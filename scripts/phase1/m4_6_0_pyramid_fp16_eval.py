"""M4.6.0: Pyramid_m1_base 完整 e2e FP32 vs FP16 实测.

复用 HEAL inference.py 结构, 加:
  - --fp16 标志: model.half() + batch_data 转 fp16
  - 每帧 CUDA event timing
  - 输出 AP30/50/70 + latency 统计 (mean/p50/p99)

输出:
  results/m4_6_0_pyramid_{fp32|fp16}_eval.txt — AP + latency 统计
  results/m4_6_0_pyramid_eval.csv — 一行 (precision, AP30/50/70, lat_mean/p50/p99 ms)

用法:
  PYTHONPATH=/home/jichengzhi/heal_research/HEAL python scripts/phase1/m4_6_0_pyramid_fp16_eval.py \
      --model_dir /home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12 \
      --precision fp16
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
# heter assignment 用 cwd-relative path, 切到 HEAL_ROOT 让相对路径生效
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils  # type: ignore
from opencood.tools import train_utils, inference_utils  # type: ignore
from opencood.data_utils.datasets import build_dataset  # type: ignore
from opencood.utils import eval_utils  # type: ignore
from opencood.utils.common_utils import update_dict  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--fusion_method", default="intermediate")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 = all 2170; <2170 = quick smoke test")
    p.add_argument("--range", default="102.4,102.4")
    p.add_argument("--save_vis_interval", type=int, default=10**9,
                   help="effectively disable vis (set big)")
    p.add_argument("--save_npy", action="store_true")
    p.add_argument("--no_score", action="store_true")
    p.add_argument("--note", default="")
    return p.parse_args()


def to_half_recursive(x):
    """递归把 dict / list / tensor 中的 float tensor 转 fp16, 保留 long/bool."""
    if isinstance(x, dict):
        return {k: to_half_recursive(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_half_recursive(v) for v in x]
    if torch.is_tensor(x) and x.is_floating_point():
        return x.half()
    return x


def main():
    opt = parse_args()
    is_fp16 = (opt.precision == "fp16")

    # ── HEAL inference.py 的 hypes 加载逻辑 ──
    hypes = yaml_utils.load_yaml(None, opt)
    if "heter" in hypes:
        x_min, x_max = -float(opt.range.split(",")[0]), float(opt.range.split(",")[0])
        y_min, y_max = -float(opt.range.split(",")[1]), float(opt.range.split(",")[1])
        opt.note += f"_{x_max}_{y_max}"
        new_cav_range = [
            x_min, y_min,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        })
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        for name, func in yaml_utils_lib.__dict__.items():
            if name == hypes["yaml_parser"]:
                hypes = func(hypes)

    hypes["validate_dir"] = hypes["test_dir"]
    left_hand = "OPV2V" in hypes["test_dir"] or "V2XSET" in hypes["test_dir"]
    print(f"left_hand={left_hand}")

    if "box_align" in hypes:
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    print("Creating Model")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Model from checkpoint")
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    print(f"resume from {resume_epoch} epoch")
    opt.note += f"_epoch{resume_epoch}_{opt.precision}"

    model.cuda().eval()
    if is_fp16:
        model = model.half()
        print("✅ model.half() — FP16")

    print("Dataset Building")
    np.random.seed(303)
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    n_total = len(dataset)
    if opt.max_samples > 0:
        print(f"⚠️  max_samples={opt.max_samples} 限定快速测试 (full {n_total})")

    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}

    # CUDA event timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []

    n_done = 0
    t_start = time.time()
    for i, batch_data in enumerate(loader):
        if opt.max_samples > 0 and i >= opt.max_samples:
            break
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if is_fp16:
                batch_data = to_half_recursive(batch_data)

            torch.cuda.synchronize()
            starter.record()

            if opt.fusion_method == "intermediate":
                infer_result = inference_utils.inference_intermediate_fusion(
                    batch_data, model, dataset)
            elif opt.fusion_method == "no":
                infer_result = inference_utils.inference_no_fusion(
                    batch_data, model, dataset)
            elif opt.fusion_method == "single":
                infer_result = inference_utils.inference_no_fusion(
                    batch_data, model, dataset, single_gt=True)
            else:
                raise NotImplementedError(opt.fusion_method)

            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

            pred_box_tensor = infer_result["pred_box_tensor"]
            gt_box_tensor = infer_result["gt_box_tensor"]
            pred_score = infer_result["pred_score"]

            for iou in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score,
                                            gt_box_tensor, result_stat, iou)
        n_done += 1
        if i % 100 == 0:
            print(f"  [{n_done:4d}/{opt.max_samples or n_total}] cum_lat_mean={np.mean(timings):.2f} ms")
        torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n=== inference done — {n_done} samples in {elapsed:.1f}s ===")

    # AP — eval_final_results returns (ap30, ap50, ap70)
    ap30, ap50, ap70 = eval_utils.eval_final_results(
        result_stat, opt.model_dir, opt.fusion_method + opt.note)

    timings_arr = np.array(timings)
    stats = {
        "precision": opt.precision,
        "n_samples": n_done,
        "AP30": float(ap30),
        "AP50": float(ap50),
        "AP70": float(ap70),
        "lat_mean_ms": float(np.mean(timings_arr)),
        "lat_p50_ms": float(np.percentile(timings_arr, 50)),
        "lat_p99_ms": float(np.percentile(timings_arr, 99)),
        "lat_std_ms": float(np.std(timings_arr)),
        "lat_min_ms": float(np.min(timings_arr)),
        "lat_max_ms": float(np.max(timings_arr)),
    }

    print(f"\n=== M4.6.0 result ({opt.precision}) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 写报告 + CSV (append mode)
    out_txt = RESULTS_DIR / f"m4_6_0_pyramid_{opt.precision}_eval.txt"
    with out_txt.open("w") as f:
        f.write(f"# M4.6.0 Pyramid_m1_base 完整 e2e — {opt.precision}\n")
        f.write(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"model_dir: {opt.model_dir}\n")
        f.write(f"max_samples: {opt.max_samples} (0=all)\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print(f"\n✅ Wrote {out_txt}")

    out_csv = RESULTS_DIR / "m4_6_0_pyramid_eval.csv"
    import pandas as pd
    row = pd.DataFrame([stats])
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        # 删掉同 precision 旧行 (重测覆盖)
        prev = prev[prev["precision"] != opt.precision]
        df = pd.concat([prev, row], ignore_index=True)
    else:
        df = row
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}")


if __name__ == "__main__":
    main()
