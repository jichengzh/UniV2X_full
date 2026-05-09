"""M4.6.2: Pyramid 多模块 mask-based pruning (按 framework adapter 5 模块映射).

Framework adapter 的 5 模块 → HEAL 实际模块:
  backbone   → encoder_m1.* + backbone_m1.*
  encoder    → pyramid_backbone.resnet.layer0  (ResNeXt 第 1 stage)
  decoder    → pyramid_backbone.resnet.layer1 + layer2  (后 2 stages, 主算量)
  heads      → cls_head + reg_head + dir_head + pyramid_backbone.single_head_*
  v2x_comm   → pyramid_backbone.shrink_conv  (multi-scale fusion conv, 近似 aligner)

运行 4 个 multi-module 配置 (来自 framework adapter 的合理选择):
  baseline                            : 全 0 (sanity)
  uniform_30                          : 5 模块都 30% L1 prune
  framework_recommended (front-heavy) : backbone 50% + encoder 50% + decoder 30% + heads 30% + v2x_comm 20%
  framework_aggressive (back-heavy)   : backbone 30% + encoder 30% + decoder 70% + heads 50% + v2x_comm 20%

输出:
  results/m4_6_2_pyramid_multimodule.csv
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils  # type: ignore
from opencood.tools import train_utils  # type: ignore
from opencood.data_utils.datasets import build_dataset  # type: ignore
from opencood.utils import eval_utils  # type: ignore
from opencood.utils.common_utils import update_dict  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
OUT_CSV = RESULTS_DIR / "m4_6_2_pyramid_multimodule.csv"


# ── Framework 5 模块 → HEAL 实际模块 name pattern 映射 ──
MODULE_PATTERNS = {
    "backbone":  ["encoder_m1.", "backbone_m1."],
    "encoder":   ["pyramid_backbone.resnet.layer0."],
    "decoder":   ["pyramid_backbone.resnet.layer1.", "pyramid_backbone.resnet.layer2."],
    "heads":     ["cls_head", "reg_head", "dir_head", "pyramid_backbone.single_head_"],
    "v2x_comm":  ["pyramid_backbone.shrink_conv"],
}

# 4 个测试配置 (5 模块 prune rate)
CONFIGS = {
    "baseline":               {"backbone": 0.0,  "encoder": 0.0,  "decoder": 0.0,  "heads": 0.0,  "v2x_comm": 0.0},
    "uniform_30":             {"backbone": 0.3,  "encoder": 0.3,  "decoder": 0.3,  "heads": 0.3,  "v2x_comm": 0.3},
    "framework_front_heavy":  {"backbone": 0.5,  "encoder": 0.5,  "decoder": 0.3,  "heads": 0.3,  "v2x_comm": 0.2},
    "framework_back_heavy":   {"backbone": 0.3,  "encoder": 0.3,  "decoder": 0.7,  "heads": 0.5,  "v2x_comm": 0.2},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--bn_recal_batches", type=int, default=50)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--range", default="102.4,102.4")
    p.add_argument("--note", default="")
    p.add_argument("--configs", default="all", help="comma-separated subset of CONFIGS keys, or 'all'")
    return p.parse_args()


def make_hypes(opt):
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
    if "box_align" in hypes:
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]
    return hypes


def get_convs_for_framework_module(model, fw_module: str) -> list:
    """根据 framework module 名 (backbone/encoder/decoder/heads/v2x_comm) 找出 HEAL conv layers."""
    patterns = MODULE_PATTERNS.get(fw_module, [])
    found = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        if m.groups != 1:    # skip grouped conv (group constraint)
            continue
        if m.weight.shape[0] <= 4:   # skip small heads (e.g. single_head_X has out=1)
            continue
        if any(name.startswith(p) or p in name for p in patterns):
            found.append((name, m))
    return found


def apply_multimodule_pruning(model, prune_rates: dict) -> dict:
    """对每个 framework module 按 prune_rate 套 mask, 返回统计."""
    stats = {}
    total_layers = 0
    total_filters = 0
    total_pruned = 0
    for fw_module, rate in prune_rates.items():
        if rate <= 0:
            continue
        convs = get_convs_for_framework_module(model, fw_module)
        n_layers = len(convs)
        n_filters = sum(m.weight.shape[0] for _, m in convs)
        for name, conv in convs:
            prune.ln_structured(conv, name="weight", amount=rate, n=1, dim=0)
        stats[fw_module] = {"rate": rate, "n_layers": n_layers,
                             "n_filters": n_filters, "n_pruned": int(n_filters * rate)}
        total_layers += n_layers
        total_filters += n_filters
        total_pruned += int(n_filters * rate)
    return {"per_module": stats, "total_layers": total_layers,
            "total_filters": total_filters, "total_pruned": total_pruned}


def make_model(opt, hypes):
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(opt.model_dir, model)
    model.cuda().eval()
    return model


def bn_recalibrate(model, dataset, n_batches: int):
    if n_batches <= 0:
        return
    print(f"  BN recal: {n_batches} batches")
    model.train()
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=True, pin_memory=False, drop_last=False)
    seen = 0
    with torch.no_grad():
        for batch_data in loader:
            if seen >= n_batches: break
            if batch_data is None: continue
            try:
                batch_data = train_utils.to_device(batch_data, "cuda")
                _ = model(batch_data["ego"])
                seen += 1
            except Exception:
                continue
    model.eval()
    print(f"  BN recal done: {seen}/{n_batches}")


@torch.no_grad()
def eval_one(model, dataset, max_samples: int = 0) -> dict:
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    n_done = 0
    from collections import OrderedDict as _OD
    for i, batch_data in enumerate(loader):
        if max_samples > 0 and i >= max_samples: break
        if batch_data is None: continue
        batch_data = train_utils.to_device(batch_data, "cuda")
        torch.cuda.synchronize()
        starter.record()
        out = model(batch_data["ego"])
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))
        try:
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, _OD([("ego", out)]))
        except Exception as e:
            print(f"  [post_process error i={i}: {e}]")
            continue
        for iou in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
        n_done += 1
        if i % 200 == 0 and i > 0:
            print(f"  [{n_done}/{max_samples or len(dataset)}] cum_lat_p50={np.percentile(timings,50):.2f}ms")
        torch.cuda.empty_cache()
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, "/tmp", "m4_6_2")
    timings_arr = np.array(timings)
    return {
        "n_samples": n_done,
        "AP30": float(ap30), "AP50": float(ap50), "AP70": float(ap70),
        "lat_mean_ms": float(np.mean(timings_arr)),
        "lat_p50_ms": float(np.percentile(timings_arr, 50)),
        "lat_p99_ms": float(np.percentile(timings_arr, 99)),
    }


def main():
    opt = parse_args()
    if opt.configs == "all":
        config_keys = list(CONFIGS.keys())
    else:
        config_keys = opt.configs.split(",")

    print("=" * 60)
    print(f"M4.6.2: Pyramid multi-module pruning")
    print(f"  configs: {config_keys}")
    print(f"  BN recal: {opt.bn_recal_batches} batches")
    print("=" * 60)

    hypes = make_hypes(opt)
    np.random.seed(303)
    dataset = build_dataset(hypes, visualize=True, train=False)

    rows = []
    for config_name in config_keys:
        rates = CONFIGS[config_name]
        print(f"\n=== {config_name}: {rates} ===")
        model = make_model(opt, hypes)
        if any(v > 0 for v in rates.values()):
            stats = apply_multimodule_pruning(model, rates)
            print(f"  total: {stats['total_layers']} layers, "
                  f"{stats['total_pruned']}/{stats['total_filters']} filters")
            bn_recalibrate(model, dataset, opt.bn_recal_batches)
        eval_res = eval_one(model, dataset, opt.max_samples)
        row = {
            "config_name": config_name,
            **{f"prune_rate__{k}": v for k, v in rates.items()},
            "bn_recal_batches": opt.bn_recal_batches if any(v>0 for v in rates.values()) else 0,
            **eval_res,
        }
        rows.append(row)
        print(f"  → AP30/50/70: {row['AP30']:.4f}/{row['AP50']:.4f}/{row['AP70']:.4f} "
              f"| lat_p50: {row['lat_p50_ms']:.2f}ms")
        del model
        torch.cuda.empty_cache()

    df_new = pd.DataFrame(rows)
    df_new["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        df = pd.concat([prev, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Wrote {OUT_CSV} ({len(df)} total rows)")
    print(f"\n=== M4.6.2 摘要 ===")
    show = df_new[["config_name", "AP30", "AP50", "AP70", "lat_p50_ms"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
