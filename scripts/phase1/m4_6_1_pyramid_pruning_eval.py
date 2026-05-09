"""M4.6.1: Pyramid L1 mask-based channel pruning + BN recalibration AP curve.

技术说明:
  - 用 torch.nn.utils.prune.ln_structured 给 PyramidFusion 内 Conv2d 套 mask
  - 只剪 groups=1 的 1x1 conv (conv1/conv3/downsample); 跳过 groups=32 的 conv2 (避 group constraint)
  - BN recalibration: forward N batches in train() mode, 让 BN 重统计 mean/var, 部分恢复 AP
  - 测 AP30/50/70 + 完整 e2e latency on OPV2V test 2170 samples

注意:
  - mask-based pruning 的 GPU latency 跟 baseline 几乎一样 (PyTorch dense kernel 不跳过 zero weights)
  - 真正减小 latency 要 M4.6.2 (重 build smaller PyramidFusion + truncate weights)
  - 本步只验证: framework adapter 准则 (L1) 在 Pyramid CNN decoder 上的 AP-rate 曲线

输出:
  results/m4_6_1_pyramid_pruning.csv (每行: prune_rate, criterion, bn_recal, AP, lat)
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
from torch.utils.data import DataLoader, Subset

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils  # type: ignore
from opencood.tools import train_utils, inference_utils  # type: ignore
from opencood.data_utils.datasets import build_dataset  # type: ignore
from opencood.utils import eval_utils  # type: ignore
from opencood.utils.common_utils import update_dict  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV = RESULTS_DIR / "m4_6_1_pyramid_pruning.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--prune_rates", default="0.0,0.25,0.5,0.75,0.9",
                   help="comma-separated list of prune rates to sweep")
    p.add_argument("--criterion", choices=["l1", "fpgm"], default="l1")
    p.add_argument("--bn_recal_batches", type=int, default=50,
                   help="BN recalibration: N batches forward in train() mode")
    p.add_argument("--bn_recal_off", action="store_true", help="skip BN recal")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--range", default="102.4,102.4")
    p.add_argument("--note", default="")
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


def get_prunable_convs(model: torch.nn.Module) -> list[tuple[str, torch.nn.Conv2d]]:
    """返回 (name, conv) 列表 — 只对 groups=1 的 1x1 conv 剪 output channel.

    跳过:
      - conv2 (groups=32 grouped, 剪 output 会破坏 group structure)
      - single_head_X (out_channels=1 太小, 不可剪)
      - encoder_m1 (PointPillar VFE, 不在 framework decoder 范围)
    """
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        # 只剪 pyramid_backbone (M4.7 的 decoder/encoder 模块) 和 shrink_conv
        if not (name.startswith("pyramid_backbone") or name.startswith("shrink_conv")):
            continue
        # 跳过 grouped conv
        if m.groups != 1:
            continue
        # 跳过 output_channel <= 4 (没意义)
        if m.weight.shape[0] <= 4:
            continue
        # 跳过 1x1 expansion conv (only kernel size = 1x1 is OK; we want main 3x3 trunk too if exists)
        out.append((name, m))
    return out


def fpgm_score(weight: torch.Tensor) -> torch.Tensor:
    """FPGM (filter pruning via geometric median) score per output filter.
    The filter closest to all others (smallest distance sum) gets lowest score → pruned first.
    """
    n = weight.shape[0]
    flat = weight.view(n, -1)
    # pairwise L2 distances among filters
    dist = torch.cdist(flat, flat, p=2)  # (n, n)
    score = dist.sum(dim=1)  # higher = more unique
    return score


def apply_pruning(model, prune_rate: float, criterion: str = "l1") -> dict:
    """对所有 prunable conv 套 mask, 返回统计."""
    prunable = get_prunable_convs(model)
    n_layers = len(prunable)
    n_total_filters = sum(m.weight.shape[0] for _, m in prunable)
    n_pruned_filters = 0

    for name, conv in prunable:
        amount = prune_rate
        if criterion == "l1":
            # L1 norm pruning on output channel (dim=0)
            prune.ln_structured(conv, name="weight", amount=amount, n=1, dim=0)
        elif criterion == "fpgm":
            # custom FPGM: 给 weight 打分, mask 最小 amount 比例
            with torch.no_grad():
                score = fpgm_score(conv.weight.data)
                k = int(score.numel() * amount)
                if k == 0:
                    continue
                _, prune_idx = torch.topk(score, k, largest=False)
                mask = torch.ones_like(conv.weight.data)
                mask[prune_idx] = 0.0
                prune.custom_from_mask(conv, "weight", mask=mask)
        n_pruned_filters += int(conv.weight.shape[0] * amount)

    return {
        "n_layers_pruned": n_layers,
        "n_total_filters": n_total_filters,
        "n_pruned_filters": n_pruned_filters,
    }


def make_model_and_loader(opt, hypes):
    model = train_utils.create_model(hypes)
    resume_epoch, model = train_utils.load_saved_model(opt.model_dir, model)
    model.cuda().eval()

    np.random.seed(303)
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    return model, dataset, loader, resume_epoch


def bn_recalibrate(model, dataset, n_batches: int):
    """BN recalibration: forward N batches in train() mode 让 BN 重统计 running mean/var."""
    if n_batches <= 0:
        return
    print(f"  BN recalibration: forward {n_batches} batches in train() mode")
    model.train()
    # 关 dropout 等 (但 BN 仍重统计)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=True, pin_memory=False, drop_last=False)
    seen = 0
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if seen >= n_batches:
                break
            if batch_data is None:
                continue
            try:
                batch_data = train_utils.to_device(batch_data, "cuda")
                _ = model(batch_data["ego"])
                seen += 1
            except Exception as e:
                # 一些 batch 可能边界问题, skip
                continue
    model.eval()
    print(f"  BN recal done: {seen}/{n_batches} batches forwarded")


@torch.no_grad()
def eval_one(model, dataset, loader, max_samples: int = 0) -> dict:
    """跑 inference + AP + per-frame timing, 返回 dict."""
    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = []
    n_done = 0
    from collections import OrderedDict as _OD

    for i, batch_data in enumerate(loader):
        if max_samples > 0 and i >= max_samples:
            break
        if batch_data is None:
            continue
        batch_data = train_utils.to_device(batch_data, "cuda")
        torch.cuda.synchronize()
        starter.record()
        out = model(batch_data["ego"])
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

        output_dict = _OD([("ego", out)])
        try:
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, output_dict)
        except Exception as e:
            # 某些极端 prune rate 可能导致 NMS 失败 (零 boxes)
            print(f"  [post_process error at i={i}: {e}]")
            continue
        for iou in (0.3, 0.5, 0.7):
            eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
        n_done += 1
        if i % 200 == 0 and i > 0:
            print(f"  [{n_done:4d}/{max_samples or len(dataset)}] cum_lat_p50={np.percentile(timings,50):.2f}ms")
        torch.cuda.empty_cache()

    ap30, ap50, ap70 = eval_utils.eval_final_results(
        result_stat, "/tmp", "m4_6_1")  # save_path 不重要, 写 /tmp
    timings_arr = np.array(timings)
    return {
        "n_samples": n_done,
        "AP30": float(ap30),
        "AP50": float(ap50),
        "AP70": float(ap70),
        "lat_mean_ms": float(np.mean(timings_arr)),
        "lat_p50_ms": float(np.percentile(timings_arr, 50)),
        "lat_p99_ms": float(np.percentile(timings_arr, 99)),
    }


def main():
    opt = parse_args()
    rates = [float(r) for r in opt.prune_rates.split(",")]
    do_bn = not opt.bn_recal_off

    print("=" * 60)
    print(f"M4.6.1: Pyramid pruning AP curve")
    print(f"  criterion: {opt.criterion}")
    print(f"  prune_rates: {rates}")
    print(f"  BN recal: {opt.bn_recal_batches if do_bn else 'OFF'}")
    print(f"  max_samples: {opt.max_samples or 'all'}")
    print("=" * 60)

    hypes = make_hypes(opt)

    rows = []
    for rate in rates:
        print(f"\n=== prune_rate={rate} ===")
        # 重新加载 fresh model (避免上一轮 prune 累积)
        model, dataset, loader, _ = make_model_and_loader(opt, hypes)

        if rate > 0:
            stats = apply_pruning(model, rate, opt.criterion)
            print(f"  pruned {stats['n_layers_pruned']} layers, "
                  f"{stats['n_pruned_filters']}/{stats['n_total_filters']} filters")
            if do_bn:
                bn_recalibrate(model, dataset, opt.bn_recal_batches)

        eval_res = eval_one(model, dataset, loader, opt.max_samples)
        row = {
            "prune_rate": rate,
            "criterion": opt.criterion if rate > 0 else "none",
            "bn_recal_batches": opt.bn_recal_batches if (do_bn and rate > 0) else 0,
            **eval_res,
        }
        rows.append(row)
        print(f"  → AP30/50/70: {row['AP30']:.4f}/{row['AP50']:.4f}/{row['AP70']:.4f} "
              f"| lat_p50: {row['lat_p50_ms']:.2f}ms")

        # 释放
        del model, loader
        torch.cuda.empty_cache()

    # 输出 CSV (append 模式)
    df_new = pd.DataFrame(rows)
    df_new["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        df = pd.concat([prev, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Wrote {OUT_CSV} ({len(df)} total rows)")
    print(f"\n=== M4.6.1 {opt.criterion} 摘要 ===")
    print(df_new[["prune_rate", "criterion", "AP30", "AP50", "AP70", "lat_p50_ms"]].to_string(index=False))


if __name__ == "__main__":
    main()
