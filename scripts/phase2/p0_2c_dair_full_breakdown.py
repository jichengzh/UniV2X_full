"""P0 — full per-module latency breakdown on DAIR-V2X (post-CUDA-NMS).

Reuses the proven CUDA-Event timing replay from m4_8_pyramid_per_stage_timing.py
(encoder/backbone/aligner/pyramid/shrink/heads + postproc decode/dir/corners/nms),
but boots the DAIR g8 baseline with its native geometry (no OPV2V ±102.4 range
override). Because P0.1 integrated CUDA nms_rotated into HEAL box_utils, the NMS
sub-stage here reflects the fast path -> this is the real DAIR e2e AFTER the NMS fix.

Goal: rank every prunable module (encoder / backbone / pyramid ResNeXt / shrink) by
real DAIR cost, to inform whole-network pruning design.

Run on idle GPU (C10):
  cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=6 python \
    /home/jichengzhi/UniV2X/scripts/phase2/p0_2c_dair_full_breakdown.py --measure 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, "/home/jichengzhi/UniV2X/scripts/phase1")

from m4_8_pyramid_per_stage_timing import (  # noqa: E402
    CudaTimer, replay_forward_timed,
)

DATA_DIR = Path("/home/jichengzhi/UniV2X/paper_learning/2. AAAI最终故事/data")
P64_DIR = HEAL_ROOT / "opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45"
P64_CKPT = "net_epoch_bestval_at19.pth"
P64_NUM_FILTERS = [64, 128, 256]


def stats(arr):
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=150)
    ap.add_argument("--out", default=str(DATA_DIR / "p0_2c_dair_full_breakdown.json"))
    args = ap.parse_args()

    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    cfg = yaml_utils.load_yaml(str(P64_DIR / "config.yaml"))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = P64_NUM_FILTERS
    cfg["validate_dir"] = cfg.get("validate_dir") or cfg.get("data_dir")

    print("[dair-bd] building model", flush=True)
    model = train_utils.create_model(cfg)
    _, model = train_utils.load_saved_model(str(P64_DIR), model)
    model.cuda().eval()

    print("[dair-bd] building DAIR val dataset", flush=True)
    ds = build_dataset(cfg, visualize=False, train=False)
    total = args.warmup + args.measure
    n_use = min(total, len(ds))
    loader = DataLoader([ds[i] for i in range(n_use)], batch_size=1, num_workers=2,
                        collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False)

    fwd_labels = ["start", "encoder", "backbone", "aligner", "pyramid", "shrink",
                  "heads", "postproc"]
    fwd_timer = CudaTimer(fwd_labels)
    sub_timers = {"decode": [], "dir": [], "corners": [], "nms": [], "range_mask": []}
    e2e_ms, record_len_list = [], []
    post_processor = ds.post_processor
    device = torch.device("cuda")

    print(f"[dair-bd] warmup={args.warmup} measure={args.measure}", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= total:
                break
            batch = train_utils.to_device(batch, device)
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                _ = replay_forward_timed(model, batch, post_processor, fwd_timer, sub_timers)
            except Exception as e:
                if i < 3:
                    print(f"  sample {i} failed: {type(e).__name__}: {str(e)[:120]}")
                continue
            torch.cuda.synchronize()
            if i >= args.warmup:
                e2e_ms.append((time.time() - t0) * 1000)
                record_len_list.append(int(batch["ego"]["record_len"][0]))
            if i + 1 == args.warmup:
                fwd_timer.records = {k: [] for k in fwd_timer.records}
                sub_timers = {k: [] for k in sub_timers}
                print("  [warmup done] reset", flush=True)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{total}", flush=True)

    report = {
        "dataset": "DAIR-V2X val", "ckpt": str(P64_DIR / P64_CKPT),
        "record_len_mean": float(np.mean(record_len_list)) if record_len_list else None,
        "e2e_walltime": stats(e2e_ms),
        "forward_stages": {k: stats(v) for k, v in fwd_timer.records.items()},
        "postproc_substages": {k: stats(v) for k, v in sub_timers.items()},
        "device": torch.cuda.get_device_name(0),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    e2e = report["e2e_walltime"]["mean"]
    print("\n============== DAIR full per-module breakdown (post CUDA-NMS) ==============")
    print(f"record_len mean = {report['record_len_mean']:.2f} agents/scene")
    print(f"{'stage':14s} {'mean(ms)':>10s} {'p50':>8s} {'p99':>8s} {'%e2e':>7s}")
    for k in ["encoder", "backbone", "aligner", "pyramid", "shrink", "heads", "postproc"]:
        s = report["forward_stages"].get(k)
        if s:
            print(f"{k:14s} {s['mean']:10.3f} {s['p50']:8.3f} {s['p99']:8.3f} "
                  f"{100*s['mean']/e2e:6.1f}%")
    print("  -- postproc sub --")
    for k in ["decode", "dir", "corners", "nms", "range_mask"]:
        s = report["postproc_substages"].get(k)
        if s:
            print(f"  {k:12s} {s['mean']:10.3f} {s['p50']:8.3f} {s['p99']:8.3f} "
                  f"{100*s['mean']/e2e:6.1f}%")
    print(f"{'e2e walltime':14s} {e2e:10.3f} {report['e2e_walltime']['p50']:8.3f}")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
