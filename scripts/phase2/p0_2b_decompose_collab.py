"""P0 — decompose pyramid_backbone.forward_collab into its real sub-stages.

Closes the measurement gap behind "pyramid_backbone is the optimizable bottleneck":
the prunable ResNeXt (get_multiscale_feature) vs the V2X fusion (single_head +
weighted_fuse warp) vs decode were NEVER separately measured. The 14.45ms
forward_collab was only ever split by subtracting a cross-condition 3.3ms submodule
number — an inference, not a measurement.

This script monkey-patches forward_collab with a CUDA-Event-instrumented replica
(no HEAL source change) and reports per-sub-stage mean/p50/p99:
  get_multiscale  (3-stage ResNeXt, prunable via num_filters/plane)
  single_head     (3x occ heads, 1x1 conv)
  weighted_fuse   (3x warp_affine grid_sample + cross-agent weighted sum) -> NOT prunable
  decode          (3x deblock convtranspose + concat)

Run on an idle GPU (C10):
  cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=<idle> python \
    /home/jichengzhi/UniV2X/scripts/phase2/p0_2b_decompose_collab.py --measure 150
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

DATA_DIR = Path("/home/jichengzhi/UniV2X/paper_learning/2. AAAI最终故事/data")
P64_DIR = HEAL_ROOT / "opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45"
P64_CKPT = "net_epoch_bestval_at19.pth"
P64_NUM_FILTERS = [64, 128, 256]

# accumulators (per forward_collab call)
SUB = {k: [] for k in ["get_multiscale", "single_head", "weighted_fuse",
                        "decode", "collab_total"]}


def make_instrumented_forward_collab(pyramid):
    """Return an instrumented forward_collab bound to `pyramid`, mirroring the
    real body (LiDAR path; cam_crop ignored as DAIR g8 is LiDAR-only)."""
    from opencood.models.fuse_modules.pyramid_fuse import weighted_fuse

    def fwd(spatial_features, record_len, affine_matrix,
            agent_modality_list=None, cam_crop_info=None):
        ev = {k: torch.cuda.Event(enable_timing=True)
              for k in ["s", "gm", "sh_sum", "wf_sum", "dec"]}
        sh_ms = 0.0
        wf_ms = 0.0
        ev["s"].record()
        feature_list = pyramid.get_multiscale_feature(spatial_features)
        ev["gm"].record()

        fused_feature_list = []
        occ_map_list = []
        for i in range(pyramid.num_levels):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            e0.record()
            occ_map = eval(f"pyramid.single_head_{i}")(feature_list[i])
            occ_map_list.append(occ_map)
            score = torch.sigmoid(occ_map) + 1e-4
            e1.record()
            fused_feature_list.append(
                weighted_fuse(feature_list[i], score, record_len,
                              affine_matrix, pyramid.align_corners))
            e2.record()
            torch.cuda.synchronize()
            sh_ms += e0.elapsed_time(e1)
            wf_ms += e1.elapsed_time(e2)
        ev["sh_sum"].record()  # marker only; per-level sums tracked above

        fused_feature = pyramid.decode_multiscale_feature(fused_feature_list)
        ev["dec"].record()
        torch.cuda.synchronize()

        SUB["get_multiscale"].append(ev["s"].elapsed_time(ev["gm"]))
        SUB["single_head"].append(sh_ms)
        SUB["weighted_fuse"].append(wf_ms)
        SUB["decode"].append(ev["sh_sum"].elapsed_time(ev["dec"]))
        SUB["collab_total"].append(ev["s"].elapsed_time(ev["dec"]))
        return fused_feature, occ_map_list

    return fwd


def stats(arr):
    a = np.asarray(arr, dtype=float)
    return {"n": len(a), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=150)
    ap.add_argument("--out", default=str(DATA_DIR / "p0_2b_collab_decompose.json"))
    args = ap.parse_args()

    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils, inference_utils
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    cfg = yaml_utils.load_yaml(str(P64_DIR / "config.yaml"))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = P64_NUM_FILTERS
    cfg["validate_dir"] = cfg.get("validate_dir") or cfg.get("data_dir")

    print("[decompose] building model", flush=True)
    model = train_utils.create_model(cfg)
    sd = torch.load(P64_DIR / P64_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()

    print("[decompose] building DAIR val dataset", flush=True)
    val_ds = build_dataset(cfg, visualize=False, train=False)
    total = args.warmup + args.measure
    n_use = min(total, len(val_ds))
    loader = DataLoader([val_ds[i] for i in range(n_use)], batch_size=1, num_workers=2,
                        collate_fn=val_ds.collate_batch_test, shuffle=False)

    model.pyramid_backbone.forward_collab = make_instrumented_forward_collab(
        model.pyramid_backbone)

    print(f"[decompose] warmup={args.warmup} measure={args.measure}", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= total:
                break
            batch = train_utils.to_device(batch, device)
            try:
                _ = inference_utils.inference_intermediate_fusion(batch, model, val_ds)
            except Exception as e:
                if i < 3:
                    print(f"  sample {i} failed: {type(e).__name__}: {str(e)[:100]}")
            if i + 1 == args.warmup:
                for k in SUB:
                    SUB[k].clear()
                print("  [warmup done] reset", flush=True)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{total}", flush=True)

    report = {k: stats(v) for k, v in SUB.items() if v}
    Path(args.out).write_text(json.dumps(report, indent=2))

    tot = report["collab_total"]["mean"]
    print("\n============== forward_collab internal decomposition ==============")
    print(f"{'sub-stage':16s} {'mean(ms)':>10s} {'p50':>8s} {'p99':>8s} {'%collab':>8s}")
    for k in ["get_multiscale", "single_head", "weighted_fuse", "decode", "collab_total"]:
        s = report[k]
        pct = 100 * s["mean"] / tot if tot else 0
        print(f"{k:16s} {s['mean']:10.3f} {s['p50']:8.3f} {s['p99']:8.3f} {pct:7.1f}%")
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
