"""P0.2 Q1 — capture REAL spatial_features feeding get_multiscale_feature.

Root cause A (plan6 §三): plan v5's MinMaxCalibrator.get_batch() returned None and
only read a stale reused cache -> TRT had no scales for the actual tensors -> INT8
fell back to FP16/FP32. The fix is to calibrate with the ANCHOR'S OWN real
intermediate features.

This script boots the full HeterPyramidCollab model, hooks get_multiscale_feature
to record its input tensor on N real DAIR-V2X val samples, splits per-agent into
(1, C, H, W) batches, and caches them for the INT8 calibrator. It also reports the
true (C, H, W) so the TRT engine input shape is verified (not assumed 256x256).

Output: /tmp/plan6_p0_2_calib/<tag>_calib.npy  (float32, shape (M, C, H, W))
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))

CALIB_DIR = Path("/tmp/plan6_p0_2_calib")
CALIB_DIR.mkdir(parents=True, exist_ok=True)

P64_BASELINE_DIR = HEAL_ROOT / "opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45"
P64_CKPT = "net_epoch_bestval_at19.pth"
P64_NUM_FILTERS = [64, 128, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p64_baseline")
    ap.add_argument("--model-dir", default=str(P64_BASELINE_DIR))
    ap.add_argument("--ckpt", default=P64_CKPT)
    ap.add_argument("--num-filters", type=int, nargs=3, default=P64_NUM_FILTERS)
    ap.add_argument("--n-samples", type=int, default=100,
                    help="DAIR val samples to harvest features from")
    ap.add_argument("--max-batches", type=int, default=256,
                    help="cap on cached per-agent feature batches")
    args = ap.parse_args()

    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils, inference_utils
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    model_dir = Path(args.model_dir)
    cfg = yaml_utils.load_yaml(str(model_dir / "config.yaml"))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = list(args.num_filters)
    cfg["validate_dir"] = cfg.get("validate_dir") or cfg.get("data_dir")

    print(f"[Q1] building model {args.tag}", flush=True)
    model = train_utils.create_model(cfg)
    sd = torch.load(model_dir / args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()

    print(f"[Q1] building DAIR val dataset", flush=True)
    val_ds = build_dataset(cfg, visualize=False, train=False)
    n_use = min(args.n_samples, len(val_ds))
    loader = DataLoader([val_ds[i] for i in range(n_use)], batch_size=1, num_workers=0,
                        collate_fn=val_ds.collate_batch_test, shuffle=False)

    # hook: capture input to get_multiscale_feature
    pyramid = model.pyramid_backbone
    orig_get = pyramid.get_multiscale_feature
    captured = []

    def hooked_get(x):
        captured.append(x.detach().to(torch.float32).cpu())
        return orig_get(x)

    pyramid.get_multiscale_feature = hooked_get

    print(f"[Q1] harvesting features over {n_use} samples", flush=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            batch = train_utils.to_device(batch, device)
            try:
                _ = inference_utils.inference_intermediate_fusion(batch, model, val_ds)
            except Exception as e:
                print(f"  sample {i} fwd failed: {type(e).__name__}: {str(e)[:80]}")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{n_use}  captured {len(captured)} tensors", flush=True)

    pyramid.get_multiscale_feature = orig_get  # restore

    # split each (N_agents, C, H, W) into per-agent (1, C, H, W)
    per_agent = []
    shapes = set()
    for t in captured:
        for a in range(t.shape[0]):
            per_agent.append(t[a:a + 1].numpy())
            shapes.add(tuple(t.shape[1:]))
        if len(per_agent) >= args.max_batches:
            break

    if not per_agent:
        print("[Q1] ERROR: no features captured")
        return 1

    arr = np.concatenate(per_agent, axis=0).astype(np.float32)  # (M, C, H, W)
    out = CALIB_DIR / f"{args.tag}_calib.npy"
    np.save(out, arr)
    print(f"\n==================== P0.2 Q1 calibration harvest ====================")
    print(f"input shapes seen (C,H,W): {sorted(shapes)}")
    print(f"cached {arr.shape[0]} per-agent batches, dtype={arr.dtype}, "
          f"each {arr.shape[1:]}")
    print(f"value range: min={arr.min():.3f} max={arr.max():.3f} "
          f"mean={arr.mean():.4f} std={arr.std():.4f}")
    print(f"[Q1] wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
