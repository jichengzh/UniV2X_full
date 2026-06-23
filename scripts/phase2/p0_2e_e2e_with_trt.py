"""P0.2 (e2e half) — realized DAIR e2e wall-clock with pyramid get_multiscale on TRT.

The submodule bench (p0_2d) showed TRT-FP16 is 6.28x / INT8 8.08x faster than
PyTorch on get_multiscale. This script measures whether that translates to e2e:
patch pyramid_backbone.get_multiscale_feature with the 3-output TRTBackbone3 and
time the full inference_intermediate_fusion (incl. CUDA NMS from P0.1) on DAIR val.

Modes: pytorch (no patch) | trt_fp16 | trt_int8.

Run on idle GPU (C10):
  cd /home/jichengzhi/heal_research/HEAL && CUDA_VISIBLE_DEVICES=6 python \
    /home/jichengzhi/UniV2X/scripts/phase2/p0_2e_e2e_with_trt.py --measure 150
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO / "scripts" / "phase2"))

from p0_2_build_eval import TRTBackbone3  # noqa: E402
from p0_2_multiscale_trt import ENGINE_DIR, P64_BASELINE  # noqa: E402


def stats(arr):
    a = np.asarray(arr, dtype=float)
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p99": round(float(np.percentile(a, 99)), 3)}


def run_mode(model, ds, loader, device, mode, engine, warmup, measure):
    orig = model.pyramid_backbone.get_multiscale_feature
    if mode != "pytorch":
        model.pyramid_backbone.get_multiscale_feature = TRTBackbone3(engine)
    from opencood.tools import train_utils, inference_utils
    e2e = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= warmup + measure:
                break
            batch = train_utils.to_device(batch, device)
            torch.cuda.synchronize(); t0 = time.time()
            try:
                _ = inference_utils.inference_intermediate_fusion(batch, model, ds)
            except Exception as ex:
                if i < 3:
                    print(f"  [{mode}] sample {i} failed: {type(ex).__name__}: {str(ex)[:100]}")
                continue
            torch.cuda.synchronize()
            if i >= warmup:
                e2e.append((time.time() - t0) * 1000)
    model.pyramid_backbone.get_multiscale_feature = orig  # restore
    return stats(e2e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=150)
    ap.add_argument("--modes", nargs="*", default=["pytorch", "trt_fp16", "trt_int8"])
    args = ap.parse_args()

    tag, model_dir, num_filters, ckpt_name = P64_BASELINE
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    cfg = yaml_utils.load_yaml(str(model_dir / "config.yaml"))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = num_filters
    cfg["validate_dir"] = cfg.get("validate_dir") or cfg.get("data_dir")
    print("[e2e] building model", flush=True)
    model = train_utils.create_model(cfg)
    sd = torch.load(model_dir / ckpt_name, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    device = torch.device("cuda"); model.to(device).eval()

    print("[e2e] building DAIR val dataset", flush=True)
    ds = build_dataset(cfg, visualize=False, train=False)
    n_use = min(args.warmup + args.measure, len(ds))
    loader = DataLoader([ds[i] for i in range(n_use)], batch_size=1, num_workers=2,
                        collate_fn=ds.collate_batch_test, shuffle=False)

    eng = {"trt_fp16": ENGINE_DIR / f"{tag}_fp16.engine",
           "trt_int8": ENGINE_DIR / f"{tag}_int8.engine"}
    res = {}
    for m in args.modes:
        print(f"[e2e] mode={m}", flush=True)
        res[m] = run_mode(model, ds, loader, device, m, eng.get(m),
                          args.warmup, args.measure)
        print(f"  -> {res[m]}", flush=True)

    base = res.get("pytorch", {}).get("mean")
    out = DATA_DIR / "p0_2e_e2e_with_trt.json"
    out.write_text(json.dumps(res, indent=2))
    print("\n============== DAIR e2e with pyramid get_multiscale on TRT ==============")
    print(f"{'mode':12s} {'e2e mean(ms)':>13s} {'p50':>8s} {'p99':>8s} {'vs pytorch':>11s}")
    for m in args.modes:
        r = res[m]
        sp = f"{base/r['mean']:.2f}x" if base else "-"
        print(f"{m:12s} {r['mean']:13.3f} {r['p50']:8.3f} {r['p99']:8.3f} {sp:>11s}")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
