"""P0 — Generate random Pyramid candidates + bench lat (no AP, no finetune).

Pipeline per candidate:
    1. Pick (s0, s1, s2) triplet from 98 valid (gradient + power-of-2 constraints)
    2. Pick precision (FP16 or INT8)
    3. structural_prune_pyramid → pruned ckpt (no FT, weights-loaded only)
    4. tools/export_onnx_pyramid_collab → ONNX
    5. m4_8_trt_build_bench → TRT engine + lat bench (200 warmup + 200 measure)
    6. Append (config_features, lat_p50_ms) to data/pyramid_random_bench.parquet

Output: ~50-100 (config, lat) data points for LGB training.

Note: lat doesn't depend on weight values, only architecture, so we skip
finetune (saves ~5 min × 50 = 4 hours). AP eval also skipped (LGB lat predictor only).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# 1. Enumerate valid triplets (98)
# ============================================================
POW2 = {1, 2, 4, 8, 16, 32}
BASE_PLANES = (64, 128, 256)


def planes_legal(planes):
    if planes <= 0 or planes % 8 != 0:
        return False
    width = int(planes * 4 / 64) * 32
    if width == 0:
        return False
    wpg = width // 32   # groups = 32
    return wpg in POW2


def enumerate_triplets():
    s0 = [p for p in range(8, 64 + 1, 8) if planes_legal(p)]
    s1 = [p for p in range(8, 128 + 1, 8) if planes_legal(p)]
    s2 = [p for p in range(8, 256 + 1, 8) if planes_legal(p)]
    triplets = []
    for p0 in s0:
        r0 = (64 - p0) / 64
        for p1 in s1:
            r1 = (128 - p1) / 128
            if abs(r0 - r1) > 0.30 + 1e-3:
                continue
            for p2 in s2:
                r2 = (256 - p2) / 256
                if abs(r1 - r2) > 0.30 + 1e-3:
                    continue
                triplets.append((p0, p1, p2))
    return triplets


# ============================================================
# 2. Per-candidate pipeline
# ============================================================

PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
ORIG_CKPT_DIR = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
CACHE_ROOT = REPO_ROOT / "models/p0_random_cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def signature(s0, s1, s2):
    return f"{s0:03d}_{s1:03d}_{s2:03d}"


def prune_ckpt(s0, s1, s2, gpu="0"):
    """Run structural_prune_pyramid → produce ckpt without finetune."""
    sig = signature(s0, s1, s2)
    out_dir = CACHE_ROOT / f"prune_{sig}"
    if (out_dir / "config.yaml").exists() and (out_dir / "net_epoch_bestval_at23.pth").exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "tools/structural_prune_pyramid.py",
        "--orig-dir", ORIG_CKPT_DIR,
        "--out-dir", str(out_dir),
        "--num-filters-new", f"{s0},{s1},{s2}",
    ]
    env = {"CUDA_VISIBLE_DEVICES": gpu}
    import os
    env = {**os.environ, **env}
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"prune failed: {r.stderr[-500:]}")
    return out_dir


def export_onnx(prune_dir, gpu="0"):
    """Run export_onnx_pyramid_collab → ONNX."""
    sig = prune_dir.name.replace("prune_", "")
    onnx = CACHE_ROOT / f"onnx_{sig}.onnx"
    if onnx.exists():
        return onnx
    cmd = [
        PYTHON, "tools/export_onnx_pyramid_collab.py",
        "--ckpt", str(prune_dir / "net_epoch_bestval_at23.pth"),
        "--hypes", str(prune_dir / "config.yaml"),
        "--out", str(onnx),
        "--feat-h", "128", "--feat-w", "256",
    ]
    import os
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"export failed: {r.stderr[-500:]}")
    return onnx


def build_bench(onnx, precision, gpu="0"):
    """Build TRT engine + bench lat. Returns dict with p50_ms / engine_size_mb."""
    sig = onnx.stem.replace("onnx_", "")
    engine = CACHE_ROOT / f"engine_{sig}_{precision}.engine"
    report = CACHE_ROOT / f"bench_{sig}_{precision}.json"
    if report.exists():
        with open(report) as f:
            return json.load(f)
    cmd = [
        PYTHON, "scripts/phase1/m4_8_trt_build_bench.py",
        "--onnx", str(onnx),
        "--precision", precision,
        "--engine", str(engine),
        "--report", str(report),
        "--input-shape", "2,64,128,256",
        "--extra-input-shape", "t_ego:2,2,3",
        "--n-warmup", "100", "--n-measure", "200",
    ]
    if precision == "int8":
        cmd += [
            "--calib-multi", "spatial_features:calibration/pyramid_dair_collab_spatial.npy",
            "--calib-multi", "t_ego:calibration/pyramid_dair_collab_tego.npy",
            "--calib-cache", str(CACHE_ROOT / f"calib_{sig}.cache"),
        ]
    import os
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"build/bench failed: {r.stderr[-500:]}")
    with open(report) as f:
        return json.load(f)


# ============================================================
# 3. Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-candidates", type=int, default=50,
                   help="how many triplets to sample (max 98)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="4")
    p.add_argument("--precisions", default="fp16,int8",
                   help="comma-separated precisions to bench per triplet")
    p.add_argument("--output", default="data/pyramid_random_bench.parquet")
    args = p.parse_args()

    import random
    random.seed(args.seed)

    triplets = enumerate_triplets()
    print(f"[P0] enumerated {len(triplets)} valid triplets (gradient + pow2 filtered)")

    # Sample + ensure baseline (64,128,256) included
    if (64, 128, 256) in triplets:
        triplets_sample = [(64, 128, 256)] + random.sample(
            [t for t in triplets if t != (64, 128, 256)],
            min(args.n_candidates - 1, len(triplets) - 1)
        )
    else:
        triplets_sample = random.sample(triplets, min(args.n_candidates, len(triplets)))
    print(f"[P0] sampled {len(triplets_sample)} triplets")

    precs = [p.strip() for p in args.precisions.split(",")]
    rows = []
    t_total_start = time.time()
    for i, (s0, s1, s2) in enumerate(triplets_sample, 1):
        sig = signature(s0, s1, s2)
        rates = [(64-s0)/64, (128-s1)/128, (256-s2)/256]
        print(f"\n[{i}/{len(triplets_sample)}] triplet ({s0},{s1},{s2}) "
              f"rates={[round(r,2) for r in rates]}")
        t0 = time.time()
        try:
            prune_dir = prune_ckpt(s0, s1, s2, gpu=args.gpu)
            print(f"  prune done +{time.time()-t0:.1f}s")
            t1 = time.time()
            onnx = export_onnx(prune_dir, gpu=args.gpu)
            print(f"  onnx done +{time.time()-t1:.1f}s")
            for prec in precs:
                t2 = time.time()
                report = build_bench(onnx, prec, gpu=args.gpu)
                print(f"  {prec} done +{time.time()-t2:.1f}s "
                      f"p50={report['p50_ms']:.3f} ms engine={report['engine_size_mb']:.2f} MB")
                rows.append({
                    "triplet_sig": sig,
                    "stage0_planes": s0,
                    "stage1_planes": s1,
                    "stage2_planes": s2,
                    "stage0_rate": round(rates[0], 4),
                    "stage1_rate": round(rates[1], 4),
                    "stage2_rate": round(rates[2], 4),
                    "precision": prec,
                    "lat_p50_ms": report["p50_ms"],
                    "lat_p99_ms": report["p99_ms"],
                    "lat_mean_ms": report["mean_ms"],
                    "engine_size_mb": report["engine_size_mb"],
                    "build_secs": report.get("build_secs", -1),
                })
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        elapsed = time.time() - t_total_start
        print(f"  cumulative: {elapsed:.0f}s ({elapsed/60:.1f} min) for {i} triplets")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n[P0] wrote {len(df)} rows -> {out}")
    print(df[["triplet_sig", "precision", "lat_p50_ms", "engine_size_mb"]].head(20))


if __name__ == "__main__":
    main()
