"""Part A — Per-stage 量化有效性验证 bench.

按 plan2.md Part A 设计: 3 triplet × 6 per-stage config (3-8) = 18 bench 点.
全局 FP16 (config 1) 和全局 INT8 (config 2) 已在 100-anchor parquet 中, 无需重测.

Triplet 选择:
    T_baseline = (64, 128, 256)
    T_prune50  = (32, 64, 136)   # avg_rate ≈ 0.49, 替代不存在的 (32,64,128)
    T_prune75  = (16, 32, 64)

Per-stage config:
    #3 INT8|FP16|FP16  → --mixed-fp16-substr "/resnet/layer1/;/resnet/layer2/"
    #4 FP16|INT8|FP16  → --mixed-fp16-substr "/resnet/layer0/;/resnet/layer2/"
    #5 FP16|FP16|INT8  → --mixed-fp16-substr "/resnet/layer0/;/resnet/layer1/"
    #6 INT8|INT8|FP16  → --mixed-fp16-substr "/resnet/layer2/"   ← H1 主假设
    #7 INT8|FP16|INT8  → --mixed-fp16-substr "/resnet/layer1/"
    #8 FP16|INT8|INT8  → --mixed-fp16-substr "/resnet/layer0/"

输出: data/perstage_quant_bench.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CACHE_ROOT = REPO_ROOT / "models/p0_random_cache"
OUT_DIR = REPO_ROOT / "models/perstage_quant_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIPLETS = [
    ("T_baseline", 64, 128, 256),
    ("T_prune50",  32,  64, 136),
    ("T_prune75",  16,  32,  64),
]

# Each entry: (label, s0_prec, s1_prec, s2_prec, fp16_substrs)
# fp16_substrs is None for global FP16/INT8 (skip — already in main parquet)
PERSTAGE_CONFIGS = [
    ("c3_I_F_F", "INT8", "FP16", "FP16", "/resnet/layer1/;/resnet/layer2/"),
    ("c4_F_I_F", "FP16", "INT8", "FP16", "/resnet/layer0/;/resnet/layer2/"),
    ("c5_F_F_I", "FP16", "FP16", "INT8", "/resnet/layer0/;/resnet/layer1/"),
    ("c6_I_I_F", "INT8", "INT8", "FP16", "/resnet/layer2/"),
    ("c7_I_F_I", "INT8", "FP16", "INT8", "/resnet/layer1/"),
    ("c8_F_I_I", "FP16", "INT8", "INT8", "/resnet/layer0/"),
]

CALIB_SPATIAL = "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO    = "calibration/pyramid_dair_collab_tego.npy"


def run_bench(onnx_path: Path, label: str, fp16_substrs: str, sig: str, gpu: str = "0"):
    """Run m4_8_trt_build_bench with --precision mixed + --mixed-fp16-substr."""
    engine_out = OUT_DIR / f"engine_{sig}_{label}.engine"
    report_out = OUT_DIR / f"bench_{sig}_{label}.json"
    if report_out.exists():
        print(f"  [{label}] cache hit -> {report_out.name}")
        with open(report_out) as f:
            return json.load(f)

    calib_cache = OUT_DIR / f"calib_{sig}_{label}.cache"
    cmd = [
        PYTHON, "scripts/phase1/m4_8_trt_build_bench.py",
        "--onnx", str(onnx_path),
        "--precision", "mixed",
        "--mixed-fp16-substr", fp16_substrs,
        "--engine", str(engine_out),
        "--report", str(report_out),
        "--input-shape", "2,64,128,256",
        "--extra-input-shape", "t_ego:2,2,3",
        "--n-warmup", "100", "--n-measure", "200",
        "--calib-multi", f"spatial_features:{CALIB_SPATIAL}",
        "--calib-multi", f"t_ego:{CALIB_TEGO}",
        "--calib-cache", str(calib_cache),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    print(f"  [{label}] building + bench ...")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                        timeout=900, env=env)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  [{label}] FAILED in {elapsed:.0f}s")
        print(f"    stderr tail: {r.stderr[-800:]}")
        return None
    with open(report_out) as f:
        report = json.load(f)
    print(f"  [{label}] done in {elapsed:.0f}s -> p50={report['p50_ms']:.3f} ms"
          f" engine={report['engine_size_mb']:.2f} MB")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default="0")
    p.add_argument("--output", default="data/perstage_quant_bench.parquet")
    p.add_argument("--triplet-filter", default=None,
                   help="只跑指定 triplet (T_baseline/T_prune50/T_prune75)")
    args = p.parse_args()

    rows = []
    t_total = time.time()

    for tri_label, s0, s1, s2 in TRIPLETS:
        if args.triplet_filter and tri_label != args.triplet_filter:
            continue
        sig = f"{s0:03d}_{s1:03d}_{s2:03d}"
        onnx_path = CACHE_ROOT / f"onnx_{sig}.onnx"
        if not onnx_path.exists():
            print(f"[{tri_label}] missing ONNX {onnx_path} — skip")
            continue
        print(f"\n=== {tri_label}: ({s0},{s1},{s2}) [{sig}] ===")

        for c_label, p0_prec, p1_prec, p2_prec, fp16_substrs in PERSTAGE_CONFIGS:
            report = run_bench(onnx_path, c_label, fp16_substrs, sig, gpu=args.gpu)
            if report is None:
                continue
            rows.append({
                "triplet": tri_label,
                "triplet_sig": sig,
                "stage0_planes": s0, "stage1_planes": s1, "stage2_planes": s2,
                "config_label": c_label,
                "stage0_prec": p0_prec, "stage1_prec": p1_prec, "stage2_prec": p2_prec,
                "fp16_substrs": fp16_substrs,
                "lat_p50_ms": report["p50_ms"],
                "lat_p99_ms": report["p99_ms"],
                "lat_mean_ms": report["mean_ms"],
                "engine_size_mb": report["engine_size_mb"],
                "build_secs": report.get("build_secs", -1),
            })
            elapsed = time.time() - t_total
            print(f"  cumulative {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if not rows:
        print("[ERR] no successful bench points")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"\n[done] wrote {len(df)} rows -> {out_path}")
    print()
    print(df[["triplet", "config_label", "stage0_prec", "stage1_prec", "stage2_prec",
              "lat_p50_ms", "engine_size_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
