"""E2E bench v1 orchestrator: 8 triplet × 4 prec = 32 anchor.

For each triplet:
  1. Export e2e ONNX (if not cached)
  2. Build engines for FP32, FP16, INT8_mm, INT8_ent (skip if cached)
  3. Bench each engine via e2e_bench_pyramid.py
  4. Append anchor row to combined parquet/csv

Output:
  paper_learning/2. AAAI最终故事/data/e2e_bench_v1.parquet
  paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
DATA_OUT = REPO_ROOT / "paper_learning/2. AAAI最终故事/data"
ONNX_CACHE = REPO_ROOT / "models/e2e_cache"
ENGINE_CACHE = REPO_ROOT / "models/e2e_cache"
ONNX_CACHE.mkdir(parents=True, exist_ok=True)
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"

CALIB_DIR = REPO_ROOT / "calibration/pyramid_dair_e2e_32k"
DAIR_ROOT = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
MAX_VOX = 32000

TRIPLETS = [
    # (tag, ckpt, hypes, planes_sig)
    ("T1_base", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml",
     (64, 128, 256)),
    ("T2_p25", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch_bestval_at25.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/config.yaml",
     (48, 96, 192)),
    ("T3_p37", str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/config.yaml"),
     (40, 80, 160)),
    ("T4_p50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at23.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/config.yaml",
     (32, 64, 128)),
    ("T5_p62", str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/config.yaml"),
     (24, 56, 128)),
    ("T6_p75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at25.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/config.yaml",
     (16, 32, 64)),
    ("T7_wide_shallow", str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/config.yaml"),
     (48, 64, 128)),
    ("T8_narrow_deep", str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/config.yaml"),
     (24, 48, 192)),
]

# (q_tag, prec_flag, extra build args)
Q_CONFIGS = [
    ("Q_fp32", "fp32", []),
    ("Q_fp16", "fp16", []),
    ("Q_int8_mm", "int8", ["--calibrator", "minmax"]),
    ("Q_int8_ent", "int8", ["--calibrator", "entropy"]),
]


def run(cmd, check=True, **kw):
    print(f"[run] {' '.join(cmd[:5])} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0 and check:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"cmd failed rc={r.returncode}")
    return r


def export_onnx(tag, ckpt, hypes, out_path):
    if out_path.exists():
        print(f"[onnx] {tag} cached")
        return
    t0 = time.time()
    run([PYTHON, str(REPO_ROOT / "tools/export_onnx_pyramid_e2e.py"),
         "--ckpt", ckpt, "--hypes", hypes,
         "--max-voxels", str(MAX_VOX),
         "--out", str(out_path)])
    print(f"[onnx] {tag} exported in {time.time()-t0:.0f}s")


def build_engine(tag, q_tag, prec_flag, extra, onnx_path, engine_path,
                 calib_cache_path):
    if engine_path.exists():
        print(f"[build] {tag}/{q_tag} cached")
        # Try to read previous build report
        report = engine_path.with_suffix(".build.json")
        if report.exists():
            return json.loads(report.read_text())
        return {"build_secs": None, "engine_size_mb": engine_path.stat().st_size / 1e6}
    t0 = time.time()
    report_path = engine_path.with_suffix(".build.json")
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx_path),
           "--precision", prec_flag,
           "--engine", str(engine_path),
           "--report", str(report_path),
           "--workspace-mb", "4096",
           "--skip-bench"]  # bench in m4_8 uses random data → OOB on int32 inputs
    if prec_flag == "int8":
        cmd += ["--calib-cache", str(calib_cache_path),
                "--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
                "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
                "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
                "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
                "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy"]
    cmd += extra
    run(cmd)
    rep = json.loads(report_path.read_text())
    print(f"[build] {tag}/{q_tag}: {time.time()-t0:.0f}s, "
          f"engine={rep.get('engine_size_mb', 0):.1f}MB")
    return rep


def bench_engine(tag, q_tag, ckpt_dir, engine_path, bench_report_path,
                 n_warmup, n_measure):
    if bench_report_path.exists():
        print(f"[bench] {tag}/{q_tag} cached")
        return json.loads(bench_report_path.read_text())
    t0 = time.time()
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase2/e2e_bench_pyramid.py"),
           "--engine", str(engine_path),
           "--ckpt-dir", str(ckpt_dir),
           "--dair-root", DAIR_ROOT,
           "--max-voxels", str(MAX_VOX),
           "--n-warmup", str(n_warmup),
           "--n-measure", str(n_measure),
           "--report", str(bench_report_path)]
    run(cmd)
    rep = json.loads(bench_report_path.read_text())
    print(f"[bench] {tag}/{q_tag}: e2e={rep['lat_e2e_ms']['mean']:.2f}ms "
          f"trt={rep['lat_trt_ms']['mean']:.2f}ms "
          f"post={rep['lat_postproc_ms']['mean']:.2f}ms "
          f"({time.time()-t0:.0f}s)")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-measure", type=int, default=80)
    ap.add_argument("--triplets", default="all",
                    help="comma-separated list of triplet tags, or 'all'")
    ap.add_argument("--q-configs", default="all")
    args = ap.parse_args()

    triplets = TRIPLETS if args.triplets == "all" else \
        [t for t in TRIPLETS if t[0] in args.triplets.split(",")]
    q_configs = Q_CONFIGS if args.q_configs == "all" else \
        [q for q in Q_CONFIGS if q[0] in args.q_configs.split(",")]

    rows = []
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    parquet_path = DATA_OUT / "e2e_bench_v1.parquet"
    csv_path = DATA_OUT / "e2e_bench_v1.csv"

    # Load existing if present (for resume)
    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        existing_keys = set(zip(existing["triplet"], existing["q_tag"]))
        rows = existing.to_dict("records")
        print(f"[resume] loaded {len(rows)} existing rows")
    else:
        existing_keys = set()

    for tag, ckpt, hypes, planes in triplets:
        ckpt_dir = str(Path(ckpt).parent)
        onnx_path = ONNX_CACHE / f"{tag}.onnx"
        try:
            export_onnx(tag, ckpt, hypes, onnx_path)
        except Exception as e:
            print(f"[err] {tag} onnx export failed: {e}")
            continue

        for q_tag, prec_flag, extra in q_configs:
            if (tag, q_tag) in existing_keys:
                print(f"[skip] {tag}/{q_tag} done")
                continue
            engine_path = ONNX_CACHE / f"{tag}_{q_tag}.engine"
            calib_cache = ONNX_CACHE / f"{tag}_{q_tag}_calib.cache"
            bench_path = ONNX_CACHE / f"{tag}_{q_tag}_bench.json"

            try:
                bld = build_engine(tag, q_tag, prec_flag, extra,
                                   onnx_path, engine_path, calib_cache)
                bench = bench_engine(tag, q_tag, ckpt_dir, engine_path,
                                     bench_path, args.n_warmup, args.n_measure)
            except Exception as e:
                print(f"[err] {tag}/{q_tag} failed: {e}")
                rows.append({
                    "triplet": tag, "q_tag": q_tag, "prec_flag": prec_flag,
                    "stage0_planes": planes[0], "stage1_planes": planes[1],
                    "stage2_planes": planes[2],
                    "build_success": False, "fail_reason": str(e)[:200],
                    "ts": datetime.now().isoformat(timespec="seconds"),
                })
                continue

            row = {
                "triplet": tag, "q_tag": q_tag, "prec_flag": prec_flag,
                "stage0_planes": planes[0], "stage1_planes": planes[1],
                "stage2_planes": planes[2],
                "max_voxels": MAX_VOX,
                "n_collected": bench["n_collected"],
                "n_skipped": bench["n_skipped"],
                "lat_e2e_mean_ms": bench["lat_e2e_ms"]["mean"],
                "lat_e2e_p50_ms": bench["lat_e2e_ms"]["p50"],
                "lat_e2e_p99_ms": bench["lat_e2e_ms"]["p99"],
                "lat_trt_mean_ms": bench["lat_trt_ms"]["mean"],
                "lat_trt_p50_ms": bench["lat_trt_ms"]["p50"],
                "lat_trt_p99_ms": bench["lat_trt_ms"]["p99"],
                "lat_postproc_mean_ms": bench["lat_postproc_ms"]["mean"],
                "lat_postproc_p50_ms": bench["lat_postproc_ms"]["p50"],
                "real_voxels_mean": bench["real_voxels"]["mean"],
                "real_voxels_p99": bench["real_voxels"]["p99"],
                "engine_size_mb": bld.get("engine_size_mb"),
                "build_secs": bld.get("build_secs"),
                "build_success": True, "fail_reason": "",
                "device": bench["device"],
                "ts": datetime.now().isoformat(timespec="seconds"),
            }
            rows.append(row)
            existing_keys.add((tag, q_tag))

            # Persist after each anchor (resume-safe)
            df = pd.DataFrame(rows)
            df.to_parquet(parquet_path, index=False)
            df.to_csv(csv_path, index=False)
            print(f"  [persist] {len(rows)} rows -> {csv_path.name}")

    print(f"\n=== e2e bench v1 complete: {len(rows)} rows ===")
    print(f"  parquet: {parquet_path}")
    print(f"  csv:     {csv_path}")
    df = pd.DataFrame(rows)
    if not df.empty:
        ok = df[df["build_success"] == True]
        print(f"\n  successes: {len(ok)} / {len(df)}")
        if len(ok) > 0:
            print(ok[["triplet", "q_tag", "lat_e2e_mean_ms", "lat_trt_mean_ms",
                      "lat_postproc_mean_ms", "engine_size_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
