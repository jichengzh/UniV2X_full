"""Augment e2e_bench_v1.csv with all search-space dimensions + AP + speedup.

Reference: paper_learning/2. AAAI最终故事/搜索空间一览.md

Added columns:
  # B 剪枝 (per-stage 已有 stage0/1/2_planes)
  prune_object       — channel / 2:4 / none (all current anchors are "channel" since
                       T1-T8 vary planes; T1_base technically "none" with full planes)
  sparse_mask        — dense / mask_s0_only / 2:4_all / ... (all current = "dense")

  # Q 量化
  q_bits             — FP32 / FP16 / INT8 (derived from prec_flag)
  q_bits_per_stage   — (s0,s1,s2) for Q1' per-stage; NaN for global Q
  q_granularity      — per-tensor / per-channel / none
  q_object           — W-only / W+A / none

  # D 部署
  d_scheme           — single IP routing scheme (4090 = GPU)
  d_tactic           — TRT tactic_sources (default / no_cudnn / cublas_lt / all_enabled / edge_only)
  d_workspace_gb     — int 1/2/4/8/16

  # Hardware
  hardware           — rtx4090 / orin_agx_64gb

  # Performance (AP from class_a_pyramid_full.parquet via triplet × q_tag match)
  ap30 / ap50 / ap70

  # Speedup (relative to T1_base Q_fp32 baseline)
  speedup_e2e_vs_baseline   = lat_e2e_baseline / lat_e2e_current
  speedup_trt_vs_baseline   = lat_trt_baseline / lat_trt_current
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
BENCH_DIR = REPO_ROOT / "paper_learning/2. AAAI最终故事/data"
AP_SOURCE = REPO_ROOT / "data/_by_class/class_a_pyramid_full.parquet"

# Triplet name mapping between e2e_bench_v1 and class_a (T7/T8 differ)
TRIPLET_NAME_MAP = {
    "T7_wide_shallow": "T7_wide",
    "T8_narrow_deep": "T8_deep",
}


def derive_q_dims(prec_flag: str, q_tag: str) -> dict:
    """Map prec_flag + q_tag to (q_bits, q_bits_per_stage, q_granularity, q_object).

    Decisions reflect TRT default behavior for global precision modes:
      * FP32: no quantization, all fields None
      * FP16: TF/HMMA tensor cores, no calibration. q_granularity='per-tensor' (no
              calibration table; weights kept at FP16). q_object='W+A' since FP16 affects both.
      * INT8 (minmax): IInt8MinMaxCalibrator → per-tensor activation, per-tensor weight
              by default. q_object='W+A'.
      * INT8 (entropy): IInt8EntropyCalibrator2 → similar default per-tensor (TRT 10
              chooses per-channel for some Conv weights automatically).
    """
    if prec_flag == "fp32":
        return {"q_bits": "FP32", "q_bits_per_stage": None,
                "q_granularity": "none", "q_object": "none"}
    if prec_flag == "fp16":
        return {"q_bits": "FP16", "q_bits_per_stage": None,
                "q_granularity": "per-tensor", "q_object": "W+A"}
    if prec_flag == "int8":
        # Both minmax and entropy use TRT default per-tensor activation +
        # per-channel weight (TRT 10 default for Conv layers).
        return {"q_bits": "INT8", "q_bits_per_stage": None,
                "q_granularity": "per-tensor", "q_object": "W+A"}
    return {"q_bits": "Unknown", "q_bits_per_stage": None,
            "q_granularity": "Unknown", "q_object": "Unknown"}


def derive_b_dims(planes: tuple) -> dict:
    """Pruning object label."""
    if planes == (64, 128, 256):
        return {"prune_object": "none", "sparse_mask": "dense"}
    return {"prune_object": "channel", "sparse_mask": "dense"}


def lookup_ap(triplet: str, q_tag: str, ap_df: pd.DataFrame) -> tuple:
    """Match (triplet, q_tag) against class_a AP source.

    Returns (ap30, ap50, ap70) or (NaN, NaN, NaN) if no match.
    Uses D_ws4 (workspace=4GB) to match our e2e bench's --workspace-mb 4096.
    """
    src_triplet = TRIPLET_NAME_MAP.get(triplet, triplet)
    rows = ap_df[(ap_df["triplet_tag"] == src_triplet)
                 & (ap_df["q_label"] == q_tag)
                 & (ap_df["d_label"] == "D_ws4")]
    if len(rows) == 0:
        # Try any D since AP is roughly D-invariant
        rows = ap_df[(ap_df["triplet_tag"] == src_triplet)
                     & (ap_df["q_label"] == q_tag)]
    if len(rows) == 0:
        return (np.nan, np.nan, np.nan)
    r = rows.iloc[0]
    return (float(r["ap30"]), float(r["ap50"]), float(r["ap70"]))


def main():
    csv_path = BENCH_DIR / "e2e_bench_v1.csv"
    parquet_path = BENCH_DIR / "e2e_bench_v1.parquet"
    df = pd.read_csv(csv_path)
    print(f"loaded e2e_bench: {len(df)} rows, {len(df.columns)} cols")

    ap_df = pd.read_parquet(AP_SOURCE)
    print(f"loaded AP source: {len(ap_df)} rows from {AP_SOURCE.name}")

    # ─── 1. Baseline = T1_base + Q_fp32 ───────────────────────────────────────
    base_row = df[(df["triplet"] == "T1_base") & (df["q_tag"] == "Q_fp32")]
    if len(base_row) == 0:
        print("[err] no T1_base Q_fp32 row, cannot compute speedup")
        sys.exit(1)
    base_e2e = float(base_row["lat_e2e_mean_ms"].iloc[0])
    base_trt = float(base_row["lat_trt_mean_ms"].iloc[0])
    print(f"baseline: T1_base+Q_fp32 -> e2e={base_e2e:.3f}ms, trt={base_trt:.3f}ms")

    # ─── 2. Build extension columns ───────────────────────────────────────────
    rows = []
    for _, r in df.iterrows():
        planes = (int(r["stage0_planes"]), int(r["stage1_planes"]), int(r["stage2_planes"]))
        out = dict(r)

        # B dims
        out.update(derive_b_dims(planes))

        # Q dims
        out.update(derive_q_dims(r["prec_flag"], r["q_tag"]))

        # D dims (constant across all 32 anchors in v1: GPU + default tactic + 4GB workspace)
        out["d_scheme"] = "GPU"
        out["d_tactic"] = "default"
        out["d_workspace_gb"] = 4

        # Hardware
        out["hardware"] = "rtx4090"

        # AP from class_a source
        ap30, ap50, ap70 = lookup_ap(r["triplet"], r["q_tag"], ap_df)
        out["ap30"] = ap30
        out["ap50"] = ap50
        out["ap70"] = ap70

        # Speedup
        out["speedup_e2e_vs_baseline"] = base_e2e / float(r["lat_e2e_mean_ms"]) \
            if pd.notna(r["lat_e2e_mean_ms"]) else np.nan
        out["speedup_trt_vs_baseline"] = base_trt / float(r["lat_trt_mean_ms"]) \
            if pd.notna(r["lat_trt_mean_ms"]) else np.nan

        # Throughput (single-IP single-instance)
        if pd.notna(r["lat_e2e_mean_ms"]):
            out["throughput_fps"] = 1000.0 / float(r["lat_e2e_mean_ms"])
        else:
            out["throughput_fps"] = np.nan

        rows.append(out)

    out_df = pd.DataFrame(rows)
    # Reorder columns: coords first, then dims, then perf, then resources, then meta
    cols_order = [
        # ─ ID + coord (B) ─
        "triplet", "stage0_planes", "stage1_planes", "stage2_planes",
        "prune_object", "sparse_mask",
        # ─ coord (Q) ─
        "q_tag", "prec_flag", "q_bits", "q_bits_per_stage",
        "q_granularity", "q_object",
        # ─ coord (D) ─
        "d_scheme", "d_tactic", "d_workspace_gb",
        # ─ Hardware ─
        "hardware", "device",
        # ─ Capacity / sample size ─
        "max_voxels", "n_collected", "n_skipped",
        "real_voxels_mean", "real_voxels_p99",
        # ─ Performance: latency ─
        "lat_e2e_mean_ms", "lat_e2e_p50_ms", "lat_e2e_p99_ms",
        "lat_trt_mean_ms", "lat_trt_p50_ms", "lat_trt_p99_ms",
        "lat_postproc_mean_ms", "lat_postproc_p50_ms",
        # ─ Performance: speedup + throughput ─
        "speedup_e2e_vs_baseline", "speedup_trt_vs_baseline", "throughput_fps",
        # ─ Performance: AP ─
        "ap30", "ap50", "ap70",
        # ─ Resource ─
        "engine_size_mb", "build_secs",
        # ─ Meta ─
        "build_success", "fail_reason", "ts",
    ]
    # Keep only cols that exist in df
    cols_order = [c for c in cols_order if c in out_df.columns]
    # Add any leftover at end
    cols_order += [c for c in out_df.columns if c not in cols_order]
    out_df = out_df[cols_order]

    # Round latencies to 3 decimals for readability
    for c in out_df.columns:
        if c.startswith("lat_") or c.startswith("speedup_") or c in ("throughput_fps",):
            if c in out_df.columns:
                out_df[c] = pd.to_numeric(out_df[c], errors="coerce").round(4)

    out_df.to_csv(csv_path, index=False)
    out_df.to_parquet(parquet_path, index=False)
    print(f"\nwrote {len(out_df)} rows × {len(out_df.columns)} cols to:")
    print(f"  {csv_path}")
    print(f"  {parquet_path}")

    # Print summary
    print("\n=== speedup summary (vs T1_base Q_fp32) ===")
    print(out_df[["triplet", "q_tag", "lat_e2e_mean_ms", "lat_trt_mean_ms",
                  "speedup_e2e_vs_baseline", "speedup_trt_vs_baseline",
                  "ap50"]].to_string(index=False))


if __name__ == "__main__":
    main()
