"""Compute estimated TRT deployment latency v4 — calibrated against real Plan C measurements.

Updated speedup factors (replacing v1 theoretical 4×):

quant_speedup_factors (calibrated against tiny BEV encoder Plan C measurements 2026-05-04):
  PyTorch FP32:        1.0  (no speedup)
  PyTorch fake-quant:  1.0  (overhead, often slower)
  TRT FP16:            1.66 (实测 tiny BEV: 25.95→15.67 ms)
  TRT INT8:            1.66 (实测同 FP16, plugin FP16-only限制)

D-space (3 dims):
  d_runtime ∈ {pytorch_fp32, pytorch_fakequant, trt_fp16, trt_int8}
  d_pipelined_get_bevs: binary (PyTorch -6%, TRT -22% via backbone-BEV overlap)
  d_temporal_cache_int8: binary (AMOTA +0.006, latency 0)

Module-level baselines (R101+DCN, measured in baseline.json):
  backbone (R101+DCN): 30.66 ms (FP32) — DCN forces FP16 only in TRT
  bev_encoder:         50.54 ms (FP32) → 15.67 ms TRT FP16
  decoder:              7.63 ms (FP32)
  seg_head:            53.68 ms (FP32) → contains heads
  other (Python loop): 372 ms (e2e tracking, mostly Python/CPU)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/phase4/stage5_baseline_v3.csv"

# Calibrated TRT speedup vs PyTorch FP32 (实测 2026-05-04)
TRT_SPEEDUP = {
    "pytorch_fp32":      1.0,
    "pytorch_fakequant": 1.0,    # fake-quant overhead ≈ no speedup
    "trt_fp16":          1.66,   # 实测 BEV: 25.95→15.67 ms
    "trt_int8":          1.66,   # 实测 ≈ FP16 (plugin limit)
}

# Pipelined GetBevs speedup factor (1.3 v4 实测)
PIPELINED_SPEEDUP = {
    "pytorch_fp32":      0.94,   # 1.06× → 6% reduction
    "pytorch_fakequant": 0.94,
    "trt_fp16":          0.78,   # backbone+BEV overlap = 22% reduction
    "trt_int8":          0.78,
}

# Backbone TRT speedup is limited by DCN (FP16 only, no INT8 kernel)
BACKBONE_TRT_SPEEDUP = {
    "pytorch_fp32":      1.0,
    "pytorch_fakequant": 1.0,
    "trt_fp16":          1.4,    # DCN plugin FP16
    "trt_int8":          1.4,    # No INT8 path for DCN
}


def load_baseline_modules() -> dict:
    """Module-level baseline latencies (R101+DCN, measured in baseline.json)."""
    return {
        "backbone": 30.664,
        "neck":      1.302,
        "bev_encoder": 50.541,
        "decoder":   7.633,
        "seg_head": 53.680,
        "other":  371.724,    # = e2e_total - sum(modules)
    }


def compute_lat_v4(row: pd.Series, base: dict) -> float:
    """Compute estimated TRT e2e latency for one config given d_runtime + d_pipelined."""
    runtime = str(row.get("d_runtime", "pytorch_fp32"))
    pipelined = bool(row.get("d_pipelined_get_bevs", False))
    cache_int8 = bool(row.get("d_temporal_cache_int8", False))

    # Module-level latencies after pruning + quantization + runtime
    bb_speedup = BACKBONE_TRT_SPEEDUP[runtime]
    other_speedup = TRT_SPEEDUP[runtime]
    bb_lat = base["backbone"] * (1.0 - float(row.get("prune_rate__backbone", 0))) / bb_speedup
    enc_lat = base["bev_encoder"] * (1.0 - float(row.get("prune_rate__encoder_ffn", 0))) / other_speedup
    dec_lat = base["decoder"] * (1.0 - float(row.get("prune_rate__decoder_ffn", 0))) / other_speedup
    heads_lat = base["seg_head"] * (1.0 - float(row.get("prune_rate__heads_mid", 0))) / other_speedup
    neck_lat = base["neck"]
    other_lat = base["other"]

    e2e = bb_lat + neck_lat + enc_lat + dec_lat + heads_lat + other_lat

    # Pipelined backbone-BEV overlap (saves max(bb_lat, enc_lat) → both run concurrent)
    if pipelined:
        e2e *= PIPELINED_SPEEDUP[runtime]

    # cache_int8 doesn't affect latency
    return e2e


def expand_with_dspace(df: pd.DataFrame) -> pd.DataFrame:
    """For each base config (23 rows), generate D-space variants:
    4 runtimes × 2 pipelined × 2 cache_int8 = 16 D combinations.
    Output: 23 × 16 = 368 rows."""
    runtimes = ["pytorch_fp32", "pytorch_fakequant", "trt_fp16", "trt_int8"]
    pipelined_vals = [False, True]
    cache_int8_vals = [False, True]

    base = load_baseline_modules()
    rows = []
    for _, row in df.iterrows():
        for rt in runtimes:
            for pipe in pipelined_vals:
                for ci8 in cache_int8_vals:
                    new_row = row.to_dict()
                    new_row["d_runtime"] = rt
                    new_row["d_pipelined_get_bevs"] = int(pipe)
                    new_row["d_temporal_cache_int8"] = int(ci8)
                    new_row["est_trt_latency_v4_ms"] = compute_lat_v4(pd.Series(new_row), base)
                    # AMOTA boost from temporal cache
                    if ci8 and "amota" in new_row and pd.notna(new_row.get("amota")):
                        new_row["amota_v4"] = new_row["amota"] + 0.006
                    else:
                        new_row["amota_v4"] = new_row.get("amota")
                    new_row["config_full_id"] = f"{row['config_id']}__{rt}__pipe{int(pipe)}__cache{int(ci8)}"
                    rows.append(new_row)
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} base configs from {CSV_PATH}")

    expanded = expand_with_dspace(df)
    print(f"Expanded to {len(expanded)} rows (23 base × 16 D-space combinations)")

    out_csv = ROOT / "data/phase4/stage5_v4_dspace.csv"
    expanded.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Summary
    print(f"\nest_trt_latency_v4_ms range: [{expanded.est_trt_latency_v4_ms.min():.1f}, {expanded.est_trt_latency_v4_ms.max():.1f}] ms")
    print(f"  PyTorch FP32 baseline (no D):  {expanded[(expanded.d_runtime=='pytorch_fp32') & (expanded.d_pipelined_get_bevs==0) & (expanded.config_id=='baseline')].est_trt_latency_v4_ms.iloc[0]:.1f} ms")
    print(f"  TRT INT8 + pipelined (best):   {expanded[(expanded.d_runtime=='trt_int8') & (expanded.d_pipelined_get_bevs==1) & (expanded.config_id=='baseline')].est_trt_latency_v4_ms.iloc[0]:.1f} ms")

    # Per-runtime statistics
    print(f"\nPer-runtime min latency (best D config):")
    for rt in ["pytorch_fp32", "pytorch_fakequant", "trt_fp16", "trt_int8"]:
        sub = expanded[expanded.d_runtime == rt]
        print(f"  {rt:20s}  min={sub.est_trt_latency_v4_ms.min():.1f}  max={sub.est_trt_latency_v4_ms.max():.1f}")

    # Speedup vs PyTorch baseline
    pytorch_baseline = expanded[(expanded.d_runtime=='pytorch_fp32') & (expanded.d_pipelined_get_bevs==0) & (expanded.config_id=='baseline')].est_trt_latency_v4_ms.iloc[0]
    best_overall = expanded.est_trt_latency_v4_ms.min()
    print(f"\nMax speedup vs PyTorch baseline: {pytorch_baseline / best_overall:.2f}× (baseline {pytorch_baseline:.1f}ms → best {best_overall:.1f}ms = -{(1-best_overall/pytorch_baseline)*100:.1f}%)")


if __name__ == "__main__":
    main()
