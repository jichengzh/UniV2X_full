"""Compute estimated TRT deployment latency for each config.

Logic:
  est_trt_latency = sum over modules:
      module_baseline_latency
      × (1 - prune_rate)            # pruning reduces compute proportionally
      × quant_speedup(bits, target) # quantization gives N× speedup

quant_speedup ratios (industry standard for backbone/transformer modules on GPU):
  FP32 (32-bit):  1.0 × (no speedup)
  FP16 (16-bit):  2.0 × (Tensor Cores)
  INT8 (8-bit):   4.0 × (Tensor Cores INT8 path)
  INT4 (4-bit):   8.0 × (theoretical, hardware-dependent)
  W-only INT8:    1.5 × (only weight load benefits, compute is FP)

Adds est_trt_latency_ms column to stage5_baseline_v3.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/phase4/stage5_baseline_v3.csv"
LATENCY_DIR = ROOT / "output/plan_b/latency"

# Read baseline module latencies (from baseline.json)
def load_baseline_modules() -> dict:
    p = LATENCY_DIR / "baseline.json"
    d = json.loads(p.read_text())
    mods = d.get("modules", {})
    return {
        "backbone": mods.get("backbone", {}).get("mean", 30.0),
        "neck": mods.get("neck", {}).get("mean", 1.3),
        "bev_encoder": mods.get("bev_encoder", {}).get("mean", 50.0),
        "decoder": mods.get("track_head_decoder", {}).get("mean", 7.0),
        "seg_head": mods.get("seg_head", {}).get("mean", 50.0),
        "ego_other": d.get("e2e_ms_mean", 515.0) - sum([
            mods.get("backbone", {}).get("mean", 30.0),
            mods.get("neck", {}).get("mean", 1.3),
            mods.get("bev_encoder", {}).get("mean", 50.0),
            mods.get("track_head_decoder", {}).get("mean", 7.0),
            mods.get("seg_head", {}).get("mean", 50.0),
        ]),
    }


def quant_speedup(bits: int, target: str) -> float:
    """How much faster the module gets after quantization."""
    if bits >= 32:
        return 1.0
    if bits == 16:
        return 2.0
    if bits == 8:
        return 1.5 if target == "W" else 4.0
    if bits == 4:
        return 8.0
    return 1.0  # default


def compute_row_lat(row: pd.Series, base: dict) -> float:
    """Compute estimated TRT e2e latency for one config."""
    bb_speedup = quant_speedup(int(row.get("q_bits__backbone", 32)), str(row.get("q_target", "none")))
    enc_speedup = quant_speedup(int(row.get("q_bits__encoder", 32)), str(row.get("q_target", "none")))
    dec_speedup = quant_speedup(int(row.get("q_bits__decoder", 32)), str(row.get("q_target", "none")))
    heads_speedup = quant_speedup(int(row.get("q_bits__heads", 32)), str(row.get("q_target", "none")))

    bb_lat = base["backbone"] * (1.0 - float(row.get("prune_rate__backbone", 0.0))) / bb_speedup
    neck_lat = base["neck"]  # neck never pruned/quantized in our space
    enc_lat = base["bev_encoder"] * (1.0 - float(row.get("prune_rate__encoder_ffn", 0.0))) / enc_speedup
    dec_lat = base["decoder"] * (1.0 - float(row.get("prune_rate__decoder_ffn", 0.0))) / dec_speedup
    heads_lat = base["seg_head"] * (1.0 - float(row.get("prune_rate__heads_mid", 0.0))) / heads_speedup
    other_lat = base["ego_other"]

    return bb_lat + neck_lat + enc_lat + dec_lat + heads_lat + other_lat


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    base = load_baseline_modules()
    print(f"Baseline modules: {base}")
    print(f"Baseline sum: {sum(base.values()):.1f}ms")

    df["est_trt_latency_ms"] = df.apply(lambda r: compute_row_lat(r, base), axis=1)
    df.to_csv(CSV_PATH, index=False)

    print(f"\nWrote {len(df)} rows × {len(df.columns)} cols")
    print(df[["config_id", "amota", "lat_e2e_ms", "est_trt_latency_ms"]].sort_values("est_trt_latency_ms").to_string(index=False))
    print(f"\nest_trt_latency_ms range: [{df['est_trt_latency_ms'].min():.1f}, {df['est_trt_latency_ms'].max():.1f}]")


if __name__ == "__main__":
    main()
