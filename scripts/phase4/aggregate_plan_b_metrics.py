"""Aggregate Plan B per-config metrics into LightGBM-ready CSV.

Reads:
  - data/phase4/stage5_v3/<id>.json  (AMOTA + tracking metrics per config)
  - prune_configs/active_<id>.json   (prune feature schema)
  - quant_configs/active_<id>.json   (quant feature schema)

Writes:
  - data/phase4/stage5_baseline_v3.csv  (config_id, prune features, quant features, AMOTA, ...)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "data/phase4/stage5_v3"
PRUNE_DIR = ROOT / "prune_configs"
QUANT_DIR = ROOT / "quant_configs"
OUT_CSV = ROOT / "data/phase4/stage5_baseline_v3.csv"


def load_prune_features(cfg_id: str) -> dict:
    """Extract prune ratios per module."""
    f = PRUNE_DIR / f"active_{cfg_id}.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {
        "prune_rate__backbone": d.get("backbone", {}).get("channel_pruning_ratio", 0.0),
        "prune_rate__encoder_ffn": 1.0 - d.get("encoder", {}).get("ffn_mid_ratio", 1.0),
        "prune_rate__encoder_attn": d.get("encoder", {}).get("attn_proj_ratio", 0.0),
        "prune_rate__encoder_heads": d.get("encoder", {}).get("head_pruning_ratio", 0.0),
        "prune_rate__decoder_ffn": 1.0 - d.get("decoder", {}).get("ffn_mid_ratio", 1.0),
        "prune_rate__decoder_attn": d.get("decoder", {}).get("attn_proj_ratio", 0.0),
        "prune_rate__decoder_heads": d.get("decoder", {}).get("head_pruning_ratio", 0.0),
        "prune_rate__heads_mid": 1.0 - d.get("heads", {}).get("head_mid_ratio", 1.0),
        "decoder_num_layers": d.get("decoder", {}).get("num_layers", 6),
    }


def load_quant_features(cfg_id: str) -> dict:
    """Extract quant bits per module."""
    f = QUANT_DIR / f"active_{cfg_id}.json"
    if not f.exists():
        return {
            "q_bits__global_w": 32, "q_bits__global_a": 32,
            "q_bits__backbone": 32, "q_bits__encoder": 32,
            "q_bits__decoder": 32, "q_bits__heads": 32,
            "q_bits__v2x_comm": 32,
            "q_target": "none",
            "q_granularity_w": "per_tensor", "q_granularity_a": "per_tensor",
        }
    d = json.loads(f.read_text())
    g = d.get("global", {})
    m = d.get("modules", {})
    out = {
        "q_bits__global_w": g.get("default_w_bits", 32),
        "q_bits__global_a": g.get("default_a_bits", 32),
        "q_bits__backbone": m.get("backbone", {}).get("bits", g.get("default_w_bits", 32)),
        "q_bits__encoder": m.get("encoder", {}).get("bits", g.get("default_w_bits", 32)),
        "q_bits__decoder": m.get("decoder", {}).get("bits", g.get("default_w_bits", 32)),
        "q_bits__heads": m.get("heads", {}).get("bits", g.get("default_w_bits", 32)),
        "q_bits__v2x_comm": m.get("v2x_comm", {}).get("bits", g.get("default_w_bits", 32)),
        "q_target": g.get("default_quant_target", "W+A"),
        "q_granularity_w": g.get("default_w_granularity", "per_tensor"),
        "q_granularity_a": g.get("default_a_granularity", "per_tensor"),
    }
    return out


def main() -> None:
    rows = []
    for json_path in sorted(METRICS_DIR.glob("*.json")):
        cfg_id = json_path.stem
        m = json.loads(json_path.read_text())
        if "amota" not in m:
            print(f"skip {cfg_id}: no AMOTA in metrics", file=sys.stderr)
            continue
        if m.get("has_error"):
            print(f"skip {cfg_id}: has_error=True", file=sys.stderr)
            continue

        # Special handling for P1_xx (manually-imported 1.2 datapoints)
        if cfg_id.startswith("P1_"):
            pct = int(cfg_id[3:])  # 20, 30, 40, 50, 60
            ratio = pct / 100.0
            features = {
                "prune_rate__backbone": 0.0,
                "prune_rate__encoder_ffn": ratio,
                "prune_rate__encoder_attn": 0.0,
                "prune_rate__encoder_heads": 0.0,
                "prune_rate__decoder_ffn": ratio,
                "prune_rate__decoder_attn": 0.0,
                "prune_rate__decoder_heads": 0.0,
                "prune_rate__heads_mid": 0.0,
                "decoder_num_layers": 6,
                "q_bits__global_w": 32, "q_bits__global_a": 32,
                "q_bits__backbone": 32, "q_bits__encoder": 32,
                "q_bits__decoder": 32, "q_bits__heads": 32,
                "q_bits__v2x_comm": 32,
                "q_target": "none",
                "q_granularity_w": "per_tensor", "q_granularity_a": "per_tensor",
                "source": "1.2_P1_FFN",
            }
        else:
            features = load_prune_features(cfg_id)
            features.update(load_quant_features(cfg_id))
            features["source"] = "plan_b_active"

        features["config_id"] = cfg_id
        # Metrics
        for k in ["amota", "amotp", "recall", "mota", "tp", "fp", "fn", "gt", "mAP", "NDS", "car_ap_4m"]:
            features[k] = m.get(k)
        rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")
    print(df[["config_id", "amota", "prune_rate__encoder_ffn", "q_bits__global_w", "source"]].to_string(index=False))
    print(f"\nAMOTA span: [{df['amota'].min():.4f}, {df['amota'].max():.4f}] (Δ={df['amota'].max()-df['amota'].min():.4f})")


if __name__ == "__main__":
    main()
