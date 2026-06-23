"""Aggregate Plan C tiny metrics into LightGBM-ready CSV.

Reads:
  - logs/plan_c/<id>.log (tiny PyTorch eval logs containing car_AP_4m metrics)
  - logs/tiny_validation/baseline_amota.log (baseline)
  - prune_configs/active_<id>.json + quant_configs/active_<id>.json (config schema)

Writes:
  - data/phase4/stage5_baseline_v4_tiny.csv (configs + car_AP_4m + mAP)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRUNE_DIR = ROOT / "prune_configs"
QUANT_DIR = ROOT / "quant_configs"
LOG_DIR = ROOT / "logs/plan_c"
BASELINE_LOG = ROOT / "logs/tiny_validation/baseline_amota.log"
OUT_CSV = ROOT / "data/phase4/stage5_baseline_v4_tiny.csv"


def extract_metrics(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    out = {}
    m = re.search(r"car_AP_dist_4\.0'?: ([0-9]+\.[0-9]+)", text)
    if m:
        out["car_AP_4m"] = float(m.group(1))
    m = re.search(r"pts_bbox_NuScenes/mAP\s+([0-9]+\.[0-9]+)", text)
    if m:
        out["mAP"] = float(m.group(1))
    m = re.search(r"pts_bbox_NuScenes/amota\s+([0-9]+\.[0-9]+)", text)
    if m:
        out["amota"] = float(m.group(1))
    return out


def load_prune(cfg_id: str) -> dict:
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


def load_quant(cfg_id: str) -> dict:
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
    return {
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


def main():
    rows = []
    cfgs = ["baseline", "B1", "B2", "B3", "C1", "C2", "C3", "C4",
            "D1", "D2", "D3", "D4", "D5",
            "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12",
            "A1", "A2", "A3", "A4", "A5", "A6"]

    for cfg in cfgs:
        log = BASELINE_LOG if cfg == "baseline" else (LOG_DIR / f"{cfg}.log")
        metrics = extract_metrics(log)
        if "car_AP_4m" not in metrics:
            print(f"  skip {cfg}: no metrics in log", file=sys.stderr)
            continue

        if cfg == "baseline":
            features = {
                "prune_rate__backbone": 0.0,
                "prune_rate__encoder_ffn": 0.0, "prune_rate__encoder_attn": 0.0,
                "prune_rate__encoder_heads": 0.0,
                "prune_rate__decoder_ffn": 0.0, "prune_rate__decoder_attn": 0.0,
                "prune_rate__decoder_heads": 0.0,
                "prune_rate__heads_mid": 0.0, "decoder_num_layers": 6,
                "q_bits__global_w": 32, "q_bits__global_a": 32,
                "q_bits__backbone": 32, "q_bits__encoder": 32,
                "q_bits__decoder": 32, "q_bits__heads": 32, "q_bits__v2x_comm": 32,
                "q_target": "none",
                "q_granularity_w": "per_tensor", "q_granularity_a": "per_tensor",
            }
        else:
            features = {**load_prune(cfg), **load_quant(cfg)}

        features["config_id"] = cfg
        features["car_AP_4m"] = metrics["car_AP_4m"]
        features["mAP"] = metrics.get("mAP")
        features["amota"] = metrics.get("amota", 0.0)
        features["source"] = "plan_c_tiny"
        rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    print(f"car_AP_4m range: [{df.car_AP_4m.min():.4f}, {df.car_AP_4m.max():.4f}] (span {df.car_AP_4m.max()-df.car_AP_4m.min():.4f})")
    print()
    print(df[["config_id", "car_AP_4m", "prune_rate__encoder_ffn",
              "q_bits__global_w", "q_target"]].sort_values("car_AP_4m", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
