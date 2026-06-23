"""LightGBM v4-tiny — train predictors on tiny baseline 26 configs.

Two predictors:
  1. car_AP@4m (proxy for accuracy, replaces AMOTA which is truncated to 0)
  2. trt_bev_ms (real measured TRT BEV encoder latency)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data/phase4/stage5_v4_tiny_full.csv"
OUT_DIR = ROOT / "results"

FEATURES = [
    "prune_rate__backbone", "prune_rate__encoder_ffn", "prune_rate__encoder_attn",
    "prune_rate__encoder_heads", "prune_rate__decoder_ffn", "prune_rate__decoder_attn",
    "prune_rate__decoder_heads", "prune_rate__heads_mid", "decoder_num_layers",
    "q_bits__global_w", "q_bits__global_a", "q_bits__backbone", "q_bits__encoder",
    "q_bits__decoder", "q_bits__heads", "q_bits__v2x_comm",
    "q_target", "q_granularity_w", "q_granularity_a",
]


def encode_cat(df, cols):
    for c in cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.Categorical(df[c]).codes
    return df


def train(X, y, target_name, model_out):
    print(f"\n=== {target_name} ({len(y)} rows) ===")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}] (span {y.max()-y.min():.4f})")
    if y.max() == y.min():
        print(f"  WARN: y has zero span, can't train")
        return None
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    params = {
        "objective": "regression", "metric": "mae", "verbosity": -1,
        "learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 1,
        "feature_pre_filter": False, "num_threads": 2,
    }
    for train_idx, test_idx in loo.split(X):
        booster = lgb.train(params, lgb.Dataset(X[train_idx], y[train_idx]), num_boost_round=80)
        preds[test_idx] = booster.predict(X[test_idx])
    rho, p = spearmanr(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"  Spearman = {rho:.3f} (p={p:.4g}), MAE = {mae:.4f}")

    # Final model on all data
    final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=80)
    final.save_model(str(model_out))
    print(f"  Saved: {model_out}")
    return {"spearman": float(rho), "p": float(p), "mae": float(mae),
            "n": int(len(y)), "y_min": float(y.min()), "y_max": float(y.max())}


def main():
    df = pd.read_csv(CSV)
    df = encode_cat(df, ["q_target", "q_granularity_w", "q_granularity_a"])
    print(f"Loaded {len(df)} configs from {CSV}")

    X = df[FEATURES].values.astype(np.float64)

    # Predictor 1: car_AP_4m
    y_ap = df["car_AP_4m"].values.astype(np.float64)
    metrics_ap = train(X, y_ap, "car_AP@4m", ROOT / "models/lgb_predictor_v4_tiny_acc.txt")

    # Predictor 2: TRT BEV latency
    y_lat = df["trt_bev_ms"].values.astype(np.float64)
    mask = ~np.isnan(y_lat)
    metrics_lat = train(X[mask], y_lat[mask], "TRT BEV latency (ms)",
                       ROOT / "models/lgb_predictor_v4_tiny_lat.txt")

    # Predictor 3: PyTorch e2e (more signal due to encoder ffn pruning visible)
    y_pyt = df["tiny_e2e_pytorch_ms"].values.astype(np.float64)
    metrics_pyt = train(X, y_pyt, "PyTorch e2e (ms)",
                       ROOT / "models/lgb_predictor_v4_tiny_pyt_e2e.txt")

    out_json = OUT_DIR / "phase4_lgb_v4_tiny_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"car_AP_4m": metrics_ap, "trt_bev_lat": metrics_lat,
                   "pytorch_e2e": metrics_pyt}, f, indent=2)
    print(f"\nSaved metrics: {out_json}")

    print("\n=== R101+DCN vs tiny baseline 对照 ===")
    print(f"  R101+DCN (v3): n=23, AMOTA span=0.094, Spearman 0.636")
    print(f"  tiny     (v4-tiny): n={len(df)}, car_AP_4m span={df.car_AP_4m.max()-df.car_AP_4m.min():.4f}, Spearman {metrics_ap['spearman']:.3f}")
    print(f"  R101+DCN latency span: 9.3 ms (baseline TRT 实测)")
    print(f"  tiny TRT BEV span:    {df.trt_bev_ms.max()-df.trt_bev_ms.min():.2f} ms (剪枝/量化对 BEV TRT 几乎无影响)")


if __name__ == "__main__":
    main()
