"""LightGBM v4 — train AMOTA + latency predictors with D-space dimensions.

Two outputs:
  models/lgb_predictor_v4_amota.txt    — predicts amota_v4 (includes d_temporal_cache_int8 boost)
  models/lgb_predictor_v4_latency.txt  — predicts est_trt_latency_v4_ms

Trained on 368 rows (23 base configs × 16 D-space combinations).
Note: AMOTA only varies by d_temporal_cache (per config), so amota predictor only needs
that single D dim added to original 19 features.
"""
from __future__ import annotations

import json
import sys
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
V4_CSV = ROOT / "data/phase4/stage5_v4_dspace.csv"
OUT_DIR = ROOT / "results"

# Original 19 features (from v3 schema)
PRUNE_QUANT_FEATURES = [
    "prune_rate__backbone", "prune_rate__encoder_ffn", "prune_rate__encoder_attn",
    "prune_rate__encoder_heads", "prune_rate__decoder_ffn", "prune_rate__decoder_attn",
    "prune_rate__decoder_heads", "prune_rate__heads_mid", "decoder_num_layers",
    "q_bits__global_w", "q_bits__global_a", "q_bits__backbone", "q_bits__encoder",
    "q_bits__decoder", "q_bits__heads", "q_bits__v2x_comm",
    "q_target", "q_granularity_w", "q_granularity_a",
]


def encode_categorical(df, cols):
    for c in cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.Categorical(df[c]).codes
    return df


def train_predictor(X, y, target_name, feature_names, model_out):
    print(f"\n=== Train {target_name} ({len(y)} rows, {X.shape[1]} features) ===")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}] (span {y.max()-y.min():.4f})")

    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    params = {
        "objective": "regression", "metric": "mae", "verbosity": -1,
        "learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 1,
        "feature_pre_filter": False, "num_threads": 2,
    }

    # For 368 rows LOOCV is too slow; use 5-fold instead
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        train_set = lgb.Dataset(X[train_idx], y[train_idx])
        booster = lgb.train(params, train_set, num_boost_round=80)
        preds[test_idx] = booster.predict(X[test_idx])

    rho, p = spearmanr(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"  Spearman = {rho:.3f} (p={p:.4g}), MAE = {mae:.4f}")

    # Train final on all data
    final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=80)
    final.save_model(str(model_out))
    print(f"  Saved: {model_out}")

    return {"spearman": float(rho), "p_value": float(p), "mae": float(mae),
            "n_rows": int(len(y)), "y_min": float(y.min()), "y_max": float(y.max()),
            "y_span": float(y.max() - y.min()), "feature_count": int(X.shape[1])}


def main():
    df = pd.read_csv(V4_CSV)
    print(f"Loaded {len(df)} rows from {V4_CSV}")

    # ==== Predictor 1: AMOTA (includes d_temporal_cache_int8) ====
    feat_amota = PRUNE_QUANT_FEATURES + ["d_temporal_cache_int8"]
    feat_amota_df = df[feat_amota].copy()
    feat_amota_df = encode_categorical(feat_amota_df, ["q_target", "q_granularity_w", "q_granularity_a"])
    X_amota = feat_amota_df.values.astype(np.float64)
    y_amota = df["amota_v4"].values.astype(np.float64)

    mask = ~np.isnan(y_amota)
    metrics_amota = train_predictor(
        X_amota[mask], y_amota[mask], "AMOTA v4 (with cache_int8)",
        feat_amota, ROOT / "models/lgb_predictor_v4_amota.txt",
    )

    # ==== Predictor 2: Latency ====
    feat_lat = PRUNE_QUANT_FEATURES + ["d_runtime", "d_pipelined_get_bevs"]
    feat_lat_df = df[feat_lat].copy()
    feat_lat_df = encode_categorical(feat_lat_df, ["q_target", "q_granularity_w",
                                                     "q_granularity_a", "d_runtime"])
    X_lat = feat_lat_df.values.astype(np.float64)
    y_lat = df["est_trt_latency_v4_ms"].values.astype(np.float64)

    metrics_lat = train_predictor(
        X_lat, y_lat, "Latency v4 (TRT + D-space)",
        feat_lat, ROOT / "models/lgb_predictor_v4_latency.txt",
    )

    # Save metrics
    out_json = OUT_DIR / "phase4_lgb_v4_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"amota_predictor": metrics_amota, "latency_predictor": metrics_lat}, f, indent=2)
    print(f"\nSaved metrics: {out_json}")

    # Summary comparison
    print("\n=== v3 vs v4 Comparison ===")
    print(f"  v3 latency span (no D-space):  0.0934 → Spearman 0.971")
    print(f"  v4 latency span (with D):      {metrics_lat['y_span']:.1f} ms → Spearman {metrics_lat['spearman']:.3f}")
    print(f"  v3 AMOTA span:                 0.0934 → Spearman 0.636")
    print(f"  v4 AMOTA span (with cache):    {metrics_amota['y_span']:.4f} → Spearman {metrics_amota['spearman']:.3f}")


if __name__ == "__main__":
    main()
