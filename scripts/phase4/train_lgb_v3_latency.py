"""LightGBM v3 with latency target — predict e2e_ms given config.

Trains a SECOND predictor (alongside AMOTA predictor) so NSGA-II can do
multi-objective accuracy×latency search.
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
V3_CSV = ROOT / "data/phase4/stage5_baseline_v3.csv"
OUT_DIR = ROOT / "results"
MODEL_OUT = ROOT / "models/lgb_predictor_v3_latency.txt"


def main() -> None:
    df = pd.read_csv(V3_CSV)
    print(f"Loaded {len(df)} rows from {V3_CSV}")

    target = "est_trt_latency_ms"
    if target not in df.columns:
        print(f"ERROR: {target} column missing — run compute_est_trt_latency.py first")
        return

    print(f"{target} range: [{df[target].min():.1f}, {df[target].max():.1f}] ms")

    drop_cols = ["config_id", "source", "amota", "amotp", "recall", "mota", "tp", "fp", "fn",
                 "gt", "mAP", "NDS", "car_ap_4m", target,
                 "lat_e2e_ms", "lat_e2e_std", "lat_backbone_ms", "lat_neck_ms",
                 "lat_bev_encoder_ms", "lat_seg_head_ms", "lat_track_head_decoder_ms",
                 "params_after_M", "params_reduction"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    for c in cat_cols:
        feature_df[c] = pd.Categorical(feature_df[c]).codes
    print(f"Features: {len(feature_df.columns)}")

    X = feature_df.values.astype(np.float64)
    y = df[target].values.astype(np.float64)

    mask = ~np.isnan(y)
    if not mask.all():
        print(f"Dropping {(~mask).sum()} rows with NaN {target}")
        X, y, df = X[mask], y[mask], df[mask].reset_index(drop=True)

    if len(y) < 5:
        print(f"ERROR: only {len(y)} rows")
        return

    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_data_in_leaf": 1,
        "feature_pre_filter": False,
        "num_threads": 2,
    }
    for train_idx, test_idx in loo.split(X):
        train_set = lgb.Dataset(X[train_idx], y[train_idx])
        booster = lgb.train(params, train_set, num_boost_round=80)
        preds[test_idx] = booster.predict(X[test_idx])

    rho, p = spearmanr(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"\n=== LightGBM v3 LATENCY LOOCV ({len(y)} rows) ===")
    print(f"  Spearman rho = {rho:.3f} (p={p:.3f})")
    print(f"  MAE          = {mae:.2f} ms")
    print(f"  y range      = [{y.min():.1f}, {y.max():.1f}] ms (span {y.max()-y.min():.1f} ms)")

    print("\nPer-config OOF:")
    for cfg, true_v, pred_v in zip(df["config_id"], y, preds):
        print(f"  {cfg:12s}  true={true_v:7.1f}  pred={pred_v:7.1f}  err={abs(true_v-pred_v):6.1f}")

    final_train = lgb.Dataset(X, y)
    final = lgb.train(params, final_train, num_boost_round=80)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODEL_OUT))
    print(f"\nSaved final latency model: {MODEL_OUT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "phase4_stage5_lgb_v3_latency_metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_rows": int(len(y)),
            "spearman": float(rho),
            "spearman_p": float(p),
            "mae_ms": float(mae),
            "y_min_ms": float(y.min()),
            "y_max_ms": float(y.max()),
            "y_span_ms": float(y.max() - y.min()),
            "target": target,
            "feature_count": int(X.shape[1]),
        }, f, indent=2)
    print(f"Saved metrics: {out_json}")


if __name__ == "__main__":
    main()
