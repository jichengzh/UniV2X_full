"""LightGBM v3 training on Plan B (R101+DCN baseline) AMOTA data.

Targets `amota` (clean signal, no merge pipeline noise).
Spearman target: ≥ 0.7 (vs v2's -0.36 on tiny baseline).
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
MODEL_OUT = ROOT / "models/lgb_predictor_v3.txt"


def main() -> None:
    df = pd.read_csv(V3_CSV)
    print(f"Loaded {len(df)} rows from {V3_CSV}")
    print(f"AMOTA range: [{df['amota'].min():.4f}, {df['amota'].max():.4f}]")

    target = "amota"
    drop_cols = ["config_id", "source", "amotp", "recall", "mota", "tp", "fp", "fn",
                 "gt", "mAP", "NDS", "car_ap_4m", target]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode categorical features (q_target, q_granularity_*)
    cat_cols = [c for c in feature_df.columns if feature_df[c].dtype == "object"]
    for c in cat_cols:
        feature_df[c] = pd.Categorical(feature_df[c]).codes
    print(f"Encoded {len(cat_cols)} categorical: {cat_cols}")
    print(f"Total features: {len(feature_df.columns)}: {list(feature_df.columns)}")

    X = feature_df.values.astype(np.float64)
    y = df[target].values.astype(np.float64)

    # Drop rows with NaN AMOTA (failed configs)
    mask = ~np.isnan(y)
    if not mask.all():
        print(f"Dropping {(~mask).sum()} rows with NaN AMOTA")
        X = X[mask]
        y = y[mask]
        df = df[mask].reset_index(drop=True)

    if len(y) < 5:
        print(f"ERROR: only {len(y)} rows, need ≥ 5 for LOOCV")
        return

    # LOOCV
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
        "num_threads": 2,  # don't thrash CPU when GPU eval is running concurrently
    }

    for train_idx, test_idx in loo.split(X):
        train_set = lgb.Dataset(X[train_idx], y[train_idx])
        booster = lgb.train(params, train_set, num_boost_round=80)
        preds[test_idx] = booster.predict(X[test_idx])

    rho, p = spearmanr(y, preds)
    mae = mean_absolute_error(y, preds)

    print(f"\n=== LightGBM v3 LOOCV ({len(y)} rows) ===")
    print(f"  Spearman rho = {rho:.3f} (p={p:.3f})")
    print(f"  MAE          = {mae:.4f}")
    print(f"  y range      = [{y.min():.4f}, {y.max():.4f}] (span {y.max()-y.min():.4f})")
    print()
    print("Per-config OOF:")
    for cfg, true_v, pred_v in zip(df["config_id"], y, preds):
        print(f"  {cfg:12s}  true={true_v:.4f}  pred={pred_v:.4f}  err={abs(true_v-pred_v):.4f}")

    # Final model on all rows
    final_train = lgb.Dataset(X, y)
    final = lgb.train(params, final_train, num_boost_round=80)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODEL_OUT))
    print(f"\nSaved final model: {MODEL_OUT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "phase4_stage5_lgb_v3_metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_rows": int(len(y)),
            "spearman": float(rho),
            "spearman_p": float(p),
            "mae": float(mae),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_span": float(y.max() - y.min()),
            "target": target,
            "feature_count": int(X.shape[1]),
            "configs": df["config_id"].tolist(),
        }, f, indent=2)
    print(f"Saved metrics: {out_json}")

    # Compare to v2
    v2_path = OUT_DIR / "phase4_stage5_lgb_v2_metrics.json"
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text())
        print()
        print("=== v2 vs v3 ===")
        print(f"  v2 (tiny):      n={v2.get('n_rows')}, span={v2.get('y_max',0)-v2.get('y_min',0):.4f}, Spearman={v2.get('spearman',0):.3f}")
        print(f"  v3 (R101+DCN):  n={len(y)},       span={y.max()-y.min():.4f}, Spearman={rho:.3f}")
        print(f"  Improvement: span ×{(y.max()-y.min())/(v2.get('y_max',1)-v2.get('y_min',0)):.1f}, Spearman {rho-v2.get('spearman',0):+.3f}")


if __name__ == "__main__":
    main()
