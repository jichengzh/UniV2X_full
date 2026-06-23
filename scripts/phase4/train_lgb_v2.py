"""LightGBM v2 smoke training on Stage 5.1 data (7 rows so far).

Combines the new Stage 5.1 data with optional baseline_unified rows for an
LOOCV evaluation. Targets `real_f1_overall` (Stage 5 cleanest signal).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.feature_encoder import CATEGORICAL_COLS, encode_baseline_df  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

V2_CSV = ROOT / "data/phase4/stage5_baseline_v2.csv"
OUT_DIR = ROOT / "results"
MODEL_OUT = ROOT / "models/lgb_predictor_v2.txt"


def main() -> None:
    df = pd.read_csv(V2_CSV)
    print(f"Loaded {len(df)} rows from {V2_CSV}")

    # Use real_f1_overall as the target (clean signal, no AMOTA truncation).
    target = "real_f1_overall"
    feature_df = df.drop(columns=[c for c in df.columns if c.startswith("real_") or c in {
        "config_id", "source", "id", "goal", "n_pred_overall", "tp_overall", "n_gt_overall",
        "physical_violation", "empirical_violations", "soft_violations",
    }], errors="ignore")
    encoded = encode_baseline_df(feature_df)
    print(f"Encoded {len(encoded.columns)} features")

    # Convert categorical dtypes to integer codes (LightGBM-friendly).
    cat_cols = [c for c in encoded.columns if str(encoded[c].dtype) == "category"]
    for c in cat_cols:
        encoded[c] = encoded[c].cat.codes
    print(f"Encoded {len(cat_cols)} categorical cols as int codes")

    X = encoded.values.astype(np.float64)
    y = df[target].values.astype(np.float64)

    # LOOCV (leave-one-out, ok for tiny n)
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
    }

    for train_idx, test_idx in loo.split(X):
        train_set = lgb.Dataset(X[train_idx], y[train_idx])
        booster = lgb.train(params, train_set, num_boost_round=80)
        preds[test_idx] = booster.predict(X[test_idx])

    rho, p = spearmanr(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"\n=== LightGBM v2 LOOCV ({len(y)} rows) ===")
    print(f"  Spearman rho = {rho:.3f} (p={p:.3f})")
    print(f"  MAE          = {mae:.4f}")
    print(f"  y range      = [{y.min():.4f}, {y.max():.4f}] (span {y.max()-y.min():.4f})")
    print()
    print("Per-config OOF:")
    for cfg, true_v, pred_v in zip(df["config_id"], y, preds):
        print(f"  {cfg:10s}  true={true_v:.4f}  pred={pred_v:.4f}  err={abs(true_v-pred_v):.4f}")

    # Train final on ALL rows.
    final_train = lgb.Dataset(X, y)
    final = lgb.train(params, final_train, num_boost_round=80)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODEL_OUT))
    print(f"\nSaved final model: {MODEL_OUT}")

    # Save metrics
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "phase4_stage5_lgb_v2_metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "n_rows": int(len(y)),
            "spearman": float(rho),
            "spearman_p": float(p),
            "mae": float(mae),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "target": target,
            "feature_count": int(encoded.shape[1]),
        }, f, indent=2)
    print(f"Saved metrics: {out_json}")


if __name__ == "__main__":
    main()
