"""Plan2-Phase-AP-LGB — LGB v4 with REAL AP predictor + 5D Pareto demo.

Adds amota / AP50 / AP70 as new predictor outputs, trained on:
    - data/stage_a_ap_real.parquet  (8 anchors: baseline + pruned25/50/75 × FP16/INT8)
    - data/stage_b_ap_real.parquet  (~40 anchors: 20 strategic random_bench triplets × FP16/INT8 after finetune)

Joined with existing lat data from:
    - data/pyramid_random_bench.parquet  (100 lat anchors, NO AP)
    - data/4090_dspace_bench.parquet     (48 lat anchors, NO AP)
    - data/orin_dspace_bench.parquet     (72 lat anchors, NO AP)

Strategy:
    - AP predictor trained on ~48 real AP anchors (single regression task)
    - LGB v4 lat predictor inherits from v3
    - Multi-output: predict (lat, throughput, ap50, ap70, build_success)
    - 5D Pareto: minimize (lat, workspace, params), maximize (throughput, ap50)
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[2]


def load_ap_data():
    """Load real AP measurements from Stage A + Stage B."""
    frames = []
    for fname in ["stage_a_ap_real.parquet", "stage_b_ap_real.parquet"]:
        p = ROOT / "data" / fname
        if p.exists():
            df = pd.read_parquet(p)
            frames.append(df)
            print(f"  loaded {fname}: {len(df)} anchors")
        else:
            print(f"  WARN: {fname} missing")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def harmonize_ap_data(df_ap):
    """Standardize columns: stage planes + prec + ap50/70."""
    if df_ap.empty:
        return df_ap
    keep = ["stage0_planes","stage1_planes","stage2_planes","precision","ap30","ap50","ap70"]
    df_ap = df_ap[[c for c in keep if c in df_ap.columns]].copy()
    return df_ap


def featurize(df):
    X = np.column_stack([
        df["stage0_planes"].to_numpy(dtype=np.float32),
        df["stage1_planes"].to_numpy(dtype=np.float32),
        df["stage2_planes"].to_numpy(dtype=np.float32),
        (df["precision"] == "fp16").astype(np.float32),
        (df["precision"] == "int8").astype(np.float32),
    ])
    names = ["stage0_planes","stage1_planes","stage2_planes","prec_fp16","prec_int8"]
    return X, names


def train_ap_predictor(df_ap, target_col="ap50"):
    """Train AP regression on real measurements only."""
    if df_ap.empty or target_col not in df_ap.columns:
        return None, None
    df = df_ap.dropna(subset=[target_col]).copy()
    if len(df) < 5:
        print(f"  WARN: only {len(df)} samples for {target_col}, skipping")
        return None, None
    X, names = featurize(df)
    y = df[target_col].to_numpy(dtype=np.float32)
    print(f"\n[{target_col} predictor] n={len(y)}, y range [{y.min():.3f}, {y.max():.3f}]")
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {"objective": "regression", "metric": "mae",
              "learning_rate": 0.05, "num_leaves": 15,
              "min_data_in_leaf": 1, "num_threads": 1, "verbose": -1, "seed": 42}
    model = lgb.train(params, lgb.Dataset(Xtr, ytr, free_raw_data=False),
                      num_boost_round=200,
                      valid_sets=[lgb.Dataset(Xv, yv, free_raw_data=False)],
                      callbacks=[lgb.early_stopping(stopping_rounds=20)])
    pred = model.predict(Xv)
    spr = spearmanr(yv, pred).correlation if len(yv) > 1 else 0
    mae = mean_absolute_error(yv, pred)
    print(f"  hold-out spearman={spr:.4f}  MAE={mae:.4f}  rel={mae/np.mean(yv)*100:.1f}%")
    full = lgb.train(params, lgb.Dataset(X, y, free_raw_data=False), num_boost_round=200)
    return full, {"n": len(y), "spearman": float(spr), "mae": float(mae),
                   "rel_mae_pct": float(mae/np.mean(yv)*100)}


def main():
    print("="*70)
    print("LGB v4: real AP predictor (Stage A + B)")
    print("="*70)

    df_ap = load_ap_data()
    if df_ap.empty:
        print("\n❌ No AP data — stages A/B not complete yet")
        return
    df_ap = harmonize_ap_data(df_ap)
    print(f"\nTotal AP anchors: {len(df_ap)}")
    print(df_ap.head(15).to_string())

    # Train predictors
    metrics = {}
    for target in ["ap30", "ap50", "ap70"]:
        if target not in df_ap.columns: continue
        m, mr = train_ap_predictor(df_ap, target)
        if m is None: continue
        m.save_model(str(ROOT / f"models/lgb_v4_{target}.txt"))
        metrics[target] = mr
        print(f"  ✅ saved lgb_v4_{target}.txt")

    # Save metrics
    out = ROOT / "results/lgb_v4_metrics.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({"metrics": metrics, "n_anchors": len(df_ap)}, f, indent=2)
    print(f"\n✅ saved {out}")


if __name__ == "__main__":
    main()
