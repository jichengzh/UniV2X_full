"""P0.5 — Train Pyramid_DAIR_m1 single-model LGB lat predictor.

Input:  data/pyramid_random_bench.parquet (~100 rows from p0_pyramid_random_bench.py)
        + 12 hand-pick anchor data from results/m4_9_framework_pareto_v2.csv (optional merge)

Features (5):
    stage0_planes     numeric (16-64)
    stage1_planes     numeric (16-128)
    stage2_planes     numeric (16-256)
    precision_fp16    {0, 1}
    precision_int8    {0, 1}

Target: lat_p50_ms

Train: LightGBM regression, leave-one-out CV (since only ~100 samples) +
       5-fold CV for MAE/spearman.

Output:
    models/lgb_pyramid_lat.txt
    results/lgb_pyramid_metrics.json
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
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]


def featurize(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix (N, 5) from per-row stage planes + precision."""
    names = ["stage0_planes", "stage1_planes", "stage2_planes",
             "precision_fp16", "precision_int8"]
    X = np.column_stack([
        df["stage0_planes"].to_numpy(dtype=np.float32),
        df["stage1_planes"].to_numpy(dtype=np.float32),
        df["stage2_planes"].to_numpy(dtype=np.float32),
        (df["precision"] == "fp16").astype(np.float32).to_numpy(),
        (df["precision"] == "int8").astype(np.float32).to_numpy(),
    ])
    return X, names


def train_one(X, y, names, seed: int = 42) -> tuple[lgb.Booster, dict]:
    """Train LightGBM regression with random 80/20 hold-out (KFold causes
    internal state corruption with this small dataset, so use simple split)."""
    n = len(y)
    print(f"  training samples: {n}, features: {len(names)} ({names})")

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,           # 102 samples / 31 leaves ≈ 3.3 samples/leaf
        "min_data_in_leaf": 2,
        "num_threads": 1,           # avoid OMP deadlock with PyTorch/CUDA in env
        "verbose": -1,
        "seed": seed,
    }

    # 80/20 hold-out
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(0.8 * n)
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    train_set = lgb.Dataset(X[tr_idx], y[tr_idx], free_raw_data=False)
    m = lgb.train(params, train_set, num_boost_round=200)
    val_pred = m.predict(X[val_idx])
    val_mae = mean_absolute_error(y[val_idx], val_pred)
    val_spr = spearmanr(y[val_idx], val_pred).correlation
    val_rel_mae = val_mae / np.mean(y[val_idx]) * 100
    print(f"  hold-out (n={len(val_idx)}): spearman={val_spr:.4f}  MAE={val_mae:.4f} ms  ({val_rel_mae:.1f}% relative)")

    # Final train on all data
    final_set = lgb.Dataset(X, y, free_raw_data=False)
    final = lgb.train(params, final_set, num_boost_round=200)

    in_sample_pred = final.predict(X)
    in_sample_mae = mean_absolute_error(y, in_sample_pred)
    in_sample_spr = spearmanr(y, in_sample_pred).correlation
    print(f"  in-sample (n={n}): spearman={in_sample_spr:.4f}  MAE={in_sample_mae:.4f} ms")

    cv_mae = val_mae
    cv_spr = val_spr
    cv_rel_mae = val_rel_mae

    return final, {
        "n_train": n,
        "n_features": len(names),
        "feature_names": names,
        "cv_5fold_spearman": cv_spr,
        "cv_5fold_mae_ms": cv_mae,
        "cv_5fold_rel_mae_pct": cv_rel_mae,
        "in_sample_spearman": in_sample_spr,
        "in_sample_mae_ms": in_sample_mae,
    }


def main():
    print("=" * 60)
    print("P0.5 — Train Pyramid_DAIR_m1 LGB lat predictor")
    print("=" * 60)

    bench_path = ROOT / "data/pyramid_random_bench.parquet"
    if not bench_path.exists():
        raise SystemExit(f"❌ {bench_path} not found — run p0_pyramid_random_bench.py first")

    df_random = pd.read_parquet(bench_path)
    print(f"\n  loaded {len(df_random)} rows from {bench_path.name}")

    # Augment with M4.9 v2 hand-pick anchors that random search missed
    # (uniform prune 50% / 75% finetuned variants). These add valuable training
    # signal at canonical Pareto points without costing extra bench time.
    extra_anchors = [
        # (stage0, stage1, stage2, precision, lat_p50_ms) — from m4_9_framework_pareto_v2.csv
        # A3 / A4 prune50 FT (uniform 50% on baseline 64/128/256 → 32/64/128)
        (32, 64, 128, "fp16", 1.009),
        (32, 64, 128, "int8", 0.776),
        # A11 / A10 prune75 FT — these are within random space already, skip
    ]
    extra_rows = []
    for s0, s1, s2, prec, lat in extra_anchors:
        extra_rows.append({
            "triplet_sig": f"{s0:03d}_{s1:03d}_{s2:03d}",
            "stage0_planes": s0, "stage1_planes": s1, "stage2_planes": s2,
            "stage0_rate": (64-s0)/64, "stage1_rate": (128-s1)/128, "stage2_rate": (256-s2)/256,
            "precision": prec, "lat_p50_ms": lat,
            "lat_p99_ms": lat, "lat_mean_ms": lat,
            "engine_size_mb": 0.0, "build_secs": 0.0,
        })
    df_extra = pd.DataFrame(extra_rows)
    print(f"  augmenting with {len(df_extra)} hand-pick anchors (A3/A4 prune50)")
    df = pd.concat([df_random, df_extra], ignore_index=True)
    print(f"  total training rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    print(f"  precisions: {df['precision'].value_counts().to_dict()}")
    print(f"  stage0_planes: {sorted(df['stage0_planes'].unique().tolist())}")
    print(f"  stage1_planes: {sorted(df['stage1_planes'].unique().tolist())}")
    print(f"  stage2_planes: {sorted(df['stage2_planes'].unique().tolist())}")
    print(f"  lat range: {df['lat_p50_ms'].min():.3f} – {df['lat_p50_ms'].max():.3f} ms")

    # Drop rows with missing lat (failed builds)
    df = df.dropna(subset=["lat_p50_ms"]).reset_index(drop=True)
    if len(df) < 20:
        print(f"⚠️  only {len(df)} valid rows — too few for LGB")
        return

    X, names = featurize(df)
    y = df["lat_p50_ms"].to_numpy(dtype=np.float32)

    print(f"\n[Train LGB]")
    model, metrics = train_one(X, y, names, seed=42)

    # Save
    MODELS_DIR = ROOT / "models"
    RESULTS_DIR = ROOT / "results"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODELS_DIR / "lgb_pyramid_lat.txt"))
    with open(RESULTS_DIR / "lgb_pyramid_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  ✅ saved -> models/lgb_pyramid_lat.txt")
    print(f"  ✅ saved -> results/lgb_pyramid_metrics.json")

    # Feature importance
    print(f"\n  feature importance (split):")
    fi = sorted(zip(names, model.feature_importance(importance_type="split")),
                key=lambda x: -x[1])
    for n, v in fi:
        print(f"    {n:25s}  {v}")

    # Hold-out 12 anchor sanity (sample top 5 / bottom 5 by lat)
    print(f"\n  hold-out spot check:")
    df_sorted = df.sort_values("lat_p50_ms")
    for _, r in df_sorted.head(3).iterrows():
        x = np.array([[r["stage0_planes"], r["stage1_planes"], r["stage2_planes"],
                       int(r["precision"] == "fp16"), int(r["precision"] == "int8")]],
                      dtype=np.float32)
        pred = model.predict(x)[0]
        err = pred - r["lat_p50_ms"]
        print(f"    ({int(r['stage0_planes'])},{int(r['stage1_planes'])},{int(r['stage2_planes'])}) "
              f"{r['precision']:5s}  measured={r['lat_p50_ms']:.3f}  pred={pred:.3f}  err={err:+.3f} ms")
    for _, r in df_sorted.tail(3).iterrows():
        x = np.array([[r["stage0_planes"], r["stage1_planes"], r["stage2_planes"],
                       int(r["precision"] == "fp16"), int(r["precision"] == "int8")]],
                      dtype=np.float32)
        pred = model.predict(x)[0]
        err = pred - r["lat_p50_ms"]
        print(f"    ({int(r['stage0_planes'])},{int(r['stage1_planes'])},{int(r['stage2_planes'])}) "
              f"{r['precision']:5s}  measured={r['lat_p50_ms']:.3f}  pred={pred:.3f}  err={err:+.3f} ms")


if __name__ == "__main__":
    main()
