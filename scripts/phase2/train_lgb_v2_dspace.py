"""Plan2-Step4 — 重训 LGB v2 跨硬件 (4090 + Orin) + D 维度.

Input datasets:
    data/pyramid_random_bench.parquet      (100 anchor, 4090, D=default+8GB 单点)
    data/perstage_quant_bench.parquet      (18 anchor, 4090, per-stage Q)
    data/4090_dspace_bench.parquet         (48 anchor, 4090, D_tactic × D_workspace)
    data/orin_dspace_bench.parquet         (Orin, D_scheme × D_tactic × D_workspace)

Three LGB boosters trained:
    - lat_ms (regression, 只用 build_success=True 样本)
    - throughput_fps (留 placeholder, 等 multi-engine bench 数据)
    - build_success (binary classification, 用全部样本 含失败)

Features (12):
    stage0_planes, stage1_planes, stage2_planes,
    precision_fp16, precision_int8,
    d_scheme_A, d_scheme_B0, d_scheme_B1,        (one-hot, 4090 全是 A)
    d_tactic_default, d_tactic_other,            (one-hot 简化版)
    d_workspace_gb,                              (numeric)
    hardware_rtx4090, hardware_orin_agx,         (one-hot)

Output:
    models/lgb_v2_lat.txt
    models/lgb_v2_build_success.txt
    results/lgb_v2_metrics.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]


def load_all_data():
    """Load + harmonize all four bench datasets."""
    frames = []

    # 1. pyramid_random_bench (100 anchor, 4090, D=default+8GB)
    p = pd.read_parquet(ROOT / "data/pyramid_random_bench.parquet")
    p = p.assign(
        d_scheme="A_gpu",
        d_tactic="default",
        d_workspace_gb=8,
        hardware="rtx4090",
        build_success=True,
    )
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","engine_size_mb"]])

    # 2. perstage_quant_bench (18 anchor, 4090, per-stage Q 当作 fp16-int8 hybrid)
    p = pd.read_parquet(ROOT / "data/perstage_quant_bench.parquet")
    # For per-stage, classify by majority precision
    p["precision"] = p.apply(lambda r: "int8" if [r.stage0_prec,r.stage1_prec,r.stage2_prec].count("INT8") >= 2 else "fp16", axis=1)
    p = p.assign(
        d_scheme="A_gpu", d_tactic="default", d_workspace_gb=8,
        hardware="rtx4090", build_success=True,
    )
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","engine_size_mb"]])

    # 3. 4090_dspace_bench (48 anchor, 4090, D_tactic × D_workspace)
    p = pd.read_parquet(ROOT / "data/4090_dspace_bench.parquet")
    p = p.assign(d_scheme="A_gpu", hardware="rtx4090")
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","engine_size_mb"]])

    # 4. orin_dspace_bench (72 anchor, Orin, D_scheme × D_tactic × D_workspace)
    p = pd.read_csv(ROOT / "data/orin_dspace_bench.csv")
    p["build_success"] = p["build_success"].astype(bool)
    # Parse triplet_sig → stage planes
    parts = p["triplet_sig"].str.split("_", expand=True).astype(int)
    p["stage0_planes"] = parts[0]
    p["stage1_planes"] = parts[1]
    p["stage2_planes"] = parts[2]
    p["d_workspace_gb"] = p["d_workspace"].str.replace("GB","").astype(int)
    p = p.assign(hardware="orin_agx", engine_size_mb=None)
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","engine_size_mb"]])

    df = pd.concat(frames, ignore_index=True)
    return df


def featurize(df):
    """11 + 2 hw = 13 features."""
    X = np.column_stack([
        df["stage0_planes"].to_numpy(dtype=np.float32),
        df["stage1_planes"].to_numpy(dtype=np.float32),
        df["stage2_planes"].to_numpy(dtype=np.float32),
        (df["precision"] == "fp16").astype(np.float32),
        (df["precision"] == "int8").astype(np.float32),
        (df["d_scheme"] == "A_gpu").astype(np.float32),
        (df["d_scheme"] == "B_dla0").astype(np.float32),
        (df["d_scheme"] == "B_dla1").astype(np.float32),
        (df["d_tactic"] == "default").astype(np.float32),
        (df["d_tactic"] != "default").astype(np.float32),  # any non-default
        df["d_workspace_gb"].to_numpy(dtype=np.float32),
        (df["hardware"] == "rtx4090").astype(np.float32),
        (df["hardware"] == "orin_agx").astype(np.float32),
    ])
    names = ["stage0_planes","stage1_planes","stage2_planes",
             "prec_fp16","prec_int8",
             "scheme_A","scheme_B0","scheme_B1",
             "tactic_default","tactic_other",
             "workspace_gb","hw_4090","hw_orin"]
    return X, names


def train_lat_predictor(X_succ, y_lat, names):
    """Train lat regression on build_success=True samples."""
    n = len(y_lat)
    print(f"\n[Lat predictor] n_samples={n}")
    Xtr, Xv, ytr, yv = train_test_split(X_succ, y_lat, test_size=0.2, random_state=42)
    train_set = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    val_set = lgb.Dataset(Xv, yv, free_raw_data=False)
    params = {"objective": "regression", "metric": "mae",
              "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 2, "num_threads": 1,
              "verbose": -1, "seed": 42}
    model = lgb.train(params, train_set, num_boost_round=300,
                      valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(stopping_rounds=20)])
    pred = model.predict(Xv)
    mae = mean_absolute_error(yv, pred)
    spr = spearmanr(yv, pred).correlation
    print(f"  hold-out spearman={spr:.4f}  MAE={mae:.4f}ms  (rel {mae/np.mean(yv)*100:.1f}%)")
    # Final on full
    full = lgb.train(params, lgb.Dataset(X_succ, y_lat, free_raw_data=False),
                     num_boost_round=300)
    in_sample_pred = full.predict(X_succ)
    in_mae = mean_absolute_error(y_lat, in_sample_pred)
    in_spr = spearmanr(y_lat, in_sample_pred).correlation
    print(f"  in-sample spearman={in_spr:.4f}  MAE={in_mae:.4f}ms")
    return full, {"n_train": n,
                   "holdout_spearman": float(spr),
                   "holdout_mae_ms": float(mae),
                   "holdout_rel_mae_pct": float(mae/np.mean(yv)*100),
                   "in_sample_spearman": float(in_spr),
                   "in_sample_mae_ms": float(in_mae)}


def train_build_success_predictor(X_all, y_all, names):
    """Binary classifier on all samples (含 failure)."""
    print(f"\n[Build_success predictor] n_samples={len(y_all)}, positive_rate={y_all.mean():.2%}")
    if len(np.unique(y_all)) < 2:
        print("  WARN: only one class, skip training")
        return None, {"n_train": int(len(y_all)), "skipped": True}
    Xtr, Xv, ytr, yv = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    train_set = lgb.Dataset(Xtr, ytr, free_raw_data=False)
    val_set = lgb.Dataset(Xv, yv, free_raw_data=False)
    params = {"objective": "binary", "metric": "binary_logloss",
              "learning_rate": 0.05, "num_leaves": 15,
              "min_data_in_leaf": 3, "num_threads": 1,
              "verbose": -1, "seed": 42}
    model = lgb.train(params, train_set, num_boost_round=200,
                      valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(stopping_rounds=20)])
    pred_p = model.predict(Xv)
    pred = (pred_p > 0.5).astype(int)
    acc = accuracy_score(yv, pred)
    auc = roc_auc_score(yv, pred_p) if len(np.unique(yv)) > 1 else None
    print(f"  hold-out accuracy={acc:.4f}  AUC={auc}")
    full = lgb.train(params, lgb.Dataset(X_all, y_all, free_raw_data=False),
                     num_boost_round=200)
    return full, {"n_train": int(len(y_all)),
                  "positive_rate": float(y_all.mean()),
                  "holdout_accuracy": float(acc),
                  "holdout_auc": float(auc) if auc else None}


def main():
    print("="*70)
    print("LGB v2 跨硬件 D 空间预测器训练")
    print("="*70)

    df = load_all_data()
    print(f"\nTotal samples: {len(df)}")
    print("By hardware × build_success:")
    print(df.groupby(["hardware","build_success"]).size())

    # Drop rows with nan lat where build_success=True (data quality issue)
    df_succ = df[df.build_success & df.lat_p50_ms.notna()].copy()
    print(f"\n[Lat training pool]: {len(df_succ)} samples (build_success=True, lat not nan)")
    print("  hardware:", df_succ.hardware.value_counts().to_dict())
    print("  precision:", df_succ.precision.value_counts().to_dict())
    print("  d_scheme:", df_succ.d_scheme.value_counts().to_dict())
    print("  d_tactic:", df_succ.d_tactic.value_counts().to_dict())

    X_succ, names = featurize(df_succ)
    y_lat = df_succ["lat_p50_ms"].to_numpy(dtype=np.float32)

    lat_model, lat_metrics = train_lat_predictor(X_succ, y_lat, names)

    # build_success predictor
    X_all, _ = featurize(df)
    y_all = df["build_success"].astype(int).to_numpy()
    bs_model, bs_metrics = train_build_success_predictor(X_all, y_all, names)

    # Save
    MODELS = ROOT / "models"
    RESULTS = ROOT / "results"
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    lat_model.save_model(str(MODELS / "lgb_v2_lat.txt"))
    if bs_model is not None:
        bs_model.save_model(str(MODELS / "lgb_v2_build_success.txt"))

    metrics = {
        "feature_names": names,
        "lat_predictor": lat_metrics,
        "build_success_predictor": bs_metrics,
        "data_summary": {
            "total_samples": int(len(df)),
            "build_success_count": int(df.build_success.sum()),
            "lat_training_count": int(len(df_succ)),
            "by_hardware": df.hardware.value_counts().to_dict(),
            "by_hw_x_success": {
                f"{hw}_{succ}": int(c) for (hw, succ), c in
                df.groupby(["hardware","build_success"]).size().items()
            },
        },
    }
    with open(RESULTS / "lgb_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n✅ saved -> models/lgb_v2_lat.txt + lgb_v2_build_success.txt")
    print(f"✅ saved -> results/lgb_v2_metrics.json")

    # Feature importance
    print(f"\n[Lat predictor feature importance (split)]:")
    fi = sorted(zip(names, lat_model.feature_importance(importance_type="split")),
                key=lambda x: -x[1])
    for n, v in fi:
        print(f"  {n:20s}  {v}")


if __name__ == "__main__":
    main()
