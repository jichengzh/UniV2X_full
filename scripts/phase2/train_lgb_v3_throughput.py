"""Plan2-Phase3 — LGB v3: 加 throughput predictor (3rd booster).

新增数据源:
    results/orin_multi_engine_result.json   (prune75 scheme B 实测 throughput)
    results/orin_multi_engine_batch.json    (5 triplet × scheme A+B 实测 throughput)
    results/orin_3ip_prune75.json           (scheme C 3-IP 实测)
    results/orin_3ip_prune50.json           (scheme C 3-IP 实测)

策略:
    - 单 IP throughput = 1/lat_mean (从所有 single-engine bench 派生)
    - 双 IP (scheme B) throughput = 实测 multi-engine
    - 三 IP (scheme C) throughput = 实测 3-engine

LGB v3 booster:
    - lat_p50_ms (regression) — 继承 v2
    - throughput_fps (regression) — 新增
    - build_success (binary) — 继承 v2

关键论点:
    throughput 跟 lat 不是 1/lat 关系! 双/三 IP 流水线让 throughput > 1/lat。
    LGB v3 必须学到 (config, scheme) → throughput 的非线性映射。
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[2]


def load_lat_data():
    """Re-use lgb_v2 data sources for lat (same as train_lgb_v2_dspace)."""
    frames = []

    # 1. pyramid_random_bench (4090, B × Q, D 单点)
    p = pd.read_parquet(ROOT / "data/pyramid_random_bench.parquet")
    p = p.assign(d_scheme="A_gpu", d_tactic="default", d_workspace_gb=8,
                  hardware="rtx4090", build_success=True)
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","lat_mean_ms",
                     "engine_size_mb"]])

    # 2. perstage (per-stage Q)
    p = pd.read_parquet(ROOT / "data/perstage_quant_bench.parquet")
    p["precision"] = p.apply(lambda r: "int8" if [r.stage0_prec,r.stage1_prec,r.stage2_prec].count("INT8") >= 2 else "fp16", axis=1)
    p = p.assign(d_scheme="A_gpu", d_tactic="default", d_workspace_gb=8,
                  hardware="rtx4090", build_success=True)
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","lat_mean_ms",
                     "engine_size_mb"]])

    # 3. 4090 D-space (48 anchor)
    p = pd.read_parquet(ROOT / "data/4090_dspace_bench.parquet")
    p = p.assign(d_scheme="A_gpu", hardware="rtx4090")
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","lat_mean_ms",
                     "engine_size_mb"]])

    # 4. Orin D-space (72 anchor)
    p = pd.read_csv(ROOT / "data/orin_dspace_bench.csv")
    p["build_success"] = p["build_success"].astype(bool)
    parts = p["triplet_sig"].str.split("_", expand=True).astype(int)
    p["stage0_planes"] = parts[0]; p["stage1_planes"] = parts[1]; p["stage2_planes"] = parts[2]
    p["d_workspace_gb"] = p["d_workspace"].str.replace("GB","").astype(int)
    p = p.assign(hardware="orin_agx", engine_size_mb=None)
    frames.append(p[["stage0_planes","stage1_planes","stage2_planes",
                     "precision","d_scheme","d_tactic","d_workspace_gb",
                     "hardware","build_success","lat_p50_ms","lat_mean_ms",
                     "engine_size_mb"]])

    df = pd.concat(frames, ignore_index=True)
    # Derive single-IP throughput from lat_mean_ms
    df["throughput_fps"] = df.apply(
        lambda r: 1000.0 / r.lat_mean_ms if pd.notna(r.lat_mean_ms) else None,
        axis=1
    )
    return df


def load_multi_engine_throughput():
    """Load multi-engine実測 throughput (scheme B / C)."""
    rows = []

    # Single anchor: prune75 scheme B (from results/orin_multi_engine_result.json)
    p = ROOT / "results/orin_multi_engine_result.json"
    if p.exists():
        d = json.loads(p.read_text())
        s, sig = (16, 32, 64), "016_032_064"
        rows.append({
            "stage0_planes": s[0], "stage1_planes": s[1], "stage2_planes": s[2],
            "precision": "fp16", "d_scheme": "B_dla0",
            "d_tactic": "default", "d_workspace_gb": 2,  # Orin 8.5 prefer 2GB
            "hardware": "orin_agx", "build_success": True,
            "lat_p50_ms": d["scheme_B_pipeline"]["single_frame_p50_ms"],
            "lat_mean_ms": 1000.0 / d["scheme_B_pipeline"]["throughput_fps"],
            "throughput_fps": d["scheme_B_pipeline"]["throughput_fps"],
            "engine_size_mb": None,
        })

    # Batch: 5 triplet × scheme A/B (from orin_multi_engine_batch.json)
    p = ROOT / "results/orin_multi_engine_batch.json"
    if p.exists():
        d = json.loads(p.read_text())
        for sig, r in d.items():
            if r.get("skipped") or r.get("error"):
                continue
            parts = sig.split("_"); s = (int(parts[0]), int(parts[1]), int(parts[2]))
            # Scheme A (integrated GPU)
            rows.append({
                "stage0_planes": s[0], "stage1_planes": s[1], "stage2_planes": s[2],
                "precision": "fp16", "d_scheme": "A_gpu",
                "d_tactic": "default", "d_workspace_gb": 2,
                "hardware": "orin_agx", "build_success": True,
                "lat_p50_ms": r["scheme_A_full"]["p50_ms"],
                "lat_mean_ms": r["scheme_A_full"]["mean_ms"],
                "throughput_fps": r["scheme_A_full"]["throughput_fps"],
                "engine_size_mb": None,
            })
            # Scheme B (DLA0 backbone + GPU collab) — 实测 pipeline
            rows.append({
                "stage0_planes": s[0], "stage1_planes": s[1], "stage2_planes": s[2],
                "precision": "fp16", "d_scheme": "B_dla0",
                "d_tactic": "default", "d_workspace_gb": 2,
                "hardware": "orin_agx", "build_success": True,
                "lat_p50_ms": r["scheme_B_pipeline"]["single_frame_p50_ms"],
                "lat_mean_ms": 1000.0 / r["scheme_B_pipeline"]["throughput_fps"],
                "throughput_fps": r["scheme_B_pipeline"]["throughput_fps"],
                "engine_size_mb": None,
            })

    # Scheme C: 3-IP pipeline (prune75 + prune50)
    for sig_str, planes in [("016_032_064", (16,32,64)), ("032_064_136", (32,64,136))]:
        p = ROOT / f"results/orin_3ip_{['prune50','prune75'][sig_str=='016_032_064']}.json"
        if p.exists():
            d = json.loads(p.read_text())
            rows.append({
                "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                "precision": "fp16", "d_scheme": "C_3ip",
                "d_tactic": "default", "d_workspace_gb": 2,
                "hardware": "orin_agx", "build_success": True,
                "lat_p50_ms": d["scheme_C_pipeline"]["single_frame_p50_ms"],
                "lat_mean_ms": 1000.0 / d["scheme_C_pipeline"]["throughput_fps"],
                "throughput_fps": d["scheme_C_pipeline"]["throughput_fps"],
                "engine_size_mb": None,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def featurize(df):
    X = np.column_stack([
        df["stage0_planes"].to_numpy(dtype=np.float32),
        df["stage1_planes"].to_numpy(dtype=np.float32),
        df["stage2_planes"].to_numpy(dtype=np.float32),
        (df["precision"] == "fp16").astype(np.float32),
        (df["precision"] == "int8").astype(np.float32),
        (df["d_scheme"] == "A_gpu").astype(np.float32),
        (df["d_scheme"] == "B_dla0").astype(np.float32),
        (df["d_scheme"] == "B_dla1").astype(np.float32),
        (df["d_scheme"] == "C_3ip").astype(np.float32),  # NEW: 3-IP scheme
        (df["d_tactic"] == "default").astype(np.float32),
        (df["d_tactic"] != "default").astype(np.float32),
        df["d_workspace_gb"].to_numpy(dtype=np.float32),
        (df["hardware"] == "rtx4090").astype(np.float32),
        (df["hardware"] == "orin_agx").astype(np.float32),
    ])
    names = ["stage0_planes","stage1_planes","stage2_planes",
             "prec_fp16","prec_int8",
             "scheme_A","scheme_B0","scheme_B1","scheme_C_3ip",
             "tactic_default","tactic_other",
             "workspace_gb","hw_4090","hw_orin"]
    return X, names


def train_regression(X, y, target_name, names, params_override=None):
    """Generic regression trainer."""
    print(f"\n[{target_name} predictor] n={len(y)}")
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {"objective": "regression", "metric": "mae",
              "learning_rate": 0.05, "num_leaves": 31,
              "min_data_in_leaf": 2, "num_threads": 1,
              "verbose": -1, "seed": 42}
    if params_override: params.update(params_override)
    model = lgb.train(params, lgb.Dataset(Xtr, ytr, free_raw_data=False),
                      num_boost_round=300,
                      valid_sets=[lgb.Dataset(Xv, yv, free_raw_data=False)],
                      callbacks=[lgb.early_stopping(stopping_rounds=20)])
    pred = model.predict(Xv)
    mae = mean_absolute_error(yv, pred)
    spr = spearmanr(yv, pred).correlation
    print(f"  hold-out spearman={spr:.4f}  MAE={mae:.4f}  rel={mae/np.mean(yv)*100:.1f}%")
    full = lgb.train(params, lgb.Dataset(X, y, free_raw_data=False), num_boost_round=300)
    return full, {"n_train": len(y),
                   "holdout_spearman": float(spr),
                   "holdout_mae": float(mae),
                   "holdout_rel_mae_pct": float(mae/np.mean(yv)*100)}


def main():
    print("="*70)
    print("LGB v3: 加 throughput predictor + 三 IP scheme C 数据")
    print("="*70)

    df_lat = load_lat_data()
    df_thr_multi = load_multi_engine_throughput()
    print(f"\nlat data:   {len(df_lat)} rows  (4090 + Orin single-IP)")
    print(f"multi-IP throughput data: {len(df_thr_multi)} rows")

    # Combined dataset
    df_full = pd.concat([df_lat, df_thr_multi], ignore_index=True)
    print(f"combined:   {len(df_full)} rows")
    print(f"by d_scheme: {df_full.d_scheme.value_counts().to_dict()}")

    # Lat training (build_success=True + lat not nan)
    df_lat_ok = df_full[df_full.build_success & df_full.lat_p50_ms.notna()].copy()
    print(f"\nLat training pool: {len(df_lat_ok)} samples")
    X_lat, names = featurize(df_lat_ok)
    y_lat = df_lat_ok["lat_p50_ms"].to_numpy(dtype=np.float32)
    lat_model, lat_metrics = train_regression(X_lat, y_lat, "Lat", names)

    # Throughput training (build_success=True + throughput not nan)
    df_thr_ok = df_full[df_full.build_success & df_full.throughput_fps.notna()].copy()
    print(f"\nThroughput training pool: {len(df_thr_ok)} samples")
    print(f"  by d_scheme: {df_thr_ok.d_scheme.value_counts().to_dict()}")
    X_thr, _ = featurize(df_thr_ok)
    y_thr = df_thr_ok["throughput_fps"].to_numpy(dtype=np.float32)
    thr_model, thr_metrics = train_regression(X_thr, y_thr, "Throughput", names,
                                                params_override={"num_leaves": 15, "min_data_in_leaf": 1})

    # Build_success classifier (full dataset incl. failures)
    X_all, _ = featurize(df_full)
    y_bs = df_full["build_success"].astype(int).to_numpy()
    print(f"\nBuild_success: n={len(y_bs)}, positive_rate={y_bs.mean():.2%}")
    Xtr, Xv, ytr, yv = train_test_split(X_all, y_bs, test_size=0.2, random_state=42, stratify=y_bs)
    bs_params = {"objective": "binary", "metric": "binary_logloss",
                  "learning_rate": 0.05, "num_leaves": 15,
                  "min_data_in_leaf": 3, "num_threads": 1, "verbose": -1, "seed": 42}
    bs_model = lgb.train(bs_params, lgb.Dataset(Xtr, ytr, free_raw_data=False),
                          num_boost_round=200,
                          valid_sets=[lgb.Dataset(Xv, yv, free_raw_data=False)],
                          callbacks=[lgb.early_stopping(stopping_rounds=20)])
    bs_pred = (bs_model.predict(Xv) > 0.5).astype(int)
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(yv, bs_pred)
    auc = roc_auc_score(yv, bs_model.predict(Xv)) if len(np.unique(yv))>1 else None
    bs_model = lgb.train(bs_params, lgb.Dataset(X_all, y_bs, free_raw_data=False), num_boost_round=200)
    print(f"  hold-out acc={acc:.4f} AUC={auc}")
    bs_metrics = {"n_train": len(y_bs), "holdout_accuracy": acc, "holdout_auc": float(auc) if auc else None}

    # Save
    MODELS = ROOT / "models"; RESULTS = ROOT / "results"
    lat_model.save_model(str(MODELS / "lgb_v3_lat.txt"))
    thr_model.save_model(str(MODELS / "lgb_v3_throughput.txt"))
    bs_model.save_model(str(MODELS / "lgb_v3_build_success.txt"))

    metrics = {
        "feature_names": names,
        "lat_predictor": lat_metrics,
        "throughput_predictor": thr_metrics,
        "build_success_predictor": bs_metrics,
        "data_summary": {
            "total_combined": int(len(df_full)),
            "lat_training": int(len(df_lat_ok)),
            "throughput_training": int(len(df_thr_ok)),
            "build_success_training": int(len(df_full)),
            "throughput_by_scheme": df_thr_ok.d_scheme.value_counts().to_dict(),
        },
    }
    with open(RESULTS / "lgb_v3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n✅ saved lgb_v3_lat / throughput / build_success + metrics")

    # Feature importance
    print(f"\n[Throughput predictor feature importance]:")
    fi = sorted(zip(names, thr_model.feature_importance(importance_type="split")), key=lambda x: -x[1])
    for n, v in fi:
        print(f"  {n:20s} {v}")


if __name__ == "__main__":
    main()
