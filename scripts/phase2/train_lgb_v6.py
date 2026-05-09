"""Phase 2.4 v6: 跨模型 LGB 预测器 (扩展 v5.1, 加 model_class + Pyramid 锚点).

v5.1 → v6 改进:
  (1) 加 'm4_5_pyramid_configs', 'm4_6_0_pyramid_full_e2e',
      'm4_6_1_pruning', 'm4_6_2_multimodule', 'm4_6_3_pareto_validation' 到 ALL_SOURCES
  (2) 加 model_class one-hot (univ2x_full / uniad_tiny_variant / pyramid_fusion)
       让 LGB 学到 "Pyramid baseline e2e ~30ms / amota=0.96, univ2x baseline ~5640ms / amota=0.32"
       模型类别差异
  (3) 不过滤 lat_e2e_ms outlier (Pyramid 完整 e2e 30ms 跟 univ2x_full FP32 5640ms 同时入训, 让 LGB 学跨数量级 generalization)

输出:
  models/lgb_v6_amota.txt
  models/lgb_v6_latency.txt
  results/phase2_4_lgb_v6_metrics.json
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
sys.path.insert(0, str(ROOT))

from framework.config_schema import UNIV2X_MODULES

REAL_INPUT = ROOT / "data/baseline_4090.parquet"
SYNTH_INPUT = ROOT / "data/phase4/stage5_v4_dspace.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


SUBNET_TYPE_PYRAMID = {
    "backbone": "CNN", "encoder": "CNN", "decoder": "CNN",      # ★ Pyramid decoder 是 CNN
    "heads": "MLP", "v2x_comm": "CNN",                            # ★ Pyramid v2x 也是 CNN
}
SUBNET_TYPE_UNIV2X = {
    "backbone": "CNN", "encoder": "CNN", "decoder": "Transformer",
    "heads": "MLP", "v2x_comm": "MLP",
}
CNN_CRITERIA = ("L1", "FPGM")
TRANSFORMER_CRITERIA = ("Taylor", "Wanda")
MLP_CRITERIA = ("L1",)
ALL_BITS = ("FP32", "FP16", "INT8")
ALL_RUNTIMES = ("pytorch_fp32", "pytorch_fp16", "trt_fp16", "trt_int8")
ALL_SOURCES = (
    "1.1_quant", "1.2_prune", "1.2_prune_ft", "1.2_pareto", "1.2_joint",
    "1.2_P1_FFN", "1.3_d", "plan_b_active", "synth_dspace",
    # v6 新增 Pyramid 锚点 source
    "m4_5_pyramid_configs", "m4_6_0_pyramid_full_e2e",
    "m4_6_1_pruning", "m4_6_2_multimodule", "m4_6_3_pareto_validation",
)
ALL_MODEL_CLASSES = ("univ2x_full", "uniad_tiny_variant", "pyramid_fusion")


def featurize_v6(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """v6 增加 model_class one-hot, 其余跟 v5.1 一致."""
    feats: list[np.ndarray] = []
    names: list[str] = []

    for m in UNIV2X_MODULES:
        # 用 univ2x subnet type 作 base; v6 让 LGB 自己学跨模型差异 (不显式区分)
        st = SUBNET_TYPE_UNIV2X[m]
        feats.append(df[f"prune_rate__{m}"].fillna(0.0).to_numpy().reshape(-1, 1))
        names.append(f"prune_rate__{m}")
        crit_col = df[f"prune_criterion__{m}"].fillna("none")
        # v6: 用 union of CNN+Transformer+MLP criteria (LGB 自己挑)
        all_criteria = list(set(CNN_CRITERIA + TRANSFORMER_CRITERIA + MLP_CRITERIA))
        for c in all_criteria:
            feats.append((crit_col == c).astype(int).to_numpy().reshape(-1, 1))
            names.append(f"crit_{m}_{c}")
        qb_col = df[f"q_bits__{m}"].fillna("FP32")
        for b in ALL_BITS:
            feats.append((qb_col == b).astype(int).to_numpy().reshape(-1, 1))
            names.append(f"qbits_{m}_{b}")
        qg_col = df[f"q_granularity__{m}"].fillna("none")
        for g in ("per-tensor", "per-channel"):
            feats.append((qg_col == g).astype(int).to_numpy().reshape(-1, 1))
            names.append(f"qgran_{m}_{g}")
        dr_col = df[f"d_routing__{m}"].fillna("GPU")
        feats.append(dr_col.str.startswith("DLA").astype(int).to_numpy().reshape(-1, 1))
        names.append(f"d_dla_{m}")

    po_col = df["prune_object"].fillna("none")
    for p in ("none", "channel", "2:4"):
        feats.append((po_col == p).astype(int).to_numpy().reshape(-1, 1))
        names.append(f"prune_object_{p}")

    if "d_pipelined" in df.columns:
        feats.append(df["d_pipelined"].fillna(0).astype(int).to_numpy().reshape(-1, 1))
        names.append("d_pipelined")
    if "d_temporal_cache_int8" in df.columns:
        feats.append(df["d_temporal_cache_int8"].fillna(0).astype(int).to_numpy().reshape(-1, 1))
        names.append("d_temporal_cache_int8")
    if "d_runtime" in df.columns:
        rt_col = df["d_runtime"].fillna("pytorch_fp32")
        for r in ALL_RUNTIMES:
            feats.append((rt_col == r).astype(int).to_numpy().reshape(-1, 1))
            names.append(f"d_runtime_{r}")

    src_col = df["source"].fillna("unknown")
    for s in ALL_SOURCES:
        feats.append((src_col == s).astype(int).to_numpy().reshape(-1, 1))
        names.append(f"source_{s}")

    # v6 ★ model_class one-hot
    if "model_class" in df.columns:
        mc_col = df["model_class"].fillna("unknown")
        for mc in ALL_MODEL_CLASSES:
            feats.append((mc_col == mc).astype(int).to_numpy().reshape(-1, 1))
            names.append(f"model_class_{mc}")

    X = np.hstack(feats).astype(np.float64)
    return X, names


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(REAL_INPUT)
    print(f"baseline 行数: {len(df)}")
    print(f"  model_class 分布: {df['model_class'].value_counts().to_dict()}")
    return df


def train_one(X, y, names, target_name: str, weights=None) -> tuple[lgb.Booster, dict]:
    """单目标训练 + KFold(5) evaluation (替代 LOOCV — 避免 early_stopping single-sample valid 问题)."""
    from sklearn.model_selection import KFold
    params = {
        "objective": "regression",
        "metric": "l1",
        "num_leaves": 31,
        "min_data_in_leaf": 3,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "num_threads": 4,
    }
    n = len(y)
    n_folds = min(5, n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    preds = np.zeros(n)
    for train_idx, test_idx in kf.split(X):
        train_w = weights[train_idx] if weights is not None else None
        booster = lgb.train(
            params,
            lgb.Dataset(X[train_idx], y[train_idx], weight=train_w),
            num_boost_round=200,  # 固定 round, 不 early_stop
        )
        preds[test_idx] = booster.predict(X[test_idx])

    sr, _ = spearmanr(preds, y)
    mae = mean_absolute_error(y, preds)

    # 全数据训终模型
    final = lgb.train(params, lgb.Dataset(X, y, weight=weights), num_boost_round=200)
    return final, {"target": target_name, "n_train": n,
                   "cv_spearman": float(sr), "cv_mae": float(mae),
                   "n_folds": n_folds}


def main() -> None:
    print("=" * 60)
    print("Phase 2.4 v6 — 跨模型 LGB (加 model_class one-hot + Pyramid 锚点)")
    print("=" * 60)
    df = load_data()
    df_a = df[df["amota"].notna()].reset_index(drop=True)
    df_l = df[df["lat_e2e_ms"].notna()].reset_index(drop=True)
    print(f"\n  amota 训练样本: {len(df_a)} (model_class: {df_a['model_class'].value_counts().to_dict()})")
    print(f"  latency 训练样本: {len(df_l)} (model_class: {df_l['model_class'].value_counts().to_dict()})")

    # featurize
    Xa, names_a = featurize_v6(df_a)
    Xl, names_l = featurize_v6(df_l)
    ya = df_a["amota"].to_numpy()
    yl = df_l["lat_e2e_ms"].to_numpy()
    print(f"  amota X shape: {Xa.shape}, latency X shape: {Xl.shape}")

    # 权重: 完整 e2e 实测 + Pyramid M4.6 实测 高权重 (real anchor); 子模块 / synth 低权重
    weight_a = np.ones(len(df_a))
    real_anchor_sources = ("m4_6_0_pyramid_full_e2e", "m4_6_1_pruning",
                           "m4_6_2_multimodule", "m4_6_3_pareto_validation")
    for src in real_anchor_sources:
        weight_a[df_a["source"] == src] = 3.0  # 实测 anchor 权重 3×

    print(f"\n  amota weights: real anchor={int((weight_a==3.0).sum())}, normal={int((weight_a==1.0).sum())}")

    booster_a, m_a = train_one(Xa, ya, names_a, "amota", weights=weight_a)
    print(f"\n  amota CV: spearman={m_a['cv_spearman']:.4f} MAE={m_a['cv_mae']:.4f}")

    booster_l, m_l = train_one(Xl, yl, names_l, "lat_e2e_ms")
    print(f"  latency CV: spearman={m_l['cv_spearman']:.4f} MAE={m_l['cv_mae']:.2f} ms")

    # Save
    booster_a.save_model(str(MODELS_DIR / "lgb_v6_amota.txt"))
    booster_l.save_model(str(MODELS_DIR / "lgb_v6_latency.txt"))
    print(f"\n✅ Saved {MODELS_DIR}/lgb_v6_{{amota,latency}}.txt")

    metrics = {"v6_features": Xa.shape[1], "amota": m_a, "latency": m_l}
    metrics_path = RESULTS_DIR / "phase2_4_lgb_v6_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"✅ {metrics_path}")

    # Pyramid 锚点上的训练-集 fit 准度 (in-sample, 不是 CV)
    print(f"\n=== Pyramid 锚点上的训练 fit 准度 (in-sample) ===")
    py_idx = df_a[df_a["model_class"] == "pyramid_fusion"].index.tolist()
    if py_idx:
        py_pred = booster_a.predict(Xa[py_idx])
        py_actual = ya[py_idx]
        py_mae = mean_absolute_error(py_actual, py_pred)
        py_sr, _ = spearmanr(py_pred, py_actual)
        print(f"  pyramid_fusion N={len(py_idx)}: spearman={py_sr:.4f} MAE={py_mae:.4f}")
        for ii, i in enumerate(py_idx):
            if ii < 8 or ii >= len(py_idx) - 4:
                print(f"    {df_a.loc[i, 'config_id']:<35} actual={ya[i]:.4f} pred={py_pred[ii]:.4f}")
            elif ii == 8:
                print(f"    ... ({len(py_idx) - 12} more)")


if __name__ == "__main__":
    main()
