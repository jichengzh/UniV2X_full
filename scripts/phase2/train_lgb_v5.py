"""Phase 2.4: 精度预测器 v5 (按 v1.5 §0.5 子网准则重做)

对应 v1.5 §0.5 — 不同子网的特征重要性应分别建模:
  - CNN 子网 (backbone, encoder): L1/FPGM 准则
  - Transformer 子网 (decoder): Taylor/Wanda 准则
  - MLP 子网 (heads, v2x_comm): L1 准则

输入: data/baseline_4090.parquet (52/57 行有 amota, 38/57 有 lat_e2e_ms)
模型: LightGBM (LOOCV due to 小样本)
输出:
  - models/lgb_v5_amota.txt
  - models/lgb_v5_latency.txt
  - results/phase2_4_lgb_v5_metrics.json
  - results/phase2_4_lgb_v5_feature_importance.csv
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

INPUT = ROOT / "data/baseline_4090.parquet"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


# ---------- v1.5 §0.5 子网类型分类 ----------

SUBNET_TYPE = {
    "backbone": "CNN",       # ResNet101 backbone
    "encoder": "CNN",        # BEV encoder (CNN-based)
    "decoder": "Transformer",  # Track decoder (attention)
    "heads": "MLP",          # Detection / Map / Motion / Plan heads
    "v2x_comm": "MLP",       # V2X message passing
}

CNN_CRITERIA = ("L1", "FPGM")
TRANSFORMER_CRITERIA = ("Taylor", "Wanda")
MLP_CRITERIA = ("L1",)
ALL_CRITERIA = ("L1", "FPGM", "Taylor", "Wanda", "none")
ALL_BITS = ("FP32", "FP16", "INT8")
ALL_GRAN = ("per-tensor", "per-channel", "none")
ALL_QOBJ = ("W-only", "W+A", "none")
ALL_RUNTIMES = ("pytorch_fp32", "trt_fp16", "trt_int8")


# ---------- 特征工程 (按 v1.5 §0.5 子网类型) ----------

def featurize(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """按子网类型构造特征向量.

    每个 module 5 个特征:
      - prune_rate (continuous)
      - criterion (one-hot, 类型对齐: CNN→L1/FPGM, Transformer→Taylor/Wanda, MLP→L1)
      - q_bits (one-hot 3)
      - q_granularity (one-hot 2: per-tensor / per-channel)
      - d_routing_is_dla (binary, 0/1)

    全局 4 特征:
      - prune_object (one-hot 3: none/channel/2:4)
      - d_pipelined (binary)
      - d_temporal_cache_int8 (binary)
      - d_runtime (one-hot 3)
    """
    feats: list[np.ndarray] = []
    names: list[str] = []

    for m in UNIV2X_MODULES:
        st = SUBNET_TYPE[m]

        # prune_rate (continuous)
        feats.append(df[f"prune_rate__{m}"].fillna(0.0).to_numpy().reshape(-1, 1))
        names.append(f"prune_rate__{m}")

        # criterion (按子网类型)
        crit_col = df[f"prune_criterion__{m}"].fillna("none")
        if st == "CNN":
            for c in CNN_CRITERIA:
                feats.append((crit_col == c).astype(int).to_numpy().reshape(-1, 1))
                names.append(f"crit_{m}_{c}")
        elif st == "Transformer":
            for c in TRANSFORMER_CRITERIA:
                feats.append((crit_col == c).astype(int).to_numpy().reshape(-1, 1))
                names.append(f"crit_{m}_{c}")
        else:  # MLP
            for c in MLP_CRITERIA:
                feats.append((crit_col == c).astype(int).to_numpy().reshape(-1, 1))
                names.append(f"crit_{m}_{c}")

        # q_bits one-hot (3)
        qb_col = df[f"q_bits__{m}"].fillna("FP32")
        for b in ALL_BITS:
            feats.append((qb_col == b).astype(int).to_numpy().reshape(-1, 1))
            names.append(f"qbits_{m}_{b}")

        # q_granularity (2 = per-tensor / per-channel)
        qg_col = df[f"q_granularity__{m}"].fillna("none")
        for g in ("per-tensor", "per-channel"):
            feats.append((qg_col == g).astype(int).to_numpy().reshape(-1, 1))
            names.append(f"qgran_{m}_{g}")

        # d_routing_is_dla
        dr_col = df[f"d_routing__{m}"].fillna("GPU")
        feats.append(dr_col.str.startswith("DLA").astype(int).to_numpy().reshape(-1, 1))
        names.append(f"d_dla_{m}")

    # 全局
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

    X = np.hstack(feats).astype(np.float64)
    return X, names


# ---------- 训练 + LOOCV ----------

def train_with_loocv(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], target: str,
) -> tuple[lgb.Booster, dict]:
    print(f"\n=== Train {target} ({len(y)} rows × {X.shape[1]} feat) ===")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}] (span {y.max() - y.min():.4f})")

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "min_data_in_leaf": 2,
        "feature_pre_filter": False,
        "num_threads": 2,
        "lambda_l2": 0.1,
    }

    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=np.float64)
    for train_idx, test_idx in loo.split(X):
        train_set = lgb.Dataset(X[train_idx], y[train_idx])
        booster = lgb.train(params, train_set, num_boost_round=80)
        preds[test_idx] = booster.predict(X[test_idx])

    # 全部数据训最终模型
    final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=80)

    rho, p = spearmanr(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    rel_mae = mae / max(y.max() - y.min(), 1e-6)

    metrics = {
        "target": target,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_span": float(y.max() - y.min()),
        "spearman_loocv": float(rho),
        "p_value": float(p),
        "mae_loocv": float(mae),
        "rmse_loocv": rmse,
        "relative_mae_pct": float(rel_mae * 100),
    }
    print(f"  Spearman = {rho:.3f} (p={p:.4g})")
    print(f"  MAE      = {mae:.4f}  (relative {rel_mae*100:.1f}% of y span)")
    print(f"  RMSE     = {rmse:.4f}")
    return final, metrics


def feature_importance(booster: lgb.Booster, names: list[str]) -> pd.DataFrame:
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    df = pd.DataFrame({"feature": names, "gain": gain, "split": split})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)

    def subnet_of(fname: str) -> str:
        for m, st in SUBNET_TYPE.items():
            if f"_{m}" in fname or fname.endswith(m):
                return st
        if fname.startswith("d_") or fname.startswith("prune_object"):
            return "Global"
        return "Other"

    df["subnet_type"] = df["feature"].apply(subnet_of)
    return df


# ---------- 主流程 ----------

def main() -> None:
    print("=" * 60)
    print("Phase 2.4 — LGB v5 (v1.5 §0.5 子网准则)")
    print("=" * 60)

    df = pd.read_parquet(INPUT)
    print(f"✅ Loaded {INPUT}: {len(df)} rows × {len(df.columns)} cols")

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_all: dict[str, dict] = {}

    # ---- amota predictor ----
    df_a = df[df["amota"].notna()].reset_index(drop=True).copy()
    Xa, fnames = featurize(df_a)
    ya = df_a["amota"].to_numpy()
    booster_a, m_a = train_with_loocv(Xa, ya, fnames, "amota")
    booster_a.save_model(str(MODELS_DIR / "lgb_v5_amota.txt"))
    metrics_all["amota"] = m_a

    fi_a = feature_importance(booster_a, fnames)
    print("\n  Top 10 features (amota):")
    print(fi_a.head(10).to_string(index=False))
    print("\n  Subnet type aggregated gain:")
    print(fi_a.groupby("subnet_type")["gain"].sum().sort_values(ascending=False).to_string())

    # ---- latency predictor ----
    df_l = df[df["lat_e2e_ms"].notna()].reset_index(drop=True).copy()
    Xl, _ = featurize(df_l)
    yl = df_l["lat_e2e_ms"].to_numpy()
    booster_l, m_l = train_with_loocv(Xl, yl, fnames, "lat_e2e_ms")
    booster_l.save_model(str(MODELS_DIR / "lgb_v5_latency.txt"))
    metrics_all["lat_e2e_ms"] = m_l

    fi_l = feature_importance(booster_l, fnames)
    print("\n  Top 10 features (latency):")
    print(fi_l.head(10).to_string(index=False))
    print("\n  Subnet type aggregated gain:")
    print(fi_l.groupby("subnet_type")["gain"].sum().sort_values(ascending=False).to_string())

    # 保存 metrics + importance
    metrics_path = RESULTS_DIR / "phase2_4_lgb_v5_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\n✅ Saved metrics → {metrics_path}")

    fi_combined = pd.concat([
        fi_a.assign(target="amota"),
        fi_l.assign(target="lat_e2e_ms"),
    ], ignore_index=True)
    fi_path = RESULTS_DIR / "phase2_4_lgb_v5_feature_importance.csv"
    fi_combined.to_csv(fi_path, index=False)
    print(f"✅ Saved feature importance → {fi_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  amota predictor:   spearman={m_a['spearman_loocv']:.3f}  MAE={m_a['mae_loocv']:.4f}  ({m_a['relative_mae_pct']:.1f}% span)")
    print(f"  latency predictor: spearman={m_l['spearman_loocv']:.3f}  MAE={m_l['mae_loocv']:.1f} ms ({m_l['relative_mae_pct']:.1f}% span)")


if __name__ == "__main__":
    main()
