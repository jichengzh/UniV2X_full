"""Phase 2.4 v5.1: 改进精度预测器

v5 → v5.1 改进:
  (1) 过滤 lat_e2e_ms > 1000 outliers (排 FP32 PyTorch 5640ms 等量级污染)
  (2) 加 source 列作为 one-hot 分类特征 (实验设置异质性建模)
  (3) 引入 stage5_v4_dspace.csv 368 行加权 (real_weight=10, synth_weight=1)
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

LATENCY_OUTLIER_THRESHOLD = 1000.0  # ms


SUBNET_TYPE = {
    "backbone": "CNN", "encoder": "CNN", "decoder": "Transformer",
    "heads": "MLP", "v2x_comm": "MLP",
}
CNN_CRITERIA = ("L1", "FPGM")
TRANSFORMER_CRITERIA = ("Taylor", "Wanda")
MLP_CRITERIA = ("L1",)
ALL_BITS = ("FP32", "FP16", "INT8")
ALL_RUNTIMES = ("pytorch_fp32", "trt_fp16", "trt_int8")
ALL_SOURCES = ("1.1_quant", "1.2_prune", "1.2_prune_ft", "1.2_pareto",
               "1.2_joint", "1.2_P1_FFN", "1.3_d", "plan_b_active",
               "synth_dspace")  # v5.1 新增 synth


def featurize(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feats: list[np.ndarray] = []
    names: list[str] = []

    for m in UNIV2X_MODULES:
        st = SUBNET_TYPE[m]
        feats.append(df[f"prune_rate__{m}"].fillna(0.0).to_numpy().reshape(-1, 1))
        names.append(f"prune_rate__{m}")
        crit_col = df[f"prune_criterion__{m}"].fillna("none")
        if st == "CNN":
            criteria = CNN_CRITERIA
        elif st == "Transformer":
            criteria = TRANSFORMER_CRITERIA
        else:
            criteria = MLP_CRITERIA
        for c in criteria:
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

    # v5.1 新增: source one-hot
    src_col = df["source"].fillna("unknown")
    for s in ALL_SOURCES:
        feats.append((src_col == s).astype(int).to_numpy().reshape(-1, 1))
        names.append(f"source_{s}")

    X = np.hstack(feats).astype(np.float64)
    return X, names


def load_real() -> pd.DataFrame:
    """加载 baseline_4090.parquet 真实样本."""
    df = pd.read_parquet(REAL_INPUT)
    df["is_synth"] = False
    return df


def load_synth() -> pd.DataFrame:
    """把 stage5_v4_dspace.csv 368 行映射到 baseline_4090 schema."""
    src = pd.read_csv(SYNTH_INPUT)
    rows = []
    bit2str = {32: "FP32", 16: "FP16", 8: "INT8"}
    for _, r in src.iterrows():
        row: dict = {
            "config_id": "synth_" + str(r["config_full_id"]),
            "source": "synth_dspace",
            "is_synth": True,
            "is_real_measured": False,
            "amota": float(r.get("amota_v4", r.get("amota", np.nan))),
            "lat_e2e_ms": float(r.get("est_trt_latency_v4_ms", r.get("lat_e2e_ms", np.nan))),
            "prune_object": "channel" if any([
                float(r.get(f"prune_rate__{x}", 0) or 0) > 0
                for x in ("backbone", "encoder_ffn", "encoder_attn", "decoder_ffn", "heads_mid")
            ]) else "none",
        }
        # 5 module 粗粒度
        row["prune_rate__backbone"] = float(r.get("prune_rate__backbone", 0) or 0)
        enc_rates = [float(r.get(f"prune_rate__encoder_{s}", 0) or 0) for s in ("ffn", "attn", "heads")]
        dec_rates = [float(r.get(f"prune_rate__decoder_{s}", 0) or 0) for s in ("ffn", "attn", "heads")]
        row["prune_rate__encoder"] = max(enc_rates)
        row["prune_rate__decoder"] = max(dec_rates)
        row["prune_rate__heads"] = float(r.get("prune_rate__heads_mid", 0) or 0)
        row["prune_rate__v2x_comm"] = 0.0
        for m in UNIV2X_MODULES:
            row[f"prune_criterion__{m}"] = "L1" if row[f"prune_rate__{m}"] > 0 else "none"
            row[f"q_bits__{m}"] = bit2str.get(int(r.get(f"q_bits__{m}", 32) or 32), "FP32")
            gw = str(r.get("q_granularity_w", "per_tensor") or "per_tensor")
            row[f"q_granularity__{m}"] = "per-tensor" if gw == "per_tensor" else "per-channel"
            qt = str(r.get("q_target", "none") or "none")
            row[f"q_object__{m}"] = "W+A" if "A" in qt else ("W-only" if qt == "W" else "none")
            row[f"d_routing__{m}"] = "GPU"
        row["d_runtime"] = str(r.get("d_runtime", "pytorch_fp32"))
        row["d_pipelined"] = int(r.get("d_pipelined_get_bevs", 0) or 0)
        row["d_temporal_cache_int8"] = int(r.get("d_temporal_cache_int8", 0) or 0)
        rows.append(row)
    return pd.DataFrame(rows)


def train_with_loocv(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray,
    feature_names: list[str], target: str, eval_mask: np.ndarray | None = None,
) -> tuple[lgb.Booster, dict]:
    """LOOCV 训练. 用 weights 给真实样本更高权重.

    eval_mask: 只在这些索引上算 LOOCV 指标 (默认全部). 用于"只在真实样本上评估"模式.
    """
    print(f"\n=== Train {target} ({len(y)} rows × {X.shape[1]} feat) ===")
    real_n = int((weights > 1).sum())
    synth_n = len(y) - real_n
    print(f"  real={real_n} (w=10), synth={synth_n} (w=1)")
    print(f"  y range: [{y.min():.4f}, {y.max():.4f}] (span {y.max() - y.min():.4f})")

    if eval_mask is None:
        eval_mask = np.ones(len(y), dtype=bool)
    eval_idx = np.where(eval_mask)[0]
    print(f"  eval on {len(eval_idx)} rows (real-only LOOCV)")

    params = {
        "objective": "regression", "metric": "mae", "verbosity": -1,
        "learning_rate": 0.05, "num_leaves": 7, "min_data_in_leaf": 2,
        "feature_pre_filter": False, "num_threads": 2, "lambda_l2": 0.1,
    }

    preds_eval = np.zeros(len(eval_idx), dtype=np.float64)
    for i, test_i in enumerate(eval_idx):
        train_idx = np.delete(np.arange(len(y)), test_i)
        train_set = lgb.Dataset(X[train_idx], y[train_idx], weight=weights[train_idx])
        booster = lgb.train(params, train_set, num_boost_round=80)
        preds_eval[i] = booster.predict(X[test_i:test_i + 1])[0]

    final = lgb.train(params, lgb.Dataset(X, y, weight=weights), num_boost_round=80)

    y_eval = y[eval_idx]
    rho, p = spearmanr(y_eval, preds_eval)
    mae = mean_absolute_error(y_eval, preds_eval)
    rmse = float(np.sqrt(np.mean((y_eval - preds_eval) ** 2)))
    rel_mae = mae / max(y_eval.max() - y_eval.min(), 1e-6)

    metrics = {
        "target": target, "n_rows_train": int(len(y)),
        "n_rows_eval_real": int(len(eval_idx)),
        "n_real": real_n, "n_synth": synth_n,
        "n_features": int(X.shape[1]),
        "y_min": float(y_eval.min()), "y_max": float(y_eval.max()),
        "spearman_loocv": float(rho), "p_value": float(p),
        "mae_loocv": float(mae), "rmse_loocv": rmse,
        "relative_mae_pct": float(rel_mae * 100),
    }
    print(f"  Spearman = {rho:.3f} (p={p:.4g})")
    print(f"  MAE      = {mae:.4f}  (relative {rel_mae*100:.1f}% of eval span)")
    print(f"  RMSE     = {rmse:.4f}")
    return final, metrics


def feature_importance(booster: lgb.Booster, names: list[str]) -> pd.DataFrame:
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    df = pd.DataFrame({"feature": names, "gain": gain, "split": split})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)

    def subnet_of(fname: str) -> str:
        if fname.startswith("source_"):
            return "Source"
        for m, st in SUBNET_TYPE.items():
            if f"_{m}" in fname or fname.endswith(m):
                return st
        if fname.startswith("d_") or fname.startswith("prune_object"):
            return "Global"
        return "Other"
    df["subnet_type"] = df["feature"].apply(subnet_of)
    return df


def main() -> None:
    print("=" * 60)
    print("Phase 2.4 v5.1 — outlier filter + source feature + dspace 加权")
    print("=" * 60)

    real = load_real()
    synth = load_synth()
    print(f"✅ Real:  {len(real)} rows from baseline_4090.parquet")
    print(f"✅ Synth: {len(synth)} rows from stage5_v4_dspace.csv")

    # 列对齐 (synth 缺列填默认)
    union_cols = set(real.columns) | set(synth.columns)
    for col in union_cols:
        if col not in real.columns:
            real[col] = np.nan
        if col not in synth.columns:
            synth[col] = np.nan
    common_cols = sorted(union_cols)
    real = real[common_cols]
    synth = synth[common_cols]

    combined = pd.concat([real, synth], ignore_index=True)
    print(f"   Combined: {len(combined)} rows")

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_all: dict = {}

    # ---- amota predictor ----
    df_a = combined[combined["amota"].notna()].reset_index(drop=True).copy()
    Xa, fnames = featurize(df_a)
    ya = df_a["amota"].to_numpy()
    wa = np.where(df_a["is_synth"].fillna(False), 1.0, 10.0)
    eval_mask_a = ~df_a["is_synth"].fillna(False).to_numpy()
    booster_a, m_a = train_with_loocv(Xa, ya, wa, fnames, "amota", eval_mask=eval_mask_a)
    booster_a.save_model(str(MODELS_DIR / "lgb_v5_1_amota.txt"))
    metrics_all["amota"] = m_a

    fi_a = feature_importance(booster_a, fnames)
    print("\n  Top 10 (amota):")
    print(fi_a.head(10).to_string(index=False))
    print("\n  Subnet aggregated gain (amota):")
    print(fi_a.groupby("subnet_type")["gain"].sum().sort_values(ascending=False).to_string())

    # ---- latency predictor (过滤 outlier) ----
    df_l = combined[combined["lat_e2e_ms"].notna()].reset_index(drop=True).copy()
    pre_filter = len(df_l)
    df_l = df_l[df_l["lat_e2e_ms"] <= LATENCY_OUTLIER_THRESHOLD].reset_index(drop=True)
    print(f"\n  Latency outlier filter: {pre_filter} → {len(df_l)} (cut {pre_filter - len(df_l)} rows > {LATENCY_OUTLIER_THRESHOLD}ms)")

    Xl, _ = featurize(df_l)
    yl = df_l["lat_e2e_ms"].to_numpy()
    wl = np.where(df_l["is_synth"].fillna(False), 1.0, 10.0)
    eval_mask_l = ~df_l["is_synth"].fillna(False).to_numpy()
    booster_l, m_l = train_with_loocv(Xl, yl, wl, fnames, "lat_e2e_ms", eval_mask=eval_mask_l)
    booster_l.save_model(str(MODELS_DIR / "lgb_v5_1_latency.txt"))
    metrics_all["lat_e2e_ms"] = m_l

    fi_l = feature_importance(booster_l, fnames)
    print("\n  Top 10 (latency):")
    print(fi_l.head(10).to_string(index=False))
    print("\n  Subnet aggregated gain (latency):")
    print(fi_l.groupby("subnet_type")["gain"].sum().sort_values(ascending=False).to_string())

    metrics_path = RESULTS_DIR / "phase2_4_lgb_v5_1_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"\n✅ Saved metrics → {metrics_path}")

    fi_combined = pd.concat([
        fi_a.assign(target="amota"),
        fi_l.assign(target="lat_e2e_ms"),
    ], ignore_index=True)
    fi_path = RESULTS_DIR / "phase2_4_lgb_v5_1_feature_importance.csv"
    fi_combined.to_csv(fi_path, index=False)
    print(f"✅ Saved FI → {fi_path}")

    print("\n" + "=" * 60)
    print("v5.1 vs v5 改进")
    print("=" * 60)
    with open(RESULTS_DIR / "phase2_4_lgb_v5_metrics.json") as f:
        v5 = json.load(f)
    print(f"  amota   spearman: v5={v5['amota']['spearman_loocv']:.3f} → v5.1={m_a['spearman_loocv']:.3f}")
    print(f"  amota   MAE:      v5={v5['amota']['mae_loocv']:.4f} → v5.1={m_a['mae_loocv']:.4f}")
    print(f"  latency spearman: v5={v5['lat_e2e_ms']['spearman_loocv']:.3f} → v5.1={m_l['spearman_loocv']:.3f}")
    print(f"  latency MAE (ms): v5={v5['lat_e2e_ms']['mae_loocv']:.1f} → v5.1={m_l['mae_loocv']:.1f}")


if __name__ == "__main__":
    main()
