"""Stage 3 LightGBM 精度预测器训练 + 评估 (S3.2/S3.3 sanity).

22 行真实数据下用 LOOCV (留一交叉验证),输出:
- Spearman ρ (排序相关性) — 主指标 (目标 > 0.85, sanity 阶段达 0.5+ 即可)
- MAE — 绝对误差
- Pareto rank correlation
- feature importance — 指导 Stage 2.5 主动采样

设计简化 (针对 22 行小数据):
- 用 sklearn LGBMRegressor (比 lgb.train 简单稳定)
- 固定 n_estimators=50,不做 early stopping (val 集太小会死循环)
- 不做 bagging (行数太少)
- 不做 LOOCV 内 train/val 二次切分

输出:
- models/lgb_predictor_v0.txt          — LightGBM 模型
- models/lgb_predictor_v0_meta.json    — 训练 metadata
- results/phase2_stage3_sanity.csv     — 每折 OOF 预测
- results/phase2_stage3_metrics.json   — 评估指标
- results/phase2_stage3_feat_imp.csv   — 特征重要性
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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework.feature_encoder import (
    CATEGORICAL_COLS,
    encode_baseline_df,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# LightGBM 参数 — 22 行数据保守配置
# ============================================================
LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.05,
    num_leaves=7,
    min_data_in_leaf=1,
    min_data_in_bin=1,
    feature_fraction=1.0,       # 不做特征采样
    bagging_fraction=1.0,       # 不做 bagging
    bagging_freq=0,
    lambda_l1=0.1,
    lambda_l2=0.1,
    n_estimators=50,
    verbosity=-1,
    n_jobs=1,                   # 单线程,避免小数据并行 overhead
)


def pareto_rank(y: np.ndarray) -> np.ndarray:
    return pd.Series(y).rank(method="average").values


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    spear, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pr_true = pareto_rank(y_true)
    pr_pred = pareto_rank(y_pred)
    pr_spear, _ = spearmanr(pr_true, pr_pred)
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = float("nan")
    return dict(
        spearman_rho=float(spear) if spear == spear else 0.0,
        mae=float(mae),
        pareto_rank_rho=float(pr_spear) if pr_spear == pr_spear else 0.0,
        r2=float(r2) if r2 == r2 else 0.0,
    )


def main() -> None:
    print("=" * 60)
    print("Phase 2 Stage 3 — LightGBM 精度预测器 sanity")
    print("=" * 60, flush=True)

    # ---------- 1. 加载数据 ----------
    df = pd.read_parquet(ROOT / "data" / "phase1" / "baseline_unified.parquet")
    print(f"\n[data] {len(df)} 行 / sources: {dict(df.source.value_counts())}", flush=True)

    X_full = encode_baseline_df(df)
    y_full = df["accuracy_diff_vs_fp32"].astype(float).values
    n_feat = X_full.shape[1]
    n_cat = sum(c in CATEGORICAL_COLS for c in X_full.columns)
    print(f"[features] {n_feat} 列 ({n_cat} 类别 + {n_feat - n_cat} 数值)", flush=True)
    print(f"[target] accuracy_diff_vs_fp32 range [{y_full.min():.4f}, {y_full.max():.4f}]", flush=True)

    cat_cols = [c for c in X_full.columns if c in CATEGORICAL_COLS]

    # ---------- 2. LOOCV ----------
    print("\n[LOOCV] 开始留一交叉验证...", flush=True)
    loo = LeaveOneOut()
    oof_pred = np.full(len(y_full), np.nan)
    train_maes = []

    for fold_i, (tr, te) in enumerate(loo.split(X_full)):
        X_tr = X_full.iloc[tr]
        X_te = X_full.iloc[te]
        y_tr = y_full[tr]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_tr, y_tr, categorical_feature=cat_cols)
        oof_pred[te] = model.predict(X_te)[0]
        train_maes.append(mean_absolute_error(y_tr, model.predict(X_tr)))

        if (fold_i + 1) % 5 == 0 or fold_i == 0:
            print(f"  fold {fold_i+1:2d}/{len(y_full)} done", flush=True)

    print(f"[LOOCV] 完成 {len(y_full)} 折", flush=True)

    # ---------- 3. 评估 ----------
    metrics = evaluate(y_full, oof_pred)
    metrics["mean_train_mae"] = float(np.mean(train_maes))
    metrics["n_samples"] = int(len(y_full))
    metrics["target_spearman"] = 0.85
    metrics["sanity_threshold"] = 0.50

    print("\n" + "-" * 60)
    print("LOOCV 结果")
    print("-" * 60)
    print(f"  Spearman ρ:           {metrics['spearman_rho']:+.4f}  (sanity > 0.50, 终目标 > 0.85)")
    print(f"  Pareto rank ρ:        {metrics['pareto_rank_rho']:+.4f}  (终目标 > 0.75)")
    print(f"  MAE:                  {metrics['mae']:.4f}            (终目标 < 0.005)")
    print(f"  R²:                   {metrics['r2']:+.4f}")
    print(f"  mean train MAE:       {metrics['mean_train_mae']:.4f}            (远 << OOF MAE 说明过拟合)")

    # ---------- 4. 全量训练 (用于 feature importance & 固化模型) ----------
    print("\n[full-fit] 全量数据训练用于 feature importance...", flush=True)
    full_model = lgb.LGBMRegressor(**LGB_PARAMS)
    full_model.fit(X_full, y_full, categorical_feature=cat_cols)

    imp = pd.DataFrame({
        "feature": full_model.booster_.feature_name(),
        "gain": full_model.booster_.feature_importance(importance_type="gain"),
        "split": full_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    print("\n" + "-" * 60)
    print("Feature importance (Top-10 by gain)")
    print("-" * 60)
    for _, r in imp.head(10).iterrows():
        if r.gain > 0:
            print(f"  {r['feature']:40s}  gain={r['gain']:8.2f}  splits={int(r['split']):3d}")

    n_used = (imp["gain"] > 0).sum()
    print(f"\n  [info] 实际被用到的特征: {n_used}/{n_feat}")

    # ---------- 5. 落盘 ----------
    out_models = ROOT / "models"
    out_results = ROOT / "results"
    out_models.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    full_model.booster_.save_model(str(out_models / "lgb_predictor_v0.txt"))
    meta = {
        "stage": "S3.2 sanity",
        "n_samples": int(len(y_full)),
        "n_features": int(n_feat),
        "n_features_used": int(n_used),
        "categorical_feature_names": cat_cols,
        "lgb_params": {k: v for k, v in LGB_PARAMS.items()},
        "metrics_loocv": metrics,
    }
    with open(out_models / "lgb_predictor_v0_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    oof_df = pd.DataFrame({
        "config_id": df["config_id"].values,
        "source": df["source"].values,
        "y_true": y_full,
        "y_pred": oof_pred,
        "abs_err": np.abs(y_full - oof_pred),
    }).sort_values("abs_err", ascending=False).reset_index(drop=True)
    oof_df.to_csv(out_results / "phase2_stage3_sanity.csv", index=False)

    with open(out_results / "phase2_stage3_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    imp.to_csv(out_results / "phase2_stage3_feat_imp.csv", index=False)

    # ---------- 6. sanity 判定 ----------
    print("\n" + "=" * 60)
    if metrics["spearman_rho"] >= 0.85:
        print(f"[FINAL TARGET REACHED] Spearman ρ = {metrics['spearman_rho']:.4f} ≥ 0.85,可直接进 Stage 4")
    elif metrics["spearman_rho"] > metrics["sanity_threshold"]:
        print(f"[sanity PASS] Spearman ρ = {metrics['spearman_rho']:.4f} > {metrics['sanity_threshold']}")
        print("  下一步: Stage 2.5 补数据到 100+ 行后重训")
    else:
        print(f"[sanity SOFT-FAIL] Spearman ρ = {metrics['spearman_rho']:.4f} ≤ {metrics['sanity_threshold']}")
        print("  常见原因: (a) 数据过少 (22 行); (b) target 跨度窄; (c) 特征工程缺关键交互项")
        print("  下一步: Stage 2.5 主动采样,然后回炉重训")
    print("=" * 60)
    print("\n输出:")
    print(f"  {out_models / 'lgb_predictor_v0.txt'}")
    print(f"  {out_models / 'lgb_predictor_v0_meta.json'}")
    print(f"  {out_results / 'phase2_stage3_sanity.csv'}")
    print(f"  {out_results / 'phase2_stage3_metrics.json'}")
    print(f"  {out_results / 'phase2_stage3_feat_imp.csv'}")


if __name__ == "__main__":
    main()
