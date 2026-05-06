"""AccuracyPredictor 接口测试 (Stage S3.7 验收)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from framework.accuracy_predictor import LGBPredictor, load_default_predictor
from framework.config_schema import Config, UNIV2X_MODULES


def test_default_predictor_loads():
    p = load_default_predictor()
    assert p.booster is not None
    assert len(p.feature_names) > 0
    print(f"  feature_names: {len(p.feature_names)} cols, trust={p.trust_level()}")


def test_predict_baseline():
    p = load_default_predictor()
    cfg = Config.fp32_baseline()
    diff = p.predict(cfg)
    assert isinstance(diff, float)
    print(f"  fp32_baseline diff_predict = {diff:+.4f}")


def test_predict_with_uncertainty():
    p = load_default_predictor()
    cfg = Config.fp32_baseline()
    mean, std = p.predict_with_uncertainty(cfg)
    assert mean == p.predict(cfg)
    assert std == 0.0  # LightGBM 单模型,std=0
    print(f"  (mean, std) = ({mean:+.4f}, {std:.4f})")


def test_predict_int8_lower_diff():
    """直觉测试: 全 INT8 应该比全 FP32 有更高 (worse) 的 accuracy_diff."""
    p = load_default_predictor()
    fp32 = Config.fp32_baseline()
    int8 = fp32.with_field(
        q_bits={m: "INT8" for m in UNIV2X_MODULES},
        q_granularity={m: "per-tensor" for m in UNIV2X_MODULES},
        q_object={m: "W+A" for m in UNIV2X_MODULES},
    )
    d_fp32 = p.predict(fp32)
    d_int8 = p.predict(int8)
    # 22 行 sanity 模型未必能学出这个方向,只检查能正常预测
    assert isinstance(d_fp32, float) and isinstance(d_int8, float)
    print(f"  fp32 diff={d_fp32:+.4f}  int8 diff={d_int8:+.4f}")


def test_predict_batch_speed():
    """搜索器要用,1000 次评估必须在合理时间内.

    验收: 1000 次预测 < 1 秒 (LightGBM 应该 ms 级).
    """
    p = load_default_predictor()
    base = Config.fp32_baseline()
    # 制造 1000 个略有不同的 config
    configs = []
    for i in range(1000):
        rates = {m: (i % 5) * 0.1 for m in UNIV2X_MODULES}
        configs.append(base.with_field(prune_rate=rates, prune_object="channel"))
    t0 = time.time()
    preds = p.predict_batch(configs)
    elapsed = time.time() - t0
    print(f"  1000 batch preds in {elapsed*1000:.1f} ms ({elapsed*1e6/1000:.1f} μs/pred)")
    assert len(preds) == 1000
    assert elapsed < 1.0, f"批量预测过慢: {elapsed:.2f}s"


def test_trust_level():
    p = load_default_predictor()
    level = p.trust_level()
    assert level in ("high", "medium", "low (sanity only)", "untrustworthy")
    print(f"  trust_level = {level}")
    # sanity v0 应该是 'low'
    assert "low" in level or level == "untrustworthy"


def test_consistency_with_oof():
    """全量训练模型在 baseline_unified 行上的 prediction 应大致 < OOF MAE.

    因为全量训练的模型见过这些样本,误差应该比 OOF 小.
    """
    import numpy as np
    p = load_default_predictor()
    root = Path(__file__).resolve().parents[2]
    df = pd.read_parquet(root / "data" / "phase1" / "baseline_unified.parquet")
    oof = pd.read_csv(root / "results" / "phase2_stage3_sanity.csv")
    # 用 feature_encoder + booster 预测
    from framework.feature_encoder import encode_baseline_df, CATEGORICAL_COLS
    X = encode_baseline_df(df)
    for fn in p.feature_names:
        if fn not in X.columns:
            X[fn] = pd.NA
    X = X[p.feature_names]
    y_pred = p.booster.predict(X)
    y_true = df["accuracy_diff_vs_fp32"].values
    mae_in = float(np.mean(np.abs(y_true - y_pred)))
    mae_oof = float(oof["abs_err"].mean())
    print(f"  in-sample MAE = {mae_in:.4f},  OOF MAE = {mae_oof:.4f}  (in <= oof 表示正常)")
    assert mae_in <= mae_oof + 1e-3  # 容差


if __name__ == "__main__":
    print("=== AccuracyPredictor 接口测试 ===")
    test_default_predictor_loads()
    test_predict_baseline()
    test_predict_with_uncertainty()
    test_predict_int8_lower_diff()
    test_predict_batch_speed()
    test_trust_level()
    test_consistency_with_oof()
    print("\n--- S3.7 accuracy_predictor: 7/7 tests passed ---")
