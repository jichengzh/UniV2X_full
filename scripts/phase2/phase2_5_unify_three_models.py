"""Phase 2.5 三模型联合 Pareto 骨架.

整合三个 50 候选 Pareto:
  - A7  (univ2x_full):       results/phase1a_pareto.csv         50 候选, 5 Pareto
  - M4.7 (pyramid_fusion):    results/phase2_pareto_pyramid.csv   50 候选, 1 Pareto
  - M5.6 (uniad_tiny_variant): results/phase2_pareto_uniad_tiny.csv 33 候选, 6 Pareto

输出:
  results/phase2_5_three_models_pareto.csv  — 统一表 (latency × amota / params)
  results/phase2_5_three_models_pareto.parquet

注意:
  - 三模型 latency 跨 4 个数量级: pyramid (~3-5ms) → uniad_tiny (~520ms) → univ2x_full (~90-5640ms)
  - amota 维度:
      * A7: 用 LightGBM v3/v4 预测的 est_amota (univ2x baseline 训)
      * M4.7: amota_pending (待 M4.5 baseline AP + 抽样验证)
      * M5.6: amota_pending (待 M5.7 抽样实测 + v5.1 预测)
  - Pareto 各自前沿 (model_class 内部) 已标在源 csv; 联合表保留 is_pareto 列
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

A7_CSV = RESULTS_DIR / "phase1a_pareto.csv"
M47_CSV = RESULTS_DIR / "phase2_pareto_pyramid.csv"
M56_CSV = RESULTS_DIR / "phase2_pareto_uniad_tiny.csv"

OUT_CSV = RESULTS_DIR / "phase2_5_three_models_pareto.csv"
OUT_PARQUET = RESULTS_DIR / "phase2_5_three_models_pareto.parquet"


def normalize_a7(df: pd.DataFrame) -> pd.DataFrame:
    """A7 列 → 统一 schema."""
    out = df.copy()
    out["model_class"] = "univ2x_full"
    out["est_lat_4090_ms"] = out["est_lat_e2e_ms"]
    out["est_amota"] = out["est_amota"]  # 已是 LGB 预测
    out["est_orin_fp16_ms"] = np.nan  # A7 时 M2 f 还没拟合, 后续可重算
    out["est_orin_int8_ms"] = np.nan
    out["est_params_M"] = np.nan
    return out


def normalize_m47(df: pd.DataFrame) -> pd.DataFrame:
    """M4.7 列 → 统一 schema."""
    out = df.copy()
    out["est_lat_4090_ms"] = out["est_lat_4090_pytorch_ms"]
    out["est_amota"] = np.nan  # 待 M4.5 baseline + 抽样实测
    return out


def normalize_m56(df: pd.DataFrame) -> pd.DataFrame:
    """M5.6 列 → 统一 schema."""
    out = df.copy()
    out["est_lat_4090_ms"] = out["est_lat_4090_pytorch_ms"]
    out["est_amota"] = np.nan  # 待 M5.7 抽样实测 + v5.1 预测
    return out


def main() -> None:
    print("=" * 60)
    print("Phase 2.5 — 三模型联合 Pareto 骨架")
    print("=" * 60)

    a7 = pd.read_csv(A7_CSV)
    m47 = pd.read_csv(M47_CSV)
    m56 = pd.read_csv(M56_CSV)

    print(f"\n输入:")
    print(f"  A7  (univ2x_full):        {len(a7)} 候选, Pareto {a7['is_pareto'].sum()} 点")
    print(f"  M4.7 (pyramid_fusion):    {len(m47)} 候选, Pareto {m47['is_pareto'].sum()} 点")
    print(f"  M5.6 (uniad_tiny_variant): {len(m56)} 候选, Pareto {m56['is_pareto'].sum()} 点")

    a7n = normalize_a7(a7)
    m47n = normalize_m47(m47)
    m56n = normalize_m56(m56)

    # 统一列 (取交集 + 标准化)
    common_cols = [
        "config_id", "model_class", "is_pareto",
        "est_lat_4090_ms", "est_orin_fp16_ms", "est_orin_int8_ms",
        "est_amota", "est_params_M",
        "prune_object",
    ]
    # 加各模块的 prune/quant 配置
    for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
        for p in ("prune_rate", "prune_criterion", "q_bits", "q_granularity", "q_object", "d_routing"):
            common_cols.append(f"{p}__{m}")

    def select(df: pd.DataFrame) -> pd.DataFrame:
        avail = [c for c in common_cols if c in df.columns]
        return df[avail]

    df = pd.concat([select(a7n), select(m47n), select(m56n)], ignore_index=True)
    print(f"\n联合表: {len(df)} 行 × {len(df.columns)} 列")

    # 摘要统计
    print(f"\n=== latency 跨度 (4090 PyTorch) ===")
    for mc in df["model_class"].unique():
        sub = df[df["model_class"] == mc]
        if sub["est_lat_4090_ms"].notna().any():
            print(f"  {mc:<22}  "
                  f"min={sub['est_lat_4090_ms'].min():>8.2f}  "
                  f"max={sub['est_lat_4090_ms'].max():>8.2f}  "
                  f"mean={sub['est_lat_4090_ms'].mean():>8.2f}  ms")

    print(f"\n=== amota 维度状态 ===")
    for mc in df["model_class"].unique():
        sub = df[df["model_class"] == mc]
        n_amota = sub["est_amota"].notna().sum()
        if n_amota > 0:
            print(f"  {mc:<22}  amota 已估: {n_amota}/{len(sub)}  "
                  f"range [{sub['est_amota'].min():.3f}, {sub['est_amota'].max():.3f}]")
        else:
            print(f"  {mc:<22}  amota TBD (待 M4.5 / M5.7)")

    print(f"\n=== Pareto 前沿 (各模型内部) ===")
    for mc in df["model_class"].unique():
        sub = df[(df["model_class"] == mc) & (df["is_pareto"] == True)]
        print(f"  {mc:<22}  {len(sub)} Pareto 点")

    # 输出
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n✅ Wrote {OUT_CSV}")
    print(f"✅ Wrote {OUT_PARQUET}")

    # 提示后续工作
    print(f"\n=== 后续 (M4.5 / M5.7 完成后) ===")
    print(f"  1. M4.5 baseline AP 回填 → pyramid_baseline 行 amota anchor")
    print(f"  2. 抽样 5-10 个 Pareto-optimal 候选实测 AP (M4.5/M5.7)")
    print(f"  3. v5.1 LightGBM 给 50 候选预测 amota (训练时加入 pyramid + uniad_tiny anchor)")
    print(f"  4. 重跑此脚本, est_amota 填入, 三模型 Pareto 图可画")


if __name__ == "__main__":
    main()
