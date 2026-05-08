"""M5.2: univ2x-tiny 4090 latency 数据整合 + M2 f 估算 Orin AGX latency.

输入:
  data/phase4/stage5_baseline_v3.csv  — 23 configs, 4090 PyTorch FP32 实测
                                         (plan_b_active 18 + 1.2_P1_FFN 5)
  results/m2_latency_mapping_f.json   — M2 拟合的 4090↔Orin 函数

输出:
  data/uniad_tiny_baseline.csv  — 标 source='uniad_tiny_variant' + Orin latency 估算

注意 (extrapolation warning):
  M2 f 函数在 ResNet18-101 (1-7ms range) 上拟合, R²>0.99 但样本范围窄.
  univ2x-tiny e2e latency 在 502-562ms 范围, 是 M2 训练范围的 ~80x 外推.
  线性外推 R² 可信但绝对值有风险, 需 Phase 3 用真实 Orin trtexec 验证.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_V3 = ROOT / "data/phase4/stage5_baseline_v3.csv"
INPUT_F = ROOT / "results/m2_latency_mapping_f.json"
OUTPUT = ROOT / "data/uniad_tiny_baseline.csv"

EXTRAPOLATION_THRESHOLD_MS = 10.0  # 超过 M2 训练范围 7× 标 warning


def main() -> None:
    print("=" * 60)
    print("M5.2 — univ2x-tiny baseline + Orin latency 估算")
    print("=" * 60)

    df = pd.read_csv(INPUT_V3)
    print(f"✅ Loaded {INPUT_V3.name}: {len(df)} configs")
    print(f"   source: {df['source'].value_counts().to_dict()}")
    print(f"   lat_e2e_ms (4090 PyTorch FP32): "
          f"{df['lat_e2e_ms'].min():.1f} - {df['lat_e2e_ms'].max():.1f} ms")

    with open(INPUT_F, "r") as f:
        fits = json.load(f)["fits"]
    fp16 = fits["f_fp16_from_4090_fp32"]
    int8 = fits["f_int8_from_4090_fp32"]
    print(f"✅ Loaded M2 f: f_fp16(R²={fp16['r_squared']:.3f}), f_int8(R²={int8['r_squared']:.3f})")

    # 估算 Orin latency
    lat_4090 = df["lat_e2e_ms"].to_numpy()
    est_orin_fp16 = fp16["intercept_a"] + fp16["slope_b"] * lat_4090
    est_orin_int8 = int8["intercept_a"] + int8["slope_b"] * lat_4090

    # 不确定度: M2 stderr × x (slope 不确定度 × 输入 latency)
    est_orin_fp16_uncertainty = fp16["stderr"] * np.abs(lat_4090)
    est_orin_int8_uncertainty = int8["stderr"] * np.abs(lat_4090)

    # 外推 warning
    extrapolation_warning = lat_4090 > EXTRAPOLATION_THRESHOLD_MS

    # 重命名 source 标记
    df_out = df.copy()
    df_out = df_out.rename(columns={"lat_e2e_ms": "lat_4090_pytorch_fp32_ms"})
    df_out["model_class"] = "uniad_tiny_variant"
    df_out["source_original"] = df_out["source"]  # 保留原 source
    df_out["source"] = "uniad_tiny_variant"
    df_out["est_orin_fp16_ms"] = est_orin_fp16
    df_out["est_orin_fp16_uncertainty_ms"] = est_orin_fp16_uncertainty
    df_out["est_orin_int8_ms"] = est_orin_int8
    df_out["est_orin_int8_uncertainty_ms"] = est_orin_int8_uncertainty
    df_out["extrapolation_warning"] = extrapolation_warning
    df_out["m2_f_extrapolation_factor"] = lat_4090 / 7.0  # M2 训练上限 ~7ms

    # 列顺序 (重要列优先)
    front = [
        "config_id", "source", "source_original", "model_class",
        "amota", "amotp", "mAP", "NDS",
        "lat_4090_pytorch_fp32_ms",
        "est_orin_fp16_ms", "est_orin_fp16_uncertainty_ms",
        "est_orin_int8_ms", "est_orin_int8_uncertainty_ms",
        "extrapolation_warning", "m2_f_extrapolation_factor",
        "params_after_M",
    ]
    rest = [c for c in df_out.columns if c not in front]
    df_out = df_out[[c for c in front if c in df_out.columns] + rest]

    OUTPUT.parent.mkdir(exist_ok=True)
    df_out.to_csv(OUTPUT, index=False)
    print(f"\n✅ Wrote {OUTPUT}")
    print(f"   shape: {df_out.shape}")

    # 摘要
    print(f"\n=== Orin latency 估算摘要 ===")
    print(f"  est_orin_fp16:  {df_out['est_orin_fp16_ms'].min():.1f} - {df_out['est_orin_fp16_ms'].max():.1f} ms"
          f" (mean {df_out['est_orin_fp16_ms'].mean():.1f}, ±{df_out['est_orin_fp16_uncertainty_ms'].mean():.1f})")
    print(f"  est_orin_int8:  {df_out['est_orin_int8_ms'].min():.1f} - {df_out['est_orin_int8_ms'].max():.1f} ms"
          f" (mean {df_out['est_orin_int8_ms'].mean():.1f}, ±{df_out['est_orin_int8_uncertainty_ms'].mean():.1f})")
    print(f"  extrapolation_warning: {df_out['extrapolation_warning'].sum()} / {len(df_out)} rows")
    print(f"  m2_f_extrapolation_factor: avg {df_out['m2_f_extrapolation_factor'].mean():.0f}× (M2 训练 1-7ms, 当前 502-562ms)")

    print(f"\n=== Top-5 + Bottom-5 (按 est_orin_int8_ms ↑) ===")
    sorted_df = df_out.sort_values("est_orin_int8_ms")
    cols = ["config_id", "source_original", "amota", "lat_4090_pytorch_fp32_ms",
            "est_orin_fp16_ms", "est_orin_int8_ms"]
    print("Top 5:")
    print(sorted_df[cols].head(5).to_string(index=False))
    print("\nBottom 5:")
    print(sorted_df[cols].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
