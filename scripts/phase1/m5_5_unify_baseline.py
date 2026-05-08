"""M5.5: 整合三模型 baseline + Orin latency 估算 → 统一 baseline_4090.parquet.

输入:
  data/baseline_4090.parquet      — 当前 57 行 (无 model_class)
  results/m4_3_pyramid_fusion_4090_latency.csv — 2 行 Pyramid (M4.3)
  results/m2_latency_mapping_f.json — M2 f 函数

变更:
  - 加 model_class 列: 'univ2x_full' (22) / 'uniad_tiny_variant' (35) / 'pyramid_fusion' (2 新增)
  - 加 est_orin_fp16_ms / est_orin_int8_ms / est_orin_uncertainty_ms 列
  - 加 extrapolation_warning 列 (4090 lat > 10ms 标 True, 出 M2 训练范围)
  - 加 m2_f_factor 列 (lat_4090 / 7ms, M2 训练上限)

输出: 替换 data/baseline_4090.parquet + .csv (从 57 → 59 行 × 新增 5 列)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "data/baseline_4090.parquet"
PYRAMID_CSV = ROOT / "results/m4_3_pyramid_fusion_4090_latency.csv"
M2_F_JSON = ROOT / "results/m2_latency_mapping_f.json"

# source → model_class 映射 (M5.5 audit 结果)
# univ2x_full: R101 + DCNv2 + 200x200 BEV (1.1/1.2 章节实验, PyTorch FP32 e2e ~5640ms)
# uniad_tiny_variant: R50 + 50x50 BEV, 无 DCN (stage5_v3 plan_b_active + 1.2_P1_FFN + 1.3_d, ~500ms)
SOURCE_TO_MODEL_CLASS = {
    "1.1_quant": "univ2x_full",
    "1.2_prune": "univ2x_full",
    "1.2_prune_ft": "univ2x_full",
    "1.2_pareto": "univ2x_full",
    "1.2_joint": "univ2x_full",
    "1.2_P1_FFN": "uniad_tiny_variant",
    "1.3_d": "uniad_tiny_variant",
    "plan_b_active": "uniad_tiny_variant",
}

EXTRAPOLATION_THRESHOLD_MS = 10.0  # M2 训练上限 7ms, 超 10ms 标 warning


def main() -> None:
    print("=" * 60)
    print("M5.5 — 统一 baseline_4090.parquet (model_class + Orin 估算)")
    print("=" * 60)

    df = pd.read_parquet(BASELINE)
    print(f"✅ Loaded baseline_4090: {len(df)} rows × {len(df.columns)} cols")

    # 1. 加 model_class
    df["model_class"] = df["source"].map(SOURCE_TO_MODEL_CLASS).fillna("unknown")
    print(f"\n=== model_class 分布 ===")
    print(df["model_class"].value_counts().to_string())

    # 2. 加 Pyramid Fusion (M4.3 输出)
    pyramid = pd.read_csv(PYRAMID_CSV)
    print(f"\n✅ Loaded {PYRAMID_CSV.name}: {len(pyramid)} rows (Pyramid Fusion)")
    pyramid_rows = []
    for _, r in pyramid.iterrows():
        row = {col: np.nan for col in df.columns}
        row["config_id"] = f"pyramid_{r['precision']}_baseline"
        row["source"] = "m4_3_pyramid_baseline"
        row["model_class"] = "pyramid_fusion"
        row["is_real_measured"] = True
        row["notes"] = f"M4.3 PyramidFusion {r['precision']} (3.76M params, lidar_pyramid config)"
        row["amota"] = np.nan  # Pyramid 用 OPV2V AP30/AP50, 不是 amota; 留空
        row["lat_e2e_ms"] = float(r["mean_ms"])
        row["params_after_M"] = float(r["n_params_M"])
        # B1/B2/D 都是 baseline (不剪/不量化, 但 q_bits 反映 PyTorch precision)
        row["prune_object"] = "none"
        for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
            row[f"prune_rate__{m}"] = 0.0
            row[f"prune_criterion__{m}"] = "none"
            row[f"q_granularity__{m}"] = "none"
            row[f"q_object__{m}"] = "none"
            row[f"d_routing__{m}"] = "GPU"
            row[f"q_bits__{m}"] = "FP16" if r["precision"] == "fp16" else "FP32"
        row["d_runtime"] = "pytorch_fp16" if r["precision"] == "fp16" else "pytorch_fp32"
        row["d_pipelined"] = 0
        row["d_temporal_cache_int8"] = 0
        pyramid_rows.append(row)

    pyramid_df = pd.DataFrame(pyramid_rows)
    df = pd.concat([df, pyramid_df], ignore_index=True)
    print(f"   Combined: {len(df)} rows")

    # 3. 应用 M2 f 函数计算 Orin 估算 (只对有 lat_e2e_ms 的行)
    with open(M2_F_JSON, "r") as f:
        fits = json.load(f)["fits"]
    fp16 = fits["f_fp16_from_4090_fp32"]
    int8 = fits["f_int8_from_4090_fp32"]

    has_lat = df["lat_e2e_ms"].notna()
    lat = df.loc[has_lat, "lat_e2e_ms"].to_numpy()

    df["est_orin_fp16_ms"] = np.nan
    df["est_orin_int8_ms"] = np.nan
    df["est_orin_uncertainty_ms"] = np.nan
    df["extrapolation_warning"] = False
    df["m2_f_factor"] = np.nan

    df.loc[has_lat, "est_orin_fp16_ms"] = fp16["intercept_a"] + fp16["slope_b"] * lat
    df.loc[has_lat, "est_orin_int8_ms"] = int8["intercept_a"] + int8["slope_b"] * lat
    df.loc[has_lat, "est_orin_uncertainty_ms"] = fp16["stderr"] * np.abs(lat)
    df.loc[has_lat, "extrapolation_warning"] = lat > EXTRAPOLATION_THRESHOLD_MS
    df.loc[has_lat, "m2_f_factor"] = lat / 7.0  # M2 训练上限

    # 4. 重排列顺序 (重要列前置)
    front = [
        "config_id", "model_class", "source", "is_real_measured", "notes",
        "amota", "amotp", "mAP", "lat_e2e_ms",
        "est_orin_fp16_ms", "est_orin_int8_ms",
        "est_orin_uncertainty_ms", "extrapolation_warning", "m2_f_factor",
        "params_after_M", "mem_peak_mb",
    ]
    rest = [c for c in df.columns if c not in front]
    df = df[[c for c in front if c in df.columns] + rest]

    # 5. 保存
    df.to_parquet(BASELINE, index=False)
    df.to_csv(BASELINE.with_suffix(".csv"), index=False)
    print(f"\n✅ Wrote {BASELINE} ({len(df)} rows × {len(df.columns)} cols)")
    print(f"✅ Wrote {BASELINE.with_suffix('.csv')}")

    # 6. 摘要报告
    print(f"\n=== 整合后 baseline 分组 ===")
    print(df.groupby(["model_class", "source"]).size().to_string())

    print(f"\n=== Orin latency 估算覆盖 ===")
    print(f"  有 4090 lat 的行: {has_lat.sum()} / {len(df)}")
    print(f"  有 Orin 估算的行: {df['est_orin_fp16_ms'].notna().sum()} / {len(df)}")
    print(f"  extrapolation_warning: {df['extrapolation_warning'].sum()} / {len(df)} 行 (lat > 10ms)")
    print()
    print(f"=== 各 model_class 的 latency 范围 ===")
    for mc in df["model_class"].unique():
        sub = df[df["model_class"] == mc]
        if sub["lat_e2e_ms"].notna().any():
            print(f"  {mc:<22}  4090 lat: {sub['lat_e2e_ms'].min():>7.1f} - {sub['lat_e2e_ms'].max():>7.1f} ms"
                  f"  | Orin FP16 估算: {sub['est_orin_fp16_ms'].min():>7.1f} - {sub['est_orin_fp16_ms'].max():>7.1f} ms")


if __name__ == "__main__":
    main()
