"""把 M4.6.1+2+3 的 16 个真实测试点整合到 baseline_4090.parquet, 给 LGB v6 训练用.

输入:
  results/m4_6_1_pyramid_pruning.csv (7 行单模块 prune)
  results/m4_6_2_pyramid_multimodule.csv (4 行多模块 prune)
  results/m4_6_3_pyramid_pareto_validation.csv (5 行 framework Pareto validation)

输出:
  data/baseline_4090.parquet 加 16 行 (source='m4_6_1_pruning' / 'm4_6_2_multimodule' / 'm4_6_3_pareto_validation')
  全部 model_class='pyramid_fusion', amota=AP50, lat=p50

注意:
  - mask-based pruning lat 不真减, 但 AP 是真实测的, 适合给 amota 模型训练
  - latency 字段记录但不必给 LGB latency 模型, 因为 mask-based 不是真 latency 信号
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "data/baseline_4090.parquet"
M4_6_1 = REPO_ROOT / "results/m4_6_1_pyramid_pruning.csv"
M4_6_2 = REPO_ROOT / "results/m4_6_2_pyramid_multimodule.csv"
M4_6_3 = REPO_ROOT / "results/m4_6_3_pyramid_pareto_validation.csv"


def m4_6_1_to_rows(df: pd.DataFrame, baseline_cols: list) -> list[dict]:
    rows = []
    for i, r in df.iterrows():
        row = {col: np.nan for col in baseline_cols}
        rate = float(r["prune_rate"])
        crit = r["criterion"] if rate > 0 else "none"
        row["config_id"] = f"m4_6_1_pyramid_{crit}_p{int(rate*100):02d}"
        row["source"] = "m4_6_1_pruning"
        row["model_class"] = "pyramid_fusion"
        row["is_real_measured"] = True
        row["notes"] = (f"M4.6.1 single-module ({crit}) p={rate:.2f} mask-based + BN recal 50; "
                       f"AP30/50/70={r['AP30']:.4f}/{r['AP50']:.4f}/{r['AP70']:.4f}")
        row["amota"] = float(r["AP50"])
        row["lat_e2e_ms"] = float(r["lat_p50_ms"])
        row["params_after_M"] = 5.50  # 完整 Pyramid_m1_base (mask 不真减 size)
        # 5 module: M4.6.1 只剪 pyramid_backbone+shrink_conv (映射到 framework decoder + v2x_comm)
        row["prune_object"] = "channel" if rate > 0 else "none"
        for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
            # mask-based pruning 在 framework 5 模块中近似只影响 decoder + v2x_comm
            if m in ("decoder", "v2x_comm"):
                row[f"prune_rate__{m}"] = rate
                row[f"prune_criterion__{m}"] = crit if rate > 0 else "none"
            else:
                row[f"prune_rate__{m}"] = 0.0
                row[f"prune_criterion__{m}"] = "none"
            row[f"q_bits__{m}"] = "FP32"
            row[f"q_granularity__{m}"] = "none"
            row[f"q_object__{m}"] = "none"
            row[f"d_routing__{m}"] = "GPU"
        row["d_runtime"] = "pytorch_fp32"
        row["d_pipelined"] = 0
        row["d_temporal_cache_int8"] = 0
        rows.append(row)
    return rows


def m4_6_2_to_rows(df: pd.DataFrame, baseline_cols: list) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        row = {col: np.nan for col in baseline_cols}
        row["config_id"] = f"m4_6_2_{r['config_name']}"
        row["source"] = "m4_6_2_multimodule"
        row["model_class"] = "pyramid_fusion"
        row["is_real_measured"] = True
        row["notes"] = (f"M4.6.2 multi-module ({r['config_name']}) mask + BN recal 50; "
                       f"AP30/50/70={r['AP30']:.4f}/{r['AP50']:.4f}/{r['AP70']:.4f}")
        row["amota"] = float(r["AP50"])
        row["lat_e2e_ms"] = float(r["lat_p50_ms"])
        row["params_after_M"] = 5.50
        any_prune = any(float(r.get(f"prune_rate__{m}", 0)) > 0 for m in ("backbone","encoder","decoder","heads","v2x_comm"))
        row["prune_object"] = "channel" if any_prune else "none"
        for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
            rate = float(r.get(f"prune_rate__{m}", 0) or 0)
            row[f"prune_rate__{m}"] = rate
            row[f"prune_criterion__{m}"] = "L1" if rate > 0 else "none"
            row[f"q_bits__{m}"] = "FP32"
            row[f"q_granularity__{m}"] = "none"
            row[f"q_object__{m}"] = "none"
            row[f"d_routing__{m}"] = "GPU"
        row["d_runtime"] = "pytorch_fp32"
        row["d_pipelined"] = 0
        row["d_temporal_cache_int8"] = 0
        rows.append(row)
    return rows


def m4_6_3_to_rows(df: pd.DataFrame, baseline_cols: list) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        row = {col: np.nan for col in baseline_cols}
        row["config_id"] = f"m4_6_3_{r['config_id']}"
        row["source"] = "m4_6_3_pareto_validation"
        row["model_class"] = "pyramid_fusion"
        row["is_real_measured"] = True
        row["notes"] = (f"M4.6.3 framework Pareto candidate {r['config_id']} "
                       f"(predicted_pareto={r['is_pareto_predicted']}); "
                       f"AP30/50/70={r['AP30']:.4f}/{r['AP50']:.4f}/{r['AP70']:.4f}; "
                       f"prec_tag={r['prec_tag']}")
        row["amota"] = float(r["AP50"])
        row["lat_e2e_ms"] = float(r["lat_p50_ms"])
        row["params_after_M"] = 5.50
        row["prune_object"] = r["prune_object"]
        for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
            rate = float(r.get(f"prune_rate__{m}", 0) or 0)
            row[f"prune_rate__{m}"] = rate
            row[f"prune_criterion__{m}"] = "L1" if rate > 0 else "none"
            # int8_proxy_fp16 → FP16 (因 PyTorch GPU 不支持 INT8)
            row[f"q_bits__{m}"] = "FP16" if "fp16" in r["prec_tag"] else "FP32"
            row[f"q_granularity__{m}"] = "none"
            row[f"q_object__{m}"] = "none"
            row[f"d_routing__{m}"] = "GPU"
        row["d_runtime"] = "pytorch_fp16" if "fp16" in r["prec_tag"] else "pytorch_fp32"
        row["d_pipelined"] = 0
        row["d_temporal_cache_int8"] = 0
        rows.append(row)
    return rows


def main() -> None:
    print("=" * 60)
    print("M4.6.1+2+3 整合到 baseline_4090.parquet")
    print("=" * 60)

    df = pd.read_parquet(BASELINE)
    print(f"baseline 当前 {len(df)} rows")

    # 删旧的 m4_6 行 (避免重复)
    before = len(df)
    df = df[~df["source"].isin(["m4_6_1_pruning", "m4_6_2_multimodule", "m4_6_3_pareto_validation"])].reset_index(drop=True)
    if before != len(df):
        print(f"  清掉旧 m4_6_* 行: {before - len(df)} 行")

    cols = list(df.columns)
    new_rows = []
    for csv, fn, name in [(M4_6_1, m4_6_1_to_rows, "m4_6_1"),
                          (M4_6_2, m4_6_2_to_rows, "m4_6_2"),
                          (M4_6_3, m4_6_3_to_rows, "m4_6_3")]:
        if csv.exists():
            sub = pd.read_csv(csv)
            rows = fn(sub, cols)
            print(f"  {name}: 加 {len(rows)} 行")
            new_rows.extend(rows)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        out = pd.concat([df, new_df], ignore_index=True)
        out.to_parquet(BASELINE, index=False)
        out.to_csv(BASELINE.with_suffix(".csv"), index=False)
        print(f"\n✅ Wrote {BASELINE} ({len(out)} rows, +{len(new_rows)})")

        py = out[out["model_class"] == "pyramid_fusion"]
        print(f"\n=== pyramid_fusion 训练锚点 ({len(py)} rows) ===")
        print(py[["config_id", "source", "amota", "lat_e2e_ms", "prune_object"]].to_string(index=False))


if __name__ == "__main__":
    main()
