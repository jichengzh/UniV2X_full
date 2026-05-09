"""M4.6.0 post: 把完整 Pyramid e2e 实测结果合入 baseline_4090.parquet.

输入: results/m4_6_0_pyramid_eval.csv (FP32 + FP16 两行)
输出: data/baseline_4090.parquet 加 2 新行 (source='m4_6_0_pyramid_full_e2e')
      不动现有 pyramid_baseline 行 (那是 PyramidFusion 子模块, 保留参考)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "data/baseline_4090.parquet"
EVAL_CSV = REPO_ROOT / "results/m4_6_0_pyramid_eval.csv"


def main() -> None:
    df = pd.read_parquet(BASELINE)
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"baseline {len(df)} rows | eval {len(eval_df)} rows")

    new_rows = []
    for _, r in eval_df.iterrows():
        prec = r["precision"]
        row = {col: np.nan for col in df.columns}
        row["config_id"] = f"m4_6_0_pyramid_full_e2e_{prec}"
        row["source"] = "m4_6_0_pyramid_full_e2e"
        row["model_class"] = "pyramid_fusion"
        row["is_real_measured"] = True
        row["notes"] = (f"M4.6.0 完整 e2e 实测 OPV2V test 2170 samples; "
                       f"AP30={r['AP30']:.4f}/AP50={r['AP50']:.4f}/AP70={r['AP70']:.4f}; "
                       f"4090 PyTorch {prec} (autocast for fp16)")
        row["amota"] = float(r["AP50"])  # 用 AP50 作 amota anchor
        row["lat_e2e_ms"] = float(r["lat_p50_ms"])  # 用 p50 (median, 代表性)
        row["params_after_M"] = 5.50  # 完整 Pyramid_m1_base
        row["prune_object"] = "none"
        for m in ("backbone", "encoder", "decoder", "heads", "v2x_comm"):
            row[f"prune_rate__{m}"] = 0.0
            row[f"prune_criterion__{m}"] = "none"
            row[f"q_granularity__{m}"] = "none"
            row[f"q_object__{m}"] = "none"
            row[f"d_routing__{m}"] = "GPU"
            row[f"q_bits__{m}"] = "FP32" if prec == "fp32" else "FP16"
        row["d_runtime"] = "pytorch_fp32" if prec == "fp32" else "pytorch_fp16"
        row["d_pipelined"] = 0
        row["d_temporal_cache_int8"] = 0
        new_rows.append(row)

    new_df = pd.DataFrame(new_rows)
    out = pd.concat([df, new_df], ignore_index=True)
    out.to_parquet(BASELINE, index=False)
    out.to_csv(BASELINE.with_suffix(".csv"), index=False)
    print(f"✅ Wrote {BASELINE} ({len(out)} rows, +2 from M4.6.0)")

    # 摘要
    print(f"\n=== Pyramid baseline data points ===")
    py = out[out["model_class"] == "pyramid_fusion"]
    print(py[["config_id", "source", "lat_e2e_ms", "amota", "params_after_M"]].to_string(index=False))


if __name__ == "__main__":
    main()
