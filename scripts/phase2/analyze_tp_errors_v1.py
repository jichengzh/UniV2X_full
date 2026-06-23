"""Analysis of tp_errors_v1.csv: compare AP70 vs mATE/mASE/mAOE signals.

Shows whether TP geometric errors have more monotonic signal across pruning rate
and quantization than AP70 (which is "flat" in the AP plateau region).

Usage:
    python scripts/phase2/analyze_tp_errors_v1.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

def main():
    csv_path = REPO_ROOT / "results/tp_errors_v1.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run eval_tp_errors_v1.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(df.to_string(index=False))
    print()

    # 1. Check monotonicity across pruning rate (FP16 only)
    fp16 = df[df["precision"] == "fp16"].copy()
    anchor_order = ["base", "pruned25", "pruned50", "pruned75"]
    fp16["anchor_ord"] = fp16["anchor"].map({a: i for i, a in enumerate(anchor_order)})
    fp16 = fp16.sort_values("anchor_ord")

    print("=" * 60)
    print("FP16 across pruning rate (monotonicity check)")
    print("=" * 60)
    for col in ["ap70", "mATE", "mASE", "mAOE"]:
        vals = fp16[col].values
        diffs = np.diff(vals)
        n_increases = np.sum(diffs > 0)
        n_decreases = np.sum(diffs < 0)
        trend = "↑ monotone" if n_increases == len(diffs) else \
                "↓ monotone" if n_decreases == len(diffs) else "non-monotone"
        span = float(np.max(vals) - np.min(vals))
        print(f"  {col:<8}: {trend:<15} span={span:.4f}  "
              f"vals=[{', '.join(f'{v:.4f}' for v in vals)}]")

    # 2. INT8 effect per anchor (INT8 - FP16)
    print()
    print("=" * 60)
    print("INT8 degradation (INT8 - FP16 per anchor)")
    print("=" * 60)
    merged = df[df["precision"] == "fp16"][["anchor", "ap70", "mATE", "mASE", "mAOE"]].merge(
        df[df["precision"] == "int8"][["anchor", "ap70", "mATE", "mASE", "mAOE"]],
        on="anchor", suffixes=("_fp16", "_int8")
    )
    for col in ["ap70", "mATE", "mASE", "mAOE"]:
        merged[f"delta_{col}"] = merged[f"{col}_int8"] - merged[f"{col}_fp16"]

    for _, row in merged.iterrows():
        print(f"\n  {row['anchor']}:")
        print(f"    AP70:  fp16={row['ap70_fp16']:.4f}  int8={row['ap70_int8']:.4f}  delta={row['delta_ap70']:+.4f}")
        print(f"    mATE:  fp16={row['mATE_fp16']:.4f}m  int8={row['mATE_int8']:.4f}m  delta={row['delta_mATE']:+.4f}m")
        print(f"    mASE:  fp16={row['mASE_fp16']:.4f}  int8={row['mASE_int8']:.4f}  delta={row['delta_mASE']:+.4f}")
        print(f"    mAOE:  fp16={row['mAOE_fp16']:.4f}rad  int8={row['mAOE_int8']:.4f}rad  delta={row['delta_mAOE']:+.4f}rad")

    # 3. Summary: which metric has the highest normalized span?
    print()
    print("=" * 60)
    print("Signal strength: normalized span (max-min)/mean")
    print("=" * 60)
    for col in ["ap70", "mATE", "mASE", "mAOE"]:
        all_vals = df[col].values
        span = np.max(all_vals) - np.min(all_vals)
        mean_val = np.mean(all_vals)
        norm_span = span / mean_val if mean_val > 0 else float("nan")
        print(f"  {col:<8}: span={span:.4f}  mean={mean_val:.4f}  norm_span={norm_span:.3f}")

    # 4. Conclusion
    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    fp16_sorted = fp16.sort_values("anchor_ord")
    ap70_monotone = np.all(np.diff(fp16_sorted["ap70"].values) <= 0)
    mAOE_monotone = np.all(np.diff(fp16_sorted["mAOE"].values) >= 0)
    mATE_monotone = np.all(np.diff(fp16_sorted["mATE"].values) >= 0)
    mASE_monotone = np.all(np.diff(fp16_sorted["mASE"].values) >= 0)

    ap70_span_norm = (fp16_sorted["ap70"].max() - fp16_sorted["ap70"].min()) / fp16_sorted["ap70"].mean()
    mAOE_span_norm = (fp16_sorted["mAOE"].max() - fp16_sorted["mAOE"].min()) / fp16_sorted["mAOE"].mean()

    print(f"  AP70 monotone (↓ with pruning): {ap70_monotone}")
    print(f"  mATE monotone (↑ with pruning): {mATE_monotone}")
    print(f"  mASE monotone (↑ with pruning): {mASE_monotone}")
    print(f"  mAOE monotone (↑ with pruning): {mAOE_monotone}")
    print(f"  AP70 normalized span: {ap70_span_norm:.3f}")
    print(f"  mAOE normalized span: {mAOE_span_norm:.3f}")

    verdict = "SIGNAL FOUND" if (mAOE_monotone or mATE_monotone or mASE_monotone) else "SIGNAL FLAT"
    if mAOE_span_norm > ap70_span_norm * 1.5 or mATE_span_norm > ap70_span_norm * 1.5:
        print(f"\n  VERDICT: {verdict} — TP errors have {mAOE_span_norm/ap70_span_norm:.1f}x more signal than AP70")
    else:
        print(f"\n  VERDICT: {verdict} — TP errors span ≈ AP70 span, consider alternative metrics")

    # Save analysis
    out = {
        "source": str(csv_path),
        "n_configs": len(df),
        "fp16_ap70_vals": fp16_sorted["ap70"].tolist(),
        "fp16_mATE_vals": fp16_sorted["mATE"].tolist(),
        "fp16_mASE_vals": fp16_sorted["mASE"].tolist(),
        "fp16_mAOE_vals": fp16_sorted["mAOE"].tolist(),
        "ap70_monotone": bool(ap70_monotone),
        "mATE_monotone": bool(mATE_monotone),
        "mASE_monotone": bool(mASE_monotone),
        "mAOE_monotone": bool(mAOE_monotone),
        "ap70_norm_span": float(ap70_span_norm),
        "mAOE_norm_span": float(mAOE_span_norm),
    }
    out_path = REPO_ROOT / "results/tp_errors_v1_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved analysis to {out_path}")


if __name__ == "__main__":
    main()
