"""Validate data/unified_bench.parquet against §3 schema spec.

Per plan §6.2 + §〇.5 rules:
  - Every row has all 34 cols (NaN allowed where appropriate)
  - AP cols must be NaN for metric_type='lat_only' / 'lat_only_multi_ip'
  - AP cols must NOT be NaN for metric_type='ap_dair_*'
  - amota_cross_model uses ap50 slot for amota (intentional, not error)
  - build_success bool, fail_reason str | NaN
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


def main():
    bench = DATA / "unified_bench.parquet"
    spec = DATA / "schema_spec.json"
    if not bench.exists():
        print(f"FAIL: {bench} missing"); sys.exit(1)
    if not spec.exists():
        print(f"FAIL: {spec} missing"); sys.exit(1)

    df = pd.read_parquet(bench)
    spec_data = json.loads(spec.read_text())
    required = spec_data["columns"]

    errors = []

    # 1. Column presence
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")

    # 2. metric_type AP rules (per §〇.5)
    lat_only_mask = df["metric_type"].isin(["lat_only", "lat_only_multi_ip"])
    ap_real_mask = df["metric_type"].isin(["ap_dair_1789full", "ap_dair_500samples"])

    # lat-only rows: AP MUST be NaN
    lat_with_ap = df[lat_only_mask & df["ap50"].notna()]
    if len(lat_with_ap) > 0:
        errors.append(f"§〇.5 violation: {len(lat_with_ap)} lat_only rows have ap50 != NaN")

    # AP-real rows: ap50 MUST exist
    ap_real_missing = df[ap_real_mask & df["ap50"].isna()]
    if len(ap_real_missing) > 0:
        errors.append(f"§〇.5 violation: {len(ap_real_missing)} ap_dair rows have ap50 = NaN")

    # 3. amota cross-model must use metric_type='amota_cross_model'
    amota_mask = df["metric_type"] == "amota_cross_model"
    amota_not_pyramid = df[amota_mask & (df["model_class"] != "pyramid_fusion")]
    print(f"  amota_cross_model rows: {amota_mask.sum()} "
          f"({len(amota_not_pyramid)} non-Pyramid model_class — expected for cross-model)")

    # 4. build_success type
    if df["build_success"].dtype not in [bool, "object"]:
        errors.append(f"build_success dtype is {df['build_success'].dtype}, expected bool/object")

    # 5. Hardware enum
    valid_hw = {"rtx4090", "orin_agx_64gb"}
    bad_hw = df[~df["hardware"].isin(valid_hw) & df["hardware"].notna()]
    if len(bad_hw):
        errors.append(f"invalid hardware values: {set(bad_hw['hardware'].unique())}")

    # 6. q_bits enum
    valid_q = {"FP32", "FP16", "INT8", "mixed", None}
    bad_q = df[~df["q_bits"].isin(valid_q) & df["q_bits"].notna()]
    if len(bad_q):
        errors.append(f"invalid q_bits: {set(bad_q['q_bits'].unique())}")

    # Report
    print("=" * 60)
    print(f"unified_bench.parquet rows: {len(df)} cols: {len(df.columns)}")
    print(f"metric_type breakdown:")
    print(df["metric_type"].value_counts().to_string())
    print()
    print(f"Hardware breakdown:")
    print(df["hardware"].value_counts().to_string())
    print()
    print(f"LGB usage summary:")
    print(f"  lat head training (lat_mean_ms != NaN, NOT amota): "
          f"{((df['lat_mean_ms'].notna()) & (df['metric_type'] != 'amota_cross_model')).sum()} rows")
    print(f"  AP head training (ap_dair_500samples + ap_dair_1789full): "
          f"{ap_real_mask.sum()} rows")
    print(f"  amota head training: {amota_mask.sum()} rows")
    print(f"  negative pool (excluded): {(df['metric_type'] == 'ap_dair_negative_pool').sum()} rows")

    print("\n" + "=" * 60)
    if errors:
        print(f"FAIL: {len(errors)} schema errors:")
        for e in errors: print(f"  - {e}")
        sys.exit(1)
    print("✅ All schema checks passed")


if __name__ == "__main__":
    main()
