"""Bench v1 view — 14 figure report on e2e_bench_v1.csv (4704 anchor).

Adapter around `dataset_stats_report.py`: loads the production bench dataset
(21 triplet x 7 Q x 32 D, no FT axis) and writes the same 14-figure suite into
stats_v3_bench/.

Key differences from the plan-v3 stats:
  - All 4704 anchors are FT-equilibrium (each triplet's own bestval ckpt,
    epoch 23-47, varying per triplet — see e2e_bench_v1_sanity_report.md).
  - D has 32 real levels (vs plan v3's 1), so it stays as a feature.
  - No noise study attached (E1 borrows plan v3 sigma_noise as a reference
    only, since bench triplets differ from T_g8_p97).

Run:
    python scripts/phase2/dataset_stats_report_bench.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import dataset_stats_report as base  # noqa: E402  (sibling script)

REPO = Path("/home/jichengzhi/UniV2X")
BENCH_CSV = REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv"
DEFAULT_OUT = REPO / "paper_learning/2. AAAI最终故事/data/stats_v3_bench"

# Sentinel FT level so DEPLOYMENT_FT filter passes (we don't filter by FT).
FT_SENTINEL = 8

# 21 triplet -> (s0,s1,s2) lookup, from e2e_bench_v1_schema.md §1 + Track A v2.
TRIPLET_PLANES = {
    "T1_base":        (64, 128, 256),
    "T2_p25":         (48,  96, 192),
    "T3_p37":         (40,  80, 160),
    "T4_p50":         (32,  64, 128),
    "T5_p62":         (24,  56, 128),
    "T6_p75":         (16,  32,  64),
    "T7_wide_shallow": (48, 64, 128),
    "T8_narrow_deep": (24,  48, 192),
    "T10_p11":        (16, 128, 256),
    "T11_p14":        (64,  64, 256),
    "T12_p21":        (32,  64, 256),
    "T13_p29":        (32,  32, 256),
    "T14_p36":        (16,  16, 256),
    "T15_p39":        (16, 128, 128),
    "T16_p54":        (16,  64, 128),
    "T17_p57":        (32,  32, 128),
    "T18_p64":        (16,  16, 128),
    "T19_p71":        (32,  32,  64),
    "T20_p79":        (16,  16,  64),
    "T21_p82":        (16,  32,  32),
    "T22_p89":        (16,  16,  16),
}


def load_bench(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    for c in ["ap30", "ap50", "ap70", "throughput_fps", "stage0_planes",
              "stage1_planes", "stage2_planes"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    rows = []
    base_total = 64 + 128 + 256
    for _, r in raw.iterrows():
        # Triplet column already encodes the canonical short name.
        trip = r["triplet"]
        planes = TRIPLET_PLANES.get(trip)
        if planes is None:
            # Fall back to csv stage*_planes (Track A v2 anchors may carry
            # explicit values).
            planes = (int(r["stage0_planes"]) if pd.notna(r["stage0_planes"]) else 0,
                      int(r["stage1_planes"]) if pd.notna(r["stage1_planes"]) else 0,
                      int(r["stage2_planes"]) if pd.notna(r["stage2_planes"]) else 0)
        s0, s1, s2 = planes
        # Strip Q_ prefix to align with plan-v3 nomenclature.
        q = r["q_tag"][2:] if isinstance(r["q_tag"], str) and r["q_tag"].startswith("Q_") else r["q_tag"]
        lat = (1000.0 / r["throughput_fps"]) if pd.notna(r["throughput_fps"]) and r["throughput_fps"] > 0 else None
        rows.append({
            "triplet": trip,
            "ft": FT_SENTINEL,           # sentinel; not a real search axis
            "q": q,
            "d": r["d_tag"],
            "seed": -1,
            "ap50": float(r["ap50"]) if pd.notna(r["ap50"]) else None,
            "lat_p50_ms": float(lat) if lat is not None else None,
            "src": "e2e_bench_v1",
            "planes_s1": int(s0),
            "planes_s2": int(s1),
            "planes_s3": int(s2),
            "total_prune_pct": 1.0 - (s0 + s1 + s2) / base_total,
            "build_success": bool(r["build_success"]) if pd.notna(r["build_success"]) else False,
        })
    return pd.DataFrame(rows)


def bench_featurize(df: pd.DataFrame):
    """Override base.featurize: keep D one-hot (32 levels), drop FT.

    FT is a sentinel constant here, so it would carry zero variance.
    D *does* vary across 32 levels (5 tactic x 4 ws x 3 BL = 60 cells, of which
    32 are populated) and is a legitimate feature.
    """
    sub = df.dropna(subset=["ap50"]).reset_index(drop=True)
    feat = sub[["planes_s1", "planes_s2", "planes_s3",
                "total_prune_pct"]].copy()
    q_oh = pd.get_dummies(sub["q"], prefix="q")
    d_oh = pd.get_dummies(sub["d"], prefix="d")
    X = pd.concat([feat, q_oh, d_oh], axis=1).fillna(0)
    y = sub["ap50"].values.astype(float)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BENCH_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--v3-noise", default="/tmp/a11_noise/noise.json",
                    help="Path to plan v3 noise.json for E1 reference panel")
    args = ap.parse_args()

    df = load_bench(Path(args.csv))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Wire base module to our paths / settings.
    base.OUT = out
    base.DEPLOYMENT_MODE = True       # drop FT axis from A1/A3/B3 panels
    base.DEPLOYMENT_FT = {FT_SENTINEL}  # allow our sentinel through filters
    base.featurize = bench_featurize    # keep D as feature

    df.to_csv(out / "all_anchors.csv", index=False)
    print(f"[bench-stats] loaded N={len(df)} anchors, "
          f"AP avail={df['ap50'].notna().sum()}, "
          f"lat avail={df['lat_p50_ms'].notna().sum()}")
    print(f"[bench-stats] out_dir={out}")
    print(f"[bench-stats] triplet={df['triplet'].nunique()}, "
          f"q={df['q'].nunique()}, d={df['d'].nunique()}")

    results = {}
    print("[stats] A1 marginal coverage"); results["A1"] = base.fig_A1_marginal_coverage(df)
    print("[stats] A2/A3 grid heatmaps"); results["A2_A3"] = base.fig_A2_A3_grids(df)
    print("[stats] A4 PCA hull");         results["A4"] = base.fig_A4_hull_pca2d(df)
    print("[stats] B1 target hist");      results["B1"] = base.fig_B1_target_hist(df)
    print("[stats] B2 class balance");    results["B2"] = base.fig_B2_class_balance(df)
    print("[stats] B3 axis boxplots");    results["B3"] = base.fig_B3_by_axis_box(df)
    print("[stats] C1 MI/Spearman");      results["C1"] = base.fig_C1_mi_spearman(df)
    print("[stats] C2 corr matrix");      results["C2"] = base.fig_C2_corr_matrix(df)
    print("[stats] C3 interaction H");    results["C3"] = base.fig_C3_interaction_h(df)
    print("[stats] D1 5-fold CV");        results["D1"] = base.fig_D1_cv_r2(df)
    print("[stats] D2 learning curve");   results["D2"] = base.fig_D2_learning_curve(df)
    print("[stats] D3 OOD");              results["D3"] = base.fig_D3_ood(df)
    print("[stats] E1 noise floor (plan v3 reference)")
    results["E1"] = base.fig_E1_noise_floor(df, Path(args.v3_noise))

    import json
    (out / "stats_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    base.write_report(df, results)


if __name__ == "__main__":
    main()
