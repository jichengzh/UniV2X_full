"""Plan 4 Phase D — 合并 Phase A 119 + Stage 1 96 = 215 anchor 出 14 图.

复用 dataset_stats_report.py 的 fig_* 函数, 配合:
  - DEPLOYMENT_MODE = True (FT=8 锁定 不在特征里)
  - GroupKFold by triplet (5-fold CV 协议)
  - 不含 D 维度 (AP 主表 D=D1 固定)

Output: paper_learning/2. AAAI最终故事/data/stats_v3_plan4/
  - 14 PNG (01-14)
  - all_anchors.csv (215 行)
  - dataset_v4_stats.md (auto-gen)
  - stats_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import dataset_stats_report as base

REPO = Path("/home/jichengzhi/UniV2X")
PHASE_A_CSV = Path("/tmp/plan4_phaseA/phase_a_anchors.csv")
STAGE_1_CSV = Path("/tmp/plan4_phaseC_stage1/stage1_mix.csv")
DEFAULT_OUT = REPO / "paper_learning/2. AAAI最终故事/data/stats_v3_plan4"

FT_SENTINEL = 8  # FT 锁定 8


def load_plan4_anchors() -> pd.DataFrame:
    """Combine Phase A + Stage 1, normalize to dataset_stats_report schema."""
    df_a = pd.read_csv(PHASE_A_CSV)
    df_s1 = pd.read_csv(STAGE_1_CSV)
    df = pd.concat([df_a, df_s1], ignore_index=True)
    for c in ["ap50", "planes_s0", "planes_s1", "planes_s2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ap50"]).reset_index(drop=True)

    rows = []
    base_total = 64 + 128 + 256
    for _, r in df.iterrows():
        s0 = int(r["planes_s0"]); s1 = int(r["planes_s1"]); s2 = int(r["planes_s2"])
        rows.append({
            "triplet": r["triplet"],
            "ft": FT_SENTINEL,
            "q": r["q"],
            "d": r["d"],
            "seed": -1,
            "ap50": float(r["ap50"]),
            "lat_p50_ms": None,
            "src": "plan4_phaseA_phaseCstage1",
            "planes_s1": s0,        # rename to dataset_stats_report schema
            "planes_s2": s1,        # (script uses planes_s1/s2/s3 starting at stage0)
            "planes_s3": s2,
            "total_prune_pct": 1.0 - (s0 + s1 + s2) / base_total,
            "build_success": True,
        })
    return pd.DataFrame(rows)


def plan4_featurize(df: pd.DataFrame):
    """Drop FT (锁定) + D (单档无信号), keep planes + Q one-hot."""
    sub = df.dropna(subset=["ap50"]).reset_index(drop=True)
    feat = sub[["planes_s1", "planes_s2", "planes_s3",
                "total_prune_pct"]].copy()
    q_oh = pd.get_dummies(sub["q"], prefix="q")
    X = pd.concat([feat, q_oh], axis=1).fillna(0)
    y = sub["ap50"].values.astype(float)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    df = load_plan4_anchors()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Wire base module
    base.OUT = out
    base.DEPLOYMENT_MODE = True
    base.DEPLOYMENT_FT = {FT_SENTINEL}
    base.featurize = plan4_featurize

    df.to_csv(out / "all_anchors.csv", index=False)
    print(f"[plan4 D] N={len(df)} anchors, triplet={df['triplet'].nunique()}, "
          f"q={df['q'].nunique()}, d={df['d'].nunique()}")
    print(f"[plan4 D] out_dir={out}")

    results = {}
    print("[stats] A1 marginal coverage")
    results["A1"] = base.fig_A1_marginal_coverage(df)
    print("[stats] A2/A3 grid heatmaps")
    results["A2_A3"] = base.fig_A2_A3_grids(df)
    print("[stats] A4 PCA hull")
    results["A4"] = base.fig_A4_hull_pca2d(df)
    print("[stats] B1 target hist")
    results["B1"] = base.fig_B1_target_hist(df)
    print("[stats] B2 class balance")
    results["B2"] = base.fig_B2_class_balance(df)
    print("[stats] B3 axis boxplots")
    results["B3"] = base.fig_B3_by_axis_box(df)
    print("[stats] C1 MI/Spearman")
    results["C1"] = base.fig_C1_mi_spearman(df)
    print("[stats] C2 corr matrix")
    results["C2"] = base.fig_C2_corr_matrix(df)
    print("[stats] C3 interaction H")
    results["C3"] = base.fig_C3_interaction_h(df)
    print("[stats] D1 5-fold CV")
    results["D1"] = base.fig_D1_cv_r2(df)
    print("[stats] D2 learning curve")
    results["D2"] = base.fig_D2_learning_curve(df)
    print("[stats] D3 OOD")
    results["D3"] = base.fig_D3_ood(df)
    print("[stats] E1 noise floor")
    results["E1"] = base.fig_E1_noise_floor(df, Path("/tmp/plan4_phase0/noise.json"))

    (out / "stats_results.json").write_text(
        json.dumps(results, indent=2, default=str))
    base.write_report(df, results)


if __name__ == "__main__":
    main()
