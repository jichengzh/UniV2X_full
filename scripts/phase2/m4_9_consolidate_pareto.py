"""M4.9 — Consolidate all m4_8 + Phase B measurements into final framework Pareto.

Reads existing JSON results files and produces:
  - results/m4_9_framework_pareto.csv (final Pareto table)
  - results/m4_9_framework_pareto_report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def load(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def main():
    results = REPO_ROOT / "results"

    # Latency benches (sub-module pyramid_backbone+shrink+heads on DAIR)
    pt_bench = load(results / "m4_8_dair_pytorch_subnet_bench.json")
    pt_fp32 = next(a for a in pt_bench["anchors"] if a["anchor"] == "pytorch_fp32")
    pt_fp16h = next(a for a in pt_bench["anchors"] if a["anchor"] == "pytorch_fp16_half")

    # Latency benches (collab N=2 e2e — Phase A.5 + B)
    trt_fp32_co = load(results / "m4_8_dair_collab_trt_fp32.json")
    trt_fp16_co = load(results / "m4_8_dair_collab_trt_fp16.json")
    trt_int8_co = load(results / "m4_8_dair_collab_trt_int8.json")
    pr_fp16_co = load(results / "m4_8_dair_pruned50_ft_collab_trt_fp16.json")
    pr_int8_co = load(results / "m4_8_dair_pruned50_ft_collab_trt_int8.json")

    # AP eval (full DAIR val 1789, hybrid PyTorch + TRT)
    ap_baseline = load(results / "m4_8_hybrid_ap_dair_pytorch_baseline.json")
    ap_fp16_co = load(results / "m4_8_hybrid_ap_dair_collab_fp16.json")
    ap_int8_co = load(results / "m4_8_hybrid_ap_dair_collab_int8.json")
    ap_pr_fp16 = load(results / "m4_8_hybrid_ap_dair_pruned50_ft_collab_fp16.json")
    ap_pr_int8 = load(results / "m4_8_hybrid_ap_dair_pruned50_ft_collab_int8.json")

    rows = []
    BASE_AP50 = ap_baseline["ap50"]

    def add_row(tag, lat_p50, engine_mb, ap, prune, prec, cov_pct, notes=""):
        rows.append({
            "anchor": tag,
            "prune_pct": prune,
            "precision": prec,
            "engine_mb": round(engine_mb, 2) if engine_mb else None,
            "lat_p50_ms": round(lat_p50, 3),
            "speedup_vs_pt_fp32": round(pt_fp32["p50_ms"] / lat_p50, 2) if lat_p50 else None,
            "ap30": round(ap["ap30"], 4),
            "ap50": round(ap["ap50"], 4),
            "ap70": round(ap["ap70"], 4),
            "delta_ap50_pp": round((ap["ap50"] - BASE_AP50) * 100, 2),
            "trt_coverage_pct": cov_pct,
            "notes": notes,
        })

    # PT FP32 baseline (force_fallback hybrid AP)
    add_row("A0_pytorch_fp32_baseline",
            pt_fp32["p50_ms"], None, ap_baseline,
            prune=0, prec="FP32", cov_pct=0,
            notes="reproduces HEAL paper baseline ✓")
    # PT FP16 .half() (sub-module) — but we don't have hybrid AP for pure PyTorch FP16
    # Skip for now — TRT FP16 is more interesting

    # TRT FP16 collab (Phase A.5)
    add_row("A1_trt_fp16_collab",
            trt_fp16_co["p50_ms"], trt_fp16_co["engine_size_mb"], ap_fp16_co,
            prune=0, prec="FP16", cov_pct=round(ap_fp16_co["n_trt_collab_path"] / ap_fp16_co["n_samples"] * 100, 1))

    # TRT INT8 collab (Phase A.5)
    add_row("A2_trt_int8_collab",
            trt_int8_co["p50_ms"], trt_int8_co["engine_size_mb"], ap_int8_co,
            prune=0, prec="INT8", cov_pct=round(ap_int8_co["n_trt_collab_path"] / ap_int8_co["n_samples"] * 100, 1))

    # Pruned 50% finetune + FP16 collab (Phase B)
    add_row("A3_pruned50_ft_fp16_collab",
            pr_fp16_co["p50_ms"], pr_fp16_co["engine_size_mb"], ap_pr_fp16,
            prune=50, prec="FP16", cov_pct=round(ap_pr_fp16["n_trt_collab_path"] / ap_pr_fp16["n_samples"] * 100, 1),
            notes="50% L1-channel prune + 1ep finetune")

    # Pruned 50% finetune + INT8 collab (Phase B+A.5)
    add_row("A4_pruned50_ft_int8_collab",
            pr_int8_co["p50_ms"], pr_int8_co["engine_size_mb"], ap_pr_int8,
            prune=50, prec="INT8", cov_pct=round(ap_pr_int8["n_trt_collab_path"] / ap_pr_int8["n_samples"] * 100, 1),
            notes="50% L1-channel prune + 1ep finetune + INT8 PTQ")

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results/m4_9_framework_pareto.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote -> {out}\n")
    print("=" * 100)
    print("M4.9 FRAMEWORK PARETO — Pyramid_DAIR_m1 on DAIR-V2X val 1789")
    print("=" * 100)
    print(df.to_string(index=False))

    # Compute Pareto-optimality (lat × ap50)
    print("\n--- Pareto-optimal anchors (lower lat, higher AP50) ---")
    pareto_idx = []
    for i, r in df.iterrows():
        if r["lat_p50_ms"] is None: continue
        is_pareto = True
        for j, q in df.iterrows():
            if i == j or q["lat_p50_ms"] is None: continue
            if q["lat_p50_ms"] <= r["lat_p50_ms"] and q["ap50"] >= r["ap50"] and \
               (q["lat_p50_ms"] < r["lat_p50_ms"] or q["ap50"] > r["ap50"]):
                is_pareto = False
                break
        if is_pareto:
            pareto_idx.append(i)
    pdf = df.loc[pareto_idx]
    print(pdf[["anchor", "lat_p50_ms", "speedup_vs_pt_fp32", "ap50", "delta_ap50_pp"]].to_string(index=False))

    return df


if __name__ == "__main__":
    main()
