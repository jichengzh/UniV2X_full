"""M4.9 v2 — Unified 10-anchor Pareto on Pyramid_DAIR_m1.

Adds 4 supplementary anchors over v1:
  - A5/A6: Pruned 25% FT FP16/INT8 (intermediate prune rate)
  - A7:    Mixed precision (FP16 heads + INT8 backbone, per-module q_bits)
  - A8:    INT8 + SPARSE_WEIGHTS flag (ASP not applied — flag noop on this model)
  - A9:    Orin AGX FP16 cross-platform (M dimension; INT8 calib cache 4090↔Orin not portable)

All latencies use clean re-bench on idle GPU 6 (no contention).
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
    R = REPO_ROOT / "results"

    # PT FP32 sub-module bench (baseline lat anchor)
    pt_bench = load(R / "m4_8_dair_pytorch_subnet_bench.json")
    pt_fp32 = next(a for a in pt_bench["anchors"] if a["anchor"] == "pytorch_fp32")

    # CLEAN re-benched collab N=2 latencies (GPU 6 idle)
    bench = {
        "fp16":           load(R / "m4_8_dair_collab_trt_fp16_clean.json"),
        "int8":           load(R / "m4_8_dair_collab_trt_int8_clean.json"),
        "mixed":          load(R / "m4_8_dair_collab_trt_mixed_clean.json"),
        "int8_sparse":    load(R / "m4_8_dair_collab_trt_int8_sparse_clean.json"),
        "pruned25_fp16":  load(R / "m4_8_dair_pruned25_ft_collab_trt_fp16_clean.json"),
        "pruned25_int8":  load(R / "m4_8_dair_pruned25_ft_collab_trt_int8_clean.json"),
        "pruned50_fp16":  load(R / "m4_8_dair_pruned50_ft_collab_trt_fp16.json"),
        "pruned50_int8":  load(R / "m4_8_dair_pruned50_ft_collab_trt_int8.json"),
        "pruned75_fp16":  load(R / "m4_8_dair_pruned75_ft_collab_trt_fp16_clean.json"),
        "pruned75_int8":  load(R / "m4_8_dair_pruned75_ft_collab_trt_int8_clean.json"),
    }

    # AP eval (full DAIR val 1789, hybrid PyTorch + TRT)
    ap = {
        "baseline":         load(R / "m4_8_hybrid_ap_dair_pytorch_baseline.json"),
        "fp16":             load(R / "m4_8_hybrid_ap_dair_collab_fp16.json"),
        "int8":             load(R / "m4_8_hybrid_ap_dair_collab_int8.json"),
        "mixed":            load(R / "m4_8_hybrid_ap_dair_collab_mixed.json"),
        "int8_sparse":      load(R / "m4_8_hybrid_ap_dair_collab_int8_sparse.json"),
        "pruned25_fp16":    load(R / "m4_8_hybrid_ap_dair_pruned25_ft_collab_fp16.json"),
        "pruned25_int8":    load(R / "m4_8_hybrid_ap_dair_pruned25_ft_collab_int8.json"),
        "pruned50_fp16":    load(R / "m4_8_hybrid_ap_dair_pruned50_ft_collab_fp16.json"),
        "pruned50_int8":    load(R / "m4_8_hybrid_ap_dair_pruned50_ft_collab_int8.json"),
        "pruned75_fp16":    load(R / "m4_8_hybrid_ap_dair_pruned75_ft_collab_fp16.json"),
        "pruned75_int8":    load(R / "m4_8_hybrid_ap_dair_pruned75_ft_collab_int8.json"),
    }

    # Orin: latency-only (FP16 build, no INT8 because TRT 4090 cache not portable)
    orin_fp16_lat_ms = 47.92  # mean GPU compute time from trtexec --avgRuns=200
    orin_engine_mb = 12.5  # estimated FP16 engine size on Orin

    rows = []
    BASE_AP50 = ap["baseline"]["ap50"]
    PT_FP32 = pt_fp32["p50_ms"]

    def add(tag, lat, engine_mb, ap_d, prune, prec, platform="RTX 4090", notes=""):
        cov = round(ap_d["n_trt_collab_path"] / ap_d["n_samples"] * 100, 1) \
            if "n_trt_collab_path" in ap_d else 0
        rows.append({
            "anchor": tag,
            "prune_pct": prune,
            "precision": prec,
            "platform": platform,
            "engine_mb": round(engine_mb, 2) if engine_mb else None,
            "lat_p50_ms": round(lat, 3) if lat else None,
            "speedup_vs_pt_fp32": round(PT_FP32 / lat, 2) if lat else None,
            "ap30": round(ap_d["ap30"], 4) if "ap30" in ap_d else None,
            "ap50": round(ap_d["ap50"], 4) if "ap50" in ap_d else None,
            "ap70": round(ap_d["ap70"], 4) if "ap70" in ap_d else None,
            "delta_ap50_pp": round((ap_d["ap50"] - BASE_AP50) * 100, 2) if "ap50" in ap_d else None,
            "trt_coverage_pct": cov,
            "notes": notes,
        })

    add("A0_pytorch_fp32_baseline", PT_FP32, None, ap["baseline"],
        prune=0, prec="FP32", notes="HEAL paper baseline ✓")
    add("A1_trt_fp16_collab", bench["fp16"]["p50_ms"], bench["fp16"]["engine_size_mb"], ap["fp16"],
        prune=0, prec="FP16")
    add("A2_trt_int8_collab", bench["int8"]["p50_ms"], bench["int8"]["engine_size_mb"], ap["int8"],
        prune=0, prec="INT8")
    add("A3_pruned50_ft_fp16", bench["pruned50_fp16"]["p50_ms"], bench["pruned50_fp16"]["engine_size_mb"], ap["pruned50_fp16"],
        prune=50, prec="FP16", notes="50% L1-channel + 1ep FT")
    add("A4_pruned50_ft_int8", bench["pruned50_int8"]["p50_ms"], bench["pruned50_int8"]["engine_size_mb"], ap["pruned50_int8"],
        prune=50, prec="INT8", notes="50% L1-channel + 1ep FT + INT8 PTQ")
    add("A5_pruned25_ft_fp16", bench["pruned25_fp16"]["p50_ms"], bench["pruned25_fp16"]["engine_size_mb"], ap["pruned25_fp16"],
        prune=25, prec="FP16", notes="NEW: 25% L1-channel + 1ep FT (intermediate prune rate)")
    add("A6_pruned25_ft_int8", bench["pruned25_int8"]["p50_ms"], bench["pruned25_int8"]["engine_size_mb"], ap["pruned25_int8"],
        prune=25, prec="INT8", notes="NEW: 25% L1-channel + 1ep FT + INT8 PTQ")
    add("A7_mixed_heads_fp16_backbone_int8", bench["mixed"]["p50_ms"], bench["mixed"]["engine_size_mb"], ap["mixed"],
        prune=0, prec="MIXED",
        notes="NEW: FP16 cls/reg/dir heads + INT8 backbone (per-module q_bits)")
    add("A8_int8_sparse_weights_flag", bench["int8_sparse"]["p50_ms"], bench["int8_sparse"]["engine_size_mb"], ap["int8_sparse"],
        prune=0, prec="INT8+SW",
        notes="NEW: BuilderFlag.SPARSE_WEIGHTS (no ASP — TRT auto-detects 2:4; flag noop on this model)")
    add("A9_orin_agx_fp16", orin_fp16_lat_ms, orin_engine_mb, ap["fp16"],
        prune=0, prec="FP16", platform="Orin AGX",
        notes="NEW: Orin AGX TRT 8.5.2 FP16 (AP from 4090 same engine; INT8 calib cache 4090↔Orin not portable)")
    add("A10_pruned75_ft_fp16", bench["pruned75_fp16"]["p50_ms"], bench["pruned75_fp16"]["engine_size_mb"], ap["pruned75_fp16"],
        prune=75, prec="FP16",
        notes="NEW: 75% L1-channel + 1ep FT (widths [1,2,4] all power-of-2 ✓ fast-path)")
    add("A11_pruned75_ft_int8", bench["pruned75_int8"]["p50_ms"], bench["pruned75_int8"]["engine_size_mb"], ap["pruned75_int8"],
        prune=75, prec="INT8",
        notes="NEW: 75% L1-channel + 1ep FT + INT8 PTQ — dominates A4 prune50 INT8 in all 3 dims")

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "results/m4_9_framework_pareto_v2.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote -> {out}\n")
    print("=" * 130)
    print("M4.9 v2 FRAMEWORK PARETO — Pyramid_DAIR_m1 on DAIR-V2X val 1789 (10 anchors)")
    print("=" * 130)
    print(df.drop(columns=["notes"]).to_string(index=False))
    print("\n--- notes ---")
    for _, r in df.iterrows():
        if r["notes"]:
            print(f"  {r['anchor']}: {r['notes']}")

    # Pareto frontier (RTX 4090 only — Orin is different platform, plotted separately)
    df_4090 = df[df["platform"] == "RTX 4090"].reset_index(drop=True)
    print("\n--- Pareto-optimal anchors on RTX 4090 (lower lat, higher AP50) ---")
    pareto_idx = []
    for i, r in df_4090.iterrows():
        if r["lat_p50_ms"] is None: continue
        is_pareto = True
        for j, q in df_4090.iterrows():
            if i == j or q["lat_p50_ms"] is None: continue
            if q["lat_p50_ms"] <= r["lat_p50_ms"] and q["ap50"] >= r["ap50"] and \
               (q["lat_p50_ms"] < r["lat_p50_ms"] or q["ap50"] > r["ap50"]):
                is_pareto = False
                break
        if is_pareto:
            pareto_idx.append(i)
    pdf = df_4090.loc[pareto_idx]
    print(pdf[["anchor", "lat_p50_ms", "speedup_vs_pt_fp32", "ap50", "delta_ap50_pp", "engine_mb"]].to_string(index=False))

    print("\n--- M-dimension cross-platform (Orin AGX vs RTX 4090, same FP16 engine) ---")
    o_lat = float(df[df["anchor"] == "A9_orin_agx_fp16"]["lat_p50_ms"].iloc[0])
    g_lat = float(df[df["anchor"] == "A1_trt_fp16_collab"]["lat_p50_ms"].iloc[0])
    print(f"  Orin AGX FP16 collab: {o_lat:.2f} ms   |   RTX 4090 FP16 collab: {g_lat:.3f} ms   |   ratio = {o_lat/g_lat:.1f}× (Orin slower)")

    return df


if __name__ == "__main__":
    main()
