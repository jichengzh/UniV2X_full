#!/usr/bin/env python3
"""Phase 1 (E6) — Pyramid-LiDAR RSU-perception P x Q Pareto assembly (v2, real anchors).

DELIVERABLE = deployable Pareto over the joint P x Q software space
(343 widths x 3 precisions = 1029), with three extreme points
(AP-max / latency-min / energy-min) whose lat+energy+AP are all REAL-measured
at ONE unified caliber.

Real-measured anchor front (this is the empirical Pareto backbone):
  - 4 gold-anchor widths [16,32,64] [32,64,128] [48,96,192] [64,128,256]
    (the AP-carrying w0 diagonal), each built + measured at the UNIFIED
    caliber: H800 / TVM, im2col + WMMA (fp16 --cast-fp16-source, tensorcore_gate=True,
    144-180 wmma intrinsics), input [2,64,256,256] (real AP-shape), batch=2,
    fresh per-width process, SERIAL exclusive measurement (no cross-lane contention).
  - fp16 latency + energy: DIRECTLY real-measured (this run).
  - AP70: gold converged (stage_a_ap_real, DAIR val n=1789), per-anchor exact
    (anchors ARE the gold anchor widths) -> AP axis is real per-point, not estimated.

int8_tc Q-axis:
  - int8 latency/energy = fp16 anchor value x REAL network-level int8/fp16 ratio,
    where the ratio is measured from the 60-width original60 corpus (both fp16 and
    int8_tc real-measured there) at the matching w0. Labeled
    'int8tc_real_corpus_ratio_derived' (ratio is real; direct anchor int8_tc .so
    build via the scale-aware native pipeline = documented GAP G-int8).
  - int8 AP70 = gold converged int8 (stage_a_ap_real), NOT fake-quant.

Search context:
  - The joint P x Q surrogate front over the 1029 space is produced by the
    implemented framework NSGA-II (framework/run_pqs_ablation.py / search_three_arm.py),
    cost-model warm-started on the 180-LUT (lat spearman 0.991). The anchor diagonal
    is the frontier neighborhood that this run real-measures.

Three extremes are taken over the DIRECT-real fp16 anchors so each is fully
real-measured; int8 (derived) is shown to extend the front further.

Outputs (shared fs):
  results/phase1_pyramid_pareto.csv
  results/phase1_pyramid_pareto.json
  multi_agent/figure/fig_phase1_pyramid_pareto.png
"""
import json, os, csv, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jichengzhi/V2X"
GEN = os.path.join(ROOT, "multi_agent/data/stage2_lut_generation_v1/generated/"
                   "original60_quant_20260627")
ANCHOR_DIR = os.path.join(GEN, "anchor_g1")
CORPUS = os.path.join(GEN, "cost_model/train/original60_training_table_latest.json")
OUT_RESULTS = os.path.join(ROOT, "results")
OUT_FIG = os.path.join(ROOT, "multi_agent/figure")

# gold converged AP70 (stage_a_ap_real, DAIR val n=1789) keyed by w0 (stage0 planes)
GOLD_AP = {
    16: {"fp16": 0.5300, "int8": 0.5236},
    32: {"fp16": 0.5641, "int8": 0.5542},
    48: {"fp16": 0.5905, "int8": 0.5841},
    64: {"fp16": 0.6309, "int8": 0.6228},
}
ANCHORS = [[16, 32, 64], [32, 64, 128], [48, 96, 192], [64, 128, 256]]
LAT_CAL = ("measured_h800_tvm_wmma_ap256x256_batch2_serial")


def wkey(w):
    return "x".join(str(x) for x in w)


INT8_CAL = ("measured_h800_tvm_int8tc_ap256x256_batch2_serial "
            "(--rewrite-dtype int8 --cast-fp16-source, tensorcore_gate=True, "
            "latency-only dummy scale = scale-independent real int8_tc kernel latency)")


def build_points():
    flat = json.load(open(os.path.join(ANCHOR_DIR, "anchor_fp16_lat_collected.json")))
    fen = json.load(open(os.path.join(ANCHOR_DIR, "anchor_fp16_energy_collected.json")))
    ilat = json.load(open(os.path.join(ANCHOR_DIR, "anchor_int8_lat_collected.json")))
    ien = json.load(open(os.path.join(ANCHOR_DIR, "anchor_int8_energy_collected.json")))
    pts = []
    ratio_log = {}
    for w in ANCHORS:
        k = wkey(w)
        w0 = w[0]
        f_lat = round(float(flat[k]["lat_wmma_ms"]), 4)
        f_en = round(float(fen[k]), 4)
        i_lat = round(float(ilat[k]["lat_int8tc_ms"]), 4)
        i_en = round(float(ien[k]), 4)
        assert flat[k]["tensorcore_gate"] is True, f"anchor {k} fp16 not tensorized"
        assert ilat[k]["tensorcore_gate"] is True, f"anchor {k} int8 not tensorized"
        ratio_log[k] = {"int8_fp16_lat": round(i_lat / f_lat, 4),
                        "int8_fp16_energy": round(i_en / f_en, 4)}
        # fp16 anchor: all three axes DIRECT real-measured
        pts.append({
            "width": w, "precision": "fp16", "s": "tuned_wmma",
            "latency_ms": f_lat, "energy_j": f_en, "ap70": GOLD_AP[w0]["fp16"],
            "lat_source": LAT_CAL, "energy_source": LAT_CAL,
            "ap_source": "gold_stage_a_converged_ap70_n1789",
            "status": "measured",
        })
        # int8_tc anchor: lat/energy DIRECT real-measured (int8 tensor cores); AP = gold int8
        pts.append({
            "width": w, "precision": "int8_tc", "s": "tuned_int8_tc",
            "latency_ms": i_lat, "energy_j": i_en, "ap70": GOLD_AP[w0]["int8"],
            "lat_source": INT8_CAL, "energy_source": INT8_CAL,
            "ap_source": "gold_stage_a_converged_ap70_n1789_int8",
            "status": "measured",
        })
    return pts, ratio_log


def pareto_mask(pts):
    """3-objective: maximize ap70, minimize latency, minimize energy."""
    for a in pts:
        dom = False
        for b in pts:
            if b is a:
                continue
            if (b["ap70"] >= a["ap70"] and b["latency_ms"] <= a["latency_ms"]
                    and b["energy_j"] <= a["energy_j"]
                    and (b["ap70"] > a["ap70"] or b["latency_ms"] < a["latency_ms"]
                         or b["energy_j"] < a["energy_j"])):
                dom = True
                break
        a["on_pareto"] = not dom


def main():
    os.makedirs(OUT_RESULTS, exist_ok=True)
    os.makedirs(OUT_FIG, exist_ok=True)
    pts, ratio_log = build_points()
    pareto_mask(pts)

    # three extremes over ALL anchors (fp16 + int8_tc all DIRECT real-measured)
    real = [p for p in pts if p["status"] == "measured"]
    assert len(real) == len(pts) == 8, "all 8 anchors must be real-measured"
    latmin = min(real, key=lambda p: p["latency_ms"])
    emin = min(real, key=lambda p: p["energy_j"])
    apmax = max(real, key=lambda p: p["ap70"])
    for p in pts:
        p["extreme"] = []
    latmin["extreme"].append("latency_min")
    emin["extreme"].append("energy_min")
    apmax["extreme"].append("ap_max")

    payload = {
        "phase": "1_E6_pyramid_lidar_rsu_pareto",
        "model": "Pyramid_DAIR_m1_base_2023_08_14_11_42_29 (DAIR gold)",
        "space": "343 widths x 3 precisions = 1029 (joint P x Q); s=inner-tuned",
        "caliber": {
            "latency_energy": "H800 / TVM tensorcore, input [2,64,256,256] AP-shape, "
                              "batch=2, fresh per-width process, SERIAL exclusive measure; "
                              "fp16=im2col+WMMA (--cast-fp16-source, gate=True), "
                              "int8_tc=MatmulInt8Tensorization (--rewrite-dtype int8 "
                              "--cast-fp16-source, gate=True, scale-independent latency)",
            "ap70": "gold converged (stage_a_ap_real, DAIR val n=1789), per-anchor exact",
        },
        "anchor_front_n_real": len(real),
        "int8_fp16_ratio_per_anchor_measured": ratio_log,
        "search_context": {
            "engine": "framework/run_pqs_ablation.py + search_three_arm.py NSGA-II "
                      "(pop=8, budget=60); exhaustive exact surrogate Pareto over the "
                      "full 1029 -> results/phase1_surrogate_search_1029.json",
            "verified_vs_predicted": "This file = the REAL-MEASURED Pareto (8 diagonal "
                    "anchors [w0,2w0,4w0], all 3 axes real). The 1029 surrogate (under the "
                    "established w0-dominant-AP finding) PREDICTS minimal-neck configs "
                    "[w0,16,16] dominate the diagonal (same w0-AP at lower lat), front int8=7 "
                    "/ fp16=3 / fp32=0. Those minimal-neck points are the co-design search's "
                    "next real-measure targets (need AP finetune at minimal-neck to confirm "
                    "w0-only-AP holds there) = GAP G-minimal-neck. The diagonal anchors are "
                    "the conservative VERIFIED frontier neighborhood.",
        },
        "extreme_points": {
            "ap_max": {"width": apmax["width"], "precision": apmax["precision"],
                       "latency_ms": apmax["latency_ms"], "energy_j": apmax["energy_j"],
                       "ap70": apmax["ap70"], "source": "ALL REAL-measured"},
            "latency_min": {"width": latmin["width"], "precision": latmin["precision"],
                            "latency_ms": latmin["latency_ms"], "energy_j": latmin["energy_j"],
                            "ap70": latmin["ap70"], "source": "ALL REAL-measured"},
            "energy_min": {"width": emin["width"], "precision": emin["precision"],
                           "latency_ms": emin["latency_ms"], "energy_j": emin["energy_j"],
                           "ap70": emin["ap70"], "source": "ALL REAL-measured"},
        },
        "gaps": {
            "G_int8_ap_finetune": "int8_tc AP70 uses gold converged int8 (stage_a_ap_real, "
                                  "real calibration, NOT fake-quant); per-anchor int8 AP is "
                                  "exact at gold anchor widths. int8 latency/energy DIRECT "
                                  "real-measured (scale-independent kernel).",
            "G_minimal_neck": "1029 surrogate predicts minimal-neck [w0,16,16] dominates the "
                              "measured diagonal IF w0-only-AP holds there (prior pruning "
                              "finding, not gold-verified at minimal-neck). Next real-measure "
                              "target: AP finetune + lat/energy at minimal-neck front widths.",
            "G_corpus_caliber": "180-LUT corpus (fp16@256 vs int8@128, tuned~=default) = "
                                "cost-model warm-start / surrogate-shape source only; NOT on "
                                "the unified deployable latency axis.",
        },
        "points": pts,
    }
    json.dump(payload, open(os.path.join(OUT_RESULTS, "phase1_pyramid_pareto.json"),
                            "w"), indent=2, ensure_ascii=False)

    cols = ["width", "precision", "s", "latency_ms", "energy_j", "ap70",
            "on_pareto", "status", "extreme", "lat_source", "ap_source"]
    with open(os.path.join(OUT_RESULTS, "phase1_pyramid_pareto.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(cols)
        for p in pts:
            wtr.writerow([wkey(p["width"]), p["precision"], p["s"], p["latency_ms"],
                          p["energy_j"], p["ap70"], p["on_pareto"], p["status"],
                          "|".join(p["extreme"]), p["lat_source"], p["ap_source"]])

    # figure: latency (x) vs AP70 (y), colour=energy, marker=precision
    fig, ax = plt.subplots(figsize=(9, 6))
    for prec, mk, lbl in (("fp16", "s", "fp16 WMMA (real-measured)"),
                          ("int8_tc", "^", "int8_tc (real-measured)")):
        ps = [p for p in pts if p["precision"] == prec]
        sc = ax.scatter([p["latency_ms"] for p in ps], [p["ap70"] for p in ps],
                        c=[p["energy_j"] for p in ps], cmap="viridis", marker=mk,
                        s=130, edgecolors="k", linewidths=0.6, vmin=0.4, vmax=2.6,
                        label=lbl, zorder=3)
    fp16pts = sorted([p for p in pts if p["precision"] == "fp16"],
                     key=lambda p: p["latency_ms"])
    ax.plot([p["latency_ms"] for p in fp16pts], [p["ap70"] for p in fp16pts], "b-",
            lw=1.4, alpha=0.7, label="fp16 front", zorder=2)
    i8s = sorted([p for p in pts if p["precision"] == "int8_tc"],
                 key=lambda p: p["latency_ms"])
    ax.plot([p["latency_ms"] for p in i8s], [p["ap70"] for p in i8s], "r--", lw=1.2,
            alpha=0.6, label="int8_tc front", zorder=2)
    for p in pts:
        for e in p["extreme"]:
            ax.annotate(f"{e}\n{wkey(p['width'])} {p['precision']}",
                        (p["latency_ms"], p["ap70"]), fontsize=7,
                        xytext=(6, 6), textcoords="offset points", zorder=4)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("energy (J/inference, measured)")
    ax.set_xlabel("latency (ms, H800 TVM WMMA, AP-shape 256x256, batch2, real-measured)")
    ax.set_ylabel("AP70 (gold converged, DAIR val n=1789)")
    ax.set_title("Phase 1 / E6 - Pyramid-LiDAR RSU Pareto (P x Q joint)\n"
                 "fp16 (WMMA) + int8_tc all REAL-measured (lat+energy+AP), unified caliber")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_FIG, "fig_phase1_pyramid_pareto.png"), dpi=140)

    print("=== Phase 1 Pareto (v2, real anchors, fp16+int8 all measured) ===")
    print(f"points: {len(pts)}  (all real-measured 3-axis)")
    print(f"on_pareto: {sum(1 for p in pts if p['on_pareto'])}")
    print(f"AP-max     : {wkey(apmax['width'])} {apmax['precision']} "
          f"AP70={apmax['ap70']} lat={apmax['latency_ms']}ms E={apmax['energy_j']}J [REAL]")
    print(f"lat-min    : {wkey(latmin['width'])} {latmin['precision']} "
          f"lat={latmin['latency_ms']}ms AP70={latmin['ap70']} E={latmin['energy_j']}J [REAL]")
    print(f"energy-min : {wkey(emin['width'])} {emin['precision']} "
          f"E={emin['energy_j']}J AP70={emin['ap70']} lat={emin['latency_ms']}ms [REAL]")


if __name__ == "__main__":
    main()
