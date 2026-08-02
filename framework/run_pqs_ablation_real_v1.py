"""P×Q×S three-axis ablation -- REAL CALIBER (per-width measured int8_tc).

Sibling of framework/run_pqs_ablation.py (the shipped PROXY version: HV
56.7% / 85.3% / 100.0% for A-noS-PQS / A-serial-PQS / A-joint-PQS, Wilcoxon
p=4.88e-4. That script and its output artifacts (results/pqs_ablation_results.json,
multi_agent/figure/pqs_hv_boxplot.png, multi_agent/figure/pqs_pareto_int8.png)
are UNTOUCHED by this file).

WHAT CHANGES vs the proxy version
----------------------------------
The proxy version prices every buildable-int8 width at the SAME uniform
constant, ``UNIFORM_INT8_SPEEDUP = 1.449`` (H800 fp16/int8 ratio for a single
*stage0-only conv submodule* microbenchmark, results/q_int8_ms_stage0_result.json
-- see results/q_int8_dp4a_pairs.csv: identical 150.1us stage0 latency recorded
for pad64/mix_b/s1_64/s2_128, i.e. the old measurement cannot even tell these
widths apart).

This script instead uses ``framework.q_lookup_real.RealQLookup``, which loads
results/q_int8tc_real_ratios_v1.json: REAL, per-width, FULL-BACKBONE
(latency_kind="pyramid_backbone_subnet") H800 measurements from the original60
SMBO loop (multi_agent/data/stage2_lut_generation_v1/generated/
original60_quant_20260627/smbo_loop/), using the int8_tc (tensorcore, im2col+MMA,
tensorcore_gate=true, wmma_count>0) backend -- NOT native/untensorized SIMT
int8. For each width with a real (fp16, int8_tc) measured pair (same SMBO
batch, so backend-consistent with each other), the REAL ratio
fp16_lat_tuned_ms / int8tc_lat_tuned_ms is applied multiplicatively to
whatever fp16 latency the ablation's own (unchanged) LatencyLUT resolves for
that width. Real per-inference energy_j is also carried through for both
precisions as informational metadata (NOT folded into the 2-objective
(latency, AP) NSGA-II/HV machinery -- that would be a scope change to
search_three_arm.py, out of scope here).

COVERAGE (see results/q_int8tc_real_ratios_v1.json, rebuild via
``python -m scripts.phase2.build_q_int8tc_real_ratios_v1``):
    covered  (4/9): base [64,128,256], p50 [32,64,128], p75 [16,32,64],
                    trap25 [48,96,192]
    missing  (5/9): pad64 [64,96,192], mix_b [48,64,256], s1_64 [64,64,256],
                    mix_d [48,128,128], s2_128 [64,128,128]
Missing widths get NO assumed int8 speedup by default (int8_lat returns the
fp16 latency unchanged, tagged lat_source="...unmeasured_neutral_fp16_latency"
by the EXISTING, unmodified CostModelPQS.evaluate() logic in
search_three_arm.py) -- unless --fallback-proxy is explicitly passed, which
degrades those 5 widths to the SAME proxy constant, clearly labeled, for a
"best-effort full-grid" comparison run. See ``results/
pqs_ablation_real_v1_missing_measurements.json`` for the exact
``measure_config.py --width ... --precision {fp16,int8_tc} --gpu <idle>``
commands needed to close each gap.

NARRATIVE SHIFT: categorical -> quantitative
---------------------------------------------
The proxy version's "structural_q_claim" is CATEGORICAL: W_g (s0=48,
in_per_g=3) is *structurally* int8-unbuildable (in_per_g % 4 != 0, the legacy
NCHWc dp4a pack-4 constraint) while P_g (s0=64) is buildable, so A-serial (which
locks W_g under fp16) can never reach an int8 point at all, while A-joint can.

2026-07-03 finding (real int8_tc measurements): int8_tc BUILDS on every width
tried so far, INCLUDING s0=48 (trap25 here: build_success=true,
tensorcore_gate=true, wmma_count=180). The dp4a pack-4 wall does not exist
under the current (int8_tc) backend. This script therefore defaults
``enforce_int8_buildable=False`` (pass --legacy-buildability-wall to force the
old rule for a side-by-side run). Under the real-caliber default, the
CATEGORICAL claim is NOT APPLICABLE (flagged explicitly in the output JSON,
not silently dropped) -- what real data CAN show is a QUANTITATIVE claim:
per-width int8_tc speedup is real but NON-UNIFORM (1.05x-1.23x across the 4
covered widths, vs the single 1.449x the proxy assumes everywhere), computed
via the existing, unmodified ``detect_q_rank_flip_pairs()`` helper fed with
the RealQLookup instance.

OPEN GAP NOT ADDRESSED HERE: the int8 AP delta (_INT8_AP_DELTA_MEDIAN=-0.008
in search_three_arm.py) remains a historical-TRT-derived proxy; AP under the
real int8_tc backbone has not been separately re-evaluated. Flagged in meta,
not fixed by this upgrade.

Run:
    python -m framework.run_pqs_ablation_real_v1 [--seeds 12] [--budget 60]
        [--pop 8] [--fallback-proxy 1.095] [--legacy-buildability-wall] [--quiet]

Outputs (distinct from the proxy version's artifacts):
    results/pqs_ablation_results_real_v1.json
    results/pqs_ablation_real_v1_missing_measurements.json
    multi_agent/figure/pqs_hv_boxplot_real_v1.png
    multi_agent/figure/pqs_pareto_int8_real_v1.png
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from framework.search_three_arm import (
    APModel, LatencyLUT, SCHEDULES, QUANT_MODES,
    candidate_widths, compute_hv_ref_pqs,
    run_joint_pqs, run_noS_pqs, run_serial_pqs,
    detect_q_rank_flip_pairs, detect_wg_pg_pairs, load_seed_grid,
)
from framework.run_b4_ablation import wilcoxon_signed_rank, aggregate_convergence
from framework.run_pqs_ablation import plot_hv_boxplot, _ref_hv_scalar
from framework.q_lookup_real import RealQLookup, REAL_RATIO_JSON

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "multi_agent" / "figure"


# ---------------------------------------------------------------------------
# Real-caliber-specific plot (own copy, NOT framework.run_pqs_ablation's
# plot_pareto_int8, which hardcodes proxy-specific labels: "INT8 = H800 fp16 /
# 1.449" and "s0=48 INT8 UNBUILDABLE" -- both wrong/misleading under real
# caliber where enforce_int8_buildable defaults to False and the ratio is
# per-width, not a constant.)
# ---------------------------------------------------------------------------
def plot_pareto_int8_real(joint_visited, serial_visited, apm, lut, qlut: RealQLookup,
                           q_pairs, hv_ref, path: Path):
    """Scatter: fp16 vs int8 points, real-caliber. Annotates each detected
    W_g/P_g pair with whether its int8 point is REAL-measured or
    proxy/unmeasured (per qlut.has_direct_h800), instead of assuming a
    categorical buildability split."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    def split(visited):
        pts = {"fp16": ([], []), "int8": ([], [])}
        for (w, s, q) in visited:
            if q == "int8" and qlut.enforce_int8_buildable and not qlut.can_build_int8(w):
                continue   # legacy-wall mode only: unbuildable -> no real point
            lat = qlut.int8_lat(lut.latency(w, s), w, sched=s) if q == "int8" else lut.latency(w, s)
            ap = apm.ap70(w) - (0.008 if q == "int8" else 0.0)
            pts[q][0].append(lat); pts[q][1].append(ap)
        return pts

    jp = split(joint_visited)
    sp = split(serial_visited)
    ax.scatter(jp["fp16"][0], jp["fp16"][1], s=70, marker="o", facecolors="none",
               edgecolors="#2c7fb8", linewidths=1.5, label="A-joint fp16")
    ax.scatter(jp["int8"][0], jp["int8"][1], s=120, marker="o",
               color="#2c7fb8", alpha=0.85, label="A-joint INT8_tc (reached)")
    ax.scatter(sp["fp16"][0], sp["fp16"][1], s=42, marker="x",
               color="#d95f0e", label="A-serial fp16")
    ax.scatter(sp["int8"][0], sp["int8"][1], s=60, marker="P",
               color="#d95f0e", edgecolors="k", label="A-serial INT8_tc")

    for k, p in enumerate(q_pairs):
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        pg_lat = qlut.int8_lat(lut.latency(pg, "tuned"), pg, sched="tuned")
        pg_ap = apm.ap70(pg) - 0.008
        real_pg = qlut.has_direct_h800(pg, "tuned")
        real_wg = qlut.has_direct_h800(wg, "tuned")
        ax.scatter([pg_lat], [pg_ap], s=320, marker="*", color="#2c7fb8",
                   edgecolors="k", zorder=6,
                   label=("P_g tuned+INT8_tc (real-caliber)" if k == 0 else None))
        ax.annotate(
            f"P_g={list(pg)} int8_tc={'REAL' if real_pg else 'unmeasured'}\n"
            f"W_g={list(wg)} int8_tc={'REAL' if real_wg else 'unmeasured'}",
            (pg_lat, pg_ap), textcoords="offset points", xytext=(10, -38),
            fontsize=7.2, color="#2c7fb8",
            arrowprops=dict(arrowstyle="->", color="#2c7fb8"))

    ax.set_xscale("log")
    ax.set_xlabel("latency (µs, log)  —  INT8_tc = REAL per-width H800 measured "
                  "ratio where available, else unmeasured/neutral")
    ax.set_ylabel("AP70")
    ax.set_title("P×Q×S REAL CALIBER: int8_tc coverage 4/9 widths "
                 "(base/p50/p75/trap25); no dp4a buildability wall (2026-07-03)")
    ax.legend(fontsize=7.2, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(n_seeds=12, budget=60, pop=8, verbose=True,
        fallback_proxy: float | None = None,
        legacy_buildability_wall: bool = False):
    lut = LatencyLUT()
    apm = APModel()
    qlut = RealQLookup()
    key_scale = 1

    # Real-caliber defaults: no categorical buildability wall (2026-07-03: int8_tc
    # builds everywhere measured so far, incl. s0=48/32/16), no uniform-constant
    # fallback for the 5 uncovered widths (honest: unmeasured -> neutral fp16
    # latency, no assumed speedup) unless explicitly requested.
    qlut.enforce_int8_buildable = legacy_buildability_wall
    qlut.uniform_int8_speedup = fallback_proxy

    grid = candidate_widths(lut, apm)
    seed_grid = load_seed_grid(key_scale=key_scale)
    hv_ref = compute_hv_ref_pqs(grid, lut, apm, qlut)

    ps_pairs = detect_wg_pg_pairs(grid, lut, apm)
    q_pairs = detect_q_rank_flip_pairs(grid, lut, apm, qlut, sched="tuned")

    # label map: seed_grid (8 probe points) UNION real_ratio json's own labels
    # (covers the 4 widths -- mix_b/mix_d/s1_64/s2_128 -- that ARE in the
    # searchable candidate_widths() grid but NOT in the 8-point seed_grid probe
    # file, so seed_grid.get() alone would print raw tuples for them).
    label_map: dict[tuple, str] = {w: v["label"] for w, v in seed_grid.items()}
    for e in qlut._real_ratio.values():
        label_map[tuple(int(x) for x in e["num_filters"])] = e["label"]
    for e in qlut.missing_widths():
        wk = tuple(int(x) for x in e.get("num_filters", e.get("width", [])))
        if wk:
            label_map[wk] = e.get("label", str(wk))

    def lbl(w):
        return label_map.get(tuple(w), str(tuple(w)))

    build_table = [{
        "width": list(w),
        "label": lbl(w),
        "s0": int(w[0]), "in_per_g": qlut.in_per_g(w),
        "int8_buildable_legacy_dp4a_rule": qlut.can_build_int8(w),
        "int8_tc_real_measured": qlut.has_direct_h800(w, "tuned"),
        "int8_tc_real_ratio": (qlut.real_ratio_record(w) or {}).get("ratio_int8tc_over_fp16"),
    } for w in grid]

    real_lut = lut.mode != "seed"
    real_ap = apm.mode == "b2"
    covered_labels = [b["label"] for b in build_table if b["int8_tc_real_measured"]]
    missing_labels = [b["label"] for b in build_table if not b["int8_tc_real_measured"]]

    if verbose:
        print("=" * 84)
        print("P×Q×S THREE-AXIS ABLATION DRIVER -- REAL CALIBER (int8_tc)")
        print("=" * 84)
        print(f"grid ({len(grid)} widths): " + ", ".join(lbl(w) for w in grid))
        print(f"LUT mode={lut.mode} (real={real_lut})  AP mode={apm.mode} (real={real_ap})")
        print(f"INT8_tc real-measured widths (4/9 expected): {covered_labels}")
        print(f"INT8_tc UNMEASURED widths (fallback_proxy={fallback_proxy}): {missing_labels}")
        print(f"enforce_int8_buildable(legacy dp4a wall)={legacy_buildability_wall} "
              "(real-caliber default: False -- 2026-07-03 int8_tc builds everywhere)")
        print(f"ref_hv={hv_ref}  seeds={n_seeds} budget={budget}\n")

    hv = {a: [] for a in ("A-joint-PQS", "A-noS-PQS", "A-serial-PQS")}
    conv = {a: [] for a in hv}
    joint_visited, serial_visited, nos_visited = set(), set(), set()
    rep = {}

    for sd in range(n_seeds):
        rj = run_joint_pqs(lut, apm, qlut, grid, budget, random.Random(sd), hv_ref, pop)
        rn = run_noS_pqs(lut, apm, qlut, grid, budget, random.Random(sd), hv_ref, pop)
        rs = run_serial_pqs(lut, apm, qlut, grid, budget, random.Random(sd), hv_ref, pop)
        hv["A-joint-PQS"].append(rj.final_hv); conv["A-joint-PQS"].append(rj.cost.conv_log)
        hv["A-noS-PQS"].append(rn.final_hv);   conv["A-noS-PQS"].append(rn.cost.conv_log)
        hv["A-serial-PQS"].append(rs.final_hv); conv["A-serial-PQS"].append(rs.cost.conv_log)
        joint_visited.update(rj.visited)
        nos_visited.update(rn.visited)
        serial_visited.update(rs.visited)
        if sd == 0:
            rep = {"A-joint-PQS": rj, "A-noS-PQS": rn, "A-serial-PQS": rs}

    # ---- statistics ----
    wilcox = wilcoxon_signed_rank(hv["A-joint-PQS"], hv["A-serial-PQS"])
    conv_agg = {a: aggregate_convergence(conv[a]) for a in conv}

    # ---- lat_source provenance tally (real / proxy / unmeasured-neutral / other)
    # across every (width,sched,quant) point the three arms actually evaluated
    # this run -- this is the concrete "distinguish real/proxy/estimated" answer.
    lat_source_tally: dict[str, int] = {}
    for arm_res in rep.values():
        for r in arm_res.cost.eval_log:
            lat_source_tally[r["lat_source"]] = lat_source_tally.get(r["lat_source"], 0) + 1

    # ---- categorical Q-claim (legacy view; kept for continuity/comparison,
    # NOT the real-caliber headline claim -- see quantitative_q_claim below) ----
    def reaches(visited, width, quant):
        return any(w == tuple(width) and q == quant for (w, s, q) in visited)

    per_pair = []
    for p in ps_pairs:
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        per_pair.append({
            "wg": list(wg), "pg": list(pg), "ap70": p["ap70"],
            "wg_in_per_g": qlut.in_per_g(wg), "pg_in_per_g": qlut.in_per_g(pg),
            "wg_int8_buildable_legacy_dp4a_rule": qlut.can_build_int8(wg),
            "pg_int8_buildable_legacy_dp4a_rule": qlut.can_build_int8(pg),
            "wg_int8_tc_real_measured": qlut.has_direct_h800(wg, "tuned"),
            "pg_int8_tc_real_measured": qlut.has_direct_h800(pg, "tuned"),
            "joint_reaches_pg_int8": reaches(joint_visited, pg, "int8"),
            "serial_reaches_pg_int8": reaches(serial_visited, pg, "int8"),
            "ps_iso_ap_latency_ratio": p["iso_ap_latency_ratio"],
        })
    cat_pass = bool(ps_pairs) and all(
        (not pp["wg_int8_buildable_legacy_dp4a_rule"]) and pp["pg_int8_buildable_legacy_dp4a_rule"]
        and pp["joint_reaches_pg_int8"] and not pp["serial_reaches_pg_int8"]
        for pp in per_pair)
    serial_pareto = rep["A-serial-PQS"].pareto if rep else []
    serial_misaligned_int8_on_pareto = any(
        r["quant"] == "int8" and not qlut.can_build_int8(tuple(r["width"]))
        for r in serial_pareto)
    serial_locked = [tuple(w) for w in (rep["A-serial-PQS"].locked_widths or [])] if rep else []
    serial_locked_all_unbuildable = bool(serial_locked) and all(
        not qlut.can_build_int8(w) for w in serial_locked)
    n_joint_int8_widths = len({w for (w, s, q) in joint_visited
                               if q == "int8" and qlut.can_build_int8(w)})
    n_serial_int8_widths = len({w for (w, s, q) in serial_visited
                                if q == "int8" and qlut.can_build_int8(w)})

    # ---- quantitative Q-claim (the real-caliber headline): per-width REAL
    # speedup ratio, non-uniform, from detect_q_rank_flip_pairs() fed with
    # RealQLookup (source field distinguishes real-measured vs proxy/other). ----
    real_ratio_summary = [
        {"label": b["label"], "width": b["width"],
         "ratio_int8tc_over_fp16": b["int8_tc_real_ratio"]}
        for b in build_table if b["int8_tc_real_measured"]
    ]

    # ---- plots ----
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p_box = FIGDIR / "pqs_hv_boxplot_real_v1.png"
    p_par = FIGDIR / "pqs_pareto_int8_real_v1.png"
    plot_hv_boxplot(hv, hv_ref_scalar := _ref_hv_scalar(grid, lut, apm, qlut, hv_ref), p_box)
    plot_pareto_int8_real(joint_visited, serial_visited, apm, lut, qlut,
                          q_pairs, hv_ref, p_par)

    # ---- JSON ----
    def stats(v):
        a = np.array(v)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "pct_of_joint": float(a.mean() / np.mean(hv["A-joint-PQS"]) * 100)}

    missing_widths_report = qlut.missing_widths()

    results = {
        "meta": {
            "experiment": "P×Q×S three-axis coupling ablation -- REAL CALIBER (int8_tc)",
            "caliber": "real_v1",
            "proxy_sibling": "framework/run_pqs_ablation.py (untouched; HV 56.7/85.3/100.0, "
                             "Wilcoxon p=4.88e-4)",
            "data_status": "real" if (real_lut and real_ap) else "placeholder_seed_fallback",
            "int8_backend": "h800_tvm_int8_rewritten_tensorcore (im2col+MMA, "
                            "tensorcore_gate=true; NOT native/untensorized SIMT int8)",
            "int8_latency_model": (
                "per-width REAL measured ratio (fp16_lat_tuned_ms/int8tc_lat_tuned_ms, "
                "same SMBO batch) applied multiplicatively to the ablation's own "
                "LatencyLUT fp16 value, for the 4/9 grid widths with a real "
                f"measured pair (source: {REAL_RATIO_JSON.relative_to(ROOT)}); "
                f"remaining 5/9 widths: {'fallback proxy=' + str(fallback_proxy) if fallback_proxy else 'NO assumed speedup (neutral fp16 latency, honestly unmeasured)'}"
            ),
            "int8_tc_coverage": {
                "n_grid": len(grid), "n_covered": len(covered_labels),
                "n_missing": len(missing_labels),
                "covered_labels": covered_labels, "missing_labels": missing_labels,
            },
            "real_ratio_summary": real_ratio_summary,
            "int8_ap_delta": -0.008,
            "int8_ap_source": "historical_trt_evidence:median_DAIR_val_1789_int8_minmax_ap_delta "
                              "-- STILL A PROXY under real caliber; AP has not been "
                              "re-measured under the int8_tc backbone. Open gap.",
            "structural_constraint": "in_per_g = s0//16 % 4 (legacy dp4a NCHWc pack-4 rule)",
            "enforce_int8_buildable_legacy_dp4a_wall": legacy_buildability_wall,
            "categorical_claim_applicable": legacy_buildability_wall,
            "caveat": (
                "Real-caliber default (enforce_int8_buildable_legacy_dp4a_wall=False): "
                "the CATEGORICAL Q-coupling claim (structural buildability wall) is "
                "NOT APPLICABLE -- 2026-07-03 real int8_tc measurements build "
                "successfully on every width tried, including s0=48 (trap25: "
                "build_success=true, tensorcore_gate=true, wmma_count=180), which the "
                "legacy dp4a rule called unbuildable. See structural_q_claim below for "
                "the legacy-rule view (kept for continuity, not the headline claim) and "
                "quantitative_q_claim / real_ratio_summary for the real, non-uniform "
                "per-width speedup story (1.05x-1.23x across 4 covered widths, vs the "
                "single 1.449x proxy constant)."
            ),
            "lat_source_tally": lat_source_tally,
            "missing_measurements_file": str((RESULTS / 'pqs_ablation_real_v1_missing_measurements.json').relative_to(ROOT)),
        },
        "int8_buildability_table": build_table,
        "q_rank_flip_pairs": [{
            "wg": list(p["wg"]), "pg": list(p["pg"]), "ap70": p["ap70"],
            "wg_int8_us": p["wg_int8_us"], "pg_int8_us": p["pg_int8_us"],
            "q_rank_flip_ratio": p["q_rank_flip_ratio"], "source": p["source"],
        } for p in q_pairs],
        "ps_rank_flip_pairs": [{
            "wg": list(p["wg"]), "pg": list(p["pg"]), "ap70": p["ap70"],
            "iso_ap_latency_ratio": p["iso_ap_latency_ratio"],
        } for p in ps_pairs],
        "hv_distribution": {a: stats(hv[a]) for a in hv},
        "hv_raw": {a: hv[a] for a in hv},
        "wilcoxon_joint_vs_serial": wilcox,
        "structural_q_claim": {
            "applicable": legacy_buildability_wall,
            "note": ("This block reproduces the PROXY version's categorical logic "
                     "verbatim (legacy in_per_g%4==0 dp4a rule) for continuity/"
                     "comparison. It is NOT the real-caliber headline result unless "
                     "--legacy-buildability-wall was passed. See quantitative_q_claim."),
            "n_pairs": len(ps_pairs),
            "per_pair": per_pair,
            "categorical_pass": cat_pass,
            "serial_misaligned_int8_on_pareto": serial_misaligned_int8_on_pareto,
            "serial_locked_widths": [list(w) for w in serial_locked],
            "serial_locked_all_int8_unbuildable": serial_locked_all_unbuildable,
            "n_joint_int8_widths": n_joint_int8_widths,
            "n_serial_int8_widths": n_serial_int8_widths,
        },
        "quantitative_q_claim": {
            "note": ("Real-caliber headline: per-width int8_tc/fp16 speedup is REAL "
                     "and NON-UNIFORM (not a single constant). Built from "
                     "detect_q_rank_flip_pairs() (unmodified, in search_three_arm.py) "
                     "fed with RealQLookup; 'source' distinguishes real-measured vs "
                     "proxy/other for each detected W_g/P_g pair. real_ratio_summary "
                     "gives the plain per-width ratio for all 4 covered widths "
                     "regardless of whether they form a detected rank-flip pair."),
            "q_rank_flip_pairs_source_breakdown": {
                src: sum(1 for p in q_pairs if p["source"] == src)
                for src in {p["source"] for p in q_pairs}
            } if q_pairs else {},
            "real_ratio_summary": real_ratio_summary,
        },
        "visited_int8": {
            "A-joint-PQS": sorted(
                [lbl(w), s] for (w, s, q) in joint_visited if q == "int8"),
            "A-serial-PQS": sorted(
                [lbl(w), s] for (w, s, q) in serial_visited if q == "int8"),
        },
        "figures": {"hv_boxplot": str(p_box), "pareto_int8": str(p_par)},
    }
    out_json = RESULTS / "pqs_ablation_results_real_v1.json"
    out_json.write_text(json.dumps(results, indent=2))

    missing_out = RESULTS / "pqs_ablation_real_v1_missing_measurements.json"
    missing_out.write_text(json.dumps({
        "note": "Widths in the P×Q×S ablation grid lacking a real (fp16,int8_tc) "
                "measured pair. Run these on H800 (idle GPU only -- confirm via "
                "nvidia-smi, util 0%/mem<=50MiB before latency/energy measurement) "
                "then rerun scripts/phase2/build_q_int8tc_real_ratios_v1.py to "
                "refresh results/q_int8tc_real_ratios_v1.json, then rerun this driver.",
        "missing": missing_widths_report,
    }, indent=2))

    if verbose:
        _print_report(results)
        print(f"\nartifacts:\n  {out_json}\n  {missing_out}\n  {p_box}\n  {p_par}")
    return results


def _print_report(r: dict) -> None:
    print("=" * 84)
    print("REAL-CALIBER RESULTS")
    print("=" * 84)
    for a, s in r["hv_distribution"].items():
        print(f"  {a:14s} mean={s['mean']:.4e} pct_of_joint={s['pct_of_joint']:.1f}% "
              f"std={s['std']:.2e} [{s['min']:.3e},{s['max']:.3e}]")
    w = r["wilcoxon_joint_vs_serial"]
    print(f"\nWilcoxon A-joint vs A-serial:")
    print(f"  method={w['method']} stat={w['statistic']} p={w['p_value']} "
          f"rank-biserial={w['effect_size_rank_biserial']:.3f} "
          f"mean_diff={w['mean_diff']:.3e}")
    print("\n--- int8_tc real-measurement coverage ---")
    cov = r["meta"]["int8_tc_coverage"]
    print(f"  covered {cov['n_covered']}/{cov['n_grid']}: {cov['covered_labels']}")
    print(f"  MISSING {cov['n_missing']}/{cov['n_grid']}: {cov['missing_labels']} "
          f"(see {r['meta']['missing_measurements_file']})")
    print("\n--- quantitative Q-claim: real per-width int8_tc/fp16 ratio ---")
    for rr in r["meta"]["real_ratio_summary"]:
        print(f"  {rr['label']:8s} {rr['width']}: ratio={rr['ratio_int8tc_over_fp16']}")
    print(f"\n--- categorical claim (legacy dp4a view; applicable="
          f"{r['structural_q_claim']['applicable']}) ---")
    sc = r["structural_q_claim"]
    for pp in sc["per_pair"]:
        print(f"  W_g={pp['wg']}(int8_tc_real={pp['wg_int8_tc_real_measured']}) "
              f"/ P_g={pp['pg']}(int8_tc_real={pp['pg_int8_tc_real_measured']}) "
              f"@AP{pp['ap70']}: joint→(P_g,int8)={pp['joint_reaches_pg_int8']} "
              f"serial→(P_g,int8)={pp['serial_reaches_pg_int8']}")
    print(f"\n  legacy categorical_pass={sc['categorical_pass']} "
          f"(reference only unless legacy_buildability_wall=True)")
    print(f"\n--- lat_source provenance tally (seed 0, all 3 arms) ---")
    for src, n in sorted(r["meta"]["lat_source_tally"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:4d}  {src}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="P×Q×S three-axis ablation driver -- REAL CALIBER (int8_tc)")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--fallback-proxy", type=float, default=None,
                    help="if set, apply this uniform speedup to the 5 widths "
                         "WITHOUT a real int8_tc measurement (degraded/best-effort "
                         "full-grid mode). Default: none -- those widths get NO "
                         "assumed int8 speedup (honest, neutral fp16 latency).")
    ap.add_argument("--legacy-buildability-wall", action="store_true",
                    help="force the legacy dp4a in_per_g%%4==0 categorical "
                         "buildability rule (pre-2026-07-03 view), for a "
                         "side-by-side comparison against the proxy script's "
                         "structural_q_claim. Real-caliber default: off.")
    args = ap.parse_args()
    run(n_seeds=args.seeds, budget=args.budget, pop=args.pop,
        verbose=not args.quiet, fallback_proxy=args.fallback_proxy,
        legacy_buildability_wall=args.legacy_buildability_wall)
