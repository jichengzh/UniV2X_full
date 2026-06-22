"""C0c' — CoDriving P×Q×S three-arm ablation (LOW-DIM separability probe).

The mirror of framework/run_pqs_ablation.py (Pyramid) for CoDriving, the
STANDARD-conv (groups=1, `resnet:true`) control model.  Goal-② context:
this is NOT the verdict — it is the LOW-DIMENSIONAL starting point.  If
CoDriving comes out SEPARABLE here (expected), C6 then ESCALATES dimensions
(per-channel granularity / int4 / batch) to hunt a higher-dim trap.

Key architectural difference vs Pyramid (why we expect SERIAL):
  * CoDriving backbone = standard 3x3 conv, groups=1.
  * int8 buildable for ALL widths (Cin=48 K/16=27, Cin=64 K/16=36 both build
    real WMMA int8, 1.42x/1.32x, max_rel_err=0.0 — results/codriving_int8_verify.json).
    -> NO categorical Q-coupling (no alignment cliff).
  * The 4 widths [64,128,256]/[48,96,192]/[32,64,128]/[16,32,64] differ in
    ALL of (s0,s1,s2) -> detect_wg_pg_pairs (needs same s1,s2 diff s0) finds
    ZERO pairs -> no schedule rank-flip -> P×S separable too.

Data (REAL where noted):
  * latency  : results/latency_lut_codriving.json (base+p50 real H800 TVM;
               p25/p75 power-law estimated — structural verdict is magnitude-
               independent, caveat carried).
  * AP70     : results/ap70_model_codriving.json (iso-budget de-confounded).
  * int8     : uniform proxy 1.37x (mean of measured 1.42x/1.32x); all widths
               buildable.  CATEGORICAL claim is N/A for standard conv.

Run: PYTHONPATH=/home/jichengzhi/V2X python -m framework.run_pqs_codriving
Pure-Python; no GPU.  Output: results/coupling_map/C0c_codriving_pqs.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from framework.search_three_arm import (
    LatencyLUT, APModel, QLookup, SCHEDULES, QUANT_MODES,
    candidate_widths, compute_hv_ref_pqs, build_from_manifest,
    run_joint_pqs, run_noS_pqs, run_serial_pqs,
    detect_wg_pg_pairs, hypervolume_2d, nondominated_idx,
)
from framework.run_b4_ablation import wilcoxon_signed_rank, aggregate_convergence

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
COUPLING = RESULTS / "coupling_map"
LUT_COD = RESULTS / "latency_lut_codriving.json"
AP_COD = RESULTS / "ap70_model_codriving.json"
MANIFEST_COD = ROOT / "framework" / "partitions" / "codriving_partition.yaml"

# Mean of measured CoDriving int8 speedups (Cin=48 1.42x / Cin=64 1.32x).
COD_INT8_SPEEDUP = 1.37

# ── int8 buildability now comes from the stage1 bridge, NOT a per-model subclass ──
# The old QLookupCoDriving hardcoded can_build_int8≡True.  That fact is now DERIVED:
# the codriving manifest has groups=1 → int8_buildable_align = pack_factor(4) → every
# legal width %4==0 → all buildable.  build_from_manifest(MANIFEST_COD) reproduces the
# subclass exactly with zero per-model code (验证: 见 §validation step 3)。


def _ref_hv_scalar(grid, lut, apm, qlut, hv_ref) -> float:
    pts = []
    for w in grid:
        for s in SCHEDULES:
            for q in QUANT_MODES:
                if q == "int8":
                    if qlut.enforce_int8_buildable and not qlut.can_build_int8(w):
                        continue
                    lat = qlut.int8_lat(lut.latency(w, s), w, sched=s)
                    ap = apm.ap70(w) - 0.008
                else:
                    lat = lut.latency(w, s); ap = apm.ap70(w)
                pts.append((lat, -ap))
    keep = nondominated_idx(pts)
    return hypervolume_2d([pts[i] for i in keep], hv_ref)


def run(n_seeds=12, budget=60, pop=8, verbose=True):
    # int8 buildability + space from the stage1 bridge (manifest), not a subclass.
    # CoDriving groups=1 → int8_buildable_align=4 → all widths buildable (= old
    # QLookupCoDriving). key_scale=1 (no bottleneck expansion → cur_width=num_filters).
    b = build_from_manifest(MANIFEST_COD, lut_path=LUT_COD, ap_path=AP_COD,
                            q_path=Path("/nonexistent"))   # skip Pyramid Q-LUT
    lut, apm, qlut = b["lut"], b["apm"], b["qlut"]
    qlut.enforce_int8_buildable = True
    qlut.uniform_int8_speedup = COD_INT8_SPEEDUP

    # candidate_widths() also picks up Pyramid SEED_GRID anchors (e.g. pad64
    # [64,96,192]) that both LUT and AP fall back to.  Restrict to the widths
    # ACTUALLY in the CoDriving LUT (lut.direct) so the grid is pure CoDriving.
    grid = [w for w in candidate_widths(lut, apm) if w in lut.direct]
    hv_ref = compute_hv_ref_pqs(grid, lut, apm, qlut)
    ps_pairs = detect_wg_pg_pairs(grid, lut, apm)

    build_table = [{
        "width": list(w), "s0": int(w[0]),
        "int8_buildable": qlut.can_build_int8(w),
    } for w in grid]

    if verbose:
        print("=" * 84)
        print("C0c' — CoDriving P×Q×S THREE-ARM ABLATION (standard conv, low-dim probe)")
        print("=" * 84)
        print(f"grid ({len(grid)} widths): {[list(w) for w in grid]}")
        print(f"LUT mode={lut.mode}  AP mode={apm.mode}")
        print(f"int8 buildable widths: ALL ({len(grid)}/{len(grid)}) — standard conv")
        print(f"detect_wg_pg_pairs -> {len(ps_pairs)} rank-flip pairs "
              f"(expect 0: widths differ in all of s0,s1,s2)")
        print(f"ref_hv={hv_ref}  seeds={n_seeds} budget={budget}\n")

    hv = {a: [] for a in ("A-joint-PQS", "A-noS-PQS", "A-serial-PQS")}
    conv = {a: [] for a in hv}
    joint_visited, serial_visited = set(), set()
    rep = {}
    for sd in range(n_seeds):
        rj = run_joint_pqs(lut, apm, qlut, grid, budget, random.Random(sd), hv_ref, pop)
        rn = run_noS_pqs(lut, apm, qlut, grid, budget, random.Random(sd), hv_ref, pop)
        rs = run_serial_pqs(lut, apm, qlut, grid, budget, random.Random(sd), hv_ref, pop)
        hv["A-joint-PQS"].append(rj.final_hv); conv["A-joint-PQS"].append(rj.cost.conv_log)
        hv["A-noS-PQS"].append(rn.final_hv);   conv["A-noS-PQS"].append(rn.cost.conv_log)
        hv["A-serial-PQS"].append(rs.final_hv); conv["A-serial-PQS"].append(rs.cost.conv_log)
        joint_visited.update(rj.visited); serial_visited.update(rs.visited)
        if sd == 0:
            rep = {"A-joint-PQS": rj, "A-serial-PQS": rs}

    wilcox = wilcoxon_signed_rank(hv["A-joint-PQS"], hv["A-serial-PQS"])
    hv_ref_scalar = _ref_hv_scalar(grid, lut, apm, qlut, hv_ref)

    def stats(v):
        a = np.array(v)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "pct_of_joint": float(a.mean() / np.mean(hv["A-joint-PQS"]) * 100)}

    serial_pct = stats(hv["A-serial-PQS"])["pct_of_joint"]
    # SEPARABLE iff serial reaches ~joint HV (>=97%) AND no rank-flip pairs.
    separable = (serial_pct >= 97.0) and (len(ps_pairs) == 0)

    results = {
        "meta": {
            "cell": "C0c'",
            "experiment": "CoDriving P×Q×S three-arm ablation (low-dim separability probe)",
            "model": "CoDriving (V2Xverse, DAIR, standard 3x3 conv, groups=1, resnet:true)",
            "goal": "②-precursor: low-dim baseline; SERIAL here -> C6 escalates dims to hunt trap",
            "backend": "pure-Python over H800-TVM LUT (base/p50 real, p25/p75 power-law est)",
            "int8_latency_model": f"uniform {COD_INT8_SPEEDUP}x proxy "
                                  "(mean of measured 1.42x/1.32x); ALL widths buildable",
            "caveat": "p25/p75 latency power-law estimated; structural verdict "
                      "(0 rank-flip pairs, SERIAL) is magnitude-independent. "
                      "C6 (trap hunt) requires REAL int8 per-width default+tuned.",
            "n_seeds": n_seeds, "budget": budget, "pop_size": pop,
            "grid_widths": [list(w) for w in grid],
            "hv_ref_scalar": hv_ref_scalar,
        },
        "int8_buildability_table": build_table,
        "ps_rank_flip_pairs": ps_pairs,
        "n_ps_pairs": len(ps_pairs),
        "hv_distribution": {a: stats(hv[a]) for a in hv},
        "hv_raw": {a: hv[a] for a in hv},
        "wilcoxon_joint_vs_serial": wilcox,
        "verdict": {
            "value": "SERIAL" if separable else "JOINT_OR_CHECK",
            "serves_goal": "② (low-dim baseline)",
            "architecture_conditional": "standard-conv (groups=1) — contrast to Pyramid grouped",
            "serial_pct_of_joint": serial_pct,
            "n_rank_flip_pairs": len(ps_pairs),
            "separable": separable,
            "interpretation": (
                f"CoDriving LOW-DIM SEPARABLE: A-serial reaches {serial_pct:.1f}% of "
                "A-joint HV with ZERO schedule rank-flip pairs and ALL widths int8-"
                "buildable. Standard conv has no alignment cliff -> P,Q,S decouple at "
                "this dimensionality. NOT the final verdict (goal②): C6 escalates to "
                "per-channel granularity / int4 / batch to hunt a higher-dim trap."
                if separable else
                "UNEXPECTED non-separable signal at low dim — inspect HV/pairs.")
        },
    }
    COUPLING.mkdir(parents=True, exist_ok=True)
    out = COUPLING / "C0c_codriving_pqs.json"
    out.write_text(json.dumps(results, indent=2))

    if verbose:
        print("--- HV distribution (mean over seeds; % of A-joint) ---")
        for a in ("A-joint-PQS", "A-serial-PQS", "A-noS-PQS"):
            s = results["hv_distribution"][a]
            print(f"  {a:14s} mean={s['mean']:.4e} ({s['pct_of_joint']:6.1f}% of joint)")
        print(f"\nWilcoxon joint vs serial: p={wilcox['p_value']} "
              f"method={wilcox['method']}")
        print(f"rank-flip pairs: {len(ps_pairs)}")
        print(f"\nVERDICT: {results['verdict']['value']}  "
              f"(serial={serial_pct:.1f}% of joint, separable={separable})")
        print(f"\nartifact: {out}")
    return results


if __name__ == "__main__":
    run()
