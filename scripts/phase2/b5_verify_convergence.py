"""B5 — verify each arm's convergence solution is a REAL-measured point.

The three-arm search (B3/B4) runs on a cost model whose latency LUT and AP
model are BOTH real measurements:
  * latency  -> results/latency_lut_pyramid.json  (direct H800 TVM grid:
               gap1_grid_corrected + lut_results_grid relaunch, real default/tuned us)
  * AP70     -> results/ap70_model_pyramid.json `table` (stage_a DepGraph fp16 +
               DepGraph expansion finetune; weight-identity within each pair)

So "B5 real-test" = AUDIT that every config on each arm's converged Pareto is
backed by a real-measured (latency, AP70) datapoint — NOT a model extrapolation
or interpolation — and confirm the convergence story:
   A-serial converges to W_g (misaligned-s0, tunes ~2x, slow) at each pair AP,
   A-joint converges to P_g (aligned-s0, tunes ~7.9x, fast) at the same AP,
   headline = iso-AP70 latency ratio (W_g_tuned / P_g_tuned).

Pure-Python audit, no GPU. Writes results/b5_convergence_verification.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from framework.search_three_arm import (
    APModel, LatencyLUT, candidate_widths, detect_wg_pg_pairs, compute_hv_ref,
    run_joint, run_serial, run_noS,
)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"


def _real_latency(lut: LatencyLUT, w, sched) -> dict:
    """Is this (width, sched) latency a directly-measured grid point?"""
    w = tuple(int(x) for x in w)
    direct = w in lut.direct
    key = "default_us" if sched == "default" else "tuned_us"
    return {"width": list(w), "sched": sched,
            "real_measured": bool(direct),
            "source": "lut.direct (H800 TVM grid)" if direct else "NOT-direct",
            "lat_us": (lut.direct[w][key] if direct else lut.latency(w, sched))}


def _real_ap(apm: APModel, w) -> dict:
    """Is this width's AP70 a real finetuned anchor (exact), not interpolated?"""
    w = tuple(int(x) for x in w)
    exact = w in apm.exact
    return {"width": list(w), "real_measured": bool(exact),
            "source": "apm.exact (stage_a / DepGraph finetune)" if exact
                      else "INTERPOLATED",
            "ap70": apm.ap70(w)}


def main() -> None:
    lut = LatencyLUT(); apm = APModel()
    grid = candidate_widths(lut, apm)
    pairs = detect_wg_pg_pairs(grid, lut, apm)
    hv_ref = compute_hv_ref(grid, lut, apm)

    # converged solutions of each arm (representative seed=0, generous budget)
    rj = run_joint(lut, apm, grid, 60, random.Random(0), hv_ref, 8)
    rs = run_serial(lut, apm, grid, 60, random.Random(0), hv_ref, 8)
    rn = run_noS(lut, apm, grid, 60, random.Random(0), hv_ref, 8)

    # ---- audit 1: every converged-Pareto config is real-measured ----
    audit = {}
    for arm, res in (("A-joint", rj), ("A-serial", rs), ("A-noS", rn)):
        rows = []
        for r in res.pareto:
            lat = _real_latency(lut, r["width"], r["sched"])
            ap = _real_ap(apm, r["width"])
            rows.append({**lat, "ap70": ap["ap70"],
                         "ap_real": ap["real_measured"], "ap_source": ap["source"],
                         "lat_source": lat["source"]})
        all_real = all(x["real_measured"] and x["ap_real"] for x in rows)
        audit[arm] = {"all_configs_real_measured": all_real, "pareto": rows}

    # ---- audit 2: per-pair convergence story (W_g vs P_g at iso-AP70) ----
    # global tuned Pareto over the full grid (for reporting/classification only)
    grec = [{"width": w, "sched": "tuned", "ap70": apm.ap70(w),
             "lat_us": lut.latency(w, "tuned")} for w in grid]
    from framework.search_three_arm import _front_records
    global_front = {tuple(r["width"]) for r in _front_records(grec)}

    def arm_best_tuned_at_ap(res, ap_level, tol=1e-4):
        """Lowest-latency TUNED config the arm ships at this AP70 level."""
        best = None
        for r in res.pareto:
            if abs(r["ap70"] - ap_level) <= tol and r["sched"] == "tuned":
                if best is None or r["lat_us"] < best["lat_us"]:
                    best = r
        return best

    pair_story = []
    for p in pairs:
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        j = arm_best_tuned_at_ap(rj, p["ap70"])
        s = arm_best_tuned_at_ap(rs, p["ap70"])
        # A "shipped co-design win" requires P_g to be on the GLOBAL tuned Pareto
        # (else P_g is dominated by a higher-AP width and never shipped by anyone
        #  — the pair then only demonstrates the rank-flip MECHANISM).
        pg_on_global = pg in global_front
        joint_ships_pg = bool(j and tuple(j["width"]) == pg)
        serial_ships_wg = bool(s and tuple(s["width"]) == wg)
        kind = ("shipped_codesign_win" if (pg_on_global and joint_ships_pg
                                           and serial_ships_wg)
                else "mechanism_only")
        pair_story.append({
            "ap70": p["ap70"], "wg": list(wg), "pg": list(pg),
            "kind": kind,
            "pg_on_global_tuned_pareto": pg_on_global,
            "joint_ships": (list(j["width"]) if j else None),
            "joint_ships_lat_us": (round(j["lat_us"], 1) if j else None),
            "joint_converges_to_Pg": joint_ships_pg,
            "serial_ships": (list(s["width"]) if s else None),
            "serial_ships_lat_us": (round(s["lat_us"], 1) if s else None),
            "serial_converges_to_Wg": serial_ships_wg,
            "iso_ap70_latency_ratio": p["iso_ap_latency_ratio"],
            "wg_tuned_us": p["wg_tuned_us"], "pg_tuned_us": p["pg_tuned_us"],
        })

    wins = [ps for ps in pair_story if ps["kind"] == "shipped_codesign_win"]
    story_ok = len(wins) >= 1          # >=1 clean shipped win suffices for headline
    all_real = all(audit[a]["all_configs_real_measured"] for a in audit)

    out = {
        "summary": {
            "all_converged_configs_real_measured": all_real,
            "has_shipped_codesign_win": story_ok,
            "headline_shipped_ratios": [ps["iso_ap70_latency_ratio"] for ps in wins],
            "mechanism_only_ratios": [ps["iso_ap70_latency_ratio"] for ps in pair_story
                                      if ps["kind"] == "mechanism_only"],
            "n_pairs": len(pairs), "n_shipped_wins": len(wins),
            "note": ("Headline = shipped co-design wins (P_g on global Pareto, "
                     "joint ships it, serial ships W_g). mechanism_only pairs show "
                     "the rank-flip + structural exclusion but P_g is globally "
                     "dominated (e.g. pad64 dominated by s1_64) so not a shipped ratio."),
        },
        "provenance": {
            "latency": "results/latency_lut_pyramid.json — direct H800 TVM grid "
                       "(gap1_grid_corrected + lut_results_grid relaunch), real default/tuned us",
            "ap70": "results/ap70_model_pyramid.json `table` — stage_a DepGraph fp16 "
                    "anchors + DepGraph expansion finetune (mix_b/s1_64, mix_d); "
                    "within-pair AP equality by zero-pad weight-identity",
            "note": "No GPU re-run needed: convergence solutions ARE the real-measured "
                    "grid points. For maximal rigor one may re-tune each shipped config "
                    "once more on H800 (fresh workdir, builder timeout=300) — optional.",
        },
        "real_measured_audit": audit,
        "per_pair_convergence_story": pair_story,
    }
    out_path = RESULTS / "b5_convergence_verification.json"
    out_path.write_text(json.dumps(out, indent=2))

    # ---- console ----
    print("=" * 78)
    print("B5 — convergence-solution real-measured audit")
    print("=" * 78)
    print(f"all converged configs real-measured : {all_real}")
    print(f"has >=1 shipped co-design win : {story_ok}")
    print(f"headline shipped ratios : {out['summary']['headline_shipped_ratios']}")
    print(f"mechanism-only ratios   : {out['summary']['mechanism_only_ratios']}\n")
    for ps in pair_story:
        print(f"  [{ps['kind']}] @AP70={ps['ap70']}: "
              f"A-serial→{ps['serial_ships']} ({ps['serial_ships_lat_us']}µs, =W_g "
              f"{ps['serial_converges_to_Wg']}) vs A-joint→{ps['joint_ships']} "
              f"({ps['joint_ships_lat_us']}µs, =P_g {ps['joint_converges_to_Pg']}) "
              f"| P_g on global Pareto={ps['pg_on_global_tuned_pareto']} "
              f"→ {ps['iso_ap70_latency_ratio']}×")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
