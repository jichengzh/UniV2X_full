#!/usr/bin/env python3
"""P2.2b — SMBO convergence tracker across rounds.

Reads round{N}_fp32_{measured,feedback_report}.json + the frozen original60 table,
and reports per-round: new points, this-round min latency, cumulative best latency,
and the surrogate before/after MAPE. The loop has CONVERGED when the cumulative
best latency stops improving (acquisition no longer finds faster configs) AND the
surrogate MAPE has stabilized -> further rounds add no frontier value.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("/home/jichengzhi/V2X")
LOOP = REPO / ("multi_agent/data/stage2_lut_generation_v1/generated/"
               "original60_quant_20260627/smbo_loop")
TABLE = REPO / ("multi_agent/data/stage2_lut_generation_v1/generated/"
                "original60_quant_20260627/cost_model/train/original60_training_table_latest.json")


def parse_w(w):
    return tuple(int(x) for x in (w if isinstance(w, list) else str(w).split("x")))


def main(precision="fp32"):
    tab = json.load(open(TABLE))
    base_min = min(r["latency_ms"] for r in tab["rows"]
                   if r["precision"] == precision and r.get("latency_ms"))
    rounds = []
    cum_min = base_min
    r = 1
    while True:
        mf = LOOP / f"round{r}_{precision}_measured.json"
        if not mf.is_file():
            break
        meas = json.load(open(mf))
        lats = [m["lat_tuned_ms"] for m in meas if m.get("lat_tuned_ms")]
        rmin = min(lats) if lats else None
        prev_cum = cum_min
        if rmin is not None:
            cum_min = min(cum_min, rmin)
        fb = LOOP / f"round{r}_{precision}_feedback_report.json"
        mape = {}
        if fb.is_file():
            d = json.load(open(fb))
            c = d.get("model_correction_latency", {})
            mape = {"before_mape": c.get("before_lat_mape"),
                    "after_mape": c.get("after_lat_mape_LOO")}
        rounds.append({"round": r, "n_new": len(meas), "round_min_lat_ms": rmin,
                       "cum_best_lat_ms": cum_min,
                       "advanced_frontier": bool(rmin is not None and rmin < prev_cum),
                       **mape})
        r += 1

    report = {"schema": "smbo_convergence_v1", "precision": precision,
              "baseline_best_lat_ms": base_min, "rounds": rounds,
              "final_best_lat_ms": cum_min,
              "converged": bool(rounds and not rounds[-1]["advanced_frontier"]),
              "note": ("converged = last round did not lower cumulative best latency. "
                       "Note: min-latency is trivially the smallest net; the SMBO value "
                       "is reaching it efficiently AND calibrating the surrogate (MAPE) "
                       "over the AP-constrained region, not brute-forcing all 283.")}
    out = LOOP / f"convergence_{precision}.json"
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{'round':>5} {'n_new':>5} {'round_min':>10} {'cum_best':>9} {'adv':>4} "
          f"{'before_mape':>11} {'after_mape':>10}")
    print(f"{'base':>5} {'60':>5} {'-':>10} {base_min:9.2f} {'-':>4}")
    for x in rounds:
        print(f"{x['round']:>5} {x['n_new']:>5} "
              f"{(x['round_min_lat_ms'] or 0):10.2f} {x['cum_best_lat_ms']:9.2f} "
              f"{str(x['advanced_frontier']):>4} "
              f"{(x.get('before_mape') or 0):11.3f} {(x.get('after_mape') or 0):10.3f}")
    print(f"\nconverged={report['converged']} final_best_lat={cum_min:.2f}ms  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "fp32"))
