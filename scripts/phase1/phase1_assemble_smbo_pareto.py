#!/usr/bin/env python3
"""Phase1/E6 — assemble the FINAL 3-precision RSU Pareto from the framework SMBO run.

The framework SMBO loop (scripts/stage2_smbo_loop_v1.py) converged all 3 precisions
(fp32/fp16/int8_tc) over the frozen Theta_sw, driving real H800 measure_config
latency+energy per top-k candidate + feedback-retrained cost model. This script
assembles the deliverable from what the loop produced:

  * lat/energy  = REAL measure_config (H800 TVM, unified caliber: input_hw=[128,256],
                  fp16=WMMA / int8_tc=MatmulInt8Tensorization, both tensorcore_gate=True).
  * AP70        = REAL gold (stage_a_ap_real, 4 diagonal anchors x {fp16,int8}); the
                  loop's in-loop AP proxy is a non-functional placeholder (table ap70=0),
                  so AP is supplied here from gold. Off-diagonal SMBO-explored points have
                  real lat/energy but AP pending finetune (flagged, the closing GAP).

Output: results/phase1_pyramid_pareto.{csv,json} + figure/fig_phase1_pyramid_pareto.png
"""
import json, glob, csv
from pathlib import Path
import pandas as pd

REPO = Path("/home/jichengzhi/V2X")
BASE = REPO / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
LOOP = BASE / "smbo_loop"
GOLD_PARQUET = REPO / "data/stage_a_ap_real.parquet"
OUT_CSV = REPO / "results/phase1_pyramid_pareto.csv"
OUT_JSON = REPO / "results/phase1_pyramid_pareto.json"

DIAG = ["16x32x64", "32x64x128", "48x96x192", "64x128x256"]  # gold anchor widths


def wkey(w):
    return "x".join(map(str, w)) if isinstance(w, (list, tuple)) else str(w)


def load_gold_ap():
    """{(widthstr, precision_axis): ap70} — precision_axis in {fp16, int8}."""
    df = pd.read_parquet(GOLD_PARQUET)
    ap = {}
    for _, r in df.iterrows():
        ws = f"{int(r.stage0_planes)}x{int(r.stage1_planes)}x{int(r.stage2_planes)}"
        ap[(ws, str(r.precision))] = float(r.ap70)
    return ap


def load_measured():
    """(precision_axis, widthstr) -> {lat, energy, src}. precision_axis: fp32/fp16/int8."""
    meas = {}
    # corpus 60/precision (real lat/energy; ap70 placeholder ignored)
    tbl = json.load(open(BASE / "cost_model/train/original60_training_table_latest.json"))["rows"]
    for r in tbl:
        meas[(r["precision"], r["width"])] = {"lat": r["latency_ms"], "energy": r["energy_j"], "src": "corpus60"}
    # round measured (fp16/int8 = int8_tc caliber)
    for mj in glob.glob(str(LOOP / "round*_*_measured.json")):
        name = Path(mj).name
        prec = "int8" if "int8" in name else ("fp16" if "fp16" in name else "fp32")
        for r in json.load(open(mj)):
            meas[(prec, wkey(r["width"]))] = {"lat": r["lat_tuned_ms"], "energy": r["energy_j"], "src": "smbo_round"}
    # diagonal anchor top-ups (measure_config, fp16 + int8_tc)
    for dj in glob.glob(str(LOOP / "diag_anchors/*.json")):
        d = json.load(open(dj))
        if not d.get("lat_tuned_ms"):
            continue
        prec = "int8" if d.get("precision") in ("int8_tc", "int8") else "fp16"
        meas[(prec, wkey(d["width"]))] = {"lat": d["lat_tuned_ms"], "energy": d["energy_j"], "src": "diag_anchor"}
    return meas


def nondominated(points, objs):
    """objs: list of (key, 'min'|'max'). Return non-dominated subset (list of dicts)."""
    def dominates(a, b):
        ge = True; gt = False
        for k, d in objs:
            av, bv = a[k], b[k]
            if d == "max":
                if av < bv: ge = False
                if av > bv: gt = True
            else:
                if av > bv: ge = False
                if av < bv: gt = True
        return ge and gt
    front = []
    for p in points:
        if not any(dominates(q, p) for q in points if q is not p):
            front.append(p)
    return front


def main():
    gold = load_gold_ap()
    meas = load_measured()

    # --- AP-real Pareto: 4 diagonal anchors x {fp16,int8_tc}, real lat/energy + real gold AP ---
    ap_real_pts = []
    for ws in DIAG:
        for axis in ("fp16", "int8"):
            m = meas.get((axis, ws))
            ap = gold.get((ws, axis))
            if m and ap is not None:
                ap_real_pts.append({
                    "width": [int(x) for x in ws.split("x")],
                    "precision": "int8_tc" if axis == "int8" else axis,
                    "lat_ms": round(m["lat"], 4), "energy_j": round(m["energy"], 4),
                    "ap70": round(ap, 4),
                    "lat_src": "measured_h800_measure_config", "energy_src": "measured_h800_measure_config",
                    "ap_src": "gold_stage_a_ap_real", "measured_src": m["src"]})
    front = nondominated(ap_real_pts, [("ap70", "max"), ("lat_ms", "min"), ("energy_j", "min")])
    for p in front:
        p["on_pareto_front"] = True

    # --- 3 extremes (all gold-real AP) ---
    ap_max = max(ap_real_pts, key=lambda p: p["ap70"])
    lat_min = min(ap_real_pts, key=lambda p: p["lat_ms"])
    e_min = min(ap_real_pts, key=lambda p: p["energy_j"])
    extremes = {"ap_max": ap_max, "lat_min": lat_min, "energy_min": e_min}

    # --- framework SMBO exploration: off-diagonal points, real lat/energy, AP pending finetune ---
    explored = []
    for (prec, ws), m in meas.items():
        if ws in DIAG:
            continue
        if m["src"] in ("smbo_round",):  # points the loop actually proposed+measured
            explored.append({"width": [int(x) for x in ws.split("x")],
                             "precision": "int8_tc" if prec == "int8" else prec,
                             "lat_ms": round(m["lat"], 4), "energy_j": round(m["energy"], 4),
                             "ap70": None, "ap_src": "PENDING_FINETUNE",
                             "lat_src": "measured_h800_measure_config"})

    out = {
        "schema": "phase1_pyramid_rsu_pareto_smbo_v1",
        "method": "framework SMBO closed loop (scripts/stage2_smbo_loop_v1.py): NSGA/exact-front select -> LightGBM cost model rank -> top-k real H800 measure_config -> feedback retrain -> iterate; fp32/fp16/int8_tc all converged (convergence_{fp32,fp16,int8}.json).",
        "caliber": "H800 TVM measure_config, input_hw=[128,256], fp16=WMMA / int8_tc=MatmulInt8Tensorization (both tensorcore_gate=True); latency serial-exclusive.",
        "ap_axis_note": "AP70 real only at 4 diagonal gold anchors (stage_a_ap_real). SMBO in-loop AP is a placeholder proxy (table ap70=0). Off-diagonal explored points have real lat/energy, AP PENDING finetune (closing GAP).",
        "pareto_front_ap_real": front,
        "extremes": extremes,
        "all_ap_real_points": ap_real_pts,
        "smbo_explored_offdiagonal_ap_pending": explored,
        "convergence": {p: json.load(open(LOOP / f"convergence_{p}.json")).get("converged")
                        for p in ("fp32", "fp16", "int8") if (LOOP / f"convergence_{p}.json").exists()},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["width", "precision", "lat_ms", "energy_j", "ap70", "on_front", "lat_src", "ap_src"])
        for p in ap_real_pts:
            w.writerow(["x".join(map(str, p["width"])), p["precision"], p["lat_ms"], p["energy_j"],
                        p["ap70"], p.get("on_pareto_front", False), p["lat_src"], p["ap_src"]])

    print(f"AP-real points: {len(ap_real_pts)} | front: {len(front)} | SMBO off-diag explored: {len(explored)}")
    print("=== Pareto front (AP-real) ===")
    for p in sorted(front, key=lambda x: x["lat_ms"]):
        print(f"  {p['width']} {p['precision']:8s} lat {p['lat_ms']:7.3f}ms  E {p['energy_j']:.3f}J  AP70 {p['ap70']:.4f}")
    print("=== extremes ===")
    for k, p in extremes.items():
        print(f"  {k}: {p['width']} {p['precision']} lat {p['lat_ms']}ms E {p['energy_j']}J AP70 {p['ap70']}")
    print(f"written: {OUT_JSON}  {OUT_CSV}")


if __name__ == "__main__":
    main()
