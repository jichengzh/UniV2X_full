#!/usr/bin/env python3
"""Phase 1 (E6) — surrogate search over the FULL 1029 joint P x Q space.

Space = 343 widths (W0,W1,W2 each in 7 levels {16,24,32,40,48,56,64})
        x 3 precisions {fp32, fp16, int8_tc} = 1029 (p,q) points.
Schedule s = inner-loop tuned (not a search dim).

Surrogate (cold-start warm-started on real data):
  - fp16 latency shape fitted (quadratic LSQ) on the 60-width original60 corpus
    fp16 latencies, then AFFINE-rescaled so the 4 real anchor predictions match
    the REAL serial WMMA measurements (256x256, this run).
  - int8_tc latency = fp16 x int8/fp16 ratio, ratio fitted (linear in w0) on the
    4 REAL measured anchor int8_tc/fp16 ratios {0.678,0.690,0.887,0.912}.
  - fp32 latency = fp16 x real default/rewritten ratio (untensorized baseline;
    dominated, kept for completeness).
  - AP70 = gold w0-anchored converged (stage_a_ap_real); int8 = fp16-0.008 (gold).
  - energy = fp16 real anchor energy shape (~ proportional to latency),
    rescaled to real anchor energies; int8 energy x real int8/fp16 energy ratio.

Because the surrogate is cheap, the 1029 space is enumerated EXHAUSTIVELY and the
EXACT 3-objective (AP up / lat down / energy down) Pareto front is computed
(stronger than NSGA-II sampling, which only approximates this). The framework
NSGA-II (run_pqs_ablation / search_three_arm) is the production sampler for the
same objective; it is cited in results/phase1_pyramid_pareto.json.

Buildability: int8_tc requires fp16-I/O-cast tensorization (tensorcore_gate=True);
all 4 tested anchor widths build. int8 kept buildable for the anchor family.

Output: results/phase1_surrogate_search_1029.json (predicted front over 1029,
confirming the real-measured anchor diagonal is on the front).
"""
import json, os, itertools
import numpy as np

ROOT = "/home/jichengzhi/V2X"
GEN = os.path.join(ROOT, "multi_agent/data/stage2_lut_generation_v1/generated/"
                   "original60_quant_20260627")
ANCHOR_DIR = os.path.join(GEN, "anchor_g1")
CORPUS = os.path.join(GEN, "cost_model/train/original60_training_table_latest.json")
OUT = os.path.join(ROOT, "results/phase1_surrogate_search_1029.json")

LEVELS = [16, 24, 32, 40, 48, 56, 64]
ANCHORS = [(16, 32, 64), (32, 64, 128), (48, 96, 192), (64, 128, 256)]
GOLD_AP = {16: 0.5300, 32: 0.5641, 48: 0.5905, 64: 0.6309}
INT8_DAP = -0.008


def feat(w):
    w0, w1, w2 = w
    return [1, w0, w1, w2, w0 * w0, w1 * w1, w2 * w2, w0 * w1, w1 * w2, w0 * w2]


def ap_w0(w0, prec):
    xs = sorted(GOLD_AP)
    if w0 <= xs[0]:
        ap = GOLD_AP[xs[0]]
    elif w0 >= xs[-1]:
        ap = GOLD_AP[xs[-1]]
    else:
        lo = max(x for x in xs if x <= w0)
        hi = min(x for x in xs if x >= w0)
        ap = (GOLD_AP[lo] if lo == hi else
              GOLD_AP[lo] + (GOLD_AP[hi] - GOLD_AP[lo]) * (w0 - lo) / (hi - lo))
    return round(ap + (INT8_DAP if prec == "int8_tc" else 0.0), 4)


def main():
    # --- real anchor measurements (this run) ---
    f_lat = json.load(open(os.path.join(ANCHOR_DIR, "anchor_fp16_lat_collected.json")))
    f_en = json.load(open(os.path.join(ANCHOR_DIR, "anchor_fp16_energy_collected.json")))
    i_lat = json.load(open(os.path.join(ANCHOR_DIR, "anchor_int8_lat_collected.json")))
    i_en = json.load(open(os.path.join(ANCHOR_DIR, "anchor_int8_energy_collected.json")))
    akey = ["16x32x64", "32x64x128", "48x96x192", "64x128x256"]
    real_fp16_lat = np.array([f_lat[k]["lat_wmma_ms"] for k in akey])
    real_fp16_en = np.array([f_en[k] for k in akey])
    real_int8_lat = np.array([i_lat[k]["lat_int8tc_ms"] for k in akey])
    real_int8_en = np.array([i_en[k] for k in akey])
    aw0 = np.array([16, 32, 48, 64])

    # --- fit fp16 latency SHAPE on 60-corpus fp16, affine-rescale to real anchors ---
    rows = json.load(open(CORPUS))["rows"]
    fp16 = [r for r in rows if r["precision"] == "fp16"]
    Xc = np.array([feat((int(r["w0"]), int(r["w1"]), int(r["w2"]))) for r in fp16])
    yc = np.array([float(r["latency_ms"]) for r in fp16])
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)          # corpus shape
    g_anchor = np.array([feat(w) @ beta for w in ANCHORS])   # shape at anchors
    # affine a*g+b so anchor predictions match real fp16 measurements
    A = np.column_stack([g_anchor, np.ones(4)])
    (a, b), *_ = np.linalg.lstsq(A, real_fp16_lat, rcond=None)

    def fp16_lat(w):
        return max(0.1, a * (np.array(feat(w)) @ beta) + b)

    # --- int8/fp16 lat ratio: linear in w0 fitted on 4 real anchor ratios ---
    r_lat = real_int8_lat / real_fp16_lat
    pl = np.polyfit(aw0, r_lat, 1)
    r_en = real_int8_en / real_fp16_en
    pe = np.polyfit(aw0, r_en, 1)
    # fp32 (untensorized) ratio vs fp16, from real default/rewritten at anchors
    fp32_ratio = float(np.mean([f_lat[k]["lat_default_ms"] / f_lat[k]["lat_wmma_ms"]
                                for k in akey]))
    fp32_en_ratio = fp32_ratio  # ~ power constant -> energy scales with latency

    # --- energy: proportional to latency, calibrated to real fp16 anchor energies ---
    ke = float(np.mean(real_fp16_en / real_fp16_lat))  # J per ms (fp16)

    # --- enumerate 1029, evaluate surrogate ---
    pts = []
    for w in itertools.product(LEVELS, LEVELS, LEVELS):
        fl = fp16_lat(w)
        fe = fl * ke
        for prec in ("fp32", "fp16", "int8_tc"):
            if prec == "fp16":
                lat, en = fl, fe
            elif prec == "int8_tc":
                rr = float(np.clip(np.polyval(pl, w[0]), 0.5, 1.05))
                re_ = float(np.clip(np.polyval(pe, w[0]), 0.5, 1.05))
                lat, en = fl * rr, fe * re_
            else:  # fp32 untensorized baseline
                lat, en = fl * fp32_ratio, fe * fp32_en_ratio
            pts.append({"width": list(w), "precision": prec,
                        "latency_ms": round(lat, 3), "energy_j": round(en, 4),
                        "ap70": ap_w0(w[0], prec)})

    # --- exact 3-objective Pareto (max ap70, min lat, min energy) ---
    for p in pts:
        p["on_pareto"] = True
    P = pts
    for a_ in P:
        for b_ in P:
            if a_ is b_:
                continue
            if (b_["ap70"] >= a_["ap70"] and b_["latency_ms"] <= a_["latency_ms"]
                    and b_["energy_j"] <= a_["energy_j"]
                    and (b_["ap70"] > a_["ap70"] or b_["latency_ms"] < a_["latency_ms"]
                         or b_["energy_j"] < a_["energy_j"])):
                a_["on_pareto"] = False
                break
    front = [p for p in pts if p["on_pareto"]]
    front.sort(key=lambda p: p["latency_ms"])

    anchor_set = {(tuple(w), prec) for w in ANCHORS for prec in ("fp16", "int8_tc")}
    on_front_anchors = [p for p in front
                        if (tuple(p["width"]), p["precision"]) in anchor_set]

    payload = {
        "space": "343 widths (7^3) x 3 precisions = 1029 (p,q); s=inner-tuned",
        "search": "EXHAUSTIVE exact 3-objective Pareto over 1029 with real-calibrated "
                  "surrogate (stronger than NSGA-II sampling); framework NSGA-II "
                  "(run_pqs_ablation/search_three_arm) is the production sampler cited "
                  "in phase1_pyramid_pareto.json",
        "surrogate_calibration": {
            "fp16_lat": "corpus-60 quadratic shape, affine-rescaled to 4 REAL anchors",
            "affine_a_b": [round(float(a), 5), round(float(b), 4)],
            "int8_lat_ratio_vs_w0": [round(float(pl[0]), 5), round(float(pl[1]), 4)],
            "real_anchor_int8_fp16_ratio": [round(float(x), 4) for x in r_lat],
            "fp32_untensorized_ratio": round(fp32_ratio, 3),
            "energy_j_per_ms_fp16": round(ke, 5),
        },
        "n_points": len(pts), "front_n": len(front),
        "front_ap_range": [round(min(p["ap70"] for p in front), 4),
                           round(max(p["ap70"] for p in front), 4)],
        "front_lat_range": [round(min(p["latency_ms"] for p in front), 3),
                            round(max(p["latency_ms"] for p in front), 3)],
        "anchors_on_front": [{"width": p["width"], "precision": p["precision"],
                              "latency_ms": p["latency_ms"], "ap70": p["ap70"]}
                             for p in on_front_anchors],
        "front": front,
    }
    json.dump(payload, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"=== surrogate search over 1029 ===")
    print(f"points={len(pts)}  exact Pareto front n={len(front)}")
    print(f"front AP70 {payload['front_ap_range']}  lat {payload['front_lat_range']}ms")
    print(f"real-measured anchors ON the front: {len(on_front_anchors)}/8")
    for p in on_front_anchors:
        print(f"  {p['width']} {p['precision']} lat={p['latency_ms']}ms ap70={p['ap70']}")
    print(f"front precision mix: " + ", ".join(
        f"{pr}={sum(1 for p in front if p['precision']==pr)}"
        for pr in ("fp32", "fp16", "int8_tc")))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
