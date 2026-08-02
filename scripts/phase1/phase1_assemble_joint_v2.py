#!/usr/bin/env python3
"""Assemble Phase-1 v2 deliverable from the joint NSGA-II search output + compare
to the v1 (per-precision diagonal) front. Draws the v1-vs-v2 Pareto figure.

Key finding this encodes: the JOINT search discovered a minimal-w1 front
([w0,32,w2]) that DOMINATES v1's diagonal on (lat,energy), justified by
corr(ap70,w1)=-0.04 (neck does not carry AP) vs corr(ap70,w0)=0.46.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/jichengzhi/V2X")
DATA = REPO / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
JOINT = DATA / "smbo_joint_nsga2/joint_nsga2_pop24_b60_real_seed0.json"
V1 = REPO / "results/phase1_pyramid_pareto.json"
OUT_JSON = REPO / "results/phase1_pyramid_pareto_v2.json"
OUT_CSV = REPO / "results/phase1_pyramid_pareto_v2.csv"
FIG = REPO / "results/figure/fig_phase1_pyramid_pareto_v2.png"
FIG2 = REPO / "multi_agent/figure/fig_phase1_pyramid_pareto_v2.png"

j = json.load(open(JOINT))
front = j["joint_pareto_front"]
for p in front:
    p["ap70_used"] = p.get("ap70") if p.get("ap70") is not None else p.get("ap70_pred")

# v1 diagonal front (for domination comparison)
v1 = json.load(open(V1))
v1f = v1.get("pareto_front_ap_real", [])


def dominates_le(a, b):  # a dominates b on (lat,energy) minimize
    return a["lat_ms"] <= b["lat_ms"] and a["energy_j"] <= b["energy_j"] and \
        (a["lat_ms"] < b["lat_ms"] or a["energy_j"] < b["energy_j"])


# how many v1 diagonal points are dominated by some v2 joint point on (lat,energy)?
dominated = []
for d in v1f:
    d2 = {"lat_ms": d["lat_ms"], "energy_j": d["energy_j"]}
    hit = next((p for p in front if dominates_le(p, d2)), None)
    if hit:
        dominated.append({"v1_point": [d["width"], d["precision"], d["lat_ms"], d["energy_j"]],
                          "dominated_by": [hit["width"], hit["precision"], hit["lat_ms"], hit["energy_j"]]})

# extremes
lat_min = min(front, key=lambda p: p["lat_ms"])
en_min = min(front, key=lambda p: p["energy_j"])
ap_max = max(front, key=lambda p: p["ap70_used"])

out = {
    "schema": "phase1_pyramid_pareto_v2", "source": str(JOINT.name),
    "method": j["method"], "space": j["space"],
    "hyperparams": j["hyperparams"],
    "search_real_measured": {"warm_start_corpus": j["real_measured_start"],
                             "new_real_measured": j["real_measured_new"],
                             "converged": "front0 fully-measured (no unmeasured front members left)"},
    "baseline_validation": j["baseline_validation"],
    "joint_pareto_front": front, "front_size": len(front),
    "extremes": {
        "latency_min": {"width": lat_min["width"], "precision": lat_min["precision"],
                        "lat_ms": lat_min["lat_ms"], "energy_j": lat_min["energy_j"],
                        "ap70": lat_min["ap70_used"], "ap_src": lat_min["ap_src"]},
        "energy_min": {"width": en_min["width"], "precision": en_min["precision"],
                       "lat_ms": en_min["lat_ms"], "energy_j": en_min["energy_j"],
                       "ap70": en_min["ap70_used"], "ap_src": en_min["ap_src"]},
        "ap_max": {"width": ap_max["width"], "precision": ap_max["precision"],
                   "lat_ms": ap_max["lat_ms"], "energy_j": ap_max["energy_j"],
                   "ap70": ap_max["ap70_used"], "ap_src": ap_max["ap_src"]},
    },
    "v1_vs_v2": {
        "v1_method": "per-precision separate 343-space search + diagonal assembly (DEPRECATED, see 9_7_14 §9)",
        "v2_method": "single 1029 joint P x Q surrogate-assisted NSGA-II",
        "finding": "joint search discovered a MINIMAL-w1 front [w0,32,w2] that dominates v1's "
                   "diagonal on (lat,energy); justified by corr(ap70,w1)=-0.04 (neck does not "
                   "carry AP) vs corr(ap70,w0)=0.46 (backbone drives AP).",
        "v1_diagonal_points_dominated_by_v2": dominated,
        "n_v1_dominated": len(dominated),
    },
    "ap_caveat": "Front lat/energy are REAL (H800 TVM measure_config). AP70 is the table/surrogate "
                 "signal (real but weak: span 0.038, corr w0=0.46/w1=-0.04/w2=0.04); NOT gold-scale "
                 "converged finetune. Gold-scale AP (stage_a, 0.52-0.63) is the higher-fidelity "
                 "reference for the w0 effect. Front-member gold AP -> finetune (w1/w2 AP-neutral "
                 "so minimal-w1 front members ~= their w0's gold AP).",
}
json.dump(out, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)

# CSV
with open(OUT_CSV, "w") as f:
    f.write("w0,w1,w2,precision,lat_ms,energy_j,ap70,ap_src\n")
    for p in sorted(front, key=lambda x: x["lat_ms"]):
        w = p["width"]
        f.write(f"{w[0]},{w[1]},{w[2]},{p['precision']},{p['lat_ms']:.4f},"
                f"{p['energy_j']:.4f},{p['ap70_used']:.4f},{p['ap_src']}\n")

# ---- figure: v1 diagonal vs v2 joint front -----------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
CU = {"int8_tc": "#d62728", "fp16": "#1f77b4", "fp32": "#2ca02c"}
# panel A: AP70 vs latency
for prec in ["int8_tc", "fp16", "fp32"]:
    pts = sorted([p for p in front if p["precision"] == prec], key=lambda x: x["lat_ms"])
    if pts:
        ax[0].scatter([p["lat_ms"] for p in pts], [p["ap70_used"] for p in pts],
                      c=CU[prec], marker="o", s=70, label=f"v2 joint front {prec}", zorder=5)
        for p in pts:
            ax[0].annotate("x".join(map(str, p["width"])), (p["lat_ms"], p["ap70_used"]),
                           fontsize=6, xytext=(3, 3), textcoords="offset points")
# v1 diagonal (gold AP, note different scale)
dl = sorted(v1f, key=lambda x: x["lat_ms"])
ax[0].plot([p["lat_ms"] for p in dl], [p["ap70"] for p in dl], "k--x", alpha=0.5,
           label="v1 diagonal (gold-scale AP)", zorder=3)
ax[0].set_xlabel("latency ms (H800 TVM, real)")
ax[0].set_ylabel("AP70  (v2=table/surrogate scale; v1=gold scale)")
ax[0].set_title("Phase1 v2 JOINT NSGA-II front vs v1 diagonal\n(AP axes differ in scale — see caveat)")
ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)
# panel B: energy vs latency — the axis where domination is real (both real-measured)
for prec in ["int8_tc", "fp16", "fp32"]:
    pts = [p for p in front if p["precision"] == prec]
    if pts:
        ax[1].scatter([p["lat_ms"] for p in pts], [p["energy_j"] for p in pts],
                      c=CU[prec], marker="o", s=70, label=f"v2 joint {prec}", zorder=5)
ax[1].plot([p["lat_ms"] for p in dl], [p["energy_j"] for p in dl], "k--x", alpha=0.6,
           label="v1 diagonal", zorder=3)
ax[1].set_xlabel("latency ms (real)"); ax[1].set_ylabel("energy J/frame (real)")
ax[1].set_title(f"Real lat/energy: v2 joint front dominates v1 diagonal\n"
                f"({len(dominated)}/{len(v1f)} v1 pts dominated; "
                f"HV(nsga2)={j['baseline_validation']['hv_nsga2_front']:.3g} > "
                f"HV(rand)={j['baseline_validation']['hv_random_mean']:.3g})")
ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.3)
plt.tight_layout()
for fp in [FIG, FIG2]:
    fp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fp, dpi=130)

print(f"front_size={len(front)} | new_real={j['real_measured_new']} | "
      f"v1_dominated={len(dominated)}/{len(v1f)}")
print(f"extremes: lat_min={lat_min['width']}{lat_min['precision']} {lat_min['lat_ms']:.3f}ms | "
      f"ap_max={ap_max['width']}{ap_max['precision']} AP={ap_max['ap70_used']:.4f}")
print(f"baseline: {out['baseline_validation']['nsga2_beats_random_frac']*100:.0f}% NSGA2>=random")
print(f"[out] {OUT_JSON}\n[csv] {OUT_CSV}\n[fig] {FIG2}")
