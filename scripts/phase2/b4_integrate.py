"""B4 integration: build direct-grid latency LUT + consolidated real-AP table
from gap1 (8 widths) + grid2 relaunch (new H800 latencies) + DepGraph AP expansion.

Direct grid (NOT additive — additive failed). Excludes failed (-1) tuned rows.
Real-AP table = seed anchors (gap1 stage_a) + expansion (mix_b/mix_d + padded
partners s1_64/s2_128 by weight-identity).  Pairs:
  pair1 trap25/pad64  @AP0.5905   (gap1)
  pair2 mix_b/s1_64   @AP0.6362   (expansion)
  pair3 mix_d/s2_128  @AP0.6369   (s2_128 latency FAILED -> incomplete)
"""
import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results"

# label -> num_filters (the grid2 widths)
LABEL_W = {
    "s0_16": [16,128,256], "s0_32": [32,128,256], "s1_32": [64,32,256],
    "s1_64": [64,64,256],  "s2_64": [64,128,64],  "s2_128": [64,128,128],
    "mix_a": [32,96,192],  "mix_b": [48,64,256],  "mix_c": [16,128,128],
    "mix_d": [48,128,128], "mix_e": [64,96,128],  "mix_f": [32,64,64],
}

# ---- 1. direct-grid latency LUT ----
gap1 = json.loads((R / "gap1_grid_corrected.json").read_text())
widths = {}  # num_filters tuple-str -> row
for g in gap1["grid"]:
    w = tuple(int(x) for x in g["num_filters"])
    widths[w] = {"num_filters": list(w), "label": g["label"],
                 "default_us": float(g["default_us"]), "tuned_us": float(g["tuned_us"])}

# grid2 csv (take LAST valid row per label; skip tuned<=0)
for row in csv.DictReader((R / "lut_results_grid.csv").read_text().splitlines()):
    lab = row["label"]; tun = float(row["tuned_us"])
    if lab not in LABEL_W or tun <= 0:
        continue
    w = tuple(LABEL_W[lab])
    widths[w] = {"num_filters": list(w), "label": lab,
                 "default_us": float(row["default_us"]), "tuned_us": tun}

lut = {"_format": "direct grid (per-width real H800 tuned/default us)",
       "_source": "gap1_grid_corrected + lut_results_grid (relaunch); -1 rows excluded",
       "widths": sorted(widths.values(), key=lambda r: r["num_filters"])}
(R / "latency_lut_pyramid.json").write_text(json.dumps(lut, indent=2))
print(f"[LUT] {len(widths)} priceable widths -> latency_lut_pyramid.json")
for r in lut["widths"]:
    print(f"   {r['num_filters']} {r['label']:8s} def={r['default_us']:.0f} tun={r['tuned_us']:.0f} ratio={r['default_us']/r['tuned_us']:.2f}")

# ---- 2. consolidated real-AP table ----
exp = json.loads((R / "ap70_depgraph_expansion.json").read_text())
def find_ap(tag):
    for fp in exp.get("results", []):
        if fp.get("tag") == tag: return round(float(fp["ap70"]), 4)
    return None
ap_mixb = find_ap("mix_b"); ap_mixd = find_ap("mix_d")

table = [
    {"num_filters":[64,128,256],"ap70":0.6309,"src":"stage_a"},
    {"num_filters":[32,64,128], "ap70":0.5641,"src":"stage_a"},
    {"num_filters":[16,32,64],  "ap70":0.5300,"src":"stage_a"},
    {"num_filters":[48,96,192], "ap70":0.5905,"src":"stage_a trap25 W_g(pair1)"},
    {"num_filters":[64,96,192], "ap70":0.5905,"src":"pad64 P_g(pair1) weight-identity"},
    {"num_filters":[48,64,256], "ap70":ap_mixb,"src":"expansion mix_b W_g(pair2)"},
    {"num_filters":[64,64,256], "ap70":ap_mixb,"src":"s1_64 P_g(pair2) weight-identity"},
    {"num_filters":[48,128,128],"ap70":ap_mixd,"src":"expansion mix_d W_g(pair3)"},
    {"num_filters":[64,128,128],"ap70":ap_mixd,"src":"s2_128 P_g(pair3) weight-identity"},
]
apm = json.loads((R / "ap70_model_pyramid.json").read_text())
apm["table"] = table  # _load_b2 reads data["table"] as EXACT points
(R / "ap70_model_pyramid.json").write_text(json.dumps(apm, indent=2))
print(f"\n[AP] consolidated table = {len(table)} real-AP widths (mix_b={ap_mixb}, mix_d={ap_mixd})")
print("[AP] note: s2_128 has AP but NO latency (crashed) -> candidate_widths will drop it")
