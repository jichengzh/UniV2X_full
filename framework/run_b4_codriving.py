"""B4-CoDriving — T3 separable-model contra-arm.

Same three-arm search kernel as run_b4_ablation.py (Pyramid), but fed
CoDriving-specific data:
  * LUT: results/latency_lut_codriving.json
    - base + p50: REAL H800 TVM measurements (cod_e2e.csv, P2)
    - p25 + p75:  ESTIMATED via power-law channel scaling (caveat in JSON)
  * AP model: results/ap70_model_codriving.json
    - iso-budget de-confounded AP70 (codriving_isobudget_verdict.csv)

Physics expectation (P4 HANDOFF_codriving_tvm_migration_v1.md §1.6):
  CoDriving backbone = standard 3x3 ResNet conv (K=9W >> N=W deep reduction).
  TVM tiling argmin (16x16x32) is UNIVERSAL across all prune widths (0.0%
  loss from a fixed tile choice, 6 rotor knobs tested, 3x rep validated).
  => Tuning improvement ratio (~2.1-2.3x) is near-constant across widths.
  => NO latency rank-flip between default and tuned schedule.
  => detect_wg_pg_pairs should return 0 pairs.
  => A-joint ≈ A-serial (separable regime).

CONTRAST with Pyramid (run_b4_ablation.py B4 result):
  Pyramid grouped bottleneck: tuning ratios span 1.96x (trap25, misaligned)
  to 10.3x (p75, well-aligned). Large rank-flip pairs (3.29-3.5x iso-AP
  ratio). A-joint 99.6% vs A-serial 86.0% HV, Wilcoxon p=4.9e-4.

CoDriving EXPECTED result:
  0 rank-flip pairs => A-joint ≈ A-serial ≈ (A-noS excluded since tuning
  still helps uniformly across widths). HV gap A-joint vs A-serial ≈ 0.
  Wilcoxon p >> 0.05 (not significant), confirming separable regime.

Honest judging (HANDOFF_codesign_nextstage_v1.md §4 T3 iron-rule):
  A-joint ≈ A-serial IS THE CORRECT RESULT for CoDriving. Do not force
  a gap. The model-dependent separability is the scientific finding.

Run: python -m framework.run_b4_codriving [--seeds 12] [--budget 60] [--pop 8]
Output: results/b4_codriving_ablation_results.json + 3 figures in figure/
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse kernel + Pyramid driver utilities
from framework.search_three_arm import (
    APModel, LatencyLUT, candidate_widths, detect_wg_pg_pairs, compute_hv_ref,
    reference_pareto, run_joint, run_noS, run_serial, load_seed_grid, SCHEDULES,
)

# Import Wilcoxon + iso-AP utilities from the Pyramid B4 driver
from framework.run_b4_ablation import (
    wilcoxon_signed_rank, iso_ap70_ratio, aggregate_convergence,
    plot_hv_boxplot, plot_convergence,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "multi_agent" / "figure"

# CoDriving-specific LUT + AP model paths
COD_LUT_JSON = RESULTS / "latency_lut_codriving.json"
COD_AP_JSON = RESULTS / "ap70_model_codriving.json"

# Fallback inline data (in case JSON files not found)
_INLINE_LUT_WIDTHS = [
    {"num_filters": [64, 128, 256], "label": "base",
     "default_us": 16680.47, "tuned_us": 8057.79},
    {"num_filters": [48, 96, 192], "label": "p25",
     "default_us": 8874.0, "tuned_us": 4033.0},
    {"num_filters": [32, 64, 128], "label": "p50",
     "default_us": 3643.15, "tuned_us": 1609.10},
    {"num_filters": [16, 32, 64], "label": "p75",
     "default_us": 795.0, "tuned_us": 361.0},
]
_INLINE_AP_TABLE = [
    {"num_filters": [64, 128, 256], "ap70": 0.4063},  # base_isobudget
    {"num_filters": [48, 96, 192], "ap70": 0.3661},   # p25
    {"num_filters": [32, 64, 128], "ap70": 0.3845},   # p50 (seed1)
    {"num_filters": [16, 32, 64],  "ap70": 0.4049},   # p75
]


def _load_codriving_lut() -> LatencyLUT:
    """Load CoDriving LUT. Falls back to inline if file missing."""
    if COD_LUT_JSON.exists():
        lut = LatencyLUT(COD_LUT_JSON, RESULTS / "gap1_grid_corrected.json"
                         if (RESULTS / "gap1_grid_corrected.json").exists()
                         else COD_LUT_JSON)
        # Override: direct widths from COD_LUT_JSON
        data = json.loads(COD_LUT_JSON.read_text())
        rows = data.get("widths", [])
        for r in rows:
            w = tuple(int(x) for x in r["num_filters"])
            lut.direct[w] = {"default_us": float(r["default_us"]),
                              "tuned_us": float(r["tuned_us"])}
        lut.mode = "codriving_real+estimated"
        return lut
    # inline fallback
    lut = LatencyLUT.__new__(LatencyLUT)
    lut.seed = {}
    lut.mode = "codriving_inline_fallback"
    lut.direct = {}
    lut.per_stage = {}
    lut.additive_error = None
    for r in _INLINE_LUT_WIDTHS:
        w = tuple(int(x) for x in r["num_filters"])
        lut.direct[w] = {"default_us": float(r["default_us"]),
                          "tuned_us": float(r["tuned_us"])}
    return lut


def _load_codriving_apm() -> APModel:
    """Load CoDriving AP model. Falls back to inline if file missing."""
    ap_path = COD_AP_JSON if COD_AP_JSON.exists() else RESULTS / "latency_lut_codriving.json"
    apm = APModel.__new__(APModel)
    apm.exact = {}
    apm.mode = "codriving"

    if COD_AP_JSON.exists():
        data = json.loads(COD_AP_JSON.read_text())
        rows = data.get("table", [])
        for r in rows:
            apm.exact[tuple(int(x) for x in r["num_filters"])] = float(r["ap70"])
    else:
        for r in _INLINE_AP_TABLE:
            apm.exact[tuple(int(x) for x in r["num_filters"])] = float(r["ap70"])

    apm._build_interp()
    return apm


def plot_point_cloud_cod(joint_visited, serial_visited, ref_front,
                          apm, lut, grid, pairs, path: Path):
    """Point cloud plot for CoDriving (no W_g/P_g pairs expected)."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    labels = {(64,128,256): "base", (48,96,192): "p25",
              (32,64,128): "p50", (16,32,64): "p75"}

    def pts(visited):
        xs, ys = [], []
        for (w, s) in visited:
            xs.append(lut.latency(w, s)); ys.append(apm.ap70(w))
        return xs, ys

    jx, jy = pts(joint_visited)
    sx, sy = pts(serial_visited)
    ax.scatter(jx, jy, s=130, marker="o", facecolors="none",
               edgecolors="#2c7fb8", linewidths=1.8, label="A-joint visited")
    ax.scatter(sx, sy, s=40, marker="x", color="#d95f0e", label="A-serial visited")

    # annotate each width
    for w in grid:
        lab = labels.get(w, str(w))
        for s in SCHEDULES:
            try:
                x, y = lut.latency(w, s), apm.ap70(w)
                ax.annotate(f"{lab}/{s[0]}", (x, y), fontsize=7, alpha=0.6)
            except Exception:
                pass

    rx = [r["lat_us"] for r in ref_front]; ry = [r["ap70"] for r in ref_front]
    order = np.argsort(rx)
    ax.plot(np.array(rx)[order], np.array(ry)[order], ls=":", color="green",
            lw=1, label="reference Pareto (hidden from arms)")

    title = ("B4-CoDriving T3: point cloud — standard conv separable\n"
             "A-joint and A-serial explore near-identical regions (0 rank-flip pairs)")
    ax.set_xscale("log")
    ax.set_xlabel("latency (µs, log) — TVM H800 backbone")
    ax.set_ylabel("AP70 (DAIR val 1789, iso-budget)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.text(0.98, 0.05,
            "CoDriving: TVM ratio 2.07-2.26× uniform\n"
            "Pyramid: TVM ratio 1.96-10.3× width-dependent\n"
            "=> 0 rank-flip pairs => A-joint ≈ A-serial",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(n_seeds=12, budget=60, pop=8, n_starts=3, verbose=True):
    lut = _load_codriving_lut()
    apm = _load_codriving_apm()

    # Determine grid from the loaded data
    grid_w = sorted(apm.exact.keys())
    # Keep only widths whose latency is resolvable (both default + tuned)
    grid = []
    for w in grid_w:
        try:
            lut.latency(w, "default"); lut.latency(w, "tuned"); grid.append(w)
        except Exception:
            pass

    pairs = detect_wg_pg_pairs(grid, lut, apm)
    hv_ref = compute_hv_ref(grid, lut, apm)
    ref_front, ref_hv = reference_pareto(grid, lut, apm, hv_ref)

    data_status = ("real+estimated" if COD_LUT_JSON.exists() and COD_AP_JSON.exists()
                   else "inline_fallback")
    n_real = sum(1 for w in [(64,128,256),(32,64,128)] if w in [tuple(x) for x in grid])

    if verbose:
        print("=" * 80)
        print("B4-CoDriving: T3 SEPARABLE CONTRA-ARM")
        print("=" * 80)
        print(f"Model: CoDriving (standard 3x3 ResNet backbone)")
        print(f"LUT mode: {lut.mode} (2 real H800 TVM + 2 estimated)")
        print(f"AP mode: {apm.mode}")
        print(f"Grid ({len(grid)} widths): " +
              ", ".join(f"{w}" for w in grid))
        print(f"AP70 per width: " +
              ", ".join(f"{apm.ap70(w):.4f}" for w in grid))
        print(f"Default latency (µs): " +
              ", ".join(f"{lut.latency(w,'default'):.0f}" for w in grid))
        print(f"Tuned latency (µs):   " +
              ", ".join(f"{lut.latency(w,'tuned'):.0f}" for w in grid))
        print(f"Tuning ratios:        " +
              ", ".join(f"{lut.latency(w,'default')/lut.latency(w,'tuned'):.2f}×"
                        for w in grid))
        print(f"\ndetect_wg_pg_pairs → {len(pairs)} pair(s): "
              + ("; ".join(f"{p['wg']}/{p['pg']}@AP{p['ap70']}({p['iso_ap_latency_ratio']}×)"
                          for p in pairs) if pairs else "NONE (separable: no rank-flip)"))
        print(f"ref_hv={ref_hv:.4e}  seeds={n_seeds}  budget={budget}\n")

    hv = {a: [] for a in ("A-joint", "A-noS", "A-serial")}
    conv = {a: [] for a in hv}
    joint_visited, serial_visited, nos_visited = set(), set(), set()
    multistart = []
    rep = {}

    # start widths for multi-start
    start_widths = []
    def _add(w):
        w = tuple(int(x) for x in w)
        if w in grid and w not in start_widths:
            start_widths.append(w)
    if grid:
        _add(grid[-1]); _add(grid[0])
        for p in pairs:
            _add(p["wg"]); _add(p["pg"])
        mid = grid[len(grid) // 2]
        if len(start_widths) < n_starts:
            _add(mid)

    for sd in range(n_seeds):
        rj = run_joint(lut, apm, grid, budget, random.Random(sd), hv_ref, pop)
        rn = run_noS(lut, apm, grid, budget, random.Random(sd), hv_ref, pop)
        rs = run_serial(lut, apm, grid, budget, random.Random(sd), hv_ref, pop)
        hv["A-joint"].append(rj.final_hv)
        hv["A-noS"].append(rn.final_hv)
        hv["A-serial"].append(rs.final_hv)
        conv["A-joint"].append(rj.cost.conv_log)
        conv["A-noS"].append(rn.cost.conv_log)
        conv["A-serial"].append(rs.cost.conv_log)
        joint_visited.update(rj.visited)
        nos_visited.update(rn.visited)
        serial_visited.update(rs.visited)
        if sd == 0:
            rep = {"A-joint": rj, "A-noS": rn, "A-serial": rs}
        for sw in start_widths:
            rss = run_serial(lut, apm, grid, budget, random.Random(sd), hv_ref,
                             pop, start_width=sw)
            serial_visited.update(rss.visited)
            multistart.append({
                "seed": sd, "start_width": list(sw),
                "locked": [list(w) for w in (rss.locked_widths or [])],
            })

    wilcox = wilcoxon_signed_rank(hv["A-joint"], hv["A-serial"])
    iso = iso_ap70_ratio(joint_visited, serial_visited, apm, lut, grid)
    conv_agg = {a: aggregate_convergence(conv[a]) for a in conv}

    # structural claim (generalized, works with 0 pairs)
    def _locked_tuples(m):
        return [tuple(w) for w in m["locked"]]
    per_pair = []
    for p in pairs:
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        pg_dropped = [pg not in _locked_tuples(m) for m in multistart]
        wg_locked = [wg in _locked_tuples(m) for m in multistart]
        per_pair.append({
            "wg": list(wg), "pg": list(pg), "ap70": p["ap70"],
            "iso_ap70_latency_ratio": p["iso_ap_latency_ratio"],
            "joint_reaches_pg_tuned": (pg, "tuned") in joint_visited,
            "serial_reaches_pg_tuned": (pg, "tuned") in serial_visited,
            "serial_drops_pg_all_starts": (all(pg_dropped) if pg_dropped else None),
            "wg_is_locked_representative": (all(wg_locked) if wg_locked else None),
        })
    all_pass = bool(pairs) and all(
        pp["joint_reaches_pg_tuned"] and not pp["serial_reaches_pg_tuned"]
        and pp["serial_drops_pg_all_starts"] for pp in per_pair)
    structural = {
        "n_pairs": len(pairs),
        "per_pair": per_pair,
        "all_pairs_pass": all_pass,
        "headline_ratios": [pp["iso_ap70_latency_ratio"] for pp in per_pair],
        "separability_verdict": (
            "SEPARABLE: 0 rank-flip pairs detected. A-joint ≈ A-serial by construction."
            if len(pairs) == 0 else
            f"{len(pairs)} rank-flip pair(s) found — UNEXPECTED for CoDriving."
        ),
    }

    # plots
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p_box = FIGDIR / "b4_cod_hv_boxplot.png"
    p_conv = FIGDIR / "b4_cod_convergence.png"
    p_cloud = FIGDIR / "b4_cod_pointcloud.png"
    plot_hv_boxplot(hv, ref_hv, p_box)
    plot_convergence(conv_agg, ref_hv, p_conv)
    plot_point_cloud_cod(joint_visited, serial_visited, ref_front,
                          apm, lut, grid, pairs, p_cloud)

    def stats(v):
        a = np.array(v)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "pct_of_ref": float(a.mean() / ref_hv * 100)}

    # visited sets as labels
    lab_map = {(64,128,256):"base",(48,96,192):"p25",(32,64,128):"p50",(16,32,64):"p75"}
    def _vis_labels(vis):
        return sorted([[lab_map.get(w, str(list(w))), s] for (w,s) in vis])

    results = {
        "meta": {
            "model": "CoDriving (standard 3x3 ResNet backbone, DAIR val)",
            "purpose": "T3 separable contra-arm: expected A-joint ≈ A-serial (0 rank-flip pairs)",
            "data_status": data_status,
            "lut_mode": lut.mode,
            "ap_mode": apm.mode,
            "lut_caveat": ("base+p50 REAL H800 TVM (cod_e2e.csv P2); "
                           "p25+p75 ESTIMATED via power-law channel scaling, 2.20x ratio"),
            "n_seeds": n_seeds, "budget": budget, "pop_size": pop,
            "grid_widths": [list(w) for w in grid],
            "ap70_per_width": {str(list(w)): round(apm.ap70(w), 4) for w in grid},
            "default_us_per_width": {str(list(w)): round(lut.latency(w,"default"),1) for w in grid},
            "tuned_us_per_width": {str(list(w)): round(lut.latency(w,"tuned"),1) for w in grid},
            "tuning_ratio_per_width": {
                str(list(w)): round(lut.latency(w,"default")/lut.latency(w,"tuned"),2)
                for w in grid},
            "reference_hv": ref_hv,
            "hv_ref_nadir": list(hv_ref),
            "physics_explanation": (
                "CoDriving 3x3 ResNet: K=9W >> N=W (deep reduction). P4 experiment "
                "showed tile argmin=16x16x32 is optimal for ALL prune widths (0% loss). "
                "Tuning ratio ~2.1-2.3x is near-constant across widths. "
                "=> NO latency rank-flip between default and tuned. "
                "=> A-serial stage-1 default-Pareto contains same widths as A-joint Pareto. "
                "=> A-serial stage-2 (tune locked widths) = A-joint tuned outcome. "
                "=> A-joint ≈ A-serial (separable). "
                "CONTRAST: Pyramid grouped bottleneck has 1.96x-10.3x span of tuning ratios "
                "-> strong rank-flip (3.29x iso-AP ratio) -> A-joint >> A-serial."
            ),
        },
        "wg_pg_pairs": [{
            "wg": list(p["wg"]), "pg": list(p["pg"]), "ap70": p["ap70"],
            "iso_ap_latency_ratio": p["iso_ap_latency_ratio"],
        } for p in pairs],
        "hv_distribution": {a: stats(hv[a]) for a in hv},
        "hv_raw": {a: hv[a] for a in hv},
        "wilcoxon_joint_vs_serial": wilcox,
        "iso_ap70_latency_ratio": iso,
        "structural_claim": structural,
        "visited": {
            "A-joint": _vis_labels(joint_visited),
            "A-noS": _vis_labels(nos_visited),
            "A-serial": _vis_labels(serial_visited),
        },
        "figures": {
            "hv_boxplot": str(p_box),
            "convergence": str(p_conv),
            "point_cloud": str(p_cloud),
        },
    }

    out_json = RESULTS / "b4_codriving_ablation_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    if verbose:
        _print_report(results)
        print(f"\nartifacts:\n  {out_json}\n  {p_box}\n  {p_conv}\n  {p_cloud}")
    return results


def _print_report(r: dict):
    ref_hv = r["meta"]["reference_hv"]
    print("=" * 80)
    print("B4-CoDriving: T3 CONTRA-ARM RESULTS")
    print("=" * 80)
    print(f"Model: {r['meta']['model']}")
    print(f"Pairs detected: {r['structural_claim']['n_pairs']}")
    print(f"Separability verdict: {r['structural_claim']['separability_verdict']}")
    print(f"\nTuning ratios (CoDriving vs Pyramid):")
    for w_str, ratio in r['meta']['tuning_ratio_per_width'].items():
        print(f"  {w_str}: {ratio:.2f}x")
    print("  Pyramid range: 1.96x–10.3x (trap25–pad64) → RANK FLIP")
    print("  CoDriving: {:.2f}x–{:.2f}x → NO RANK FLIP → SEPARABLE".format(
        min(r['meta']['tuning_ratio_per_width'].values()),
        max(r['meta']['tuning_ratio_per_width'].values())))

    print("\n--- Layer 2: HV distribution (mean over seeds) ---")
    for a in ("A-joint", "A-serial", "A-noS"):
        s = r["hv_distribution"][a]
        print(f"  {a:9s} mean={s['mean']:.4e} ({s['pct_of_ref']:5.1f}% ref) "
              f"std={s['std']:.2e} [{s['min']:.3e},{s['max']:.3e}]")

    print("\n--- Layer 3: Wilcoxon A-joint vs A-serial ---")
    w = r["wilcoxon_joint_vs_serial"]
    print(f"  method={w['method']} stat={w['statistic']} p={w['p_value']} "
          f"rank-biserial={w['effect_size_rank_biserial']:.3f} "
          f"mean_diff={w['mean_diff']:.3e}")
    if w.get("note"):
        print(f"  note: {w['note']}")

    hjv = r["hv_distribution"]["A-joint"]["mean"]
    hsv = r["hv_distribution"]["A-serial"]["mean"]
    hnv = r["hv_distribution"]["A-noS"]["mean"]
    print(f"\n  HV ratio A-joint/A-serial = {hjv/hsv:.4f} (expect ~1.00 for separable)")
    print(f"  HV ratio A-joint/A-noS    = {hjv/hnv:.4f} "
          f"(> 1 expected: tuning helps uniformly)")

    print("\n--- CROSS-MODEL COMPARISON ---")
    print("  Model          | A-joint HV%ref | A-serial HV%ref | A-noS HV%ref | "
          "Wilcoxon p | n_pairs | max_ratio")
    jpc = r["hv_distribution"]["A-joint"]["pct_of_ref"]
    spc = r["hv_distribution"]["A-serial"]["pct_of_ref"]
    npc = r["hv_distribution"]["A-noS"]["pct_of_ref"]
    print(f"  CoDriving (T3) | {jpc:14.1f} | {spc:15.1f} | {npc:12.1f} | "
          f"{w['p_value'] if w['p_value'] else 'N/A':>10} | "
          f"{r['structural_claim']['n_pairs']:>7} | "
          f"{max(r['meta']['tuning_ratio_per_width'].values()):.2f}x")
    print("  Pyramid (B4)   |          99.6 |            86.0 |         48.0 | "
          "4.9e-04    |       2 | 10.34x")
    print("  => CoDriving: separable (low TVM ratio, 0 pairs, A-joint~A-serial)")
    print("  => Pyramid: coupled  (high TVM ratio on aligned widths, 2 pairs, "
          "A-joint>>A-serial)")
    print("  => Co-design value = f(coupling strength) = f(conv type) [model-dependent]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="B4-CoDriving: T3 separable contra-arm")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(n_seeds=args.seeds, budget=args.budget, pop=args.pop,
        n_starts=args.starts, verbose=not args.quiet)
