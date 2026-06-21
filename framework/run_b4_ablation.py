"""B4 — three-arm ablation evidence driver.

Consumes the B3 kernel (framework/search_three_arm.py) and produces the FOUR
evidence layers of doc4 §2.3–2.5 / §6:

  1. Multi-seed × multi-start × greedy-order matrix
       - A-joint / A-noS / A-serial × N>=10 seeds.
       - A-serial multi-start: >=3 different initial greedy width picks per seed.
       - Greedy order: P->S is the only non-degenerate axis in our P×S space
         (S is a per-width binary {default,tuned}; "S->P" would mean picking a
         global schedule before a width, which is ill-defined when each width has
         its own tuned engine -> documented as degenerate, not faked).
  2. Per-arm hypervolume distribution -> boxplot.  Reference Pareto + HV nadir
     computed OFFLINE for scoring only; never fed to any arm (discipline).
  3. Wilcoxon signed-rank A-joint vs A-serial paired by seed + effect size + p.
  4. Three PNGs -> multi_agent/figure/:
       (a) HV boxplot per arm
       (b) convergence curves (eval-count vs running-best HV, mean±band)
       (c) evaluated point-cloud (AP70, latency): A-joint vs A-serial; shows
           A-serial's cloud lacks the P_g (pad64-tuned) region.
  + structured results -> results/b4_ablation_results.json

Grid-agnostic: widths come from kernel.candidate_widths(lut, apm), which auto
picks up the real B1 LUT + B2 AP model when present, else the seed fallback.

Honest judging (doc4 §5): A-joint≈A-serial is a VALID result (separable
regime).  This driver reports the gap as-is; it does not manufacture one.

Run:  python -m framework.run_b4_ablation [--seeds 12] [--budget 60] [--pop 8]
Pure-Python + matplotlib (+ scipy if present; clean permutation fallback else).
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

from framework.search_three_arm import (
    APModel, LatencyLUT, candidate_widths, detect_wg_pg_pairs, compute_hv_ref,
    reference_pareto, run_joint, run_noS, run_serial, load_seed_grid, SCHEDULES,
)
from scripts.phase2.closedloop_objective_query import (
    backbone_to_e2e_latency, driving_score,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "multi_agent" / "figure"


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank (scipy if available; exact permutation fallback)
# ---------------------------------------------------------------------------
def wilcoxon_signed_rank(a: list[float], b: list[float]) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    nz = [d for d in diffs if d != 0]
    out = {"n": len(a), "n_nonzero": len(nz),
           "mean_diff": float(np.mean(diffs)) if diffs else 0.0}
    if not nz:
        out.update({"method": "none", "statistic": None, "p_value": None,
                    "effect_size_rank_biserial": 0.0,
                    "note": "all paired differences are zero (separable regime)"})
        return out
    # rank-biserial effect size (works even when all diffs share a sign)
    ranks = _rankdata([abs(d) for d in nz])
    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    w_neg = sum(r for r, d in zip(ranks, nz) if d < 0)
    total = w_pos + w_neg
    rb = (w_pos - w_neg) / total if total else 0.0
    try:
        from scipy.stats import wilcoxon
        # zsplit handles ties/constant-sign cleanly
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided",
                       mode="auto")
        out.update({"method": "scipy.wilcoxon", "statistic": float(res.statistic),
                    "p_value": float(res.pvalue)})
    except Exception as e:  # permutation fallback (sign-flip exact for small n)
        stat, p = _perm_signed_rank(nz, ranks)
        out.update({"method": "permutation_fallback", "statistic": float(stat),
                    "p_value": float(p), "scipy_error": str(e)})
    out["effect_size_rank_biserial"] = float(rb)
    return out


def _rankdata(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1   # average rank (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _perm_signed_rank(nz: list[float], ranks: list[float]) -> tuple[float, float]:
    """Exact sign-flip permutation on signed ranks (n<=20), else normal approx."""
    n = len(nz)
    w_pos = sum(r for r, d in zip(ranks, nz) if d > 0)
    if n <= 20:
        observed = w_pos
        count = 0
        total = 0
        for mask in range(1 << n):
            s = sum(ranks[i] for i in range(n) if mask & (1 << i))
            total += 1
            # two-sided: |s - mean| >= |observed - mean|
            mean = sum(ranks) / 2
            if abs(s - mean) >= abs(observed - mean) - 1e-9:
                count += 1
        return observed, count / total
    # normal approximation
    rsum = sum(ranks)
    mean = rsum / 2
    var = sum(r * r for r in ranks) / 4
    z = (w_pos - mean) / math.sqrt(var) if var else 0.0
    from math import erfc
    p = erfc(abs(z) / math.sqrt(2))
    return w_pos, p


# ---------------------------------------------------------------------------
# iso-AP70 latency ratio  (A-joint vs A-serial, matched AP70)
# ---------------------------------------------------------------------------
def iso_ap70_ratio(joint_visited: set, serial_visited: set,
                   apm: APModel, lut: LatencyLUT, grid: list) -> dict:
    levels = sorted({round(apm.ap70(w), 6) for w in grid})

    def best_at(visited, v, threshold):
        lats = []
        for (w, s) in visited:
            ap = round(apm.ap70(w), 6)
            if (ap >= v) if threshold else (ap == v):
                lats.append(lut.latency(w, s))
        return min(lats) if lats else None

    per_level = []
    for v in levels:
        je = best_at(joint_visited, v, False)
        se = best_at(serial_visited, v, False)
        jt = best_at(joint_visited, v, True)
        st = best_at(serial_visited, v, True)
        per_level.append({
            "ap70": v,
            "joint_lat_exact": je, "serial_lat_exact": se,
            "ratio_exact": (se / je) if (je and se) else None,
            "joint_lat_threshold": jt, "serial_lat_threshold": st,
            "ratio_threshold": (st / jt) if (jt and st) else None,
        })
    exact = [r for r in per_level if r["ratio_exact"]]
    thr = [r for r in per_level if r["ratio_threshold"]]
    head_e = max(exact, key=lambda r: r["ratio_exact"]) if exact else None
    head_t = max(thr, key=lambda r: r["ratio_threshold"]) if thr else None
    return {
        "per_level": per_level,
        "headline_exact": head_e,        # isolates the structurally-excluded branch
        "headline_threshold": head_t,    # AP70>=v: higher-AP substitution allowed
    }


# ---------------------------------------------------------------------------
# Convergence curve aggregation (step-interpolate onto common eval grid)
# ---------------------------------------------------------------------------
def aggregate_convergence(conv_logs: list[list[dict]]) -> dict:
    max_n = max((c[-1]["n"] for c in conv_logs if c), default=0)
    if max_n == 0:
        return {"n": [], "mean": [], "lo": [], "hi": []}
    grid_n = list(range(1, max_n + 1))
    curves = []
    for c in conv_logs:
        pts = {d["n"]: d["hv"] for d in c}
        cur, last = [], 0.0
        for n in grid_n:
            if n in pts:
                last = pts[n]
            cur.append(last)
        curves.append(cur)
    arr = np.array(curves)
    return {"n": grid_n, "mean": arr.mean(0).tolist(),
            "lo": arr.min(0).tolist(), "hi": arr.max(0).tolist()}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_hv_boxplot(hv: dict, ref_hv: float, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    arms = ["A-joint", "A-serial", "A-noS"]
    data = [hv[a] for a in arms]
    bp = ax.boxplot(data, labels=arms, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], ["#2c7fb8", "#d95f0e", "#999999"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(ref_hv, ls="--", color="green", lw=1,
               label=f"reference (global P×S) HV = {ref_hv:.3g}")
    ax.set_ylabel("hypervolume (higher = better)")
    ax.set_title("B4 layer-2: per-arm HV distribution over seeds")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_convergence(conv: dict, ref_hv: float, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"A-joint": "#2c7fb8", "A-serial": "#d95f0e", "A-noS": "#999999"}
    for arm, agg in conv.items():
        if not agg["n"]:
            continue
        n = agg["n"]
        ax.plot(n, agg["mean"], color=colors[arm], lw=1.8, label=arm)
        ax.fill_between(n, agg["lo"], agg["hi"], color=colors[arm], alpha=0.18)
    ax.axhline(ref_hv, ls="--", color="green", lw=1, label="reference HV")
    ax.set_xlabel("unique cost-model evaluations")
    ax.set_ylabel("running-best hypervolume")
    ax.set_title("B4 layer-2/4: convergence (mean±[min,max] band over seeds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_point_cloud(joint_visited: set, serial_visited: set, ref_front: list,
                     apm: APModel, lut: LatencyLUT, seed_grid: dict, pairs: list,
                     path: Path):
    fig, ax = plt.subplots(figsize=(7.6, 5.0))

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

    # highlight every detected W_g/P_g rank-flip pair
    for k, p in enumerate(pairs):
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        pg_lat = lut.latency(pg, "tuned"); pg_ap = apm.ap70(pg)
        wg_lat = lut.latency(wg, "tuned"); wg_ap = apm.ap70(wg)
        lbl_p = "P_g (tuned) joint-only" if k == 0 else None
        lbl_w = "W_g (tuned) serial-locked, dominated" if k == 0 else None
        ax.scatter([pg_lat], [pg_ap], s=320, marker="*", color="#2c7fb8",
                   edgecolors="k", zorder=5, label=lbl_p)
        ax.scatter([wg_lat], [wg_ap], s=120, marker="P", color="#d95f0e",
                   edgecolors="k", zorder=5, label=lbl_w)
        ax.annotate(f"P_g={list(pg)}\n{p['iso_ap_latency_ratio']}× faster @AP{p['ap70']}\n"
                    "A-serial NEVER visits",
                    (pg_lat, pg_ap), textcoords="offset points", xytext=(12, -40),
                    fontsize=7.5, color="#2c7fb8",
                    arrowprops=dict(arrowstyle="->", color="#2c7fb8"))
        ax.annotate(f"W_g={list(wg)}\nserial-locked",
                    (wg_lat, wg_ap), textcoords="offset points", xytext=(8, 12),
                    fontsize=7.5, color="#d95f0e",
                    arrowprops=dict(arrowstyle="->", color="#d95f0e"))
    rx = [r["lat_us"] for r in ref_front]; ry = [r["ap70"] for r in ref_front]
    order = np.argsort(rx)
    ax.plot(np.array(rx)[order], np.array(ry)[order], ls=":", color="green",
            lw=1, label="reference Pareto (hidden from arms)")

    ax.set_xscale("log")
    ax.set_xlabel("latency (µs, log)")
    ax.set_ylabel("AP70")
    ax.set_title("B4 layer-3 (headline): evaluated point-cloud — "
                 "A-serial's cloud lacks the P_g region")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Closed-loop driving-score axis (model-estimated; AP axis is NOT replaced)
# ---------------------------------------------------------------------------
def _closedloop_report(pairs: list, rep: dict, apm: APModel, lut: LatencyLUT) -> dict:
    """Per-pair W_g-vs-P_g DS contrast (equal AP) + per-arm shipped-Pareto DS.

    DS is MODEL-ESTIMATED (CoDriving τ-curve + H800→Orin linear scaling, β=0).
    It is an additional descriptive axis, NOT folded into hypervolume/search;
    the AP70 objective axis is retained.  See closedloop_b4_plugin_v1.md.
    """
    def ds_of(width, sched):
        ap = apm.ap70(width)
        e2e = backbone_to_e2e_latency(lut.latency(width, sched), "fp32")
        return round(driving_score(ap, e2e, beta=0.0), 2), round(e2e, 1)

    contrast = []
    for p in pairs:
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        wg_ds, wg_e2e = ds_of(wg, "tuned")
        pg_ds, pg_e2e = ds_of(pg, "tuned")
        contrast.append({
            "wg": list(wg), "pg": list(pg), "ap70": p["ap70"],
            "wg_tuned_e2e_ms_est": wg_e2e, "pg_tuned_e2e_ms_est": pg_e2e,
            "wg_tuned_ds": wg_ds, "pg_tuned_ds": pg_ds,
            "ds_gain_pg_over_wg": round(pg_ds - wg_ds, 2),
        })

    per_arm = {}
    for arm, res in (rep or {}).items():
        rows = []
        for r in res.pareto:
            w, s = tuple(r["width"]), r["sched"]
            ds = r.get("ds_model"); e2e = r.get("e2e_orin_ms_est")
            if ds is None:                       # fallback if rec lacked CL fields
                ds, e2e = ds_of(w, s)
            rows.append({"width": list(w), "sched": s, "ap70": round(r["ap70"], 4),
                         "lat_us": round(r["lat_us"], 1),
                         "e2e_orin_ms_est": e2e, "ds_model": ds})
        per_arm[arm] = rows

    return {
        "method": "model-estimated (NOT real closed-loop sim)",
        "caveat": ("DS from CoDriving τ_perc curve + H800→Orin linear scaling "
                   "(±30% non-base); β=0 (no real AP→DS data). Pyramid closed-loop "
                   "needs CARLA (not installed). AP70 axis retained as primary."),
        "scale_tier": "fp32", "beta": 0.0,
        "per_pair_ds_contrast": contrast,
        "per_arm_shipped_pareto_ds_seed0": per_arm,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(n_seeds=12, budget=60, pop=8, n_starts=3, verbose=True):
    lut = LatencyLUT()
    apm = APModel()
    grid = candidate_widths(lut, apm)
    pairs = detect_wg_pg_pairs(grid, lut, apm)     # auto W_g/P_g rank-flip pairs
    seed_grid = load_seed_grid()
    hv_ref = compute_hv_ref(grid, lut, apm)
    ref_front, ref_hv = reference_pareto(grid, lut, apm, hv_ref)

    real_lut = lut.mode != "seed"
    real_ap = apm.mode == "b2"
    placeholder = not (real_lut and real_ap)

    if verbose:
        print("=" * 80)
        print("B4 THREE-ARM ABLATION DRIVER")
        print("=" * 80)
        print(f"grid ({len(grid)} widths): "
              + ", ".join(seed_grid.get(w, {}).get('label', str(w)) for w in grid))
        print(f"LUT mode={lut.mode} (real={real_lut})  "
              f"AP mode={apm.mode} (real={real_ap})")
        print(f"{'REAL DATA' if not placeholder else 'PLACEHOLDER (seed fallback)'}"
              f"  ref_hv={ref_hv:.4e}  seeds={n_seeds} budget={budget}")
        print(f"detected {len(pairs)} W_g/P_g rank-flip pair(s): "
              + "; ".join(f"{list(p['wg'])}/{list(p['pg'])}@AP{p['ap70']}"
                          f"({p['iso_ap_latency_ratio']}×)" for p in pairs) + "\n")

    cl_kw = dict(use_closedloop=True, cl_beta=0.0, cl_scale_tier="fp32")
    hv = {a: [] for a in ("A-joint", "A-noS", "A-serial")}
    conv = {a: [] for a in hv}
    joint_visited, serial_visited, nos_visited = set(), set(), set()
    multistart = []   # per (seed,start): lock set
    rep = {}          # seed-0 ArmResults for per-arm Pareto/DS reporting

    # multi-start widths: extremes + EVERY W_g and P_g (so we test that A-serial
    # drops each P_g even when greedy is forced to START from that very P_g).
    start_widths = []
    def _add_start(w):
        w = tuple(int(x) for x in w)
        if w in grid and w not in start_widths:
            start_widths.append(w)
    if grid:
        _add_start(grid[-1]); _add_start(grid[0])     # widest, narrowest
        for p in pairs:
            _add_start(p["wg"]); _add_start(p["pg"])
        mid = grid[len(grid) // 2]                     # pad toward n_starts
        if len(start_widths) < n_starts:
            _add_start(mid)

    for sd in range(n_seeds):
        rj = run_joint(lut, apm, grid, budget, random.Random(sd), hv_ref, pop, **cl_kw)
        rn = run_noS(lut, apm, grid, budget, random.Random(sd), hv_ref, pop, **cl_kw)
        rs = run_serial(lut, apm, grid, budget, random.Random(sd), hv_ref, pop, **cl_kw)
        hv["A-joint"].append(rj.final_hv); conv["A-joint"].append(rj.cost.conv_log)
        hv["A-noS"].append(rn.final_hv); conv["A-noS"].append(rn.cost.conv_log)
        hv["A-serial"].append(rs.final_hv); conv["A-serial"].append(rs.cost.conv_log)
        joint_visited.update(rj.visited)
        nos_visited.update(rn.visited)
        serial_visited.update(rs.visited)
        if sd == 0:
            rep = {"A-joint": rj, "A-noS": rn, "A-serial": rs}
        # multi-start for A-serial
        for sw in start_widths:
            rss = run_serial(lut, apm, grid, budget, random.Random(sd), hv_ref,
                             pop, start_width=sw, **cl_kw)
            serial_visited.update(rss.visited)
            multistart.append({
                "seed": sd, "start_width": list(sw),
                "locked": [list(w) for w in (rss.locked_widths or [])],
            })

    # ---- statistics ----
    wilcox = wilcoxon_signed_rank(hv["A-joint"], hv["A-serial"])
    iso = iso_ap70_ratio(joint_visited, serial_visited, apm, lut, grid)
    conv_agg = {a: aggregate_convergence(conv[a]) for a in conv}

    # ---- structural confirmation: per-pair (generalized, no hardcoded pair) ----
    def _locked_tuples(m):
        return [tuple(w) for w in m["locked"]]
    per_pair = []
    for p in pairs:
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        # Core structural check (robust, grid-agnostic): P_g is discarded by
        # stage-1 default ranking on EVERY start -> its tuned branch is never
        # built.  (Whichever width A-serial actually locks, P_g is excluded.)
        pg_dropped = [pg not in _locked_tuples(m) for m in multistart]
        # Informational nuance: is the misaligned partner W_g the locked
        # representative?  True for pairs where W_g sits on the default-Pareto
        # (pair2); False when a third width dominates W_g under default too
        # (pair1: mix_b dominates trap25) — both still exclude P_g.
        wg_locked = [wg in _locked_tuples(m) for m in multistart]
        per_pair.append({
            "wg": list(wg), "pg": list(pg), "ap70": p["ap70"],
            "iso_ap70_latency_ratio": p["iso_ap_latency_ratio"],
            "wg_tuned_us": p["wg_tuned_us"], "pg_tuned_us": p["pg_tuned_us"],
            "wg_tuned_ratio": p["wg_tuned_ratio"], "pg_tuned_ratio": p["pg_tuned_ratio"],
            "joint_reaches_pg_tuned": (pg, "tuned") in joint_visited,
            "serial_reaches_pg_tuned": (pg, "tuned") in serial_visited,
            "serial_drops_pg_all_starts": (all(pg_dropped) if pg_dropped else None),
            "wg_is_locked_representative": (all(wg_locked) if wg_locked else None),
            "n_starts_drop_pg": sum(pg_dropped), "n_starts": len(pg_dropped),
        })
    all_pass = bool(pairs) and all(
        pp["joint_reaches_pg_tuned"] and not pp["serial_reaches_pg_tuned"]
        and pp["serial_drops_pg_all_starts"] for pp in per_pair)
    structural = {
        "n_pairs": len(pairs),
        "per_pair": per_pair,
        "all_pairs_pass": all_pass,
        "headline_ratios": [pp["iso_ap70_latency_ratio"] for pp in per_pair],
        "n_multistart_runs": len(multistart),
        "n_starts_per_seed": len(start_widths),
    }

    # ---- closed-loop driving-score axis (model-estimated; AP axis retained) ----
    closedloop = _closedloop_report(pairs, rep, apm, lut)

    # ---- plots ----
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p_box = FIGDIR / "b4_hv_boxplot.png"
    p_conv = FIGDIR / "b4_convergence.png"
    p_cloud = FIGDIR / "b4_pointcloud.png"
    plot_hv_boxplot(hv, ref_hv, p_box)
    plot_convergence(conv_agg, ref_hv, p_conv)
    plot_point_cloud(joint_visited, serial_visited, ref_front, apm, lut,
                     seed_grid, pairs, p_cloud)

    # ---- JSON ----
    def stats(v):
        a = np.array(v)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "pct_of_ref": float(a.mean() / ref_hv * 100)}
    results = {
        "meta": {
            "data_status": "real" if not placeholder else "placeholder_seed_fallback",
            "caveat": ("Magnitudes are PLACEHOLDER on seed-fallback data; structural "
                       "conclusions (visited sets / lock invariance) are valid. "
                       "Re-run when B1 LUT + B2 AP model land for headline numbers."
                       if placeholder else "Real B1/B2 data."),
            "lut_mode": lut.mode, "ap_mode": apm.mode,
            "n_seeds": n_seeds, "budget": budget, "pop_size": pop,
            "grid_widths": [list(w) for w in grid],
            "grid_labels": [seed_grid.get(w, {}).get("label", str(w)) for w in grid],
            "hv_ref_nadir": list(hv_ref),
            "reference_hv": ref_hv,
            "reference_pareto": [{"label": seed_grid.get(r["width"], {}).get(
                "label", str(r["width"])), "sched": r["sched"],
                "ap70": r["ap70"], "lat_us": r["lat_us"]} for r in ref_front],
            "greedy_order_note": ("P->S is the only non-degenerate order: S is a "
                "per-width binary {default,tuned}, so a global 'S->P' pre-pick is "
                "ill-defined (each width owns its tuned engine). Reported P->S; "
                "S->P documented as degenerate, not run."),
        },
        "wg_pg_pairs": [{
            "wg": list(p["wg"]), "pg": list(p["pg"]), "ap70": p["ap70"],
            "wg_tuned_us": p["wg_tuned_us"], "pg_tuned_us": p["pg_tuned_us"],
            "wg_tuned_ratio": p["wg_tuned_ratio"], "pg_tuned_ratio": p["pg_tuned_ratio"],
            "iso_ap_latency_ratio": p["iso_ap_latency_ratio"],
        } for p in pairs],
        "hv_distribution": {a: stats(hv[a]) for a in hv},
        "hv_raw": {a: hv[a] for a in hv},
        "wilcoxon_joint_vs_serial": wilcox,
        "iso_ap70_latency_ratio": iso,
        "structural_claim": structural,
        "closedloop_ds_axis": closedloop,
        "multistart": multistart,
        "visited": {
            "A-joint": sorted([seed_grid.get(w, {}).get("label", str(w)), s]
                              for (w, s) in joint_visited),
            "A-noS": sorted([seed_grid.get(w, {}).get("label", str(w)), s]
                            for (w, s) in nos_visited),
            "A-serial": sorted([seed_grid.get(w, {}).get("label", str(w)), s]
                               for (w, s) in serial_visited),
        },
        "figures": {"hv_boxplot": str(p_box), "convergence": str(p_conv),
                    "point_cloud": str(p_cloud)},
    }
    out_json = RESULTS / "b4_ablation_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    if verbose:
        _print_report(results, ref_hv)
        print(f"\nartifacts:\n  {out_json}\n  {p_box}\n  {p_conv}\n  {p_cloud}")
    return results


def _print_report(r: dict, ref_hv: float):
    print("--- layer 2: HV distribution (mean over seeds) ---")
    for a in ("A-joint", "A-serial", "A-noS"):
        s = r["hv_distribution"][a]
        print(f"  {a:9s} mean={s['mean']:.4e} ({s['pct_of_ref']:5.1f}% ref) "
              f"std={s['std']:.2e} [{s['min']:.3e},{s['max']:.3e}]")
    print("\n--- layer 3: Wilcoxon A-joint vs A-serial (paired by seed) ---")
    w = r["wilcoxon_joint_vs_serial"]
    print(f"  method={w['method']} stat={w['statistic']} p={w['p_value']} "
          f"rank-biserial={w['effect_size_rank_biserial']:.3f} "
          f"mean_diff={w['mean_diff']:.3e}")
    if w.get("note"):
        print(f"  note: {w['note']}")
    print("\n--- headline: per-pair iso-AP70 latency ratio (W_g_tuned / P_g_tuned) ---")
    s = r["structural_claim"]
    if not s["per_pair"]:
        print("  (no W_g/P_g rank-flip pair in grid — nothing to headline)")
    for pp in s["per_pair"]:
        print(f"  pair W_g={pp['wg']} / P_g={pp['pg']} @ iso-AP70={pp['ap70']}: "
              f"{pp['iso_ap70_latency_ratio']:.2f}× "
              f"(W_g tuned {pp['wg_tuned_us']:.0f}µs/{pp['wg_tuned_ratio']:.1f}× vs "
              f"P_g tuned {pp['pg_tuned_us']:.0f}µs/{pp['pg_tuned_ratio']:.1f}×)")
    print("\n--- layer 1/4: structural claim (per pair, generalized) ---")
    for pp in s["per_pair"]:
        print(f"  {pp['wg']}→{pp['pg']}: joint reaches P_g/tuned={pp['joint_reaches_pg_tuned']} "
              f"| serial reaches P_g/tuned={pp['serial_reaches_pg_tuned']} (expect False) "
              f"| serial drops P_g {pp['n_starts_drop_pg']}/{pp['n_starts']} starts="
              f"{pp['serial_drops_pg_all_starts']} "
              f"| W_g locked-repr={pp['wg_is_locked_representative']}")
    print(f"  ({s['n_pairs']} pair(s), {s['n_starts_per_seed']} starts/seed, "
          f"{s['n_multistart_runs']} multi-start runs)")
    ok = s["all_pairs_pass"]
    print(f"\n  STRUCTURAL RESULT: {'PASS' if ok else 'CHECK'} — "
          + (f"A-serial structurally excludes every P_g across all seeds/starts; "
             f"A-joint reaches each. headline ratios={s['headline_ratios']}"
             if ok else "see per-pair fields above."))
    cl = r.get("closedloop_ds_axis")
    if cl:
        print("\n--- closed-loop DS axis (model-estimated; AP axis retained) ---")
        for c in cl["per_pair_ds_contrast"]:
            print(f"  {c['wg']}/{c['pg']} @AP{c['ap70']}: "
                  f"W_g DS={c['wg_tuned_ds']} (e2e {c['wg_tuned_e2e_ms_est']}ms) vs "
                  f"P_g DS={c['pg_tuned_ds']} (e2e {c['pg_tuned_e2e_ms_est']}ms) "
                  f"→ +{c['ds_gain_pg_over_wg']} DS for P_g")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="B4 three-arm ablation driver")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run(n_seeds=args.seeds, budget=args.budget, pop=args.pop,
        n_starts=args.starts, verbose=not args.quiet)
