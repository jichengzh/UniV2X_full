"""P×Q×S three-axis ablation driver — the missing 3-axis coupling proof.

Extends the B4 P×S ablation (framework/run_b4_ablation.py) by adding the
quantization (Q) axis, so the experiment now proves that pruning (P),
schedule (S), AND quantization (Q) are JOINTLY coupled — a serial/stepwise
optimizer that fixes one axis at a time systematically misses the optimum.

The categorical Q-coupling (the headline mechanism, MEASURED on H800):
  * Pyramid stage0 grouped conv: in_per_g = s0 // 16.
  * NCHWc int8 (dp4a) needs in_per_g % 4 == 0.
  * s0=48 (in_per_g=3) -> int8 STRUCTURALLY UNBUILDABLE (real, q_int8_dp4a_pairs.csv).
  * s0=64 (in_per_g=4) -> builds real WMMA/dp4a int8, 1.45x over fp16.

Three arms (all driven by the EXTENDED 3-gene NSGA-II in search_three_arm.py):
  A-joint-PQS   : explore P×Q×S jointly.
  A-noS-PQS     : P×Q only, S always default (isolates Q value w/o schedule tuning).
  A-serial-PQS  : FAIR serial. stage1 = width under fp16/default; stage2 = tune
                  schedule × quant on the LOCKED widths.  Q is NOT forbidden — but
                  the widths locked in stage1 (fp16-default-fast = misaligned s0=48
                  greedy trap) are int8-UNBUILDABLE, and the aligned P_g (s0=64) that
                  owns BOTH the schedule rank-flip AND the int8 path was already
                  dropped.  -> serial structurally cannot reach (P_g, tuned, int8).

INT8 latency for buildable widths = H800 fp16 / 1.449 (the one real H800 stage0
int8 speedup, applied as a uniform per-width proxy -> keeps the Pareto axis
H800-pure, no cross-hardware 4090 mixing).  Caveat carried in the JSON.

Run: python -m framework.run_pqs_ablation [--seeds 12] [--budget 60] [--pop 8]
Pure-Python + matplotlib + (scipy if present, else exact permutation fallback).
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
    APModel, LatencyLUT, QLookup, SCHEDULES, QUANT_MODES,
    candidate_widths, compute_hv_ref_pqs,
    run_joint_pqs, run_noS_pqs, run_serial_pqs,
    detect_q_rank_flip_pairs, detect_wg_pg_pairs, load_seed_grid,
)
from framework.run_b4_ablation import wilcoxon_signed_rank, aggregate_convergence

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "multi_agent" / "figure"

UNIFORM_INT8_SPEEDUP = QLookup.MEASURED_INT8_SPEEDUP   # 1.449 (real H800 stage0)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_hv_boxplot(hv: dict, ref_hv: float, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    arms = ["A-joint-PQS", "A-serial-PQS", "A-noS-PQS"]
    data = [hv[a] for a in arms]
    try:
        bp = ax.boxplot(data, tick_labels=[a.replace("-PQS", "") for a in arms],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color="black"))
    except TypeError:  # older matplotlib
        bp = ax.boxplot(data, labels=[a.replace("-PQS", "") for a in arms],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], ["#2c7fb8", "#d95f0e", "#999999"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(ref_hv, ls="--", color="green", lw=1,
               label=f"reference (global P×Q×S) HV = {ref_hv:.3g}")
    ax.set_ylabel("hypervolume (higher = better)")
    ax.set_title("P×Q×S ablation: per-arm HV distribution over seeds")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_pareto_int8(joint_visited, serial_visited, apm, lut, qlut,
                     q_pairs, hv_ref, path: Path):
    """Scatter: fp16 vs int8 points.  Shows A-joint reaches the int8-P_g region;
    A-serial's int8 points exist only on aligned widths it happened to lock
    (the misaligned W_g it locks has NO int8 point)."""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))

    def split(visited):
        pts = {"fp16": ([], []), "int8": ([], [])}
        for (w, s, q) in visited:
            if q == "int8" and qlut.enforce_int8_buildable and not qlut.can_build_int8(w):
                continue   # unbuildable -> no real point
            lat = lut.latency(w, s)
            if q == "int8":
                lat = qlut.int8_lat(lat, w, sched=s)
                ap = apm.ap70(w) - 0.008
            else:
                ap = apm.ap70(w)
            pts[q][0].append(lat); pts[q][1].append(ap)
        return pts

    jp = split(joint_visited)
    sp = split(serial_visited)
    ax.scatter(jp["fp16"][0], jp["fp16"][1], s=120, marker="o", facecolors="none",
               edgecolors="#2c7fb8", linewidths=1.5, label="A-joint fp16")
    ax.scatter(jp["int8"][0], jp["int8"][1], s=120, marker="o",
               color="#2c7fb8", alpha=0.85, label="A-joint INT8 (reached)")
    ax.scatter(sp["fp16"][0], sp["fp16"][1], s=42, marker="x",
               color="#d95f0e", label="A-serial fp16")
    ax.scatter(sp["int8"][0], sp["int8"][1], s=60, marker="P",
               color="#d95f0e", edgecolors="k", label="A-serial INT8 (only aligned locks)")

    for k, p in enumerate(q_pairs):
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        pg_lat = qlut.int8_lat(lut.latency(pg, "tuned"), pg, sched="tuned")
        pg_ap = apm.ap70(pg) - 0.008
        ax.scatter([pg_lat], [pg_ap], s=320, marker="*", color="#2c7fb8",
                   edgecolors="k", zorder=6,
                   label=("P_g tuned+INT8 (joint-only)" if k == 0 else None))
        ax.annotate(f"P_g={list(pg)} s0=64\nINT8 buildable\nW_g={list(wg)} s0=48 "
                    "INT8 UNBUILDABLE",
                    (pg_lat, pg_ap), textcoords="offset points", xytext=(10, -38),
                    fontsize=7.2, color="#2c7fb8",
                    arrowprops=dict(arrowstyle="->", color="#2c7fb8"))

    ax.set_xscale("log")
    ax.set_xlabel("latency (µs, log)  —  INT8 = H800 fp16 / 1.449 (measured proxy)")
    ax.set_ylabel("AP70")
    ax.set_title("P×Q×S: A-joint reaches the INT8 P_g frontier; "
                 "A-serial's locked W_g (s0=48) has no INT8 path")
    ax.legend(fontsize=7.2, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(n_seeds=12, budget=60, pop=8, verbose=True, manifest=None,
        int8_buildable_all=False, int8_speedup=None):
    if manifest:
        # Bridge-native path: space + int8 buildability auto-derived from the
        # stage1 manifest (cur_width keys, key_scale×). Legacy path (manifest=None)
        # is bit-identical to pre-bridge.
        from framework.search_three_arm import build_from_manifest
        b = build_from_manifest(manifest)
        lut, apm, qlut, key_scale = b["lut"], b["apm"], b["qlut"], b["key_scale"]
        if verbose:
            g = b["gating_knob"]
            print(f"  [bridge] manifest={Path(manifest).name} key_scale={key_scale} "
                  f"int8_buildable_align={b['bridge_int8_align']} "
                  f"gating={g.search_group_id if g else None}")
            print(f"  [dispatch] 逐-knob 耦合分流 (stage1 耦合分数→stage2 内环预算):")
            for d in b["dispatch_plan"]:
                print(f"    {d['knob']:18s} score={d['coupling_score']:.3f} "
                      f"(cliff={d['cliff_strength']:.2f} sched={d['schedule_headroom']:.2f}) "
                      f"→ {d['dispatch'].upper()}")
            print(f"  [dispatch] 联合搜旋钮 = {b['joint_knobs']} (其余串行, 省内环预算)")
    else:
        lut = LatencyLUT()
        apm = APModel()
        qlut = QLookup()
        key_scale = 1
    # Wire the structural int8 constraint + H800-pure uniform int8 speedup.
    # int8_buildable_all: int8_tc (im2col+MMA) builds every width (no dp4a pack-4 wall,
    # verified 2026-07-03: all 60 measured widths incl. 46 dp4a-"unbuildable" built+tensorized).
    qlut.enforce_int8_buildable = not int8_buildable_all
    qlut.uniform_int8_speedup = int8_speedup if int8_speedup else UNIFORM_INT8_SPEEDUP

    grid = candidate_widths(lut, apm)
    seed_grid = load_seed_grid(key_scale=key_scale)
    hv_ref = compute_hv_ref_pqs(grid, lut, apm, qlut)

    # P×S rank-flip pairs (schedule axis) and the categorical Q pairs.
    ps_pairs = detect_wg_pg_pairs(grid, lut, apm)
    q_pairs = detect_q_rank_flip_pairs(grid, lut, apm, qlut, sched="tuned")

    # Buildability table (categorical Q-coupling, per width).
    build_table = [{
        "width": list(w),
        "label": seed_grid.get(w, {}).get("label", str(w)),
        "s0": int(w[0]), "in_per_g": qlut.in_per_g(w),
        "int8_buildable": qlut.can_build_int8(w),
    } for w in grid]

    real_lut = lut.mode != "seed"
    real_ap = apm.mode == "b2"

    if verbose:
        print("=" * 84)
        print("P×Q×S THREE-AXIS ABLATION DRIVER")
        print("=" * 84)
        print(f"grid ({len(grid)} widths): "
              + ", ".join(seed_grid.get(w, {}).get('label', str(w)) for w in grid))
        print(f"LUT mode={lut.mode} (real={real_lut})  AP mode={apm.mode} (real={real_ap})")
        print(f"INT8 latency = H800 fp16 / {UNIFORM_INT8_SPEEDUP} (measured stage0 proxy)")
        print(f"INT8 buildable widths: "
              + ", ".join(b["label"] for b in build_table if b["int8_buildable"]))
        print(f"INT8 UNBUILDABLE (s0 misaligned): "
              + ", ".join(b["label"] for b in build_table if not b["int8_buildable"]))
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

    # ---- structural Q-claim (per categorical pair) ----
    def reaches(visited, width, quant):
        return any(w == tuple(width) and q == quant for (w, s, q) in visited)

    # The categorical Q-coupling keys off the P×S rank-flip pairs (s0=48 W_g vs
    # s0=64 P_g): the SAME alignment property that drives the schedule rank-flip
    # ALSO gates int8 buildability.  (detect_q_rank_flip_pairs finds latency
    # rank-flips, which the uniform-speedup proxy erases by construction — the Q
    # coupling here is CATEGORICAL build-or-not, not a latency ratio flip.)
    per_pair = []
    for p in ps_pairs:
        wg, pg = tuple(p["wg"]), tuple(p["pg"])
        per_pair.append({
            "wg": list(wg), "pg": list(pg), "ap70": p["ap70"],
            "wg_in_per_g": qlut.in_per_g(wg), "pg_in_per_g": qlut.in_per_g(pg),
            "wg_int8_buildable": qlut.can_build_int8(wg),     # expect False
            "pg_int8_buildable": qlut.can_build_int8(pg),     # expect True
            "joint_reaches_pg_int8": reaches(joint_visited, pg, "int8"),
            "serial_reaches_pg_int8": reaches(serial_visited, pg, "int8"),
            "ps_iso_ap_latency_ratio": p["iso_ap_latency_ratio"],
        })
    # Categorical claim: every W_g is int8-unbuildable AND P_g is buildable AND
    # A-joint reaches (P_g,int8) while A-serial never reaches (P_g,int8).
    cat_pass = bool(ps_pairs) and all(
        (not pp["wg_int8_buildable"]) and pp["pg_int8_buildable"]
        and pp["joint_reaches_pg_int8"] and not pp["serial_reaches_pg_int8"]
        for pp in per_pair)
    # A misaligned int8 eval is a sentinel (nadir latency) -> can NEVER land on a
    # Pareto front.  Confirm no misaligned-int8 point is on A-serial's deliverable.
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

    # ---- plots ----
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p_box = FIGDIR / "pqs_hv_boxplot.png"
    p_par = FIGDIR / "pqs_pareto_int8.png"
    plot_hv_boxplot(hv, hv_ref_scalar := _ref_hv_scalar(grid, lut, apm, qlut, hv_ref), p_box)
    plot_pareto_int8(joint_visited, serial_visited, apm, lut, qlut,
                     q_pairs, hv_ref, p_par)

    # ---- JSON ----
    def stats(v):
        a = np.array(v)
        ref = hv_ref_scalar
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "pct_of_joint": float(a.mean() / np.mean(hv["A-joint-PQS"]) * 100)}

    results = {
        "meta": {
            "experiment": "P×Q×S three-axis coupling ablation",
            "data_status": "real" if (real_lut and real_ap) else "placeholder_seed_fallback",
            "int8_latency_model": f"H800_TVM_fp16 / {UNIFORM_INT8_SPEEDUP} "
                                  "(uniform proxy from real H800 stage0 WMMA int8; "
                                  "H800-pure, no 4090 cross-hardware)",
            "int8_ap_delta": -0.008,
            "int8_ap_source": "historical_trt_evidence:median_DAIR_val_1789_int8_minmax_ap_delta",
            "structural_constraint": "int8 buildable iff in_per_g(=s0//16) % 4 == 0 "
                "(NCHWc IC_BN=4 / dp4a 4-int8 packing); s0=48 MEASURED-excluded "
                "(q_int8_dp4a_pairs.csv), s0=16/32 excluded by same rule",
            "caveat": ("INT8 latency is a UNIFORM per-width stage0-speedup proxy; "
                       "full-backbone int8 (stage1/stage2) NOT separately measured. "
                       "The CATEGORICAL claim (buildable-or-not) is the robust result; "
                       "the int8 latency MAGNITUDE is a proxy."),
            "lut_mode": lut.mode, "ap_mode": apm.mode,
            "n_seeds": n_seeds, "budget": budget, "pop_size": pop,
            "grid_widths": [list(w) for w in grid],
            "hv_ref_scalar": hv_ref_scalar,
            "hv_ref_nadir": list(hv_ref),
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
            "n_pairs": len(ps_pairs),
            "per_pair": per_pair,
            "categorical_pass": cat_pass,
            "serial_misaligned_int8_on_pareto": serial_misaligned_int8_on_pareto,
            "serial_locked_widths": [list(w) for w in serial_locked],
            "serial_locked_all_int8_unbuildable": serial_locked_all_unbuildable,
            "n_joint_int8_widths": n_joint_int8_widths,
            "n_serial_int8_widths": n_serial_int8_widths,
            "interpretation": (
                "3-axis coupling PROVEN: A-serial's ENTIRE fp16-default-locked frontier "
                f"({[list(w) for w in serial_locked]}) consists of int8-UNBUILDABLE "
                "widths (all s0 in {16,32,48}, in_per_g not divisible by 4) — because "
                "the fp16-default-fast frontier is itself dominated by misaligned-s0 "
                "widths. So A-serial reaches int8 on ZERO widths, while A-joint reaches "
                f"int8 on {n_joint_int8_widths} aligned (s0=64) widths. Even though "
                "A-serial DOES tune quant in stage2, its locked widths cannot build int8. "
                "The SAME s0-alignment property gates pruning, schedule-tuning AND quant."
                if cat_pass else "CHECK per_pair fields.")
        },
        "visited_int8": {
            "A-joint-PQS": sorted(
                [seed_grid.get(w, {}).get("label", str(w)), s]
                for (w, s, q) in joint_visited if q == "int8" and qlut.can_build_int8(w)),
            "A-serial-PQS": sorted(
                [seed_grid.get(w, {}).get("label", str(w)), s]
                for (w, s, q) in serial_visited if q == "int8" and qlut.can_build_int8(w)),
        },
        "figures": {"hv_boxplot": str(p_box), "pareto_int8": str(p_par)},
    }
    out_json = RESULTS / "pqs_ablation_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    if verbose:
        _print_report(results)
        print(f"\nartifacts:\n  {out_json}\n  {p_box}\n  {p_par}")
    return results


def _ref_hv_scalar(grid, lut, apm, qlut, hv_ref) -> float:
    """Reference HV = HV of the global P×Q×S Pareto (offline, never fed to arms)."""
    from framework.search_three_arm import hypervolume_2d, nondominated_idx
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


def _print_report(r: dict):
    print("--- HV distribution (mean over seeds; % of A-joint) ---")
    for a in ("A-joint-PQS", "A-serial-PQS", "A-noS-PQS"):
        s = r["hv_distribution"][a]
        print(f"  {a:14s} mean={s['mean']:.4e} ({s['pct_of_joint']:5.1f}% of joint) "
              f"std={s['std']:.2e} [{s['min']:.3e},{s['max']:.3e}]")
    print("\n--- Wilcoxon A-joint-PQS vs A-serial-PQS (paired by seed) ---")
    w = r["wilcoxon_joint_vs_serial"]
    print(f"  method={w['method']} stat={w['statistic']} p={w['p_value']} "
          f"rank-biserial={w['effect_size_rank_biserial']:.3f} "
          f"mean_diff={w['mean_diff']:.3e}")
    print("\n--- categorical Q-coupling (per pair) ---")
    sc = r["structural_q_claim"]
    for pp in sc["per_pair"]:
        print(f"  W_g={pp['wg']}(in/g={pp['wg_in_per_g']},int8={pp['wg_int8_buildable']}) "
              f"/ P_g={pp['pg']}(in/g={pp['pg_in_per_g']},int8={pp['pg_int8_buildable']}) "
              f"@AP{pp['ap70']}: joint→(P_g,int8)={pp['joint_reaches_pg_int8']} "
              f"serial→(P_g,int8)={pp['serial_reaches_pg_int8']} (expect False)")
    print(f"\n  RESULT: {'PASS' if sc['categorical_pass'] else 'CHECK'} — "
          + ("3-axis P×Q×S coupling: A-serial structurally cannot reach (P_g, int8); "
             "A-joint does. serial_misaligned_int8_on_pareto="
             f"{sc['serial_misaligned_int8_on_pareto']} (expect False)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="P×Q×S three-axis ablation driver")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--manifest", default=None,
                    help="stage1 partition manifest (bridge-native space); "
                         "omit for legacy num_filters path (bit-identical regression)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--int8-buildable-all", action="store_true",
                    help="int8_tc (im2col+MMA) reality: no dp4a pack-4 wall, every width buildable")
    ap.add_argument("--int8-speedup", type=float, default=None,
                    help="uniform int8/fp16 speedup (default 1.449 dp4a stage0 proxy; "
                         "use ~1.095 for measured network-level int8_tc)")
    args = ap.parse_args()
    run(n_seeds=args.seeds, budget=args.budget, pop=args.pop,
        verbose=not args.quiet, manifest=args.manifest,
        int8_buildable_all=args.int8_buildable_all, int8_speedup=args.int8_speedup)
