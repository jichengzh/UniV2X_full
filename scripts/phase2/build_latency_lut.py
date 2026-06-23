"""
build_latency_lut.py  —  B1 LUT builder

Reads all per-stage and validation-combo tuning results (from H800 csv +
gap1_grid_corrected.json), fits additive per-stage latency tables, validates
additivity, and writes:
  results/latency_lut_pyramid.json  — per-stage f_k tables + additivity report
  scripts/phase2/latency_lut_query.py — query function

Additive model:
  Lat(s0, s1, s2, sched) ≈ f_0(s0, sched) + f_1(s1, sched) + f_2(s2, sched)

Where base=[64,128,256] is the anchor:
  f_k(base_k, sched) is fixed by the base measurement; other f_k values are
  inferred from iso_s* measurements (one stage varied at a time).

Strictly per-stage f_k is underdetermined (we can only measure Σ f_k).
We set f_k(w, sched) = Lat(base-except-sk=w, sched) - Lat(base, sched) + f_k(base_k, sched)
i.e., Δ_k(w) = Lat(iso_sk, sched) - Lat(base, sched).
Then: Lat_pred(s0,s1,s2) = Lat_base + Δ_0(s0) + Δ_1(s1) + Δ_2(s2)
  where Δ_k(base_k) = 0.

Usage:
  python scripts/phase2/build_latency_lut.py --lut_csv /exdata/jichengzhi/s2_tvm/lut_results.csv \
    [--json_out results/latency_lut_pyramid.json]

Or call build_lut() programmatically with dicts of raw measurements.
"""
from __future__ import annotations
import json, csv, argparse, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Base widths
BASE = [64, 128, 256]  # [s0, s1, s2]

# Mapping: label -> (num_filters, role)
# role: "single_s0" | "single_s1" | "single_s2" | "combo"
LABEL_META = {
    # Existing (from gap1_grid_corrected.json) — anchor + single-stage
    "base":            ([64, 128, 256], "anchor"),
    "base_retest":     ([64, 128, 256], "anchor"),
    "iso_s0":          ([48, 128, 256], "single_s0"),
    "iso_s0_retest":   ([48, 128, 256], "single_s0"),
    "iso_s0_v2":       ([48, 128, 256], "single_s0"),   # fresh verify run
    "iso_s1":          ([64,  96, 256], "single_s1"),
    "iso_s1_retest":   ([64,  96, 256], "single_s1"),
    "iso_s2":          ([64, 128, 192], "single_s2"),
    "iso_s2_retest":   ([64, 128, 192], "single_s2"),
    # New single-stage sweep
    "s0_16":           ([16, 128, 256], "single_s0"),
    "s0_32":           ([32, 128, 256], "single_s0"),
    "s1_32":           ([64,  32, 256], "single_s1"),
    "s1_64":           ([64,  64, 256], "single_s1"),
    "s2_64":           ([64, 128,  64], "single_s2"),
    "s2_128":          ([64, 128, 128], "single_s2"),
    # Multi-stage validation combos
    "p50":             ([32,  64, 128], "combo"),
    "p50_retest":      ([32,  64, 128], "combo"),
    "trap25":          ([48,  96, 192], "combo"),
    "trap25_retest":   ([48,  96, 192], "combo"),
    "pad64":           ([64,  96, 192], "combo"),
    "pad64_retest":    ([64,  96, 192], "combo"),
    "p75":             ([16,  32,  64], "combo"),
    "p75_retest":      ([16,  32,  64], "combo"),
    "mix_a":           ([32,  96, 192], "combo"),
    "mix_b":           ([48,  64, 256], "combo"),
    "mix_c":           ([16, 128, 128], "combo"),
}


def load_gap1_json(path: str | Path) -> dict:
    """Load gap1_grid_corrected.json, return {label: {default_us, tuned_us, num_filters}}."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for entry in data.get("grid", []):
        lbl = entry["label"]
        # Normalise: "iso_s0" in JSON corresponds to iso_s0_retest label in csv
        out[lbl] = {
            "num_filters": entry["num_filters"],
            "default_us": entry["default_us"],
            "tuned_us": entry["tuned_us"],
        }
    return out


def load_lut_csv(path: str | Path) -> dict:
    """Load H800 lut_results.csv, return {label: {default_us, tuned_us}}."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row["label"]
            try:
                out[lbl] = {
                    "default_us": float(row["default_us"]),
                    "tuned_us": float(row["tuned_us"]),
                }
            except (ValueError, KeyError):
                pass
    return out


def resolve_measurements(gap1: dict, lut_csv: dict) -> dict:
    """
    Merge sources, preferring H800 csv (freshest) over gap1_json.
    Returns {label: {num_filters, default_us, tuned_us}}.
    """
    merged = {}

    # Start with gap1 json (verified data)
    for lbl, entry in gap1.items():
        merged[lbl] = {**entry}

    # Overlay with H800 csv (may have retest / new labels)
    for lbl, entry in lut_csv.items():
        # Determine num_filters from LABEL_META, or fall back to previously merged entry
        nf = None
        if lbl in LABEL_META:
            nf = LABEL_META[lbl][0]   # LABEL_META[lbl] = ([num_filters], "role")
        if nf is None and lbl in merged:
            nf = merged[lbl]["num_filters"]
        if nf is not None:
            merged[lbl] = {
                "num_filters": nf,
                "default_us": entry["default_us"],
                "tuned_us": entry["tuned_us"],
            }

    return merged


def pick_best(measurements: dict, preferred_labels: list) -> dict | None:
    """Return first available label from preferred_labels."""
    for lbl in preferred_labels:
        if lbl in measurements and measurements[lbl]["default_us"] > 0 and measurements[lbl]["tuned_us"] > 0:
            return {**measurements[lbl], "_label": lbl}
    return None


def build_per_stage_tables(measurements: dict) -> dict:
    """
    Build per-stage delta tables from single-stage measurements.

    Returns:
      {
        "base_default_us": float,   # Lat(base, default)
        "base_tuned_us": float,     # Lat(base, tuned)
        "s0": {16: {Δdef, Δtun, def_abs, tun_abs}, 32: ..., 48: ..., 64: ...},
        "s1": {...},
        "s2": {...},
      }
    """
    # Anchor: base
    base_entry = pick_best(measurements, ["base_retest", "base"])
    if base_entry is None:
        raise ValueError("No base measurement found")

    base_def = base_entry["default_us"]
    base_tun = base_entry["tuned_us"]
    print(f"  Anchor (base=[64,128,256]): default={base_def:.1f}µs  tuned={base_tun:.1f}µs")

    # Stage-0: vary s0, others=base(128,256)
    # Labels: base(s0=64), iso_s0/iso_s0_retest(s0=48), s0_16, s0_32
    s0_entries = {
        64: pick_best(measurements, ["base_retest", "base"]),
        48: pick_best(measurements, ["iso_s0_retest", "iso_s0_v2", "iso_s0"]),
        32: pick_best(measurements, ["s0_32"]),
        16: pick_best(measurements, ["s0_16"]),
    }

    # Stage-1: vary s1, others=base(64,256)
    s1_entries = {
        128: pick_best(measurements, ["base_retest", "base"]),
        96:  pick_best(measurements, ["iso_s1_retest", "iso_s1"]),
        64:  pick_best(measurements, ["s1_64"]),
        32:  pick_best(measurements, ["s1_32"]),
    }

    # Stage-2: vary s2, others=base(64,128)
    s2_entries = {
        256: pick_best(measurements, ["base_retest", "base"]),
        192: pick_best(measurements, ["iso_s2_retest", "iso_s2"]),
        128: pick_best(measurements, ["s2_128"]),
        64:  pick_best(measurements, ["s2_64"]),
    }

    def build_stage_table(entries: dict) -> dict:
        table = {}
        for width, entry in sorted(entries.items()):
            if entry is None:
                table[width] = None
                continue
            table[width] = {
                "default_us": round(entry["default_us"], 2),
                "tuned_us":   round(entry["tuned_us"], 2),
                "delta_default_us": round(entry["default_us"] - base_def, 2),
                "delta_tuned_us":   round(entry["tuned_us"]   - base_tun, 2),
                "_source_label": entry.get("_label", "?"),
            }
        return table

    s0_table = build_stage_table(s0_entries)
    s1_table = build_stage_table(s1_entries)
    s2_table = build_stage_table(s2_entries)

    print(f"\n  s0 table (others=[128,256]):")
    for w, v in sorted(s0_table.items()):
        status = f"def={v['default_us']:.0f} tun={v['tuned_us']:.0f} Δdef={v['delta_default_us']:+.0f} Δtun={v['delta_tuned_us']:+.0f}" if v else "MISSING"
        print(f"    s0={w}: {status}")
    print(f"\n  s1 table (others=[64,256]):")
    for w, v in sorted(s1_table.items()):
        status = f"def={v['default_us']:.0f} tun={v['tuned_us']:.0f} Δdef={v['delta_default_us']:+.0f} Δtun={v['delta_tuned_us']:+.0f}" if v else "MISSING"
        print(f"    s1={w}: {status}")
    print(f"\n  s2 table (others=[64,128]):")
    for w, v in sorted(s2_table.items()):
        status = f"def={v['default_us']:.0f} tun={v['tuned_us']:.0f} Δdef={v['delta_default_us']:+.0f} Δtun={v['delta_tuned_us']:+.0f}" if v else "MISSING"
        print(f"    s2={w}: {status}")

    return {
        "base_default_us": round(base_def, 2),
        "base_tuned_us": round(base_tun, 2),
        "s0": s0_table,
        "s1": s1_table,
        "s2": s2_table,
    }


def interpolate_or_nearest(table: dict, width: int) -> dict | None:
    """Linear interpolation between nearest known widths for LUT lookup."""
    available = {w: v for w, v in table.items() if v is not None}
    if not available:
        return None
    if width in available:
        return available[width]
    widths = sorted(available.keys())
    # Find nearest neighbours
    lo = max((w for w in widths if w <= width), default=None)
    hi = min((w for w in widths if w >= width), default=None)
    if lo is None:
        return available[hi]
    if hi is None:
        return available[lo]
    # Interpolate
    alpha = (width - lo) / (hi - lo)
    v_lo, v_hi = available[lo], available[hi]
    return {
        "default_us": v_lo["default_us"] + alpha * (v_hi["default_us"] - v_lo["default_us"]),
        "tuned_us":   v_lo["tuned_us"]   + alpha * (v_hi["tuned_us"]   - v_lo["tuned_us"]),
        "delta_default_us": v_lo["delta_default_us"] + alpha * (v_hi["delta_default_us"] - v_lo["delta_default_us"]),
        "delta_tuned_us":   v_lo["delta_tuned_us"]   + alpha * (v_hi["delta_tuned_us"]   - v_lo["delta_tuned_us"]),
        "_interpolated": True,
    }


def predict_latency(lut: dict, s0: int, s1: int, s2: int, sched: str = "tuned") -> float:
    """Additive prediction: Lat_base + Δ_0(s0) + Δ_1(s1) + Δ_2(s2)."""
    base_val = lut["base_tuned_us"] if sched == "tuned" else lut["base_default_us"]
    delta_key = "delta_tuned_us" if sched == "tuned" else "delta_default_us"

    v0 = interpolate_or_nearest(lut["s0"], s0)
    v1 = interpolate_or_nearest(lut["s1"], s1)
    v2 = interpolate_or_nearest(lut["s2"], s2)

    if v0 is None or v1 is None or v2 is None:
        return float("nan")

    return base_val + v0[delta_key] + v1[delta_key] + v2[delta_key]


def validate_additivity(lut: dict, measurements: dict) -> list:
    """
    Compare additive prediction vs real measurement for multi-stage combos.
    Returns list of {label, num_filters, real_default, real_tuned, pred_default, pred_tuned,
                     err_default_us, err_tuned_us, err_default_pct, err_tuned_pct}
    """
    combo_labels_priority = {
        "p50":    (["p50_retest", "p50"],       [32, 64, 128]),
        "trap25": (["trap25_retest", "trap25"],  [48, 96, 192]),
        "pad64":  (["pad64_retest", "pad64"],    [64, 96, 192]),
        "p75":    (["p75_retest", "p75"],        [16, 32, 64]),
        "mix_a":  (["mix_a"],                    [32, 96, 192]),
        "mix_b":  (["mix_b"],                    [48, 64, 256]),
        "mix_c":  (["mix_c"],                    [16, 128, 128]),
    }

    results = []
    for combo_name, (label_list, nf) in combo_labels_priority.items():
        entry = pick_best(measurements, label_list)
        if entry is None:
            results.append({
                "label": combo_name, "num_filters": nf, "status": "MISSING"
            })
            continue

        s0, s1, s2 = nf
        real_def = entry["default_us"]
        real_tun = entry["tuned_us"]
        pred_def = predict_latency(lut, s0, s1, s2, "default")
        pred_tun = predict_latency(lut, s0, s1, s2, "tuned")

        err_def_us  = pred_def - real_def
        err_tun_us  = pred_tun - real_tun
        err_def_pct = 100 * abs(err_def_us) / real_def if real_def > 0 else float("nan")
        err_tun_pct = 100 * abs(err_tun_us) / real_tun if real_tun > 0 else float("nan")

        results.append({
            "label": combo_name,
            "num_filters": nf,
            "status": "OK",
            "real_default_us": round(real_def, 2),
            "real_tuned_us":   round(real_tun, 2),
            "pred_default_us": round(pred_def, 2),
            "pred_tuned_us":   round(pred_tun, 2),
            "err_default_us":  round(err_def_us, 2),
            "err_tuned_us":    round(err_tun_us, 2),
            "err_default_pct": round(err_def_pct, 1),
            "err_tuned_pct":   round(err_tun_pct, 1),
            "_real_source": entry.get("_label", "?"),
        })

    return results


def summarise_additivity(val_results: list) -> dict:
    pcts = [r["err_tuned_pct"] for r in val_results if r.get("status") == "OK" and isinstance(r.get("err_tuned_pct"), float)]
    def_pcts = [r["err_default_pct"] for r in val_results if r.get("status") == "OK" and isinstance(r.get("err_default_pct"), float)]
    n_ok = sum(1 for r in val_results if r.get("status") == "OK")
    n_total = len(val_results)

    verdict = "USABLE"
    if pcts:
        max_err = max(pcts)
        mean_err = sum(pcts) / len(pcts)
        if max_err > 30:
            verdict = "BROKEN — fall back to real-measure-on-visit"
        elif max_err > 15:
            verdict = "MARGINAL — use with caution, prefer real-measure for top-k candidates"
        else:
            verdict = "USABLE — additive LUT within acceptable error"
    else:
        verdict = "INCOMPLETE — not enough validation combos available yet"
        max_err = mean_err = float("nan")

    return {
        "n_validation_combos_ok": n_ok,
        "n_validation_combos_total": n_total,
        "max_tuned_err_pct": round(max_err, 1) if isinstance(max_err, float) else max_err,
        "mean_tuned_err_pct": round(mean_err, 1) if isinstance(mean_err, float) else mean_err,
        "max_default_err_pct": round(max(def_pcts), 1) if def_pcts else float("nan"),
        "mean_default_err_pct": round(sum(def_pcts) / len(def_pcts), 1) if def_pcts else float("nan"),
        "verdict": verdict,
    }


def build_lut(gap1_json_path: str, lut_csv_path: str, out_json: str):
    print(f"\n=== B1 Latency LUT Builder ===")
    print(f"  gap1 json: {gap1_json_path}")
    print(f"  lut csv:   {lut_csv_path}")

    gap1 = load_gap1_json(gap1_json_path)
    lut_csv = load_lut_csv(lut_csv_path)

    print(f"\n  gap1 labels: {sorted(gap1.keys())}")
    print(f"  lut csv labels: {sorted(lut_csv.keys())}")

    # Resolve into unified dict
    measurements = resolve_measurements(gap1, lut_csv)

    # Build per-stage tables
    print("\n--- Per-stage tables ---")
    per_stage = build_per_stage_tables(measurements)

    # Validate additivity
    print("\n--- Additivity validation ---")
    val_results = validate_additivity(per_stage, measurements)
    summary = summarise_additivity(val_results)

    print(f"\n  Validation results:")
    for r in val_results:
        if r.get("status") == "OK":
            print(f"  {r['label']:10s} nf={r['num_filters']}  "
                  f"real_tun={r['real_tuned_us']:.0f}  pred_tun={r['pred_tuned_us']:.0f}  "
                  f"err={r['err_tuned_us']:+.0f}µs ({r['err_tuned_pct']:.1f}%)  "
                  f"[def err={r['err_default_pct']:.1f}%]")
        else:
            print(f"  {r['label']:10s} nf={r['num_filters']}: {r['status']}")

    print(f"\n  SUMMARY: n_ok={summary['n_validation_combos_ok']}/{summary['n_validation_combos_total']} "
          f"max_tuned_err={summary['max_tuned_err_pct']}% mean_tuned_err={summary['mean_tuned_err_pct']}%")
    print(f"  VERDICT: {summary['verdict']}")

    # Also include verification run comparison (if available)
    verify_note = {}
    iso_v2 = measurements.get("iso_s0_v2")
    iso_ref = measurements.get("iso_s0_retest") or measurements.get("iso_s0")
    if iso_v2 and iso_ref:
        diff_def = iso_v2["default_us"] - iso_ref["default_us"]
        diff_tun = iso_v2["tuned_us"]   - iso_ref["tuned_us"]
        pct_def  = 100 * abs(diff_def) / iso_ref["default_us"]
        pct_tun  = 100 * abs(diff_tun) / iso_ref["tuned_us"]
        verify_note = {
            "iso_s0_v2_default_us": round(iso_v2["default_us"], 2),
            "iso_s0_v2_tuned_us":   round(iso_v2["tuned_us"],   2),
            "iso_s0_ref_default_us": round(iso_ref["default_us"], 2),
            "iso_s0_ref_tuned_us":   round(iso_ref["tuned_us"],   2),
            "diff_default_us": round(diff_def, 2),
            "diff_tuned_us":   round(diff_tun, 2),
            "run_to_run_noise_default_pct": round(pct_def, 1),
            "run_to_run_noise_tuned_pct":   round(pct_tun, 1),
        }
        print(f"\n  VERIFICATION iso_s0_v2 vs ref: "
              f"default {iso_v2['default_us']:.0f} vs {iso_ref['default_us']:.0f} ({pct_def:.1f}%), "
              f"tuned   {iso_v2['tuned_us']:.0f} vs {iso_ref['tuned_us']:.0f} ({pct_tun:.1f}%)")
        if pct_tun > 15:
            print(f"  *** WARNING: verification divergence {pct_tun:.1f}% > 15% — STOP and report ***")
        else:
            print(f"  Verification within run-to-run noise: OK")

    # Assemble output JSON
    output = {
        "_description": "B1 additive per-stage latency LUT for Pyramid backbone (H800 TVM MetaSchedule)",
        "_hardware": "H800 Hopper, TVM 0.20.dev1070, backbone-only, idle GPU, min/mean over 500 reps",
        "_model": "Lat(s0,s1,s2,sched) ≈ base_lat(sched) + Δ_0(s0,sched) + Δ_1(s1,sched) + Δ_2(s2,sched)",
        "_anchor": "base=[64,128,256]; all Δ defined relative to base (Δ_k(base_k)=0)",
        "_query": "Use lookup(s0, s1, s2, sched) from scripts/phase2/latency_lut_query.py",
        "per_stage_tables": per_stage,
        "additivity_validation": {
            "summary": summary,
            "per_combo": val_results,
        },
        "verification_run": verify_note,
    }

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Written -> {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap1_json",
                        default=str(REPO_ROOT / "results/gap1_grid_corrected.json"))
    parser.add_argument("--lut_csv",
                        default=str(REPO_ROOT / "results/lut_results_h800.csv"),
                        help="Path to the new H800 lut_results.csv (scp'd locally)")
    parser.add_argument("--retune_csv",
                        default=str(REPO_ROOT / "results/gap1_grid_retune_h800.csv"),
                        help="Path to existing gap1_grid_retune_h800.csv for retest labels")
    parser.add_argument("--json_out",
                        default=str(REPO_ROOT / "results/latency_lut_pyramid.json"))
    args = parser.parse_args()

    # Load both CSVs and merge
    gap1 = load_gap1_json(args.gap1_json)
    lut_csv = load_lut_csv(args.retune_csv)
    lut_new  = load_lut_csv(args.lut_csv)
    lut_csv.update(lut_new)  # new data wins

    # Resolve: re-use shared logic via direct call
    print(f"\n  gap1 json: {args.gap1_json}")
    print(f"  retune csv: {args.retune_csv}")
    print(f"  lut csv: {args.lut_csv}")
    print(f"\n  gap1 labels: {sorted(gap1.keys())}")
    print(f"  lut csv combined labels: {sorted(lut_csv.keys())}")

    measurements = resolve_measurements(gap1, lut_csv)

    print("\n--- Per-stage tables ---")
    per_stage = build_per_stage_tables(measurements)

    print("\n--- Additivity validation ---")
    val_results = validate_additivity(per_stage, measurements)
    summary = summarise_additivity(val_results)

    print(f"\n  Validation results:")
    for r in val_results:
        if r.get("status") == "OK":
            print(f"  {r['label']:10s} nf={r['num_filters']}  "
                  f"real_tun={r['real_tuned_us']:.0f}  pred_tun={r['pred_tuned_us']:.0f}  "
                  f"err={r['err_tuned_us']:+.0f}µs ({r['err_tuned_pct']:.1f}%)  "
                  f"[def err={r['err_default_pct']:.1f}%]")
        else:
            print(f"  {r['label']:10s} nf={r['num_filters']}: {r['status']}")

    print(f"\n  SUMMARY: n_ok={summary['n_validation_combos_ok']}/{summary['n_validation_combos_total']} "
          f"max_tuned_err={summary['max_tuned_err_pct']}% mean_tuned_err={summary['mean_tuned_err_pct']}%")
    print(f"  VERDICT: {summary['verdict']}")

    # Verification check
    verify_note = {}
    iso_v2  = measurements.get("iso_s0_v2")
    iso_ref = measurements.get("iso_s0_retest") or measurements.get("iso_s0")
    if iso_v2 and iso_ref:
        diff_def = iso_v2["default_us"] - iso_ref["default_us"]
        diff_tun = iso_v2["tuned_us"]   - iso_ref["tuned_us"]
        pct_def  = 100 * abs(diff_def) / iso_ref["default_us"]
        pct_tun  = 100 * abs(diff_tun) / iso_ref["tuned_us"]
        verify_note = {
            "iso_s0_v2_default_us":   round(iso_v2["default_us"],   2),
            "iso_s0_v2_tuned_us":     round(iso_v2["tuned_us"],     2),
            "iso_s0_ref_default_us":  round(iso_ref["default_us"],  2),
            "iso_s0_ref_tuned_us":    round(iso_ref["tuned_us"],    2),
            "diff_default_us": round(diff_def, 2),
            "diff_tuned_us":   round(diff_tun, 2),
            "run_to_run_noise_default_pct": round(pct_def, 1),
            "run_to_run_noise_tuned_pct":   round(pct_tun, 1),
        }
        print(f"\n  VERIFICATION iso_s0_v2 vs ref: "
              f"default {iso_v2['default_us']:.0f} vs {iso_ref['default_us']:.0f} ({pct_def:.1f}%), "
              f"tuned {iso_v2['tuned_us']:.0f} vs {iso_ref['tuned_us']:.0f} ({pct_tun:.1f}%)")
        if pct_tun > 15:
            print(f"  *** WARNING: verification divergence {pct_tun:.1f}% > 15% — STOP and report ***")
        else:
            print(f"  Verification within run-to-run noise: OK")

    output = {
        "_description": "B1 additive per-stage latency LUT for Pyramid backbone (H800 TVM MetaSchedule)",
        "_hardware": "H800 Hopper, TVM 0.20.dev1070, backbone-only, idle GPU, min/mean over 500 reps",
        "_model": "Lat(s0,s1,s2,sched) ≈ base_lat(sched) + Δ_0(s0,sched) + Δ_1(s1,sched) + Δ_2(s2,sched)",
        "_anchor": "base=[64,128,256]; all Δ defined relative to base (Δ_k(base_k)=0)",
        "_query": "Use lookup(s0, s1, s2, sched) from scripts/phase2/latency_lut_query.py",
        "per_stage_tables": per_stage,
        "additivity_validation": {
            "summary": summary,
            "per_combo": val_results,
        },
        "verification_run": verify_note,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Written -> {out_path}")


if __name__ == "__main__":
    main()
