#!/usr/bin/env python3
"""
L1 Integration: merge real H800 TVM int8 latencies into latency_lut_pyramid_q.json,
then re-run P×Q×S three-arm ablation with real H800 INT8 data.

Usage:
  python scripts/l1_integrate_h800_int8.py [--dry-run]

Inputs (must exist):
  results/q_int8_base_gate.csv   — base [64,128,256] H800 TVM int8 latency
  results/q_int8_pairs.csv       — key-pair H800 TVM int8 latencies (default+tuned)
  results/q_int8_pairs_pg.csv    — P_g key-pair latencies (pad64/s1_64/s2_128)
  results/q_int8_ap.json         — real int8 AP on DAIR val (or {"source":"not_measured"})

Outputs:
  results/latency_lut_pyramid_q.json  — updated with h800_tvm_int8_* fields
  results/smoke_pqs_real.json         — P×Q×S ablation with real H800 INT8 data

CALIBER NOTE:
  FP16 reference for ratio comparison = results/latency_lut_pyramid.json (H800 TVM tuned_us)
  NOT latency_lut_pyramid_q.json fp16_p50_ms (that is 4090 TRT, wrong caliber).
"""
import argparse
import json
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
Q_LUT    = RESULTS / "latency_lut_pyramid_q.json"
# !! CALIBER: FP16 reference MUST come from headline LUT (H800 TVM), NOT Q_LUT (4090 TRT)
FP16_LUT = RESULTS / "latency_lut_pyramid.json"


def load_q_lut():
    with open(Q_LUT) as f:
        return json.load(f)


def parse_pairs_csv(csv_path):
    """Returns dict: {(tuple(num_filters), sched) -> lat_us}
    Skips rows where lat_us is RUNNING/BLOCKED/non-numeric."""
    out = {}
    skipped = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nf_raw = row.get("num_filters", "")
            if not nf_raw.strip("[]").strip():
                continue
            try:
                nf = tuple(int(x) for x in nf_raw.strip("[]").split(","))
                sched = row["sched"].strip()
                lat = float(row["lat_us"])   # raises ValueError if RUNNING/BLOCKED
                out[(nf, sched)] = lat
            except (ValueError, KeyError):
                skipped += 1
    if skipped:
        print(f"[INFO] {csv_path.name}: skipped {skipped} RUNNING/BLOCKED rows")
    return out


def parse_base_gate_csv(csv_path):
    """Returns {(tuple(num_filters), sched) -> lat_us} for base gate.
    Skips rows where lat_us is RUNNING/BLOCKED/non-numeric."""
    out = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nf_raw = row.get("width", "")
            if not nf_raw.strip("[]").strip():
                continue
            try:
                nf = tuple(int(x) for x in nf_raw.strip("[]").split(","))
                sched = row.get("sched", "tuned").strip()
                lat = float(row["lat_us"])
                out[(nf, sched)] = lat
            except (ValueError, KeyError):
                pass
    return out


def parse_int8_ap(json_path):
    """Returns list of width records from q_int8_ap.json, or [] if not_measured."""
    with open(json_path) as f:
        data = json.load(f)
    if data.get("source") == "not_measured":
        print(f"[INFO] q_int8_ap.json: not_measured (reason={data.get('reason')})")
        return []
    return data.get("widths", [])


def load_fp16_lut_h800():
    """Load headline FP16 LUT (H800 TVM tuned_us) as {tuple(num_filters): tuned_us}."""
    with open(FP16_LUT) as f:
        d = json.load(f)
    return {tuple(e["num_filters"]): e.get("tuned_us") for e in d.get("widths", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be merged without writing")
    args = ap.parse_args()

    # ── Check inputs ──────────────────────────────────────────────────────────
    base_gate_path = RESULTS / "q_int8_base_gate.csv"
    pairs_path     = RESULTS / "q_int8_pairs.csv"
    pairs_pg_path  = RESULTS / "q_int8_pairs_pg.csv"   # P_g widths from ab403502
    ap_path        = RESULTS / "q_int8_ap.json"

    required = [base_gate_path, pairs_path, ap_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"[ERROR] Missing input files:")
        for p in missing:
            print(f"  {p}")
        print("Waiting for hw-optimizer and sw-optimizer to complete STEP 1/2/3.")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_q_lut()
    fp16_h800 = load_fp16_lut_h800()   # H800 TVM FP16 tuned_us — correct caliber
    pairs = parse_pairs_csv(pairs_path)
    if pairs_pg_path.exists():
        pairs.update(parse_pairs_csv(pairs_pg_path))
        print(f"[INFO] Loaded P_g pairs from {pairs_pg_path}")
    pairs.update(parse_base_gate_csv(base_gate_path))
    ap_records = parse_int8_ap(ap_path)

    print(f"[INFO] Loaded pairs: {list(pairs.keys())}")
    print(f"[INFO] Loaded AP records: {len(ap_records)} widths")

    # ── Merge into Q-LUT ──────────────────────────────────────────────────────
    updated = 0
    for entry in data.get("widths", []):
        nf = tuple(entry["num_filters"])
        if (nf, "default") in pairs:
            entry["h800_tvm_int8_default_us"] = pairs[(nf, "default")]
            entry["_int8_default_source"] = "H800_TVM_int8_real"
            updated += 1
        if (nf, "tuned") in pairs:
            entry["h800_tvm_int8_tuned_us"] = pairs[(nf, "tuned")]
            entry["_int8_tuned_source"] = "H800_TVM_int8_real"

    # Merge AP data
    ap_by_width = {tuple(r["num_filters"]): r for r in ap_records}
    for entry in data.get("widths", []):
        nf = tuple(entry["num_filters"])
        if nf in ap_by_width:
            rec = ap_by_width[nf]
            entry["real_int8_ap70"] = rec["ap70"]
            entry["real_int8_ap50"] = rec.get("ap50")
            entry["real_int8_delta_ap70"] = rec.get("delta_ap70")
            entry["_int8_ap_calib"] = rec.get("calib_method", "unknown")

    # Add metadata
    data["_h800_int8_integrated"] = True
    data["_h800_int8_source"] = "q0_int8_spike + q_int8_pairs (fresh_workdir, subprocess_isolated)"

    print(f"\n[INFO] Updated {updated} widths with H800 TVM int8 latencies.")

    if args.dry_run:
        print("\n[DRY RUN] Would write updated Q-LUT:")
        for entry in data.get("widths", []):
            nf = entry["num_filters"]
            h8d = entry.get("h800_tvm_int8_default_us")
            h8t = entry.get("h800_tvm_int8_tuned_us")
            fp16_sp = entry.get("int8_speedup")
            if h8d or h8t:
                print(f"  {nf}: default={h8d}µs tuned={h8t}µs (4090_speedup={fp16_sp}x)")
        return

    # Write updated Q-LUT
    with open(Q_LUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Written updated Q-LUT to {Q_LUT}")

    # ── Q-rank-flip analysis with real H800 TVM INT8 ─────────────────────────
    print("\n" + "=" * 72)
    print("Q-RANK-FLIP ANALYSIS (H800 TVM INT8 real vs 4090 estimated)")
    print("=" * 72)

    pairs_to_check = [
        ("trap25", (48, 96, 192), "pad64",  (64, 96, 192),  "s1=96"),
        ("mix_b",  (48, 64, 256), "s1_64",  (64, 64, 256),  "s1=64"),
        ("mix_d",  (48,128, 128), "s2_128", (64,128, 128), "s2=128 ★Q-flip pair"),
    ]
    for wg_label, wg, pg_label, pg, shared in pairs_to_check:
        wg_d = pairs.get((wg, "default"))
        wg_t = pairs.get((wg, "tuned"))
        pg_d = pairs.get((pg, "default"))
        pg_t = pairs.get((pg, "tuned"))

        # FP16 H800 TVM (from latency_lut_pyramid.json for comparison)
        print(f"\n  Pair ({shared}): {wg_label} vs {pg_label}")
        if wg_d and pg_d:
            flip_default = pg_d < wg_d
            ratio_d = wg_d / pg_d if pg_d > 0 else float("inf")
            print(f"    INT8 default: {wg_label}={wg_d:.0f}µs vs {pg_label}={pg_d:.0f}µs  "
                  f"ratio={ratio_d:.3f}x  Q-rank-flip={flip_default}")
        if wg_t and pg_t:
            flip_tuned = pg_t < wg_t
            ratio_t = wg_t / pg_t if pg_t > 0 else float("inf")
            print(f"    INT8 tuned:   {wg_label}={wg_t:.0f}µs vs {pg_label}={pg_t:.0f}µs  "
                  f"ratio={ratio_t:.3f}x  Q-rank-flip={flip_tuned}")

        # Compare INT8 speedups — FP16 ref MUST be H800 TVM (fp16_h800), NOT 4090 TRT
        for label, nf in [(wg_label, wg), (pg_label, pg)]:
            fp16_ref_us = fp16_h800.get(nf)       # H800 TVM FP16 tuned (µs) ← correct caliber
            int8_t      = pairs.get((nf, "tuned")) # H800 TVM INT8 tuned (µs)
            if fp16_ref_us and int8_t:
                sp = fp16_ref_us / int8_t          # both in µs, no unit conversion needed
                print(f"    {label} INT8-tuned/FP16-tuned speedup (H800 TVM, same caliber): {sp:.3f}x")

    # ── Q-AMPLIFICATION TABLE (headline result for L1) ───────────────────────
    print("\n" + "=" * 72)
    print("Q-AMPLIFICATION TABLE — H800 TVM INT8-tuned vs FP16-tuned ratio")
    print("(one caliber: both from H800 TVM MetaSchedule)")
    print("=" * 72)
    print(f"  {'Pair':10} {'W_g':8} {'P_g':8} {'FP16_Wg':10} {'FP16_Pg':10} {'FP16_r':8} | "
          f"{'INT8_Wg':10} {'INT8_Pg':10} {'INT8_r':8} {'Δratio':8} {'amplified?':12}")
    for wg_label, wg, pg_label, pg, shared in pairs_to_check:
        fp16_wg = fp16_h800.get(wg)
        fp16_pg = fp16_h800.get(pg)
        int8_wg = pairs.get((wg, "tuned"))
        int8_pg = pairs.get((pg, "tuned"))
        if fp16_wg and fp16_pg:
            fp16_r = fp16_wg / fp16_pg
            if int8_wg and int8_pg:
                int8_r = int8_wg / int8_pg
                delta  = int8_r - fp16_r
                amp    = "YES ↑ AMP" if int8_r > fp16_r else "no"
                print(f"  {shared:10} {wg_label:8} {pg_label:8} "
                      f"{fp16_wg:10.0f} {fp16_pg:10.0f} {fp16_r:8.3f}x | "
                      f"{int8_wg:10.0f} {int8_pg:10.0f} {int8_r:8.3f}x {delta:+8.3f} {amp}")
            else:
                print(f"  {shared:10} {wg_label:8} {pg_label:8} "
                      f"{fp16_wg:10.0f} {fp16_pg:10.0f} {fp16_r:8.3f}x | "
                      f"{'MISSING':10} {'MISSING':10} {'?':8}")
        else:
            print(f"  {shared:10}: FP16 reference missing in headline LUT — check latency_lut_pyramid.json")

    # ── Run detect_q_rank_flip_pairs with real H800 TVM INT8 ────────────────
    print("\n" + "=" * 72)
    print("Q-RANK-FLIP DETECTION WITH REAL H800 TVM INT8")
    print("=" * 72)
    try:
        sys.path.insert(0, str(ROOT))
        from framework.search_three_arm import (
            LatencyLUT, APModel, QLookup, detect_q_rank_flip_pairs,
            SMOKE_WIDTHS_PQS, Q_LUT_JSON
        )
        lut  = LatencyLUT()
        apm  = APModel()
        qlut = QLookup(q_path=Q_LUT)   # uses freshly written Q-LUT with H800 int8
        # detect at default schedule (where structural Q-rank-flip happens)
        q_pairs_d = detect_q_rank_flip_pairs(SMOKE_WIDTHS_PQS, lut, apm, qlut, sched="default")
        q_pairs_t = detect_q_rank_flip_pairs(SMOKE_WIDTHS_PQS, lut, apm, qlut, sched="tuned")
        print(f"Q-rank-flip @ default: {len(q_pairs_d)} pairs")
        for p in q_pairs_d:
            print(f"  {p['wg']} vs {p['pg']} ratio={p['q_rank_flip_ratio']}x  source={p['source']}")
        print(f"Q-rank-flip @ tuned:   {len(q_pairs_t)} pairs")
        for p in q_pairs_t:
            print(f"  {p['wg']} vs {p['pg']} ratio={p['q_rank_flip_ratio']}x  source={p['source']}")

        print("\n>>> NOW RUN FINAL P×Q×S ABLATION:")
        print("  python -m framework.search_three_arm --q-mode --seeds 12 --budget 90 --pop 10")
        print("  2>&1 | tee results/smoke_pqs_real.txt")
    except ImportError as e:
        print(f"[WARN] Could not import framework: {e}")
        print("Run manually: python -m framework.search_three_arm --q-mode --seeds 12")


if __name__ == "__main__":
    main()
