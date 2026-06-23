"""L4 update: integrate new pair latency + AP data into LUT and AP model JSON.

After hw-optimizer reports l4_b1_latency_new_pairs.csv and sw-optimizer reports
l4_b2_ap_finetune.json, this script:

1. Adds 5 new latency entries to results/latency_lut_pyramid.json
2. Adds 6 new rows to results/ap70_model_pyramid.json "table"
   (W_g row from finetune AP, P_g row with SAME AP by zero-pad identity)
3. Re-runs detect_wg_pg_pairs to verify new pairs are detected
4. Reports pairs + shipped status + AP70 bands

Run after results arrive:
  python scripts/phase2/l4_update_lut_and_ap.py

Or with explicit data (for testing before results are ready):
  python scripts/phase2/l4_update_lut_and_ap.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LUT_JSON = RESULTS / "latency_lut_pyramid.json"
AP_JSON = RESULTS / "ap70_model_pyramid.json"
LAT_CSV = RESULTS / "l4_b1_latency_new_pairs.csv"
AP_FINETUNE_JSON = RESULTS / "l4_b2_ap_finetune.json"

# Mapping: label → (W_g_width, P_g_width)
PAIR_DEFS = {
    "pair4": ((48, 96, 256), (64, 96, 256)),  # P_g=iso_s1 already in LUT
    "pair5": ((48, 32, 128), (64, 32, 128)),
    "pair6": ((48, 64, 192), (64, 64, 192)),
}

# iso_s1=[64,96,256] is already in LUT — don't re-add
ALREADY_IN_LUT = {(64, 96, 256)}


def load_lut():
    data = json.loads(LUT_JSON.read_text())
    existing = {tuple(w["num_filters"]): w for w in data["widths"]}
    return data, existing


def load_ap_model():
    data = json.loads(AP_JSON.read_text())
    existing_table = {tuple(r["num_filters"]): r for r in data.get("table", [])}
    return data, existing_table


def parse_latency_csv(path: Path) -> dict:
    """Returns {tuple(num_filters): {"default_us": float, "tuned_us": float, "label": str}}."""
    result = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nf_str = row["num_filters"].strip("[]").replace(" ", "")
            nf = tuple(int(x) for x in nf_str.split(","))
            result[nf] = {
                "label": row["label"],
                "default_us": float(row["default_us"]),
                "tuned_us": float(row["tuned_us"]),
                "gpu": row.get("gpu", "H800"),
                "n_trials": row.get("n_trials", "?"),
                "notes": row.get("notes", ""),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Use placeholder data if results not ready")
    parser.add_argument("--lat-csv", default=str(LAT_CSV))
    parser.add_argument("--ap-json", default=str(AP_FINETUNE_JSON))
    args = parser.parse_args()

    lat_csv_path = Path(args.lat_csv)
    ap_json_path = Path(args.ap_json)

    # -----------------------------------------------------------------------
    # Check data availability
    # -----------------------------------------------------------------------
    if not lat_csv_path.exists():
        if args.dry_run:
            print(f"[DRY-RUN] {lat_csv_path} not found; using placeholder data")
            lat_data = {
                (48, 96, 256): {"label": "wg_pair4", "default_us": 44000.0, "tuned_us": 20500.0},
                (48, 32, 128): {"label": "wg_pair5", "default_us": 28000.0, "tuned_us": 13500.0},
                (64, 32, 128): {"label": "pg_pair5", "default_us": 32000.0, "tuned_us": 3200.0},
                (48, 64, 192): {"label": "wg_pair6", "default_us": 37000.0, "tuned_us": 18000.0},
                (64, 64, 192): {"label": "pg_pair6", "default_us": 43000.0, "tuned_us": 4800.0},
            }
        else:
            print(f"ERROR: {lat_csv_path} not found. Run hw-optimizer first or use --dry-run")
            sys.exit(1)
    else:
        lat_data = parse_latency_csv(lat_csv_path)
        print(f"Loaded latency data for {len(lat_data)} widths from {lat_csv_path}")

    if not ap_json_path.exists():
        if args.dry_run:
            print(f"[DRY-RUN] {ap_json_path} not found; using placeholder AP data")
            ap_data = {
                "wg_pair4": {"num_filters": [48, 96, 256], "ap70": 0.6140, "ap50": 0.779},
                "wg_pair5": {"num_filters": [48, 32, 128], "ap70": 0.5750, "ap50": 0.745},
                "wg_pair6": {"num_filters": [48, 64, 192], "ap70": 0.5990, "ap50": 0.763},
            }
        else:
            print(f"ERROR: {ap_json_path} not found. Run sw-optimizer first or use --dry-run")
            sys.exit(1)
    else:
        ap_data = json.loads(ap_json_path.read_text())
        print(f"Loaded AP finetune data for {list(ap_data.keys())} from {ap_json_path}")

    # -----------------------------------------------------------------------
    # Verify rank-flip structure before updating
    # -----------------------------------------------------------------------
    print("\n=== Rank-flip verification ===")
    for pair_name, (wg, pg) in PAIR_DEFS.items():
        wg_lat = lat_data.get(wg)
        pg_lat = lat_data.get(pg)

        # For pair4, P_g=iso_s1 is already in LUT
        if pg in ALREADY_IN_LUT and pg_lat is None:
            lut_data, lut_existing = load_lut()
            if pg in lut_existing:
                pg_lat = {
                    "label": "iso_s1",
                    "default_us": lut_existing[pg]["default_us"],
                    "tuned_us": lut_existing[pg]["tuned_us"],
                }
                print(f"  {pair_name}: P_g={pg} already in LUT: default={pg_lat['default_us']}µs tuned={pg_lat['tuned_us']}µs")

        if wg_lat is None or pg_lat is None:
            print(f"  {pair_name}: MISSING data wg={wg_lat is not None} pg={pg_lat is not None}")
            continue

        wd, wt = wg_lat["default_us"], wg_lat["tuned_us"]
        pd, pt = pg_lat["default_us"], pg_lat["tuned_us"]

        default_flip = wd < pd  # W_g faster at default
        tuned_flip = pt < wt    # P_g faster at tuned (rank flip)
        flip_exists = default_flip and tuned_flip

        status = "RANK-FLIP ✓" if flip_exists else "NO FLIP ✗"
        ratio = wt / pt if pt > 0 else float("inf")
        print(f"  {pair_name}: W_g={wg} default={wd:.0f}µs tuned={wt:.0f}µs "
              f"| P_g={pg} default={pd:.0f}µs tuned={pt:.0f}µs "
              f"| iso-AP-ratio={ratio:.2f}× | {status}")
        if not flip_exists:
            print(f"    WARNING: No rank-flip for {pair_name}! "
                  f"default: W_g {'faster' if default_flip else 'slower'}, "
                  f"tuned: P_g {'faster' if tuned_flip else 'slower'}")

    # -----------------------------------------------------------------------
    # Update latency_lut_pyramid.json
    # -----------------------------------------------------------------------
    print("\n=== Updating latency LUT ===")
    lut_json_data, lut_existing = load_lut()

    added_lut = []
    for nf, info in lat_data.items():
        if nf in ALREADY_IN_LUT:
            print(f"  SKIP {nf} (already in LUT as iso_s1)")
            continue
        if nf in lut_existing:
            print(f"  SKIP {nf} (already in LUT with default={lut_existing[nf]['default_us']:.0f}µs)")
            continue
        new_entry = {
            "num_filters": list(nf),
            "label": info["label"],
            "default_us": info["default_us"],
            "tuned_us": info["tuned_us"],
            "_source": f"L4 H800 TVM fp16 {info.get('gpu','H800')} {info.get('n_trials','?')}trials",
            "_note": info.get("notes", ""),
        }
        lut_json_data["widths"].append(new_entry)
        lut_existing[nf] = new_entry
        added_lut.append(nf)
        print(f"  + {info['label']} {nf}: default={info['default_us']:.0f}µs tuned={info['tuned_us']:.0f}µs")

    if not args.dry_run and added_lut:
        LUT_JSON.write_text(json.dumps(lut_json_data, indent=2))
        print(f"Saved {LUT_JSON} (+{len(added_lut)} entries)")
    elif args.dry_run:
        print(f"[DRY-RUN] Would add {len(added_lut)} entries to {LUT_JSON}")

    # -----------------------------------------------------------------------
    # Update ap70_model_pyramid.json table
    # -----------------------------------------------------------------------
    print("\n=== Updating AP model table ===")
    ap_json_data, ap_existing = load_ap_model()

    added_ap = []
    for pair_name, (wg, pg) in PAIR_DEFS.items():
        wg_key = f"wg_{pair_name.replace('pair','pair')}"  # e.g., wg_pair4
        if wg_key not in ap_data:
            print(f"  SKIP {pair_name}: no AP data for {wg_key}")
            continue

        ap70 = ap_data[wg_key]["ap70"]

        # Add W_g
        if wg not in ap_existing:
            wg_entry = {
                "num_filters": list(wg),
                "ap70": ap70,
                "src": f"L4 finetune {wg_key} stage_a protocol DAIR_val_1789",
            }
            ap_json_data["table"].append(wg_entry)
            ap_existing[wg] = wg_entry
            added_ap.append(wg)
            print(f"  + W_g {wg}: AP70={ap70:.4f}")
        else:
            print(f"  SKIP W_g {wg} (already in table)")

        # Add P_g with SAME AP70 (zero-pad identity — AP exactly equal by construction)
        if pg not in ap_existing:
            pg_entry = {
                "num_filters": list(pg),
                "ap70": ap70,
                "src": f"L4 zero-pad identity from {wg_key} (AP=W_g AP by construction)",
            }
            ap_json_data["table"].append(pg_entry)
            ap_existing[pg] = pg_entry
            added_ap.append(pg)
            print(f"  + P_g {pg}: AP70={ap70:.4f} (zero-pad identity, same as W_g)")
        else:
            # For iso_s1=[64,96,256] which might already be in table from calibration
            existing_ap = ap_existing[pg]["ap70"]
            print(f"  NOTE P_g {pg} already in table with AP70={existing_ap:.4f}; "
                  f"will update to {ap70:.4f} to enforce zero-pad identity")
            ap_existing[pg]["ap70"] = ap70
            ap_existing[pg]["src"] += f" [updated L4: zero-pad identity from {wg_key}]"

    if not args.dry_run and added_ap:
        AP_JSON.write_text(json.dumps(ap_json_data, indent=4))
        print(f"Saved {AP_JSON} (+{len(added_ap)} entries to table)")
    elif args.dry_run:
        print(f"[DRY-RUN] Would add {len(added_ap)} entries to {AP_JSON} table")

    # -----------------------------------------------------------------------
    # Verify detect_wg_pg_pairs would find the new pairs
    # -----------------------------------------------------------------------
    print("\n=== Running detect_wg_pg_pairs on updated data ===")
    sys.path.insert(0, str(ROOT))

    # Reload LUT with new data in memory (don't re-read from disk in dry-run)
    if args.dry_run:
        # Simulate: inject the new entries into the LatencyLUT
        pass

    try:
        from framework.search_three_arm import LatencyLUT, APModel, detect_wg_pg_pairs, candidate_widths
        lut = LatencyLUT()
        apm = APModel()

        # Inject new lat data into lut.direct (in case dry-run)
        for nf, info in lat_data.items():
            lut.direct[nf] = {"default_us": info["default_us"], "tuned_us": info["tuned_us"]}
        # Also inject iso_s1 if not there
        if (64, 96, 256) not in lut.direct:
            lut.direct[(64, 96, 256)] = {"default_us": 51622.0, "tuned_us": 6911.0}

        # Inject new AP data
        for pair_name, (wg, pg) in PAIR_DEFS.items():
            wg_key = f"wg_{pair_name}"
            if wg_key in ap_data:
                ap70 = ap_data[wg_key]["ap70"]
                apm.exact[wg] = ap70
                apm.exact[pg] = ap70  # zero-pad identity
                print(f"  Injected AP70={ap70:.4f} for pair {pair_name} ({wg}/{pg})")

        apm._build_interp()

        # Get candidate widths (all in LUT)
        all_widths = list(lut.direct.keys())
        pairs = detect_wg_pg_pairs(all_widths, lut, apm)

        print(f"\n  detect_wg_pg_pairs found {len(pairs)} pairs:")
        for p in pairs:
            wg_r = p["wg_tuned_ratio"]
            pg_r = p["pg_tuned_ratio"]
            ratio = p["iso_ap_latency_ratio"]
            ap = p["ap70"]
            print(f"    wg={p['wg']} pg={p['pg']} AP70={ap:.4f} "
                  f"wg_tune={wg_r:.2f}× pg_tune={pg_r:.2f}× iso_ratio={ratio:.2f}×")

        # Shipped analysis
        print(f"\n=== Global Pareto + shipped analysis ===")
        pareto = [
            (483.63, 0.530, "p75/tuned"),
            (3010.51, 0.564, "p50/tuned"),
            (5506.57, 0.637, "s2_128/tuned"),
        ]
        for pair in pairs:
            pg = pair["pg"]
            pg_lat = pair["pg_tuned_us"]
            pg_ap = pair["ap70"]
            dominated = any(
                pg_lat >= plat and pg_ap <= pap
                for plat, pap, _ in pareto
            )
            status = "SHIPPED (on global Pareto)" if not dominated else "mechanism-only"
            print(f"  P_g={pg} lat={pg_lat:.0f}µs AP={pg_ap:.4f} → {status}")

    except Exception as e:
        print(f"  detect_wg_pg_pairs simulation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== AP cliff summary ===")
    cliff_keys = ["cliff_a_wpg8", "cliff_b_wpg8"]
    print("prune_config | num_filters | approx_reduction | AP70")
    # Existing stage_a anchors
    anchors = [
        ("base",  [64,128,256], "0%",   0.6309),
        ("p25",   [48,96,192],  "25%",  0.5905),
        ("p50",   [32,64,128],  "50%",  0.5641),
        ("p75",   [16,32,64],   "75%",  0.5300),
    ]
    for label, nf, red, ap in anchors:
        print(f"  {label:12} {nf} {red:6} {ap:.4f}")
    for k in cliff_keys:
        if k in ap_data:
            info = ap_data[k]
            nf = info["num_filters"]
            ap70 = info.get("ap70")
            status = info.get("status", "?")
            if ap70:
                # Estimate reduction
                base = [64, 128, 256]
                red = sum(b - n for b, n in zip(base, nf)) / sum(base) * 100
                print(f"  {k:12} {nf} ~{red:.0f}% {ap70:.4f}  ({status})")
            else:
                print(f"  {k:12} {nf} [INFEASIBLE or PENDING] {status}")
        else:
            print(f"  {k:12} PENDING")


if __name__ == "__main__":
    main()
