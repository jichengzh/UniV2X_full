#!/bin/bash
# L1 STEP4 final assembly — run this after q_int8_base_gate.csv + q_int8_pairs.csv arrive with REAL data
# Usage: bash scripts/l1_final_step4.sh

set -e
cd /home/jichengzhi/V2X

PYTHON=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python

echo "=== STEP 4A: Validate input files ==="
$PYTHON - << 'PYEOF'
import csv, sys

def check(path, required_cols):
    rows = list(csv.DictReader(open(path)))
    print(f"  {path}: {len(rows)} rows")
    for r in rows:
        for col in required_cols:
            val = r.get(col, '')
            if val in ('BLOCKED', '', 'nan', 'None'):
                print(f"  ERROR: {path} row {r} has invalid {col}={val!r}")
                sys.exit(1)
        try:
            float(r['lat_us'])
        except (ValueError, KeyError):
            print(f"  ERROR: {path} non-numeric lat_us: {r}")
            sys.exit(1)
    print(f"  OK: {len(rows)} rows, all lat_us numeric")

check("results/q_int8_base_gate.csv", ['lat_us', 'sched'])
check("results/q_int8_pairs.csv", ['lat_us', 'num_filters', 'sched'])
print("All inputs valid — proceeding to integration")
PYEOF

echo ""
echo "=== STEP 4B: Merge H800 TVM int8 into Q-LUT ==="
$PYTHON scripts/l1_integrate_h800_int8.py

echo ""
echo "=== STEP 4C: Final P×Q×S ablation (12 seeds, 9 widths, real H800 TVM int8) ==="
$PYTHON -m framework.search_three_arm --q-mode --seeds 12 --budget 90 --pop 10 \
  2>&1 | tee results/smoke_pqs_real.txt

echo ""
echo "=== STEP 4D: Save structured final results ==="
$PYTHON - << 'PYEOF'
import json, re, numpy as np
from scipy.stats import wilcoxon

content = open("results/smoke_pqs_real.txt").read()
joint  = [float(x) for x in re.findall(r'A-joint-PQS\s+HV=([\d.e+]+)', content)]
serial = [float(x) for x in re.findall(r'A-serial-PQS\s+HV=([\d.e+]+)', content)]
nos    = [float(x) for x in re.findall(r'A-noS-PQS\s+HV=([\d.e+]+)', content)]

if not joint:
    print("ERROR: could not parse HV values from smoke_pqs_real.txt")
    exit(1)

diffs = [j - s for j, s in zip(joint, serial)]
stat, p = wilcoxon(diffs)

result = {
    "_meta": {"status": "FINAL (real H800 TVM int8)", "n_widths": 9, "seeds": 12},
    "hv": {
        "A-joint-PQS":  {"mean": float(np.mean(joint)),  "pct": 100.0},
        "A-serial-PQS": {"mean": float(np.mean(serial)), "pct": float(np.mean(serial)/np.mean(joint)*100)},
        "A-noS-PQS":    {"mean": float(np.mean(nos)),    "pct": float(np.mean(nos)/np.mean(joint)*100)},
    },
    "wilcoxon": {"stat": float(stat), "p": float(p), "n": len(diffs), "all_positive": all(d>0 for d in diffs)},
}
with open("results/smoke_pqs_final.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== FINAL P×Q×S RESULT (real H800 TVM int8) ===")
print(f"  A-joint:  {np.mean(joint):.0f} HV (100%)")
print(f"  A-serial: {np.mean(serial):.0f} HV ({np.mean(serial)/np.mean(joint)*100:.1f}%)")
print(f"  A-noS:    {np.mean(nos):.0f} HV ({np.mean(nos)/np.mean(joint)*100:.1f}%)")
print(f"  Wilcoxon: stat={stat}, p={p:.4e}, all_pos={all(d>0 for d in diffs)}")
print(f"  Saved: results/smoke_pqs_final.json")
PYEOF

echo ""
echo "=== DONE === Report results/smoke_pqs_final.json to team-lead"
