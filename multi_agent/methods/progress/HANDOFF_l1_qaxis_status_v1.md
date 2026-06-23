# HANDOFF: L1 Q-axis Integration Status

**Date**: 2026-06-21
**Owner**: data-orchestrator
**Status**: PARTIAL — STEP 3+4-prelim done; STEP 1+2 (H800 TVM int8) pending hw-optimizer GPU slot

---

## What's done

### STEP 3 — Real INT8 AP ✅
- Source: `results/q_int8_ap.json` (DAIR val 1789, TRT MinMaxCalibrator, 4dp)
- base: ap70=0.622809 (Δ=-0.0081), p50: 0.554228 (Δ=-0.0099), p75: 0.523636 (Δ=-0.0064)
- Median Δap70 = **-0.008** (updated from -0.007 estimate in framework)
- `search_three_arm.py::_INT8_AP_DELTA_MEDIAN` = -0.008 ✅

### STEP 4 — P×Q×S Preliminary ✅ (estimated INT8, 9 widths)
**9 widths**: base/p50/p75 + pair1(trap25/pad64) + pair2(mix_b/s1_64) + pair3(mix_d/s2_128)

| Arm | Mean HV | % of joint |
|-----|---------|------------|
| A-joint-PQS | 7580 | 100.0% |
| A-serial-PQS | 6448 | **85.1%** |
| A-noS-PQS | 4380 | 57.8% |

Wilcoxon joint vs serial: **stat=0, p=4.9e-4**, all 12 seeds positive.

Q-structural claim: **PASS** — A-serial never explores INT8 by construction.

Q-rank-flip (4090 TRT estimated): **1 pair** — mix_d[48,128,128] vs s2_128[64,128,128]:
- FP16 default: mix_d=42447µs (faster), s2_128=47435µs
- INT8 default: mix_d=36826µs, **s2_128=34060µs** (flip!)
- INT8 speedup: mix_d=1.153×, s2_128=**1.393×**
- Q-flip ratio: 1.081× (labeled MEASURED: both have 4090 TRT data)

### Dataset ✅
- Added p75 INT8 row (`pyr_16-32-64_INT8`): lat=0.639ms, ap70=0.523636
- Total: 66 rows, 60 real complete (lat+AP), 27 INT8 complete

---

## CRITICAL CAVEAT — H800 TVM INT8 may flip the Q-rank-flip!

`results/s2_2b_screenA_int8.csv` shows H800 TVM INT8 single-conv:
- **w48: 3.172µs vs w64: 4.437µs → w48 is 40% FASTER on H800!**
- This is **OPPOSITE** to 4090 TRT pattern (where s0=48 alignment trap causes smaller speedup)

If H800 TVM INT8 full backbone follows single-conv pattern:
- mix_d[48,128,128] might get **larger** INT8 speedup than estimated (>1.153×)
- s2_128[64,128,128] might get **smaller** INT8 speedup (<1.393×)
- **Q-rank-flip may disappear on H800 TVM**

This does NOT invalidate the structural claim (A-serial still never explores INT8).
It may reduce the Q-contribution to HV gap (but P×S alone is already Wilcoxon significant).

---

## What's pending

### STEP 1 — H800 TVM int8 base gate
- Script: hw-optimizer running `q0_int8_spike.py` on H800
- Blocked: H800 GPUs 4-7 busy with other TVM experiments
- Output: `results/q_int8_base_gate.csv` (base [64,128,256] INT8 latency)

### STEP 2 — H800 TVM int8 key-pair latencies
- Targets: trap25[48,96,192], pad64[64,96,192], mix_b[48,64,256], s1_64[64,64,256], mix_d[48,128,128], s2_128[64,128,128]
- Both default and tuned schedule, fresh workdir, subprocess isolation
- Output: `results/q_int8_pairs.csv`

### Final STEP 4 (after 1+2 arrive)
```bash
python scripts/l1_integrate_h800_int8.py  # merge H800 int8 into Q-LUT
python -m framework.search_three_arm --q-mode --seeds 12 --budget 90 --pop 10 2>&1 | tee results/smoke_pqs_real.txt
```
Then check if Q-rank-flip survives with real H800 data, and compute final HV gap.

---

## Key files

| File | Purpose |
|------|---------|
| `results/q_int8_ap.json` | Real INT8 AP (gold) |
| `results/latency_lut_pyramid_q.json` | 4090 TRT Q-ratios + H800 TVM FP16 |
| `results/smoke_pqs_v2_summary.json` | Preliminary P×Q×S results |
| `scripts/l1_integrate_h800_int8.py` | Merge H800 TVM int8 → re-run ablation |
| `framework/search_three_arm.py` | Updated SMOKE_WIDTHS_PQS (9 widths), QLookup H800 int8 |
| `figure/pqs_ablation_preliminary.png` | Ablation bar chart + Q-rank-flip |
| `multi_agent/data/dataset_v2.csv` | 66 rows, includes p75 INT8 |
