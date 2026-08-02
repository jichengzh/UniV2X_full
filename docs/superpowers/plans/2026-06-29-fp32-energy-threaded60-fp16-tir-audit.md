# FP32 Energy Threaded60 And FP16 TIR Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remeasure all 60 FP32 energy points with the FP16/INT8-style `threaded_window` sampler and audit representative FP16 TIR lowering evidence.

**Architecture:** Keep old FP32 `post_sync_per_iter` rows as historical evidence. Write threaded FP32 60-point rows/raw to separate artifacts first, then promote them into the canonical table only after coverage and sanity checks pass. Audit FP16 lowering from the same representative labels used in the 5-point energy smoke.

**Tech Stack:** Python, TVM Relax/MetaSchedule, H800 CUDA, `nvidia-smi` power telemetry, Stage2 LUT JSONL/Markdown exports.

---

### Task 1: Launch FP32 Threaded60 Energy Remeasure

**Files:**
- Read: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json`
- Read: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_original60_energy_remeasured_rows_v1.jsonl`
- Write: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_original60_energy_threaded60_rows_v1.jsonl`
- Write: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp32_energy_remeasure/20260629_fp32_threaded_window_original60_60labels/`

- [ ] **Step 1: Build a 60-label queue from canonical summary**

Run a Python launcher that maps each FP32 label to canonical width/candidate metadata, ONNX path `/exdata/jichengzhi/s2_tvm/models/{label}_backbone.onnx`, and workdir `/exdata/jichengzhi/s2_tvm/workdirs/{label}`.

- [ ] **Step 2: Run four H800 lanes**

Use GPUs `0,3,5,6`. For most labels use `--energy-schedule-policy metaschedule_tuned`; use `default` only for labels already known to fail tuned route: `s2_096`, `lhc_17`, `frontier_25`, `frontier_26`.

Every run must include:

```bash
--energy-sampling-mode threaded_window \
--energy-warmup-iters 20 \
--energy-measure-iters 300 \
--energy-min-active-s 5 \
--energy-sync-interval-iters 50
```

- [ ] **Step 3: Verify coverage**

Run:

```bash
python - <<'PY'
from pathlib import Path
p=Path('/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_original60_energy_threaded60_rows_v1.jsonl')
print(sum(1 for _ in p.open()))
PY
```

Expected: `60`.

### Task 2: Audit FP16 TIR Lowering

**Files:**
- Read: `/exdata/jichengzhi/s2_tvm/workdirs/{label}/`
- Read: `/exdata/jichengzhi/s2_tvm/fp16_true_smoke_20260627/20260628_fp16_true_original60_energy_batch001_60labels/{label}/layer_precision_summary.json`
- Write: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_tir_lowering_5label_audit_latest.md`
- Write: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_tir_lowering_5label_audit_latest.json`

- [ ] **Step 1: Inspect representative labels**

Inspect `s0_024`, `lhc_07`, `lhc_20`, `s0_056`, `frontier_25`.

- [ ] **Step 2: Search TIR artifacts**

For each label, search workdir and available TVM artifacts for `wmma`, `mma`, `tensorcore`, `float16`, `tir`, `cast`, and `layout`.

- [ ] **Step 3: Record conclusion**

Classify each label as one of:
- `tensorcore_evidence_present`
- `fp16_dtype_present_no_tensorcore_evidence`
- `tir_artifact_missing`
- `mixed_or_inconclusive`

### Task 3: Generate Review And Update Canonical Table

**Files:**
- Write: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_threaded_window_energy_60label_review_latest.md`
- Write: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp32_threaded_window_energy_60label_review_latest.json`
- Modify if validated: `/home/jichengzhi/V2X/scripts/stage2_generate_original60_quant_state_coverage.py`

- [ ] **Step 1: Compare old FP32, threaded FP32, and FP16**

Compute dynamic watt and J/inference medians, min/max, and schedule counts.

- [ ] **Step 2: Promote threaded60 rows only if all 60 rows are measured**

If 60/60 rows pass sanity checks, update canonical generation to prefer `fp32_original60_energy_threaded60_rows_v1.jsonl`.

- [ ] **Step 3: Regenerate total table**

Run:

```bash
python scripts/stage2_generate_original60_quant_state_coverage.py \
  --output-root /home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627
```

Expected: `energy_rows: 180`, FP32 energy `60/60 measured`, FP32 energy source `true_measurement`.
