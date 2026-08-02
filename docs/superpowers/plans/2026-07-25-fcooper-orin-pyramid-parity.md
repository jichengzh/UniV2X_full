# F-Cooper Orin Pyramid-Parity Table 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the five F-Cooper Orin controls under the same TensorRT 8.5 default-builder protocol used by Pyramid and populate the reserved F-Cooper cells in paper Table 2 from measured evidence.

**Architecture:** Keep the frozen F-Cooper subnet, sources, inputs, AP bridge, latency protocol, and energy formula unchanged. Replace only the unsupported builder-level 0/5 contract with an explicit `trt85_default` policy and `builder_optimization_level=null`, then regenerate the manifest and run all four TRT arms plus the native control on Orin. The fail-closed finalizer remains the only source allowed to provide Table 2 values.

**Tech Stack:** Python, pytest, TensorRT 8.5.2.2, PyTorch/CUDA, tegrastats, OPV2V, LaTeX.

## Global Constraints

- Only F-Cooper is in scope; do not rerun Pyramid, CoDriving, H800, TVM, or CPU experiments.
- Preserve OPV2V full-2170, sample batch 1, dense agent batch 5, input `[5,64,512,512]`, and `post_scatter_backbone_shrinker`.
- Four TRT rows use the target Orin's TensorRT 8.5 default builder; do not set or claim builder optimization levels 0/5.
- Schedule-only remains strict FP32 with TF32 disabled and inspector evidence; the other three TRT controls remain FP16.
- Preserve warmup 20, iterations 300, repeats 5, CUDA-event compute-only latency and VIN_SYS_5V0 energy.
- Populate Table 2 only after the finalizer emits complete measured evidence; never estimate missing values.
- Do not alter Table 1 or Table 3 values, table font size, or three-table page layout.

---

### Task 1: Encode the TensorRT 8.5 default-builder contract

**Files:**
- Modify: `framework/tests/test_fcooper_orin_five_config.py`
- Modify: `framework/tests/test_fcooper_orin_runner.py`
- Modify: `framework/tests/test_fcooper_orin_finalize.py`
- Modify: `tools/orin_deploy/fcooper_orin_five_config.py`
- Modify: `tools/orin_deploy/fcooper_orin_runner.py`
- Modify: `tools/orin_deploy/fcooper_orin_finalize.py`

**Interfaces:**
- Consumes: frozen five-arm identity and source SHA contract.
- Produces: manifest/build/finalizer evidence with `builder_policy="trt85_default"` and `builder_optimization_level=null`.

- [ ] Add failing tests that reject non-default or fabricated builder levels.
- [ ] Run focused tests and confirm the new assertions fail.
- [ ] Update manifest, runner, receipt, and finalizer validation.
- [ ] Run focused tests and syntax checks.

### Task 2: Update the formal handoff protocol

**Files:**
- Modify: `multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/36_7_14_交接文档_FCooper_Orin五配置实测反思与Table2补齐计划_v1.md`

**Interfaces:**
- Consumes: verified Pyramid TRT 8.5 build protocol.
- Produces: an explicit record that level 0/5 parity is not claimed and that all three Orin models use the same default-builder protocol.

- [ ] Replace level 0/5 requirements and the obsolete hard blocker.
- [ ] Preserve precision, numerical, AP, latency, and energy gates.
- [ ] Record the evidence limitation for cross-hardware scheduling claims.

### Task 3: Regenerate and run F-Cooper evidence

**Files:**
- Update artifacts under: `results/lane_c_orin_fcooper_stage6_five_config_20260724/`

**Interfaces:**
- Consumes: frozen sources, real held-out tensor, Orin runner, secure ephemeral sudo input.
- Produces: four local Orin engines, numeric reports, five latency/energy reports, five full-2170 AP reports, and final evidence CSV/JSON.

- [ ] Regenerate the canonical manifest under the revised protocol.
- [ ] Build and inspect the four TRT engines locally on Orin.
- [ ] Run held-out numerical gates.
- [ ] Run five 20/300/5 latency and VIN_SYS_5V0 energy measurements.
- [ ] Run five full-2170 AP bridges.
- [ ] Run the fail-closed finalizer and inspect all SHA bindings.

### Task 4: Populate and verify paper Table 2

**Files:**
- Modify: `multi_agent/paper/Latex/AnonymousSubmission2027.tex`

**Interfaces:**
- Consumes: finalizer-generated five-row F-Cooper table.
- Produces: 15 measured F-Cooper cells formatted to two decimals.

- [ ] Snapshot Table 1 and Table 3 numerical content.
- [ ] Replace only the 15 reserved F-Cooper Table 2 cells.
- [ ] Preserve the existing `\small` font and three-table page layout.
- [ ] Compile LaTeX and compare Table 1/Table 3 snapshots.
- [ ] Run focused tests, SHA audit, and final evidence completeness checks.
