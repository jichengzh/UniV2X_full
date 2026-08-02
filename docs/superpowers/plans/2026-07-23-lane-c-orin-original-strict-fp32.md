# Lane C Orin Original Strict-FP32 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the Original Pyramid `(64,128,256)` multiscale backbone on Orin in strict FP32 under the same numerical, latency, AP, and power boundaries used for the deployable `(16,32,64)` FP16 result.

**Architecture:** Extend the existing Lane C backbone runner with an explicit strict-FP32 build contract that clears TensorRT TF32 and rejects FP16/INT8 inspector evidence. Bind the three-output channel contract to `(64,128,256)`, run a freshly built Orin engine on the locked real held-out inputs, and reuse the full_1789 bridge only after strengthening its output-shape gate. Seal all evidence in a new immutable result root.

**Tech Stack:** Python 3, NumPy, PyTorch, ONNX, ONNX Runtime/TensorRT 8.5.2.2, CUDA events, tegrastats, unittest.

## Global Constraints

- Scope is only Pyramid `get_multiscale_feature` / multiscale-backbone subnet; fusion, shrinker, cls/reg/dir heads remain PyTorch in AP evaluation.
- Source checkpoint SHA is `4ccc6fe1f7cc13b5d1294f74014b01e849cc8e90b69cbded14158e1999fa42b2`.
- Source config SHA is `f55a6ad8cff9fa522ce74fe9ef4d85d682314455278240ce7d5ec998d567f540`.
- Source ONNX is `results/stage35_gold32_supplement_v1_20260713/pyramid_sources_shape_repaired_v1/064x128x256/pyramid_064x128x256_multiscale.onnx`, SHA `3b125b3c77f7484d8b3a46ae674446f30e663c53fa38431c3f2cb96ae9cbb0ff`.
- ONNX input is `[2,64,128,256]`; ordered outputs are `[2,64,128,256]`, `[2,128,64,128]`, `[2,256,32,64]`.
- The primary held-out input must be re-exported as float32 real DAIR spatial features from the Original epoch23 checkpoint and must carry scene/split/source SHA evidence.
- Prior held-out SHA `40f64b1086c0a4a9635b423d11e64b827543a15470418c96f2aba8f8c5985867` came from the pruned checkpoint. It may be retained only as a cross-configuration stress input and must not be promoted to the source-exact Original numerical gate.
- Strict FP32 means TensorRT FP16 and INT8 flags are unset and TensorRT `BuilderFlag.TF32` is explicitly cleared.
- Primary latency is batch 2, warmup 20, iterations 300, repeat 5, CUDA event, engine compute, no data transfer.
- AP is full_1789, `eval_range=102.4,51.2`, zero failed samples, zero fallback samples, and exactly three validated multiscale outputs per processed sample.
- Orin power is reported by named tegrastats rails; it is not compared as the same physical quantity as H800 NVML board power.
- The deployable comparison row is the prior Orin Pyramid `(16,32,64)` TensorRT FP16 result; INT8 remains a failed diagnostic.
- The comparison is a combined structure-plus-precision deployment comparison, not an isolated pruning or dtype causal claim.
- New artifacts are written under `$V2X_RESULTS_DIR/lane_c_orin_original_fp32_20260723/`.
- Existing engines are never copied or reused; a fresh engine must be built locally on the configured Orin target (`$LANE_C_ORIN_HOST`).

---

### Task 1: Add a strict-FP32 TensorRT build contract

**Files:**
- Modify: `tools/orin_deploy/lane_c_backbone_parity_runner.py`
- Modify: `framework/tests/test_lane_c_backbone_parity_runner.py`

**Interfaces:**
- Consumes: existing `build_engine`, `builder_flags_for_precision`, inspector parser, and output-path confinement.
- Produces: `precision=fp32`, explicit TF32 clearing, channel-signature validation, and strict-FP32 evidence in the build receipt.

- [ ] **Step 1: Write failing unit tests**

Add tests that require:

```python
self.assertEqual(
    builder_flags_for_precision("fp32"),
    ["fp32", "tf32_disabled"],
)
self.assertEqual(
    validate_output_channel_signature(
        [[2, 64, 128, 256], [2, 128, 64, 128], [2, 256, 32, 64]],
        expected=(64, 128, 256),
    ),
    [64, 128, 256],
)
```

Add a fake TensorRT config test proving `configure_builder_precision(..., "fp32")` calls `clear_flag(BuilderFlag.TF32)` and never calls `set_flag` for FP16 or INT8. Add inspector tests proving FP16, INT8, or TF32 evidence blocks strict-FP32 status.

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
PYTHONPATH=. python -m unittest framework.tests.test_lane_c_backbone_parity_runner
```

Expected: failure because `fp32`, TF32 clearing, and output-channel helpers do not yet exist.

- [ ] **Step 3: Implement the minimal strict-FP32 path**

Implement immutable helpers with these contracts:

```python
def builder_flags_for_precision(precision: str) -> list[str]:
    if precision == "fp32":
        return ["fp32", "tf32_disabled"]
    if precision == "fp16":
        return ["fp16"]
    if precision == "int8":
        return ["int8", "fp16_fallback"]
    raise ValueError(f"unsupported precision: {precision}")


def configure_builder_precision(trt, builder, config, precision: str) -> None:
    if precision == "fp32":
        config.clear_flag(trt.BuilderFlag.TF32)
        return
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        return
    if precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("TensorRT reports no fast INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        return
    raise ValueError(f"unsupported precision: {precision}")
```

Report `strict_fp32`, `tf32_allowed`, `output_shapes`, `output_channel_signature`, and a list of forbidden inspector matches. Require `--expected-output-channels 64,128,256` for this run and reject a mismatching engine.

- [ ] **Step 4: Run unit and regression tests**

Run:

```bash
PYTHONPATH=. python -m unittest \
  framework.tests.test_lane_c_backbone_parity_runner \
  framework.tests.test_lane_c_orin_runner
```

Expected: all tests pass.

### Task 2: Strengthen the full_1789 bridge output contract

**Files:**
- Modify: `scripts/stage3_trt_multiscale_ap_bridge_v3.py`
- Modify: `framework/tests/test_stage3_trt_multiscale_ap_bridge_v3.py`

**Interfaces:**
- Consumes: `TrtMultiscaleBackboneBridge`, `output_error_record`, `engine_ap_gate`, and report generation.
- Produces: explicit `(64,128,256)` output channel validation and a fail-closed AP claim.

- [ ] **Step 1: Write failing bridge tests**

Add an Original happy-path fake runner with output channels `64/128/256`. Add a wrong-channel fake runner and assert that the bridge raises before returning tensors. Extend `engine_ap_gate` tests so an output summary with fewer than `processed_samples * 3` compared records, a shape mismatch, or a non-finite output blocks `engine_ap_claim`.

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
PYTHONPATH=. python -m unittest framework.tests.test_stage3_trt_multiscale_ap_bridge_v3
```

Expected: failure because the channel contract and output-summary gate are absent.

- [ ] **Step 3: Implement the channel and claim gates**

Add `--expected-output-channels`, parse it into an immutable tuple, pass it into `TrtMultiscaleBackboneBridge`, and verify the spatially ordered output shapes before slicing. Extend the protocol with the expected channels. Extend `engine_ap_gate` to require:

```python
num_records == processed_samples * 3
num_compared == processed_samples * 3
all_finite is True
shape_mismatch_count == 0
```

Store the validated output contract in the AP report.

- [ ] **Step 4: Run bridge and existing AP regression tests**

Run:

```bash
PYTHONPATH=. python -m unittest \
  framework.tests.test_stage3_trt_multiscale_ap_bridge_v3 \
  framework.tests.test_lane_c_remote_ap_bridge
```

Expected: all tests pass.

### Task 3: Freeze source, held-out, and strict-FP32 reference evidence

**Files:**
- Create: `tools/orin_deploy/lane_c_original_fp32_finalize.py`
- Create: `framework/tests/test_lane_c_original_fp32_finalize.py`
- Create: `results/lane_c_orin_original_fp32_20260723/contracts/experiment_contract.json`

**Interfaces:**
- Consumes: exact source/checkpoint/config, a newly exported Original held-out receipt, and prior FP16 comparison artifacts.
- Produces: a fail-closed completion audit, summary tables, and final artifact manifest.

- [ ] **Step 1: Write failing finalizer tests**

Build temporary fixtures and require the finalizer to reject wrong checkpoint, config, ONNX, held-out source checkpoint, non-float32 held-out data, output channels, TF32-enabled build receipts, incomplete latency samples, partial AP, and a result claiming isolated pruning or dtype causality.

- [ ] **Step 2: Run the tests and observe RED**

Run:

```bash
PYTHONPATH=. python -m unittest framework.tests.test_lane_c_original_fp32_finalize
```

Expected: module import failure before implementation.

- [ ] **Step 3: Implement the fail-closed finalizer**

The finalizer must validate every Global Constraint, emit `completion_audit.json`, `comparison.json`, `numerical_summary.csv`, `latency_power_summary.csv`, `ap_summary.csv`, and `lane_c_orin_original_fp32_summary.md`, and refuse to report a cross-hardware or single-factor speedup.

- [ ] **Step 4: Run finalizer tests**

Run:

```bash
PYTHONPATH=. python -m unittest framework.tests.test_lane_c_original_fp32_finalize
```

Expected: all tests pass.

### Task 4: Export Original held-out data, then build and measure the fresh Orin strict-FP32 engine

**Files:**
- Create: `results/lane_c_orin_original_fp32_20260723/orin/fp32/*`
- Create: `results/lane_c_orin_original_fp32_20260723/reference/*`

**Interfaces:**
- Consumes: frozen ONNX, newly exported Original held-out input, hardened runner, Orin TensorRT 8.5.2.2.
- Produces: fresh engine/build/inspector SHAs, server reference, three-level numerical report, 1500 latency samples, and tegrastats rails.

- [ ] **Step 1: Audit Orin idle state and capabilities**

Run read-only remote checks for hostname, TensorRT/CUDA, GPU identity, disk, process occupancy, nvpmodel, jetson_clocks, temperature, and whether `BuilderFlag.TF32` plus `clear_flag` are available. Save the result in the new capability directory.

- [ ] **Step 2: Export source-exact Original held-out data**

Using checkpoint SHA `4ccc6fe1...42b2`, config SHA `f55a6ad8...f540`, and the same frozen DAIR scene split policy, run the Original model only through pillar VFE and scatter. Save float32 `[N,2,64,128,256]` batches plus a receipt containing checkpoint/config/split SHA, ordered scene IDs, record lengths, exclusion policy, dtype, shape, and output SHA. Assert the selected scenes are not used for any calibration.

- [ ] **Step 3: Transfer only frozen source, Original held-out, and scripts**

Copy the exact ONNX, held-out NPY, runner, bridge, and source receipt into a fresh remote root. Verify every received SHA before building. Do not transfer an engine.

- [ ] **Step 4: Build strict FP32 locally on Orin**

Invoke:

```bash
python lane_c_backbone_parity_runner.py build \
  --precision fp32 \
  --expected-output-channels 64,128,256 \
  --onnx source/pyramid_064x128x256_multiscale.onnx \
  --engine orin/fp32/original_strict_fp32.engine \
  --inspector-json orin/fp32/engine_inspector.json \
  --artifact-json orin/fp32/build_receipt.json \
  --artifact-root .
```

Require `strict_fp32=true`, `tf32_allowed=false`, zero FP16/INT8 inspector matches, and output channels `64/128/256`.

- [ ] **Step 5: Generate and compare real held-out outputs**

Generate the exact ONNX FP32 reference on a server runtime, execute the Orin engine on all 16 held-out batches, and report cosine, nRMSE, MAE, RMSE, finite status, range, and clipping proxies for all three levels.

- [ ] **Step 6: Run the primary latency and power protocol**

Run warmup 20, iterations 300, repeat 5 with CUDA events and tegrastats. Save all 1500 samples, per-repeat statistics, raw rail telemetry, and the measurement receipt.

### Task 5: Run full_1789 AP and seal the deployment comparison

**Files:**
- Create: `results/lane_c_orin_original_fp32_20260723/ap_full/orin/fp32/*`
- Create: final summary and SHA manifest under `results/lane_c_orin_original_fp32_20260723/`

**Interfaces:**
- Consumes: fresh strict-FP32 engine, Original checkpoint/config, hardened bridge, DAIR val split.
- Produces: full_1789 AP50/AP70 and the Original-FP32 versus compact-FP16 deployment table.

- [ ] **Step 1: Run the direct Orin AP bridge**

Use `precision-tag=fp32`, `expected-output-channels=64,128,256`, `eval-range=102.4,51.2`, and the exact Original checkpoint directory. Require `processed_samples=1789`, `failed_samples=0`, `fallback_samples=0`, and `num_compared=5367`.

- [ ] **Step 2: Run the finalizer**

Require all source, strict-FP32, numerical, latency, power, and AP gates to pass. Label the comparison `combined_structure_and_precision_deployment_comparison`.

- [ ] **Step 3: Verify and seal artifacts**

Run:

```bash
PYTHONPATH=. python -m unittest \
  framework.tests.test_lane_c_backbone_parity_runner \
  framework.tests.test_stage3_trt_multiscale_ap_bridge_v3 \
  framework.tests.test_lane_c_original_fp32_finalize
python -m py_compile \
  tools/orin_deploy/lane_c_backbone_parity_runner.py \
  tools/orin_deploy/lane_c_original_fp32_finalize.py \
  scripts/stage3_trt_multiscale_ap_bridge_v3.py
git diff --check
```

Generate `artifact_sha256_manifest.txt` last and verify every listed file with `sha256sum -c`.
