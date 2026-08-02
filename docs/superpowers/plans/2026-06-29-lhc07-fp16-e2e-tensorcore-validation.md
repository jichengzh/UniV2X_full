# lhc_07 FP16 End-to-End Tensor-Core Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove, on one concrete original60 config (`lhc_07`), whether real FP16 tensor-core lowering changes algorithm-visible backbone/subnet inference latency.

**Architecture:** Use a two-gate design. Gate A is a directly measurable end-to-end counterfactual: full default FP16 engine latency minus default implementation of the target 1x1 convblock plus tensor-core implementation of the same convblock. Gate B attempts a true ONNX graph rewrite so the full engine itself contains matmul-form 1x1 convs and can be timed as one TVM VM.

**Tech Stack:** Python, ONNX, TVM Relax, TVM dlight `MatmulTensorization`, H800 CUDA target, JSON/Markdown evidence artifacts.

---

### Task 1: Add lhc_07 End-to-End Counterfactual Gate

**Files:**
- Modify: `/home/jichengzhi/V2X/scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`
- Output: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_e2e_tensorcore_validation_latest.{json,md}`

- [ ] **Step 1: Extend script with `--mode e2e-counterfactual`**

Add a mode that measures:

```text
full_default_fp16_latency_us
target_conv_default_latency_us
target_conv_tensorcore_latency_us
counterfactual_latency_us = full_default - target_conv_default + target_conv_tensorcore
counterfactual_speedup = full_default / counterfactual_latency
```

- [ ] **Step 2: Measure default target convblock without tensor-core**

Use the same `M=16384,K=48,N=256` matmul+bias+relu shape, but build through default `relax.build` without dlight tensorization. Record `wmma=0,tvm_mma_sync=0` if available and latency.

- [ ] **Step 3: Reuse tensor-core target convblock measurement**

Use the existing legalize/fuse + `MatmulTensorization` route. Gate requires `wmma>0`, `tvm_mma_sync>0`, successful VM run, and max error below `1e-3`.

- [ ] **Step 4: Run full default FP16 engine**

Use `/exdata/jichengzhi/s2_tvm/models/lhc_07_backbone.onnx`, input `spatial_features=[2,64,128,256]`, and `relax.build` default route. Record latency and export path.

- [ ] **Step 5: Write JSON/MD evidence**

The MD must state clearly that Gate A is a measured counterfactual, not yet a single rewritten full-engine binary. It is still algorithm-level because it uses full engine latency and the exact target convblock from the same `lhc_07` graph.

### Task 2: Attempt True Full-Engine ONNX Rewrite

**Files:**
- Create or modify: `/home/jichengzhi/V2X/scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`
- Output: H800 raw rewritten ONNX under `/exdata/jichengzhi/s2_tvm/fp16_tensorcore_convblock_engine_20260629/`

- [ ] **Step 1: Rewrite selected 1x1 Conv nodes**

For each eligible Conv with `group=1`, `kernel_shape=[1,1]`, `pads=[0,0,0,0]`, and `strides=[1,1]`, replace:

```text
NCHW input -> Transpose NCHW->NHWC -> Reshape [N*H*W,C] -> MatMul W^T -> Add bias if present -> Reshape [N,H,W,O] -> Transpose NHWC->NCHW
```

- [ ] **Step 2: Import rewritten ONNX into TVM**

Run default `relax.build` first. Then try selective dlight scheduling. Do not claim tensor-core unless scheduled TIR or generated code contains `wmma/tvm_mma_sync`.

- [ ] **Step 3: Validate output parity**

Run original full default engine and rewritten full engine on the same random input. Compare every output tensor shape and max/mean absolute error.

- [ ] **Step 4: Write success or blocker**

If rewritten engine builds and runs, record true full-engine latency. If it fails, save the rewritten ONNX, traceback, and exact blocker so the next worker can continue.

### Task 3: Final Interpretation Document

**Files:**
- Modify or create: `/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_e2e_tensorcore_validation_latest.md`

- [ ] **Step 1: Separate proof levels**

Use these labels:

```text
measured_full_default_engine
measured_convblock_tensorcore_counterfactual
true_rewritten_full_engine_attempt
```

- [ ] **Step 2: State conclusion**

If counterfactual speedup is positive, state that FP16 tensor-core lowering has a measurable latency impact for `lhc_07`, but the production full-engine speedup still requires full graph rewrite success.

- [ ] **Step 3: State relation to FP32/INT8**

Explain that the old FP16-vs-FP32 similarity is caused by missing tensor-core lowering in the full engine, while INT8 slowness is a related backend-route problem with different numeric/scale blockers.
