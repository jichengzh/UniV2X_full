# Stage2 Native INT8 Backbone/Subnet Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current QDQ-heavy / float32-heavy INT8 path with a verifiable native INT8 backbone/subnet route, then measure `base`, `s0_024`, and `s1_048` on H800 with latency reported in `ms`.

**Architecture:** Treat the existing ONNXRuntime static QDQ route as a baseline and diagnostic artifact only. Add a separate native INT8 route under `raw/int8_native_route/` that first probes TVM CUDA INT8/QNN capability, then builds a QNN or TensorIR-backed backbone/subnet artifact with int8/uint8 tensors and int32 accumulation through the conv-heavy path. Only after evidence gates pass should row generators import native INT8 latency/energy/AP data.

**Tech Stack:** Python, ONNX, TVM Relax/Relay/TIR, CUDA H800, existing Stage2 JSONL row tables, `unittest`, `nvidia-smi` telemetry.

---

## Current Blocker

The measured INT8 smoke path is not a clean quantized acceleration path. It currently uses ONNXRuntime static QDQ quantization with synthetic min/max calibration, imports `QuantizeLinear` / `DequantizeLinear` into TVM, and lowers to a graph where float32 tokens and dequantize functions still dominate the hot path.

Representative evidence:

- Current route script: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_route/stage2_h800_int8_route_attempt.py`
- Current route recipe: `raw/int8_route/20260627_083401/{base,s0_024,s1_048}/quant_recipe.json`
- Current lowered inventory: `raw/quant_speed_root_cause/20260627_quant_speed_profile_v1/{base,s0_024,s1_048}/int8/tvm_operator_inventory.json`
- Manual evidence: `exports/quant_speed_manual_lowered_evidence_int8_qdq_v1.json`

This explains why INT8 latency is not meaningfully faster, but the next stage should solve the implementation problem instead of only explaining it.

## Success Gates

| gate | required evidence | pass condition |
|---|---|---|
| Native route identity | `native_int8_route_manifest.json`, artifact registry row | `quant_method=h800_tvm_native_int8_backbone_subnet`, not `static_qdq_synthetic_minmax` |
| QDQ removal | ONNX/Relax op count and lowered text scan | `QuantizeLinear` / `DequantizeLinear` absent from hot path; boundary quant/dequant only if explicitly marked |
| Dtype lowering | `tvm_operator_inventory.json`, lowered IR snippets | conv-heavy path uses int8/uint8 inputs/weights with int32 accumulation; float32 limited to scales, final dequant, or unsupported boundary ops |
| Schedule fairness | matched schedule audit | native INT8, FP32, and true-FP16 comparisons state schedule policy and artifact path; `base` is not used for causal claims until this audit passes |
| Latency | H800 TVM/TVM VM repeated smoke | `latency_ms` is lower than current QDQ INT8 and FP32 for `s0_024` and `s1_048`; target initial threshold is at least 10% faster before calling it acceleration |
| Scope honesty | row fields and summaries | `full_network_claim=false`, `quant_scope=backbone_subnet_native_int8` or `backbone_subnet_partial_native_int8` |
| AP/energy | row source audit | AP is not claimed measured unless a real TVM/TVM VM eval source exists; energy follows only after the native latency route passes |

## File Structure

- Create: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_capability_probe.py`
  - Probe whether the local TVM build can compile and run tiny CUDA INT8/QNN conv workloads.
- Create: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_route_attempt.py`
  - Build native INT8 backbone/subnet artifacts for `base`, `s0_024`, and `s1_048`.
- Create: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_latency_smoke.py`
  - Measure native INT8 artifacts with H800 TVM/TVM VM and write `latency_result.json`.
- Create: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_profile.py`
  - Export lowered graph, dtype inventory, QDQ count, and VM/profile evidence for native artifacts.
- Create: `framework/tests/test_stage2_native_int8_route_evidence.py`
  - Unit tests for classifying native INT8 vs QDQ-heavy routes and rejecting overclaims.
- Modify later: `scripts/stage2_probe_tvm_int8_artifact_route.py`
  - Keep the old QDQ route, but label it explicitly as baseline/diagnostic.
- Modify later: `scripts/stage2_generate_quant_anchor_smoke.py`
  - Import native INT8 rows only when all evidence gates pass.
- Modify later: `scripts/stage2_generate_original60_quant_state_coverage.py`
  - Add native INT8 coverage fields without mixing them with old QDQ INT8 rows.

## Task 1: Evidence Schema And Native/QDQ Classifier

**Files:**
- Create: `framework/tests/test_stage2_native_int8_route_evidence.py`
- Create or modify: a small helper colocated with the native route scripts

- [ ] Add tests that reject a route when `quant_method=static_qdq_synthetic_minmax`.
- [ ] Add tests that reject a route when lowered inventory has dequantize-heavy function names such as `fused_*quantize*dequantize*` on the hot path.
- [ ] Add tests that require `native_int8_route_manifest.json` fields: `label`, `quant_method`, `quant_scope`, `full_network_claim`, `artifact_digest`, `lowered_inventory_digest`, `q_or_dq_hot_path_count`, `int8_kernel_evidence`, `float32_boundary_evidence`, `schedule_policy`.
- [ ] Implement the classifier with conservative defaults: unknown artifacts fail closed and remain diagnostic-only.
- [ ] Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m unittest framework.tests.test_stage2_native_int8_route_evidence
```

## Task 2: TVM CUDA INT8 Capability Probe

**Files:**
- Create: `raw/int8_native_route/stage2_h800_native_int8_capability_probe.py`

- [ ] Build a tiny int8/uint8 conv workload with int32 accumulation and optional requantization.
- [ ] Probe both QNN frontend availability and direct TensorIR fallback availability.
- [ ] Write `raw/int8_native_route/<run_id>/capability_probe.json` with target, TVM version, CUDA target, build status, run status, lowered dtype tokens, and failure traceback if any.
- [ ] Declare one of three outcomes: `qnn_native_int8_available`, `tensorir_int8_required`, or `blocked_by_tvm_cuda_int8_capability`.
- [ ] Run on H800:

```bash
PYTHONPATH=/home/jichengzhi/V2X /exdata/jichengzhi/tvm310/bin/python \
  multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_capability_probe.py \
  --run-id 20260627_native_int8_capability_v1 \
  --gpu 3
```

## Task 3: Native INT8 Route Prototype For `s0_024`

**Files:**
- Create: `raw/int8_native_route/stage2_h800_native_int8_route_attempt.py`

- [ ] Start with `s0_024`, because current FP32/FP16/INT8 timings are close enough for a useful controlled comparison.
- [ ] Prefer a TVM QNN route that keeps intermediate conv-heavy tensors quantized and only dequantizes at unsupported boundaries.
- [ ] If QNN cannot produce CUDA INT8 kernels, implement a narrow TensorIR prototype for the dominant conv shapes and report `quant_scope=backbone_subnet_partial_native_int8`.
- [ ] Write per-label artifacts under `raw/int8_native_route/<run_id>/s0_024/`: `native_int8_route_manifest.json`, `attempts.json`, `lowered_graph.txt`, `tvm_operator_inventory.json`, and compiled artifact if successful.
- [ ] Do not append Stage2 measured rows from this task. This task only proves the route can build.

## Task 4: Native INT8 H800 Latency Smoke

**Files:**
- Create: `raw/int8_native_route/stage2_h800_native_int8_latency_smoke.py`

- [ ] Measure native INT8 `s0_024` with the same warmup/repeat style as current FP32/FP16/INT8 smoke.
- [ ] Report latency externally only as `latency_ms`.
- [ ] Compare against current values:
  - FP32 `s0_024`: `39.522546 ms`
  - true-FP16 `s0_024`: `37.734585 ms`
  - current QDQ INT8 `s0_024`: `41.040342 ms`
- [ ] Pass condition for moving forward: native INT8 beats current QDQ INT8 and FP32, with initial target at least 10% faster than FP32 for a credible acceleration claim.
- [ ] If this fails, return to Task 3 and inspect lowered dtype tokens, schedule policy, VM overhead, and layout conversions before expanding to other labels.

## Task 5: Expand To `base` And `s1_048`

**Files:**
- Same native route and smoke scripts

- [ ] Build and measure `s1_048` after `s0_024` passes, because it is an original60 point and has comparable current smoke evidence.
- [ ] Build and measure `base`, but keep the causal claim gated until matched schedule policy is verified because current `base` FP32 is much faster than FP16/INT8 and likely uses a different schedule/route.
- [ ] Write a `matched_schedule_policy_audit.json` for all three labels.
- [ ] Keep failures as blocker rows or diagnostic artifacts; do not silently fall back to QDQ and call it native.

## Task 6: Profile And Close The Lowering Problem

**Files:**
- Create: `raw/int8_native_route/stage2_h800_native_int8_profile.py`

- [ ] Export lowered graph and dtype token counts for each native INT8 artifact.
- [ ] Record q/dq hot-path count, float32 token locations, int8/int32 function names, and unsupported boundary ops.
- [ ] Compare native route inventory against current QDQ route inventory.
- [ ] Produce `exports/native_int8_lowering_audit_latest.md` and `.json`.
- [ ] Reviewer gate: the audit must make it visually obvious that the native route is not QDQ-heavy / float32-heavy.

## Task 7: Row Integration After Evidence Gates

**Files:**
- Modify: `scripts/stage2_generate_quant_anchor_smoke.py`
- Modify: `scripts/stage2_generate_original60_quant_state_coverage.py`

- [ ] Append native INT8 latency rows only after Tasks 1-6 pass.
- [ ] Use a new precision/route label such as `native-int8` or `int8_native_backbone_subnet`; do not overwrite the old QDQ INT8 rows.
- [ ] Required row fields: `latency_ms`, `measurement_source=true_measurement_smoke`, `claim_status=claimable_true_measurement_smoke`, `quality_gate_status=native_int8_tvm_latency_smoke_only`, `full_network_claim=false`, `quant_method=h800_tvm_native_int8_backbone_subnet`.
- [ ] Add digest fields for compiled artifact, route manifest, lowered inventory, and capability probe.
- [ ] Keep AP empty/no-claim until a real TVM/TVM VM AP eval source exists.

## Task 8: Energy And AP Follow-Up

**Files:**
- Extend native INT8 energy/AP scripts only after latency route passes

- [ ] Reuse the H800 idle baseline + active telemetry energy pattern for native INT8.
- [ ] Write native INT8 energy rows separately from old QDQ INT8 energy rows.
- [ ] Run AP only through a compliant TVM/TVM VM eval path. If unavailable, preserve `AP70 no measured data` and write blocker/quarantine evidence.
- [ ] Final expected table target: three native INT8 latency rows, three native INT8 energy rows, and three native INT8 AP measured rows or explicit AP blockers.

## Agent Team Split

| agent | responsibility | output |
|---|---|---|
| Main experiment agent | Build capability probe, native route, H800 latency/energy/AP experiments, and row integration after gates pass | artifacts, latency/energy/AP rows or blockers, native lowering audit |
| Critic agent | Attack the conclusion and route classification; check QDQ fallback, float32-heavy lowering, schedule mismatch, AP overclaim, unit mix, and digest mismatch | reviewer notes, accept/reject decision, required reruns |

The critic agent must treat any QDQ fallback, unknown lowered dtype, missing digest, or AP source without TVM/TVM VM evidence as a blocking issue.

## Next `/goal` Command

```text
/goal 启动 agent team 执行 Stage2 native INT8 backbone/subnet 加速计划。使用 multi_agent_v1.spawn_agent 启动两个 sub-agent: 一个主实验 agent(agent_type=worker), 一个批判 agent(agent_type=reviewer)。

工作目录: /home/jichengzhi/V2X

主目标:
- 解决当前 INT8 的 QDQ-heavy / float32-heavy lowering 问题。
- 为 base/s0_024/s1_048 构建 native INT8 backbone/subnet route。
- 在 H800 上证明 native INT8 latency_ms 相比当前 QDQ INT8 与 FP32 有真实加速。
- 后续再补 native INT8 energy_J 与 AP70; AP 没有 real TVM/TVM VM eval source 时必须写 blocker, 不得声明 measured。

启动命令模板:

multi_agent_v1.spawn_agent(
  agent_type="worker",
  message="主实验 agent: 在 /home/jichengzhi/V2X 中执行 docs/superpowers/plans/2026-06-27-stage2-native-int8-backbone-subnet-acceleration.md。优先实现 raw/int8_native_route 的 TVM CUDA INT8 capability probe、native INT8 route attempt、native INT8 latency smoke 和 lowering audit。不要把 static QDQ route 当 native INT8; 所有 latency 对外统一写 ms; full_network_claim=false。"
)

multi_agent_v1.spawn_agent(
  agent_type="reviewer",
  message="批判 agent: 只读审阅主实验 agent 的 native INT8 route、H800 日志、row、artifact digest、lowered inventory 与结论。重点检查 QDQ fallback、float32-heavy lowering、schedule 不公平、AP overclaim、us/ms 混用、digest 缺失和 base 因果过度外推。输出 accept/reject reviewer notes 和必须补的实验。"
)
```

## Stop Conditions

- **Success:** `s0_024` and `s1_048` native INT8 pass evidence gates and beat FP32/current QDQ INT8 latency; `base` has measured native INT8 data plus a schedule fairness audit.
- **Blocked:** TVM/H800 cannot build or run any native INT8/QNN/TensorIR CUDA route; write capability-probe blocker and keep existing QDQ INT8 as diagnostic-only.
- **Not acceptable:** updating summary rows to claim native INT8 acceleration while the lowered route is still QDQ-heavy or float32-heavy.
