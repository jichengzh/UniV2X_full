# 40_6_28_交接文档_阶段三_CalibratedINT8APSmoke非空与RowGate修复

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `39_6_28_交接文档_阶段三_H800ReferenceRange实测与ScaleAwareGate通过.md`
- `38_6_28_交接文档_阶段三_ReferenceRangeCapturePlan与FP16INT8_AP收口.md`

本轮核心进展: 将 calibration v2 接入 native INT8 AP bridge 的 TVM output dequant 路径, 并在 H800 上完成 `s0_024` calibrated AP smoke。第一次只覆盖 `pyramid_level0` 时仍空预测；随后采集到 `pyramid_level2` 的 115 个 reference ranges, 生成覆盖 `pyramid_level0/1/2` 的 calibration v2 后, AP smoke 产生非空预测。与此同时修复 row gate: 1-sample smoke 不能把 `ap_row_allowed` 置 true。

## 0. 当前权威覆盖状态

总表覆盖状态仍未变化:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 5/60 measured |
| native INT8 | 60/60 measured | 60/60 measured | 0/60 measured |
| FP32 | 58/60 measured | 0/60 measured | 0/60 measured |

仍不得生成或导入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

本轮的 INT8 AP 是 smoke gate 进展, 不是 full AP measured row。

latency 口径继续固定:

```text
latency_ms = H800 + TVM backbone/subnet compiled module end-to-end latency
input = spatial_features
output = multiscale backbone/subnet outputs
full_network_claim = false
```

## 1. 本轮代码变更

修改:

```text
scripts/stage2_h800_native_int8_real_activation_bridge.py
framework/tests/test_stage2_native_int8_route.py
```

新增 helper:

```text
dequantize_tvm_output_uint8(...)
attach_tensor_quant_params_to_worker_request(...)
build_full_ap_report_gate_fields(...)
```

新增 CLI:

```text
--tensor-quant-params-path
--ap-row-min-samples
```

行为变化:

1. TVM worker request 会记录:

```text
tensor_quant_params_path
tensor_quant_params_digest
output_dequant_policy = per_output_tensor_quant_params_v2
```

2. bridge 对 TVM uint8 graph outputs 反量化时, 优先按 output tensor name 查 calibration v2:

```text
pyramid_level0 -> tensor_quant_params_v2
pyramid_level1 -> tensor_quant_params_v2
pyramid_level2 -> tensor_quant_params_v2
```

缺失时才 fallback 到旧 activation quant, 并写入 `output_dequant_summary.json`。

3. row gate 修复:

```text
smoke_gate_passed 可以为 true
但 ap_row_allowed 必须同时满足 processed_samples >= ap_row_min_samples
默认 ap_row_min_samples = 1789
```

因此 1-sample smoke 不会再被标成可导入 AP row。

## 2. Full multiscale reference range capture

H800 实测目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/reference_range_capture_h800_pyramid_level2_v1/
```

关键文件:

```text
runner_stdout.txt
runner_stderr.txt
tensor_reference_range_targets_to_pyramid_level2_v1.json
tensor_reference_range_capture_plan_to_pyramid_level2_v1.json
tensor_reference_ranges_to_pyramid_level2_v1.json
tensor_quant_params_calibration_v2_to_pyramid_level2.json
tensor_reference_range_capture_summary.json
```

capture summary:

```text
schema = native_int8_reference_range_capture_summary_v1
status = ready_for_scale_aware_simulator
range_count = 115
COMMAND_RC = 0
full_network_claim = false
ap_measured = false
```

capture plan:

```text
summary = {total: 115, hook_ready: 115, blocked: 0}
```

range payload:

```text
status = ready_for_calibration
range_count = 115
missing_count = 0
```

op coverage:

| op_type | count |
|---|---:|
| Conv | 51 |
| Relu | 48 |
| Add | 16 |

calibration v2 已覆盖:

```text
spatial_features scale = 0.05309228710099763, zero_point = 0
pyramid_level0 scale = 0.1627870995228685, zero_point = 128
pyramid_level1 scale = 0.07742052754079264, zero_point = 128
pyramid_level2 scale = 0.07146926939956785, zero_point = 128
```

## 3. AP smoke 结果

### 3.1 只覆盖 pyramid_level0 的 smoke

目录:

```text
ap_smoke_calibrated_output_dequant_v1/
```

结果:

```text
processed_samples = 1
pred_nonempty_count = 0
pred_total_count = 0
gate = blocked: empty_predictions_all_samples
```

审计结论:

```text
pyramid_level0 使用 tensor_quant_params_v2
pyramid_level1/2 fallback 到 activation_quant
```

因此只校准 `pyramid_level0` 不足以让 AP bridge 产生有效预测。

### 3.2 覆盖 pyramid_level0/1/2 的 smoke

目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_smoke_calibrated_pyramid_level2_v2_rowgate/
```

关键文件:

```text
runner_stdout.txt
runner_stderr.txt
real_activation_bridge_report.json
full_ap_eval_report.json
native_int8_s0_024_full_ap_blocker.json
output_dequant_summary.json
worker_response_summary.json
postprocess_summary.json
head_output_summary.json
bridge_call_000/agent_000/native_int8_worker_request.json
```

结果:

```text
COMMAND_RC = 0
processed_samples = 1
failed_samples = 0
pred_nonempty_count = 1
pred_total_count = 77
ap30 = 0.0
ap50 = 0.0
ap70 = 0.0
smoke_gate_passed = true
ap_row_allowed = false
ap_row_block_reason = full_eval_num_samples_1_lt_1789
```

三路输出反量化:

```text
pyramid_level0 scheme = tensor_quant_params_v2
pyramid_level1 scheme = tensor_quant_params_v2
pyramid_level2 scheme = tensor_quant_params_v2
```

worker request 审计:

```text
tensor_quant_params_path = .../reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json
tensor_quant_params_digest = 9173c87514722a2f2be528f631de16871321f683d0d944340fc6889d85b75b26
output_dequant_policy = per_output_tensor_quant_params_v2
```

## 4. 当前结论

已经解决:

```text
1. s0_024 layer0/pyramid_level0 saturation blocker。
2. pyramid_level1/2 output dequant 缺 calibration 导致空预测的问题。
3. 1-sample smoke 被误标成 AP row allowed 的 row gate 风险。
```

仍未完成:

```text
1. native INT8 full AP measured row 仍为 0/60。
2. s0_024 还没有跑 full 1789-sample INT8 AP eval。
3. s0_040/s1_048 尚未做 full multiscale reference range capture + AP smoke。
4. FP16 AP 仍有 55 个 cell 未补齐。
```

重要边界:

```text
本轮接入的是 AP bridge 输出反量化 calibration。
TVM .so 内部的量化 scale 是否完全按 calibration v2 重建, 仍需后续 artifact rebuild 路径审查。
```

但从 smoke 角度, native INT8 AP bridge 已经从 `empty_predictions_all_samples` 推进到 `pred_nonempty_count > 0`。

## 5. 下一步计划

### 5.1 s0_024 full AP eval

使用:

```text
--tensor-quant-params-path reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json
--num-samples 1789
--full-ap-min-samples 1789
--ap-row-min-samples 1789
```

目标:

```text
processed_samples = 1789
pred_nonempty_count > 0
AP30/AP50/AP70 finite
ap_row_allowed = true
```

若 AP 仍极低或为 0, 仍可以是 measured 结果, 但必须保留 full eval raw artifact; 后续再判断质量与数值原因。

### 5.2 三点 smoke 推广

对:

```text
s0_040
s1_048
```

重复:

```text
reference range capture to pyramid_level2
calibration v2
calibrated AP smoke
```

### 5.3 original60 收口

三点通过后, 推广到 original60:

```text
native INT8 AP70 60/60 measured
FP16 AP70 60/60 measured
```

继续保持:

```text
latency_ms 统一 ms
energy 统一 J / inference
full_network_claim = false
AP 禁止 predicted/interpolated/TRT-reference 冒充 H800 TVM measured
```

## 6. 验证记录

本地测试:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest \
  framework.tests.test_stage2_native_int8_route \
  framework.tests.test_stage2_original60_quant_completion

Ran 60 tests in 7.919s
OK
```

语法检查:

```text
python -m py_compile \
  framework/stage2/native_int8_full_onnx.py \
  framework/stage2/original60_quant_completion.py \
  scripts/stage2_h800_native_int8_op_alignment.py \
  scripts/stage2_h800_native_int8_real_activation_bridge.py \
  scripts/stage2_generate_fp16_int8_original60_completion_queue.py \
  scripts/stage2_generate_original60_quant_ap_true_eval_queue.py

exit 0
```

artifact 断言:

```text
range_count = 115
pred_nonempty_count > 0
pred_total_count > 0
smoke_gate_passed = true
ap_row_allowed = false
all graph outputs use tensor_quant_params_v2
```

H800 远端:

```text
reference range capture to pyramid_level2 COMMAND_RC 0
calibrated AP smoke COMMAND_RC 0
```

## 7. 下一轮 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage3 original60 FP16/native INT8 三指标收口, 单 agent 执行, 不启动 agent team。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/40_6_28_交接文档_阶段三_CalibratedINT8APSmoke非空与RowGate修复.md
- multi_agent/methods/design/auto-tuning/progress/6_27/39_6_28_交接文档_阶段三_H800ReferenceRange实测与ScaleAwareGate通过.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md

当前事实:
- native INT8 backbone/subnet latency-energy route 已打通, original60 latency=60/60 measured, energy=60/60 measured。
- 当前 latency_ms 口径是 H800 + TVM backbone/subnet compiled module end-to-end, input=spatial_features, output=multiscale backbone/subnet outputs, full_network_claim=false。
- FP16 AP70 只有 5/60 measured, 剩余 55 个 true FP16 AP cell。
- native INT8 AP70 仍 0/60 measured, rows/native_int8_original60_ap_rows_v1.jsonl 不得生成或导入, 直到 true native INT8 full eval gate 通过。
- s0_024 full multiscale reference ranges 已在 H800 实测采集到 pyramid_level2: range_count=115, missing_count=0。
- s0_024 calibrated AP smoke 已非空: pred_nonempty_count=1, pred_total_count=77, smoke_gate_passed=true。
- 1-sample smoke row gate 已修复: ap_row_allowed=false, reason=full_eval_num_samples_1_lt_1789。

优先任务:
1. 在 H800 上运行 s0_024 native INT8 full AP eval: num_samples=1789, full_ap_min_samples=1789, ap_row_min_samples=1789, tensor_quant_params_path 指向 reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json。
2. 若 s0_024 full eval 通过, 生成合规 native INT8 AP row; 若失败, 保留 stdout/stderr、worker request/response、postprocess、numeric summary 和 blocker。
3. 对 s0_040 和 s1_048 重复 full multiscale reference range capture -> calibration v2 -> calibrated AP smoke。
4. 三点通过后推广 native INT8 original60 AP queue。
5. 并行补齐 FP16 true AP eval 剩余 55 个 cell; 不复用 suspect FP16-tagged row。

硬目标:
- FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 60/60 measured。
- native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 60/60 measured。
- 刷新 rows/latency_original60_quant_rows_v1.jsonl、rows/energy_original60_quant_rows_v1.jsonl、rows/ap_original60_quant_rows_v1.jsonl。
- 刷新 exports/original60_quant_three_metric_summary_latest.md/.json, fp16_int8 completion review/gap report。

失败处理:
- 遇到任一配置失败不得直接停止整批任务。
- 必须保存 runner command、stdout/stderr、GPU id、raw artifact、failure reason。
- 必须分类失败、做最小复现或 op-level blocker、修 runner/job queue 后重试。
- 只有经过反思、审查和问题解决仍不可解时, 才对该 cell 写 quarantine/no-claim; 其他 cell 继续运行。
```
