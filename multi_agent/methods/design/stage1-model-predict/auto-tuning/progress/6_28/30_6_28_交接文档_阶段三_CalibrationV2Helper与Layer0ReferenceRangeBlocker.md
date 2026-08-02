# 36_6_28_交接文档_阶段三_CalibrationV2Helper与Layer0ReferenceRangeBlocker

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `35_6_28_交接文档_阶段三_ScaleAwarePrefixTrace诊断与中间ReferenceRange计划.md`
- `34_6_28_交接文档_阶段三_CompletionQueue刷新与ScaleAwareSimulator基线补齐.md`

本轮核心进展: 用 TDD 补齐了 reference range -> tensor quant params 的 calibration v2 helper, 并基于现有 `output_dequant_sanity5_v1` 的 PyTorch reference summaries 生成了 partial calibration artifact。结果显示当前可用 reference range 只有 `pyramid_level0/1/2`, 仍缺 layer0 内部 Conv/Add/Relu 输出范围, 因此 scale-aware prefix gate 继续 blocked。没有生成或导入任何 native INT8 AP measured row。

## 0. 当前权威覆盖状态不变

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
```

仍不得生成或导入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

latency 口径仍固定:

```text
latency_ms = H800 + TVM backbone/subnet compiled module end-to-end latency
full_network_claim = false
```

## 1. 本轮代码新增

修改:

```text
scripts/stage2_h800_native_int8_op_alignment.py
framework/tests/test_stage2_native_int8_route.py
```

新增 helper:

```text
build_tensor_quant_params_from_reference_ranges(...)
_reference_range_name_and_bounds(...)
_scale_from_reference_range(...)
```

新增测试:

```text
test_reference_ranges_build_tensor_quant_params_without_scale_collapse
```

红测结果:

```text
ImportError: cannot import name 'build_tensor_quant_params_from_reference_ranges'
```

绿测后语义:

```text
输入 observed reference tensor min/max
输出 schema=native_int8_tensor_quant_params_calibration_v2
中间 tensor 默认使用 zero_point=128
scale = max(abs(min), abs(max)) / 127, 并受 min_scale 下限保护
graph input 保留原 activation quant scale/zero_point
```

示例测试中:

```text
min=0.0, max=12.7 -> scale=0.1, zero_point=128
min=0.0, max=25.4 -> scale=0.2, zero_point=128
```

这解决的是第35轮发现的问题之一: 不再让 output_scale 只靠 `input_scale * weight_scale` 连乘后退化到 `1e-14` 或 `1e-20`。

## 2. 本轮 artifact

目录:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1/
```

新增:

```text
tensor_reference_ranges_available_outputs_v1.json
tensor_quant_params_calibration_v2_available_outputs.json
tensor_quant_params_calibration_v2_blocker.json
```

来源:

```text
output_dequant_sanity5_v1/numeric_sanity_summary.json
output_dequant_sanity5_v1/bridge_call_000/agent_000/activation_quant_summary.json
```

可用 reference tensors:

| tensor | reference_min | reference_max | scale | zero_point |
|---|---:|---:|---:|---:|
| `spatial_features` | n/a | n/a | 0.05318125182507085 | 0 |
| `pyramid_level0` | 0.0 | 21.525503158569336 | 0.16949215085487665 | 128 |
| `pyramid_level1` | 0.0 | 12.446399688720703 | 0.09800314715528112 | 128 |
| `pyramid_level2` | 0.0 | 10.00213623046875 | 0.07875697819266732 | 128 |

artifact 状态:

```text
tensor_reference_ranges_available_outputs_v1.json:
  status = partial_missing_layer0_intermediate_ranges
  range_count = 3

tensor_quant_params_calibration_v2_available_outputs.json:
  status = blocked_missing_layer0_intermediate_reference_ranges
  range_count = 3

tensor_quant_params_calibration_v2_blocker.json:
  status = blocked
  failure_reason = missing_layer0_intermediate_reference_ranges
```

缺失项:

```text
layer0.0 Conv/Relu/Add outputs
layer0.1 Conv/Relu/Add outputs
layer0.2 Conv/Relu/Add outputs
```

## 3. 当前 blocker 判断

第35轮已经证明:

```text
naive/default scale propagation 在 /resnet/layer0/layer0.1/Add_output_0 首次重饱和
pyramid_level0 max_fraction = 0.7581348419189453
```

第36轮进一步证明:

```text
现有 output_dequant_sanity5_v1 只够生成 pyramid_level0/1/2 的 output scale,
不能为 layer0 内部 22 个 prefix ops 提供 calibrated output_scale。
```

因此当前 INT8 AP gate 的最精确 blocker 是:

```text
missing_layer0_intermediate_reference_ranges
```

而不是:

```text
TVM 无法 build INT8 route
INT8 latency/energy 没跑通
缺 Python scale-aware simulator
缺 output-level reference range
```

## 4. 下一步计划

### 4.1 采集 layer0 中间 reference ranges

需要扩展:

```text
scripts/stage2_h800_native_int8_op_alignment.py
```

建议新增参数:

```text
--collect-reference-ranges
--reference-range-stop-output pyramid_level0
--reference-range-out tensor_reference_ranges_layer0_v1.json
```

实现要求:

1. 对 ONNX prefix 到 `pyramid_level0` 的每个 Conv/Add/Relu output 匹配 PyTorch module 或 activation point。
2. 注册 hooks 或在 PyTorch forward 中捕获中间 tensor。
3. 产出:

```text
tensor_reference_ranges_layer0_v1.json
tensor_quant_params_calibration_v2.json
```

4. 每个 tensor 至少包含:

```text
tensor_name
op_name
op_type
shape
dtype
min
max
mean
std
recommended_uint8_scale
recommended_zero_point
sample_count
source_module_name or source_hook
```

### 4.2 rerun scale-aware simulator

输入:

```text
tensor_quant_params_calibration_v2.json
cached activation_uint8.npy
onnx_op_records.json
runtime_weights_int8.npz
```

准入:

```text
record_count >= 22
last_output_name = pyramid_level0
no tensor before pyramid_level0 has max_fraction >= 0.5
pyramid_level0 max_fraction < 0.5
```

通过后再进入:

```text
TVM route builder scale-aware requant
H800 rebuild
output sanity5
20-sample AP smoke
s0_024 full AP
original60 expansion
```

### 4.3 FP16 AP 并行主线

继续:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl: 5 -> 60
```

从:

```text
jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl
```

筛出 55 个 `precision=fp16` blocked job, 做 checkpoint recovery, 找到后运行:

```text
scripts/stage2_h800_true_fp16_ap_eval.py
```

## 5. 验证

新增 targeted test:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route.Stage2NativeInt8RouteTest.test_reference_ranges_build_tensor_quant_params_without_scale_collapse
Ran 1 test
OK
```

相关完整测试:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 53 tests in 7.489s
OK
```

语法检查:

```text
python -m py_compile framework/stage2/native_int8_full_onnx.py framework/stage2/original60_quant_completion.py scripts/stage2_h800_native_int8_op_alignment.py scripts/stage2_generate_fp16_int8_original60_completion_queue.py scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
exit code = 0
```

artifact 校验:

```text
calibration_v2_artifact_json = ok
status = blocked_missing_layer0_intermediate_reference_ranges
range_count = 3
pyramid_level0_scale = 0.16949215085487665
blocker_reason = missing_layer0_intermediate_reference_ranges
```

## 6. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。当前 authoritative 状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured; latency 统一使用 latency_ms, 口径是 H800 TVM backbone/subnet compiled module end-to-end, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明完整感知 pipeline 或真实 RSU 物理边缘设备 latency。第35轮已在 s0_024 centered-Conv route cached activation 上运行 scale-aware prefix trace 到 pyramid_level0, status=blocked, record_count=22, first_heavily_saturated_output=/resnet/layer0/layer0.1/Add_output_0, pyramid_level0 max_fraction=0.7581348419189453。第36轮已用 TDD 补齐 build_tensor_quant_params_from_reference_ranges(...) helper, 生成 tensor_quant_params_calibration_v2_available_outputs.json, 但该 artifact 仅含 spatial_features 与 pyramid_level0/1/2 的 reference-derived scale: pyramid_level0=0.16949215085487665, pyramid_level1=0.09800314715528112, pyramid_level2=0.07875697819266732; 当前 blocker 为 missing_layer0_intermediate_reference_ranges, artifact 位于 raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1/tensor_quant_params_calibration_v2_blocker.json。下一步先扩展 scripts/stage2_h800_native_int8_op_alignment.py, 增加 --collect-reference-ranges / --reference-range-stop-output pyramid_level0, 采集 layer0.0/0.1/0.2 每个 Conv/Add/Relu 输出 range, 生成 tensor_reference_ranges_layer0_v1.json 和 tensor_quant_params_calibration_v2.json; 然后 rerun simulate_scale_aware_int8_graph_prefix, 要求到 pyramid_level0 前无 max_fraction>=0.5, 通过后再下沉 TVM route builder、H800 rebuild、output sanity5、20-sample AP smoke、s0_024 full AP、original60 60 labels 扩展。FP16 主线继续从 AP queue 中 55 个 fp16 blocked job 做 checkpoint recovery, 找到 checkpoint 后运行 scripts/stage2_h800_true_fp16_ap_eval.py 并追加 rows/fp16_true_original60_ap_rows_v1.jsonl。遇到任何 build/eval/import/SSH/checkpoint/adapter/TVM/telemetry 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 做失败分类、最小复现、反思审查、runner/adapter/route/queue 修复并补跑失败 label, 其他 label 继续推进; 只有证明当前环境确实不可解, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker 或 quarantine/no-claim row。
```
