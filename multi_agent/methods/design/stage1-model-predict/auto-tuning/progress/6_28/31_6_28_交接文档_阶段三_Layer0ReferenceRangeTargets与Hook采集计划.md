# 37_6_28_交接文档_阶段三_Layer0ReferenceRangeTargets与Hook采集计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `36_6_28_交接文档_阶段三_CalibrationV2Helper与Layer0ReferenceRangeBlocker.md`
- `35_6_28_交接文档_阶段三_ScaleAwarePrefixTrace诊断与中间ReferenceRange计划.md`

本轮核心进展: 将“采集 layer0 中间 reference ranges”从文档计划推进成可执行入口。新增了 `prefix_reference_range_targets(...)` 和 `--collect-reference-ranges` CLI 轻量模式, 已基于 `s0_024` centered-Conv route 的 `onnx_op_records.json` 生成到 `pyramid_level0` 为止的 22 个 reference-range target。当前状态仍是 blocked, 因为本轮只生成 target 清单, 尚未在 H800/PyTorch forward 中注册 hooks 采集真实中间 tensor ranges。

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
prefix_reference_range_targets(...)
build_reference_range_targets_payload(...)
```

新增 CLI:

```text
--collect-reference-ranges
--reference-range-stop-output
--reference-range-out
```

新增测试:

```text
test_prefix_reference_range_targets_stop_at_requested_output
```

红测结果:

```text
ImportError: cannot import name 'prefix_reference_range_targets'
```

绿测后语义:

1. 从 ONNX `op_records` 顺序扫描。
2. 选择 `Conv / Relu / Add` 输出作为 reference-range targets。
3. 遇到 `stop_output_names` 中的输出立即停止。
4. 每条 target 写入:

```text
op_index
op_type
op_name
output_name
input_name / input_names
requires_reference_range = true
stop_matched
```

## 2. 本轮 artifact

命令:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_h800_native_int8_op_alignment.py \
  --label s0_024 \
  --ckpt-dir /unused \
  --raw-dir multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1 \
  --route-dir multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024 \
  --collect-reference-ranges \
  --reference-range-stop-output pyramid_level0 \
  --reference-range-out tensor_reference_range_targets_layer0_v1.json
```

输出:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1/tensor_reference_range_targets_layer0_v1.json
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1/tensor_reference_range_collection_blocker.json
```

artifact summary:

```text
schema = native_int8_reference_range_targets_v1
status = blocked
failure_reason = reference_range_targets_recorded_without_pytorch_hooks
target_count = 22
stop_output_names = ["pyramid_level0"]
first target = /resnet/layer0/layer0.0/conv1/Conv_output_0
last target = pyramid_level0
last target stop_matched = true
full_network_claim = false
ap_measured = false
```

op coverage:

| op_type | count |
|---|---:|
| Conv | 10 |
| Relu | 9 |
| Add | 3 |

22 个 targets 覆盖:

```text
layer0.0 conv1/relu/conv2/relu_1/conv3/downsample/Add/relu_2
layer0.1 conv1/relu/conv2/relu_1/conv3/Add/relu_2
layer0.2 conv1/relu/conv2/relu_1/conv3/Add/relu_2
pyramid_level0
```

## 3. 当前 blocker 判断

当前 INT8 AP gate blocker 仍是:

```text
missing_layer0_intermediate_reference_ranges
```

本轮把 blocker 细化为:

```text
reference_range_targets_recorded_without_pytorch_hooks
```

含义:

1. 已明确需要采集哪 22 个 ONNX prefix outputs。
2. 还没有把这些 ONNX outputs 映射到 PyTorch hooks / forward capture。
3. Add output 不是简单 Module hook, 需要额外处理 residual add pre-ReLU 或建立近似/替代策略。
4. 不能用现有 `pyramid_level0/1/2` output ranges 直接替代 layer0 中间 ranges。

## 4. 下一步计划

### 4.1 Hook 映射

在 `scripts/stage2_h800_native_int8_op_alignment.py` 中继续扩展:

```text
--collect-reference-ranges
```

让它不只写 targets, 而是在 H800 PyTorch forward 中实际采集:

```text
tensor_reference_ranges_layer0_v1.json
```

建议分三类处理:

| target type | capture strategy |
|---|---|
| Conv output | register_forward_hook on resolved Conv module |
| Relu output | register_forward_hook on resolved ReLU module when module exists |
| Add output | capture block pre/post residual add; 如果 pre-ReLU add 无 hook, 先写 per-target blocker, 不伪造 range |

输出字段:

```text
tensor_name
op_index
op_type
op_name
shape
dtype
min
max
mean
std
recommended_uint8_scale
recommended_zero_point
sample_count
source_module_name or source_capture
capture_status
```

### 4.2 Calibration v2

采集真实 ranges 后生成:

```text
tensor_quant_params_calibration_v2.json
```

并 rerun:

```text
simulate_scale_aware_int8_graph_prefix(..., tensor_quant_params=tensor_quant_params_calibration_v2)
```

准入:

```text
record_count >= 22
last_output_name = pyramid_level0
no tensor before pyramid_level0 has max_fraction >= 0.5
pyramid_level0 max_fraction < 0.5
```

### 4.3 之后再下沉 TVM

只有 Python simulator gate 通过后, 才改:

```text
raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
```

进入:

```text
H800 rebuild -> output sanity5 -> 20-sample AP smoke -> s0_024 full AP -> original60 INT8 AP 60 labels
```

## 5. 验证

新增 targeted test:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route.Stage2NativeInt8RouteTest.test_prefix_reference_range_targets_stop_at_requested_output
Ran 1 test
OK
```

相关完整测试:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 54 tests in 7.279s
OK
```

语法检查:

```text
python -m py_compile framework/stage2/native_int8_full_onnx.py framework/stage2/original60_quant_completion.py scripts/stage2_h800_native_int8_op_alignment.py scripts/stage2_generate_fp16_int8_original60_completion_queue.py scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
exit code = 0
```

artifact 校验:

```text
tensor_reference_range_targets_layer0_v1.json blocked 22 reference_range_targets_recorded_without_pytorch_hooks
tensor_reference_range_collection_blocker.json blocked 22 reference_range_targets_recorded_without_pytorch_hooks
```

## 6. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。当前 authoritative 状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured; latency 统一使用 latency_ms, 口径是 H800 TVM backbone/subnet compiled module end-to-end, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明完整感知 pipeline 或真实 RSU 物理边缘设备 latency。第36轮已用 TDD 补齐 build_tensor_quant_params_from_reference_ranges(...) helper, 但 calibration_v2_available_outputs 仅含 pyramid_level0/1/2, blocker=missing_layer0_intermediate_reference_ranges。第37轮已新增 prefix_reference_range_targets(...) 与 --collect-reference-ranges / --reference-range-stop-output / --reference-range-out CLI, 并基于 s0_024 centered-Conv route 生成 tensor_reference_range_targets_layer0_v1.json: target_count=22, op coverage Conv=10, Relu=9, Add=3, first target=/resnet/layer0/layer0.0/conv1/Conv_output_0, last target=pyramid_level0, status=blocked, failure_reason=reference_range_targets_recorded_without_pytorch_hooks。下一步继续扩展 scripts/stage2_h800_native_int8_op_alignment.py, 将这 22 个 ONNX prefix targets 映射到 H800 PyTorch hooks/forward capture, 生成 tensor_reference_ranges_layer0_v1.json; Conv/Relu 尽量用 module hooks, Add 若无法捕获 pre-ReLU residual add 必须写 per-target blocker, 不能伪造 range。采集后生成 tensor_quant_params_calibration_v2.json, rerun simulate_scale_aware_int8_graph_prefix, 要求到 pyramid_level0 前无 max_fraction>=0.5; 通过后再下沉 TVM route builder、H800 rebuild、output sanity5、20-sample AP smoke、s0_024 full AP、original60 60 labels 扩展。FP16 主线继续从 AP queue 中 55 个 fp16 blocked job 做 checkpoint recovery, 找到 checkpoint 后运行 scripts/stage2_h800_true_fp16_ap_eval.py 并追加 rows/fp16_true_original60_ap_rows_v1.jsonl。遇到任何 build/eval/import/SSH/checkpoint/adapter/TVM/telemetry 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 做失败分类、最小复现、反思审查、runner/adapter/route/queue 修复并补跑失败 label, 其他 label 继续推进; 只有证明当前环境确实不可解, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker 或 quarantine/no-claim row。
```
