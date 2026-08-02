# 39_6_28_交接文档_阶段三_H800ReferenceRange实测与ScaleAwareGate通过

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `38_6_28_交接文档_阶段三_ReferenceRangeCapturePlan与FP16INT8_AP收口.md`
- `37_6_28_交接文档_阶段三_Layer0ReferenceRangeTargets与Hook采集计划.md`
- `36_6_28_交接文档_阶段三_CalibrationV2Helper与Layer0ReferenceRangeBlocker.md`

本轮核心进展: 在 H800 上完成 `s0_024` layer0 reference ranges 实测采集, 22/22 target 全部 hook-ready 且全部 captured。基于这 22 个真实中间 range 生成 calibration v2 后, 离线 scale-aware prefix simulator 到 `pyramid_level0` 不再出现 heavy saturation。此前 `missing_layer0_intermediate_reference_ranges` blocker 对 `s0_024` 已解除。

## 0. 当前权威覆盖状态

总表覆盖状态仍是:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 5/60 measured |
| native INT8 | 60/60 measured | 60/60 measured | 0/60 measured |
| FP32 | 58/60 measured | 0/60 measured | 0/60 measured |

注意:

```text
本轮没有生成 native INT8 AP measured row。
rows/native_int8_original60_ap_rows_v1.jsonl 仍不得导入总表。
```

latency 口径继续固定:

```text
latency_ms = H800 + TVM backbone/subnet compiled module end-to-end latency
input = spatial_features
output = multiscale backbone/subnet outputs
full_network_claim = false
```

它仍只能作为 `RSU-side backbone/subnet compute proxy on H800`, 不能写成 full perception network 或真实物理 RSU 设备端到端 latency。

## 1. 本轮代码变更

修改:

```text
scripts/stage2_h800_native_int8_op_alignment.py
framework/tests/test_stage2_native_int8_route.py
```

新增能力:

1. `reference_range_capture_plan(...)` 支持 reused ReLU call-index 映射。
2. residual `Add` 不再只能 blocked: 若同一 block 后继 ReLU target 存在, 使用 `module_forward_pre_hook_call_index` 捕获 pre-ReLU input, 即 ONNX Add output。
3. 新增 `build_reference_range_capture_payload(...)`。
4. 新增 `build_pytorch_module_inventory_payload(...)`。
5. 新增 CLI:

```text
--execute-reference-range-capture
--module-inventory-out
--reference-range-plan-out
--reference-ranges-out
--reference-range-calibration-out
```

兼容性:

```text
--collect-reference-ranges 不加 --execute-reference-range-capture 时仍保持轻量 target metadata 模式, 不加载 PyTorch/H800 model。
```

新增测试:

```text
test_reference_range_capture_plan_maps_reused_relu_and_add_by_call_index
test_build_reference_range_capture_payload_summarizes_ranges
test_build_pytorch_module_inventory_payload_records_module_classes
```

## 2. H800 实测 artifact

H800 实测目录已拉回本地:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/reference_range_capture_h800_abs_v2/
```

关键文件:

```text
runner_stdout.txt
runner_stderr.txt
pytorch_module_inventory_layer0_v1.json
tensor_reference_range_targets_layer0_v1.json
tensor_reference_range_capture_plan_layer0_v1.json
tensor_reference_ranges_layer0_v1.json
graph_input_quant_reference_range_capture_v1.json
tensor_quant_params_calibration_v2.json
tensor_reference_range_capture_summary.json
```

命令状态:

```text
COMMAND_RC 0
schema = native_int8_reference_range_capture_summary_v1
status = ready_for_scale_aware_simulator
range_count = 22
full_network_claim = false
ap_measured = false
```

module inventory:

```text
schema = native_int8_pytorch_module_inventory_v1
status = measured
module_count = 158
scope = model.pyramid_backbone
```

capture plan:

```text
schema = native_int8_reference_range_capture_plan_v1
status = ready_for_capture
summary = {total: 22, hook_ready: 22, blocked: 0}
```

range payload:

```text
schema = native_int8_reference_ranges_v1
status = ready_for_calibration
range_count = 22
missing_count = 0
```

op coverage:

| op_type | count |
|---|---:|
| Conv | 10 |
| Relu | 9 |
| Add | 3 |

代表性 range:

```text
first tensor:
  tensor_name = /resnet/layer0/layer0.0/conv1/Conv_output_0
  source_module_name = resnet.layer0.0.conv1
  min = -11.642651557922363
  max = 10.394745826721191
  recommended_uint8_scale = 0.09167442171592412

pyramid_level0:
  tensor_name = pyramid_level0
  source_module_name = resnet.layer0.2.relu
  source_call_index = 2
  min = 0.0
  max = 20.673961639404297
  recommended_uint8_scale = 0.1627870995228685
```

Add capture 口径:

```text
/resnet/layer0/layer0.0/Add_output_0 -> resnet.layer0.0.relu pre-hook call_index=2
/resnet/layer0/layer0.1/Add_output_0 -> resnet.layer0.1.relu pre-hook call_index=2
/resnet/layer0/layer0.2/Add_output_0 -> resnet.layer0.2.relu pre-hook call_index=2
```

这避免了直接修改 HEAL block forward, 也没有伪造 Add range。

## 3. Scale-aware gate 结果

新增本地 trace 目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_h800_calibration_v2/
```

关键文件:

```text
scale_aware_prefix_trace_records.json
scale_aware_prefix_trace_summary.json
```

summary:

```text
schema = native_int8_scale_aware_prefix_trace_summary_v2
status = passed_no_heavy_saturation_to_pyramid_level0
record_count = 22
first_heavily_saturated_output = null
pyramid_level0 max = 242
pyramid_level0 max_fraction = 0.0
pyramid_level0 zero_fraction = 0.0
full_network_claim = false
ap_measured = false
```

对比第 35-38 份文档中的旧 blocker:

```text
旧: pyramid_level0 max_fraction = 0.7581348419189453
新: pyramid_level0 max_fraction = 0.0
```

因此 `missing_layer0_intermediate_reference_ranges` 对 `s0_024` 的 scale-aware prefix gate 已解除。

## 4. 已知问题与边界

1. 本轮只证明 `s0_024` layer0 prefix scale-aware gate 通过, 还不是 full AP measured。
2. scale-aware trace 使用 cached spatial activation 运行 simulator, 不是完整 val set AP。
3. native INT8 TVM worker/AP runner 还没有接入 `tensor_quant_params_calibration_v2.json`。
4. original60 的 native INT8 AP 仍为 0/60 measured。
5. FP16 AP 仍只有 5/60 measured, 剩余 55 个 true FP16 AP cell 要继续跑。

## 5. 下一步计划

### 5.1 把 calibration v2 接入 native INT8 AP worker

优先任务:

```text
输入:
  reference_range_capture_h800_abs_v2/tensor_quant_params_calibration_v2.json

目标:
  native INT8 TVM worker 使用 calibrated per-tensor scale/zero_point,
  替代 naive input_scale * weight_scale propagation。
```

实现后先跑 smoke:

```text
s0_024 native INT8 AP smoke
```

gate:

```text
pred_nonempty_count > 0
AP30/AP50/AP70 finite
processed_samples 满足 smoke 或 full eval 约束
full_network_claim = false
```

如果 smoke 仍失败, 必须写:

```text
worker request/response
stdout/stderr
postprocess summary
numeric sanity summary
op-level blocker
```

### 5.2 推广到三点 INT8 AP smoke

s0_024 通过后, 继续:

```text
s0_040
s1_048
```

每个 label 都先采集 layer0 reference ranges, 生成 calibration v2, 再跑 AP smoke。

### 5.3 original60 收口

三点通过后再推广 original60:

```text
native INT8 AP70: 60/60 measured
FP16 AP70: 60/60 measured
```

latency/energy 当前已有:

```text
FP16 latency/energy = 60/60 measured
native INT8 latency/energy = 60/60 measured
```

但仍要保留 artifact/digest/telemetry 审核: 发现缺 raw 或口径不一致的 cell 要重跑或 quarantine。

## 6. 验证记录

本地验证:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest \
  framework.tests.test_stage2_native_int8_route \
  framework.tests.test_stage2_original60_quant_completion

Ran 58 tests in 7.471s
OK
```

语法检查:

```text
python -m py_compile \
  framework/stage2/native_int8_full_onnx.py \
  framework/stage2/original60_quant_completion.py \
  scripts/stage2_h800_native_int8_op_alignment.py \
  scripts/stage2_generate_fp16_int8_original60_completion_queue.py \
  scripts/stage2_generate_original60_quant_ap_true_eval_queue.py

exit 0
```

artifact 断言:

```text
capture status = ready_for_scale_aware_simulator
range_count = 22
plan = {blocked:0, hook_ready:22, total:22}
trace status = passed_no_heavy_saturation_to_pyramid_level0
```

H800 远端验证:

```text
scripts/stage2_h800_native_int8_op_alignment.py py_compile exit 0
reference range capture COMMAND_RC 0
```

## 7. 下一轮 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage3 original60 FP16/native INT8 三指标收口, 单 agent 执行, 不启动 agent team。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/39_6_28_交接文档_阶段三_H800ReferenceRange实测与ScaleAwareGate通过.md
- multi_agent/methods/design/auto-tuning/progress/6_27/38_6_28_交接文档_阶段三_ReferenceRangeCapturePlan与FP16INT8_AP收口.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md

当前事实:
- native INT8 backbone/subnet latency-energy route 已打通, original60 latency=60/60 measured, energy=60/60 measured。
- 当前 latency_ms 口径是 H800 + TVM backbone/subnet compiled module end-to-end, input=spatial_features, output=multiscale backbone/subnet outputs, full_network_claim=false。
- 它只能作为 RSU-side backbone/subnet compute proxy on H800, 不能写成 full perception network 或物理 RSU 设备端到端 latency。
- FP16 AP70 只有 5/60 measured, 剩余 55 个 true FP16 AP cell。
- native INT8 AP70 仍 0/60 measured, rows/native_int8_original60_ap_rows_v1.jsonl 不得生成或导入, 直到 true native INT8 AP gate 通过。
- s0_024 layer0 reference ranges 已在 H800 实测采集: range_count=22, missing_count=0。
- s0_024 capture plan: total=22, hook_ready=22, blocked=0。
- s0_024 calibration v2 已生成, scale-aware prefix simulator 到 pyramid_level0 已通过: max_fraction=0.0, first_heavily_saturated_output=null。

优先任务:
1. 把 reference_range_capture_h800_abs_v2/tensor_quant_params_calibration_v2.json 接入 native INT8 TVM worker/AP runner, 替代 naive scale propagation。
2. 先跑 s0_024 native INT8 AP smoke, 要求 pred_nonempty_count>0 且 AP30/AP50/AP70 finite; 不满足则保存 worker request/response、stdout/stderr、numeric sanity、postprocess 和 op-level blocker。
3. s0_024 通过后, 对 s0_040 和 s1_048 重复 H800 reference range capture -> calibration v2 -> native INT8 AP smoke。
4. 三点通过后推广 native INT8 original60 AP queue。
5. 并行补齐 FP16 true AP eval 剩余 55 个 cell; 不复用 suspect FP16-tagged row。
6. 保留并复核 FP16/native INT8 latency 和 energy 60/60 measured artifact, 对缺 raw/digest/telemetry 的 cell 重跑或 quarantine。

硬目标:
- FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 60/60 measured。
- native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 60/60 measured。
- 刷新三张 rows 表和 exports/original60_quant_three_metric_summary_latest.md/.json, 同时刷新 fp16_int8 completion review/gap report。

失败处理:
- 遇到任一配置失败不得直接停止整批任务。
- 必须保存 runner command、stdout/stderr、GPU id、raw artifact、failure reason。
- 必须分类失败、做最小复现或 op-level blocker、修 runner/job queue 后重试。
- 只有经过反思、审查和问题解决仍不可解时, 才对该 cell 写 quarantine/no-claim; 其他 cell 继续运行。
```
