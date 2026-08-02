# 38_6_28_交接文档_阶段三_ReferenceRangeCapturePlan与FP16INT8_AP收口

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `37_6_28_交接文档_阶段三_Layer0ReferenceRangeTargets与Hook采集计划.md`
- `36_6_28_交接文档_阶段三_CalibrationV2Helper与Layer0ReferenceRangeBlocker.md`
- `33_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_EnergyAP大范围补点计划.md`

本轮核心进展: 把 layer0 的 22 个 reference-range target 进一步转成 capture plan。新增 `reference_range_capture_plan(...)`, 并修复 ONNX `Relu` op name 到 PyTorch module suffix 的规范化。当前仍未产出真实 layer0 reference ranges, 因为本地没有 H800 上完整 `model.pyramid_backbone.named_modules()` inventory, 且 residual `Add` 需要 block-level capture, 不能用普通 module hook 伪造。

## 0. 两个口径确认

### 0.1 INT8 backbone/subnet route 是否已经打通

可以确认的是:

```text
H800 + TVM native INT8 full ONNX backbone/subnet route 已经打通 latency/energy 批量测量。
native INT8 original60 latency = 60/60 measured
native INT8 original60 energy = 60/60 measured
```

不能过度声明的是:

```text
native INT8 AP70 = 0/60 measured
INT8 AP gate 仍 blocked at missing_layer0_intermediate_reference_ranges
rows/native_int8_original60_ap_rows_v1.jsonl 仍不得生成或导入
```

所以当前表述应为: `native INT8 backbone/subnet latency-energy route 已打通`, 但 `native INT8 accuracy/AP route 尚未闭环`。

### 0.2 latency 是否是 backbone 端到端推理速度

当前所有 quant latency row 的统一口径是:

```text
latency_ms = H800 + TVM backbone/subnet compiled module end-to-end latency
input = spatial_features
output = multiscale backbone/subnet outputs
full_network_claim = false
```

这不是完整感知网络端到端 latency, 不包含 dataset loading、voxelization、postprocess、NMS、通信、调度等链路。它可以作为 RSU-side backbone/subnet 计算段的 H800 proxy, 但不能直接写成物理 RSU 边缘设备的真实端到端推理速度。若对外表述为 RSU 侧速度, 必须带限定词: `RSU-side backbone/subnet compute proxy on H800`。

## 1. 当前权威覆盖状态

来自:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
```

覆盖状态:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 5/60 measured |
| native INT8 | 60/60 measured | 60/60 measured | 0/60 measured |
| FP32 | 58/60 measured | 0/60 measured | 0/60 measured |

completion queue:

```text
total_jobs = 120
precision_counts = fp16:60, int8:60
latency measured = 120/120
energy measured = 120/120
AP measured = 5/120
jobs_requiring_action = 115
```

下一阶段主缺口不是 latency/energy, 而是:

```text
FP16 AP70 true eval: 55 remaining
native INT8 AP70 true eval: 60 remaining
```

如果发现 energy artifact 缺 digest、路径、raw telemetry 或口径不一致, 该 cell 需要重跑或写 quarantine；否则 energy 当前按 60/60 measured 继承。

## 2. 本轮代码与测试

修改:

```text
scripts/stage2_h800_native_int8_op_alignment.py
framework/tests/test_stage2_native_int8_route.py
```

新增 helper:

```text
reference_range_capture_plan(...)
```

同步修复:

```text
module_suffix_candidates(...) 现在会去掉 terminal /Conv 和 /Relu
```

新增测试:

```text
test_reference_range_capture_plan_maps_conv_relu_and_blocks_add
```

RED:

```text
ImportError: cannot import name 'reference_range_capture_plan'
```

GREEN 后语义:

| op_type | capture plan 行为 |
|---|---|
| Conv | 若能解析到 PyTorch module, 标记 `hook_ready` |
| Relu | 若能解析到 PyTorch module, 标记 `hook_ready`; 后续真实采集仍需处理 reused ReLU call index |
| Add | 标记 `blocked`, `failure_reason=add_output_requires_block_level_capture` |
| other | 标记 `blocked`, `failure_reason=unsupported_reference_range_target_op_type` |

## 3. 本轮新增 artifact

目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1/
```

新增:

```text
tensor_reference_range_capture_plan_layer0_v1.json
tensor_reference_range_capture_plan_blocker.json
```

capture plan summary:

```text
schema = native_int8_reference_range_capture_plan_v1
status = blocked
failure_reason = partial_module_inventory_only_add_relu_need_block_level_capture
target_count = 22
hook_ready = 2
blocked = 20
module_inventory_source = op_level_selected_ops_only_not_full_pytorch_named_modules
full_network_claim = false
ap_measured = false
```

已具备现有实测 module 映射证据的 target:

| op_index | op_type | op_name | source_module_name |
|---:|---|---|---|
| 0 | Conv | `/resnet/layer0/layer0.0/conv1/Conv` | `resnet.layer0.0.conv1` |
| 5 | Conv | `/resnet/layer0/layer0.0/downsample/downsample.0/Conv` | `resnet.layer0.0.downsample.0` |

仍 blocked 的原因:

1. 当前只复用了先前 op-level alignment 选中 Conv 的 module 名, 不是完整 PyTorch module inventory。
2. layer0 其余 Conv 需要 H800 上导出完整 `model.pyramid_backbone.named_modules()` 后再解析。
3. ReLU 在 HEAL ResBlock/Bottleneck 中是同一个 `self.relu` 多次复用, 真实采集要按 call index 对齐 `/relu`, `/relu_1`, `/relu_2`。
4. ONNX `Add` 对应 residual add, 不是普通 named module, 必须在 block forward 或 wrapper 中捕获 pre-ReLU add output；不能伪造 range。

## 4. 当前 INT8 AP blocker

最精确 blocker 仍是:

```text
missing_layer0_intermediate_reference_ranges
```

本轮进一步细化为:

```text
partial_module_inventory_only_add_relu_need_block_level_capture
```

已经排除的错误方向:

```text
不是 TVM 完全无法 build INT8 route
不是 native INT8 latency/energy 没跑通
不是只缺 output-level pyramid_level0/1/2 range
不是可以继续使用旧 QDQ/float32-heavy AP 结果替代 native INT8 AP
```

需要继续解决的是: 为 layer0 prefix 的 Conv/Relu/Add 输出采集真实 PyTorch reference min/max, 生成 scale-aware calibration v2, 再让 native INT8 AP gate 通过。

## 5. 下一阶段执行计划

执行模式: 单 agent, 不启用 agent team。

### 5.1 Native INT8 AP gate unblock

1. 在 H800 上导出完整 module inventory:

```text
pytorch_module_inventory_layer0_v1.json
source = model.pyramid_backbone.named_modules()
```

2. 扩展 `--collect-reference-ranges`, 输出真实:

```text
tensor_reference_ranges_layer0_v1.json
```

捕获策略:

| target | strategy |
|---|---|
| Conv output | `register_forward_hook` on resolved Conv module |
| ReLU output | hook reused ReLU module and record call index |
| Add output | block-level wrapper / forward patch capture pre-ReLU residual add |

3. 生成:

```text
tensor_quant_params_calibration_v2.json
```

4. rerun scale-aware prefix simulator, gate 条件:

```text
不再出现 layer0 Add heavy saturation
pyramid_level0 max_fraction 不再维持 0.758 级别重饱和
输出误差进入可解释范围, 否则写 op-level blocker 继续定位
```

5. 先跑 native INT8 AP smoke:

```text
s0_024
s0_040
s1_048
```

三点通过后推广 original60 native INT8 AP queue。

### 5.2 FP16 AP true eval 收口

FP16 latency/energy 已 60/60 measured, 下一阶段补齐:

```text
FP16 AP70: 55 remaining
```

执行要求:

1. 不复用 suspect FP16-tagged row。
2. 每个 FP16 AP row 必须证明 model/batch/runner 是 true FP16 或合规 AMP FP16。
3. 如果 checkpoint 缺失, 先查 inventory 和 label mapping；仍缺失时写 per-label blocker, 不中断其他 label。

### 5.3 rows/exports 收口

最终必须刷新:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_original60_ap_rows_v1.jsonl
rows/latency_original60_quant_rows_v1.jsonl
rows/energy_original60_quant_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
```

## 6. 硬目标

最终交付目标收口为:

```text
original60 x fp16:
  latency_ms = 60/60 measured
  energy_J_per_inference = 60/60 measured
  AP70 = 60/60 measured

original60 x native_int8:
  latency_ms = 60/60 measured
  energy_J_per_inference = 60/60 measured
  AP70 = 60/60 measured
```

也就是 120 个 config-precision cell 的三指标闭环。任何失败 cell 都必须保留:

```text
runner command
stdout/stderr
GPU id / env
raw artifact path
failure reason
debug attempt
重试或修复记录
```

遇到问题的处理规则:

1. 不因为单点失败停止全队列。
2. 先分类失败: checkpoint/data/shape/dtype/TVM build/runtime/numeric/AP adapter。
3. 做最小复现或 op-level blocker。
4. 修改 runner/job queue 后重试。
5. 只有经过反思、审查和问题解决仍不可解, 才写 quarantine/no-claim；其他 cell 继续跑。

## 7. 下一轮 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage3 original60 FP16/native INT8 三指标收口, 单 agent 执行, 不启动 agent team。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/38_6_28_交接文档_阶段三_ReferenceRangeCapturePlan与FP16INT8_AP收口.md
- multi_agent/methods/design/auto-tuning/progress/6_27/37_6_28_交接文档_阶段三_Layer0ReferenceRangeTargets与Hook采集计划.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md

当前事实:
- native INT8 backbone/subnet latency-energy route 已打通, original60 latency=60/60 measured, energy=60/60 measured。
- 当前 latency_ms 口径是 H800 + TVM backbone/subnet compiled module end-to-end, input=spatial_features, output=multiscale backbone/subnet outputs, full_network_claim=false。
- 它只能作为 RSU-side backbone/subnet compute proxy on H800, 不能写成 full perception network 或物理 RSU 设备端到端 latency。
- FP16 latency/energy 已 60/60 measured, FP16 AP70 只有 5/60 measured。
- native INT8 AP70 仍 0/60 measured, rows/native_int8_original60_ap_rows_v1.jsonl 不得生成或导入, 直到 true native INT8 AP gate 通过。
- 当前 INT8 AP blocker 是 missing_layer0_intermediate_reference_ranges, 最新细分 blocker 是 partial_module_inventory_only_add_relu_need_block_level_capture。
- 已有 artifact tensor_reference_range_capture_plan_layer0_v1.json: total=22, hook_ready=2, blocked=20。

优先任务:
1. 在 H800 上导出完整 model.pyramid_backbone.named_modules() inventory, 生成 pytorch_module_inventory_layer0_v1.json。
2. 扩展 --collect-reference-ranges, 对 layer0 Conv/Relu/Add 输出采集真实 PyTorch reference ranges; ReLU 要按 reused module call index 对齐, Add 要用 block-level capture, 不得伪造。
3. 生成 tensor_reference_ranges_layer0_v1.json 和 tensor_quant_params_calibration_v2.json。
4. rerun scale-aware prefix simulator, 消除 layer0 Add heavy saturation 后, 先跑 native INT8 AP smoke: s0_024, s0_040, s1_048。
5. INT8 AP smoke 通过后推广 native INT8 original60 AP queue。
6. 并行补齐 FP16 true AP eval 剩余 55 个 cell; 不复用 suspect FP16-tagged row。
7. 保留并复核 FP16/native INT8 latency 和 energy 60/60 measured artifact, 对缺 raw/digest/telemetry 的 cell 重跑或 quarantine。

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
