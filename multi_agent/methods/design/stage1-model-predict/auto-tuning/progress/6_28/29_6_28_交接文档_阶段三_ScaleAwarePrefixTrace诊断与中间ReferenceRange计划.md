# 35_6_28_交接文档_阶段三_ScaleAwarePrefixTrace诊断与中间ReferenceRange计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `34_6_28_交接文档_阶段三_CompletionQueue刷新与ScaleAwareSimulator基线补齐.md`
- `32_6_28_交接文档_阶段三_INT8CenteredConv修复与DynamicScaleRequant计划.md`

本轮核心进展: 使用第34轮补齐的 `simulate_scale_aware_int8_graph_prefix(...)`, 对 `s0_024` centered-Conv route 的 cached activation 运行 scale-aware prefix trace 到 `pyramid_level0`。结果显示 naive/default scale propagation 会在 layer0 内部重新产生重饱和, 因此当前 blocker 从“缺 scale-aware simulator”进一步收敛为“缺中间 PyTorch reference tensor ranges / calibrated per-tensor output_scale”。本轮没有写入任何 AP measured row, `rows/native_int8_original60_ap_rows_v1.jsonl` 仍不得生成。

## 0. 当前权威覆盖状态不变

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
```

completion queue 状态:

```text
jobs/fp16_int8_original60_completion_queue_v1.jsonl = 120
latency measured = 120
energy measured = 120
AP measured = 5
AP no_claim = 115
jobs_requiring_action = 115
```

AP true-eval queue 状态:

```text
jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl = 115
  fp16 = 55
  int8 = 60
quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl = 115
ready_for_import = 0
blocked = 115
```

latency 口径仍固定:

```text
latency_ms = H800 + TVM backbone/subnet compiled module end-to-end latency
full_network_claim = false
```

## 1. 本轮 scale-aware prefix trace artifact

输入 route:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/
```

输入文件:

```text
onnx_op_records.json
tvm_operator_inventory.json
runtime_weights_int8.npz
output_dequant_sanity5_v1/bridge_call_000/agent_000/activation_uint8.npy
output_dequant_sanity5_v1/bridge_call_000/agent_000/activation_quant_summary.json
```

输出 artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1/
```

新增文件:

```text
scale_aware_prefix_trace_records.json
scale_aware_prefix_trace_summary.json
tensor_quant_params_calibration_v1.json
scale_aware_prefix_blocker.json
```

本轮 scale policy:

```text
graph_input: use cached activation minmax scale, zero_point=0
Conv: output_scale default = input_scale * weight_scale
Relu: inherit input scale/zero_point unless target params provided
Add: output_scale default = min(lhs_scale, rhs_scale), zero_point=128
```

注意: 这是 diagnostic policy, 不是 final calibration policy。

## 2. 诊断结果

执行结果:

```text
status = blocked
record_count = 22
last_output_name = pyramid_level0
first_heavily_saturated_output = /resnet/layer0/layer0.1/Add_output_0
failure_reason = scale_aware_default_propagation_heavy_saturation
```

关键 tensor:

| tensor | op_index | scale | mean | max_fraction | zero_fraction |
|---|---:|---:|---:|---:|---:|
| `/resnet/layer0/layer0.1/Add_output_0` | 13 | `3.648162474337463e-14` | 161.601452 | 0.633322 | 0.364821 |
| `pyramid_level0` | 21 | `5.1913579852445536e-20` | 224.283125 | 0.758135 | 0.000000 |

`pyramid_level0` trace summary:

```text
shape = [1, 24, 256, 256]
dtype = uint8
min = 128
max = 255
mean = 224.28312492370605
std = 54.383055451461004
max_fraction = 0.7581348419189453
```

对比第32轮 centered-Conv fixed `/256` route:

```text
centered-Conv fixed /256 prefix 到 pyramid_level0 无 max_fraction >= 0.5 的重饱和点,
但 output sanity5 correlation 仍 blocked。

本轮 naive scale-aware propagation 反而在 layer0.1/Add 重新重饱和,
说明不能只用 input_scale * weight_scale 逐层传播作为 output_scale。
```

## 3. 根因判断

已确认:

```text
scale-aware arithmetic helper 可运行
cached activation / route records / runtime weights 足够跑 prefix trace
默认传播 policy 不足以作为 final calibration
```

当前 blocker 更精确地写成:

```text
native INT8 route needs calibrated per-tensor output_scale from observed intermediate PyTorch reference ranges; naive propagation collapses layer0 scales and saturates residual Add outputs.
```

为什么会这样:

1. 每层 Conv 的 `input_scale * weight_scale` 是 accumulator real scale, 不是天然合适的 output tensor scale。
2. ReLU/Conv 多层后 scale 数值快速收缩到 `1e-14` 到 `1e-20` 级别。
3. Residual Add 两个分支使用过小 common output_scale 量化后, 大量值被 clip 到 0/255。
4. 当前已有 op-level artifact 只捕获了两个 spatial Conv 的输入, 没有捕获 layer0 每个 Conv/Add/Relu output 的 PyTorch reference range。

因此下一步必须采集:

```text
layer0.0 conv1/relu/conv2/relu_1/conv3/downsample/Add/relu_2 outputs
layer0.1 conv1/relu/conv2/relu_1/conv3/Add/relu_2 outputs
layer0.2 conv1/relu/conv2/relu_1/conv3/Add/relu_2 outputs
pyramid_level0
```

并基于这些 observed reference ranges 生成 `tensor_quant_params_calibration_v2.json`。

## 4. 验证

Artifact 校验:

```text
scale_aware_artifact_json = ok
status = blocked
record_count = 22
last_output_name = pyramid_level0
first_heavy = /resnet/layer0/layer0.1/Add_output_0
pyramid_level0_max_fraction = 0.7581348419189453
```

测试:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 52 tests in 7.994s
OK
```

语法检查:

```text
python -m py_compile framework/stage2/native_int8_full_onnx.py framework/stage2/original60_quant_completion.py scripts/stage2_h800_native_int8_op_alignment.py scripts/stage2_generate_fp16_int8_original60_completion_queue.py scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
exit code = 0
```

## 5. 下一步计划

### 5.1 增加中间 reference range 采集

修改目标:

```text
scripts/stage2_h800_native_int8_op_alignment.py
```

新增一个模式或参数, 对指定 output tensor / op name 批量注册 PyTorch hooks, 采集每个中间 tensor 的:

```text
shape
dtype
min
max
mean
std
recommended_uint8_scale
recommended_zero_point
raw reference path if persisted
```

输出:

```text
tensor_reference_ranges_layer0_v1.json
tensor_quant_params_calibration_v2.json
```

TDD 要求:

1. 先加单测, 构造小型 reference range 输入, 验证生成的 output_scale 不会退化到 `1e-14`。
2. 再实现 calibration helper。
3. 最后在 `s0_024` cached activation 上 rerun scale-aware prefix trace。

### 5.2 scale-aware simulator gate

rerun:

```text
simulate_scale_aware_int8_graph_prefix(..., tensor_quant_params=tensor_quant_params_calibration_v2)
```

准入:

```text
no max_fraction >= 0.5 through pyramid_level0
pyramid_level0 uint8 distribution not collapsed to all 128/255
if reference float tensor available: corrcoef improves vs centered-Conv fixed /256
```

### 5.3 TVM route builder gate

只有 Python simulator 通过后, 再改:

```text
raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
```

要求:

```text
Conv requant 使用 per-tensor output_scale
Add 做 residual branch scale alignment
Relu 使用 tensor zero_point
tvm_operator_inventory.json 记录每个 tensor scale/zero_point policy
```

### 5.4 AP 主线

FP16:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl: 5 -> 60
继续 55 个 label checkpoint recovery + true FP16 AP eval
```

native INT8:

```text
rows/native_int8_original60_ap_rows_v1.jsonl: 0 -> 60
scale-aware simulator -> H800 rebuild -> output sanity5 -> 20-sample AP smoke -> s0_024 full AP -> original60
```

## 6. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。当前 authoritative 状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured; latency 统一使用 latency_ms, 口径是 H800 TVM backbone/subnet compiled module end-to-end, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明完整感知 pipeline 或真实 RSU 物理边缘设备 latency。completion queue 已刷新: total_jobs=120, latency measured=120, energy measured=120, AP measured=5/no_claim=115; AP true-eval queue=115, 全部 blocked, fp16=55, int8=60。第34轮已补齐 QuantTensor scale-aware prefix simulator; 本轮已在 s0_024 centered-Conv route cached activation 上运行 scale-aware prefix trace 到 pyramid_level0, artifact 位于 raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/scale_aware_prefix_trace_cached_activation_v1/。结果: status=blocked, record_count=22, first_heavily_saturated_output=/resnet/layer0/layer0.1/Add_output_0, pyramid_level0 max_fraction=0.7581348419189453。结论: naive/default scale propagation 不是 final calibration; 当前 INT8 blocker 精确为缺少 layer0 中间 PyTorch reference tensor ranges / calibrated per-tensor output_scale, 不能据此生成 rows/native_int8_original60_ap_rows_v1.jsonl。下一步先 TDD 增加中间 reference range / tensor_quant_params_calibration_v2 生成逻辑, 采集 layer0.0/0.1/0.2 每个 Conv/Add/Relu 输出 range, rerun simulate_scale_aware_int8_graph_prefix 并要求 pyramid_level0 前无 max_fraction>=0.5; 通过后再下沉 TVM route builder, H800 rebuild, output sanity5, 20-sample AP smoke, s0_024 full AP, 然后扩展 original60 60 labels。FP16 主线继续从 AP queue 中 55 个 fp16 blocked job 做 checkpoint recovery, 找到 checkpoint 后运行 scripts/stage2_h800_true_fp16_ap_eval.py 并追加 rows/fp16_true_original60_ap_rows_v1.jsonl。遇到任何 build/eval/import/SSH/checkpoint/adapter/TVM/telemetry 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 做失败分类、最小复现、反思审查、runner/adapter/route/queue 修复并补跑失败 label, 其他 label 继续推进; 只有证明当前环境确实不可解, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker 或 quarantine/no-claim row。
```
