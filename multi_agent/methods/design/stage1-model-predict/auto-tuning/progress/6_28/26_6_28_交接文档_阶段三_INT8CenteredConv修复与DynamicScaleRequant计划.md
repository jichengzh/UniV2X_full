# 32_6_28_交接文档_阶段三_INT8CenteredConv修复与DynamicScaleRequant计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `31_6_28_交接文档_阶段三_INT8ZeroPointAwareAddRelu修复与ScalePropagationBlocker.md`

本轮新增进展: 在上一轮 zero-point aware `Relu/Add` 的基础上, 继续修复 native INT8 route 中 Conv 输入 zero point 未居中的问题。H800 已完成 centered-Conv route build/run 和 5-sample output sanity。结果显示三层 output RMSE 大幅下降, 但 correlation gate 仍未通过。因此当前 blocker 进一步收敛为固定 `/256` requant 缺少 per-tensor dynamic scale, 而不是权重、Relu/Add 或 Conv input zero point。

## 0. 当前权威三指标状态不变

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
total latency = 120/120 measured
total energy = 120/120 measured
total AP70 = 5/120 measured
```

仍不得生成或导入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

本轮 centered-Conv route 仍是 AP adapter 修复验证产物, 使用 debug row-write mode, 不写 canonical original60 latency/energy rows, 不产生 AP measured row。

## 1. 本轮代码修复

修改:

```text
scripts/stage2_h800_native_int8_op_alignment.py
framework/tests/test_stage2_native_int8_route.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
```

新增测试:

```text
test_native_prefix_simulator_centers_intermediate_conv_input_zero_point
```

修复内容:

1. `simulate_native_int8_graph_prefix(...)` 新增 tensor zero-point propagation。
2. graph input `spatial_features` 默认 zero point = 0。
3. Conv/Relu/Add/Identity 输出记录并传播 `output_zero_point`。
4. 中间 Conv 输入按 `input_zero_point` 居中后再卷积。
5. TVM route builder 维护 `tensor_zero_points`, 并在 `conv2d_native_int8(...)` 前插入 `x - input_zero_point` compute。
6. Conv op record 额外记录:

```text
input_zero_point
output_zero_point
```

修复前问题:

```text
Relu/Add 已经使用 128 作为中间 tensor zero point, 但后续 Conv 仍把 uint8 128 当作正激活卷进去。
```

修复后语义:

```text
spatial_features input zero_point = 0
intermediate Conv/Relu/Add output zero_point = 128
Conv accumulator input = uint8_input - input_zero_point
Relu = max(x, input_zero_point)
Add = lhs + rhs - lhs_zero_point - rhs_zero_point + output_zero_point
```

## 2. 本地 cached activation trace

artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_zp128_add_relu_v1/s0_024/native_prefix_trace_cached_activation_zp128_add_relu_centered_conv_v1/
```

到 `pyramid_level0` 无 `max_fraction >= 0.5` 的重饱和点。

关键输出:

| op | mean | max_fraction | min | max | input_zp | output_zp |
|---|---:|---:|---:|---:|---:|---:|
| layer0.0/Add | 123.335223 | 0.000002 | 0 | 255 | n/a | 128 |
| layer0.1/Add | 131.135383 | 0.000000 | 122 | 253 | n/a | 128 |
| layer0.2/Add | 131.438208 | 0.000000 | 125 | 253 | n/a | 128 |
| pyramid_level0 | 131.519782 | 0.000000 | 128 | 253 | 128 | 128 |

对比第31轮 zero-point aware Add/Relu 但未 centered Conv:

| op | 第31轮 max_fraction | 本轮 max_fraction |
|---|---:|---:|
| layer0.0/Add | 0.118992 | 0.000002 |
| layer0.1/Add | 0.258383 | 0.000000 |
| layer0.2/Add | 0.171961 | 0.000000 |
| pyramid_level0 | 0.171961 | 0.000000 |

结论:

```text
Conv input zero-point 居中是必要修复, 能继续消除 layer0 内部的 uint8 饱和。
```

## 3. H800 centered-Conv route build/run

run id:

```text
20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1
```

artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/
```

结果:

```text
status = success
latency_ms = 11.430506
energy_J = 2.7004522109892797
artifact_digest = 12e41f0216171108e78081b591a2c47264137127fdde05a01e540612a2105d49
op_counts = {"Add": 16, "Conv": 51, "Relu": 48}
full_network_claim = false
ap_measured = false
```

注意:

```text
该 run 使用 --allow-non-h800-debug 作为 row-write guard, 不写 canonical rows。
latency/energy 仅用于 route 修复对比, 不替代 original60 canonical native INT8 latency/energy 60/60 rows。
```

## 4. H800 op-level trace

artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/op_level_alignment_trace_v1/
```

first spatial Conv alignment:

```text
status = passed
processed_samples = 1
selected_op_count = 2
record_count = 4
records_passed = 4
records_failed = 0
weight_record_count = 2
weight_records_passed = 2
weight_records_failed = 0
```

native prefix trace 到 `pyramid_level0`:

```text
record_count = 22
first max_fraction >= 0.5 = null
```

关键 Add 输出:

| op | mean | max_fraction | min | max | output_zp |
|---|---:|---:|---:|---:|---:|
| layer0.0/Add | 123.290057 | 0.000002 | 0 | 255 | 128 |
| layer0.1/Add | 131.122494 | 0.000000 | 122 | 253 | 128 |
| layer0.2/Add | 131.428125 | 0.000000 | 125 | 253 | 128 |
| pyramid_level0 | 131.508041 | 0.000000 | 128 | 253 | 128 |

## 5. H800 5-sample output sanity

artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/output_dequant_sanity5_v1/
```

结果:

```text
status = blocked
processed_samples = 5
records = 15
ap_measured = false
full_network_claim = false
```

三层 output:

| output | samples | mae_mean | rmse_mean | corrcoef_mean | reference_range_mean | candidate_scale_mean | passed |
|---|---:|---:|---:|---:|---:|---:|---|
| pyramid_level0 | 5 | 1.027640 | 1.422918 | 0.198135 | 20.473965 | 0.162492 | false |
| pyramid_level1 | 5 | 0.156292 | 0.428120 | 0.177085 | 11.303640 | 0.128613 | false |
| pyramid_level2 | 5 | 0.049219 | 0.259391 | 0.066447 | 9.457661 | 0.579605 | false |

对比第31轮:

| output | 第31轮 rmse_mean | 本轮 rmse_mean | 第31轮 corrcoef_mean | 本轮 corrcoef_mean |
|---|---:|---:|---:|---:|
| pyramid_level0 | 10.198283 | 1.422918 | -0.342580 | 0.198135 |
| pyramid_level1 | 4.971554 | 0.428120 | -0.023631 | 0.177085 |
| pyramid_level2 | 2.642796 | 0.259391 | 0.353887 | 0.066447 |

解释:

1. `level0/level1/level2` 的 RMSE 都显著下降, 说明 Conv zero-point 居中有效。
2. `corrcoef_mean` 仍低于 0.5 gate, 说明 tensor ordering/relative magnitude 还没有恢复。
3. 这时不应放宽 AP gate 或直接跑 1789 measured AP; 应继续修 requant scale。

## 6. 当前根因判断

已解决:

```text
AP checkpoint 与 ONNX 权重一致性
Conv+BN fused initializer audit
first spatial Conv alignment
Relu zero-point clamp
Add zero-point double-counting
Conv input zero-point centering
```

仍未解决:

```text
fixed requant_u8 = floor(acc / 256) + 128 过于粗糙
Conv output scale 没有由 input_scale * weight_scale / output_scale 决定
Add 只做 zero-point alignment, 尚未做不同分支 scale alignment
runtime inventory 没有记录每个 tensor 的 scale/zero_point policy
```

当前 blocker:

```text
native INT8 route has correct zero-point propagation, but lacks dynamic per-tensor scale-aware requantization.
```

## 7. 下一阶段技术计划

### 7.1 实现 scale-aware simulator

先在 Python simulator 中完成, 不直接改 TVM:

```text
QuantTensor:
  values_uint8
  scale
  zero_point
  tensor_name
```

Conv:

```text
acc_int32 = conv2d(input_uint8 - input_zp, weight_int8)
real_scale = input_scale * weight_scale
output_uint8 = round(acc_int32 * real_scale / output_scale) + output_zp
```

Relu:

```text
dequant -> max(x, 0) -> quantize(output_scale, output_zp)
```

Add:

```text
lhs_real + rhs_real -> quantize(common output_scale, output_zp)
```

### 7.2 需要的 calibration 证据

对 `s0_024` 先采集 5-sample reference tensor ranges:

```text
layer0.0 conv/add/relu outputs
layer0.1 conv/add/relu outputs
layer0.2 conv/add/relu outputs
pyramid_level0/1/2
```

输出:

```text
tensor_quant_params_calibration_v1.json
scale_aware_prefix_trace_records.json
scale_aware_output_sanity_summary.json
```

准入:

```text
pyramid_level0/1/2 corrcoef_mean >= 0.5
rmse_mean <= 0.25 * reference_range_mean
```

### 7.3 再下沉到 TVM route builder

只有 scale-aware simulator 通过后, 再改:

```text
stage2_h800_native_int8_full_onnx_route.py
```

route builder 必须:

```text
读取或生成 tensor_quant_params
Conv 使用 scale-aware requant, 不再固定 /256
Add 对分支 scale 做 rescale
Relu 使用 tensor zero point
tvm_operator_inventory.json 写入每个 tensor 的 scale/zero_point
```

### 7.4 gate 顺序

```text
1. scale-aware simulator 5-sample all passed
2. H800 rebuild TVM route
3. H800 output sanity5 all passed
4. 20-sample AP smoke finite AP 且 pred_nonempty_count > 0
5. s0_024 1789-frame full AP gate
6. 写 rows/native_int8_original60_ap_rows_v1.jsonl 首行
7. 扩展 original60 60 行
```

## 8. 验证

本地:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 49 tests in 9.290s
OK
```

本地语法:

```text
python -m py_compile framework/stage2/native_int8_full_onnx.py framework/tests/test_stage2_native_int8_route.py scripts/stage2_h800_native_int8_op_alignment.py multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
exit code = 0
```

H800:

```text
py_compile = passed
targeted unittest = 2 tests OK
centered-Conv route build/run = success
op-level first spatial alignment = passed
output sanity5 = blocked
```

## 9. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。当前权威状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured; latency 统一使用 latency_ms, 口径是 H800 TVM backbone/subnet module end-to-end, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明为完整感知 pipeline 或真实 RSU 物理设备 latency。本轮已完成 checkpoint-consistent s0_024 route 的 Conv input zero-point 居中修复: graph input spatial_features zero_point=0, intermediate Conv/Relu/Add output zero_point=128, Conv accumulator input=uint8_input-input_zero_point。H800 centered-Conv route build/run 成功, latency_ms=11.430506, energy_J=2.7004522109892797, artifact_digest=12e41f0216171108e78081b591a2c47264137127fdde05a01e540612a2105d49, 但该 run 使用 debug row-write mode, 不写 canonical rows。H800 prefix trace 到 pyramid_level0 已无 max_fraction>=0.5 的重饱和点; output sanity5 相比第31轮显著改善但仍 blocked: pyramid_level0 rmse=1.422918 corr=0.198135, pyramid_level1 rmse=0.428120 corr=0.177085, pyramid_level2 rmse=0.259391 corr=0.066447。因此 rows/native_int8_original60_ap_rows_v1.jsonl 仍不得生成。下一步主线是实现真正 scale-aware requant: 先在 Python simulator 中引入 QuantTensor(values_uint8, scale, zero_point), Conv 使用 input_scale*weight_scale/output_scale requant, Add 做分支 scale alignment, Relu 使用 tensor zero_point; 采集 s0_024 5-sample reference tensor ranges 生成 tensor_quant_params_calibration_v1.json, scale-aware simulator 通过 pyramid_level0/1/2 corr>=0.5 且 rmse<=0.25*reference_range 后, 再下沉到 TVM route builder 并重建 H800 route。gate 顺序: scale-aware simulator all passed -> H800 rebuild -> output sanity5 all passed -> 20-sample AP smoke -> s0_024 1789-frame full AP -> 首行 native_int8 AP row -> 扩展 original60 60 行。并行继续 FP16 AP checkpoint recovery, 当前 rows/fp16_true_original60_ap_rows_v1.jsonl 为 5 行, 剩余 55 个 label 必须从备份/归档/用户提供路径恢复 checkpoint 后跑 true FP16 full AP。每批刷新 original60_quant summary/review/gap, 审计 energy rows 保持 120/120 measured; 若任何 row 被 summary 拒绝则按 label 补跑。遇到 build/eval/import/SSH/checkpoint/adapter/TVM 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 反思根因, 审查最近 artifact, 做最小复现, 修复并补跑失败 label/gate; 只有证明当前环境中确实无法解决, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker。
```
