# 31_6_28_交接文档_阶段三_INT8ZeroPointAwareAddRelu修复与ScalePropagationBlocker

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `30_6_28_交接文档_阶段三_CheckpointConsistentINT8Route通过与OutputScaleBlocker.md`

本轮新增进展: 已把 INT8 AP blocker 从“末端 output dequant 不对”进一步定位到 native route 内部 `Relu/Add` 的 zero-point 语义错误, 并完成一轮修复、H800 rebuild 和 5-sample sanity。修复后明显降低了 layer0 residual Add 饱和, 但 output sanity 仍未通过, 因此当前 blocker 继续收敛为 per-layer scale propagation / requant scale 问题。

## 0. 当前权威状态不变

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

本轮 `s0_024` 新 route 是 AP adapter 修复验证产物, 使用 debug row-write mode, 没有写 canonical original60 rows。它不能当作 native INT8 AP measured row。

## 1. 本轮修复内容

### 1.1 op-level diagnostic 能力增强

修改:

```text
scripts/stage2_h800_native_int8_op_alignment.py
framework/tests/test_stage2_native_int8_route.py
```

新增能力:

```text
select_conv_records_from_op_records(...)
simulate_native_int8_graph_prefix(...)
summarize_uint8_trace_tensor(...)
--op-name-regex
--conv-indices
--trace-native-prefix
--trace-stop-output
```

用途:

1. 支持选择 deeper Conv, 不再只能 probe 直接消费 `spatial_features` 的两个 Conv。
2. 支持用 ONNX op_records + runtime int8 weights 从真实 `activation_uint8` 开始模拟 native INT8 prefix。
3. trace 每个 Conv/Relu/Add 输出的 `mean/min/max/zero_fraction/max_fraction`, 用来定位第一个明显饱和或尺度漂移点。

### 1.2 route builder 修复 zero-point aware Relu/Add

修改:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
```

旧语义:

```text
Relu: uint8 passthrough
Add: clip(lhs + rhs, 0, 255)
```

修复后语义:

```text
UINT8_ZERO_POINT = 128
Relu: max(x, 128)
Add: clip(lhs + rhs - 128, 0, 255)
```

原因:

`conv2d -> requant_u8` 使用 `floor(conv / 256) + 128`, 也就是 128 表示近似零点。旧 route 把 Relu 写成 passthrough 会保留小于 128 的负值; 旧 Add 直接相加会把两个分支的 zero point 也相加, 导致 residual Add 大面积饱和到 255。

## 2. 本地 cached activation 证据

旧 prefix trace artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/native_prefix_trace_cached_activation_v1/
```

旧语义下的关键饱和:

| op | mean | max_fraction |
|---|---:|---:|
| layer0.0/Add | 210.240610 | 0.497298 |
| layer0.1/Add | 234.031701 | 0.830791 |
| layer0.2/Add | 246.869972 | 0.875048 |
| pyramid_level0 | 246.869972 | 0.875048 |

新 zero-point aware simulator artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/native_prefix_trace_cached_activation_zp128_add_relu_v1/
```

新语义下到 `pyramid_level0` 没有 `max_fraction >= 0.5` 的 tensor:

| op | mean | max_fraction |
|---|---:|---:|
| layer0.0/Add | 130.270036 | 0.118992 |
| layer0.1/Add | 146.417304 | 0.258383 |
| layer0.2/Add | 133.269732 | 0.171961 |
| pyramid_level0 | 163.895142 | 0.171961 |

结论:

```text
Relu/Add zero-point 修复是必要修复, 能显著降低 layer0 的 uint8 饱和;
但这只是 route 语义修复, 还没有证明 AP 数值闭环通过。
```

## 3. H800 新 route build/run

run id:

```text
20260628_checkpoint_consistent_s0_024_native_int8_route_zp128_add_relu_v1
```

artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_zp128_add_relu_v1/s0_024/
```

结果:

```text
status = success
latency_ms = 9.477227
energy_J = 2.1911997284834084
artifact_digest = b2fa57aee4ad1a7fa1d26462915ee0ec1cffb3547ed7579cf6d5b0bc5e293f18
op_counts = {"Add": 16, "Conv": 51, "Relu": 48}
full_network_claim = false
ap_measured = false
```

注意:

```text
本 run 使用 --allow-non-h800-debug 作为 row-write guard, 即使在 H800 上运行也不写 canonical rows。
```

## 4. H800 op-level trace

artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_zp128_add_relu_v1/s0_024/op_level_alignment_trace_v1/
```

op-level first spatial conv:

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

| op | mean | max_fraction | zero_fraction | min | max |
|---|---:|---:|---:|---:|---:|
| layer0.0/Add | 130.299347 | 0.118982 | 0.126558 | 0 | 255 |
| layer0.1/Add | 146.419898 | 0.258407 | 0.083694 | 0 | 255 |
| layer0.2/Add | 133.271067 | 0.171963 | 0.041667 | 0 | 255 |
| pyramid_level0 | 163.895850 | 0.171963 | 0.0 | 128 | 255 |

结论:

```text
旧 route 的 residual Add 大面积饱和已经被修复。
```

## 5. H800 5-sample output sanity

artifact:

```text
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_zp128_add_relu_v1/s0_024/output_dequant_sanity5_v1/
```

结果:

```text
status = blocked
processed_samples = 5
records = 15
ap_measured = false
full_network_claim = false
```

三层 output summary:

| output | samples | mae_mean | rmse_mean | corrcoef_mean | reference_range_mean | candidate_scale_mean | passed |
|---|---:|---:|---:|---:|---:|---:|---|
| pyramid_level0 | 5 | 6.232413 | 10.198283 | -0.342580 | 20.474607 | 0.161217 | false |
| pyramid_level1 | 5 | 3.040501 | 4.971554 | -0.023631 | 11.294200 | 0.088931 | false |
| pyramid_level2 | 5 | 1.177394 | 2.642796 | 0.353887 | 9.453263 | 0.074435 | false |

对比上一轮旧 route:

| output | old rmse_mean | new rmse_mean | old corrcoef_mean | new corrcoef_mean |
|---|---:|---:|---:|---:|
| pyramid_level0 | 19.066432 | 10.198283 | -0.415779 | -0.342580 |
| pyramid_level1 | 11.050527 | 4.971554 | -0.307276 | -0.023631 |
| pyramid_level2 | 0.253762 | 2.642796 | null | 0.353887 |

解释:

1. level0/level1 的误差明显下降, 说明 Add/ReLU zero-point 修复有效。
2. level2 旧结果是全 255 饱和导致的偶然低 MAE, 不能视为真实通过; 新 route 后 level2 变成有动态范围, 但仍未对齐。
3. 三层 output 仍未全部 passed, 因此仍不能跑 1789-frame full AP measured row。

## 6. 当前根因判断

已解决:

```text
旧 benchmark ONNX 与 AP checkpoint 权重不一致
Conv+BN fused weight alignment
first spatial Conv alignment
Relu uint8 passthrough 错误
residual Add zero-point double-counting 导致的大面积饱和
```

仍未解决:

```text
per-layer output scale propagation
fixed requant_u8 divisor = 256 是否匹配每层 accumulator/weight/input scale
不同 residual 分支的 scale alignment
跨 layer0 -> layer1 -> layer2 的 output scale 继承
只靠三层末端 minmax dequant 无法恢复 tensor ordering/correlation
```

当前主 blocker 应写成:

```text
native INT8 route has zero-point-correct Relu/Add, but still lacks calibrated per-layer scale propagation/requant policy.
```

## 7. 下一阶段最小技术计划

### 7.1 先实现 scale-aware prefix simulator

目标:

```text
在 Python simulator 中为每个 tensor 维护 scale/zero_point, 不只维护 uint8 值。
```

建议:

1. 从 graph input `spatial_features` minmax 得到 activation scale/zero_point。
2. 每个 Conv 使用:

```text
accumulator_real_scale = input_scale * weight_scale
choose output_scale per tensor
requant = round(accumulator * accumulator_real_scale / output_scale) + output_zero_point
```

3. ReLU:

```text
max(x, output_zero_point)
```

4. Add:

```text
dequantize or rescale lhs/rhs to common output_scale, then add and requant
```

5. 输出每层:

```text
scale
zero_point
clip ratio
rmse/corrcoef against PyTorch block/module output where available
```

### 7.2 再把 scale policy 下沉到 TVM route builder

只有 simulator 能让 `pyramid_level0/1/2` sanity passed 后, 再改:

```text
stage2_h800_native_int8_full_onnx_route.py
```

修复项:

```text
Conv requant 不再固定 /256
Add 不只做 lhs+rhs-128, 而是做 branch scale alignment
Relu 使用对应 tensor zero_point
runtime inventory 写入每个 tensor 的 scale/zero_point
```

### 7.3 gate 顺序

```text
1. scale-aware simulator: 5-sample pyramid_level0/1/2 all passed
2. H800 rebuild route
3. op-level trace: no heavy saturation, deeper block outputs improve
4. H800 5-sample output sanity: all passed
5. 20-sample AP smoke: finite AP, pred_nonempty_count > 0
6. s0_024 1789-frame full AP gate
7. 首行 rows/native_int8_original60_ap_rows_v1.jsonl
8. 扩展 original60 60 行
```

## 8. 验证

本地:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 48 tests in 6.348s
OK
```

本地语法:

```text
python -m py_compile framework/stage2/native_int8_full_onnx.py framework/tests/test_stage2_native_int8_route.py scripts/stage2_h800_native_int8_op_alignment.py multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
exit code = 0
```

H800:

```text
py_compile route/op_alignment = passed
targeted unittest = 2 tests OK
route build/run = success
op-level first spatial alignment = passed
output sanity5 = blocked
```

## 9. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。当前权威状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured; latency 统一使用 latency_ms, 口径是 H800 TVM backbone/subnet module end-to-end, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明为完整感知 pipeline 或真实 RSU 物理设备 latency。本轮已完成 checkpoint-consistent s0_024 route 的 zero-point aware Relu/Add 修复: Relu=max(x,128), Add=clip(lhs+rhs-128,0,255); H800 rebuild 成功, latency_ms=9.477227, energy_J=2.1911997284834084, artifact_digest=b2fa57aee4ad1a7fa1d26462915ee0ec1cffb3547ed7579cf6d5b0bc5e293f18, 但该 run 使用 debug row-write mode, 不写 canonical rows。H800 prefix trace 到 pyramid_level0 已无 max_fraction>=0.5 的重饱和点, layer0.1/Add max_fraction 从旧语义 0.830791 降到 0.258407; 但 5-sample output sanity 仍 blocked: pyramid_level0 rmse=10.198283 corr=-0.342580, pyramid_level1 rmse=4.971554 corr=-0.023631, pyramid_level2 rmse=2.642796 corr=0.353887, 因此 rows/native_int8_original60_ap_rows_v1.jsonl 仍不得生成。下一步主线是实现 scale-aware prefix simulator 和 route builder: 每个 tensor 维护 scale/zero_point, Conv requant 不再固定 /256, Add 做 residual 分支 scale alignment, Relu 使用对应 zero_point, 并把 tensor scale/zero_point 写入 runtime inventory。gate 顺序: scale-aware simulator 5-sample all passed -> H800 rebuild -> deeper op trace -> 5-sample output sanity all passed -> 20-sample AP smoke -> s0_024 1789-frame full AP -> 首行 native_int8 AP row -> 扩展 original60 60 行。并行继续 FP16 AP checkpoint recovery, 当前 rows/fp16_true_original60_ap_rows_v1.jsonl 为 5 行, 剩余 55 个 label 必须从备份/归档/用户提供路径恢复 checkpoint 后跑 true FP16 full AP。每批刷新 original60_quant summary/review/gap, 审计 energy rows 保持 120/120 measured; 若任何 row 被 summary 拒绝则按 label 补跑。遇到 build/eval/import/SSH/checkpoint/adapter/TVM 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 反思根因, 审查最近 artifact, 做最小复现, 修复并补跑失败 label/gate; 只有证明当前环境中确实无法解决, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker。
```
