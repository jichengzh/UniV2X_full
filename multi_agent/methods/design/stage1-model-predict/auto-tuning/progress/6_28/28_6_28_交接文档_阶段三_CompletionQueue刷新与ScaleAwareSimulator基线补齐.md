# 34_6_28_交接文档_阶段三_CompletionQueue刷新与ScaleAwareSimulator基线补齐

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `33_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_EnergyAP大范围补点计划.md`
- `32_6_28_交接文档_阶段三_INT8CenteredConv修复与DynamicScaleRequant计划.md`

本轮核心进展: 已按当前 canonical rows 刷新 FP16/INT8 original60 completion queue、completion review、gap report、AP true-eval source audit 和 AP blocker 列表。同时修复了 native INT8 诊断基线中的缺口: `QuantTensor` 和 scale-aware requant/add helper 已存在, 但 `simulate_scale_aware_int8_graph_prefix(...)` 缺失, 导致 native INT8 route 测试红。本轮已补齐该 Python scale-aware prefix simulator, 并通过相关测试。

## 0. 过期事实校正

早期第10份文档中写过:

```text
FP16 latency/energy measured 只有 s0_024/s1_048 两点
INT8 latency/energy measured 只有 s0_024/s1_048 两点且是旧 QDQ/TVM VM smoke
```

该事实已经过期。当前 authoritative rows 和 exports 显示:

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
```

后续必须以当前 rows/exports 为准, 不能再按“只有两点 latency/energy measured”的旧状态制定队列。

## 1. 当前口径仍不变

INT8 backbone/subnet native route 状态:

```text
H800 + TVM native INT8 full ONNX backbone/subnet route 已打通 build/run
original60 native INT8 latency rows = 60/60
original60 native INT8 energy rows = 60/60
native INT8 AP rows = 0/60
```

latency 口径:

```text
latency_ms = H800 + TVM measured backbone/subnet compiled module end-to-end latency
scope = spatial_features -> backbone/subnet multiscale outputs
full_network_claim = false
```

该 latency 可以作为 RSU-side backbone/subnet workload proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理边缘设备 latency。

## 2. 本轮刷新产物

### 2.1 Completion queue / review / gap report

执行:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

刷新:

```text
jobs/fp16_int8_original60_completion_queue_v1.jsonl
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_gap_report_latest.json
exports/fp16_int8_original60_gap_report_latest.md
```

刷新后 summary:

```text
total_jobs = 120
precision_counts = {"fp16": 60, "int8": 60}
latency = {"measured": 120}
energy = {"measured": 120}
ap = {"measured": 5, "no_claim": 115}
jobs_requiring_action = 115
```

completion queue 行数:

```text
jobs/fp16_int8_original60_completion_queue_v1.jsonl = 120
  fp16 = 60
  int8 = 60
```

required actions:

```text
run_true_fp16_ap_eval = 55 jobs
run_native_int8_ap_eval = 60 jobs
complete/no action = 5 jobs
```

结论: 下一阶段主缺口已经收敛为 AP, latency/energy 保持 120/120 measured, 只需要继续做 raw artifact / digest / telemetry 审计, 如发现不合规 row 再单点补跑。

### 2.2 AP true-eval queue / blockers

执行:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
```

刷新:

```text
jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl
quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl
exports/fp16_int8_original60_ap_true_eval_source_audit_latest.json
exports/fp16_int8_original60_ap_true_eval_source_audit_latest.md
```

刷新后 summary:

```text
total_jobs = 115
source_rows = {"fp16": 75, "int8": 4}
blocked = 115
ready_for_import = 0
```

AP true-eval queue 行数:

```text
jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl = 115
  fp16 = 55
  int8 = 60
```

blocker 行数:

```text
quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl = 115
  no_compliant_true_fp16_model_eval_source = 55
  no_compliant_native_int8_model_eval_source = 60
```

结论: 默认源中没有可直接 import 的合规 AP row。下一步不是继续等 import source, 而是实际运行:

```text
FP16 true eval: 55 labels
native INT8 eval: 先过 s0_024 scale-aware/output sanity/AP smoke gate, 再扩展 60 labels
```

## 3. 本轮代码修复

修改:

```text
scripts/stage2_h800_native_int8_op_alignment.py
```

新增/补齐:

```text
_tensor_quant_params(...)
_conv2d_accumulator_int32(...)
simulate_scale_aware_int8_graph_prefix(...)
```

已有 helper:

```text
QuantTensor(values_uint8, scale, zero_point, tensor_name)
requantize_int32_scale_aware(...)
scale_aware_add_uint8(...)
```

`simulate_scale_aware_int8_graph_prefix(...)` 当前语义:

1. graph input 以 `QuantTensor` 进入。
2. Conv 使用 centered uint8 activation、int8 weight 和 int32 accumulator。
3. Conv requant 使用:

```text
output_uint8 = round(acc_int32 * input_scale * weight_scale / output_scale) + output_zero_point
```

4. Add 使用真实 scale alignment:

```text
lhs_real = (lhs_uint8 - lhs_zp) * lhs_scale
rhs_real = (rhs_uint8 - rhs_zp) * rhs_scale
out_uint8 = round((lhs_real + rhs_real) / output_scale) + output_zp
```

5. Relu 先 dequant 到 real, 再 clamp `>=0`, 再按输出 tensor scale/zero_point requant。
6. Identity 继承或使用目标 tensor quant params。
7. trace record 写:

```text
schema = native_int8_scale_aware_prefix_trace_record_v1
op_index
conv_index
op_type
op_name
input_names
output_name
scale
zero_point
tensor summary
```

注意: 这只是 Python simulator 诊断基线, 还没有下沉到 TVM route builder, 不能据此写 native INT8 AP row。

## 4. 验证

Completion queue 生成:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
status = ok
total_jobs = 120
jobs_requiring_action = 115
```

AP true-eval queue 生成:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
status = ok
total_jobs = 115
blocked = 115
```

单测:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_original60_quant_completion
Ran 12 tests
OK
```

scale-aware targeted test:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route.Stage2NativeInt8RouteTest.test_scale_aware_prefix_simulator_uses_tensor_quant_params
Ran 1 test
OK
```

相关完整测试:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 52 tests
OK
```

语法检查:

```text
python -m py_compile framework/stage2/native_int8_full_onnx.py framework/stage2/original60_quant_completion.py scripts/stage2_h800_native_int8_op_alignment.py scripts/stage2_generate_fp16_int8_original60_completion_queue.py scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
exit code = 0
```

## 5. 下一步计划

### 5.1 FP16 AP 主线

目标:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl: 5 -> 60
remaining = 55
```

执行:

1. 读取 `jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl` 中 `precision=fp16` 的 55 个 blocked job。
2. 按 label 做 checkpoint recovery inventory, 覆盖 H800 当前目录、历史 manifest、备份路径、训练产物目录和别名命名。
3. 找到 checkpoint 后运行 `scripts/stage2_h800_true_fp16_ap_eval.py`。
4. 每批追加 `rows/fp16_true_original60_ap_rows_v1.jsonl`, 再刷新三指标总表和 completion review。
5. 找不到 checkpoint 的 label 写 per-label blocker, 不阻塞其他 label。

### 5.2 native INT8 AP 主线

目标:

```text
rows/native_int8_original60_ap_rows_v1.jsonl: 0 -> 60
```

当前 gate:

```text
scale-aware Python simulator 已补齐并有单测
仍需采集/生成 s0_024 tensor_quant_params_calibration_v1.json
仍需用真实 cached activation 跑 scale-aware prefix trace
仍需下沉 TVM route builder 并通过 output sanity5
```

执行顺序:

1. 用 `s0_024` cached activation 和 ONNX op records 跑 `simulate_scale_aware_int8_graph_prefix(...)`。
2. 生成:

```text
tensor_quant_params_calibration_v1.json
scale_aware_prefix_trace_records.json
scale_aware_output_sanity_summary.json
```

3. scale-aware simulator 通过 `pyramid_level0/1/2` sanity gate 后, 修改 TVM route builder。
4. H800 rebuild native INT8 route。
5. output sanity5 passed 后跑 20-sample AP smoke。
6. `s0_024` 1789-frame full AP 通过后, 写首行 native INT8 AP row。
7. 扩展 original60 60 labels。

### 5.3 失败处理

任一 label 或 gate 失败时:

1. 保存 stdout/stderr、runner command、GPU id、env、raw artifact、digest、failure reason。
2. 分类为 checkpoint、artifact、TVM build/runtime、shape、dtype、scale alignment、postprocess、AP import、SSH/env、telemetry 等。
3. 构造最小复现。
4. 修复 runner/adapter/route/queue 后补跑失败 label。
5. 其他 label 继续推进。
6. 只有证明当前环境不可解时, 才写 quarantine/no-claim blocker。

## 6. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。当前最新 authoritative 状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured; latency 统一使用 latency_ms, 口径是 H800 TVM backbone/subnet compiled module end-to-end, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明完整感知 pipeline 或真实 RSU 物理边缘设备 latency。本轮已刷新 jobs/fp16_int8_original60_completion_queue_v1.jsonl: total_jobs=120, fp16=60, int8=60, latency measured=120, energy measured=120, AP measured=5/no_claim=115, jobs_requiring_action=115; 已刷新 AP true-eval queue: jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl=115, fp16=55, int8=60, 全部 blocked, quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl=115, blocker 原因为 no_compliant_true_fp16_model_eval_source=55 和 no_compliant_native_int8_model_eval_source=60。不要把默认 source rows 直接 import 成 AP measured。INT8 技术基线: scripts/stage2_h800_native_int8_op_alignment.py 已补齐 QuantTensor scale-aware prefix simulator, 包括 _tensor_quant_params、_conv2d_accumulator_int32、simulate_scale_aware_int8_graph_prefix; 相关测试 framework.tests.test_stage2_native_int8_route 与 framework.tests.test_stage2_original60_quant_completion 共 52 tests OK, py_compile OK。下一阶段硬目标仍是 original60 60 labels x {fp16,native_int8} 的 AP/energy/latency 全量收口: latency/energy 保持 120/120 measured 并持续审计 raw artifact/digest/telemetry, FP16 AP 从 5/60 补到 60/60, native INT8 AP 从 0/60 补到 60/60。FP16 主线从 AP true-eval queue 中 55 个 fp16 blocked job 做 checkpoint recovery, 找到 checkpoint 后运行 scripts/stage2_h800_true_fp16_ap_eval.py 并追加 rows/fp16_true_original60_ap_rows_v1.jsonl。INT8 主线先用 s0_024 cached activation 和 ONNX op records 跑 simulate_scale_aware_int8_graph_prefix, 生成 tensor_quant_params_calibration_v1.json、scale_aware_prefix_trace_records.json、scale_aware_output_sanity_summary.json; simulator 通过 pyramid_level0/1/2 sanity 后下沉 TVM route builder, H800 rebuild, output sanity5 passed, 20-sample AP smoke finite/pred_nonempty, s0_024 1789-frame full AP, 写首行 rows/native_int8_original60_ap_rows_v1.jsonl, 再扩展 original60 60 labels。遇到任何 build/eval/import/SSH/checkpoint/adapter/TVM/telemetry 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 做失败分类、最小复现、反思审查、runner/adapter/route/queue 修复并补跑失败 label, 其他 label 继续推进; 只有证明当前环境确实不可解, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker 或 quarantine/no-claim row。
```
