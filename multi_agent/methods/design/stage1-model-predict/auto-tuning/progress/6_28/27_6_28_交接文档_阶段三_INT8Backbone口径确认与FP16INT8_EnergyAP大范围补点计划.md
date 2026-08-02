# 33_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_EnergyAP大范围补点计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `24_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_AP收口计划.md`
- `30_6_28_交接文档_阶段三_CheckpointConsistentINT8Route通过与OutputScaleBlocker.md`
- `31_6_28_交接文档_阶段三_INT8ZeroPointAwareAddRelu修复与ScalePropagationBlocker.md`
- `32_6_28_交接文档_阶段三_INT8CenteredConv修复与DynamicScaleRequant计划.md`

本轮核心结论: 两个口径问题都可以继续向前推进, 但必须写清边界。INT8 的 `backbone/subnet native route` 已经能在 H800+TVM 上 build/run, 并且 original60 的 native INT8 latency/energy 行已达到 60/60; 但 INT8 AP 数值闭环尚未完成, 当前 blocker 是 scale-aware requant / residual branch scale alignment。当前所有量化 latency 都统一按 `latency_ms` 审阅, 口径是 H800+TVM backbone/subnet module end-to-end, 可作为 RSU-side backbone/subnet workload 的 proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理边缘设备实测 latency。

## 0. 口径确认

### 0.1 INT8 backbone 实现是否已经打通

结论: 是, 但只限于 `backbone/subnet native INT8 route build/run 与 latency/energy 实测`。

当前可声明:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
route_spec = full_onnx_topology_conv_relu_add_identity_v1
device = H800
runtime = TVM graph executor / TE route
full_network_claim = false
```

已经完成的 original60 measured rows:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60 rows
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 rows
```

不能声明:

```text
INT8 AP measured 已完成
INT8 full perception pipeline 已完成
INT8 route 已经数值等价或 AP 可直接 claim
真实 RSU 物理边缘设备 latency 已实测
```

原因: 当前 latest centered-Conv route 虽然已消除主要 zero-point 饱和, 但 `output_dequant_sanity5` 仍 blocked:

| output | rmse_mean | corrcoef_mean | status |
|---|---:|---:|---|
| pyramid_level0 | 1.422918 | 0.198135 | blocked |
| pyramid_level1 | 0.428120 | 0.177085 | blocked |
| pyramid_level2 | 0.259391 | 0.066447 | blocked |

当前 INT8 AP gate 的主 blocker:

```text
native INT8 route has correct zero-point propagation, but lacks dynamic per-tensor scale-aware requantization.
```

### 0.2 当前 latency 是否都是 backbone 端到端推理速度

结论: 是, 但这里的端到端是 `backbone/subnet compiled module` 的端到端。

统一写法:

```text
latency_ms = H800 + TVM measured backbone/subnet module end-to-end latency
scope = spatial_features -> backbone/subnet multiscale outputs
full_network_claim = false
```

解释:

1. 当前 latency 不是 per-op 估算, 而是 compiled backbone/subnet module 的实测运行时间。
2. 当前 latency 不包含 dataloader、完整 encoder、head、NMS、postprocess、dataset eval 或 V2X 全链路 IO。
3. 当前 latency 可以作为 RSU-side backbone/subnet dense workload 的 server-side proxy。
4. 当前 latency 不能写成真实 RSU 物理边缘段设备的绝对速度, 因为实测硬件是 H800。

## 1. 当前权威覆盖状态

已复核行数:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl = 60
rows/fp16_true_original60_energy_rows_v1.jsonl = 60
rows/fp16_true_original60_ap_rows_v1.jsonl = 5
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60
rows/native_int8_original60_ap_rows_v1.jsonl = missing / 0
```

按 precision 拆分:

| metric | FP16 | native INT8 | total |
|---|---:|---:|---:|
| latency_ms | 60/60 measured | 60/60 measured | 120/120 |
| energy_J_per_inference | 60/60 measured | 60/60 measured | 120/120 |
| AP70 | 5/60 measured | 0/60 measured | 5/120 |

权威 review:

```text
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
```

当前 review 状态:

```text
total_jobs = 120
precision_counts = {"fp16": 60, "int8": 60}
jobs_requiring_action = 115
```

## 2. 下一阶段硬目标

下一阶段不是继续证明 INT8 backbone 能否 build, 而是完成 `fp16` 和 `native INT8` 两种量化方式在 original60 上的 energy/AP 实测补点, 并把结果收口到可审阅总表。

最终验收目标:

```text
configs = original60 existing 60 labels
precisions = fp16, native_int8
latency unit = latency_ms
latency rows = 120/120 measured and audited
energy rows = 120/120 measured and audited
AP rows = 120/120 measured
full_network_claim = false for all backbone/subnet rows
jobs_requiring_action = 0 unless a per-label blocker proves unsolvable in current environment
```

更具体地说:

| output | current | target |
|---|---:|---:|
| FP16 energy | 60/60 | 60/60 audited, invalid rows single-label rerun |
| FP16 AP70 | 5/60 | 60/60 measured |
| native INT8 energy | 60/60 | 60/60 audited, invalid rows single-label rerun |
| native INT8 AP70 | 0/60 | 60/60 measured |

总表刷新目标:

```text
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

## 3. 单 agent 执行计划

本阶段只使用一个执行 agent, 不启动双 agent / reviewer agent team。执行 agent 必须自己完成实验、审查、反思、修复和补跑。

### 3.1 第一批: 状态审计与队列重建

目标: 建立 60 labels x 2 precision 的唯一执行队列, 不从零发明配置。

执行:

1. 从 `exports/fp16_int8_original60_completion_review_latest.json` 反推所有 no_claim cell。
2. 校验 FP16/native INT8 latency rows 的字段统一使用 `latency_ms`。
3. 校验 FP16/native INT8 energy rows 的 raw artifact、digest、telemetry source、`full_network_claim=false`。
4. 对 artifact 缺失、digest 不一致、字段不合规的 energy row 做单 label rerun。
5. 生成或刷新:

```text
jobs/fp16_int8_original60_energy_ap_completion_queue_v1.jsonl
```

每条 job 至少包含:

```text
label
precision
width
latency_status
energy_status
ap_status
checkpoint_path_or_blocker
native_int8_route_manifest
raw_artifact_dir
last_failure_reason
next_action
```

### 3.2 第二批: FP16 energy/AP 补点

FP16 energy 当前已有 60/60 measured, 下一阶段以审计为主; 若任一 row 缺 telemetry/raw evidence, 只补跑该 label。

FP16 AP 是主缺口:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl: 5 -> 60
remaining = 55 labels
```

执行顺序:

1. 从 completion review 中筛出 FP16 `AP=no_claim` 的 55 个 label。
2. 对每个 label 做 checkpoint recovery inventory, 覆盖 H800 当前目录、历史 manifest、备份目录、训练产物目录和别名命名。
3. 找到 checkpoint 后运行 true FP16 AP eval, 不允许把 suspect FP16-tagged 或 FP32 row 当作 true FP16 AP。
4. 每完成一批就追加/刷新:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_gap_report_latest.md
```

FP16 AP row 必须包含:

```text
label
precision = fp16
AP30
AP50
AP70
num_samples
checkpoint_path
checkpoint_digest
eval_config
raw_artifact
precision_evidence
full_network_claim = false
```

### 3.3 第三批: native INT8 energy/AP 补点

native INT8 energy 当前已有 60/60 measured, 下一阶段以审计为主; 若任一 row 缺 native route evidence 或 telemetry/raw evidence, 只补跑该 label。

native INT8 AP 当前是 0/60, 不允许直接生成 AP row。必须先解决 scale-aware requant gate:

```text
rows/native_int8_original60_ap_rows_v1.jsonl: missing / 0 -> 60
```

INT8 AP gate 顺序:

1. 在 Python simulator 中实现或复核 `QuantTensor(values_uint8, scale, zero_point)`。
2. Conv 使用:

```text
output_uint8 = round(acc_int32 * input_scale * weight_scale / output_scale) + output_zero_point
```

3. Add 做 residual 分支 scale alignment, 不再只做 zero-point alignment。
4. Relu 使用 tensor zero point, 不固定假设所有 tensor 都是 128。
5. 采集 `s0_024` 5-sample reference tensor ranges, 生成:

```text
tensor_quant_params_calibration_v1.json
scale_aware_prefix_trace_records.json
scale_aware_output_sanity_summary.json
```

6. scale-aware simulator 通过后, 再下沉到 TVM route builder。
7. H800 rebuild native INT8 route, 通过 output sanity5。
8. 跑 20-sample AP smoke, 要求 finite AP 且 pred_nonempty_count > 0。
9. 跑 `s0_024` 1789-frame full AP, 写首行 native INT8 measured AP。
10. 参数化 label/width/manifest/raw_dir, 扩展 original60 60 个 label。

native INT8 AP row 必须包含:

```text
label
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
AP30
AP50
AP70
num_samples
native_int8_route_manifest
runtime_weight_archive_digest
activation_quant_summary
worker_request_response_summary
raw_artifact
full_network_claim = false
```

## 4. 失败处理规则

遇到任何 build/eval/import/SSH/checkpoint/adapter/TVM/telemetry 问题, 不允许直接停止整个阶段。执行 agent 必须按下面顺序处理:

1. 保存 stdout/stderr、runner command、GPU id、env、config、raw artifact、digest、failure reason。
2. 分类失败:

```text
missing_checkpoint
missing_artifact
energy_telemetry_invalid
TVM_build_failure
TVM_runtime_failure
shape_mismatch
dtype_mismatch
scale_alignment_failure
postprocess_failure
AP_import_failure
SSH_or_env_failure
```

3. 构造最小复现, 只复现当前失败 label、当前失败 op 或当前失败 AP sample。
4. 审查最近 artifact, 明确是否存在 FP32/FP16/INT8 口径混淆。
5. 修复 runner、adapter、route builder、queue 或 import script 后重试失败 label。
6. 其他 label 继续推进, 不因为单点失败阻塞整批。
7. 只有在证明当前环境中无法解决时, 才允许留下 per-label blocker 或 quarantine/no-claim row。

可接受的无法解决条件示例:

```text
checkpoint 实体不存在且所有可访问备份均未找到
H800 环境或数据权限不可访问且当前 agent 无法恢复
某 label 的 ONNX/权重文件缺失且无可用来源
```

即使进入 blocker, 也必须写出:

```text
searched_paths
candidate_paths
failed_command
failure_type
failure_evidence
next_recovery_action
why_unsolved_in_current_environment
```

## 5. 预期交付物

最终必须能审阅到:

```text
rows/fp16_true_original60_energy_rows_v1.jsonl = 60 valid measured rows
rows/fp16_true_original60_ap_rows_v1.jsonl = 60 valid measured rows
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 valid measured rows
rows/native_int8_original60_ap_rows_v1.jsonl = 60 valid measured rows
exports/fp16_int8_original60_completion_review_latest.md shows jobs_requiring_action = 0
exports/original60_quant_three_metric_summary_latest.md uses latency_ms consistently
```

如果存在无法解决 blocker, 则完成标准改为:

```text
all solvable labels measured
all unsolved labels have per-label blocker artifact
completion review does not silently mark blocker rows as measured
```

## 6. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。两个口径先固定: 1) INT8 backbone/subnet native route 已经在 H800+TVM 上打通 build/run, original60 native INT8 latency/energy rows 已为 60/60, 但 INT8 AP 数值闭环尚未完成, 当前 blocker 是 scale-aware requant 与 residual branch scale alignment; 2) 当前所有量化 latency 对外统一使用 latency_ms, 口径是 H800 TVM backbone/subnet compiled module end-to-end, scope=spatial_features 到 multiscale backbone/subnet outputs, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明为完整感知 pipeline 或真实 RSU 物理边缘设备 latency。当前权威状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured, rows/native_int8_original60_ap_rows_v1.jsonl 仍不得在 gate 通过前生成。下一阶段硬目标: 完成 original60 现有 60 个配置 x {fp16,native_int8} 的 energy 和 AP 实测补点, latency rows 保持 120/120 并审计 latency_ms 单位; energy rows 保持/补齐到 FP16 60/60 与 native INT8 60/60 valid measured; AP rows 从当前 5/120 补到 120/120 measured, 即 rows/fp16_true_original60_ap_rows_v1.jsonl=60 且 rows/native_int8_original60_ap_rows_v1.jsonl=60。先从 exports/fp16_int8_original60_completion_review_latest.json 重建 jobs/fp16_int8_original60_energy_ap_completion_queue_v1.jsonl, 审计 energy raw artifact/digest/telemetry/full_network_claim, invalid row 单 label rerun。FP16 主线: 对剩余 55 个 AP=no_claim label 做 checkpoint recovery inventory, 找到 checkpoint 后运行 true FP16 AP eval, 追加 FP16 AP measured rows 并刷新 summary/review/gap。INT8 主线: 不再重复证明 backbone buildability, 先解决 scale-aware requant gate: Python simulator 维护 QuantTensor(values_uint8, scale, zero_point), Conv 使用 input_scale*weight_scale/output_scale requant, Add 做 residual 分支 scale alignment, Relu 使用 tensor zero_point; 采集 s0_024 5-sample tensor ranges, 生成 tensor_quant_params_calibration_v1.json、scale_aware_prefix_trace_records.json、scale_aware_output_sanity_summary.json, 通过 sanity 后下沉 TVM route builder, H800 rebuild, output sanity5 passed, 20-sample AP smoke finite/pred_nonempty, s0_024 1789-frame full AP, 写首行 native INT8 AP, 再扩展 original60 60 labels。遇到任何 build/eval/import/SSH/checkpoint/adapter/TVM/telemetry 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 做失败分类、最小复现、反思审查、runner/adapter/route/queue 修复并补跑失败 label, 其他 label 继续推进; 只有证明当前环境中确实无法解决, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker 或 quarantine/no-claim row。
```
