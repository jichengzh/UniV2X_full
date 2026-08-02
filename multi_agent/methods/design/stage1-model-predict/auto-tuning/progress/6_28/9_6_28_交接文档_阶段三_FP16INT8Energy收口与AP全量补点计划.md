# 15_6_28_交接文档_阶段三_FP16INT8Energy收口与AP全量补点计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/14_6_28_交接文档_阶段三_FP16Latency60收口与EnergyAP全量计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.md`

## 0. 两个口径结论

### 0.1 INT8 backbone/subnet route 是否已经打通

是, 但只限于当前 Stage2 定义的 backbone/subnet module。

当前可声明的 INT8 route 是:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
engine_kind = tvm_graph_executor
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

这个 route 已经不是旧的 QDQ-heavy / float32-heavy lowering。Original60 的 native INT8 latency 和 energy 均已有 H800 TVM measured row。

不能扩大的声明:

```text
不是 full perception network end-to-end
不是 RSU 物理边缘设备实测速度
不是 AP/eval route 已经打通
```

### 0.2 当前所有 latency 是否都是 backbone/subnet module 端到端推理速度

是, 但必须限定为 H800 TVM 上的 backbone/subnet module 推理。

当前导出总表统一使用 `ms`:

```text
latency_ms = latency_p50_us / 1000
```

原始 latency row 中仍保留 `latency_unit=us` 和 `latency_p50_us` 等采样字段, 这是底层 runner 的记录格式; 对外审阅和总表使用 ms。

当前不能写成:

```text
近似 RSU 边缘段设备推理速度
```

除非后续明确把 RSU edge backend 定义为同一 H800 TVM backend, 或在真实 RSU 硬件/边缘 backend 上补测并建立换算依据。

## 1. 本轮新增结果

本轮在上一版 FP16 latency 60/60、INT8 latency 60/60、INT8 energy 60/60 的基础上, 又完成 FP16 true route original60 energy 60/60。

### 1.1 FP16 energy full batch

run_id:

```text
20260628_fp16_true_original60_energy_batch001_60labels
```

远端执行结果:

```json
{
  "returncode": 0,
  "run_id": "20260628_fp16_true_original60_energy_batch001_60labels"
}
```

本地产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260628_fp16_true_original60_energy_batch001_60labels/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260628_fp16_true_original60_energy_batch001_60labels_remote_run_log/
```

行级校验:

```text
rows = 60
unique_labels = 60
energy_J range = 2.0808905856729973 to 12.121410915628456
contract_failures = 0
```

每行保持:

```text
precision = fp16
backend = h800_tvm_power_telemetry
measurement_status = measured
quant_method = h800_tvm_true_fp16_onnx_relax
quant_scope = backbone_only
engine_kind = tvm_vm
full_network_claim = false
source_files includes idle_power_samples.csv and active_power_samples.csv
```

### 1.2 INT8 energy full batch

已有产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch002_retry2_57labels/
```

行级状态:

```text
rows = 60
energy_J range = 0.2598788607420836 to 2.796973770305826
precision = int8
backend = h800_tvm_power_telemetry
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
engine_kind = tvm_graph_executor
full_network_claim = false
```

## 2. 当前 canonical 三表状态

已刷新:

```text
rows/latency_original60_quant_rows_v1.jsonl
rows/energy_original60_quant_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/original60_quant_three_metric_summary_latest.csv
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

全量三精度表状态:

| metric | measured | no_claim | total |
|---|---:|---:|---:|
| latency | 178 | 2 | 180 |
| energy | 120 | 60 | 180 |
| AP70 | 0 | 180 | 180 |

解释:

- latency 的 2 个 `no_claim` 是 FP32 遗留缺口, 不属于 FP16/INT8 本阶段收口范围。
- energy 的 60 个 `no_claim` 是 FP32 energy 未跑, 不属于 FP16/INT8 本阶段收口范围。
- FP16/INT8 的 AP 仍全部是 `no_claim`。

FP16/INT8 completion review 状态:

| precision set | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 + INT8 original60 | 120/120 measured | 120/120 measured | 0/120 measured |

Completion summary:

```json
{
  "latency": {"measured": 120},
  "energy": {"measured": 120},
  "ap": {"no_claim": 120}
}
```

## 3. Energy 和 AP 数据位置

### 3.1 FP16 energy

合规 measured row:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl
```

raw telemetry:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260628_fp16_true_original60_energy_batch001_60labels/<label>/
```

每个 label 目录应包含:

```text
energy_result_metaschedule_tuned.json
telemetry_payload_metaschedule_tuned.json
idle_power_samples.csv
active_power_samples.csv
layer_precision_summary.json
*_backbone_true_fp16.onnx
```

### 3.2 INT8 energy

合规 measured row:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

raw telemetry:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/<label>/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch002_retry2_57labels/<label>/
```

每个 label 目录应包含:

```text
energy_result.json
idle_power_samples.csv
active_power_samples.csv
tvm_operator_inventory.json
*_native_int8_full_onnx_tvm_graph.so
```

### 3.3 FP16/INT8 AP

当前没有合规 measured AP row。

当前 AP 状态只在 canonical 表中体现为:

```text
rows/ap_original60_quant_rows_v1.jsonl
FP16 AP70 = 0/60 measured
INT8 AP70 = 0/60 measured
```

因此不能把任何 energy smoke、latency smoke、TRT reference、预测值或插值值写成 AP measured。

## 4. 下一阶段唯一核心目标

下一阶段不再把 energy 作为主要缺口, 而是收口 FP16/INT8 的 original60 AP 实测。

具体目标:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 60/60 measured |
| INT8 | 60/60 measured | 60/60 measured | 60/60 measured |

下一阶段只允许以下两类结果:

1. 产出 120 条合规 AP measured row, 并刷新 canonical 表到 `AP measured=120`。
2. 如果某一路由确实不可运行, 产出逐点 blocker artifact, 包括失败命令、stdout/stderr、缺失接口、最小复现、已尝试修复和下一步方案; 不能只写 "无法完成"。

## 5. 下一阶段执行计划

### 5.1 先补 AP ingestion

当前 `scripts/stage2_generate_original60_quant_state_coverage.py` 对 AP 仍是固定 `no_claim` 逻辑。需要先补 AP measured row ingestion:

```text
新增 CLI:
--fp16-ap-rows
--int8-ap-rows

新增或扩展测试:
framework/tests/test_stage2_lut_productization.py
framework/tests/test_stage2_original60_quant_completion.py
```

测试必须覆盖:

```text
合规 FP16 AP row 进入 canonical AP 表
合规 INT8 AP row 进入 canonical AP 表
predicted/interpolated/TRT-reference AP row 被拒绝
缺少 dataset split / checkpoint / eval command / raw eval output 的 AP row 被拒绝
```

### 5.2 再打通 FP16 AP eval

目标输出:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
raw/ap_eval_original60/fp16_true_<run_id>/
```

每条 row 必须包含:

```text
precision = fp16
quant_method = h800_tvm_true_fp16_onnx_relax
quant_scope = backbone_only 或明确的 eval route scope
dataset split
checkpoint / ckpt digest
eval command
raw eval output
AP30 / AP50 / AP70
full_network_claim
```

如果 AP eval 需要 full model 而当前 latency/energy 只有 backbone module, 必须把 AP route scope 写清楚, 不能把 backbone module latency scope 误写为 full network scope。

### 5.3 打通 native INT8 AP eval

目标输出:

```text
rows/native_int8_full_onnx_original60_ap_rows_v1.jsonl
raw/ap_eval_original60/native_int8_full_onnx_<run_id>/
```

每条 row 必须包含:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8 或明确的 eval route scope
dataset split
checkpoint / ckpt digest
eval command
raw eval output
AP30 / AP50 / AP70
full_network_claim
```

如果 native INT8 artifact 不能直接接入 eval, 必须先做最小复现和接口修复, 不要直接停在 "AP backend missing"。

### 5.4 刷新总表并做审阅产物

AP row 合规后刷新:

```text
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/original60_quant_three_metric_summary_latest.csv
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

验收状态必须达到:

```text
FP16/INT8 latency measured = 120/120
FP16/INT8 energy measured = 120/120
FP16/INT8 AP measured = 120/120
jobs_requiring_action = 0
```

### 5.5 问题处理要求

遇到 build/eval/import 失败时, 下一阶段 agent 不应直接停止。必须按以下顺序处理:

```text
1. 保存失败命令、stdout、stderr、环境变量摘要和 artifact 路径
2. 判断失败属于代码 bug、数据缺口、接口缺口、环境缺口还是理论不可行
3. 对代码 bug 或接口缺口先做最小修复和单点复测
4. 单点通过后再批量补跑失败 label
5. 只有外部依赖缺失、权限不可恢复、数据源不存在等确实无法在当前环境解决的问题, 才写 blocker
```

## 6. 下一阶段 /goal 命令

```text
/goal 继续在 ${V2X_ROOT} 执行 Stage2 original60 FP16/INT8 AP 全量实测收口。不要启动 agent team, 仅使用单 agent 完成实验、修复和审查闭环。当前已完成 FP16/INT8 latency 120/120 measured、energy 120/120 measured; AP 仍为 0/120 measured。下一阶段目标是: 1) 为 scripts/stage2_generate_original60_quant_state_coverage.py 补齐合规 AP row ingestion, 包括拒绝 predicted/interpolated/TRT-reference AP 的测试; 2) 产出 rows/fp16_true_original60_ap_rows_v1.jsonl 和 rows/native_int8_full_onnx_original60_ap_rows_v1.jsonl, 每种 precision 60 行、共 120 行, 每行包含 dataset split、checkpoint/digest、eval command、raw eval output、AP30/AP50/AP70、precision route、full_network_claim; 3) 刷新 original60_quant_20260627 的 summary/review/gap/queue, 验收 FP16/INT8 latency=120/120 measured、energy=120/120 measured、AP=120/120 measured、jobs_requiring_action=0; 4) 遇到任何 build/eval/import 问题, 先保存 blocker artifact 并进行反思、审查、最小复现、修复和失败 label 补跑, 不允许直接停止, 除非证明当前环境中确实不可解决。
```

