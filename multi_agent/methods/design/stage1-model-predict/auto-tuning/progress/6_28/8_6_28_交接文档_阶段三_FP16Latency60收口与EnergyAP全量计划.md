# 14_6_28_交接文档_阶段三_FP16Latency60收口与EnergyAP全量计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/13_6_28_交接文档_阶段三_INT8Original60LatencyEnergy收口与FP16AP剩余计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.md`

## 0. 先回答两个口径问题

### 0.1 INT8 backbone/subnet 是否已经打通

是, 但只限于当前 Stage2 定义的 backbone/subnet module:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
engine_kind = tvm_graph_executor
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

当前 native INT8 full-ONNX original60 已完成:

| metric | measured | target | status |
|---|---:|---:|---|
| latency_ms | 60 | 60 | complete |
| energy_J / inference | 60 | 60 | complete |
| AP70 | 0 | 60 | pending |

结论: TVM native INT8 backbone/subnet route 已经不是 QDQ-heavy / float32-heavy lowering 的旧路线; original60 的 latency 和 energy 已有真实 H800 TVM measured row。

### 0.2 当前 latency 是否是端到端 RSU 边缘设备速度

不是。

当前所有 canonical latency row 的统一口径是:

```text
latency_ms = H800 + TVM measured backbone/subnet module inference latency
full_network_claim = false
```

更具体地说:

- FP32 latency: `quant_scope=backbone_only`, `engine_kind=tvm_vm`。
- FP16 latency: `quant_scope=backbone_only`, `engine_kind=tvm_vm`。
- INT8 latency: `quant_scope=backbone_subnet_native_int8`, `engine_kind=tvm_graph_executor`。

它们是 backbone/subnet module 的端到端推理时间, 不是完整 perception network 的端到端 latency, 也不是 RSU 物理边缘设备的绝对推理速度。除非后续明确把 RSU 边缘段 backend 定义为同一 H800 TVM backend, 否则不能写成 "近似 RSU 边缘设备速度"。

## 1. 本轮新增结果

本轮在上一轮 INT8 latency/energy 60/60 的基础上, 又完成 FP16 true route original60 latency 60/60。

### 1.1 H800 FP16 latency batch

run_id:

```text
20260628_fp16_true_original60_latency_batch002_60labels
```

远端执行结果:

```json
{
  "returncode": 0,
  "started_at": "2026-06-28T01:58:57+08:00",
  "ended_at": "2026-06-28T02:22:33+08:00",
  "labels_requested": 60
}
```

本地产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260628_fp16_true_original60_latency_batch002_60labels/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260628_fp16_true_original60_latency_batch002_60labels_remote_run_log/
```

行级校验:

```text
rows = 60
unique_labels = 60
missing_labels = []
extra_labels = []
contract_failures = 0
latency_ms range = 12.451139 to 51.759077
```

每行保持:

```text
precision = fp16
measurement_status = measured
quant_method = h800_tvm_true_fp16_onnx_relax
quant_scope = backbone_only
engine_kind = tvm_vm
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

刷新后计数:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 2/60 measured | 0/60 measured |
| INT8 | 60/60 measured | 60/60 measured | 0/60 measured |

Completion summary:

```json
{
  "latency": {"measured": 120},
  "energy": {"measured": 62, "no_claim": 58},
  "ap": {"no_claim": 120}
}
```

解释:

- `latency measured=120` = FP16 60 + INT8 60。
- `energy measured=62` = FP16 smoke 2 + INT8 native 60。
- AP 对 FP16/INT8 仍没有合规 measured source。

## 3. 仍然缺什么

### 3.1 FP16 energy

当前只有 2 个 original60 FP16 energy measured:

```text
s0_024
s1_048
```

现有 smoke 文件:

```text
rows/fp16_true_ap_energy_smoke_rows_v1.jsonl
raw/fp16_true_energy/
```

注意: `fp16_true_ap_energy_smoke_rows_v1.jsonl` 文件名里有 `ap`, 但内容是 energy telemetry row, 不包含 AP70 measured。

下一阶段应该产品化 original60 FP16 energy runner, 输出:

```text
rows/fp16_true_original60_energy_rows_v1.jsonl
raw/fp16_true_energy_original60/<run_id>/
```

最低验收:

```text
FP16 energy measured = 60/60
backend = h800_tvm_power_telemetry
energy_unit = joule_per_inference
source_files includes idle_power_samples.csv and active_power_samples.csv
full_network_claim = false
```

### 3.2 FP16/INT8 AP70

当前:

```text
FP16 AP70 measured = 0/60
INT8 AP70 measured = 0/60
```

AP 不能由 backbone latency/energy 推断。下一阶段必须实现或恢复合规 eval/import route, 输出:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_full_onnx_original60_ap_rows_v1.jsonl
raw/ap_eval_original60/<run_id>/
```

每条 AP measured row 必须包含:

```text
dataset split
checkpoint / ckpt digest
eval command
precision route
AP30 / AP50 / AP70
raw eval output
full_network_claim
```

禁止:

```text
predicted AP
interpolated AP
model-fit AP
TRT reference 冒充 H800 TVM measured
```

如果 eval route 无法直接连接 native INT8 backbone artifact, 必须写出逐点 blocker artifact, 包括失败命令、stdout/stderr、缺失接口、最小复现和下一步修复。

### 3.3 AP row 进入 canonical 表的代码缺口

当前 `scripts/stage2_generate_original60_quant_state_coverage.py` 对 AP 仍是固定 no_claim 逻辑。下一阶段需要先加 AP row ingestion, 再跑全量 AP:

```text
新增 CLI:
--fp16-ap-rows
--int8-ap-rows

新增测试:
framework/tests/test_stage2_lut_productization.py
  - compliant FP16 AP row enters canonical AP table
  - compliant INT8 AP row enters canonical AP table
  - predicted/interpolated/TRT-reference AP row is rejected
```

## 4. 下一阶段收口目标

最终目标不是再做 smoke, 而是把 existing original60 的 FP16/INT8 两种量化方式收口为全量三指标:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 60/60 measured |
| INT8 | 60/60 measured | 60/60 measured | 60/60 measured |

当前距离目标:

| precision | metric | missing |
|---|---|---:|
| FP16 | energy | 58 |
| FP16 | AP70 | 60 |
| INT8 | AP70 | 60 |

如果下一阶段选择重跑 FP16 energy 全量 60 行来替换 smoke 的 2 行, 也可以; 但最终 canonical 表必须显示 FP16 energy 60/60 measured。

## 5. 遇到问题时的处理规则

不要因为一个 label 失败就停止整批。标准处理:

1. 保存失败 label、命令、GPU id、stdout/stderr、raw artifact、returncode。
2. 先判断是环境问题、模型/ONNX 问题、TVM lowering/build 问题、runtime 问题、telemetry 问题还是 eval pipeline 问题。
3. 对单个 label 做最小复现, 修复后只重跑失败 label。
4. 如果同一问题连续修复失败, 写出 blocker artifact 和精确 root cause, 再决定 quarantine/no_claim。
5. 只有在有明确 root cause 且当前会话无法解决时, 才允许把该 label 标记为 no_claim/blocker; 不能无解释停止。

## 6. 下一轮 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 补点收口。不要启动 agent team, 单 agent 执行即可。

先阅读:
- multi_agent/methods/design/auto-tuning/progress/6_27/14_6_28_交接文档_阶段三_FP16Latency60收口与EnergyAP全量计划.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.md
- multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl

当前已完成:
- native INT8 full-ONNX original60 latency 60/60 measured。
- native INT8 full-ONNX original60 energy 60/60 measured。
- true FP16 original60 latency 60/60 measured。
- canonical latency/energy/AP 三表已刷新。
- completion review/gap 已刷新。
- 当前 latency 统一为 H800 TVM backbone/subnet module latency, 单位 ms, full_network_claim=false; 不得写成完整网络端到端或 RSU 物理设备绝对速度。

下一阶段唯一目标:
完成 existing original60 的 FP16/INT8 两种量化方式三指标全量收口, 使 canonical/review 显示:
- FP16 latency 60/60 measured, energy 60/60 measured, AP70 60/60 measured。
- INT8 latency 60/60 measured, energy 60/60 measured, AP70 60/60 measured。

具体执行:
1. 产品化 FP16 original60 energy runner, 使用 H800 power telemetry 和 idle-subtracted 方法, 输出 rows/fp16_true_original60_energy_rows_v1.jsonl 与 raw/fp16_true_energy_original60/<run_id>/。
2. 更新 canonical generator, 增加 --fp16-ap-rows 和 --int8-ap-rows, 并用单元测试证明 compliant AP row 能进入 AP 表, predicted/interpolated/TRT-reference AP row 会被拒绝。
3. 恢复或实现 FP16 AP eval/import route, 输出 rows/fp16_true_original60_ap_rows_v1.jsonl, 每行带 dataset split、ckpt、eval command、AP30/AP50/AP70、raw eval output。
4. 恢复或实现 native INT8 AP eval/import route, 输出 rows/native_int8_full_onnx_original60_ap_rows_v1.jsonl, 每行带同等证据。
5. 每完成一个批次立即刷新 original60_quant_three_metric_summary_latest.* 和 fp16_int8_original60_completion_review/gap。
6. 遇到任何失败不要直接停止: 先做 root cause 反思、日志审查、最小复现、修复并重跑失败 label; 只有确认当前无法解决时才写 blocker/no_claim, 且 blocker 必须有 stdout/stderr、命令、raw artifact 和明确原因。

最终交付:
- rows/fp16_true_original60_energy_rows_v1.jsonl = 60 measured rows
- rows/fp16_true_original60_ap_rows_v1.jsonl = 60 measured rows
- rows/native_int8_full_onnx_original60_ap_rows_v1.jsonl = 60 measured rows
- exports/original60_quant_three_metric_summary_latest.md 显示 FP16/INT8 三指标均 60/60 measured
- exports/fp16_int8_original60_completion_review_latest.md 无 FP16/INT8 no_claim
- 新交接文档记录所有 run_id、raw artifact、失败修复和最终校验
```
