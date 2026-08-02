# 18_6_28_交接文档_阶段三_FP16INT8_EnergyAP补点收口执行计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/17_6_28_交接文档_阶段三_APTrueEval队列与单点Adapter计划.md`
- `multi_agent/methods/design/auto-tuning/progress/6_27/16_6_28_交接文档_阶段三_APIngestion完成与TrueEval收口计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`

## 0. 对两个关键问题的确认

### 0.1 INT8 backbone 实现是否已经打通

结论: 已打通, 但声明范围必须限定为 Stage2 当前的 `backbone/subnet` native INT8 route。

当前可声明的 INT8 measured route:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
engine_kind = tvm_graph_executor
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

当前证据:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60 rows
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 rows
exports/native_int8_full_onnx_audit_latest.md
```

这一路线已经不是旧的 QDQ-heavy / float32-heavy lowering。最新 native INT8 full ONNX topology audit 中, dtype inventory 记录 `float32=0`, `quantize=0`, `dequantize=0`, 且 latency/energy 均已在 H800 TVM route 上产生 measured rows。

仍然不能扩大声明为:

```text
完整感知全网络 end-to-end 已打通
AP/eval route 已经完成
真实 RSU 物理边缘设备实测
```

### 0.2 当前 latency 是否是基于 backbone 的端到端推理速度

结论: 是, 但这里的端到端是 `backbone/subnet module` 输入到输出的模块端到端, 不是完整 V2X 感知 pipeline 端到端。

当前 canonical latency 表的共同口径:

```text
full_network_claim = false
optimized_scope = backbone_only
latency_ms = latency_p50_us / 1000
```

因此可以写成:

```text
H800 TVM backbone/subnet module latency in ms.
```

如果需要对应 RSU 边缘段, 建议谨慎写成:

```text
作为 RSU edge-segment backbone/subnet 推理速度的 server-side proxy。
```

不能直接写成:

```text
真实 RSU 物理设备端到端推理速度
完整感知网络端到端 latency
```

除非后续在真实 RSU 硬件或已定义等价的 RSU backend 上补测, 并给出硬件映射依据。

## 1. 当前权威状态

权威数据根目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/
```

FP16 + INT8 当前状态:

| axis | FP16 | INT8 | 合计 |
|---|---:|---:|---:|
| latency | 60/60 measured | 60/60 measured | 120/120 measured |
| energy | 60/60 measured | 60/60 measured | 120/120 measured |
| AP70 | 0/60 measured | 0/60 measured | 0/120 measured |

全三精度 canonical summary:

| axis | measured | no_claim | total |
|---|---:|---:|---:|
| latency | 178 | 2 | 180 |
| energy | 120 | 60 | 180 |
| AP70 | 0 | 180 | 180 |

解释:

- FP16/INT8 latency 已经完成 120/120 measured。
- FP16/INT8 energy 已经完成 120/120 measured。
- FP16/INT8 AP 仍然是 0/120 measured, 这是下一阶段核心缺口。
- FP32 遗留缺口不属于本阶段 FP16/INT8 收口目标。

## 2. 数据位置

### 2.1 FP16 latency

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl
```

### 2.2 FP16 energy

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260628_fp16_true_original60_energy_batch001_60labels/
```

### 2.3 INT8 latency

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
```

### 2.4 INT8 energy

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch002_retry2_57labels/
```

### 2.5 FP16/INT8 AP 目标位置

当前这两个文件还没有合规 measured rows:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

目标 raw artifact 根目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/
```

历史 TRT / hybrid / predicted / model-fit / interpolated AP source 不能作为本阶段 measured AP 导入。

## 3. 下一阶段具体目标

下一阶段目标不是泛泛地继续补点, 而是把现有 original60 的 FP16 和 INT8 两条量化路线收口到可审阅的三指标状态。

最终验收状态:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 60/60 measured |
| INT8 | 60/60 measured | 60/60 measured | 60/60 measured |

具体新增产出:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl = 60 measured rows
rows/native_int8_original60_ap_rows_v1.jsonl = 60 measured rows
```

复核产出:

```text
rows/fp16_true_original60_energy_rows_v1.jsonl = 60 measured rows
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 measured rows
```

刷新后的总体验收:

```text
FP16/INT8 latency = 120/120 measured
FP16/INT8 energy = 120/120 measured
FP16/INT8 AP70 = 120/120 measured
jobs_requiring_action = 0
```

## 4. 执行路线

### 4.1 单点先行

先做 `s0_024` 单点, 不直接批量跑 120 个 AP jobs。

FP16 单点验收:

```text
precision = fp16
quant_method = h800_tvm_true_fp16_onnx_relax
backend = model_eval
measurement_source = true_eval
metric = AP70
secondary_metrics includes AP30/AP50
dataset / eval_split / ckpt_path / ckpt_digest / eval_command / raw_artifact / source_files complete
full_network_claim = false
```

INT8 单点验收:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
backend = model_eval
measurement_source = true_eval
metric = AP70
secondary_metrics includes AP30/AP50
dataset / eval_split / ckpt_path / ckpt_digest / eval_command / raw_artifact / source_files complete
full_network_claim = false
```

### 4.2 AP runner 与失败日志

已有准备:

```text
scripts/stage2_h800_true_fp16_ap_eval.py
```

当前本地 row-conversion 单测和 py_compile 已通过, 但 H800 AP smoke 尚未形成可导入 measured row。最近一次 H800 SSH 重试被远端断开, 因此下一阶段第一步是补齐 `s0_024` FP16 smoke 的稳定运行日志或 `ap_eval_blocker.json`, 不能把未落盘失败当成结论。

必须落盘:

```text
runner_command.json
gpu_preflight.json
ap_eval_report.json 或 ap_eval_blocker.json
layer_precision_summary.json
stdout/stderr 或等价 traceback
```

### 4.3 单点通过后扩展到 original60

单点通过后, 依次扩展:

```text
FP16 AP: s0_024 -> 60 labels
INT8 AP: s0_024 -> 60 labels
```

批量执行时不能因为某个 label 失败而停止整个队列。失败 label 写 blocker, 其余 label 继续跑; 批后回收失败 label, 做最小复现和修复。

### 4.4 导入并刷新 canonical

AP rows 生成后执行:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py \
  --fp32-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_latency_smoke_rows_v1.jsonl \
  --fp32-latency-remap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_latency_original60_remapped_rows_v1.jsonl \
  --fp16-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_latency_smoke_rows_v1.jsonl \
  --fp16-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl \
  --fp16-energy-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_ap_energy_smoke_rows_v1.jsonl \
  --fp16-energy-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl \
  --int8-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_latency_smoke_rows_v1.jsonl \
  --int8-energy-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_ap_energy_smoke_rows_v1.jsonl \
  --native-int8-full-onnx-latency-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl \
  --native-int8-full-onnx-energy-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl \
  --fp16-ap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl \
  --int8-ap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl

PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

必须复核:

```text
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/original60_quant_three_metric_summary_latest.csv
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

## 5. 问题处理要求

下一阶段只使用单 agent 执行实验、修复、自审和导入闭环, 不启动 agent team。

遇到任何 build / eval / import 失败, 不允许直接停止。必须按以下顺序处理:

```text
1. 保存失败命令、stdout/stderr、traceback、环境摘要、GPU 状态和 raw artifact 路径
2. 判断失败属于代码 bug、数据缺口、接口缺口、环境缺口还是当前理论不可行
3. 对代码 bug 或接口缺口做最小复现
4. 补测试或补验证脚本
5. 修复后先复测单点
6. 单点通过后继续批量补跑失败 label
7. 刷新 canonical/review/gap/queue
8. 只有权限不可恢复、数据源不存在、环境依赖无法安装等当前环境确实无法解决的问题, 才写最终 blocker
```

每个 blocker 必须包含:

```text
label
precision
failed command
stdout/stderr or traceback
raw artifact path
attempted fixes
minimal reproduction
next executable fix
```

## 6. 下一阶段 /goal 命令

```text
/goal 继续在 ${V2X_ROOT} 执行 Stage2 original60 FP16/INT8 量化补点收口, 只使用单 agent, 不启动 agent team。当前权威状态: INT8 backbone/subnet native route 已打通, FP16/INT8 latency 已完成 120/120 measured, FP16/INT8 energy 已完成 120/120 measured, latency 口径是 H800 TVM backbone/subnet module 端到端推理时间且 full_network_claim=false; AP 仍为 0/120 measured。下一阶段具体目标: 1) 复核并保持 FP16/INT8 energy 各 60 行 measured, 必要时补跑缺失或异常 label; 2) 先完成 s0_024 FP16 true AP eval 与 native INT8 AP eval 单点闭环, 每个单点必须产出 ap_eval_report.json 或精确 ap_eval_blocker.json; 3) 单点通过后扩展到 original60 60 个配置, 产出 rows/fp16_true_original60_ap_rows_v1.jsonl=60 measured rows 和 rows/native_int8_original60_ap_rows_v1.jsonl=60 measured rows, 每行包含 AP30/AP50/AP70、dataset/split、checkpoint/digest、eval_command、raw_artifact、source_files、precision route、full_network_claim=false; 4) 用 scripts/stage2_generate_original60_quant_state_coverage.py 的 --fp16-ap-rows 和 --int8-ap-rows 导入 AP rows 并刷新 summary/review/gap/queue; 5) 最终验收 FP16/INT8 latency=120/120 measured、energy=120/120 measured、AP70=120/120 measured、jobs_requiring_action=0。遇到任何 build/eval/import 问题, 先保存 blocker artifact, 再进行反思、审查、最小复现、修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境确实无法解决。
```
