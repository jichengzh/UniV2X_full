# 16_6_28_交接文档_阶段三_APIngestion完成与TrueEval收口计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/15_6_28_交接文档_阶段三_FP16INT8Energy收口与AP全量补点计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.md`

## 0. 口径确认

### 0.1 INT8 backbone/subnet 是否已经打通

是, 但声明范围必须保持为当前 Stage2 的 `backbone/subnet` route。

当前可声明的 INT8 measured route:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
engine_kind = tvm_graph_executor
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

它已经不是旧的 QDQ-heavy / float32-heavy lowering。Original60 的 native INT8 latency 和 energy 均已有 60/60 H800 TVM measured rows。

不能扩大的声明:

```text
不是 full perception network end-to-end
不是 RSU 物理边缘设备实测
不是 AP/eval route 已完成
```

### 0.2 当前 latency 是否可理解为 RSU 边缘段设备推理速度

不能直接这样理解。

当前总表中的 latency 统一为 `ms`, 计算口径为:

```text
latency_ms = latency_p50_us / 1000
```

它表示 H800 TVM 上对应 `backbone/subnet module` 的模块端到端推理时间。这里的端到端是指该模块输入到模块输出, 不是完整感知网络、不是完整 V2X pipeline, 也不是 RSU 物理边缘设备实测。

如果论文或报告中需要映射到 RSU 边缘段, 推荐写成:

```text
H800 TVM backbone/subnet module latency, used as a server-side edge-segment proxy.
```

不能写成:

```text
真实 RSU 边缘设备 latency
```

除非后续在真实 RSU 硬件或已定义等价的 RSU backend 上补测并建立等价依据。

## 1. 本轮新增工程状态

### 1.1 AP ingestion 已补齐

`scripts/stage2_generate_original60_quant_state_coverage.py` 已新增合规 AP row ingestion:

```text
--fp16-ap-rows
--int8-ap-rows
```

新增合规 gate:

```text
is_compliant_original60_ap_row(row, precision)
measured_ap_index(paths, precision)
```

合规 AP measured row 必须满足:

```text
precision / quant_policy 匹配 fp16 或 int8
measurement_status = measured
backend = model_eval
metric = AP70
metric_value 非空
measurement_source = true_eval
full_network_claim = false
dataset / eval_split / ckpt_path / ckpt_digest / eval_command / raw_artifact 非空
source_files 非空
secondary_metrics 包含 AP30 和 AP50
```

明确拒绝以下 AP 来源:

```text
predicted
model_fit / model-fit
interpolated
trt
simulated
```

FP16 AP 额外要求:

```text
quant_method = h800_tvm_true_fp16_onnx_relax
quality/source/layer/eval_command 中必须有 true_fp16 或 float16 证据
```

INT8 AP 额外要求:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
```

### 1.2 新增测试

新增测试位于:

```text
framework/tests/test_stage2_lut_productization.py
```

覆盖内容:

```text
合规 FP16 AP row 可进入 canonical AP 表
合规 native INT8 AP row 可进入 canonical AP 表
TRT / predicted / model-fit AP row 保持 no_claim
no_claim AP row 不再被误写为 imported
```

验证命令:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_lut_productization.Stage2Original60QuantCoverageCliTest framework.tests.test_stage2_original60_quant_completion
python -m py_compile scripts/stage2_generate_original60_quant_state_coverage.py framework/tests/test_stage2_lut_productization.py scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

验证结果:

```text
Ran 16 tests in 2.606s
OK
py_compile OK
```

## 2. 当前 canonical 状态

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

FP16 + INT8 completion review:

| axis | measured | no_claim | total |
|---|---:|---:|---:|
| latency | 120 | 0 | 120 |
| energy | 120 | 0 | 120 |
| AP70 | 0 | 120 | 120 |

全三精度 summary:

| axis | measured | no_claim | total |
|---|---:|---:|---:|
| latency | 178 | 2 | 180 |
| energy | 120 | 60 | 180 |
| AP70 | 0 | 180 | 180 |

解释:

- FP16/INT8 latency 已完成 120/120 measured。
- FP16/INT8 energy 已完成 120/120 measured。
- FP16/INT8 AP 仍是 0/120 measured。
- 全表中的 FP32 energy 和 FP32 AP 缺口不属于下一阶段 FP16/INT8 收口范围。

## 3. 数据位置

### 3.1 FP16 latency

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl
```

### 3.2 FP16 energy

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_true_smoke/20260628_fp16_true_original60_energy_batch001_60labels/
```

### 3.3 INT8 latency

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
```

### 3.4 INT8 energy

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch001/
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_full_onnx_original60_batch002_retry2_57labels/
```

### 3.5 FP16/INT8 AP

当前没有合规 measured AP row。

目标产物位置:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/
```

现有历史 AP / TRT AP / predicted AP / model-fit AP 不能作为 measured AP 导入。

## 4. 下一阶段唯一目标

下一阶段不再把 energy 作为主要缺口; energy 只作为验收项复核。唯一核心目标是产出并导入 FP16/INT8 original60 的真实 AP eval 数据。

必须收口到以下具体状态:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 60/60 measured |
| INT8 | 60/60 measured | 60/60 measured | 60/60 measured |

总验收:

```text
FP16/INT8 latency = 120/120 measured
FP16/INT8 energy = 120/120 measured
FP16/INT8 AP70 = 120/120 measured
jobs_requiring_action = 0
```

如果某一路由确实不可运行, 也不能直接停止。必须给出逐点 blocker artifact:

```text
失败 label
失败命令
stdout / stderr
环境摘要
raw artifact 路径
已尝试的修复
最小复现
下一步可执行修复方案
```

## 5. 下一阶段执行计划

### 5.1 FP16 true AP eval

目标:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
60 rows
```

每行必须包含:

```text
precision = fp16
quant_method = h800_tvm_true_fp16_onnx_relax
backend = model_eval
measurement_status = measured
measurement_source = true_eval
dataset / eval_split
ckpt_path / ckpt_digest
eval_command
raw_artifact
source_files
AP30 / AP50 / AP70
full_network_claim = false
```

注意: AP eval 可能需要完整模型路径, 而当前 latency/energy 是 backbone/subnet module。若 eval route 不是同一模块, 必须在 row 中把 eval route scope 写清楚, 不得把 backbone latency scope 扩大为 full network latency claim。

### 5.2 Native INT8 AP eval

目标:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
60 rows
```

每行必须包含:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
backend = model_eval
measurement_status = measured
measurement_source = true_eval
dataset / eval_split
ckpt_path / ckpt_digest
eval_command
raw_artifact
source_files
AP30 / AP50 / AP70
full_network_claim = false
```

如果 native INT8 artifact 不能直接接入 eval, 下一阶段必须做接口修复和最小复现, 不能停在 `tvm_int8_ap_eval_backend_missing`。

### 5.3 导入 AP rows 并刷新总表

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

刷新后必须复核:

```text
exports/original60_quant_three_metric_summary_latest.md
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_gap_report_latest.md
jobs/fp16_int8_original60_completion_queue_v1.jsonl
```

## 6. 失败处理要求

下一阶段只使用单 agent 执行实验、修复和审查闭环。

遇到任何 build / eval / import 失败, 按以下顺序推进:

```text
1. 保存失败命令、stdout、stderr、环境摘要和 artifact 路径
2. 判断失败属于代码 bug、数据缺口、接口缺口、环境缺口还是理论不可行
3. 对代码 bug 或接口缺口做最小复现
4. 写或补测试, 修复最小复现
5. 单点复测通过后再批量补跑失败 label
6. 刷新 canonical/review/gap/queue
7. 只有外部依赖缺失、权限不可恢复、数据源不存在等当前环境确实无法解决的问题, 才写 blocker
```

## 7. 下一阶段 /goal 命令

```text
/goal 继续在 ${V2X_ROOT} 执行 Stage2 original60 FP16/INT8 AP true-eval 全量收口。不要启动 agent team, 仅使用单 agent 完成实验、修复、审查和导入闭环。当前已完成 FP16/INT8 latency 120/120 measured、energy 120/120 measured; AP 仍为 0/120 measured。已完成 scripts/stage2_generate_original60_quant_state_coverage.py 的合规 AP row ingestion, 支持 --fp16-ap-rows 和 --int8-ap-rows, 并已用测试覆盖合规导入与 TRT/predicted/model-fit 拒绝。下一阶段目标是: 1) 产出 rows/fp16_true_original60_ap_rows_v1.jsonl 和 rows/native_int8_original60_ap_rows_v1.jsonl, 每种 precision 60 行、共 120 行, 每行包含 dataset split、checkpoint/digest、eval command、raw eval output、AP30/AP50/AP70、precision route、full_network_claim=false; 2) 用已新增 AP ingestion 刷新 original60_quant_20260627 的 summary/review/gap/queue, 验收 FP16/INT8 latency=120/120 measured、energy=120/120 measured、AP=120/120 measured、jobs_requiring_action=0; 3) 遇到任何 build/eval/import 问题, 先保存 blocker artifact 并进行反思、审查、最小复现、修复和失败 label 补跑, 不允许直接停止, 除非证明当前环境中确实不可解决。
```
