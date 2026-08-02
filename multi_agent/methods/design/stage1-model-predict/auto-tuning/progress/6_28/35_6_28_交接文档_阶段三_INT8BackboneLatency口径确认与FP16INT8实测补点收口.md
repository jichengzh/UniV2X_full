# 41_6_28_交接文档_阶段三_INT8BackboneLatency口径确认与FP16INT8实测补点收口

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `40_6_28_交接文档_阶段三_CalibratedINT8APSmoke非空与RowGate修复.md`
- `39_6_28_交接文档_阶段三_H800ReferenceRange实测与ScaleAwareGate通过.md`
- `33_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_EnergyAP大范围补点计划.md`

本轮核心目的: 回答两个口径问题, 并把下一阶段目标收口为一个可执行的单 agent 任务: 在 existing original60 的 60 个配置上, 完成 FP16 与 native INT8 两种量化方式的 energy/AP 实测补点与总表刷新。遇到失败不能直接停止; 必须保存证据、复现、反思、审查、修复和补跑, 只有当前环境确实无法恢复时才写 per-label blocker。

## 0. 两个口径问题的明确回答

### 0.1 INT8 backbone 实现是否已经打通

结论: 是, 但声明范围限定为 `H800 + TVM native INT8 backbone/subnet route build/run`。

当前可以声明:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
route_spec = full_onnx_topology_conv_relu_add_identity_v1
device = H800
runtime = TVM graph executor / TE route
full_network_claim = false
```

已经完成的 measured rows:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60 rows
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 rows
```

新增 AP bridge 进展:

```text
s0_024 calibrated native INT8 AP smoke 已产生非空预测
processed_samples = 1
pred_nonempty_count = 1
pred_total_count = 77
smoke_gate_passed = true
ap_row_allowed = false
ap_row_block_reason = full_eval_num_samples_1_lt_1789
```

这说明 native INT8 backbone/subnet 输出已经能接入后续 AP bridge 做 smoke 验证, 但还不能把 INT8 AP 写成 measured row。

当前不能声明:

```text
native INT8 AP measured 已完成
native INT8 full 1789-sample AP eval 已完成
完整 HEAL / V2X perception pipeline TVM INT8 已完成
真实 RSU 物理边缘设备 latency 已实测
```

### 0.2 当前所有配置的 latency 是否是 backbone 端到端推理速度

结论: 是, 但这里的端到端只指 `backbone/subnet compiled module` 的模块端到端, 不是完整 perception pipeline 端到端。

统一口径:

```text
latency_ms = latency_p50_us / 1000
measurement = H800 + TVM compiled backbone/subnet module runtime
input = spatial_features
output = multiscale backbone/subnet outputs
full_network_claim = false
```

可以使用的表述:

```text
H800 TVM backbone/subnet module latency in ms.
可以作为 RSU-side backbone/subnet workload 的 server-side proxy。
```

不能使用的表述:

```text
完整感知网络端到端 latency
包含 dataloader / encoder / head / NMS / postprocess / dataset eval 的 full pipeline latency
真实 RSU 物理边缘设备绝对 latency
```

注意: 当前 rows 底层字段仍保留 `latency_p50_us`; 对外总表、审阅表和论文/汇报材料必须统一展示换算后的 `latency_ms`。

## 1. 当前权威覆盖状态

本轮复核的 rows 文件行数:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 5/60 measured |
| native INT8 | 60/60 measured | 60/60 measured | 0/60 measured |

合计:

```text
latency measured = 120/120
energy measured = 120/120
AP measured = 5/120
jobs_requiring_action = 115
```

当前 `native_int8_original60_ap_rows_v1.jsonl` 仍不存在; 不得把 smoke、blocker、hybrid AP、FP16/FP32 AP 或 predicted AP 导入为 native INT8 measured AP。

权威 review 文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
```

权威 rows 文件:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl
rows/fp16_true_original60_energy_rows_v1.jsonl
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

## 2. 下一阶段硬目标

下一阶段最终目标收口为:

```text
configs = existing original60 60 labels
precisions = fp16, native_int8
latency = 120/120 measured, 对外统一 latency_ms
energy = 120/120 measured and audited
AP = 120/120 measured
full_network_claim = false for all backbone/subnet rows
jobs_requiring_action = 0, unless a per-label blocker proves unsolvable in current environment
```

更具体的补点目标:

| output | current | target | action |
|---|---:|---:|---|
| FP16 latency | 60/60 | 60/60 | 保持并统一展示 `latency_ms` |
| FP16 energy | 60/60 | 60/60 audited | 复核 raw artifact / digest, 异常单点补跑 |
| FP16 AP70 | 5/60 | 60/60 measured | 补 55 个 full true-eval rows |
| native INT8 latency | 60/60 | 60/60 | 保持并统一展示 `latency_ms` |
| native INT8 energy | 60/60 | 60/60 audited | 复核 native route + telemetry evidence, 异常单点补跑 |
| native INT8 AP70 | 0/60 | 60/60 measured | 从 calibrated smoke 到 full 1789-sample AP eval, 补 60 rows |

目标输出文件:

```text
rows/latency_original60_quant_rows_v1.jsonl
rows/energy_original60_quant_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_original60_ap_rows_v1.jsonl
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
```

## 3. 单 agent 执行策略

本阶段只使用一个执行 agent。该 agent 需要自己完成实验、审查、反思、修复和补跑, 不再启动两个 agent 或 reviewer team。

### 3.1 第一批: 状态审计与队列刷新

目标: 建立 `60 labels x 2 precisions` 的唯一执行视图, 不从零发明配置。

执行:

1. 从 `exports/fp16_int8_original60_completion_review_latest.json` 读取 `AP=no_claim` 的 115 个 cell。
2. 校验 latency 对外字段统一为 `latency_ms`, 底层 `latency_p50_us` 仅保留为 raw artifact 字段。
3. 校验 FP16/native INT8 energy rows 均有 raw artifact、digest、telemetry source、`full_network_claim=false`。
4. 对 artifact 缺失、digest 不一致或字段不合规的 energy row 做单 label rerun, 不大面积重跑已合规 rows。
5. 刷新:

```text
jobs/fp16_int8_original60_completion_queue_v1.jsonl
jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_gap_report_latest.md
```

### 3.2 第二批: FP16 AP 补 55 行

FP16 AP 当前缺口:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl = 5 rows
remaining = 55 labels
```

执行顺序:

1. 对 55 个 label 做 checkpoint recovery inventory, 覆盖 H800 当前目录、历史 manifest、备份目录、stage1/stage2 变体和别名命名。
2. 找到 checkpoint 后运行 true FP16 full AP eval, `num_samples=1789`。
3. FP16 runner 继续使用 `model_half` 或 `amp_fp16`, 但 post_process 前将 model outputs cast 到 FP32, 保持 HEAL postprocessor 的 FP32 anchor/postprocess math。
4. 每完成一批立即刷新 `rows/fp16_true_original60_ap_rows_v1.jsonl` 和 `rows/ap_original60_quant_rows_v1.jsonl`。
5. 对找不到 checkpoint 的 label, 写 per-label blocker artifact, 至少包含 `label`, `width`, `searched_paths`, `reason`, `next_candidate_paths`, `recovery_action`。

FP16 AP row 必须包含:

```text
label
precision = fp16
AP30 / AP50 / AP70
num_samples = 1789
checkpoint_path
checkpoint_digest
eval_config
raw_artifact
precision_evidence
full_network_claim = false
```

### 3.3 第三批: native INT8 AP 补 60 行

native INT8 AP 当前缺口:

```text
rows/native_int8_original60_ap_rows_v1.jsonl = missing / 0 rows
remaining = 60 labels
```

必须遵守的 gate:

```text
1-sample smoke 只能证明 bridge 非空, 不能入 AP row。
full AP row 必须 num_samples >= 1789。
full AP row 必须 processed_samples >= 1789。
full AP row 必须 AP30/AP50/AP70 有限。
ap_row_allowed 必须为 true。
```

优先从 `s0_024` 开始完成 full eval, 因为它已经具备 calibrated smoke 非空证据:

```text
route:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/

calibration:
reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json

smoke:
ap_smoke_calibrated_pyramid_level2_v2_rowgate/full_ap_eval_report.json
```

执行顺序:

1. 对每个 label 检查 native INT8 route 是否存在且和 checkpoint/width 一致。
2. 若缺 output calibration, 先执行 full multiscale reference range capture, 至少覆盖 `pyramid_level0/1/2`。
3. 执行 1-sample calibrated AP smoke, 要求 `pred_nonempty_count > 0` 且 `ap_row_allowed=false`。
4. smoke 通过后执行 full 1789-sample INT8 AP eval。
5. full eval 通过后才生成或追加:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
```

native INT8 AP row 必须包含:

```text
label
precision = native_int8
AP30 / AP50 / AP70
num_samples = 1789
native_int8_route_dir
native_int8_route_digest_or_manifest
tensor_quant_params_path
tensor_quant_params_digest
output_dequant_policy
raw_artifact
worker_request
worker_response_summary
postprocess_summary
full_network_claim = false
```

## 4. 失败时的强制处理闭环

遇到任何问题不能直接停止。每个失败必须进入以下闭环:

```text
1. 保存 runner_command、stdout、stderr、worker request/response、postprocess summary、eval report。
2. 标注失败类型: checkpoint_missing / route_missing / build_failed / calibration_missing / smoke_empty / full_eval_failed / metric_gate_failed / artifact_invalid。
3. 做最小复现, 优先单 label、单 batch、1-sample。
4. 反思根因, 明确是数据、checkpoint、route、quant scale、bridge、postprocess 还是总表 ingestion。
5. 修复代码或配置后补充 focused test。
6. 重跑同一 label 的 smoke 或 full eval。
7. 只有在当前环境无法恢复 checkpoint/route, 或 full eval 多次复现同一不可修复失败后, 才写 per-label blocker。
```

per-label blocker 不是停止理由, 而是继续推进其他 label 的同时保留可审查证据。最终允许剩余 blocker 的唯一条件是 blocker 能证明当前环境无法解决, 并给出下一步人工输入或数据恢复需求。

## 5. 下一步第一动作

第一动作应为 `s0_024` native INT8 full 1789-sample AP eval:

```text
label = s0_024
num_samples = 1789
full_ap_min_samples = 1789
ap_row_min_samples = 1789
tensor_quant_params_path = .../reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json
raw_dir = .../s0_024/ap_full_calibrated_pyramid_level2_v1/
```

若通过:

```text
生成 native INT8 s0_024 AP measured row
刷新 rows/native_int8_original60_ap_rows_v1.jsonl
刷新 rows/ap_original60_quant_rows_v1.jsonl
刷新 completion review / gap report
复制同一流程到 s0_040、s1_048, 再扩展到 remaining 57 labels
```

若失败:

```text
不得停止。
保存完整 raw artifact。
按第 4 节失败闭环定位 full eval 失败点。
修复后先重跑 1-sample smoke, 再重跑 1789-sample full eval。
```

## 6. /goal 命令

```text
/goal 阶段三收口: 单 agent 在 original60 现有 60 个配置上完成 FP16 与 native INT8 的 energy/AP 实测补点。保持 latency 统一使用 latency_ms; 当前 latency 口径固定为 H800+TVM backbone/subnet compiled module end-to-end, full_network_claim=false, 不声明完整 perception pipeline 或真实 RSU 物理设备 latency。目标是 FP16 latency/energy 60/60 保持并审计、native INT8 latency/energy 60/60 保持并审计、FP16 AP 从 5/60 补到 60/60、native INT8 AP 从 0/60 补到 60/60, 最终刷新 rows 与 exports 总表并使 jobs_requiring_action=0。INT8 AP 必须先通过 calibrated smoke, 再通过 1789-sample full eval; 禁止把 smoke/no-claim/blocker/hybrid/predicted AP 导入 measured row。遇到任何失败先保存 artifacts、复现、反思、审查、修复和补跑; 只有 checkpoint/route 在当前环境不可恢复或 full eval 形成可审查不可解 blocker 时才标记 blocked。
```

