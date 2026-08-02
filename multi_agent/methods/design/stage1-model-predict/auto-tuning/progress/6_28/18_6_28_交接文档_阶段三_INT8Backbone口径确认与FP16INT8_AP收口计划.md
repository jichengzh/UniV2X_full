# 24_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_AP收口计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `21_6_28_交接文档_阶段三_INT8Backbone口径澄清与FP16INT8补点收口计划.md`
- `22_6_28_交接文档_阶段三_INT8APAdapter根因诊断与数值Route计划.md`
- `23_6_28_交接文档_阶段三_INT8APShapeRealWeightRouteProbe成功与WorkerBridge计划.md`

本轮核心结论: INT8 的 backbone/subnet native route 已经打通, original60 的 FP16/INT8 latency 与 energy 也已经全量 measured; 但 AP 闭环还没有完成。下一阶段不要再停留在证明 INT8 backbone 能否 build, 而是要把 native INT8 backbone route 接到 HEAL AP eval, 同时补齐 FP16 AP。

## 0. 两个口径确认

### 0.1 INT8 backbone 实现是否已经打通

结论: 是, 但声明范围必须限定在 `backbone/subnet native INT8 route`。

当前可声明:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

已经完成的 original60 measured rows:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60 rows
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 rows
```

更接近 AP eval 边界的 `s0_024` probe 也已经 build/run 成功:

```text
run_id = 20260628_native_int8_apshape_s0_024_realweight_probe_v2
input_shape_override = spatial_features [1,64,256,256]
outputs =
  [1,24,256,256]
  [1,128,128,128]
  [1,256,64,64]
runtime_weights_int8.npz = 51 个 ONNX initializer quantized int8 weights
```

不能声明:

```text
完整感知 full pipeline INT8 end-to-end 已打通
INT8 AP measured 已完成
TVM INT8 route 已经直接替换 HEAL postprocess 全链路
```

### 0.2 目前所有 latency 是否都是基于 backbone 端到端推理

结论: 是, 但这里的端到端是 `backbone/subnet module` 的端到端, 不是完整感知 pipeline 的端到端。

当前 latency 共同口径:

```text
device = H800
runtime = TVM compiled module
scope = spatial_features -> backbone/subnet multiscale outputs
unit for review/export = latency_ms
full_network_claim = false
```

因此这些 latency 可以作为 `RSU edge-segment backbone/subnet 推理速度` 的 server-side proxy, 但不能写成真实 RSU 物理边缘设备实测速度, 也不能写成包含 dataloader/head/NMS/postprocess 的完整端到端感知速度。

## 1. 当前权威覆盖状态

权威 review:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
```

当前 summary:

```text
total_jobs = 120
precision_counts = {"fp16": 60, "int8": 60}
latency measured = 120/120
energy measured = 120/120
AP measured = 5/120
jobs_requiring_action = 115
```

按 precision 拆分:

| axis | FP16 | INT8 | 合计 |
|---|---:|---:|---:|
| latency | 60/60 measured | 60/60 measured | 120/120 measured |
| energy | 60/60 measured | 60/60 measured | 120/120 measured |
| AP70 | 5/60 measured | 0/60 measured | 5/120 measured |

## 2. 当前数据位置

FP16:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl = 60 rows
rows/fp16_true_original60_energy_rows_v1.jsonl = 60 rows
rows/fp16_true_original60_ap_rows_v1.jsonl = 5 rows
raw/ap_eval_original60/
raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

INT8:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60 rows
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 rows
rows/native_int8_original60_ap_rows_v1.jsonl = missing / 0 rows
raw/int8_native_route/
raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/
```

刷新目标表:

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

## 3. 下一阶段总目标

最终收口目标:

```text
original60 labels = 60
precisions = fp16, int8
metrics per label/precision = latency_ms, energy_J_per_inference, AP70
target cells = 120 config-precision cells
target measured metric rows =
  FP16 latency 60/60
  FP16 energy 60/60
  FP16 AP70 60/60
  INT8 latency 60/60
  INT8 energy 60/60
  INT8 AP70 60/60
```

当前 latency/energy 已达到 60/60 + 60/60, 下一阶段仍需做 raw artifact/digest/字段一致性复核; 如果复核发现某个 label 的 energy 或 latency artifact 不完整, 只补跑该 label, 不回退已经有充分证据的 rows。真正的大缺口是:

```text
FP16 AP70: 5/60 -> 60/60, 还缺 55 行
INT8 AP70: 0/60 -> 60/60, 还缺 60 行
```

## 4. 下一阶段执行计划

### 4.1 Energy 与 latency 复核

目标: 保持 `latency=120/120 measured`、`energy=120/120 measured`, 并确认所有对外 latency 单位都是 ms。

执行:

1. 校验四个 rows 文件各 60 行。
2. 校验每行都有 `label`、`precision`、`latency_ms` 或 `energy_J`、`raw_artifact`、digest/source evidence、`full_network_claim=false`。
3. 校验 canonical summary 没有把 raw `latency_p50_us` 当成最终审阅 latency。
4. 对缺 raw artifact 或 digest 不一致的 label 单点补跑, 写入新的 raw run 目录, 再刷新 latest summary/review。

### 4.2 FP16 AP 补齐

目标: `rows/fp16_true_original60_ap_rows_v1.jsonl = 60 measured rows`。

当前 blocker: 55 个 label 主要缺少可用 finetuned checkpoint, blocker 已落盘在:

```text
raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

执行:

1. 从 completion review 中筛出 FP16 `AP=no_claim` 的 55 个 label。
2. 继续做 checkpoint recovery inventory, 覆盖 H800 当前目录、历史 manifest、备份路径、训练产物目录和可能的别名命名。
3. 找到 checkpoint 的 label 立即运行 `scripts/stage2_h800_true_fp16_ap_eval.py`。
4. 每批完成后拉回 raw artifact 和 rows, 刷新 completion review。
5. 找不到 checkpoint 的 label 不停止整批任务, 写 per-label blocker, 包含 searched_paths、candidate_paths、failure_reason、next_recovery_action。

AP row 必须包含:

```text
AP30/AP50/AP70
num_samples
checkpoint path/digest
eval config
raw_artifact
precision evidence
full_network_claim=false
```

### 4.3 INT8 AP 补齐

目标: `rows/native_int8_original60_ap_rows_v1.jsonl = 60 measured rows`。

当前状态: `s0_024` AP-shape native INT8 backbone route 已经 build/run 成功, 但仍未接入真实 HEAL activation 和 postprocess。

先做单点 gate:

```text
label = s0_024
route = h800_tvm_native_int8_backbone_subnet
bridge = UniV2X torch process -> tvm310 worker -> PyTorch head/postprocess
acceptance = one-sample postprocess smoke 成功, 或写出精确 blocker
```

one-sample smoke 必须产出:

```text
native_int8_worker_request.json
native_int8_worker_response.json
activation_quant_summary.json
multiscale_output_summary.json
head_output_summary.json
postprocess_summary.json
```

单点 smoke 通过后:

1. 跑 `s0_024` 1789-frame full AP。
2. 成功后追加 `rows/native_int8_original60_ap_rows_v1.jsonl` 第一行 measured。
3. 将 bridge 参数化为 label/width/manifest/raw_dir。
4. 批量扩展 original60 其余 label。
5. 每批刷新 canonical AP rows、completion review 和 gap report。

INT8 AP row 必须包含:

```text
AP30/AP50/AP70
num_samples
native_int8_route_manifest
runtime_weight_archive_digest
activation quantization evidence
worker request/response summary
raw_artifact
full_network_claim=false
```

## 5. 失败处理规则

遇到任一配置失败, 不允许直接停止整个阶段。必须按以下顺序处理:

1. 保存 stdout/stderr、runner command、GPU id、raw artifact、failure reason。
2. 分类失败: missing checkpoint、missing artifact、TVM build failure、TVM runtime failure、shape mismatch、dtype mismatch、postprocess failure、AP import failure、SSH/env failure。
3. 构造最小复现, 只复现当前失败 label 或当前失败 op。
4. 审查当前结论是否把 FP32/FP16/INT8 口径混淆。
5. 修改 runner/job queue 后重试失败 label, 其他 label 继续推进。
6. 只有证明当前执行 agent 无法解决时, 才写 per-cell quarantine/no-claim blocker。

## 6. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 三指标收口, 单 agent 执行, 不启动 agent team。当前口径: INT8 backbone/subnet native route 已打通, 但 full_network_claim=false; latency 是 H800 TVM backbone/subnet module 的端到端推理时间, 对外统一 latency_ms, 可作为 RSU edge-segment backbone/subnet server-side proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理设备实测。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; total AP measured 5/120, jobs_requiring_action=115。下一阶段硬目标: 完成 original60 60 个配置 x {fp16,int8} 的 latency_ms、energy_J_per_inference、AP70 全量 measured; latency/energy 先做 raw artifact/digest/字段一致性复核, 发现缺口单点补跑; FP16 AP 从 5/60 补到 60/60; INT8 AP 从 0/60 补到 60/60。FP16 路线: 从 completion review 筛出 55 个 AP=no_claim label, 继续 H800 checkpoint recovery inventory, 找到 checkpoint 即运行 scripts/stage2_h800_true_fp16_ap_eval.py, 找不到 checkpoint 写 per-label blocker 并继续其他 label。INT8 路线: 不再重复证明 backbone buildability, 直接基于 20260628_native_int8_apshape_s0_024_realweight_probe_v2 实现 s0_024 TVM worker bridge, UniV2X torch 进程保存真实 HEAL spatial_features activation, tvm310 worker 加载 TVM .so + runtime_weights_int8.npz 运行 native INT8 backbone/subnet, 返回三层 multiscale feature, 再接 PyTorch head/postprocess; one-sample smoke 产出 worker_request/response、activation_quant_summary、multiscale_output_summary、head_output_summary、postprocess_summary, 通过后跑 s0_024 1789-frame full AP 并追加 rows/native_int8_original60_ap_rows_v1.jsonl 首行 measured, 再扩展到 original60 全量。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 必须保存 blocker artifact/stdout/stderr, 做失败分类、最小复现、反思审查、runner/job queue 修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
