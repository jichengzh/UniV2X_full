# 21_6_28_交接文档_阶段三_INT8Backbone口径澄清与FP16INT8补点收口计划

更新时间: 2026-06-28

本文件是当前最新交接入口。上一份 `20_6_28_交接文档_阶段三_FP16AP首批4点TrueEval导入.md` 中 AP 计数已过期: 当前 FP16 AP measured row 已从 4 行更新到 5 行。

## 0. 关键口径澄清

### 0.1 INT8 backbone 是否已经打通

结论: 已打通, 但声明范围必须限定为 Stage2 的 `backbone/subnet native INT8 route`。

当前可声明的 measured route:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
route_spec = full_onnx_topology_conv_relu_add_identity_v1
full_network_claim = false
```

当前 evidence:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60 rows
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60 rows
```

`s0_024` native INT8 manifest/inventory 示例:

```text
onnx_path = ${V2X_DATA_ROOT}/s2_tvm/models/s0_024_backbone.onnx
input_shape = spatial_features [2, 64, 128, 256]
output_shape =
  /resnet/layer0/layer0.2/relu_2/Relu_output_0 [2, 24, 128, 256]
  /resnet/layer1/layer1.4/relu_2/Relu_output_0 [2, 128, 64, 128]
  /resnet/layer2/layer2.7/relu_2/Relu_output_0 [2, 256, 32, 64]
dtype_inventory = dequantize 0, float32 0, int32 217, int8 301, quantize 0, uint8 250
op_counts = Conv 51, Relu 48, Add 16, Identity 46
```

因此它已经不是旧的 QDQ-heavy / float32-heavy lowering。下一阶段不能再把这个问题描述成 INT8 backbone 没有 build 出来, 而应该聚焦到: 如何把 native INT8 backbone/subnet route 接入 AP eval, 并完成 original60 全量 AP measured rows。

不能声明:

```text
完整感知全网络 end-to-end 已打通
完整 HEAL pipeline TVM INT8 已打通
INT8 AP measured 已完成
```

### 0.2 当前所有 latency 是否是 backbone 端到端推理

结论: 是, 但这里的端到端只指 `backbone/subnet module` 输入到输出的模块端到端, 不是完整 V2X 感知 pipeline 端到端。

当前 FP16/INT8 latency rows 的共同口径:

```text
full_network_claim = false
raw latency field = latency_p50_us
对外审阅字段 = latency_ms = latency_p50_us / 1000
```

FP16 latency:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl = 60 rows
quant_scope = backbone_only
full_network_claim = false for all 60 rows
```

INT8 latency:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60 rows
quant_scope = backbone_subnet_native_int8
full_network_claim = false for all 60 rows
```

后续所有对外总表和审阅表必须统一展示 `ms`, 不能把 raw `us` 字段直接作为最终 latency 指标展示。

可以使用的表述:

```text
H800 TVM backbone/subnet module latency in ms.
可以作为 RSU edge-segment backbone/subnet 推理速度的 server-side proxy。
```

不能使用的表述:

```text
真实 RSU 物理设备实测 latency
完整感知网络端到端 latency
包含 head / postprocess / dataloader / NMS 的 full pipeline latency
```

## 1. 当前权威覆盖状态

权威 review:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
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

注意: energy 当前已经是 FP16 60/60 + INT8 60/60 measured。下一阶段仍要复核 energy raw artifact 与汇总表一致性, 但核心缺口已经转移到 AP measured rows。

## 2. 当前 measured 数据位置

FP16 latency:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl
```

FP16 energy:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl
```

INT8 latency:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
```

INT8 energy:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

FP16 AP:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/
```

INT8 AP 目标文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

当前该 INT8 AP rows 文件尚不存在, 不能导入历史 no-claim / blocker AP 作为 measured。

## 3. 当前 FP16 AP measured rows

FP16 AP runner 当前策略: model/input 保持 `model_half` 或 `amp_fp16`, 但 model output 在 `dataset.post_process` 前 cast 到 FP32, 以匹配 HEAL postprocessor 的 anchor/postprocess FP32 math。该策略不改变 latency 口径, AP row 仍保持 `full_network_claim=false`。

| label | width | AP30 | AP50 | AP70 | num_samples | run_id |
|---|---:|---:|---:|---:|---:|---|
| s0_024 | [24,128,256] | 0.7962363235542628 | 0.7545584375996155 | 0.5961445248627925 | 1789 | 20260628_fp16_true_ap_s0_024_full_amp_fp16_v1 |
| s0_040 | [40,128,256] | 0.7876717273233904 | 0.7454286072525822 | 0.5897363386141050 | 1789 | 20260628_fp16_true_ap_s0_040_full_amp_fp16_v1 |
| s0_056 | [56,128,256] | 0.7951631598005301 | 0.7537173526202081 | 0.5870672005342719 | 1789 | 20260628_fp16_true_ap_s0_056_full_amp_fp16_v1 |
| s1_048 | [64,48,256] | 0.7860478968707385 | 0.7467321998477431 | 0.5992276271081318 | 1789 | 20260628_fp16_true_ap_s1_048_full_amp_fp16_v1 |
| s2_160 | [64,160,256] | 0.7927004774506324 | 0.7513990229771785 | 0.5925278912642500 | 1789 | 20260628_fp16_true_ap_s2_160_full_amp_fp16_v1 |

## 4. 已知缺口和不能绕开的 blocker

### 4.1 FP16 AP 剩余 55 行

H800 checkpoint inventory 已确认: 在当前可见 checkpoint 目录中, 原 56 个剩余 FP16 label 只有 `s2_160` 找到了可直接 full eval 的 checkpoint, 并已完成 measured row。剩余 55 个 label 主要 blocker 是缺少可用 finetuned checkpoint 目录。

下一阶段不能因为 checkpoint 缺失直接停止。必须按以下顺序处理:

1. 继续做 H800 checkpoint inventory, 包括历史命名、备份目录、stage1/stage2 变体和 manifest 中记录的路径。
2. 对找到 checkpoint 的 label 立即运行 FP16 true-eval 并写入 `fp16_true_original60_ap_rows_v1.jsonl`。
3. 对找不到 checkpoint 的 label 写出 per-label blocker artifact, 至少包含 label、width、searched_paths、reason、next_candidate_paths。
4. 若确认 checkpoint 不存在, 明确提出恢复路径: 从历史训练产物恢复、重新训练/finetune、或将该 label 标为需要人工提供 checkpoint。不能伪造 measured AP。

### 4.2 INT8 AP 剩余 60 行

当前 native INT8 artifact 输出的是 backbone multi-scale features, 不是 HEAL postprocessor 直接需要的 head outputs。HEAL AP path 需要 `cls_preds`、`reg_preds`、可选 `dir_preds` 进入 `dataset.post_process`。

因此 INT8 AP adapter 的正确目标是:

```text
TVM native INT8 backbone/subnet -> PyTorch shrink/head -> HEAL post_process -> AP30/AP50/AP70
```

或者进一步 build 出包含 head 的 INT8 route。无论选哪条路线, 都必须先做 `s0_024` 单点闭环, 产出 `ap_eval_report.json` 或精确 `ap_eval_blocker.json`。不能把历史 FP32/FP16 AP、hybrid AP、predicted AP、no-claim AP 导入为 INT8 measured AP。

## 5. 下一阶段具体计划

### 5.1 保持 latency/energy 已完成状态

目标:

```text
FP16/INT8 latency = 120/120 measured
FP16/INT8 energy = 120/120 measured
```

执行要求:

1. 复核四个 rows 文件均为 60 行, 且 `full_network_claim=false`。
2. 对外总表统一展示 `latency_ms`, 保留 raw `latency_p50_us` 只作为底层 artifact 字段。
3. 如果发现某个 label raw artifact 缺失或 digest 不一致, 补跑对应 label, 不直接删除 measured row。

### 5.2 FP16 AP 从 5/60 补到 60/60

执行顺序:

1. 用当前 queue 找出 FP16 `ap_status=no_claim` 的 55 个 label。
2. 在 H800 上做 checkpoint recovery inventory。
3. 对 checkpoint 存在的 label 批量运行 `scripts/stage2_h800_true_fp16_ap_eval.py`。
4. 每一批结束后 rsync raw artifact 和 rows 回本地。
5. 刷新 review, 确认 FP16 AP measured count 增长。
6. 对 checkpoint 缺失 label 写 blocker artifact, 并继续处理其他 label。

验收:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl = 60 measured rows
每行包含 AP30/AP50/AP70、num_samples、checkpoint/digest、eval config、raw_artifact、full_network_claim=false
```

### 5.3 INT8 AP 从 0/60 补到 60/60

单点 gate:

```text
label = s0_024
route = h800_tvm_native_int8_backbone_subnet
adapter = TVM INT8 backbone/subnet output -> PyTorch head/postprocess
required output = ap_eval_report.json or ap_eval_blocker.json
```

单点通过后批量扩展:

1. 将 adapter 参数化为 label/width/manifest/raw_dir。
2. 逐 label 读取 native INT8 route manifest 和 TVM artifact。
3. 运行 AP eval, 写入 `rows/native_int8_original60_ap_rows_v1.jsonl`。
4. 对 shape mismatch、dtype mismatch、missing artifact、postprocess failure 分别写 blocker artifact。
5. 每批刷新 canonical/review/queue。

验收:

```text
rows/native_int8_original60_ap_rows_v1.jsonl = 60 measured rows
每行包含 AP30/AP50/AP70、num_samples、native_int8_manifest、adapter route、raw_artifact、full_network_claim=false
```

### 5.4 每批刷新总表

AP rows 发生变化后执行:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py \
  --fp16-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl \
  --fp16-energy-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl \
  --native-int8-full-onnx-latency-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl \
  --native-int8-full-onnx-energy-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl \
  --fp16-ap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl \
  --int8-ap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

随后刷新 completion queue:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

## 6. 最终收口目标

最终目标必须具体到以下状态:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 60/60 measured |
| INT8 | 60/60 measured | 60/60 measured | 60/60 measured |

最终 review 验收:

```text
total_jobs = 120
jobs_requiring_action = 0
latency measured = 120/120
energy measured = 120/120
AP measured = 120/120
所有对外 latency 指标使用 ms
所有 measured rows 均有 raw artifact 和 source evidence
```

遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 不能直接停止。必须先保存 blocker artifact 和 stdout/stderr, 然后进行反思、审查、最小复现、修复和失败 label 补跑。只有在证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决时, 才能把该 label 标记为需要外部输入。

## 7. 下一阶段 /goal 命令

```text
/goal 继续在 ${V2X_ROOT} 执行 Stage2 original60 FP16/INT8 量化补点收口, 单 agent 执行。当前权威状态: INT8 backbone/subnet native route 已打通, 但范围仅限 backbone/subnet, full_network_claim=false; 当前所有 latency 是 H800 TVM backbone/subnet module 端到端推理时间, 对外统一使用 latency_ms, 可作为 RSU edge-segment backbone/subnet 推理速度 proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理设备端到端 latency。当前覆盖: FP16/INT8 latency=120/120 measured, energy=120/120 measured, AP=5/120 measured, jobs_requiring_action=115。下一阶段具体目标: 1) 复核并保持 FP16/INT8 latency 和 energy 各 60 行 measured, 确保总表 latency 全部用 ms; 2) 将 FP16 AP 从 5/60 补到 60/60, 对剩余 label 先做 H800 checkpoint inventory/recovery, 有 checkpoint 就跑 scripts/stage2_h800_true_fp16_ap_eval.py, 无 checkpoint 则写 per-label blocker artifact 并继续其他 label; 3) 将 INT8 AP 从 0/60 补到 60/60, 先完成 s0_024 native INT8 AP adapter 单点闭环, route 为 TVM native INT8 backbone/subnet -> PyTorch head/postprocess -> AP30/AP50/AP70, 单点必须产出 ap_eval_report.json 或精确 ap_eval_blocker.json, 通过后扩展 original60; 4) 每批 AP rows 后刷新 scripts/stage2_generate_original60_quant_state_coverage.py 和 scripts/stage2_generate_fp16_int8_original60_completion_queue.py, 更新 canonical summary/review/gap/queue; 5) 最终验收 rows/fp16_true_original60_ap_rows_v1.jsonl=60 measured rows, rows/native_int8_original60_ap_rows_v1.jsonl=60 measured rows, FP16/INT8 latency=120/120 measured, energy=120/120 measured, AP=120/120 measured, jobs_requiring_action=0。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 先保存 blocker artifact/stdout/stderr, 再反思、审查、最小复现、修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
