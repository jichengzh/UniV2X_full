# 17_6_28_交接文档_阶段三_APTrueEval队列与单点Adapter计划

日期: 2026-06-28

继承文档:

- `multi_agent/methods/design/auto-tuning/progress/6_27/16_6_28_交接文档_阶段三_APIngestion完成与TrueEval收口计划.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_ap_true_eval_source_audit_latest.md`

## 0. 当前权威状态

旧的 9/10 号交接文档记录的是早期三点或两点状态, 不能再作为当前覆盖率依据。当前权威状态以 `original60_quant_20260627` 的 latest rows/exports 为准。

FP16 + INT8 original60 当前覆盖:

| axis | measured | no_claim / blocked | total |
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

重要口径:

- INT8 native route 已对 original60 完成 60/60 latency 和 60/60 energy。
- latency 是 H800 TVM `backbone/subnet module` 的模块端到端推理时间, 对外单位为 `ms`。
- 这些 latency/energy row 均保持 `full_network_claim=false`。
- AP 不能由 latency/energy 推断, 也不能使用 TRT/hybrid/predicted/model-fit/interpolated source 冒充 measured。

## 1. 本轮新增产物

### 1.1 AP true-eval queue 生成脚本

新增:

```text
scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
```

作用:

1. 读取 `jobs/fp16_int8_original60_completion_queue_v1.jsonl`。
2. 只筛选 FP16/INT8 中 `ap_status != measured` 的 cell。
3. 审计默认历史 AP sources 和显式传入的 AP source rows。
4. 使用与 canonical AP ingestion 对齐的合规规则判定是否可导入。
5. 输出 AP true-eval queue、source audit 和 blocker rows。

它不会写 AP measured row, 也不会把历史 AP 强行导入 canonical。

### 1.2 新增输出

实际执行命令:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_ap_true_eval_queue.py \
  --output-root multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627 \
  --created-at 2026-06-28T00:00:00Z
```

输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_ap_true_eval_source_audit_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_ap_true_eval_source_audit_latest.md
```

实际结果:

```text
AP true-eval queue rows = 120
ready_for_import = 0
blocked = 120
blocker rows = 120
FP16 blockers = 60 x no_compliant_true_fp16_model_eval_source
INT8 blockers = 60 x no_compliant_native_int8_model_eval_source
```

默认审计 source rows:

```text
FP16 source rows inspected = 75
INT8 source rows inspected = 4
```

结论: 当前没有可直接导入的合规 FP16/INT8 AP measured source。

### 1.3 H800 AP 环境 preflight

本轮对 H800 做了只读实时 preflight, 输出:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/h800_ap_env_preflight_20260628_true_eval_stdout.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/h800_ap_env_preflight_20260628_true_eval_latest.json
```

解析结果:

```text
hostname = <PRIVATE_HOST>
status = pass
blocked_reasons = []
env_python exists = true
dataset_val exists = true
base_ckpt exists = true
```

本次 preflight 使用 `--skip-gpu`, 因此它证明 AP 软件环境、数据集和 checkpoint 可见, 不证明 GPU 当前空闲。AP eval 不要求 GPU 完全空闲, 但正式跑 eval 时仍必须记录 GPU 状态。

## 2. 根因判断

当前 AP 缺口不是 H800 访问问题, 也不是 DAIR-V2X 数据集或 base checkpoint 缺失。

更具体的根因是:

```text
repo 中已有 AP runner 主要是 TRT/hybrid chain:
structural_prune_pyramid -> train_ddp --half -> ONNX export -> TRT FP16 build -> hybrid AP eval
```

该链路在历史阶段可作为真实 eval source, 但不满足本阶段的合规要求:

- FP16 AP 需要 `quant_method=h800_tvm_true_fp16_onnx_relax` 或明确 true-FP16 非 TRT eval/import 证据。
- INT8 AP 需要 `quant_method=h800_tvm_native_int8_backbone_subnet`, `quant_scope=backbone_subnet_native_int8`, 并能说明 native INT8 backbone/subnet 如何接入 eval pipeline。
- 当前 historical AP source 中出现 TRT/hybrid、predicted/model-fit/interpolated 相关 token 或缺少 true-FP16/native-INT8 route marker, 因此全部保持 no-claim/blocker。

## 3. 下一步最小闭环

下一阶段不要直接批量跑 120 个 AP jobs。必须先做单点 adapter, 单点通过后再扩到 original60。

### 3.1 FP16 单点 adapter

第一目标 label:

```text
s0_024
```

目标:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl 至少新增 s0_024 1 行 measured
```

要求:

```text
precision = fp16
quant_method = h800_tvm_true_fp16_onnx_relax
backend = model_eval
measurement_source = true_eval
metric = AP70
secondary_metrics 包含 AP30/AP50
dataset / eval_split / ckpt_path / ckpt_digest / eval_command / raw_artifact / source_files 完整
full_network_claim = false
```

建议实现路线:

1. 以 HEAL `opencood/tools/inference.py` 为参考, 新增 Stage2 wrapper, 产出 JSON AP report。
2. 加入明确 true-FP16 mode: model/input/runner 的 dtype evidence 写入 `layer_precision_summary.json` 或等价 raw evidence。
3. 不经过 TRT engine, 不写 `.engine` 作为 source evidence。
4. 单点输出 `raw/ap_eval_original60/fp16_true_s0_024/`。
5. 用 `--fp16-ap-rows rows/fp16_true_original60_ap_rows_v1.jsonl` 刷新 canonical, 验证 AP 从 0 变成 1。

### 3.2 Native INT8 单点 adapter

第一目标 label:

```text
s0_024
```

目标:

```text
rows/native_int8_original60_ap_rows_v1.jsonl 至少新增 s0_024 1 行 measured 或精确接口 blocker
```

要求:

```text
precision = int8
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
backend = model_eval
measurement_source = true_eval
metric = AP70
secondary_metrics 包含 AP30/AP50
dataset / eval_split / ckpt_path / ckpt_digest / eval_command / raw_artifact / source_files 完整
full_network_claim = false
```

建议实现路线:

1. 复用已有 native INT8 TVM graph executor artifact。
2. 明确 native INT8 backbone/subnet 的输入输出接口。
3. 实现或验证与 HEAL downstream eval pipeline 的连接。
4. 如果无法连接, blocker 必须写清楚具体接口差异: input name、shape、dtype、输出 tensor、downstream 期望、失败 traceback。
5. 不允许把旧 QDQ route、TRT INT8、simulated/predicted AP 写成 measured。

### 3.3 单点通过后的批量策略

单点 FP16 和 INT8 均通过后:

```text
FP16 AP: s0_024 -> 60 labels
INT8 AP: s0_024 -> 60 labels
```

每批失败 label 继续运行其他 labels, 并写入:

```text
quarantine/fp16_int8_original60_ap_true_eval_blockers_v1.jsonl
raw/ap_eval_original60/<precision>_<label>/
```

## 4. 刷新 canonical 的命令

AP row 产出后执行:

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

PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_ap_true_eval_queue.py \
  --output-root multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627
```

验收:

```text
FP16/INT8 latency = 120/120 measured
FP16/INT8 energy = 120/120 measured
FP16/INT8 AP70 = 120/120 measured
jobs_requiring_action = 0
AP true-eval queue blocked = 0
```

## 5. 验证记录

本轮已验证:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest \
  framework.tests.test_stage2_original60_quant_completion.Stage2Original60QuantCompletionTest.test_ap_true_eval_queue_cli_marks_compliant_sources_ready_and_blocks_bad_sources

python -m py_compile scripts/stage2_generate_original60_quant_ap_true_eval_queue.py
```

结果:

```text
Ran 1 test in 0.171s
OK
py_compile OK
```

后续完整收尾前还需要跑:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest \
  framework.tests.test_stage2_lut_productization.Stage2Original60QuantCoverageCliTest \
  framework.tests.test_stage2_original60_quant_completion

python -m py_compile \
  scripts/stage2_generate_original60_quant_state_coverage.py \
  scripts/stage2_generate_original60_quant_ap_true_eval_queue.py \
  scripts/stage2_generate_fp16_int8_original60_completion_queue.py \
  framework/tests/test_stage2_lut_productization.py \
  framework/tests/test_stage2_original60_quant_completion.py
```

## 6. 下一阶段 /goal 命令

```text
/goal 继续在 ${V2X_ROOT} 执行 Stage2 original60 FP16/INT8 AP true-eval 收口。不要启动 agent team, 单 agent 执行即可。当前 FP16/INT8 latency=120/120 measured、energy=120/120 measured, AP=0/120 measured。已新增 AP ingestion 和 AP true-eval queue/source audit; 当前 jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl 为 120 rows, 全部 blocked, H800 AP 软件环境 preflight pass, 根因是缺少合规 true-FP16/native-INT8 model_eval adapter, 不是环境或数据集缺失。下一阶段先完成 s0_024 单点 FP16 true-eval AP row 和 s0_024 单点 native INT8 AP row 或精确接口 blocker, 每行必须包含 dataset/eval_split/checkpoint/digest/eval_command/raw_artifact/source_files/AP30/AP50/AP70/precision route/full_network_claim=false, 禁止 TRT、predicted、model-fit、interpolated AP。单点通过后扩到 original60 60 labels x {fp16,int8}; 每个失败 label 必须保存 stdout/stderr、runner command、GPU 状态、raw artifact、failure reason, 先做最小复现和修复后重试, 不允许直接停止。
```
