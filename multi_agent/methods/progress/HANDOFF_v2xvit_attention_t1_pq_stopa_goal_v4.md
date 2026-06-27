# 中文交接 — V2X-ViT Attention T1-P/T1-Q 直到 Stop-A 通过 (v4, 2026-06-23)

> 这是下一次用 `/goal` 启动工作的入口文档。目标不是开始 T2，而是继续完成 T1，直到 `attention-p50-int8/mixed` 的完整 TVM mixed-INT8 端到端行通过 Stop-A validator。

---

## 0. /goal 建议目标

建议新窗口直接用下面的目标启动：

```text
完成 V2X-ViT attention T1-P/T1-Q 的最终 Stop-A：实现 full-model 或 fusion-subgraph TVM mixed-INT8 runner，生成真实 attention-p50-int8/mixed 端到端 latency/AP 行，并让 results/attention_e2e_pq_v1.json 通过 Stop-A validator。期间不得进入 T2，不得用 subnet/逐算子/fake-quant 冒充 e2e。
```

当前 gate:

```text
T1_PQ_INCOMPLETE_DO_NOT_START_T2
```

Stop-A 通过前禁止进入：

```text
Phase T2 — stage1 扩到 attention
```

---

## 1. 当前真实状态

已经完成：

- T0: attention/fusion breakdown 已完成。
- T1-S: MSwin `relative_indices` CPU buffer 问题已定位并修复，S 轴已有实现修复收益。
- T1-P: HMSA/MSwin 结构化 head pruning scanner 和 50% attention pruning 已完成。
- T1-P full-val: `attention-p50-shortft-fp16` 已有完整 DAIR val 结果，可作为下一步 TVM mixed-INT8 checkpoint。
- T1-Q subnet: static-HMSA v3 TVM bench 已完成，attention subnet mixed-INT8 相比 same-prune FP16 有小幅正收益。

仍未完成：

- `attention-p50-int8/mixed` 的 full-model e2e latency/AP 行仍缺失。
- dynamic HMSA type dispatch 仍未被 static-HMSA v3 覆盖。
- TVM mixed-INT8 的 AP 保真仍未实测。
- Stop-A validator 仍应返回 `REJECT`，这是正确保护。

---

## 2. 当前关键证据

### 2.1 Full-val FP16 checkpoint 证据

来源：

- `results/attention_e2e_checkpoint_eval_full_v1.json`
- `results/attention_e2e_checkpoint_eval_full_v1.csv`
- `results/attention_e2e_checkpoint_eval_full_v1.md`

| config | prune | finetune | latency p50 ms | speedup | AP50 | AP70 | ΔAP50 | ΔAP70 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 0% | official ckpt | 38.6740 | 1.0000x | 0.71013 | 0.52161 | 0 | 0 |
| attention-p50-fp16 no-ft | 50% | none | 38.9024 | 0.9941x | 0.66884 | 0.46274 | -0.0413 | -0.0589 |
| attention-p50-shortft-fp16 | 50% | 100 steps lr=1e-4 | 35.8636 | 1.0784x | 0.69729 | 0.51322 | -0.0128 | -0.0084 |

结论：

- 50% attention pruning 需要 short finetune 才能满足 AP guardrail。
- 下一步必须基于 shortft checkpoint，而不是 no-ft checkpoint。

### 2.2 Static-HMSA v3 TVM subnet 证据

来源：

- `results/attention_full_tvm_bench_v3_static.json`
- `logs/attention_e2e_pq_v1/attention_full_tvm_bench_v3_static.log`
- `results/attention_subnet_accel_v2_static.{json,csv,md}`

HMSA v3 scope:

```text
full_attention_hmsa_static_2agent_qkv_relation_core_mixed_int8
```

覆盖：

- q/k/v projection: 已覆盖
- relation_att/msg core: 已覆盖
- output projection: 已覆盖
- dynamic type dispatch: 未覆盖

TVM subnet 数值：

| config | prune | quant | subnet p50 ms | speedup vs base FP16 | speedup vs same-prune FP16 |
|---|---:|---|---:|---:|---:|
| attention-subnet-base-fp16 | 0% | fp16 | 0.928630 | 1.0000x | 1.0000x |
| attention-subnet-base-mixed-int8 | 0% | mixed_int8 | 0.913952 | 1.0161x | 1.0161x |
| attention-subnet-p50-fp16 | 50% | fp16 | 0.564448 | 1.6452x | 1.0000x |
| attention-subnet-p50-mixed-int8 | 50% | mixed_int8 | 0.542453 | 1.7119x | 1.0405x |

结论：

```text
subnet 层面不触发 Stop-C，可以继续做 full-model/fusion-subgraph TVM mixed-INT8 runner。
```

注意：

- 这不是 full-model e2e。
- 这不是 AP 证据。
- 这不能写成最终 prune+quant 加速结论。

### 2.3 当前 final staging 表

来源：

- `results/attention_final_acceptance_v2_static.json`
- `results/attention_final_acceptance_v2_static.csv`
- `results/attention_final_acceptance_v2_static.md`

当前状态：

```text
BLOCKED_TVM_MIXED_INT8_E2E_MISSING
```

当前缺失行：

```text
attention-p50-int8/mixed
```

Stop-A validator 预期：

```text
REJECT
```

预期 reject 原因只应集中在：

- missing e2e latency
- missing speedup
- missing AP50/AP70
- missing ΔAP50/ΔAP70
- missing latency/AP command
- missing latency/AP log
- latency_scope 不是 `e2e`

---

## 3. 下一阶段必须产出的文件

目标产物：

- `results/attention_p50_tvm_mixed_int8_row_v1.json`
- `results/attention_e2e_pq_v1.json`
- `results/attention_e2e_pq_v1.csv`
- `results/attention_e2e_pq_v1.md`
- TVM runner 的 latency/AP 原始日志，放在 `logs/attention_e2e_pq_v1/`

建议新增或修改的脚本位置：

- 新增 runner 优先放在 `scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py`
- 或扩展已有 `scripts/phase2/attention_e2e_checkpoint_eval.py`
- 不要把逻辑塞进报告脚本，`attention_final_acceptance_report.py` 只负责组表

最终 row 必须包含：

```json
{
  "config": "attention-p50-int8/mixed",
  "attention_prune_pct": 50,
  "quant": "int8/mixed",
  "quant_backend": "TVM Relax int8/mixed",
  "checkpoint_path": "models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth",
  "manifest_path": "models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json",
  "latency_scope": "e2e",
  "dataset_split": "DAIR val",
  "n_samples": 1789
}
```

数值字段必须来自真实运行：

- `e2e_latency_ms`
- `speedup`
- `ap50`
- `ap70`
- `delta_ap50`
- `delta_ap70`
- `latency_command`
- `latency_log`
- `ap_command`
- `ap_log`

---

## 4. 推荐执行路线

### Step 1: 先保留两条 agent 轨道

继续使用两个角色：

| agent | 角色 | 职责 |
|---|---|---|
| Agent-A `attention-tvm-executor` | 执行 | 实现/运行 full-model 或 fusion-subgraph TVM mixed-INT8 runner，生成 row 和日志 |
| Agent-B `attention-pq-reviewer` | 审查 | 只读审查 row 是否真 TVM、真 e2e、同协议、不是 subnet 冒充 |

要求：

- Agent-A 可以改代码和跑实验。
- Agent-B 不改代码，只审查证据。
- 每次生成最终表前先让 Agent-B 查 row。

### Step 2: 实现 runner 的最小可接受版本

runner 必须满足：

- 加载 `attention-p50-shortft` checkpoint。
- 使用同一 DAIR val split。
- latency 路径必须跑完整 model forward，而不是只测 attention subnet。
- AP 路径必须用同一个 runner 产出的 detection output 计算 AP。
- TVM mixed-INT8 必须实际参与 inference path。
- 日志必须记录命令、设备、样本数、checkpoint、manifest、TVM report 或 build artifact。

可以先接受的工程折中：

- full model 仍由 PyTorch 包裹，但 attention fusion 子图由 TVM runtime 执行。
- dynamic HMSA dispatch 如果暂时不完整，可以先固定 DAIR val 实际 type order 的可证路径，但必须在 row 里写清 `dynamic_type_dispatch_coverage`。
- 如果只覆盖 static `[0,1]` 而 DAIR val 实际运行也固定这个顺序，需要在日志里证明。

不接受：

- 只用 `attention_subnet_accel_v2_static` 推算 e2e。
- 只把 subnet latency 从 PyTorch latency 中相减/相加。
- fake quant AP。
- direct matmul speedup。
- TensorRT/TRT/QDQ-ONNX 替代 TVM。

### Step 3: 生成 final report

runner 生成 `results/attention_p50_tvm_mixed_int8_row_v1.json` 后执行：

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_final_acceptance_report.py \
  --e2e-report results/attention_e2e_checkpoint_eval_full_v1.json \
  --subnet-report results/attention_subnet_accel_v2_static.json \
  --tvm-e2e-row results/attention_p50_tvm_mixed_int8_row_v1.json \
  --out-json results/attention_e2e_pq_v1.json \
  --out-csv results/attention_e2e_pq_v1.csv \
  --out-md results/attention_e2e_pq_v1.md
```

然后执行：

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_e2e_pq_v1.json
```

---

## 5. Stop-A 通过标准

必须同时满足：

```text
validator verdict == ACCEPTABLE_E2E_SCHEMA
attention-p50-int8/mixed speedup >= 1.10
delta_ap50 >= -0.02
delta_ap70 >= -0.02
```

并且人工审查确认：

- `latency_scope == e2e`
- `quant_backend` 明确是 `TVM Relax int8/mixed`
- AP 不是 simulated/fake/prior
- baseline、attention-p50-fp16、attention-p50-int8/mixed 使用同一 DAIR val split
- shortft checkpoint 和 manifest 对应同一 50% structured attention head pruning
- row 中 latency/AP command 和 log path 可复现

---

## 6. 如果 full-model mixed-INT8 未达标

不要包装成成功。按下面顺序处理：

1. 如果 runner 还没真正接入 TVM attention path，继续修 runner。
2. 如果 TVM 已接入但 speedup < 1.10，先做 profiling，确认瓶颈是 MSwin overhead、HMSA dispatch、Q/DQ 开销、还是非 attention 模块。
3. 如果 AP 下降超过 -0.02，尝试 calibrated mixed-INT8 或 QAT/短微调，但必须重新生成 AP。
4. 如果多次真实 runner 后仍不满足 Stop-A，写 Stop-C 负证据，不能进入 T2。

可作为新 T1-Q 分支尝试的替代方案：

- 只量化 QKV/FFN/output Linear，softmax/LN 保 FP16。
- W8A16 或 weight-only INT8。
- TVM fused QKV + Q/DQ + output projection。
- TVM MetaSchedule/TensorIR 自定义 attention schedule。
- dynamic HMSA dispatch lowering。

这些方案只有在生成 full-model e2e row 并通过 validator 后才算成功。

---

## 7. 本地验证基线

当前最近一次验证：

```text
py_compile: passed
phase2 selected pytest: 43 passed, 2 warnings
Stop-B validator: ACTIONABLE_BLOCKER
Stop-A v2_static staging: expected REJECT
static_subnet_assertions_ok
```

可复跑：

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m py_compile \
  scripts/phase2/t1_attention_pq_feasibility.py \
  scripts/phase2/t1_attention_tvm_bench.py \
  scripts/phase2/t1_attention_e2e_pq.py \
  scripts/phase2/t1_attention_p50_short_finetune.py \
  scripts/phase2/attention_subnet_accel_report.py \
  scripts/phase2/attention_e2e_checkpoint_eval.py \
  scripts/phase2/attention_final_acceptance_report.py \
  scripts/phase2/attention_e2e_pq_validator.py

/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest \
  tests/phase2/test_t1_attention_pq_feasibility.py \
  tests/phase2/test_t1_attention_tvm_bench.py \
  tests/phase2/test_t1_attention_e2e_pq.py \
  tests/phase2/test_t1_attention_p50_short_finetune.py \
  tests/phase2/test_attention_subnet_accel_report.py \
  tests/phase2/test_attention_e2e_checkpoint_eval.py \
  tests/phase2/test_attention_final_acceptance_report.py \
  tests/phase2/test_attention_e2e_pq_validator.py -q
```

---

## 8. 关键文件清单

设计和交接：

- `multi_agent/methods/design/plan_transformer_into_framework_v1.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_tvm_v3.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v4.md`

runner/report 脚本：

- `scripts/phase2/t1_attention_tvm_bench.py`
- `scripts/phase2/attention_subnet_accel_report.py`
- `scripts/phase2/attention_e2e_checkpoint_eval.py`
- `scripts/phase2/attention_final_acceptance_report.py`
- `scripts/phase2/attention_e2e_pq_validator.py`

已有结果：

- `results/attention_full_tvm_bench_v3_static.json`
- `results/attention_subnet_accel_v2_static.json`
- `results/attention_final_acceptance_v2_static.json`
- `results/attention_e2e_checkpoint_eval_full_v1.json`
- `results/attention_e2e_pq_blocker_v2_static.md`
- `results/attention_e2e_pq_review_v2_static.md`

checkpoint：

- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth`
- `models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json`

---

## 9. 硬边界

- 不要进入 T2。
- 不要把 subnet v2 static 写成 full-model e2e。
- 不要用 fake-quant prior AP。
- 不要用 TensorRT/TRT/QDQ-ONNX 替代 TVM。
- 不要声称 dynamic HMSA dispatch 已完成，除非 runner 日志证明。
- 不要覆盖 `results/attention_full_tvm_bench_v2.json` 或 v3 static 原始结果。
- 不要在文档里写远程密码。

