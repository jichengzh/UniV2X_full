# HANDOFF — V2X-ViT Attention T1-P/T1-Q TVM Final Gate (v2, 2026-06-23)

> 2026-06-23 update: static-HMSA v3 subnet stage has been completed. New windows should now start from
> [`HANDOFF_v2xvit_attention_t1_pq_tvm_v3.md`](HANDOFF_v2xvit_attention_t1_pq_tvm_v3.md).

> 新窗口接手先读本页。本页是清上下文后的唯一启动入口。v1 交接仍保留作历史细节, 但下一步以本 v2 为准。
>
> 背景设计文档: [`plan_transformer_into_framework_v1.md`](../design/plan_transformer_into_framework_v1.md)。

---

## 0. 当前结论

**不要进入 T2。当前 gate 仍是:**

```text
T1_PQ_INCOMPLETE_DO_NOT_START_T2
```

用户最终验收标准已经明确:

1. 完整 TVM 加速实现。
2. 完整剪枝 + 量化加速实现。
3. 先统计 attention subnet 加速, 但只看 attention 部分。
4. 再做 full-model end-to-end 加速效果。
5. 不能再用逐算子/direct-matmul 或 fake-quant prior 冒充端到端结论。

当前真实状态:

- T1-P: HMSA/MSwin 手写结构化 scanner 已补。
- T1-Q: TVM-only 路线已替代旧 TensorRT 口径。
- attention-p50 short-finetune FP16 full DAIR val 已有真实端到端证据, 通过当前 AP guardrail。
- TVM mixed-INT8 full-model e2e row 仍缺失, 所以 Stop-A 未完成。
- 最新代码已补 HMSA fixed-type `[0,1]` static TVM target, 覆盖 q/k/v projection + relation core + output projection, 但还没有在 H800 重测。

---

## 1. 接手后第一步

先拉起两个 agent, 且分工必须保持互相制衡:

| agent | 角色 | 任务 | 输出 |
|---|---|---|---|
| Agent-A `attention-tvm-executor` | 方案执行 | 跑 H800 static-HMSA v3 subnet bench; 生成 subnet v2; 若通过再实现 full-model/fusion-subgraph TVM mixed-INT8 runner | JSON/CSV/MD 实验产物 + 命令日志 |
| Agent-B `attention-pq-reviewer` | 批判性验收 | 审查 Agent-A 的结果是否同协议、是否真 TVM、是否端到端、是否把 subnet/逐算子冒充 full-model | reviewer verdict + 需要纠正的问题 |

工作节奏:

1. Agent-A 先跑/实现, Agent-B 不改代码, 只审查证据。
2. 每个新增表格都必须经过 Agent-B 审查后再写入 HANDOFF/报告。
3. 如果 mixed-INT8 在同剪枝 FP16 对比下仍更慢, 直接 Stop-C/负证据, 不包装成收益。

---

## 2. 已完成的关键产物

### 2.1 T1-P scanner

文件:

- [`scripts/phase2/t1_attention_pq_feasibility.py`](../../../scripts/phase2/t1_attention_pq_feasibility.py)
- [`results/attention_axis_feasibility_v1.json`](../../../results/attention_axis_feasibility_v1.json)
- [`results/attention_axis_feasibility_v1.md`](../../../results/attention_axis_feasibility_v1.md)

scanner 枚举:

| family | count | 说明 |
|---|---:|---|
| `hmsa_head` | 3 | 3 个 encoder layer, 每层 HMSA 8 heads × dim_head 32 |
| `mswin_head` | 9 | 3 层 × window_size {4,8,16}; heads 分别 {16,8,4} |
| `encoder_depth` | 3 | depth pruning 候选 |
| `embed_dim` | 1 | dim=256, round_to=32 的全局通道候选 |

P gate:

```text
P_axis_READY_WITH_MANUAL_SCANNER_DEPGRAPH_PARTIAL
```

注意: MSwin isolated DepGraph 仍失败, 后续不能依赖通用 DepGraph 自动切 MSwin, 必须用手写 slice 规则。

### 2.2 Attention pruning + short finetune

文件:

- [`scripts/phase2/t1_attention_e2e_pq.py`](../../../scripts/phase2/t1_attention_e2e_pq.py)
- [`scripts/phase2/t1_attention_p50_short_finetune.py`](../../../scripts/phase2/t1_attention_p50_short_finetune.py)
- [`models/v2xvit_attention_t1/attention_p50_manifest_v1.json`](../../../models/v2xvit_attention_t1/attention_p50_manifest_v1.json)
- [`models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json`](../../../models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json)
- [`models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth`](../../../models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth)

剪枝方法:

- HMSA: 结构化 head pruning, 切 q/k/v Linear 输出通道、a_linears 输入通道、relation_att/relation_msg 的 head 维。
- MSwin: 结构化 head pruning, 切 to_qkv 中 Q/K/V 的 head 片段和 to_out[0] 输入通道。
- residual 输出维度保持 256。
- pos_embedding/relative_indices 不随 head pruning 改 shape。

full DAIR val FP16 结果:

| config | prune | finetune | latency p50 ms | speedup | AP50 | AP70 | ΔAP50 | ΔAP70 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 0% | official ckpt | 38.6740 | 1.0000× | 0.71013 | 0.52161 | 0 | 0 |
| attention-p50-fp16 no-ft | 50% | none | 38.9024 | 0.9941× | 0.66884 | 0.46274 | -0.0413 | -0.0589 |
| attention-p50-shortft-fp16 | 50% | 100 steps, lr=1e-4 | 35.8636 | 1.0784× | 0.69729 | 0.51322 | -0.0128 | -0.0084 |

结论: 当前 pruning+shortft FP16 可以作为后续 TVM mixed-INT8 的 checkpoint candidate, 但它本身不是最终验收。

### 2.3 TVM attention bench

文件:

- [`scripts/phase2/t1_attention_tvm_bench.py`](../../../scripts/phase2/t1_attention_tvm_bench.py)
- [`results/attention_axis_tvm_bench_v1.json`](../../../results/attention_axis_tvm_bench_v1.json)
- [`results/attention_full_tvm_bench_v2.json`](../../../results/attention_full_tvm_bench_v2.json)
- [`logs/attention_e2e_pq_v1/attention_full_tvm_bench_v2.log`](../../../logs/attention_e2e_pq_v1/attention_full_tvm_bench_v2.log)

已实测 H800 TVM v2:

| target | FP16 p50 ms | mixed INT8 p50 ms | mixed/FP16 speedup | 解释 |
|---|---:|---:|---:|---|
| `mswin_bwa_full_attention` | 0.456245 | 0.465418 | 0.9803× | 负/中性 |
| `mswin_bwa_p50_full_attention` | 0.244672 | 0.257077 | 0.9517× | 负/中性 |
| `hmsa_full_attention` minimal relation core | 0.332906 | 0.348373 | 0.9556× | 负/中性, minimal core only |
| `hmsa_p50_full_attention` minimal relation core | 0.199093 | 0.205259 | 0.9700× | 负/中性, minimal core only |

已新增但未重测:

- `make_hmsa_static_2agent_param_plan`
- `build_relax_hmsa_static_2agent_fp16`
- `build_relax_hmsa_static_2agent_mixed_int8`
- `bench_hmsa_static_2agent_attention`

这个 static target 固定 type order `[0,1]`, 覆盖 q/k/v projection、relation_att/msg core、output projection。仍不覆盖 dynamic `types` dispatch。

### 2.4 Attention subnet acceleration v1

文件:

- [`scripts/phase2/attention_subnet_accel_report.py`](../../../scripts/phase2/attention_subnet_accel_report.py)
- [`results/attention_subnet_accel_v1.json`](../../../results/attention_subnet_accel_v1.json)
- [`results/attention_subnet_accel_v1.md`](../../../results/attention_subnet_accel_v1.md)

v1 仍基于 HMSA minimal relation-core, 不是 static-HMSA v3:

| config | prune | quant | subnet p50 ms | speedup vs base fp16 | speedup vs same-prune fp16 |
|---|---:|---|---:|---:|---:|
| `attention-subnet-base-fp16` | 0% | fp16 | 0.789151 | 1.0000× | 1.0000× |
| `attention-subnet-base-mixed-int8` | 0% | mixed_int8 | 0.813791 | 0.9697× | 0.9697× |
| `attention-subnet-p50-fp16` | 50% | fp16 | 0.443765 | 1.7783× | 1.0000× |
| `attention-subnet-p50-mixed-int8` | 50% | mixed_int8 | 0.462336 | 1.7069× | 0.9598× |

结论: pruning 对 subnet 有收益, 但当前 mixed-INT8 慢于 same-prune FP16, 不能当作量化收益。

### 2.5 Final acceptance staging table

文件:

- [`scripts/phase2/attention_final_acceptance_report.py`](../../../scripts/phase2/attention_final_acceptance_report.py)
- [`results/attention_final_acceptance_v1.json`](../../../results/attention_final_acceptance_v1.json)
- [`results/attention_final_acceptance_v1.md`](../../../results/attention_final_acceptance_v1.md)
- [`scripts/phase2/attention_e2e_pq_validator.py`](../../../scripts/phase2/attention_e2e_pq_validator.py)

当前表:

| config | prune | quant | latency p50 ms | speedup | AP50 | AP70 | ΔAP50 | ΔAP70 | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0% | fp16 | 38.6740 | 1.0000× | 0.7101 | 0.5216 | 0 | 0 | real full-val |
| attention-p50-fp16 | 50% | fp16 | 35.8636 | 1.0784× | 0.6973 | 0.5132 | -0.0128 | -0.0084 | real full-val |
| attention-p50-int8/mixed | 50% | TVM mixed INT8 | missing | missing | missing | missing | missing | missing | `MISSING_TVM_E2E` |

Stop-A validator 当前预期返回 `REJECT`, 且错误只应集中在 `attention-p50-int8/mixed` 缺少 e2e latency/AP/log/command。

---

## 3. 本机环境事实

当前本机有空闲 RTX 4090, 但没有可 import 的 TVM:

```text
${V2X_ROOT}/tvm_venv310/bin/python: ModuleNotFoundError: No module named 'tvm'
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python: ModuleNotFoundError: No module named 'tvm'
```

因此 TVM bench 必须去 H800 TVM 环境跑。不要在本机伪造 TVM 结果。

---

## 4. 新窗口启动命令

### 4.1 先确认本地测试

```bash
cd ${V2X_ROOT}

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m py_compile \
  scripts/phase2/t1_attention_pq_feasibility.py \
  scripts/phase2/t1_attention_tvm_bench.py \
  scripts/phase2/t1_attention_e2e_pq.py \
  scripts/phase2/t1_attention_p50_short_finetune.py \
  scripts/phase2/attention_subnet_accel_report.py \
  scripts/phase2/attention_e2e_checkpoint_eval.py \
  scripts/phase2/attention_final_acceptance_report.py \
  scripts/phase2/attention_e2e_pq_validator.py

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m pytest \
  tests/phase2/test_t1_attention_pq_feasibility.py \
  tests/phase2/test_t1_attention_tvm_bench.py \
  tests/phase2/test_t1_attention_e2e_pq.py \
  tests/phase2/test_t1_attention_p50_short_finetune.py \
  tests/phase2/test_attention_subnet_accel_report.py \
  tests/phase2/test_attention_e2e_checkpoint_eval.py \
  tests/phase2/test_attention_final_acceptance_report.py \
  tests/phase2/test_attention_e2e_pq_validator.py -q
```

已验证结果:

```text
42 passed, 2 warnings
```

### 4.2 确认当前 Stop-B/Stop-A gate

```bash
cd ${V2X_ROOT}

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py --mode stop-b

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_final_acceptance_v1.json
```

预期:

```text
Stop-B: ACTIONABLE_BLOCKER
Stop-A: REJECT, because attention-p50-int8/mixed row is missing e2e metrics/logs
```

### 4.3 T1 gate 断言

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python - <<'PY'
import json
from pathlib import Path
data=json.loads(Path('results/attention_axis_feasibility_v1.json').read_text())
assert data['gate']['overall']['verdict']=='T1_PQ_INCOMPLETE_DO_NOT_START_T2'
assert data['gate']['q_axis']['verdict']=='Q_axis_TVM_DIRECT_INT8_PARTIAL_FULL_ATTN_BLOCKED'
assert data['p_axis']['manual_scanner']['counts']['hmsa_head']==3
assert data['p_axis']['manual_scanner']['counts']['mswin_head']==9
assert any(row['prune_rate_pct']==50 and row['quant']=='int8' for row in data['coupling_table'])
print('gate_assertions_ok')
PY
```

已验证:

```text
gate_assertions_ok
```

---

## 5. 下一步执行计划

### Step 1: H800 重跑 static-HMSA v3 subnet

同步最新脚本到 H800 TVM 环境, 运行:

```bash
cd ${V2X_DATA_ROOT}/v2x_t1_attention
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path)

CUDA_VISIBLE_DEVICES=6 ${V2X_DATA_ROOT}/tvm310/bin/python \
  scripts/t1_attention_tvm_bench.py \
  --out-json results/attention_full_tvm_bench_v3_static.json \
  --full-attention --full-only --number 3 --repeat 3 \
  > logs/attention_full_tvm_bench_v3_static.log 2>&1
```

传回本地后生成 subnet v2:

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_subnet_accel_report.py \
  --tvm-report results/attention_full_tvm_bench_v3_static.json \
  --out-json results/attention_subnet_accel_v2_static.json \
  --out-csv results/attention_subnet_accel_v2_static.csv \
  --out-md results/attention_subnet_accel_v2_static.md
```

Agent-B 必查:

- `hmsa_full_attention` 和 `hmsa_p50_full_attention` 的 scope 是否是 static qkv/relation/out, 不是旧 minimal core。
- p50 mixed-INT8 是否超过 same-prune FP16。
- 如果仍低于 same-prune FP16, 不要继续写 positive quantization claim。

### Step 2: 若 subnet mixed-INT8 仍失败, Stop-C

停止条件:

```text
attention-subnet-p50-mixed-int8 speedup_vs_same_prune_fp16 <= 1.0
```

产物:

- `results/attention_e2e_pq_blocker_v2.md`
- 更新 reviewer 文档
- 明确写: pruning 有效, 当前 TVM mixed-INT8 量化路径无收益或负收益, 不进入 full-model TVM mixed-INT8 包装。

### Step 3: 若 subnet mixed-INT8 通过, 做 full-model/fusion-subgraph runner

目标 row:

```text
config: attention-p50-int8/mixed
checkpoint_path: models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth
manifest_path: models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json
quant_backend: TVM Relax int8/mixed
latency_scope: e2e
dataset_split: DAIR val
n_samples: 1789
```

产物:

- `results/attention_p50_tvm_mixed_int8_row_v1.json`
- `results/attention_e2e_pq_v1.json`
- `results/attention_e2e_pq_v1.csv`
- `results/attention_e2e_pq_v1.md`
- 原始 latency/AP logs under `logs/attention_e2e_pq_v1/`

生成最终表:

```bash
cd ${V2X_ROOT}
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_final_acceptance_report.py \
  --e2e-report results/attention_e2e_checkpoint_eval_full_v1.json \
  --subnet-report results/attention_subnet_accel_v2_static.json \
  --tvm-e2e-row results/attention_p50_tvm_mixed_int8_row_v1.json \
  --out-json results/attention_e2e_pq_v1.json \
  --out-csv results/attention_e2e_pq_v1.csv \
  --out-md results/attention_e2e_pq_v1.md

${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_e2e_pq_v1.json
```

Stop-A 成功条件:

```text
validator verdict == ACCEPTABLE_E2E_SCHEMA
attention-p50-int8/mixed speedup >= 1.10
delta_ap50 >= -0.02
delta_ap70 >= -0.02
```

---

## 6. 关键文件清单

| 文件 | 用途 |
|---|---|
| [`multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_tvm_v2.md`](HANDOFF_v2xvit_attention_t1_pq_tvm_v2.md) | 当前交接入口 |
| [`multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_tvm_v1.md`](HANDOFF_v2xvit_attention_t1_pq_tvm_v1.md) | 历史长版细节 |
| [`scripts/phase2/t1_attention_tvm_bench.py`](../../../scripts/phase2/t1_attention_tvm_bench.py) | TVM direct/MSwin/HMSA static bench |
| [`scripts/phase2/attention_subnet_accel_report.py`](../../../scripts/phase2/attention_subnet_accel_report.py) | subnet 加速表生成 |
| [`scripts/phase2/attention_e2e_checkpoint_eval.py`](../../../scripts/phase2/attention_e2e_checkpoint_eval.py) | baseline/no-ft/shortft checkpoint full-val eval |
| [`scripts/phase2/attention_final_acceptance_report.py`](../../../scripts/phase2/attention_final_acceptance_report.py) | Stop-A 最终表生成 |
| [`scripts/phase2/attention_e2e_pq_validator.py`](../../../scripts/phase2/attention_e2e_pq_validator.py) | Stop-A/Stop-B validator |
| [`results/attention_final_acceptance_v1.json`](../../../results/attention_final_acceptance_v1.json) | 当前 blocked final staging table |
| [`results/attention_e2e_checkpoint_eval_full_v1.json`](../../../results/attention_e2e_checkpoint_eval_full_v1.json) | full-val FP16 checkpoint 证据 |
| [`results/attention_subnet_accel_v1.json`](../../../results/attention_subnet_accel_v1.json) | 旧 v1 subnet, minimal HMSA core |
| [`results/attention_full_tvm_bench_v2.json`](../../../results/attention_full_tvm_bench_v2.json) | H800 v2 TVM 原始结果 |
| [`results/attention_e2e_pq_blocker_v1.md`](../../../results/attention_e2e_pq_blocker_v1.md) | 当前 blocker |
| [`results/attention_e2e_pq_review_v1.md`](../../../results/attention_e2e_pq_review_v1.md) | 当前 reviewer 记录 |

---

## 7. 最容易犯错的点

1. 不要把 direct-matmul speedup 写成 attention fusion speedup。
2. 不要把 `attention_subnet_accel_v1` 写成 full-model e2e。
3. 不要把 fake-quant prior AP 写成 TVM INT8 AP。
4. 不要说 TVM mixed-INT8 已经有收益; 当前 v1/v2 证据显示 same-prune FP16 更快。
5. 不要覆盖 `attention_full_tvm_bench_v2.json`; static-HMSA 新结果应落 `attention_full_tvm_bench_v3_static.json`。
6. 不要进入 T2, 除非 `results/attention_e2e_pq_v1.json` 通过 Stop-A validator。

---

## 8. 当前验证记录

最近一次本地验证:

```text
py_compile: passed
pytest phase2 selected: 42 passed, 2 warnings
attention_e2e_pq_validator --mode stop-b: ACTIONABLE_BLOCKER
attention_e2e_pq_validator --mode stop-a --report-json results/attention_final_acceptance_v1.json: expected REJECT
gate_assertions_ok
```

Stop-A reject 的原因全部是 `attention-p50-int8/mixed` 缺失真实 e2e 指标、命令和日志。这是预期保护。
