# Stage2 H800 Multi-Agent LUT Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已经完成的 coverage-first 多 agent 机制接到 H800 真实 artifact 和测量环境上, 生成非空 latency/AP/energy 队列, 并持续扩大 H800 LUT 的 unique config 覆盖。

**Architecture:** 本计划继承 `PLAN_stage2_multi_agent_lut_coverage_pipeline_v1_zh.md` 的 durable queue、artifact gate、axis follower 和 supervisor gate。生产阶段不再把 24 轮 repeat 当作进展, 而是让 artifact-agent 先把候选变成 artifact-ready, latency-agent 扩展 unique latency cells, energy/AP-agent 追赶同一 candidate bundle, supervisor-agent 每轮用 unique coverage 和 quarantine 状态决定是否继续。

**Tech Stack:** Python JSONL queues, Stage2 LUT productization scripts, H800 TVM/Relax/MetaSchedule, latency/energy GPU idle preflight, AP memory/OOM preflight, H800 power telemetry, AP true eval/import/finetune, ssh/rsync, dmux/Codex 多 agent 会话。

---

## 0. 前置结论

上一个计划已经完成的内容:

1. `scripts/stage2_summarize_lut_coverage.py`
2. `scripts/stage2_supervisor_poll.py`
3. `scripts/stage2_generate_coverage_candidates.py`
4. `scripts/stage2_plan_artifact_tasks.py`
5. `scripts/stage2_generate_latency_coverage_jobs.py`
6. `scripts/stage2_generate_energy_coverage_jobs.py`
7. `scripts/stage2_generate_ap_coverage_jobs.py`
8. `RUNBOOK_stage2_multi_agent_lut_polling_v1_zh.md`

当前未完成的内容:

1. H800 上 60 个新 coverage candidates 尚未变成 artifact-ready。
2. 当前本地 artifact gate 输出 `missing_artifact_config_count=60`。
3. 当前 latency/AP/energy ready job queue 都是 0。
4. 因为历史数据 repeat ratio=0.914, supervisor gate 当前是 `NO_GO`。

本计划的第一目标不是直接 claim 大规模启动, 而是把 gate 从“空队列 NO_GO”推进到“有 artifact-ready 候选、有非空三轴队列、能持续轮询”的状态。

## 1. 必须遵守的规则

### 1.1 H800 与安全规则

1. 只有 H800 远端可以写 `backend=h800_tvm` 或 H800 energy telemetry measured row。
2. 本地 4090 只能做脚本、队列、文档和离线检查。
3. 不在 prompt、日志、JSONL、Markdown 中记录真实密码; 示例统一写 `<H800_PASSWORD>`。
4. H800 访问方式固定读取 `RUNBOOK_stage2_h800_server_access_v1_zh.md`。

### 1.2 GPU 使用规则

| agent | 默认 GPU | 规则 |
|---|---|---|
| latency-agent | GPU0/1/4/5 | 每 GPU 串行, 每 job 先做 idle preflight |
| energy-agent | GPU2 | 不与 latency 共用同一卡 |
| AP-agent import/replay | GPU5 或任一已释放 GPU | 只运行 true eval/import/replay, 缺 source 写 no-claim |
| AP-agent stable smoke finetune | GPU0/1/2/3/4 | Phase C 首批 5 个新配置五卡并行微调, 每卡一个配置 |
| supervisor-agent | 不占 GPU | 只读 queue/state/rows/quarantine/export |
| artifact-agent | 按需 | 编译/导出阶段必须声明占用 GPU/CPU, 不写 measured row |

latency/energy measured job 启动前必须保存:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv
nvidia-smi pmon -c 1
```

若 latency/energy 目标 GPU utilization > 5%、有非本 job compute process、或 memory used > 1024 MiB 且不是系统保留, 写 `preflight_blocked`, 不启动 measured job。

AP finetune/AP eval 不套用 latency/energy 的 GPU idle gate。AP 的要求是:

1. 启动前仍保存 `nvidia-smi` 和 `nvidia-smi pmon -c 1` 作为并发记录。
2. 目标 GPU 显存必须足够, 不得存在明显会导致 OOM 的大进程。
3. 微调和 AP eval 允许在 GPU 非完全空闲时运行; 并发只影响耗时, 不直接污染 AP 指标。
4. 若发生 OOM、训练崩溃、eval 崩溃或 AP 漂移异常, 只隔离该配置, 不停止整批。

### 1.3 数据质量规则

1. AP measured row 只能来自真实 eval/import source, predicted AP 只能 report-only。
2. energy telemetry 不稳定时写 no-claim 或 quality flag, 不伪造 measured energy。
3. coverage 阶段同一 `(model,width,quant_policy,schedule_policy,optimized_scope,backend)` 默认不 repeat。
4. 只有 `paper_retest`, `outlier_retest`, `cross_gpu_drift_check` 可以突破 repeat 上限。
5. 任何 CUDA illegal memory / bad DB 必须写 quarantine, 同 config 不进入 continuation。

## 2. 文件和目录责任

生产目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/
```

关键文件:

| 文件 | 责任 |
|---|---|
| `candidates/candidate_queue.jsonl` | coverage candidates, 由 supervisor/artifact-agent 扩展 |
| `artifacts/artifact_tasks.jsonl` | artifact-agent 待处理任务 |
| `artifacts/artifact_state.jsonl` | 每个 candidate 的 latest artifact readiness |
| `artifacts/artifact_registry_v1.jsonl` | latency/AP/energy 共享的 artifact bundle |
| `jobs/latency_job_queue_gpu*.jsonl` | latency-agent 消费的 per-GPU 队列 |
| `jobs/energy_job_queue.jsonl` | energy-agent 消费的队列 |
| `jobs/ap_job_queue.jsonl` | AP-agent 消费的队列 |
| `jobs/*_job_state.jsonl` | append-only job 状态 |
| `rows/latency_lut_rows_v1.jsonl` | 新生产 latency rows |
| `rows/energy_lut_rows_v1.jsonl` | 新生产 energy rows |
| `rows/ap_anchor_rows_v1.jsonl` | 新生产或导入 AP rows |
| `quarantine/*.jsonl` | bad DB/outlier/missing artifact 隔离记录 |
| `exports/readiness_gate_latest.json` | supervisor 机器可读 gate |
| `exports/supervisor_report_latest.md` | 人读报告 |

## 3. Phase A: 同步脚本并做 H800 preflight

**Files:**
- Read: `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`
- Sync: `scripts/stage2_*.py`
- Sync: `framework/stage2/*.py`
- Sync: `multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/`

- [ ] **Step 1: 从本地同步脚本和 coverage pipeline 目录到 H800**

```bash
cd ${V2X_ROOT}
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
rsync -avR \
  -e 'ssh -p 30001 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60' \
  scripts/stage2_summarize_lut_coverage.py \
  scripts/stage2_supervisor_poll.py \
  scripts/stage2_generate_coverage_candidates.py \
  scripts/stage2_plan_artifact_tasks.py \
  scripts/stage2_generate_latency_coverage_jobs.py \
  scripts/stage2_generate_energy_coverage_jobs.py \
  scripts/stage2_generate_ap_coverage_jobs.py \
  scripts/stage2_lut_worker.py \
  scripts/stage2_h800_run_measurement_job.py \
  framework/stage2 \
  multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1 \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/
unset password-based SSH (disabled; use an SSH key)
```

Expected:

```text
rsync exits 0
H800:${V2X_ROOT} contains updated scripts and coverage_pipeline_v1
```

- [ ] **Step 2: H800 单连接 preflight**

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
ssh -p 30001 \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=60 \
  -o ServerAliveInterval=30 \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  'cd ${V2X_ROOT} && hostname && pwd && python3 --version && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv && nvidia-smi pmon -c 1'
unset password-based SSH (disabled; use an SSH key)
```

Expected:

```text
hostname contains <PRIVATE_HOST>
pwd is ${V2X_ROOT}
GPU0/1/2/3/4/5 are idle by the rules in Section 1.2
```

## 4. Phase B: H800 artifact discovery 与候选重锚

**Files:**
- Read: `candidates/candidate_queue.jsonl`
- Write: `artifacts/h800_artifact_inventory_v1.txt`
- Write: `artifacts/artifact_tasks.jsonl`
- Write: `artifacts/artifact_state.jsonl`
- Write: `artifacts/artifact_registry_v1.jsonl`
- Write: `quarantine/missing_artifact_v1.jsonl`

- [ ] **Step 1: 在 H800 上生成 artifact inventory**

```bash
cd ${V2X_ROOT}
mkdir -p multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts
find ${V2X_DATA_ROOT}/s2_tvm \
  \( -name '*.onnx' -o -name 'database_workload.json' -o -name 'database_tuning_record.json' \) \
  -print | sort > multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/h800_artifact_inventory_v1.txt
```

Expected:

```text
h800_artifact_inventory_v1.txt contains ONNX and MetaSchedule DB paths under ${V2X_DATA_ROOT}/s2_tvm
```

- [ ] **Step 2: 在 H800 上重跑 artifact planner**

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
python3 scripts/stage2_plan_artifact_tasks.py \
  --candidate-queue "$BASE/candidates/candidate_queue.jsonl" \
  --quarantine-file "$BASE/quarantine/bad_db_quarantine_v1.jsonl" \
  --artifact-root ${V2X_DATA_ROOT}/s2_tvm \
  --artifact-tasks-out "$BASE/artifacts/artifact_tasks.jsonl" \
  --artifact-state-out "$BASE/artifacts/artifact_state.jsonl" \
  --artifact-registry-out "$BASE/artifacts/artifact_registry_v1.jsonl" \
  --missing-artifact-out "$BASE/quarantine/missing_artifact_v1.jsonl" \
  --created-at 2026-06-26T00:00:00Z
```

Expected:

```text
artifact_state.jsonl and artifact_registry_v1.jsonl are rewritten on H800
ready candidates are explicit
missing/quarantined candidates are explicit and do not enter measurement queues
```

- [ ] **Step 3: 审查 artifact readiness**

```bash
cd ${V2X_ROOT}
python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
base = Path("multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1")
rows = [json.loads(line) for line in (base / "artifacts/artifact_state.jsonl").read_text().splitlines() if line.strip()]
print("artifact_rows", len(rows))
print("artifact_status", dict(Counter(row.get("artifact_status") for row in rows)))
print("ap_status", dict(Counter(row.get("ap_status") for row in rows)))
for row in rows[:10]:
    print(row.get("label"), row.get("artifact_status"), row.get("missing_artifacts"), row.get("ap_status"))
PY
```

Gate:

```text
If artifact_status=ready count >= 12, proceed to Phase C.
If ready count is 1-11, run a very small validation batch and ask artifact-agent to top up.
If ready count is 0, do not start latency/AP/energy workers; artifact-agent must re-anchor candidates to existing H800 artifacts or compile/export new artifacts first.
```

## 5. Phase C: 生成非空 latency/AP/energy 队列

### 5.0 Phase C AP Stable Smoke 硬规则

Phase C 中 AP 的首要目标不是“给队列塞满 AP 数值”, 而是稳定产出首批可追溯 AP 配置。首批 stable smoke 固定采用 10 个已有 source/import 配置 + 5 个 original60 新配置微调:

```text
已有 source/import: base, p50, p75, trap25, mix_b, mix_d, iso_s0, iso_s1, iso_s2, p50b2_136
新微调配置: s0_024, s0_040, s0_056, s1_048, s2_160
```

#### Phase C stable smoke 一页执行规范

这部分是后续 AP-agent 的硬入口。若本节与其他旧文档冲突, 以本节为准。

| 项 | 固定规则 |
|---|---|
| 目标 | 稳定产出首批可追溯 AP, 不是填 predicted/model-fit AP |
| 首批新配置数 | 5 个 original60 新配置 |
| 微调次数 | 每个配置默认 1 次 finetune |
| 训练 epoch | `epoches=31` |
| 初始化 | DAIR base checkpoint 结构化剪枝后的 init@23 |
| 实际训练长度 | 约 8 个 epoch, 从 epoch 23 续训到 epoch 31 |
| 并行方式 | H800 GPU0-4 五卡并行, 每卡一个配置, 单卡 DDP |
| GPU idle 要求 | AP finetune/eval 不要求 GPU 完全空闲 |
| GPU 必要门控 | 必须记录 `nvidia-smi`/`pmon`; 显存明显不足或 OOM 风险时换卡/延后/隔离 |
| 数据集 | DAIR-V2X-C train split 微调, DAIR-V2X-C `val_1789` full eval |
| AP 指标 | AP70 为主, AP30/AP50 为 secondary metrics |
| 成功产物 | `claimable_true_eval` 或带 digest 待补的 `true_eval` AP row |
| 失败处理 | 单配置 quarantine, 其他配置继续跑 |

首批 5 个新配置的固定 GPU 绑定如下:

| job_id | label | width | GPU | master_port | finetune_runs | epoches | train split | eval split |
|---|---|---|---:|---:|---:|---:|---|---|
| ap_ft_01 | s0_024 | [24,128,256] | 0 | 29700 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_02 | s0_040 | [40,128,256] | 1 | 29701 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_03 | s0_056 | [56,128,256] | 2 | 29702 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_04 | s1_048 | [64,48,256] | 3 | 29703 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |
| ap_ft_05 | s2_160 | [64,128,160] | 4 | 29704 | 1 | 31 | DAIR-V2X-C train.json | DAIR-V2X-C val_1789 |

微调流程必须是完整 full-model 链路:

```text
DAIR base ckpt
  -> structural_prune_pyramid.py 按 width 生成 config/init ckpt
  -> 检查 config.yaml 的 epoches=31
  -> 检查/修复 init ckpt 为 flat state_dict
  -> train_ddp.py --half 单卡 DDP 微调
  -> 选择 best checkpoint
  -> export_onnx_pyramid_collab.py 导出 full-model ONNX
  -> m4_8_trt_build_bench.py 构建 TRT FP16 engine
  -> m4_8_hybrid_infer_ap.py 跑 DAIR-V2X-C val_1789 full AP
  -> 写 AP row 或 quarantine row
```

AP 微调的稳定性不靠“默认多次重复”解决。首批只做 1 次 finetune; 只有以下情况才追加第 2 seed 或延长 epoch:

| 触发条件 | 补救动作 |
|---|---|
| AP70 明显高于 base 或违背邻域趋势 | 同配置补第 2 seed, 标记 `reason=ap_trend_anomaly` |
| best checkpoint 出现在最后 epoch | 延长到 `epoches=48`, 标记 `reason=not_converged_at31` |
| 延长后仍不稳定或进入 paper claim 子集 | 延长到 `epoches=70` 并做 2 seed |
| 训练/eval/ONNX/TRT 任一步失败 | 当前配置 quarantine, 不写 measured AP |
| GPU 显存不足/OOM 风险 | 换到 GPU5 或等待释放; 若仍失败, quarantine 当前配置 |

数据集必须固定为:

```text
logical dataset: DAIR-V2X
H800 HEAL path: ${V2X_HOME}/heal_research/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure
H800 preferred storage: ${V2X_DATA_ROOT}/DAIR-V2X/DAIR-V2X-C/cooperative-vehicle-infrastructure
train split: train.json, historical count ~= 4811
eval split: val.json, canonical name val_1789, num_samples=1789
```

最终 `ap_stability_summary.md` 必须能直接审查:

```text
label, width, gpu_id, master_port, finetune_runs, epoches,
dataset_train, eval_split, ckpt_path, config_path,
AP30, AP50, AP70, claim_status, quarantine_reason/raw_artifact
```

#### AP 首批稳定产出的完整链路

每个新配置必须走同一条链路:

```text
width spec
  -> structural_prune_pyramid 生成 full-model config + init ckpt
  -> 检查 init ckpt 是 flat state_dict
  -> train_ddp.py --half 单卡微调
  -> 锁定 best checkpoint
  -> 导出 ONNX
  -> 构建 TRT FP16 engine
  -> DAIR-V2X val_1789 full AP eval
  -> 写 canonical AP row 或 quarantine
```

不能把以下内容写成 measured AP:

```text
predicted AP
model-fit AP
interpolated AP
只有 backbone ONNX/TVM artifact、没有 full-model ckpt 的 AP
随机初始化或 ckpt 格式错误导致的 eval AP
```

#### 微调次数和 epoch 预算

| 项 | Phase C stable smoke 默认值 | 后续升级条件 |
|---|---|---|
| 每个新配置 finetune 次数 | 1 次 | AP 异常、趋势反常、bestval 落在最后 epoch、或 paper claim 时补第 2 seed |
| finetune epoch | `epoches=31` | 未收敛或异常点延长到 `epoches=48` 或 `epoches=70` |
| 初始化 epoch | base/pruned init 视为 `net_epoch_bestval_at23.pth` | 必须记录实际 init ckpt |
| 实际微调预算 | 从 epoch 23 到 epoch 31, 约 8 个 finetune epochs | paper/final claim 再加 repeat/seed |
| seed | 首批默认不强制多 seed | 若 runner 支持 seed, 固定并记录 `20260626`; 若不支持, 记录环境默认 seed |

这意味着首批 5 个新配置默认总共跑 5 次 finetune, 不是每个配置多轮重复。repeat 只用于异常排查和 paper-grade 子集, 不用于扩充 raw row count。

#### 5GPU 并行配置

| job | label | width | GPU | finetune 次数 | epoches | eval split | 产出 |
|---|---|---|---:|---:|---:|---|---|
| ap_ft_01 | s0_024 | [24,128,256] | 0 | 1 | 31 | DAIR-V2X val_1789 | true_eval row 或 quarantine |
| ap_ft_02 | s0_040 | [40,128,256] | 1 | 1 | 31 | DAIR-V2X val_1789 | true_eval row 或 quarantine |
| ap_ft_03 | s0_056 | [56,128,256] | 2 | 1 | 31 | DAIR-V2X val_1789 | true_eval row 或 quarantine |
| ap_ft_04 | s1_048 | [64,48,256] | 3 | 1 | 31 | DAIR-V2X val_1789 | true_eval row 或 quarantine |
| ap_ft_05 | s2_160 | [64,128,160] | 4 | 1 | 31 | DAIR-V2X val_1789 | true_eval row 或 quarantine |

GPU 约束:

1. H800 GPU0-4 同时启动, 每张卡一个单卡 DDP 训练进程。
2. `CUDA_VISIBLE_DEVICES` 必须固定到上表 GPU。
3. `master_port` 使用 `29700 + GPU`, 避免五个单卡 DDP 互相冲突。
4. GPU5 默认留给 base/p50/mix_d fresh replay、eval 补救、digest 扫描或 supervisor; 若 GPU5 不可用, eval 等待任意 finetune GPU 释放。
5. AP 微调和 AP eval 不要求 GPU 完全空闲, 但必须记录并发状态; 显存不足/OOM 时只 quarantine 当前配置。

#### 数据集和 split

H800 上必须使用 HEAL/OpenCOOD 当前 DAIR-V2X-C 数据口径:

```text
dataset logical name: DAIR-V2X
dataset physical root: ${V2X_HOME}/heal_research/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure
H800 preferred target: ${V2X_DATA_ROOT}/DAIR-V2X/DAIR-V2X-C/cooperative-vehicle-infrastructure
train split: train.json, historical count ~= 4811
eval split: val.json, canonical name val_1789, num_samples=1789
```

AP canonical row 中必须写:

```text
dataset=DAIR-V2X
eval_split=val_1789
num_samples=1789
metric=AP70
secondary_metrics.AP30
secondary_metrics.AP50
finetune_protocol=structural_prune_pyramid_train_ddp_half_epoches31_TRT_FP16_DAIR_val_1789
schedule_policy=not_applicable
```

不允许用 mini-val、小样本 smoke 或其他 split 的 AP 替代 `val_1789` measured AP。

#### 微调命令模板

每个配置先从 DAIR base checkpoint 结构化剪枝:

```bash
CUDA_VISIBLE_DEVICES=<GPU> \
PYTHONPATH=${V2X_HOME}/heal_research/HEAL \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  tools/structural_prune_pyramid.py \
  --orig-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29 \
  --out-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_<label>_2026_06_26 \
  --num-filters-new <s0>,<s1>,<s2> \
  --width-per-group 4 \
  --groups 32
```

训练前必须确认输出 config 的训练预算为 31:

```text
config.yaml 中 train_params.epoches 或等价字段必须为 31
model_dir 内 init ckpt 必须能被 HEAL load_saved_model 正确加载
net_epoch_bestval_at23.pth 必须是 flat state_dict; 若是 {"model_state_dict": ...} 包裹格式, 先 unwrap 后再训练
```

单卡 DDP 微调模板:

```bash
CUDA_VISIBLE_DEVICES=<GPU> \
PYTHONPATH=${V2X_HOME}/heal_research/HEAL \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  -m torch.distributed.launch \
  --nproc_per_node=1 \
  --use_env \
  --master_port=<29700+GPU> \
  ${V2X_HOME}/heal_research/HEAL/opencood/tools/train_ddp.py \
  --hypes_yaml ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_<label>_2026_06_26/config.yaml \
  --model_dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_<label>_2026_06_26 \
  --half
```

微调完成后必须导出 ONNX、构建 TRT FP16 engine, 再跑 full val AP。若某一步失败, 写:

```text
jobs/ap_finetune_smoke_job_state_v1.jsonl
quarantine/ap_unstable_or_unclaimable_v1.jsonl
raw/<label>/{command.json,stdout.log,stderr.log,manifest.json}
```

#### Phase C AP stable smoke stop 条件

Phase C AP 子阶段只有满足以下条件才算完成:

1. 10 个已有 source/import 配置都有 canonical AP row 或明确 replay/quarantine 状态。
2. 5 个新配置全部完成 `structural_prune_pyramid -> flat ckpt check -> epoches=31 finetune -> ONNX -> TRT FP16 -> DAIR val_1789 AP eval`, 或者每个失败点都有明确 quarantine reason。
3. 至少 3 个 fresh/import replay anchor 通过 `abs(AP70_observed - AP70_expected) <= 0.003`。
4. measured AP row 中 0 条来自 predicted/model-fit/interpolation。
5. `ap_stability_summary.md` 能直接看到 label、width、GPU、finetune 次数、epoches、dataset/split、AP30/AP50/AP70、claim_status。

**Files:**
- Read: `artifacts/artifact_state.jsonl`
- Read: `artifacts/artifact_registry_v1.jsonl`
- Read: existing rows under `generated/overnight_6h_20260626/merged/`
- Write: `jobs/latency_job_queue_gpu0.jsonl`
- Write: `jobs/latency_job_queue_gpu1.jsonl`
- Write: `jobs/latency_job_queue_gpu4.jsonl`
- Write: `jobs/latency_job_queue_gpu5.jsonl`
- Write: `jobs/energy_job_queue.jsonl`
- Write: `jobs/ap_job_queue.jsonl`
- Write: `exports/energy_axis_gap_report.*`
- Write: `exports/ap_axis_gap_report.*`

- [ ] **Step 1: 生成 latency per-GPU queues**

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
MERGED=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged
python3 scripts/stage2_generate_latency_coverage_jobs.py \
  --candidates "$BASE/candidates/candidate_queue.jsonl" \
  --artifact-state "$BASE/artifacts/artifact_state.jsonl" \
  --artifact-registry "$BASE/artifacts/artifact_registry_v1.jsonl" \
  --latency-rows "$MERGED/latency_lut_rows_merged_v1.jsonl" \
  --latency-rows "$BASE/rows/latency_lut_rows_v1.jsonl" \
  --out-dir "$BASE/jobs" \
  --rows-out-jsonl "$BASE/rows/latency_lut_rows_v1.jsonl" \
  --raw-root "$BASE/raw/latency" \
  --gpus 0,1,4,5 \
  --phase coverage_pipeline_v1 \
  --tag coverage \
  --created-at 2026-06-26T00:00:00Z \
  --manifest-path "$BASE/candidates/candidate_queue.jsonl" \
  --registry-path "$BASE/artifacts/artifact_registry_v1.jsonl" \
  --manifest-digest coverage_pipeline_v1 \
  --warmup-iters 1 \
  --measure-iters 500 \
  --repeat 5
```

Gate:

```text
Total latency queue rows >= 12 before starting latency workers.
No latency queue file for GPU2 unless energy-agent is paused explicitly.
```

- [ ] **Step 2: 生成 AP queue 和 AP gap report**

AP 分两类队列:

1. `coverage_pipeline_v1/jobs/ap_job_queue.jsonl`: 只处理已有 source/import 或 no-claim gap report。
2. `ap_stability_20260626/jobs/ap_finetune_smoke_queue_v1.jsonl`: 只处理 Phase C stable smoke 的 5 个新配置微调。

二者不能混用。已有 source/import 可以快速生成 canonical row; 新配置必须走 5.0 节的完整 finetune/eval 链路。

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
MERGED=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged
python3 scripts/stage2_generate_ap_coverage_jobs.py \
  --candidate-queue "$BASE/candidates/candidate_queue.jsonl" \
  --artifact-registry "$BASE/artifacts/artifact_registry_v1.jsonl" \
  --ap-rows "$MERGED/ap_anchor_rows_merged_v1.jsonl" \
  --ap-rows "$BASE/rows/ap_anchor_rows_v1.jsonl" \
  --out-dir "$BASE/jobs" \
  --out-jsonl "$BASE/jobs/ap_job_queue.jsonl" \
  --gap-report-csv "$BASE/exports/ap_axis_gap_report.csv" \
  --gap-report-json "$BASE/exports/ap_axis_gap_report.json" \
  --run-id ap_coverage_pipeline_v1 \
  --created-at 2026-06-26T00:00:00Z \
  --rows-out-jsonl "$BASE/rows/ap_anchor_rows_v1.jsonl" \
  --manifest-path "$BASE/candidates/candidate_queue.jsonl" \
  --registry-path "$BASE/artifacts/artifact_registry_v1.jsonl" \
  --manifest-digest coverage_pipeline_v1
```

Gate:

```text
AP queue can be smaller than latency queue.
Every blocked AP row must have source_missing/no_claim reason.
No predicted AP measured rows are allowed.
Phase C stable smoke new configs must also exist in ap_stability_20260626/jobs/ap_finetune_smoke_queue_v1.jsonl.
```

- [ ] **Step 3: 生成 energy queue 和 energy gap report**

Energy 跟随 latency。第一轮若还没有新 latency rows, 只能对 priority-selected artifact-ready candidate 做小批量 energy; 更稳妥的流程是 latency 第一批完成后再运行本步骤。

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
MERGED=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged
python3 scripts/stage2_generate_energy_coverage_jobs.py \
  --candidates "$BASE/candidates/candidate_queue.jsonl" \
  --artifact-registry "$BASE/artifacts/artifact_registry_v1.jsonl" \
  --latency-rows "$MERGED/latency_lut_rows_merged_v1.jsonl" \
  --latency-rows "$BASE/rows/latency_lut_rows_v1.jsonl" \
  --energy-rows "$MERGED/energy_lut_rows_merged_v1.jsonl" \
  --energy-rows "$BASE/rows/energy_lut_rows_v1.jsonl" \
  --out-queue "$BASE/jobs/energy_job_queue.jsonl" \
  --out-gap-csv "$BASE/exports/energy_axis_gap_report.csv" \
  --out-gap-json "$BASE/exports/energy_axis_gap_report.json" \
  --gpu-id 2 \
  --run-id-prefix energy_coverage_pipeline_v1 \
  --tag coverage \
  --rows-out-jsonl "$BASE/rows/energy_lut_rows_v1.jsonl" \
  --raw-root "$BASE/raw/energy" \
  --phase coverage_pipeline_v1 \
  --manifest-path "$BASE/candidates/candidate_queue.jsonl" \
  --registry-path "$BASE/artifacts/artifact_registry_v1.jsonl" \
  --manifest-digest coverage_pipeline_v1 \
  --energy-warmup-iters 50 \
  --energy-measure-iters 1500
```

Gate:

```text
Energy queue rows >= min(5, completed new latency cells) before starting energy worker.
Telemetry-noisy configs write quality/no-claim, not fake measured rows.
```

## 6. Phase D: 启动多 agent 轮询生产

**Files:**
- Read: `RUNBOOK_stage2_multi_agent_lut_polling_v1_zh.md`
- Write: `jobs/*_job_state.jsonl`
- Write: `raw/latency/`, `raw/energy/`, `raw/ap/`
- Write: `logs/`

- [ ] **Step 1: latency-agent 在 GPU0/1/4/5 启动 worker**

每个 GPU 一个 worker, 每个 worker 串行消费自己的 queue:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
mkdir -p "$BASE/logs"
for GPU in 0 1 4 5; do
  nohup python3 scripts/stage2_lut_worker.py \
    --job-plan "$BASE/jobs/latency_job_queue_gpu${GPU}.jsonl" \
    --job-state "$BASE/jobs/latency_job_state_gpu${GPU}.jsonl" \
    --max-hours 6 \
    --resume \
    --log-dir "$BASE/logs/latency_gpu${GPU}" \
    --quarantine-db "$BASE/quarantine/bad_db_quarantine_v1.jsonl" \
    --require-gpu-idle \
    > "$BASE/logs/latency_gpu${GPU}.out" 2>&1 &
  echo $! > "$BASE/jobs/latency_gpu${GPU}.pid"
done
```

- [ ] **Step 2: AP-agent 启动 import/replay worker**

该 worker 只用于已有 source/import/replay 队列, 不负责 5GPU finetune。AP import/replay 不需要 `--require-gpu-idle`; 只记录 GPU 并发状态和 raw artifact。

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
mkdir -p "$BASE/logs"
CUDA_VISIBLE_DEVICES=5 nohup python3 scripts/stage2_lut_worker.py \
  --job-plan "$BASE/jobs/ap_job_queue.jsonl" \
  --job-state "$BASE/jobs/ap_job_state.jsonl" \
  --max-hours 6 \
  --resume \
  --log-dir "$BASE/logs/ap_import_replay_gpu5" \
  --quarantine-db "$BASE/quarantine/bad_db_quarantine_v1.jsonl" \
  > "$BASE/logs/ap_import_replay_gpu5.out" 2>&1 &
echo $! > "$BASE/jobs/ap_import_replay_gpu5.pid"
```

- [ ] **Step 3: AP-agent 启动 stable smoke 5GPU finetune**

Phase C stable smoke 使用独立队列:

```text
multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626/jobs/ap_finetune_smoke_queue_v1.jsonl
```

启动规则:

```text
GPU0: s0_024, [24,128,256], epoches=31, 1 finetune
GPU1: s0_040, [40,128,256], epoches=31, 1 finetune
GPU2: s0_056, [56,128,256], epoches=31, 1 finetune
GPU3: s1_048, [64,48,256], epoches=31, 1 finetune
GPU4: s2_160, [64,128,160], epoches=31, 1 finetune
```

每个 GPU 只跑自己的单卡 DDP 训练进程; master port 使用 `29700 + GPU`。该步骤不要求 GPU 完全空闲, 但必须在 raw 目录保存启动前的 `nvidia-smi` 和 `pmon` 记录。显存不足/OOM/训练失败只隔离对应配置, 其他 GPU 继续跑。

优先使用专用 runner:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_h800_ap_finetune_smoke_runner.py \
  --base-dir "$BASE" \
  --max-workers 5 \
  --execute \
  > "$BASE/logs/ap_finetune_smoke_runner.out" 2>&1 &
echo $! > "$BASE/jobs/ap_finetune_smoke_runner.pid"
```

该 runner 会消费 `jobs/ap_finetune_smoke_queue_v1.jsonl`, 为每个配置写 `raw/<label>/`、`jobs/ap_finetune_smoke_job_state_v1.jsonl`、`quarantine/ap_unstable_or_unclaimable_v1.jsonl`, 成功后追加 `rows/ap_anchor_rows_v1.jsonl`。如果 runner 尚未同步到 H800, agent 必须先 rsync `scripts/stage2_h800_ap_finetune_smoke_runner.py` 和依赖脚本; 不得把这 5 个新配置交给只会 import/eval 的 generic `stage2_lut_worker.py`。

- [ ] **Step 4: energy-agent 在 latency 第一批完成后启动 worker**

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
mkdir -p "$BASE/logs"
nohup python3 scripts/stage2_lut_worker.py \
  --job-plan "$BASE/jobs/energy_job_queue.jsonl" \
  --job-state "$BASE/jobs/energy_job_state.jsonl" \
  --max-hours 6 \
  --resume \
  --log-dir "$BASE/logs/energy_gpu2" \
  --quarantine-db "$BASE/quarantine/bad_db_quarantine_v1.jsonl" \
  --require-gpu-idle \
  > "$BASE/logs/energy_gpu2.out" 2>&1 &
echo $! > "$BASE/jobs/energy_gpu2.pid"
```

- [ ] **Step 5: supervisor-agent 每 10-30 分钟轮询一次**

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
MERGED=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged
python3 scripts/stage2_supervisor_poll.py \
  --latency-rows "$MERGED/latency_lut_rows_merged_v1.jsonl" \
  --latency-rows "$BASE/rows/latency_lut_rows_v1.jsonl" \
  --ap-rows "$MERGED/ap_anchor_rows_merged_v1.jsonl" \
  --ap-rows "$BASE/rows/ap_anchor_rows_v1.jsonl" \
  --energy-rows "$MERGED/energy_lut_rows_merged_v1.jsonl" \
  --energy-rows "$BASE/rows/energy_lut_rows_v1.jsonl" \
  --job-plan "$BASE/jobs/latency_job_queue_gpu0.jsonl" \
  --job-plan "$BASE/jobs/latency_job_queue_gpu1.jsonl" \
  --job-plan "$BASE/jobs/latency_job_queue_gpu4.jsonl" \
  --job-plan "$BASE/jobs/latency_job_queue_gpu5.jsonl" \
  --job-plan "$BASE/jobs/energy_job_queue.jsonl" \
  --job-plan "$BASE/jobs/ap_job_queue.jsonl" \
  --job-state "$BASE/jobs/latency_job_state_gpu0.jsonl" \
  --job-state "$BASE/jobs/latency_job_state_gpu1.jsonl" \
  --job-state "$BASE/jobs/latency_job_state_gpu4.jsonl" \
  --job-state "$BASE/jobs/latency_job_state_gpu5.jsonl" \
  --job-state "$BASE/jobs/energy_job_state.jsonl" \
  --job-state "$BASE/jobs/ap_job_state.jsonl" \
  --axis-gap-report "$BASE/exports/energy_axis_gap_report.json" \
  --axis-gap-report "$BASE/exports/ap_axis_gap_report.json" \
  --quarantine-rows "$BASE/quarantine/bad_db_quarantine_v1.jsonl" \
  --missing-artifact-rows "$BASE/quarantine/missing_artifact_v1.jsonl" \
  --out-dir "$BASE/exports"
```

Supervisor stop rules:

```text
repeat_ratio remains high and new unique cells do not grow: stop repeat, generate new candidates.
latency_ready_jobs < 10: artifact-agent must top up.
energy/AP cells < 40% latency cells: pause low-value latency expansion and let energy/AP catch up.
quarantine grows: pause affected workdir/template.
```

## 7. Phase E: Top-up 和容错循环

**Files:**
- Modify: `candidates/candidate_queue.jsonl`
- Modify: `artifacts/artifact_state.jsonl`
- Modify: `jobs/*.jsonl`
- Modify: `exports/readiness_gate_latest.json`

- [ ] **Step 1: latency queue 低水位时补 candidate**

触发条件:

```text
latency_ready_jobs < 10
and active quarantine is not growing
and artifact-agent can provide at least 12 ready candidates
```

操作:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
MERGED=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged
python3 scripts/stage2_generate_coverage_candidates.py \
  --limit 60 \
  --existing-jsonl "$MERGED/latency_lut_rows_merged_v1.jsonl" \
  --existing-jsonl "$BASE/rows/latency_lut_rows_v1.jsonl" \
  --out-jsonl "$BASE/candidates/candidate_queue_next.jsonl" \
  --created-at 2026-06-26T00:00:00Z \
  --model Pyramid-LiDAR \
  --schedule-policy metaschedule_tuned \
  --optimized-scope backbone_only
```

把 `candidate_queue_next.jsonl` 通过 artifact-agent gate 后再 append 到主 `candidate_queue.jsonl`; 不直接让未经 artifact gate 的候选进入测量队列。

- [ ] **Step 2: 处理 missing artifact**

若 `missing_artifact_v1.jsonl` 对新候选持续增长:

1. artifact-agent 先从 H800 已有 workdir 生成 ready-first candidate。
2. 对真正缺 ONNX/DB 的候选生成 compile/export task。
3. 编译完成后单点 smoke, 再更新 `artifact_state=ready`。
4. 未完成 smoke 的 artifact 不能进入 latency/energy queue。

- [ ] **Step 3: 处理 outlier 和 bad DB**

若出现 CUDA illegal memory 或 latency outlier:

1. 写 `jobs/*_job_state.jsonl` failed/preflight_blocked。
2. 写 `quarantine/bad_db_quarantine_v1.jsonl` 或 `quarantine/outlier_quarantine_v1.jsonl`。
3. supervisor 从后续 queue 移除同 config/workdir。
4. 只有重建 artifact 并单点复测通过后才能恢复。

## 8. 大规模生产 GO 判定

满足以下条件后, 才能说“可以开始大规模 LUT 完善”:

| Gate | 判定 |
|---|---|
| artifact ready | `artifact_status=ready` 候选 >= 30 |
| latency queue | latency ready jobs >= 20 |
| AP queue/gap | AP jobs 或明确 no-claim 覆盖 >= 10 |
| energy queue/gap | energy jobs 或明确 no-claim 覆盖 >= 10 |
| GPU preflight | latency/energy 目标 GPU 符合 idle 规则; AP finetune/eval 通过 memory/OOM preflight 并记录并发状态 |
| AP claim | 0 个 predicted AP measured row |
| energy claim | telemetry 缺失时写 no-claim, 不写 measured |
| quarantine | active quarantine 未持续增长 |
| repeat | 新增 measured rows 中至少 70% 来自未覆盖 unique axis cell |
| supervisor | readiness gate 至少达到 `CONDITIONAL_GO` |

阶段目标:

```text
Day 0 validation: 12-20 artifact-ready configs, latency 必测, AP/energy subset.
Day 1 coverage: 60-100 unique latency cells, 25-40 energy cells, 25-40 AP/no-claim cells.
Predictor seed: 120-180 unique latency cells, 50-80 energy cells, 50-80 AP/no-claim cells.
Paper-grade subset: 20-40 configs 做 3-5 次 repeat 和 cross-GPU drift check.
```

## 9. Agent 分工提示

artifact-agent:

```text
你负责把 candidate_queue.jsonl 变成 artifact_state/artifact_registry。不要写 measured row。优先在 H800 上发现已有 ONNX/workdir/MetaSchedule DB, 让 ready-first candidates 先进入测量队列。缺 artifact 写 missing_artifact_v1.jsonl; bad DB 写 quarantine。不要记录真实密码。
```

latency-agent:

```text
你只消费 artifact-ready 的 latency_job_queue_gpu0/1/4/5。每个 job 先做 GPU idle preflight, 只写 H800 measured latency row。目标是扩大 unique latency coverage, 不做机械 repeat。失败写 job_state/quarantine 后继续下一项。
```

energy-agent:

```text
你使用 GPU2, 只对 latency 已成功或 priority-selected artifact-ready candidate 跑 energy。telemetry 不稳定写 no-claim/quality flag。energy 与 latency 必须共享同一 candidate/artifact bundle。
```

AP-agent:

```text
你负责真实 AP rows, 包括已有 source/import/replay 和 Phase C stable smoke 微调。已有 source/import/replay 可使用 GPU5 或任一已释放 GPU; Phase C 首批新配置固定用 H800 GPU0-4 五卡并行微调: s0_024->GPU0, s0_040->GPU1, s0_056->GPU2, s1_048->GPU3, s2_160->GPU4。每个新配置默认 1 次 finetune, epoches=31, 使用 DAIR-V2X train split 训练并在 DAIR-V2X val_1789 上评测 AP30/AP50/AP70。AP finetune/eval 不要求 GPU 完全空闲, 但必须记录 nvidia-smi/pmon 和显存状态; OOM 或失败只 quarantine 当前配置。必须保留 dataset/split/ckpt/protocol/source_path。缺 AP source 写 no-claim, 继续下一项。禁止 predicted/model-fit/interpolated AP measured row。
```

supervisor-agent:

```text
你每 10-30 分钟运行 supervisor poll, 审查 unique config/cell、repeat ratio、queue 低水位、quarantine 增长、AP/energy lag。你的输出是 readiness_gate_latest.json、coverage_dashboard_latest.csv、supervisor_report_latest.md 和下一轮调度建议。
```

## 10. 执行后汇报格式

每轮 agent team 汇报必须包含:

```text
readiness decision:
artifact ready / missing / quarantined:
latency ready/running/succeeded/failed:
AP ready/running/succeeded/blocked/no-claim:
energy ready/running/succeeded/blocked/no-claim:
unique latency/AP/energy cells:
new unique cells in this round:
repeat ratio for new rows:
active quarantine:
top missing axes:
next action:
```

禁止只汇报 raw row count。
