# Stage2 Multi-Agent Coverage-First LUT Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一个多 agent 持续轮询的 Stage2 H800 LUT 生产流水线, 以快速扩大搜索空间配置覆盖为第一目标, latency/AP/energy 三轴同步追赶, repeat 只作为质量控制而不是数据量扩张手段。

**Architecture:** 不再依赖“一条长 worker 命令保活”, 而是由 artifact-agent、latency-agent、energy-agent、AP-agent 和 supervisor-agent 持续轮询 durable queues。每个 agent 只负责一个边界清晰的职责, 通过 append-only state、artifact registry、quarantine registry 和 supervisor review table 协作。coverage 指标以 unique config / unique axis cell 为准, row count 只作为 raw evidence 数量。

**Tech Stack:** Python JSONL durable queues, Stage2 LUT productization scripts, H800 TVM/Relax/MetaSchedule, H800 power telemetry, AP eval/import evidence, artifact registry, evidence registry, dmux/Codex 多 agent 会话, rsync/SSH H800 runbook。

---

## 0. 完成状态更新

更新时间: 2026-06-26

本计划的第 8 节“实施任务”已经完成，完成范围是:

1. 建立 coverage-first durable queue 目录契约。
2. 实现 seed coverage summary 与 supervisor dashboard/gate。
3. 实现 FP16 Pyramid backbone-only candidate generator。
4. 实现 artifact-agent task planner 和 artifact registry/state 输出。
5. 实现 latency/AP/energy coverage job generator。
6. 实现 energy/AP axis gap report。
7. 实现 multi-agent polling runbook。
8. 对实现做回归验证和 reviewer 问题修复。

边界说明:

1. 本计划完成的是“多 agent 生产机制”和“可执行队列/门控脚手架”，不是 120-180 个 H800 measured LUT 点已经生成完成。
2. 当前本地 artifact gate 对 60 个新候选全部给出 `artifact_status=missing`，因此生成的 latency/AP/energy job queue 为 0，这是 fail-closed 行为。
3. 下一阶段必须在 H800 上补齐或重新验证 ONNX/TVM workdir/MetaSchedule DB/AP source，使候选进入 `artifact_status=ready` 后再启动测量。
4. 新的执行计划见 `PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md`。

当前 gate snapshot:

| 指标 | 值 |
|---|---:|
| readiness decision | `NO_GO` |
| raw measured rows | 1318 |
| unique config count | 113 |
| unique latency cells | 70 |
| unique AP cells | 22 |
| unique energy cells | 21 |
| repeat ratio | 0.914 |
| generated coverage candidates | 60 |
| local missing artifact configs | 60 |
| latency/AP/energy ready jobs | 0 / 0 / 0 |

已验证命令:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s framework/tests -p 'test_stage2_*coverage*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest framework.tests.test_stage2_artifact_task_planner framework.tests.test_stage2_lut_productization framework.tests.test_stage2_large_scale_readiness framework.tests.test_stage2_supervisor_poll -v
python3 -m py_compile scripts/stage2_summarize_lut_coverage.py scripts/stage2_supervisor_poll.py scripts/stage2_generate_coverage_candidates.py scripts/stage2_plan_artifact_tasks.py scripts/stage2_generate_latency_coverage_jobs.py scripts/stage2_generate_energy_coverage_jobs.py scripts/stage2_generate_ap_coverage_jobs.py
```

## 1. 为什么 800+ rows 仍然核心数据有限

当前 overnight 结果:

| 数据 | rows | unique label | unique key | 说明 |
|---|---:|---:|---:|---|
| latency | 888 merged | 18 | 70 | 大量 continuation repeat; 多数核心 label 有约 25 次重复 |
| energy | 408 merged | 16 | 21 | 16 个配置约 25 次重复 |
| AP | 22 merged | 11 | 22 | AP 是 true anchor import, 没有盲目 repeat |

结论:

1. 800+ 行主要证明测量链路、稳定性和 quarantine 能工作。
2. 它不能等价为 800+ 个搜索空间配置。
3. 对性能预测器最有价值的是 distinct config 覆盖, 不是同一配置 24 轮 repeat。
4. 下一阶段的优化目标必须从 `row_count` 改成 `unique_config_count`、`axis_cell_coverage`、`search_space_spread`。

## 2. 新生产原则

### 2.1 Coverage-first

默认目标:

| 阶段 | unique config 目标 | latency | energy | AP | repeat 上限 |
|---|---:|---|---|---|---:|
| coverage smoke | 12-20 | 必测 | subset | subset/import | 1 |
| coverage calibration | 60-100 | 必测 | 25-40 selected | 25-40 selected | 1-2 |
| predictor seed | 120-180 | 必测 | 50-80 selected | 50-80 selected | 1-2 |
| paper-grade | 20-40 | 必测 | 必测 | 必测 | 3-5 |

硬规则:

1. coverage 阶段同一 `(model,width,quant_policy,schedule_policy,optimized_scope,backend)` 的 measured repeat 默认不超过 2。
2. 只有 `paper_retest`, `outlier_retest`, `cross_gpu_drift_check` 三类显式 tag 可以突破 repeat 上限。
3. supervisor 每轮审查 repeat ratio: 新增 measured rows 中至少 70% 必须来自此前未覆盖的 unique axis cell。
4. continuation queue 禁止机械复制全量成功 job 24 轮。

### 2.2 统一 candidate 语义

latency、energy、AP 必须消费同一个 candidate/config bundle:

```text
candidate_id
model
width / pruning policy / quant policy / schedule policy
optimized_scope
source manifest / ckpt / ONNX / TVM workdir
```

其中:

1. latency 和 energy 共享同一 H800 TVM build/runtime artifact, 只是 measurement method 不同。
2. AP 使用同一 candidate 对应的 ckpt/eval protocol, 不允许用语义不同的模型配置拼接。
3. AP 若只能 import historical true anchor, 必须保留 source/provenance; 不能写 predicted measured AP。
4. energy 若 telemetry 不稳定, 写 no-claim/quality flag, 不阻塞 latency coverage。

### 2.3 行数和配置数分离

所有 dashboard 必须同时显示:

```text
raw_row_count
unique_config_count
unique_latency_cells
unique_energy_cells
unique_ap_cells
repeat_rows
repeat_ratio
quarantined_config_count
blocked_config_count
```

不允许只用 `latency rows = N` 作为进展汇报。

## 3. Durable Queue 目录结构

建议新目录:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/
  candidates/
    candidate_queue.jsonl
    candidate_state.jsonl
    search_space_frontier.jsonl
  artifacts/
    artifact_tasks.jsonl
    artifact_state.jsonl
    artifact_registry_v1.jsonl
  jobs/
    latency_job_queue.jsonl
    latency_job_state.jsonl
    energy_job_queue.jsonl
    energy_job_state.jsonl
    ap_job_queue.jsonl
    ap_job_state.jsonl
  leases/
    artifact_agent.lock.json
    latency_agent.lock.json
    energy_agent.lock.json
    ap_agent.lock.json
  rows/
    latency_lut_rows_v1.jsonl
    energy_lut_rows_v1.jsonl
    ap_anchor_rows_v1.jsonl
  quarantine/
    bad_db_quarantine_v1.jsonl
    missing_artifact_v1.jsonl
    outlier_quarantine_v1.jsonl
  raw/
    latency/
    energy/
    ap/
  exports/
    coverage_dashboard_latest.csv
    quick_review_latest.csv
    readiness_gate_latest.json
    supervisor_report_latest.md
```

## 4. Queue Schema 草案

### 4.1 Candidate row

```json
{
  "schema": "stage2_candidate_row_v1",
  "candidate_id": "coverage:pyramid_lidar:w32x96x192:fp16:metaschedule_tuned",
  "model": "Pyramid-LiDAR",
  "label": "mix_a",
  "width": [32, 96, 192],
  "arm": "P",
  "quant_policy": "fp16",
  "schedule_policy": "metaschedule_tuned",
  "optimized_scope": "backbone_only",
  "priority": 80,
  "axes_required": ["latency", "energy", "ap"],
  "repeat_policy": "coverage",
  "max_latency_repeats": 1,
  "max_energy_repeats": 1,
  "ap_policy": "true_eval_or_true_import_only",
  "created_at": "2026-06-26T00:00:00Z"
}
```

### 4.2 Artifact state

```json
{
  "schema": "stage2_artifact_state_v1",
  "candidate_id": "coverage:pyramid_lidar:w32x96x192:fp16:metaschedule_tuned",
  "artifact_status": "ready",
  "onnx_path": "/exdata/jichengzhi/s2_tvm/models/mix_a_backbone.onnx",
  "tvm_work_dir": "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_a",
  "database_workload_path": "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_a/database_workload.json",
  "database_tuning_record_path": "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_a/database_tuning_record.json",
  "ap_source_path": "multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl",
  "quality_status": "ready",
  "failure_reason": null,
  "updated_at": "2026-06-26T00:00:00Z"
}
```

### 4.3 Lease row

每个 agent 轮询前用 `flock` 或 atomic rename 获取短租约:

```json
{
  "schema": "stage2_agent_lease_v1",
  "agent": "latency-agent",
  "owner": "hostname:pid",
  "lease_started_at": "2026-06-26T00:00:00Z",
  "lease_expires_at": "2026-06-26T00:10:00Z",
  "current_task": "latency_job_queue.jsonl:job_id"
}
```

规则:

1. lease TTL 默认 10 分钟。
2. supervisor 可以回收超时 lease。
3. agent 写 state 时必须包含 `agent`, `lease_id`, `attempt`。

## 5. Agent 职责

### 5.1 artifact-agent

目标: 让 candidate 先变成 artifact-ready, 避免 latency/energy/AP agent 在运行时才发现缺 ONNX/DB/source。

输入:

```text
candidates/candidate_queue.jsonl
quarantine/*.jsonl
```

输出:

```text
artifacts/artifact_state.jsonl
artifacts/artifact_registry_v1.jsonl
quarantine/missing_artifact_v1.jsonl
```

职责:

1. 检查 ONNX 是否存在。
2. 检查 TVM workdir/MetaSchedule DB 是否存在且 schema 可读。
3. 检查 AP true source 或 eval command 是否存在。
4. 对缺 artifact 的 candidate 写 `artifact_status=missing`, 不进入测量队列。
5. 对已知 bad DB 的 candidate 写 `artifact_status=quarantined`, 不进入 continuation。
6. 对新 width 负责触发导出/编译/调度数据库生成任务, 但生成后必须先单点 smoke 验证。

质量门:

| 检查 | 失败处理 |
|---|---|
| ONNX missing | `missing_artifact` |
| TVM DB missing | `missing_artifact` 或 `needs_tuning` |
| DB 读失败 | `quarantined_bad_db` |
| AP source missing | AP axis blocked, latency 可继续 |
| known illegal memory config | artifact quarantined, 不进测量队列 |

### 5.2 latency-agent

目标: 快速扩大 search space 的 latency 覆盖面积。

输入:

```text
artifacts/artifact_registry_v1.jsonl
jobs/latency_job_queue.jsonl
rows/latency_lut_rows_v1.jsonl
```

输出:

```text
rows/latency_lut_rows_v1.jsonl
jobs/latency_job_state.jsonl
raw/latency/<run_id>/
quarantine/bad_db_quarantine_v1.jsonl
```

职责:

1. 只消费 artifact-ready candidate。
2. 每个 unique config 默认跑 default + tuned 各 1 次。
3. 发现 coverage gap 时向 supervisor 请求新 candidate, 不自行复制 24 轮 repeat。
4. 保留 GPU idle preflight raw。
5. 发生 CUDA illegal memory 时写 quarantine, 并通知 artifact-agent 更新 artifact_status。
6. latency row 使用 ms 汇报, raw 可以保留 us。

coverage 策略:

1. 先覆盖 FP16 P/S width space。
2. stage-axis 和 coupled-axis 交错采样, 避免只测 p50/p75。
3. 每 20 个新 unique latency config 后, supervisor 选择 3-5 个做 energy/AP 补齐。
4. 对预测器而言, 比起同一点 24 repeats, 更优先新增邻域点:
   ```text
   s0 sweep: [16,24,32,40,48,56,64] x fixed s1/s2
   s1 sweep: fixed s0/s2 x [32,48,64,80,96,112,128]
   s2 sweep: fixed s0/s1 x [64,96,128,160,192,224,256]
   coupled samples: Latin-hypercube / frontier-neighborhood
   ```

### 5.3 energy-agent

目标: 使用与 latency 相同的 candidate/artifact bundle 生成 H800 telemetry energy rows, 并建立 energy quality gate。

输入:

```text
artifacts/artifact_registry_v1.jsonl
rows/latency_lut_rows_v1.jsonl
jobs/energy_job_queue.jsonl
```

输出:

```text
rows/energy_lut_rows_v1.jsonl
jobs/energy_job_state.jsonl
raw/energy/<run_id>/
exports/energy_quality_report_latest.json
```

职责:

1. 只对 latency 已成功或 artifact-ready 的 candidate 跑 energy。
2. 共享 latency 的 TVM workdir/VM artifact; 不使用语义不同的 engine。
3. 每个 coverage config 默认 1 次 energy; 仅 paper/quality retest 才重复。
4. 保存 idle power、active power samples、sample window、GPU id、preflight。
5. 建立 per-config energy sanity:
   - `joule_per_inference >= 0`
   - active watt 与 idle watt 差值合理
   - sample count 达标
   - 同 config repeat CV 超阈值则 `no_claim`

energy quality status:

| 状态 | 含义 |
|---|---|
| `claimable` | telemetry/schema/preflight/repeat quality 通过 |
| `calibration_only` | 可用于 predictor, 不进 paper claim |
| `no_claim` | 数据存在但质量或环境不足 |
| `failed` | telemetry 或 TVM runtime 失败 |

### 5.4 AP-agent

目标: 使用同一 candidate/config 生成或导入真实 AP row, 并负责 AP 质量门控。

输入:

```text
artifacts/artifact_registry_v1.jsonl
jobs/ap_job_queue.jsonl
```

输出:

```text
rows/ap_anchor_rows_v1.jsonl
jobs/ap_job_state.jsonl
raw/ap/<run_id>/
exports/ap_quality_report_latest.json
```

职责:

1. AP row 必须记录 dataset/split/ckpt/finetune protocol。
2. true import 可以用, 但必须 source-traceable。
3. predicted AP 只能 report-only, 不能写 measured row。
4. AP source missing 时写 failed/no-claim, 不阻断 latency coverage。
5. AP-agent 与 latency-agent 使用同一 candidate id; 不能自己改 width/ckpt 后仍写同 config。

AP 优先补齐:

```text
pad64, s0_16, s0_32, s1_32, s2_64, s2_128, mix_c, mix_e
```

### 5.5 supervisor-agent

目标: 持续审查各 agent 的工作, 防止重复测量膨胀、artifact 问题外溢、以及 silent failure。

输入:

```text
candidates/*.jsonl
artifacts/*.jsonl
jobs/*_state.jsonl
rows/*.jsonl
quarantine/*.jsonl
exports/*.json/jsonl/csv
```

输出:

```text
exports/coverage_dashboard_latest.csv
exports/supervisor_report_latest.md
exports/readiness_gate_latest.json
```

职责:

1. 每 10-30 分钟轮询一次, 不高频 SSH 打爆 H800 MaxStartups。
2. 汇总 latest status, 不被旧 `running` state 误导。
3. 计算 unique coverage 和 repeat ratio。
4. 若 repeat ratio > 30%, 暂停 repeat/top-up, 要求 artifact-agent 生成新 candidate。
5. 若某 config 被 quarantine, 将其从后续 queue 移除。
6. 若 latency 队列低于阈值, 生成新的 coverage candidate request。
7. 若 energy/AP 落后 latency coverage 太多, 调整优先级让 energy/AP 追赶。
8. 每次向用户汇报必须同时给:
   ```text
   unique configs
   latency cells
   AP cells
   energy cells
   repeat ratio
   blocked/quarantined count
   top missing axes
   ```

## 6. 多 agent 运行模式

推荐用 dmux 或多个 Codex/Claude 会话, 每个 agent 一个 pane/session。

### 6.1 dmux pane 分工

```text
Pane 1: supervisor-agent
Pane 2: artifact-agent
Pane 3: latency-agent
Pane 4: energy-agent
Pane 5: AP-agent
```

### 6.2 各 pane 启动提示模板

artifact-agent:

```text
你是 Stage2 H800 LUT artifact-agent。先阅读 RUNBOOK_stage2_h800_server_access_v1_zh.md 和 PLAN_stage2_multi_agent_lut_coverage_pipeline_v1_zh.md。你的唯一职责是轮询 coverage_pipeline_v1/candidates, 为 candidate 准备/验证 ONNX、TVM workdir、MetaSchedule DB、AP source, 更新 artifact_state 和 artifact_registry。不要写 latency/AP/energy measured row。遇到 missing/bad DB 写 quarantine/block, 不要静默跳过。
```

latency-agent:

```text
你是 Stage2 H800 LUT latency-agent。先阅读 H800 runbook 和 coverage pipeline plan。只消费 artifact-ready candidate, 使用 generate_latency_lut 主路径在 H800 上生成 latency rows。目标是扩大 unique config coverage, coverage 阶段同一 config 不超过 1 次 default+tuned; 不要机械 repeat。每 job 必须 GPU idle preflight, 失败写 job_state/quarantine 后继续。默认使用 H800 GPU0/1/4/5; GPU2 留给 energy-agent, 只有 energy 暂停或显式 override 时才把 GPU2 加入 latency 队列。
```

energy-agent:

```text
你是 Stage2 H800 LUT energy-agent。先阅读 H800 runbook 和 coverage pipeline plan。使用与 latency 相同的 candidate/artifact bundle 和 H800 TVM runtime 生成 energy telemetry rows。负责 energy quality gate, 缺质量时写 no-claim/quality flag。不要因为一个 config 失败停止队列。注意你可以使用H800服务器上的GPU=2
```

AP-agent:

```text
你是 Stage2 H800 LUT AP-agent。先阅读 H800 runbook 和 coverage pipeline plan。为 artifact-ready candidate 生成或导入真实 AP rows。必须保留 dataset/split/ckpt/finetune protocol。禁止把 predicted AP 写为 measured。缺 AP source 写 failed/no-claim, 继续下一 candidate。注意你可以使用H800服务器上的GPU=3
```

supervisor-agent:

```text
你是 Stage2 H800 LUT supervisor-agent。先阅读 H800 runbook、overnight handoff 和 coverage pipeline plan。每 10 分钟轮询各队列/state/rows/quarantine/export, 计算 unique coverage 和 repeat ratio, 审查各 agent 是否偏离 coverage-first。发现 queue 太短时生成 candidate request; 发现 repeat 过多时暂停 repeat; 发现 quarantined config 时从后续 queue 移除。你的输出是 coverage dashboard、supervisor report 和下一轮调度建议。
```

## 7. 容错机制

| 故障 | 责任 agent | 处理 |
|---|---|---|
| SSH rate limit | supervisor/all | 15-30 秒退避, 合并检查命令, 禁止并行 SSH 风暴 |
| GPU busy | latency/energy | 写 `preflight_blocked`, 延迟重试, 不写 measured row |
| ONNX missing | artifact | `artifact_status=missing`, 不入测量队列 |
| TVM DB missing | artifact | `needs_tuning`, 由 artifact-agent 生成/调度 |
| CUDA illegal memory | latency/energy | failed state + bad_db quarantine + artifact quarantined |
| AP source missing | AP | AP failed/no-claim, latency/energy 可继续 |
| energy telemetry noisy | energy | calibration_only/no_claim, 入 quiet retest queue |
| old running state 干扰 | supervisor | 使用 latest-status summarizer |
| queue 过短 | supervisor | 生成新 candidate request, 不靠 24 轮 repeat 保活 |
| agent crash | supervisor | stale lease 回收, 重启对应 pane/session |

## 8. 实施任务

### Task 1: 建立 coverage pipeline 目录和 seed dashboard

**Files:**
- Create: `multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/README.md`
- Create: `scripts/stage2_summarize_lut_coverage.py`
- Test: `framework/tests/test_stage2_lut_coverage_summary.py`

- [x] **Step 1: 写 coverage summary 测试**

验证同一 config 25 次 repeat 只计为 1 个 unique cell, 并输出 repeat ratio。

- [x] **Step 2: 实现 `stage2_summarize_lut_coverage.py`**

输入 latency/AP/energy JSONL, 输出:

```json
{
  "raw_row_count": 888,
  "unique_config_count": 18,
  "unique_latency_cells": 70,
  "repeat_rows": 818,
  "repeat_ratio": 0.921
}
```

- [x] **Step 3: 用 overnight 数据跑 summary**

```bash
cd /home/jichengzhi/V2X
python3 scripts/stage2_summarize_lut_coverage.py \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/latency_lut_rows_merged_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/ap_anchor_rows_merged_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/energy_lut_rows_merged_v1.jsonl \
  --out-json multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/coverage_summary_seed.json
```

Expected: 显示 latency rows 远大于 unique configs, 并标记 repeat-heavy。

### Task 2: 建立 candidate queue generator

**Files:**
- Create: `scripts/stage2_generate_coverage_candidates.py`
- Test: `framework/tests/test_stage2_coverage_candidates.py`

- [x] **Step 1: 写 candidate schema 测试**

测试 candidate row 包含 `candidate_id/model/width/arm/quant_policy/schedule_policy/axes_required/max_*_repeats`。

- [x] **Step 2: 实现 FP16 coverage candidate generator**

第一版只生成 Pyramid FP16 backbone-only:

```text
single-axis s0/s1/s2 sweep
coupled Latin-hypercube samples
frontier-neighborhood samples
```

- [x] **Step 3: 输出 `candidate_queue.jsonl`**

默认生成 60 个 unique candidates, 与 existing labels 去重。

### Task 3: artifact-agent job planner

**Files:**
- Create: `scripts/stage2_plan_artifact_tasks.py`
- Reuse: `framework/stage2/artifact_registry.py`
- Test: `framework/tests/test_stage2_artifact_task_planner.py`

- [x] **Step 1: 从 candidate queue 生成 artifact tasks**

每个 candidate 输出一个 artifact task, 检查 ONNX/workdir/AP source。

- [x] **Step 2: 缺 artifact 不进入测量 job queue**

`p50b2_136` 这类缺 ONNX 配置应输出 `artifact_status=missing`, 不生成 latency job。

- [x] **Step 3: bad DB 不进入 continuation**

`mix_a/pad64/s1_64` 若在 quarantine, artifact status 为 `quarantined`。

### Task 4: coverage-first latency job generator

**Files:**
- Create: `scripts/stage2_generate_latency_coverage_jobs.py`
- Test: `framework/tests/test_stage2_latency_coverage_jobs.py`

- [x] **Step 1: 输入 artifact-ready candidates**

只为 `artifact_status=ready` 的 candidate 生成 latency jobs。

- [x] **Step 2: repeat 上限生效**

coverage 阶段已存在 latency measured cell 时, 不再生成重复 job, 除非 tag 是 `paper_retest/outlier_retest/cross_gpu_drift_check`。

- [x] **Step 3: 输出 per-GPU job queues**

默认按 GPU0/1/4/5 分配, 每 GPU 内部串行。GPU2 默认留给 energy-agent; 只有 energy 队列暂停时才可显式加入 latency 队列。

### Task 5: energy/AP axis followers

**Files:**
- Create: `scripts/stage2_generate_energy_coverage_jobs.py`
- Create: `scripts/stage2_generate_ap_coverage_jobs.py`
- Test: `framework/tests/test_stage2_energy_coverage_jobs.py`
- Test: `framework/tests/test_stage2_ap_coverage_jobs.py`

- [x] **Step 1: energy 跟随 latency/artifact**

仅对 latency 已成功或 priority-selected artifact-ready candidate 生成 energy job。

- [x] **Step 2: AP 只生成 true eval/import jobs**

缺 AP source 的 candidate 写 AP blocked/no-claim, 不生成 fake measured job。

- [x] **Step 3: 生成 axis gap report**

输出每个 candidate 的 missing axes:

```text
candidate_id, latency_status, energy_status, ap_status, next_action
```

### Task 6: supervisor polling report

**Files:**
- Create: `scripts/stage2_supervisor_poll.py`
- Test: `framework/tests/test_stage2_supervisor_poll.py`

- [x] **Step 1: latest-status summarizer**

对 append-only job_state, 每个 `job_id` 只取最后一条状态。

- [x] **Step 2: coverage dashboard**

输出:

```text
unique configs
latency cells
AP cells
energy cells
repeat ratio
blocked/quarantined count
top missing axes
```

- [x] **Step 3: supervisor action recommendations**

规则:

| 条件 | 建议 |
|---|---|
| repeat ratio > 30% | stop repeat, generate new candidates |
| latency queue < 10 ready jobs | ask artifact-agent for next batch |
| energy coverage < 40% latency coverage | prioritize energy subset |
| AP coverage < 40% latency coverage | prioritize AP source/eval |
| quarantine count rises | pause related workdir/template |

### Task 7: 多 agent runbook

**Files:**
- Create: `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_multi_agent_lut_polling_v1_zh.md`

- [x] **Step 1: 写 dmux pane 启动说明**

包含 artifact/latency/energy/AP/supervisor 五个 pane 的 prompt。

- [x] **Step 2: 写轮询节奏**

建议:

```text
artifact-agent: 15 min
latency-agent: queue-driven, 每批 10-20 jobs
energy-agent: queue-driven, 每批 5-10 jobs
AP-agent: source-driven, 每批 5-10 jobs
supervisor-agent: 10-30 min
```

- [x] **Step 3: 写停止/暂停条件**

包括 SSH rate limit、GPU busy、illegal memory、repeat ratio 超限、readiness NO_GO。

## 9. 下一轮执行建议

立即不要再做 24 轮 continuation。下一轮应按以下顺序:

1. 运行 coverage summary, 给当前数据打上 repeat-heavy 标签。
2. 生成 60 个新的 FP16 coverage candidates。
3. artifact-agent 先筛掉 missing/quarantined。
4. latency-agent 跑第一批 20-30 个 unique configs。
5. energy-agent/AP-agent 只追赶 priority subset。
6. supervisor 每轮输出 coverage dashboard, 用 unique coverage 决定是否继续。

短期目标:

```text
unique latency configs: 18 -> 60+
unique energy configs: 16 -> 30+
unique AP configs: 11 -> 25+
repeat ratio: >80% -> <30%
artifact quarantined configs: 明确隔离, 不再反复进队列
```

最终目标:

```text
建立 120-180 个 unique config 的 H800 LUT predictor seed,
其中至少 50-80 个有 energy,
至少 50-80 个有 AP 或明确 no-claim,
并保留 20-40 个 paper-grade repeat config。
```
