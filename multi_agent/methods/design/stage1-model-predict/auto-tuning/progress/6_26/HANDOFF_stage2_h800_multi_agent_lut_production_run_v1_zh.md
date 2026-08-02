# Stage2 H800 Multi-Agent LUT Production Run Handoff v1

日期: 2026-06-26

对应计划: `PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md`

## 1. 本轮目标

按照计划第 9 节的 agent 分工, 启动 artifact-agent、latency-agent、energy-agent、AP-agent、supervisor-agent, 执行第 3-7 节 Phase A-E:

1. Phase A: 同步脚本并做 H800 preflight。
2. Phase B: H800 artifact discovery 与候选重锚。
3. Phase C: 生成非空 latency/AP/energy 队列。
4. Phase D: 启动多 agent 轮询生产。
5. Phase E: Top-up 和容错循环。

## 2. Agent 分工和结论

| agent | 角色 | 结论 |
|---|---|---|
| Nash | artifact-agent | `DONE_WITH_CONCERNS`: 本地 artifact planner 可运行, 但 60 个候选全部 `artifact_status=missing`; 需要 H800 discovery 或重锚到 known-good workdir |
| Euler | latency-agent | `BLOCKED / NO_GO`: latency GPU0/1/4/5 queue 全 0, 因为无 artifact-ready candidate |
| Darwin | energy-agent | `BLOCKED / NO_GO`: energy queue 0, gap 60, 全部 `fix_missing_artifact`; GPU2 规则确认未被 latency 占用 |
| Ampere | AP-agent | `BLOCKED / NO_GO`: AP queue 0, gap 60, 全部 no-claim blocked; 未写 predicted AP measured row |
| Rawls | supervisor-agent | `NO_GO`: Phase A 凭据/免密 SSH 阻塞, 后续 B-E 不能进入真实生产 |

## 3. 当前执行结果

本地已重新执行并刷新:

```text
scripts/stage2_plan_artifact_tasks.py
scripts/stage2_generate_latency_coverage_jobs.py
scripts/stage2_generate_ap_coverage_jobs.py
scripts/stage2_generate_energy_coverage_jobs.py
scripts/stage2_supervisor_poll.py
```

当前产物路径:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/
```

核心统计:

| 指标 | 当前值 |
|---|---:|
| candidate rows | 60 |
| artifact_state rows | 60 |
| artifact_status=ready | 0 |
| artifact_status=missing | 60 |
| AP source_missing | 60 |
| missing_artifact rows | 60 |
| latency GPU0/1/4/5 queue rows | 0 / 0 / 0 / 0 |
| AP queue rows | 0 |
| energy queue rows | 0 |
| supervisor decision | `NO_GO` |
| raw measured rows in merged evidence | 1318 |
| unique config count | 113 |
| unique latency/AP/energy cells | 70 / 22 / 21 |
| repeat ratio | 0.914 |

Top missing axes:

| axis | status | count |
|---|---|---:|
| AP | blocked | 60 |
| energy | blocked | 60 |
| latency | missing | 60 |

## 4. Phase A-E 状态

### Phase A: 同步脚本并做 H800 preflight

状态: `BLOCKED`

原因:

1. 本地 `password-based SSH (disabled; use an SSH key)` 和 `H800_PASSWORD` 环境变量均不存在。
2. `ssh -o BatchMode=yes` 免密登录失败: `Permission denied (publickey,password)`。
3. 按 runbook 安全规则, 本轮没有在命令、日志或文件中写入真实密码。

结论:

1. 未执行 rsync。
2. 未执行 H800 GPU preflight。
3. 未启动任何 H800 worker。

### Phase B: H800 artifact discovery 与候选重锚

状态: `本地 fail-closed 完成, H800 discovery BLOCKED`

本地结果:

```text
candidates=60
artifact_status missing=60
ap_status source_missing=60
```

关键原因:

1. 当前 `candidate_queue.jsonl` 没有显式 `onnx_path/tvm_work_dir/database_*`。
2. planner 只能回退到 `${V2X_DATA_ROOT}/s2_tvm/models/<label>_backbone.onnx` 和 `${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>`。
3. 本地没有 H800 `${V2X_DATA_ROOT}/s2_tvm`, 因此本地只能得到 missing。
4. 需要在 H800 上生成 artifact inventory, 或把候选重锚到 overnight registry 中的 known-good workdir。

### Phase C: 生成非空 latency/AP/energy 队列

状态: `fail-closed 完成`

结果:

```text
latency jobs=0
AP jobs=0, blocked_no_claim=60
energy jobs=0, gap_rows=60
```

结论:

1. 队列没有伪造任务。
2. AP 没有 predicted measured row。
3. energy 没有 measured row claim。
4. latency 没有绕过 artifact-ready gate。

### Phase D: 启动多 agent 轮询生产

状态: `NOT_STARTED`

原因:

1. Phase A 未解锁。
2. Phase B 没有 artifact-ready candidate。
3. Phase C 三轴队列均为空。

结论:

不能启动 worker。启动空队列或 missing artifact 队列没有意义, 也会掩盖真实阻塞。

### Phase E: Top-up 和容错循环

状态: `NOT_STARTED`

原因:

1. top-up 只能在 artifact-agent gate 后 append。
2. 当前没有至少 12 个 ready candidates。
3. 当前首要任务不是继续生成更多未重锚 candidate, 而是让已有 candidate 变成 artifact-ready 或生成 ready-first candidate。

## 5. 下一步解阻路径

### Step 1: 解锁 H800 访问

用户需要提供一种不落盘的临时登录方式:

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
```

命令执行完成后必须:

```bash
unset password-based SSH (disabled; use an SSH key)
```

不允许把真实密码写入仓库、Markdown、JSONL、job plan 或日志。

### Step 2: H800 单连接 preflight

在 H800 可登录后先执行:

```bash
cd ${V2X_ROOT}
hostname
pwd
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv
nvidia-smi pmon -c 1
```

Gate:

```text
hostname contains <PRIVATE_HOST>
pwd is ${V2X_ROOT}
GPU0/1/2/3/4/5 satisfy idle rules
```

### Step 3: H800 artifact discovery

在 H800 上执行:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
mkdir -p "$BASE/artifacts" "$BASE/quarantine"

find ${V2X_DATA_ROOT}/s2_tvm \
  \( -name '*.onnx' -o -name 'database_workload.json' -o -name 'database_tuning_record.json' \) \
  -print | sort > "$BASE/artifacts/h800_artifact_inventory_v1.txt"

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

Gate:

```text
artifact_status=ready >= 12 -> 可以进入 Phase C 小批量
artifact_status=ready >= 30 -> 满足大规模生产 artifact gate
artifact_status=ready = 0 -> 不启动 workers, 先重锚候选或生成 compile/export tasks
```

### Step 4: 重新生成队列

只有 artifact-ready 后才重新运行:

```text
stage2_generate_latency_coverage_jobs.py
stage2_generate_ap_coverage_jobs.py
stage2_generate_energy_coverage_jobs.py
stage2_supervisor_poll.py
```

Gate:

```text
latency ready jobs >= 12 才能启动 latency small batch
AP jobs 或 no-claim coverage >= 10
energy jobs 或 no-claim coverage >= 10
readiness gate 至少 CONDITIONAL_GO
```

## 6. 本轮安全和质量结论

1. 未写入真实密码。
2. 未生成 H800 measured row。
3. 未启动远端 worker。
4. 未把 predicted AP 写成 measured AP。
5. 未把 telemetry 缺失写成 measured energy。
6. queue generator fail-closed 行为正常。
7. 当前阻塞不是脚本执行失败, 而是 H800 访问和 artifact-ready 缺失。

## 7. 清上下文后启动顺序

先读:

```text
RUNBOOK_stage2_h800_server_access_v1_zh.md
PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md
HANDOFF_stage2_h800_multi_agent_lut_production_run_v1_zh.md
```

然后只做三件事:

1. 用不落盘方式解锁 H800 登录。
2. 在 H800 上做 artifact discovery/re-anchor。
3. artifact-ready 后重新生成非空三轴队列, 再启动 workers。
