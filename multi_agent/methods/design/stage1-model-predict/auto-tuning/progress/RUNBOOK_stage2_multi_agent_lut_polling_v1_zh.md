# Stage2 H800 LUT 多 Agent 轮询 Runbook v1

## 目标

本 runbook 用于把 Stage2 LUT 生成从“长 worker 命令保活”切换为 coverage-first 多 agent 轮询。所有 state/row/quarantine 文件都按 append-only 写入，supervisor 每轮只看 latest state 和 unique coverage。

安全边界:

- 不在 prompt、日志、JSONL、Markdown 中记录真实密码。
- 示例密码只能写成 `<H800_PASSWORD>`。
- 不把 predicted AP 写为 measured AP。
- 不因单个 H800 job 失败停止全局流水线；失败写 state/quarantine 后继续下一项。

## dmux 启动

建议在仓库根目录启动:

```bash
cd ${V2X_ROOT}
dmux new stage2-lut-coverage
```

五个 pane 分工如下。

### Pane 1: artifact-agent

Prompt:

```text
你是 Stage2 H800 LUT artifact-agent。先阅读 RUNBOOK_stage2_h800_server_access_v1_zh.md、PLAN_stage2_multi_agent_lut_coverage_pipeline_v1_zh.md 和 coverage_pipeline_v1/README.md。你的唯一职责是轮询 coverage_pipeline_v1/candidates，为 candidate 准备或验证 ONNX、TVM workdir、MetaSchedule DB、AP source，并更新 artifact_state/artifact_registry。不要写 latency/AP/energy measured row。遇到 missing artifact 或 bad DB，写 quarantine/block state，不要静默跳过。任何 H800 登录示例只允许使用 <H800_PASSWORD>，不要记录真实密码。
```

### Pane 2: latency-agent

Prompt:

```text
你是 Stage2 H800 LUT latency-agent。先阅读 H800 access runbook、coverage pipeline plan 和 coverage_pipeline_v1/README.md。只消费 artifact-ready candidate，使用 stage2_generate_latency_lut 主路径生成 latency rows。目标是扩大 unique latency config/cell coverage；coverage 阶段同一 cell 默认不做 repeat，除非 tag 是 paper_retest/outlier_retest/cross_gpu_drift_check。每个 job 必须 GPU idle preflight，失败写 job_state/quarantine 后继续。默认使用 H800 GPU=0,1,4,5；只有 energy-agent 未占用 GPU2 时才可显式把 GPU2 加入 latency 队列。不要写或记录真实密码，示例只能使用 <H800_PASSWORD>。
```

### Pane 3: energy-agent

Prompt:

```text
你是 Stage2 H800 LUT energy-agent。先阅读 H800 access runbook、coverage pipeline plan 和 coverage_pipeline_v1/README.md。使用与 latency 相同的 candidate/artifact bundle 和 H800 TVM runtime 生成 energy telemetry rows。负责 energy quality gate，telemetry 不稳定时写 no-claim/quality flag，不伪造 measured energy。不要因为一个 config 失败停止队列。默认使用 H800 GPU=2，且不与 latency-agent 共享同一 GPU。不要写或记录真实密码，示例只能使用 <H800_PASSWORD>。
```

### Pane 4: AP-agent

Prompt:

```text
你是 Stage2 H800 LUT AP-agent。先阅读 H800 access runbook、coverage pipeline plan 和 coverage_pipeline_v1/README.md。为 artifact-ready candidate 生成或导入真实 AP rows，必须保留 dataset/split/ckpt/finetune protocol。缺 AP source 写 failed/no-claim state，继续下一 candidate。禁止把 predicted AP 写为 measured。你可以使用 H800 GPU=3。不要写或记录真实密码，示例只能使用 <H800_PASSWORD>。
```

### Pane 5: supervisor-agent

Prompt:

```text
你是 Stage2 H800 LUT supervisor-agent。先阅读 H800 access runbook、overnight handoff、coverage pipeline plan 和 coverage_pipeline_v1/README.md。每 10-30 分钟轮询各 queue/state/rows/quarantine/export，运行 stage2_summarize_lut_coverage.py 和 stage2_supervisor_poll.py，计算 unique coverage、repeat ratio、latest job state、queue 低水位和 quarantine 增长。发现 queue 太短时请求 artifact-agent 补下一批 candidate；发现 repeat 过多时暂停 repeat；发现 quarantined config 时要求从后续 queue 移除。输出 coverage dashboard、supervisor report、readiness gate 和下一轮调度建议。不要连接 H800 执行测量，不要写或记录真实密码，示例只能使用 <H800_PASSWORD>。
```

## 轮询节奏

建议节奏:

```text
artifact-agent: 每 15 min 扫 candidate/artifact gap
latency-agent: queue-driven，每批 10-20 jobs
energy-agent: queue-driven，每批 5-10 jobs
AP-agent: source-driven，每批 5-10 jobs
supervisor-agent: 每 10-30 min 生成 latest dashboard/report/gate
```

supervisor 每轮至少执行:

```bash
python3 scripts/stage2_supervisor_poll.py \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/latency_lut_rows_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/ap_anchor_rows_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/energy_lut_rows_v1.jsonl \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/latency_job_queue_gpu0.jsonl \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/latency_job_queue_gpu1.jsonl \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/latency_job_queue_gpu4.jsonl \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/latency_job_queue_gpu5.jsonl \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/energy_job_queue.jsonl \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/ap_job_queue.jsonl \
  --job-state multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/latency_job_state.jsonl \
  --job-state multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/energy_job_state.jsonl \
  --job-state multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/ap_job_state.jsonl \
  --axis-gap-report multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/energy_axis_gap_report.json \
  --axis-gap-report multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/ap_axis_gap_report.json \
  --quarantine-rows multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/quarantine/bad_db_quarantine_v1.jsonl \
  --missing-artifact-rows multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/quarantine/missing_artifact_v1.jsonl \
  --out-dir multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports
```

汇报格式固定包含:

```text
unique configs
latency/AP/energy cells
repeat_ratio
ready/running/blocked queue counts
active quarantined config count
readiness decision
action recommendations
```

## 暂停和停止条件

立即暂停相关 pane:

- SSH rate limit: 15-30 秒退避，合并检查命令，禁止并行 SSH 风暴。
- GPU busy: 不抢占，不混跑；写 preflight_blocked 或 requeue。
- CUDA illegal memory / device-side assert: 写 failed state 和 active quarantine，暂停同 workdir/template。
- repeat_ratio > 30%: 停止 repeat continuation，请求新 coverage candidates。
- latency ready queue < 10: 暂停等待 artifact-agent 补 candidate/job，不靠 repeat 保活。
- energy/AP cells < 40% latency cells: 暂停低价值 latency 扩张，优先追赶 energy/AP subset。
- readiness gate 为 `NO_GO`: 不继续 claim 或 paper-grade 输出，先处理 stop 级 recommendation。

全局停止条件:

- active quarantine 持续增长且指向同一 workdir/template。
- latest job_state 显示同一 job 连续失败且无新 config 进展。
- supervisor 连续两轮 `repeat-heavy` 且 unique latency cells 不增长。

恢复条件:

- quarantine 对应 workdir/template 已重建或移除出队列。
- supervisor gate 从 `NO_GO` 降为 `CONDITIONAL_GO` 或 `GO`。
- 下一批 candidate/job plan 已补足 low-water queue。
