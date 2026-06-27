# Stage2 LUT Coverage Pipeline v1

本目录用于 Stage2 H800 LUT coverage-first 轮询流水线。目标不是堆叠 repeat 行数，而是扩大 unique config、latency/AP/energy axis cell 覆盖，并把 quarantine/block 状态显式暴露给 supervisor。

## 目录契约

建议布局:

```text
coverage_pipeline_v1/
  candidates/
    candidate_queue.jsonl
  artifacts/
    artifact_tasks.jsonl
    artifact_state.jsonl
    artifact_registry_v1.jsonl
  jobs/
    artifact_job_state.jsonl
    latency_job_queue_gpu0.jsonl
    latency_job_queue_gpu1.jsonl
    latency_job_queue_gpu4.jsonl
    latency_job_queue_gpu5.jsonl
    latency_job_state.jsonl
    energy_job_queue.jsonl
    energy_job_state.jsonl
    ap_job_queue.jsonl
    ap_job_state.jsonl
  rows/
    latency_lut_rows_v1.jsonl
    energy_lut_rows_v1.jsonl
    ap_anchor_rows_v1.jsonl
  quarantine/
    bad_db_quarantine_v1.jsonl
    outlier_quarantine_v1.jsonl
  exports/
    coverage_summary_seed.json
    coverage_dashboard_latest.csv
    readiness_gate_latest.json
    supervisor_report_latest.md
```

JSONL 文件保持 append-only。`job_state` 同一 `job_id` 允许多条记录，supervisor 只以最后一条为准。

## Coverage Summary

本地汇总命令:

```bash
python3 scripts/stage2_summarize_lut_coverage.py \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/latency_lut_rows_merged_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/ap_anchor_rows_merged_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/energy_lut_rows_merged_v1.jsonl \
  --quarantine-rows multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl \
  --out-json multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/coverage_summary_seed.json
```

关键字段:

```text
raw_row_count
measured_row_count
unique_config_count
unique_latency_cells
unique_energy_cells
unique_ap_cells
repeat_rows
repeat_ratio
repeat_policy_label
quarantined_config_count
blocked_job_count
```

`repeat_ratio > 0.30` 标为 `repeat-heavy`。同一 axis cell 的多次 measured row 只贡献 1 个 coverage cell，其余计入 repeat。

## Supervisor Poll

本地 supervisor 输出:

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

输出:

- `coverage_dashboard_latest.csv`: 扁平 metric/value 表，便于 tail、diff 和粘贴汇报。
- `supervisor_report_latest.md`: 人读报告，含 coverage、queue、quarantine 和 action recommendations。
- `readiness_gate_latest.json`: 机器可读 gate，含 `GO`、`CONDITIONAL_GO` 或 `NO_GO`。

## Supervisor Actions

默认规则:

```text
repeat_ratio > 30%                  -> stop repeat, generate new candidates
latency_ready_jobs < 10             -> ask artifact-agent for next batch
energy coverage < 40% latency       -> prioritize energy subset
AP coverage < 40% latency           -> prioritize AP source/eval
active quarantine config count rises -> pause related workdir/template
```

supervisor 只生成报告和建议，不连接 H800，不写密码，不把 predicted AP 写成 measured AP。
