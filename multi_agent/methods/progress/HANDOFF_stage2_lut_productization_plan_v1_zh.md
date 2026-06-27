# HANDOFF — Stage2 Latency / AP / Energy LUT 产品化补点计划 v1

日期: 2026-06-25

本文继承:

- `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v2_zh.md`
- `framework/stage2/evidence_registry.py`
- `scripts/stage2_prepare_evidence.py`

目标是把下一阶段优先级收敛到 **latency LUT、AP anchors、energy LUT 的产品化生成路线**，并明确 LUT 表格格式、补点计划、后台长时间运行方式和 registry 更新规则。

---

## 0. 当前结论

下一阶段不应先扩大搜索算法，而应先把 evidence 数据生产路线固定下来:

```text
Stage2SearchSpace
  -> candidate/config_id 生成
  -> latency/AP/energy job queue
  -> 长时间后台补点
  -> row-level LUT tables
  -> coverage summary
  -> Stage2EvidenceRegistry
  -> Stage2 runner / paper table
```

本计划优先完成三类 LUT:

1. `latency_lut_rows_v1` — latency / build / schedule evidence。
2. `ap_anchor_rows_v1` — AP / accuracy evidence。
3. `energy_lut_rows_v1` — power / energy evidence。

严格约束:

- Stage1 manifest 只提供结构和硬件合法性，不提供实测 latency / AP / energy。
- H800 TVM/Relax/MetaSchedule 是新增 measured 主后端。
- TRT 只能作为 historical 或单独 backend scope。
- proxy / demo / estimated 不能写成论文正式 measured 结果。
- energy 缺失时继续 no-claim。
- AP 与 latency 可能是预测值时，DS lookup 默认 report-only。
- 每个 row 必须可追踪到 `config_id`、`run_id`、`provenance`、`measurement_status` 和 `coverage`。

---

## 1. 目录与产物布局

建议统一放在:

```text
results/stage2/<model>/evidence/
```

每个模型至少包含:

```text
results/stage2/<model>/evidence/
  evidence_registry.json
  lut_manifest.json
  jobs/
    lut_job_plan_v1.jsonl
    lut_job_state_v1.jsonl
  latency/
    latency_lut_rows_v1.jsonl
    latency_lut_summary_v1.json
    raw/
  ap/
    ap_anchor_rows_v1.jsonl
    ap_anchor_summary_v1.json
    raw/
  energy/
    energy_lut_rows_v1.jsonl
    energy_lut_summary_v1.json
    raw/
  logs/
```

说明:

- `*_rows_v1.jsonl` 是 canonical table，每行一个不可变证据点。
- `*_summary_v1.json` 是 registry 和报告读取的轻量摘要。
- `raw/` 保存原始 stdout、bench trace、eval JSON、power telemetry 或压缩采样文件。
- `lut_job_plan_v1.jsonl` 是计划队列。
- `lut_job_state_v1.jsonl` 是 append-only 状态日志，支持断点续跑。
- CSV / Parquet 可以后续导出，但不作为 canonical 源。

---

## 2. 统一字段约定

### 2.1 共享字段

所有 LUT row 必须包含:

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema` | string | `latency_lut_row_v1` / `ap_anchor_row_v1` / `energy_lut_row_v1` |
| `row_id` | string | 唯一 row id，建议 `<kind>:<model>:<config_id>:<backend>:<run_id>` |
| `config_id` | string | 跨 latency/AP/energy 对齐主键 |
| `model` | string | 例如 `pyramid_lidar` / `codriving` |
| `manifest_digest` | string | Stage1 manifest sha256 |
| `search_space_schema` | string | 例如 `stage2_search_space_v1` |
| `candidate_id` | string | Stage2SearchSpace software candidate id |
| `software_point_id` | string | width/quant point id |
| `dense_stage` | string | `stage1` / `stage2` / `stage3` / `neck` 等 |
| `optimized_scope` | string | 当前应为 `rsu_dense_core` 或明确 scope |
| `width` | array[int] | 模型原生 width tuple |
| `quant_policy` | string | `fp16` / `int8` / `mixed` / `not_applicable` |
| `schedule_policy` | string | `default` / `metaschedule_tuned` / `not_applicable` |
| `hardware_target` | string | 例如 `H800 Hopper` |
| `backend` | string | `h800_tvm` / `historical_trt` / `model_eval` / `power_telemetry` |
| `measurement_status` | string | `measured` / `estimated` / `proxy` / `historical` / `failed` / `not_done` |
| `provenance` | string | 人类可读来源说明 |
| `run_id` | string | 一次 job 执行唯一 id |
| `created_at` | string | ISO-8601 时间 |
| `source_files` | array[string] | 原始结果文件路径 |
| `raw_artifact` | string/null | raw trace 或 eval JSON 路径 |
| `failure_reason` | string/null | 成功为 null，失败必须填写 |
| `notes` | string | caveat，不允许空泛 |

### 2.2 `config_id` 规则

`config_id` 必须稳定、可复现、跨表一致。建议格式:

```text
<model>__<candidate_id>__<software_point_id>__q_<quant_policy>__s_<schedule_policy>
```

示例:

```text
pyramid_lidar__bev_encoder.s2__bev_encoder.s2:w128:int8__q_int8__s_metaschedule_tuned
codriving__backbone.s2__backbone.s2:w128:fp16__q_fp16__s_default
```

禁止:

- 使用 `row index` 作为唯一 ID。
- 使用会随运行顺序变化的随机 id。
- latency / AP / energy 三张表使用不同 config id 命名同一候选。

---

## 3. Latency LUT 表格格式

文件:

```text
results/stage2/<model>/evidence/latency/latency_lut_rows_v1.jsonl
```

每行 schema:

```json
{
  "schema": "latency_lut_row_v1",
  "row_id": "latency:pyramid_lidar:...:run_20260625_001",
  "config_id": "pyramid_lidar__bev_encoder.s2__...__q_fp16__s_metaschedule_tuned",
  "model": "pyramid_lidar",
  "manifest_digest": "sha256...",
  "search_space_schema": "stage2_search_space_v1",
  "candidate_id": "bev_encoder.s2",
  "software_point_id": "bev_encoder.s2:w128:fp16",
  "dense_stage": "stage3",
  "optimized_scope": "rsu_dense_core",
  "width": [64, 128, 128],
  "quant_policy": "fp16",
  "schedule_policy": "metaschedule_tuned",
  "hardware_target": "H800 Hopper",
  "backend": "h800_tvm",
  "measurement_status": "measured",
  "latency_unit": "us",
  "latency_p50_us": 5506.57,
  "latency_p90_us": 5620.12,
  "latency_mean_us": 5528.44,
  "latency_std_us": 47.31,
  "latency_min_us": 5430.02,
  "latency_max_us": 5711.90,
  "warmup_iters": 50,
  "measure_iters": 200,
  "repeat": 5,
  "batch_size": 2,
  "input_shape": {"spatial_features": [2, 64, 128, 256]},
  "tvm_target": "cuda",
  "tvm_strategy": "relax_metaschedule",
  "build_status": "success",
  "build_time_s": 1673.0,
  "provenance": "H800 TVM Relax/MetaSchedule measured dense-core LUT",
  "run_id": "run_20260625_001",
  "created_at": "2026-06-25T00:00:00+08:00",
  "source_files": ["results/raw/.../bench.json"],
  "raw_artifact": "results/stage2/pyramid_lidar/evidence/latency/raw/run_20260625_001.json",
  "failure_reason": null,
  "notes": "dense-core only; not full-model latency"
}
```

失败样本必须写 row，不能静默跳过:

```json
{
  "schema": "latency_lut_row_v1",
  "config_id": "pyramid_lidar__...__q_int8__s_metaschedule_tuned",
  "backend": "h800_tvm",
  "measurement_status": "failed",
  "build_status": "failed",
  "latency_p50_us": null,
  "failure_reason": "tvm_build_failed:int8_layout_not_buildable",
  "notes": "counts as failed_cells in coverage"
}
```

---

## 4. AP Anchor 表格格式

文件:

```text
results/stage2/<model>/evidence/ap/ap_anchor_rows_v1.jsonl
```

每行 schema:

```json
{
  "schema": "ap_anchor_row_v1",
  "row_id": "ap:pyramid_lidar:...:eval_20260625_001",
  "config_id": "pyramid_lidar__bev_encoder.s2__...__q_fp16__s_not_applicable",
  "model": "pyramid_lidar",
  "manifest_digest": "sha256...",
  "search_space_schema": "stage2_search_space_v1",
  "candidate_id": "bev_encoder.s2",
  "software_point_id": "bev_encoder.s2:w128:fp16",
  "dense_stage": "stage3",
  "optimized_scope": "rsu_dense_core",
  "width": [64, 128, 128],
  "quant_policy": "fp16",
  "schedule_policy": "not_applicable",
  "hardware_target": "not_hardware_specific",
  "backend": "model_eval",
  "measurement_status": "measured",
  "metric": "AP70",
  "metric_value": 0.6369,
  "secondary_metrics": {"AP50": 0.841, "AP30": null},
  "dataset": "DAIR-V2X",
  "eval_split": "val",
  "num_samples": 1789,
  "ckpt_path": "checkpoints/...",
  "ckpt_digest": "sha256...",
  "finetune_protocol": "weight_identity_or_short_ft",
  "training_budget": "none",
  "eval_command": "python scripts/phase2/...",
  "provenance": "DAIR validation AP anchor",
  "run_id": "eval_20260625_001",
  "created_at": "2026-06-25T00:00:00+08:00",
  "source_files": ["results/raw/.../metrics.json"],
  "raw_artifact": "results/stage2/pyramid_lidar/evidence/ap/raw/eval_20260625_001.json",
  "failure_reason": null,
  "notes": "AP anchor; schedule-independent"
}
```

AP 关键约束:

- `finetune_protocol` 不同的 AP 不得混在同一正式比较表。
- `measurement_status=predicted` 的 AP 只能用于 screening / ranking，不可替代 final measured AP。
- AP anchor 与 latency/energy 使用相同 `config_id`，但 `schedule_policy` 可写 `not_applicable`。
- 量化影响 AP 时，`quant_policy` 必须明确，不能把 FP16 AP 复用为 INT8 AP。

---

## 5. Energy LUT 表格格式

文件:

```text
results/stage2/<model>/evidence/energy/energy_lut_rows_v1.jsonl
```

每行 schema:

```json
{
  "schema": "energy_lut_row_v1",
  "row_id": "energy:pyramid_lidar:...:power_20260625_001",
  "config_id": "pyramid_lidar__bev_encoder.s2__...__q_fp16__s_metaschedule_tuned",
  "model": "pyramid_lidar",
  "manifest_digest": "sha256...",
  "search_space_schema": "stage2_search_space_v1",
  "candidate_id": "bev_encoder.s2",
  "software_point_id": "bev_encoder.s2:w128:fp16",
  "dense_stage": "stage3",
  "optimized_scope": "rsu_dense_core",
  "width": [64, 128, 128],
  "quant_policy": "fp16",
  "schedule_policy": "metaschedule_tuned",
  "hardware_target": "H800 Hopper",
  "backend": "h800_tvm_power_telemetry",
  "measurement_status": "measured",
  "energy_unit": "joule_per_inference",
  "joule_per_inference": 1.38,
  "watt_avg": 250.4,
  "watt_p50": 248.9,
  "watt_p90": 263.1,
  "idle_watt_avg": 71.2,
  "idle_baseline_policy": "subtract_idle_avg",
  "sample_window_ms": 5000,
  "telemetry_source": "nvidia_smi_or_nvml",
  "power_cap_watt": null,
  "clock_policy": "default",
  "latency_config_id": "same_as_config_id",
  "latency_run_id": "run_20260625_001",
  "warmup_iters": 50,
  "measure_iters": 200,
  "repeat": 5,
  "provenance": "H800 power telemetry aligned with latency run",
  "run_id": "power_20260625_001",
  "created_at": "2026-06-25T00:00:00+08:00",
  "source_files": ["results/raw/.../power.csv"],
  "raw_artifact": "results/stage2/pyramid_lidar/evidence/energy/raw/power_20260625_001.csv",
  "failure_reason": null,
  "notes": "same config_id as latency row; dense-core only"
}
```

Energy 关键约束:

- 必须与 latency row 使用同一 `config_id`。
- 最好与 latency 同 run 采样；若不是，必须记录 `latency_run_id` 和差异说明。
- telemetry source 不同的数据不能混写为同一 measured population。
- idle baseline 是否扣除必须显式写明。
- energy 缺失时 registry 继续输出 `energy_claim_allowed=false`。

---

## 6. Job Queue 表格格式

文件:

```text
results/stage2/<model>/evidence/jobs/lut_job_plan_v1.jsonl
results/stage2/<model>/evidence/jobs/lut_job_state_v1.jsonl
```

`lut_job_plan_v1.jsonl` 每行:

```json
{
  "schema": "lut_job_plan_row_v1",
  "job_id": "latency:pyramid_lidar:bev_encoder.s2:w128:fp16:tuned",
  "model": "pyramid_lidar",
  "lut_kind": "latency",
  "job_type": "generate_latency_lut",
  "priority": 10,
  "config_id": "pyramid_lidar__bev_encoder.s2__...__q_fp16__s_metaschedule_tuned",
  "manifest_path": "framework/partitions/pyramid_lidar_partition.yaml",
  "registry_path": "results/stage2/pyramid_lidar/evidence/evidence_registry.json",
  "candidate_id": "bev_encoder.s2",
  "software_point_id": "bev_encoder.s2:w128:fp16",
  "depends_on": [],
  "expected_output": "results/stage2/pyramid_lidar/evidence/latency/latency_lut_rows_v1.jsonl",
  "command": [
    "python",
    "scripts/stage2_generate_latency_lut.py",
    "--job-id",
    "latency:pyramid_lidar:bev_encoder.s2:w128:fp16:tuned",
    "--config-id",
    "pyramid_lidar__bev_encoder.s2__...__q_fp16__s_metaschedule_tuned",
    "--manifest",
    "framework/partitions/pyramid_lidar_partition.yaml",
    "--out-jsonl",
    "results/stage2/pyramid_lidar/evidence/latency/latency_lut_rows_v1.jsonl"
  ],
  "max_attempts": 2,
  "timeout_s": 21600,
  "resource": {"gpu": "any_h800", "exclusive": true},
  "created_at": "2026-06-25T00:00:00+08:00"
}
```

`job_type` 必须显式区分 LUT 生成与历史结果导入:

| job_type | 作用 | 输出 |
|---|---|---|
| `generate_latency_lut` | 调用 latency measurement/build command，产生新的 latency LUT 点 | `latency_lut_row_v1` |
| `generate_ap_lut` | 调用 AP eval/anchor command，产生新的 AP anchor 点 | `ap_anchor_row_v1` |
| `generate_energy_lut` | 调用 energy telemetry command，产生新的 energy LUT 点 | `energy_lut_row_v1` |
| `import_existing_latency_lut` | 迁移已有 latency JSON，不能替代新补点 | `latency_lut_row_v1` |
| `import_existing_ap_lut` | 迁移已有 AP JSON，不能替代新补点 | `ap_anchor_row_v1` |
| `import_existing_energy_lut` | 迁移已有 telemetry CSV/JSONL，不能替代新补点 | `energy_lut_row_v1` |

因此，正式补点队列必须以 `generate_*_lut` 为主；`import_existing_*` 只用于把历史产物转换为 canonical row，不能作为产品化 LUT 生成路线的完成标准。

生成入口文件必须固定为:

```text
scripts/stage2_generate_latency_lut.py
scripts/stage2_generate_ap_lut.py
scripts/stage2_generate_energy_lut.py
```

这三个 generator 负责调用真实 measurement/eval/telemetry 命令并追加 LUT row；importer 只负责迁移已有产物。

`lut_job_state_v1.jsonl` 是 append-only 状态日志:

```json
{
  "schema": "lut_job_state_row_v1",
  "job_id": "latency:pyramid_lidar:bev_encoder.s2:w128:fp16:tuned",
  "status": "running",
  "attempt": 1,
  "worker_id": "hostA:pid12345:gpu0",
  "started_at": "2026-06-25T00:10:00+08:00",
  "finished_at": null,
  "log_path": "results/stage2/pyramid_lidar/evidence/logs/job.log",
  "output_row_id": null,
  "failure_reason": null
}
```

状态枚举:

```text
queued
running
succeeded
failed
skipped
stale_lock_requeued
```

后台长跑要求:

- runner 启动时读取 plan + state，跳过已 succeeded job。
- running job 若超过 `timeout_s` 且 worker 不存在，标为 `stale_lock_requeued`。
- 每个 job 完成后 append 一条 state，不原地覆盖旧状态。
- 每个成功 job 追加一个 LUT row。
- 每个失败 job 也追加 failed row 或 state failure reason，不能静默丢失。

---

## 7. 补点规模计划

### 7.1 Smoke 阶段

目标: 验证 schema、runner、断点续跑、registry 更新。

| 模型 | latency cells | AP anchors | energy cells | 说明 |
|---|---:|---:|---:|---|
| Pyramid | 12-24 | 6-10 | 12-24 | 覆盖 baseline、P-hub、INT8 buildability cliff、default/tuned |
| CoDriving | 12-24 | 6-10 | 12-24 | 覆盖 base/p25/p50/p75、serial/default、estimated cell 复测 |

### 7.2 Calibration 阶段

目标: 支撑正式 Stage2 runner 和 Pareto 排序。

| 模型 | latency cells | AP anchors | energy cells | 说明 |
|---|---:|---:|---:|---|
| Pyramid | 60-120 | 15-30 | 60-120 | 覆盖 P anchors、Q modes、schedule policies |
| CoDriving | 60-120 | 15-30 | 60-120 | 补齐现有 estimated latency cells，并验证 separability |

### 7.3 Paper 阶段

目标: 只补论文正式表格需要的点。

| 模型 | latency cells | AP anchors | energy cells | 说明 |
|---|---:|---:|---:|---|
| Pyramid | 120-300 | 40-80 | 120-300 | 只覆盖最终声明范围，不做无意义笛卡尔积 |
| CoDriving | 120-300 | 40-80 | 120-300 | 明确 measured/proxy/historical 边界 |

---

## 8. 后台长跑启动方式

推荐先 dry-run 生成队列:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_plan_lut_jobs.py \
  --model pyramid_lidar \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --registry results/stage2/pyramid_lidar/evidence/evidence_registry.json \
  --phase smoke \
  --out-jsonl results/stage2/pyramid_lidar/evidence/jobs/lut_job_plan_v1.jsonl \
  --dry-run
```

后台运行:

```bash
mkdir -p logs
nohup env PYTHONPATH=/home/jichengzhi/V2X \
  python scripts/stage2_lut_worker.py \
    --job-plan results/stage2/pyramid_lidar/evidence/jobs/lut_job_plan_v1.jsonl \
    --job-state results/stage2/pyramid_lidar/evidence/jobs/lut_job_state_v1.jsonl \
    --max-hours 12 \
    --resume \
  > logs/stage2_lut_pyramid_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > logs/stage2_lut_pyramid.pid
```

监控:

```bash
tail -f logs/stage2_lut_pyramid_*.log
```

恢复:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_lut_worker.py \
  --job-plan results/stage2/pyramid_lidar/evidence/jobs/lut_job_plan_v1.jsonl \
  --job-state results/stage2/pyramid_lidar/evidence/jobs/lut_job_state_v1.jsonl \
  --resume \
  --max-hours 12
```

长跑前置验收:

- `--dry-run` 能打印 expected job counts。
- `lut_job_plan_v1.jsonl` 无重复 `job_id`。
- `config_id` 能在 latency/AP/energy 三类任务间对齐。
- worker 可安全跳过 succeeded job。
- worker 对失败任务写 state 和 failure row。
- registry update 不会把 failed/proxy 写成 measured。

---

## 9. Registry 更新规则

补点完成后运行:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_update_registry_from_luts.py \
  --registry results/stage2/pyramid_lidar/evidence/evidence_registry.json \
  --latency-rows results/stage2/pyramid_lidar/evidence/latency/latency_lut_rows_v1.jsonl \
  --ap-rows results/stage2/pyramid_lidar/evidence/ap/ap_anchor_rows_v1.jsonl \
  --energy-rows results/stage2/pyramid_lidar/evidence/energy/energy_lut_rows_v1.jsonl \
  --out-json results/stage2/pyramid_lidar/evidence/evidence_registry.json
```

更新规则:

- `measurement_status=measured` 只来自 measured rows。
- `failed` rows 计入 `failed_cells`。
- `proxy` / `estimated` rows 计入 `proxy_cells` 或 estimated 统计，不覆盖 measured。
- energy 只有 measured rows 数量大于 0 且 telemetry source 一致时，才允许 `energy_claim_allowed=true`。
- latency/AP 缺 required measured rows 时，正式 runner fail closed。

---

## 10. 实施优先级

P0:

1. 写 schema validator，覆盖 latency/AP/energy 三类 row。
2. 写 job plan generator，输出 `generate_latency_lut`、`generate_ap_lut`、`generate_energy_lut` 三类 job，并给出 `expected_output`。
3. 写 latency LUT generator 的最小版本: 从单个 `lut_job_plan_row_v1` 读取 `config_id`，调用 latency measurement command，追加 `latency_lut_row_v1`。
4. 写 AP LUT generator 的最小版本: 从同一 `config_id` 调用 AP eval/anchor command，追加 `ap_anchor_row_v1`。
5. 写 energy LUT generator 的最小版本: 从同一 `config_id` 调用 telemetry command，追加 `energy_lut_row_v1`。
6. 写 energy no-claim/claim validator: 只有 measured telemetry row 存在时，registry 才允许 `energy_claim_allowed=true`。
7. 写历史 latency/AP/energy importer，作为已有文件迁移路径，不能替代 generator。

P1:

1. 写后台 worker，执行 `generate_*_lut` job，支持断点续跑。
2. 接入真实 latency measurement/build command，产出 measured latency row。
3. 接入真实 AP eval command，产出 measured AP anchor row。
4. 接入真实 energy telemetry command，产出 measured energy row。
5. 写 registry update script，每批生成后刷新 coverage 和 claim gate。

P2:

1. 扩展到 CoDriving estimated latency cells 复测。
2. 扩展到 H800 Q evidence。
3. 扩展到 DS top-K closed-loop validation。
4. 同步 export repo 和 docs。

---

## 11. 不在本计划范围

- 不重新定义 Stage2 public input。
- 不把 `evidence_registry_path` 暴露为用户级 CLI 参数。
- 不启动真实后台长跑。
- 不承诺 energy 已有 measured 数据。
- 不把 DS lookup 升级成主 Pareto 目标。
- 不处理 full-model speedup claim。

---

## 12. P1/P2 执行落地状态

日期: 2026-06-25

已落地 P1 基础设施:

- `framework/stage2/lut_productization.py`
  - row/schema validator。
  - `coverage_from_rows()`。
  - `job_plan_row()` / `latest_job_status()` / `next_queued_jobs()`。
  - `energy_claim_allowed_from_rows()`。
- `scripts/stage2_plan_lut_jobs.py`
  - 生成 `generate_latency_lut`、`generate_ap_lut`、`generate_energy_lut` 三类主路径 job。
  - command JSON 由调用方传入，真实 measurement/eval/telemetry 命令必须 stdout 输出 JSON object。
- `scripts/stage2_generate_latency_lut.py`
  - 调用 latency measurement command，追加 `latency_lut_row_v1`。
- `scripts/stage2_generate_ap_lut.py`
  - 调用 AP eval command，追加 `ap_anchor_row_v1`。
- `scripts/stage2_generate_energy_lut.py`
  - 调用 energy telemetry command，追加 `energy_lut_row_v1`。
- `scripts/stage2_lut_worker.py`
  - 支持 `--resume`、`--max-jobs`、`--max-hours`。
  - append-only 写 `lut_job_state_v1.jsonl`。
  - 跳过 `succeeded` / `skipped` job。
- `scripts/stage2_update_registry_from_luts.py`
  - 从 canonical row JSONL 计算 coverage。
  - 刷新 latency/AP/energy source path、backend、measurement_status 和 coverage。
  - energy 只有合法 measured telemetry row 存在时才允许 claim。
- `scripts/stage2_import_latency_lut.py`
- `scripts/stage2_import_ap_anchors.py`
- `scripts/stage2_import_energy_lut.py`
  - 作为历史产物迁移路径，不能替代 `generate_*_lut`。

已同步文档:

- `docs/stage2-evidence-registry.zh-CN.md`
  - canonical row tables。
  - command JSON stdout 合约。
  - job plan / worker / registry update 命令。
  - energy no-claim 规则。
  - 大规模 LUT 启动门槛。

P2 当前状态:

- CoDriving estimated latency cells 复测: 队列/worker/row schema 已支持；真实复测还需要在 H800 目标环境提交 CoDriving job plan。
- H800 Q evidence: 仍应作为后续独立 quant evidence source 接入；不得继续把 historical TRT Q 作为 measured。
- DS top-K closed-loop validation: 仍保持 validation/report-only；AP 或 latency 任一为 predicted 时，DS LUT 不得升级为主 Pareto 目标。
- export repo/docs: 本仓库 docs 已同步；外部 export repo 未在本轮修改。

大规模补点启动判断:

- 可以开始 `smoke` 级后台补点: 是，前提是三类 command JSON 已替换成真实命令，并在目标环境先跑 `--max-jobs 1`。
- 可以开始大规模 LUT 完善: 不能直接开始。必须先通过 smoke gate:
  1. latency/AP/energy 各至少 1 个真实 job 成功。
  2. 生成 measured canonical rows。
  3. registry updater 刷新后 measured coverage 正确。
  4. energy claim gate 行为符合预期。
  5. worker resume 不重复执行已 succeeded job。
