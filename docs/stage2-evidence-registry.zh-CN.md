# Stage2 Evidence Registry 与 LUT 产品化运行说明

> 可移植执行约定：下文保留的 `/home/jichengzhi/V2X` 是历史实验记录；在新 checkout 中以 `$V2X_ROOT` 替换，并先按 [`REPRODUCIBILITY.zh-CN.md`](REPRODUCIBILITY.zh-CN.md) 设置环境变量。

日期: 2026-06-25

本文说明 Stage2 latency/AP/energy LUT 的产品化生成路线。正式补点主路径是 `generate_*_lut`，历史数据迁移路径 `import_existing_*` 只用于把已有产物转换成 canonical JSONL row，不能作为新 LUT 生成完成标准。

## 1. Canonical 产物

推荐目录:

```text
results/stage2/<model>/evidence/
  evidence_registry.json
  jobs/
    lut_job_plan_v1.jsonl
    lut_job_state_v1.jsonl
  latency/latency_lut_rows_v1.jsonl
  ap/ap_anchor_rows_v1.jsonl
  energy/energy_lut_rows_v1.jsonl
  logs/
```

三张 canonical 表:

- `latency_lut_row_v1`: H800 TVM/Relax/MetaSchedule latency/build evidence.
- `ap_anchor_row_v1`: DAIR/CoDriving AP anchor/eval evidence.
- `energy_lut_row_v1`: H800 power telemetry energy evidence.

核心对齐规则:

- latency/AP/energy 必须共享稳定 `config_id`。
- `measurement_status=measured` 只能来自真实 measurement/eval/telemetry。
- `proxy`、`estimated`、`historical` 可以参与 coverage，但不能覆盖 measured claim。
- energy 只有存在合法 measured telemetry row 时才允许 `energy_claim_allowed=true`。

## 2. 真实命令 stdout 合约

`scripts/stage2_generate_*_lut.py` 通过 command JSON 执行真实命令。真实命令必须向 stdout 打印一个 JSON object。

Latency command 至少输出:

```json
{"latency_p50_us": 123.4}
```

可选字段包括 `latency_p90_us`、`latency_mean_us`、`latency_std_us`、`warmup_iters`、`measure_iters`、`repeat`、`raw_artifact`。

AP command 至少输出:

```json
{"metric": "AP70", "metric_value": 0.63, "dataset": "DAIR-V2X", "eval_split": "val", "ckpt_path": "ckpts/smoke.pt"}
```

Energy command 至少输出:

```json
{"joule_per_inference": 1.2, "telemetry_source": "nvidia_smi", "raw_artifact": "logs/energy.json"}
```

## 3. 生成 job plan

示例:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_plan_lut_jobs.py \
  --model pyramid_lidar \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --registry results/stage2/pyramid_lidar/evidence/evidence_registry.json \
  --config-id cfg_smoke \
  --candidate-id smoke \
  --software-point-id smoke:w64x128x256:fp16 \
  --dense-stage neck \
  --width 64,128,256 \
  --quant-policy fp16 \
  --schedule-policy default \
  --latency-measurement-command-json '["python","-c","import json; print(json.dumps({\"latency_p50_us\":123.4}))"]' \
  --ap-eval-command-json '["python","-c","import json; print(json.dumps({\"metric_value\":0.63,\"dataset\":\"DAIR-V2X\",\"eval_split\":\"val\",\"ckpt_path\":\"ckpts/smoke.pt\"}))"]' \
  --energy-telemetry-command-json '["python","-c","import json; print(json.dumps({\"joule_per_inference\":1.2,\"telemetry_source\":\"nvidia_smi\",\"raw_artifact\":\"logs/energy.json\"}))"]' \
  --out-jsonl results/stage2/pyramid_lidar/evidence/jobs/lut_job_plan_v1.jsonl
```

计划文件必须包含三类生成 job:

- `generate_latency_lut`
- `generate_ap_lut`
- `generate_energy_lut`

## 4. 后台 worker

前台 smoke:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_lut_worker.py \
  --job-plan results/stage2/pyramid_lidar/evidence/jobs/lut_job_plan_v1.jsonl \
  --job-state results/stage2/pyramid_lidar/evidence/jobs/lut_job_state_v1.jsonl \
  --max-jobs 1 \
  --resume
```

后台长跑:

```bash
mkdir -p logs
nohup env PYTHONPATH=/home/jichengzhi/V2X \
  python scripts/stage2_lut_worker.py \
    --job-plan results/stage2/pyramid_lidar/evidence/jobs/lut_job_plan_v1.jsonl \
    --job-state results/stage2/pyramid_lidar/evidence/jobs/lut_job_state_v1.jsonl \
    --max-hours 12 \
    --resume \
  > logs/stage2_lut_pyramid_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

worker 行为:

- append-only 写 `lut_job_state_v1.jsonl`。
- 自动跳过 `succeeded` / `skipped` job。
- 失败 job 写 `failed` 和 `failure_reason`。
- 可用 `--max-jobs` 控制 smoke，`--max-hours` 控制长跑窗口。

## 5. Registry 更新

补点后刷新 registry:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage2_update_registry_from_luts.py \
  --registry results/stage2/pyramid_lidar/evidence/evidence_registry.json \
  --latency-rows results/stage2/pyramid_lidar/evidence/latency/latency_lut_rows_v1.jsonl \
  --ap-rows results/stage2/pyramid_lidar/evidence/ap/ap_anchor_rows_v1.jsonl \
  --energy-rows results/stage2/pyramid_lidar/evidence/energy/energy_lut_rows_v1.jsonl \
  --out-json results/stage2/pyramid_lidar/evidence/evidence_registry.json
```

更新规则:

- coverage 从 row-level JSONL 重新计算。
- latency measured backend 必须是 `h800_tvm`。
- AP measured backend 必须是 `model_eval`。
- energy measured backend 必须是 `h800_tvm_power_telemetry`。
- energy 空表、failed-only、telemetry source 不匹配时继续 no-claim。

## 6. 大规模 LUT 启动门槛

大规模补点前必须按顺序满足:

1. Schema gate: `test_stage2_lut_productization`、registry/integration contract、`py_compile` 全部通过。
2. Queue gate: `stage2_plan_lut_jobs.py` 生成的正式队列只包含 `generate_*_lut` 主路径 job。
3. Smoke gate: 每个模型至少 1 个 config 完成 latency/AP/energy 三类真实命令，并产生 measured canonical rows。
4. Registry gate: updater 刷新后 latency/AP measured coverage 大于 0；energy 只有 measured telemetry row 存在时 claim allowed。
5. P2 CoDriving gate: CoDriving estimated latency cells 至少完成一轮 H800 复测，estimated rows 不直接替代 measured rows。
6. P2 Q evidence gate: H800 Q evidence 单独建表或作为后续 quant evidence source，不再把 historical TRT Q 当 measured。
7. P2 DS gate: DS top-K closed-loop validation 只作为 validation/report-only gate；AP/latency 任一为 predicted 时不得把 DS lookup 升级为主 Pareto 目标。

结论: 基础设施可以支持后台长时间补点；是否开始大规模运行取决于 smoke gate 是否在目标 H800 环境用真实命令通过。
