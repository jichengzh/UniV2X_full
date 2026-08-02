# Stage2 6 小时并行 LUT 生成计划 v1

日期：2026-06-26

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用 GPU0-5 连续约 6 小时并行生成 H800 Stage2 latency/AP/energy LUT，使明早可审查一个相对完善、带缺口标记和 claim gate 的 LUT。

**Architecture:** 本轮不再等待 AP/energy 完全稳定后才跑 latency，而采用三轴同步流水线：latency 负责快速扩展配置覆盖，AP/energy 按相同 config 队列追赶；任何 AP/energy 失败只写 failed/no-claim/quarantine，不阻断后续配置。数据仍统一写入 `multi_agent/data/stage2_lut_generation_v1/`，所有新点必须经 schema validator、outlier detector、quick review、registry/readiness gate 汇总。

**Tech Stack:** Python Stage2 LUT scripts, H800 TVM/MetaSchedule, AP eval/import command, H800 power telemetry, JSONL job plan/state, artifact/evidence registry, latency outlier detector.

---

## 1. 本轮判定修正

之前 `P/S ready subset` 只放行 `p50/p75/base/iso_s1` 是三轴最小安全子集，不是 calibration 的完整范围。本轮目标是 6 小时受控批量生产，所以配置范围必须回到 `PLAN_stage2_dataset_v2_aligned_lut_generation_v1_zh.md` 的 calibration 分布：

| 类别 | 本轮处理方式 |
|---|---|
| latency | 立即扩展到 base/p25/p50/p75/p50b2/single-axis/coupled/cliff，不拘泥 p50/p75 |
| AP | 与 latency 使用同一 config 队列；能真实 eval/import 就写 measured row，不能可信测量就写 failed/no-claim |
| energy | 与 latency 使用同一 config 队列；可与 latency 并行但不能共用同一 GPU；不可信就写 failed/no-claim |
| Q/INT8 | artifact ready 前不纳入主队列；若中途找到 ready artifact，可进入低优先级追加队列 |
| outlier | trap25、历史 unstable repeat 单独 retest；失败不停止全局生产 |

本轮可以启动的是：

```text
H800 FP16 latency/AP/energy 受控 6 小时并行 LUT production: GO
full unattended P/Q/S tri-axis all-scope production: 仍不是本轮目标
```

## 2. GPU 调度策略

用户已确认 GPU0-5 可用。本轮采用 6 卡并行，不要求整机完全空闲，但 latency/energy 的 target GPU 必须在 job 启动前空闲。

### 2.1 固定分工

| GPU | 默认任务 | 原因 |
|---:|---|---|
| 0 | latency worker A | 跑 anchor + single-axis |
| 1 | latency worker B | 跑 coupled/cliff + repeats |
| 2 | latency worker C | 跑补充点/CoDriving/低优先级点 |
| 3 | energy worker | energy 与 latency 并行，但每次仅一张 target GPU 做 telemetry |
| 4 | AP worker A | AP eval/import 跟随 latency 队列 |
| 5 | AP worker B | AP eval/import 跟随 latency 队列 |

### 2.2 运行中重分配规则

| 情况 | 处理 |
|---|---|
| AP 队列积压超过 latency 已完成配置数的 50% | GPU2 从 latency 切给 AP 1 小时 |
| energy 队列积压超过 6 个 config | GPU2 或 GPU5 临时切给 energy，但同一时刻 energy target GPU 不超过 2 张 |
| latency 失败率超过 25% | 暂停低优先级 latency，只跑 known-good/retest 队列 |
| energy 出现连续 2 个 CUDA illegal address | energy 进入 salvage/failed 标记模式，跳过同 DB 模板配置 |
| AP eval 数据集/ckpt 不可用 | 写 AP failed/no-claim row，继续下一配置 |

### 2.3 preflight 硬规则

每个 latency/energy job 启动前保存：

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv
nvidia-smi pmon -c 1
```

target GPU 判定：

| 条件 | 判定 |
|---|---|
| util > 5% | 当前 job 不启动，写 `preflight_blocked`，转下一个 config |
| 有非本 job compute process | 当前 job 不启动，写 `preflight_blocked` |
| memory used > 1024 MiB 且不是系统保留 | 当前 job 不启动，写 `preflight_blocked` |
| AP job 占用 GPU4/5 | 不影响 GPU0-3 的 latency/energy claim；但 energy row 标记 `quality=calibration_parallel`，paper 前再复测 |

## 3. 6 小时配置分布

### 3.1 Priority A：必须覆盖的主锚点

这些配置必须优先完成 latency，并尽量同步补 AP/energy：

| label | width | 目的 | latency | AP | energy |
|---|---|---|---|---|---|
| base | 64/128/256 | H800 tuned 主锚 | 必跑 repeat | 必跑 | 必跑 |
| p50 | 32/64/128 | pruning 主锚 | 必跑 repeat | 必跑 | 必跑 |
| p75 | 16/32/64 | 强剪枝端点 | 必跑 repeat | 必跑 | 必跑 |
| trap25 | 48/96/192 | outlier/quarantine 判定 | 必跑 retest | 必跑 | 能跑则跑，异常则 no-claim |
| pad64 | 64/96/192 | W_g/P_g anchor | 必跑 | 必跑 | 先跳过旧失败 DB，能用 known-good path 再跑 |
| p50b2_136 | 32/64/136 | dataset_v2 对齐点 | 必跑 | 必跑 | 可选 |

### 3.2 Priority B：single-axis coverage

| group | width | 目的 | 本轮策略 |
|---|---|---|---|
| s0_16 | 16/128/256 | stage0 sensitivity | latency + AP，energy 若有余量 |
| s0_32 | 32/128/256 | stage0 sensitivity | latency + AP，energy 若有余量 |
| iso_s0 | 48/128/256 | stage0 near-base | latency + AP + energy |
| s1_32 | 64/32/256 | stage1 sensitivity | latency + AP，energy 若有余量 |
| s1_64 | 64/64/256 | 已知 CUDA illegal memory 风险 | 默认 quarantine，不进入夜间主队列 |
| iso_s1 | 64/96/256 | stage1 near-base | latency + AP + energy |
| s2_64 | 64/128/64 | stage2 sensitivity | latency + AP，energy 若有余量 |
| s2_128 | 64/128/128 | stage2 sensitivity | latency + AP，energy 若有余量 |
| iso_s2 | 64/128/192 | stage2 near-base | latency + AP + energy |

### 3.3 Priority C：coupled/cliff coverage

| label | width | 目的 | 本轮策略 |
|---|---|---|---|
| mix_a | 32/96/192 | coupled candidate | latency + AP + energy |
| mix_b | 48/64/256 | cliff / slow tuned 点 | latency repeat + AP + energy |
| mix_c | 16/128/128 | coupled candidate | latency + AP，energy 若有余量 |
| mix_d | 48/128/128 | cliff / slow tuned 点 | latency repeat + AP + energy |
| mix_e | 64/64/128 | p1/p2 新点 | latency + AP，energy 若有余量 |

### 3.4 Priority D：低优先级追加

| 类别 | 条件 | 处理 |
|---|---|---|
| CoDriving base/p50 | latency 队列提前完成 | 作为 negative anchor latency/AP 补点 |
| INT8/Q base/p50 | artifact registry 出现 ready artifact | 加入低优先级 Q 队列 |
| DS top-K | AP+latency 均完成且还有时间 | 只做 report-only validation，不进入主 claim |

## 4. 6 小时节奏

| 时间窗 | latency | AP | energy | 每小时审查 |
|---|---|---|---|---|
| 0:00-0:30 | base/p50/p75/trap25/pad64/p50b2 | base/p50 | base | preflight + first rows schema |
| 0:30-2:00 | single-axis A/B | p75/trap25/pad64/p50b2 | p50/p75 | 检查 AP/energy 是否跟上主锚点 |
| 2:00-4:00 | single-axis + coupled/cliff | single-axis | iso_s0/iso_s1/iso_s2/mix_a | 跑 outlier detector，隔离异常模板 |
| 4:00-5:15 | repeats + missing latency gaps | coupled/cliff AP | mix_b/mix_d/缺口 energy | 优先补齐已有 latency 的 AP/energy |
| 5:15-5:45 | 只补最小缺口，不开高风险新 DB | 只补缺口 | 只补缺口 | 停止启动长 job |
| 5:45-6:00 | freeze | freeze | freeze | export registry/quick review/readiness |

## 5. 失败不停止策略

### 5.1 单配置失败处理

| 失败类型 | 写入 | 后续 |
|---|---|---|
| latency command 非零退出 | job_state=`failed`，raw artifact 保存 stderr/stdout | 同 config AP/energy 暂缓，进入 latency quarantine |
| latency repeat outlier | latency row `claim_status=no_claim/conditional`，outlier report 标记 | 继续跑其他 config，后续单独 retest |
| AP eval 缺 ckpt/dataset | AP failed/no-claim row 或 job_state=`failed` | 继续下一 AP config |
| AP 数值缺关键字段 | AP row 不进入 claim；写 failure_reason | 继续下一 AP config |
| energy telemetry 失败但 payload 存在 | 自动 salvage canonical energy row | 继续下一 energy config |
| energy CUDA illegal address | job_state=`failed`，config/DB 模板 quarantine | 继续下一 energy config |
| energy 数值异常过低/过高 | row 标记 `quality_warning`，paper 前复测 | 不阻塞 calibration coverage |

### 5.2 夜间不允许的行为

1. 不因为一个 AP/energy 失败停止整批。
2. 不把 AP placeholder 当 measured AP。
3. 不把 failed/no-claim energy 当 claimable energy。
4. 不把 Q/INT8 historical/proxy 行当 H800 measured Q evidence。
5. 不把 trap25 unstable repeat 当 paper-grade latency claim。

## 6. 预期产出

6 小时后期望看到：

| 产物 | 目标 |
|---|---:|
| 新增/复测 latency canonical rows | 18-30 |
| 新增 AP rows 或 failed/no-claim AP evidence | 12-20 |
| 新增 energy rows 或 failed/no-claim energy evidence | 8-14 |
| quick review 表覆盖 config | 18+ |
| outlier report | 必须生成 |
| artifact/evidence registry | 必须更新 |
| readiness gate | 至少给出 latency/AP/energy 分项判定 |

明早可以直接查看：

```text
multi_agent/data/stage2_lut_generation_v1/exports/quick_review_latest.csv
multi_agent/data/stage2_lut_generation_v1/exports/outlier_report_latest.csv
multi_agent/data/stage2_lut_generation_v1/exports/readiness_gate_latest.json
multi_agent/data/stage2_lut_generation_v1/registry/evidence_registry_latest.json
multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_summary_v1.json
```

## 7. 启动前检查清单

- [ ] 保存 GPU0-5 preflight：
  ```bash
  cd ${V2X_ROOT}
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv
  nvidia-smi pmon -c 1
  ```
- [ ] 确认 GPU0-5 没有非本轮 compute process。
- [ ] 确认 `s1_64` 和 pad64 energy 旧失败模板在 quarantine/skip list 中。
- [ ] 确认 job 输出目录使用新 run id，避免覆盖旧 state：
  ```text
  multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/
  ```
- [ ] 确认三类输出 JSONL 分开：
  ```text
  generated/overnight_6h_20260626/latency/latency_lut_rows_v1.jsonl
  generated/overnight_6h_20260626/ap/ap_anchor_rows_v1.jsonl
  generated/overnight_6h_20260626/energy/energy_lut_rows_v1.jsonl
  ```
- [ ] 启动前创建目录和空 JSONL，保证失败轴也能进入收尾导出：
  ```bash
  cd ${V2X_ROOT}
  mkdir -p \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/latency \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/ap \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/energy \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/raw \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/logs
  touch \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/latency/latency_lut_rows_v1.jsonl \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/ap/ap_anchor_rows_v1.jsonl \
    multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/energy/energy_lut_rows_v1.jsonl \
    multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl
  ```

## 8. 推荐 worker 启动方式

实际启动时应生成三个队列文件，而不是把所有 job 塞进一个串行 worker：

```text
generated/overnight_6h_20260626/jobs/latency_gpu012_job_plan.jsonl
generated/overnight_6h_20260626/jobs/ap_gpu45_job_plan.jsonl
generated/overnight_6h_20260626/jobs/energy_gpu3_job_plan.jsonl
```

每个 worker 使用独立 state/log：

```bash
cd ${V2X_ROOT}

nohup python3 scripts/stage2_lut_worker.py \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs/latency_gpu012_job_plan.jsonl \
  --job-state multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs/latency_gpu012_job_state.jsonl \
  --resume \
  --max-hours 5.75 \
  --require-gpu-idle \
  --quarantine-db multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl \
  --log-dir multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/logs \
  > multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/logs/latency_gpu012.nohup.log 2>&1 &

nohup python3 scripts/stage2_lut_worker.py \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs/ap_gpu45_job_plan.jsonl \
  --job-state multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs/ap_gpu45_job_state.jsonl \
  --resume \
  --max-hours 5.75 \
  --quarantine-db multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl \
  --log-dir multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/logs \
  > multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/logs/ap_gpu45.nohup.log 2>&1 &

nohup python3 scripts/stage2_lut_worker.py \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs/energy_gpu3_job_plan.jsonl \
  --job-state multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs/energy_gpu3_job_state.jsonl \
  --resume \
  --max-hours 5.75 \
  --require-gpu-idle \
  --quarantine-db multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl \
  --log-dir multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/logs \
  > multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/logs/energy_gpu3.nohup.log 2>&1 &
```

如果 AP/energy 队列落后，1 小时检查点后追加：

```text
generated/overnight_6h_20260626/jobs/ap_gpu2_extra_job_plan.jsonl
generated/overnight_6h_20260626/jobs/energy_gpu5_extra_job_plan.jsonl
```

## 9. 收尾导出

6 小时结束前必须执行：

```bash
cd ${V2X_ROOT}

python3 scripts/stage2_build_artifact_registry.py \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/latency/latency_lut_rows_v1.jsonl \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/latency/latency_lut_rows_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/ap/ap_anchor_rows_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/energy/energy_lut_rows_salvaged_from_payload_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/energy/energy_lut_rows_v1.jsonl \
  --quarantine-rows multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl \
  --out-jsonl multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_v1.jsonl \
  --summary-json multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_summary_v1.json

python3 scripts/stage2_detect_latency_outliers.py \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/latency/latency_lut_rows_v1.jsonl \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/latency/latency_lut_rows_v1.jsonl \
  --out-json multi_agent/data/stage2_lut_generation_v1/exports/outlier_report_latest.json \
  --out-csv multi_agent/data/stage2_lut_generation_v1/exports/outlier_report_latest.csv

python3 scripts/stage2_export_quick_review.py \
  --artifact-registry multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_v1.jsonl \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/latency/latency_lut_rows_v1.jsonl \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/latency/latency_lut_rows_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/ap/ap_anchor_rows_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/energy/energy_lut_rows_salvaged_from_payload_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/energy/energy_lut_rows_v1.jsonl \
  --outlier-report multi_agent/data/stage2_lut_generation_v1/exports/outlier_report_latest.json \
  --out-json multi_agent/data/stage2_lut_generation_v1/exports/quick_review_latest.json \
  --out-csv multi_agent/data/stage2_lut_generation_v1/exports/quick_review_latest.csv

mkdir -p multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged
cat \
  multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/latency/latency_lut_rows_v1.jsonl \
  multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/latency/latency_lut_rows_v1.jsonl \
  > multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/latency_lut_rows_merged_v1.jsonl
cat \
  multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl \
  multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/ap/ap_anchor_rows_v1.jsonl \
  > multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/ap_anchor_rows_merged_v1.jsonl
cat \
  multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/energy/energy_lut_rows_salvaged_from_payload_v1.jsonl \
  multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/energy/energy_lut_rows_v1.jsonl \
  > multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/energy_lut_rows_merged_v1.jsonl

python3 scripts/stage2_update_registry_from_luts.py \
  --registry multi_agent/data/stage2_lut_generation_v1/registry/evidence_registry_base_v1.json \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/latency_lut_rows_merged_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/ap_anchor_rows_merged_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/merged/energy_lut_rows_merged_v1.jsonl \
  --out-json multi_agent/data/stage2_lut_generation_v1/registry/evidence_registry_latest.json

python3 scripts/stage2_readiness_gate.py \
  --artifact-registry multi_agent/data/stage2_lut_generation_v1/artifacts/artifact_registry_v1.jsonl \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/latency/latency_lut_rows_v1.jsonl \
  --latency-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/latency/latency_lut_rows_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl \
  --ap-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/ap/ap_anchor_rows_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1/energy/energy_lut_rows_salvaged_from_payload_v1.jsonl \
  --energy-rows multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/energy/energy_lut_rows_v1.jsonl \
  --job-plan multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626/jobs/latency_gpu012_job_plan.jsonl \
  --evidence-registry multi_agent/data/stage2_lut_generation_v1/registry/evidence_registry_latest.json \
  --outlier-report multi_agent/data/stage2_lut_generation_v1/exports/outlier_report_latest.json \
  --out-json multi_agent/data/stage2_lut_generation_v1/exports/readiness_gate_latest.json
```

若某个导出脚本参数与本地版本不一致，以 `python3 scripts/<script>.py --help` 为准修正，但导出目标文件名保持上述 latest 路径。

## 10. 明早验收标准

| 验收项 | 通过标准 |
|---|---|
| LUT 是否相对完善 | quick review 至少 18 个 config，其中 base/p50/p75/trap25/pad64/p50b2 有 latency |
| AP 是否跟上 | 至少 12 个 config 有 AP measured 或 explicit failed/no-claim |
| energy 是否跟上 | 至少 8 个 config 有 energy measured 或 explicit failed/no-claim |
| 失败是否可解释 | job_state 中失败 config 有 failure_reason/raw log |
| latency 是否可信 | outlier report 生成，unstable 行不进入 claimable |
| 大规模下一步 | readiness gate 能明确给出 `GO_LATENCY_CONTINUE`、`GO_AP_CONTINUE`、`GO_ENERGY_CONTINUE` 或具体阻塞 |
