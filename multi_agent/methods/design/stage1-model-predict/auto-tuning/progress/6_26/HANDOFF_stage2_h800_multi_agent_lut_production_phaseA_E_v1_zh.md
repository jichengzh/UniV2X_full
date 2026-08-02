# Stage2 H800 Multi-Agent LUT Production Phase A-E 交接 v1

日期: 2026-06-26  
主目录: `${V2X_ROOT}`  
H800 远端: `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`  
H800 hostname: `<PRIVATE_HOST>`  
密码记录规则: 不写真实密码; 命令示例统一使用 `<H800_PASSWORD>`。

## 0. 最新状态: original60 artifact factory 已完成

更新时间: 2026-06-26 16:58 CST  
状态优先级: 本节覆盖下方旧章节中“原始 60 点仍 missing / 不能启动”的历史结论。

本轮已完成 `coverage_pipeline_v1/candidates/candidate_queue.jsonl` 中原始 60 个 candidates 的工程产物生成。当前 stop gate 已满足:

```text
original60_artifact_ready_count = 60
missing_artifact_count = 0
active_quarantine_count = 0
artifact_state_original60_v1.jsonl = 60 rows, ready=60
artifact_registry_original60_v1.jsonl = 60 rows, ready=60
artifact_tasks_original60_v1.jsonl = 60 rows, ready=60
missing_artifact_original60_v1.jsonl = 0 rows
local DB snapshot = 120 JSON files, bad=0
```

H800 工程产物路径:

```text
${V2X_DATA_ROOT}/s2_tvm/models/<label>_backbone.onnx
${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>/database_workload.json
${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>/database_tuning_record.json
```

本地同步结果:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_onnx/
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_h800_workdirs/
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/artifact_state_original60_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/artifact_registry_original60_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_artifact_readiness_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_artifact_readiness_latest.md
```

完整 60 点 artifact 表在:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_artifact_readiness_latest.md
```

执行中修复了两个关键机制问题:

1. TVM runtime 环境必须前置:
   ```text
   PATH=/usr/local/cuda-12.2/bin:$PATH
   LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib:$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path):$LD_LIBRARY_PATH
   ```
   否则会复现 `cudaGraphAddDependencies_v2` undefined symbol。
2. artifact launcher 必须每个 job 使用独立 worker 子进程。单个 Python/TVM worker 连续处理多个 job 会保留 CUDA context, 下一 job 的 `pmon` preflight 会把自身识别为 GPU 忙, 产生自阻塞。

新增/更新脚本:

```text
scripts/stage2_original60_plan_artifact_build.py
scripts/stage2_original60_export_onnx.py
scripts/stage2_original60_tvm_artifact_worker.py
scripts/stage2_original60_artifact_readiness.py
scripts/stage2_original60_h800_launch_artifacts.sh
```

新的完成交接文档:

```text
multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_original60_artifact_factory_complete_v1_zh.md
```

下一阶段才能进入 original60 的 latency/AP/energy 数据生产。注意: 本轮没有写 latency/AP/energy measured row。

## 1. 本轮结论

本轮按 `PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md` 的 Phase A-E 执行了 agent team 分工和真实 H800 链路验证。

关键结论:

1. 原始 `candidate_queue.jsonl` 的 60 个 coverage synthetic candidates 仍然是 `60/60 artifact_status=missing`, 不能进入 latency/AP/energy worker。
2. 已新建并执行 ready-first 重锚分支: `candidate_queue_ready_first.jsonl`, 共 6 个 H800 已有 artifact-ready candidates。
3. ready-first latency 已完成真实 H800 测量: 6 个 width x 2 个 schedule = 12 条 measured latency row。
4. AP 队列为 0, 6 个 ready-first candidate 都因缺真实 AP source 写入 no-claim gap, 没有 predicted AP measured row。
5. energy 初始 2 个任务失败, 根因是系统 `python3` 缺 TVM `PYTHONPATH`; 已修复代码并验证 H800 上 `python3` 能 import TVM。
6. 修复后已生成可重跑的 `energy_job_queue_ready_first_envfix_after_latency.jsonl`, 共 6 个 energy jobs, 但当前 H800 GPU 被 `user01` 的 Qwen 服务占用, 按 GPU idle 规则不能启动 energy。
7. 当前 ready-first supervisor gate 是 `CONDITIONAL_GO`: latency 链路通过, energy 可重跑但等 GPU 空闲, AP 等真实 source。

## 2. Agent Team 分工与结果

| agent | 分工 | 本轮结果 |
|---|---|---|
| artifact-agent | 发现 H800 artifact, 生成 artifact_state/registry, 不写 measured row | 原始 60 点确认 missing; ready-first 6 点全部 ready |
| latency-agent | 消费 GPU0/1/4/5 latency queue, 扩大 unique latency coverage | 12/12 succeeded |
| energy-agent | GPU2, 跟随 latency 成功 cell 生成 energy | 初始 2/2 failed; env 已修复; envfix 6-job queue ready; 等 GPU 空闲 |
| AP-agent | GPU3 或 true import, 只写真实 AP row | 0 queued; 6 no-claim, reason=`missing AP source_path` |
| supervisor-agent | 汇总 queue/state/rows/gap/quarantine | ready-first gate=`CONDITIONAL_GO`; 原始 60 点仍 `NO_GO` |

## 3. Phase A-E 执行状态

### Phase A: 同步脚本与 H800 Preflight

已完成:

- 同步 Stage2 LUT 脚本、framework/stage2 和 ready-first candidate 到 H800。
- H800 Python: `3.10.12`。
- 初始 preflight 中 GPU0/1/2/4/5 符合空闲规则; GPU3 曾有进程, 但本轮 AP queue=0, 未使用 GPU3。

注意:

- H800 登录侧会出现 SSH `kex_exchange_identification: Connection closed by remote host` 限流。实际操作必须降低连接频率, 单连接批量执行, 连接间隔建议 3-10 分钟。

### Phase B: Artifact Discovery 与候选重锚

原始 60 点:

```text
candidates=60
artifact_status_counts={"missing":60}
ap_status_counts={"source_missing":60}
```

原因: candidate label/config/width 与 H800 已有 registry 无可验证 overlap, 且默认 fallback 路径 `${V2X_DATA_ROOT}/s2_tvm/models/<label>_backbone.onnx` / `workdirs/<label>` 不存在。

ready-first 6 点:

```text
candidates=6
artifact_status_counts={"ready":6}
ap_status_counts={"source_missing":6}
```

文件:

- `multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/candidates/candidate_queue_ready_first.jsonl`
- `.../artifacts/artifact_state_ready_first.jsonl`
- `.../artifacts/artifact_registry_ready_first_v1.jsonl`

### Phase C: 生成 latency/AP/energy 队列

生成结果:

| 队列 | 文件 | 数量 | 状态 |
|---|---|---:|---|
| latency | `jobs/ready_first_latency/latency_job_queue_gpu0.jsonl` | 3 | completed |
| latency | `jobs/ready_first_latency/latency_job_queue_gpu1.jsonl` | 3 | completed |
| latency | `jobs/ready_first_latency/latency_job_queue_gpu4.jsonl` | 3 | completed |
| latency | `jobs/ready_first_latency/latency_job_queue_gpu5.jsonl` | 3 | completed |
| AP | `jobs/ap_job_queue_ready_first.jsonl` | 0 | no-claim gap only |
| energy 初始 | `jobs/energy_job_queue_ready_first.jsonl` | 2 | old env failed |
| energy 修复后 | `jobs/energy_job_queue_ready_first_envfix_after_latency.jsonl` | 6 | ready, waiting GPU idle |

### Phase D: 启动 worker

latency:

- GPU0/1/4/5 worker 已启动并完成。
- 最新状态按 job_id 去重后为 `12 succeeded / 0 failed`。

energy:

- 初始 GPU2 worker 启动 2 个任务, 均失败。
- 失败根因: `ModuleNotFoundError("No module named 'tvm'")`。
- 已修复并同步:
  - `framework/stage2/lut_productization.py`: `build_tvm_runtime_env()` 现在前置 repo root 和 `${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages`。
  - `scripts/stage2_h800_run_measurement_job.py`: `configure_tvm_env()` 不再覆盖 `PYTHONPATH`, 会保留 repo root、TVM site-packages 和已有路径。
- H800 验证: 系统 `python3` 在新 env 下可 import TVM, 版本 `0.20.dev1070+gb628d91fa`。

AP:

- 未启动 worker, 因 AP queue=0。
- 6 个 candidate 已写 no-claim gap, 禁止将 predicted AP 作为 measured AP。

### Phase E: Top-up 与容错

已完成:

1. 用 ready-first 分支规避原始 60 点 artifact missing。
2. latency smoke 先每卡 1 job, 通过后展开剩余队列。
3. energy 失败后定位到 env bug, 修复并生成新 tag 队列 `coverage_envfix`, 避免旧 failed state 的 max-attempt 阻塞。
4. supervisor ready-first 分支输出:

```text
decision=CONDITIONAL_GO
unique_latency_cells=12
unique_ap_cells=0
unique_energy_cells=0
repeat_ratio=0.0
missing_artifact_config_count=0
latency_planned_jobs=12
latency_succeeded_jobs=12
energy_planned_jobs=6
energy_ready_jobs=6
ap_planned_jobs=0
```

输出目录:

- `.../exports/ready_first_supervisor_envfix/readiness_gate_latest.json`
- `.../exports/ready_first_supervisor_envfix/coverage_dashboard_latest.csv`
- `.../exports/ready_first_supervisor_envfix/supervisor_report_latest.md`

## 4. 当前实测数据表

单位统一为 ms。energy/AP 当前没有 measured row。

| label | width | default latency ms | tuned latency ms | tuned/default | AP 状态 | energy 状态 |
|---|---:|---:|---:|---:|---|---|
| `mix_f` | `[32,64,64]` | 22.347 | 19.418 | 1.151x | no-claim: missing AP source | envfix queue ready |
| `wg_pair5` | `[48,32,128]` | 28.162 | 24.714 | 1.140x | no-claim: missing AP source | envfix queue ready |
| `wg_pair6` | `[48,64,192]` | 37.268 | 21.157 | 1.762x | no-claim: missing AP source | envfix queue ready |
| `wg_pair4` | `[48,96,256]` | 46.529 | 23.737 | 1.960x | no-claim: missing AP source | envfix queue ready |
| `pg_pair5` | `[64,32,128]` | 33.150 | 11.291 | 2.936x | no-claim: missing AP source | envfix queue ready |
| `pg_pair6` | `[64,64,192]` | 42.210 | 42.819 | 0.986x | no-claim: missing AP source | envfix queue ready |

Latency row 文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/latency_lut_rows_ready_first_v1.jsonl
```

## 5. 当前阻塞项

### 5.1 H800 GPU 不满足 idle 规则

当前 H800 上有非本轮实验进程占用所有 GPU:

```text
PID 1543941
USER user01
CMD python serve_qwen_ra_grpo.py --base /data/planner_study/models/qwen2.5-3b-instruct ...
```

该进程挂在 GPU0-7, GPU2 memory 约 3261 MiB, pmon 有 compute process。按规则:

```text
GPU util <= 5%
memory used <= 1024 MiB
无非本 job compute process
```

因此不能启动 energy worker, 也不能继续新的 latency/AP GPU job。不要 kill 该进程, 除非用户明确确认这是可停止进程。

最新状态更新: 用户已反馈 H800 GPU 已重新空出。下一轮仍必须先做真实 `nvidia-smi` / `pmon` preflight, 但执行目标不再优先补 energy; GPU 资源优先用于原始 60 个 coverage candidates 的工程产物生成。

### 5.2 AP 缺真实 source

ready-first 6 点没有 `ap_source_path`, 只能写 no-claim gap。下一步需要:

1. 从已有 AP true eval/import artifact 反向补 `ap_source_path`。
2. 或对同一 candidate/width 跑真实 AP eval/import。
3. 继续禁止 predicted AP measured row。

### 5.3 原始 60 点仍缺 artifact

原始 coverage queue 不可启动。下一步必须二选一:

1. artifact-agent 从 H800 已有 artifact 继续重锚更多 ready-first candidates, 目标 >=30 ready。
2. 对原始 60 点生成 compile/export backlog, 完成 ONNX/workdir/DB 后再单点 smoke。

## 6. GPU 空闲后的恢复命令

先检查 H800:

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
ssh -p 30001 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60 \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  'cd ${V2X_ROOT} && nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv && nvidia-smi pmon -c 1'
unset password-based SSH (disabled; use an SSH key)
```

若 GPU2 满足 idle 规则, 启动 envfix energy:

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
ssh -p 30001 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60 \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  'cd ${V2X_ROOT}
   BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
   mkdir -p "$BASE/logs/ready_first_energy"
   nohup python3 scripts/stage2_lut_worker.py \
     --job-plan "$BASE/jobs/energy_job_queue_ready_first_envfix_after_latency.jsonl" \
     --job-state "$BASE/jobs/energy_job_state_ready_first_envfix_after_latency.jsonl" \
     --max-hours 6 \
     --resume \
     --log-dir "$BASE/logs/ready_first_energy/gpu2_envfix_after_latency" \
     --quarantine-db "$BASE/quarantine/bad_db_quarantine_v1.jsonl" \
     --require-gpu-idle \
     > "$BASE/logs/ready_first_energy/gpu2_envfix_after_latency.out" 2>&1 &
   echo $! > "$BASE/jobs/energy_gpu2_ready_first_envfix_after_latency.pid"'
unset password-based SSH (disabled; use an SSH key)
```

完成后拉回结果:

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
printf '%s\n' \
  multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/energy_job_state_ready_first_envfix_after_latency.jsonl \
  multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/rows/energy_lut_rows_ready_first_v1.jsonl \
| rsync -av --files-from=- \
  -e 'ssh -p 30001 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60' \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/ \
  ${V2X_ROOT}/
unset password-based SSH (disabled; use an SSH key)
```

## 7. 下一阶段计划: 原始 60 候选 artifact factory 优先

### 7.1 目标重锚

下一轮唯一 P0 目标:

```text
为 coverage_pipeline_v1/candidates/candidate_queue.jsonl 中的原始 60 个 candidates
全部生成 H800 工程产物, 并通过 artifact gate。
```

本阶段不启动以下任务:

```text
不启动 latency LUT measurement worker
不启动 energy telemetry worker
不启动 AP eval/import worker
不继续 ready-first 6 点补齐
不把 partial artifact 当作 ready
```

### 7.2 原始 60 点现状

当前原始 queue:

```text
candidate_queue.jsonl: 60 rows
artifact_state.jsonl: 60 rows, artifact_status=missing
missing_artifact_v1.jsonl: 60 rows
```

候选分布:

| group | count | 说明 |
|---|---:|---|
| `s0_*` | 3 | stage0 width sweep |
| `s1_*` | 4 | stage1 width sweep |
| `s2_*` | 3 | stage2 width sweep |
| `lhc_*` | 25 | Latin-hypercube coverage |
| `frontier_*` | 25 | frontier/coverage expansion |

每个 candidate 必须最终具备:

| artifact | 目标路径规则 | ready 条件 |
|---|---|---|
| ONNX | `${V2X_DATA_ROOT}/s2_tvm/models/<label>_backbone.onnx` | 文件存在, 非空, 可被 ONNX loader 读取 |
| TVM workdir | `${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>` 或明确 registry path | 目录存在 |
| MetaSchedule workload | `<tvm_work_dir>/database_workload.json` | 文件存在, 非空, JSON 可读 |
| MetaSchedule tuning record | `<tvm_work_dir>/database_tuning_record.json` | 文件存在, 非空, JSON/JSONL 可读 |
| artifact registry row | `artifact_registry_v1.jsonl` | `artifact_status=ready`, `missing_artifacts=[]` |

### 7.3 P0 执行步骤

P0-A: H800 preflight 与同步。

```text
1. 单连接登录 H800, 确认 hostname=<PRIVATE_HOST>。
2. 检查 GPU0-5 idle; 若某 GPU 不 idle, 从 artifact worker pool 排除该 GPU。
3. 同步最新 scripts/framework/stage2/coverage_pipeline_v1。
4. 不在任何文件中写真实密码。
```

P0-B: 构建 original60 artifact build queue。

输出文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/artifact_build_queue_original60.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/jobs/artifact_build_state_original60.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/artifact_build_manifest_original60.jsonl
```

要求:

```text
queue 必须包含 60 个 candidate_id
每个 candidate 显式记录 label,width,onnx_path,tvm_work_dir,database_workload_path,database_tuning_record_path
不得把 ready-first 6 点混入 original60 queue
```

P0-C: 并行生成 ONNX。

建议策略:

```text
artifact-agent 使用 GPU0-5 并行, 每 GPU 串行处理一个 candidate
先导出所有缺失 ONNX
每个 ONNX 写 export log, 包括 label,width,source command,returncode,sha256,size_bytes
ONNX export 成功但 TVM DB 未生成时, artifact_status 仍不能是 ready
```

P0-D: 并行生成 TVM workdir / MetaSchedule DB。

建议策略:

```text
每个 candidate 使用同一 label 的 ONNX 输入
workdir 统一落到 ${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>
生成 database_workload.json 和 database_tuning_record.json
若调参时间过长, 允许低 trial smoke DB 先完成 artifact-ready seed, 但必须在 registry 中标注 tuning_budget/trials
发生 CUDA illegal memory / bad DB 时写 quarantine, 修复后继续, 不允许跳过 candidate 后宣布完成
```

P0-E: artifact-only validation。

验证不写 latency/AP/energy measured row, 只检查工程产物可用:

```text
1. ONNX load pass
2. TVM Relax from_onnx pass
3. MetaSchedule DB path exists
4. ApplyDatabase 或等价 DB load smoke pass
5. artifact planner 返回 60/60 ready
```

P0-F: 更新 registry/state 并同步回本地。

目标文件:

```text
artifacts/artifact_state_original60_v1.jsonl
artifacts/artifact_registry_original60_v1.jsonl
artifacts/artifact_tasks_original60_v1.jsonl
quarantine/missing_artifact_original60_v1.jsonl
exports/artifact_original60_readiness_v1.json
exports/artifact_original60_report_v1.md
```

完成后再将 original60 ready registry 合并/提升到主 artifact registry, 但只有在 60/60 ready 后才能覆盖主 `artifact_state.jsonl` / `artifact_registry_v1.jsonl`。

### 7.4 Stop 条件

下一轮 stop 条件必须是:

```text
original60_artifact_ready_count == 60
original60_missing_artifact_count == 0
original60_quarantine_active_count == 0
60 个 ONNX 路径在 H800 真实存在且非空
60 个 TVM workdir 在 H800 真实存在
60 个 database_workload.json 和 database_tuning_record.json 在 H800 真实存在且非空
artifact planner / validator 对 original60 输出 60/60 ready
结果已 rsync 回本地
本交接文档已更新最终 artifact 表
```

未满足上述条件时, 不得把任务总结为“完成”。如果单个 candidate 失败, 下一步必须是修复该 candidate 的 export/tuning/DB, 而不是转向 latency/AP/energy 补点。

### 7.5 后续阶段顺序

只有 P0 stop 条件满足后, 才进入后续阶段:

| 阶段 | 内容 | 进入条件 |
|---|---|---|
| P1 | 用 original60 artifact 生成 latency queues | 60/60 artifact ready |
| P2 | latency measured rows 扩展 | latency queues 非空且 GPU idle |
| P3 | energy follower | 对应 latency 已成功 |
| P4 | AP true source/eval/import | AP source 明确存在 |
| P5 | registry export / predictor seed | 三轴 rows 或 no-claim gap 完整 |

## 8. 本轮验证命令

本地通过:

```bash
python3 -m unittest \
  framework.tests.test_stage2_lut_productization.Stage2LutProductizationTest.test_tvm_runtime_env_preserves_repo_and_tvm_pythonpath \
  framework.tests.test_stage2_energy_coverage_jobs \
  framework.tests.test_stage2_lut_productization.Stage2EnergyClaimGateTest
```

结果:

```text
Ran 7 tests
OK
```

语法检查通过:

```bash
python3 -m py_compile \
  framework/stage2/lut_productization.py \
  scripts/stage2_h800_run_measurement_job.py \
  scripts/stage2_lut_worker.py
```

H800 验证通过:

```text
TVM_IMPORT_RC 0
TVM_IMPORT_OUT 0.20.dev1070+gb628d91fa
```

## 9. 大规模启动判定

当前不能宣布大规模启动。

原因:

1. ready-first 只有 6 个 artifact-ready candidates, 小于大规模 gate 的 >=30。
2. latency 已通过但队列已消费完, 下一批需要 artifact top-up。
3. energy env 已修复但还没有 measured energy row, 当前被 GPU 占用阻塞。
4. AP 只有 no-claim gap, 没有真实 AP measured row。
5. 原始 60 点仍然是 compile/export backlog, 不能直接跑。

当前可以宣布:

```text
H800 latency 产品化链路可用;
artifact-ready 分支可从 candidate -> queue -> worker -> row 闭环;
energy Python/TVM 环境问题已修复并有 6-job retriable queue;
下一步首要任务已重锚为 original60 artifact factory, 暂不补 energy/AP/latency。
```

## 10. 下一轮 /goal 命令

建议下一次清空上下文后直接使用以下 `/goal`:

```text
/goal 在 ${V2X_ROOT} 中执行 Stage2 H800 original60 artifact factory。先阅读 multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_h800_multi_agent_lut_production_phaseA_E_v1_zh.md 和 PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md。使用 H800 远端 ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>, 密码由用户提供但禁止写入任何文件。首要且唯一 P0 任务是为 multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/candidates/candidate_queue.jsonl 中的原始 60 个 candidates 生成工程产物: ${V2X_DATA_ROOT}/s2_tvm/models/<label>_backbone.onnx, ${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>, database_workload.json, database_tuning_record.json, 并更新 original60 artifact build queue/state/manifest/registry/report。可以使用 GPU0-5 并行, 但每个 artifact job 必须先做 nvidia-smi 和 pmon preflight; 忙碌 GPU 不能使用。不启动 latency/AP/energy measured worker, 不补 ready-first 6 点, 不把 partial artifact 标成 ready。stop 条件必须是 original60_artifact_ready_count==60, missing_artifact_count==0, active_quarantine_count==0, 60 个 ONNX/TVM workdir/MetaSchedule DB 路径都在 H800 真实存在且非空, artifact planner/validator 输出 60/60 ready, 结果 rsync 回本地, 并更新交接文档。未达到这个条件不得总结为完成。
```
