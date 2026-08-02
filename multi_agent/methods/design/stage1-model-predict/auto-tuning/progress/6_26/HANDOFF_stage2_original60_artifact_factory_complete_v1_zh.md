# Stage2 original60 artifact factory 完成交接 v1

日期: 2026-06-26  
主目录: `${V2X_ROOT}`  
H800: `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`  
H800 hostname: `<PRIVATE_HOST>`  
密码记录规则: 不写真实密码; 命令示例使用 `<H800_PASSWORD>`。

## 1. 本轮完成结论

本轮首要目标已经完成: `coverage_pipeline_v1/candidates/candidate_queue.jsonl` 中原始 60 个候选点的工程产物已全部生成并通过 H800 artifact gate。

最终 gate:

| gate | 结果 |
|---|---:|
| original60 candidates | 60 |
| H800 ONNX ready | 60 |
| H800 TVM workdir ready | 60 |
| `database_workload.json` ready | 60 |
| `database_tuning_record.json` ready | 60 |
| `artifact_state_original60_v1.jsonl` ready | 60 |
| `artifact_registry_original60_v1.jsonl` ready | 60 |
| active quarantine | 0 |
| missing artifact | 0 |
| local DB snapshot JSON files | 120 |
| local DB snapshot bad JSON | 0 |

本轮没有启动 latency/AP/energy measured worker, 没有写 measured LUT row。

## 2. 产物位置

H800 远端工程产物:

```text
${V2X_DATA_ROOT}/s2_tvm/models/<label>_backbone.onnx
${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>/database_workload.json
${V2X_DATA_ROOT}/s2_tvm/workdirs/<label>/database_tuning_record.json
```

本地同步产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_onnx/
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_h800_workdirs/
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/artifact_tasks_original60_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/artifact_state_original60_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/artifact_registry_original60_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/quarantine/missing_artifact_original60_v1.jsonl
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_artifact_readiness_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_artifact_readiness_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/logs/original60_artifacts/
```

完整 60 点表格见:

```text
multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/exports/original60_artifact_readiness_latest.md
```

## 3. 执行链路

新增/更新脚本:

| 脚本 | 作用 |
|---|---|
| `scripts/stage2_original60_plan_artifact_build.py` | 从 original60 candidate queue 生成 artifact build queue 和 GPU shard |
| `scripts/stage2_original60_export_onnx.py` | 本地 UniV2X/HEAL 导出 60 个 Pyramid backbone ONNX |
| `scripts/stage2_original60_tvm_artifact_worker.py` | H800 artifact-only worker, 生成 TVM workdir 和 MetaSchedule DB |
| `scripts/stage2_original60_artifact_readiness.py` | H800 artifact readiness validator/report |
| `scripts/stage2_original60_h800_launch_artifacts.sh` | H800 后台 launcher, GPU idle gate 后并行补齐 60 点 |

关键执行记录:

```text
ONNX export: 60/60, 59 new + 1 cached
ONNX local size: 371M
DB snapshot local size: 110M
H800 launcher log: logs/original60_artifacts/launcher_20260626_163718.log
```

## 4. 问题与修复

### 4.1 SSH `MaxStartups`

H800 会偶发:

```text
kex_exchange_identification: banner line 0: Exceeded MaxStartups
```

处理方式:

```text
减少 SSH 连接频率
rsync/ssh 使用低频重试
把远端操作合并成少量连接
```

### 4.2 TVM runtime CUDA 符号错误

初始 worker 失败:

```text
cudaGraphAddDependencies_v2, version libcudart.so.12
```

根因: launcher 没有注入 TVM310 CUDA runtime `LD_LIBRARY_PATH`。

固定环境:

```bash
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib:$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path):${LD_LIBRARY_PATH:-}
```

已写入:

```text
scripts/stage2_original60_tvm_artifact_worker.py
scripts/stage2_original60_h800_launch_artifacts.sh
```

### 4.3 Worker 自阻塞

初始并行 worker 一个进程串行跑多个 job。每个 job 都做 preflight, 但第一个 TVM job 后 Python 进程保留 CUDA context, 下一 job 的 `pmon` 会看到自身 PID, 导致 `preflight_blocked`。

修复: launcher 改为每个 GPU 一个 controller, controller 每轮只调用一次:

```text
stage2_original60_tvm_artifact_worker.py --max-jobs 1 --require-gpu-idle
```

子进程退出后 CUDA context 释放, 再进入下一 job。最终 6 个 GPU shard 各 10 点全部 succeeded。

## 5. 快速审查命令

本地:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
python3 - <<'PY'
import json
from pathlib import Path
base=Path('multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1')
r=json.loads((base/'exports/original60_artifact_readiness_latest.json').read_text())
print({k:r.get(k) for k in ['original60_artifact_ready_count','missing_artifact_count','active_quarantine_count','is_complete']})
print('state_rows', sum(1 for l in (base/'artifacts/artifact_state_original60_v1.jsonl').open() if l.strip()))
print('db_snapshot_files', len(list((base/'artifacts/original60_h800_workdirs').glob('*/database_*.json'))))
PY
```

预期:

```text
{'original60_artifact_ready_count': 60, 'missing_artifact_count': 0, 'active_quarantine_count': 0, 'is_complete': True}
state_rows 60
db_snapshot_files 120
```

远端:

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
ssh -p 30001 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60 \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  'cd ${V2X_ROOT}
   BASE=multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1
   python3 scripts/stage2_original60_artifact_readiness.py \
     --candidate-queue "$BASE/candidates/candidate_queue.jsonl" \
     --artifact-root ${V2X_DATA_ROOT}/s2_tvm \
     --job-state-glob "$BASE/jobs/artifact_build_state_original60_gpu*.jsonl" \
     --quarantine-file "$BASE/quarantine/missing_artifact_original60_v1.jsonl" \
     --json-out "$BASE/exports/original60_artifact_readiness_recheck.json" \
     --md-out "$BASE/exports/original60_artifact_readiness_recheck.md" \
     --require-complete'
unset password-based SSH (disabled; use an SSH key)
```

## 6. 下一阶段计划

现在可以进入 original60 的 measured LUT 数据生产阶段, 但仍需遵守:

```text
latency/energy 测量前 GPU 必须 idle
AP 只能 true eval 或 true import, 禁止 predicted AP 写 measured row
energy 与 latency 可以并行, 但 energy row 必须记录 telemetry/provenance
任何不可信结果写 no-claim 或 quarantine, 不阻断其它配置继续生产
```

建议顺序:

| 优先级 | 任务 | stop 条件 |
|---|---|---|
| P1 | 为 original60 生成 latency job plan 并跑 latency-only coverage | 60 个 candidate 至少 tuned/default latency row 或明确 quarantine |
| P1 | energy follower 与 latency 对齐 | 已有 latency 的 candidate 同步产出 energy row 或 no-claim/quarantine |
| P1 | AP source/eval 补齐 | true AP rows 或 no-claim source gap |
| P2 | supervisor 合并三轴 coverage dashboard | latency/AP/energy registry 一致, outlier gate 通过 |

下一阶段 `/goal` 建议:

```text
/goal 在 ${V2X_ROOT} 中基于已完成的 original60 artifact registry 启动 Stage2 H800 original60 measured LUT 生产。先阅读 multi_agent/methods/design/auto-tuning/progress/HANDOFF_stage2_original60_artifact_factory_complete_v1_zh.md 和 PLAN_stage2_h800_multi_agent_lut_production_v1_zh.md。使用 H800 远端 ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>, 密码由用户提供但禁止写入任何文件。前置条件是 original60_artifact_ready_count==60 且 missing_artifact_count==0。优先生成 original60 latency job plan 并在 GPU idle gate 通过后跑 latency-only coverage; energy follower 与 latency 进度对齐; AP 只允许 true eval/import, 缺 source 则写 no-claim。不得覆盖本轮 artifact registry。stop 条件是 original60 的 latency/AP/energy 三轴状态全部有 measured row 或 no-claim/quarantine 解释, supervisor dashboard 更新完成, outlier gate 通过, 结果 rsync 回本地并更新交接文档。
```
