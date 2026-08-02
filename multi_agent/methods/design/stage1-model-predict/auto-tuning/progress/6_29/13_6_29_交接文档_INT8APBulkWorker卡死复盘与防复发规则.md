# 67_6_29_交接文档_INT8APBulkWorker卡死复盘与防复发规则

## 0. 本文用途

本文记录 2026-06-29 夜间 `native_int8` AP 大规模补点 worker 的卡死问题、人工检查结论、反思和后续防复发规则。

本文是冷启动必读文档之一。后续继续 INT8 AP 大规模补点前，必须先读本文，再读：

```text
multi_agent/methods/design/auto-tuning/progress/6_27/58_6_29_冷启动交接_INT8_APFullVal补点与BNAwareSmoke复现流程.md
multi_agent/methods/design/auto-tuning/progress/6_27/60_6_29_Codex核查_INT8近无损AP实测证据链.md
multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md
```

固定口径：

```text
1. 当前路线是 H800 + TVM native INT8 backbone/subnet bridge + PyTorch FP32 head/postprocess。
2. 这不是 TRT 路线，也不是 full-network INT8。
3. row 中必须保持 full_network_claim=false。
4. AP measured row 必须来自 full-val 1789 samples，5-sample smoke 不能作为正式 AP row。
5. 单个 label 失败或卡死不能阻塞整批补点，必须 fail/skip 后继续下一个 label。
```

## 1. 当前问题概览

本轮尝试使用 H800 GPU3-6 对 original60 INT8 AP 进行批量补点。目标是复用已有 exact-label checkpoint，生成 AP 所需前置资源，并跑 full-val AP 后导入：

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

当前已经有：

```text
native_int8_original60_ap_rows_v1.jsonl: 25 rows, 25 unique labels
```

但是最新 recovery worker 没有继续产出 row。人工检查发现当前卡住点不在 AP metric 本身，而在 worker 生命周期和前置资源生成阶段。

## 2. 已发生的两类卡死

### 2.1 第一类：full-val report 已成功写出，但子进程不退出

第一轮 bulk worker 中，以下 4 个 label 已经成功写出 full-val report：

| label | AP70 | processed_samples | 结论 |
|---|---:|---:|---|
| `frontier_19` | 0.5950394125 | 1789 | report 有效，已手工导入 |
| `lhc_01` | 0.5789925241 | 1789 | report 有效，已手工导入 |
| `frontier_22` | 0.5965525044 | 1789 | report 有效，已手工导入 |
| `lhc_03` | 0.5887124856 | 1789 | report 有效，已手工导入 |

当时现象：

```text
1. full_ap_eval_report.json 已存在。
2. report status=success。
3. processed_samples=1789。
4. ap_row_allowed=true。
5. 父进程卡在 do_wait。
6. 子进程主线程已类似 zombie，但仍有 PyTorch/CUDA 残留线程占用 CPU。
7. 因 subprocess.run() 不返回，没有写 rc.txt，也没有进入 import_row。
```

判断：

```text
这不是 AP 评估结果错误，也不是 TVM route 数值失败。
本质是 full-val 子进程完成业务输出后，Python/PyTorch/CUDA runtime 没有干净退出，导致父 worker 永久等待。
```

已做处理：

```text
1. 手工导入 4 个已完成 report。
2. kill 旧的 stuck worker。
3. 修改 scripts/stage2_native_int8_ap_bulk_pipeline.py，增加 run_step_until_report()。
4. 设计为：如果 full_ap_eval_report.json 已达到 success + processed_samples>=1789，则等待 grace period 后强制结束进程组，并继续 import row。
```

### 2.2 第二类：recovery worker 在前置 export / BN-aware capture 阶段卡死

修改后启动 recovery worker：

```text
raw/int8_native_route/20260629_native_int8_ap_bulk_launcher_20260629_recover_234143
```

最新人工检查看到 GPU3-6 父进程均卡在 `do_wait`，但这次并没有 full-val report。子进程位置如下：

| GPU | label | 卡住阶段 | 子进程状态 | 说明 |
|---:|---|---|---|---|
| 3 | `frontier_16` | checkpoint-consistent ONNX export | `Dl` | 内核不可中断等待 |
| 4 | `frontier_18` | checkpoint-consistent ONNX export | `Rl` | 仍在运行，但长时间无 row 产出 |
| 5 | `frontier_25` | BN-aware reference range capture | `Dl` | 内核不可中断等待 |
| 6 | `frontier_26` | checkpoint-consistent ONNX export | `Dl` | 内核不可中断等待 |

同时 `nvidia-smi` 查询 30 秒未返回。这个信号说明问题可能已经触及 CUDA/GPU driver、文件系统等待或模型加载路径，而不是单纯 Python 业务逻辑慢。

判断：

```text
recovery 版补丁只覆盖 full-val report 已写出后的退出问题。
它没有覆盖 export、bootstrap、range capture、numeric sanity 等前置阶段的长时间卡死。
因此当前 stuck 是第二类问题：前置 step 无强超时、无隔离、无 fail-and-continue。
```

## 3. 反思

### 3.1 错误一：把 retry label 直接放回主 lane

`frontier_16/frontier_18/frontier_25/frontier_27` 等 label 在上一轮已经出现过 export 或 numeric_sanity 失败。recovery 时把它们直接放回 GPU3-6 主 lane，导致每条 lane 的第一个任务就可能卡死。

正确做法应该是：

```text
1. 已知失败 label 不进入主生产队列。
2. 已知失败 label 进入 quarantine/retry 队列。
3. retry 队列一次只跑少量 label，并设置严格 timeout。
4. retry 再失败时写 no-claim/fail artifact，不阻塞主队列。
```

### 3.2 错误二：只修了 full-val 退出，没有给所有 step 加 timeout

本轮只针对 full-val 进程不退出做了 report-success 强制收尾，但没有给以下前置 step 统一加 timeout：

```text
1. checkpoint-consistent ONNX export
2. bootstrap route
3. BN-aware reference range capture
4. scale-aware calibrated route build
5. numeric sanity
6. AP smoke
7. full-val AP
8. import row
```

只要任意 step 使用 `subprocess.run()` 且没有 timeout，整条 GPU lane 都可能永久阻塞。

### 3.3 错误三：没有把“卡死”和“失败”作为一等状态写入 registry

现在 row 只记录 measured AP，对失败配置的状态记录不足。这样下一轮调度时无法稳定区分：

```text
1. 从未尝试。
2. 尝试失败。
3. 超时。
4. 进程卡死。
5. report 成功但 import 未完成。
6. numeric sanity 不可信。
7. AP 结果可疑，应 quarantine。
```

这会导致清空上下文后，agent 重新把危险 label 当成普通待测 label。

### 3.4 错误四：没有把 GPU/driver 异常作为 stop 条件

当 `nvidia-smi` 本身无法在 30 秒内返回时，说明不能继续盲目启动更多 GPU 任务。此时应该进入巡检和隔离流程，而不是继续补点。

## 4. 防复发规则

### 4.1 主生产队列规则

主生产队列只允许进入满足以下条件的 label：

```text
1. 有合法 exact-label checkpoint。
2. 不在 quarantine label list。
3. 最近一次尝试不是 export_hang / capture_hang / gpu_driver_suspect。
4. 同一 label 没有正在运行的 active job。
5. 同一 label 没有已完成且待导入的 valid full_ap_eval_report.json。
```

以下 label 暂时必须进入 quarantine，不允许作为 lane 首个任务直接启动：

```text
frontier_16
frontier_18
frontier_25
frontier_27
frontier_01
```

说明：

```text
frontier_16/frontier_18/frontier_27: 曾出现 export failure 或 export hang。
frontier_25: 曾出现 numeric_sanity failure，本轮又在 range capture 卡住。
frontier_01: 已有 AP70=0.0 的可疑 measured row，必须单独复查，不能作为正常近无损证据。
```

### 4.2 每个 step 必须有 timeout

后续修改 worker 时，所有外部命令必须走统一 runner，不允许直接裸用 `subprocess.run()`。

建议 timeout 初值：

| step | 建议 timeout | 超时后的状态 |
|---|---:|---|
| ONNX export | 20 min | `export_timeout` |
| bootstrap route | 20 min | `bootstrap_timeout` |
| BN-aware range capture | 30 min | `range_capture_timeout` |
| calibrated route build | 30 min | `route_build_timeout` |
| numeric sanity | 15 min | `numeric_sanity_timeout` |
| 5-sample AP smoke | 20 min | `ap_smoke_timeout` |
| full-val AP | 180 min | `full_ap_timeout` |
| import row | 5 min | `import_timeout` |

超时处理必须满足：

```text
1. kill 子进程组。
2. 写 rc.txt。
3. 写 failure_report.json。
4. 写 label_attempt_state.json。
5. 刷新 queue/coverage。
6. 继续下一个 label。
```

### 4.3 report-success 优先规则

full-val AP 阶段使用特殊规则：

```text
如果 full_ap_eval_report.json 已经存在，且满足：
1. status=success
2. processed_samples>=1789
3. failed_samples=0
4. ap_row_allowed=true

则视为业务成功。
即使评估进程没有自然退出，也应在 grace period 后 kill process group，然后执行 import row。
```

这条规则只适用于 full-val AP，不适用于 export/range capture 等前置阶段。

### 4.4 GPU 健康门控

启动任何 H800 worker 前必须执行：

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  'timeout 30s nvidia-smi && timeout 30s nvidia-smi pmon -c 1'
```

如果 `nvidia-smi` 30 秒内不返回：

```text
1. 不启动新任务。
2. 不继续排队补点。
3. 记录 gpu_driver_suspect。
4. 检查是否有 D-state CUDA/Python 子进程。
5. 必要时请求人工处理 GPU/driver 状态。
```

### 4.5 lane 调度规则

后续 GPU3-6 不应使用“每个 GPU 一长串 label 串行跑到底”的朴素模式。建议改为：

```text
1. 每个 label 是独立 job。
2. 每个 job 有 job_id、label、gpu_id、step、timeout、attempt_id。
3. 每个 GPU lane 一次只取一个 non-quarantine label。
4. job 完成、失败或超时后立即释放 lane。
5. supervisor 每 5-10 分钟轮询 active jobs。
6. stuck job 不允许阻塞后续 label。
```

## 5. 当前产物位置

H800 远端：

```text
V2X=${V2X_ROOT}
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
RAW_PARENT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
ROWS=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

当前 latest launcher：

```text
$RAW_PARENT/native_int8_ap_bulk_latest_launcher_dir.txt
```

截至本文记录，latest launcher 指向：

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260629_native_int8_ap_bulk_launcher_20260629_recover_234143
```

当前 stuck 父进程：

| GPU | parent pid | 状态 |
|---:|---:|---|
| 3 | 3646109 | `do_wait` |
| 4 | 3646114 | `do_wait` |
| 5 | 3646120 | `do_wait` |
| 6 | 3646125 | `do_wait` |

注意：这些 pid 是本文撰写时的现场状态。后续窗口必须重新巡检，不要假设 pid 仍然存在。

## 6. 下一步处理计划

### P0：先止血，不继续盲跑

1. 重新巡检 H800 GPU3-6：

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  'timeout 30s nvidia-smi; timeout 30s nvidia-smi pmon -c 1'
```

2. 如果 `nvidia-smi` 仍不返回，暂停所有新任务。
3. 如果 stuck 子进程仍在 `D` 状态，不要反复 kill；记录后交给人工处理或等待内核态释放。
4. 将 `frontier_16/frontier_18/frontier_25/frontier_27/frontier_01` 写入 quarantine 清单。

### P1：修 worker runner

修改：

```text
scripts/stage2_native_int8_ap_bulk_pipeline.py
```

要求：

```text
1. 所有 step 统一走 timeout runner。
2. 所有 step 写 step_start/step_end/rc/stdout/stderr/failure_report。
3. timeout 后 kill process group。
4. timeout 后继续下一个 label。
5. full-val AP 保留 report-success 优先规则。
6. job 结束后刷新 coverage/queue。
```

### P2：重建生产队列

生成两个队列：

```text
1. production_queue: 非 quarantine、未测、checkpoint 合法的 label。
2. quarantine_queue: 曾失败/超时/卡死/AP 可疑的 label。
```

生产优先级：

```text
1. 先跑 production_queue，快速扩大 LUT 覆盖。
2. quarantine_queue 单独开小批次验证，不得阻塞主生产。
3. 每个 GPU lane 的第一个任务必须是低风险 label。
```

### P3：重新启动补点

只有满足以下条件后，才能重新启动 GPU3-6 AP bulk：

```text
1. nvidia-smi 30 秒内正常返回。
2. 无未知用户任务被误杀风险。
3. runner 已覆盖所有 step timeout。
4. quarantine 生效。
5. latest queue 中没有把危险 label 放在每条 lane 的第一个任务。
```

## 7. 冷启动检查命令

### 7.1 查看最新 launcher 和进程

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
cd ${V2X_ROOT}
LAUNCH=$(cat multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/native_int8_ap_bulk_latest_launcher_dir.txt 2>/dev/null || true)
echo LAUNCH=$LAUNCH
for g in 3 4 5 6; do
  pid=$(cat $LAUNCH/gpu${g}.pid 2>/dev/null || true)
  echo GPU$g pid=$pid
  ps -p $pid -o pid,ppid,stat,wchan:24,etime,cmd 2>/dev/null || true
  ps --ppid $pid -o pid,ppid,stat,wchan:24,etime,pcpu,pmem,cmd --forest 2>/dev/null || true
done
'
```

### 7.2 查看 INT8 AP row 数

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
cd ${V2X_ROOT}
python3 - <<PY
import json
from pathlib import Path
root=Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627")
p=root/"rows/native_int8_original60_ap_rows_v1.jsonl"
rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
print("rows", len(rows), "unique", len({r.get("label") for r in rows}))
for r in rows[-20:]:
    print(r.get("label"), r.get("metric_value"), r.get("run_id"))
PY
'
```

### 7.3 查找成功 report 但未导入的结果

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
cd ${V2X_ROOT}
find multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route \
  -name full_ap_eval_report.json -printf "%T@ %p\n" | sort -n | tail -40
'
```

## 8. 给下一位 agent 的明确要求

下一位 agent 不要直接说“继续大规模补点”。必须先完成：

```text
1. 确认 H800 GPU/driver 健康。
2. 隔离 quarantine labels。
3. 修复 worker 全 step timeout。
4. 验证一个 non-quarantine label 可以完整跑通并自动 import。
5. 再启动 GPU3-6 并行补点。
```

如果发现 `nvidia-smi` 卡住或大量 Python/CUDA 进程处于 `D` 状态，应向用户汇报“需要人工处理 GPU/driver 状态”，而不是继续启动新任务。

