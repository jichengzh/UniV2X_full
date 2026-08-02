# Stage2 H800 服务器访问与实验运行 Runbook v1

日期: 2026-06-26

目的: 固定 H800 服务器连接方式、远端路径、实验启动前检查和常见故障处理。清空上下文后, 先读本文件, 再读最新交接文档。

## 1. 服务器信息

| 项 | 值 |
|---|---|
| SSH host | `<PRIVATE_HOST>` |
| SSH port | `30001` |
| SSH user | `<LOCAL_USER>` |
| 已验证 hostname | `<PRIVATE_HOST>` |
| GPU | 8 x NVIDIA H800, 每卡约 81559 MiB |
| 主要远端工作目录 | `${V2X_ROOT}` |
| TVM/H800 artifact 根目录 | `${V2X_DATA_ROOT}/s2_tvm` |
| TVM Python | `${V2X_DATA_ROOT}/tvm310/bin/python` |

安全约束:

1. 不要把 H800 密码写入仓库、job plan、脚本或日志。
2. 需要非交互登录时使用环境变量 `password-based SSH (disabled; use an SSH key)`, 命令结束后 `unset password-based SSH (disabled; use an SSH key)`。
3. 清空上下文后若不知道密码, 向用户确认; 不要猜测或把历史口述密码落盘。

## 2. 推荐连接命令

交互登录:

```bash
ssh -p 30001 \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=60 \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

非交互单条命令:

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
ssh -p 30001 \
  -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=60 \
  -o ServerAliveInterval=30 \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST> \
  'hostname; pwd; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv; nvidia-smi pmon -c 1'
unset password-based SSH (disabled; use an SSH key)
```

从本地同步脚本或结果:

```bash
export password-based SSH (disabled; use an SSH key)='<H800_PASSWORD>'
rsync -avR \
  -e 'ssh -p 30001 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60' \
  scripts/stage2_lut_worker.py \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/
unset password-based SSH (disabled; use an SSH key)
```

## 3. 远端环境约定

Stage2 H800 TVM 测量统一使用:

```bash
cd ${V2X_ROOT}
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib:$LD_LIBRARY_PATH
```

常用 artifact 路径:

```text
${V2X_DATA_ROOT}/s2_tvm/models
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_base
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_p50
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_p75_retest
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_trap25
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_pad64_retest
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_iso_s0_retest
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_iso_s1_retest
${V2X_DATA_ROOT}/s2_tvm/ms_work_2e_iso_s2_retest
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_s0_16
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_s0_32
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_s1_32
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_s2_64
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_s2_128_v2
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_mix_a
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_mix_b
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_mix_c
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_mix_d
${V2X_DATA_ROOT}/s2_tvm/ms_bumped_mix_e
```

注意:

1. 远端 `${V2X_ROOT}` 可能是 rsync 工作副本, 不一定是权威 git repo。
2. 本地 `${V2X_ROOT}` 是文档和脚本编辑的主路径; 改脚本后需要显式 rsync 到 H800。
3. H800 measured rows 只能在 H800 远端生成; 本地 RTX 4090 不能写 `backend=h800_tvm` 的 measured row。

## 4. GPU preflight 硬规则

每个 latency/energy job 启动前必须保存:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv
nvidia-smi pmon -c 1
```

空闲判定:

| 条件 | 处理 |
|---|---|
| target GPU utilization > 5% | 不启动 measured job, 写 `preflight_blocked` |
| target GPU 有非本 job compute process | 不启动 measured job |
| target GPU memory used > 1024 MiB 且不是系统保留 | 不启动 measured job |
| `pmon` 无进程但 worker PID 仍在 | 可能处于 build/preflight/切换阶段, 等 30-60 秒再看 |

`nvidia-smi` 输出常见格式是带单位的, 如 `4 MiB`, `0 %`。preflight parser 必须提取数字, 不能直接 `float("4 MiB")`。

## 5. 常见问题

### 5.1 SSH rate limit

症状:

```text
kex_exchange_identification: Connection closed by remote host
Connection closed by <PRIVATE_HOST> port 30001
```

原因: 远端 SSH `MaxStartups` / rate-limit。处理:

1. 停止并行 SSH/rsync。
2. 等 15-30 秒再单连接重试。
3. 长时间实验减少轮询频率, 一次 SSH 内完成多项检查。

### 5.2 H800 上看不到进程

不要直接判定失败。先检查:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626
for p in "$BASE"/jobs/*.pid; do
  echo "$p $(cat "$p" 2>/dev/null)"
  ps -p "$(cat "$p" 2>/dev/null)" -o pid=,stat=,etime=,cmd= || true
done
python3 -m json.tool "$BASE/exports/current_experiment_report_v1.json" | sed -n '1,120p'
```

有限队列跑完后 worker 会退出。`--max-hours 6` 是最长运行上限, 不是保持 GPU 忙 6 小时。若要持续运行, 必须生成足够长的 queue 或 top-up continuation queue。

### 5.3 TVM/CUDA illegal memory access

典型症状:

```text
CUDA: an illegal memory access was encountered
```

处理:

1. 当前 job 写 failed state 和 raw artifact。
2. 对应 config/workdir 进入 bad-DB quarantine。
3. 后续 worker 应跳过该 config, 不能继续写 measured claim。
4. 如需恢复, 必须重建 ONNX/MetaSchedule DB 或换 known-good workdir 后单点复测。

当前已知风险:

| config | 风险 |
|---|---|
| `s1_64` | 历史 tuned latency illegal memory 风险 |
| `pad64` | continuation repeat 中复现 illegal memory, 当前 quarantined |
| `mix_a` | energy 与部分 continuation latency 中复现 illegal memory, 当前 quarantined |

### 5.4 AP source 缺失

AP measured row 只能来自真实 eval/import source。缺 source 时写 failed job_state, 不能写 predicted AP measured row。

当前已有 true AP anchor 的 label:

```text
base, p50, p75, trap25, p50b2_136, iso_s0, iso_s1, iso_s2, mix_a, mix_b, mix_d
```

当前缺 true AP source 的 label:

```text
pad64, s0_16, s0_32, s1_32, s2_64, s2_128, mix_c, mix_e
```

### 5.5 Missing artifact

`p50b2_136` 当前有 AP anchor, 但 H800 latency artifact 缺失:

```text
p50b2_136_backbone.onnx / work_dir missing
```

补测前必须先在 artifact registry 中变成 ready, 再启动 latency/energy。

## 6. 当前关键数据目录

本轮 H800 6h/continuation 实验远端原始目录:

```text
H800:${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626
```

本地已同步副本:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626
```

快速审查:

```bash
cd ${V2X_ROOT}
BASE=multi_agent/data/stage2_lut_generation_v1/generated/overnight_6h_20260626
python3 -m json.tool "$BASE/exports/readiness_gate_latest.json" | sed -n '1,120p'
python3 -m json.tool "$BASE/registry/artifact_registry_summary_v1.json"
head -n 20 "$BASE/exports/quick_review_v1.csv"
wc -l "$BASE"/merged/*.jsonl "$BASE"/registry/artifact_registry_v1.jsonl
```
