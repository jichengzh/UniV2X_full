# 44_6_28_交接文档_阶段三_INT8FullAPv2运行中与FP16CheckpointInventory

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `43_6_28_交接文档_阶段三_EmptyPredictionSummary修复与s0_024INT8FullAP重启.md`
- `42_6_28_交接文档_阶段三_s0_024NativeINT8FullAP启动与监控计划.md`

本轮核心进展: 继续监控 `s0_024` native INT8 full AP v2 run。v2 已越过 v1 的 `sample_000125` empty summary blocker, 当前仍在运行, 暂无 sample blocker 和 full report。同时完成一次 H800 只读 FP16 AP checkpoint inventory, 证明当前可见 checkpoint/results/AP raw 根目录只包含 5 个已测 FP16 AP label 的候选, remaining 55 label 当前没有可直接评估的 checkpoint 候选。

## 0. 当前权威覆盖状态

覆盖状态仍未变化:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 5/60 measured |
| native INT8 | 60/60 measured | 60/60 measured | 0/60 measured |

合计:

```text
latency measured = 120/120
energy measured = 120/120
AP measured = 5/120
jobs_requiring_action = 115
```

本轮没有新增 measured AP row。`native_int8_original60_ap_rows_v1.jsonl` 仍不得生成/导入。

latency 口径继续固定:

```text
latency_ms = H800 + TVM backbone/subnet compiled module end-to-end latency
full_network_claim = false
```

## 1. s0_024 native INT8 full AP v2 运行状态

v2 raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix/
```

启动信息:

```text
runner_pid = 3810190
python_pid = 3810196
COMMAND_START = 2026-06-28T08:19:11+08:00
restart_reason = empty_numpy_summary_fix
num_samples = 1789
full_ap_min_samples = 1789
ap_row_min_samples = 1789
gpu_id = 0
```

最新监控:

```text
time = 2026-06-28T08:35:16+08:00
elapsed = 00:16:05
process_state = running
python_state = Sl
gpu0_memory_used_MB = 8677
sample_blocker_count = 0
max_worker_tmp = 000199
full_ap_eval_report = absent
```

判断:

```text
v2 已越过 v1 的 sample_000125 blocker。
当前没有 sample_*_blocker.json。
full report 尚未写出, 因此还不能判断 AP row gate。
```

注意:

```text
runner_stderr.txt 中的 git diff usage warning 是 run_full_ap.sh 启动时打印 diff 的噪声。
Python eval 进程已继续运行; 该 warning 不代表 full eval 失败。
```

## 2. FP16 AP checkpoint inventory

新增审阅产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_ap_checkpoint_inventory_20260628_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_ap_checkpoint_inventory_20260628_latest.md
```

搜索范围:

```text
${V2X_HOME}/heal_research/checkpoints
${V2X_DATA_ROOT}/heal_research/checkpoints
${V2X_ROOT}/results
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60
```

inventory summary:

| item | count |
|---|---:|
| original60 labels | 60 |
| FP16 AP measured labels | 5 |
| labels with checkpoint/raw candidates | 5 |
| unmeasured labels with candidates | 0 |
| labels without candidates | 55 |

labels with candidates:

```text
s0_024
s0_040
s0_056
s1_048
s2_160
```

结论:

```text
当前 H800 可见 checkpoint/results/AP raw 根目录只包含 5 个已测 FP16 AP label 的候选。
remaining 55 FP16 AP label 当前没有可直接评估的 checkpoint 候选。
```

这不是停止理由。下一步需要:

```text
1. 扩大 checkpoint recovery 范围: 其他机器、备份盘、训练产物归档、manifest 历史路径。
2. 若能恢复 checkpoint, 立即运行 true FP16 AP eval。
3. 若当前环境确实无法恢复, 写 per-label checkpoint_missing blocker, 但继续推进 native INT8 AP 和其他可解 cell。
```

## 3. 下一步执行顺序

优先级 1: 继续监控 `s0_024` native INT8 full AP v2。

通过条件:

```text
processed_samples >= 1789
ap_row_allowed = true
AP30/AP50/AP70 finite
pred_nonempty_count > 0
output_dequant_summary shows tensor_quant_params_v2 for pyramid_level0/1/2
full_network_claim = false
```

如果通过:

```text
生成 s0_024 native INT8 AP measured row。
刷新 rows/native_int8_original60_ap_rows_v1.jsonl。
刷新 rows/ap_original60_quant_rows_v1.jsonl。
刷新 completion review / gap report / three metric summary。
复制流程到 s0_040、s1_048。
```

如果失败:

```text
保存 full raw artifact。
读取 sample blocker 或 full report blocker。
做最小复现, 优先指定失败 sample_index。
修复后重跑 smoke 和 full eval。
不得因为单点失败停止其他 labels。
```

优先级 2: FP16 AP remaining 55 checkpoint recovery。

当前本机/H800 可见路径结论已经是:

```text
unmeasured_labels_with_candidates = []
labels_without_candidates = 55
```

下一步必须向外部 checkpoint source 或训练恢复路径推进, 不能伪造 measured AP。

## 4. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 AP 收口, 单 agent 执行, 不启动 agent team。当前 latency/energy 已是 FP16 60/60 与 native INT8 60/60 measured, latency 对外统一 latency_ms, scope 固定为 H800+TVM backbone/subnet compiled module end-to-end, full_network_claim=false。当前 AP 是 FP16 5/60、native INT8 0/60。s0_024 native INT8 full AP v2 正在 H800 运行: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, pid=3810190, 最新监控到 bridge_call_000199, sample_blocker_count=0, full_report_absent。继续监控到 full_ap_eval_report.json 产生; 若通过 gate, 生成 native INT8 s0_024 AP measured row 并刷新 AP 总表。FP16 AP checkpoint inventory 已落盘, 当前 H800 可见路径只有 5 个已测 label 有候选, remaining 55 无候选; 下一步需要恢复 checkpoint 或写 per-label checkpoint_missing blocker, 但不得停止 native INT8 AP 推进。
```

