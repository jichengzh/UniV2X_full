# 47_6_28_交接文档_阶段三_FullAPv2进度532与后续Route盘点

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `46_6_28_交接文档_阶段三_FullAPv2长跑监控与Importer远端就绪.md`
- `45_6_28_交接文档_阶段三_NativeINT8APImporter就绪与FullAPv2继续运行.md`

本轮核心进展: 继续监控 `s0_024` native INT8 full AP v2。该 run 仍在 H800 GPU0 上运行, Python 子进程确认活跃, 已推进到 `bridge_call_000532`, 当前无 sample blocker, full report 尚未写出。同时完成 `s0_040/s1_048` 的只读 route 准备度盘点: 两者有 checkpoint 和 ONNX, 但尚无 checkpoint-consistent centered-conv native INT8 route 目录。

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

仍不得生成或导入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

直到 `full_ap_eval_report.json` 通过 full 1789-sample gate。

## 1. s0_024 native INT8 full AP v2 最新状态

raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix/
```

进程:

```text
runner_pid = 3810190
python_pid = 3810196
COMMAND_START = 2026-06-28T08:19:11+08:00
gpu_id = 0
```

最新精确监控:

```text
time = 2026-06-28T09:02:43+08:00
elapsed = 00:43:31
runner_state = S
python_state = Sl
python_cpu = 365%
gpu0_pmon_pid = 3810196
sample_blocker_count = 0
max_worker_tmp = 000532
full_ap_eval_report = absent
```

判断:

```text
v2 仍在正常运行, 不是僵尸进程。
已越过 v1 的 sample_000125 blocker。
当前没有 sample_*_blocker.json。
full report 尚未写出, 因此不能导入 AP row。
```

速度估算:

```text
43.5 min -> bridge_call_000532
full target = 1789 samples
粗略估计 total runtime 约 2.4-2.6 hours, 还需继续监控。
```

## 2. s0_040 / s1_048 route 准备度盘点

只读盘点结论:

```text
s0_040 checkpoint exists:
${V2X_DATA_ROOT}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_040_2026_06_26

s1_048 checkpoint exists:
${V2X_DATA_ROOT}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s1_048_2026_06_26

s0_040 ONNX exists:
${V2X_DATA_ROOT}/s2_tvm/models/s0_040_backbone.onnx

s1_048 ONNX exists:
${V2X_DATA_ROOT}/s2_tvm/models/s1_048_backbone.onnx
```

但当前 route 目录盘点显示:

```text
checkpoint-consistent centered-conv native INT8 route currently exists only for s0_024.
s0_040 and s1_048 do not yet have matching route dirs under raw/int8_native_route.
```

因此, `s0_040/s1_048` 不能直接复用当前 AP bridge full eval 命令。下一步要么生成 checkpoint-consistent native INT8 route, 要么先把 route builder 参数化到这两个 label。

## 3. Importer 状态

native INT8 AP importer 已就绪:

```text
scripts/stage2_import_native_int8_full_ap_row.py
```

已在本地和 H800 远端验证:

```text
gated full report -> native_int8 AP measured row: pass
1-sample smoke / ap_row_allowed=false -> rejected with blocker: pass
```

只有 v2 full report 满足以下条件时才运行 importer:

```text
processed_samples >= 1789
ap_row_allowed = true
ap_row_min_samples >= 1789
pred_nonempty_count > 0
AP30/AP50/AP70 numeric
output_dequant_summary covers pyramid_level0/1/2
output dequant scheme = tensor_quant_params_v2 for all graph outputs
full_network_claim = false
```

## 4. 下一步

### 4.1 继续监控 s0_024 v2

```bash
export '\\061\\062\\063\\064\\065\\066\\067\\070')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
RAW="${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix"
PY="${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python"
echo DATE=$(date -Iseconds)
ps -p "$(cat "$RAW/runner.pid")" -o pid,stat,etime,cmd || true
ps --ppid "$(cat "$RAW/runner.pid")" -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd || true
nvidia-smi pmon -c 1 | sed -n "1,12p"
echo BLOCKER_COUNT=$(find "$RAW" -maxdepth 1 -name "sample_*_blocker.json" | wc -l)
echo MAX_WORKER_TMP=$(find "$RAW/worker_tmp" -maxdepth 1 -type d -name "bridge_call_*" 2>/dev/null | sed "s#.*/bridge_call_##" | sort | tail -1)
if [ -f "$RAW/full_ap_eval_report.json" ]; then "$PY" -m json.tool "$RAW/full_ap_eval_report.json"; fi
'
unset password-based SSH (disabled; use an SSH key)
```

### 4.2 如果 full report 通过 gate

在 H800 上运行 importer:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python \
  scripts/stage2_import_native_int8_full_ap_row.py \
  --label s0_024 \
  --width 24,128,256 \
  --raw-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix \
  --route-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024 \
  --tensor-quant-params-path ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json \
  --rows-out ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl \
  --run-id 20260628_native_int8_s0_024_full_ap_v2 \
  --created-at 2026-06-28T00:00:00Z
```

然后刷新总表与 review:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

### 4.3 如果 full report 失败或 importer 拒绝

不得停止整批任务。执行:

```text
1. 保存 full raw artifact。
2. 读取 sample blocker / full report blocker / native_int8_ap_row_import_blocker.json。
3. 按 failure type 做最小复现。
4. 修复后重跑 smoke 和 full eval。
```

## 5. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 AP 收口, 单 agent 执行, 不启动 agent team。当前 latency/energy 已是 FP16 60/60 与 native INT8 60/60 measured, latency 对外统一 latency_ms, scope 固定为 H800+TVM backbone/subnet compiled module end-to-end, full_network_claim=false。当前 AP 是 FP16 5/60、native INT8 0/60。s0_024 native INT8 full AP v2 正在 H800 GPU0 运行: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, runner_pid=3810190, python_pid=3810196, 最新监控到 bridge_call_000532, sample_blocker_count=0, full_report_absent。native INT8 AP importer 已同步到 H800 并通过远端测试。s0_040/s1_048 有 checkpoint 和 ONNX, 但尚无 checkpoint-consistent centered-conv native INT8 route 目录, 不能直接跑 full AP bridge。继续监控 s0_024 到 full_ap_eval_report.json 产生; 若 gate 通过, 运行 importer 生成 native INT8 s0_024 AP measured row, 再刷新 AP 总表和 completion review。若失败, 保存完整 raw artifact, 做最小复现和根因定位, 修复后补跑。
```

