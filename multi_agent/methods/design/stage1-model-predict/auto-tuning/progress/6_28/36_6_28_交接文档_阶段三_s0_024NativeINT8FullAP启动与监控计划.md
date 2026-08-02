# 42_6_28_交接文档_阶段三_s0_024NativeINT8FullAP启动与监控计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `41_6_28_交接文档_阶段三_INT8BackboneLatency口径确认与FP16INT8实测补点收口.md`
- `40_6_28_交接文档_阶段三_CalibratedINT8APSmoke非空与RowGate修复.md`

本轮核心进展: 已按单 agent 路线在 H800 上启动 `s0_024` native INT8 full 1789-sample AP eval。该任务使用已通过 1-sample calibrated smoke 的 `pyramid_level0/1/2` calibration v2, 并设置 full AP row gate 为 `1789` samples。当前任务仍在运行, 尚未产生 measured AP row。

## 0. 当前权威状态

当前 completion review 仍为:

```text
latency measured = 120/120
energy measured = 120/120
AP measured = 5/120
jobs_requiring_action = 115
```

按 precision:

| precision | latency | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 5/60 measured |
| native INT8 | 60/60 measured | 60/60 measured | 0/60 measured |

重要边界:

```text
latency_ms = H800 + TVM backbone/subnet compiled module end-to-end latency
full_network_claim = false
```

不得把当前 running full eval、历史 smoke、blocker 或 FP16 AP 写成 native INT8 measured AP。

## 1. 已启动的 H800 full AP eval

任务:

```text
label = s0_024
precision = native_int8
num_samples = 1789
full_ap_min_samples = 1789
ap_row_min_samples = 1789
gpu_id = 0
status = running at 2026-06-28T08:06:40+08:00
pid = 3654045
python_pid = 3654049
```

远端 raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v1/
```

route:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/
```

calibration:

```text
reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json
```

checkpoint:

```text
${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26
```

已落盘启动证据:

```text
runner_command.json
launcher_state.json
runner.pid
run_full_ap.sh
runner_stdout.txt
runner_stderr.txt
bridge_call_000/activation_float32.npy
```

启动 command 已写入 `runner_command.json`; shell runner 已写入 `run_full_ap.sh`。文档和 artifact 中不得写入 H800 密码明文。

## 2. 初始监控结果

第一次监控:

```text
time = 2026-06-28T08:05:27+08:00
pid = 3654045
process_state = running
gpu0_memory_used_MB = 1821
python_child = 3654049
```

第二次监控:

```text
time = 2026-06-28T08:06:40+08:00
pid = 3654045
process_state = running
gpu0_memory_used_MB = 3905
python_child = 3654049
new_artifact = bridge_call_000/activation_float32.npy
```

第三次监控:

```text
time = 2026-06-28T08:09:53+08:00
pid = 3654045
elapsed = 00:04:57
process_state = running
gpu0_memory_used_MB = 5417
python_child = 3654049
full_ap_eval_report = not_yet_written
stderr_traceback = none
```

stdout tail:

```text
COMMAND_START 2026-06-28T08:04:57+08:00
gpu_id=0 raw=.../ap_full_calibrated_pyramid_level2_v1 route=.../s0_024 calibration=.../tensor_quant_params_calibration_v2_to_pyramid_level2.json
```

stderr tail 只有 `timm` 和 `pkg_resources` warning, 未见 immediate failure traceback。

## 3. Full AP row gate

只有满足以下条件, 才允许生成或导入 native INT8 AP measured row:

```text
processed_samples >= 1789
failed_samples == 0 or failed_samples has documented acceptable policy
pred_nonempty_count > 0
AP30/AP50/AP70 finite
smoke_gate_passed = true
ap_row_allowed = true
ap_row_min_samples = 1789
output_dequant_summary shows pyramid_level0/1/2 use tensor_quant_params_v2
full_network_claim = false
```

通过后写入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_gap_report_latest.md
exports/original60_quant_three_metric_summary_latest.md
```

未通过时不得写 measured row。

## 4. 下一步监控命令

使用密钥/环境变量方式提供 SSH 密码, 不要把密码写入文档或命令历史。

```bash
export '\\061\\062\\063\\064\\065\\066\\067\\070')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
RAW="${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v1"
echo DATE=$(date -Iseconds)
cat "$RAW/runner.pid"
ps -p "$(cat "$RAW/runner.pid")" -o pid,stat,etime,cmd || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -8
find "$RAW" -maxdepth 2 -type f -printf "%P %s\n" | sort | head -120
tail -80 "$RAW/runner_stdout.txt" || true
tail -120 "$RAW/runner_stderr.txt" || true
if [ -f "$RAW/full_ap_eval_report.json" ]; then cat "$RAW/full_ap_eval_report.json"; fi
'
unset password-based SSH (disabled; use an SSH key)
```

## 5. 完成或失败后的处理

### 5.1 如果 full eval 成功

执行:

1. 拉取或审阅 `full_ap_eval_report.json`, `real_activation_bridge_report.json`, `output_dequant_summary.json`, `worker_response_summary.json`, `postprocess_summary.json`。
2. 校验 `processed_samples=1789`, `ap_row_allowed=true`, `AP70` 有限。
3. 生成 `s0_024` native INT8 AP measured row。
4. 刷新 native INT8 AP rows 和 canonical AP 总表。
5. 将同一流程复制到 `s0_040` 和 `s1_048`, 再扩展到 remaining 57 labels。

### 5.2 如果 full eval 失败

不得停止整批任务。执行:

1. 保存 `runner_stdout.txt`, `runner_stderr.txt`, `runner_command.json`, `launcher_state.json`, worker request/response, postprocess summary。
2. 判断 failure type: `full_eval_failed`, `smoke_empty`, `metric_gate_failed`, `route_runtime_failed`, `calibration_missing`, `postprocess_failed`。
3. 做最小复现: 同 route + 同 calibration + `num_samples=1`。
4. 若 1-sample 仍通过, 定位 full eval 独有问题: sample index、memory、dataset item、postprocess accumulation 或 eval aggregation。
5. 修复后重跑 1-sample smoke, 再重跑 1789-sample full eval。
6. 只有形成可审查且当前环境无法解决的 blocker, 才写 per-label blocker; 其他 labels 继续推进。

## 6. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 AP 收口, 单 agent 执行, 不启动 agent team。当前 latency/energy 已是 FP16 60/60 与 native INT8 60/60 measured, latency 对外统一 latency_ms, scope 固定为 H800+TVM backbone/subnet compiled module end-to-end, full_network_claim=false。当前 AP 是 FP16 5/60、native INT8 0/60。先监控 H800 上 s0_024 native INT8 full 1789-sample AP eval: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v1, pid=3654045。若 full eval 通过, 校验 ap_row_allowed=true、processed_samples>=1789、AP30/AP50/AP70 有限、output_dequant_summary 使用 tensor_quant_params_v2, 然后生成 native INT8 s0_024 AP measured row 并刷新 AP 总表和 completion review。若失败, 保存完整 raw artifact, 做 1-sample 最小复现和根因定位, 修复后补跑; 不得因单点失败停止其他 labels。
```
