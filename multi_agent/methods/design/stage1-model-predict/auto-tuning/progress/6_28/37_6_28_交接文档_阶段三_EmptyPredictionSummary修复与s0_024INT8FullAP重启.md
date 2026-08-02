# 43_6_28_交接文档_阶段三_EmptyPredictionSummary修复与s0_024INT8FullAP重启

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `42_6_28_交接文档_阶段三_s0_024NativeINT8FullAP启动与监控计划.md`
- `41_6_28_交接文档_阶段三_INT8BackboneLatency口径确认与FP16INT8实测补点收口.md`

本轮核心进展: `s0_024` native INT8 full AP v1 run 在 sample 125 暴露出一个 bridge 代码健壮性问题: 空预测框 tensor 在 summary 阶段触发 `np.min(empty)`。该问题不是 TVM native INT8 worker 失败, worker request/response 已成功。已用 TDD 修复 `summarize_numpy_array` 对 zero-size array 的处理, 同步到 H800, 终止旧 v1 run, 并启动 v2 full AP run。v2 已越过旧失败点, 当前仍在运行。

## 0. 当前权威覆盖状态

当前 measured 覆盖仍未变化:

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

本轮没有新增 native INT8 AP measured row。`native_int8_original60_ap_rows_v1.jsonl` 仍不得生成/导入, 直到 full 1789-sample gate 通过。

## 1. v1 run 的 blocker

v1 raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v1/
```

v1 旧 PID:

```text
runner_pid = 3654045
python_pid = 3654049
```

blocker:

```text
sample_000125_blocker.json
failure_reason = ValueError:zero-size array to reduction operation minimum which has no identity
```

根因栈:

```text
run_bridge -> tensor_summary(pred_box_tensor) -> summarize_numpy_array -> np.min(empty)
```

关键判断:

```text
native_int8_worker_response.status = success
outputs = pyramid_level0/1/2 uint8 tensors
worker_stderr.txt = empty
```

因此这是 AP bridge summary 对空预测不健壮, 不是 native INT8 route build/run 失败, 也不是 TVM worker 输出失败。

v1 已保留 interruption artifact:

```text
interruption_state.json
status = terminated_for_code_fix
reason = empty_numpy_summary_fix_requires_process_restart
preserved_blocker = sample_000125_blocker.json
terminated_at = 2026-06-28T08:18:27+08:00
```

## 2. 代码修复

修改:

```text
scripts/stage2_h800_native_int8_real_activation_bridge.py
framework/tests/test_stage2_native_int8_route.py
```

修复行为:

```text
summarize_numpy_array(...) 现在记录 size。
非空数组保持 min/max/mean/std 数值。
空数组返回 min/max/mean/std = None, 不再抛 ValueError。
```

新增回归测试:

```text
Stage2NativeInt8RouteTest.test_real_activation_bridge_summarizes_empty_numpy_array
```

本地验证:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 61 tests in 6.010s
OK

python -m py_compile scripts/stage2_h800_native_int8_real_activation_bridge.py framework/tests/test_stage2_native_int8_route.py
exit 0
```

远端 H800 验证:

```text
PYTHONPATH=${V2X_ROOT} ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m unittest framework.tests.test_stage2_native_int8_route.Stage2NativeInt8RouteTest.test_real_activation_bridge_summarizes_empty_numpy_array
Ran 1 test in 0.146s
OK
```

## 3. v2 full AP run

v2 raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix/
```

v2 启动信息:

```text
label = s0_024
precision = native_int8
num_samples = 1789
full_ap_min_samples = 1789
ap_row_min_samples = 1789
gpu_id = 0
runner_pid = 3810190
python_pid = 3810196
COMMAND_START = 2026-06-28T08:19:11+08:00
restart_reason = empty_numpy_summary_fix
```

v2 监控状态:

```text
time = 2026-06-28T08:30:54+08:00
elapsed = 00:11:29
process_state = running
gpu0_memory_used_MB = 6997
max_observed_worker_tmp = worker_tmp/bridge_call_000147
sample_blockers = none
full_ap_eval_report = not_yet_written
```

重要说明:

```text
v2 已越过 v1 的 sample_000125_blocker 点。
截至 08:30:54 未出现 sample_*_blocker.json。
runner_stderr.txt 中的 git diff usage warning 是 runner 启动时打印 diff 的噪声, 不影响 Python eval 进程; 实际 Python 进程已继续运行。
```

## 4. 下一步监控与 gate

继续监控 v2:

```bash
export '\\061\\062\\063\\064\\065\\066\\067\\070')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
RAW="${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix"
echo DATE=$(date -Iseconds)
cat "$RAW/runner.pid"
ps -p "$(cat "$RAW/runner.pid")" -o pid,stat,etime,cmd || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -8
find "$RAW" -maxdepth 1 -name "sample_*_blocker.json" -printf "%f %s\n" | sort | head
find "$RAW" -maxdepth 4 -type f -printf "%P %s\n" | sort | tail -120
if [ -f "$RAW/full_ap_eval_report.json" ]; then cat "$RAW/full_ap_eval_report.json"; fi
'
unset password-based SSH (disabled; use an SSH key)
```

只有满足以下条件才允许写 native INT8 AP measured row:

```text
processed_samples >= 1789
ap_row_allowed = true
ap_row_min_samples = 1789
AP30/AP50/AP70 finite
pred_nonempty_count > 0
output_dequant_summary shows tensor_quant_params_v2 for pyramid_level0/1/2
full_network_claim = false
```

如果 v2 成功:

```text
1. 生成 s0_024 native INT8 AP measured row。
2. 刷新 rows/native_int8_original60_ap_rows_v1.jsonl。
3. 刷新 rows/ap_original60_quant_rows_v1.jsonl。
4. 刷新 completion review / gap report / three metric summary。
5. 推广到 s0_040、s1_048, 再扩展到 remaining labels。
```

如果 v2 失败:

```text
1. 保留 full raw artifact, 不写 measured row。
2. 读取最新 sample blocker 或 full report blocker。
3. 区分 empty prediction 合法样本、postprocess bug、dataset sample bug、TVM output issue、metric aggregation issue。
4. 做 1-sample 或指定 sample_index 最小复现。
5. 修复后再重跑 smoke 和 full eval。
```

## 5. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 AP 收口, 单 agent 执行, 不启动 agent team。当前 latency/energy 已是 FP16 60/60 与 native INT8 60/60 measured, latency 对外统一 latency_ms, scope 固定为 H800+TVM backbone/subnet compiled module end-to-end, full_network_claim=false。当前 AP 是 FP16 5/60、native INT8 0/60。已修复 s0_024 native INT8 full AP v1 在 sample_000125 的 empty pred_box_tensor summary bug: summarize_numpy_array 对空数组返回 size=0 且 min/max/mean/std=None。本地 61 个相关测试通过, 远端 empty summary 回归测试通过。现在监控 v2 run: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, pid=3810190。若 full eval 通过, 校验 processed_samples>=1789、ap_row_allowed=true、AP30/AP50/AP70 finite、output_dequant_summary 使用 tensor_quant_params_v2, 然后生成 native INT8 s0_024 AP measured row 并刷新 AP 总表和 completion review。若失败, 保存完整 raw artifact, 做最小复现和根因定位, 修复后补跑; 不得因单点失败停止其他 labels。
```

