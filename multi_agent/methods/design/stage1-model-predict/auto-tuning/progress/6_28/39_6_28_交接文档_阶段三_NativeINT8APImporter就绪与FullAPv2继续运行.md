# 45_6_28_交接文档_阶段三_NativeINT8APImporter就绪与FullAPv2继续运行

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `44_6_28_交接文档_阶段三_INT8FullAPv2运行中与FP16CheckpointInventory.md`
- `43_6_28_交接文档_阶段三_EmptyPredictionSummary修复与s0_024INT8FullAP重启.md`

本轮核心进展: `s0_024` native INT8 full AP v2 仍在 H800 上运行, 已继续推进到 `bridge_call_000271`, 当前无 sample blocker, 但尚未写出 full report。等待期间已新增 native INT8 full AP row importer, 用于在 full report 通过 gate 后生成合规 `native_int8_original60_ap_rows_v1.jsonl` row。该 importer 已有通过路径和 smoke 拒绝路径测试。

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

本轮没有导入 native INT8 AP measured row。必须等 full report 通过 gate 后再入表。

## 1. s0_024 native INT8 full AP v2 当前状态

v2 raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix/
```

启动信息:

```text
runner_pid = 3810190
python_pid = 3810196
COMMAND_START = 2026-06-28T08:19:11+08:00
num_samples = 1789
full_ap_min_samples = 1789
ap_row_min_samples = 1789
gpu_id = 0
```

最新监控:

```text
time = 2026-06-28T08:41:22+08:00
elapsed = 00:22:11
process_state = running
gpu0_memory_used_MB = 11425
sample_blocker_count = 0
max_worker_tmp = 000271
full_ap_eval_report = absent
```

判断:

```text
v2 已越过 v1 sample_000125 blocker。
当前没有 sample_*_blocker.json。
full report 尚未写出, 不能导入 AP row。
```

## 2. Native INT8 AP row importer

新增:

```text
scripts/stage2_import_native_int8_full_ap_row.py
```

新增/更新测试:

```text
framework/tests/test_stage2_original60_quant_completion.py
```

覆盖行为:

```text
1. 将通过 gate 的 full_ap_eval_report.json 转成 native_int8 AP70 measured row。
2. 拒绝 1-sample smoke 或 ap_row_allowed=false 的 report。
3. 要求 output_dequant_summary 中 pyramid_level0/1/2 均使用 tensor_quant_params_v2。
4. 输出 row 保持 full_network_claim=false。
5. 输出 row 禁止 TRT 字样。
```

importer gate:

```text
processed_samples >= 1789
ap_row_allowed = true
ap_row_min_samples >= 1789
pred_nonempty_count > 0
AP30/AP50/AP70 numeric
output_dequant_summary covers pyramid_level0/1/2
output dequant scheme = tensor_quant_params_v2 for all graph outputs
```

测试与验证:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_original60_quant_completion framework.tests.test_stage2_native_int8_route
Ran 63 tests in 7.619s
OK

python -m py_compile scripts/stage2_import_native_int8_full_ap_row.py framework/tests/test_stage2_original60_quant_completion.py
exit 0
```

## 3. v2 成功后的入表命令

仅在 `full_ap_eval_report.json` 存在且 gate 通过后运行:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_import_native_int8_full_ap_row.py \
  --label s0_024 \
  --width 24,128,256 \
  --raw-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix \
  --route-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024 \
  --tensor-quant-params-path ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json \
  --rows-out ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl \
  --run-id 20260628_native_int8_s0_024_full_ap_v2 \
  --created-at 2026-06-28T00:00:00Z
```

随后刷新:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

注意: 如果 gate 不通过, importer 会写:

```text
native_int8_ap_row_import_blocker.json
```

并且不会写 measured row。

## 4. 下一步监控命令

```bash
export '\\061\\062\\063\\064\\065\\066\\067\\070')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
RAW="${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix"
echo DATE=$(date -Iseconds)
cat "$RAW/runner.pid"
ps -p "$(cat "$RAW/runner.pid")" -o pid,stat,etime,cmd || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | head -8
echo BLOCKER_COUNT=$(find "$RAW" -maxdepth 1 -name "sample_*_blocker.json" | wc -l)
echo MAX_WORKER_TMP=$(find "$RAW/worker_tmp" -maxdepth 1 -type d -name "bridge_call_*" 2>/dev/null | sed "s#.*/bridge_call_##" | sort | tail -1)
if [ -f "$RAW/full_ap_eval_report.json" ]; then cat "$RAW/full_ap_eval_report.json"; fi
'
unset password-based SSH (disabled; use an SSH key)
```

## 5. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 AP 收口, 单 agent 执行, 不启动 agent team。当前 latency/energy 已是 FP16 60/60 与 native INT8 60/60 measured, latency 对外统一 latency_ms, scope 固定为 H800+TVM backbone/subnet compiled module end-to-end, full_network_claim=false。当前 AP 是 FP16 5/60、native INT8 0/60。s0_024 native INT8 full AP v2 正在 H800 运行: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, pid=3810190, 最新监控到 bridge_call_000271, sample_blocker_count=0, full_report_absent。native INT8 AP row importer 已实现: scripts/stage2_import_native_int8_full_ap_row.py, 相关 63 个测试通过。继续监控到 full_ap_eval_report.json 产生; 若 gate 通过, 运行 importer 生成 native INT8 s0_024 AP measured row, 再刷新 AP 总表和 completion review。若失败, 保存完整 raw artifact, 做最小复现和根因定位, 修复后补跑。
```

