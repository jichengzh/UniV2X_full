# 49_6_28_交接文档_阶段三_INT8APRouteReadiness与FullAPv2进度710

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `48_6_28_交接文档_阶段三_INT8Backbone口径纠偏与FP16INT8_APEnergy收口计划.md`
- `47_6_28_交接文档_阶段三_FullAPv2进度532与后续Route盘点.md`

本轮核心进展:

```text
1. 继续监控 s0_024 native INT8 full AP v2, 当前仍在 H800 GPU0 活跃运行。
2. 生成 native INT8 AP route readiness 报告, 明确 60/60 latency/energy route manifest 不等于 60/60 AP-ready route。
3. 发现并纠正 canonical coverage 裸跑默认参数会回写 no_claim 的刷新风险; 已用显式权威 rows 重刷, 恢复 review 正确状态。
```

## 0. 当前权威覆盖状态

本轮重刷并复核后:

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

复核命令输出摘要:

```text
rows/latency_original60_quant_rows_v1.jsonl:
  fp16 measured = 60
  int8 measured = 60

rows/energy_original60_quant_rows_v1.jsonl:
  fp16 measured = 60
  int8 measured = 60

rows/ap_original60_quant_rows_v1.jsonl:
  fp16 measured = 5
  fp16 no_claim = 55
  int8 no_claim = 60

exports/fp16_int8_original60_completion_review_latest.json:
  axis_status_counts = {
    "latency": {"measured": 120},
    "energy": {"measured": 120},
    "ap": {"measured": 5, "no_claim": 115}
  }
```

仍不得生成或导入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

直到 full 1789-sample INT8 AP report 通过 importer gate。

## 1. s0_024 native INT8 full AP v2 最新状态

raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix
```

最新监控:

```text
time = 2026-06-28T09:16:09+08:00
runner_pid = 3810190
python_pid = 3810196
elapsed = 00:56:57
python_state = Rl
python_cpu = 350%
sample_blocker_count = 0
max_worker_tmp = bridge_call_000710
full_ap_eval_report.json = absent
```

判断:

```text
进程仍活跃, 不是僵尸。
已经越过 v1 的 sample_000125 empty prediction summary blocker。
当前没有 sample_*_blocker.json。
full report 尚未写出, 因此不能导入 AP row。
```

继续监控命令:

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
RAW="${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix"
PY="${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python"
echo DATE=$(date -Iseconds)
ps -p "$(cat "$RAW/runner.pid")" -o pid,stat,etime,cmd || true
ps --ppid "$(cat "$RAW/runner.pid")" -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd || true
echo BLOCKER_COUNT=$(find "$RAW" -maxdepth 1 -name "sample_*_blocker.json" | wc -l)
echo MAX_WORKER_TMP=$(find "$RAW/worker_tmp" -maxdepth 1 -type d -name "bridge_call_*" -printf "%f\n" 2>/dev/null | sort | tail -1)
echo REPORT_EXISTS=$([ -f "$RAW/full_ap_eval_report.json" ] && echo yes || echo no)
if [ -f "$RAW/full_ap_eval_report.json" ]; then "$PY" -m json.tool "$RAW/full_ap_eval_report.json"; fi
'
unset password-based SSH (disabled; use an SSH key)
```

## 2. Native INT8 AP route readiness 报告

新增产物:

```text
exports/native_int8_ap_route_readiness_20260628_latest.json
exports/native_int8_ap_route_readiness_20260628_latest.md
```

核心结论:

```text
label_count = 60
full_onnx_route_manifest_labels = 60
checkpoint_consistent_centered_conv_route_labels = 1
tensor_quant_params_v2_labels = 1
ap_output_dequant_smoke_labels = 1
full_ap_v2_running_gate_pending_labels = ["s0_024"]
route_missing_labels = []
labels_needing_ap_route_assets = 59
```

解释:

```text
original60 native INT8 latency/energy full-ONNX route manifests 已覆盖 60/60。
但 AP-ready 的 checkpoint-consistent centered-conv route、tensor_quant_params_v2 calibration、AP output-dequant smoke 资产目前只覆盖 s0_024。
因此不能把 60/60 latency/energy route manifest 误当成 60/60 native INT8 AP-ready route。
```

后续批量 native INT8 AP 的正确顺序:

```text
1. 等 s0_024 full AP v2 完成并通过 importer gate。
2. 将 s0_024 的 checkpoint-consistent centered-conv route + tensor_quant_params_v2 calibration + bridge 参数固化为模板。
3. 对剩余 59 个 label 先补 AP-ready route/calibration/smoke 资产。
4. 再逐 label 跑 full 1789-sample AP eval。
5. importer gate 通过后才写 rows/native_int8_original60_ap_rows_v1.jsonl。
```

## 3. 刷新脚本防踩坑

本轮发现:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
```

裸跑默认参数会按保守策略回写 canonical rows, 结果是 FP16/INT8 latency/energy/AP 全部变成 `no_claim`。原因是默认参数不会自动读取以下权威 rows:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl
rows/fp16_true_original60_energy_rows_v1.jsonl
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
rows/native_int8_original60_ap_rows_v1.jsonl
```

已用显式参数重刷恢复正确状态。后续每次刷新都必须使用 `48_...` 文档第 4 节的显式命令, 不要裸跑默认参数。

恢复后的 completion review:

```text
axis_status_counts = {
  "ap": {"measured": 5, "no_claim": 115},
  "energy": {"measured": 120},
  "latency": {"measured": 120}
}
jobs_requiring_action = 115
```

## 4. 如果 full report 产生

先检查 gate:

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

通过后在 H800 上运行 importer:

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

然后用显式 rows 参数刷新 canonical/review, 不能裸跑默认 coverage 命令。

## 5. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 energy/AP 收口, 单 agent 执行, 不启动 agent team。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP 0/60 measured; completion review jobs_requiring_action=115。latency_ms 口径固定为 H800 TVM compiled backbone/subnet module end-to-end, input=spatial_features, output=backbone/subnet multiscale outputs, full_network_claim=false, 不能声明为完整 perception pipeline 或真实 RSU 物理边缘设备绝对 latency。H800 上 s0_024 native INT8 full AP v2 正在运行: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, runner_pid=3810190, python_pid=3810196, 最新监控到 2026-06-28T09:16:09+08:00, bridge_call_000710, sample_blocker_count=0, full_report_absent。新增 readiness 报告: exports/native_int8_ap_route_readiness_20260628_latest.md/.json, 结论是 original60 60/60 有 latency/energy full-ONNX route manifest, 但只有 s0_024 有 AP-ready checkpoint-consistent centered-conv route + tensor_quant_params_v2 + output-dequant smoke, 剩余 59 个 label 需要先补 AP-ready route/calibration/smoke 资产。注意 canonical 刷新不能裸跑 scripts/stage2_generate_original60_quant_state_coverage.py, 必须显式传入 fp16_true_original60_* 和 native_int8_full_onnx_original60_* rows, 否则会回写 no_claim。下一步继续监控 s0_024 full AP v2; full_report 产生后先审 gate, 通过后 importer 写入 native_int8_original60_ap_rows_v1.jsonl, 再用显式 rows 参数刷新 canonical/review。随后把 s0_024 的 AP-ready route/calibration/bridge 模板推广到剩余 59 个 label, 同时继续 FP16 AP checkpoint recovery/full true-eval。遇到任何失败不能直接停止: 保存 raw artifact/stdout/stderr/runner command/GPU id/failure reason, 做最小复现和根因修复, smoke 后 full eval 补跑; 只有当前环境无法恢复时才写 per-label blocker。
```
