# 52_6_28_交接文档_阶段三_s0040APSmoke阻断与ScaleAwareRoute下一步

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `51_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8全量AP收口计划.md`
- `50_6_28_交接文档_阶段三_Coverage默认刷新修复与FullAPv2进度812.md`
- `40_6_28_交接文档_阶段三_CalibratedINT8APSmoke非空与RowGate修复.md`

本轮核心进展:

```text
1. 继续监控 s0_024 native INT8 full AP v2: 仍在运行, 无 sample blocker, full report 尚未写出。
2. 将 s0_040 推进到 checkpoint-consistent route + tensor_quant_params_v2 + output-dequant smoke artifact。
3. s0_040 5-sample AP smoke 被 empty_predictions_all_samples 阻断, 未进入 full AP。
4. numeric sanity 证明 PyTorch reference head/postprocess 有效, TVM native INT8 输出与 reference 不对齐。
5. prefix trace 证明没有 heavy saturation, 但高层特征动态范围向 128 附近塌缩, 下一步应做内部 scale-aware/dynamic requant route, 不应盲跑 s0_040 full AP。
```

## 0. 当前总表覆盖状态

权威覆盖仍是:

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured

native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured

completion review:
  latency = 120/120 measured
  energy = 120/120 measured
  AP = 5/120 measured
  jobs_requiring_action = 115
```

口径必须保持:

```text
latency_ms = H800 TVM compiled backbone/subnet module end-to-end latency
input = spatial_features
output = multiscale backbone/subnet outputs
full_network_claim = false
```

仍不能声明:

```text
native INT8 AP measured
完整 perception pipeline latency
真实 RSU 物理设备绝对 latency
```

## 1. s0_024 full AP v2 状态

raw:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix
```

最新监控:

```text
time = 2026-06-28T10:01:50+08:00
runner_pid = 3810190
python_pid = 3810196
elapsed = 01:42:38
max_worker_tmp = bridge_call_001283
sample_blocker_count = 0
full_ap_eval_report.json = absent
```

判断:

```text
进程仍在推进。
未发现 sample blocker。
full report 尚未写出, 因此不得导入 native_int8_original60_ap_rows_v1.jsonl。
```

## 2. s0_040 本轮实测进展

route root:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040
```

### 2.1 Reference range capture

第一次失败:

```text
raw_dir = reference_range_capture_h800_pyramid_level2_v1
failure_reason = CUDA error: invalid device ordinal
root_cause = CUDA_VISIBLE_DEVICES=1 后进程内只剩 device 0, 但命令仍 --gpu-id 1
```

修复:

```text
重跑时设置 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, 保持 --gpu-id 1。
```

成功目录:

```text
reference_range_capture_h800_pyramid_level2_v2_cuda_visible_all/
```

关键结果:

```text
schema = native_int8_reference_range_capture_summary_v1
status = ready_for_scale_aware_simulator
range_count = 115
full_network_claim = false
ap_measured = false
calibration_path = tensor_quant_params_calibration_v2_to_pyramid_level2.json
```

### 2.2 Calibrated AP smoke

第一次失败:

```text
raw_dir = ap_smoke_calibrated_pyramid_level2_v1
failure_type = tvm_worker_failure
failure_reason = worker looked for s0_024_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so under s0_040 route dir
root_cause = bridge 默认 artifact name 仍指向 s0_024, s0_040 需要显式传 artifact/inventory/runtime weights
```

修复:

```text
显式传入:
--artifact-path s0_040_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
--inventory-path tvm_operator_inventory.json
--runtime-weight-archive-path runtime_weights_int8.npz
```

1-sample smoke:

```text
raw_dir = ap_smoke_calibrated_pyramid_level2_v2_explicit_artifact
worker = success
output_dequant = pyramid_level0/1/2 all tensor_quant_params_v2
head/postprocess = success
processed_samples = 1
pred_nonempty_count = 0
ap_row_allowed = false
ap_row_block_reason = empty_predictions_all_samples
```

5-sample smoke:

```text
raw_dir = ap_smoke5_calibrated_pyramid_level2_v1
processed_samples = 5
failed_samples = 0
pred_nonempty_count = 0
pred_total_count = 0
smoke_gate_passed = false
gate.reason = empty_predictions_all_samples
```

结论:

```text
s0_040 不是单帧偶然 empty。
worker/dequant/head/postprocess 已通, 但 TVM native INT8 multiscale feature 数值不足以通过 postprocess threshold。
```

## 3. s0_040 numeric sanity 结论

目录:

```text
numeric_sanity_calibrated_pyramid_level2_v1/
```

关键信息:

```text
numeric_sanity_only = true
processed_samples = 1
status = blocked
```

重要观察:

```text
numeric sanity 模式下, bridge 仍调用 TVM worker 并记录 TVM-vs-reference 对齐误差, 但返回 PyTorch reference multiscale outputs 给 head/postprocess。
该 reference path 同一帧 AP 输出:
  AP30 = 0.80
  AP50 = 0.80
  AP70 = 0.69
```

因此:

```text
checkpoint/head/postprocess 本身有效。
empty prediction 的直接原因是 TVM native INT8 route 输出与 PyTorch reference multiscale outputs 不对齐。
```

对齐摘要:

| tensor | corrcoef | rmse | mae | status |
|---|---:|---:|---:|---|
| pyramid_level0 | 0.4684436294 | 1.3131599698 | 0.7939837891 | blocked |
| pyramid_level1 | 0.1091737599 | 0.4445688924 | 0.1577737108 | blocked |
| pyramid_level2 | 0.3740354864 | 0.2365620197 | 0.0399594842 | blocked |

## 4. s0_040 prefix trace 结论

目录:

```text
native_prefix_trace_to_pyramid_level2_v1/
```

summary:

```text
schema = native_int8_prefix_trace_summary_v1
status = recorded
record_count = 115
first_heavily_saturated_output = null
```

代表性 trace:

```text
op_index=0 Conv output:
  min=0, max=246, mean=125.92

pyramid_level2:
  min=128, max=159, mean=128.04
```

解释:

```text
问题不是典型的 max_fraction >= 0.5 heavy saturation。
更像是固定 requant / 内部 scale propagation 导致高层特征动态范围逐层压缩到 128 附近。
这解释了 head cls logits 偏低和 postprocess 全空。
```

## 5. Readiness 报告已更新

更新文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_ap_route_readiness_20260628_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/native_int8_ap_route_readiness_20260628_latest.md
```

当前摘要:

```text
label_count = 60
full_onnx_route_manifest_labels = 60
checkpoint_consistent_centered_conv_route_labels = 2
tensor_quant_params_v2_labels = 2
ap_output_dequant_smoke_labels = 2
ap_output_dequant_smoke_passed_labels = [s0_024]
ap_output_dequant_smoke_blocked_labels = [s0_040]
labels_needing_ap_route_assets = 58
labels_needing_ap_calibration_or_smoke = 58
labels_needing_numeric_fix_before_full_ap = [s0_040]
```

## 6. 下一步计划

### 6.1 不要盲跑 s0_040 full AP

原因:

```text
s0_040 5-sample smoke 已全空。
numeric sanity 已证明 TVM output misalignment。
直接跑 1789-frame full AP 只会浪费 GPU 时间, 不会生成合规 measured row。
```

### 6.2 优先实现内部 scale-aware / dynamic requant route

目标:

```text
让 native INT8 route 在内部 Conv / Add / ReLU 之间使用 tensor_quant_params_v2 或 equivalent dynamic scale propagation, 而不是固定 /256 requant 后一路把高层特征压到 128 附近。
```

最小验收顺序:

```text
1. 在 s0_040 上复用现有 tensor_quant_params_calibration_v2_to_pyramid_level2.json。
2. 构建 scale-aware route 或 dynamic requant route。
3. 先跑 numeric_sanity_only:
   pyramid_level0/1/2 corrcoef 显著提升, status 不再 blocked。
4. 再跑 5-sample AP smoke:
   pred_nonempty_count > 0
   smoke_gate_passed = true
   ap_row_allowed = false, 因为 samples < 1789
5. 通过后再启动 s0_040 1789-frame full AP。
```

### 6.3 继续监控 s0_024 full AP v2

report 写出后审 gate:

```text
processed_samples >= 1789
ap_row_allowed = true
ap_row_min_samples >= 1789
pred_nonempty_count > 0
AP30/AP50/AP70 numeric
output_dequant_summary covers pyramid_level0/1/2
full_network_claim = false
```

gate 通过后才允许 importer 写:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

### 6.4 FP16 AP 仍需并行补齐

当前 FP16 AP 仍是:

```text
5/60 measured
55/60 no_claim
```

继续 checkpoint recovery + `scripts/stage2_h800_true_fp16_ap_eval.py`, 每批后刷新 coverage/review。

## 7. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 energy/AP 全量收口, 单 agent 执行, 不启动 agent team。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; completion review latency=120/120, energy=120/120, AP=5/120, jobs_requiring_action=115。latency_ms 口径固定为 H800 TVM compiled backbone/subnet module end-to-end, input=spatial_features, output=multiscale backbone/subnet outputs, full_network_claim=false; 不能声明为完整 perception pipeline 或真实 RSU 物理设备绝对 latency。s0_024 native INT8 full AP v2 仍在 H800 GPU0 运行: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, runner_pid=3810190, python_pid=3810196, latest_monitor_time=2026-06-28T10:01:50+08:00, max_worker_tmp=bridge_call_001283, sample_blocker_count=0, full_report_absent。s0_040 已完成 checkpoint-consistent native INT8 route、reference range capture、tensor_quant_params_v2 calibration 和 output-dequant smoke artifact; route latency_ms=12.291168, energy_J=2.7866748651142528; 但 5-sample AP smoke blocked: processed_samples=5, pred_nonempty_count=0, reason=empty_predictions_all_samples。numeric_sanity_only 证明 PyTorch reference path 同帧 AP70≈0.69, 但 TVM output misaligned: pyramid_level1 corrcoef=0.109, pyramid_level2 corrcoef=0.374; prefix trace 到 pyramid_level2 无 heavy saturation 但动态范围压缩到 128-159。下一步不要盲跑 s0_040 full AP, 先实现/构建内部 scale-aware 或 dynamic requant native INT8 route, 使用 tensor_quant_params_v2 修复 Conv/Add/ReLU 内部尺度传播; 先 numeric_sanity gate, 再 5-sample smoke gate, 通过后再 full AP。继续监控 s0_024 full AP report; report gate 通过后 importer 写 native_int8_original60_ap_rows_v1.jsonl 并刷新 coverage/review。FP16 AP 继续从 5/60 补到 60/60。遇到任何 build/eval/import/SSH/checkpoint/adapter/gate 问题, 不允许直接停止: 保存 raw artifact、command、stdout/stderr、runner pid、GPU id、failure reason, 做最小复现、根因判断、反思审查、脚本或队列修复、smoke 和失败 label 补跑; 只有证明当前环境或 checkpoint 缺失无法由执行 agent 解决时才写 per-label blocker 并继续其他 label。
```
