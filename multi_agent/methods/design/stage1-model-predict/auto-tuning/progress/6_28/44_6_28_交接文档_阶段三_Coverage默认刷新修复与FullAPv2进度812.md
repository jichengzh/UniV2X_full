# 50_6_28_交接文档_阶段三_Coverage默认刷新修复与FullAPv2进度812

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `49_6_28_交接文档_阶段三_INT8APRouteReadiness与FullAPv2进度710.md`
- `48_6_28_交接文档_阶段三_INT8Backbone口径纠偏与FP16INT8_APEnergy收口计划.md`

本轮核心进展:

```text
1. 修复 scripts/stage2_generate_original60_quant_state_coverage.py 默认刷新会误回写 no_claim 的问题。
2. 新增回归测试, 验证 output_root/rows 下已有权威 measured rows 时, 裸跑 coverage 也会自动导入。
3. 用真实 original60_quant_20260627 目录裸跑 coverage + completion queue, 复核 FP16/INT8 latency/energy/AP 覆盖保持正确。
4. 继续监控 H800 s0_024 native INT8 full AP v2, 当前推进到 bridge_call_000832, 无 blocker, full report 尚未写出。
```

## 0. 当前权威覆盖状态

真实目录裸跑 coverage 后复核:

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
```

completion review:

```text
axis_status_counts = {
  "latency": {"measured": 120},
  "energy": {"measured": 120},
  "ap": {"measured": 5, "no_claim": 115}
}
jobs_requiring_action = 115
precision_counts = {"fp16": 60, "int8": 60}
total_jobs = 120
```

## 1. Coverage 默认刷新修复

问题:

```text
之前裸跑:
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py

会因为没有显式传入 rows 参数而忽略 output_root/rows 下已有的 true-FP16/native-INT8 权威 rows, 从而把 canonical rows 回写成 no_claim。
```

修复:

```text
scripts/stage2_generate_original60_quant_state_coverage.py
```

新增默认路径聚合逻辑:

```text
当对应 CLI 参数为空时, 自动读取 output_root/rows 下已存在的权威 row 文件:

fp32_latency_original60_remapped_rows_v1.jsonl
fp32_latency_smoke_rows_v1.jsonl
fp16_true_original60_latency_rows_v1.jsonl
fp16_true_latency_smoke_rows_v1.jsonl
fp16_true_original60_energy_rows_v1.jsonl
fp16_true_ap_energy_smoke_rows_v1.jsonl
int8_latency_smoke_rows_v1.jsonl
int8_ap_energy_smoke_rows_v1.jsonl
native_int8_full_onnx_original60_latency_rows_v1.jsonl
native_int8_full_onnx_latency_rows_v1.jsonl
native_int8_full_onnx_original60_energy_rows_v1.jsonl
native_int8_full_onnx_energy_rows_v1.jsonl
fp16_true_original60_ap_rows_v1.jsonl
native_int8_original60_ap_rows_v1.jsonl
```

显式传参仍然优先, 不会被默认路径覆盖。

新增测试:

```text
framework/tests/test_stage2_lut_productization.py
  test_original60_quant_state_coverage_defaults_to_authoritative_rows_under_output_root
```

RED 结果:

```text
AssertionError: 'no_claim' != 'measured'
```

GREEN/验证:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_lut_productization.Stage2Original60QuantCoverageCliTest.test_original60_quant_state_coverage_defaults_to_authoritative_rows_under_output_root
Ran 1 test ... OK

PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_lut_productization.Stage2Original60QuantCoverageCliTest
Ran 10 tests ... OK

python -m py_compile scripts/stage2_generate_original60_quant_state_coverage.py framework/tests/test_stage2_lut_productization.py
OK
```

真实目录验证:

```text
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py

结果仍为 latency 120/120 measured, energy 120/120 measured, AP 5/120 measured, jobs_requiring_action=115。
```

## 2. s0_024 native INT8 full AP v2 最新状态

raw 目录:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix
```

最新监控:

```text
time = 2026-06-28T09:24:54+08:00
runner_pid = 3810190
python_pid = 3810196
sample_blocker_count = 0
max_worker_tmp = bridge_call_000832
full_ap_eval_report.json = absent
```

判断:

```text
进程仍在推进。
当前没有 sample_*_blocker.json。
full report 尚未写出, 因此不能导入 rows/native_int8_original60_ap_rows_v1.jsonl。
```

## 3. Native INT8 AP route readiness 状态

readiness 产物:

```text
exports/native_int8_ap_route_readiness_20260628_latest.json
exports/native_int8_ap_route_readiness_20260628_latest.md
```

核心结论保持:

```text
label_count = 60
full_onnx_route_manifest_labels = 60
checkpoint_consistent_centered_conv_route_labels = 1
tensor_quant_params_v2_labels = 1
ap_output_dequant_smoke_labels = 1
route_missing_labels = []
labels_needing_ap_route_assets = 59
```

解释:

```text
60/60 native INT8 latency/energy full-ONNX route manifest 只证明 latency/energy route 资产覆盖。
AP-ready 的 checkpoint-consistent centered-conv route + tensor_quant_params_v2 calibration + output-dequant smoke 目前仍只有 s0_024。
```

## 4. 下一步

1. 继续监控 s0_024 full AP v2 到 `full_ap_eval_report.json` 写出。
2. report 写出后先审 gate:

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

3. gate 通过后运行 importer, 写入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

4. 运行默认 coverage + completion queue 即可; 默认路径读取已修复:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

5. 若 full AP 失败或 importer 拒绝, 保存 raw artifact/blocker, 做最小复现和修复, 不能停止整阶段。
6. s0_024 通过后, 将 AP-ready route/calibration/bridge 模板推广到剩余 59 个 label。

## 5. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 energy/AP 收口, 单 agent 执行, 不启动 agent team。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP 0/60 measured; completion review jobs_requiring_action=115。latency_ms 口径固定为 H800 TVM compiled backbone/subnet module end-to-end, input=spatial_features, output=backbone/subnet multiscale outputs, full_network_claim=false, 不能声明为完整 perception pipeline 或真实 RSU 物理边缘设备绝对 latency。本轮已修复 scripts/stage2_generate_original60_quant_state_coverage.py 默认路径: 裸跑 coverage 会自动读取 output_root/rows 下 fp16_true_original60_*、native_int8_full_onnx_original60_* 和 AP rows, 不再回写 no_claim; 已通过 Stage2Original60QuantCoverageCliTest 10 tests 和真实目录裸跑验证。H800 上 s0_024 native INT8 full AP v2 正在运行: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, runner_pid=3810190, python_pid=3810196, 最新监控到 2026-06-28T09:24:54+08:00, bridge_call_000832, sample_blocker_count=0, full_report_absent。readiness 报告仍显示 original60 60/60 有 latency/energy full-ONNX route manifest, 但只有 s0_024 有 AP-ready checkpoint-consistent centered-conv route + tensor_quant_params_v2 + output-dequant smoke, 剩余 59 个 label 需要先补 AP-ready route/calibration/smoke 资产。下一步继续监控 s0_024 full AP v2; full_report 产生后审 gate, 通过后 importer 写入 native_int8_original60_ap_rows_v1.jsonl, 再刷新 coverage/review。随后把 s0_024 的 AP-ready route/calibration/bridge 模板推广到剩余 59 个 label, 同时继续 FP16 AP checkpoint recovery/full true-eval。遇到任何失败不能直接停止: 保存 raw artifact/stdout/stderr/runner command/GPU id/failure reason, 做最小复现和根因修复, smoke 后 full eval 补跑; 只有当前环境无法恢复时才写 per-label blocker。
```
