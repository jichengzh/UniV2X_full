# 51_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8全量AP收口计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `50_6_28_交接文档_阶段三_Coverage默认刷新修复与FullAPv2进度812.md`
- `49_6_28_交接文档_阶段三_INT8APRouteReadiness与FullAPv2进度710.md`
- `41_6_28_交接文档_阶段三_INT8BackboneLatency口径确认与FP16INT8实测补点收口.md`

本轮核心目的:

```text
1. 明确回答 INT8 backbone/subnet 是否打通。
2. 明确 latency_ms 的测量口径, 避免误写为完整感知 pipeline 或真实 RSU 物理设备绝对 latency。
3. 把下一阶段收口成一个具体目标: existing original60 60 个配置 x {fp16,native_int8} 的 energy/AP 全量实测和总表刷新。
4. 明确失败处理原则: 遇到问题先保存证据、最小复现、反思、审查、修复和补跑, 不直接停止; 只有当前环境或 checkpoint 缺失无法解决时才写 per-label blocker。
```

## 0. 两个口径问题的结论

### 0.1 INT8 backbone 实现是否已经打通

结论: **已打通, 但范围限定为 `H800 + TVM native INT8 backbone/subnet route`。**

可以声明:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
route_spec = full_onnx_topology_conv_relu_add_identity_v1
runtime = TVM graph executor / TE route
device = H800
full_network_claim = false
```

当前权威 measured 覆盖:

```text
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
```

新增 AP-ready route 进展:

```text
s0_024:
  checkpoint-consistent centered-conv native INT8 route 已有
  tensor_quant_params_v2 calibration 已有
  output-dequant smoke 已通过
  1789-frame full AP v2 正在 H800 运行, 尚未写出 full_ap_eval_report.json

s0_040:
  checkpoint-consistent multiscale ONNX export 成功
  checkpoint-consistent native INT8 full-ONNX route 已成功
  latency_ms = 12.291168
  energy_J = 2.7866748651142528
  artifact_digest = a8451ec81d54bb7620fc2a05b5c81e9215cad7b658ae3cd0f418b8a6f23bf8f5
  还未做 reference range calibration / output-dequant smoke / full AP
```

当前不能声明:

```text
native INT8 AP measured 已完成
native INT8 full 1789-sample AP eval 已完成
完整 HEAL / V2X perception pipeline TVM INT8 已完成
真实 RSU 物理边缘设备 latency 已实测
```

### 0.2 当前 latency 是否是 backbone 端到端推理速度

结论: **是, 但这里的端到端只指 `backbone/subnet compiled module` 的模块端到端, 不是完整 perception pipeline 端到端。**

统一口径:

```text
latency_ms = latency_p50_us / 1000
measurement = H800 + TVM compiled backbone/subnet module runtime
input = spatial_features
output = multiscale backbone/subnet outputs
full_network_claim = false
```

可以写:

```text
H800 TVM backbone/subnet module latency in ms.
可以作为 RSU-side backbone/subnet workload 的 server-side proxy。
```

不能写:

```text
完整感知网络端到端 latency
包含 dataloader / encoder / head / NMS / postprocess / dataset eval 的 full pipeline latency
真实 RSU 物理边缘设备绝对 latency
```

如果要写“近似于 RSU 边缘段设备速度”, 必须加限定: 它近似的是 RSU-side backbone/subnet workload 在当前 H800 TVM route 上的推理速度, 不是任何真实 RSU 物理设备的绝对推理速度。

## 1. 当前权威覆盖状态

权威 review:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
```

当前 summary:

```text
latency measured = 120/120
energy measured = 120/120
AP measured = 5/120
jobs_requiring_action = 115
precision_counts = {"fp16": 60, "int8": 60}
```

分精度状态:

| precision | latency_ms | energy | AP70 |
|---|---:|---:|---:|
| FP16 | 60/60 measured | 60/60 measured | 5/60 measured |
| native INT8 | 60/60 measured | 60/60 measured | 0/60 measured |

权威 rows:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl
rows/fp16_true_original60_energy_rows_v1.jsonl
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

当前仍不能生成或导入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

原因: 还没有任何 native INT8 1789-frame full AP gate 通过。smoke、hybrid AP、FP16/FP32 AP、predicted AP 或 blocker 都不能写成 native INT8 measured AP row。

## 2. 最新 H800 运行状态

### 2.1 s0_024 native INT8 full AP v2

raw:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix
```

最新监控:

```text
time = 2026-06-28T09:43:57+08:00
runner_pid = 3810190
python_pid = 3810196
elapsed = 01:24:46
sample_blocker_count = 0
full_ap_eval_report.json = absent
```

判断:

```text
进程仍在运行。
当前没有 sample_*_blocker.json。
full report 尚未写出, 所以不能导入 native_int8_original60_ap_rows_v1.jsonl。
```

### 2.2 s0_040 checkpoint-consistent native INT8 route

run root:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1
```

export:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/checkpoint_consistent_s0_040_multiscale_export_v1/export_report.json
```

route result:

```text
status = success
route_probe = full_onnx_route_probe.json
manifest = s0_040/native_int8_route_manifest.json
artifact = s0_040/s0_040_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
latency_ms = 12.291168
energy_J = 2.7866748651142528
op_counts = {"Add": 16, "Conv": 51, "Relu": 48}
full_network_claim = false
```

本轮修复过的远端接口漂移:

```text
1. H800 route 进程最初使用 UniV2X Python, import tvm 失败。
   修复: route 部分改用 ${V2X_DATA_ROOT}/tvm310/bin/python, 并设置 LD_LIBRARY_PATH=$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path):.../tvm/lib。

2. 远端 completion queue 不存在, route 脚本无法解析 s0_040 width。
   修复: 在 run root 下写最小 width metadata queue:
   s0_040_route_width_queue_min.jsonl

3. 远端 stage2_generate_latency_lut.py / stage2_generate_energy_lut.py 版本滞后, 不接受 --layer-precision-summary-digest。
   修复: 从本地同步两个 generator 脚本, py_compile 通过。

4. 远端 framework/stage2/lut_productization.py 版本滞后, validator 仍强制 INT8 rows 必须 engine_kind=tvm_vm。
   修复: 从本地同步 validator, 并在远端通过:
   framework.tests.test_stage2_native_int8_route.Stage2NativeInt8RouteTest.test_native_int8_latency_row_allows_graph_executor
   framework.tests.test_stage2_native_int8_route.Stage2NativeInt8RouteTest.test_non_native_int8_latency_row_still_requires_tvm_vm
```

注意: `s0_040` 目前只是 AP-ready 资产的第二步。下一步还需要 reference range capture、tensor_quant_params_v2 calibration、output-dequant smoke, 之后才可跑 full AP。

## 3. 下一阶段硬目标

最终目标收口为:

```text
configs = existing original60 60 labels
precisions = fp16, native_int8
latency_ms = 120/120 measured, 保持并统一展示为 ms
energy = 120/120 measured + evidence audited; 发现坏行只做单点补跑
AP70 = 120/120 measured
jobs_requiring_action = 0
full_network_claim = false for all backbone/subnet rows
```

如果某个 label 因 checkpoint 缺失、环境不可恢复或 full AP gate 永久失败无法完成, 不能直接停止; 必须写 per-label blocker, 且 blocker 至少包含:

```text
label
precision
width
raw_dir
command
stdout/stderr
failure_reason
minimal_repro
actions_tried
why_unsolved_in_current_environment
next_recovery_candidate
```

## 4. 具体执行计划

### 4.1 INT8 AP 路线

1. 继续监控 `s0_024` full AP v2。
2. 一旦 `full_ap_eval_report.json` 写出, 先审 gate:

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

3. gate 通过后运行 importer, 只追加第一行 native INT8 AP measured row:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

4. 对 `s0_040` 继续做:

```text
reference range capture to pyramid_level2
tensor_quant_params_calibration_v2_to_pyramid_level2.json
output-dequant smoke
1789-frame full AP
```

5. `s0_024` 与 `s0_040` 两点都通过后, 批量推广到剩余 58 个 label:

```text
checkpoint export -> checkpoint-consistent native INT8 route -> reference range capture -> output-dequant smoke -> full AP -> importer
```

### 4.2 FP16 AP 路线

当前 FP16 AP:

```text
5/60 measured
55/60 no_claim
```

执行:

```text
1. 从 completion review 筛出 FP16 AP=no_claim 的 55 个 label。
2. 做 checkpoint recovery inventory, 搜索 H800 stage1/stage2 checkpoint、历史 manifest、备份目录和 label alias。
3. 找到 checkpoint 后运行 scripts/stage2_h800_true_fp16_ap_eval.py。
4. 每完成一批刷新 fp16_true_original60_ap_rows_v1.jsonl 和 canonical ap_original60_quant_rows_v1.jsonl。
5. 找不到 checkpoint 的 label 写 per-label blocker, 继续其他 label, 不阻塞整批。
```

### 4.3 Energy 路线

当前 energy 已是:

```text
FP16 energy = 60/60 measured
native INT8 energy = 60/60 measured
```

下一阶段不是盲目重跑全部 energy, 而是做 evidence audit:

```text
1. 检查每行 raw_artifact 是否存在。
2. 检查 digest / telemetry source / engine_kind / quant_method / full_network_claim=false。
3. 对缺 artifact、digest 不一致、字段不合规或 telemetry 异常的 label 做单点 rerun。
4. 总表保留 energy_J_per_inference, latency 统一显示 latency_ms。
```

## 5. 刷新命令

每批补点后都运行:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

审阅入口:

```text
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/native_int8_ap_route_readiness_20260628_latest.md
exports/native_int8_ap_route_readiness_20260628_latest.json
```

## 6. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 energy/AP 全量收口, 单 agent 执行, 不启动 agent team。当前口径必须固定: latency_ms 是 H800 TVM compiled backbone/subnet module end-to-end, input=spatial_features, output=multiscale backbone/subnet outputs, full_network_claim=false; 可以作为 RSU-side backbone/subnet workload 的 server-side proxy, 不能声明为完整 perception pipeline latency 或真实 RSU 物理设备绝对 latency。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; completion review latency=120/120, energy=120/120, AP=5/120, jobs_requiring_action=115。INT8 backbone/subnet native route 已打通: original60 60/60 有 native INT8 latency/energy full-ONNX route measured; s0_024 有 checkpoint-consistent route + tensor_quant_params_v2 + output-dequant smoke, full AP v2 正在 H800 运行, raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, runner_pid=3810190, python_pid=3810196, 2026-06-28T09:43:57+08:00 仍 running, sample_blocker_count=0, full_report_absent。s0_040 已完成 checkpoint-consistent ONNX export 和 native INT8 full-ONNX route, run_root=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1, route status=success, latency_ms=12.291168, energy_J=2.7866748651142528, artifact_digest=a8451ec81d54bb7620fc2a05b5c81e9215cad7b658ae3cd0f418b8a6f23bf8f5; 它还不是 AP measured row, 下一步需要 reference range capture -> tensor_quant_params_v2 -> output-dequant smoke -> full AP。下一阶段硬目标: 完成 existing original60 60 labels x {fp16,native_int8} 的 latency_ms/energy/AP70 全量 measured 和 evidence audit; latency 120/120 保持, energy 120/120 做 raw artifact/digest/telemetry/field audit 并只补跑坏行, FP16 AP 从 5/60 补到 60/60, native INT8 AP 从 0/60 补到 60/60, 最终 jobs_requiring_action=0。遇到任何 build/eval/import/SSH/checkpoint/adapter/gate 问题, 不允许直接停止: 必须保存 raw artifact、command、stdout/stderr、runner pid、GPU id、failure reason, 做最小复现、根因判断、反思审查、脚本或队列修复、smoke 和失败 label 补跑; 只有证明当前环境或 checkpoint 缺失无法由执行 agent 解决时才写 per-label blocker 并继续其他 label。
```
