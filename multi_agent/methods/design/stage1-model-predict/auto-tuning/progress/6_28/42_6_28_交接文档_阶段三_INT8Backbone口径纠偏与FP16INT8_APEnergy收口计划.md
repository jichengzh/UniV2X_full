# 48_6_28_交接文档_阶段三_INT8Backbone口径纠偏与FP16INT8_APEnergy收口计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `47_6_28_交接文档_阶段三_FullAPv2进度532与后续Route盘点.md`
- `41_6_28_交接文档_阶段三_INT8BackboneLatency口径确认与FP16INT8实测补点收口.md`
- `33_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_EnergyAP大范围补点计划.md`

本轮核心目的: 回答 INT8 backbone 是否打通、latency 是否能按 RSU 边缘段速度理解这两个口径问题, 并把下一阶段目标收口到一个非常具体的单 agent 任务: 在 existing original60 的 60 个配置上完成 `fp16` 与 `native_int8` 的 energy/AP 审计、实测补点和总表刷新。遇到任何问题不能直接停止; 必须保存证据、最小复现、反思、审查、修复和补跑。只有当前环境确实无法恢复时, 才写 per-label blocker。

## 0. 两个口径问题的明确回答

### 0.1 INT8 的 backbone 实现是否已经打通

结论: 是, 但声明范围必须限定为 `H800 + TVM native INT8 backbone/subnet build/run/latency/energy route`。

当前可以声明:

```text
quant_method = h800_tvm_native_int8_backbone_subnet
quant_scope = backbone_subnet_native_int8
route_spec = full_onnx_topology_conv_relu_add_identity_v1
runtime = TVM graph executor / TE route
hardware_target = H800 Hopper
input = spatial_features
output = backbone/subnet multiscale outputs
full_network_claim = false
```

已经完成的 original60 measured rows:

```text
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60
```

当前还不能声明:

```text
native INT8 AP measured 已完成
native INT8 full 1789-sample AP eval 已完成
完整 HEAL/V2X perception pipeline TVM INT8 已完成
真实 RSU 物理边缘设备 latency 已实测
```

最新 INT8 AP 状态:

```text
s0_024 native INT8 full AP v2 仍在 H800 GPU0 运行
runner_pid = 3810190
python_pid = 3810196
latest_monitor_time = 2026-06-28T09:07:34+08:00
elapsed = 00:48:23
max_worker_tmp = bridge_call_000591
sample_blocker_count = 0
full_ap_eval_report.json = absent
```

判断: native INT8 backbone/subnet 到 AP bridge 的路径已经越过 smoke 和 v1 empty-prediction summary blocker, 但 full AP gate 尚未完成, 所以 `rows/native_int8_original60_ap_rows_v1.jsonl` 仍不得生成或导入。

### 0.2 当前所有配置的 latency 是否是 backbone 端到端推理速度, 是否近似 RSU 边缘段设备推理速度

结论分两层:

```text
是: 当前 latency 是 compiled backbone/subnet module 的模块端到端实测。
否: 当前 latency 不是完整 perception pipeline 端到端, 也不能直接等同真实 RSU 物理边缘设备绝对速度。
```

统一口径:

```text
latency_ms = latency_p50_us / 1000
measurement = H800 + TVM compiled backbone/subnet module runtime
input_shape = spatial_features [2, 64, 128, 256]
fp16 quant_scope = backbone_only
native_int8 quant_scope = backbone_subnet_native_int8
full_network_claim = false
```

可以写成:

```text
H800 TVM backbone/subnet module latency in ms.
可以作为 RSU-side backbone/subnet dense workload 的 server-side proxy 或趋势对比。
```

不能写成:

```text
完整 V2X 感知网络端到端 latency
包含 dataloader / encoder / head / NMS / postprocess / dataset eval 的 full pipeline latency
真实 RSU 物理边缘设备绝对 latency
```

原因: 目前 latency rows 的硬件是 `H800 Hopper`, runtime 是 TVM VM 或 TVM graph executor, 口径只覆盖 `spatial_features -> backbone/subnet multiscale outputs`。如果要声明真实 RSU 边缘设备速度, 后续必须在目标 RSU 硬件或等价边缘 GPU/SoC 上重新 build/run, 并把 hardware target 单独写入 rows。

## 1. 当前权威覆盖状态

本轮复核的 rows 文件:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl = 60
rows/fp16_true_original60_energy_rows_v1.jsonl = 60
rows/fp16_true_original60_ap_rows_v1.jsonl = 5
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60
rows/native_int8_original60_ap_rows_v1.jsonl = missing / 0
```

按 precision 拆分:

| metric | FP16 | native INT8 | total |
|---|---:|---:|---:|
| latency_ms | 60/60 measured | 60/60 measured | 120/120 |
| energy_J_per_inference | 60/60 measured | 60/60 measured | 120/120 |
| AP70 | 5/60 measured | 0/60 measured | 5/120 |

当前 review 状态:

```text
exports/fp16_int8_original60_completion_review_latest.md
created_at = 2026-06-27T22:58:16Z
total_jobs = 120
jobs_requiring_action = 115
precision_counts = {"fp16": 60, "int8": 60}
axis_status_counts = {
  "latency": {"measured": 120},
  "energy": {"measured": 120},
  "ap": {"measured": 5, "no_claim": 115}
}
```

FP16 AP checkpoint inventory:

```text
exports/fp16_ap_checkpoint_inventory_20260628_latest.md
label_count = 60
measured_fp16_ap_labels = s0_024, s0_040, s0_056, s1_048, s2_160
unmeasured_labels_with_candidates = 0
labels_without_candidates = 55
```

## 2. 下一阶段硬目标

下一阶段最终目标收口为:

```text
configs = existing original60 60 labels
precisions = fp16, native_int8
latency = 120/120 measured, 对外统一 latency_ms
energy = 120/120 measured and audited, 异常/证据缺失单点补跑
AP = 120/120 measured
jobs_requiring_action = 0, unless a per-label blocker proves unsolvable in current environment
full_network_claim = false for all backbone/subnet latency/energy rows
```

更具体地说:

| output | current | target | action |
|---|---:|---:|---|
| FP16 latency | 60/60 | 60/60 | 保持, 只对外展示 `latency_ms` |
| FP16 energy | 60/60 | 60/60 audited | 复核 raw artifact/digest/telemetry, 异常单点补跑 |
| FP16 AP70 | 5/60 | 60/60 measured | 补 55 个 full true-eval rows |
| native INT8 latency | 60/60 | 60/60 | 保持, 只对外展示 `latency_ms` |
| native INT8 energy | 60/60 | 60/60 audited | 复核 native route/telemetry, 异常单点补跑 |
| native INT8 AP70 | 0/60 | 60/60 measured | 从 s0_024 full gate 扩展到 60 个 full AP rows |

目标输出文件:

```text
rows/latency_original60_quant_rows_v1.jsonl
rows/energy_original60_quant_rows_v1.jsonl
rows/ap_original60_quant_rows_v1.jsonl
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_original60_ap_rows_v1.jsonl
exports/original60_quant_three_metric_summary_latest.md
exports/original60_quant_three_metric_summary_latest.json
exports/fp16_int8_original60_completion_review_latest.md
exports/fp16_int8_original60_completion_review_latest.json
exports/fp16_int8_original60_gap_report_latest.md
exports/fp16_int8_original60_gap_report_latest.json
```

## 3. 单 agent 执行计划

本阶段只使用一个执行 agent。该 agent 需要自己完成实验、审查、反思、修复和补跑, 不启动双 agent 或 reviewer team。

### 3.1 第一批: 继续监控并收口 s0_024 native INT8 full AP gate

H800 监控命令:

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '
RAW="${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix"
PY="${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python"
echo DATE=$(date -Iseconds)
ps -p "$(cat "$RAW/runner.pid")" -o pid,stat,etime,cmd || true
ps --ppid "$(cat "$RAW/runner.pid")" -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd || true
echo BLOCKER_COUNT=$(find "$RAW" -maxdepth 1 -name "sample_*_blocker.json" | wc -l)
echo MAX_WORKER_TMP=$(find "$RAW/worker_tmp" -maxdepth 1 -type d -name "bridge_call_*" 2>/dev/null | sed "s#.*/bridge_call_##" | sort | tail -1)
if [ -f "$RAW/full_ap_eval_report.json" ]; then "$PY" -m json.tool "$RAW/full_ap_eval_report.json"; fi
'
unset password-based SSH (disabled; use an SSH key)
```

如果 full report 通过 gate, 在 H800 上导入 AP row:

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

必须满足的 native INT8 AP row gate:

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

如果 full report 失败或 importer 拒绝, 不得停止整阶段。执行:

```text
1. 保存 full raw artifact。
2. 读取 sample blocker / full report blocker / native_int8_ap_row_import_blocker.json。
3. 按 failure type 做最小复现。
4. 写测试或最小断言复现问题。
5. 修复后先跑 smoke, 再重跑 full eval。
6. 只有确认当前环境无法恢复时, 写 per-label blocker。
```

### 3.2 第二批: FP16 AP 补 55 行

当前 FP16 AP 缺口:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl = 5
remaining = 55 labels
```

执行顺序:

```text
1. 扩展 checkpoint recovery inventory, 覆盖 H800 当前目录、/exdata、历史 manifest、raw AP eval 输出和备份目录。
2. 对找到 checkpoint 的 label 运行 FP16 full true-eval, num_samples=1789。
3. postprocess 前保持 HEAL postprocessor 所需 FP32 math, 不把 AP runner 的后处理误当成 FP16 kernel。
4. 每完成一批就刷新 fp16_true_original60_ap_rows_v1.jsonl 和 ap_original60_quant_rows_v1.jsonl。
5. 找不到 checkpoint 的 label 必须写 per-label blocker, 包含 searched_paths、reason、next_candidate_paths、recovery_action。
```

FP16 AP row 必须包含:

```text
label
precision = fp16
AP30/AP50/AP70
num_samples = 1789
checkpoint_path
checkpoint_digest
eval_config
raw_artifact
measurement_source = true_eval
full_network_claim = false
```

### 3.3 第三批: native INT8 AP 从 s0_024 扩展到 60 行

执行顺序:

```text
1. 以 s0_024 full AP gate 为模板, 固化 route builder、calibration、bridge、importer 参数。
2. 对 s0_040/s1_048 等已有 checkpoint/ONNX 的 label 先生成 checkpoint-consistent native INT8 route, 不复用不匹配 route。
3. 对每个 label 执行 calibrated full 1789-sample native INT8 AP eval。
4. 通过 importer gate 后写入 rows/native_int8_original60_ap_rows_v1.jsonl。
5. 每批刷新 completion review 和 gap report。
```

native INT8 AP 不允许的替代项:

```text
1-sample smoke AP
hybrid FP16/FP32 AP
simulated AP
predicted AP
没有 full 1789-sample gate 的 partial eval
不匹配 checkpoint 的 route 输出
```

### 3.4 第四批: energy 审计与异常补跑

虽然当前 FP16/native INT8 energy 均为 60/60 measured, 下一阶段仍需做证据审计:

```text
1. 校验 raw_artifact 存在。
2. 校验 telemetry_source / active_power_samples / idle_power_samples 存在。
3. 校验 row 的 latency_config_id 或 route digest 能追溯到对应 latency/route。
4. 校验 full_network_claim=false。
5. 对 artifact 缺失、digest 不一致、功耗窗口异常或硬件口径不一致的 label 做单点补跑。
```

验收口径:

```text
FP16 energy = 60/60 measured and audited
native INT8 energy = 60/60 measured and audited
所有异常补跑都有 raw artifact 和审计记录
```

## 4. 总表刷新与验收

每完成一批 AP/energy 更新后运行。注意: `stage2_generate_original60_quant_state_coverage.py` 不能裸跑默认参数; 默认参数不会自动读取 60/60 的 true-FP16 与 native-INT8 rows, 会把 canonical 表回写成保守 `no_claim`。必须显式传入当前权威 rows:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py \
  --fp32-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_latency_smoke_rows_v1.jsonl \
  --fp32-latency-remap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp32_latency_original60_remapped_rows_v1.jsonl \
  --fp16-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_latency_smoke_rows_v1.jsonl \
  --fp16-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_latency_rows_v1.jsonl \
  --fp16-energy-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_ap_energy_smoke_rows_v1.jsonl \
  --fp16-energy-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_energy_rows_v1.jsonl \
  --int8-latency-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_latency_smoke_rows_v1.jsonl \
  --int8-energy-smoke-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/int8_ap_energy_smoke_rows_v1.jsonl \
  --native-int8-full-onnx-latency-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl \
  --native-int8-full-onnx-energy-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl \
  --fp16-ap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl \
  --int8-ap-rows multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

如果脚本参数已变化, 先用:

```bash
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_original60_quant_state_coverage.py --help
PYTHONPATH=${V2X_ROOT} python scripts/stage2_generate_fp16_int8_original60_completion_queue.py --help
```

阶段验收标准:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl = 60
rows/native_int8_original60_ap_rows_v1.jsonl = 60
FP16 energy audit pass = 60/60
native INT8 energy audit pass = 60/60
exports/original60_quant_three_metric_summary_latest.md 展示 latency_ms, energy_J_per_inference, AP70
exports/fp16_int8_original60_completion_review_latest.md jobs_requiring_action = 0
```

如果仍有 blocker:

```text
blocker_count 必须小于等于确实无法恢复的 label 数
每个 blocker 必须有 raw artifact、最小复现、失败栈、已尝试修复、下一步恢复动作
不得用 blocker 代替未尝试的实验
```

## 5. 下一步 /goal 命令

```text
/goal 继续 Stage2 original60 FP16/native INT8 energy/AP 收口, 单 agent 执行, 不启动 agent team。先明确口径: INT8 backbone/subnet native route 已在 H800+TVM 上打通 latency/energy build-run, 但 native INT8 AP full gate 尚未完成; latency_ms 是 H800 TVM compiled backbone/subnet module 端到端, input=spatial_features, output=backbone/subnet multiscale outputs, full_network_claim=false, 不能声明为完整 perception pipeline 或真实 RSU 物理边缘设备绝对 latency。当前权威覆盖: FP16 latency 60/60, FP16 energy 60/60, FP16 AP 5/60; native INT8 latency 60/60, native INT8 energy 60/60, native INT8 AP 0/60; completion review jobs_requiring_action=115。H800 上 s0_024 native INT8 full AP v2 正在运行: raw_dir=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix, runner_pid=3810190, python_pid=3810196, 最新监控到 2026-06-28T09:07:34+08:00, bridge_call_000591, sample_blocker_count=0, full_report_absent。下一阶段硬目标: 完成 existing original60 60 个配置的 fp16 与 native_int8 AP 实测补点到 120/120 AP measured, 并完成 FP16/native INT8 energy 120/120 measured rows 的 raw artifact/telemetry/digest 审计和异常单点补跑; 总表最终刷新到 latency_ms、energy_J_per_inference、AP70 三指标齐全, jobs_requiring_action=0。遇到任何失败不能直接停止: 必须保存 raw artifact, 做最小复现, 写测试或断言, 根因分析, 修复, smoke 验证, full eval 补跑; 只有当前环境无法恢复时才写 per-label blocker。
```
