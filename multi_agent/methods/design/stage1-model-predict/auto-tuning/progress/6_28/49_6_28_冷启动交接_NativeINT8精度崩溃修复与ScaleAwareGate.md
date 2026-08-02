# 55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate

## 0.0 2026-06-28 修订摘要: INT8 AP 恢复与 5 点 smoke

本节是最新执行口径, 覆盖本文后续章节中仍残留的过期 importer-first 说法。

```text
用户已确认:
1. 下一阶段 AP smoke 按 5 个 original60 配置点执行, 不是只在一个配置上跑 5 samples。
2. 接受先做 AP 可恢复实验: scale-aware route 暂时允许使用 float32 requant / Add / ReLU 实值换算来验证 AP 是否恢复。
3. rows 去重/precision 口径不是主线; 只有它阻塞下一步实验或 coverage/review 时才优先处理。
4. 精度收口标准是: 非空 + AP30/AP50/AP70 有合理非零趋势 + head/postprocess 分布与 FP16 baseline 可解释。
```

当前执行优先级:

```text
P0: 只读确认 s0_024 importer/rows 状态, 不重复把 importer schema 当主线 blocker。
P1: 以 s0_040 为主根因点, 重建 scale-aware route, 跑 numeric_sanity -> 5-sample AP smoke。
P2: 如果 s0_040 AP smoke 通过, 扩展到 5 个 original60 配置点:
    s0_040, s0_024, s0_056, s1_048, s2_160。
P3: 如果 AP 恢复但 TIR 仍 float32-heavy, 不能宣称最终 native INT8 加速完成;
    后续继续把 requant 收口为 integer fixed-point multiplier/shift route。
```

最新已知远端审计结论:

```text
s0_024:
- full AP 已完成, processed_samples=1789, pred_nonempty_count=1787。
- rows/native_int8_original60_ap_rows_v1.jsonl 中已观察到 s0_024 两条 row:
  1) precision=native_int8
  2) precision=int8
- 远端 importer 已观察到支持 output_dequant_summary["items"]。
- 因此下一阶段不要把 s0_024 importer 修复作为第一主线。

s0_040:
- 旧 route 5-sample smoke: pred_nonempty_count=0, AP30/AP50/AP70=0。
- 旧 numeric_sanity corrcoef 约为:
  pyramid_level0=0.468, pyramid_level1=0.109, pyramid_level2=0.374。
- prefix trace 显示后段输出逐渐塌缩到 zero_point 附近, 更像逐层 requant/scale 漂移,
  不是简单后处理阈值问题。
```

## 0. 任务边界

本窗口只推进 native INT8 精度崩溃修复, 不处理 FP16 AP checkpoint recovery。目标是解决 native INT8 backbone/subnet AP 路线中的:

```text
s0_040 TVM native INT8 输出动态范围塌缩
empty_predictions_all_samples / pred_nonempty_count=0
Conv/Add/ReLU scale-aware requant 不完整
QDQ-heavy / float32-heavy lowering 风险
```

禁止盲跑 full AP。必须按:

```text
numeric_sanity -> 5-sample AP smoke -> 1789-frame full AP
```

逐级 gate。

## 0.1 启动初期必读材料

新窗口启动后先读以下材料, 再改代码或启动 GPU 任务:

```text
本文件:
multi_agent/methods/design/auto-tuning/progress/6_27/55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md

共享背景:
multi_agent/methods/design/auto-tuning/progress/6_27/53_6_28_交接文档_阶段三_GPU45_FP16Checkpoint阻断与INT8ScaleAwareBuilder就绪.md

最近 INT8 关键交接:
multi_agent/methods/design/auto-tuning/progress/6_27/40_6_28_交接文档_阶段三_CalibratedINT8APSmoke非空与RowGate修复.md
multi_agent/methods/design/auto-tuning/progress/6_27/45_6_28_交接文档_阶段三_NativeINT8APImporter就绪与FullAPv2继续运行.md
multi_agent/methods/design/auto-tuning/progress/6_27/52_6_28_交接文档_阶段三_s0040APSmoke阻断与ScaleAwareRoute下一步.md

当前 completion 状态:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json

s0_024 full AP artifact:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix/full_ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix/output_dequant_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix/native_int8_ap_row_import_blocker.json

s0_040 route/numeric/smoke artifact:
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040/native_int8_route_manifest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040/full_onnx_route_attempt.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040/numeric_sanity_calibrated_pyramid_level2_v1/numeric_sanity_summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040/ap_smoke5_calibrated_pyramid_level2_v1/full_ap_eval_report.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040/reference_range_capture_h800_pyramid_level2_v2_cuda_visible_all/tensor_quant_params_calibration_v2_to_pyramid_level2.json

必须读源码与测试:
scripts/stage2_import_native_int8_full_ap_row.py
framework/tests/test_stage2_original60_quant_completion.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
scripts/stage2_h800_native_int8_real_activation_bridge.py
framework/stage2/native_int8_full_onnx.py
framework/tests/test_stage2_native_int8_route.py
```

快速阅读命令:

```bash
cd ${V2X_ROOT}
sed -n '1,260p' multi_agent/methods/design/auto-tuning/progress/6_27/55_6_28_冷启动交接_NativeINT8精度崩溃修复与ScaleAwareGate.md
sed -n '1,180p' scripts/stage2_import_native_int8_full_ap_row.py
sed -n '720,920p' framework/tests/test_stage2_original60_quant_completion.py
sed -n '1,220p' framework/stage2/native_int8_full_onnx.py
```

## 0.2 固定规则与红线

1. 单 agent 执行, 不启动 agent team。
2. 使用系统化调试: 先读错误、复现、定位根因, 再修复。不能猜测式连改。
3. 涉及 importer 或量化逻辑的行为变更必须 TDD: 先加失败测试, 看见失败, 再做最小修复, 再跑相关测试。
4. 启动 GPU0 任务前必须保存 `nvidia-smi` 与 compute-apps 审计结果。不能 kill 或抢占未知任务。
5. 不明文写 H800 密码。只使用 `export '<REDACTED_LEGACY_SECRET>')"`。
6. 禁止盲跑 full AP。必须先过 numeric_sanity, 再过 5-sample AP smoke, 且 `pred_nonempty_count>0` 后才允许 1789-frame full AP。
7. s0_024 full AP row import 成功只说明有一条 measured native INT8 AP row; 其 AP 极低, 不能宣称 INT8 精度崩溃已解决。
8. experimental scale-aware route 在 gate 通过前不能写 canonical latency/energy/AP rows; 如使用 `--allow-non-h800-debug`, 只用于避免写正式 rows, 不能当正式测量。
9. 所有 rows 保持 `full_network_claim=false`; latency_ms 口径固定为 H800 TVM compiled backbone/subnet module end-to-end。
10. 必须保存 raw artifact、command、stdout/stderr、runner pid、GPU id、failure reason、numeric summary、smoke/full AP report。
11. 如果 numeric 仍塌缩, 继续定位 Conv/Add/ReLU requant、zero_point、round/clip、graph_input static quant、TE lowering 与 runtime scale 参数, 不直接停止。
12. 不使用破坏性 git/文件操作, 不回滚用户已有改动。

## 1. 当前权威状态

当前 completion review:

```text
FP16 latency = 60/60 measured
FP16 energy  = 60/60 measured
FP16 AP      = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy  = 60/60 measured
native INT8 AP      = 0/60 measured
completion review AP = 5/120
```

latency_ms 口径固定:

```text
H800 TVM compiled backbone/subnet module end-to-end
input = spatial_features
output = multiscale backbone/subnet outputs
full_network_claim = false
```

所有 latency 指标统一使用 ms。

## 2. H800 最新资源状态

2026-06-28T11:07:42+08:00 重新审计:

```text
GPU0: 4 MiB, 0%
GPU1: 4 MiB, 0%
GPU2: 4 MiB, 0%
GPU3: 4 MiB, 0%
GPU4: 4 MiB, 0%
GPU5: 4 MiB, 0%
GPU6: 8 MiB, 0%
GPU7: 4 MiB, 0%
compute-apps: empty
```

也就是说 GPU0 当前可用于 INT8 修复。但冷启动后仍必须再次审计, 不用本快照直接启动。

连接方式, 不要明文写密码:

```bash
export '<REDACTED_LEGACY_SECRET>')"
ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

远端常用路径:

```text
V2X_ROOT=${V2X_ROOT}
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
TVM_PY=${V2X_DATA_ROOT}/tvm310/bin/python
RAW_ROOT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
ROWS=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl
```

## 3. 已完成代码准备

已在本地与 H800 同步过的关键文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
scripts/stage2_h800_native_int8_real_activation_bridge.py
framework/stage2/native_int8_full_onnx.py
framework/tests/test_stage2_native_int8_route.py
```

新增能力:

```text
--tensor-quant-params-path
load_tensor_quant_params(...)
tensor_quant_params_for(...)
requant_u8_scale_aware(...)
relu_u8_scale_aware(...)
add_u8_scale_aware(...)
build_full_onnx_te_spec(..., tensor_quant_params=..., graph_input_quant_params=...)
route_spec = full_onnx_topology_scale_aware_tensor_quant_params_v2
quantize_activation_uint8_static(...)
bridge: tensor_quant_params["spatial_features"] -> calibration_static_uint8 input quant
```

已通过测试:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route
```

远端 py_compile 也已通过。冷启动后如有疑问先重跑, 不要重写已有实现。

## 4. 先处理 s0_024 full AP import blocker

s0_024 full AP 已完成, 不是 AP gate 失败:

```text
raw artifact:
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix

status = success
processed_samples = 1789
pred_nonempty_count = 1787
pred_total_count = 54235
ap30 = 0.0016761650535494891
ap50 = 0.0011790254856126326
ap70 = 0.00018487973338387518
ap_row_allowed = true
ap_row_block_reason = ""
full_network_claim = false
```

当前 import blocker:

```text
RuntimeError:output_dequant_summary missing outputs: ['pyramid_level0', 'pyramid_level1', 'pyramid_level2']
```

根因判断:

```text
scripts/stage2_import_native_int8_full_ap_row.py::output_dequant_schemes()
当前只读取 output_dequant_summary["outputs"] 或 ["dequant_outputs"]。
实际 s0_024 output_dequant_summary.json 使用的是 ["items"] list。
items 中包含 pyramid_level0/1/2, scheme=tensor_quant_params_v2。
```

这是 importer schema 适配问题, 不是 full AP 无效。

### TDD 修复步骤

1. 在 `framework/tests/test_stage2_original60_quant_completion.py` 增加一个测试: `output_dequant_summary.json` 只提供 `items` list 时, importer 仍能生成 compliant native INT8 AP row。
2. 先运行该单测, 确认失败。
3. 修改 `scripts/stage2_import_native_int8_full_ap_row.py::output_dequant_schemes()`:

```python
outputs = output_dequant_summary.get("outputs")
if not isinstance(outputs, list):
    outputs = output_dequant_summary.get("dequant_outputs")
if not isinstance(outputs, list):
    outputs = output_dequant_summary.get("items")
if not isinstance(outputs, list):
    outputs = []
```

4. 重跑:

```bash
cd ${V2X_ROOT}
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_original60_quant_completion.Stage2Original60QuantCompletionTest.test_native_int8_full_ap_import_cli_converts_gated_report_to_compliant_ap_row
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_original60_quant_completion
python -m py_compile scripts/stage2_import_native_int8_full_ap_row.py
```

### 同步并导入 s0_024

```bash
export '<REDACTED_LEGACY_SECRET>')"
scp -P 30001 scripts/stage2_import_native_int8_full_ap_row.py \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/scripts/stage2_import_native_int8_full_ap_row.py

ssh -o StrictHostKeyChecking=accept-new -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>
```

远端执行:

```bash
cd ${V2X_ROOT}
PY=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python
RAW24=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024/ap_full_calibrated_pyramid_level2_v2_empty_summary_fix
ROUTE24=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_centered_conv_v1/s0_024
QP24=$ROUTE24/reference_range_capture_h800_pyramid_level2_v1/tensor_quant_params_calibration_v2_to_pyramid_level2.json
ROWS=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl

$PY scripts/stage2_import_native_int8_full_ap_row.py \
  --label s0_024 \
  --width 24,128,256 \
  --raw-dir "$RAW24" \
  --route-dir "$ROUTE24" \
  --tensor-quant-params-path "$QP24" \
  --rows-out "$ROWS" \
  --run-id native_int8_original60_full_ap_s0_024_20260628_fullapv2 \
  --created-at 2026-06-28T02:45:30Z \
  --min-samples 1789
```

成功后刷新:

```bash
PYTHONPATH=${V2X_ROOT} $PY scripts/stage2_generate_original60_quant_state_coverage.py
PYTHONPATH=${V2X_ROOT} $PY scripts/stage2_generate_fp16_int8_original60_completion_queue.py
```

## 5. s0_040 scale-aware gate

s0_040 已有 checkpoint-consistent route 和 tensor quant params:

```text
OLD_ROUTE=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040

QP40=$OLD_ROUTE/reference_range_capture_h800_pyramid_level2_v2_cuda_visible_all/tensor_quant_params_calibration_v2_to_pyramid_level2.json
```

关键 scale:

```text
spatial_features scale = 0.049095408121744795
spatial_features zero_point = 0
param_count = 116
pyramid_level0/1/2 also have scale/zero_point
```

已有参考 artifact:

```text
$OLD_ROUTE/native_int8_route_manifest.json
$OLD_ROUTE/full_onnx_route_attempt.json
$OLD_ROUTE/numeric_sanity_calibrated_pyramid_level2_v1/numeric_sanity_summary.json
$OLD_ROUTE/ap_smoke5_calibrated_pyramid_level2_v1/full_ap_eval_report.json
$OLD_ROUTE/reference_range_capture_h800_pyramid_level2_v2_cuda_visible_all/tensor_quant_params_calibration_v2_to_pyramid_level2.json
```

先审计旧命令:

```bash
cd ${V2X_ROOT}
OLD_ROUTE=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040
for f in \
  "$OLD_ROUTE/numeric_sanity_calibrated_pyramid_level2_v1/runner_command.json" \
  "$OLD_ROUTE/ap_smoke5_calibrated_pyramid_level2_v1/runner_command.json" \
  "$OLD_ROUTE/full_onnx_route_attempt.json"
do
  echo "==== $f ===="
  test -f "$f" && ${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python -m json.tool "$f" | head -n 120 || true
done
```

### 重建 scale-aware route

建议新 run_id, 不覆盖旧 route:

```bash
cd ${V2X_ROOT}
TVM_PY=${V2X_DATA_ROOT}/tvm310/bin/python
RAW_PARENT=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route
QP40=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040/reference_range_capture_h800_pyramid_level2_v2_cuda_visible_all/tensor_quant_params_calibration_v2_to_pyramid_level2.json
QUEUE=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/fp16_int8_original60_completion_queue_v1.jsonl

CUDA_VISIBLE_DEVICES=0 \
STAGE2_NATIVE_INT8_RAW_PARENT="$RAW_PARENT" \
"$TVM_PY" multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py \
  --run-id 20260628_checkpoint_consistent_s0_040_native_int8_route_scale_aware_v1 \
  --gpu 0 \
  --labels s0_040 \
  --completion-queue "$QUEUE" \
  --input-shape-override spatial_features=1,64,256,256 \
  --tensor-quant-params-path "$QP40" \
  --number 3 \
  --repeat 3 \
  --energy-iters 50 \
  --allow-non-h800-debug \
  --continue-on-failure
```

说明:

```text
--allow-non-h800-debug 这里用于避免把实验 route 的 latency/energy 写入正式 rows。
确认 numeric/smoke gate 通过前, 不写 canonical latency/energy/AP row。
```

新 route 预期:

```text
NEW_ROUTE=${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_scale_aware_v1/s0_040
route_spec = full_onnx_topology_scale_aware_tensor_quant_params_v2
tensor_quant_params_count > 0
```

## 6. numeric_sanity 与 smoke gate

用旧 `runner_command.json` 作为模板, 把 route/artifact 路径改到 NEW_ROUTE。顺序:

```text
1. numeric_sanity_only
2. 5-sample AP smoke
3. full AP
```

numeric_sanity 必查:

```text
pyramid_level0/1/2 corrcoef
RMSE
输出动态范围 min/max/std
非零比例
dequant scheme 是否 tensor_quant_params_v2
旧 route vs scale-aware route 对比
```

放行条件:

```text
pyramid_level1/2 corrcoef 和动态范围相较旧 route 明显改善
输出不是大范围塌缩到常数或近零
5-sample AP smoke pred_nonempty_count > 0
full AP 前必须保存 numeric_sanity_summary.json 和 smoke full_ap_eval_report.json
```

如果 numeric 仍失败, 不要直接 full AP。按顺序排查:

```text
Conv requant scale: input_scale * weight_scale / output_scale
Add 两输入 scale/zero_point 对齐
ReLU zero_point 与 clip 下界
round/clip u8 语义
graph_input static quant 是否真的使用 spatial_features scale/zp
TE lowering 是否仍混入 float32-heavy 或 QDQ-heavy route
runtime scale 参数是否被常量折叠/遗漏
```

## 7. 冷启动验收

本窗口短期验收:

```text
s0_024 row 状态只读确认完成; 如果重复 row 或 precision 口径阻塞 coverage/review, 完成最小去重/口径修正
s0_040 scale-aware route 重建成功
s0_040 numeric_sanity 相较旧 route 有明确改善
s0_040 5-sample smoke pred_nonempty_count > 0
完成 5 个 original60 配置点 AP smoke: s0_040, s0_024, s0_056, s1_048, s2_160
5 点 smoke 均满足: 非空 + AP30/AP50/AP70 有合理非零趋势 + head/postprocess 分布与 FP16 baseline 可解释
```

最终验收:

```text
native INT8 AP 从 0/60 推进到 60/60 measured
每个 AP row 有 raw artifact、route manifest、tensor_quant_params、output_dequant_summary、AP30/AP50/AP70、num_samples、pred_nonempty_count
不声明 full_network_claim=true
如果 AP 可恢复 route 仍是 float32-heavy, 继续推进 integer fixed-point native INT8 route, 不把 float32-heavy 结果当最终加速结论
```

## 8. 冷启动 /goal

```text
/goal 只推进 Stage2 original60 native INT8 精度崩溃修复和 5 点 AP smoke, 不处理 FP16。启动前先阅读本文件 0.0 修订摘要和初期必读材料, 重新审计 H800 GPU0 与 compute-apps, 不依赖旧快照, 不 kill 未知任务。先只读确认 s0_024 importer/rows 状态: 远端应已支持 output_dequant_summary["items"], rows/native_int8_original60_ap_rows_v1.jsonl 里可能已有 s0_024 两条 row; 如果重复 row 或 precision=native_int8/int8 口径阻塞 coverage/review, 用最小 TDD 修复或去重收口, 否则不要把 s0_024 importer 当主线 blocker。主线任务是修复 s0_040 native INT8 动态范围塌缩: 用 tensor_quant_params_calibration_v2_to_pyramid_level2.json 重建 scale-aware route, route_spec 必须为 full_onnx_topology_scale_aware_tensor_quant_params_v2, 依次跑 numeric_sanity -> 5-sample AP smoke。numeric_sanity 必须检查 pyramid_level0/1/2 corrcoef、RMSE、min/max/std、非零比例、dequant scheme, 并与旧 route corrcoef 约 0.468/0.109/0.374 对比; 如果仍失败, 用 op-level/prefix trace 二分定位第一个 corrcoef/动态范围崩塌的 op 或 residual block, 继续排查 Conv requant、Add scale/zero_point、ReLU clip、round/clip、graph_input static quant、TE lowering 和 runtime scale 参数。用户已确认可以先接受 AP 可恢复实验, 因此 scale-aware route 暂时允许 float32 requant 来验证 AP 是否恢复, 但每次 build 必须保存 TIR dtype inventory; 若 AP 恢复但 float32-heavy, 不能宣称最终 native INT8 加速完成, 后续要改 integer fixed-point requant。s0_040 通过后扩展到 5 个 original60 配置点 AP smoke: s0_040、s0_024、s0_056、s1_048、s2_160。5 点 smoke 的收口标准是 processed_samples>=5、pred_nonempty_count>0、AP30/AP50/AP70 有合理非零趋势、head_output_summary 和 postprocess_summary 不塌缩且与 FP16 baseline 分布差异可解释。遇到 build/eval/import/adapter/gate 问题不停止, 保存 raw artifact、command、stdout/stderr、runner pid、GPU id、failure reason、numeric summary、smoke report 和 FP16-vs-INT8 分布对比, 做根因判断和脚本修复后重跑 gate。
```
