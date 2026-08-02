# 27_6_28_交接文档_阶段三_INT8FullAPGateSmoke与OutputDequantBlocker

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `26_6_28_交接文档_阶段三_RealActivationBridge成功与INT8FullAPGate计划.md`

本轮新增进展: `s0_024` real activation bridge 已扩展为 full AP gate runner, 支持 `--num-samples`、AP 统计累计、gate 判定和 blocker 落盘。H800 已完成 3-sample gate smoke。结果说明当前主要 blocker 已从“bridge 能否跑通”转移到“TVM uint8 multiscale output 如何正确 dequantize 回 PyTorch head 可消费的数值范围”。

## 0. 当前权威 measured 状态

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
completion jobs requiring action = 115
```

口径仍保持:

```text
latency unit = ms
energy unit = J / inference
latency scope = H800 TVM backbone/subnet module end-to-end
full_network_claim = false
```

## 1. 本轮新增代码能力

新增/修改:

```text
framework/stage2/native_int8_full_onnx.py
framework/tests/test_stage2_native_int8_route.py
scripts/stage2_h800_native_int8_real_activation_bridge.py
scripts/stage2_native_int8_tvm_worker.py
```

新增 gate helper:

```text
native_int8_full_ap_gate(report, min_samples=1789)
build_full_ap_blocker(...)
```

`scripts/stage2_h800_native_int8_real_activation_bridge.py` 新增参数:

```text
--num-samples
--full-ap-min-samples
--keep-detailed-samples
```

runner 当前能力:

1. patch `model.pyramid_backbone.get_multiscale_feature`。
2. 每帧捕获真实 HEAL `spatial_features`。
3. 每帧调用 tvm310 worker 输出三层 native INT8 multiscale feature。
4. 接回 PyTorch head/postprocess。
5. 用 HEAL `eval_utils.caluclate_tp_fp` 累计 result_stat。
6. 用 HEAL `eval_utils.eval_final_results` 生成 AP30/AP50/AP70。
7. 对 partial eval、全空预测、非 finite AP 写 blocker, 不写 measured row。
8. 默认只保留第一帧详细 `.npy`, 后续帧保留 summary, 避免 full eval raw 目录过大。

## 2. H800 3-sample full AP gate smoke

raw artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/full_ap_gate_smoke3_v1/
```

关键文件:

```text
full_ap_eval_report.json
native_int8_s0_024_full_ap_blocker.json
native_int8_s0_024_output_dequant_blocker.json
activation_quant_summary.json
multiscale_output_summary.json
worker_response_summary.json
head_output_summary.json
postprocess_summary.json
eval_s0_024_native_int8_bridge.yaml
```

H800 command:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${V2X_ROOT}:${V2X_HOME}/heal_research/HEAL \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python scripts/stage2_h800_native_int8_real_activation_bridge.py \
  --label s0_024 \
  --ckpt-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26 \
  --raw-dir .../full_ap_gate_smoke3_v1 \
  --gpu-id 0 \
  --num-samples 3 \
  --keep-detailed-samples 1
```

结果:

```text
status = success
processed_samples = 3
failed_samples = 0
AP30 = 0.0
AP50 = 0.0
AP70 = 0.0
pred_nonempty_count = 0
pred_total_count = 0
gate.status = blocked
gate.reason = partial_eval_num_samples_3_lt_1789
ap_row_allowed = false
rows/native_int8_original60_ap_rows_v1.jsonl = still missing
```

summary 覆盖:

```text
activation_quant_items = 3
worker_response_items = 3
multiscale_output_summary_items = 9
```

## 3. 当前 blocker 解释

当前已经解决:

```text
native INT8 full ONNX backbone route build/run
AP-shape route build/run
TVM worker 跨进程 load/run/output
真实 HEAL activation capture
真实 activation -> TVM worker -> PyTorch head/postprocess
AP gate runner 能累计 AP 并写 blocker
```

当前没有写 INT8 AP measured row 的原因:

1. 3-sample smoke 是 partial eval, `processed_samples=3 < 1789`。
2. 3-sample 全部预测为空, `pred_nonempty_count=0`。
3. 更重要的是 output dequant policy 仍不成立: 当前 TVM route 内部每层 `conv2d_native_int8 -> requant_u8`, 但 worker response 没有每个 multiscale output 的真实 scale/zero_point。
4. bridge 临时使用 activation scale 对 TVM uint8 output 反量化, 导致 head logits 数值异常:

```text
cls_preds sample0:
  max = -21.3475341796875
  mean = -84.4428939819336
  min = -108.17803955078125
```

因此不能盲目跑 1789 帧并把 AP=0 或任何 AP 写入 measured row。必须先建立 output dequant/calibration evidence。

新增 blocker:

```text
native_int8_s0_024_output_dequant_blocker.json
failure_type = output_dequant_policy_blocker
failure_reason = temporary activation-scale dequantization is insufficient for AP measured claim; 3-sample smoke produced pred_nonempty_count=0 and strongly negative cls logits
```

## 4. 下一步计划

### 4.1 INT8 output dequant calibration gate

目标: 在 `s0_024` 上建立可审计的 output dequant policy。

建议执行:

1. 新增 5-sample numeric sanity runner。
2. 同一帧同时运行:

```text
PyTorch reference model.pyramid_backbone.get_multiscale_feature(spatial_features)
TVM worker native INT8 multiscale output
```

3. 对三层 output 分别统计:

```text
PyTorch float output min/max/mean/std
TVM uint8 output min/max/mean/std
finite ratio
candidate dequant scale/zero_point
range alignment error
```

4. 输出:

```text
output_dequant_calibration_summary.json
numeric_sanity_summary.json
```

5. 如果三层 output 可建立稳定 dequant, 再跑 20-sample AP smoke。
6. 如果仍全空或数值不可解释, 写 stronger blocker, 不跑 1789 full eval。

### 4.2 INT8 s0_024 full AP gate

只有 output dequant policy 通过后, 才运行:

```text
--num-samples 1789
```

成功条件:

```text
processed_samples = 1789
failed_samples = 0 或有逐样本 blocker 且不影响有效 eval
AP30/AP50/AP70 finite
pred_nonempty_count > 0
output_dequant_calibration evidence present
```

满足后才允许追加:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

### 4.3 FP16 AP checkpoint recovery

继续:

```text
FP16 AP 5/60 -> 60/60
```

当前 blocker 目录:

```text
raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

## 5. 验证

本地:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 36 tests in 6.037s
OK

python -m py_compile framework/stage2/native_int8_full_onnx.py framework/tests/test_stage2_native_int8_route.py scripts/stage2_h800_native_int8_real_activation_bridge.py scripts/stage2_native_int8_tvm_worker.py
exit 0
```

artifact gate:

```text
processed_samples = 3
failed_samples = 0
ap70 = 0.0
pred_nonempty_count = 0
gate.status = blocked
gate.reason = partial_eval_num_samples_3_lt_1789
rows/native_int8_original60_ap_rows_v1.jsonl does not exist
```

## 6. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 三指标收口, 单 agent 执行, 不启动 agent team。当前口径: INT8 backbone/subnet native route 已打通, full_network_claim=false; latency 是 H800 TVM backbone/subnet module 的端到端推理时间, 对外统一 latency_ms, 可作为 RSU edge-segment backbone/subnet server-side proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理设备实测。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; total AP measured 5/120, jobs_requiring_action=115。本轮新增: s0_024 full AP gate runner 已实现, scripts/stage2_h800_native_int8_real_activation_bridge.py 支持 --num-samples、AP result_stat 累计、gate 判定和 blocker 落盘; H800 3-sample gate smoke 已完成, raw artifact 位于 raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/full_ap_gate_smoke3_v1/, processed_samples=3, failed_samples=0, AP30/AP50/AP70=0, pred_nonempty_count=0, gate=blocked(partial_eval_num_samples_3_lt_1789), rows/native_int8_original60_ap_rows_v1.jsonl 仍不存在。当前主要 blocker 是 output dequant policy: TVM route 每层 conv2d_native_int8 -> requant_u8, 但 worker 没有三层 multiscale output 的真实 scale/zero_point, 当前临时用 activation scale 反量化导致 cls logits strongly negative 且空预测; blocker 已写 native_int8_s0_024_output_dequant_blocker.json。下一步硬目标: 先做 s0_024 5-sample numeric sanity/output dequant calibration, 同一帧同时运行 PyTorch reference get_multiscale_feature 与 TVM worker, 对三层 output 分别记录 PyTorch float range、TVM uint8 range、finite ratio、candidate scale/zero_point、range alignment error, 产出 output_dequant_calibration_summary.json 和 numeric_sanity_summary.json。只有 output dequant policy 通过后, 才跑 20-sample AP smoke 和 1789-frame full AP gate; 若仍全空预测或数值不可解释, 写 stronger blocker, 不允许把 AP=0 或 partial AP 写入 measured row。并行继续 FP16 AP checkpoint recovery: 当前 FP16 AP 5/60, 缺 55 行, blocker 在 raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/; 找到 checkpoint 即运行 true FP16 AP, 找不到写 per-label blocker 并继续其他 label。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 必须保存 blocker artifact/stdout/stderr, 做失败分类、最小复现、反思审查、runner/job queue 修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
