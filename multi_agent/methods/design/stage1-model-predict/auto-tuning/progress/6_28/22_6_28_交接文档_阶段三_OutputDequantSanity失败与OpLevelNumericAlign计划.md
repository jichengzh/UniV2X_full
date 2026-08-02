# 28_6_28_交接文档_阶段三_OutputDequantSanity失败与OpLevelNumericAlign计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `27_6_28_交接文档_阶段三_INT8FullAPGateSmoke与OutputDequantBlocker.md`

本轮新增进展: 已完成 `s0_024` 5-sample numeric sanity/output dequant calibration。结果为 blocked: PyTorch reference `get_multiscale_feature` 与 TVM worker 的三层 uint8 output 在 minmax output dequant 后仍严重不对齐。因此下一步不能直接跑 20-sample AP smoke 或 1789-frame full AP; 必须先做 op-level numeric alignment。

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
framework/tests/test_stage2_native_int8_route.py
scripts/stage2_h800_native_int8_real_activation_bridge.py
```

新增 helper:

```text
build_output_dequant_sanity_record(...)
summarize_dequant_sanity(...)
```

新增 runner mode:

```text
--numeric-sanity-only
```

该模式会:

1. patch `model.pyramid_backbone.get_multiscale_feature`。
2. 同一帧先运行 PyTorch reference `get_multiscale_feature(spatial_features)`。
3. 再运行 TVM worker native INT8 route。
4. 对三层 output 分别比较 reference float output 与 TVM uint8 output。
5. 生成 candidate minmax dequant scale/zero_point。
6. 计算 MAE/RMSE/max_abs/corrcoef。
7. 返回 reference output 给模型继续跑完, 因此 sanity mode 里 HEAL 打印的 AP 是 reference path 的 AP, 不是 INT8 measured AP。

## 2. H800 5-sample numeric sanity 结果

raw artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/output_dequant_sanity5_v1/
```

关键文件:

```text
real_activation_bridge_report.json
numeric_sanity_summary.json
output_dequant_calibration_summary.json
native_int8_s0_024_numeric_alignment_blocker.json
activation_quant_summary.json
worker_response_summary.json
multiscale_output_summary.json
```

H800 command:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${V2X_ROOT}:${V2X_HOME}/heal_research/HEAL \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python scripts/stage2_h800_native_int8_real_activation_bridge.py \
  --label s0_024 \
  --ckpt-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26 \
  --raw-dir .../output_dequant_sanity5_v1 \
  --gpu-id 0 \
  --num-samples 5 \
  --keep-detailed-samples 1 \
  --numeric-sanity-only
```

结果:

```text
processed_samples = 5
failed_samples = 0
records = 15
output_dequant_calibration_summary.status = blocked
rows/native_int8_original60_ap_rows_v1.jsonl = still missing
```

注意: H800 stdout 中显示 `AP30=0.82, AP50=0.82, AP70=0.73` 是 numeric sanity mode 返回 PyTorch reference output 后的 reference path AP, 不是 INT8 measured AP, 不能写入 INT8 AP row。

## 3. 三层 output 对齐结果

| tensor | samples | rmse_mean | mae_mean | corrcoef_mean | reference_range_mean | passed |
|---|---:|---:|---:|---:|---:|---|
| `/resnet/layer0/layer0.2/relu_2/Relu_output_0` | 5 | 19.508309804565794 | 19.45357278926415 | -0.1319593869416177 | 20.474607467651367 | false |
| `/resnet/layer1/layer1.4/relu_2/Relu_output_0` | 5 | 11.109477609991199 | 11.088143867755011 | -0.04836128584113113 | 11.313913154602051 | false |
| `/resnet/layer2/layer2.7/relu_2/Relu_output_0` | 5 | 9.397421905642236 | 9.393676109948832 | 0.006688677727582757 | 9.443699645996094 | false |

解释:

```text
RMSE 接近 reference range 本身, corrcoef 接近 0 或为负。
这不是简单缺少一组三层 output scale/zero_point 可以解释的问题。
当前 native INT8 route 的数值语义与 PyTorch reference 尚未对齐。
```

新增 blocker:

```text
native_int8_s0_024_numeric_alignment_blocker.json
failure_type = numeric_alignment_blocker
failure_reason = PyTorch reference multiscale features and TVM uint8 outputs are not numerically aligned under minmax output dequant calibration; all three outputs failed sanity thresholds
```

## 4. 当前结论

已经解决:

```text
TVM native INT8 backbone build/run
AP-shape route build/run
TVM worker 跨进程 load/run/output
真实 HEAL activation capture
真实 activation -> TVM worker -> PyTorch head/postprocess
full AP gate runner 和 partial/empty blocker
reference-vs-TVM numeric sanity runner
```

未解决:

```text
TVM uint8 multiscale output 数值未对齐 PyTorch reference
当前 fixed requant_u8 rule 缺少 layer/output quantization evidence
INT8 AP measured row 仍为 0/60
```

因此下一步不是直接跑 full AP, 而是做 op-level numeric alignment。

## 5. 下一步计划

### 5.1 Op-level numeric alignment

目标: 找到 TVM route 与 PyTorch reference 的第一个数值发散点。

建议执行:

1. 用 `s0_024` 第一帧真实 `spatial_features` 和当前 checkpoint。
2. 对 PyTorch `pyramid_backbone.get_multiscale_feature` 加 hook, 记录:

```text
resnet/layer0/layer0.0/conv1 output
resnet/layer0/layer0.0/relu output
layer0 final output
layer1 final output
layer2 final output
```

3. 在 TVM route 中增加 debug output 或拆子图, 至少对第一层 Conv/ReLU/requant 输出落盘:

```text
tvm_first_conv_int32_or_requant_uint8.npy
tvm_first_relu_or_requant_uint8.npy
```

4. 对齐输入和权重:

```text
activation quant summary
ONNX initializer original float stats
runtime_weights_int8.npz stats
weight quant scale/zero_point
```

5. 生成:

```text
op_level_numeric_alignment_summary.json
op_level_numeric_alignment_blocker.json 或 op_level_numeric_alignment_pass.json
```

### 5.2 修正 route 数值策略

根据 op-level 结果决定:

1. 如果第一层就不对齐, 优先检查:

```text
activation quantization semantics
weight layout OIHW vs HWIO
group conv layout
requant_u8 固定除以 256 + 128 是否错误
```

2. 如果第一层对齐但后续发散, 检查:

```text
Add residual branch scale mismatch
Relu/requant order
per-layer output scale propagation
```

3. 修正后重跑:

```text
5-sample numeric sanity
20-sample AP smoke
1789-frame full AP gate
```

只有 numeric sanity 通过且 AP gate 满足条件, 才写:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

### 5.3 FP16 AP checkpoint recovery

继续:

```text
FP16 AP 5/60 -> 60/60
```

当前 blocker:

```text
raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

## 6. 验证

本地:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 37 tests in 7.066s
OK

python -m py_compile framework/stage2/native_int8_full_onnx.py framework/tests/test_stage2_native_int8_route.py scripts/stage2_h800_native_int8_real_activation_bridge.py scripts/stage2_native_int8_tvm_worker.py
exit 0
```

artifact gate:

```text
real_activation_bridge_report.schema = native_int8_output_dequant_numeric_sanity_report_v1
processed_samples = 5
output_dequant_calibration_summary.status = blocked
numeric_sanity_summary records = 15
rows/native_int8_original60_ap_rows_v1.jsonl does not exist
```

## 7. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 三指标收口, 单 agent 执行, 不启动 agent team。当前口径: INT8 backbone/subnet native route 已打通, full_network_claim=false; latency 是 H800 TVM backbone/subnet module 的端到端推理时间, 对外统一 latency_ms, 可作为 RSU edge-segment backbone/subnet server-side proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理设备实测。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; total AP measured 5/120, jobs_requiring_action=115。本轮新增: s0_024 5-sample output dequant numeric sanity 已完成, raw artifact 位于 raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/output_dequant_sanity5_v1/, processed_samples=5, records=15, output_dequant_calibration_summary.status=blocked; 三层 TVM uint8 output 与 PyTorch reference get_multiscale_feature 在 minmax output dequant 后仍严重不对齐, rmse_mean 接近 reference_range_mean, corrcoef_mean 接近 0 或为负; blocker 已写 native_int8_s0_024_numeric_alignment_blocker.json。注意 sanity mode stdout 的 AP 是 reference path AP, 不是 INT8 measured AP, 不允许写 rows/native_int8_original60_ap_rows_v1.jsonl。下一步硬目标: 做 s0_024 op-level numeric alignment, 用第一帧真实 spatial_features 和当前 checkpoint, hook PyTorch layer0/layer1/layer2 中间输出, 同时让 TVM route 暴露或拆出第一层 Conv/ReLU/requant debug output, 对齐 activation quant、ONNX initializer float stats、runtime_weights_int8.npz stats、weight quant scale/zero_point, 找到第一个数值发散点, 产出 op_level_numeric_alignment_summary.json 和 pass/blocker。优先检查 activation quant semantics、weight layout OIHW/HWIO、group conv layout、fixed requant_u8 除以 256 加 128、residual Add scale mismatch。只有 op-level 对齐修复后, 才重跑 5-sample numeric sanity、20-sample AP smoke、1789-frame full AP gate; 通过后才允许追加 rows/native_int8_original60_ap_rows_v1.jsonl。并行继续 FP16 AP checkpoint recovery: 当前 FP16 AP 5/60, 缺 55 行, blocker 在 raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/; 找到 checkpoint 即运行 true FP16 AP, 找不到写 per-label blocker 并继续其他 label。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 必须保存 blocker artifact/stdout/stderr, 做失败分类、最小复现、反思审查、runner/job queue 修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
