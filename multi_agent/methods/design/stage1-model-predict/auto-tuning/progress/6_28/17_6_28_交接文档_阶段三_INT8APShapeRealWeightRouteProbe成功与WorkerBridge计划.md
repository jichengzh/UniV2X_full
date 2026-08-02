# 23_6_28_交接文档_阶段三_INT8APShapeRealWeightRouteProbe成功与WorkerBridge计划

更新时间: 2026-06-28

本文件继承:

- `22_6_28_交接文档_阶段三_INT8APAdapter根因诊断与数值Route计划.md`

本轮新增进展: `s0_024` native INT8 AP-shape full-ONNX backbone route 已在 H800 build/run 成功, 并持久化真实 ONNX initializer 的 int8 量化权重 archive。仍然没有新增 INT8 AP measured row, 因为还没有接入 HEAL eval 的真实 activation 和 postprocess。

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

注意: 本轮新增 AP-shape probe 不是 official original60 latency/energy measured row, 也不是 AP measured row。它是 INT8 AP adapter 的 build/run feasibility evidence。

## 1. 本轮新增代码能力

新增/修改:

```text
framework/stage2/native_int8_full_onnx.py
framework/tests/test_stage2_native_int8_route.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
```

新增 helper:

```text
parse_input_shape_overrides(["spatial_features=1,64,256,256"])
build_runtime_arg_plan(...)
```

route script 新增能力:

```text
--input-shape-override spatial_features=1,64,256,256
```

并在 raw artifact 中写入:

```text
input_shape_overrides
runtime_input_sources
runtime_arg_plan
runtime_weight_archive_path
runtime_weight_archive_digest
runtime_weight_archive_keys
```

实现边界:

1. graph input activation 目前仍是 synthetic random uint8 probe input。
2. Conv weight inputs 已从 ONNX initializer 做 per-tensor symmetric absmax int8 quantization。
3. weight archive 已持久化, 可供下一阶段 TVM worker 复用。

## 2. H800 AP-shape real-weight probe 结果

run:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/
```

关键产物:

```text
full_onnx_route_probe.json
s0_024/tvm_operator_inventory.json
s0_024/native_int8_route_manifest.json
s0_024/runtime_weights_int8.npz
s0_024/s0_024_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
s0_024/latency_result.json
s0_024/energy_result.json
```

H800 run result:

```text
status = success
label = s0_024
input_shape_override = spatial_features [1,64,256,256]
latency_ms = 9.343423
energy_J = 2.3741115398678865
op_counts = Conv 51, Relu 48, Add 16, Identity 46
full_network_claim = false
```

inventory 关键字段:

```text
input_shape = {"spatial_features": [1,64,256,256]}
output_shape =
  /resnet/layer0/layer0.2/relu_2/Relu_output_0 [1,24,256,256]
  /resnet/layer1/layer1.4/relu_2/Relu_output_0 [1,128,128,128]
  /resnet/layer2/layer2.7/relu_2/Relu_output_0 [1,256,64,64]
runtime_input_sources =
  synthetic_random_uint8_activation: 1
  onnx_initializer_quantized_int8: 51
runtime_arg_plan roles =
  graph_input: 1
  weight_input: 51
  graph_output: 3
runtime_weights_int8.npz keys = 51
```

这说明上一轮 blocker 已缩小:

```text
已解决: AP eval shape 的 native INT8 route 可以 build/run
已解决: route 可以使用真实 ONNX initializer 的 int8 quantized weight inputs
未解决: HEAL torch activation -> uint8 quantization -> TVM worker -> feature dequant/bridge -> PyTorch head/postprocess
```

## 3. 仍不能写 INT8 AP measured row 的原因

当前仍不能写 `rows/native_int8_original60_ap_rows_v1.jsonl`, 因为:

1. 本轮 probe 的 graph activation input 还是 synthetic random uint8, 不是 HEAL `aligner_m1` 真实输出。
2. 还没有定义 activation quantization scale/zero point, 也没有记录 calibration evidence。
3. TVM route 输出是 uint8 multiscale feature, 还没有 dequantize 回 PyTorch head 可消费的 tensor。
4. HEAL eval 环境与 TVM runtime 仍是两个 Python 环境:
   - UniV2X 有 torch/opencood, 无 TVM。
   - tvm310 有 TVM, 无 torch。
5. 还没有实现 persistent TVM worker 或等价 bridge, 所以不能在 1789-frame AP eval 中高效调用该 route。

## 4. 下一步具体实现计划

### 4.1 TVM worker bridge

先实现 `s0_024` one-sample worker, 不直接 full AP:

```text
HEAL UniV2X process:
  1. run encoder_m1/backbone_m1/aligner_m1
  2. save aligned spatial_features float32 npy
  3. write activation quant params
  4. call persistent or one-shot tvm310 worker

TVM worker:
  1. load s0_024_native_int8_full_onnx...so
  2. load runtime_weights_int8.npz
  3. quantize/load activation uint8
  4. run TVM main
  5. save 3 multiscale outputs + dequant metadata

HEAL UniV2X process:
  1. load/dequant multiscale outputs
  2. pass through existing weighted_fuse/decode_multiscale_feature/shrink_conv/heads
  3. run dataset.post_process
```

### 4.2 one-sample acceptance

one-sample smoke must produce:

```text
native_int8_worker_request.json
native_int8_worker_response.json
activation_quant_summary.json
multiscale_output_summary.json
head_output_summary.json
postprocess_summary.json
```

Required shapes:

```text
input = [1,64,256,256]
multiscale outputs =
  [1,24,256,256]
  [1,128,128,128]
  [1,256,64,64]
head outputs =
  cls_preds [1,2,256,256]
  reg_preds [1,14,256,256]
  dir_preds [1,4,256,256]
```

如果 one-sample postprocess 成功, 仍只写 smoke report, 不写 measured AP row。

### 4.3 full AP gate

只有以下条件满足后才追加 measured AP row:

```text
num_samples = 1789
AP30/AP50/AP70 present
raw_artifact includes worker request/response or summary
runtime_weight_archive_digest recorded
activation quantization evidence recorded
full_network_claim = false
quality_gate_status = native_int8_full_onnx_original60_ap_eval
```

## 5. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 INT8/FP16 AP 补点收口, 单 agent 执行, 不启动 agent team。当前权威状态: FP16/INT8 latency=120/120 measured, energy=120/120 measured, AP=5/120 measured, jobs_requiring_action=115; latency 是 H800 TVM backbone/subnet module latency, 对外统一 ms, full_network_claim=false。本轮已完成 s0_024 native INT8 AP-shape real-weight route probe: route 支持 --input-shape-override spatial_features=1,64,256,256, H800 run 20260628_native_int8_apshape_s0_024_realweight_probe_v2 成功, output shape 与 HEAL HeterPyramidCollab AP 边界一致, runtime_weights_int8.npz 已持久化 51 个 ONNX initializer quantized int8 weights, runtime_arg_plan 已记录 graph_input/weight_input/graph_output。下一步不要再重复证明 backbone buildability, 直接实现 s0_024 TVM worker bridge: UniV2X torch 进程保存 aligner_m1 后的真实 activation, tvm310 worker 加载 .so + runtime_weights_int8.npz 运行 native INT8 route, 输出三层 multiscale feature, 再回到 PyTorch weighted_fuse/decode/shrink/head/postprocess。先做 one-sample smoke, 产出 worker_request/response、activation_quant_summary、multiscale_output_summary、head_output_summary、postprocess_summary; one-sample postprocess 成功后再跑 s0_024 1789-frame full AP, 成功才追加 rows/native_int8_original60_ap_rows_v1.jsonl 首行 measured。并行继续 FP16 AP checkpoint recovery, 当前 55 个 missing checkpoint blocker 已在 raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/。每批后刷新 original60_quant_three_metric_summary_latest、fp16_int8_original60_completion_review_latest、gap_report_latest。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 必须保存 blocker artifact/stdout/stderr, 再反思、审查、最小复现、修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
