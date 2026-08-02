# 25_6_28_交接文档_阶段三_TVMWorkerSyntheticSmoke成功与RealActivationBridge计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `24_6_28_交接文档_阶段三_INT8Backbone口径确认与FP16INT8_AP收口计划.md`

本轮新增进展: 已实现并验证 `s0_024` native INT8 TVM worker 的跨进程运行路径。该 smoke 使用 synthetic uint8 activation, 因此仍不能写 INT8 AP measured row; 但它证明了下一步 HEAL 真实 activation bridge 所需的核心 worker 能力已经具备。

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
scripts/stage2_native_int8_tvm_worker.py
```

新增 helper:

```text
build_tvm_worker_request(...)
validate_tvm_worker_response(...)
```

新增 worker script:

```text
scripts/stage2_native_int8_tvm_worker.py
```

worker 输入:

```text
native_int8_worker_request.json
artifact_path = TVM exported .so
runtime_weight_archive_path = runtime_weights_int8.npz
activation_npy_path = quantized uint8 activation
runtime_arg_plan = graph_input + weight_input + graph_output argument order
expected_output_shapes = expected graph output tensor shapes
```

worker 输出:

```text
native_int8_worker_response.json
out_resnet_layer0_layer0_2_relu_2_Relu_output_0.npy
out_resnet_layer1_layer1_4_relu_2_Relu_output_0.npy
out_resnet_layer2_layer2_7_relu_2_Relu_output_0.npy
```

实现细节:

1. TVM import 只发生在 `run_worker()`, 本地 unit test 不依赖 TVM。
2. worker 按 `runtime_arg_plan` 顺序构造 TVM args。
3. graph input 使用 `activation_uint8.npy`。
4. weight input 使用 `runtime_weights_int8.npz` 里的 quantized int8 ONNX initializer。
5. graph output 使用 preallocated uint8 TVM tensor, run 后保存为 `.npy`。
6. 兼容 tvm310 中 `tvm.nd` 不存在的情况, fallback 到 `tvm.runtime.tensor(value, device=dev)`。

## 2. H800 synthetic worker smoke 结果

raw artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/worker_smoke_synthetic_v1/
```

关键文件:

```text
activation_uint8.npy
native_int8_worker_request.json
native_int8_worker_response.json
out_resnet_layer0_layer0_2_relu_2_Relu_output_0.npy
out_resnet_layer1_layer1_4_relu_2_Relu_output_0.npy
out_resnet_layer2_layer2_7_relu_2_Relu_output_0.npy
```

response:

```text
status = success
output_count = 3
```

outputs:

| tensor | dtype | shape |
|---|---|---|
| `/resnet/layer0/layer0.2/relu_2/Relu_output_0` | uint8 | [1,24,256,256] |
| `/resnet/layer1/layer1.4/relu_2/Relu_output_0` | uint8 | [1,128,128,128] |
| `/resnet/layer2/layer2.7/relu_2/Relu_output_0` | uint8 | [1,256,64,64] |

这说明已解决:

```text
TVM .so 可以在独立 tvm310 worker 进程 load_module
runtime_weights_int8.npz 可以按 runtime_arg_plan 注入
worker 可以输出三层 multiscale uint8 feature .npy
worker request/response contract 已可本地单元测试
```

仍未解决:

```text
activation 仍是 synthetic uint8, 不是 HEAL aligner_m1 真实 spatial_features
activation quantization scale/zero_point 尚未从真实 activation 记录
TVM 输出尚未 dequantize/bridge 回 PyTorch head
尚未产出 head_output_summary/postprocess_summary
尚未跑 1789-frame INT8 AP full eval
```

因此本轮 smoke 不是 AP measured row, 不能追加:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

## 3. 本轮验证

本地:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route
Ran 19 tests in 0.131s
OK

python -m py_compile scripts/stage2_native_int8_tvm_worker.py framework/tests/test_stage2_native_int8_route.py
exit 0
```

H800:

```text
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=${V2X_ROOT}
LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib
${V2X_DATA_ROOT}/tvm310/bin/python scripts/stage2_native_int8_tvm_worker.py --request-json .../worker_smoke_synthetic_v1/native_int8_worker_request.json
```

H800 首次失败:

```text
failure_reason = module 'tvm' has no attribute 'nd'
```

修复:

```text
make_tvm_array() fallback to tvm.runtime.tensor(value, device=dev)
```

复跑结果:

```text
status = success
output_count = 3
```

## 4. 下一步计划

### 4.1 INT8 real activation one-sample bridge

目标: 用真实 HEAL `spatial_features` 替代 synthetic activation。

执行:

1. 在 UniV2X/HEAL AP eval 进程内定位 `HeterPyramidCollab.forward_collab` 的 `spatial_features` 边界。
2. 捕获单帧真实 `spatial_features` tensor, shape 应为 `[1,64,256,256]`。
3. 做 uint8 quantization, 写出:

```text
activation_float32.npy
activation_uint8.npy
activation_quant_summary.json
```

4. 调用 `scripts/stage2_native_int8_tvm_worker.py`。
5. 读取 3 个 uint8 multiscale output, 写:

```text
multiscale_output_summary.json
```

6. 将输出 dequantize/bridge 为 PyTorch tensor, 接入 weighted_fuse/decode/shrink/head。
7. 写:

```text
head_output_summary.json
postprocess_summary.json
```

通过标准:

```text
one-sample postprocess 成功, 或落精确 blocker
```

### 4.2 INT8 s0_024 full AP gate

one-sample bridge 通过后, 才运行:

```text
s0_024 1789-frame full AP eval
```

成功后追加首行:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

AP row 必须包含:

```text
AP30/AP50/AP70
num_samples
native_int8_route_manifest
runtime_weight_archive_digest
activation quantization evidence
worker request/response summary
raw_artifact
full_network_claim=false
```

### 4.3 FP16 AP checkpoint recovery

并行继续:

```text
FP16 AP 5/60 -> 60/60
```

当前 blocker 目录:

```text
raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

不要因为 missing checkpoint 停止整批。找到 checkpoint 的 label 立即跑 true FP16 AP; 找不到的 label 写 per-label blocker, 继续其他 label。

## 5. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 三指标收口, 单 agent 执行, 不启动 agent team。当前口径: INT8 backbone/subnet native route 已打通, full_network_claim=false; latency 是 H800 TVM backbone/subnet module 的端到端推理时间, 对外统一 latency_ms, 可作为 RSU edge-segment backbone/subnet server-side proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理设备实测。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; total AP measured 5/120, jobs_requiring_action=115。本轮新增: scripts/stage2_native_int8_tvm_worker.py 已实现, s0_024 synthetic uint8 activation worker smoke 已在 H800 成功, raw artifact 位于 raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/worker_smoke_synthetic_v1/, worker 成功加载 TVM .so + runtime_weights_int8.npz 并输出 3 个 uint8 multiscale feature, shape 分别为 [1,24,256,256]、[1,128,128,128]、[1,256,64,64]。注意该 smoke 使用 synthetic activation, 不是 AP measured row, 不允许追加 rows/native_int8_original60_ap_rows_v1.jsonl。下一步硬目标: 实现 s0_024 real activation one-sample bridge, 在 UniV2X/HEAL AP eval 进程捕获真实 spatial_features [1,64,256,256], 写 activation_float32.npy、activation_uint8.npy、activation_quant_summary.json, 调用 tvm310 worker, 读取三层 multiscale output, dequantize/bridge 回 PyTorch weighted_fuse/decode/shrink/head/postprocess, 产出 multiscale_output_summary.json、head_output_summary.json、postprocess_summary.json; one-sample postprocess 成功后跑 s0_024 1789-frame full AP, 成功才追加 rows/native_int8_original60_ap_rows_v1.jsonl 首行 measured, 再扩展到 original60 全量 INT8 AP 60/60。并行继续 FP16 AP checkpoint recovery: 当前 FP16 AP 5/60, 缺 55 行, blocker 在 raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/; 找到 checkpoint 即运行 true FP16 AP, 找不到写 per-label blocker 并继续其他 label。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 必须保存 blocker artifact/stdout/stderr, 做失败分类、最小复现、反思审查、runner/job queue 修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
