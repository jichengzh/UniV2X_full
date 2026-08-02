# 26_6_28_交接文档_阶段三_RealActivationBridge成功与INT8FullAPGate计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `25_6_28_交接文档_阶段三_TVMWorkerSyntheticSmoke成功与RealActivationBridge计划.md`

本轮新增进展: `s0_024` native INT8 route 已从 synthetic activation 推进到真实 HEAL `spatial_features` one-sample bridge, 并成功接回 PyTorch head/postprocess。仍不能写 INT8 AP measured row, 因为当前只处理 1 个样本, 且输出 dequant policy 仍是 smoke 级临时策略。

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

新增 helper:

```text
quantize_activation_uint8(...)
summarize_numpy_array(...)
```

新增 bridge script:

```text
scripts/stage2_h800_native_int8_real_activation_bridge.py
```

bridge 路线:

```text
HEAL UniV2X process
  -> patch model.pyramid_backbone.get_multiscale_feature
  -> capture real spatial_features [1,64,256,256]
  -> asymmetric_minmax_uint8 quantization
  -> call tvm310 worker
  -> load 3 uint8 multiscale outputs
  -> temporary dequant bridge to PyTorch tensors
  -> PyTorch decode/shrink/head
  -> dataset.post_process
```

## 2. H800 real activation one-sample bridge 结果

raw artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/real_activation_bridge_s0_024_v1/
```

关键文件:

```text
real_activation_bridge_report.json
activation_quant_summary.json
multiscale_output_summary.json
worker_response_summary.json
head_output_summary.json
postprocess_summary.json
bridge_call_000/activation_float32.npy
bridge_call_000/agent_000/activation_float32.npy
bridge_call_000/agent_000/activation_uint8.npy
bridge_call_000/agent_000/activation_quant_summary.json
bridge_call_000/agent_000/native_int8_worker_request.json
bridge_call_000/agent_000/native_int8_worker_response.json
bridge_call_000/agent_000/out_resnet_layer0_layer0_2_relu_2_Relu_output_0.npy
bridge_call_000/agent_000/out_resnet_layer1_layer1_4_relu_2_Relu_output_0.npy
bridge_call_000/agent_000/out_resnet_layer2_layer2_7_relu_2_Relu_output_0.npy
```

report:

```text
status = success
processed_samples = 1
ap_measured = false
resume_epoch = 29
ckpt = ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26/net_epoch_bestval_at29.pth
```

activation quant:

```text
scheme = asymmetric_minmax_uint8
shape = [1,64,256,256]
scale = 0.053168685763489966
zero_point = 0
```

worker outputs:

| tensor | dtype | shape |
|---|---|---|
| `/resnet/layer0/layer0.2/relu_2/Relu_output_0` | uint8 | [1,24,256,256] |
| `/resnet/layer1/layer1.4/relu_2/Relu_output_0` | uint8 | [1,128,128,128] |
| `/resnet/layer2/layer2.7/relu_2/Relu_output_0` | uint8 | [1,256,64,64] |

head outputs:

```text
cls_preds = [1,2,256,256]
reg_preds = [1,14,256,256]
dir_preds = [1,4,256,256]
occ_single_list = list length 3
```

postprocess:

```text
status = success
gt_box_tensor = [20,8,3]
pred_box_tensor = None
pred_score = None
```

解释: postprocess 成功但该单样本没有预测框输出。由于这是 one-sample smoke 且 dequant policy 仍为临时策略, 不能据此生成 AP measured row。

## 3. 本轮验证

本地:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 34 tests in 5.890s
OK

python -m py_compile framework/stage2/native_int8_full_onnx.py framework/tests/test_stage2_native_int8_route.py scripts/stage2_h800_native_int8_real_activation_bridge.py scripts/stage2_native_int8_tvm_worker.py
exit 0
```

artifact gate:

```text
real_activation_bridge_report.status = success
processed_samples = 1
activation shape = [1,64,256,256]
worker output shapes = [1,24,256,256], [1,128,128,128], [1,256,64,64]
postprocess_summary.status = success
rows/native_int8_original60_ap_rows_v1.jsonl = still missing
```

## 4. 当前剩余 blocker

### 4.1 INT8 AP measured row 尚未生成

当前已经解决:

```text
native INT8 backbone build/run
AP-shape route build/run
worker process load/run/output
real HEAL activation capture
real activation -> TVM worker -> PyTorch head/postprocess one-sample route
```

仍未解决:

```text
1789-frame full AP eval 尚未跑
输出 dequant policy 仍是 temporary activation minmax scale, 不是严谨 output quantization metadata
当前 one-sample pred_box_tensor=None, 需要 full eval 前审查数值有效性
```

下一步不能直接把 one-sample smoke 导入 AP 表。必须先做 `s0_024` full AP gate 或写明确 blocker。

### 4.2 FP16 AP 仍缺 55 行

当前 FP16 AP:

```text
5/60 measured
55/60 no_claim / missing checkpoint blocker
```

blocker 目录:

```text
raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

继续要求: 找到 checkpoint 的 label 立即跑 true FP16 AP; 找不到的 label 写 per-label blocker 并继续其他 label。

## 5. 下一步计划

### 5.1 INT8 s0_024 full AP gate

目标:

```text
rows/native_int8_original60_ap_rows_v1.jsonl 首行 measured, 或落 full AP blocker
```

执行:

1. 将 `scripts/stage2_h800_native_int8_real_activation_bridge.py` 扩展为 `--num-samples` full-eval mode。
2. 对每帧复用当前 bridge, 累计 `result_stat`。
3. 写 raw:

```text
full_ap_eval_report.json
activation_quant_summary_stream.jsonl
worker_response_summary_stream.jsonl
postprocess_summary_stream.jsonl
```

4. 若 AP eval 成功且 `num_samples=1789`, 再追加:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

5. 若出现全空预测或数值异常, 不写 measured row, 写:

```text
native_int8_s0_024_full_ap_blocker.json
```

blocker 必须包含:

```text
failure_type
failure_reason
sample_count
pred_nonempty_count
activation quant summary
worker response summary
head output range summary
postprocess summary
```

### 5.2 INT8 output dequant policy 审查

当前 one-sample smoke 使用:

```text
temporary activation minmax scale applied to uint8 TVM outputs
```

这是 shape/postprocess smoke 可接受, 但 AP measured 前必须审查:

1. 是否需要为每个 TVM output 独立记录 output scale/zero_point。
2. 是否可以从 PyTorch reference multiscale feature 对齐输出范围。
3. 是否需要先做 5-sample 数值 sanity, 比较 PyTorch get_multiscale_feature vs TVM bridge output 的 shape/range/finite ratio。
4. 若无法建立合理 dequant, full AP 可以运行但不能 claim 为数值有效 measured AP; 必须写 blocker。

### 5.3 FP16 AP checkpoint recovery

继续:

```text
FP16 AP 5/60 -> 60/60
```

## 6. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 三指标收口, 单 agent 执行, 不启动 agent team。当前口径: INT8 backbone/subnet native route 已打通, full_network_claim=false; latency 是 H800 TVM backbone/subnet module 的端到端推理时间, 对外统一 latency_ms, 可作为 RSU edge-segment backbone/subnet server-side proxy, 不能声明为完整感知 pipeline 或真实 RSU 物理设备实测。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; total AP measured 5/120, jobs_requiring_action=115。本轮新增: s0_024 real activation one-sample bridge 已在 H800 成功, raw artifact 位于 raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/real_activation_bridge_s0_024_v1/; 真实 HEAL spatial_features [1,64,256,256] 已捕获并做 asymmetric_minmax_uint8 quantization, TVM worker 成功输出 3 个 uint8 multiscale feature [1,24,256,256]、[1,128,128,128]、[1,256,64,64], PyTorch head 输出 cls/reg/dir shape 正确, dataset.post_process status=success。注意该 one-sample smoke 不是 AP measured row, 不允许追加 rows/native_int8_original60_ap_rows_v1.jsonl。下一步硬目标: 扩展 scripts/stage2_h800_native_int8_real_activation_bridge.py 为 s0_024 full AP gate, 支持 --num-samples, 累计 result_stat, 写 full_ap_eval_report.json 和 stream summaries; 若 num_samples=1789 且 AP30/AP50/AP70 成功, 才追加 rows/native_int8_original60_ap_rows_v1.jsonl 首行 measured; 若全空预测、数值异常或 output dequant policy 不足以支撑 measured claim, 必须写 native_int8_s0_024_full_ap_blocker.json, 包含 failure_type/failure_reason/sample_count/pred_nonempty_count/activation quant/worker/head/postprocess summaries。并行继续 FP16 AP checkpoint recovery: 当前 FP16 AP 5/60, 缺 55 行, blocker 在 raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/; 找到 checkpoint 即运行 true FP16 AP, 找不到写 per-label blocker 并继续其他 label。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 必须保存 blocker artifact/stdout/stderr, 做失败分类、最小复现、反思审查、runner/job queue 修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
