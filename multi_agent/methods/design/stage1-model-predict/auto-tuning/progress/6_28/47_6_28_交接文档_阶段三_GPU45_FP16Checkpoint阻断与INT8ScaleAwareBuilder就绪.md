# 53_6_28_交接文档_阶段三_GPU45_FP16Checkpoint阻断与INT8ScaleAwareBuilder就绪

## 0. 本轮结论

1. GPU4/GPU5 当前空闲, 但 FP16 AP 55 个 no_claim label 没有可直接 full eval 的 checkpoint, 所以不能启动合法的 FP16 AP 大规模补点。
2. 现有 missing-checkpoint blocker 已落盘 55 个 label, 位置:
   `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/`
3. 早前 INT8 s0_024 full AP v2 占用过 GPU0; 后续执行前必须重新审计 GPU0/GPU4/GPU5 和 runner artifact, 不用旧快照直接判断资源状态。
4. 已完成 INT8 scale-aware route builder 代码准备: `stage2_h800_native_int8_full_onnx_route.py` 支持 `--tensor-quant-params-path`, Conv/Add/ReLU 可使用 tensor_quant_params_v2 进行 scale-aware requant。
5. bridge 已支持 calibration-static input quant: 当 tensor_quant_params 中存在 `spatial_features` 时, 不再使用 per-sample minmax, 而是使用 calibration scale/zero_point。
6. 新 route builder、bridge 和 framework helper 已同步到 H800, 远端 py_compile 通过。

## 1. 当前覆盖状态

权威 completion review:

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
completion review: latency = 120/120, energy = 120/120, AP = 5/120, jobs_requiring_action = 115
```

latency_ms 口径保持不变: H800 TVM compiled backbone/subnet module end-to-end, input=spatial_features, output=multiscale backbone/subnet outputs, full_network_claim=false。不能声明为完整 perception pipeline latency 或真实 RSU 物理设备绝对 latency。

## 2. GPU 状态快照与新调度约束

2026-06-28T10:40:55+08:00 历史监控:

```text
GPU0: s0_024 native INT8 full AP v2 still running, python pid=3810196, runner pid=3810190, elapsed=02:21:44
GPU4: free
GPU5: free
s0_024 sample blocker count = 0
s0_024 full_ap_eval_report.json = absent
```

该快照只作为历史记录。新的调度约束是:

```text
GPU4/GPU5: 用于 FP16 AP 补点与 checkpoint recovery 后的分片队列。
GPU0: 用于 native INT8 精度崩溃修复; 启动前重新确认是否空闲, 若仍有有效旧任务则等待或记录阻塞, 不误杀。
所有 lane: 启动前保存 nvidia-smi/ps/runner artifact 审计结果。
```

## 3. FP16 AP lane

当前 FP16 AP measured labels:

```text
s0_024, s0_040, s0_056, s1_048, s2_160
```

FP16 no_claim labels = 55。默认 checkpoint root 与扩展 inventory 仍未找到这些 label 的 `Pyramid_DAIR_m1_stage2_ap_<label>` checkpoint。

已审计的可见 stage1 checkpoint 宽度包含:

```text
[48,128,256], [64,96,256], [64,128,192], [48,64,256], [48,128,128],
[64,128,256], [32,64,136], [48,96,192], [32,64,128], [16,32,64],
[24,128,256], [40,128,256], [56,128,256], [64,48,256], [64,128,160]
```

这些不是 55 个 no_claim label 的可直接 exact-label AP checkpoint。不能用不匹配 checkpoint 写 original60 FP16 AP measured row。

下一步:

1. 继续 checkpoint recovery, 搜索备份/归档/人工提供路径。
2. 找到 exact-label checkpoint 后, 立即在 GPU4/GPU5 跑:
   `scripts/stage2_h800_true_fp16_ap_eval.py --execute --precision-mode amp_fp16`
3. 每个成功 label 才能追加 `rows/fp16_true_original60_ap_rows_v1.jsonl`。
4. 未恢复 checkpoint 的 label 保持 per-label blocker, 不阻塞 INT8 lane。

## 4. INT8 lane

### 4.1 已完成代码准备

修改并同步到 H800:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
scripts/stage2_h800_native_int8_real_activation_bridge.py
framework/stage2/native_int8_full_onnx.py
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

测试:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route
Ran 50 tests OK

H800:
py_compile stage2_h800_native_int8_full_onnx_route.py OK
py_compile native_int8_full_onnx.py and real_activation_bridge.py OK
static input quant smoke: [-1,0,1] -> [126,128,130] at scale=0.5, zero_point=128
```

### 4.2 s0_040 calibration

s0_040 已有 tensor_quant_params_v2:

```text
${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_040_native_int8_route_centered_conv_v1/s0_040/reference_range_capture_h800_pyramid_level2_v2_cuda_visible_all/tensor_quant_params_calibration_v2_to_pyramid_level2.json
```

关键输入:

```text
spatial_features scale = 0.049095408121744795
spatial_features zero_point = 0
param_count = 116
pyramid_level0/1/2 also have scale/zero_point
```

### 4.3 仍未完成的 gate

还没有跑:

```text
s0_040 scale-aware route rebuild
s0_040 numeric_sanity
s0_040 5-sample AP smoke
s0_040 full AP
```

执行约束: 不再假设旧 GPU0 状态仍然有效。启动前重新审计 GPU0; 若空闲则直接跑 s0_040 gate, 若仍被有效任务占用则等待或记录资源阻塞, 不误杀。

下一步在 GPU0 可用后立即执行:

```text
1. 用 --tensor-quant-params-path 重建 s0_040 scale-aware route。
2. 跑 numeric_sanity_only, 对比 pyramid_level0/1/2 corrcoef/RMSE/动态范围。
3. 若 corrcoef 和动态范围显著改善, 跑 5-sample AP smoke。
4. 只有 pred_nonempty_count > 0 后, 才允许 1789-frame full AP。
5. 若 numeric 仍差, 下一处修复应转向 TE lowering/rounding 或 runtime scale 参数; bridge input quantization 已先改为 calibration-static graph_input_quant。
```

## 5. 修订后的 /goal

```text
/goal 继续 Stage2 original60 FP16/native INT8 AP 全量收口, 单 agent 执行, 允许并行启动两条互不抢资源的 H800 lane。启动前先做一次 nvidia-smi/ps/runner artifact 审计, 记录 GPU0/GPU4/GPU5 是否空闲、已有 pid、对应 run_id 和是否可等待; 不误杀已有任务。Lane A 是 FP16 AP 补点, 绑定 H800 GPU4/GPU5, 目标是把 rows/fp16_true_original60_ap_rows_v1.jsonl 从当前 FP16 AP 5/60 推进到 60/60: 先复查 fp16_missing_checkpoint_blockers_v1 和 checkpoint inventory, 对已有 exact-label checkpoint 的 label 立即生成 GPU4/GPU5 分片队列并启动 scripts/stage2_h800_true_fp16_ap_eval.py --execute --precision-mode amp_fp16; 对仍缺 exact-label checkpoint 的 label 并行做 checkpoint recovery, 找到后动态补入队列。禁止用不匹配 checkpoint、旧 smoke AP 或非 full eval 结果写 measured row; 每个成功 label 必须保存 command、stdout/stderr、runner pid、GPU id、raw report, 追加 AP row 后刷新 coverage/review。Lane B 是 native INT8 精度崩溃修复, 绑定 H800 GPU0, 目标是把 INT8 AP 崩溃从 QDQ-heavy/float32-heavy 或 scale mismatch 根因推进到可解释、可复现、可补点的 native INT8 AP 路线: 使用已同步的 scale-aware route builder 和 bridge, 用 s0_040 tensor_quant_params_calibration_v2_to_pyramid_level2.json 重建 scale-aware route, 依次跑 numeric_sanity -> 5-sample AP smoke -> full AP。numeric_sanity 必须检查 pyramid_level0/1/2 corrcoef、RMSE、动态范围、非零比例和 dequant scheme; 只有 corrcoef/动态范围相较旧 route 明显改善且 5-sample smoke pred_nonempty_count>0, 才允许启动 1789-frame full AP。若 numeric 仍失败, 不停止: 定位 Conv/Add/ReLU requant、round/clip、zero_point、graph_input static quant、TE lowering 和 runtime scale 参数, 最小复现后修复脚本并重跑 numeric gate。当前权威覆盖写入交接状态: FP16 latency 60/60, FP16 energy 60/60, FP16 AP 5/60; native INT8 latency 60/60, native INT8 energy 60/60, native INT8 AP 0/60 或按最新 import 后覆盖刷新; completion review AP 当前约 5/120, jobs_requiring_action 约 115。latency_ms 口径固定为 H800 TVM compiled backbone/subnet module end-to-end, input=spatial_features, output=multiscale backbone/subnet outputs, full_network_claim=false。所有 latency 指标统一使用 ms。遇到 build/eval/import/SSH/checkpoint/adapter/gate 问题, 保存 raw artifact、command、stdout/stderr、runner pid、GPU id、failure reason, 做根因判断、反思审查、脚本或队列修复、smoke 和失败 label 补跑; 只有证明当前环境或 checkpoint 缺失无法由执行 agent 解决时才写 per-label blocker 并继续其他 label。
```
