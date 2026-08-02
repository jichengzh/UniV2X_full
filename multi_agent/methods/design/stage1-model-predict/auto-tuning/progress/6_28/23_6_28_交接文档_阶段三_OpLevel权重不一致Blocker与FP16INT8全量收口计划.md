# 29_6_28_交接文档_阶段三_OpLevel权重不一致Blocker与FP16INT8全量收口计划

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `28_6_28_交接文档_阶段三_OutputDequantSanity失败与OpLevelNumericAlign计划.md`

本轮新增进展: 已完成 `s0_024` op-level numeric alignment probe。结果进一步收敛了 INT8 AP blocker: 第一层直接消费 `spatial_features` 的两个 Conv 输出已经与 PyTorch reference 不相关, 并且 `runtime_weights_int8.npz` 反量化权重与当前 AP checkpoint 的 PyTorch module weight 也不相关。因此当前问题不应继续描述为单纯 output dequant policy, 更具体的主 blocker 是 checkpoint/initializer/weight-binding 不一致或 route 权重来源不一致。

## 0. 对当前两个口径问题的确认

### 0.1 INT8 backbone 是否已经打通

是, 但只限于 backbone/subnet native INT8 route 的 build/run/latency/energy 口径:

```text
TVM native INT8 backbone build/run = 已打通
AP-shape route build/run = 已打通
TVM worker 加载 .so + runtime_weights_int8.npz = 已打通
真实 HEAL spatial_features -> TVM worker -> PyTorch head/postprocess = 已打通 smoke
native INT8 AP measured row = 未打通, 0/60
```

因此可以说: INT8 backbone 实现路径已经存在并能在 H800 上运行; 但不能说 native INT8 已经完成 AP 有效闭环。

### 0.2 latency 是否是 backbone 端到端推理速度

当前所有 original60 量化 latency/energy 收口口径是:

```text
latency unit = ms
latency scope = H800 TVM backbone/subnet module end-to-end
input boundary = spatial_features / backbone graph input
output boundary = multiscale backbone features / backbone graph outputs
full_network_claim = false
```

也就是说, 它是 backbone/subnet 图内从输入到输出的端到端推理时间, 可作为 RSU 边缘段 backbone/subnet server-side proxy。它不是完整感知 pipeline 的端到端延迟, 不包含 raw point cloud preprocessing、collaboration data movement、head/postprocess/NMS 等完整 `raw input -> boxes` 链路, 也不能声明为真实 RSU 物理设备实测。

## 1. 当前权威三指标状态

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
total latency = 120/120 measured
total energy = 120/120 measured
total AP70 = 5/120 measured
completion jobs requiring action = 115
```

当前不可写:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

该文件仍不存在, 这是正确状态。不能把 3-sample AP=0、numeric-sanity-only stdout 的 reference AP, 或任何 partial AP 写入 INT8 measured row。

## 2. 新增代码与验证 artifact

新增/修改:

```text
scripts/stage2_h800_native_int8_op_alignment.py
scripts/stage2_h800_native_int8_real_activation_bridge.py
framework/tests/test_stage2_native_int8_route.py
```

新增 helper:

```text
build_op_level_alignment_record(...)
```

新增 H800 artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/op_level_alignment_s0_024_v1/
```

关键文件:

```text
op_level_numeric_alignment_summary.json
op_level_alignment_records.json
op_level_weight_alignment_records.json
op_level_selected_ops.json
spatial_activation_alignment_inputs.json
native_int8_s0_024_op_level_numeric_alignment_blocker.json
```

## 3. Op-level probe 结果

H800 probe 只处理 `s0_024` 第一帧, hook PyTorch `pyramid_backbone.resnet` 中两个直接消费 `spatial_features` 的 Conv:

```text
/resnet/layer0/layer0.0/conv1/Conv -> resnet.layer0.0.conv1
/resnet/layer0/layer0.0/downsample/downsample.0/Conv -> resnet.layer0.0.downsample.0
```

summary:

```text
schema = native_int8_op_level_numeric_alignment_summary_v1
status = blocked
processed_samples = 1
selected_op_count = 2
record_count = 4
records_passed = 0
records_failed = 4
weight_record_count = 2
weight_records_passed = 0
weight_records_failed = 2
ap_measured = false
full_network_claim = false
```

### 3.1 首层 Conv output 对齐

候选输出使用当前 native INT8 route 语义:

```text
activation_uint8 + weight_int8 -> conv2d -> floor(conv / 256) + 128 -> clip uint8
```

然后把候选 uint8 minmax 对齐到 PyTorch reference range。即使消除了尺度范围差异, correlation 仍为负:

| op | candidate | rmse | corrcoef | reference_range | passed |
|---|---|---:|---:|---:|---|
| layer0.0.conv1 | raw_u8_minmax_aligned | 1.750815 | -0.052945 | 22.041054 | false |
| layer0.0.conv1 | centered_input_minmax_aligned | 1.750815 | -0.052945 | 22.041054 | false |
| layer0.0.downsample.0 | raw_u8_minmax_aligned | 2.524492 | -0.065477 | 30.676826 | false |
| layer0.0.downsample.0 | centered_input_minmax_aligned | 2.524492 | -0.065477 | 30.676826 | false |

补充观察:

```text
spatial_features min = 0.0
activation zero_point = 0
raw_u8 与 centered_input 结果相同
```

所以第一帧上 activation zero_point 不是主要解释。

### 3.2 权重对齐

将 `runtime_weights_int8.npz` 中的 int8 权重按记录 scale 反量化, 再 minmax 对齐到当前 AP checkpoint 的 PyTorch module weight range。结果也不相关:

| op weight | rmse | corrcoef | reference_range | passed |
|---|---:|---:|---:|---|
| layer0.0.conv1 weight | 0.275909 | 0.005799 | 1.287873 | false |
| layer0.0.downsample.0 weight | 0.398720 | 0.011277 | 1.767006 | false |

这比 output dequant blocker 更关键: 如果权重本身与 AP checkpoint 不一致, 后续无论如何调 output scale, 都不能得到可信 INT8 AP。

## 4. 当前根因判断

当前最高优先级假设:

```text
native INT8 AP-shape route 使用的 ONNX initializer / runtime_weights_int8.npz
与 HEAL AP eval 当前 checkpoint
${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26/net_epoch_bestval_at29.pth
不是同一组权重, 或 initializer-to-module binding 顺序/名称存在错绑。
```

已降低优先级的假设:

```text
仅仅缺少三层 output scale/zero_point
仅仅 activation zero_point 未减
仅仅 partial AP sample 太少
```

仍需排查但必须排在权重一致性之后:

```text
fixed requant_u8 的 /256 + 128 是否正确
OIHW/HWIO 或 group conv layout
Add residual 分支 scale mismatch
Relu/requant 顺序
per-layer output scale propagation
```

## 5. 下一阶段硬目标

最终收口目标保持非常具体:

```text
完成现有 original60 的 FP16 与 native INT8 两种量化方式三指标补点:
FP16 latency 60/60, energy 60/60, AP70 60/60
native INT8 latency 60/60, energy 60/60, AP70 60/60
最终总表具备 120 行 latency_ms、120 行 energy_J_per_inference、120 行 AP70 measured 数据
jobs_requiring_action = 0
```

当前 latency/energy 已经满足 120/120, 下一阶段实际工作集中在:

```text
FP16 AP70: 5/60 -> 60/60
native INT8 AP70: 0/60 -> 60/60
```

若某些 label 的 checkpoint 确认无法恢复, 必须有 per-label blocker、检索记录、失败分类和替代处理建议; 不能因为一个 label 缺失而停止其它 label。

## 6. 下一步执行计划

### 6.1 先修 INT8 checkpoint-consistent route

目标: 不再使用可能不同源的 benchmark ONNX/runtime weight 做 AP adapter。

执行顺序:

1. 从 `s0_024` AP checkpoint 重新导出或定位严格同源 backbone ONNX。
2. 重新生成 native INT8 AP-shape route 与 `runtime_weights_int8.npz`。
3. 跑 `stage2_h800_native_int8_op_alignment.py`。
4. 准入阈值:

```text
op_level_weight_alignment_records.json: selected spatial conv weight corrcoef >= 0.95
op_level_alignment_records.json: selected spatial conv output corrcoef >= 0.5 且 rmse <= 0.25 * reference_range
```

5. 如果权重仍不一致, 停止 AP gate, 继续修 export/initializer binding。

### 6.2 权重一致后再修 requant/dequant

通过权重一致性后, 才继续:

```text
activation quant semantics
weight scale 使用方式
fixed requant_u8 /256 + 128
per-layer output scale
residual Add scale
三层 multiscale output dequant
```

必须依次跑:

```text
1-frame op-level alignment
5-sample output numeric sanity
20-sample INT8 AP smoke
1789-frame s0_024 full AP gate
```

只有 full AP gate 通过, 才允许写首行:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

### 6.3 扩展到 original60 INT8 AP

当 `s0_024` native INT8 AP row 合规后:

1. 为 original60 每个 label 生成 checkpoint-consistent AP-shape native INT8 route。
2. 对每个 label 至少保留:

```text
route manifest
runtime_weights_int8.npz
op-level alignment summary
numeric sanity summary
full AP report
row source files
```

3. 批量写入 60 行 INT8 AP measured rows。
4. 每批刷新:

```text
original60_quant_three_metric_summary_latest.json/.md/.csv
fp16_int8_original60_completion_review_latest.json/.md
gap_report_latest
```

### 6.4 并行继续 FP16 AP recovery

当前 FP16 AP:

```text
5/60 measured
55/60 no_claim / missing checkpoint blocker
```

已有 blocker:

```text
raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

执行要求:

1. 不再盲搜已审计失败的默认路径。
2. 从 H800、备份、训练输出、归档路径恢复 per-label checkpoint/config。
3. 找到 checkpoint 即跑:

```text
scripts/stage2_h800_true_fp16_ap_eval.py --precision-mode amp_fp16
```

4. 找不到则写 per-label blocker 并继续其它 label。

## 7. 问题处理规则

下一阶段遇到任何问题不能直接停止。必须按以下顺序处理:

```text
1. 保存 stdout/stderr/traceback/raw artifact
2. 写 blocker JSON, 标明 full_network_claim=false, ap_measured=false
3. 失败分类: SSH / env / checkpoint / export / build / runtime / numeric / AP gate / row ingestion
4. 做最小复现
5. 审查是否是 runner bug、route bug、数据源 bug 或 checkpoint 缺失
6. 修复后重跑失败 label
7. 只有证明当前环境或缺失 checkpoint 确实不可由执行 agent 解决, 才允许将该 label 标记为 unresolved blocker
```

## 8. 验证记录

本地验证:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
Ran 40 tests
OK

python -m py_compile scripts/stage2_h800_native_int8_op_alignment.py scripts/stage2_h800_native_int8_real_activation_bridge.py framework/tests/test_stage2_native_int8_route.py
exit 0
```

H800 验证:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${V2X_ROOT}:${V2X_HOME}/heal_research/HEAL \
${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python scripts/stage2_h800_native_int8_op_alignment.py \
  --label s0_024 \
  --ckpt-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_s0_024_2026_06_26 \
  --raw-dir .../op_level_alignment_s0_024_v1 \
  --gpu-id 0
```

结果:

```text
status = blocked
records_failed = 4
weight_records_failed = 2
ap_measured = false
```

## 9. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 三指标收口, 单 agent 执行, 不启动 agent team。当前口径: latency 统一使用 latency_ms; 当前 latency/energy 是 H800 TVM backbone/subnet module 从 spatial_features/backbone graph input 到 multiscale backbone graph outputs 的图级端到端推理时间, full_network_claim=false, 可作为 RSU edge-segment backbone/subnet server-side proxy, 不能声明为完整 raw input->boxes 感知 pipeline 延迟或真实 RSU 物理设备实测。当前权威覆盖: FP16 latency 60/60 measured, FP16 energy 60/60 measured, FP16 AP70 5/60 measured; native INT8 latency 60/60 measured, native INT8 energy 60/60 measured, native INT8 AP70 0/60 measured; total latency 120/120, energy 120/120, AP70 5/120, jobs_requiring_action=115。INT8 backbone route 已打通 build/run/latency/energy, 但 INT8 AP 未闭环。本轮新增 s0_024 op-level numeric alignment artifact: raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/op_level_alignment_s0_024_v1/, status=blocked, selected_op_count=2, records_failed=4, weight_records_failed=2; 两个首层 Conv output 在 minmax 对齐后 corrcoef 为负, 且 runtime_weights_int8.npz 反量化权重与当前 AP checkpoint PyTorch module weight corrcoef 仅约 0.0058/0.0113。下一步最高优先级不是直接 full AP, 而是生成 checkpoint-consistent native INT8 AP-shape route: 从 s0_024 AP checkpoint 重新导出或定位严格同源 backbone ONNX, 重新生成 runtime_weights_int8.npz, 先跑 op-level weight/output alignment, 权重 corrcoef>=0.95 且首层输出通过后再跑 5-sample numeric sanity、20-sample AP smoke、1789-frame full AP gate; 只有 full AP gate 通过才允许创建/追加 rows/native_int8_original60_ap_rows_v1.jsonl。随后扩展到 original60 60 个 INT8 AP 点。并行继续 FP16 AP checkpoint recovery, 从当前 5/60 补到 60/60; 找到 checkpoint 即跑 true FP16 AP, 找不到写 per-label blocker 并继续其它 label。最终硬目标: 现有 original60 配置完成 FP16 与 native INT8 两种量化方式的 latency_ms、energy_J_per_inference、AP70 measured 全量补点, 即 FP16 60 行三指标 + native INT8 60 行三指标, jobs_requiring_action=0。遇到 SSH/env/checkpoint/export/build/runtime/numeric/AP gate/row ingestion 任何问题, 必须保存 artifact/stdout/stderr/blocker, 做反思、审查、最小复现、修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
