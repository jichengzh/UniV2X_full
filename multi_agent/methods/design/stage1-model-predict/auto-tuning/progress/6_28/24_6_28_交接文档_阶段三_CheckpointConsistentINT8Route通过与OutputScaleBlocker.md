# 30_6_28_交接文档_阶段三_CheckpointConsistentINT8Route通过与OutputScaleBlocker

更新时间: 2026-06-28

本文件是当前最新交接入口, 继承:

- `29_6_28_交接文档_阶段三_OpLevel权重不一致Blocker与FP16INT8全量收口计划.md`

本轮新增进展: `s0_024` 的 native INT8 AP route 已从“权重不一致 blocker”推进到“checkpoint-consistent route build/run + 首层 op-level alignment 通过”。旧 route 与 AP checkpoint 权重不一致的问题已经被实验证明并绕开; 新 route 的 Conv+BN fused weight audit 51/51 通过, TVM native INT8 artifact 成功产出并完成 latency/energy smoke。当前主要 blocker 已收敛为后续层输出尺度传播问题, 尤其是 `layer0/layer1` 内部的 requant、residual Add 和 multiscale output dequant。

## 0. 先回答当前两个口径问题

### 0.1 INT8 backbone 实现是否已经打通

可以说已经打通, 但必须带上边界:

```text
native INT8 backbone/subnet route build = 已打通
native INT8 backbone/subnet route run = 已打通
original60 native INT8 latency rows = 60/60 measured
original60 native INT8 energy rows = 60/60 measured
checkpoint-consistent s0_024 AP-shape route = 已 build/run
checkpoint-consistent route first-op alignment = passed
native INT8 AP measured rows = 0/60 measured
```

因此准确表述是:

```text
INT8 backbone/subnet 的 TVM native INT8 实现路径已经存在并能运行;
但 native INT8 AP 有效闭环还没有完成, 不能把当前结果写成 INT8 AP measured。
```

### 0.2 当前 latency 是否都是 backbone 端到端推理速度

是, 但不是完整感知网络端到端, 也不是物理 RSU 设备实测。

当前 original60 量化 latency/energy 的统一口径:

```text
latency unit = ms
latency metric = latency_ms
hardware/backend = H800 + TVM
scope = backbone/subnet module end-to-end
input boundary = spatial_features / backbone graph input
output boundary = multiscale backbone features / backbone graph outputs
full_network_claim = false
```

这里的“端到端”只指 backbone/subnet 模块内部从输入到输出, 不包含:

```text
raw point cloud preprocessing
collaboration data movement
detection head
postprocess / NMS
complete raw input -> boxes pipeline
real RSU physical edge device runtime
```

可以在报告中写成:

```text
H800 TVM 上的 RSU-side backbone/subnet workload latency proxy
```

不要写成:

```text
真实 RSU 物理边缘设备 latency
完整感知网络端到端 latency
```

## 1. 当前权威三指标状态

行数复核:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl = 60
rows/fp16_true_original60_energy_rows_v1.jsonl = 60
rows/fp16_true_original60_ap_rows_v1.jsonl = 5
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl = 60
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl = 60
rows/native_int8_original60_ap_rows_v1.jsonl = not produced
```

汇总状态:

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

解释:

1. latency 和 energy 对 FP16/native INT8 已经各 60 行。
2. 下一阶段仍要把 energy 纳入验收, 但不是主缺口; 主缺口是 AP。
3. 如果刷新总表后 energy 出现 missing 或 schema 不合规, 需要按失败 label 补跑, 不能直接忽略。
4. native INT8 AP row 文件仍不存在, 这是正确状态; 当前不能导入 reference AP、partial AP、numeric-sanity-only AP 或 AP=0。

## 2. 本轮新增证据链

### 2.1 旧 native INT8 AP-shape route 被证明不是 AP checkpoint-consistent

artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024/checkpoint_weight_audit_v1/
```

结果:

```text
status = blocked
records = 8
records_passed = 0
records_failed = 8
```

结论:

```text
旧 /exdata/.../models/s0_024_backbone.onnx 与当前 AP checkpoint 不是同一组权重,
不能再用于 native INT8 AP adapter 证明。
```

### 2.2 已从 AP checkpoint 导出同源 multiscale ONNX

新增脚本:

```text
scripts/stage2_h800_export_checkpoint_multiscale_onnx.py
```

artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/checkpoint_consistent_s0_024_multiscale_export_v1/
```

导出报告:

```text
export_report.json
status = success
onnx_digest = 90af3348071cfe93cdfa24aac2436054ca2d2ebc84b059c1e3bb750fc07d7e12
op_counts = {"Add": 16, "Conv": 51, "Relu": 48}
input_shape = [1, 64, 256, 256]
output_names = pyramid_level0, pyramid_level1, pyramid_level2
pyramid_level0 shape = [1, 24, 256, 256]
pyramid_level1 shape = [1, 128, 128, 128]
pyramid_level2 shape = [1, 256, 64, 64]
ap_measured = false
full_network_claim = false
```

### 2.3 Conv+BN fused weight audit 全部通过

新增脚本:

```text
scripts/stage2_h800_native_int8_checkpoint_weight_audit.py
```

artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/checkpoint_consistent_s0_024_multiscale_export_v1/checkpoint_weight_audit_fused_bn_all51_v1/
```

结果:

```text
status = passed
records = 51
records_passed = 51
records_failed = 0
skipped = 0
```

结论:

```text
checkpoint-consistent ONNX 的所有 Conv initializer 与 AP checkpoint 中 Conv+BN folded weight 对齐。
权重来源问题已经解决。
```

### 2.4 checkpoint-consistent native INT8 route 已 build/run

route builder 已支持环境变量覆盖 model root:

```text
STAGE2_NATIVE_INT8_MODEL_ROOT
```

artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/
```

关键文件:

```text
s0_024_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so
runtime_weights_int8.npz
tvm_operator_inventory.json
onnx_op_records.json
latency_result.json
energy_result.json
full_onnx_route_attempt.json
```

smoke 结果:

```text
status = success
latency_ms = 9.380352
energy_J = 2.1689717560120227
op_counts = {"Add": 16, "Conv": 51, "Relu": 48}
full_network_claim = false
```

注意:

```text
这次 checkpoint-consistent route 是 AP adapter 修复验证产物, 使用了 debug row-write mode,
没有写入 canonical original60 rows。
canonical original60 native INT8 latency/energy 60/60 已由既有 batch 产出。
```

### 2.5 first-op op-level alignment 已通过

artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/op_level_alignment_v1/
```

结果:

```text
status = passed
processed_samples = 1
selected_op_count = 2
record_count = 4
records_passed = 4
records_failed = 0
weight_record_count = 2
weight_records_passed = 2
weight_records_failed = 0
ap_measured = false
full_network_claim = false
```

结论:

```text
第一层直接消费 spatial_features 的 Conv 与 downsample Conv 已经通过当前 sanity gate。
后续不应再优先怀疑旧的权重错配, 应转向 deeper layer quantization propagation。
```

## 3. 当前 blocker: output scale / requant / residual Add

5-sample output dequant sanity artifact:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/output_dequant_sanity5_v1/
```

summary:

```text
status = blocked
processed_samples = 5
records = 15
ap_measured = false
full_network_claim = false
```

三层输出:

| output | samples | mae_mean | rmse_mean | corrcoef_mean | reference_range_mean | candidate_scale_mean | passed |
|---|---:|---:|---:|---:|---:|---:|---|
| pyramid_level0 | 5 | 18.829709 | 19.066432 | -0.415779 | 20.474501 | 0.103137 | false |
| pyramid_level1 | 5 | 11.008266 | 11.050527 | -0.307276 | 11.224117 | 0.105888 | false |
| pyramid_level2 | 5 | 0.046756 | 0.253762 | null | 9.432849 | 1.0 | true |

解释:

```text
level0/level1 即使做 minmax range alignment, correlation 仍为负;
level2 通过不能证明整体 AP route 可用, 因为 head 主要依赖完整 multiscale 数值结构。
```

当前最可能的问题:

```text
1. 每层固定 requant_u8 = floor(conv / 256) + 128 的尺度过粗或缺少 per-layer output scale。
2. residual Add 直接在 uint8 空间相加, 两个分支 scale/zero_point 没有对齐。
3. Relu / requant / Add 的顺序或 saturating policy 与 PyTorch reference 图不一致。
4. layer0/layer1 中间层已经出现尺度和排序破坏, 最终输出 dequant 无法靠三层末端 minmax 修复。
```

已经降低优先级的问题:

```text
旧 benchmark ONNX 权重不一致
AP checkpoint 与 ONNX initializer 不同源
首层 spatial_features activation zero-point
仅三层 output scale 缺失
```

## 4. 下一阶段主目标

最终收口目标必须非常具体:

```text
完成 original60 现有 60 个配置的 FP16 和 native INT8 补点:

FP16:
  latency_ms = 60/60 measured
  energy_J_per_inference = 60/60 measured
  AP70 = 60/60 measured

native INT8:
  latency_ms = 60/60 measured
  energy_J_per_inference = 60/60 measured
  AP70 = 60/60 measured

总表最终验收:
  FP16/native INT8 latency rows = 120/120 measured
  FP16/native INT8 energy rows = 120/120 measured
  FP16/native INT8 AP rows = 120/120 measured
  jobs_requiring_action = 0
```

当前已完成:

```text
latency rows = 120/120 measured
energy rows = 120/120 measured
AP rows = 5/120 measured
```

下一阶段实际缺口:

```text
FP16 AP: 5/60 -> 60/60
native INT8 AP: 0/60 -> 60/60
energy: 保持 120/120 并在总表刷新后审计; 若出现 missing/schema drift, 按 label 补跑
```

## 5. 下一阶段执行计划

### 5.1 INT8: 先修 deeper layer quantization propagation

目标:

```text
让 checkpoint-consistent native INT8 route 的 multiscale outputs 数值可接 AP eval。
```

执行顺序:

1. 扩展 `scripts/stage2_h800_native_int8_op_alignment.py`, 支持选择更深的 op。
2. 优先 tracing:

```text
resnet.layer0.0.conv1
resnet.layer0.0.conv2
resnet.layer0.0.conv3
resnet.layer0.0.downsample.0
resnet.layer0.0 residual Add
resnet.layer0.1 conv/add path
resnet.layer0.2 final output
resnet.layer1.0 downsample/add path
```

3. 每个 probe 同时保存:

```text
PyTorch reference tensor stats
TVM uint8 tensor stats
candidate dequant tensor stats
scale / zero_point / clipping ratio
correlation / RMSE / MAE
selected op mapping evidence
```

4. 针对第一个失败 op 做最小修复, 不要直接跳到 1789-frame full AP。
5. 重点审查并修复:

```text
requant_u8 divisor and zero_point
per-layer output scale propagation
Add 输入分支 scale alignment
saturating Add policy
Relu before/after requant order
uint8/int32 accumulator clipping
```

通过 gate:

```text
deeper op-level alignment: selected deeper ops correlation >= 0.5 或明确解释低 correlation 的稀疏/饱和原因
5-sample output sanity: pyramid_level0/1/2 全部 passed
20-sample AP smoke: finite AP, pred_nonempty_count > 0, no structural failure
s0_024 1789-frame full AP: processed_samples=1789, AP30/AP50/AP70 finite
```

只有满足这些条件, 才允许写:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

### 5.2 INT8: 首行 AP 通过后扩展 original60

扩展顺序:

1. `s0_024` full AP 首行 measured。
2. 选择 3 个分布不同的 label 做 smoke:

```text
s0_024
s1_048
base 或 frontier label
```

3. smoke 通过后批量跑 original60。
4. 每个 label 必须保留:

```text
checkpoint path / digest
ONNX digest
native INT8 route artifact digest
eval command
raw eval output
AP30 / AP50 / AP70
processed_samples
full_network_claim = false
precision route = native_int8_checkpoint_consistent_backbone_subnet
```

5. 任何失败 label 先落 blocker, 再定位、修复、补跑; 不允许因为一个 label 失败停止其它 label。

### 5.3 FP16: 继续 AP checkpoint recovery 和 full eval

当前 FP16 AP:

```text
5/60 measured
55/60 missing
```

已知 blocker:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/
```

下一步:

1. 不重复盲搜已审计 checkpoint root。
2. 从备份目录、历史训练输出、归档机器或用户提供路径恢复缺失 checkpoint/config。
3. 每恢复一个 label, 运行 true FP16 full AP eval。
4. 成功后追加到:

```text
rows/fp16_true_original60_ap_rows_v1.jsonl
```

5. 每批刷新 summary/review/gap。
6. 若 checkpoint 确认不可恢复, 必须保留 per-label blocker、检索位置、命令、时间、失败分类和后续替代建议; 不能伪造 measured AP。

### 5.4 Energy 和 latency 总表收口

虽然当前 FP16/native INT8 energy 已经 120/120 measured, 下一阶段仍要做表级验收:

1. 重新生成 original60 quant summary/review/gap。
2. 检查:

```text
latency_ms > 0
energy_J_per_inference > 0
measurement_status = measured
full_network_claim = false
unit = ms / J
no latency_p50_us-only final display
```

3. 如果某个 label 的 energy row schema 不一致、缺字段或被 summary 拒绝, 只补跑该 label。
4. 最终 MD 审阅表仍需要提供, 方便人工审阅。

## 6. 遇到问题时的处理原则

下一阶段不能遇到 blocker 就直接停止。标准闭环:

```text
1. 保存 blocker artifact, 包括 stdout/stderr/config/checkpoint/route digest。
2. 写 failure_type 和 failure_reason。
3. 做最小复现, 确认是 export、TVM build、worker runtime、eval adapter、checkpoint 还是 ingestion 问题。
4. 审查最近修改和 route artifact, 避免把旧 artifact 混入新结论。
5. 修复后只补跑失败 label 或失败 gate。
6. 刷新 summary/review/gap。
7. 只有证明当前环境中确实无法解决, 例如 checkpoint 实体不存在且无可访问备份, 才允许把该 label 保持为 blocker。
```

禁止:

```text
把 partial AP 写成 measured AP
把 reference AP 写成 INT8 AP
把 QDQ/TRT/simulated route 写成 native INT8 measured
把 H800 latency 写成真实 RSU 物理设备 latency
把 checkpoint missing 当作全任务停止理由
```

## 7. 关键文件清单

新增/修改过的脚本与测试:

```text
scripts/stage2_h800_native_int8_checkpoint_weight_audit.py
scripts/stage2_h800_export_checkpoint_multiscale_onnx.py
scripts/stage2_h800_native_int8_op_alignment.py
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
framework/tests/test_stage2_native_int8_route.py
```

关键 raw artifacts:

```text
raw/int8_native_route/checkpoint_consistent_s0_024_multiscale_export_v1/export_report.json
raw/int8_native_route/checkpoint_consistent_s0_024_multiscale_export_v1/checkpoint_weight_audit_fused_bn_all51_v1/checkpoint_weight_alignment_summary.json
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/full_onnx_route_attempt.json
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/op_level_alignment_v1/op_level_numeric_alignment_summary.json
raw/int8_native_route/20260628_checkpoint_consistent_s0_024_native_int8_route_v1/s0_024/output_dequant_sanity5_v1/output_dequant_calibration_summary.json
```

当前 rows:

```text
rows/fp16_true_original60_latency_rows_v1.jsonl
rows/fp16_true_original60_energy_rows_v1.jsonl
rows/fp16_true_original60_ap_rows_v1.jsonl
rows/native_int8_full_onnx_original60_latency_rows_v1.jsonl
rows/native_int8_full_onnx_original60_energy_rows_v1.jsonl
```

尚未允许生成或导入:

```text
rows/native_int8_original60_ap_rows_v1.jsonl
```

## 8. 验证

本轮已跑:

```text
PYTHONPATH=${V2X_ROOT} python -m unittest framework.tests.test_stage2_native_int8_route framework.tests.test_stage2_original60_quant_completion
```

结果:

```text
Ran 46 tests in 7.230s
OK
```

语法检查:

```text
python -m py_compile \
  framework/stage2/native_int8_full_onnx.py \
  framework/tests/test_stage2_native_int8_route.py \
  scripts/stage2_h800_native_int8_real_activation_bridge.py \
  scripts/stage2_native_int8_tvm_worker.py \
  scripts/stage2_h800_native_int8_op_alignment.py \
  scripts/stage2_h800_native_int8_checkpoint_weight_audit.py \
  scripts/stage2_h800_export_checkpoint_multiscale_onnx.py \
  multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py
```

结果:

```text
exit code = 0
```

## 9. /goal 命令

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/native INT8 三指标补点收口, 单 agent 执行, 不启动 agent team。当前权威状态: FP16 latency=60/60 measured, FP16 energy=60/60 measured, FP16 AP=5/60 measured; native INT8 latency=60/60 measured, native INT8 energy=60/60 measured, native INT8 AP=0/60 measured; latency 统一使用 latency_ms, 口径是 H800 TVM backbone/subnet module end-to-end, full_network_claim=false, 可作为 RSU-side backbone/subnet workload proxy, 但不能声明为完整感知 pipeline 或真实 RSU 物理设备 latency。当前 INT8 backbone/subnet route 已打通: original60 native INT8 latency/energy 60/60 已有 canonical rows; 新增 checkpoint-consistent s0_024 AP-shape route 已从 AP checkpoint 导出 ONNX, Conv+BN fused weight audit 51/51 passed, TVM native INT8 build/run 成功, first-op op-level alignment passed。当前 native INT8 AP blocker 是 deeper layer output scale/requant/residual Add propagation: output_dequant_sanity5 中 pyramid_level0/1 failed, pyramid_level2 passed, 因此 rows/native_int8_original60_ap_rows_v1.jsonl 仍不得生成。下一阶段目标非常具体: 完成 original60 现有 60 个配置的 FP16 和 native INT8 补点, 最终 FP16/native INT8 latency=120/120 measured、energy=120/120 measured、AP=120/120 measured、jobs_requiring_action=0, 并提供总表审阅 md。执行顺序: 1) 扩展 scripts/stage2_h800_native_int8_op_alignment.py 到 layer0/layer1 deeper ops, 定位第一个破坏数值结构的 requant/Add/Relu/scale 点; 2) 修复 native INT8 per-layer output scale propagation、residual Add 分支 scale alignment、requant_u8 divisor/zero_point 和 saturating policy; 3) 依次通过 deeper op-level alignment、5-sample output sanity、20-sample AP smoke、s0_024 1789-frame full AP gate; 4) 只有 s0_024 full AP gate 通过后, 才写 rows/native_int8_original60_ap_rows_v1.jsonl 首行 measured, 然后扩展到 original60 60 行; 5) 并行继续 FP16 AP checkpoint recovery, 当前 rows/fp16_true_original60_ap_rows_v1.jsonl 为 5 行, 剩余 55 个 label 必须从备份/归档/用户提供路径恢复 checkpoint 后跑 true FP16 full AP, 成功后追加 measured row; 6) 每批刷新 original60_quant summary/review/gap, 审计 energy rows 保持 120/120 measured, 若任何 energy/latency row 被 summary 拒绝则按 label 补跑。遇到 build/eval/import/SSH/checkpoint/adapter/TVM 问题时, 不允许直接停止: 必须保存 blocker artifact/stdout/stderr/config/digest, 反思根因, 审查最近 artifact, 做最小复现, 修复并补跑失败 label/gate; 只有证明当前环境中确实无法解决, 例如 checkpoint 实体不存在且无可访问备份, 才允许留下 per-label blocker。
```
