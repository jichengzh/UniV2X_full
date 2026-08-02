# 22_6_28_交接文档_阶段三_INT8APAdapter根因诊断与数值Route计划

更新时间: 2026-06-28

本文件继承:

- `21_6_28_交接文档_阶段三_INT8Backbone口径澄清与FP16INT8补点收口计划.md`
- `20_6_28_交接文档_阶段三_FP16AP首批4点TrueEval导入.md`

本轮没有新增 measured AP row。新增的是 `s0_024` native INT8 AP adapter 的远端接口诊断和精确 blocker artifact, 用来把后续实现目标从“笼统接 AP”收窄到“先构建数值可用的 native INT8 AP-shape route”。

## 0. 当前权威状态

```text
FP16 latency = 60/60 measured
FP16 energy = 60/60 measured
FP16 AP70 = 5/60 measured
native INT8 latency = 60/60 measured
native INT8 energy = 60/60 measured
native INT8 AP70 = 0/60 measured
completion jobs requiring action = 115
```

总表和 review:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_completion_review_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_int8_original60_gap_report_latest.json
```

## 1. 本轮新增诊断产物

### 1.1 HEAL model boundary probe

远端 H800 单样本 forward probe 已成功, 本地同步位置:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/native_int8_adapter_probe_s0_024/interface_probe.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/native_int8_adapter_probe_s0_024/pyramid_boundary_probe.json
```

关键结论:

```text
model_class = HeterPyramidCollab
record_len = [1]
agent_modality_list = ["m1"]
forward_collab input spatial_features = [1, 64, 256, 256], float32
get_multiscale_feature output =
  [1, 24, 256, 256]
  [1, 128, 128, 128]
  [1, 256, 64, 64]
decode_multiscale_feature output = [1, 384, 256, 256]
shrink_conv output = [1, 256, 256, 256]
cls_head output = [1, 2, 256, 256]
reg_head output = [1, 14, 256, 256]
dir_head output = [1, 4, 256, 256]
```

因此 INT8 AP adapter 的正确插入点不是完整模型入口, 而是 `PyramidFusion.forward_collab` 内部的:

```text
get_multiscale_feature(spatial_features)
```

adapter 必须保留:

```text
record_len
pairwise_t_matrix -> normalize_pairwise_tfm
agent_modality_list
weighted_fuse
decode_multiscale_feature
shrink_conv
cls_head / reg_head / dir_head
dataset.post_process
```

### 1.2 s0_024 native INT8 AP blocker

本轮新增 blocker:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/native_int8_s0_024_adapter_probe_v1/ap_eval_blocker.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/native_int8_s0_024_adapter_probe_v1/runner_command.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/native_int8_s0_024_adapter_probe_v1/stdout.txt
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/native_int8_s0_024_adapter_probe_v1/stderr.txt
```

blocker 摘要:

```text
schema = stage2_native_int8_ap_eval_blocker_v1
label = s0_024
precision = int8
status = blocked_no_measured_ap_row
failure_type = adapter_backend_not_numerically_ready
failure_reason = current native INT8 full ONNX artifact is a benchmark backbone/subnet route, not a calibrated numerical AP eval backend
no_row_appended = true
full_network_claim = false
```

### 1.3 FP16 AP checkpoint inventory 与逐点 blocker

本轮重新在 H800 checkpoint root 做了 inventory, 并与当前 FP16 AP no_claim 的 55 个 label 对照。结果: 当前可见 checkpoint 目录中没有匹配剩余 55 个 label 的 `Pyramid_DAIR_m1_stage2_ap_<label>...` checkpoint/config。

inventory 与对照结果:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_checkpoint_inventory_20260628_v2/h800_checkpoint_inventory.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_checkpoint_inventory_20260628_v2/fp16_missing_checkpoint_audit.json
```

逐点 blocker:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/summary.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/<label>/ap_eval_blocker.json
```

blocker 计数:

```text
fp16 AP remaining labels = 55
checkpoint matched = 0
missing_checkpoint blockers written = 55
```

这些 blocker 不能当作 measured AP row。后续若恢复某个 label 的 checkpoint/config, 应只重跑该 label, 成功后追加 measured AP row 并刷新 review/gap。

## 2. INT8 AP 当前真正 blocker

当前问题不是 TVM 不能 build INT8 backbone。native INT8 backbone/subnet latency/energy 已经 60/60 measured, 且 QDQ/float32-heavy lowering 问题已经在这条 route 上解决。

当前 INT8 AP blocker 是: 现有 native INT8 route 还不是可以参与 AP eval 的数值后端。

具体原因:

1. `HeterPyramidCollab` 的 AP eval 需要 `cls_preds/reg_preds/dir_preds`, 当前 native INT8 artifact 只输出 PyramidFusion resnet 的三层 multiscale feature。
2. 当前 native INT8 latency/energy artifact 的 input/output shape 来自 benchmark ONNX:

```text
input_shape = spatial_features [2, 64, 128, 256]
output_shape =
  [2, 24, 128, 256]
  [2, 128, 64, 128]
  [2, 256, 32, 64]
```

而 `s0_024` AP eval 单样本实际边界是:

```text
input = [1, 64, 256, 256]
output =
  [1, 24, 256, 256]
  [1, 128, 128, 128]
  [1, 256, 64, 64]
```

3. 当前 native INT8 full ONNX route 脚本把 Conv weights 建成 TE placeholder, latency/energy benchmark 阶段喂的是 synthetic runtime weights/input。这足以测 backbone/subnet route 的 latency/energy, 但不能作为 AP 数值后端。
4. 当前 route 的 calibration 记录是 `none_direct_native_int8_synthetic_inputs`, 没有 activation/weight scale、zero point、dequant policy, 因此不能把其输出直接用于 AP。
5. 环境也有实际工程限制: HEAL eval Python 环境有 torch/opencood 但没有 TVM; TVM Python 环境有 TVM runtime 但没有 torch。直接把 TVM runtime site-packages 注入 HEAL 环境会破坏 torch import。后续需要 persistent TVM worker 或重新布置 runtime 环境, 不能靠一次性混 PYTHONPATH。

## 3. 下一步实现目标

下一阶段目标不是重新证明 INT8 backbone latency/energy, 而是把 `s0_024` native INT8 AP adapter 从 blocker 推到 one-sample smoke。

### 3.1 单点实现 gate

`s0_024` 必须先完成:

```text
HEAL torch encoder/backbone_m1/aligner_m1 -> aligned spatial_features
aligned spatial_features -> numerical native INT8 PyramidFusion multiscale route
multiscale tensors -> existing PyTorch weighted_fuse/decode/shrink/head
head outputs -> dataset.post_process
one-sample smoke -> ap_eval_report.json or precise blocker
```

one-sample smoke 通过后再跑 1789-frame full AP。不能跳过 one-sample gate 直接跑全量。

### 3.2 数值 route 必须补齐的内容

必须补齐:

1. AP eval shape override 或重新 export/build AP-shape ONNX route。
2. 从 checkpoint/ONNX initializer 绑定真实权重, 不能再用 runtime random weights。
3. activation/weight quantization recipe, 包括 scale、zero point、clip/dequant policy。
4. route 输出 tensor 顺序必须和 `PyramidFusion.get_multiscale_feature` 一致。
5. HEAL torch 环境和 TVM runtime 的 bridge:
   - 推荐先做 persistent subprocess worker, 用 tvm310 Python 常驻加载 artifact, 通过 stdin/stdout 或 mmap/npy 传输入输出。
   - 不建议把 tvm310 site-packages 直接塞进 UniV2X Python, 已实测会破坏 torch import。

### 3.3 验证顺序

1. 本地单测: blocker schema、shape contract、forbidden AP row import。
2. H800 one-sample diagnostic:

```text
input shape = [1,64,256,256]
output shapes = [1,24,256,256], [1,128,128,128], [1,256,64,64]
head outputs = cls/reg/dir shapes match FP32 path
dataset.post_process succeeds
```

3. H800 one-sample AP smoke:

```text
num_samples = 1
write ap_eval_report.json only as smoke evidence
do not append measured AP row
```

4. H800 full AP:

```text
num_samples = 1789
append rows/native_int8_original60_ap_rows_v1.jsonl only if full eval succeeds
```

## 4. FP16 AP 剩余缺口

FP16 AP 当前 5/60 measured。剩余 55 个 label 的主要问题仍是 checkpoint recovery。

已 measured:

```text
s0_024
s0_040
s0_056
s1_048
s2_160
```

下一步继续:

1. 不再重复盲搜同一个 checkpoint root; 先根据 `fp16_missing_checkpoint_audit.json` 扩展新的候选来源, 例如备份目录、历史训练输出、归档机器或手动提供路径。
2. 找到 checkpoint 就跑 `scripts/stage2_h800_true_fp16_ap_eval.py`。
3. 当前 55 个 missing checkpoint label 已有 per-label blocker; checkpoint 恢复后应按 label 重跑并替换 no-claim 状态, 不得伪造 measured AP。

## 5. 下一阶段 /goal

```text
/goal 在 ${V2X_ROOT} 中继续 Stage2 original60 FP16/INT8 AP 补点收口, 单 agent 执行, 不启动 agent team。当前权威状态: FP16/INT8 latency=120/120 measured, energy=120/120 measured, AP=5/120 measured, jobs_requiring_action=115; latency 是 H800 TVM backbone/subnet module latency, 对外统一使用 ms, full_network_claim=false。本轮已完成 s0_024 native INT8 AP adapter 根因诊断: HEAL 模型是 HeterPyramidCollab, AP 边界 input=[1,64,256,256], multiscale output=[1,24,256,256]/[1,128,128,128]/[1,256,64,64]; 已落 blocker multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60/native_int8_s0_024_adapter_probe_v1/ap_eval_blocker.json。关键 blocker: 当前 native INT8 latency/energy route 是 benchmark backbone/subnet route, 使用 benchmark shape 和 TE placeholder/synthetic runtime weights, calibration_source=none_direct_native_int8_synthetic_inputs, 不能直接作为 AP 数值后端。FP16 AP 方面, 已重新做 H800 checkpoint inventory, 当前剩余 55 个 FP16 AP label 在默认 checkpoint root 中匹配数为 0, 已写 55 个 missing_checkpoint blocker 到 raw/ap_eval_original60/fp16_missing_checkpoint_blockers_v1/。下一步先实现 s0_024 numerical native INT8 AP adapter: 1) build 或重建 AP-shape PyramidFusion native INT8 route; 2) 绑定真实 checkpoint/ONNX weights, 补齐 activation/weight scale、zero point、clip/dequant policy; 3) 通过 persistent TVM worker 或等价 bridge 接入 HEAL torch eval 环境, 不要混用会破坏 torch 的 PYTHONPATH; 4) 先跑 one-sample smoke, 验证 multiscale shape、head cls/reg/dir shape、dataset.post_process; 5) one-sample 通过后跑 s0_024 1789-frame full AP, 成功后写 rows/native_int8_original60_ap_rows_v1.jsonl 首行 measured; 6) 再扩展 INT8 original60 60 行。并行继续 FP16 AP checkpoint recovery, 但不要重复盲搜已审计 root, 应从备份/归档/人工路径恢复 checkpoint 后按 label 重跑。每批后刷新 original60_quant_three_metric_summary_latest、fp16_int8_original60_completion_review_latest、gap_report_latest。遇到任何 build/eval/import/SSH/checkpoint/adapter 问题, 必须先保存 blocker artifact/stdout/stderr, 再反思、审查、最小复现、修复和失败 label 补跑; 不允许直接停止, 除非证明当前环境或缺失 checkpoint 确实无法由执行 agent 解决。
```
