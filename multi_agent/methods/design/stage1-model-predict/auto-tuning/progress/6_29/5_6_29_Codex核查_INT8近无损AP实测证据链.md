# 60_6_29_Codex核查_INT8近无损AP实测证据链

## 0. 核查结论

针对 `59_6_29_交接文档_INT8近无损存疑问题_待codex核查.md` 中 Claude 给出的 4 个 INT8 full-val AP 实测点，我做了本地行表、H800 远端原始 artifact、TIR lowered 代码、worker response、dequant gate 和导入脚本的独立核查。

结论分三层：

1. **4 个 INT8 AP full-val 结果本身有原始证据支撑**：远端存在 `full_ap_eval_report.json`，均为 1789 帧 full-val，`status=success`、`failed_samples=0`、`pred_nonempty_count=1789`、`ap_row_allowed=true`，AP70 与本地 row/总表一致。
2. **这不是 full-network INT8**：实现只 hook 了 `model.pyramid_backbone.get_multiscale_feature`，后续 head/postprocess 仍是 PyTorch/FP32；row 和总表中 `full_network_claim=False`、`quant_scope=backbone_subnet_native_int8` 是正确的。
3. **“INT8 近无损”只能暂时表述为“backbone/subnet native INT8 bridge 在这 4 个配置上的 AP70 接近 FP16”**。尚不能上升为“整网 INT8 近无损”或“INT8 量化普遍无损”，因为还缺少破坏性量化对照实验来证明 AP 评测对 backbone 量化扰动足够敏感。

## 1. 本地 row 与 FP16 对比复核

数据源：

- `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/native_int8_original60_ap_rows_v1.jsonl`
- `${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl`

核查结果：

| label | FP16 AP70 | INT8 AP70 | INT8-FP16 | INT8 gate |
|---|---:|---:|---:|---|
| s0_024 | 0.5961445249 | 0.5962153156 | +0.0000707907 | native_int8_full_ap_eval |
| s0_040 | 0.5897363386 | 0.5902004388 | +0.0004641002 | native_int8_full_ap_eval |
| s0_056 | 0.5870672005 | 0.5865872843 | -0.0004799162 | native_int8_full_ap_eval |
| s1_048 | 0.5992276271 | 0.5998638386 | +0.0006362115 | native_int8_full_ap_eval |

本地 row 字段显示：

- `measurement_source=true_eval`
- `quant_method=h800_tvm_native_int8_backbone_subnet`
- `quant_scope=backbone_subnet_native_int8`
- `optimized_scope=native_int8_backbone_torch_head_ap_eval`
- `full_network_claim=False`
- `num_samples=1789`
- `pred_nonempty_count=1789`

附带发现：`fp16_true_original60_ap_rows_v1.jsonl` 当前是 61 行、60 个 unique label，`frontier_27` 有重复行。这个问题不影响上述 4 个 INT8 点的对比，但后续冻结 FP16 AP 总表前应单独清理。

## 2. H800 远端原始 full-val 报告复核

本地 row 的 `source_files` 在 4090 工作区均不存在，因此不能只依赖本地二次导入行。已改为核查 H800 远端同路径原始产物。

远端路径前缀：

`${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/int8_native_route/`

| label | raw artifact | processed | failed | pred_nonempty | pred_total | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| s0_024 | `20260628_s0_024_scaleaware_bnaware_bias_v1/.../ap_fullval_1789_bnaware_bias_v1_gpu3_20260629` | 1789 | 0 | 1789 | 41025 | 0.8006229491 | 0.7574909085 | 0.5962153156 |
| s0_040 | `20260628_s0_040_scaleaware_bnaware_bias_v1/.../ap_fullval_1789_bnaware_bias_v1_gpu4_20260629` | 1789 | 0 | 1789 | 38811 | 0.7899000291 | 0.7473765795 | 0.5902004388 |
| s0_056 | `20260628_s0_056_scaleaware_bnaware_bias_v1/.../ap_fullval_1789_bnaware_bias_v1_gpu5_20260629` | 1789 | 0 | 1789 | 38454 | 0.7957009340 | 0.7536073902 | 0.5865872843 |
| s1_048 | `20260628_s1_048_scaleaware_bnaware_bias_v1/.../ap_fullval_1789_bnaware_bias_v1_gpu6_20260629` | 1789 | 0 | 1789 | 38592 | 0.7918883116 | 0.7513142188 | 0.5998638386 |

注意：`full_ap_eval_report.json` 内部的 `ckpt_digest` 为 `None`，但导入脚本会用 `ckpt_path` 对 checkpoint 文件重新计算 sha256 并写入 row。这个不是 AP 数值的致命问题，但属于元数据弱点，建议后续 full report 原始文件也直接写入 `ckpt_digest`。

## 3. TIR 与 runtime 证据

### 3.1 卷积主计算确认为整数累加

4 个 label 的 TIR 都存在，且 artifact `.so` 路径存在。TIR dtype 计数如下：

| label | uint8 | int8 | int32 | float16 | float32 |
|---|---:|---:|---:|---:|---:|
| s0_024 | 231 | 51 | 817 | 0 | 3044 |
| s0_040 | 231 | 51 | 817 | 0 | 3044 |
| s0_056 | 231 | 51 | 817 | 0 | 3044 |
| s1_048 | 229 | 50 | 806 | 0 | 3032 |

关键 TIR 片段显示：

- `conv2d_nchw = T.sblock_alloc_buffer(..., "int32")`
- `group_conv2d_nchw = T.sblock_alloc_buffer(..., "int32")`
- 更新表达式形如：
  - `conv2d_nchw[...] = conv2d_nchw[...] + pad_temp[...] * T.Cast("int32", weight_...)`
  - `group_conv2d_nchw[...] = group_conv2d_nchw[...] + pad_temp_1[...] * T.Cast("int32", weight_...)`

因此 Claude 更正后的判断成立：卷积主路径不是 FP32 conv fallback，而是 `uint8/int8 -> int32 accumulation`。TIR 中大量 `float32` 主要来自 requant/dequant scale，不等于卷积主计算走 FP32。

### 3.2 没有硬件 INT8 加速内核证据

4 个 TIR 中以下关键词均为 0：

- `dp4a`
- `__dp4a`
- `int8x4`
- `conv2d_NCHWc_int8`
- `qnn`
- `wmma`
- `tensorcore`
- `mma`

所以这条路线可以称为 native INT8 数值路径，但不能称为已经使用 H800 上高效 INT8 kernel。它解释 AP 数值，不解释 latency/energy 加速。

### 3.3 worker response 支持实际执行 INT8 `.so`

每个 full-val raw artifact 中都有 `worker_response_summary.json`。复核结果：

| label | worker items | status | returncode | 输出 dtype |
|---|---:|---|---|---|
| s0_024 | 3407 | 全部 success | 全部 0 | `pyramid_level0/1/2` 均为 uint8 |
| s0_040 | 3407 | 全部 success | 全部 0 | `pyramid_level0/1/2` 均为 uint8 |
| s0_056 | 3407 | 全部 success | 全部 0 | `pyramid_level0/1/2` 均为 uint8 |
| s1_048 | 3407 | 全部 success | 全部 0 | `pyramid_level0/1/2` 均为 uint8 |

`native_int8_route_manifest.json` 中 `artifact_path` 指向的 `_tvm_graph.so` 均存在，`full_network_claim=False`，`route_spec=full_onnx_topology_scale_aware_tensor_quant_params_v2`。

## 4. dequant gate 复核

每个点的 `output_dequant_summary.json` 中：

- `full_network_claim=False`
- 每个 label 的 `items=10221`
- `pyramid_level0/pyramid_level1/pyramid_level2` 的 scheme 全部是 `tensor_quant_params_v2`
- 每个 pyramid 输出各有 3407 条记录，对应 worker 调用记录

导入脚本 `${V2X_ROOT}/scripts/stage2_import_native_int8_full_ap_row.py` 的 gate 要求：

- `processed_samples >= 1789`
- `ap_row_allowed=true`
- `ap_row_min_samples >= 1789`
- `pred_nonempty_count > 0`
- `ap30/ap50/ap70` 必须是数值
- `pyramid_level0/1/2` 的 dequant scheme 必须是 `tensor_quant_params_v2`

上述 4 个点均满足。

## 5. 实现路径复核

INT8 AP 评测脚本 `${V2X_ROOT}/scripts/stage2_h800_native_int8_real_activation_bridge.py` 中：

- 行 494-520：每次调用通过 `subprocess.run` 启动 TVM worker，worker returncode 非 0 会抛错。
- 行 757-777：加载 PyTorch 模型、checkpoint、dataset，然后把 `model.pyramid_backbone.get_multiscale_feature` 替换为 `NativeInt8BackboneBridge`。
- 行 797-816：执行 `model(batch_data["ego"])`，随后使用 `dataset.post_process` 和 `eval_utils.caluclate_tp_fp` 计算 AP。
- 行 807：postprocess 前将输出中的浮点 tensor cast 到 FP32。

这说明当前评测是“INT8 backbone bridge + PyTorch head/postprocess”的端到端 AP 评测，不是纯 TVM full-network 推理，也不是全图 INT8。

## 6. 对 Claude 结论的逐项判定

| 待核查项 | Codex 判定 | 说明 |
|---|---|---|
| A. 卷积累加是否 int32 整数 MAC | 支持 | TIR buffer 与乘加表达式确认 int32 accumulation |
| B. 是否执行 int8 `.so`，非 FP32 fallback | 基本支持 | `.so` 存在，worker 全 success，输出为 uint8 pyramid features；仍建议后续保存每帧 worker 使用的 artifact digest |
| C. AP 近 FP16 是否真实 | 部分支持 | 4 个 full-val AP 报告真实存在且通过 gate；但还缺破坏性量化对照证明 AP 对 backbone 量化扰动敏感 |
| D. per-sample AP 累积是否正确 | 基本支持 | 1789 processed、failed=0、pred_nonempty=1789、使用同一 eval_utils；尚未做逐帧 det/gt hash 对账 |
| E. float32 requant 是否使 INT8 名不副实 | 不支持“名不副实”的强说法 | 卷积主计算为 int32，但 requant/dequant 使用 float32 scale；应描述为“非 fully-integer INT8 route” |
| F. backbone-only 是否 low-ball 量化损失 | 支持 | `full_network_claim=False`，head/postprocess FP32 会低估 full-network INT8 损失 |

## 7. 当前总表状态补充

`${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/original60_quant_three_metric_summary_latest.json` 中 4 个 INT8 row：

- `ap_status=measured`
- `ap70` 与 row/full report 一致
- `quality_gate_status` 包含 `native_int8_full_ap_eval`
- `full_network_claim=False`
- `quant_scope=backbone_subnet_native_int8`
- `latency_ms` 有值：s0_024=9.210731、s0_040=9.646859、s0_056=10.466485、s1_048=8.070635
- `energy_J=None`

因此，至少在当前 latest JSON 总表里，INT8 AP 和 latency 已合并，但 energy 字段没有具体数值。后续如果声称 INT8 energy 60/60 measured，需要从 energy row 或 raw result 重新合并到 summary，并确保 `energy_J` 非空。

## 8. 下一步建议

P0：补一个破坏性量化对照实验。选 `s0_024`，保留同一 checkpoint 与评测流程，故意使用错误 scale 或强行 4-bit/极端 clipping。若 AP 明显下降，说明当前评测链路能感知 backbone 量化破坏，4 个近无损点可信度上升；若 AP 仍不降，应优先排查 bridge 是否被绕过、postprocess 是否过强、AP 累积是否存在重复/误用。

P1：同步远端 raw artifact 到本地或在 row 中增加 `remote_source_files`。当前本地 `source_files` 全部不存在，不利于审计复现。

P2：在 `full_ap_eval_report.json` 原始报告中写入 `ckpt_digest`、`artifact_digest`、`route_manifest_digest`，不要只依赖导入脚本补写。

P3：清理 `fp16_true_original60_ap_rows_v1.jsonl` 的 `frontier_27` 重复行，避免后续 full summary 冻结时出现隐性重复。

P4：如果目标是“真实 INT8 加速”，下一阶段重点不应再纠结 AP row 是否存在，而应让 TVM lowering 走高效 int8 kernel 或持久 worker，并明确区分：

- AP 数值验证：当前 4 点基本成立；
- latency/energy 加速验证：当前无 dp4a/tensorcore 证据，不应声称已经实现硬件 INT8 加速。
