# 1_6_30 交接文档: 阶段三 FP16 加速结论、AP-Shape 缺口与下一阶段计划

日期: 2026-06-30

本文用于新窗口冷启动继续推进 FP16 真实加速与 AP-safe 收口。当前结论必须拆成三层:

```text
1. FP16 加速效果不明显的根因已经明确:
   default FP16 full-engine 没有把主要耗时 group conv lower 到 TensorCore fast path。

2. 真实 TensorCore 加速已经在 lhc_07 speed-gate shape 上证明:
   [2,64,128,256] full-engine rewritten latency 18.6305 ms -> 12.7965 ms,
   speedup=1.4559, wmma=72, tvm_mma_sync=2。

3. AP runtime shape 的 AP-safe TensorCore route 尚未闭合:
   AP runtime activation shape 是 [2,64,256,256]。
   当前 AP-shape bridge smoke 已通, 但 AP-shape engine 没有命中 TensorCore gate。
```

因此下一阶段不能再重复证明“FP16 能不能加速”。下一阶段的核心目标是:

```text
把 AP runtime shape [2,64,256,256] 的 backbone/subnet full-engine 也修到真正 TensorCore rewritten route,
并完成 lhc_07 full-val AP 证明。
```

---

## 0. 启动初期必须阅读

新窗口启动后先读:

```text
multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md
multi_agent/methods/design/auto-tuning/progress/6_27/65_6_29_交接文档_阶段三_FP16真实FullEngineTensorCore加速完成与APSafe下一步.md
multi_agent/methods/design/auto-tuning/progress/6_27/66_6_29_交接文档_阶段三_FP16APSafeBridgeSmoke与FullVal启动.md
multi_agent/methods/design/auto-tuning/progress/6_27/61_6_29_交接文档_阶段三_FP16INT8加速原因差异与端到端图内加速计划.md
```

核心脚本:

```text
scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
scripts/stage2_fp16_h800_rewrite_suite_runner.py
scripts/stage2_fp16_h800_gate_check.py
scripts/stage2_fp16_tvm_worker.py
scripts/stage2_h800_fp16_rewritten_activation_bridge.py
scripts/stage2_h800_export_checkpoint_multiscale_onnx.py
```

关键测试:

```bash
python -m unittest framework.tests.test_stage2_h800_fp16_rewritten_activation_bridge framework.tests.test_stage2_fp16_tvm_worker -v
python -m unittest framework.tests.test_stage2_fp16_h800_gate_check framework.tests.test_stage2_fp16_h800_rewrite_suite_runner -v
python -m py_compile scripts/stage2_h800_fp16_rewritten_activation_bridge.py scripts/stage2_fp16_tvm_worker.py scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
```

---

## 1. 固定口径

1. latency 统一使用 `ms`。
2. 当前速度对象是 `H800 + TVM compiled backbone/subnet full-engine`, 输入为 `spatial_features`, 输出为 backbone/subnet multiscale feature。
3. 不能写成完整 perception pipeline latency, 也不能写成真实 RSU 物理设备端到端绝对速度。当前仍是 `full_network_claim=false`。
4. AP-safe 与 speed-gate 分开记录。speed-gate true 不自动代表 AP-safe true。
5. 只有完整 AP runtime shape engine 自身满足以下条件, 才能写成 `AP-shape FP16 TensorCore full-engine measured`:
   - H800/sm90 target;
   - rewritten latency 小于 default latency;
   - `wmma/tvm_mma_sync > 0`;
   - output error 可解释或 AP full-val 可接受;
   - `.so`、TIR、stdout/stderr、report、raw artifact 完整保存。
6. 如果 full-val 仍在后台运行, 不得把 1-sample smoke 写成 full-val。

---

## 2. FP16 加速效果不明显的原因

### 2.1 表象

此前 original60 总表里 FP16 latency 与 FP32 接近, 部分点甚至看起来 FP16 没有优势。

错误解释是:

```text
FP16 本身对 H800/TVM 没有加速价值。
```

当前证据已经证明这个解释不成立。

### 2.2 真实原因

真实原因是:

```text
default FP16 full-engine 虽然 dtype 是 FP16,
但主要耗时 group conv 没有进入 TensorCore fast path。
```

核心证据:

| 证据 | 结果 | 结论 |
|---|---:|---|
| default lhc_07 FP16 full-engine | `wmma=0`, `tvm_mma_sync=0` | default engine 不是 TensorCore engine |
| 真实 1x1 convblock | `wmma=36`, `tvm_mma_sync=1` | TVM/H800 能生成 FP16 TensorCore |
| all-1x1 counterfactual | 收益约 `1.004x` | 1x1 不是主要耗时 |
| group conv 单层/full im2col | 单层有显著收益 | 主要空间在 3x3/group conv |
| speed-gate shape full-engine group-conv rewrite | `18.6305 ms -> 12.7965 ms`, `1.4559x`, `wmma=72` | 根因已在真实 full-engine 上闭合 |

一句话结论:

```text
FP16 速度不明显不是量化无效, 而是默认 lowering/schedule 没把主要耗时 group conv 送入 TensorCore。
```

---

## 3. 已采用的解决方法

### 3.1 1x1 rewrite 作为 route proof

先将可匹配的 1x1 conv 重写为:

```text
NCHW -> NHWC -> reshape -> matmul -> bias/add -> reshape -> NCHW
```

该路线证明了 TVM/H800 可以稳定生成 FP16 TensorCore, 但 1x1 不是主要瓶颈, 因此对 full-engine latency 收益很小, 甚至可能被 transpose/reshape overhead 抵消。

### 3.2 转向主要耗时 group conv

随后选择 full-engine 中主要耗时的 3x3/group conv PrimFunc 做 TIR replacement。

speed-gate shape 中命中的 replacement:

```text
fused_conv2d12_add8_relu6
fused_conv2d16_add8_relu6
```

方法:

```text
1. 在 full-engine TIR 中定位目标 group conv PrimFunc。
2. 用 same-signature full-im2col/matmul/restore replacement 替换目标 PrimFunc。
3. main graph 和其它 PrimFunc 保持不变。
4. 对 rewritten full-engine 应用 selective MatmulTensorization。
5. 保存 rewritten TIR、scheduled TIR、.so、latency、output_compare、stdout/stderr。
```

### 3.3 修复 selective tensorization 漏调度

曾经出现 replacement 后 latency 变快, 但 `wmma=0/tvm_mma_sync=0` 的情况。

根因:

```text
_apply_selective_matmul_tensorization 依赖 func.attrs["s_tir"] 判断 TIR PrimFunc。
H800 远端 TVM 0.20 生成的 PrimFunc 没有该 attr。
因此包含 T.sblock("matmul") 的 replacement PrimFunc 被误判 skipped_non_tir。
```

修复:

```text
只要对象有 script(), 就进入文本判定。
含 T.sblock("matmul") 或函数名含 matmul 的 PrimFunc 尝试 MatmulTensorization。
其它 PrimFunc 保持原样。
```

修复后 speed-gate shape 出现真实 TensorCore:

```text
fused_conv2d12_add8_relu6: wmma=36, tvm_mma_sync=1
fused_conv2d16_add8_relu6: wmma=36, tvm_mma_sync=1
total: wmma=72, tvm_mma_sync=2
```

---

## 4. 已达到的解决效果

### 4.1 Speed-Gate Shape: TensorCore 加速已闭合

审阅文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_rewrite_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_rewrite_latest.md
```

H800 raw:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewrite_suite_20260629_h800_tensorcore_fix_20260629_130126/lhc07_full_engine_group_conv_rewrite/
```

关键结果:

| item | value |
|---|---:|
| input shape | `[2,64,128,256]` |
| output0 shape | `[2,24,128,256]` |
| output1 shape | `[2,48,64,128]` |
| output2 shape | `[2,128,32,64]` |
| default latency | `18.630492333333333 ms` |
| rewritten latency | `12.79650124 ms` |
| speedup | `1.4559051715712046x` |
| wmma | `72` |
| tvm_mma_sync | `2` |
| tensorcore_gate | `true` |

output error:

| output | max_abs_err | mean_abs_err | mean_abs_err/ref_abs_mean |
|---|---:|---:|---:|
| output0 | `0.0` | `0.0` | `0.0` |
| output1 | `0.0` | `0.0` | `0.0` |
| output2 | `0.72705078125` | `0.07954635471105576` | `0.1895030289888382` |

结论:

```text
FP16 TensorCore 加速不是理论 counterfactual, 已经有 H800/sm90 真实 rewritten full-engine measured latency。
```

限制:

```text
该 speed-gate engine 的 input shape 是 [2,64,128,256],
不能直接用于当前 AP runtime activation shape [2,64,256,256]。
```

### 4.2 AP-Shape Bridge: AP 路线已打通, 但 TensorCore 未闭合

AP checkpoint/runtime 中 `lhc_07` 的 `get_multiscale_feature` activation shape 是:

```text
[2,64,256,256]
```

为此重新导出了 checkpoint-consistent AP-shape ONNX:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260629/lhc_07_apshape_multiscale.onnx
```

输出 shape:

```text
pyramid_level0=[2,24,256,256]
pyramid_level1=[2,48,128,128]
pyramid_level2=[2,128,64,64]
```

AP-shape rewrite 审阅文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_apshape_lhc07_rewrite_20260629/fp16_lhc07_full_engine_group_conv_rewrite_latest.json
```

AP-shape `.so`:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_shape_lhc07_20260629/rewrite/lhc07_full_engine_group_conv_rewrite/rewritten_full_engine.so
```

AP-shape 当前结果:

| item | value |
|---|---:|
| input shape | `[2,64,256,256]` |
| default latency | `36.48966236 ms` |
| rewritten latency | `36.4855846 ms` |
| wmma | `0` |
| tvm_mma_sync | `0` |
| tensorcore_gate | `false` |
| replace_records | `[]` |

这说明:

```text
AP-shape engine build/run 成功, 但当前 replacement matcher 没有命中 AP-shape 的 group conv。
因此 AP-shape 路线目前是 correctness/bridge 证据, 不是 TensorCore speed 证据。
```

### 4.3 AP Smoke: TVM output -> head/postprocess/AP eval 已通

审阅文件:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_smoke_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_smoke_latest.md
```

H800 raw:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_smoke_v3
```

结果:

| item | value |
|---|---:|
| processed_samples | `1` |
| failed_samples | `0` |
| pred_nonempty_count | `1` |
| pred_total_count | `18` |
| AP30 | `0.7406249999999999` |
| AP50 | `0.7406249999999999` |
| AP70 | `0.6293154761904762` |
| smoke_gate_passed | `true` |
| ap_measured | `false` |

AP-shape smoke output error:

| output | shape | max_abs_err | mean_abs_err | mean_abs_err/ref_abs_mean |
|---|---:|---:|---:|---:|
| output0 | `[1,24,256,256]` | `0.029115676879882812` | `0.00121464638505131` | `0.001472104568857695` |
| output1 | `[1,48,128,128]` | `0.035881638526916504` | `0.0008529233746230602` | `0.002676371086409351` |
| output2 | `[1,128,64,64]` | `0.02682185173034668` | `0.00022862741025164723` | `0.0023954663096400694` |

注意:

```text
该 smoke 的 output error 很小, 是因为 AP-shape engine 当前没有命中 TensorCore replacement。
它证明 bridge 和 AP eval 路径可用, 但不能证明 AP-shape TensorCore route 已 AP-safe。
```

---

## 5. 当前主要卡点

当前卡点不是“FP16 能否加速”, 而是:

```text
speed-gate shape 和 AP runtime shape 不一致。
```

具体表现:

| 项目 | speed-gate shape | AP runtime shape |
|---|---:|---:|
| input | `[2,64,128,256]` | `[2,64,256,256]` |
| output0 | `[2,24,128,256]` | `[2,24,256,256]` |
| output1 | `[2,48,64,128]` | `[2,48,128,128]` |
| output2 | `[2,128,32,64]` | `[2,128,64,64]` |
| TensorCore gate | true | false |
| AP bridge | 不适配 | smoke 已通 |

导致的工程后果:

```text
1. 65 号 speed-gate engine 不能直接拿来跑 AP eval。
2. AP-shape engine 虽然能接入 AP eval, 但 replacement matcher 没命中, replace_records=[]。
3. 所以下一阶段必须修 AP-shape group conv matcher/rewrite, 而不是再跑旧 speed-gate shape。
```

---

## 6. 下一阶段计划

### P0: 冻结现有证据和口径

目标:

```text
禁止把 AP-shape smoke 写成 TensorCore speed 证据;
禁止把 speed-gate shape 写成 AP-safe full-val 证据。
```

要做:

1. 确认并保留 65/66 号文档。
2. 所有新表格新增至少以下字段:
   - `input_shape`
   - `route_identity`
   - `tensorcore_gate`
   - `wmma`
   - `tvm_mma_sync`
   - `ap_measured`
   - `full_network_claim`
3. 对外表述必须写:

```text
speed-gate closed; AP-shape TensorCore not yet closed.
```

### P1: 检查 AP-shape full-val v2 是否完成

已启动的 full-val v2:

```text
raw_dir=${V2X_DATA_ROOT}/s2_tvm/fp16_rewritten_ap_bridge_20260629_lhc07_apshape_fullval_v2
export_report_json=multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_full_engine_ap_fullval_latest.json
```

注意:

```text
截至本文撰写时, 本地尚未看到 fp16_lhc07_rewritten_full_engine_ap_fullval_latest.json。
因此不能宣布 full-val 完成。
```

下一步:

1. 远端检查 v2 是否生成 report 或 blocker。
2. 如果完成:
   - 同步 report、stdout/stderr、summary JSON 到本地。
   - 生成 `fp16_lhc07_rewritten_full_engine_ap_fullval_latest.md`。
   - 检查 `processed_samples`、`failed_samples`、`pred_nonempty_count`、AP30/AP50/AP70、score/box distribution。
3. 如果失败:
   - 保留 blocker、traceback、worker request/response、activation、stdout/stderr。
   - 按失败原因修复后重跑。

### P2: 修 AP-shape group conv matcher/rewrite

目标:

```text
让 AP-shape [2,64,256,256] full-engine 也出现 replace_records>0,
并让 scheduled_counts 中 wmma/tvm_mma_sync > 0。
```

具体步骤:

1. 打开 AP-shape `full_engine_legalize_fuse_before_rewrite.py`。
2. 对比 speed-gate shape 的 `full_engine_legalize_fuse_before_rewrite.py`。
3. 找出 AP-shape 中对应 group conv PrimFunc 的实际 shape/name/script pattern。
4. 修复 matcher:
   - 不依赖固定 PrimFunc 名称;
   - 使用 shape、group、kernel、layout、callsite 等结构特征匹配;
   - 记录所有候选和未命中原因。
5. 先只替换一个 AP-shape group conv, 做 smoke:
   - build 成功;
   - `.so` 成功 export;
   - `wmma/tvm_mma_sync > 0`;
   - output error 可解释。
6. 再替换两个主要 group conv, 跑 latency gate。

验收标准:

```text
AP-shape default latency > AP-shape rewritten latency
wmma/tvm_mma_sync > 0
replace_records 非空
output_compare 有记录
raw TIR/.so/stdout/stderr 完整保存
```

### P3: AP-shape TensorCore AP smoke

在 P2 完成后, 重新跑 AP bridge smoke。

目标:

```text
processed_samples >= 1
failed_samples = 0
pred_nonempty_count > 0
AP30/AP50/AP70 非空且趋势合理
output0/1/2 error 可解释
```

如果 output2 drift 大:

1. 区分是单 PrimFunc indexing 问题, 还是 repeated callsite 误差累积。
2. 分别跑:
   - downsample-only rewrite;
   - repeated-only rewrite;
   - callsite-aware rewrite。
3. 若单层误差小但 AP 下降明显, 保存 head/postprocess distribution 对照。

### P4: AP-shape TensorCore full-val

AP smoke 通过后, 跑 lhc_07 full-val。

目标:

```text
完成 lhc_07 FP16 AP-shape TensorCore full-val:
AP30/AP50/AP70
prediction count
score/box distribution
output error
latency_ms
raw artifacts
```

full-val 完成后才能写:

```text
lhc_07 FP16 AP-shape TensorCore route AP-safe evidence available.
```

### P5: 推广到 original60

在 `lhc_07` AP-shape TensorCore route 闭合后, 再推广到 original60。

计划:

1. 先选 3 个代表点:
   - 低延迟点;
   - 中位点;
   - 高延迟点。
2. 每个点都要求:
   - latency_ms;
   - TensorCore gate;
   - output error;
   - AP smoke;
   - raw artifact。
3. 3 点稳定后再扩展到 60 点。

---

## 7. 下一窗口建议 /goal

```text
/goal 基于 ${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_30/1_6_30_交接文档_阶段三_FP16加速结论_APShape缺口与下一阶段计划.md 继续推进 FP16 AP-shape TensorCore 收口。固定前提: lhc_07 speed-gate shape [2,64,128,256] 已证明 H800/sm90 真实 FP16 TensorCore full-engine 加速, latency 18.6305 ms -> 12.7965 ms, wmma=72, tvm_mma_sync=2; 不要重复证明 speed-gate。当前目标是修复 AP runtime shape [2,64,256,256] 的 group conv matcher/rewrite, 让 AP-shape full-engine 也产出 replace_records>0、wmma/tvm_mma_sync>0、rewritten latency 小于 default latency, 然后用 scripts/stage2_h800_fp16_rewritten_activation_bridge.py 完成 lhc_07 AP smoke 和 full-val。必须保存 TIR/.so/stdout/stderr/report/blocker/output_compare/AP30/AP50/AP70/prediction distribution。遇到失败不能停止, 先保留 raw artifact 并做 shape/matcher/callsite/output2 drift 根因分析。
```

---

## 8. 最重要的交接提醒

1. 不要再把 FP16 加速问题解释为“FP16 不适合 H800/TVM”。证据显示问题在 default lowering/schedule。
2. 不要把 `wmma=0` 的 AP-shape engine 当作 TensorCore engine。
3. 不要把 1-sample smoke 写成 full-val。
4. 不要把 `[2,64,128,256]` speed-gate engine 直接写成 AP runtime engine。
5. 下一阶段最优先的技术动作是修 AP-shape group conv matcher, 不是重新测旧 shape latency。
