# 65_6_29 交接文档: 阶段三 FP16 真实 Full-Engine TensorCore 加速完成与 APSafe 下一步

日期: 2026-06-29

本文用于新窗口冷启动继续推进 FP16 加速研究。当前最重要结论:

```text
lhc_07 的 FP16 速度/TensorCore gate 已经从 counterfactual 推进到 H800/sm90 真实 rewritten full-engine measured latency。
当前已经证明: 默认 FP16 不快的根因是 full-engine lowering 没有把主要耗时 group conv 送入 TensorCore fast path。
通过 full-engine TIR replacement + selective MatmulTensorization, 已得到真实 full-engine 加速:
18.630492333333333 ms -> 12.79650124 ms, speedup=1.4559051715712046, wmma=72, tvm_mma_sync=2。
```

但还不能写成 AP-safe FP16 route:

```text
output2 drift 仍明显, 且 TVM rewritten backbone/subnet 到 head/postprocess/AP eval bridge 仍缺。
下一阶段重点不是继续证明能否加速, 而是 AP-safe bridge 与 output2 drift 收口。
```

---

## 0. 启动初期必须阅读

```text
multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md
multi_agent/methods/design/auto-tuning/progress/6_27/59_6_29_交接文档_阶段三_FP16TensorCore端到端验证与下一步加速计划.md
multi_agent/methods/design/auto-tuning/progress/6_27/64_6_29_交接文档_阶段三_FP16INT8加速根因与端到端图内加速冷启动计划.md
```

核心脚本:

```text
scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
scripts/stage2_fp16_h800_rewrite_suite_runner.py
scripts/stage2_fp16_h800_gate_check.py
```

最新关键产物:

```text
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_rewrite_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_rewrite_latest.md
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_h800_gate_check_latest.json
multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_h800_gate_check_latest.md
```

H800 raw artifacts:

```text
${V2X_DATA_ROOT}/s2_tvm/fp16_rewrite_suite_20260629_h800_tensorcore_fix_20260629_130126/lhc07_full_engine_group_conv_rewrite/
${V2X_DATA_ROOT}/s2_tvm/fp16_rewrite_suite_20260629_h800_full_20260629_123740/
```

---

## 1. 固定口径

1. latency 统一使用 `ms`。
2. 当前速度对象是 `H800 + TVM compiled backbone/subnet full-engine`, 输入为 `spatial_features`, 输出为 backbone/subnet multiscale feature。
3. 不能写成完整 perception pipeline latency, 也不能写成真实 RSU 物理设备端到端绝对速度。当前仍是 `full_network_claim=false`。
4. default FP16 engine 不能写成 TensorCore engine。
5. 只有 rewritten full-engine 自身满足:
   - H800/sm90 target;
   - `rewritten_latency_ms < default_latency_ms`;
   - `wmma/tvm_mma_sync > 0`;
   - raw artifact/TIR/.so/stdout/stderr 完整保存;
   才能写成 `FP16 TensorCore full-engine measured latency`。
6. AP-safe 与速度 gate 分开。速度/TensorCore gate 已闭合, AP-safe gate 未闭合。

---

## 2. FP16 加速效果不明显的原因

### 2.1 历史表象

此前 original60 中 FP16 latency 与 FP32 接近, 甚至部分点看起来 FP16 没有优势。

容易误判的错误解释:

```text
FP16 本身对 TVM/H800 没有加速价值。
```

当前证据证明这个解释不成立。

### 2.2 真正原因

真实原因是:

```text
完整 lhc_07 default FP16 engine 虽然 dtype 是 FP16, 但 full-engine lowering 没有把主要耗时算子送入 TensorCore fast path。
```

分阶段证据:

| 证据 | 结果 | 结论 |
|---|---:|---|
| lhc_07 真实 1x1 convblock | `wmma=36`, `tvm_mma_sync=1` | TVM/H800 能生成 FP16 TensorCore |
| 全部 eligible 1x1 Conv counterfactual | 约 `0.076 ms`, `1.004x` 收益 | 1x1 不是主要耗时 |
| default full FP16 engine | `wmma=0`, `tvm_mma_sync=0`, latency 约 `19 ms` | default full-engine 不是 TensorCore engine |
| group conv 单层/full im2col | 单层可显著快于 default group conv | 主要加速空间在 3x3/group conv |
| 修复后 full-engine group-conv rewrite | `wmma=72`, `tvm_mma_sync=2`, `1.4559x` | 根因被实证闭合 |

一句话总结:

```text
FP16 不是量化无效, 而是默认 full-engine lowering 没有把主要耗时 group conv 变成 TensorCore 可执行路径。
```

---

## 3. 采用的解决方法

### 3.1 第一阶段: 1x1 rewrite 作为 route proof

目标:

```text
将 stride=1/group=1/kernel=1x1/pad=0 的 Conv 重写为:
NCHW -> NHWC -> Reshape -> MatMul -> Add -> Reshape -> NCHW
```

结果:

```text
rewrite-onnx-1x1 可以生成 rewritten ONNX、shape inference、TVM build/run。
但 H800 latest 1x1 rewritten full-engine:
default_ms=18.488221013333327
rewritten_ms=19.483181546666668
speedup=0.9489323378242829
wmma=0
tvm_mma_sync=0
```

解释:

```text
1x1 rewrite 证明了 route 可走, 但不是 lhc_07 的加速解。
它也再次说明主要耗时不在 eligible 1x1 Conv。
```

### 3.2 第二阶段: 选择主要耗时 group conv

根据 counterfactual 和 profile, 继续选择主要耗时的 3x3/group conv。

关键 replacement:

```text
fused_conv2d12_add8_relu6
fused_conv2d16_add8_relu6
```

方法:

```text
1. 在 full-engine TIR 中定位目标 group conv PrimFunc。
2. 用 same-signature full-im2col/matmul/restore replacement 替换目标 PrimFunc。
3. main graph 其它部分保持不变, 生成真实 rewritten full-engine binary。
4. 对 rewritten full-engine 应用 selective MatmulTensorization。
5. 保存 rewritten TIR、ONNX、.so、stdout/stderr、latency 和输出误差。
```

### 3.3 第三阶段: 修复 selective tensorization 漏调度问题

首次 H800 full suite 得到:

```text
default_ms=18.480157860000002
rewritten_ms=12.88581602
speedup=1.4341472694718795
target_arch=sm_90
wmma=0
tvm_mma_sync=0
tensorcore_gate=false
```

这说明 replacement 已经让 full-engine 变快, 但不满足 59 号文档要求的 TensorCore gate。

根因:

```text
_apply_selective_matmul_tensorization 依赖 func.attrs["s_tir"] 判断 TIR PrimFunc。
H800 远端 TVM 0.20 生成的 PrimFunc 没有该 attr。
因此所有 TIR PrimFunc 被标记为 skipped_non_tir。
replacement 中实际存在 T.sblock("matmul"), 但被提前跳过, 所以 wmma/tvm_mma_sync=0。
```

修复:

```text
scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
删除 _apply_selective_matmul_tensorization 中对 attrs["s_tir"] 的硬依赖。
只要对象有 script(), 就进入文本判定:
  - 含 T.sblock("matmul") 或函数名含 matmul -> 尝试 MatmulTensorization
  - 其它 PrimFunc 保持原样
```

验证:

```bash
python -m py_compile scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py
python -m unittest framework.tests.test_stage2_fp16_h800_gate_check framework.tests.test_stage2_fp16_h800_rewrite_suite_runner -v
```

---

## 4. 解决效果

### 4.1 H800/sm90 真实 full-engine TensorCore 加速

修复后在 H800 GPU4 上重跑:

```bash
cd ${V2X_ROOT}
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib:$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path 2>/dev/null):${LD_LIBRARY_PATH:-}
${V2X_DATA_ROOT}/tvm310/bin/python \
  scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode full-engine-group-conv-rewrite \
  --gpu 4 \
  --reps 30 \
  --full-reps 30 \
  --raw-dir ${V2X_DATA_ROOT}/s2_tvm/fp16_rewrite_suite_20260629_h800_tensorcore_fix_20260629_130126 \
  --export-dir multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports \
  --cast-fp16-source \
  --group-conv-rewrite-filter all
```

结果:

| item | value |
|---|---:|
| target_arch | `sm_90` |
| default_latency_ms | `18.630492333333333` |
| rewritten_latency_ms | `12.79650124` |
| latency_delta_ms | `5.833991093333333` |
| speedup_ratio | `1.4559051715712046` |
| rewritten_wmma | `72` |
| rewritten_tvm_mma_sync | `2` |
| tensorcore_gate | `true` |

selective tensorization records:

| PrimFunc | status | wmma | tvm_mma_sync |
|---|---|---:|---:|
| `fused_conv2d12_add8_relu6` | tensorized | 36 | 1 |
| `fused_conv2d16_add8_relu6` | tensorized | 36 | 1 |

结论:

```text
59 号文档中的速度目标已完成:
counterfactual 已变成 H800/sm90 真实 rewritten full-engine measured latency。
rewritten full-engine 本身出现 wmma/tvm_mma_sync, 且 latency 明显低于 default。
```

### 4.2 输出误差

输出对比:

| output | shape | max_abs_err | mean_abs_err | mean_err/orig_mean |
|---:|---|---:|---:|---:|
| 0 | `[2, 24, 128, 256]` | 0.0 | 0.0 | 0.0 |
| 1 | `[2, 48, 64, 128]` | 0.0 | 0.0 | 0.0 |
| 2 | `[2, 128, 32, 64]` | 0.72705078125 | 0.07954635471105576 | 0.1895030289888382 |

解释:

```text
速度/TensorCore gate 已闭合。
但 output2 drift 仍明显, 不能直接写成 AP-safe route。
下一阶段必须做 TVM rewritten backbone/subnet -> head/postprocess/AP eval bridge。
```

### 4.3 final gate 状态

已运行:

```bash
python3 scripts/stage2_fp16_h800_gate_check.py \
  --export-dir multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports
```

最新 gate:

| check | status | 说明 |
|---|---|---|
| `h800_suite_preflight` | pass | H800/sm90 环境通过 |
| `self_test_no_tvm_mapping` | pass | no-TVM 映射自测通过 |
| `group_conv_accum_compare` | pass | H800/sm90 诊断通过 |
| `rewritten_1x1_full_engine` | fail | 1x1 不是加速解, 不作为 final speed solution |
| `group_conv_rewritten_full_engine` | pass | H800/sm90 真实 full-engine TensorCore speed gate 已通过 |
| `fp16_rewritten_ap_safe` | missing | 缺 TVM rewritten backbone/subnet 到 head/postprocess/AP eval bridge |

总状态:

```text
status=incomplete
```

原因:

```text
速度/TensorCore gate 已闭合。
AP-safe gate 未闭合。
```

---

## 5. 必须避免的误写

不能写:

```text
FP16 AP-safe 已解决。
rewritten full-engine 已可直接扩展 original60。
当前 latency 是完整 perception pipeline latency。
当前结果是 RSU 物理设备端到端速度。
output2 drift 可以忽略。
```

可以写:

```text
H800/sm90 lhc_07 FP16 full-engine TensorCore speed gate 已完成。
FP16 之前加速不明显的直接原因是 default full-engine lowering 没有进入主要耗时 group conv 的 TensorCore fast path。
通过 group conv TIR replacement + selective MatmulTensorization, lhc_07 真实 full-engine latency 从 18.6305 ms 降到 12.7965 ms, speedup 1.4559x。
当前 AP-safe 仍未验证, output2 drift 需要下一阶段收口。
```

---

## 6. 下一阶段计划

### P0: 冻结当前速度证据

1. 将 `fp16_lhc07_full_engine_group_conv_rewrite_latest.json/md` 作为 lhc_07 FP16 speed/TensorCore 证据。
2. 在任何总表或报告中明确:
   - `backend=h800_tvm`;
   - `scope=backbone/subnet full-engine`;
   - `full_network_claim=false`;
   - `ap_safe=false/unknown`;
   - `output2_drift_present=true`。
3. 不要再用旧 sm89 结果或旧 `wmma=0` H800 结果覆盖 latest。

### P1: 建立 TVM rewritten AP bridge

当前缺口:

```text
已有 PyTorch true-FP16 AP eval, 但它不能证明 TVM rewritten full-engine AP-safe。
必须把 TVM rewritten backbone/subnet output 接入 head/postprocess/AP eval。
```

执行步骤:

1. 阅读并复用 INT8 real activation bridge:
   ```text
   scripts/stage2_h800_native_int8_real_activation_bridge.py
   scripts/stage2_native_int8_ap_bulk_pipeline.py
   ```
2. 为 FP16 rewritten route 建立同类 bridge:
   - PyTorch pipeline 负责 dataset/head/postprocess/AP;
   - backbone/subnet 特征由 TVM rewritten full-engine 输出替换;
   - 保存 prediction 数量、score/box 分布、AP30/AP50/AP70。
3. 对 `lhc_07` 先跑 5 sample AP smoke。
4. 再跑 full-val 或更大 sample。

验收:

```text
prediction 非空;
AP30/AP50/AP70 有合理非零趋势;
与 default FP16/PyTorch baseline 的 head/postprocess 分布可解释;
raw artifact/stdout/stderr/traceback 全部保存。
```

### P2: output2 drift 收口

当前 output2:

```text
mean_err/orig_mean = 0.1895030289888382
```

排查顺序:

1. 判断 output2 drift 是否实际影响 AP。
2. 如果 AP 可接受, 标记为 speed/AP-safe route, 记录 drift 可接受边界。
3. 如果 AP 不可接受, 回到 TIR 层:
   - default-like reduction/order;
   - FP32 accumulation hybrid;
   - partial callsite rewrite;
   - 只启用 AP-safe callsite;
   - 对 residual/add/relu downstream 放大路径做 intermediate debug。

### P3: 扩展 original60 的前置条件

只有 lhc_07 同时满足:

```text
speed_gate=true
tensorcore_gate=true
AP-safe=true 或 AP degradation 可接受且有解释
```

才能扩展 original60。

扩展时每个配置至少记录:

| 字段 | 要求 |
|---|---|
| config_id | original60 label |
| default_latency_ms | H800/sm90 实测 |
| rewritten_latency_ms | H800/sm90 实测 |
| speedup_ratio | rewritten 相对 default |
| wmma/tvm_mma_sync | rewritten TIR 计数 |
| output_error | output shape/max_abs_err/mean_abs_err |
| AP30/AP50/AP70 | rewritten AP eval |
| raw artifact | TIR/ONNX/.so/stdout/stderr/manifest |

### P4: INT8 与 FP16 总结口径

当前 FP16 与 INT8 的差异仍应这样写:

```text
FP16: default full-engine lowering 没把主要耗时 group conv 送入 TensorCore fast path; 现在 lhc_07 speed gate 已通过。
INT8: 旧问题是 QDQ-heavy/float32-heavy route; native INT8 route 已推进, 但 AP/scale/head-postprocess 仍需收口。
```

---

## 7. 下一阶段 /goal 建议

```text
/goal 基于 65_6_29 交接文档继续推进 FP16 APSafe 收口。固定口径: H800/sm90 TVM backbone/subnet full-engine latency 已证明 lhc_07 group-conv rewritten TensorCore speed gate 通过, 不再重复证明能否加速。下一阶段目标是建立 TVM rewritten backbone/subnet output 到 head/postprocess/AP eval 的 bridge, 对 lhc_07 完成 FP16 rewritten AP smoke/full-val, 输出 AP30/AP50/AP70、prediction 数量、score/box 分布、output0/1/2 error 和 raw artifact。若 AP 不可接受, 不停止, 继续定位 output2 drift: default-like reduction/order、FP32 accumulation hybrid、partial AP-safe callsite rewrite、residual/add/relu downstream 放大路径。验收条件: rewritten prediction 非空, AP30/AP50/AP70 趋势合理或明确不可接受且有根因, 所有失败保存 stdout/stderr/traceback/TIR/ONNX/.so/manifest。
```

