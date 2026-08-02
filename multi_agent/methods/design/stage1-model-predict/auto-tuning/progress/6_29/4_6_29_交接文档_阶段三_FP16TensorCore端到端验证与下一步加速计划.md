# 59_6_29 交接文档: 阶段三 FP16 TensorCore 端到端验证与下一步加速计划

日期: 2026-06-29

本交接文档用于清空上下文后继续推进 `original60` 三精度实测中 FP16 加速问题。当前最重要结论是: **TVM/H800 可以稳定生成 FP16 tensor-core matmul, lhc_07 的真实 1x1 convblock 也可以稳定进入 tensor-core path, 但完整 ONNX backbone/subnet 默认 FP16 engine 仍没有进入 tensor-core fast path, 因此 FP16 推理加速不明显。**

---

## 0. 启动初期必须阅读的文档

新窗口启动后先读:

1. `multi_agent/methods/design/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md`
2. `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_forced_lowering_next_step_review_latest.md`
3. `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_convblock_tensorcore_and_engine_probe_latest.md`
4. `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_e2e_tensorcore_validation_latest.md`
5. `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_e2e_all1x1_tensorcore_validation_latest.md`
6. `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_fp32_int8_speed_root_cause_explanation_20260629.md`
7. `docs/superpowers/plans/2026-06-29-lhc07-fp16-e2e-tensorcore-validation.md`

核心脚本:

- `scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`

---

## 1. 固定规则和口径

1. latency 指标统一使用 `ms` 对外汇报；内部 raw 可以保留 `us`。
2. 当前速度口径是 **H800 + TVM backbone/subnet compiled module**, 输入为 `spatial_features`, 输出为 backbone/subnet multiscale feature。不得写成完整 perception pipeline latency, 也不得写成真实 RSU 物理边缘设备绝对速度。
3. 只有完整 engine 自身出现 `wmma/tvm_mma_sync` 并完成 build/run/latency/energy 实测后, 才能写成 `tensor-core FP16 full-engine measured`。
4. 当前 `lhc_07` 端到端结果是 **measured counterfactual**, 不是单一 rewritten full-engine binary。可以用于证明“FP16 tensor-core lowering 会影响完整算法推理时间”, 但不能替代真实 rewritten engine latency row。
5. 不得把 default FP16 engine 的 `.so` 写成 tensor-core engine。当前 default full engine `.so` 没有 `wmma/tvm_mma_sync` 证据。
6. 遇到 build/eval/import/TVM/SSH 问题时, 保存 raw artifact、command、stdout/stderr、traceback、TIR/script, 做最小复现后继续推进；不要因为单点失败直接停止。

---

## 2. 当前已完成的实验证据

### 2.1 TVM/H800 FP16 matmul tensor-core 正例已打通

已证明:

- dlight `MatmulTensorization` 可以在 H800 上生成 FP16 tensor-core TIR。
- 正确 route 是:

```python
import tvm.s_tir.dlight as dl
from tvm.s_tir.dlight.gpu.matmul import MatmulTensorization

with target, tvm.transform.PassContext(opt_level=3):
    scheduled = dl.ApplyDefaultSchedule(MatmulTensorization())(mod)
with target, tvm.transform.PassContext(opt_level=3):
    ex = tvm.compile(scheduled, target=target)
```

注意:

- 直接 `MatmulTensorization().apply(func, target, False)` 可以生成 WMMA TIR, 但 raw `sch.mod` 编译可能因 `global_symbol`/entry wrapping 失败。
- 稳定方式是对完整 Relax/IRModule 用 `dl.ApplyDefaultSchedule(MatmulTensorization())`。

### 2.2 lhc_07 单个真实 1x1 convblock 已稳定 tensor-core

目标 block:

- label: `lhc_07`
- block: `fused_conv2d11_add7_relu5`
- 原始 conv: input NCHW=`[2,48,64,128]`, weight OIHW=`[256,48,1,1]`, output NCHW=`[2,256,64,128]`
- 等价 matmul: `M=16384, K=48, N=256`

关键结果:

| 变体 | 重复 | wmma | tvm_mma_sync | latency 量级 | 结论 |
|---|---:|---:|---:|---:|---|
| pure matmul K=48 | 3/3 success | 36 | 1 | 约 13.4-13.9 us | 稳定 tensor-core |
| matmul + bias/relu K=48 | 3/3 success | 36 | 1 | 约 13.4-14.0 us | 融合后仍稳定 |
| K=64 padded | 3/3 success | 36 | 1 | 约 14.5 us | padding 不是必要条件 |

结论:

- `K=48` 本身不是 blocker。
- 之前失败的直接 probe 是因为没有先做 `LegalizeOps -> AnnotateTIROpPattern -> FuseOps -> FuseTIR`。dlight 必须看到合法 fuse 后的 matmul TIR pattern 才能 tensorize。

证据:

- `exports/fp16_lhc07_convblock_tensorcore_and_engine_probe_latest.md`
- `exports/fp16_lhc07_convblock_tensorcore_and_engine_probe_latest.json`

### 2.3 完整 lhc_07 默认 FP16 engine 已导出, 但不是 tensor-core engine

输入 ONNX:

- `${V2X_DATA_ROOT}/s2_tvm/models/lhc_07_backbone.onnx`
- shape_dict: `spatial_features=[2,64,128,256]`

完整默认 engine:

- route: `default_relax_build`
- status: success
- latency_mean_us: `19013.5739` 或近似 `18.98-19.01 ms`
- TIR/tensor-core 证据: `wmma=0`, `tvm_mma_sync=0`
- export: `${V2X_DATA_ROOT}/s2_tvm/fp16_tensorcore_convblock_engine_20260629/lhc07_full_fp16_engine/default_relax_build.so`

结论:

- 这个 `.so` 是完整 FP16 default engine, 不是 tensor-core FP16 engine。
- 当前 full engine 仍没有自动把 NCHW conv lowering 到 dlight tensor-core matmul path。

### 2.4 lhc_07 端到端 counterfactual 验证已完成

单 block counterfactual:

| 项 | latency_mean_us | 证据 |
|---|---:|---|
| full default FP16 engine | 18974.78758 | 完整 default Relax build |
| target conv default | 20.309823 | `wmma=0`, `tvm_mma_sync=0` |
| target conv tensor-core | 13.47955 | `wmma=36`, `tvm_mma_sync=1` |
| counterfactual full engine | 18967.957307 | measured counterfactual |

单 block 影响:

- absolute_delta_ms = `0.006830273`
- speedup_ratio = `1.000360095`
- target_conv_default_share = `0.001070358`

解释: 目标 block 进入 tensor-core 后确实更快, 但单个 block 只占完整 engine 约 `0.107%`, 因此端到端影响很小。

全部 eligible 1x1 Conv counterfactual:

| 项 | 数值 |
|---|---:|
| eligible 1x1 Conv 数量 | 33 |
| covered nodes | 33 |
| failed nodes | 0 |
| full_default_latency_mean_us | 18985.37351 |
| eligible_default_sum_us | 491.644458 |
| eligible_tensorcore_sum_us | 415.643466 |
| absolute_delta_us | 76.000992 |
| absolute_delta_ms | 0.076000992 |
| counterfactual_latency_mean_ms | 18.909372518 |
| speedup_ratio | 1.004019223 |

结论:

- FP16 tensor-core lowering 对 `lhc_07` 完整 backbone/subnet latency 有可量化影响。
- 但仅替换全部 eligible stride=1/group=1/1x1 Conv 的理论收益仍只有约 `0.076 ms`、`1.004x`。
- 这说明主要耗时不在这些 1x1 Conv, 而在 3x3/group conv、layout/default lowering、fusion 或 VM/graph overhead。

证据:

- `exports/fp16_lhc07_e2e_tensorcore_validation_latest.md/json`
- `exports/fp16_lhc07_e2e_all1x1_tensorcore_validation_latest.md/json`

---

## 3. FP16 推理加速不明显的原因

目前最可信原因:

1. 当前 original60 的 FP16 latency 大多来自完整 ONNX/default Relax build 或历史 full-module MetaSchedule route。
2. 这些 route 的 dtype 虽然是 FP16, 但完整 engine 的关键算子没有进入 FP16 tensor-core fast path。
3. 完整 `lhc_07` default engine 的直接证据是 `wmma=0`, `tvm_mma_sync=0`, latency 约 `19 ms`。
4. 单独把真实 1x1 convblock 改写为 legalize/fuse 后的 matmul, 可以稳定出现 `wmma=36`, `tvm_mma_sync=1`, 说明 TVM/H800 并非不能加速 FP16。
5. all-1x1 counterfactual 显示即使全部 eligible 1x1 Conv 都进入 tensor-core, 端到端也只快约 `0.076 ms`, 因此 lhc_07 的主要耗时还在其它算子或 graph/runtime overhead。

一句话总结:

**FP16 当前不是“量化无效”, 而是“完整 backbone/subnet lowering 没有把主要耗时算子送进 tensor-core fast path”。**

---

## 4. 与 INT8 之前推理加速不明显的区别

### 4.1 相同点

FP16 和 INT8 的共同表象是:

- precision 字段变了, 但完整推理没有明显变快。
- 不能只看 `precision=fp16/int8` 就认为使用了硬件快路径。
- 必须审查 TIR/CUDA/runtime 证据:
  - FP16 看 `wmma/tvm_mma_sync`。
  - INT8 看 native int8/dp4a/int8 conv route, 同时排除 QDQ/float32-heavy lowering。

### 4.2 不同点

| 维度 | FP16 当前问题 | INT8 之前问题 |
|---|---|---|
| 核心 blocker | 完整 ONNX NCHW conv lowering 没有自动进入 tensor-core matmul path | 早期 route 是 QDQ-heavy / float32-heavy lowering, 或 native INT8 scale/requant/layout 未收口 |
| 是否已有局部硬件快路径正例 | 有, lhc_07 1x1 convblock 已稳定 `wmma/tvm_mma_sync` | 有, native INT8 build/run 和部分 full ONNX latency/energy route 已打通 |
| 端到端主要缺口 | 把局部 tensor-core lowering 真正嵌入 full engine, 并覆盖主要耗时算子 | AP 数值一致性、scale/requant/residual Add/output dequant、head/postprocess bridge |
| 当前 latency 解释 | FP16 default engine 没有 tensor-core, all-1x1 tensor-core counterfactual 收益小 | INT8 早期慢主要是 QDQ/float32-heavy, 后续 native route 速度可跑但 AP/数值仍需收口 |
| 下一步工程重点 | graph/TIR rewrite + layout/fusion + 3x3/group conv fast path | scale-aware native INT8 AP full-val + 数值对齐 |

结论:

- 两者同属于“没有真正吃到目标硬件快路径或快路径没有覆盖主要耗时”的问题。
- 但 FP16 现在更偏 lowering/schedule/fusion 工程。
- INT8 更偏 native quant route 的数值一致性和 AP bridge 工程。

---

## 5. 下一步如何实现端到端图里的真实加速

目标不是继续证明单个 matmul 能 tensor-core, 而是把加速写进完整 `lhc_07` graph 或 TIR。

### P0: 冻结当前证据和禁用错误结论

必须保持:

- 当前完整 default FP16 engine 不得标为 tensor-core engine。
- 当前 all-1x1 结果是 measured counterfactual, 不是真实 rewritten full-engine measured latency。
- 总表中若需要写结论, 应写:
  - `fp16_dtype_present`
  - `full_engine_tensorcore_evidence_absent`
  - `1x1_tensorcore_counterfactual_positive_but_small`

### P1: 做真实 full-engine rewrite v1

优先从 1x1 Conv rewrite 开始, 因为已经证明所有 eligible 1x1 shapes 都能 tensor-core。

Rewrite 规则:

```text
Conv(NCHW, W[O,I,1,1], bias)
  -> Transpose NCHW->NHWC
  -> Reshape [N*H*W, I]
  -> MatMul [N*H*W, I] x [I, O]
  -> Add bias if exists
  -> Reshape [N,H,W,O]
  -> Transpose NHWC->NCHW
```

Gate:

1. rewritten ONNX 能被 TVM import。
2. rewritten full engine 能 build/run。
3. full engine scheduled TIR 或 generated module 有 `wmma/tvm_mma_sync`。
4. 同一 random input 下, 原始 default engine 与 rewritten engine 输出 shape 一致。
5. 输出 max/mean abs error 在 FP16 可解释范围内。
6. rewritten full engine latency_ms 小于 default full engine, 且收益与 all-1x1 counterfactual 同向。

如果 rewritten ONNX 失败, 必须保存:

- rewritten ONNX
- shape inference 结果
- TVM import traceback
- failing node name
- fallback plan

### P2: 处理真正主要耗时: 3x3/group conv

all-1x1 只带来约 `1.004x`, 说明大头不是 1x1。下一步必须盘点并处理:

1. group conv 3x3: `group=32`, weight shape 如 `[256,8,3,3]` 等。
2. stride=2 downsample/group conv。
3. layout transform 与 memory movement。
4. full engine VM overhead 与 fusion boundary。

候选路线:

- 对 group conv 做 explicit im2col + batched/grouped matmul lowering, 再尝试 tensor-core。
- 对 NCHW/NHWC layout 做全图统一, 避免每个 rewrite 引入大量 transpose。
- 先选最大 FLOP/latency 的 group conv block 做 single-block tensor-core/im2col 正例, 再纳入 full-engine counterfactual。
- 若 im2col memory overhead 太大, 需要比较 direct conv schedule、implicit GEMM、Winograd/TC route 三种可能。

### P3: 真实 rewritten full-engine latency/energy

只有 P1/P2 成功后, 才启动:

1. `lhc_07` rewritten full-engine latency 重测。
2. `lhc_07` rewritten full-engine energy 重测。
3. 输出误差 gate。
4. 若有 checkpoint/AP route, 再接 PyTorch head/postprocess 做 AP smoke。

目标产物:

- `exports/fp16_lhc07_rewritten_full_engine_tensorcore_validation_latest.md/json`
- H800 raw dir:
  `${V2X_DATA_ROOT}/s2_tvm/fp16_tensorcore_convblock_engine_20260629/lhc07_rewritten_full_engine_*`
- rewritten `.so`
- scheduled TIR dump
- output parity report
- latency/energy report

---

## 6. 建议的下一阶段执行顺序

1. 基于当前脚本新增 `rewrite-onnx-1x1` 模式, 只处理 stride=1/group=1/1x1 Conv。
2. 在 H800 上生成 `lhc_07_backbone_1x1_matmul_rewrite.onnx`。
3. 用 TVM import rewritten ONNX, 先 default build/run, 再尝试 dlight `MatmulTensorization`。
4. 若 full module dlight 仍因非 matmul PrimFunc assert 失败, 不要全模块套 `MatmulTensorization`; 改为:
   - import 后 extract/fuse TIR
   - 只对 matmul PrimFunc 应用 `MatmulTensorization`
   - 对其它 PrimFunc 用 `Fallback/GeneralReduction`
5. 做原始 default full engine vs rewritten full engine 输出误差对比。
6. 做 rewritten full engine latency。
7. 若 1x1 rewritten full engine 只带来小收益, 进入 group conv/im2col tensor-core 正例。
8. 只有真实 rewritten full engine 有 `wmma/tvm_mma_sync` 且 latency 下降, 才考虑向 original60 推广。

---

## 7. 当前关键命令

H800 环境:

```bash
cd ${V2X_ROOT}
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${V2X_DATA_ROOT}/tvm310/lib/python3.10/site-packages/tvm/lib:$(cat ${V2X_DATA_ROOT}/tvm_nvlibs.path 2>/dev/null):${LD_LIBRARY_PATH:-}
```

单 block / full default / all-1x1 验证命令:

```bash
CUDA_VISIBLE_DEVICES=4 ${V2X_DATA_ROOT}/tvm310/bin/python \
  scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode e2e-all1x1 \
  --gpu 4 \
  --reps 100 \
  --full-reps 20
```

本地同步结果:

```bash
rsync -av -e 'ssh -p 30001 -o StrictHostKeyChecking=accept-new' \
  ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_e2e_all1x1_tensorcore_validation_latest.* \
  ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/
```

---

## 8. 下一窗口 /goal

```text
/goal 阅读并严格遵守 ${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_27/59_6_29_交接文档_阶段三_FP16TensorCore端到端验证与下一步加速计划.md。当前结论: TVM/H800 FP16 tensor-core matmul 已打通, lhc_07 真实 1x1 convblock 和全部 eligible 1x1 Conv 均可稳定产出 wmma=36/tvm_mma_sync=1; 但完整 lhc_07 default FP16 engine 仍 wmma=0/tvm_mma_sync=0, latency 约 19ms, all-1x1 measured counterfactual 仅带来约 0.076ms/1.004x 收益, 因此 FP16 加速不明显的原因是完整 backbone/subnet lowering 没有把主要耗时算子送入 tensor-core fast path。下一阶段目标: 在 lhc_07 上把 counterfactual 变成真实 rewritten full-engine 加速。执行顺序: 1) 基于 scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py 新增 rewrite-onnx-1x1 模式, 将 stride=1/group=1/kernel=1x1/pad=0 的 Conv 重写为 NCHW->NHWC->Reshape->MatMul->Add->Reshape->NCHW; 2) 在 H800 生成 rewritten ONNX 并做 shape inference; 3) TVM import/build/run rewritten full engine, 保存 rewritten ONNX、scheduled TIR、.so、stdout/stderr; 4) 对比原始 default engine 与 rewritten engine 的输出 shape/max_abs_err/mean_abs_err; 5) 实测 rewritten full-engine latency_ms, 要求 full engine 中出现 wmma/tvm_mma_sync 证据; 6) 若全模块 dlight 因非 matmul PrimFunc assert 失败, 改为只对 matmul PrimFunc 应用 MatmulTensorization, 其它 PrimFunc 走 Fallback/GeneralReduction; 7) 若 1x1 rewritten full engine 仍收益很小, 继续选择主要耗时的 3x3/group conv 做 im2col/implicit-GEMM tensor-core 正例和 counterfactual。固定规则: 当前结果是 measured counterfactual, 不得写成真实 rewritten full-engine measured latency; default FP16 engine 不能标为 tensor-core engine; latency 对外统一 ms, 口径为 H800 TVM backbone/subnet compiled module, full_network_claim=false; 遇到失败必须保存 raw artifact/traceback/TIR/ONNX 并做最小复现, 不要直接停止。
```
