# 60_6_29 交接文档: 阶段三 FP16 rewritten 1x1 full-engine 结果与 3x3/group conv 下一步

日期: 2026-06-29

本交接文档承接 `59_6_29_交接文档_阶段三_FP16TensorCore端到端验证与下一步加速计划.md`。当前已经把 all-1x1 counterfactual 推进为真实 rewritten ONNX full-engine build/run: full engine 自身可以出现 FP16 WMMA/TensorCore 证据, 但仅 rewrite 1x1 Conv 没有带来端到端加速。

## 1. 本轮必须保留的口径

1. latency 对外统一使用 `ms`。
2. 当前速度口径仍是 H800 TVM backbone/subnet compiled module, `full_network_claim=false`。
3. `fp16_lhc07_rewritten_1x1_full_engine_latest.*` 是真实 rewritten full-engine 结果, 不是 counterfactual。
4. 本轮可访问的原始 ONNX 是本地 coverage 目录 FP32 ONNX; 因 `/exdata` 当前不可写/不可用, 使用 `--cast-fp16-source` 生成 FP16-cast ONNX 后再 rewrite。不得把这个写成直接复用了 `${V2X_DATA_ROOT}/s2_tvm/models/lhc_07_backbone.onnx`。
5. 结论必须写成: 1x1 rewritten full engine 已能产生 TensorCore TIR, 但 1x1 rewrite 不是 lhc_07 的加速解; 下一步应处理主要耗时的 3x3/group conv。

## 2. 代码改动

核心脚本:

- `scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py`

新增能力:

1. `--mode rewrite-onnx-1x1`
   - 将 eligible `stride=1/group=1/kernel=1x1/pad=0` Conv 改写为:
     `NCHW -> NHWC -> Reshape -> MatMul -> Add? -> Reshape -> NCHW`
   - 保存 rewritten ONNX 和 shape inference ONNX。
   - build/run 原始 default engine 与 rewritten engine。
   - 保存 TIR、`.so`、JSON/MD、traceback。
2. `--cast-fp16-source`
   - 将可访问的 FP32 ONNX 的 float initializer 和 tensor type 转成 FP16。
   - 输入 feed 按 ONNX input dtype 生成, 当前 `spatial_features=float16`。
3. selective matmul scheduler
   - full-module `dl.ApplyDefaultSchedule(MatmulTensorization(), ...)` 会在非匹配 PrimFunc 上触发 `assert index_maps is not None`。
   - 已改为失败后自动 fallback 到 per-PrimFunc selective schedule: 只对 matmul PrimFunc 尝试 `MatmulTensorization`, 其它 PrimFunc 保留原 TIR。

环境修复:

- `tvm_venv310` 原本没有 `onnx`, 且仓库根目录下的 `onnx/` 模型目录会遮蔽 Python 包。
- 已执行:

```bash
uv pip install --python ${V2X_ROOT}/tvm_venv310/bin/python onnx==1.16.2
```

- 脚本内新增 `_avoid_local_onnx_shadow()` 防止从 repo root 运行时导入错误的 namespace `onnx`。

## 3. 最新可信结果

命令:

```bash
${V2X_ROOT}/tvm_venv310/bin/python scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode rewrite-onnx-1x1 \
  --cast-fp16-source \
  --gpu 0 \
  --full-reps 30 \
  --reps 20 \
  --onnx ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_onnx/lhc_07_backbone.onnx \
  --raw-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629
```

结果文件:

- JSON: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_1x1_full_engine_latest.json`
- MD: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_rewritten_1x1_full_engine_latest.md`
- raw dir: `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629/lhc07_full_fp16_engine_1x1_rewrite/`

关键数值:

| engine | status | wmma | tvm_mma_sync | latency_mean_ms | export |
|---|---|---:|---:|---:|---|
| original FP16-cast default | success | 0 | 0 | 38.592574846666665 | success |
| rewritten 1x1 matmul selective tensorcore | success | 468 | 13 | 42.530549300000004 | success |

输出误差:

| output | shape | max_abs_err | mean_abs_err |
|---:|---|---:|---:|
| 0 | `[2,24,128,256]` | 0.01953125 | 0.0010029313853010535 |
| 1 | `[2,48,64,128]` | 0.013671875 | 0.0009546744986437261 |
| 2 | `[2,128,32,64]` | 0.01171875 | 0.000909475318621844 |

调度记录:

- full-module dlight: failed, `AssertionError` at `MatmulTensorization.apply`, raw traceback 已保存。
- selective per-matmul: success。
- 44 个 IRModule function record 中:
  - 13 个 matmul PrimFunc tensorized。
  - 30 个 non-matmul PrimFunc 保留原 TIR。
  - 1 个非 TIR 函数跳过。

## 4. 当前结论

1. TVM/H800 并非不能在 full engine 中生成 FP16 TensorCore。rewritten full engine 自身已经有 `wmma=468`、`tvm_mma_sync=13`。
2. 仅对 1x1 Conv 做 ONNX-level MatMul rewrite 不会加速 lhc_07。30-rep 下 rewritten latency 为 `42.5305 ms`, 原始 FP16-cast default 为 `38.5926 ms`, rewritten 反而慢约 `3.938 ms`。
3. 原因是 1x1 rewrite 引入了大量 `Transpose/Reshape` 和 graph/runtime overhead, 同时主要耗时仍在未 tensorize 的 3x3/group conv。当前 scheduled TIR 仍有 `conv2d=81`。
4. 这与之前 all-1x1 counterfactual 一致: 1x1 算子本身可以进 TensorCore, 但不是 lhc_07 full-engine 的主要瓶颈。

## 4.1 追加结果: 3x3/group conv im2col TensorCore 单点正例

新增脚本模式:

```bash
${V2X_ROOT}/tvm_venv310/bin/python scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode group-conv-im2col \
  --gpu 0 \
  --reps 30 \
  --conv-candidate <candidate> \
  --raw-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629
```

已跑候选:

| candidate | implicit GEMM shape | default group conv ms | im2col TC ms | wmma | tvm_mma_sync | schedule-only delta ms | ideal compute-only delta ms | ideal speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fused_conv2d4_add10_relu6` | `G=32,M=4096,K=72,N/group=8` | 1.9251404867 | 0.0315118667 | 36 | 1 | 0.0012015000 | 1.8936286200 | 61.0926x |
| `fused_conv2d6_add10_relu6` | `G=32,M=4096,K=72,N/group=8` | 1.9217544733 | 0.0315645867 | 36 | 1 | 0.0011283067 | 1.8901898867 | 60.8832x |

结果文件:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_im2col_tensorcore_fused_conv2d4_add10_relu6.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_im2col_tensorcore_fused_conv2d4_add10_relu6.json`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_im2col_tensorcore_fused_conv2d6_add10_relu6.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_im2col_tensorcore_fused_conv2d6_add10_relu6.json`

raw artifacts:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629/lhc07_group_conv_im2col_tensorcore/fused_conv2d4_add10_relu6/`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629/lhc07_group_conv_im2col_tensorcore/fused_conv2d6_add10_relu6/`

解释:

1. 这两个主要 3x3/group conv 的 im2col/batched MatMul 都能稳定进入 TensorCore, 证据为 `wmma=36/tvm_mma_sync=1`。
2. 输出误差在 FP16 范围内, max_abs_err 约 `4.6e-6` 到 `4.8e-6`, mean_abs_err 约 `5e-7`。
3. `schedule-only delta` 很小, 只表示 im2col MatMul 默认 schedule 换成 TensorCore schedule 的增量, 不能代表整层替换收益。
4. `ideal compute-only delta` 很大, 但它是假设输入已经 im2col 且不计 NCHW restore 的理论下界。下一步必须实测 im2col materialization/restore 或做 TIR-level implicit-GEMM rewrite, 否则不能写成 full-engine 加速。

## 4.2 追加结果: 完整单层 im2col materialization + TensorCore + NCHW restore 已跑通

新增脚本模式:

```bash
${V2X_ROOT}/tvm_venv310/bin/python scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode group-conv-full-im2col \
  --gpu 0 \
  --reps 30 \
  --conv-candidate <candidate> \
  --raw-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629
```

该模式构造完整 TVM 单层路径:

```text
NCHW input
  -> call_tir(im2col_materialize)
  -> batched MatMul + bias + ReLU, selective MatmulTensorization
  -> call_tir(restore_nchw)
  -> NCHW output
```

结果:

| candidate | default group conv ms | full im2col TC ms | wmma | tvm_mma_sync | max_abs_err vs default | delta ms | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fused_conv2d4_add10_relu6` | 1.3132121467 | 0.1719978467 | 36 | 1 | 7.62939453125e-06 | 1.1412143000 | 7.6350x |
| `fused_conv2d6_add10_relu6` | 1.8013593333 | 0.1265107133 | 36 | 1 | 7.62939453125e-06 | 1.6748486200 | 14.2388x |

结果文件:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_full_im2col_tensorcore_fused_conv2d4_add10_relu6.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_full_im2col_tensorcore_fused_conv2d4_add10_relu6.json`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_full_im2col_tensorcore_fused_conv2d6_add10_relu6.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_group_conv_full_im2col_tensorcore_fused_conv2d6_add10_relu6.json`

raw artifacts:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629/lhc07_group_conv_full_im2col_tensorcore/fused_conv2d4_add10_relu6/`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629/lhc07_group_conv_full_im2col_tensorcore/fused_conv2d6_add10_relu6/`

解释:

1. 这已经不是 precomputed im2col; 单层计时包含 GPU 上的 im2col materialization 和 NCHW restore。
2. 两个主要 group conv 都明显快于 default group conv, 且 full single-layer module 自身有 `wmma=36/tvm_mma_sync=1`。
3. 这仍不是 lhc_07 full-engine latency。下一步应把这两个替换嵌入 full engine, 或至少做 full-engine measured counterfactual。
4. 当前两层单独 delta 合计约 `2.816 ms`, 但 full-engine 中可能因 kernel launch、layout、fusion、数据依赖而不同, 不能直接写成 full-engine 加速。

## 4.3 追加结果: full-engine measured counterfactual 已完成

新增脚本模式:

```bash
${V2X_ROOT}/tvm_venv310/bin/python scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode full-engine-group-conv-counterfactual \
  --cast-fp16-source \
  --gpu 0 \
  --full-reps 30 \
  --onnx ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_onnx/lhc_07_backbone.onnx \
  --raw-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629
```

结果文件:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_counterfactual_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_counterfactual_latest.json`

raw artifacts:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629/lhc07_full_engine_group_conv_counterfactual/`

关键结果:

| item | latency ms | 说明 |
|---|---:|---|
| full default engine | 32.6907901867 | FP16-cast local ONNX, `wmma=0/tvm_mma_sync=0` |
| default group conv sum | 3.1145714800 | `fused_conv2d4 + fused_conv2d6` 两层默认 group conv 单层实测合计 |
| replacement sum | 0.2985085600 | 两层完整 im2col TensorCore 单层实测合计 |
| counterfactual full engine | 29.8747272667 | `full_default - default_sum + replacement_sum` |
| counterfactual delta | 2.8160629200 | measured counterfactual, 不是真实 rewritten engine |
| counterfactual speedup | 1.0942623809x | measured counterfactual |

解释:

1. 这是 full-engine measured counterfactual, 不是 rewritten full-engine binary。
2. default full engine 仍不能标为 TensorCore engine。
3. 该结果说明: 如果两个主要 group conv 的完整单层替换能真正嵌入 full engine, lhc_07 有约 `2.816 ms / 1.094x` 的可观收益空间。
4. 下一步必须把 replacement 真正嵌入 full engine, 保存 rewritten IRModule/TIR/.so/stdout/stderr, 并对比输出 shape/max_abs_err/mean_abs_err 与 latency_ms。

## 4.4 追加结果: 真实 full-engine group-conv rewrite 已 build/run 并产生加速, 但 output2 误差仍需收口

新增脚本模式:

```bash
${V2X_ROOT}/tvm_venv310/bin/python scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py \
  --mode full-engine-group-conv-rewrite \
  --cast-fp16-source \
  --gpu 0 \
  --full-reps 30 \
  --onnx ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/artifacts/original60_onnx/lhc_07_backbone.onnx \
  --raw-dir ${V2X_ROOT}/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629
```

结果文件:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_rewrite_latest.md`
- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports/fp16_lhc07_full_engine_group_conv_rewrite_latest.json`

raw artifacts:

- `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/fp16_tensorcore_rewrite_20260629/lhc07_full_engine_group_conv_rewrite/`

代码修复:

1. `replace_records=[]` 的旧结果来自未重跑/未命中状态。按当前 legalize/fuse 后的 PrimFunc script shape 识别后, 实际命中:
   - `fused_conv2d12_add8_relu6` -> `fused_conv2d4_add10_relu6`
   - `fused_conv2d16_add8_relu6` -> `fused_conv2d6_add10_relu6`
2. same-signature TE replacement 初版失败原因:
   - `te.sum(...).astype("float16")` 触发 `Reductions are only allowed at the top level of compute`。
   - 修复为 reduction 保持顶层。
3. 第二版失败原因:
   - replacement 输出 buffer 被推断为 `float32`, 与原 `call_tir` 期望 `float16` 不一致。
   - 修复为 `out_nchw` 阶段 cast 回 `float16`。
4. 为贴近原始 TIR 和单层正例, matmul accumulation 改为 FP16 accumulation, scheduled TIR 中 WMMA accumulator 为 `float16`。

关键结果:

| item | default full engine | rewritten full engine |
|---|---:|---:|
| latency_mean_ms | 33.4367142533 | 22.2562589333 |
| wmma | 0 | 72 |
| tvm_mma_sync | 0 | 2 |
| speedup_ratio | N/A | 1.5023510624x |
| latency_delta_ms | N/A | 11.1804553200 |
| tensorcore_gate | false | true |

替换记录:

| PrimFunc | candidate | status |
|---|---|---|
| `fused_conv2d12_add8_relu6` | `fused_conv2d4_add10_relu6` | replaced + tensorized |
| `fused_conv2d16_add8_relu6` | `fused_conv2d6_add10_relu6` | replaced + tensorized |

输出对比:

| output | max_abs_err | mean_abs_err | max_err/orig_max | mean_err/orig_mean |
|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 0.7744140625 | 0.0811934769 | 0.2662861049 | 0.1934267730 |

诊断对照:

| rewrite_filter | replaced PrimFunc | latency_delta_ms | speedup | wmma/tvm_mma_sync | output2 max_abs_err | output2 mean_abs_err | output2 mean_err/orig_mean |
|---|---|---:|---:|---:|---:|---:|---:|
| `downsample` | `fused_conv2d12_add8_relu6` | 2.0454187000 | 1.0651582612x | 36/1 | 0.2521972656 | 0.0198994912 | 0.0474064499 |
| `repeated` | `fused_conv2d16_add8_relu6` | 9.3440368600 | 1.3932101786x | 36/1 | 0.6774902344 | 0.0797374994 | 0.1899581999 |
| `all` | both | 11.1804553200 | 1.5023510624x | 72/2 | 0.7744140625 | 0.0811934769 | 0.1934267730 |

解释:

1. 这是第一条真实 rewritten full-engine binary 证据, 不再是 counterfactual。
2. 速度 gate 已通过: rewritten full engine 自身有 `wmma=72/tvm_mma_sync=2`, latency 从 `33.4367 ms` 降到 `22.2563 ms`。
3. 但 correctness gate 尚未完全收口: output0/output1 完全一致, output2 有明显差异。
4. 诊断结果显示主要速度收益和主要 output2 误差都来自 `repeated` 路径: 只替换 `fused_conv2d16_add8_relu6` 已有 `9.3440 ms / 1.3932x` 收益, output2 mean relative error 约 `0.1900`; 只替换 downsample 的 mean relative error 约 `0.0474`。
5. 当前最可能原因是 `fused_conv2d16_add8_relu6` 在 full engine 中被多次 call_tir 复用; 替换单个 PrimFunc 等价于替换所有 callsite, 小的层内差异会沿后续 residual/conv 放大。下一步必须做 callsite-aware rewrite 或 AP smoke, 不能直接把该结果写成 AP-safe 的最终 FP16 engine。
6. 该结果可以作为“FP16 TensorCore lowering 会显著影响 lhc_07 backbone/subnet latency”的实测证据; 但若要写入总表或作为生产测量, 仍需完成 output2 误差解释/校准或 AP smoke。

## 5. 下一阶段计划

P0: 冻结当前 1x1 rewritten full-engine 证据。

- 不要再把 1x1 counterfactual 当作真实 engine。
- 不要把本轮 FP16-cast ONNX 误写成 `/exdata` 原始 FP16 ONNX。
- 保留当前 raw artifacts 作为正例: rewritten full engine 可以出现 WMMA, 但 1x1 不加速。

P1: 已完成主要耗时 3x3/group conv 的 im2col TensorCore 最小正例和完整单层正例。

- `fused_conv2d4_add10_relu6` 和 `fused_conv2d6_add10_relu6` 均已证明 TensorCore 可行。
- 完整单层 `NCHW -> im2col -> TensorCore MatMul -> NCHW` 已明显快于 default group conv。
- 下一步不需要继续证明 MatMul 能 tensorize, 而要把替换落到 full engine。

P2: 已完成 full-engine measured counterfactual, 并已打通真实 rewritten full-engine 加速路径; 下一步做 correctness 收口。

- 已完成 measured counterfactual: `full_default - selected_group_conv_default + selected_group_conv_full_im2col_tensorcore`。
- 已完成真实 full-engine rewrite: 替换 `fused_conv2d12_add8_relu6` 和 `fused_conv2d16_add8_relu6`, 产出 `wmma=72/tvm_mma_sync=2`, latency `22.2563 ms`。
- 已完成 one-PrimFunc filter 对照: 主要问题集中在 repeated `fused_conv2d16_add8_relu6`。
- 下一步必须解释 output2 误差: 优先做 callsite-aware rewrite 或 AP smoke, 判断误差来自多 callsite 同时替换、TensorCore FP16 accumulation 差异, 还是 im2col indexing/fusion 边界。
- 如果 correctness 无法收口, 必须保留该结果为 speed-positive/correctness-pending, 不写成 AP-safe final engine。
- 对 full engine 继续使用 selective scheduling, 不再直接 full-module MatmulTensorization。

P3: 收口标准。

- 最低成功: 已达成。至少两个主要 3x3/group conv 的完整单层 im2col TensorCore 正例, 有 TIR 证据、latency_ms、误差。
- 更高成功: 部分达成。rewritten full engine latency 小于 default FP16-cast engine, 且 full engine 自身有 `wmma/tvm_mma_sync`; 但 output2 误差仍需解释/校准。
- 若仍不能加速, 必须给出按算子 share 和 TIR lowering 的解释, 不能只停在 build 失败。

## 6. 可直接使用的 /goal

```text
/goal 阅读并严格遵守 ${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_27/59_6_29_交接文档_阶段三_FP16TensorCore端到端验证与下一步加速计划.md 和 ${V2X_ROOT}/multi_agent/methods/design/auto-tuning/progress/6_27/60_6_29_交接文档_阶段三_FP16Rewritten1x1FullEngine结果与3x3下一步.md。当前已完成: rewrite-onnx-1x1 mode 已打通; FP16-cast lhc_07 rewritten 1x1 full engine 自身出现 wmma=468/tvm_mma_sync=13, 但 latency 42.5305ms 慢于 default 38.5926ms, 说明 1x1 rewrite 不是加速解; 两个主要 3x3/group conv (`fused_conv2d4_add10_relu6`, `fused_conv2d6_add10_relu6`) 的完整单层 `NCHW -> im2col materialize -> batched MatMul TensorCore -> NCHW restore` 已完成, 均有 wmma=36/tvm_mma_sync=1, 且分别比 default group conv 快约 7.64x 和 14.24x; full-engine measured counterfactual 已完成: default full engine 32.6908ms, 替换两层后的 counterfactual 29.8747ms, delta 2.8161ms, speedup 1.0943x, 但这仍不是 rewritten full-engine binary。下一阶段目标: 将这两个完整单层替换真正嵌入 lhc_07 full engine, 产出 rewritten full engine 的 IRModule/TIR/.so/stdout/stderr/latency_ms/输出 shape/max_abs_err/mean_abs_err, 并证明 full engine 自身有 wmma/tvm_mma_sync 且 latency 小于 default。固定规则: latency 统一 ms; 口径为 H800 TVM backbone/subnet compiled module; full_network_claim=false; 不得把 counterfactual 写成真实 engine; 不得把 FP16-cast ONNX 写成 /exdata 原始 FP16 ONNX; full-module dlight assert 时必须使用 selective per-matmul/conv schedule 并保存 raw TIR/traceback; 遇到失败继续做最小复现和原因定位。
```
