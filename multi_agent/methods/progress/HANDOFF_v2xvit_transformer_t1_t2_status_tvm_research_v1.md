# 中文交接 - V2X-ViT Transformer/Attention T1-T2 当前状态与 TVM 调研 (v1, 2026-06-24)

> 清理上下文后从本文档继续。本文档合并了 `plan_transformer_into_framework_v1.md` 的 T1/T2 计划、当前真实实验结果、Stop-A 验收状态、后续停止目标，以及 TVM 加速 Transformer 的外部研究证据。

---

## 0. 新窗口冷启动结论

当前结论必须拆成两层：

1. **作为 targeted experimental path，当前代码已经可以把 V2X-ViT 的部分 Transformer attention 接到 TVM 上并得到真实端到端收益。**
   - Stop-A 最终报告：`results/attention_e2e_pq_v1.json`
   - Validator：`python scripts/phase2/attention_e2e_pq_validator.py --mode stop-a --report-json results/attention_e2e_pq_v1.json`
   - 结果：`ACCEPTABLE_E2E_SCHEMA`
   - 可接受最终 row 是 `attention-p50-int8/mixed`，但 TVM scope 是 **MSwin-only**，不是全 attention。

2. **作为通用加速框架能力，Transformer 架构尚未完整集成到 stage1/stage2 搜索框架。**
   - `framework/stage1/auto_trace.py` 仍把 `fusion_net (V2XTransformer: HMSA+MSwin, multi-agent, 不可 trace -> 不剪)` 作为 skip。
   - `framework/stage1/standard_conv_probe_plan.py` 仍把 `attention_fusion_coverage_anchor` 标为未覆盖。
   - 当前 TVM runner 是 `scripts/phase2/` 下的手写实验路径，不是通用 manifest/bridge/search 自动发现路径。

因此对用户问题“加速框架现在可不可以对 Transformer 架构加速”的回答是：

```text
可以做 V2X-ViT attention 的定向实验加速，且 MSwin-only + 50% head pruning + W8A16 TVM mixed-INT8 已通过 Stop-A。
但还不能声称通用框架已经完整支持 Transformer。T2 的 stage1 adapter/graph_scan/bridge/search 集成尚未完成。
```

还差多少：

- **速度门槛层面**：all-attention scope 已真实跑通但 speedup `1.0828x`，距离 Stop-A 速度阈值 `1.10x` 还差 `0.0172x`，约差 `1.6%` 相对门槛。
- **工程集成层面**：T2 的通用框架接入基本仍是待办；当前只有 phase2 手写 runner 与报告链路，不是 stage1 自动扫描 + bridge manifest + 搜索器闭环。
- **后端优化层面**：无剪枝无量化 FP32 TVM 是负收益；FP16 端到端有收益但 subnet 负收益，说明当前 TVM Relax 路径主要吃掉 Python/PyTorch dispatch/einsum/einops 开销，并没有证明 isolated attention kernel 本身比 PyTorch 更快。
- **适用范围层面**：本文档讨论的是当前框架准备纳入的 OpenCOOD/HEAL/V2X-ViT/AttFuse/Where2comm 等中间融合方法。对这批方法，融合前的本地感知主干主要仍是 CNN/PointPillar/BEV backbone，attention 主要集中在跨 agent `fusion_net`。但不要写成“所有 V2X 或所有 RSU 感知都没有 attention”；roadside/infrastructure-side Transformer/BEVFormer 类方法是外部反例。

---

## 1. 对照原计划的完成情况

原计划：`multi_agent/methods/design/plan_transformer_into_framework_v1.md`

| 阶段 | 原计划目标 | 当前状态 | 结论 |
|---|---|---|---|
| T0 | 拆 V2X-ViT fusion/attention 算子级瓶颈 | 已完成 | 原始 fusion_net 是瓶颈；定位到 MSwin relative_indices CPU buffer 问题和 attention 子模块分布 |
| T1-S | 实测 attention schedule/backend 可行性，先修 eager 低效 | 已完成 | `relative_indices` 改为 registered buffer 后，MSwin synthetic 从约 `279.9 ms` 降到 `30.9 ms`，约 `9.1x` |
| T1-P | 定义/实现 Transformer 剪枝轴 | 已完成到 Stop-A 可用 | 手写结构化 scanner 覆盖 HMSA/MSwin head pruning；50% attention pruning manifest 已生成 |
| T1-Q | TVM 量化/混合精度 attention e2e 验收 | 有条件完成 | MSwin-only W8A16 TVM mixed-INT8 通过 Stop-A；all-scope 低于速度门槛；FP32 TVM 负收益 |
| Stop-A | 真实 DAIR val 端到端 latency/AP row，通过 validator | 已通过，但有 scope 限定 | Validator 返回 `ACCEPTABLE_E2E_SCHEMA`；必须注明 TVM scope 是 `mswin` |
| T2 | stage1 adapter/graph_scan 扩展到 Transformer/fusion | 未完成 | 下一阶段应做 T2-A，不能把 phase2 runner 当成通用框架集成 |
| T3 | 注意力 latency/AP LUT 与代价模型 | 未开始 | 需要 T2 能吐出可搜索 knobs 后再做 |
| T4 | 接入三臂搜索并做完整 e2e Pareto | 未开始 | 依赖 T2/T3 |

---

## 2. 当前 Stop-A 真实端到端结果

报告文件：

- `results/attention_e2e_pq_v1.json`
- `results/attention_e2e_pq_v1.csv`
- `results/attention_e2e_pq_v1.md`

Validator：

```bash
python scripts/phase2/attention_e2e_pq_validator.py \
  --mode stop-a \
  --report-json results/attention_e2e_pq_v1.json
```

结果：

```text
verdict: ACCEPTABLE_E2E_SCHEMA
errors: []
warnings: []
```

| config | prune | quant/backend | TVM scope | latency p50 ms | speedup | AP50 | AP70 | dAP50 | dAP70 |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| baseline | 0% | PyTorch FP16 | none | 23.0752 | 1.0000x | 0.4501 | 0.2747 | 0.0000 | 0.0000 |
| attention-p50-fp16 | 50% | PyTorch FP16 | none | 25.0184 | 0.9223x | 0.5913 | 0.3621 | +0.1412 | +0.0874 |
| attention-p50-int8/mixed | 50% | TVM Relax int8/mixed W8A16 | MSwin-only | 20.0382 | 1.1516x | 0.591491 | 0.362288 | +0.1414 | +0.0876 |

必须解释：

- AP 不降反升不是 pruning/quantization/TVM 本身带来的。主要原因是 `attention-p50-fp16` 与 `attention-p50-int8/mixed` 使用了 500-step fine-tune 后的 checkpoint。
- 同一 fine-tune checkpoint 下，TVM mixed-INT8 相比 PyTorch FP16 的 AP 基本持平：AP50 `0.591491` vs `0.591314`，AP70 `0.362288` vs `0.362065`。
- `attention-p50-fp16` 的 latency 比 baseline 更慢，说明“剪枝后未加 TVM/量化”并不会自然加速端到端；现有收益来自 TVM mixed-INT8 + MSwin path 替换。

TVM runtime evidence：

| measurement | total TVM calls | MSwin calls | HMSA calls | fallback | compile |
|---|---:|---:|---:|---:|---|
| final Stop-A row | 16416 | 16416 | 0 | 0 | MSwin 9, HMSA 0 |

最终 row 的 caveat：

```text
TVM scope=mswin; quant_policy=w8a16;
MSwin q/k/v W8A16 in TVM with FP16 softmax/output;
HMSA TVM disabled.
```

---

## 3. all-attention scope 与无剪枝无量化对照

### 3.1 剪枝+W8A16 TVM all-attention scope

文件：`results/attention_p50_tvm_mixed_int8_measurement_all_w8a16_h800_current_v1.json`

| config | TVM scope | quant | latency p50 ms | speedup | AP50 | AP70 | runtime |
|---|---|---|---:|---:|---:|---:|---|
| attention-p50-int8/mixed | all = MSwin+HMSA | W8A16 | 21.3111 | 1.0828x | 0.590081 | 0.361700 | total 21888, MSwin 16416, HMSA 5472, fallback 0 |

结论：

- all-scope 能跑通，fallback 为 0，说明 HMSA static `[0,0]` TVM path 在 DAIR val 上机械可用。
- 但 speedup `1.0828x < 1.10x`，没有达到 Stop-A 速度门槛。
- DAIR val 的 HMSA type coverage 是 static `[0,0]`，`covers_dynamic_type_dispatch=false`；不能声称覆盖所有动态 type dispatch。

### 3.2 无剪枝、无量化，FP16 TVM all-attention

文件：

- `results/attention_fp16_tvm_no_prune_ablation_v1.json`
- `results/attention_fp16_tvm_no_prune_ablation_rerun_e2e_v1.json`

| scope | PyTorch FP16 p50 ms | TVM FP16 p50 ms | speedup |
|---|---:|---:|---:|
| e2e model forward, run1 | 15.4440 | 13.5284 | 1.1416x |
| e2e model forward, rerun | 16.6806 | 13.9680 | 1.1942x |

Subnet 对照：

| subnet | PyTorch FP16 p50 ms | TVM FP16 p50 ms | speedup |
|---|---:|---:|---:|
| MSwin | 0.326808 | 0.455272 | 0.7178x |
| HMSA static | 0.409115 | 0.476672 | 0.8583x |
| subnet sum | 0.735923 | 0.931944 | 0.7897x |

解释：

- FP16 e2e 看起来加速，但 isolated subnet 是负收益。
- 这说明当前收益更可能来自替换 PyTorch/einops/einsum/Python dispatch 的整图调用开销，而不是 TVM 单个 attention kernel 更快。
- e2e 口径是 `model(batch["ego"])`，不包含 dataloading、postprocess/NMS、AP aggregation。

### 3.3 无剪枝、无量化，FP32 TVM all-attention

文件：

- `results/attention_fp32_tvm_no_prune_ablation_v1.json`
- `results/attention_fp32_tvm_no_prune_ablation_rerun_e2e_v1.json`

| scope | PyTorch FP32 p50 ms | TVM FP32 p50 ms | speedup |
|---|---:|---:|---:|
| e2e model forward, run1 | 17.4717 | 19.4886 | 0.8965x |
| e2e model forward, rerun | 17.1333 | 19.3413 | 0.8858x |

Subnet 对照：

| subnet | PyTorch FP32 p50 ms | TVM FP32 p50 ms | speedup |
|---|---:|---:|---:|
| MSwin | 0.623898 | 0.977667 | 0.6381x |
| HMSA static | 0.665155 | 0.939881 | 0.7077x |
| subnet sum | 1.289053 | 1.917548 | 0.6722x |

解释：

- FP32 TVM 不是当前可用加速路径。
- 当前 Relax VM + 默认/dlight FP32 lowering 输给 PyTorch/cuBLAS/ATen 调度，且多模块 VM/DLPack 调用开销不可忽略。
- 后续如果要证明 TVM 对 Transformer kernel 本身有收益，需要 MetaSchedule/Ansor/DietCode 类 tuning 或 flash-attention 式 fused schedule，而不是只依赖默认 Relax build。

---

## 4. 剪枝、量化、微调方法说明

### 4.1 当前剪枝到底是什么

当前方法是手写结构化 head pruning，不是 unstructured sparsity，也不是随机删 tensor：

- scanner 枚举 V2X-ViT fusion encoder 中的 3 个 HMSA attention 和 9 个 MSwin `BaseWindowAttention`。
- 剪枝维度是 **attention head**，保留外部 embed/channel contract，manifest 中 `dim_256_preserved=true`。
- 当前 p50 manifest：
  - `method=manual_structured_attention_head_pruning`
  - `prune_rate_pct=50`
  - `hmsa_head=3`
  - `mswin_head=9`
  - `hmsa_keep_heads=4`
  - `mswin_keep_heads={ws4:8, ws8:4, ws16:2}`
- 这类策略的依据是 Transformer head redundancy 的历史工作；典型参考包括 Michel et al. “Are Sixteen Heads Really Better than One?” 和 Voita et al. “Analyzing Multi-Head Self-Attention”，二者都指出部分 attention heads 可结构化移除且不必然造成大幅精度下降。

### 4.2 当前量化是什么

当前最终接受的是 TVM Relax mixed-INT8 W8A16：

- MSwin q/k/v weight int8，activation FP16，TVM 内 dequant。
- softmax/output 保持 FP16。
- HMSA 在最终 row 中关闭；all-scope row 里 HMSA static `[0,0]` 可跑，但速度不达标。
- 全程没有使用 TensorRT；最终报告里 `quant_backend=TVM Relax int8/mixed`。

### 4.3 当前微调是什么

最终 row 使用：

```text
models/v2xvit_attention_t1/attention_p50_shortft_steps500_lr0.0001_seed20260624.pth
```

微调设置：

```text
500 steps, lr=1e-4, seed=20260624, all params
```

需要提醒：

- AP 大幅高于 baseline 主要来自这个 fine-tune checkpoint 与当前评测协议组合，不应在论文/报告中写成“剪枝量化提升精度”。
- 更严谨的表述应是：“在同一 DAIR val 协议下，50% head pruning 经短微调后 AP 恢复并高于该 baseline row；TVM W8A16 对该 checkpoint 的 AP 基本无损。”

---

## 5. 为什么 latency 有时不变或 TVM 反而慢

当前结果显示三件事：

1. `attention-p50-fp16` 剪枝后端到端速度反而慢于 baseline：25.0184 ms vs 23.0752 ms。
   - 原因可能是结构化 head pruning 保持外部 256 dim contract，减少的 head 内部计算没有充分转化为下游 dense GEMM 或 kernel launch 数减少。
   - fine-tune checkpoint 本身不应增加推理后处理；latency 口径是 model forward，不含 postprocess/NMS。速度变慢更可能来自 forward path 的实际 kernel/shape/dispatch 变化，而不是后处理时间。

2. FP16 no-prune TVM e2e 有收益，但 subnet 负收益。
   - 说明 TVM 替换整段 forward path 后减少了一部分 PyTorch/einops/einsum/Python overhead。
   - 不能据此断言 TVM isolated attention kernel 已经优于 PyTorch。

3. FP32 no-prune TVM e2e 与 subnet 都负收益。
   - 说明默认 Relax FP32 对当前 attention shape 不够好。
   - 如果后续目标是“完整 TVM 加速实现”，必须引入 tuned schedule，而不能只复用当前默认 lowering。

---

## 6. 当前框架范围与 Transformer 集成反思

### 6.1 目前要集成的方法范围

当前讨论不要泛化为“所有 V2X 算法”。本项目当前要集成的是已有 stage1/stage2 框架覆盖或准备覆盖的一批 OpenCOOD/HEAL 风格中间融合方法：

- V2X-ViT / V2XTransformer
- AttFuse
- Where2comm
- F-Cooper / MaxFusion
- V2VNet / DiscoNet 等 routing/fusion baseline

在这批方法中，融合前的感知计算通常是：

```text
agent/RSU local encoder = PointPillar / CNN / BEV backbone
cross-agent reasoning = fusion_net
```

因此下一步框架集成的合理优先级是：

1. 保持 stage1 对 dense CNN/PointPillar/BEV backbone 的 channel pruning / quant scanning。
2. 把 `fusion_net` 从 skipped subgraph 升级为 attention/fusion sidecar manifest。
3. 在 sidecar 里记录不同 fusion 家族的结构化 knobs，而不是把所有 attention 都当成同一个 `nn.MultiheadAttention`。

必须避免的表述：

```text
错误：RSU 侧感知计算都没有 attention，attention 只用于车端融合。
```

更严谨的表述：

```text
在当前 OpenCOOD/HEAL 中间融合基线中，RSU/vehicle 的本地感知 encoder 多为 CNN/PointPillar/BEV backbone，
attention 主要集中在跨 agent fusion_net。但更广义的 roadside/infrastructure perception 已经存在
Transformer/BEVFormer/cross-attention 方法，所以通用框架需要预留 local perception attention 的扫描入口。
```

### 6.2 Transformer 接入 stage1/stage2 的主要问题

当前障碍不是“找不到 attention”，而是现有框架的搜索对象和 Transformer/fusion 的真实结构不匹配：

| 问题 | 影响 |
|---|---|
| stage1 原搜索轴是 CNN channel | Transformer 需要新增 `head`, `window`, `sequence/token`, `attention_tvm_scope`, `quant_policy` 等 knobs |
| 多 agent 输入动态 | `record_len`, agent 数、pairwise transform、mask、type dispatch 让 trace/export 不稳定 |
| V2X-ViT attention 是自定义结构 | HMSA 有 per-type `q/k/v/a_linears` 和 relation tensor；MSwin 有 window attention 和 relative position，不能简单按标准 MHA 处理 |
| 剪枝要保持外部 contract | 当前 head pruning 必须同步裁 q/k/v/out/relation tensor，同时保持外部 256 dim，不等于普通 channel pruning |
| TVM runner 仍是 phase2 手写路径 | 目前靠 monkey-patch/runner 替换，不是 stage1 manifest -> bridge -> search -> build 的闭环 |
| latency 归因容易混淆 | subnet 负收益但 e2e 有收益时，可能只是减少 PyTorch/einops/Python dispatch，不代表 TVM kernel 本身更快 |

因此 T2A 的真实目标不是“一步支持全部 Transformer”，而是先建立稳定 schema：

```text
skipped fusion_net
  -> typed attention/fusion sidecar
  -> per-family knobs
  -> bridge/coupling score
  -> optional backend candidates: pytorch / tvm_fp16 / tvm_w8a16 / later tuned schedule
```

### 6.3 为什么 TensorRT 通常能加速 Transformer

TensorRT 对 NVIDIA GPU 推理是专门工程化过的后端，Transformer 加速来自几个机制叠加：

- 标准 MHA pattern / `IAttention` 能触发 fused attention，把 `QK^T -> softmax -> AV` 合并到专门 kernel。
- builder 会按目标 GPU、shape、precision 搜索 tactic，选择实际最快实现。
- FP16/INT8/FP8 等低精度路径天然对接 Tensor Core。
- layer fusion 减少 kernel launch 和显存读写。
- 标准 Transformer/LLM 图在 TensorRT 中有成熟优化路径；NVIDIA 文档明确列出 fused attention、transformer pattern、precision/tactic selection。

参考：

- NVIDIA TensorRT Fused Attention: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-fused-attention.html
- NVIDIA TensorRT performance optimization / tactic selection: https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- NVIDIA TensorRT documentation overview: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html

这解释了为什么 TRT 在标准 Transformer 上经常比 eager PyTorch 快：它不是只把 matmul 换一个 backend，而是做了 fused attention、tactic tuning、低精度 Tensor Core 和图级 fusion。

### 6.4 为什么当前 TVM 加速效果差

当前 TVM 结果差，不代表 TVM 不能加速 Transformer，而是当前实现还处在“最小可跑/局部替换”阶段：

1. **没有真正 fused full attention。**
   - Stop-A 接受 row 是 `MSwin q/k/v W8A16 in TVM + FP16 softmax/output`。
   - HMSA 在最终 row 关闭；all-scope 能跑但速度不到门槛。
   - 这与 TRT fused MHA 不同。

2. **没有 shape-specific auto-tuning。**
   - 当前主要依赖 Relax/default/dlight lowering。
   - FP32 no-prune 端到端和 subnet 都负收益，说明默认 schedule 对当前 shape 不够好。
   - 后续若要证明 TVM kernel 本身更快，需要 MetaSchedule/Ansor/DietCode 或 flash-attention 风格 schedule。

3. **调用粒度太碎。**
   - 当前是 Python monkey-patch 局部替换，每个 attention 子模块调用 TVM runtime。
   - VM/DLPack/dispatch overhead 会吃掉小 kernel 的收益。

4. **PyTorch/cuBLAS/ATen FP16 baseline 已经很强。**
   - no-prune FP16 e2e 有收益，但 isolated subnet 是负收益。
   - 这说明 TVM 可能减少了一些 Python/einops/einsum dispatch，而不是 isolated attention kernel 本身更优。

5. **剪枝没有自然变成硬件收益。**
   - 当前 head pruning 保持外部 256 dim contract。
   - 如果内部 head 变少但 GEMM shape、kernel launch、输出 projection 没有同步形成更优 schedule，端到端 latency 可能不降甚至上升。

下一步判断标准：

- 如果要声称“TVM 对 Transformer attention 有 kernel 级加速”，必须给出 subnet 级 positive speedup，并说明 fused/scheduled kernel 覆盖了哪些 op。
- 如果只看到 e2e positive、subnet negative，只能写成“减少了框架/dispatch/局部图开销”，不能写成“TVM attention kernel 更快”。
- 如果继续采用 MSwin-only policy，必须明确 HMSA 仍是 fallback 或 future tuned schedule 对象。

---

## 7. TVM 加速 Transformer 的外部研究调研

结论：**存在 TVM/TVM ecosystem 加速 Transformer 的研究和系统，但加速效果依赖 tuned schedule、shape specialization、dynamic-shape tuning、融合策略和量化。默认 TVM lowering 不保证快。**

| 来源 | 对象 |  reported acceleration / evidence | 对本项目的启发 |
|---|---|---|---|
| Apache TVM blog: [Deploying Quantized BERT with TVM and PyTorch](https://tvm.apache.org/2020/07/14/bert-pytorch-tvm) | BERT GPU inference | tuned TVM 后整体从 PyTorch `6.5-7 ms` 到约 `6.2 ms`，整体约 `5%-10%` speedup；部分 op 从 `0.2 ms` 到 `0.13-0.15 ms` | TVM 可以加速 Transformer，但收益不是自动巨大，需要 tuning；5%-10% 与当前 Stop-A `1.15x` 是同量级 |
| Apache TVM blog: [Introduction to Auto-scheduler](https://tvm.apache.org/2021/03/03/intro-auto-scheduler) | 多模型 FP32 single-batch，包括 BERT | auto-scheduler 相比 AutoTVM 有 `1.02x-8.95x` 提升；BERT-base@GPU 是极端案例之一 | 当前默认 Relax/FP32 慢，下一步应考虑 MetaSchedule/Ansor 风格 auto-tuning |
| MLSys 2022 DietCode: [Automatic Optimization for Dynamic Tensor Programs](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html) | BERT dynamic-shape workloads | auto-scheduling 时间最多降 `5.88x`；runtime 最多比 Ansor 好 `69.5%`、比 vendor library 好 `18.6%` | V2X-ViT 多 agent/窗口 attention 有 shape specialization 与动态分支问题，DietCode 类方法比单一静态默认 schedule 更贴近需求 |
| Relax paper: [Relax: Composable Abstractions for End-to-End Dynamic Machine Learning](https://arxiv.org/pdf/2311.02103) | LLM/Transformer deployment | Relax/TVM stack 报告 LLM decode latency 可有明显收益，例如 NVIDIA decode latency reduction up to `27%`，AMD 7900 XTX batch-1 optimized performance up to `1.50x`，移动端相对 llama.cpp throughput up to `55%` better | 说明 Relax 可承载 Transformer/LLM，但高收益来自 operator fusion、partial library lowering、CUDA graph/offload 等组合，不是当前最小实现 |
| Apache TVM tutorial: [Optimize Large Language Model](https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html) | TinyLlama/Relax workflow | 官方教程展示 LLM 从模型导入到 Relax IRModule、优化、build、deploy 的流程；不是对比 benchmark | 证明 TVM 对 Transformer/LLM 是官方支持方向；本项目应把 phase2 runner 收敛成 framework manifest/build 流程 |
| Apache TVM docs: [End-to-End Optimize Model](https://tvm.apache.org/docs/how_to/tutorials/e2e_opt_model.html) | 通用 e2e model optimization | 官方文档提示默认 e2e optimization may not suit complex models | 与本项目 FP32/default Relax 负收益一致：复杂 Transformer 需要专门 schedule |
| MLC LLM docs: [Introduction](https://llm.mlc.ai/docs/get_started/introduction) 与 WebLLM paper [arXiv 2412.15803](https://arxiv.org/html/2412.15803v2) | TVM/MLC LLM deployment | MLC 是 TVM ecosystem 的高性能 LLM deployment engine；WebLLM 可保持 MLC-LLM native decoding throughput 的约 `71%-80%` | 说明 TVM ecosystem 对 Transformer 规模模型有成熟路线，但我们当前 V2X attention 还没有用到 MLC/flash/fused attention 级优化 |

额外 pruning 参考：

- Michel, Levy, Neubig, [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)
- Voita et al., [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418)

---

## 8. 下一阶段建议停止目标

下一阶段不要泛泛“开始 T2”。建议拆成两个明确 stop。

### Stop-T1-AllAttention-Decision

目标：对全 attention TVM scope 做最终判定，是继续优化 HMSA，还是正式采用 MSwin-only policy。

验收：

- 重跑或优化 `tvm_scope=all, quant_policy=w8a16`。
- 若 all-scope speedup `>=1.10x` 且 AP guardrail 通过，则升级最终 row 为 all-scope。
- 若仍 `<1.10x`，则形成负证据报告：HMSA TVM static path 在当前实现下拖慢，最终策略保留 MSwin-only，并把 HMSA 留给 T3 tuned schedule。
- 必须保留 runtime stats：MSwin/HMSA call count、compile count、fallback count。

### Stop-T2A-Framework-Integration

目标：把当前 phase2 手写 attention 能力接入 stage1/stage2 框架，而不是继续堆实验脚本。

验收：

- `framework/stage1/adapters.py` / `auto_trace.py` 不再把 V2XTransformer 简单 skip；至少能生成 fusion/attention sidecar manifest。
- `framework/stage1/graph_scan.py` 能识别并输出 Transformer knobs：
  - `hmsa_head`
  - `mswin_head`
  - `attention_tvm_scope`
  - `quant_policy`
  - 可选：`encoder_depth`, `embed_dim`, `ffn_dim`
- AttFuse/Where2comm 这类非 V2X-ViT fusion 至少保持 typed family 信息，不得误标为 HMSA/MSwin；若实现 AttFuse scanner，应输出 `scaled_dot_product_attention` / `bmm_softmax_bmm` 类 knob，而不是套用 head pruning。
- `framework/stage1_bridge.py` 能给 attention knobs 计算 coupling score，并标出 buildability/align cliff。
- 搜索入口能读到这些 knobs，但 T2A 不强制要求跑完整搜索 Pareto。
- 测试至少覆盖：
  - manifest 中出现 attention knobs；
  - skip reason 不再把 V2XTransformer 直接判为 unsupported；
  - AttFuse 仍可被识别为 attention/fusion family，且不会被错误归类为 V2X-ViT HMSA/MSwin；
  - bridge 对 attention knobs 有稳定 schema；
  - 旧 conv-only 模型不被破坏。

---

## 9. 推荐双 agent 分工

新窗口建议拉起两个 agent：

1. **Agent-A / Executor**
   - 负责执行 Stop-T1-AllAttention-Decision 或 Stop-T2A-Framework-Integration。
   - 如果做 T2A，优先改 stage1 adapter/graph_scan/bridge schema 和测试。
   - 不负责最终验收结论，避免自证。

2. **Agent-B / Critical Reviewer**
   - 独立检查实验结果与表述。
   - 必查项：
     - 是否把 subnet 当 e2e；
     - 是否把 MSwin-only 写成 all-attention；
     - 是否把 fine-tune AP 提升写成 pruning/quantization 提升；
     - 是否混用了旧 H800 baseline 与当前 baseline；
     - TVM runtime calls/fallback 是否真实；
     - latency 是否包含 postprocess/NMS，若不包含必须写明。
     - 是否把当前 OpenCOOD/HEAL 中间融合范围错误外推为“所有 V2X/RSU 感知都没有 attention”。
     - 是否把 TRT 的 fused attention/tactic/低精度能力和当前 TVM 的局部 q/k/v patch 混为同等后端能力。

---

## 10. 关键文件索引

设计/交接：

- `multi_agent/methods/design/plan_transformer_into_framework_v1.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_tvm_v3.md`
- `multi_agent/methods/progress/HANDOFF_v2xvit_attention_t1_pq_stopa_goal_v6.md`
- 本文档：`multi_agent/methods/progress/HANDOFF_v2xvit_transformer_t1_t2_status_tvm_research_v1.md`

最终报告：

- `results/attention_e2e_pq_v1.json`
- `results/attention_e2e_pq_v1.csv`
- `results/attention_e2e_pq_v1.md`

关键 measurement：

- `results/attention_p50_tvm_mixed_int8_measurement_mswin_w8a16_h800_current_v1.json`
- `results/attention_p50_tvm_mixed_int8_measurement_all_w8a16_h800_current_v1.json`
- `results/attention_fp16_tvm_no_prune_ablation_v1.json`
- `results/attention_fp16_tvm_no_prune_ablation_rerun_e2e_v1.json`
- `results/attention_fp32_tvm_no_prune_ablation_v1.json`
- `results/attention_fp32_tvm_no_prune_ablation_rerun_e2e_v1.json`
- `results/attention_e2e_checkpoint_eval_h800_current_v1.json`

关键脚本：

- `scripts/phase2/attention_e2e_pq_validator.py`
- `scripts/phase2/attention_final_acceptance_report.py`
- `scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py`
- `scripts/phase2/attention_fp16_tvm_no_prune_ablation.py`
- `scripts/phase2/t1_attention_e2e_pq.py`
- `scripts/phase2/t1_attention_tvm_bench.py`

模型：

- `models/v2xvit_attention_t1/attention_p50_shortft_steps500_lr0.0001_seed20260624.pth`
- `models/v2xvit_attention_t1/attention_p50_shortft_steps500_manifest_v1.json`

---

## 11. 新窗口建议 prompt

```text
阅读 multi_agent/methods/progress/HANDOFF_v2xvit_transformer_t1_t2_status_tvm_research_v1.md。
先复核 Stop-A 当前状态和 TVM scope caveat，然后按文档中的 Stop-T1-AllAttention-Decision 或 Stop-T2A-Framework-Integration 继续。
拉起两个 agent：一个执行实现，一个批判性验收实验/表述；不得把 subnet 当 e2e，不得把 MSwin-only 写成 all-attention，不得把 fine-tune AP 提升归因给 pruning/quantization/TVM。
同时注意：当前框架范围是 OpenCOOD/HEAL 中间融合方法，融合前本地感知主干主要是 CNN/PointPillar/BEV backbone；不要外推成所有 V2X/RSU 感知都没有 attention。解释 TRT/TVM 差异时必须区分 TRT fused attention/tactic/低精度能力与当前 TVM 局部 q/k/v patch。
```
