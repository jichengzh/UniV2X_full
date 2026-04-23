# 剪枝 × 硬件可配置维度交互调研

> 场景: UniV2X 子任务 1.2 已完成可配置剪枝 (结构化通道剪枝 + L1/Taylor/FPGM 重要性准则 + 分层剪枝率搜索)。本文聚焦"剪枝方案确定后，硬件侧还剩哪些可调维度，二者如何反向耦合"。
>
> 已有调研覆盖: Phi (ISCA'25, SNN 层次化稀疏)、SOFA (MICRO'24, 动态稀疏跨阶段 tiling)、TASDER (MLSys'25, 非结构化稀疏映射到结构化加速器)、MicroScopiQ (ISCA'25, outlier-aware 量化+剪枝) 等——本文不再重复，只在对比时引用。

---

## 1. 稀疏模式 ↔ 硬件支持

### 1.1 Ampere/Hopper Sparse Tensor Core (2:4)

**核心事实 (NVIDIA 官方):**

- **2:4 半结构化稀疏**: 每 4 个连续权重中恰有 2 个为零，50% 固定稀疏率。Sparse Tensor Core 仅处理非零值 + metadata (2-bit/非零元素指示原位置)，理论 **2× 计算吞吐、2× 权重带宽**。
- **硬件代际**:
  - Ampere (A100, SM 8.0, 2020): 首次引入稀疏 Tensor Core，FP16/BF16/INT8/TF32 (TF32 下降级为 1:2)。
  - Hopper (H100, SM 9.0, 2022): 第四代 Tensor Core，稀疏吞吐在 FP8 上达到 **3957.8 TFLOPS**，新增 FP8 E4M3/E5M2 稀疏支持。
  - Ada/Blackwell (SM 8.9/10.x/12.x): cuSPARSELt 0.9+ 已扩展支持 SM 8.9、9.0、10.0、10.1、11.0、12.0、12.1。
- **Hopper 并未引入 "4:8"**: 4:8 是软件层模式 (SparseGPT/Wanda 支持的训练端模式)，硬件侧 Sparse Tensor Core 仍仅支持 2:4。4:8 必须在运行时解压或投影到 2:4 才能获得硬件加速。

**实测加速比 (关键):**

| 场景 | dense 基线 | 2:4 稀疏加速 | 来源 |
|------|-----------|------------|------|
| OPT-175B CUTLASS matmul, bs=2048 | cuBLAS dense | **1.54×–1.79×** (单层) | SparseGPT NeurIPS'23 |
| Llama 2:4 + FP8, vLLM E2E | BF16 dense | **1.7× prefill / 1.5× decode** | RedHat Neural Magic 2024 |
| Llama 2:4 SparseGPT + FP8 (PyTorch) | FP16 dense | **1.27× E2E serving** | HPC-AI benchmark 2025 |
| CUTLASS 3.6 FP8 sparse kernel | cuBLAS BF16 | **2.5× peak** | RedHat 2024-12 |
| Search engine int8 + 2:4 | dense int8 | **1.4×** E2E | NVIDIA blog (PNR 案例) |
| Jetson Orin DLA RetinaNet R-34 | 密集 DLA | **1.36×** (低功耗模式更高) | NVIDIA DLA repo |

**几个容易误读的点:**
1. **单层 2× 只是 peak**。大 GEMM (高算术强度) 能逼近 1.8×；端到端因为 attention、softmax、规范化、KV cache 等未加速部分被拉低到 1.2–1.7×。
2. **仅对权重维度 (通常 A 矩阵) 稀疏有效**——激活侧稀疏无法直接利用 Sparse Tensor Core。
3. **对齐要求严格**: 2:4 模式要求在 **K 维度 (GEMM 的 reduction 轴)** 上每连续 4 个元素中有 2 个零。权重存储布局必须配合；若权重被转置 (BN 等)，则需要重新 prune 或 pattern 转换。
4. **TensorRT 触发条件**: `trtexec --sparsity=enable/force`；FP16 要求 I/O 通道对齐到 16 (稠密是 8)；INT8 要求对齐到 32，且 "输出通道 > 128" 才倾向选择稀疏 kernel (否则工作量不够摊销 metadata 开销)。

### 1.2 Block-sparse 库

GPU 上三种主流稀疏库，面向不同稀疏粒度：

| 库 | 稀疏格式 | 典型粒度 | 代表场景 | 切换到稠密的稀疏率阈值 |
|----|---------|---------|---------|------------------|
| **cuSPARSELt** (NVIDIA) | 固定 2:4 压缩 (CSR-like metadata) | 元素级 | Ampere/Hopper 半结构化 | 固定 50% |
| **cuSPARSE Block-SpMM** | Blocked-ELL | 16/32/64/128 块 | 科学计算 + 稀疏 DNN | V100 块=32 时密度 <40%；A100 块=32 时 <50% 即可快于 cuBLAS |
| **Magicube / cuTeSpMM** | Block + 向量化 tile | 变块 | 图神经网络 SpMM | TCU-Synergy 模型预测；高非零结构规则性才有效 |
| **Sputnik** (Google) | CSR + row-swizzle | 行级 | 高稀疏率 DNN (80-95%) | 90% 稀疏率下几何平均 3.4×，95% 稀疏率 5.4× |
| **SparseRT** (arXiv 2020) | 非结构化代码生成 | 元素级 | 90%+ 稀疏率 1×1 Conv/FC | 只在 >90% 有效 |
| **Megablocks** | BCSR + 转置索引 | 128×128 块 | MoE dropless | 与负载倾斜正相关 |
| **MatX / nvmath-python UST** (NVIDIA 2026) | 通用 UST DSL (COO/CSR/CSC/BSR 可定制) | 可配置 | Ampere+ 统一 sparse 接口 | — |

**关键经验法则**:
- **<50% 稀疏率**: 基本唯一可行路径是 2:4 (cuSPARSELt)。Blocked-ELL 与 Sputnik 在该稀疏率下几乎无收益。
- **50-90% 稀疏率**: Block-sparse 开始有效，但必须块对齐 (16/32)。非结构化稀疏仍无法跑赢 cuBLAS。
- **>90% 稀疏率**: Sputnik / SparseRT / TorchSparse++ 这类 CUDA Core 稀疏 kernel 才显著获利，但它们 **绕开了 Tensor Core**，无法与 Tensor Core 峰值竞争——这是为什么科学计算稀疏 kernel 在 DNN 场景 "看起来快但实际输给 cuBLAS" 的主因。

### 1.3 非结构化稀疏的硬件瓶颈

为什么 NVIDIA GPU 几乎不能从随机 50-70% 非结构化稀疏获得加速？

**根本原因 (Sputnik/SparseRT 系列论文反复强调):**

1. **Warp divergence (束内分歧)**: GPU 以 32 线程为 warp 执行。若每行非零数可变，同一 warp 内线程工作量不均，SIMT 必须等最慢线程完成——load imbalance within a warp。
2. **非对齐 load-store**: GPU 通过 memory coalescing 把 32 个线程的访问合并为少量 transaction。稀疏场景的 indirect addressing (column_index 解引用) 破坏连续性，出现 uncoalesced access，带宽利用率可跌到 1/6 到 1/8 (同算法 6× 性能差，见 Dev.to 2026 基准)。
3. **Tensor Core 难以启用**: Tensor Core 需要固定 shape 的小矩阵乘 (MMA 16×16×16 等)。非结构化稀疏无法在 register tile 层面保证 2 个零正好落在 4 连续位——只能回退 CUDA Core，性能上限降到 FP16 峰值的 1/8–1/16。
4. **Metadata 开销反噬**: 每 pruned 元素需要 1–2 bit 索引。稀疏率 <90% 时，metadata load 与解压开销抵消计算节省。
5. **L2 cache 压力**: B 矩阵 (稠密激活) 按间接索引访问，cache line 局部性差。

**结论 (工业界共识)**: 非结构化稀疏适合 **CPU (DeepSparse) 和定制 ASIC**；在 GPU 上除非稀疏率 >90%，否则应该强制约束到 2:4 或块稀疏。

### 1.4 结构化 vs 半结构化 vs 非结构化 对比

| 维度 | 通道/Filter 剪枝 (结构化) | 2:4 N:M (半结构化) | 非结构化 (随机) |
|------|--------------|---------|--------|
| 稀疏粒度 | 整通道/滤波器 | 4 连续元素内的 2 个 | 单元素 |
| 精度损失 (同稀疏率) | 最大 | 中 | 最小 |
| GPU kernel | 直接复用 cuBLAS (通道数变小) | cuSPARSELt / CUTLASS sparse | 需要 Sputnik/SparseRT 等定制 |
| 实测加速 (50%) | **完全正比于通道压缩** (线性) | **1.4-1.8× matmul** | **几乎零加速** (甚至更慢) |
| 内存占用 | 正比通道数 | 压缩 2× + metadata 2-bit/非零 | CSR/CSC 存储膨胀 |
| TensorRT 开箱 | ✅ | ✅ (`--sparsity=enable`) | ❌ (需要 cuSPARSELt 手动集成) |
| DLA (Jetson Orin) | ✅ | ✅ (1.36× 实测) | ❌ |
| 精度对齐要求 | 通道数 8/16/32 倍数 | K 维 4 对齐 | 无 |
| 动态 shape 友好 | ✅ | △ (重新压缩需要时间) | ❌ |
| 量化叠加 | 直接 + PTQ/QAT | ✅ FP8/INT8 成熟 (2.5× peak) | 通常不叠加 |

**UniV2X 实操建议**: 通道剪枝 → 得到稠密小模型；再对特定大层 (backbone、BEV decoder) 尝试 2:4。两者叠加时，先剪通道 (改变 shape)，再对保留通道跑 2:4 mask。

---

## 2. 剪枝率 ↔ 内核选择交叉点

### 2.1 Sparse kernel 切换阈值 (crossover point)

**cuSPARSE Block-SpMM 实测 (NVIDIA blog 2021):**

> "The speedup ratio compared to cuBLAS is nearly linear to the sparsity... When the block size is 32, the kernel is faster than cuBLAS if the density is <40% on Volta and <50% on Ampere."

意味着:
- **Volta (V100)**: 要获得任何收益，block 密度必须 <40% (即块级稀疏率 >60%)。
- **Ampere (A100)**: 放宽到 50% 密度，得益于 Tensor Core 和更好的 L2。
- 块大小越大 (64/128)，crossover 门槛越低，但模型精度越敏感。

**Sputnik 的 "中等稀疏率" 观察:**

> "Existing GPU kernels for sparse linear algebra are primarily optimized for 99%+ sparsity. With the moderate levels (50-90%) found in DNNs, these kernels are not able to outperform their dense counterparts."

因此，Sputnik 的核心贡献是针对 70-95% 稀疏率优化。

**决策树 (给定剪枝率 s):**

```
s < 0.3  → 使用 dense + 通道剪枝 (降低通道数比稀疏化划算)
0.3 ≤ s < 0.5 → 2:4 固定模式 (cuSPARSELt)
0.5 ≤ s < 0.7 → Block-SpMM (块=32/64) 或 2:4 + 通道剪枝组合
0.7 ≤ s < 0.9 → Sputnik / cuTeSpMM (CUDA core + 可选 TCU)
s ≥ 0.9  → SparseRT / DeepSparse (CPU) / 定制 ASIC
```

### 2.2 cuSPARSELt algorithm selection

cuSPARSELt 将算子执行设计为 **Plan (描述) → Algorithm Selection (自动调优) → Execute** 的三阶段：

```
cusparseLtMatmulAlgSelectionInit(..., CUSPARSELT_MATMUL_ALG_DEFAULT)
cusparseLtMatmulSearch()    // 试跑 N 种 algorithm ID，找最快
cusparseLtMatmulAlgSetAttribute(..., CUSPARSELT_MATMUL_ALG_CONFIG_ID, &best_alg)
cusparseLtMatmulPlanInit()
```

**关键属性:**
- `CUSPARSELT_MATMUL_SEARCH_ITERATIONS`: 每个 alg 试跑次数 (默认 10)。
- `CUSPARSELT_MATMUL_ALG_CONFIG_ID`: 枚举 alg ID (实际可选数取决于 SM 架构、数据类型、M/N/K、epilogue)。
- 支持 **Split-K** (大 K 的小 M/N matmul 拆分)。
- 支持 **epilogue 融合** (activation+bias+scaling)——对应 TensorRT 的 `Convolution+BN+ReLU` 融合模式。

**与 cuBLASLt 的对比 (设计同构):**
- cuBLASLt: dense GEMM 的 heuristic → algorithm id → descriptor-based 执行。
- cuSPARSELt: 复用了同样范式，额外封装 prune/compress 步骤。
- 实务: 在小 M/N (如 BEV decoder 的小 batch 推理) 上，`MatmulSearch` 是必需的；不同 alg 性能差异可达 30-50%。cuSPARSELt 0.7+ 已改进 search 效率并去除内部内存分配。

**算法选择 × 剪枝率的交叉影响:**
- cuSPARSELt 只在 50% (2:4) 工作，稀疏率不可调。
- 若模型真实稀疏率 <50%，直接用 dense kernel；若 >50%，多余的零被浪费在 metadata 里。
- UniV2X 通道剪枝场景: 若层 L 的 L1 准则给出 40% 通道可剪，剪完后得到窄通道 dense 模型——是否再叠加 2:4？关键看该层 K 维剩余大小是否足够 (K < 64 时 2:4 收益骤降)。

---

## 3. 剪枝 ↔ 调度 / 流水线

### 3.1 动态剪枝的流水影响

**三种动态剪枝的硬件调度特征:**

| 方法 | 剪枝粒度 | 决策时机 | GPU 友好度 |
|------|--------|--------|-----------|
| DynamicViT (NeurIPS'21) | token 级 | 预测模块每 N 层选 top-k | ⭐⭐⭐ 仍 dense GEMM，但 seq_len 变小 |
| SkipNet / BlockDrop | 层级 (skip residual) | gating network 决定跳过 | ⭐⭐ 分支依赖打断 graph fusion |
| Mixture-of-Depths (MoD, 2024) | token + 层 | router 分配 compute 预算 | ⭐⭐ 需要 gather/scatter |
| CoDeNet / SBNet | 空间稀疏 | 输入驱动 mask | ⭐ 非规则 mask 回到非结构化稀疏 |

**DynamicViT 的关键观察:**
- Token 剪枝后序列缩短 → 剩余 token 仍组成 **dense tensor**，attention/FFN 直接用 cuBLAS，无需稀疏 kernel。
- 这是实现 "dynamic 但 hardware-friendly" 的关键——把动态稀疏转化为动态 shape。
- 实测 66% token 剪枝 → 31-37% FLOPs 降低、**40% 吞吐提升**，精度损失 <0.5%。

**分支预测 / 控制流开销:**
- GPU 不支持硬件分支预测 (除了 warp-level predicate)。
- 动态决策 (if x > threshold) 在 kernel 内造成 **warp divergence**。
- 解决: 把决策放在 **两个 kernel 之间** (先跑 scorer kernel 产出 mask，再用 mask 做 compact 再跑下一个 kernel)，让每个 kernel 内部依然 uniform。
- TensorRT 对动态 shape 的支持通过 **optimization profile**: min/opt/max shape。每个 profile 会 build 独立 engine tensor，profile 切换有 overhead (TensorRT 官方文档明确警告)。

### 3.2 MoE 与 head pruning 的流水

**Megablocks (dropless MoE, 2022):**
- 核心: 把 MoE 重构为单一 **block-sparse GEMM** (BCSR 格式，128×128 块)。
- 相对 Tutel (带 capacity factor + padding/drop) 训练加速 **1.4×**；相对 Megatron-LM dense baseline **2.4×**。
- 硬件维度: block size 必须 tune (太小不足以饱和 Tensor Core；太大会限制 expert 粒度)。经验 128 最优。
- Hopper 版推荐 `grouped GEMM` 路径 (利用 TMA + WGMMA)。

**多流 (multi-stream) 执行 (TensorRT-LLM PR #11520, 2026):**
- MoE 和 MLA attention 已引入 **multi-stream orchestration**: shared expert 在 auxiliary CUDA stream 执行，与 routed expert 并行。
- 必须用 begin/end/wait stream marker 显式同步。
- 对 head pruning 场景的意义: 剪掉一部分头后，**剩余 heads 的 batched matmul 仍然序列化**——若 heads 异构 (如 GQA、DuoAttention retrieval vs streaming)，可以分配到不同 stream 并行。
- DuoAttention (arXiv 2410.10819): retrieval heads (full KV cache) + streaming heads (constant KV) 两套，chunked prefill 天然适合 dual-stream。

**Head 剪枝与 BEV encoder 的思考:**
- UniV2X 的 BEV decoder 多头自注意力 (MHA) 若剪掉 25% 头，直接在 QKV projection 维度缩减。
- 此时 attention matmul shape 变为 `(batch, heads_kept, seq, head_dim)` — 依然是 batched dense GEMM。
- 加速比: FLOPs 降 25% → 延迟降 ~20% (考虑 memory-bound 层和 softmax 开销)。
- 若想进一步并行剩余 heads，需要手写 CUDA kernel 或 FlashAttention variant。

---

## 4. 剪枝 ↔ 内存管理

### 4.1 稀疏存储格式的内存足迹

| 格式 | 存储 (K × N 矩阵，稀疏率 s) | 适合场景 |
|------|----------------------|---------|
| Dense (baseline) | K × N × bw | 密度 > 50% |
| CSR (compressed sparse row) | (1-s)·KN·bw + (1-s)·KN·4 + (K+1)·4 | 行不规则非结构化 |
| CSC | (1-s)·KN·bw + (1-s)·KN·4 + (N+1)·4 | 列不规则 |
| ELL (ELLPACK) | max_nnz_per_row × K × (bw+4) | 行 nnz 较均匀 |
| Blocked-ELL | 适合块对齐 | block GEMM |
| BCSR (块 CSR) | 块数 × block_bw + 块索引 | Megablocks / block-sparse DNN |
| 2:4 packed | (KN/2)·bw + KN·0.25 (2-bit metadata) | Sparse Tensor Core |
| WCSR (AsyncSparse 2026) | 窗口压缩 | Hopper 异步 kernel |

**关键观察**:
- 2:4 存储减半 + 少量 metadata，**权重存储固定节省 ~44% (fp16)**。
- CSR 在稀疏率 <70% 时实际比 dense 更大 (index 膨胀)。
- BCSR 块越大 metadata 比例越小，但内部零填充比例越大，实际节省空间有非单调曲线。

### 4.2 剪枝后激活 / workspace 收缩

**通道剪枝对内存峰值的三层影响:**

1. **权重内存**: 直接线性缩减 (剪 30% → 节省 30%)。
2. **激活 tensor**: 通道数变小 → activation shape (H×W×C) 的 C 变小 → **前向 workspace 线性缩减**。
3. **TensorRT workspace**: TensorRT builder 为每个候选 kernel 预留 workspace；通道数变小后有更多小 kernel 可选，workspace 总峰值一般也下降。

**陷阱**:
- TensorRT implicit padding (官方文档确认): 若通道数不对齐 (如 127)，会 pad 到下一个对齐边界 (128 for INT8)，**padding 不增加计算但增加内存**。剪得太不规则反而扩大内存峰值。
- 建议剪枝率搜索时，把 **对齐约束 (channels % alignment == 0)** 作为硬约束而非软约束。

**KV cache 与剪枝交互 (LLM 场景):**
- 剪枝后 hidden_dim 变小 → KV cache 按 `(batch × seq × layers × 2 × n_heads × head_dim)` 线性缩小。
- 但 GQA/MQA、DuoAttention 等 head 共享进一步减少 KV cache — 可能与 head pruning **冲突**: 若已经 GQA 共享，再剪 head 收益递减。
- 最新做法: 先做 GQA (跨 head 合并)，再对 `n_kv_heads` 维剪枝；DHA 论文 (proceedings.com/079017) 展示该路线。

### 4.3 稀疏权重预加载与 re-allocation

- cuSPARSELt 要求 `cusparseLtSpMMAPrune` + `cusparseLtSpMMACompress` **一次性预处理**，之后 compressed weight 固定 shape。
- 这与 dynamic pruning 不兼容 — 若每 token 重算 mask，需要 re-compress，单次 compress 耗时远高于 matmul 本身。
- 实际部署: static 2:4 只做一次 (compile time)；dynamic 必须用 unstructured mask + MaskedGEMM kernel (无硬件加速)。

---

## 5. 剪枝 ↔ 异构加速

### 5.1 DLA 稀疏支持 (NVIDIA Jetson Orin)

**官方 DLA-SW 仓库 (github.com/NVIDIA/Deep-Learning-Accelerator-SW) 明确:**

- ✅ Orin DLA **支持 2:4 结构化稀疏**；RetinaNet-ResNet34 实测 **1.36×** 加速 (INT8)。
- ❌ 不支持非结构化稀疏，也不支持其他 N:M 模式。
- 规律: "The more math-bound a layer in dense operation, the higher the expected dense→sparse speedup"。
- 低功耗模式下 DLA TOP/Byte 比降低 → 更多层从 memory-bound 转为 math-bound → 稀疏收益更高 (15W 模式下 74% INT8 TOPS 来自 DLA)。

**Jetson AGX Orin 用户实测反馈** (developer forum 2024-01):
- 某用户报告 "Sparsity does not provide any speedup for TensorRT on DLA" — 定位为 memory-bound 小模型，层已经 bandwidth-limited，稀疏计算并非瓶颈。
- 启示: 稀疏化 **不能缓解 bandwidth bound**——需要配合量化 (INT8→INT4) 或激活剪枝。

### 5.2 剪枝决策对 DLA offload 可行性的影响

DLA 相对 GPU iGPU 更挑剔:

| 剪枝操作 | 对 DLA offload 的影响 |
|---------|------------------|
| 通道数变成奇数/非 32 倍数 | DLA compiler 可能失败 → fallback 到 GPU |
| 插入 reshape/transpose (剪枝形状不一致) | 断 DLA 连续子图 → 部分 offload 带来多次 DLA-GPU 往返 |
| 2:4 稀疏启用 | 多层 math-bound conv 受益，memory-bound 层无收益 |
| Per-channel 量化 | Orin DLA 支持 per-channel INT8 和 sparsity，但混用 kernel 路径受限 |
| 深度可分离卷积剪枝 | DWConv 在 DLA 上本来就 memory-bound，剪枝收益小 |

**UniV2X 建议**: 若目标包括 Orin DLA offload，剪枝搜索空间应强约束 `channels % 32 == 0`，并在 candidate 评估时调用 `trtexec --useDLACore=0 --allowGPUFallback` 实测有多少层真落到 DLA。

### 5.3 FPGA / ASIC 稀疏加速器

**专用架构 (学术/产业):**

| 加速器 | 稀疏支持 | 关键机制 | 适配的剪枝方案 |
|-------|--------|--------|-----------|
| **SIGMA** (HPCA'20, GT) | 任意稀疏率 + 不规则 GEMM | Flex-DPE + Forwarding Adder Network (FAN) | 随机非结构化 (10-90%) |
| **Cambricon-S / X** | Step-function 稀疏 | sparse index-based MAC | 非结构化 + L1 |
| **Phi** (ISCA'25) | 模式化层次稀疏 | 动态稀疏模式生成 | SNN outlier 减少后的模式 |
| **SIGMA-like FPGA (CVPR'23W)** | 固定块 + systolic 对齐 | 剪枝与 systolic array Nin×Nout 对齐 | 通道剪枝 rounded to array size |
| **TASDER** (MLSys'25) | 非结构化 → 结构化近似 | 分配律拆分为多个 2:4 序列 | 非结构化 prune + 结构化 HW |
| **SPADE / SparTA** | Pattern DSL + kernel gen | 规则稀疏 pattern encoding | 任意 + compiler 自动切 |
| **MicroScopiQ** (ISCA'25) | outlier + microscaling | 多精度 INT PE + ReCoN NoC | outlier-aware pruning |

**对软件剪枝的反向指导:**
- **SIGMA 路线 (flexible)**: 剪枝可任意，硬件吃任意稀疏——但面积大。
- **Cambricon-X 路线 (structured)**: 要求剪枝产出规则稀疏 (固定稀疏率 per group)。
- **TASDER 路线 (bridge)**: 保留非结构化算法灵活性，运行时近似分解为 N 个结构化 pass，把 GPU Sparse Tensor Core 当作 8:32 或更高比率的"块稀疏"近似装置。

---

## 6. 硬件感知剪枝搜索

### 6.1 经典方法

| 方法 | 硬件信号 | 搜索机制 | 关键贡献 |
|------|-------|--------|--------|
| **AMC** (ECCV'18, MIT Song Han) | 端侧真实 latency (mobile) | DDPG RL + continuous action | 首个在手机上以 latency 为 reward 做分层剪枝率搜索 |
| **NetAdapt** (ECCV'18, Google) | 目标设备 latency | 迭代 prune + measure | 不依赖 RL，工业工具链友好 |
| **MetaPruning** (ICCV'19) | latency lookup table | PruningNet 生成权重 + 进化搜索 | 不需重训，可在 Titan Xp latency 约束下快速搜 |
| **AutoSlim** (2019) | FLOPs proxy | slimmable supernet | 一次训练覆盖多宽度 |
| **HALP** (NeurIPS'22, NVIDIA) | latency LUT per filter | augmented knapsack + saliency × latency | ResNet-50 在 GPU 上 **1.6× 吞吐** + 0.3% acc 提升 |
| **LightPrune** (ICCV'25) | 可微分 latency predictor | 端到端梯度优化 | 不用反复 profile，预测延迟 |
| **PuRL** (AutoML'20) | sparsity 本身 | DQN (比 AMC MDP 更简单) | ResNet-50 80% 稀疏，Top-1 75.37% |

**LUT (Look-Up Table) 路线的价值**:
所有上述方法都依赖 "**把每个候选层配置 → 真实 latency**" 的表。UniV2X 可以沿用: 预先对 (通道数, 稀疏模式, 量化精度) 三元组测 TensorRT 实测延迟，建表供搜索器查询。该表本身就是 1.3 "硬件搜索" 的知识图。

### 6.2 LLM 剪枝 (SparseGPT / Wanda / Sheared)

**SparseGPT (ICML'23):**
- 一次性无需训练，OBS (Optimal Brain Surgeon) 思想对 Hessian 近似求解。
- 支持非结构化、2:4、4:8 三种。
- 175B OPT 模型 **<4.5 小时** 剪到 50-60% 稀疏，几乎无 perplexity 损失。
- 实测 GPU 加速 (CUTLASS 2:4): **1.54-1.79×** 单 matmul。

**Wanda (ICLR'24):**
- 极简: 重要性 = `|w| × ||x||` (激活感知)。不需要反向传播或二阶信息。
- LLaMA-2-7B 2:4 Wanda perplexity 11.02 (vs dense 5.12，非结构化 50% 6.42)。
- 代码支持 CUTLASS 与 cuSPARSELt 双后端切换。

**Sheared LLaMA (ICLR'24):**
- 目标形状剪枝 (targeted): 从 LLaMA2-7B 剪到 1.3B/2.7B 固定架构。
- 结构化: 删层、删 head、删 intermediate dim、删 hidden dim。
- 只用 50B tokens (原始预训练的 3%) 达到 SOTA 3B 性能。
- **对硬件的价值**: 产出的是 **dense 小模型**，立即可用 cuBLAS + 常规 TensorRT，不依赖稀疏 kernel。

**三者对比 (对硬件部署的含义):**

| 方法 | 输出模型形态 | 部署栈 | 加速实际来源 |
|------|----------|------|----------|
| SparseGPT 50% unstructured | 稀疏矩阵 | DeepSparse (CPU) / Sputnik | 带宽 & 存储 |
| SparseGPT 2:4 | 2:4 packed | cuSPARSELt / CUTLASS / vLLM | Sparse Tensor Core |
| Wanda 2:4 | 2:4 packed | 同上 | 同上 |
| Sheared LLaMA | Dense (小) | cuBLAS / TensorRT-LLM | 纯 shape 缩减 |

### 6.3 联合剪枝 + 量化

**APQ (CVPR'20, MIT):**
- 搜索空间: architecture × channel pruning × mixed-precision quant。
- 核心: **quantization-aware accuracy predictor**。先训 FP32 predictor (便宜，大量样本)，再 transfer 到 INT8 predictor (少量 QAT 样本)。
- 对比分离式 (ProxylessNAS + AMC + HAQ): 同样 latency 下 **精度高 2.3%、GPU-hours 少 600×**。
- 使用 hardware latency LUT 驱动进化搜索。

**HALP + QAT 扩展** (NVIDIA 后续工作): 把 latency 预测扩展到 int8 kernel (不同于 fp16 kernel 的 LUT)。

**QD-BEV (ICCV'23)**: BEV 检测的量化 + distillation 联合；剪枝部分通常是预处理。

**UPAQ (unified pruning + quantization) 系列**: 统一可微分搜索，使用 Gumbel softmax 选 bit-width + channel mask。

**FGMP (NVIDIA arXiv 2025)**: FP4/FP8 block 级混合精度，与 VMAC datapath 协同。**稀疏可作为正交维度叠加**，但还未开源集成。

---

## 7. 关键洞察与交互矩阵

### 7.1 十条核心洞察

1. **2:4 稀疏是目前 GPU 上唯一"免费"的稀疏**——其他模式要么无加速，要么需要库集成。但它固定 50%，无法调节。
2. **通道剪枝 vs N:M 不是竞争关系**——应串联: 先通道剪枝得到窄 dense 模型 (shape 线性省)，再对足够大的层叠加 2:4 (额外 1.4-1.8×)。
3. **剪枝率非线性映射到加速**。<30% 用稠密 + 小通道；30-50% 用 2:4；50-70% block-sparse；>70% 学术库 (Sputnik)；>90% CPU/ASIC。
4. **通道数对齐 (8/16/32) 是硬约束**，不是软建议。TensorRT 会隐式 pad，打破剪枝节省的计算与内存。
5. **Sparse kernel 切换的临界点受 K 维影响**——K 小时 metadata 开销相对变大，cuSPARSELt 可能比 cuBLAS 还慢。小 GEMM 不要强行开稀疏。
6. **DLA 支持 2:4，但只对 math-bound 层有效**。memory-bound 层 (小卷积、DWConv) 开 sparsity 等于白开。剪枝搜索应先识别 math-bound layer。
7. **动态剪枝首选 token 级 (shape-reducing)**，避免 GPU warp divergence。DynamicViT 范式把"动态稀疏"转成"动态 shape"，是最友好的形式。
8. **LUT-driven 搜索 (HALP/LightPrune) 是工程上最稳的路线**——把"搜索空间"与"硬件 profile"解耦，搜索器看到的是纯函数。
9. **cuSPARSELt 的 algorithm search 不可省略**。相同 2:4 不同 alg ID 性能差 30-50%；Plan 建立时做一次 MatmulSearch 就能用很久。
10. **LLM 剪枝落地的真实速度通常低于论文预期**——SparseGPT/Wanda 报告的 1.5-1.8× matmul 加速，到 vLLM E2E 只剩 1.2-1.7× (attention、softmax 未稀疏；KV cache 仍然稠密)。

### 7.2 剪枝 × 硬件维度交互矩阵

| 剪枝维度\硬件维度 | Kernel 选择 | 内存布局 | 调度/流 | 融合 | DLA offload | 对齐约束 |
|----------------|----------|-------|-------|------|-----------|---------|
| **通道剪枝率** | 无变化 (仍 dense) | 线性收缩 | 无影响 | 改变可融合模式 | 必须 % 32==0 | 硬约束 |
| **通道选择** | cuDNN 可能选不同 algo | layout 可能需重排 | 无 | `Conv+BN+ReLU` 依赖权重 | 相同 | — |
| **2:4 N:M 模式** | cuSPARSELt / CUTLASS sparse | 2-bit metadata + 压缩 | 无影响 | 支持 epilogue (bias/act) | ✅ 1.36× | K%4=0 |
| **Block sparse** | cuSPARSE Block-SpMM | BCSR/Blocked-ELL | 无影响 | 有限融合 | ❌ | 块 16/32/64 |
| **非结构化 (>90%)** | Sputnik / SparseRT | CSR/CSC | 无影响 | 难融合 | ❌ | 无 |
| **Head 剪枝** | batched matmul shape 变 | QKV 权重缩 | ✅ 多流 | FlashAttention 兼容 | ✅ (头数仍 dense) | head_dim 对齐 |
| **Token 剪枝** | dense (seq 变短) | KV cache 动态 | 需要 compact kernel | FlashAttention decode 兼容 | 部分 | — |
| **层剪枝 (depth)** | 完整省去 kernel launch | 省激活 | 改变流水 | 前后层融合窗口变 | ✅ | — |
| **动态剪枝** | 每 token 调度 | 运行时 realloc | 严重依赖 | 不可融合 | ❌ | — |
| **量化 + 剪枝共搜** | INT8/FP8 + 2:4 kernel | compressed + scale | 无额外影响 | reformatting 风险 | ✅ DLA INT8 sparse | 32 倍数 |

### 7.3 对 UniV2X 子任务 1.3 的直接建议

1. **把 "硬件可配置维度"分两层定义**:
   - **TensorRT 可直接控制**: `--sparsity=enable/force`、`--fp16 / --int8`、`--useDLACore`、`--memPoolSize=workspace:2048`、optimization profile min/opt/max shape。
   - **TensorRT 自动决策，但可通过软件配置影响**: kernel algorithm (通过通道数对齐/稀疏模式影响可选集)、融合模式 (通过剪枝后图拓扑影响)、reformatting 层 (通过混合精度边界影响)。

2. **剪枝搜索空间约束清单 (硬件侧硬过滤):**
   - [ ] `channels % 32 == 0` (INT8) / `% 16 == 0` (FP16 sparse) / `% 8 == 0` (FP16 dense)
   - [ ] 若启用 2:4: `K_dim % 4 == 0` 且 `K_dim >= 64` (否则加速消失)
   - [ ] 剪枝后通道 "输出 > 128"，避免工作量不足以摊销稀疏 metadata (NVIDIA 官方建议)
   - [ ] DLA 路径上禁止产生 reshape/transpose 把连续子图打断
   - [ ] 跨层剪枝率梯度 < 30% (避免插入 reformatting 或破坏融合)

3. **把 cuSPARSELt MatmulSearch 纳入"硬件配置"搜索维度**: 即便 2:4 模式固定，alg ID 选择也能带来 30% 差异。

4. **LUT-based 代理模型**: 借鉴 HALP/LightPrune，对每层测 `(剪枝率, 稀疏模式, 量化) → latency` 小规模网格，建表；搜索时查表。UniV2X 子任务 1.3 的"代理模型"与之同构。

5. **动态剪枝暂缓**: Jetson/A100 上 dynamic pruning 实施复杂度 >> 静态 2:4 + 通道剪枝。除非 UniV2X 有明确 token/region 动态性 (如 BEV sparse query)，否则优先静态方案。

---

## 参考资料索引 (按章节)

### §1 稀疏模式 × 硬件

- [Structured Sparsity in the NVIDIA Ampere Architecture and Applications in Search Engines](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)
- [Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt)
- [Accelerating Matrix Multiplication with Block Sparse Format and NVIDIA Tensor Cores](https://developer.nvidia.com/blog/accelerating-matrix-multiplication-with-block-sparse-format-and-nvidia-tensor-cores/)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth.md/)
- [Deploying 2:4 Sparsity with FP8 on Hopper — M'Tech 2026](https://www.mtechresearch.tech/deploying-24-sparsity-with-fp8-on-hopper-a-production-cookbo)
- [2:4 Sparse Llama FP8 on Hopper (Red Hat Developer, 2024-12)](https://developers.redhat.com/articles/2024/12/18/24-sparse-llama-fp8-sota-performance-nvidia-hopper-gpus)
- [Explore 2:4 Semi-Structured Sparsity with 1.27x Inference Speedup (HPC-AI 2025)](https://company.hpc-ai.com/blog/explore-24-semi-structured-sparsity-with-1.27x-inference-speedup-on-nvidia-gpus)
- Pool et al. 2021, "Accelerating Sparse Deep Neural Networks with 2:4 Sparsity" — arXiv:2104.08378
- Gale et al. 2020, "Sparse GPU Kernels for Deep Learning" (Sputnik) — SC2020
- SparseRT: arXiv:2008.11849
- cuTeSpMM: arXiv:2504.06443
- AsyncSparse (Hopper TMA+WGMMA): arXiv:2604.17834
- TorchSparse++ (MIT Han Lab): MICRO 2023
- [cuSPARSELt docs](https://docs.nvidia.com/cuda/cusparselt/) / [Release Notes](https://docs.nvidia.com/cuda/cusparselt/release_notes.html)
- [Simplify Sparse Deep Learning with UST in nvmath-python (2026-04)](https://developer.nvidia.com/blog/simplify-sparse-deep-learning-with-universal-sparse-tensor-in-nvmath-python/)

### §2 剪枝率 × 内核选择

- [Sparsity in INT8 TensorRT Workflow](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/)
- [TensorRT Overhead of Shape Change and Optimization Profile](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/overhead-layer-optimization.html)
- DA-SpMM: arXiv:2202.08556

### §3 动态剪枝 / MoE / Head pruning

- Rao et al. NeurIPS'21, "DynamicViT" — [项目页](https://github.com/raoyongming/DynamicViT)
- Gale et al. 2022, "MegaBlocks" — arXiv:2211.15841 / [ROCm/Megablocks](https://github.com/ROCm/Megablocks/) / [nomic-ai/megablocks](https://github.com/nomic-ai/megablocks)
- [TensorRT-LLM PR #11520: Multi-stream MOE and MLA attention (2026-02)](https://github.com/NVIDIA/TensorRT-LLM/pull/11520)
- DuoAttention: arXiv:2410.10819
- DHA (Decoupled Head Attention): proceedings.com/079017

### §4 内存管理

- [NVIDIA Jetson Maximizing Memory Efficiency (2026-04)](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
- [Memory Coalescing: 6x Performance Difference (Dev.to 2026-04)](https://dev.to/codinginavan/memory-coalescing-same-computation-6x-performance-difference-339)

### §5 异构加速

- [NVIDIA Deep-Learning-Accelerator-SW (Orin DLA sparse case study)](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW/blob/main/README.md)
- [Sparsity does not provide any speedup for TensorRT on DLA — Jetson Forum](https://forums.developer.nvidia.com/t/sparsity-does-not-provide-any-speedup-for-tensorrt-on-dla/278355)
- SIGMA (HPCA'20): [paper](https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2020/01/sigma_hpca2020.pdf)
- Hardware-Aware Pruning for FPGA (CVPR'23W): [paper](https://openaccess.thecvf.com/content/CVPR2023W/EVW/papers/Plochaet_Hardware-Aware_Pruning_for_FPGA_Deep_Learning_Accelerators_CVPRW_2023_paper.pdf)

### §6 硬件感知剪枝搜索

- AMC (ECCV'18): [mit-han-lab/amc](https://github.com/mit-han-lab/AMC)
- MetaPruning (ICCV'19): arXiv:1903.10258
- HALP (NeurIPS'22): arXiv:2110.10811 / [项目页](https://halp-neurips.github.io/)
- LightPrune (ICCVW'25): [paper](https://openaccess.thecvf.com/content/ICCV2025W/EVW/papers/Belhadi_LightPrune_Latency-Aware_Structured_Pruning_for_Efficient_Deep_Inference_on_Embedded_ICCVW_2025_paper.pdf)
- PuRL (AutoML'20): [paper](https://www.automl.org/wp-content/uploads/2020/07/AutoML_2020_paper_36.pdf)
- SparseGPT (ICML'23): arXiv:2301.00774
- Wanda (ICLR'24): [locuslab/wanda](https://github.com/locuslab/wanda)
- Sheared LLaMA (ICLR'24): arXiv:2310.06694 / [princeton-nlp/LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing)
- APQ (CVPR'20): arXiv:2006.08509 / [mit-han-lab/apq](https://github.com/mit-han-lab/apq)
