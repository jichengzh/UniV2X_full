# 量化 × 硬件可配置维度交互调研

> **目的**: 补足「子任务 1.1 可配置量化」之后的认知盲点 —— 给定量化方案后,在固定硬件(Orin GPU/DLA、Ampere/Ada/Hopper 数据中心卡)上还有哪些硬件维度可调,这些维度如何与量化决策互相影响。
>
> **与既有调研的区别**:
> - `hw_sw_codesign_quant+prune.md` 偏「量化/剪枝联合搜索空间」
> - `survey_raw/agent_2_quant.md` 偏「学术加速器论文(MicroScopiQ/Panacea/BitMoD...)」
> - **本文聚焦**: 在 NVIDIA 固定平台上,量化格式 ↔ kernel ↔ 调度 ↔ 内存 ↔ DLA 的实际耦合关系,以及量化决策会如何改变这些硬件维度的最优取值。

---

## 0. 总体结论(TL;DR)

1. **量化本身不等于加速**。在 NVIDIA GPU 上,是否能真正得到 speedup 取决于:
   (a) 该代硬件是否有对应精度的 Tensor Core 指令 (IMMA/HMMA/QGMMA/WGMMA),
   (b) 权重和激活是否都被量化(否则只能做 dequant→FP16 计算),
   (c) 量化粒度是否和 kernel 支持的 scale 布局匹配。
2. **固定硬件上,量化决策至少反向影响五类硬件维度**:kernel tactic、执行调度(batch/并发)、内存分配(workspace/KV cache/激活)、DLA/GPU 加速器选择、reformat 数量。
3. **最容易被忽视的耦合**:
   - per-channel activation 在 GPU GEMM 上几乎不可用 → SmoothQuant 的「把难度转移到权重」才成立;
   - INT4 在 Hopper 上反而会回退到 CUDA core(IMAD),Ampere/Ada 才能真正用 IMMA;
   - DLA INT8 强制 per-tensor + HWC4 布局,只要一层不满足就全图 fallback 到 GPU;
   - 小 batch 下 INT8 量化/反量化开销会吃掉 kernel 带来的 speedup,roofline 模型往往高估收益(XLA Issue #40680)。
4. **2024-2026 的新趋势是「算法与硬件布局同步设计」**: MX/NVFP4 block-scaled 格式要求量化器直接写入 tcgen05 打包布局;FlashAttention-3 FP8 要求 Q/K/V 同时量化,否则整个 fused kernel 回退。

---

## 1. 量化格式 ↔ 内核选择

### 1.1 Tensor Core 指令与精度映射(按代际)

不同 GPU 代际对量化精度的支持呈「马赛克」形态,某个精度在 A 代有 Tensor Core、在 B 代却退化成 CUDA core。关键 SASS 指令对照表(来自对 mma/wgmma 指令的 SASS 反汇编分析,arXiv 2402.13499):

| 代际 | FP16 指令 | INT8 指令 | INT4 指令 | FP8 指令 |
|------|----------|-----------|-----------|----------|
| Ampere (sm80) | HMMA.16816 | IMMA.16832.S8.S8 | IMMA.16832.S4.S4 | —(不支持) |
| Ada (sm89) | HMMA | IMMA | IMMA.16832.S4.S4 | WGMMA/QGMMA 部分支持 |
| Hopper (sm90) | HGMMA.64x*x16 | IGMMA.64x256x32.S8.S8 | **退化为 IMAD → CUDA core** | QGMMA(E4M3/E5M2) |
| Blackwell (sm10x) | 同上 + tcgen05 | IMMA + 块缩放 | MXFP4 / NVFP4 / MXFP6 | MXFP8(E8M0 scale) |

**关键意外点**:
- **Hopper 上 INT4 不是更快而是更慢**: `mma` 指令在 Hopper 被编译为一连串 IMAD,性能显著低于 Ampere/Ada 上真正的 Tensor Core INT4 路径(这就是为什么 TensorRT-LLM 把 INT4 AWQ 标注为「仅 Ampere 及以后 GPU」而 W4A8 AWQ 要求「Ada/Hopper 及以后」—— 后者权重 INT4 也是先 dequant 到 FP8 再算)。
- **FP4/FP8 必须走 wgmma/tcgen05**,这些指令只存在于 Hopper/Blackwell,Ada 仅部分支持。

### 1.2 TensorRT tactic 选择机制

- TensorRT 对每个层(或层簇)枚举所有可用的 kernel 实现(tactics),在 build 期实测延迟选择最快。`BuilderFlag::kEDITABLE_TIMING_CACHE` 可以把 tactic 决策 dump 出来并手动覆盖。
- `FP16 / INT8 / TF32` 三个 precision flag 互相独立,**TensorRT 仍然可能为了更快或者因为没有低精度实现,而选择更高精度的 kernel**——也就是说开 `--int8` 不代表每层都跑 INT8。
- 对 `IMatrixMultiplyLayer` 的累加精度默认由输入类型决定,但「至少和输入一样宽」。如要强制 FP32 accumulation(为了 outlier 精度),需要用 `strongly typed mode` 并把输入 cast 到 FP32,TensorRT 会把 cast 融合进 GEMM,得到「FP16 输入 + FP32 累加」的单 kernel。
- **TRT 与 cuBLASLt 共享底层 GEMM 库**(NVIDIA 官方确认 TensorRT issue #3259),所以 cuBLASLt 的 COL32 / 4R4 INT8 排布、tile 结构、`cublasLtMatmulAlgoGetHeuristic` 的调优效果,TRT 都会享受到 —— 但 PyTorch `matmul` 不暴露 `cublasLtMatmulAlgoGetHeuristic`,因此用 PyTorch 当基线 benchmark cuBLAS 会低估。

### 1.3 per-tensor vs per-channel 对 kernel 的影响

- **per-tensor 权重 + per-tensor 激活(SmoothQuant O3)**: 可直接走 off-the-shelf INT8 GEMM(cuBLASLt / TRT 通用),**无需定制 kernel**。
- **per-channel 权重 + per-tensor 激活(SmoothQuant O2, INT8 SQ 默认)**: 主流 GPU GEMM 支持 per-output-channel scale(通过 epilogue 乘回),仍然走 IMMA。
- **per-channel 激活**: 在 GPU 上基本不可用 —— 需要在 GEMM 内层循环里做 per-token 反量化,打断 tile 复用;SmoothQuant 论文反复强调「per-channel activation quantization is not compatible with INT8 GEMM kernels」正是这个原因,因此它把难度「迁移」到了权重。
- **per-group(例如 group=128)**: AWQ 的 W4A16 用这种格式。需要 fused dequant kernel(如 vLLM 的 AWQ Marlin kernel),在 shared memory 内完成 INT4→FP16 反量化后再进 MMA,Marlin 比朴素 AWQ kernel 快 2-4×(Ampere/Hopper)。

### 1.4 对称 vs 非对称

- **对称量化(zero-point=0)**: GEMM 计算仅为 `A_int × B_int + bias`,直接使用 IMMA。
- **非对称量化(zero-point≠0)**: 需要展开 `(A-zA)(B-zB) = AB - zA·B - zB·A + zA·zB`,其中 `zA·B` 和 `zB·A` 是额外的向量求和项,通常被折叠进 bias;但如果激活是非对称 per-token,则需要 per-token 的 `sum(B)` 项,kernel 路径复杂度显著上升。
- **结论**: 在 TensorRT / TRT-LLM 里,激活对称是「默认推荐」;非对称激活只在学术工作(Panacea HPCA'25)里见到定制加速器支持。

### 1.5 Blackwell 的新约束:块缩放布局

- MXFP8 block size = 32,共享 E8M0 scale;NVFP4 block size = 16,共享 E4M3 FP8 scale + per-tensor scale。
- **硬性要求**: scale 字节必须以 tcgen05 期望的「packed layout」存在,否则 block-scaled GEMM 要先跑一次 reshape/permute kernel。fal.ai 在 Blackwell B200 上的 MXFP8 quantizer 就是直接把 scale 写入 tcgen05 布局,达 6+ TB/s;TransformerEngine 的 dense (M, K/32) 布局则需要额外打包步骤。
- 对 UniV2X 的启示: 即使我们不上 Blackwell,这种「量化器输出布局必须与 Tensor Core 输入布局一致」的现象,在 Ampere/Ada 的 INT8 COL32 / per-channel scale 布局上也存在。

---

## 2. 量化 ↔ 执行调度

### 2.1 混合精度下 kernel 的非单调性

- 直觉:INT8 kernel 比 FP16 快。实际:**小 shape 下 INT8 经常更慢**。XLA Issue #40680(2026-04)量化了这一点 —— 在 RTX 3080 上,INT8 vs FP32 的理论峰值比约 4×,但对小/中等 shape,kernel launch overhead + quant/dequant 插入的 fixed cost 吃掉了峰值优势,roofline 模型「显著高估 INT8 GEMM 的 speedup」。
- **TensorRT 的行为**: 开 `--int8` 时,如果 INT8 kernel 实测慢于 FP16,tactic selector 会选 FP16。因此「配置成 INT8 但 profile 显示还是 HMMA」是正常现象,不是 bug。
- **cuDLA 上观察到相反**: Orin DLA 对 INT8 卷积特别优化,约为 FP16 的 15×(稀疏 30×),因此 YOLOv5 on DLA 如果某三层回退到 FP16,会观察到 **显著** 性能下降(NVIDIA 技术博客,2023)—— DLA 的 int8→fp16 fallback 比 GPU 更昂贵。

### 2.2 Batch size 最优点迁移

- LLM roofline(arXiv 2402.16363):
  - FP16 batch-size 分界点 ≈ 240(低于这个点是 memory-bound,高于是 compute-bound)。
  - **INT8 权重 + BF16 激活**(weight-only):分界点降到 ≈ 120,因为权重加载量减半,算数强度翻倍。
  - **INT8 W+A**: 理论分界点和 FP16 差不多(≈ 243),但 INT8 权重 + INT8 激活同时减少内存+计算,在 bandwidth-bound 区是 2× 加速。
- **批大小策略**随量化改变:
  - W4A16(AWQ):对 batch=1 的 decode 阶段收益最大(memory-bound),batch=16+ 时收益急剧减小 —— TensorRT-Model-Optimizer 官方推荐 batch≥16 用 W8A8 或 FP8,而不是 INT4。
  - W8A8(SmoothQuant):无论 batch 大小都有加速,因为同时利用 INT8 带宽和 TOPS。
  - FP8 per-tensor(Ada/Hopper):吞吐双倍 Tensor Core peak,对所有 batch 友好。

### 2.3 KV cache 容量放大效应

这是 LLM 推理侧最直接的硬件维度耦合:
- 对 Llama-3.1-8B 在 32K context:FP16 KV = 2.0GB,FP8 KV = 1.0GB,**TurboQuant TQ4 (INT4 per-token-head) = 0.51GB**(vLLM PR #39008, 2026-04)。
- **KV cache 缩小 → KV blocks 翻倍 → concurrent request 翻倍**。TurboQuant 在 Llama-3.1-8B 上 vLLM GPU KV blocks 从 ~4200(FP8)提到 8167(TQ4),约 1.94× blocks。
- **调度影响**: vLLM PagedAttention 的 block table 需要改尺寸;continuous batching 的最大 batch 上限提高;context window 可以翻倍而不加 GPU。
- **kernel 耦合**: FP8 KV cache 上,FlashAttention-3 要求 Q 也必须是 FP8(否则 kernel 拒绝);FlashInfer XQA 则允许 Q 是 FP16/BF16。因此**同一个量化选择(KV=FP8)会迫使不同的 attention backend 选择**(vLLM PR #29661, 2025-11)。实践做法是在 attention custom op 的 forward() 里才做 Q 的量化,否则无法兼容两个 backend。
- FA3 的另一个限制:Q/K/V descale 分散在 softmax 和 epilogue 里;如果只有 K 的 descale 没有 Q 的,会默默走 BF16 compute path(GH issue #1848,用户从 Nsight Systems 才发现 —— HMMA.16816.F32.BF16 出现即表示 FP8 路径没被激活,应该看到 QGMMA)。

---

## 3. 量化 ↔ 内存管理

### 3.1 Calibration workspace

- **TRT Model Optimizer 的已知 OOM 点**(Issue #107, 2024-11):`from modelopt.torch._deploy._runtime import RuntimeRegistry` 会触发 torch+tensorrt 库加载,吃掉 20GB GPU 内存 —— 与被校准模型大小无关,是**框架侧的固定开销**。
- PTQ calibration 的内存峰值≠推理峰值:entropy/percentile calibration 要保存每层的 histogram(典型 2048 bins × #channels × 4B),对 ViT-L 级别的模型可达数 GB。一个常见 workaround 是降低 calibration 输入分辨率(如 512→224)。
- ModelOpt 提供了 `--low_memory_mode`(仅 FP8/NVFP4 + max calibration):先把权重压到低精度再做 calibration,以避免同时持有 FP32 + INT8 两份权重。对 W8A8 entropy calibration 不适用。
- `--calibrate_per_node` 在大模型上节省显存,代价是 calibration 时间增加。

### 3.2 激活/权重量化的内存足迹差异

- **权重量化**(INT4/INT8 W-only): 静态内存(模型载入)下降到 25%/50%,**激活仍然 FP16**,所以中间激活的峰值内存不变。
- **激活 + 权重量化**(W8A8):模型和激活都 50%,显存峰值同步下降,但**需要 quant/dequant 节点插入计算图**,这些节点自身占 workspace。
- **W4A8**(NVFP4 weight + FP8 activation): 新趋势,权重存储 25%、激活 50%,但要求 Blackwell/Ada 的 FP8 Tensor Core + NVFP4 解码。在 Ada 上是「权重 INT4 dequant→FP8 → FP8 GEMM」,仍然比 FP16 快 1.5×+。

### 3.3 TensorRT workspace 内存分类

- `setMaxWorkspaceSize()` 控制的是 **kernel workspace**(tactic 算 GEMM 时需要的临时 buffer),与 **量化 workspace** 是两套。
- 量化 workspace 主要包括:
  - Q/DQ 节点的临时 tensor(激活量化后的 INT8 张量 + scale 的中间存储)
  - mixed-precision reformat 节点(CHW32 ↔ CHW16 ↔ CHW4)
  - calibration cache 文件(外部存储,不占运行时内存)
- **实战经验**: 混合精度模型的 workspace 比纯 FP16 大,原因是多了 reformat 节点;如果 workspace 给太小,TensorRT 会选次优 tactic 而非报错,所以要给足。

---

## 4. 量化 ↔ 异构加速(DLA / 多加速器)

### 4.1 DLA 量化约束与 fallback 成本

DLA 是 Orin 上的专用 CNN 加速器,对量化有严格约束:

| 维度 | DLA 限制 |
|------|---------|
| 精度 | 只支持 FP16 和 INT8,**不支持 FP32/BF16** |
| 量化粒度 | INT8 强制 per-tensor(不支持 per-channel) |
| 输入布局 | INT8 推荐 `dla_hwc4`(通道数 ≤4 且被 conv 消费时);否则用 `chw32` |
| 算子集合 | 有限(主要 conv/deconv/FC/pool/bn/activation),transformer 常见算子大多不支持 |
| 动态形状 | TRT 8.5 在 Orin 上对 DLA 显式 batch 动态形状会 fallback(GH #4413) |
| 精度混用 | INT8 和 FP16 层之间需要 reformatting(copyNode),某些版本会失败(Orin 论坛讨论 #343304) |

**fallback 代价**:
- 当某层不满足 DLA 约束,TRT 自动切到 GPU(需 `--allowGPUFallback`)。但 **DLA↔GPU 切换需要 DMA 拷贝 + 布局转换**,一个 fallback 可能让整个 subgraph 失去 DLA 全图优势。
- YOLOv5 在 Orin DLA 上的经典案例:若最后三个 conv fall back 到 FP16(DLA 也支持 FP16 但是 INT8 快 15×),整体吞吐明显下降;必须通过 insert Q/DQ at every layer(PTQ scales)把所有层都留在 INT8。
- **PTQ 经验法则**(cuDLA 技术博客):**「The more available scales, the lower the latency」**,与 GPU 上「Q/DQ 太多会打断 TRT fusion」恰好相反 —— 同一份 ONNX 给 DLA 和 GPU 用需要两套策略。

**易踩坑**:
- `DLA INT8 模式下计算格式默认是 kDLA_HWC4`,若 GPU 部分用 `kLINEAR`,相邻层边界会插入格式转换(GH #3799 的一个子问题)。
- **QAT 的 scale 必须导出为 calibration cache 给 DLA 用**(QAT2PTQ),否则 DLA INT8 精度会完全错乱(GH #3799: 相同 ONNX 在 GPU INT8 正确、DLA FP16 正确,但 DLA INT8 结果是错的,根因是 scale 缺失)。

### 4.2 多加速器协同下的量化策略

Jetson Orin 同时有 GPU + 2 DLA。工业界典型调度:
- **DLA 跑密集 CNN backbone**(INT8,per-tensor,HWC4 输入);
- **GPU 跑 transformer / attention / 动态形状**(INT8 per-channel 或 FP16);
- **量化配置自然分裂**: DLA 部分必须 per-tensor,GPU 部分可以 per-channel,中间通过 reformat 层桥接。因此 UniV2X 的可配置量化粒度需要**按 subgraph 区分默认值**,而不是全图统一。

---

## 5. 量化自动搜索 / 自动部署工作

### 5.1 经典工作(2018-2021)

| 工作 | 核心思路 | 硬件维度反馈 |
|------|---------|--------------|
| **HAQ** (CVPR'19 oral) | RL agent 决定每层 bit-width,用硬件模拟器给 latency/energy 反馈 | **直接硬件测量**(非 FLOPs 代理),发现「edge 和 cloud 的最优策略根本不同」 |
| **HAWQ-V1/V2** (ICCV'19, NeurIPS'20) | Hessian 特征值当敏感度 → 决定层 bit-width | 无直接硬件反馈(V1/V2),只用敏感度 |
| **HAWQ-V3** (arXiv 2011.10680) | 把 bit-width 搜索写成 ILP,约束是 latency/BOPS/size | **硬件感知**: ILP 约束可以是实测延迟,秒级求解 |
| **DNAS** (Wu et al. 2018) | Gumbel-Softmax 可微 bit-width 搜索 | 延迟当 regularization,端到端可训 |
| **APQ** (CVPR'20) | 联合搜 NAS + pruning + quant,用 accuracy predictor 省算力 | 硬件通过 latency predictor 反馈 |

**核心教训**: HAWQ-V3 的 ILP formulation 是把量化和硬件维度结合的最简洁方案 —— **目标是最小化二阶 perturbation,约束是实测延迟/能耗/模型大小**,在「可解释 + 可复现 + 秒级求解」三个维度上比 RL/DNAS 都占优。UniV2X 的可配置量化搜索如果需要自动化,ILP 应当是首选工具。

### 5.2 LLM 量化部署(2022-2026)

| 方法 | 精度 | 硬件路径 | 算法-硬件耦合点 |
|------|------|---------|----------------|
| **LLM.int8()** | W8A8 + FP16 outlier | CUDA 自定义 kernel(分离 outlier 计算) | 因为 **outlier 处理需要 gather/scatter**,比 FP16 还慢,被 SmoothQuant 淘汰 |
| **SmoothQuant** | W8A8(per-channel W + per-tensor A) | off-the-shelf INT8 GEMM | **per-channel activation 不可行 → 把难度迁移到权重**;直接复用 cuBLASLt |
| **GPTQ** | W4A16 / W3A16 | 自定义 dequant kernel + FP16 GEMM | 4-bit 权重必须 **fused dequant + FP16 matmul**,否则 DRAM 写回开销吃掉加速 |
| **AWQ** | W4A16(per-group=128) | Marlin kernel(vLLM), TinyChat kernel | Marlin 用 Ampere/Hopper 的 `ldmatrix` + 重排 INT4 → **shared memory 内反量化再 MMA**,比朴素 AWQ 快 2-4× |
| **FP8 (E4M3/E5M2)** | W8A8(per-tensor 或 per-channel W) | cuBLASLt FP8 GEMM (Ada/Hopper) | 只在 sm89/sm90+ 可用;要求 Tensor Core QGMMA |
| **INT4-FP8 AWQ (W4A8)** | W4 + A8 (FP8) | TRT-LLM custom kernel | W 先 dequant → FP8,再 FP8 Tensor Core;**只有 Ada/Hopper 及以后** |
| **MXFP4/NVFP4** | W4A4 块缩放 | Blackwell tcgen05 块缩放 GEMM | scale 必须按 tcgen05 打包布局写入;CUTLASS 4.3+ 支持 |
| **TurboQuant TQ4** | KV cache W4(per-token-head) | reuse 既有 INT8 attention kernel(unpack 到 INT8) | **不改 attention kernel** 是 deliverable 的关键,3.76× vs FP16 |

**选型表**(来自 TRT Model Optimizer 官方文档):
- **batch ≤ 4(memory-bound)**: 权重 only 量化已经够,推荐 INT4 AWQ(Ampere+)或 FP8(Ada+)
- **batch ≥ 16(compute-bound)**: 必须 W+A 都量化,推荐 FP8(精度好)或 INT8 SQ(老 GPU)或 INT4-FP8 AWQ(最激进)
- **Ampere 及之前**: INT4 AWQ 或 INT8 SmoothQuant
- **Ada/Hopper**: FP8 per-tensor 优先,不够再上 INT4-FP8 AWQ

### 5.3 工业工具链

| 工具链 | 搜索空间 | 硬件约束能力 |
|-------|---------|-------------|
| **TensorRT Model Optimizer** (NVIDIA) | FP8 / NVFP4 / INT8 SQ / INT4 AWQ / INT4-FP8 AWQ; 层级 `nvfp4_mlp_only` / `nvfp4_experts_only` / `nvfp4_omlp_only` 等预设 | 输出直接被 TRT/TRT-LLM 消费,不需要再 tactic 搜索 |
| **NVIDIA NeMo Quantization Toolkit** | 聚焦 LLM,PTQ + QAT | 同上 |
| **llm-compressor** (Neural Magic/RedHat) | W8A8 / W4A16 / KV cache per-tensor 或 per-attention-head FP8 | 直接导出 vLLM 可用格式(PR #30141) |
| **TorchAO** | 和 torch.compile 集成的 weight-only/weight+activation 量化 | 通过 torch.compile 触发 Triton kernel 或外部 kernel |
| **Intel Extension for PyTorch** | RTN/AWQ/GPTQ for Intel GPU,W4 全范围 | 调用 `HGEMM_INT4_COMMON_DISPATCH` 分派到定制 kernel |

**对 UniV2X 的启发**:
- 如果目标部署栈是 TRT(Orin GPU)或 TRT-DLA,**ModelOpt 就是事实标准**,我们自研的「可配置量化」最好是能输出 ModelOpt / ONNX Q/DQ 格式,而非自造格式。
- ModelOpt 的层级预设(`nvfp4_mlp_only` 等)说明:工业界的可配置量化不是「每层一个 bit-width 参数」,而是「一组实战验证过的粒度模板」,这个抽象层次值得借鉴。

---

## 6. 关键洞察与交互矩阵

### 6.1 交互矩阵(量化决策 × 硬件维度)

| 量化决策 \ 硬件维度 | kernel tactic | batch 最优点 | KV/激活内存 | DLA 可用性 | reformat 数量 |
|---|---|---|---|---|---|
| **FP16 → INT8 per-tensor** | IMMA 激活,Ampere+ | 几乎不变 | 激活 ÷2 | 可 DLA | 不变 |
| **INT8 per-tensor → per-channel W** | 仍 IMMA(epilogue scale) | 不变 | 不变 | **DLA fallback** | 不变 |
| **INT8 → INT4 W-only (AWQ)** | 需定制 dequant kernel(Marlin) | batch↓ 更有利 | 权重 ÷4 | DLA 不支持 | reformat 可能插入 |
| **FP16 → FP8** | QGMMA(仅 Ada/Hopper+) | 分界点保持 | ÷2 | DLA 不支持 | reformat 可能插入 |
| **INT8 对称 → 非对称 A** | 复杂度上升,TRT 多数情况 fallback | 不变 | 多 scale/zp | DLA fallback | 增加 |
| **per-tensor → per-group(g=128)** | 定制 group-GEMM kernel | 不变 | 多 scale | DLA fallback | 可能增加 |
| **KV cache FP16 → FP8** | FA3 要求 Q 也 FP8;FlashInfer XQA 放宽 | **batch↑** | KV ÷2 | 不适用(LLM) | Q 量化点迁移 |
| **KV cache FP16 → TQ4** | 仅需解包到 INT8,复用现有 attn kernel | batch↑↑ | KV ÷3.76 | 不适用 | 插入解包 |
| **激活量化点从输入→每层** | Q/DQ 太多会破坏 TRT fusion(GPU) | 不变 | 多 scale | **DLA 更快**(more scales = lower latency) | **依 runtime 而定** |

### 6.2 可落地的五条经验法则

1. **「量化粒度 = 硬件天花板」**: 选粒度前先看最保守的部署目标。Orin DLA 是 per-tensor INT8;想上 per-channel 就不能在 DLA 上跑那一层。
2. **「权重和激活必须一起量化才能用 Tensor Core peak TOPS」**: 只量化权重(W4A16/W8A16)本质是 memory-bound 加速,对 batch≥16 收益急剧衰减;compute-bound 场景必须 W+A 都量化。
3. **「同一份 ONNX 不要指望同时最优 DLA 和 GPU」**: Q/DQ 数量对两者需求相反 —— GPU 少插点(保融合),DLA 多插点(保 INT8);实战中需要两份配置。
4. **「Hopper 用 INT4 权重是反模式,应该上 FP8」**: Ampere 的 INT4 是 Tensor Core,Hopper 的 INT4 会退化成 CUDA core。W4A8 AWQ 在 Hopper 上实际是「权重 INT4 dequant → FP8 GEMM」,Tensor Core 跑的是 FP8 不是 INT4。
5. **「开 --int8 不代表每层 INT8」**: TRT tactic selector 会在小 shape 上主动选 FP16。不要相信 `--int8` 就断言 speedup,一定要看 dumpProfile 或 Nsight 的 SASS 指令确认是 IMMA 还是 HMMA(FP8 路径要看 QGMMA 而非 HMMA.F32.BF16)。

### 6.3 UniV2X 的方向性建议

1. **可配置量化的「默认值」应该按 subgraph 分裂**:图像分支(准备给 DLA 跑)默认 per-tensor INT8 对称 + MinMax;transformer 分支(给 GPU 跑)默认 per-channel INT8 对称 + Percentile。
2. **引入「硬件可用性检查」前置门**:在搜索量化配置时,先用一个白名单过滤掉硬件不支持的组合(例如 Hopper+INT4+Tensor Core 直接判不可行),避免无效 calibration。
3. **对接 ModelOpt / ONNX Q/DQ**:自研格式长期成本高,能输出 Q/DQ ONNX 就能复用 TRT Model Optimizer 全栈。
4. **把 KV cache 量化当成独立维度**:如果 UniV2X 后续接 LLM 部件,KV 量化(FP8/INT4)的价值往往比权重量化更高,且它改变的是「并发数/context 长度」而非单 query 延迟。
5. **Profile 一定看 SASS**:HMMA / IMMA / QGMMA / IGMMA 是判断量化是否真正生效的唯一可靠信号;trtexec `--dumpProfile` 和 Nsight Systems 都能拿到。

---

## 参考资料(关键来源摘要)

### 官方文档 / 博客
- NVIDIA TRT Model Optimizer: [Choosing Quantization Methods](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/_choosing_quant_methods.html) — batch vs 方法选型表
- NVIDIA TensorRT 10.x: [Algorithm Selection and Reproducible Builds / Precision Control](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html) — tactic cache、FP16/INT8/TF32 互斥逻辑
- NVIDIA Developer Blog: [Deploying YOLOv5 on Orin with cuDLA](https://developer.nvidia.com/blog/deploying-yolov5-on-nvidia-jetson-orin-with-cudla-quantization-aware-training-to-inference/) — DLA INT8 15× 于 FP16、dla_hwc4 布局、Q/DQ 规则
- NVIDIA Developer Blog: [Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) — cublasLtMatmulAlgoGetHeuristic 调优、sm80+ gemmAlgo 废弃
- PyTorch Blog: [FlashAttention-3](https://pytorch.org/blog/flashattention-3) — Hopper FP8 kernel, WGMMA, 1.2 PFLOPS
- fal.ai Blog: [Chasing 6+ TB/s: an MXFP8 quantizer on Blackwell](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/) — 量化器输出必须对齐 tcgen05 布局

### 学术论文
- **HAQ** (Wang et al., CVPR'19 oral): [arxiv.org/abs/1811.08886](https://arxiv.org/abs/1811.08886)
- **HAWQ / HAWQ-V2 / HAWQ-V3** (Dong/Yao et al., ICCV'19/NeurIPS'20/ICML'21): ILP formulation [arXiv 2011.10680](https://arxiv.org/pdf/2011.10680)
- **AWQ** (Lin et al., MLSys'24 best paper): on-the-fly dequant + kernel fusion
- **SmoothQuant** (Xiao et al., ICML'23): per-channel A → 把难度迁移到 W
- **LLM Inference Unveiled: Roofline Model Insights** ([arXiv 2402.16363](https://arxiv.org/html/2402.16363v5)): INT8 roofline、quant 操作对分界点的影响
- **SASS 级 mma/wgmma 分析** ([arXiv 2402.13499](https://arxiv.org/pdf/2402.13499v1)): Hopper INT4 退化为 IMAD 的根因

### GitHub Issues / PRs(实战信号)
- [NVIDIA/TensorRT#3799](https://github.com/NVIDIA/TensorRT/issues/3799) — DLA INT8 精度错乱 = scale 缺失,需 QAT2PTQ cache
- [NVIDIA/TensorRT#3211](https://github.com/NVIDIA/TensorRT/issues/3211) — 同一 ResNet50,FP16 全 DLA,INT8 部分 fallback
- [NVIDIA/Model-Optimizer#107](https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/107) — ModelOpt 加载即占 20GB,`--low_memory_mode` / `--calibrate_per_node`
- [Dao-AILab/flash-attention#1848](https://github.com/Dao-AILab/flash-attention/issues/1848) — FP8 FA3 悄悄走 BF16 路径,需看 SASS 确认
- [vllm-project/vllm#29661](https://github.com/vllm-project/vllm/pull/29661) — FP8 KV cache 迫使 attention backend 选择分裂
- [vllm-project/vllm#39008](https://github.com/vllm-project/vllm/pull/39008) — TurboQuant TQ4 KV,3.76× vs FP16,不改 attn kernel
- [openxla/xla#40680](https://github.com/openxla/xla/issues/40680) — roofline 模型高估 INT8 GEMM speedup,小 shape 下 overhead 占主导
- [NVIDIA/TensorRT#3259](https://github.com/NVIDIA/TensorRT/issues/3259) — TRT 和 cuBLASLt 共享 GEMM 库,PyTorch 不暴露 algoGetHeuristic
