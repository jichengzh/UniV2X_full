# Agent-2: 量化/剪枝/NAS硬件协同 (10篇)

> Agent使用了WebSearch,结果质量较高

### 1. MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization
- **Venue:** ISCA 2025
- **链接:** https://arxiv.org/abs/2411.05282
- **协同维度:** 量化 × 剪枝 × 加速器设计
- **简介:** outlier-aware microscaling量化与剪枝联合,多精度INT PE阵列+ReCoN NoC,推理速度提升3倍、能耗降低2倍。

### 2. Panacea: Novel DNN Accelerator using Accuracy-Preserving Asymmetric Quantization and Energy-Saving Bit-Slice Sparsity
- **Venue:** HPCA 2025
- **链接:** https://arxiv.org/abs/2412.10059
- **协同维度:** 非对称量化 × bit-slice稀疏性 × 加速器
- **简介:** 非对称量化bit-slice GEMM,加速器支持稀疏/稠密工作负载,较Sibia能效提升1.97倍。

### 3. M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type
- **Venue:** HPCA 2025
- **链接:** https://arxiv.org/abs/2502.18755
- **协同维度:** 自适应数据类型 × 混合精度 × 脉动阵列
- **简介:** 数学自适应数值类型MANT,集成脉动阵列支持2/4/8-bit混合精度,加速2.99倍。

### 4. BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration
- **Venue:** HPCA 2025
- **链接:** https://arxiv.org/abs/2411.11745
- **协同维度:** 混合数据类型 × bit-serial计算 × 加速器
- **简介:** bit-serial混合数据类型量化,统一处理2-8bit精度,较ANT和OliVe分别加速1.69倍和1.48倍。

### 5. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression
- **Venue:** MLSys 2024 (Best Paper)
- **链接:** https://arxiv.org/abs/2306.00978
- **协同维度:** 激活感知量化 × 端侧部署 × 推理框架
- **简介:** 基于激活分布保护1%显著权重通道,配套TinyChat框架较FP16加速3倍,首次移动端部署70B模型。

### 6. OliVe: Accelerating LLMs via Hardware-friendly Outlier-Victim Pair Quantization
- **Venue:** ISCA 2023
- **链接:** https://arxiv.org/abs/2304.07493
- **协同维度:** 离群值量化 × 内存对齐编码 × 加速器
- **简介:** outlier-victim pair量化实现内存对齐,集成脉动阵列和tensor core,较GOBO加速4.5倍。

### 7. SOFA: Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling
- **Venue:** MICRO 2024
- **链接:** https://arxiv.org/abs/2407.10416
- **协同维度:** 动态稀疏性 × 跨阶段协同 × 注意力加速器
- **简介:** 跨阶段协同tiling策略处理Transformer动态稀疏,较A100加速9.5倍,能效提升71.5倍。

### 8. Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency SNNs
- **Venue:** ISCA 2025
- **链接:** https://doi.org/10.1145/3695053.3731035
- **协同维度:** 层次化稀疏性 × 模式感知 × SNN加速器
- **简介:** 算法端减少离群值增强模式化稀疏,架构端动态生成稀疏模式,加速3.45倍。

### 9. FGMP: Fine-Grained Mixed-Precision Quantization for Hardware-Accelerated LLM Inference
- **Venue:** arXiv preprint 2025 (NVIDIA)
- **链接:** https://arxiv.org/abs/2504.14152
- **协同维度:** 细粒度混合精度 × FP4/FP8 × 硬件协同
- **简介:** block级混合精度NVFP4/FP8量化,协同设计VMAC数据通路,能耗降低14%。

### 10. TASDER: Enabling Unstructured Sparse Acceleration on Structured Sparse Accelerators
- **Venue:** MLSys 2025
- **链接:** https://arxiv.org/abs/2403.07953
- **协同维度:** 非结构化稀疏 × 结构化硬件 × 张量分解
- **简介:** 利用线性代数分配律将非结构化稀疏近似为结构化稀疏序列,EDP改善83%。
