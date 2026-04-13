# Agent-1: LLM/Transformer专用加速 (10篇)

> 注意: Agent未能调用WebSearch,结果基于训练知识,链接需PM验证

### 1. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Venue:** ICLR 2024
- **链接:** https://arxiv.org/abs/2307.08691
- **协同维度:** 稀疏注意力 × GPU SRAM分层 × IO感知调度
- **简介:** 优化注意力计算的GPU线程块划分与warp级并行策略,减少非矩阵乘FLOPs占比,相比FlashAttention提速约2倍。

### 2. vLLM: Efficient Memory Management for LLM Serving with PagedAttention
- **Venue:** SOSP 2023
- **链接:** https://arxiv.org/abs/2309.06180
- **协同维度:** KV-cache分页管理 × 虚拟内存抽象 × GPU显存调度
- **简介:** 借鉴OS虚拟内存分页机制管理KV cache,消除内存碎片,支持共享与copy-on-write。

### 3. SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference
- **Venue:** ASPLOS 2024
- **链接:** https://arxiv.org/abs/2305.09781
- **协同维度:** 投机解码 × 树状并行验证 × GPU批量推理
- **简介:** 使用多个小模型构建token树投机推理,大模型一次性并行验证整棵树。

### 4. SGLang: Efficiently Programming Large Language Models
- **Venue:** NeurIPS 2024
- **链接:** https://arxiv.org/abs/2312.07104
- **协同维度:** KV-cache复用 × RadixAttention × serving系统调度
- **简介:** RadixAttention自动复用前缀KV cache,大幅提升复杂LLM程序吞吐。

### 5. Megablocks: Efficient Sparse Training with Mixture-of-Experts
- **Venue:** MLSys 2023
- **链接:** https://arxiv.org/abs/2211.15841
- **协同维度:** MoE稀疏计算 × block-sparse矩阵运算 × GPU kernel优化
- **简介:** 将MoE不均匀expert分配转换为block-sparse矩阵乘,避免padding和token-drop。

### 6. FlashDecoding++
- **Venue:** MLSys 2024
- **链接:** https://arxiv.org/abs/2311.01282
- **协同维度:** 解码阶段attention × 异步softmax × GPU flat GEMM
- **简介:** 针对LLM解码阶段提出统一最大值技巧消除同步开销,优化flat-shape GEMM。

### 7. DistServe: Disaggregating Prefill and Decoding for LLM Serving
- **Venue:** OSDI 2024
- **链接:** https://arxiv.org/abs/2401.09670
- **协同维度:** prefill/decode分离 × 异构资源调度 × serving系统
- **简介:** 将prefill与decoding分配到不同GPU实例,消除干扰,最大化goodput。

### 8. Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- **Venue:** ISCA 2024
- **链接:** https://arxiv.org/abs/2311.18677
- **协同维度:** 推理阶段拆分 × 异构硬件映射 × 集群调度
- **简介:** 分析prefill(compute-bound)与decode(memory-bound)特性差异,映射到异构机器。

### 9. Orca: A Distributed Serving System for Transformer-Based Models
- **Venue:** OSDI 2022
- **链接:** https://www.usenix.org/conference/osdi22/presentation/yu
- **协同维度:** 连续批处理 × iteration-level调度 × GPU利用率优化
- **简介:** 提出iteration-level scheduling实现连续批处理,成为后续LLM serving基础。

### 10. FlexGen: High-Throughput Generative Inference with a Single GPU
- **Venue:** ICML 2023
- **链接:** https://arxiv.org/abs/2303.06865
- **协同维度:** KV-cache offloading × CPU-GPU-Disk分层存储 × 线性规划调度
- **简介:** 通过线性规划搜索GPU/CPU/Disk间最优tensor放置策略,单GPU高吞吐推理。
