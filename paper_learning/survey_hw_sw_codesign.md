# 深度学习软硬件协同加速综述 (2023-2026)

> 调研时间: 2026-04-09
> 论文范围: 2023年至今,优先顶会顶刊
> 论文总数: 49篇 (去重后)

---

## 1. 概述

深度学习模型的计算需求持续增长,单纯依靠硬件算力提升或算法优化已难以满足实时性和能效要求。软硬件协同加速(HW-SW Co-design)通过联合优化算法、编译器、系统软件和硬件架构,在多个层次上协调设计决策,以突破单层优化的天花板。

本综述覆盖2023-2026年间的前沿工作,按以下5个方向分类,并单独设章介绍自动驾驶/车路协同场景的实践:

| 分类 | 核心问题 | 典型协同维度 |
|------|---------|-------------|
| A. LLM/Transformer专用加速 | 大模型推理的内存墙、延迟、吞吐优化 | KV-cache管理 × GPU调度 × serving系统 |
| B. 模型压缩与硬件协同 | 量化/剪枝/NAS与加速器的联合设计 | 量化策略 × 稀疏硬件 × 精度-效率权衡 |
| C. 编译与系统软件栈 | 算子融合、调度、内存规划与硬件适配 | 编译器 × 计算图优化 × 通信调度 |
| D. 新型计算架构 | CIM/Chiplet/FPGA等非传统架构的软件协同 | 架构设计 × 映射/编译 × DSE框架 |
| E. 自动驾驶与V2X | 感知模型的边缘部署与实时推理 | 模型压缩 × 嵌入式部署 × 硬件加速器 |

---

## 2. LLM/Transformer专用加速

大语言模型推理面临compute-bound的prefill阶段与memory-bound的decode阶段的特性差异,以及KV-cache的内存管理挑战。该方向论文密度最高,涵盖从单GPU优化到集群级serving系统。

### [1] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Venue:** ICLR 2024
- **链接:** [arXiv](https://arxiv.org/abs/2307.08691)
- **协同维度:** 注意力计算 × GPU SRAM分层 × IO感知调度
- **简介:** 优化注意力计算的GPU线程块划分与warp级并行策略,利用GPU存储层次减少HBM访问,相比FlashAttention提速约2倍。

### [2] vLLM: Efficient Memory Management for LLM Serving with PagedAttention
- **Venue:** SOSP 2023
- **链接:** [arXiv](https://arxiv.org/abs/2309.06180)
- **协同维度:** KV-cache分页 × 虚拟内存抽象 × GPU显存调度
- **简介:** 借鉴OS虚拟内存的分页机制管理KV cache,消除内存碎片并支持copy-on-write共享,显存利用率接近理论最优。

### [3] SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference
- **Venue:** ASPLOS 2024
- **链接:** [arXiv](https://arxiv.org/abs/2305.09781)
- **协同维度:** 投机解码 × 树状并行验证 × GPU批量推理
- **简介:** 使用多个小模型构建token树进行投机推理,大模型一次性并行验证整棵树,显著降低自回归解码延迟。

### [4] SGLang: Efficiently Programming Large Language Models
- **Venue:** NeurIPS 2024
- **链接:** [arXiv](https://arxiv.org/abs/2312.07104)
- **协同维度:** KV-cache复用 × RadixAttention × serving系统
- **简介:** 提出RadixAttention自动复用前缀KV cache,结合编程前端与运行时优化,大幅提升复杂LLM程序吞吐。

### [5] Megablocks: Efficient Sparse Training with Mixture-of-Experts
- **Venue:** MLSys 2023
- **链接:** [arXiv](https://arxiv.org/abs/2211.15841)
- **协同维度:** MoE稀疏计算 × block-sparse矩阵运算 × GPU kernel
- **简介:** 将MoE的不均匀expert分配转换为block-sparse矩阵乘,避免padding和token-drop,GPU上高效稀疏训练。

### [6] FlashDecoding++: Faster LLM Inference with Asynchrony and Flat GEMM Optimization
- **Venue:** MLSys 2024
- **链接:** [arXiv](https://arxiv.org/abs/2311.01282)
- **协同维度:** 解码attention × 异步softmax × GPU flat GEMM
- **简介:** 针对LLM解码阶段提出统一最大值技巧消除同步开销,优化flat-shape GEMM执行效率。

### [7] DistServe: Disaggregating Prefill and Decoding for LLM Serving
- **Venue:** OSDI 2024
- **链接:** [arXiv](https://arxiv.org/abs/2401.09670)
- **协同维度:** prefill/decode分离 × 异构资源 × serving系统
- **简介:** 将prefill与decoding分配到不同GPU实例,消除二者干扰,在满足SLO条件下最大化goodput。

### [8] Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- **Venue:** ISCA 2024
- **链接:** [arXiv](https://arxiv.org/abs/2311.18677)
- **协同维度:** 推理阶段拆分 × 异构硬件映射 × 集群调度
- **简介:** 分析prefill(compute-bound)与decode(memory-bound)的特性差异,将两阶段映射到异构机器以提升集群利用率。

### [9] Orca: A Distributed Serving System for Transformer-Based Models
- **Venue:** OSDI 2022 (开创性工作)
- **链接:** [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
- **协同维度:** 连续批处理 × iteration-level调度 × GPU利用率
- **简介:** 提出iteration-level scheduling实现请求的连续批处理,消除静态batching的气泡,成为后续LLM serving系统的基础。

### [10] FlexGen: High-Throughput Generative Inference with a Single GPU
- **Venue:** ICML 2023
- **链接:** [arXiv](https://arxiv.org/abs/2303.06865)
- **协同维度:** KV-cache offloading × CPU-GPU-Disk分层存储 × 线性规划
- **简介:** 通过线性规划搜索GPU/CPU/Disk间的最优tensor放置策略,在单GPU上实现大模型高吞吐离线推理。

---

## 3. 模型压缩与硬件协同

该方向聚焦"让模型更小更快"的算法与硬件联合设计,涵盖量化(特别是LLM的低比特量化)、稀疏性加速器、以及联合压缩策略。

### [11] MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization
- **Venue:** ISCA 2025
- **链接:** [arXiv](https://arxiv.org/abs/2411.05282)
- **协同维度:** 量化 × 剪枝 × 加速器设计
- **简介:** outlier-aware microscaling量化与Hessian剪枝联合,设计多精度INT PE阵列+ReCoN NoC,推理速度提升3倍、能耗降低2倍。

### [12] Panacea: Novel DNN Accelerator using Accuracy-Preserving Asymmetric Quantization
- **Venue:** HPCA 2025
- **链接:** [arXiv](https://arxiv.org/abs/2412.10059)
- **协同维度:** 非对称量化 × bit-slice稀疏性 × 加速器
- **简介:** 首次提出非对称量化bit-slice GEMM(AQS-GEMM),加速器支持稀疏/稠密工作负载,较Sibia能效提升1.97倍。

### [13] M-ANT: Efficient Low-bit Group Quantization for LLMs via Mathematically Adaptive Numerical Type
- **Venue:** HPCA 2025
- **链接:** [arXiv](https://arxiv.org/abs/2502.18755)
- **协同维度:** 自适应数据类型 × 混合精度 × 脉动阵列
- **简介:** 数学自适应数值类型MANT支持灵活编码,集成脉动阵列实现2/4/8-bit混合精度,平均加速2.99倍。

### [14] BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration
- **Venue:** HPCA 2025
- **链接:** [arXiv](https://arxiv.org/abs/2411.11745)
- **协同维度:** 混合数据类型 × bit-serial计算 × 加速器
- **简介:** bit-serial混合数据类型量化方案,统一处理2-8bit精度,较ANT和OliVe分别加速1.69倍和1.48倍。

### [15] AWQ: Activation-aware Weight Quantization for On-Device LLM Compression
- **Venue:** MLSys 2024 (Best Paper)
- **链接:** [arXiv](https://arxiv.org/abs/2306.00978)
- **协同维度:** 激活感知量化 × 端侧部署 × 推理框架
- **简介:** 基于激活分布保护1%显著权重通道,配套TinyChat推理框架较FP16加速3倍,首次移动端部署70B LLaMA-2。

### [16] OliVe: Accelerating LLMs via Hardware-friendly Outlier-Victim Pair Quantization
- **Venue:** ISCA 2023
- **链接:** [arXiv](https://arxiv.org/abs/2304.07493)
- **协同维度:** 离群值量化 × 内存对齐编码 × 加速器
- **简介:** outlier-victim pair量化牺牲相邻不重要值容纳离群值,实现内存对齐,较GOBO加速4.5倍、能效提升4.0倍。

### [17] SOFA: Compute-Memory Optimized Sparsity Accelerator via Cross-Stage Coordinated Tiling
- **Venue:** MICRO 2024
- **链接:** [arXiv](https://arxiv.org/abs/2407.10416)
- **协同维度:** 动态稀疏性 × 跨阶段协同 × 注意力加速器
- **简介:** 跨阶段协同tiling策略处理Transformer大规模token并行的动态稀疏,较A100加速9.5倍,能效提升71.5倍。

### [18] Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency SNNs
- **Venue:** ISCA 2025
- **链接:** [DOI](https://doi.org/10.1145/3695053.3731035)
- **协同维度:** 层次化稀疏性 × 模式感知 × SNN加速器
- **简介:** 算法端减少离群值增强模式化稀疏,架构端动态生成稀疏模式,实现3.45倍加速和4.93倍能效提升。

### [19] FGMP: Fine-Grained Mixed-Precision Quantization for Hardware-Accelerated LLM Inference
- **Venue:** arXiv preprint 2025 (NVIDIA)
- **链接:** [arXiv](https://arxiv.org/abs/2504.14152)
- **协同维度:** 细粒度混合精度 × FP4/FP8 × 硬件协同
- **简介:** block级混合精度NVFP4/FP8量化,协同设计VMAC数据通路和在线激活量化单元,推理能耗降低14%。

### [20] TASDER: Enabling Unstructured Sparse Acceleration on Structured Sparse Accelerators
- **Venue:** MLSys 2025
- **链接:** [arXiv](https://arxiv.org/abs/2403.07953)
- **协同维度:** 非结构化稀疏 × 结构化硬件 × 张量分解
- **简介:** 利用线性代数分配律将非结构化稀疏近似为结构化稀疏序列,弥合软件偏好与硬件约束的鸿沟,EDP改善83%。

---

## 4. 编译与系统软件栈优化

编译器和运行时系统是连接算法与硬件的桥梁。该方向近年呈现两大趋势: 算子融合范围持续扩展,以及LLM驱动的编译优化新范式。

### [21] Souffle: Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions
- **Venue:** ASPLOS 2024
- **链接:** [DOI](https://doi.org/10.1145/3617232.3624858)
- **协同维度:** 编译器 × 算子融合 × GPU代码生成
- **简介:** 基于张量表达式构建全局依赖图,跨算子边界进行数据流分析与调度优化。A100上较TensorRT加速3.7倍,较XLA加速7.8倍。

### [22] MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators
- **Venue:** SC 2024
- **链接:** [DOI](https://doi.org/10.1109/SC41406.2024.00040)
- **协同维度:** 算子融合 × 搜索空间 × 性能模型
- **简介:** 针对内存密集型计算密集算子链构建高层搜索空间结合解析性能模型。A100上较Ansor加速5.9倍,调优时间降低70倍。

### [23] GraphPipe: Improving Performance and Scalability of DNN Training with Graph Pipeline Parallelism
- **Venue:** ASPLOS 2025
- **链接:** [DOI](https://doi.org/10.1145/3669940.3707220)
- **协同维度:** 计算图拓扑 × 流水线并行 × 调度优化
- **简介:** 图流水线并行将DNN分割为DAG形式流水线阶段,保留模型拓扑实现计算独立算子并发执行,较PipeDream加速1.6倍。

### [24] ClusterFusion: Expanding Operator Fusion Scope for LLM Inference
- **Venue:** NeurIPS 2025
- **链接:** [arXiv](https://arxiv.org/abs/2508.18850)
- **协同维度:** 算子融合 × 片上通信 × LLM推理
- **简介:** 引入集群级通信原语将QKV投影+注意力+输出投影融合为单一kernel,避免片外内存访问。H100上延迟降低1.61倍。

### [25] Arnold: Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment
- **Venue:** NeurIPS 2025
- **链接:** [arXiv](https://arxiv.org/abs/2509.15940)
- **协同维度:** 网络拓扑 × 通信调度 × 分布式训练
- **简介:** 分析通信模式与数据中心拓扑的不匹配,设计拓扑感知调度算法。9600+GPU生产环境性能提升10.6%。

### [26] Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving
- **Venue:** NeurIPS 2025
- **链接:** [arXiv](https://arxiv.org/abs/2506.01374)
- **协同维度:** LLM推理 × 编译器自动调优 × MCTS搜索
- **简介:** 用LLM生成编译变换提议以MCTS引导搜索,36个样本即达2.5倍加速,样本效率比TVM进化搜索高16倍。

### [27] FuseFlow: Fusion-Centric Compilation for Sparse Deep Learning on Streaming Dataflow
- **Venue:** ASPLOS 2026
- **链接:** [DOI](https://doi.org/10.1145/3779212.3790165)
- **协同维度:** 稀疏编译器 × 跨表达式融合 × 数据流架构
- **简介:** 首个支持稀疏算子跨表达式融合的端到端编译器,将PyTorch稀疏模型编译为融合数据流图,加速最高2.7倍。

### [28] RedFuser: Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators
- **Venue:** ASPLOS 2026
- **链接:** [DOI](https://doi.org/10.1145/3779212.3790209)
- **协同维度:** 级联归约 × 符号推导 × 硬件感知代码生成
- **简介:** 级联归约融合的形式化方法,符号推导引擎自动识别可融合模式并生成增量计算,较TVM加速2-5倍。

### [29] DeepCompile: Compiler-Driven Approach to Optimizing Distributed Deep Learning Training
- **Venue:** arXiv preprint
- **链接:** [arXiv](https://arxiv.org/abs/2504.09983)
- **协同维度:** 分布式训练 × 计算图变换 × 通信重叠
- **简介:** 编译图变换插入/重排分布式算子实现通信-计算重叠,较ZeRO-3加速1.28倍。

---

## 5. 新型计算架构

存内计算(CIM/PIM)、Chiplet异构架构、FPGA加速等非传统计算范式正在从原型走向实用,其配套的软件栈(编译器、映射工具、DSE框架)是落地的关键瓶颈。

### [30] CIM-MLC: A Multi-level Compilation Stack for Computing-In-Memory Accelerators
- **Venue:** ASPLOS 2024
- **链接:** [arXiv](https://arxiv.org/abs/2401.12428)
- **协同维度:** CIM硬件抽象 × 多层级编译 × DNN映射调度
- **简介:** 面向通用CIM架构的多层级编译框架,建立统一的硬件层次化抽象与计算模式表示,平均加速3.2倍。

### [31] CIMFlow: Integrated Framework for Systematic Design and Evaluation of Digital CIM
- **Venue:** DAC 2025
- **链接:** [arXiv](https://arxiv.org/abs/2505.01107)
- **协同维度:** 数字CIM架构 × MLIR编译流 × ISA仿真
- **简介:** 基于MLIR的端到端数字CIM设计评估框架,层次化ISA衔接编译与仿真,支持系统化设计空间探索。

### [32] DB-PIM: Efficient SRAM-PIM Architecture by Exploiting Unstructured Bit-Level Sparsity
- **Venue:** DAC 2024
- **链接:** [arXiv](https://arxiv.org/abs/2404.09497)
- **协同维度:** 位级稀疏算法 × SRAM-PIM宏架构 × 算法-架构协同
- **简介:** 利用非结构化位级稀疏性设计专用PIM宏单元与CSD加法树,最高7.69倍加速与83.43%能耗节省。

### [33] Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators
- **Venue:** HPCA 2024
- **链接:** [arXiv](https://arxiv.org/abs/2312.16436)
- **协同维度:** Chiplet架构 × 层级流水映射 × D2D通信优化
- **简介:** 大规模Chiplet加速器的映射与架构协同探索框架,模拟退火优化Die间通信,性能提升1.98倍、能效提升1.41倍。

### [34] INDM: Chiplet-Based Interconnect Network and Dataflow Mapping for DNN Accelerators
- **Venue:** IEEE TCAD 2024
- **链接:** [DOI](https://doi.org/10.1109/TCAD.2023.3332832)
- **协同维度:** Chiplet互连拓扑 × 数据流映射 × 架构分区
- **简介:** 层次化多环片上网络+通信感知数据流映射减少层切换流量拥塞,EDP降低26-74%,延迟降低27-80%。

### [35] DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators
- **Venue:** MICRO 2023
- **链接:** [DOI](https://doi.org/10.1145/3613424.3623797)
- **协同维度:** 可微性能模型 × 硬件DSE × 映射联合优化
- **简介:** 可微分析模型捕获映射与性能关系,梯度下降同时探索硬件设计与映射空间,比贝叶斯优化快12.59倍。

### [36] Understanding the Potential of FPGA-Based Spatial Acceleration for LLM Inference
- **Venue:** ACM TRETS 2024
- **链接:** [arXiv](https://arxiv.org/abs/2312.15159)
- **协同维度:** FPGA空间架构 × LLM算子专用化 × 数据流通信
- **简介:** 系统评估FPGA空间架构用于LLM推理,为不同算子定制硬件单元,GPT解码较A100加速1.9倍、能效提升5.7倍。

### [37] Allo: A Programming Model for Composable Accelerator Design
- **Venue:** PLDI 2024
- **链接:** [DOI](https://doi.org/10.1145/3656401)
- **协同维度:** 加速器编程模型 × 可组合定制原语 × 跨函数优化
- **简介:** 可组合空间加速器编程模型,将计算/存储/通信/数据类型定制解耦为独立原语,支持自底向上类型安全组合。

### [38] Chiplet-Gym: Optimizing Chiplet-based AI Accelerator Design with RL
- **Venue:** IEEE TC 2024
- **链接:** [arXiv](https://arxiv.org/abs/2406.00858)
- **协同维度:** Chiplet PPAC建模 × 强化学习DSE × 2.5D/5.5D封装
- **简介:** 将Chiplet加速器设计空间建模为RL问题,结合模拟退火,吞吐提升1.52倍、能耗降低73%。

### [39] hls4ml: Flexible Open-Source Platform for Deep Learning Acceleration on Reconfigurable Hardware
- **Venue:** ACM TRETS 2026
- **链接:** [arXiv](https://arxiv.org/abs/2512.01463)
- **协同维度:** HLS工具链 × 量化感知编译 × 多框架多后端
- **简介:** 开源ML-to-HLS转换平台,支持PyTorch/Keras/ONNX及量化变体,生成面向多厂商FPGA的数据流加速器设计。

---

## 6. 自动驾驶与车路协同中的实践

自动驾驶感知模型(BEV、点云3D检测、多模态融合)对实时性和功耗有严格要求,软硬件协同加速在该场景从"可选优化"变为"部署刚需"。该方向的特点是: 算法侧以量化/蒸馏为主,硬件侧以嵌入式GPU(Jetson)和FPGA为主要目标平台。

### [40] Q-PETR: Quant-aware Position Embedding Transformation for Multi-View 3D Detection
- **Venue:** arXiv preprint 2025
- **链接:** [arXiv](https://arxiv.org/abs/2502.15488)
- **协同维度:** 量化算法 × BEV感知模型 × TensorRT部署
- **简介:** 针对PETR系列模型INT8量化后mAP下降58.2%的问题,提出量化友好的位置编码变换架构,精度损失控制在1%以内。

### [41] QD-BEV: Quantization-aware View-guided Distillation for Multi-view 3D Object Detection
- **Venue:** ICCV 2023
- **链接:** [ICCV](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_QD-BEV__Quantization-aware_View-guided_Distillation_for_Multi-view_3D_Object_Detection_ICCV_2023_paper.pdf)
- **协同维度:** 量化感知训练 × 知识蒸馏 × BEV感知
- **简介:** 首次系统分析BEV网络量化问题,提出视角引导蒸馏稳定QAT训练。W4A6实现8倍压缩率,37.2% NDS。

### [42] SPADE: Sparse Pillar-based 3D Object Detection Accelerator for Autonomous Driving
- **Venue:** HPCA 2024
- **链接:** [arXiv](https://arxiv.org/abs/2305.07522)
- **协同维度:** 稀疏加速器 × 点云3D检测 × ASIC硬件
- **简介:** 针对PointPillars设计专用ASIC加速器,含动态向量剪枝、稀疏坐标管理硬件和稀疏感知数据流,能效提升4.1-28.8倍。

### [43] PlanKD: Compressing End-to-End Motion Planner for Autonomous Driving
- **Venue:** CVPR 2024
- **链接:** [arXiv](https://arxiv.org/abs/2403.01238)
- **协同维度:** 知识蒸馏 × 端到端规划 × 模型压缩
- **简介:** 首个端到端运动规划器蒸馏框架,基于信息瓶颈提取规划相关特征。26.3M学生模型超越52.9M教师,推理减半至39.7ms。

### [44] QuantV2X: A Fully Quantized Multi-Agent System for Cooperative Perception
- **Venue:** arXiv preprint 2025
- **链接:** [arXiv](https://arxiv.org/abs/2509.03704)
- **协同维度:** 量化 × V2X协同感知 × 通信压缩
- **简介:** 首个全量化V2X协同感知系统,统一模型级量化(INT4权重/INT8激活)与通信级码本量化,保持99.8%原始精度。

### [45] PTQAT: Hybrid Parameter-Efficient Quantization for 3D Perception Tasks
- **Venue:** ICCV 2025 Workshop
- **链接:** [arXiv](https://arxiv.org/abs/2508.10557)
- **协同维度:** PTQ+QAT混合量化 × 3D感知 × 部署优化
- **简介:** 混合量化策略对关键层做QAT微调、其余PTQ,冻结近50%可量化层即达全QAT性能,支持多位宽多架构。

### [46] LiFT: Lightweight, FPGA-Tailored 3D Object Detection Based on LiDAR Data
- **Venue:** DASIP 2025
- **链接:** [arXiv](https://arxiv.org/abs/2501.11159)
- **协同维度:** FPGA硬件 × 点云3D检测 × INT8量化
- **简介:** 面向FPGA 30 GMAC算力约束设计轻量LiDAR检测器,2D cell替代3D体素,AMD Kria K26上实时推理。

### [47] DeployFusion: Deployable Monocular 3D Detection with Multi-Sensor Fusion for Edge
- **Venue:** Sensors 2024
- **链接:** [MDPI](https://www.mdpi.com/1424-8220/24/21/7007)
- **协同维度:** BEV多传感器融合 × TensorRT加速 × 嵌入式部署
- **简介:** EdgeNeXt骨干+两阶段融合网络,TensorRT部署于Jetson Orin NX,138ms/帧实时3D检测。

### [48] UPAQ: A Framework for Real-Time and Energy-Efficient 3D Object Detection in AVs
- **Venue:** arXiv preprint 2025
- **链接:** [arXiv](https://arxiv.org/abs/2501.04213)
- **协同维度:** 剪枝+量化 × 3D检测 × Jetson嵌入式
- **简介:** 半结构化剪枝+混合精度量化,Jetson Orin Nano上对PointPillars实现5.62倍压缩、1.97倍加速。

### [49] DiMA: Distilling Multi-modal Large Language Models for Autonomous Driving
- **Venue:** CVPR 2025
- **链接:** [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Hegde_Distilling_Multi-modal_Large_Language_Models_for_Autonomous_Driving_CVPR_2025_paper.pdf)
- **协同维度:** 多模态蒸馏 × LLM压缩 × 端到端驾驶
- **简介:** 将多模态LLM知识蒸馏至轻量视觉规划器,联合训练结合掩码重建与场景编辑。推理时无需LLM,效率和鲁棒性均优。

---

## 7. 方法论分析: 优化范式、执行模式与性能评估

### 7.1 优化范式: 多目标优化 vs 约束优化

多目标优化的定义不取决于硬件是否可变,而取决于**是否同时优化多个相互冲突的目标**。即便在固定GPU平台上,量化位宽选择(精度 vs 延迟 vs 内存)、剪枝比例分配(精度 vs 计算量)、编译器调度(延迟 vs 内存峰值 vs 吞吐)都是多目标问题。

从本综述的49篇论文来看,多目标问题的处理方式可分为三类:

| 处理方式 | 含义 | 典型论文 | 搜索算法 |
|---------|------|---------|---------|
| **显式多目标Pareto搜索** | 同时优化多个目标,维护Pareto前沿,不预设权重 | CASCO, DOSA, Gemini, Chiplet-Gym, MicroScopiQ | MoBo, NSGA-II, 模拟退火, RL |
| **加权标量化** | 将多目标加权合并为单一标量后求解 | CASCO的EDP=Energy×Delay, DOSA的加权PPA, SOFA | 遗传算法, 梯度下降 |
| **目标固定+约束转化** | 固定部分目标为约束条件,优化剩余单目标 | AWQ(固定位宽→最大化精度), FlashAttention-2(固定精度→最小化延迟), vLLM(最大化吞吐) | 贪心, 启发式, 解析优化 |

**关键发现:** 大部分工作选择了"目标固定+约束转化"的简化路径,将本质上的多目标问题转化为单目标求解。这降低了搜索难度,但可能错过了更优的设计点。真正维护Pareto前沿做完整多目标搜索的集中在加速器DSE方向,数量较少但方法论更完整。

### 7.2 执行模式: 串行 vs 单向感知 vs 双向交互

软件侧优化(量化/剪枝/编译)与硬件侧优化(架构设计/映射/调度)之间的执行关系是协同设计的核心问题:

| 执行模式 | 占比 | 典型论文 | 信息流向 |
|---------|------|---------|---------|
| **完全串行** | ~50% | AWQ→TinyChat, QD-BEV→TensorRT, PlanKD→部署 | 算法优化 → 编译优化 → 硬件执行,各步独立 |
| **单向感知(HW→SW)** | ~30% | FlashAttention-2, OliVe, Panacea, Souffle | 算法设计时**感知**硬件特性(SRAM层级/内存对齐/roofline),但不改变硬件 |
| **双向交互** | ~20% | CASCO, DOSA, Gemini, Chiplet-Gym, CIMFlow | 硬件配置和软件策略在搜索过程中相互影响,形成闭环 |

```
完全串行:
  算法优化(量化/蒸馏) → 固定 → 编译器优化 → 固定 → 硬件执行
  后步无法反馈前步

单向感知:
  硬件特性(SRAM大小/对齐要求/带宽) → 指导算法设计
  例: FlashAttention利用GPU SRAM层级做tiling
  例: OliVe设计内存对齐的量化编码
  算法"看到"硬件约束,但硬件本身不变

双向交互:
  硬件配置 ←→ 软件策略 (每轮迭代中相互影响)
  例: CASCO中HW配置约束融合组大小,融合EDP反馈驱动HW搜索方向
  例: DOSA中可微模型同时对HW参数和映射参数求梯度
```

**为什么串行和单向感知占主导?** 因为目前大部分团队面对的是固定硬件平台(NVIDIA GPU/Jetson),没有"硬件可调"的自由度。双向交互主要出现在定制加速器(ASIC/FPGA/CIM/Chiplet)的设计场景中。

### 7.3 性能评估: 如何快速得到优化性能

快速且准确的性能评估是搜索效率的核心瓶颈。评估一个设计点越快,搜索算法能探索的空间就越大。从本综述看,有以下四种主要方法:

**方法A: 解析性能模型 (最成熟)**

| 工具 | 建模对象 | 使用论文 | 评估速度 |
|------|---------|---------|---------|
| Timeloop | 给定HW+映射,估算能耗/延迟/面积 | CASCO, DOSA, Gemini, INDM | 毫秒级 |
| MAESTRO | 数据流级别性能建模 | AIRCHITECT V2, DIGAMMA | 毫秒级 |
| Optimus | 层融合的片外通信代价 | CASCO | 毫秒级 |
| Roofline模型 | GPU算子的计算/内存瓶颈判断 | MCFuser, Souffle | 微秒级 |

优点: 极快,可嵌入搜索循环内层。缺点: 与真实硬件有精度gap,需校准。

**方法B: 可微分析模型 (最先进)**

DOSA(MICRO 2023)将Timeloop式分析模型改造为可微分形式:
- 传统: 分析模型是黑盒 → 只能用贝叶斯优化/遗传算法(采样效率低)
- DOSA: 分析模型可微 → 梯度下降同时优化HW+映射参数
- 附加: 数据驱动模型补偿分析模型与真实硬件的差距
- 结果: 比贝叶斯优化快12.59倍

**方法C: 学习型代理模型 (新兴方向)**

| 代理模型 | 论文 | 优势 |
|---------|------|------|
| 高斯过程 | CASCO(MoBo), Coflex | 少量样本即可建模,内置不确定性估计 |
| 强化学习 | Chiplet-Gym | 适合序列决策,可学习跨步骤依赖 |
| 扩散模型 | DiffuSE | 学习从目标到参数的逆映射,一次采样多个候选 |
| 神经网络 | AIRCHITECT V2 | 在大量模拟数据上预训练,泛化到新工作负载 |

优点: 可逐步提高精度。缺点: 需要初始训练数据,冷启动问题。

**方法D: 编译器级快速评估 (固定硬件场景)**

对固定GPU平台不需要硬件级建模,编译器级评估即可:

| 方法 | 论文 | 做法 |
|------|------|------|
| 解析cost model | MCFuser, Souffle | 基于roofline和内存带宽估算kernel性能 |
| Profiling引导 | DeepCompile, GraphPipe | 先profiling各算子开销,再优化调度 |
| LLM驱动搜索 | Reasoning Compiler | LLM生成变换提议+MCTS,36个样本达2.5倍加速 |

**总结:** 性能评估方法的选择取决于场景:
- 定制硬件DSE → Timeloop/MAESTRO解析模型 + 代理模型加速
- 固定GPU优化 → Roofline/profiling + 编译器cost model
- 追求极致搜索效率 → DOSA的可微模型或DiffuSE的扩散模型

---

## 8. 趋势与总结

### 8.1 各方向技术趋势

| 方向 | 关键趋势 |
|------|---------|
| **LLM推理加速** | prefill/decode阶段分离成为共识; KV-cache管理从"单机优化"走向"集群级调度"; 投机解码正在从研究走向生产 |
| **量化与稀疏** | LLM量化从均匀精度走向混合精度/混合数据类型; 离群值处理成为核心挑战; 非结构化稀疏与结构化硬件的鸿沟正在被弥合 |
| **编译器** | 算子融合范围持续扩大(从单算子到跨表达式); LLM被用于驱动编译优化(Reasoning Compiler); 稀疏模型的编译支持开始完善 |
| **新型架构** | CIM从器件级走向系统级(需要编译栈配套); Chiplet成为大规模AI加速器的主流封装方案; FPGA在LLM推理中展现能效优势 |
| **自动驾驶** | BEV模型的量化部署是当前热点; 点云处理已有专用ASIC(SPADE); V2X协同感知的量化压缩刚刚起步 |

### 8.2 尚未充分探索的方向

1. **V2X协同感知的软硬件协同**: 目前仅有QuantV2X一篇尝试,多车/路侧协同的通信-计算联合优化空间巨大
2. **BEV模型的专用加速器**: 现有工作集中在量化/蒸馏的算法侧,BEV view transform等特殊算子的硬件加速尚未充分探索
3. **端到端自动驾驶模型的全栈优化**: 从感知到规划的端到端模型正在兴起,其部署优化才刚刚开始(PlanKD, DiMA)
4. **CIM/PIM上的LLM推理**: 存内计算对memory-bound的LLM解码有天然优势,但系统级工作很少
5. **跨层全栈协同**: 同时优化网络结构+量化策略+编译调度+硬件配置的闭环框架仍然稀缺
