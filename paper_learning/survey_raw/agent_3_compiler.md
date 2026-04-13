# Agent-3: 编译器与系统软件栈优化 (9篇)

> Agent使用了WebSearch,结果质量较高

### 1. Souffle: Optimizing Deep Learning Inference via Global Analysis and Tensor Expressions
- **Venue:** ASPLOS 2024
- **链接:** https://doi.org/10.1145/3617232.3624858
- **协同维度:** 编译器 × 算子融合 × GPU代码生成
- **简介:** 基于张量表达式构建全局依赖图,跨算子边界数据流分析与调度优化。A100上较TensorRT加速3.7倍。

### 2. MCFuser: High-Performance and Rapid Fusion of Memory-Bound Compute-Intensive Operators
- **Venue:** SC 2024
- **链接:** https://doi.org/10.1109/SC41406.2024.00040
- **协同维度:** 算子融合 × 搜索空间 × 性能模型
- **简介:** 针对内存密集型算子链构建高层搜索空间结合解析性能模型。A100上较Ansor加速5.9倍。

### 3. GraphPipe: Improving Performance and Scalability of DNN Training with Graph Pipeline Parallelism
- **Venue:** ASPLOS 2025
- **链接:** https://doi.org/10.1145/3669940.3707220
- **协同维度:** 计算图拓扑 × 流水线并行 × 调度优化
- **简介:** 图流水线并行将DNN分割为DAG形式流水线阶段,保留拓扑实现并发执行。较PipeDream加速1.6倍。

### 4. ClusterFusion: Expanding Operator Fusion Scope for LLM Inference
- **Venue:** NeurIPS 2025
- **链接:** https://arxiv.org/abs/2508.18850
- **协同维度:** 算子融合 × 片上通信 × LLM推理
- **简介:** 引入集群级通信原语将QKV投影+注意力+输出投影融合为单一kernel。H100上延迟降低1.61倍。

### 5. Arnold: Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment
- **Venue:** NeurIPS 2025
- **链接:** https://arxiv.org/abs/2509.15940
- **协同维度:** 网络拓扑 × 通信调度 × 分布式训练
- **简介:** 拓扑感知调度算法对齐通信组与物理网络。9600+GPU生产环境性能提升10.6%。

### 6. Reasoning Compiler: LLM-Guided Optimizations for Efficient Model Serving
- **Venue:** NeurIPS 2025
- **链接:** https://arxiv.org/abs/2506.01374
- **协同维度:** LLM推理 × 编译器自动调优 × MCTS搜索
- **简介:** 用LLM生成变换提议以MCTS引导搜索,36个样本达2.5倍加速,样本效率比TVM高16倍。

### 7. FuseFlow: A Fusion-Centric Compilation Framework for Sparse Deep Learning on Streaming Dataflow
- **Venue:** ASPLOS 2026
- **链接:** https://doi.org/10.1145/3779212.3790165
- **协同维度:** 稀疏编译器 × 跨表达式融合 × 数据流架构
- **简介:** 首个支持稀疏算子跨表达式融合的端到端编译器,稀疏模型加速2.7倍。

### 8. RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators
- **Venue:** ASPLOS 2026
- **链接:** https://doi.org/10.1145/3779212.3790209
- **协同维度:** 级联归约 × 符号推导 × 硬件感知代码生成
- **简介:** 级联归约融合形式化方法,符号推导引擎自动识别可融合模式,较TVM加速2-5倍。

### 9. DeepCompile: A Compiler-Driven Approach to Optimizing Distributed Deep Learning Training
- **Venue:** arXiv preprint
- **链接:** https://arxiv.org/abs/2504.09983
- **协同维度:** 分布式训练 × 计算图变换 × 通信重叠
- **简介:** 编译图变换实现通信-计算重叠,较ZeRO-3加速1.28倍。
