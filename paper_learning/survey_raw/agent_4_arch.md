# Agent-4: 新型计算架构协同 (10篇)

> Agent使用了WebSearch,结果质量较高

### 1. CIM-MLC: A Multi-level Compilation Stack for Computing-In-Memory Accelerators
- **Venue:** ASPLOS 2024
- **链接:** https://arxiv.org/abs/2401.12428
- **协同维度:** CIM硬件抽象 × 多层级编译 × DNN映射调度
- **简介:** 面向通用CIM架构的多层级编译框架,统一CIM硬件层次化抽象,平均加速3.2倍。

### 2. CIMFlow: Integrated Framework for Systematic Design and Evaluation of Digital CIM Architectures
- **Venue:** DAC 2025
- **链接:** https://arxiv.org/abs/2505.01107
- **协同维度:** 数字CIM架构 × MLIR编译流 × ISA仿真评估
- **简介:** 基于MLIR的端到端数字CIM设计评估框架,层次化ISA衔接编译与仿真。

### 3. DB-PIM: Towards Efficient SRAM-PIM Architecture Design by Exploiting Unstructured Bit-Level Sparsity
- **Venue:** DAC 2024
- **链接:** https://arxiv.org/abs/2404.09497
- **协同维度:** 位级稀疏算法 × SRAM-PIM宏架构 × 算法-架构协同
- **简介:** 利用非结构化位级稀疏性设计PIM宏单元,最高7.69倍加速与83.43%能耗节省。

### 4. Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators
- **Venue:** HPCA 2024
- **链接:** https://arxiv.org/abs/2312.16436
- **协同维度:** Chiplet架构探索 × 层级流水映射 × D2D通信优化
- **简介:** 大规模Chiplet加速器映射与架构协同探索,模拟退火优化Die间通信,性能提升1.98倍。

### 5. INDM: Chiplet-Based Interconnect Network and Dataflow Mapping for DNN Accelerators
- **Venue:** IEEE TCAD 2024
- **链接:** https://doi.org/10.1109/TCAD.2023.3332832
- **协同维度:** Chiplet互连拓扑 × 数据流映射 × 架构分区探索
- **简介:** 层次化多环片上网络+通信感知数据流映射,EDP降低26-74%。

### 6. DOSA: Differentiable Model-Based One-Loop Search for DNN Accelerators
- **Venue:** MICRO 2023
- **链接:** https://doi.org/10.1145/3613424.3623797
- **协同维度:** 可微性能模型 × 硬件DSE × 映射空间联合优化
- **简介:** 可微分析模型用梯度下降同时探索硬件设计与映射空间,比贝叶斯优化快12.59倍。

### 7. Understanding the Potential of FPGA-Based Spatial Acceleration for LLM Inference
- **Venue:** ACM TRETS 2024
- **链接:** https://arxiv.org/abs/2312.15159
- **协同维度:** FPGA空间架构 × LLM算子专用化 × 数据流通信优化
- **简介:** 系统评估FPGA空间架构用于LLM推理,GPT解码较A100加速1.9倍、能效提升5.7倍。

### 8. Allo: A Programming Model for Composable Accelerator Design
- **Venue:** PLDI 2024
- **链接:** https://doi.org/10.1145/3656401
- **协同维度:** 加速器编程模型 × 可组合定制原语 × 跨函数层次优化
- **简介:** 可组合空间加速器编程模型,解耦计算/存储/通信/数据类型定制为独立原语。

### 9. Chiplet-Gym: Optimizing Chiplet-based AI Accelerator Design with RL
- **Venue:** IEEE TC 2024
- **链接:** https://arxiv.org/abs/2406.00858
- **协同维度:** Chiplet PPAC建模 × 强化学习DSE × 2.5D/5.5D封装
- **简介:** RL+模拟退火优化Chiplet资源分配和封装架构,吞吐提升1.52倍、能耗降低73%。

### 10. hls4ml: A Flexible, Open-Source Platform for Deep Learning Acceleration on Reconfigurable Hardware
- **Venue:** ACM TRETS 2026
- **链接:** https://arxiv.org/abs/2512.01463
- **协同维度:** HLS工具链 × 量化感知编译 × 多框架多后端协同
- **简介:** 开源ML-to-HLS转换平台,支持多框架和量化变体,生成FPGA数据流加速器。
