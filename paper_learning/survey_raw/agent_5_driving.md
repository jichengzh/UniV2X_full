# Agent-5: 自动驾驶/V2X部署加速 (10篇)

> Agent使用了WebSearch,结果质量较高

### 1. Q-PETR: Quant-aware Position Embedding Transformation for Multi-View 3D Object Detection
- **Venue:** arXiv preprint 2025
- **链接:** https://arxiv.org/abs/2502.15488
- **协同维度:** 量化算法 × BEV感知模型 × TensorRT部署
- **简介:** 针对PETR系列INT8量化精度下降58.2%问题,提出量化友好位置编码变换,mAP和NDS下降控制1%以内。

### 2. QD-BEV: Quantization-aware View-guided Distillation for Multi-view 3D Object Detection
- **Venue:** ICCV 2023
- **链接:** https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_QD-BEV__Quantization-aware_View-guided_Distillation_for_Multi-view_3D_Object_Detection_ICCV_2023_paper.pdf
- **协同维度:** 量化感知训练 × 知识蒸馏 × BEV感知
- **简介:** 首次系统分析BEV网络量化,视角引导蒸馏稳定QAT,W4A6实现8倍压缩。

### 3. SPADE: Sparse Pillar-based 3D Object Detection Accelerator for Autonomous Driving
- **Venue:** HPCA 2024
- **链接:** https://arxiv.org/abs/2305.07522
- **协同维度:** 稀疏加速器 × 点云3D检测 × ASIC硬件
- **简介:** 针对PointPillars设计专用加速器,动态向量剪枝+稀疏感知数据流,能效提升4.1-28.8倍。

### 4. PlanKD: Compressing End-to-End Motion Planner for Autonomous Driving
- **Venue:** CVPR 2024
- **链接:** https://arxiv.org/abs/2403.01238
- **协同维度:** 知识蒸馏 × 端到端规划 × 模型压缩
- **简介:** 首个端到端规划器蒸馏框架,26.3M学生模型超越52.9M教师,推理39.7ms。

### 5. QuantV2X: A Fully Quantized Multi-Agent System for Cooperative Perception
- **Venue:** arXiv preprint 2025
- **链接:** https://arxiv.org/abs/2509.03704
- **协同维度:** 量化 × V2X协同感知 × 通信压缩
- **简介:** 首个全量化V2X协同感知系统,INT4权重/INT8激活+通信码本量化,保持99.8%精度。

### 6. PTQAT: A Hybrid Parameter-Efficient Quantization Algorithm for 3D Perception Tasks
- **Venue:** ICCV 2025 Workshop
- **链接:** https://arxiv.org/abs/2508.10557
- **协同维度:** PTQ+QAT混合量化 × 3D感知 × 部署优化
- **简介:** 混合量化策略,冻结近50%可量化层即达全QAT性能,支持4-bit多架构。

### 7. LiFT: Lightweight, FPGA-Tailored 3D Object Detection Based on LiDAR Data
- **Venue:** DASIP 2025
- **链接:** https://arxiv.org/abs/2501.11159
- **协同维度:** FPGA硬件 × 点云3D检测 × INT8量化
- **简介:** 面向FPGA约束设计轻量LiDAR检测器,2D cell替代3D体素,AMD Kria K26上实时推理。

### 8. DeployFusion: Deployable Monocular 3D Detection with Multi-Sensor Fusion for Edge Devices
- **Venue:** Sensors 2024
- **链接:** https://www.mdpi.com/1424-8220/24/21/7007
- **协同维度:** BEV多传感器融合 × TensorRT加速 × 嵌入式部署
- **简介:** EdgeNeXt骨干+两阶段融合,TensorRT部署于Jetson Orin NX,138ms/帧实时3D检测。

### 9. UPAQ: A Framework for Real-Time and Energy-Efficient 3D Object Detection in Autonomous Vehicles
- **Venue:** arXiv preprint 2025
- **链接:** https://arxiv.org/abs/2501.04213
- **协同维度:** 剪枝+量化 × 3D检测 × Jetson嵌入式
- **简介:** 半结构化剪枝+混合精度量化,Jetson Orin Nano上5.62倍压缩、1.97倍加速。

### 10. DiMA: Distilling Multi-modal Large Language Models for Autonomous Driving
- **Venue:** CVPR 2025
- **链接:** https://openaccess.thecvf.com/content/CVPR2025/papers/Hegde_Distilling_Multi-modal_Large_Language_Models_for_Autonomous_Driving_CVPR_2025_paper.pdf
- **协同维度:** 多模态蒸馏 × LLM压缩 × 端到端驾驶
- **简介:** 多模态LLM蒸馏至轻量视觉规划器,推理时无需LLM,鲁棒性和效率均优。
