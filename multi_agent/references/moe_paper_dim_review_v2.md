# MoE 论文优化维度 × 我们搜索空间 — 逐节对照分析 v2.0

> 作者: hw-optimizer
> 日期: 2026-06-04
> 来源论文: "Toward Optimal Mixture of Experts System for 3D Object Detection: A Game of Accuracy, Efficiency and Adaptivity"
> 期刊: IEEE TPAMI Vol.48 No.1, January 2026 (pp.914-931)
> 对照基准: `dims_hardware_v2.md` + `NEXT_PHASE_plan_v1.md §二.3` + `team_charter_v1.md §3 已验证勘误`
>
> **v2 修正说明 (ISS-022)**:
> - ① 删除 v1 中编造数字"BEVDet 496MB/8306ms" — PDF 全表不存在;改用论文实际值
> - ② 6064→397ms 改为 EMOS 同模型 ONNX 系统优化自身前后对比(§V-C line 1144),不是模型架构差异;消除 v1 §6.2 自相矛盾
> - ③ H-NEW1 场景自适应 engine 切换补 caveat: 与我方 pathE 负结果 ISS-018 冲突,属下一阶段更难模型工作

---

## 0. 论文结构速览

| TPAMI 页 | 章节 | 核心内容 |
|---------|------|---------|
| p.914-915 | §I Introduction | EMOS 动机; 两大设计思路; 贡献摘要 |
| p.915-917 | §II Related Work | §II-A 3D 检测算法; §II-B ML 编译平台(ONNX/TVM/TRT/OneDNN/TorchDynamo) |
| p.917-918 | §III-A | Expert Utility 评估函数 Ua/Ue |
| p.918-920 | §III-B | Expert 类型定义: EPEs/LAPEs/FAPEs/Emergency Expert |
| p.919-920 | §III-C | MoE Ensemble Strategy; 场景分配函数 π(s) |
| p.919-920 | §III-D | Adaptive Multimodal Expert Router (AMER): AMDB + MoE Switch |
| p.919-921 | §III-E | 分层训练策略; 长尾效应缓解; 正则化 |
| p.921 | §IV-A | System Innovation 1/2: 稀疏卷积 + Multiscale Pooling |
| p.921-923 | §IV-B | System Innovation 3/4: Memory Opt + 计算图优化 |
| p.923-928 | §V A-D | 实验: 数据集/指标/硬件/结果/消融 |
| p.928 | §VI Limitations | 当前局限; 未来工作 |

---

## 1. 主对照表 — EMOS 每个优化方法 × 我方搜索空间

分类代码:
- **(a) 已覆盖/等价** — 我方有对应真测实现
- **(b) TRT-auto 已覆盖** — TRT-auto build 已包含;额外操作无增益
- **(c) 平台差异/非真冲突** — 场景不同,机制差异,不推翻我方结论
- **(d) 潜在新维度** — 值得评估但尚未纳入

| 论文章节/页 | 优化方法名称 | 一句话说明 | 论文声称效果 [平台/模型] | 我方对应处理 | **分类** | 我方真测/待测状态 |
|------------|-------------|-----------|-------------------------|-------------|---------|--------------|
| §I Abstract (p.914) | **Accelerator-aware scheduling** (ONNX/TensorRT) | 用 ONNX+TensorRT 后端执行 MoE 推理,加速器感知调度 | 15.35× vs efficiency-oriented baselines on nuScenes [Jetson AGX Orin, EMOS 333MB/32MB] [论文声称, p.925] | 我方全流程基础设施即 ONNX→TRT build;pipeline 完全等价 | **(a) 已覆盖** | ✅ 全流程真测 (stage_a_cache, collab2 engines) |
| §I Abstract (p.914) | **Zero-copy buffering** (I/O) | CPU-GPU 传输避免冗余拷贝,减少 I/O latency | 未单独量化 [抽象描述] [论文声称] | F3 unified memory (Orin UMA 零拷贝); 4090 data_ptr 直传 | **(a) 已覆盖** | ✅ 文档覆盖 (dims_hardware_v2 §F3) |
| §I Abstract (p.914) | **Overlapped I/O-compute** (chunk prefetch) | batch 切 chunk,处理 chunk i 同时 prefetch i+1 | 未单独量化;贡献于整体 e2e 降低 [论文声称] | **⚠ 见 §3 深析** — 与我方 E_pipeline 1.08× 证伪**机制不同**(非真冲突) | **(c) 平台差异** | ✅ 我方 E_pipeline 已真测;batch=1 H2D≈0 |
| §II-B (p.917) | **ONNX Runtime 图优化** (ONNX [74]) | 利用 ONNX Runtime 的 multi-level IR 做算子融合、图重写 | 论文仅引 ONNX 文献,无独立量化 [引用文献层面] | 我方 ONNX→TRT build 路径已包含;TRT 内部图优化覆盖 | **(b) TRT-auto 覆盖** | ✅ TRT 内建;不需额外操作 |
| §II-B (p.917) | **TensorRT 部署后端** (TRT [30]) | 用 TRT 做算子融合、量化、kernel tuning 实现低延迟推理 | 论文直接用作后端;无论文独立量化 [引用文献] | 我方核心基础设施;完全等价 | **(a) 已覆盖** | ✅ TRT 10.x (4090) + TRT 8.5 (Orin) 全链路真测 |
| §III-A (p.917) | **Accuracy Utility Function U_a** | 量化 accuracy-per-parameter + accuracy-per-data (对数惩罚) | 用于 expert 排序和 ensemble 策略设计 [方法定义层] | 我方无此函数;我方直接用 (AP70, latency, energy) Pareto 轴 | **(d) 概念参考** | ⚠ 我方有效 Pareto 轴,U_a 形式化不在优先列表 |
| §III-A (p.917) | **Efficiency Utility Function U_e** | 量化 accuracy-per-latency-per-size (含跨平台缩放系数 ν_h) | 用于 expert 排序;ν_h=3.25 (RTX3080→Orin), 4.35 (A4000→Orin) [EMOS 论文数值] | 我方 M2 f-函数 (R²>0.995, 4090→Orin 映射) 等价;比 ν_h 单系数更精确 | **(a) 已覆盖 (更优)** | ✅ M2 f-函数真测 (results/m2_latency_mapping_f.json) |
| §III-B (p.918-919) | **Efficiency-Prioritized Experts (EPEs)** | LiDAR-only 轻量专家,BEV 2D encoder, 最快;仅在所有目标清晰时激活 | EPE alone: 32MB, KITTI AP car-hard 75.33 [Jetson, EMOS] [论文声称, Table V/VI] | 单模型 PyramidFusion 不含多专家架构;EPE 等价于低精度/低剪枝配置 | **(d) 概念迁移,见§4** | ⚠ 见 H-NEW1 caveat (ISS-018 冲突) |
| §III-B (p.918-919) | **Lightweight Accuracy Experts (LAPEs)** | LiDAR+图像轻量双模专家;处理部分困难/远距目标 | tri-expert: 333MB, 397.3ms [Jetson, EMOS] [论文声称, p.924] | 等价于中等精度配置;单模型架构无法动态选择 | **(d) 概念迁移** | ⚠ 见 H-NEW1 caveat |
| §III-B (p.919) | **Full-feature Accuracy Experts (FAPEs)** | 完整多模态专家,最重;用于不清晰+远距目标 | **论文未单列 FAPE 单独尺寸**; 论文实值: tri-expert 333MB / EPE-only 32MB (Table V p.927) [论文声称] | 等价于高精度完整配置 | **(d) 概念迁移** | ⚠ 见 H-NEW1 caveat |
| §III-B (p.919) | **Emergency Expert** | 速度/距离触发的应急路径;跳过预处理直接消费原始传感器数据 | 安全保障路径;延迟约束触发 [EMOS 架构] | 我方无应急路径;与框架功能不对应 | 不适用 | — |
| §III-C (p.919) | **MoE 场景分配函数 π(s)** | 基于场景难度 F̂(s) 阈值 τ 动态路由到 EPEs 或 APEs | 动态路由使平均模型大小 ≈ 333.3MB (激活加权 eq.14, p.923) [EMOS 数据] | 单模型无此函数;概念上对应"runtime engine 选择" | **(d) 概念迁移** | ⚠ 见 H-NEW1 caveat |
| §III-D (p.919-920) | **Adaptive Multimodal Data Bridge (AMDB)** | LiDAR+图像双分支融合;预生成 proposal regions + 置信度分数供 MoE Switch 使用 | 生成 proposal regions 供 expert 选择 [架构组件] | 我方 PyramidFusion 的 pyramid_backbone + weighted_fuse 等价于特征融合路径 | **(a) 等价架构** | ✅ 已真测 (stage_a_cache collab2 engines) |
| §III-D (p.919-920) | **MoE Switch (场景感知 expert 调度)** | 基于 proposal region confidence + 目标距离,在 EPE/LAPE/FAPE 间动态切换 | 减少平均推理时间同时维持精度 [EMOS 架构] | 单模型无需切换;若做 runtime engine 切换需此类路由逻辑 | **(d) 概念迁移** | ⚠ 见 H-NEW1 caveat |
| §III-E (p.920-921) | **层次化多模态 backpropagation** | 对 LiDAR 分支/图像分支/最终输出 三路独立监督,加速收敛 (Fig.3) | 前20个 epoch loss 曲线明显收敛更快 [EMOS 消融 Fig.9, p.927] [论文声称] | 我方训练方法独立,不依赖 EMOS 架构;HEAL 训练有自己的损失结构 | 不适用(训练策略差异) | — |
| §III-E (p.921) | **长尾效应缓解:数据子集划分** | 按 expert 将训练数据分为 K 个子集,每个 expert 等频曝光 | 防止 EPEs 在稀少场景上欠训练 [EMOS 训练] | 我方单模型训练;无 expert 专属子集需求 | 不适用 | — |
| §III-E (p.921) | **Expert 正则化损失 R_cls/R_reg** | focal loss 变体 + L2 正则防 expert 遗忘,格式见 eq.10-13 | 配合长尾缓解;参数 α=0.55 car / γ=0.2-0.8 | 我方无 expert 专属正则;HEAL 训练有标准 focal loss | 不适用 | — |
| §IV-A §Innov.1 (p.921) | **Edge-optimized 3D Sparse Convolution** | 自定义硬件感知稀疏 3D conv,仅计算非空 voxel,并行 FMA kernel | 65-80% fewer ops vs dense conv [EMOS 声称, p.921] [论文声称] | 我方 PyramidFusion 已用 spconv 库,等价受益;ONNX 路径已优化 | **(a) 已覆盖** | ✅ spconv 路径已真测;TRT engine 内建 |
| §IV-A §Innov.2 (p.921) | **Multiscale Pooling (cache-efficient fusion)** | Sobel 引导的自适应下采样:图像特征从 ~1M → ~60K voxel,再与 LiDAR 20K voxel 融合 | 6064.2ms → **397.3ms** (EMOS 同模型 ONNX 系统优化前后对比,含本方法) [Jetson, nuScenes, EMOS 同模型] [论文声称, §V-C 消融, p.928] | 我方 PyramidFusion 无摄像头分支;无此 voxel cache 问题;不适用 | 不适用(模态差异) | — |
| §IV-B §Innov.3-1 (p.921-922) | **Memory Optimization 1: Shared Memory/L1L2 配置** | 配置 ONNX 编译器最大化 shared memory 命中率;优化 warp-level 执行顺序 | 减少全局内存访问,提升 SM 利用 [EMOS 架构] [论文声称] | TRT tactic 选择 (B1) 已内含 warp-level 优化;我方 tactic 扫描实测差异 5-9% | **(b) TRT-auto 覆盖** | ✅ H2 B1 tactic 扫描真测;无额外增益 |
| §IV-B §Innov.3-2 (p.922) | **Memory Optimization 2: Parallel Communication+Computation (Chunk Prefetch)** | batch 切 chunk,chunk i+1 异步 prefetch 同时计算 chunk i;隐藏 CPU-GPU 传输延迟 | Thread mgmt Fig.6: LiDAR Branch 216→75ms (-65%); FAPEs 825→282ms (-66%) [Jetson, EMOS 系统] [论文声称] | **见§3 深析** — EMOS 隐藏 CPU→GPU PCIe H2D 传输(异构操作);我方证伪的是 GPU-only SM 时间复用。**机制不同,不冲突** | **(c) 平台差异** | ✅ 我方 Orin UMA = H2D≈0;4090 pipeline 1.08× 已真测 |
| §IV-B §Innov.3-3 (p.922) | **Memory Optimization 3: Thread Management (4-boundary staged release)** | 定义 4 个执行边界,每到边界 CPU 终止完成线程并释放其地址空间;最小化 peak memory | Fig.6 数据为系统综合效果(含所有 Innovation),无单独量化 [论文声称] | CPU 线程管理是 deployment 约束,非搜索变量;我方 GPU/TRT 执行不需要此机制 | **(b) 不适用/TRT 内建** | — |
| §IV-B §Innov.3-4 (p.922) | **Memory Optimization 4: GPU Prefix Scan for Sparse CNN** | 利用 GPU prefix scan 将稀疏累加 O(N)→O(log N),提升 load balance | 参考文献 [88] 算法描述;无单独量化 [论文声称] | spconv 库已内建前缀扫描等稀疏优化;TRT/ONNX 路径受益 | **(a) 已覆盖** | ✅ spconv 路径已使用 |
| §IV-B §Innov.4-a (p.923) | **Computation Graph Optimization: Graph Rewriting** | constant folding / redundant node elimination / semantic-preserving operator fusion (Conv+Add → single node) | 第 1 阶段图重写 (分区前);贡献于综合加速 [EMOS 系统] [论文声称] | TRT build 内建 constant folding + 算子融合;PyramidFusion 无 LayerNorm 块(仅 BatchNorm,被 TRT 折叠进 conv),额外增益极小 | **(b) TRT-auto 覆盖** | ✅ TRT 内建;对 Pyramid 无额外操作需要 |
| §IV-B §Innov.4-b (p.923) | **Computation Graph Optimization: Complex Node Fusion** | GEMM + activation 融合消除中间算术;LayerNorm 融合 (ReduceMean+Sub+Pow+Sqrt+Div → LayerNormalization) | 第 2 阶段复杂节点融合 (分区后);贡献于综合加速 [EMOS 系统] [论文声称] | LayerNorm 融合对含 Transformer 模型有效;PyramidFusion 无 self-attention/LayerNorm 块(仅 agent-加权 softmax 融合,单算子 TRT 原生)→ 无额外增益 | **(b) TRT-auto 覆盖 (Pyramid 不适用)** | ⚠ 对 V2X-ViT (未来 Transformer 阶段) 可能有价值 |
| §V-A (p.924) | **ONNX+TensorRT 部署后端** (inference 全链路) | ONNX Runtime + TensorRT 实现 end-to-end 推理;workspace + FP16 precision mode 针对 Orin 优化 | Jetson Orin AGX 上真测; workspace=max / FP16 [论文声称, §V-A] | 我方完全相同路径;4090 TRT 10.x + Orin TRT 8.5 | **(a) 已覆盖** | ✅ 全链路真测 |
| §V-A (p.924) | **CPU thread 限制 (≤4 线程)** | Jetson 推理限制最多 4 个 CPU 线程以保证 GPU/DLA 利用稳定 | 保证跨 CPU-GPU 并发稳定 [Jetson 部署约束] [论文声称] | 部署约束,非搜索维度;建议在 orin_agx.yaml 补充 `cpu.max_inference_threads` | 不适用(部署约束) | — |
| §V-A (p.924) | **跨平台 latency 缩放系数 ν_h** | 从 desktop GPU (RTX3080/A4000) 经验缩放外推到 Jetson Orin 推理延迟 | ν_h=3.25 (RTX3080→Orin); ν_h=4.35 (A4000→Orin) [EMOS 经验单点系数] [论文声称] | 我方 M2 f-函数 (R²>0.995, 4090→Orin) 更精确;已用于跨平台预测 | **(a) 已覆盖 (更精确)** | ✅ M2 f-函数真测 |
| §V-C (p.925) | **EMOS on KITTI** | tri-expert 系统在 KITTI 上 AP cars +1.95% / pedestrians +1.31% / cyclists +6.24%; 371.3ms@Jetson (2.6×@comparable) | [Jetson AGX Orin, KITTI, EMOS 333.3MB] [论文声称] | 不同平台/模型/数据集;我方 DAIR 单帧 PyramidFusion 不可比较 | 参考 | — |
| §V-C (p.925) | **EMOS on nuScenes (15.35×)** | EMOS 在 nuScenes 上 397.3ms/frame @Jetson, 15.35× faster than efficiency-oriented baselines, 67% of parameter count | [Jetson AGX Orin, nuScenes, EMOS 333.3MB, vs efficiency-oriented SOTAs] [论文声称] | **❌ v1 误写 BEVDet 496MB/8306ms 已删除** ← 这一比较对象是 efficiency-oriented baselines (非 accuracy-oriented BEVDet) | 参考 (不同平台/模型) | — |
| §V-C ablation (p.928) | **ONNX-based system optimization 效果 (6064→397ms)** | 同一 EMOS 模型,开启 vs 关闭 ONNX 系统优化(multiscale pooling + 计算图融合)的消融对比 | **同模型** off→on: 6064.2ms → 397.3ms [Jetson, nuScenes, EMOS 同模型] [论文声称] | **❌ v1 误归因"模型架构差异"已修正** ← 这是 EMOS 自身系统优化(work-reduction + 图融合),非模型换型 | **(c) 我方 spconv+TRT-auto 覆盖其机制** | ✅ 我方 TRT-auto 已内含类似优化 |

---

## 2. ISS-022 三处修正的完整说明

### ① 删除编造数字 "BEVDet 496MB/8306ms"

**v1 错误**: "论文对比的 baseline 是 BEVDet (large, 496MB, 8306ms)"

**PDF 核实**: 全文 grep "496" 和 "8306" → **均不存在**。BEVDet 在论文中仅作精度参考方法出现(相关工作 p.916, 图表 p.926)。论文实值: tri-expert ensemble ≈ 333MB, EPE alone = 32MB (Table V p.927); **FAPE 单独尺寸论文未单列**,不填近似数字。

**正确数字来源**:
- 15.35× speedup: §V-C p.925 原文: "15.35 times faster than **efficiency-oriented baselines**" (非 BEVDet)
- EMOS 模型大小: §V-A p.924 原文: "approximately **333.3 MB** with 32-bit (FP32) precision" (tri-expert); Table V p.927: EPE only = **32 MB**
- 397.3ms: Table IV nuScenes Inference Time, §V-C p.925 原文

### ② 6064→397ms 正确归因: 同模型 ONNX 系统优化

**v1 错误**: §4 表内标"BEVDet large vs EMOS 模型架构不同";§6.2 又正确写"work-reduction",自相矛盾。

**PDF 核实**: §V-C 消融部分(p.928)原文: "applying **ONNX-based system optimization** decreases the end-to-end inference time **from 6064.2ms to 397.3ms**" — 明确是 EMOS 自身系统优化前后对比(**同一模型**)。

**正确解读**:
- 6064→397ms = EMOS 自身 multiscale pooling (1M→60K voxel, 减算量) + ONNX 计算图融合 (常数折叠+算子融合) 的综合效果
- 主体是 **work-reduction** (multiscale pooling 大幅减少 camera feature 计算量) + 图融合,不是 stage SM 调度并行
- 与我方"单 GPU pipeline 1.08× 已证伪"**不冲突** —— 6064→397 靠减算量,不靠 SM 并行;我方证伪的是 SM 时间复用

### ③ H-NEW1 场景自适应 engine 切换 + ISS-018 冲突 caveat

**v1 错误**: 将 H-NEW1 (runtime engine-switching) 标为 P2 低优先级 backlog,未说明与我方实验结论的冲突。

**修正**: H-NEW1 的前提是"困难帧需要更重的 engine"。但我方 **pathE 负结果 (ISS-018)** 显示:

> 在过参数化 PyramidFusion/DAIR 上,INT8 在全距离箱近免费(Δap70 -0.0046),剪枝代价在难样本上**不放大**。即"困难帧 ≠ 需要重 engine"在本数据集上不成立 → 场景自适应路由的 AP 收益 **为零**。

**结论**: H-NEW1 的适用条件是更难/欠参数化模型(如 V2X-ViT 上若 INT8 AP 崩塌),而非本阶段 Pyramid/DAIR。**属下一阶段换模型工作,非本阶段 Phase H**。

---

## 3. Chunk Prefetch vs 我方 E_pipeline — 为何非真冲突

**EMOS chunk prefetch** (§IV-B §Innov.3-2):
- 隐藏对象: **CPU→GPU PCIe H2D 传输延迟** (batch 数据搬运)
- 物理机制: CPU 异步 DMA 传输 chunk i+1 到 GPU 设备内存,同时 GPU 计算 chunk i
- 这是**真正异构操作**: CPU DMA controller ≠ GPU SM,不竞争

**我方 E_pipeline 证伪** (E_pipeline_singlegpu_4090.csv):
- 证伪对象: **GPU-only SM 时间复用** (stage-level pipeline parallelism,两帧不同 stage 共享同一 SM 池)
- 物理机制: memory-bound backbone 虽算力闲但仍占满 SM occupancy + 共享 L2/DRAM 控制器 → 没有可用的并发空间

**总结**:
| | EMOS chunk prefetch | 我方 E_pipeline |
|---|---|---|
| 并行的资源 | CPU DMA ∥ GPU SM | GPU stage0 SM ∥ GPU stage1 SM |
| 是否相交 | **不相交** (PCIe ≠ SM) | **相交** (共享同一 SM 池) |
| 效果 | 真并行,有效 | 时间复用,1.08× 极弱 |
| 冲突? | **不冲突** | — |

**注意**: Orin UMA 架构下 H2D≈0 (CPU/GPU 共享物理内存,指针传递),EMOS chunk prefetch 在 Orin 上的实际增益也会因此大幅降低。

---

## 4. MoE/Expert 路由对我方单模型 PyramidFusion 的迁移可行性评估 (更新 v2)

### 4.1 架构不兼容的核心原因 (不变)

EMOS 多子网架构(EPE 32MB / tri-expert 333MB)需要独立参数 + AMDB 提案生成 + runtime dispatch 逻辑。PyramidFusion 单路融合架构无法直接插入 MoE gate。

### 4.2 H-NEW1 (Runtime engine-switching) — 更新后的判断

**概念**: 基于 PyramidFusion head 置信度分数, runtime 选择 INT8+pruned / FP16+pruned / FP16+base 三档预编译 engine。

**v1 判断**: P2 低优先级 backlog

**v2 修正 (ISS-018 + ISS-022 冲突)**:

> **⚠ H-NEW1 前提与我方 pathE 负结果冲突**。我方实验显示:在 Pyramid/DAIR 上,INT8 在困难样本(远距离、高遮挡)的 AP 代价**不大于**简单样本 —— 即"困难场景需要更重 engine"这一前提在本数据集上**不成立**。场景路由带来的 AP 增益 ≈ 0。
>
> H-NEW1 的适用场景 = **欠参数化/更难模型**上有 AP trade-off 时(如 V2X-ViT INT8 AP 从 0.75→0.30 崩塌的情形)。
>
> **结论**: **H-NEW1 属下一阶段换模型(V2X-ViT)工作,非本阶段 Phase H**。当前 Pyramid/DAIR 无 AP 收益前提,不应投入工程量。

---

## 5. 对我方搜索空间的净影响总结

| 结论类型 | EMOS 维度 | 对我方影响 |
|---------|---------|-----------|
| **支持/验证** | work-reduction (sparse conv + multiscale pooling) 是主要加速手段 | ✅ 与我方"量化×剪枝是真杠杆"结论一致 |
| **支持/验证** | 单点 ν_h 系数跨平台外推 | 我方 M2 f-函数 R²>0.995 更精确,说明跨平台预测是可行维度 |
| **不适用** | MoE multi-expert 架构/路由 | 单模型架构不兼容;本阶段 Pyramid/DAIR AP 近常数,路由无收益 |
| **不冲突** | chunk prefetch (H2D 隐藏) | 与我方 E_pipeline 证伪机制不同(PCIe DMA ≠ SM 时间复用);不推翻我方 1.08× 结论 |
| **不适用** | LayerNorm fusion / complex node fusion | Pyramid backbone 无 LayerNorm (仅 BatchNorm,TRT 折叠);V2X-ViT 阶段再考虑 |
| **不适用** | Multiscale pooling (camera voxel 缩减) | 我方 PyramidFusion 无摄像头分支 |
| **Phase H 搜索空间不变** | 论文所有系统维度 | H1-H5 (2:4/workspace/batch/DLA路由/nvpmodel) 仍是正确主搜索轴 |

---

## 6. 相关工作引用定位

EMOS 在我方论文中的定位:

**Related Work §System-Level Optimization 章节** — 引用为"MoE-based adaptive inference 的代表":
- EMOS 解决算法层 MoE 多专家设计 + edge ONNX/TRT 系统优化
- 我方解决搜索空间层三维联合耦合陷阱 (B1×B2×D kernel-cliff, ISS-014)
- 差异化: EMOS 静态设计多 expert; 我方在**固定单模型**上做 **Pareto 搜索 + 耦合陷阱刻画**

**论文引用时注意区分两个数字**:
- "15.35×" = vs efficiency-oriented baselines on nuScenes [EMOS 论文, Jetson]
- "6064→397ms" = EMOS 自身 ONNX 系统优化 off→on 消融 [EMOS 论文, Jetson, nuScenes, 同模型]

---

*文档路径: `multi_agent/references/moe_paper_dim_review_v2.md`(2026-06-04 doc-curator 自 methods/design/ 迁入: 外部论文对照归 references)*
*交付给: supervisor (核验) → data-orchestrator + team-lead (知会)*
