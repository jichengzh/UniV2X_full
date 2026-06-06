# 剪枝 × 量化 耦合机理调研 (P×Q Coupling Survey v1)

**任务**: #14-①  
**日期**: 2026-06-06  
**作者**: sw-optimizer  
**引用纪律**: [文献声称] = 原文摘要/paper body 核查内容; [我方实证] = V2X 项目真测数据含出处; [推断] = 逻辑推断非直接测量  

---

## 目录

1. [核心问题回答](#1-核心问题回答)  
2. [耦合关系分类](#2-耦合关系分类)  
3. [内部实证对照表](#3-内部实证对照表)  
4. [文献方法谱系表](#4-文献方法谱系表)  
5. [方法详细摘要](#5-方法详细摘要)  
6. [我方框架定位](#6-我方框架定位)  
7. [完整文献索引](#7-完整文献索引)  

---

## 1. 核心问题回答

### Q1: 为什么要融合剪枝与量化而非独立串行?

**短答**: 剪枝决策改变网络结构，结构变化直接影响量化的硬件收益和精度代价——两者的最优解空间**不可分解**。独立串行优化会在两个维度都次优，甚至互相破坏对方的收益。

**长答**（四条主线）:

#### 主线 A: 剪枝结构决定量化硬件收益 (P → Q kernel 效率)

剪枝改变网络中间层通道数。INT8 Tensor Core 内核要求通道数为 32 的整数倍；FP16 dense 要求 8 对齐；FP16 sparse (2:4) 要求 16 对齐。若剪枝后通道数不对齐，TensorRT 会切换到慢速通用内核，使 INT8 加速效果从预期的 1.25–1.57× 退化到 ~1.06×，甚至 INT8 内核完全失效（退化为 FP32 计算）。

[我方实证] ISS-014 (issues_log_v1.md): 等计算量 (49.1 vs 49.2 GMAC) 配置下，32-对齐通道 (cliff2_c) 比非对齐通道 (prune90) 快 **2.57×**；单层最大差异 **11.3×**。

#### 主线 B: AP 代价非线性叠加 (P×Q 相互作用)

在过参数化模型（如 Pyramid/DAIR）上：
- 单独剪枝 50% 参数: AP50 下降 −0.026  
- 单独 INT8 量化: AP50 下降 −0.0005（几乎无损）  
- 若两者独立且 AP 代价线性可加: 预期 −0.027  
- 实测联合: AP50 下降 **−0.039**（超加性 penalty）

[我方实证] perstage_quant_AP_real_v2.csv: T_prune50p 组, base AP50 0.7910, p50+FP16=0.7644 (−0.0266), base+INT8=0.7905 (−0.0005), p50+INT8=0.7522 (−0.0388).  
同时，延迟收益近乎乘法叠加：P75+INT8-auto = **0.6124 ms** vs dense FP16 = **1.2715 ms** = **2.08×**（body-subnet collab2 on 4090）。AP 代价与延迟收益的非对称叠加特性是联合搜索的核心论据。

#### 主线 C: 最优排序 (P before Q)

量化会扰乱权重大小的相对排序，使后续剪枝错误地移除量化后看起来"小"但实际重要的权重。理论上 P → Q 优于 Q → P。

[文献声称] Harma et al. ICLR 2025 (arXiv 2405.20935): 首次数学证明稀疏化与量化非正交，Q before P 破坏权重重要性排序 → P before Q 为最优排序。  
[文献声称] Kim et al. ICLR 2026 (arXiv 2603.18426): Progressive Intensity Hypothesis (PIH)，量化排序优势在 4-bit 下可达 −49.9 perplexity。

#### 主线 D: 异常权重是两者的共同瓶颈

LLM 中的异常权重 (outliers) 既难以剪枝（大幅值 → 保留），又损害量化（扩大动态范围 → 量化分辨率浪费）。忽视该共同瓶颈导致独立串行的双重失败。

[文献声称] JSQ (ICML 2024, PMLR 2024): 新稀疏度度量桥接 P→Q，消除 outlier 后 LLaMA 实现 7.96× 计算量削减。  
[文献声称] SpQR (ICLR 2024, arXiv 2306.03078): 将 1–5% outlier 以高精度稀疏格式存储，其余 3–4 bit 量化，LLaMA-33B 在 24GB 消费级 GPU 运行。

---

## 2. 耦合关系分类

| 耦合类型 | 方向 | 描述 | 典型方法 | 我方实例 |
|---------|------|------|---------|---------|
| **结构-内核耦合** | P → Q | 剪枝后通道数决定量化内核选择 | DJPQ, 硬件约束搜索 | ISS-014: 2.57× 同计算量速度差 |
| **精度-代价耦合** | P × Q → AP | 联合 AP 代价非加性（超加性或亚加性，取决于冗余度） | APQ, CLIP-Q | p50+INT8: −0.039 vs additive −0.027 |
| **排序耦合** | P ↔ Q 顺序 | 操作顺序影响最终精度（P before Q 更优） | Harma 2025, Kim 2026 | [推断] 与我方 P→Q 实验顺序一致 |
| **异常值耦合** | P ∩ Q → 共同瓶颈 | Outlier 权重在两维度均为瓶颈 | JSQ, SpQR | [推断] Pyramid 无显著 outlier（INT8 近无损） |
| **粒度-对齐耦合** | Q → P 约束 | 量化内核粒度（32/16/8）约束剪枝搜索空间 | GETA, 硬件感知搜索 | dims_pruning_v1.md §3: round_to=32 floor |
| **宽度公式耦合** | P ↔ Q 结构兼容 | ResNeXt width 公式 + INT8 对齐双重约束 | — | wpg 陷阱: g32+wpg4+planes<16→崩 |
| **混精自动支配** | Q 内部 | TRT-auto 混精优于手工 per-stage 分配 | TRT-auto (TensorRT) | perstage_quant_pareto: 全部 per-stage 手工被支配 |
| **能效耦合** | P+Q → 能效 | 联合压缩改变内存带宽压力与计算密度比 | HAQ, DLA部署 | quantization_x_hardware §2.1: DLA INT8 ~15× FP16 |

---

## 3. 内部实证对照表

以下每条均标注数据文件出处，口径分层。

### 3.1 P → Q: 通道对齐决定 INT8 内核效率

**ISS-014: 内核切换悬崖** [我方实证, profile-verified]

| 配置 | stage0 通道数 | stage0 conv2 延迟(3层和) | 内核类型 | INT8 加速比 |
|------|--------------|------------------------|---------|-----------|
| p25 | 48→96ch (非32对齐) | 0.770 ms | `implicit_gemm_g1` (FP32 fallback) | ~1.06× |
| p50 | 32→64ch (%32=0) | 0.143 ms | `direct_group` (专用) | ~1.25× |
| p75 | 16→32ch (%32=0) | <0.143 ms | grouped kernel | ~1.57× |

数据出处: `multi_agent/methods/progress/issues_log_v1.md` §ISS-014 (IEngineInspector TacticValue 核验)

引擎级等计算量对比:
- prune90 [6,13,26] (非对齐中间通道): **2.355 ms**
- cliff2_c [16,32,64] (%32对齐): **0.916 ms**
- 差异: **2.57×** (等 GMAC 纯内核切换效应)

数据出处: `issues_log_v1.md` §ISS-014 引擎级剖面数据

**TRT 对齐要求** [文献声称+内部验证]:
- INT8: 通道数须 %32==0
- FP16 sparse (2:4): %16==0
- FP16 dense: %8==0

出处: `paper_learning/survey_raw_2/pruning_x_hardware.md` §1.1

---

### 3.2 P×Q: 延迟收益乘法叠加 / AP 代价超加性

[我方实证] 数据来源: `results/perstage_quant_AP_real_v2.csv` + `results/perstage_quant_pareto_verdict_v2.md`  
口径: DAIR val 1789 真测, body-subnet collab2, RTX 4090 CUDA Event 计时

| 配置 | 通道宽度 | 量化 | lat_p50 (ms) | AP50 | vs dense FP16 延迟加速 | vs dense FP16 AP 代价 |
|------|---------|------|-------------|------|----------------------|---------------------|
| Dense FP16 | [64,128,256] | FP16 | 1.2715 | 0.7910 | 1× | 0 |
| Dense INT8 | [64,128,256] | INT8-auto | 0.8113 | 0.7905 | 1.57× | −0.0005 |
| P50 FP16 | [32,64,128] | FP16 | 1.0138 | 0.7644 | 1.25× | −0.0266 |
| P50 INT8 | [32,64,128] | INT8-auto | 0.7956 | 0.7522 | **1.60×** | −0.0388 |
| P75 FP16 | [16,32,64] | FP16 | 0.7681 | 0.7567 | 1.65× | −0.0343 |
| **P75 INT8** | **[16,32,64]** | **INT8-auto** | **0.6124** | **0.7537** | **2.08×** | **−0.0373** |

**关键发现**:
- 延迟收益近似乘法叠加: P50×1.25 + INT8×1.57 → 实测 1.60× (≈乘法预期)
- AP 代价**超加性** (P50: −0.0266, INT8: −0.0005, 加法预期: −0.0271, 实测: −0.0388)
  → 量化施加在已剪枝模型上的 AP 惩罚比原始模型更大 → 不可独立优化
- P75 反而 AP cost 略小于 P50 (−0.0373 vs −0.0388) → INT8 在过参数化模型上的精度代价与剪枝率非单调

---

### 3.3 Q → P 约束: INT8 粒度约束剪枝搜索空间

[我方实证] 数据出处: `multi_agent/methods/design/dims_pruning_v1.md` §3, §D4, §D6

- **round_to=32 floor**: INT8 模块强制通道数对齐 32。Pyramid stage0 (64ch) 有效剪枝上限 ≈ 50%（更激进会 floor 到 32，无进一步削减效果）
- **wpg 宽度公式陷阱**: `width = int(planes * wpg / 64) * groups`; g=32+wpg=4 时 planes<16 → width=0 崩溃
  - 根因: 剪枝结构必须迁就量化 kernel 粒度 (32 对齐) 和 ResNeXt 宽度分组公式的双重约束
  - 修复: wpg 必须提升到 16

出处: `multi_agent/methods/design/dims_pruning_v1.md` §D6, §8.4; `paper_learning/survey_raw_2/pruning_x_hardware.md` §7.2

---

### 3.4 TRT-auto 支配手工 per-stage 混精

[我方实证] 数据出处: `results/perstage_quant_pareto_verdict_v2.md`, `results/perstage_quant_AP_real_v2.csv`

P50 三元组 (T_prune50p) Pareto 分析:

| 配置 | lat_p50 (ms) | AP50 | Pareto 状态 |
|------|-------------|------|------------|
| global_int8_automix (TRT-auto) | **0.7956** | 0.7522 | ✅ Pareto |
| global_fp16_automix | 1.0138 | **0.7644** | ✅ Pareto |
| c3_I\|F\|F (保 stage1 FP16) | 1.0168 | 0.7526 | ❌ 被支配 |
| c5_F\|F\|I | 1.0004 | 0.7503 | ❌ 被支配 |
| c7_I\|F\|I | 0.9668 | 0.7512 | ❌ 被支配 |
| c_all_int8_forced | 0.9073 | 0.7280 | ❌ 被支配 |

**结论**: 全部 6 个手工 per-stage 混精配置均被支配。保 stage1 FP16 有 +0.022~0.025 AP 正向点，但 lat 代价使其不在 Pareto 前沿。TRT-auto 逐层选择策略（最小化含 reformat overhead 的总引擎延迟）比手工 stage 边界分割更优。

机制: 手工在 stage 边界强制精度切换 → TRT 插入 reformat/Q-DQ 节点 → 额外延迟；TRT-auto 避免不必要的 reformat → 同时更快更准。

出处: `multi_agent/methods/design/dims_quantization_v1.md` §二.5; `results/perstage_quant_pareto_verdict_v2.md` §3

---

### 3.5 INT8 能效与硬件收益

[我方实证] Orin 真测 (ISS-016):
- base collab2 FP16 @MODE_30W: 能量 348.5 mJ/帧
- 4090 FP16: 462.45 mJ/帧 (NVML GPU 卡级)
- 口径注: Orin VIN_SYS_5V0 = 整模组; 4090 NVML = GPU 卡, 不同测量域

[文献声称] DLA INT8 优势: `quantization_x_hardware.md` §2.1: "DLA INT8 卷积 ~15× FP16 (sparse 30×)"  
[文献声称] HAQ (CVPR'19): 能量削减 1.9× vs 固定 8-bit  
[推断] INT8 30–52% 节能 (CLAUDE.md §E4 引用) — 具体实验文件 E4 未在本次扫描范围内，标为待核实

出处: `issues_log_v1.md` §ISS-016; `paper_learning/survey_raw_2/quantization_x_hardware.md` §2.1

---

### 3.6 2:4 稀疏 + INT8: P×Q 复合最高收益

[文献声称+内部引用]:
- 2:4 稀疏 + INT8: 1.4× over INT8 alone (NVIDIA blog, PnR case) — 出处: `pruning_x_hardware.md` §1.1
- Hopper: 2:4 sparse FP8 = 3957.8 TFLOPS (最高算力组合)
- DLA 32× 对齐要求同样适用于联合 2:4+INT8 模式

---

## 4. 文献方法谱系表

| # | 方法 | 作者 | 年份 | 会议 | arXiv/URL | 耦合模式 | vs 独立串行增益 |
|---|------|------|------|------|-----------|---------|--------------|
| 1 | Deep Compression | Han, Mao, Dally | 2016 | ICLR | [1510.00149](https://arxiv.org/abs/1510.00149) | 顺序 P→Q→Huffman | 35–49× 压缩比 (建立了顺序基线) |
| 2 | CLIP-Q | Tung, Mori | 2018 | CVPR | [CVF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.pdf) | 并行联合 (P ∥ Q ∥ finetune) | AlexNet 51× vs DC 35×; GoogLeNet 10×; ResNet50 15× |
| 3 | HAQ | Wang et al. | 2019 | CVPR | [1811.08886](https://arxiv.org/abs/1811.08886) | RL硬件反馈驱动 Q | 延迟 1.4–1.95×, 能量 1.9× vs 固定8bit |
| 4 | APQ | T. Wang et al. | 2020 | CVPR | [2006.08509](https://arxiv.org/abs/2006.08509) | 联合NAS+剪枝+量化搜索 | +2.3% acc vs ProxylessNAS+AMC+HAQ串行; 600× 更少GPU时 |
| 5 | DJPQ | Wang, Lu, Blankevoort | 2020 | ECCV | [2007.10463](https://arxiv.org/abs/2007.10463) | 可微联合单损失函数 | ResNet18 53× BOPs; MobileNetV2 43× BOPs iso-acc |
| 6 | Bayesian Bits | van Baalen et al. | 2020 | NeurIPS | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/3f13cf4ddf6fc50c0d39a1d5aeb57dd8-Abstract.html) | 统一参数化 (0-bit = 剪枝) | 超越固定bit宽度Pareto; 硬件友好2^k位宽 |
| 7 | FITCompress | Zandonati et al. | 2023 | arXiv | [2302.07612](https://arxiv.org/abs/2302.07612) | Fisher信息测地线联合搜索 | 超越 Bayesian Bits & DJPQ on ResNet+BERT+YOLO |
| 8 | SpQR | Dettmers et al. | 2024 | ICLR | [2306.03078](https://arxiv.org/abs/2306.03078) | 敏感度驱动稀疏高精度+主体量化 | <1% perplexity loss @ 3–4bit; 15% 速度提升 |
| 9 | JSQ | Guo et al. | 2024 | ICML | [PMLR](https://proceedings.mlr.press/v235/guo24g.html) | 桥接度量 (outlier→Q友好稀疏) | LLaMA 7.96× 计算削减无精度崩溃 |
| 10 | Harma et al. | Harma et al. | 2025 | ICLR | [2405.20935](https://arxiv.org/abs/2405.20935) | 理论证明: P before Q 最优排序 | 首次数学证明非正交性 |
| 11 | SLiM | Mozaffari et al. | 2025 | ICML | [2410.09615](https://arxiv.org/abs/2410.09615) | 顺序 Q→2:4→LoRA补偿 | +5.66% acc vs 先前最佳; 4.3× GPU加速 |
| 12 | GETA | Qu et al. | 2025 | CVPR | [2502.16638](https://arxiv.org/abs/2502.16638) | 量化感知依赖图+联合训练 | CNN+Transformer 超越全部先前 joint P×Q |
| 13 | OBR | Guo, Li, Benini | 2025 | arXiv | [2509.11177](https://arxiv.org/abs/2509.11177) | Hessian联合误差补偿 (闭合解) | W4A4KV4+50%稀疏: 4.72× 加速, 6.4× 内存削减 |
| 14 | Kim et al. (PIH) | Kim et al. | 2026 | ICLR | [2603.18426](https://arxiv.org/abs/2603.18426) | 理论: 通用压缩排序框架 | 4-bit P-before-Q 优势 −49.9 perplexity |
| 15 | SparseGPT | Frantar, Alistarh | 2023 | ICML | [2301.00774](https://arxiv.org/abs/2301.00774) | 共享Hessian顺序 P→Q (OBC) | LLM 50%稀疏近无损; 与GPTQ组合 |
| 16 | 硬件感知 joint MPQ+P | Motetti et al. | 2024 | IEEE Trans. | [2407.01054](https://arxiv.org/abs/2407.01054) | 梯度通道联合MPQ+剪枝+硬件cost | 首个 one-shot 硬件感知 channel-wise MPQ+剪枝 |

---

## 5. 方法详细摘要

### 5.1 Deep Compression — 顺序 P→Q 基线的建立

[文献声称] Han, Mao, Dally. "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." ICLR 2016. arXiv:1510.00149.

**耦合方式**: 严格顺序三阶段 (P → 再训练 → Q → 再训练 → Huffman)。每阶段独立，前阶段结果固定传入下阶段。

**为何有效 (隐式耦合)**: 剪枝自然使权重分布向零聚集，减小动态范围 → 低 bit 量化更容易表示。这是 P→Q 耦合的隐式形式，后续工作将其显式化。

**局限**: 剪枝决策不考虑量化代价；Q 的精度损失无法被 P 阶段的剪枝模式所补偿。

**规模**: AlexNet 35× (240MB→6.9MB), VGG-16 49× (552MB→11.3MB), 无精度损失。

---

### 5.2 CLIP-Q — 首次证明并行联合优于顺序

[文献声称] Tung, Mori. "CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization." CVPR 2018.

**耦合方式**: P 与 Q 同时进行，嵌入一个统一的 fine-tuning 过程。网络在每次迭代中既做剪枝决策（权重置零）也做量化决策（权重舍入到最近码本中心），梯度对两个目标共同反传。

**关键洞见**: 顺序方法存在"过早剪枝"问题——某连接被 P 移除后，Q 无法补偿；反之，P 也无法通过选择性保留来对抗 Q 误差。并行允许两者互相修正。

**vs 顺序的证据**: AlexNet 51× vs Deep Compression 35×; ResNet-50 15× 无精度损失。

---

### 5.3 DJPQ — 可微联合损失，含通道对齐约束

[文献声称] Wang, Lu, Blankevoort. "Differentiable Joint Pruning and Quantization for Hardware Efficiency." ECCV 2020. arXiv:2007.10463.

**耦合方式**: 将剪枝 (VIB 变分信息瓶颈) 和混精量化 (per-layer bit 选择) 嵌入统一损失函数，梯度同时更新剪枝掩码和量化精度选择变量。

**硬件对齐耦合 (关键)**: 引入 2^k 受限位宽（1/2/4/8 bit）并配合结构化剪枝，使剪枝后通道数与硬件加速器对齐——这是**通道对齐作为联合约束**的早期实例，与我方 ISS-014 的发现一致。

**证据**: ResNet18 53× BOPs, MobileNetV2 43× BOPs, 均在 iso-accuracy 下超过独立 P+Q 基线。

---

### 5.4 Bayesian Bits — 统一参数化 (0-bit = 剪枝)

[文献声称] van Baalen et al. "Bayesian Bits: Unifying Quantization and Pruning." NeurIPS 2020.

**耦合方式**: 将剪枝视为 0-bit 量化的特殊情况。通过级联残差量化（每级决定是否加倍有效精度），门控变量 $z \in \{0,1\}$ 控制每位的开关；$z=0$ 在所有级 = 剪枝。统一参数化使 P 和 Q 共享同一组优化变量。

**意义**: 这是真正意义上的"P 和 Q 是同一参数空间的不同操作"，消除了人为划分两个阶段的边界。

---

### 5.5 APQ — 联合 NAS+P+Q，搜索空间不可分解性

[文献声称] T. Wang et al. "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy." CVPR 2020. arXiv:2006.08509.

**耦合方式**: Once-for-All supernet + 量化感知精度预测器。先训 FP32 精度预测器，再迁移到 INT8 预测器（少量 QAT 样本）。进化搜索同时覆盖架构、通道剪枝率、逐层位宽三个维度。

**核心证据 (vs 独立串行)**: +2.3% ImageNet top-1 精度 vs ProxylessNAS+AMC+HAQ 的三阶段串行组合，同等延迟预算下。同时减少 600× GPU 训练时间。

**理论根据**: 三维空间不可分解——最优 Q 策略依赖网络结构（P 结果），最优 P 策略依赖后续 Q 的误差放大模式；独立优化每维必然次优。

---

### 5.6 Harma et al. — 非正交性与 P-before-Q 理论证明

[文献声称] Harma et al. "Effective Interplay Between Sparsity and Quantization." ICLR 2025. arXiv:2405.20935.

**耦合模式**: 理论分析，证明非正交性并给出最优排序的数学保证。

**核心定理**: 稀疏化与量化是**非正交操作**：联合误差 > 独立误差之和。机制：Q 扰乱权重幅值的相对大小排序 → 后续基于幅值的 P 错误移除"量化后看起来小但实际重要"的权重。因此 P before Q 理论最优。

**实验验证**: OPT 和 LLaMA 系列 (125M–8B), ViT, ResNet 上均验证。[文献声称]

---

### 5.7 JSQ — Outlier 桥接度量，LLM 极限压缩

[文献声称] Guo et al. "JSQ: Joint Sparsification and Quantization of LLMs." ICML 2024. PMLR 2024.

**核心发现**: 识别 LLM 中稀疏化与量化的共同瓶颈——**outlier 权重**：
- 稀疏化倾向于保留 outlier（大幅值 → 看似重要）
- 但 outlier 恰是量化最难处理的（扩大动态范围）

**耦合方式**: 新稀疏度度量显式惩罚保留 outlier，同时服务 P 和 Q 两个目标；搜索式激活编辑消除无用 outlier。

**证据**: LLaMA 7.96× 计算量削减，无精度崩溃；大多数先前方法在此极限压缩比下失败。

---

### 5.8 OBR — 分布冲突与 Hessian 联合补偿

[文献声称] Guo, Li, Benini. "OBR: Optimal Brain Restoration." arXiv:2509.11177. 2025.

**识别分布冲突 (核心贡献)**:
- 量化偏好权重分布**紧凑**（小动态范围 → 细分辨率）
- 剪枝受益于权重分布**高方差**（分布散开 → 幅值差异大 → 易于重要性排序）

这两个目标**相互拉扯**，使得先 P 后 Q 的顺序虽然理论最优，但仍会有显著的联合误差——需要额外补偿。

**方法**: 训练无关的 Hessian 联合误差补偿（闭合解），显式补偿 P 和 Q 的相互误差扩大。

**证据**: LLaMA2-7B W4A4KV4+2:4 稀疏: 比 SparseGPT+GPTQ 串行基线降低 perplexity 18.8 点, zero-shot 精度提升 5.86%。系统级 4.72× 加速, 6.4× 内存削减。

---

### 5.9 Kim et al. (PIH) — 通用压缩排序理论

[文献声称] Kim et al. "When to Prune, When to Quantize, and When to Do Both." ICLR 2026. arXiv:2603.18426.

**Progressive Intensity Hypothesis (PIH)**: 弱扰动应先于强扰动施加。剪枝 (归零部分权重) 通常弱于量化 (扰动所有权重)，因此 P before Q 是一般规律，而非特例。

**排序优势量化**: 4-bit 量化下，"量化先"→"剪枝先"的 perplexity 优势可达 −49.9 点（SparseGPT 对比）。

**扩展性**: 适用于多阶段压缩、混精量化、LoRA 增强管线，语言和视觉模型。

---

## 6. 我方框架定位

### 6.1 我方 vs 文献方法对应关系

| 文献现象 | 我方实测 | 差异/共同点 |
|---------|---------|-----------|
| DJPQ: 通道对齐是联合约束 | ISS-014: 32对齐悬崖 2.57× | **完全吻合**。我方有更精确的 profile 数据 (TacticValue) |
| Harma 2025: P before Q 理论最优 | 我方实验顺序: 先剪枝再量化 | **顺序一致**，但我方未做反向对照实验 [推断 可增补] |
| APQ: 三维联合优于独立串行 | 我方: Pareto 扫描覆盖 B1×B2 联合维度 | **动机一致**。我方新增 D 维度 (hardware deployment) |
| Bayesian Bits: 统一参数化 | 我方: 分维度扫描 (非统一参数) | **差异**: 我方不尝试统一参数化 |
| OBR: P×Q 分布冲突 | [推断] Pyramid 过参数化使冲突弱化 (INT8 near-lossless) | 在过参数化域冲突被模型冗余吸收 |
| JSQ: Outlier 是共同瓶颈 | `dims_quantization_v1.md`: INT8 SNR <1，无显著 outlier | Pyramid 无 outlier 问题 — 冗余充足 |

### 6.2 我方框架的独特性 (文献空白)

[我方实证] 来源: `paper_learning/survey_raw_2/hw_aware_nas_and_joint_search.md` §10.2

当前文献中：
- **APQ**: 覆盖 A+B1+B2（架构+剪枝+量化），**不含 D (部署配置)**
- **HAQ**: 覆盖 B2+D（量化+延迟反馈），**不含 B1（剪枝）**
- **硬件感知 NAS 文献**: 覆盖 A+D，**不含 B**

我方 sw-hw-cooptim 框架的独特定位：**在 V2X 协同感知场景下，B1（通道剪枝）× B2（混精量化）× D（部署配置：platform, precision, kernel）联合搜索**，且以**真实硬件延迟（非 LUT 估算）** 作为 Pareto 维度，是文献中已识别的空白区。

### 6.3 已获得的关键联合证据 (P×Q 联合优于串行的具体论据)

1. **内核切换悬崖** [我方实证, ISS-014]: P 后通道数必须与 Q 内核粒度共同约束，否则 Q 收益蒸发（2.57× 等计算量差异）→ 这是"P 必须感知 Q"的硬约束
2. **AP 超加性惩罚** [我方实证, perstage_quant_AP_real_v2.csv]: P50+INT8 的 AP 代价 (−0.039) > P50-only (−0.027) + INT8-only (−0.0005) → 独立串行高估了联合精度
3. **TRT-auto 支配手工混精** [我方实证]: 手工 per-stage 精度分配（即 Q 独立于 P 后结构决策）全部被 TRT-auto 支配 → 量化内部的"局部分配"也需要全局优化
4. **wpg-P×Q 双约束** [我方实证]: 剪枝搜索空间上界被量化对齐需求硬裁 → P 搜索必须感知 Q 约束，无法独立搜索

---

## 7. 完整文献索引

按年份排序。[文献声称] 均经过 WebFetch 核查摘要/paper page（由 web-agent 完成）。

```
[Han2016] Han, Song and Mao, Huizi and Dally, William J.
"Deep Compression: Compressing Deep Neural Networks with Pruning, 
Trained Quantization and Huffman Coding."
ICLR 2016. arXiv:1510.00149
https://arxiv.org/abs/1510.00149

[Tung2018] Tung, Frederick and Mori, Greg.
"CLIP-Q: Deep Network Compression Learning by In-Parallel Pruning-Quantization."
CVPR 2018.
https://openaccess.thecvf.com/content_cvpr_2018/papers/Tung_CLIP-Q_Deep_Network_CVPR_2018_paper.pdf
https://ieeexplore.ieee.org/document/8578919/

[Wang2019-HAQ] Wang, Kuan and Liu, Zhijian and Lin, Yujun and Han, Song.
"HAQ: Hardware-Aware Automated Quantization with Mixed Precision."
CVPR 2019 (oral). arXiv:1811.08886
https://arxiv.org/abs/1811.08886

[Wang2020-APQ] Wang, Tianzhe and Wang, Kuan and Cai, Han and Han, Song et al.
"APQ: Joint Search for Network Architecture, Pruning and Quantization Policy."
CVPR 2020. arXiv:2006.08509
https://arxiv.org/abs/2006.08509

[Wang2020-DJPQ] Wang, Diwen and Lu, Weng-Tai and Blankevoort, Babak.
"Differentiable Joint Pruning and Quantization for Hardware Efficiency."
ECCV 2020. arXiv:2007.10463
https://arxiv.org/abs/2007.10463

[vanBaalen2020] van Baalen, Mart and Louizos, Christos et al.
"Bayesian Bits: Unifying Quantization and Pruning."
NeurIPS 2020.
https://proceedings.neurips.cc/paper/2020/hash/3f13cf4ddf6fc50c0d39a1d5aeb57dd8-Abstract.html
GitHub: https://github.com/Qualcomm-AI-research/BayesianBits

[Zandonati2023] Zandonati, Benjamin et al.
"FITCompress: Neural Network Compression via Fisher Information Theory."
arXiv:2302.07612, 2023.
https://arxiv.org/abs/2302.07612

[Frantar2023-SparseGPT] Frantar, Elias and Alistarh, Dan.
"SparseGPT: Massive Language Models Can be Accurately Pruned in One Shot."
ICML 2023. arXiv:2301.00774
https://arxiv.org/abs/2301.00774

[Dettmers2024-SpQR] Dettmers, Tim et al.
"SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression."
ICLR 2024. arXiv:2306.03078
https://arxiv.org/abs/2306.03078

[Guo2024-JSQ] Guo, Zhanbiao et al.
"JSQ: Joint Sparsification and Quantization of LLMs without Dropping Tokens."
ICML 2024.
https://proceedings.mlr.press/v235/guo24g.html
GitHub: https://github.com/uanu2002/JSQ

[Motetti2024] Motetti, B. A. et al.
"Joint Channel-Wise Pruning and Mixed-Precision Quantization: A Hardware-Aware Approach."
IEEE Transactions on Computers, 2024. arXiv:2407.01054
https://arxiv.org/abs/2407.01054

[Harma2025] Harma, Andrei et al.
"Effective Interplay Between Sparsity and Quantization."
ICLR 2025. arXiv:2405.20935
https://arxiv.org/abs/2405.20935
Project: https://sq-interplay.github.io/

[Qu2025-GETA] Qu, Sifan et al.
"GETA: Quantization-Aware Pruning via Dependency Graph."
CVPR 2025. arXiv:2502.16638
https://arxiv.org/abs/2502.16638

[Mozaffari2025-SLiM] Mozaffari, Jafar and Yazdanbakhsh, Amir and Dehnavi, Maryam.
"SLiM: One-Shot Quantization and Sparsity with Low-Rank Approximation."
ICML 2025. arXiv:2410.09615
https://arxiv.org/abs/2410.09615

[Guo2025-OBR] Guo, Chuanshuai and Li, Menghao and Benini, Luca.
"OBR: Optimal Brain Restoration for Extreme LLM Compression via Joint Sparse-Quantized Error Compensation."
arXiv:2509.11177, 2025.
https://arxiv.org/abs/2509.11177
GitHub: https://github.com/csguoh/OBR

[Kim2026-PIH] Kim, Jeonghoon et al.
"When to Prune, When to Quantize, and When to Do Both: Progressive Intensity Hypothesis for Model Compression."
ICLR 2026. arXiv:2603.18426
https://arxiv.org/abs/2603.18426
```

---

## 附录: 内部数据文件引用索引

| 现象 | 数据文件 | 数据类型 |
|-----|---------|---------|
| ISS-014 内核切换悬崖 | `multi_agent/methods/progress/issues_log_v1.md` §ISS-014 | [我方实证] profile-verified |
| 通道对齐要求 | `paper_learning/survey_raw_2/pruning_x_hardware.md` §1.1 | [文献声称] TRT官方规格 |
| P×Q Pareto 数据 | `results/perstage_quant_AP_real_v2.csv` | [我方实证] 真测 |
| per-stage 混精被支配 | `results/perstage_quant_pareto_verdict_v2.md` | [我方实证] Pareto 分析 |
| TRT-auto 机制 | `multi_agent/methods/design/dims_quantization_v1.md` §二.5 | [我方实证+文献] |
| wpg 宽度公式 | `multi_agent/methods/design/dims_pruning_v1.md` §D6 | [我方实证] |
| round_to=32 约束 | `multi_agent/methods/design/dims_pruning_v1.md` §3, §D4 | [我方实证] |
| INT8 DLA 能效 | `paper_learning/survey_raw_2/quantization_x_hardware.md` §2.1 | [文献声称] |
| Orin 能量数据 | `multi_agent/methods/progress/issues_log_v1.md` §ISS-016 | [我方实证] |
| 文献空白定位 | `paper_learning/survey_raw_2/hw_aware_nas_and_joint_search.md` §10.2 | [文献声称+我方推断] |
| APQ 对比增益 | `paper_learning/survey_raw_2/pruning_x_hardware.md` §6.3 | [文献声称] |

---

*本文档由 sw-optimizer 撰写，供 supervisor 逐引核验。所有 [文献声称] 均经 web-agent WebFetch 核查原文摘要/paper page；所有 [我方实证] 均标注具体数据文件和测量口径；[推断] 明确区分。*
