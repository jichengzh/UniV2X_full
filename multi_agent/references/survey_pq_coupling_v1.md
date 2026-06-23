# 剪枝 × 量化 耦合机理调研 (P×Q Coupling Survey v1.3)

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
6. [**融合加速的实现机制分类学** ★v1.3新增](#6-融合加速的实现机制分类学)  
7. [我方框架定位](#7-我方框架定位)  
8. [完整文献索引](#8-完整文献索引)  

---

## 1. 核心问题回答

### Q1: 为什么要融合剪枝与量化而非独立串行?

**短答**: 剪枝决策改变网络结构，结构变化直接影响量化的硬件收益和精度代价——两者的最优解空间**不可分解**。独立串行优化会在两个维度都次优，甚至互相破坏对方的收益。

### Q1-0. 多策略融合的优势汇总（逐条带引用） [v1.2 增补]

> ★[2026-06-06 team-lead 增补] 本节为集中回答"为什么做多策略融合、优势是什么"，**全部数字来自本文档其他节已核验的引用**（文献编号见 §8 索引；我方数字见附录数据文件索引），无新声称。

**为什么必须融合 — 因果链三句话**:
1. 单一策略有收益天花板: 剪枝只减计算量、量化只降精度位宽，各自收益上限受模型冗余形态限制（[我方实证] Pyramid 剪枝单独最高 1.65×、INT8 单独 1.57×，见 §3.2 表）。
2. 两者收益**近似乘法叠加**而 AP 代价**不按比例叠加**（过参数化域），联合空间存在单策略到不了的 Pareto 区域（[我方实证] P75+INT8 = 2.08×@4090 / 6.56×@Orin-FP32基线，AP 代价仅 −0.037，§3.2）。
3. 但耦合是双刃: 不联合优化时，剪枝会**摧毁**量化的硬件收益（[我方实证] ISS-014 内核悬崖，等计算量 2.57× 差异，§3.1）——所以"融合"不仅是为了叠加收益，更是为了**避免互相破坏**。

**优势量化对比表（联合 vs 独立串行，全部带引用）**:

| 优势维度 | 证据（联合 vs 串行/单策略） | 来源 |
|---------|---------------------------|------|
| 压缩率 | CLIP-Q 并行联合 AlexNet **51×** vs Deep Compression 顺序串行 **35×** | [文献声称] [Tung2018]; [Han2016] |
| 精度（同延迟预算） | APQ 联合搜索 **+2.3% ImageNet top-1** vs ProxylessNAS+AMC+HAQ 三阶段串行 | [文献声称] [Wang2020-APQ] |
| 搜索成本 | APQ 联合 **600× 更少 GPU 时** vs 三阶段串行 | [文献声称] [Wang2020-APQ] |
| 计算量（iso-acc） | DJPQ 可微联合 ResNet18 **53× BOPs 削减**，超独立 P+Q 基线 | [文献声称] [Wang2020-DJPQ] |
| LLM 极限压缩 | OBR 联合误差补偿(W4A4KV4+50%稀疏): 系统级 **4.72× 加速, 6.4× 内存削减**(摘要已证); vs SparseGPT+GPTQ 串行的 ppl/zero-shot 增益见 §5.8【待核:正文】 | [文献声称] [Guo2025-OBR] |
| 延迟（我方） | P75+INT8 联合 **2.08×**@4090（vs 单 INT8 1.57× / 单 P75 1.65×）; Orin 上 vs FP32 基线 **6.56×** | [我方实证] §3.2; E7 |
| 能耗（我方） | INT8 叠加在剪枝模型上仍省 **29.6%** J/frame（279.78→196.88 mJ; base 上省 51.6%） | [我方实证] §3.5; E4_energy_4090.csv T_prune75 fp16/int8 行(cudagraph 源, batch=2, supervisor 复算确认) |
| 避免收益蒸发 | 联合感知对齐约束可避免内核悬崖（不联合: 等计算量下慢 **2.57×**，INT8 增益从 1.57× 跌到 ~1.06×） | [我方实证] §3.1; ISS-014 |

**优势的边界（诚实声明）**: 联合收益并非处处乘法——P50+INT8 实测 1.60× **低于**乘法预期 1.96×（次乘法，§3.2），且 AP 代价超加性（−0.039 > −0.027，§3.2）。这恰恰说明：**联合空间必须搜索而非按独立收益外推**——这正是做联合优化框架（而非手工串行调参）的根本理由。

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
[文献声称] Kim et al. ICLR 2026 (arXiv 2603.18426): Progressive Intensity Hypothesis (PIH)，量化排序优势在 4-bit 下可达 −49.9 perplexity [待核:正文]。

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

| 配置 | stage0 通道数 | stage0 conv2 延迟(3层和) | 内核类型 | INT8 加速比 (同级 FP16→INT8) |
|------|--------------|------------------------|---------|-----------|
| base (dense) | 64→128→256ch (%32=0) | — | `direct_group` | **1.57×** |
| p25 | 48→96ch (非32对齐) | 0.770 ms | `implicit_gemm_g1` (FP32 fallback) | ~1.06× |
| p50 | 32→64ch (%32=0) | 0.143 ms | `direct_group` (专用) | **1.27×** |
| p75 | 16→32ch (%32=0) | <0.143 ms **[推断]** | grouped kernel | **1.25×** |

**趋势**: INT8 加速比随剪枝深度单调下降 (base 1.57× → p50 1.27× → p75 1.25×)，即**剪枝削弱了 INT8 的独立收益**。这是 P×Q 次乘法交互的结构性根因，而非独立叠加。

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
- 延迟收益**次乘法叠加(P50) / 近似乘法叠加(P75)** — INT8 增益随剪枝级衰减是核心耦合证据:
  - **P50+INT8 = 1.60×** < 乘法预期 **1.96×** (= P50剪枝加速 1.25× × base INT8加速 1.57×): **次乘法(sub-multiplicative)**
    根因: INT8 在 P50 剪枝模型上的独立加速**从 1.57× 衰减至 1.27×** (P50-FP16→P50-INT8: 1.0138→0.7956ms); 若以衰减后的 INT8 增益 1.27× 计: 1.25×1.27≈1.59≈1.60 — 但这恰恰说明 INT8 的边际收益被剪枝结构变化所抑制，是不可加性的直接证据
  - **P75+INT8 = 2.08×** ≈ 乘法预期 **2.07×** (= P75剪枝加速 1.655× × P75-FP16→INT8 增益 1.254×): **近似乘法** [P75 conv2 profile 数据不足→1.25× 为 **[推断]**]
  - 延迟联合收益对比 — **4090 与 Orin 口径必须分开标注**:
    - **RTX 4090** (body-subnet collab2): dense-FP16 1.2715ms → P75+INT8 0.6124ms = **2.08×** [我方实证, `results/perstage_quant_AP_real_v2.csv`, CUDA Event p50, FP16基线对比]
    - **Orin MODE_30W 612MHz** (body-subnet collab2): FP32-base 131.33ms → P75+INT8 20.02ms = **6.56×** [我方实证, `results/E7_orin_e2e_baseline_vs_best.csv`, trtexec loadEngine GPU_Compute_median warmup=200/runs=200; 注: 此为 FP32→INT8+P75 跨精度基线对比]
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

[我方实证] RTX 4090 INT8 能效实测 (`results/E4_energy_4090.csv`, NVML GPU 卡级, B=1):
- T1_base FP16: **291.16 mJ/帧** → T1_base INT8: **141.02 mJ/帧** = **−51.6%** 能量削减
- (T1_base_fp16.engine n=218 runs; T1_base_int8.engine n=216 runs; doe6 源; idle_power=28.89W 已剔除)
- **INT8 叠加在剪枝模型上仍省能** [v1.2 增补, supervisor 复算确认]: T_prune75 FP16 **279.78 mJ/帧** → INT8 **196.88 mJ/帧** = **−29.6%**(E4 csv T_prune75 fp16/int8 行, cudagraph 源, batch=2)— 即 P×Q 联合下能耗收益保留(虽较 base 上的 −51.6% 衰减, 与延迟侧"INT8 增益随剪枝衰减"方向一致且**非独立证据**: 能耗≈功率×延迟, 延迟侧衰减直接传导至能耗侧)

[文献声称] DLA INT8 优势: `quantization_x_hardware.md` §2.1: "DLA INT8 卷积 ~15× FP16 (sparse 30×)"  
⚠️ **与我方实测冲突**: ISS-007 记录 0/12 INT8 build 失败 (kDIRECT_IO + bank 超限, **非算子不兼容** — FP16 同模型 8/12 可 build 恰证明算子兼容); **DLA INT8 ~15× 优势在我方模型上不可达**。[文献声称] 优势适用于满足 DLA 资源/IO 约束的网络。

[文献声称] HAQ (CVPR'19): 能量削减 1.9× vs 固定 8-bit

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
| 14 | Kim et al. (PIH) | Kim et al. | 2026 | ICLR | [2603.18426](https://arxiv.org/abs/2603.18426) | 理论: 通用压缩排序框架 | 4-bit P-before-Q 优势 −49.9 perplexity [待核:正文] |
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

**证据**: LLaMA2-7B W4A4KV4+2:4 稀疏: 比 SparseGPT+GPTQ 串行基线降低 perplexity 18.8 点, zero-shot 精度提升 5.86%【待核:正文 — 摘要无此两数, 摘要已证为系统级 4.72× 加速 + 6.4× 内存削减】。

---

### 5.9 Kim et al. (PIH) — 通用压缩排序理论

[文献声称] Kim et al. "Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression." ICLR 2026. arXiv:2603.18426.

**Progressive Intensity Hypothesis (PIH)**: 弱扰动应先于强扰动施加。剪枝 (归零部分权重) 通常弱于量化 (扰动所有权重)，因此 P before Q 是一般规律，而非特例。

**排序优势量化**: 4-bit 量化下，"量化先"→"剪枝先"的 perplexity 优势可达 −49.9 点（SparseGPT 对比）[待核:正文]。

**扩展性**: 适用于多阶段压缩、混精量化、LoRA 增强管线，语言和视觉模型。

---

## 6. 融合加速的实现机制分类学

> **v1.3 新增章节** — 回答"具体怎么把 P 和 Q 融合起来实现加速"，从实现机制而非耦合原理角度分类。  
> **引用纪律同前**: [文献声称] = 原文/WebFetch核查; [我方实证] = 真测含文件出处; [待核:正文] = 摘要/HTML未见、正文待确认; [推断] = 逻辑推断。

---

### 6.1 顺序流水实现 (Sequential Pipeline)

**代表方法**: Deep Compression (Han et al. ICLR 2016), SparseGPT + GPTQ

#### Deep Compression 三阶段管线 [文献声称]

三个独立阶段严格顺序执行，每段均有 finetune：

**阶段①: 剪枝 (Pruning)**
- 按权重幅值排序, 低于阈值的连接置零 (unstructured sparsity)
- 剪后再训练 (retrain with L2 regularization on remaining connections)
- 压缩效果: AlexNet 9×, VGG-16 13× 参数削减 [文献声称, arXiv:1510.00149 摘要]

**阶段②: 量化 (Trained Quantization)**
- 对每层非零权重做 **K-means 聚类**, 得到 k 个中心点 codebook（每连接 2–8 bit）[待核:正文—k选择标准]
- 前向: 每权重用最近中心点代替
- 反向: 梯度**累积**到其所属的 centroid（非个别权重），更新 centroid 值

**阶段③: Huffman 编码**
- 对 codebook 索引做 Huffman 编码进一步压缩存储（高频 index 用短码）
- 最终压缩比: AlexNet 35× (240MB→6.9MB), VGG-16 49× (552MB→11.3MB) [文献声称]

**顺序耦合的隐式形式**: 剪枝后权重向零聚集 → 量化动态范围缩小 → 低 bit codebook 更精确表示 → 这是 P→Q 的隐式依赖，后续方法将其**显式化**。

---

#### SparseGPT + GPTQ 共享 Hessian 复用 [文献声称]

**核心创新: 同一层的 Hessian H 可被 P 和 Q 两阶段复用**

- Hessian 估计: **H = 2·X·Xᵀ** (X 为层输入激活矩阵，~128 calibration samples) [待核:正文—精确系数和样本数]
- **SparseGPT 剪枝更新** (OBS, Optimal Brain Surgeon):
  移除权重 w_q 后，对残余权重的补偿:
  `Δw_R = −w_q · H⁻¹_{R,q} / H⁻¹_{q,q}` (对每个保留权重 R，按 Hessian 逆矩阵插值) [待核:正文]
- **GPTQ 量化** 使用同一 H:
  每权重量化后，对同行其余权重补偿量化舍入误差，公式形式相同 [待核:正文]
- **Hessian 复用的收益**: 计算一次 H⁻¹ 分摊到 P 和 Q 两步 → 顺序串行的**计算成本被均摊**
- 适用场景: LLM 一次性压缩（不再训练），需 calibration 数据无需梯度

---

### 6.2 联合可微实现 (Joint Differentiable)

**代表方法**: DJPQ (Wang, Lu, Blankevoort. ECCV 2020), CLIP-Q (Tung, Mori. CVPR 2018)

#### DJPQ 统一损失函数 [文献声称, arXiv:2007.10463 摘要核查]

将 VIB 剪枝和混精量化嵌入**单一可微损失**，梯度同时更新两类变量:

```
L_total = L_task + α·L_prune_VIB + β·L_quant_bit
```

**L_prune_VIB** (变分信息瓶颈剪枝项) [待核:正文—精确公式]:
- z ∈ {0,1}^C 为通道级重要性门控变量; q(z|W) = 变分后验; p(z) = ∏Bernoulli(π_target) 为稀疏先验
- 正则项 = KL[q(z|W) || p(z)], 鼓励通道被关闭
- 训练时用 RelaxedBernoulli (Hard Concrete) 参数化, 使梯度可通过离散掩码流过 ∂L/∂θ_z

**L_quant_bit** (量化位宽代价项) [待核:正文—精确公式]:
- 每层位宽 B_l 从 {1,2,4,8} bit 候选中软选择 (Gumbel-Softmax 离散松弛)
- 代价 = 期望位宽 E[B_l], 鼓励选低 bit
- 温度退火: 训练末期 τ→0, 软选择收敛为 hard one-hot

**梯度同时更新两类变量**:
- ∂L/∂θ_z: 通过 RelaxedBernoulli 的 STE 反传 → 更新剪枝门控
- ∂L/∂θ_B: 通过 Gumbel-Softmax 梯度 → 更新位宽选择分布
- **联合优化意义**: 位宽选择会影响哪些通道值得保留; 通道是否被剪影响量化误差的分布 → 两者互相反馈

**2^k 位宽约束** [文献声称, GitHub 核查]:
- 位宽强制为 2 的幂次 (1/2/4/8 bit), 天然满足 TRT INT8 内核 %32 对齐要求

---

#### CLIP-Q 并行联合 (P ∥ Q ∥ finetune) [文献声称, CVF PDF 核查]

不分阶段, P 和 Q **在每次 mini-batch 迭代中同时执行**:
- 每步前向: 对当前权重 (i) 查找最近 codebook centroid (量化), (ii) 根据重要性决策是否置零 (剪枝)
- 每步反向: 梯度对 centroid 和剪枝掩码同时更新
- **消除"过早剪枝"问题**: 被 P 移除的连接 Q 无法补偿; 并行允许两者互相修正, 可撤销早期错误决策
- 效果: AlexNet **51×** 压缩比 vs Deep Compression 顺序串行 **35×** (+46%) [文献声称]

---

### 6.3 统一参数化实现 (Unified Parameterization)

**代表方法**: Bayesian Bits (van Baalen et al. NeurIPS 2020)

**核心思想: 0-bit 量化 = 剪枝**, 将 P 和 Q 统一到同一参数空间

#### 级联残差量化分解 [文献声称, GitHub + NeurIPS 摘要核查]

**分解式**: b-bit 量化 = k 个独立的 1-bit 量化残差级联叠加:
```
x_quant = Σ_{j=1}^{k} z_j · quant_{1-bit}(residual_j)
```
其中 residual_j = x − Σ_{i<j}(之前各级量化值), z_j ∈ {0,1} 为门控变量 [待核:正文—精确公式符号]

**门控精度选择**:
- z_j 由 learnable stochastic gate 控制, 受稀疏先验正则 KL[q(z_j)||p(z_j)]
- 有效精度 = 2^(Σ z_j): z=(1,0,0,...) → 1-bit; z=(1,1,0,...) → 2-bit; ...
- **z_j=0 for all j → 有效精度=0 → 权重完全被抑制 = 剪枝**

**统一优化**:
- 单一损失: L_task + λ·Σ KL[q(z_j)||Bernoulli(π_target)]
- 优化变量: 模型权重 W + 门控参数 θ_z (同时优化)
- 硬件对齐: 位宽限制在 2^k 形式, 满足 Tensor Core 要求

**意义**: 业界首次将"剪枝"和"量化"统一为同一优化变量的不同取值, 无人为阶段边界 [文献声称]

---

### 6.4 预测器/搜索实现 (Accuracy Predictor + Evolutionary Search)

**代表方法**: APQ (T. Wang et al. CVPR 2020)

#### 量化感知精度预测器训练 [文献声称, arXiv:2006.08509 摘要核查]

**阶段①: FP32 精度预测器**
- 从 OFA (Once-for-All) 超网随机采样子网 → 直接在 ImageNet val 子集评估精度 (无需重新训练, 利用超网权重共享)
- 样本数: [待核:正文—原文约 160K 子网]
- 训练 MLP 回归器: 输入 = 网络配置向量 (架构/剪枝率/量化位宽), 输出 = 预测精度

**阶段②: INT8 精度预测器迁移**
- FP32 预测器已习得网络结构先验 → INT8 预测器只需少量 QAT 样本做知识迁移 (~1000 samples) [待核:正文]
- 收益: 避免对每个 INT8 候选做完整 QAT (极大降低搜索成本)

**进化搜索编码三维配置** [文献声称]:
- 染色体 = [per-layer 架构配置, per-layer 剪枝比例, per-layer 量化位宽] 拼接向量 [待核:正文—精确编码格式]
- 搜索约束: 延迟/能量预算 (LUT 查找或在线实测)
- 进化: 候选种群, top 保留交叉变异, 多代后收敛 [待核:正文—种群大小和代数]
- **三维不可分解性 (§1 APQ 引证)**: 最优 Q 依赖 P 结果, 最优 P 依赖 Q 误差放大模式; 进化搜索在联合空间找最优, 比串行独立优化每维效率高 **600×** [文献声称]

#### ★定位澄清: APQ 是"搜索级联合", 不是"优化级融合" [v1.5 增补, 应用户质疑]

用户质疑成立一半, 需精确化: APQ **是** P×Q 联合方案(标题/摘要明示 "jointly" 优化 architecture/pruning/quantization, 已核 [arXiv:2006.08509](https://arxiv.org/abs/2006.08509) + [CVPR 开放获取版](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.html)), 但其"联合"的实现层次与 6.2/6.3 类方法**本质不同**:

| 维度 | 优化级融合 (DJPQ/Bayesian Bits/GETA) | **搜索级联合 (APQ, 本节)** |
|------|--------------------------------------|---------------------------|
| P 和 Q 在哪里相遇 | 同一个损失函数/同一组参数, 训练时梯度直接交互 | **不在联合损失中相遇**(量化仅出现于预测器数据采集的短期 finetune, 原文 Fig 2 "Quant-Aware Fine-Tuning") — 主要作为配置向量的两段, 在搜索器和预测器中相遇 |
| "剪枝"的形态 | 对已训练网络施加掩码/结构删减 | **OFA 超网的弹性通道宽度选择**(architecture-embedded channel pruning), 更接近 NAS 而非经典后训练剪枝 |
| 耦合怎么被捕获 | 机理式: 损失函数显式建模交互 | **统计式: 量化感知精度预测器从 (配置, 实测精度) 样本中学到 P×Q 交互**, 搜索器据此避开坏组合 |
| 联合的证据形式 | 联合训练 vs 串行训练的端到端对比 | 联合搜索 vs 串行搜索(ProxylessNAS+AMC+HAQ)的搜索效率/结果对比 |

**对本报告分类学的意义**: §6 的 7 类实现范式实际横跨一条"**耦合实现层次轴**": 训练层(6.2/6.3)→ 补偿层(6.5)→ **搜索层(6.4)**→ 部署层(6.6)。APQ 归在搜索层是正确归类, 但引用它时**不应表述为"剪枝量化融合算法"**(那是 6.2/6.3 的措辞), 应表述为"P×Q 联合搜索方案"。

**对我方框架定位的意义** [推断]: 我方框架(DepGraph 后训练结构剪枝 × TRT INT8 × D 部署维, 预测器驱动搜索)与 APQ 同属**搜索级联合**——耦合同样由预测器统计捕获而非损失函数机理建模。差异: ①我方 P 是经典后训练剪枝(非超网宽度选择), 更贴近部署现实; ②我方耦合信号来自**真硬件在环测量**(非 LUT); ③我方多出 D 维。这意味着 APQ 是我方最近的方法学亲戚, 与它的对比(而非与 DJPQ 类的对比)才是 related work 中的正确对位。

---

### 6.5 训练无关补偿实现 (Training-Free Compensation)

**代表方法**: OBR (Guo, Li, Benini. 2025), SLiM (Mozaffari et al. 2025)

#### OBR Hessian 联合误差补偿 (闭合解) [文献声称, arXiv:2509.11177 HTML 核查 ✓]

**目标函数** (输出扰动最小化):
```
min ΔL ≈ (1/2) · Σᵢ E[Δwᵢ · H · Δwᵢᵀ]
H = 2·X·Xᵀ  (X 为层输入激活, 128 samples from WikiText2, seq_len=2048)
```

**闭合解** — 对每行权重, 分"保留集 R"和"驱逐集 E":
```
ΔwR* = −H_RR⁻¹ · H_RE · e_E
```
- **剪枝补偿**: e_E = w_E (被剪权重本身), 补偿传递信息到保留权重
- **量化补偿**: e_E = w̄_E − quant(w̄_E) (量化舍入误差), 同一公式形式
- **两种误差共享同一闭合解框架** — 这是"联合"的数学本质

**特性**: 训练无关 (calibration-only, 无梯度更新); 应用于 Llama2-7B W4A4KV4+50%稀疏, 实现 4.72× 加速/6.4× 内存削减 [文献声称]

---

#### SLiM one-shot Q → 2:4 → LoRA 补偿管线 [文献声称, arXiv:2410.09615 HTML 核查 ✓]

严格顺序三步, 均为 one-shot (无迭代再训练):

**步骤①: SLiM-Quant 均匀量化**
- 概率优化框架: 将量化表述为最小化期望平方误差; 对权重分布做数值积分求最优截断 α
- 多格点精化 (multi-grid refinement): 先粗搜范围 (0, max|W|) 10均匀点, 迭代细化误差最小区域
- 输出: 量化后权重 W_q

**步骤②: 2:4 半结构化稀疏**
- 对已量化权重 W_q 调用现成 one-shot 剪枝方法 (如 Wanda)
- 施加 2:4 pattern: 每连续 4 个权重中保留 2 个非零 → 满足 NVIDIA Sparse Tensor Core 要求

**步骤③: SLiM-LoRA 闭合解适配器**
- saliency 函数: **F(W) = diag(x̄)·W** (x̄ = 校准集平均绝对激活, 衡量权重对输出的贡献)
- LoRA 适配器权重 A,B 通过 SVD 分解 saliency 矩阵得到闭合解 (无反向传播)
- 需要校准数据计算 x̄; rank 为可调超参数 [文献声称]

**关键顺序**: Q first → 2:4 on top of quantized weights → LoRA compensates joint Q+P error

---

### 6.6 部署级融合实现 (Deployment-Level Pipeline)

**最贴近我方工作的实现形态**: DepGraph 结构剪枝 → ONNX → TRT INT8 MinMax 校准 → Engine build

#### 我方管线 [我方实证, 完整口径]

**步骤①: DepGraph 结构化剪枝**
- 工具: `tools/configurable/depgraph_pyramid.py`
- 准则: L1/FPGM/Taylor/Wanda (可配置); DepGraph 全网依赖图确保剪枝不破坏结构一致性 (backbone+neck+deblock 全可剪)
- **关键约束**: `round_to=32` — 剪枝后通道数须 %32==0, 确保后续 INT8 Tensor Core 选快速内核
- 输出: flat state_dict 格式 checkpoint (非包裹格式, 否则 HEAL load 时 key missing)
- finetune: 剪枝后必须 finetune 才算有效 AP 数据 (未 finetune 必崩)

**步骤②: ONNX 导出**
- 已剪枝模型导出为 ONNX (真结构化重建, 非 mask-based)
- 关键: **必须是真结构剪枝** — mask-based 的稀疏 (权重=0 但矩阵形状不变) 被 TRT 视为 dense, 无延迟收益且浪费 24h 实验 (已排查, nouse)

**步骤③: TRT INT8 MinMax 校准**
- 校准策略: **MinMax calibrator** (Entropy calibrator 在我方模型上 AP 崩塌, ISS-017/ISS-023)
- 数据: DAIR val 子集作为校准集
- TRT-auto 混精: TRT 逐层自动选 INT8/FP16 (延迟驱动), 不手工指定 per-stage

**步骤④: Engine build + 延迟验证**
- trtexec loadEngine; warmup=200/runs=200; CUDA Event 或 GPU_Compute_median 计时
- 验证文件: `results/E7_orin_e2e_baseline_vs_best.csv` (Orin p75+INT8=20.02ms)
  `results/perstage_quant_AP_real_v2.csv` (4090 p75+INT8=0.6124ms)

**实现约束验证** [我方实证, ISS-014]:
- 剪枝通道不对齐 (p25: 48→96ch, 非%32) → TRT 选慢速 `implicit_gemm_g1` kernel → INT8 增益从 1.57×→~1.06×
- 因此: DepGraph 的 `round_to=32` 是 P 和 Q 的**实现层耦合点** — 剪枝阶段必须感知量化的对齐约束

---

#### 2:4 Sparse Tensor Core + INT8 硬件级复合融合 [文献声称]

NVIDIA Ampere/Hopper 上可同时启用结构化 2:4 稀疏和 INT8 量化:
- 先做 2:4 结构化剪枝 (每 4 个权重保留 2), 再 INT8 量化
- 稀疏 Tensor Core 自动处理非零权重的 INT8 运算
- 理论叠加: INT8 alone (约 2×) × 2:4 sparse (约 2×) ≈ 4× over FP16-dense [文献声称, `pruning_x_hardware.md` §1.1]
- 我方状态: **未实测** (模型未做 2:4 剪枝); DLA INT8 路径已知不可达 (ISS-007, §3.5)
- 实现门槛: Pyramid/SpVoxelNet 卷积层需确认是否满足 2:4 pattern 约束

---

### 6.7 实现层对比表

| 实现范式 | 代表方法 | 训练成本 | 是否需要数据 | 硬件感知程度 | 典型适用场景 |
|---------|---------|---------|------------|------------|------------|
| **顺序流水** | Deep Compression, SparseGPT+GPTQ | 低-中 (分段 finetune/无梯度) | 需要 (finetune 或 calibration) | 低 (无在线硬件反馈) | 有完整训练集; 精度优先; LLM 一次性压缩 |
| **联合可微** | DJPQ, CLIP-Q | 高 (联合全程训练) | 需要 (完整训练集) | 中 (通过 HW-cost proxy 损失项) | CNN 中小模型; 精度极致优化 |
| **统一参数化** | Bayesian Bits | 高 (统一训练) | 需要 (完整训练集) | 中 (2^k 位宽硬件友好) | 位宽+稀疏率联合搜索; 研究工具 |
| **预测器搜索** | APQ | 中 (预测器+进化搜索) | 需要 (OFA 超网+少量 QAT) | 高 (LUT/实测延迟约束) | NAS+P+Q 联合; 大规模产品搜索 |
| **训练无关补偿** | OBR, SLiM | 极低 (闭合解, 无梯度) | 需要 (校准集 ~128–1000 samples) | 低-中 (结构不变) | LLM 快速部署; 无训练资源场景 |
| **部署级融合** | **我方管线** (TRT) | 低 (校准无梯度; finetune 仅在 P 后) | 需要 (DAIR val 校准集) | **高** (真实硬件 latency 反馈, CUDA Event 实测) | **边缘部署 (Orin/4090); 真实 Pareto 测量** |

**我方管线的独特性**: 是上表中唯一以**真实硬件延迟**（非 LUT 估算）作为 Pareto 反馈的范式，且覆盖 B1(剪枝)×B2(量化)×D(部署配置) 三维联合搜索空间（§7.2 文献空白）。

---

## 7. 我方框架定位

### 7.1 我方 vs 文献方法对应关系

| 文献现象 | 我方实测 | 差异/共同点 |
|---------|---------|-----------|
| DJPQ: 通道对齐是联合约束 | ISS-014: 32对齐悬崖 2.57× | **完全吻合**。我方有更精确的 profile 数据 (TacticValue) |
| Harma 2025: P before Q 理论最优 | 我方实验顺序: 先剪枝再量化 | **顺序一致**，但我方未做反向对照实验 [推断 可增补] |
| APQ: 三维联合优于独立串行 | 我方: Pareto 扫描覆盖 B1×B2 联合维度 | **动机一致**。我方新增 D 维度 (hardware deployment) |
| Bayesian Bits: 统一参数化 | 我方: 分维度扫描 (非统一参数) | **差异**: 我方不尝试统一参数化 |
| OBR: P×Q 分布冲突 | [推断] Pyramid 过参数化使冲突弱化 (INT8 near-lossless) | 在过参数化域冲突被模型冗余吸收 |
| JSQ: Outlier 是共同瓶颈 | `dims_quantization_v1.md`: INT8 SNR <1，无显著 outlier | Pyramid 无 outlier 问题 — 冗余充足 |

### 7.2 我方框架的独特性 (文献空白)

[我方实证] 来源: `paper_learning/survey_raw_2/hw_aware_nas_and_joint_search.md` §10.2

当前文献中：
- **APQ**: 覆盖 A+B1+B2（架构+剪枝+量化），**不含 D (部署配置)**
- **HAQ**: 覆盖 B2+D（量化+延迟反馈），**不含 B1（剪枝）**
- **硬件感知 NAS 文献**: 覆盖 A+D，**不含 B**

我方 sw-hw-cooptim 框架的独特定位：**在 V2X 协同感知场景下，B1（通道剪枝）× B2（混精量化）× D（部署配置：platform, precision, kernel）联合搜索**，且以**真实硬件延迟（非 LUT 估算）** 作为 Pareto 维度，是文献中已识别的空白区。

### 7.3 已获得的关键联合证据 (P×Q 联合优于串行的具体论据)

1. **内核切换悬崖** [我方实证, ISS-014]: P 后通道数必须与 Q 内核粒度共同约束，否则 Q 收益蒸发（2.57× 等计算量差异）→ 这是"P 必须感知 Q"的硬约束
2. **AP 超加性惩罚** [我方实证, perstage_quant_AP_real_v2.csv]: P50+INT8 的 AP 代价 (−0.039) > P50-only (−0.027) + INT8-only (−0.0005) → 独立串行高估了联合精度
3. **TRT-auto 支配手工混精** [我方实证]: 手工 per-stage 精度分配（即 Q 独立于 P 后结构决策）全部被 TRT-auto 支配 → 量化内部的"局部分配"也需要全局优化
4. **wpg-P×Q 双约束** [我方实证]: 剪枝搜索空间上界被量化对齐需求硬裁 → P 搜索必须感知 Q 约束，无法独立搜索

---

## 8. 完整文献索引

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

[Wang2020-DJPQ] Wang, Ying and Lu, Yadong and Blankevoort, Tijmen.
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

[Harma2025] Harma, Simla Burcu et al.
"Effective Interplay Between Sparsity and Quantization."
ICLR 2025. arXiv:2405.20935
https://arxiv.org/abs/2405.20935
Project: https://sq-interplay.github.io/

[Qu2025-GETA] Qu, Sifan et al.
"GETA: Quantization-Aware Pruning via Dependency Graph."
CVPR 2025. arXiv:2502.16638
https://arxiv.org/abs/2502.16638

[Mozaffari2025-SLiM] Mozaffari, Mohammad et al.  [待核:正文—一作名原记录为"Jafar"存疑, 已据supervisor指示改为"Mohammad"]
"SLiM: One-Shot Quantization and Sparsity with Low-Rank Approximation."
ICML 2025. arXiv:2410.09615
https://arxiv.org/abs/2410.09615

[Guo2025-OBR] Guo, Hang and Li, Yawei and Benini, Luca.
"OBR: Optimal Brain Restoration for Extreme LLM Compression via Joint Sparse-Quantized Error Compensation."
arXiv:2509.11177, 2025.
https://arxiv.org/abs/2509.11177
GitHub: https://github.com/csguoh/OBR

[Kim2026-PIH] Kim, Jeonghoon et al.
"Prune-then-Quantize or Quantize-then-Prune? Understanding the Impact of Compression Order in Joint Model Compression."
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
| INT8 能效 −51.6% (4090) | `results/E4_energy_4090.csv` (T1_base_fp16/int8, NVML, B=1) | [我方实证] |
| 联合延迟 4090 2.08× + Orin 6.56× | `results/perstage_quant_AP_real_v2.csv` + `results/E7_orin_e2e_baseline_vs_best.csv` | [我方实证] |
| 文献空白定位 | `paper_learning/survey_raw_2/hw_aware_nas_and_joint_search.md` §10.2 | [文献声称+我方推断] |
| APQ 对比增益 | `paper_learning/survey_raw_2/pruning_x_hardware.md` §6.3 | [文献声称] |

---

*本文档由 sw-optimizer 撰写，供 supervisor 逐引核验。所有 [文献声称] 均经 web-agent WebFetch 核查原文摘要/paper page；所有 [我方实证] 均标注具体数据文件和测量口径；[推断] 明确区分。*
