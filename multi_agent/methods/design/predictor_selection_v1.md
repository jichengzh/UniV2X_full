# 预测器选型与训练方案 v1 — 为什么是 GBDT (LightGBM), 以及怎么训

> 创建: 2026-06-05 | 状态: 设计决议 (基于文献 + 框架内已有实证)
> 角色: 框架闭环中的 f: Config(B,Q,D) → {AP, latency, throughput, energy, model_size} 性能预测器, 供搜索器在不真测的情况下评估候选配置, 并驱动 active-learning 采样。
> 相关文档: `doe_design_v1.md` (采样策略), `dims_*.md` (特征来源), `../../background/00_研究目标与实验档案_v1.md` (5 指标定义)
> 文末附完整参考文献; 正文引用以 [n] 标注。**凡是框架内实证, 均标注数据文件出处, 与文献证据区分。**

---

## 一、任务规格 — 我们的预测问题长什么样

预测器选型不能照搬 NAS 文献的默认答案, 必须先从我们自己的数据约束出发。框架的预测任务有 7 个硬特征:

| # | 特征 | 事实依据 (框架内) |
|---|------|------------------|
| R1 | **极小样本**: 完整点 (lat+AP) 仅 33, 加 latency-only 共 ~200 行; 论文周期内乐观上限 O(10²~10³) | `multi_agent/data/dataset_v2.csv` 65 行感知主表(×93 列, 含闭环 wire) + ~140 行 D 维 bench;★训练用精简学习视图 `dataset_v2_learning.csv`(每维度1规范列: ap70/lat_p50_ms/throughput_fps/energy_per_frame_mj/engine_size_mb) |
| R2 | **表格型混合特征**: 数值 (stage planes, workspace_gb, prune_rate) + 类别 (calibrator, d_scheme, d_tactic, q_mode, hardware) + 布尔 (build_success), 共 ~30 列 | `schema_v2.md` |
| R3 | **目标函数高度非光滑**: kernel 对齐悬崖 (32-对齐 INT8 加速 11–19% / 8-对齐 ≈0)、entropy 校准崩塌 (AP 0.79→0.14)、DLA INT8 build 全失败 (0/12) — 都是阶跃, 不是平滑趋势 | `dataset_analysis.md` 综合反思; `dims_quantization_v1.md` ISS-014 |
| R4 | **条件/受限空间**: C1–C11 耦合约束使空间非笛卡尔 (如 q_bits=FP32 ⇒ granularity=none; 2:4 ⇒ 必须 sparse_tc IP) | `搜索空间一览.md §四` |
| R5 | **多目标 + 不同难度**: lat/throughput/energy 信号强 (~3× span), AP 在非崩塌区几乎平坦 (4pp 窄带) — 单一模型同时拟合是浪费 | `创新点.md §1.2` 信息密度表 |
| R6 | **需要不确定性输出**: 采样预算昂贵 (每个完整点 = TRT build + 1789 帧 AP 真测 ≈ 4–6 min GPU 时), 必须 active learning, acquisition 需要预测方差 | `doe_design_v1.md`; 创新点 Stage 1b |
| R7 | **固定拓扑、变配置**: 我们搜的不是网络结构图 (没有变化的算子图), 而是同一拓扑下的宽度/精度/部署配置 — 特征天然是表格, 不是图 | `dims_pruning_v1.md` (planes/wpg 是标量轴) |

R7 一条就排除了一大类 NAS 文献的默认方案 (GCN/GNN 编码架构图), 下面逐类对比。

---

## 二、候选方法谱系与逐项裁决

| 方法 | 代表文献 | 机制 | 对照任务规格的裁决 |
|------|---------|------|--------------------|
| **查表 (LUT)** | FBNet [9], ProxylessNAS [17] | 逐 op 实测延迟求和 | ❌ 假设延迟逐层可加。我们实测明确违背: TRT 跨层融合 + tactic 全局选择使整网延迟非逐层和 (TRT FP32 3.36ms ≠ PyTorch 逐层和的任何缩放); per-stage 混精 18/18 比全局 INT8 慢 17–43% 正是"边界效应不可加"的反例 (`results/perstage_quant_pareto.md`) |
| **kernel 级分解预测** | nn-Meter [6] | 按融合规则切 kernel, 每类 kernel 单独回归 | ⚠️ 思想正确 (它承认融合), 但需要对每个目标设备做大规模 kernel 基准库 (nn-Meter 每设备数万次测量), 我们 2 个平台 O(10²) 预算建不起; 且它只预测 latency, 不覆盖 AP/energy。其"kernel 对齐特征"思想我们以特征工程形式吸收 (见 §五) |
| **GP / 贝叶斯优化** | Snoek et al. [14] | 高斯过程后验 + acquisition | ⚠️ 不确定性原生支持是优点; 但标准 GP 核假设平滑性 (违背 R3 的阶跃悬崖), 对类别/条件特征 (R2/R4) 需要专门核设计, 文献共识是这类空间 RF/GBDT 代理更稳 — SMAC 用 RF 替代 GP 的动机正是"条件、类别、非平滑" [13] |
| **MLP 回归** | OFA [8] | 全连接网络拟合 (arch, lat) | ❌ OFA 的 MLP 预测器用 **16K 样本**训练; 我们 O(10²) 样本远低于神经网络起效区。Grinsztajn et al. [15] 的系统对比: 中小规模表格数据 + 不规则目标函数上树模型稳定优于深度模型, 原因恰是树对"非光滑/阶跃"目标的归纳偏置 (对应我们 R3) |
| **GCN/GNN 架构编码** | BRP-NAS [5], TPU 学习代价模型 [19] | 把算子图编码成图神经网络输入 | ❌ 解决的是"拓扑可变"的搜索空间 (NAS-Bench cell 结构); 我们是固定拓扑 + 表格配置 (R7), 图编码退化为定长向量, GNN 只剩开销没有收益 |
| **元学习跨设备** | HELP [7] | 少样本适配新硬件的 latency 预测 | ⏳ 当前不需要 (只有 4090/Orin 两个平台, 各自直接训练); 若未来扩到 >5 类边缘设备, 这是首选升级路径 (见 §六) |
| **GBDT (LightGBM/XGBoost)** | LightGBM [1], XGBoost [2] | 梯度提升决策树 | ✅ **逐条命中**: 树分裂天然拟合阶跃 (R3); 原生类别特征 + 对缺失/条件维度鲁棒 (R2/R4); 小样本表格是其主场 [15]; per-target 独立训练成本可忽略 (R5); quantile/ensemble 提供不确定性 (R6, 见 §五) |

### 文献侧的横向证据 (不止单篇背书)

1. **NAS 预测器系统评测**: White et al. [11] 在 NeurIPS 2021 对 31 种性能预测器做了统一评测, 核心结论是预测器优劣强依赖训练预算 regime —— 在**中小训练预算 (数十~数百个标注架构)** 下, 基于树的模型 (XGBoost/LGBoost/NGBoost/RF) 属于持续第一梯队, 而 GCN/深度预测器需要更大预算才反超。我们的 O(10²) 预算正落在树模型的优势区。
2. **代理基准的工程选择**: NAS-Bench-301 [12] 为 10^18 规模空间建代理模型时, GBDT (XGBoost/LightGBM) 与 GIN 同列最优家族, 且 GBDT 训练成本低一个量级以上。
3. **张量编译器的工业实践**: AutoTVM 的代价模型就是 XGBoost (Chen et al. [3]); TenSet [4] 在大规模程序性能数据上系统对比, GBDT 与 MLP 在 kernel 延迟预测上同档, 而 GBDT 无需 GPU 训练、可解释 (feature importance 直接审计)。**预测"编译后硬件延迟"这一具体任务, 工业主流就是 GBDT。**
4. **表格数据上的树模型优势**: Grinsztajn et al. [15] (NeurIPS 2022 D&B) 用 45 个数据集证明中等规模表格任务上树模型系统性优于深度模型, 机制分析指出树对**不规则 (non-smooth) 目标函数**的偏置是关键 — 这正是我们 R3 (kernel 悬崖/校准崩塌) 的形状。

---

## 三、框架内已有实证 — 我们自己踩过的坑就是选型依据

文献只回答"哪一类模型", 训练协议必须由框架内的失败案例决定:

| 实证 | 事实 | 对训练方案的约束 |
|------|------|------------------|
| **lgb_v6_latency 崩** | 跨数量级 (Pyramid ~ms 级与 UniV2X ~百 ms 级混训) 直接给出负延迟预测, 已弃用 | **禁止跨模型量级混训**: 按 model_class 分层训练, 或 target 取 log 并加 per-model offset; 见 `CLAUDE.md §四` |
| **f_lat 天花板 R²≈0.73** | 8 个简单特征下 LGB 容量扫描 + 加密 Q 特征都突破不了 0.73 (oracle 实验) | 瓶颈是**特征缺口不是模型容量** — 缺的是结构-硬件交互特征 (如 `n_non_pow2_groups`), 不是更深的树; 换更大模型无意义 (memory: `project_f_lat_ceiling`) |
| **`n_non_pow2_groups` 特征发现** | T2/T3 延迟反弹根因 = ResNeXt grouped conv `in_per_group` 非 2 的幂触发 cuDNN fallback; `stage*_planes` 原始值预测不了这个 | 特征工程必须显式编码 kernel 对齐性 (`dataset_analysis.md` 反思 #5) |
| **AP × Q 信息密度极不均** | 非 entropy 配置 AP 全在 4pp 窄带; entropy 16/32 崩到 0.14–0.27 | AP 预测器**必须分解**: 平坦区回归 + 崩塌分类, 单一回归器会被崩塌点拖垮整体 (创新点 Stage 1c 三分解) |
| **per-stage 混精 18/18 负样本** | 全部比全局 INT8 慢 17–43%, 但保留在数据集 | 负样本/失败 cell (build_fail, AP 崩) 是预测器的关键训练数据, 不清洗掉 |
| **lgb_v6_amota 指标污染 (ISS-026)** | Pyramid 段拿 AP 当 AMOTA 拟合 | 每个 target 列必须物理同质; 跨模型预测前先做指标审计 |

---

## 四、决议 — 预测器架构

**主干: per-target 独立 LightGBM [1], 按 (model_class, hardware) 分层, 五目标五组模型 + 两个分类头。**

```
f_feasible : Config → {build_success, AP_crash}        ← LGB 二分类 ×2 (先于一切回归)
f_lat      : Config → log(lat_p50)                     ← LGB 回归, 分层 by (model, hw)
f_tput     : Config → log(throughput)                  ← batch>1 数据就绪前 = 1000/lat 推导, 不单独训
f_energy   : Config → log(J_frame)                     ← LGB 回归 (E4/E5/E6 数据)
f_AP       : Config → ΔAP vs 同 ckpt baseline          ← LGB 回归, 仅在 f_feasible 判"不崩"的区域有效
f_size     : Config → params_kb / engine_size_mb       ← params 解析公式直接算, engine_size 用 LGB
```

设计理由逐条:

1. **先分类后回归 (两段式)**: 崩塌/失败是阶跃事件, 让回归器去拟合 0.79→0.14 的跳变会污染平坦区精度 (框架内实证 §三)。f_feasible 把空间切成"可用/不可用", 回归器只在可用区训练。这等价于 White et al. [11] 中表现稳健的"分而治之"实践, 也与 SMAC 在受限空间先建可行性模型的做法一致 [13]。
2. **lat 取 log + 分层**: 直接回应 lgb_v6_latency 跨数量级崩的事故; log 域中树的相对误差均匀, 且杜绝负值预测。
3. **AP 预测 ΔAP 而非绝对值**: AP 绝对值由 ckpt 质量主导 (base 0.791 是任务上限), 配置效应是小信号 (span ~0.04); 预测残差比预测绝对值的信噪比高一个量级 (`stage_a_ap_real.parquet` 的信号结构)。
4. **throughput 暂不独立训**: 当前 corr(tput, 1/lat) = 0.9996 (`dims_hardware_v2.md §0.5`), 独立预测器是冗余; **batch 轴数据落地后升级为独立 LGB** (那时 tput 才与 1/lat 解耦)。
5. **不确定性 = quantile LightGBM 三分位 (α=0.1/0.5/0.9) 或 5-seed bagging ensemble**: LightGBM 原生支持 quantile objective; ensemble 方差作为 acquisition 信号的做法在深度与树模型上均已验证 [16][18]。两者都比给 GBDT 套贝叶斯后验便宜且够用 — active learning 只需要**相对**不确定性排序, 不需要校准的绝对后验。

**特征工程清单 (v1 必含, 来自 §三 实证):**

- 结构: `stage{0,1,2}_planes`, `prune_rate`, `params_kb`, `wpg`, `deblocks_keep`, `shrink_keep`
- **对齐交互 (关键)**: `n_non_pow2_groups` (0–3), `min_plane_mod32` (planes 是否 32 整除), `is_mix_q` (bool)
- 量化: `q_mode`(类别), `calibrator`(类别 — entropy 是崩塌分类头的主特征), `q_granularity`, `q_object`
- 部署: `hardware`(类别), `d_scheme`(类别), `d_tactic`(类别), `d_workspace_gb`, `opt_level`
- 元数据: `ckpt_status`(类别 — ckpt_source × entropy 耦合的载体), `latency_kind` (口径隔离, 同口径内训练)

**训练协议:**

- 验证: **GroupKFold, group = triplet** (同一剪枝架构的不同量化/部署变体不允许跨 fold 泄漏 — 否则 R² 虚高; 这是 NAS 预测器评测的标准做法 [11])
- 同 `latency_kind` 内训练, 禁止 body_subnet 与 e2e 口径混合 (口径差 ≈10ms 会成为伪信号)
- 超参: 小样本下用浅树强正则 (num_leaves ≤ 15, min_data_in_leaf ≥ 5, feature_fraction 0.8), 容量扫描已证明加大无收益 (§三 f_lat 天花板)
- 每次重训输出 feature importance 审计表, 出现物理不可解释的 top 特征 (如 config_id 泄漏) 即停

**Active learning 闭环 (与 doe_design_v1 对接):**

1. pilot 集 (现有 ~200 行) 训 v0 → 2. quantile 宽度 × Pareto 邻近度加权 acquisition 选下一批 K=8~16 → 3. data-orchestrator 下采样指令真测 → 4. 并回 dataset_v2 重训。敏感度分层采样 (高方差的 B×Q、Q×D 耦合区多采, AP 平坦区少采) 的离线模拟显示 K=32 即可到 R²≥0.85 (创新点 Stage 1a 设计, 待跑)。

---

## 五、对比结论一览 (为什么不是别的)

| 备选 | 一票否决点 | 证据 |
|------|-----------|------|
| LUT 逐层求和 | TRT 融合使延迟不可加 | per-stage 18/18 反例 (框架内) + [6] 的动机章节 |
| nn-Meter kernel 级 | 每设备需 O(10⁴) kernel 基准, 预算不允许; 不覆盖 AP/energy | [6] 实验设置 |
| GP | 平滑核 vs 阶跃目标; 类别/条件特征支持差 | [13] 的 RF-代理动机; R3/R4 |
| MLP | 需要 10³~10⁴ 样本起步 | [8] 用 16K 样本; [15] 小表格数据系统对比 |
| GCN (BRP-NAS) | 我们拓扑固定, 无图可编码 | R7; [5] 适用前提是 cell 拓扑可变 |
| HELP 元学习 | 当前只有 2 平台, 无"新设备少样本"问题 | [7] 的问题设定; 留作升级路径 |
| **LightGBM (选定)** | — | [1][3][4][11][12][15] + §三全部框架内实证 |

## 六、何时换方法 (升级触发条件, 预先写死防漂移)

1. **设备种类 > 5** (如加入 Thor/Xavier/嵌入式 GPU 矩阵) → 引入 HELP 式元学习层 [7], LGB 退为单设备基模型。
2. **搜索空间引入拓扑变化** (如算子替换/层数可变) → 特征表格化失效, 评估 GCN 编码 [5][19]。
3. **数据量过 5×10³ 且 R² 仍卡** → 重新评估 MLP/深度模型 (届时进入 [15] 中深度模型开始追平的数据 regime)。
4. **AP 崩塌模式增多到分类头 F1 < 0.9** → 崩塌机制建模升级 (per-机制子分类器, 如 entropy/对齐/DLA 分开)。

---

## 参考文献

[1] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 2017.

[2] T. Chen, C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.

[3] T. Chen, L. Zheng, E. Yan, Z. Jiang, T. Moreau, L. Ceze, C. Guestrin, A. Krishnamurthy. *Learning to Optimize Tensor Programs.* NeurIPS 2018. (AutoTVM 的 XGBoost 代价模型)

[4] L. Zheng, R. Liu, J. Shao, T. Chen, J. E. Gonzalez, I. Stoica, A. H. Ali. *TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers.* NeurIPS 2021 Datasets and Benchmarks Track.

[5] Ł. Dudziak, T. Chau, M. S. Abdelfattah, R. Lee, H. Kim, N. D. Lane. *BRP-NAS: Prediction-based NAS using GCNs.* NeurIPS 2020.

[6] L. L. Zhang, S. Han, J. Wei, N. Zheng, T. Cao, Y. Yang, Y. Liu. *nn-Meter: Towards Accurate Latency Prediction of Deep-Learning Model Inference on Diverse Edge Devices.* MobiSys 2021 (Best Paper).

[7] H. Lee, S. Lee, S. Chong, S. J. Hwang. *HELP: Hardware-Adaptive Efficient Latency Prediction for NAS via Meta-Learning.* NeurIPS 2021.

[8] H. Cai, C. Gan, T. Wang, Z. Zhang, S. Han. *Once-for-All: Train One Network and Specialize It for Efficient Deployment.* ICLR 2020.

[9] B. Wu, X. Dai, P. Zhang, Y. Wang, F. Sun, Y. Wu, Y. Tian, P. Vajda, Y. Jia, K. Keutzer. *FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search.* CVPR 2019.

[10] M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, Q. V. Le. *MnasNet: Platform-Aware Neural Architecture Search for Mobile.* CVPR 2019.

[11] C. White, A. Zela, B. Ru, Y. Liu, F. Hutter. *How Powerful are Performance Predictors in Neural Architecture Search?* NeurIPS 2021.

[12] J. Siems, L. Zimmer, A. Zela, J. Lukasik, M. Keuper, F. Hutter. *NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search.* arXiv:2008.09777, 2020.

[13] F. Hutter, H. H. Hoos, K. Leyton-Brown. *Sequential Model-Based Optimization for General Algorithm Configuration (SMAC).* LION 5, 2011.

[14] J. Snoek, H. Larochelle, R. P. Adams. *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS 2012.

[15] L. Grinsztajn, E. Oyallon, G. Varoquaux. *Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?* NeurIPS 2022 Datasets and Benchmarks Track.

[16] T. Duan, A. Anand, D. Y. Ding, K. K. Thai, S. Basu, A. Ng, A. Schuler. *NGBoost: Natural Gradient Boosting for Probabilistic Prediction.* ICML 2020.

[17] H. Cai, L. Zhu, S. Han. *ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.* ICLR 2019.

[18] B. Settles. *Active Learning Literature Survey.* Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009.

[19] S. Kaufman, P. Phothilimthana, Y. Zhou, C. Mendis, S. Roy, A. Sabne, M. Burrows. *A Learned Performance Model for Tensor Processing Units.* MLSys 2021.

[20] B. Lakshminarayanan, A. Pritzel, C. Blundell. *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles.* NeurIPS 2017.
