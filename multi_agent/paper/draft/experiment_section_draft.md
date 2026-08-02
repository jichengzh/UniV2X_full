# 实验章节草稿与实验空白方案

> 目标：本文实验章节预计只有两页左右，因此正文最多保留四个小节。其中实验配置小节只在最终论文中简短撰写，本草稿不展开；本文件重点梳理后续结果章节已有基础和需要补充的实验。

## 章节结构占位

### 4.1 实验配置

[正文待写，本文件暂不展开。最终论文中只需用较短篇幅说明模型、数据集、硬件、baseline、公平比较协议和指标定义。]

### 4.2 端到端有效性与消融实验

[正文待写。该小节承担实验章节的主结果：证明 SHCoSearch 相比 TensorRT、AutoTVM/TVM、compression-only、schedule-only 和 serial search 能得到更好的部署候选，并通过消融说明提升来自软硬件联合搜索机制。跨模型、跨硬件加速效果也建议合并到本节中。]

### 4.3 搜索空间、搜索预算与实测回流分析

[正文待写。该小节承担机制和敏感性分析：说明搜索空间分区如何降低复杂度，搜索预算和 Top-K 采样规模如何影响结果，以及真实硬件测量回流是否改善代理模型和候选排序。]

### 4.4 可选补充验证与方法边界

[正文待写或删除。如果篇幅不足，本节整体移至 Appendix 或 Discussion。可选内容包括 V2XVerse 闭环时延验证、CoDriving 阴性结果、buildability 日志和更完整的机制分析。]

## 实验空白方案

### 一、4.2 端到端有效性与消融实验

这个小节需要证明两个核心问题：第一，框架整体是否有效；第二，效果是否确实来自联合搜索，而不是来自某个已有后端、剪枝、量化或调度组件。

#### 已有基础

1. **Pyramid + H800 是当前最完整的主实验基础。**  
   已有 FP32、FP16、INT8 多个候选配置的真实 latency 数据，并且已经对 FP16 输入尺寸不一致、INT8 tensorization 等问题做过修正。当前 Figure 1(a) 已经能够说明通道宽度、精度和实测 latency 之间存在交叉和非单调关系，可以作为本文方法动机和主实验背景。

2. **P×S 消融已经具备较强证据。**  
   已有 A-joint、A-serial、A-noS 的 HV 对比和显著性结果，可用于证明“先压缩再调度”的串行流程会错过部分更优候选。该结果可以作为 4.2 中联合搜索消融的主证据之一。

3. **P×Q×S 消融已有 12 seed 结果，但正文使用前需要复核口径。**  
   当前已有 P×Q×S 的 12 seed ablation 结果，能够说明剪枝、量化和调度联合搜索优于串行搜索。但其中 INT8 latency 仍有 proxy 或阶段性口径限制。如果要作为正文主结果，需要补 full-backbone INT8/FP16 统一口径；否则更适合放入机制分析或 Appendix。

4. **V2XVerse 时延实验已经证明 latency 优化具有下游意义。**  
   Figure 1(b) 已经展示 perception latency 增大会导致 driving score 下降、碰撞或 timeout 风险上升。这个结果可以支撑为什么需要优化 latency，但不建议在 4.2 中重复展开。

5. **部分跨模型或跨硬件证据已经存在，但不够统一。**  
   当前 Pyramid/H800 最完整；CoDriving 有可分离或弱耦合证据；4090、Orin、H800 上也有不同阶段的测量记录。但这些结果是否同一输入尺寸、同一 batch、同一 full-backbone/full-model 范围，需要进一步核实。

#### 需要补充的实验

1. **最终 measured Pareto frontier。**  
   这是 4.2 最重要的实验空白。需要冻结一组最终候选，并确保所有进入正文 Pareto 图或表的点都满足：
   - latency 是目标硬件真实测量；
   - energy 使用统一口径；
   - AP 是真实 validation 或 finetune 后复核结果；
   - 不使用 surrogate AP 或 proxy latency 作为最终坐标。

   建议正文主图使用 AP70-latency Pareto，energy 用颜色、点大小或表格辅助表示。如果 energy 口径仍未完全稳定，不要强行声称完整三目标最优，只能说在 AP-latency 主轴上取得优势，并报告 energy 辅助结果。

2. **与 TensorRT、AutoTVM/TVM 和默认部署的公平对比。**  
   需要明确这些 baseline 到底比较什么：
   - TensorRT：作为部署后端 baseline，比较同一模型或同一候选的 best valid engine；
   - AutoTVM/TVM：作为 schedule tuning baseline，比较固定软件配置下的调度优化能力；
   - Default：原始模型或默认后端优化；
   - Serial：先选择剪枝/量化，再进行调度；
   - SHCoSearch：软件候选选择和硬件调度反馈联合进行。

   这个实验的关键不是 baseline 数量越多越好，而是每个 baseline 必须公平。至少要保证最终 AP、latency、energy 的测量口径一致。如果 TensorRT 和 TVM 不能完全统一后端，则正文中要明确说明这是 deployment framework baseline，而不是严格同一调度器内部消融。

3. **compression-only、schedule-only、serial 和 SHCoSearch 的消融表。**  
   需要在同一搜索空间和相近测量预算下比较：
   - compression-only：只搜索剪枝和量化，不使用内层调度反馈；
   - schedule-only：固定软件配置，只优化调度；
   - serial：先确定软件配置，再调度；
   - SHCoSearch：软件配置、调度和测量反馈共同影响候选排序。

   如果时间有限，正文中至少保留 A-joint、A-serial、A-noS 的 HV 表；compression-only 和 schedule-only 可以合并为简化 baseline 或移至 Appendix。

4. **跨模型和跨硬件加速表。**  
   用户希望覆盖 Pyramid、CoDriving、F-Cooper 等约 5 个模型，以及 H800、4090、Orin。建议不要把它拆成独立小节，而是合并到 4.2 中，用类似 ALT 7.2 的 compact end-to-end benchmark 表展示。

   需要补充的信息包括：
   - 每个模型的 baseline latency；
   - SHCoSearch 或本文优化后 latency；
   - AP 变化；
   - energy 变化；
   - speedup；
   - 使用的硬件和后端。

   如果完整的 5 模型 × 3 硬件矩阵来不及完成，正文可以先放 3--5 个 representative model-hardware pairs，完整矩阵放 Appendix。不能把不同口径的结果混成一个平均 speedup。

5. **代表性部署点。**  
   建议从最终 Pareto 中挑 2--3 个 operating points，分别对应：
   - high-accuracy setting；
   - balanced setting；
   - low-latency 或 low-energy setting。

   这些点可以帮助 reviewer 直观看到本文方法输出的不是单一最优，而是一组可部署折中方案。

#### 最低可接受版本

如果时间和篇幅都紧，4.2 至少需要完成：

- 一张主平台 measured Pareto 图；
- 一个 baseline 对比表；
- 一个 A-joint / A-serial / A-noS 消融表；
- 2--3 个代表性部署点。

跨模型和跨硬件若数据不完整，可以作为小表或 Appendix 支撑，不应挤占主结果。

### 二、4.3 搜索空间、搜索预算与实测回流分析

这个小节需要证明本文搜索机制不是简单工程堆叠，而是有必要的搜索空间约束、预算控制和真实测量反馈。

#### 已有基础

1. **搜索空间规模已有明确构造。**  
   当前主线使用 3-stage 粗分区和全局精度选择，软件空间规模为 `7^3 × 3 = 1029`。这可以说明本文不是在原始指数空间中盲目搜索，而是通过计算图分区和硬件约束形成可探索的候选空间。

2. **SMBO 实测回流已有直接证据。**  
   FP32 round 1--3 已经完成。已有结果显示，真实测量回流可以降低 latency surrogate 的预测误差。例如 round 1 中 latency MAPE 从 32.4% 降到 18.2%。同时，回流纠正了 surrogate 对窄宽度区域过度乐观的外推。

3. **已有“更小不一定更快”的机制证据。**  
   Figure 1(a) 和 SMBO 后续测量都说明，通道进一步变窄并不一定继续降低 latency，窄配置可能因为 GPU 利用率、kernel 触发或 launch overhead 而变慢。这可以作为搜索空间和实测回流必要性的机制解释。

4. **W_g/P_g 或通道对齐例子可以解释串行搜索为什么会失败。**  
   已有同 AP 下不同通道配置 latency 相差数倍的证据，能够说明单独看剪枝率、FLOPs 或默认调度结果不足以判断最终部署性能。

#### 需要补充的实验

1. **分区粒度敏感性。**  
   需要补一个轻量对照，说明为什么使用 3-stage 粗分区。该实验不需要非常复杂，建议只比较 2--3 个粒度：
   - whole-model/global width：空间最小，但表达能力弱；
   - 3-stage partition：本文默认设置；
   - finer partition：空间显著增大，若没有明显收益则说明默认设置合理。

   至少需要报告每种设置的搜索空间规模、测量成本和最终 best Pareto/HV 或 best latency。若 finer partition 未完成，不要声称 3-stage 无损，只能说它在当前实验中提供了有效的成本-性能折中。

2. **搜索预算或 Top-K 采样敏感性。**  
   参考 ALT 7.3.2 Parameter Sensitivity，补少量预算点即可。建议选择：
   - small Top-K；
   - default Top-K；
   - large Top-K。

   每个设置报告 measured candidates 数量、best latency/HV、是否命中相近 Pareto。重点不是证明 default 永远最优，而是说明过小预算可能漏掉候选，过大预算收益有限，本文设置是合理折中。

3. **实测回流有效性整理成表。**  
   已有结果需要整理成适合正文的小表，例如：
   - round；
   - 新测候选数量；
   - MAPE before；
   - MAPE after；
   - best latency 或前沿是否推进；
   - 主要结论。

   正文中不要过度强调“每轮都刷新最优”，因为已有结果表明多轮搜索主要贡献是校准模型，而不是持续刷新 latency 最优。更准确的写法是：实测回流修正了代理模型对窄宽度区域的错误排序，使搜索避免继续浪费测量预算。

4. **机制例子压缩。**  
   如果 4.3 还有空间，可以加入一个 2--4 行的小机制表，展示同 AP 或相近 AP 下两个候选因为通道对齐、kernel 触发或低精度执行路径不同而 latency 明显不同。

   如果篇幅不足，这部分只引用 Figure 1(a)，完整 W_g/P_g 表和 buildability 日志放 Appendix。

#### 最低可接受版本

4.3 至少需要完成：

- 一个搜索空间规模对照或说明；
- 一个预算/Top-K 敏感性小结果；
- 一个实测回流 MAPE before/after 表。

机制例子可选，但建议保留一句话解释，否则 4.3 容易变成纯参数调节而不是方法机制分析。

### 三、4.4 可选补充验证与边界

这个小节不是必须。如果两页篇幅不足，应优先保留 4.2 和 4.3，把 4.4 移到 Appendix 或 Discussion。

#### 已有基础

1. **V2XVerse 闭环时延实验已经完成较多。**  
   现有结果显示 perception latency 增大会降低 driving score，并提高 collision 或 timeout 风险。该证据已经被用于 Introduction Figure 1(b)。

2. **AP×latency 的闭环安全网格已有部分结果。**  
   已有 DS、collision rate、route completion 等结果，但 composed DS 与 AP 之间存在非单调和噪声，不能直接支持所有强结论。

3. **CoDriving 阴性结果可以作为方法边界。**  
   CoDriving 在某些低维 P×Q×S 空间中表现为可分离或弱耦合。这说明本文框架不应宣称所有模型都必然强耦合，而应表述为能够识别并利用模型-硬件相关的耦合。

#### 需要补充的实验

1. **闭环时延结果是否进入正文。**  
   如果 4.4 保留，只需要 2--3 句话说明 latency 对 V2X 闭环驾驶有实际影响。完整 DS/collision 网格不建议放正文。

2. **AP×latency 安全网格是否作为附录。**  
   如果要支持“AP 退化会改变安全风险”这一主张，需要补齐 cliff band 中缺失的 AP/latency cells，并明确使用 collision rate 而不是 composed DS 作为主要安全指标。否则只作为附录探索性结果。

3. **CoDriving 阴性结果是否写入边界。**  
   这部分可以帮助论文更诚实，但也会占篇幅。建议正文不展开，只在 Discussion 或 Appendix 中说明：不同模型的耦合强度不同，SHCoSearch 的价值在于用统一框架搜索和验证这种耦合，而不是预设所有模型都强耦合。

#### 最低可接受版本

如果只剩很少篇幅，4.4 可以完全删除。闭环时延证据已经在 Introduction Figure 1(b) 中发挥作用，实验正文不必重复。

## 当前最需要优先补齐的实验

1. **主平台最终 measured Pareto。**  
   这是整篇实验的核心。如果没有真实 AP、latency、energy 坐标，4.2 很难成立。

2. **公平 baseline 协议。**  
   必须明确 TensorRT、AutoTVM/TVM、compression-only、schedule-only、serial 和 SHCoSearch 分别如何获得候选，以及最终如何测量。

3. **联合搜索消融表。**  
   至少需要 A-joint / A-serial / A-noS；如果 P×Q×S full-backbone 口径来不及，就先用 P×S 作为正文主消融，P×Q×S 放 Appendix 或标注为机制证据。

4. **跨模型跨硬件 compact 表。**  
   如果要声称方法具备通用性，需要至少准备若干 representative model-hardware pairs。没有统一口径的结果不能混算平均值。

5. **搜索预算和实测回流表。**  
   这部分已有较好基础，整理成本低，且能直接支撑 Method 中的 measurement feedback 设计。
