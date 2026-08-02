# SHCoS Abstract 撰写计划

## 1. 写作目标

- **目标会议**：AAAI 主会。
- **摘要形式**：单段、无引用、无公式、无小标题。
- **目标篇幅**：**约 200--220 个英文词，11 个句子**；词数用于控制节奏，不作为强制约束。
- **结构主线**：通用感知部署需求 -> 软硬件耦合缺口 -> V2X 时延动机 -> SHCoS 总体方案 -> 三项核心设计 -> 通用能力结论。
- **一句话核心论点**：通用感知模型的边缘部署需要联合建模软件压缩与硬件调度；SHCoS 通过受约束的搜索空间、硬件探针增强的外层搜索和条件化内层调度，在 AP、延迟与能耗之间寻找可部署的 Pareto 候选，V2X 协同感知则作为代表性的时延敏感验证场景。

当前版本约 213 词。后续修改以论证完整性和句间衔接为优先；已经确认的句子不为满足精确词数而连带压缩，只有明显冗余时才进一步删减。

## 2. AAAI-26 优秀论文摘要观察

AAAI 官方公布的 5 篇 AAAI-26 Outstanding Papers 摘要具有以下统计特征：

| 论文 | 词数 | 句数 | 主要结构 |
|---|---:|---:|---|
| Model Change for Description Logic Concepts | 109 | 4 | 问题定义 -> 概念划分 -> 理论主张 -> 结果 |
| Causal Structure Learning for Dynamical Systems | 161 | 7 | 背景 -> 双重缺口 -> 方法 -> 技术依据 -> 实验 |
| ReconVLA | 163 | 8 | 背景 -> 经验问题 -> 方法 -> 机制 -> 数据 -> 实验 |
| High-Pass Matters | 138 | 7 | 背景 -> 缺口 -> 理论洞察 -> 方法 -> 实验 |
| LLM2CLIP | 172 | 6 | 背景 -> 研究问题 -> 框架 -> 实现 -> 多任务结果 |

其共同特点不是固定句式，而是较稳定的信息顺序：前 1--2 句迅速建立问题和缺口，在第 3--4 句完成方法命名，随后只解释决定方法成立的核心设计，最后用实验结论收束。它们通常不在摘要中展开相关工作，也不逐项复述 Introduction 的贡献列表。

参考来源：[AAAI-26 Outstanding Paper Awards](https://aaai.org/about-aaai/aaai-awards/aaai-conference-paper-awards-and-recognition/)、[ReconVLA](https://ojs.aaai.org/index.php/AAAI/article/view/38921)、[LLM2CLIP](https://ojs.aaai.org/index.php/AAAI/article/view/37427)、[High-Pass Matters](https://ojs.aaai.org/index.php/AAAI/article/view/39469)、[CaDyT](https://ojs.aaai.org/index.php/AAAI/article/view/40999)、[Model Change](https://ojs.aaai.org/index.php/AAAI/article/view/39008)。

## 3. 十一句话的参考词数分配

| 句子 | 词数 | 句子任务 | 内容大纲 | 写作约束 |
|---|---:|---|---|---|
| S1 | 19 | 场景扩展 | 说明智能感知应用正向车端、路侧等边缘场景拓展，高效部署因此日益重要。 | 以通用感知应用为主，并通过车端和路侧保留 V2X 场景联系。 |
| S2 | 21 | 资源矛盾 | 说明复杂感知网络提升性能的同时，加剧计算需求与有限边缘资源之间的矛盾。 | 从应用趋势自然收敛到需要解决的部署问题。 |
| S3 | 14 | 优化需求 | 指出解决上述矛盾需要统一优化软件侧模型压缩与硬件侧执行调度。 | 用 `Addressing this conflict` 明确承接 S2。 |
| S4 | 26 | 现有缺口 | 指出现有研究割裂处理压缩与调度，忽视压缩引起的执行变化及后端对部署收益的制约。 | 解释为什么统一优化仍未被现有方法充分解决。 |
| S5 | 17 | 框架定位 | 提出 SHCoS，并将其定义为面向模型压缩与执行调度的软件--硬件协同优化框架。 | 直接突出协同优化主旨，避免重复叠加 hardware-guided 和 multi-objective。 |
| S6 | 11 | 两模块总述 | 说明框架由设计空间构建和多目标探索两个模块组成。 | 使用不同模块名称，避免连续重复 `search-space`。 |
| S7 | 22 | 搜索空间构造 | 说明计算图依赖分区与后端能力刻画如何形成满足结构和部署约束的紧凑压缩空间。 | 不在摘要中展开底层配置和公式。 |
| S8 | 29 | 双层探索架构 | 探索模块引入双层探索架构（BEA），通过将压缩搜索和调度细化分别组织在外层与内层，避免展平庞大的软硬件联合空间。 | 按“模块引入架构 -> 分层组织方式 -> 避免空间膨胀”的顺序表达。 |
| S9 | 14 | 多指标代价模型 | 为降低评估成本，引入硬件探针增强的多指标代价模型（HP-MCMs）预测 AP、延迟与能耗。 | 缩写与正文的硬件探针特征和多指标模型直接对应。 |
| S10 | 13 | 搜索策略 | 基于 HP-MCMs 的预测，代理辅助 NSGA-II 排序候选并探索 Pareto 前沿。 | 使用标准算法名称，不另造搜索策略缩写。 |
| S11 | 27 | 通用能力结论 | 概括框架寻找硬件自适应 Pareto 配置并提供精度、延迟和能耗折中的能力。 | 不列具体模型、后端或数值，但表述范围必须由最终实验支撑。 |
| **合计** | **约 213** |  |  |  |

## 4. 建议的信息推进

摘要可以划分为四个连续语义块，但排版上保持一个自然段：

1. **S1--S4：Why**。从通用感知模型的边缘部署需求出发，依次引出优化手段、软硬件耦合缺口和 V2X 场景中的时延重要性。
2. **S5--S6：What**。先定义 SHCoS 的软硬件协同优化对象，再单独交代两个主要模块。
3. **S7--S10：How**。依次说明空间构建、双层探索架构、硬件探针增强的多指标代价模型和代理辅助搜索策略。
4. **S11：Capability and scope**。用能力型结论收束全文，不列具体实验对象，但将适用范围限制在最终完成验证的感知工作负载与边缘平台内。

## 5. 结论句证据要求

结论句不列出具体模型、后端和数值，但以下表述仍需实验支撑：

| 摘要表述 | 最低证据要求 |
|---|---|
| `hardware-adaptive Pareto configurations` | 不同目标硬件上的候选或执行配置存在可验证差异 |
| `diverse perception workloads` | 多个感知模型或网络结构使用统一协议完成评测 |
| `heterogeneous edge platforms` | 多类设备或后端使用一致口径完成真实测量 |
| `accuracy, latency, and energy trade-offs` | 三个指标均来自最终候选的统一实测坐标 |

若跨模型、跨硬件或能耗实验未闭合，应相应收紧 `diverse`、`heterogeneous` 或三指标折中的表述，而不是依赖 proxy 结果扩大方法范围。

## 6. 术语与表述锁定

| 概念 | 摘要中的统一写法 | 备注 |
|---|---|---|
| 方法名 | `SHCoS` | 摘要采用该名称；Introduction 和 Method 中仍存在 `SHCoSearch`，后续需全文统一。 |
| 任务 | `edge perception deployment` | 作为方法的主要适用问题。 |
| 代表场景 | `latency-sensitive applications such as V2X cooperative perception` | 用于说明时延重要性，不作为方法专用边界。 |
| 联合优化 | `software--hardware co-search` | 避免与较宽泛的 `co-optimization` 来回切换。 |
| 软件变量 | `pruning and quantization configurations` | 摘要不区分 pruning ratio 与 channel width。 |
| 硬件变量 | `hardware schedules` | 不枚举 tiling、layout 和 vectorization。 |
| 多目标指标 | `AP, latency, and energy` | AP 若目标读者可能不熟悉，可在正文首次定义为 average precision。 |
| 输出 | `Pareto deployment candidates` | 避免 `optimal solution`，因为输出不是单一加权最优点。 |

## 7. 删除规则与最终检查

- 不使用引用；AAAI LaTeX 模板明确要求摘要中不包含 references。
- 不复述 Figure 1 的 60 个候选或 V2XVerse 时延曲线，这些证据属于 Introduction。
- 不写公式、搜索空间数量级、损失函数或调度旋钮。
- 不使用 `novel`, `first`, `comprehensive`, `significantly`，除非有明确范围和统计证据。
- 每句只承担上表中的一个主要任务，避免用多个 `and` 串联方法与结果。
- 结论句可以不列具体数值，但其中每个范围词都必须与实验章节一致。
- 最终检查顺序：术语一致 -> 逻辑与衔接 -> 结论范围与实验一致 -> 无引用 -> 无未定义缩写 -> 总篇幅不过长。

## 8. 推荐撰写顺序

先写 S11，锁定论文能够支持的通用能力和适用范围；再写 S5--S10，使方法描述只服务于该能力；最后写 S1--S4，并检查通用感知背景与车端、路侧场景之间是否自然衔接。摘要完成后，应从后向前检查每个范围词是否能在实验章节找到对应证据。
