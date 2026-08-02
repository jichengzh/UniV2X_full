# Background and Method AAAI-Style Review

以下审查仅基于 `method_zh_ieee_v1.md` 文本，不评价实验、结果、伦理、可复现性或整篇接收概率。本文档已根据作者对 Method main weaknesses 的逐条修改与回复进行更新，当前判断反映修订后版本的主要风险。

## 1. Background Review

### Overall assessment

Background 有明确的问题意识：模型压缩与硬件调度不能串行独立优化，实际部署性能取决于二者耦合。但目前更像 Method section 中的“动机与系统概述”，而不是完整 AAAI 论文 Background。它能提出现象和工程动机，但对 AI 方法层面的抽象问题、现有方法的系统性不足、以及本文方法为何必要的理论铺垫仍不够。

### Main strengths

- 通过 roofline 视角把推理速度拆成计算负载与硬件有效执行能力，问题入口清楚。
- 表格案例有效说明“低 FLOPs 不等于低 latency”“全 INT8 不一定优于混合 INT8”，这是有说服力的动机。
- 能自然引出软硬件联合搜索，而不是串行剪枝、量化、编译。
- 背景与 Method 中的剪枝、量化、调度、Pareto 搜索存在直接联系。

### Main weaknesses

- Background 过于贴近项目实现，缺少对通用 AI 问题的抽象，例如“硬件条件化模型压缩搜索”“多目标代理辅助搜索”“结构约束下的联合离散优化”等。
- 现有方法不足说得较概括，只分为软件压缩与硬件编译两类，但没有明确指出 NAS、NetAdapt、AutoTVM、硬件感知剪枝/量化等相关范式各自失败在哪里。
- gap 目前是经验性现象，而不是形式化研究 gap。文本说明了“两个维度耦合”，但未清楚说明现有方法为何不能建模这种耦合。
- 表格没有编号、来源、实验设置或说明，作为 Background 证据略显突兀。若这是正文表 1，需要给出简洁说明；若只是动机案例，应压缩。
- V2X 感知出现较晚，缺少说明该任务相比一般视觉模型部署有什么结构性特殊性，例如多分支、多模态、稠密 BEV/协同融合算子是否导致搜索空间更复杂。文中未说明。

### Suggested revision

- 将 Background 压缩为 3 段：
  - 研究问题：协同式 V2X 感知模型在边缘硬件部署中需要同时优化 AP、latency、energy。
  - 技术挑战：剪枝、量化、调度之间存在条件依赖，FLOPs/bit-width 不是可靠代理。
  - 方法动机：需要一个结构约束、硬件条件化、多目标代理辅助的联合搜索框架。
- 明确 gap：现有硬件感知压缩通常把硬件反馈作为静态查表或单指标约束；现有编译调优通常在固定模型结构下优化调度；二者都不能有效处理“软件配置改变合法调度空间”的条件化搜索问题。
- 把表格从“项目化发现”改写为“motivating observation”，并解释其抽象含义：channel alignment、kernel availability、mixed precision support 使得软件配置与后端执行路径非单调耦合。

## 2. Method Review

### Overall assessment

Method 已经具备较完整的形式化框架：定义了软件空间、调度空间、联合候选、多目标 Pareto 目标、硬件约束、图分区、代价模型和闭环搜索。相较初稿，当前版本已经明显强化了“条件化联合搜索”的主线：外层生成剪枝与量化候选，候选排序依赖其诱导的调度空间和内层硬件反馈，而不是默认部署指标。附录伪代码、符号/指标列表、AP 残差预测依据和方法边界也在一定程度上缓解了严谨性问题。当前剩余主要风险不再是流程缺失，而是需要进一步把“约束条件化双层协同搜索机制”作为本文自己的方法贡献在 Introduction 与 Method 中持续凸显。

### Main strengths

- 输入和输出大体清楚：输入为部署模型和目标硬件，输出为满足场景约束的 Pareto 候选集合。
- 联合空间定义较完整：$x=(p,q)$、$c=(x,s)$、$\Theta_{\mathrm{joint}}$、$\Sigma_{\mathrm{sch}}(x)$ 等符号形成了方法骨架。
- 多目标优化目标与论文目标一致，覆盖 AP、latency、energy。
- “条件化联合搜索”是目前最有潜力的方法贡献：软件配置诱导调度空间，外层搜索不只看默认调度指标，而看内层调度反馈。
- 方法边界写得较严谨，明确代价模型只是排序信号，最终 Pareto 需要真实测量。

### Main weaknesses

- 完整算法伪代码已放入附录 C，正文 3.3.1 也说明了外层候选生成、进入内层调度、真实测量回流和代价模型更新的流程。当前问题降级为：正文需要用一两句话更明确地指向附录算法，并说明正文只保留主机制，细节由附录承载。
- 关键变量定义已有改进。$p,q,s,\rho,K,\mathcal{B}_{\rho},\psi_{\mathrm{hw}},\psi_{\mathrm{sw}},\Omega_k$ 等在正文或附录 D 中得到补充。当前仍需注意的是：正文中首次出现的符号不宜过多依赖附录，特别是 $\phi_{\mathrm{hp}}(x)$、$g_\rho(c)$ 和 $\mathcal{B}_{\rho}$ 应保留最低限度的解释。
- 新贡献与已有技术边界仍是最主要风险。当前正文已经体现“条件化联合搜索”，但尚未在 Method 中单独命名本文的新机制；LightGBM、NSGA-II、AutoTVM、XGBoost、NetAdapt、DepGraph 的角色仍需要被明确写成“实例化组件”，而不是贡献主体。
- 模块之间“串联感”已有缓解。3.3.1 已补充内层真实评测结果回流外层、影响后续排序与采样；3.3.4 也说明新增观测用于更新外层代价模型。当前不建议在 3.3.1 中引入过多未定义接口符号，但可在现有自然语言基础上继续强化“反馈改变候选排序”的机制句。
- 公式删减方向合理。普通数据回流记账式公式不必保留在正文，Pareto 目标、结构/硬件过滤集合、rank loss 和 AP 残差公式更值得保留。
- AP 预测风险已有缓解。当前版本用已有压缩/量化研究和本文预实验共同支撑“AP 波动相对受限”的观察，并明确最终 Pareto 候选仍需真实 AP 评测。剩余风险是该预实验观察最好在实验或附录中给出简短证据。
- 通用性问题已有澄清。正文方法主要面向 V2X 协同感知模型，并未明显绑定 Pyramid 或 CoDriving；TVM/AutoTVM 应表述为内层调度器的一种实例化，避免写成方法依赖特定后端机制。TensorRT 若已不属于当前框架，应避免出现在正文或核心附录叙事中。

### Suggested revision

- 在 Introduction 的 contribution/overview 中完整说明贡献边界：本文不重新设计单独剪枝、量化、进化优化或调度算法，而是提出约束条件化的双层协同搜索机制。
- Method 中不宜重复大段“不是做什么”的贡献声明，只需在 3.3.1 用一句机制定位承接：外层在结构与硬件约束过滤后的软件空间中搜索，内层在给定软件配置后实例化条件调度空间，并将真实硬件反馈回流至外层代价模型。
- 保持 3.3.1 第二段的流程阐述，不强行加入 $\mathcal{D}_t$、$z_i$ 等提前出现的符号；通过“回流更新代价模型，并影响后续软件候选的排序与采样”这类自然语言即可回应模块接口问题。
- 在 3.3.2/3.3.3/3.3.4 中把已有技术写成实例化选择：LightGBM 是外层多指标代价模型的实例化，NSGA-II 是 Pareto 候选生成器的实例化，AutoTVM 风格策略是条件调度空间求解器的实例化。

## 3. Background-Method Logic Check

| Background 中提出的问题/挑战 | Method 中对应的技术设计 | 是否对应充分 | 修改建议 |
|---|---|---:|---|
| FLOPs 降低不必然降低 latency | 硬件能力刻画、微基准探针、$\psi_{\mathrm{hw}}$ 过滤 | 部分充分 | 需要明确 $\psi_{\mathrm{hw}}$ 如何捕捉 channel alignment、kernel trigger 等非单调条件 |
| 剪枝、量化会改变硬件执行路径 | 条件调度空间 $\Sigma_{\mathrm{sch}}(x)$，候选 $c=(x,s)$ | 较充分 | 建议将其提升为核心方法定义，并形式化为硬件条件化评价 |
| 硬件调度收益不能脱离模型形状判断 | 内层 AutoTVM 风格调度搜索；真实测量回流外层观测集合 | 较充分 | 当前已说明内层返回真实评测结果并影响后续排序；建议避免在内层段落重复外层交互 |
| 需要同时优化 AP、latency、energy | Pareto 多目标优化，NSGA-II | 充分 | 建议补充约束处理方式，如不可行候选如何惩罚或过滤 |
| 搜索空间过大，不能直接枚举 | 图分区、硬件过滤、上层搜索分区 $\mathcal{G}'$ | 较充分 | 需要说明分区合并准则，“语义位置相近且具有相同硬件约束”仍偏模糊 |
| 协同式 V2X 模型包含复杂算子与模块 | 计算图扫描与依赖分区 | 部分充分 | Background 应提前说明这些结构为何导致普通剪枝/量化搜索失效 |
| 真实硬件测量昂贵 | LightGBM 代价模型、证据回流、预测 Pareto 前沿邻域筛选 | 较充分 | 需要继续突出与普通 surrogate search 的区别：硬件探针增强特征和条件化调度反馈会改变软件候选排序 |

## 4. AAAI Method Contribution Diagnosis

当前最接近类别：**B / C 之间**。更准确地说，修订后文本已经从单纯的 **C. 面向特定任务的系统框架** 向 **B. 对已有 AI 方法的有效改进** 靠近，但尚未稳定达到 A 类“明确新颖 AI 方法”。其核心潜力在于硬件反馈驱动的条件化双层 Pareto 搜索，而不是单个剪枝、量化、代理模型或调度算法。

### 判断依据

- 方法主体仍使用已有组件：结构化剪枝、训练后量化、DepGraph、AutoTVM/XGBoost、LightGBM、NSGA-II、Pareto 优化。
- 修订后文本已经更清楚地说明外层软件候选依赖内层硬件调度反馈进行排序，而不是只依据默认调度或静态软件复杂度。
- 附录 C 的伪代码和附录 D 的符号/指标列表增强了可执行性，但正文中的新机制命名和贡献边界仍需进一步锐化。

### 本文真正的新方法是什么

当前文本中真正的新方法应被概括为：面向 V2X 协同感知模型的约束条件化双层多目标搜索机制，其中软件压缩配置的评价由其诱导的硬件调度空间和真实硬件反馈共同决定，而不是由默认模型复杂度、静态 FLOPs 或孤立的软件指标决定。

### 新颖点是否能用一句话概括

可以，但需要更锐化：

> 本文提出一种约束条件化的双层 Pareto 搜索方法，将剪枝/量化候选的评价从静态软件指标提升为经目标硬件调度反馈校正后的多目标部署表现。

### 新颖性是否足以支撑 AAAI 方法贡献

相比初稿已有明显增强，但仍需进一步压实。若 Introduction 明确贡献边界，Method 明确命名“约束条件化双层搜索”，并用附录 C 支撑完整算法流程，则可以从系统框架提升为较清楚的方法改进。当前最需要避免的是让已有组件名称淹没本文自己的机制。

### 是否只是组合已有技术

仍有被认为是组合剪枝、量化、搜索、编译、代理模型和 Pareto 优化的风险，但风险已降低。关键在于持续强调组合后产生的机制：软件候选不是被静态评价，而是经条件调度空间和真实硬件反馈重新排序。

### 如果是组合，核心机制在哪里

- 软件候选 $x$ 不直接评价，而通过 $\Sigma_{\mathrm{sch}}(x)$ 条件化获得硬件响应。
- 外层搜索空间不展开硬件调度维度，而通过有限内层探针估计软件配置的可达部署表现。
- 硬件能力与图依赖共同压缩合法搜索空间，而不是搜索后再过滤无效候选。

### 如何强化为更清楚的 AAAI 方法贡献

- 在 Introduction 中给方法命名，例如“约束条件化双层协同搜索”或“硬件反馈驱动的条件化双层 Pareto 搜索”。
- 在 Method 3.3.1 中保持简洁流程，不必提前引入过多接口符号，但必须保留“内层反馈影响后续排序与采样”的机制句。
- 在 3.3.2 中说明 $\phi_{\mathrm{hp}}(x)$ 来自硬件探针和内层反馈，而不只是普通软件特征。
- 明确相比硬件感知 NAS、NetAdapt、AutoTVM 的本质区别：本文不是固定模型调度，也不是固定硬件查表压缩，而是软件结构与调度空间互相条件化。

## 5. Formalization and Module Review

### Missing definitions

- $p,q$ 已在正文简化定义为剪枝率/通道宽度与量化位宽，当前篇幅下可以接受。
- $s$ 已说明包含 tile、layout、vectorization 等后端优化旋钮，正文粒度基本够用；更细的调度参数可留在附录。
- $\rho$ 可保持为部署场景编号或场景索引，但正文应说明其至少包含目标硬件和部署预算约束。
- $g_\rho(c)$ 与 $\mathcal{B}_\rho$ 仍偏抽象，建议正文保留一句说明 $\mathcal{B}_{\rho}$ 可包括 latency、energy、内存、后端支持能力等约束。
- $\phi_{\mathrm{hp}}(x)$ 是仍需最小补充的关键特征，建议说明其由软件配置编码、硬件能力过滤结果和少量调度探针反馈组成；详细指标列表可放在附录 D。
- $\Omega_k$、$P_k^{\mathrm{hw}}$、$P_k$ 的候选粒度当前已基本可读，若篇幅允许，可在附录 D 中补充示例。
- $x_{\mathrm{ref}}$ 已补充为同一模型和评测协议下的已测参考软件配置，当前够用。
- 内层调度搜索的完整预算、返回值、失败处理可由附录 C/D 承载，正文不必全部展开。
- NSGA-II 的种群大小、交叉变异算子、终止条件不必进入正文，除非它们构成本文方法贡献。

### Ambiguous symbols

- $\mathcal{B}_1,\mathcal{B}_2$ 与 $\mathcal{B}_\rho$ 符号相近，容易混淆。前者是分区，后者是约束预算；若有空间，建议将部署预算改为 $\mathcal{C}_\rho$ 或 $\mathcal{Q}_\rho$。
- 多指标符号歧义已基本修正：通用指标使用 $r\in\{\mathrm{AP},\ell,e\}$，延迟/能耗损失使用 $a\in\{\ell,e\}$，从而避免 AP 被误读为进入对数值预测目标。
- $\hat{u}_i^y$ 和 $\hat{r}^y(x)$ 是两个模型还是同一 LightGBM 的两个输出，文中未说明。
- $\widehat{m}^y(x)$ 与 $\widehat{\mathbf{m}}(x)$ 的关系需要统一。
- 需继续检查正文与附录符号是否一致，特别是 $\psi_{\mathrm{hw}}$、$\psi_{\mathrm{sw}}$、$\Omega_k$、$K$ 等符号。

### Weak or unnecessary formulas

- 普通数据回流记账式公式不建议放在正文；用自然语言描述证据回流即可。
- LightGBM 加法树公式是标准模型定义，对创新贡献有限；若篇幅紧张，可压缩为模型实例化与特征定义。
- AP 残差公式有用，当前已补充锚点解释和最终真实 AP 复核，风险已降低。
- Pareto 目标公式应保留，这是全文核心形式化之一。

### Unclear modules

- “搜索空间构建模块”和“搜索空间探索模块”清楚，但子模块间接口不够清楚。
- “硬件能力刻画”到底输出能力描述、过滤谓词、LUT、探针特征，还是都输出，正文未分清。
- “计算图扫描与分区”中 $\mathcal{G}$ 到 $\mathcal{G}'$ 的合并规则是关键，但目前只是自然语言说明。
- “多指标代价模型”与“候选方案生成”之间仍可更明确说明：只有预测 Pareto 前沿邻域候选进入内层调度与真实测量。
- “冷启动”标题下目前主要写证据回流，冷启动本身展开不多；若不详细讨论历史观测迁移，可考虑将小节标题改为“证据回流”。

### Recommended improvements

- 保留附录 D 的符号/指标列表，正文只保留必要定义，避免符号前置过多。
- 保留附录 C 的算法伪代码，并在正文 3.3.1 明确引用。
- 将 $\phi_{\mathrm{hp}}(x)$ 的构造做最小补充；$\psi_{\mathrm{hw}}$、$\psi_{\mathrm{sw}}$ 当前正文解释基本够用。
- 统一正文与附录符号，避免审稿人认为方法定义不稳定。
- 若篇幅受限，压缩 LightGBM/NSGA-II/AutoTVM 的标准介绍，把篇幅让给本文独有的条件化搜索机制。

## 6. Writing and Narrative Review

- 存在一定项目化语言，例如“系统概览”“框架有效性依赖”“硬件在环评估”“能力描述文件”等。这些适合 Implementation Details，不宜占据 Method 主体太多空间。
- Background 中表格很有用，但现在放置突然，且 Markdown 表头异常，正式论文中需要规范表号、caption 和解释。
- 3.2 和 3.3 段落偏长，层级虽清楚，但每个小节内部缺少“定义-机制-作用”的结构。
- 术语需要统一：软硬件协同搜索、联合搜索、条件化联合搜索、双层探索、SA-MOEA、SMBO、NSGA-II 等术语同时出现，容易造成方法主线分散。
- 附录 B 若仍保留 DepGraph/DeepGraph 详细原理，应避免重复解释已有方法；正文只需说明其作为依赖分析工具的接口。
- TVM/AutoTVM 应表述为内层调度器实例化，Pyramid/CoDriving 应主要出现在实验对象或实现细节中。正文当前并未明显绑定具体模型，这一点较初稿风险降低。
- AP 残差预测段落已加入已有研究与预实验依据，写作上仍建议避免过强表述，例如“不会引起 AP 的大幅波动”可改为“通常仅产生有限 AP 波动”。

## 7. Page-Limit Compression Advice

| 内容位置/段落 | 当前作用 | 处理方式 | 理由 |
|---|---|---|---|
| Background 3.1.1 第 1 段 | 从 roofline 引出计算负载与硬件执行能力 | 保留但压缩 | 是动机入口，但可更短 |
| Background 表格 | 展示非单调部署现象 | 保留但规范化 | 证据有力，但需要 caption 和简洁解释 |
| Background 3.1.1 表后长段 | 说明剪枝/量化/调度耦合 | 保留 | 是本文核心 gap |
| 3.1.2 框架概述 | 总览流程 | 压缩 | 与 3.2/3.3 重复，可改成 1 段贡献概述 |
| 3.2 搜索空间构建开头 | 定义 $x,c,\Theta_{\mathrm{joint}}$ | 保留 | 核心问题建模 |
| Pareto 目标公式 | 定义优化目标 | 保留 | 必须保留 |
| 3.2.3 硬件能力刻画 | 构建硬件约束 | 保留但压缩 | 保留抽象机制，移走实现细节 |
| 3.2.4 计算图扫描与分区 | 结构合法性约束 | 保留 | 对剪枝/量化合法性重要 |
| $\mathcal{G}\to\mathcal{G}'$ 合并说明 | 降低搜索维度 | 强化而非压缩 | 目前是关键但不够清楚 |
| 3.3.1 双层探索架构 | 内外层机制 | 保留并轻量强化 | 是方法贡献核心；正文保留流程，完整伪代码放附录 C |
| 3.3.2 LightGBM 公式 | 代理模型定义 | 压缩 | 标准模型公式贡献有限 |
| Rank loss | 候选排序机制 | 保留 | 比普通回归更有方法意义 |
| AP 残差预测 | 降低 AP 评估噪声 | 保留但补定义 | 有用，但需说明锚点选择 |
| 3.3.3 NSGA-II | 多目标候选生成 | 压缩 | 标准算法，保留作用即可 |
| 3.3.4 证据回流 | 闭环更新 | 保留但压缩 | 机制重要，避免重复 3.3.1 的流程描述 |
| 3.4 方法边界 | 限定外推范围 | 压缩保留 | 严谨性好，但正文可短 |
| 附录 A | 硬件能力实现流程 | 保留在附录并压缩工程细节 | 作为可复现补充可以保留，不应回流正文 |
| 附录 B | DepGraph/DeepGraph 原理 | 保留接口说明，压缩已有方法原理 | 不是本文核心新方法 |

### Priority 1 - Keep

- 核心问题定义：软硬件配置条件依赖导致静态 FLOPs/bit-width 不可靠。
- 联合空间：$x=(p,q)$、$s\in\Sigma_{\mathrm{sch}}(x)$、$c=(x,s)$。
- Pareto 多目标公式。
- 硬件过滤与图分区约束。
- 双层搜索机制。
- 外层代理模型如何使用内层硬件反馈。
- 与已有方法的本质区别。
- 附录 C 中的完整算法流程。

### Priority 2 - Compress

- roofline 背景。
- 表格案例解释。
- LightGBM 标准公式。
- NSGA-II 标准介绍。
- AP 残差预测动机和文献/预实验依据。
- 方法边界说明。

### Priority 3 - Move to Appendix / Implementation Details

- TensorRT/TVM 构建流程。
- CUDA-event 延迟、engine size、backend version。
- Pyramid、CoDriving 探针细节。
- DeepGraph 依赖传播完整推导。
- 硬件能力描述文件 schema。

### Priority 4 - Remove or Greatly Weaken

- 宣传式“统一框架”“提升可靠性”等泛化表述，除非紧跟机制解释。
- 对已有方法的常识性介绍。
- 与核心算法无关的部署流程细节。
- 重复说明“真实测量为准”的段落。

### 简短建议

- Background 建议压缩成 3 段：问题背景、技术 gap、本文方法动机。
- Method 优先保留：问题形式化、搜索空间构建、硬件/结构约束、双层条件化搜索、代理模型训练与更新。
- 可合并模块：3.3.3 和 3.3.4 可合并为“Candidate Selection and Feedback Update”。
- 必须保留公式：联合空间定义、Pareto 目标、结构/硬件过滤集合、rank loss 或候选排序准则。
- 伪代码已放入附录 C；正文需明确引用，不必在正文重复展开。
- 有限篇幅内，应把“条件化硬件反馈如何改变软件候选评价”作为 AAAI 方法贡献主线；已有组件只作为实例化方式出现。

## 8. Most Critical Problems

### Problem 1

**Problem:** 核心新方法仍需更明确命名，仍有被认为是已有技术组合的风险。  
**Why it matters for AAAI:** AAAI 主会需要明确方法贡献，而不只是系统集成。  
**Evidence from the text:** 文中使用 DepGraph、AutoTVM、XGBoost、LightGBM、NSGA-II、NetAdapt 等已有技术；虽然 3.2 和 3.3.1 已经体现条件化双层搜索，但方法名称和贡献边界仍不够突出。  
**How to revise:** 在 Introduction 中明确命名“约束条件化双层协同搜索”或“硬件反馈驱动的条件化双层 Pareto 搜索”；Method 中把已有技术写成实例化组件。  
**Severity:** Critical

### Problem 2

**Problem:** 少数关键接口变量仍需最低限度解释。  
**Why it matters for AAAI:** 方法严谨性依赖清楚的问题建模，变量不清会削弱可信度。  
**Evidence from the text:** $p,q,s,\psi_{\mathrm{hw}},\psi_{\mathrm{sw}}$ 已基本说明，附录 D 也可承载详细指标；但 $\phi_{\mathrm{hp}}(x)$、$g_\rho(c)$、$\mathcal{B}_\rho$ 在正文中仍偏抽象。  
**How to revise:** 正文补一两句最低限度解释，详细取值和指标列表放附录 D。  
**Severity:** Major

### Problem 3

**Problem:** Background 的 AI 方法 gap 不够抽象。  
**Why it matters for AAAI:** 评审会判断论文是在解决 AI 方法问题，还是只是部署工程问题。  
**Evidence from the text:** 背景主要讲 roofline、硬件 kernel、通道配置和 INT8 现象，但较少讨论现有硬件感知压缩/搜索方法的建模缺陷。  
**How to revise:** 明确指出现有压缩搜索、硬件调度、NAS/NetAdapt 类方法为何无法处理“软件配置诱导调度空间”的条件依赖。  
**Severity:** Major

### Problem 4

**Problem:** 双层搜索流程已有附录伪代码，但正文与附录之间的承接还可更清楚。  
**Why it matters for AAAI:** 没有伪代码或精确流程，方法很难被评审判断为完整算法。  
**Evidence from the text:** 3.3.1 用自然语言描述外层 SMBO、SA-MOEA、内层 AutoTVM，并指向附录 C；正文已说明候选进入内层和反馈回流，但预算和终止条件主要依赖附录。  
**How to revise:** 在 3.3.1 末尾保留“完整迭代过程见附录 C”，并确保附录 C 写清输入、初始化、candidate generation、inner tuning、measurement、surrogate update、Pareto update。  
**Severity:** Moderate

### Problem 5

**Problem:** AP 稳定性依据需要在实验或附录中留下证据锚点。  
**Why it matters for AAAI:** AP 残差预测会影响候选排序与测量预算分配，评审需要知道该近似在哪些候选范围内成立。  
**Evidence from the text:** 当前文本已引用已有压缩/量化研究并提及本文预实验，但预实验的范围、模型和观察结果尚未在 Method 正文展开。  
**How to revise:** 不需要把完整实验结果放入 Method；可在实验章节或附录中给出一个简短预实验表/说明，支撑“部署可行候选范围内 AP 波动相对受限”的使用条件。  
**Severity:** Moderate

## 9. Final Reviewer-Style Comment

The revised Background and Method identify an important deployment-oriented AI problem: model compression decisions and hardware scheduling decisions are conditionally coupled, so FLOPs, bit-width, or default latency are unreliable proxies for deployed performance. The motivating observation is relevant, and the formulation of a joint space over pruning, quantization, and scheduling now provides a clearer methodological backbone. The Method also shows improved awareness of structural validity, hardware feasibility, multi-objective optimization, measurement feedback, and the boundary between surrogate prediction and final real measurement.

The revision substantially mitigates several earlier concerns. The method now better explains that the outer search ranks pruning and quantization candidates according to hardware feedback induced by the inner scheduling space; appendix material can carry the complete algorithm and symbol definitions; and the AP residual predictor is no longer presented as an unsupported assumption. Nevertheless, the paper should still sharpen the boundary between the proposed mechanism and the existing components used to instantiate it. The central contribution should be explicitly named and framed as a constraint-conditioned bi-level Pareto search mechanism, while DepGraph, LightGBM, NSGA-II, AutoTVM/XGBoost, and NetAdapt-style ideas should be presented as supporting components rather than the source of novelty.

当前 Background 与 Method **已经接近 AAAI 主会论文的方法表述要求，但仍未完全稳固**。最需要优先修改的是：在 Introduction 中明确给出本文方法的贡献边界，在 Method 中持续围绕“约束条件化双层协同搜索/硬件反馈驱动的条件化 Pareto 搜索”组织叙事，并对 $\phi_{\mathrm{hp}}(x)$、$\mathcal{B}_{\rho}$、$g_{\rho}(c)$ 等少数关键接口变量保留最小正文解释。
