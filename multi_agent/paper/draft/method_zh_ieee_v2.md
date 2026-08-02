# 方法

## 3.1 概述

### 3.1.1 背景与动机

从屋脊线模型（roofline model）的视角看，算法在目标硬件上的部署推理速度主要受两个因素共同限制：一是软件侧的计算负载，即完成一次前向推理所需的计算总量；二是硬件侧的有效执行能力，即硬件在给定计算图下能够实际发挥的最高算力。现有加速方法也围绕这两个方面展开：一类是软件压缩方法，通过减少通道宽度、删除冗余计算或降低数值精度来降低模型计算负载；另一类是硬件编译与调度方法，通过选择合适的数据布局、循环划分、算子融合策略，提高模型在目标硬件上的执行效率。
|||||
|-|-|-|-|
|方案|剪枝后通道配置|数值精度|latency|
|（1）|[48, 96, 192]|fp32|21.6 ms|
|（2）|[64, 96, 192]|fp32|6.1 ms|
|（3）|[48, 64, 256]|fp32|19.40 ms|
|（4）|[64, 64, 256]|fp32|5.9 ms|
|（5）|[48, 96, 192]|mix int8|2.7ms|
|（6）|[48, 96, 192]|full int8|4.3ms|

然而，上述两个维度并非彼此独立。表1展示了关键的实测发现，结果表明降低计算负载并不必然带来部署加速，硬件调度收益也不能脱离模型形状单独判断。例如，scheme(1) 与 scheme(3) 采用更窄的通道配置，理论计算负载更低，但由于未能有效触发目标硬件上的高效 kernel，其实际延迟反而显著高于通道补齐后的 scheme(2) 与 scheme(4)；类似地，在相同通道配置下，scheme(5) 的混合 INT8 量化延迟低于 scheme(6) 的全 INT8 量化。这些结果表明，软件侧的剪枝、量化选择会改变硬件侧可用的执行路径，而硬件侧的调度与低精度支持又会反过来决定某个压缩配置是否真正有效。因此，本文不将软件压缩与硬件调度视为两个串行独立步骤，而是在联合空间中搜索满足目标硬件约束的 Pareto 最优实现。

### 3.1.2 框架概述

基于上述发现，本文提出一个面向协同式 V2X 感知模型的软硬件协同优化框架，其系统概览如图 1 所示。本文的框架将软件侧的模型压缩空间和硬件侧的执行调度空间统一建模，并围绕两个核心模块展开：搜索空间构建模块和搜索空间探索模块。具体流程如下。首先，系统以部署模型和目标硬件为输入，分析计算图结构和硬件执行能力，并构建满足结构约束与硬件约束的配置空间。其次，搜索空间探索模块在该空间中联合选择剪枝、量化和调度配置，并以精度、延迟和能耗作为优化目标。为了降低搜索成本，框架引入代价模型，对候选配置和调度方案进行快速评估，仅将优良的候选送入真实硬件测量和精度复核。最后，系统输出一组适配目标硬件的 Pareto 候选方案，并将新增测量结果回流更新代价模型，以提升后续搜索的可靠性。



## 3.2 搜索空间构建

本文的联合设计空间由软件压缩配置与硬件调度配置共同构成。在软件压缩维度，本文以结构化剪枝和训练后量化作为主要优化旋钮。与知识蒸馏或量化感知训练相比，这两类方法能够减少计算与存储开销，同时降低对重训练和网络结构重构的依赖，因此具有更好的迁移性。
在此基础上，本文将软件配置记为 $x=(p,q)$，其中 $p$ 表示剪枝率或对应的通道宽度，$q$ 表示模型级量化策略，包括精度模式及其覆盖方式；所有满足硬件能力约束和计算图结构约束的软件配置共同构成软件搜索空间 $\Theta_{\mathrm{sw}}$。在硬件调度维度，本文遵循现有张量程序自动调优方法中的调度空间定义~\cite{chen2018learning,shao2022metaschedule}，并将硬件调度配置记为 $s$。该配置包含循环分块、访存布局和向量化等后端优化选项。由此，一个完整的部署候选可表示为 $c=(x,s)$。


给定部署场景 $\rho$，本文将联合设计问题表述为一个受约束的多目标优化任务。
$\mathcal{P}^{\star}_{\rho}=\mathrm{Pareto}\{(-\mathrm{AP}(x),\ell(c),e(c))\mid x\in\Theta_{\mathrm{sw}},\ c=(x,s),\ g_{\rho}(c)\in\mathcal{C}_{\rho}\}.$
其中 $\mathrm{AP}(x)$ 为检测精度，主要由剪枝和量化配置决定；$\ell(c)$ 和 $e(c)$ 分别表示完整部署候选 $c$ 在目标硬件上的延迟和能耗；$\mathcal{C}_{\rho}$ 表示硬件能力和部署预算约束，$g_{\rho}(c)$ 表示将候选 $c$ 映射到上述约束中的可行性条件。$\mathcal{P}^{\star}_{\rho}$ 表示满足上述约束的 Pareto 最优候选集合 。

由于剪枝率在搜索空间中近似连续，且协同式 V2X 感知模型包含多类复杂算子与功能模块，软件压缩配置本身已具有较大的组合规模；当量化位宽和硬件调度进一步耦合后，原始联合空间会迅速膨胀至难以直接搜索。为此，本文通过硬件能力刻画和计算图依赖分区显式构建软硬件约束，对原始空间进行压缩，形成规模受控且可由后续优化器高效探索的配置集合。

### 3.2.3 硬件能力刻画

本文将目标硬件能力描述为 $\mathcal{H}$，主要通过三个步骤进行刻画：解析目标硬件与编译后端的能力信息，获得可用计算单元、支持算子集合、数值精度类型、量化粒度以及访存约束等基础能力描述；随后对候选算子和子图进行编译验证，确认其是否能被目标后端接受；最后运行少量微基准探针，识别通道打包和低精度 kernel 触发条件等离散约束。

基于 $\mathcal{H}$，原始候选被转化为部署相关的离散配置。以第 $k$ 个上层搜索分区为例，经硬件能力过滤后的候选集合写为
$
P_k^{\mathrm{hw}}=\{x_k\mid x_k\in\Omega_k,\ \psi_{\mathrm{hw}}(x_k,\mathcal{H})=1\},
$
其中，$P_k^{\mathrm{hw}}$ 表示经硬件能力过滤后保留下来的局部软件候选集合；$\Omega_k$ 表示该分区在约束前的候选集合，包含剪枝和量化选择；$\psi_{\mathrm{hw}}$ 表示由硬件能力 $\mathcal{H}$ 给出的可部署性判别函数，用于排除硬件或后端不支持的软件配置。该集合只刻画软件配置在目标硬件上的可部署性，结构依赖约束由后续计算图分区进一步过滤；具体调度搜索由第 3.3 节的内层调度过程完成。

### 3.2.4 计算图扫描与分区

为避免剪枝操作破坏计算图结构一致性与张量维度合法性，本文采用基于计算图依赖约束结构化剪枝方法 \cite{depgraph}，通过扫描网络计算图，将稠密、定形计算核心中存在通道依赖关系的算子整合为若干依赖块。每个依赖块对应一个最小剪枝分区，记为 $\mathcal{B}_1$ 中的一个元素；剪枝配置旋钮只能作用于这些分区，而不能任意挂载到单个网络层。量化分区不独立重新划分，而是由一个或多个完整剪枝分区合并形成，记为 $\mathcal{B}_2$，以避免量化边界切断通道耦合关系。因此，本文将计算图分区表示为 $\mathcal{G}=\langle\mathcal{B}_1,\mathcal{B}_2\rangle$。
直接以所有底层依赖块作为搜索变量仍会造成过大的组合空间（$10^{30}量级$）。因此，本文进一步将语义位置相近、且具有相同硬件约束的依赖块归并为上层搜索分区 $\mathcal{G}'$。对第 $k$ 个上层搜索分区，其候选配置集合写为
$
P_k=\{x_k\in P_k^{\mathrm{hw}}\mid \psi_{\mathrm{sw}}(x_k,\mathcal{G}')=1\},
$
其中 $P_k$ 表示同时满足硬件能力和计算图分区约束的局部候选配置集合；$\psi_{\mathrm{sw}}$ 表示由计算图分区 $\mathcal{G}'$ 给出的软件结构判别函数，用于排除破坏剪枝分区、量化边界或跨层通道依赖的配置。
最终，所有上层搜索分区的候选配置共同组成软件搜索空间 $\Theta_{\mathrm{sw}}$，其中 $K$ 为上层搜索分区的数量：
$
\Theta_{\mathrm{sw}}=\prod_{k=1}^{K}P_k .
$

因此，搜索空间构建阶段的输出是可枚举的软件候选空间 $\Theta_{\mathrm{sw}}$ 及其硬件/结构约束描述；硬件调度维度由 $s$ 表示，具体调度优化和候选部署表现评估由第 3.3 节完成。



## 3.3 搜索空间探索
为了探索联合搜索空间，本文需要：（1）快速评估已访问候选的多指标部署质量；（2）高效访问候选点。为降低昂贵的硬件测量和精度复核开销，本文构建多指标代价模型预测候选性能，在统一框架下刻画精度、延迟和能耗之间的耦合关系；同时采用 surrogate-assisted NSGA-II 引导候选访问，在无需预设固定权重的情况下高效探索多指标 Pareto 候选。

### 3.3.1 内外双层探索架构

软硬件协同搜索的主要挑战来自两个方面。第一，剪枝和量化配置会改变算子形状、内存访问模式和合法调度空间，使硬件调度依赖于给定的软件配置。第二，硬件调度旋钮数量庞大，如果直接把软件旋钮和硬件旋钮展平到同一空间，搜索维度会急剧膨胀，并产生大量无效候选。为同时保留这种条件依赖关系并控制搜索维度，构建外层软硬件联合探索与内层硬件调度细化相结合的双层架构。

具体而言，外层搜索基于Sequential Model-Based Optimization（SMBO），构建代理模型辅助的多目标进化优化（SA-MOEA）框架，以剪枝和量化配置 $x$ 为显式搜索变量，并通过少量调度探针获得对应的硬件响应信号。搜索初期，已测候选及其多指标测量结果被用作外层代价模型的初始监督信号；随后，多目标进化策略在软件搜索空间 $\Theta_{\mathrm{sw}}$ 内生成候选配置$\mathcal{X}_{\tau}$，代价模型快速评价候选配置，只有位于预测 Pareto 前沿邻域的候选进入内层调度与真实硬件测量；内层返回的真实评测结果继续回流到外层观测集合，用于更新代价模型，并影响后续软件候选的排序与采样。需要注意，外层并不穷尽底层调度空间，而是在候选排序阶段引入硬件反馈，使剪枝和量化配置能够根据硬件执行表现持续重排序。完整的迭代过程见附录 C。

内层调度搜索采用 AutoTVM 风格的自动调优策略 \cite{chen2018learning}：通过进化搜索生成调度候选，并以 XGBoost 代价模型预测候选性能。少量目标硬件测量用于校准该预测器并选择较优调度实现。



### 3.3.2 多指标代价模型
与传统神经架构搜索中面向时延的单目标搜索相比，多指标联合搜索面临样本获取成本高、目标间耦合以及指标响应非平滑等问题。平滑回归模型通常难以刻画此类非线性行为，而基于多层感知机或图卷积网络的预测器往往需要更大规模的训练数据。

因此，本文采用树集成模型，为外层搜索分别构建面向不同优化目标的代价预测器，该类模型能够刻画混合配置特征之间的非线性交互与阈值效应，因而适用于本文有限样本且局部非平滑的评估场景。

对于第 $\tau$ 轮第$i$个候选 $x_{\tau,i}$，指标 $r$ 的预测值如式~\ref{eq}所示。
$
\widehat{m}^{r}_{\tau}(x_{\tau,i})=
\mathcal{E}_{r}^{(\tau)}\!\left(\phi(x_{\tau,i})\right),
\quad
x_{\tau,i}\in\mathcal{X}_{\tau},\ 
r\in\{\ell,e,\mathrm{AP}_{70}\},
$
其中，$\mathcal{E}_{r}^{(\tau)}$ 表示第 $\tau$ 轮针对指标 $r$ 独立训练的树集成预测器。各预测器共享同一输入表示 $\phi(x_{\tau,i})$，但不共享模型参数或监督目标，从而分别适应精度、时延和能耗不同的尺度与测量特性。

由于压缩配置$x$到计算图的映射具有模型依赖性和非线性，相近的压缩配置在不同感知模型中可能对应不同的计算与存储特征。仅依赖配置变量，预测器难以在有限样本下学习上述差异，从而增加跨模型性能预测与候选排序的难度。为解决上述问题，本文引入图感知特征输入 $\phi(x_{\tau,i})$ ，该输入由配置特征和图级统计特征共同构成，其具体形式如式~\ref{eq}所示：

$
\phi(x_{\tau,i})=
\left[
\phi_{\mathrm{cfg}}(x_{\tau,i}),\
\phi_{\mathrm{graph}}(x_{\tau,i})
\right].
$

其中，$\phi_{\mathrm{cfg}}(x_{\tau,i})$ 表示由候选 $x_{\tau,i}$ 对应的压缩配置编码得到的数值向量，作为代价模型的基础输入；$\phi_{\mathrm{graph}}(x_{\tau,i})$，用于汇总该候选在当前压缩配置下的计算图结构信息，包括算子组成与数量、计算量和参数规模等。这使预测器能够直接基于实际计算结构学习其与精度、时延和能耗之间的非线性关系。

为提高小样本条件下多指标回归的数值稳定性与泛化能力，需要根据不同指标的统计特性对训练目标进行适当变换，并选择与之匹配的预测器。本文采用外层五折、内层三折的嵌套式分组交叉验证（nested grouped cross-validation）框架进行联合选择。最终确定时延和能耗采用独立的ExtraTrees 回归器，并以 log1p 变换后的数值标签作为训练目标，以压缩长尾测量值的动态范围。$\mathrm{AP}$ 采用基于 Huber 损失训练的 LightGBM 锚点残差预测器， 将训练折内同一感知模型各样本的实测 $\mathrm{AP}$ 中位数作为锚点，并以样本相对于该锚点的残差作为预测目标，使预测器聚焦于模型内部的相对精度偏移，同时抑制跨模型基础精度差异对回归过程的主导作用。上述变换仅用于改善小样本回归的数值条件，推理阶段的预测结果均恢复至原始指标空间。完整候选集合、选择准则及详细比较结果见附录 E。

### 3.3.3 候选方案生成

在本框架中，候选生成需要在多指标目标之间保持多样化折中，而不是把 AP、延迟、能耗和模型规模预先压缩为单一标量。因此，本文采用 NSGA-II \cite{deb2002nsga} 作为外层候选生成方法。NSGA-II 基于 $\widehat{m}^{r}_{\tau}(x_{\tau,i})$ 对当前候选集合 $\mathcal{X}_{\tau}$ 执行非支配排序和拥挤距离选择，并从 $\Theta_{\mathrm{sw}}$ 中产生下一轮候选集合 $\mathcal{X}_{\tau+1}$。该机制无需预设固定指标权重，能够持续维护覆盖不同部署偏好的 Pareto 候选。

### 3.3.4 证据回流与冷启动

框架的有效性依赖于代理模型和真实测量之间的持续校正。被选中的候选进入硬件在环评估：内层调度器在目标硬件上搜索对应实现，并获得延迟、能耗等真实部署指标；需要进入最终前沿的候选进一步执行统一协议下的微调和 AP 评测。第 $\tau$ 轮新增观测包括候选配置、真实多指标测量结果以及构建状态、kernel 触发和回退情况等辅助硬件反馈，这些观测被加入历史样本，并用于更新下一轮预测器 $\mathcal{E}_{r}^{(\tau+1)}$，以评估候选集合 $\mathcal{X}_{\tau+1}$。上述闭环最终输出满足目标场景约束的 Pareto 候选集合 $\mathcal{P}^{\star}_{\rho}$，而不是单个加权最优配置。

## 3.4 方法边界

本文方法将搜索空间合法性、代理模型预测和最终实验结论明确分离。硬件能力刻画与计算图分区只能说明候选在结构上和后端语义上可被考虑；代价模型只能在搜索过程中提供低成本排序信号；最终精度、延迟和能耗结论必须来自声明范围内的真实测量。对于未被计算图分区覆盖的子图、缺少训练权重的模型，或尚未在目标硬件上完成测量的配置，本文不外推其完整模型级加速性质。该边界使所提出的软硬件协同优化框架既能利用代理模型提高搜索效率，又能保持最终 Pareto 前沿的可验证性。

---

## 当前方法草稿的待修正项（作者工作注释，不属于论文正文）

本节用于记录当前文字与真实实现、最新实验状态之间的不一致。正式成稿前应逐项修正，并在修正完成后删除本节。

### M1. 开头 INT8 表格的证据口径不完整

第 3.1.1 节表格中的 `mix int8 = 2.7 ms` 和 `full int8 = 4.3 ms` 未标明 backend、graph boundary、是否 hand-rewrite、是否自动调优、AP 状态和 evidence trust。该组结果目前只能作为历史 motivating observation，不能作为自动 RouteB INT8 已实现加速的最终证据。

最新同构自动后端结果为：FP16 BlockBuilder `1.8083 ms`；top25 automatic INT8 region `2.0282 ms`，且未通过 numerical gate。因此正式正文必须：

1. 为历史表格补充完整口径并明确其只用于说明“压缩配置与执行路径存在耦合”；或
2. 使用最终通过速度、数值和 AP 门禁的新结果替换该表格。

### M2. 软件搜索变量不能继续简化为纯位宽 `q`

当前写法 `x=(p,q)` 将 `q` 仅定义为量化位宽，无法表示最新框架中的量化策略。当前外层 genome 应抽象为：

```text
x = ([p1,p2,p3], q_mode, mixed_policy_id)
```

其中：

- `[p1,p2,p3]` 或等价 width vector 表示各搜索分区的结构化剪枝配置；
- `q_mode` 表示 FP16、INT8 或 mixed INT8 等逻辑精度模式；
- `mixed_policy_id` 表示由搜索器选择的模型级量化覆盖策略，而不是具体 Conv 名单；
- 具体 Conv、INT8 region、scale 粒度和 fusion 由 compiler/backend realization 根据计算图和 capability profile 自动生成。

正式方法可以继续用抽象符号 `q`，但必须将其定义为 quantization policy，而不是单一 bit-width。

### M3. 量化分区描述与 automatic region formation 不一致

当前第 3.2.4 节称量化分区由一个或多个剪枝分区直接合并形成。最新实现中，量化逻辑 policy 与物理 INT8 region 需要分开：

```text
logical mixed policy
-> stable Conv selection manifest
-> ONNX dataflow analysis
-> automatic INT8 region formation
```

automatic region 会沿唯一消费者的 `Conv -> Relu -> selected Conv` 合并，并在 Add、fan-out、graph output、未选 Conv 或不支持的算子处自动切断。正式正文应分别定义：

- 搜索器可见的 logical quantization partition/policy；
- 编译器生成的 physical INT8 execution region。

二者不能使用同一个符号而不加区分。

### M4. pairwise rank loss（v2 已解决）

v1 曾给出未接入主搜索器的显式 `pairwise rank loss`。v2 已删除该训练目标；Spearman 仅作为 nested grouped cross-validation 中模型选择分数的一部分，不再声称独立训练 value predictor 与 rank predictor。

### M5. AP residual predictor（v2 已解决）

Stage4 模型选择和 Stage5 production search 已采用训练折内按感知模型中位数构造的 AP model-anchor residual，并冻结 `lgbm_huber_residual` 为 $\mathrm{AP}_{70}$ 预测头。v2 已按该实现更新正文，同时保留最终前沿候选必须经过统一 gold AP 真实评测的证据边界。

### M6. 内层调度器和 backend realization 需要按实际工具链重写

当前正文统一写成 AutoTVM-style evolutionary search + XGBoost。项目实际存在多条实现路径：Phase1 `measure_config.py` 的条件化真调优、FP16/INT8 TensorCore lowering、DLight/MetaSchedule 相关路径，以及正在验证的 automatic mixed-region lowering。正式方法必须明确论文主实验到底采用哪一条 canonical pipeline，并给出：

```text
compiler/backend version
schedule search method
tuning budget
database reuse policy
fresh-workdir policy
buildability/correctness gate
fallback policy
```

不能将不同阶段使用过的 AutoTVM、MetaSchedule、DLight、CUTLASS 或 hand-rewrite 结果混写成一个统一实现。

### M7. AP 稳定性前提（v2 已解决）

v1 曾将“适度剪枝或混合精度量化不会引起 AP 大幅波动”写成一般性前提。v2 已删除该假设，仅将残差预测用于外层候选筛选，并明确最终前沿的 AP 坐标必须来自统一协议下的真实评测。

建议改为：AP 对剪枝与量化策略的响应具有模型、层和 realization 依赖性；surrogate 只负责低成本采样，最终 Pareto AP 必须由统一协议真实评测确认。

### M8. 当前 P0--P3 的证据边界

截至当前实现：

| phase | status | 可写入论文的结论 |
|---|---|---|
| P0 per-node calibration | complete | scale 已按稳定 node/tensor/hash 绑定，旧 shape-signature 聚合问题已修复 |
| P1 automatic region formation | mechanism complete | region 可由通用 dataflow 自动形成，top25 Q/DQ 从 7/7 降到 4/4 |
| P2 speed/numerical gate | not passed | 当前 automatic INT8 仍慢于 FP16，rank-2/top25 数值门失败 |
| P3 full AP/Pareto | not started | 不能声称 automatic RouteB INT8 已同时获得速度和精度收益 |

P0/P1 属于机制正确性证据，不等价于最终性能贡献。最终 Abstract、Introduction 和 Experiments 只能在 P2/P3 门禁闭合后加入 INT8 速度/AP 主张。

### M9. 正式方法章节建议重组

建议最终 Method 按真实执行顺序重组为：

```text
3.1 Problem formulation and framework overview
3.2 Capability-conditioned search-space construction
    3.2.1 Structured pruning genome
    3.2.2 Logical quantization policy
    3.2.3 Hardware/compiler capability scan
    3.2.4 Automatic backend realization and region formation
3.3 Surrogate-assisted outer search
3.4 Conditional schedule tuning and hardware-in-the-loop measurement
3.5 Gold AP validation and measured evidence feedback
3.6 Output frontier and evidence boundary
```

该结构将“搜索器决定什么”和“编译器自动实现什么”分开，也能避免把尚未进入 genome 的 backend 名称、具体 Conv ID 和人工规则写成方法先验。

---

## 附录 E：多指标代价模型的目标构造与模型选择

本附录补充正文第 3.3.2 节中多指标代价模型的目标构造、候选模型集合和 nested grouped cross-validation 过程。这些模型均作为外层搜索的低成本代理预测器，不替代最终候选的真实硬件测量和完整 AP 评测。

### E.1 指标特定目标构造

时延和能耗测量均为非负值，且不同配置之间可能存在明显的长尾分布。为减弱少量高代价样本对回归损失的主导作用，本文分别构造
$
y_{\ell}=\log(1+\ell),
\qquad
y_{e}=\log(1+e).
$
预测阶段通过 $\operatorname{expm1}(\cdot)$ 将模型输出恢复至原始时延和能耗量纲。

对于 $\mathrm{AP}_{70}$，不同感知模型的基础精度差异可能大于同一模型内部由剪枝和量化引起的变化。本文因此在每个训练折内按感知模型构建中位数锚点。对于模型 $g$，其锚点定义为
$
a_g=
\operatorname{median}
\left\{
\mathrm{AP}_{70}(x_i)
\mid
g(x_i)=g,\ i\in\mathcal{I}_{\mathrm{train}}
\right\}.
$
相应的残差目标和预测恢复过程为
$
\delta_{\mathrm{AP}}(x)
=
\mathrm{AP}_{70}(x)-a_{g(x)},
$
$
\widehat{\mathrm{AP}}_{70}(x)
=
a_{g(x)}
+
\widehat{\delta}_{\mathrm{AP}}(x).
$
锚点仅由当前训练折计算，不使用内部验证折或外层测试折的信息。该构造使预测器主要学习同一感知模型内部的相对精度偏移，而不是直接拟合跨模型的绝对精度差异。

### E.2 Nested grouped cross-validation

模型选择基于 Gold176 数据中的 44 个完整 $(\mathrm{model},\mathrm{width})$ 组。每组包含同一模型和通道配置下的不同部署臂；划分过程中，同组样本始终位于同一侧，以避免相近配置同时出现在训练集和验证集中。

评估采用 $5$ 折外层、$3$ 折内层的 nested grouped cross-validation。对于每个外层折，模型选择过程如下：

1. 保留一个完整外层测试折，仅用于估计泛化性能。
2. 在其余外层训练数据上执行三折 grouped inner validation。
3. 分别训练全部候选模型，并在每个内部验证折上计算联合选择分数。
4. 对三个内部验证折的分数取平均。
5. 选择平均分数最低的候选，在完整外层训练集上重新训练。
6. 在外层测试折上计算 OOF MAE 和 Spearman。
7. 重复五个外层折，并统计各候选被选中的次数。

内部选择分数定义为
$
S_{\mathrm{inner}}
=
\frac{\mathrm{MAE}}
{\max\!\left(Q_{0.9}(y)-Q_{0.1}(y),10^{-9}\right)}
+0.25\left(1-\rho_{\mathrm{S}}\right),
$
其中，$Q_{0.9}(y)-Q_{0.1}(y)$ 表示当前内部验证折真实标签的稳健跨度，$\rho_{\mathrm{S}}$ 为 Spearman 相关系数。第一项衡量相对于目标动态范围的绝对预测误差，第二项惩罚错误的候选排序；分数越低越好。

该分数仅在同一目标内部用于模型选择。由于时延、能耗和 $\mathrm{AP}_{70}$ 分别使用各自的标签跨度，不同目标之间的 $S_{\mathrm{inner}}$ 数值不能直接比较。

### E.3 候选集合与选择结果

候选集合由模型类别、训练目标和标签构造方式共同确定，如表 E.1 所示。

**表 E.1：候选代价模型集合**

| 目标 | 模型类别 | 训练目标 | 标签构造 |
|---|---|---|---|
| 时延、能耗 | ExtraTrees | squared-error | raw、log1p |
| 时延、能耗 | LightGBM | MAE | raw、log1p |
| 时延、能耗 | LightGBM | Huber | raw、log1p |
| 时延、能耗 | LightGBM | median quantile | raw、log1p |
| $\mathrm{AP}_{70}$ | ExtraTrees | squared-error | raw、model-anchor residual |
| $\mathrm{AP}_{70}$ | LightGBM | MAE | raw、model-anchor residual |
| $\mathrm{AP}_{70}$ | LightGBM | Huber | raw、model-anchor residual |
| $\mathrm{AP}_{70}$ | LightGBM | median quantile | raw、model-anchor residual |

图 E.1 给出了全部 24 个“目标—候选模型”组合的比较。颜色表示候选在当前目标内部的相对表现，红框表示最终保留的预测器。由于颜色归一化在每个目标内部独立完成，不应根据颜色深浅跨目标比较模型性能。

**图 E.1：不同目标下候选代价模型的比较。** 单元格依次给出固定候选的 OOF MAE、OOF Spearman、平均内部验证分数和外层折入选次数。颜色越深表示该候选在当前列和当前目标内的相对表现越好。红框表示最终保留的模型配置。

![图 E.1：不同目标下候选代价模型的比较](../../figure/cost_model_selection/cost_model_selection_heatmap.png)

最终选择结果见表 E.2。

**表 E.2：最终代价模型配置及模型选择结果**

| 指标 | 最终配置 | 标签构造 | 平均内部验证分数 ↓ | 外层入选 | 其他外层入选候选 | 固定候选 OOF MAE ↓ | 固定候选 OOF Spearman ↑ |
|---|---|---|---:|---:|---|---:|---:|
| 时延 | **ExtraTrees / squared-error** | **log1p** | **0.1024** | **5/5** | 无 | **0.9697 ms** | **0.9720** |
| 能耗 | **ExtraTrees / squared-error** | **log1p** | **0.1412** | **4/5** | LightGBM Huber raw：1/5 | **0.2083 J** | **0.9281** |
| $\mathrm{AP}_{70}$ | **LightGBM / Huber** | **model-anchor residual** | **0.1836** | **3/5** | LightGBM MAE residual：2/5 | **0.0448** | **0.8137** |

表 E.2 中的 OOF MAE 和 Spearman 是将单一候选配置固定后得到的诊断结果，用于解释不同模型的泛化行为；它们不是事后选择最终预测器的依据。最终模型由每个外层训练集内部的 $S_{\mathrm{inner}}$ 决定。因此，$\mathrm{AP}_{70}$ 的 Huber residual 虽然不具有最低的固定候选 OOF MAE，但获得了最低的平均内部验证分数，并在五个外层折中的三个折被选中；MAE residual 则在其余两个折被选中。该结果支持将 Huber residual 固定为后续外层搜索中的 $\mathrm{AP}_{70}$ 预测头。
