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
在此基础上，本文将软件配置记为 $x=(p,q)$，其中 $p$ 表示剪枝率或对应的通道宽度，$q$ 表示量化位宽；所有满足硬件能力约束和计算图结构约束的软件配置共同构成软件搜索空间 $\Theta_{\mathrm{sw}}$。在硬件调度维度，本文遵循现有张量程序自动调优方法中的调度空间定义~\cite{chen2018learning,shao2022metaschedule}，并将硬件调度配置记为 $s$。该配置包含循环分块、访存布局和向量化等后端优化选项。由此，一个完整的部署候选可表示为 $c=(x,s)$。


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

具体而言，外层搜索基于Sequential Model-Based Optimization（SMBO），构建代理模型辅助的多目标进化优化（SA-MOEA）框架，以剪枝和量化配置 $x$ 为显式搜索变量，并通过少量调度探针获得对应的硬件响应信号。搜索初期，已测候选及其多指标测量结果被用作外层代价模型的初始监督信号；随后，多目标进化策略在软件搜索空间 $\Theta_{\mathrm{sw}}$ 内生成候选配置，代价模型快速评价候选配置，只有位于预测 Pareto 前沿邻域的候选进入内层调度与真实硬件测量；内层返回的真实评测结果继续回流到外层观测集合，用于更新代价模型，并影响后续软件候选的排序与采样。需要注意，外层并不穷尽底层调度空间，而是在候选排序阶段引入硬件反馈，使剪枝和量化配置能够根据硬件执行表现持续重排序。完整的迭代过程见附录 C。

内层调度搜索采用 AutoTVM 风格的自动调优策略 \cite{chen2018learning}：通过进化搜索生成调度候选，并以 XGBoost 代价模型预测候选性能。少量目标硬件测量用于校准该预测器并选择较优调度实现。



### 3.3.2 多指标代价模型

相比传统 NAS 中以延迟为主的单指标搜索，在多指标联合搜索空间中存在：实测样本获取成本高，各项指标之间相互耦合，且部分指标呈现明显非平滑变化等问题。传统平滑回归模型难以刻画这种非线性关系；多层感知机MLP或GCN图编码预测器通常依赖较大规模训练样本。基于上述考虑，本文采用 LightGBM 作为外层多指标代价模型。
$
\widehat{m}^{r}(x)=
F_{\mathrm{LGBM}}^{r}\!\left(\phi_{\mathrm{hp}}(x)\right)=
\sum_{t=1}^{T_r}\eta_t^{r}h_t^{r}\!\left(\phi_{\mathrm{hp}}(x)\right),
\quad
x\in\Theta_{\mathrm{sw}},\ 
r\in\{\mathrm{AP},\ell,e\},
$
其中，$\phi_{\mathrm{hp}}(x)$ 表示对软件配置 $x$ 进行硬件探针增强后的向量特征，$h_t^{r}$ 为指标 $r$ 对应模型中的第 $t$ 棵回归树，$\eta_t^{r}$ 为其权重。树模型的分裂结构使其能够直接处理离散变量、条件特征和局部非连续响应，因而适合本文的小样本软硬件联合评估场景。

与上述异质性相对应，本文不对所有指标使用统一损失，而是按指标信噪比和使用方式分别建模。由于外层搜索变量为软件配置 $x$，下述代价模型均在 $x$ 层面预测其经条件化调度后可达到的部署表现。对延迟和能耗，本文采用值预测与序预测相结合的训练目标 \cite{chen2018learning}。值预测器估计 $\log \ell$ 与 $\log e$ 以减弱大数值样本对回归损失的主导作用，并给出搜索阶段的 Pareto 坐标和约束判断，其损失写为：
$
\mathcal{J}_{\mathrm{val}}^{a}=
\sum_i
\left(\hat{u}_i^{a}-\log m_i^{a}\right)^2,\quad
a\in\{\ell,e\}.
$
其中 $m_i^{a}$ 表示软件配置 $x_i$ 经内层调度后获得的指标 $a$ 的观测值，$\hat{u}_i^{a}$ 表示对应的对数值预测。上述值预测仅用于延迟和能耗，AP 不进入对数值预测目标。
序预测器在相同模型和硬件分组内学习候选间的相对优劣，其 pairwise rank loss 定义为
$
\mathcal{J}_{\mathrm{rank}}^{a}=
\sum_{(i,j)\in\mathcal{R}_a}
\log\left(1+\exp\left[-\mathrm{sign}(m_i^{a}-m_j^{a})(\hat{o}^{a}(x_i)-\hat{o}^{a}(x_j))\right]\right),
$
其中 $\mathcal{R}_a$ 表示由同一模型和硬件分组内的软件配置 $x_i,x_j\in\Theta_{\mathrm{sw}}$ 构成的训练样本对，$\hat{o}^{a}(x)$ 为软件配置 $x$ 在指标 $a$ 上的序预测分数，该值越小越好；由于 $\ell$ 和 $e$ 越小越优，上式通过候选对的相对次序约束模型学习低代价配置与高代价配置之间的排序关系。最终训练目标为 $\mathcal{J}^{a}=\mathcal{J}_{\mathrm{val}}^{a}+\lambda\mathcal{J}_{\mathrm{rank}}^{a}$。值预测给出可解释的指标坐标，序预测用于稳定候选排序，从而降低尺度差异对搜索决策的影响。**在实现上，值预测器和序预测器共享相同的输入特征 $\phi_{\mathrm{hp}}(x)$，但分别训练对应的回归目标和排序目标。**


已有感知模型压缩与量化研究以及本文预实验均表明，V2X 协同感知网络通常具有一定结构冗余，在适度剪枝或混合精度量化下不会引起 AP 的大幅波动 \cite{he2018amc,wang2019haq}。同时，绝对AP易受 checkpoint 质量、finetune 收敛和评测波动影响，因此本文采用基于真测锚点的 AP 残差预测 \cite{eccv2018NetAdapt}。
$
\Delta \mathrm{AP}(x)=\mathrm{AP}(x)-\mathrm{AP}(x_{\mathrm{ref}}),
\quad
\widehat{\mathrm{AP}}(x)=\mathrm{AP}(x_{\mathrm{ref}})+\widehat{\Delta \mathrm{AP}}(x).
$
其中 $x_{\mathrm{ref}}$ 表示同一模型和评测协议下的已测参考软件配置，$\Delta \mathrm{AP}(x)$ 刻画候选软件配置 $x$ 相对该锚点的精度变化。搜索阶段以 $\widehat{\mathrm{AP}}(x)$ 作为排序和采样依据，用于前沿邻域选择与测量预算分配。最终进入 Pareto 前沿的候选仍需经过统一训练或微调协议下的真实 AP 评测，其 AP 坐标以真实评测结果为准。

### 3.3.3 候选方案生成

在本框架中，候选生成需要在多指标目标之间保持多样化折中，而不是把 AP、延迟、能耗和模型规模预先压缩为单一标量。因此，本文采用 NSGA-II \cite{deb2002nsga} 作为外层候选生成方法。NSGA-II 基于代价模型预测的 $\widehat{m}^{r}(x)$ 执行非支配排序和拥挤距离选择，从 $\Theta_{\mathrm{sw}}$ 中产生下一轮软件候选。该机制无需预设固定指标权重，能够持续维护覆盖不同部署偏好的 Pareto 候选。

### 3.3.4 证据回流与冷启动

框架的有效性依赖于代理模型和真实测量之间的持续校正。被选中的候选进入硬件在环评估：内层调度器在目标硬件上搜索对应实现，并获得延迟、能耗等真实部署指标；需要进入最终前沿的候选进一步执行统一协议下的微调和 AP 评测。第 $t$ 轮新增观测包括候选配置、真实多指标测量结果以及构建状态、kernel 触发和回退情况等辅助硬件反馈，这些观测被加入历史样本并用于更新外层代价模型。上述闭环最终输出满足目标场景约束的 Pareto 候选集合 $\mathcal{P}^{\star}_{\rho}$，而不是单个加权最优配置。

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

### M4. pairwise rank loss 尚未在 Phase1 主搜索器中实现

当前第 3.3.2 节给出了显式 `pairwise rank loss`，并声称 value predictor 与 rank predictor 分别训练。实际 `stage2_smbo_joint_nsga2_v1.py` 的 Phase1 主路径使用 `LGBMRegressor(objective="regression")`：latency/energy 使用回归目标，AP 也使用直接回归或弱信号 fallback；当前没有独立训练的 pairwise rank predictor。

正式处理只能二选一：

1. 按真实实现删除 pairwise rank loss，只保留 value surrogate、uncertainty/acquisition 和 NSGA-II 非支配排序；或
2. 真正实现 rank predictor，完成 held-out rank、搜索 HV 和预算效率消融后再保留该公式。

在完成选项 2 前，摘要不能声称提出了 value-and-rank joint training。

### M5. AP residual predictor 不是当前 Phase1 主实现

当前第 3.3.2 节写成基于真测锚点的 `Delta AP` 残差学习，但 Phase1 joint NSGA-II 主搜索器当前使用直接 AP regression；AP 信号不足时使用 plateau-style fallback，最终前沿候选再通过统一 gold finetune/AP evaluation 验证。

正式正文应按真实闭环改写为：

```text
AP surrogate for acquisition
-> frontier-neighborhood candidate selection
-> gold AP evaluation for final frontier
-> measured feedback updates the surrogate
```

只有在残差模型接入主搜索器并完成比较实验后，才能保留现有残差公式。

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

### M7. “适度量化不会引起 AP 大幅波动”需要收紧

当前第 3.3.2 节将“适度剪枝或混合精度量化不会引起 AP 大幅波动”写成一般性前提。最新 CoDriving 实验已经观察到多条 TVM INT8/mixed INT8 路线 AP70 为 0，以及 rank-2/top25 numerical gate 失败。因此该句不能作为普遍假设。

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
