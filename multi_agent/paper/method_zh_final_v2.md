# 方法（Method）v2

我们将协同式 V2X 感知的软硬件协同优化建模为一个两阶段流水线。阶段一进行一次性、面向部署的网络-硬件协同刻画：在给定模型配置、checkpoint 与目标硬件能力描述后，框架生成可审计的 trace 边界，扫描可追踪的稠密计算核心，并将硬件约束、结构依赖、验证状态和模型级证据边界写入 manifest。阶段二在该 manifest 定义的剪枝、量化、调度和路由空间上执行联合搜索，输出按部署场景条件化的 Pareto 前沿方案。

## 阶段一：网络-硬件协同刻画与模型分类

合法配置集合由网络结构与目标加速器共同决定。剪枝率只能挂载在结构上必须联合剪除的通道组上；一个模块可行的量化位宽和粒度取决于其目标硬件 IP 与后端；调度收益也随宽度、精度和 batch 共同变化。因此，阶段一不直接优化模型，而是作为前置分析环节依次完成三个子阶段：硬件能力刻画、计算图扫描与分区、模型分类。

**1. 硬件能力刻画。** 对于每个目标加速器，我们以机器可读的 capability YAML 记录其能力，并形式化为
$
\mathcal{H}=\langle\mathcal{I}, \mathbf{a}, \mathcal{Q}, \mathcal{T}\rangle
$。
其中，$\mathcal{I}$ 枚举可用计算 IP、支持精度和算子可达性；$\mathbf{a}$ 描述不同数值精度下的通道对齐、pack factor 和 INT8 buildability 约束；$\mathcal{Q}$ 规定可用位宽、量化粒度、对称性和逐通道支持；$\mathcal{T}$ 锁定后端与工具链版本。该静态 capability 层用于离线判断结构合法性和搜索空间边界，但不替代目标硬件上的实测。本文新增测量统一采用 H800 TVM/Relax/MetaSchedule；TRT 数据仅作为 historical evidence 进入报告，不作为新增测量后端或默认分类依据。

**2. 计算图扫描与分区。** 给定模型名、config、checkpoint 和最小模型加载入口后，阶段一加载完整模型并生成 trace plan。默认流程通过模块树扫描与 dense candidate path 检测，隔离可进入 dense DepGraph 的稠密计算核心，并排除 sparse VFE、scatter、geometry projection、routing、attention、fusion 和 postprocess 等不适合直接结构化剪枝的子图。对于 HeterModelBaseline 系列，框架使用 `TraceBoundaryDetector.heter_baseline_v1` 生成 trace plan；已有特殊模型仍可保留 legacy wrapper 或专用 wrapper，但其输出均被规范化为统一的 `trace_plan` schema。

在选中的 trace candidate 上，框架执行一次 forward dry-run，并基于 `torch-pruning` DepGraph 提取三类结构视图：（B1）剪枝组，即结构上必须联合剪除的最小层集合；（B2）量化单元，由完整 B1 组的并集构成，避免量化边界切分通道耦合组；（D）路由标注，逐算子记录硬件可达性。随后，B1 剪枝组按语义桶和 stage 聚合为少量搜索旋钮，每个旋钮继承其成员中最严格的剪枝率上界；逐算子路由标注则折叠为最大连续可路由子图，以匹配实际路由决策粒度。该整合过程只减少自由度，不放宽结构约束。每份 manifest 还通过一次 fail-fast 验证记录其可用性，包括 wrapper 前向、DepGraph 构建、物理试剪、输出形状和接口一致性检查。需要强调的是，上述验证覆盖的是可 trace 的 dense candidate，而不是完整模型 AP 或所有 skipped subgraph 的端到端行为。

**3. 模型分类。** 阶段一最后将结构 manifest 与 S2--S4 证据合并为模型级结论边界。S2/S4 的 H800 TVM measured cells 用于判断代表性单元上是否存在 P/Q/S/batch 耦合信号；S3 绑定量化敏感性证据；TRT 相关结果仅保留为历史证据。校准预测器（`calibrated_predictor`）据此记录证据类别、触发规则、阻塞项、后续验证门和不支持外推的结论；最终分类器（`model_classifier`）进一步读取默认模型的 manifest 与 evidence directory，生成三类模型状态：

1. `CO_ACCELERATION_REQUIRED`，即需要协同加速或至少需要 pair/joint 级校准；
2. `SEPARABLE_ACCELERATION`，即仅在声明 scope 内支持可分离加速；
3. `SCAN_FAILED`，即缺少 trained checkpoint 或扫描只达到 architecture-only，不能作为训练模型分类。

分类器同时保留 `classification`、`scope`、`evidence_level`、`blockers`、`required_next_probe_or_gate`、`historical_evidence_sources`、`measured_h800_tvm` 和 `unsupported_conclusions` 等字段。`no_overpromotion=True` 是硬约束：`groups=1`、no cliff、bridge 层的低风险描述或 skipped-subgraph 未覆盖，都不能被提升为模型级可分离证明。

最终，阶段一输出可表示为
$
\mathcal{M}=\langle\mathcal{H},\mathcal{G},\Theta,\lambda,v,\mathcal{E},\mathcal{C}\rangle
$。
其中 $\mathcal{H}$ 是硬件能力，$\mathcal{G}=\langle\mathrm{B1},\mathrm{B2},\mathrm{D}\rangle$ 是三视图结构分区，$\Theta$ 是整合后的搜索变量，$\lambda$ 是 trace-net 延迟先验，$v$ 是运行时验证状态，$\mathcal{E}$ 是 S2--S4 证据链，$\mathcal{C}$ 是模型级分类与禁止外推项。

## 阶段二：软硬件协同优化

阶段二消费阶段一 manifest 定义的低维但合法的搜索空间。一个候选配置由剪枝宽度、量化策略、设备路由和调度选择共同决定；其中调度空间不是固定笛卡尔积，而是由选定的宽度与精度重新诱导。因此，先剪枝、再量化、再调度的串行贪心会丢失一类关键候选：某个宽度在默认调度下不是最优，但经 MetaSchedule 调优后可能反超默认最优宽度。S2/S4 的 H800 schedule-anchor 证据正是用来标注这种 P/Q/S/batch 耦合是否在代表性单元上出现。

**搜索空间构建。** 外环搜索在 manifest 的 B1 search knobs、B2 quant units 和 D routing segments 上枚举候选；内环调度空间随每个 $(W,Q)$ 组合重建。合法性谓词提前裁掉不可构建配置：例如 INT8 pack factor、per-group 输入通道、分组卷积组数和硬件对齐共同决定某个剪枝宽度是否能构建 INT8；被路由到 DLA/NPU 的模块继承更严格的量化和 op whitelist 约束。对当前目标硬件没有真实加速收益的半结构化或非结构化剪枝不被纳入 Stage1 默认剪枝对象。

**分层代价模型与真实测量回灌。** 调度内环复用 TVM/MetaSchedule 代价模型和实测 latency cells；外环用模型/硬件条件化的统计模型估计延迟、吞吐、能耗和构建成功概率。精度不作为纯预测值直接写入最终 Pareto 前沿：AP 相关结论必须来自经训练或 finetune 收敛后的整帧真测锚点，并回灌更新代理模型。对于缺 checkpoint 的 architecture-only scan，框架只允许输出结构扫描状态，不能输出 trained checkpoint classification。

**场景条件化 Pareto 前沿。** 评估器为每个候选输出精度、延迟、吞吐、能耗和模型体积等部署指标；在闭环驾驶场景中，还可加入驾驶得分和路线完成率。不同部署场景只改变目标集和约束集，不改变 Stage1 manifest 的合法性定义。前沿搜索采用外环多目标搜索与内环调度查表/调优结合的方式；当 LUT 未命中或不确定性接近 Pareto 前沿时，主动学习回路选择少量候选进行真实测量并回灌。

## 方法边界

当前框架已经实现 9 模型默认 manifest/classifier 流程，但它不是一个对任意未知模型都保证正确的学习型分类器。新模型接入时，用户必须提供可构建的 config、checkpoint 和模型加载入口，并审核 Stage1 生成的 trace boundary manifest；新硬件接入时，静态 capability YAML 可离线运行结构扫描，但任何延迟、吞吐、AP 或能耗结论都必须在目标硬件上通过对应后端实测获得。TRT 相关数据只可作为 historical evidence；新增测量、默认后端和验收后端统一使用 H800 TVM/Relax/MetaSchedule，除非显式修改 backend policy 并重新验收。
