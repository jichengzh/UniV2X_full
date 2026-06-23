# 方法（第二部分）· 阶段二：软硬件协同优化

> 论文 Method 第二节中文稿 (v2, 2026-06-20)。相对 v1 的核心演进：搜索平台迁移到 **TVM**，主搜索轴由"剪枝 × 量化 × 部署"重定为 **剪枝（P）× 调度（schedule, S）联合搜索**（TVM MetaSchedule），量化（Q/INT8）因 TVM relax 暂无 INT8 pass 降为 future/次要轴；搜索器由单层进化升级为 **嵌套双环**（外环软件多目标 NSGA-II × 内环 schedule MetaSchedule）；并据 ALT / CHaNAS / AutoTVM 三篇文献重构空间构建、代价模型与有效性论证。设计依据：`multi_agent/methods/design/auto-tuning/{1_design_space_building,2_design_cost_model,3_design_exploring_tvm_integration,4_design_ablation_proof}_v1.md`，文献研读 `references/study_joint_search_methods_v1.md`。承接阶段一产物 `framework/partitions/*_partition.yaml`。

---

## 3.2 阶段二：软硬件协同优化

### 动机：为什么必须联合搜索

阶段一把一个理论上可配置的网络翻译成具体的搜索变量及其合法取值——但"哪些可调"不等于"哪个最优"。压缩配置由两侧构成：**软件侧**决定"算什么"（剪枝改变通道宽度、量化改变数值格式），**调度侧**决定"怎么算"（张量化 tile、循环序、算子融合、layout 变换）。本节的根本主张是：这两侧不能各自独立优化再拼装，**也不能以"先定软件、再调调度"的串行流水求解**，原因有二，且第二条比第一条更深。

**(一) 跨维耦合陷阱。** 一个维度的局部最优会被另一个维度抵消。典型如某模块剪到非对齐宽度后，其分组卷积的 `in_per_group` 不再是 2 的幂，无法命中 tensor-core 快核而触发回退核，使*更激进*的剪枝反而*更慢*；又如对齐回填（padding）会同时吞掉剪枝收益与量化增益。这些耦合点使任何"分项最优之和"都偏离真实前沿。

**(二) 串行贪心的结构性局部最优（核心原理）。** 比"陷阱存在"更本质的是：一个**串行搜索器**会因其搜索*结构*而系统性地够不到全局最优解，这与它是否"足够聪明"无关。设串行先在剪枝维 P 上搜索——此时它只能在 **默认调度（default schedule）** 下观测各宽度的延迟，于是贪心选中在该口径下 (AP, latency) 最优的宽度，记为 $W_g$。但关键事实是：**各宽度的调度调优余量并不均匀**（剪枝宽度一变，合法 tile 因子集、可命中的快核、可融合性、可张量化性都随之重建——此即 ALT "layout 变则 loop 空间重建" 的同构现象）。因此各宽度在默认调度下的延迟排序，**不等于**它们各自被 MetaSchedule 调优到底后的排序：某个在默认调度下平庸（余量大、尚未被利用）的宽度 $P_g$，调优后可能反超成为全局最优。串行一旦在第一阶段按默认调度锁死了 $W_g$，在后续调度阶段就**永远不会回头改宽度**，于是 $P_g$ 整条分支从它的可达空间里被结构性地切除。

我们把这一原理形式化为一对配置点：$W_g$ = **单维（贪心所搜那一维）最优、但多维（全局）非最优** 的点（贪心吸引子）；$P_g$ = 单维次优、被 $W_g$ 支配，但只有它在联合调优后才成为全局最优的点（被错过的全局点）。串行的失败不是"运气差选了坏点"，而是"先锁一侧"的搜索结构使 $W_g$ 的吸引盆地不含 $P_g$。这与 ALT-FP/BP 微基准"串联算子单向传 layout，第一个算子的最优 layout 对第二个次优"的论证完全同构，也与 CHaNAS"最优架构随调度漂移、Pareto 彼此相交"的观察一致。**正是这条原理决定了三维必须纳入同一搜索器联合求解，而非串行流水。**

由此，阶段二由四部分构成：在阶段一的合法集合上构建可枚举且非笛卡尔的**搜索空间**、训练一个免逐候选真测的分层**性能评估器（代价模型）**、定义多目标下的**优化目标（Pareto 前沿）**、以及在上述之上以嵌套双环求解的**联合搜索器**。

### 搜索空间构建（两阶段定义 × 块分解 × 合法性收缩）

我们将一个候选的*软件—部署*配置形式化为三元组 $c=\langle B_1,B_2,D\rangle$（剪枝、量化、部署路由）；对每个 $c$，其确定的通道宽度 $W$ 与数据格式 $P$ 进一步诱导一个**调度子空间** $\Sigma_{\mathrm{sch}}(W,P)$，一个完整部署点是 $\langle c,\,S^\star(c)\rangle$，其中 $S^\star$ 是该子空间的内层最优调度。理论上三维（外环）× 调度（内环）的联合规模为 $\sim\!10^{15}\times O(10^9)$（调度模板空间本身即 $O(10^9)$），呈指数爆炸，既不可穷举也无法朴素采样覆盖。可枚举性来自**对阶段一刻画结果的压缩**与**三篇文献的空间分解机制**，而非任意裁剪。

**(1) 两阶段空间定义（借 ALT cross-exploration）。** 空间不是 $\Sigma_{\mathrm{sw}}\times\Sigma_{\mathrm{sch}}$ 的固定笛卡尔积，而是分两层：外层软件空间 $\Sigma_{\mathrm{sw}}=P_{\text{prune}}\times Q_{\text{quant}}$（多目标，交外环 NSGA-II）；每选定一个 $(W,P)$，**重建**该配置下的合法调度子空间 $\Sigma_{\mathrm{sch}}(W,P)$（即"联合"语义的空间根据——调度空间是软件配置的*函数*，而非独立维度）。这对应 ALT 的 joint stage（每选一个上游配置即重建下游空间）+ 可选的 schedule-only 精修阶段（软件配置冻结后再多搜几轮调度）。沿用 ALT"只为复杂算子建空间"的原则，我们**只为稠密通道耦合核心（conv / grouped-conv / GEMM）建调度空间**；稀疏编码器（VFE/LSS）与协同融合 neck（通道保持）不进调度搜索空间，与阶段一"只追踪稠密核心"的边界一致。

**(2) 按块分解 + 调度查找表（借 CHaNAS R·B·S + R^B）。** 外环每个候选都真跑一次内环调度搜索（分钟级）不可行。我们沿用 CHaNAS 的块级预调度：把网络切成 $B$ 个**块**（Pyramid backbone 的 per-stage；双 agent 的 RSU 段 / 车端段），对每个 $(\text{块},\,W,\,P)$ 组合**离线预调度一次**（内层 MetaSchedule 搜得最优调度），把"最优调度 + 实测 (latency, energy)"存入**调度 LUT**。其代价是 $B\cdot|W|\cdot|P|$（**线性**），而非 $(|W|\cdot|P|\cdot|S|)^{B}$（指数）。外环在线组合时延迟由查表加性合成 $\textit{Lat}_{\text{net}}=\sum_{b}\textit{Lat}_{\mathrm{LUT}}[b,W_b,P_b]$，不再触发内环。这把"$P\times$ schedule 联合搜"从指数压回"线性建表 + 多项式外环查表"，是整套方法工程可行的核心。

**(3) 合法性谓词 + 依赖传播（借 ALT layout propagation + CHaNAS divisible + AutoTVM 模板）。** 空间是非笛卡尔的，合法子空间由两类机制在采样阶段裁定：

- **合法性谓词** `legal(W,P,hw)`：把"对齐"从单点（÷32）一般化为分组卷积的真根因——对每个分组卷积层（groups $=g$），命中 tensor-core 快核当且仅当 $\mathrm{in\_per\_g}(W)=2\cdot\textit{num\_filters}(W)/g\in\{2^k\}$。这与 CHaNAS"split factor 限可整除、不可整除更差"在调度内环的自限是同一现象在剪枝宽度轴的前移；并用解析谓词（而非真跑）提前裁掉非法宽度，等价 ALT 用数据复用分析裁 layout 子空间。叠加来自阶段一硬件能力 $\mathcal{H}$ 的精度门控谓词（敏感单元 head/attn/grid_sample/VFE 门控为 FP16-only；DLA 路由 $\Rightarrow$ INT8 $\wedge$ per-tensor $\wedge$ HWC4）。
- **三跳依赖传播**（剪枝 $W\to$ 量化 $P\to$ schedule）：剪枝决定通道宽度 / $\mathrm{in\_per\_g}$ / `round_to` floor；量化决定数据格式与粒度（INT8$\to$对齐 32，FP16$\to$8；DLA$\Rightarrow$per-tensor）；二者**共同重建**合法调度子空间（tile 因子须整除宽度、能否 INT8-tensorize、能否融合均被传播裁定）。这是 §"两阶段空间定义"中"空间重建"的传播实现——不是每个调度参数自由搜，而是宽度/格式定后合法调度集被传播确定。该机制要求调度层落在 TVM（MetaSchedule + 自定义 layout-transform pass）而非黑盒，方能在 IR 的 pass 里改写张量访问索引而不手写 kernel。

**(4) 轴塌缩与硬件注入。** 剪枝对象塌缩为常量 `channel`（半结构化 2:4 与非结构化 element 在目标硬件拿不到真加速，剔除）；重要性准则、剪枝粒度、量化对象/校准器等只决定"剪哪些/如何执行"而不改最终通道数，对前沿零位移，固定不入空间。各旋钮的剪枝率上界、量化单元可行位宽、路由段 DLA 可达性，均由阶段一汇合的 $\mathcal{H}$ 给定；无 DLA 的目标（如 RTX 4090）路由轴退化为单 GPU 段。

经此压缩，软件侧搜索空间由 $\sim\!10^{33}$（结构耦合组视图）降至 $\sim\!10^{4}$，调度搜索成本由指数降为 $B\cdot|W|\cdot|P|$ 的线性建表，外环对 $\sim\!10^{4}$ 配置查表加性组合即可遍历。阶段一附带的**逐单元延迟权重**进一步把"该不该搜某旋钮"前置为优先级，并支持 CHaNAS 式的子空间预筛——在合法性过滤之后、内环建表之前，用评估器估计各子空间的 (AP, latency) 联合 CDF，优先把建表与搜索预算投向有利子空间，避免在注定差的子空间上空耗内环预算。

### 性能评估器构建（分层异构代价模型）

联合搜索的内核是一个免逐候选真测的评估器，使搜索器无需为每个候选都执行 TRT/TVM 构建、$1789$ 帧 AP 真测或一次闭环仿真。它**不是单个回归器，而是一个分层异构集合**——这一选择由我们任务相对三篇文献的根本差异决定：三篇皆为**单目标延迟**（精度作约束或一次性 one-shot 评估），我们是**多目标 + AP 不可纯预测**。

**(1) 内外分层的根因。** 内层（调度，单 $(W,P)$ 下、同数量级、µs 级 span，特征是 TIR 的 AST 结构）与外层（跨 $W$/精度、跨数量级、特征是表格标量配置）在数量级与特征类型上根本不同——把二者合一个回归器正是早期延迟预测器"跨数量级混训给负值预测"崩塌的根因。故：
- **内层调度代价模型直接复用 TVM MetaSchedule 的 XGBoost cost model**（AutoTVM 后继），其特征即 `feature_extractor` 的 buffer-touch / arithmetic intensity / loop 结构等 low-level AST 特征，自带 rank-style 目标 + evolutionary_search + 在线实测回灌。调度级特征是 AST 的函数而非表格配置，自建等于重写编译器栈——"不重造轮子"。由 ALT cross-exploration 决定：宽度一变 AST 即变，内层模型**不能跨 $W$ 复用同一 fit**（每个 $(\text{块},W,P)$ 各自 tune）。
- **外层代价模型用 per-(model, hardware) 分层的梯度提升决策树（LightGBM）**：树分裂天然拟合阶跃（kernel 对齐悬崖、校准崩塌、构建失败等非光滑跳变）、原生支持类别与条件特征、在中小规模表格数据上系统性优于深度模型。固定拓扑变配置排除了 GNN 架构编码，非光滑与条件特征排除了高斯过程平滑核，小样本排除了 MLP。

**(2) 外层各目标的建模方式（信噪比驱动）。** 延迟/能耗的 span 达数量级（信号强），AP 的配置效应仅 $\sim\!0.04$ 且贴近任务上限（信号弱）——故二者用不同机制：

$$
f_{\text{feasible}}:\ c\mapsto\{\textit{build\_success},\,\textit{AP\_crash}\}\quad(\text{先于一切回归的二分类门})
$$
$$
f_{\text{lat}},\,f_{\text{energy}}:\ \textbf{lambdarank（序头）}\ +\ \log\text{-回归（值头）双头}
$$
$$
f_{\text{AP}}:\ c\mapsto\Delta\textit{AP}\ \ (\text{残差、仅排序用}),\qquad f_{\text{size}}:\ \text{解析公式（params）}
$$

延迟/能耗采用 AutoTVM 的 **rank/pairwise loss** $\mathcal{L}_{\text{rank}}=\sum_{i,j}\log(1+e^{-\,\mathrm{sign}(c_i-c_j)(\hat f_i-\hat f_j)})$：搜索的选择阶段只关心相对快慢序，rank loss 对跨数量级混训**结构性免疫**（以 $(\textit{model},\textit{hw})$ 为 group，跨数量级配对不进同一 group，从机制上杜绝非法比较）。同时保留一个 $\log(\textit{lat}_{p50})$ 分层回归头供 Pareto 坐标轴与 iso-AP 延迟读数——**回归头给坐标、rank 头给序**，冲突以 rank 为准；校验指标用 Spearman/Kendall-$\tau$/NDCG@k 而非仅 R²。AP 预测**残差** $\Delta\textit{AP}$ 而非绝对值（绝对值由检查点质量主导、配置效应是小信号，预测残差信噪比高一个量级），且**只用于在搜索中给候选排序、筛掉明显劣者，绝不作为最终 Pareto 的 AP 真值**。吞吐在 batch 轴数据就绪前由 $1/\textit{lat}$ 派生，模型尺寸由参数量解析公式直算。

**(3) AP 真测漏斗（相对三篇文献的最大扩展）。** AP 不像延迟可纯预测：预测信号弱、真值代价极高（每点 = finetune 收敛 + $1789$ 帧）、且不 finetune 的裸 AP 是悲观假象（剪枝裸 AP 暴跌、finetune 后恢复）。三篇文献都回避了"必须 finetune 真测的指标如何进搜索循环"。我们的原则是 **AP 不进"纯预测的代价模型"，而以"漏斗粗筛 + 真测锚点 + 真测回灌"混合形态进入搜索**，按代价从低到高分层，让昂贵的 finetune 真测尽量少：

$$
\text{NSGA-II 候选}\ \xrightarrow{\ \Delta\textit{AP}\ \text{残差} + \text{廉价代理（零真测成本）}\ }\ \text{前沿邻域 Top-K}
$$
$$
\xrightarrow{\ \text{早停 finetune 粗排（中成本，仅排序）}\ }\ \text{前沿候选}\ \xrightarrow{\ \text{收敛 finetune}+1789\ \text{帧真测（高成本锚点）}\ }\ \text{回灌重训}
$$

关于 CHaNAS 式 one-shot weight-sharing：剪枝子网天然是 base 检查点的"子网"，概念上适配 weight-sharing，但裸继承权重未经"渐进蒸馏使所有子网共享后皆近最优"的超网训练，其 AP 是悲观偏置——故仅用于剔除极端崩塌点，精确 AP 仍须 finetune 真测（是否值得另训一个剪枝鲁棒超网换取免逐候选 finetune，取决于搜索候选规模，留作扩展）。

**(4) 跨硬件迁移（借 AutoTVM global+local 分解）。** 新硬件（如 Orin）的延迟代价模型按 $\hat f_{\text{Orin}}(c)=\hat f^{\text{global}}(c)+\hat f^{\text{local}}_{\text{Orin}}(c)$ 分解：$\hat f^{\text{global}}$ 来自 4090 大样本 + 已拟合的跨平台映射（冷启动，不需 Orin 数据），$\hat f^{\text{local}}_{\text{Orin}}$ 是 Orin 少量实测拟合的残差头。invariant 表示 = 配置标量中对硬件不变的部分（剪枝率、位宽、参数量、对齐特征）进 global 头，硬件相关部分（路由方案、DLA 可达性）进 local 头；内层调度的 invariant 表示直接用 MetaSchedule 的 AST context 特征。

**(5) 驾驶轴两级预测。** 驾驶得分 $\mathrm{DS}$ 与路线完成率 $\mathrm{RC}$ 不直接从配置回归，而是两级：先由感知头给出 $\widehat{\mathrm{AP}},\widehat{\ell}$，再经闭环仿真标定的响应面 $g$（"驾驶得分–注入时延"曲线族，按感知质量调制）换算——把闭环采样从"每配置一次"降为"标定一族曲线"，并使 $\mathrm{DS}$ 显式成为延迟与精度的可微下游函数。

**(6) 训练与在线更新。** 内层 MetaSchedule 在 tune 过程中自产自训（AutoTVM Algorithm 1）。外层以 pilot 集训 v0，不确定性以 quantile 三分位或 bagging 集成方差给出，驱动 **active-learning** 采样回路：以 quantile 宽度 × Pareto 邻近度加权的 acquisition 选下一批 $K=8\text{–}16$ 候选交真测、回灌主表重训，并保留 $\sim\!5\text{–}10\%$ ε-greedy 随机配置（尤其对齐边界附近）防搜索锁死在已知好区。训练协议以 `triplet` 为组做 GroupKFold（杜绝同一剪枝架构的量化/部署变体跨折泄漏）、同延迟口径内训练（禁子模块与端到端混训）、小样本下用浅树强正则。

### 优化目标：regime 条件的多目标 Pareto 前沿

"一个配置是否更优"由**多目标 Pareto 前沿**判定：没有任何单一标量能正确排序在相互冲突目标上各有取舍的配置——更激进的压缩以精度换延迟与能耗，更高的吞吐可能以单帧延迟劣化为代价——任何预设权重的综合分都会掩盖分项信号并把方法论锁死在某一部署偏好上。这里需厘清三个层次。

**(i) 评估器对每个配置输出 5 个感知指标 + 2 个端任务驾驶指标，共 7 项。** 感知侧记 $\mathbf{m}_{\text{perc}}(c)=(\mathrm{AP},\ \ell,\ \tau,\ e,\ s)$，依次为精度、延迟、吞吐、能耗、模型体积（规范度量为 $\mathrm{AP}_{70}$、$\ell_{p50}$、QPS、每帧能量、引擎体积；过参数化区间内 $\mathrm{AP}_{30/50}$ 近饱和故主用 $\mathrm{AP}_{70}$）。驾驶侧记 $\mathbf{m}_{\text{drive}}(c)=(\mathrm{DS},\ \mathrm{RC})$，即时延感知 CARLA 闭环下的驾驶得分与路线完成率。**驾驶指标是本研究的端任务度量、与感知指标并列进入预测器与 Pareto，而非仅作下游验证**：感知侧 AP/延迟只是代理，闭环仿真的全部目的正是把"该配置真实推理时延对驾驶安全的影响"压缩成 $\mathrm{DS}$——优化 $\mathrm{DS}$ 等于直接优化端到端驾驶安全。

**(ii) 前沿是 regime 条件的多目标 Pareto，目标维度随场景而变。** 选择器按部署场景 $\rho$ 把 7 指标划分为**目标集 $\mathcal{O}_\rho$ 与约束集 $\mathcal{S}_\rho$**，仅在目标集上计算支配关系 $c\prec c'$、在约束集上做可行性过滤。**精度轴恒驻目标集、从不充当约束**——本研究的根本命题即"压缩在多大精度代价下换多少加速"，精度是代价的第一度量；即便当前模型/数据上精度轴信息量偏低，也以"前沿沿精度轴塌缩"诚实呈现而非删轴。驾驶安全场景下 $\mathrm{DS}$ 升为目标轴；因 $\mathrm{DS}$ 已整合延迟对安全的代价，该场景把延迟降为约束（$\ell\le\ell_{\max}$）以避免与 $\mathrm{DS}$ 冗余——**延迟与 DS 不在同一 regime 同时充当目标**，冗余由 regime 划分消解。

| 场景 regime | 绑定 | 目标轴 $\mathcal{O}_\rho$（AP 常驻） | 约束轴 $\mathcal{S}_\rho$ |
|---|---|---|---|
| 车载单机 (ego) | 延迟 | $(\mathrm{AP},\ \ell,\ e)$ | $\tau\ge\tau_{\min},\ s\le s_{\max}$ |
| 路侧多车 (RSU) | 吞吐 | $(\mathrm{AP},\ \tau,\ e)$ | $\ell\le\ell_{\max},\ s\le s_{\max}$ |
| 能耗敏感 | 能耗 | $(\mathrm{AP},\ e,\ \ell)$ | $\ell\le\ell_{\max},\ \tau\ge\tau_{\min},\ s\le s_{\max}$ |
| 驾驶安全闭环 | 端任务安全 | $(\mathrm{AP},\ \mathrm{DS},\ e)$ | $\mathrm{RC}\ge\mathrm{RC}_{\min},\ \ell\le\ell_{\max},\ s\le s_{\max}$ |

同一组预测的 7 指标，仅切换"目标/约束"的划分即适配不同部署，无需重训评估器。其中**吞吐是独立维度而非延迟的冗余倒数**：仅在单流、批大小为一且无跨帧流水时退化为 $1000/\ell$，经空间维（多车批处理）或时间维（异构 GPU∥DLA 跨帧流水）解耦后即脱离该式。朝向回归误差等真阳性几何误差项作预测器的剪枝退化参考信号入库，其相对动态范围远小于成本轴，既不作前沿成本轴、也不参与可行性筛选。闭环驾驶指标与感知指标分属不同测量口径（CARLA 闭环 vs 真测引擎），在数据集中以独立列与溯源标记并存。

### 联合搜索器：嵌套双环

在上述目标定义下，搜索器以**嵌套双环**求 regime 条件的 Pareto 前沿，用 ALT cross-exploration 的"上游变 → 下游空间重建"结构把外环软件轴与内环调度轴粘合：

$$
\underbrace{\text{外环 NSGA-II}}_{P\times Q,\ \text{多目标}}\ \xrightarrow{\ \text{每选定 } c\Rightarrow(W,P)\ }\ \underbrace{\text{内环 MetaSchedule}}_{\text{schedule},\ \text{单目标 lat/energy}}\ \xrightarrow{\ \text{回灌最优 }(\ell,e)\ }\ \text{该 }c\text{ 的真实硬件代价}
$$

**外环用 NSGA-II 而非 ALT 的 PPO**：PPO 需大量 on-policy 轨迹，在极小样本上不收敛，且多目标 RL 须标量化奖励（违背"AP 恒驻目标轴、不许加权综合分"的纪律）；NSGA-II 是 population-based 进化，天然多目标、无需可微奖励、与 CHaNAS 进化外环同源。**内环用 MetaSchedule 而非自写退火**：AutoTVM 的并行模拟退火 + XGBoost cost model 的工程化后继即 MetaSchedule（evolutionary_search + xgb_model），内环单目标延迟正是其强项。外环每个候选先查调度 LUT（命中即取 tuned 调度 + 实测代价），未命中才触发内环真搜并回灌 LUT——这把"每候选重搜调度"降为"块级一次性建表 + 在线查表"。

**合法性由两个组件在采样阶段保证**：约束过滤器区分三类硬度——物理硬约束（违反即构建失败，硬过滤）、经验硬约束（实测拐点，硬过滤）、软约束（偏好引导，作惩罚项）；以及 D↔B2 双向传播——被路由到 DLA 的模块强制锁死量化轴（INT8/per-tensor/HWC4），反之选择 per-channel 的模块禁止路由到 DLA。两者均在候选生成时即剔除非法配置，避免把预算浪费在注定构建失败的点上。这构成与纯量化方法的关键差异：**耦合崩溃在搜索的事前由约束与传播规避，而非在量化之后才被发现。**

**延迟的查表组合按拓扑结构选择算子**，而非无条件加法——这是对 CHaNAS 纯加性模型的修正（其加性隐含"块间无重叠、无跨块融合、无同步 barrier"，在我们的真实场景部分成立部分崩坏）：

$$
\textit{Lat}_{\text{net}}(c)=\underbrace{\sum_{\text{串行块 }b}\textit{Lat}_{\mathrm{LUT}}[b,W_b,P_b]}_{\text{(A) 串行段：加性}}+\underbrace{\sum_{\text{融合段 }g}\textit{Lat}_{\mathrm{LUT}}[g(\text{整段, 含 join})]}_{\text{(C) 融合/协同 join 段：整测不拆}}+\underbrace{\sum_{\text{并行段 }p}\big(\max_{\text{branch}}\textit{Lat}_{\mathrm{LUT}}[\cdot]+\textit{handoff}\big)}_{\text{(D) 异构 GPU∥DLA：取 max + handoff}}
$$

LUT 的键须含 `latency_kind`（口径隔离铁律：body_subnet / collab2 / e2e / 板级不可混加）。LUT 是机制加速器而非真值替代——外环 Pareto 前沿 Top-K 仍须在阶段三整网真测验证，LUT 只用于搜索内圈快速排序。

**联合优于串行的设计依据**正是动机节的结构性原理：外环在 $P\times$ schedule 联合空间中能搜到"默认调度下次优、调优后全局最优"的 $P_g$ 分支，而串行先按默认调度锁死 $W_g$ 后该分支被结构性排除。为使这一优势可归因，搜索须遵循"不靠枚举"的纪律：全局对搜索器隐藏（搜索器只能经代价模型 + 有限真测预算访问环境，绝不喂入全局 Pareto）；串行 baseline 须为合理强 baseline（每阶段都取当下最优、非故意选差）。校验通过的 regime 条件前沿——一组带预测性能的部署候选——即为阶段三 Top-K 真测验证的输入。

---

## 图 2 配图说明（Figure 2 caption）

**图 2. 阶段二：软硬件协同优化的嵌套双环搜索架构。** 左侧承接阶段一的分区清单（整合搜索旋钮 + 合法取值 + 逐单元延迟权重 + 硬件能力 $\mathcal{H}$ 摘要）。中部"搜索空间构建"将理论的 $P\times Q\times$ schedule（$\sim\!10^{15}\times O(10^9)$）按三机制压缩为可枚举的非笛卡尔合法空间：两阶段定义（外层软件 $P\times Q$，每选定 $(W,P)$ 重建内层调度子空间 $\Sigma_{\mathrm{sch}}(W,P)$——ALT cross-exploration）、按块分解为线性调度 LUT（$B\cdot|W|\cdot|P|$——CHaNAS R·B·S）、合法性谓词（$\mathrm{in\_per\_g}\in 2^k$ + 精度门控）与三跳依赖传播（剪枝→量化→schedule）。上方为外环 regime 条件 NSGA-II（软件轴 $P\times Q$，多目标）：每个候选 $c$ 经约束过滤（物理/经验硬约束 + 软惩罚）与 D↔B2 双向传播后查调度 LUT，命中即取、未命中触发下方内环 TVM MetaSchedule（evolutionary_search + XGBoost cost model，AutoTVM 后继）真搜该 $(W,P)$ 的最优调度并回灌 LUT。右侧为分层异构代价模型——内层调度复用 MetaSchedule xgb（rank + AST 特征），外层 per-(model,hw) LightGBM（latency/energy rank+log 双头修跨数量级崩塌、AP 预测 $\Delta$AP 残差仅排序用 + 真测漏斗、可行性二分类门、size 解析、跨硬件 global+local 分解）输出 5 感知指标，再经两级头把 $(\widehat{\mathrm{AP}},\widehat{\ell})$ 经闭环标定响应面 $g$ 换算端任务驾驶指标 $(\mathrm{DS},\mathrm{RC})$——合计 7 项 + 不确定性。选择器按 regime $\rho$ 划分目标集 $\mathcal{O}_\rho$（AP 常驻；驾驶安全 regime 以 DS 为目标、延迟降约束）与约束集 $\mathcal{S}_\rho$，仅在目标集做非支配排序、在约束集做可行性过滤，得 regime 条件多目标 Pareto；并以 quantile 宽度 × 前沿邻近度 + ε-greedy 驱动 active-learning 采样回路（虚线）回灌重训。前沿 Top-K 输出至阶段三真测验证。延迟查表组合按拓扑切换加性/max+handoff/整测三算子。

### 图 2 示意（供绘图参考）

```
  阶段一 partition manifest
  (搜索旋钮 + 合法取值 + 延迟权重 + H 摘要)
              │
              ▼
  ┌──────────────────────────────────────────┐
  │  搜索空间构建  理论 P×Q×schedule ~10^15×O(10^9) │
  │   ├─ 两阶段定义: 外软件P×Q / 每(W,P)重建Σ_sch  │  [ALT cross-explore]
  │   ├─ 按块分解: 调度LUT B·|W|·|P| (线性)        │  [CHaNAS R·B·S]
  │   ├─ 合法性谓词 in_per_g∈2^k + 精度门控        │  [CHaNAS divisible+ALT analytic]
  │   ├─ 三跳传播 剪枝→量化→schedule (非笛卡尔)     │  [ALT propagation]
  │   └─ 轴塌缩(对象=channel) + H注入合法取值       │
  │  → 软件侧 ~10^4 + 调度LUT线性                  │
  └───────────────┬──────────────────────────────┘
                  ▼
  ┌─ 外环 NSGA-II (P×Q, 多目标) ────────────┐      ┌─ 分层异构代价模型 ─────────────────┐
  │  候选 c=⟨B1,B2,D⟩                        │      │ 内层(schedule): MetaSchedule xgb   │
  │   ├─ 约束过滤(物理/经验/软)              │ 查LUT │   rank + AST特征 + 进化搜 (复用TVM) │
  │   ├─ D↔B2 双向传播(DLA⇒INT8/pt)         │ ◀──▶  │ 外层(P×Q): per-(model,hw) LGB      │
  │   └─ 查调度LUT 命中取/未命中触发内环 ────┼──────│   lat/energy: rank+log双头(修崩)   │
  │                                         │ 触发  │   AP: ΔAP残差(仅排序)+真测漏斗      │
  │  内环 MetaSchedule (schedule单目标) ─────┼──────▶   feasible分类门 / size解析        │
  │   evo_search+xgb → 最优schedule+(ℓ,e)   │ 回灌  │   跨硬件 global+local 分解          │
  │   → 写回LUT; Lat_net=Σ/max+handoff/整测 │      │ 两级头 g(AP^,ℓ^)→(DS,RC) 闭环标定  │
  │  regime划分 O_ρ(AP常驻)/S_ρ(约束)        │ 7指标 │ → 5感知+2驾驶 + 不确定性(quantile) │
  └───────────────┬──────────────────────────┘+方差 └──────────────┬─────────────────────┘
                  ▼                                                │
   O_ρ 非支配排序 + S_ρ 可行性过滤   ┄┄ acquisition(quantile×邻近+ε) ┄┄▶ 真测/闭环回灌重训
        → regime 条件 多目标 Pareto 前沿
                  │
                  ▼
          前沿 Top-K → 阶段三 真测验证 → 编译 engine (TVM .so)
```
> ★ 平台说明：主搜索轴 = **prune（P）× schedule（S）**，全程在 TVM-MetaSchedule 同一编译栈内（外环改 IRModule/重导 ONNX，内环调 PrimFunc schedule）。量化（Q/INT8）为 future/次要轴：relax 暂无 INT8 pass，INT8 须走 BYOC-TRT 且与 TVM 调度口径分裂，故不进 MVP 主流水线，引入时按独立口径分键、不与 TVM 数混加 $\textit{Lat}_{\text{net}}$。
> 闭环标定：DS/RC 标签来自时延感知 CARLA 闭环（注入真实延迟 → DS/RC），但只用于**标定响应面 $g$**（一族曲线，覆盖若干精度档），而非逐候选真跑；标定后 DS/RC 作端任务目标轴随 $(\widehat{\mathrm{AP}},\widehat{\ell})$ 在线换算，进入 Pareto。

---

## 与三篇文献的机制映射（方法依据一览）

| 文献机制 | 本方法的落地 |
|---|---|
| ALT cross-exploration（上游变 → 下游空间重建） | 两阶段空间定义 + 嵌套双环；外环 $(W,P)$ 定后内环调度空间重建，内外层代价模型解耦、外层标签 = 内层调优后延迟 |
| ALT layout propagation | 剪枝→量化→schedule 三跳依赖传播，合法调度集被传播裁定而非自由搜 |
| ALT "只为复杂算子建空间" | 只为稠密通道耦合核心（conv/grouped-conv/GEMM）建调度空间 |
| ALT-FP/BP 微基准（串行单向传 layout 次优） | 串行贪心结构性局部最优（$W_g$/$P_g$）的因果同构 |
| CHaNAS R·B·S + block-LUT | 按块分解调度 LUT 防指数爆炸；延迟查表组合（加性/max+handoff/整测按拓扑切换） |
| CHaNAS divisible-split（不可整除更差） | 合法性谓词 $\mathrm{in\_per\_g}\in 2^k$ 的他证（对齐约束前移到宽度轴） |
| CHaNAS Fig 2（最优架构随调度漂移） | 默认调度排序 ≠ 调优后排序 → 必漏只在 tuned schedule 显现的 $P_g$ |
| CHaNAS 子空间 CDF 划分 | 评估器估 (AP,latency) 联合 CDF 优先有利子空间建表/搜 |
| CHaNAS one-shot 精度预测 | AP weight-sharing：概念适配但裸权重悲观，仅剔极端点；精确 AP 仍真测 |
| AutoTVM rank loss > regression | 外层 latency/energy 改 lambdarank 序头 + log 回归值头，修跨数量级崩塌 |
| AutoTVM XGBoost cost model（AST 特征） | 内层调度直接复用 MetaSchedule xgb_model，不自建 |
| AutoTVM Algorithm 1 在线回灌 + ε-greedy | 外层 active-learning 闭环（quantile×Pareto 邻近 acquisition + ε 随机） |
| AutoTVM Eq.4 global+local 分解 | 跨硬件迁移 $\hat f_{\text{Orin}}=\hat f^{\text{global}}+\hat f^{\text{local}}_{\text{Orin}}$ |
| 三篇共识：单目标 latency | 本方法扩为多目标 7 指标 + AP 真测漏斗（最大原创扩展） |
