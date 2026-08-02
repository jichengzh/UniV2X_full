# AnonymousSubmission2027 实验结果两页中文草稿（v1）

## 1. 写作目标与篇幅约束

本稿用于规划论文正文的 Experimental Results，目标是在**双栏约两整页**内完成以下证据链：

1. 交代模型、数据集、硬件、后端和统一测量边界；
2. 说明成本模型及目标变换并非经验指定，而是由嵌套式分组交叉验证选出；
3. 围绕统一 Table 1 的 (a)、(b)、(c) 三个部署 profile 说明 GEAR 的主要优势；
4. 从结构—精度—后端耦合角度解释解耦方法在当前实验合同下表现受限的原因；
5. 明确 Orin FP16 control、功耗 rail，以及 TVM 结果具有模型依赖性等结论边界。

考虑到现有结果已合并为一张较大的跨栏主表，正文**不再新增实验平台表或成本模型选择表**。平台合同和模型选择结果均压缩为正文中的高密度段落。建议篇幅分配如下：

- 实验设置与成本模型选择：约 0.45 页；
- Table 1 跨栏表格：约 0.45--0.55 页；
- 三个 panel 的结果分析与结论边界：约 0.85--1.00 页。

核心论点为：

> 在统一精度约束和模型内测量协议下，GEAR 通过联合考虑压缩结构、数值精度与后端实现，在 TensorRT/H800 的三个模型以及 TVM/H800 的 Pyramid、CoDriving 上获得优于解耦和顺序优化策略的可行解；Orin与TVM实验进一步表明，最优部署配置取决于模型、目标硬件和后端能力，不存在能够预先写死并支配所有profile的单一压缩或调度规则。

## 2. 术语与数值规范

- 方法统一写为 **GEAR**；不在正文继续使用 `Joint SHCoSearch` 等旧名称。
- 精度指标统一写为 \(\mathrm{AP}_{70}\)，不再与笼统的 \(\mathrm{AP}\) 混用。
- 对照方法统一为 `Original/default`、`Compression only`、`Schedule only` 和 `Compress \(\rightarrow\) Tune`。
- 所有实数及百分比统一保留小数点后两位；百分比由未舍入的原始测量值计算。
- Table 1 中 (a)、(b)、(c) 分别表示 TensorRT/H800、TensorRT/Orin frozen-subnet 和 TVM/H800。
- Panel (b) 的联合行统一称为 **GEAR-selected structure (FP16 control)**，不得表述为H800 INT8结果的同精度复现。
- 表中仅对满足精度约束的时延和能耗排序：最优值加粗，第二优值加下划线；\(\mathrm{AP}_{70}\) 不参与字体标注。

## 3. 两页正文中文草稿

### 3.1 Experimental Setup and Cost-model Selection

我们在 Pyramid、CoDriving 和 F-Cooper 等多种协同感知模型上评估 GEAR，以验证其通用性。根据不同模型的评测设置，分别采用 DAIR-V2X 和 OPV2V 数据集进行验证，评测样本规模分别为 1789 和 2170条。实验覆盖NVIDIA H800和Jetson AGX Orin硬件，以及TensorRT和TVM后端。为隔离结构与后端实现的影响，时延仅统计各模型冻结部署子网的CUDA计算时间，不包含数据加载和主机—设备传输。H800能耗由NVML板级功率计算，Orin能耗采用同一活动窗口内的`VIN_SYS_5V0`功率；由于二者对应不同的电气边界，本文不报告跨硬件能耗倍率。

成本模型及其目标变换通过外层五折、内层三折的嵌套式分组交叉验证确定。验证使用 176 条真实测量记录，覆盖 44 个完整模型—结构分组，并保证同组样本不跨数据划分。最终结果如图所示。图中横轴为平均内层验证得分（Mean inner-validation score），对候选预测器及目标变换组合\((m,t)\)定义为
\[
\mathrm{MIS}(m,t)=\frac{1}{5}\sum_{k=1}^{5}\frac{1}{3}\sum_{j=1}^{3}S_{k,j}(m,t),
\]
\[
S_{k,j}=\frac{\mathrm{MAE}_{k,j}}
{\max\!\left(Q_{0.9}(y_{k,j})-Q_{0.1}(y_{k,j}),10^{-9}\right)}
+0.25\left(1-\rho_{\mathrm{S},k,j}\right),
\]
其中，\(\mathrm{MAE}_{k,j}\)为该折平均绝对误差，\(Q_{0.9}(y_{k,j})-Q_{0.1}(y_{k,j})\)为真实标签的稳健跨度，\(\rho_{\mathrm{S},k,j}\)为Spearman秩相关系数。第一项衡量归一化预测误差，第二项惩罚候选排序错误；因此平均得分越低，表示预测器及目标变换组合越优。结果表明，采用 log1p 目标的 ExtraTrees 适用于时延和能耗预测，而基于模型锚点残差和 Huber 损失的 LightGBM 适用于AP70预测。


### 3.2 Cross-Hardware and Backend Generalization



尽管已有软件—硬件协同优化方法开始联合搜索网络结构与执行调度\cite{chen2021chanas,xu2022alt}，其优化目标、候选空间和评估指标与本文并不一致，难以直接进行对比。为保证实验的可控性和可复现性，本文按照典型模型部署流程设置四类对比方案。Original/default 表示直接部署原始模型，不进行压缩或调度优化；Compression only 表示仅采用剪枝和量化等模型压缩方法\cite{yang2018netadapt,xiao2023haloc}；Schedule only 保持原始计算图不变，仅通过后端自动调优搜索高效执行实现\cite{chen2018learning,shao2022metaschedule}；Compress \(\rightarrow\) Tune表示现有工程中常见的串行工作流程流程\cite{zhao2025quantv2x,balasubramaniam2025upaq}。

考虑到不同方法在优化目标和结果形式上存在差异，本文采用统一的代表性点比较协议。具体而言：在相同的\(\Delta\mathrm{AP}_{70}\leq0.10\)精度预算下，从每种方法实际产生的结果中选取一个满足约束的代表性最优点，并在相同硬件、后端和测量边界下比较其AP、时延与能耗。该协议既保留各方法自身的优化结果，又避免因目标定义不同而进行不对等的多目标指标比较，具体结果见Table 1。


<!-- 结果显示，在H800与TensorRT条件下，GEAR在所有模型上均取得可行候选中的最低时延和能耗。相对于最强可行解耦基线，GEAR在Pyramid、CoDriving和F-Cooper上平均降低了21.34%的时延和31.93%的能耗。同时，在几乎不损失精度的前提下，选定配置相较于原始部署在三个模型上分别实现了约 (7.16\times)、(6.17\times) 和 (17.35\times) 的推理加速。

同时结果还说明，算法的加速收益并非简单来自 FLOPs 减少。Compression only 在三个模型上均停留于 FP16，并未从 INT8 路径获得稳定收益；同时选择了比GEAR更加保守的通道裁剪策略\(e.g. ...\)。这与后端探针结果一致：单层 INT8 卷积在不进行调度优化的前提下相对 FP16 的时延增加63%. 这是因为更低的位宽并不能保证所有算子可以触发高效的低精度计算单元，同时还会再算子之间引入精度变换带来的额外时间消耗。类似的，更小的计算图并不能保证压缩后的通道形状可以触发高效的TensorRT tactic，并满足 Tensor Core 通道对齐要求。Compress (\rightarrow) Tune 虽然能够在计算图压缩后继续优化算子实现、内存布局等调度优化，但其串行决策机制在计算图冻结后无法将后端实测结果反馈至压缩阶段。因此，该流程与 Compression only 面临相似的问题。即可能在Compress 决策阶段提前排除经过软件—硬件协同优化后实际更高效的通道宽度与数值精度组合。这导致最终搜索得到的最优方案相比于GEAR慢71.88%。Schedule only 仅在固定计算图和数值精度的条件下优化执行调度，无法通过削减冗余计算和降低数值精度来降低理论计算强度。对于复杂网络优化效果有限。 -->

Table~\ref{tab:main-results}(a)显示，以H800为硬件平台，GEAR在所有模型上均取得可行候选中的最低时延和能耗。相对于最强可行解耦基线，GEAR在Pyramid、CoDriving和F-Cooper上平均降低了34.48%的时延和33.79%的能耗。同时，在几乎不损失精度的前提下，选定配置相较于原始部署在三个模型上分别实现了约 (16.86$\times$)、(7.70$\times$) 和 (13.60$\times$) 的推理加速。

同时结果还说明，算法的加速收益并非简单来自 FLOPs 减少。Compression only 在三个模型上均停留于 FP16，并未从 INT8 路径获得稳定收益；同时选择了比GEAR更加保守的通道裁剪策略\(e.g. 在 Pyramid上选择(24,128,64) 而 GEAR选择(16,32,64)\)。这与后端探针结果一致：单层 INT8 卷积在不进行调度优化的前提下相对 FP16 的时延增加63%. 这是因为更低的位宽并不能保证所有算子可以触发高效的低精度计算单元，同时还会再算子之间引入精度变换带来的额外时间消耗。类似的，更小的计算图并不能保证压缩后的通道形状可以触发高效的TensorRT tactic，并满足 Tensor Core 通道对齐要求。Compress (\rightarrow) Tune 虽然能够在计算图压缩后继续优化算子实现、内存布局等调度优化，但其串行决策机制在计算图冻结后无法将后端实测结果反馈至压缩阶段。因此，该流程与 Compression only 面临相似的问题。即可能在Compress 决策阶段提前排除经过软件—硬件协同优化后实际更高效的通道宽度与数值精度组合。Schedule only 仅在固定计算图和数值精度的条件下优化执行调度，无法通过削减冗余计算和降低数值精度来降低理论计算强度。对于复杂网络优化效果有限。

此外，GEAR 的内层优化并不绑定特定部署后端，而是通过统一接口调用与目标硬件相匹配的编译和调优机制。在 NVIDIA 平台上，框架可以采用 TensorRT 完成硬件调度优化以实现最优效果，如Table~\ref{tab:main-results}(b)展所示。同时也支持接入Ansor、MetaSchedule 等更加先进的自动硬件调度方法。

Table~\ref{tab:main-results}(c)显示，在 Jetson AGX Orin 上，GEAR 同样获得了满足精度约束的最优边缘部署结果。感知子网在 Pyramid、CoDriving 和 F-Cooper 上分别实现了 (12.12\times)、(12.06\times) 和 (14.63\times) 的加速，根据figure2中展示的既有剖析结果。在驾驶传输时延为 (50.00) ms 的条件下，Amdahl 折算表明，GEAR 可将三个模型的系统推理时延提升1.74$\times$~2.59$\times$。基于 Figure 1(a) 中闭环仿真结果进行情景映射。针对计算密集型的F-Cooper模型将端到端时延849.53 ms降低到328.44 ms 预计可以获得49.09%的DS提高。针对计算轻量型Pyramid和CoDriving，其本身端到端时延位于相对稳定的时延平台区间。GEAR虽然不能带来明显的DS提升，但是扩大相对于时延崩溃节点的安全裕量，让系统面对通信波动更加稳定，同时支持更加低成本的边缘计算平台。


<!-- TVM实验进一步检验了后端泛化性。在Pyramid上，GEAR相对于Compression only将时延和能耗分别降低80.05%和76.62%；在CoDriving上，相对于最强顺序基线Compress \(\rightarrow\) Tune分别降低18.03%和19.23%。但在F-Cooper的TVM配置上，Compress \(\rightarrow\) Tune优于GEAR，说明联合搜索并不支配所有模型—后端组合。更重要的是，GEAR在TensorRT上选择INT8，而在TVM上选择FP16，表明方法并未预设低精度必然更快，而是依据目标后端的算子降级能力和真实执行代价选择结构—精度组合。固定原始图的调度调优只能带来有限收益，显著加速仍需要结构、精度与后端实现之间的反馈耦合。跨后端结果因此支持“能力条件化搜索”的结论，而不支持某一后端或某一配置在所有场景下均最优的泛化表述。 -->

### 3.3 Online Search Component Ablation

| 变体 | 代理模型引导 | 在线实测反馈 | 后端感知特征 | $\Delta$HV-AUC $\uparrow$ | 相对 GEAR-core |
|---|:---:|:---:|:---:|---:|---:|
| **GEAR-core** | ✓ | ✓ | ✓ | **15.90 $\pm$ 4.47** | — |
| w/o surrogate | — | ✓ | ✓ | 10.50 $\pm$ 6.14 | $-5.40$ |
| w/o measured feedback | ✓ | — | ✓ | 8.44 $\pm$ 0.62 | $-7.45$ |
| Backend-blind features | ✓ | ✓ | — | 5.87 $\pm$ 2.51 | $-10.03$ |

**Table 2：在线搜索组件消融。** `✓` 和 `—` 分别表示启用和移除相应组件；数值为三个随机种子的均值 $\pm$ 标准差，最后一列为相对 GEAR-core 的变化。GEAR-core 不包含尚未纳入的候选级能力扫描器。

为区分代理模型引导、在线实测反馈和后端感知表征对有限预算搜索过程的独立贡献，Table 2 对 GEAR 的三个外层搜索组件进行了消融。本实验采用 $\Delta$HV-AUC 衡量完整搜索轨迹上的 Pareto 前沿扩张效率，其计算为

$$
\Delta\mathrm{HV}_{\tau}
=\left[\mathrm{HV}\!\left(\mathcal{D}_{\mathrm{out}}^{\tau}\right)
-\mathrm{HV}\!\left(\mathcal{D}_{\mathrm{out}}^{0}\right)\right]_+.
$$

$$
\Delta\mathrm{HV}\text{-}\mathrm{AUC}
=2\sum_{\tau=1}^{4}
\left(\Delta\mathrm{HV}_{\tau-1}+\Delta\mathrm{HV}_{\tau}\right).
$$

其中，$\mathcal{D}_{\mathrm{out}}^{0}$ 和 $\mathcal{D}_{\mathrm{out}}^{\tau}$ 分别表示冷启动数据与截至第 $\tau$ 轮的累计实测数据；$\mathrm{HV}(\cdot)$ 表示相对于参考点、在 $(\mathrm{latency},\mathrm{energy},-\mathrm{AP}_{70})$ 三目标空间中计算的超体积；$[\cdot]_+$ 表示仅保留括号内差值的非负部分。每轮包含四个选择名额，因此预算节点为 $(0,4,8,12,16)$，梯形积分系数化简为 2。该指标同时度量前沿扩张的幅度和出现时间；越早发现高质量 Pareto 候选且前沿扩张越大，$\Delta$HV-AUC 越高。

实验结果显示，删除后端感知特征造成最大 $\Delta\mathrm{HV}\text{-}\mathrm{AUC}$ 降幅 63.1%，表明缺少 backend/capability/profile 信息时，代价模型难以辨别不同配置在目标硬件上的执行差异，进而使外层搜索难以发现高潜力方案。
在另外两项独立消融中，移除代理模型引导并改用均匀随机采样后，采样点不再按照预测的精度—时延—能耗权衡进行优先排序，导致有限测量预算难以集中于高潜力候选，从而使该指标下降 34.0%；冻结在线实测反馈后，代价模型无法利用新增的真实图特征和硬件测量持续修正冷启动预测误差，使该指标下降 46.9%。

## 4. 压缩时必须保留的句子

如果英文翻译后超出两页，优先删除实现性细节，但应保留以下四类信息：

1. 三个模型的数据集、AP样本数和不同部署子网边界；
2. `5-fold outer / 3-fold inner`分组交叉验证及三个最终预测器；
3. Table 1中GEAR相对最强解耦基线的六个改进百分比；
4. Orin的FP16 control与功耗rail边界，以及TVM中TensorRT选择INT8、TVM选择FP16和F-Cooper未由GEAR取胜的核心观察。

可优先压缩或移至附录的内容：

- 每个预测器的MAPE解释可缩为一句；
- Orin中间特征误差的具体数值可移至附录；
- Tensor Core粒度、内存布局等机制可合并为“hardware-unaware execution inefficiency”；
- TVM/F-Cooper中GEAR与Compress \(\rightarrow\) Tune的具体差值可移至附录，但必须保留“GEAR并非在所有profile上取胜”的边界。
