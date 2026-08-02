# CoptV2X: Coupling-Aware Bilevel Optimization for V2X Perception Model Compression and Backend Scheduling——附录中文审阅稿

> 本文件为中文审阅稿，暂不合并至论文主文件。

## 附录 A：Figure 1 与 Figure 2 的测量方法

在 Figure 1(a) 所采用的零注入 V2XVerse 闭环基线中，CARLA 以 20 Hz 同步模式推进，即每个仿真步对应 $\delta_{\mathrm{sim}}=50$ ms。系统在每个仿真步采集自车状态与传感器观测，并按固定每四个仿真步执行一次完整的协同感知与规划。该四帧间隔只是 V2XVerse 触发完整推理的原生策略，不改变时延注入模块以 50 ms 仿真步为单位的时间索引。执行推理时，自车、其他协作车辆和路侧单元的观测首先被组织为 CoDriving 的多智能体输入，随后依次完成特征提取、空间对齐与协同融合、目标检测、占用表示构建和轨迹规划，并由控制器将预测轨迹转换为转向、油门和制动指令。在两次完整推理之间，车辆依据最近一次可用规划结果和当前自车状态更新控制；控制指令返回后，CARLA 才推进到下一仿真步。因此，推理计算耗时不会改变后续车辆状态演化，该基线难以直观呈现推理时延对闭环驾驶行为与行驶安全的影响。

相较于上述基线，本文在“原始观测采集”与“感知预处理”之间增加了自车观测时延注入模块，而感知、融合、规划和控制模块本身保持不变。历史帧库以仿真步编号，并独立于完整推理的触发策略在每个 50 ms 仿真步更新。在仿真步 $k$，该模块将当前自车输入拷贝至历史帧库，并根据名义感知时延 $\tau_{\mathrm{perc}}$ 计算目标延迟 $\Delta_{\mathrm{perc}}=\lceil\tau_{\mathrm{perc}}/\delta_{\mathrm{sim}}\rceil$。当仿真步 $k$ 触发完整推理时，模块选择满足 $j\leq k-\Delta_{\mathrm{perc}}$ 的最新缓存帧 $j$ 替换当前自车输入。由此，从感知特征提取到控制输出的整个下游链路均基于历史自车观测作出决策，而仿真世界继续按正常时钟演化，该机制模拟了感知信息滞后带来的影响。

Figure 1(a) 在 CoDriving、NVIDIA H800 和 V2XVerse full-traffic 配置下测试上述注入机制。名义时延取 0--700 ms、步长 50 ms，并增加 800、900 和 1,000 ms 三个高时延压力点；这一取点方式原用于密集观察退化起点并覆盖高时延失效区。实验首先从候选路线中筛选零时延下至少两次重复达到 DS $\geq99.9$、无碰撞且无超时的路线，再取所有时延配置均有记录的路线交集，最终得到 35 条 matched routes。每条路线独立运行 3 次。图中蓝线为 Honest Driving Score：采用CARLA Leaderboard默认的计算方法确定，超时或未产生有效分数的运行按 DS=0 计入均值，误差条为 $1.96s/\sqrt{n}$ 计算的 95% 置信区间。红线为已完成运行中的碰撞比例，只要行人、车辆或场景布局碰撞任一项非零即记为碰撞。紫线表示全部运行中因超时未完成评估或未生成有效 DS 的运行比例，这通常意味着仿真过程中发生了严重的事故导致车辆停在了路线中间，没有到达终点。


Figure 2 将 V2X 感知流水线划分为输入编码、Backbone + Neck、特征融合和检测四个阶段，本附录沿用相同的四阶段计时合同，以统一 Pyramid、CoDriving 和 F-Cooper 的模块归因。对于 LiDAR 模型，输入编码包含体素化或柱编码及 scatter；对于相机模型，该阶段对应图像处理前端。Neck、deblock、上采样和 shrink convolution 统一计入 Backbone。检测阶段则同时包含预测头和后处理：前者包括分类、框回归和方向预测，后者包括阈值过滤、top-$k$、框解码、方向修正、角点转换、旋转 NMS 和范围过滤。

Figure 2 与正文 Table 1 使用不同的硬件、执行后端和计时对象。Figure 2 的阶段分解在 NVIDIA GeForce RTX 4090 上采用原生 PyTorch/CUDA 路径完成，用于分析完整感知流水线中各模块的计算占比。模块计时从已经位于 GPU 上的输入开始，因此数据集读取、CPU 预处理、Host-to-Device 传输和进程间通信不属于 Figure 2 的测量边界。各模块使用 CUDA events 记录起止时间，并在读取 elapsed time 前执行设备同步；完整区间围绕相同的设备端 forward 与后处理路径测量。阶段表报告算术平均值，同时在底层记录中保留 median 和 p99。profiling 协议采用 50 次预热和 200 次正式测量。CoDriving 的输入编码与检测项采用边界一致的 profiling 数值，以保持四阶段归因完整。


| 模型 | 输入编码 | Backbone + Neck | 特征融合 | 检测（预测头 + 后处理） | 已归因总时延 |
|---|---:|---:|---:|---:|---:|
| Pyramid | 2.40 ms（13.80%） | 10.02 ms（57.65%） | 2.05 ms（11.79%） | 2.91 ms（16.76%） | 17.39 ms |
| CoDriving | 2.40 ms（16.49%） | 8.40 ms（57.75%） | 0.81 ms（5.58%） | 2.94 ms（20.18%） | 14.55 ms |
| F-Cooper | 2.98 ms（14.32%） | 10.34 ms（49.67%） | 1.57 ms（7.52%） | 5.93 ms（28.49%） | 20.82 ms |

> **表注：** CoDriving 的输入编码与检测项采用边界一致的 profiling 数值，以保持四阶段归因完整。

## 附录 B：CoptV2X 双层软硬件协同搜索

正文已经给出搜索空间构建、代价模型结构和候选生成机制，因此本附录仅将 Figure 5 的双层数据流写成可复现的算法步骤。算法沿用正文符号：$\Theta_{\mathrm{sw}}$ 为场景相关的合法软件空间，$\mathcal{D}_{\mathrm{out}}^0$ 为冷启动外层观测，$R$ 和 $B$ 分别为外层迭代次数和每轮候选数，$\mathcal{C}_{\mathrm{in}}$ 为后端相关的内层停止准则。外层先由 NSGA-II 生成覆盖不同多目标权衡的候选，再通过预测 Pareto 前沿筛选形成真实测量批次；内层调优则由目标后端的统一接口实现。

**算法 1：CoptV2X 双层软硬件协同搜索**

```text
输入：合法软件空间 Θ_sw，冷启动外层观测 D_out^0，
      外层迭代次数 R，每轮候选数 B，目标部署场景 ρ，
      后端相关的内层停止准则 C_in
输出：由真实测量记录构成的 Pareto 集 P*_ρ

1:  for τ = 1, ..., R do
2:      (E_AP^τ,E_ℓ^τ,E_e^τ) ← FIT-OUTER(D_out^(τ−1))
3:      φ(x) ← [φ_cfg(x),φ_graph(x)],  ∀x ∈ Θ_sw
4:      C_τ ← NSGA-II(Θ_sw,−E_AP^τ,E_ℓ^τ,E_e^τ)
5:      X_τ ← FRONTIER-SCREEN(C_τ,B,D_out^(τ−1))
6:      ΔD_out^τ ← ∅
7:      for each x ∈ X_τ do
8:          Θ_hw(x,ρ) ← CONDITIONAL-SPACE(x,ρ)
9:          s_x ← INNER-TUNE(x,Θ_hw(x,ρ),C_in)
10:         y_x ← MEASURE_ρ(x,s_x) = (AP(x),ℓ(x,s_x),e(x,s_x))
11:         ΔD_out^τ ← ΔD_out^τ ∪ {((x,s_x),y_x)}
12:     end for
13:     D_out^τ ← D_out^(τ−1) ∪ ΔD_out^τ
14: end for
15: P*_ρ ← PARETO({((x,s),y) ∈ D_out^R | (x,s) ∈ Θ_ρ},(−AP,ℓ,e))
16: return P*_ρ

17: procedure INNER-TUNE(x,Θ_hw(x,ρ),C_in)
18:     if backend(ρ) = default then
19:         D_in ← INITIAL-PROBES(x,Θ_hw(x,ρ))
20:         while ¬C_in(D_in) do
21:             E_in ← FIT-XGBOOST(D_in)
22:             s ← SIMULATED-ANNEALING(Θ_hw(x,ρ),E_in)
23:             D_in ← D_in ∪ {(s,MEASURE-LATENCY_ρ(x,s))}
24:         end while
25:         return arg min_{(s,ℓ_s)∈D_in} ℓ_s
26:     else
27:         return NATIVE-BACKEND-TUNE(x,Θ_hw(x,ρ),C_in)
28:     end if
29: end procedure
```

`FRONTIER-SCREEN` 优先保留预测非支配或接近预测 Pareto 前沿的候选，并结合候选多样性选择 $B$ 个尚未测量的软件配置。每轮的 $B$ 个候选完成后，其 $\mathrm{AP}$、时延和能耗记录作为一个完整批次并入 $\mathcal{D}_{\mathrm{out}}^\tau$，再用于下一轮代价模型更新。`INNER-TUNE` 在 default profile 中采用正文所述的 XGBoost 时延预测与模拟退火，而 TensorRT profile 调用其原生 tactic/timing 选择机制；因此，XGBoost 与模拟退火不是所有后端共享的固定实现。

## 附录 C：对比流程与消融设置

本节仅定义正文主结果表中的部署流程和在线搜索消融，数据集、代价模型选择以及各平台的软件环境分别在正文、附录 D 和附录 E 中说明。主结果表中的每一行表示一条完整的部署流程，而不是在同一结果上依次叠加若干独立优化算子。所有流程采用相同的软件候选定义、目标硬件、部署后端、精度约束和测量边界，但软件配置与后端配置的确定方式不同。

Original/default 固定原始模型和数值精度，并采用后端默认配置完成部署。Compression only 根据软件侧指标选择压缩候选，随后使用标准后端配置进行测量，不执行额外的后端调优。Schedule only 保持原始模型结构不变，仅搜索更优的后端执行配置。Compression $\rightarrow$ Tune 先确定并锁定压缩候选，再针对该候选执行后端调优。CoptV2X 则根据批次级真实硬件反馈联合更新软件候选选择和条件化后端调优，从而在搜索过程中同时考虑精度、时延和能耗。

在固定总预算下，Compression only 将预算用于覆盖更多压缩候选，而 Compression $\rightarrow$ Tune 将较多预算集中于少量候选的后端调优，因此后者即使改善了已选候选的执行配置，也可能因软件候选覆盖范围较小而得到较差的最终代表点。

所有方法采用相同的代表点选择协议：在相对于原始部署的 $\Delta\mathrm{AP}_{70}\leq0.10$ 约束下，从每种方法中选择时延最低的可行输出。构建或运行失败的候选不参与代表点选择。各方法的结果均来自各自预先定义的候选集合和预算，因而 Compression only 与 Compression $\rightarrow$ Tune 并不是同一候选在调优前后的成对比较；在观察最终评测结果后再取两条流程中的较优结果，会形成具有更大有效搜索预算的混合基线，而不再是原定义的串行流程。

正文消融实验分别考察代价模型引导、在线实测反馈和图特征的作用。所有变体使用相同的候选池、冷启动数据、真实硬件测量流程、随机种子和搜索预算，仅移除被考察的组件。完整方法在每批真实测量完成后更新代价模型并选择下一批候选；w/o cost model 以均匀随机采样替代代价模型引导；w/o feedback 冻结由冷启动数据训练的代价模型，不使用后续实测结果更新模型或候选排序；w/o graph features 保留代价模型和在线更新，但仅使用剪枝与量化配置作为预测输入。

上述三个消融臂分别用于检验代价模型相对于随机选点的有效性、在线硬件反馈对候选排序的修正作用，以及图特征对不同部署配置执行差异的表征能力。具体结果及组件组合已在正文消融表中报告，相关运行次数和统计口径见附录 E。

## 附录 D：代价模型与目标变换选择

本文通过五个外层折和三个内层折的 nested grouped cross-validation 联合选择代价模型类别与目标变换。该分析使用 176 条真实测量，覆盖 44 个完整的 model--structure groups；在每一次划分中，同一 group 的全部样本始终位于训练侧或验证侧的同一侧，从而避免同一模型结构的信息跨折泄漏。

在每个 outer fold 内，首先根据三个 inner folds 上的平均得分对 predictor--transformation pairs 进行排序。排名第一的组合随后在完整 outer-training set 上重新拟合，并仅在 held-out outer fold 上评估。Figure D.1 展示了最终选择结果；其横轴为五个 outer splits 上聚合的 mean inner-validation score：

$$
\mathrm{MIS}(m,t)=\frac{1}{5}\sum_{u=1}^{5}\frac{1}{3}\sum_{v=1}^{3}S_{u,v}(m,t),
$$

其中，$(m,t)$ 表示 predictor 与 target-transformation 的组合，$S_{u,v}$ 表示 outer fold $u$ 中 inner fold $v$ 的 fold-level score：

$$
S_{u,v}
=\frac{\mathrm{MAE}_{u,v}}{\max(\Delta_{u,v},10^{-9})}
+0.25\left(1-\mathrm{SRCC}_{u,v}\right).
$$

其中，$\mathrm{MAE}_{u,v}$ 为 inner fold 上的 mean absolute error；$\Delta_{u,v}=Q_{0.9}(y_{u,v})-Q_{0.1}(y_{u,v})$ 为 inner fold $(u,v)$ 的 robust label span；$\mathrm{SRCC}_{u,v}$ 为 Spearman rank correlation coefficient。第一项衡量归一化预测误差，第二项惩罚错误的候选排序，因此该分数越低表示预测性能越好。该选择准则只用于比较同一目标指标下的 predictor--transformation pairs，不用于比较 latency、energy 与 $\mathrm{AP}_{70}$ 三种不同标签之间的绝对难度。

该实验共比较 24 个 predictor--transformation pairs，并固定使用 seed `20260716` 完成一次 5-by-3 nested grouped-CV run。Latency 和 energy 最终均选择使用 `log1p` target 的 ExtraTrees regressor；$\mathrm{AP}_{70}$ 最终选择采用 Huber loss 和 model-anchored residual target 的 LightGBM predictor。最终 ExtraTrees 使用 160 trees、maximum depth 8、minimum leaf size 2 和 0.8 feature subsampling；LightGBM 使用 120 trees、learning rate 0.04、7 leaves、maximum depth 4、minimum child size 4，以及 0.1/0.1 的 $\ell_1/\ell_2$ regularization。上述组合、fold protocol、seed 和最终参数在在线搜索开始前冻结。

**Figure D.1：Nested grouped cross-validation 下的代价模型与目标变换选择。** 横轴为 mean inner-validation score；同一目标指标内数值越低越好，但不同目标指标之间不可直接比较。图文件为 `Latex/Figures/cost_model_selection.pdf`。

正式英文附录使用 `draft/final/cost_model_selection.pdf` 作为该图的唯一图源，并已将其原样复制到 `Latex/Figures/cost_model_selection.pdf`。该文件为横向三联图，原始宽度约 12.27 in；若整体压缩至 AAAI 单栏，图中文字约缩小至 5 pt，无法满足投稿可读性。因此当前采用双栏 `figure*`。若后续必须改为单栏，应将三个面板拆分并分别排版，而不应整体缩小。

> **图源核查：** 指定 PDF 的字体已嵌入，双栏排版后标签和坐标轴可读；但 AP70 面板最下方数值 `0.207` 紧贴右侧裁切边界。该现象来自原始 PDF，本稿按“正式图片”要求未擅自重绘。若最终导出源文件仍可调整，建议仅增加 AP70 面板的右侧留白，并保持全部数值不变。

最终 LaTeX 合并时应保留两个独立公式及其标签，并使用以下图环境：

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Figures/cost_model_selection.pdf}
\caption{Cost-model and target-transformation selection under nested grouped
cross-validation. The horizontal axis reports the mean inner-validation
score. Lower scores are better within the same target, but scores are not
comparable across targets.}
\label{fig:cost-model-selection}
\end{figure*}
```

本附录明确给出数据量、group 数、fold 数、候选组合数量、选择准则、随机种子、最终 predictor--transformation pairs 和对应超参数，用于支撑 reproducibility checklist 中关于 hyperparameter selection、random seed setting、evaluation criterion 和 algorithm-run protocol 的说明。模型训练、量化和 backend-internal tuning 的完整参数不在本附录中逐项展开，因此“全部最终超参数”仍应保持为 `partial`。

## 附录 E：复现性清单补充说明

本节集中补充正文和前述附录中尚未完整展开的复现信息，包括场景相关候选空间的实例化结果、搜索预算、随机种子、独立运行次数、统计口径、评价指标和计算环境。数据集、基础方法及其完整引用已在正文给出，不在本附录重复列示。代码实现与发布信息由随论文提交的独立代码仓库提供，本附录不再重复维护代码清单或提交待办。

### E.1 场景相关候选空间

CoptV2X 不预设固定的剪枝层数、剪枝维度或剪枝率。对于每个 model--backend profile，计算图扫描首先识别通道依赖耦合的结构组，随后由结构合法性、模型物化和后端能力约束共同生成合法宽度配置。软件候选仍记为 $x=(p,q)$，其中 p 是扫描与过滤后得到的具体通道配置，而不是预先指定的剪枝率向量；$q$ 是目标后端支持的数值精度。在本文使用的 NVIDIA profiles 中，保留的精度候选为 FP16 和 INT8。

| 模型实例 | 图扫描得到的剪枝结构组 | 量化旋钮 |
|---|---|---:|
| Pyramid | 3 个 Backbone 结构组，每组包含 7 个合法宽度选项 | FP16,INT8  |
| CoDriving | 3 个 Backbone 结构组，每组包含 7 个合法宽度选项 | FP16,INT8 |
| F-Cooper | 5 个结构组，包括 3 个 Backbone 组和 2 个 Neck 组 | FP16,INT8 |

上述结构组数量和候选数量仅描述本文三个实验实例的扫描结果，不是 CoptV2X 预设的网络超参数。对于新的模型、硬件或后端，结构组及其合法宽度集合均由同一图扫描与能力过滤流程重新生成。模型训练和后端硬件调度搜索中未枚举的参数在这里并没有逐个枚举。

<!--
E.1 候选空间复核索引（供草稿审阅，不进入最终排版）：
1. Pyramid 的三个 7 值宽度轴与完整网格：
   framework/feasibility_gate.py
2. Pyramid 的 50 个结构仅为 60 个已物化结构排除 10 个 already_measured
   后的在线未测子集，不代表完整设计空间：
   results/stage5_single_target_search_v2_20260718/S5-PYR-TVM/candidate_manifest.json
3. CoDriving 的 343 个结构及其来源注册表：
   results/stage5_two_model_search_v1_20260717/candidate_source_registry.json
4. CoDriving 的 320 个结构为排除 21 个 already_measured 和 2 个
   frozen_independent_holdout 后的在线未测子集：
   results/stage5_single_target_search_v2_20260718/S5-COD-TVM/candidate_manifest.json
5. F-Cooper 五组定义、1,792 个结构与 3,584 个 precision genomes：
   results/fcooper_workpackage_a_20260723/contracts/candidate_source_registry.json
   results/fcooper_workpackage_a_20260723/contracts/frozen_contract.json
-->



### E.2 随机种子、算法运行次数与统计报告

本节出现的 `Stage` 编号仅用于对应代码仓库中的内部工作包，并不表示 CoptV2X 由这些阶段顺序组成。下文均先说明相应功能，仅在定位实现或归档记录时保留内部编号。

#### E.2.1 随机种子绑定

**代价模型选择。** 代价模型类别和目标变换通过 nested grouped cross-validation 确定，整个过程采用固定的基准 seed `20260716`。该 seed 同时控制外层与内层的分组划分、各候选预测器的训练以及所选预测器的最终重拟合。ExtraTrees 和 LightGBM 均采用由该基准种子确定的固定随机初始化，并以单线程方式运行，以减少并行计算引入的不确定性；针对不同预测目标训练的分位数预测模型则使用确定性派生的独立随机种子。因此，在数据、配置和软件环境保持一致的条件下，模型选择过程能够重复执行，无需在正文中进一步列出与搜索算法无关的折编号或模型编号。

**主实验中的在线候选搜索。** 每个 model--backend 搜索任务从其 manifest 读取固定的 base seed。初始代价模型使用该 seed，后续各轮重拟合仅根据轮次确定性地派生新的 seed。基于 predicted frontier 和 diversity 的生产候选选择器本身不调用 NumPy 或 PyTorch 的随机数生成器；校准 group 则通过稳定哈希排序确定。作为对照，`w/o cost model` 使用由同一任务 seed 初始化的局部随机数生成器，在尚未评测的候选中进行均匀无放回采样，因此不会受到进程级全局随机状态的影响。

**在线搜索组件消融。** 三条独立消融轨迹分别采用 `20260718`、`20260719` 和 `20260720`。所有消融变体复用这组三个 seed，并沿用与主搜索相同的逐轮派生规则，使不同变体能够在配对的随机条件下进行比较。这里的 seed 数量表示独立搜索轨迹的数量，不能与同一候选的 warm-up、timing iterations 或重复测量次数混为一谈。

**训练脚本与确定性边界。** 恢复训练脚本同时设置 Python、NumPy、PyTorch CPU 和 CUDA 的随机 seed，并关闭 `cudnn.benchmark`、启用 `cudnn.deterministic`。由于当前实现尚未强制所有算子仅采用确定性实现，本文仅声称各主要随机入口均已受到显式控制，不保证在不同 CUDA、cuDNN版本或硬件环境下获得逐位完全一致的结果。

<!-- 上述设置分别对应 `framework/stage4/cost_model_selection_v1.py`、`framework/stage5/production_search_v1.py`、`framework/stage7/online_component_ablation_v1.py`、`framework/stage7/search_policy_v1.py` 和 `scripts/stage5_advance_task_round_v2.py`。最终匿名代码附件仍需保留这些入口、运行配置和版本信息；在附件尚未完成前，已提交 checklist 中的 random-seed 项仍保持为 `partial`。 -->

#### E.2.2 独立算法运行次数

本文严格区分独立算法运行、独立测量进程和进程内计时重复。独立算法运行是指从一个独立 seed 或独立初始化开始，并完成全部搜索或训练预算的一次完整执行。独立测量进程用于复核同一个已冻结的部署配置，而 warm-up、正式计时迭代和同一进程内的重复执行仅用于估计测量分布；后两类重复均不增加算法运行次数。这主要是考虑到真实硬件搜索涉及反复编译、部署和测量成本过高。与此同时，本方法在不同模型和后端上的单次生产搜索中均表现出一致且幅度较大的优势，表明主要结论不太可能由某一特定随机种子的偶然波动所解释。

**主实验比较（Table 1）。** 对于模型、后端和比较方法的每种组合，Table 1 报告一次完整搜索运行中按照统一准则选出的代表点。候选选定后的重复硬件测量仅用于稳定估计 AP、时延和能耗，不能视为额外的搜索运行。当前记录已覆盖最终表格的数据来源以及 Pyramid、CoDriving 和 F-Cooper 部分代表点的独立测量复核；提交前仍需确保每个代表点均能与其搜索过程、候选配置和原始测量结果一一对应。

**在线组件消融（Table 2）。** 每个消融变体采用三个独立 seed，因此形成三次独立搜索运行。每次运行包含四轮在线搜索，每轮选择四个候选，即每个 seed 对应 16 次真实硬件评测。四个消融变体共需完成 12 次独立搜索运行。最终结果报告每个 seed 的 $\Delta\mathrm{HV}$-AUC、三个 seed 的均值和标准差，以及相对于完整方法的配对差值。

**闭环时延注入实验（Figure 1(a)）。** 每个感知时延设置均通过独立的闭环 episode 进行评测。该图包含 18 个时延设置，每个设置均由 35 条 matched routes 分别重复 3 次，共计 105 个 episodes。Honest DS 根据 episode 级样本计算均值和 95% confidence interval。这些 episodes 属于闭环仿真的独立运行，不应与部署候选的进程内计时重复混淆。

**压缩配置时延实验（Figure 1(b)）。** 每个通道配置与数值精度的组合形成一条独立 benchmark 记录。该图包含 60 个通道配置和三种数值精度，共计 180 条时延记录。其中每条记录重复测量三次。这 180 条记录表示不同的固定待测配置，而不是 180 次独立搜索。

**流水线时延分解（Figure 2）。** Figure 2 对 Pyramid、CoDriving 和 F-Cooper 采用同一四阶段归因合同，并报告输入编码、Backbone + Neck、特征融合以及检测的时延占比。三种模型沿用相同的硬件类别、软件路径、计时边界和统计口径；CoDriving 的输入编码与检测项采用边界一致的 profiling 数值，以保持四阶段归因完整。

**代价模型选择。** 代价模型选择执行一次固定 seed 的 nested grouped cross-validation。该过程使用 176 条真实测量，覆盖 44 个完整的 model--structure groups。

部分代表点还通过三个独立测量进程进行了复核，Orin 上的 F-Cooper 也采用了单独的 warm-up、重复测量和计时迭代设置。这些数量仅描述相应候选的硬件测量协议，不能推广至 Table 1 的全部结果，也不能用于增加算法运行次数。在线组件消融报告逐 seed 结果、均值、标准差和相对于完整方法的配对差值，但不进行统计显著性声明；Figure 1(a) 则报告 episode 级 95% confidence interval。

### E.3 评价指标与选择动机

本文从感知质量、实时执行成本和能效三个维度评价部署候选。$\mathrm{AP}_{70}$ 衡量 3D detection 在 IoU threshold 0.70 下的平均精度，用于约束模型压缩造成的检测质量损失，并与正文采用的 V2X 感知评测口径保持一致。Latency 以毫秒衡量目标设备上的实际执行时间，直接反映部署方案能否满足实时推理需求。Energy 以焦耳衡量相同 active inference window 内的单次推理能耗，用于区分具有相近时延但功率开销不同的部署方案。三项指标均来自真实部署测量，而不以 FLOPs、参数量或理论算力作为替代，并且仅在硬件平台、后端和测量边界一致时进行比较。

由于上述三个目标之间不存在预先给定的固定权重，本文使用 hypervolume（HV）综合评价 Pareto 解集的质量，避免通过人为选择单一代表点掩盖不同的精度--时延--能耗权衡。在线搜索消融进一步采用 $\Delta\mathrm{HV}$-AUC，同时衡量 Pareto 前沿相对于冷启动前沿的扩张幅度及其出现时间。下面给出 HV 及 $\Delta\mathrm{HV}$-AUC 的完整计算过程。对于任一成功完成真实硬件评测的候选 $c$，首先将三个目标统一写成最小化形式：

$$
\mathbf{z}(c)
=\left(\ell(c),\,e(c),\,-\mathrm{AP}_{70}(c)\right),
$$

其中 $\ell(c)$ 和 $e(c)$ 分别表示时延与能耗。按照正文口径，HV 并不直接在上述原始数值上计算，而是先映射到任务相关的归一化目标空间：

$$
\widetilde{\mathbf{z}}(c)
=\mathcal{N}_{\mathrm{task}}\!\left(\mathbf{z}(c)\right).
$$

这里 $\mathcal{N}_{\mathrm{task}}$ 表示针对当前 model--hardware--backend 任务冻结的逐维归一化；同一任务的全部消融变体、随机种子和预算检查点均使用相同参数。若 $\widetilde{\mathbf{z}}(c_i)$ 在三个维度上均不大于 $\widetilde{\mathbf{z}}(c_j)$，并且至少在一个维度上严格更小，则称 $c_i$ 支配 $c_j$。从累计成功测量集合 $\mathcal{D}$ 中删除被支配点后得到非支配集合 $\mathcal{P}(\mathcal{D})$。编译失败、数值验证失败或未获得完整目标值的候选仍消耗相应搜索预算，但不进入非支配集合，也不贡献超体积。

归一化目标空间中的任务相关参考点 $\widetilde{\mathbf r}$ 在搜索开始前固定，并在同一任务的全部变体和 seed 之间保持不变。参考点不会根据某个消融变体的在线结果重新调整，从而保证不同变体使用相同的 HV 边界。

令 $\mathcal{P}_{\widetilde{\mathbf r}}(\mathcal{D})$ 表示三个维度均严格优于参考点的非支配候选集合。每个有效非支配点 $\widetilde{\mathbf{z}}(c)$ 与参考点 $\widetilde{\mathbf r}$ 共同确定一个三维长方体：

$$
\mathcal{B}_{\widetilde{\mathbf r}}(c)
=\prod_{k=1}^{3}[\widetilde z_k(c),\widetilde r_k].
$$

超体积定义为这些长方体并集的体积：

$$
\mathrm{HV}(\mathcal{D})
=\lambda_3\!\left(
\bigcup_{c\in\mathcal{P}_{\widetilde{\mathbf r}}(\mathcal{D})}
\mathcal{B}_{\widetilde{\mathbf r}}(c)
\right),
$$

其中 $\lambda_3(\cdot)$ 表示三维体积。实际计算时先按时延坐标对非支配点排序，再沿时延轴累积能耗--负精度平面上的二维支配面积，因此重叠区域只计算一次。只有能够扩展现有 Pareto 前沿的候选才会增加 HV。

令 $\mathcal{D}_{\mathrm{out}}^\tau$ 表示冷启动集合与截至第 $\tau$ 轮获得的全部成功在线测量之并集，则

$$
\Delta\mathrm{HV}_{\tau}
=\max\!\left(
0,\,
\mathrm{HV}\!\left(\mathcal{D}_{\mathrm{out}}^\tau\right)
-\mathrm{HV}\!\left(\mathcal{D}_{\mathrm{out}}^0\right)
\right).
$$

四轮搜索分别在累计预算 $b_\tau\in\{0,4,8,12,16\}$ 处形成五个检查点。采用梯形法计算增量超体积曲线下面积：

$$
\begin{aligned}
\Delta\mathrm{HV}\text{-}\mathrm{AUC}
&=\sum_{\tau=1}^{4}
\frac{b_\tau-b_{\tau-1}}{2}
\left(\Delta\mathrm{HV}_{\tau-1}
+\Delta\mathrm{HV}_{\tau}\right)\\
&=2\sum_{\tau=1}^{4}
\left(\Delta\mathrm{HV}_{\tau-1}
+\Delta\mathrm{HV}_{\tau}\right).
\end{aligned}
$$

$\Delta\mathrm{HV}$ 衡量某一预算节点相对于冷启动前沿的扩张量，而 $\Delta\mathrm{HV}$-AUC 进一步奖励更早出现的前沿改进。与正文一致，HV 在归一化的 $(\mathrm{latency},\mathrm{energy},-\mathrm{AP}_{70})$ 目标空间内、相对于固定的任务相关参考点计算。绝对 HV 和 $\Delta\mathrm{HV}$-AUC 只在采用相同归一化、冷启动集合和参考点的同一 model--hardware--backend 任务内比较。

> **核实说明：** 正文只明确了“归一化三目标空间”和“固定任务相关参考点”，未给出 $\mathcal{N}_{\mathrm{task}}$ 的具体边界；当前归档实现仍可见原始目标空间路径。为避免附录与正文再次冲突，本稿不再写入未经最终结果 manifest 证明的 min--max 边界或 5% 参考点构造式。若后续需要将 E.3 提升为完整可复现定义，必须从生成正文 Table 2 的最终结果记录补入冻结的归一化参数和参考点。

### E.4 已核验计算环境

实验时的软件版本来自归档 runtime records，CPU、内存和操作系统等稳定主机属性由只读设备复核补齐。为与正文 Table 1 完全一致，本节沿用 `Default/H800`、`TensorRT/H800` 和 `TensorRT/Jetson AGX Orin` 三个 profile 名称。每个 profile 内的所有比较方法均使用相同的硬件、执行后端、软件环境和测量边界。

**Default/H800。** Table 1(a) 的实验运行于配备 $8\times$ NVIDIA H800 80 GB GPU、$2\times$ Intel Xeon Platinum 8480+ CPU 和 2 TB RAM 的服务器。主机使用 Ubuntu 22.04.2、Linux 5.15.0-1029-nvidia 和 NVIDIA driver 535.54.03，GPU power limit 设为 700 W。且该执行路径不经由 cuDNN dispatch。这里的 `Default` 是正文采用的 profile 名称：其未调优参照采用后端默认执行配置；对于明确包含 backend tuning 的比较方法，调优仍在同一 环境和相同测量合同下完成。

**TensorRT/H800。** Table 1(b) 使用与 Default/H800 相同的 H800 服务器、操作系统、驱动和 700 W power limit，但在独立的软件环境中构建和测量 TensorRT engine。该环境使用 Python 3.9.25、PyTorch 2.0.1+cu118、CUDA 11.8、cuDNN 8.7.0 和 TensorRT <PRIVATE_HOST>。所有 TensorRT engine 均在该环境内生成并执行。

尽管上述两个 H800 profile 共享同一物理服务器，其 Python、CUDA 和 backend runtime 均相互隔离。采用独立环境是必要的，因为不同后端调度具有不同的环境和运行时依赖以及 CUDA 和 cuDNN 兼容关系。强行使用完全相同的软件栈可能导致编译或运行不兼容，也可能迫使其中一个后端偏离已经核验的稳定依赖组合，从而将环境适配问题引入性能结果。为此，我们为两个后端分别冻结经过核验的软件环境；一次实验只激活其中一个 backend-specific environment，模型构建、后端调优、warm-up、正式计时和能耗采样均在该环境内闭合完成，任何候选的时延或能耗结果都不会从另一个 H800 profile 复用。该设置不改变 profile 内部比较的公平性，因为同一 profile 下的所有方法仍共享完全相同的硬件、软件栈和测量合同。

**TensorRT/Jetson AGX Orin。** Table 1(c) 运行于配备 12-core Arm Cortex-A78AE CPU 和 64 GB unified LPDDR5 memory 的 Jetson AGX Orin。设备固定为 MODE_30W 并启用 `jetson_clocks`，系统使用 Linux 5.10.216-tegra、L4T 35.6.1、Python 3.8.10、PyTorch 1.12.0a0+2c916ef.nv22.3、CUDA 11.4、cuDNN <PRIVATE_HOST> 和 TensorRT <PRIVATE_HOST>。Orin engine 在该设备的软件环境中独立构建和测量，不复用 H800 上生成的 TensorRT engine 或性能记录。

H800 latency 排除数据读取和 Host-to-Device transfer，energy 使用 NVML board power；Orin latency 仅覆盖 TensorRT engine execution，energy 使用 active-window `VIN_SYS_5V0` rail。由于两种功率源具有不同的电气边界，energy ratio 只在相同 profile 和相同测量协议内报告，不跨 H800 与 Orin 汇总。
