# 联合/协同搜索方法研究笔记 v1 — ALT / CHaNAS / AutoTVM

> 目的: 为本项目 route2(把**编译调度** tiling/loop-order/layout/tactic/fusion 变成**可搜索的硬件轴**, 与软件旋钮 prune×quant 联合搜索)做相关工作定位、方法借鉴、差异化锐化。
> 标注约定: **[文献声称]** = 论文/二手源原文主张; **[启发,推断]** = 我对本项目的推断, 非原文。
> 下载/阅读状态见文末附录。本笔记基于 PDF 全文精读(ALT/AutoTVM)+ NACOS 综述(arXiv:2408.04116, 含 CHaNAS 专段)+ ACM 摘要 + web 检索(CHaNAS 无 arXiv, PDF 未取到)。

---

## 一、ALT (EuroSys'23) — 打破 layout↔loop 的墙

- **arXiv**: 2210.12415(预印本题为 *"Boosting Deep Learning Performance by Breaking the Wall between Graph and Operator Level Optimizations"*; EuroSys'23 正式版改题为 *"…between Data Layout and Loop Optimizations…"*, 同一篇)
- **DOI**: 10.1145/3552326.3587440
- 作者: Zhiying Xu 等(南京大学 + 华为)

### (a) 一句话方法
[文献声称] ALT 是一个深度编译器, 把**图级 data layout 优化**与**算子级 loop 优化**放进**同一个 auto-tuning 框架**联合搜索, 打破了前人"先定 layout, 再 tune loop"的单向、一次性流水, 平均比 Ansor 单算子快 1.5×、端到端快 1.4×。

### (b) 搜索空间 + 联合的是哪两侧
- **两侧 = ① graph-level data layout(张量存储格式, 如 C2D 的 NOHW / NHWO / HWON, 含多级 tiling layout)+ ② operator-level loop(loop tiling / reorder / vectorize / unroll / fuse)**。
- [文献声称] 关键 motivation(Obs.1): 最优 layout 对 loop 性能影响巨大(选对 layout 让 loop 优化在 Intel CPU 提升 55.9%、GPU 87.2%、ARM 48.8%), 但无 loop 反馈时无法判断哪个 layout 好 → 必须联合。
- [文献声称] 联合空间巨大: 单 C2D 算子 layout 变换约 O(10^19)、loop 约 O(10^7), 组合后爆炸。
- **降空间手段**: ① 只为"复杂算子"(conv / GEMM, layout-sensitive)建 layout 变换空间, 其它张量用 **layout propagation** 传播已搜 layout, 不再搜; ② 用基于"layout 如何影响数据复用/cache"的分析裁出**有前途子空间的 tuning template**(C2D 剪到 O(10^6) 6 个可调参数: h_t/w_t/o_t/i_t/i'_t/o'_t)。

### (c) 搜索算法 + 代价模型
- **核心创新 = "cross-exploration architecture"(交叉探索架构)**: 把联合 tuning 拆成两阶段——**joint stage**(同时搜 layout+loop)与 **loop-only stage**(layout 固定只搜 loop)。在 joint stage 用 **PPO 强化学习**[文献声称]:
  - **Layout Agent**(actor): 对每个可调 split 维输出动作 a_s∈(0,1), 映射成 split factor F=R(D·a_s), 产出一串 layout 原语序列(split/reorder/unfold)。
  - **Loop Agents**: 对每个 layout 候选, 重建 loop 空间后做多轮 random-walk loop 探索(类 FlexTensor), 取最佳 loop 性能反馈给 Layout Agent 作 reward。
  - 用一个**全局共享 critic 网络**建模众多 actor/primitive 间的相互影响。
  - reward 统一为 r = U − l(U 常数, l 为该 layout 多轮 loop 后的最佳延迟)。
- **关键设计点(对我们最有用)**: layout 一变, loop 空间就要重建, 上一轮搜过的点在新空间里失效 → 直接迭代探索低效。ALT 的对策正是 "joint stage 用 cross-exploration(每选一个 layout → 重建 loop 空间 → 多轮 loop 优化 → 把最优 loop 性能回灌作 reward)" 实现 layout↔loop 的**双向、统一**优化流。
- **代价模型**: XGBoost 树集成(同 Ansor), 输入是 program 的 loop 结构 + 访问表达式特征, 预测 throughput; 探索时只在真硬件上测每个 batch 的 top-k(或一条 RL 轨迹), 这些实测又用于在线训练 cost model。
- **transformation 基础设施**: 把 layout 操作抽象成可复用原语(split/reorder/fuse + unfold/pad/store_at 及其逆), 在 TE→TVMIR 的 pass 里改写所有张量访问索引——**关键: 改 layout 不需要重写算子实现**(这是前人做不到 joint 的根因: 改 layout 要手动重实现算子, 成本太高)。

### (d) 联合 vs 分离/串行的证据
- [文献声称] **消融 ALT-OL**(loop-only, 砍掉 joint stage, layout 固定 NHWO/NDHWO): ALT 相对 ALT-OL 端到端约 **1.3× 提升**——证明 layout 联合 tuning 带来真实独立收益, 不是 loop tuning 的副产物。
- [文献声称] **消融 ALT-WP**(只消除相邻算子间的 conversion 算子, 不做 joint layout tuning): ALT 相对 ALT-WP **平均 1.3×**, 个别无提升——证明把 layout 也纳入 tuning 比"只省转换"强。
- [文献声称] **micro-benchmark ALT-FP / ALT-BP**(两个串联 C2D, 一个先 tune 把 layout 单向传给另一个): ALT(分别 tune 两算子并在中间插 conversion)>ALT-FP/ALT-BP——证明**第一个算子的最优输出 layout 对第二个算子是次优的**, 即"串行单向传 layout"会陷次优, 联合/带反馈才好。
- [文献声称] case study(Table 3): 不同 layout 下同一 loop 优化后的指令数/L1-miss 差异显著, 最优 layout(N H/h_t W/w_t O/o_t h_t w_t o_t)只 2% cache miss, 优于 NHWO。

### (e) ★对本项目的启发
- [启发,推断] **ALT 的 layout↔loop 墙 ≈ 我们的 quant-alignment↔schedule 墙**: 我们的"÷32 通道对齐 kernel-cliff"本质就是一个 **layout/数据格式约束**(INT8 tensor-core 要求通道对齐), 而它能否被 schedule(tiling/tactic)消化, 取决于调度——这正是 ALT 说的"layout 与 loop 耦合, 不能分开定"。⇒ 我们可以把"对齐耦合"重新表述为 **"prune 决定的通道宽度(=layout)× quant 精度(=数据格式)× schedule(tiling/tactic)三者联合, 单独定任一个都次优"**。
- [启发,推断] **cross-exploration = 我们 searcher 的可借鉴骨架**: 软件旋钮(prune rate, 决定通道宽度)一变, schedule 空间(合法 tiling/tactic)就重建——这与"layout 一变 loop 空间重建"完全同构。⇒ Gap1 的搜索器可学 ALT 分两阶段: **joint stage**(prune×quant 选定 → 重建 schedule 搜索空间 → 用 TVM MetaSchedule 搜 schedule → 把最优 latency 回灌作 reward)+ **schedule-only stage**(软件旋钮固定再精调 schedule)。
- [启发,推断] **"改 layout 不需重写算子" → 我们要的正是 TVM 这层抽象**: TRT-auto 是黑盒, 改不了 layout/tactic 的可见枚举; 迁到 TVM(MetaSchedule schedule_rule + 自定义 layout transform pass)后, 通道宽度变化触发的 tiling 重建可以自动化, 不必手写 kernel。
- [启发,推断] **Gap1 消融直接照搬 ALT-OL/ALT-WP 范式**: 设"schedule-only(软件旋钮固定 = 串行)" vs "joint(prune×quant×schedule 联合)"两条 Pareto, 期望像 ALT-OL 一样看到 joint 前沿支配 schedule-only。注意 ALT 的 1.3× 是 latency 单目标; 我们是 (AP, latency, energy) 多目标, 证据形式应是 **Pareto 前沿支配/hypervolume**, 不是单点倍率。

---

## 二、CHaNAS (LCTES'21) — 架构 × 编译调度协同搜索 ★已获原文 PDF 全文精读

- **正式出处**: LCTES'21(ACM SIGPLAN/SIGBED, 2021-06-22, Virtual Canada), 题 *"CHaNAS: Coordinated Search for Network Architecture and Scheduling Policy"*, **DOI 10.1145/3461648.3463846**。
  - ⚠️ **勘误**: 原笔记据 NACOS 二手源写的 "ACM TECS'22 / DOI 10.1145/3533251" 与用户提供的 PDF 不符 —— PDF 是 **LCTES'21** 会议版。本节据该 PDF 全文 12 页精读重写, 旧二手源 caveat 已全部撤除。
- 作者: Weiwei Chen, Ying Wang, Gangliang Lin, Chengsi Gao, Cheng Liu, Lei Zhang(中科院计算所)。用 **TVM 0.7.dev** 做代码生成。

### (a) 一句话方法
[文献声称] **首个**把 NN 架构 + "把模型映射到目标硬件的编译调度策略"放进**同一联合搜索空间**搜索的框架; 用 block-based 分层超网 + 块级预调度 + 进化搜索。在 ImageNet 上, **iso-accuracy** 下相对同精度 SOTA HW-NAS(MobileNet-v3)在 P100/Xeon/Note10 各快 **1.6× / 1.9× / 1.7×**。

### (b) 搜索空间 + 联合的是哪两侧
- **两侧 = ① NN 架构 {arch}**(elastic MBConv 超网: depth D / channel expand ratio E / kernel size k / input resolution; 基本单元 = block_i)**× ② 每块编译调度 C={c1..cm}**(graph 级: layer fusion + data layout transform; operator 级: Table 1 十种原语 split/reorder/fuse/unroll/vectorize/parallel/cache/bind/compute_at/inline 及参数)。
- **目标(Eq 1)**: (arch*, c\*_arch\*) ∈ argmax_{arch∈{arch}, c∈C} A(arch,c)  **s.t.** P(arch\*,c\*) < P_thres。⇒ **精度 A 是目标, 延迟 P 是约束** —— 搜索本身是 constrained-single-objective(不是多目标 Pareto; Pareto 只在 eval 阶段展示, Fig 7)。
- ★**关键降空间(分解)**: 原单级联合空间 = **R^B · S^B**(R 架构/块, S 调度/块, B 块)→ 分层分解为 **R·B·S + R^B**(块级独立预调度 R·B·S 个 + 架构组合 R^B), 指数级 → 可控。**这是它能把"架构×调度"联合搜索做到可行的核心 trick。**

### (c) 搜索算法 + 代价模型(三组件, Fig 3)
1. **Elastic super-net**: 权重共享 one-shot, 子网直接继承超网权重 → **免逐候选训练**, 快速评精度。超网训一次(16×P100, 4.5 天, 知识蒸馏渐进 finetune); **块优化与超网训练并行**(块优化独立于训练)。
2. **块级预调度(Step 2, Fig 5)**: 调度优化器 F(A1..Ab, D_hard) → (c\*_block, P_c\*_block)。每块抽计算子图 → graph 优化(fusion + layout transform)+ operator 调度搜(**高斯过程 GP 代价模型** + 启发式 mutation, 借块结构相似性跨块复用 GP)→ 把每块最优调度 + 延迟存进 **block 性能 LUT**; 全网延迟 **Lat_net = Σ_i Lat_block_i**(Eq 3, 加性可组合)。
   - ★**mutation 规则**(对我们最关键的一句): **split factor 限制为"可整除(divisible)的切分选择"** —— 原文 "we find that a split factor is efficient in most cases, while other split choices have inferior results. Hence, we limit the split factors to divisible split choices."
3. **协同探索(Step 3)**: 先按部署约束(网络推理延迟 CDF)**自动划子空间** —— 发现 **input resolution(In_size∈{160,176,192,208,224})+ model expansion ratio(Md_E∈{1.0,1.1,1.2,1.3})** 是两个最重要因子 → 分成 5×4=20 个子空间, 随机采样 λ 网络比较 latency CDF, 选平均延迟最低的子空间 → 再在该子空间用 **进化搜索**(arch 编码成 genome 向量, 随机取 15K 子网测 10K val 精度训 **MLP 精度预测器**, 配 block-LUT latency 预测器算 fitness, mutation + crossover 迭代)。

### (d) 联合 vs 分离/串行的证据 ★原文有干净消融(此前二手源说"未见"是错的, 现据 PDF 更正)
- **CHaNAS-W vs CHaNAS-W/O(Table 2, Note10)**: 同一超网抽出的解, **W/O = 不做调度级再优化**(= NAS only, 固定调度), **W = 加块级调度协同优化**。结果: W/O **27.5ms** vs W **16.4ms**, 且 **iso-accuracy**(ImageNet top1 76.0 vs 76.2)、W 的 **MACs 反而更多(240M vs 224M)却更快** ⇒ **纯调度协同在等精度下省 1.68× 延迟**。这是"联合 > 串行"**最干净的同超网消融**(变量只有"是否做调度协同")。
- **Fig 7(三平台 Pareto)**: CHaNAS-W 支配 CHaNAS-W/O 支配 MobileNetV3/Fbnet/Mnasnet(P100/Xeon/Note10)。
- ★**Fig 2(motivation = 我们的 argmin 漂移测试同构!)**: 200 个 OFA 随机模型在 Note10 上跑 3 种调度(A=TF_Lite 默认, B/C 改 loop split 旋钮), **三条 Pareto 前沿彼此相交** —— 没有任一调度的 NAS 基线恒优; **最优架构随延迟约束切换**(>35ms 时 Schedule-B 优, 20ms 时 Schedule-A 优)。即"最优架构依赖调度、反之亦然" = 架构×调度耦合 = **我们 E-couple "argmin 随软件配置漂移"的同一逻辑**(只是它换成 arch×schedule, 我们是 prune-width×tile)。
- **协同成本**: 超网训一次 + 块优化与训练并行; 新场景重部署仅 **+50 GPU-hours($150)** vs Mnasnet 每场景 4000 GPU-hours($12250)。

### (e) ★对本项目的启发
- [启发,推断] **CHaNAS 联合 arch×schedule; 我们联合 prune×quant×schedule** —— 我们**不动架构拓扑**(固定 PyramidFusion/CoDriving), 只动**压缩旋钮**, 比 NAS 便宜得多(无需重训超网)且贴合"已部署模型协同加速"。**定位句**: "NACOS/CHaNAS 改架构(贵、需重训超网), 我们改压缩配置(prune×quant, 便宜、支持已训练模型), 同样联合编译调度, 但搜索成本低一个量级。"
- [启发,推断] **CHaNAS-W/W/O 消融 = 我们 Gap1 的直接模板**: 同一基底, "有/无调度协同"两条线, 用 **iso-accuracy latency**(Table 2 那样)作最legible headline。我们对应: 同一 prune×quant 配置, "schedule 固定(serial/现状)vs schedule 协同(joint)", 报 **iso-AP latency**。
- [启发,推断] **CHaNAS Fig 2 = 我们 E-couple 的 published 先例**: "最优架构随调度/约束切换"恰是我们想 claim 的 model-dependent 耦合的同构证据 —— related-work 里可直接引为"耦合驱动联合搜索"的范式来源。
- [启发,推断] **分解 R·B·S + R^B + block-LUT + Lat_net=ΣLat_block = 让我们联合搜可行的关键工程**: 别把 prune×quant×schedule 当全交叉积枚举(指数爆炸)。按"块"(= Pyramid backbone stage / 双 agent RSU·车端段)**预调度每个 (宽度,精度) 组合 → 存 schedule-LUT → 外层 NSGA-II 加性组合 Lat_net**。这直接抄 CHaNAS 的分层。
- [启发,推断] **divisible-split 规则 = 我们 ÷32 对齐纪律的 primary-source 锚**: CHaNAS 自己就把 split factor 限制为可整除, 因为不可整除"结果更差" —— 我们的 trap25(非÷32 → tensor-core 失配 → 慢)是同一现象在 INT8 边界的放大版。论文里"对齐是跨轴硬约束"可引 CHaNAS 的 divisible-split 作他证。
- [启发,推断] **CHaNAS 是 constrained-single-objective(精度目标+延迟约束); 我们是多目标 5 指标** —— 可借它的"约束化单目标"做干净的 iso-AP latency 切片, 同时另报多目标 hypervolume。它只用 latency-LUT 不联合 AP, 我们 AP 必 finetune 真测, 是相对它的强化。

---

## 三、AutoTVM (NeurIPS'18) — 学习式张量程序优化(Ansor 前身)

- **arXiv**: 1805.08166(*"Learning to Optimize Tensor Programs"*)
- 作者: Tianqi Chen, Lianmin Zheng, Eddie Yan 等(UW + SJTU)
- 开源: apache/tvm 的 `python/tvm/autotvm` 模块(路径见下)

### (a) 一句话方法
[文献声称] AutoTVM 把"为给定硬件选最优张量程序实现"形式化为 **arg min_{s∈S_e} f(g(e,s))**(e=compute 表达式, s=schedule, g=代码生成器, f=真硬件运行时延), 用**领域特定统计代价模型**引导在 O(10^9) 程序变体里搜索, 用**迁移学习**跨 workload 加速 2–10×, 性能可与手工库(cuDNN 等)竞争。

### (b) 搜索空间(注意: AutoTVM 不联合架构/layout, 是纯 schedule 搜索)
- **搜索空间 S_e = schedule 模板(template)**: 由 Halide/TVM 原语张成——multi-level loop tiling(每个 loop 轴的 tile 因子)、loop ordering、shared-memory caching、unrolling、vectorization 标注。单 GPU 算子 O(10^9) 个实现。
- ⚠️ **关键定位**: AutoTVM 本身**不是联合搜索**(不搜架构、不搜 layout), 它是 ALT/CHaNAS 所用的"算子级 loop tuning"那一侧的**基础引擎**。我们研究它是为了**借它的搜索算法 + 代价模型 + 迁移学习**, 这三块是任何 schedule 搜索器的核心。

### (c) 搜索算法 + 代价模型(本篇精华)
- **代价模型 f̂(x)**(两选一):
  - **GBT(XGBoost)**: 从 low-level AST 抽领域特征(内存访问次数、数据复用率、loop 结构 + unroll/vectorize/thread-binding 标注)。基于特征、CPU 快、预测快。
  - **TreeGRU**: 递归把 AST 编码成 embedding 向量再线性映射到 cost。无需手工特征、可扩展, 但训练/预测慢, 需 GPU 批处理。
- **训练目标函数**: 不用回归 loss(∑(f̂−c)²), 而用 **rank loss**(∑log(1+e^{−sign(c_i−c_j)(f̂_i−f̂_j)}))——因为选择阶段只关心相对排序, 不关心绝对值。[文献声称] 实验(Fig.5)rank ≥ regression。
- **探索算法(Algorithm 1)**: 用 **parallel 模拟退火(simulated annealing)**, 以 f̂(x) 为能量函数走 Markov 链; 每轮取预测 top-k batch 到真硬件实测 → 更新 f̂; Markov 链状态跨 f̂ 更新持久化; 加 **ε-greedy**(εb 随机点)保探索。
- **diversity-aware 探索**: 把 schedule s 分解成 m 个分量, 用子模函数 L(S)=−∑f̂ + α∑|∪{s_j}| 选既低 cost 又多样的 batch(贪心近似)。[文献声称] 多数 workload 无显著影响, 个别有益, 故保留。
- [文献声称] 统计代价模型显著优于黑盒(GA/random): GBT/TreeGRU 找到的算子比 random 快 2×、收敛更快(Fig.4)。

### (d) "学习式 + 迁移" vs 朴素搜索的证据
- [文献声称] **代价模型 vs 黑盒(Fig.4)**: GBT/TreeGRU 比 GA、random 收敛快且终值好——证明领域特定建模有效(这点与传统超参调优"模型法≈random"的结论相反, 因为这里有领域结构可利用)。
- [文献声称] **迁移学习(Fig.8)**: 用 invariant 表示(low-level AST 的 context relation 特征, 对搜索空间不变)共享 cost model, f̂ = f̂^global + f̂^local; 跨 workload 迁移省 **2–10×** trials。Fig.9 显示 context-relation 特征跨算子类型也能迁移, flatten-AST 只能同算子类型内迁。
- [文献声称] **端到端(Fig.10)**: 无外部算子库下, AutoTVM 生成的程序在 server GPU / 嵌入式 CPU / mobile GPU 上与 cuDNN/TFLite/ARM ComputeLib 竞争甚至更优; 整体端到端 1.2–3.8×。

### (e) ★对本项目的启发
- [启发,推断] **AutoTVM 是我们 route2 schedule 搜索的"标准件"**: 我们要把 schedule 变可搜轴, 最现实的落地就是 TVM 的 AutoTVM(模板)或 MetaSchedule(无模板, 进化搜索, AutoTVM 的后继)。MetaSchedule 用进化搜索 + XGBoost cost model, 与 AutoTVM 同源——我们的 schedule 轴可直接复用其 `search_strategy/evolutionary_search` + `cost_model/xgb_model`。
- [启发,推断] **rank loss > regression loss 的教训直接适用**: 我们的 LGB latency 预测器(已知"跨数量级回归崩, 给负值", 见 CLAUDE.md §四)就是回归 loss 受害者; **改用 rank/pairwise loss(只学相对快慢)很可能修好 latency 预测器**——这是 AutoTVM 给我们最直接可落地的一条。
- [启发,推断] **迁移学习的 invariant 表示 → 我们跨硬件(4090↔Orin)迁移的方法论**: 我们已有 "M2 f 函数跨平台拟合 R²>0.995", 但若要把 schedule cost model 从 4090 迁到 Orin, 应学 AutoTVM 用对硬件不变的 AST/特征表示 + global+local 分解。
- [启发,推断] **模拟退火 + ε-greedy + 在线更新 cost model 的探索循环**, 可作为我们 searcher_v0 升级到 Stage4(NSGA-II/BoTorch)时 schedule 子搜索的内循环模板。
- [启发,推断] **AutoTVM 单目标(latency); 我们多目标**——不能直接抄它的 arg min f, 要把它的 schedule 搜索嵌进我们的多目标外循环(prune×quant 为外, schedule 为内)。

---

## 四、跨方法对比表

| 维度 | ALT | CHaNAS | AutoTVM |
|---|---|---|---|
| 出处 | EuroSys'23 / arXiv 2210.12415 / DOI 10.1145/3552326.3587440 | **LCTES'21 / DOI 10.1145/3461648.3463846**(原文已精读) | NeurIPS'18 / arXiv 1805.08166 |
| 联合的两侧 | **graph layout × operator loop** | **NN 架构 × 编译调度(block 粒度)** | **不联合**(纯 schedule/loop tuning) |
| 是否动架构 | 否 | **是**(super-network NAS) | 否 |
| 是否动 layout | **是**(核心) | 是(block 内 layout transform) | 否(layout 固定) |
| 是否动 schedule/loop | 是 | 是 | **是**(唯一轴) |
| 搜索算法 | PPO 强化学习(Layout Agent + Loop Agents, cross-exploration 两阶段) | 进化算法(HW-NAS)+ 类进化启发式(ACO) | 并行模拟退火 + ε-greedy + diversity-aware |
| 代价模型 | XGBoost 树集成(在线训练) | GP(ACO)+ latency LUT & 精度预测器(NAS) | XGBoost / TreeGRU, **rank loss** |
| 联合>串行的证据 | ALT-OL/ALT-WP 消融(1.3×)+ ALT-FP/BP micro(单向传 layout 次优) | **CHaNAS-W vs W/O 1.68× iso-acc(Table2)+ Fig2 最优架构随调度漂移 + Fig7 三平台 Pareto 支配** | 代价模型 vs 黑盒(2×)+ 迁移(2–10×); **非 joint 证据** |
| 加速幅度 | 单算子 1.5× / e2e 1.4× vs Ansor | 1.6–1.9× vs 同精度 baseline | e2e 1.2–3.8× vs 手工库 |
| 目标数 | 单目标 latency | latency(+精度约束) | 单目标 latency |
| 开源 | 未找到公开 artifact 仓库 | 综述标 "Yes/TVM", 但未定位到公开仓库 | **apache/tvm `python/tvm/autotvm`** |
| 对我们的角色 | **方法骨架**(耦合两侧 + cross-exploration + 联合消融范式) | **定位对照**(NACOS 改架构, 我们改压缩) | **工具基础**(schedule 搜索引擎 + rank loss + 迁移) |

---

## 五、综合: 这三篇对本项目 Gap1(联合搜 vs 串行搜 Pareto 对比)实验设计的直接启发

> Gap1 = 证明"prune×quant×schedule **联合搜**"的 Pareto 前沿支配"**串行搜**(先定 prune×quant 再单独 tune schedule, 即当前 TRT-auto 黑盒做法)"。
> ★**完整可执行的 Gap1 实验设计(三臂消融 + 内外环 + 分解 + 度量 + 载体 + 决策规则 + 落地步骤)已落盘**: [`../methods/design/gap1_joint_vs_serial_design_v1.md`](../methods/design/gap1_joint_vs_serial_design_v1.md)。本节是其文献依据摘要。

1. **[启发,推断] 消融对照范式直接抄 ALT-OL/ALT-WP**。设三条线: ① **串行(serial)**= 软件旋钮固定后才 tune schedule(≈现状 TRT-auto, 我们的"schedule-only");② **联合(joint)**= prune×quant×schedule 同搜;③(可选)**无 schedule 轴**= schedule 固定默认(≈我们当前 searcher_v0, D 只路由不调度)。期望 joint 前沿在 (AP, latency, energy) 空间支配 serial——这正是 ALT 用 ALT-OL 证 layout 联合有独立收益的同构实验。
   - ⚠️ 度量改为**多目标 Pareto 支配 / hypervolume**, 不是单点倍率(ALT/CHaNAS/AutoTVM 都是单目标 latency, 我们必须升级)。

2. **[启发,推断] 耦合的"墙"要选对——我们的墙是"÷32 通道对齐 × INT8 数据格式 × tiling/tactic"**。ALT 教我们: 联合的价值只在**两侧强耦合**时显现(layout↔loop)。我们必须先用实验确认 prune(通道宽度)与 schedule(tiling)在 INT8 边界确实强耦合(对齐 cliff), 否则 joint 退化成 serial(回想 CLAUDE.md 已记录的教训: 剪枝率轴上 Pareto 退化"剪到底即最优", 因为耦合弱)。⇒ **Gap1 的 sweet spot 应放在 INT8 对齐边界附近**, 那里耦合最强、joint 最可能赢。

3. **[启发,推断] 内外循环结构 = ALT 的 cross-exploration + AutoTVM 的 schedule 内循环**。外循环(prune×quant, 多目标 NSGA-II/BoTorch)每选一个软件配置 → 通道宽度确定 → **重建 schedule 搜索空间** → 内循环用 TVM MetaSchedule(进化 + XGBoost, AutoTVM 后继)搜 schedule → 把最优 (latency,energy) 回灌作该配置的真实指标。这把"软件旋钮变→schedule 空间重建"的同构关系(对应 ALT "layout 变→loop 空间重建")落进我们的搜索器。

4. **[启发,推断] cost model 用 rank loss + 在线更新(AutoTVM)修我们的 latency 预测器**。当前 LGB latency "跨数量级回归崩"——改 pairwise/rank loss(只学相对快慢)是低成本高确定性的修法。schedule 内循环的 cost model 直接复用 TVM 的 `cost_model/xgb_model`(XGBoost + rank), 不重造轮子。

5. **[启发,推断] 工具栈决策: 从 TRT-auto(黑盒, 不可搜)迁到 TVM MetaSchedule(schedule 可见可搜)**。这是把 schedule 变"可搜索硬件轴"的唯一现实路径——TRT 逐层精度是延迟驱动的封闭 tuner(CLAUDE.md §0.3 已确认), 无法暴露 tiling/layout 枚举; TVM 的 `schedule_rule/multi_level_tiling` + `space_generator/post_order_apply` + `search_strategy/evolutionary_search` 正好把 ALT 那套"layout/loop 原语搜索"开放出来。**注意**: 我们 memory 已记录"CoDriving/Pyramid TVM 迁移"两条线的真实坑(grouped conv 8–10× vs 标准 2×, fusion neck 导入 blocked), Gap1 实验须建在这些已打通的 TVM 基线上。

6. **[启发,推断] CHaNAS-W/W/O 给 Gap1 一个比 ALT 更贴切的 headline 度量 = iso-AP latency**。CHaNAS 是 arch↔schedule(更接近我们 config↔schedule)。其 Table 2"同超网、有/无调度协同、等精度下省 1.68× 延迟"= 我们应做"同 prune×quant 配置、schedule 固定(serial)vs 协同(joint)、**iso-AP latency**"切片。**两个度量都报**: ① iso-AP latency 倍率(legible headline)② 多目标 hypervolume / Pareto 支配(严谨)。

7. **[启发,推断] CHaNAS 分解 R·B·S + R^B 解我们联合搜的"指数爆炸"**。prune×quant×schedule 全交叉积不可行; 抄 CHaNAS: 按"块"(Pyramid backbone stage / 双 agent RSU·车端段)**预调度每个 (宽度,精度) 组合 → 存 schedule-LUT → 外层加性组合 Lat_net=ΣLat_block**。内环 schedule 搜只跑有限块×档组合一次, 外环 NSGA-II 查表组合, 不必每候选重跑 MetaSchedule。

8. **[启发,推断] CHaNAS Fig 2 = E-couple 的 published 先例; divisible-split = ÷32 对齐他证**。Fig 2"最优架构随调度/约束切换"可在 related-work 引为耦合范式来源; CHaNAS 自限 split 为可整除("不可整除结果更差")是我们"对齐跨轴硬约束/trap25 失配"的 primary-source 他证。

---

## 附录: 下载/仓库/阅读状态

### 下载的 PDF(目录 `/home/jichengzhi/V2X/multi_agent/references/papers/`)
| 文件 | 大小 | 状态 |
|---|---|---|
| `ALT_2210.12415.pdf` | 1.76 MB | ✅ arXiv 全文, 已精读 1–12 页(全文) |
| `AutoTVM_1805.08166.pdf` | 4.36 MB | ✅ arXiv 全文, 已精读 1–8 页(正文+实验, 后续为补充材料) |
| `NAS_codeopt_survey_2408.04116.pdf` | 0.55 MB | ✅ NACOS 综述(背景二手源), 已精读全 13 页 |
| `CHaNAS.pdf` | 4.5 MB | ✅ **用户提供原文(LCTES'21)**, 已精读全 12 页; §二 已据此重写, 旧二手源 caveat 撤除 |

### 开源仓库
- **AutoTVM**: 属 `apache/tvm`, **未克隆整库**(过大), 关键模块路径(已在 v0.17.0 tag 核验):
  - `python/tvm/autotvm/tuner/xgboost_cost_model.py` — XGBoost 代价模型(本笔记 §三c)
  - `python/tvm/autotvm/tuner/xgboost_tuner.py`, `sa_model_optimizer.py` — 模拟退火探索
  - `python/tvm/autotvm/tuner/{ga_tuner,model_based_tuner,index_based_tuner}.py`
  - `python/tvm/autotvm/task/space.py` — schedule 模板空间定义; `task/dispatcher.py` + `tophub.py` — 迁移/历史库
  - ⚠️ **勘误**: apache/tvm **main 分支已删除 autotvm**, 重构为 `python/tvm/s_tir/meta_schedule/`(AutoTVM 的后继 MetaSchedule)。要用 AutoTVM 须 checkout ≤ v0.17.x 的 release tag。MetaSchedule 关键路径(route2 直接相关): `search_strategy/evolutionary_search.py`、`schedule_rule/multi_level_tiling.py`、`cost_model/xgb_model.py`、`space_generator/post_order_apply.py`、`tune.py`、`relax_integration.py`。
- **ALT**: GitHub / gh / MCP 搜索均**未找到官方公开 artifact 仓库**(EuroSys'23 论文未挂公开代码链接)。
- **CHaNAS**: **未找到公开仓库**(NACOS 综述 Table1 对另一篇 [36] 标 open-source, CHaNAS 行未明确; 我方搜索亦无果)。

### 引用
- ALT: Xu et al., EuroSys'23. arXiv:2210.12415. DOI 10.1145/3552326.3587440.
- CHaNAS: Chen et al., LCTES'21(ACM SIGPLAN/SIGBED), 2021. DOI 10.1145/3461648.3463846.(原文 PDF 已精读; 注: 另有 DOI 10.1145/3533251 疑为期刊扩展版, 未读)
- AutoTVM: Chen et al., NeurIPS'18. arXiv:1805.08166.
- 二手源: Bachiri et al., "Combining NAS and Automatic Code Optimization: A Survey", arXiv:2408.04116, 2024.
