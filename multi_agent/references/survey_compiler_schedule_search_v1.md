# 编译调度作为"可搜硬件轴"调研 v1
> Task: route2 / S2(TVM) 支撑文献 | 2026-06-17
> 主题: **把编译调度(tiling / loop-order / memory-layout / tactic / 算子融合)本身变成被搜索的硬件维度, 而不是甩给黑盒 auto-tuner(TRT trtexec)**
> 纪律: [文献声称](WebSearch/原文核) / [我方实证](本项目真测) / [启发](推断, 明确标注) 严格分层
> 关联: [survey_hwsw_codesign_v1.md](survey_hwsw_codesign_v1.md)(协同优化机理总纲) · memory `project-hwsw-codesign-route2.md`(S0b/S2.1 实测) · [HANDOFF_tvm_migration_route2_v1.md](../methods/progress/HANDOFF_tvm_migration_route2_v1.md)

---

## §0 为什么调研这条线 (我们的 gap)

[我方实证] S0/S0b 已证: 我们目前**把硬件调度全委托给 TRT-auto**(封闭 tuner, 只能在自带 tactic 里"选", 无 API 定义 tile/loop-order/layout/fusion)。结果 D 维几乎塌缩成常量, TRT 上唯一立得住的协同耦合只剩"通道对齐×INT8"一个(trap25 失配 48_96_192 → fp16 比 p50 慢 2.9× / INT8 仅 1.08×), 且对齐模型的 INT8 层数随 build 旋钮波动=auto 延迟选择而非干净轴 ⇒ **TRT 太薄**。

本批工作(Ansor / CHaNAS / FAST / TLP / Roller / Hidet / ALT)的**共同区别于我们**之处:
> **他们把 tiling / loop-order / memory-layout / tactic / 算子融合枚举成显式搜索点; 我们委托 TRT-auto。** 最高杠杆的修法 = 不调 trtexec auto, 改用 TVM/Ansor(或 Roller 构造式 / ALT 联合 layout+loop), 让 tile/layout/fusion 成为**可见、可与软件维(剪枝率×量化档)联合搜索**的硬件维。

这正是 route2 的技术内核: 调度可搜 → prune/quant × schedule 的耦合才能被发现/量化/缓解(List B B1–B7), 而不是被 auto 抹平。

---

## §1 七篇基本内容 + 启发 (按机制分组)

### 组 A — 搜索式 auto-scheduler (枚举调度 + 代价模型)

**A1. Ansor (OSDI'20)** — *Generating High-Performance Tensor Programs for DL*, Zheng et al. (UCB)
- [文献声称] 自动生成 tensor program: **分层搜索空间**(sketch 高层结构 + annotation 低层 tiling/loop 细节), **进化搜索 + 学习代价模型**, 外加 **task scheduler** 在子图间分配调优预算。无需 AutoTVM 那样的人写模板。速度 vs 模板搜索: **CPU 3.8× / ARM 2.6× / GPU 1.7×**。
- [启发] **这就是把 tiling/loop-order/compute-location 变成自动搜索点的奠基工作**, 也是我们 S2.2 的现成引擎(TVM 内置 MetaSchedule 是其后继)。但 Ansor **先定 layout 再调 loop**(单向), 是它的天花板(见 ALT 打破之)。对我们: Ansor 是把调度从"委托 auto"升级为"显式搜索"的最小可行路径。

**A2. TLP (ASPLOS'23)** — *A Deep Learning-based Cost Model for Tensor Program Tuning*, Zhai et al.
- [文献声称] 从**调度原语(schedule primitives)**直接抽特征(把调度当"张量语言", 延迟预测转成 NLP 回归任务), 不依赖硬件架构专家特征。搜索时间 **CPU 9.1× / GPU 3.0×** 提速 vs SOTA。配套 MTL-TLP 用多任务学习解"跨硬件代价模型不可迁移"问题。
- [启发] 与 [我方实证] f_lat 预测器上限 R²~0.73 ([[project_f_lat_ceiling]]) 呼应: **特征工程决定可学习性**。若 S2.2/S3 要建 schedule×prune/quant 的代价模型, TLP 的"调度原语即特征"思路比手工硬件特征更可迁移。次要但对预测器阶段有用。

### 组 B — 构造式 (不盲搜, 按硬件对齐"构造"内核) ★重点

**B1. Roller (OSDI'22)** ★ — *Fast and Efficient Tensor Compilation for DL*, Zhu et al. (MSR)
- [文献声称] 核心 = **rTile**: 一种"封装了**与底层加速器关键特征对齐**的 tensor 形状"的 tile 抽象; 从 tensor 表达式抽形状 + 用硬件规格构造 rTile(硬件对齐的building block)。**递归构造算法**生成 rProgram, 用微观性能模型免实测评估 → **秒级生成内核**(vs 数小时搜索), GPU 上与 SOTA 相当, 不成熟加速器(IPU)上更优。
- [启发 ★ 对我们最关键]: **Roller 的 rTile = 我们"对齐×INT8 陷阱"的正面机理化**。我们 S0b 发现的 trap25(48_96_192 非÷32 → tensor-core 不可用 → fallback 慢 2.9×) 本质就是 **rTile 无法与 MMA 形状对齐**的失配案例。Roller 把"tile 与硬件单元对齐"上升为**第一性设计原则**, 而非搜索副产物。⇒ 我们的 co-design 应把 **tile/layout 对齐**做成显式可搜/可构造的硬件轴; 对齐陷阱 = Roller 框架下的"未对齐 rTile"。**且 Roller 秒级构造 → S2.2 不必非走漫长 Ansor autotune, 可用构造式直接对比"对齐 vs 失配"rTile 的 tensor-core 占用/延迟, 更便宜、更确定地复现耦合**(见 §3)。

**B2. Hidet (ASPLOS'23)** — *Task-Mapping Programming Paradigm for DL Tensor Programs*, Ding et al.
- [文献声称] 把调度嵌入 tensor program, 用 **task mapping** 直接定义计算分配与顺序 → 支持**语句级**细粒度优化(loop-oriented 调度做不到)。post-scheduling fusion 自动融合; 硬件中心调度空间**与输入尺寸无关**, 大幅减调优时间。vs ONNX Runtime / AutoTVM / Ansor **最高 1.48× / 平均 1.22×**, 调优时间比 AutoTVM/Ansor **少 20× / 11×**。独立编译器(不依赖 TVM), 仅 GPU。
- [启发] 提供比 loop-schedule 更细的控制粒度(语句级), 对我们 List B B5(fusion×Q/DQ 放置)、B2(tensorize MMA 映射)是更强的表达。但 **Hidet 仅 GPU 无 DLA**, 与 [survey_hwsw_codesign_v1.md] 记录的 Hidet 局限一致 → 边缘异构(Orin GPU∥DLA)用不上, 定位为 GPU 侧机理工具/对照。

### 组 C — 打破层间墙 (联合 layout+loop / graph+operator) ★重点

**C1. ALT (EuroSys'23)** ★ — *Breaking the Wall between Data Layout and Loop Optimizations for DL Compilation*, Xu et al.
- [文献声称] 现有编译器**先定张量 layout 再调 loop**(单向、一次性, 把 graph 级与 operator 级优化强行分到不同系统层, 错失联合调优)。ALT 提供通用变换模块用易用原语**同时操纵 layout 和 loop**, 集成 auto-tuning **联合优化 graph 级 layout + operator 级 loop**。vs SOTA(含 Ansor): 单算子**平均 1.5× 加速** + 端到端更优。
- [启发 ★ 直接验证我们 List B B1]: 我们 List B B1 假设"layout co-search 能把对齐硬悬崖变可调旋钮"——**ALT 已用发表结果证明: 联合搜 layout+loop 严格优于 Ansor 那种'先定 layout 再调 loop'(=TRT 的分离做法)**。这给我们 B1 一个**published precedent + 量化锚(1.5×)**。⇒ 论文里"TRT 固定 layout = 看不见对齐耦合"的论点有了文献支撑; 且指明我们若只用 Ansor(固定 layout)仍不够, **layout 必须进搜索空间**(ALT 思想), 这是 B1 缓解对齐悬崖(pad/repack)的理论依据。

### 组 D — 协同搜索 (网络结构 × 编译调度 / 全栈加速器) ★重点

**D1. CHaNAS (TECS'22)** ★ — *A Framework for Neural Network Architecture and Compile Co-optimization*, dl.acm.org/10.1145/3533251
- [文献声称] **联合搜索神经网络架构 + 对应的编译器调度策略**(把模型映射到目标硬件): hardware-aware NAS 阶段用 lookup table, 自动代码优化阶段用**高斯过程代价模型**。co-design 解相对同精度基线: **NVIDIA P100 1.6× / Intel Xeon 8163 1.9× / Samsung Note10 1.7×** 性能提升。
- [启发 ★ 我们最贴近的 precedent/模板]: CHaNAS = **"软件配置(网络结构) × 编译调度 联合搜索"** 的范式样板, 与我们"prune/quant × schedule 联合搜索"同构。它证明了**网络×调度联合搜索可发表、且严格优于同精度单侧基线**(1.6–1.9×)。⇒ 直接背书我们 co-design 的方法论合法性。**差异化 = 我们的占位**: CHaNAS 在服务器/手机 GPU 做 NAS×schedule; 我们做 **剪枝×量化 × schedule 在边缘 V2X 协同感知**, 且以**对齐×量化耦合(可量化性悬崖)**为具体机理抓手——CHaNAS 没碰量化对齐这个轴。

**D2. FAST (ASPLOS'22)** — *A Full-Stack Search Technique for Domain Optimized DL Accelerators*, Zhang et al. (Google Brain)
- [文献声称] 在**软硬件栈**上定义宽优化环境: **硬件 datapath + 软件调度 + 编译器 pass(算子融合、tensor padding)** 一起搜。针对 EfficientNet/BERT 瓶颈设计加速器: 单负载 **Perf/TDP 3.7×** / 多负载 **2.4×** vs TPU-v3。
- [启发] FAST 是**可重构加速器**全栈搜索(与 [survey_hwsw_codesign_v1.md] 记录的"真协同搜索靠可重构 target"一致, =我们 route1 的硬件不可重构张力)。**但有一个对我们极有用的点: FAST 把 `tensor padding` 当成显式编译器搜索决策**——这正是缓解我们对齐陷阱的 co-design 旋钮: **把失配通道(48)pad 到 ÷32(64)以恢复 tensor-core 资格, 用 FLOPs 换对齐**。固定硅片上我们搜不了 datapath, 但 **padding/layout/tile 这些编译 pass 仍可搜**(FAST 把它们和 datapath 平级列为搜索维)。

---

## §2 重点三篇对我们工作的具体启发 (CHaNAS / Roller / ALT)

| 论文 | 给我们的核心启发 | 映射到我方 |
|---|---|---|
| **CHaNAS** | "软件配置×编译调度"联合搜索是**已发表、被验证严格优于单侧**的范式(1.6–1.9×) | = 我们 co-design 方法论的 precedent; 我们差异化 = 剪枝×量化×schedule + **量化对齐耦合**机理 + 边缘 V2X 场景(CHaNAS 未碰量化对齐) |
| **Roller** | tile **与硬件单元对齐**是第一性原则(rTile), 不是搜索副产物; 秒级**构造**而非盲搜 | 我们 S0b 的 trap25(非÷32→fallback) = rTile 失配的实例; ⇒ ① 把 tile/layout 对齐做成显式可搜/可构造硬件轴; ② **S2.2 可用构造式直接对比对齐 vs 失配的 tensor-core 占用/延迟, 免漫长 autotune** |
| **ALT** | layout 必须**与 loop 联合搜**(打破"先定 layout 再调 loop"); 比 Ansor 1.5× | 直接验证 List B B1; 指出**只用 Ansor(固定 layout)不够, layout 要进搜索空间**; 给"TRT 固定 layout=看不见对齐耦合"论点文献支撑 |

**三篇合起来给我们一条清晰的论文逻辑线**:
1. [CHaNAS] 软件×调度联合搜索是合法且增益严格的 co-design 范式 →
2. [我方 S0b] 在**固定硅片 + TRT-auto** 下, 调度被黑盒抹平, 只剩对齐×INT8 一个能观察不能调的耦合(=TRT 太薄) →
3. [Roller] 该耦合的机理 = tile/layout 与硬件 MMA 单元的**对齐**(rTile); [ALT] 解法 = 把 **layout 提进可搜空间**与 loop/precision 联合搜; [FAST] 具体旋钮含 **tensor padding**(pad 到 ÷32 换对齐) →
4. [route2/S2] ⇒ 我们改用 TVM(Ansor/MetaSchedule, 必要时 Roller 式构造 + ALT 式 layout 联合)把对齐/tile/layout/fusion 变成**可见、可与剪枝率×量化档联合搜索**的硬件轴, 从而**发现并缓解** TRT-auto 抹平的多耦合(List B B1–B5)。

---

## §3 对下一步工作计划的影响 (是否改 S2.2?)

**结论: 不推翻 S2.2, 但显著优化其做法, 并强化论文论点。**

1. **[Roller 启发 → 降本] S2.2 可加一条"构造式对照", 不必只靠漫长 Ansor autotune。**
   原 S2.2 计划 = Ansor/MetaSchedule 搜 layout+tile 复现 B1/B2 对齐机理。Roller 表明: **对齐 vs 失配的差异是可由 tile-硬件对齐关系确定性推出的**(秒级构造 + 微性能模型), 不必跑数小时搜索。⇒ S2.2 先做**便宜的构造式/手工 layout 对照实验**: 对 p50(32_64_128÷32, rTile 可对齐 MMA) vs trap25(48_96_192✗÷32, rTile 失配) 在 TVM 里**显式指定对齐 vs 不对齐的 layout/tile + INT8 tensorize**, 直接量出 tensor-core 占用/延迟差 → 比 autotune 更确定、更便宜地复现"对齐×可量化性"耦合。autotune 作为补充上界。
   > [我方资产已就绪] 三个 backbone 子图(base/p50/trap25)已在 H800 relax 数值对齐 PASS, 可直接做该对照。

2. **[ALT + FAST 启发 → 加 layout/padding 旋钮] 缓解 demo 的具体动作 = pad/repack。**
   List B B1/B2 的"把对齐硬悬崖变可调旋钮"现在有明确手段: **layout 联合搜(ALT)** + **tensor padding 48→64(FAST)**。S2.2 的"协同 vs 串行" demo 落地为: 串行(prune-then-quant, trap25 失配 48ch → INT8 撞 MMA 墙 → fallback) vs **协同(TVM co-search: pad/repack 48→对齐 + INT8 tensorize, 救回 tensor-core)**, 给出协同后延迟 vs TRT-auto fallback 的对比。这是"硬件维度被协同地配置"的真 demo(回应用户之前"S0b 只证耦合存在、未证硬件被协同配置"的关键质疑)。

3. **[CHaNAS 启发 → 论文定位] related-work 与差异化清晰了。**
   related-work 主线 = CHaNAS(NN×schedule co-search)/ALT(layout+loop)/Roller(rTile 对齐)/Ansor(autoschedule); 我们的**空白占位** = 剪枝×量化×schedule 联合 + **量化对齐耦合**机理 + **边缘 V2X 协同感知**多指标 Pareto(AP/lat/throughput/energy/size)。CHaNAS/FAST 不碰量化对齐; ALT/Roller 不碰量化/剪枝软件维; 我们把两侧接起来。

4. **[TLP/Hidet → 次要] 暂不影响 S2.2 主线。** TLP(代价模型)留给 S3 预测器阶段; Hidet(仅 GPU 无 DLA)作 GPU 侧机理对照, 不进边缘异构主线。

**不变的纪律**: 延迟必空闲 GPU 实测; TVM 绝对延迟常追不上 TRT, 价值=**发现/量化/缓解耦合**非刷 SOTA(Roller/ALT 也强调对不成熟硬件/被忽视维度的增益, 非绝对峰值); 区分真测/估算/仅声明; supervisor 跨配置验稳定。

---

## §4 引用索引
- Ansor: USENIX OSDI'20, arXiv:2006.06762 — https://www.usenix.org/conference/osdi20/presentation/zheng
- CHaNAS: ACM TECS'22, 10.1145/3533251 — https://dl.acm.org/doi/full/10.1145/3533251
- FAST: ASPLOS'22, 10.1145/3503222.3507767, arXiv:2105.12842 — https://dl.acm.org/doi/10.1145/3503222.3507767
- TLP: ASPLOS'23, 10.1145/3575693.3575737, arXiv:2211.03578 — https://dl.acm.org/doi/abs/10.1145/3575693.3575737
- Roller: USENIX OSDI'22, arXiv — https://www.usenix.org/conference/osdi22/presentation/zhu
- Hidet: ASPLOS'23, 10.1145/3575693.3575702, arXiv:2210.09603 — https://dl.acm.org/doi/10.1145/3575693.3575702
- ALT: EuroSys'23, 10.1145/3552326.3587440, arXiv:2210.12415 — https://dl.acm.org/doi/10.1145/3552326.3587440
