# 消融实验设计 — 证明"单侧/串行贪心搜索陷局部最优, 联合搜索达全局更优" (v1, 2026-06-19)

> ★**2026-06-19 两点更正(用户强调, 全文据此理解)**:
> 1. **W_g 定义更正**: W_g = **单维(贪心所搜那一维)最优、但多维(全局)非最优**的点 = 单侧贪心搜索落入并卡住的**局部最优**。[旧稿把 W_g 写成"单维被支配但联合解锁全局"——那描述的是**被错过的全局点**, 现记为 **P_g**(单维次优/多维最优); **W_g 改为贪心吸引子(单维最优/多维非最优)**。] 贪心选 W_g 因它在所搜单维上最优; 它非全局, 因到达 P_g 需接受贪心已丢弃的单维次优选择。
> 2. **平台: 已迁移 TVM, 不以 TRT/INT8 为重心**。W_g 的机制改在 **TVM prune × schedule** 轴上: W_g = 某宽度在 **default dlight schedule 下最优**(贪心据此选它), 但其 tuned 后非全局最优; P_g = 另一宽度 default 下次优、但 **MetaSchedule-tuned 后才是全局最优**(因各宽度 schedule 调优余量不均匀→tuned 重排序)。原稿"FP16 平庸但 INT8 解锁(需 TRT INT8)"的机制**降为 future/次要**, 不作主依赖。

> **定位**: 本文是论文**有效性核心命题**的实验设计。
> **核心命题 (claim)**: **单侧搜索(串行/贪心)会因贪心而陷入结构性局部最优, 达不到全局最优; 联合搜索能到达串行够不到的全局更优解。**
> **与 gap1 旧设计的关系**: [`../methods/design/gap1_joint_vs_serial_design_v1.md`](../methods/design/gap1_joint_vs_serial_design_v1.md) 提供了"三臂 S0/S1/S2 + 内外双环 + 块分解 LUT + iso-AP latency + 对齐边界 sweet spot"的工程骨架。本文**不重复**那套工程, 而是**纠正其论证方式**: 旧设计的 §10.5 首轮结果本质是"枚举几个对齐陷阱点(trap25)来说明陷阱存在"; 用户已明确指出**这不够** —— 真正要证的是"**搜索算法在不知道全局的情况下, 串行因贪心被困**", 这必须**真正消融掉一侧搜索空间、各自跑完整搜索过程**, 比它们**收敛到的解**, 而不是枚举已知点。本文专注补这个论证缺口。
> **文献依据**: [`../references/study_joint_search_methods_v1.md`](../references/study_joint_search_methods_v1.md)(ALT/CHaNAS/AutoTVM)。
> **标注**: [借鉴 ALT]/[借鉴 CHaNAS]/[我们的设计]/[待定]。

---

## §0 为什么"枚举点对比"不够 — 论证方式的根本纠正 [我们的设计]

旧 gap1 §10.5 的 trap25 demo 是这样的:**我们(实验者)已经知道 base/p50/trap25 三个点的 (AP, latency)**, 然后指着 trap25 说"看, 非÷32 宽度慢 7.18×, 串行如果剪到这里就踩坑"。这是**结果点对比**, 有两个致命弱点:

1. **它假设了全局已知**。枚举=我们已经把全局 Pareto 测出来了, 再事后指认哪个点差。但论文的命题是关于**搜索算法**的: 一个搜索器在**不知道全局**时, 串行会不会被贪心困住? 枚举回答不了这个 —— 一个聪明的串行搜索"明明可以不剪到 25%"。审稿人会反驳: "串行搜索又不傻, 它在 P 维搜的时候 trap25 的 latency 已经暴露了, 它根本不会选 trap25"。**枚举无法排除"串行只是这次运气差"**。

2. **它证的是"陷阱存在", 不是"贪心必然陷入"**。用户原话: "找耦合陷阱不是为了证明陷阱存在"。陷阱存在是**环境属性**(耦合的客观事实); 贪心陷入是**算法属性**(搜索过程的结构性失败)。论文要的是后者。

**正确的论证 = 让两个搜索器在同一个未知环境里真跑搜索过程, 比它们各自收敛到哪。** 关键是: **串行搜索的"贪心"不在于它选了坏点, 而在于它的搜索结构(先锁一侧, 再搜另一侧)使某些全局最优解从一开始就落在它的可达空间之外** —— 它再怎么努力搜后续维度也到不了。这正是:
- [借鉴 ALT] ALT-FP/BP 微基准(Fig 11)的逻辑: 串联两个 C2D, ALT-FP 先 tune 第一个 C2D 把其**最优输出 layout 单向传给**第二个 → 论文实测"**第一个算子的最优输出 layout 对第二个是次优的, 反之亦然**" → 串行单向传 layout 被联合(分别 tune + 中间插转换)支配。**这不是枚举, 是真跑了两条 tuning 流程比收敛解。** 串行的失败是**结构性的**: 一旦第一个算子的 layout 被贪心锁死, 第二个算子的可达 loop 空间就被限制, 怎么搜都到不了联合解。
- [借鉴 CHaNAS] CHaNAS Fig 2 motivation: 200 个随机模型 × 3 种调度的 Pareto **彼此相交**, "最优架构随调度切换" → 没有任何单侧固定能恒优 → 锁死一侧必然在某约束下次优。

**本文的全部设计都围绕"如何用搜索过程(而非枚举)暴露并量化这个结构性局部最优"展开。**

---

## §1 三个消融臂 — 消掉一侧搜索空间 [借鉴 CHaNAS-WO + ALT-OL]

我们的完整联合搜索空间 = **P(剪枝宽度) × S(schedule: tiling/loop-order/fusion/tensorize, 即 TVM MetaSchedule)**。**主轴 = prune × schedule(已迁 TVM)**; 量化轴 Q(TRT INT8)作 future/次要扩展, 默认不进主搜索空间(见标题更正②、§4.2、§8)。三个消融臂对应"消掉哪一侧 / 怎么串":

| 臂 | 名称 | 搜索空间 | 搜索过程 | 对应文献 |
|---|---|---|---|---|
| **A-joint** | 完整联合 | P × S 同搜 | 单一多目标搜索器在 prune×schedule 联合空间里搜 | = CHaNAS-W / ALT joint |
| **A-noS** | 消掉 schedule 轴 | P (S 固定 default dlight) | 只搜 P, schedule 永远用 dlight 默认 | = CHaNAS-W/O(无调度再优化) |
| **A-serial** | 串行贪心(消掉联合性) | P → 锁 → S | **分阶段贪心**: 先在 P 维搜到当下最优(default-S 下)并锁定, 最后才在锁定宽度上搜 S | = 经典"先 NAS/剪枝后编译"串行流水 |

> 注: 量化轴 Q 若启用(future), A-joint = P×Q×S 同搜 / A-serial = P→Q→S 逐一锁定; 但主线消融只跑 prune×schedule。

**三臂的变量隔离**(这是消融干净的关键):
- **A-joint vs A-noS**: 唯一变量 = "schedule 是否进搜索空间"。证 schedule 轴有独立收益(= CHaNAS-W vs W/O, ALT vs ALT-OL)。
- **A-joint vs A-serial**: 唯一变量 = "两轴同搜 vs 先后锁定串行"。**这一对是本文核心** —— 证"贪心串行的结构性局部最优"。两臂用**完全相同的搜索空间、cost model、explorer 内核、总评估预算**, 只差"是否分阶段锁定"。

**期望**: A-joint ⪰ A-serial ⪰ A-noS(Pareto 支配)。但**期望不是论点**; 论点是下面 §2 的"为什么 A-serial 必然到不了 A-joint 的解"。

> ⚠️ 与旧 gap1 S0/S1/S2 的映射: A-noS≈S0, A-serial≈S1, A-joint≈S2。**改名 + 重新定义重点**: 旧 S1 串行只是"先定软件再 tune schedule"(两阶段); 本文 A-serial 强调**逐轴贪心锁定**(prune→schedule, Q 启用时再插一段), 且重点从"S2 比 S1 快多少"转到"A-serial 的可达解集被 A-joint 的解结构性排除"。

---

## §2 ★如何证"贪心→结构性局部最优"而非"偶然差" [我们的设计 + 借鉴 ALT]

这是全文最关键的一节。要证的不是"A-serial 这次跑出来差", 而是"A-serial 的搜索**结构**使它**系统地、可重复地、跨起点地**到不了 A-joint 的解"。设计四层证据, 层层加码:

### 2.1 锁死机制的因果链(定性, 解释为什么)[借鉴 ALT-FP/BP]

串行 A-serial 的第一阶段在 **P 维**搜索时, 它看到的 latency 是"P×**default-schedule(dlight)**"下的 latency。它会**贪心选当下 (AP, latency) 最优的剪枝宽度 = W_g**。但各宽度在 default-schedule 下的延迟排序 **≠** 它们各自被 MetaSchedule tune 到底后的排序(ALT: "layout 一变 loop 空间重建"; 我们: "剪枝宽度一变, 合法 tiling / 可命中的快核 / schedule 调优余量都变")。

**关键机理**(★按更正①重写): 设贪心吸引子 = **W_g(在 P 单维 default-schedule 下最优, 故被贪心选中; 但 tuned 后多维非全局最优)**; 真正的全局点 = **P_g(在 P 单维 default-schedule 下次优、被 W_g 支配, 但只有它在 MetaSchedule-tuned 后才成为全局最优)**。两者排序翻转的物理来源 = **各宽度 schedule 调优余量不均匀**(实测 base 9.06× / p50 8.72× / trap25 1.96×, 见 `results/gap1_schedule_lut.json`): 调优余量大的宽度在 default 下不起眼, tune 后反超。串行一旦在第一阶段按 default-schedule 锁死 W_g, 它在后续 schedule 阶段**永远不会回头改 W**, 于是 P_g 整条分支从它的可达空间里被切掉 → 结构性到不了全局最优 P_g。这与 ALT-FP/BP 的 "第一个算子最优 layout 对第二个次优" 完全同构: **单维(default-schedule)最优的 W_g ≠ 联合最优, 锁死 W_g 即排除 P_g**。

我们已有的实证锚点(来自 gap1 §10.5, 但**重新解读为机理而非枚举**): 剪枝到 25% → 宽度 48 → grouped conv(g=32) in_per_g={3,6,12} 非 2 的幂 → kernel-cliff, MetaSchedule 也救不回(21615µs)。**这说明"宽度选择"对下游 schedule 可达性有硬约束** —— 正是锁死机制的物理基础。但注意: 这里**不是**说"串行会蠢到剪 25%"; 而是更微妙的 §2.2。

### 2.2 ★构造"default-schedule 排序≠tuned 排序"的 rank-flip 环境, 让贪心必踩 [我们的设计, 最关键]

§2.1 的 trap25 太粗(串行在 P 维就看到它慢, 会避开)。要暴露**结构性**局部最优, 必须构造一个环境, 其中 **default-schedule 下的宽度排序与 tuned 后的宽度排序发生翻转**:

> **存在两个宽度 W_g 与 P_g: 在 P 单维(default dlight schedule)下 W_g 最优、P_g 次优(被 W_g 支配, AP 略低或 latency 略高), 但因各宽度 schedule 调优余量不均匀, P_g 在 MetaSchedule-tuned 后反超成全局最优, 而 W_g tuned 后非最优。串行在 P 阶段按 default-schedule 贪心锁了 W_g, 永远走不到 P_g 这条分支 → 结构性到不了全局最优。**

物理可行性(为什么这种 rank-flip 存在, [借鉴 ALT/CHaNAS]):
- **prune × schedule 的调优余量耦合(主机制)**: schedule 调优余量按宽度差异巨大 —— 实测 base 9.06× / p50 8.72× / trap25 1.96×(`results/gap1_schedule_lut.json`)。某宽度在 default dlight 下平庸(余量大、未被利用), 但 MetaSchedule tune 到底后大幅提速反超; 串行在 P 阶段只看 default-schedule 延迟, **看不到这个宽度的 schedule 潜力**, 于是不选它(选了 default 下更快的 W_g)。→ 这就是"单维(default-schedule)次优、tuned 后全局最优"的 P_g 的真实来源。
- [借鉴 CHaNAS Fig 2] 最优架构随调度漂移: 同一宽度在不同 schedule 下排序翻转 → 用 default-schedule 排序选宽度 = 用错的排序, 必然漏掉只在 tuned schedule 下才显现的宽度 P_g。
- **(future/次要)对齐 × 量化耦合**: 若启用 Q 轴, 某些宽度在 FP16 下平庸但恰好对齐 INT8 tensor-core MMA(÷32 且 in_per_g=2^k), 在 (INT8, tensorize) 下大幅加速 —— 同类 rank-flip 的另一来源, 但需真 TRT INT8(见 §8), 不作主依赖。

**构造步骤**(主线 prune×schedule):
1. 在搜索空间里**确保包含**至少一组这样的 (W_g, P_g) rank-flip 宽度对(先用小规模探针验证其存在 —— 见 §2.5 / §8 开放问题, 探针计划见下方引用)。
2. 让 A-serial 的 P 阶段在 default-schedule 下评估 —— 它会按 default 排序选 W_g ≠ P_g。
3. A-joint 在 prune×schedule 联合空间能搜到 (P_g, tuned-S_g)。
4. 比两者收敛解 → A-joint 支配, **且差距来自 P_g 这条被 A-serial 结构性排除的分支**(可追溯: 报告 A-serial 收敛解的 W=W_g 与 A-joint 收敛解的 W=P_g 不同, 且 A-serial 从未评估过 (P_g, tuned-S) 高质量组合)。

> **W_g 存在性探针**(rank-flip 是否真存在的硬前提): 见 [`design_wg_probe_int8_plan_v1.md`](../../../paper/design_wg_probe_int8_plan_v1.md)(已更新为 TVM prune×schedule 版)。

### 2.3 跨起点 / 跨 seed / 跨贪心顺序的鲁棒性(统计, 证"必然"而非"偶然")[我们的设计]

单次跑无法区分"结构性"与"运气"。设计**重复实验矩阵**:

- **多 seed**: A-serial 与 A-joint 各跑 N≥10 个随机 seed(explorer 的随机初始化/采样种子)。报告每臂收敛解的 **hypervolume 分布**(箱线图), 而非单值。
- **多起点**: A-serial 的第一阶段从不同初始剪枝档起步(贪心起点扰动)。若**无论从哪起点, A-serial 都收敛到含 W_g 而不含 P_g 的解** → 局部最优是结构性的(吸引盆地是 W_g、不含全局最优 P_g), 不是起点不好。
- **多贪心顺序**: 串行的轴顺序换成 P→S / S→P 两种(Q 启用时再加 P→Q→S / Q→P→S 等)。若**所有顺序都到不了 A-joint 的解** → 不是"顺序选错", 是"先锁任意一侧"本身的缺陷。
- **统计显著性**: A-joint vs A-serial 的 hypervolume 差做配对检验(同 seed 配对, Wilcoxon signed-rank)。报告 p 值 + 效应量。**判读**: 若差异显著且 A-serial 的 hypervolume 分布**整体低于** A-joint(分布不重叠或重叠很小)→ 结构性局部最优成立; 若分布大幅重叠 → 耦合弱, 退故事 B(§5)。

### 2.4 收敛过程曲线(动态, 证"搜够了还是到不了")[借鉴 AutoTVM Fig 4]

只比终值会被质疑"A-serial 预算不够"。必须画**收敛曲线**: 横轴 = 评估次数(真实测/cost-model query 数), 纵轴 = 当前最优 hypervolume。

- 两臂**同总预算**。
- 若 A-serial 的曲线**早早 plateau 在低于 A-joint 终值的水平**, 且**再给它更多预算也不再上升**(因为它的可达空间封顶了)→ 这是"到不了", 不是"没搜够"。**这是排除'预算不够'反驳的关键证据。**
- [借鉴 AutoTVM] AutoTVM 用收敛曲线(Fig 4)证 cost model 比 random 收敛快且终值好 —— 我们同形式, 证 A-serial 收敛快但**封顶低**(贪心的典型病征: 快速收敛到局部最优)。

### 2.5 可达空间可视化(直观, 证"分支被排除")[我们的设计]

把 A-serial 实际**评估过的所有 (W, S) 配置**(Q 启用时为 (W,Q,S))与 A-joint 评估过的, 投影到 (AP70, latency) 平面叠画。预期看到: A-serial 的评估点云**整体缺失 P_g 所在区域**(因为第一阶段按 default-schedule 锁死 W_g 后再不访问 P_g 分支), 而 A-joint 的点云覆盖到该区域并在那里(tuned-S 下)找到支配点。**这张图直接可视化"贪心把全局最优分支 P_g 切掉了"** —— 比任何倍率数字都有说服力, 且明确区别于"枚举几个点"(这是搜索器**自己访问过的点**, 不是我们事后摆的)。

---

## §3 不靠枚举的设计纪律 [我们的设计]

明文写进论文方法节, 防止退回枚举:
1. **全局对搜索器隐藏**: 搜索器只能通过 cost model + 有限真测预算访问环境, **不喂给它全局 Pareto**。我们(实验者)可以离线另测一个 reference Pareto 用于**评估**(算 hypervolume 的参考前沿), 但**这个 reference 绝不进入任何一臂的搜索过程**。
2. **同内核同预算**: 三臂共用同一 explorer(NSGA-II / 约束单目标内核)、同一 cost model、同一总评估预算。差异只在搜索空间结构(消哪轴 / 是否串行锁定)。
3. **报搜索过程产物**: 收敛曲线(§2.4)、评估点云(§2.5)、跨 seed 分布(§2.3) —— 全是"搜索过程"的产物, 不是"结果点"。
4. **A-serial 必须是合理的强串行 baseline**: 它在每一阶段都用当下能拿到的最优(不是故意选差点), 这样它的失败才归因于"串行结构"而非"baseline 太弱"。

---

## §4 载体、平台、指标 [借鉴 CHaNAS + 我们已确立素材]

### 4.1 载体模型(顺便证 model-dependent 判据)
两个模型对比, 本身就是"协同价值条件化"的数据点:
- **Pyramid (HEAL, grouped conv g=32)** = **预期显耦合**。grouped conv 的 in_per_g=2^k 约束 + INT8 对齐使"宽度↔schedule↔量化"强耦合 → A-serial 锁死宽度后下游受限明显 → **预期 A-joint 显著支配 A-serial**。已有公平 AP70 曲线: 0.631→0.590→0.564→0.530(stage_a, finetuned, DAIR val 1789), 配 TRT fp16 延迟。
- **CoDriving (标准 conv)** = **预期可分离**。标准 conv 对宽度不挑, schedule 旋钮价值低(TVM 实测 2× vs Pyramid grouped 8-10×) → 三轴近独立 → **预期 A-joint ≈ A-serial**。已有 iso-budget 公平对照(base_isobudget AP50 0.626 > 所有剪枝档)。
- **跨模型对比的论点**: Pyramid 显耦合(A-joint≫A-serial)+ CoDriving 可分离(A-joint≈A-serial)= **"协同搜索的价值 = 耦合强度的函数"** 这一 model-dependent 判据的直接证据。两个结果都"有用": 一个证联合必要, 一个划定可分离适用区。

### 4.2 平台口径(铁律, 同旧 gap1 §8.3)
- **主平台 = H800 TVM**(已迁移): schedule 搜索 + 相对延迟在 TVM 坐实机理(MetaSchedule evolutionary_search + xgb cost model); prune×schedule 是主轴。
- **三臂必须同平台同口径**(否则消融不干净); 不拼跨平台 e2e。
- **(future/次要)量化轴 Q**: 本设计**不以 INT8/TRT 为重心**, Q 默认不进主搜索空间。若日后启用 Q 轴, INT8 须 BYOC-TRT(relax 无 INT8 pass), 量化臂延迟走 TRT 口径不与 TVM 数混加; 且 gap1 §10.5 已记录"INT8 simulated 不可信", 故 Q 轴要么真 TRT INT8, 要么标"待补"(§5/§8)。
- **AP 平台无关**(来自 4090/DAIR finetuned 真测); **延迟 H800 relative**。
- 空闲 GPU(mem≤50MiB), 微秒 kernel min-of-N。

### 4.3 指标(双度量 + 过程指标)
- **headline**: iso-AP70 latency 倍率(A-joint vs A-serial)[借鉴 CHaNAS Table 2 iso-acc 1.68×]。
- **严谨**: (AP70, latency, energy) 三目标 hypervolume + Pareto 支配关系 [我们的多目标升级, 三篇文献都没做]。
- **过程**(§2 核心): 收敛曲线、跨 seed hypervolume 分布 + 配对显著性、评估点云覆盖图。
- **AP 轴用 AP70**(信号最强, span 0.10); energy 用 J/frame(已有 E4 真测基础)。

---

## §5 预期结果 + 诚实退路 [我们的设计, 防 p-hacking]

**跑实验前把判读规则定死**:

| 结果 | 解读 | 论文 framing |
|---|---|---|
| Pyramid: A-joint 的 hypervolume 分布显著高于 A-serial(分布几乎不重叠), 且 §2.5 点云显示 P_g 分支被 A-serial(锁死 W_g 后)排除, 跨 seed/起点/顺序一致 | **结构性局部最优成立** | 故事 A 强版本: co-design 框架有直接 payoff, 联合搜不可省 |
| CoDriving: A-joint ≈ A-serial(分布大幅重叠) | **耦合弱 → 可分离** | **不是失败**, 是判据数据点(故事 B): 此模型属"可分离适用区", 串行够用。明确划定 co-design 何时必要 |
| 两者都 ≈(含 Pyramid) | 我们设计的 sweet spot 没构造出真 trade-off, 或耦合普遍弱 | 退 characterization: 报"协同价值条件化"+ 失败的诚实分析。**仍可发表**, 但 framing 调整 |

**判读铁律**:
- A-joint ≈ A-serial **本身就是结论**(协同价值条件化), 不是 bug。切勿硬凑成"联合大赢"(违纪, 一个反例打穿)。
- 区分"A-serial 没搜够"(收敛曲线还在涨)与"A-serial 到不了"(曲线 plateau)。只有后者支持局部最优论点。
- **预判**(诚实): 从已有证据, A-joint≫A-serial **很可能只在 Pyramid + 对齐/量化耦合边界显著**, CoDriving 平坦区≈串行。这正是"条件化"的预期形态, 也是双模型设计的目的。

---

## §6 落地步骤 [借鉴 gap1 §8 + 本文重点调整]

1. **小规模探针: 验证 rank-flip(W_g/P_g)存在**(§2.2/§2.5 开放问题①的前提)。先在 Pyramid 上离线扫一小批宽度的 (default-schedule 延迟, tuned 延迟), 确认存在"default 下 W_g 优、tuned 后 P_g 反超"的 rank-flip 宽度对。**若不存在, 整个论点无载体 → 先解决这个**。计划见 [`design_wg_probe_int8_plan_v1.md`](../../../paper/design_wg_probe_int8_plan_v1.md)。
2. **给 searcher_v0 加 schedule 轴 + 多目标评估器 + 串行/联合两种调度模式**(现 searcher_v0 仅 P/Q/D-routing, 无 schedule 维、无 metric 评估、无 Pareto)。三臂共用此内核, 只切"消哪轴(主线只消 schedule) / 是否分阶段锁定"。
3. **建 schedule-LUT**[借鉴 CHaNAS R·B·S + block-LUT]: 对每 (块, W) 跑一次内环 MetaSchedule 存最优 schedule + 实测延迟(Q 启用时再乘量化档)。外环加性组合 Lat_net=ΣLat_block, 防指数爆炸。复用 H800 已打通的 `{base,p50,trap25}_backbone.onnx` + `s2_2*` 脚本, 调优余量数据见 `results/gap1_schedule_lut.json`。
4. **cost model 修复**[借鉴 AutoTVM rank loss]: 外环 latency 预测器改 pairwise rank loss(现 LGB 回归跨数量级崩); 内环复用 TVM xgb_model。
5. **跑三臂 × N seed × 多起点 × 多贪心顺序**(§2.3 矩阵), 同预算同 cost model。
6. **算双度量 + 三种过程产物**(收敛曲线/点云/分布)。
7. **按 §5 判读规则定 framing**。

---

## §7 每篇文献 → 本设计的映射 [一览]

| 文献 | 启发 | 落到本设计 |
|---|---|---|
| **ALT-FP/BP**(Fig 11 微基准) | 串联算子单向传 layout 次优, 真跑两 tuning 流比收敛解(非枚举) | §2.1 锁死机制因果链; §0 "搜索过程而非枚举"的直接先例 |
| **ALT** "layout 变→loop 空间重建" | 单维变化重建下游空间 | §2.1 "剪枝宽度变→Q/S 空间重建" 同构; A-serial 锁死即排除分支 |
| **ALT-OL** 消融 | 砍 joint stage 只剩 loop, 证联合有独立收益 | §1 A-noS(消 schedule 轴)对照 |
| **CHaNAS-W vs W/O**(Table 2) | 同超网有/无调度协同, iso-acc 1.68× | §1 A-joint↔A-noS; §4.3 iso-AP70 headline |
| **CHaNAS Fig 2** | 最优架构随调度漂移, Pareto 相交 | §2.2 "default-schedule 排序错→漏掉只在 tuned schedule 显现的 P_g"; rank-flip / 锁死 W_g 必次优的立论 |
| **CHaNAS** R·B·S+block-LUT | 分解防指数爆炸 | §6.3 块级 schedule-LUT + 加性组合 |
| **CHaNAS** divisible-split | 限可整除("不可整除更差") | §2.1 in_per_g=2^k 对齐约束的他证 |
| **AutoTVM** Fig 4 收敛曲线 | cost model vs random 收敛快+终值好 | §2.4 收敛曲线证 A-serial "收敛快但封顶低"(贪心病征) |
| **AutoTVM** rank loss | rank > regression(只需相对快慢) | §6.4 修 latency 预测器 |

---

## §8 开放问题

1. **rank-flip(W_g/P_g)是否真存在?**(最大风险) 必须先用小规模探针(§6.1)确认 Pyramid 搜索空间里存在"default-schedule 下 W_g 优、tuned 后 P_g 反超"的宽度对。若不存在 —— 即各宽度的 default-schedule 排序与 tuned 排序一致(贪心按 default 选中的就是 tuned 后最优)—— 则 A-serial 不会被困, 论点无载体。**这是全设计的硬前提, 须最先验证。** 主候选来源: schedule 调优余量按宽度不均(`results/gap1_schedule_lut.json` base 9.06×/p50 8.72×/trap25 1.96×); future 候选: 只在 INT8+tensorize 下才显优的对齐宽度(default-FP16 下平庸)。探针计划见 [`design_wg_probe_int8_plan_v1.md`](../../../paper/design_wg_probe_int8_plan_v1.md)。
2. **(future/次要)量化轴 Q 的真实代价**: 本设计主线 = prune×schedule, **不依赖 Q**。若日后把 Q 作为另一条 rank-flip 来源(§2.2 future 项), 则必须有**真 INT8 AP + 真 INT8 延迟**(TRT INT8 / TensorRT-ModelOpt; gap1 已记录 INT8 simulated 不可信)。在拿到真 INT8 前, Q 轴标"待补", 不进主搜索空间。
3. **串行的"贪心"定义**: A-serial 第一阶段按什么准则选 W_g? 若按 (AP, default-schedule-latency) Pareto 选一个集合(而非单点)再进下一阶段, 它的可达空间更大, 可能削弱局部最优效应。需定义"串行"的强度: 单点锁定(最强局部最优) vs Top-k 锁定(较弱)。建议主报单点锁定(经典串行), 辅报 Top-k(展示即便放松仍不及联合)。
4. **预算公平性的口径**: A-serial 分阶段, 各阶段预算如何分配才算"与 A-joint 同总预算"? 是否给 A-serial 阶段间预算自由再分配的权利(否则被质疑预算切分不公)? 建议: 总评估次数相等, A-serial 阶段内预算可自适应分配, 排除"切分不公"反驳。
5. **多目标下"局部最优"的定义**: 单目标局部最优清晰; 多目标下 A-serial 收敛到的是一个 Pareto 集, "被结构性排除"如何严格定义? 建议用 hypervolume 间隙 + 被排除区域的支配点是否存在来操作化。
6. **energy 轴是否进搜索过程**: energy 与 latency 强相关(INT8 同时省两者), 进搜索可能冗余。建议 energy 作评估指标(报 hypervolume)但不必单独作搜索目标轴 — 待定。

---

## §9 ★实验结果 — B1–B5 真数据落地 (results v1, 2026-06-21)

> 本节回写 §2 四层证据的**实测结果**。全部基于真测: 延迟 = H800 TVM 直接网格(`results/latency_lut_pyramid.json`, gap1_grid_corrected + lut_results_grid relaunch, 真 default/tuned µs); AP70 = stage_a DepGraph fp16 锚点 + DepGraph 扩展 finetune(`results/ap70_model_pyramid.json` `table`, DAIR val 1789)。驱动器 `framework/run_b4_ablation.py` + 内核 `framework/search_three_arm.py`; 审计 `scripts/phase2/b5_verify_convergence.py`。产物: `results/b4_ablation_results.json` / `results/b5_convergence_verification.json` / `multi_agent/figure/b4_{hv_boxplot,convergence,pointcloud}.png`。

### 9.1 搜索网格 + rank-flip 对(§2.2 构造成功)
真测网格 = 8 宽度(均真 AP + 真延迟): p75/p50/base/trap25/pad64/mix_b/s1_64/mix_d。`detect_wg_pg_pairs` 从网格**自动识别** 2 组 default→tuned rank-flip 对(共享 (s1,s2), s0 失配 vs 补齐, 同 AP 由零填充权重恒等保证):

| 对 | W_g(default 更快, 被串行锁/排) | P_g(tuned 更快) | 同 AP70 | W_g tuned 余量 | P_g tuned 余量 | iso-AP 倍率 |
|---|---|---|---|---|---|---|
| 1 | trap25 [48,96,192] | pad64 [64,96,192] | 0.5905 | 1.96× | 7.71× | 3.51× |
| 2 | mix_b [48,64,256] | s1_64 [64,64,256] | 0.6362 | 2.14× | 7.88× | 3.29× |

rank-flip 真实存在(§8 开放问题①闭合): 失配 s0(in_per_g=3)tuned 调优余量仅 ~2×, 补齐 s0→64(in_per_g=4)tuned 余量 ~7.9× → tuned 后 P_g 远快于 W_g, 与 default 排序翻转。(pair3 mix_d/s2_128 因 s2_128 TVM 持续 CUDA 崩降级, 见 §9.5。)

### 9.2 层 2/3 — HV 分布 + 配对显著性(§2.3, 12 seed)
| 臂 | mean HV | %of ref | std |
|---|---|---|---|
| **A-joint** | 6.928e3 | **99.6%** | 4.3e1 |
| **A-serial** | 5.978e3 | **86.0%** | ~0 |
| **A-noS** | 3.337e3 | **48.0%** | 0 |

- **Wilcoxon A-joint vs A-serial: p=4.9e-4, rank-biserial=1.0**(12 seed 全偏向联合, 分布不重叠)。
- A-serial std≈0: 锁死机制确定性(每 seed 收敛到同一被限可达集)。**这是 framing-independent 的主结果** —— 串行系统性、可重复地劣于联合, 与具体 pair 解读无关。期望序 A-joint ⪰ A-serial ⪰ A-noS 成立。

### 9.3 层 1/4 — 结构性排除(§2.5, 跨 72 起点)
每对均 **PASS**(2 pair × 12 seed × 6 起点 = 72 多起点 run):
- A-joint 访问 (P_g, tuned): 两对都 **True**。
- A-serial 访问 (P_g, tuned): 两对都 **False**(结构性排除 = stage-1 default 排序丢弃 P_g → 其 tuned 引擎从不构建)。
- A-serial 跨全部 72 起点都丢弃 P_g(含**强制从 P_g 起步**的起点) → 结构性、非偶然/非起点差。
- 注: pair1 中 W_g(trap25)本身也被 mix_b 在 default 下支配(`W_g locked-repr=False`), pair2 中 W_g(mix_b)是被锁代表(`True`)。**核心排除判据是"P_g 被丢弃", 不依赖"W_g 恰被锁"** —— 已据此泛化(旧硬编码单对 pad64 判据已修)。

### 9.4 ★诚实区分: shipped 协同赢 vs 仅机理(B5 审计)
B5 审计**所有臂收敛 Pareto 解均为真测点**(latency∈直接网格, AP∈真 finetune)。但**两对的论证强度不同**:

| 对 | 类型 | A-serial 出货 | A-joint 出货 | P_g 在全局 Pareto? | 倍率 |
|---|---|---|---|---|---|
| **2 (s1_64)** | **shipped 协同赢** | mix_b/tuned 19405µs | **s1_64/tuned 5895µs** | **是** | **3.29×** |
| 1 (pad64) | 仅机理 | (无, trap25/pad64 均被支配) | (无) | **否(被 s1_64 支配)** | 3.51× |

- **pair2 是干净 headline**: iso-AP70=0.6362 下, A-joint 出货 s1_64/tuned(在全局 Pareto 上, 5895µs), A-serial 因 s1_64 未被锁→从不调优→只能出货 mix_b/tuned(19405µs)= **同精度 3.29× 更慢**, 两端点都真测、都在各自臂的 Pareto 上。这是真正的"串行漏掉一个联合能到的全局 Pareto 点"。
- **pair1 仅机理**: trap25/pad64 清晰展示 rank-flip(2× vs 7.7×)+ 结构性排除, **但 pad64 在本网格被 s1_64 全局支配**(s1_64 更快 5895<6152 且 AP 更高 0.6362>0.5905)→ pad64 不在任何臂的出货 Pareto 上 → 3.51× 是"同 AP niche 内"的机理示数, **非出货解倍率**。诚实标注, 不当 headline。
- **教训**: 真网格变大后(8 宽度), 低 AP 的 P_g 可能被高 AP 宽度全局支配。"shipped 协同赢"要求 P_g 在全局 Pareto 上。

### 9.5 闭环 DS 轴(用户要求, 已接入; AP 轴保留为主轴)
方案 B(`closedloop_b4_plugin_v1.md`)接入: `CostModel.evaluate` 的 rec 附带 `ds_model`/`e2e_orin_ms_est`(model-estimated, 不进 HV/搜索)。**AP70 仍为主目标轴, 未删**。每对同 AP 下 P_g 驾驶分显著高于 W_g:

| 对 | W_g tuned DS (e2e) | P_g tuned DS (e2e) | P_g 增益 |
|---|---|---|---|
| 1 (AP0.5905) | 85.6 (535.6ms) | 95.6 (215.0ms) | **+10.0 DS** |
| 2 (AP0.6362) | 87.8 (489.8ms) | 95.7 (209.7ms) | **+7.95 DS** |

seed0 各臂出货 Pareto 的 DS: A-joint 在 AP0.6362 出 DS=95.7 vs A-serial 出 DS=87.8(同精度差 ~8 DS); A-noS(无调度)全程 DS 50–92(最差)。**闭环从驾驶安全独立支撑结构性论点**: 串行锁 W_g 的代价不止延迟, 还有驾驶分。**口径**: model-estimated(CoDriving τ_perc 曲线 + H800→Orin 线性缩放 ±30%), β=0(无真 AP→DS 数据), 非真 Pyramid 闭环(需装 CARLA), 不得写 "real closed-loop"。

> ★**[2026-06-21 DS 计算勘误 — 用户指出, 当前 DS 模型有结构性缺陷, 后续估算必须修]**
> 当前 `DS = DS_lat(latency)`(β=0)= **把 Pyramid 延迟投影到 CoDriving 实测 τ→DS 曲线上, 这隐含假设 Pyramid 的感知精度/特征提取 = CoDriving** —— 不成立(两者不同模型、不同任务 DAIR vs V2Xverse)。后果: β=0 下**两个同延迟但不同 AP 的配置得到相同 DS**, 但低 AP = 漏检多 = 即便同延迟驾驶也更不安全。**DS 必须由 (AP, latency) 共同决定**, 不能只看延迟。
> - **本节内 W_g/P_g 对比仍然有效**: 对内 W_g 与 P_g **AP 严格相等**(零填充权重恒等), 故 +7.95/+10 DS 纯由延迟差驱动, 与 β 取值无关 —— 这个结论可信、可留。
> - **失效的是跨 AP 的 DS 比较**: 上面"各臂出货 Pareto DS"表跨配置 AP 不同(尤其 A-noS), β=0 下不可信, 仅作占位; 真正结论以 HV/Wilcoxon(§9.2)+ 对内 DS 差为准。
> - **后续估算修法**: ① 至少给 β>0 让 AP 进入(但 β 无实测标定, 仍是假设); ② 正解 = 建 **Pyramid 专属 DS(AP, τ) 曲面**(真闭环里同时扫 AP 与延迟, 见 §9.7 + 新交接 `HANDOFF_codesign_nextstage_v1.md`), 替换借来的 CoDriving 纯延迟曲线。在此之前, DS 仅用于"同 AP 对内"对比, 不做跨 AP 论断。

### 9.6 与 §5 判读规则对照 + 必带 caveat
- **落点 = 故事 A 强版本**(Pyramid 显耦合): A-joint HV 分布显著高于 A-serial(p=4.9e-4 不重叠), 点云显示 P_g 分支被 A-serial 排除, 跨 72 起点一致 → **结构性局部最优成立**, 联合搜不可省。
- **caveat(写论文必带)**: ① 网格 8 宽度偏小, 防"枚举"质疑靠"结构性排除在任何含 rank-flip 对的网格上成立 + 搜索器自主锁 W_g(非手挑)"立论(§3); 更大搜索需更多真 finetune(贵, 模型引导策略本为省它), 标 future。② AP 轴弱(过参数化, 单/混合剪枝 finetune 后 AP≈0.63 几乎不掉, 仅激进均匀剪枝才掉)但**保留为目标轴**。③ 延迟全程 H800, 不跨平台混。④ 闭环 = 估算非真 sim。⑤ pair1 是机理示范非出货倍率(§9.4)。

### 9.7 未完成 / 下一步
> **执行计划 + agent 分工见 `multi_agent/methods/progress/HANDOFF_codesign_nextstage_v1.md`。**
- **量化轴 Q**(§8 开放问题②, 用户强调别忘): prune×**quant**×schedule 三轴, 需真 TRT INT8(relax 无 INT8 pass, simulated 不可信), 跨口径不混。INT8 会放大 W_g/P_g 但不改本质。
- **pair3 补全**: s2_128 [64,128,128] TVM 调优持续 CUDA illegal-access 崩(2× exit134), 需排查后补第 3 个 AP 档的对。
- **CoDriving 对照臂**(§4.1): 预期 A-joint≈A-serial(可分离), 跑通则坐实"协同价值=耦合强度函数"的双模型判据。
- **真闭环 + Pyramid DS 曲面**: Pyramid→V2Xverse 移植 + CARLA 装好后, ① 用真 Orin e2e τ_perc sweep 替换 DS 估算; ② 建 **Pyramid 专属 DS(AP, τ) 二维曲面**(同时扫 AP 与延迟), 修 §9.5 勘误的"DS 须由 AP+latency 共同决定"。
