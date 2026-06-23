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
- **CoDriving (标准 conv)** = **已证可分离(SERIAL)** ★[2026-06-22 复核确认]。标准 conv 对宽度不挑, schedule 旋钮价值低(TVM 实测 2× vs Pyramid grouped 8-10×) → 三轴近独立。**C0c' 三臂实测 A-joint=A-serial=100%(SERIAL)** + C1cod 无宽度陷阱 + C6 高维 batch×width 无稳健陷阱 ⇒ 不再是"预期", 是干净阴性定论(详见 §9.8d')。已有 iso-budget 公平对照(base_isobudget AP50 0.626 > 所有剪枝档)。
- **跨模型对比的论点**: Pyramid 显耦合(A-joint≫A-serial, **已实证** §9)+ CoDriving **预期**可分离(A-joint≈A-serial, **当前欠实测, 待 L3 多点真测定论**, 见 §9.8a)= **"协同搜索的价值 = 耦合强度的函数"** 这一 model-dependent 判据。两个结果都"有用": 一个证联合必要, 一个划定可分离适用区(若真可分离)。**叙事修正**: 框架对两模型都优化, 差别在"是否必须联合搜" → 真正贡献 = **可从架构预测联合搜必要性的判据**(对齐敏感度: grouped/depthwise 或 INT8 → 耦合), 非"1:1 只对 Pyramid 有效"。详见 §9.8a。

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
真测网格 = 9 宽度(均真 AP + 真延迟): p75/p50/base/trap25/pad64/mix_b/s1_64/mix_d/**s2_128**。`detect_wg_pg_pairs` 从网格**自动识别** 3 组 default→tuned rank-flip 对(共享 (s1,s2), s0 失配 vs 补齐, 同 AP 由零填充权重恒等保证):

| 对 | W_g(default 更快, 被串行锁/排) | P_g(tuned 更快) | 同 AP70 | W_g tuned 余量 | P_g tuned 余量 | iso-AP 倍率 |
|---|---|---|---|---|---|---|
| 3 | mix_d [48,128,128] | s2_128 [64,128,128] | 0.6369 | 2.01× | 8.61× | **3.83×**(最大) |
| 1 | trap25 [48,96,192] | pad64 [64,96,192] | 0.5905 | 1.96× | 7.71× | 3.51× |
| 2 | mix_b [48,64,256] | s1_64 [64,64,256] | 0.6362 | 2.14× | 7.88× | 3.29× |

rank-flip 真实存在(§8 开放问题①闭合): 失配 s0(in_per_g=3)tuned 调优余量仅 ~2×, 补齐 s0→64(in_per_g=4)tuned 余量 ~7.7–8.6× → tuned 后 P_g 远快于 W_g, 与 default 排序翻转。**三对独立复现, 跨三个 AP70 档(0.5905/0.6362/0.6369)。**

**★[2026-06-21 收尾] pair3 已并入 3 对 ablation 重跑(L2 完成)**: s2_128 [64,128,128] 的 TVM 调优崩(seed=0 进化搜索为某 grouped conv 选出 OOB kernel, 在完整图 compile 时触发 CUDA illegal-access)已用 **seed=42 + fresh workdir + subprocess 验证前置** 修通(真测 `results/s2_128_fresh_tune_seed42.csv`: default 47435µs/tuned 5506µs/8.6×)。iso-AP 恒等已核实: s2_128 = mix_d 的 s0 零填充(48→64), AP70=0.6369(`ap70_model_pyramid.json` `table` 已含, 旧 T2 报的 0.590 系误填, 已弃)。**§9.2–9.4 的 HV/Wilcoxon/B5 已是 3 对网格重跑结果**(`run_b4_ablation --seeds 12` + `b5_verify_convergence`, 主控复跑核验)。

### 9.1b ★HV(hypervolume / 超体积)指标:计算方法 + 直白解读
> 三臂对比的主指标。下文先给精确算法(对应 `framework/search_three_arm.py::hypervolume_2d`),再用大白话说什么时候高/低。

**(1) 目标空间**:每个候选配置 = 平面上一个点 `(延迟 lat_µs, AP70)`。我们要"延迟越低越好 + AP70 越高越好"。为统一成最小化问题,第二轴取 `−AP70`,于是点写成 `(lat, −AP70)`,**两轴都是越小越好**。

**(2) 参考点 nadir `ref`**(右上角"最差角"):`ref = (网格内最大延迟 ×1.05, −(最小 AP70 − 0.02))`。它**离线算一次、对所有臂相同、从不喂给搜索器**(纪律:只用于打分,不泄露给搜索)。

**(3) HV = 一个臂的 Pareto 前沿所"支配/覆盖"的面积**(以 nadir 为边界):
1. 取该臂访问过的全部点 → 求非支配前沿(扔掉被别人"又快又准"盖过的点);
2. 只留比 `ref` 在两轴都更优的点;
3. 按延迟降序累加矩形:`HV = Σ (prev_lat − lat) × (ref_y − (−AP70))`。
单位 = 延迟(µs)× AP,是**相对量**,只用于同口径(同一 `ref`)比臂。报表里 `%of ref` = 该臂 HV ÷ **全局真 Pareto** 的 HV(离线算的理论上界)。

**(4) 什么时候 HV 高(大白话)**:**又快、又准、且前沿覆盖面广**。具体三种贡献叠加 ——
- 前沿在**每个 AP 档都压到真正低的延迟**(例:找到了 tuned 后的 P_g 引擎,同精度延迟砍到 ~1/4);
- 前沿**跨越完整 trade-off 范围**(从便宜低 AP 到昂贵高 AP 都有解);
- 非支配点多、把曲线填满。

**(5) 什么时候 HV 低**:**解慢、或精度低、或前沿有缺口(漏掉本可达的好点)**。例如 ——
- 解被支配(又慢又不准):A-serial 锁死失配宽度 W_g,只能用它"调优很差"的引擎 → 同 AP 下延迟高一大截 → 覆盖面积缩水;
- 前沿太短/有洞:A-serial 结构性**漏掉低延迟的 P_g** → 中段 AP 的延迟下不来;A-noS **从不调 schedule** → 几乎所有点都慢 → 面积最小。

**(6) 为什么用它当主指标**:一个标量就把"延迟 × 精度 × 覆盖广度"三件事压成可比数,**不用人为挑某个工作点**(避免"挑对自己有利的工作点"的偏见)。本文落点正是如此:A-joint 拿到 tuned-P_g 低延迟点 → 面积最大(≈100%);A-serial 结构性漏 P_g → 中段延迟高 → 面积小一截(85%);A-noS 不调 schedule → 全慢 → 面积最小(48–57%)。HV 差距 + Wilcoxon 配对显著性(§9.2)共同支撑"串行系统性劣于联合"。

### 9.2 层 2/3 — HV 分布 + 配对显著性(§2.3, 12 seed, 3 对网格, ref_hv=6.9913e3)
| 臂 | mean HV | %of ref | std |
|---|---|---|---|
| **A-joint** | 6.984e3 | **99.9%** | 2.4e1 |
| **A-serial** | 5.978e3 | **85.5%** | ~0 |
| **A-noS** | 3.337e3 | **47.7%** | 0 |

- **Wilcoxon A-joint vs A-serial: p=4.88e-4, rank-biserial=1.0**(12 seed 全偏向联合, 分布不重叠)。
- A-serial std≈0: 锁死机制确定性(每 seed 收敛到同一被限可达集)。**这是 framing-independent 的主结果** —— 串行系统性、可重复地劣于联合, 与具体 pair 解读无关。期望序 A-joint ⪰ A-serial ⪰ A-noS 成立。

### 9.3 层 1/4 — 结构性排除(§2.5, 跨 96 起点/对)
每对均 **PASS**(3 pair × 12 seed × 8 起点 = 96 多起点 run/对, STRUCTURAL RESULT: PASS):
- A-joint 访问 (P_g, tuned): 三对都 **True**。
- A-serial 访问 (P_g, tuned): 三对都 **False**(结构性排除 = stage-1 default 排序丢弃 P_g → 其 tuned 引擎从不构建)。
- A-serial 跨全部 96 起点都丢弃 P_g(含**强制从 P_g 起步**的起点, 96/96) → 结构性、非偶然/非起点差。
- 注: pair3(mix_d)/pair2(mix_b)的 W_g 是被锁代表(`W_g locked-repr=True`), pair1 中 W_g(trap25)本身也被 mix_b 在 default 下支配(`False`)。**核心排除判据是"P_g 被丢弃", 不依赖"W_g 恰被锁"** —— 已据此泛化(旧硬编码单对 pad64 判据已修)。

### 9.4 ★诚实区分: shipped 协同赢 vs 仅机理(B5 审计, 3 对)
B5 审计**所有臂收敛 Pareto 解均为真测点**(latency∈直接网格, AP∈真 finetune)。**三对论证强度不同**(`b5_convergence_verification.json`):

| 对 | 类型 | A-serial 出货 | A-joint 出货 | P_g 在全局 Pareto? | 倍率 |
|---|---|---|---|---|---|
| **3 (s2_128)** | **shipped 协同赢** | mix_d/tuned 21071µs | **s2_128/tuned 5506µs** | **是** | **3.83×** |
| 1 (pad64) | 仅机理 | (无, trap25/pad64 均被支配) | (无) | **否(被 s2_128 支配)** | 3.51× |
| 2 (s1_64) | 仅机理 | mix_b/tuned 19405µs | (无, s1_64 被 s2_128 支配) | **否(被 s2_128 支配)** | 3.29× |

- **pair3 是干净 headline(新, 取代旧 pair2)**: iso-AP70=0.6369 下, A-joint 出货 s2_128/tuned(在全局 Pareto 上, 5506µs), A-serial 因 s2_128 未被锁→从不调优→只能出货 mix_d/tuned(21071µs)= **同精度 3.83× 更慢**, 两端点都真测、都在各自臂的 Pareto 上。这是真正的"串行漏掉一个联合能到的全局 Pareto 点", 且是三对中最大倍率。
- **pair2 由 shipped 降为仅机理**: 加入 s2_128 后, s2_128(5506µs, AP0.6369)**全局支配** s1_64(5895µs, AP0.6362)—— 更快且 AP 更高 → s1_64 不再在全局 Pareto 上 → pair2 的 3.29× 退为机理示数(原 §9 旧版以 pair2 为 shipped headline, 已被 pair3 取代)。
- **pair1 仍仅机理**: trap25/pad64 清晰展示 rank-flip(2× vs 7.7×)+ 结构性排除, **但 pad64 被 s2_128 全局支配** → 3.51× 是"同 AP niche 内"机理示数, 非出货解倍率。
- **教训**: 真网格变大后(9 宽度), 低 AP 的 P_g 可能被更高 AP 且更快的宽度全局支配。"shipped 协同赢"要求 P_g 在全局 Pareto 上 —— 当前唯一满足者 = **pair3(3.83×)**。

### 9.5 闭环 DS 轴(用户要求, 已接入; AP 轴保留为主轴)
方案 B(`closedloop_b4_plugin_v1.md`)接入: `CostModel.evaluate` 的 rec 附带 `ds_model`/`e2e_orin_ms_est`(model-estimated, 不进 HV/搜索)。**AP70 仍为主目标轴, 未删**。每对同 AP 下 P_g 驾驶分显著高于 W_g:

| 对 | W_g tuned DS (e2e) | P_g tuned DS (e2e) | P_g 增益 |
|---|---|---|---|
| 3 (AP0.6369) | 86.2 (524.3ms) | 95.8 (201.7ms) | **+9.67 DS** |
| 1 (AP0.5905) | 85.6 (535.6ms) | 95.6 (215.0ms) | **+10.0 DS** |
| 2 (AP0.6362) | 87.8 (489.8ms) | 95.7 (209.7ms) | **+7.95 DS** |

seed0 各臂出货 Pareto 的 DS: A-joint 在 AP0.6362 出 DS=95.7 vs A-serial 出 DS=87.8(同精度差 ~8 DS); A-noS(无调度)全程 DS 50–92(最差)。**闭环从驾驶安全独立支撑结构性论点**: 串行锁 W_g 的代价不止延迟, 还有驾驶分。**口径**: 当前表中数字 = model-estimated(CoDriving τ_perc 曲线 + H800→Orin 线性缩放 ±30%), β=0, **仅作占位**。★[2026-06-21 方法更新, 用户拍板] **延迟的正确口径 = Orin 真实 e2e 实测(不再用 H800→Orin 估算)**;真闭环 DS 实验(含 Orin 真测 τ)**已委托其他 agent 进行**, 产出后替换本表估算值。在此之前不得写 "real closed-loop"。

> ★**[2026-06-21 DS 计算勘误 — 用户指出, 当前 DS 模型有结构性缺陷, 后续估算必须修]**
> 当前 `DS = DS_lat(latency)`(β=0)= **把 Pyramid 延迟投影到 CoDriving 实测 τ→DS 曲线上, 这隐含假设 Pyramid 的感知精度/特征提取 = CoDriving** —— 不成立(两者不同模型、不同任务 DAIR vs V2Xverse)。后果: β=0 下**两个同延迟但不同 AP 的配置得到相同 DS**, 但低 AP = 漏检多 = 即便同延迟驾驶也更不安全。**DS 必须由 (AP, latency) 共同决定**, 不能只看延迟。
> - **本节内 W_g/P_g 对比仍然有效**: 对内 W_g 与 P_g **AP 严格相等**(零填充权重恒等), 故 +7.95/+10 DS 纯由延迟差驱动, 与 β 取值无关 —— 这个结论可信、可留。
> - **失效的是跨 AP 的 DS 比较**: 上面"各臂出货 Pareto DS"表跨配置 AP 不同(尤其 A-noS), β=0 下不可信, 仅作占位; 真正结论以 HV/Wilcoxon(§9.2)+ 对内 DS 差为准。
> - **修法(★已委托其他 agent 进行, 不列入本文档/handoff 的下一步计划)**: ① 延迟用 **Orin 真实 e2e 实测**(替换 H800→Orin 估算); ② 正解 = 建 **Pyramid 专属 DS(AP, τ) 曲面**(真闭环里同时扫 AP × Orin 真测延迟), 替换借来的 CoDriving 纯延迟曲线。**该真闭环 DS 实验由其他 agent 负责**(本团队不重复安排)。在此之前, 本文档 DS 仅用于"同 AP 对内"对比, 不做跨 AP 论断。
> - **注**: 上述只针对**闭环 DS 的延迟口径**;**ablation rank-flip 延迟(§9.1–9.4)仍是 H800 TVM**(耦合本身是 TVM 调优余量现象, 非边缘部署数), 两者不混。

### 9.6 与 §5 判读规则对照 + 必带 caveat
- **落点 = 故事 A 强版本**(Pyramid 显耦合): A-joint HV 分布显著高于 A-serial(p=4.9e-4 不重叠), 点云显示 P_g 分支被 A-serial 排除, 跨 72 起点一致 → **结构性局部最优成立**, 联合搜不可省。
- **caveat(写论文必带)**: ① 网格 9 宽度偏小, 防"枚举"质疑靠"结构性排除在任何含 rank-flip 对的网格上成立 + 搜索器自主锁 W_g(非手挑)"立论(§3); 更大搜索需更多真 finetune(贵, 模型引导策略本为省它), 标 future。② AP 轴 = **中段高原 + 高段 soft-knee**(§9.8f, 2026-06-22 真测): wholenet 剪枝 finetune 后 AP70 在 ≤84% 近平(高原, 坐实 iso-AP 框架——co-design 收益纯在延迟维), 84–93% 加速下降(AP70 0.585→0.537, total range 0.094=94× 噪声)→ AP 轴**有真信息非"平到没用"**;保留为目标轴。③ 延迟全程 H800, 不跨平台混。④ 闭环 = 估算非真 sim。⑤ pair1/pair2 是机理示范非出货倍率, **唯一 shipped headline = pair3(3.83×)**(§9.4)。

### 9.8 ★[2026-06-22 更新] 跨阶段结果落地
> **本轮(2026-06-22)全部完成**: (a) CoDriving 可分离 ✅ / (b) Q 量化轴真 int8 ✅ / (d) **三轴 P×Q×S ablation ✅(补齐"三轴强耦合"缺口)** / (e) L4 latency 网格扩充 ✅(3 对 + 边界刻画) / (f) **L4 AP 崖口 ✅(soft_knee, range 0.094)** / (c) 闭环 smoke ✅。全部未 commit(等用户授权)。

**(a) CoDriving 对照臂 — ✅ [2026-06-22] fp16+int8 双轴均可分离(L3 复验完成, `codriving_coupling_verdict.md`)**
复用同一三臂内核换 CoDriving 数据:A-joint=A-serial=100.0% / A-noS=96.3% / 0 rank-flip 对。**早先"欠实测暂不接受"已用真测 s0 探针 + int8 复验补齐**:

| 模型 | A-joint | A-serial | A-noS | rank-flip 对 | 耦合类型 | 证据 |
|---|---|---|---|---|---|---|
| CoDriving | 100.0% | 100.0% | 96.3% | 0 | **可分离/弱** | ✅ s0 探针无 rank-flip(真测)+ int8 两宽度均 build 无陷阱(真测) |
| Pyramid(本文) | 99.9% | 85.5% | 47.7% | 3 | **强/定性** | ✅ 多宽度真测 + 12 seed + 三轴 P×Q×S(§9.8d) |

**可分离的真测依据(两条独立)**: ① **fp16 s0 失配探针**(固定 s1=128/s2=256, 仅变 s0, `codriving_s0probe_fp16.csv`): s0_32/48/64 default 单调、tuned 无对齐 rank-flip(s0_48 tuned 崩 = CUDA artifact 非耦合)。② **int8 复验**(G1, batch=2, groups=1, `codriving_int8_verify.json`): Cin=48(K÷16=27✓)和 Cin=64(K÷16=36✓)**均 build 成功**(int8 1.42×/1.32×, max_rel_err=0.0), **排序不变无 rank-flip** → 标准 conv int8 至多软效率惩罚, **无 Pyramid 那种定性 build-or-not 陷阱**。⇒ **CoDriving 可分离已坐实**(注: int8 WMMA 确认走 MetaSchedule DB trace + 数值 0.0, 非直接 CUDA grep —— relax VMExecutable 工具限制, 见 verdict §3 caveat; 但"能 build"的定性对照不依赖 WMMA/dp4a 核路径)。残留 caveat: 网格仍偏小, p25/p75 fp16 绝对值跨 batch 不可直接比(标记在案)。

**★叙事修正(关键): 不是"1:1 对照(框架只对 Pyramid 有效)", 而是"可从架构预测联合搜必要性的判据"**:
- 框架对**两个模型都优化**(CoDriving 也拿 2.07–2.26× TVM 加速);差别在**是否必须联合搜**。"co-design 何时必要"本身就是贡献, 不是"1 赢 1 输"。
- **架构判据(原理)**: `prune×schedule 耦合 ⟺ 高效核可用性依赖某通道数是硬件 tile 整数倍`。Pyramid grouped conv 每组通道(in_per_g)须对齐 MMA-K → 剪枝改变 tensor-core 路径可用性 → 调优余量随宽度剧变(2× vs 7.9×)→ 耦合;标准稠密 conv(CoDriving)归约维连续, 任意宽度都良好 tile → 余量近常数 → 解耦。机理 iso 消融(stage0 主导)支持此判据。
- **普适性**: grouped/depthwise conv 统治高效边缘网络(MobileNet/EfficientNet/ResNeXt/RegNet/ShuffleNet)→ 框架目标人群多数耦合;纯稠密 conv 是少数。
- **★INT8 放大假设 — [2026-06-22] 已测, 对标准 conv 证否**: 原假设"INT8 WMMA K 需更严对齐 → 量化态连标准 conv 都变耦合"。**L3 int8 复验证否(对 CoDriving)**: Cin=48(K=432, ÷16=27)和 Cin=64(K=576, ÷16=36)都对齐 WMMA(K=Cin×9, Cin≥2 时 K÷16 总成立)→ 标准 conv 在 int8 下**仍可分离, 无 rank-flip**。⇒ 量化**不会**把标准 conv 变成 Pyramid 那种定性陷阱; 耦合仍是 **grouped conv 专属**(根因 = grouped 的 per-group in_per_g 极小 → NCHWc IC_BN=4 直接除不尽; 标准 conv 的 K=Cin×9 永远够大)。这反而**强化**了架构判据: 耦合 ⟺ 高效核可用性依赖小通道数对齐, 是 grouped/depthwise 的结构特征。

**(b) Q 量化轴 — ✅ [2026-06-22] 真 int8 已实现 + 定性耦合机理证实(QDQ-ONNX 路线证否 → 直接 topi NCHWc dp4a/WMMA)**
- **★关键负结果(已定论)**: **TVM relax QDQ-ONNX 路线对 grouped conv 出的是假 int8**(FP32-GEMM + QDQ overhead: TIR conv buffer=float32, int8-default 普遍比 fp16 慢 11–44%; 之前 `screenA` 的"WMMA/int8"是 groups=1 稠密 conv, 不能外推到 grouped)。⇒ QDQ-ONNX→relax 路线作废, 不可作 headline。
- **★解决(本轮工程成果)**: **直接用 topi `conv2d_NCHWc_int8` 建 int8 grouped conv = 真 int8**, 主控独立核验(`results/q_int8_ms_stage0_result.json` + `int8_correctness_verify.json` + `q_int8_dp4a_pairs.csv` + `q_tvm_int8_verdict.md`):
  - 生成 CUDA 含 `__dp4a` + PTX `dp4a.u32.s32`;MetaSchedule 在 sm90 选了更快的 **WMMA INT8**(`wmma::mma_sync` + INT8 fragment + INT32 累加 + 动态 shared mem)。
  - **数值精确**: max_rel_error = **0.0**(int8×int8→int32 精确整数), spot 521.0==521.0。
  - stage0 单 conv: int8 **150.1µs vs fp16 217.5µs = 1.45×**(H800-GPU6 真测)。
- **★定性耦合(头条机理 — 比 fp16 的定量 rank-flip 更强)**: NCHWc int8 须 `in_per_g=s0/16` 整除 4(dp4a 4-int8 打包)。**s0=48(in_per_g=3)→ NCHWc IC_BN=4 结构性不可能 → int8 不可 build**(`q_int8_dp4a_pairs.csv` 真测 NOT_APPLICABLE);**s0=64(in_per_g=4)→ 可 build, 1.45×**。⇒ 对齐耦合从 fp16 定量(2× vs 7.9×)**升为 int8 定性(能/不能 build)**: 串行按 fp16-default 锁失配 W_g(s0=48)= 失去**快的 NCHWc-dp4a** int8 路径(★见 §9.8d' 精修: NCHW int8 fallback 仍在但与 fp16-tuned 打平→失配宽度无 int8 加速,对齐宽度才有 1.45×,这正是耦合本身)。
- **int8 AP**(`results/q_int8_ap.json`): DAIR val 1789, Δap70 ≈ **-0.008**(TRT MinMax 真测, 非 simulated)。
- **口径限制(务必带)**: int8 延迟数 = **stage0 单 conv 微基准(s0=64), 非全 backbone**; 3 个 P_g 对 stage0 维度相同 → 同 150µs, 不区分对。全 backbone int8(stage1/stage2)= 未做的后续工作。定性的"能/不能 build"判据是 robust 结果, int8 延迟量级是 proxy。

**(d) ★[2026-06-22] 三轴 P×Q×S ablation — ✅ PASS(三轴强耦合已证, 补齐缺失的量化证明)**
把定性 int8 buildability 约束接进 `framework/search_three_arm.py` 的 `CostModelPQS`(`enforce_int8_buildable` + 真测 1.449× 作 H800-pure uniform int8 proxy), 由 `framework/run_pqs_ablation.py` 跑 12 seed 三臂 P×Q×S(`results/pqs_ablation_results.json`):

| 臂 | HV(% of A-joint) | 解释 |
|---|---|---|
| **A-joint-PQS** | **100.0%** | 同搜 P×Q×S → 在 4 个对齐宽度(s0=64)上拿到 int8 |
| **A-serial-PQS**(公平串行) | **85.3%** | stage1 锁 fp16-default 宽度 → stage2 调 schedule×quant(**不禁 int8**) |
| A-noS-PQS | 56.7% | P×Q 不调 schedule |

- **Wilcoxon A-joint vs A-serial: p=4.88e-4, rank-biserial=1.0**(12 seed 配对全胜)。
- **★三轴耦合机理(最强形态)**: A-serial 的 stage1 fp16-default 锁定前沿 = `[16,32,64],[32,64,128],[48,64,256],[48,128,128]` —— **全部 int8 不可 build**(s0∈{16,32,48}, in_per_g 不整除 4), 因为 **fp16-default-fast 前沿本身就被失配-s0 宽度支配**。⇒ A-serial 即使在 stage2 主动调 quant, 锁定的宽度也建不出 int8 → **int8 命中 0 个宽度**;A-joint 在 **4 个对齐宽度**上拿到 int8。**同一个 s0 对齐属性同时 gate 剪枝宽度选择、schedule 可调性、int8 可 build 性 = 三轴强耦合**(`categorical_pass=True`, `serial_misaligned_int8_on_pareto=False`)。
- 图: `multi_agent/figure/pqs_hv_boxplot.png` + `pqs_pareto_int8.png`。
- **口径**: int8 延迟 = H800 fp16/1.449 uniform proxy(全 H800, 不混 4090); 定性 buildability 是 robust 主结果, 延迟量级是 proxy。这补齐了 §0 诚实表里"三轴强耦合 ❌ 未证"的缺口 → **现为 ✅ 已证**。

**(d') ★[2026-06-22] 耦合叙事统一 — 三臂消融 = 耦合"度量仪"; 主张从"三轴不可约"改为"内外环(软×硬)耦合, P-hub"**

★**口径统一(与 doc1 §0.1 一致)**: 我们要主张的协同**不是 P/Q/S 三轴各自独立不可约**, 而是 **内环(硬件调度 S)↔外环(软件 P×Q)的强耦合**, 经 **P(IC_BN)枢纽**。而 **(d) 的三臂消融(A-joint vs A-serial vs A-noS)本身就是普适的"耦合度量仪"**: HV 比量化每个架构的内外环耦合强度 —— **"耦合/可分离"是被这把尺子测量出来的, 不是手写规则**。Pyramid 测出强耦合(A-serial 85.3%<A-joint), CoDriving 测出可分离(C0c' A-serial=A-joint=100%), V2X-ViT 居中。`grouped conv→小 IC_BN→强耦合` 是**机理 insight(科学发现)**, 非 operative 规则。

(d) 的三臂 ablation PASS(A-joint 100% > A-serial 85.3%, Wilcoxon p=4.88e-4)**本身有效不撤** —— 它真实证明了"联合搜 > 串行搜"(即内外环不可独立优化)。但后续 coupling-map 逐 cell 深挖(`coupling_map_v1.md` + `results/coupling_map/`, trackA+main 双核验)**精修了机制解读**, 把曾经设想的"三维独立不可约"诚实降级:

| 机制 | 原设想 | 复核后真相 | 是否经 P |
|---|---|---|---|
| mech1 (int8 buildability gate) | 硬约束/独立腿 | **format 障壁,延迟中性** —— C3 best-vs-best(全 tuned): misaligned IC=96 上 FP16-tuned(139.29µs)≈ INT8-NCHW(140.92µs)打平, **都快过**对齐后 padded-NCHWc-tuned(149.45µs); "建不出 int8"实为"建不出**快的 NCHWc-dp4a** int8", padding 修对齐不划算; INT8 优势仅 vs 未调优 FP16 | 经 P(IC_BN) |
| mech2 (IC_BN→MS gain scaling) | 第三条独立腿 | **REAL, 唯一存活的真核心** —— C2 确认 argmin_S rank-flip(native-WMMA@IC_BN16 → padded-WMMA@IC_BN4), 最优 (Q,S) 对 IC_BN 依赖 | 经 P(IC_BN) |
| mech3 (Q×S 独立于 P) | "不经 P"的最强腿 | **FALSE = BACKEND_ARTIFACT** —— FP16 **也走 WMMA half**(g8 gain 8.59×/g32 1.64×); 原"FP16-MS 0/100 valid"是 `count_db_valid()` 解析 bug + `write_c7_verdict.py:44` 硬编码 literal(g8 从未测) | — |

**诚实结论**:
- (d) 的三轴耦合**机理全部经 s0/IC_BN(=P)枢纽**(line 329 原文"同一个 s0 对齐属性同时 gate 三者"本就是 P-hub 表述)。深挖后确认: **没有"不经 P 的不可约 Q×S 耦合"的干净证据**(mech3 倒)。⇒ 正确表述 = **P(IC_BN) 是 hub 的 P×(Q+S) 联合依赖**, 框架仍须联合搜(因 P 牵动 Q、S 的可行域与最优 schedule), 但**不是**三轴彼此独立纠缠的"三维不可约"。
- (d) 的 int8 proxy(fp16/1.449)代表**NCHWc-dp4a 张量化 int8**; A-serial"int8 命中 0 宽度"应精确读作"**快 NCHWc-int8** 命中 0", A-joint 的优势 = 在对齐宽度上锁定**快 int8 路径**。这不削弱 ablation(joint 仍独得快 int8), 但把二元"可/不可 build int8"修正为"NCHWc-张量化可/不可"。
- **CoDriving 对照(标准 conv groups=1)**: C0c' 三臂 SERIAL(A-joint=A-serial=100%) + C1cod 无宽度陷阱 + C6 高维 batch×width 无稳健陷阱 ⇒ **三轴耦合是 grouped-conv(Pyramid)特有, 非普适**; 标准 conv 可分离。这是 goal② 的干净阴性结论。
- 本轮复核拦下 3 处假阳性(C1cod p25 cast-artifact / C6 batch 噪声 rank-flip / mech3 DB-bug+硬编码), 共同教训: **int8-vs-fp16 绝对速度不是耦合判据(cast-chain artifact); 后端规则覆盖缺失 ≠ 算法本征不可约**。汇总矩阵: `results/coupling_map_matrix.json`(9/9 cell)。

**(e) ★[2026-06-22] L4 网格扩充 — rank-flip 是 (s1,s2) 条件性的(边界刻画, 诚实负结果)**
G2 在 GPU6 真测扩了 8 对(s0=48 vs s0=64), 结果: **仅原 3 对是严格 rank-flip, 新增 5 对均非翻转**(`results/l4_new_widths.csv` + `latency_lut_pyramid_l4.json`):

| pair | W_g(s0=48) | P_g(s0=64) | default 赢家 | P_g tuned | flip? | AP70 |
|---|---|---|---|---|---|---|
| 1 trap25/pad64 | 42355 | 47407 | **s0=48** | 7.71× | ✅ | 0.5905 |
| 2 mix_b/s1_64 | 41460 | 46445 | **s0=48** | 7.88× | ✅ | 0.6362 |
| 3 mix_d/s2_128 | 42447 | 47435 | **s0=48** | 8.61× | ✅ | 0.6369 |
| 5 [48,32,128]/[64,32,128] | 43203 | 40468 | s0=64 | 3.58× | ✗ | — |
| 7 [48,128,192]/[64,128,192] | 60129 | 51992 | s0=64 | 7.18× | ✗ | — |
| 8 [48,96,128]/[64,96,128] | 49382 | 42786 | s0=64 | 2.91× | ✗ | — |
| 4 [48,96,256]/[64,96,256] | 54060(min, 干净复测) | 51622 | **s0=64** | 7.47× | ✗ | 0.5996 |

- **诚实结论**: rank-flip **不是普适的, 是 (s1,s2)-regime 特定的**。翻转需两条件同时成立: ① s0=48 在 default 下领先(latency 更低); ② s0=64 tuned 余量 7-8×(而 s0=48 仅 ~2×)。新 (s1,s2) 组合里 **dlight 默认已偏好 s0=64**(8 对中 5 对 s0=64 default 更快)→ 贪心直接选对齐宽度 → 无 trap。
- **机理精化**: 失配(s0=48, in_per_g=3)不仅 tuned 余量小, 连 **default schedule 也常被惩罚** —— 即使通道更少 FLOPs 更低, 对某些 (s1,s2) 其 default 延迟反而 > 对齐的 s0=64。trap 只出现在"失配的 default 惩罚还没盖过 FLOPs 优势"的窄区(原 3 对的 (s1,s2))。
- **pair4 干净复测定论(主控亲跑, GPU6)**: 原 default=103297µs 确是 contention glitch, 但**干净复测 default = 61137µs(mean)/54060µs(min), 仍 > partner iso_s1 的 51622µs** → s0=64 default-faster → **pair4 确认 non-flip**(`b1_results/wg_pair4_default_remeasure.json`)。它有真 AP70=0.5996 但因 default 排序不构成贪心陷阱, **不 ship**。⇒ 严格 rank-flip 锁定为 **3 对**(诚实, 不强凑第 4)。
- **这反而让故事更强(不是更弱)**: 我们能**刻画 trap 何时发生**(贪心陷阱的边界条件), 而非空泛宣称"剪枝总是陷阱"。3 对是 shipped 证据(已在 §9.2–9.4 ablation 内), 不需重跑。

**(f) ★[2026-06-22] L4 AP 崖口(DELIVERABLE A)— ✅ 完成: soft-knee, AP 轴从高原变真曲线(`ap_cliff_l4.json`)**
G3 在 4090 跑 **wholenet 剪枝**(backbone+deblocks+shrink 一起剪, 之前 backbone-only 封顶 ≤67% 是"无崖"假象的根因)到 84/89/93% 总压缩 + stage_a finetune(DAIR val 1789 真测, 全部 weights_loaded_verified=True/rc=0):

| total prune% | params | AP70 | slope/5% |
|---|---|---|---|
| 0.0%(base) | 5.46M | 0.6309 | — |
| 75.3%(all3_hard) | 1.35M | 0.5900 | −0.0027 |
| 83.9%(wn_80) | 0.879M | 0.5850 | −0.0029(高原) |
| 89.2%(wn_87) | 0.593M | 0.5613 | −0.0224(**8× 加速**) |
| 93.3%(wn_93) | 0.368M | 0.5369 | −0.0298(继续陡) |

- **判读 = `soft_knee`**(非 hard cliff): `cliff_found=False`(无单步 >0.03), 但 **AP70 total range = 0.094 = ~94× 管线噪声(0.001)** → **AP 轴携带真信息**, 拐点在 **~84–89%**(高原后斜率 8× 加速)。
- **★对 headline 的意义(双向都有用)**: ① **75–84% 高原区**(AP 几乎不动)**坐实 iso-AP 框架** —— 我们的 3 对 rank-flip/三轴耦合都建在"同 AP 不同延迟"的 iso-AP 恒等上, AP 在中等剪枝率近平 = co-design 收益纯在**延迟维**, 正是这条故事线的前提。② **>84% 的加速下降**给 AP 轴**真实操作范围**(span 0.094), 把"AP 轴弱"(§9.6 caveat ②)从"平到没信息"修正为"中段平→高段有真 trade-off"。
- **诚实**: 这是 soft knee 非戏剧性悬崖(DAIR 对 Pyramid 过参数化, 与 CLAUDE.md §7 一致); backbone-only ref(cliff_a 62.3% AP0.5755)在 JSON 里标为**不同剪枝策略**, 不混入 wholenet 崖口曲线。图: `multi_agent/figure/ap_cliff_wholenet.png`。

**(c) 真闭环 RSU smoke — ✅ G1 PASS(可行性证明)**
Pyramid→V2Xverse 移植 P1–P5 经核验**实际已完成**(原 handoff "CARLA 未装/闭环不可运行" 是 stale)。RSU-enabled Pyramid 闭环 r0 真跑通: **DS=100 / RC=100 / status=Completed, 无 Traceback**(`results_driving_pyr_rsu_smoke/.../results.json`)→ Pyramid+RSU 闭环可行性确认。τ_perc sweep 4 档 config 已建。**DS(AP,τ) 2D 曲面(修 §9.5 勘误)仍待 G3–G6**(见 §9.7)。

### 9.7 未完成 / 下一步
> **下一阶段执行计划 + agent 并行分工见 `multi_agent/methods/progress/HANDOFF_codesign_nextstage_v2.md`(取代 v1)。**

| 项 | 状态 | 下一步 |
|---|---|---|
| pair3 补全(L2) | ✅ **完成** — 延迟 rank-flip 3.83× 真测 + 并入 3 对 b4/b5 重跑 + iso-AP0.6369 恒等确认 + shipped headline 转 pair3 | — |
| CoDriving 对照臂 | ✅ **[2026-06-22] 完成** — fp16 s0 探针无 rank-flip + int8 两宽度均 build 无陷阱(1.42×/1.32×, max_rel_err=0.0)= 可分离坐实(`codriving_coupling_verdict.md`, `codriving_int8_verify.json`) | (可选)更大网格;int8 WMMA 直接 CUDA grep(relax VM 工具限制待解) |
| Q 量化轴 + 三轴 ablation | ✅ **[2026-06-22] 完成** — 真 int8(dp4a/WMMA, max_rel_err=0.0, 1.45×)+ 定性耦合(s0=48 结构性建不出**快 NCHWc-dp4a** int8;NCHW fallback 仍在但失对齐加速,§9.8d')+ **三轴 P×Q×S ablation PASS**(A-joint 100%/A-serial 85.3%, Wilcoxon p=4.88e-4, §9.8d;机制为 P-hub 非三维独立,§9.8d') | 全 backbone int8(可选, 把延迟从 proxy 升真测)|
| 真闭环 DS 曲面 | 🔄 G1 smoke PASS | ★**已委托其他 agent**(本团队不安排)。延迟口径 = **Orin 真测**(非估算);DS 须 (AP,latency) 共同决定。本团队仅按需提供 body 延迟/AP |
| 网格扩充 T1b(L4 latency) | ✅ **[2026-06-22] 实测定论** — 扩 8 对真测, **仅原 3 对 rank-flip**; 新 5 对非翻转(含 pair4 干净复测 default 54060>51622 = 确认 non-flip)→ **rank-flip 是 (s1,s2) 条件性**(边界刻画, §9.8e) | — (诚实锁定 3 对, 不强凑) |
| AP 崖口(L4 DELIVERABLE A) | ✅ **[2026-06-22] 完成** — wholenet 84/89/93% finetune(DAIR val 1789): AP70 0.585/0.561/0.537, **soft_knee**(range 0.094=94×噪声, 拐点 84-89%), AP 轴从"高原"变"真曲线"(§9.8f, `ap_cliff_l4.json`) | (可选)补 78/86% 加密拐点 |

- **DS(AP,τ) 修法**: 真闭环里同时扫 AP(剪枝/量化档)× 延迟(τ)→ 建 Pyramid 专属 2D 曲面替换借来的 CoDriving 纯延迟曲线, 修 §9.5 勘误。
