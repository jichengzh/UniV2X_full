> ⚠️ **已被 [HANDOFF_codesign_unified_v1.md](HANDOFF_codesign_unified_v1.md) 取代(2026-06-18)** —— 本文档的叙事/gap 内容已并入该统一交接的 Part II(并新增 §13 Gap1 实验设计讨论)。接手只读 unified 版; 本页仅作历史留存。

# HANDOFF — 论文故事综合 + 自洽性检验 + gap/补实验 (v1, 2026-06-18)

> **定位**: 不是实验交接, 是**把目前所有结论组织成顶会论文的叙事架构 + 诚实的 gap 分析**。接手据此判断"故事是否完整自洽 + 还要补哪些实验"。
> 衔接: 既有 method 稿 `paper/stage1_method_zh_v1.md`(可配置网络)+ `paper/stage2_method_zh_v1.md`(协同优化+耦合陷阱); 研究总纲 `background/00_研究目标与实验档案_v1.md`; TVM 线 `HANDOFF_tvm_codesign_crossmodel_v1.md`。

---

## §0 一句话诊断

既有故事主线 = **"P(剪枝)×Q(量化)×D(部署)存在耦合陷阱 → 串行优化次优 → 必须联合搜"**。本会话新证据对它**既加固也削弱**:
- **加固**: 对齐(÷32)是跨 P/Q/D 三轴的硬约束(trap25 在剪枝 0.29×/量化 1.08×/调度全输); kernel-cliff 真实。
- **削弱**: 耦合强度其实**弱且 model-dependent** —— Pyramid tile 耦合仅"中等有界"(折中 tile 吃掉大部分), CoDriving 6 旋钮**全可分离**; 且过参数化使 AP 轴近塌缩。
- **⇒ 结论: 原"耦合普遍存在→必须联合搜"命题若硬撑会被审稿人证伪。必须把故事升级为条件化版本(下 §2 推荐), 否则不自洽。**

---

## §1 资产盘点 (所有已确立结论, 正负都列, 按主题)

**A. 耦合证据(故事核心轴)**
- ✅强: 对齐×INT8 硬耦合(S0b: trap25 非÷32 → INT8 仅 1.08× vs p50 1.32×, 4090 真测)。
- ✅强: prune×quant 耦合(CoDriving: INT8 ΔAP50 随剪枝加深放大 base −0.003→p75 −0.062; 同时 INT8 加速随剪枝缩水 base 1.61×→p50 1.25×)。
- ⚠️中: tile×剪枝宽度耦合(Pyramid E-couple: 16×16×16 在 W48/64 最优 W128 最差, +21% rank-reversal; **但折中 tile 7.3% 吃掉大部分**)。
- ⚠️弱/负: L1/L2 弱, L3/L4 不可判; CoDriving 6 旋钮**全可分离**。
- ★发现: **co-design 耦合强度 model-dependent**(非标算子=grouped bottleneck 显耦合; 标准深归约 conv 可分离)。

**B. 加速证据(系统轴)**
- ✅ backbone 编译: Pyramid tuned/default 10.34×(grouped 默认差), CoDriving 2.07×(标准 conv 默认已好)。
- ✅ 剪枝编译级加速: Pyramid p50 2.07×, CoDriving ~5×(对齐档); 失配 trap25 0.29×(反而慢)。
- ✅ 量化加速(TRT): 对齐 1.32–1.60×, trap25 1.08×。
- ✅ Amdahl 真相: backbone 占全协同 pipeline 仅 13.6%(Pyramid)/RSU 50%车端 50%(CoDriving); **fusion neck 55%/16% 是真瓶颈且 TVM 导入 BLOCKED**。
- ✅ RSU 段: VFE 计算编译化 >200×(eager 18ms→85µs), scatter NCHW 向量化 2–4×(数值=0, 边缘卡避免布局转置), decode-fix 5–6×。
- ✅ 能耗: INT8 省 30–52% J/frame。Orin 异构 GPU∥DLA 真并行 1.34×。

**C. 精度证据(AP 轴, 当前最弱)**
- ⚠️负: Pyramid/DAIR 过参数化 → 剪枝 AP 无悬崖(AP50 0.79→0.76), INT8 近无损 → **AP 轴近塌缩, 无 trade-off**。
- ⚠️ CoDriving: iso-budget 后剪枝↑AP=训练协议 artifact(非剪枝功劳); 真 Pareto 极浅。
- 📋计划: V2X-ViT(欠参数化, INT8 会崩 AP30 57.4→40.0)把精度轴做实 —— **进行中/未完成**。

**D. 闭环/驾驶证据(end-task 轴, 未接入)**
- ✅ τ_perc 感知延迟→驾驶分有单调退化曲线; τ_ego(规划延迟)是 null。
- ❌ **加速后的延迟从未接进闭环看驾驶分** —— co-design 加速→驾驶收益的闭环未打通。

**E. 工具链/方法学(支撑, 非卖点)**
- TVM relax 导入+对齐+MS tune 全通; fusion neck 两模型都 BLOCKED(grid_sample); relax 无 INT8 pass(须 BYOC-TRT)。
- 确定性手写 schedule 消融方法(rule-drop 被预算混淆 → 弃)。

---

## §2 三个候选故事线 + 推荐

### 故事 A(原线)— "耦合驱动的 P×Q×D 联合搜索框架"
- 主张: 耦合陷阱使串行次优, 联合搜达更优 Pareto。
- ❌**问题**: CoDriving 可分离 + Pyramid 弱耦合 + 过参数化 AP 塌缩 ⇒ "耦合普遍且强"的前提**站不住**, 审稿人一个反例(CoDriving)就动摇全文。**当前证据不支持这个故事的强版本。**

### 故事 B(推荐)— "条件化 co-design: 何时该联合搜, 何时可分离"
- 主张: co-design 不是"总有用", 而是**价值条件化于模型算子结构 + 对齐约束**。贡献 = 给出**判据**(何时联合、何时分离)+ 一个 **coupling-aware 搜索器**(检测到耦合→联合, 否则分离省算力), 对齐(÷32)作跨轴硬约束。
- ✅**自洽**: 把所有负结果(CoDriving 可分离/Pyramid 弱耦合/AP 塌缩)从"缺陷"变成"判据的数据点"。强证据(对齐硬约束/model-dependent)正好是核心发现。
- ✅顶会卖点: "we characterize WHEN co-design pays off" 比 "yet another co-design framework" 更有洞察、更难被单反例推翻。

### 故事 C — "V2X 协同感知边缘部署: 瓶颈再定位 + RSU/车端分段加速"
- 主张: 真瓶颈是 fusion neck(车端)+ eager 前后处理, 不是大家都优化的 backbone; 按 RSU/车端分段。
- ⚠️**问题**: 加速手段偏工程(向量化/CUDA 化); fusion neck 还没解决(最大瓶颈 BLOCKED) ⇒ 系统论文完整性不足, 单独撑不起顶会。

### ★推荐 = **B 为骨, C 为肉, A 的强证据(对齐)作支柱**
统一标题候选: **"When Does Hardware–Software Co-Design Pay Off for V2X Collaborative Perception? A Coupling Characterization and a Conditional Search Framework"**
一句话贡献: 系统刻画 V2X 协同感知 P×Q×D 三轴耦合, 发现耦合**条件化于算子结构**且**对齐是跨轴硬约束**, 据此提出按需联合/分离的 coupling-aware 搜索, 并在多模型/多边缘平台真测 Pareto(理想上接闭环驾驶分)。

---

## §3 推荐故事的论文骨架 (章节 → 用哪些结论)

1. **Intro**: V2X 协同感知边缘部署的 P×Q×D 优化; 既有工作各自孤立优化或假设可分离; 我们问"何时该联合搜"。
2. **Motivation/Background**: 协同感知 pipeline 结构(RSU 感知→车端融合); 边缘约束; QuantV2X/V2X-ViT INT8 崩(崩溃-规避 motivation, §1-C 文献)。
3. **Coupling Characterization(核心①)**: 三轴耦合系统刻画 —— 对齐×INT8(S0b)、prune×quant(CoDriving ΔAP)、tile×宽度(Pyramid E-couple); **跨模型对比表** → model-dependent 判据。【资产 A】
4. **The Alignment Constraint(核心②)**: ÷32 失配在 P/Q/D 三轴全输(trap25)= 跨方法硬约束。【A + B 的 trap25】
5. **Bottleneck Relocation(核心③)**: Amdahl 真相, 真瓶颈=fusion/eager 非 backbone; RSU/车端分段。【B: Amdahl + RSU 段】
6. **Conditional Co-Design Search(框架)**: 判据驱动的联合/分离搜索 + 对齐硬约束 + 目标对准真瓶颈; searcher_v0。【框架 + Gap1 待补】
7. **Evaluation**: 多模型(Pyramid/CoDriving/+V2X-ViT 待)× 多平台(4090/Orin/H800)Pareto(AP-lat-energy-throughput); 联合搜 vs 串行搜增益(Gap1); 理想上闭环驾驶分(Gap5)。
8. **Conclusion**: co-design 价值条件化; 判据 + 框架。

---

## §4 自洽性检验 (核心 claim × 证据 / gap)

| # | 论文要 claim 的 | 现有证据 | 状态 | 缺口 |
|---|---|---|---|---|
| C1 | 三轴存在耦合 | S0b 对齐×INT8 / CoDriving prune×quant / Pyramid tile | ✅ 够 | — |
| C2 | 耦合 model-dependent(判据) | Pyramid 显耦合 vs CoDriving 可分离(2 模型) | ⚠️ 样本少 | **Gap2**: 需 3+ 模型验判据 |
| C3 | 对齐是跨 P/Q/D 硬约束 | trap25 三轴全输 | ✅ 强 | — |
| C4 | 真瓶颈是 fusion/eager 非 backbone | Amdahl(13.6%)+ RSU/车端分段 | ✅ 够(Pyramid/CoDriving) | fusion 未加速(划界 or 补) |
| C5 | **联合搜 > 串行搜(co-design 真 payoff)** | searcher_v0 存在, 但**无联合 vs 串行 Pareto 对比** | ❌ **缺** | **Gap1(最关键)** |
| C6 | 精度轴有真 trade-off | Pyramid/CoDriving 过参数化 AP 塌缩 | ❌ 当前模型不支持 | **Gap3**: V2X-ViT 做实 |
| C7 | 加速→驾驶收益 | τ_perc 曲线存在, 但未接加速 | ❌ 未打通 | **Gap5**: 接闭环 |
| C8 | 边缘真 e2e 加速 | 跨平台分段(H800 TVM/Orin torch), **拼不出单一数** | ⚠️ 部分 | **Gap4**: Orin 整段实测 |

**自洽性判断**: 故事 B 的骨架(C1/C3/C4)证据已足; 但 **C5(联合搜 payoff)、C6(精度轴)、C2(判据普适)是故事完整性的三根缺腿** —— 不补, 论文会停在"刻画了耦合但没证明 co-design 值得做"。

---

## §5 补实验清单 (按对故事完整性的必要性排序)

**MUST(不补则故事不成立)**
1. **[Gap1] 联合搜 vs 串行搜 Pareto 对比** —— 在显耦合模型(Pyramid)上, 真跑①联合搜 P×Q×D ②串行(先 P 后 Q 后 D), 比最终 (AP,lat,energy) Pareto。证联合搜拿到串行够不到的点(哪怕仅在对齐边界附近)。**这是 co-design "有用"的唯一直接证据, 当前完全缺。** 复用 searcher_v0。工作量中(框架已在)。
2. **[Gap3] V2X-ViT 把精度轴做实** —— 欠参数化 + INT8 崩(已知 AP30 57.4→40.0), 提供真 AP-latency trade-off + 验证判据(V2X-ViT 是 attention-heavy, 预测显耦合)。工作量中(原计划已授权)。

**SHOULD(显著加固)**
3. **[Gap2] 第三/四个模型验判据** —— F-Cooper(标准 conv, 预测可分离)/ UniV2X。把"算子结构→耦合强度"从 2 点变成趋势。可复用现有 E-couple/P4 脚本。工作量中。
4. **[Gap4] Orin 整段 RSU/车端 e2e** —— TRT 串 VFE+scatter+backbone, 拿单一边缘 e2e(消除跨平台拼接的诚实窟窿)。工作量大(VFE/scatter plugin)。

**NICE(顶会竞争力跃升)**
5. **[Gap5] 加速→闭环驾驶分** —— 把 co-design 加速后的延迟接进 CARLA 闭环, 沿 τ_perc 曲线看驾驶分提升。把"快了 X×"升级成"驾驶安全/通过率提升 Y" = V2X 论文最强落地说服力。基础设施已有(τ_perc 曲线 + 闭环仿真)。工作量中-大。
6. **fusion neck 编译** —— BYOC-TRT GridSample plugin 碰车端 55%/16% 瓶颈; 或诚实划 future work + 量化其影响上界。

---

## §6 风险 + 诚实呈现策略

- **负结果是资产不是负债**: CoDriving 可分离 / 过参数化 AP 塌缩 / TVM 消解 TRT 耦合 —— 在故事 B 下全是"判据的数据点"。**切勿硬凑成"耦合很强"**(会被一个反例打穿, 也违背纪律)。
- **不夸大 co-design 增益**: 即便补了 Gap1, 增益可能仅在对齐边界/特定模型显著。诚实表述"条件化收益", 比"普遍大增益"更可信、更难驳。
- **跨平台口径纪律**: H800(TVM)/4090(TRT)/Orin(边缘)数据**分平台报, 不拼单一 e2e**; 量化全是 TRT 参考(relax 无 INT8 pass)。
- **AP 轴诚实**: 当前模型 AP 近常数须明说"前沿沿精度轴塌缩", 靠 V2X-ViT(Gap3)才把精度轴做实。
- **最大单点风险 = Gap1(联合搜 payoff)**: 若真跑出来联合搜 ≈ 串行搜(因耦合弱), 则故事退为纯 characterization(measurement 论文), co-design 框架贡献削弱 —— 那时主打"判据 + 何时不必 co-design"仍可发表, 但定位要随结果诚实调整。**先跑 Gap1 探明, 再定最终 framing。**

---

## §7 与既有 method 稿的衔接 / 需修正处
- `stage2_method_zh_v1.md` 的"耦合陷阱使串行次优"动机段 **需从绝对命题改为条件化** —— 补 model-dependent 判据 + 对齐硬约束, 把 CoDriving 可分离作为"分离适用区"诚实写入。
- `background/00` 的"换 V2X-ViT 做实精度轴(进行中)"= 本文档 Gap3, 升为 MUST。
- 新增章节(§3 骨架 3/5): Coupling Characterization 跨模型表 + Bottleneck Relocation(RSU/车端 Amdahl)是本会话新增、原 method 稿没有的, 应正式纳入。

---

## §8 接手第一步
1. 与用户确认 framing: 故事 B(条件化 co-design)是否接受为主线。
2. **优先跑 Gap1**(联合搜 vs 串行搜 Pareto, Pyramid)—— 这是决定故事是"框架论文"还是"characterization 论文"的分水岭, 探明后再定最终 framing。
3. 并行启动 Gap3(V2X-ViT 精度轴)+ Gap2(第三模型验判据)。
4. Gap4/Gap5 视投稿 deadline 与人力排期。
