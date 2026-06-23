# Gap1 实验设计 — 联合搜 vs 串行搜 Pareto 对比 (v1, 2026-06-18)

> **定位**: 把 ALT/CHaNAS/AutoTVM 三篇联合搜索文献的启发, 全部落到 Gap1 这一个具体可执行实验上。
> **Gap1 = 论文 claim C5 的唯一直接证据**: "prune×quant×schedule **联合搜** 的 Pareto 前沿支配 **串行搜**(先定软件配置再单独 tune schedule = 现状 TRT-auto 黑盒)"。是决定论文是"co-design 框架"还是"characterization"的分水岭(见 [HANDOFF_codesign_unified_v1.md](../progress/HANDOFF_codesign_unified_v1.md) §13)。
> **文献依据**: [study_joint_search_methods_v1.md](../../references/study_joint_search_methods_v1.md)(ALT/CHaNAS/AutoTVM 研读)。
> **代码基底**: `framework/searcher_v0.py`(现状: 随机采样+传播+约束过滤, **无 schedule 轴/无评估器/无 Pareto**, D 仅 GPU/DLA 路由) → Gap1 = 把它升级到"带 schedule 轴 + 评估器 + Pareto"的 Stage 4。

---

## §0 一句话 + 设计总览

Gap1 不是"跑两个搜索比一下", 而是一个**三臂消融**(抄 CHaNAS-W/W/O + ALT-OL), 内外双环搜索(抄 ALT cross-exploration + AutoTVM 内环), 搜索空间按块分解(抄 CHaNAS R·B·S+R^B), 双度量(iso-AP latency + hypervolume), **sweet spot 钉在 INT8 对齐边界**(我方 trap25 + ALT/FAST padding), 并预设**诚实的决策规则**(联合≈串行 → 退 characterization)。

---

## §1 三臂消融 (★核心实验骨架, 抄 CHaNAS-W/W/O + ALT-OL/WP)

在**同一搜索预算 + 同一模型 + 同一平台**下跑三条线, 比最终 Pareto:

| 臂 | 名称 | schedule 轴 | 软件轴 (prune×quant) | 对应文献 | 我方现状 |
|---|---|---|---|---|---|
| **S0** | no-schedule baseline | 固定默认(dlight / TRT-auto) | 搜 | = CHaNAS-W/O(无调度再优化) | ≈ 现 searcher_v0 |
| **S1** | serial(串行) | 先定软件配置, **再**单独 tune schedule | 先贪心搜(在默认 schedule 上) | = "先 NAS 后编译"串行流程 | 当前 TRT-auto 实际做法 |
| **S2** | joint(联合) | 与软件轴**同时**搜 | 同时搜 | = CHaNAS-W / ALT joint | **本 Gap1 要新建** |

**期望**: S2 Pareto ⪰ S1 ⪰ S0, 且 **S2 相对 S1 的增益集中在 INT8 对齐边界**(见 §5)。
**关键**: S1 与 S2 的唯一变量 = "软件配置定了之后 schedule 才搜" vs "两者同搜" —— 这是干净的"是否联合"消融(对齐 CHaNAS Table 2 只变"是否做调度协同")。

---

## §2 搜索结构 — 外环 × 内环 (抄 ALT cross-exploration + AutoTVM 内环)

**同构关系(ALT 的核心洞察)**: ALT 中"layout 一变 → loop 搜索空间重建"; 我们"prune 选定通道宽度 → 合法 schedule(tiling/tactic)空间重建"。完全同构 ⇒ 直接套 ALT 的 cross-exploration 两阶段。

```
外环 (软件轴, 多目标):  prune_rate × quant_bits  —— NSGA-II / 约束化单目标
   │  每选定一个软件配置 → 通道宽度 W、精度 P 确定
   ▼
[joint stage]  重建该 (W,P) 的合法 schedule 搜索空间
   │
   ▼
内环 (schedule 轴):  TVM MetaSchedule(evolutionary_search + XGBoost cost model)
   │  搜 tiling / loop-order / fusion / tensorize
   ▼
   回灌: 把内环最优 (latency, energy) 作为该软件配置的"真实"硬件指标 (reward)
   │
   ▼
[schedule-only stage] (可选精修): 软件配置固定, 再多搜几轮 schedule
```

- **S2(joint)** = 外环每个候选都触发内环 schedule 重搜 → 回灌真实延迟。
- **S1(serial)** = 外环先用**默认 schedule 的延迟**贪心定软件配置, 选定后才对**那一个**配置跑内环 schedule 搜。
- **S0** = 外环全程用默认 schedule 延迟, 从不跑内环。
- 内环引擎 = TVM MetaSchedule(AutoTVM 后继): `search_strategy/evolutionary_search` + `cost_model/xgb_model` + `schedule_rule/multi_level_tiling`。**不重造轮子**(AutoTVM 启发)。

---

## §3 搜索空间分解 — 防指数爆炸 (★抄 CHaNAS R·B·S + R^B + block-LUT)

**问题**: prune×quant×schedule 全交叉积 = 指数爆炸(CHaNAS 原始 R^B·S^B)。**解**: 抄 CHaNAS 分解为 **R·B·S + R^B**。

具体到我们:
1. **"块" = backbone stage**(Pyramid 3 stage / CoDriving stage)或**双 agent 的 RSU·车端段**。
2. **块级预调度(一次性, 离线建表)**: 对每个块 × 每个 (宽度档 W, 精度档 P) 组合, 跑一次内环 MetaSchedule, 把**最优 schedule + 实测 (latency, energy)** 存进 **schedule-LUT**。组合数 = B × |W| × |P|(线性, 非指数)。
3. **外环搜索(在线)**: NSGA-II 组合各块的 (W,P) 选择, **延迟用加性组合** `Lat_net = Σ_block Lat_LUT[block, W, P]`(CHaNAS Eq 3), AP 用预测器/真测, **不必每个外环候选都重跑 MetaSchedule**。
4. 这让 S2(joint)可行: 内环搜 = 建 LUT 一次性成本; 外环 = 查表组合。S1/S0 共用同一 LUT 的"默认 schedule"列, 保证三臂可比。

---

## §4 度量 — 双度量都报 (抄 CHaNAS iso-AP latency + 升级多目标)

文献都是单目标(latency, 精度作约束); 我们多目标, 必须升级, 但保留 CHaNAS 的 legible headline:

1. **iso-AP latency(legible headline, 抄 CHaNAS Table 2)**: 在匹配 AP 的水平切一刀, 报 S2 vs S1 vs S0 的延迟倍率。CHaNAS 那句"等精度省 1.68×"就是这么来的, 审稿人一眼懂。
2. **多目标 Pareto 支配 / hypervolume(严谨)**: 在 (AP, latency, energy) 空间报三臂前沿的支配关系与 hypervolume 差。**这是 ALT/CHaNAS/AutoTVM 都没做、我们必须做的升级**(它们单目标)。
3. **对齐边界放大镜**: 单独报"serial 踩坑配置(非÷32 → INT8 fallback)"vs"joint 救坑(pad/repack → INT8 tensorize)"的点对点延迟差(见 §5)。

---

## §5 载体 + Sweet Spot — Gap1 最微妙处 (我方 trap25 + ALT/FAST padding + §13 讨论)

**ALT 的硬道理**: 联合的价值**只在两侧强耦合处显现**。耦合弱则 joint 退化成 serial(= 我方已踩的"剪枝率轴 Pareto 退化, 剪到底即最优")。⇒ Gap1 成败取决于把 sweet spot 选对。

### 5.1 联合赢串行的机理前提(§13.2)
联合能赢, 必须存在**"单维次优、组合最优"**的点: 即串行贪心会踩坑、联合会避坑/利用坑。
- **坑(对齐边界)**: 剪枝选了非÷32 宽度(如 48) → INT8 撞 tensor-core MMA 对齐墙 → fallback 慢(我方 trap25 实证: 剪枝 0.29×/量化 1.08× 双输)。
- **串行会踩吗?** 若 P 维单独搜本就不选非÷32(它本来就慢、AP 不更好), 串行天然避坑 → 联合无优势。**所以必须人为把 sweet spot 构造在"非÷32 宽度恰好 AP 有优势"的区域**, 逼出 trade-off。
- **联合的救法(抄 ALT layout 联合 + FAST tensor padding)**: joint 搜到"pad/repack 48→64 恢复对齐 + INT8 tensorize 救回 tensor-core", 用少量 FLOPs 换对齐 → 整体更快。**这就是"硬件维度被协同地配置"的真 demo**(回应"S0b 只证耦合存在、未证硬件被协同配置"的质疑)。

### 5.2 载体选择 — Gap1 ⟸ Gap3 (★§13.1 核心; 2026-06-19 勘误)
- ❌**早先两版假设都已修正**: ①"V2X-ViT 欠参数化 INT8 崩 57.4→40.0"(那是 V2XSet/OPV2V 文献值非 DAIR); ②(2026-06-18 一度写的)"DAIR AP 轴全塌缩、剪枝/INT8 近无损" —— **也被复测推翻**(用户质疑正确)。
- ★**2026-06-19 复测裁决(`results/v2xvit_ap_corrected.json` + `v2xvit_int8_ap_sim.json`)**:
  - **剪枝真崩 AP**(权重验证真加载 missing=0): V2X-ViT p50 原始剪枝 AP50 0.710→**0.025**(−68.6%), p75 AP70 0.521→**0.009**; **finetune 后恢复 ~0.73**。⇒ "剪枝↑AP/无损"是 post-finetune + iso-budget 假象; 真相 = **DAIR 过参数化使 finetune 能完全恢复**。
  - **INT8 真实代价未测定**: 3 个 simulated fake-quant 变体全近无损(±0.01), **但都失真**(FP32 累加 / 注意力 matmul 粗糙 proxy scale / 12 层激活未覆盖 / 静态 scale 反而更宽松)→ 不可信, 系统偏向无损。
- ✅**2026-06-19 定稿: 剪枝轴 AP 载体已成立(2 模型公平实测), 用户拍板用 Pyramid + CoDriving 两条曲线推进**:
  - **Pyramid 公平 AP-latency Pareto**(stage_a, base=官方收敛 ckpt): AP70 平滑单调 0.631→0.590→0.564→0.530(span 0.10); 配 TRT fp16 延迟 1.27/2.87/1.01/0.78ms。★**p25(48 破 ÷32)被 base 严格支配**(AP↓+延迟↑ 2.26×)= 对齐耦合实证 + 串行踩坑案例。**这就是 Gap1 要的"软件旋钮真移动 AP、且对齐制造被支配点"的载体。**
  - **CoDriving iso-budget**: base_isobudget(公平 base)AP50 0.626 > 所有剪枝档 → 剪枝轴真伤 AP(浅但方向明确)。
- ⇒ **Gap1 主轴 = 剪枝×schedule(×对齐)**, 在 Pyramid + CoDriving 上做。AP 轴用 **AP70**(信号最强); 对齐 trap(p25 类非÷32)是联合搜该避、串行搜会踩的关键耦合点(= §5.1 sweet spot 的实锤)。
- **INT8 轴(次要/待补)**: 真实代价仍未测定(simulated 不可信), 需真 TRT INT8 / TensorRT-ModelOpt; 作为第二软件轴可后补。
- **V2X-ViT(暂缓)**: 需 iso-budget 重训 base 才能得公平曲线(同 CoDriving 做法), GPU 空闲后补, 非当前主线。
- **8h/近期可做**(GPU 约束: 干净 latency 须 idle 卡): ① 用现有 Pyramid 公平 Pareto + 对齐 trap 直接搭三臂消融的"串行踩 p25 vs 联合避坑"demo; ② schedule-LUT 建表(内环 MetaSchedule)等 idle GPU。

---

## §6 Cost Model — 修我们崩掉的预测器 (★抄 AutoTVM rank loss)

- **外环 latency 预测器**: 当前 LGB latency "跨数量级回归崩、给负值"(CLAUDE.md §四 + memory [[project_f_lat_ceiling]])。**AutoTVM 实证 rank loss > regression loss**(选择阶段只需相对快慢)。⇒ **改 pairwise/rank loss** —— 低成本高确定性的修法。
- **内环 schedule cost model**: 直接复用 TVM MetaSchedule 的 `cost_model/xgb_model`(XGBoost + rank), 不重造。
- **跨平台迁移(4090↔Orin)**: 若要把 cost model 迁移, 学 AutoTVM 的 invariant 表示 + global+local 分解(我方已有 M2 f 函数跨平台 R²>0.995 基础)。

---

## §7 决策规则 — 诚实预设 (★§14 最大单点风险)

**跑 Gap1 前先把判读规则定死, 防 p-hacking**:
- **若 S2 显著支配 S1**(hypervolume 增益 > 噪声, 且 iso-AP latency 倍率明显, 尤其在对齐边界): → 故事 A/B 的强版本成立, co-design 框架有直接 payoff。
- **若 S2 ≈ S1**(联合≈串行, 因耦合弱): → **这本身就是故事 B 的核心数据点**(co-design 价值条件化, 此模型/此区间属"可分离适用区")。论文诚实退为 characterization + 判据, **仍可发表**, 但 framing 随结果调整。**切勿硬凑成"联合大赢"**(违背纪律, 一个反例就被打穿)。
- **预判**(诚实): 从已有证据(Pyramid 弱耦合/CoDriving 可分离/TVM 消解耦合), S2 相对 S1 的增益**很可能只在对齐边界/V2X-ViT 显著**, 平坦区≈串行。这正是"条件化"的预期形态。

---

## §8 落地步骤 + 复用资产 + 平台口径

### 8.1 步骤(建议顺序)
1. **(前置)推进 Gap3/V2X-ViT 就绪** —— 否则 Gap1 无真 AP 轴(§5.2)。或先 Pyramid latency-only 机制验证。
2. **给 searcher_v0 加 schedule 轴 + 评估器** —— 现 searcher_v0 只有 prune/quant/D-routing, 无 schedule 维、无 metric 评估。Gap1 = 实现其 docstring 说的 "Stage 4 NSGA-II/BoTorch"。
3. **建 schedule-LUT(§3)** —— 对每个 (块, W, P) 跑内环 MetaSchedule 存最优 schedule+实测延迟。复用 H800 已打通的 `{base,p50,trap25}_backbone.onnx` + `s2_2*` 脚本。
4. **构造对齐边界 sweet spot(§5.1)** —— 含非÷32 宽度档 + 其 pad/repack 变体, 逼出 trade-off。
5. **跑三臂 S0/S1/S2(§1)**, 同预算同 seed。
6. **算双度量(§4)** + 对齐边界放大镜。
7. **按 §7 决策规则判读**, 定最终 framing。

### 8.2 复用资产
- `framework/searcher_v0.py`(外环骨架 + 对齐网格 `ALIGNED_PRUNE_RATES`/`NEAR_ALIGNED_PRUNE_RATES` 已建 —— 正好喂 §5 sweet spot)。
- H800 TVM env + `{base,p50,trap25}_backbone.onnx` + MetaSchedule 工具链(已打通)。
- `s2_2{d,e,f,g}` 脚本(tile/knob 消融, 已是内环雏形)。
- E-couple 结论(哪些旋钮耦合 = 内环该重点搜哪些)。

### 8.3 平台口径纪律(铁律)
- **schedule 搜索/延迟在 H800 TVM 坐实相对/机理**; 边缘绝对延迟须 4090/Orin 真测。
- **INT8 须 BYOC-TRT**(relax 无 INT8 pass) —— 量化臂的延迟是 TRT 口径, 不与 H800 TVM 数混加。
- **不拼跨平台单一 e2e**; 三臂必须在**同一平台同一口径**内比(否则消融不干净)。
- 空闲 GPU 实测(mem≤50MiB), 微秒 kernel min-of-N 兜竞争尖峰。

---

## §9 每篇论文 → Gap1 设计决策映射表 (★一览)

| 文献 | 启发 | 落到 Gap1 的具体决策 |
|---|---|---|
| **CHaNAS** | W vs W/O 同基底"有/无调度协同"消融, iso-acc 1.68× | §1 三臂中 S0(=W/O)↔S2(=W); §4 iso-AP latency 作 headline |
| **CHaNAS** | Fig 2 最优架构随调度漂移 = 耦合 motivation | related-work 引为先例; §1 三臂的立论依据 |
| **CHaNAS** | 分解 R·B·S + R^B + block-LUT + Lat_net=ΣLat_block | §3 按块预调度建 schedule-LUT, 外环加性组合, 防指数爆炸 |
| **CHaNAS** | mutation 限可整除 split("不可整除更差") | §5 ÷32 对齐 sweet spot 的他证; 内环 schedule 空间剪枝规则 |
| **CHaNAS** | constrained-single-objective(精度目标+延迟约束) | §4 iso-AP latency 切片(配多目标 hypervolume) |
| **ALT** | layout↔loop 墙; cross-exploration 双阶段 | §2 外环(软件)×内环(schedule)双环; "宽度变→schedule 空间重建" |
| **ALT** | ALT-OL/WP 消融范式(联合有独立收益 1.3×) | §1 S1(serial)vs S2(joint)的消融形式 |
| **ALT** | layout 必须进搜索空间(只调 loop 不够) | §5.1 救坑动作 = layout/pad 联合, 不只调 tiling |
| **ALT/FAST** | tensor padding 当显式编译决策(pad 换对齐) | §5.1 joint 救坑 = pad/repack 48→64 + INT8 tensorize |
| **AutoTVM** | rank loss > regression loss | §6 改 LGB latency 预测器为 pairwise rank loss |
| **AutoTVM** | XGBoost cost model + 进化/退火 + 在线更新 | §2/§6 内环复用 TVM MetaSchedule xgb_model, 不重造 |
| **AutoTVM** | 迁移学习 invariant 表示(global+local) | §6 cost model 跨 4090↔Orin 迁移的方法 |
| **三篇共识** | 联合价值只在强耦合处显现 | §5 sweet spot 钉对齐边界; §7 决策规则(弱耦合→退 characterization) |

---

## §10.5 ★首轮结果 (2026-06-19, H800 全空 schedule 轴三臂)

第一个 **联合>串行** 机制实证(Pyramid backbone, H800 TVM, `results/gap1_schedule_lut.json`):

**schedule-LUT(default dlight vs MetaSchedule-tuned, 干净空闲卡)**:
| width | tuned µs | AP70 | 对齐 |
|---|---|---|---|
| base 64/128/256 | 6214 | 0.631 | ✓ |
| trap25 48/96/192 | **21615** | 0.590 | ✗ |
| p50 32/64/128 | 3011 | 0.564 | ✓ |

**三臂**: S1 串行(剪到 25%→trap25→tune)= 21615µs; **S2a 联合(对齐感知改选 p50)= 3011µs = 7.18× faster, AP70 仅 −0.026**。

**★根因精炼(比"÷32"更准)**: grouped conv(g=32)需 **in_per_g = 2·num_filters/32 为 2 的幂**才命中快核; num_filters=48 → in_per_g={3,6,12}(全三阶段非 2 的幂)→ kernel-cliff, MetaSchedule 救不回。base={4,8,16}/p50={2,4,8} 全 2 的幂。

**★pad 救援 = 诚实负结果(强化论点)**: 局部 pad stage0(48→64, in_per_g 3→4)无效(47362µs, 反更慢), 因 stage1/2 仍 6/12。⇒ **对齐是全网系统属性, 不可局部修补 → 必须整网联合搜(剪枝选宽度时就考虑 in_per_g∈2^k)**。这正是联合 vs 串行的核心价值。

**诚实边界**: ① 7.18× 是**对齐陷阱罚分**(串行盲目剪到 25% 落在被支配的 48ch), 非泛化"联合搜处处更快"; trap25 实际被 base 双轴支配(AP↑+延迟↓)。② backbone-only, Amdahl 稀释到 e2e(backbone ~13.6%)。③ AP 来自 4090/DAIR(平台无关), 延迟 H800(relative 坐实)。④ 仅剪枝×schedule 轴, 量化轴未并入。

## §10 风险 + 开放问题
1. **载体风险(最大)**: Gap1 须 Gap3/V2X-ViT 就绪。V2X-ViT 迁移/训练进度未知 —— 这是排序的硬前提(§13.4 待用户确认)。
2. **sweet spot 构造风险**: 若构造不出"非÷32 宽度 AP 有优势"的点, 串行天然避坑, 联合无可赢之处 → 须先小规模探针确认该点存在。
3. **内环成本**: 建 schedule-LUT 的 MetaSchedule 搜索耗时(每块×档一次)。靠分解(§3)控制在线性, 但 H800 GPU 抢占要排期。
4. **INT8 口径**: BYOC-TRT 的量化臂能否与 TVM schedule 臂在同一框架内比 —— 可能须把 schedule 臂也走 TRT(或量化臂只在 4090 TRT 口径单独报), 是工程开放问题。
5. **多目标 vs 约束单目标**: NSGA-II(多目标)还是 CHaNAS 式约束单目标? 建议**主用约束单目标产 iso-AP latency**(干净), **辅用 NSGA-II 产 hypervolume**(严谨)。
