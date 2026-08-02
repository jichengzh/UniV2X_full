# Exploring（空间探索）+ ★TVM 集成架构设计 v1

> ★**2026-06-19 两点更正(用户强调, 全文据此理解)**:
> 1. **W_g 定义**: W_g = **单维(贪心所搜那一维)最优、但多维(全局)非最优**的点 = 单侧贪心搜索落入并卡住的**局部最优**(贪心选它因它单维最优; 非全局因到达真全局需接受贪心已丢弃的单维次优选择)。另设 P_g = 被错过的全局点(单维次优/多维最优)。
> 2. **平台: 已迁移 TVM, 不以 TRT/INT8 为重心**。架构/实验主轴 = **TVM prune × schedule**(MetaSchedule); 量化(INT8)因 relax 无 INT8 pass 降为 **future/次要**, 不作主依赖。本文涉及 TRT/INT8 处相应降级为 future-note。

> 3. ★**耦合口径统一(2026-06-22, 见 `1_design_space_building_v1.md §0.1`)**: 本文的外环(软件 P×Q, NSGA)↔内环(硬件 schedule, MetaSchedule)正是协同的两环。协同主张 = **内外环强耦合(经 P-hub)**, 强度按架构用**三臂消融测量**(非"if groups==1"规则); 可分离架构(如 CoDriving)是测量出 ≈1.0 的结果, 框架据耦合分数自适应分配内环预算(分流=连续预算非二元开关)。

> **定位**: 把"导入一个网络 → 输出最优模型"做成**一条流水线**, 而不是现在 searcher_v0 那种"随机采样 + 约束过滤"或人工不断枚举试错。本设计严格基于三篇文献(ALT / CHaNAS / AutoTVM, 研读笔记 `references/study_joint_search_methods_v1.md`)与我们已打通的 TVM 基底(H800 TVM 0.20 改版 Unity, MetaSchedule 在 `tvm.s_tir.meta_schedule`)。
>
> **标注约定**: `[借鉴X]` = 直接来自某篇论文的方法; `[我们的设计]` = 本项目自建/改造; `[待定]` = 工程开放问题(结尾汇总)。
>
> **承接关系**: 本文是 `stage2_method_zh_v1.md`(阶段二总稿)与 `gap1_joint_vs_serial_design_v1.md`(联合 vs 串行消融)的**架构落地层** —— 前者定"搜什么/评什么/Pareto 怎么定", Gap1 定"怎么做消融证联合有价值", 本文定"探索算法 + TVM 集成 + 流水线组件"的具体形态。代码基底 = `framework/{searcher_v0,nsga2_pareto_search_v4,accuracy_predictor,latency_estimator}.py`。

---

## §1 三篇的 Exploring 怎么做 + 我们选哪种

### 1.1 三篇探索机制对比

| 维度 | **ALT** (EuroSys'23) | **CHaNAS** (LCTES'21) | **AutoTVM** (NeurIPS'18) |
|---|---|---|---|
| 探索的是什么 | graph layout × operator loop | NN 架构 × 编译调度(block 粒度) | 纯 schedule/loop(单算子) |
| 探索算法 | **PPO 强化学习**: Layout Agent(actor)出 layout 原语序列, Loop Agents 对每个 layout 重建 loop 空间多轮 random-walk, 共享 critic; **cross-exploration 两阶段**(joint stage + loop-only stage) | **进化搜索**: 先按部署延迟 CDF **自动划子空间**(In_size × Md_E → 20 子空间), 选平均延迟最低子空间, 再在该子空间 genome 编码 + mutation/crossover; 配 **块级预调度**(GP 代价模型)建 LUT | **并行模拟退火** + **ε-greedy** + **diversity-aware** 选 batch; 每轮取预测 top-k 真测回灌 cost model |
| 代价模型 | XGBoost(在线训练) | NAS 侧: MLP 精度预测器 + block-latency-LUT; 调度侧: GP | **XGBoost / TreeGRU + rank loss**(只学相对快慢) |
| 目标数 | 单目标 latency | 约束化单目标(精度目标 + 延迟约束) | 单目标 latency |
| 对我们的角色 | **内外双环的"粘合"骨架**(上游变 → 下游空间重建) | **分解防爆炸 + 外环算法 + 子空间划分** | **schedule 内环引擎 + cost-model 修法** |

### 1.2 我们的选择 — 内外双环 [我们的设计, 三篇缝合]

三篇都是**单目标**(latency 或精度+延迟约束), 而我们是**多目标 7 指标 Pareto**(AP, latency, throughput, energy, size + DS, RC; 见 `stage2_method_zh_v1.md` §优化目标), 不能照抄任何一篇的标量探索。我们的方案是**嵌套双环**:

```
外环(软件轴 P×Q, 多目标 Pareto):  prune_rate × quant_bits × (per-单元精度)
   算法 = NSGA-II 进化     [借鉴 CHaNAS 进化 + 我们的多目标升级]
   │  每选定一个软件配置 c → 通道宽度 W、精度 P 确定
   ▼  ★ ALT cross-exploration 同构: "上游(P/Q)变 → 下游(schedule)空间重建"
内环(硬件 schedule 轴, 单目标 latency/energy):  tiling / loop-order / fusion / layout / tensorize
   算法 = TVM MetaSchedule evolutionary_search + XGBoost(rank)   [借鉴 AutoTVM 后继, 直接复用不重造]
   │  搜该 (W,P) 下的最优实现
   ▼
   回灌: 内环最优 (latency, energy) 作为该软件配置 c 的"真实"硬件代价 (reward)
```

- **为什么外环用 NSGA-II 而非 ALT 的 PPO** [我们的设计]: PPO 需大量 on-policy 轨迹(rollout)训 actor-critic, 在我们极小样本(完整点 ~33–58)上不收敛; 多目标 RL 还要标量化奖励(违背"AP 恒驻目标轴、不许加权综合分"的纪律, `stage2` §iii)。NSGA-II 是 population-based 进化, 天然多目标、无需可微奖励、与 CHaNAS 的进化外环同源 —— 且 `framework/nsga2_pareto_search_v4.py` 已落地骨架(pymoo NSGA2)。
- **为什么内环用 MetaSchedule 而非自写退火** [借鉴 AutoTVM]: AutoTVM 的并行模拟退火 + XGBoost cost model 的后继就是 MetaSchedule(evolutionary_search + xgb_model), TVM 已实现且工程化。"不重造轮子"是 AutoTVM 给我们的直接教训。内环单目标(latency)正是它的强项。
- **ALT 的两阶段** [借鉴 ALT]: joint stage = 外环每个候选都触发内环 schedule 重搜(对应 Gap1 的 S2-joint); 可选的 schedule-only stage = 软件配置固定后再精修 schedule(对应 ALT 的 loop-only stage)。
- **CHaNAS 子空间划分** [借鉴 CHaNAS, 可选]: 若软件空间大, 可先按"最重要因子"(我们 = 剪枝率档 × 精度档)粗划子空间、用延迟代理排序, 在最优子空间内细搜 —— 降外环采样成本。当前我们空间已压到 ~10^4(`stage2` §搜索空间), 可暂不分; 留作扩展。

### 1.3 防指数爆炸 — 按块分解 + schedule-LUT [借鉴 CHaNAS R·B·S + R^B]

外环每个候选都跑一次内环 MetaSchedule(分钟级)不可行。抄 CHaNAS 的**块级预调度**:

1. **"块"** = backbone stage(Pyramid 3 stage / CoDriving stage)或双 agent 的 RSU·车端段。
2. **离线建表(一次性)**: 对每个 `(块, 宽度档 W, 精度档 P)` 组合跑一次内环 MetaSchedule, 把**最优 schedule + 实测 (latency, energy)** 存进 **schedule-LUT**。组合数 = `B × |W| × |P|`(**线性**, 非指数)。
3. **在线外环**: NSGA-II 组合各块的 (W,P), 延迟用**加性组合** `Lat_net = Σ_block LUT[block, W, P]`(CHaNAS Eq.3), 不必每候选重跑 MetaSchedule。
4. S0/S1/S2 三臂(Gap1)共用同一 LUT 的不同列(default schedule vs tuned schedule), 保证消融可比。

> 已有首轮实证(`results/gap1_schedule_lut.json`, Gap1 §10.5): Pyramid backbone 三宽度的 default-vs-tuned LUT 已建, 联合(对齐感知改选 p50)相对串行(盲剪到 trap25=48ch)快 7.18×。这就是 schedule-LUT 的最小可用形态。

---

## §2 ★核心: 能否基于 TVM 现有框架搭建? — P×Q 整合进 schedule 搜索的 HOW

### 2.1 关键认识: P/Q 与 schedule 在 TVM 里位于不同层 [我们的设计]

AutoTVM/MetaSchedule **已经把 schedule(tile/loop/fusion/layout/tensorize)做成可搜的硬件旋钮**了 —— 这一侧我们**直接复用, 不重造**。我们要做的是把**剪枝 P、量化 Q** 整合进去。但三者在 TVM 抽象栈里位置不同:

```
层级                         TVM 对象 / 入口                   谁来改
───────────────────────────────────────────────────────────────────
① 模型结构(剪枝 P 改通道宽度)   ONNX / relax IRModule(算子拓扑)   ★我们(重建子图)
② 数值精度(量化 Q 改 dtype)    relax dtype + Q/DQ 节点          ★我们(BYOC-TRT or relax pass)
─── 以上 = 外环改"算什么" ──────────────────────────────────────────
③ 实现/调度(tile/layout/...)   PrimFunc + s_tir.Schedule       TVM MetaSchedule(复用)
─── 以上 = 内环改"怎么算" ──────────────────────────────────────────
```

**P 是改 ①(IRModule 算子的通道数), Q 是改 ②(dtype + tensorize 选择), schedule 是改 ③(PrimFunc 的循环结构)**。三者不在同一 TVM API 层, 不能简单"加一个搜索维度"塞进 MetaSchedule 的 space_generator —— 必须用**外环改上游 IR、内环调下游 schedule** 的嵌套。这正是 ALT cross-exploration "上游变 → 下游空间重建"的结构, 我们用它粘合 P×Q(上游)与 schedule(下游)。

### 2.2 具体集成点 — 挂在 TVM 哪一层 [我们的设计 + 借鉴 ALT/FAST]

| 旋钮 | TVM 集成点 | 怎么挂 | 文献依据 |
|---|---|---|---|
| **P 剪枝率** | **外环, 在 ONNX 导入之前** | 用 DepGraph 在 PyTorch 端 rebuild 出窄通道子图 → 重新 export ONNX → `from_onnx` 进 relax。**每个剪枝宽度 = 一个新 IRModule**。(我们已有 `{base,p50,trap25}_backbone.onnx` 即此产物) | [我们的设计] 剪枝改拓扑, 必在 IR 重建层 |
| **Q 量化(INT8)** | **relax 层 / BYOC-TRT** | relax **无 INT8 量化 pass**(已核验, 跨模型 handoff §1.5) → INT8 走 **BYOC-TRT**(`relax.transform` 把量化子图交 TRT 编译); fp16 走 `relax.transform.ToMixedPrecision`。tensorize 选择(`MMA_i8i8i32_INTRIN`)在内环表达 | [借鉴 AutoTVM] 精度即 dtype; [我们的设计] INT8 路径补 BYOC |
| **Q 单元精度选择** | **外环, 子图切分** | 哪些块走 INT8 vs FP16 = 外环离散变量; 决定该块送 BYOC-TRT(INT8) 还是 relax-MetaSchedule(fp16) | [借鉴 CHaNAS] 块级精度 |
| **schedule(tile/loop/fusion)** | **内环, `tune_relax` / space_generator** | `ri.tune_relax(mod, params, target, work_dir, max_trials_global)` → `MetaScheduleApplyDatabase` → `tvm.compile`。space_generator 默认 `post_order_apply` + `multi_level_tiling` schedule_rule, 不改 | [借鉴 AutoTVM 后继] 直接复用 |
| **layout / 对齐救援(pad)** | **内/外环之间, relax pass** | `ConvertLayout` + `pad_einsum` 把非÷32 宽度 pad 回对齐 → 救回 tensor-core(ALT layout 联合 + FAST tensor padding) | [借鉴 ALT + FAST] |

**集成流(单个外环候选 c 的处理)**:

```
c=⟨P,Q,D⟩
  │  ① P: DepGraph rebuild → 窄通道 ONNX                    [我们]
  ▼
relax IRModule(窄网络, fp32)
  │  ② Q: per-块切分 → {fp16 块经 ToMixedPrecision; INT8 块标记 BYOC-TRT}   [我们]
  ▼
混精 IRModule
  │  ③ Legalize: Sequential([LegalizeOps, AnnotateTIROpPattern, FuseOps, FuseTIR])  ← relax-op 降成 TIR PrimFunc(否则 "No tasks to tune", 已踩坑)
  ▼
  │  ④ 查 schedule-LUT[块, W, P] — 命中则直接取 tuned schedule + (lat,energy)   [借鉴 CHaNAS LUT]
  │     未命中(新组合)则触发内环:
  ▼
内环 MetaSchedule: ri.tune_relax(evolutionary_search + xgb_model)   [复用 TVM]
  │  (INT8 块: BYOC-TRT build + trtexec bench)                       [我们补]
  ▼
  ⑤ 回灌: (lat,energy) 写回 LUT; 加性组合 Lat_net = Σ_block            [借鉴 CHaNAS Eq.3]
```

**关键工程坑(已在 H800 踩过, 跨模型 handoff §1)**: target 必用 `Target.from_device(tvm.cuda(0))`(裸 dict 缺 max_threads_per_block 崩); tune 前必 legalize; 手写 schedule 消融前必打 `func.with_attr("tirx.is_scheduled", True)`; shared 暂存必配 cooperative fetch 否则 24× 假慢。

### 2.3 "先搭起来、cost model 后补"是否可行? — 可行, 给出 MVP [我们的设计]

**结论: 可行, 且这是正确的工程顺序。** 理由: schedule 内环的 cost model 不用我们训 —— TVM MetaSchedule 自带 XGBoost(`cost_model/xgb_model`), 它在每次 `tune_relax` 内部边搜边在线训练(每轮取 top-k 真测回灌, AutoTVM 范式)。我们**唯一要补的 cost model 是外环的 AP/latency 预测器**, 而这恰好可以"先用真测/粗代理跑通、再迭代":

**最小可行 pipeline (MVP) 组件清单**:

| 组件 | MVP 用什么 | 复用 / 自建 | 迭代方向 |
|---|---|---|---|
| 输入 | ONNX / ckpt(已有 `{base,p50,trap25}_backbone.onnx`) | 复用现有导出脚本 | 完整 e2e 导出(neck 待解) |
| 空间构建 | `searcher_v0` 的合法采样 + 约束/传播 | **复用** `constraints.py`+`propagation.py` | 加 schedule 轴的合法取值 |
| 外环探索 | NSGA-II(`nsga2_pareto_search_v4.py`) | **复用骨架**(已有 pymoo NSGA2) | 把 `d_runtime` 粗轴升级为真 schedule-LUT 索引 |
| 内环 schedule | TVM `ri.tune_relax`(自带 xgb cost model) | **复用 TVM 现成** | — (已工程化) |
| schedule-LUT | 离线对 (块,W,P) 建表, 存 json | **自建**(已有 `gap1_schedule_lut.json` 雏形) | 扩档位 + 多平台列 |
| 外环 latency 预测 | **MVP: 直接用 LUT 加性组合的真测**(免训) | 复用 LUT | 训 LGB(rank loss)插值未测组合 |
| 外环 AP 预测 | **MVP: 真测 / `lgb_v7_ap` 粗代理** | 复用 `accuracy_predictor.py` | active-learning 补点重训 |
| energy / throughput | MVP: energy 真测(E4 已起), tput=1/lat 派生 | 复用 | batch 轴真测后单独建模 |
| 输出 | Pareto 前沿 + Top-K 编译 engine | 自建汇总 | DeployPlan 导出 |

**MVP 主轴 = TVM prune × schedule(外环 NSGA-II prune × 内环 MetaSchedule), 策略 = "内环 cost model 用 TVM 现成 + 外环先真测/粗代理 + LUT 缓存避免重复内环"**。先把流水线跑通(导入 → 外环 NSGA-II 查 LUT → 出 Pareto), cost model 的精度迭代(外环 LGB 改 rank loss、active-learning 补点)是**后续增量**, 不阻塞 pipeline 成型。这正是 AutoTVM "边搜边训 cost model" 的思路放到外环。

> ★**量化轴(INT8 / Q)为 future, 不进 MVP 主流水线**(见首部更正2)。MVP 与主搜索轴只走 **prune(P)× schedule** 两轴, 全程在 TVM-MetaSchedule 同一编译栈内; 凡涉及 INT8 必走 BYOC-TRT、TRT 口径分裂等集成点(本文 §2.1②/§2.2 Q 两行/§3 INT8 块分支/§5 风险1·3)均为 **future(量化轴日后)**, 不作 MVP 依赖。

---

## §3 "导入网络 → 输出最优模型" 端到端流水线 [我们的设计, 缝合三篇]

```
┌─ 输入 ───────────────────────────────────────────────────────────┐
│  ONNX / PyTorch ckpt  +  硬件能力 H(yaml)  +  阶段一 partition manifest │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌─ ① 空间构建 ────────────────────────────────────────────────────┐   [复用 stage1 + searcher_v0]
│  理论 B1×B2×D ~10^15 → H 注入合法取值 + 轴塌缩(对象=channel)        │
│  + C1–C11 耦合约束(非笛卡尔) → 合法空间 ~10^4                      │
│  + ★新增 schedule 轴的合法取值(每个剪枝宽度对应一组合法 tile/layout) │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌─ ② 探索: 外环 P×Q × 内环 schedule (双环) ──────────────────────┐
│                                                                  │
│  外环 NSGA-II [借鉴CHaNAS进化+多目标升级]                         │
│   候选 c=⟨P,Q,D⟩ → 约束过滤 + D↔B2 传播(constraints/propagation) │   [复用]
│        │                                                         │
│        │  ★ ALT cross-exploration: P/Q 定 → 通道宽度/精度定        │
│        ▼                                                         │
│   查 schedule-LUT[块,W,P] ── 命中→取 ── 未命中↓                   │   [借鉴CHaNAS LUT]
│        │                                                         │
│        ▼  内环 TVM MetaSchedule [复用TVM, AutoTVM后继]            │
│   重建该(W,P)合法 schedule 空间 → tune_relax(evo+xgb) →           │
│   最优 schedule + 实测(lat,energy) → 写回 LUT                     │
│   (INT8 块: BYOC-TRT build+bench)                  [我们补]        │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌─ ③ cost model 引导 ───────────────────────────────────────────┐
│  外环评估器 f: c → 7 指标 + 不确定性                              │   [复用 accuracy_predictor + 改造 latency]
│   - latency: LUT 加性组合 Lat_net=ΣLat_block (MVP免训)            │   [借鉴CHaNAS Eq.3]
│              → 后期 LGB rank-loss 插值                            │   [借鉴AutoTVM rank loss]
│   - AP: lgb ΔAP 预测(残差) / 真测                                │   [复用 lgb_v7_ap]
│   - energy/tput/size: 真测/派生/解析                             │
│   - DS/RC: 两级 g(AP^,ℓ^)→驾驶分 (闭环标定响应面)                 │   [我们的设计]
│  active-learning: quantile宽度×前沿邻近 选K=8–16 真测回灌重训(虚线)│   [借鉴AutoTVM在线更新]
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌─ ④ 输出 ─────────────────────────────────────────────────────┐
│  regime条件 多目标 Pareto 前沿(O_ρ:AP常驻 / S_ρ约束)             │   [我们的设计]
│  → Top-K 配置 → 阶段三真测验证 → 编译好的 engine(TVM .so / TRT)  │
└──────────────────────────────────────────────────────────────────┘
```

**哪些复用 TVM 现成 / 哪些自建**:

| 模块 | TVM 现成 | 我们自建 |
|---|---|---|
| schedule 搜索引擎 | ✅ `meta_schedule.relax_integration.tune_relax`(evolutionary_search) | — |
| schedule cost model | ✅ `cost_model/xgb_model`(XGBoost + rank, 在线训) | — |
| schedule 空间生成 | ✅ `space_generator/post_order_apply` + `multi_level_tiling` | — |
| 手写 schedule 原语(对齐救援) | ✅ `s_tir.Schedule`(split/tensorize/transform_layout/pad_einsum) | 调用编排 |
| INT8 tensorize | ✅ `tensor_intrin.cuda.MMA_i8i8i32_INTRIN` | BYOC-TRT 接线 |
| 剪枝 P(改拓扑) | ❌ | DepGraph rebuild → ONNX(已有) |
| 外环多目标探索 | ❌ | NSGA-II(pymoo, 已有骨架) |
| 外环 7 指标评估器 | ❌ | LGB(AP ΔAP / latency rank / 两级 DS) |
| schedule-LUT 缓存 + 加性组合 | ❌ | json LUT(已有雏形) |
| 约束/传播 | ❌ | `constraints.py` / `propagation.py`(已有) |
| regime 条件 Pareto / active-learning | ❌ | 选择器 + acquisition(设计待实现) |

---

## §4 与 searcher_v0 的衔接 — 升级成什么 [我们的设计]

`searcher_v0.py` 现状: **随机采样 + 双向传播 + 硬约束过滤**, 无 schedule 轴、无评估器、无 Pareto(其 docstring 自陈"真正的多目标搜索器在 Stage 4")。同目录已有 `nsga2_pareto_search_v4.py`(pymoo NSGA-II, 但 D 维只是 `d_runtime ∈ {pytorch, fakequant, trt_fp16, trt_int8}` 的粗 4 档枚举, **不是真 schedule 搜索**)。升级路径 = 把 v0 的合法采样作为 NSGA-II 的初始种群生成器, 并补三件事:

| 缺口 | 现状 | Stage 4 升级 | 复用 |
|---|---|---|---|
| **schedule 轴** | 无(D 仅 GPU/DLA 路由 + d_runtime 粗 4 档) | 加 schedule-LUT 索引轴: 每个 (块,W,P) 的最优 tuned schedule, 由内环 MetaSchedule 离线产出 | searcher_v0 的 `ALIGNED_PRUNE_RATES`/`NEAR_ALIGNED_PRUNE_RATES` 正好喂对齐 sweet-spot |
| **评估器** | 无(只过滤合法性) | 接 `accuracy_predictor`(ΔAP) + 改造 `latency_estimator`(LUT 加性组合 / LGB rank) + energy/DS | 复用现有两个预测器接口 |
| **Pareto** | 无 | regime 条件多目标非支配排序(O_ρ/S_ρ 切分) + hypervolume | 复用 `nsga2_pareto_search_v4` 的 pymoo NSGA2 |

**衔接职责边界**: `searcher_v0` 保留为"合法空间采样器/初始种群发生器"(它的约束+传播是 NSGA-II 采样阶段的合法性保证); `nsga2_pareto_search_v4` 升级为带 schedule-LUT + 7 指标评估器 + regime Pareto 的 **Stage 4 联合搜索器**。两者通过 `constraints.is_legal_for_hardware` + `propagation.propagate_with_report` 共享合法性逻辑。

**与 Gap1 三臂的对应**: S0(no-schedule) = 查 LUT 的 default 列; S1(serial) = 先在 default 列上贪心定 P×Q, 再对该单点跑内环; S2(joint) = 外环每候选都查/触发内环 tuned 列。三臂共用同一升级后的搜索器, 只切"何时调用内环"。

---

## §5 开放问题 / TVM 集成技术风险点 [待定]

1. **★[future — 量化轴日后] INT8 在同一框架内可比性**: ★平台已迁移 TVM, 量化(INT8/Q)非本架构主轴(见首部更正2), 本风险点仅在日后引入量化轴时才相关、**不阻塞 MVP(prune×schedule)**。relax **无 INT8 量化 pass**(已核验), INT8 必走 **BYOC-TRT**。于是 fp16 schedule 走 TVM-MetaSchedule 口径、INT8 走 TRT 口径 —— 两者**不在同一编译栈**, 内环回灌的 (lat,energy) 口径不一致, 加性组合 `Lat_net` 会混口径。日后需决策: (a) 量化臂只在 4090 TRT 口径单独报、不与 TVM schedule 臂混 `Lat_net`; 或 (b) schedule 臂也统一走 TRT。Gap1 §10 风险4 同此。
2. **fusion/neck 导入 BLOCKED**: 两模型(Pyramid grid_sample / CoDriving where2comm+scatter)的 attention neck 都导不进 relax(shape 推断 bug)。当前所有 schedule 实验都在 **backbone-only 子图**上做。neck 占 e2e 16–55%(Amdahl), 不解决则 schedule 搜索的加速被严重稀释。路径 = relax torch 前端 / BYOC-TRT GridSample plugin。
3. **TVM 绝对延迟常追不上 TRT**(conv-dense): TVM 的价值在**发现/量化/缓解耦合**(layout/tile/tensorize 可见可搜), 不是刷 SOTA 延迟 —— 这正是本架构以 **TVM prune × schedule 为主轴**(见首部更正2)的依据: 探索/co-design 用 TVM, 不追 TRT 的绝对延迟。"最终部署 engine 是否换 TRT" 属 **future(量化轴/部署后端日后)**, MVP 流水线"输出 engine" 默认 TVM(.so), 不依赖 TRT。
4. **schedule-LUT 加性可组合性假设**: CHaNAS 的 `Lat_net=ΣLat_block` 假设块间无重叠/无融合跨块收益。我们 backbone stage 间可能有 layer fusion 收益被加性模型漏掉 → LUT 低估联合优化空间。需验证加性误差(对几个组合做整网 tune 对照)。
5. **外环 cost model 的样本极小**: 完整点 ~33–58, NSGA-II 进化需大量评估。若全靠真测则慢、靠 LGB 则样本不足。MVP 用 LUT 加性组合规避(只需块级真测, 组合免训), 但跨块外推精度待验; active-learning 补点的收敛性 [待定]。
6. **耦合强度 model-dependent → 联合搜价值条件化**(跨模型 handoff §2): Pyramid(grouped conv)显耦合 = 双环有价值; CoDriving(标准 conv)显可分离 = 内外环可退化为串行、co-design 价值≈0。流水线需能**自动判定**某模型属哪类(否则对可分离模型白跑内环)。判定信号 = "argmin schedule 是否随剪枝宽度漂移"(E-couple 已有方法), 但自动化阈值 [待定]。
7. **平台口径纪律**: schedule 搜索/相对加速在 H800 TVM 坐实; 边缘绝对延迟须 4090/Orin 真测。流水线跨平台时 cost model 迁移须学 AutoTVM 的 invariant 表示(global+local 分解), 否则 4090 的 LUT 不能直接用于 Orin [待定, 已有 M2 f 函数 R²>0.995 基础]。

---

> **一句话总结**: 基于 TVM 现有框架**可以**搭建 —— **主搜索轴 = prune × schedule**(外环 NSGA-II 剪枝 × 内环 MetaSchedule, 见首部更正2): schedule 内环(tile/layout/fusion/tensorize + cost model + 进化搜索)**整套复用 TVM MetaSchedule**, 我们只需自建**外环 P(改 IRModule/重导 ONNX) + schedule-LUT 缓存 + 多目标 NSGA-II + 7 指标评估器**, 用 ALT cross-exploration 的"上游变→下游空间重建"结构把外环剪枝 P 与内环 schedule 嵌套粘合。MVP 可"先搭起来、cost model 后补"(内环 cost model 用 TVM 现成、外环先真测/LUT 加性组合)。**量化(Q/INT8)轴为 future**: INT8 必走 BYOC-TRT、与 TVM schedule 臂口径分裂, 是日后引入量化轴才面对的风险, 不进 MVP 主流水线。
