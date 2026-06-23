# Cost Model 设计 — 代价模型构建与训练 (v1, 2026-06-19)

> ★**2026-06-19 两点更正(用户强调, 全文据此理解)**:
> 1. **W_g 定义**: W_g = **单维(贪心所搜那一维)最优、但多维(全局)非最优**的点 = 单侧贪心搜索落入并卡住的**局部最优**(贪心选它因它单维最优; 非全局因到达真全局需接受贪心已丢弃的单维次优选择)。另设 P_g = 被错过的全局点(单维次优/多维最优)。
> 2. **平台: 已迁移 TVM, 不以 TRT/INT8 为重心**。架构/实验主轴 = **TVM prune × schedule**(MetaSchedule); 量化(INT8)因 relax 无 INT8 pass 降为 **future/次要**, 不作主依赖。本文涉及 TRT/INT8 处相应降级为 future-note。

> 3. ★**耦合口径统一(2026-06-22, 见 `1_design_space_building_v1.md §0.1`)**: 本代价模型服务的"协同"= **内环(硬件调度 S, 本文真测对象)↔外环(软件 P×Q)的耦合**, 经 P(IC_BN)枢纽; 耦合强度按架构用三臂消融**测量**(非规则)。本文 block-LUT 解耦内外环正是为了在测得耦合时仍能高效联合搜。

> **定位**: 为软硬件协同搜索框架(prune×quant×schedule 联合搜)设计**代价模型(性能评估器)**的构建与训练方案, 严格基于 ALT(EuroSys'23)/ CHaNAS(LCTES'21)/ AutoTVM(NeurIPS'18) 三篇文献的可迁移机制。
> **上游事实源**: 文献研读 `../../references/study_joint_search_methods_v1.md`(尤 §三c AutoTVM / §二c CHaNAS / §一c ALT); 预测器选型 `../../methods/design/predictor_selection_v1.md`; 现状方法稿 `stage2_method_zh_v1.md`; 联合搜实验 `../../methods/design/gap1_joint_vs_serial_design_v1.md`。
> **核心痛点(必须正面解决)**: ① LGB latency 预测器"跨数量级回归崩、给负值"已弃用(`CLAUDE.md §四`); ② Pyramid e2e f_lat 预测 R² 上限 ~0.73, 特征缺口非容量问题(memory `project_f_lat_ceiling`); ③ AP 必须 finetune 真测, 预测不准(配置效应仅 ~0.04, 在任务上限 0.791 附近)。
> **标注约定**: `[借鉴-AutoTVM/CHaNAS/ALT]` = 直接搬某篇机制; `[我们的设计]` = 本项目自定; `[待定]` = 开放/需实验定。

---

## §0 一句话总览 + 核心立场

我们的代价模型**主服务对象 = TVM**(内环 schedule 用 TVM MetaSchedule 的 xgb_model, 外环 prune × 可选 quant), **不是单个回归器, 而是一个分层异构集合**: 内层(schedule, latency 单目标, 同数量级)直接复用 **TVM MetaSchedule 的 XGBoost cost model**(AutoTVM 后继, 自带 rank loss + 在线更新); 外层(prune × 可选 quant, 多目标 5/7 指标, 跨数量级)用 **per-target LightGBM**, 其中 **latency/energy 改用 rank/pairwise loss 修崩**, **AP 走真测/finetune + ΔAP 残差预测**, 二者由 **block-LUT 加性组合**(CHaNAS, 但严格界定适用边界)在外层 Pareto 搜索中拼装。（量化轴为 future/次要, 见 H1 更正 2。）

**核心立场(与三篇的根本差异)**: 三篇都是**单目标 latency**(精度作约束或一次性 one-shot 评估), 我们是**多目标 + AP 不可纯预测**。因此 **AP 不能进"纯预测的 cost model", 必须以"真测锚点 + 残差预测 + active-learning"混合形态进入搜索循环** —— 这是本设计相对三篇文献最大的扩展点, 见 §5。

```
                         ┌─────────────────────────────────────────────┐
  外层 (prune×quant)      │  外层 cost model  (per-target LightGBM)       │
  多目标 NSGA-II ────────▶│   f_lat / f_energy : rank/pairwise loss(修崩) │
   候选 c=⟨B1,B2,D⟩       │   f_AP : ΔAP 残差回归 (排序用, 不当真值)       │
                         │   f_feasible : build_success / AP_crash 分类  │
                         │   f_size : 解析公式                           │
                         └───────────────┬─────────────────────────────┘
                                         │ 延迟/能耗用 block-LUT 加性组合
                                         │ Lat_net = Σ_block Lat_LUT[block,W,P]   (CHaNAS, 见 §4 边界)
                                         ▼
  内层 (schedule)        ┌─────────────────────────────────────────────┐
  每选定 (W,P) → 重建    │  内层 cost model  (TVM MetaSchedule)          │
  schedule 空间 ────────▶│   cost_model/xgb_model (XGBoost + rank loss)  │
   (ALT cross-explore)   │   evolutionary_search + 在线实测回灌 (AutoTVM)│
                         │   → 该 (block,W,P) 最优 schedule + 实测 lat   │ → 回填 LUT
                         └─────────────────────────────────────────────┘
  AP 轴 (横切两层)       ┌─────────────────────────────────────────────┐
                         │  真测锚点 (finetune 后 1789 帧) + ΔAP 残差预测 │
                         │  + active-learning 补点 (§5: 代理/早停/one-shot)│
                         └─────────────────────────────────────────────┘
```

---

## §1 借 AutoTVM — 三条直接可落地的机制

### 1.1 rank/pairwise loss > regression loss —— 直接修我们崩掉的 LGB latency 预测器 `[借鉴-AutoTVM]`

**文献依据**: AutoTVM §3 不用回归 loss $\sum(\hat f - c)^2$, 而用 **rank loss**

$$\mathcal{L}_{\text{rank}}=\sum_{i,j}\log\!\big(1+e^{-\,\mathrm{sign}(c_i-c_j)\,(\hat f_i-\hat f_j)}\big)$$

理由: 搜索的选择阶段**只关心相对快慢排序**, 不关心延迟绝对值。AutoTVM Fig.5 实测 "rank ≥ regression"(rank 绕开了"建模绝对 cost 值"这一更难的任务), 故 rank 是其默认目标。

**`[我们的设计]` 改造方案 — 修 `lgb_v6_latency` 崩塌**:

我们的 LGB latency 崩塌**根因 = 回归 loss + 跨数量级混训**(Pyramid ~ms 与 UniV2X ~百 ms 混训 → 负值预测)。rank loss 对此**结构性免疫**: 它只学序关系, 数量级差异不再污染。具体三步:

1. **目标从绝对值改为序关系**: LightGBM 切换到 **`objective=lambdarank` / `rank_xendcg`**(LightGBM 原生支持 LambdaRank), 或自定义 pairwise logistic loss(同 AutoTVM 公式)。`label = lat`(连续值仅用于产生 pairwise sign), `group = (model_class, hardware)` —— **关键: 同一 (model,hw) 内才两两比较, 跨数量级对不进同一 group**, 从机制上杜绝"Pyramid vs UniV2X 比延迟"的非法配对。
2. **双头互补**: rank 头只给序(供 NSGA-II 支配判定足够), 但 Pareto 的**坐标轴/iso-AP latency headline 仍需绝对延迟值**。故保留一个 **`f_lat: log(lat_p50)` 分层回归头**(`predictor_selection_v1 §四` 已定的 log+分层方案)产绝对值, **用 rank 头做序的最终裁决**(冲突时以 rank 为准)。即: **回归头给坐标, rank 头给序**。
3. **校验指标改 Spearman/Kendall-τ + NDCG@k**, 不只看 R²。我们已知 f_lat 的 R² 天花板 0.73 是特征缺口(`project_f_lat_ceiling`), 但**排序质量(Kendall-τ)可能远高于 R²** —— 对搜索而言后者才是真正指标。**这条很可能直接把"R² 0.73 的废预测器"救成"排序够用的可用预测器"**(低成本高确定性, Gap1 §6 已预判)。

> **`[待定]`**: rank loss 是否也能缓解 f_lat 的 0.73 天花板?天花板归因于缺 `n_non_pow2_groups` 类结构-硬件交互特征(memory)。rank loss 不补特征, 故**绝对值 R² 大概率仍卡 0.73, 但排序 Kendall-τ 待测** —— 需一次离线实验(在现有 ~200 行 latency 数据上对比 regression vs rank 的 Kendall-τ)。

### 1.2 内层 schedule cost model 直接复用 TVM MetaSchedule `xgb_model` —— 不重造轮子 `[借鉴-AutoTVM]`

**文献依据**: AutoTVM 的 cost model $\hat f(x)$ 从 **low-level loop AST 抽领域特征**: 内存访问次数、数据复用率、loop 结构 + unroll/vectorize/thread-binding 标注(GBT 路线), 或 TreeGRU 递归编码 AST(无需手工特征但慢)。这些特征**对 schedule 搜索是不可替代的**(它们编码了 tiling/unroll 如何影响 cache 与并行)。

**`[我们的设计]`**: 我们**完全不自建** schedule 级 cost model。route2 内层 schedule 搜索走 TVM **MetaSchedule**(AutoTVM 在 apache/tvm main 的后继, 路径 `meta_schedule/cost_model/xgb_model.py`), 它:
- 自带 **XGBoost cost model**, 特征 = `meta_schedule/feature_extractor`(per-BlockRV 的 buffer touch / arith intensity / loop 结构 — 即 AutoTVM 那套 low-level AST 特征的现代实现);
- 自带 **rank-style 目标 + evolutionary_search**(AutoTVM 模拟退火的后继, 同样在线实测回灌, 见 §2.1);
- 我们只需提供 workload(剪枝后的 backbone TIR)与 target(H800/4090/Orin)。

**理由**: schedule 级特征是 AST 结构的函数, **不是我们外层那种表格配置**, 自建等于重写 TVM 编译器栈。我们的贡献在**外层 + AP 轴 + 多目标**, 不在重造算子级 cost model。Gap1 §10.5 首轮已用 MetaSchedule 的 default-dlight vs tuned 跑出 7.18× 对齐陷阱实证, 路径已打通。

### 1.3 在线训练循环(探索时实测 top-k 回灌)`[借鉴-AutoTVM + ALT]`

**文献依据**: AutoTVM Algorithm 1 —— 每轮用 cost model $\hat f$ 跑并行模拟退火选 **预测 top-k batch** → **真硬件实测** → 收集 $(e,s,c)$ 加入 $\mathcal{D}$ → **更新 $\hat f$** → Markov 链状态跨更新持久化 + ε-greedy(εb≈0.05)保探索。ALT 同样在线训练 XGBoost(探索时只实测每 batch 的 top-k, 这些实测回灌训 cost model)。

**`[我们的设计]` — 两层都用在线更新, 但循环不同**:
- **内层(schedule)**: 直接是 MetaSchedule 的内置循环(它本就 AutoTVM Algorithm 1 的实现), 我们不另写。
- **外层(prune×quant)**: 这是我们的 **active-learning 闭环**(`predictor_selection_v1 §四` + `stage2_method §性能评估器` 已设计), 与 AutoTVM Algorithm 1 同构:
  ```
  pilot 集 (~200 行) 训外层 cost model v0
    └▶ NSGA-II 用 cost model 选 Pareto 前沿 Top-K (= AutoTVM 的"预测 top-k batch")
        └▶ data-orchestrator 下采样指令 → sw/hw 专家真测 (= "真硬件实测")
            └▶ 回灌 dataset_v2 重训 cost model (= "更新 f̂")
  ```
  - **acquisition `[我们的设计]`**: 不是纯 top-k, 而是 **quantile 宽度 × Pareto 邻近度** 加权(`predictor_selection §四`), 即 AutoTVM top-k 的不确定性升级版(AutoTVM §3 自己也说"uncertainty estimate 是 worthy candidate, 但他们未用"; 我们用 quantile LGB / bagging 方差补上)。
  - **ε-greedy 保探索 `[借鉴-AutoTVM]`**: 每批留 ~5–10% 随机配置(尤其对齐边界附近), 防 cost model 把搜索锁死在已知好区、漏掉 §5 的"非÷32 sweet spot"。

---

## §2 借 ALT — cross-exploration 决定 cost model 的"调用结构"

> ALT 本身的 cost model 就是 XGBoost 在线训练(同 AutoTVM, §1 已覆盖)。ALT 对**我们 cost model 的独特贡献不在模型本身, 而在"何时调用哪个 cost model"的结构**。

### 2.1 "宽度变 → schedule 空间重建" 决定内外层 cost model 解耦 `[借鉴-ALT]`

**文献依据**: ALT 核心洞察 —— layout 一变, loop 搜索空间就重建, 上一轮搜过的点在新空间失效。对策 = cross-exploration: joint stage 每选一个 layout → 重建 loop 空间 → 多轮 loop 优化 → 最优 loop 性能回灌作 reward。

**`[我们的设计]`**: 我们的 "prune 选定通道宽度 W → 合法 schedule(tiling/tactic)空间重建" 与之**完全同构**。这直接决定 cost model 的两层**必须解耦、不能合一**:
- 内层 cost model(MetaSchedule xgb)的特征是**给定 W 后的具体 TIR AST** —— W 一变, AST 变, 内层 cost model 的输入分布变 → **内层 cost model 不能跨 W 复用同一个 fit**(每个 (block,W,P) 各自 tune)。
- 外层 cost model(LGB)的特征是**配置标量**(prune_rate / q_bits / 路由), 它**消费内层回灌的"该 W 下最优 schedule 的实测延迟"作为标签** —— 即外层 LGB 学的是"配置 → (该配置内层调优后的) 最优延迟", 而非"配置 → 默认 schedule 延迟"。这正是 ALT "把最优 loop 性能回灌作 reward" 的搬运。

**含义**: 我们的外层 cost model 标签 = **内层 schedule 调优后的延迟**(LUT 查表值), 不是裸 PyTorch/默认延迟。这保证外层学到的是"软硬件协同后"的真前沿, 而非串行前沿。

### 2.2 invariant 表示(global + local 分解)→ 跨硬件(4090↔Orin)迁移 `[借鉴-AutoTVM/ALT 共享]`

**文献依据**: AutoTVM §4(Eq.4) —— 跨 workload/硬件迁移的关键是 **transferable representation that is invariant**: 用 low-level AST 的 **context-relation 特征**(对搜索空间不变, 公式 $R^{(ij)}_i=\max_{k:Z_{kj}<\beta_t}Z_{ki}$), 然后 cost model 分解为

$$\hat f(x)=\hat f^{(\text{global})}(x)+\hat f^{(\text{local})}(x)$$

$\hat f^{\text{global}}$ 在历史数据 $\mathcal{D}'$ 上训(给冷启动初值), $\hat f^{\text{local}}$ 在当前域少量数据上微调。Fig.8 跨 workload 省 2–10× trials。

**`[我们的设计]` — 跨硬件 cost model 迁移**:
- **我们已有的资产**: M2 f 函数跨平台 latency 拟合 **R²>0.995**(4090↔Orin, `CLAUDE.md §3.2`)。这本身就是一个极强的 **global 项**: $\hat f^{\text{global}}_{\text{Orin}}(c)\approx \phi(\hat f_{4090}(c))$, $\phi$ = 已拟合的跨平台映射。
- **分解落地 `[我们的设计]`**: 新硬件(如 Orin)的 latency cost model = **`f̂_Orin(c) = f̂_global(c) + f̂_local,Orin(c)`**, 其中 `f̂_global` 来自 4090 大样本 + M2 映射(冷启动, 不需 Orin 数据), `f̂_local,Orin` 是 Orin 少量实测(O(10) 点)拟合的残差。这把 Orin 的标注预算从"重建整个 cost model"降到"拟合一个残差头"(对应 AutoTVM 2–10× trials 节省)。
- **invariant 特征 `[我们的设计]`**: 我们外层的 invariant 表示 = **配置标量中对硬件不变的部分**(prune_rate, q_bits, params, `n_non_pow2_groups`, `min_plane_mod32`), 硬件相关的(d_scheme, DLA 可达性)进 local 头。内层 schedule 的 invariant 表示直接用 MetaSchedule 的 AST context 特征(已是 AutoTVM 那套)。
- **`[待定]`**: M2 f 函数当前是**整模型级**线性映射(R²>0.995), 是否 per-block / per-config 仍成立?若 per-block 也成立, 则 block-LUT(§4)可直接跨平台移植, 只需对少数 block 在 Orin 上校准。需小规模验证。

---

## §3 借 CHaNAS — block-LUT + 加性延迟模型(认真评估"看着不靠谱"的疑虑)

> **用户原话**: "CHaNAS 的 LAT 方法看着不靠谱, 但对我们是否也有启发?" 本节正面回答: **加性延迟模型 `Lat_net=ΣLat_block` 在我们的场景有明确的"可用区"与"会崩区", 不能无条件搬, 但分块预调度的工程骨架对我们极有价值。**

### 3.1 CHaNAS 做了什么 `[借鉴-CHaNAS]`

**文献依据**: CHaNAS 把单级联合空间 $R^B\cdot S^B$ 分解为 $R\cdot B\cdot S + R^B$: 对每个 block × 调度组合**预调度一次**(GP cost model + 启发式 mutation, split factor 限可整除), 把"每 block 最优调度 + 延迟"存进 **block 性能 LUT**, 全网延迟 **`Lat_net = Σ_i Lat_block_i`(Eq.3, 加性)**。精度侧另用 one-shot 超网 + MLP 精度预测器。

### 3.2 "加性延迟"对我们成立吗? —— 分情形裁决 `[我们的设计]`

加性模型 `Lat_net=ΣLat_block` 的隐含假设 = **block 之间无重叠、无跨 block 融合、无全局 tactic 选择、无同步 barrier 干扰**。逐条对照我们的真实场景:

| 场景 | 加性是否成立 | 依据 | 处置 |
|---|---|---|---|
| **(A) backbone 内 stage 间串行(单 agent, 单设备, 各 stage 独立 TRT/TVM 编译)** | ✅ **近似成立** | stage 串行执行, 上一 stage 输出喂下一 stage, 时间确实近似相加(忽略层间 launch 间隙) | **可用**: 这是我们 LUT 的主用区。Gap1 §3 的 backbone stage 分块即此 |
| **(B) TRT 跨 stage 算子融合 / 全局 tactic 选择** | ❌ **崩** | `predictor_selection §二` 已实证: TRT 跨层融合 + tactic 全局选择 → 整网延迟 ≠ 逐层和(TRT FP32 3.36ms ≠ 逐层和); per-stage 混精 18/18 比全局 INT8 慢 17–43% = "边界 reformat 不可加"反例 | **禁用纯加性**: 若内层用 **TRT-auto** 编译整网, 加性失效。**对策**: 要么按"已融合的子图"为 block 粒度(融合发生在 block 内, block 间不再融合), 要么 LUT 存"整段(含融合)实测"而非逐 stage 和 |
| **(C) 协同融合 neck / 同步 barrier(双 agent body 在 fusion 处合并)** | ❌ **崩** | `CLAUDE.md §0.6`: 跨 agent fuse 是同步 barrier = 并行收益硬上限; fusion 是 join 点, 不是顺序 block | **禁用加性**: fusion 段必须作为**单个不可分 block** 整体进 LUT(测它的真实延迟), 不能拆成 RSU 段 + 车端段相加。CoDriving fusion 一半是 backbone 重跑(memory `tvm_codesign_crossmodel`), 更不可加 |
| **(D) 异构 GPU∥DLA / 跨进程流水(Orin 多进程 1.34×)** | ❌ **崩(且方向相反)** | `CLAUDE.md §0.9`: GPU∥DLA 重叠真实(1.34×)但需多进程编排; 此时 `Lat_net = max(段) 而非 Σ` | **改 max-模型**: 并行段用 `Lat = max(Lat_GPU_branch, Lat_DLA_branch) + handoff`, 不是加。handoff 体积依赖(016 拆分 1.29× / 032 拆分 0.91× 负结果) → LUT 须含 handoff 项 |

### 3.3 我们的 LUT 设计 + 适用边界 `[我们的设计]`

**结论**: 我们**采用 CHaNAS 的分块预调度骨架**(R·B·S 离线建表防指数爆炸), 但**组合算子不是无条件加法**, 而是 **"加性为默认、按拓扑结构分段切换 max / 整段实测"** 的混合算子:

```
Lat_net(配置 c) =  Σ_{串行 block b}  Lat_LUT[b, W_b, P_b]          # (A) 串行段加性 — 主用区
                 + Σ_{融合段 g}      Lat_LUT[g(整段, 含 join), W, P] # (C) fusion 段整测, 不拆
                 + Σ_{并行段 p}      ( max_branch Lat_LUT[...] + handoff(体积) )  # (D) 异构并行段取 max
```

**LUT 的键**: `(block_id, width_W, precision_P, hardware, latency_kind)`; **值**: `(最优 schedule, 实测 lat_p50, energy)`。**latency_kind 必须入键**(口径隔离铁律, `predictor_selection §四`: body_subnet / collab2 / e2e / 板级不可混加)。

**适用边界(写死, 防误用)**:
1. **加性仅用于"无跨 block 融合的串行段"**(A)。若整网走 TRT-auto 整体编译(B), **直接存整网实测**, 不查 LUT 加。
2. **fusion / 协同 join 段(C)永远作单个不可分 block 整测**。
3. **异构并行(D)用 max + handoff, 不用 Σ**。
4. **LUT 是"机制加速器", 不是"真值替代"**: 外层 Pareto 前沿 Top-K **仍须阶段三整网真测验证**(`stage2_method §联合搜索器`), LUT 只用于搜索内圈快速排序。

> **为什么"看着不靠谱"但仍有启发**: CHaNAS 的纯加性确实在我们 (B)(C)(D) 三类崩 —— 用户的直觉对。**但其真正价值是"分块预调度 + LUT 缓存"避免外环每候选重跑内环 schedule 搜**(把指数 $R^B S^B$ 降到线性 $R\cdot B\cdot S$), 这个工程骨架与"加性是否精确"正交。我们保留骨架、换掉组合算子, 即取其精华去其糟粕。

### 3.4 CHaNAS 精度侧(one-shot 超网 + MLP 精度预测器)→ 我们 AP 的部分启发 `[借鉴-CHaNAS, 部分]`

CHaNAS 用 one-shot 权重共享超网免逐候选训练、子网继承权重快速评精度 + MLP 精度预测器。**对我们 AP 轴的启发见 §5(weight-sharing 是否可行)**。注意: CHaNAS **不联合 AP 进 latency cost model**(精度是约束/另一预测器), 这与我们一致 —— **AP 与 latency 用不同机制, 不混进一个 cost model**。

---

## §4 我们的分层 cost model 整体结构

### 4.1 结构总览(谁预测什么、用什么、标签从哪来)`[我们的设计]`

| 层 | 预测对象 | 模型 | 目标/loss | 标签来源 | 借鉴 |
|---|---|---|---|---|---|
| **内层** | schedule→latency(单 W,P, 同数量级) | TVM MetaSchedule `xgb_model` | rank-style + 进化搜 | 内层实测回灌(在线) | AutoTVM |
| **外层** | 配置→latency(跨 W/精度) | per-(model,hw) LightGBM | **lambdarank(序) + log 回归(值)双头** | 内层 LUT 加性组合值(§3) | AutoTVM rank + CHaNAS LUT |
| **外层** | 配置→energy | per-(model,hw) LightGBM | rank + log 回归 | E4/E5/E6 真测 | AutoTVM rank |
| **外层** | 配置→throughput | 派生 `1000/lat`(batch=1) / 独立 LGB(batch>1) | — | 1/lat 或 batch 实测 | — |
| **外层** | 配置→model_size | **解析公式**(params) + LGB(engine_size) | — | 计算/真测 | — |
| **可行性** | build_success / AP_crash | LightGBM 二分类 ×2 | logloss | 真测(含负样本) | predictor_selection |
| **AP 轴** | 配置→ΔAP | LightGBM 回归(**仅排序用**) + **真测锚点** | 回归(残差) | **finetune 后 1789 帧真测** | §5 专论 |
| **驾驶轴** | (AP,ℓ)→DS/RC | 两级: 感知头 + 闭环标定响应面 g | — | 闭环标定一族曲线 | stage2_method |

### 4.2 为什么这样分层 `[我们的设计]`

1. **内外分层的根因 = 数量级 + 特征类型不同**。内层同数量级(单 W,P 下的 schedule 变体, µs 级 span)、特征是 AST(连续结构)→ XGBoost 回归/rank 都行, 用 TVM 现成。外层跨数量级(prune 0%→90% + FP32→INT8, 延迟 span 数量级)、特征是表格标量 → **必须 rank loss + 分层**(否则重蹈 lgb_v6 崩塌)。**把这两者合一个 cost model 是 lgb_v6 崩塌的根因之一** —— 分层是修法。
2. **latency/energy 用 rank, AP 用残差回归 —— 因为信噪比天差地别**。latency/energy span ~3×(信号强), AP 配置效应仅 ~0.04 且在任务上限附近(信号弱, `predictor_selection R5`)。AP 若也强行进 cost model 当真值预测, 会被噪声主导 → **AP 的预测只配做"排序/筛掉明显差的", 真值必须 finetune 真测**(§5)。
3. **size 解析、throughput 派生 —— 不浪费样本**。params 是宽度的确定函数(公式直算); throughput 当前 corr(1/lat)=0.9996, batch 轴数据就绪前不独立建(`stage2_method`)。

### 4.3 如何训练 — 数据从哪来、冷启动、在线更新、需要多少点 `[我们的设计]`

**(a) 训练数据来源(分层)**:
- 内层 schedule cost model: **不需要我们提供训练集**, MetaSchedule 在 tune 过程中自产自训(AutoTVM Algorithm 1)。建 LUT 时每个 (block,W,P) tune 一次(几百~上千 trial), 产物入 LUT。
- 外层 latency/energy: 来自 `dataset_v2`(当前完整点 58 行, latency-only ~200 行)+ block-LUT 加性组合的合成点。
- 外层 AP: 来自 finetune 后真测(`stage_a_ap_real.parquet` 金标准 4 位真值 + 各剪枝档)。

**(b) 冷启动 `[借鉴-AutoTVM global 项 + CHaNAS pilot]`**:
- 内层: MetaSchedule 自带 `database`(历史 tune 记录), 同类 block 可 warm-start(AutoTVM tophub 思想)。
- 外层: 用现有 ~200 行 pilot 训 v0(`predictor_selection §四`)。跨硬件用 §2.2 的 `f̂_global`(4090 大样本 + M2 映射)给 Orin 冷启动。

**(c) 在线更新(§1.3 闭环)**: NSGA-II 选 Top-K → 真测 → 回灌重训。

**(d) 需要多少点 `[待定, 有离线估计]`**:
- 外层: `predictor_selection` 的离线模拟显示 **敏感度分层采样 K=32 即可到 R²≥0.85**(待跑验证)。结合 rank loss(只需序), **实际需求可能更低**(rank 比 regression 省样本, AutoTVM 实证)。
- AP: 受 finetune 成本硬约束(每点数 GPU-时), **目标是 O(10) 个真测锚点 + 残差预测插值**, 不是密集采样。
- 内层 LUT: 组合数 = B × |W| × |P|(线性), Pyramid ~3 stage × ~5 宽度档 × ~3 精度档 ≈ 45 次 tune, 可控。

---

## §5 AP 这种"必须 finetune 真测"的指标怎么进搜索循环(本设计最难点)`[我们的设计]`

> **问题本质**: AP 不像 latency 可纯预测。① 预测信号弱(配置效应 ~0.04 vs 任务上限 0.791); ② 真值代价极高(每点 = finetune 收敛 + 1789 帧, 数 GPU-时); ③ 不 finetune 的 AP 是假象(`CLAUDE.md §0.7`: 剪枝裸 AP 暴跌、finetune 后恢复, V2X-ViT p50 0.71→0.025→finetune 0.73)。三篇文献都**回避了这个问题**(单目标 latency / 一次性 one-shot 精度), 我们必须自己解。

我们采用**四层组合策略**, 按代价从低到高排, 让昂贵的 finetune 真测尽量少:

### 5.1 ΔAP 残差预测当"粗筛排序器", 不当真值 `[我们的设计 + 借 AutoTVM rank 思路]`
AP 预测器(LGB)预测 **ΔAP vs 同 ckpt baseline**(残差, 信噪比高一个量级, `predictor_selection §四.3`), **只用于在 NSGA-II 中给候选排序、筛掉明显劣的**, 绝不作为最终 Pareto 的 AP 真值。配 rank loss 思路(只需相对 AP 优劣序)。**作用**: 把需要 finetune 真测的候选从"全空间"缩到"预测前沿邻域 Top-K"。

### 5.2 代理指标(proxy)做早期粗筛 `[我们的设计, 待定具体代理]`
在 finetune 前, 用**廉价代理**预筛:
- **不 finetune 的裸 AP / 少步 finetune AP**: 已知裸 AP 是悲观偏置(暴跌), 但**相对序可能保留**(剪得越狠裸 AP 越低)→ 作下界代理。
- **重要性度量代理**: 剪枝保留的 L1 norm 能量占比 / 通道冗余度, 与可恢复 AP 弱相关。
- **`[待定]`**: 需实验确认代理与 finetune 后 AP 的 Kendall-τ(若 τ 太低则代理无效)。`CLAUDE.md §0.7` 警示: DAIR 过参数化使 finetune 几乎总能恢复 → **代理在过参数化区可能全饱和无区分度**(同剪枝悬崖找不到的困境)。故代理主要用于**剔除极端崩塌点**, 不用于精排。

### 5.3 早停 finetune `[我们的设计]`
对进入精排的候选, **不 finetune 到完全收敛**, 而是 finetune 固定少 epoch(1–2)取 AP 趋势 + 外推。风险: `CLAUDE.md §0.4` 记录"昨夜非单调 AP 是欠拟合噪声, 收敛后消失" → **早停 AP 有欠拟合噪声**, 须配 ±0.01 噪声带判读, 只用于排序不用于绝对值。最终 Pareto 锚点仍须收敛 finetune。

### 5.4 CHaNAS one-shot weight-sharing 精度预测 —— 可行性分析 `[借鉴-CHaNAS, 部分可行]`
CHaNAS 用 one-shot 超网, 子网直接继承超网权重免逐候选训练。**对我们是否可行?**

| 维度 | CHaNAS(NAS) | 我们(prune×quant) | 可行性裁决 |
|---|---|---|---|
| 搜的是什么 | 架构拓扑(深度/通道/kernel) | 固定拓扑 + 剪枝率 + 量化 | 剪枝 = 通道子集, **天然是"子网"** → weight-sharing **概念上适配** |
| 权重来源 | 超网联合训练(16×P100, 4.5 天) | 已有 base ckpt(官方收敛) | **我们已有"超网"= base ckpt**, 剪枝子网 = 从中选通道, **不需重训超网** ✅ |
| 评精度 | 子网继承权重 + val 直接测(免训) | 剪枝子网继承 base 权重直接测 | ✅ **可行但是"裸 AP"**(= 5.2 的悲观下界), 因剪枝后未 finetune |
| 与 finetune 的 gap | 超网渐进蒸馏使继承权重接近 finetuned | base 权重剪完没 finetune, gap 大(0.71→0.025) | ❌ **关键障碍**: 我们的剪枝子网继承权重 ≠ finetuned 权重, 裸 AP 严重低估真 AP |

**结论 `[我们的设计]`**: CHaNAS 式 one-shot weight-sharing **概念适配(剪枝=子网)但精度上不可直接用** —— 因为我们的"超网"(base ckpt)**没有经过 NAS 超网那种"让所有子网共享后都接近最优"的渐进蒸馏训练**, 裸继承权重的 AP 是悲观偏置。**两条改造路径(待定)**:
- **(i) `[待定]` 训一个"剪枝鲁棒超网"**: 借 CHaNAS 渐进蒸馏思路, 对 base 做一次 slimmable/once-for-all 式训练(随机宽度子网联合训), 使任意剪枝宽度的继承权重都接近其 finetuned 值 → 之后 weight-sharing 评 AP 才准。**成本**: 一次性超网训练(类似 CHaNAS 4.5 天量级), 是否值得取决于搜索候选数。
- **(ii) `[future/次要]` 量化轴用 weight-sharing 更可行**: PTQ(post-training quant)本就不重训权重, **INT8/FP16 的 AP 可在同一 finetuned 权重上一次校准多档评估**(QuantV2X harness), 比剪枝轴的 weight-sharing 障碍小。（量化轴日后纳入再说, 当前不作主轴依赖, 见 H1 更正 2。）

### 5.5 AP 进搜索循环的最终流程 `[我们的设计]`
```
NSGA-II 候选
  └▶ (5.1) ΔAP 残差预测 + (5.2) 代理: 筛掉明显崩塌点      [零真测成本]
      └▶ 进入前沿邻域的 Top-K
          └▶ (5.3) 早停 finetune 粗排 (1-2 epoch, 排序用)  [中成本]
              └▶ 最终前沿候选 (O(10) 个)
                  └▶ 收敛 finetune + 1789 帧真测 (真值锚点) [高成本, 阶段三]
                      └▶ 回灌重训 AP 残差预测器 (active-learning)
```
**核心思想**: AP 不是"预测进 cost model", 而是 **"预测+代理做漏斗粗筛 → 真测只投给前沿锚点 → 真测回灌改进预测"**。这是相对三篇文献(都不需要昂贵真测精度)的本质扩展。

---

## §6 三篇 → 本设计的决策映射表(一览)

| 文献机制 | 本 cost model 的落地决策 | 标注 |
|---|---|---|
| AutoTVM rank loss > regression | 外层 LGB latency/energy 改 lambdarank(序) + log 回归(值)双头 → 修 lgb_v6 崩塌 | 借鉴-AutoTVM(§1.1) |
| AutoTVM XGBoost cost model(AST 特征) | 内层 schedule 直接用 TVM MetaSchedule `xgb_model`, 不自建 | 借鉴-AutoTVM(§1.2) |
| AutoTVM Algorithm 1 在线实测回灌 + ε-greedy | 外层 active-learning 闭环(quantile×Pareto 邻近 acquisition + ε 随机) | 借鉴-AutoTVM+ALT(§1.3) |
| AutoTVM Eq.4 global+local 分解 + invariant 表示 | 跨硬件: `f̂_Orin = f̂_global(4090+M2) + f̂_local,Orin`(残差头, 省 Orin 标注) | 借鉴-AutoTVM(§2.2) |
| ALT cross-exploration(宽度变→空间重建) | 内外层 cost model 解耦; 外层标签 = 内层调优后延迟(非默认) | 借鉴-ALT(§2.1) |
| CHaNAS block-LUT + Lat_net=ΣLat_block | 分块预调度建 LUT(防指数爆炸), 但组合算子 = 加性/max/整测混合(按拓扑) | 借鉴-CHaNAS, 改造(§3) |
| CHaNAS divisible-split 限制 | 内层 schedule 空间剪枝 + 外层对齐特征(`n_non_pow2_groups`); ÷32 对齐他证 | 借鉴-CHaNAS(§3.3) |
| CHaNAS one-shot 精度预测 | AP weight-sharing: 概念适配但裸权重悲观, 需 slimmable 超网或仅用于量化轴 | 借鉴-CHaNAS, 受限(§5.4) |
| 三篇共识: 单目标 latency | **我们扩多目标 + AP 真测漏斗** = 本设计最大原创扩展 | 我们的设计(§5) |

---

## §7 开放问题(待实验/待定)

1. **`[待定]` rank loss 救得了 f_lat 吗?** R² 天花板 0.73 是特征缺口, rank 不补特征 → 绝对值大概率仍卡; 但**排序 Kendall-τ 是否够搜索用**需在现有 ~200 行上做 regression vs rank 的离线对比。这是修预测器的第一个验证实验。
2. **`[待定]` 加性 LUT 的崩区边界要实测标定**: (B) TRT 融合到底使 Σ 偏离多少?(C) fusion 段整测 vs 拆和的误差?(D) max+handoff 模型的 handoff 项怎么参数化(已知 016 拆分 1.29× / 032 拆分 0.91× 负结果)?需小规模 ablation 量化加性误差, 定"何时退回整网真测"的阈值。
3. **`[待定]` M2 跨平台映射是否 per-block 成立?** 若成立, block-LUT 可跨 4090↔Orin 移植(只校准少数 block); 若只整模型级成立, 跨硬件迁移退到整模型 global 项。
4. **`[待定]` AP 代理指标的有效性**: 裸 AP / 重要性度量与 finetune 后 AP 的 Kendall-τ?DAIR 过参数化是否使代理在主用区全饱和(同剪枝悬崖找不到的困境)→ 代理可能只能剔极端点。
5. **`[待定]` 是否值得训"剪枝鲁棒超网"(slimmable/OFA)** 换 AP weight-sharing?成本(一次性超网训练)vs 收益(免逐候选 finetune)取决于搜索候选规模 —— 候选少则不值, 候选多(>O(100) finetune)才划算。
6. **`[future/次要]` 内层 MetaSchedule 的 INT8 口径**: relax 无 INT8 pass, 量化臂须 BYOC-TRT(Gap1 §8.3) → 内层 cost model 的 INT8 latency 是 TRT 口径, 与 TVM schedule 口径不可混加 → LUT 须按 (latency_kind, 编译后端) 分键。**注: 量化轴非主轴, 此 INT8/BYOC-TRT 口径为 future-note, 不作主依赖(见 H1 更正 2)**; cost model 当前主轴 = TVM prune × schedule。
7. **`[待定]` energy cost model 数据量**: 当前仅 E4 起步(INT8 省 30–52% J/frame), energy 的 rank 预测器需要多少跨档真测点才稳?
8. **`[待定]` 驾驶轴 DS/RC 进 cost model 的两级响应面 g 标定**: 当前闭环行 0(gated), 待 τ_perc 曲线实验补全; g 是延迟+精度的下游函数, 其拟合质量决定 DS 能否进 Pareto(`stage2_method §备注`)。

---

> **与现有事实源的关系**: 本文是 `predictor_selection_v1.md`(选 LightGBM 的理由)与 `gap1_joint_vs_serial_design_v1.md`(§6 cost model 那一小节)的**展开与统一**, 聚焦"代价模型如何构建与训练"。预测器**选型**(为何 GBDT 不为 GP/MLP/GCN)以 `predictor_selection_v1.md` 为准, 本文不重复论证, 只在其选定的 GBDT 上叠加三篇文献的 loss/迁移/LUT/在线机制, 并新增 AP 真测漏斗(§5)这一三篇未覆盖的扩展。
