# Space Building 设计 — 搜索空间构建与收缩 (v1, 2026-06-19)

> **定位**: 为软硬件协同搜索框架的 **auto-tuning 之前的"空间构建"环节** 写一份严格基于 ALT / CHaNAS / AutoTVM 三篇论文的设计稿。本稿专注三轴 **P(剪枝)× Q(量化)× schedule(TVM 调度)** 的合法空间如何 **快速构建 + 分解 + 收缩到可搜规模**，承接 `stage1_method_zh_v1.md`(刻画) 与 `stage2_method_zh_v1.md`(搜索器/评估器)之间的缺口，并把 `gap1_joint_vs_serial_design_v1.md` 的实验骨架落到"空间定义"这一层。
>
> **标注约定**: 每条标 `[借鉴 ALT/CHaNAS/AutoTVM 哪个机制]` / `[我们的设计]` / `[待定]`。
>
> **三篇文献依据**: `references/study_joint_search_methods_v1.md`(PDF 精读 digest)。现状代码: `framework/searcher_v0.py`(随机采样+传播+约束过滤, **无 schedule 轴/无评估器/无 Pareto**)。
>
> **纪律提醒**: 对齐(`in_per_g∈2^k`)只是空间几何的 **一种** 合法性约束, 不是只盯 trap25 一个点; 下文一律把它一般化为"合法性谓词", trap25/p25 仅作其实例。

---

## §0 总览 — 我们的"空间构建"对应文献的哪一步

| 文献 | 它的"空间构建"做了什么 | 我们对应的空间构建 |
|---|---|---|
| **ALT** | 只为复杂算子(conv/GEMM)建 layout 变换空间 + layout propagation 传播其余张量 + analytic 裁出有前途子空间(C2D O(10^19)→O(10^6) 6 参数) | 软件旋钮(剪枝宽度/量化档)一变 → **合法 schedule 空间重建**; 两阶段 joint/schedule-only |
| **CHaNAS** | R^B·S^B → 分解为 **R·B·S + R^B**; block-level pre-scheduling 存 LUT; 按 input-res×expand-ratio 划子空间选 CDF 最低 | 按 **块**(Pyramid stage / 双 agent 段)预调度每个 **(宽度,精度)** 组合建 **schedule-LUT**; 外层加性组合; 子空间预筛 |
| **AutoTVM** | schedule 模板空间 S_e(tiling/reorder/unroll/vectorize, O(10^9)); divisible split factor; rank loss | 内环 schedule 轴的 **可调原语清单** + 合法 split 网格(divisible) |

**一句话**: 我们的 Space Building = **两阶段空间定义**(借 ALT cross-exploration) × **按块分解 + schedule-LUT**(借 CHaNAS) × **divisible/对齐合法性 + propagation 预筛**(借 ALT propagation + CHaNAS divisible + AutoTVM 模板)。最终把理论 P×Q×schedule(~10^15 × O(10^9) 指数爆炸) 收缩为 **块级 LUT(线性) + 外环组合(~10^4)**。

---

### §0.1 协同的本质 —— 内外环(软件优化×硬件优化)耦合, 强度按架构"测量"而非"规则" (★统一口径, 2026-06-22)

本框架是**内外双环搜索**: **外环 = 软件优化轴 (剪枝 P × 量化 Q)**, NSGA-II 查 LUT 加性组合; **内环 = 硬件优化轴 (TVM 调度 S)**, 按块跑 MetaSchedule 真搜。我们要主张的"协同"**不是 P/Q/S 三轴各自独立的强纠缠**, 而是 **内环(硬件调度)与外环(软件 P×Q)的强耦合** —— 即 **外环的 (P,Q) 排序离不开内环 S 的结果, 反之亦然**。

- **机理枢纽 P(IC_BN)**: 外环的结构选择(宽度/groups → IC_BN=IC/groups)决定内环调度的可张量化性与对齐合法性(§3 对齐谓词正是此机理的前移)。这是把"外环结构"灌入"内环可调性"的**唯一枢纽**; 故协同 = **P-hub 的 外环×内环 耦合**, 而非三轴对称纠缠。`grouped conv → 小 IC_BN → 强耦合 / groups=1 → IC_BN 恒大 → 弱耦合` 是我们实测得到的**机理 insight (作为科学发现报告)**, **不是写进框架的 operative 规则**。
- **耦合强度是被"测量"的, 不是被"声明"的**: 同一套三臂消融 (A-joint vs A-serial vs A-noS 的 HV 比) **就是普适的耦合度量仪** —— 任意架构跑同一度量, CoDriving 自动落到 ≈1.0 (可分离), Pyramid 落到 <1 (耦合), V2X-ViT 居中。**"可分离"是测量结果, 不是手写的 `if groups==1` 规则**; 框架因此**普适**而非工程拼装。
- **搜索据耦合分数自适应**(设计目标, 见 §2 内环预算): 内环(调度)预算按每块测得/预测的耦合分数连续分配 —— 弱耦合块退化为"搜一次", 强耦合块付全量 joint。**分流不是二元硬开关, 是耦合分数驱动的连续预算分配**。

> 详见 `coupling_map_v1.md` (逐 cell 实证: P-hub, mech3 后端 artifact 已证伪) 与 `4_design_ablation_proof_v1.md §9.8d'` (三臂消融=耦合度量仪)。本稿下文"三轴 P×Q×schedule"均指**搜索的三个维度轴**(空间定义层面), 与上述"耦合**主张**=内外环"不矛盾: 三者是被搜的轴, 耦合发生在内外环之间、经 P 枢纽。

---

## §1 ALT 映射 — 两阶段空间定义(joint stage / schedule-only stage)

### 1.1 ALT 怎么做的 [借鉴 ALT cross-exploration]

`[借鉴 ALT]` ALT 把"图级 data layout × 算子级 loop"放进同一 auto-tuning, 核心洞察是 **layout 一变, loop 搜索空间就要重建**(上一轮搜过的 loop 点在新 layout 下失效)。它的空间构建有三个动作:

1. **只为复杂算子建空间**: 仅 conv/GEMM 这类 layout-sensitive 算子建 layout 变换空间, 其余张量不搜。
2. **layout propagation**: 一个算子定了 layout, 沿数据流传播到相邻张量, 不再独立搜(避免每个张量都成自由变量)。
3. **analytic 裁子空间**: 用"layout 如何影响数据复用/cache"的分析, 把 C2D 的 O(10^19) layout × O(10^7) loop 裁成 O(10^6) 的 6 参数 tuning template(h_t/w_t/o_t/i_t/i'_t/o'_t)。
4. **cross-exploration 两阶段**: **joint stage**(同搜 layout+loop, 每选一个 layout→重建 loop 空间→多轮 loop 优化→最优 loop 性能回灌作 reward) + **loop-only stage**(layout 固定只搜 loop)。

### 1.2 同构关系 [我们的设计]

`[我们的设计]` **ALT 的 layout↔loop 墙 ≡ 我们的 (剪枝宽度 × 量化数据格式)↔schedule 墙**:

- ALT 的 **layout**(张量存储格式) ≈ 我们的 **剪枝决定的通道宽度 W + 量化决定的数据格式 P**(INT8 tensor-core 要求通道对齐 = 一种 layout 约束)。
- ALT 的 **loop**(tiling/reorder/...) ≈ 我们的 **schedule**(TVM tile/loop-order/fusion/tensorize)。
- ALT 的"layout 变→loop 空间重建" ≈ 我们的 **"(W,P) 变→合法 schedule 空间重建"**(通道宽度变了, 合法 tile 因子集、能否 tensorize INT8、能否融合都变)。

⇒ 我们直接套 ALT 的两阶段空间定义。

### 1.3 我们的两阶段空间定义 [我们的设计]

```
外层软件空间  Σ_sw = P(剪枝宽度) × Q(量化档)        ← 多目标, NSGA-II
   │  每选定一个 (W, P)
   ▼
[joint stage]   重建该 (W,P) 的合法 schedule 子空间  Σ_sch(W,P)
   │            (★非笛卡尔: Σ_sch 依赖 (W,P), 不是固定集合)
   ▼
内层调度空间  Σ_sch(W,P) = {合法 tile 因子 × loop-order × fusion × tensorize}
   │  内环 MetaSchedule 搜 → 最优 (latency, energy)
   ▼
   回灌该 (W,P) 的真实硬件指标
   │
   ▼
[schedule-only stage] (可选精修)  (W,P) 冻结, 再多搜几轮 Σ_sch
```

- **joint stage 的空间** = `Σ_sw × Σ_sch(W,P)`, 但 **Σ_sch 随 (W,P) 重建**(借 ALT 核心)。这是"联合"语义的空间根据: 软件旋钮与 schedule 不在两个独立笛卡尔积里, 而是 schedule 空间是软件配置的 **函数**。
- **schedule-only stage 的空间** = 固定 (W,P) 下的 `Σ_sch`(= ALT loop-only / CHaNAS-W/O / 我们 S0-S1)。
- `[借鉴 ALT "只为复杂算子建空间"]` 我们 **只为稠密通道耦合核心(conv/grouped-conv/GEMM)建 schedule 空间**; 稀疏编码器(VFE/LSS)、协同融合(通道保持)不进 schedule 搜索空间(与 stage1 "只追踪稠密核心"边界一致)。
- `[借鉴 ALT layout propagation]` 见 §3.2(剪枝→量化→schedule 的依赖传播)。

### 1.4 [待定]
- **joint stage 是否对每个 (W,P) 都真重建并重搜 schedule?** — 不。借 CHaNAS 把 schedule 搜下沉到 **块级 LUT**(§2), joint stage 实际是"查 LUT + 加性组合", 只有建 LUT 时才真跑内环。这是 ALT(每候选重搜) 与 CHaNAS(预调度查表) 的折中, 待 §2 定。

---

## §2 CHaNAS 映射 — 分块 + 空间分解 + schedule-LUT

### 2.1 CHaNAS 怎么做的 [借鉴 CHaNAS]

`[借鉴 CHaNAS]`
- **联合两侧** = NN 架构 {arch}(elastic MBConv: depth/expand-ratio/kernel/input-res) × 每块编译调度 C(graph: fusion+layout transform; operator: 10 种原语)。
- **★空间分解(核心 trick)**: 原单级联合空间 = **R^B·S^B**(R 架构/块, S 调度/块, B 块) → 分解为 **R·B·S + R^B**: 块级独立预调度 R·B·S 个 + 架构组合 R^B。指数 → 可控。
- **block-level pre-scheduling**: 每块抽计算子图 → graph 优化(fusion+layout) + operator 调度搜(GP 代价模型 + 启发式 mutation, 跨块复用 GP) → 把每块最优调度+延迟存进 **block 性能 LUT**; 全网 **Lat_net = Σ_i Lat_block_i**(加性可组合)。
- **mutation 限 divisible split**: "split factor 限制为可整除切分, 不可整除结果更差"。
- **子空间预筛(CDF)**: 按 input-res(5)×expand-ratio(4)=20 子空间, 随机采样 λ 网络比 latency CDF, **选平均延迟最低的子空间** 再进化搜。

### 2.2 我们的"块"定义 [我们的设计]

`[我们的设计]` 我们 **不动架构拓扑**(固定 PyramidFusion/CoDriving), 把 CHaNAS 的"架构旋钮 R/块"换成 **(剪枝宽度档 W, 量化精度档 P)/块**:

**块的两种切法**(模型相关, 取并):
1. **backbone stage 块**: Pyramid backbone 3 个 stage(num_filters [64,128,256] → 每 stage 一块); CoDriving backbone stage。这是 P 轴(per-stage num_filters)与 schedule 的天然耦合粒度。
2. **双 agent 部署段块**: RSU 段 / 车端段(ego)。这不是 D 轴路由 + 异构(GPU∥DLA)的天然粒度，因为两个agent不可能用一块GPU完成计算。两者**不冲突**: stage 块是 P×schedule 的载体, 部署段块是 D×异构的载体; 一个块在某个部署段上。

与 stage1 一致性: 块 = stage1 DepGraph 给出的"最小耦合组整合后的 per-stage 剪枝旋钮" 的承载单元(B1 视图), schedule 在其上重建。

### 2.3 我们的分解公式 [我们的设计 + 借鉴 CHaNAS Eq3]

设:
- `B` = 块数(Pyramid backbone = 3 stage; 双 agent = +RSU/ego 段)。
- `|W|` = 每块的剪枝宽度档数(锚点, 见 §4.1)。
- `|P|` = 每块的量化精度档数(见 §4.2)。
- `|S|` = 每个 (块,W,P) 内环 schedule 搜索预算(trial 数, 非枚举)。

**理论联合空间(指数爆炸)**:
```
|Σ_full| = (|W|·|P|·|S|)^B        ← 全交叉积, 不可行
```

**CHaNAS 式分解 [借鉴 CHaNAS R·B·S + R^B]**:
```
建表(离线, 线性):   N_LUT = B · |W| · |P|        个 (块,宽度,精度) 组合
                    每个组合跑一次内环 schedule 搜(预算 |S| trial), 存最优 schedule+(lat,energy)
外环(在线, 组合):   N_outer ≤ (|W|·|P|)^B           个全网配置
                    但延迟用加性组合查表:  Lat_net = Σ_{b=1..B} Lat_LUT[b, W_b, P_b]   (Eq3)
                    不必为每个外环候选重跑 schedule 搜
```

`[我们的设计]` 关键: **schedule 搜索成本从 `(...)^B` 降为 `B·|W|·|P|`(线性)**, 这是让"P×Q×schedule 联合搜可行"的工程核心。外环 NSGA-II 只查表 + 加性组合, 不触发内环。

**数量级估算**(Pyramid backbone, §4 取值):
- `B=3`, `|W|≈4`(对齐锚点), `|P|≈3`(FP32/FP16/INT8) → `N_LUT = 3·4·3 = 36` 次内环 schedule 搜(可离线一次性建)。
- 外环全网组合 `(4·3)^3 ≈ 1728`(单 backbone), 加上量化单元/路由段后到 `~10^4`(与 stage2 现有估算一致), NSGA-II 查表可搜。

### 2.4 子空间预筛 [借鉴 CHaNAS CDF]

`[借鉴 CHaNAS CDF + 我们的设计]` CHaNAS 用 input-res×expand-ratio 的 latency-CDF 选最优子空间。我们对应: 用 **(全局剪枝率档 × 全局精度档)** 划子空间, 在每个子空间随机采样 λ 个配置, 用 **评估器(stage2 LightGBM, 非真测)** 算 latency/AP 的 CDF, **优先把搜索预算投到 CDF 有利的子空间**。注意我们是多目标, 不能像 CHaNAS 单看 latency CDF, 应看 **(AP, latency) 联合 CDF 或预测 hypervolume**。这与 stage1 的"延迟权重优先投高占比单元"是同一思路的子空间版。

### 2.5 [待定]
- 双 agent 段的 `Lat_net` 是否仍加性? — RSU∥ego 若异构并行(GPU∥DLA 多进程 1.34×)则 **非加性**, 是 `max(Lat_RSU, Lat_ego) + handoff`。加性公式只对串行块成立。这是 CHaNAS 加性假设在我们异构流水下的 **真实偏离**, 待 §4.3 D 轴与流水形态确定后修正。
这里我还是要再次强调两个agent不能放在一块去考虑，因为两个部分是不同的设备计算的，一个是路测设备一个是车端设备。这不是说在异构GPU上就不是并行的时延问题。而是只要提到时延，就要把RSU和ego分开讨论，RSU给出一个时延，RSU给一个时延
---

## §3 快速确定 + 收缩搜索空间 — 三件套 pipeline

### 3.1 合法性约束 — 对齐谓词一般化 [借鉴 CHaNAS divisible + ALT analytic + 我们的设计]

`[我们的设计 — 一般化, 不只 trap25]` 把"对齐"从单点(÷32/trap25)上升为 **合法性谓词** `legal(W, P, hw)`:

**对齐谓词(分组卷积的真根因, 比"÷32"更准)**:
```
对每个块的每个分组卷积层(groups=g):
  in_per_g(W) = 2 · num_filters(W) / g
  命中 tensor-core 快核  ⟺  in_per_g(W) ∈ {2^k}   (★ 2 的幂, 不是简单 ÷32)
```
- 这是 `gap1 §10.5` 实测根因: num_filters=48,g=32 → in_per_g={3,6,12} 全非 2 的幂 → kernel-cliff(21615µs); base={4,8,16}/p50={2,4,8} 全 2 的幂(快)。
- `[借鉴 CHaNAS divisible-split]` CHaNAS 自限 split factor 为可整除("不可整除更差") = 同一现象在 schedule 内环的他证。我们的对齐谓词 = 该规则在 **剪枝宽度轴** 的前移(把 divisible 约束从 schedule 内环提到外环宽度选择)。
- `[借鉴 ALT analytic 裁子空间]` 用解析谓词(而非真跑)提前裁掉非法宽度, 等价 ALT 用"数据复用分析"裁 layout 子空间。

**精度门控谓词**(硬件能力注入, 来自 stage1 H):
```
q_bits(unit) ∈ Q(hw, unit)          ← 敏感单元(head/attn/grid_sample/VFE) 门控为 FP16-only
DLA 路由 ⇒ q_bits=INT8 ∧ per-tensor ∧ HWC4   ← D↔B2 硬传播
DLA × INT8 在 Pyramid 上 0/12 build 成功 ⇒ DLA 有效精度 = FP16-only(经验硬约束)
```

**现状代码对接**: `searcher_v0.py` 已有 `ALIGNED_PRUNE_RATES`(÷8 网格) + `NEAR_ALIGNED_PRUNE_RATES`(非对齐) + `is_legal_for_hardware`(按 yaml `alignment_enforcement` 动态升降级)。`[待定]` 现网格是"256 通道下 ÷8"的近似, **应改为按每块每层真实 (num_filters,g) 算 in_per_g∈2^k 的精确谓词**, 而非全局剪枝率网格(精确化是 Space Building 落地的第一步代码改动)。

### 3.2 propagation — 剪枝→量化→schedule 的依赖传播 [借鉴 ALT layout propagation]

`[借鉴 ALT layout propagation + 我们的设计]` ALT 把一个算子定的 layout 沿数据流传播到相邻张量, 不重复搜。我们的依赖传播链 **三跳**:

```
(1) 剪枝 W  ──决定──▶  通道宽度 / in_per_g / round_to floor
(2) 量化 P  ──决定──▶  数据格式(INT8→对齐 a=32, FP16→a=8) + 粒度(DLA⇒per-tensor)
(3) schedule ──受约束──▶  合法 tile 因子(须整除宽度) / 能否 INT8-tensorize / 能否 fusion
```

- **跳(1)→(2) [现状已有]**: `propagation.py` 的 D↔B2 双向传播(DLA⇒INT8/per-tensor; per-channel⇒禁 DLA)。`round_to` 由 q_bits 派生(INT8→32/FP16→8)。
- **跳(2)→(3) [我们的设计, 待建]**: 量化数据格式 + 通道宽度 **共同重建** schedule 子空间。这是 §1.3 joint stage 的"空间重建"的传播实现: 不是每个 schedule 参数自由搜, 而是宽度/格式定了后, **合法 tile 因子集被传播裁定**(tile 因子须整除宽度; INT8 块的 tensorize intrinsic 须 in_per_g∈2^k)。
- `[借鉴 ALT "改 layout 不需重写算子"]` 这要求 schedule 层是 TVM(MetaSchedule + 自定义 layout-transform pass), 而非 TRT-auto 黑盒(改不了可见枚举)。propagation 在 TVM IR 的 pass 里改写张量访问索引, 不手写 kernel。

### 3.3 子空间预筛 — CDF 式 [借鉴 CHaNAS]

见 §2.4。`[我们的设计]` 收缩 pipeline 把预筛放在合法性过滤 **之后**、内环建表 **之前**: 先用谓词裁非法, 再用评估器 CDF 选有利子空间, 只对入选子空间建 schedule-LUT, 避免在注定差的子空间上花内环预算。

### 3.4 空间收缩 pipeline 全景 [我们的设计]

```
理论空间  P×Q×schedule  (~10^15 × O(10^9), 指数爆炸)
   │
   │ ① stage1 整合: 结构耦合组 → per-stage 旋钮 (10^33 → 10^4 软件侧)        [stage1 已落地]
   ▼
   │ ② 合法性谓词裁剪 (§3.1): in_per_g∈2^k + 精度门控 + DLA 经验硬约束       [借 CHaNAS divisible + ALT analytic]
   ▼
   │ ③ propagation (§3.2): 剪枝→量化→schedule 三跳依赖传播, 非笛卡尔        [借 ALT layout propagation]
   ▼
   │ ④ 按块分解 (§2.3): schedule 搜下沉块级 LUT, B·|W|·|P| 线性建表        [借 CHaNAS R·B·S+R^B]
   ▼
   │ ⑤ 子空间 CDF 预筛 (§2.4/3.3): 评估器选有利子空间优先建表/搜             [借 CHaNAS CDF]
   ▼
可搜空间  外环 ~10^4 (查 LUT 加性组合) + 内环 schedule-LUT 线性
   │
   ▼
NSGA-II 多目标搜索 (stage2 §联合搜索器)
```

---

## §4 三轴具体取值与维度 + 空间大小估算

### 4.1 P 轴(剪枝) — 宽度锚点 [我们的设计]

- **可调对象**: per-stage `num_filters`(Pyramid backbone [64,128,256]; CoDriving 对应 stage)。`[轴塌缩]` 剪枝对象 = `channel`(2:4/element 在目标硬件无真加速, 已塌缩, 见 stage2)。
- **宽度锚点(对齐感知)**: 不是连续剪枝率, 而是 **满足 in_per_g∈2^k 的离散宽度档**:
  - Pyramid stage0(g=32): num_filters ∈ {64, 32, 16, ...} 使 in_per_g=2·nf/32 ∈ {4,2,1} (2 的幂) → 合法锚点。
  - 非对齐档(如 48, in_per_g=3) = **trap 档, 默认排除**, 仅 Gap1 sweet spot 实验显式纳入(逼出"串行踩坑/联合避坑")。
- **每块宽度档数** `|W| ≈ 4`(base + 2~3 个对齐压缩档)。`[待定]` 是否纳入 trap 档随实验目的开关(Gap1 纳入, 生产搜索排除)。
- **粒度**: per-stage(整合后 Pyramid backbone ~3-5 旋钮; 全模型 4-6, 见 stage2)。
- `[借鉴 CHaNAS]` 锚点选取学 CHaNAS 把 expand-ratio 离散成 {1.0,1.1,1.2,1.3} 的做法: 我们离散成对齐合法宽度, 不搜连续率。

### 4.2 Q 轴(量化) — 精度档 [我们的设计]

- **位宽** `q_bits ∈ {FP32, FP16, INT8}`, per-量化单元(语义桶: backbone/BEV-encoder/neck/heads, ~3-4 单元)。
- **单元精度选择**: 粗粒度 on/off(哪些单元走 INT8 vs FP16); **逐层精度委托 TRT-auto**(stage2 已定论: per-stage 强制混精被 TRT-auto 延迟驱动支配, 不枚举)，注意我们现在使用TVM而不是TRT进行加速了。
- **粒度**: weight per-tensor/per-channel(DLA⇒per-tensor); activation 恒 per-tensor。`[派生约束]` 由 D↔B2 传播定, 非自由搜。
- **每块精度档数** `|P| ≈ 3`(FP32/FP16/INT8, 敏感单元门控后可能 2)。
- `[待定]` INT8 真实 AP 代价 **未测定**(gap1 §5.2: simulated fake-quant 全失真不可信), 需真 TRT INT8 / TensorRT-ModelOpt。空间里 INT8 档 **结构上保留**, 但其 AP 标签待真测。

### 4.3 schedule 轴 — 可调原语 [借鉴 AutoTVM + ALT]

- `[借鉴 AutoTVM 模板空间 S_e]` 算子级原语: **multi-level tiling**(每 loop 轴 tile 因子, 须整除宽度)、**loop reorder**、**unroll**、**vectorize**、**(GPU) thread-binding / shared-mem cache**。
- `[借鉴 ALT]` graph 级: **fusion**(相邻 conv-bn-relu / neck 融合, 注: Pyramid/CoDriving fusion neck TVM 导入有已知坑)、**layout transform / pad-repack**(对齐救援: 48→64 + INT8 tensorize)。
- **INT8 专属**: **tensorize**(映射到 tensor-core MMA intrinsic, 须 in_per_g∈2^k)。
- **引擎**: TVM **MetaSchedule**(AutoTVM 后继): `space_generator/post_order_apply` + `schedule_rule/multi_level_tiling` + `search_strategy/evolutionary_search` + `cost_model/xgb_model`。INT8 走 BYOC-TRT(relax 无 INT8 pass)。`[借鉴 AutoTVM]` **不重造轮子**。
- **每 (块,W,P) 的 schedule 搜预算** `|S|` = MetaSchedule trial 数(如 512~2048), **非枚举**(O(10^9) 空间靠进化+cost model 引导搜)。
- `[借鉴 AutoTVM divisible split]` 内环 split factor 限可整除(同 §3.1 谓词)。

### 4.4 空间大小估算汇总 [我们的设计]

| 层次 | 空间大小 | 收缩来源 |
|---|---|---|
| 理论 P×Q×schedule(全交叉) | `(|W|·|P|·|S|)^B`, 含 O(10^9) schedule → 指数爆炸 | — |
| stage1 整合后软件侧 | ~10^4(剪枝旋钮 ~5 + 量化单元 + 路由段) | stage1 per-stage 绑定 + 轴塌缩 |
| schedule-LUT(离线建表) | `B·|W|·|P| ≈ 3·4·3 = 36` 次内环搜(Pyramid backbone) | CHaNAS R·B·S 分解 |
| 外环可搜空间 | ~10^4 配置, 查表加性组合 | CHaNAS R^B 组合 + 合法性谓词 + propagation |
| 内环单次 schedule 搜 | O(10^9) 模板空间, 搜 `|S|≈512-2048` trial | AutoTVM cost-model 引导(非枚举) |

**结论**: 三轴联合在 **分解后** 是"线性建表(36 次内环) + 多项式外环(~10^4 查表)", 而非指数。这正是 CHaNAS 让"架构×调度"可行的同款工程, 移植到"P×Q×schedule"。

---

## §5 与现状代码 / 实验的接口

- **现状 `searcher_v0.py`**: 已有 ① 软件轴随机采样 ② D↔B2 propagation ③ 合法性过滤(动态 alignment_enforcement)。**缺**: schedule 轴、评估器、Pareto、按块 LUT。
- **Space Building 落地的第一批改动**(本稿指向):
  1. 把 `ALIGNED_PRUNE_RATES`(全局 ÷8 网格)→ **按每块 (num_filters,g) 精确算 in_per_g∈2^k 的宽度锚点**(§3.1/§4.1)。
  2. 加 **块级 schedule-LUT 数据结构** `LUT[block][W][P] = (best_schedule, lat, energy)`(§2.3)。
  3. 加 **跳(2)→(3) 传播**: (W,P) → 合法 schedule 子空间(§3.2)。
  4. 外环延迟改 **加性查表 Lat_net=ΣLat_LUT**(§2.3), 异构段改 max+handoff(§2.5 待定)。
- **Gap1 实验复用**(`gap1 §10.5` 已有首轮): schedule-LUT 已在 H800 建过 base/trap25/p50 三档; Space Building 把它一般化为 `B·|W|·|P|` 全表。

---

## §6 待用户拍板的开放问题

1. **块切法的粒度** — backbone 按 stage 切(3 块)是否够? 还是要细到 conv-block? CHaNAS 用 MBConv block(粗); 我们 Pyramid 只 3 stage, 太粗可能掩盖 stage 内耦合, 太细则 LUT 膨胀。**建议默认 per-stage(3 块), 与 stage1 剪枝旋钮对齐**, 请确认。

2. **schedule-LUT 是否对每个 (W,P) 真建?** — 36 次内环 MetaSchedule 搜(每次数十分钟 GPU)是否可接受? 还是先只建对齐档(排除 trap)+ Gap1 sweet spot 显式补 trap? **建议生产空间只建对齐档(|W|=3~4), trap 档按实验目的补**。

3. **trap/非对齐宽度档默认进不进生产空间?** — 进则空间含被支配点(浪费预算), 不进则 Gap1"串行踩坑"demo 需单独构造。**建议: 生产搜索排除 trap, Gap1 消融显式纳入**(一个开关), 请确认这个双模式。

4. **异构段 Lat_net 加性 vs max(§2.5)** — RSU∥ego 走 GPU∥DLA 多进程并行(非加性)还是串行(加性)? 这决定外环组合公式。**待 D 轴流水形态(单设备串行 vs 异构并行)定**。

5. **INT8 档 AP 标签缺口(§4.2)** — INT8 真实 AP 代价未测定(simulated 不可信)。空间结构保留 INT8 档, 但搜索能否信任其 AP 预测? **建议: INT8 档先只参与 latency/energy 搜索, AP 标签待真 TRT INT8 补齐再开放为目标轴**, 请确认是否接受这个分期。

6. **子空间 CDF 预筛用单目标还是多目标 CDF(§2.4)** — CHaNAS 单看 latency CDF; 我们多目标。用预测 hypervolume 选子空间成本更高。**建议: 先用 (AP, latency) 二维联合 CDF 粗筛, hypervolume 留作 Gap1 严谨度量**, 请确认。
