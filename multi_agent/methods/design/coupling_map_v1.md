# 协同搜索空间 — 耦合地图 (Coupling Map) v1.0

## §0 真正目的(用户 2026-06-22 核心澄清 —— 一切实验的第一性前提)
绘制这张耦合地图,**最底层最核心的目的有两个**。它们不是"分类/效率",而是要**证明多维联合耦合这一现象本身**:

**① 首要目标 — 让 framework 真正做到 P×Q×S 三维全面联合搜索(不止"一个通道维度的结论")。**
当前"三轴强耦合"实质 = 三轴**经由单一机制(P 通道对齐)**耦合:P 是枢纽,P×S 与 P×Q 各自 JOINT,但 **Q×S 是否独立耦合(不经 P)未知**。这**还不是**真正的三维联合 —— 若空间能因子分解成"P-hub + 其余独立",框架只需"仔细搜 P",并非全维联合。**首要目标 = 证明 P×Q×S 的联合是不可约的(存在多条耦合机制、Q×S 也独立耦合),使全维联合搜索成为真正必要。** 地图的每个 cell 都服务于此:不断挖出"超出 P-对齐"的新耦合机制,直到三维不可约。

**② 次要目标 — 证明 CoDriving 也存在耦合陷阱(通过更高维度的耦合找到它)。**
当前"CoDriving 可分离"只是 **1–2 维(P×S、P×Q)** 视角下的观察。**次要目标 = 在更高维度的组合里搜出 CoDriving 的陷阱** —— 工作假设:标准 conv 在单维无对齐陷阱,但在某个高维组合(如 Q-granularity×P、bit-width×S、或某三维联合)里仍会出现 serial 漏掉的最优。**找到 → 耦合比"grouped-conv 专属"更普适**;高维彻底搜完仍无 → 也是诚实负结果(但必须**真搜够维度**才能下此结论,不能像之前 1 维就断"可分离")。⇒ CoDriving 的 cell 一律按**"高维陷阱猎取"**设计,不是"确认可分离"。

**③ "避开两个极端 / 搜索可分解性"的正确定位(修正我之前的主次颠倒)。**
把搜索拆成"只在耦合块联合、其余因子化"以省成本 —— 那是**地图日后被框架搜索器使用时的下游作用**,**不是绘制地图最底层的目的**。先有 ① ②(证明多维联合耦合真实且不可约 / 在高维挖出对照模型陷阱),搜索效率才是之后的应用。

---

> **本图的"方法"(服务于上面 §0 的目的)**: 在 **TVM(MetaSchedule)** 后端上,对每个维度/维度对裁决耦合结构 —— **JOINT 必须联合搜** / **SERIAL 可串行** / **SINGLE 只搜单一** / **PAIR 某两维就够**。
> **后端 = TVM,不是 TRT。** TVM 是**生成式调度器**,schedule(S)是被搜的一等轴;TRT 是封闭 auto-tuner(故 TRT 上"per-stage 混精被 auto 支配 / D 维塌缩"等结论**不外推到 TVM**)。
> 通道对齐 = 地图里的**一个 JOINT cell**,不是全部 —— 目标是用更多 cell 撑起 §0① 的"不可约三维联合"和 §0② 的"CoDriving 高维陷阱"。

> ★**口径统一更新 (2026-06-22, 与 `1_design_space_building_v1.md §0.1` 一致)**: 复核后(见 §3/§5)主张已从"P/Q/S 三轴独立不可约"诚实降级为 **内环(硬件调度 S)↔外环(软件 P×Q)的强耦合, 经 P(IC_BN)枢纽**。**"耦合/可分离"是用三臂消融(度量仪)测量出来的, 不是手写规则** —— CoDriving 测出可分离(§0② 阴性闭合)是测量结果而非"if groups==1"。`grouped conv→耦合` 是机理 insight 非框架规则。本图各 cell 即这把度量仪在不同维度对上的读数。

---

## 1. 轴定义(只列在 TVM 上真能动 latency/AP 且可搜的维度)
| 组 | 轴 | dims 文档字段 | TVM 上的取值域 | 说明 |
|---|---|---|---|---|
| **P 剪枝** | width(各 stage 通道) | D1 prune_rate / num_filters | 连续档 [16..64]×stage | 改 FLOPs + 对齐 |
| | wpg(ResNeXt 每组宽) | D6 width_per_group | {4,8,16} | 与 groups 共同定 in_per_g |
| | round_to(通道对齐) | D4 round_to | {1,8,16,32} | 显式对齐旋钮 |
| | cardinality | resnext_groups | {8,32}(架构超参) | 决定 grouped 程度 |
| **Q 量化** | bit-width | Q0 q_bits | {fp16, int8, int4} | TVM 上 = dp4a/WMMA tensorize 路径 |
| | 粒度 | Q2 q_granularity | {per-tensor, per-channel} | 影响 scale + AP |
| **S 调度** | MetaSchedule | (tile/loop-order/fusion/tensorize) | 生成式搜索 | **TVM 一等轴** |
| **H 硬件** | device | A1 routing | {4090, Orin-GPU, Orin-DLA} | 改 tile/tensor-core 可用性 |
| | batch | — | {1,2,4} | 吞吐 regime |

> 注:D2 prune_object/D3 criterion/D5 granularity 主要是"怎么剪"而非"剪后形状",对 latency-耦合影响经由 width 体现,先并入 width 轴;Q1 per-module 混精/Q3 object/Q4 calibrator 二期。

## 2. 裁决判据(怎么把一个 cell 归类,TVM 上可测)
**核心 = argmin 漂移 / 交互效应检验**(P×S 通道对齐工作已用的那套,泛化):
- 两轴 (A,B) **耦合** ⟺ `argmin_B`(最优 B 配置)随 A 的取值**改变**(最优重排序)。TVM 测法:在 A 的网格上,各自用 MetaSchedule 真调 B,看赢家是否翻转。
- **量化**:对该子集跑三臂 —— `joint(A,B)` HV vs `serial(A→B)` HV(只在 {A,B} 上)。gap 大 → JOINT;gap≈0 → SERIAL/SINGLE。
- 4 类裁决:
  - **JOINT**:argmin 漂移 + serial HV 显著 < joint(如通道对齐 rank-flip)。
  - **SERIAL**:argmin 不漂移但两轴都影响 HV → 先搜 A 锁定再搜 B 不丢最优(标推荐顺序)。
  - **SINGLE**:B 几乎不动 HV(一个维度主导)。
  - **PAIR**:三轴里某两轴 JOINT、第三轴可独立加(空间部分因子分解)。
- **铁律**:延迟全程 TVM/同硬件;int8 必 dump CUDA grep dp4a/wmma + 数值校验(承袭 [[feedback-no-premature-impossible]]);fresh workdir 逐宽度。

## 3. 已知 cells(真测填写)
> ★地图是 **architecture-conditional** 的 —— grouped(Pyramid)与 standard(CoDriving)给两张不同的图。这正是"可从架构预测联合搜必要性"的判据。

| 维度对 | Pyramid(grouped conv, ResNeXt g32/wpg4) | CoDriving(standard conv, g1) |
|---|---|---|
| **P-width × S** | **JOINT** ✅ —— argmin schedule 随对齐漂移,rank-flip(default-fast 失配宽 vs tuned-fast 对齐宽),iso-AP 延迟比 3.83×(`b4_ablation_results.json`) | **SERIAL/SINGLE** ✅ —— 无 rank-flip,标准 conv 任意宽度都良好 tile(`codriving_s0probe_fp16.csv`) |
| **P-width × Q(int8 build)** | **JOINT** ✅(定性/categorical)—— s0=48(in_per_g=3)NCHWc IC_BN=4 **结构性建不出 int8**,只 s0=64 可(`q_int8_dp4a_pairs.csv`) | **SERIAL** ✅ —— Cin=48/64 **都 build int8**(1.42×/1.32×),软惩罚非定性(`codriving_int8_verify.json`) |
| **P×Q×S 三轴** | **JOINT(全)** ✅ —— A-joint 100%/A-serial 85.3%, Wilcoxon p=4.88e-4(`pqs_ablation_results.json`);★**这是"三轴经由 P(IC_BN)枢纽耦合"**,P 是 hub,**非三维独立不可约**(见下勘误) | **SERIAL** ✅ —— C0c' 三臂 A-joint=A-serial=100%(`C0c_codriving_pqs.json`) |
| **Q × S** (固定 P=base) | **P-mediated(非独立于P)** ⚠️ —— 原报 "FP16-MS 0/100 valid → Q×S 不经P耦合" **已证伪**:FP16 **也走 WMMA m16n16k16(half)**,gain 8.59×(g8)/1.64×(g32);原 0-valid = DB 计数 bug + 硬编码 literal(见 mech3 勘误)。INT8 相对 FP16 的 MS 优势(1.37×@IC_BN16 → 1.04×@IC_BN4)随 IC_BN 变 = **经 P-hub**(`mech3_fp16_verify.json`) | **NO_WIDTH_SPECIFIC_TRAP** ✅ —— p25"软陷阱"=cast-artifact 假阳性(`C1_QxS_codriving.json`) |
| **wpg × Q(INT8 buildability)** | **FORMAT 障壁,延迟中性** ⚠️ —— IC_BN<4 → NCHWc dp4a 路径封锁(groups=64 FAIL);★C3 best-vs-best(全 tuned): misaligned IC=96 上 FP16-tuned(139.29µs)≈ INT8-NCHW(140.92µs)打平,**都快过** padded-128-NCHWc-tuned(149.45µs)→ padding 修对齐不划算, mech1 是 format gate **不是延迟代价**(`C2_C3_pyramid.json`) | N/A (标准 conv groups=1, IC_BN=Cin, 不受此限) |
| **wpg × Q × S** (IC_BN gain scaling) | **JOINT** ✅ **核心真机制** —— IC_BN 控制 MS gain 量级(INT8 11.75×/4.59×/1.70×;FP16 8.59×/—/1.64×);C2 确认 argmin_S rank-flip(native-WMMA@IC_BN16 → padded-WMMA@IC_BN4)。**经 P(IC_BN)hub**(`C2_C3_pyramid.json`) | N/A |

**★[2026-06-22 勘误 — 复核后撤回"三维独立不可约",改判 P-hub] (trackA+main 双核验)**

原 §3 声称 "P×Q×S 三维不可约,3 条独立机制(含 Q×S 不经 P)" **已部分撤回**。真测复核:

| 机制 | 复核后状态 | 是否经 P |
|------|-----------|---------|
| mech1 (P×Q, IC_BN≥4 NCHWc dp4a) | **FORMAT 障壁,延迟中性** —— NCHW int8 fallback 吸收,padding 不划算(C3 同-INT8-图) | 经 P |
| mech2 (wpg×Q×S, IC_BN→MS gain scaling) | **REAL,核心** —— C2 rank-flip 确认 | 经 P(IC_BN) |
| mech3 (Q×S 独立,不经 P) | **FALSE = BACKEND_ARTIFACT** —— FP16 也用 WMMA half;原"0 valid"是 `count_db_valid()` 解析 bug + `write_c7_verdict.py:44` 硬编码 literal(g8 从未测) | — |

⇒ **唯一存活的真耦合(mech2)是 P-mediated**;mech1 软可消除;mech3 不存在。**没有"不经 P 的不可约耦合"的干净证据。** 诚实图谱 = **P(IC_BN) 是 hub 的 P×(Q+S) 联合依赖**(框架仍须联合搜,但因为 P 是 hub 牵动 Q、S,而非三维彼此独立纠缠)。证据:`mech3_fp16_verify.json` + `C2_C3_pyramid.json` + C7 json 顶层 `mech3_verdict/claim_downgraded`。

## 4. cells 状态总览(按 §0 目标排序)
| # | cell | 状态 | 判决 | 文件 |
|---|---|---|---|---|
| **C1** | **Q × S**(固定 P=base) | ✅ **完成(复核改判)** | **P-mediated** — 原"FP16-MS 0% gain"证伪(FP16 也 WMMA);Q×S 优势随 IC_BN 变=经 P | `C1_QxS_pyramid_final.json` + `mech3_fp16_verify.json` |
| **C7** | **wpg×Q×S**(原称三维不可约) | ✅ **完成(降级)** | **P-hub,非三维独立不可约** — mech1 软/mech2 真核心/mech3 假(后端 artifact) | `C7_pyramid_irreducible.json` |
| **C4** | **Q-granularity × P** (V2X-ViT) | ✅ **完成** | **JOINT** — per-channel vs per-tensor: AP Δ+0.0085 @ p75(static,confound 已排除) | `C4_QgranxP_v2xvit.json` |
| **C5** | **Q routing × stage** (V2X-ViT) | ✅ **完成** | **SERIAL/PAIR** — grid_sample/einsum 触发 DLA blacklist,仅 ~4% 可路由 | `C5_routing_v2xvit.json` |
| **C6** | **CoDriving 高维陷阱** | ✅ **完成(阴性)** | **NO_ROBUST_HIGHDIM_TRAP** — batch×width "rank-flip" 是 1.2% 噪声级近平局,跟 size 不跟对齐 | `C6_codriving_highdim.json` |
| **C0c'** | **CoDriving P×Q×S 三臂** | ✅ **完成** | **SERIAL** — A-joint=A-serial=100% HV,0 rank-flip | `C0c_codriving_pqs.json` |
| **C2** | **wpg × S** | ✅ **完成** | **CONFIRMED rank-flip(P-hub)** — argmin_S native-WMMA(IC_BN16)→padded-WMMA(IC_BN4);CUDA reindex_shared_ vs reindex_pad_shared_ | `C2_C3_pyramid.json` |
| **C3** | **round_to × P** (对齐消除) | ✅ **完成(best-vs-best)** | **mech1=FORMAT 障壁,延迟中性** — IC=96 上 FP16-tuned(139.29)≈INT8-NCHW(140.92)>padded-INT8(149.45),padding 不划算;INT8 优势仅 vs 未调优 FP16 | `C2_C3_pyramid.json` + `trap25_fp16_tuned.json` |
| **C8** | **device/batch × {P,Q,S}** | ⏳ 二期 | DLA routing × op-blacklist;batch 吞吐 regime | — |

## 5. 进度与下一步

### ★目标① 状态: **未达成(诚实) —— P-hub,非真三维不可约** (2026-06-22 复核)
原 "已达成" 判断 **撤回**。复核后:存在的真耦合(mech2/C2 wpg×Q×S gain scaling)**经 P(IC_BN)hub**;mech1 是 **format 障壁且延迟中性**(C3 同-INT8-图:NCHW fallback 反而最快,padding 不划算);**唯一声称"不经 P"的 mech3 是后端 artifact**(FP16 也走 WMMA,原"0 valid"是解析 bug+硬编码 literal)。
- ⇒ **"不经 P 的不可约耦合"无干净证据**。可辩护 claim = **P(IC_BN)-hub 的 P×(Q+S) 联合依赖**(框架仍须联合搜:P 是 hub 牵动 Q、S)。
- 若要真三维不可约,需找一个 P 固定下 Q、S 仍纠缠的**算法本征**机制(非后端规则覆盖);当前未找到。

### ★目标② 状态: **已闭合(阴性)** (2026-06-22)
CoDriving 标准 conv(groups=1)低维(C0c' SERIAL)+ 宽度(C1cod 无陷阱)+ 高维 batch×width(C6 无稳健陷阱)三者一致 ⇒ **耦合是 grouped-conv(Pyramid)特有,非普适**。"找了但没找到"=干净阴性结论。

### 剩余工作
- [ ] 补 trap25-FP16-tuned 闭合 C3 net-speedup 量级(trackA 测量中)
- [ ] 汇总矩阵 `results/coupling_map_matrix.json`(已建)+ doc4 §9 主叙事重构

## 6. 与既有结论的关系(诚实)
- 旧"三轴强耦合 ✅"= (P×S)+(P×Q)+(P×Q×S) 三个 JOINT cell,**经由 P 单一枢纽**,int8 延迟用 1.449 uniform proxy(抹掉 Q×S)。复核后确认:**这是 P-hub 耦合,不是"真三维不可约"**;原以为 C7 补上了"不经P"那条腿,但 mech3 复核证伪 → **那条腿不存在**。
- 旧"CoDriving 可分离 ✅"**现已是定论**:高维(C6)搜过仍无陷阱,符合 §0② "高维搜够仍无 ⇒ 可分离"。
- 旧 TRT 结论(per-stage 混精被 auto 支配 / 2:4 弱 / D 维塌缩)= TRT 后端 cell,与本 TVM 地图分列,不混。
- ★方法学:本轮三处假阳性被复核拦下 —— (1) C1cod p25 软陷阱=cast-artifact;(2) C6 batch rank-flip=噪声;(3) mech3 Q×S 不经P=DB-count bug+硬编码。共同教训:**int8-vs-fp16 绝对速度不是耦合判据(cast-chain artifact);后端规则覆盖缺失 ≠ 算法本征不可约**。
