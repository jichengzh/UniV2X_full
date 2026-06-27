# HANDOFF — Stage2 实验补充指南：AAAI 审稿视角 v1

日期: 2026-06-25

本文替换原先“实验收口与验证”版本。原文偏工程交接，重点是 smoke、demo、脚本和已有数字复核；这不足以支撑 AAAI 论文。本文从审稿人的角度回答一个更核心的问题:

```text
还需要补哪些实验，才能证明本文提出的软硬件协同优化框架确实有效、
能显著提升推理速度，并且相比单轴/串行优化更优？
```

本文参考:

- `multi_agent/references/papers/ALT_2210.12415.pdf`
- `multi_agent/references/papers/CHaNAS.pdf`
- `multi_agent/references/study_joint_search_methods_v1.md`
- `multi_agent/methods/design/auto-tuning/1_design_space_building_v1.md`
- `multi_agent/methods/design/auto-tuning/2_design_cost_model_v1.md`
- `multi_agent/methods/design/auto-tuning/4_design_ablation_proof_v1.md`
- `multi_agent/methods/progress/HANDOFF_stage2_gap_search_space_v1_zh.md`
- `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v1_zh.md`
- `multi_agent/methods/progress/HANDOFF_stage2_gap_stage1_stage2_integration_v1_zh.md`

---

## 0. 审稿人会先问什么

如果论文主张是“一个面向 V2X 模型的软硬件协同优化框架”，AAAI 审稿人不会只接受一个模型上的局部加速或一个工程 smoke。审稿人会要求看到四条证据链:

1. **有效性**: 在真实模型、真实 checkpoint、真实硬件上，框架能在基本保持 AP/精度的前提下显著降低推理延迟。
2. **协同必要性**: 联合搜索不是把单轴优化结果重新包装；它要稳定优于强串行 baseline、P-only、Q-only、S-only 或 P/Q 后再调 schedule 的流程。
3. **可解释性**: 为什么某些模型需要协同，某些模型可以串行？论文要证明这是结构和硬件交互造成的，而不是 cherry-pick。
4. **可推广性与边界**: 结论至少跨模型类别或硬件设置成立；不成立的模型也要解释清楚，并由框架自动给出 conservative verdict。

因此，下一阶段实验不应围绕“如何写 smoke 脚本”，而应围绕以下科学命题组织:

```text
Claim A: Stage2 可以显著降低 V2X 模型推理延迟，同时保持 AP。
Claim B: 对存在 P(IC_BN)-hub 或 schedule-sensitive 结构的模型，joint search 明显优于 serial/single-axis search。
Claim C: 对可分离模型，框架能识别 serial 已足够，而不是盲目声称 joint 永远更好。
Claim D: Stage1 scan + Stage2 search + evidence/cost model 的组合能在新模型上给出可审计、保守的优化建议。
```

---

## 1. 从 ALT 和 CHaNAS 借鉴的实验范式

### 1.1 ALT 的启发

ALT 的实验不是只报告“某个 kernel 快了”。它做了三类关键证据:

- **主结果**: 相对 Ansor / 现有编译器，在 single-operator 和 end-to-end inference 上报告平均加速。
- **消融**: ALT-OL、ALT-WP 等变体，证明 layout 与 loop 的 joint tuning 有独立收益，不只是 loop tuning 或去掉 conversion overhead 的副产物。
- **机制实验**: 串联 C2D 的 ALT-FP/BP micro-benchmark，证明单向串行传 layout 会让后续算子陷入次优，joint/cross-exploration 才能找到更优解。

对应到本项目，不能只展示 Pyramid 的一个 faster point。必须展示:

- joint search 相对 serial/noS 的真实 Pareto/HV 优势。
- 这个优势来自 `prune width -> schedule/tensorization space` 的耦合，而不是实验者事后枚举出一个快点。
- 搜索过程本身访问不到/访问得到哪些分支，能够解释 serial 为什么被锁死。

### 1.2 CHaNAS 的启发

CHaNAS 的实验设计更贴近本项目。它证明“架构 × 调度”协同有效，主要靠:

- **CHaNAS-W vs CHaNAS-W/O**: 同一超网，同精度下，有调度协同的解明显更快。
- **多硬件 Pareto**: P100、Xeon、Note10 多后端上展示 Pareto frontier。
- **Fig.2 风格 motivation**: 不同 schedule 下 Pareto frontier 交叉，说明最优架构依赖调度，固定 schedule 的 NAS 会错过最优。
- **block LUT + 搜索成本**: 说明联合搜索不是不可承受的指数爆炸，而是通过 block-level pre-scheduling 和 LUT 变成可行过程。

对应到本项目，论文实验应同时有:

- iso-AP latency speedup，作为最容易读的 headline。
- 多目标 Pareto / hypervolume，作为严谨度量。
- joint / serial / noS 的公平消融。
- 搜索成本和测量成本，说明方法不是“用无限实验换加速”。
- 至少一个阴性模型，证明框架不是无条件宣称 joint 更好。

---

## 2. 当前已有证据的审稿风险

### 2.1 已有强证据

Pyramid P×S 是当前最强证据:

- `A-joint` mean HV: `6984.217`，约 `99.90% ref`。
- `A-serial` mean HV: `5978.342`，约 `85.51% ref`。
- `A-noS` mean HV: `3336.904`，约 `47.73% ref`。
- Wilcoxon `p=0.00048828125`，rank-biserial `1.0`。
- shipped win: `[48,128,128] -> [64,128,128]`，AP70 `0.6369`，latency `21071.43us -> 5506.57us`，`3.827x`。

这可以支撑:

```text
Pyramid grouped conv 存在 P(IC_BN)-hub 型宽度-调度耦合；
在这个模型和 H800 TVM scope 下，joint 明显优于 serial。
```

CoDriving P×Q×S 是当前阴性证据:

- `A-joint = A-serial = 100%`。
- `A-noS = 97.07%`。
- rank flip = `0`。
- all widths INT8 buildable。

这可以支撑:

```text
标准卷积 dense envelope 中，当前 CoDriving 的 P/Q/S 低维空间基本可分离；
框架可以输出 serial 足够，而不是所有模型都强推 joint。
```

### 2.2 还不足以支撑 AAAI 的地方

从审稿视角，当前证据仍有明显缺口:

1. **模型数量不足**: Stage2 真正完成加速实验的主要是 Pyramid 和 CoDriving。两模型可以形成正例/阴性对照，但还不足以证明“框架”泛化。
2. **full-model 证据不足**: 许多结果是 dense-core / backbone / stage0，不能直接写成 full e2e speedup。
3. **P×Q×S 口径未收口**: 当前 `results/pqs_ablation_results.json` 是 4 seed，旧文档曾写 12 seed。论文表格不能混用。
4. **Q 轴证据边界不稳**: H800 TVM stage0 INT8、TRT historical AP、proxy QLookup 不能混成同一个“INT8 joint result”。
5. **cost model / evidence registry 还没成为论文级实验基础设施**: latency LUT、AP anchors、energy、DS 的 schema、coverage、provenance 和 uncertainty 还在补。
6. **跨硬件实测不足**: 目前可以做 static capability 判断，但新硬件没有 measured latency LUT 时不能声称加速结论。
7. **搜索成本没有系统报告**: 需要说明比 exhaustive、省多少测量、比 serial/random/单轴多花多少成本，是否值得。

---

## 3. 必补实验总表

下面是建议的 AAAI 实验补充矩阵。P0 是论文主线必须补，P1 是强烈建议，P2 是有空间再补。

| 编号 | 实验 | 证明什么 | 优先级 |
|---|---|---|---|
| E1 | 主结果: end-to-end / dense-core 加速与 AP 保持 | 框架真的提升推理速度，且没有牺牲主要精度 | P0 |
| E2 | joint vs serial/noS/single-axis 公平消融 | 协同优化优于单轴或串行优化 | P0 |
| E3 | rank-flip / P(IC_BN)-hub 机制实验 | 为什么 joint 会赢，排除 cherry-pick | P0 |
| E4 | 跨模型泛化: 正例、阴性、边界模型 | 框架能判断何时 joint 必要、何时 serial 足够 | P0 |
| E5 | 搜索效率和测量成本 | joint 不是靠不可承受的测量预算赢 | P0 |
| E6 | cost model / LUT / AP anchor 质量 | 搜索输入可靠，预测/查表不会误导结论 | P1 |
| E7 | 跨硬件实验 | 软硬件协同不是只对 H800 单点成立 | P1 |
| E8 | full-model coverage / Amdahl 分解 | dense-core 加速能兑现到 e2e 的比例，防止过度声明 | P1 |
| E9 | energy / throughput / power | 证明硬件效率，而不只是 latency | P1 |
| E10 | downstream DS / closed-loop top-K 复核 | 若论文强调 V2X driving value，需要证明加速不会破坏闭环表现 | P2 |
| E11 | ablation of Stage1 classifier gate | 证明 Stage1 分类器对 Stage2 策略有用 | P1 |
| E12 | 与外部 baseline 对比 | 证明不是只比弱 baseline | P0 |

---

## 4. P0 实验设计

### E1. 主结果: 加速与 AP 保持

**审稿问题**

```text
你的框架最终能让真实 V2X 模型更快吗？精度是否保持？
```

**实验对象**

至少包含:

- Pyramid: 当前正例，必须保留。
- CoDriving: 当前阴性/可分离对照，必须保留。
- 至少再补 1 个 trained checkpoint 模型，优先 F-Cooper 或 AttFuse。
- V2X-ViT 可作为 attention/fusion 边界模型，但若 full pipeline 未闭合，应放在 limitation / case study，不要作为主加速表唯一新增模型。

**对照组**

每个模型至少比较:

- Original / baseline model。
- P-only: 只剪枝，default schedule。
- Q-only: 只量化或 mixed precision，若 Q evidence 不足则标 blocked。
- S-only: 不改模型，只调 TVM schedule。
- Serial: 先 P/Q，再 S。
- Joint: P/Q/S joint 或 P/S joint。

**指标**

主指标:

- AP70 / AP50，按模型任务选择并说明。
- latency p50 / p90。
- iso-AP latency speedup。
- Pareto frontier / hypervolume。

辅指标:

- throughput FPS。
- model size。
- build success rate。
- energy，若 evidence 可用。

**需要的技术**

- Stage1 manifest 和 `Stage2SearchSpace`。
- latency LUT: H800 TVM measured。
- AP anchors: finetuned / imported measured AP。
- Stage2 search driver。
- full-model 或 dense-core latency runner。

**当前技术状态**

- Pyramid P×S 已有强证据。
- CoDriving dense envelope 已有阴性结论。
- `Stage2SearchSpace` 已在 `framework/stage1_bridge.py` 增加 `load_stage2_search_space(path)` / `stage2_search_space()`，并有 `test_stage2_search_space_contract.py`。
- latency/AP evidence registry 还没有完全产品化。

**缺口**

- 新增 F-Cooper / AttFuse 的 Stage2 latency LUT 与 AP anchors。
- full-model e2e runner 或 Amdahl 分解。
- P×Q×S 统一 seed 与 backend/scope。
- 每个结果需要明确 `optimized_scope`，不能把 dense-core 写成 full e2e。

---

### E2. joint vs serial/noS/single-axis 公平消融

**审稿问题**

```text
joint search 是否真的比强串行 baseline 更好？还是只是比弱 baseline 好？
```

**实验设计**

用同一 search space、同一 evidence、同一 total budget，比较:

| Arm | 含义 | 对应文献启发 |
|---|---|---|
| `A-noS` | 消掉 schedule 轴，只用 default schedule | CHaNAS-W/O / ALT-OL |
| `A-serial` | 先锁 P/Q，再调 schedule | 经典串行流水 |
| `A-joint` | P/Q/S 同搜或 P/S 同搜 | CHaNAS-W / ALT joint |
| `P-only` | 只调剪枝宽度 | 单轴 baseline |
| `Q-only` | 只调量化策略 | 单轴 baseline |
| `S-only` | 只调 schedule | 编译器 baseline |
| `Random-joint` | 同空间随机搜索 | 排除搜索器弱基线 |

P/Q 证据不足时，主线先用 P/S，Q 放 P1 或 marked unavailable。

**统计要求**

- 至少 10-12 seeds，推荐 12。
- 同 seed 配对比较。
- Wilcoxon signed-rank。
- rank-biserial effect size。
- HV distribution boxplot。
- convergence curve。

**需要的技术**

- `framework/search_three_arm.py` 的统一三臂搜索。
- 固定 reference point 的 HV 计算。
- search trace logging: 每个 arm 访问过哪些 candidate。
- seed/budget/pop 统一配置。

**当前技术状态**

- Pyramid P×S 已有 12 seed 结果。
- Pyramid P×Q×S 当前落盘是 4 seed，需要收口。
- CoDriving P×Q×S 已有 serial=joint 阴性结果。

**缺口**

- `P-only/Q-only/S-only/Random-joint` 还没有统一放进论文表格。
- P×Q×S 需要重跑或找回 12 seed artifact。
- 搜索 trace 需要可导出，支撑 E3 的机制图。

---

### E3. rank-flip / P(IC_BN)-hub 机制实验

**审稿问题**

```text
joint 为什么赢？是否只是你挑了一个特殊点？
```

**实验设计**

对 Pyramid 这类 grouped conv 模型，做三层证据:

1. **rank-flip 表**: 列出 W_g 与 P_g。W_g 在 default schedule 下看起来更优，但 tuned 后被 P_g 支配。
2. **search trace 点云**: 画出 A-serial 和 A-joint 实际访问的点。证明 serial 不是慢，而是结构上排除了 P_g 分支。
3. **IC_BN / schedule gain 关系图**: x 轴为 `IC_BN` 或 width alignment，y 轴为 default latency、tuned latency、tuning gain。

**对照**

- Pyramid grouped conv: 应出现 rank flip。
- CoDriving standard conv: 应无 rank flip 或显著更弱。
- 可选 F-Cooper / AttFuse: 看其结构落在哪一类。

**需要的技术**

- Stage2SearchSpace 中的 width anchor、diagnostic trap anchor。
- latency LUT 同时包含 default/tuned。
- 每个 candidate 的 `IC_BN`、groups、width、schedule gain metadata。
- search trace exporter。

**当前技术状态**

- Pyramid 已有 shipped pair `[48,128,128] -> [64,128,128]`，可作为主机制例。
- Search space 文档已要求 width candidate 从真实层结构和硬件合法性生成，而不是固定 25/50/75。

**缺口**

- 需要把机制图从 handoff 数字整理成论文图。
- 需要 CoDriving 的对应阴性机制图。
- 需要自动生成 rank-flip table，不靠手工挑点。

---

### E4. 跨模型泛化

**审稿问题**

```text
这是不是只对 Pyramid 有效？框架能否处理新模型？
```

**建议模型分组**

| 类别 | 模型 | 目的 | 当前状态 |
|---|---|---|---|
| grouped/dense positive | Pyramid lidar/camera | 证明 joint 必要 | Stage2 证据最强 |
| standard dense negative | CoDriving | 证明 serial 足够时框架能识别 | 已有阴性结果 |
| fusion/attention boundary | AttFuse / V2X-ViT | 证明边界和 blocker 可审计 | Stage1 有扫描/分类，Stage2 证据不足 |
| cooperative baseline | F-Cooper | 增加 V2X 模型覆盖 | ckpt ok，但 Stage2 LUT/AP 待补 |
| missing ckpt | Where2comm / V2VNet / DiscoNet | 证明 fail-closed | 不能作为 checkpoint 优化主实验 |

**最低论文要求**

P0 目标应至少达到:

```text
2 个完整 Stage2 模型 + 1 个新增 checkpoint 模型的部分 Stage2 结果 + 3 个 scan-failed/fail-closed 案例
```

更理想:

```text
Pyramid + CoDriving + F-Cooper + AttFuse
```

其中 Pyramid/CoDriving 给完整三臂，F-Cooper/AttFuse 至少给 Stage1->Stage2 search space + latency/AP anchors + top-K validation。

**需要的技术**

- 新模型 Stage1 manifest。
- Stage2SearchSpace 自动生成。
- latency LUT builder / importer。
- AP anchors builder / importer。
- classifier gate。

**当前技术状态**

- Stage1 classifier 已能输出 9 模型三分类。
- F-Cooper、AttFuse ckpt ok。
- Where2comm / V2VNet / DiscoNet 缺 ckpt，正确状态是 `SCAN_FAILED` 或 architecture-only。

**缺口**

- F-Cooper/AttFuse 缺 Stage2 measured latency LUT。
- 缺 AP anchors。
- attention/fusion 子图的 trace/TVM runner 不完整。

---

### E5. 搜索效率与测量成本

**审稿问题**

```text
joint search 是否只是因为用了更多测量预算？实际是否可承受？
```

**实验设计**

报告:

- 每个 arm 的 total evaluated candidates。
- TVM/MetaSchedule trials。
- latency LUT cells。
- AP anchor count。
- wall-clock time。
- GPU-hours。
- final HV vs budget curve。

对照:

- exhaustive grid，若空间很小可跑；若空间大，报告 estimated exhaustive size。
- random search。
- serial greedy。
- joint with LUT。
- joint without LUT，若不可行则报告理论成本。

**需要的技术**

- evidence registry 记录 measured cells / failed cells / proxy cells。
- Stage2 search logs。
- LUT coverage stats。
- wall-clock and hardware metadata。

**当前技术状态**

- cost model 文档已提出 smoke/calibration/paper 三档 LUT 规模。
- 但 registry coverage 还未成为统一产物。

**缺口**

- 缺搜索成本表。
- 缺 measurement budget fairness 检查。
- 缺 “joint 的额外成本 vs speedup 收益” 结论。

---

### E12. 外部 baseline 对比

**审稿问题**

```text
你只比了自己设计的 baseline，是否足够？
```

**建议 baseline**

至少需要以下几类:

- PyTorch eager baseline。
- TVM default / DLight baseline。
- TVM MetaSchedule S-only。
- prune-only baseline。
- quant-only baseline，若 evidence 足够。
- serial pipeline: prune/quant 后再 schedule。
- random search within same space。

若能实现，可加:

- 手工工程优化 baseline。
- existing pruning method + TVM tune。
- TensorRT historical 只作 context，不作为 H800 TVM 新实测主 baseline。

**需要的技术**

- baseline runner。
- same input shape / batch / hardware / repeat policy。
- AP evaluation protocol。

**缺口**

- 当前 TRT 只能 historical，不能混入 H800 TVM 主实验。
- pruning method baseline 若引入外部算法，需要统一 finetune budget。

---

## 5. P1/P2 实验设计

### E6. Cost model / LUT / AP anchor 质量

**审稿问题**

```text
你的搜索依赖 LUT 和 AP anchors，它们准确吗？能泛化到未测候选吗？
```

**实验设计**

- latency LUT holdout: measured vs predicted / interpolated。
- AP anchor holdout: predicted / imported AP vs measured AP。
- ranking metric: Spearman、Kendall tau、top-k recall。
- uncertainty calibration: predicted interval 是否覆盖真实值。
- sensitivity: LUT cells 从 20/60/120/300 增加时，HV 和推荐配置是否稳定。

**需要的技术**

- Stage2EvidenceRegistry。
- latency/AP schema。
- rank/pairwise cost model 或 LUT interpolation。
- AP evaluation protocol。

**当前状态**

- `LatencyLUT`、`APModel`、`QLookup` 已存在。
- cost model handoff 已加入 energy、DS 和 LUT 数据规模计划。

**缺口**

- registry 未落地。
- AP anchors 产品化未完成。
- latency LUT builder 未产品化。

---

### E7. 跨硬件实验

**审稿问题**

```text
软硬件协同是否真的依赖硬件？换硬件后框架还能工作吗？
```

**实验设计**

至少增加一个 secondary hardware:

- RTX 3090 / 4090 / Orin 任选其一，优先实际可接入的设备。
- 对 Pyramid 与 CoDriving 各跑小规模 latency LUT。
- 比较 search policy 是否变化、top config 是否变化、joint/serial verdict 是否稳定或合理改变。

**指标**

- latency speedup。
- rank correlation between H800 and new hardware。
- top-k overlap。
- hardware-specific blocker。

**需要的技术**

- hardware YAML。
- hardware probe。
- measured latency LUT。
- backend/scope 记录。

**当前状态**

- static capability YAML 可以做合法性判断。
- 但没有实机测量时不能输出新硬件加速结论。

**缺口**

- 3090/4090/Orin measured LUT。
- new hardware evidence registry。

---

### E8. full-model coverage / Amdahl 分解

**审稿问题**

```text
dense-core 快了，完整模型真的快吗？
```

**实验设计**

对 Pyramid 和 CoDriving 做 full pipeline profiling:

- preprocessing。
- dense core / backbone。
- fusion / routing / attention。
- decode / postprocess。
- data movement / copy。
- launch overhead。

然后报告:

```text
dense-core speedup
full e2e speedup
theoretical Amdahl upper bound
unoptimized bottleneck
```

**需要的技术**

- full-model profiler。
- CUDA event / wall-clock breakdown。
- Stage1 skipped_subgraphs manifest。
- optimized_scope claim boundary。

**当前状态**

- CoDriving 已知 full pipeline 受 eager/decode/fusion/launch/copy 稀释。
- Stage1 manifest 能记录 skipped subgraphs。

**缺口**

- Pyramid/CoDriving 统一 full e2e breakdown。
- fusion/attention 子图单独 LUT 或 blocker accounting。

---

### E9. energy / throughput / power

**审稿问题**

```text
你说硬件协同，只看 latency 是否够？能耗是否恶化？
```

**实验设计**

对最终推荐 top-K 和 baseline 测:

- joule per inference。
- average power。
- peak power。
- latency-energy Pareto。
- throughput under batch=1 或实际 batch。

**需要的技术**

- energy LUT 或同步 power telemetry。
- 与 latency LUT 共用 `config_id`。
- idle baseline policy。

**当前状态**

- cost model handoff 已把 energy LUT 列为空白。

**缺口**

- energy measurement pipeline 未产品化。
- 没有 energy evidence 时不能写 energy improvement。

---

### E10. downstream DS / closed-loop top-K 复核

**审稿问题**

```text
V2X 加速是否会影响下游驾驶表现？
```

**实验设计**

不要把 DS 作为主 Pareto 轴，除非 evidence 足够。建议:

1. 用现有 CoDriving AP×latency->DS map 做 report-only ranking。
2. 选 top-K 推荐配置做真实 closed-loop rerun。
3. 报告 DS、route completion、collision。

**需要的技术**

- `multi_agent/real_test/ds_ap_latency_all_measured.csv`。
- DS query wrapper。
- top-K closed-loop runner。
- uncertainty / cliff-band 标注。

**当前状态**

- 已有 CoDriving/Town05/clean6 65-cell DS map。

**缺口**

- 只能用于 CoDriving/Town05/clean6 scope。
- predicted AP + predicted latency -> DS 默认 report-only。
- Pyramid 或其他模型不能外推。

---

### E11. Stage1 classifier gate 的有效性

**审稿问题**

```text
Stage1 分类器是否真的帮助 Stage2 决策，还是只是事后标签？
```

**实验设计**

比较:

- 无 gate: 所有模型都跑 joint。
- classifier gate: `SCAN_FAILED` fail closed，`SEPARABLE` serial/low budget，`CO_ACCELERATION_REQUIRED` joint。
- oracle gate: 用完整三臂结果事后给最优策略。

指标:

- wasted search budget。
- invalid optimization attempts。
- achieved HV / speedup。
- false joint / false serial。

**需要的技术**

- `model_classifier` output。
- `Stage2SearchSpace.model_search_policy`。
- Stage2Input/Output contract。

**当前状态**

- classifier 已有三分类。
- 集成 handoff 指出 runtime gate 还未正式接入。

**缺口**

- `scripts/stage2_optimize_model.py` 统一入口未完成。
- classifier gate fixture 未完成。

---

## 6. 论文 Experiment 部分建议结构

建议按以下结构写，而不是按脚本或阶段编号写。

### 6.1 Experimental Setup

写清:

- Models: Pyramid、CoDriving、F-Cooper/AttFuse 等。
- Dataset / metrics: DAIR-V2X 或对应数据集，AP70/AP50。
- Hardware: H800 TVM/Relax/MetaSchedule 主后端；其他硬件如有实测再列。
- Search space: P/Q/S 定义、dense-core scope、full-model caveat。
- Measurement protocol: warmup、repeat、p50/p90、AP finetune protocol。

### 6.2 Main Results: Accuracy-Preserving Acceleration

主表:

| Model | Scope | Method | AP | latency p50 | speedup | backend | evidence |
|---|---|---|---:|---:|---:|---|---|

写法:

- 先给 iso-AP latency speedup。
- 再给 Pareto/HV。
- dense-core 与 full e2e 分开列。

### 6.3 Does Co-Design Beat Serial Optimization?

核心消融表:

| Model | A-noS HV | A-serial HV | A-joint HV | joint/serial | p-value | verdict |
|---|---:|---:|---:|---:|---:|---|

配图:

- Pareto frontier。
- HV boxplot。
- convergence curve。

这是论文最重要一节。

### 6.4 Why Does Joint Search Help?

机制节:

- Pyramid rank-flip table。
- IC_BN vs schedule gain。
- A-serial / A-joint search trace point cloud。
- CoDriving negative mechanism: no rank-flip, all widths buildable。

这节对应 ALT micro-benchmark 和 CHaNAS Fig.2 的角色。

### 6.5 Generalization Across Models and Hardware

写:

- 正例、阴性、边界模型。
- 新模型 Stage1->Stage2 的自动 pipeline。
- 如果有第二硬件，列 top-K overlap / policy change。
- 如果没有第二硬件实测，只能写 limitation，不能包装成结论。

### 6.6 Search Efficiency and Cost Model Quality

写:

- search budget vs HV。
- LUT size / measured cells。
- random/exhaustive/serial comparison。
- latency/AP ranking accuracy。
- measurement overhead。

这节对应 CHaNAS 的 search cost 和 block LUT 可行性论证。

### 6.7 Optional: Energy and Downstream Driving

只有 evidence 足够时写。

- energy: latency-energy Pareto。
- DS: report-only map + top-K closed-loop rerun。

如果 evidence 不足，这节应放到 limitation，而不是硬写主结果。

### 6.8 Limitations and Scope

必须主动写:

- dense-core 不等于 full-model。
- H800 TVM 是新增实测主后端。
- TRT 是 historical / separate backend。
- missing ckpt 模型不进入 checkpoint optimization。
- attention/fusion full optimization 是后续工作。

---

## 7. 技术依赖与缺口总表

| 实验 | 需要的技术 | 当前进展 | 缺口 |
|---|---|---|---|
| E1 主结果 | Stage2SearchSpace、latency LUT、AP anchors、search driver | Pyramid/CoDriving 局部可用；search space contract 已更新 | 新模型 LUT/AP；full-model runner |
| E2 joint 消融 | A-noS/A-serial/A-joint、HV、seed control | Pyramid P×S 强；CoDriving 阴性可用 | P×Q×S 12 seed 收口；single-axis baselines |
| E3 机制实验 | rank-flip detector、IC_BN metadata、search trace | Pyramid 机制已有数字 | 自动图表；CoDriving 阴性机制图 |
| E4 跨模型 | Stage1 scan/classifier、Stage2 manifest bridge | 9 模型分类已做 | F-Cooper/AttFuse Stage2 measured evidence |
| E5 搜索成本 | registry coverage、wall-clock logging、budget logging | 规模计划已写入 cost model handoff | coverage 统计和 budget fairness test |
| E6 cost model | evidence registry、holdout split、ranking metrics | LatencyLUT/APModel/QLookup 已有 | registry、builder/importer、uncertainty |
| E7 跨硬件 | hardware YAML、hardware probe、measured LUT | static capability 有基础 | 新硬件实机测量 |
| E8 full-model | profiler、Amdahl 分解、skipped subgraph accounting | Stage1 skipped_subgraphs 有基础 | full e2e breakdown 和 fusion/attention accounting |
| E9 energy | energy LUT、power telemetry | 仅设计空白 | measurement pipeline |
| E10 DS | DS LUT、closed-loop runner、uncertainty | CoDriving 65-cell map 已有 | top-K rerun；不可跨模型 |
| E11 gate | model_classifier runtime gate、Stage2 I/O contract | classifier 已有；集成计划已写 | CLI 和 fixture 测试 |
| E12 external baselines | PyTorch/TVM/default/serial/random runners | 部分已有 | 统一 baseline table |

---

## 8. 下一阶段执行优先级

### 第一优先级: 让主 claim 可成立

1. 收口 Pyramid P×Q×S 12 seed 或正式声明只用 P×S 主线。
2. 为 F-Cooper 或 AttFuse 补一条 Stage2 measured evidence 链。
3. 完成 single-axis baselines: P-only、S-only、serial、random。
4. 生成主表: AP-preserving latency speedup + HV。

### 第二优先级: 让“协同必要性”可解释

1. 自动生成 Pyramid rank-flip table。
2. 生成 search trace point cloud。
3. 生成 CoDriving negative mechanism figure。
4. 把 P(IC_BN)-hub 写成机制，不写成手工规则。

### 第三优先级: 让框架而非个例成立

1. 接入 classifier gate。
2. 输出 Stage2SearchSpace + Stage2Output。
3. 补 evidence registry coverage。
4. 新模型 fail-closed 和 checkpoint-ok 的不同路径都要展示。

### 第四优先级: 提升论文完整度

1. energy LUT。
2. full e2e Amdahl。
3. secondary hardware。
4. DS top-K closed-loop rerun。

---

## 9. 本方向 `/goal` 启动指令

```text
/goal 按 AAAI 审稿标准推进 Stage2 实验补充。目标不是写 smoke/demo 脚本，而是补齐论文所需的科学证据链：主结果 AP-preserving acceleration、joint vs serial/noS/single-axis 公平消融、P(IC_BN)-hub/rank-flip 机制图、跨模型泛化、搜索成本、cost model/evidence 质量，以及 full-model/energy/downstream 的边界。优先收口 Pyramid P×Q×S seed 冲突或明确 P×S 为主线；为 F-Cooper 或 AttFuse 补 Stage2 measured latency/AP evidence；增加 P-only/S-only/random/serial baselines；生成论文 Experiment 结构所需的主表、Pareto/HV 图、收敛曲线和机制图。严格约束：H800 TVM 是新增实测主后端；TRT 只能 historical/separate backend；dense-core 不外推 full-model；proxy/demo 不进入论文正式结果；没有 energy/DS 实测时不输出对应 claim。
```
