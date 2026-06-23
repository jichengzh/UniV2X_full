# 激活 AP 故事 — 策略与 ROI 对比 (v1, 2026-06-03)

> ## ★★ 最终结论 (2026-06-03, 用户拍板 + supervisor 独立核验 ISS-018) — 本文降为过程记录
> **A/B/E 三路免训实证均为负结果: Pyramid/DAIR 上 AP 无可提取成本-trade-off。**
> - **A**: forced-INT8 是 Pareto 被支配点(auto 双轴碾压 ISS-013); cliff/backbone 禁拼曲线; 4dp-vs-2dp 假象勿读"INT8 提升 AP"。
> - **B**: head-INT8 Δap70 -0.0046 **不崩** → 护栏-规避叙事**不能在 Pyramid conv head 声称**(崩在 transformer/MSDA)。
> - **E**: 距离难度 spread 0.147→0.074 **反放大**(floor, n_gt 3172), 按 ISS-015 = **非 trade-off**(smoke 3.6× 是 60 样本假象)。
> **决策(★阶段性, 非论文最终 scope)**: 本阶段 AP 作约束(非目标), 主前沿转 (latency, energy, throughput)+耦合陷阱; **本阶段不训 V2X-ViT, 但 V2X-ViT/换模型 = 下一阶段确定开展、方案待讨论的实验(非取消)**。A/B/E 入表作 ablation/predictor(regime=ablation)。**诚实负结果(加分科学), 非失败。**
> **差异化两支柱(阶段性)**: ①(主, 硬核)成本轴多目标联合 + 耦合陷阱 kernel-cliff(ISS-014); ②(本阶段 = motivation 层)崩溃-规避 = 本阶段引 QuantV2X 文献(V2X-ViT 75.1→29.9)+ Pyramid head-INT8 不崩诚实对比, **本阶段不声称自有模型 demonstrated**; **经 V2X-ViT/transformer 真正 demonstrated = 下阶段计划**。详见宪章 §1 + background §0′ + ISS-018。下文为决策前过程分析, 保留备查。
>
> 编制: data-orchestrator 综合 sw-optimizer 软件侧方案 v0 + 数据/硬件视角。
> 上游: team-lead 战略批准"投入激活 AP 故事"(用户接受扩范围)。
> **流程纪律: 本文是设计/选型, 交 team-lead 审 + supervisor 把关, team-lead 报用户确认后才投训练算力。不直接开训。**
> 关联: `pareto_definition_v1.md §七`(AP 轴近平的已知事实约束)· `dims_pruning_v1.md §8`(剪枝无悬崖)· `dims_quantization_v1.md`(敏感层护栏)。

> ## ★★★ 2026-06-04 Phase M 更新 — 精度目标的转变(换指标实测定论, supervisor ISS-020 终核 PASS)
> 用户提出"AP 信息量低可能是评价指标问题"→ Phase M 用 **NDS TP 误差项(mATE/mASE/mAOE)** 在 4 锚点(base/p25/p50/p75)× {FP16,INT8} × DAIR val 1789 真测验证。结论(经 sw 真测 → data 独立复算 → supervisor 独立复跑 三层核验, 含 ISS-024 权重污染修正 + 共同 GT 子集消幸存者偏差 + 帧级 bootstrap CI):
> 1. **剪枝轴: mAOE 信号成立 ✅** — 严格单调, 共同 GT 子集(固定 2849 框)SNR≈7.5×(>5× 阈值), base↔p75 CI 严格不重叠; 幸存者偏差方向保守(真退化只多不少)。**mAOE 捕捉到 AP70 低估的朝向退化**(剪枝/量化先伤连续回归)。⚠️ 幅度数字以全1789 epoch25 干净版终核为准(在跑)。
> 2. **量化轴: 无信号 ✗** — INT8 单步 ΔmAOE 非单调变号(SNR 0.28–0.9×, 噪声)。**"量化在 Pyramid/DAIR 近无损"是真结论, 不是 mAP 指标盲区造成的假象** —— 换再多精度指标也测不出量化 trade-off。
> 3. **精度目标定位(转变后)**: 精度轴(AP/mAOE)**仍是约束, 不是 Pareto 目标轴**(mAOE 相对范围仍 ≪ 成本轴 5×; ISS-018 维持)。**mAOE 升格为预测器"剪枝退化约束信号"入库**(dataset_v2 新列, 带 mAOE_basis 溯源 + CI; 详见 schema_v2.md)。信号仅在全剪枝程(base↔p75)可分辨, 相邻细档(p25↔p50)CI 重叠。
> 4. **判据纪律(宪章 MUST-NOT-5)**: 信号强弱只认 "Δ vs 自身 bootstrap CI(≥5×)", **禁用 "×AP70" 比值判据**(两噪声相除=假精确, 曾被撤回两次)。
> 5. **路线决策(用户拍板)**: 真正激活精度-成本 trade-off 须**换模型/换数据**。顺序 = **V2X-ViT 先行(Task#8, 已授权)**; UniV2X/V2X-Seq+AMOTA 暂缓(Task#7, 难启动, 待讨论 — 注: 真 AMOTA 只能来自有跟踪头的 UniV2X 家族, PyramidFusion 物理上不产 AMOTA, 见 ISS-026 勘误)。
> 出处: `results/tp_errors_v1.csv` · `tp_errors_ci_v1.csv` · `common_gt_v1.csv` / `common_gt_corrected.csv` · `scripts/phase2/eval_tp_errors*.py` · issues_log ISS-020/024/025/026。

---

## 0. 问题陈述
现状: 整个剪枝+量化空间 **AP70 span 仅 0.138(0.493–0.631), 无悬崖**, AP50 近饱和(span 0.068)。根因 = Pyramid 对 DAIR **严重过参数化**。⇒ selector 实际把 AP 当近常数约束, 真正拉开差距的是成本轴(lat/energy/throughput/size)。
**目标**: 让 AP 出现**可分辨单调下降(信号 ≫ 管线噪声 ~0.001)**, 且最服务"B1×B2×D 联合 + 耦合陷阱规避崩溃" vs QuantV2X 的差异化卖点。
**★两层目标须分清(ISS-013)**: ① **预测器 AP 信号 + 差异化消融** —— Path A/B(拆护栏)即可达, 已验真信号; ② **可部署 Pareto 前沿出现 AP-cost trade-off** —— Path A/B 达不到(forced 是被支配点, 前沿仍是 auto-int8 AP 高原), **只能靠 Path E(难子集)/ C(V2X-ViT)/ 兜底(更难模型)**。本轮先吃 ①(免训、低风险), ② 视 E 结果与用户意愿再投。

**★关键判断(降低决策门槛)**: 下面**首选的 A + B 两条路径都不需要新训练算力** —— 全部用现成 finetuned ckpt + TRT build flag, 本质是 P0 式 build+eval 采集。**训练算力决策只在"A/B 信号不足 → 换更难模型/任务"的兜底才出现。**

---

## 1. 四条候选路径 ROI 对比

| 路径 | 做法 | 训练成本 | 工时(纯计算) | 信号强度 | TRT 可 build / 真测 latency | 对差异化对齐度 | 裁决 |
|---|---|---|---|---|---|---|---|
| **A 剪枝×forced-INT8 深扫** | 3 级 backbone 档(0.364M/0.270M/0.088M)各出 FP16+**forced-INT8** AP, 画 (pyramid_backbone params × 精度) AP 曲面 | **零(免训)** | ~1.5h(3 build+eval) | **强(分段, 已验)**: forced ΔAP70 低剪枝(base/p25)平缓~-0.010, 高剪枝(p50/p75)放大 -0.018/-0.034(10-34×噪声) | ✅ 全 collab2 可 build = 完整点; **但 forced=被支配点(标 ablation, 非前沿)** | ★高(预测器信号 + 护栏消融差异化; **非前沿 trade-off**) | **主选(并行, 仅预测器+消融)** |
| **B 拆护栏强量化敏感层** | 故意把 cls/reg/dir head + encoder(VFE)压 INT8(改 SENSITIVE_FP16_SUBSTR) | 零(免训) | ~同量级 | **强且可控**: encoder INT8 ~0.12 e2e AP 损失, head INT8 score 崩 | ✅ 可 build | ★高(直接对位"事前约束 vs QuantV2X 事后崩") | **主选(并行, 定位=消融)** |
| **E 难子集重 eval** | 现有 engine 在 DAIR val 难子集(远距/遮挡/夜间)重跑 AP, 看剪枝/INT8 在难样本掉幅是否放大 | **零(免训)** | 中 ~3-5h 工程(切分+重 eval) | 中-强(小/远目标 INT8 预期掉更多); **新颖零训练放大器** | ✅(复用现有 engine) | 中-高(放大 A/B 信号) | **第二选(信号放大器)**; 线索: repo 已有 `cliff3_*_hard/xhard` 难度概念可复用切分 |
| **C V2X-ViT INT8** | 引 transformer 模型, INT8 量化 | 中(需校准/可能 finetune) | **高 ~2-4 天** | 最强(QuantV2X 实测 75.1→29.9) | ❌ MSDA/attn 自定义 plugin **TRT 无 INT8** → engine build 不出 → **无 latency** | 高(QuantV2X 同款模型) | **后置/related-work + 可选 PyTorch AP-only 复现** |
| **D INT4 / W4A8** | 更低位宽 | 零(免训) | — | 最强 AP 信号但脏 | ❌ Ada sm89 TRT10 **无 INT4 IMMA** → build 不出 → AP-only; percentile 校准对自训长尾激活有 entropy 同类崩风险 | 中 | **仅消融脚注, 不进主 Pareto** |

> **推荐执行序(sw+data 一致)**: **A+B 并行立即跑(免训、双轴、覆盖 B1×B2 双侧)→ 看信号 → E 难子集作零训练放大器补强 → C related-work/AP-only → D 脚注。** 仅当 A+B+E 信号仍不足才触发兜底训练(换更难模型), 届时单独报 team-lead。

---

## 2. 推荐方案: A 主 + B 辅

### Path A(主) — 剪枝×**forced-INT8** 深扫, 激活**预测器 AP 信号 + 护栏消融对照**(★非前沿 trade-off, ISS-013)
- **假设(已部分验证, 分段)**: 模型越窄(剪枝率越高)→ 冗余越少 → INT8 量化误差越无处吸收 → **forced-INT8 的 ΔAP 在剪枝到一定程度后放大**(非 base 起即单调)。
- **★[2026-06-03 supervisor ISS-013 纠偏 — 必读] forced-int8 是 Pareto 被支配点, 不构成可部署前沿的 AP trade-off**:
  - **全档系统性支配(supervisor 实查升级, base/p50/p75 三档都成立)**: auto/uniform-int8 在延迟↓+AP↑两轴同时支配 forced-int8 ——
    - base 0.8113/0.6228 ≻ 0.9492/0.6203
    - p50 0.7956/0.5542 ≻ 0.9073/0.5465
    - p75 0.6124/0.5236 ≻ 0.6963/0.4956
    物理因: TRT-auto 延迟驱动选层=构造上最快, 且保敏感层 FP16=AP 更高。**naive INT8(QuantV2X 式)跨所有剪枝率都被 auto-routing 碾压**, 不是单点偶然。
  - ⇒ **Path A 的三重价值要分清**: ① ✓ 给 AP 预测器真训练信号(单调 ΔAP); ② ✓ "护栏 OFF"消融=差异化证据(**支配关系反让叙事更硬: forced/naive INT8 = QuantV2X 式做法, 被我们 auto-routing 双轴碾压**); ③ ✗ **不会让可部署 Pareto 前沿出现 AP trade-off**(前沿最优仍是 auto-int8 的 AP 高原)。
  - **标注纪律(与 Path B 同, ISS-013 责成)**: forced-int8 点入主表标 `regime=ablation_guardrail_off` / 非前沿, **仅作预测器训练 + 消融**, 不得在 Pareto 叙事里当"前沿出现 AP trade-off"。
  - **真·前沿级 AP trade-off 只能靠 Path E(难子集)/ C(V2X-ViT)/ 兜底(更难模型)** —— 当前 Pyramid/DAIR 上 auto-int8 前沿的 AP 仍是近常数高原。
- **★[2026-06-03 已验证, 关键修正] 必须用 forced-all-int8, 不能用 auto-int8**:
  - 实查金标准/perstage_v2(含新 p25 forced 点), **forced-all-int8 的 ΔAP70(vs FP16)分段放大**: base **-0.0106** / p25 **-0.0100** / p50 **-0.0176** / p75 **-0.0344**(噪声 ~0.001 的 **10-34×**, 可分辨)。**低剪枝档(base/p25)惩罚平缓 ~-0.010, 高剪枝档(p50/p75)显著放大** —— 物理上合理: 冗余足够时 INT8 误差有处吸收, 剪到一定程度才开始崩。**注意: 非 "base→p75 严格单调"(base≈p25), 写论文勿表述为严格单调, 防审稿反驳。**
  - 而 **auto-int8 非单调且整体更小**: base -0.0081 / p25 -0.0009 / p50 -0.0099 / p75 -0.0064。
  - 物理因: auto-int8 延迟驱动地把"不划算"层(恰=敏感层)留 FP16 = **免费护栏**, 各架构保哪些层不同。**forced 拆掉这个 auto 护栏 → 纯剪枝×量化交互, 高剪枝档惩罚放大。**
  - ⇒ **Path A 与 Path B 本质统一: 都是"拆护栏"** —— A=全局 forced(uniform INT8), B=定向敏感层(head/encoder)。auto-int8 则保留作 Pareto 上的"免费护栏"最优点。
- **弹药(sw 核实, 全 flat 已验, 免训)**:
  | tag | ckpt(bestval) | planes | total | pyramid_backbone | FP16 AP | INT8(forced) AP |
  |---|---|---|---|---|---|---|
  | pruned75 | `pruned75_2026_05_10/net_epoch_bestval_at31.pth` | [16,32,64] | 2.072M | **0.364M** | ✅金标准 | ✅金标准(已测) |
  | cliff2_c | `cliff2_c_2026_06_02/net_epoch_bestval_at39.pth` | [16,32,64] | 2.072M | **0.364M** | ✅ap_cliff2 | ❌待测 |
  | prune90 | `prune90_2026_06_02/net_epoch_bestval_at45.pth` | [~6,13,26] | 1.978M | **0.270M** | ✅ap_cliff | ❌待测 |
  | prune95 | `prune95_2026_06_02/net_epoch_bestval_at39.pth` | [~4,6,13] | 1.796M | **0.088M** | ✅ap_cliff | ❌待测 |
  - **★[2 个数据设计要点]** ① **pruned75 ≈ cliff2_c 同架构**([16,32,64] pb=0.364M)非独立点 → 真正不同 backbone 档 = **{0.364M, 0.270M, 0.088M} 三级**。② **total params 几乎不动(2.07→1.80M)而 pb 0.364→0.088M** → **Path A 剪枝轴必须用 pyramid_backbone params(或剪枝率), 不能用 total**(否则看着没剪)。
  - **新增工作量**: 仅 cliff2_c/prune90/prune95 各 1 个 forced-INT8 build+eval(3×~30min, 免训); FP16 AP 都已有(ap_cliff json)。
  - 已有 forced 点: base / p50 / p75(perstage_v2)。p25 forced **正在跑**(sw, 落 `results/p0_1_p25_forced_int8_ap.json`)。
- **交付(双轴完整点)**: sw 出每 ckpt 的 FP16+forced-INT8 DAIR val 1789 AP; hw 出对应 INT8 engine collab2 latency+energy → **lat+AP 完整点**, 直接并主表 + 喂预测器。
- **判据**: forced-INT8 分支在高剪枝档从 FP16 分支"剥离放大"(已确认 p50/p75 放大, base≈p25 平), 推到 prune90/95/cliff 看放大是否延续甚至触 AP 悬崖。

### Path B(辅) — 拆护栏消融, 坐实约束价值
- **做法**: 故意取消框架对 cls/reg/dir head + encoder 的 FP16 护栏, 强压 INT8 → 制造可控 AP 悬崖。
- **定位**: **消融实验(with-护栏 vs without-护栏), 不进主 Pareto 前沿**。
- **价值**: 直接证明"我们事前约束/传播 vs QuantV2X 事后才发现崩溃"的差异化, 在 **B2 量化轴**坐实(与 A/P0-1 的 B1 剪枝对齐轴互补, 双轴覆盖耦合陷阱)。

### C / D — 后置, 标清限制
- **C(V2X-ViT)**: TRT 无 transformer-INT8 → 无 latency, 作 related-work; 可选 PyTorch INT8 AP-only 复现崩溃叙事(对位 QuantV2X), 并展示"我们的约束传播会**事前**把它标为禁区"。
- **D(INT4/W4A8)**: 无 TRT build → AP-only 违反真测 latency 红线, 仅消融。

---

## 3. 成本与算力总结(给 team-lead 报用户)
- **A + B: 零训练算力**, 仅 TRT build + DAIR val eval(A 新增 ~1.5h 纯计算; B 同量级; GPU 占用同 P0)。**可立即批准, 风险低。**
- **E(难子集): 零训练**, 但 ~3-5h 工程(切分+重 eval)。作 A/B 信号放大器, 第二批。
- **C: 中等 ~2-4 天**(校准/可能 finetune + 大概率拿不到 latency), 建议后置/related-work。
- **D: 零训练但无 latency**, 仅消融脚注。
- **兜底**(仅 A/B/E 信号不足才触发): 换更难模型/任务 —— **这一步才需训练算力, 届时单独报 team-lead**。

---

## 4. 待办 / 门控
- [x] sw 回填 Path A 弹药清单(见 §2 表, 已填)。
- [ ] team-lead 审 + supervisor 把关 → 报用户确认 A+B(免训, 低门槛)。
- [ ] 批准后建 Task: A(sw FP16+forced-INT8 AP × hw collab2 latency 配对, 3 档)+ B(护栏消融)。
- [ ] E(难子集)第二批, 复用 repo `cliff3_*_hard` 难度切分线索, 降工程量。
- [ ] C/D 标 related-work/消融脚注, 不占主线算力。

---

## 5. ★最终结论 (2026-06-03 全量实测后 + 用户战略拍板)

**A/B/E 全量结果一致表明: Pyramid/DAIR 上 AP 没有可提取的成本-trade-off, AP 作约束而非目标轴。** 这是**有效科学结论, 非失败**(用户定调)。

| 路径 | 全量结果(实测) | 判定 |
|---|---|---|
| **A 剪枝×forced-INT8** | cliff 系 forced ap70 0.62/0.61/0.60(0.364/0.270/0.088M), 无悬崖, ≈各自 FP16 plateau。**cliff 系≠backbone 系不可拼曲线; ap_cliff FP16 是 2dp stdout 不可算 ΔAP70** | 作 ablation/predictor 点; INT8 近免费再确认 |
| **B 拆护栏 head-INT8** | base Δap70 **-0.0046** / p50 **+0.0044**(噪声内)→ **head-INT8 不崩** | 护栏必要性在 Pyramid conv head **不成立**(差异化叙事只在 transformer/V2X-ViT 才立, 但不投训) |
| **E 难子集重 eval** | 剪枝 spread 随距离 **shrink**(near -0.147 → r80plus **-0.074**, n_gt 3172); INT8 全箱免费(r80plus -0.0008) | **floor 效应/平移, 非难样本 trade-off**(smoke 3.6× 是 60 样本假象, 全量推翻)→ **未激活前沿 trade-off** |

**根因**: Pyramid 对 DAIR 严重过参数化 → INT8 处处近免费、剪枝无悬崖、难样本 spread 被 AP 地板压缩。

**新论文主线(用户拍板)**:
1. **耦合陷阱(P0-1)** —— tactic-level kernel-selection cliff 铁证(非解析、必须真测, 打 APQ/HAQ LUT)。
2. **(latency, energy, throughput) 多目标 Pareto 为主前沿; AP 作满足的约束**(过参数化下 AP≈常数, 这正由 A/B/E 实证支撑)。throughput 经 batch 空间维 sweep 脱离 1/lat 成真目标轴。
3. A/B = 护栏消融(regime=ablation, 非前沿); E = 诚实负结果(本节)。
4. **兜底(V2X-ViT)取消**, 仅 related-work 一句提及("框架规避的 QuantV2X 式 transformer-INT8 崩溃")。

⇒ **"激活 AP 故事"在 Pyramid/DAIR 上证伪; 转为"AP 近常数约束 + 成本轴协同优化"主线。** 这比强行造 AP trade-off 更诚实、更站得住。

---
*维护: A/B 信号实测后更新本文; 若触发兜底训练再补 §兜底详案。*
