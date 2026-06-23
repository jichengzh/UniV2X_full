# AP 不敏感问题 — 最终保证方案 (plan v3)

> **状态**: ✅ **完整 plan 完成 + 修订 (2026-05-25)** — 用户审查后修订为 2-tier 叙事:
> - **Tier A** (FT=0/2 collapse anchor, 21 个): 作为 "FT 不够会崩" 部署 warning, 不进 predictor
> - **Tier B** (FT∈{8,15} unique B×Q, 15 anchor): 真实 deployment predictor, **真 R²=0.56** (不是原报告的虚高 0.94)
>
> 详见 [`phase_v3_finding.md`](./phase_v3_finding.md) (修订) + [`stats_v3/图片说明_中文.md`](./stats_v3/图片说明_中文.md) (deployment view 中文 review 指南).
>
> **背景**: plan v2 (`AP不敏感解决方案_plan2.md`) 已 ✅ 完成. 实测 AP std 从 plan v1 的 0.011 提升至 0.05 (4-5×), 触发 path A' (B×Q 协同崩溃) + path C (predictor 有信号). 但仍有 3 个缺口:
> 1. **strict path A 未触发** — FP32 单维 AP<0.30 没出现 (p97+FT4=0.367, 距阈值 7pp)
> 2. **collapse 点只有 1-2 个** — 数据集 std=0.05 仍偏小, 训 predictor 信号弱
> 3. **D 维度 (TRT tactic) 完全空白** — 当前 (B, Q, FT) 三维, 缺 D
>
> **plan v3 目标**: 用 **三机制并行** 把 AP std 推到 ≥0.15, range ≥ 0.5, 给 framework predictor 提供**统计学可证明充分**的数据集. 任一 mechanism fail 仍有另两个保底, 整 plan 不会全 fail.
>
> **三机制**:
> 1. **Mechanism 1 — FT-underfit**: FT=0 (raw) + FT=2 必崩, 用噪声地板锚定 random AP
> 2. **Mechanism 2 — wpg=32 baseline + planes=2** (fallback): 推到架构真下限
> 3. **Mechanism 3 — D 维度真测**: TRT builder tactic + batch size + sparsity, 直接复用 plan v2 ckpt
>
> **关联文档**:
> - 前置 plan v2: `AP不敏感解决方案_plan2.md` (✅ 完成)
> - plan v2 finding: `phase3_finding.md`
> - 问题清单: `问题.md` §7.7 + §7.9
> - 数据集: `数据集制作_plan.md` v1.5

---

## 一、范围与约束 (用户确认事项)

- ✅ 包含 FT=0/FT=2 underfit 区域 eval (Mechanism 1, P0)
- ✅ 包含 D 维度真测 (Mechanism 3, P0)
- ✅ 包含 5-seed 噪声估计 (1 anchor 即可)
- 🟡 wpg=32 baseline 重训 (Mechanism 2) 仅在 Phase 1 后 AP std<0.10 才触发, **conditional P1**
- ✅ 包含统计学 dataset validation report (强制门槛, 不过不能进 framework training)
- ❌ 不包含 INT4 (TRT 10.13 不支持)
- ❌ 不包含跨任务 (OPV2V) — 留 future work

---

## Phase 1 — 把 plan v2 ckpt 榨干 (~3.5h, 0 新基础训练)

**目标**: 基于已有 baseline_g8 + 3 triplet × 15 FT ckpts, 横向加 3 个新子实验, 直接生成 65 个新 anchor.

### 1.1 FT-underfit 子实验 (锚定 AP collapse 下限) — ~30 min

**输入 ckpts** (复用 plan v2):
- 3 triplet × FT=0 (raw, prune 后直接 eval, 已存于 `models/dataset_a_cache_g8/ft_T_g8_p{87,93,97}_raw/net_epoch_bestval_at*.pth` 的 epoch 0 / raw)
- 3 triplet × FT=2 (plan v2 FT 训练时已有 epoch 25/26 中间 ckpt, 等价 FT=2)

**实验矩阵**:

| 锚 | 数量 | 期望 AP | path A 触发? |
|---|---|---|---|
| FT=0 raw × 3 triplet × 4 Q (fp16/int8_mm/int8_pc_wo/int8_ent) | 12 | **~0.01-0.10** | ✅ 必触发 |
| FT=2 × 3 triplet × 4 Q | 12 | **0.10-0.40** | ✅ 多个触发 |

**实现**: 新建 `scripts/phase2/a11_eval_underfit.py`, 6 GPU 并行 dispatcher, ckpt sweep + 4 Q sweep.

**成功信号**:
- 24 个 AP eval 跑完
- 至少 6 个 anchor AP < 0.30 (strict path A 达成)
- 至少 3 个 anchor AP < 0.10 (random AP region 锚定)

**失败回退**: FT=2 epoch 中间 ckpt 不存在 → 用 plan v2 raw ckpt 重 train 2 epoch (额外 +1h × 3 GPU 并行).

### 1.2 D-tactic 子实验 (加第三维度真测) — ~1.5h

**目标**: 给 framework (B, Q, FT, D) 四维数据.

**D 配置 3 档** (trtexec build flags):
- D_default: `--avgRuns=200 --workspace=2048`
- D_sparse: 加 `--sparsity=enable` (TRT 自动稀疏化)
- D_bs8: `--minShapes=...:8x... --maxShapes=...:8x...` (batch=8 而非 1)

**实验矩阵**:

3 triplet × FT=8 (plan v2 中等档) × 4 Q × 3 D = **36 anchor**

**实现**: 新建 `scripts/phase2/a11_d_tactic_dispatcher.py`, 复用 ONNX (plan v2 Phase 3b 已 export), 仅重 TRT build + lat eval. **顺便采 latency 数据** (CLAUDE.md §四 latency predictor 崩的问题, 这步顺手补).

**成功信号**:
- 36 个 (engine_build + lat + ap_eval) 完成
- D 维度 AP std ≥ 0.02 (相同 B+Q+FT 下 D 至少有微小影响) — 不强制, 即使 D 无影响也是有效 finding
- Lat 维度: 同一 ckpt 不同 D 的 lat 跨度 ≥ 1.3× (验证 D 对 latency 有效)

**失败回退**: 若 TRT sparse build fail (groups=8 不支持 sparsity) → D 只测 default + bs8 两档, anchor 数减为 24.

### 1.3 多种子噪声估计 (1 anchor × 5 seed FT) — ~1h on 5 GPU

**目标**: 量化 finetune noise floor σ_noise, 决定 predictor R² 上限.

**选锚**: T_g8_p97 + FT=4 (plan v2 中 AP=0.367, 最敏感点)

**实验**: 5 个 seed (current + 0/1/2/3) 重 FT 4 epoch, 测 5 个 AP_FP32.

**实现**: 新建 `scripts/phase2/a11_noise_seed_dispatcher.py`, 复用 plan v2 raw_p97 ckpt, 改 train_ddp.py 的 random seed.

**成功信号**:
- 5 个 seed 完成
- σ_noise 计算: std(AP_FP32 across 5 seed)
- σ_noise ≤ 0.03 → R² ceiling ≥ 0.90 (健康)
- σ_noise ∈ (0.03, 0.05] → R² ceiling 0.75-0.90 (可用但勉强)
- σ_noise > 0.05 → **任何 predictor R² 都不会超过 0.5**, 触发扩 noise 估计 (多 anchor 重测)

### 1.4 Phase 1 产出汇总

- 新 anchor 总数: 24 (1.1) + 36 (1.2) + 5 (1.3) = **65**
- 加 plan v2 的 21 = **86 anchor total**
- 同时获得 36 个 (B, Q, D) lat 实测值 — 顺手解决 latency 问题
- **强制保底**: FT=0 必给 random AP, AP range 必到 [0.01, 0.572] = **0.56 wide** (plan v1 的 11×)

---

## Phase 2 — wpg=32 baseline + planes=2 极限 (conditional, ~3h)

**触发条件**: Phase 1 完成后, 若 AP std < 0.10 或 collapse anchor (<0.30) 数 < 5, 触发本 phase. 否则跳过.

### 2.1 重训 wpg=32 baseline_g8

**配置**: 新 hypes `lidar_pyramid_dair_v2x_basedair_g8_wpg32.yaml`, 仅改 `width_per_group: 32`.
- 解锁更小 planes: p=2 时 width = int(2*32/64)*8 = 8 (非零)
- 30 epoch DDP 4 GPU, ~110 min

### 2.2 极限 prune (4,2,2) + (2,2,2)

- T_g32wpg_p99 (4,2,2): ~99% prune
- T_g32wpg_p99_5 (2,2,2): ~99.5% prune, 架构真硬限

每个 prune + FT 8 epoch (60 min × 2 并行).

### 2.3 AP eval

- 2 triplet × 4 Q = 8 新 anchor
- 期望: 至少 1 anchor AP < 0.20 (硬路径 A)

**Phase 2 失败回退**: 若 (2,2,2) raw 后 finetune 不收敛 → 只用 (4,2,2), anchor 数减为 4.

---

## Phase 3 — 数据集封口 + statistical validation + predictor 训练 (~2h)

### 3.1 统计学 dataset validation report

**实现**: 新建 `scripts/phase2/dataset_stats_report.py`, 跑一次产 `dataset_v3_stats.md` + 6 张图.

**5 类指标 + 强制门槛** (任一不过 → 触发 Phase 4 补点):

**所有图片保存路径**: `paper_learning/2. AAAI最终故事/data/stats_v3/*.png` (1200×800 px, 300 dpi, matplotlib + seaborn)

#### 3.1.1 覆盖度 (Coverage)
- 边际归一化熵: 每维 (B, Q, FT, D) **≥ 0.85**
- Cell density: B×Q×FT 网格 filled ≥ **70%**
- Covering radius (standardized): ≤ **0.3**

#### 3.1.2 分布健康度 (Distribution Health)
- AP std ≥ **0.15**, range ≥ **0.5**
- Skewness |s| < 1, Kurtosis ∈ [1, 5]
- Dip test p < 0.05 (multimodal, 验证有 collapse cluster)
- 类平衡: 崩 (<0.30) / 降级 (0.30-0.50) / healthy (≥0.50) 各 ≥ **10%**

#### 3.1.3 信息量 (Mutual Information & Correlation)
- 至少 3 个特征 |Spearman ρ| > 0.3
- 任意特征 pair |Pearson r| < 0.95 (no redundancy)
- 至少 1 个 H-statistic > 0.1 (B×Q 交互验证)

#### 3.1.4 可学习性 (Learnability)
- 5-fold CV R² ≥ **0.75**, MAE ≤ **0.04**
- Label-shuffle baseline R² ≈ 0 (±0.05)
- 学习曲线 60% 处 R² ≥ 0.65
- OOD test:
  - Hold out p97 全部, train p87+p93, MAE ≤ 1.5 × in-sample
  - Hold out int8_pc_wo 全部, train 其他 Q, MAE ≤ 1.5 × in-sample

#### 3.1.5 噪声/可重复性 (Noise Floor)
- σ_noise (来自 Phase 1.3, 5 seed) ≤ **0.02**
- R² ceiling = 1 - σ²_noise/σ²_total ≥ **0.95** (给定 σ_total ~ 0.1)

#### 3.1.6 图片输出清单 (供人工 review)

`dataset_stats_report.py` 自动生成以下 **14 张图**, 每张图含 (内容 / 看什么 / 通过判据). 全部存 `paper_learning/2. AAAI最终故事/data/stats_v3/`.

**A 组 — 覆盖度 (4 张)**

| # | 文件名 | 内容 | 你要看什么 | 通过判据 |
|---|---|---|---|---|
| A1 | `01_coverage_marginal.png` | 4 子图: B/Q/FT/D 各维 histogram + 边际熵标注 | 每维分布是否过度集中在某一档 | 每维 bar 高度差 < 3×, 熵标注 ≥ 0.85 |
| A2 | `02_coverage_grid_BxQ.png` | B×Q 网格 heatmap (cell 值=anchor 数), 空白 cell 灰色 | 是否有大片空白角落 | 空白 cell 比例 ≤ 30% |
| A3 | `03_coverage_grid_BxFT.png` | B×FT 网格 heatmap | 极端 prune 是否覆盖 FT 全档 | 同上 |
| A4 | `04_coverage_hull_pca2d.png` | PCA 2D 投影, anchor 散点 + 凸包多边形 | 是否有"孤岛"或"洞" | 凸包外 anchor 比例 < 5% |

**B 组 — Target 分布 (3 张)**

| # | 文件名 | 内容 | 你要看什么 | 通过判据 |
|---|---|---|---|---|
| B1 | `05_target_histogram.png` | AP histogram + KDE 曲线 + 3 类阈值竖线 (0.30, 0.50) | 是否多峰 (健康) 或单峰扎堆 | 至少看到 2 个 mode, 不全堆在 [0.5, 0.6] |
| B2 | `06_target_class_balance.png` | 横向 stacked bar: 崩 / 降级 / healthy 三类计数 + 百分比 | 是否有类比例 < 10% | 三类各 ≥ 10% |
| B3 | `07_target_by_axis_box.png` | 4 子图, 各维度的 AP boxplot (e.g. 不同 Q 下 AP box) | 哪个轴对 AP 影响最大 | FT 轴 box 跨度 > 0.3, Q 轴 box 跨度 > 0.05 |

**C 组 — 信息量 (3 张)**

| # | 文件名 | 内容 | 你要看什么 | 通过判据 |
|---|---|---|---|---|
| C1 | `08_mi_spearman_per_feature.png` | 横向 bar, 每特征 MI + Spearman ρ 双色 bar | 哪些特征是 dead 维, 哪些是 dominant | ≥ 3 个特征 \|ρ\|>0.3, 没有全 dead |
| C2 | `09_corr_matrix.png` | 特征 × 特征 Pearson heatmap (-1 ~ +1, RdBu) | 是否有冗余特征 (深色非对角块) | 对角线外 \|r\| 全部 < 0.95 |
| C3 | `10_interaction_hstat.png` | Top-5 特征 pair 的 H-statistic bar | 是否有 B×Q 交互证据 | 至少 1 pair H > 0.1 |

**D 组 — 可学习性 (3 张)**

| # | 文件名 | 内容 | 你要看什么 | 通过判据 |
|---|---|---|---|---|
| D1 | `11_cv_r2_per_fold.png` | 5-fold CV R² + MAE 双子图 bar (含 shuffle baseline 对照) | fold 间是否稳定, vs shuffle 是否显著拉开 | R² mean ≥ 0.75 且 R²_real - R²_shuffle > 0.5 |
| D2 | `12_learning_curve.png` | 训练集大小 sweep, R² 曲线 + shuffle baseline 双曲线 | 是否 saturate, 是否还在涨 | 60% 数据时 R² ≥ 0.65, 曲线趋平 |
| D3 | `13_ood_predict_vs_real.png` | 散点图, 3 个 hold-out scenario (p97 / pc_wo / 混合) 三色 + y=x 对角线 + ±σ_noise band | OOD 预测是否系统偏离 | 90% 点落在 ±2σ_noise 内 |

**E 组 — 噪声 (1 张)**

| # | 文件名 | 内容 | 你要看什么 | 通过判据 |
|---|---|---|---|---|
| E1 | `14_noise_floor.png` | 5 seed × T_g8_p97_FT4 的 AP 散点 + mean line + σ band + R²_ceiling 标注 | seed 间方差是否过大 | σ_noise ≤ 0.02, R²_ceiling ≥ 0.95 |

**图片生成统一规范** (写入 `dataset_stats_report.py`):
- 字体: DejaVu Sans, 标题 16pt, 轴标 12pt
- 配色: seaborn `colorblind` palette, 强调色 #C0392B (红, 标 fail), #27AE60 (绿, 标 pass)
- 每图右下角标 "auto-generated YYYY-MM-DD HH:MM"
- 每图标题包含通过状态: e.g. "AP Class Balance ✅ (崩 14 / 降级 21 / healthy 51)"
- 配套 `dataset_v3_stats.md` 在每节插入对应图链接 `![A1](stats_v3/01_coverage_marginal.png)`

**人工 review 流程**:
1. 跑完 `dataset_stats_report.py` 后, 进 `stats_v3/` 目录
2. 按 A→B→C→D→E 顺序看 14 张图 (~5 min)
3. 看每图标题的 ✅/❌, 红色标的就是要触发 Phase 4 补点的指标
4. 跟 `dataset_v3_stats.md` 对照, ✅ 数 ≥ 12 → 进 Phase 3.2 训 predictor

### 3.2 Predictor 训练 (LGB v7)

仅在 §3.1 所有门槛通过后执行:

- Train: 86 anchor (或 Phase 4 补点后 N)
- Features: B (planes 三档 + total_prune%) + Q (4 cat + calib_type one-hot) + FT (epoch_count) + D (sparse/bs/tactic one-hot) = ~10 dim
- Target: AP50
- 同时训 latency predictor LGB v7_lat (用 Phase 1.2 的 36 lat 实测点 + plan v2 baseline 83 点)

**输出**:
- `models/lgb_v7_ap.txt`
- `models/lgb_v7_latency.txt`
- `models/lgb_v7_feature_importance.csv`
- `results/lgb_v7_cv_metrics.json`

### 3.3 dataset_v3_stats.md 输出模板

```md
# Dataset v3 Stats Report (auto-generated YYYY-MM-DD)

## Summary
- N anchors: 86
- Pass/fail per metric: [N ✅ / 12]

## Coverage
- Marginal entropy: B=0.92, Q=1.00, FT=0.89, D=1.00 ✅
- Cell density: 28/36 = 78% ✅
- Covering radius: 0.27 ✅

## Distribution
- AP std: 0.17 ✅, range: 0.56 ✅, range 0.01→0.57
- Class balance: 崩 14 / 降级 21 / healthy 51 ✅
- Dip test p=0.02 (multimodal) ✅

## Information
- Top features by MI: planes_3 (0.45), q_calib_type (0.32), ft_epoch (0.28) ✅
- B×Q H-stat: 0.18 ✅ (interaction confirmed)

## Learnability
- 5-fold R²: 0.81 ± 0.04 ✅ (shuffle: 0.02)
- OOD p97 hold-out MAE: 0.045 ✅
- OOD pc_wo hold-out MAE: 0.039 ✅

## Noise
- σ_noise (T_g8_p97_FT4, 5 seed): 0.018 ✅
- R² ceiling: 0.96 ✅

## Verdict: ✅ Ready for framework training
```

---

## Phase 4 — 兜底补点 (conditional, ~1-2h)

**触发条件**: Phase 3.1 任一统计门槛不过.

**策略** (按违反指标 case-by-case):

| 不过的指标 | 补点策略 | 补几个 |
|---|---|---|
| Cell density < 70% | 列出空白 cell, 各补 1 anchor | 视空白数 (~5-15) |
| AP std < 0.15 | 加 1 个新 triplet (8,4,2) × 4 Q × 2 FT | 8 |
| Class 平衡: 崩 < 10% | 加 FT=1 区, 4 anchor | 4 |
| OOD MAE 超标 | 加 hold-out cell 邻近 anchor | 5-8 |
| R² < 0.75 | 综合补点 (重跑 §3.1 评估) | 10-20 |

**Phase 4 完成后**: 重跑 §3.1, 若仍不过 → 写 `phase4_finding_limits.md`, 说明数据集统计学验证的真实极限, 转入 framework 分析阶段.

---

## 整体 /goal 终止条件

✅ **整体 plan 完成判定** (任一路径):

1. **完整成功**: Phase 1 ✓ + Phase 3 statistical ✓ + Phase 3 predictor 训练 ✓, 写 `phase_v3_finding.md`
2. **Phase 2 触发后成功**: 完整成功 + Phase 2 ckpts/anchor 也并入数据集
3. **Phase 4 触发后成功**: 完整成功 + Phase 4 补点纳入
4. **早停 (Phase 1.3 噪声爆掉)**: σ_noise > 0.05, 写 `phase1_noise_finding.md`, 说明 finetune noise 决定了 predictor R² 上限不可逾越

❌ **整体 plan 失败** (需人工介入):
- Phase 1 任一 sub-phase 反复 OOM/异常
- Phase 4 重跑后仍 R² < 0.6 (说明数据本质不可学, 需重设特征工程)
- 任意 phase 超出预算 2× (Phase 1 > 7h, Phase 2 > 6h, Phase 3 > 4h, Phase 4 > 4h)

---

## 时间预算汇总

| Phase | 工作 | 预算 wall | GPU 占用 | 是否必走 |
|---|---|---|---|---|
| 1.1 | FT=0 + FT=2 × 3 triplet × 4 Q (24 anchor eval) | ~30 min | 6 GPU | ✅ |
| 1.2 | D-tactic 36 anchor (TRT build + lat + ap) | ~1.5h | 4 GPU 并行 | ✅ |
| 1.3 | 5 seed × FT4 × T_g8_p97 噪声估计 | ~1h | 5 GPU 并行 | ✅ |
| 3.1 | dataset_stats_report.py + 6 张图 | ~10 min | 1 CPU | ✅ |
| 3.2 | LGB v7 ap + latency 训练 + CV | ~30 min | 1 CPU | ✅ |
| 2 | wpg=32 baseline + (4,2,2)/(2,2,2) finetune | ~3h | 4 GPU DDP | 🟡 conditional |
| 4 | 兜底补点 | ~1-2h | 视情况 | 🟡 conditional |
| **必走总计** | (Phase 1 + 3) | **~3.5h** | 4-6 GPU | — |
| **最大总计** | (Phase 1 + 2 + 3 + 4) | **~6.5h** | 4-6 GPU | — |

---

## 关键脚本/路径清单

| 文件 | 类型 | 状态 |
|---|---|---|
| `scripts/phase2/a11_eval_underfit.py` | Phase 1.1 dispatcher (24 anchor) | 🔲 待写 |
| `scripts/phase2/a11_d_tactic_dispatcher.py` | Phase 1.2 dispatcher (36 anchor + lat) | 🔲 待写 |
| `scripts/phase2/a11_noise_seed_dispatcher.py` | Phase 1.3 dispatcher (5 seed) | 🔲 待写 |
| `scripts/phase2/dataset_stats_report.py` | Phase 3.1 stats + **14 张图 (A1-E1)** | 🔲 待写 |
| `paper_learning/2. AAAI最终故事/data/stats_v3/` | 14 张 PNG 输出目录 | 🔲 待生成 |
| `scripts/phase2/a11_train_lgb_v7.py` | Phase 3.2 LGB ap + lat 训练 | 🔲 待写 |
| `scripts/phase2/a11_wpg32_train.py` | Phase 2 dispatcher (conditional) | 🔲 待写 |
| `scripts/phase2/a11_phase4_补点.py` | Phase 4 dispatcher (conditional) | 🔲 待写 |
| `paper_learning/2. AAAI最终故事/data/dataset_v3_stats.md` | Phase 3.1 输出 | 🔲 待生成 |
| `paper_learning/2. AAAI最终故事/data/phase_v3_finding.md` | 终态报告 | 🔲 待生成 |
| `models/lgb_v7_ap.txt` | Phase 3.2 输出 | 🔲 待生成 |
| `models/lgb_v7_latency.txt` | Phase 3.2 输出 | 🔲 待生成 |

---

## 已有可复用资源 (来自 plan v2)

| 资源 | 路径 | 用途 |
|---|---|---|
| baseline_g8 ckpts (30 epoch) | `/home/jichengzhi/heal_research/HEAL/opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/` | Phase 1.3 seed init |
| 3 triplet × 15 FT ckpts | `/home/jichengzhi/UniV2X/models/dataset_a_cache_g8/ft_T_g8_p{87,93,97}_raw/` | Phase 1.1 (FT=0/2 提取) + 1.2 (复用 FT=8 ckpt) |
| ONNX exports (3 triplet × FT=8) | plan v2 Phase 3b 已生成 (路径见 a10_eval_3b_b_q_combo.py) | Phase 1.2 复用 (避免重 export) |
| Calibration data (5 input) | plan v2 Phase 3b 已生成 | Phase 1.2 复用 |
| Plan v2 21 anchor 数据 | `/tmp/a10_phase3a/spread_3a.csv` + `/tmp/a10_phase3b/combo_3b.csv` | Phase 3.1 + 3.2 输入 |
| Plan v1 三模型 baseline lat | `data/baseline_4090.parquet` (83 行) | Phase 3.2 latency predictor 训练数据 |

---

## 风险与回退策略

| 风险 | 概率 | 影响 | 回退 |
|---|---|---|---|
| FT=2 中间 ckpt 缺失 (plan v2 save_freq 可能没存) | 中 | Phase 1.1 部分 fail | 用 raw ckpt 重 train 2 epoch (+1h × 3 GPU 并行) |
| TRT sparse build fail (groups=8 不兼容) | 低 | Phase 1.2 减点 | D 维度只测 default + bs8, anchor 数减 12 |
| 5 seed 训练 GPU 不够 (其他用户占) | 中 | Phase 1.3 串行化 | 串行 5 次, wall 5h, 或减为 3 seed |
| Phase 1 全跑完仍 AP std < 0.10 | 低 | Phase 2 触发 | 走 Phase 2 wpg=32 路径 (+3h) |
| Phase 4 补点后仍 R² < 0.6 | 低 | 数据本质不可学 | 写 `limits_finding.md`, 转入 framework 分析报告 |
| 任意 phase 超预算 2× | 中 | 影响整 plan 时间 | 暂停 + 写 partial finding, 不强推 |

---

## 验证清单 (执行中检查点)

### Phase 1
- [ ] Phase 1.1: 24 anchor (FT=0/2 × 3 triplet × 4 Q) eval 完成
- [ ] Phase 1.1: 至少 6 anchor AP<0.30 (strict path A 达成)
- [ ] Phase 1.1: 至少 3 anchor AP<0.10 (random region 锚定)
- [ ] Phase 1.2: 36 anchor (D × FT=8 × Q × triplet) lat + ap eval 完成
- [ ] Phase 1.2: 同一 ckpt 不同 D 的 lat 跨度 ≥ 1.3× (D 对 lat 有效)
- [ ] Phase 1.3: 5 seed × T_g8_p97_FT4 训练完成
- [ ] Phase 1.3: σ_noise 计算并入 stats report

### Phase 3
- [ ] Phase 3.1: dataset_stats_report.py 跑通, 输出 dataset_v3_stats.md
- [ ] Phase 3.1: **14 张图 (A1-E1) 全部生成于 `stats_v3/`** (供人工 review)
  - [ ] A 组 4 张 (coverage_marginal / grid_BxQ / grid_BxFT / hull_pca2d)
  - [ ] B 组 3 张 (target_histogram / class_balance / by_axis_box)
  - [ ] C 组 3 张 (mi_spearman / corr_matrix / interaction_hstat)
  - [ ] D 组 3 张 (cv_r2 / learning_curve / ood_predict_vs_real)
  - [ ] E 组 1 张 (noise_floor)
- [ ] Phase 3.1: 12 个强制门槛全部 ✅ (或触发 Phase 4)
- [ ] **Phase 3.1: 人工 review 14 张图后 confirm** (/goal pause 等待用户 ack)
- [ ] Phase 3.2: LGB v7_ap 训练 + 5-fold CV R² ≥ 0.75
- [ ] Phase 3.2: LGB v7_lat 训练 + 5-fold CV R² ≥ 0.75
- [ ] Phase 3.2: OOD test 通过 (p97 + pc_wo hold-out)

### Phase 2 (conditional)
- [ ] Phase 2: wpg=32 baseline 30 epoch 完成
- [ ] Phase 2: (4,2,2) + (2,2,2) finetune 完成
- [ ] Phase 2: 8 新 anchor eval, 至少 1 个 AP<0.20

### Phase 4 (conditional)
- [ ] Phase 4: 列出违反指标 + 对应补点 anchor list
- [ ] Phase 4: 补点 eval 完成 + 重跑 stats report

### 终态
- [ ] 写 `phase_v3_finding.md` 报告
- [ ] 更新 `paper_learning/2. AAAI最终故事/data/问题.md` §7.10 加 plan v3 结果
- [ ] 更新 `数据集制作_plan.md` 至 v1.6 (含 v3 86+ anchor 数据)
- [ ] commit + 推 branch

---

## 启动前 sanity check (人工)

启动 /goal 前确认:

1. ✅ baseline_g8 ckpts 仍在 `/home/jichengzhi/heal_research/HEAL/opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/` (未被清理)
2. ✅ 3 triplet FT ckpts 仍在 `models/dataset_a_cache_g8/ft_T_g8_p*_raw/` (未被清理)
3. ✅ plan v2 spread_3a.csv + combo_3b.csv 仍在 `/tmp/` (可能已清, 需重 eval 或从 git 找)
4. ✅ 4-6 GPU 可用 (4090 host, 看 `nvidia-smi`)
5. ✅ 磁盘 ≥ 50GB 空闲 (Phase 2 wpg=32 训练 + 中间 ONNX/engine)

启动后, /goal 应按 Phase 1 → 3 → (2/4 conditional) 顺序推进, 每个 sub-phase 独立 task, 失败不阻断后续 (除非 fatal).

---

## 与 plan v1/v2 对比

| 维度 | plan v1 (g32) | plan v2 (g8 + Q) | **plan v3 (g8 + Q + D + FT-underfit)** |
|---|---|---|---|
| anchor 数 | 5 | 21 | **86 (必走) + Phase 2/4 补点** |
| AP std | 0.011 | 0.05 | **≥0.15 (target)** |
| AP range | 0.052 | 0.205 | **≥0.5 (target)** |
| Collapse anchor (<0.30) | 0 | 0 (弱 path A) | **≥6 (强 path A)** |
| 维度数 | 1 (B) | 3 (B, Q, FT) | **4 (B, Q, FT, D)** |
| Latency 数据 | 0 (只 baseline) | 0 (plan v2 没测 lat) | **36 实测 lat 点 (D 维度) + 83 baseline** |
| Predictor R² (ap) | 不可训 | 未测 | **≥0.75 强制门槛** |
| Predictor R² (lat) | 0 (崩) | 未测 | **≥0.75 强制门槛** |
| 统计学 validation | 无 | 无 | **12 强制指标, 自动 report** |
| 论文价值 | (无信号) | path A' 协同 | **(B,Q,FT,D) 完整 + statistical guarantee** |

---

## 论文 §C 重写预期 (plan v3 完成后)

> Pyramid Fusion + DAIR-V2X 任务在 framework 协同搜索空间 (B, Q, FT, D) 下展现出从 over-parameterized region 到 capacity-bounded region 的清晰过渡:
>
> - **过 FT region** (FT≥8, prune ≤ 89%, FP32): 高度 over-parameterized, AP 几乎不变 (验证 plan v1 现象).
> - **极端 B region** (planes ≤ 8, g=8): AP 与 FT 强耦合, FT 不足时 AP 跌至 0.37 (plan v2 finding).
> - **B×Q 协同 region** (planes=4 + pc_wo INT8): -5pp 协同损失, 单维不可见 (plan v2 path A').
> - **strict collapse region** (FT≤2 + 极端 prune + 错 Q 配): AP 跌至 < 0.30, 触发硬路径 A (**plan v3 新增**).
> - **D 维度影响** (TRT sparsity / batch / tactic): lat 跨度 1.3-2.0×, AP 几乎不变 (plan v3 finding).
>
> Framework 的 LGB predictor 在 86+ anchor 数据集上达到 5-fold CV R²=0.81 (vs label-shuffle baseline R²=0.02), OOD test 通过 — **统计学层面证明可学**.

---

**plan v3 完, 等待 /goal 启动.**
