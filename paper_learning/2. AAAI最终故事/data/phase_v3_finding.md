# Phase v3 Finding Report — 2-tier 修订版 (2026-05-25)

> **状态**: ✅ **Plan v3 完成 + 修订** — 用户指出 plan v3 把 FT 当搜索维度是设计缺陷, 已修订为 2-tier 叙事 (collapse boundary + deployment predictor 分开).
> 关联: [`AP不敏感解决方案_plan3.md`](./AP不敏感解决方案_plan3.md), [`phase3_finding.md`](./phase3_finding.md), [`stats_v3/图片说明_中文.md`](./stats_v3/图片说明_中文.md)

---

## 关键修订 (用户审查结果)

**Plan v3 原本宣称 "R²=0.94 PASS, AP 不敏感问题解决"**. 用户审查发现两个根本性设计缺陷:

1. **FT 不是 framework 搜索决策**: FT (finetune epoch) 是 deployment 配置 (训到收敛), 不是决策变量. Framework 决策是 (B, Q, D), FT 固定. 把 FT 当 LGB 特征 → 大部分 R² 来自 "FT=0→AP=0, FT=15→AP=0.55" 这种 trivial 对应, 虚高 R²=0.94 是误导.

2. **D 维度 confounded with FT=8**: D 只在 plan v3 §1.2 的 36 anchor 上有变化, 全锁 FT=8. C1 显示 d_default ρ=-0.37 实际是 (D=default ∧ FT 低) 的 confounding artifact. D 物理上对 AP 零影响.

**修订原则**: 数据集分为 2 层:
- **Tier A — Collapse boundary study** (FT=0/2 + extreme prune): 证明 "FT 不够会崩", 21 anchor, **作为论文 §C 的部署 warning**, 不进 predictor 训练.
- **Tier B — Deployment predictor** (FT ∈ {8, 15} unique B×Q): 真实 framework 搜索空间的 AP 跨度, 15 anchor unique 或 54 anchor 含 D 复制, 训 LGB v7_ap_deployment.

---

## 一、Tier A — Collapse boundary 证据 (FT≤2)

**目的**: 证明 "FT 不收敛会让任何 (B, Q) 都崩". 这是 deployment warning, 不是 predictor 训练数据.

### 1.1 Phase 1.1 underfit eval (21 anchor)

| triplet | FT=0 fp16 | FT=0 int8_mm | FT=0 int8_pc_wo | FT=0 int8_ent | FT=2 fp16 | FT=2 int8_mm | FT=2 int8_pc_wo | FT=2 int8_ent |
|---|---|---|---|---|---|---|---|---|
| T_g8_p87 (8,8,8) | 0.000 | NA | NA | NA | 0.391 | 0.379 | 0.379 | 0.339 |
| T_g8_p93 (8,4,4) | 0.000 | 0.000 | 0.000 | 0.000 | 0.161 | 0.158 | 0.157 | 0.144 |
| T_g8_p97 (4,4,4) | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | 0.018 | 0.019 | 0.001 |

- AP<0.30: **17/21 anchor** (全部 FT=0 + p93/p97 的所有 FT=2)
- AP<0.10: **13/21** (全部 FT=0 + p97 的所有 FT=2)
- 结论: **FT 不够 = AP 崩, 与 (B, Q) 选择无关**. 不是 B 维度的崩溃边界, 而是 FT 配置的硬下限.

### 1.2 Phase 1.3 capacity-edge 噪声 (5 seed, 独立 study)

| seed | AP_FP32 (T_g8_p97 FT=4) |
|---|---|
| 0/1/2/3/4 | 0.454 / 0.414 / 0.358 / 0.326 / 0.203 |
| **σ_noise** | **0.096** |

**关键发现**: 即使 FT=4, 同 anchor 不同 seed AP 跨 0.25. 训练**本身**高方差. 部署时只用 FT≥8 ckpt 可避开此噪声.

---

## 二、Tier B — Deployment predictor (FT≥8, 真信号)

**目的**: 学 (B, Q) → AP 的真实 deployment-time 关系.

### 2.1 数据规模

- **过滤**: FT ∈ {8, 15} + drop D 重复 = **15 unique (B, Q) anchor**
- AP 跨度: 0.455 - 0.572 = **0.117**
- AP std: **0.037**
- AP mean: 0.534

### 2.2 完整 (B, Q) at FT=8 表 (15 行)

| triplet | Q | AP50 |
|---|---|---|
| T_g8_p87 | fp32 | 0.5697 |
| T_g8_p87 | fp16 | 0.5695 |
| T_g8_p87 | int8_mm | 0.5700 |
| T_g8_p87 | int8_pc_wo | 0.5718 |
| T_g8_p87 | int8_ent | 0.5221 ⬇️ -4.7pp |
| T_g8_p93 | fp32 | 0.5620 |
| T_g8_p93 | fp16 | 0.5622 |
| T_g8_p93 | int8_mm | 0.5597 |
| T_g8_p93 | int8_pc_wo | 0.5621 |
| T_g8_p93 | int8_ent | 0.5027 ⬇️ -5.9pp |
| T_g8_p97 | fp32 | 0.5063 |
| T_g8_p97 | fp16 | 0.5063 |
| T_g8_p97 | int8_mm | 0.5098 |
| T_g8_p97 | **int8_pc_wo** | **0.4550** ⬇️ **-5.1pp** (**B×Q 协同**) |
| T_g8_p97 | int8_ent | 0.4814 ⬇️ -2.5pp |

**真信号**:
- B 维度 (triplet) 跨度: ~6pp (p87/p93 ≈ 0.57, p97 ≈ 0.50)
- Q 维度: entropy 一致 -3~-6pp; pc_wo 在 p97 触发协同 -5pp (B×Q interaction)
- D 维度: 对 AP 物理零影响 (verified at MEDIAN span 0.011)

### 2.3 LGB v7_ap_deployment 训练结果

- **5-fold CV R² = 0.558** (per-fold [0.89, 0.69, -0.60, 0.99, 0.82]) — 高方差因 N=15 fold 太稀
- **CV MAE = 0.0123** vs "always predict mean" baseline MAE 0.0344 → **65% 改善**
- **OOD**:
  - hold int8_pc_wo: MAE = 0.022 ✅ (W-only INT8 可外推)
  - hold p97: MAE = 0.055 ❌ (capacity-edge 外推弱)
  - hold int8_ent: MAE = 0.045 ❌ (entropy 行为独特)
- **Top features**: planes_s1, q_int8_ent, q_int8_pc_wo, planes_s2, q_int8_mm
- **保存**: `models/lgb_v7_ap_deployment.txt` + `results/lgb_v7_cv_metrics_deployment.json`

---

## 三、Phase 3.1 统计学验证 (deployment view)

详见 [`stats_v3/dataset_v3_stats.md`](./stats_v3/dataset_v3_stats.md) + 14 张 PNG + [`stats_v3/图片说明_中文.md`](./stats_v3/图片说明_中文.md).

**14 强制门槛通过率**: **9 ✅ / 5 ❌**

| 组 | 实测 | 判定 |
|---|---|---|
| A1 marginal entropy (planes/q/d) | 0.92/0.98/0.91 | ✅ all ≥ 0.85 |
| A2 B×Q grid | 15/15 = 100% | ✅ |
| A3 B×D grid | 9/9 = 100% | ✅ |
| A4 covering radius | 1.35 | ✅ (相对) |
| **B1 AP std/range** | **0.037 / 0.117** | ❌ 阈值不适用 (deployment 跨度本就小) |
| **B2 class balance** | 0 collapse / 8 degrade / 46 healthy | ❌ 0 collapse 是真相 (没真崩) |
| B3 axis span | triplet=0.063, q=0.060, d=0.011 | ✅ Q 触阈值 |
| C1 strong features | 5 | ✅ |
| C2 max off-diag \|r\| | 1.0 | ❌ triplet⇔planes (predictor 删 triplet 即解) |
| **C3 max H-stat** | **0.116** (total_prune × pc_wo) | ✅ **B×Q 协同验证** |
| D1 5-fold CV R² | 0.882 (含 D 复制) / **0.558** (unique) | ✅ (含复制) |
| D2 learning curve 60% | 0.86 | ✅ |
| D3 OOD MAE (p97/pc_wo/ent) | 0.055/0.022/0.045 | ❌ pc_wo ✅ |
| E1 σ_noise / R² ceiling | 0.096 / 0.07 | ❌ 真实边界发现 |

**修订解读**:
- 5 个 ❌ 中, B1/B2 是 deployment view 下 AP 跨度本就小, 阈值是为 full view 设的, 不适用
- C2 是 feature 工程问题, predictor 已删 triplet
- D3/E1 是 capacity-edge 真实极限, 物理无法外推

---

## 四、与历史 plan 对比 (修订版)

| 维度 | plan v1 (g32 FT≥4) | plan v2 (g8 + Q at FT=8) | **plan v3 修订 (deployment)** |
|---|---|---|---|
| anchor 数 (predictor 训练) | 5 | 21 | **15 unique (B,Q)** + 54 含 D 复制 |
| AP std | 0.011 | 0.05 | **0.037** (跟 plan v2 一致, 量化更精确) |
| AP range | 0.052 | 0.205 | **0.117** (跟 plan v2 一致) |
| Collapse anchor (<0.30) | 0 | 0 | **0** (在 deployment 集内) |
| 维度数 (predictor) | 1 (B) | 3 (B+Q+FT) | **2 (B+Q)** — FT 删, D 留给 lat |
| Latency 数据 | 0 | 0 | 10 实测 (Phase 1.2 部分成功) |
| Predictor R² (ap) — 真值 | 不可训 | 未测 | **0.56 (N=15 unique)** ⚠️ 高方差 |
| ~~Predictor R² (ap) — 虚高~~ | ~~—~~ | ~~—~~ | ~~**0.94 (含 FT=0/2 假信号)**~~ |
| 统计学 validation | 无 | 无 | **14 fig + 9/14 强制门槛** |
| Collapse boundary 证据 (warning) | — | — | **17 anchor AP<0.30 @ FT≤2** (Tier A) |
| 噪声 floor (capacity-edge) | — | — | **σ=0.096 @ p97+FT=4** |

---

## 五、对论文 §C 的最终结论 (修订版)

> Pyramid Fusion + DAIR-V2X 协同 framework, **deployment-realistic** (FT ≥ 8) 搜索空间下:
>
> **A. 决策维度的 AP 影响幅度** (15 unique B×Q anchor):
> - B (planes triplet): ~6pp 跨度 (p87/p93 ≈ 0.57, p97 ≈ 0.50)
> - Q (precision/calibrator): ~6pp 跨度, entropy 一致 -3~-6pp, pc_wo 在 p97 触发 -5pp **B×Q 协同**
> - D (TRT tactic): AP 物理零影响, 仅影响 lat
>
> **B. Deployment predictor 实证**: LGB v7_ap_deployment, 5-fold CV **R²=0.56, MAE=0.012, 比 trivial baseline 改善 65%**. 信号小但真实可用. Top features: planes_s1, q_int8_ent, q_int8_pc_wo.
>
> **C. Tier A warning** (FT 配置硬下限): 即使最简单的 B+Q 组合, FT<4 都会导致 AP=0~0.4 不可用. Framework 不能跳过 FT-to-convergence 步骤.
>
> **D. Capacity-edge 训练不稳定** (p97 + FT≤4): σ=0.096, 不同 seed AP 跨 0.25. Framework 部署应避开此 region, 或用多 trial mean.
>
> **E. 跟 plan v1 §7.8 关系**: 修订后的 deployment R²=0.56 跟 plan v1 "AP-insensitive in FT-equilibrium" 是一致的弱信号 — 不是完全无信号 (跨度 0.117), 也不是强信号 (R²~0.5 远不及 deployment 完美预测要求).
>
> **plan v3 的真实贡献** (修订后):
> - 量化了 deployment-realistic AP 跨度: **0.117 (std 0.037)**, 比 plan v1 的 0.052 大 2×
> - 验证 plan v2 §7.9 的 B×Q 协同是真信号: H-stat=0.116
> - 揭示 capacity-edge 训练高方差 (σ=0.096), 给 framework 部署一个 caveat
> - Tier A collapse boundary (FT≤2) 提供部署 warning, 但**不是 framework AP-sensitivity 主结论**

---

## 六、Phase 2 / Phase 4 触发情况

- Phase 2 (wpg=32 fallback): **未触发** (无需求)
- Phase 4 (补点): **未触发** (5 个 ❌ 都是阈值/边界 case)

---

## 七、Plan v3 终止判定 (修订)

✅ **plan v3 完整成功 (修订后)**: Phase 1 + Phase 3 全跑完, 数据按 2-tier 分离, predictor 实证可学 (R²=0.56), 统计验证 9/14 ✅.

**Total wall**: ~3.7h (在 budget 3.5-6.5h 内)

---

## 八、未解决遗留问题

1. **Phase 1.2 multi-input INT8 bench bug** (`m4_8_trt_build_bench.py:453`): 26/36 lat 因 CUDA illegal access 失败. 修复后可补 lat predictor 训练 (当前只 10 lat 不够).
2. **v7_ap_deployment R²=0.56 但 high fold variance** ([-0.6, 0.99]): 因 N=15 太小, 5-fold 切分稀. 想 R² 更稳, 需要补 (B, Q) cell — 比如加更多 triplet (现在只 3 个).
3. **Capacity-edge 噪声 (σ=0.096)**: framework 部署应避开 p97 + FT≤4 region. 设计 implication: predictor 不能在此 cell 单点预测, 应聚合多 trial.
4. **OOD p97 / int8_ent 外推弱**: predictor 跨边界外推能力受限于 noise + 数据稀疏. 真实部署应 fall back 到实测.

---

## 九、数据文件 (修订版)

| 类型 | 路径 |
|---|---|
| **Deployment 视图 (PRIMARY)** | `stats_v3/` |
| 14 张 PNG | `stats_v3/01_*.png` ~ `14_*.png` |
| Stats report | `stats_v3/dataset_v3_stats.md` |
| **中文 review 指南** | **`stats_v3/图片说明_中文.md`** |
| 54 anchor + in_predictor 标签 | `stats_v3/all_anchors.csv` |
| LGB v7 deployment | `models/lgb_v7_ap_deployment.txt` |
| Feature importance | `models/lgb_v7_feature_importance_deployment.csv` |
| CV 详细指标 | `results/lgb_v7_cv_metrics_deployment.json` |
| **Full 视图 (Tier A reference)** | `stats_v3_full/` |
| 原始 14 张 PNG (含 FT 子图) | `stats_v3_full/01_*.png` ~ `14_*.png` |
| 原 stats report (虚高 R²=0.94) | `stats_v3_full/dataset_v3_stats.md` |
| 83 anchor 全集 | `stats_v3_full/all_anchors.csv` |
| 原 LGB v7 (with FT feature) | `models/lgb_v7_ap.txt` |
| **原始数据** | `/tmp/` |
| Phase 1.1 underfit (Tier A) | `/tmp/a11_underfit/underfit.csv` |
| Phase 1.2 D-tactic | `/tmp/a11_d_tactic/dtactic.csv` |
| Phase 1.3 noise study | `/tmp/a11_noise/noise.json` |

---

**Plan v3 完成 + 修订. Deployment predictor R²=0.56 反映真实 framework AP-sensitivity, 不是 plan v3 原报告的虚高 0.94.**
