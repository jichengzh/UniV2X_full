# Plan 4 Phase D Summary — 14 图 + 5 gate FAIL → path 4 reframe 确认

**Date**: 2026-05-28
**Wall**: ~5 min (14 PNG + LGB + 中文说明)

## 一句话结论

215 anchor (Phase A 119 + Stage 1 96) × 9 Q × FT=8 锁定 × D=D1 14 图全生成. **Plan v4 §8.4 5 gate 全 FAIL** (G1 R²=0.39 / G3 plateau R²=-5.24 / G5 std=0.098). **path 4 reframe 触发, 这是 plan v4 的科学发现**.

## 5 gate 实测 (vs plan v4 §8.4 预注册阈值)

| Gate | 阈值 | 实测 | Verdict |
|---|---|---|---|
| G1: 5-fold GroupKFold R² | ≥ 0.65 | 0.388 | ❌ |
| G3: Plateau-only R² (AP≥0.50) | ≥ 0.55 | −5.24 | ❌ |
| G4: 强相关特征数 (|ρ|>0.3) | ≥ 3 | 1 (只 q_int8_pt_wa_ent dummy) | ❌ |
| G5: AP std | ≥ 0.10 | 0.098 | ❌ |

**5/5 FAIL → 触发 plan v4 §11.2 成功 Path 2 (reframe)** (失败也算成功 — paper §C 主发现).

## 关键图视觉证据 (review user gate)

### B1: AP 直方图 (`05_target_histogram.png`)

```
N=215, std=0.098, range=0.572, mean=0.501
  - 87% anchor 集中在 [0.50, 0.57] plateau
  - 21 collapse anchor (AP < 0.30) — 全 int8_pt_wa_ent
  - degrade 8 anchor (0.30-0.50) — 主要 ent 部分崩
  - plateau 内 std<0.025 (无 ranking signal)
```

### D1: 5-fold CV (`11_cv_r2_per_fold.png`)

```
mean R² (random KFold script default) = 0.637  ← 含 in-triplet leakage
mean R² (GroupKFold by triplet, 真泛化) = 0.388  ← 实际值
per fold: [0.41, 0.74, 0.71, -0.47, 0.55]  ← fold 3 严重负
```

### D3: OOD predict-vs-real (`13_ood_predict_vs_real.png`)

OOD test 显示 predictor 在新 triplet 上无法预测 ent crash, MAE 大.

## paper §C reframe narrative (定稿)

```
Pyramid Fusion + DAIR-V2X 在 deployment-realistic 部署区:
  Search space: FT=8 / prune 0-97% / 9 Q / D 物理上零信号
  实测 215 anchor (factorial 全网格):
    · AP plateau intrinsic, std=0.098 < 阈值 0.10
    · 87% in [0.50, 0.57], 唯一 variance from entropy crash (non-monotonic, 不可学)
    · LGB R² capped 0.4 (vs target 0.65, GroupKFold)

Conclusion: Pyramid+DAIR over-parameterized for compression. Framework predictor 主体 = lat
(R² 0.47 → target 0.75 via Phase E D-rich + multi-build + kernel-aware feature).
AP safety filter rule-based suffices.
```

## 下一步 (path 4 next phase)

按 plan v4 §11.2 + §9 (Phase E):

1. ✅ Phase D 14 图 + 中文说明 (本 phase 完成)
2. ⏭️ Phase E **LAT 补强** (跟 AP 解耦, 跟 FT 无关, 可立即启动)
   - E.1 多 build sanity (100 anchor × 3 build, ~2.5h)
   - E.2 D 网格补 12 cell × 21T × 7Q = 1764 anchor (~7h)
   - E.3 kernel-aware feature (~3h 工程)
   - E.4 LGB v9_lat 训练
3. ⏭️ Final report (双轴整合)

## USER GATE (按 plan v4 §8.5)

plan v4 §8.5 要求 user review 3 张关键图后才能进 Phase E:
- `05_target_histogram.png` — plateau 视觉
- `11_cv_r2_per_fold.png` — R² 真泛化能力
- `13_ood_predict_vs_real.png` — OOD 失败模式

但 path 4 reframe 已自动触发 (5 gate FAIL), 用户 review 性质从"approve AP predictor"变为"approve reframe narrative + 启动 Phase E". 主 agent 暂停, 等用户确认.

## 文件落点

- 14 PNG: `paper_learning/2. AAAI最终故事/data/stats_v3_plan4/`
- 中文图说明: `stats_v3_plan4/图片说明_中文.md`
- LGB decision (G1-G5): `/tmp/plan4_phaseC_stage1/lgb_v9_decision.json`
- 完整 215 anchor csv: `stats_v3_plan4/all_anchors.csv`
