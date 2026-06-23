# Plan 4 Phase C path 2b Stage 1 Summary — mix Q 扩展失败 → 触发 path 4 reframe

**Date**: 2026-05-28
**Wall**: ~30 min (96 anchor × 5.5 min on 7 GPU)

## 一句话结论

24 triplet × 4 new mix Q (s1/s01/s02/s12) × FT=8 实测 96/96 anchor — **全在 plateau (range 0.459-0.573, std=0.023)**, 加入 Phase A 119 后 LGB GroupKFold R² 从 0.41 反而下降到 **0.39**, 5 gate 全 FAIL. **触发 plan v4 §6 path 4 reframe**.

## 关键数字

| 指标 | Phase A only | Phase A + Stage 1 |
|---|---|---|
| N anchor | 119 | 215 |
| Q variants | 5 | 9 (加 4 mix) |
| AP mean | 0.481 | 0.479 |
| AP std | 0.127 | 0.098 (↓ 因加入 stage 1 plateau anchor 稀释) |
| LGB 5-fold R² | 0.4147 | **0.3881** (↓) |
| Plateau-only R² | 0.572 | **−5.24** (爆炸 negative, 完全无法泛化) |
| SS(B) | 12.6% | 10.5% |
| SS(Q) | 55.9% | 56.9% (仍 ent 主导) |
| SS(B:Q) | 33.1% | 33.7% |

LGB per-fold: `[0.414, 0.743, 0.707, -0.474, 0.550]` — fold 3 严重负, 跨 triplet 极不稳.

## 4 mix Q 96 anchor 的 AP 分布 (范围 [0.459, 0.573])

| triplet | mix_s1 | mix_s01 | mix_s02 | mix_s12 | std |
|---|---|---|---|---|---|
| T15_p39 | 0.4912 | 0.4884 | 0.4585 | 0.4595 | 0.024 (最高 std) |
| T5_p62 | 0.5067 | 0.4912 | 0.4870 | 0.4912 | 0.014 |
| T19_p71 | 0.4937 | 0.4961 | 0.4885 | 0.4961 | 0.004 |
| (其他 21 个 std < 0.01, 全 plateau) | | | | | |
| T_g8_p87 | 0.5704 | 0.5718 | 0.5732 | 0.5689 | 0.002 |
| T22_p89 | 0.5430 | 0.5430 | 0.5435 | 0.5435 | 0.0003 |
| T12_p21 | 0.5354 | 0.5345 | 0.5346 | 0.5346 | 0.0004 |

**Mix Q 完全没有引入新崩塌或新 ranking**, 平均 std 跟 Phase A fp32 std=0.021 持平.

## 排除 ent (192 anchor, 8 Q)

```
std = 0.0232  (跟 Phase A 4 Q 排除 ent 的 std 几乎一样)
range = [0.4585, 0.5732]
```

**信号比噪声 (σ_FT8=0.018) 大 1.3×**, LGB 无法学到稳定 ranking pattern.

## 决策树触发

按 plan v4 §6 + Phase D §8.4:
- Stage 1 dispatcher 内置触发: 8 anchor < 0.50 → CONTINUE to LGB ✓ (但门槛过松)
- LGB 5 gate 验收 (基于 215 anchor):
  - G1 5-fold R²≥0.65: **0.388 FAIL**
  - G3 plateau R²≥0.55: **-5.24 FAIL**
  - G5 AP std≥0.10: **0.098 FAIL**
- → plan v4 §8.4 "任一 gate 不达 → 回 Phase A 重归因, **路径 4 reframe 是合法回退**"

## 不升级 Stage 2 的依据

Stage 2 含 KLD/percentile/head-fallback (~5h 工程 + 数据):
- **KLD/percentile**: 同 ent 一样是 calibrator 选 scale → 大概率重复 ent 随机崩塌, 不引入 ranking signal
- **head-fallback**: 把 cls/reg/dir 头强制 FP16 → **修好 ent crash**, AP 全回 plateau → **消除唯一区分信号**, 反向证明 plateau intrinsic

继续 Stage 2 的可能产出有 2 种:
1. 修好 ent → 24T × 9Q 全 plateau (std<0.005) — 印证 path 4 reframe
2. 新 calibrator 引入新随机崩塌 — 同样无法学

**两种结果都通向 path 4**. 不再投资工程, 直接 reframe + Phase E lat.

## 触发 path 4 reframe

按 plan v4 §6.5:
> **目标**: 不补 AP 数据, 全力补 lat (Phase E), reframe paper §C narrative.
> **新工作**: 0 新 AP. Phase C 跳过 (Stage 1 已跑完, 数据保留), 直接 Phase D 用 Phase A + Stage 1 215 anchor 出 14 图 + 写 reframe report.

## 下一步 (自动 trigger)

1. ✅ 写本 summary
2. ⏭️ 更新 state.json: phase_b_path="4", phase_c_done=true
3. ⏭️ Phase D 用 215 anchor 出 14 图 + reframe report (`stats_v3_plan4/`)
4. ⏭️ Phase D 5 gate 已知会 FAIL (跟此处一致), 但 14 图 + report 是 paper §C narrative 关键
5. ⏭️ Phase E 启动 (LAT 补强, 13h on 6 GPU, target lat R²≥0.75)

## 文件落点

- 96 anchor csv: `/tmp/plan4_phaseC_stage1/stage1_mix.csv`
- 96 ap.json: `/tmp/plan4_phaseC_stage1/*_ap.json`
- LGB v9 decision: `/tmp/plan4_phaseC_stage1/lgb_v9_decision.json`
- Combined 215 anchor 数据: 可合并 `phase_a_anchors.csv` + `stage1_mix.csv` → `plan4_combined_anchors.csv` (Phase D 用)
