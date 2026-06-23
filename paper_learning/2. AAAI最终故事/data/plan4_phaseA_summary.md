# Plan 4 Phase A Summary — B × Q 因素归因 (120 anchor @ FT=8)

**Date**: 2026-05-28
**Wall**: ~50 min (24 ONNX export 6 min + 119/120 anchor 真测 on 7 GPU)

## 一句话结论

24 triplet × 5 Q × FT=8 实测 119 anchor: **AP 信号 100% 来自 Q_int8_pt_wa_ent 在部分 triplet 上的随机崩塌**, **B/Q 单维度 都无 ranking signal (std<0.06)**. LGB GroupKFold R²=0.41 边缘. 触发**预注册 path 2b** (但 path 4 reframe 也合理).

## ANOVA 关键数字

| 量 | 值 |
|---|---|
| N anchor | 119/120 (T6_p75 + int8_pt_wa_ent eval fail) |
| AP mean | 0.4806 |
| AP std | 0.1267 |
| AP range | [0.001, 0.5696] |
| SS(B) / SS(total) | **12.6%** |
| SS(Q) / SS(total) | **55.9%** ← Q 主导 SS, 但是… |
| SS(B:Q) / SS(total) | **33.1%** |
| SS(err) | -1.5% (full grid 无 residual) |
| LGB 5-fold GroupKFold R² | **0.4147** (per fold [0.48, 0.82, 0.71, -0.51, 0.58]) |

## 关键诊断: Q SS 全部来自 ent crash dummy

排除 int8_pt_wa_ent 看其他 4 Q (fp32/fp16/int8_pt_wa_mm/int8_pc_wo) 的 AP std:

| triplet | 4 Q AP | std (排除 ent) |
|---|---|---|
| T1_base | [0.5493, 0.5491, 0.5460, 0.5447] | **0.0023** |
| T22_p89 | [0.5431, 0.5415, 0.5433, 0.5428] | **0.0008** |
| T_g8_p97 | [0.5063, 0.5065, 0.5100, 0.5124] | **0.0029** |

**非 ent Q 跨度仅 0.0008-0.0029, 是 σ_FT8 (0.018) 的 1/6 — 比 noise floor 小一个数量级**.

A.1 B main effect (24 triplet × Q=fp32): std = **0.0211**, 仍 < threshold 0.06.

## 24 triplet × Q=fp32 AP (B main effect 数据)

| triplet | AP50 | triplet | AP50 |
|---|---|---|---|
| T1_base | 0.5491 | T13_p29 | 0.5378 |
| T2_p25 | 0.5407 | T14_p36 | 0.5378 |
| T3_p37 | 0.5260 | T15_p39 | 0.5174 |
| T4_p50 | 0.5375 | T16_p54 | 0.5085 |
| T5_p62 | 0.5067 | T17_p57 | 0.5232 |
| T6_p75 | 0.5583 | T18_p64 | 0.5081 |
| T7_wide_shallow | 0.5031 | T19_p71 | 0.4917 |
| T8_narrow_deep | 0.5432 | T20_p79 | 0.5067 |
| T10_p11 | 0.5384 | T21_p82 | 0.5215 |
| T11_p14 | 0.5260 | T22_p89 | 0.5415 |
| T12_p21 | 0.5346 | T_g8_p87 | 0.5696 |
|  |  | T_g8_p93 | 0.5622 |
|  |  | T_g8_p97 | 0.5063 |

跨 24 triplet (含 g8 极端到 97% prune) — **AP 全在 [0.49, 0.57] 内**, plateau intrinsic.

## entropy crash 触发情况 (21/24 triplet 不同程度崩塌)

| triplet | AP_ent | triplet | AP_ent |
|---|---|---|---|
| T1_base | 0.2395 (-31pp) | T13_p29 | 0.5230 (-1pp) |
| T2_p25 | 0.4860 (-5pp) | T14_p36 | 0.0010 (CRASH!) |
| T3_p37 | 0.1346 (CRASH) | T15_p39 | 0.0791 (CRASH) |
| T4_p50 | 0.2603 (-28pp) | T16_p54 | 0.1535 (CRASH) |
| T5_p62 | 0.4669 (-4pp) | T17_p57 | 0.4842 (-4pp) |
| T6_p75 | FAIL | T18_p64 | 0.0125 (CRASH) |
| T7 | 0.3813 (-12pp) | T19_p71 | 0.2807 (-21pp) |
| T8 | 0.1934 (CRASH) | T20_p79 | 0.0680 (CRASH) |
| T10 | 0.0127 (CRASH) | T21_p82 | 0.4055 (-12pp) |
| T11 | 0.4294 (-10pp) | T22 | 0.2402 (-30pp) |
| T12 | 0.4996 (-4pp) | T_g8_p87 | 0.5221 (-5pp) |
|  |  | T_g8_p93 | 0.5037 (-6pp) |
|  |  | T_g8_p97 | 0.4704 (-4pp) |

**8 个 triplet 严重崩塌 (<0.2), 16 个轻度降级, 跟 plane 大小无单调关系**. 这跟问题 §7.12.4 完全一致.

## 决策树触发情况 (按 §6 预注册优先级)

| 条件 | 触发? | 数字 |
|---|---|---|
| cond_1 (B 主导, std>0.12 AND SS(B)>30%) | ❌ | std=0.021, ss_b=12.6% |
| cond_2a (Q 排除 ent 信号 ≥2 triplet AND SS(Q)>30%) | ❌ | q_signal_count=0, ss_q=55.9% |
| cond_3 (B:Q 协同 SS>30% AND SS(B)<30% AND SS(Q)<30%) | ❌ | ss_q=55.9% > 30% |
| cond_2b (Q SS 但只 ent crash) | **✅** | ss_q=55.9% 全部来自 ent |
| (cond_4 fallback) | (n/a) | LGB R²=0.41 > 0.40 threshold |

**按预注册阈值, 自动选 path 2b**.

## ⚠️ 但 path 2b 风险评估

**path 2b** = 扩 Q 到 16 variants (KLD calibrator + percentile + per-stage mix + head-fallback). 工程 ~17h.

风险: 新加的 8 个 Q variant **大概率重复 ent 的随机崩塌模式**:
- KLD/percentile calibrator 本质跟 entropy 同类 (scale 选择敏感)
- per-stage mix (mix_s01/s02/s12) 看 bench v1 数据, 跟 mix_s0/mix_s2 同 plateau (std=0.01)
- head-fallback 可能"修好" ent crash → AP 全回到 plateau, 反而**消除信号**

实际上, 24T × 4Q (排除 ent) 实测 std<0.003 vs σ_FT8=0.018 — **信号比噪声小一个数量级**, 任何 Q 扩展都很难突破这个比例.

## ⚖️ 用户决策点 (Phase B 路径)

| 选项 | 工程 wall | 期望产出 | 风险 |
|---|---|---|---|
| **path 2b** (预注册自动选, 扩 Q 16 variants) | ~17h (含工具) | 期望 R² > 0.65, 但**可能仍 plateau** | 新 Q 重复 ent crash 模式, R² 提升<0.10 |
| **path 4 reframe** (诚实承认 intrinsic plateau) | 0h AP, 全力 Phase E lat 补强 (~13h) | paper §C 主发现: "Pyramid+DAIR over-parameterized for deployment-realistic compression" | 工程沉没成本不亏, lat 仍可单轴突破 |
| **path 3 mini** (B × Q 加密 grid, 仅 5-6 新 Q) | ~6h | 中间方案, 加新 Q 不超 6 个看是否触发新 ranking | 还是有可能 plateau |

**我倾向 path 4 reframe**: ANOVA 数字非常清楚, 非 ent Q 跨 24 triplet std<0.003. 在这种"信号比噪声小 6×"的数据上做 predictor 是浪费工程. paper §C narrative 改成 "intrinsic plateau + Q 维度仅有 entropy 随机崩塌 (binary 不可学)" 是 honest, 而且 problem §7.12 已经支撑这一点.

**但用户可能想看 path 2b 真跑出来的数据再下结论** — 这也合理. 工程预算允许的话, 走 path 3 mini (~6h, 中间方案) 可能是平衡选项.

## 下游影响

- Phase A 119 anchor + Pre-Phase 0 5 seed AP 已落地 `/tmp/plan4_phaseA/`, `/tmp/plan4_phase0/`
- 24 ONNX + 119 engine 已 cache (engine + ap.json + build.json)
- T2/T4/T6 epoch31 ckpt 在 HEAL checkpoints/ (FT=8 锁定 ckpt, 可复用)

## 文件落点

- 120 行 csv: `/tmp/plan4_phaseA/phase_a_anchors.csv`
- ANOVA: `/tmp/plan4_phaseA/attribution.json`
- 119 anchor 详细 AP JSON: `/tmp/plan4_phaseA/*_ap.json`
- 119 engine: `/tmp/plan4_phaseA/*.engine`
