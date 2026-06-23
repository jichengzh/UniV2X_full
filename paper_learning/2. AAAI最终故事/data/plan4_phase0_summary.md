# Plan 4 Pre-Phase 0 Summary — σ_FT8 实测

**Date**: 2026-05-27
**Wall**: ~15 min (5 seed × 2 epoch finetune ~5 min + 5 seed AP eval ~4 min)

## 结论 (1 句)

FT=8 噪声地板 σ = **0.0183**, 满足 `< 0.02` tight 阈值 → Phase A 启用 `threshold_signal = 0.06`.

## 关键数字

| 指标 | 值 |
|---|---|
| anchor | T_g8_p97 + FT=8 + Q=fp32 + D=D1 |
| seeds | 0, 1, 2, 3, 4 (PYTHONHASHSEED = 1000 + seed × 17) |
| 5 AP50 | [0.5557, 0.5269, 0.5666, 0.5690, 0.5710] |
| mean AP | 0.5579 |
| **σ_noise** | **0.0183** |
| range | 0.0441 |
| verdict | PASS_TIGHT |

## 对照历史

| 实验 | σ |
|---|---|
| FT=4 (plan v3 §1.3, 不可学) | 0.096 |
| FT=6 (plan v3 §7.11) | 0.023 |
| **FT=8 (本 phase 0)** | **0.0183** |

FT=8 比 FT=6 噪声小 21%, 比 FT=4 小 5.2×. 印证 §7.11 "FT=8 σ<0.02 (估)" 的推测.

## 下游影响

- Phase A 阈值: **std(AP across treatment) > 0.06** 才视为"有信号"
- Phase A SS attribution 阈值: > 30%
- Phase D AP gate: 5-fold GroupKFold R² ≥ 0.65, MAE ≤ 0.04

## 注意事项 (跨 phase)

- T_g8_p97 + FT=8 mean AP = 0.5579, 跟 bench v1 (T22_p89 + FT=bestval = 0.5414) 同量级 → FT=8 epoch 27 跟 bestval 在 plateau 内 AP 几乎等价
- 5 seed FT=8 ckpt 已落地 `models/dataset_a_cache_g8/noise_ft8_p97_seed{0..4}/net_epoch27.pth`, **可在 Phase A 中作为 g8 T_g8_p97 anchor 复用** (节约 1 个 ckpt 不必重训)

## 文件落点

- 5 anchor JSON: `/tmp/plan4_phase0/plan4_ph0_seed{0..4}_ap.json`
- 整合: `/tmp/plan4_phase0/noise.json`
- 状态: `plan4_state.json` 已更新 `phase_0_done=true`, `sigma_ft8=0.0183`, `threshold_signal=0.06`
- 5 seed ckpt: `models/dataset_a_cache_g8/noise_ft8_p97_seed{0..4}/net_epoch27.pth`
