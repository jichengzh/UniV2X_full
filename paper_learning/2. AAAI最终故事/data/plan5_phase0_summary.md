# Plan v5 Phase 0 — Pre-flight summary

**Run date**: 2026-05-29
**Result**: **all_pass=True**, 进入 Phase A

## 6 项 check 结果

| check | status | 详情 |
|---|---|---|
| g8_baseline_ckpt | ✓ PASS | `Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/net_epoch_bestval_at19.pth` (30.48 MB, loadable) |
| g32_baseline_ckpt | ✓ PASS | `Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth` (22.12 MB) |
| calibration_data | ✓ PASS | `pyramid_dair_calib_minmax.cache` + 4 其他 cache 存在 |
| tensorrt | ✓ PASS | v10.13.0.35, SPARSE_WEIGHTS + INT8 flag 都 supported |
| sparsity_tool | ✓ PASS (fallback) | apex 不可用 → `torch.nn.utils.prune` manual 2:4 mask 路径 |
| dair_val_n1789 | ✓ PASS | val.json 1789 sample 精确匹配 |

## 2 个 soft warning (不阻塞, 但需 Phase A/B 实施时 work around)

1. **trtexec binary 未装** — TRT 是 pip wheel install (没 binary). Phase A/B bench 必须用 TRT Python API (IBuilder + IBuilderConfig + IExecutionContext.execute_v2 + CUDA event 测时). Plan §5.3.3 / §7.3.2 里 trtexec 命令需要 Python wrapper 替代.

2. **apex.contrib.sparsity 未装** — Phase B 不能用 ASP.init_model_for_pruning + ASP.compute_sparse_masks 自动闭环. 改用 `torch.nn.utils.prune` 手动实现 2:4 structured mask:
   ```python
   for name, module in model.named_modules():
       if isinstance(module, nn.Conv2d):
           prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=1)
   ```
   功能等价但 finetune loop 需 hand-roll.

## 预注册阈值 (写入 plan5_state.json, 不允许 phase 跑完才改)

| gate | 阈值 |
|---|---|
| G_A_smooth_r2_min | 0.85 |
| G_B_sparsity_reduction_min | 0.30 (3/5 plane PASS) |
| G_C_ap_gap_max | 0.02 |
| G_D_dominate_lat_factor | 0.70 |
| G_D_dominate_ap_factor | 0.92 |

## 下一步 — Phase A

State: `current_phase=phase_A`, 进入 §5.

Phase A 分 4 step:
- A.1 prune 4 plane (sequential, ~10 min)
- A.2 finetune 4 ckpt epoch 19→27 (4 GPU 并发, ~20h wall)
- A.3 bench 54 cell (sequential TRT Python API, ~6h)
- A.4 smooth poly fit + gate G_A 判 (~1h)
