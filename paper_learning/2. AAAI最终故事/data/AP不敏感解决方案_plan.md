# AP 不敏感解决方案 — Fixed-FT 数据集重建计划

> **状态 (2026-05-21 21:00 更新)**: ✅ **Phase 1 早停 — sweet spot 不存在**.
> Pilot 实测 3 triplet × 5 FT (4/6/8/10/15) 全部满足 AP[T11]>0.50 但 AP[T11]-AP[T22] gap 永远 < 0.025 (远低于 0.05 阈值).
> Pyramid Fusion + DAIR-V2X 在 backbone 维度过度 over-parameterized — 任意 FT≥4 + 任意 prune ratio 都几乎同 AP.
> **Phase 2/3 不执行** (按 plan §整体成功信号 #2 早停路径).
> 完整发现报告: `/tmp/a9_pilot/eval/phase1_finding.md`. 问题.md §7.7 已同步.


> **背景**: 当前 e2e_bench_v1.csv 4704 anchor AP 分布过度集中 (83.5% 样本在 [0.52, 0.54], std=0.03). 根因是每个 triplet 都被 finetune 25-47 epoch 到 equilibrium, B/Q/D 维度信号被抹平. 详见 `问题.md` §7.
>
> **目标**: 用固定较小 FT epoch 重训 21 triplet, 让 AP 在 (T, Q) 空间内保留可学习的 variance (重 prune 有可量化退化, 轻 prune 不崩), 同时控制训练成本.
>
> **决策硬约束** (来自用户 2026-05-21):
> - FT 不能太少 (避开 single-seed noise): **FT ≥ 4**
> - FT 不能太多 (信号被抹平): **FT ≤ 20**
> - 轻度 prune **不能崩**: AP[T11_p14, 14% prune] > 0.50
> - 重度 prune **可以崩** (但要有理由): AP[T22_p89, 89% prune] 可显著低于 T11
> - **Single seed** (不重复跑取平均)
>
> **关联文档**: `paper_learning/2. AAAI最终故事/data/问题.md` §7, `数据集制作_plan.md` v1.4

---

## Phase 1 — Pilot: 找 FT* sweet spot (~50 min on 6 GPU)

### 1.1 目标

实验确定 FT* ∈ [4, 20], 满足:
1. **spread(FT*) maximized** = std(AP[T11], AP[T17], AP[T22]) 最大
2. AP[T11_p14] > 0.50 (轻 prune 不崩)
3. AP[T22_p89] < AP[T11_p14] - 0.05 (重 prune 有差异化)

### 1.2 执行步骤

#### 1.2.1 启用 HEAL per-epoch checkpoint 保存

修改 `/home/jichengzhi/heal_research/HEAL/opencood/tools/train_ddp.py` 让 ckpt 每 epoch save 而非每 2 epoch:
- 找到 save logic (搜 `if epoch % `)
- 改为 every epoch save, 或参数化 `--save-every 1`
- **避免污染原始训练流程**: 用 monkey-patch 或临时 fork

**成功信号**: 训练运行后, `models/dataset_a_cache/ft_*/net_epoch{N}.pth` 出现 N=24,25,26,...,38 连续序列 (不再跳 2).

#### 1.2.2 训练 T11 / T17 raw → FT=15 epoch (3 GPU 并行)

```bash
# T11_p14: raw ckpt 已存在 /home/jichengzhi/UniV2X/models/dataset_a_cache/ft_064_064_256_raw
# T17_p57: 需先 structural_prune
# T22_p89: raw ckpt 已存在 /home/jichengzhi/UniV2X/models/dataset_a_cache/ft_016_016_016_raw
```

预 prune T17 (新工作):
```bash
mkdir -p models/dataset_a_cache/ft_032_032_128_raw
python tools/structural_prune_pyramid.py \
    --orig-dir /home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29 \
    --out-dir models/dataset_a_cache/ft_032_032_128_raw \
    --num-filters-new 32,32,128
```

启动 3 个 finetune (每个 ~25 min on 1 GPU, 15 epoch):
- T11 raw → 15 epoch on GPU 0
- T17 raw → 15 epoch on GPU 1
- T22 raw → 15 epoch on GPU 2 (补 FT=15 数据)

**成功信号**:
- 每个 ft_dir 出现 `net_epoch24.pth`, `net_epoch25.pth`, ..., `net_epoch38.pth` (24+15=38, 因 HEAL 从 baseline epoch 23 续训)
- 训练日志无 fail/OOM
- 训练 wall 时间 < 35 min per triplet

#### 1.2.3 5 个 FT 档位的 AP eval (16 GPU·min)

对每个 (triplet, FT) 组合:
1. Export ONNX from `net_epoch{23+FT}.pth`
2. Build TRT FP32 engine
3. Run `e2e_eval_ap.py --n-samples 1789`

数据点 (15 个):
- T11 × FT={4, 6, 8, 10, 15}
- T17 × FT={4, 6, 8, 10, 15}
- T22 × FT={15} (已有 FT={2, 4, 6, 8, 10, 14, 18, 24})

可直接用 `scripts/phase2/a8_finetune_curve.py` 模式 (扩展 TARGETS).

**成功信号**:
- 15 个 `*_ap.json` 文件产生于 `/tmp/finetune_curve/`
- 所有 AP50 ∈ [0, 0.8] (sanity), 无 NaN

#### 1.2.4 决策表生成

输出 `/tmp/finetune_curve/spread_table.csv`:

|  | FT=4 | FT=6 | FT=8 | FT=10 | FT=15 |
|---|---|---|---|---|---|
| T11 (14% prune) | a₁₁ | a₁₂ | a₁₃ | a₁₄ | a₁₅ |
| T17 (57% prune) | b₁₁ | ... | ... | ... | ... |
| T22 (89% prune) | 0.520 | 0.558 | 0.543 | 0.556 | c₁₅ |
| **std** | σ₁ | σ₂ | σ₃ | σ₄ | σ₅ |
| **AP_max - AP_min** | Δ₁ | ... | ... | ... | ... |
| **constraint (a) AP[T11]>0.5** | bool | ... | ... | ... | ... |
| **constraint (c) AP[T11]-AP[T22]>0.05** | bool | ... | ... | ... | ... |

**FT* = argmax σ s.t. (a) ∧ (c)**

### 1.3 Phase 1 成功信号 (整体)

✅ Phase 1 完成判定:
1. `/tmp/finetune_curve/spread_table.csv` 已生成且包含所有 5×3=15 数据点
2. 至少 1 个 FT 满足 constraint (a) ∧ (c) (找到 sweet spot)
3. FT* ∈ [4, 20] 已确定并写入 `/tmp/finetune_curve/FT_decision.json`:
   ```json
   {
     "FT_chosen": <int>,
     "spread_std": <float>,
     "AP_T11": <float>,
     "AP_T17": <float>,
     "AP_T22": <float>,
     "constraint_a_satisfied": true,
     "constraint_c_satisfied": true,
     "reasoning": "<one sentence>"
   }
   ```

❌ Phase 1 失败判定 (任一即失败, 停止后续):
- 所有 5 个 FT 档都不满足 (a) ∧ (c) → 没有 sweet spot
- 任何 triplet 训练 fail/OOM
- spread σ < 0.03 across all FT → 任意 FT 都无信号, 需要 reframe 整个 plan

**失败时回退**: 标记 `phase1_failed.md` 记录原因, 不进 Phase 2.

---

## Phase 2 — 用 FT* 重训全 21 triplet (~2.5h on 6 GPU)

### 2.1 前置: Phase 1 已成功 (FT_decision.json 存在)

### 2.2 执行步骤

#### 2.2.1 复用现有 raw ckpts (T10-T22) + 重新 prune T1-T8

T10-T22 raw ckpts 在 Track A v2 时已生成或可重新 prune. 检查/补全:

```bash
# 应有 21 个 _raw 目录:
ls models/dataset_a_cache/ft_*_raw/  # 14 已有: T10-T22 + T22_p89
# 需补 T1-T8 (8 个) raw ckpt
```

补 T1-T8 (8 个 structural_prune, ~5 min):
- T1: (64, 128, 256) — 跟 baseline 一样, 不需要 prune. 直接用 baseline ckpt
- T2-T8: 跑 structural_prune_pyramid 各一遍

**成功信号**: 21 个 `ft_*_raw/net_epoch_bestval_at23.pth` 文件就绪.

#### 2.2.2 每 triplet finetune FT* epoch (6 GPU 并行, ~6 min × 4 batch ≈ 25 min)

写 `scripts/phase2/a9_fixed_ft_train.py`:
- 输入: FT_chosen (来自 FT_decision.json), 21 triplet list
- 每 triplet: 用 raw ckpt + train_ddp.py 训 FT* epoch, --half AMP
- 输出: 21 个 ft_dir 含 net_epoch{23+FT*}.pth

**成功信号**:
- 21 个目录都有 `net_epoch{23+FT_chosen}.pth` 文件
- 总 wall < 45 min (6 GPU 并行 4 批)
- 无失败 triplet

#### 2.2.3 重新 export ONNX (21 个) + AP eval × 7 Q (147 anchor)

类似 a5 phase 1 流程:
1. Export 21 ONNX (CPU, ~15 min serial 或 5 min parallel)
2. 对每个 (triplet, Q) 跑 fresh AP eval (覆盖 csv 原 reused_csv 值)
3. lat 测量保持不变 (现有 D-config × triplet × Q lat data 不动)

**成功信号**:
- `results/a2_d_expand/T*_Q_*_D1_default_4gb_ap.json` 全部更新 (21×7=147 个 fresh AP)
- AP50 值 std 跨 (T, Q) 显著大于 0.03 (现有 csv 水平)

### 2.3 Phase 2 成功信号 (整体)

✅ Phase 2 完成判定:
1. 21 triplet 全部用 FT* finetune 训完 (21 个 ckpt 就绪)
2. 147 个 (T, Q) fresh AP eval 已跑 (`*_ap.json` 时间戳更新)
3. AP50 分布跨 (T, Q):
   - **std > 0.05** (远大于 0.03 baseline)
   - **range > 0.20** (max-min)
   - 有 anchor AP < 0.30 (崩溃 anchor 存在)
   - 有 anchor AP > 0.50 (正常 anchor 存在)
4. 崩溃 anchor 与 prune % 正相关: 用 logistic regression 验 P(collapse | prune%) 有显著斜率 (p < 0.05)

❌ Phase 2 失败判定:
- 任何 triplet FT* finetune fail/OOM
- AP50 std ≤ 0.03 (说明 FT* 选错, 仍 over-finetune)
- 崩溃只发生在轻 prune (违反"崩溃需有理由"原则)

---

## Phase 3 — 重建 csv + 重训 predictor (~1h)

### 3.1 执行步骤

#### 3.1.1 csv 替换 AP 列

写 `scripts/phase2/a9_rebuild_csv_with_fixed_ft.py`:
- 读现有 csv (4704 anchor)
- 对每个 (triplet, q_tag), 用新 147 个 AP 值替换 csv 中所有 32 D 变体的 ap30/ap50/ap70
- d_tag 列、lat 列、resources 列保持不变
- 增加新列 `finetune_epochs` (全填 FT*)
- 备份原 csv → `e2e_bench_v1.csv.bak.pre_ft_pivot`

**成功信号**:
- csv 行数仍 4704
- 新列 `finetune_epochs` 存在且所有值 == FT*
- AP std > 0.05 in healthy subset

#### 3.1.2 重训 LGB predictor

跑 `scripts/phase2/predictor_efficiency_study.py`:
- 期望 f_AP_full R² **降至 0.6-0.8** (从虚高 0.96 → 真信号)
- 期望 f_lat R² **保持 ~0.66** (lat 不变)

**成功信号**:
- f_AP_full R² ∈ [0.5, 0.85] (说明有真信号, 不再虚高)
- AP 预测 MAE 增加但 R² 反映真实可学习性
- f_AP_baseline (无 Q, 无 D) R² > 0.20 (B 维度对 AP 有真实预测力, 不再 0.07 baseline)

### 3.2 Phase 3 成功信号 (整体)

✅ Phase 3 完成判定:
1. csv 备份 + 重建完成
2. predictor 重训完成, summary.json 更新
3. f_AP_full R² 在 [0.5, 0.85] 区间 (真信号)
4. 更新 `paper_learning/2. AAAI最终故事/data/数据集制作_plan.md` 至 v1.5 (含 fixed-FT 决策)
5. 更新 `paper_learning/2. AAAI最终故事/data/问题.md` §7 标 ✅ 已解决

---

## 整体成功信号 — /goal 终止条件

✅ **整体 plan 完成判定** (任一即成功结束):

1. **完整路径**: Phase 1 ✓ + Phase 2 ✓ + Phase 3 ✓, AP distribution 重建为有 variance 的形式, predictor 反映真信号.

2. **早停 (Phase 1 失败)**: spread σ < 0.03 across all FT ∈ [4, 15]. 这是有价值的发现 (确认 Pyramid Fusion 在任意 FT 下 B 维度都无信号), 论文 §C 需 reframe. 写入 `phase1_failed.md` 报告.

❌ **整体 plan 失败** (需人工介入):
- Phase 2 中任意 triplet FT* finetune 反复 OOM/fail
- Phase 3 后 predictor R² 仍异常虚高 (> 0.95) — 数据替换没生效
- 任意 phase 超过预算时间 2× (Phase 1 > 1.5h, Phase 2 > 5h, Phase 3 > 2h)

---

## 时间预算汇总

| Phase | 工作 | 预算 wall | GPU 占用 |
|---|---|---|---|
| 1 | Pilot (T11/T17 finetune 15 epoch + 15 AP eval) | ~50 min | 6 GPU |
| 2 | 21 triplet × FT* finetune + 147 fresh AP eval | ~2.5h | 6 GPU |
| 3 | csv 重建 + predictor 重训 + 文档更新 | ~1h | 1 GPU (LGB train) |
| **总计** | | **~4.5h** | 6 GPU 独占 |

---

## 关键脚本/路径清单

| 文件 | 状态 |
|---|---|
| `tools/structural_prune_pyramid.py` | ✅ 已有 |
| `tools/export_onnx_pyramid_e2e.py` | ✅ 已有 |
| `scripts/phase1/m4_8_trt_build_bench.py` | ✅ 已有 |
| `scripts/phase2/e2e_eval_ap.py` | ✅ 已有 |
| `scripts/phase2/a8_finetune_curve.py` | ✅ 已有 (Phase 1 模板) |
| `scripts/phase2/a9_fixed_ft_train.py` | 🔲 待写 (Phase 2.2.2) |
| `scripts/phase2/a9_rebuild_csv_with_fixed_ft.py` | 🔲 待写 (Phase 3.1.1) |
| HEAL `train_ddp.py` save-freq patch | 🔲 待改 (Phase 1.2.1) |

---

## 风险与回退策略

| 风险 | 概率 | 影响 | 回退 |
|---|---|---|---|
| Phase 1 无 sweet spot (任意 FT 都无 spread) | 中 | paper §C reframe | 写报告说明 "Pyramid+DAIR 无 B-维度 AP 信号" 这本身是论文级发现 |
| FT*=4 但 T11 < 0.50 (轻 prune 也崩) | 低 | 违反约束 (a) | 上调 FT* 到 6 或更高 |
| 21 triplet 中某些 finetune 不收敛 | 低 | 数据缺失 | 跳过该 triplet, 用剩余的 |
| predictor 重训后 f_lat R² 也降 (cross-time noise 仍存在) | 中 | 影响 paper §C | 不影响本 plan, 独立服务器到位后 multi-build agg 解决 |

---

## 验证清单 (执行中检查点)

- [ ] Phase 1.1: HEAL train_ddp.py patched, T11/T17/T22 raw ckpt 就绪
- [ ] Phase 1.2: 3 个 triplet 训完 FT=15, 15 个 ckpt 全部存在
- [ ] Phase 1.3: 15 个新 AP eval 完成, spread_table.csv 生成
- [ ] Phase 1.4: FT_decision.json 写入 FT* 及理由
- [ ] Phase 2.1: 21 个 raw ckpt 全部就绪
- [ ] Phase 2.2: 21 triplet × FT* 训练完成
- [ ] Phase 2.3: 147 个 fresh AP eval 完成, std > 0.05
- [ ] Phase 3.1: csv 备份 + 重建完成
- [ ] Phase 3.2: predictor R² ∈ [0.5, 0.85] (真信号)
- [ ] 文档更新: 数据集制作_plan.md → v1.5, 问题.md §7 标 ✅
