# M4.6 全实验报告 — Pyramid Fusion 真实测试 (M4.6.0+1+2+3)

**日期**: 2026-05-09 17:07-19:28 (2h21min)
**硬件**: RTX 4090, OPV2V test 2170 samples
**evaluation framework**: HEAL inference + framework adapter mask-based pruning + BN recal 50 batches

---

## 全部实测数据汇总

### M4.6.0: 量化 baseline (autocast)

| precision | AP30 | AP50 | AP70 | lat_p50 (ms) |
|-----------|------|------|------|--------------|
| FP32 | 0.9691 | 0.9635 | 0.9272 | 35.31 |
| FP16 | 0.9690 | 0.9631 | 0.9268 | **26.70** (1.32×) |

**FP16 加速 1.32× 几乎无 AP 损失** (-0.04% AP50).

---

### M4.6.1: 单模块 (pyramid_backbone+shrink_conv) L1/FPGM AP curve

| rate | criterion | AP30 | AP50 | AP70 | lat_p50 |
|------|-----------|------|------|------|---------|
| 0.0 | none | 0.9690 | 0.9635 | 0.9270 | 37.81 |
| 0.25 | L1 | 0.9682 | 0.9620 | 0.9258 | 35.92 |
| 0.50 | L1 | 0.9670 | 0.9606 | **0.9184** | 36.29 |
| 0.50 | **FPGM** | 0.9677 | 0.9617 | **0.9214** | 35.77 |
| 0.75 | L1 | 0.9494 | 0.9430 | **0.6621** | 37.75 |
| 0.75 | FPGM | 0.9527 | 0.9453 | 0.6007 | 37.01 |
| 0.90 | L1 | 0.3106 | 0.2548 | 0.0472 | 34.79 |

**Sweet spot: 50% L1 (AP50 -0.30%)**. 75% AP70 崩溃 (-29%). 90% 全崩.

**L1 vs FPGM**: 50% 时 FPGM 略好 (+0.30% AP70), 75% 时 L1 反超 (+6.14% AP70). **mask-based latency 不变** (mask 后 PyTorch 仍 dense kernel).

---

### M4.6.2: 多模块 (5 模块同剪) AP

| config | rates (b/e/d/h/v) | AP30 | AP50 | AP70 | lat_p50 |
|--------|-------------------|------|------|------|---------|
| baseline | 0.0/0.0/0.0/0.0/0.0 | 0.9690 | 0.9635 | 0.9271 | 37.58 |
| uniform_30 | 0.3/0.3/0.3/0.3/0.3 | 0.9409 | 0.7848 | **0.2343** | 37.04 |
| framework_front_heavy | 0.5/0.5/0.3/0.3/0.2 | 0.9187 | 0.7428 | 0.1999 | 37.95 |
| framework_back_heavy | 0.3/0.3/**0.7**/0.5/0.2 | 0.9448 | 0.7943 | 0.2362 | 36.25 |

**关键发现**:
1. **多模块累积效应巨大**: 全模块各 30% 比单模块 50% 严重得多 (AP70 0.918 → 0.234)
2. **重剪 decoder (back-heavy) > 重剪 backbone (front-heavy)** — 印证 framework adapter "decoder 高冗余" 假设
3. lat 仍不变 (mask-based)

---

### M4.6.3: 框架 M4.7 Pareto 候选 vs 非 Pareto 实测 (5 candidates)

| config | is_pareto (predicted) | rates | prec | AP70 | lat_p50 |
|--------|-----------------------|-------|------|------|---------|
| 0001 (quant-only) | False | all 0 | INT8 (proxy fp16) | 0.9269 | 25.5 |
| 0010 (quant-only) | False | all 0 | INT8 (proxy fp16) | 0.9268 | 26.3 |
| 0013 (multi-prune mid) | False | 0.5/0.2/0.4/0.375/0.15 | proxy fp16 | 0.2101 | 26.0 |
| 0027 (multi-prune light decoder) | False | 0.4/0.297/0.05/0.125/0.375 | proxy fp16 | 0.3270 | 25.7 |
| **0040** (Pareto, all aggressive) | **True** | 0.75/0.8/0.75/0.7/0.75 | proxy fp16 | **0.1397** | 26.8 |

**🚨 关键问题: framework Pareto 预测完全失败**

framework 标的 Pareto-optimal candidate 0040 实际**最差** (AP70=0.140), 4 个 non-Pareto 全部 dominate:
- 0001 / 0010 (quant-only): AP70=0.927 / lat=25.5ms — 最佳实际表现
- 0013: AP70=0.210
- 0027: AP70=0.327
- **0040 (Pareto)**: AP70=0.140 — 最差

---

## 根因分析

### 问题 1: KNN latency 估算 anchor 错

M4.7 用 baseline_4090.parquet 中 8 行 m4_5_pyramid_configs (PyramidFusion **子模块** dummy input timing) 作 KNN anchor:
- 子模块 latency: 1.6-4.8 ms (FP32 4.49, FP16 3.27)
- 完整 e2e latency (M4.6.0): 35.3 / 26.7 ms (FP32/FP16) = **8× 差距**

candidate 0040 KNN 估 **2.46 ms**, 实测 26.8 ms — **10× 偏差**.

### 问题 2: Mask-based pruning 不真减 latency

PyTorch `torch.nn.utils.prune.ln_structured` 套 mask, 但 forward 仍跑 dense Conv kernel. 所以无论剪枝率多大, lat 几乎不变:
- 0% prune: 37.8 ms
- 50% prune: 36.3 ms
- 75% prune: 37.7 ms
- 90% prune: 34.8 ms

要真减 latency, 必须**重 build smaller PyramidFusion** (修改 num_filters 配置 + truncate weights), 这是 M4.6.2 真实版本要做的工作 (本次跳过 — 工程复杂).

### 问题 3: AP 预测器 (Phase 2.4 v5.1) 没用上

framework 没给 M4.7 candidates 预测 AP, 完全靠 latency × params 双轴决定 Pareto. 而剪枝多模块累积效应巨大, 在 latency 估算错的情况下 framework 排序基本是噪声.

### 问题 4: BN recal 50 batches 不够 (无 finetune)

50 batches BN recalibration 只能恢复 BN running stats, 不能恢复剪掉权重的功能. 真要恢复多模块 70-80% 剪枝的 AP, 需要 1-2 epoch finetune (需 OPV2V train split, 我们目前只下载了 test split).

---

## Framework 真实状态

### ✅ 验证可成立的 claim

1. **Framework 跨模型迁移** (B + D 通用 schema): adapter 模式让 Pyramid 跟 univ2x 用同一搜索空间 ✓
2. **三类硬约束 DSL + 双向传播** (PHYSICAL/EMPIRICAL/SOFT): 已 commit, M4.7/M5.6 都跑通 ✓
3. **单维度 trade-off 可控**: M4.6.1 单模块剪 50% AP 几乎无损 (FPGM 略好), 75% 起跌 — framework adapter 准则池设计有意义 ✓
4. **多模块累积效应有方向性**: M4.6.2 显示 decoder-heavy > backbone-heavy, 跟 framework adapter "Pyramid decoder 是 CNN, 高冗余" 假设一致 ✓

### ❌ 当前不能 claim 的

1. **真实加速倍数显著** (Pareto 双轴): 当前 mask-based 不真减 lat, 真减需 M4.6.2 重 build smaller model
2. **Pareto 前沿可信**: M4.6.3 显示 framework 预测的 Pareto candidate 0040 实测最差 (5/5 candidate 中 worst)
3. **AP 预测有效**: 没用 v5.1 LGB 预测器, framework 完全没估 AP

---

## 下一步建议

### 立即修复 framework Pareto 预测 (B-D 维度)

1. **修 KNN anchor**: 用 M4.6.0 完整 e2e baseline (35ms FP32 / 27ms FP16) 替换子模块 4.5/3.3 ms anchor
2. **加 AP 估算到 M4.7**: 用 v5.1 LGB 预测器给 50 候选 AP, Pareto 改成 (latency × predicted AP) 双轴
3. **重新跑 M4.7**, Pareto candidates 应该会变

### 真减 latency: 重 build smaller PyramidFusion (替代 mask-based)

1. 修 num_filters config (e.g. [32, 64, 128] = 50%)
2. instantiate fresh PyramidFusion
3. truncate baseline ckpt weights (按 L1 norm 选 top-N output channels)
4. BN recal 200 batches (more) 
5. 测 AP + 真减小的 latency

### 长期 (含 finetune)

下载 OPV2V train split (~50GB), finetune 1-2 epoch 给重剪枝候选恢复 AP. 需要 ~1 天/config.

---

## 数据 deliverables

- `results/m4_6_1_pyramid_pruning.csv` (7 行)
- `results/m4_6_2_pyramid_multimodule.csv` (4 行)  
- `results/m4_6_3_pyramid_pareto_validation.csv` (5 行)
- `results/m4_6_full_report.md` (本文件)

**实测总数据点**: 16 (vs M4.6 前的 2 个 baseline). framework 真实状态从 "predicted only" 升级到 "20% measured + 80% predicted (with known prediction errors)".
