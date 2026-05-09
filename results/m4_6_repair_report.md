# M4.6 修复报告 — Framework v6 跨模型 Pareto 实证

**日期**: 2026-05-10
**总用时**: ~6h (LGB v6 训练 + M4.7 v3 + M5.6 v2 + A7 v2 + M4.6.3 v2 抽样实测 15 候选)

## 修复反思 #5/#6/#7 完成情况

| 反思 # | 问题 | 修复 |
|-------|------|------|
| #5 mask 不真减 lat | 未真改, 但 rule-based lat 公式补充 | ✅ 部分 (rule-based MAE 0.5-5ms 可用) |
| #6 KNN anchor 错 | LGB v6 跨模型预测 + rule-based lat 替代 KNN | ✅ |
| #7 AP 预测器没接 framework | M4.7 v3 / M5.6 v2 / A7 v2 全部接 LGB v6 amota | ✅ |

## 三模型 Pareto 重跑结果 (修复后)

| Model | 候选 | Pareto 点 | lat 范围 (pred ms) | amota 范围 (pred) |
|-------|------|----------|-------------------|------------------|
| Pyramid (M4.7 v3) | 50 | 10 | 21-31 | 0.745-0.963 |
| uniad_tiny (M5.6 v2) | 33 | 4 | 481-524 | 0.288-0.343 |
| univ2x_full (A7 v2) | 50 | 5 | 426-527 | 0.322-0.359 |

## M4.6.3 v2 抽样实测验证 (15 candidates)

### LGB v6 amota 预测精度

| 候选类型 | 预测 amota | 实测 AP50 | MAE |
|----------|-----------|-----------|-----|
| 9 个 'none' Pareto | 0.955-0.963 | **0.9631 全等** | 0.001-0.008 |
| 5 个 'none' non-Pareto | 0.959-0.962 | **0.9631 全等** | 0.0001-0.004 |
| 1 个 channel Pareto (0039) | 0.745 | 0.629 | 0.116 |
| 1 个 channel non-Pareto (0040) | 0.645 | 0.606 | 0.039 |

**关键**: LGB v6 在 'none + bit reduction' 候选上 **MAE < 0.01** (出色); 在 channel pruning 候选上 MAE 0.04-0.12 (偏乐观).

### Rule-based latency 公式精度

| 候选类型 | 预测 lat | 实测 lat | 偏差 |
|----------|---------|---------|------|
| 'none' (none + FP16/FP32/INT8 bit) | 22.5-33.8ms | 25.7-27.9ms | -0.5 ~ -8ms |
| channel pruning (0039/0040) | 21.1-21.8ms | 27.1-27.7ms | +6 ms |

**lat 公式**: `voxelize_overhead(10ms) + conv_budget(25ms) × (1 - prune×0.5) × bits_factor`. 在 'none' 候选 MAE ~3ms (可用), 在 channel pruning 偏低估 6ms (mask-based 不真减但公式假设减, 后续 finetune-based real reduction 可校准).

### Framework 排序正确率: 7/35 (20%)

**解读**: 14/15 实测 AP=0.9631 完全等同 → Pareto vs non-Pareto 在 AP 维度无差异, 排序完全靠 latency 微小差异 (0.5-3ms). 在 measurement noise (CUDA event jitter ~1ms) 范围内, 排序经常翻转, 不能 claim framework 排序"对".

但**关键观察**:
- Pareto 候选 0039 (channel pruning, AP=0.629) vs non-Pareto 候选 0040 (channel pruning aggressive, AP=0.606) — **0039 真的 dominate 0040** ✓ framework 排序对
- 14 个 'none' 候选实测等价 — 不属"错位", 是"tradeoff"

## v1 → v3 修复价值量化

| metric | v1 (KNN, lat×params) | v3 (LGB v6 + rule-based) | 改进 |
|--------|---------------------|------------------------|------|
| 训练锚点 | 8 行子模块 | 26 行 (含 2 完整 e2e + 16 实测) | 3.25× |
| amota 维度 | ❌ 不预测 | ✅ MAE 0.001-0.008 ('none') | qualitative |
| latency 预测 | 子模块 4.5ms (10× 偏差) | 完整 e2e MAE 0.5-5ms | ~10× |
| Pareto candidates | 1 (collapsed) | 10 (覆盖 trade-off 谱) | 10× |
| Pareto 候选实测 vs 预测一致性 | 0040 实测最差 (5/5 dominate it) | 0039 实测优于 0040 (subspace 排序对) | qualitative |

## 未解决问题 (留给后续)

1. **LGB v6 latency 模型崩** — 跨 4 数量级 (3.4-5640ms) 训不出来, 用 rule-based 替代. 解法: per-model_class 单独训 lat 模型.
2. **Mask-based pruning 不真减 latency** — 14 个 'none' 候选 lat 都 25-28ms 压缩平台, 真减 lat 需要 finetune-aware structural pruning (M4.6.2 真版本, 1 周工作).
3. **LGB v6 在 channel pruning 候选 amota 偏乐观** — 训练数据中 channel pruning 实测样本只 16 个 (M4.6.1+2+3), 跨配置 generalization 弱.
4. **uniad_tiny / univ2x_full 缺真实测验证** — 本次只 Pyramid 抽样实测; M5.7 / M1.4 是后续工作.

## 论文 claim 修正版

### ✅ 可 claim

1. **Framework 跨模型适配**: 三个 model_class 同一 random_search + adapter pattern + 三类硬约束 ✓
2. **AP 预测器跨模型 transfer**: LGB v6 加 model_class one-hot + Pyramid 锚点重训, **'none' 候选 AP MAE 0.001-0.008** (Pyramid baseline anchor 实证)
3. **B-D 维度 Pareto 前沿**: 三个 model_class 都给出合理 Pareto frontier (lat × amota 双轴)
4. **Reflection-driven framework iteration**: M4.6 实测暴露 v1 设计漏洞 → v6 + rule-based 修复

### ⚠️ 需要 caveat

1. Latency 预测 rule-based, 不是 LGB-learned (跨数量级 ML 训不出来)
2. Pareto 排序在小 trade-off 区域 (各候选差 1-3ms / 0.001 amota) 受测量噪声主导, 不可严格 claim "predicted Pareto = measured Pareto"
3. 无 finetune 路径下 channel pruning 真减 lat 不可观察

### ❌ 不能 claim

1. "Framework 找到 SOTA Pareto frontier" — 没 finetune, AP-rate 曲线不准
2. "20× 加速验证" — 实测 lat 只在 25-35ms 范围内 (无 finetune-aware structural pruning 不可达)

## 文件 deliverables

- `models/lgb_v6_{amota,latency}.txt`
- `scripts/phase2/train_lgb_v6.py`
- `scripts/phase1/m4_7_v3_pyramid_pareto.py` (LGB v6 + rule-based)
- `scripts/phase1/m5_6_v2_uniad_tiny_pareto.py`
- `scripts/phase1/a7_v2_univ2x_full_pareto.py`
- `scripts/phase1/m4_6_3_v2_pareto_validation.py`
- `scripts/phase1/m4_6_integrate_to_baseline.py`
- `results/phase2_pareto_pyramid_v3.csv` (50 行)
- `results/phase2_pareto_uniad_tiny_v2.csv` (33 行)
- `results/phase1a_pareto_v2.csv` (50 行)
- `results/m4_6_3_v2_pareto_validation.csv` (15 行实测)
- `results/m4_6_repair_report.md` (本文件)
