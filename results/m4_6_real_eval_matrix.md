# Framework 真实测试矩阵 — 三模型抽样验证 plan

**目的**: 把 framework 的 "predicted Pareto 前沿" 升级成 "实测验证过的 Pareto 前沿"

**核心问题**: 不能只测 baseline, 必须覆盖 framework 声称能找到的 trade-off 点 (剪枝 / 量化 / multi-module). 否则 framework 的 ranking 没有实证支撑.

**总工作量**: 30 个真实测试 (Pyramid 15 + uniad_tiny 10 + univ2x_full 5), 估 10-15 小时实测时间, 可分 5-7 天分批做.

---

## A. Pyramid Fusion (M4.6 完整版, 优先)

**已完成 (M4.6.0)**:

| # | config | precision | prune | lat_p50 | AP50 | 状态 |
|---|--------|-----------|-------|---------|------|------|
| 1 | pyramid_baseline | FP32 | none | 35.3 ms | 0.9635 | ✅ |
| 2 | pyramid_baseline | FP16 | none | 26.7 ms | 0.9631 | ✅ |

**待测 13 个 (M4.6.1-M4.6.3)**:

### A1. 量化维度 (M4.6.1, 验证 INT8 hook, ~3 个)

| # | config | precision | prune | 预期 lat | 验证什么 |
|---|--------|-----------|-------|---------|----------|
| 3 | pyramid_int8_ptq_static | INT8 | none | ~15 ms (估) | INT8 PTQ 是否 work + AP drop |
| 4 | pyramid_int8_dynamic | INT8 dynamic | none | ~20 ms (估) | dynamic 比 static 简单, 对比 |
| 5 | pyramid_mixed_int8_fp16 | mixed | none | ~22 ms (估) | per-module precision (encoder INT8 + heads FP16) |

**为什么抽样这 3**: 量化是 framework 主推 D 维度之一. INT8 PTQ 是工业最常用的, dynamic 是简单 fallback, mixed precision 验证 framework "per-module q_bits" 设计的实际意义.

### A2. 通道剪枝维度 (M4.6.2, 验证 prune hook, ~5 个)

| # | config | criterion | 模块 | rate | 预期 lat | 验证什么 |
|---|--------|-----------|------|------|---------|----------|
| 6 | pyramid_l1_25 | L1 | pyramid_backbone | 25% | ~32 ms | 低剪枝率, AP 应几乎不掉 |
| 7 | pyramid_l1_50 | L1 | pyramid_backbone | 50% | ~26 ms | 中剪枝率, 找 sweet spot |
| 8 | pyramid_l1_75 | L1 | pyramid_backbone | 75% | ~22 ms | 高剪枝率, AP 应明显跌 |
| 9 | pyramid_fpgm_50 | FPGM | pyramid_backbone | 50% | ~26 ms | 不同准则 vs L1 比较 |
| 10 | pyramid_l1_50_no_finetune | L1 | pyramid_backbone | 50% | ~26 ms | 不 finetune, 量化 AP 跌的 ablation |

**为什么抽样这 5**: 
- 0%/25%/50%/75% 跨度 → 看 framework Pareto 是否单调
- L1 vs FPGM → 验证 framework adapter 准则池设计 (Pyramid CNN 用 L1/FPGM, 不是 Taylor/Wanda)  
- finetune ablation → 验证训练成本是否必要

### A3. M4.7 框架搜索结果验证 (M4.6.3, ~5 个)

| # | config | 来自 | 验证什么 |
|---|--------|------|----------|
| 11 | m4_7_pyramid_0040 | M4.7 Pareto 唯一点 | INT8 + 80% encoder + 75% decoder + 70% heads |
| 12 | m4_7_pyramid_0013 | M4.7 channel-only #2 | 第 2 个 channel 剪枝候选 |
| 13 | m4_7_pyramid_0027 | M4.7 channel-only #3 | 第 3 个 |
| 14 | m4_7_pyramid_0001 | M4.7 非 Pareto, none + INT8 | **Negative control**: framework 说不在 Pareto 上, 实测应该确实差 |
| 15 | m4_7_pyramid_0010 | M4.7 非 Pareto, FP32 baseline | **Negative control**: 应该比 #1 (我们的 baseline FP32) 差 |

**为什么抽样这 5**:
- #11-13: framework 标的 "Pareto-optimal" 候选实测能否真在 Pareto 前沿 (核心 framework claim 验证)
- #14-15: **negative control** — framework 标的 "不是 Pareto" 候选, 实测应确实差. 否则 framework 的 ranking 就没意义.

---

## B. uniad_tiny variant (M5.7, 中优先)

**已有**: baseline_4090.parquet 有 23 行有 lat+params (Phase 1.2_P1_FFN + 1.3_d 实验), 但都没 AP. 需要实测 AP.

**待测 10 个**:

| # | config | 来自 | 验证什么 |
|---|--------|------|----------|
| 1 | uniad_tiny_baseline | stage5_v3 plan_b_active | baseline AP anchor |
| 2 | uniad_tiny_FFN_50 | 1.2_P1_FFN | FFN 剪 50% (Transformer 子网准则) |
| 3 | uniad_tiny_FFN_75 | 1.2_P1_FFN | FFN 剪 75% |
| 4 | uniad_tiny_d_GPU_only | 1.3_d_routing | D 全 GPU |
| 5 | uniad_tiny_d_DLA_split | 1.3_d_routing | D 部分 DLA (验证 D-routing) |
| 6-7 | M5.6 Pareto 2 个 | m5_6_uniad_tiny_0018 (channel) + 0004 (none) | framework Pareto 验证 |
| 8-9 | M5.6 非 Pareto 2 个 | 随机选 | **Negative control** |
| 10 | uniad_tiny_int8_full | 1.2 quant | INT8 全模型 |

**为什么这样抽样**:
- 利用 Phase 1.2_P1_FFN / 1.3_d 已实验过的 configs (我们已知 latency, 只缺 AP), 抽样代价小
- 测 D-routing 实证 (DLA vs GPU 真实 latency 差距)
- M5.6 Pareto + 非 Pareto 对比 (跟 Pyramid 同样设计)

---

## C. univ2x_full (M1.4, 低优先)

**已有**: baseline_4090.parquet 22 行, 22/22 amota, 3/22 lat. 缺 e2e PyTorch FP16 在 4090 上的真实测.

**待测 5 个** (univ2x_full ckpt 大, finetune 慢, 抽样必须小):

| # | config | 来自 | 验证什么 |
|---|--------|------|----------|
| 1 | univ2x_full_baseline_pytorch_fp16 | 1.1_quant 已有但没 lat | baseline 第 4 个 lat anchor (除了 90/5640 极值) |
| 2 | univ2x_full_int8_trt_full | 1.1_quant | INT8 TRT 全模型 |
| 3 | A7 Pareto 1 | phase1a_pareto.csv is_pareto=True 第 1 个 | framework Pareto 验证 |
| 4 | A7 Pareto 2 | 第 2 个 | 同 |
| 5 | A7 非 Pareto | **Negative control** | 验证 LGB v3/v4 预测排序的可靠性 |

**为什么 univ2x_full 抽样最小**:
- ckpt 巨大 (R101 + DCN + 200x200 BEV), 跑一次 e2e ~5640ms FP32
- finetune 单 config 要 1-2 天 4090 (vs Pyramid 几小时)
- 但 univ2x_full 已经是我们 Phase 1A/1B/2.4 的主战场, 数据基础好, 不必再大量抽样

---

## 抽样设计的核心原则

1. **覆盖 prune rate 全范围** (0%/25%/50%/75%): 验证 framework Pareto 单调性
2. **量化 axis 全覆盖** (FP32/FP16/INT8/mixed): 验证 D 维度 framework 设计
3. **Pareto 前沿候选** vs **非 Pareto 候选** 配对: framework 的 ranking 必须可证伪
4. **跨 model 一致测试** (Pyramid + uniad_tiny + univ2x_full): 验证 framework 迁移性是真的
5. **利用已有数据** (1.1/1.2/1.3 实验已有 lat 但缺 AP) 降低重测成本
6. **Pareto 抽样优先于密集网格**: 30 个点足以验证 framework, 不必 50+50+50

---

## 时间表

| 阶段 | 内容 | 估时 | 产出 |
|------|------|------|------|
| M4.6.1 (今/明) | A1 量化 (3 个 configs) | 4-6h | INT8 PTQ + AP |
| M4.6.2 (1-2 天) | A2 通道剪枝 (5 个 configs, 含 finetune) | 1-2 天 | L1/FPGM 实测 |
| M4.6.3 (1 天) | A3 M4.7 framework 验证 (5 个 configs) | 1 天 | framework Pareto 真实性 |
| M5.7 (1 天) | B uniad_tiny 抽样 (10 个) | 1 天 | 跨模型一致性 |
| M1.4 (2 天) | C univ2x_full 抽样 (5 个) | 2 天 | univ2x_full 实测 anchor |

**总计**: ~5-7 天可完成所有 30 个抽样实测, 之后 framework Pareto 真实性可信.
