# 搜索空间总规模报告 v1.0

> 角色: 数据生成师 (Search-Space & Data Generation Engineer) 交付物 1
> 输入: 三份可配置维度清单 (`methods/design/dims_pruning_v1.md` / `dims_quantization_v1.md` / `dims_hardware_v2.md`;硬件清单 v1 已被 v2 取代, 原文存 `multi_agent/archive/`) + 共享契约 `framework/config_schema.py` + 搜索器 `framework/searcher_v0.py`
> 数字来源标注: [清单]=维度清单枚举 / [实测]=searcher_v0 真实统计 / [估算]=笛卡尔积推算 / [覆盖]=已有真测数据
> 生成日期: 2026-05-31

---

## 0. 一句话总览

搜索空间 = **B 剪枝 × Q 量化 × D 部署**, 取值域由**目标硬件 capability 门控**。
- **Pyramid (单模块 `model`)**: 4090 **4.5 万**档 / Orin **11.5 万**档 [估算]。
- **UniV2X (5 模块)**: 量级 **10^15–10^16** [估算] (5 模块独立组合 → 维度爆炸)。
- 经 searcher_v0 约束过滤后**有效规模**: UniV2X pass_rate ≈ 54–65% [实测], Pyramid 单模块 pass_rate ≈ 0.05–0.3% [实测] (单模块下 `channel_align_32` 硬约束几乎一票否决, 见 §4)。
- **当前真测覆盖**: `e2e_bench_v1.csv` 4704 行 (8 triplet × Q × D 网格, 全 4090), 但**全部是 prune+quant+D-cell 的笛卡尔展开**, 真实独立 anchor ≈ 32 (8 triplet × 4 精度)。本报告 DoE 在此之上加 INT8/percentile/W-only/跨硬件锚点。

---

## 1. 各维度取值域 (据三份清单)

### 1.1 B 剪枝 (`dims_pruning_v1.md`)
| 维度 | 字段 | 取值数 (有效) | 来源 |
|------|------|--------------|------|
| D1 prune_rate | `prune_rate[m]` | 15 档 (searcher `COMBINED_PRUNE_RATES`: 7 对齐 + 8 近对齐) | [清单] §2 / searcher L45 |
| D2 prune_object | `prune_object` (全局) | 4090: 3 (channel/2:4/none); Orin: 2 (channel/none, sparse_tc=False) | [清单] §2 + 硬件 sparse_tc |
| D3 criterion | `prune_criterion[m]` | 4 (L1/Taylor/FPGM/Wanda; none 仅 rate=0) | [清单] §2 |
| D4 round_to | (派生 q_bits) | 2 (INT8→32, FP16/32→8) | [清单] §2 D4 |

> element (非结构化) 不上 TRT 已剔除; 2:4 仅 sparse tensor core 硬件 (4090 sparse_tc=True, Orin=False [实测])。

### 1.2 Q 量化 (`dims_quantization_v1.md`)
| 维度 | 字段 | 取值数 (有效) | 来源 |
|------|------|--------------|------|
| Q0 bits | `q_bits[m]` | 3 (INT8/FP16/FP32) | [清单] Q0 |
| Q2 gran(W) | `q_granularity[m]` | 2 (per-channel/per-tensor; none 仅 FP32) | [清单] Q2 |
| Q3 object | `q_object[m]` | 2 (W+A/W-only; none 仅 FP32) | [清单] Q3 |
| Q4 calibrator | `q_calibrator[m]` | 2 (minmax/percentile_99_99; **entropy 禁用**, none 仅非 INT8) | [清单] Q4 |

> 单模块有效量化组合 = INT8(2×2×2=8) + FP16(1) + FP32(1) = **10 档** [估算]。
> activation 恒 per-tensor (硬约束, 不计入组合); per-group(AWQ) 仅 4090 TRT10, 当前未纳入枚举。

### 1.3 D 部署 (档位计数推导见 `archive/dims_hardware_v1.md` §6;现行维度定义以 `dims_hardware_v2.md` 为准, 其 §0.5 可搜性裁决指出 tactic/workspace 应交 TRT-auto, 真·可搜 D 维 = A1 routing + batch)
| 平台 | 路由 D2 | tactic D4 | workspace D5 | D 档数 |
|------|--------|-----------|--------------|--------|
| RTX 4090 | 1 (GPU only) | 5 | 5 | **25** [清单] |
| Orin AGX | 6 (GPU+DLA0/1×fallback) | 4 | 4 | **96** [清单] |

> precision/granularity (D6/D8) 由 Q 维度固定, 不在 D 本体重复计数 (清单 §6 口径)。

---

## 2. 笛卡尔积总规模 (按模型 × 硬件)

> 公式: **总规模 = (B 剪枝组合) × (Q 量化组合) × (D 部署档数)**。
> 单模块 B×Q (粗上界, 未约束剪枝) = prune_rate(15) × prune_object(n) × criterion(4) × Q组合(10)。

### 2.1 Pyramid (单模块 `model`, PYRAMID_M1_MODULES)

| 平台 | B×Q (单模块) | × D | **总笛卡尔积** [估算] |
|------|-------------|-----|----------------------|
| RTX 4090 | 15×3×4×10 = 1,800 | ×25 | **45,000** |
| Orin AGX | 15×2×4×10 = 1,200 | ×96 | **115,200** |

> 与 CLAUDE.md / 维度清单 "4090 ~25 / Orin ~96 D 档" 口径一致 (D 本体), 乘上 B×Q 后得全空间。

### 2.2 UniV2X (5 模块 backbone/encoder/decoder/heads/v2x_comm)

5 模块的 prune_rate/criterion/bits/gran/obj/calib **各自独立** (per-module dict), prune_object 全局:
单模块 B×Q (无全局 prune_object) = prune_rate(15) × criterion(4) × Q组合(10) = 600。

| 平台 | (单模块 600)^5 × prune_object × D | **总笛卡尔积** [估算] |
|------|----------------------------------|----------------------|
| RTX 4090 | 600^5 × 3 × 25 | **≈ 5.8 × 10^15** |
| Orin AGX | 600^5 × 2 × 96 | **≈ 1.5 × 10^16** |

> 这是"搜索器为什么需要"的量化论据: UniV2X 全空间 10^15-10^16, 不可枚举, 必须代理模型 (LGB) + 多目标搜索 (NSGA-II/BoTorch)。

---

## 3. 约束过滤后的有效规模 (searcher_v0 实测 pass_rate)

> 用 `framework/searcher_v0.random_search` 实跑, 采样 → 双向传播 → `is_legal_for_hardware` 过滤, 统计 pass_rate。
> 命令: `random_search(hw, n_candidates=500, max_attempts=50000, seed=1)`。

| 模型 | 平台 | pass_rate [实测] | propagate 改写率 [实测] | 解读 |
|------|------|-----------------|------------------------|------|
| Pyramid | RTX 4090 | **0.05%** | 0.0% | 单模块 + hard alignment → `channel_align_32` 几乎一票否决 (见 §4) |
| Pyramid | Orin AGX | **0.29%** | 55.4% | soft alignment 放松, 但单模块去重后独立空间仍小 (collected 仅 144) |
| UniV2X | RTX 4090 | **64.6%** | 0.0% | 5 模块大空间, 合法组合充裕 |
| UniV2X | Orin AGX | **54.1%** | 97.8% | DLA 路由触发大量传播改写 (跨 IP 精度归一) |

**有效规模估算** = 总笛卡尔积 × pass_rate:
- Pyramid 4090: 45,000 × 0.0005 ≈ **~23 个独立合法配置** (与 searcher 去重后 collected=25 吻合 [实测])。
- Pyramid Orin: 115,200 × 0.0029 ≈ **~330**。
- UniV2X 4090: 5.8e15 × 0.646 ≈ **3.8 × 10^15** (仍不可枚举)。
- UniV2X Orin: 1.5e16 × 0.541 ≈ **8.1 × 10^15**。

---

## 4. 关键发现: Pyramid 单模块 pass_rate 崩塌 (给搜索器 owner)

searcher_v0 对 Pyramid 单模块采样时, 从 `COMBINED_PRUNE_RATES` (含 8 个**近对齐**率如 0.05/0.297/0.55) 抽一个全局 rate。单模块 → 这个 rate 直接决定唯一 `model` 的通道数。4090 `alignment_enforcement=hard` 下, 近对齐率剪后通道数非 32 倍数 → `[empirical/channel_align_32]` 约束否决 (50000 次采样里 ~35000 次因此失败 [实测])。

含义:
1. **Pyramid 真实可搜空间极小** (~23 个 4090 合法点), 这正是为什么 Pyramid 适合**穷举式真测 DoE** (本报告 §交付物 2 的 25 anchor 已接近覆盖核心), 而 UniV2X 必须靠代理模型。
2. 搜索器在单模块场景应**只从对齐网格 `ALIGNED_PRUNE_RATES` (7 档) 采样** (跳过近对齐), pass_rate 会从 0.05% 提升到接近 prune_object 合法比例。这是给 framework owner 的优化建议 (本报告不改 searcher)。
3. Orin soft alignment 让近对齐率部分通过 (propagate 改写率 55%), 印证 dims_hardware §D13 "soft 让搜索空间 ~30% 解放"。

---

## 5. 当前真测覆盖率

| 数据源 | 行数 | 独立 anchor | 模型 | 硬件 | 维度覆盖 |
|--------|------|------------|------|------|---------|
| `data/e2e_bench_v1.csv` | 4704 | ~32 (8 triplet × 4 精度) | Pyramid DAIR | 4090 | B(8 档剪枝) × Q(FP32/FP16/INT8mm/INT8ent) × D(部分网格展开) |
| `data/baseline_4090.parquet` | 83 | ~30 | 三模型 spectrum | 4090 | M4.6 mask-based (latency 同质, 已知失败) |

**覆盖率估算** (独立 anchor / 有效规模):
- Pyramid 4090: 32 真测 anchor / ~23 有效配置 → 核心空间**已基本覆盖** (真测 anchor 含 entropy 等已被约束排除的点)。
- Pyramid Orin: **0 真测** (跨硬件未覆盖, 本 DoE §cross_hw 补 DLA 锚点)。
- UniV2X: **0 真测** (本阶段聚焦 Pyramid; UniV2X 留代理模型阶段)。

**未覆盖的关键边角** (本 DoE 补齐):
- INT8 percentile_99_99 校准器 (e2e_bench 只有 minmax/entropy)。
- INT8 W-only / weight per-tensor 粒度轴 (灵敏度分层缺这两个 OAT 点)。
- 跨硬件 Orin DLA 路由 (latency 预测器跨硬件外推的配对样本)。
- 剪枝 × INT8 交互拐点 (p25/p50/p75 × INT8) 的真实 build 数据 (受 GPU 占用限制, 当前 dry_run, 见 §交付物 4)。

---

## 6. 交付物路径

- 本报告: `paper_learning/2. AAAI最终故事/data/search_space_size_v1.md`
- DoE 设计 + anchor 清单: `paper_learning/2. AAAI最终故事/data/doe_design_v1.md` + `tools/configurable/doe_anchors.py`
- 生成管线: `tools/configurable/generate_dataset.py`
- 生成数据 (dry_run + prune_real): `output/doe_dataset_v1/doe_dataset_v1.csv` + `output/doe_dataset_v1/manifests/`
