# LGB 预测器训练数据集制作计划

> **关联文档**:
> - 搜索空间标准: `paper_learning/2. AAAI最终故事/搜索空间一览.md`
> - 已实验数据汇总: `paper_learning/2. AAAI最终故事/result/Plan2_完整实验结果.md` + `Phase1_5_完整通宵汇报.md` + `Plan2_PhaseAP_5D_Pareto.md`
> - 反思与设计纪律: `paper_learning/2. AAAI最终故事/reflection_mistakes.md`(尤其 #19-#31)
> - 整体实验路线: `paper_learning/2. AAAI最终故事/00_故事评估与实验路线_v1.md`
>
> **当前版本**: v1.1 (2026-05-13) — 加 §〇.5 旧数据可用性诊断 + §〇.6 ABCD 推进现状审计
> **状态**: P0.1 (8 ckpt 库) + P0.2 (144 anchor 主网格) 已完成, **Class A 缺口 176 anchor, B/C/D 未启动**

---

## 〇、为什么要做这份计划

当前"345 行真测"覆盖搜索空间 ~562,666 cell, **覆盖率 0.06%**, 且严重不均衡:

| 缺口 | 严重度 | 影响 |
|------|--------|------|
| **AP × D 维度联测真值 = 0** | 🔴 致命 | AP predictor 完全感知不到 D 维度 (paper §C Pareto demo 的 AP 轴是预测+估算, 不是真测) |
| **AP × Orin 真值 = 0** | 🔴 致命 | 跨硬件 AP predictor 全靠"4090 AP = Orin AP"假设 |
| **AP 真测覆盖 B 中间档 = 0** | 🟡 严重 | 当前 8 个 anchor 全在 baseline / p25 / p50 / p75 标准 ratio 上, 中间 (40,72,128) 等无 |
| **mixed precision × AP = 0** | 🟡 严重 | Part A 测了 18 个 per-stage lat 但都没 AP |
| **跨模型 LGB 缺充足正样本** | 🟢 中等 | baseline_4090.parquet 83 行 amota 跨 3 model 但样本质量不均 (反思 #22) |

加上反思 #23/24/25/28:
- **训练数据必须 random_search 均匀采样**, 不能 hand-pick (selection bias)
- 单模型至少需要 ~80 anchor (~20% 覆盖) 让 LGB CV MAE 接近 in-sample MAE
- **closed_loop demo 必须用 random_search + LGB**, 不是 enumeration

→ 必须**有目的地**补一批高质量真测数据.

---

## 〇.5、关于"345 行旧数据可用性"的严肃诊断 (新增 2026-05-13)

> 不能再含糊说"已有 345 真测". 用户问及"这些旧数据本来含 AP 吗", 答案是**绝大多数没有**. 必须明确列出每个 parquet 的可用范围, 否则后续 LGB 训练会重蹈"用 lat-only 数据骗自己有 AP 真值"的坑.

### 〇.5.1 按"对 LGB 多目标 (lat+AP+throughput+resource) 训练的有效性"分级

| 文件 | 行数 | 有 AP 真测? | AP 可用? | lat 可用? | 跨硬件? | 给 LGB 的角色 |
|------|------|------------|----------|-----------|---------|--------------|
| `data/stage_a_ap_real.parquet` | 8 | ✅ 是 (DAIR 1789) | **✅ 唯一完整 AP 真值** | ✅ | 4090 | **AP predictor 唯一可信训练源**, 但 8 行不足以单独训练 |
| `data/stage_b_ap_real.parquet` | 4 | ✅ 是 但 AP≈0.06 | ❌ negative (finetune 不足导致崩塌) | ✅ | 4090 | 仅作"未收敛 ckpt → AP 崩"的负样本, **不进主训练集** |
| `data/pyramid_random_bench.parquet` | 100 | ❌ | NaN | ✅ | 4090 (单 D) | 仅 lat predictor 可用 (AP 列必须 NaN) |
| `data/perstage_quant_bench.parquet` | 18 | ❌ | NaN | ✅ | 4090 (mixed Q) | 仅 lat predictor 可用 |
| `data/4090_dspace_bench.parquet` | 48 | ❌ | NaN | ✅ | 4090 (5 tactic × workspace) | 仅 lat predictor 可用 |
| `data/orin_dspace_bench.parquet` | 72 | ❌ | NaN | ✅ (含 32 fail) | Orin | 仅 Orin lat predictor 可用 |
| `results/orin_multi_engine_batch.json` | 10 | ❌ | NaN | ✅ | Orin (双 IP) | 仅 lat predictor, 仅 Orin |
| `results/orin_3ip_*.json` | 2 | ❌ | NaN | ✅ | Orin (三 IP) | 仅 lat predictor, 仅 Orin |
| `data/baseline_4090.parquet` | 83 | ❌ AP / ✅ amota 跨模型 | **amota ≠ DAIR AP, 不可混用** | ✅ | 4090 | 仅"跨模型 amota predictor"训练, **不喂主 Pyramid AP** |
| **小计** | **345** | **AP 真测: 12 行 (8 可用 + 4 negative)** | **337 行无 AP** | | | |

### 〇.5.2 严肃结论 (用户原意复述)

> **"如果不含有 AP 指标, 这些数据就没有用了才对啊"** — 用户

**部分正确**:
- 对 **AP predictor**: 345 行里只有 8 行 (2.3%) 真可用. 这是为什么 plan §1.1 必须新做 320 anchor (Class A) 来给 AP predictor 训练样本.
- 对 **lat predictor**: 337 行 lat-only 数据仍可用 — 训练时 LGB v5 支持 multi-task, lat 头吃完整 345 行, AP 头只吃 8+320=328 行. **但前提是 schema 里 AP 列必须显式 NaN, 严禁用 fp16 等价或 baseline AP 填充** (反思 #2 教训).
- 对 **throughput predictor**: 345 行中所有 build_success=True 的行 throughput=1000/lat_mean 派生可用 (单 IP 时), 双/三 IP 需独立测.
- 对 **resource predictor (engine_size / peak_gpu_mem / params)**: 部分行已有, 但 peak_gpu_mem 在多个老 parquet 里缺. P0.4 整合时必须显式标 NaN, 不补估.

**实操规则 (写进 P0.4)**:
1. `data/unified_bench.parquet` 必须按 §3 schema 每行 18+ 列, **缺测列填 NaN + `fail_reason` 字段标原因**, 不允许"猜值".
2. LGB v5 训练时按列 NaN 自动 mask, lat 头 ~1065 样本可训, AP 头只用 ~330 (8 stage_a + 320 Class A + 待补).
3. 论文 §C Pareto 主表上的每个数据点必须可追溯到 unified_bench 中具体行号 + source 列, 不允许出现"345 行"这种含糊表述. 报告时口径用 "AP 真测 N 行 / lat 真测 M 行", **不再讲合并行数**.

### 〇.5.3 跟"720 anchor 完整测试"的关系

720 anchor 计划保留, 但要点修订:
- 720 anchor 是**完整 4 类指标 (lat+AP+throughput+resource) 真测**的目标行数 (Class A 320 + B 120 + C 170 + D 100, 不依赖 345 旧行).
- 345 旧行不计入"720 完整 anchor", 但 lat-only 部分仍喂 lat predictor (作为额外样本).
- **总样本池修正**: lat predictor 训练 ≤ 720 + 337 = **1057 行可用**(其中 720 完整 + 337 lat-only); AP predictor 训练 ≤ 320 + 8 = **328 行可用**.

---

## 〇.6、ABCD 推进现状审计 (新增 2026-05-13)

> 用户原话: **"我看好像还没有完成 AB 子集的构建"** — 完全正确, 现状如下.

### 〇.6.1 P0.1/P0.2 实际产出 vs 计划

| 计划目标 | 状态 | 实际产出 | 偏差 |
|----------|------|----------|------|
| P0.1 8 triplet 充分训练 (Class A ckpt 库) | ✅ 完成 (2026-05-13 上午) | 8 ckpt 全过 AP gate (AP50 0.74-0.79) | 0 |
| P0.2 Class A 4090 主网格 (8×6×3=144) | ✅ 完成 (2026-05-13 10:32, 60.7 min) | 144 anchor | 第一批 |
| **P0.2.b 补 4090 Class A 176 anchor (4 新 Q + 1 新 D)** | **✅ 完成 (2026-05-13, 79.3 min wall, 0 fail)** | **176/176 OK, 合并 → 320 完整 4090 anchor** | **0** |
| P0.3 Class A Orin 250 anchor | 🔲 未启动 | 0 | -250 |
| P0.4 整合旧数据到 unified_bench | 🔲 未启动 | 0 | -345 (待整理) |
| P1 Class B 4090 D-space 120 | 🔲 未启动 | 0 | -120 (实际 48 旧 lat-only 已有, 待补 72 + AP) |
| P1 Class C Orin D-space 170 | 🔲 未启动 | 0 | -170 (实际 72 旧 lat-only 已有, 待补 98) |
| P2 Class D 跨模型 100 | 🔲 未启动 | 0 | -100 |
| **完整 4 指标 anchor 累计** | | **320 + 8 (stage_a) = 328** | **缺 392** |

### 〇.6.2.b P0.2.b 完成后 Q 维度新发现

10 Q × 4 D × 8 T 全笛卡尔覆盖后, AP spread 维持 76.69pp, 但 Q 维度结构性发现:

| Q 类型 | mean AP50 | 现象 |
|-------|-----------|------|
| Q_fp32 / Q_fp16 / Q_int8_mm | 0.75-0.76 | **安全路径**, AP 保 99% baseline |
| Q_int8_ent | 0.31 | **entropy 校准崩塌**: T7/T8/T1 触发, T6 稳 |
| Q_mix_s0..s12 (6 个混合) | 0.29-0.36 | **混合 INT8 路径全受 entropy 拖累** |

→ paper §C "搜索空间 ≠ Pareto frontier" 直接证据: 10 Q 中 **7 Q 含 AP<0.1 anchor**, ~63% Q 子空间有崩塌 cell. minmax + FP16 是真正"可信路径".

### 〇.6.2 P0.2 的 cell 缩水分析

P0.2 (`scripts/phase2/dataset_a_main_grid_4090.py`) 实际跑的是 6 Q × 3 D = 18 cell/triplet, 不是计划的 10 Q × 4 D = 40 cell/triplet, 原因:
- Q 只覆盖 6 个 (Q_fp16, Q_int8_ent, Q_int8_mm, Q_mix_s0, Q_mix_s2, Q_mix_s01), 缺 Q_int8_pc, Q_fp16_pc, Q_int8_wa_pt, Q_fp16_pt (per-channel × W+A 组合)
- D 只覆盖 3 个 workspace (1/4/8 GB), 缺 16GB + tactic 维度 (no_cudnn, all_enabled)
- **缩水合理性**: 主网格优先快出 AP 多样性证据 (76.69pp spread 已证), 但**论文 §C 主表需要完整 320 anchor 才能给 LGB 训练充分覆盖 §2.2 全部 10 Q**.

→ **必须补 176 anchor (P0.2.b 任务)** 才算 Class A 完成.

### 〇.6.3 修订后的 ABCD 进度看板 (锁定到 〇.5 重定义)

| Class | 计划行 | 完成 | 缺口 | 下一步 |
|-------|--------|------|------|--------|
| **A** Pyramid 4090 + Orin (B×Q×D × AP) | **320 + 250 = 570** | **320 (4090 完整 ✓)** + 0 (Orin) = **320** | **250** (仅 Orin) | P0.3 Orin 250 (~24h Orin 独占) |
| **B** 4090 D-space lat 扩展 | **120** | 0 | 120 | P1.a (旧 48 lat-only 整入 NaN-AP, P0.4 处理) + 72 新 random_search anchor |
| **C** Orin D-space lat 扩展 | **170** | 0 | 170 | P1.b (旧 72 lat-only 整入 + 98 新) |
| **D** UniAD-tiny + UniV2X | **100** | 0 | 100 | P2 (UniAD-tiny 60 + UniV2X 40 cross-model) |
| **合计完整 4-指标 anchor** | **720** | **328** (含 stage_a 8) | **392** | 见 §〇.6.4 |
| 旧 lat-only 数据 (并行 P0.4 整入) | — | 337 (NaN-AP) + 4 (stage_b negative) | — | P0.4 schema 统一 |

### 〇.6.4 修订 P 级任务执行优先级 (Day-1 ~ Day-N)

| 任务 | 工时估 | 完成则解锁 |
|------|--------|-----------|
| ~~P0.2.b~~ 补 4090 Class A 176 anchor | **✅ 完成 (79.3 min, 0 fail)** | 4090 Class A 320 完整 ✓ |
| ~~P0.4~~ 整合 337 lat-only + 4 negative + 83 amota 到 unified_bench schema | **✅ 完成 (10 min, 660 rows, schema 0 errors)** | LGB 三 head 训练源全部就绪 ✓ |
| **P0.3** Orin Class A 8 T × 6 Q × 9 single-IP D = 432 potential (trtexec, 跳 D7-D9 多 IP 留 P1.b) | **🟡 跑中 (Orin PID 146250)** ~3-7h wall | Class A 跨硬件完整 |
| **P1.a** 4090 D-space 扩 72 anchor (workspace 16GB + 2:4 sparse + edge_only tactic) | 4h | B-class 120 完成 |
| **P1.b** Orin D-space 扩 98 anchor (workspace 4/8GB + 双 IP × 4 + 三 IP × 4 + fallback policy) | 12h Orin | C-class 170 完成 |
| **P2** Class D cross-model 100 anchor | 8h (4090 8GPU) | D-class 100 完成 |
| **P3** LGB v5 重训 + Pareto demo | 2h | 论文 §C 数据就绪 |
| **剩余 wall (P0.3 串行 + P0.4/P1/P2 并行 + P3)** | ~36h ≈ 1.5 工作日 (Orin 24h 是瓶颈, 4090 闲时可干 P0.4/P1.a/P2) | |

---

## 一、总目标规模

### 1.1 数据总量

**总目标: ~720 个完整真测 anchor**(覆盖 Pyramid + UniAD-tiny + UniV2X 三模型, 跨 RTX 4090 + Orin AGX 两硬件).

| 子集 | anchor 数 | 主网格 | 用途 |
|------|-----------|--------|------|
| **A — Pyramid 充分训练 × B × Q × D** | **320** | 8 triplet × 10 Q × 4 D | LGB v5 主训练集 (lat / throughput / AP / build_success) |
| **B — 4090 D-space 扩展 (lat-only)** | **120** | 现有 48 → 100, 加 sparse + workspace 档 | LGB D 维度泛化加强 |
| **C — Orin D-space 扩展 (lat-only)** | **180** | 现有 72 → 180, 加 workspace + 双/三 IP | LGB Orin 端泛化 |
| **D — 跨模型 paired (UniAD-tiny + UniV2X)** | **100** | UniAD-tiny 60 + UniV2X 40 | 跨模型 LGB (cross-model adapter) |
| **总计** | **720** | | |

附加 / 已有: ~345 真测点(见 §3.1), 总样本池 ~1065 行.

### 1.2 覆盖率 vs 搜索空间

| 硬件 | 理论 cell | 目标真测 | 覆盖率 | LGB CV 需求(反思 #23) |
|------|-----------|---------|--------|------------------------|
| RTX 4090 (Pyramid) | 45,050 | 200 | 0.44% | ≥80 anchor 单模型 ✓ |
| Orin AGX (Pyramid) | 517,616 | 250 | 0.05% | ≥80 anchor 单模型 ✓ |
| UniAD-tiny / UniV2X (4090) | ~~10^5~~ (待算) | 100 | — | 跨模型补集 |

**注意**: 0.05% Orin 覆盖率绝对值低, 但 LGB 训练核心需求是**单模型 ≥80 anchor 且分布均匀**, 不是绝对覆盖率. 反思 #23 已论证.

---

## 二、维度覆盖要求(每条 anchor 必标注的搜索空间坐标)

### 2.1 B 维度(剪枝)

| 子维度 | 取值数 | 采样策略 |
|--------|--------|----------|
| `prune_object` | 3 | random, 但 channel 占 80% / 2:4 占 15% / none 占 5% |
| `stage0_planes` × `stage1_planes` × `stage2_planes` | 7×11×15 = 1155 (Orin) | 受 C8 (wpg pow2 / soft alignment) + C9 (planes mod 8) + C10 (跨层梯度 ≤0.30) 过滤 |
| 2:4 per-stage mask | 2³ = 8 | random when prune_object=2:4 |

**Class A 的 8 个 triplet 选定原则** (覆盖 B 维度 spectrum):

```
T1 baseline:    (64, 128, 256)   — 0% prune, AP 上限锚
T2 p25:         (48,  96, 192)   — 25% prune, Stage A 已覆盖, 拓展 D
T3 p37 (新):    (40,  80, 160)   — 37% prune, 填中间档 (反思: AP 真测无 (40,72,128) 等)
T4 p50:         (32,  64, 128)   — 50% prune, Stage A 已覆盖, 拓展 D
T5 p62 (新):    (24,  56, 128)   — 62% prune, 填中间档
T6 p75:         (16,  32,  64)   — 75% prune, Stage A 已覆盖, 拓展 D
T7 wide-shallow:(48,  64, 128)   — 非标 ratio (stage0/1 减得不一致, 测 framework 自适应)
T8 narrow-deep: (24,  48, 192)   — 非标 ratio (stage2 保, 强化 BEV head)
```

**T1-T6 是单调 ratio**, **T7-T8 是非单调 ratio** — 让 LGB 学到非线性 B 维度 lat / AP 关系.

### 2.2 Q 维度(量化)

每 triplet 测 10 个 Q-cfg (覆盖 Q1 全局 9 + Q1' per-stage 8 选 1):

```
Q1: FP32 / per-tensor / W-only     ← FP32 baseline
Q2: FP16 / per-tensor / W-only
Q3: FP16 / per-channel / W-only
Q4: FP16 / per-channel / W+A
Q5: INT8 / per-tensor / W-only
Q6: INT8 / per-tensor / W+A         ← 当前主用
Q7: INT8 / per-channel / W-only
Q8: INT8 / per-channel / W+A
Q9: mixed s0=INT8 s1=FP16 s2=FP16  ← Q1' 代表
Q10: mixed s0=FP16 s1=INT8 s2=INT8 ← Q1' 代表
```

(反思 #19/20: q_granularity 和 q_object 是全局变量, 不是 1.)

### 2.3 D 维度(部署)

**RTX 4090**: 测 4 个代表 D-cfg

```
D1: default tactic + 4GB workspace      ← Stage A 已用
D2: with_cudnn tactic + 8GB workspace   ← Part B-4090.2 最优组合
D3: CUBLAS_LT only + 16GB workspace     ← 新档, 测 16GB 边界
D4: all_enabled tactic + 1GB workspace  ← 资源 Pareto 端
```

**Orin AGX**: 测 12 个代表 D-cfg

```
单 IP × tactic × workspace:
D1: GPU + default + 2GB
D2: GPU + default + 8GB     ← 新档 8GB 测试
D3: GPU + no_cudnn + 4GB
D4: DLA0 (单 IP) + default + 1GB
D5: DLA1 (单 IP) + default + 1GB
D6: DLA0 + no_cudnn + 2GB

双 IP / 三 IP:
D7: 双 IP A (backbone DLA0 + collab GPU) + 2GB
D8: 双 IP B (backbone DLA1 + collab GPU) + 2GB
D9: 三 IP C (stage01 DLA0 + stage2 DLA1 + collab GPU) + 2GB

资源 Pareto 端:
D10: GPU + default + 1GB     ← 最小 workspace
D11: GPU + default + 8GB     ← 最大 workspace, 跟 D2 比 spread
D12: DLA0 + permissive fallback + 2GB  ← 测 fallback policy (反思: 暂未实验)
```

### 2.4 HW 维度(硬件)

每 anchor 标注 `hardware ∈ {rtx4090, orin_agx_64gb}`.
跨硬件 paired 的 anchor 用 `paired_id` 链接(同 (B,Q) 在两边各跑一次).

### 2.5 Model 维度(架构)

`model_class ∈ {pyramid_fusion, uniad_tiny, univ2x_full}`, 主体是 pyramid_fusion (700 anchor), 另两个共 100 anchor 做跨模型验证.

---

## 三、每 anchor 必测的"全方面"性能指标

按用户明确要求(吞吐量 + AP 精度 + 资源使用 + 全维度), 每行至少 18 列:

### 3.1 输入坐标列(11 列)

```
model_class        : str  ∈ {pyramid_fusion, uniad_tiny, univ2x_full}
stage0_planes      : int
stage1_planes      : int
stage2_planes      : int
prune_object       : str  ∈ {channel, 2:4, none}
sparse_mask        : str  ∈ {dense, mask_s0_only, mask_s0_s1, ..., 2:4_all}  ← when prune_object=2:4
q_bits             : str  ∈ {FP32, FP16, INT8, mixed}
q_granularity      : str  ∈ {per-tensor, per-channel, none}
q_object           : str  ∈ {W-only, W+A, none}
d_scheme           : str  ∈ {GPU, DLA0, DLA1, dual_A, dual_B, triple_C}
d_tactic           : str  ∈ {default, no_cudnn, with_cudnn, cublas_lt, all_enabled, edge_only}
d_workspace_gb     : int  ∈ {1, 2, 4, 8, 16}
hardware           : str  ∈ {rtx4090, orin_agx_64gb}
```

### 3.2 性能指标列(7 列必测)

| 列名 | 类型 | 测量来源 | 用户要求 |
|------|------|----------|----------|
| **lat_p50_ms** | float | TRT engine + cudaEvent (n_warmup=100, n_measure=200) | ✓ 延迟主指标 |
| **lat_p99_ms** | float | 同上 | ✓ 长尾监控 |
| **lat_mean_ms** | float | 同上 | ✓ 平均 |
| **throughput_fps** | float | multi-IP 时独立测(异步 stream 并发); 单 IP 时 = 1000/lat_mean_ms | ✓ **吞吐量主指标** |
| **ap30** | float | DAIR-V2X val 1789 samples 真测 (m4_8_hybrid_infer_ap.py) | ✓ **AP 精度主指标** |
| **ap50** | float | 同上 | ✓ 同上 |
| **ap70** | float | 同上 | ✓ 长尾 IoU 监控 |

### 3.3 资源使用指标列(4 列必测)

| 列名 | 类型 | 测量来源 | 用户要求 |
|------|------|----------|----------|
| **engine_size_mb** | float | os.path.getsize(engine_file) | ✓ **存储资源** |
| **peak_gpu_mem_mb** | float | nvidia-smi 监控 / cudaMemGetInfo 在 inference 时取样 | ✓ **运行时显存** |
| **params_kb** | float | 从 ckpt 算 (analyse model) | ✓ 模型参数量 |
| **build_secs** | float | TRT engine build 耗时 | ✓ 编译开销 (cold-start 资源) |

### 3.4 元数据列(可选但推荐)

```
build_success       : bool    True/False, 失败 anchor 也保留
fail_reason         : str     when build_success=False (DLA banks limit / quantization fail / OOM 等)
n_trt_path          : int     有多少 sample 走 TRT 真路径(对比 fallback)
n_pytorch_fallback  : int     PyTorch fallback sample 数
calibration_method  : str     ∈ {entropy, minmax, adaround, none}  ← INT8 校准方法
finetune_epochs     : int     该 ckpt 微调的 epoch 数(0 = baseline, 25 = 收敛)
source              : str     生成脚本名(用于复现追踪)
timestamp           : str     ISO 8601 时间戳
```

---

## 四、数据子集分阶段实施计划

### 4.1 Class A — Pyramid 充分训练 × B × Q × D (~320 anchor, P0)

**这是论文 §C 主表的数据来源, 最高优先级**.

#### 4.1.1 准备阶段: 8 个 triplet 全量训练 (~8 GPU-day on 4090 × 8)

每个 triplet 走完整 HEAL 训练流程:
1. **Stage 1 (~12 epoch)**: 训练剪枝后的 LiDAR-only backbone (collaborative pretraining)
2. **Stage 2 (~12 epoch)**: 训练 V2X collab head
3. 总 ~25 epoch, 单 GPU 30 min/epoch → 12.5 h/triplet
4. 8 triplet × 12.5 h / 4 GPU 并行 = ~25 h wall (~1.5 day)

输出: 8 ckpts in `models/dataset_a_cache/{triplet_sig}/net_epoch_bestval_atN.pth`

**反思 #24/25 应用**: 这 8 个 triplets 是 **hand-pick representative**, **不能用作 LGB 训练数据本身** — 而是作为 **ckpt 库**, 后续 B × Q × D anchor 在这 8 个 ckpt 之上做 (TRT build × AP eval), 形成 320 个 anchor.

#### 4.1.2 主网格扫描: 8 triplet × 10 Q × 4 D = 320 anchor on 4090

每 anchor 工作量:
- ONNX export (复用): 30 s 一次性
- TRT engine build: 1-2 min
- Latency bench: 1 min
- AP eval (DAIR-V2X val 1789): 5 min
- Resource measurement: ~10 s
- 合计: ~8 min/anchor

总计: 320 × 8 min = 2560 min = ~43 h on 1 GPU → 8 GPU 并行 ~5.5 h wall.

并行设计: 8 triplet 分到 8 GPU, 每 GPU 处理 40 anchor (10 Q × 4 D). AP eval 串行(避免 dataloader 抢 I/O, 反思: Stage B v1/v2 的 timeout 教训).

#### 4.1.3 Orin AGX 扩展: 8 triplet × 6 Q (剪) × 12 D = 576 anchor → 收敛到 ~250 真测 (P0)

Orin 上不重复全部 10 Q (因为 ONNX → TRT 8.5 兼容性差), 只测核心 6 个 Q:
- {FP16/per-tensor/W-only, FP16/per-channel/W-only, FP16/per-channel/W+A, INT8/per-tensor/W-only, INT8/per-tensor/W+A, INT8/per-channel/W+A}

12 D-cfg 见 §2.3.

8 × 6 × 12 = 576 potential, 但 build_success ≈ 40% (DLA failures) → ~230 真测 + ~340 build_fail negative samples.

工作量: Orin 上 build (3-5 min) + lat bench (1 min) + AP eval **NOT done on Orin in this plan** (因为 Stage A 已证 AP cross-hardware close, INT8 跨 IP 差异 < 1pp). Orin 只测 lat / throughput / build_success / 资源.

总: 576 × 5 min = ~48 h on Orin (独占, 2 day wall).

### 4.2 Class B — 4090 D-space lat 扩展 (~70 new anchor, P1)

已有 `data/4090_dspace_bench.parquet` 48 行. 补:

| 新增维度 | anchor 数 | 用途 |
|----------|-----------|------|
| workspace = {1GB, 16GB} 扩档 | 24 (3 triplet × 2 prec × 4 tactic × 2 ws) | 现在 doc 标 5 档 workspace, 实测只覆盖 2 档 |
| 2:4 sparse INT8 路径 | 24 (3 triplet × 4 tactic × 2 ws) | sparse 路径 spread 可能 10%+ (反思: 当前无测) |
| edge_only / cublas_lt only tactic | 24 | 5 tactic 全覆盖 |

总: ~72 new + 48 old = 120 (反思 #28: random_search 而非 hand-pick, 实际 3 triplet 仍是 hand-pick 是历史遗留, 后续 P1 补 random_search 30 个 unseen triplet 各 4 D-cfg).

### 4.3 Class C — Orin D-space lat 扩展 (~108 new anchor, P1)

已有 72 行. 补:

| 新增维度 | anchor 数 |
|----------|-----------|
| workspace = {4GB, 8GB} 扩档 | 36 |
| 双 IP scheme {A, B} × 4 不同 triplet × 2 prec | 16 |
| 三 IP scheme C × 4 不同 triplet × 2 prec | 16 |
| GPU fallback policy {strict, permissive} × 5 anchor | 10 (Orin 独有维度) |
| 单 IP DLA0/1 全图测 × 5 unseen triplet × 2 prec | 20 |

总: ~98 new + 72 old = 170 (留 10 buffer 失败重测).

### 4.4 Class D — 跨模型对照 (~100 anchor, P2)

**UniAD-tiny (univ2x-tiny variant) on 4090 — 60 anchor**:
- 复用现有 5 个 `univ2x_tiny_*` ONNX (M5.1-M5.6 输出) 作为 baseline
- random_search 12 个 prune ratio × 5 D-cfg = 60 anchor
- 全测 lat + throughput + AP (在合适的 V2X dataset 上, 可能不能用 DAIR, 改 OPV2V-coop val 4K samples)

**UniV2X (DCN 反例) on 4090 — 40 anchor**:
- Backbone 不能剪 (DCN), 只测 B = {none, 2:4} × Q × D
- 40 anchor 当 framework 的"硬约束实证"

### 4.5 已有数据的整合与清洗 (P0, 与 4.1 并行)

已有 ~345 行散在多个 parquet, 需统一 schema 到 §3.1-§3.4 标准列:

| 现有数据 | 行数 | 缺失列 | 处理 |
|----------|------|--------|------|
| `data/pyramid_random_bench.parquet` | 100 | 无 AP / throughput | 补 AP 真测 (Stage A 已部分覆盖 4 + 2 = 6) 或标 [no_AP_data] |
| `data/perstage_quant_bench.parquet` | 18 | 无 AP | 选 6 个代表点补 AP 真测 |
| `data/4090_dspace_bench.parquet` | 48 | 无 AP | 同上 |
| `data/orin_dspace_bench.parquet` | 72 | 无 AP, 无 throughput | 单 IP 时 throughput = 1000/lat; AP 标 [estimated_from_4090] |
| `data/stage_a_ap_real.parquet` | 8 | 无 D-cfg variation | 主真值表 ✓ |
| `data/stage_b_ap_real.parquet` | 4 | finetune 不足, AP≈0.06 | 标 [insufficient_finetune], 不进 LGB 训练但保留作 negative |
| `results/orin_multi_engine_batch.json` | 10 | 无 AP | 标 [estimated_from_4090] |
| `results/orin_3ip_*.json` | 2 | 无 AP | 同上 |
| `data/baseline_4090.parquet` | 83 | 跨模型, amota 而非 AP | 标 model_class, 单独喂跨模型 LGB |

**整合产出**: `data/unified_bench.parquet` (~1065 行 = 345 existing + 720 new), schema 按 §3.

---

## 五、工程时间预算

> **注**: 本节是 "未做加速优化" 的 baseline 估算 (4.5 day). 实际执行按 §8.0 训练/eval 加速协议, **总 wall 降到 2 day**, 详见 §8.0.5.

按 4 GPU 4090 + 1 Orin AGX 独占的可用资源:

| Phase | 任务 | 单机工时 | 并行 wall | 累计 wall |
|-------|------|----------|-----------|-----------|
| **P0.1** | 8 triplet 充分训练 (Class A 准备) | 100 GPU-h | 4 GPU 并行 = 25 h | **1 day** |
| **P0.2** | Class A 4090 主网格 320 anchor | 43 GPU-h | 8 GPU 并行 = 5.5 h | 1.2 day |
| **P0.3** | Class A Orin 扩展 250 anchor | 48 h (Orin 独占) | 不可并行 | 3.2 day |
| **P0.4** | 整合现有 345 行清洗 + AP 补测 | 15 GPU-h | 1 GPU = 15 h | 3.8 day |
| **P1** | Class B + C D-space 扩展 (~178 anchor) | 30 GPU-h + 12 Orin-h | 8 GPU + Orin = 4 h + 12 h | 4.3 day |
| **P2** | Class D 跨模型 (~100 anchor) | 20 GPU-h | 8 GPU 并行 = 2.5 h | 4.5 day |
| **P3** | LGB v5 重训 + 5-fold CV + Pareto demo 复跑 | 1 GPU-h | 0.5 h | 4.5 day |

**总 wall time: ~4.5 工作日** (4090 4 GPU + Orin 独占, 中间穿插).

**关键时间节点**:
- Day 1 end: P0.1 完成, 8 个收敛的 Pyramid ckpts 入库
- Day 2 end: Class A 4090 320 anchor 完成 (论文主表数据出炉)
- Day 4 end: Class A Orin 完成 (跨硬件 paired 数据完整)
- Day 5: LGB 重训 + Pareto demo + 写 result/ 报告

---

## 六、数据落点 + Schema 标准

### 6.1 文件路径

```
data/
├── unified_bench.parquet         ← 主数据集, schema 见 §3
├── unified_bench.csv             ← 同上 CSV 版本(便于人工查阅)
├── _by_class/
│   ├── class_a_pyramid_full.parquet      ← 320 行
│   ├── class_b_4090_dspace_v2.parquet    ← 120 行 (含历史 48)
│   ├── class_c_orin_dspace_v2.parquet    ← 170 行 (含历史 72)
│   ├── class_d_cross_model.parquet       ← 100 行
│   └── legacy/                            ← 历史 parquet 原文件 (不删)
├── schema_spec.json              ← 列定义 + dtype + 取值集合
└── failed_anchors.parquet        ← build_success=False 的 negative samples
```

### 6.2 Schema 校验脚本

```python
# scripts/phase2/validate_dataset_schema.py
REQUIRED_COLUMNS = [
    # 坐标列 (11)
    "model_class", "stage0_planes", "stage1_planes", "stage2_planes",
    "prune_object", "sparse_mask", "q_bits", "q_granularity",
    "q_object", "d_scheme", "d_tactic", "d_workspace_gb", "hardware",
    # 性能 (7)
    "lat_p50_ms", "lat_p99_ms", "lat_mean_ms", "throughput_fps",
    "ap30", "ap50", "ap70",
    # 资源 (4)
    "engine_size_mb", "peak_gpu_mem_mb", "params_kb", "build_secs",
    # 元数据
    "build_success", "fail_reason", "n_trt_path", "n_pytorch_fallback",
    "calibration_method", "finetune_epochs", "source", "timestamp",
]
```

每次 anchor 写入前必经此 validator, 缺列就 raise.

### 6.3 测量来源 traceability

每行 `source` 字段记录生成脚本名 (e.g., `class_a_main_grid_v1.py:run_anchor()`), 便于复现追踪. 反思 #8 教训: "数据呈现没标 caveat" 不能再犯.

---

## 七、反思驱动的设计约束(必须遵守)

| 反思 # | 教训 | 本计划如何避免 |
|--------|------|---------------|
| **#1** (核心) | 多指标 Pareto 评估, 不能只测速度 | §3 强制每 anchor 测 lat + throughput + AP + 资源, 4 类指标缺 1 即视为不完整 |
| **#5** | mask-based pruning 不真减 latency | §4.1 用 **structural pruning** (channel L1 删除真实减少 weight), 不是 mask |
| **#6** | 预测 anchor 选错(子模块 vs e2e) | §3.2 latency 测**完整 e2e**(含 voxelize + backbone + collab), 不是子模块 dummy |
| **#8** | 表格数据没标 caveat | §6.3 source / timestamp / calibration_method / finetune_epochs 必填 |
| **#9** | 子模块 vs e2e 没标口径 | 同上, 单独列 lat_subgraph_ms vs lat_e2e_ms 区分 |
| **#23** | 5% LGB 覆盖不够, 应 ≥20% | 单模型 Pyramid 目标 200 anchor (>500 cell 中代表), 反思 #23 的 80 ≤ 200 ✓ |
| **#24** | hand-pick 训 LGB 是 selection bias | §4.1.1 hand-pick 只用于"提供 ckpt 库", LGB 训练数据用主网格 320 anchor (B × Q × D 笛卡尔积, 不是 Pareto 选点) |
| **#25** | validation hand-pick vs training random 混淆 | 主网格 320 是 training (笛卡尔均匀), Pareto 验证另用 NSGA-II Top-K (不重叠) |
| **#28** | closed-loop demo 用 enumeration 而非 random_search | P3 阶段 LGB Pareto demo 必须 random_search 1000+ candidates, 不是 enumerate |
| **#29** | 搜索空间 ≠ Pareto frontier, 不能预删"不优" cell | per-stage Q 即使 18/18 比全 INT8 慢仍纳入 (作 LGB 负样本), 不预删. 数据集策略同步: 17 个 Q + 6 个 D_scheme 全留 |
| **#30** | DLA 不兼容是 per-operator, 不是 per-model | 单 IP DLA0/1 anchor 留在数据集 (即使大概率 build_success=False), 让 LGB 学 fallback 阈值 |
| **#31** | B-Orin.3 流水线 deferred 是工程懒 | Class C 必须真测双 IP / 三 IP, 不能用估算 |
| **新** (本计划) | finetune budget 必须充足 | §4.1.1 8 triplet 走完整 ~25 epoch HEAL 训练, 不是 1-4 epoch (Stage B 错误教训) |

---

## 八之零、训练加速与精度保障协议(必须遵守)

历史 Stage B 教训:
1. **微调 epoch 不够 → AP 崩 (0.06)**: M4.9 random_search 输出 `bestval@23` 是裸结构剪枝 ckpt (L1-norm 删 channel 后未微调), 用 1-4 epoch 短微调救不回来, AP 直接崩.
2. **重训只用单 GPU → 速度慢**: HEAL `train.py` 默认 single-GPU, 单 triplet 25 epoch 在 1 张 4090 上 ~12.5 h, 8 triplet 100 GPU-h 不可接受.

本计划必须执行以下 4 + 4 防护措施.

### 8.0.1 精度保障 — 4 层防护(防 AP 崩)

| # | 防护手段 | 应用阶段 | 预期效果 |
|---|----------|----------|----------|
| **P1** | **完整 HEAL stage1 训练 (~25 epoch)** | Class A 8 triplet 全用 | AP50 vs baseline 差距 ≤ 2pp(50% 剪枝) |
| **P2** | **Knowledge Distillation 加速** | Class D 跨模型 / 时间紧的额外 triplet | Teacher=baseline 收敛 ckpt, Student=剪枝 ckpt, distill loss + det loss → 5-8 epoch 收敛 |
| **P3** | **Iterative Magnitude Pruning (IMP)** | Class D random triplet 大量场景 | 5%×15 step 渐进剪枝, 每 step 内 2 epoch 微调, 累积 30 epoch 但 AP 平滑下降 |
| **P4** | **Convergence Gate 硬验收** | 所有 ckpt 入库前 | 不接受 `AP50 < baseline_AP50 - 5pp` 的 ckpt; 不收敛标 `finetune_status='not_converged'`, **不进 LGB 训练集** |

**具体执行规则**:
- 训练时**每 epoch 在 DAIR-V2X val 子集 (~500 sample) 评 AP50**, 监控收敛
- **连续 3 epoch ΔAP50 < 0.5% 才视为收敛**, 否则强制延 10 epoch
- 收敛后保存 `net_epoch_bestval_atN.pth`(N 是 best epoch), **同时记录 finetune_epochs 进 anchor row**(反思 #8: 标 caveat)
- 极端剪枝 (16,16,16) 或低于 (24,24,24) 通道不入 Class A 主力(实测 ckpt 不稳 + TRT engine 路径异常)

### 8.0.2 训练加速 — 4 项工程化

| # | 加速手段 | 实现方式 | 预期加速 |
|---|----------|----------|----------|
| **S1** | **DDP 多卡并行 (4 GPU per triplet)** | `torchrun --nproc_per_node=4 train.py --dist-url env://` 替代 `CUDA_VISIBLE_DEVICES=N python train.py` | **3.5× ↑**(单 triplet 12.5h → 3.5h) |
| **S2** | **混合精度训练 (AMP)** | `torch.cuda.amp.autocast()` + `GradScaler`, 添 train.py 的 forward+backward 包装 | **1.5-2× ↑**, 显存 -40% |
| **S3** | **batch_size 上调** | DDP 4 卡每卡 batch_size=4, effective batch=16 (单卡默认 batch=2) | 1.3× ↑(GPU 利用率从 60% → 90%) |
| **S4** | **DAIR-V2X 数据 voxelization 预缓存** | 一次性把 train + val 1789 sample 的 voxel + LiDAR feature 存 npz, 训练时 skip per-batch 重算 | 1.2-1.5× ↑(I/O bound 节省) |

**S1+S2+S3+S4 复合加速**: 12.5h / triplet → **2-3h / triplet** (~5× 加速)

**实际部署**:
- 8 张 4090 分两批: 2 个 triplet 并行 × 4 卡 DDP each → 4 batch 顺序 → 8h wall (8 triplet 全跑完)
- **vs 之前 plan 的 25h wall (1 triplet/GPU 8 卡并行单卡训) → 3× 加速**

### 8.0.3 AP Eval 加速 — 消除 PyTorch fallback

Stage B 6/40 anchor 撞 3600s timeout 教训: AP eval 慢主要因为 PyTorch fallback (TRT path=0).

**根因**: `m4_8_hybrid_infer_ap.py` 路由:
```
N=1 sample + trt_subnet ≠ None → 单 agent TRT 引擎 (快)
N=2 sample + trt_collab ≠ None → 协同 TRT 引擎 (快)
else                            → PyTorch fallback (慢, 每 sample 1-6s)
```

Stage A/B 只 build 了 collab 引擎 (`--engine-collab`), 没 build 单 agent 引擎 (`--engine`). DAIR val 1789 里有 ~9.6%-15% N=1 sample → 全 fall back PyTorch → 1789 × ~3s = 89 min/anchor.

**修复**: 每个 ckpt build **两套**引擎 + AP eval 两个 engine 都传:

```python
# scripts/phase2/dataset_a_main_grid_4090.py 中:
def build_anchor(triplet, prec, d_cfg):
    # 1. ONNX export (一次性, 双 subgraph)
    onnx_subnet = export_subnet_onnx(ckpt)    # 单 agent: spatial_features → cls/reg/dir
    onnx_collab = export_collab_onnx(ckpt)     # 协同: spatial × N=2 + t_ego → cls/reg/dir

    # 2. TRT engines build (双套)
    engine_subnet = trt_build(onnx_subnet, prec=prec, d_cfg=d_cfg)
    engine_collab = trt_build(onnx_collab, prec=prec, d_cfg=d_cfg)

    # 3. AP eval (双 engine 都传, 0 PyTorch fallback)
    subprocess.run([..., "--engine", engine_subnet, "--engine-collab", engine_collab, ...])
```

**预期 AP eval 时间**:
- 之前 (单 collab engine, 15% PyTorch fallback): 5-7 min/anchor
- 修复后 (双 engine, 0% fallback): **2-3 min/anchor** (~2× 加速)

### 8.0.4 AP Eval 二段策略 — Sweep vs Final

| 阶段 | n_samples | 单 anchor 用时 | 用途 |
|------|-----------|-------------|------|
| **Grid sweep** (Class A 主网格 320 anchor) | **500** (DAIR val 子集均匀采样) | 1-1.5 min | 快速过完所有 anchor, AP MAE 估计 ±1-2pp |
| **Final validation** (NSGA-II Pareto Top-20) | **1789** (full DAIR val) | 3-4 min | 论文 §3 主表数字必须用 1789 |

→ Class A 320 anchor sweep: 320 × 1.5 min / 4 GPU = 2h wall (比原计划 5.5h 再快 3×).

### 8.0.5 综合时间预算修订

| Phase | 原计划 wall | 修订后 wall | 加速来源 |
|-------|------------|------------|----------|
| P0.1 8 triplet 全量训练 | 25 h | **8 h** | DDP S1 + AMP S2 + batch S3 + cache S4 |
| P0.2 4090 主网格 320 anchor | 5.5 h | **2 h** | 双 engine 消除 fallback + 500-sample sweep |
| P0.3 Orin 250 anchor | 48 h (Orin 独占) | **24 h** | 双 engine + Orin 不做 AP eval (假设 cross-HW AP 等价) |
| P0.4 整合现有数据 | 15 h | 15 h | 不变 |
| P1 + P2 + P3 | 6 h | 6 h | 不变 |
| **总 wall** | **4.5 day** | **2 day** | (4090 4 卡 + Orin 独占) |

---

## 八、关键风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 8 个 triplet 全量训练 25 h wall 超时 | **低**(修订后) | §8.0.2 S1+S2+S3+S4 已降到 8 h wall; checkpoint resume 容错; 若 1 个 triplet 训不收敛, 走 §8.0.1 P4 标 `not_converged`, 不丢全 plan |
| Orin AGX 上 AP eval 时间过长(单 sample 慢) | 高 | **不在 Orin 上做 AP eval**(假设 AP 跨 hardware FP16 等价, INT8 ±1pp 误差 paper 标 caveat) |
| DAIR-V2X val 1789 samples 全跑 AP 太慢 | **低**(修订后) | §8.0.4 二段策略: sweep 500 sample(1.5 min) + Pareto Top-20 跑 1789(3 min) |
| **AP eval PyTorch fallback 撞 timeout** | **低**(修订后) | §8.0.3 修复: 每 ckpt build 双 engine (单 agent + collab), 0% fallback, AP eval 2-3 min/anchor |
| **微调不充分导致 AP 崩 (Stage B 0.06 教训)** | **低**(修订后) | §8.0.1 P1+P4: 完整 25 epoch + AP50 收敛 gate, 不收敛标 not_converged 丢 negative pool |
| 跨模型(UniAD-tiny / UniV2X) baseline 不收敛 | 中 | 复用现有 ckpt(univ2x-tiny variant), 不重训; UniV2X 只测 backbone=baseline 不剪枝 |
| 资源使用(peak_gpu_mem)采样不稳 | 低 | nvidia-smi pmon 采样 + 取 inference 阶段 max; 每 anchor 重复 3 次取 median |
| Orin GPU fallback policy 实验失败 | 低 | strict / permissive 是 TRT 配置选项, 影响 build, 失败 anchor 标 fail_reason='dla_strict_fail' 入 negative set |
| 数据 schema 后续要扩列 | 中 | schema_spec.json 版本化(v1, v2), 旧数据用 NaN 填新列 |

---

## 九、验收标准 (P0 完成时必须满足)

1. **行数**: `data/unified_bench.parquet` ≥ 720 行 (Class A+B+C+D)
2. **覆盖率**:
   - Pyramid on 4090: AP 真测 ≥ 100 anchor (Class A 子集), 占 4090 搜索空间 0.22%
   - Pyramid on Orin: lat 真测 ≥ 200 anchor, build_success 标注 ≥ 250 anchor (含 negative)
   - 跨模型: ≥ 100 paired anchor
3. **Schema 校验通过**: `python scripts/phase2/validate_dataset_schema.py` 0 errors
4. **每 anchor 4 类指标都有 (或明确 NaN + caveat)**:
   - Lat ≥ 95% rows 有 (build_success=True 时必填)
   - Throughput ≥ 95% rows 有
   - AP ≥ 30% rows 有 (Class A 应 100%, B/C/D 可部分 NaN)
   - 资源 (engine_size_mb 等) ≥ 95% 有
5. **LGB v5 重训性能**:
   - lat predictor 5-fold CV Spearman ≥ 0.95 (现 v3 是 0.968 ✓)
   - AP predictor 5-fold CV Spearman ≥ 0.85 (现 v4 in-sample 1.00, 但 8-anchor 不可信)
   - build_success AUC ≥ 0.95 (现 1.0 ✓)
6. **5D Pareto demo (lat × throughput × AP × ws × params)**:
   - 4090 + Orin 各产 ≥ 20 个 Pareto 点
   - 真测 AP 验证 Top-5 Pareto 点的预测 AP MAE ≤ 5 pp

---

## 十、当前已有的真测 anchor 资产清单(起点, 2026-05-13 更新)

按 §6 schema 整理后的现有数据:

| 文件 | 行数 | 必测指标完整度 | 备注 |
|------|------|---------------|------|
| **class_a_pyramid_full** (P0.2 新出) | **144** | lat (待重读 build JSON key) / thr (派生) / **AP ✓ (500 sample)** / 资源 ✓ | **P0.2 产出**, 8 triplet × 6 Q × 3 D, AP spread 77pp |
| stage_a_ap_real | 8 | lat ✓ / thr ✓ / **AP ✓ (1789 full)** / 资源 ✓ | **paper-grade**, 充分微调 (基准锚) |
| stage_b_ap_real | 4 | lat ✓ / thr ✓ / AP (≈0.06) / 资源 ✓ | **negative**, 1-4 epoch finetune 不足, 不喂 AP 头 |
| pyramid_random_bench | 100 | lat ✓ / thr (派生) / **AP ✗** / 资源 部分 | 4090 单 D, lat 头可用, AP 列必须 NaN |
| perstage_quant_bench | 18 | lat ✓ / thr (派生) / **AP ✗** / 资源 部分 | 4090 mixed Q, 同上 |
| 4090_dspace_bench | 48 | lat ✓ / thr (派生) / **AP ✗** / 资源 ✓ | 4090 D, 同上 |
| orin_dspace_bench | 72 | lat ✓ / thr (派生) / **AP ✗** / 资源 部分 | Orin D, 含 32 fail, 同上 |
| orin_multi_engine_batch | 10 | lat ✓ / thr ✓ / **AP ✗** / 资源 ✗ | Orin 双 IP, 仅 lat 头 |
| orin_3ip_* | 2 | lat ✓ / thr ✓ / **AP ✗** / 资源 ✗ | Orin 三 IP, 仅 lat 头 |
| baseline_4090 | 83 | lat ✓ / **amota ✓ 跨模型** / 资源 部分 | 跨模型 amota, **不混入 Pyramid AP** |

**当前真测 anchor 池修订**:
- **完整 4 指标 (lat+AP+throughput+resource) anchor**: **152 行** = 8 (stage_a 1789) + 144 (Class A 500 sample). 仅作 AP predictor 训练源.
- **lat-only 可喂 lat predictor**: 337 行 (pyramid_random + perstage_quant + 4090_dspace + orin_dspace + multi_engine + 3ip).
- **跨模型 amota**: 83 行 (baseline_4090), 不混 Pyramid AP, 单独跨模型头.
- **negative 池**: 4 行 (stage_b, AP≈0.06), 仅作"未收敛 ckpt 负样本".

→ **离 §1.1 的 720 完整 anchor 目标缺 568 行**, 必须按 §〇.6.4 推进 P0.2.b/P0.3/P1/P2.

---

## 十一、对应实施脚本(占位, 待 P0 启动时建)

```
scripts/phase2/
├── dataset_a_prepare_ckpts.py        ← P0.1 8 triplet 全量训练
├── dataset_a_main_grid_4090.py       ← P0.2 4090 320 anchor 主网格
├── dataset_a_main_grid_orin.py       ← P0.3 Orin 250 anchor
├── dataset_b_4090_dspace_expand.py   ← P1 4090 D 扩展
├── dataset_c_orin_dspace_expand.py   ← P1 Orin D 扩展
├── dataset_d_cross_model.py          ← P2 UniAD-tiny + UniV2X
├── dataset_unify_and_clean.py        ← P0.4 整合现有 345 行
├── validate_dataset_schema.py        ← schema 校验
└── train_lgb_v5.py                   ← P3 LGB 重训
```

---

## 十二、修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-12 | 初版, 综合 搜索空间一览 + result/ + reflection_mistakes + 00_故事路线 |
| v1.1 | 2026-05-13 | + §〇.5 旧 345 数据可用性诊断 (AP 真测仅 8 行) + §〇.6 ABCD 推进现状审计 (Class A 144/320 缺 176, BCD 全未启动, 总进度 152/720) |
| v1.2 | 2026-05-13 | + §十三 当前执行进度快照 (P0.1/P0.2/P0.2.b/P0.4 完成, P0.3 Orin 跑中 280/432 OK=89, BCD 未启动) |

---

## 十三、当前执行进度快照 (2026-05-13 完整审计)

> **目的**: 用户要求重新评估实验路径正确性, 本节冻结当前真实状态, 不做评价.

### 13.1 任务级完成情况

| 任务 | 计划目标 | 实际产出 | 状态 | wall time | 关键产出文件 |
|------|----------|----------|------|-----------|--------------|
| P0.1 | 4 NEW Pyramid triplet × 25 epoch 训练 (T3/T5/T7/T8) | 4 ckpt 全过 AP gate (AP50 0.74-0.79) | ✅ 完成 | ~12h (2026-05-13 上午) | `models/dataset_a_cache/ft_{sig}/net_epoch_bestval_at33.pth` × 4 |
| P0.2 | 4090 Class A 主网格 (8×6 Q×3 D=144 anchor) | 144/144 OK, AP spread 76.69pp | ✅ 完成 | 60.7 min | `data/_by_class/class_a_pyramid_full.parquet` (初版) + `results/dataset_a_main/*.json` × 144 |
| P0.2.b | 4090 Class A 补充 (4 NEW Q × 4 D + 6 OLD Q × 1 NEW D = 176 anchor) | 176/176 OK, 合并 → 320 完整 4090 anchor | ✅ 完成 | 79.3 min | `data/_by_class/class_a_pyramid_full.parquet` (合并 320 行 = 10 Q × 4 D × 8 T 全笛卡尔) |
| P0.4 | 整合 9 旧数据源到 unified_bench schema | 660 rows × 34 cols, schema 0 errors | ✅ 完成 | ~10 min | `data/unified_bench.parquet` + `data/unified_bench.csv` + `data/schema_spec.json` |
| P0.3 | Orin Class A 8 T × 6 Q × 9 single-IP D = 432 anchor (lat-only) | **🟡 跑中 280/432 (65%), OK=89, fail=191** | 进行中 | ~7h wall 已耗 (cum) | `/home/jichengzhi/orin_class_a/results/*.json` (89 OK + 191 fail) |
| P1.a | Class B 4090 D-space 扩展 72 new anchor | 0 | 🔲 未启动 | — | — |
| P1.b | Class C Orin D-space 扩展 98 new anchor (workspace 4/8GB + 双 IP + 三 IP + fallback policy) | 0 | 🔲 未启动 | — | — |
| P2 | Class D 跨模型 (UniAD-tiny 60 + UniV2X 40 = 100 anchor) | 0 | 🔲 未启动 | — | — |
| P3 | LGB v5 重训 + 5-fold CV + 5D Pareto demo | 0 | 🔲 未启动 | — | — |

### 13.2 真实数据资产盘点 (绝对真测行数)

**A. 完整 4 指标 (lat + AP + throughput + resource) Pyramid anchor**:
- **Class A 4090**: **320 行** (P0.2 + P0.2.b 合并, AP eval 500-sample DAIR val)
- **Stage A paper-grade**: **8 行** (1789-sample full DAIR val, FP32/FP16 baseline 各 4)
- **Stage B negative**: **4 行** (AP≈0.06, finetune 不足, 仅作 LGB negative pool, **不入主 AP head**)
- → **完整 4 指标 anchor 累计**: **328 行** (320 + 8, 不含 negative)

**B. lat-only (AP=NaN, 仅 LGB lat head 可用)**:
- `pyramid_random_bench.parquet`: 100 行 (4090, 单 D)
- `perstage_quant_bench.parquet`: 18 行 (4090, mixed Q)
- `4090_dspace_bench.parquet`: 48 行 (4090, D 维度)
- `orin_dspace_bench.parquet`: 72 行 (Orin, D 维度, 含 32 build_fail)
- `orin_multi_engine_batch.json`: 10 行 (Orin, 双 IP)
- `orin_3ip_*.json`: 2 行 (Orin, 三 IP)
- **P0.3 Orin (跑中)**: **89 行 OK** + 191 fail (lat-only, 主要 INT8 因 cache 跨版本崩) ← 持续累积中
- → **lat-only 累计 (含 P0.3 当前)**: **339 行** (250 旧 + 89 新)

**C. 跨模型 amota (不混 Pyramid AP)**:
- `baseline_4090.parquet`: 83 行 (跨 3 model_class, amota 指标)
- → **跨模型 amota**: **83 行**

**D. 总计真测行数 (2026-05-13 当前)**: **750 行** = 328 完整 + 339 lat-only + 83 amota
- 距 §1.1 "720 完整 anchor" 目标: 完整 4 指标只 328, **缺 392** (B/C/D 完整真测全未启动)
- lat-only LGB lat head 可训样本数: 660 (P0.4 unified) + 89 (P0.3 当前) = **749 lat 样本**

### 13.3 Orin P0.3 当前结果分布 (280/432 已完成)

| Q 类型 | 完成数 | OK 数 | fail 数 | 主要失败原因 |
|-------|-------|-------|---------|--------------|
| Q_fp32 | 48 (6 triplet × 8 D) | 42 | 6 | D12 dla0_strict 无 fallback fail |
| Q_fp16 | 48 | 42 | 6 | 同上 |
| Q_int8_mm | 48 | 0 | 48 | **TRT cache 跨版本 (4090=10.13 vs Orin=8.5) 不兼容** |
| Q_int8_ent | 48 | 0 | 48 | 同上 |
| Q_best | 48 | 0 | 48 | 同上 (需 calib) |
| Q_int8_nofallback | 40 (T1-T5) | 0 | 40 | 同上 |
| **小计** | **280** | **84 (30%)** | **196** | |

> 注: T6 段 Q_fp16 仍在跑, 数字会再增. INT8 全失败属已知 gap, 待 P0.3.b 补救 (patched cache magic TRT-101300 → TRT-8502).

### 13.4 关键工程产出 (脚本 / 文档)

**脚本** (`scripts/phase2/`):
- `dataset_a_prepare_ckpts.py` — P0.1 4 triplet 训练
- `dataset_a_main_grid_4090.py` — P0.2 主网格
- `dataset_a_main_grid_4090_supplement.py` — P0.2.b 补充网格
- `dataset_unify_and_clean.py` — P0.4 9-source schema 合并
- `validate_dataset_schema.py` — P0.4 schema 校验
- `orin_class_a_bench_trtexec.py` — P0.3 Orin trtexec 版本 (无 pycuda 依赖)
- `orin_class_a_int8_sweep.py` — P0.3.b INT8 补救 (patched cache, 待启动)
- `build_inspect_csv.py` — 最终 720 anchor CSV 聚合器

**文档** (`paper_learning/2. AAAI最终故事/`):
- `数据集制作_plan.md` (本文件, v1.2)
- `搜索空间一览.md` (4090 25 / Orin 96 + 5 维 D)
- `Pyramid_架构与剪枝边界_audit.md` — Pyramid 7 子模块完整解剖 + 剪枝可达边界 + 速度瓶颈 (B1 NMS 60% / B2 PointPillar 30% / B3 backbone 9%)
- `00_故事评估与实验路线_v1.md` (v2.5)

### 13.5 ⚠️ 已知重大限制 (需用户决策实验路径前知悉)

1. **ONNX subnet 仅占 e2e ~9-10%** (Pyramid 架构 audit 实证): 剪 backbone 50% 在 e2e 上仅 6.5% 加速; **paper §C "剪枝 + 量化 Pareto 主表" 数据仅反映 9% e2e 的优化**, 不是 e2e 整体加速
2. **prune 30-37% 区间 lat 反向变慢** (T2/T3 实证): 非 32/64 对齐 channel → TRT 内核反优化, 搜索空间内"可行但崩"cell 实证
3. **Orin INT8 完全失败 (cache 跨版本)**: 当前 280/432 OK=84 仅含 FP32+FP16+DLA, 缺整个 INT8 Q 维度. P0.3.b 已准备 patched cache 但未启动
4. **Class B/C/D 完整真测全未启动**: 当前 328 完整 anchor 仅来自 Class A, 距 720 目标差 392 行 (61% 工作量未做)
5. **跨硬件 AP 假设未验证**: Class A 4090 AP 真测, Orin 不做 AP eval, 假设 "INT8 跨 HW ±1pp" 是 §4.1.3 caveat, **未实证**
6. **真实速度瓶颈不在我们的搜索空间**: NMS (60% e2e) + PointPillar encoder (30%) 都不在剪枝量化范围, 协同框架的 Q 维度天花板就是 ~1.1× e2e

### 13.6 下一步候选路径 (待用户决策)

> **本节不预设, 仅列出可走选项**.

| 路径 | 实质 | 工作量 | 价值 |
|------|------|-------|------|
| **A. 当前路径加速** | 等 P0.3 跑完 (~3-4h), 启 P0.3.b INT8 补救 (~4h), 然后 P1.a Class B 72 anchor, P1.b Orin D 扩展 98 anchor, P2 跨模型 100 anchor, P3 LGB 重训 | ~30h wall | 完成 720 anchor 计划, 维持 paper §C "剪枝量化 Pareto" 叙事 |
| **B. 重新定位 paper §C** | 承认 ONNX subnet 仅 9% e2e, 重写 §C 为 "**框架感知硬件 + D 调度** 是 Pareto 主轴, 剪枝量化是 Q 子维度". 数据上停止扩 720, 重点跑 D 维度对比 (GPU vs DLA vs 多 IP) | ~15h wall + paper 重写 | 故事更诚实, 但 paper 主表换结构 |
| **C. 攻 e2e 真瓶颈** | 改 NMS 为 CUDA kernel (~5 人日工程), 或换 encoder 为 dense (失精度), 真拿 e2e 2-3× 加速. **抛弃当前剪枝量化路径** | ~10 工作日 + 训练验证 | 风险高, 但 e2e 数字漂亮 |
| **D. 数据集回炉 — 加 e2e timing breakdown** | 把现 320 anchor + 后续 anchor 每个补 cudaEvent encoder/subnet/postproc 三段时间, paper 直接报 breakdown | ~3h 加 instrumentation + 重跑 320 anchor | 数据可信度大幅提升, paper 可承认 ONNX subnet 仅 10% e2e 同时给 D 维度证据 |
| **E. 混合 (B+D)** | 重定位 §C 故事 + 补 e2e timing breakdown | ~10h wall + paper 重写 | 推荐 — 数据可信 + 故事诚实 |

---

*快照时间: 2026-05-13. 之后由用户重新评估实验路径决定下一步.*

---

*维护脚本: 见 §11. 本计划批准后启动 P0.1.*
