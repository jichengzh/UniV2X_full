# LGB 预测器训练数据集制作计划

> **关联文档**:
> - 搜索空间标准: `paper_learning/2. AAAI最终故事/搜索空间一览.md`
> - 已实验数据汇总: `paper_learning/2. AAAI最终故事/result/Plan2_完整实验结果.md` + `Phase1_5_完整通宵汇报.md` + `Plan2_PhaseAP_5D_Pareto.md`
> - 反思与设计纪律: `paper_learning/2. AAAI最终故事/reflection_mistakes.md`(尤其 #19-#31)
> - 整体实验路线: `paper_learning/2. AAAI最终故事/00_故事评估与实验路线_v1.md`
>
> **当前版本**: v1.4 (2026-05-17)
> **状态**: **e2e_bench_v1.csv 1764 行 ✅ canonical** (Pyramid DAIR, e2e n=1789 AP, 21 T × 7 Q × 12 D). Track A v2 13 新 triplet 训练 + a5 1092 新 anchor 已完成 (2026-05-17). Stage 1 预测器研究: **f_AP_full R²=0.989 ✅ 达标**, **f_lat R²=0.729 ❌ 卡 0.73 上限** (问题 6: TRT 内核选择噪声 + D 网格稀疏). Orin 0/250 (TRT cache 跨版本阻塞).
> **创新点**: paper §C "sensitivity-stratified DoE + active learning, K=32-64 达 R²≥0.9". Stage 1 验证: sensitivity_stratified > random > sobol > lhs (best K=128 R²_lat=0.568, R²_ap=0.618). 后续靠 D 维度精细化 (步骤 1+2 = 补完 5×4 网格 + 加 builderOptimizationLevel) 突破 f_lat 上限.

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

## 〇.5、数据集状态 (2026-05-17 v1.4 修订)

> v1.0-v1.2 时期的 §〇.5 (345 行旧数据诊断) + §〇.6 (ABCD 审计) 已被本节替代. 历史决策见 git log + 反思文档.

### 〇.5.1 当前 canonical 主表

**`data/e2e_bench_v1.csv`** (1764 行, 31 列, 4090):
- 21 triplet × 7 Q × 12 D = 1764 anchor (B × Q × D 笛卡尔子集)
- 21 triplet = 原 T1-T8 (8) + Track A v2 T10-T22 (13). T9 因 stage1 grow 不可行被排除.
- 12 D-config = 5 tactic × 1-4 workspace (5×4 网格只覆盖 12/20, **v1.4 待补完**)
- 全部 e2e 单 engine 管线, AP n=1789 (DAIR-V2X val 全量), throughput_fps 单一 lat 指标
- 详见 `e2e_bench_v1_schema.md` (v1.3, 31 列)

### 〇.5.2 已废弃数据源 (不喂 LGB AP 头)

| 旧文件 | 行数 | 废弃原因 |
|--------|------|---------|
| `data/_by_class/class_a_pyramid_full.parquet` (v1.2 P0.2+P0.2.b) | 320 | **AP 来自 dataset_a_main subnet+collab 管线 (n=500), 与 e2e lat 不可比** (问题 5) |
| `data/unified_bench.parquet` AP 列 | 660 | 同上 (混管线) — **lat 列仍可喂 LGB lat 头**, AP 列必须清零 |
| 各 lat-only parquet (pyramid_random / perstage / 4090_dspace / orin_dspace 等) | 252 | 单 lat 维度可入 lat predictor 扩样本池, AP=NaN |
| `baseline_4090.parquet` (跨模型 amota) | 83 | amota ≠ DAIR AP, 仅"跨模型 amota predictor"单独使用 |

### 〇.5.3 Class A/B/C/D 当前进度 (v1.4)

| Class | 范围 | 计划 | 已完成 | 状态 |
|-------|------|------|--------|------|
| **A** Pyramid 4090 + Orin | 21T × 7Q × 20D (v1.4 含 D 扩) | ~2940 | **1764 (4090)** + 0 (Orin) | 4090: 21T 训练完 ✅, D 网格 12/20 覆盖 (待补 8 D + BL 维度) |
| **B** 4090 D-space 扩 (v1.4 重定义为 D 网格补完 + builderOptLevel) | 8 新 (tactic,ws,BL=3) + 12 (D, BL) | ~2940 | 0 | 步骤 1+2 待启动 (本次) |
| **C** Orin D-space 扩 | workspace / 双三 IP / fallback | 170 | 0 | ⛔ 等 Orin 解锁 |
| **D** 跨模型 (V2X HEAL baselines) | F-Cooper / AttFuse / V2X-ViT × 16 | ~48 | 0 | 未启动 (v1.3 撤回原 UniAD-tiny / UniV2X) |
| **合计 4-指标 anchor** | | **~5742** | **1764** | **31%** (Stage 2e 后 → 4704 = 82%) |

---

## 一、总目标规模

### 1.1 数据总量

**总目标 (v1.3): ~908 完整 4-指标 anchor** (Pyramid Fusion 主, 跨 RTX 4090 + Orin AGX 两硬件; 加 V2X HEAL baseline F-Cooper / AttFuse / V2X-ViT transfer 验证).

| 子集 | anchor 数 | 主网格 | 用途 |
|------|-----------|--------|------|
| **A** Pyramid Fusion (4090 + Orin) | **~570** | 4090: 8T × ~10Q × ~5D (含 sparse/edge_only); Orin: 8T × 6Q × 12D | LGB lat/AP/throughput/resource 主训练集 |
| **B** 4090 D-space lat 扩 | **~120** | sparse INT8 + edge_only/cublas_lt tactic + random_search 12 unseen triplet | D 维度泛化 |
| **C** Orin D-space lat 扩 | **~170** | workspace + 双/三 IP + fallback policy | Orin 端泛化 |
| **D** 跨模型 V2X baselines | **~48** | F-Cooper / AttFuse / V2X-ViT 各 16 anchor (4B × 4Q × 1D) | predictor 跨架构 transfer (R²≥0.85) |
| **总计 4-指标** | **~908** | | |

> **v1.3 关键修改**:
> - D 类: **撤回 UniAD-tiny / UniV2X** (用户 2026-05-16 纠正: UniAD-tiny 不适合加速且非 V2X 算法, UniV2X 是端到端 planning), **改 V2X HEAL baseline (F-Cooper / AttFuse / V2X-ViT)** 各 16 anchor 验证 predictor transfer.
> - 总目标从 v1.2 的 720 调到 ~908 (Class A 实际需要更细 Q × D, D 类大幅缩减但加 transfer 验证).

### 1.2 覆盖率 vs 搜索空间

| 硬件 | 理论 cell | 目标真测 | 覆盖率 | 创新点配合 |
|------|-----------|---------|--------|-----------|
| RTX 4090 (Pyramid) | 45,050 | ~370 | 0.82% | 用 K=32-64 anchor 训预测器达 R²≥0.9 (§4.6) |
| Orin AGX (Pyramid) | 517,616 | ~250 | 0.05% | sensitivity-stratified DoE 摊薄成本 |
| V2X baselines (F-Cooper/AttFuse/V2X-ViT × 4090) | ~30,000 each | 16 each | 0.05% | predictor transfer (无需重新满 sweep) |

**v1.3 创新点定调**: 不再追求满笛卡尔积, 用 §4.6 的 sensitivity-stratified DoE + active learning 在每模型上用 ~32-64 anchor 训出 R²≥0.9 预测器.

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

### 2.3 D 维度(部署) — v1.4 扩到 3 子维度

D 维度现由 **3 个子维度** 联合构成 (问题 6 解决方案):

```
d_tactic                ∈ {default, with_cudnn, cublas_lt, all_enabled, edge_only}     # 5 档
d_workspace_gb          ∈ {1, 4, 8, 16}                                                # 4 档
d_builder_opt_level     ∈ {0, 3, 5}                                                    # 3 档 (v1.4 新, 默认 3)
```

**RTX 4090 D-config 全集 (60 个 = 5×4×3, 当前覆盖 12/60)**:

| 阶段 | D-config 数 | tactic × ws × BL | 覆盖 |
|------|----|----|----|
| **v1.3 现状 (已采)** | 12 | 5 tactic × 1-4 ws × BL=3 (子集) | 12/60 |
| **步骤 1 补完网格 (v1.4)** | +8 | (with_cudnn,1GB), (with_cudnn,16GB), (cublas_lt,1GB), (all_enabled,8GB), (all_enabled,16GB), (edge_only,1GB/8GB/16GB), BL=3 | 20/60 |
| **步骤 2 加 builderOptLevel (v1.4)** | +12 | 4 代表 D (default_4gb, cublas_lt_8gb, all_enabled_4gb, edge_only_4gb) × {BL=0, BL=3, BL=5} | 32/60 |
| **v1.4 目标** | 32 | (其中 4 个 (D, BL=3) 与现有 12 个里 4 个重叠 → 重测做噪声地板控制) | 32/60 = 53% |

**D-config 命名约定**:
- BL=3 (默认): `D{N}_{tactic}_{ws}gb` (例: D5_default_1gb)
- BL≠3: `D{N}_{tactic}_{ws}gb_BL{0,5}` (例: D21_default_4gb_BL0)
- 详见 `scripts/phase2/a2_run_one_anchor.py` 的 D_MAP

**Orin AGX**: (v1.3 设计不变, 待解锁)

```
D1: GPU + default + 2GB ... D12: DLA0 + permissive fallback + 2GB
```

> Orin AGX 详细 D-config 见原 v1.3 §2.3 (单 IP / 双 IP / 三 IP × workspace / fallback 共 12 档), Orin 解锁后启用.

### 2.4 HW 维度(硬件)

每 anchor 标注 `hardware ∈ {rtx4090, orin_agx_64gb}`.
跨硬件 paired 的 anchor 用 `paired_id` 链接(同 (B,Q) 在两边各跑一次).

### 2.5 Model 维度(架构) — v1.3 修订

`model_class ∈ {pyramid_fusion, fcooper, attfuse, v2xvit}`, 主体 pyramid_fusion (~860 anchor), V2X HEAL baseline 各 16 anchor 验证 predictor transfer. v1.0-v1.2 时代的 uniad_tiny / univ2x_full 已撤回 (不是 V2X 协同检测算法).

---

## 三、每 anchor 必测的"全方面"性能指标

按用户明确要求(吞吐量 + AP 精度 + 资源使用 + 全维度), 每行至少 18 列:

### 3.1 输入坐标列(11 列)

```
model_class        : str  ∈ {pyramid_fusion, fcooper, attfuse, v2xvit}  # v1.3 修订
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

### 4.1 Class A — Pyramid Fusion 主数据集

**这是 paper §C 主表的数据来源, 最高优先级**.

#### 4.1.1 ckpt 库 (P0.1, ✅ 完成 2026-05-13)

8 个 triplet 通过完整 HEAL 训练 (~25 epoch DDP), 全部过 AP gate (subnet AP50 0.74-0.79). 输出 `models/dataset_a_cache/{triplet_sig}/net_epoch_bestval_atN.pth`. 8 triplet 是 hand-pick 代表 (反思 #24, 仅作 ckpt 库, 不直接喂 LGB).

#### 4.1.2 4090 主网格 e2e 管线 (P0.2 e2e, ✅ 部分完成 224/320)

**已完成 (2026-05-15/16)**: 8T × 7Q × 4D = 224 anchor, e2e_bench_v1.csv 主表.

| 模块 | 范围 | anchor 数 | 状态 |
|------|------|----------|------|
| A.0 baseline 笛卡尔 | 8T × {fp32, fp16, int8_mm, int8_ent} × default D | 32 | ✅ 2026-05-14 |
| A.1 phase1 mix | 8T × {mix_s0, mix_s2} × default D | 16 | ✅ 2026-05-15 |
| A.1 phase2 W-only | 8T × Q_int8_pc_wo × default D | 8 | ✅ 2026-05-15 |
| A.2 D 维度扩 | 8T × 7Q × {with_cudnn/8GB, cublas_lt/16GB, all_enabled/1GB} | 168 | ✅ 2026-05-16 |
| 问题 5 修复 | A.0 32 anchor e2e AP n=1789 重测 + 回填 128 行 | — | ✅ 2026-05-16 |

**待补 (~96 anchor)**:

| 模块 | 范围 | anchor 数 | 阻塞 |
|------|------|----------|------|
| A.1 phase2 pt 变体 | 8T × {pt_wa, pt_wo, pc_wa} × default D | 24 | ⛔ 问题 4 (ONNX Q/DQ 破坏融合) 走方案 2 TRT C++ API ~6h |
| A.4 sparse INT8 | 8T × {2:4 sparse} × {fp16, int8} × default | 16 | 需先实现 ASP weight reload + 联合校准 |
| A.5 random_search 补 unseen triplet | 12 random (B, Q, D) 真测 + 重 baseline | 12-48 | 反思 #28 避免 hand-pick selection bias |

#### 4.1.3 Orin AGX 跨硬件扩展 (P0.3, ⛔ 阻塞)

**当前状态**: 0/250. **阻塞点**: TRT cache 跨版本 (4090 build=10.13.0.35 vs Orin runtime=8.5.2.2), engine 不兼容. 需 patched cache magic TRT-101300 → TRT-8502 (~4h 工程).

**解锁后采集计划**: 8T × 6Q (核心 minmax + 3 mix) × 12D (单 IP × 9 + 双 IP × 4 + 三 IP × 4 + fallback policy × 2) = ~576 potential, build_success ≈ 40% → ~230 真测 + ~340 build_fail negative.

**Orin AP 策略**: 不在 Orin 测 AP (跨硬件 INT8 ±1pp 差异由 4090 AP + ε caveat 处理).

### 4.2 Class B — 4090 D-space lat 扩展 (~120 anchor)

Class A 的 D 维度只覆盖 {default/4, with_cudnn/8, cublas_lt/16, all_enabled/1}. 补:

| 子模块 | 范围 | anchor 数 |
|--------|------|----------|
| B.1 workspace 扩档 {2GB, 32GB} | 3T × 2 prec × 4 tactic × 2 ws | 48 |
| B.2 2:4 sparse INT8 路径 | 3T × 4 tactic × 2 ws | 24 |
| B.3 edge_only single tactic | 3T × 2 prec × 4 anchor | 24 |
| B.4 random_search unseen triplet | 12 random (B, Q, D) | 24 |

总: ~120 anchor. **反思 #28**: B.4 必须 random_search, 不能 hand-pick.

### 4.3 Class C — Orin D-space lat 扩展 (~170 anchor)

依赖 A.3 解锁. 补:

| 子模块 | 范围 | anchor 数 |
|--------|------|----------|
| C.1 workspace 扩档 {4GB, 8GB} | 8T × 2 prec × 2 ws | 32 |
| C.2 双 IP scheme {A, B} | 4T × 2 prec × 2 scheme | 16 |
| C.3 三 IP scheme C | 4T × 2 prec × 2 ws | 16 |
| C.4 fallback policy {strict, permissive} | 5 anchor × 2 policy | 10 |
| C.5 单 IP DLA0/1 全图 | 5 unseen T × 2 prec × 2 DLA target | 20 |
| C.6 INT8 全量补 (Q_int8_mm/ent on Orin) | 8T × 2 Q × 4 D | 64 |
| C.7 buffer 失败重测 | — | 12 |

总: ~170 anchor (含 ~50% build_fail negative).

### 4.4 Class D — 跨模型 V2X HEAL baselines (~48 anchor)

> **v1.3 重大修订**: 撤回原 UniAD-tiny / UniV2X. UniAD-tiny 验证不适合加速且不是 V2X 算法, UniV2X 是端到端 planning 非 V2X 协同检测.

**真正跨模型范围**: 4 个 V2X HEAL baseline (Pyramid m1 已做, 还有 F-Cooper, AttFuse, V2X-ViT). 详见 `model/V2X_baselines_横向对比_v3.md`.

| Model | model class | dataset | 真权重 | 范围 | anchor 数 |
|-------|-------------|---------|-------|------|----------|
| Pyramid m1 | HeterPyramidCollab | DAIR + OPV2V | ✅ | 主数据集 (Class A) | (224 完成) |
| F-Cooper | HeterModelBaseline (MaxFusion) | OPV2V | ✅ HEAL HF | 4 prune × 4 Q × 1 D | 16 |
| AttFuse | HeterModelBaseline (AttFusion) | OPV2V | ✅ HEAL HF | 4 prune × 4 Q × 1 D | 16 |
| V2X-ViT | HeterModelBaseline (V2XViTFusion) | DAIR | ✅ HEAL HF | 4 prune × 4 Q × 1 D | 16 |

**总计**: 48 anchor (3 model × 16). **用途**: 验证 predictor 跨架构 transfer (R²≥0.85), 而不是给 3 个新模型分别训完整满 sweep predictor.

**前置工程**:
1. F-Cooper / AttFuse ONNX export sanity (fusion_net = MaxFusion / AttFusion 兼容性未验证)
2. V2X-ViT MHA INT8 量化掉点 (paper §C anti-pattern 数据点, "framework 自动避开 transformer INT8")
3. Dataset mismatch (F-Cooper/AttFuse 在 OPV2V, V2X-ViT 在 DAIR): predictor 加 `dataset` 作为 feature

### 4.5 旧数据处理 (废弃 v1.2 整合方案)

> v1.2 §4.5 计划 "整合 345 行旧数据到 unified_bench" — **AP 列因 问题 5 同类问题 (混管线) 全部不可信**.

- **e2e_bench_v1.csv 224 行 = canonical 主表**, paper §C 主表只用此
- `data/unified_bench.parquet` (660 行) lat 列仍可作 LGB lat 头扩样本池, **AP 列必须清零**
- `data/baseline_4090.parquet` (83 行 amota) 单独喂"跨模型 amota predictor", 不混 Pyramid AP

### 4.6 创新点 — Sensitivity-stratified DoE + Active Learning (v1.3 新)

**Paper §C 核心 claim**: "在 B×Q×D 三维搜索空间上, 用 K=32-64 anchor (而非满笛卡尔 224) 训 LGB lat/AP 预测器达到 R²≥0.9, sampling 效率比 random 高 3-5×."

**方法论分阶段**:

| Phase | 内容 | 数据需求 | 工时 |
|-------|------|---------|------|
| **α — Sampling efficiency study** | 用现 224 anchor 当 ground truth. 对比 4 sampling (random / LHS / Sobol / sensitivity-stratified) × 6 K size (16/32/48/64/96/128) → LGB R²(lat) + R²(ap) on hold-out | **0 新数据** | 3-4h |
| **β — Active learning loop** | α 最优 sampling 起点 K=32, predictor 跑全 224 cell uncertainty, 选 top-8 加入, 重训, K=32→40→…→96, 看 R² 收敛 | 0 新数据 | 2h |
| **γ — Per-dim sensitivity decomposition** | 训 3 子预测器: f_lat(B,Q,D), f_AP_baseline(B), f_AP_crash(Q, ckpt_source). 证 AP 主要跟 (ckpt_source, Q=entropy) 相关, 跟 B/D 几乎独立 → 减少采样 | 0 新数据 | 1h |

**Paper §C 主图**: K-R² 曲线 (random vs LHS vs sensitivity-stratified) + active learning 收敛速度 + per-dim importance 直方图.

**产出脚本**: `scripts/phase2/predictor_efficiency_study.py` (待写).

---

## 五、工程时间预算 (v1.3 新)

> 6 GPU 4090 + 1 Orin AGX 独占可用. 224 anchor 现状下剩余工作量, 统一 Stage 命名 (与 `创新点.md` §2/§4 一致):

| Stage | 任务 | 数据需求 | 工时 | 累计 wall | 状态 |
|-------|------|---------|------|-----------|------|
| **Stage 1 — 方法论验证 (基于 oracle, 0 新数据)** | | | | | |
| 1a | sampling efficiency study (4 sampling × 6 K × 5 repeat) | 0 | 3-4h CPU | 0.5 day | ✅ 完成 (2026-05-17) |
| 1b | active learning loop (K=32→96, step=8) | 0 | 2h | 0.6 day | ✅ 完成 |
| 1c | sub-predictor decomposition (f_lat, f_AP_baseline, f_AP_crash) | 0 | 1h | 0.7 day | ✅ 完成. f_AP_full R²=0.989 ✅, **f_lat R²=0.729 ❌ 卡上限 (问题 6)** |
| **Stage 2 — Pyramid 数据集补全** | | | | | |
| 2a (旧) | A.1 phase2 pt 变体 (TRT C++ API set_input_range 解锁) | +24 | 7h | 1.0 day | 🟡 deferred (Q_int8_pc_wo 8 anchor 已够) |
| 2b (旧) | A.3 Orin 跨硬件 (TRT cache patch + 250 anchor 采集) | +250 | 4h 工程 + 24h Orin | 2.5 day | ⛔ 阻塞 |
| 2c (旧) | B/C class D-space (4090 sparse/edge_only + Orin workspace/多 IP) | +290 | 4h 4090 + 12h Orin | 3.5 day | 🟡 部分完成 (4090 sparse 已撤回, edge_only 已并入 D 维度补全) |
| **2d (v1.4 新) — Track A v2 + a5 (B 维度扩 + 21 T × 7 Q × 12 D)** | +1092 | 12h | 1.5 day | ✅ 完成 (2026-05-17, 21 T × 7 Q × 12 D = 1764 csv) |
| **2e (v1.4 新) — D 维度精细化 (解决问题 6)** | | | | | |
| 2e.1 步骤 1 | 补完 5×4 网格 (8 新 (tactic, ws) × 21T × 7Q, BL=3 默认) | +1176 | **~5h on 6 GPU** (实测 BL=3 每 anchor ~100s) | 1.9 day | 🔲 待启动 |
| 2e.2 步骤 2 | 加 builderOptimizationLevel (4 代表 D × {BL=0, BL=3, BL=5} × 21T × 7Q, BL=3 588 anchor 是噪声地板控制重测) | +1764 | **~15h on 6 GPU** (BL=5 build 388s vs BL=3 72s, 5×!) | 2.5 day | 🔲 待启动 |
| **Stage 3 — 跨模型 transfer (V2X HEAL baseline)** | F-Cooper / AttFuse / V2X-ViT 各 16 anchor | +48 | 6h | 2.1 day | 🔲 待用户决策 (F-Cooper/AttFuse 只有 OPV2V 版, 需重训 DAIR) |
| **Stage 4 — LGB 重训 + Pareto demo** | 训练 + 5-fold CV + paper §C 主图 | 0 | 2h | 2.2 day | 🔲 待 Stage 2e + 3 完成 |

**总 wall (剩余)**: **~20h (2e 步骤 1+2 = 2940 新 anchor)** → ~26h (含 Stage 3 V2X-ViT-only) → 3 day 全部完成. (BL=5 build 时间是主要瓶颈, 占 step 2 wall 75%.)

**关键里程碑**:
- ✅ Day 0.7: Stage 1 完成, 创新点 K-R² 曲线已绘 (sensitivity_stratified 最优)
- ✅ Day 1.5: Stage 2d 完成, oracle 224 → 1764 anchor
- 🔲 Day 1.9: Stage 2e 完成, 期待 f_lat R² 0.73 → 0.85+
- 🔲 Day 2.2: paper §C 主图全部数据就绪

---

## 六、数据落点 + Schema 标准

### 6.1 文件路径 (v1.3 更新)

```
paper_learning/2. AAAI最终故事/data/
├── e2e_bench_v1.csv / .parquet      ← canonical 主表 (224 行, 31 列, paper §C 数据源)
├── e2e_bench_v1_schema.md            ← schema 定义 (v1.3, 31 列)
├── dataset_analysis.md               ← 每模块完成后的 sanity 分析
├── 问题.md                            ← 累积问题清单 (5 条)
├── 纪律.md                            ← 实验纪律 self-check
├── PROGRESS.md                       ← 模块级进度看板
└── figs/                             ← BQD 3D 分布图 等

data/  (旧路径, deprecated AP, lat-only 仍可用)
├── unified_bench.parquet            ← 660 行, AP 列废弃, lat 列仍喂 LGB lat 头
└── _by_class/legacy/                ← v1.2 时代散落 parquet
```

### 6.2 Schema 简明 (e2e_bench_v1, 32 列 v1.4)

坐标 13 列: `triplet, stage{0,1,2}_planes, prune_object, sparse_mask, q_tag, prec_flag, q_bits, q_bits_per_stage, q_granularity, q_object, d_scheme, d_tactic, d_workspace_gb, d_builder_opt_level (v1.4 新, 默认 3), hardware`.

性能 1 列: `throughput_fps` (单一 lat 指标, 问题 2 §2.5 精简).

AP 3 列: `ap30, ap50, ap70` (e2e 单 engine n=1789, 问题 5 修复后 canonical).

资源/状态 6 列: `engine_size_mb, build_secs, build_success, fail_reason, n_collected, n_skipped, max_voxels, device, real_voxels_{mean,p99}, ts`.

详细列定义见 `e2e_bench_v1_schema.md`.

### 6.3 traceability

每行 `ts` (ISO 8601 时间戳) + `device` 字段记录采集环境. 复现路径: anchor `(triplet, q_tag, d_tactic, d_workspace_gb)` 唯一索引 + `models/e2e_cache*/{tag}.engine` + `results/{a0_refresh_ap, a1_q_expand, a2_d_expand}/{tag}_ap.json` 详细 AP 输出.

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

本节是 v1.0-v1.1 时代的训练加速协议, P0.1 完成后已经落地, 详细细节可见 git log. 当前阶段 (v1.3) 已用不到大批量重训, 仅 β.1 pt 变体可能需要短 finetune, 沿用既定 4-层精度保障:

- **P1**: 完整 HEAL stage1 训练 (~25 epoch), Class A 8 triplet 已应用
- **P4**: Convergence Gate 硬验收 — `AP50 ≥ baseline - 5pp` 否则标 `not_converged` 不进 LGB
- **训练加速**: DDP 4 卡 + AMP + batch_size 上调 + voxel 预缓存, 复合 ~5× ↑ (单 triplet 12.5h → 2.5h, 已用于 P0.1)
- **AP Eval**: 双 engine (subnet + collab) 消除 PyTorch fallback, 当前已升级为 e2e 单 engine n=1789 (问题 5 修复后)

---

## 九、验收标准 (v1.3 修订)

1. **canonical 主表**: `data/e2e_bench_v1.csv` ≥ 600 行 (Pyramid 4090 ~320 + Orin ~250 + V2X baseline 48)
2. **覆盖率**:
   - Pyramid 4090: AP 真测 ≥ 248 anchor (Class A 4090 完整), 占 4090 搜索空间 0.55%
   - Pyramid Orin: lat 真测 ≥ 200 anchor, build_success 标注 ≥ 250 anchor (含 negative)
   - V2X baseline transfer: 48 anchor (3 model × 16)
3. **创新点验证 (§4.6)**: K=32-64 anchor 训预测器 lat R² ≥ 0.9, AP R² ≥ 0.85 on hold-out
4. **AP 管线统一**: 所有 AP 列来自 `e2e_eval_ap.py` (e2e 单 engine n=1789), 不混 subnet
5. **跨模型 transfer**: predictor 在 V2X baseline 各 16 anchor 上 R² ≥ 0.85 (允许 fine-tune)
6. **Pareto demo**: 4090 + Orin 各产 ≥ 20 个 Pareto 点, 真测 AP 验证 Top-5 MAE ≤ 5 pp

---

## 十、对应实施脚本 (v1.3)

```
scripts/phase2/
├── a0_refresh_ap.py                  ← ✅ 问题 5 修复 (A.0 32 anchor e2e AP n=1789 重测)
├── a1_run_one_anchor.py              ← ✅ A.1 phase1 mix orchestrator
├── a1_run_one_anchor_phase2.py       ← ✅ A.1 phase2 W-only orchestrator
├── a1_dispatch_parallel.sh           ← ✅ A.1 6-GPU 并行 dispatcher
├── a2_run_one_anchor.py              ← ✅ A.2 D-dim orchestrator (含 --tactic CLI)
├── a2_dispatch_parallel.py           ← ✅ A.2 6-GPU 并行 dispatcher
├── a2_merge_rows.py                  ← ✅ A.2 row JSON 合并到 csv
├── plot_bqd_3d.py                    ← ✅ BQD 三维分布可视化
├── onnx_full_qdq_preprocess.py       ← ✅ ONNX W+A Q/DQ 预处理 (走 问题 4 方案 1 用 --w-only)
│
├── predictor_efficiency_study.py     ← 🔲 Phase α (创新点验证, 待写)
├── a14_phase2_pt_via_trt_capi.py     ← 🔲 β.1 pt 变体 TRT C++ API 解锁
├── orin_class_a_int8_sweep.py        ← 🟡 β.2 Orin INT8 补救 (patched cache 待启动)
├── dataset_b_4090_dspace_v2.py       ← 🔲 β.3 4090 B class sparse + edge_only
├── dataset_c_orin_dspace_v2.py       ← 🔲 β.3 Orin C class workspace + 多 IP
├── dataset_d_v2x_baselines.py        ← 🔲 γ 跨模型 V2X baseline transfer
└── train_lgb_v6_efficiency_study.py  ← 🔲 δ LGB 重训 + Pareto demo

scripts/phase1/
└── m4_8_trt_build_bench.py           ← ✅ TRT build + bench (含 --tactic, --w-only flags)
```

---

## 十一、修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-12 | 初版 |
| v1.1 | 2026-05-13 | + §〇.5 旧数据可用性诊断 + §〇.6 ABCD 推进现状审计 |
| v1.2 | 2026-05-13 | + §十三 进度快照 (P0.1/P0.2/P0.2.b/P0.4 完成, P0.3 Orin 跑中) |
| **v1.3** | **2026-05-16** | **重写 §〇.5 (废弃 subnet 320 行, e2e_bench_v1.csv 224 行成主表), §一 (撤回 720 满笛卡尔, 加 ~908 总计 + 创新点), §四 (Class A/B/C/D 全部修订, 撤回 UniAD-tiny/UniV2X, 加 V2X baseline transfer, 加 §4.6 sampling efficiency), §五 (新执行路线 α→β→γ→δ 4 day), §六 (主表换 e2e_bench_v1.csv), §八之零 / §十三 (精简过时细节). 删除 §十三 v1.2 历史快照** |
| **v1.4** | **2026-05-17** | **Track A v2 + a5 完成 (224 → 1764 anchor, 21 T × 7 Q × 12 D). Stage 1 完成 (f_AP_full R²=0.989 ✅, f_lat R²=0.729 ❌). 加 §2.3 D 第 3 子维度 d_builder_opt_level (TRT 10.13 builderOptimizationLevel 0/3/5). 加 §五 Stage 2d (完成) + Stage 2e (步骤 1 补完 5×4 + 步骤 2 加 BL). §六 schema 32 列. 加 问题 6 (f_lat 上限根因 + 解决方案).** |

---

## 十二、关键风险 (v1.3)

| 风险 | 应对 |
|------|------|
| Orin TRT cache 跨版本不解锁 | β.2 阻塞: patched cache magic TRT-101300 → TRT-8502, ~4h 工程 (orin_class_a_int8_sweep.py 已准备) |
| A.1 phase2 pt 变体被 问题 4 阻塞 | 走方案 2 TRT C++ API (~6h 工程) 或 accept 8 行 pc_wo 已足够 |
| V2X baseline ONNX export 不兼容 (F-Cooper/AttFuse fusion_net) | γ 前先 export sanity, 失败 fallback 到 OPV2V Pyramid OPV2V_orig sweep 补 (P0.2 时代未启用的 OPV2V Pyramid ckpt 可启用) |
| 创新点 K-R² 曲线不显著 | Phase α 失败则砍 paper §C sampling 创新点, 回退到"完整笛卡尔" Pareto 叙事 |
| AP 管线再出问题 | 任何新 anchor 走 e2e_eval_ap.py n=1789 + 跟 csv 抽样校验 (问题 5 教训) |

---

*plan 完整版本控制见 git log. 当前 working set 见 §4 + §5.*
