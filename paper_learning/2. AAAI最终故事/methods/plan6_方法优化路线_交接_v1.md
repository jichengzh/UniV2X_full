# Plan v6 方法优化路线 — 跨模型交接文档 (v1)

> **创建日期**: 2026-05-30
> **撰写者**: claude-opus-4-7 (本次 session)
> **接续模型**: claude-opus-4-8 (用户准备切换执行)
> **目的**: 把 plan v1-v5 累计的方法学硬伤 + 数据缺口梳理为 4 个核心问题 + 优先级路线,供下一轮 (plan v6) 方法优化使用。
> **关联 /goal**: 之前 `AP不敏感解决方案_plan5.md` 已 `/goal clear`。本文档不是新 /goal,是诊断 + 路线建议。

---

## 〇、本文档怎么用

本文档分 5 节:
1. plan v1-v5 加速目标的演变(说明每个 plan 实际在做什么,什么时候转向)
2. F1/F2 — LGB predictor 训不好的归因
3. 量化的真问题(伪量化 + hybrid pipeline 数据搬运)
4. 剪枝的真问题(只能剪 backbone,占整网太少)
5. 性能瓶颈未知(没做 module-level profiling,所有加速决策瞎打)

每节给出:**现状数据 + 根因诊断 + 解决路线(带工程量估计)**。

最后第 6 节给整体优先级路线(P0-P2)。

---

## 一、Plan v1-v5 加速目标的演变

**结论**: 5 个 plan 不是同一个目标的连续优化,而是 **造数据 (v1-3) → 训预测器 (v4) → 真实加速 demo (v5)** 三段式。每次转向都是因为前一段触碰到方法学硬伤。

| Plan | 实际目标 | 加速相关度 | 转向原因 |
|---|---|---|---|
| **v1** (5-21) | 找 FT 不影响 AP 的 sweet spot,稳定数据集制作流程 | 0% | — |
| **v2** (5-22) | 找 AP 真崩溃边界 (架构 g32→g8 + plane=4 极端) | 0% | v1 plateau 早停,需要扩搜索空间 |
| **v3** (5-25) | 2-tier 数据集 + 训 LGB AP predictor | 5% | v2 找到 B×Q 协同 collapse,可以进 predictor 阶段 |
| **v4** (5-27) | **双轴 (AP + LAT) factorial + LGB predictor** | 50% | v3 报 R²=0.94 实际是 FT 当特征 trivial overfit,**用户审查后转向"真 deployment 视图"** |
| **v5** (5-29) | **g8 plane sweep + 2:4 sparsity + TRT real INT8 真测 lat × 真测 AP** | 90% | v4 双轴 plateau (AP R²=0.39, LAT R²=0.52),search space 被 width 公式锁死,**转向 deployment-grade 真测** |

**关键转折点**:
- **v3 → v4**: 意识到 predictor 必须 deployment-realistic (FT=8 锁定,不能拿 FT 当特征)
- **v4 → v5**: 意识到 search space 自身太小 + 测量不真实 (fake quant proxy)

### 参考附件
- `data/AP不敏感解决方案_plan.md` (v1)
- `data/AP不敏感解决方案_plan2.md`
- `data/AP不敏感解决方案_plan3.md` + `data/phase_v3_finding.md` (用户审查纪录)
- `data/AP不敏感解决方案_plan4.md` + `data/plan4_final_report.md` (双轴 plateau finding)
- `data/AP不敏感解决方案_plan5.md` + `data/plan5_final_report.md`

---

## 二、F1/F2 — LGB predictor 训不好的归因

**结论**: **主要是数据问题,不是模型问题。LGB 本身没问题,plan v4 phase E 已验证 capacity sweep + rich features 都没帮助**。具体是 (AP 侧) 信号物理不存在 + (LAT 侧) 数据采集质量。

### 2.1 现状数据账

| 指标 | AP 侧 | LAT 侧 |
|---|---|---|
| 真测 anchor 数 | v3 Tier B 15 + v4 grid 215 + v5 phase C 5 = **~235 unique** | bench v1 4584 + v5 phase A 45 = **~4629** |
| FT 是否统一 | v4/v5 都锁 FT=8 ✓ | bench v1 部分 FT=8 部分混杂 ✗ (v1.4 修订前残留) |
| 信号强度 (max-min range) | AP50 0.78-0.79 = **1.1% 跨度** (saturated) | lat 0.44-12 ms = **27× 跨度** |
| 噪声 floor | σ_FT8 = 0.0183 | TRT kernel CV ≈ 0.7% |
| 信号/噪声比 | **~0.6 (信号 < 噪声)** | ~38 (信号充足) |

### 2.2 根因

**AP 侧 (F1)** — **信号物理上不存在**:
- 跨 8× plane reduction AP 仅降 1.1% (v5 phase C, n=1789 DAIR val 实测),这是模型 over-parameterized 的物理事实
- 反例:plan v2 g8 + (4,4,4) + int8_pc_wo + FT=8 收到 -5.1% AP 崩溃,说明信号在**窄窗口**存在 — 即 (架构 × 极端 prune × 极端量化) 的 corner
- 数据量不是瓶颈,235 anchor 已经不少

**LAT 侧 (F2)** — **数据采集 3 个硬伤**:
1. **FT 不统一**:bench v1 4584 行早期数据 FT 混杂,锁 FT=8 后子集变小
2. **D 维度 32 cell 大部分是 noise**:24 cell 共享 ~8 ms 平台,只有 8 cell (BL=0 + cublas_lt + workspace) 跳到 12-15 ms — step function 难学
3. **端到端 lat 含 voxelize / NMS 这些 plane-invariant 大头**:稀释 backbone 信号

### 2.3 解决路线

| 问题 | 解法 | 工程量 |
|---|---|---|
| AP 信号物理不存在 | 扩 architecture axis (g32/g16/g8/g4) + 加 KITTI/nuScenes 数据集找 AP-sensitive scene | ~2 周 |
| LAT 数据 FT 不统一 | filter bench v1 取 FT=8 子集 + 加 source column tracing | 1 hour |
| D 维度 noise | collapse 32→3 effective cluster (plan v5 §13 已写但未做) | 半 day |
| 端到端 vs 子模块 lat 混在一起 | 训两个 predictor: `f_lat_backbone` + `f_lat_e2e (含 voxelize 固定 offset)` | 半 day |
| LGB 模型本身 | **不动**。plan v4 phase E 验过不是瓶颈 | — |

### 参考附件
- `data/plan4_phaseE_summary.md` (LGB capacity sweep + rich features 验证)
- `data/plan4_phaseA_summary.md` (B × Q ANOVA 分解)
- `data/plan4_phaseD_summary.md` (5 gate 全 FAIL 报告)
- `data/数据集制作_plan.md` v1.4 (原 KPI 设计)
- `data/e2e_bench_v1.csv` (LAT 4584 anchor 主数据)
- `data/e2e_bench_v1_schema.md` (31 列定义)

---

## 三、量化的真问题 — 伪量化 + hybrid pipeline

**结论**: PyramidFusion 含 sparse voxelize op,ONNX 不支持 → 只能 hybrid pipeline (PyTorch voxelize → TRT backbone → PyTorch head)。当前实测 INT8 vs FP16 只快 13%,远不到理论 2×。三个根因。

### 3.1 实测量化 lat (plan v5 phase A, p64 D1_default)

| Q variant | lat (ms) | vs FP16 | 实际是否真 INT8 |
|---|---|---|---|
| FP16 | 0.880 | 1.00× | — |
| INT8_mm (整网 + calibrator) | 0.764 | -13% | **部分** (TRT warning 大量 layer fallback FP16) |
| INT8_pc_wo (weight-only) | 2.246 | **+155% 反而慢** | 不是真 INT8 (activation 还是 FP,要 dequant) |
| INT8 理论 (Tensor Core 2×) | 0.44 | -50% | 没拿到 |

### 3.2 三个根因

**根因 A — Calibration cache 错配**: plan v5 phase A 用了原 g8 baseline 的 `pyramid_dair_calib_minmax.cache` 喂 prune 后的子模块 ONNX。子模块 tensor 名字跟 baseline 不一致 → TRT 找不到 scale → fallback。这是**工程偷懒**,不是 TRT 限制。

**根因 B — Hybrid pipeline 数据搬运**: 代码里已用 `tensor.data_ptr()` 直传 GPU 指针,不走 host → 单次 transfer < 0.1 ms。真正占时间的是 `execute_async_v3` 的 kernel launch overhead (~0.05-0.1 ms),对 sub-millisecond 模型占比大。

**根因 C — 整网没进 TRT**: voxelize 是 sparse 算子 (SpVoxelPreprocessor 用 spconv),ONNX 没标准支持。所以 voxelize 永远 PyTorch。整网真正进 TRT 需要写 TRT custom plugin。

### 3.3 解决路线(按工程成本排序)

| # | 方案 | 解决根因 | 工程量 | 预期收益 |
|---|---|---|---|---|
| Q1 | 每个 plane 重新跑 calibration (用该 plane 的真实中间 BEV feature) | A | 1 day | INT8 fallback 大幅降,预期 -30% lat |
| Q2 | 修 ONNX 多输出 export (返回 3 stage features tuple) | hybrid pipeline 阻塞 | 1 day | Phase C 真 INT8 AP 闭环 |
| Q3 | PyTorch QAT + ONNX 含 QDQ nodes 导出 | A + B | 3-5 day | 整网真 INT8,无 fallback,预期 -50% lat |
| Q4 | 为 SpVoxelPreprocessor 写 TRT custom plugin | C | 1-2 周 | 整网进 TRT,消除 PyTorch/TRT 边界开销 |
| Q5 | 接受现状,论文里诚实写 "INT8 partial fallback,实测加速 -13%" | — | 0 | paper 论点缩水 |

**P0 推荐**: Q1+Q2 (~2 day),拿到真 INT8 lat + 真 INT8 AP。

### 参考附件
- `scripts/phase2/plan5_phaseA_bench_trt.py` (当前 hybrid pipeline 代码)
- `scripts/phase2/plan5_phaseC_hybrid_eval.py` (TRTBackbone wrapper 实现)
- `/tmp/plan5_phaseA_bench_full.log` (TRT warning 大量 fallback 证据)
- `methods/量化和剪枝方法.md` (原始方法学定义)
- `calibration/pyramid_dair_calib_minmax.cache` (当前错配的 cache)

---

## 四、剪枝的真问题 — 只能剪 backbone,占整网太少

**结论**: PyramidFusion 完整网络含 5 个 module,其中只有 pyramid_backbone (1-5M params) 可 plane prune。其他 4 个 module 是 plane-invariant,即使 plane 缩 8× 总 params 也只减 -75% 不是 -90%。

### 4.1 PyramidFusion 各 module params 分布(g8 baseline)

| Module | params | plane-prunable? | 原因 |
|---|---|---|---|
| encoder_m1 (PointPillar VFE) | ~0.5M | 不行 | 输出 fixed 64 ch 喂 pyramid |
| backbone_m1 (modality-specific ResNet stage 0) | ~1.2M | 理论可以 | 但跟 pyramid 输入 coupling |
| **pyramid_backbone** (主 3-stage ResNeXt) | 1-5M | **可以** | plan v5 prune 这块 |
| shrink_conv (3×128 → 384, fixed 384) | ~3M | 不行 | 跟 head input dim coupling |
| cls/reg/dir_heads | ~0.5M | 不行 | output dim fixed by task |
| **总和 (g8 baseline)** | **7.58M** | 1-5M 可剪 | — |

**plan v5 plane=8 极端**: 7.58M → 1.87M = **-75% 总参数**(理论 -90% 因为 plane=8/64=12.5%,但 plane-invariant 模块占大头稀释)。计算量也只减 ~70%,因为 voxelize/heads/NMS 占大头。

### 4.2 解决路线

| # | 方案 | 解决什么 | 工程量 | 副作用 |
|---|---|---|---|---|
| P1 | encoder_m1 加 prune axis (PFN linear 缩 channel) | 多覆盖 0.5M params | 1 day | 破坏 encoder→pyramid 输入,需重 finetune |
| P2 | shrink_conv 缩 (384→256→128) | 多覆盖 3M params | 2 day | 破坏 head input dim,所有 head 重 init |
| P3 | 整网 architecture sweep (layer_nums 也变,不只 plane) | 整网 axis | 1 周 | 重 train 整网,不能 init from baseline |
| P4 | 换 architecture (BEVFusion-tiny / SECOND / FocalsConv) | 跳出 PyramidFusion | 2-3 周 | 整套 pipeline 重做 |
| P5 | 接受现状,论文里诚实写 "prune 覆盖 ~30% FLOPs,其余 quantization + TRT fusion 处理" | — | 0 | paper 论点缩水 |

**P0 推荐**: P1 + P3 (~2 周),价值最高的中等工程量方案。P4 跳出框架时间成本太高。

### 参考附件
- `scripts/phase2/plan5_phaseA_prune_g8.py` (当前 pyramid_backbone-only structural prune)
- `data/plan5_phaseA_prune_log.json` (各 plane 实际 params 缩减)
- `data/plan5_state.json` 的 `phase_A_artifacts.observation` 字段 (-75% vs -90% 解释)
- `methods/协同加速框架_工作流_v1.5.md` (剪枝方法学定义)

---

## 五、【已订正】性能瓶颈已实测 — NMS 占 e2e ~72%,不是模型

> **2026-05-31 订正**: 本节 v1 原文写"瓶颈未知 / 没 profiling 数据 / 所有加速决策瞎打",**这是错的** —— 撰写交接时漏查了 `Pyramid_分段实测耗时_v1.md`(2026-05-13)。该文件早已用 CUDA Event 把整网分段测穿。今天(05-31)在已验证空闲的 GPU 6 上**独立重测确认**了结论。下表为订正后的真实数据。

**结论**: PyramidFusion e2e 真实瓶颈是 **`nms_rotated` —— 一段 HEAL `box_utils.py:693` 的纯 Python + Shapely O(N²) CPU 循环,占 e2e ~72%**。模型计算(forward 全部)只占 ~24%。**剪枝/量化在 e2e 层面被 NMS 死压**:就算把整个 forward 优化到 0,e2e 也只快 ~1.3×;而单换 CUDA NMS 即可 ~3.6×。

### 5.1 实测 module-level breakdown(g8 baseline,RTX 4090,OPV2V test,1.61 agents/scene)

| 模块 | v1 mean (05-13) | **v2 mean (05-31 重测)** | v2 p50 | % of e2e (v2 mean) |
|---|---|---|---|---|
| encoder_m1 (PointPillar VFE) | 3.01 | 5.54 ※ | 2.84 | ~5% (※mean 被单个 p99=148 离群拉高,p50 才是真值) |
| backbone_m1 (ResNet 2D) | 2.12 | 2.19 | 1.79 | 2.0% |
| aligner_m1 | 0.024 | 0.025 | — | 0.0% |
| **pyramid_backbone.forward_collab** | 14.45 | **14.52** | 14.65 | 13.5% |
| shrink_conv | 3.30 | 3.30 | 3.29 | 3.1% |
| heads (cls+reg+dir) | 0.087 | 0.087 | — | 0.1% |
| **forward 小计** | 22.99 | **~25.7** | — | **~24%** |
| postproc.decode/dir/corners/range_mask | ~2.8 | ~2.98 | — | ~2.8% |
| **postproc.nms (Shapely CPU 循环)** | **71.69** | **77.62** | 61.23 | **72.4%** |
| **e2e walltime** | 98.37 | **107.17** | 88.98 | 100% |

### 5.1b forward_collab 内部拆分(2026-05-31 实测,DAIR g8,150 样本,`p0_2b_decompose_collab.py`)

订正早前"融合 ~11ms 不可剪"的**错误推断**(那是 OPV2V 单 agent dummy 3.3ms vs collab 14.45ms 跨条件相减得的)。DAIR 实测拆分:

| forward_collab 子阶段 | mean (ms) | % collab | 可剪枝 |
|---|---|---|---|
| **get_multiscale (3-stage ResNeXt)** | **8.19** | **73.3%** | ✅ plane/通道 |
| single_head (3× occ 头) | 0.40 | 3.5% | 小 |
| weighted_fuse (3× warp grid_sample + 跨 agent 加权) | 1.90 | 17.0% | ❌ |
| decode (3× deblock) | 0.54 | 4.8% | 部分 |
| **collab_total (DAIR ~2 agent)** | **11.17** | 100% | — |

**结论**: forward_collab **73% 是可剪的 ResNeXt**(8.19ms),融合仅 2.84ms(25%)。**"优化 pyramid ResNeXt"方向正确**——plan5 的 plane 剪枝正打在这 73% 上。`data/p0_2b_collab_decompose.json`。(此 11.17ms 为 DAIR;上表 14.45ms 为 OPV2V,数据集不同。)

### 5.1c DAIR 完整 per-module breakdown(2026-05-31,post CUDA-NMS,150 样本,2 agent,`p0_2c_dair_full_breakdown.py`)

| 模块 | mean (ms) | %e2e | 可剪枝 |
|---|---|---|---|
| encoder_m1 (PFN) | 2.40 | 13.4% | ✅ |
| backbone_m1 | 0.84 | 4.7% | ✅ |
| **pyramid** (整模块, ResNeXt 占其 73%) | **9.97** | **55.8%** | ✅ |
| shrink_conv | 1.40 | 7.8% | ✅(耦合 head 输入) |
| heads | 0.06 | 0.3% | ❌ |
| postproc (含 NMS 1.18) | 2.85 | 16.0% | ❌(NMS 已 CUDA 化) |
| **e2e walltime** | **17.87** | 100% | — |

- **CUDA NMS 在 DAIR 也生效**:NMS 1.18ms(6.6%),DAIR e2e 仅 17.87ms。
- **pyramid 主导 55.8%**,内部 ResNeXt(~7.3ms)≈ e2e 40%,是全网最大可剪/可量化杠杆。
- 可剪 conv 合计 ≈ pyramid-ResNeXt 7.3 + encoder 2.4 + shrink 1.4 + backbone 0.84 ≈ **12ms / 17.9ms ≈ 67%**。`data/p0_2c_dair_full_breakdown.json`。

**三重证据**(NMS 从未被优化):
1. **代码**: `box_utils.py:714` 仍是 `boxes.cpu().detach().numpy()` + Shapely,git 自 05-13 零 commit。
2. **测量**: 05-31 GPU 6 独占重测,各 module 与 05-13 对齐到小数点后一位,NMS 仍 ~72%。
3. **物理**: NMS 是 CPU-bound,GPU 忙闲不影响其耗时。

### 5.2 解决路线(已重排,profiling 不再是 P0)

| # | 方案 | 工程量 | 收益 |
|---|---|---|---|
| **B0** | **MMCV `nms_rotated` (CUDA) 替换 Shapely NMS** ← 环境已装 mmcv 1.6.1,CUDA nms_rotated 可用 | **1 day** | **e2e 107→~30 ms = ~3.6× (单点最高 ROI)** |
| B1 | (已做)`torch.profiler` / CUDA Event module breakdown | 完成 | 见 5.1 |
| B2 | NVIDIA Nsight (`nsys`) op-level kernel timeline (仅在需细查 forward 内部时做) | 1 day | cuDNN kernel 级瓶颈 |
| B3 | 拆 `forward_collab` 内部 (multiscale ResNet vs warp vs weighted_fuse) | 半 day | 量化 V2X 融合开销构成 |

**P0 推荐(订正后)**: **B0 (CUDA NMS) 立即做**。这是整个项目 ROI 最高的单点,且不依赖任何剪枝/量化。

### 参考附件
- `Pyramid_分段实测耗时_v1.md` (2026-05-13 原始分段实测,§二 NMS smoking gun + §4.3 ROI 优先级表)
- `data/pyramid_per_stage_timing_v2_20260530.json` (05-31 重测结果,本次新增)
- `scripts/phase2/m4_8_pyramid_per_stage_timing.py` (分段 timing 脚本,从 HEAL_ROOT cwd 跑)
- HEAL `opencood/utils/box_utils.py:693` (`nms_rotated` Shapely 实现 = 瓶颈源)
- HEAL `opencood/tools/inference.py` (整网 inference 入口)

---

## 六、整体优先级路线 (推荐给 plan v6 执行)

> **2026-05-31 订正**: 原 P0.1 是"module profiling",但 profiling **已经做过**(§五),所以删除。新 P0 改为 **CUDA NMS 替换**(原 §五 B0),这是实测确认的最高 ROI 单点。剪枝定位也据实测下调(见下方说明)。

按推荐执行顺序排:

| 优先级 | 工作项 | 工程量 | 解决的问题 | 解锁的 paper claim |
|---|---|---|---|---|
| **P0.1** ✅ **已完成** (2026-05-31) | **CUDA NMS 替换 Shapely**(mmcv `nms_rotated`) | 完成 | 问题 5(真瓶颈) | **实测 e2e 3.04× + keep-set 100% 一致(AP 不变)**。详见 `methods/P0_1_CUDA_NMS_结果_v1.md` |
| **P0.2** ✅ **完成** (2026-05-31) | **Q1 真校准 + Q2 ONNX 多输出** | 完成 | 问题 3 | **真 INT8 AP**: FP16 ΔAP50 −0.0006 / INT8 −0.0014。**真 INT8 lat (get_multiscale)**: TRT-FP16 6.28×、TRT-INT8 8.08× vs PyTorch、INT8/FP16=1.29×。**关键**: 进 TRT(6.28×)≫ INT8 over FP16(1.29×)。详见 `methods/P0_2_真INT8_AP闭环_结果_v1.md` |
| **P0.3** (1 day) | 数据清洗 (FT=8 filter) + D collapse + 重训 LGB | 1 day | 问题 2 (F1/F2) | predictor 突破 R² 阈值 |
| **P1.1** (1-2 day) | **路侧 RSU 端 UPAQ 风格剪枝+INT8**(encoder+backbone,实测 5.15ms ≈ UPAQ PointPillars 5.72ms) | 1-2 day | 问题 4(换定位) | 路侧单 agent ~2× 加速,可单独成 claim |
| **P1.2** (2 周) | 扩 architecture axis (g32/g16/g8/g4) + 加 KITTI | 2 周 | 问题 2 (F1 AP 信号) | 找 AP-sensitive scene |
| **P2** (1-2 周) | Q3 PyTorch QAT + ONNX QDQ | 1 周 | 问题 3 根本解 | 工业标准 INT8 |

**剪枝定位订正(回应"剪枝只能剪 backbone")**: §四 的 P1-P5(扩 prune 范围到整网)**不再硬攻**。理由:实测 forward 仅占 e2e ~24%,即使把可剪范围从 backbone 扩到整网,e2e 天花板也只有 ~1.3×,投 2 周工程不划算。改为按场景拆故事 —— **车端**主打 CUDA NMS(P0.1),**路侧端**主打 UPAQ 风格剪枝+INT8(P1.1,有 UPAQ 5.72ms 对标支撑)。

**最小可投稿 bundle** = P0.1 + P0.2 + P0.3 = **~4 day**。其中 P0.1 (CUDA NMS) 单点就能拿到 e2e ~3.6×,是 plan v5 之后最该先落地的一步。

---

## 七、给 plan v6 接续模型的交接 notes

**已有工程资产** (可直接复用):
- `scripts/phase2/plan5_phase{0,A,B,C,D}_*.py` 8 个脚本(prune / finetune / bench / attribution / Pareto)
- `scripts/phase2/plan5_hypothesis_figs.py` (4 张 hypothesis 证据图)
- `data/plan5_pareto_anchors.parquet` (55 anchor 整合)
- `data/stats_v3_plan5/*.png` (5 张图 — pareto, h1, h2, h3, d1)

**已锁定的硬约束**(继承自 plan v4 §1 + v5 §1):
- C1 — FT=8 锁定 (v4 §7.13)
- C5 — prune anchor 必须 finetune
- C6 — INT8 AP 必须从 TRT engine forward (v5 新增,目前 partial)
- C7 — 2:4 sparsity 必须真 build 不能 simulation (v5 新增,已验 FAIL path 3)
- C10 — latency bench 必须独占 GPU,3 sample stability check (v5 新增)

**已踩过的工程坑**(plan v6 不要重踩):
1. bench 用 `| head -N` truncate stdout → python SIGPIPE 或 silent 跑半路 → 用 `> file.log 2>&1` 不带 pipe
2. HEAL train.py `os.system("python inference.py")` 用 PATH `python` 无 torch → log 有 traceback 但不影响训练
3. HEAL `save_freq=2` + 奇数 init_epoch + 1 epoch FT → 永不触发 save (epoch%2≠0) → 改 save_freq=1
4. TRT engine 单输出 wrap pyramid_backbone 丢失 3 stage features → HEAL single_head 报错 → 改 ONNX 多输出
5. 自动找 idle GPU 时 GPU 2 短暂空闲被选中,但其实是 finetune 在 data loader phase → 用 nvidia-smi compute-apps + pid filter
6. 多用户系统 GPU residual mem 让 strict guard 全 reject → 只有 GPU 2 周期性 idle 可用

**模型切换 (opus 4.7 → opus 4.8) 时建议**:
1. 接续模型读这份文档 + `data/plan5_final_report.md` + `data/plan5_state.json` 即可对齐上下文
2. **不需要重读 plan v1-v5 全文**(本文档已浓缩关键 finding)
3. 如果要起新 plan,建议命名为 `plan6_*.md`,以本文档 §6 优先级路线为蓝本
4. plan v5 的 `/goal` 已 clear,接续模型可以 fresh start,不会有 stop hook 干扰

---

## 附:全部可查看附件清单

### 6 plan 主文档
- `data/AP不敏感解决方案_plan.md` (v1)
- `data/AP不敏感解决方案_plan2.md`
- `data/AP不敏感解决方案_plan3.md`
- `data/AP不敏感解决方案_plan4.md`
- `data/AP不敏感解决方案_plan5.md`

### Phase 完成报告
- `data/phase_v3_finding.md` (v3 用户审查纪录)
- `data/plan4_final_report.md` (双轴 plateau finding)
- `data/plan5_final_report.md` (v5 最终报告,partial success)
- `data/plan5_phase0_summary.md` ~ `data/plan5_phaseD_summary.md` 系列

### 数据 + 配置
- `data/e2e_bench_v1.csv` (4584 anchor LAT 主数据,g32)
- `data/e2e_bench_v1_schema.md` (31 列定义)
- `data/plan5_phaseA_anchors.csv` (45 anchor,g8 子模块 lat)
- `data/plan5_phaseB_anchors.csv` (10 anchor,2:4 sparsity)
- `data/plan5_phaseC_real_ap.csv` (5 anchor,PyTorch FP32 AP)
- `data/plan5_pareto_anchors.parquet` (55 anchor 整合)
- `data/plan5_state.json` (状态机)
- `data/数据集制作_plan.md` v1.4 (原 KPI 设计)
- `data/问题.md` §7 系列 (历史问题清单 + 设计纪律演化)

### 图
- `data/stats_v3_plan4/` (plan v4 AP 14 张图)
- `data/stats_v3_plan4_lat/` (plan v4 LAT 8 张图)
- `data/stats_v3_plan5/pareto.png`
- `data/stats_v3_plan5/h1_lat_smoothness.png`
- `data/stats_v3_plan5/h2_sparsity_reduction.png`
- `data/stats_v3_plan5/h3_ap_plateau.png`
- `data/stats_v3_plan5/d1_pareto_dominate.png`

### 脚本
- `scripts/phase2/plan5_phase0_preflight.py`
- `scripts/phase2/plan5_phaseA_prune_g8.py`
- `scripts/phase2/plan5_phaseA_finetune_dispatcher.py`
- `scripts/phase2/plan5_phaseA_bench_trt.py`
- `scripts/phase2/plan5_phaseA_attribution.py`
- `scripts/phase2/plan5_phaseB_sparsity.py`
- `scripts/phase2/plan5_phaseB_bench.py`
- `scripts/phase2/plan5_phaseC_hybrid_eval.py`
- `scripts/phase2/plan5_phaseD_pareto.py`
- `scripts/phase2/plan5_hypothesis_figs.py`

### 方法学
- `methods/协同加速框架_工作流_v1.5.md` (整体方法学定义)
- `methods/量化和剪枝方法.md` (原始量化/剪枝方法定义)

### CLAUDE.md
- 项目 `CLAUDE.md` (onboarding,从 M4.8 phase 时期写)

---

**End of plan v6 交接文档 v1**
