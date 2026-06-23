# 论文局限性 (Living Document)

> AAAI 2026 论文 limitations + caveats 总览. **每次实验后必须更新此文件**, 反映新的已知/未知边界.
>
> **更新协议** (强制):
> 1. 跑完一个新 anchor / 一个新 ckpt / 一个新 platform → 在 §"更新日志" 末尾追加一行
> 2. 如果新实验**消除**已有 limitation → 把对应条目从 §"已知局限性" 移到 §"已解决局限性", 并标注解决方式 + 实证日期
> 3. 如果新实验**暴露**新 limitation → 在 §"已知局限性" 添加新条目, 引用具体测量 ID
> 4. 任何对外报告 (paper draft / reviewer 回复 / slide deck) 引用 limitation 之前, 必须确认本文件是当前 session 最新版
>
> 关联: `results/m4_9_framework_pareto_report_v2.md` Caveats 段是本文的一个子集摘录, 论文 §6 "Limitations" 段也从此文件抽取.

---

## 0. 文件元信息

| 字段 | 值 |
|---|---|
| 创建日期 | 2026-05-10 |
| 最后更新 | 2026-05-11 (P0 完成: 100 random + LGB 训练) |
| 当前 framework Pareto 版本 | M4.9 v2 + P0 (12 anchor + 100 random bench + LGB lat predictor) |
| 当前实测 anchor 总数 | 12 hand-pick (RTX 4090: 11, Orin: 1) + 100 random bench (RTX 4090) |
| 已 framework-level 修复的限制数 | 1 (width_per_group power-of-2) |
| 当前 Pareto 前沿 anchor 数 | 4 (A2 / A6 / A10 / A11) |
| **LGB lat predictor (Pyramid 单模型)** | **hold-out spearman 0.88 / MAE 0.038 ms (4.9% rel)** |
| **LGB vs rule-based 对比** | **LGB MAE 0.012 vs rule 0.115 ms (10× better)** on 6 hand-pick anchors |

---

## 1. 已知局限性 (Active limitations)

### 1.1 Baseline checkpoint provenance (HEAL 发布 ckpt)

**事实**:
- 我们用 HEAL 作者 Yifan Lu 在 HuggingFace (`yifanlu0227/HEAL`) 发布的 `Pyramid_DAIR_m1_base_2023_08_14_11_42_29` 作 baseline
- 训练时间 2023-08-14, 本机下载 2024-01-25
- HEAL 作者用 spconv 1.2.1 训, 我们环境 spconv 2.3.8

**HEAL README 原文警告**:
> "Those checkpoints has a faulty input channel number for **SECOND related models**, but you can still run them with spconv 1.2.1... To develop your model, please do not use these checkpoints." — Issue #20

**警告所指 "SECOND 相关模型" 具体范围** (实证 grep):

| 类别 | 文件清单 |
|---|---|
| 模型类 | `opencood/models/second.py`, `second_ssfa.py`, `second_intermediate.py`, `second_ssfa_uncertainty.py`, `heter_encoders.py::class SECOND` |
| m3 modality yaml | `*/HEAL/stage{1,2}/m3_pyramid.yaml`, `*/MoreModality/3_modality_*/m1m2m3_*.yaml` (7 个), `4_modality_*/m1m2m3m4_*.yaml` (7 个), `Single/m3_SECOND32_pretrain.yaml`, `dairv2x/Single/DAIR_single_m2_second.yaml` |
| 受影响发布 ckpt | `checkpoints/stage2/m3_alignto_m1` |

**bug 出在哪里**: SECOND 的 VoxelFeatureExtractor (VFE) → 3D Sparse Convolution 接口处, input channel 数被错误标注. spconv 1.2.1 没 sanity check 仍能跑 (silent degradation), spconv 2.x 严格 check 会拒载或行为异常.

**对我们工作的实证影响**:
- 我们用 m1 = **PointPillar** (`core_method: point_pillar`), **不在 SECOND 警告名单**
- m1 VFE 用 `nn.Linear` + `scatter_max` (PyTorch 原生 op), **不依赖 spconv 3D 稀疏卷积**
- 实测复现: AP50 = **0.7907** (DAIR-V2X val 1789, force_fallback 全 PyTorch) 与 HEAL paper 报告一致 → ckpt 在 m1 上**功能正常**
- 跨 spconv 1.2.1 → 2.3.8 对 m1 是 benign

**残留风险**:
- HEAL 作者还有更广 disclaimer "To develop your model, please do not use these checkpoints" — 我们做 framework 开发不是纯复现, 严格说应该自训
- Reviewer 可能要求消除 provenance 风险 → 自训 m1 baseline (~10h on 4090)

**论文应对方式**: caveat 段透明陈述 + claim "framework Pareto 是相对该 baseline 的 trade-off, 不是绝对 SOTA"

### 1.2 跨平台 M 维度只有 1 个 Orin anchor (FP16 only)

**事实**:
- M 维度实测 anchors: RTX 4090 上 9 个, Orin AGX 上 **仅 1 个** (A9, FP16 collab)
- Orin AGX INT8 calibration cache 与 RTX 4090 (TRT 10.13) 不可移植 (Orin TRT 8.5.2)
- DLA0 / DLA1 / CPU 路由全部 **未实测**

**残留风险**: 跨平台 ratio 仅有 FP16 一档实测 (37.7×), 不足以验证 M2 fitting (R²>0.995) 在 INT8 / pruned 场景下仍成立

**未来可能的解决**: Orin 上重做 INT8 calibration (transfer .npy + Python int8 calibrator, ~1h SSH 工程量)

### 1.3 搜索空间未覆盖维度

| 维度 | 已覆盖 | **未覆盖** |
|---|---|---|
| `prune_object` | channel + none | head / element / 真 2:4 (ASP retrain) |
| `prune_criterion` | L1 only | Taylor / FPGM / Wanda |
| `q_object` | none + SPARSE_WEIGHTS flag (noop) | 真 2:4 sparse weights |
| `d_routing` | GPU only | DLA0 / DLA1 / CPU |
| Platform | RTX 4090, Orin AGX (FP16 only) | Orin Nano, Jetson series, A100 等 |

**残留风险**: cube 覆盖率仅 ~30%, 论文 reviewer 可能 "为何只 L1 不 Taylor"

**论文应对**: 强调 framework 是 **可扩展架构**, capability YAML 是 plug-in (新硬件/新算子加 yaml + 新 constraint), 而非 minimal coverage claim

### 1.4 跨模型只 Pyramid_m1 一个

**事实**: M5 跨模型 (UniV2X full + uniad_tiny) **未启动**

**残留风险**: 论文 framework 通用性 claim 需要 ≥ 3 model

**计划**: M5 跨模型扩展, 把 closed-loop pipeline + 10-anchor 模板套到另两个模型 (各 ~半天工程)

### 1.5 Predictor 是 rule-based 不是 learned

**事实**: M4.9 v2 closed-loop demo 用的 predictor (`scripts/phase2/m4_9_closed_loop_v2.py::predict_lat_ms` / `predict_ap50`) 是 **rule-based hand-tuned**, 不是 LGB 学习模型

**原因**: Pyramid 实测 anchors 仅 10 个, 不够训 LGB (我们之前 LGB v6 amota predictor 是用 UniV2X 200+ anchors 训的, in-sample MAE 0.010)

**残留风险**: Reviewer "为何不用 learned predictor" → 答: 数据稀疏 + 跨模型时各模型 anchor 数都不够; 只 rule-based + 实测验证

**论文应对**: 把 framework 论点定位为 "search-space-and-constraint structure" 而非 "predictor accuracy", lat/AP MAE 都靠实测验证

### 1.6 Multi-agent fallback 9.6%

**事实**: hybrid AP eval 中 ~171/1789 (9.6%) 样本走 PyTorch fallback (record_len=[1] single-agent 样本, collab N=2 engine 不适用)

**残留风险**: 严格说 lat 报告里这 9.6% 跟 90.4% TRT 不在同一坐标系, AP 报告也是 hybrid

**论文应对**: 已 transparent 报告 `n_trt_collab_path` / `n_pytorch_fallback`, 不 hide

### 1.7 prune25 lat 异常 (已 framework-level 修复但 anchor 保留)

**事实**: A5/A6 prune25 lat = 2.71-2.87 ms 比 baseline 慢 3.35× (width_per_group=3 出 IMMA fast-path)

**framework 修复**: 添加 `_check_resnext_width_pow2` empirical constraint, 后续 search 不会再产生 width_per_group ∉ {1,2,4,8,16,32} 的配置

**anchor 保留意义**: 作为 "framework 不带 hardware-aware constraint 时会推荐的坑" 的反例数据点

### 1.8 Pyramid baseline AP50 0.7907 vs HEAL 论文报告

**事实**: 我们复现 AP50 = 0.7907 在 DAIR-V2X val 1789 上, 与 HEAL paper 报告一致

**残留风险**: 没有自训 baseline 对照, 万一 HEAL ckpt 跨 spconv 1.2.1→2.3.8 有 0.5pp 级 silent degradation, 我们检测不到

**未来可能的解决**: 自训 m1 baseline (~10h finetune) 完全消除风险

---

## 2. 已解决局限性 (Resolved)

### 2.1 ✅ width_per_group 非 power-of-2 (R: 2026-05-10)

**原 limitation**: prune25 实测 lat 反而比 baseline 慢 3.35× (width_per_group=3 不在 IMMA fast path)

**解决方式**:
- `framework/constraints.py::_check_resnext_width_pow2` (empirical constraint)
- `framework/capability_schema.py::Features.tensor_core` (hw-feature flag)
- `configs/hardware/rtx4090.yaml::features.tensor_core: true`
- `framework/constraints.py::_check_cross_layer_gradient` 修正 (只对多个被剪枝模块检查 spread)

**实证**: M4.9 v2 closed-loop demo (`scripts/phase2/m4_9_closed_loop_v2.py`) 跑 21 raw configs → 12 个被新 constraint 拒绝, 9 个 legal. predicted vs measured 误差: lat MAE 0.001 ms / AP50 MAE 0.015 pp. 相关报告: `results/m4_9_closed_loop_v2_report.md`

### 2.2 ✅ A2 measurement context stale state (R: 2026-05-09)

**原 limitation**: TrtCollabN2 复用 execution context 导致 AP 0.32 (broken)

**解决方式**: `scripts/phase1/m4_8_hybrid_infer_ap.py::TrtCollabN2.__call__` 每次调用创建 fresh `engine.create_execution_context()`, 从 input tensor pointer 分配 buffer (匹配 `trt_run_collab` 验证模式)

**实证**: AP 从 0.32 (broken) → 0.78 (correct, 与 force_fallback baseline 一致)

---

## 3. 不 claim 列表

1. ❌ "2:4 sparsity 给 free 加速" — A8 是 SPARSE_WEIGHTS flag noop, 需 ASP retrain
2. ❌ "Pruning 越多越快" — A5 (25%) 比 A2 (0%) 反而慢, 是反例
3. ❌ "Orin INT8 PTQ 已 ready" — 需 cross-platform calibration
4. ❌ "复现/超越 HEAL paper SOTA" — 我们用 HEAL 发布 m1 ckpt 作 framework 输入, 不重训, 只 claim 相对 baseline trade-off
5. ❌ "跨模型 framework 通用性已实证" — 当前仅 1 个 model (Pyramid_m1), M5 未启动
6. ❌ "DLA / NPU 路由优化已实证" — D 维度仅 GPU 实测
7. ❌ "Learned predictor 比 rule-based 更准" — 当前 demo 用 rule-based, 数据不够训 LGB

---

## 4. 与对比工作的差异声明

### 4.1 vs QuantV2X (CVPR 2025)

| 维度 | QuantV2X | 我们 (M4.9 v2) |
|---|---|---|
| 量化粒度 | per-layer + AdaRound | per-tensor INT8 + per-layer mixed precision |
| Sparsity | block-wise structured | L1-channel structured + SPARSE_WEIGHTS flag |
| Codebook | ✅ V2X 通信压缩 | ❌ 不在范围内 |
| 跨平台 M | ❌ 仅 4090 | ✅ 4090 + Orin AGX (FP16) |
| Framework search | ❌ hand-tuned ablation | ✅ random_search + constraint propagation + Pareto |
| 主实验 backbone | **PointPillar m1** (DAIR/OPV2V) | **PointPillar m1** (DAIR-V2X val 1789) |
| SECOND 主实验 | ❌ DAIR/OPV2V 不用 (仅 V2X-Real m3/m4) | ❌ 不用 |

**对比公平性**: 我们和 QuantV2X 都在 DAIR-V2X 上用 PointPillar m1, 评测同源 — 直接可比.

**SECOND 警告对双方对比无影响**: 双方 DAIR 主实验都不涉及 SECOND.

### 4.2 vs HEAL (paper baseline)

我们**不是** HEAL 的 reproduce / improve, 而是把 HEAL 模型当 framework optimize 的输入. 实证 baseline AP50 0.7907 与 HEAL 一致. Framework 给 (lat, AP) trade-off 是相对 baseline 的, 不是 SOTA claim.

---

## 5. 更新协议 (Update protocol)

### 5.1 何时必须更新本文件

- [ ] 跑完一个新 anchor (新 prune rate / 新 precision / 新 platform)
- [ ] 跑完一个新 model 的 closed-loop demo
- [ ] 加新 hardware-aware constraint 到 `framework/constraints.py`
- [ ] 加新 platform 的 capability YAML
- [ ] HEAL / QuantV2X / 其他对比工作发布更新, 影响本文件 §4

### 5.2 更新内容要求

每次更新必须:
1. 在 §0 文件元信息更新 "最后更新" + "当前 anchor 总数"
2. 在 §6 更新日志末尾追加一行 (不删旧行)
3. 如果消除限制 → §1 → §2 转移
4. 如果暴露新限制 → §1 添加, 引用 measurement file

### 5.3 不要做的事

- ❌ 不删除已有 limitation (即便已解决, 也保留在 §2 历史)
- ❌ 不模糊化具体数据 ("AP 略有变化" → 必须给具体 pp 值)
- ❌ 不无 evidence 添加 limitation (每条都必须能 trace 到具体 measurement / file)

---

## 6. 更新日志

| 日期 | 事件 | 影响 limitation # | 文件引用 |
|---|---|---|---|
| 2026-05-10 | 创建本文件, 整合 M4.8 + M4.9 v1 + M4.9 v2 caveats | 全部 §1 + §2 | `results/m4_9_framework_pareto_report_v2.md` |
| 2026-05-10 | M4.9 v2 closed-loop demo 完成 (10 anchors → 9 legal + 12 rejected by new constraint) | §2.1 (新增) + §1.7 修复 framework 层面 | `results/m4_9_closed_loop_v2_report.md` |
| 2026-05-10 | A9 Orin AGX FP16 collab 实测 (47.92 ms, 37.7× slower than 4090 FP16) | §1.2 (跨平台覆盖度从 0 → 1 anchor) | `results/orin_fp16_build_bench.log` |
| 2026-05-10 | A8 INT8 + SPARSE_WEIGHTS 实测 (0.820 ms, AP50 0.7910 — flag noop) | §1.3 q_object 列实证为 noop | `results/m4_8_dair_collab_trt_int8_sparse.json` |
| 2026-05-10 | A7 mixed precision (heads-FP16 / backbone-INT8) 实测 (0.957 ms, AP50 0.7903) | §1.3 q_bits 列从全局精度扩到 per-module | `results/m4_8_hybrid_ap_dair_collab_mixed.json` |
| 2026-05-10 | A5/A6 prune25 实测 (lat 2.71-2.87 ms — 反而慢 3.35×) | §1.7 (新增) — 暴露 width_per_group 问题 | `results/m4_8_dair_pruned25_ft_collab_trt_*_clean.json` |
| 2026-05-10 (晚) | A10 (prune75 FT FP16) + A11 (prune75 FT INT8) 实测. A11 lat **0.636 ms / engine 4.41 MB / AP50 0.7684** dominate A4 prune50 INT8 in 全部 3 维 → Pareto frontier 从 3 anchor 扩到 **4 anchor (A2/A6/A10/A11)**, A4 掉出 Pareto. 证实 `_check_resnext_width_pow2` constraint 价值: prune75 widths [1,2,4] 全 power-of-2 → IMMA fast-path → 比 prune50 还快 | §1.7 加固 (实证消除该限制必要性已 framework-level 解决) | `results/m4_8_dair_pruned75_ft_collab_trt_{fp16,int8}_clean.json` + `results/m4_8_hybrid_ap_dair_pruned75_ft_collab_{fp16,int8}.json` |
| 2026-05-11 | **P0 完成 — Pyramid 单模型 LGB 闭环**. 步骤: (P0.1) 加 `PYRAMID_M1_MODULES=("model",)` schema; (P0.2) 实证 enumerate 出 98 个合法 (s0,s1,s2) triplets; (P0.3+P0.4) random_search 取 50 个 triplets, 每个 build FP16+INT8 = **100 真实 lat 数据点** (`data/pyramid_random_bench.parquet`, GPU 4 sequential ~167 min); (P0.5) 训 LGB lat predictor (102 sample = 100 random + 2 prune50 anchor), hold-out **spearman 0.88 / MAE 0.038 ms (4.9% rel)**, in-sample MAE 0.004 ms; (P0.6) 接入 `m4_9_closed_loop_v2.py::predict_lat_ms_lgb` 替换 rule-based, 6-anchor 对比 LGB MAE **0.012 vs rule 0.115 ms (10× better)**. 关键 bug 修: LGB OMP threading deadlock 需要 `num_threads=1` | §1.5 部分解决 (Pyramid 单模型 LGB 已训出, 跨模型仍是 P1 的事) | `data/pyramid_random_bench.parquet`, `models/lgb_pyramid_lat.txt`, `results/lgb_pyramid_metrics.json`, `scripts/phase2/{p0_pyramid_random_bench, train_lgb_pyramid}.py` |

---

## 7. 待办: paper writing 时检查清单

写论文 §6 Limitations 段时, 必须从本文件抽取以下内容:

- [ ] §1.1 baseline ckpt provenance + SECOND warning 完整段落
- [ ] §1.2 跨平台 M 维度仅 FP16 1 anchor (Orin INT8 calib portability is future work)
- [ ] §1.3 cube ~30% 覆盖率 (强调 framework 可扩展, 非 minimal claim)
- [ ] §1.5 rule-based predictor (强调 framework focus on search structure not predictor accuracy)
- [ ] §3 不 claim 列表全部 7 条 (paper 末尾标 explicit non-claim)
- [ ] §4.1 与 QuantV2X 对比一致性 (强调 SECOND 警告对对比无 confounding)

写完后再扫一遍本文件, 确认每条 limitation 都在论文 §6 体现 (或在 §A appendix 有 trace).
