# Pyramid m1 (LiDAR/PointPillar, DAIR-V2X) 全网结构审计 v1

> 编制: sw-optimizer (主笔) · hw-optimizer (INT8/TRT 列协助)
> 时间: 2026-06-05
> 目的: 用户审阅用 — 与 v2xvit_structure_audit_v1.md 格式对齐, 作为模型选型对照
> 格式规范: 与 `multi_agent/model/v2xvit_structure_audit_v1.md` 对齐
> **数据集严格分块**: DAIR 口径 ≠ OPV2V 口径, 各表独立, 严禁混拼

---

## 零、基础信息

| 项目 | 数值 | 来源 |
|---|---|---|
| 模型类 | `HeterPyramidCollab` (HEAL m1) | 代码 |
| Encoder | PointPillar VFE (稀疏, m1) | yaml 配置 |
| Fusion | PyramidFusion ResNeXt 多尺度加权融合 | yaml 配置 |
| **总参数 (DAIR 配置, CPU 真算)** | **5,464,791 (5.465M)** | 【真测·CPU】`create_model` + 本文 (2026-06-05) |
| DAIR 金标准 AP (base, FP16) | AP30=0.8332 / AP50=0.7910 / AP70=0.6309 | 【真测·AP金标准】`stage_a_ap_real.parquet` (DAIR val 1789) |
| ckpt (DAIR 现役) | `Pyramid_DAIR_m1_base_2023_08_14_11_42_29/` | 盘上已提取 |

> ⚠️ **Zoo survey §1.1 勘误**: 旧表写"总参数 14.45M / backbone 6.58M"均为错误值(混入了 V2X-ViT backbone 或 OPV2V timing 数字)。本文 5.465M 是 DAIR yaml CPU 真算, 为正确值。

---

## 一、全模块分解表

### 1.1 顶层模块

| # | 模块 | 网络类型 | 参数量 | 占比 | 计算时间 (ms) | 口径 | 可剪枝? | 可量化 INT8? |
|---|---|---|---|---|---|---|---|---|
| 1 | **encoder_m1** (PointPillar VFE+Scatter) | Linear+BN (稀疏算子) | **768** | 0.01% | **未测** (稀疏 VFE, 需真实 voxel 输入; DAIR 约 2-4ms 估算) | — | ⚠️ 极少参数, 无意义 | ⚠️ spconv 稀疏算子不进 TRT export scope |
| 2 | **backbone_m1** (BaseBEVBackbone) | Conv2d+BN (standard, stride-2) | **226,176** | 4.14% | **0.35** | 【真测·hook】fp32_pytorch CUDA-Event, random weight, GPU0, W=20+M=200 | ✅ DepGraph 可剪 | ✅ 低风险 (标准 Conv) |
| 3 | **aligner_m1** | identity | **0** | 0.00% | — | — | N/A | N/A |
| 4 | **pyramid_backbone** (PyramidFusion, ResNeXt grouped conv) | Grouped Conv2d (g=32) + multi-scale | **3,757,635** | 68.76% | **5.25** | 【真测·hook】同上 | ✅ DepGraph 全网可剪 (已验证) | ✅ 主体 INT8 可行; ⚠️ stage0 非32对齐 kernel cliff (ISS-014) |
| 5 | **shrink_conv** (DownsampleConv) | Conv2d+BN (stride=1) | **1,475,072** | 26.99% | **0.34** | 【真测·hook】同上 | ✅ 随 pyramid_backbone 联动 | ✅ 低风险 |
| 6 | **heads** (cls+reg+dir, 1×1 Conv) | Conv2d 1×1 | **5,140** | 0.09% | **0.07** | 【真测·hook】同上 | ❌ 无意义 (参数极少) | ✅ 低风险 |

**profiling 口径**:

> 【真测·hook】= fp32_pytorch CUDA-Event chain-subtract, DAIR 2-agent dummy 随机权重 (random weight, 非 DAIR ckpt), GPU0 (util=0%/mem=1MiB), warmup=20+measure=200。
> 预授权: HANDOFF §二 2.2 「内部子模块 hook 计时若缺→profiling 微跑【此微跑已由 team-lead 在本条消息预授权, 引用本句即可; 须空闲卡 nvidia-smi 实查 + 启动宣告】」
> 产物: `results/pyramid_m1_submodule_profile.json` (2026-06-05)

**全体参数小计**:
```
总参数: 5,464,791 (5.465M)
  pyramid_backbone (ResNeXt):  3,757,635  68.76%  ← 参数主体
  shrink_conv:                 1,475,072  26.99%
  backbone_m1 (BaseBEV):         226,176   4.14%
  encoder_m1 (VFE):                  768   0.01%
  heads+aligner:               ~5,140     0.09%
```

**全体 body 时间小计**:
```
full body (fp32 random weight profiling):  5.69 ms
  pyramid_backbone:  5.25 ms  92.1%  ← 时间主体
  backbone_m1:       0.35 ms   6.2%
  shrink_conv:       0.34 ms   6.0%
  heads:             0.07 ms   1.2%
  encoder_m1:        未测 (稀疏 VFE)
  NMS:               未测 (CPU 单独)
```

> ⚠️ **与 M4.3 基准核对**: M4.3 单 agent pyramid_backbone FP32 p50=4.28ms / mean=4.49ms。本 profiling 用 2-agent 联合 forward_collab，故 5.25ms 包含 2-agent 协同融合计算(non-trivial warp+weighted_fuse), 高于单 agent 4.28ms 合理。两数字**不可直接比较**, 口径不同。

---

### 1.2 pyramid_backbone 内部 (PyramidFusion ResNeXt)

**架构**: multi-scale ResNeXt, 3 stage pyramid, grouped conv (g=32), deblocks upsample, weighted feature fusion

**通道配置 (base)**:

| stage | num_filters (base) | output channels after 3×conv1+conv2+conv3 | grouped conv groups | DepGraph 对齐要求 |
|---|---|---|---|---|
| stage0 | 64 | 128 (2× num_filters) | g=32 | 需 32 对齐 |
| stage1 | 128 | 256 (2× num_filters) | g=32 | 需 32 对齐 |
| stage2 | 256 | 512 (2× num_filters) | g=32 | 需 32 对齐 |
| deblocks | → 128 each | 128 | — | — |
| shrink_conv input | 384 (=3×128) | 256 | — | — |

> **⚠️ ISS-014 耦合陷阱**: stage0 num_filters[0]=48 非 32 对齐(p25 剪枝档), 2×=96 通道 → TRT INT8 kernel selection cliff: `implicit_gemm`(generic GEMM) 替代 `direct_group`(专用 grouped kernel) → p25 INT8 几乎无加速(1.06×)。这是框架"联合搜索必要性"的硬核实证。根因 **不是 padding**，是 **TRT grouped-conv kernel 选择断崖**(ISS-014, profile+tactic 实证, supervisor 独立复核)。

---

## 二、DAIR 体系延迟数据

### 2.1 单 agent body 延迟 (M4.3 PyTorch, pyramid_backbone only)

> 【真测·PyTorch】M4.3, RTX4090, PyTorch FP32/FP16 CUDA Event, 200 iter, 单 agent [1,64,128,256]

| 精度 | mean_ms | p50_ms | p99_ms | 参数量 | 口径 |
|---|---|---|---|---|---|
| FP32 | 4.494 | 4.282 | 6.517 | 3.758M | 单 agent body (pyramid_backbone only) |
| FP16 (autocast) | 3.274 | 3.108 | 5.003 | 3.758M | 同上 |

来源: `results/m4_3_pyramid_fusion_4090_latency.csv`

### 2.2 DAIR collab2 body 延迟 (TRT, body_subnet_collab2 口径)

> 口径: `body_subnet_collab2` = 双 agent 融合网络体(encoder 外), TRT engine, input shape [2,64,128,256]
> **latency_kind = `body_subnet_collab2`** (≠ e2e; 须叠加 encoder_m1 voxelize+NMS 才是 e2e)

| 配置 | 精度 | lat_p50_ms | lat_p99_ms | engine_size_MB | AP50 (DAIR val 1789) | AP70 | 口径 |
|---|---|---|---|---|---|---|---|
| **base** | FP32 (TRT) | 3.369 | — | 28.45 | 0.791 | 0.631 | lat:【真测·TRT】`m4_8_dair_collab_trt_fp32.json`; AP:【stage_a 复用】 |
| **base** | FP16 (TRT) | 1.269 | 1.275 | 12.48 | 0.791 | 0.631 | lat:【真测·TRT】`m4_8_dair_collab_trt_fp16_clean.json`; AP:【stage_a 复用】 |
| **base** | INT8 (TRT MinMax) | 0.809 | — | 8.59 | 0.790 | 0.623 | lat:【真测·TRT】`m4_8_dair_collab_trt_int8_clean.json`; AP:【stage_a 复用】 |
| base mixed (TRT-auto) | Mixed FP16/INT8 | 0.957 | — | 8.72 | — | — | 【真测·TRT】`m4_8_dair_collab_trt_mixed_clean.json`; AP 未独立测 |
| **p25 FP16** | FP16 (TRT) | 2.91 | — | — | 0.777 | 0.590 | lat:【真测·TRT】dataset_v2 `pyr_48-96-192_FP16`; AP:【stage_a 复用】; ⚠️ 非 32 对齐 kernel cliff |
| **p25 INT8** | INT8 (TRT MinMax) | 2.74–2.91 | — | — | 0.775–0.777 | 0.580–0.590 | lat:【真测·TRT】dataset_v2 T2_p25 forced/auto; AP:【stage_a 复用】 |
| **p50 FP16** | FP16 (TRT) | 1.014 | — | — | 0.764 | 0.564 | 【真测·TRT】dataset_v2 T4_p50 |
| **p50 INT8** | INT8 (TRT MinMax) | 0.7956 | — | — | 0.7522 | 0.5542 | lat:【真测·TRT】dataset_v2 `pyr_32-64-128_INT8`; AP:【stage_a 复用】; ⚠️ ISS-029 |
| p50 finetune (FT) | FP16 (TRT) | 1.009 | — | — | 0.7644 | 0.5641 | lat:【真测·TRT】`m4_8_dair_pruned50_ft_collab_trt_fp16.json`; AP:【真测·AP】`m4_8_hybrid_ap_dair_pruned50_ft_collab_fp16.json` |
| p50 finetune (FT) | INT8 (TRT MinMax) | 0.776 | — | — | — | — | 【真测·TRT】`m4_8_dair_pruned50_ft_collab_trt_int8.json`; AP 未独立测 |
| **p75 FP16** | FP16 (TRT) | 0.7681 | — | — | 0.757 | 0.530 | lat:【真测·TRT】dataset_v2 `pyr_16-32-64_FP16`; AP:【stage_a 复用】 |
| p75 FP16 (FT) | FP16 (TRT) | 0.776 | — | — | — | — | 【真测·TRT】`m4_8_dair_pruned75_ft_collab_trt_fp16_clean.json`; AP 未独立测 |

**核心结论 (DAIR collab2 body)**:
- **INT8 精度近无损**: base FP16→INT8 ap50 0.791→0.790 (Δ=-0.001), lat 1.27→0.81ms (1.57×)
- **p25 kernel cliff**: p25 FP16=2.91ms vs base FP16=1.27ms (比 base 慢!), p25 INT8=2.74ms vs base INT8=0.81ms (3.4×差距); 根因 ISS-014 grouped-conv kernel-selection cliff
- **剪枝收益单调(p50/p75)**: p50→p75 FP16 1.01→0.78ms; AP50 0.764→0.757 (微弱下降)
- **per-stage 混精 Pareto 被 TRT-auto 支配**: forced per-stage 在 stage 边界插 reformat 增延迟; TRT-auto mixed 0.957ms vs per-stage best 1.08ms (detail: `results/perstage_quant_AP_real_v2.csv`)

---

### 2.3 DAIR AP 数据 (gold standard, stage_a_ap_real.parquet)

> **金标准**: DAIR val 1789 全集真测 (已 finetune ckpt), 4dp 精度
> 来源: `multi_agent/data/stage_a_ap_real.parquet` (白名单, ✅ 可直接引用)

| 配置 | 精度 | AP50 (4dp) | AP70 (4dp) | AP30 | 口径 |
|---|---|---|---|---|---|
| **base** | FP16 | **0.7910** | **0.6309** | **0.8332** | 【金标准】DAIR val 1789, TRT collab 混合管线 (n_trt_path=1618) |
| **base** | INT8 | **0.7905** | **0.6228** | — | 【金标准】同上; INT8 Δap50=-0.0005 (近无损) |
| **p25** (num_filters=[48,96,192]) | FP16 | 0.7769 | 0.5905 | — | 【金标准】DAIR val 1789, 已 finetune |
| **p50** (num_filters=[32,64,128]) | FP16 | 0.7644 | 0.5641 | — | 【金标准】同上; ⚠️ ISS-029: 与 corrected_full 两源差+0.009, 本表用 stage_a 源 |
| **p50** | INT8 | 0.7522 | 0.5542 | — | 【金标准】同上; Δap50=-0.012; ⚠️ ISS-029: 与 corrected_full 两源差+0.009, 本表用 stage_a 源 |
| **p75** (num_filters=[16,32,64]) | FP16 | 0.7567 | 0.5300 | — | 【金标准】同上 |
| prune85 (num_filters=[10,20,40], wpg=16) | FP16 | 0.74 | 0.59 | 0.79 | 【真测·2dp】`results/ap_cliff_converged.json` |
| prune90 (num_filters=[6,13,26], wpg=16) | FP16 | 0.75 | 0.58 | 0.79 | 【真测·2dp】同上 |
| prune95 (num_filters=[4,6,13], wpg=16) | FP16 | 0.74 | 0.57 | 0.78 | 【真测·2dp】同上 |
| cliff2_a ([32,64,128], wpg=4, pb=-90%) | FP16 | 0.75 | 0.60 | 0.79 | 【真测·2dp】`results/ap_cliff2_converged.json` |
| cliff2_b ([24,48,96], wpg=4, pb=-82%) | FP16 | 0.74 | 0.59 | 0.78 | 【真测·2dp】同上 |

**关键结论 (AP 轴)**:
- **INT8 近无损**: base FP16→INT8 Δap50=-0.0005 (≪噪声水平 ±0.001)
- **剪枝在可达区无悬崖**: pb 从 base 到 -90% (cliff2_a), AP50 全稳 0.74-0.75; 根因 = **模型对 DAIR 严重过参数化**
- **AP 轴信号弱**: base AP50 仅 0.791, 全范围 span 仅 ~0.04; AP70 信号稍强 (0.63→0.53, span 0.10); 论文主用 AP70
- **INT8 精度轴无轴信号**: 4档中 3档噪声级+非单调 (ISS-020/024 终核)

---

## 三、OPV2V 体系延迟数据 (⚠️ OPV2V 口径, 与 DAIR 严禁混表)

> ⚠️ 以下数据均为 **OPV2V test split (2170 samples)** 口径, 不可与 DAIR 数字拼表比较。

### 3.1 OPV2V 完整 e2e (M4.6.0)

| 精度 | AP30 | AP50 | AP70 | lat_mean_ms | lat_p50_ms | lat_p99_ms | 口径 |
|---|---|---|---|---|---|---|---|
| FP32 | 0.9691 | 0.9635 | 0.9272 | 43.00 | 35.31 | 119.97 | 【真测·PyTorch】e2e (含 NMS CPU overhead) |
| FP16 (autocast) | 0.9690 | 0.9631 | 0.9268 | 30.96 | 26.70 | 80.23 | 【真测·PyTorch】同上 |

来源: `results/m4_6_0_pyramid_fp32_eval.txt` + `m4_6_0_pyramid_fp16_summary.md`

**NMS 占比 (OPV2V 实测)**: NMS ~72.4% of e2e → CUDA 化 3.04× e2e 加速 (P0, OPV2V 口径; DAIR 口径待测)

### 3.2 OPV2V 全网剪枝 collab subnet 延迟 (P1 实测)

> 口径: PyTorch CUDA Event, collab subnet body, RTX4090 GPU5/7 idle
> **lat 口径: OPV2V schema** (P1_wholenet_prune_real.csv; record_len≈OPV2V multi-agent)
> **AP 口径: DAIR val 1789** (stage_a / `ap_source=inference.py_DAIR_val_1789`) — ⚠️ AP 与 lat 数据集来源不同, 括号内 AP 来自 DAIR 金标准, 非 OPV2V 评测; "schema 待考" 标注
> 来源: `results/P1_wholenet_prune_real.csv`

| 配置 | prune_scope | num_filters | subnet_params | lat_fp16_p50_ms | lat_fp32_p50_ms | AP50 [DAIR, 非OPV2V] |
|---|---|---|---|---|---|---|
| dense_baseline | none | [64,128,256] | 5.238M | 4.01 | 5.40 | (0.791 from stage_a) |
| backbone_only_p50 | backbone_only | [32,64,128] | 2.581M | 5.13 | 4.40 | (0.764 from stage_a) |
| wholenet_light | bb+deblocks+shrink | [32,64,128], deblocks=112×3 | 2.198M | 3.86 | 4.11 | (0.75 2dp) |
| wholenet_p50 | bb+deblocks+shrink | [32,64,128], deblocks=96×3 | 1.860M | 3.75 | 3.82 | (0.75 2dp) |
| wholenet_aggr | bb+deblocks+shrink | [32,64,128], deblocks=64×3 | 1.323M | 3.30 | 3.09 | (0.75 2dp) |

> ⚠️ **backbone_only_p50 lat anomaly**: 5.13ms FP16 > dense_baseline 4.01ms → 对齐问题(p50 num_filters[0]=32, deblocks 维持128 → shrink 输入维度 384→128 但维度比例变化). 这不是 DAIR collab2 的非对齐陷阱, 是 OPV2V collab subnet 不同 scope 的维度效应。

---

## 四、剪枝分析

### 4.1 剪枝工具与配置

| 工具 | 目标 | 方法 | 已验证 |
|---|---|---|---|
| `tools/configurable/depgraph_pyramid.py` | pyramid_backbone (全网) | DepGraph L1 norm + round_to=32 | ✅ 已真测收敛 AP |
| DepGraph ckpt 格式 | flat state_dict | 剪枝 ckpt 须为 flat 格式, `{"model_state_dict":...}` 包裹格式会 full key missing | ✅ 已确认 |

**剪枝支持范围** (已验证):
- ✅ `pyramid_backbone` (68.76% 参数): 全网可剪, 含 deblocks/shrink_conv
- ✅ `backbone_m1` (4.14%): DepGraph 可剪但参数少, ROI 极低
- ⚠️ `encoder_m1` (0.01%): 参数极少, 无意义; spconv 稀疏算子 DepGraph 不支持

### 4.2 剪枝精度结论 (DAIR 实测)

| 剪枝范围 | 剪枝率 | 结论 | 依据 |
|---|---|---|---|
| pyramid_backbone | 0–90% | **无精度悬崖** (AP50 0.791→0.74, span ~0.05) | `ap_cliff{,2}_converged.json`; CLAUDE §〇.7 |
| pyramid_backbone | p25(num_filters[0]=48, 非32对齐) | AP 有轻微 Δ; **主要问题在延迟(ISS-014 kernel cliff)** | stage_a_ap_real.parquet |
| 全网 (backbone+deblocks+shrink) | 74.7% | AP50≈0.75 (2dp), lat FP16 从 4.01→3.30ms (-17.7%) | P1_wholenet_prune_real.csv |

**根因诊断 (精度无悬崖)**: DAIR 场景 base AP50 仅 0.791 → **模型对任务严重过参数化**. 剪掉 pyramid_backbone 90% 参数 AP50 仍 0.74-0.75 (基础任务容量远超所需). 这是"剪 pyramid_backbone Pareto 退化(剪到底即最优)"的根因。真正 AP trade-off 在量化边界/更难任务。

### 4.3 HEAL ResNeXt 宽度公式陷阱

```
width = int(p * wpg / 64) * g    # p=plane, wpg=width_per_group, g=groups
```

| wpg | g | p 下限 | 风险 |
|---|---|---|---|
| 4 | 8 | p<16 → width=0 → 崩溃 | ⚠️ g8+wpg=4: p<16 全崩 |
| 16 | 8 | p<8 → width=0 | ✅ wpg=16 安全 |

→ 需要缩 plane 时**必须把 wpg 加到 16**。已入 MEMORY `project_heal_resnext_width_formula.md`。

---

## 五、量化分析

### 5.1 各模块 INT8 风险

| 模块 | 网络类型 | INT8 风险档 | 依据 |
|---|---|---|---|
| encoder_m1 (VFE) | Linear+BN (稀疏) | **N/A** (spconv 不进 TRT scope) | 代码分析 |
| backbone_m1 | Conv2d+BN | **🟢 低** | 标准 conv; 类比 Pyramid base INT8 Δap50=-0.001 |
| pyramid_backbone stage0 (非32对齐) | Grouped Conv g=32 | **🔴 高 (kernel cliff)** | ISS-014 实证: p25 INT8 几乎无加速(1.06× vs 1.57×) |
| pyramid_backbone 对齐档 | Grouped Conv g=32 | **🟢 低-中** | base/p50/p75 INT8 正常 (1.57×/1.07×/—) |
| shrink_conv | Conv2d+BN | **🟢 低** | 同 backbone |
| heads | Conv2d 1×1 | **🟢 低** | 标准 1×1 |

### 5.2 校准器选择

| 校准器 | AP 结果 | 结论 |
|---|---|---|
| **MinMax** | ap50 0.791→0.790 (Δ=-0.001) | ✅ **主路径, 推荐** |
| Entropy | AP 崩塌 (ap50<0.5) | ❌ 已证伪, 禁用 |
| Percentile | 未系统测试 | ⚠️ 未测 |

来源: `multi_agent/methods/design/dims_quantization_v1.md §二.3`; commit `3de2167` (Entropy AP 崩塌诊断 + MinMax 主路径决议)

### 5.3 per-stage 混精量化

| 场景 | AP 结果 | Pareto 地位 |
|---|---|---|
| 剪枝档 T_prune50p 保 stage1 FP16 | AP50 比 forced-INT8 高 +0.022~0.025 | ✅ AP 轴有正向点 |
| baseline 上保 stage1 FP16 | AP50 近无损 → 无价值 (baseline INT8 本就无损) | — |
| **TRT-auto mixed** | AP≥ 任何手工 per-stage; lat 更快 | **Pareto 支配**: 手工 per-stage 无一在前沿 |

**结论**: 框架应用 **TRT-auto** (自动混精), 不枚举手工 per-stage。

---

## 六、硬件部署 (TRT/Orin)

### 6.1 TRT 部署配置

| 项目 | 内容 | 来源 |
|---|---|---|
| ONNX export scope | pyramid_backbone (含 backbone_m1+shrink_conv+heads), input [2,64,128,256] | 实测已 export |
| encoder_m1 (spconv) | **不在 TRT scope** — 稀疏算子不可 ONNX | 代码分析 |
| TRT FP16 engine | `models/pyramid_dair_m1_collab_n2_fp16.engine` (12.5MB) | `m4_8_dair_collab_trt_fp16_clean.json` |
| TRT INT8 engine | `models/pyramid_dair_m1_collab_n2_int8.engine` (8.6MB) | `m4_8_dair_collab_trt_int8_clean.json` |
| INT8 calibrator | MinMax (主路径) | dims_quantization_v1.md |
| Orin 跨平台映射 | R²>0.995, `results/m2_latency_mapping_f.json` | 【真测·拟合】 |

### 6.2 Orin DLA 状态

| 配置 | Build 结果 | 延迟 |
|---|---|---|
| DLA FP16 | ⚠️ 8/12 成功 (bank 限制, kDIRECT_IO) | 比 GPU FP16 快 ~22% (部分) |
| **DLA INT8** | **❌ 0/12 build 成功** (bank 超限) | N/A |

**结论**: Pyramid 在 Orin 上 **DLA 路由只能 FP16**, INT8 完全不可行 (非 ResNet50 的结论, 不可外推)。

---

## 七、瓶颈结论

### 7.1 时间瓶颈

```
DAIR body 分解 (fp32_pytorch random weight profiling):
├── pyramid_backbone (PyramidFusion ResNeXt)   5.25 ms  92.1%  ← 绝对主体
│   ├── stage0 grouped conv (g=32, 3×3)        (层级未单独测)
│   ├── deblocks upsample                      (层级未单独测)
│   └── weighted_fuse multi-scale              (层级未单独测)
├── backbone_m1 (BaseBEVBackbone stride-2)     0.35 ms   6.2%
├── shrink_conv (DownsampleConv)               0.34 ms   6.0%
├── heads (cls+reg+dir 1×1 conv)              0.07 ms   1.2%
├── encoder_m1 (VFE sparse)                   未测 (约 2-4 ms 估算)
└── NMS (CPU)                                  未测 (OPV2V 占 e2e ~72%)

DAIR collab2 TRT body: base FP16=1.27ms / INT8=0.81ms
OPV2V e2e: FP32=35.31ms / FP16=26.70ms (含 NMS, ⚠️ OPV2V 口径)
```

### 7.2 参数瓶颈

| 排序 | 模块 | 参数量 | 剪枝 ROI |
|---|---|---|---|
| 1 | pyramid_backbone | 3.758M (68.8%) | **中** (剪到底几乎免费, 但 Pareto 退化) |
| 2 | shrink_conv | 1.475M (27.0%) | **中** (联动剪可省, 但 P1 全网剪已含) |
| 3 | backbone_m1 | 226K (4.1%) | **低** (少, 剪收益小) |
| 4 | encoder_m1 | 768 (0.01%) | **无意义** |

### 7.3 精度敏感点

| 维度 | 发现 | 来源 |
|---|---|---|
| backbone 剪枝 (0-90%) | **免费**: AP50 span ~0.05 (0.791→0.74), 近无损 | stage_a_ap_real + cliff 实测 |
| INT8 (base) | **近免费**: Δap50=-0.001 | stage_a_ap_real (金标准) |
| INT8 (p50) | Δap50=-0.012 (轻微放大) | stage_a_ap_real |
| p25 非对齐 INT8 | **kernel cliff**: lat 几乎无加速(1.06×) | ISS-014 profile+tactic 实证 |
| per-stage 混精 | AP 有正向点(剪枝档+0.022); 但 Pareto 被 TRT-auto 支配 | perstage_quant_AP_real_v2.csv |
| Entropy 校准 | **AP 崩塌** | `dims_quantization_v1.md §二.3`; commit `3de2167` |

**交叉结论**:
- "动哪里最值(时间)": pyramid_backbone 占 body 92%, 首选压缩目标
- "动哪里最安全(精度)": INT8 (base) 几乎无损; 剪枝(对齐档) 无悬崖 → **联合剪+INT8 可叠加收益**
- **耦合陷阱**: 剪到非32对齐档 + INT8 = kernel cliff (ISS-014) → 框架必须联合搜索规避

---

## 八、已知实验事实附录

### 8.1 Phase M 剪枝 AP 数据 (4dp gold standard, dataset_v2.csv)

参见 §二 2.3 完整表格。关键引用点:
- base INT8: ap50=0.7905, ap70=0.6228 (MinMax, `pyr_64-128-256_INT8`, is_real_measured=True)
- p50 FP16: ap50=0.7644, ap70=0.5641
- p75 FP16: ap50=0.7567, ap70=0.5300

### 8.2 QuantV2X 外部文献对照

来源: arXiv:2509.03704 Table 1, DAIR-V2X, PTQ (已原文核验)

| 配置 | AP30 | AP50 |
|---|---|---|
| Pyramid FP32 (外部) | 75.1 | 68.2 |
| Pyramid INT8 (外部, PTQ) | 74.6 | 67.8 |
| Pyramid INT4W (外部) | 74.2 | 66.7 |

→ 外部与我方 INT8 结论一致 (AP 近无损); 互证可信。
> ⚠️ 旧引 "75.1→29.9" 已证伪为跨模型拼接幻觉 (ISS-031); 正确值: **Pyramid INT8 74.6 (Δ-0.5)**, V2X-ViT INT8 40.0 (Δ-17.4)。

### 8.3 DLA Orin 实测

来源: `data/orin_dspace_bench.parquet` + ISS-007

- DLA INT8: 0/12 build 成功 (kDIRECT_IO + bank 超限)
- DLA FP16: 8/12 build 成功, 比 GPU FP16 快 ~22%
- GPU0 (Orin sm87): TorchVision ResNet 代理模型跨平台拟合 R²>0.995, `results/m2_latency_mapping_f.json` (非 Pyramid 直接拟合; 代理模型用于 Orin latency 估算)

---

*文档版本: v1.0 (2026-06-05)*
*产物: `results/pyramid_m1_submodule_profile.json` (sub-module hook timing)*
*核验标准: 污染值(2849/7.7×/0.5078)未引用; AP 用 4dp 金标准源; latency 带 latency_kind; DAIR/OPV2V 严禁混表*
