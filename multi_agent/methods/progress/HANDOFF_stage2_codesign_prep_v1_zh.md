# Stage2 软硬件协同优化框架启动前信息整理 v1

日期: 2026-06-24

本文用于进入 Stage2 前快速恢复上下文。它整理三件事:

1. Stage1 与 Stage2 当前通过什么渠道交互。
2. Stage2 软硬件协同优化框架的整体流程是什么。
3. 目前 Pyramid 与 CoDriving 两个模型上已经完成的重要实验结果。

本文只整理现状与边界，不改动已有代码和结果文件。

---

## 1. 当前结论摘要

Stage1 与 Stage2 已经有一条直接数据通道: **Stage1 partition manifest -> `framework/stage1_bridge.py` -> `SpaceSpec` -> `framework/search_three_arm.py` / PQS 搜索器**。这条通道把网络扫描得到的剪枝旋钮、量化单元、硬件对齐约束、routing 段和 trace 边界变成 Stage2 可搜索空间。

同时还有一条校准/证据通道: **Stage2/S2/S4 等测量脚本 -> `results/stage1_model_predict/` 证据 JSON -> `calibrated_predictor` / `model_classifier`**。这条通道把 H800 TVM 测得的 schedule coupling、coverage gate、三臂验证等证据回灌到 Stage1 分类器，用来输出“需要协同加速 / 可分离加速 / 扫描失败”、blocker 和下一步 gate。

需要注意: **Stage2 搜索器当前直接消费的是 manifest，经 bridge 派生搜索空间；`model_classifier` 的三分类 JSON 目前更多是 gate/report 层，还没有完全作为 Stage2 search runtime 的硬输入。** 下一阶段应把分类器输出正式接入 Stage2 调度策略，例如对 `SCAN_FAILED` 禁止搜索、对 `SEPARABLE_ACCELERATION` 默认串行/低预算、对 `CO_ACCELERATION_REQUIRED` 开启 joint 或 per-knob joint。

---

## 2. Stage1 给 Stage2 提供什么

### 2.1 主要输入: partition manifest

典型路径:

- `framework/partitions/codriving_partition.yaml`
- `framework/partitions/pyramid_lidar_partition.yaml`
- `framework/partitions/pyramid_camera_partition.yaml`
- `results/autoscan_fcooper_partition.yaml`
- `results/autoscan_attfuse_partition.yaml`
- `results/autoscan_where2comm_partition.yaml`
- `results/autoscan_v2vnet_partition.yaml`
- `results/autoscan_disconet_partition.yaml`

Stage2 最直接使用的是 manifest 中这些结构化字段:

- `scan_status` / `ckpt_status`: 判断扫描是否可用、是否是真 checkpoint。
- `hw_capability`: 硬件能力，例如 `int8_align`、`fp16_align`、`int8_pack_factor`、legal bits、DLA whitelist 等。
- `trace_plan`: dense candidate、skipped / ignored / rejected 子图、是否 manual override、是否需要 review。
- `trace.skipped_subgraphs`: sparse、fusion、attention、routing、custom/postprocess 等未进入 dense core 的子图。
- `view_b1_prune_groups`: 细粒度剪枝组，包含 `cin/cout/groups/ic_bn/kernel/stride/op_types/fanout_buckets` 等特征。
- `view_b1_search_groups`: Stage2 的剪枝搜索旋钮，包含 `widths`、`round_to`、`int8_buildable_align`、`bucket`、`max_rate`、聚合 feature。
- `view_b2_quant_units`: 量化单元和 legal precision / granularity。
- `view_d_routing_segments`: 部署/routing 段。
- `view_latency.coverage`: trace-net coverage 口径。它不是 full-model latency coverage。

### 2.2 Bridge 的实际代码通道

核心代码:

- `framework/stage1_bridge.py`
- `framework/search_three_arm.py::build_from_manifest`
- `framework/run_pqs_ablation.py`
- `framework/run_pqs_codriving.py`

桥接流程:

```text
Stage1 manifest
    ↓
SpaceSpec.from_manifest()
    ↓
KnobSpec / QuantUnit / RoutingSegment
    ↓
legal_widths()
buildable_int8_widths()
has_int8_buildability_cliff()
dispatch_plan()
    ↓
search_three_arm.build_from_manifest()
    ↓
LatencyLUT + APModel + QLookup + joint_knobs
    ↓
Stage2 三臂 / PQS 搜索
```

关键实现点:

- `KnobSpec.legal_widths()` 根据 `round_to` 和 `max_rate` 给出结构上可剪的宽度。
- `KnobSpec.buildable_int8(w)` 根据 manifest 里的 `int8_buildable_align` 判断该宽度能否进入 H800 TVM NCHWc/dp4a/WMMA INT8 快路径。
- `SpaceSpec.has_int8_buildability_cliff()` 判断“可剪但不可建 INT8”的结构性前提。它不是模型级可分离证明。
- `SpaceSpec.dispatch_plan()` 生成逐 knob 的 `coupling_score` 和 `joint/serial` 分流建议。
- `build_from_manifest()` 选择有 cliff 的 gating knob，派生 `key_scale`、`bridge_int8_align`，并返回 Stage2 搜索器需要的 LUT/AP/Q lookup。

当前 bridge 的原则是: **不写 `if model == ...` 的模型硬编码，尽量从 manifest 字段派生。** 例如 CoDriving 的 `groups=1` 会自然落成 `int8_buildable_align=pack_factor=4`，所有合法宽度都可 build；Pyramid 分组卷积会落成更严格的对齐约束。

### 2.3 Stage1 分类器提供的 gate 信息

主要产物:

- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`

当前 9 模型三分类:

| model | acceleration_class | 当前含义 |
|---|---|---|
| CoDriving | 可分离加速 | 只在 CoDriving 已测 dense ResNet backbone envelope 内可分离，不跨模型外推 |
| F-Cooper | 需要协同加速 | H800 TVM latency pair calibration / fusion gate 未闭合 |
| AttFuse | 需要协同加速 | attention/fusion coverage 未闭合 |
| V2X-ViT | 需要协同加速 | fake-quant + routing gate，true INT8/AP 仍 blocked |
| Pyramid lidar | 需要协同加速 | P-hub/grouped-conv context，TRT 只作 historical evidence |
| Pyramid camera | 需要协同加速 | 同上 |
| Where2comm | 扫描失败 | 缺 checkpoint，只是 architecture-only/random-init sidecar |
| V2VNet | 扫描失败 | 同上 |
| DiscoNet | 扫描失败 | 同上 |

Stage2 下一阶段应该把这张表变成正式 gate:

- `SCAN_FAILED`: 不进入 Stage2 自动搜索，只输出“需提供 config+ckpt / 修 trace boundary”。
- `SEPARABLE_ACCELERATION`: 默认串行或低预算搜索，用三臂度量仪抽检即可。
- `CO_ACCELERATION_REQUIRED`: 开启 per-knob joint budget 或完整三臂验证。

---

## 3. Stage2 如何利用 Stage1 信息

Stage2 使用 Stage1 信息主要做四件事。

### 3.1 构建合法搜索空间

Stage1 已经把原始网络图扫描成若干搜索旋钮。Stage2 不再手写 `num_filters` 组合，而是从 manifest 读取:

- 当前宽度 `cur_widths`
- 剪枝粒度 `round_to`
- 最大剪枝率 `max_rate`
- 是否 grouped conv
- INT8 可建对齐 `int8_buildable_align`
- 量化 legal bits / granularity
- routing/device 段

这一步把“理论 P×Q×S 巨空间”收缩成每个模型自己的合法空间。

### 3.2 判断哪些 knob 需要 joint 搜索

Stage2 不是对所有模型、所有 knob 都一律付 full joint 成本。bridge 会给每个 knob 计算:

- `cliff_strength`: `int8_buildable_align / round_to` 的对齐悬崖强度。
- `schedule_headroom`: TVM default 到 MetaSchedule tuned 的潜在收益；没有实测时用结构先验，grouped/非标卷积更高，标准卷积更低。
- `coupling_score`: 综合 P×Q 与 P×S 信号。
- `dispatch`: `joint` 或 `serial`。

这对应 auto-tuning 设计文档中“耦合强度是被测量/估计出来的，不是由 `groups=1` 这样的静态规则直接宣布”的原则。

### 3.3 驱动三臂验证和 PQS 搜索

Stage2 目前的主要实验内核:

- `framework/search_three_arm.py`
- `framework/run_b4_ablation.py`
- `framework/run_pqs_ablation.py`
- `framework/run_pqs_codriving.py`

三臂定义:

| 臂 | 含义 |
|---|---|
| `A-joint` | P 与 S 同搜；PQS 中是 P×Q×S 同搜 |
| `A-serial` | 先在 default / 局部口径下锁 P，再搜 S 或 Q×S |
| `A-noS` | 消掉 schedule 轴，只用 default schedule |

Stage1 manifest 给出搜索空间和可建性，Stage2 用 LUT/AP/Q lookup 对候选做 Pareto/HV 评估。

### 3.4 回灌 Stage1 证据

S2/S4 等证据脚本虽然在 `scripts/phase2/` 下，但它们的结果写入 `results/stage1_model_predict/`，供 Stage1 calibrated predictor / model classifier 使用。

典型证据:

- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.json`
- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.json`
- `results/stage1_model_predict/s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json`
- `results/stage1_model_predict/s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json`
- `results/stage1_model_predict/s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json`

对应脚本:

- `scripts/phase2/stage1_s2_anchor_runner.py`: 在 H800/TVM 上跑低成本 schedule anchors，使用 synthetic representative shapes，不导出 full model。
- `scripts/phase2/stage1_s4_three_arm_validation.py`: 不新增测量，读取已有 S2 24-cell matrix，比较 `local_only`、`pair_search`、`joint_search`。
- `scripts/phase2/stage1_s2_5_s3_evidence_report.py`: 生成 S2.5 coverage gates 和 S3 quant sensitivity 报告。

因此当前系统是双向的:

```text
Stage1 scan manifest  ──→  Stage2 search space / legal constraints
Stage2/S2/S4 evidence ──→  Stage1 calibrated predictor / model classifier
```

---

## 4. Stage2 软硬件协同优化框架整体流程

根据 `multi_agent/methods/design/auto-tuning/`，当前 Stage2 正确主线是:

```text
输入: model config + checkpoint + Stage1 manifest + hardware capability
    ↓
空间构建:
  从 manifest 读取 B1/B2/D 视图、对齐约束、trace boundary
    ↓
外环软件空间:
  P = 剪枝宽度 / channel width
  Q = 量化精度/格式，当前作为补充或 future，不是唯一主轴
    ↓
内环硬件空间:
  S = TVM Relax / MetaSchedule schedule
  tile / loop order / fusion / tensorize / layout
    ↓
块级 LUT:
  对每个 (block, width, precision) 做 MetaSchedule 或读取已有测量
    ↓
搜索:
  A-joint / A-serial / A-noS 或 NSGA-II 多目标搜索
    ↓
评估:
  latency / AP / energy / throughput / model size / DS/RC
    ↓
验证:
  B5 收敛解真测审计、三臂显著性、coverage/gate 回灌
    ↓
输出:
  Pareto 前沿、推荐配置、blocker、下一步真测 gate
```

### 4.1 空间构建

设计稿 `1_design_space_building_v1.md` 的核心是:

- 外环 P/Q 变了，内环 schedule 空间也要重建，这借鉴 ALT 的 cross-exploration。
- 不枚举全量 P×Q×S，而是按块建 schedule-LUT，借鉴 CHaNAS 的 `R·B·S + R^B` 分解。
- 合法性约束前移，例如 `in_per_g`、`int8_buildable_align`、precision gate、routing/DLA gate。
- `groups=1` 或 `no cliff` 不是模型级可分离证明；真正可分离要通过三臂度量仪测出来。

### 4.2 Cost model

设计稿 `2_design_cost_model_v1.md` 的核心是分层:

- 内层 schedule cost model 不重造，使用 TVM MetaSchedule 的 XGBoost/rank-style cost model。
- 外层 prune/quant 多目标评估器使用 LUT、AP 真测/残差预测、energy/throughput/size 派生或真测。
- latency 排序优先 rank/pairwise loss，避免跨数量级回归崩。
- AP 不能纯预测成真值，必须用 finetune/评测锚点，预测器只能用于筛选或排序。

### 4.3 探索算法

设计稿 `3_design_exploring_tvm_integration_v1.md` 的核心是:

- 外环用 NSGA-II 或三臂搜索，不用 PPO。
- 内环用 TVM Relax/MetaSchedule。
- `P` 改网络结构，通常要重新导出/重建 IRModule。
- `S` 改 TIR/schedule，由 TVM 负责。
- `Q` 当前不作为主依赖。历史文档里有 TRT/INT8 路线，但 Stage1 新证据口径必须与 H800 TVM 区分清楚。

### 4.4 三臂证明

设计稿 `4_design_ablation_proof_v1.md` 的核心是:

- 证明的不是“某几个枚举点差异很大”，而是“搜索流程中串行贪心会结构性漏掉 joint 可达的分支”。
- 要报告 HV 分布、Wilcoxon、收敛曲线、访问点云、B5 真测审计。
- Pyramid 是强耦合载体；CoDriving 是标准卷积可分离对照。

---

## 5. Pyramid 已有重要结果

### 5.1 P×S 三臂验证: joint 明显优于 serial

主要文件:

- `results/b4_ablation_results.json`
- `results/b5_convergence_verification.json`
- `framework/search_three_arm.py`
- `multi_agent/methods/design/auto-tuning/4_design_ablation_proof_v1.md`

当前 B4 artifact:

- 数据状态: `real`
- LUT: `b1_direct`
- AP: `b2`
- seeds: 12
- budget: 60
- reference HV: `6991.31033945`

HV 结果:

| arm | mean HV | ref 占比 | std |
|---|---:|---:|---:|
| `A-joint` | 6984.217 | 99.90% | 23.52 |
| `A-serial` | 5978.342 | 85.51% | ~0 |
| `A-noS` | 3336.904 | 47.73% | 0 |

显著性:

- `A-joint` vs `A-serial`: Wilcoxon `p=0.00048828125`
- rank-biserial effect size: `1.0`
- 12 个 seed 全部偏向 joint。

### 5.2 三组 W_g/P_g rank-flip 对

| 对 | W_g | P_g | AP70 | W_g tuned us | P_g tuned us | iso-AP latency ratio |
|---|---|---|---:|---:|---:|---:|
| 3 | `[48,128,128]` | `[64,128,128]` | 0.6369 | 21071.43 | 5506.57 | **3.827x** |
| 1 | `[48,96,192]` | `[64,96,192]` | 0.5905 | 21614.80 | 6152.04 | 3.513x |
| 2 | `[48,64,256]` | `[64,64,256]` | 0.6362 | 19404.95 | 5894.56 | 3.292x |

B5 审计结论:

- 所有收敛配置都有真测点支撑。
- 只有 pair3 是真正 shipped co-design win: joint 出货 `[64,128,128]` tuned，serial 出货 `[48,128,128]` tuned，同 AP70=0.6369 下 joint 快 3.827x。
- pair1/pair2 是机制示范，不是最终出货倍率，因为它们的 P_g 被全局更优点支配。

### 5.3 结构性排除证据

在 3 对 × 12 seed × 8 起点的多起点统计中:

- joint 到达 `P_g+tuned`: True。
- serial 到达 `P_g+tuned`: False。
- serial 在所有起点下都丢弃 P_g: True。
- `all_pairs_pass=True`。

这支持“串行贪心结构性漏掉 joint 分支”，而不是单次搜索运气差。

### 5.4 P×Q×S 结果: 已有结构性证据，但当前 artifact 需要复核 seed 数

主要文件:

- `results/pqs_ablation_results.json`
- `results/q_int8_ms_stage0_result.json`
- `results/q_int8_dp4a_pairs.csv`
- `results/q_int8_ap.json`
- `results/coupling_map_matrix.json`

当前 `results/pqs_ablation_results.json` 记录:

- `n_seeds=4`
- `budget=20`
- INT8 latency model: H800 TVM FP16 / 1.449 uniform proxy
- INT8 AP delta: `-0.008`
- 结构约束: `s0=48` 的 grouped conv `in_per_g=3`，不可进入 NCHWc IC_BN=4 快路径；`s0=64` 的 `in_per_g=4` 可 build。

HV 结果:

| arm | mean HV | joint 占比 |
|---|---:|---:|
| `A-joint-PQS` | 7512.733 | 100.0% |
| `A-serial-PQS` | 6277.770 | 83.56% |
| `A-noS-PQS` | 4268.032 | 56.81% |

当前 JSON 的 Wilcoxon:

- `n=4`
- `p=0.125`
- effect size = `1.0`

注意: 6 月 22 日交接文字中曾写过 P×Q×S 12 seed、A-serial 约 85.3%、p=4.88e-4；但当前落盘 JSON 是 4 seed。下一阶段写论文或作为正式表格前，需要以脚本重跑或找回 12 seed artifact，统一口径。

结构性结果仍然有用:

- `categorical_pass=True`
- serial 锁定前沿: `[16,32,64]`, `[32,64,128]`, `[48,64,256]`, `[48,128,128]`
- serial 锁定宽度全部 INT8 unbuildable。
- joint 能访问 4 个 aligned `s0=64` INT8 width。
- 解释: 同一个 `s0 / IC_BN` 对齐属性同时 gate 剪枝宽度、schedule 调优和 INT8 快路径。

### 5.5 H800 TVM INT8 stage0 微基准

`results/q_int8_ms_stage0_result.json`:

- scope: stage0 grouped conv only，不是 full-backbone。
- `lat_int8_us=150.1`
- `lat_fp16_ref_us=217.5`
- FP16 over INT8 speedup: `1.449x`
- path: WMMA
- numerical correctness: max_rel_error = 0.0

这个结果只能用于:

- 证明 `s0=64` aligned path 能进入 H800 TVM int8 快路径。
- 给 PQS ablation 一个 H800-pure uniform proxy。

不能用于:

- 全 backbone INT8 延迟。
- 区分 pair1/pair2/pair3 的完整延迟。

### 5.6 Pyramid AP 曲线: soft knee，不是 hard cliff

`results/ap_cliff_l4.json`:

| total prune | params | AP70 |
|---:|---:|---:|
| 0.0% | 5.46M | 0.6309 |
| 75.3% | 1.35M | 0.5900 |
| 83.9% | 0.879M | 0.5850 |
| 89.2% | 0.593M | 0.5613 |
| 93.3% | 0.368M | 0.5369 |

结论:

- `cliff_type=soft_knee`
- `cliff_found=False`
- AP70 total range = 0.094，约 94x pipeline noise。
- 84% 前后是拐点: 中段较平，高剪枝段 AP 下降加速。

这说明 AP 轴不是完全无信息；但在中等剪枝区，co-design 收益主要体现在 latency，而不是 AP 同时大幅变化。

### 5.7 Coupling map 的最新科学解释

`results/coupling_map_matrix.json` 给出的最终综合:

- Pyramid 的耦合是真实的，但主要是 **P(IC_BN)-hub**。
- 干净结论不是“三轴彼此独立不可约”，而是“P/IC_BN 改变了 Q/S 的可行域与收益”。
- mech2 即 `wpg × S` rank-flip / gain scaling 是真实核心。
- mech1 即 NCHWc buildability 是 format gate，latency 上未必总收益。
- mech3 即“不经 P 的 Q×S 独立耦合”被判定为 backend artifact。

论文表述应改成:

> Pyramid grouped conv shows a P-hub hardware-software coupling: pruning changes IC_BN, which changes schedule/tensorization availability and tuning gain. Joint search is required because serial search can lock a width that later blocks the fast hardware path.

不应写成:

> P/Q/S are independently irreducibly coupled in all directions.

---

## 6. CoDriving 已有重要结果

### 6.1 CoDriving 是标准卷积可分离对照

主要文件:

- `results/coupling_map/C0c_codriving_pqs.json`
- `framework/run_pqs_codriving.py`
- `results/codriving_int8_verify.json`
- `results/coupling_map_matrix.json`

PQS 三臂结果:

| arm | mean HV | joint 占比 |
|---|---:|---:|
| `A-joint-PQS` | 1164.826 | 100.0% |
| `A-serial-PQS` | 1164.826 | 100.0% |
| `A-noS-PQS` | 1130.655 | 97.07% |

其它:

- rank-flip pairs = 0。
- all widths INT8 buildable。
- verdict = `SERIAL` / separable。
- Wilcoxon 无法计算显著差异，因为 paired difference 全为 0。

这说明 CoDriving 在当前低维 P×Q×S 空间中是可分离的。`coupling_map_matrix.json` 进一步记录 C1cod、C6 的高维 trap 也没有稳定成立。

### 6.2 CoDriving INT8 验证

`results/codriving_int8_verify.json`:

| config | FP16 us | INT8 us | speedup | buildable | rank flip |
|---|---:|---:|---:|---|---|
| Cin=48 | 76.49 | 53.98 | 1.42x | True | False |
| Cin=64 | 89.61 | 67.89 | 1.32x | True | False |

两者:

- 都 build 成功。
- 都使用 WMMA 证据。
- numerical max_rel_err = 0.0。
- FP16 排序和 INT8 排序一致。

结论: 标准 3x3 dense conv 的 `K=Cin×9` 足够大，INT8 对齐不形成 Pyramid 那种 grouped conv per-group 小通道悬崖。

### 6.3 CoDriving AP 与训练 confound

`results/codriving_isobudget_verdict.csv`:

| model | protocol | AP50 |
|---|---|---:|
| base 原始 | scratch 30ep | 0.5607 |
| base-isobudget | unpruned + same second anneal | **0.6263** |
| p25 | prune + same second anneal | 0.5864 |
| p50 | prune + same second anneal | 0.6148 |
| p75 | prune + same second anneal | 0.6182 |

结论:

- “剪枝提高 AP”是训练协议 confound。
- 在 iso-budget 下，不剪枝 base 的 AP50 高于所有剪枝档。
- p50 多 seed 约 0.606-0.615，finetune AP50 噪声约 ±0.005。

### 6.4 CoDriving body-level 硬件结果与全网瓶颈

`results/codriving_dair_grid_8point_clean.csv`:

| config | FP16 body ms | INT8 body ms | energy FP16/INT8 J | engine FP16/INT8 MB |
|---|---:|---:|---:|---:|
| base | 0.466 / 0.471 remeasure | 0.290 / 0.276 remeasure | 0.179 / 0.081 | 16.04 / 9.06 |
| p25 | 0.329 | 0.252 | 0.114 / 0.073 | 9.63 / 5.94 |
| p50 | 0.262 / 0.272 remeasure | 0.210 / 0.215 remeasure | 0.081 / 0.055 | 5.36 / 3.67 |
| p75 | 0.274 / 0.280 remeasure | 0.211 / 0.225 remeasure | 0.079 / 0.052 | 3.12 / 2.12 |

全网/部署口径:

- hybrid e2e: base FP16 28.75 ms, base INT8 29.15 ms, p50 FP16 26.5 ms, p75 FP16 24.1 ms。
- pure PyTorch: 98.4 ms。
- body core 只占 e2e 很小一部分，body latency Pareto 到部署口径会被 eager 前后处理稀释。

已定位瓶颈:

- 全网 eager 有大量 CPU/launch/copy 开销。
- `generate_predicted_boxes` 每帧 CPU meshgrid + `.to(device)`，缓存后单 call 2.05 ms -> 0.086 ms。
- 下一步若要让剪枝/量化兑现到全网，必须先减少 decode/eager 开销或整段 TRT/TVM/CUDA 化。

### 6.5 CoDriving 与 Pyramid 的对照意义

当前不是“框架只对 Pyramid 有效，CoDriving 失败”。正确解释是:

- Pyramid grouped conv: P-hub 耦合强，joint 搜索必要。
- CoDriving standard conv: P/Q/S 近似可分离，serial 达到 joint。
- 框架贡献之一是能通过同一套三臂度量仪区分“何时必须协同搜索，何时串行足够”。

---

## 7. 当前局限与下一阶段建议

### 7.1 Stage1/Stage2 接口层面的局限

1. `model_classifier` 尚未成为 Stage2 search runtime 的一等输入。
2. Stage2 仍有不少历史脚本直接读 `latency_lut_*` / `ap70_model_*`，而不是完全从 manifest + evidence registry 自动发现。
3. `dispatch_plan()` 已经存在，但下一阶段还需要把它真正用于预算分配，而不是只输出报告。
4. `view_latency.coverage` 是 trace-net coverage，不是 full-model coverage。Stage2 若要 full-model 推荐，必须显式处理 skipped fusion/sparse/routing。

### 7.2 实验口径层面的局限

1. H800 TVM 是当前新增实测主后端；TRT 相关 AP/latency 在 Stage1 分类中只能是 historical evidence，不能混成新 Stage1 后端。
2. Pyramid P×Q×S 当前 JSON 是 4 seed；文字 handoff 中有 12 seed 说法，需复核。
3. Pyramid INT8 stage0 微基准不是 full-backbone INT8。
4. CoDriving body-level 加速没有直接等于 full e2e 加速，eager/decode/fusion 开销会稀释。
5. DS/闭环目前部分是 model-estimated，不是 Pyramid 专属真实 DS(AP, latency) 曲面。

### 7.3 下一阶段建议工作包

建议 Stage2 下一阶段不要先大规模增加模型，而是先把框架闭环做实:

1. **Stage1 classifier -> Stage2 gate 接入**
   - `SCAN_FAILED` 阻止自动搜索。
   - `SEPARABLE_ACCELERATION` 默认 serial/low-budget。
   - `CO_ACCELERATION_REQUIRED` 启用 per-knob joint/三臂验证。

2. **统一 Stage2 evidence registry**
   - manifest、latency LUT、AP model、Q lookup、energy/throughput、B5 audit 均有 schema。
   - 避免脚本各自手写路径和模型别名。

3. **让 `dispatch_plan()` 真正驱动预算**
   - 高 `coupling_score` knob 跑 joint / MetaSchedule。
   - 低 `coupling_score` knob 跑 serial 或复用默认 schedule。
   - 输出每个 knob 的预算原因。

4. **重跑/复核 Pyramid P×Q×S**
   - 统一 4 seed vs 12 seed artifact。
   - 如果要写论文表，必须锁定最终 JSON。

5. **补 full-backbone 或 full-segment 证据**
   - Pyramid: stage0 INT8 之外，补 stage1/stage2 或 backbone-level INT8/Tuned 证据。
   - CoDriving: 修 decode/eager 开销后重测 full e2e，确认 body 优化能否兑现。

6. **把输出从“实验脚本结果”升级为“推荐配置”**
   - 输入: model name + manifest + hardware + evidence registry。
   - 输出: 推荐 search policy、top configs、需要真测的 probes、blockers、unsupported conclusions。

---

## 8. 下一阶段 `/goal` 启动指令草案

```text
/goal 进入 Stage2 软硬件协同优化框架完善阶段。基于当前已收口的 Stage1 manifest/model_classifier 与 auto-tuning 设计，完成 Stage1→Stage2 的正式集成：把 model_classifier 三分类作为 Stage2 gate，把 stage1_bridge.SpaceSpec/dispatch_plan 作为搜索预算分配输入，统一 latency/AP/Q/evidence registry schema，并重跑或复核 Pyramid P×Q×S artifact 的 seed 口径。严格约束：H800 TVM/Relax/MetaSchedule 是新增实测主后端；TRT/AP/latency 只能按历史或单独口径标注，不能混入 Stage1 新后端；CoDriving 的可分离结论只在已测标准卷积 dense envelope 内成立；Pyramid 的耦合表述为 P(IC_BN)-hub，不写成三轴独立不可约。验收包括：Stage2 gate tests、bridge dispatch budget tests、Pyramid/CoDriving 两模型推荐策略报告、P×Q×S seed 复核、以及一份中文 Stage2 方法/实验进展 v2 文档。
```

---

## 9. 清空上下文后优先读取

1. `multi_agent/methods/progress/HANDOFF_stage2_codesign_prep_v1_zh.md`
2. `multi_agent/methods/design/auto-tuning/1_design_space_building_v1.md`
3. `multi_agent/methods/design/auto-tuning/2_design_cost_model_v1.md`
4. `multi_agent/methods/design/auto-tuning/3_design_exploring_tvm_integration_v1.md`
5. `multi_agent/methods/design/auto-tuning/4_design_ablation_proof_v1.md`
6. `framework/stage1_bridge.py`
7. `framework/search_three_arm.py`
8. `results/b4_ablation_results.json`
9. `results/b5_convergence_verification.json`
10. `results/pqs_ablation_results.json`
11. `results/coupling_map/C0c_codriving_pqs.json`
12. `results/coupling_map_matrix.json`
13. `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`

