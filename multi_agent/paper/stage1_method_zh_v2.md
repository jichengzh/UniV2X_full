# 方法（第一部分）v2：面向部署的网络--硬件协同刻画与模型分类

> 论文 Method 第一节中文稿 v2。对应实现：`framework/stage1/`、`scripts/stage1_classify_models.py`、`scripts/phase2/stage1_*`，主要产物为 Stage1 manifest、calibrated predictor 报告与 model classifier 报告。v2 相对 v1 的核心更新是：Stage1 不再被描述为“用户手工指定 dense core 的薄 adapter 流程”，而是“用户提供模型加载入口，框架自动生成可审计 trace boundary，用户审核 manifest，必要时手工 override”。

---

## 3.1 阶段一目标

阶段一的任务是在正式搜索之前确定两个问题：

1. **哪些变量可以合法搜索。** 剪枝率只能挂载在结构上必须联合剪除的通道组；量化位宽与粒度不能切碎通道依赖组；设备路由必须满足目标硬件的 op whitelist 和精度约束。
2. **当前证据能支持哪些模型级结论。** 一个 dense core scan 通过，并不等于 full-model separability。若模型存在未闭合的 fusion、attention、routing、sparse frontend 或 custom 子图，分类器必须保留 blocker 和下一步 gate。

因此，阶段一由五个实际子层组成：

```text
硬件 capability 扫描
        ↓
模型加载与 trace boundary 生成
        ↓
dense candidate DepGraph 扫描与 validation
        ↓
S2/S2.5/S3/S4 证据校准
        ↓
model classifier 三分类输出
```

## 3.2 硬件能力刻画

硬件能力由 capability YAML 给出，并归一化为
$
\mathcal{H}=\langle\mathcal{I},\mathbf{a},\mathcal{Q},\mathcal{T}\rangle
$。
其中 $\mathcal{I}$ 表示 GPU/DLA/NPU 等计算 IP、支持精度和 op whitelist；$\mathbf{a}$ 表示 INT8/FP16 通道对齐、pack factor 与 alignment enforcement；$\mathcal{Q}$ 表示可行位宽、量化粒度、对称性和逐通道限制；$\mathcal{T}$ 表示后端和工具链版本。

这个 YAML 是静态知识层，适合完全离线运行结构合法性扫描。若用户给出 3090、4090、Orin 或新硬件，只要 YAML 足够描述其 capability，Stage1 可以离线判断“结构上哪些宽度、位宽和路由选择可能合法”。但它不能产生该硬件上的延迟、吞吐或 AP 结论；任何 measured claim 必须接入真实硬件并运行 probe。当前新增实测后端策略固定为 H800 TVM/Relax/MetaSchedule；TRT 只保留为 historical evidence。

## 3.3 模型加载与 trace boundary 生成

当前 Stage1 的正确入口是：

```text
config + checkpoint + model name + minimal loader
        ↓
加载完整 torch.nn.Module
        ↓
扫描 full-model module tree
        ↓
生成 dense candidate path
        ↓
排除 sparse / fusion / routing / attention / postprocess
        ↓
合成 wrapper candidate
        ↓
输出 trace_plan
```

框架中的核心组件是 `TraceBoundaryDetector`。它先用 `ModuleTreeScanner` 收集模块树，再由 `HeuristicTagger` 给模块打标签，例如 dense、sparse frontend、fusion、attention、routing、postprocess、head 或 custom；随后 `DensePathFinder` 生成 candidate，并把不可纳入 dense DepGraph 的子图写成 typed `skipped_subgraphs`。对于 HeterModelBaseline 系列，当前实现使用 `TraceBoundaryDetector.heter_baseline_v1`；其他模型可走 generic detector 或 legacy wrapper 兼容路径。

trace plan 的关键字段包括：

```text
schema
detector
trace_confidence
coverage_scope
manual_override_used
review_required
review_reasons
selected_candidate
included_modules
ignored_layers
skipped_subgraphs
rejected_candidates
module_inventory
```

这里的设计重点是“自动生成、可审计、可复核”。用户不是先手工告诉框架扫描哪些层，而是提供模型加载入口；框架给出候选边界，用户审核 `included / ignored / skipped / rejected` 是否合理。若自动候选错误，才通过手写 `TraceAdapter`、手写 wrapper 或 detector/plugin override 兜底，并在 manifest 中留下 `manual_override_used` 或低置信度 review 标记。

## 3.4 dense candidate 的 DepGraph 扫描

选中 trace candidate 后，`graph_scan.scan()` 在该 dense wrapper 上执行 Stage1 图扫描：

1. **S0 forward dry-run。** 构造 dummy dense input，执行一次前向，记录 entry shape 和 output shape。
2. **S1 DepGraph build。** 使用 `torch-pruning.DependencyGraph` 建图，统计 prunable groups。
3. **S2 B1 prune groups。** 提取结构上必须联合剪除的最小层集合，记录通道宽度、硬件对齐下限、最大剪枝率、分组卷积上下文、op types、fanout buckets 等。
4. **S2b search groups。** 将 B1 结构真相按语义桶和 stage 聚合为搜索旋钮；聚合后的 feature 至少包含 `min_ic_bn/max_groups/op_types/fanout_buckets`。
5. **S3 B2 quant units。** 以完整 B1 组并集构造量化单元，避免量化边界切开耦合组。
6. **S4 routing segments。** 根据硬件 op whitelist 和算子类型生成逐节点路由标注，并折叠成连续路由段。
7. **S5 stats and validation。** 统计参数分布、检查对齐约束，并执行 pruning dry-run。
8. **S5b latency profile。** 可选地对 trace net 做逐层延迟刻画，明确标注为 trace-net coverage。

扫描产物是 partition manifest。它同时包含结构事实视图和搜索视图：

```text
trace
trace_plan
view_b1_prune_groups
view_b1_search_groups
view_b2_quant_units
view_d_routing
view_d_routing_segments
view_latency
checks
scan_status
```

## 3.5 runtime validation

当前 validation 是 Stage1 manifest 的一部分，而不是额外人工报告。它验证 dense candidate 是否能作为后续结构搜索的输入：

```text
wrapper_forward_dryrun
output_shape_sanity
depgraph_build
prune_dryrun
interface_invariant_check
n_prunable_groups
latency_coverage_annotation
```

具体执行方式是：先对 wrapper 做 forward dry-run；再构建 DepGraph；随后重新构造模型，使用 `MetaPruner(pruning_ratio=0.5, round_to=32)` 做一次物理通道剪除，并再次 forward；最后检查 grouped conv 的 channel/groups 一致性。`attach_runtime_validation()` 会把 forward、DepGraph 和 pruning 结果写回 `trace_plan.selected_candidate.validation`。

该 validation 的边界必须写清楚：它证明的是“所选 dense candidate 可前向、可建图、可试剪”，不证明 skipped sparse/fusion/attention/routing/custom 子图已经被覆盖，也不证明完整模型 AP 或 HV。`view_latency.coverage` 同样只表示 trace-net latency coverage，不是 full-model latency coverage。

## 3.6 结构视图到搜索变量

B1 prune groups 是结构真相，但若每个 B1 组都作为独立剪枝变量，空间会指数爆炸。因此 Stage1 生成两层视图：

- `view_b1_prune_groups`：保留 DepGraph 发现的结构事实；
- `view_b1_search_groups`：按语义桶和 stage 绑定成少量搜索旋钮。

绑定只减少自由度，不放宽合法性。每个 search group 继承成员中最严格的剪枝率上界和硬件对齐约束。路由标注也从逐算子节点折叠成最大连续可路由段，因为真实部署中每次 GPU/DLA 切换都会产生 handoff/reformat 成本。最终 manifest 将高维结构空间压缩为阶段二能枚举的低维搜索空间，同时保留 B1 原始结构作为可审计依据。

## 3.7 S2/S2.5/S3/S4 证据层

Stage1 manifest 只说明“扫描到什么”和“dense candidate 是否可用”。模型级结论还需要后续证据校准。当前实现使用以下证据：

**S2 schedule anchor。** `stage1_s2_anchor_runner.py` 在 H800 TVM/Relax/MetaSchedule 上跑两个 synthetic anchors：标准 Conv2d 和 ConvTranspose2d/deblock 代表。它枚举 P、Q、S 和 batch，测量 default、own tuned 与 schedule-swap attempt 的 latency，用来判断 schedule coupling signal 是否存在。

**S2 probe completion。** `stage1_s2_probe_completion_report.py` 读取 probe queue 和 S2 measured scan，生成 S2 完成度审计报告。它标出哪些 probe 已测、哪些 blocked、哪些只是 existing evidence binding。该报告当前主要承担 provenance 和 stop-condition 说明，不是最终三分类的核心规则输入。

**S2.5 coverage gates。** `stage1_s2_5_s3_evidence_report.py` 的 S2.5 部分关闭 targeted gates。F-Cooper MaxFusion 可被 shape-preserving proof 和 existing timing bounded，但该结论只适用于 F-Cooper MaxFusion，不能外推到所有 fusion，也不能证明 full-model separability。routing、attention 和 learned fusion 仍作为模型级 coverage gate。

**S3 quant sensitivity。** 同一脚本的 S3 部分绑定 Pyramid 和 V2X-ViT 的量化敏感性证据。Pyramid 的 TRT AP/latency 只作为 historical evidence；V2X-ViT 仍是 fake-quant 与 structural routing gate，true TRT INT8 AP/per-channel closure 仍 blocked。

**S4 three-arm validation。** `stage1_s4_three_arm_validation.py` 复用 S2 的 H800 TVM measured cells，比较 local-only、pair-search 和 joint-search。它的结论是 latency-only：在两个高风险 synthetic anchors 上，local-only winner 会变化，pair-search 在该 mini matrix 中可匹配 joint，因此后续至少需要 pair-level schedule calibration。它不提供 AP/HV，也不证明 full-model irreducible coupling。

## 3.8 calibrated predictor 与 model classifier

`calibrated_predictor` 是证据校准层。它读取 v0 predictions 和 S2/S2.5/S3/S4 证据，为每个模型写入：

```text
verdict
scope
evidence_level
evidence_categories
rule_triggers
blockers
required_next_probe_or_gate
supported_conclusions
unsupported_conclusions
historical_evidence_sources
```

它的功能不是最终三分类，而是标注“当前证据支持什么、不支持什么、哪些结论会违反证据边界”。例如，F-Cooper 由于 S2 schedule coupling 与 S4 local-only unsafe，被标为至少需要 pair-level calibration；Pyramid 被保留为 historical TRT evidence + P-hub context；V2X-ViT 保留 fake-quant/structural routing gate；Where2comm/V2VNet/DiscoNet 因缺 checkpoint 只能 architecture-only。

`model_classifier` 是最终分类层。当前实现直接读取 manifest 和 evidence directory，并调用 v0 predictor 生成基础记录，再在 `_model_record()` 中写入最终分类字段。概念上它消费的是 calibrated evidence policy；实现上并不强制先读取 `stage1_coupling_predictions_v1.json`。输出三类：

```text
CO_ACCELERATION_REQUIRED  需要协同加速
SEPARABLE_ACCELERATION    可分离加速
SCAN_FAILED              扫描失败
```

三分类之外，报告还保留 `classification`、`scope`、`evidence_level`、`measured_h800_tvm`、`historical_evidence_sources`、`blockers`、`required_next_probe_or_gate` 和 `unsupported_conclusions`。这样即使最终表格只有三类，也不会丢失证据边界。

## 3.9 当前默认模型集合与分类口径

当前默认分类器覆盖 9 个模型：

```text
CoDriving
F-Cooper
AttFuse
V2X-ViT
Pyramid lidar
Pyramid camera
Where2comm
V2VNet
DiscoNet
```

分类口径如下：

- CoDriving：只允许 scoped separable，范围限定在已测 CoDriving dense ResNet backbone envelope，不跨模型外推。
- F-Cooper：MaxFusion 局部 gate 可被 bounded，但 S2/S4 表明 schedule calibration 仍需要 pair-level 或更强证据，因此归入需要协同加速。
- AttFuse：attention/fusion 未完整 trace，dense scan 不可提升为 full-model separability。
- V2X-ViT：fake-quant 与 structural routing gate 仍存在，true TRT INT8 AP 未闭合。
- Pyramid lidar/camera：P-hub/grouped-conv context 和 historical TRT AP/latency 只作为历史证据，不能作为新增测量后端。
- Where2comm/V2VNet/DiscoNet：缺 trained checkpoint，目前是 architecture-only scan，因此归入扫描失败，而不是 trained checkpoint classification。

## 3.10 no-overpromotion 规则

分类器显式禁止以下外推：

- `groups=1` 不能作为模型级可分离证明。
- no int8 buildability cliff 不能作为模型级可分离证明。
- bridge 层低 cliff risk 不能写成 full-model separable。
- skipped fusion/attention/routing/custom subgraph 未闭合时，不能输出 full-model separability。
- random-init timing 不能代表 trained model evidence。
- static capability YAML 不能代表新硬件实测。
- TRT 不能作为新增实测后端或分类器默认后端。
- S4 latency-only 不能证明 AP/HV joint-vs-serial 行为。

`no_overpromotion=True` 是最终报告的验收条件。如果模型有 blocker 但输出 full-model separable verdict，则视为 overpromotion。

## 3.11 输出形式

阶段一最终输出包括两类文件。

第一类是扫描 manifest：

```text
framework/partitions/*_partition.yaml
results/autoscan_*_partition.yaml
```

它们用于定义阶段二搜索空间，包含硬件能力、结构视图、trace plan、validation 和 latency coverage。

第二类是证据与分类报告：

```text
results/stage1_model_predict/stage1_coupling_predictions_v1.json
results/stage1_model_predict/calibrated_predictor_report_v1.json
results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json
results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

它们用于说明当前模型级证据状态、三分类结果、blocker、下一步 gate/probe 和禁止外推项。

## 图 1 配图说明（v2）

**图 1. Stage1 面向部署的网络--硬件协同刻画与证据分类流程。** 左侧为硬件 capability YAML，输出静态硬件能力 $\mathcal{H}$；右侧为模型入口，用户提供 config、checkpoint 和 loader，Stage1 加载完整模型并由 `TraceBoundaryDetector` 生成 trace plan。中部对 selected dense candidate 执行 forward dry-run、DepGraph build、B1/B2/D 三视图提取、0.5 pruning dry-run 和 trace-net latency profiling，输出 partition manifest。下部 S2/S2.5/S3/S4 证据进入 calibrated predictor，形成 evidence categories、rule triggers、blockers 和 unsupported conclusions；最终 model classifier 输出三分类，同时保留 next gate/probe 与 no-overpromotion 检查。灰色分支表示 sparse、fusion、attention、routing 和 postprocess 等 skipped subgraphs：它们被记录和 gate，而不是被静默视为已覆盖。

```text
Hardware YAML ──▶ H = <IP, align, quant, toolchain>
                         │
                         ▼
config + ckpt + model name + loader
                         │
                         ▼
              full model module tree scan
                         │
                         ▼
        TraceBoundaryDetector / DensePathFinder
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
 selected dense candidate        skipped / rejected
          │                             │
          ▼                             ▼
 forward + DepGraph + prune dry-run   review gates
          │
          ▼
 B1 prune groups + B2 quant units + D routing
          │
          ▼
 partition manifest
          │
          ▼
 S2/S2.5/S3/S4 evidence
          │
          ▼
 calibrated predictor
          │
          ▼
 model classifier: co-accel / separable / scan failed
```

---

## 实现状态（2026-06-24）

当前实现已经完成：`TraceBoundaryDetector`、`BoundaryValidator.graph_scan_integrated_v1`、manifest predictor fields、H800 TVM evidence ingestion policy、三分类 `model_classifier`、中文/英文开源 README 与新增模型 adapter 教程。需要继续推进的部分是：对更多新模型提供 detector/plugin override 机制，对 attention/fusion/routing 子图做更完整 trace 或独立证据闭合，以及让 `model_classifier` 显式消费 `stage1_coupling_predictions_v1.json`，使“校准层 → 分类层”的工程边界更干净。
