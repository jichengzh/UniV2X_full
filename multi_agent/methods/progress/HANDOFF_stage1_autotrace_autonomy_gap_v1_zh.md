# 交接文档：Stage1 AutoTraceAdapter 自治化缺口与下一步重构

日期：2026-06-24

## 0. 工作边界

从本交接开始，`github/stage1-model-scanner-aaai/` 由另一个窗口维护。后续本窗口的代码修改不应改动该目录，除非用户重新明确授权。

本文档只记录主仓库当前 Stage1 实验进展、AutoTraceAdapter 的真实状态、存在的问题，以及下一轮应实现的自动 trace-boundary 识别流程。

## 1. 当前实验进展

Stage1 端到端模型分类器 v1 已经跑通，当前主线目标从 Safe Predictor v0 / Calibrated Predictor v1 收敛到模型级分类器：

```text
Stage1 manifest
  + S2/S2.5/S3/S4 evidence
  -> calibrated / model classifier
  -> 三分类 + 细粒度 verdict + blocker + next gate/probe + unsupported conclusions
```

当前分类器输出两层结果：

- `acceleration_class`：最终用户可读的三分类。
- `classification` / `verdict`：细粒度证据状态，用于审计和论文方法说明。

三分类定义：

| enum | 中文标签 | 含义 |
|---|---|---|
| `CO_ACCELERATION_REQUIRED` | 需要协同加速 | 模型存在剪枝、量化、调度、fusion、routing、attention 或 P-hub 等跨阶段/跨模块耦合，需要联合处理。 |
| `SEPARABLE_ACCELERATION` | 可分离加速 | 只在明确证据范围内成立。当前仅 CoDriving 的 scoped dense ResNet backbone envelope 满足。 |
| `SCAN_FAILED` | 扫描失败 | 没有形成有效 trained-checkpoint model classification；常见原因是缺 ckpt、只有 architecture-only scan、random-init sidecar 或 trace coverage 不足。 |

当前 9 模型分类状态：

| 模型 | 三分类 | 当前解释 |
|---|---|---|
| `codriving` | `SEPARABLE_ACCELERATION` | 仅限 CoDriving measured dense ResNet backbone envelope，不外推到其他模型或 skipped subgraph。 |
| `fcooper` | `CO_ACCELERATION_REQUIRED` | MaxFusion 形状/时延上下文已绑定，但 H800 S2/S4 仍显示 schedule calibration required。 |
| `attfuse` | `CO_ACCELERATION_REQUIRED` | attention/fusion 未完整纳入 Stage1 trace，不能输出 full-model separable。 |
| `v2xvit` | `CO_ACCELERATION_REQUIRED` | V2XTransformer/routing/fake-quant gate 未闭合，true TRT INT8 AP 仍 blocked/not done。 |
| `pyramid_lidar` | `CO_ACCELERATION_REQUIRED` | P-hub/grouped-conv context 与 historical TRT Q/AP 只作为历史证据，不能升级为新后端结论。 |
| `pyramid_camera` | `CO_ACCELERATION_REQUIRED` | 同 Pyramid lidar，且 camera/LSS 路径 trace coverage 更受限。 |
| `where2comm` | `SCAN_FAILED` | 当前只有 architecture-only dense-core scan + random-init fusion sidecar，不是 trained checkpoint classification。 |
| `v2vnet` | `SCAN_FAILED` | 同上。 |
| `disconet` | `SCAN_FAILED` | 同上。 |

已落实的后端纪律：

- 新增实测后端统一为 H800 TVM / Relax / MetaSchedule。
- `allowed_new_measurement_backends == ["h800_tvm"]`。
- TRT 只能进入 historical evidence，不能进入 `measured_h800_tvm`。
- Pyramid TRT AP/latency 只能是 `historical_trt_evidence`。
- V2X-ViT true TRT INT8/AP 仍是 blocked / not done。
- `groups=1`、`no cliff`、`bridge SEPARABLE` 均不能作为模型级可分离证明。

最近重点验证：

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py
```

最近结果：`23 passed`。

CLI 生成也已通过：

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

## 2. 当前 AutoTraceAdapter 到底实现了什么

当前 `AutoTraceAdapter` 位于：

- `framework/stage1/auto_trace.py`

它的目标是减少每个模型都手写 `_Net` 子类的成本。当前它已经实现了以下能力：

- 统一 `TraceAdapter` 接口：`build_trace_net()`、`ignored_layers()`、`semantic_bucket()`。
- 通过 `build_fn(config_path, ckpt_path, device)` 构建 trace-ready dense net。
- 根据 `bev_shape` 自动生成 dummy dense BEV input。
- 对常见 head 名称进行 ignored layer 自动探测：`cls_head`、`reg_head`、`dir_head`、`single_head`。
- 复用 `_generic_bucket()` 做 layer semantic bucket。
- 提供通用 wrapper：
  - `_CoDrivingTraceNet`
  - `_HeterBaselineTraceNet`
- 注册了 9 个模型的 AutoTraceAdapter 条目：
  - CoDriving
  - F-Cooper
  - AttFuse
  - V2X-ViT
  - Pyramid lidar/camera
  - Where2comm
  - V2VNet
  - DiscoNet

当前它确实比完全手写 `TraceAdapter` 少了样板代码，但它仍然不是完整自动扫描器。

## 3. 当前 AutoTraceAdapter 的核心问题

当前最主要的问题是：它仍然要求用户或开发者自己指定 Stage1 应该扫描什么。

具体表现：

| 事项 | 当前由谁决定 | 问题 |
|---|---|---|
| `build_fn` | 人手写 | 人决定 full model 如何变成 trace-ready dense net。 |
| `bev_shape` | 人写或半手工推断 | Stage1 没有可靠自动推断 dense entry shape 的统一机制。 |
| `skipped_desc` / skipped modules | 人写 | sparse/fusion/routing/postprocess 的排除逻辑没有成为框架级判断。 |
| wrapper candidate | 人手写或复用少数 wrapper | 框架没有自动从 module tree 生成 wrapper candidate。 |
| dense candidate path | 人通过模型知识指定 | 没有通用算法从完整模型中发现 backbone/neck/head dense path。 |
| ignored interface layers | 名称启发 + 人补充 | 仅能覆盖 head 名称，不能可靠识别 fusion 输入保护层、LayerNorm/ConvNeXt 等结构约束。 |
| trace coverage | manifest 记录结果 | 但没有自动给出“为什么选择这些 included/ignored/skipped”的完整审计报告。 |

因此，当前 AutoTraceAdapter 的“auto”主要体现在：

- 统一 adapter 调用；
- 自动 head 探测；
- 复用通用 wrapper；
- 减少一部分模型 glue code。

它没有做到：

- 输入完整模型后自动识别 dense core；
- 自动排除 sparse / fusion / routing / postprocess；
- 自动生成 wrapper；
- 自动判定 skipped 子图对 full-model verdict 的影响；
- 让用户只负责提供 config、ckpt 和审计结果。

这也是下一轮重构最重要的方向。

## 4. 正确的目标流程

期望流程应是：

```text
用户提供 config + ckpt + model name
        ↓
Stage1 加载完整模型
        ↓
自动扫描模块树
        ↓
自动识别 dense candidate path
        ↓
自动排除 sparse / fusion / routing / postprocess
        ↓
自动生成 wrapper candidate
        ↓
dry-run 验证
        ↓
trace / DepGraph 验证
        ↓
生成 included / ignored / skipped manifest
        ↓
分类器保守判断 evidence level
```

用户的默认职责应该只有：

1. 提供模型配置、checkpoint 和 model name。
2. 审核 Stage1 生成的扫描报告是否合理。
3. 如果报告不合理，再通过明确接口或教程提供手工 override。

手工 adapter 不应是默认路径，而应是 fallback。

## 5. 框架应具备的基本判断逻辑

Stage1 应该内置一套可解释的 module tagging 规则，而不是让用户直接声明哪些该扫、哪些该跳过。

### 5.1 默认 included 候选

优先纳入 dense candidate path：

- `Conv2d`
- `ConvTranspose2d`
- `Linear`
- `BatchNorm2d`
- 常见 dense backbone / neck / shrinker：
  - `backbone`
  - `resnet`
  - `base_bev_backbone`
  - `pyramid_backbone`
  - `neck`
  - `deblock`
  - `shrink`
  - `shrinker`

这些模块不是天然可剪/可量化结论，只是 dense path 候选。必须通过 forward sanity、DepGraph build、prune dry-run 和 shape check 才能进入最终 included manifest。

### 5.2 默认 skipped 候选

默认排除并标记为 blocker 或 coverage cap：

| 类型 | 关键词/结构信号 | manifest type |
|---|---|---|
| sparse / geometry preprocess | `pillar_vfe`, `voxel`, `scatter`, `sparse`, `quickcumsum`, `cumsum`, `lift`, `splat`, `geometry` | `sparse_or_geometry_preprocess` |
| fusion / alignment | `fusion`, `warp`, `affine`, `pairwise_t_matrix`, `record_len`, `collab` | `fusion_or_alignment` |
| attention / transformer | `attention`, `transformer`, `HMSA`, `MSwin`, `attfuse` | `attention_or_routing_fusion` |
| routing / communication | `where2comm`, `v2v`, `disco`, `communication`, `routing`, `message_passing` | `attention_or_routing_fusion` |
| postprocess / decoding | `postprocess`, `nms`, `decode`, `box_coder`, `proposal` | `postprocess_or_decode` |
| dataset / loss / metric | `loss`, `assigner`, `target`, `metric`, `eval` | `training_or_eval_only` |

默认原则：

- skipped 不代表不重要。
- skipped 代表当前 Stage1 dense-core trace 没覆盖它。
- 如果 skipped 是 fusion/attention/routing，分类器不能输出 full-model separability。
- 如果 skipped 是 sparse preprocess，但后续 dense BEV entry 有明确语义和 shape，可以允许 scoped dense-core verdict。

### 5.3 默认 ignored 候选

默认 ignored layer 应包括：

- 输出 heads：`cls_head`, `reg_head`, `dir_head`, `single_head`
- occupancy / auxiliary heads
- 固定接口层：
  - fusion 输入通道保护层；
  - 需要保持 downstream shape 的 shrinker 输出层；
  - 已知结构化剪枝会破坏 normalized shape 的 LayerNorm / ConvNeXt adapter；
  - camera/LSS aligner 中无法安全 prune 的接口模块。

ignored 与 skipped 不同：

- ignored 仍在 wrapper forward 中，可用于建立依赖或保持输出。
- skipped 不进入 trace graph，只作为 coverage/blocker 记录。

## 6. 建议的重构设计

建议新增以下抽象，而不是继续扩大 `AutoTraceAdapter` registry。

### 6.1 ModelSpec

用户输入最小化为：

```yaml
model_name: my_model
config_path: /path/to/config.yaml
ckpt_path: /path/to/checkpoint.pth
code_root: /path/to/model/repo
model_builder: optional.import.path:build_model
```

其中 `model_builder` 是可选 fallback。理想情况下 Stage1 可以从 config 或项目约定发现 builder；发现不了再要求用户补充。

### 6.2 FullModelLoader

职责：

- 加载 config；
- 构建 full model；
- 加载 checkpoint；
- 标记 `ckpt_status`；
- 不做 trace 边界决策。

输出：

```python
LoadedModel(
    model_name,
    full_model,
    config,
    ckpt_status,
    provenance
)
```

### 6.3 ModuleTreeScanner

职责：

- 遍历 `full_model.named_modules()`；
- 记录模块名、类型、参数量、子树结构；
- 初步打 tag：
  - dense candidate；
  - sparse candidate；
  - fusion/routing candidate；
  - postprocess candidate；
  - head candidate；
  - unknown custom candidate。

输出 module inventory。

### 6.4 DensePathFinder

职责：

- 从 module inventory 中寻找可 trace dense path。
- 优先识别 backbone -> neck/shrinker -> heads。
- 尝试定位 dense entry：
  - post-scatter BEV；
  - backbone input；
  - camera BEV encoder output；
  - user-provided sample input fallback。

输出多个 wrapper candidate，而不是单个强行结论：

```json
[
  {
    "candidate_id": "dense_bev_backbone_path",
    "entry": "post_scatter_bev",
    "included_modules": ["backbone_m1", "shrinker_m1", "cls_head", "reg_head"],
    "ignored_layers": ["cls_head", "reg_head"],
    "skipped_subgraphs": ["pillar_vfe", "scatter", "fusion_net"],
    "input_shape": [1, 64, 512, 512],
    "confidence": "medium"
  }
]
```

### 6.5 WrapperSynthesizer

职责：

- 根据 wrapper candidate 生成可执行 wrapper。
- 首轮可以不做源码生成，而用通用 `ModulePathWrapper`：
  - 顺序执行若干 module path；
  - 支持 dict I/O adapter；
  - 支持手动/自动 entry tensor name；
  - 支持 head outputs。

如果遇到复杂 forward，如 Pyramid `forward_single` 或 CoDriving dict I/O，先使用 framework 内置 wrapper pattern；不要要求用户一开始手写完整 adapter。

### 6.6 BoundaryValidator

每个 candidate 必须通过分级验证：

1. full model load sanity；
2. wrapper forward dry-run；
3. output shape sanity；
4. torch_pruning DepGraph build；
5. prune dry-run；
6. interface invariant check；
7. latency coverage annotation。

验证失败不应导致整体扫描直接失败，而应输出 rejected candidate：

```json
{
  "candidate_id": "dense_bev_backbone_path",
  "status": "rejected",
  "failed_at": "depgraph_build",
  "error": "LayerNorm normalized_shape mismatch after prune",
  "suggested_override": "mark aligner_m2 as ignored"
}
```

### 6.7 TracePlan / ManifestWriter

最终 manifest 需要明确：

- `included_modules`
- `ignored_layers`
- `skipped_subgraphs`
- `rejected_candidates`
- `selected_candidate`
- `selection_reason`
- `coverage_scope`
- `trace_confidence`
- `manual_override_used`
- `review_required`

分类器必须消费这些字段，而不是只看模型名。

## 7. 分类器如何接入新 trace plan

分类器当前已经保守处理 blocker，但下一步应该减少模型名分支，改为依赖 trace/evidence 状态。

建议新 evidence level：

| evidence level | 条件 |
|---|---|
| `trace_auto_validated_dense_core_only` | 自动候选通过 DepGraph/dry-run，但有 skipped subgraph。 |
| `trace_auto_validated_full_dense_path` | 自动候选覆盖模型主要 dense path，且 skipped 不影响目标结论。 |
| `trace_manual_override_required` | 自动候选失败或 confidence 低，需要用户确认。 |
| `architecture_only_missing_ckpt` | 缺 checkpoint，只能结构扫描。 |
| `measured_h800_tvm_latency_bound` | 已有 H800 TVM latency evidence。 |
| `historical_trt_evidence_only` | 只有 historical TRT，不可作为新增实测后端。 |

三分类应由以下原则决定：

- 有 fusion/attention/routing skipped blocker 且未闭合 evidence：`CO_ACCELERATION_REQUIRED`。
- 缺 checkpoint 或 only random-init：`SCAN_FAILED`。
- 自动 trace 覆盖 scope 明确、无 blocker、且有同 scope measured evidence：才允许 `SEPARABLE_ACCELERATION`。
- 任何 `manual_override_required` 默认不允许 full-model separable。

## 8. 对当前 AutoTraceAdapter 的处理建议

不要直接删除当前 AutoTraceAdapter。建议把它重新定位为过渡层：

```text
AutoTraceAdapter v0
  当前 registry + build_fn 模式
  用于复现实验和已有 9 模型

TraceBoundaryDetector v1
  新的默认路径
  输入 config + ckpt + model name
  自动产生 TracePlan

ManualTraceAdapter
  fallback / override
  仅当自动候选失败或用户审核不通过时使用
```

重构顺序建议：

1. 定义 `TracePlan` schema 和测试。
2. 写 `ModuleTreeScanner`，先只输出 inventory，不改扫描逻辑。
3. 写 `HeuristicTagger`，将 module 标为 dense/sparse/fusion/routing/postprocess/head/unknown。
4. 写 `DensePathFinder`，先支持 HEAL `HeterModelBaseline`。
5. 写 `BoundaryValidator`，复用现有 `graph_scan.scan` 的 forward/DepGraph/dryrun 能力。
6. 让 `graph_scan.scan` 接受 `TracePlanAdapter`。
7. 分类器消费 `trace_confidence`、`manual_override_used`、`coverage_scope`。
8. 保留旧 AutoTraceAdapter registry 做 regression baseline。

## 9. 下一轮最小可交付

建议下一轮不要一次性追求任意模型全自动，而是做一个可验证原型：

目标：

```text
输入 HEAL HeterModelBaseline config + ckpt + model_name
自动生成 TracePlan
自动识别:
  included = backbone_m1 + shrinker_m1 + heads
  skipped = pillar_vfe + scatter + fusion_net
  ignored = cls/reg/dir heads + 必要 interface layers
跑通 forward dry-run + DepGraph build + prune dry-run
输出 included / ignored / skipped manifest
```

推荐测试：

- F-Cooper：应识别 MaxFusion skipped，ckpt ok。
- AttFuse：应识别 AttFusion skipped，ckpt ok。
- Where2comm：应识别 routing/fusion skipped，缺 ckpt -> architecture_only，不允许 trained classification。
- V2VNet / DiscoNet：同上。

验收条件：

- 用户不需要手写 `skipped_desc`。
- 用户不需要手写 `_HeterBaselineTraceNet`。
- 用户只提供 config、ckpt、model name。
- 自动报告中必须解释为什么某些模块 included、ignored、skipped。
- 自动候选失败时必须给出 rejected candidate 和建议 override。
- 分类器不能因为自动 trace 成功就输出 full-model separable。

## 10. 关键文件

当前实现相关：

- `framework/stage1/adapters.py`
- `framework/stage1/auto_trace.py`
- `framework/stage1/graph_scan.py`
- `framework/stage1/latency_profile.py`
- `framework/stage1/model_classifier.py`
- `scripts/autoscan_reproduce_check.py`
- `scripts/stage1_classify_models.py`

已有测试：

- `framework/tests/test_stage1_autoscan_extensions.py`
- `framework/tests/test_stage1_manifest_predictor_fields.py`
- `framework/tests/test_stage1_model_classifier_contract.py`
- `framework/tests/test_coupling_predictor_static.py`
- `framework/tests/test_calibrated_predictor_v1.py`

建议新增测试：

- `framework/tests/test_stage1_trace_plan_schema.py`
- `framework/tests/test_stage1_module_tree_scanner.py`
- `framework/tests/test_stage1_trace_boundary_detector_heter_baseline.py`
- `framework/tests/test_stage1_trace_plan_classifier_gates.py`

## 11. 交接重点

当前 Stage1 的实验和分类器已经足够支撑论文阶段的保守分类报告，但 AutoTraceAdapter 不能被描述成“完整自动模型扫描器”。

准确说法应是：

```text
当前 AutoTraceAdapter 是一个减少手写 wrapper 的过渡层。
它仍然依赖人工提供 dense-core 构建函数、BEV shape 和 skipped module 描述。
下一步需要把 trace boundary discovery 变成框架级能力：
由 Stage1 自动扫描完整模型模块树，提出 dense wrapper candidate，
验证后生成 included/ignored/skipped manifest，
用户只负责提供 config/ckpt/model name 并审核报告。
```

这是下一轮 Stage1 自动化工作的核心。

## 12. 本轮 trace-boundary 自治化原型进展

日期：2026-06-24

本轮在主仓库完成了 Stage1 trace-boundary 自治化的第一阶段原型；仍然遵守边界：未修改 `github/stage1-model-scanner-aaai/`，该目录继续由另一个窗口维护。

### 12.1 已新增的核心代码

- 新增 `framework/stage1/trace_plan.py`
  - 定义 `stage1_trace_plan_v1` schema。
  - 实现 `ModuleTreeScanner`：遍历 full model 的 `named_modules()`，记录 path/type/param/tree depth。
  - 实现 `HeuristicTagger`：基于模块名、类型和结构启发式标记 dense candidate、sparse/geometry、fusion/alignment、attention/routing、postprocess、head/ignored candidate。
  - 实现 `TraceBoundaryDetector`：首个 detector 支持 HEAL `HeterModelBaseline` 风格模型。
  - 实现 `legacy_trace_plan_from_manifest()`：旧 manifest 没有 `trace_plan` 时可规范化成同一 schema，便于 classifier 使用统一字段。

- 更新 `framework/stage1/auto_trace.py`
  - `AutoTraceAdapter` 增加 `trace_plan` 与 `build_trace_plan()`。
  - `_build_heter_baseline()` 不再只返回 wrapper，而是在加载完整 HEAL `HeterModelBaseline` 后先运行 `TraceBoundaryDetector`，再把 plan 附加到 trace wrapper。
  - 旧 registry 仍保留为 regression baseline；没有 full-model detector 的条目会被规范化为 legacy boundary plan。

- 更新 `framework/stage1/graph_scan.py`
  - `scan()` 会把 `trace_plan` 写入 manifest。
  - 已有 S0 forward、S1 DepGraph、S5 prune dry-run 的结果会回填到 `trace_plan.selected_candidate.validation`：
    - `wrapper_forward_dryrun`
    - `depgraph_build`
    - `prune_dryrun`
    - `n_prunable_groups`

- 更新 `framework/stage1/coupling_predictor.py`
  - 预测前保证 manifest 一定有 `trace_plan`。
  - 将 plan 中的 skipped subgraphs 合并到统一 skipped 口径。
  - 将 `manual_override_used`、`trace_confidence=low`、`review_required`、`missing ckpt` 转成保守 blocker / next gate。

- 更新 `framework/stage1/model_classifier.py`
  - 分类报告输出新增：
    - `trace_confidence`
    - `coverage_scope`
    - `manual_override_used`
    - `review_required`
    - `trace_plan`
  - 缺 checkpoint 的 trace plan 会保持 `SCAN_FAILED`，不会变成 trained checkpoint classification。

### 12.2 已覆盖的验收点

新增测试：

- `framework/tests/test_stage1_trace_boundary_detector_heter_baseline.py`

覆盖内容：

- `ModuleTreeScanner` / `HeuristicTagger` 能识别：
  - `backbone_m1`、`shrinker_m1`、heads 为 dense / ignored candidates；
  - `encoder_m1.pillar_vfe`、`encoder_m1.scatter` 为 sparse/geometry skipped；
  - `fusion_net` 为 fusion/attention/routing skipped。
- F-Cooper 原型：
  - 自动识别 `MaxFusion` 为 skipped；
  - `ckpt_status=ok`；
  - included 包含 `backbone_m1`、`shrinker_m1`、`cls_head`、`reg_head`、`dir_head`；
  - ignored 包含 `cls_head`、`reg_head`、`dir_head`；
  - `manual_override_used=False`；
  - `coverage_scope=dense_core_only`。
- AttFuse 原型：
  - 自动识别 `AttFusion` 为 skipped；
  - 标记 `attention_fusion_coverage_anchor`；
  - 作为 full-model verdict blocker。
- Where2comm 原型：
  - 自动识别 routing/fusion skipped；
  - `ckpt_status=missing_architecture_scan_only`；
  - classifier 输出 `SCAN_FAILED`；
  - 不允许输出 trained checkpoint classification。

### 12.3 本轮验证结果

已运行：

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py \
  framework/tests/test_stage1_autoscan_extensions.py \
  framework/tests/test_stage1_trace_boundary_detector_heter_baseline.py
```

结果：`32 passed`。

额外 detector-only 真实 HEAL config 验收也已跑通：

- F-Cooper：真实 ckpt 加载成功，`fusion_net` 标为 `maxfusion_coverage_anchor`，`ckpt_status=ok`。
- AttFuse：真实 ckpt 加载成功，`fusion_net` 标为 `attention_fusion_coverage_anchor`，`ckpt_status=ok`。
- Where2comm：无 ckpt，`fusion_net` 标为 `routing_fusion_coverage_anchor`，`ckpt_status=missing_architecture_scan_only`。
- V2VNet / DiscoNet：无 ckpt，均识别 `backbone_m1`、`shrinker_m1`、heads included，`fusion_net` 为 routing fusion skipped，`pillar_vfe` / `scatter` 为 sparse/geometry skipped。

已运行 CLI / schema / py_compile：

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md

python -m py_compile \
  framework/stage1/coupling_predictor.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage1/model_classifier.py \
  framework/stage1/trace_plan.py \
  framework/stage1/auto_trace.py \
  framework/stage1/graph_scan.py \
  scripts/stage1_classify_models.py
```

结果：通过。分类报告仍满足：

- `backend_policy.default_new_measurement_backend == "h800_tvm"`
- `allowed_new_measurement_backends == ["h800_tvm"]`
- TRT 只保留为 historical evidence
- `no_overpromotion == True`

### 12.4 当前仍然存在的限制

本轮完成的是可审计原型，不是任意模型的全自动扫描器。

仍未完全自动化的部分：

- full model builder discovery 还没有完全自动化；HEAL HeterModelBaseline 目前走内置 loader，任意外部模型仍需要 builder/plugin。
- wrapper synthesizer 仍复用 `_HeterBaselineTraceNet`，还没有实现通用源码无关的顺序 wrapper / dict I/O wrapper 合成器。
- dense entry shape 对 HEAL lidar 模型可以从 config 推断，但任意模型的 sample input / dense entry 仍需要下一阶段完善。
- 旧 YAML 结果没有重新跑 autoscan，因此当前 classifier 对旧 manifest 会生成 `legacy_trace_adapter_manifest_normalizer` trace plan；要获得 full-model module-tree derived plan，需要重跑 autoscan 并保存新 manifest。
- detector 当前只把 fusion/attention/routing 未闭合变成 blocker/review gate；不解决这些子图本身的可 trace 化。

下一步建议：

1. 重跑 `scripts/autoscan_reproduce_check.py --models a1,a2 --save-yaml` 或等价流程，刷新 F-Cooper / AttFuse / Where2comm / V2VNet / DiscoNet manifest，使结果文件携带 full-model-derived `trace_plan`。
2. 将 `TraceBoundaryDetector` 扩展到 CoDriving、Pyramid、V2X-ViT，而不是只依赖 legacy plan normalization。
3. 实现真正的 `DensePathFinder` / `WrapperSynthesizer` / `BoundaryValidator` 三段式，使用户只提供 config + ckpt + model name 后即可产生多个 wrapper candidate 和 rejected candidate。
4. 在 classifier 中逐步减少模型名硬编码，更多依赖 `trace_confidence`、`coverage_scope`、`manual_override_used`、`ckpt_status`、`skipped_subgraphs` 和 H800 TVM evidence。
