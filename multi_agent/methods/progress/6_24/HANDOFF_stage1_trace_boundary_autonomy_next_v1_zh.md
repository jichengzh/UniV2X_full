# 交接文档：Stage1 Trace-Boundary 自治化下一轮完整计划

日期：2026-06-24

## 0. 工作边界

从本交接开始继续遵守以下边界：

- 不修改 `github/stage1-model-scanner-aaai/`。该目录由另一个窗口维护。
- 本窗口只修改主仓库中的 `framework/`、`scripts/`、`results/stage1_model_predict/`、`multi_agent/methods/progress/` 等当前 Stage1 主线文件。
- 新增实测后端仍统一为 H800 TVM / Relax / MetaSchedule。
- TRT 只能作为 historical evidence，不能作为新增 probe backend、分类器默认后端或新验收后端。

## 1. 当前已完成状态

上一轮已经完成 Stage1 trace-boundary 自治化第一阶段原型：

- 新增 `framework/stage1/trace_plan.py`
  - `stage1_trace_plan_v1` schema
  - `ModuleTreeScanner`
  - `HeuristicTagger`
  - `TraceBoundaryDetector`
  - `legacy_trace_plan_from_manifest()`
- `AutoTraceAdapter` 已能持有 `trace_plan`。
- `graph_scan.scan()` 已能把 `trace_plan` 写入 manifest，并回填 forward / DepGraph / prune dry-run validation。
- `coupling_predictor` / `model_classifier` 已开始消费：
  - `trace_confidence`
  - `coverage_scope`
  - `manual_override_used`
  - `review_required`
  - `trace_plan`
- HEAL `HeterModelBaseline` 原型已覆盖：
  - F-Cooper
  - AttFuse
  - Where2comm
  - V2VNet
  - DiscoNet

已验证：

- F-Cooper：真实 ckpt 加载成功，`fusion_net` 标为 `maxfusion_coverage_anchor`，`ckpt_status=ok`。
- AttFuse：真实 ckpt 加载成功，`fusion_net` 标为 `attention_fusion_coverage_anchor`，`ckpt_status=ok`。
- Where2comm：无 ckpt，`fusion_net` 标为 `routing_fusion_coverage_anchor`，`ckpt_status=missing_architecture_scan_only`。
- V2VNet / DiscoNet：无 ckpt，均识别 `backbone_m1`、`shrinker_m1`、heads included，`fusion_net` 为 routing fusion skipped，`pillar_vfe` / `scatter` 为 sparse/geometry skipped。

最近验证命令结果：

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py \
  framework/tests/test_stage1_autoscan_extensions.py \
  framework/tests/test_stage1_trace_boundary_detector_heter_baseline.py
```

结果：`32 passed`。

CLI / schema / py_compile 已通过：

```bash
PYTHONPATH=${V2X_ROOT} \
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

## 2. 为什么还不是完整自动扫描器

当前系统已经能自动生成可审计的 trace boundary plan，但还没有完成“用户只提供 config + ckpt + model name 后，完整自动生成并验证扫描候选”的目标。

当前限制：

- HEAL `HeterModelBaseline` 支持较完整，但 CoDriving / Pyramid / V2X-ViT 仍主要依赖旧 adapter 或 legacy manifest normalization。
- full model loader 还不是通用发现机制；当前 HEAL 模型走内置 loader，任意外部模型仍需要 builder/plugin。
- wrapper 还不是自动合成；当前检测器能生成 `TracePlan`，但真正 trace 的 wrapper 仍复用 `_HeterBaselineTraceNet`。
- dense entry shape 只对 HEAL lidar BEV 路径有较可靠推断，任意模型的 sample input / dense entry discovery 还不完整。
- fusion / attention / routing 当前只自动识别为 blocker，还没有被纳入 full-model trace closure。
- 默认分类报告当前仍读取旧 autoscan YAML；这些 YAML 需要重跑 autoscan 才能携带 full-model-derived `trace_plan`。

## 3. 下一轮目标

下一轮目标是把第一阶段原型推进为完整的 trace-boundary 自动候选生成与验证流程：

```text
用户提供 config + ckpt + model name
        ↓
Stage1 加载 full model
        ↓
ModuleTreeScanner 生成 module inventory
        ↓
HeuristicTagger 标注 dense / sparse / fusion / routing / postprocess / head
        ↓
DensePathFinder 生成一个或多个 dense path candidate
        ↓
WrapperSynthesizer 自动生成 wrapper candidate
        ↓
BoundaryValidator 执行 forward dry-run / DepGraph / prune dry-run
        ↓
TracePlan 写入 included / ignored / skipped / rejected_candidates
        ↓
Stage1 manifest 写入 trace_plan
        ↓
Classifier 基于 trace_confidence / coverage_scope / manual_override_used / evidence level 保守三分类
```

完成后的默认用户职责应压缩为：

1. 提供 `config_path`、`ckpt_path`、`model_name`。
2. 审核 Stage1 自动生成的扫描报告是否合理。
3. 只有当自动报告不合理时，才使用 override/plugin 教程修正边界。

## 4. 下一轮实施计划

### 4.1 刷新现有 autoscan manifest

先重跑当前已经支持的 HEAL HeterBaseline 模型，使磁盘结果从 legacy normalized plan 变成 full-model-derived `trace_plan`。

目标模型：

- F-Cooper
- AttFuse
- Where2comm
- V2VNet
- DiscoNet

建议命令：

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
python scripts/autoscan_reproduce_check.py \
  --models fcooper,attfuse,where2comm,v2vnet,disconet \
  --device cpu \
  --save-yaml
```

验收：

- `results/autoscan_fcooper_partition.yaml` 等输出 manifest 包含 `trace_plan.schema == stage1_trace_plan_v1`。
- F-Cooper / AttFuse 的 `trace_plan.detector == TraceBoundaryDetector.heter_baseline_v1`。
- Where2comm / V2VNet / DiscoNet 保持 `ckpt_status=missing_architecture_scan_only`。

### 4.2 实现 DensePathFinder

新增或扩展：

- `framework/stage1/trace_plan.py`
- 可拆分新增 `framework/stage1/trace_boundary.py`

职责：

- 从 module inventory 产生多个候选 dense path，而不是只选择一个 HeterBaseline 模板。
- 候选至少包含：
  - `candidate_id`
  - `entry`
  - `included_modules`
  - `ignored_layers`
  - `skipped_subgraphs`
  - `input_shape`
  - `confidence`
  - `selection_reason`

优先规则：

- backbone / neck / shrinker / heads 优先作为 dense core。
- sparse / geometry preprocess 默认 skipped。
- fusion / attention / routing 默认 skipped 且作为 full-model blocker。
- postprocess / decode 默认 skipped。
- heads 默认 ignored，但仍可保留在 wrapper forward 中用于输出 sanity。

### 4.3 实现 WrapperSynthesizer

新增：

- `framework/stage1/wrapper_synthesizer.py` 或放入 `trace_plan.py` 的小型原型。

第一版支持：

- HEAL `HeterModelBaseline`
  - `backbone_m1`
  - `shrinker_m1`
  - `cls_head`
  - `reg_head`
  - `dir_head`
- CoDriving
  - backbone dict I/O
  - optional shrink
  - cls/reg heads
- fallback `ModulePathWrapper`
  - 按 module path 顺序执行
  - 支持 dict input / tensor input 的最小适配

验收：

- 新 wrapper 不能只依赖人工 `_HeterBaselineTraceNet`。
- 若 wrapper 生成失败，必须输出 rejected candidate，而不是静默回退成可用结论。

### 4.4 实现 BoundaryValidator

新增：

- `framework/stage1/boundary_validator.py` 或合入 `trace_plan.py`

每个 candidate 分级验证：

1. full model load sanity
2. wrapper forward dry-run
3. output shape sanity
4. torch_pruning DepGraph build
5. prune dry-run
6. interface invariant check
7. latency coverage annotation

失败输出：

```json
{
  "candidate_id": "dense_bev_backbone_path",
  "status": "rejected",
  "failed_at": "depgraph_build",
  "error": "...",
  "suggested_override": "..."
}
```

### 4.5 扩展到 9 模型统一 TracePlan

下一轮应逐步把以下模型也接入 full-model-derived `trace_plan`：

- CoDriving
- Pyramid lidar
- Pyramid camera
- V2X-ViT

验收原则：

- CoDriving 仍只能输出 scoped dense ResNet backbone envelope，不允许跨模型外推。
- Pyramid 仍保留 P-hub / grouped-conv context，不允许把 historical TRT 证据当新后端。
- V2X-ViT true TRT INT8 AP 仍 blocked / not done。
- attention / routing / fusion 未闭合时，不能输出 full-model separable。

### 4.6 分类器减少模型名硬编码

修改：

- `framework/stage1/coupling_predictor.py`
- `framework/stage1/model_classifier.py`
- `framework/stage1/calibrated_predictor.py`

方向：

- 更多依赖 `trace_plan` 字段：
  - `ckpt_status`
  - `trace_confidence`
  - `coverage_scope`
  - `manual_override_used`
  - `review_required`
  - `skipped_subgraphs[*].full_model_verdict_blocker`
  - `skipped_subgraphs[*].blocker_gate`
- 保留必要模型特例，但不能让模型名绕过 blocker。

分类原则：

- 缺 ckpt 或 random-init only：`SCAN_FAILED`
- fusion / attention / routing skipped 且未闭合：`CO_ACCELERATION_REQUIRED`
- 只有同 scope measured evidence 且无 blocker：才可 `SEPARABLE_ACCELERATION`
- `manual_override_used=True` 或 `trace_confidence=low` 默认需要 review，不允许 full-model separable。

## 5. 完成目标

下一轮完成标准如下：

### 5.1 功能完成

- 用户输入可抽象为：

```yaml
model_name: fcooper
config_path: /path/to/config.yaml
ckpt_path: /path/to/net.pth
```

- Stage1 能自动：
  - 加载 full model；
  - 扫描 module tree；
  - 自动识别 dense candidate path；
  - 自动排除 sparse / fusion / routing / postprocess；
  - 自动生成 wrapper candidate；
  - 执行 forward dry-run；
  - 执行 DepGraph build；
  - 执行 prune dry-run；
  - 输出 included / ignored / skipped / rejected manifest。

### 5.2 模型级验收

- F-Cooper：
  - `MaxFusion` 自动 skipped；
  - ckpt ok；
  - `backbone_m1` / `shrinker_m1` / heads included；
  - cls/reg/dir heads ignored；
  - 不输出 full-model separable。
- AttFuse：
  - `AttFusion` 自动 skipped；
  - ckpt ok；
  - `attention_fusion_coverage_anchor` 生效；
  - 不输出 full-model separable。
- Where2comm：
  - routing/fusion 自动 skipped；
  - 缺 ckpt 时 `architecture_only`；
  - 分类必须是 `SCAN_FAILED`；
  - 不允许 trained classification。
- V2VNet / DiscoNet：
  - routing/fusion 自动 skipped；
  - 缺 ckpt 时 `architecture_only`；
  - 分类必须是 `SCAN_FAILED`。
- CoDriving：
  - 只能 scoped separable；
  - 不允许跨模型外推。
- Pyramid lidar/camera：
  - P-hub / grouped-conv context 保留；
  - TRT 只作为 historical evidence。
- V2X-ViT：
  - transformer/routing/fake-quant gate 未闭合时仍 blocked；
  - true TRT INT8 AP 不得写成完成。

### 5.3 报告与 schema

分类报告必须包含：

- `trace_plan`
- `trace_confidence`
- `coverage_scope`
- `manual_override_used`
- `review_required`
- `included_modules`
- `ignored_layers`
- `skipped_subgraphs`
- `rejected_candidates`
- `no_overpromotion=True`

## 6. 测试命令

基础回归：

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
pytest -q \
  framework/tests/test_stage1_trace_boundary_detector_heter_baseline.py \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py \
  framework/tests/test_stage1_autoscan_extensions.py
```

新增测试建议：

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
pytest -q \
  framework/tests/test_stage1_dense_path_finder.py \
  framework/tests/test_stage1_wrapper_synthesizer.py \
  framework/tests/test_stage1_boundary_validator.py \
  framework/tests/test_stage1_trace_plan_nine_model_contract.py
```

CLI：

```bash
PYTHONPATH=${V2X_ROOT} \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

schema 检查：

```bash
PYTHONPATH=${V2X_ROOT} python - <<'PY'
import json
from pathlib import Path

p = Path("results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json")
d = json.loads(p.read_text())
assert d["schema"] == "stage1_model_classification_v1"
assert d["backend_policy"]["default_new_measurement_backend"] == "h800_tvm"
assert "trt" not in [x.lower() for x in d["backend_policy"]["allowed_new_measurement_backends"]]
assert d["no_overpromotion"] is True
assert len(d["models"]) == 9
for item in d["models"]:
    assert "trace_plan" in item
    assert "trace_confidence" in item
    assert "coverage_scope" in item
    assert "manual_override_used" in item
    assert "review_required" in item
    assert "classification" in item
    assert "scope" in item
    assert "blockers" in item
    assert "historical_evidence_sources" in item
print("stage1_trace_plan_classifier_ok")
PY
```

py_compile：

```bash
python -m py_compile \
  framework/stage1/trace_plan.py \
  framework/stage1/auto_trace.py \
  framework/stage1/graph_scan.py \
  framework/stage1/coupling_predictor.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage1/model_classifier.py \
  scripts/stage1_classify_models.py
```

## 7. 产物路径

代码：

- `framework/stage1/trace_plan.py`
- `framework/stage1/auto_trace.py`
- `framework/stage1/graph_scan.py`
- `framework/stage1/coupling_predictor.py`
- `framework/stage1/model_classifier.py`
- 可新增：
  - `framework/stage1/wrapper_synthesizer.py`
  - `framework/stage1/boundary_validator.py`

测试：

- `framework/tests/test_stage1_trace_boundary_detector_heter_baseline.py`
- 可新增：
  - `framework/tests/test_stage1_dense_path_finder.py`
  - `framework/tests/test_stage1_wrapper_synthesizer.py`
  - `framework/tests/test_stage1_boundary_validator.py`
  - `framework/tests/test_stage1_trace_plan_nine_model_contract.py`

结果：

- `results/autoscan_fcooper_partition.yaml`
- `results/autoscan_attfuse_partition.yaml`
- `results/autoscan_where2comm_partition.yaml`
- `results/autoscan_v2vnet_partition.yaml`
- `results/autoscan_disconet_partition.yaml`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`

文档：

- `multi_agent/methods/progress/HANDOFF_stage1_trace_boundary_autonomy_next_v1_zh.md`

## 8. /goal 启动指令

```text
/goal 阅读 multi_agent/methods/progress/HANDOFF_stage1_trace_boundary_autonomy_next_v1_zh.md，并基于该交接文档完成 Stage1 trace-boundary 自治化下一轮实现。严格约束：不要修改 github/stage1-model-scanner-aaai/，该目录由另一个窗口维护；新增实测后端统一为 H800 TVM/Relax/MetaSchedule；TRT 只能作为 historical evidence，不能作为新增 probe backend、分类器默认后端或新验收后端。目标是把第一阶段 TracePlan 原型推进为完整自动候选生成与验证流程：用户只提供 config + ckpt + model name，Stage1 加载 full model，ModuleTreeScanner 生成 module inventory，HeuristicTagger 标注 dense/sparse/fusion/routing/postprocess/head，DensePathFinder 生成多个 dense path candidate，WrapperSynthesizer 自动生成 wrapper candidate，BoundaryValidator 执行 forward dry-run、output shape sanity、DepGraph build、prune dry-run、interface invariant check，并将 included/ignored/skipped/rejected_candidates/review_required 写入 stage1_trace_plan_v1 manifest。刷新 F-Cooper、AttFuse、Where2comm、V2VNet、DiscoNet autoscan YAML，使其携带 full-model-derived trace_plan；扩展或规划接入 CoDriving、Pyramid lidar/camera、V2X-ViT 的统一 trace_plan；分类器必须基于 trace_confidence、coverage_scope、manual_override_used、ckpt_status、skipped_subgraphs 和 evidence level 保守三分类。验收必须证明：F-Cooper 自动识别 MaxFusion skipped 且 ckpt ok；AttFuse 自动识别 AttFusion skipped 且 ckpt ok；Where2comm/V2VNet/DiscoNet 自动识别 routing/fusion skipped，缺 ckpt 时必须为 architecture_only/SCAN_FAILED，不允许 trained classification；fusion/attention/routing 未闭合时不得输出 full-model separable；CoDriving 只能 scoped separable 且不得跨模型外推；Pyramid TRT 只作为 historical evidence；V2X-ViT true TRT INT8 AP 仍 blocked/not done。完成后运行 pytest、CLI schema 检查、py_compile，更新 results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json/.md 和中文 handoff/progress 文档。
```

## 9. 本轮完成记录

日期：2026-06-24

本轮已完成 `/goal` 中要求的下一阶段主线实现。仍然未修改 `github/stage1-model-scanner-aaai/`；该目录保持由另一个窗口维护。

### 9.1 已完成代码

- `framework/stage1/trace_plan.py`
  - 新增 `TraceCandidate`。
  - 新增 `DensePathFinder`：从 tagged module inventory 生成 dense path candidate。
  - 新增 `GeneratedTraceWrapper`。
  - 新增 `WrapperSynthesizer`：根据 selected candidate 生成可执行 wrapper。
  - 新增 `BoundaryValidator`：可独立执行 wrapper forward dry-run、output shape sanity、DepGraph build、prune dry-run、interface invariant check。
  - `TraceBoundaryDetector` 已改为调用 `DensePathFinder`，并在 `trace_plan` 中写入：
    - `candidates`
    - `selected_candidate`
    - `included_modules`
    - `ignored_layers`
    - `skipped_subgraphs`
    - `rejected_candidates`
    - `review_required`

- `framework/stage1/auto_trace.py`
  - HEAL `HeterModelBaseline` builder 已从旧 `_HeterBaselineTraceNet` 切换为 `WrapperSynthesizer().synthesize(...)`。
  - 旧 `_HeterBaselineTraceNet` 保留为回归/fallback 参考，但默认 HEAL path 已走自动 wrapper candidate。

- `framework/stage1/graph_scan.py`
  - `attach_runtime_validation()` 现在写入 BoundaryValidator 风格 validation 字段：
    - `boundary_validator=BoundaryValidator.graph_scan_integrated_v1`
    - `wrapper_forward_dryrun`
    - `output_shape_sanity`
    - `depgraph_build`
    - `prune_dryrun`
    - `interface_invariant_check`
    - `n_prunable_groups`
    - `out_shapes`

- `framework/stage1/model_classifier.py`
  - 模型记录顶层新增：
    - `trace_confidence`
    - `coverage_scope`
    - `manual_override_used`
    - `review_required`
    - `included_modules`
    - `ignored_layers`
    - `skipped_subgraphs`
    - `rejected_candidates`
    - `trace_plan`
  - Markdown 报告新增 `Trace Plan Summary` 表。

### 9.2 已刷新 autoscan manifest

已运行：

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
python scripts/autoscan_reproduce_check.py \
  --models fcooper,attfuse,where2comm,v2vnet,disconet \
  --device cpu \
  --save-yaml
```

已刷新：

- `results/autoscan_fcooper_partition.yaml`
- `results/autoscan_attfuse_partition.yaml`
- `results/autoscan_where2comm_partition.yaml`
- `results/autoscan_v2vnet_partition.yaml`
- `results/autoscan_disconet_partition.yaml`

自动扫描输出摘要：

| model | detector | wrapper | ckpt_status | included | ignored | skipped | validation |
|---|---|---|---|---|---|---|---|
| F-Cooper | `TraceBoundaryDetector.heter_baseline_v1` | `heter_baseline_dense_path` | `ok` | `backbone_m1`, `shrinker_m1`, heads | `cls_head`, `reg_head`, `dir_head` | `pillar_vfe`, `scatter`, `fusion_net(MaxFusion)` | S0/S1/S5 ok |
| AttFuse | `TraceBoundaryDetector.heter_baseline_v1` | `heter_baseline_dense_path` | `ok` | `backbone_m1`, `shrinker_m1`, heads | `cls_head`, `reg_head`, `dir_head` | `pillar_vfe`, `scatter`, `fusion_net(AttFusion)` | S0/S1/S5 ok |
| Where2comm | `TraceBoundaryDetector.heter_baseline_v1` | `heter_baseline_dense_path` | `missing_architecture_scan_only` | `backbone_m1`, `shrinker_m1`, heads | `cls_head`, `reg_head`, `dir_head` | `pillar_vfe`, `scatter`, `fusion_net(routing)` | S0/S1/S5 ok |
| V2VNet | `TraceBoundaryDetector.heter_baseline_v1` | `heter_baseline_dense_path` | `missing_architecture_scan_only` | `backbone_m1`, `shrinker_m1`, heads | `cls_head`, `reg_head`, `dir_head` | `pillar_vfe`, `scatter`, `fusion_net(routing)` | S0/S1/S5 ok |
| DiscoNet | `TraceBoundaryDetector.heter_baseline_v1` | `heter_baseline_dense_path` | `missing_architecture_scan_only` | `backbone_m1`, `shrinker_m1`, heads | `cls_head`, `reg_head`, `dir_head` | `pillar_vfe`, `scatter`, `fusion_net(routing)` | S0/S1/S5 ok |

所有五个 HEAL manifest 的 `selected_candidate.validation` 均包含：

- `boundary_validator=BoundaryValidator.graph_scan_integrated_v1`
- `wrapper_forward_dryrun=ok`
- `depgraph_build=ok`
- `prune_dryrun=ok`
- `output_shape_sanity=ok`

### 9.3 非 HEAL 模型的扫描接入状态

这一小节只记录自动扫描框架状态，不记录分类器结论。

| model | 当前扫描状态 | trace_plan 来源 | 下一步要做 |
|---|---|---|---|
| CoDriving | 已有旧 trace adapter，可生成 legacy schema plan | `legacy_trace_adapter_manifest_normalizer` | 写 CoDriving full-model loader + CoDriving wrapper candidate |
| Pyramid lidar | 已有旧 trace adapter，可生成 legacy schema plan | `legacy_trace_adapter_manifest_normalizer` | 写 Pyramid full-model detector，处理 P-hub / grouped-conv / pyramid heads |
| Pyramid camera | 已有旧 trace adapter，可生成 legacy schema plan | `legacy_trace_adapter_manifest_normalizer` | 写 camera/LSS skipped boundary + BEV entry detector |
| V2X-ViT | 已有旧 trace adapter，可生成 legacy schema plan | `legacy_trace_adapter_manifest_normalizer` | 写 transformer/routing skipped boundary + backbone wrapper candidate |

也就是说，本轮“自动网络扫描输出”真正完成的是 HEAL HeterBaseline 五模型；CoDriving / Pyramid / V2X-ViT 只是被纳入统一 `trace_plan` schema，尚未完成 full-model-derived detector。

### 9.4 新增测试

新增：

- `framework/tests/test_stage1_dense_path_finder.py`
- `framework/tests/test_stage1_wrapper_synthesizer.py`
- `framework/tests/test_stage1_boundary_validator.py`
- `framework/tests/test_stage1_trace_plan_nine_model_contract.py`

覆盖：

- `DensePathFinder` 能生成 HEAL dense path candidate。
- `WrapperSynthesizer` 能生成可执行 wrapper。
- `BoundaryValidator` 能真实执行 forward / DepGraph / prune dry-run。
- 九模型报告都暴露 `trace_plan` contract。
- HEAL 五模型使用 `TraceBoundaryDetector.heter_baseline_v1`。
- Where2comm / V2VNet / DiscoNet 的扫描输出明确 `ckpt_status=missing_architecture_scan_only`。
- CoDriving / Pyramid / V2X-ViT 的扫描输出仍是 legacy normalized plan，不伪装成 full-model-derived detector。

### 9.5 已运行验证

完整 pytest：

```bash
PATH=${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=${V2X_ROOT} \
pytest -q \
  framework/tests/test_stage1_dense_path_finder.py \
  framework/tests/test_stage1_wrapper_synthesizer.py \
  framework/tests/test_stage1_boundary_validator.py \
  framework/tests/test_stage1_trace_plan_nine_model_contract.py \
  framework/tests/test_stage1_trace_boundary_detector_heter_baseline.py \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py \
  framework/tests/test_stage1_autoscan_extensions.py
```

结果：`36 passed`。

CLI / schema：

```bash
PYTHONPATH=${V2X_ROOT} \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

额外 schema 检查已通过：

- `schema == stage1_model_classification_v1`
- `backend_policy.default_new_measurement_backend == h800_tvm`
- `allowed_new_measurement_backends` 不包含 TRT
- `no_overpromotion == True`
- `len(models) == 9`
- 每个模型包含：
  - `trace_plan`
  - `trace_confidence`
  - `coverage_scope`
  - `manual_override_used`
  - `review_required`
  - `included_modules`
  - `ignored_layers`
  - `skipped_subgraphs`
  - `rejected_candidates`

说明：以上 CLI/schema 检查属于分类器报告合同验证；自动网络扫描输出本身以上面的 autoscan YAML 与 `trace_plan.selected_candidate.validation` 为准。

py_compile 已通过：

```bash
python -m py_compile \
  framework/stage1/trace_plan.py \
  framework/stage1/auto_trace.py \
  framework/stage1/graph_scan.py \
  framework/stage1/coupling_predictor.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage1/model_classifier.py \
  scripts/stage1_classify_models.py
```

### 9.6 剩余限制

当前已经完成 HEAL HeterBaseline 的自动候选生成、自动 wrapper 和验证闭环；但仍不是任意模型的完全自动扫描器。

剩余限制：

- CoDriving / Pyramid / V2X-ViT 还没有 full-model-derived detector。
- full model builder discovery 仍不是通用机制；外部模型仍需要 loader/plugin。
- fusion / attention / routing 仍是自动识别为 skipped blocker，没有实现 full-model trace closure。
- `WrapperSynthesizer` 当前第一版主要支持 BEV dense path：backbone dict I/O -> shrinker -> heads；更复杂 forward 还需要扩展。
