# HANDOFF: Stage1 模型分类器计划收口判断与下一阶段交接

日期：2026-06-24  
原计划：`multi_agent/methods/design/stage1-model-predict/stage1_model_classifier_implementation_plan_v1.md`

## 0. 收口判断

**结论：`stage1_model_classifier_implementation_plan_v1.md` 按工程验收标准已经正式收口。**

收口依据不是“所有科学问题都解决”，而是该 plan 明确要求的工程目标已经达成并通过当前验收：

- 建立了端到端 `model_classifier` 契约。
- 新增实测后端统一为 H800 TVM / Relax / MetaSchedule。
- TRT 仅作为 historical evidence，不进入新增测量后端、默认后端或 `measured_h800_tvm`。
- manifest predictor fields 已补齐 typed skip、B1/B1-search feature、trace-net latency coverage。
- S2/S2.5/S3/S4 既有证据已进入 calibrated / classifier 报告链。
- bridge/search 中误导性的 separable 文案已替换。
- 默认 9 模型分类报告已生成，且 `no_overpromotion=True`。
- 对 CoDriving 外推、Pyramid TRT 误用、V2X-ViT true TRT blocked、Where2comm/V2VNet/DiscoNet architecture-only 等边界均保守处理。

**不应把下一阶段研究问题误判为本计划未收口。** 自动 trace boundary 的进一步产品化、attention/fusion/routing 子图闭合、AP/HV 三臂验证、更多硬件实测、`model_classifier` 显式读取 `stage1_coupling_predictions_v1.json`，都属于下一阶段增强，不阻塞本 plan closure。

## 1. 当前关键产物

### 1.1 代码入口

- Safe predictor v0：`framework/stage1/coupling_predictor.py`
- Calibrated predictor v1：`framework/stage1/calibrated_predictor.py`
- End-to-end model classifier：`framework/stage1/model_classifier.py`
- Classifier CLI：`scripts/stage1_classify_models.py`
- Legacy CLI compatibility：
  - `scripts/stage1_predict_coupling.py`
  - `scripts/stage1_predict_coupling_v1.py`

### 1.2 Trace boundary / manifest 相关

- 自动 trace plan：`framework/stage1/trace_plan.py`
- Auto adapter：`framework/stage1/auto_trace.py`
- DepGraph scan：`framework/stage1/graph_scan.py`
- Latency coverage：`framework/stage1/latency_profile.py`
- Legacy adapters：`framework/stage1/adapters.py`

当前扫描流程为：

```text
用户提供 config + checkpoint + model name + loader
        ↓
Stage1 加载完整模型
        ↓
TraceBoundaryDetector 扫描模块树并生成 dense candidate
        ↓
WrapperSynthesizer 合成 wrapper candidate
        ↓
graph_scan 执行 forward / DepGraph / B1-B2-D / pruning dry-run
        ↓
manifest 写入 trace_plan + validation + typed skipped_subgraphs
        ↓
predictor / classifier 保守分类
```

### 1.3 报告产物

- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`
- `results/stage1_model_predict/stage1_coupling_predictions_v1.json`
- `results/stage1_model_predict/stage1_coupling_predictions_v1.md`
- `results/stage1_model_predict/calibrated_predictor_report_v1.json`
- `results/stage1_model_predict/calibrated_predictor_report_v1.md`
- `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`
- `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md`

### 1.4 开源子仓库

`github/stage1-model-scanner-aaai/` 已按当前框架更新过一版，包括 README、中文 README、新增模型 adapter 教程、`trace_plan.py`、`auto_trace.py`、`graph_scan.py`、`model_classifier.py` 等。用户已说明后续该目录由另一个窗口维护；本窗口下一阶段如无明确要求，不应再修改该目录。

## 2. 当前模型分类结果

默认 9 模型来自 `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`：

| model | acceleration_class | 当前解释 |
|---|---|---|
| `codriving` | `SEPARABLE_ACCELERATION` | 仅限 CoDriving measured dense ResNet backbone envelope；不跨模型外推。 |
| `fcooper` | `CO_ACCELERATION_REQUIRED` | MaxFusion 局部 bounded，但 H800 TVM S2/S4 表明 dense anchors 至少需要 pair-level schedule calibration。 |
| `attfuse` | `CO_ACCELERATION_REQUIRED` | attention/fusion 未完整 trace，dense scan 不能提升为 full-model separability。 |
| `v2xvit` | `CO_ACCELERATION_REQUIRED` | fake-quant / structural routing gate 仍在，true TRT INT8 AP 未闭合。 |
| `pyramid_lidar` | `CO_ACCELERATION_REQUIRED` | P-hub/grouped-conv context + historical TRT evidence；不能作为新增测量后端或 separability proof。 |
| `pyramid_camera` | `CO_ACCELERATION_REQUIRED` | 同 Pyramid lidar，且 checkpoint 口径为 OPV2V-only/no DAIR。 |
| `where2comm` | `SCAN_FAILED` | `missing_architecture_scan_only`，不能写成 trained checkpoint scan。 |
| `v2vnet` | `SCAN_FAILED` | `missing_architecture_scan_only`，fusion timing sidecar 不能代表 trained evidence。 |
| `disconet` | `SCAN_FAILED` | `missing_architecture_scan_only`，DAIR single-agent sidecar 不能代表多 agent fusion。 |

根报告状态：

- `schema = stage1_model_classification_v1`
- `models = 9`
- `backend_policy.default_new_measurement_backend = h800_tvm`
- `backend_policy.allowed_new_measurement_backends = ["h800_tvm"]`
- `no_overpromotion = True`

## 3. 原计划逐项收口判断

| Task | 状态 | 判断 |
|---|---|---|
| Task 1: Classifier contract and backend policy | 已收口 | `model_classifier.py` 和 contract tests 存在；backend policy 为 H800 TVM；TRT historical-only。 |
| Task 2: Manifest predictor fields | 已收口 | typed `skipped_subgraphs`、legacy `skipped_modules`、B1 feature、B1 search rollup、trace-net coverage 已由测试覆盖。 |
| Task 3: H800 TVM evidence ingestion | 已收口 | S2/S4 latency 标为 H800 TVM；Pyramid TRT historical-only；V2X-ViT true TRT INT8/AP blocked。未单独创建 `h800_tvm_evidence.py`，因为当前代码规模未要求拆分。 |
| Task 4: Replace separability wording | 已收口 | bridge/search 旧文案 `SEPARABLE (全旋钮可串行)` 和 `No cliff (separable...)` 已验收为不存在。 |
| Task 5: End-to-end CLI and reports | 已收口 | `scripts/stage1_classify_models.py` 可生成 JSON/MD；schema check 通过。 |
| Task 6: Validation and handoff docs | 已收口 | validation doc 和 safe predictor handoff 均存在；本文件补充最终 plan closure 判断。 |
| Task 7: Full verification | 已收口 | 当前轮已重新跑计划中的 pytest、CLI、schema、bridge wording、py_compile。 |

## 4. 本轮实际验收结果

已在 2026-06-24 当前工作区重新执行：

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py
```

结果：`23 passed in 25.39s`

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q framework/tests/test_stage1_autoscan_extensions.py
```

结果：`4 passed`，仅有 matplotlib/pyparsing deprecation warnings。

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

结果：`stage1_model_classifier_ok models=9 ...`

Schema check：

```text
stage1_model_classifier_ok
```

Bridge wording acceptance：

```text
bridge_wording_ok
```

Py compile：

```bash
python -m py_compile \
  framework/stage1/coupling_predictor.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage1/model_classifier.py \
  scripts/stage1_classify_models.py
```

结果：exit 0。

目录放置检查：

- `multi_agent/methods/design/stage1-model-predict/` 顶层未发现非 Markdown 文件。
- model classifier 结果位于 `results/stage1_model_predict/model_classifier/`。

## 5. 仍需保持的证据边界

这些不是本 plan 的失败项，而是后续工作必须继续遵守的边界：

1. **Stage1 validation 不是 full-model validation。**  
   当前 validation 只证明 selected dense candidate 可 forward、可建 DepGraph、可 pruning dry-run。它不证明 skipped sparse/fusion/attention/routing/custom 子图被覆盖。

2. **`view_latency.coverage` 是 trace-net coverage。**  
   不能写成 full-model latency coverage；skipped subgraphs 必须 separately gated。

3. **S4 是 latency-only。**  
   S4 支持“local-only 不安全、至少需要 pair-level schedule calibration”的 latency 结论；不支持 AP/HV，也不支持 full-model irreducible coupling。

4. **TRT 是 historical evidence。**  
   Pyramid TRT AP/latency 只进入 `historical_evidence_sources`；不能作为新增 probe backend、默认 classifier backend 或新验收后端。

5. **CoDriving 不外推。**  
   CoDriving 是 scoped separable，仅限已测 dense ResNet backbone envelope。

6. **缺 checkpoint 的模型是扫描失败。**  
   Where2comm / V2VNet / DiscoNet 当前是 `missing_architecture_scan_only`，不能写成 trained checkpoint classification。

7. **`groups=1` 和 no-cliff 不是模型级 separability proof。**  
   它们最多描述 traced dense space 中没有某类 INT8 buildability cliff。

## 6. 下一阶段建议

下一阶段不应继续围绕本 plan 做“补收口”；应启动新目标。建议优先级如下。

### 6.1 自动 trace boundary 产品化

目标：把当前 `TraceBoundaryDetector` 从现有模型族推广为更稳健的新模型接入流程。

建议任务：

- 明确 detector/plugin API。
- 将用户需要提供的信息压缩为 config、checkpoint、model name、loader 和可选 input-shape hint。
- 让框架自动输出 candidate ranking、rejected reason、manual override suggestion。
- 增加“用户审核 manifest”教程和失败样例。
- 对新模型增加 no-checkpoint / random-init / full-checkpoint 三种状态的区别测试。

### 6.2 Attention/fusion/routing 证据闭合

目标：减少目前阻断 full-model conclusion 的 skipped subgraph。

建议任务：

- AttFuse attention/fusion trace 或独立 latency/AP gate。
- V2X-ViT HMSA/MSwin / routing gate 的真实后端测量。
- Where2comm/V2VNet/DiscoNet trained checkpoint 配置补齐后重跑 scan。
- 区分 fusion shape-bound、latency-bound、AP-bound 三种证据等级。

### 6.3 Classifier 工程边界清理

当前 `model_classifier` 概念上消费 calibrated evidence policy，但实现上仍直接调用 v0 predictor 并读取 evidence directory。下一阶段可改成：

```text
Stage1 manifest
    ↓
coupling_predictor v0
    ↓
calibrated_predictor v1
    ↓
model_classifier consumes stage1_coupling_predictions_v1.json explicitly
```

这样能让“证据校准层”和“最终三分类层”边界更清楚。

### 6.4 新硬件接入

目标：把 3090/4090/Orin/H800 等硬件 capability 与 measured evidence 分层做清楚。

建议任务：

- capability YAML 只做离线结构合法性。
- 新硬件 measured latency/AP 必须接入对应硬件运行 probe。
- 不允许用静态 YAML 生成 measured claim。
- 若引入 H800 之外的新测量后端，必须更新 backend policy 和测试。

## 7. 下一阶段 `/goal` 建议

```text
/goal 基于当前已收口的 Stage1 model classifier，实现下一阶段 Stage1 trace-boundary autonomy 产品化。严格约束：不要重新打开 stage1_model_classifier_implementation_plan_v1 的收口项；新增工作只围绕自动 trace boundary、detector/plugin API、用户审核 manifest、manual override fallback 和新模型接入教程。目标流程是：用户提供 config + checkpoint + model name + loader，Stage1 自动扫描完整模型模块树，生成 dense candidate、ignored、skipped、rejected、validation 和 next override suggestion；分类器继续保守消费 manifest，不因 dense candidate 通过而输出 full-model separability。验收必须包含：新 detector/plugin contract tests、BoundaryValidator validation tests、至少一个模拟新模型失败案例、README/教程更新，以及 no_overpromotion=True。
```

## 8. 清空上下文前的最小恢复清单

下一窗口如果只读 5 个文件即可恢复状态，建议按顺序读：

1. `multi_agent/methods/progress/HANDOFF_stage1_model_classifier_plan_closure_v1_zh.md`
2. `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md`
3. `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`
4. `multi_agent/paper/stage1_method_zh_v2.md`
5. `github/stage1-model-scanner-aaai/README.zh-CN.md`（若下一步涉及开源子仓库）

## 9. 工作区注意事项

当前主仓库工作区不是干净状态，存在大量实验产物、未跟踪文件和此前已有修改。不要在下一阶段用 destructive git 命令回滚工作区。若只需要继续 Stage1 trace-boundary autonomy，应优先限定 diff 范围到：

- `framework/stage1/trace_plan.py`
- `framework/stage1/auto_trace.py`
- `framework/stage1/graph_scan.py`
- `framework/tests/test_stage1_*`
- `docs/` 或 `multi_agent/methods/progress/`

`github/stage1-model-scanner-aaai/` 后续由另一个窗口维护；除非用户明确要求同步，否则不要修改该目录。
