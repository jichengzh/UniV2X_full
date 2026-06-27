# 交接文档：Stage1 模型分类器与 Trace/硬件自动化

日期：2026-06-24

## 0. 当前状态

Stage1 端到端模型分类器 v1 已实现并生成产物。当前分类器有两层输出：

- `acceleration_class`：主分类字段，三分类。
- `classification`：细粒度证据 verdict，保留用于兼容旧报告和审计。

三分类定义如下：

| enum | 中文标签 | 含义 |
|---|---|---|
| `CO_ACCELERATION_REQUIRED` | 需要协同加速 | 该模型需要联合/协同考虑剪枝、量化、调度、路由或 fusion/attention gate。 |
| `SEPARABLE_ACCELERATION` | 可分离加速 | 只有在证据范围明确且没有 blocker 时才允许。当前只适用于 CoDriving 的 scoped dense backbone envelope。 |
| `SCAN_FAILED` | 扫描失败 | 没有形成有效 trained-model classification；通常是缺 ckpt、只有 architecture-only scan 或证据不足。 |

当前产物：

- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`

主要实现：

- `framework/stage1/model_classifier.py`
- `scripts/stage1_classify_models.py`

相关验证/说明文档：

- `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`
- `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md`

## 1. 当前 9 个模型分类结果

| 模型 | 三分类 | 细粒度 verdict | scope |
|---|---|---|---|
| `codriving` | `SEPARABLE_ACCELERATION` / 可分离加速 | `SCOPED_MEASURED_NEGATIVE_ANCHOR` | `codriving_measured_resnet_backbone_envelope` |
| `fcooper` | `CO_ACCELERATION_REQUIRED` / 需要协同加速 | `PAIR_CALIBRATION_REQUIRED_FULL_MODEL_GATED` | `full_model_schedule_coupling_detected` |
| `attfuse` | `CO_ACCELERATION_REQUIRED` / 需要协同加速 | `FUSION_UNCOVERED_STATIC_DENSE_PROMOTION_BLOCKED` | `traced_dense_subgraph` |
| `v2xvit` | `CO_ACCELERATION_REQUIRED` / 需要协同加速 | `PAIR_OR_JOINT_GATE_FAKE_QUANT_TRUE_TRT_BLOCKED` | `full_model_gate_blocked` |
| `pyramid_lidar` | `CO_ACCELERATION_REQUIRED` / 需要协同加速 | `P_HUB_CONTEXT_WITH_HISTORICAL_TRT_Q_AP_BOUND` | `full_model_p_hub_context` |
| `pyramid_camera` | `CO_ACCELERATION_REQUIRED` / 需要协同加速 | `P_HUB_CONTEXT_WITH_HISTORICAL_TRT_Q_AP_BOUND` | `full_model_p_hub_context` |
| `where2comm` | `SCAN_FAILED` / 扫描失败 | `ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN` | `traced_dense_subgraph` |
| `v2vnet` | `SCAN_FAILED` / 扫描失败 | `ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN` | `traced_dense_subgraph` |
| `disconet` | `SCAN_FAILED` / 扫描失败 | `ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN` | `traced_dense_subgraph` |

注意：这里的 `SCAN_FAILED` 不是 Python 扫描程序崩溃，而是没有形成有效的 trained checkpoint 模型分类。Where2comm / V2VNet / DiscoNet 当前只有 architecture-only dense-core scan 和 random-init sidecar，因此不能写成 trained model classification，且 `measured_h800_tvm` 必须为空。

## 2. 后端与证据口径

当前后端政策：

- 新增实测后端只允许 H800 TVM / Relax / MetaSchedule。
- `backend_policy.allowed_new_measurement_backends == ["h800_tvm"]`。
- TRT 只能作为 historical evidence。
- TRT 不能出现在 `measured_h800_tvm`。
- TRT 不能作为 classifier 默认后端，也不能作为新增 probe backend。

已经完成的口径修正：

- Pyramid 的 TRT AP/latency 证据只标记为 `historical_trt_evidence_plus_p_hub_context`。
- V2X-ViT true TRT INT8 AP 仍然是 blocked / not done。
- `search_three_arm.py` 不再默认使用 4090 TRT ratio 作为 active H800 latency fallback；historical 4090 TRT ratio fallback 默认禁用。

## 3. 当前硬件扫描的真实能力

当前 Stage1 的 `hardware_scan` 不是完整自动硬件 benchmark 系统，而是本地硬件 capability loader。

相关文件：

- `framework/stage1/hardware_scan.py`
- `framework/capability_schema.py`
- `configs/hardware/rtx4090.yaml`
- `configs/hardware/orin_agx.yaml`
- `configs/hardware/schema.yaml`

它读取本地 YAML，并归一化以下字段：

- 硬件名称、架构、SM；
- GPU / DLA 是否存在；
- INT8 / FP16 alignment；
- INT8 pack factor；
- legal bits / quant granularity；
- DLA op whitelist；
- memory budget。

它当前不做：

- 自动识别任意新 GPU；
- 自动跑 TVM/TRT buildability probe；
- 自动测 latency / AP；
- 在没有目标硬件实测证据时保证新硬件分类正确。

离线能力边界：

- 如果硬件 YAML、模型代码、checkpoint、依赖环境、evidence 文件都在本地，系统可以完全断网运行。
- 如果没有目标硬件，只能做 static capability-based scan。
- 对 RTX 3090 这类新硬件，要得到可靠 measured classification，需要接入 3090 并跑本地 probe，或者提供已经离线保存的 3090 evidence bundle。

理想硬件自动化架构：

```text
HardwareRegistry
  本地硬件模板库
HardwareDiscoverer
  nvidia-smi / torch.cuda / TVM / trtexec / lscpu 本机发现
CapabilityResolver
  registry 模板 + 本机发现 + schema 校验
HardwareProbeRunner
  本地 micro-probe，验证 buildability 和 latency
EvidenceStore
  保存 backend、日期、driver、commit、shape、provenance
Stage1GraphScan
  在 effective hardware capability 下扫描模型结构
ModelClassifier
  输出三分类、证据等级、blocker、next probe/gate
```

## 4. 当前 TraceAdapter 的真实状态

当前 trace 系统有两代：

- 手写 adapter：`framework/stage1/adapters.py`
- 半自动 adapter：`framework/stage1/auto_trace.py`

当前 `TraceAdapter` 仍然承担了比最终理想状态更多的职责，包括：

- 模型加载路径；
- checkpoint 路径和状态；
- trace-ready wrapper；
- dummy BEV input shape；
- skipped sparse/fusion/attention/custom modules；
- ignored output/interface layers；
- semantic bucket 映射。

当前一些具体例子：

- CoDriving：从 post-scatter dense BEV 开始 trace，跳过 `pillar_vfe`、`scatter`、`fusion_net`。
- V2X-ViT：跳过 sparse encoder 和 V2XTransformer fusion，并冻结 shrinker 输出层，保持 fusion 输入通道稳定。
- Pyramid Camera：跳过 LiftSplatShoot / QuickCumsum，并冻结 aligner/heads，因为 trace 和结构化剪枝不稳定。
- F-Cooper / AttFuse / Where2comm / V2VNet / DiscoNet：使用 `_HeterBaselineTraceNet` 通用 wrapper，但仍需要 registry 条目提供 config、ckpt、BEV shape、skipped modules 和 trace note。

因此，当前实现还不是“输入模型后全自动发现 dense core”的系统。它是一个保守的混合方案：

```text
model-specific loader / thin adapter
  -> trace-ready dense-core wrapper
  -> DepGraph + manifest generation
```

## 5. 期望的 Trace 自动化方向

最终应把 `TraceAdapter` 压缩成最小模型加载器 / input spec provider。dense-core 边界发现应交给自动组件：

```text
Full model / config / ckpt
  -> ModelIntrospector
  -> SampleInputResolver
  -> TraceAttempt
  -> GraphSegmenter
  -> BoundaryValidator
  -> ManifestWriter
```

各组件职责：

| 组件 | 职责 |
|---|---|
| `ModelIntrospector` | 读取 module tree、config、checkpoint 状态、forward signature。 |
| `SampleInputResolver` | 从 config、数据样例或用户 input spec 生成 sample input。 |
| `TraceAttempt` | 尝试 torch.fx / torch.export / forward hook trace。 |
| `GraphSegmenter` | 将 full model 切分为 traceable dense core、sparse preprocess、fusion/attention、custom ops、heads/interface layers。 |
| `BoundaryValidator` | forward sanity、DepGraph build、prune dry-run、output/interface shape check。 |
| `ManifestWriter` | 输出 `trace.skipped_subgraphs`、B1/B2/D views、coverage、feature blocks。 |

理想输出是机器可读 trace plan，例如：

```json
{
  "trace_entry": "post_scatter_bev",
  "trace_modules": ["backbone_m1", "shrinker_m1", "cls_head", "reg_head"],
  "ignored_layers": ["cls_head", "reg_head"],
  "skipped_subgraphs": [
    {"name": "pillar_vfe", "type": "sparse_or_geometry_preprocess"},
    {"name": "fusion_net", "type": "attention_or_routing_fusion"}
  ],
  "input_shape": [1, 64, 512, 512],
  "confidence": "auto_detected_with_forward_sanity"
}
```

## 6. 核心设计原则

分类器不能把静态知识伪装成实测证据。

对于新硬件和新模型，框架应该优先拒判/降级，而不是过度宣称：

- `STATIC_ONLY`
- `MEASURED_ON_TARGET_HARDWARE`
- `HISTORICAL_EVIDENCE`
- `ARCHITECTURE_ONLY`
- `UNKNOWN_NEEDS_PROBE`

只有 `MEASURED_ON_TARGET_HARDWARE` 才能支撑强的目标硬件结论。

## 7. 建议的下一步目标

建议下一轮目标：

```text
把 Stage1 trace boundary 和 hardware capability 处理重构为更自治的离线系统。

实现：
1. HardwareRegistry 和 CapabilityResolver，读取 configs/hardware/*.yaml。
2. 可选 HardwareDiscoverer，使用本地 nvidia-smi / torch.cuda / TVM 可用性，不联网。
3. HardwareProbeRunner 接口，支持 no-op/static mode 和 measured mode。
4. TracePlan schema，把模型加载和 trace-boundary 决策拆开。
5. HEAL HeterModelBaseline 的 TraceBoundaryDetector 原型：
   - 自动识别 backbone/shrinker/heads；
   - 自动将 sparse/fusion/attention 模块标记为 skipped_subgraphs；
   - 尽可能从 config 推断 BEV shape；
   - 用 forward sanity 和 DepGraph build 验证。
6. 更新 model_classifier，消费 trace plan confidence 和 evidence grade，而不是依赖模型名分支。
```

建议优先阅读文件：

- `framework/stage1/hardware_scan.py`
- `framework/capability_schema.py`
- `configs/hardware/*.yaml`
- `framework/stage1/adapters.py`
- `framework/stage1/auto_trace.py`
- `framework/stage1/graph_scan.py`
- `framework/stage1/model_classifier.py`
- `framework/tests/test_stage1_model_classifier_contract.py`
- `framework/tests/test_stage1_manifest_predictor_fields.py`

建议验证命令：

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_stage1_autoscan_extensions.py
```

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

## 8. 最近验证状态

集成三分类输出后，最近一次重点验证通过：

```text
23 passed
```

命令：

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py
```

CLI 生成通过：

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

schema 检查已通过：

- `acceleration_class_labels` 存在；
- CoDriving = `SEPARABLE_ACCELERATION`；
- F-Cooper / AttFuse / V2X-ViT / Pyramid = `CO_ACCELERATION_REQUIRED`；
- Where2comm / V2VNet / DiscoNet = `SCAN_FAILED`；
- A2 模型 `measured_h800_tvm == {}`。

