# HANDOFF: Stage1 Model Classifier + Trace/Hardware Autonomy

Date: 2026-06-24

## 0. Current State

Stage1 model classifier v1 is implemented and generated. It now has two layers of classification:

- `acceleration_class`: coarse three-class output used as the main model classification.
- `classification`: fine-grained evidence verdict kept for compatibility and auditability.

The three coarse classes are:

| enum | label | intended meaning |
|---|---|---|
| `CO_ACCELERATION_REQUIRED` | 需要协同加速 | The scanned/evidenced model needs coupled or at least pair-level acceleration decisions. |
| `SEPARABLE_ACCELERATION` | 可分离加速 | Only supported when evidence is scoped and no blocker is triggered. Currently only CoDriving's measured dense backbone envelope. |
| `SCAN_FAILED` | 扫描失败 | No valid trained-model classification; current evidence is architecture-only, missing ckpt, or insufficient. |

Generated artifacts:

- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`

Implementation:

- `framework/stage1/model_classifier.py`
- `scripts/stage1_classify_models.py`

Validation docs:

- `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`
- `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md`

## 1. Current Nine-Model Classification

| model | acceleration_class | fine-grained verdict | scope |
|---|---|---|---|
| `codriving` | `SEPARABLE_ACCELERATION` | `SCOPED_MEASURED_NEGATIVE_ANCHOR` | `codriving_measured_resnet_backbone_envelope` |
| `fcooper` | `CO_ACCELERATION_REQUIRED` | `PAIR_CALIBRATION_REQUIRED_FULL_MODEL_GATED` | `full_model_schedule_coupling_detected` |
| `attfuse` | `CO_ACCELERATION_REQUIRED` | `FUSION_UNCOVERED_STATIC_DENSE_PROMOTION_BLOCKED` | `traced_dense_subgraph` |
| `v2xvit` | `CO_ACCELERATION_REQUIRED` | `PAIR_OR_JOINT_GATE_FAKE_QUANT_TRUE_TRT_BLOCKED` | `full_model_gate_blocked` |
| `pyramid_lidar` | `CO_ACCELERATION_REQUIRED` | `P_HUB_CONTEXT_WITH_HISTORICAL_TRT_Q_AP_BOUND` | `full_model_p_hub_context` |
| `pyramid_camera` | `CO_ACCELERATION_REQUIRED` | `P_HUB_CONTEXT_WITH_HISTORICAL_TRT_Q_AP_BOUND` | `full_model_p_hub_context` |
| `where2comm` | `SCAN_FAILED` | `ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN` | `traced_dense_subgraph` |
| `v2vnet` | `SCAN_FAILED` | `ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN` | `traced_dense_subgraph` |
| `disconet` | `SCAN_FAILED` | `ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN` | `traced_dense_subgraph` |

Important boundary: `SCAN_FAILED` here does not mean the Python scan crashed. It means the current system does not have a valid trained-checkpoint model classification. For Where2comm / V2VNet / DiscoNet, the evidence is architecture-only dense-core scan plus random-init sidecar, and `measured_h800_tvm` must remain empty.

## 2. Backend Evidence Policy

New measured backend policy:

- New measurement backend: H800 TVM / Relax / MetaSchedule only.
- `backend_policy.allowed_new_measurement_backends == ["h800_tvm"]`.
- TRT is historical evidence only.
- TRT must not appear in `measured_h800_tvm`.
- TRT must not be used as classifier default backend or new probe backend.

Current policy corrections already made:

- Pyramid TRT AP/latency is labeled `historical_trt_evidence_plus_p_hub_context`.
- V2X-ViT true TRT INT8 AP remains blocked / not done.
- Search code no longer defaults to active `H800_TVM x 4090_Q_ratio` style fallback; historical 4090 TRT ratio fallback is disabled by default.

## 3. Hardware Scan Reality

Current Stage1 hardware scan is not a full automatic hardware benchmark system. It is primarily a local hardware capability loader:

- Code: `framework/stage1/hardware_scan.py`
- Schema: `framework/capability_schema.py`
- Local hardware YAML:
  - `configs/hardware/rtx4090.yaml`
  - `configs/hardware/orin_agx.yaml`
  - `configs/hardware/schema.yaml`

It reads local YAML and exposes normalized fields:

- hardware name / arch / SM
- GPU / DLA availability
- INT8 and FP16 alignment
- INT8 pack factor
- legal bits and granularity
- DLA op whitelist
- memory budget

It does not currently:

- discover an arbitrary new GPU by itself;
- run TVM/TRT buildability probes automatically;
- measure latency/AP;
- guarantee correct performance classification for a new hardware target without measured evidence.

Offline behavior:

- The system can run fully offline if the hardware YAML, model code, checkpoints, dependencies, and evidence files are local.
- Without target hardware, only static capability-based scan is possible.
- For a new target such as RTX 3090, correct measured classification needs either an attached 3090 and local probe runner, or a pre-existing offline evidence bundle.

Desired future design:

```text
HardwareRegistry
  local known hardware templates
HardwareDiscoverer
  nvidia-smi / torch / tvm / trtexec / lscpu detection
CapabilityResolver
  registry template + local discovery + schema validation
HardwareProbeRunner
  local micro-probes for buildability and latency
EvidenceStore
  backend, date, driver, commit, shape, provenance
Stage1GraphScan
  model structure scan under effective hardware capability
ModelClassifier
  three-class output with evidence grade and abstention policy
```

## 4. TraceAdapter Reality

Current trace system has two generations:

- Manual adapters: `framework/stage1/adapters.py`
- Semi-automatic adapters: `framework/stage1/auto_trace.py`

Current `TraceAdapter` still does more than the final design should. It currently defines or provides:

- model loading path;
- checkpoint path and status;
- trace-ready wrapper;
- dummy BEV input shape;
- skipped sparse/fusion/attention/custom modules;
- ignored output/interface layers;
- semantic bucket mapping.

Examples of current manual or semi-manual choices:

- CoDriving starts from post-scatter dense BEV and skips `pillar_vfe`, `scatter`, `fusion_net`.
- V2X-ViT skips sparse encoder and V2XTransformer fusion, and freezes the shrinker output layer to keep fusion input channels stable.
- Pyramid Camera skips LiftSplatShoot / QuickCumsum and freezes aligner/heads due trace and pruning instability.
- F-Cooper / AttFuse / Where2comm / V2VNet / DiscoNet use `_HeterBaselineTraceNet`, but still require registry entries for config, ckpt, BEV shape, skipped modules, and trace notes.

This means the current implementation is not yet the ideal "input model and automatically discover dense core" system. It is a conservative hybrid:

```text
model-specific loader / thin adapter
  -> trace-ready dense-core wrapper
  -> DepGraph and manifest generation
```

## 5. Desired Trace Automation

The intended final architecture should reduce `TraceAdapter` to a minimal model loader / input-spec provider. Dense-core boundary discovery should move into automatic components:

```text
Full model / config / ckpt
  -> ModelIntrospector
  -> SampleInputResolver
  -> TraceAttempt
  -> GraphSegmenter
  -> BoundaryValidator
  -> ManifestWriter
```

Component responsibilities:

| component | responsibility |
|---|---|
| `ModelIntrospector` | Read module tree, config, checkpoint status, forward signature. |
| `SampleInputResolver` | Generate sample input from config, dataset sample, or user-provided input spec. |
| `TraceAttempt` | Try torch.fx / torch.export / forward hook trace. |
| `GraphSegmenter` | Split full model into traceable dense core, sparse preprocess, fusion/attention, custom ops, heads/interface layers. |
| `BoundaryValidator` | Run forward sanity, DepGraph build, pruning dry-run, output/interface shape checks. |
| `ManifestWriter` | Emit `trace.skipped_subgraphs`, B1/B2/D views, coverage, feature blocks. |

The output should be a machine-readable trace plan:

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

## 6. Key Design Principle

The classifier must not pretend that static knowledge is measured evidence.

For both new hardware and new models, the framework should prefer abstention over overclaiming:

- `STATIC_ONLY`
- `MEASURED_ON_TARGET_HARDWARE`
- `HISTORICAL_EVIDENCE`
- `ARCHITECTURE_ONLY`
- `UNKNOWN_NEEDS_PROBE`

Only `MEASURED_ON_TARGET_HARDWARE` can support strong target-hardware conclusions.

## 7. Immediate Next Work

Recommended next goal:

```text
Refactor Stage1 trace boundary and hardware capability handling toward an autonomous offline system.

Implement:
1. HardwareRegistry and CapabilityResolver over configs/hardware/*.yaml.
2. Optional HardwareDiscoverer using local nvidia-smi / torch.cuda / TVM availability, no network.
3. HardwareProbeRunner interface with no-op/static mode and measured mode.
4. TracePlan schema that separates model loading from trace-boundary decisions.
5. TraceBoundaryDetector prototype for HEAL HeterModelBaseline:
   - auto-detect backbone/shrinker/heads;
   - mark sparse/fusion/attention modules as skipped_subgraphs;
   - infer BEV shape from config when possible;
   - validate with forward sanity and DepGraph build.
6. Update model_classifier to consume trace plan confidence and evidence grade, not model-name branches.
```

Suggested files to inspect first:

- `framework/stage1/hardware_scan.py`
- `framework/capability_schema.py`
- `configs/hardware/*.yaml`
- `framework/stage1/adapters.py`
- `framework/stage1/auto_trace.py`
- `framework/stage1/graph_scan.py`
- `framework/stage1/model_classifier.py`
- `framework/tests/test_stage1_model_classifier_contract.py`
- `framework/tests/test_stage1_manifest_predictor_fields.py`

Suggested verification after changes:

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

## 8. Last Verified State

Last focused verification after integrating three-class output:

```text
23 passed
```

Command:

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py
```

Also passed:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

Schema check passed for:

- `acceleration_class_labels`
- CoDriving = `SEPARABLE_ACCELERATION`
- F-Cooper / AttFuse / V2X-ViT / Pyramid = `CO_ACCELERATION_REQUIRED`
- Where2comm / V2VNet / DiscoNet = `SCAN_FAILED`
- A2 `measured_h800_tvm == {}`

