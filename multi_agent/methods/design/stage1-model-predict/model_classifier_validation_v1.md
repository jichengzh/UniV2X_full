# Stage1 Model Classifier Validation v1

Date: 2026-06-24

## Contract

The end-to-end classifier consumes Stage1 manifests plus existing S2/S2.5/S3/S4 evidence and emits one row per default model. It does not launch new probes.

Required output artifacts:

- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`

Required schema fields per model:

- `model`
- `ckpt_status`
- `acceleration_class`
- `acceleration_class_label`
- `acceleration_class_reason`
- `classification`
- `scope`
- `backend_policy`
- `evidence_level`
- `evidence_sources`
- `historical_evidence_sources`
- `blockers`
- `required_next_probe_or_gate`
- `unsupported_conclusions`
- `no_overpromotion`

The integrated coarse model classification uses exactly three machine-readable
classes:

- `CO_ACCELERATION_REQUIRED`: 需要协同加速
- `SEPARABLE_ACCELERATION`: 可分离加速
- `SCAN_FAILED`: 扫描失败

`classification` remains as the fine-grained evidence verdict for compatibility.

## Backend Policy

New measured evidence is restricted to H800 TVM / Relax / MetaSchedule:

- `default_new_measurement_backend = h800_tvm`
- `allowed_new_measurement_backends = ["h800_tvm"]`
- `no_overpromotion = true`

TRT is retained only as historical evidence. Any Pyramid TRT AP/latency evidence from S3 is stored under `historical_evidence_sources` / `historical_trt_evidence`; it is not a probe backend, default classifier backend, or new acceptance backend. V2X-ViT true TRT INT8 AP remains blocked / not done.

## Manifest Normalization

The classifier and scanner both preserve legacy `trace.skipped_modules` while adding typed `trace.skipped_subgraphs`. Each typed skip records:

- `name`
- `type`
- `full_model_verdict_blocker`
- `blocker_gate`
- `source`

The B1 prune-group feature block includes `cin/cout/groups/ic_bn/kernel/stride/op_types/fanout_buckets`. B1 search-group feature rollup includes `min_ic_bn/max_groups/op_types/fanout_buckets`.

`view_latency.coverage` is explicitly trace-net coverage, not full-model coverage:

- `coverage_scope = trace_net_only`
- `full_model_latency_pct = null`
- skipped sparse/fusion/attention/custom subgraphs remain separately gated.

## Classification Boundaries

The report forbids these promotions:

- `groups=1` as model-level separability proof.
- `no int8 buildability cliff` as model-level separability proof.
- the former bridge all-serial/separable label as an architecture verdict.
- S4 H800 TVM latency-only matrix as AP/HV or full-model irreducible-coupling evidence.
- random-init fusion timing as trained-checkpoint evidence.
- architecture-only scans for Where2comm / V2VNet / DiscoNet as trained model classifications.
- `measured_h800_tvm` evidence on Where2comm / V2VNet / DiscoNet while they remain `missing_architecture_scan_only`.
- CoDriving scoped measured anchor as cross-model evidence.

The active three-arm search path also keeps historical TRT ratio context disabled as a default latency backend. Missing H800 INT8 data uses neutral FP16 latency unless real H800 TVM data or the explicit H800 stage0 proxy is selected.

## Acceptance Commands

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py
```

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q framework/tests/test_stage1_autoscan_extensions.py
```

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m framework.stage1_bridge
```

Expected bridge result: no former all-serial/separable architecture label.
