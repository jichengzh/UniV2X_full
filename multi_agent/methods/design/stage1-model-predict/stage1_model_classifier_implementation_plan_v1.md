# Stage1 End-to-End Model Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converge Safe Predictor v0 and Calibrated Predictor v1 into an end-to-end Stage1 model classifier that consumes Stage1 manifests plus S2/S2.5/S3/S4 evidence and emits conservative per-model classifications.

**Architecture:** Keep Stage1 manifest generation, evidence ingestion, and classification as separate surfaces. New measurements are normalized to a single backend policy (`h800_tvm`); TRT evidence is retained only as historical evidence and must not become a new probe backend, default classifier backend, or acceptance backend. The classifier is evidence-first: static dense-trace facts, low-cost H800 TVM latency anchors, coverage gates, and blocked/historical evidence are separated so no model is overpromoted.

**Tech Stack:** Python 3, PyYAML/JSON, existing `framework/stage1/*` predictor pipeline, existing Stage1 manifests, pytest, existing `results/stage1_model_predict/*` evidence artifacts, Markdown reports.

---

## Core Constraints

- 新增实测后端统一为 **H800 TVM / Relax / MetaSchedule**。
- `backend_policy.default_backend == "h800_tvm"`。
- `backend_policy.allowed_new_measurement_backends == ["h800_tvm"]`。
- TRT 只能作为 historical evidence 写入 `historical_evidence_sources` 或 `historical_trt_evidence`。
- TRT 不得进入 `measured_h800_tvm`，不得作为新增 probe backend、分类器默认后端或新验收后端。
- 不再允许 `groups=1`、`no cliff`、`bridge SEPARABLE` 被解释为模型级可分离。
- `no_overpromotion=True` 必须保留并由测试覆盖。
- Attention/fusion 未完整接入 trace 时，AttFuse / V2X-ViT / Where2comm / V2VNet / DiscoNet 不能输出 full-model separable。
- Where2comm / V2VNet / DiscoNet 是 `missing_architecture_scan_only`，不能写成 trained checkpoint classification。
- S4 是 H800 TVM latency-only，不含 AP/HV；只能支持 pair-level schedule calibration required，不能支持 full-model irreducible coupling。

## File Responsibilities

- Modify `framework/stage1/coupling_predictor.py`
  - Preserve v0 guardrails.
  - Add backend policy constants/helpers shared with the classifier.
  - Make evidence wording H800 TVM-first and demote TRT wording to historical-only where applicable.
- Modify `framework/stage1/calibrated_predictor.py`
  - Keep v1 report compatibility.
  - Rename measured evidence categories so S2/S4 latency is `measured_h800_tvm_latency`.
  - Move Pyramid TRT AP/latency categories into historical evidence.
- Create `framework/stage1/model_classifier.py`
  - Own the end-to-end classifier contract.
  - Load manifests and S2/S2.5/S3/S4 artifacts.
  - Emit per-model `classification`, `scope`, `evidence_level`, `evidence_sources`, `historical_evidence_sources`, `blockers`, `required_next_probe_or_gate`, `unsupported_conclusions`, and `no_overpromotion`.
- Optional create `framework/stage1/h800_tvm_evidence.py`
  - Keep evidence artifact parsing isolated if `model_classifier.py` grows too large.
- Modify `framework/stage1/adapters.py`
  - Add typed skip metadata defaults where adapters still expose only legacy `skipped_modules`.
- Modify `framework/stage1/auto_trace.py`
  - Ensure auto adapters expose both legacy skip strings and typed skipped subgraph information.
- Modify `framework/stage1/graph_scan.py`
  - Add per-B1 structural `feature` fields.
  - Add search-group feature rollups.
- Modify `framework/stage1/latency_profile.py`
  - Add `view_latency.coverage` with trace-net coverage wording.
- Modify `framework/stage1_bridge.py`
  - Replace `SEPARABLE (全旋钮可串行)` with non-model-separability wording.
- Modify `framework/search_three_arm.py`
  - Replace `No cliff (separable...)` language with traced dense-space buildability wording.
- Modify `framework/run_pqs_ablation.py`
  - Replace any search/report wording that implies no-cliff means separable.
- Create `scripts/stage1_classify_models.py`
  - End-to-end CLI for model classification.
- Preserve compatibility in `scripts/stage1_predict_coupling.py` and `scripts/stage1_predict_coupling_v1.py`
  - Existing CLIs must continue to run.
- Create `framework/tests/test_stage1_model_classifier_contract.py`
  - Contract tests for backend policy, historical TRT handling, output schema, and no-overpromotion.
- Modify `framework/tests/test_stage1_manifest_predictor_fields.py`
  - Manifest field tests for typed skips, B1 features, search rollups, and trace-net latency coverage.
- Update `framework/tests/test_coupling_predictor_static.py`
  - Keep v0 safety tests aligned with the new wording.
- Update `framework/tests/test_calibrated_predictor_v1.py`
  - Ensure v1 no longer labels TRT as new measured evidence.
- Create `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`
  - Validation rationale and critic acceptance notes.
- Update `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md`
  - Handoff status, artifacts, and remaining blockers.

## Input Evidence Artifacts

The classifier must read existing artifacts only:

- `results/stage1_model_predict/s2_schedule_anchor_audit_v1.json`
- `results/stage1_model_predict/s2_probe_results/stage1_s2_probe_completion_v1.json`
- `results/stage1_model_predict/s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json`
- `results/stage1_model_predict/s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json`
- `results/stage1_model_predict/s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json`
- `results/stage1_model_predict/stage1_coupling_predictions_v1.json`

## Default Model Set

The default CLI run must classify exactly these 9 manifests:

- CoDriving
- F-Cooper
- AttFuse
- V2X-ViT
- Pyramid lidar
- Pyramid camera
- Where2comm
- V2VNet
- DiscoNet

Who2com is excluded because it lacks a directly buildable config/ckpt combination.

## Output Artifacts

- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`
- `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`
- `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`
- `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md`

The design directory must contain Markdown only. JSON/YAML results go under `results/stage1_model_predict/`. Python code goes under `framework/` or `scripts/`.

---

## Task 1: Establish Classifier Contract And Backend Policy

**Files:**
- Modify: `framework/stage1/coupling_predictor.py`
- Modify: `framework/stage1/calibrated_predictor.py`
- Create: `framework/stage1/model_classifier.py`
- Create: `framework/tests/test_stage1_model_classifier_contract.py`

- [ ] **Step 1: Write failing contract tests**
  - Assert classifier schema is `stage1_model_classification_v1`.
  - Assert root backend policy contains:
    - `default_backend == "h800_tvm"`
    - `default_new_measurement_backend == "h800_tvm"`
    - `allowed_new_measurement_backends == ["h800_tvm"]`
  - Assert no allowed new backend contains `trt`.
  - Assert each model item contains:
    - `model`
    - `ckpt_status`
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
  - Assert `no_overpromotion is True`.
  - Assert TRT paths/categories appear only in historical fields.

- [ ] **Step 2: Run contract tests red**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q framework/tests/test_stage1_model_classifier_contract.py
```

Expected: fail because `framework/stage1/model_classifier.py` and contract fields do not exist yet.

- [ ] **Step 3: Implement minimal contract**
  - Add backend policy constants.
  - Add `classify_models(...)` and `build_classification_report(...)`.
  - Return a valid schema for the default 9 models using existing v1 predictions as seed context.
  - Preserve conservative blockers and unsupported conclusions.

- [ ] **Step 4: Run contract tests green**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q framework/tests/test_stage1_model_classifier_contract.py
```

Expected: pass.

## Task 2: Add Stage1 Manifest Predictor Fields

**Files:**
- Modify: `framework/stage1/adapters.py`
- Modify: `framework/stage1/auto_trace.py`
- Modify: `framework/stage1/graph_scan.py`
- Modify: `framework/stage1/latency_profile.py`
- Modify: `framework/tests/test_stage1_manifest_predictor_fields.py`

- [ ] **Step 1: Write failing manifest-field tests**
  - For current/default manifests, assert `trace.skipped_subgraphs` and legacy `trace.skipped_modules` are both available after normalization.
  - Assert `view_b1_prune_groups[*].feature` includes:
    - `cin`
    - `cout`
    - `groups`
    - `ic_bn`
    - `kernel`
    - `stride`
    - `op_types`
    - `fanout_buckets`
  - Assert `view_b1_search_groups[*].feature` includes:
    - `min_ic_bn`
    - `max_groups`
    - `op_types`
    - `fanout_buckets`
  - Assert `view_latency.coverage.coverage_scope == "trace_net_only"`.
  - Assert coverage notes explicitly say this is not full-model coverage.

- [ ] **Step 2: Run manifest tests red**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: fail on missing typed skips, missing feature fields, or missing coverage block.

- [ ] **Step 3: Implement typed skipped-subgraph normalization**
  - Keep legacy `skipped_modules`.
  - Add typed `skipped_subgraphs` entries with:
    - `name`
    - `type`
    - `full_model_verdict_blocker`
    - `blocker_gate`
    - `source`
  - Map attention/transformer/HMSA/MSwin to `attention_or_fusion`.
  - Map sparse/pillar/scatter/VFE to `sparse_frontend`.
  - Map geometry/QuickCumsum/custom projection to `custom_or_geometry_projection`.
  - Map learned/routing/GNN fusion to `fusion` or routing/fusion blocker.
  - Map MaxFusion/channel-preserving pooling as non-proof coverage note, not full-model pass.

- [ ] **Step 4: Implement B1 structural feature fields**
  - Compute `cin`, `cout`, `groups`, `ic_bn = cin / groups`, `kernel`, `stride`, `op_types`, and `fanout_buckets` from group members.
  - If a member lacks a shape field, use `None` rather than inventing values.
  - `fanout_buckets` should be derived from coupled buckets/member semantic buckets.

- [ ] **Step 5: Implement search-group feature rollups**
  - Aggregate member B1 features into search-group `feature`.
  - `min_ic_bn` is the minimum known member `ic_bn`.
  - `max_groups` is the maximum known member `groups`.
  - `op_types` and `fanout_buckets` are sorted unique lists.

- [ ] **Step 6: Implement trace-net latency coverage**
  - Add `view_latency.coverage` in both measured and skipped modes.
  - Include:
    - `coverage_scope: "trace_net_only"`
    - `trace_net_latency_pct`
    - `full_model_latency_pct: null`
    - `skipped_subgraphs_accounted_separately: true`
    - `note` explaining this is not full-model coverage.

- [ ] **Step 7: Run manifest tests green**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: pass.

## Task 3: Implement H800 TVM Evidence Ingestion

**Files:**
- Modify: `framework/stage1/coupling_predictor.py`
- Modify: `framework/stage1/calibrated_predictor.py`
- Modify/Create: `framework/stage1/h800_tvm_evidence.py`
- Extend: `framework/tests/test_stage1_model_classifier_contract.py`
- Extend: `framework/tests/test_calibrated_predictor_v1.py`

- [ ] **Step 1: Write failing evidence-ingestion tests**
  - Assert S2/S4 latency evidence is labeled `measured_h800_tvm_latency`.
  - Assert Pyramid TRT AP/latency is labeled `historical_trt_evidence`.
  - Assert Pyramid TRT does not appear under `measured_h800_tvm`.
  - Assert V2X-ViT true TRT INT8/AP remains blocked or not done.

- [ ] **Step 2: Run evidence tests red**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_calibrated_predictor_v1.py
```

Expected: fail on legacy TRT-as-measured categories.

- [ ] **Step 3: Implement artifact readers**
  - Read S2 audit and S2 probe completion artifacts.
  - Read S2.5 coverage gates.
  - Read S3 quant sensitivity.
  - Read S4 three-arm validation.
  - Normalize S2/S4 H800 TVM latency evidence into measured H800 fields.
  - Normalize TRT AP/latency evidence into historical-only fields.

- [ ] **Step 4: Keep blocked evidence blocked**
  - V2X-ViT `true_trt_int8_ap.status == "NOT_DONE"` must produce blocker wording.
  - Random-init timing sidecars must not become trained checkpoint evidence.
  - Architecture-only scans must stay architecture-only.

- [ ] **Step 5: Run evidence tests green**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_calibrated_predictor_v1.py
```

Expected: pass.

## Task 4: Replace Misleading Separability Wording

**Files:**
- Modify: `framework/stage1_bridge.py`
- Modify: `framework/search_three_arm.py`
- Modify: `framework/run_pqs_ablation.py`
- Extend: `framework/tests/test_stage1_model_classifier_contract.py` or create focused static wording test.

- [ ] **Step 1: Write failing wording test**
  - Assert `python -m framework.stage1_bridge` output does not contain `SEPARABLE (全旋钮可串行)`.
  - Assert source files no longer contain `No cliff (separable`.

- [ ] **Step 2: Run wording test red**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m framework.stage1_bridge
rg -n "SEPARABLE \\(全旋钮可串行\\)|No cliff \\(separable" framework/stage1_bridge.py framework/search_three_arm.py framework/run_pqs_ablation.py
```

Expected: current bridge/search wording is found.

- [ ] **Step 3: Replace wording**
  - Replace bridge verdict with `STRUCTURAL_LOW_CLIFF_RISK_NOT_MODEL_SEPARABLE_PROOF`.
  - Replace comments/logs with “no int8 buildability cliff in traced dense space”.
  - Keep meaning scoped to traced dense search space.

- [ ] **Step 4: Verify wording**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m framework.stage1_bridge
rg -n "SEPARABLE \\(全旋钮可串行\\)|No cliff \\(separable" framework/stage1_bridge.py framework/search_three_arm.py framework/run_pqs_ablation.py
```

Expected: bridge output does not contain old wording; `rg` returns no matches.

## Task 5: Add End-to-End CLI And Reports

**Files:**
- Create: `scripts/stage1_classify_models.py`
- Preserve: `scripts/stage1_predict_coupling.py`
- Preserve: `scripts/stage1_predict_coupling_v1.py`
- Extend: `framework/tests/test_stage1_model_classifier_contract.py`

- [ ] **Step 1: Write failing CLI/schema test**
  - Run classifier CLI with default manifests.
  - Assert JSON exists at `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json`.
  - Assert Markdown exists at `results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md`.
  - Assert JSON has 9 models and required fields.

- [ ] **Step 2: Run CLI test red**

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

Expected: fail because CLI does not exist.

- [ ] **Step 3: Implement CLI**
  - Default to the 9 model manifests listed above.
  - Support repeated `--manifest`.
  - Support `--evidence-dir`.
  - Support `--out-json` and `--out-md`.
  - Write parent directories.
  - Exit non-zero if no models are classified.

- [ ] **Step 4: Generate outputs**

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

Expected: writes both output artifacts.

- [ ] **Step 5: Run schema check**

```bash
PYTHONPATH=/home/jichengzhi/V2X python - <<'PY'
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
    assert "classification" in item
    assert "scope" in item
    assert "blockers" in item
    assert "historical_evidence_sources" in item
print("stage1_model_classifier_ok")
PY
```

Expected: prints `stage1_model_classifier_ok`.

## Task 6: Generate Validation And Handoff Documents

**Files:**
- Create: `multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md`
- Update: `multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md`

- [ ] **Step 1: Write validation document**
  - State that H800 TVM is the only new measurement backend.
  - State that TRT is historical evidence only.
  - Include per-model classification summary.
  - Include critic checks:
    - no blocked/pass confusion
    - no TRT backend misuse
    - no CoDriving cross-model extrapolation
    - no random-init timing promotion
    - no architecture-only scan written as checkpoint scan

- [ ] **Step 2: Update handoff**
  - Link classifier JSON/Markdown.
  - Link validation document.
  - Record remaining blockers exactly:
    - attention/fusion not fully traced
    - Where2comm/V2VNet/DiscoNet missing ckpts
    - Who2com excluded
    - S4 latency-only
    - TRT historical-only

- [ ] **Step 3: Verify design/result placement**

```bash
find multi_agent/methods/design/stage1-model-predict -maxdepth 1 -type f ! -name '*.md' -print
find results/stage1_model_predict/model_classifier -maxdepth 1 -type f -print
```

Expected: design directory contains only Markdown; model classifier results are under `results/stage1_model_predict/model_classifier/`.

## Task 7: Full Verification

**Files:**
- All files touched above.

- [ ] **Step 1: Run base static tests**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py
```

Expected: pass.

- [ ] **Step 2: Run old autoscan extension regression**

```bash
PATH=/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin:$PATH \
PYTHONPATH=/home/jichengzhi/V2X \
pytest -q framework/tests/test_stage1_autoscan_extensions.py
```

Expected: pass.

- [ ] **Step 3: Run CLI acceptance**

```bash
PYTHONPATH=/home/jichengzhi/V2X \
python scripts/stage1_classify_models.py \
  --out-json results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-md results/stage1_model_predict/model_classifier/stage1_model_classification_v1.md
```

Expected: exit 0 and writes both artifacts.

- [ ] **Step 4: Run report schema check**

```bash
PYTHONPATH=/home/jichengzhi/V2X python - <<'PY'
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
    assert "classification" in item
    assert "scope" in item
    assert "blockers" in item
    assert "historical_evidence_sources" in item
print("stage1_model_classifier_ok")
PY
```

Expected: prints `stage1_model_classifier_ok`.

- [ ] **Step 5: Run bridge wording acceptance**

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m framework.stage1_bridge
```

Expected: output no longer contains `SEPARABLE (全旋钮可串行)`.

- [ ] **Step 6: Run py_compile**

```bash
python -m py_compile \
  framework/stage1/coupling_predictor.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage1/model_classifier.py \
  scripts/stage1_classify_models.py
```

Expected: exit 0.

- [ ] **Step 7: Review final diff**

```bash
git diff -- \
  framework/stage1/coupling_predictor.py \
  framework/stage1/calibrated_predictor.py \
  framework/stage1/model_classifier.py \
  framework/stage1/adapters.py \
  framework/stage1/auto_trace.py \
  framework/stage1/graph_scan.py \
  framework/stage1/latency_profile.py \
  framework/stage1_bridge.py \
  framework/search_three_arm.py \
  framework/run_pqs_ablation.py \
  scripts/stage1_classify_models.py \
  framework/tests/test_stage1_model_classifier_contract.py \
  framework/tests/test_stage1_manifest_predictor_fields.py \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_calibrated_predictor_v1.py \
  multi_agent/methods/design/stage1-model-predict/model_classifier_validation_v1.md \
  multi_agent/methods/progress/HANDOFF_stage1_safe_predictor_v0.md
```

Expected: no unrelated edits; no TRT-as-new-backend wording; no full-model separable overpromotion.

---

## Expected Classifier Output Fields

Each model object must contain:

- `model`
- `ckpt_status`
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

Root report must contain:

- `schema: "stage1_model_classification_v1"`
- `backend_policy`
- `evidence_inputs`
- `models`
- `no_overpromotion`
- `unsupported_global_conclusions`

## Blockers To Preserve

- Attention/fusion 尚未完整接入 Stage1 trace，因此 AttFuse / V2X-ViT / Where2comm / V2VNet / DiscoNet 不能输出 full-model separable。
- Where2comm / V2VNet / DiscoNet 当前是 `missing_architecture_scan_only`，不能写成 trained checkpoint classification。
- Who2com 缺少可直接构建的 config/ckpt 组合，不纳入本轮分类器默认 9 模型集合。
- S4 是 H800 TVM latency-only，不含 AP/HV；只能支持 pair-level schedule calibration required，不能支持 full-model irreducible coupling。
- TRT 只能保留为历史证据；任何新增测量、分类器默认后端、probe runner 都不能使用 TRT。

## `/goal` Prompt

```text
/goal 按照当前 Stage1 handoff 与 implementation plan，完成端到端 Stage1 模型分类器实现。严格约束：新增实测后端统一为 H800 TVM/Relax/MetaSchedule；TRT 只允许作为 historical evidence，不得作为新增后端或默认分类依据。实施内容包括：补齐 manifest typed skipped_subgraphs、结构特征和 coverage 字段；建立 model_classifier 输出契约；接入 S2/S2.5/S3/S4 既有证据；替换 bridge/search 中误导性的 separable 文案；生成 results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json/.md；更新 handoff。验收必须通过 pytest、CLI schema 检查、no_overpromotion=True，并由 critic agent 审查是否存在 blocked/pass 混淆、TRT 误用、CoDriving 外推、random-init timing 误用或 architecture-only 写成 checkpoint scan。
```
