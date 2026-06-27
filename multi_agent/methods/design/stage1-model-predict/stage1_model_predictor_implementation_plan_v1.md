# Stage1 Model Separability Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-cost, mechanism-aware predictor that estimates whether a model architecture is P/Q/S-separable or coupling-prone before running full joint-vs-serial enumeration.

**Architecture:** Extend Stage1 from "manifest generation" to "manifest + coverage + coupling-risk features + optional anchor probes". The predictor must separate three concepts: static structural risk, low-cost probe evidence, and measured ablation evidence. It must never turn "no grouped-conv cliff in the traced dense subgraph" into a full-model `SEPARABLE` verdict when fusion/attention/custom operators are skipped or unprofiled.

**Tech Stack:** Python 3, PyYAML, dataclasses, existing `framework/stage1/*` manifest pipeline, existing `framework/stage1_bridge.py`, pytest, optional CUDA/TVM probe hooks for sampled anchors only.

---

## Context And Evidence Basis

This plan is grounded in the existing coupling-map evidence:

- `multi_agent/methods/design/coupling_map_v1.md`: coupling is defined by `argmin` drift and joint-vs-serial HV gap, not by hand-written architecture labels.
- `multi_agent/methods/design/auto-tuning/1_design_space_building_v1.md`: the most reliable current mechanism is P-hub coupling: `width/groups -> IC_BN`, which changes quantization format reachability and schedule search space.
- `results/coupling_map_matrix.json`: Pyramid has P-hub coupling; CoDriving is measured separable after low-dimensional and high-dimensional checks; V2X-ViT shows Q-granularity/P and routing/fusion signals.
- Current failure mode: `framework/stage1_bridge.py` treats all `grouped_conv=false` knobs as low risk and may emit `SEPARABLE` even when fusion or attention is skipped.
- `standard_conv_coupling_deepening_plan_v1.md`: `groups=1` is not a separability verdict. It only removes the known grouped-conv IC_BN hard cliff; standard Conv2d still needs shape features or low-cost anchor probes before being promoted to low-risk.
- `standard_conv_anchor_measured_results_v1.md`: CoDriving is a measured negative anchor only inside its measured scope; it must not be copied as a universal standard-conv rule for F-Cooper, AttFuse, V2X-ViT, or Pyramid mixed grouped+standard contexts.

The new predictor must be cheap. It may run small targeted probes on a few anchor shapes, but must not run full P x Q x S enumeration, full MetaSchedule grids, or multi-seed joint-vs-serial searches as part of prediction.

## File Structure

Create and modify the following files:

- Create `framework/stage1/coupling_predictor.py`
  - Owns predictor dataclasses, feature extraction from manifest, risk scoring, confidence rules, and report serialization.
- Create `framework/stage1/coupling_probe.py`
  - Owns low-cost optional anchor-probe interfaces and pure-Python probe result schemas. It must be runnable in `static-only` mode without CUDA/TVM.
- Modify `framework/stage1/graph_scan.py`
  - Add richer per-group structural metadata required by the predictor: root op shape, `cin`, `cout`, `groups`, `ic_bn`, kernel/stride, member op types, fanout buckets.
- Modify `framework/stage1/auto_trace.py`
  - Add typed skipped-subgraph metadata instead of free-text skip strings for fusion/attention/sparse/custom operators.
- Modify `framework/stage1/latency_profile.py`
  - Add trace coverage accounting: traced leaf latency, skipped estimated bucket, profile status, and model coverage class.
- Modify `framework/stage1_bridge.py`
  - Stop using `coupling_summary()` as an architecture-level verdict. Keep bridge compatibility, but add a deprecation note and route new predictions through `coupling_predictor`.
- Create `scripts/stage1_predict_coupling.py`
  - CLI: load one or more manifests, run static prediction, optionally merge probe files, emit JSON and Markdown reports.
- Create `framework/tests/test_coupling_predictor_static.py`
  - Tests static risk logic against synthetic manifests and existing small fixtures.
- Create `framework/tests/test_stage1_manifest_predictor_fields.py`
  - Tests new manifest fields are present and backward-compatible defaults are safe.
- Create `framework/tests/fixtures/coupling_predictor/`
  - Store minimal YAML fixtures for Pyramid-like P-hub, CoDriving-like standard conv, and AttFuse-like skipped fusion.

## Predictor Output Contract

The predictor returns a report with this shape:

```python
{
    "model": "attfuse",
    "scope": "traced_dense_subgraph",
    "verdict": "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE",
    "confidence": 0.42,
    "risk": {
        "risk_pq": 0.18,
        "risk_ps": 0.27,
        "risk_qs": 0.12,
        "risk_p_hub": 0.19,
        "risk_uncovered": 0.82
    },
    "evidence": [
        "all traced conv knobs have IC_BN >= 4 across legal widths",
        "fusion_net skipped with type=attention_or_fusion",
        "latency coverage is unknown because latency_status=skipped"
    ],
    "recommended_action": "run_fusion_anchor_probe_or_profile_before_full_model_verdict"
}
```

Allowed verdicts:

- `MEASURED_SEPARABLE`: only when full joint-vs-serial evidence exists and passes the existing measurement criterion.
- `PREDICTED_SEPARABLE_LOW_RISK`: reserved for a future calibrated mode where static features, measured anchor probes, and coverage checks all pass. Static-only output must not use this verdict.
- `ANCHOR_PROBED_LOW_RISK`: one scoped measured anchor shows low risk, with real latency/AP evidence and no uncovered blockers; this is not a full-model measured separability verdict.
- `P_HUB_COUPLED`: P controls Q/S legality or schedule signature through IC_BN / alignment / tensorization.
- `Q_S_COUPLED`: quantization format changes schedule signature at fixed P.
- `FUSION_UNCOVERED_UNKNOWN`: skipped fusion/attention/custom subgraph has non-trivial latency or unknown coverage.
- `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE`: static evidence is incomplete or contradictory.

The predictor must not emit `MEASURED_SEPARABLE` or `PREDICTED_SEPARABLE_LOW_RISK` from static features alone.

## Task 1: Add Predictor Fixtures And Static Contract Tests

**Files:**
- Create: `framework/tests/fixtures/coupling_predictor/pyramid_p_hub.yaml`
- Create: `framework/tests/fixtures/coupling_predictor/codriving_standard.yaml`
- Create: `framework/tests/fixtures/coupling_predictor/attfuse_skipped_fusion.yaml`
- Create: `framework/tests/test_coupling_predictor_static.py`
- Create: `framework/stage1/coupling_predictor.py`

- [ ] **Step 1: Add minimal fixture for a Pyramid-like P-hub model**

Create `framework/tests/fixtures/coupling_predictor/pyramid_p_hub.yaml`:

```yaml
stage: stage1_partition
model: pyramid_fixture
scan_status: ok
search_space_summary:
  n_b1_search_knobs: 2
  latency_status: skipped
trace:
  entry_shape: [1, 64, 128, 256]
  skipped_modules: []
  skipped_subgraphs: []
view_b1_search_groups:
  - search_group_id: bev_encoder.s0
    bucket: bev_encoder
    widths: [128]
    max_rate: 0.75
    round_to: 32
    int8_buildable_align: 128
    grouped_conv: true
    criterion_pool: [L1, FPGM]
    feature:
      cin: 128
      cout: 128
      groups: 32
      ic_bn: 4
      op_types: [Conv2d, BatchNorm2d]
      kernel: [3, 3]
      fanout_buckets: [bev_encoder]
  - search_group_id: neck
    bucket: neck
    widths: [256]
    max_rate: 0.75
    round_to: 32
    int8_buildable_align: 4
    grouped_conv: false
    criterion_pool: [L1]
    feature:
      cin: 384
      cout: 256
      groups: 1
      ic_bn: 384
      op_types: [Conv2d]
      kernel: [1, 1]
      fanout_buckets: [heads]
view_latency:
  status: skipped
  coverage:
    traced_latency_pct: null
    skipped_latency_pct: null
```

- [ ] **Step 2: Add minimal fixture for a CoDriving-like measured standard-conv model**

Create `framework/tests/fixtures/coupling_predictor/codriving_standard.yaml`:

```yaml
stage: stage1_partition
model: codriving_fixture
scan_status: ok
measured_coupling:
  source: results/coupling_map/C0c_codriving_pqs.json
  joint_vs_serial: serial_matches_joint
  rank_flip_pairs: 0
search_space_summary:
  n_b1_search_knobs: 2
  latency_status: skipped
trace:
  entry_shape: [1, 64, 192, 704]
  skipped_modules:
    - fusion_net (CoDriving 0-param channel-preserving, auto-skip)
  skipped_subgraphs:
    - name: fusion_net
      type: channel_preserving_fusion
      expected_latency_class: low
      full_model_verdict_blocker: false
view_b1_search_groups:
  - search_group_id: backbone.s0
    bucket: backbone
    widths: [64]
    max_rate: 0.5
    round_to: 32
    int8_buildable_align: 4
    grouped_conv: false
    criterion_pool: [L1, FPGM]
    feature:
      cin: 64
      cout: 64
      groups: 1
      ic_bn: 64
      op_types: [Conv2d, BatchNorm2d]
      kernel: [3, 3]
      fanout_buckets: [backbone]
  - search_group_id: neck
    bucket: neck
    widths: [128]
    max_rate: 0.75
    round_to: 32
    int8_buildable_align: 4
    grouped_conv: false
    criterion_pool: [L1]
    feature:
      cin: 384
      cout: 128
      groups: 1
      ic_bn: 384
      op_types: [Conv2d]
      kernel: [1, 1]
      fanout_buckets: [heads]
view_latency:
  status: skipped
  coverage:
    traced_latency_pct: null
    skipped_latency_pct: null
```

- [ ] **Step 3: Add minimal fixture for AttFuse-like skipped attention/fusion**

Create `framework/tests/fixtures/coupling_predictor/attfuse_skipped_fusion.yaml`:

```yaml
stage: stage1_partition
model: attfuse_fixture
scan_status: ok
search_space_summary:
  n_b1_search_knobs: 2
  latency_status: skipped
trace:
  entry_shape: [1, 64, 512, 512]
  skipped_modules:
    - fusion_net (AttFusion: multi-agent attention fusion, auto-skip)
  skipped_subgraphs:
    - name: fusion_net
      type: attention_or_fusion
      expected_latency_class: unknown
      full_model_verdict_blocker: true
view_b1_search_groups:
  - search_group_id: backbone.s0
    bucket: backbone
    widths: [64]
    max_rate: 0.5
    round_to: 32
    int8_buildable_align: 4
    grouped_conv: false
    criterion_pool: [L1, FPGM]
    feature:
      cin: 64
      cout: 64
      groups: 1
      ic_bn: 64
      op_types: [Conv2d, BatchNorm2d]
      kernel: [3, 3]
      fanout_buckets: [backbone]
  - search_group_id: neck
    bucket: neck
    widths: [256]
    max_rate: 0.75
    round_to: 32
    int8_buildable_align: 4
    grouped_conv: false
    criterion_pool: [L1]
    feature:
      cin: 384
      cout: 256
      groups: 1
      ic_bn: 384
      op_types: [Conv2d]
      kernel: [1, 1]
      fanout_buckets: [heads]
view_latency:
  status: skipped
  coverage:
    traced_latency_pct: null
    skipped_latency_pct: null
```

- [ ] **Step 4: Write failing static predictor tests**

Create `framework/tests/test_coupling_predictor_static.py`:

```python
from pathlib import Path

from framework.stage1.coupling_predictor import predict_manifest


FIXTURES = Path(__file__).parent / "fixtures" / "coupling_predictor"


def test_pyramid_fixture_predicts_p_hub_coupled():
    report = predict_manifest(FIXTURES / "pyramid_p_hub.yaml")

    assert report["verdict"] == "P_HUB_COUPLED"
    assert report["risk"]["risk_p_hub"] >= 0.70
    assert report["risk"]["risk_pq"] >= 0.40
    assert any("IC_BN" in item for item in report["evidence"])


def test_codriving_fixture_uses_measured_separable_when_evidence_exists():
    report = predict_manifest(FIXTURES / "codriving_standard.yaml")

    assert report["verdict"] == "MEASURED_SEPARABLE"
    assert report["scope"] == "full_model"
    assert report["confidence"] >= 0.90
    assert report["risk"]["risk_uncovered"] <= 0.20


def test_attfuse_fixture_blocks_full_model_separable_due_to_skipped_fusion():
    report = predict_manifest(FIXTURES / "attfuse_skipped_fusion.yaml")

    assert report["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert report["scope"] == "traced_dense_subgraph"
    assert report["risk"]["risk_uncovered"] >= 0.70
    assert "run_fusion_anchor_probe_or_profile_before_full_model_verdict" == report["recommended_action"]
```

- [ ] **Step 5: Run tests and verify they fail before implementation**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'framework.stage1.coupling_predictor'`.

- [ ] **Step 6: Implement the minimal static predictor API**

Create `framework/stage1/coupling_predictor.py` with the initial implementation:

```python
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


VERDICT_MEASURED_SEPARABLE = "MEASURED_SEPARABLE"
VERDICT_PREDICTED_SEPARABLE_LOW_RISK = "PREDICTED_SEPARABLE_LOW_RISK"
VERDICT_ANCHOR_PROBED_LOW_RISK = "ANCHOR_PROBED_LOW_RISK"
VERDICT_P_HUB_COUPLED = "P_HUB_COUPLED"
VERDICT_Q_S_COUPLED = "Q_S_COUPLED"
VERDICT_FUSION_UNCOVERED_UNKNOWN = "FUSION_UNCOVERED_UNKNOWN"
VERDICT_LOW_CONFIDENCE = "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return data


def _legal_widths(group: dict[str, Any]) -> list[int]:
    widths = [int(w) for w in group.get("widths", [])]
    if not widths:
        return []
    base = max(widths)
    round_to = max(1, int(group.get("round_to", 32)))
    max_rate = float(group.get("max_rate", 0.0))
    floor = max(round_to, int(math.ceil(base * (1.0 - max_rate) / round_to)) * round_to)
    return [w for w in range(floor, base + 1, round_to)]


def _risk_from_group(group: dict[str, Any]) -> dict[str, Any]:
    feature = group.get("feature", {}) or {}
    round_to = max(1, int(group.get("round_to", 32)))
    int8_align = max(1, int(group.get("int8_buildable_align", round_to)))
    legal = _legal_widths(group)
    unbuildable = [w for w in legal if w % int8_align != 0]
    ic_bn = float(feature.get("ic_bn", 9999.0))
    grouped = bool(group.get("grouped_conv", False))

    pq = 0.0
    if legal and unbuildable:
        pq = max(pq, len(unbuildable) / len(legal))
    if ic_bn < 4:
        pq = max(pq, 0.90)
    elif ic_bn < 8:
        pq = max(pq, 0.55)

    ps = 0.0
    if grouped:
        ps = max(ps, 0.70)
    if ic_bn < 8:
        ps = max(ps, 0.55)
    if len(feature.get("fanout_buckets", []) or []) > 1:
        ps = max(ps, 0.45)

    p_hub = max(pq, ps) if (grouped or ic_bn < 8 or unbuildable) else min(max(pq, ps), 0.25)

    return {
        "id": group.get("search_group_id", "unknown"),
        "risk_pq": round(_clip01(pq), 4),
        "risk_ps": round(_clip01(ps), 4),
        "risk_p_hub": round(_clip01(p_hub), 4),
        "evidence": {
            "ic_bn": ic_bn,
            "grouped_conv": grouped,
            "legal_widths": legal,
            "unbuildable_int8_widths": unbuildable,
        },
    }


def _skipped_risk(manifest: dict[str, Any]) -> tuple[float, list[str], bool]:
    trace = manifest.get("trace", {}) or {}
    skipped = trace.get("skipped_subgraphs", []) or []
    if not skipped:
        return 0.0, [], False

    risk = 0.0
    evidence: list[str] = []
    blocks_full_model = False
    for item in skipped:
        name = str(item.get("name", "unknown"))
        kind = str(item.get("type", "unknown"))
        blocker = bool(item.get("full_model_verdict_blocker", False))
        if blocker or kind in {"attention_or_fusion", "custom_cuda", "unknown"}:
            risk = max(risk, 0.85)
            blocks_full_model = True
        elif kind in {"sparse_vfe", "scatter"}:
            risk = max(risk, 0.35)
        else:
            risk = max(risk, 0.15)
        evidence.append(f"skipped {name} type={kind} blocker={blocker}")
    return round(risk, 4), evidence, blocks_full_model


def _measured_verdict(manifest: dict[str, Any]) -> str | None:
    measured = manifest.get("measured_coupling", {}) or {}
    if (
        measured.get("joint_vs_serial") == "serial_matches_joint"
        and int(measured.get("rank_flip_pairs", 1)) == 0
        and measured.get("latency_source") == "real"
        and measured.get("scope_coverage") == "full_model"
    ):
        return VERDICT_MEASURED_SEPARABLE
    return None


def _anchor_probe_verdict(probe: dict[str, Any] | None) -> str | None:
    if not probe:
        return None
    if probe.get("evidence_level") not in {"anchor_probe_measured", "measured_negative_anchor"}:
        return None
    if probe.get("latency_source") != "real":
        return None
    if probe.get("argmin_or_rank_flip", 1) != 0:
        return None
    if float(probe.get("hv_gap", 1.0)) > float(probe.get("noise_floor", 0.01)):
        return None
    if probe.get("uncovered_blocking_subgraphs", 1) != 0:
        return None
    return VERDICT_ANCHOR_PROBED_LOW_RISK


def predict_manifest(path: str | Path, probe: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = _load_manifest(path)
    groups = manifest.get("view_b1_search_groups", []) or []
    group_risks = [_risk_from_group(group) for group in groups]

    risk_pq = max((g["risk_pq"] for g in group_risks), default=0.0)
    risk_ps = max((g["risk_ps"] for g in group_risks), default=0.0)
    risk_p_hub = max((g["risk_p_hub"] for g in group_risks), default=0.0)
    risk_qs = 0.0

    if probe:
        risk_qs = max(risk_qs, float(probe.get("risk_qs", 0.0)))
        risk_ps = max(risk_ps, float(probe.get("risk_ps", 0.0)))
        risk_p_hub = max(risk_p_hub, float(probe.get("risk_p_hub", 0.0)))

    risk_uncovered, skipped_evidence, blocks_full_model = _skipped_risk(manifest)
    evidence = skipped_evidence[:]
    for g in group_risks:
        ev = g["evidence"]
        if ev["ic_bn"] < 8:
            evidence.append(f"{g['id']} has low IC_BN={ev['ic_bn']}")
        if ev["unbuildable_int8_widths"]:
            evidence.append(f"{g['id']} has INT8-unbuildable legal widths {ev['unbuildable_int8_widths']}")

    measured = _measured_verdict(manifest)
    anchor = _anchor_probe_verdict(probe)
    if measured:
        verdict = measured
        confidence = 0.95
        scope = "full_model"
        recommended = "use_serial_search_or_low_budget_confirmation"
    elif blocks_full_model:
        verdict = VERDICT_FUSION_UNCOVERED_UNKNOWN
        confidence = 0.40
        scope = "traced_dense_subgraph"
        recommended = "run_fusion_anchor_probe_or_profile_before_full_model_verdict"
    elif risk_p_hub >= 0.60:
        verdict = VERDICT_P_HUB_COUPLED
        confidence = 0.78
        scope = "full_model_if_coverage_ok"
        recommended = "allocate_joint_budget_to_high_p_hub_knobs"
    elif anchor and max(risk_pq, risk_ps, risk_qs, risk_uncovered) <= 0.30:
        verdict = anchor
        confidence = 0.70
        scope = probe.get("scope", "anchor_regime")
        recommended = "keep verdict scoped to measured anchor; do not promote to measured separable"
    elif max(risk_pq, risk_ps, risk_qs, risk_uncovered) <= 0.30:
        verdict = VERDICT_LOW_CONFIDENCE
        confidence = 0.52
        scope = "traced_dense_subgraph"
        recommended = "run_targeted_anchor_probe_before_any_low_risk_verdict"
    else:
        verdict = VERDICT_LOW_CONFIDENCE
        confidence = 0.50
        scope = "traced_dense_subgraph"
        recommended = "run_targeted_anchor_probe"

    return {
        "model": manifest.get("model", Path(path).stem),
        "scope": scope,
        "verdict": verdict,
        "confidence": round(_clip01(confidence), 4),
        "risk": {
            "risk_pq": round(_clip01(risk_pq), 4),
            "risk_ps": round(_clip01(risk_ps), 4),
            "risk_qs": round(_clip01(risk_qs), 4),
            "risk_p_hub": round(_clip01(risk_p_hub), 4),
            "risk_uncovered": round(_clip01(risk_uncovered), 4),
        },
        "knob_risks": group_risks,
        "evidence": evidence,
        "recommended_action": recommended,
    }
```

- [ ] **Step 7: Run static predictor tests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py
```

Expected: `3 passed`.

- [ ] **Step 8: Commit Task 1**

```bash
git add framework/stage1/coupling_predictor.py framework/tests/test_coupling_predictor_static.py framework/tests/fixtures/coupling_predictor
git commit -m "feat(stage1): add coupling risk predictor contract"
```

## Task 2: Add Typed Skipped-Subgraph Metadata To AutoTrace

**Files:**
- Modify: `framework/stage1/auto_trace.py`
- Modify: `framework/stage1/adapters.py`
- Modify: `framework/stage1/graph_scan.py`
- Test: `framework/tests/test_stage1_manifest_predictor_fields.py`

- [ ] **Step 1: Write failing tests for typed skipped-subgraph fields**

Create `framework/tests/test_stage1_manifest_predictor_fields.py`:

```python
from framework.stage1.auto_trace import get_auto_adapter
from framework.stage1.graph_scan import _manifest_trace_block


def test_attfuse_adapter_exposes_typed_fusion_skip():
    adapter = get_auto_adapter("attfuse")

    assert any(item["name"] == "fusion_net" for item in adapter.skipped_subgraphs)
    fusion = next(item for item in adapter.skipped_subgraphs if item["name"] == "fusion_net")
    assert fusion["type"] == "attention_or_fusion"
    assert fusion["full_model_verdict_blocker"] is True


def test_codriving_channel_preserving_fusion_is_not_a_full_model_blocker():
    adapter = get_auto_adapter("codriving")

    fusion = next(item for item in adapter.skipped_subgraphs if item["name"] == "fusion_net")
    assert fusion["type"] == "channel_preserving_fusion"
    assert fusion["full_model_verdict_blocker"] is False


def test_manifest_trace_block_includes_legacy_and_typed_skips():
    class Adapter:
        skipped_modules = ["fusion_net (legacy string)"]
        skipped_subgraphs = [{"name": "fusion_net", "type": "attention_or_fusion"}]

    block = _manifest_trace_block(Adapter(), [1, 64, 8, 8])

    assert block["skipped_modules"] == ["fusion_net (legacy string)"]
    assert block["skipped_subgraphs"] == [{"name": "fusion_net", "type": "attention_or_fusion"}]
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: FAIL because `skipped_subgraphs` and `_manifest_trace_block` do not exist.

- [ ] **Step 3: Extend adapter base objects with typed skipped metadata**

Modify `framework/stage1/adapters.py` base `TraceAdapter` class to define:

```python
class TraceAdapter:
    skipped_modules: list[str] = []
    skipped_subgraphs: list[dict] = []
```

Keep the existing string `skipped_modules` for backward compatibility.

- [ ] **Step 4: Add typed skips to AutoTrace adapters**

In `framework/stage1/auto_trace.py`, add `skipped_subgraphs` support to `AutoTraceAdapter.__init__`:

```python
def __init__(..., skipped_desc: Optional[list] = None,
             skipped_subgraphs: Optional[list[dict]] = None, ...):
    ...
    self.skipped_modules = skipped_desc or []
    self.skipped_subgraphs = skipped_subgraphs or []
```

For `codriving`, use:

```python
skipped_subgraphs=[
    {"name": "pillar_vfe", "type": "sparse_vfe", "full_model_verdict_blocker": False},
    {"name": "scatter", "type": "scatter", "full_model_verdict_blocker": False},
    {"name": "fusion_net", "type": "channel_preserving_fusion", "full_model_verdict_blocker": False},
],
```

For `fcooper`, use:

```python
skipped_subgraphs=[
    {"name": "pillar_vfe", "type": "sparse_vfe", "full_model_verdict_blocker": False},
    {"name": "scatter", "type": "scatter", "full_model_verdict_blocker": False},
    {"name": "fusion_net", "type": "channel_preserving_fusion", "full_model_verdict_blocker": False},
],
```

For `attfuse`, use:

```python
skipped_subgraphs=[
    {"name": "pillar_vfe", "type": "sparse_vfe", "full_model_verdict_blocker": False},
    {"name": "scatter", "type": "scatter", "full_model_verdict_blocker": False},
    {"name": "fusion_net", "type": "attention_or_fusion", "full_model_verdict_blocker": True},
],
```

For `v2xvit`, use:

```python
skipped_subgraphs=[
    {"name": "encoder_m1", "type": "sparse_vfe", "full_model_verdict_blocker": False},
    {"name": "fusion_net", "type": "attention_or_fusion", "full_model_verdict_blocker": True},
],
```

For Pyramid lidar/camera, label sparse encoders and geometry projection as:

```python
{"name": "encoder_m1", "type": "sparse_vfe", "full_model_verdict_blocker": False}
{"name": "encoder_m2", "type": "geometry_projection", "full_model_verdict_blocker": True}
```

Camera geometry projection is a blocker because it can dominate runtime and has non-conv scheduling behavior.

- [ ] **Step 5: Add manifest trace helper in graph_scan**

In `framework/stage1/graph_scan.py`, add:

```python
def _manifest_trace_block(adapter: TraceAdapter, entry_shape) -> dict:
    return {
        "entry_shape": list(entry_shape),
        "skipped_modules": list(getattr(adapter, "skipped_modules", [])),
        "skipped_subgraphs": list(getattr(adapter, "skipped_subgraphs", [])),
        "note": getattr(adapter, "trace_note", ""),
    }
```

Replace the existing manifest `"trace": {...}` assembly with:

```python
"trace": _manifest_trace_block(adapter, tuple(x.shape)),
```

- [ ] **Step 6: Run manifest field tests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: `3 passed`.

- [ ] **Step 7: Re-run one CPU scan smoke to confirm backward compatibility**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python scripts/autoscan_reproduce_check.py --models fcooper --device cpu --save-yaml
```

Expected output includes `scan_status=ok` and writes `results/autoscan_fcooper_partition.yaml`. The YAML must contain both `trace.skipped_modules` and `trace.skipped_subgraphs`.

- [ ] **Step 8: Commit Task 2**

```bash
git add framework/stage1/auto_trace.py framework/stage1/adapters.py framework/stage1/graph_scan.py framework/tests/test_stage1_manifest_predictor_fields.py results/autoscan_fcooper_partition.yaml
git commit -m "feat(stage1): add typed skipped subgraph metadata"
```

## Task 3: Add Structural Features Required By P/Q/S Mechanism Prediction

**Files:**
- Modify: `framework/stage1/graph_scan.py`
- Test: `framework/tests/test_stage1_manifest_predictor_fields.py`

- [ ] **Step 1: Add tests for per-group structural feature fields**

Append to `framework/tests/test_stage1_manifest_predictor_fields.py`:

```python
from framework.stage1.graph_scan import _structural_feature_for_members
import torch.nn as nn


def test_structural_feature_for_grouped_conv_reports_ic_bn():
    conv = nn.Conv2d(128, 128, kernel_size=3, groups=32, padding=1)
    feature = _structural_feature_for_members([("bev_encoder.block.conv", conv)], fanout_buckets=["bev_encoder"])

    assert feature["cin"] == 128
    assert feature["cout"] == 128
    assert feature["groups"] == 32
    assert feature["ic_bn"] == 4
    assert feature["op_types"] == ["Conv2d"]
    assert feature["kernel"] == [3, 3]
    assert feature["fanout_buckets"] == ["bev_encoder"]


def test_structural_feature_for_standard_conv_reports_large_ic_bn():
    conv = nn.Conv2d(64, 128, kernel_size=3, groups=1, padding=1)
    feature = _structural_feature_for_members([("backbone.conv", conv)], fanout_buckets=["backbone"])

    assert feature["cin"] == 64
    assert feature["cout"] == 128
    assert feature["groups"] == 1
    assert feature["ic_bn"] == 64
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: FAIL because `_structural_feature_for_members` is missing.

- [ ] **Step 3: Implement structural feature extraction**

Add to `framework/stage1/graph_scan.py`:

```python
def _structural_feature_for_members(members, fanout_buckets=None) -> dict:
    convs = [m for _, m in members if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    linears = [m for _, m in members if isinstance(m, nn.Linear)]
    root = convs[0] if convs else (linears[0] if linears else None)

    op_types = sorted({type(m).__name__ for _, m in members})
    if isinstance(root, (nn.Conv2d, nn.ConvTranspose2d)):
        cin = int(root.in_channels)
        cout = int(root.out_channels)
        groups = int(getattr(root, "groups", 1))
        kernel = list(root.kernel_size if isinstance(root.kernel_size, tuple) else (root.kernel_size, root.kernel_size))
        stride = list(root.stride if isinstance(root.stride, tuple) else (root.stride, root.stride))
    elif isinstance(root, nn.Linear):
        cin = int(root.in_features)
        cout = int(root.out_features)
        groups = 1
        kernel = [1, 1]
        stride = [1, 1]
    else:
        cin = 0
        cout = 0
        groups = 1
        kernel = [1, 1]
        stride = [1, 1]

    ic_bn = float(cin) / max(1, groups)
    return {
        "cin": cin,
        "cout": cout,
        "groups": groups,
        "ic_bn": ic_bn,
        "op_types": op_types,
        "kernel": kernel,
        "stride": stride,
        "fanout_buckets": sorted(set(fanout_buckets or [])),
    }
```

- [ ] **Step 4: Attach structural features to each raw B1 group**

Inside `extract_prune_groups()`, after `coupled = ...`, add:

```python
feature = _structural_feature_for_members(members, fanout_buckets=coupled)
```

Then add this field to each group dictionary:

```python
"feature": feature,
```

- [ ] **Step 5: Attach aggregate structural features to consolidated search groups**

In `consolidate_search_groups()`, add a helper:

```python
def _merge_structural_features(groups: list[dict]) -> dict:
    features = [g.get("feature", {}) for g in groups if g.get("feature")]
    if not features:
        return {}
    ic_bn_values = [float(f.get("ic_bn", 9999.0)) for f in features]
    return {
        "min_ic_bn": min(ic_bn_values),
        "median_ic_bn": sorted(ic_bn_values)[len(ic_bn_values) // 2],
        "max_groups": max(int(f.get("groups", 1)) for f in features),
        "op_types": sorted({op for f in features for op in f.get("op_types", [])}),
        "fanout_buckets": sorted({b for f in features for b in f.get("fanout_buckets", [])}),
    }
```

Add to each consolidated search group:

```python
"feature": _merge_structural_features(gs),
```

where `gs` is the list of raw groups in that consolidated search group.

- [ ] **Step 6: Update predictor to read aggregate fields**

Modify `_risk_from_group()` in `framework/stage1/coupling_predictor.py`:

```python
feature = group.get("feature", {}) or {}
ic_bn = float(feature.get("min_ic_bn", feature.get("ic_bn", 9999.0)))
grouped = bool(group.get("grouped_conv", False)) or int(feature.get("max_groups", 1)) > 1
fanout = feature.get("fanout_buckets", []) or []
```

Replace existing fanout access with `fanout`.

- [ ] **Step 7: Run tests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage1_manifest_predictor_fields.py framework/tests/test_coupling_predictor_static.py
```

Expected: all tests pass.

- [ ] **Step 8: Regenerate current partition manifests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python scripts/autoscan_reproduce_check.py --models all --device cpu --save-yaml
```

Expected:

- A0 models remain `MATCH`.
- F-Cooper and AttFuse remain `scan_status=ok`.
- New YAML files contain `view_b1_search_groups[*].feature`.

- [ ] **Step 9: Commit Task 3**

```bash
git add framework/stage1/graph_scan.py framework/stage1/coupling_predictor.py framework/tests/test_stage1_manifest_predictor_fields.py framework/tests/test_coupling_predictor_static.py results/autoscan_*_partition.yaml
git commit -m "feat(stage1): add structural coupling features to manifests"
```

## Task 4: Add Coverage-Aware Full-Model Verdict Guardrails

**Files:**
- Modify: `framework/stage1/latency_profile.py`
- Modify: `framework/stage1/coupling_predictor.py`
- Test: `framework/tests/test_coupling_predictor_static.py`

- [ ] **Step 1: Add tests for latency coverage guardrails**

Append to `framework/tests/test_coupling_predictor_static.py`:

```python
def test_unknown_latency_coverage_prevents_full_model_low_risk_for_skipped_fusion():
    report = predict_manifest(FIXTURES / "attfuse_skipped_fusion.yaml")

    assert report["scope"] == "traced_dense_subgraph"
    assert report["verdict"] == "FUSION_UNCOVERED_UNKNOWN"


def test_traced_coverage_allows_predicted_low_risk_without_blocking_skips(tmp_path):
    source = (FIXTURES / "codriving_standard.yaml").read_text()
    source = source.replace("traced_latency_pct: null", "traced_latency_pct: 0.94")
    source = source.replace("skipped_latency_pct: null", "skipped_latency_pct: 0.06")
    path = tmp_path / "covered_codriving.yaml"
    path.write_text(source)

    report = predict_manifest(path)

    assert report["verdict"] == "MEASURED_SEPARABLE"
    assert report["risk"]["risk_uncovered"] <= 0.20
```

- [ ] **Step 2: Run tests before implementation**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py
```

Expected: tests may pass for the current fixture but do not yet validate latency coverage from real manifests. Continue with implementation to make coverage explicit.

- [ ] **Step 3: Add coverage schema to latency sidecar**

In `framework/stage1/latency_profile.py`, when building `view_latency`, add:

```python
"coverage": {
    "traced_latency_pct": 1.0 if status in {"ok", "ok_but_contended", "estimated_cpu"} else None,
    "skipped_latency_pct": 0.0 if status in {"ok", "ok_but_contended", "estimated_cpu"} else None,
    "coverage_note": "trace-net leaves only; skipped full-model subgraphs accounted separately by trace.skipped_subgraphs",
},
```

This is intentionally conservative: trace-net coverage is 100% for the trace-net, not the full model. Full-model blockers still come from `trace.skipped_subgraphs`.

- [ ] **Step 4: Update predictor uncovered-risk calculation with coverage**

In `framework/stage1/coupling_predictor.py`, add:

```python
def _coverage_risk(manifest: dict[str, Any]) -> tuple[float, list[str]]:
    latency = manifest.get("view_latency", {}) or {}
    coverage = latency.get("coverage", {}) or {}
    traced = coverage.get("traced_latency_pct")
    skipped = coverage.get("skipped_latency_pct")
    if traced is None or skipped is None:
        return 0.35, ["latency coverage unknown"]
    skipped_f = float(skipped)
    if skipped_f >= 0.30:
        return 0.75, [f"skipped latency pct is high: {skipped_f:.2f}"]
    if skipped_f >= 0.10:
        return 0.45, [f"skipped latency pct is moderate: {skipped_f:.2f}"]
    return 0.05, [f"traced latency pct is {float(traced):.2f}"]
```

Then inside `predict_manifest()` merge it:

```python
coverage_risk, coverage_evidence = _coverage_risk(manifest)
risk_uncovered = max(risk_uncovered, coverage_risk)
evidence.extend(coverage_evidence)
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add framework/stage1/latency_profile.py framework/stage1/coupling_predictor.py framework/tests/test_coupling_predictor_static.py
git commit -m "feat(stage1): guard coupling predictions with coverage risk"
```

## Task 5: Add Low-Cost Anchor Probe Interface

**Files:**
- Create: `framework/stage1/coupling_probe.py`
- Modify: `framework/stage1/coupling_predictor.py`
- Create: `framework/tests/test_coupling_probe.py`

- [ ] **Step 1: Write probe tests using deterministic fake signatures**

Create `framework/tests/test_coupling_probe.py`:

```python
from framework.stage1.coupling_probe import schedule_signature_risk


def test_schedule_signature_risk_low_when_argmin_stable():
    signatures = {
        ("base", "fp16"): {"argmin_template": "wmma_native", "valid_templates": ["wmma_native"]},
        ("min", "fp16"): {"argmin_template": "wmma_native", "valid_templates": ["wmma_native"]},
        ("base", "int8"): {"argmin_template": "wmma_native", "valid_templates": ["wmma_native"]},
        ("min", "int8"): {"argmin_template": "wmma_native", "valid_templates": ["wmma_native"]},
    }

    risk = schedule_signature_risk(signatures)

    assert risk["risk_ps"] == 0.0
    assert risk["risk_qs"] == 0.0
    assert risk["risk_p_hub"] == 0.0


def test_schedule_signature_risk_high_when_p_changes_argmin():
    signatures = {
        ("base", "fp16"): {"argmin_template": "wmma_native", "valid_templates": ["wmma_native"]},
        ("min", "fp16"): {"argmin_template": "padded_wmma", "valid_templates": ["padded_wmma", "scalar"]},
        ("base", "int8"): {"argmin_template": "wmma_native", "valid_templates": ["wmma_native"]},
        ("min", "int8"): {"argmin_template": "nchw_fallback", "valid_templates": ["nchw_fallback"]},
    }

    risk = schedule_signature_risk(signatures)

    assert risk["risk_ps"] >= 0.70
    assert risk["risk_p_hub"] >= 0.70
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_probe.py
```

Expected: FAIL because `framework.stage1.coupling_probe` does not exist.

- [ ] **Step 3: Implement probe signature risk calculation**

Create `framework/stage1/coupling_probe.py`:

```python
from __future__ import annotations

from typing import Any


def _argmin(signatures: dict[tuple[str, str], dict[str, Any]], p: str, q: str) -> str | None:
    item = signatures.get((p, q), {}) or {}
    value = item.get("argmin_template")
    return str(value) if value is not None else None


def schedule_signature_risk(signatures: dict[tuple[str, str], dict[str, Any]]) -> dict[str, float]:
    base_fp16 = _argmin(signatures, "base", "fp16")
    min_fp16 = _argmin(signatures, "min", "fp16")
    base_int8 = _argmin(signatures, "base", "int8")
    min_int8 = _argmin(signatures, "min", "int8")

    risk_ps = 0.0
    if base_fp16 and min_fp16 and base_fp16 != min_fp16:
        risk_ps = max(risk_ps, 0.75)
    if base_int8 and min_int8 and base_int8 != min_int8:
        risk_ps = max(risk_ps, 0.75)

    risk_qs = 0.0
    if base_fp16 and base_int8 and base_fp16 != base_int8:
        risk_qs = max(risk_qs, 0.55)
    if min_fp16 and min_int8 and min_fp16 != min_int8:
        risk_qs = max(risk_qs, 0.55)

    risk_p_hub = max(risk_ps, risk_qs if risk_ps > 0 else 0.0)
    return {
        "risk_ps": round(risk_ps, 4),
        "risk_qs": round(risk_qs, 4),
        "risk_p_hub": round(risk_p_hub, 4),
    }
```

- [ ] **Step 4: Document probe budget boundaries in module docstring**

Add this docstring to `framework/stage1/coupling_probe.py`:

```python
"""Low-cost coupling probes for Stage1 separability prediction.

This module is intentionally not a full three-arm experiment. A probe records
schedule signatures for a small anchor set:

  P anchors: base width and most aggressive legal width near an alignment edge.
  Q anchors: fp16 and int8.
  S signature: valid template names and chosen argmin template from static or
  compile-light inspection.

The probe may later call TVM compile-light helpers, but the public contract is
the signature dictionary tested here. Full MetaSchedule grids and multi-seed
joint-vs-serial HV are outside this module.
"""
```

- [ ] **Step 5: Wire optional probe risk into predictor**

The initial `predict_manifest(path, probe=None)` already merges `risk_qs`, `risk_ps`, and `risk_p_hub` from the `probe` dictionary. Keep this interface. Add a test in `framework/tests/test_coupling_predictor_static.py`:

```python
def test_probe_risk_can_upgrade_standard_conv_to_low_confidence():
    report = predict_manifest(
        FIXTURES / "codriving_standard.yaml",
        probe={"risk_ps": 0.8, "risk_p_hub": 0.8},
    )

    assert report["verdict"] == "MEASURED_SEPARABLE"


def test_probe_risk_changes_unmeasured_standard_conv_to_p_hub(tmp_path):
    source = (FIXTURES / "codriving_standard.yaml").read_text()
    source = source.replace(
        "measured_coupling:\n  source: results/coupling_map/C0c_codriving_pqs.json\n  joint_vs_serial: serial_matches_joint\n  rank_flip_pairs: 0\n",
        "",
    )
    path = tmp_path / "unmeasured_standard.yaml"
    path.write_text(source)

    report = predict_manifest(path, probe={"risk_ps": 0.8, "risk_p_hub": 0.8})

    assert report["verdict"] == "P_HUB_COUPLED"
```

- [ ] **Step 6: Run probe and predictor tests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_probe.py framework/tests/test_coupling_predictor_static.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add framework/stage1/coupling_probe.py framework/tests/test_coupling_probe.py framework/tests/test_coupling_predictor_static.py
git commit -m "feat(stage1): add low-cost coupling probe signatures"
```

## Task 6: Add CLI For Batch Prediction Reports

**Files:**
- Create: `scripts/stage1_predict_coupling.py`
- Test: `framework/tests/test_coupling_predictor_cli.py`

- [ ] **Step 1: Write CLI tests**

Create `framework/tests/test_coupling_predictor_cli.py`:

```python
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/jichengzhi/V2X")
FIXTURE = ROOT / "framework/tests/fixtures/coupling_predictor/attfuse_skipped_fusion.yaml"


def test_stage1_predict_coupling_cli_writes_json(tmp_path):
    out = tmp_path / "report.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/stage1_predict_coupling.py"),
        "--manifest",
        str(FIXTURE),
        "--out-json",
        str(out),
    ]

    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text())
    assert report["attfuse_fixture"]["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
```

- [ ] **Step 2: Run CLI test and verify failure**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_cli.py
```

Expected: FAIL because `scripts/stage1_predict_coupling.py` is missing.

- [ ] **Step 3: Implement CLI**

Create `scripts/stage1_predict_coupling.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/home/jichengzhi/V2X")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage1.coupling_predictor import predict_manifest


def _markdown_report(reports: dict[str, dict]) -> str:
    lines = ["# Stage1 Coupling Prediction Report", ""]
    for model, report in reports.items():
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"- verdict: `{report['verdict']}`")
        lines.append(f"- scope: `{report['scope']}`")
        lines.append(f"- confidence: `{report['confidence']}`")
        risk = report["risk"]
        lines.append(
            "- risk: "
            f"PQ={risk['risk_pq']}, PS={risk['risk_ps']}, QS={risk['risk_qs']}, "
            f"P-hub={risk['risk_p_hub']}, uncovered={risk['risk_uncovered']}"
        )
        lines.append(f"- recommended_action: `{report['recommended_action']}`")
        lines.append("")
        lines.append("Evidence:")
        for item in report.get("evidence", []):
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True, help="Stage1 partition YAML path. Repeatable.")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md")
    args = parser.parse_args()

    reports = {}
    for manifest in args.manifest:
        report = predict_manifest(manifest)
        reports[str(report["model"])] = report

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False))

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_markdown_report(reports))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_cli.py
```

Expected: `1 passed`.

- [ ] **Step 5: Run CLI on current manifests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage1_predict_coupling.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --manifest framework/partitions/codriving_partition.yaml \
  --manifest framework/partitions/v2xvit_partition.yaml \
  --manifest framework/partitions/fcooper_partition.yaml \
  --manifest framework/partitions/attfuse_partition.yaml \
  --out-json results/stage1_coupling_predictions.json \
  --out-md results/stage1_coupling_predictions.md
```

Expected:

- Pyramid: `P_HUB_COUPLED`
- CoDriving: `ANCHOR_PROBED_LOW_RISK` for the measured ResNet envelope only, or `MEASURED_SEPARABLE` only when a same-scope real joint-vs-serial evidence file is attached.
- V2X-ViT: `JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND` until Q-granularity x P and routing/fusion evidence are attached.
- F-Cooper: `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` until MaxFusion coverage and at least one BaseBEVBackbone standard-conv anchor probe support dense-subgraph low risk.
- AttFuse: `FUSION_UNCOVERED_UNKNOWN`

- [ ] **Step 6: Commit Task 6**

```bash
git add scripts/stage1_predict_coupling.py framework/tests/test_coupling_predictor_cli.py results/stage1_coupling_predictions.json results/stage1_coupling_predictions.md
git commit -m "feat(stage1): add coupling prediction report CLI"
```

## Task 7: Calibrate The Predictor Against Existing Evidence

**Files:**
- Create: `multi_agent/methods/design/stage1-model-predict/calibration_protocol_v1.md`
- Modify: `framework/stage1/coupling_predictor.py`
- Test: `framework/tests/test_coupling_predictor_static.py`

- [ ] **Step 1: Write calibration protocol**

Create `multi_agent/methods/design/stage1-model-predict/calibration_protocol_v1.md`:

```markdown
# Stage1 Coupling Predictor Calibration Protocol v1

## Purpose

Calibrate the low-cost predictor against existing measured cells without turning prediction into full enumeration.

## Positive And Negative Anchors

| Model / cell | Expected predictor behavior | Evidence |
|---|---|---|
| Pyramid lidar/camera | `P_HUB_COUPLED` | P-hub, C2/C7, grouped bottleneck IC_BN signal |
| CoDriving | `MEASURED_SEPARABLE` only when same-scope real measured file is attached; otherwise `ANCHOR_PROBED_LOW_RISK` or `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` depending on anchor coverage | C0c', C1cod, C6 |
| V2X-ViT | `JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND` until Q-granularity x P and routing/fusion evidence are attached | fusion 92.4%, C4, C5 |
| F-Cooper | `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` until standard-conv anchor probes and MaxFusion coverage exist; dense-subgraph low risk can be reported only at dense-subgraph scope | standard conv backbone, MaxFusion low risk is not yet measured in this predictor framework |
| AttFuse | `FUSION_UNCOVERED_UNKNOWN` until fusion anchor probe/profile exists | attention fusion skipped |

## Metrics

- No false full-model separable verdict when `trace.skipped_subgraphs[*].full_model_verdict_blocker=true`.
- Pyramid P-hub score >= 0.70.
- Standard-conv dense backbone P-hub score <= 0.30 without probe risk.
- Measured evidence overrides static prediction only for `MEASURED_SEPARABLE`.
- Probe risk can upgrade an unmeasured model to coupled or low-confidence.

## Budget Bound

Allowed per new model:

- One Stage1 auto-scan.
- One latency-profile pass if GPU is idle.
- At most two P anchors per hot block.
- At most two Q anchors per hot block.
- No full joint-vs-serial multi-seed search inside prediction.
```

- [ ] **Step 2: Add calibration threshold constants**

In `framework/stage1/coupling_predictor.py`, add constants near the top:

```python
P_HUB_THRESHOLD = 0.60
LOW_RISK_THRESHOLD = 0.30
BLOCKING_UNCOVERED_THRESHOLD = 0.70
```

Replace literal `0.60`, `0.30`, and high uncovered checks with these constants.

- [ ] **Step 3: Add regression tests for expected current-model behavior**

Append to `framework/tests/test_coupling_predictor_static.py`:

```python
def test_threshold_constants_keep_pyramid_fixture_above_p_hub_cutoff():
    report = predict_manifest(FIXTURES / "pyramid_p_hub.yaml")

    assert report["verdict"] == "P_HUB_COUPLED"
    assert report["risk"]["risk_p_hub"] >= 0.60


def test_standard_conv_without_measured_evidence_is_predicted_not_measured(tmp_path):
    source = (FIXTURES / "codriving_standard.yaml").read_text()
    source = source.replace(
        "measured_coupling:\n  source: results/coupling_map/C0c_codriving_pqs.json\n  joint_vs_serial: serial_matches_joint\n  rank_flip_pairs: 0\n",
        "",
    )
    path = tmp_path / "unmeasured_standard.yaml"
    path.write_text(source)

    report = predict_manifest(path)

    assert report["verdict"] == "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE"
    assert report["recommended_action"] == "run_targeted_anchor_probe_before_any_low_risk_verdict"
```

- [ ] **Step 4: Run calibration tests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add framework/stage1/coupling_predictor.py framework/tests/test_coupling_predictor_static.py multi_agent/methods/design/stage1-model-predict/calibration_protocol_v1.md
git commit -m "docs(stage1): define coupling predictor calibration protocol"
```

## Task 8: Replace Architecture-Level Bridge Verdict Usage

**Files:**
- Modify: `framework/stage1_bridge.py`
- Modify: `framework/search_three_arm.py`
- Modify: `framework/run_pqs_ablation.py`
- Test: `framework/tests/test_coupling_predictor_static.py`

- [ ] **Step 1: Add bridge compatibility test**

Append to `framework/tests/test_coupling_predictor_static.py`:

```python
def test_bridge_summary_wording_does_not_claim_measured_separable_for_static_only():
    from framework.stage1_bridge import SpaceSpec

    spec = SpaceSpec.from_manifest(FIXTURES / "attfuse_skipped_fusion.yaml")
    summary = spec.coupling_summary()

    assert summary["architecture_verdict"] != "MEASURED_SEPARABLE"
```

- [ ] **Step 2: Modify bridge wording**

In `framework/stage1_bridge.py`, change `coupling_summary()` verdict strings:

```python
"architecture_verdict": (
    "STRUCTURAL_COUPLING_RISK (部分旋钮建议联合搜)" if n_joint
    else "STRUCTURAL_LOW_CLIFF_RISK (非全模型可分离证明)"
),
```

Keep `n_joint`, `dispatch_plan`, and risk scores for backward compatibility.

- [ ] **Step 3: Update search logging**

In `framework/search_three_arm.py`, replace comments and print messages that equate no cliff with separability. For example, change:

```python
# No cliff (separable, e.g. CoDriving/V2X-ViT)
```

to:

```python
# No int8 buildability cliff in the traced dense search space.
# This is not a full-model separability verdict; use coupling_predictor for that.
```

- [ ] **Step 4: Update run_pqs_ablation bridge printout**

In `framework/run_pqs_ablation.py`, when printing dispatch, add:

```python
print("  [dispatch] 注意: bridge dispatch is structural budget guidance, not a measured architecture separability verdict.")
```

- [ ] **Step 5: Run tests**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py
```

Expected: all tests pass.

- [ ] **Step 6: Smoke old bridge path**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X python -m framework.stage1_bridge
```

Expected: It still prints per-model knob tables and does not crash.

- [ ] **Step 7: Commit Task 8**

```bash
git add framework/stage1_bridge.py framework/search_three_arm.py framework/run_pqs_ablation.py framework/tests/test_coupling_predictor_static.py
git commit -m "refactor(stage1): separate structural dispatch from separability verdicts"
```

## Task 9: End-To-End Validation On Current Models

**Files:**
- Generate: `results/stage1_coupling_predictions.json`
- Generate: `results/stage1_coupling_predictions.md`
- Create: `multi_agent/methods/design/stage1-model-predict/validation_report_v1.md`

- [ ] **Step 1: Re-run current Stage1 scans**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python scripts/autoscan_reproduce_check.py --models all --device cpu --save-yaml
```

Expected:

- `codriving`, `pyramid_lidar`, and `v2xvit` remain `MATCH`.
- `fcooper` and `attfuse` remain `scan_status=ok`.
- The regenerated manifests include structural features and typed skipped subgraphs.

- [ ] **Step 2: Generate prediction reports**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage1_predict_coupling.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --manifest framework/partitions/pyramid_camera_partition.yaml \
  --manifest framework/partitions/codriving_partition.yaml \
  --manifest framework/partitions/v2xvit_partition.yaml \
  --manifest framework/partitions/fcooper_partition.yaml \
  --manifest framework/partitions/attfuse_partition.yaml \
  --out-json results/stage1_coupling_predictions.json \
  --out-md results/stage1_coupling_predictions.md
```

Expected report:

- Pyramid lidar/camera: `P_HUB_COUPLED`.
- CoDriving: `ANCHOR_PROBED_LOW_RISK` for measured anchor scope, or `MEASURED_SEPARABLE` only when full same-scope measured evidence has been embedded.
- V2X-ViT: `JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND`.
- F-Cooper: no stronger than `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` at full-model scope until standard-conv anchor probes and MaxFusion coverage are attached.
- AttFuse: `FUSION_UNCOVERED_UNKNOWN`.

- [ ] **Step 3: Write validation report**

Create `multi_agent/methods/design/stage1-model-predict/validation_report_v1.md`:

```markdown
# Stage1 Coupling Predictor Validation Report v1

## Command

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage1_predict_coupling.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --manifest framework/partitions/pyramid_camera_partition.yaml \
  --manifest framework/partitions/codriving_partition.yaml \
  --manifest framework/partitions/v2xvit_partition.yaml \
  --manifest framework/partitions/fcooper_partition.yaml \
  --manifest framework/partitions/attfuse_partition.yaml \
  --out-json results/stage1_coupling_predictions.json \
  --out-md results/stage1_coupling_predictions.md
```

## Expected Interpretation

- Pyramid remains the positive P-hub coupling anchor.
- CoDriving remains the negative measured/separable anchor only when measured evidence is attached.
- V2X-ViT and AttFuse are not allowed to receive full-model separable verdicts while attention/fusion is skipped.
- F-Cooper remains low-confidence at full-model scope until its standard-conv dense backbone has at least one anchor probe and MaxFusion coverage is explicitly marked non-blocking.

## Acceptance

This validation passes if no model with blocking skipped subgraphs receives `MEASURED_SEPARABLE` or `PREDICTED_SEPARABLE_LOW_RISK` at full-model scope.
```

- [ ] **Step 4: Run full test subset**

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_coupling_probe.py \
  framework/tests/test_coupling_predictor_cli.py \
  framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 9**

```bash
git add results/stage1_coupling_predictions.json results/stage1_coupling_predictions.md multi_agent/methods/design/stage1-model-predict/validation_report_v1.md
git commit -m "test(stage1): validate coupling predictor on current models"
```

## Final Verification

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X pytest -q \
  framework/tests/test_coupling_predictor_static.py \
  framework/tests/test_coupling_probe.py \
  framework/tests/test_coupling_predictor_cli.py \
  framework/tests/test_stage1_manifest_predictor_fields.py
```

Expected: all tests pass.

Run:

```bash
PYTHONPATH=/home/jichengzhi/V2X python scripts/stage1_predict_coupling.py \
  --manifest framework/partitions/pyramid_lidar_partition.yaml \
  --manifest framework/partitions/codriving_partition.yaml \
  --manifest framework/partitions/v2xvit_partition.yaml \
  --manifest framework/partitions/fcooper_partition.yaml \
  --manifest framework/partitions/attfuse_partition.yaml \
  --out-json results/stage1_coupling_predictions.json \
  --out-md results/stage1_coupling_predictions.md
```

Expected:

- No skipped-attention model is reported as full-model separable.
- Pyramid is flagged as P-hub coupled.
- Standard-conv dense backbones are low-risk predictions, not measured claims.
- The report contains evidence lines explaining each verdict.

## Self-Review Checklist

- Spec coverage: the plan adds manifest coverage, typed skip metadata, P/Q/S mechanism features, optional anchor probes, calibrated verdicts, and reporting.
- Placeholder scan: no unresolved placeholder labels are used; all steps include concrete files, commands, and expected outcomes.
- Type consistency: predictor API is `predict_manifest(path, probe=None)` throughout; probe API is `schedule_signature_risk(signatures)` throughout.
- Scope control: no task performs full P x Q x S enumeration or multi-seed joint-vs-serial search as part of prediction.
