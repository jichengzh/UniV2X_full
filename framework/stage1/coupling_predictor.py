#!/usr/bin/env python3
"""Safe Stage1 coupling predictor v0.

This module is deliberately conservative. Static manifest facts can add
blockers or request probes, but static-only facts cannot produce a model-level
separability verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from framework.stage1.trace_plan import legacy_trace_plan_from_manifest


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = ROOT / "results/stage1_model_predict"
COUPLING_MAP_PATH = ROOT / "results/coupling_map_matrix.json"
BACKEND_POLICY = {
    "default_backend": "h800_tvm",
    "default_new_measurement_backend": "h800_tvm",
    "allowed_new_measurement_backends": ["h800_tvm"],
    "historical_evidence_backends": ["trt"],
    "historical_trt_evidence_only": True,
}
S2_5_COVERAGE_REPORT = (
    "results/stage1_model_predict/s2_5_coverage_gates/"
    "stage1_s2_5_coverage_gate_closure_v1.json"
)
S3_QUANT_REPORT = (
    "results/stage1_model_predict/s3_quant_sensitivity/"
    "stage1_s3_quant_sensitivity_v1.json"
)

VERDICT_MEASURED_SEPARABLE = "MEASURED_SEPARABLE"
VERDICT_PREDICTED_SEPARABLE_LOW_RISK = "PREDICTED_SEPARABLE_LOW_RISK"
VERDICT_ANCHOR_PROBED_LOW_RISK = "ANCHOR_PROBED_LOW_RISK"
VERDICT_MEASURED_ENVELOPE_ONLY = "MEASURED_ENVELOPE_ONLY"
VERDICT_LOW_CONFIDENCE = "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE"
VERDICT_FUSION_UNCOVERED = "FUSION_UNCOVERED_UNKNOWN"
VERDICT_V2XVIT_GATE = "JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND"
VERDICT_PYRAMID_P_HUB = "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION"

SEPARABLE_VERDICTS = {
    VERDICT_MEASURED_SEPARABLE,
    VERDICT_PREDICTED_SEPARABLE_LOW_RISK,
    VERDICT_ANCHOR_PROBED_LOW_RISK,
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a mapping: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _load_evidence(evidence_dir: Path | str | None) -> dict[str, Any]:
    evidence_dir = Path(evidence_dir or DEFAULT_EVIDENCE_DIR)
    return {
        "evidence_dir": evidence_dir,
        "census": _load_json(evidence_dir / "standard_conv_census_v1.json"),
        "probe_queue": _load_json(evidence_dir / "standard_conv_probe_queue_v1.json"),
        "anchor": _load_json(evidence_dir / "standard_conv_anchor_measured_results_v1.json"),
        "s2_audit": _load_json(evidence_dir / "s2_schedule_anchor_audit_v1.json"),
        "s2_5": _load_json(
            evidence_dir
            / "s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json"
        ),
        "s3_quant": _load_json(
            evidence_dir
            / "s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json"
        ),
        "coupling_map": _load_json(COUPLING_MAP_PATH),
    }


def _model_gate(evidence: dict[str, Any], model: str) -> dict[str, Any]:
    gates = evidence.get("probe_queue", {}).get("verdict_gates", {}) or {}
    if model in gates:
        return dict(gates.get(model, {}) or {})
    try:
        from framework.stage1.standard_conv_probe_plan import _verdict_gates

        return dict(_verdict_gates().get(model, {}) or {})
    except Exception:
        return {}


def _required_gates(gate: dict[str, Any]) -> list[str]:
    values = []
    values.extend(_as_list(gate.get("required_before_full_model_separable")))
    values.extend(_as_list(gate.get("required_before_broader_standard_conv_claim")))
    return [str(item) for item in values if item]


def _probe_evidence_level(evidence: dict[str, Any], probe_id: str) -> str | None:
    for probe in _as_list(evidence.get("probe_queue", {}).get("probes")):
        if isinstance(probe, dict) and probe.get("probe_id") == probe_id:
            value = probe.get("evidence_level")
            return str(value) if value else None
    return None


def _s2_5_gate(evidence: dict[str, Any], gate_id: str) -> dict[str, Any]:
    gate = evidence.get("s2_5", {}).get("gates", {}).get(gate_id, {})
    return dict(gate) if isinstance(gate, dict) else {}


def _s3_model_evidence(evidence: dict[str, Any], model_key: str) -> dict[str, Any]:
    item = evidence.get("s3_quant", {}).get("models", {}).get(model_key, {})
    return dict(item) if isinstance(item, dict) else {}


def _s2_anchor_result(evidence: dict[str, Any], probe_id: str) -> dict[str, Any]:
    anchors = (
        evidence.get("s2_audit", {})
        .get("h800_anchor_scan", {})
        .get("anchors", [])
    )
    for item in _as_list(anchors):
        if isinstance(item, dict) and item.get("probe_id") == probe_id:
            return dict(item)
    return {}


def _census_model(evidence: dict[str, Any], model: str) -> dict[str, Any]:
    for item in _as_list(evidence.get("census", {}).get("models")):
        if isinstance(item, dict) and item.get("model") == model:
            return item
    return {}


def _skip_name(raw: str) -> str:
    return raw.split("(", 1)[0].strip() or raw.strip()


def _is_channel_preserving_skip(text: str) -> bool:
    return any(
        key in text
        for key in (
            "0-param",
            "channel-preserving",
            "maxfusion",
            "max pooling",
            "warp_affine",
        )
    )


def _skip_type_and_blocker(model: str, raw_type: str, name: str) -> tuple[str, bool, str | None]:
    text = f"{raw_type} {name}".lower()
    channel_preserving = _is_channel_preserving_skip(text)
    if any(key in text for key in ("attention", "attfusion", "transformer", "hmsa", "mswin")):
        return "attention_or_fusion", True, "attention_fusion_coverage_anchor"
    if any(key in text for key in ("quickcumsum", "geometry", "voxel", "custom")):
        return "custom_or_geometry_projection", True, "custom_subgraph_coverage_gate"
    if "fusion" in text:
        if channel_preserving:
            gate = "maxfusion_coverage_anchor" if model == "fcooper" else None
            return "channel_preserving_or_pooling_fusion", False, gate
        return "fusion", True, "routing_fusion_coverage_anchor"
    if any(key in text for key in ("sparse", "pillar", "scatter", "vfe")):
        return "sparse_frontend", False, None
    if raw_type in {"attention_or_transformer"}:
        return "attention_or_fusion", True, "attention_fusion_coverage_anchor"
    return raw_type or "unknown", raw_type in {"unknown"}, None


def _normalize_skipped_subgraphs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    model = str(manifest.get("model", "unknown"))
    trace = manifest.get("trace", {}) or {}
    normalized: list[dict[str, Any]] = []

    for item in _as_list(trace.get("skipped_subgraphs")):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("module") or "unknown")
        raw_type = str(item.get("type") or item.get("kind") or "unknown")
        skip_type, default_blocker, gate = _skip_type_and_blocker(model, raw_type, name)
        blocker = bool(item.get("full_model_verdict_blocker", default_blocker))
        normalized.append(
            {
                "name": name,
                "type": skip_type,
                "full_model_verdict_blocker": blocker,
                "blocker_gate": gate,
                "source": "typed",
            }
        )

    for raw in _as_list(trace.get("skipped_modules")):
        raw = str(raw)
        name = _skip_name(raw)
        skip_type, blocker, gate = _skip_type_and_blocker(model, raw, name)
        normalized.append(
            {
                "name": name,
                "type": skip_type,
                "full_model_verdict_blocker": blocker,
                "blocker_gate": gate,
                "source": "legacy_skipped_modules",
                "raw": raw,
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in normalized:
        key = (str(item["name"]), str(item["type"]))
        if key not in seen:
            deduped.append(item)
            seen.add(key)
    return deduped


def _normalize_trace_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    trace_plan = manifest.get("trace_plan")
    if isinstance(trace_plan, dict) and trace_plan.get("schema") == "stage1_trace_plan_v1":
        return dict(trace_plan)
    return legacy_trace_plan_from_manifest(manifest)


def _manifest_with_trace_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    data = dict(manifest)
    trace_plan = _normalize_trace_plan(data)
    trace = dict(data.get("trace", {}) or {})
    plan_skips = [
        dict(item)
        for item in _as_list(trace_plan.get("skipped_subgraphs"))
        if isinstance(item, dict)
    ]
    if plan_skips:
        trace["skipped_subgraphs"] = [
            *[
                dict(item)
                for item in _as_list(trace.get("skipped_subgraphs"))
                if isinstance(item, dict)
            ],
            *plan_skips,
        ]
    data["trace"] = trace
    data["trace_plan"] = trace_plan
    return data


def _blocking_skip_gates(skipped: Iterable[dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for item in skipped:
        if item.get("full_model_verdict_blocker"):
            gate = item.get("blocker_gate")
            blockers.append(str(gate or f"skipped_{item.get('type', 'unknown')}_coverage_gate"))
    return sorted(set(blockers))


def _has_grouped_conv(manifest: dict[str, Any]) -> bool:
    for group in _as_list(manifest.get("view_b1_search_groups")):
        if not isinstance(group, dict):
            continue
        feature = group.get("feature", {}) or {}
        if bool(group.get("grouped_conv")):
            return True
        if int(feature.get("max_groups", feature.get("groups", 1)) or 1) > 1:
            return True
    return False


def _latency_unknown(manifest: dict[str, Any]) -> bool:
    summary = manifest.get("search_space_summary", {}) or {}
    latency = manifest.get("view_latency", {}) or {}
    return summary.get("latency_status") == "skipped" or latency.get("status") == "skipped"


def _base_risk(model: str, manifest: dict[str, Any], skipped: list[dict[str, Any]]) -> dict[str, float]:
    has_blocking_skip = any(item.get("full_model_verdict_blocker") for item in skipped)
    grouped = _has_grouped_conv(manifest)
    latency_unknown = _latency_unknown(manifest)

    if model.startswith("pyramid"):
        return {
            "risk_pq": 0.78,
            "risk_ps": 0.70,
            "risk_qs": 0.55,
            "risk_p_hub": 0.90,
            "risk_uncovered": 0.50 if latency_unknown else 0.25,
        }
    if model == "v2xvit":
        return {
            "risk_pq": 0.72,
            "risk_ps": 0.45,
            "risk_qs": 0.68,
            "risk_p_hub": 0.25,
            "risk_uncovered": 0.88,
        }
    if model == "attfuse" or has_blocking_skip:
        return {
            "risk_pq": 0.35,
            "risk_ps": 0.35,
            "risk_qs": 0.35,
            "risk_p_hub": 0.20,
            "risk_uncovered": 0.90,
        }
    if model == "codriving":
        return {
            "risk_pq": 0.18,
            "risk_ps": 0.22,
            "risk_qs": 0.18,
            "risk_p_hub": 0.15,
            "risk_uncovered": 0.25 if latency_unknown else 0.10,
        }
    return {
        "risk_pq": 0.35,
        "risk_ps": 0.35,
        "risk_qs": 0.30,
        "risk_p_hub": 0.65 if grouped else 0.25,
        "risk_uncovered": 0.45 if latency_unknown else 0.25,
    }


def _evidence_lines(
    model: str,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    skipped: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "Static standard-conv metadata is treated as a probe selector, not a separability verdict.",
        "groups=1 is not used as a model-level separability rule.",
    ]
    census_item = _census_model(evidence, model)
    if census_item:
        lines.append(
            "standard_conv_census_v1: "
            f"{census_item.get('n_candidates', 0)} candidate groups; "
            f"skipped types={census_item.get('skipped_subgraph_types', [])}"
        )
    s2_summary = evidence.get("s2_audit", {}).get("s2_completion_summary", {})
    if s2_summary:
        lines.append(
            "S2 H800 schedule anchors: "
            f"{s2_summary.get('measured_anchor_items')} measured anchors / "
            f"{s2_summary.get('measured_anchor_cells')} cells; "
            f"coupling_signal_probe_ids={s2_summary.get('coupling_signal_probe_ids')}"
        )
    if _has_grouped_conv(manifest):
        lines.append("manifest contains grouped-conv/P-hub context that blocks local standard-conv promotion.")
    if _latency_unknown(manifest):
        lines.append("latency coverage is skipped or unknown in the manifest.")
    for item in skipped:
        marker = "blocker" if item.get("full_model_verdict_blocker") else "coverage_note"
        lines.append(f"skipped {item['name']} type={item['type']} source={item['source']} {marker}")
    return lines


def _codriving_has_anchor(evidence: dict[str, Any]) -> bool:
    anchor = evidence.get("anchor", {}) or {}
    return bool(anchor.get("codriving_fp16_dense_core") and anchor.get("codriving_pqs_and_highdim"))


def _evidence_sources(model: str, manifest_path: Path | None) -> list[str]:
    sources: list[str] = []
    if manifest_path is not None:
        sources.append(_rel(manifest_path))
    sources.extend(
        [
            "results/stage1_model_predict/standard_conv_census_v1.json",
            "results/stage1_model_predict/standard_conv_probe_queue_v1.json",
        ]
    )
    if model == "codriving":
        sources.extend(
            [
                "results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json",
                "results/stage1_model_predict/s2_schedule_anchor_audit_v1.json",
                "results/coupling_map/C0c_codriving_pqs.json",
                "results/coupling_map/C1_QxS_codriving.json",
                "results/coupling_map/C6_codriving_highdim.json",
            ]
        )
    elif model == "v2xvit":
        sources.extend(
            [
                "results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json",
                "results/stage1_model_predict/s2_schedule_anchor_audit_v1.json",
                S2_5_COVERAGE_REPORT,
                S3_QUANT_REPORT,
                "results/coupling_map_matrix.json",
                "results/coupling_map/C4_QgranxP_v2xvit.json",
                "results/coupling_map/C5_routing_v2xvit.json",
            ]
        )
    elif model.startswith("pyramid"):
        sources.extend(
            [
                "results/coupling_map_matrix.json",
                "results/coupling_map/C2_C3_pyramid.json",
                "results/coupling_map/C3_int8_same_graph.json",
                "results/coupling_map/C7_pyramid_irreducible.json",
                S3_QUANT_REPORT,
            ]
        )
    else:
        sources.extend(
            [
                "results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json",
                "results/stage1_model_predict/s2_schedule_anchor_audit_v1.json",
                S2_5_COVERAGE_REPORT,
            ]
        )
    return sorted(dict.fromkeys(sources))


def _finalize_report(
    *,
    manifest_path: Path | None,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    verdict: str,
    scope: str,
    evidence_level: str,
    blockers: list[str],
    required_next: list[str],
    risk: dict[str, float],
    evidence_lines: list[str],
    confidence: float,
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    model = str(manifest.get("model", "unknown"))
    blockers = sorted(set(str(item) for item in blockers if item))
    required_next = sorted(set(str(item) for item in required_next if item))
    if verdict == VERDICT_PREDICTED_SEPARABLE_LOW_RISK and evidence_level.startswith("static_only"):
        verdict = VERDICT_LOW_CONFIDENCE
        blockers.append("static_only_cannot_emit_predicted_separable_low_risk")
        required_next.append("run_targeted_anchor_probe_before_any_low_risk_verdict")

    return {
        "model": model,
        "manifest": _rel(manifest_path) if manifest_path else None,
        "ckpt_status": str(manifest.get("ckpt_status") or "unknown"),
        "verdict": verdict,
        "scope": scope,
        "evidence_level": evidence_level,
        "confidence": round(float(confidence), 3),
        "risk": {key: round(float(value), 3) for key, value in risk.items()},
        "blockers": blockers,
        "required_next_probe_or_gate": required_next,
        "evidence_sources": _evidence_sources(model, manifest_path),
        "evidence": evidence_lines,
        "normalized_skipped_subgraphs": skipped,
        "trace_plan": manifest.get("trace_plan"),
    }


def is_full_model_scope(scope: str | None) -> bool:
    """Return true for exact and semantically prefixed full-model scopes."""
    if not scope:
        return False
    return scope == "full_model" or scope.startswith("full_model_")


def find_overpromotions(predictions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    overpromotion = []
    for item in predictions:
        if (
            item.get("blockers")
            and is_full_model_scope(str(item.get("scope", "")))
            and item.get("verdict") in SEPARABLE_VERDICTS
        ):
            overpromotion.append(
                {
                    "model": item.get("model"),
                    "verdict": item.get("verdict"),
                    "scope": item.get("scope"),
                    "blockers": item.get("blockers"),
                }
            )
    return overpromotion


def predict_manifest(
    manifest: Path | str | dict[str, Any],
    *,
    evidence_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Predict one manifest using Safe Predictor v0 guardrails."""

    manifest_path: Path | None = None
    if isinstance(manifest, dict):
        data = manifest
    else:
        manifest_path = Path(manifest)
        data = _load_yaml(manifest_path)
    data = _manifest_with_trace_plan(data)

    model = str(data.get("model", "unknown"))
    evidence = _load_evidence(evidence_dir)
    gate = _model_gate(evidence, model)
    required = _required_gates(gate)
    skipped = _normalize_skipped_subgraphs(data)
    skip_blockers = _blocking_skip_gates(skipped)
    risk = _base_risk(model, data, skipped)
    evidence_lines = _evidence_lines(model, data, evidence, skipped)
    blockers = list(skip_blockers)
    required_next = list(required)
    confidence = 0.45
    trace_plan = data.get("trace_plan", {}) if isinstance(data.get("trace_plan"), dict) else {}
    if trace_plan.get("manual_override_used"):
        blockers.append("trace_manual_override_used_review_required")
        required_next.append("review_stage1_trace_plan_boundary")
    if str(trace_plan.get("trace_confidence")) == "low":
        blockers.append("trace_boundary_confidence_low")
        required_next.append("provide_trace_boundary_override_or_detector_plugin")
    if trace_plan.get("review_required"):
        required_next.append("review_stage1_trace_plan_boundary")
    if "missing" in str(trace_plan.get("ckpt_status") or data.get("ckpt_status") or ""):
        blockers.append("missing_checkpoint_architecture_only")
        required_next.append("provide_trained_checkpoint_for_model_scan")

    if model == "codriving" and _codriving_has_anchor(evidence):
        verdict = VERDICT_ANCHOR_PROBED_LOW_RISK
        scope = "codriving_measured_resnet_backbone_envelope"
        evidence_level = "measured_negative_anchor"
        confidence = 0.78
        blockers.extend(
            [
                "scope_limited_to_codriving_measured_resnet_backbone_envelope",
                "no_cross_model_extrapolation",
            ]
        )
        evidence_lines.append(
            "CoDriving C0c/C1cod/C6 and FP16 dense-core anchors bind only the measured CoDriving envelope."
        )
    elif model == "attfuse":
        verdict = VERDICT_FUSION_UNCOVERED
        scope = "traced_dense_subgraph"
        evidence_level = "uncovered_unknown_until_attention_integration"
        confidence = 0.84
        blockers.append("attention_fusion_coverage_anchor")
        evidence_lines.append("AttFuse attention/fusion coverage is not integrated into Stage1 v0.")
        routing_gate = _s2_5_gate(evidence, "routing_fusion_coverage_anchor")
        if routing_gate:
            evidence_lines.append(
                "S2.5 routing/fusion coverage keeps learned or attention fusion as a model-level gate; "
                f"status={routing_gate.get('status')}."
            )
    elif model == "v2xvit":
        verdict = VERDICT_V2XVIT_GATE
        scope = "full_model_gate_blocked"
        s3_v2xvit = _s3_model_evidence(evidence, "v2xvit")
        evidence_level = (
            str(s3_v2xvit.get("evidence_level"))
            if s3_v2xvit.get("evidence_level")
            else (
                _probe_evidence_level(evidence, "v2xvit_qgranularity_p_anchor")
                or "existing_measured_cell_needs_predictor_binding"
            )
        )
        confidence = 0.82
        blockers.extend(
            [
                "attention_fusion_coverage_anchor",
                "routing_fusion_coverage_anchor",
                "v2xvit_qgranularity_p_anchor",
            ]
        )
        evidence_lines.append("V2X-ViT C4 Q-granularity x P and C5 routing/fusion gates remain binding.")
        if s3_v2xvit:
            evidence_lines.append(
                "S3 V2X-ViT evidence is fake-quant plus structural routing only; "
                f"status={s3_v2xvit.get('status')}; true_trt_int8_ap="
                f"{s3_v2xvit.get('true_trt_int8_ap', {}).get('status')}."
            )
    elif model.startswith("pyramid"):
        verdict = VERDICT_PYRAMID_P_HUB
        scope = "full_model_p_hub_context"
        s3_pyramid = _s3_model_evidence(evidence, "pyramid")
        evidence_level = "historical_trt_evidence_plus_p_hub_context"
        confidence = 0.86
        blockers.append("pyramid_mixed_p_hub_context_anchor")
        evidence_lines.append("Pyramid C2/C3/C7 evidence binds local standard-conv groups to P-hub context.")
        if s3_pyramid:
            required_next = [
                gate
                for gate in required_next
                if gate != "per_stage_q_ap_sensitivity_anchor"
            ]
            evidence_lines.append(
                "S3 Pyramid historical TRT AP/latency evidence is bound only as historical evidence: "
                "forced per-stage INT8 exposes AP sensitivity, but hand-forced per-stage mixed "
                "precision is not Pareto-positive versus historical TRT automix."
            )
    elif model == "fcooper":
        verdict = VERDICT_LOW_CONFIDENCE
        maxfusion_gate = _s2_5_gate(evidence, "maxfusion_coverage_anchor")
        basebev_anchor = _s2_anchor_result(evidence, "std_basebev_backbone_schedule_anchor")
        has_maxfusion_skip = any(
            item.get("blocker_gate") == "maxfusion_coverage_anchor"
            or "maxfusion" in str(item.get("name", "")).lower()
            or "maxfusion" in str(item.get("raw", "")).lower()
            for item in skipped
        )
        if (
            has_maxfusion_skip
            and maxfusion_gate.get("status") == "STATIC_SHAPE_PROOF_PLUS_EXISTING_TIMING_BOUND"
            and basebev_anchor.get("verdict") == "COUPLING_SIGNAL_DETECTED"
        ):
            scope = "full_model_schedule_coupling_detected"
            evidence_level = "measured_s2_schedule_coupling_plus_static_maxfusion_bound"
            required_next = [
                gate
                for gate in required
                if gate not in {"maxfusion_coverage_anchor", "std_basebev_backbone_schedule_anchor"}
            ]
            blockers.extend(
                gate
                for gate in required_next
                if gate not in {"maxfusion_coverage_anchor", "std_basebev_backbone_schedule_anchor"}
            )
            required_next.append("s4_three_arm_validation_for_basebev_schedule_anchor")
            evidence_lines.append(
                "S2.5 bounds F-Cooper MaxFusion as shape-preserving with existing negligible timing, "
                "but this is F-Cooper-only and not a full-model separability pass."
            )
            evidence_lines.append(
                "S2 H800 std_basebev_backbone_schedule_anchor detected coupling; "
                f"rank_flips={basebev_anchor.get('rank_flip_count')} "
                f"gain_range=({basebev_anchor.get('gain_min')}, {basebev_anchor.get('gain_max')})."
            )
        else:
            scope = "full_model_pending_anchor_probes"
            evidence_level = "static_only_pending_anchor_probe"
            blockers.extend(required)
            evidence_lines.append("F-Cooper MaxFusion and BaseBEVBackbone anchors are not measured in v0.")
        confidence = 0.55
    elif skip_blockers:
        verdict = VERDICT_FUSION_UNCOVERED
        scope = "traced_dense_subgraph"
        evidence_level = "uncovered_unknown_until_attention_integration"
        confidence = 0.78
        evidence_lines.append("Blocking skipped subgraphs prevent full-model separability wording.")
        routing_gate = _s2_5_gate(evidence, "routing_fusion_coverage_anchor")
        if routing_gate and "routing_fusion_coverage_anchor" in set(required + skip_blockers):
            evidence_lines.append(
                "S2.5 routing/fusion coverage remains a model-level cap; "
                f"status={routing_gate.get('status')}; verdict={routing_gate.get('verdict')}."
            )
    else:
        verdict = VERDICT_LOW_CONFIDENCE
        scope = "traced_dense_subgraph"
        evidence_level = "static_only_pending_anchor_probe"
        confidence = 0.50
        if not required_next:
            required_next.append("run_targeted_anchor_probe_before_any_low_risk_verdict")
        blockers.extend(required_next)
        evidence_lines.append("No same-scope measured anchor is available for a low-risk verdict.")

    if evidence_level.startswith("static_only") and verdict == VERDICT_PREDICTED_SEPARABLE_LOW_RISK:
        blockers.append("static_only_cannot_emit_predicted_separable_low_risk")

    return _finalize_report(
        manifest_path=manifest_path,
        manifest=data,
        evidence=evidence,
        verdict=verdict,
        scope=scope,
        evidence_level=evidence_level,
        blockers=blockers,
        required_next=required_next,
        risk=risk,
        evidence_lines=evidence_lines,
        confidence=confidence,
        skipped=skipped,
    )


def predict_manifests(
    manifests: Iterable[Path | str],
    *,
    evidence_dir: Path | str | None = None,
) -> dict[str, Any]:
    evidence = _load_evidence(evidence_dir)
    reports = [
        predict_manifest(path, evidence_dir=evidence.get("evidence_dir"))
        for path in manifests
    ]
    overpromotion = find_overpromotions(reports)

    return {
        "schema": "stage1_coupling_predictions_v0",
        "evidence_dir": _rel(evidence.get("evidence_dir", DEFAULT_EVIDENCE_DIR)),
        "source_manifests": [_rel(Path(path)) for path in manifests],
        "policy": {
            "groups_1_is_not_a_separability_rule": True,
            "static_only_may_not_emit_predicted_low_risk": True,
            "codriving_anchor_does_not_cross_model_extrapolate": True,
            "blocking_skipped_subgraphs_prevent_full_model_separable": True,
        },
        "predictions": reports,
        "no_overpromotion": not overpromotion,
        "overpromotion": overpromotion,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage1 Coupling Predictions v0",
        "",
        "Safe Predictor v0 is a guardrail report. It does not run new probes and does not promote static-only evidence to model-level separability.",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- no_overpromotion: `{report.get('no_overpromotion')}`",
        "",
    ]
    for item in report.get("predictions", []):
        lines.extend(
            [
                f"## {item['model']}",
                "",
                f"- verdict: `{item['verdict']}`",
                f"- scope: `{item['scope']}`",
                f"- evidence_level: `{item['evidence_level']}`",
                f"- confidence: `{item['confidence']}`",
                f"- blockers: {', '.join(f'`{x}`' for x in item['blockers']) or '`none`'}",
                "- required_next_probe_or_gate: "
                + (", ".join(f"`{x}`" for x in item["required_next_probe_or_gate"]) or "`none`"),
                "- evidence_sources: "
                + (", ".join(f"`{x}`" for x in item.get("evidence_sources", [])) or "`none`"),
                "",
                "Evidence:",
            ]
        )
        for evidence_item in item.get("evidence", []):
            lines.append(f"- {evidence_item}")
        lines.append("")

    lines.extend(
        [
            "## No-Overpromotion Check",
            "",
            "A prediction is invalid if it has blockers, uses `scope=full_model`, and emits `MEASURED_SEPARABLE` or `PREDICTED_SEPARABLE_LOW_RISK`.",
            f"Result: `{'PASS' if report.get('no_overpromotion') else 'FAIL'}`",
            "",
        ]
    )
    return "\n".join(lines)
