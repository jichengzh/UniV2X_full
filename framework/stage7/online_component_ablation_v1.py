"""Hardware-independent contracts for the Stage7 online ablation experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from typing import Any, Mapping, Sequence

from framework.stage2.canonical_search_v3 import validate_capability_profile


SCHEMA_VERSION = "stage7_online_component_ablation_v1"
TASK_ID = "S7-PYR-TVM"
MODEL = "pyramid"
HARDWARE = "h800"
DISPATCH = "tvm_auto"
SEEDS = (20260718, 20260719, 20260720)
WIDTH_SCHEMA = ("w0", "w1", "w2")
FORBIDDEN_FEATURE_TOKENS = (
    "capability", "backend", "profile", "dispatch_key", "backend name", "capability_profile_id",
)
FORBIDDEN_LABEL_TOKENS = (
    "latency", "energy", "AP metrics", "terminal status", "failure reason", "Pareto/frontier membership",
    "current ablation results", "cache membership", "cached labels",
)
_TRUSTED_MODEL_PROVENANCE = {
    "architecture", "model_config", "model-config", "static_config", "static-config",
}


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def frozen_experiment_contracts() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return independent copies of Full and the four single-variable variants."""
    full = {
        "schema_version": SCHEMA_VERSION,
        "variant": "full",
        "task_id": TASK_ID,
        "model": MODEL,
        "hardware_id": HARDWARE,
        "dispatch_key": DISPATCH,
        "seeds": list(SEEDS),
        "batch_size": 4,
        "round_count": 4,
        "sample_budget": 16,
        "policy_name": "predicted_frontier_diversity",
        "surrogate_acquisition": "enabled",
        "feedback_refit": "enabled",
        "actual_graph_feature_feedback": "enabled",
        "backend_model_features": "enabled",
        "candidate_pool": "scan_pass",
        "scanner": "enabled",
    }
    variants = {
        "without_surrogate": {**full, "variant": "without_surrogate", "surrogate_acquisition": "uniform_random_without_replacement"},
        "without_measured_feedback": {**full, "variant": "without_measured_feedback", "feedback_refit": "frozen_initial_bundle", "actual_graph_feature_feedback": "off"},
        "backend_blind": {**full, "variant": "backend_blind", "backend_model_features": "off"},
        "without_capability_scan": {**full, "variant": "without_capability_scan", "candidate_pool": "pre_scan", "scanner": "off"},
    }
    return copy.deepcopy(full), copy.deepcopy(variants)


def audit_single_variable_isolation(full: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    """Verify variants change exactly their documented ablation switch(es)."""
    name = str(variant.get("variant") or "")
    permitted = {
        "without_surrogate": {"variant", "surrogate_acquisition"},
        "without_measured_feedback": {"variant", "feedback_refit", "actual_graph_feature_feedback"},
        "backend_blind": {"variant", "backend_model_features"},
        "without_capability_scan": {"variant", "candidate_pool", "scanner"},
    }
    if name not in permitted:
        raise ValueError("unknown Stage7 variant")
    changed = sorted({key for key in set(full) | set(variant) if full.get(key) != variant.get(key)})
    unexpected = sorted(set(changed) - permitted[name])
    if unexpected:
        raise ValueError("variant changes fields outside its ablation: " + ",".join(unexpected))
    if not set(changed) >= (permitted[name] - {"variant"}):
        raise ValueError("variant does not apply its required ablation")
    return {"variant": name, "changed_fields": changed, "verdict": "pass", "audit_sha256": _sha(changed)}


def _identity(value: Mapping[str, Any] | str) -> str:
    if isinstance(value, str):
        return value
    return str(value.get("row_id") or value.get("manifest_job_id") or "")


def a1_select_unselected(
    scan_pass_candidates: Sequence[Mapping[str, Any] | str],
    selected_ids: set[str],
    *,
    seed: int,
    batch_size: int = 4,
) -> list[str]:
    """Select a uniform, unique A1 batch without any model-facing inputs."""
    if seed not in SEEDS:
        raise ValueError("A1 seed must be one of the frozen Stage7 seeds")
    if batch_size != 4:
        raise ValueError("Stage7 A1 batch size is frozen at four")
    ids = [_identity(candidate) for candidate in scan_pass_candidates]
    if any(not candidate_id for candidate_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("scan-pass candidates require unique non-empty identities")
    available = [candidate_id for candidate_id in ids if candidate_id not in selected_ids]
    if len(available) < batch_size:
        raise ValueError("fewer than four unselected scan-pass candidates")
    return random.Random(seed).sample(available, batch_size)


def project_a2_feedback(
    initial_training_view: Sequence[Mapping[str, Any]],
    initial_graph_feature_view: Mapping[str, Mapping[str, Any]],
    *,
    anchor: Mapping[str, Any],
    bundle_sha256: str,
    selected_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record selected identities while retaining the initial A2 selection inputs."""
    if len(bundle_sha256) != 64:
        raise ValueError("A2 bundle SHA256 must be a 64-character digest")
    recorded: list[dict[str, Any]] = []
    observed: set[str] = set()
    for source in selected_results:
        row = copy.deepcopy(dict(source))
        candidate_id = _identity(row)
        if not candidate_id:
            raise ValueError("selected feedback results require an identity")
        if candidate_id not in observed:
            recorded.append(row)
            observed.add(candidate_id)
    return {
        "training_view": copy.deepcopy([dict(row) for row in initial_training_view]),
        "graph_feature_view": copy.deepcopy(dict(initial_graph_feature_view)),
        "anchor": copy.deepcopy(dict(anchor)),
        "bundle_sha256": str(bundle_sha256),
        "recorded_results": recorded,
        "feedback_refit": False,
        "actual_graph_feature_feedback": False,
    }


def _forbidden_feature(value: str) -> bool:
    lowered = value.lower().replace("_", " ")
    return any(token.replace("_", " ") in lowered for token in FORBIDDEN_FEATURE_TOKENS)


def _canonical_field(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _metric_field(name: Any) -> bool:
    field = _canonical_field(name)
    segments = tuple(segment for segment in field.split("_") if segment)
    return (
        "latency" in segments or "energy" in segments or "map" in segments
        or any(segment == "ap" or bool(re.fullmatch(r"ap\d+", segment)) for segment in segments)
    )


def _forbidden_input_field(name: Any) -> bool:
    field = _canonical_field(name)
    return (
        _metric_field(field)
        or field in {"terminal_status", "failure_reason", "current_ablation_results"}
        or field.startswith("pareto_") or field.startswith("frontier_")
        or field.startswith("cache_") or field.startswith("cached_")
    )


def _forbidden_input_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _forbidden_input_field(key):
                paths.append(path)
            paths.extend(_forbidden_input_paths(nested, path))
        return paths
    if isinstance(value, (list, tuple)):
        return [
            path
            for index, nested in enumerate(value)
            for path in _forbidden_input_paths(nested, f"{prefix}[{index}]")
        ]
    return []


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def project_backend_blind_features(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Drop capability/backend/profile-derived feature columns and audit the result."""
    full_rows: list[dict[str, float]] = []
    provenances: list[dict[str, str]] = []
    direct_identity_fields = {
        "capability_profile_id", "dispatch_key", "backend", "backend_name", "profile", "profile_id",
    }
    for source in rows:
        row = copy.deepcopy(dict(source))
        leaked = sorted(field for field in row if _canonical_field(field) in direct_identity_fields)
        if leaked:
            raise ValueError("forbidden direct identity field remains: " + ",".join(leaked))
        width = row.get("width")
        if not isinstance(width, (list, tuple)) or len(width) != 3:
            raise ValueError("backend-blind rows require three widths")
        q_mode = str(row.get("q_mode") or "")
        if q_mode not in {"fp16", "int8"}:
            raise ValueError("backend-blind rows require fp16 or int8 q_mode")
        provenance = row.get("feature_provenance") or {}
        if not isinstance(provenance, Mapping):
            raise ValueError("feature_provenance must be a mapping")
        features = {f"width:{name}": _numeric(value) for name, value in zip(WIDTH_SCHEMA, width)}
        features["q_mode"] = float(q_mode == "int8")
        for container_name, prefix in (("graph_features", "graph"), ("model_features", "model"), ("features", "feature")):
            values = row.get(container_name) or {}
            if not isinstance(values, Mapping):
                raise ValueError(f"{container_name} must be a mapping")
            for name, value in values.items():
                feature_name = str(name)
                if container_name != "graph_features":
                    provenance_value = provenance.get(feature_name, provenance.get(f"{prefix}:{feature_name}"))
                    if provenance_value is None:
                        raise ValueError(f"{container_name} feature provenance missing: {feature_name}")
                    provenance_text = str(provenance_value)
                    if not _forbidden_feature(feature_name) and not _forbidden_feature(provenance_text):
                        if _canonical_field(provenance_text) not in _TRUSTED_MODEL_PROVENANCE:
                            raise ValueError(f"{container_name} feature provenance is unknown: {feature_name}")
                features[f"{prefix}:{name}"] = _numeric(value)
        full_rows.append(features)
        provenances.append({str(key): str(value) for key, value in provenance.items()})
    full_schema = sorted({name for row in full_rows for name in row})
    removed: list[str] = []
    for name in full_schema:
        raw_name = name.split(":", 1)[-1]
        provenance_leak = any(
            _forbidden_feature(value)
            for provenance in provenances
            for key, value in provenance.items()
            if key in {raw_name, name}
        )
        if _forbidden_feature(name) or provenance_leak:
            removed.append(name)
    blind_schema = [name for name in full_schema if name not in set(removed)]
    full_matrix = [[row.get(name, 0.0) for name in full_schema] for row in full_rows]
    blind_matrix = [[row.get(name, 0.0) for name in blind_schema] for row in full_rows]
    return {
        "full_schema": full_schema,
        "blind_schema": blind_schema,
        "removed_names": removed,
        "full_matrix_sha256": _sha(full_matrix),
        "blind_matrix_sha256": _sha(blind_matrix),
        "leakage_verdict": "no_forbidden_features",
    }


def project_stage5_backend_blind_features(
    stage5_candidates: Sequence[Mapping[str, Any]], raw_profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach explicit capability-derived provenance before applying the A3 projection."""
    profile = validate_capability_profile(raw_profile)
    adapted: list[dict[str, Any]] = []
    for source in stage5_candidates:
        candidate = copy.deepcopy(dict(source))
        model_features = copy.deepcopy(dict(candidate.get("model_features") or {}))
        provenance = copy.deepcopy(dict(candidate.get("feature_provenance") or {}))
        for name, value in profile["features"].items():
            feature_name = f"capability:{name}"
            model_features[feature_name] = value
            provenance[feature_name] = "capability_profile-derived"
        for identity_field in ("capability_profile_id", "dispatch_key", "backend", "backend_name", "profile", "profile_id"):
            candidate.pop(identity_field, None)
        adapted.append({
            **candidate,
            "model_features": model_features,
            "feature_provenance": provenance,
        })
    return project_backend_blind_features(adapted)


def scanner_rule_manifest(scanner_profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe the fixed, label-free candidate-level static scanner input contract."""
    manifest = {
        "schema_version": SCHEMA_VERSION + "_scanner_rules",
        "candidate_allowed_inputs": [
            "task_id", "model", "hardware_id", "dispatch_key", "capability_profile_id", "width",
            "width_schema", "q_mode", "source_status", "source_evidence_sha256", "source_contract",
            "graph_features",
        ],
        "profile_allowed_inputs": [
            "capability_profile_id", "hardware_id", "dispatch_key", "raw_capability_digest",
            "q_mode_support", "q_mode_coverage", "static_shape_contract",
        ],
        "forbidden_inputs": list(FORBIDDEN_LABEL_TOKENS),
        "decision_values": ["pass", "reject"],
        "no_filesystem_or_historical_label_access": True,
    }
    if scanner_profile is None:
        return manifest
    bound = {
        **manifest,
        "capability_profile_id": scanner_profile["capability_profile_id"],
        "hardware_id": scanner_profile["hardware_id"],
        "dispatch_key": scanner_profile["dispatch_key"],
        "raw_capability_digest": scanner_profile["raw_capability_digest"],
        "static_shape_contract": copy.deepcopy(scanner_profile["static_shape_contract"]),
    }
    return {**bound, "rule_sha256": _sha(bound)}


def _normalize_static_shape_contract(static_rule_contract: Mapping[str, Any] | None) -> dict[str, list[int]]:
    if not isinstance(static_rule_contract, Mapping):
        raise ValueError("static shape rule contract is required")
    fields = ("width_alignment", "width_min", "width_max")
    normalized: dict[str, list[int]] = {}
    for field in fields:
        values = static_rule_contract.get(field)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("static shape rule contract is invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("static shape rule contract is invalid")
        normalized[field] = [int(value) for value in values]
    if any(normalized["width_min"][index] > normalized["width_max"][index] for index in range(3)):
        raise ValueError("static shape rule contract is invalid")
    return normalized


def derive_scanner_profile(
    raw_profile: Mapping[str, Any], static_rule_contract: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Derive label-free scanner support from an immutable formal capability profile."""
    profile = validate_capability_profile(raw_profile)
    if str(profile["hardware_target"]) != HARDWARE or str(profile["dispatch_key"]) != DISPATCH:
        raise ValueError("formal profile identity does not match the frozen Stage7 task")
    shape_contract = _normalize_static_shape_contract(static_rule_contract)
    coverage: dict[str, float] = {}
    for q_mode in ("fp16", "int8"):
        names = (f"s1p_{q_mode}_build_success_coverage", f"s1q_{q_mode}_build_success_coverage")
        values: list[float] = []
        for name in names:
            value = profile["features"].get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"formal profile missing frozen {name}")
            if float(value) < 0.0 or float(value) > 1.0:
                raise ValueError(f"formal profile invalid frozen {name}")
            values.append(float(value))
        coverage[q_mode] = min(values)
    derived = {
        "schema_version": SCHEMA_VERSION + "_scanner_profile",
        "capability_profile_id": profile["capability_profile_id"],
        "hardware_id": profile["hardware_target"],
        "dispatch_key": profile["dispatch_key"],
        "raw_capability_digest": profile["capability_digest"],
        "q_mode_support": {q_mode: coverage[q_mode] > 0.0 for q_mode in coverage},
        "q_mode_coverage": coverage,
        "static_shape_contract": shape_contract,
    }
    return {**derived, "rule_manifest": scanner_rule_manifest(derived)}


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _has_label_field(values: Mapping[str, Any]) -> bool:
    return bool(_forbidden_input_paths(values))


def _shape_contract_reasons(profile: Mapping[str, Any], width: Any) -> list[str]:
    contract = profile.get("static_shape_contract")
    if not isinstance(contract, Mapping):
        return ["missing_static_shape_contract"]
    fields = ("width_alignment", "width_min", "width_max")
    values = {field: contract.get(field) for field in fields}
    if (
        not isinstance(width, (list, tuple)) or len(width) != 3
        or any(not isinstance(values[field], (list, tuple)) or len(values[field]) != 3 for field in fields)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for field in fields for value in values[field]
        )
    ):
        return ["invalid_static_shape_contract"]
    if any(values["width_min"][index] > values["width_max"][index] for index in range(3)):
        return ["invalid_static_shape_contract"]
    reasons: list[str] = []
    if any(width[index] % values["width_alignment"][index] for index in range(3)):
        reasons.append("width_alignment_mismatch")
    if any(
        width[index] < values["width_min"][index] or width[index] > values["width_max"][index]
        for index in range(3)
    ):
        reasons.append("width_outside_static_bounds")
    return reasons


def _candidate_reasons(candidate: Mapping[str, Any], profile: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    reasons.extend(f"forbidden_input:{path}" for path in _forbidden_input_paths(candidate))
    if not str(profile.get("capability_profile_id") or ""):
        reasons.append("missing_profile_identity")
    expected = {
        "task_id": TASK_ID, "model": MODEL, "hardware_id": HARDWARE, "dispatch_key": DISPATCH,
        "capability_profile_id": profile.get("capability_profile_id"),
    }
    for field, expected_value in expected.items():
        if str(candidate.get(field) or "") != str(expected_value or ""):
            reasons.append(f"{field}_mismatch")
    if str(profile.get("hardware_id") or profile.get("hardware_target") or "") != HARDWARE:
        reasons.append("profile_hardware_mismatch")
    if str(profile.get("dispatch_key") or "") != DISPATCH:
        reasons.append("profile_dispatch_mismatch")
    width = candidate.get("width")
    if not isinstance(width, (list, tuple)) or len(width) != 3 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in width):
        reasons.append("invalid_width")
    else:
        reasons.extend(_shape_contract_reasons(profile, width))
    if tuple(candidate.get("width_schema") or ()) != WIDTH_SCHEMA:
        reasons.append("width_schema_mismatch")
    q_mode = str(candidate.get("q_mode") or "")
    if q_mode not in {"fp16", "int8"}:
        reasons.append("invalid_q_mode")
    support = profile.get("q_mode_support") or profile.get("support_by_q_mode")
    coverage = profile.get("q_mode_coverage") or profile.get("coverage_by_q_mode")
    if not isinstance(support, Mapping) or support.get(q_mode) is not True:
        reasons.append("unsupported_q_mode")
    if not isinstance(coverage, Mapping) or not coverage.get(q_mode):
        reasons.append("uncovered_q_mode")
    if candidate.get("source_status") not in {"ready", "materializable"}:
        reasons.append("invalid_source_status")
    if not _is_sha256(candidate.get("source_evidence_sha256")):
        reasons.append("missing_source_evidence_sha256")
    source_contract = candidate.get("source_contract")
    required_paths = ("checkpoint_path", "onnx_path") if str(candidate.get("model")) == MODEL else ()
    if not isinstance(source_contract, Mapping):
        reasons.append("invalid_source_contract")
    elif any(not str(source_contract.get(field) or "") for field in required_paths):
        reasons.append("missing_static_source_path")
    else:
        for field in ("checkpoint_sha256", "source_checkpoint_sha256", "onnx_sha256"):
            if source_contract.get(field) is not None and not _is_sha256(source_contract[field]):
                reasons.append("invalid_optional_source_digest")
                break
    graph = candidate.get("graph_features")
    if not isinstance(graph, Mapping):
        reasons.append("missing_graph_features")
    elif _has_label_field(graph):
        reasons.append("graph_features_contain_forbidden_label")
    return reasons


_STATIC_CANDIDATE_FIELDS = (
    "row_id", "manifest_job_id", "task_id", "model", "hardware_id", "dispatch_key",
    "capability_profile_id", "width", "width_schema", "q_mode", "source_status",
    "source_evidence_sha256", "source_contract", "graph_features",
)


def _static_candidate_projection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only scanner-approved static fields into a later selection pool."""
    return {
        field: copy.deepcopy(candidate[field])
        for field in _STATIC_CANDIDATE_FIELDS
        if field in candidate
    }


def scan_candidate_pool(
    pre_scan_candidates: Sequence[Mapping[str, Any]], *, capability_profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify candidates only from immutable source and capability support metadata."""
    profile = copy.deepcopy(dict(capability_profile))
    required_profile_fields = {
        "capability_profile_id", "hardware_id", "dispatch_key", "raw_capability_digest",
        "q_mode_support", "q_mode_coverage", "static_shape_contract", "rule_manifest",
    }
    if required_profile_fields - set(profile):
        raise ValueError("scan_candidate_pool requires an enriched scanner profile")
    if profile["rule_manifest"] != scanner_rule_manifest(profile):
        raise ValueError("scanner profile rule manifest mismatch")
    decisions: list[dict[str, Any]] = []
    observed: set[str] = set()
    for source in pre_scan_candidates:
        candidate = copy.deepcopy(dict(source))
        candidate_id = _identity(candidate)
        reasons = _candidate_reasons(candidate, profile)
        if not candidate_id or candidate_id in observed:
            reasons.append("empty_or_duplicate_identity")
        observed.add(candidate_id)
        decisions.append({
            "row_id": candidate_id, "decision": "pass" if not reasons else "reject", "reasons": sorted(reasons),
        })
    accepted = {decision["row_id"] for decision in decisions if decision["decision"] == "pass"}
    scan_pass = [
        _static_candidate_projection(row) for row in pre_scan_candidates if _identity(row) in accepted
    ]
    result = {
        "rule_manifest": copy.deepcopy(profile["rule_manifest"]),
        "scanner_profile_audit": {
            "capability_profile_id": profile["capability_profile_id"],
            "raw_capability_digest": profile["raw_capability_digest"],
            "rule_sha256": profile["rule_manifest"]["rule_sha256"],
        },
        "pre_scan_count": len(pre_scan_candidates),
        "scan_pass_count": len(scan_pass),
        "decisions": decisions,
        "scan_pass_candidates": scan_pass,
        "decision_sha256": _sha(decisions),
    }
    result["scanner_note"] = (
        "capability_scan_non_discriminative_on_frozen_pool"
        if result["pre_scan_count"] == result["scan_pass_count"]
        else "capability_scan_discriminative_on_frozen_pool"
    )
    return result


def variant_candidate_pools(
    pre_scan_candidates: Sequence[Mapping[str, Any]], scan_audit: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Expose scan-pass candidates to Full/A1/A2/A3 and the pre-scan pool to A4."""
    scan_pass = copy.deepcopy(list(scan_audit.get("scan_pass_candidates") or []))
    pre_scan = selection_candidate_view(pre_scan_candidates)
    return {
        "full": scan_pass,
        "without_surrogate": copy.deepcopy(scan_pass),
        "without_measured_feedback": copy.deepcopy(scan_pass),
        "backend_blind": copy.deepcopy(scan_pass),
        "without_capability_scan": pre_scan,
    }


CACHE_KEY_DIMENSIONS = (
    "model", "capability_profile_id", "hardware_id", "measurement_scope", "input_protocol_sha256",
    "batch_size", "genome", "q_mode", "source_checkpoint_sha256", "onnx_sha256",
    "build_protocol_sha256", "tuning_protocol_sha256", "measurement_protocol_sha256", "ap_protocol_sha256",
)
_CACHE_SHA_FIELDS = {
    "input_protocol_sha256", "source_checkpoint_sha256", "onnx_sha256", "build_protocol_sha256",
    "tuning_protocol_sha256", "measurement_protocol_sha256", "ap_protocol_sha256",
}


def build_measurement_cache_key(dimensions: Mapping[str, Any]) -> str:
    """Hash every exact measurement identity dimension; partial keys are invalid."""
    missing = [field for field in CACHE_KEY_DIMENSIONS if dimensions.get(field) in (None, "", [], {})]
    if missing:
        raise ValueError("measurement cache key missing dimensions: " + ",".join(missing))
    if any(not _is_sha256(dimensions[field]) for field in _CACHE_SHA_FIELDS):
        raise ValueError("measurement cache key requires exact SHA256 protocol dimensions")
    if not isinstance(dimensions["batch_size"], int) or dimensions["batch_size"] <= 0:
        raise ValueError("measurement cache key batch_size must be positive")
    if str(dimensions["q_mode"]) not in {"fp16", "int8"}:
        raise ValueError("measurement cache key q_mode must be fp16 or int8")
    return _sha({field: copy.deepcopy(dimensions[field]) for field in CACHE_KEY_DIMENSIONS})


def selection_candidate_view(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return selection inputs after stripping all cache state and metric labels."""
    def project(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: project(nested)
                for key, nested in value.items()
                if not _forbidden_input_field(key)
                and "cache" not in _canonical_field(key)
                and "prediction" not in _canonical_field(key)
                and "uncertainty" not in _canonical_field(key)
                and "label" not in _canonical_field(key)
            }
        if isinstance(value, list):
            return [project(nested) for nested in value]
        if isinstance(value, tuple):
            return tuple(project(nested) for nested in value)
        return copy.deepcopy(value)
    result: list[dict[str, Any]] = []
    for source in candidates:
        result.append(project(dict(source)))
    return result


def reveal_measurement_cache(
    candidate_id: str,
    selected_ids: set[str],
    dimensions: Mapping[str, Any],
    cache: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Reveal an exact cache entry only after the corresponding candidate is selected."""
    if candidate_id not in selected_ids:
        raise ValueError("cache reveal is permitted only after an ID is selected")
    cached = cache.get(build_measurement_cache_key(dimensions))
    return copy.deepcopy(dict(cached)) if cached is not None else None


def audit_cache_selection_invariance(
    candidates: Sequence[Mapping[str, Any]],
    *,
    cache: Mapping[str, Mapping[str, Any]],
    selector: Any,
) -> dict[str, Any]:
    """Prove an adapter's selected IDs do not change when cache data is populated."""
    empty_view = selection_candidate_view(candidates)
    populated_probe: list[dict[str, Any]] = []
    for original, clean in zip(candidates, empty_view):
        dimensions = original.get("measurement_cache_dimensions")
        cache_key = build_measurement_cache_key(dimensions) if isinstance(dimensions, Mapping) else None
        cached_labels = cache.get(cache_key) if cache_key is not None else None
        populated_probe.append({
            **clean,
            "cache_membership": cached_labels is not None,
            "cached_labels": copy.deepcopy(dict(cached_labels)) if cached_labels is not None else None,
        })
    empty_ids = [str(value) for value in selector(copy.deepcopy(empty_view))]
    populated_ids = [str(value) for value in selector(copy.deepcopy(populated_probe))]
    if empty_ids != populated_ids:
        raise ValueError("cache selection invariance drift")
    return {"verdict": "pass", "selected_ids": empty_ids, "audit_sha256": _sha(empty_ids)}


_BUDGET_FAILURES = {
    "backend_failure", "build_failure", "unsupported_precision", "quantization_failure",
    "numerical_failure", "candidate_runtime_capability_failure",
}
_INFRASTRUCTURE_FAILURES = {
    "gpu_unavailable", "unrelated_process_oom", "contention", "ssh_failure", "network_failure",
    "missing_source_artifact", "permission_failure", "runner_bug",
}


def classify_failure(failure_kind: str) -> dict[str, Any]:
    """Classify failure accounting without manufacturing performance measurements."""
    if failure_kind in _BUDGET_FAILURES:
        return {
            "failure_kind": failure_kind, "consumes_selected_event_budget": True,
            "same_request_retry": False, "fabricated_metrics": False,
        }
    if failure_kind in _INFRASTRUCTURE_FAILURES:
        return {
            "failure_kind": failure_kind, "consumes_selected_event_budget": False,
            "same_request_retry": True, "fabricated_metrics": False,
        }
    raise ValueError("unknown Stage7 failure kind")


def validate_result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only measured-success rows with real metric artifacts, never surrogate fills."""
    result = copy.deepcopy(dict(row))
    success_statuses = {"success", "measured_success", "measured_success_gold"}
    if result.get("terminal_status") not in success_statuses:
        forbidden_metrics = sorted(str(key) for key in result if _metric_field(key))
        if forbidden_metrics:
            raise ValueError("failure rows cannot contain latency, energy, or AP metrics/artifacts: " + ",".join(forbidden_metrics))
    for key, value in result.items():
        if any(metric in str(key).lower() for metric in ("latency", "energy", "ap")) and "source" in str(key).lower():
            if "surrogate" in str(value).lower():
                raise ValueError("surrogate-filled latency, energy, or AP is forbidden")
    if result.get("terminal_status") in success_statuses:
        for metric, artifact_field in (
            ("latency_ms", "latency_artifact_sha256"),
            ("energy_j", "energy_artifact_sha256"),
            ("ap70", "ap_artifact_sha256"),
        ):
            if not isinstance(result.get(metric), (int, float)) or isinstance(result.get(metric), bool):
                raise ValueError(f"successful result requires measured {metric}")
            if not _is_sha256(result.get(artifact_field)):
                raise ValueError(f"successful result requires {artifact_field}")
    return {"verdict": "pass", "row": result, "row_sha256": _sha(result)}
