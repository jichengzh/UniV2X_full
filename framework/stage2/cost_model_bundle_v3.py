"""Serializable baseline multi-head cost-model bundle for canonical Stage2 search."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from framework.stage2.canonical_search_v3 import validate_capability_profile


SCHEMA_VERSION = "stage2_cost_model_bundle_v3"
TARGETS = ("latency_ms", "energy_j", "ap70")


def _sha(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _profile_map(profiles: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    validated = [validate_capability_profile(profile) for profile in profiles]
    result = {profile["capability_profile_id"]: profile for profile in validated}
    if len(result) != len(validated):
        raise ValueError("duplicate capability_profile_id")
    return result


def build_feature_schema(
    rows: Sequence[Mapping[str, Any]], profiles: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    profile_by_id = _profile_map(profiles)
    graph_names = sorted(
        {
            str(name)
            for row in rows
            for name, value in (row.get("graph_features") or {}).items()
            if value is None or _finite_number(value)
        }
    )
    capability_names = sorted(
        {
            str(name)
            for profile in profile_by_id.values()
            for name in profile["features"]
        }
    )
    feature_order = [
        "width:p1",
        "width:p2",
        "width:p3",
        "q:is_int8",
        *[f"graph:{name}" for name in graph_names],
        *[f"graph_observed:{name}" for name in graph_names],
        *[f"cap:{name}" for name in capability_names],
        *[f"cap_observed:{name}" for name in capability_names],
        *[f"interaction:q_int8*cap:{name}" for name in capability_names],
        *[f"interaction:q_int8*cap_observed:{name}" for name in capability_names],
    ]
    schema = {
        "schema_version": "stage2_cost_model_feature_schema_v3",
        "feature_order": feature_order,
        "graph_feature_names": graph_names,
        "capability_feature_names": capability_names,
    }
    schema["feature_schema_sha256"] = _sha(schema)
    return schema


def _encode_row(
    row: Mapping[str, Any], schema: Mapping[str, Any], profile_by_id: Mapping[str, Mapping[str, Any]]
) -> list[float]:
    width = row.get("width")
    if not isinstance(width, (list, tuple)) or len(width) != 3:
        raise ValueError("cost-model rows require a three-element width")
    q_mode = str(row.get("q_mode") or "")
    if q_mode not in {"fp16", "int8"}:
        raise ValueError("cost-model q_mode must be fp16 or int8")
    profile_id = str(row.get("capability_profile_id") or "")
    if profile_id not in profile_by_id:
        raise ValueError(f"unknown capability_profile_id: {profile_id}")
    graph = row.get("graph_features") or {}
    capability = profile_by_id[profile_id]["features"]
    values = [float(width[0]), float(width[1]), float(width[2]), float(q_mode == "int8")]
    for name in schema["graph_feature_names"]:
        value = graph.get(name)
        values.append(float(value) if _finite_number(value) else 0.0)
    for name in schema["graph_feature_names"]:
        values.append(float(_finite_number(graph.get(name))))
    for name in schema["capability_feature_names"]:
        value = capability.get(name)
        values.append(float(value) if _finite_number(value) else 0.0)
    for name in schema["capability_feature_names"]:
        values.append(float(_finite_number(capability.get(name))))
    q_is_int8 = float(q_mode == "int8")
    for name in schema["capability_feature_names"]:
        value = capability.get(name)
        values.append(q_is_int8 * (float(value) if _finite_number(value) else 0.0))
    for name in schema["capability_feature_names"]:
        values.append(q_is_int8 * float(_finite_number(capability.get(name))))
    return values


def _fit_head(x: np.ndarray, y: np.ndarray, ridge: float) -> dict[str, Any]:
    design = np.column_stack([np.ones(x.shape[0], dtype="float64"), x])
    penalty = np.eye(design.shape[1], dtype="float64") * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    predictions = design @ coefficients
    residuals = np.abs(y - predictions)
    return {
        "model_kind": "ridge_linear_baseline",
        "coefficients": coefficients.tolist(),
        "training_rows": int(y.size),
        "residual_p90": float(np.percentile(residuals, 90.0)) if residuals.size else None,
    }


def fit_model_bundle(
    rows: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    source_rows = [copy.deepcopy(dict(row)) for row in rows]
    if not source_rows:
        raise ValueError("training rows must not be empty")
    profiles = [validate_capability_profile(profile) for profile in capability_profiles]
    profile_by_id = _profile_map(profiles)
    schema = build_feature_schema(source_rows, profiles)
    x_all = np.asarray(
        [_encode_row(row, schema, profile_by_id) for row in source_rows], dtype="float64"
    )
    heads: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        indices = [index for index, row in enumerate(source_rows) if _finite_number(row.get(target))]
        if indices:
            heads[target] = _fit_head(
                x_all[np.asarray(indices, dtype="int64")],
                np.asarray([float(source_rows[index][target]) for index in indices], dtype="float64"),
                ridge,
            )
    feasibility = np.asarray(
        [
            float(row.get("build_status") == "success" and row.get("numerical_status") == "pass")
            for row in source_rows
        ],
        dtype="float64",
    )
    heads["feasibility"] = _fit_head(x_all, feasibility, ridge)
    capability_snapshot = {
        profile["capability_profile_id"]: profile["capability_digest"] for profile in profiles
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema": schema,
        "heads": heads,
        "ridge": ridge,
        "training_row_count": len(source_rows),
        "training_data_sha256": _sha(source_rows),
        "capability_digest_by_profile": capability_snapshot,
        "training_rows": source_rows,
    }


def predict_rows(
    bundle: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected cost-model bundle schema")
    profile_by_id = _profile_map(capability_profiles)
    for profile_id, digest in bundle["capability_digest_by_profile"].items():
        if profile_id not in profile_by_id or profile_by_id[profile_id]["capability_digest"] != digest:
            raise ValueError(f"capability profile drift: {profile_id}")
    schema = bundle["feature_schema"]
    results = []
    for source in rows:
        row = copy.deepcopy(dict(source))
        features = np.asarray(_encode_row(row, schema, profile_by_id), dtype="float64")
        predictions = {}
        uncertainty = {}
        for name, head in bundle["heads"].items():
            coefficients = np.asarray(head["coefficients"], dtype="float64")
            value = float(coefficients[0] + features @ coefficients[1:])
            predictions[name] = min(1.0, max(0.0, value)) if name == "feasibility" else value
            uncertainty[name] = head["residual_p90"]
        results.append({**row, "predictions": predictions, "uncertainty_p90": uncertainty})
    return results


def update_model_bundle(
    bundle: Mapping[str, Any],
    new_rows: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    combined = [*copy.deepcopy(bundle["training_rows"]), *[copy.deepcopy(dict(row)) for row in new_rows]]
    return fit_model_bundle(combined, capability_profiles, ridge=float(bundle["ridge"]))


def backend_blind_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Project a fitted bundle onto non-capability features for the blind ablation."""

    result = copy.deepcopy(dict(bundle))
    order = result["feature_schema"]["feature_order"]
    keep_indices = [
        index
        for index, name in enumerate(order)
        if not name.startswith("cap:")
        and not name.startswith("cap_observed:")
        and not name.startswith("interaction:q_int8*cap:")
        and not name.startswith("interaction:q_int8*cap_observed:")
    ]
    result["feature_schema"]["feature_order"] = [order[index] for index in keep_indices]
    result["feature_schema"]["capability_feature_names"] = []
    schema_for_hash = {k: v for k, v in result["feature_schema"].items() if k != "feature_schema_sha256"}
    result["feature_schema"]["feature_schema_sha256"] = _sha(schema_for_hash)
    for head in result["heads"].values():
        coefficients = head["coefficients"]
        head["coefficients"] = [coefficients[0], *[coefficients[index + 1] for index in keep_indices]]
    result["ablation"] = "backend_blind_projection"
    return result


def save_model_bundle(bundle: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected cost-model bundle schema")
    return bundle


__all__ = [
    "backend_blind_bundle",
    "build_feature_schema",
    "fit_model_bundle",
    "load_model_bundle",
    "predict_rows",
    "save_model_bundle",
    "update_model_bundle",
]
