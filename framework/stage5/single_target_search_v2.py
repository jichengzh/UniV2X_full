"""Single-target, single-profile search contracts for Stage5 v2."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from framework.stage2.canonical_search_v3 import validate_capability_profile
from framework.stage5.genome_contract_v1 import (
    MODEL_WIDTH_SCHEMAS,
    validate_structure_identity,
    width_schema_for_model,
)
from framework.stage4.cost_model_selection_v1 import encode_rows
from framework.stage5.production_search_v1 import (
    ProductionBundle,
    _extra_trees,
    _graph_feature_names,
    _lgbm_huber,
    _predict_model,
    _quantile,
    _stable_calibration_groups,
)


SCHEMA_VERSION = "stage5_single_target_search_v2"
ALLOWED_MODELS = set(MODEL_WIDTH_SCHEMAS)
ALLOWED_Q_MODES = {"fp16", "int8"}
SUCCESS_STATUS = "measured_success_gold"
TRUE_FAILURE_STATUSES = {"feasibility_failure", "numerical_feasibility_failure"}
PUBLIC_FAILURE_STATUSES = {"public_runner_failure", "shared_source_failure"}
FROZEN_GOLD176_ROWS_SHA256 = "9880d625e1ac2c5e336a5de3bc1d861072d58e05d4b1bea6c79ef1cd0e93ca19"
FROZEN_GOLD176_GRAPH_FEATURES_SHA256 = "c5f03e19daba4779cb187f036d7c4bf612d3479a00453149f4eaf213412536cd"


def verify_frozen_coldstart_artifacts(
    rows_path: Path,
    graph_features_path: Path,
    *,
    expected_rows_sha256: str = FROZEN_GOLD176_ROWS_SHA256,
    expected_graph_features_sha256: str = FROZEN_GOLD176_GRAPH_FEATURES_SHA256,
) -> dict[str, str]:
    """Pin Stage5 D0 to the reviewed standalone Gold176 artifacts."""
    rows_sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    graph_sha = hashlib.sha256(graph_features_path.read_bytes()).hexdigest()
    if rows_sha != expected_rows_sha256:
        raise ValueError(f"frozen Gold176 rows SHA mismatch: {rows_sha}")
    if graph_sha != expected_graph_features_sha256:
        raise ValueError(f"frozen Gold176 graph features SHA mismatch: {graph_sha}")
    return {"rows_sha256": rows_sha, "graph_features_sha256": graph_sha}


def freeze_initial_coldstart(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the immutable Stage5 D0 view and reject non-Gold evidence."""
    if len(rows) != 176:
        raise ValueError("Stage5 initial training view must contain exactly 176 Gold rows")
    frozen = [copy.deepcopy(dict(row)) for row in rows]
    invalid_sources = sorted(
        {
            str(row.get("training_source"))
            for row in frozen
            if row.get("training_source") not in {None, "initial_coldstart"}
        }
    )
    if invalid_sources:
        raise ValueError(
            "Stage5 initial training rows must be initial_coldstart only: "
            + ",".join(invalid_sources)
        )
    identities = [str(row.get("manifest_job_id") or row.get("row_id") or "") for row in frozen]
    if any(not identity for identity in identities) or len(identities) != len(set(identities)):
        raise ValueError("Gold176 initial training identities must be non-empty and unique")
    return [{**row, "training_source": "initial_coldstart"} for row in frozen]


@dataclass(frozen=True)
class SearchTask:
    task_id: str
    target_model: str
    hardware_id: str
    capability_profile: Mapping[str, Any]
    sample_budget: int = 16
    batch_size: int = 4
    round_count: int = 4


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def validate_search_task(task: SearchTask) -> dict[str, Any]:
    if not task.task_id:
        raise ValueError("task_id must not be empty")
    if task.target_model not in ALLOWED_MODELS:
        raise ValueError(f"unsupported target model: {task.target_model}")
    if not task.hardware_id:
        raise ValueError("hardware_id must not be empty")
    profile = validate_capability_profile(task.capability_profile)
    profile_hardware = str(profile.get("hardware_target") or "").lower()
    if profile_hardware and task.hardware_id.lower() != profile_hardware:
        raise ValueError("task hardware/profile mismatch")
    if (task.sample_budget, task.batch_size, task.round_count) != (16, 4, 4):
        raise ValueError("Stage5 main-search budget is frozen at B=4, T=16, four rounds")
    payload = {
        "schema_version": "stage5_search_task_contract_v2",
        "task_id": task.task_id,
        "target_model": task.target_model,
        "hardware_id": task.hardware_id,
        "capability_profile_id": profile["capability_profile_id"],
        "capability_digest": profile["capability_digest"],
        "dispatch_key": profile["dispatch_key"],
        "genome_schema": [*width_schema_for_model(task.target_model), "q_mode"],
        "sample_budget": task.sample_budget,
        "batch_size": task.batch_size,
        "round_count": task.round_count,
        "main_search_early_stopping": False,
    }
    return {**payload, "task_sha256": _sha(payload)}


def validate_task_feedback_history(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: SearchTask,
    completed_rounds: int,
) -> dict[str, Any]:
    """Reject stale, cross-task, or non-online rows before an online refit."""
    contract = validate_search_task(task)
    expected_count = completed_rounds * task.batch_size
    if completed_rounds not in {1, 2, 3} or len(rows) != expected_count:
        raise ValueError("feedback history must contain four rows per completed round")
    identities = [_row_id(row) for row in rows]
    if any(not identity for identity in identities) or len(set(identities)) != len(rows):
        raise ValueError("feedback history identities must be non-empty and unique")
    expected = {
        "task_id": task.task_id,
        "model": task.target_model,
        "hardware_id": task.hardware_id,
        "capability_profile_id": contract["capability_profile_id"],
        "dispatch_key": contract["dispatch_key"],
        "task_sha256": contract["task_sha256"],
        "training_source": "online_feedback",
    }
    for row in rows:
        for field, value in expected.items():
            if str(row.get(field)) != str(value):
                raise ValueError(f"feedback history {field} drift")
        if row.get("terminal_status") not in {SUCCESS_STATUS, *TRUE_FAILURE_STATUSES}:
            raise ValueError("feedback history contains a non-terminal or public failure")
    return {
        "task_id": task.task_id,
        "completed_rounds": completed_rounds,
        "feedback_rows": len(rows),
        "task_sha256": contract["task_sha256"],
    }


def build_task_candidate_manifest(
    source_registry: Mapping[str, Any],
    *,
    task: SearchTask,
    measured_row_ids: set[str],
    frozen_holdout_group_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Expand source widths into independent q genomes for one fixed task."""
    if source_registry.get("schema_version") != "stage5_candidate_source_registry_v1":
        raise ValueError("unexpected candidate source registry schema")
    contract = validate_search_task(task)
    profile = validate_capability_profile(task.capability_profile)
    groups = source_registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("candidate source registry groups must be a list")
    holdout_ids = frozen_holdout_group_ids or set()
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    observed: set[str] = set()
    for source in groups:
        group = copy.deepcopy(dict(source))
        group_id = str(group.get("group_id") or "")
        if not group_id or group_id in observed:
            raise ValueError("empty or duplicate source group_id")
        observed.add(group_id)
        if str(group.get("model")) != task.target_model:
            continue
        identity = validate_structure_identity(group)
        width = list(identity.width)
        if identity.model != task.target_model:
            raise ValueError(f"source identity mismatch: {group_id}")
        if group_id in holdout_ids:
            excluded.append({"group_id": group_id, "reason": "frozen_independent_holdout"})
            continue
        if group.get("source_status") not in {"ready", "materializable"}:
            excluded.append({"group_id": group_id, "reason": "source_not_ready"})
            continue
        if not isinstance(group.get("graph_features"), Mapping):
            raise ValueError(f"candidate graph features missing: {group_id}")
        if len(str(group.get("source_evidence_sha256") or "")) != 64:
            raise ValueError(f"candidate source evidence SHA missing: {group_id}")
        for q_mode in ("fp16", "int8"):
            profile_id = str(profile["capability_profile_id"])
            row_id = f"{group_id}|q={q_mode}|profile={profile_id}"
            if row_id in measured_row_ids:
                excluded.append({"row_id": row_id, "reason": "already_measured"})
                continue
            rows.append(
                {
                    "schema_version": "stage5_candidate_row_v2",
                    "task_id": task.task_id,
                    "task_sha256": contract["task_sha256"],
                    "row_id": row_id,
                    "manifest_job_id": row_id,
                    "group_id": group_id,
                    "model": task.target_model,
                    "width": width,
                    "width_schema": list(identity.width_schema),
                    "structure_widths": dict(identity.structure_widths),
                    "genome": [*width, q_mode],
                    "strategy_id": f"q={q_mode}",
                    "q_mode": q_mode,
                    "hardware_id": task.hardware_id,
                    "capability_profile_id": profile_id,
                    "capability_digest": profile["capability_digest"],
                    "dispatch_key": profile["dispatch_key"],
                    "source_status": group["source_status"],
                    "materialization_kind": group.get("materialization_kind"),
                    "source_evidence_kind": group.get("source_evidence_kind"),
                    "source_contract": copy.deepcopy(group["source_contract"]),
                    "source_evidence_sha256": group["source_evidence_sha256"],
                    "graph_features": copy.deepcopy(group["graph_features"]),
                }
            )
    rows.sort(key=_row_id)
    return {
        "schema_version": "stage5_task_candidate_manifest_v2",
        "task_id": task.task_id,
        "task_sha256": contract["task_sha256"],
        "target_model": task.target_model,
        "capability_profile_id": profile["capability_profile_id"],
        "eligible_row_count": len(rows),
        "excluded": excluded,
        "rows": rows,
    }


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _frontier_row_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    vectors = [
        (
            float(row["predictions"]["latency_ms"]),
            float(row["predictions"]["energy_j"]),
            -float(row["predictions"]["ap70"]),
        )
        for row in rows
    ]
    return {
        _row_id(row)
        for index, row in enumerate(rows)
        if not any(
            other != index and _dominates(vectors[other], vectors[index])
            for other in range(len(rows))
        )
    }


def _feature_vector(row: Mapping[str, Any]) -> np.ndarray:
    graph = row.get("graph_features") or {}
    forbidden = {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}
    leaked = forbidden & set(graph)
    if leaked:
        raise ValueError(f"nested graph label leakage: {sorted(leaked)}")
    numeric = [
        float(value)
        for key, value in sorted(graph.items())
        if key not in {"group_id", "model", "width", "input_dims"} and _finite(value)
    ]
    return np.asarray(
        [*[float(value) for value in row["width"]], float(row["q_mode"] == "int8"), *numeric],
        dtype=float,
    )


def select_task_batch(
    predicted_rows: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    measured_graph_features: Sequence[Mapping[str, Any]],
    *,
    task: SearchTask,
) -> dict[str, Any]:
    """Select independent genomes; q_mode is a feature, never a quota or pair rule."""
    del measured_graph_features  # Candidate rows already carry the canonical graph features.
    contract = validate_search_task(task)
    candidates = [copy.deepcopy(dict(row)) for row in predicted_rows]
    if len(candidates) < task.batch_size:
        raise ValueError("not enough eligible single-genome candidates")
    forbidden = {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}
    seen: set[str] = set()
    for row in candidates:
        row_id = _row_id(row)
        if not row_id or row_id in seen:
            raise ValueError("empty or duplicate candidate row identity")
        seen.add(row_id)
        if (
            row.get("task_id") != task.task_id
            or row.get("model") != task.target_model
            or row.get("capability_profile_id") != contract["capability_profile_id"]
        ):
            raise ValueError("candidate drift from fixed search task")
        if forbidden & set(row):
            raise ValueError("candidate labels visible before measurement")
        if str(row.get("q_mode")) not in ALLOWED_Q_MODES:
            raise ValueError("unsupported q_mode")
    all_rows = [*candidates, *[dict(row) for row in measured_rows if row.get("model") == task.target_model]]
    vectors = [_feature_vector(row) for row in all_rows]
    width = max(vector.size for vector in vectors)
    padded = np.vstack([np.pad(vector, (0, width - vector.size)) for vector in vectors])
    low = np.min(padded, axis=0)
    span = np.maximum(np.ptp(padded, axis=0), 1e-12)
    normalized = (padded - low) / span
    candidate_vectors = normalized[: len(candidates)]
    measured_vectors = normalized[len(candidates) :]
    frontier = _frontier_row_ids(candidates)
    diagnostics: dict[str, dict[str, float | int]] = {}
    for index, row in enumerate(candidates):
        intervals = row["prediction_intervals"]
        uncertainty = sum(
            max(0.0, float(intervals[target]["upper"]) - float(intervals[target]["lower"]))
            / max(abs(float(row["predictions"][target])), 1e-9)
            for target in ("latency_ms", "energy_j", "ap70")
        )
        diversity = min(
            (float(np.linalg.norm(candidate_vectors[index] - ref)) for ref in measured_vectors),
            default=float(np.linalg.norm(candidate_vectors[index] - 0.5)),
        )
        diagnostics[_row_id(row)] = {
            "predicted_frontier": int(_row_id(row) in frontier),
            "feature_diversity": diversity,
            "uncertainty": uncertainty,
        }
    fallback_ranked = sorted(
        candidates,
        key=lambda row: (
            -int(diagnostics[_row_id(row)]["predicted_frontier"]),
            -float(diagnostics[_row_id(row)]["feature_diversity"]),
            -float(diagnostics[_row_id(row)]["uncertainty"]),
            _row_id(row),
        ),
    )
    prediction_matrix = np.asarray(
        [
            (
                float(row["predictions"]["latency_ms"]),
                float(row["predictions"]["energy_j"]),
                -float(row["predictions"]["ap70"]),
            )
            for row in candidates
        ],
        dtype=float,
    )
    objective_low = np.min(prediction_matrix, axis=0)
    objective_span = np.maximum(np.ptp(prediction_matrix, axis=0), 1e-12)
    objective_normalized = (prediction_matrix - objective_low) / objective_span
    selected_indices: list[int] = []
    # Latency and AP anchor exploitation. Energy is strongly correlated with
    # latency in the measured corpus, so two remaining slots are reserved for
    # shape/q maximin exploration instead of spending 3/4 of every batch on
    # potentially inaccurate objective predictions.
    for objective_index in (0, 2):
        for candidate_index in np.argsort(
            objective_normalized[:, objective_index], kind="stable"
        ):
            index = int(candidate_index)
            if index not in selected_indices:
                selected_indices.append(index)
                break
    while len(selected_indices) < task.batch_size:
        remaining = [index for index in range(len(candidates)) if index not in selected_indices]
        chosen = max(
            remaining,
            key=lambda index: (
                min(
                    float(
                        np.linalg.norm(
                            candidate_vectors[index] - candidate_vectors[selected]
                        )
                    )
                    for selected in selected_indices
                ),
                min(
                    float(
                        np.linalg.norm(
                            objective_normalized[index] - objective_normalized[selected]
                        )
                    )
                    for selected in selected_indices
                ),
                float(diagnostics[_row_id(candidates[index])]["feature_diversity"]),
                float(diagnostics[_row_id(candidates[index])]["uncertainty"]),
                _row_id(candidates[index]),
            ),
        )
        selected_indices.append(chosen)
    selected_id_set = {_row_id(candidates[index]) for index in selected_indices}
    selected = [row for row in fallback_ranked if _row_id(row) in selected_id_set]
    return {
        "schema_version": "stage5_single_genome_acquisition_v2",
        "policy": "predicted_frontier_diversity",
        "task_id": task.task_id,
        "task_sha256": contract["task_sha256"],
        "candidate_labels_visible_before_measurement": False,
        "selected_row_count": len(selected),
        "selected_row_ids": [_row_id(row) for row in selected],
        "selected_rows": selected,
        "diagnostics": diagnostics,
    }


def fit_online_bundle(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> ProductionBundle:
    """Fit the frozen Stage4 heads after independent single-genome feedback."""
    source_rows = [
        dict(row)
        for row in rows
        if str(row.get("terminal_status")) == SUCCESS_STATUS
        and all(_finite(row.get(target)) for target in ("latency_ms", "energy_j", "ap70"))
    ]
    if len(source_rows) < 20:
        raise ValueError("at least 20 finite rows are required for online fitting")
    encoded = encode_rows(source_rows, graph_features, capability_profiles)
    anchors = {
        model: float(
            np.median(
                [float(row["ap70"]) for row in source_rows if str(row["model"]) == model]
            )
        )
        for model in sorted({str(row["model"]) for row in source_rows})
    }
    value_heads: dict[str, Any] = {}
    for target in ("latency_ms", "energy_j"):
        model = _extra_trees(seed)
        model.fit(encoded.matrix, np.log1p([float(row[target]) for row in source_rows]))
        value_heads[target] = model
    ap_model = _lgbm_huber(seed)
    ap_model.fit(
        encoded.matrix,
        np.asarray(
            [float(row["ap70"]) - anchors[str(row["model"])] for row in source_rows],
            dtype=float,
        ),
    )
    value_heads["ap70"] = ap_model

    calibration_groups = _stable_calibration_groups(source_rows, seed)
    fit_indices = [
        index
        for index, row in enumerate(source_rows)
        if str(row["group_id"]) not in calibration_groups
    ]
    calibration_indices = [
        index
        for index, row in enumerate(source_rows)
        if str(row["group_id"]) in calibration_groups
    ]
    interval_heads: dict[str, tuple[Any, Any, Any]] = {}
    corrections: dict[str, dict[str, float]] = {}
    for target_index, target in enumerate(("latency_ms", "energy_j", "ap70")):
        models = tuple(
            _quantile(alpha, seed + target_index) for alpha in (0.05, 0.50, 0.95)
        )
        y_fit = np.asarray(
            [float(source_rows[index][target]) for index in fit_indices], dtype=float
        )
        for model in models:
            model.fit(encoded.matrix[fit_indices], y_fit)
        interval_heads[target] = models
        predicted = np.sort(
            np.vstack(
                [
                    _predict_model(model, encoded.matrix[calibration_indices])
                    for model in models
                ]
            ),
            axis=0,
        )
        scores: dict[str, list[float]] = {}
        for local_index, row_index in enumerate(calibration_indices):
            row = source_rows[row_index]
            model_name = str(row["model"])
            truth = float(row[target])
            score = max(
                float(predicted[0, local_index]) - truth,
                truth - float(predicted[2, local_index]),
                0.0,
            )
            scores.setdefault(model_name, []).append(score)
        corrections[target] = {
            model_name: float(max(values)) for model_name, values in scores.items()
        }
    manifest = {
        "schema_version": "stage5_online_model_bundle_manifest_v2",
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "acquisition_policy": "predicted_frontier_diversity",
        "seed": seed,
        "feature_names": list(encoded.feature_names),
        "input_row_count": len(rows),
        "value_training_row_count": len(source_rows),
        "calibration_groups": sorted(calibration_groups),
    }
    manifest["bundle_config_sha256"] = _sha(manifest)
    return ProductionBundle(
        manifest=manifest,
        feature_names=encoded.feature_names,
        graph_feature_names=_graph_feature_names(encoded.feature_names),
        value_heads=value_heads,
        interval_heads=interval_heads,
        conformal_corrections=corrections,
        model_anchors=anchors,
    )


def build_measurement_request(
    *, task: SearchTask, selected_rows: Sequence[Mapping[str, Any]], round_index: int
) -> dict[str, Any]:
    contract = validate_search_task(task)
    if round_index < 0 or round_index >= task.round_count:
        raise ValueError("round_index outside frozen task budget")
    identity_fields = (
        "schema_version",
        "task_id",
        "task_sha256",
        "row_id",
        "manifest_job_id",
        "group_id",
        "model",
        "width",
        "width_schema",
        "structure_widths",
        "genome",
        "strategy_id",
        "q_mode",
        "hardware_id",
        "capability_profile_id",
        "capability_digest",
        "dispatch_key",
        "source_status",
        "materialization_kind",
        "source_evidence_kind",
        "source_contract",
        "source_evidence_sha256",
        "graph_features",
    )
    rows = [
        {key: copy.deepcopy(row[key]) for key in identity_fields if key in row}
        for row in selected_rows
    ]
    if len(rows) != task.batch_size or len({_row_id(row) for row in rows}) != len(rows):
        raise ValueError("measurement batch must contain exactly four unique genomes")
    if any(row.get("task_sha256") != contract["task_sha256"] for row in rows):
        raise ValueError("measurement row drift from task")
    row_sha = {_row_id(row): _sha(row) for row in rows}
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": task.task_id,
        "task_sha256": contract["task_sha256"],
        "round_index": round_index,
        "batch_size": task.batch_size,
        "sample_budget": task.sample_budget,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": row_sha,
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def finalize_atomic_batch(
    request: Mapping[str, Any], feedback_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if request.get("schema_version") != "stage5_measurement_request_v2":
        raise ValueError("unexpected measurement request schema")
    requested = {_row_id(row): dict(row) for row in request.get("rows") or []}
    feedback = {_row_id(row): dict(row) for row in feedback_rows}
    if set(feedback) != set(requested):
        raise ValueError("atomic batch feedback requires all requested rows")
    for row_id, row in requested.items():
        if _sha(row) != request["row_sha256"].get(row_id):
            raise ValueError(f"request row SHA drift: {row_id}")
    for row_id, row in feedback.items():
        if row.get("measurement_request_row_sha256") != request["row_sha256"][row_id]:
            raise ValueError(f"feedback identity SHA drift: {row_id}")
        for field, value in requested[row_id].items():
            if field == "graph_features":
                continue
            if row.get(field) != value:
                raise ValueError(f"feedback identity field drift: {row_id}:{field}")
    statuses = {str(row.get("terminal_status") or "") for row in feedback.values()}
    if statuses & PUBLIC_FAILURE_STATUSES:
        return {
            "schema_version": "stage5_atomic_batch_audit_v2",
            "feedback_released": False,
            "batch_quarantined": True,
            "budget_consumed": 0,
            "resume_row_ids": list(requested),
            "reason": "public runner/shared source failure",
        }
    allowed = {SUCCESS_STATUS, *TRUE_FAILURE_STATUSES}
    if not statuses or statuses - allowed:
        raise ValueError(f"atomic batch contains non-terminal status: {sorted(statuses - allowed)}")
    for row in feedback.values():
        if row["terminal_status"] == SUCCESS_STATUS and not all(
            _finite(row.get(metric))
            for metric in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
        ):
            raise ValueError(f"successful row lacks metrics: {_row_id(row)}")
        if row["terminal_status"] in TRUE_FAILURE_STATUSES and not row.get("failure_reason"):
            raise ValueError(f"true failure lacks reason: {_row_id(row)}")
    return {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": True,
        "batch_quarantined": False,
        "budget_consumed": len(requested),
        "successful_rows": sum(
            row["terminal_status"] == SUCCESS_STATUS for row in feedback.values()
        ),
        "feasibility_terminal_rows": sum(
            row["terminal_status"] in TRUE_FAILURE_STATUSES for row in feedback.values()
        ),
        "released_feedback_rows": list(feedback.values()),
        "resume_row_ids": [],
    }
