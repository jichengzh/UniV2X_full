"""Evidence verification and comparable Pareto/HV utilities for Stage6."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


HV_MARGIN = 0.05
BACKEND_CONTEXT = {
    "trt": ("S5-PYR-TRT", "trt_engine", "h800-trt-probe-conditioned-v3"),
    "tvm": ("S5-PYR-TVM", "tvm_auto", "h800-tvm-probe-conditioned-v3"),
}
EXPECTED_MODEL = "pyramid"


def payload_sha256(payload: Any) -> str:
    """Return the canonical SHA used by Stage5/Stage6 JSON contracts."""
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _verify_file_pair(row: Mapping[str, Any], path_key: str, sha_key: str) -> str:
    path = Path(str(row.get(path_key) or ""))
    expected = str(row.get(sha_key) or "")
    if not path.is_file() or len(expected) != 64:
        raise ValueError(f"missing bound evidence: {path_key}")
    if _file_sha256(path) != expected:
        raise ValueError(f"bound evidence SHA mismatch: {path_key}")
    return str(path)


def normalize_actual_feedback_row(
    row: Mapping[str, Any], *, expected_backend: str
) -> dict[str, Any]:
    """Verify one actual-feedback row and return a paper-evidence point."""
    if expected_backend not in BACKEND_CONTEXT:
        raise ValueError(f"unsupported backend: {expected_backend}")
    task_id, dispatch_key, profile_id = BACKEND_CONTEXT[expected_backend]
    expected_context = {
        "task_id": task_id,
        "model": EXPECTED_MODEL,
        "hardware_id": "h800",
        "dispatch_key": dispatch_key,
        "capability_profile_id": profile_id,
    }
    for field, expected in expected_context.items():
        if row.get(field) != expected:
            raise ValueError(f"actual feedback context mismatch: {field}")

    stored_row_sha = str(row.get("actual_feedback_row_sha256") or "")
    unsigned = {key: value for key, value in row.items() if key != "actual_feedback_row_sha256"}
    if stored_row_sha != payload_sha256(unsigned):
        raise ValueError("actual feedback row SHA mismatch")

    width = row.get("width")
    q_mode = row.get("q_mode")
    genome = row.get("genome")
    if (
        not isinstance(width, Sequence)
        or isinstance(width, (str, bytes))
        or len(width) != 3
        or q_mode not in {"fp16", "int8"}
        or list(genome or []) != [*width, q_mode]
    ):
        raise ValueError("actual feedback genome identity mismatch")
    if row.get("terminal_status") != "measured_success_gold":
        raise ValueError("actual feedback row is not a successful measurement")

    evidence_files = {
        "performance": _verify_file_pair(
            row, "performance_result_json", "performance_result_sha256"
        ),
        "ap": _verify_file_pair(row, "ap_report_path", "ap_report_sha256"),
        "source": _verify_file_pair(
            row,
            "materialized_source_evidence_path",
            "materialized_source_evidence_sha256",
        ),
    }
    ap_payload = json.loads(Path(evidence_files["ap"]).read_text(encoding="utf-8"))
    if int(ap_payload.get("processed_samples") or 0) != 1789:
        raise ValueError("full AP evidence does not contain 1789 processed samples")
    if int(ap_payload.get("failed_samples") or 0) != 0:
        raise ValueError("full AP evidence contains failed samples")
    if int(ap_payload.get("fallback_samples") or 0) != 0:
        raise ValueError("full AP evidence contains fallback samples")

    graph_features = row.get("graph_features")
    if not isinstance(graph_features, Mapping):
        raise ValueError("actual graph features are missing")
    if graph_features.get("graph_feature_provenance") != "materialized_onnx_extracted_v1":
        raise ValueError("actual graph feature provenance mismatch")
    if payload_sha256(graph_features) != row.get("materialized_graph_features_sha256"):
        raise ValueError("actual graph feature SHA mismatch")
    candidate_features = row.get("candidate_graph_features")
    if isinstance(candidate_features, Mapping) and payload_sha256(candidate_features) != row.get(
        "candidate_graph_features_sha256"
    ):
        raise ValueError("candidate graph feature SHA mismatch")

    return {
        "manifest_job_id": str(row["manifest_job_id"]),
        "config": [int(value) for value in width] + [str(q_mode)],
        "AP70": _finite(row.get("ap70"), label="AP70"),
        "latency_ms": _finite(row.get("latency_ms"), label="latency_ms"),
        "energy_j": _finite(row.get("energy_j"), label="energy_j"),
        "terminal_status": "measured_success_gold",
        "independent_validation_passed": False,
        "evidence_sha_verified": True,
        "actual_graph_features_verified": True,
        "actual_feedback_row_sha256": stored_row_sha,
        "evidence_files": evidence_files,
    }


def normalize_closure_point(
    point: Mapping[str, Any], *, expected_backend: str
) -> dict[str, Any]:
    """Verify a Stage5 closure frontier point for Stage6 joint reuse."""
    if expected_backend not in BACKEND_CONTEXT:
        raise ValueError(f"unsupported backend: {expected_backend}")
    _, _, profile_id = BACKEND_CONTEXT[expected_backend]
    configuration_id = str(point.get("manifest_job_id") or "")
    if not configuration_id.endswith(f"profile={profile_id}"):
        raise ValueError("closure point capability profile mismatch")
    genome = point.get("genome")
    if (
        not isinstance(genome, Sequence)
        or isinstance(genome, (str, bytes))
        or len(genome) != 4
        or genome[3] not in {"fp16", "int8"}
        or point.get("q_mode") != genome[3]
    ):
        raise ValueError("closure point genome identity mismatch")
    if point.get("terminal_status") != "measured_success_gold":
        raise ValueError("closure point is not a successful measurement")
    objectives = point.get("objectives")
    if not isinstance(objectives, Mapping):
        raise ValueError("closure point objectives are missing")
    evidence_files = {
        "performance": _verify_file_pair(
            point, "performance_result_json", "performance_result_sha256"
        ),
        "ap": _verify_file_pair(point, "ap_report_path", "ap_report_sha256"),
    }
    ap_payload = json.loads(Path(evidence_files["ap"]).read_text(encoding="utf-8"))
    if int(ap_payload.get("processed_samples") or 0) != 1789:
        raise ValueError("closure full AP evidence is not a 1789-sample report")
    return {
        "manifest_job_id": configuration_id,
        "config": [int(value) for value in genome[:3]] + [str(genome[3])],
        "AP70": _finite(objectives.get("ap70"), label="AP70"),
        "latency_ms": _finite(objectives.get("latency_ms"), label="latency_ms"),
        "energy_j": _finite(objectives.get("energy_j"), label="energy_j"),
        "terminal_status": "measured_success_gold",
        "independent_validation_passed": False,
        "evidence_sha_verified": True,
        "actual_graph_features_verified": None,
        "evidence_files": evidence_files,
    }


def normalize_gold_row(
    row: Mapping[str, Any], *, expected_backend: str
) -> dict[str, Any]:
    """Verify an initial Gold176 row admitted to a Stage6 observed set."""
    if expected_backend not in BACKEND_CONTEXT:
        raise ValueError(f"unsupported backend: {expected_backend}")
    _, dispatch_key, profile_id = BACKEND_CONTEXT[expected_backend]
    if (
        row.get("model") != EXPECTED_MODEL
        or row.get("dispatch_key") != dispatch_key
        or row.get("capability_profile_id") != profile_id
    ):
        raise ValueError("Gold176 row context mismatch")
    width = row.get("width")
    q_mode = row.get("q_mode")
    if (
        not isinstance(width, Sequence)
        or isinstance(width, (str, bytes))
        or len(width) != 3
        or q_mode not in {"fp16", "int8"}
    ):
        raise ValueError("Gold176 row genome identity mismatch")
    if row.get("terminal_status") != "measured_success_gold":
        raise ValueError("Gold176 row is not a successful measurement")
    evidence_files = {
        "performance": _verify_file_pair(
            row, "performance_result_json", "performance_result_sha256"
        ),
        "ap": _verify_file_pair(row, "ap_report_path", "ap_report_sha256"),
    }
    ap_payload = json.loads(Path(evidence_files["ap"]).read_text(encoding="utf-8"))
    if int(ap_payload.get("processed_samples") or 0) != 1789:
        raise ValueError("Gold176 full AP evidence is not a 1789-sample report")
    return {
        "manifest_job_id": str(row.get("manifest_job_id") or ""),
        "config": [int(value) for value in width] + [str(q_mode)],
        "AP70": _finite(row.get("ap70"), label="AP70"),
        "latency_ms": _finite(row.get("latency_ms"), label="latency_ms"),
        "energy_j": _finite(row.get("energy_j"), label="energy_j"),
        "terminal_status": "measured_success_gold",
        "independent_validation_passed": False,
        "evidence_sha_verified": True,
        "actual_graph_features_verified": None,
        "evidence_files": evidence_files,
        "evidence_origin": "initial_coldstart",
    }


def _verify_independent_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    consistency = configuration.get("consistency")
    if not isinstance(consistency, Mapping) or consistency.get("passed") is not True:
        raise ValueError("independent validation consistency did not pass")
    rerun = consistency.get("rerun")
    if not isinstance(rerun, Mapping):
        raise ValueError("independent validation rerun metrics are missing")
    repeats = configuration.get("performance_repeats")
    if not isinstance(repeats, Sequence) or isinstance(repeats, (str, bytes)) or len(repeats) != 3:
        raise ValueError("independent validation requires exactly three repeats")
    evidence_files = []
    for repeat in repeats:
        if not isinstance(repeat, Mapping):
            raise ValueError("independent repeat record is invalid")
        evidence_files.append(
            _verify_file_pair(
                repeat, "performance_result_json", "performance_result_sha256"
            )
        )
    evidence_files.extend(
        [
            _verify_file_pair(configuration, "ap_report_path", "ap_report_sha256"),
            _verify_file_pair(configuration, "evidence_path", "evidence_sha256"),
        ]
    )
    return {
        "latency_ms": _finite(rerun.get("latency_median_ms"), label="latency_median_ms"),
        "energy_j": _finite(rerun.get("energy_median_j"), label="energy_median_j"),
        "AP70": _finite(rerun.get("ap70"), label="independent AP70"),
        "evidence_files": evidence_files,
    }


def build_independent_validation_index(
    audits: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build a SHA-verified configuration index from independent audits."""
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for audit in audits:
        schema = audit.get("schema_version")
        if schema not in {
            "stage5_independent_validation_audit_v1",
            "stage6_independent_validation_audit_v1",
        }:
            raise ValueError("unexpected independent validation audit schema")
        tasks = audit.get("tasks")
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            raise ValueError("independent validation tasks are missing")
        for task in tasks:
            if not isinstance(task, Mapping) or task.get("passed") is not True:
                continue
            task_id = str(task.get("task_id") or "")
            for configuration in task.get("configurations") or []:
                if not isinstance(configuration, Mapping):
                    raise ValueError("independent configuration record is invalid")
                configuration_id = str(configuration.get("configuration_id") or "")
                if not configuration_id:
                    raise ValueError("independent configuration ID is missing")
                arm_id = str(
                    configuration.get("arm_id")
                    or task.get("arm_id")
                    or ("joint_shcosearch" if schema == "stage5_independent_validation_audit_v1" else "")
                )
                if not arm_id:
                    raise ValueError("Stage6 independent validation arm ID is missing")
                verified = {
                    **_verify_independent_configuration(configuration),
                    "task_id": task_id,
                    "arm_id": arm_id,
                }
                key = (arm_id, configuration_id)
                if key in index and index[key] != verified:
                    raise ValueError(
                        f"conflicting independent validation evidence: {arm_id}:{configuration_id}"
                    )
                index[key] = verified
    return index


def apply_independent_validation(
    points: Sequence[Mapping[str, Any]],
    validation_index: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    arm_id: str,
) -> list[dict[str, Any]]:
    """Return points whose displayed metrics use independent rerun evidence."""
    result = []
    for source in points:
        point = copy.deepcopy(dict(source))
        configuration_id = str(point.get("manifest_job_id") or "")
        validation = validation_index.get((arm_id, configuration_id))
        if validation is None:
            point["independent_validation_passed"] = False
            result.append(point)
            continue
        point.update(
            {
                "search_latency_ms": point.get("latency_ms"),
                "search_energy_j": point.get("energy_j"),
                "search_AP70": point.get("AP70"),
                "latency_ms": validation["latency_ms"],
                "energy_j": validation["energy_j"],
                "AP70": validation["AP70"],
                "independent_validation_passed": True,
                "independent_validation_evidence": list(
                    validation.get("evidence_files") or []
                ),
            }
        )
        result.append(point)
    return result


def _objective(point: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        _finite(point.get("latency_ms"), label="latency_ms"),
        _finite(point.get("energy_j"), label="energy_j"),
        -_finite(point.get("AP70"), label="AP70"),
    )


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _frontier(points: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    objectives = [_objective(point) for point in points]
    return [
        point
        for index, point in enumerate(points)
        if not any(
            other_index != index and _dominates(other, objectives[index])
            for other_index, other in enumerate(objectives)
        )
    ]


def _hypervolume_2d(
    points: Sequence[tuple[float, float]], reference: tuple[float, float]
) -> float:
    eligible = sorted(
        {
            point
            for point in points
            if point[0] < reference[0] and point[1] < reference[1]
        }
    )
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {reference[0]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        active_y = [y for x, y in eligible if x <= left]
        area += (right - left) * max(0.0, reference[1] - min(active_y))
    return area


def _hypervolume_3d(
    points: Sequence[tuple[float, float, float]],
    reference: tuple[float, float, float],
) -> float:
    eligible = [
        point
        for point in set(points)
        if all(value < reference[index] for index, value in enumerate(point))
    ]
    eligible = [
        point
        for index, point in enumerate(eligible)
        if not any(
            other_index != index and _dominates(other, point)
            for other_index, other in enumerate(eligible)
        )
    ]
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {reference[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = [(y, z) for x, y, z in eligible if x <= left]
        volume += (right - left) * _hypervolume_2d(
            active, (reference[1], reference[2])
        )
    return float(volume)


def attach_common_hypervolume(
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach HV using one normalization domain shared by all backend arms."""
    verified_by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for arm_id, summary in arms.items():
        verified_by_arm[arm_id] = [
            point
            for point in summary.get("points") or []
            if point.get("terminal_status") == "measured_success_gold"
            and point.get("evidence_sha_verified") is True
        ]
    all_points = [point for points in verified_by_arm.values() for point in points]
    if not all_points:
        raise ValueError("no SHA-verified measured points for common HV")
    vectors = [_objective(point) for point in all_points]
    lows = tuple(min(vector[axis] for vector in vectors) for axis in range(3))
    highs = tuple(max(vector[axis] for vector in vectors) for axis in range(3))
    spans = tuple(
        high - low if high > low else max(abs(low), 1.0)
        for low, high in zip(lows, highs)
    )
    reference = tuple(
        (high - low) / span + HV_MARGIN
        for low, high, span in zip(lows, highs, spans)
    )

    result_arms = copy.deepcopy(dict(arms))
    for arm_id, points in verified_by_arm.items():
        frontier = _frontier(points)
        normalized = [
            tuple(
                (value - lows[axis]) / spans[axis]
                for axis, value in enumerate(_objective(point))
            )
            for point in frontier
        ]
        result_arms[arm_id]["HV"] = _hypervolume_3d(normalized, reference)
        result_arms[arm_id]["pareto_count"] = len(frontier)
        result_arms[arm_id]["pareto_ids"] = [
            str(point.get("manifest_job_id") or "") for point in frontier
        ]
    return {
        "arms": result_arms,
        "normalization": {
            "objective_order": ["latency_ms", "energy_j", "negative_ap70"],
            "minimum": list(lows),
            "span": list(spans),
            "reference": list(reference),
            "derived_from_point_count": len(all_points),
        },
    }
