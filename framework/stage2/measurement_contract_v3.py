"""Paper-grade Stage2 measurement row and final-frontier contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


MEASUREMENT_ROW_SCHEMA_VERSION = "stage2_measurement_row_v3"
REGISTRY_SUMMARY_SCHEMA_VERSION = "stage2_measurement_registry_summary_v3"
ARTIFACT_REGISTRY_SCHEMA_VERSION = "stage2_artifact_registry_v3"
VALID_Q_MODES = {"fp32", "fp16", "int8", "mixed_int8"}
VALID_EVIDENCE_KINDS = {"measured", "historical", "estimated", "proxy", "diagnostic"}
VALID_METRIC_STATUSES = {"measured", "not_measured", "failed", "diagnostic"}
VALID_BUILD_STATUSES = {"success", "failed", "not_run"}
VALID_NUMERICAL_STATUSES = {"pass", "fail", "not_run"}
VALID_CALIBRATION_KINDS = {"none", "formal", "diagnostic", "historical"}
VALID_LOWERING_ORIGINS = {
    "source_ir_automatic",
    "compiler_engine_automatic",
    "explicit_te_constructed",
    "historical_unknown",
}
AUTOMATIC_REALIZATION_ORIGINS = {"source_ir_automatic", "compiler_engine_automatic"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(mapping: dict[str, Any], fields: tuple[str, ...], *, context: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")


def _positive_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return number


def _pipeline_identity(row: dict[str, Any]) -> dict[str, Any]:
    _require(
        row,
        (
            "schema_version",
            "model",
            "dataset",
            "checkpoint_id",
            "width",
            "q_mode",
            "mixed_policy_id",
            "capability_profile_id",
            "pipeline",
        ),
        context="measurement row",
    )
    return {
        "schema_version": row["schema_version"],
        "model": row["model"],
        "dataset": row["dataset"],
        "checkpoint_id": row["checkpoint_id"],
        "width": row["width"],
        "q_mode": row["q_mode"],
        "mixed_policy_id": row["mixed_policy_id"],
        "capability_profile_id": row["capability_profile_id"],
        "pipeline": row["pipeline"],
    }


def compute_pipeline_fingerprint(row: dict[str, Any]) -> str:
    payload = json.dumps(
        _pipeline_identity(row),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_pipeline(row: dict[str, Any]) -> None:
    pipeline = row["pipeline"]
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be a mapping")
    _require(
        pipeline,
        (
            "onnx_sha256",
            "calibration_sha256",
            "calibration_kind",
            "lowering_origin",
            "source_ir_sha256",
            "schedule_trace_sha256",
            "tuning_database_sha256",
            "compiler_fingerprint",
            "optimized_scope",
            "input_shape",
            "batch_size",
            "tuning_budget",
        ),
        context="pipeline",
    )
    for field in ("onnx_sha256", "compiler_fingerprint"):
        if SHA256_RE.fullmatch(str(pipeline[field] or "")) is None:
            raise ValueError(f"pipeline.{field} must be a SHA256 digest")
    lowering_origin = pipeline["lowering_origin"]
    if lowering_origin not in VALID_LOWERING_ORIGINS:
        raise ValueError("pipeline.lowering_origin is invalid")
    source_ir_sha = pipeline["source_ir_sha256"]
    if lowering_origin == "compiler_engine_automatic":
        if source_ir_sha is not None and SHA256_RE.fullmatch(str(source_ir_sha)) is None:
            raise ValueError("pipeline.source_ir_sha256 must be null or a SHA256 digest")
    elif SHA256_RE.fullmatch(str(source_ir_sha or "")) is None:
        raise ValueError("pipeline.source_ir_sha256 must be a SHA256 digest")
    for field in ("schedule_trace_sha256", "tuning_database_sha256"):
        value = pipeline[field]
        if lowering_origin == "source_ir_automatic":
            if SHA256_RE.fullmatch(str(value or "")) is None:
                raise ValueError(f"automatic lowering requires pipeline.{field}")
        elif value is not None and SHA256_RE.fullmatch(str(value)) is None:
            raise ValueError(f"pipeline.{field} must be null or a SHA256 digest")
    engine_digest_fields = (
        "engine_build_config_sha256",
        "engine_inspector_sha256",
        "compiled_engine_sha256",
    )
    for field in engine_digest_fields:
        value = pipeline.get(field)
        if lowering_origin == "compiler_engine_automatic":
            if SHA256_RE.fullmatch(str(value or "")) is None:
                raise ValueError(f"automatic engine realization requires pipeline.{field}")
        elif value is not None and SHA256_RE.fullmatch(str(value)) is None:
            raise ValueError(f"pipeline.{field} must be null or a SHA256 digest")
    calibration_sha = pipeline["calibration_sha256"]
    calibration_kind = pipeline["calibration_kind"]
    if calibration_kind not in VALID_CALIBRATION_KINDS:
        raise ValueError("pipeline.calibration_kind is invalid")
    if row["q_mode"] in {"int8", "mixed_int8"}:
        if SHA256_RE.fullmatch(str(calibration_sha or "")) is None:
            raise ValueError("INT8 rows require pipeline.calibration_sha256")
        if calibration_kind == "none":
            raise ValueError("INT8 rows require a non-none pipeline.calibration_kind")
    elif calibration_sha is not None and SHA256_RE.fullmatch(str(calibration_sha)) is None:
        raise ValueError("pipeline.calibration_sha256 must be null or a SHA256 digest")
    elif calibration_kind != "none":
        raise ValueError("FP32/FP16 rows require pipeline.calibration_kind=none")
    input_shape = pipeline["input_shape"]
    if not isinstance(input_shape, list) or not input_shape:
        raise ValueError("pipeline.input_shape must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in input_shape):
        raise ValueError("pipeline.input_shape values must be positive integers")
    batch_size = pipeline["batch_size"]
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("pipeline.batch_size must be a positive integer")
    if input_shape[0] != batch_size:
        raise ValueError("pipeline.batch_size must match input_shape[0]")
    if not isinstance(pipeline["tuning_budget"], dict) or not pipeline["tuning_budget"]:
        raise ValueError("pipeline.tuning_budget must be a non-empty mapping")


def _validate_metric_fingerprint(name: str, metric: dict[str, Any], expected: str) -> None:
    status = str(metric.get("status") or "")
    if status not in VALID_METRIC_STATUSES:
        raise ValueError(f"metrics.{name}.status is invalid: {status!r}")
    fingerprint = str(metric.get("pipeline_fingerprint") or "")
    if fingerprint != expected:
        raise ValueError(f"metrics.{name}.pipeline_fingerprint mismatch")


def _validate_metrics(row: dict[str, Any]) -> None:
    metrics = row["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a mapping")
    _require(metrics, ("latency", "energy", "ap"), context="metrics")
    expected = row["pipeline_fingerprint"]
    for name in ("latency", "energy", "ap"):
        metric = metrics[name]
        if not isinstance(metric, dict):
            raise ValueError(f"metrics.{name} must be a mapping")
        _validate_metric_fingerprint(name, metric, expected)

    latency = metrics["latency"]
    if latency["status"] in {"measured", "diagnostic"}:
        p50 = _positive_number(latency.get("p50_ms"), field="metrics.latency.p50_ms")
        p90 = _positive_number(latency.get("p90_ms"), field="metrics.latency.p90_ms")
        if p90 < p50:
            raise ValueError("metrics.latency.p90_ms must be >= p50_ms")
    energy = metrics["energy"]
    if energy["status"] in {"measured", "diagnostic"}:
        _positive_number(
            energy.get("joules_per_inference"),
            field="metrics.energy.joules_per_inference",
        )
    ap_metric = metrics["ap"]
    if ap_metric["status"] in {"measured", "diagnostic"}:
        ap30 = _probability(ap_metric.get("ap30"), field="metrics.ap.ap30")
        ap50 = _probability(ap_metric.get("ap50"), field="metrics.ap.ap50")
        ap70 = _probability(ap_metric.get("ap70"), field="metrics.ap.ap70")
        if not ap30 >= ap50 >= ap70:
            raise ValueError("metrics.ap must satisfy ap30 >= ap50 >= ap70")


def final_frontier_rejection_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row["statuses"]["build"] != "success":
        reasons.append("build_not_success")
    if row["statuses"]["numerical"] != "pass":
        reasons.append("numerical_not_pass")
    if row["provenance"]["evidence_kind"] != "measured":
        reasons.append("evidence_not_measured")
    if row["q_mode"] in {"int8", "mixed_int8"} and row["pipeline"]["calibration_kind"] != "formal":
        reasons.append("calibration_not_formal")
    if row["pipeline"]["lowering_origin"] not in AUTOMATIC_REALIZATION_ORIGINS:
        reasons.append("lowering_not_automatic")
    for name in ("latency", "energy", "ap"):
        if row["metrics"][name]["status"] != "measured":
            reasons.append(f"{name}_not_measured")
    if row["trusted_for_final_frontier"] is not True:
        reasons.append("not_trusted_for_final_frontier")
    return reasons


def validate_measurement_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("measurement row must be a mapping")
    _require(
        row,
        (
            "schema_version",
            "row_id",
            "model",
            "dataset",
            "checkpoint_id",
            "width",
            "q_mode",
            "mixed_policy_id",
            "capability_profile_id",
            "pipeline",
            "pipeline_fingerprint",
            "metrics",
            "statuses",
            "provenance",
            "trusted_for_final_frontier",
        ),
        context="measurement row",
    )
    if row["schema_version"] != MEASUREMENT_ROW_SCHEMA_VERSION:
        raise ValueError(f"unexpected schema_version: {row['schema_version']!r}")
    for field in ("row_id", "model", "dataset", "checkpoint_id", "capability_profile_id"):
        if not str(row[field] or ""):
            raise ValueError(f"{field} must be non-empty")
    width = row["width"]
    if not isinstance(width, list) or len(width) != 3:
        raise ValueError("width must be a three-element list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in width):
        raise ValueError("width values must be positive integers")
    if row["q_mode"] not in VALID_Q_MODES:
        raise ValueError(f"unsupported q_mode: {row['q_mode']!r}")
    if row["q_mode"] in {"fp32", "fp16"} and row["mixed_policy_id"] != "none":
        raise ValueError("FP32/FP16 rows require mixed_policy_id=none")
    _validate_pipeline(row)
    expected_fingerprint = compute_pipeline_fingerprint(row)
    if row["pipeline_fingerprint"] != expected_fingerprint:
        raise ValueError("pipeline_fingerprint does not match canonical pipeline identity")

    statuses = row["statuses"]
    if not isinstance(statuses, dict):
        raise ValueError("statuses must be a mapping")
    _require(statuses, ("build", "numerical"), context="statuses")
    if statuses["build"] not in VALID_BUILD_STATUSES:
        raise ValueError(f"invalid build status: {statuses['build']!r}")
    if statuses["numerical"] not in VALID_NUMERICAL_STATUSES:
        raise ValueError(f"invalid numerical status: {statuses['numerical']!r}")

    provenance = row["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a mapping")
    _require(provenance, ("evidence_kind", "source_artifacts"), context="provenance")
    evidence_kind = provenance["evidence_kind"]
    if evidence_kind not in VALID_EVIDENCE_KINDS:
        raise ValueError(f"invalid evidence_kind: {evidence_kind!r}")
    source_artifacts = provenance["source_artifacts"]
    if not isinstance(source_artifacts, list) or not source_artifacts:
        raise ValueError("provenance.source_artifacts must be a non-empty list")
    for index, artifact in enumerate(source_artifacts):
        if not isinstance(artifact, dict):
            raise ValueError(f"provenance.source_artifacts[{index}] must be a mapping")
        _require(artifact, ("role", "path", "sha256"), context=f"artifact[{index}]")
        if not str(artifact["role"] or ""):
            raise ValueError(f"artifact[{index}].role must be non-empty")
        artifact_path = Path(str(artifact["path"]))
        if not artifact_path.is_file():
            raise ValueError(f"artifact[{index}] path does not exist: {artifact_path}")
        expected_sha = str(artifact["sha256"] or "")
        if SHA256_RE.fullmatch(expected_sha) is None:
            raise ValueError(f"artifact[{index}].sha256 must be a SHA256 digest")
        if sha256_file(artifact_path) != expected_sha:
            raise ValueError(f"artifact[{index}] SHA256 mismatch: {artifact_path}")

    if row["pipeline"]["lowering_origin"] == "compiler_engine_automatic":
        artifacts_by_role = {str(artifact["role"]): artifact for artifact in source_artifacts}
        engine_role_fields = {
            "engine_build_config": "engine_build_config_sha256",
            "engine_inspector": "engine_inspector_sha256",
            "compiled_engine": "compiled_engine_sha256",
        }
        for role, field in engine_role_fields.items():
            artifact = artifacts_by_role.get(role)
            if artifact is None:
                raise ValueError(f"automatic engine realization requires artifact role {role}")
            if artifact["sha256"] != row["pipeline"][field]:
                raise ValueError(f"artifact role {role} SHA256 must match pipeline.{field}")

    _validate_metrics(row)
    if row["trusted_for_final_frontier"] is True:
        if row["pipeline"]["calibration_kind"] == "diagnostic":
            raise ValueError("diagnostic calibration cannot set trusted_for_final_frontier=true")
        if row["pipeline"]["calibration_kind"] == "historical":
            raise ValueError("historical calibration cannot set trusted_for_final_frontier=true")
        if row["pipeline"]["lowering_origin"] == "explicit_te_constructed":
            raise ValueError("explicit TE lowering cannot set trusted_for_final_frontier=true")
    reasons_without_trust = [
        reason
        for reason in final_frontier_rejection_reasons({**row, "trusted_for_final_frontier": True})
        if reason != "not_trusted_for_final_frontier"
    ]
    if row["trusted_for_final_frontier"] is True and reasons_without_trust:
        if evidence_kind != "measured":
            raise ValueError(
                f"{evidence_kind} evidence cannot set trusted_for_final_frontier=true"
            )
        raise ValueError(
            "trusted_for_final_frontier=true conflicts with row status: "
            + ",".join(reasons_without_trust)
        )
    if not isinstance(row["trusted_for_final_frontier"], bool):
        raise ValueError("trusted_for_final_frontier must be a JSON boolean")
    return copy.deepcopy(row)


def is_final_frontier_eligible(row: dict[str, Any]) -> bool:
    validate_measurement_row(row)
    return not final_frontier_rejection_reasons(row)


def is_search_training_eligible(row: dict[str, Any]) -> bool:
    """Allow measured automatic latency rows into cost-model training before AP completion."""

    validate_measurement_row(row)
    if row["statuses"] != {"build": "success", "numerical": "pass"}:
        return False
    if row["provenance"]["evidence_kind"] != "measured":
        return False
    if row["pipeline"]["lowering_origin"] not in AUTOMATIC_REALIZATION_ORIGINS:
        return False
    if row["q_mode"] in {"int8", "mixed_int8"} and row["pipeline"]["calibration_kind"] != "formal":
        return False
    if any(metric["status"] == "diagnostic" for metric in row["metrics"].values()):
        return False
    return row["metrics"]["latency"]["status"] == "measured"


def filter_final_frontier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [copy.deepcopy(row) for row in rows if is_final_frontier_eligible(row)]


def measurement_evidence_view(row: dict[str, Any]) -> str:
    """Route one validated row into exactly one evidence view."""

    validate_measurement_row(row)
    if row["statuses"]["build"] != "success" or row["statuses"]["numerical"] != "pass":
        return "feasibility_failure"
    if not final_frontier_rejection_reasons(row):
        return "final_frontier"
    if is_search_training_eligible(row):
        return "gold_training"
    return "historical_prior"


def partition_measurement_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build mutually exclusive registry views without mutating source rows."""

    views = {
        "gold_training": [],
        "final_frontier": [],
        "feasibility_failure": [],
        "historical_prior": [],
    }
    for row in rows:
        view = measurement_evidence_view(row)
        views[view] = [*views[view], copy.deepcopy(row)]
    return views


def summarize_measurement_registry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    eligible = 0
    for row in rows:
        validate_measurement_row(row)
        reasons = final_frontier_rejection_reasons(row)
        if not reasons:
            eligible += 1
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "schema_version": REGISTRY_SUMMARY_SCHEMA_VERSION,
        "total_rows": len(rows),
        "final_frontier_eligible_rows": eligible,
        "rejected_rows": len(rows) - eligible,
        "rejection_reason_counts": reason_counts,
    }


def build_artifact_registry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        validate_measurement_row(row)
        for source in row["provenance"]["source_artifacts"]:
            key = (str(source["path"]), str(source["sha256"]))
            existing = artifacts.get(key)
            if existing is None:
                artifacts[key] = {
                    "path": str(source["path"]),
                    "sha256": str(source["sha256"]),
                    "roles": [str(source["role"])],
                    "row_ids": [str(row["row_id"])],
                }
                continue
            if str(source["role"]) not in existing["roles"]:
                existing["roles"] = [*existing["roles"], str(source["role"])]
            if str(row["row_id"]) not in existing["row_ids"]:
                existing["row_ids"] = [*existing["row_ids"], str(row["row_id"])]
    artifact_rows = sorted(artifacts.values(), key=lambda item: (item["path"], item["sha256"]))
    return {
        "schema_version": ARTIFACT_REGISTRY_SCHEMA_VERSION,
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
    }


__all__ = [
    "MEASUREMENT_ROW_SCHEMA_VERSION",
    "build_artifact_registry",
    "compute_pipeline_fingerprint",
    "filter_final_frontier_rows",
    "final_frontier_rejection_reasons",
    "is_final_frontier_eligible",
    "is_search_training_eligible",
    "measurement_evidence_view",
    "partition_measurement_rows",
    "summarize_measurement_registry",
    "sha256_file",
    "validate_measurement_row",
]
