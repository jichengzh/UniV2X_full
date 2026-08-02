#!/usr/bin/env python3
"""Build and close content-addressed F-Cooper Stage6 control requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUEST_SCHEMA = "stage6_fcooper_control_measurement_request_v2"
TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"
SUCCESS = "measured_success_gold"
TERMINAL_FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}
CONTROL_PHASES = {
    ("schedule_only", "measure"): (
        "S6-FCO-TRT-SCHEDULE-ONLY-MEASURE-V2",
        5,
    ),
    ("compression_only", "measure"): (
        "S6-FCO-TRT-COMPRESSION-ONLY-MEASURE-V2",
        0,
    ),
    ("compress_then_tune", "screen"): (
        "S6-FCO-TRT-COMPRESS-THEN-TUNE-SCREEN-V2",
        0,
    ),
    ("compress_then_tune", "tuned_remeasurement"): (
        "S6-FCO-TRT-COMPRESS-THEN-TUNE-TUNED-V2",
        5,
    ),
}
INDEPENDENT_AP_MAX_ABS_DRIFT = 1.0e-3
TRT_ARTIFACT_FILENAMES = {
    "compiled_engine": "compiled.engine",
    "engine_build_config": "engine_build_config.json",
    "engine_inspector": "engine_inspector.json",
    "calibration_cache": "calibration.cache",
    "calibration_manifest": "calibration_manifest.json",
}


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_pilot_reference(value: Any) -> bool:
    if isinstance(value, str):
        return PILOT_FRAGMENT in value
    if isinstance(value, Mapping):
        return any(
            contains_pilot_reference(key) or contains_pilot_reference(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return any(contains_pilot_reference(item) for item in value)
    return False


def reject_pilot_reference(value: Any, *, context: str) -> None:
    if contains_pilot_reference(value):
        raise ValueError(f"{context} contains forbidden pilot provenance")


def read_json(path: Path) -> Any:
    reject_pilot_reference(str(path), context="JSON path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    reject_pilot_reference(payload, context=f"JSON payload {path}")
    return payload


def rows_from_payload(payload: Any, field: str = "rows") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(row) for row in payload[field]]
    raise ValueError(f"expected a list or an object containing {field}")


def write_json(path: Path, payload: Any) -> None:
    reject_pilot_reference(str(path), context="output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def _validate_source_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows]
    reject_pilot_reference(selected, context="control source rows")
    identities = [_row_id(row) for row in selected]
    if not selected:
        raise ValueError("control request source rows cannot be empty")
    if any(not identity for identity in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("control request rows require unique non-empty identities")
    for row in selected:
        width = row.get("width")
        if (
            row.get("model") != "fcooper"
            or row.get("hardware_id") != "h800"
            or not isinstance(width, Sequence)
            or isinstance(width, (str, bytes))
            or len(width) != 5
            or row.get("q_mode") not in {"fp32", "fp16", "int8"}
        ):
            raise ValueError("control source row contract drift")
    return selected


def _planned_row_ids(
    five_arm_plan: Mapping[str, Any], *, arm_id: str, phase: str
) -> list[str]:
    if (
        five_arm_plan.get("schema_version")
        != "stage6_fcooper_five_arm_plan_v2"
        or five_arm_plan.get("task_id") != "S5-FCO-TRT-V2"
    ):
        raise ValueError("invalid Stage6 five-arm plan contract")
    arm = (five_arm_plan.get("arms") or {}).get(arm_id)
    if not isinstance(arm, Mapping):
        raise ValueError(f"five-arm plan lacks arm {arm_id}")
    if arm_id == "compression_only" and phase == "measure":
        values = arm.get("selected_row_ids")
    elif arm_id == "compress_then_tune" and phase == "screen":
        values = arm.get("screen_row_ids")
    elif arm_id == "compress_then_tune" and phase == "tuned_remeasurement":
        values = arm.get("tuned_row_ids")
    elif arm_id == "schedule_only" and phase == "measure":
        values = [arm.get("fixed_row_id")]
    else:
        raise ValueError(f"unsupported five-arm plan phase: {arm_id}/{phase}")
    identities = [str(value) for value in values]
    if any(not value for value in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("five-arm plan row identities are invalid")
    return identities


def validate_candidate_plan_against_five_arm_plan(
    candidate_plan: Mapping[str, Any],
    five_arm_plan: Mapping[str, Any],
    *,
    arm_id: str,
    phase: str,
    builder_optimization_level: int,
) -> list[dict[str, Any]]:
    rows = _validate_source_rows(rows_from_payload(candidate_plan))
    identities = [_row_id(row) for row in rows]
    if (
        candidate_plan.get("schema_version")
        != "stage6_fcooper_arm_candidate_plan_v2"
        or candidate_plan.get("task_id") != "S5-FCO-TRT-V2"
        or candidate_plan.get("arm_id") != arm_id
        or candidate_plan.get("phase") != phase
        or int(candidate_plan.get("builder_optimization_level", -1))
        != int(builder_optimization_level)
        or int(candidate_plan.get("row_count", -1)) != len(rows)
    ):
        raise ValueError("candidate plan metadata does not match five-arm plan")
    planned = _planned_row_ids(five_arm_plan, arm_id=arm_id, phase=phase)
    if arm_id == "schedule_only":
        row = rows[0] if len(rows) == 1 else {}
        schedule_plan = (five_arm_plan.get("arms") or {}).get("schedule_only") or {}
        if (
            identities != planned
            or row.get("manifest_job_id")
            != schedule_plan.get("fixed_manifest_job_id")
            or row.get("schedule_baseline_derivation")
            != schedule_plan.get("schedule_baseline_derivation")
            or row.get("schedule_baseline_derivation_sha256")
            != schedule_plan.get("schedule_baseline_derivation_sha256")
            or tuple(int(value) for value in row.get("width") or ())
            != (64, 128, 256, 128, 256)
            or row.get("q_mode") != "fp32"
        ):
            raise ValueError("schedule-only candidate does not match five-arm plan")
    elif identities != planned:
        raise ValueError("candidate rows do not match five-arm plan selection")
    return rows


def build_control_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm_id: str,
    phase: str,
    builder_optimization_level: int,
) -> list[dict[str, Any]]:
    key = (arm_id, phase)
    if key not in CONTROL_PHASES:
        raise ValueError(f"unsupported Stage6 control arm/phase: {key}")
    task_id, expected_builder = CONTROL_PHASES[key]
    if int(builder_optimization_level) != expected_builder:
        raise ValueError(
            f"builder level drift for {arm_id}/{phase}: "
            f"expected {expected_builder}, got {builder_optimization_level}"
        )
    selected = _validate_source_rows(rows)
    task_contract = {
        "schema_version": "stage6_fcooper_control_task_contract_v2",
        "task_id": task_id,
        "arm_id": arm_id,
        "phase": phase,
        "model": "fcooper",
        "hardware_id": "h800",
        "backend": "trt",
        "builder_optimization_level": expected_builder,
        "runner_request_kind": "stage6-control",
        "recovery_contract_required_for_pruned_rows": True,
    }
    task_sha256 = canonical_sha256(task_contract)
    rebound = []
    for source in selected:
        source_row_id = _row_id(source)
        rebound.append(
            {
                **source,
                "row_id": source_row_id,
                "manifest_job_id": source_row_id,
                "control_source_task_id": source.get("task_id"),
                "control_source_task_sha256": source.get("task_sha256"),
                "task_id": task_id,
                "task_sha256": task_sha256,
                "arm_id": arm_id,
                "phase": phase,
                "builder_optimization_level": expected_builder,
            }
        )

    requests = []
    for request_index, offset in enumerate(range(0, len(rebound), 4)):
        batch = rebound[offset : offset + 4]
        row_sha256 = {_row_id(row): canonical_sha256(row) for row in batch}
        request = {
            "schema_version": REQUEST_SCHEMA,
            "task_id": task_id,
            "task_sha256": task_sha256,
            "task_contract": task_contract,
            "arm_id": arm_id,
            "phase": phase,
            "request_index": request_index,
            "batch_size": len(batch),
            "atomic_feedback": True,
            "real_h800_measurement_required": True,
            "builder_optimization_level": expected_builder,
            "row_sha256": row_sha256,
            "rows": batch,
        }
        requests.append(
            {
                **request,
                "measurement_request_sha256": canonical_sha256(request),
            }
        )
    return requests


def write_control_requests(
    requests: Sequence[Mapping[str, Any]], *, output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for request in requests:
        request_sha = str(request["measurement_request_sha256"])
        request_index = int(request["request_index"])
        path = output_dir / f"request_{request_index:02d}_{request_sha[:16]}.json"
        write_json(path, request)
        entries.append(
            {
                "request_index": request_index,
                "path": str(path.resolve()),
                "measurement_request_sha256": request_sha,
                "batch_size": int(request["batch_size"]),
            }
        )
    manifest = {
        "schema_version": "stage6_fcooper_control_request_manifest_v2",
        "task_id": requests[0]["task_id"],
        "arm_id": requests[0]["arm_id"],
        "phase": requests[0]["phase"],
        "request_count": len(entries),
        "row_count": sum(entry["batch_size"] for entry in entries),
        "requests": entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output_dir / "request_manifest.json", manifest)
    return manifest


def requests_from_manifest(path: Path) -> list[dict[str, Any]]:
    manifest = read_json(path)
    recorded = manifest.pop("manifest_sha256", None)
    if recorded != canonical_sha256(manifest):
        raise ValueError("control request manifest SHA drift")
    requests = []
    for entry in manifest.get("requests") or []:
        request_path = Path(str(entry["path"]))
        request = read_json(request_path)
        if (
            file_sha256(request_path) == ""
            or request.get("measurement_request_sha256")
            != entry.get("measurement_request_sha256")
            or int(request.get("batch_size") or 0) != int(entry.get("batch_size") or 0)
        ):
            raise ValueError("control request manifest binding drift")
        requests.append(request)
    if (
        len(requests) != int(manifest.get("request_count") or -1)
        or sum(len(request.get("rows") or []) for request in requests)
        != int(manifest.get("row_count") or -1)
    ):
        raise ValueError("control request manifest count drift")
    return requests


def _validate_request(request: Mapping[str, Any]) -> None:
    from scripts.fcooper_execute_measurement_row_v2 import validate_control_request

    rows = request.get("rows") or []
    for row_index in range(len(rows)):
        validate_control_request(dict(request), row_index=row_index)
    task_contract = request.get("task_contract")
    if (
        not isinstance(task_contract, Mapping)
        or request.get("task_sha256") != canonical_sha256(task_contract)
        or task_contract.get("task_id") != request.get("task_id")
        or task_contract.get("arm_id") != request.get("arm_id")
        or task_contract.get("phase") != request.get("phase")
        or int(task_contract.get("builder_optimization_level", -1))
        != int(request.get("builder_optimization_level", -2))
    ):
        raise ValueError("Stage6 control task contract SHA drift")


def _feedback_path(
    artifact_root: Path, *, task_id: str, row_id: str
) -> Path:
    row_tag = hashlib.sha256(f"{task_id}:{row_id}".encode()).hexdigest()[:16]
    return artifact_root / "execution" / row_tag / "feedback_row.json"


def _validate_feedback_sha(row: Mapping[str, Any]) -> None:
    payload = dict(row)
    recorded = payload.pop("actual_feedback_row_sha256", None)
    if recorded != canonical_sha256(payload):
        raise ValueError("actual feedback row SHA drift")


def _finite(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"feedback has non-finite {name}")
    return value


def _validate_terminal_feedback(row: Mapping[str, Any]) -> None:
    status = row.get("terminal_status")
    if status == SUCCESS:
        for metric in ("ap70", "latency_ms", "energy_j"):
            _finite(row, metric)
        return
    if status in TERMINAL_FAILURES and str(row.get("failure_reason") or "").strip():
        return
    raise ValueError("feedback is not a credible terminal observation")


def aggregate_feedback(
    requests: Sequence[Mapping[str, Any]],
    *,
    artifact_root: Path,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reject_pilot_reference(str(artifact_root), context="artifact root")
    expected: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for request in requests:
        _validate_request(request)
        expected.extend((dict(row), request) for row in request["rows"])
    identities = [_row_id(row) for row, _ in expected]
    if (
        len(expected) != expected_count
        or len(set(identities)) != expected_count
        or any(not identity for identity in identities)
    ):
        raise ValueError("control feedback cannot be complete for the expected arm")

    feedback_rows = []
    feedback_paths = []
    for request in requests:
        for row_index in range(len(request["rows"])):
            feedback, path = validate_bound_feedback_row(
                request,
                row_index=row_index,
                artifact_root=artifact_root,
            )
            feedback_rows.append(feedback)
            feedback_paths.append(path)
    audit = {
        "schema_version": "stage6_fcooper_control_file_integrity_audit_v2",
        "task_ids": sorted({str(row["task_id"]) for row in feedback_rows}),
        "passed": True,
        "row_count": len(feedback_rows),
        "row_ids": [_row_id(row) for row in feedback_rows],
        "feedback_paths": [str(path.resolve()) for path in feedback_paths],
        "feedback_sha256": [file_sha256(path) for path in feedback_paths],
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return feedback_rows, audit


def validate_bound_feedback_row(
    request: Mapping[str, Any],
    *,
    row_index: int,
    artifact_root: Path,
) -> tuple[dict[str, Any], Path]:
    _validate_request(request)
    if row_index < 0 or row_index >= len(request["rows"]):
        raise IndexError(f"control row index out of range: {row_index}")
    source = dict(request["rows"][row_index])
    row_id = _row_id(source)
    path = _feedback_path(
        artifact_root, task_id=str(request["task_id"]), row_id=row_id
    )
    if not path.is_file():
        raise ValueError(f"control feedback is incomplete; missing bound row {row_id}")
    feedback = dict(read_json(path))
    _validate_feedback_sha(feedback)
    _validate_terminal_feedback(feedback)
    if (
        _row_id(feedback) != row_id
        or feedback.get("task_id") != request.get("task_id")
        or feedback.get("task_sha256") != request.get("task_sha256")
        or feedback.get("arm_id") != request.get("arm_id")
        or feedback.get("phase") != request.get("phase")
        or feedback.get("measurement_request_row_sha256")
        != request["row_sha256"][row_id]
        or int(feedback.get("builder_optimization_level", -1))
        != int(request.get("builder_optimization_level", -2))
    ):
        raise ValueError("control feedback request binding drift")
    return feedback, path


def select_complete_arm_winner(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    ap70_ref: float,
    max_ap_drop: float,
    required_task_id: str | None = None,
) -> dict[str, Any]:
    terminal = [dict(row) for row in rows]
    identities = [_row_id(row) for row in terminal]
    if (
        len(terminal) != expected_count
        or len(set(identities)) != expected_count
        or any(not identity for identity in identities)
    ):
        raise ValueError("winner selection requires complete unique arm feedback")
    for row in terminal:
        reject_pilot_reference(row, context="winner candidate")
        _validate_terminal_feedback(row)
        if required_task_id is not None and row.get("task_id") != required_task_id:
            raise ValueError("winner candidate task identity drift")
    successful = [row for row in terminal if row["terminal_status"] == SUCCESS]
    if not successful:
        failure = min(terminal, key=_row_id)
        return {**failure, "selection_status": "all_candidates_terminal_failure"}
    floor = float(ap70_ref) - float(max_ap_drop)
    feasible = [row for row in successful if float(row["ap70"]) >= floor]
    pool = feasible or successful
    minimum_latency = min(float(row["latency_ms"]) for row in pool)
    tied = [
        row
        for row in pool
        if float(row["latency_ms"]) <= minimum_latency * 1.01
    ]
    selected = min(
        tied,
        key=lambda row: (
            float(row["energy_j"]),
            float(row["latency_ms"]),
            _row_id(row),
        ),
    )
    return {
        **selected,
        "ap70_floor": floor,
        "ap_constraint_satisfied": bool(feasible),
        "selection_status": (
            "selected_feasible"
            if feasible
            else "selected_fastest_ap_constraint_violation"
        ),
    }


def repeat_spec(selection: Mapping[str, Any]) -> dict[str, Any]:
    if selection.get("terminal_status") != SUCCESS:
        return {"terminal_status": selection.get("terminal_status")}
    evidence_path = Path(str(selection["materialized_source_evidence_path"]))
    if (
        not evidence_path.is_file()
        or file_sha256(evidence_path)
        != selection.get("materialized_source_evidence_sha256")
    ):
        raise ValueError("selected repeat source-evidence SHA drift")
    evidence = read_json(evidence_path)
    onnx_path = Path(str(evidence["onnx_path"]))
    config_path = Path(str(evidence["config_path"]))
    checkpoint_path = Path(str(evidence["checkpoint_path"]))
    if (
        not onnx_path.is_file()
        or file_sha256(onnx_path) != evidence.get("onnx_sha256")
        or evidence.get("onnx_sha256")
        != (selection.get("graph_features") or {}).get("onnx_sha256")
        or not config_path.is_file()
        or file_sha256(config_path) != evidence.get("config_sha256")
        or not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != evidence.get("checkpoint_sha256")
        or evidence.get("checkpoint_sha256") != selection.get("checkpoint_sha256")
    ):
        raise ValueError("selected repeat ONNX evidence binding drift")
    width = tuple(int(value) for value in selection.get("width") or ())
    if len(width) != 5:
        raise ValueError("selected repeat is missing the five-width genome")
    if width != (64, 128, 256, 128, 256):
        recovery_path = Path(str(evidence.get("recovery_training_report_path") or ""))
        recovery_sha = evidence.get("recovery_training_report_sha256")
        if (
            not recovery_path.is_file()
            or file_sha256(recovery_path) != recovery_sha
            or recovery_sha != selection.get("recovery_training_report_sha256")
        ):
            raise ValueError("selected pruned repeat recovery evidence drift")
        recovery = read_json(recovery_path)
        from scripts.fcooper_execute_measurement_row_v2 import (
            validate_recovery_training_evidence,
        )

        validate_recovery_training_evidence(
            report_path=recovery_path,
            recovery_contract_path=Path(
                str(recovery.get("recovery_contract_path") or "")
            ),
            config_path=Path(str(recovery.get("config_path") or "")),
            initial_checkpoint_path=Path(
                str(recovery.get("initial_checkpoint_path") or "")
            ),
            recovered_checkpoint_path=Path(
                str(recovery.get("recovered_checkpoint_path") or "")
            ),
        )
    return {
        "terminal_status": SUCCESS,
        "row_id": _row_id(selection),
        "q_mode": str(selection["q_mode"]),
        "onnx_path": str(onnx_path.resolve()),
        "onnx_sha256": str(evidence["onnx_sha256"]),
        "config_path": str(config_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }


def bind_native_report(
    report: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    gpu_abs: int,
) -> dict[str, Any]:
    reject_pilot_reference(report, context="native repeat report")
    if (
        int(gpu_abs) != 7
        or report.get("status") != "success"
        or report.get("precision") != "fp32"
        or report.get("checkpoint_sha256") != contract.get("checkpoint_sha256")
        or report.get("config_sha256") != contract.get("config_sha256")
    ):
        raise ValueError("native repeat evidence binding drift")
    bound = {
        key: value
        for key, value in report.items()
        if key != "native_repeat_binding_sha256"
    }
    bound["gpu_abs"] = 7
    bound["native_repeat_binding_sha256"] = canonical_sha256(bound)
    return bound


def validate_trt_repeat_report(
    report: Mapping[str, Any],
    *,
    expected_onnx_sha256: str,
    expected_precision: str,
    expected_builder_level: int,
    expected_gpu: int,
    artifact_dir: Path,
) -> dict[str, Any]:
    reject_pilot_reference(report, context="TRT repeat report")
    artifacts = report.get("artifact_sha256")
    if (
        int(expected_gpu) != 7
        or int(report.get("gpu_abs", -1)) != expected_gpu
        or report.get("precision") != expected_precision
        or int(report.get("builder_optimization_level", -1))
        != expected_builder_level
    ):
        raise ValueError("TRT repeat GPU, precision, or builder binding drift")
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("source_onnx") != expected_onnx_sha256
        or not isinstance(artifacts.get("compiled_engine"), str)
        or len(str(artifacts["compiled_engine"])) != 64
    ):
        raise ValueError("TRT repeat artifact SHA binding drift")
    verified_artifacts: dict[str, str] = {}
    for artifact_name, recorded_sha256 in artifacts.items():
        if artifact_name == "source_onnx":
            continue
        filename = TRT_ARTIFACT_FILENAMES.get(str(artifact_name))
        if filename is None:
            raise ValueError(f"TRT repeat has unknown artifact binding: {artifact_name}")
        artifact_path = artifact_dir / filename
        if (
            not artifact_path.is_file()
            or not isinstance(recorded_sha256, str)
            or file_sha256(artifact_path) != recorded_sha256
        ):
            raise ValueError(f"TRT repeat artifact file or SHA drift: {artifact_name}")
        verified_artifacts[str(artifact_name)] = str(artifact_path.resolve())
    if {
        "compiled_engine",
        "engine_build_config",
        "engine_inspector",
    } - set(verified_artifacts):
        raise ValueError("TRT repeat is missing required artifact files")
    build_config = json.loads(
        (artifact_dir / TRT_ARTIFACT_FILENAMES["engine_build_config"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(build_config, Mapping)
        or build_config.get("source_onnx_sha256") != expected_onnx_sha256
        or build_config.get("precision") != expected_precision
        or int(build_config.get("builder_optimization_level", -1))
        != expected_builder_level
        or build_config.get("calibration_dataset") != "OPV2V-validate"
    ):
        raise ValueError("TRT repeat build-config semantic drift")
    if expected_precision == "int8":
        if (
            {"calibration_cache", "calibration_manifest"}
            - set(verified_artifacts)
            or build_config.get("calibration_manifest_sha256")
            != artifacts.get("calibration_manifest")
        ):
            raise ValueError("TRT repeat INT8 calibration semantic drift")
    elif build_config.get("calibration_manifest_sha256") is not None:
        raise ValueError("TRT repeat non-INT8 calibration semantic drift")
    for metric in ("lat_p50_ms", "energy_j"):
        _finite(report, metric)
    return {
        "schema_version": "stage6_fcooper_trt_repeat_reuse_audit_v2",
        "passed": True,
        "gpu_abs": 7,
        "precision": expected_precision,
        "builder_optimization_level": expected_builder_level,
        "source_onnx_sha256": expected_onnx_sha256,
        "compiled_engine_sha256": artifacts["compiled_engine"],
        "verified_artifacts": verified_artifacts,
    }


def validate_independent_ap_report(
    report: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    engine_path: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    reject_pilot_reference(report, context="independent AP report")
    if (
        report.get("status") != "success_full"
        or int(report.get("dataset_samples") or 0) != 2170
        or int(report.get("processed_samples") or 0) != 2170
        or int(report.get("fallback_samples", -1)) != 0
    ):
        raise ValueError("independent AP report does not satisfy full-2170 contract")
    numerical = report.get("numerical_contract")
    if (
        report.get("schema_version") != "fcooper_trt_ap_report_v1"
        or int(report.get("failed_samples", -1)) != 0
        or int(report.get("engine_samples", -1)) != 2170
        or int(report.get("engine_calls", -1)) != 2170
        or not isinstance(numerical, Mapping)
        or int(numerical.get("requested_samples", -1)) != 2170
        or int(numerical.get("fallback_samples", -1)) != 0
        or numerical.get("full_dataset_engine_execution") is not True
        or numerical.get("silent_fallback_forbidden") is not True
    ):
        raise ValueError("independent AP numerical contract drift")
    expected_hashes = {
        "engine_sha256": file_sha256(engine_path),
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    if expected_hashes["checkpoint_sha256"] != expected.get("checkpoint_sha256"):
        raise ValueError("independent AP selected checkpoint binding drift")
    for field, expected_sha256 in expected_hashes.items():
        if report.get(field) != expected_sha256:
            raise ValueError(f"independent AP {field.removesuffix('_sha256')} SHA drift")
    metrics: dict[str, float] = {}
    metric_drift: dict[str, float] = {}
    for metric in ("ap30", "ap50", "ap70"):
        measured = _finite(report, metric)
        reference = _finite(expected, metric)
        drift = abs(measured - reference)
        if drift > INDEPENDENT_AP_MAX_ABS_DRIFT:
            raise ValueError(f"independent AP metric drift: {metric}")
        metrics[metric] = measured
        metric_drift[metric] = drift
    return {
        "schema_version": "stage6_fcooper_independent_ap_reuse_audit_v2",
        "passed": True,
        "processed_samples": 2170,
        "fallback_samples": 0,
        "artifact_sha256": expected_hashes,
        "metric_abs_drift": metric_drift,
        "metric_max_abs_drift": INDEPENDENT_AP_MAX_ABS_DRIFT,
        **metrics,
    }


def validate_final_audit(audit_path: Path) -> dict[str, Any]:
    audit = read_json(audit_path)
    required_true = (
        "paper_ready",
        "all_five_arms_credible_terminal",
        "successful_repeats_verified",
        "resource_guards_verified",
    )
    if (
        audit.get("schema_version") != "stage6_fcooper_five_arm_audit_v2"
        or audit.get("task_id") != TASK_ID
        or len(audit.get("rows") or []) != 5
        or any(audit.get(field) is not True for field in required_true)
    ):
        raise ValueError("final five-arm audit is not paper-ready")
    output_dir = audit_path.resolve().parent
    csv_path = output_dir / "fcooper_stage6_trt_delta_ap_0.10_v2.csv"
    bundle_path = output_dir / "fcooper_stage6_evidence_bundle_v2.json"
    if not csv_path.is_file() or not bundle_path.is_file():
        raise ValueError("final five-arm output set is incomplete")
    bundle = read_json(bundle_path)
    if (
        bundle.get("schema_version") != "fcooper_stage6_evidence_bundle_v2"
        or bundle.get("task_id") != TASK_ID
        or Path(str(bundle.get("audit_path") or "")).resolve()
        != audit_path.resolve()
        or bundle.get("audit_sha256") != file_sha256(audit_path)
        or Path(str(bundle.get("csv_path") or "")).resolve() != csv_path.resolve()
        or bundle.get("csv_sha256") != file_sha256(csv_path)
    ):
        raise ValueError("final five-arm evidence bundle binding drift")
    return dict(audit)


def write_supervisor_complete(
    path: Path, *, final_audit_path: Path
) -> dict[str, Any]:
    if path.exists():
        existing = read_json(path)
        if (
            existing.get("final_audit_path") != str(final_audit_path.resolve())
            or existing.get("final_audit_sha256") != file_sha256(final_audit_path)
        ):
            raise ValueError(f"refusing to overwrite drifted supervisor marker: {path}")
    audit = validate_final_audit(final_audit_path)
    payload = {
        "schema_version": "fcooper_stage6_five_arm_supervisor_state_v2",
        "task_id": TASK_ID,
        "status": "complete",
        "paper_ready": True,
        "final_t16_required": True,
        "gear_source": "feedback_history_final_t16.json",
        "control_max_parallel": 4,
        "repeat_gpu": 7,
        "final_audit_path": str(final_audit_path.resolve()),
        "final_audit_sha256": file_sha256(final_audit_path),
    }
    if path.exists():
        if existing != payload:
            raise ValueError(f"refusing to overwrite drifted supervisor marker: {path}")
        return payload
    write_json(path, payload)
    return payload


def validate_original_contract(
    contract: Mapping[str, Any],
    *,
    source_config: Path,
    source_checkpoint: Path,
    contract_root: Path | None = None,
) -> dict[str, Any]:
    reject_pilot_reference(contract, context="frozen original contract")
    reject_pilot_reference(str(source_config), context="source config")
    reject_pilot_reference(str(source_checkpoint), context="source checkpoint")
    root = contract_root or Path.cwd()

    def contract_path(field: str) -> Path:
        value = contract.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"frozen original contract lacks {field}")
        path = Path(value)
        return path if path.is_absolute() else root / path

    config_evidence = contract_path("config_path")
    checkpoint_evidence = contract_path("checkpoint_path")
    ap_evidence = contract_path("ap_reference_report_path")
    checks = (
        (
            "source config",
            source_config,
            str(contract.get("config_sha256") or ""),
        ),
        (
            "contract config",
            config_evidence,
            str(contract.get("config_sha256") or ""),
        ),
        (
            "source checkpoint",
            source_checkpoint,
            str(contract.get("checkpoint_sha256") or ""),
        ),
        (
            "contract checkpoint",
            checkpoint_evidence,
            str(contract.get("checkpoint_sha256") or ""),
        ),
        (
            "AP reference",
            ap_evidence,
            str(contract.get("ap_reference_report_sha256") or ""),
        ),
    )
    for label, path, expected_sha in checks:
        if (
            len(expected_sha) != 64
            or not path.is_file()
            or file_sha256(path) != expected_sha
        ):
            raise ValueError(f"{label} evidence SHA drift")
    return {
        "schema_version": "stage6_fcooper_original_contract_preflight_v2",
        "passed": True,
        "config_sha256": contract["config_sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "ap_reference_report_sha256": contract["ap_reference_report_sha256"],
    }


def _phase_key(args: argparse.Namespace) -> tuple[str, str]:
    return str(args.arm_id), str(args.phase)


def build_command(args: argparse.Namespace) -> None:
    candidate_plan = read_json(args.candidate_plan_json)
    five_arm_plan = read_json(args.five_arm_plan_json)
    rows = validate_candidate_plan_against_five_arm_plan(
        candidate_plan,
        five_arm_plan,
        arm_id=args.arm_id,
        phase=args.phase,
        builder_optimization_level=args.builder_optimization_level,
    )
    requests = build_control_requests(
        rows,
        arm_id=args.arm_id,
        phase=args.phase,
        builder_optimization_level=args.builder_optimization_level,
    )
    manifest = write_control_requests(requests, output_dir=args.output_dir)
    print(json.dumps(manifest, sort_keys=True))


def aggregate_command(args: argparse.Namespace) -> None:
    requests = requests_from_manifest(args.request_manifest_json)
    rows, audit = aggregate_feedback(
        requests,
        artifact_root=args.artifact_root,
        expected_count=args.expected_count,
    )
    write_json(
        args.feedback_json,
        {
            "schema_version": "stage6_fcooper_control_feedback_history_v2",
            "row_count": len(rows),
            "rows": rows,
        },
    )
    write_json(args.integrity_audit_json, audit)
    print(json.dumps(audit, sort_keys=True))


def select_command(args: argparse.Namespace) -> None:
    rows = rows_from_payload(read_json(args.feedback_json))
    contract = read_json(args.contract_json)
    selected = select_complete_arm_winner(
        rows,
        expected_count=args.expected_count,
        ap70_ref=float(contract["ap70_ref"]),
        max_ap_drop=args.max_ap_drop,
        required_task_id=args.required_task_id,
    )
    write_json(args.output_json, selected)
    print(json.dumps(selected, sort_keys=True))


def validate_t16_command(args: argparse.Namespace) -> None:
    rows = rows_from_payload(read_json(args.feedback_json))
    for row in rows:
        if row.get("training_source") != "online_feedback":
            raise ValueError("GEAR T16 input must contain online feedback only")
    contract = read_json(args.contract_json)
    selected = select_complete_arm_winner(
        rows,
        expected_count=16,
        ap70_ref=float(contract["ap70_ref"]),
        max_ap_drop=args.max_ap_drop,
        required_task_id="S5-FCO-TRT-V2",
    )
    write_json(args.output_json, selected)
    print(json.dumps(selected, sort_keys=True))


def list_requests_command(args: argparse.Namespace) -> None:
    manifest = read_json(args.request_manifest_json)
    recorded = manifest.pop("manifest_sha256", None)
    if recorded != canonical_sha256(manifest):
        raise ValueError("control request manifest SHA drift")
    for entry in manifest.get("requests") or []:
        print(entry["path"])


def request_count_command(args: argparse.Namespace) -> None:
    request = read_json(args.request_json)
    _validate_request(request)
    print(len(request["rows"]))


def validate_request_complete_command(args: argparse.Namespace) -> None:
    request = read_json(args.request_json)
    rows, audit = aggregate_feedback(
        [request],
        artifact_root=args.artifact_root,
        expected_count=len(request["rows"]),
    )
    print(
        json.dumps(
            {
                **audit,
                "request_complete": True,
                "row_ids": [_row_id(row) for row in rows],
            },
            sort_keys=True,
        )
    )


def validate_request_row_complete_command(args: argparse.Namespace) -> None:
    feedback, path = validate_bound_feedback_row(
        read_json(args.request_json),
        row_index=args.row_index,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "request_row_complete": True,
                "row_id": _row_id(feedback),
                "feedback_path": str(path.resolve()),
                "feedback_sha256": file_sha256(path),
            },
            sort_keys=True,
        )
    )


def repeat_spec_command(args: argparse.Namespace) -> None:
    spec = repeat_spec(read_json(args.selection_json))
    for field in (
        "terminal_status",
        "row_id",
        "q_mode",
        "onnx_path",
        "onnx_sha256",
        "config_path",
        "checkpoint_path",
    ):
        print(spec.get(field, ""))


def bind_native_command(args: argparse.Namespace) -> None:
    bound = bind_native_report(
        read_json(args.report_json),
        contract=read_json(args.contract_json),
        gpu_abs=args.gpu,
    )
    write_json(args.report_json, bound)
    print(json.dumps(bound, sort_keys=True))


def validate_trt_repeat_command(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir or args.report_json.resolve().parent / "engine"
    audit = validate_trt_repeat_report(
        read_json(args.report_json),
        expected_onnx_sha256=args.expected_onnx_sha256,
        expected_precision=args.expected_precision,
        expected_builder_level=args.expected_builder_level,
        expected_gpu=args.gpu,
        artifact_dir=artifact_dir,
    )
    print(json.dumps(audit, sort_keys=True))


def validate_independent_ap_command(args: argparse.Namespace) -> None:
    selection = read_json(args.selection_json)
    source = repeat_spec(selection)
    row_root = args.report_json.resolve().parent
    audit = validate_independent_ap_report(
        read_json(args.report_json),
        expected=selection,
        engine_path=(
            args.engine_path
            or row_root / "same_gpu7_repeat_0/engine/compiled.engine"
        ),
        config_path=args.config_path or Path(source["config_path"]),
        checkpoint_path=args.checkpoint_path or Path(source["checkpoint_path"]),
    )
    print(json.dumps(audit, sort_keys=True))


def validate_final_audit_command(args: argparse.Namespace) -> None:
    print(json.dumps(validate_final_audit(args.audit_json), sort_keys=True))


def write_supervisor_complete_command(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            write_supervisor_complete(
                args.output_json, final_audit_path=args.audit_json
            ),
            sort_keys=True,
        )
    )


def validate_original_command(args: argparse.Namespace) -> None:
    contract = read_json(args.contract_json)
    audit = validate_original_contract(
        contract,
        source_config=args.source_config,
        source_checkpoint=args.source_checkpoint,
        contract_root=args.contract_json.resolve().parent,
    )
    if args.output_json is not None:
        write_json(args.output_json, audit)
    print(json.dumps(audit, sort_keys=True))


def combine_audits_command(args: argparse.Namespace) -> None:
    audits = [read_json(path) for path in args.audit_json]
    if not audits or any(audit.get("passed") is not True for audit in audits):
        raise ValueError("cannot combine failed control integrity audits")
    combined = {
        "schema_version": "stage6_fcooper_control_file_integrity_audit_v2",
        "passed": True,
        "row_count": sum(int(audit["row_count"]) for audit in audits),
        "component_audit_paths": [str(path.resolve()) for path in args.audit_json],
        "component_audit_sha256": [file_sha256(path) for path in args.audit_json],
    }
    combined["audit_sha256"] = canonical_sha256(combined)
    write_json(args.output_json, combined)
    print(json.dumps(combined, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--candidate-plan-json", type=Path, required=True)
    build.add_argument("--five-arm-plan-json", type=Path, required=True)
    build.add_argument("--arm-id", required=True)
    build.add_argument("--phase", required=True)
    build.add_argument("--builder-optimization-level", type=int, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(handler=build_command)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--request-manifest-json", type=Path, required=True)
    aggregate.add_argument("--artifact-root", type=Path, required=True)
    aggregate.add_argument("--expected-count", type=int, required=True)
    aggregate.add_argument("--feedback-json", type=Path, required=True)
    aggregate.add_argument("--integrity-audit-json", type=Path, required=True)
    aggregate.set_defaults(handler=aggregate_command)

    select = subparsers.add_parser("select")
    select.add_argument("--feedback-json", type=Path, required=True)
    select.add_argument("--expected-count", type=int, required=True)
    select.add_argument("--contract-json", type=Path, required=True)
    select.add_argument("--max-ap-drop", type=float, default=0.10)
    select.add_argument("--required-task-id")
    select.add_argument("--output-json", type=Path, required=True)
    select.set_defaults(handler=select_command)

    validate_t16 = subparsers.add_parser("validate-t16")
    validate_t16.add_argument("--feedback-json", type=Path, required=True)
    validate_t16.add_argument("--contract-json", type=Path, required=True)
    validate_t16.add_argument("--max-ap-drop", type=float, default=0.10)
    validate_t16.add_argument("--output-json", type=Path, required=True)
    validate_t16.set_defaults(handler=validate_t16_command)

    list_requests = subparsers.add_parser("list-requests")
    list_requests.add_argument("--request-manifest-json", type=Path, required=True)
    list_requests.set_defaults(handler=list_requests_command)

    request_count = subparsers.add_parser("request-row-count")
    request_count.add_argument("--request-json", type=Path, required=True)
    request_count.set_defaults(handler=request_count_command)

    request_complete = subparsers.add_parser("validate-request-complete")
    request_complete.add_argument("--request-json", type=Path, required=True)
    request_complete.add_argument("--artifact-root", type=Path, required=True)
    request_complete.set_defaults(handler=validate_request_complete_command)

    request_row_complete = subparsers.add_parser(
        "validate-request-row-complete"
    )
    request_row_complete.add_argument("--request-json", type=Path, required=True)
    request_row_complete.add_argument("--row-index", type=int, required=True)
    request_row_complete.add_argument("--artifact-root", type=Path, required=True)
    request_row_complete.set_defaults(
        handler=validate_request_row_complete_command
    )

    repeat = subparsers.add_parser("repeat-spec")
    repeat.add_argument("--selection-json", type=Path, required=True)
    repeat.set_defaults(handler=repeat_spec_command)

    native = subparsers.add_parser("bind-native-report")
    native.add_argument("--report-json", type=Path, required=True)
    native.add_argument("--contract-json", type=Path, required=True)
    native.add_argument("--gpu", type=int, required=True)
    native.set_defaults(handler=bind_native_command)

    trt_repeat = subparsers.add_parser("validate-trt-repeat")
    trt_repeat.add_argument("--report-json", type=Path, required=True)
    trt_repeat.add_argument("--expected-onnx-sha256", required=True)
    trt_repeat.add_argument(
        "--expected-precision", choices=("fp16", "fp32", "int8"), required=True
    )
    trt_repeat.add_argument("--expected-builder-level", type=int, required=True)
    trt_repeat.add_argument("--gpu", type=int, required=True)
    trt_repeat.add_argument("--artifact-dir", type=Path)
    trt_repeat.set_defaults(handler=validate_trt_repeat_command)

    independent_ap = subparsers.add_parser("validate-independent-ap")
    independent_ap.add_argument("--report-json", type=Path, required=True)
    independent_ap.add_argument("--selection-json", type=Path, required=True)
    independent_ap.add_argument("--engine-path", type=Path)
    independent_ap.add_argument("--config-path", type=Path)
    independent_ap.add_argument("--checkpoint-path", type=Path)
    independent_ap.set_defaults(handler=validate_independent_ap_command)

    final_audit = subparsers.add_parser("validate-final-audit")
    final_audit.add_argument("--audit-json", type=Path, required=True)
    final_audit.set_defaults(handler=validate_final_audit_command)

    complete = subparsers.add_parser("write-supervisor-complete")
    complete.add_argument("--audit-json", type=Path, required=True)
    complete.add_argument("--output-json", type=Path, required=True)
    complete.set_defaults(handler=write_supervisor_complete_command)

    original = subparsers.add_parser("validate-original-contract")
    original.add_argument("--contract-json", type=Path, required=True)
    original.add_argument("--source-config", type=Path, required=True)
    original.add_argument("--source-checkpoint", type=Path, required=True)
    original.add_argument("--output-json", type=Path)
    original.set_defaults(handler=validate_original_command)

    combine = subparsers.add_parser("combine-audits")
    combine.add_argument("--audit-json", type=Path, action="append", required=True)
    combine.add_argument("--output-json", type=Path, required=True)
    combine.set_defaults(handler=combine_audits_command)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
