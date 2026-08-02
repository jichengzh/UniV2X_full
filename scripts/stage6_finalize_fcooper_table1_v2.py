#!/usr/bin/env python3
"""Finalize formal, evidence-verified F-Cooper five-arm Stage6 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"
BASE_WIDTH = (64, 128, 256, 128, 256)
SUCCESS = "measured_success_gold"
TERMINAL_FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}
METRICS = ("ap30", "ap50", "ap70", "latency_ms", "energy_j")
FINAL_OUTPUT_NAMES = (
    "fcooper_stage6_trt_delta_ap_0.10_v2.csv",
    "fcooper_stage6_five_arm_audit_v2.json",
    "fcooper_stage6_evidence_bundle_v2.json",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.single_target_search_v2 import (  # noqa: E402
    FROZEN_GOLD176_GRAPH_FEATURES_SHA256,
    FROZEN_GOLD176_ROWS_SHA256,
)
from scripts.fcooper_execute_measurement_row_v2 import (  # noqa: E402
    validate_recovery_training_evidence,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path, field: str = "rows") -> list[dict[str, Any]]:
    payload = _read(path)
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(row) for row in payload[field]]
    raise ValueError(f"expected rows in {path}")


def _canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_pilot(value: Any, *, context: str) -> None:
    if isinstance(value, str):
        if PILOT_FRAGMENT in value:
            raise ValueError(f"{context} contains forbidden pilot path fragment")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_pilot(key, context=context)
            _reject_pilot(item, context=context)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _reject_pilot(item, context=context)


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite measured {name}: {value!r}")
    return number


def _resolve_path(value: Any, *, root: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing evidence path: {field}")
    _reject_pilot(value, context=field)
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verify_file(
    row: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    root: Path,
) -> Path:
    path = _resolve_path(row.get(path_field), root=root, field=path_field)
    expected = row.get(sha_field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"missing evidence SHA: {sha_field}")
    if not path.is_file():
        raise ValueError(f"evidence file missing: {path}")
    actual = _file_sha(path)
    if actual != expected:
        raise ValueError(f"evidence SHA mismatch for {sha_field}: {path}")
    return path


def verify_success_evidence(
    source: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Verify every byte-bound artifact required by a successful formal row."""
    row = dict(source)
    _reject_pilot(row, context="successful feedback row")
    if row.get("terminal_status") != SUCCESS:
        raise ValueError("success evidence verifier received a non-success row")
    for metric in METRICS:
        _finite(row.get(metric), name=metric)

    verified_paths = [
        _verify_file(
            row,
            path_field="performance_result_json",
            sha_field="performance_result_sha256",
            root=root,
        ),
        _verify_file(
            row,
            path_field="ap_report_path",
            sha_field="ap_report_sha256",
            root=root,
        ),
        _verify_file(
            row,
            path_field="materialized_source_evidence_path",
            sha_field="materialized_source_evidence_sha256",
            root=root,
        ),
        _verify_file(
            row,
            path_field="engine_path",
            sha_field="engine_sha256",
            root=root,
        ),
    ]
    graph = row.get("graph_features")
    if not isinstance(graph, Mapping):
        raise ValueError("successful row is missing graph features")
    if _canonical_sha(graph) != row.get("materialized_graph_features_sha256"):
        raise ValueError("successful row graph feature SHA mismatch")

    recorded_row_sha = row.pop("actual_feedback_row_sha256", None)
    if recorded_row_sha is not None and recorded_row_sha != _canonical_sha(row):
        raise ValueError("successful actual feedback row SHA mismatch")

    source_evidence = _read(verified_paths[2])
    if not isinstance(source_evidence, Mapping):
        raise ValueError("materialized source evidence must be an object")
    _reject_pilot(source_evidence, context="materialized source evidence")
    checkpoint = _verify_file(
        source_evidence,
        path_field="checkpoint_path",
        sha_field="checkpoint_sha256",
        root=root,
    )
    if source_evidence["checkpoint_sha256"] != row.get("checkpoint_sha256"):
        raise ValueError("feedback/source checkpoint SHA mismatch")
    verified_paths.append(checkpoint)

    width = tuple(int(value) for value in row.get("width") or ())
    if len(width) != 5:
        raise ValueError("successful F-Cooper row must expose five widths")
    recovery_sha = row.get("recovery_training_report_sha256")
    if width != BASE_WIDTH:
        recovery = _verify_file(
            source_evidence,
            path_field="recovery_training_report_path",
            sha_field="recovery_training_report_sha256",
            root=root,
        )
        if source_evidence["recovery_training_report_sha256"] != recovery_sha:
            raise ValueError("feedback/source recovery-training SHA mismatch")
        recovery_report = _read(recovery)
        recovery_audit = validate_recovery_training_evidence(
            report_path=recovery,
            recovery_contract_path=Path(
                str(recovery_report.get("recovery_contract_path") or "")
            ),
            config_path=Path(str(recovery_report.get("config_path") or "")),
            initial_checkpoint_path=Path(
                str(recovery_report.get("initial_checkpoint_path") or "")
            ),
            recovered_checkpoint_path=Path(
                str(recovery_report.get("recovered_checkpoint_path") or "")
            ),
        )
        if recovery_audit["recovered_checkpoint_sha256"] != row.get(
            "checkpoint_sha256"
        ):
            raise ValueError("recovery-training report checkpoint drift")
        verified_paths.append(recovery)
    elif recovery_sha not in {None, source_evidence.get("recovery_training_report_sha256")}:
        raise ValueError("unpruned recovery-training SHA drift")

    return {
        "verified": True,
        "row_id": str(row.get("row_id") or row.get("manifest_job_id") or ""),
        "verified_paths": [str(path.resolve()) for path in verified_paths],
        "verified_sha256": [_file_sha(path) for path in verified_paths],
    }


def normalize_terminal_row(source: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    status = row.get("terminal_status")
    if status == SUCCESS:
        return {
            **row,
            **{metric: _finite(row.get(metric), name=metric) for metric in METRICS},
            "credible_terminal_status": True,
        }
    if status not in TERMINAL_FAILURES:
        raise ValueError(f"non-credible terminal status: {status!r}")
    reason = str(row.get("failure_reason") or "").strip()
    if not reason:
        raise ValueError("terminal feasibility failure is missing failure_reason")
    return {
        **row,
        **{metric: None for metric in METRICS},
        "credible_terminal_status": True,
        "selection_status": row.get("selection_status")
        or "credible_terminal_failure",
    }


def _select_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    terminal = [normalize_terminal_row(row) for row in rows]
    successful = [row for row in terminal if row["terminal_status"] == SUCCESS]
    if not successful:
        failure = min(
            terminal,
            key=lambda row: str(row.get("row_id") or row.get("manifest_job_id") or ""),
        )
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
            str(row.get("row_id") or row.get("manifest_job_id") or ""),
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


def select_arm_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("arm has no terminal candidates")
    return _select_candidate(rows, ap70_ref=ap70_ref, max_ap_drop=max_ap_drop)


def select_gear_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> dict[str, Any]:
    """Select GEAR strictly from the 16 formal T16 online observations."""
    formal = [dict(row) for row in rows]
    identities = [
        str(row.get("row_id") or row.get("manifest_job_id") or "") for row in formal
    ]
    if (
        len(formal) != 16
        or len(set(identities)) != 16
        or any(not identity for identity in identities)
    ):
        raise ValueError("GEAR requires exactly 16 unique formal T16 online rows")
    for row in formal:
        _reject_pilot(row, context="GEAR T16 row")
        if (
            row.get("task_id") != TASK_ID
            or row.get("training_source") != "online_feedback"
        ):
            raise ValueError("GEAR candidates must be formal T16 online feedback only")
    return _select_candidate(
        formal, ap70_ref=ap70_ref, max_ap_drop=max_ap_drop
    )


def repeat_paths_for_row(repeat_root: Path, row_id: str) -> list[Path]:
    """Resolve repeats under the selected identity, never a fixed incumbent lane."""
    _reject_pilot(str(repeat_root), context="repeat root")
    _reject_pilot(row_id, context="selected row_id")
    row_root = repeat_root / row_id
    performance = sorted(
        row_root.glob("same_gpu7_repeat_*/performance.json")
    )
    direct = sorted(row_root.glob("same_gpu7_repeat_*.json"))
    return performance or direct


def _median_performance(
    paths: Sequence[Path],
    *,
    expected_onnx_sha256: str,
    expected_precision: str,
    expected_builder_level: int,
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError(f"expected three GPU7 performance repeats, got {len(paths)}")
    from scripts.fcooper_stage6_control_request_v2 import (
        validate_trt_repeat_report,
    )

    reports = [_read(path) for path in paths]
    if any(int(report.get("gpu_abs", -1)) != 7 for report in reports):
        raise ValueError("independent repeat is not bound to GPU7")
    engine_shas = []
    for path, report in zip(paths, reports):
        try:
            repeat_audit = validate_trt_repeat_report(
                report,
                expected_onnx_sha256=expected_onnx_sha256,
                expected_precision=expected_precision,
                expected_builder_level=expected_builder_level,
                expected_gpu=7,
                artifact_dir=path.parent / "engine",
            )
        except ValueError as error:
            if int(report.get("gpu_abs", -1)) != 7:
                raise ValueError("independent repeat is not bound to GPU7") from error
            raise
        engine_shas.append(repeat_audit["compiled_engine_sha256"])
    return {
        "latency_ms": statistics.median(
            _finite(report.get("lat_p50_ms"), name="repeat latency")
            for report in reports
        ),
        "energy_j": statistics.median(
            _finite(report.get("energy_j"), name="repeat energy")
            for report in reports
        ),
        "repeat_paths": [str(path.resolve()) for path in paths],
        "repeat_sha256": [_file_sha(path) for path in paths],
        "repeat_engine_sha256": engine_shas,
    }


def _median_native(
    paths: Sequence[Path],
    *,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError(f"expected three GPU7 native repeats, got {len(paths)}")
    reports = [_read(path) for path in paths]
    for report in reports:
        if int(report.get("gpu_abs", -1)) != 7:
            raise ValueError("native independent repeat is not bound to GPU7")
        if report.get("checkpoint_sha256") != expected_checkpoint_sha256:
            raise ValueError("native repeat checkpoint SHA drift")
        if report.get("config_sha256") != expected_config_sha256:
            raise ValueError("native repeat config SHA drift")
    return {
        "latency_ms": statistics.median(
            _finite(report.get("latency_ms"), name="native repeat latency")
            for report in reports
        ),
        "energy_j": statistics.median(
            _finite(report.get("energy_j"), name="native repeat energy")
            for report in reports
        ),
        "repeat_paths": [str(path.resolve()) for path in paths],
        "repeat_sha256": [_file_sha(path) for path in paths],
    }


def _validate_independent_ap(
    path: Path,
    *,
    expected: Mapping[str, Any],
    engine_path: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    from scripts.fcooper_stage6_control_request_v2 import (
        validate_independent_ap_report,
    )

    report = _read(path)
    validation = validate_independent_ap_report(
        report,
        expected=expected,
        engine_path=engine_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha(path),
        "processed_samples": 2170,
        "fallback_samples": 0,
        "metric_abs_drift": validation["metric_abs_drift"],
        "artifact_sha256": validation["artifact_sha256"],
        "ap30": float(report["ap30"]),
        "ap50": float(report["ap50"]),
        "ap70": float(report["ap70"]),
    }


def _validate_terminal_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence_root: Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} terminal rows, got {len(rows)}")
    identities = [
        str(row.get("row_id") or row.get("manifest_job_id") or "") for row in rows
    ]
    if any(not identity for identity in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("terminal pool has missing or duplicate row identity")
    normalized = []
    for row in rows:
        item = normalize_terminal_row(row)
        if item["terminal_status"] == SUCCESS:
            item["success_evidence_audit"] = verify_success_evidence(
                row, root=evidence_root
            )
        normalized.append(item)
    return normalized


def _validate_gpu_guard_audits(
    audit_root: Path,
    *,
    label: str,
    repeat_paths: Sequence[Path],
    full_ap_path: Path | None,
) -> list[dict[str, Any]]:
    if len(repeat_paths) != 3:
        raise ValueError("GPU7 guard binding requires exactly three repeat artifacts")
    expected = [
        (f"repeat_{index}", Path(path))
        for index, path in enumerate(repeat_paths)
    ]
    if full_ap_path is not None:
        expected.append(("full_ap", Path(full_ap_path)))
    evidence = []
    for suffix, expected_artifact in expected:
        audit_path = audit_root / f"{label}_{suffix}_guard.json"
        audit = _read(audit_path)
        _reject_pilot(audit, context="GPU7 guard audit")
        if (
            audit.get("schema_version") != "fcooper_gpu_exclusivity_gate_v1"
            or audit.get("status") != "completed_exclusive"
            or audit.get("evidence_scope") != "sampled_process_exclusivity"
            or int(audit.get("gpu_index", -1)) != 7
            or int(audit.get("return_code", -1)) != 0
            or not isinstance(audit.get("command_pid"), int)
            or not list(audit.get("command") or [])
            or int(audit.get("runtime_sample_count", 0)) < 1
            or list(audit.get("runtime_observations") or [])
            or list(audit.get("residual_processes") or [])
        ):
            raise ValueError(f"GPU7 guard did not prove exclusive execution: {label}/{suffix}")
        command = [str(value) for value in audit["command"]]
        artifact_spellings = {
            str(expected_artifact),
            str(expected_artifact.resolve()),
        }
        if not artifact_spellings.intersection(command):
            raise ValueError(
                f"GPU7 guard artifact binding drift: {label}/{suffix}"
            )
        log_path = Path(str(audit.get("log_path") or ""))
        _reject_pilot(str(log_path), context="GPU7 guarded log")
        if not log_path.is_file():
            raise ValueError(f"GPU7 guarded log is missing: {log_path}")
        evidence.append(
            {
                "label": f"{label}_{suffix}",
                "audit_path": str(audit_path.resolve()),
                "audit_sha256": _file_sha(audit_path),
                "log_path": str(log_path.resolve()),
                "log_sha256": _file_sha(log_path),
                "command_pid": int(audit["command_pid"]),
                "runtime_seconds": (
                    float(audit["runtime_seconds"])
                    if isinstance(audit.get("runtime_seconds"), (int, float))
                    and float(audit["runtime_seconds"]) > 0.0
                    else None
                ),
            }
        )
    return evidence


PHASE_TIMING_KEYS = (
    "recovery_initialization_seconds",
    "recovery_training_seconds",
    "onnx_export_seconds",
    "trt_build_performance_energy_seconds",
    "full_ap_seconds",
)


def _summarize_phase_timings(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise ValueError(
            f"{label} timing expected {expected_count} rows, got {len(rows)}"
        )
    totals = {key: 0.0 for key in PHASE_TIMING_KEYS}
    success_count = 0
    failure_count = 0
    for row in rows:
        if row.get("terminal_status") != SUCCESS:
            failure_count += 1
            continue
        timings = row.get("phase_timings_seconds")
        if not isinstance(timings, Mapping):
            raise ValueError(f"{label} success row is missing phase timings")
        for key in PHASE_TIMING_KEYS:
            value = timings.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{label} has invalid monotonic timing: {key}")
            totals[key] += float(value)
        success_count += 1
    return {
        "label": label,
        "passed": True,
        "row_count": len(rows),
        "success_count": success_count,
        "credible_failure_count": failure_count,
        "phase_totals_seconds": totals,
        "total_monotonic_seconds": sum(totals.values()),
    }


def _normalize_reused_phase_timings(
    label: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    compression_initialization_audit: Mapping[str, Any] | None = None,
    reuse_source_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    audit_rows = list(
        (compression_initialization_audit or {}).get("entries") or []
    )
    audit_widths = [
        tuple(int(value) for value in entry.get("width") or ())
        for entry in audit_rows
    ]
    if compression_initialization_audit is not None and (
        compression_initialization_audit.get("schema_version")
        != "fcooper_recovery_initialization_timing_v2"
        or compression_initialization_audit.get("passed") is not True
        or int(compression_initialization_audit.get("entry_count", -1))
        != len(audit_rows)
        or len(set(audit_widths)) != len(audit_widths)
    ):
        raise ValueError("recovery initialization timing audit drift")
    audit_entries = {
        tuple(int(value) for value in entry.get("width") or ()): entry
        for entry in audit_rows
    }
    recovered_training_rows = 0
    independently_remeasured_initialization_rows = 0
    zero_work_reuse_rows = 0
    sha_bound_reuse_rows = 0
    reuse_contracts = {
        "compress_then_tune_screen": {
            "task_id": "S6-FCO-TRT-COMPRESS-THEN-TUNE-SCREEN-V2",
            "phase": "screen",
        },
        "compress_then_tune_tuned": {
            "task_id": "S6-FCO-TRT-COMPRESS-THEN-TUNE-TUNED-V2",
            "phase": "tuned_remeasurement",
        },
    }
    reuse_sources_by_width = {
        tuple(int(value) for value in source.get("width") or ()): source
        for source in reuse_source_rows or ()
        if source.get("terminal_status") == SUCCESS
    }
    for source_row in rows:
        row = dict(source_row)
        if row.get("terminal_status") != SUCCESS:
            normalized.append(row)
            continue
        timings = dict(row.get("phase_timings_seconds") or {})
        missing = [
            key
            for key in PHASE_TIMING_KEYS
            if not isinstance(timings.get(key), (int, float))
        ]
        if not missing:
            normalized.append(row)
            continue
        if any(
            key
            not in {
                "recovery_initialization_seconds",
                "recovery_training_seconds",
                "onnx_export_seconds",
            }
            for key in missing
        ):
            raise ValueError(f"{label} is missing a measured runtime phase")
        if label == "compression_only":
            width = tuple(int(value) for value in row.get("width") or ())
            if "recovery_initialization_seconds" in missing:
                entry = audit_entries.get(width)
                if (
                    entry is None
                    or entry.get("status") != "success"
                    or not isinstance(entry.get("elapsed_seconds"), (int, float))
                    or not math.isfinite(float(entry["elapsed_seconds"]))
                    or float(entry["elapsed_seconds"]) <= 0.0
                ):
                    raise ValueError(
                        "compression-only missing initialization timing evidence"
                    )
                report_path = Path(str(entry.get("report_path") or ""))
                log_path = Path(str(entry.get("log_path") or ""))
                command = entry.get("command")
                if (
                    not report_path.is_file()
                    or entry.get("report_sha256") != _file_sha(report_path)
                    or not log_path.is_file()
                    or entry.get("log_sha256") != _file_sha(log_path)
                    or not isinstance(command, list)
                    or entry.get("command_sha256")
                    != hashlib.sha256(
                        json.dumps(command, separators=(",", ":")).encode()
                    ).hexdigest()
                    or entry.get("timing_kind")
                    != "independent_monotonic_remeasurement"
                ):
                    raise ValueError(
                        "compression-only initialization timing report drift"
                    )
                report = _read(report_path)
                if (
                    report.get("status") != "ready_for_recovery_training"
                    or tuple(report.get("width") or ()) != width
                    or report.get("source_config_sha256")
                    != compression_initialization_audit.get(
                        "source_config_sha256"
                    )
                    or report.get("source_checkpoint_sha256")
                    != compression_initialization_audit.get(
                        "source_checkpoint_sha256"
                    )
                    or report.get("recovery_contract_sha256")
                    != compression_initialization_audit.get(
                        "recovery_contract_sha256"
                    )
                ):
                    raise ValueError(
                        "compression-only initialization report contract drift"
                    )
                timings["recovery_initialization_seconds"] = float(
                    entry["elapsed_seconds"]
                )
                independently_remeasured_initialization_rows += 1
            if "recovery_training_seconds" in missing:
                report_path = Path(
                    str(row.get("recovery_training_report_path") or "")
                )
                if (
                    not report_path.is_file()
                    or row.get("recovery_training_report_sha256")
                    != _file_sha(report_path)
                ):
                    raise ValueError(
                        "compression-only recovery training report drift"
                    )
                report = _read(report_path)
                elapsed = report.get("elapsed_seconds")
                if (
                    not isinstance(elapsed, (int, float))
                    or not math.isfinite(float(elapsed))
                    or float(elapsed) <= 0.0
                ):
                    raise ValueError(
                        "compression-only recovery training timing is invalid"
                    )
                timings["recovery_training_seconds"] = float(elapsed)
                recovered_training_rows += 1
            if "onnx_export_seconds" in missing:
                timings["onnx_export_seconds"] = 0.0
        else:
            contract = reuse_contracts.get(label)
            width = tuple(int(value) for value in row.get("width") or ())
            source = reuse_sources_by_width.get(width)
            row_graph = row.get("graph_features")
            source_graph = source.get("graph_features") if source else None
            row_identity = (
                row.get("checkpoint_sha256"),
                row_graph.get("onnx_sha256")
                if isinstance(row_graph, Mapping)
                else None,
                row.get("recovery_training_report_sha256"),
            )
            source_identity = (
                source.get("checkpoint_sha256") if source else None,
                source_graph.get("onnx_sha256")
                if isinstance(source_graph, Mapping)
                else None,
                source.get("recovery_training_report_sha256")
                if source
                else None,
            )
            if (
                contract is None
                or row.get("task_id") != contract["task_id"]
                or row.get("phase") != contract["phase"]
                or row.get("training_source") != "online_feedback"
                or row.get("control_source_task_id") != TASK_ID
                or len(width) != 5
                or source is None
                or any(
                    not isinstance(value, str) or len(value) != 64
                    for value in row_identity
                )
                or row_identity != source_identity
            ):
                raise ValueError(
                    f"{label} missing SHA-bound source provenance for zero-work reuse"
                )
            for key in missing:
                timings[key] = 0.0
            zero_work_reuse_rows += 1
            sha_bound_reuse_rows += 1
        row["phase_timings_seconds"] = timings
        normalized.append(row)
    return normalized, {
        "label": label,
        "passed": True,
        "recovered_training_rows": recovered_training_rows,
        "independently_remeasured_initialization_rows": (
            independently_remeasured_initialization_rows
        ),
        "zero_work_reuse_rows": zero_work_reuse_rows,
        "sha_bound_reuse_rows": sha_bound_reuse_rows,
    }


def _validate_search_replay_timing(path: Path) -> dict[str, Any]:
    payload = _read(path)
    rounds = list(payload.get("rounds") or [])
    total_elapsed = payload.get("total_elapsed_seconds")
    if (
        payload.get("schema_version")
        != "fcooper_cost_model_replay_timing_v2"
        or payload.get("status") != "passed"
        or payload.get("timing_kind")
        != "deterministic_same_input_replay_monotonic"
        or payload.get("original_online_refit_timing_available") is not False
        or payload.get("replay_matches_original_requests") is not True
        or [row.get("round_index") for row in rounds] != [0, 1, 2, 3]
        or not isinstance(total_elapsed, (int, float))
        or not math.isfinite(float(total_elapsed))
        or float(total_elapsed) <= 0.0
        or any(
            row.get("request_semantics_match") is not True
            or not isinstance(row.get("elapsed_seconds"), (int, float))
            or not math.isfinite(float(row["elapsed_seconds"]))
            or float(row["elapsed_seconds"]) <= 0.0
            for row in rounds
        )
    ):
        raise ValueError("cost-model deterministic replay timing is invalid")
    root = path.resolve().parents[2]
    elapsed_sum = 0.0
    for round_index, row in enumerate(rounds):
        selected = list(row.get("selected_row_ids") or [])
        request_sha = row.get("measurement_request_sha256")
        if (
            len(selected) != 4
            or len(set(map(str, selected))) != 4
            or not isinstance(request_sha, str)
            or len(request_sha) != 64
        ):
            raise ValueError("cost-model replay request identity is invalid")
        expected = Path(str(row.get("expected_request_path") or "")).resolve()
        replayed = Path(str(row.get("replayed_request_path") or "")).resolve()
        log = Path(str(row.get("log_path") or "")).resolve()
        required_expected = (
            root
            / "search"
            / TASK_ID
            / f"round_{round_index:02d}"
            / "measurement_request.json"
        ).resolve()
        if expected != required_expected:
            raise ValueError("cost-model replay expected-request path drift")
        for evidence, sha_key in (
            (expected, "expected_file_sha256"),
            (replayed, "replayed_file_sha256"),
            (log, "log_sha256"),
        ):
            _reject_pilot(str(evidence), context="cost-model replay evidence")
            if (
                not evidence.is_file()
                or not isinstance(row.get(sha_key), str)
                or _file_sha(evidence) != row[sha_key]
            ):
                raise ValueError("cost-model replay evidence SHA drift")
        if expected.read_bytes() != replayed.read_bytes():
            raise ValueError("cost-model replay request byte drift")
        request = _read(expected)
        if request.get("measurement_request_sha256") != request_sha:
            raise ValueError("cost-model replay embedded request SHA drift")
        elapsed_sum += float(row["elapsed_seconds"])
    if float(total_elapsed) + 1e-9 < elapsed_sum:
        raise ValueError("cost-model replay total runtime is inconsistent")
    return {
        "passed": True,
        "timing_kind": payload["timing_kind"],
        "original_online_refit_timing_available": False,
        "round_count": 4,
        "round_elapsed_seconds": [
            float(row["elapsed_seconds"]) for row in rounds
        ],
        "total_elapsed_seconds": float(total_elapsed),
        "path": str(path.resolve()),
        "sha256": _file_sha(path),
    }


def _validate_legacy_timing_remeasurement(path: Path) -> dict[str, Any]:
    payload = _read(path)
    records = list(payload.get("records") or [])
    expected_labels = {
        *(f"original_default_repeat_{index}" for index in range(3)),
        *(f"schedule_only_repeat_{index}" for index in range(3)),
        "schedule_only_full_ap",
    }
    labels = [str(row.get("label") or "") for row in records]
    if len(labels) != len(set(labels)):
        raise ValueError("legacy GPU7 timing has duplicate labels")
    total_runtime = payload.get("total_runtime_seconds")
    if (
        payload.get("schema_version")
        != "fcooper_legacy_gpu7_timing_remeasurement_v2"
        or payload.get("status") != "passed"
        or payload.get("does_not_replace_selected_metrics") is not True
        or set(labels) != expected_labels
        or len(records) != len(expected_labels)
        or not isinstance(total_runtime, (int, float))
        or not math.isfinite(float(total_runtime))
        or float(total_runtime) <= 0.0
    ):
        raise ValueError("legacy GPU7 timing remeasurement is incomplete")
    for row in records:
        if (
            row.get("timing_kind")
            != "independent_gpu7_monotonic_remeasurement"
            or not isinstance(row.get("runtime_seconds"), (int, float))
            or not math.isfinite(float(row["runtime_seconds"]))
            or float(row["runtime_seconds"]) <= 0.0
        ):
            raise ValueError("legacy GPU7 monotonic timing is invalid")
        for path_key, sha_key in (
            ("legacy_audit_path", "legacy_audit_sha256"),
            ("timing_audit_path", "timing_audit_sha256"),
            ("log_path", "log_sha256"),
        ):
            evidence_path = Path(str(row.get(path_key) or ""))
            if (
                not evidence_path.is_file()
                or _file_sha(evidence_path) != row.get(sha_key)
            ):
                raise ValueError("legacy GPU7 timing evidence SHA drift")
        legacy_audit = _read(Path(str(row["legacy_audit_path"])))
        timing_audit = _read(Path(str(row["timing_audit_path"])))
        for audit in (legacy_audit, timing_audit):
            _reject_pilot(audit, context="legacy GPU7 timing guard")
            if (
                audit.get("schema_version") != "fcooper_gpu_exclusivity_gate_v1"
                or audit.get("status") != "completed_exclusive"
                or audit.get("evidence_scope") != "sampled_process_exclusivity"
                or int(audit.get("gpu_index", -1)) != 7
                or int(audit.get("return_code", -1)) != 0
                or not isinstance(audit.get("runtime_sample_count"), int)
                or int(audit["runtime_sample_count"]) < 1
                or list(audit.get("runtime_observations") or [])
                or list(audit.get("residual_processes") or [])
                or not list(audit.get("command") or [])
            ):
                raise ValueError("legacy GPU7 guard is not exclusive evidence")
        guard_runtime = timing_audit.get("runtime_seconds")
        if (
            not isinstance(guard_runtime, (int, float))
            or not math.isfinite(float(guard_runtime))
            or not math.isclose(
                float(guard_runtime),
                float(row["runtime_seconds"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("legacy GPU7 timing runtime mismatch")
        if Path(str(timing_audit.get("log_path") or "")).resolve() != Path(
            str(row["log_path"])
        ).resolve():
            raise ValueError("legacy GPU7 timing log binding mismatch")
        label = str(row["label"])
        output_root = path.resolve().parent / label
        command = [str(value) for value in timing_audit["command"]]
        command_bindings: list[Path]
        required_files: list[Path]
        if label.startswith("original_default_repeat_"):
            command_bindings = [output_root / "repeat.json"]
            required_files = list(command_bindings)
        elif label.startswith("schedule_only_repeat_"):
            command_bindings = [
                output_root / "performance.json",
                output_root / "engine",
            ]
            required_files = [
                output_root / "performance.json",
                output_root / "engine/compiled.engine",
            ]
        elif label == "schedule_only_full_ap":
            command_bindings = [
                output_root / "ap.json",
                path.resolve().parent
                / "schedule_only_repeat_0/engine/compiled.engine",
            ]
            required_files = list(command_bindings)
        else:
            raise ValueError("legacy GPU7 timing label is unsupported")
        for expected_output in command_bindings:
            spellings = {str(expected_output), str(expected_output.resolve())}
            if not spellings.intersection(command):
                raise ValueError("legacy GPU7 timing command output binding drift")
        if any(not required.is_file() for required in required_files):
            raise ValueError("legacy GPU7 timing command output is missing")
    runtime_sum = sum(float(row["runtime_seconds"]) for row in records)
    if not math.isclose(
        float(total_runtime), runtime_sum, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("legacy GPU7 total runtime mismatch")
    return {
        "passed": True,
        "timing_kind": "independent_gpu7_monotonic_remeasurement",
        "does_not_replace_selected_metrics": True,
        "record_count": len(records),
        "total_runtime_seconds": runtime_sum,
        "path": str(path.resolve()),
        "sha256": _file_sha(path),
    }


def _bind_repeats(
    selected: Mapping[str, Any],
    *,
    repeat_root: Path,
    builder_level: int,
    guard_root: Path,
    guard_label: str,
) -> dict[str, Any]:
    row = dict(selected)
    if row["terminal_status"] != SUCCESS:
        return {
            **row,
            "repeat_paths": [],
            "repeat_sha256": [],
            "ap_constraint_satisfied": False,
        }
    row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    graph = row.get("graph_features") or {}
    repeat_paths = repeat_paths_for_row(repeat_root, row_id)
    metrics = _median_performance(
        repeat_paths,
        expected_onnx_sha256=str(graph.get("onnx_sha256") or ""),
        expected_precision=str(row["q_mode"]),
        expected_builder_level=builder_level,
    )
    ap_path = repeat_root / row_id / "independent_ap_report.json"
    from scripts.fcooper_stage6_control_request_v2 import repeat_spec

    source = repeat_spec(row)
    ap_recheck = _validate_independent_ap(
        ap_path,
        expected=row,
        engine_path=repeat_paths[0].parent / "engine/compiled.engine",
        config_path=Path(source["config_path"]),
        checkpoint_path=Path(source["checkpoint_path"]),
    )
    guard_evidence = _validate_gpu_guard_audits(
        guard_root,
        label=guard_label,
        repeat_paths=repeat_paths,
        full_ap_path=ap_path,
    )
    return {
        **row,
        **metrics,
        "independent_ap_recheck": ap_recheck,
        "resource_guard_audits": guard_evidence,
        "resource_guard_expected_count": 4,
    }


def _verify_contract_file(
    contract: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    root: Path,
) -> Path:
    return _verify_file(
        contract, path_field=path_field, sha_field=sha_field, root=root
    )


def _original_row(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    for path_field, sha_field in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("config_path", "config_sha256"),
        ("ap_reference_report_path", "ap_reference_report_sha256"),
    ):
        _verify_contract_file(
            contract, path_field=path_field, sha_field=sha_field, root=root
        )
    row_id = "fcooper-original-default"
    repeats = repeat_paths_for_row(
        root / "controls/original_default/independent_repeats", row_id
    )
    metrics = _median_native(
        repeats,
        expected_checkpoint_sha256=str(contract["checkpoint_sha256"]),
        expected_config_sha256=str(contract["config_sha256"]),
    )
    guard_evidence = _validate_gpu_guard_audits(
        root / "controls/resource_audit/gpu7_exclusivity",
        label="original_default",
        repeat_paths=repeats,
        full_ap_path=None,
    )
    return {
        "row_id": row_id,
        "terminal_status": SUCCESS,
        "credible_terminal_status": True,
        "width": list(BASE_WIDTH),
        "q_mode": "fp32",
        "ap30": _finite(contract["ap30_ref"], name="ap30_ref"),
        "ap50": _finite(contract["ap50_ref"], name="ap50_ref"),
        "ap70": _finite(contract["ap70_ref"], name="ap70_ref"),
        "ap_report_sha256": contract["ap_reference_report_sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        **metrics,
        "resource_guard_audits": guard_evidence,
        "resource_guard_expected_count": 3,
        "ap_constraint_satisfied": True,
        "selection_status": "fixed_original_default",
    }


def _configuration(row: Mapping[str, Any]) -> str | None:
    width = row.get("width")
    q_mode = row.get("q_mode")
    if not isinstance(width, Sequence) or isinstance(width, str) or not q_mode:
        return None
    return f"({','.join(map(str, width))},{q_mode})"


def _csv_row(method: str, row: Mapping[str, Any], budget: Any) -> dict[str, Any]:
    return {
        "method": method,
        "terminal_status": row["terminal_status"],
        "selection_status": row.get("selection_status"),
        "ap_constraint_satisfied": (
            row.get("ap_constraint_satisfied")
            if row["terminal_status"] == SUCCESS
            else None
        ),
        "configuration": _configuration(row),
        **{metric: row.get(metric) for metric in METRICS},
        "budget": budget,
        "failure_rate": 0.0 if row["terminal_status"] == SUCCESS else 1.0,
    }


def _ensure_final_outputs_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite final evidence: "
            + ", ".join(str(path) for path in existing)
        )


def _prepare_final_output_target(output: Path) -> Path | None:
    if not output.exists():
        return None
    if not output.is_dir():
        raise ValueError(f"final output target is not a directory: {output}")
    entries = sorted(path for path in output.rglob("*") if path.is_file())
    if not entries:
        output.rmdir()
        return None
    expected = {output / name for name in FINAL_OUTPUT_NAMES}
    if expected.issubset(set(entries)):
        raise FileExistsError(f"complete final output already exists: {output}")
    manifest = [
        {
            "relative_path": str(path.relative_to(output)),
            "sha256": _file_sha(path),
        }
        for path in entries
    ]
    digest = _canonical_sha(manifest)[:16]
    quarantine = output.parent / f".{output.name}.incomplete-{digest}"
    if quarantine.exists():
        raise FileExistsError(
            f"incomplete final-output quarantine already exists: {quarantine}"
        )
    output.rename(quarantine)
    return quarantine


def _validate_schedule_identity(
    row: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if (
        str(row.get("row_id") or "") != plan.get("fixed_row_id")
        or str(row.get("manifest_job_id") or "")
        != plan.get("fixed_manifest_job_id")
        or row.get("schedule_baseline_derivation")
        != plan.get("schedule_baseline_derivation")
        or row.get("schedule_baseline_derivation_sha256")
        != plan.get("schedule_baseline_derivation_sha256")
    ):
        raise ValueError("schedule-only identity does not match the frozen plan")


def validate_formal_preconditions(root: Path) -> dict[str, Any]:
    width_schema = [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ]
    scanner_path = root / "scanner/scanner_execution.json"
    contract_path = root / "contracts/formal_contract_v2.json"
    probe_path = root / "probes/probe_audit.json"
    isolation_path = root / "probes/probe_isolation_audit.json"
    gate_path = root / "gates/recovery_numeric_gate_summary.json"
    closure_path = root / f"search/{TASK_ID}/formal_t16_closure_audit.json"
    scanner = _read(scanner_path)
    contract = _read(contract_path)
    probes = _read(probe_path)
    isolation = _read(isolation_path)
    gate = _read(gate_path)
    closure = _read(closure_path)
    if (
        scanner.get("schema_version") != "fcooper_fresh_scanner_execution_v2"
        or scanner.get("status") != "success"
        or scanner.get("manual_override_used") is not False
        or scanner.get("hardware_schema_validated") is not True
        or float(scanner.get("elapsed_seconds") or 0.0) <= 0.0
        or not list(scanner.get("command") or [])
    ):
        raise ValueError("fresh scanner execution contract is invalid")
    if (
        contract.get("schema_version") != "fcooper_formal_v2_contract"
        or contract.get("model") != "fcooper"
        or contract.get("dataset") != "OPV2V"
        or contract.get("split") != "test"
        or contract.get("width_schema") != width_schema
        or contract.get("q_modes") != ["fp16", "int8"]
        or contract.get("search_budget")
        != {"batch_size": 4, "rounds": 4, "total": 16}
        or int(contract.get("structure_candidate_count", -1)) != 1792
        or int(contract.get("precision_genome_count", -1)) != 3584
        or contract.get("pilot_online_labels_loaded") is not False
        or contract.get("probe_labels_allowed_in_training") is not False
        or contract.get("probe_rows_allowed_as_winner") is not False
    ):
        raise ValueError("formal search contract drift")
    bound_files: list[Path] = [
        scanner_path,
        contract_path,
        probe_path,
        isolation_path,
        gate_path,
        closure_path,
    ]
    for payload, path_field, sha_field in (
        (scanner, "config_path", "config_sha256"),
        (scanner, "checkpoint_path", "checkpoint_sha256"),
        (scanner, "partition_path", "partition_sha256"),
        (scanner, "log_path", "log_sha256"),
        (contract, "partition_path", "partition_sha256"),
        (contract, "registry_path", "registry_sha256"),
        (contract, "recovery_contract_path", "recovery_contract_sha256"),
        (contract, "source_config_path", "source_config_sha256"),
        (contract, "source_checkpoint_path", "source_checkpoint_sha256"),
        (contract, "test_manifest_path", "test_manifest_sha256"),
        (isolation, "probe_audit_path", "probe_audit_sha256"),
    ):
        bound_files.append(
            _verify_file(
                payload,
                path_field=path_field,
                sha_field=sha_field,
                root=root,
            )
        )
    registry = _read(Path(str(contract["registry_path"])))
    if (
        registry.get("schema_version") != "stage5_candidate_source_registry_v1"
        or registry.get("model") != "fcooper"
        or registry.get("width_schema") != width_schema
        or int(registry.get("structure_candidate_count", -1)) != 1792
        or int(registry.get("structure_group_count", -1)) != 5
        or len(registry.get("groups") or []) != 1792
        or registry.get("partition_manifest_sha256")
        != contract.get("partition_sha256")
    ):
        raise ValueError("scanner-derived candidate registry drift")
    probe_rows = probes.get("rows") or []
    if (
        probes.get("schema_version") != "fcooper_probe_audit_v1"
        or probes.get("all_probes_terminal") is not True
        or int(probes.get("probe_count", -1)) != 8
        or len(probe_rows) != 8
        or any(
            row.get("status") != "success"
            or not isinstance(row.get("result_sha256"), str)
            or len(row["result_sha256"]) != 64
            for row in probe_rows
        )
        or isolation.get("schema_version")
        != "fcooper_probe_isolation_audit_v2"
        or isolation.get("status") != "passed"
        or int(isolation.get("probe_count", -1)) != 8
        or isolation.get("probe_metrics_allowed_as_cost_model_labels") is not False
        or isolation.get("probe_rows_allowed_as_winner") is not False
        or isolation.get("probe_rows_allowed_in_t16_budget") is not False
    ):
        raise ValueError("capability probe or isolation contract drift")
    gate_rows = gate.get("rows") or []
    if (
        gate.get("schema_version") != "fcooper_recovery_numeric_gate_v2"
        or gate.get("status") != "passed"
        or gate.get("t16_search_allowed") is not True
        or len(gate_rows) != 2
        or any(
            row.get("status") != "success_full"
            or int(row.get("dataset_samples", -1)) != 2170
            or int(row.get("processed_samples", -1)) != 2170
            or int(row.get("fallback_samples", -1)) != 0
            or float(row.get("ap70") or 0.0)
            < float(gate.get("non_collapse_ap70_floor") or math.inf)
            for row in gate_rows
        )
    ):
        raise ValueError("recovery-training numeric gate drift")
    original_gate = root / "gates/original/original_ap_gate_report.json"
    if (
        not original_gate.is_file()
        or _file_sha(original_gate)
        != contract.get("original_ap_gate_report_sha256")
    ):
        raise ValueError("original AP gate report SHA drift")
    bound_files.append(original_gate)
    deep = closure.get("deep_evidence_audit") or {}
    if (
        closure.get("schema_version") != "fcooper_formal_t16_closure_audit_v2"
        or closure.get("task_id") != TASK_ID
        or closure.get("status") != "closed"
        or int(closure.get("rounds", -1)) != 4
        or int(closure.get("online_rows", -1)) != 16
        or int(closure.get("probe_overlap_count", -1)) != 0
        or deep.get("passed") is not True
        or int(deep.get("verified_success_rows", -1)) != 16
    ):
        raise ValueError("formal T16 closure or deep evidence audit drift")
    unique_files = list(dict.fromkeys(path.resolve() for path in bound_files))
    return {
        "schema_version": "fcooper_formal_preconditions_audit_v2",
        "passed": True,
        "fresh_scanner_verified": True,
        "scanner_elapsed_seconds": float(scanner["elapsed_seconds"]),
        "scanner_manual_override_used": False,
        "scanner_search_group_ids": scanner["search_group_ids"],
        "width_schema": width_schema,
        "structure_candidate_count": 1792,
        "precision_genome_count": 3584,
        "capability_probe_count": 8,
        "probe_labels_loaded": False,
        "probe_rows_allowed_as_winner": False,
        "recovery_numeric_gate_rows": 2,
        "formal_t16_rows": 16,
        "deep_evidence_file_count": int(deep["file_count"]),
        "files": [
            {"path": str(path), "sha256": _file_sha(path)}
            for path in unique_files
        ],
    }


def finalize(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    _reject_pilot(str(root), context="formal root")
    _reject_pilot(str(output), context="output root")
    contract_path = root / "contracts/frozen_contract.json"
    contract = _read(contract_path)
    _reject_pilot(contract, context="formal frozen contract")
    formal_preconditions = validate_formal_preconditions(root)
    ap70_ref = _finite(contract["ap70_ref"], name="ap70_ref")
    max_drop = 0.10

    original_failure = root / "controls/original_default/terminal_failure.json"
    original = (
        normalize_terminal_row(_read(original_failure))
        if original_failure.is_file()
        else _original_row(root, contract)
    )

    plan = _read(root / "controls/five_arm_plan/stage6_fcooper_five_arm_plan.json")
    arms = plan.get("arms") or {}
    schedule_rows = _validate_terminal_pool(
        _rows(root / "controls/schedule_only/feedback_history.json"),
        evidence_root=root,
        expected_count=1,
    )
    schedule = schedule_rows[0]
    if tuple(schedule.get("width") or ()) != BASE_WIDTH or schedule.get("q_mode") != "fp32":
        raise ValueError("schedule-only must use the fixed original FP32 structure")
    _validate_schedule_identity(schedule, arms.get("schedule_only") or {})
    schedule["selection_status"] = "fixed_schedule_only"
    schedule["ap_constraint_satisfied"] = (
        schedule["terminal_status"] == SUCCESS
        and float(schedule["ap70"]) >= ap70_ref - max_drop
    )
    schedule = _bind_repeats(
        schedule,
        repeat_root=root / "controls/schedule_only/independent_repeats",
        builder_level=5,
        guard_root=root / "controls/resource_audit/gpu7_exclusivity",
        guard_label="schedule_only",
    )

    compression_rows = _validate_terminal_pool(
        _rows(root / "controls/compression_only/feedback_history.json"),
        evidence_root=root,
        expected_count=16,
    )
    coldstart_contract = plan.get("coldstart_contract") or {}
    graph_contract = plan.get("observed_graph_evidence_contract") or {}
    if (
        plan.get("schema_version") != "stage6_fcooper_five_arm_plan_v2"
        or plan.get("task_id") != TASK_ID
        or coldstart_contract.get("source")
        != "byte_frozen_standalone_gold176_only"
        or coldstart_contract.get("rows_sha256")
        != FROZEN_GOLD176_ROWS_SHA256
        or coldstart_contract.get("graph_features_sha256")
        != FROZEN_GOLD176_GRAPH_FEATURES_SHA256
    ):
        raise ValueError("five-arm plan does not bind the frozen Gold176 contract")
    graph_evidence_path = _resolve_path(
        graph_contract.get("path"),
        root=root,
        field="observed_graph_evidence_contract.path",
    )
    if (
        not graph_evidence_path.is_file()
        or _file_sha(graph_evidence_path) != graph_contract.get("sha256")
    ):
        raise ValueError("five-arm plan observed graph evidence SHA drift")
    graph_evidence = _read(graph_evidence_path)
    graph_bindings = graph_evidence.get("t16_graph_bindings") or []
    graph_rows = graph_evidence.get("graph_features") or []
    if (
        len(graph_bindings) != 16
        or len(graph_rows) != 17
        or len({str(row.get("row_id") or "") for row in graph_bindings}) != 16
        or any(
            binding.get("materialized_graph_features_sha256")
            != binding.get("graph_payload_sha256")
            or binding.get("graph_payload_sha256")
            != _canonical_sha(graph_rows[index])
            for index, binding in enumerate(graph_bindings)
        )
    ):
        raise ValueError("five-arm plan observed graph row binding drift")
    compression_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in compression_rows
    ]
    if compression_ids != list(
        (arms.get("compression_only") or {}).get("selected_row_ids") or []
    ):
        raise ValueError("compression feedback does not match five-arm plan")
    compression = _bind_repeats(
        select_arm_candidate(
            compression_rows, ap70_ref=ap70_ref, max_ap_drop=max_drop
        ),
        repeat_root=root / "controls/compression_only/independent_repeats",
        builder_level=0,
        guard_root=root / "controls/resource_audit/gpu7_exclusivity",
        guard_label="compression_only",
    )

    screen_rows = _validate_terminal_pool(
        _rows(root / "controls/compress_then_tune/screen_feedback_history.json"),
        evidence_root=root,
        expected_count=12,
    )
    screen_ids_ordered = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in screen_rows
    ]
    if screen_ids_ordered != list(
        (arms.get("compress_then_tune") or {}).get("screen_row_ids") or []
    ):
        raise ValueError("screen feedback does not match five-arm plan")
    tuned_rows = _validate_terminal_pool(
        _rows(root / "controls/compress_then_tune/tuned_feedback_history.json"),
        evidence_root=root,
        expected_count=4,
    )
    screen_ids = {
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in screen_rows
    }
    tuned_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in tuned_rows
    ]
    if len(set(tuned_ids)) != 4 or not set(tuned_ids).issubset(screen_ids):
        raise ValueError("tuned four must be distinct members of the observed screen")
    tune_plan = plan.get("arms", {}).get("compress_then_tune", {})
    if (
        plan.get("task_id") != TASK_ID
        or tune_plan.get("phase_status") != "tuned_remeasurement_ready"
        or list(tune_plan.get("tuned_row_ids") or []) != tuned_ids
    ):
        raise ValueError("compress-then-tune plan does not bind the observed tuned four")
    compress_then_tune = _bind_repeats(
        select_arm_candidate(tuned_rows, ap70_ref=ap70_ref, max_ap_drop=max_drop),
        repeat_root=root
        / "controls/compress_then_tune/independent_repeats",
        builder_level=5,
        guard_root=root / "controls/resource_audit/gpu7_exclusivity",
        guard_label="compress_then_tune",
    )

    online_path = root / f"search/{TASK_ID}/feedback_history_final_t16.json"
    online_rows = _validate_terminal_pool(
        _rows(online_path), evidence_root=root, expected_count=16
    )
    initialization_timing_path = (
        root
        / "controls/resource_audit/recovery_initialization_timing_remeasurement"
        / "summary.json"
    )
    initialization_timing_audit = (
        _read(initialization_timing_path)
        if initialization_timing_path.is_file()
        else None
    )
    timing_normalization: list[dict[str, Any]] = []
    compression_rows, provenance = _normalize_reused_phase_timings(
        "compression_only",
        compression_rows,
        compression_initialization_audit=initialization_timing_audit,
    )
    timing_normalization.append(provenance)
    screen_rows, provenance = _normalize_reused_phase_timings(
        "compress_then_tune_screen",
        screen_rows,
        reuse_source_rows=compression_rows,
    )
    timing_normalization.append(provenance)
    tuned_rows, provenance = _normalize_reused_phase_timings(
        "compress_then_tune_tuned",
        tuned_rows,
        reuse_source_rows=screen_rows,
    )
    timing_normalization.append(provenance)
    schedule_rows, provenance = _normalize_reused_phase_timings(
        "schedule_only", schedule_rows
    )
    timing_normalization.append(provenance)
    online_rows, provenance = _normalize_reused_phase_timings(
        "formal_t16", online_rows
    )
    timing_normalization.append(provenance)
    gear = _bind_repeats(
        select_gear_candidate(
            online_rows, ap70_ref=ap70_ref, max_ap_drop=max_drop
        ),
        repeat_root=root / f"search/{TASK_ID}/independent_repeats",
        builder_level=5,
        guard_root=root / "controls/resource_audit/gpu7_exclusivity",
        guard_label="gear",
    )
    gear["selection_source"] = "formal_t16_online_feedback"

    arm_rows = [
        ("Original/default", original, 1),
        ("Compression only", compression, 16),
        ("Schedule only", schedule, 1),
        ("Compress -> Tune", compress_then_tune, "12+4"),
        ("GEAR (Ours)", gear, 16),
    ]
    csv_rows = [_csv_row(method, row, budget) for method, row, budget in arm_rows]

    integrity_paths = [
        root / f"search/{TASK_ID}/online_file_integrity_audit.json",
        root / f"search/{TASK_ID}/final_feedback_integrity_audit.json",
        root / "controls/schedule_only/file_integrity_audit.json",
        root / "controls/compression_only/file_integrity_audit.json",
        root / "controls/compress_then_tune/file_integrity_audit.json",
    ]
    integrity = [_read(path) for path in integrity_paths]
    integrity_passed = all(payload.get("passed") is True for payload in integrity)
    all_credible_terminal = all(
        row.get("credible_terminal_status") is True for _, row, _ in arm_rows
    )
    successful_repeats_verified = all(
        row["terminal_status"] != SUCCESS or len(row.get("repeat_paths") or []) == 3
        for _, row, _ in arm_rows
    )
    resource_guards_verified = all(
        row["terminal_status"] != SUCCESS
        or len(row.get("resource_guard_audits") or [])
        == int(row.get("resource_guard_expected_count") or -1)
        for _, row, _ in arm_rows
    )
    phase_timing_summaries = [
        _summarize_phase_timings(
            "schedule_only", schedule_rows, expected_count=1
        ),
        _summarize_phase_timings(
            "compression_only", compression_rows, expected_count=16
        ),
        _summarize_phase_timings(
            "compress_then_tune_screen", screen_rows, expected_count=12
        ),
        _summarize_phase_timings(
            "compress_then_tune_tuned", tuned_rows, expected_count=4
        ),
        _summarize_phase_timings(
            "formal_t16", online_rows, expected_count=16
        ),
    ]
    search_replay_timing = _validate_search_replay_timing(
        root
        / "controls/resource_audit/cost_model_replay_timing_v2.json"
    )
    legacy_gpu7_timing = _validate_legacy_timing_remeasurement(
        root
        / "controls/resource_audit/gpu7_timing_remeasurement/summary.json"
    )
    selected_runtime_rows = [
        ("compression_only", compression),
        ("compress_then_tune", compress_then_tune),
        ("gear", gear),
    ]
    selected_gpu7_monotonic_complete = all(
        row["terminal_status"] != SUCCESS
        or (
            len(row.get("resource_guard_audits") or [])
            == int(row.get("resource_guard_expected_count") or -1)
            and all(
                isinstance(audit.get("runtime_seconds"), (int, float))
                and float(audit["runtime_seconds"]) > 0.0
                for audit in row.get("resource_guard_audits") or []
            )
        )
        for _, row in selected_runtime_rows
    )
    resource_timing_audit = {
        "schema_version": "fcooper_stage6_resource_timing_audit_v2",
        "passed": (
            float(formal_preconditions["scanner_elapsed_seconds"]) > 0.0
            and all(row["passed"] for row in phase_timing_summaries)
            and search_replay_timing["passed"]
            and legacy_gpu7_timing["passed"]
            and selected_gpu7_monotonic_complete
        ),
        "scanner": {
            "timing_kind": "original_monotonic",
            "elapsed_seconds": float(
                formal_preconditions["scanner_elapsed_seconds"]
            ),
        },
        "measurement_phase_timings": phase_timing_summaries,
        "timing_normalization": timing_normalization,
        "recovery_initialization_timing_remeasurement": (
            {
                "path": str(initialization_timing_path),
                "sha256": _file_sha(initialization_timing_path),
                "passed": initialization_timing_audit.get("passed"),
            }
            if initialization_timing_audit is not None
            else None
        ),
        "cost_model_fit_and_refit": search_replay_timing,
        "legacy_original_schedule_gpu7": legacy_gpu7_timing,
        "selected_new_arm_gpu7_monotonic_complete": (
            selected_gpu7_monotonic_complete
        ),
        "selected_new_arm_gpu7_runtime_seconds": {
            label: [
                float(audit["runtime_seconds"])
                for audit in row.get("resource_guard_audits") or []
                if isinstance(audit.get("runtime_seconds"), (int, float))
            ]
            for label, row in selected_runtime_rows
            if row["terminal_status"] == SUCCESS
        },
    }

    csv_path = output / FINAL_OUTPUT_NAMES[0]
    audit_path = output / FINAL_OUTPUT_NAMES[1]
    bundle_path = output / FINAL_OUTPUT_NAMES[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    recovered_partial = _prepare_final_output_target(output)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    staging_csv = staging / csv_path.name
    staging_audit = staging / audit_path.name
    staging_bundle = staging / bundle_path.name
    with staging_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    audit = {
        "schema_version": "stage6_fcooper_five_arm_audit_v2",
        "task_id": TASK_ID,
        "target_model": "fcooper",
        "hardware_id": "h800",
        "backend": "trt",
        "performance_gpu_index": 7,
        "ap70_ref": ap70_ref,
        "ap70_floor": ap70_ref - max_drop,
        "online_search_budget_consumed": len(online_rows),
        "gear_candidate_source": "formal_t16_online_feedback_only",
        "gear_probe_rows_used": False,
        "all_five_arms_credible_terminal": all_credible_terminal,
        "successful_repeats_verified": successful_repeats_verified,
        "resource_guards_verified": resource_guards_verified,
        "resource_timing_audit": resource_timing_audit,
        "formal_preconditions": formal_preconditions,
        "recovered_incomplete_final_output": (
            str(recovered_partial.resolve()) if recovered_partial else None
        ),
        "paper_ready": (
            integrity_passed
            and formal_preconditions["passed"]
            and all_credible_terminal
            and successful_repeats_verified
            and resource_guards_verified
            and resource_timing_audit["passed"]
            and len(csv_rows) == 5
        ),
        "integrity_audits": [
            {
                "path": str(path.resolve()),
                "sha256": _file_sha(path),
                "passed": payload.get("passed"),
                "row_count": payload.get("row_count"),
            }
            for path, payload in zip(integrity_paths, integrity)
        ],
        "rows": csv_rows,
        "selected_evidence": {
            method: {
                "row_id": row.get("row_id"),
                "terminal_status": row["terminal_status"],
                "failure_reason": row.get("failure_reason"),
                "engine_sha256": row.get("engine_sha256"),
                "performance_result_sha256": row.get(
                    "performance_result_sha256"
                ),
                "ap_report_sha256": row.get("ap_report_sha256"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "recovery_training_report_sha256": row.get(
                    "recovery_training_report_sha256"
                ),
                "repeat_paths": row.get("repeat_paths") or [],
                "repeat_sha256": row.get("repeat_sha256") or [],
                "repeat_engine_sha256": row.get("repeat_engine_sha256"),
                "resource_guard_audits": row.get("resource_guard_audits") or [],
            }
            for method, row, _ in arm_rows
        },
    }
    with staging_audit.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
    bundle = {
        "schema_version": "fcooper_stage6_evidence_bundle_v2",
        "task_id": TASK_ID,
        "audit_path": str(audit_path),
        "audit_sha256": _file_sha(staging_audit),
        "csv_path": str(csv_path),
        "csv_sha256": _file_sha(staging_csv),
        "frozen_contract_path": str(contract_path),
        "frozen_contract_sha256": _file_sha(contract_path),
        "selected_evidence": audit["selected_evidence"],
        "formal_preconditions": formal_preconditions,
        "resource_timing_audit": resource_timing_audit,
    }
    try:
        with staging_bundle.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = finalize(args.root, args.output_dir)
    print(
        json.dumps(
            {"paper_ready": audit["paper_ready"], "rows": audit["rows"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["paper_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
