#!/usr/bin/env python3
"""Normalize fresh Stage5 performance/AP reruns into a strict validation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


TASKS = ("S5-PYR-TVM", "S5-PYR-TRT", "S5-COD-TVM", "S5-COD-TRT")
LATENCY_REL_TOL = 0.15
ENERGY_REL_TOL = 0.20
AP70_ABS_TOL = 0.01
LATENCY_CV_MAX = 0.10
ENERGY_CV_MAX = 0.15


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _nested(payload: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> float:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, bool) and math.isfinite(result):
            return result
    raise ValueError(f"finite metric missing from evidence paths: {paths}")


def _verified_payload(path: Path, expected_sha: str, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or _sha(path) != expected_sha:
        raise ValueError(f"{label} SHA mismatch")
    payload = _read(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def normalize_performance_repeat(
    *,
    task_id: str,
    configuration_id: str,
    repeat_index: int,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(str(state.get("result_json") or ""))
    expected_sha = str(state.get("result_sha256") or "")
    payload = _verified_payload(path, expected_sha, label="independent performance")
    if state.get("status") != "success" or payload.get("status") not in {
        "success",
        None,
    }:
        raise ValueError("independent performance is not successful")
    started = float(state.get("start_time_unix"))
    ended = float(state.get("end_time_unix"))
    if not (math.isfinite(started) and math.isfinite(ended) and ended > started):
        raise ValueError("independent performance timing is invalid")
    repeat_id = f"repeat-{repeat_index}"
    run_uuid = hashlib.sha256(
        f"{task_id}|{configuration_id}|{repeat_index}|{expected_sha}|{started}|{ended}".encode()
    ).hexdigest()
    return {
        "schema_version": "stage5_independent_performance_repeat_v1",
        "status": "success",
        "task_id": task_id,
        "configuration_id": configuration_id,
        "hardware_id": "h800",
        "repeat_id": repeat_id,
        "run_uuid": run_uuid,
        "started_at_utc": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
        "ended_at_utc": datetime.fromtimestamp(ended, tz=timezone.utc).isoformat(),
        "latency_ms": _nested(
            payload,
            (
                ("lat_p50_ms",),
                ("latency_ms",),
                ("latency", "latency_ms_p50"),
                ("latency", "lat_p50_ms"),
            ),
        ),
        "energy_j": _nested(
            payload,
            (
                ("energy_j",),
                ("energy", "energy_j"),
                ("energy", "joules"),
                ("energy", "joules_per_inference"),
                ("energy", "joule_per_inference"),
                ("energy", "energy_J"),
            ),
        ),
        "raw_performance_result_json": str(path),
        "raw_performance_result_sha256": expected_sha,
    }


def normalize_ap_report(
    *, task_id: str, configuration_id: str, terminal: Mapping[str, Any]
) -> dict[str, Any]:
    path = Path(str(terminal.get("report_path") or ""))
    expected_sha = str(terminal.get("report_sha256") or "")
    report = _verified_payload(path, expected_sha, label="independent full AP")
    if (
        terminal.get("stage") != "full"
        or terminal.get("status") != "success"
        or report.get("status") != "success"
        or int(report.get("processed_samples") or 0) != 1789
        or (
            report.get("requested_num_samples") is not None
            and int(report["requested_num_samples"]) != 1789
        )
        or int(report.get("failed_samples") or 0) != 0
        or int(report.get("fallback_samples") or 0) != 0
    ):
        raise ValueError("independent full AP contract mismatch")
    source = report.get("ap") if isinstance(report.get("ap"), Mapping) else report
    return {
        "schema_version": "stage5_independent_full_ap_v1",
        "status": "success",
        "task_id": task_id,
        "configuration_id": configuration_id,
        "processed_samples": 1789,
        "requested_num_samples": 1789,
        "failed_samples": 0,
        "fallback_samples": 0,
        "ap30": _nested(source, (("ap30",),)),
        "ap50": _nested(source, (("ap50",),)),
        "ap70": _nested(source, (("ap70",),)),
        "raw_ap_report_path": str(path),
        "raw_ap_report_sha256": expected_sha,
    }


def successful_state_for_configuration(
    configuration_id: str,
    *,
    jobs: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    job_ids = [
        str(job.get("job_id") or "")
        for job in jobs
        if str(job.get("manifest_job_id") or "") == configuration_id
    ]
    if len(job_ids) != 1 or not job_ids[0]:
        raise ValueError(
            f"expected one executor job for independent configuration: {configuration_id}"
        )
    matches = [
        state
        for state in states
        if str(state.get("job_id") or "") == job_ids[0]
        and state.get("status") == "success"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one successful independent repeat: {configuration_id}"
        )
    return matches[0]


def unique_successful_ap_terminal(
    configuration_id: str, states: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Accept idempotent duplicate records but reject conflicting AP evidence."""
    matches = [
        state
        for state in states
        if str(state.get("job_id") or "") == configuration_id
        and state.get("record_type") == "job_terminal"
        and state.get("stage") == "full"
        and state.get("status") == "success"
    ]
    unique = {
        (
            str(state.get("report_path") or ""),
            str(state.get("report_sha256") or ""),
        ): state
        for state in matches
    }
    if len(unique) != 1:
        raise ValueError(
            f"expected one unique successful independent full AP: {configuration_id}"
        )
    return next(iter(unique.values()))


def _relative_delta(measured: float, reference: float) -> float:
    return abs(measured - reference) / max(abs(reference), 1e-12)


def _cv(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / max(abs(mean), 1e-12)


def evaluate_consistency(
    reference: Mapping[str, Any],
    repeats: Sequence[Mapping[str, Any]],
    ap_report: Mapping[str, Any],
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    if len(repeats) != 3:
        raise ValueError("independent consistency requires exactly three repeats")
    latency_values = [float(item["latency_ms"]) for item in repeats]
    energy_values = [float(item["energy_j"]) for item in repeats]
    latency_median = float(statistics.median(latency_values))
    energy_median = float(statistics.median(energy_values))
    ap70 = float(ap_report["ap70"])
    latency_delta = _relative_delta(latency_median, float(reference["latency_ms"]))
    energy_delta = _relative_delta(energy_median, float(reference["energy_j"]))
    ap70_delta = abs(ap70 - float(reference["ap70"]))
    latency_cv = _cv(latency_values)
    energy_cv = _cv(energy_values)
    passed = (
        latency_delta <= LATENCY_REL_TOL
        and energy_delta <= ENERGY_REL_TOL
        and ap70_delta <= AP70_ABS_TOL
        and latency_cv <= LATENCY_CV_MAX
        and energy_cv <= ENERGY_CV_MAX
    )
    result = {
        "passed": passed,
        "thresholds": {
            "latency_relative": LATENCY_REL_TOL,
            "energy_relative": ENERGY_REL_TOL,
            "ap70_absolute": AP70_ABS_TOL,
            "latency_cv": LATENCY_CV_MAX,
            "energy_cv": ENERGY_CV_MAX,
        },
        "reference": {
            "latency_ms": float(reference["latency_ms"]),
            "energy_j": float(reference["energy_j"]),
            "ap70": float(reference["ap70"]),
        },
        "rerun": {
            "latency_median_ms": latency_median,
            "energy_median_j": energy_median,
            "ap70": ap70,
            "latency_cv": latency_cv,
            "energy_cv": energy_cv,
        },
        "deltas": {
            "latency_relative": latency_delta,
            "energy_relative": energy_delta,
            "ap70_absolute": ap70_delta,
        },
    }
    if not passed and raise_on_failure:
        raise ValueError(f"independent validation drift exceeds thresholds: {result}")
    return result


def _source_wrapper(
    task_id: str, config_id: str, bound_row: Mapping[str, Any]
) -> dict[str, Any]:
    source_path = Path(str(bound_row.get("source_evidence_path") or ""))
    expected_source_sha = str(bound_row.get("source_evidence_sha256") or "")
    source = _verified_payload(
        source_path,
        expected_source_sha,
        label="independent source",
    )
    if (
        source.get("status") != "ready"
        or source.get("source_plan_sha256") != bound_row.get("source_plan_sha256")
    ):
        raise ValueError(f"independent source contract mismatch: {config_id}")
    nested_pairs = (
        ("checkpoint_path", "checkpoint_sha256"),
        ("onnx_path", "onnx_sha256"),
        ("calibration_path", "calibration_sha256"),
        ("calibration_summary_path", "calibration_summary_sha256"),
    )
    for path_field, sha_field in nested_pairs:
        nested_path = Path(str(source.get(path_field) or ""))
        if (
            not nested_path.is_file()
            or _sha(nested_path) != str(source.get(sha_field) or "")
        ):
            raise ValueError(
                f"independent source nested artifact SHA mismatch: {config_id}:{path_field}"
            )
    return {
        "schema_version": "stage5_independent_source_evidence_v1",
        "status": "ready",
        "task_id": task_id,
        "configuration_id": config_id,
        "independent_from_search_measurement": True,
        "raw_source_evidence_path": str(source_path),
        "raw_source_evidence_sha256": _sha(source_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = []
    normalized_root = args.output_json.parent / "normalized"
    for task_id in TASKS:
        closure = _read(
            args.closure_root / task_id / "stage5_task_closure_audit_v3.json"
        )
        request = _read(args.validation_root / task_id / "validation_request.json")
        performance_manifest = _read(
            args.validation_root
            / task_id
            / "repeat_2/performance_manifest.json"
        )
        selected = list(closure["independent_validation_ids"])
        rows = {
            str(row["manifest_job_id"]): row for row in request.get("rows") or []
        }
        bound_rows = {
            str(row["manifest_job_id"]): row
            for row in performance_manifest.get("jobs") or []
        }
        if set(rows) != set(selected) or set(bound_rows) != set(selected):
            raise ValueError(f"validation request selection drift: {task_id}")
        repeat_states = [
            _read_jsonl(
                args.validation_root
                / task_id
                / f"repeat_{repeat_index}/performance_state.jsonl"
            )
            for repeat_index in range(3)
        ]
        repeat_jobs = [
            _read_jsonl(
                args.validation_root
                / task_id
                / f"repeat_{repeat_index}/performance_jobs.jsonl"
            )
            for repeat_index in range(3)
        ]
        ap_states = _read_jsonl(
            args.validation_root / task_id / "ap/ap_state.jsonl"
        )
        configurations = []
        frontier_points = {
            str(point["manifest_job_id"]): point
            for point in closure.get("frontier_points") or []
        }
        for config_id in selected:
            config_dir = normalized_root / task_id / hashlib.sha256(
                config_id.encode()
            ).hexdigest()[:16]
            repeats = []
            repeat_payloads = []
            for repeat_index, (jobs, states) in enumerate(
                zip(repeat_jobs, repeat_states)
            ):
                state = successful_state_for_configuration(
                    config_id, jobs=jobs, states=states
                )
                wrapper = normalize_performance_repeat(
                    task_id=task_id,
                    configuration_id=config_id,
                    repeat_index=repeat_index,
                    state=state,
                )
                repeat_payloads.append(wrapper)
                wrapper_path = config_dir / f"performance_repeat_{repeat_index}.json"
                repeats.append(
                    {
                        "repeat_id": wrapper["repeat_id"],
                        "performance_result_json": str(wrapper_path),
                        "performance_result_sha256": _write(wrapper_path, wrapper),
                    }
                )
            ap_wrapper = normalize_ap_report(
                task_id=task_id,
                configuration_id=config_id,
                terminal=unique_successful_ap_terminal(config_id, ap_states),
            )
            ap_path = config_dir / "full_ap.json"
            source_wrapper = _source_wrapper(
                task_id, config_id, bound_rows[config_id]
            )
            consistency = evaluate_consistency(
                frontier_points[config_id]["objectives"],
                repeat_payloads,
                ap_wrapper,
                raise_on_failure=False,
            )
            source_path = config_dir / "source_evidence.json"
            configurations.append(
                {
                    "configuration_id": config_id,
                    "performance_repeats": repeats,
                    "ap_report_path": str(ap_path),
                    "ap_report_sha256": _write(ap_path, ap_wrapper),
                    "evidence_path": str(source_path),
                    "evidence_sha256": _write(source_path, source_wrapper),
                    "consistency": consistency,
                }
            )
        tasks.append({
            "task_id": task_id,
            "passed": all(item["consistency"]["passed"] for item in configurations),
            "configurations": configurations,
        })
    all_tasks_passed = all(task["passed"] for task in tasks)
    audit = {
        "schema_version": "stage5_independent_validation_audit_v1",
        "all_tasks_passed": all_tasks_passed,
        "task_count": len(tasks),
        "configuration_count": sum(len(task["configurations"]) for task in tasks),
        "tasks": tasks,
    }
    _write(args.output_json, audit)
    print(json.dumps({"task_count": len(tasks), "all_tasks_passed": all_tasks_passed}, sort_keys=True))
    return 0 if all_tasks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
