#!/usr/bin/env python3
"""Finalize the four-task Stage5 v3 full-budget search from frozen files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.closure_v3 import build_stage5_closure_audit  # noqa: E402
from framework.stage5.single_target_search_v2 import (  # noqa: E402
    verify_frozen_coldstart_artifacts,
)


TASKS = (
    "S5-PYR-TVM",
    "S5-PYR-TRT",
    "S5-COD-TVM",
    "S5-COD-TRT",
)
SUCCESS_STATUS = "measured_success_gold"
FAILURE_STATUSES = {"feasibility_failure", "numerical_feasibility_failure"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}: {error.msg}") from error


def _mapping(payload: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _list(payload: Any, *, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(payload, list) or not all(
        isinstance(item, Mapping) for item in payload
    ):
        raise ValueError(f"{label} must be a JSON list of objects")
    return list(payload)


def _rows(payload: Any, *, label: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return _list(payload, label=label)
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return _list(payload["rows"], label=f"{label}.rows")
    raise ValueError(f"{label} must be a list or an object containing rows")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _artifact_path(raw_path: Any, *, base: Path, label: str) -> Path:
    value = str(raw_path or "")
    if not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _verify_file_pair(
    row: Mapping[str, Any],
    path_field: str,
    sha_field: str,
    *,
    base: Path,
    row_id: str,
) -> dict[str, str]:
    path = _artifact_path(
        row.get(path_field), base=base, label=f"{row_id}:{path_field}"
    )
    expected = str(row.get(sha_field) or "")
    if len(expected) != 64:
        raise ValueError(f"{path_field} SHA missing or malformed: {row_id}")
    if not path.is_file():
        raise ValueError(f"{path_field} artifact missing: {row_id}:{path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"{path_field} SHA mismatch: {row_id}: expected {expected}, got {actual}"
        )
    return {"path": str(path), "sha256": actual}


def _nested_finite(
    payload: Mapping[str, Any], paths: Sequence[tuple[str, ...]]
) -> float | None:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if not isinstance(value, bool):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _metric_matches(row: Mapping[str, Any], key: str, evidence: float | None) -> bool:
    try:
        measured = float(row[key])
    except (KeyError, TypeError, ValueError):
        return False
    return evidence is not None and math.isclose(
        measured, evidence, rel_tol=1e-9, abs_tol=1e-12
    )


def _verify_success_metric_evidence(
    row: Mapping[str, Any], *, formal_root: Path, row_id: str
) -> None:
    performance_path = _artifact_path(
        row.get("performance_result_json"),
        base=formal_root,
        label=f"{row_id}:performance_result_json",
    )
    performance = _mapping(
        _read_json(performance_path), label=f"{row_id} performance evidence"
    )
    latency = _nested_finite(
        performance,
        (
            ("lat_p50_ms",),
            ("latency_ms",),
            ("latency", "latency_ms_p50"),
            ("latency", "lat_p50_ms"),
        ),
    )
    energy = _nested_finite(
        performance,
        (
            ("energy_j",),
            ("energy", "energy_j"),
            ("energy", "joules"),
            ("energy", "joules_per_inference"),
            ("energy", "joule_per_inference"),
            ("energy", "energy_J"),
        ),
    )
    for key, evidence in (("latency_ms", latency), ("energy_j", energy)):
        if not _metric_matches(row, key, evidence):
            raise ValueError(f"feedback metric evidence mismatch: {row_id}:{key}")

    ap_path = _artifact_path(
        row.get("ap_report_path"), base=formal_root, label=f"{row_id}:ap_report_path"
    )
    report = _mapping(_read_json(ap_path), label=f"{row_id} AP evidence")
    ap_source = report.get("ap") if isinstance(report.get("ap"), Mapping) else report
    for key in ("ap30", "ap50", "ap70"):
        evidence = _nested_finite(ap_source, ((key,),))
        if not _metric_matches(row, key, evidence):
            raise ValueError(f"feedback metric evidence mismatch: {row_id}:{key}")
    if (
        report.get("status") != "success"
        or int(report.get("processed_samples") or 0) != 1789
        or (
            report.get("requested_num_samples") is not None
            and int(report["requested_num_samples"]) != 1789
        )
        or int(report.get("failed_samples") or 0) != 0
        or int(report.get("fallback_samples") or 0) != 0
    ):
        raise ValueError(f"feedback full-AP contract mismatch: {row_id}")


def _verify_feedback_evidence(
    rows: Sequence[Mapping[str, Any]], *, formal_root: Path
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("manifest_job_id") or row.get("row_id") or "")
        if not row_id:
            raise ValueError("feedback row identity is missing")
        pairs = [
            ("materialized_source_evidence_path", "materialized_source_evidence_sha256")
        ]
        status = str(row.get("terminal_status") or "")
        if status == SUCCESS_STATUS:
            pairs.extend(
                (
                    ("performance_result_json", "performance_result_sha256"),
                    ("ap_report_path", "ap_report_sha256"),
                )
            )
        elif status in FAILURE_STATUSES:
            pairs.append(("failure_evidence_path", "failure_evidence_sha256"))
            if status == "numerical_feasibility_failure":
                pairs.append(("performance_result_json", "performance_result_sha256"))
        else:
            raise ValueError(f"non-terminal feedback row: {row_id}:{status or '<empty>'}")
        verified_by_field = {}
        for path_field, sha_field in pairs:
            verified = _verify_file_pair(
                row, path_field, sha_field, base=formal_root, row_id=row_id
            )
            verified_by_field[path_field] = verified
            evidence.append({"row_id": row_id, "field": path_field, **verified})
        source_payload = _mapping(
            _read_json(
                Path(
                    verified_by_field["materialized_source_evidence_path"]["path"]
                )
            ),
            label=f"{row_id} materialized source evidence",
        )
        if source_payload.get("source_plan_sha256") != row.get(
            "source_evidence_sha256"
        ):
            raise ValueError(f"materialized source plan mismatch: {row_id}")
        if status == SUCCESS_STATUS:
            _verify_success_metric_evidence(
                row, formal_root=formal_root, row_id=row_id
            )
    return evidence


def _validate_task_terminal(path: Path, task_id: str) -> Mapping[str, Any]:
    terminal = _mapping(_read_json(path), label=f"{task_id} budget terminal")
    expected = {
        "schema_version": "stage5_task_budget_terminal_v3",
        "task_id": task_id,
        "status": "budget_exhausted",
        "budget_consumed": 16,
        "round_count": 4,
    }
    if any(terminal.get(field) != value for field, value in expected.items()):
        raise ValueError(f"invalid task budget terminal: {task_id}")
    return terminal


def _validate_scheduler_terminal(path: Path) -> Mapping[str, Any]:
    terminal = _mapping(_read_json(path), label="full-budget scheduler terminal")
    expected = {
        "schema_version": "stage5_full_budget_scheduler_terminal_v3",
        "status": "budget_exhausted",
        "task_count": 4,
        "round_count": 16,
        "formal_online_genomes": 64,
    }
    if any(terminal.get(field) != value for field, value in expected.items()):
        raise ValueError("invalid full-budget scheduler terminal")
    return terminal


def _validate_independent_validation(
    path: Path, validation_ids_by_task: Mapping[str, set[str]]
) -> dict[str, Any]:
    audit = _mapping(_read_json(path), label="independent validation audit")
    if audit.get("schema_version") != "stage5_independent_validation_audit_v1" or not isinstance(
        audit.get("all_tasks_passed"), bool
    ):
        raise ValueError("independent validation audit status is incomplete")
    tasks = _list(audit.get("tasks"), label="independent validation tasks")
    by_id = {str(task.get("task_id") or ""): task for task in tasks}
    if len(tasks) != 4 or set(by_id) != set(TASKS):
        raise ValueError("independent validation must contain exactly the four Stage5 tasks")

    configuration_count = 0
    repeat_count = 0
    evidence_count = 0
    for task_id in TASKS:
        task = by_id[task_id]
        if not isinstance(task.get("passed"), bool):
            raise ValueError(f"independent validation task status is incomplete: {task_id}")
        configurations = _list(
            task.get("configurations"), label=f"{task_id} validation configurations"
        )
        expected_ids = validation_ids_by_task[task_id]
        if len(configurations) != len(expected_ids):
            raise ValueError(
                f"independent validation selection count mismatch: {task_id}"
            )
        config_ids = [
            str(config.get("configuration_id") or config.get("manifest_job_id") or "")
            for config in configurations
        ]
        if (
            any(not config_id for config_id in config_ids)
            or len(set(config_ids)) != len(config_ids)
            or set(config_ids) != expected_ids
        ):
            raise ValueError(
                f"independent validation configurations must exactly match the frozen Pareto selection: {task_id}"
            )
        for config_id, config in zip(config_ids, configurations):
            consistency = config.get("consistency")
            if not isinstance(consistency, Mapping) or not isinstance(
                consistency.get("passed"), bool
            ):
                raise ValueError(
                    f"independent consistency contract mismatch: {config_id}"
                )
            thresholds = consistency.get("thresholds")
            deltas = consistency.get("deltas")
            rerun = consistency.get("rerun")
            expected_thresholds = {
                "latency_relative": 0.15,
                "energy_relative": 0.20,
                "ap70_absolute": 0.01,
                "latency_cv": 0.10,
                "energy_cv": 0.15,
            }
            if not all(
                isinstance(source, Mapping)
                for source in (thresholds, deltas, rerun)
            ) or any(
                not math.isclose(
                    float(thresholds.get(key, float("nan"))),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for key, value in expected_thresholds.items()
            ):
                raise ValueError(
                    f"independent consistency contract mismatch: {config_id}"
                )
            consistency_values = {
                "latency_relative": deltas.get("latency_relative"),
                "energy_relative": deltas.get("energy_relative"),
                "ap70_absolute": deltas.get("ap70_absolute"),
                "latency_cv": rerun.get("latency_cv"),
                "energy_cv": rerun.get("energy_cv"),
            }
            finite = all(
                _nested_finite(consistency_values, ((key,),)) is not None
                for key in expected_thresholds
            )
            computed_pass = finite and all(
                float(consistency_values[key]) <= expected_thresholds[key]
                for key in expected_thresholds
            )
            if consistency.get("passed") is not computed_pass:
                raise ValueError(
                    f"independent consistency contract mismatch: {config_id}"
                )
            repeats = _list(
                config.get("performance_repeats"),
                label=f"{task_id}:{config_id} performance repeats",
            )
            repeat_ids = [str(repeat.get("repeat_id") or "") for repeat in repeats]
            if len(repeats) != 3 or any(not repeat_id for repeat_id in repeat_ids):
                raise ValueError(
                    f"exactly 3 independent performance repeats required: {config_id}"
                )
            if len(set(repeat_ids)) != 3:
                raise ValueError(
                    f"performance repeat identities must be independent: {config_id}"
                )
            repeat_paths = {
                _artifact_path(
                    repeat.get("performance_result_json"),
                    base=path.parent,
                    label=f"{config_id}:performance_result_json",
                ).resolve()
                for repeat in repeats
            }
            if len(repeat_paths) != 3:
                raise ValueError(
                    f"performance repeat artifacts must be independent: {config_id}"
                )
            run_uuids = set()
            for repeat_id, repeat in zip(repeat_ids, repeats):
                verified_repeat = _verify_file_pair(
                    repeat,
                    "performance_result_json",
                    "performance_result_sha256",
                    base=path.parent,
                    row_id=f"{config_id}:{repeat_id}",
                )
                payload = _mapping(
                    _read_json(Path(verified_repeat["path"])),
                    label=f"{config_id}:{repeat_id} independent performance",
                )
                run_uuid = str(payload.get("run_uuid") or "")
                try:
                    started = datetime.fromisoformat(
                        str(payload.get("started_at_utc") or "")
                    )
                    ended = datetime.fromisoformat(
                        str(payload.get("ended_at_utc") or "")
                    )
                except ValueError as error:
                    raise ValueError(
                        f"independent performance timestamps invalid: {config_id}:{repeat_id}"
                    ) from error
                if (
                    payload.get("schema_version")
                    != "stage5_independent_performance_repeat_v1"
                    or payload.get("status") != "success"
                    or payload.get("task_id") != task_id
                    or payload.get("configuration_id") != config_id
                    or payload.get("hardware_id") != "h800"
                    or payload.get("repeat_id") != repeat_id
                    or not run_uuid
                    or ended <= started
                    or _nested_finite(payload, (("latency_ms",),)) is None
                    or _nested_finite(payload, (("energy_j",),)) is None
                ):
                    raise ValueError(
                        f"independent performance contract mismatch: {config_id}:{repeat_id}"
                    )
                run_uuids.add(run_uuid)
            if len(run_uuids) != 3:
                raise ValueError(
                    f"independent performance run UUIDs must be unique: {config_id}"
                )
            verified_ap = _verify_file_pair(
                config,
                "ap_report_path",
                "ap_report_sha256",
                base=path.parent,
                row_id=config_id,
            )
            ap_payload = _mapping(
                _read_json(Path(verified_ap["path"])),
                label=f"{config_id} independent AP",
            )
            if (
                ap_payload.get("schema_version") != "stage5_independent_full_ap_v1"
                or ap_payload.get("status") != "success"
                or ap_payload.get("task_id") != task_id
                or ap_payload.get("configuration_id") != config_id
                or int(ap_payload.get("processed_samples") or 0) != 1789
                or int(ap_payload.get("requested_num_samples") or 0) != 1789
                or int(ap_payload.get("failed_samples") or 0) != 0
                or int(ap_payload.get("fallback_samples") or 0) != 0
                or any(
                    _nested_finite(ap_payload, ((metric,),)) is None
                    for metric in ("ap30", "ap50", "ap70")
                )
            ):
                raise ValueError(f"independent full-AP contract mismatch: {config_id}")
            verified_source = _verify_file_pair(
                config,
                "evidence_path",
                "evidence_sha256",
                base=path.parent,
                row_id=config_id,
            )
            source_payload = _mapping(
                _read_json(Path(verified_source["path"])),
                label=f"{config_id} independent source evidence",
            )
            if (
                source_payload.get("schema_version")
                != "stage5_independent_source_evidence_v1"
                or source_payload.get("status") != "ready"
                or source_payload.get("task_id") != task_id
                or source_payload.get("configuration_id") != config_id
                or source_payload.get("independent_from_search_measurement") is not True
            ):
                raise ValueError(
                    f"independent source evidence contract mismatch: {config_id}"
                )
            configuration_count += 1
            repeat_count += 3
            evidence_count += 5
        task_computed_pass = all(
            bool(config["consistency"]["passed"]) for config in configurations
        )
        if task.get("passed") is not task_computed_pass:
            raise ValueError(f"independent validation task status mismatch: {task_id}")
    all_tasks_passed = all(bool(by_id[task_id]["passed"]) for task_id in TASKS)
    if audit.get("all_tasks_passed") is not all_tasks_passed:
        raise ValueError("independent validation aggregate status mismatch")
    return {
        "audit_path": str(path),
        "audit_sha256": _sha256(path),
        "task_count": 4,
        "configuration_count": configuration_count,
        "performance_repeat_count": repeat_count,
        "verified_evidence_count": evidence_count,
        "all_tasks_passed": all_tasks_passed,
    }


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _write_idempotent(path: Path, payload: Any) -> None:
    content = _json_bytes(payload)
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--coldstart-rows-json", type=Path, required=True)
    parser.add_argument("--coldstart-graph-features-json", type=Path, required=True)
    parser.add_argument("--expected-coldstart-rows-sha256")
    parser.add_argument("--expected-coldstart-graph-features-sha256")
    parser.add_argument("--independent-validation-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    verify_kwargs = {}
    if args.expected_coldstart_rows_sha256:
        verify_kwargs["expected_rows_sha256"] = args.expected_coldstart_rows_sha256
    if args.expected_coldstart_graph_features_sha256:
        verify_kwargs["expected_graph_features_sha256"] = (
            args.expected_coldstart_graph_features_sha256
        )
    coldstart_audit = verify_frozen_coldstart_artifacts(
        args.coldstart_rows_json,
        args.coldstart_graph_features_json,
        **verify_kwargs,
    )
    gold176 = _rows(_read_json(args.coldstart_rows_json), label="Gold176 rows")
    if len(gold176) != 176:
        raise ValueError("Gold176 input must contain exactly 176 rows")
    scheduler_terminal_path = (
        args.formal_root / "controller/full_budget_scheduler_terminal.json"
    )
    _validate_scheduler_terminal(scheduler_terminal_path)

    closures: list[dict[str, Any]] = []
    output_payloads: list[tuple[Path, Mapping[str, Any]]] = []
    all_online_ids: set[str] = set()
    validation_ids_by_task: dict[str, set[str]] = {}
    total_verified_evidence = 0
    evidence_rounds: list[dict[str, Any]] = []
    task_wall_clocks: list[dict[str, Any]] = []
    control_artifacts = [
        {
            "kind": "scheduler_terminal",
            "path": str(scheduler_terminal_path),
            "sha256": _sha256(scheduler_terminal_path),
        }
    ]
    for task_id in TASKS:
        task_root = args.formal_root / task_id
        manifest = _mapping(
            _read_json(task_root / "candidate_manifest.json"),
            label=f"{task_id} candidate manifest",
        )
        task_terminal_path = task_root / "task_budget_terminal.json"
        _validate_task_terminal(task_terminal_path, task_id)
        control_artifacts.append(
            {
                "kind": "task_terminal",
                "task_id": task_id,
                "path": str(task_terminal_path),
                "sha256": _sha256(task_terminal_path),
            }
        )
        requests = []
        feedback_batches = []
        atomic_audits = []
        task_online_ids: set[str] = set()
        task_round_clocks: list[dict[str, Any]] = []
        for round_index in range(4):
            round_root = task_root / f"round_{round_index:02d}"
            request_path = round_root / "measurement_request.json"
            feedback_path = round_root / "final/stage5_feedback_v2_final.json"
            atomic_path = round_root / "final/atomic_batch_audit.json"
            request = _mapping(
                _read_json(request_path),
                label=f"{task_id} round {round_index} measurement request",
            )
            feedback = _list(
                _read_json(feedback_path),
                label=f"{task_id} round {round_index} final feedback",
            )
            atomic = _mapping(
                _read_json(atomic_path),
                label=f"{task_id} round {round_index} atomic audit",
            )
            verified = _verify_feedback_evidence(feedback, formal_root=args.formal_root)
            total_verified_evidence += len(verified)
            started = request_path.stat().st_mtime
            ended = atomic_path.stat().st_mtime
            if ended < started:
                raise ValueError(f"round wall-clock order is invalid: {task_id}:{round_index}")
            round_clock = {
                "round_index": round_index,
                "started_at_utc": _mtime_iso(request_path),
                "ended_at_utc": _mtime_iso(atomic_path),
                "elapsed_s": float(ended - started),
                "start_evidence": "measurement_request_mtime",
                "end_evidence": "atomic_batch_audit_mtime",
            }
            task_round_clocks.append(round_clock)
            evidence_rounds.append(
                {
                    "task_id": task_id,
                    "round_index": round_index,
                    "measurement_request": {
                        "path": str(request_path),
                        "sha256": _sha256(request_path),
                    },
                    "feedback": {
                        "path": str(feedback_path),
                        "sha256": _sha256(feedback_path),
                    },
                    "atomic_batch_audit": {
                        "path": str(atomic_path),
                        "sha256": _sha256(atomic_path),
                    },
                    "feedback_artifacts": verified,
                    "wall_clock": round_clock,
                }
            )
            requests.append(request)
            feedback_batches.append(feedback)
            atomic_audits.append(atomic)
            for row in _rows(request, label=f"{task_id} request rows"):
                row_id = str(row.get("manifest_job_id") or row.get("row_id") or "")
                if row_id:
                    task_online_ids.add(row_id)
        if len(task_online_ids) != 16:
            raise ValueError(f"task must contain exactly 16 unique online rows: {task_id}")
        shared = all_online_ids & task_online_ids
        if shared:
            raise ValueError(f"exact online row IDs shared between tasks: {sorted(shared)}")
        all_online_ids.update(task_online_ids)
        closure = {
            **build_stage5_closure_audit(
                gold176, manifest, requests, feedback_batches, atomic_audits
            ),
            "budget_search_closed": True,
        }
        validation_ids_by_task[task_id] = set(
            closure["independent_validation_ids"]
        )
        closure_path = args.output_dir / task_id / "stage5_task_closure_audit_v3.json"
        output_payloads.append((closure_path, closure))
        closures.append(
            {
                "task_id": task_id,
                "closure_path": str(closure_path),
                "closure_sha256": hashlib.sha256(_json_bytes(closure)).hexdigest(),
                "online_count": 16,
                "closure": True,
            }
        )
        task_start = min(
            datetime.fromisoformat(item["started_at_utc"]).timestamp()
            for item in task_round_clocks
        )
        task_end = max(
            datetime.fromisoformat(item["ended_at_utc"]).timestamp()
            for item in task_round_clocks
        )
        task_wall_clocks.append(
            {
                "task_id": task_id,
                "started_at_utc": datetime.fromtimestamp(
                    task_start, tz=timezone.utc
                ).isoformat(),
                "ended_at_utc": datetime.fromtimestamp(
                    task_end, tz=timezone.utc
                ).isoformat(),
                "elapsed_s": float(task_end - task_start),
                "rounds": task_round_clocks,
            }
        )
    if len(all_online_ids) != 64:
        raise ValueError("full-budget closure requires exactly 64 unique online rows")

    independent_validation = None
    phase6_ready = False
    if args.independent_validation_audit is not None:
        independent_validation = _validate_independent_validation(
            args.independent_validation_audit, validation_ids_by_task
        )
        phase6_ready = independent_validation["all_tasks_passed"] is True
    wall_start = min(
        datetime.fromisoformat(item["started_at_utc"]).timestamp()
        for item in task_wall_clocks
    )
    wall_end = max(
        datetime.fromisoformat(item["ended_at_utc"]).timestamp()
        for item in task_wall_clocks
    )
    wall_clock_audit = {
        "schema_version": "stage5_wall_clock_audit_v1",
        "task_count": 4,
        "round_count": 16,
        "started_at_utc": datetime.fromtimestamp(
            wall_start, tz=timezone.utc
        ).isoformat(),
        "ended_at_utc": datetime.fromtimestamp(wall_end, tz=timezone.utc).isoformat(),
        "total_elapsed_s": float(wall_end - wall_start),
        "tasks": task_wall_clocks,
    }
    evidence_manifest_path = args.output_dir / "stage5_evidence_manifest_v3.json"
    evidence_manifest = {
        "schema_version": "stage5_evidence_manifest_v3",
        "formal_root": str(args.formal_root),
        "round_count": 16,
        "verified_feedback_artifact_count": total_verified_evidence,
        "control_artifacts": control_artifacts,
        "rounds": evidence_rounds,
    }
    summary = {
        "schema_version": "stage5_full_budget_closure_summary_v3",
        "coldstart_artifact_audit": coldstart_audit,
        "formal_task_count": 4,
        "formal_round_count": 16,
        "formal_online_rows": 64,
        "verified_formal_evidence_count": total_verified_evidence,
        "formal_evidence_manifest_path": str(evidence_manifest_path),
        "formal_evidence_manifest_sha256": hashlib.sha256(
            _json_bytes(evidence_manifest)
        ).hexdigest(),
        "wall_clock_audit": wall_clock_audit,
        "tasks": closures,
        "budget_search_closed": True,
        "engineering_search_closed": phase6_ready,
        "phase6_entry_ready": phase6_ready,
        "independent_validation": independent_validation,
    }
    for path, payload in output_payloads:
        _write_idempotent(path, payload)
    _write_idempotent(evidence_manifest_path, evidence_manifest)
    _write_idempotent(
        args.output_dir / "stage5_full_budget_closure_summary_v3.json", summary
    )
    return summary


def main() -> int:
    try:
        summary = run(parse_args())
    except (OSError, TypeError, ValueError) as error:
        print(f"stage5 full-budget closure failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
