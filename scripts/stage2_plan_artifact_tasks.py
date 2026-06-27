#!/usr/bin/env python3
"""Plan Stage2 artifact-agent tasks from a durable candidate queue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.artifact_registry import (  # noqa: E402
    ARTIFACT_REGISTRY_SCHEMA_VERSION,
    artifact_id_for_row,
    validate_artifact_registry_row,
)
from framework.stage2.lut_productization import utc_timestamp  # noqa: E402


ARTIFACT_TASK_SCHEMA = "stage2_artifact_task_v1"
ARTIFACT_STATE_SCHEMA = "stage2_artifact_state_v1"
SUMMARY_SCHEMA = "stage2_artifact_task_plan_summary_v1"

MEASUREMENT_AXES = ("latency", "energy", "ap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", required=True)
    parser.add_argument("--quarantine-file", action="append", default=[])
    parser.add_argument("--artifact-tasks-out", required=True)
    parser.add_argument("--artifact-state-out", required=True)
    parser.add_argument("--artifact-registry-out", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--base-path")
    parser.add_argument("--missing-artifact-out")
    parser.add_argument("--created-at")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.exists():
        return []
    return [
        json.loads(line)
        for line in item.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _safe(value: object) -> str:
    return str(value).replace("/", "-").replace(":", "-").replace(" ", "_")


def _label(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("label") or "")
    if value:
        return value
    candidate_id = str(candidate.get("candidate_id") or "")
    if ":" in candidate_id:
        value = candidate_id.split(":")[-1]
        if value:
            return value
    config_id = str(candidate.get("config_id") or "")
    return config_id or "unknown"


def _config_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("config_id") or candidate.get("candidate_id") or _label(candidate))


def _candidate_quarantine_keys(candidate: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for key in ("config_id", "candidate_id", "config_id_default", "config_id_tuned"):
        value = candidate.get(key)
        if value:
            keys.append(str(value))
    for value in candidate.get("config_ids", []) or []:
        if value:
            keys.append(str(value))
    deduped: list[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    return deduped


def _base_path(args: argparse.Namespace) -> Path:
    value = args.artifact_root or args.base_path or "."
    return Path(value)


def _path_value(candidate: dict[str, Any], key: str) -> str | None:
    value = candidate.get(key)
    if value:
        return str(value)
    artifacts = candidate.get("artifacts")
    if isinstance(artifacts, dict) and artifacts.get(key):
        return str(artifacts[key])
    return None


def _default_onnx_path(candidate: dict[str, Any]) -> str:
    return f"models/{_safe(_label(candidate))}_backbone.onnx"


def _default_tvm_work_dir(candidate: dict[str, Any]) -> str:
    return f"workdirs/{_safe(_label(candidate))}"


def _default_ap_source_path(candidate: dict[str, Any]) -> str:
    return f"ap/{_safe(_label(candidate))}_ap.jsonl"


def _resolve(base_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return base_path / path


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _json_readable(path: Path) -> bool:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        pass
    try:
        for line in text.splitlines():
            if line.strip():
                json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def _quarantine_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status", "active") != "active":
            continue
        for key in ("config_id", "candidate_id"):
            value = row.get(key)
            if value:
                index.setdefault(str(value), []).append(row)
    return index


def _active_quarantine_refs(
    candidate: dict[str, Any],
    quarantine_by_id: dict[str, list[dict[str, Any]]],
) -> list[str]:
    refs: list[str] = []
    for key in _candidate_quarantine_keys(candidate):
        for row in quarantine_by_id.get(key, []):
            ref = str(row.get("job_id") or row.get("config_id") or row.get("candidate_id"))
            if ref and ref not in refs:
                refs.append(ref)
    return refs


def _axes_required(candidate: dict[str, Any]) -> list[str]:
    axes = candidate.get("axes_required") or MEASUREMENT_AXES
    return [axis for axis in axes if axis in MEASUREMENT_AXES]


def _existing_source_files(paths: list[Path | None]) -> list[str]:
    files: list[str] = []
    for path in paths:
        if path and path.exists():
            text = str(path)
            if text not in files:
                files.append(text)
    return files


def _artifact_paths(
    candidate: dict[str, Any],
    base_path: Path,
) -> dict[str, Path | None]:
    onnx_path = _resolve(
        base_path,
        _path_value(candidate, "onnx_path") or _default_onnx_path(candidate),
    )
    tvm_work_dir = _resolve(
        base_path,
        _path_value(candidate, "tvm_work_dir") or _default_tvm_work_dir(candidate),
    )
    workload_path = _resolve(base_path, _path_value(candidate, "database_workload_path"))
    tuning_record_path = _resolve(
        base_path,
        _path_value(candidate, "database_tuning_record_path"),
    )
    if tvm_work_dir is not None:
        workload_path = workload_path or tvm_work_dir / "database_workload.json"
        tuning_record_path = tuning_record_path or tvm_work_dir / "database_tuning_record.json"
    ap_source_path = _resolve(
        base_path,
        _path_value(candidate, "ap_source_path") or _default_ap_source_path(candidate),
    )
    return {
        "onnx_path": onnx_path,
        "tvm_work_dir": tvm_work_dir,
        "database_workload_path": workload_path,
        "database_tuning_record_path": tuning_record_path,
        "ap_source_path": ap_source_path,
    }


def _check_artifacts(paths: dict[str, Path | None]) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    bad_db: list[str] = []
    onnx_path = paths["onnx_path"]
    tvm_work_dir = paths["tvm_work_dir"]
    workload_path = paths["database_workload_path"]
    tuning_record_path = paths["database_tuning_record_path"]

    if onnx_path is None or not onnx_path.is_file():
        missing.append("onnx_path")
    if tvm_work_dir is None or not tvm_work_dir.is_dir():
        missing.append("tvm_work_dir")
    for field, path in (
        ("database_workload_path", workload_path),
        ("database_tuning_record_path", tuning_record_path),
    ):
        if path is None or not path.is_file():
            missing.append(field)
            continue
        if not _json_readable(path):
            bad_db.append(field)
    return missing, bad_db


def _plan_candidate(
    candidate: dict[str, Any],
    *,
    base_path: Path,
    quarantine_by_id: dict[str, list[dict[str, Any]]],
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    config_id = _config_id(candidate)
    candidate_id = str(candidate.get("candidate_id") or config_id)
    paths = _artifact_paths(candidate, base_path)
    missing_artifacts, bad_db_fields = _check_artifacts(paths)
    quarantine_refs = _active_quarantine_refs(candidate, quarantine_by_id)
    ap_source_path = paths["ap_source_path"]
    ap_source_exists = ap_source_path is not None and ap_source_path.is_file()
    ap_status = "ready" if ap_source_exists else "source_missing"

    artifact_status = "ready"
    failure_reason: str | None = None
    if quarantine_refs:
        artifact_status = "quarantined"
        failure_reason = "quarantined"
    elif bad_db_fields:
        artifact_status = "quarantined"
        failure_reason = "quarantined_bad_db"
        quarantine_refs = [f"bad_db:{field}" for field in bad_db_fields]
    elif missing_artifacts:
        artifact_status = "missing"
        failure_reason = "missing_artifact"

    measurement_jobs: list[str] = []
    if artifact_status == "ready":
        for axis in _axes_required(candidate):
            if axis == "ap" and ap_status != "ready":
                continue
            measurement_jobs.append(axis)

    common = {
        "candidate_id": candidate_id,
        "config_id": config_id,
        "model": str(candidate.get("model") or candidate.get("model_name") or "unknown"),
        "model_name": str(candidate.get("model") or candidate.get("model_name") or "unknown"),
        "label": _label(candidate),
        "width": list(candidate.get("width") or []),
        "arm": str(candidate.get("arm") or "mixed"),
        "quant_policy": str(candidate.get("quant_policy") or "unknown"),
        "schedule_policy": str(candidate.get("schedule_policy") or "unknown"),
        "optimized_scope": str(candidate.get("optimized_scope") or "unknown"),
        "priority": int(candidate.get("priority") or 0),
        "axes_required": _axes_required(candidate),
        "artifact_status": artifact_status,
        "ap_status": ap_status,
        "failure_reason": failure_reason,
        "missing_artifacts": missing_artifacts,
        "quarantine_refs": quarantine_refs,
        "onnx_path": _display_path(paths["onnx_path"]),
        "tvm_work_dir": _display_path(paths["tvm_work_dir"]),
        "database_workload_path": _display_path(paths["database_workload_path"]),
        "database_tuning_record_path": _display_path(paths["database_tuning_record_path"]),
        "ap_source_path": _display_path(paths["ap_source_path"]) if ap_source_exists else None,
        "updated_at": created_at,
    }
    task = {
        "schema": ARTIFACT_TASK_SCHEMA,
        "task_id": f"artifact:{config_id}",
        **common,
        "measurement_jobs": measurement_jobs,
        "created_at": created_at,
    }
    state = {
        "schema": ARTIFACT_STATE_SCHEMA,
        **common,
        "quality_status": "ready" if artifact_status == "ready" else artifact_status,
    }

    registry = {
        "schema_version": ARTIFACT_REGISTRY_SCHEMA_VERSION,
        "artifact_id": artifact_id_for_row(common),
        "config_id": config_id,
        "candidate_id": candidate_id,
        "model_name": common["model_name"],
        "label": common["label"],
        "width": common["width"],
        "arm": common["arm"],
        "optimized_scope": common["optimized_scope"],
        "onnx_path": common["onnx_path"],
        "tvm_work_dir": common["tvm_work_dir"],
        "database_path": common["tvm_work_dir"],
        "database_workload_path": common["database_workload_path"],
        "database_tuning_record_path": common["database_tuning_record_path"],
        "ap_source_path": common["ap_source_path"],
        "energy_raw_dir": None,
        "latency_row_ids": [],
        "ap_row_ids": [],
        "energy_row_ids": [],
        "artifact_status": artifact_status,
        "ap_status": ap_status,
        "missing_artifacts": missing_artifacts,
        "quarantine_refs": quarantine_refs,
        "source_files": _existing_source_files(
            [
                paths["onnx_path"],
                paths["tvm_work_dir"],
                paths["database_workload_path"],
                paths["database_tuning_record_path"],
                paths["ap_source_path"] if ap_source_exists else None,
            ]
        ),
        "created_at": created_at,
        "updated_at": created_at,
    }
    validate_artifact_registry_row(registry)

    missing_row = None
    if artifact_status == "missing":
        missing_row = {
            "schema": "stage2_missing_artifact_v1",
            "candidate_id": candidate_id,
            "config_id": config_id,
            "model": common["model"],
            "missing_artifacts": missing_artifacts,
            "failure_reason": failure_reason,
            "created_at": created_at,
        }
    return task, state, registry, missing_row


def _summary(rows: list[dict[str, Any]], out_paths: dict[str, str]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    ap_status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["artifact_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        ap_status = str(row["ap_status"])
        ap_status_counts[ap_status] = ap_status_counts.get(ap_status, 0) + 1
    return {
        "schema": SUMMARY_SCHEMA,
        "candidates": len(rows),
        "status_counts": status_counts,
        "ap_status_counts": ap_status_counts,
        **out_paths,
    }


def main() -> int:
    args = parse_args()
    created_at = args.created_at or utc_timestamp()
    base_path = _base_path(args)
    candidates = read_jsonl(args.candidate_queue)
    quarantine_rows: list[dict[str, Any]] = []
    for path in args.quarantine_file:
        quarantine_rows.extend(read_jsonl(path))
    quarantine_by_id = _quarantine_index(quarantine_rows)

    tasks: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        task, state, registry, missing = _plan_candidate(
            candidate,
            base_path=base_path,
            quarantine_by_id=quarantine_by_id,
            created_at=created_at,
        )
        tasks.append(task)
        states.append(state)
        registry_rows.append(registry)
        if missing is not None:
            missing_rows.append(missing)

    if not args.dry_run:
        write_jsonl(args.artifact_tasks_out, tasks)
        write_jsonl(args.artifact_state_out, states)
        write_jsonl(args.artifact_registry_out, registry_rows)
        if args.missing_artifact_out:
            write_jsonl(args.missing_artifact_out, missing_rows)

    print(
        json.dumps(
            _summary(
                states,
                {
                    "artifact_tasks_out": args.artifact_tasks_out,
                    "artifact_state_out": args.artifact_state_out,
                    "artifact_registry_out": args.artifact_registry_out,
                },
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
