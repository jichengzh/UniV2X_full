#!/usr/bin/env python3
"""Generate Stage2 coverage-first H800 energy jobs from shared artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import job_plan_row  # noqa: E402


DEFAULT_BACKEND = "h800_tvm_power_telemetry"
DEFAULT_QUALITY_POLICY = "energy_quality_gate_v1"
RETEST_TAGS = {"paper_retest", "outlier_retest", "cross_gpu_drift_check"}
READY_ARTIFACT_STATUSES = {"ready", "artifact_ready"}
QUARANTINED_ARTIFACT_STATUSES = {
    "quarantined",
    "quarantine",
    "quarantined_bad_db",
    "bad_db",
    "blocked",
}
PRIORITY_SELECTED_FIELDS = {
    "energy_priority_selected",
    "priority_selected",
    "priority_selected_for_energy",
    "selected_for_energy",
}
ARTIFACT_PATH_FIELDS = (
    "onnx_path",
    "tvm_work_dir",
    "database_workload_path",
    "database_tuning_record_path",
    "vm_artifact_path",
)
REQUIRED_ARTIFACT_PATH_FIELDS = (
    "onnx_path",
    "tvm_work_dir",
    "database_workload_path",
    "database_tuning_record_path",
)
CSV_FIELDS = [
    "candidate_id",
    "label",
    "width",
    "artifact_status",
    "latency_status",
    "energy_status",
    "ap_status",
    "next_action",
    "gap_reason",
    "tag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--artifact-registry", required=True)
    parser.add_argument("--latency-rows", required=True)
    parser.add_argument("--energy-rows", required=True)
    parser.add_argument("--out-queue", required=True)
    parser.add_argument("--out-gap-csv", required=True)
    parser.add_argument("--out-gap-json", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--run-id-prefix", default="energy")
    parser.add_argument("--tag", default="coverage")
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--quality-policy", default=DEFAULT_QUALITY_POLICY)
    parser.add_argument("--rows-out-jsonl", default="rows/energy_lut_rows_v1.jsonl")
    parser.add_argument("--raw-root", default="raw/energy")
    parser.add_argument("--phase", default="coverage_pipeline_v1")
    parser.add_argument("--manifest-path", default="candidates/candidate_queue.jsonl")
    parser.add_argument("--registry-path", default="artifacts/artifact_registry_v1.jsonl")
    parser.add_argument("--manifest-digest", default="coverage_pipeline_v1")
    parser.add_argument("--energy-warmup-iters", type=int, default=50)
    parser.add_argument("--energy-measure-iters", type=int, default=1500)
    return parser.parse_args()


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{p}:{line_no} must contain a JSON object")
        rows.append(payload)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["width"] = json.dumps(csv_row.get("width", []), separators=(",", ":"))
            writer.writerow({field: csv_row.get(field, "") for field in CSV_FIELDS})


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "selected"}


def _safe_id(value: Any) -> str:
    return (
        str(value)
        .replace(":", "-")
        .replace("/", "-")
        .replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "x")
    )


def _width_tuple(row: dict[str, Any]) -> tuple[int | str, ...]:
    width = row.get("width", [])
    if isinstance(width, str):
        parts = [part.strip() for part in width.split(",") if part.strip()]
    elif isinstance(width, list):
        parts = width
    else:
        parts = [width]
    normalized: list[int | str] = []
    for part in parts:
        try:
            normalized.append(int(part))
        except (TypeError, ValueError):
            normalized.append(str(part))
    return tuple(normalized)


def _width_list(row: dict[str, Any]) -> list[int | str]:
    return list(_width_tuple(row))


def _schedule(row: dict[str, Any]) -> str:
    return str(row.get("schedule_policy") or "metaschedule_tuned")


def _cell_key(row: dict[str, Any], *, backend: str) -> tuple[Any, ...]:
    return (
        row.get("model"),
        _width_tuple(row),
        row.get("quant_policy"),
        _schedule(row),
        row.get("optimized_scope"),
        backend,
    )


def _latest_by_candidate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if candidate_id is not None:
            latest[str(candidate_id)] = row
    return latest


def _measured_latency_index(rows: list[dict[str, Any]]) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[Any, ...], dict[str, Any]],
]:
    by_candidate_schedule: dict[tuple[str, str], dict[str, Any]] = {}
    by_cell: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("measurement_status") != "measured":
            continue
        candidate_id = row.get("candidate_id")
        if candidate_id is not None:
            by_candidate_schedule[(str(candidate_id), _schedule(row))] = row
        by_cell[_cell_key(row, backend="h800_tvm")] = row
    return by_candidate_schedule, by_cell


def _measured_energy_index(rows: list[dict[str, Any]], *, backend: str) -> tuple[
    set[tuple[str, str]],
    set[tuple[Any, ...]],
]:
    by_candidate_schedule: set[tuple[str, str]] = set()
    by_cell: set[tuple[Any, ...]] = set()
    for row in rows:
        if row.get("measurement_status") != "measured":
            continue
        if row.get("backend") != backend:
            continue
        candidate_id = row.get("candidate_id")
        if candidate_id is not None:
            by_candidate_schedule.add((str(candidate_id), _schedule(row)))
        by_cell.add(_cell_key(row, backend=backend))
    return by_candidate_schedule, by_cell


def _candidate_requires_energy(candidate: dict[str, Any]) -> bool:
    axes = candidate.get("axes_required")
    if axes is None:
        return True
    if not isinstance(axes, list):
        return str(axes).lower() == "energy"
    return "energy" in {str(axis).lower() for axis in axes}


def _is_priority_selected(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> bool:
    for row in (candidate, artifact or {}):
        for field in PRIORITY_SELECTED_FIELDS:
            if _truthy(row.get(field)):
                return True
        selection = str(
            row.get("energy_selection")
            or row.get("selection_reason")
            or row.get("axis_selection")
            or ""
        ).lower()
        if selection in {"priority_selected", "selected_for_energy", "energy_priority"}:
            return True
    return False


def _artifact_status(artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "missing"
    return str(artifact.get("artifact_status") or artifact.get("status") or "missing")


def _artifact_paths(artifact: dict[str, Any]) -> dict[str, str]:
    return {
        field: str(artifact[field])
        for field in ARTIFACT_PATH_FIELDS
        if artifact.get(field) not in (None, "")
    }


def _missing_artifact_paths(artifact: dict[str, Any]) -> list[str]:
    return [
        field
        for field in REQUIRED_ARTIFACT_PATH_FIELDS
        if artifact.get(field) in (None, "")
    ]


def _config_id(candidate: dict[str, Any]) -> str:
    if candidate.get("config_id_tuned"):
        return str(candidate["config_id_tuned"])
    if candidate.get("config_id"):
        return str(candidate["config_id"])
    width = _width_list(candidate)
    return (
        f"coverage_h800_tvm_pyramid_w{width[0]}x{width[1]}x{width[2]}"
        f"_fp16_{_schedule(candidate)}"
    )


def _software_point_id(candidate: dict[str, Any]) -> str:
    if candidate.get("software_point_id"):
        return str(candidate["software_point_id"])
    width = _width_list(candidate)
    return f"pyramid_lidar:backbone:w{width[0]}x{width[1]}x{width[2]}:fp16"


def _width_csv(candidate: dict[str, Any]) -> str:
    return ",".join(str(item) for item in _width_list(candidate))


def _gap_row(
    candidate: dict[str, Any],
    *,
    artifact_status: str,
    latency_status: str,
    energy_status: str,
    next_action: str,
    gap_reason: str,
    tag: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "label": candidate.get("label"),
        "width": _width_list(candidate),
        "artifact_status": artifact_status,
        "latency_status": latency_status,
        "energy_status": energy_status,
        "ap_status": "not_evaluated_by_energy_generator",
        "next_action": next_action,
        "gap_reason": gap_reason,
        "tag": tag,
    }


def _build_job(
    candidate: dict[str, Any],
    artifact: dict[str, Any],
    *,
    latency_row: dict[str, Any] | None,
    backend: str,
    gpu_id: int,
    run_id_prefix: str,
    index: int,
    tag: str,
    quality_policy: str,
    selection_reason: str,
    created_at: str,
    rows_out_jsonl: str,
    raw_root: str,
    phase: str,
    manifest_path: str,
    registry_path: str,
    manifest_digest: str,
    energy_warmup_iters: int,
    energy_measure_iters: int,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    artifact_paths = _artifact_paths(artifact)
    run_id = f"{run_id_prefix}:{index:04d}:{_safe_id(candidate_id)}"
    job_id = f"energy:{_safe_id(candidate_id)}:{_safe_id(tag)}"
    command = [
        "python3",
        "scripts/stage2_h800_run_measurement_job.py",
        "--kind",
        "energy",
        "--model",
        str(candidate.get("model") or "Pyramid-LiDAR"),
        "--label",
        str(candidate.get("label") or candidate_id),
        "--phase",
        phase,
        "--gpu",
        str(gpu_id),
        "--onnx",
        artifact_paths["onnx_path"],
        "--work-dir",
        artifact_paths["tvm_work_dir"],
        "--width",
        _width_csv(candidate),
        "--candidate-id",
        candidate_id,
        "--software-point-id",
        _software_point_id(candidate),
        "--config-id-tuned",
        _config_id(candidate),
        "--latency-run-id",
        str(latency_row.get("run_id") if latency_row is not None else "priority_selected_without_latency"),
        "--run-id",
        run_id,
        "--raw-root",
        raw_root,
        "--out-jsonl",
        rows_out_jsonl,
        "--manifest-digest",
        manifest_digest,
        "--optimized-scope",
        str(candidate.get("optimized_scope") or "backbone_only"),
        "--quant-policy",
        str(candidate.get("quant_policy") or "fp16"),
        "--energy-warmup-iters",
        str(energy_warmup_iters),
        "--energy-measure-iters",
        str(energy_measure_iters),
    ]
    row = job_plan_row(
        job_id=job_id,
        model=str(candidate.get("model") or "Pyramid-LiDAR"),
        lut_kind="energy",
        job_type="generate_energy_lut",
        priority=int(candidate.get("priority") or 50),
        config_id=_config_id(candidate),
        manifest_path=manifest_path,
        registry_path=registry_path,
        candidate_id=candidate_id,
        software_point_id=_software_point_id(candidate),
        expected_output=rows_out_jsonl,
        command=command,
        max_attempts=1,
        timeout_s=7200,
        resource={
            "server": "h800_remote",
            "gpu": int(gpu_id),
            "exclusive": True,
            "serial_queue": True,
            "backend": backend,
            "quality_policy": quality_policy,
        },
        created_at=created_at,
    )
    row.update(
        {
        "label": candidate.get("label"),
        "width": _width_list(candidate),
        "quant_policy": candidate.get("quant_policy"),
        "schedule_policy": _schedule(candidate),
        "backend": backend,
        "optimized_scope": candidate.get("optimized_scope"),
        "artifact_paths": artifact_paths,
        "gpu_id": gpu_id,
        "run_id": run_id,
        "tag": tag,
        "quality_policy": quality_policy,
        "selection_reason": selection_reason,
        "latency_run_id": None if latency_row is None else latency_row.get("run_id"),
        "latency_config_id": None if latency_row is None else latency_row.get("config_id"),
        "repeat_policy": candidate.get("repeat_policy", "coverage"),
        }
    )
    return row


def generate_energy_jobs(
    *,
    candidates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    backend: str,
    gpu_id: int,
    run_id_prefix: str,
    tag: str,
    allow_repeat: bool,
    quality_policy: str,
    rows_out_jsonl: str = "rows/energy_lut_rows_v1.jsonl",
    raw_root: str = "raw/energy",
    phase: str = "coverage_pipeline_v1",
    manifest_path: str = "candidates/candidate_queue.jsonl",
    registry_path: str = "artifacts/artifact_registry_v1.jsonl",
    manifest_digest: str = "coverage_pipeline_v1",
    energy_warmup_iters: int = 50,
    energy_measure_iters: int = 1500,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_by_candidate = _latest_by_candidate(artifacts)
    latency_by_candidate_schedule, latency_by_cell = _measured_latency_index(latency_rows)
    energy_by_candidate_schedule, energy_by_cell = _measured_energy_index(
        energy_rows,
        backend=backend,
    )

    jobs: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    created_at = utc_timestamp()

    for candidate in candidates:
        if not _candidate_requires_energy(candidate):
            gaps.append(
                _gap_row(
                    candidate,
                    artifact_status="not_required",
                    latency_status="not_checked",
                    energy_status="not_required",
                    next_action="skip_energy_not_required",
                    gap_reason="candidate_axes_do_not_require_energy",
                    tag=tag,
                )
            )
            continue

        candidate_id = str(candidate.get("candidate_id", ""))
        artifact = artifact_by_candidate.get(candidate_id)
        artifact_status = _artifact_status(artifact)
        candidate_schedule = _schedule(candidate)
        latency_row = latency_by_candidate_schedule.get(
            (candidate_id, candidate_schedule)
        ) or latency_by_cell.get(_cell_key(candidate, backend="h800_tvm"))
        latency_status = "measured" if latency_row is not None else "missing"
        energy_measured = (candidate_id, candidate_schedule) in energy_by_candidate_schedule or _cell_key(
            candidate,
            backend=backend,
        ) in energy_by_cell

        if artifact_status in QUARANTINED_ARTIFACT_STATUSES:
            gaps.append(
                _gap_row(
                    candidate,
                    artifact_status=artifact_status,
                    latency_status=latency_status,
                    energy_status="blocked",
                    next_action="resolve_quarantine",
                    gap_reason=f"artifact_status={artifact_status}",
                    tag=tag,
                )
            )
            continue
        if artifact is None or artifact_status not in READY_ARTIFACT_STATUSES:
            gaps.append(
                _gap_row(
                    candidate,
                    artifact_status=artifact_status,
                    latency_status=latency_status,
                    energy_status="blocked",
                    next_action="fix_missing_artifact",
                    gap_reason=f"artifact_status={artifact_status}",
                    tag=tag,
                )
            )
            continue
        missing_paths = _missing_artifact_paths(artifact)
        if missing_paths:
            gaps.append(
                _gap_row(
                    candidate,
                    artifact_status=artifact_status,
                    latency_status=latency_status,
                    energy_status="blocked",
                    next_action="fix_missing_artifact",
                    gap_reason=f"missing_artifact_paths={','.join(missing_paths)}",
                    tag=tag,
                )
            )
            continue

        priority_selected = _is_priority_selected(candidate, artifact)
        if energy_measured and tag not in RETEST_TAGS and not allow_repeat:
            gaps.append(
                _gap_row(
                    candidate,
                    artifact_status=artifact_status,
                    latency_status=latency_status,
                    energy_status="measured",
                    next_action="skip_existing_energy_measurement",
                    gap_reason="energy_cell_already_measured",
                    tag=tag,
                )
            )
            continue

        if latency_row is None and not priority_selected:
            gaps.append(
                _gap_row(
                    candidate,
                    artifact_status=artifact_status,
                    latency_status=latency_status,
                    energy_status="missing",
                    next_action="wait_for_latency_or_priority_selection",
                    gap_reason="latency_not_measured_and_not_priority_selected",
                    tag=tag,
                )
            )
            continue

        if energy_measured and tag in RETEST_TAGS:
            selection_reason = tag
        elif energy_measured and allow_repeat:
            selection_reason = "allow_repeat"
        elif priority_selected and latency_row is None:
            selection_reason = "priority_selected"
            latency_status = "priority_selected_without_latency"
        else:
            selection_reason = "latency_succeeded"

        jobs.append(
            _build_job(
                candidate,
                artifact,
                latency_row=latency_row,
                backend=backend,
                gpu_id=gpu_id,
                run_id_prefix=run_id_prefix,
                index=len(jobs),
                tag=tag,
                quality_policy=quality_policy,
                selection_reason=selection_reason,
                created_at=created_at,
                rows_out_jsonl=rows_out_jsonl,
                raw_root=raw_root,
                phase=phase,
                manifest_path=manifest_path,
                registry_path=registry_path,
                manifest_digest=manifest_digest,
                energy_warmup_iters=energy_warmup_iters,
                energy_measure_iters=energy_measure_iters,
            )
        )
        gaps.append(
            _gap_row(
                candidate,
                artifact_status=artifact_status,
                latency_status=latency_status,
                energy_status="queued",
                next_action="queue_energy",
                gap_reason=selection_reason,
                tag=tag,
            )
        )

    return jobs, gaps


def main() -> int:
    args = parse_args()
    jobs, gaps = generate_energy_jobs(
        candidates=read_jsonl(args.candidates),
        artifacts=read_jsonl(args.artifact_registry),
        latency_rows=read_jsonl(args.latency_rows),
        energy_rows=read_jsonl(args.energy_rows),
        backend=args.backend,
        gpu_id=args.gpu_id,
        run_id_prefix=args.run_id_prefix,
        tag=args.tag,
        allow_repeat=args.allow_repeat,
        quality_policy=args.quality_policy,
        rows_out_jsonl=args.rows_out_jsonl,
        raw_root=args.raw_root,
        phase=args.phase,
        manifest_path=args.manifest_path,
        registry_path=args.registry_path,
        manifest_digest=args.manifest_digest,
        energy_warmup_iters=args.energy_warmup_iters,
        energy_measure_iters=args.energy_measure_iters,
    )
    write_jsonl(args.out_queue, jobs)
    write_json(args.out_gap_json, gaps)
    write_csv(args.out_gap_csv, gaps)
    print(
        json.dumps(
            {
                "schema": "stage2_energy_coverage_job_summary_v1",
                "jobs": len(jobs),
                "gap_rows": len(gaps),
                "out_queue": str(args.out_queue),
                "out_gap_csv": str(args.out_gap_csv),
                "out_gap_json": str(args.out_gap_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
