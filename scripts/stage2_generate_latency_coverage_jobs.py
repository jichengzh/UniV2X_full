#!/usr/bin/env python3
"""Generate coverage-first Stage2 H800 latency job queues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    job_plan_row,
    validate_job_plan_row,
)


DEFAULT_GPUS = [0, 1, 4, 5]
SCHEDULE_POLICIES = ("default", "metaschedule_tuned")
RETEST_TAGS = {"paper_retest", "outlier_retest", "cross_gpu_drift_check"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_job_queue(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_job_plan_row(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_gpus(value: str | Iterable[int]) -> list[int]:
    if not isinstance(value, str):
        gpus = [int(item) for item in value]
    else:
        gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("at least one GPU must be specified")
    return gpus


def _width_tuple(row: dict[str, Any]) -> tuple[int, int, int] | None:
    width = row.get("width")
    if isinstance(width, list) and len(width) == 3:
        try:
            return (int(width[0]), int(width[1]), int(width[2]))
        except (TypeError, ValueError):
            return None
    return None


def _safe(value: object) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace(":", "-")
        .replace(" ", "_")
        .replace("+", "plus")
    )


def _width_csv(candidate: dict[str, Any]) -> str:
    width = _width_tuple(candidate)
    if width is None:
        raise ValueError(f"candidate has invalid width: {candidate.get('candidate_id')}")
    return ",".join(str(item) for item in width)


def _config_id_for_width(width: tuple[int, int, int], schedule_policy: str) -> str:
    return (
        f"coverage_h800_tvm_pyramid_w{width[0]}x{width[1]}x{width[2]}"
        f"_fp16_{schedule_policy}"
    )


def _config_id(candidate: dict[str, Any], schedule_policy: str) -> str:
    width = _width_tuple(candidate)
    if width is None:
        raise ValueError(f"candidate has invalid width: {candidate.get('candidate_id')}")
    key = "config_id_default" if schedule_policy == "default" else "config_id_tuned"
    return str(candidate.get(key) or _config_id_for_width(width, schedule_policy))


def _software_point_id(candidate: dict[str, Any]) -> str:
    value = candidate.get("software_point_id")
    if value:
        return str(value)
    width = _width_tuple(candidate)
    if width is None:
        raise ValueError(f"candidate has invalid width: {candidate.get('candidate_id')}")
    return f"pyramid_lidar:backbone:w{width[0]}x{width[1]}x{width[2]}:fp16"


def _normalize_schedule(value: object) -> str:
    text = str(value or "")
    return "metaschedule_tuned" if text == "tuned" else text


def _candidate_config_ids(candidate: dict[str, Any]) -> set[str]:
    values = {
        _config_id(candidate, "default"),
        _config_id(candidate, "metaschedule_tuned"),
    }
    for item in candidate.get("config_ids", []) or []:
        if item:
            values.add(str(item))
    return values


def _artifact_matches_candidate(
    artifact: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    candidate_id = candidate.get("candidate_id")
    if candidate_id and artifact.get("candidate_id") == candidate_id:
        return True
    artifact_config = artifact.get("config_id")
    if artifact_config and str(artifact_config) in _candidate_config_ids(candidate):
        return True
    if artifact.get("label") == candidate.get("label"):
        artifact_width = _width_tuple(artifact)
        candidate_width = _width_tuple(candidate)
        if artifact_width is None or artifact_width == candidate_width:
            return True
    artifact_width = _width_tuple(artifact)
    candidate_width = _width_tuple(candidate)
    return artifact_width is not None and artifact_width == candidate_width


def _artifact_for_candidate(
    candidate: dict[str, Any],
    artifact_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for artifact in artifact_rows:
        if _artifact_matches_candidate(artifact, candidate):
            return artifact
    if candidate.get("artifact_status"):
        return candidate
    return None


def _artifact_value(
    candidate: dict[str, Any],
    artifact: dict[str, Any],
    keys: tuple[str, ...],
) -> str:
    for row in (artifact, candidate):
        for key in keys:
            value = row.get(key)
            if value:
                return str(value)
    return ""


def _ready_artifact(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> bool:
    if artifact is None:
        return False
    if artifact.get("artifact_status") != "ready":
        return False
    onnx_path = _artifact_value(candidate, artifact, ("onnx_path", "onnx"))
    work_dir = _artifact_value(
        candidate,
        artifact,
        ("tvm_work_dir", "work_dir", "database_path"),
    )
    return bool(onnx_path and work_dir)


def _measured_latency_cells(
    rows: list[dict[str, Any]],
) -> tuple[set[str], set[tuple[tuple[int, int, int], str, str, str]]]:
    config_ids: set[str] = set()
    cells: set[tuple[tuple[int, int, int], str, str, str]] = set()
    for row in rows:
        if row.get("measurement_status") != "measured":
            continue
        config_id = row.get("config_id")
        if config_id:
            config_ids.add(str(config_id))
        width = _width_tuple(row)
        schedule = _normalize_schedule(row.get("schedule_policy"))
        if width is None or not schedule:
            continue
        cells.add(
            (
                width,
                str(row.get("quant_policy") or "fp16"),
                schedule,
                str(row.get("optimized_scope") or "backbone_only"),
            )
        )
    return config_ids, cells


def _already_measured(
    *,
    candidate: dict[str, Any],
    schedule_policy: str,
    measured_config_ids: set[str],
    measured_cells: set[tuple[tuple[int, int, int], str, str, str]],
    tag: str,
) -> bool:
    if tag in RETEST_TAGS:
        return False
    width = _width_tuple(candidate)
    if width is None:
        return True
    config_id = _config_id(candidate, schedule_policy)
    if config_id in measured_config_ids:
        return True
    cell = (
        width,
        str(candidate.get("quant_policy") or "fp16"),
        schedule_policy,
        str(candidate.get("optimized_scope") or "backbone_only"),
    )
    return cell in measured_cells


def _measurement_command(
    *,
    candidate: dict[str, Any],
    artifact: dict[str, Any],
    schedule_policy: str,
    gpu: int,
    raw_root: str,
    run_id: str,
) -> list[str]:
    onnx_path = _artifact_value(candidate, artifact, ("onnx_path", "onnx"))
    onnx = Path(onnx_path)
    return [
        "python3",
        "scripts/stage2_h800_b1_latency_command.py",
        "--label",
        str(candidate["label"]),
        "--onnx-file",
        onnx.name,
        "--out-json",
        str(Path(raw_root) / run_id / "b1_latency_result.json"),
        "--gpu",
        str(gpu),
        "--schedule-policy",
        schedule_policy,
        "--models-dir",
        str(onnx.parent),
    ]


def _generator_command(
    *,
    candidate: dict[str, Any],
    artifact: dict[str, Any],
    schedule_policy: str,
    gpu: int,
    phase: str,
    tag: str,
    rows_out_jsonl: str,
    raw_root: str,
    run_id: str,
    job_id: str,
    manifest_digest: str,
    warmup_iters: int,
    measure_iters: int,
    repeat: int,
) -> list[str]:
    measurement_command = _measurement_command(
        candidate=candidate,
        artifact=artifact,
        schedule_policy=schedule_policy,
        gpu=gpu,
        raw_root=raw_root,
        run_id=run_id,
    )
    return [
        "python3",
        "scripts/stage2_generate_latency_lut.py",
        "--job-id",
        job_id,
        "--model",
        str(candidate.get("model") or "Pyramid-LiDAR"),
        "--config-id",
        _config_id(candidate, schedule_policy),
        "--candidate-id",
        str(candidate["candidate_id"]),
        "--software-point-id",
        _software_point_id(candidate),
        "--dense-stage",
        str(candidate.get("dense_stage") or "backbone"),
        "--optimized-scope",
        str(candidate.get("optimized_scope") or "backbone_only"),
        "--width",
        _width_csv(candidate),
        "--quant-policy",
        str(candidate.get("quant_policy") or "fp16"),
        "--schedule-policy",
        schedule_policy,
        "--backend",
        "h800_tvm",
        "--manifest-digest",
        manifest_digest or phase,
        "--run-id",
        run_id,
        "--warmup-iters",
        str(warmup_iters),
        "--measure-iters",
        str(measure_iters),
        "--repeat",
        str(repeat),
        "--measurement-command-json",
        json.dumps(measurement_command),
        "--out-jsonl",
        rows_out_jsonl,
    ]


def _job_row(
    *,
    candidate: dict[str, Any],
    artifact: dict[str, Any],
    schedule_policy: str,
    gpu: int,
    phase: str,
    tag: str,
    rows_out_jsonl: str,
    raw_root: str,
    manifest_path: str,
    registry_path: str,
    manifest_digest: str,
    created_at: str,
    warmup_iters: int,
    measure_iters: int,
    repeat: int,
) -> dict[str, Any]:
    label = str(candidate["label"])
    run_id = f"{_safe(phase)}_{_safe(tag)}_{_safe(label)}_{schedule_policy}_gpu{gpu}"
    job_id = f"latency_gpu{gpu}:{_safe(tag)}:{_safe(label)}:{schedule_policy}"
    command = _generator_command(
        candidate=candidate,
        artifact=artifact,
        schedule_policy=schedule_policy,
        gpu=gpu,
        phase=phase,
        tag=tag,
        rows_out_jsonl=rows_out_jsonl,
        raw_root=raw_root,
        run_id=run_id,
        job_id=job_id,
        manifest_digest=manifest_digest,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        repeat=repeat,
    )
    row = job_plan_row(
        job_id=job_id,
        model=str(candidate.get("model") or "Pyramid-LiDAR"),
        lut_kind="latency",
        job_type="generate_latency_lut",
        priority=int(candidate.get("priority") or 50),
        config_id=_config_id(candidate, schedule_policy),
        manifest_path=manifest_path,
        registry_path=registry_path,
        candidate_id=str(candidate["candidate_id"]),
        software_point_id=_software_point_id(candidate),
        expected_output=rows_out_jsonl,
        command=command,
        max_attempts=1,
        timeout_s=3600,
        resource={
            "server": "h800_remote",
            "backend": "h800_tvm",
            "gpu": int(gpu),
            "exclusive": True,
            "serial_queue": True,
            "tag": tag,
        },
        created_at=created_at,
    )
    row.update(
        {
            "label": label,
            "schedule_policy": schedule_policy,
            "tag": tag,
            "coverage_policy": "coverage_first",
            "artifact_status": "ready",
            "onnx_path": _artifact_value(candidate, artifact, ("onnx_path", "onnx")),
            "tvm_work_dir": _artifact_value(
                candidate,
                artifact,
                ("tvm_work_dir", "work_dir", "database_path"),
            ),
            "run_id": run_id,
        }
    )
    validate_job_plan_row(row)
    return row


def _sorted_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (-int(row.get("priority") or 0), str(row.get("label") or "")),
    )


def generate_latency_coverage_jobs(
    *,
    candidates: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    gpus: list[int] | tuple[int, ...] = tuple(DEFAULT_GPUS),
    created_at: str,
    phase: str,
    tag: str,
    rows_out_jsonl: str,
    raw_root: str,
    manifest_path: str,
    registry_path: str,
    manifest_digest: str = "coverage_pipeline_v1",
    warmup_iters: int = 1,
    measure_iters: int = 500,
    repeat: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    gpu_ids = parse_gpus(gpus)
    jobs_by_gpu: dict[str, list[dict[str, Any]]] = {str(gpu): [] for gpu in gpu_ids}
    measured_config_ids, measured_cells = _measured_latency_cells(latency_rows)
    emitted_widths: set[tuple[int, int, int]] = set()
    next_gpu_index = 0

    for candidate in _sorted_candidates(candidates):
        width = _width_tuple(candidate)
        if width is None or width in emitted_widths:
            continue
        artifact = _artifact_for_candidate(candidate, artifact_rows)
        if not _ready_artifact(candidate, artifact):
            continue
        assert artifact is not None
        emitted_any = False
        for schedule_policy in SCHEDULE_POLICIES:
            if _already_measured(
                candidate=candidate,
                schedule_policy=schedule_policy,
                measured_config_ids=measured_config_ids,
                measured_cells=measured_cells,
                tag=tag,
            ):
                continue
            gpu = gpu_ids[next_gpu_index % len(gpu_ids)]
            next_gpu_index += 1
            job = _job_row(
                candidate=candidate,
                artifact=artifact,
                schedule_policy=schedule_policy,
                gpu=gpu,
                phase=phase,
                tag=tag,
                rows_out_jsonl=rows_out_jsonl,
                raw_root=raw_root,
                manifest_path=manifest_path,
                registry_path=registry_path,
                manifest_digest=manifest_digest,
                created_at=created_at,
                warmup_iters=warmup_iters,
                measure_iters=measure_iters,
                repeat=repeat,
            )
            jobs_by_gpu[str(gpu)].append(job)
            emitted_any = True
        if emitted_any:
            emitted_widths.add(width)
    return jobs_by_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--artifact-state", action="append", default=[])
    parser.add_argument("--artifact-registry", action="append", default=[])
    parser.add_argument("--latency-rows", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows-out-jsonl", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--gpus", default="0,1,4,5")
    parser.add_argument("--phase", default="coverage_pipeline_v1")
    parser.add_argument("--tag", default="coverage")
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--manifest-path", default="manifest_placeholder.json")
    parser.add_argument("--registry-path", default="artifacts/artifact_registry_v1.jsonl")
    parser.add_argument("--manifest-digest", default="coverage_pipeline_v1")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=500)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    artifact_rows: list[dict[str, Any]] = []
    for path in [*args.artifact_state, *args.artifact_registry]:
        artifact_rows.extend(read_jsonl(path))
    latency_rows: list[dict[str, Any]] = []
    for path in args.latency_rows:
        latency_rows.extend(read_jsonl(path))
    jobs_by_gpu = generate_latency_coverage_jobs(
        candidates=candidates,
        artifact_rows=artifact_rows,
        latency_rows=latency_rows,
        gpus=parse_gpus(args.gpus),
        created_at=args.created_at,
        phase=args.phase,
        tag=args.tag,
        rows_out_jsonl=args.rows_out_jsonl,
        raw_root=args.raw_root,
        manifest_path=args.manifest_path,
        registry_path=args.registry_path,
        manifest_digest=args.manifest_digest,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        repeat=args.repeat,
    )
    out_dir = Path(args.out_dir)
    for gpu, jobs in jobs_by_gpu.items():
        write_job_queue(out_dir / f"latency_job_queue_gpu{gpu}.jsonl", jobs)
    counts = {gpu: len(jobs) for gpu, jobs in jobs_by_gpu.items()}
    print(
        json.dumps(
            {
                "schema": "stage2_latency_coverage_job_generation_summary_v1",
                "out_dir": str(out_dir),
                "jobs": sum(counts.values()),
                "per_gpu_counts": counts,
                "tag": args.tag,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
