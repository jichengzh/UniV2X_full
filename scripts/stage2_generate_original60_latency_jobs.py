#!/usr/bin/env python3
"""Generate direct H800 latency jobs for original60 artifact-ready candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import job_plan_row, validate_job_plan_row  # noqa: E402


DEFAULT_GPUS = [0, 1, 4, 5]


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
            validate_job_plan_row(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_gpus(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not gpus:
        raise ValueError("at least one GPU must be specified")
    return gpus


def _safe(value: object) -> str:
    return str(value).replace("/", "-").replace(":", "-").replace(" ", "_")


def _width(row: dict[str, Any]) -> list[int]:
    width = row.get("width")
    if not isinstance(width, list) or len(width) != 3:
        raise ValueError(f"invalid width for {row.get('candidate_id')}")
    return [int(item) for item in width]


def _width_csv(row: dict[str, Any]) -> str:
    return ",".join(str(item) for item in _width(row))


def _label(row: dict[str, Any]) -> str:
    return str(row.get("label") or _safe(row.get("candidate_id") or "unknown"))


def _software_point_id(row: dict[str, Any]) -> str:
    if row.get("software_point_id"):
        return str(row["software_point_id"])
    width = _width(row)
    return f"pyramid_lidar:backbone:w{width[0]}x{width[1]}x{width[2]}:fp16"


def _config_id(row: dict[str, Any], schedule: str) -> str:
    if schedule == "default" and row.get("config_id_default"):
        return str(row["config_id_default"])
    if schedule == "metaschedule_tuned" and row.get("config_id_tuned"):
        return str(row["config_id_tuned"])
    width = _width(row)
    return f"coverage_h800_tvm_pyramid_w{width[0]}x{width[1]}x{width[2]}_fp16_{schedule}"


def _measured_index(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    measured: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("measurement_status") != "measured":
            continue
        candidate_id = row.get("candidate_id")
        schedule = row.get("schedule_policy")
        if candidate_id and schedule:
            measured.add((str(candidate_id), str(schedule)))
    return measured


def _artifact_by_candidate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if candidate_id:
            out[str(candidate_id)] = row
    return out


def _ready(artifact: dict[str, Any] | None) -> bool:
    if artifact is None:
        return False
    if artifact.get("artifact_status") != "ready":
        return False
    required = (
        "onnx_path",
        "tvm_work_dir",
        "database_workload_path",
        "database_tuning_record_path",
    )
    return all(artifact.get(field) for field in required)


def build_jobs(
    *,
    candidates: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    gpus: list[int],
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
) -> dict[int, list[dict[str, Any]]]:
    measured = _measured_index(latency_rows)
    artifacts = _artifact_by_candidate(artifact_rows)
    jobs_by_gpu: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    next_gpu = 0

    for candidate in sorted(candidates, key=lambda row: (-int(row.get("priority") or 0), _label(row))):
        candidate_id = str(candidate["candidate_id"])
        if (candidate_id, "default") in measured and (candidate_id, "metaschedule_tuned") in measured:
            continue
        artifact = artifacts.get(candidate_id)
        if not _ready(artifact):
            continue
        assert artifact is not None
        gpu = gpus[next_gpu % len(gpus)]
        next_gpu += 1
        label = _label(candidate)
        run_id = f"{_safe(phase)}_{_safe(tag)}_{_safe(label)}_gpu{gpu}"
        job_id = f"original60_latency_gpu{gpu}:{_safe(tag)}:{_safe(label)}"
        command = [
            "python3",
            "scripts/stage2_h800_run_measurement_job.py",
            "--kind",
            "latency",
            "--model",
            str(candidate.get("model") or "Pyramid-LiDAR"),
            "--label",
            label,
            "--phase",
            phase,
            "--gpu",
            str(gpu),
            "--onnx",
            str(artifact["onnx_path"]),
            "--work-dir",
            str(artifact["tvm_work_dir"]),
            "--width",
            _width_csv(candidate),
            "--candidate-id",
            candidate_id,
            "--software-point-id",
            _software_point_id(candidate),
            "--config-id-tuned",
            _config_id(candidate, "metaschedule_tuned"),
            "--config-id-default",
            _config_id(candidate, "default"),
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
            "--warmup-iters",
            str(warmup_iters),
            "--measure-iters",
            str(measure_iters),
            "--repeat",
            str(repeat),
        ]
        row = job_plan_row(
            job_id=job_id,
            model=str(candidate.get("model") or "Pyramid-LiDAR"),
            lut_kind="latency",
            job_type="generate_latency_lut",
            priority=int(candidate.get("priority") or 50),
            config_id=_config_id(candidate, "metaschedule_tuned"),
            manifest_path=manifest_path,
            registry_path=registry_path,
            candidate_id=candidate_id,
            software_point_id=_software_point_id(candidate),
            expected_output=rows_out_jsonl,
            command=command,
            max_attempts=1,
            timeout_s=3600,
            resource={
                "server": "h800_remote",
                "backend": "h800_tvm",
                "gpu": gpu,
                "exclusive": True,
                "serial_queue": True,
                "tag": tag,
                "artifact_registry": "original60",
            },
            created_at=created_at,
        )
        row.update(
            {
                "label": label,
                "width": _width(candidate),
                "tag": tag,
                "coverage_policy": "original60_coverage_first",
                "artifact_status": "ready",
                "onnx_path": artifact["onnx_path"],
                "tvm_work_dir": artifact["tvm_work_dir"],
                "database_workload_path": artifact["database_workload_path"],
                "database_tuning_record_path": artifact["database_tuning_record_path"],
                "run_id": run_id,
                "emits_schedule_policies": ["default", "metaschedule_tuned"],
            }
        )
        jobs_by_gpu[gpu].append(row)
    return jobs_by_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--artifact-registry", required=True)
    parser.add_argument("--latency-rows", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows-out-jsonl", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--gpus", default="0,1,4,5")
    parser.add_argument("--phase", default="original60_measured_v1")
    parser.add_argument("--tag", default="original60")
    parser.add_argument("--manifest-path", default="candidates/candidate_queue.jsonl")
    parser.add_argument("--registry-path", default="artifacts/artifact_registry_original60_v1.jsonl")
    parser.add_argument("--manifest-digest", default="original60_measured_v1")
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--warmup-iters", type=int, default=30)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    latency_rows: list[dict[str, Any]] = []
    for path in args.latency_rows:
        latency_rows.extend(read_jsonl(path))
    jobs = build_jobs(
        candidates=read_jsonl(args.candidates),
        artifact_rows=read_jsonl(args.artifact_registry),
        latency_rows=latency_rows,
        gpus=parse_gpus(args.gpus),
        phase=args.phase,
        tag=args.tag,
        rows_out_jsonl=args.rows_out_jsonl,
        raw_root=args.raw_root,
        manifest_path=args.manifest_path,
        registry_path=args.registry_path,
        manifest_digest=args.manifest_digest,
        created_at=args.created_at,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        repeat=args.repeat,
    )
    out_dir = Path(args.out_dir)
    for gpu, rows in jobs.items():
        write_jsonl(out_dir / f"latency_job_queue_original60_gpu{gpu}.jsonl", rows)
    counts = {str(gpu): len(rows) for gpu, rows in jobs.items()}
    print(
        json.dumps(
            {
                "schema": "stage2_original60_latency_job_generation_summary_v1",
                "jobs": sum(counts.values()),
                "per_gpu_counts": counts,
                "out_dir": str(out_dir),
                "rows_out_jsonl": args.rows_out_jsonl,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
