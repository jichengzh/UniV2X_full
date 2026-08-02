#!/usr/bin/env python3
"""Build Stage3 Gold96 H800 performance batch plans from a v3 manifest."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage3_gold96_performance_batch_plan_v3"
SOURCE_MANIFEST_SCHEMA = "stage3_gold_coldstart96_manifest_v3"
RUNNER_ORDER = ("tvm_fp16", "tvm_int8", "trt_fp16", "trt_int8")
BATCH_GROUP_COUNTS = (4, 4, 4, 4, 3, 3)
PYRAMID_ONNX_ROOT = Path("/home/jichengzhi/V2X/results/stage3_gold96_v3_20260711/pyramid_sources")
PYRAMID_CALIB_ROOT = Path("/home/jichengzhi/V2X/results/stage3_gold96_v3_20260711/pyramid_calibration")
CODRIVING_ROOT = Path("/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709")
VALID_STATE_STATUSES = {"pending", "running", "failed", "success", "confirmed_failure", "measured"}


def _require_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(f"expected {SOURCE_MANIFEST_SCHEMA} manifest")
    jobs = manifest.get("jobs")
    pilot_group_ids = manifest.get("pilot_group_ids")
    if not isinstance(jobs, list):
        raise ValueError("manifest.jobs must be a list")
    if not isinstance(pilot_group_ids, list) or len(pilot_group_ids) != 2:
        raise ValueError("manifest.pilot_group_ids must be a two-item list")


def _width_key(row: Mapping[str, Any]) -> str:
    if row.get("width_key"):
        return str(row["width_key"])
    width = row.get("width")
    if not isinstance(width, Sequence) or len(width) != 3:
        raise ValueError(f"row {row.get('job_id')} missing width_key/width")
    return "x".join(str(int(value)) for value in width)


def _width_csv(width_key: str) -> str:
    return width_key.replace("x", ",")


def _padded_width(width_key: str) -> str:
    return "x".join(f"{int(value):03d}" for value in width_key.split("x"))


def _non_pilot_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    pilot_group_ids = {str(group_id) for group_id in manifest["pilot_group_ids"]}
    rows = [copy.deepcopy(dict(row)) for row in manifest["jobs"] if str(row.get("group_id")) not in pilot_group_ids]
    rows.sort(
        key=lambda row: (
            str(row.get("group_id") or ""),
            str(row.get("q_mode") or ""),
            str(row.get("capability_profile_id") or ""),
        )
    )
    if len(rows) != 88:
        raise ValueError(f"expected 88 non-pilot rows, got {len(rows)}")
    return rows


def _group_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    group_ids = sorted({str(row["group_id"]) for row in rows})
    if len(group_ids) != 22:
        raise ValueError(f"expected 22 non-pilot groups, got {len(group_ids)}")
    return group_ids


def _runner_key(row: Mapping[str, Any]) -> str:
    dispatch_key = str(row.get("dispatch_key") or "")
    q_mode = str(row.get("q_mode") or "")
    mapping = {
        ("tvm_auto", "fp16"): "tvm_fp16",
        ("tvm_auto", "int8"): "tvm_int8",
        ("trt_engine", "fp16"): "trt_fp16",
        ("trt_engine", "int8"): "trt_int8",
    }
    runner_key = mapping.get((dispatch_key, q_mode))
    if runner_key is None:
        raise ValueError(f"unsupported dispatch/q combination: {dispatch_key}/{q_mode}")
    return runner_key


def _runner_rank(row: Mapping[str, Any]) -> int:
    return RUNNER_ORDER.index(_runner_key(row))


def _onnx_and_calibration(row: Mapping[str, Any]) -> tuple[str, str]:
    model = str(row.get("model") or "")
    width_key = _width_key(row)
    if model == "pyramid":
        padded = _padded_width(width_key)
        onnx_path = PYRAMID_ONNX_ROOT / padded / f"pyramid_{padded}_multiscale.onnx"
        calibration_root = PYRAMID_CALIB_ROOT / padded
        return str(onnx_path), str(calibration_root)
    if model == "codriving":
        onnx_path = CODRIVING_ROOT / width_key / f"resnet_multiscale_{width_key}_final_fp32.onnx"
        calibration_root = CODRIVING_ROOT / width_key / "calibration_source"
        return str(onnx_path), str(calibration_root)
    raise ValueError(f"unsupported model: {model}")


def _job_output_dir(remote_artifact_root: str | Path, batch_index: int, row: Mapping[str, Any], runner_key: str) -> Path:
    return Path(remote_artifact_root) / f"batch_{batch_index:02d}" / str(row["group_id"]) / runner_key


def _build_command(
    *,
    row: Mapping[str, Any],
    batch_index: int,
    runner_key: str,
    onnx_path: str,
    calibration_root: str,
    assigned_gpu: int,
    remote_artifact_root: str | Path,
) -> list[str]:
    width_key = _width_key(row)
    width_csv = _width_csv(width_key)
    label = f"{row['model']}_{width_key}"
    out_dir = _job_output_dir(remote_artifact_root, batch_index, row, runner_key)
    if runner_key == "tvm_fp16":
        return [
            "python3",
            "scripts/stage2_route_b_fp16_auto_runner.py",
            "--phase",
            "run",
            "--label",
            label,
            "--width",
            width_csv,
            "--onnx",
            onnx_path,
            "--out-dir",
            str(out_dir),
            "--gpu",
            str(assigned_gpu),
            "--fix",
            "none",
            "--max-trials",
            "64",
            "--measure-energy",
            "--energy-iters",
            "300",
        ]
    if runner_key == "tvm_int8":
        return [
            "python3",
            "scripts/stage2_route_b_int8_auto_decomp.py",
            "--label",
            label,
            "--width",
            width_csv,
            "--onnx",
            onnx_path,
            "--out-dir",
            str(out_dir),
            "--gpu",
            str(assigned_gpu),
            "--measure-energy",
            "--energy-iters",
            "300",
            "--route-b-fp16-ms",
            "0.0",
        ]
    command = [
        "python3",
        "framework/trt_baseline/trt_profile_v1.py",
        "--onnx",
        onnx_path,
        "--precision",
        "fp16" if runner_key == "trt_fp16" else "int8",
        "--gpu",
        str(assigned_gpu),
        "--warmup",
        "20",
        "--iters",
        "300",
        "--repeat",
        "5",
        "--energy-secs",
        "5.0",
        "--artifact-dir",
        str(out_dir / "artifacts"),
        "--out",
        str(out_dir / "trt_profile_result.json"),
    ]
    if runner_key == "trt_int8":
        command.extend(["--calib-dir", calibration_root])
    return command


def _build_job(
    row: Mapping[str, Any],
    *,
    batch_index: int,
    row_index: int,
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    runner_key = _runner_key(row)
    onnx_path, calibration_root = _onnx_and_calibration(row)
    assigned_gpu = int(gpus[row_index % len(gpus)])
    command = _build_command(
        row=row,
        batch_index=batch_index,
        runner_key=runner_key,
        onnx_path=onnx_path,
        calibration_root=calibration_root,
        assigned_gpu=assigned_gpu,
        remote_artifact_root=remote_artifact_root,
    )
    return {
        "schema_version": "stage3_gold96_performance_job_v3",
        "job_id": f"{row['group_id']}|{runner_key}",
        "manifest_job_id": str(row["job_id"]),
        "group_id": str(row["group_id"]),
        "model": str(row["model"]),
        "width_key": _width_key(row),
        "q_mode": str(row["q_mode"]),
        "runner_key": runner_key,
        "dispatch_key": str(row["dispatch_key"]),
        "onnx_path": onnx_path,
        "calibration_root": calibration_root,
        "command": command,
        "assigned_gpu": assigned_gpu,
        "gpu_pool": ",".join(str(int(gpu)) for gpu in gpus),
        "remote_artifact_root": str(remote_artifact_root),
        "expected_result_json": str(_job_output_dir(remote_artifact_root, batch_index, row, runner_key) / "result.json"),
        "max_attempts": 2,
        "terminal_status": "pending",
    }


def _batch_slices(group_ids: Sequence[str]) -> list[list[str]]:
    slices: list[list[str]] = []
    cursor = 0
    for count in BATCH_GROUP_COUNTS:
        slices.append(list(group_ids[cursor : cursor + count]))
        cursor += count
    if cursor != len(group_ids):
        raise ValueError("batch partition did not consume all group ids")
    return slices


def build_all_batches(
    manifest: Mapping[str, Any],
    *,
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> list[dict[str, Any]]:
    _require_manifest(manifest)
    if not gpus:
        raise ValueError("gpus must not be empty")
    rows = _non_pilot_rows(manifest)
    group_ids = _group_ids(rows)
    groups = _batch_slices(group_ids)
    batches: list[dict[str, Any]] = []
    for batch_index, batch_group_ids in enumerate(groups, start=1):
        batch_rows = [copy.deepcopy(row) for row in rows if str(row["group_id"]) in set(batch_group_ids)]
        batch_rows.sort(key=lambda row: (batch_group_ids.index(str(row["group_id"])), _runner_rank(row)))
        jobs = [
            _build_job(
                row,
                batch_index=batch_index,
                row_index=row_index,
                remote_artifact_root=remote_artifact_root,
                gpus=gpus,
            )
            for row_index, row in enumerate(batch_rows)
        ]
        batches.append(
            {
                "schema_version": SCHEMA_VERSION,
                "batch_index": batch_index,
                "group_count": len(batch_group_ids),
                "group_ids": list(batch_group_ids),
                "manifest_row_count": len(batch_rows),
                "manifest_rows": batch_rows,
                "jobs": jobs,
                "remote_artifact_root": str(remote_artifact_root),
                "gpu_pool": ",".join(str(int(gpu)) for gpu in gpus),
            }
        )
    return batches


def build_batch_plan(
    manifest: Mapping[str, Any],
    *,
    batch_index: int,
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    batches = build_all_batches(manifest, remote_artifact_root=remote_artifact_root, gpus=gpus)
    if batch_index < 1 or batch_index > len(batches):
        raise ValueError("batch_index must be in 1..6")
    return copy.deepcopy(batches[batch_index - 1])


def aggregate_local_state(
    jobs: Sequence[Mapping[str, Any]],
    *,
    state_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_job: dict[str, dict[str, Any]] = {}
    ready_job_ids: list[str] = []
    success_job_ids: list[str] = []
    confirmed_failure_job_ids: list[str] = []
    for job in jobs:
        job_id = str(job["job_id"])
        relevant = [dict(row) for row in state_rows if str(row.get("job_id")) == job_id]
        failed_attempts = sorted(
            int(row.get("attempt", 0))
            for row in relevant
            if str(row.get("status")) == "failed"
        )
        success_attempts = sorted(
            int(row.get("attempt", 0))
            for row in relevant
            if str(row.get("status")) == "success"
        )
        latest_status = "pending"
        if relevant:
            last = relevant[-1]
            status_text = str(last.get("status") or "pending")
            latest_status = status_text if status_text in VALID_STATE_STATUSES else "pending"
        if success_attempts:
            terminal_status = "success"
        elif len(failed_attempts) >= int(job.get("max_attempts", 2)):
            terminal_status = "confirmed_failure"
        else:
            terminal_status = "pending"
        summary = {
            "job_id": job_id,
            "attempts_used": len(failed_attempts) + (1 if success_attempts else 0),
            "failed_attempts": failed_attempts,
            "success_attempts": success_attempts,
            "latest_status": latest_status,
            "terminal_status": terminal_status,
        }
        by_job[job_id] = summary
        if terminal_status == "pending":
            ready_job_ids.append(job_id)
        elif terminal_status == "success":
            success_job_ids.append(job_id)
        else:
            confirmed_failure_job_ids.append(job_id)
    return {
        "schema_version": "stage3_gold96_performance_batch_state_v3",
        "job_count": len(jobs),
        "ready_job_ids": ready_job_ids,
        "success_job_ids": success_job_ids,
        "confirmed_failure_job_ids": confirmed_failure_job_ids,
        "by_job": by_job,
    }


def read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    item = Path(path)
    if not item.exists():
        return []
    return [json.loads(line) for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return out


def parse_gpus(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not gpus:
        raise ValueError("at least one GPU must be specified")
    return gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--remote-artifact-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--state-jsonl", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    batch = build_batch_plan(
        manifest,
        batch_index=int(args.batch_index),
        remote_artifact_root=args.remote_artifact_root,
        gpus=parse_gpus(args.gpus),
    )
    state = aggregate_local_state(batch["jobs"], state_rows=read_jsonl(args.state_jsonl))
    stem = f"stage3_gold96_performance_batch_{int(args.batch_index):02d}"
    plan_json = write_json(args.output_dir / f"{stem}_plan.json", batch)
    jobs_jsonl = write_jsonl(args.output_dir / f"{stem}_jobs.jsonl", batch["jobs"])
    aggregate_json = write_json(args.output_dir / f"{stem}_state.json", state)
    launch_ready_jobs = [job for job in batch["jobs"] if str(job["job_id"]) in set(state["ready_job_ids"])]
    launch_jsonl = write_jsonl(args.output_dir / f"{stem}_launch_ready.jsonl", launch_ready_jobs)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "batch_index": int(args.batch_index),
                "dry_run": bool(args.dry_run or not args.execute),
                "plan_json": str(plan_json),
                "jobs_jsonl": str(jobs_jsonl),
                "state_json": str(aggregate_json),
                "launch_ready_jsonl": str(launch_jsonl),
                "ready_jobs": len(launch_ready_jobs),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
