#!/usr/bin/env python3
"""Run Original60 FP32 AP smoke/full-val backfill jobs on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--queue", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gpu-ids", default="3,4,5,6")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def measured_labels(rows_out: Path) -> set[str]:
    labels: set[str] = set()
    for row in read_jsonl(rows_out):
        if str(row.get("precision") or row.get("quant_policy") or "") != "fp32":
            continue
        if str(row.get("measurement_status") or "") != "measured":
            continue
        if row.get("metric_value") is None:
            continue
        label = str(row.get("label") or "")
        if label:
            labels.add(label)
    return labels


def build_command(job: dict[str, Any], *, gpu_id: int, args: argparse.Namespace) -> list[str]:
    runner = ROOT / "scripts/stage2_h800_true_fp32_ap_eval.py"
    command = [
        args.env_python,
        str(runner),
        "--label",
        str(job["label"]),
        "--width",
        str(job["width_csv"]),
        "--ckpt-dir",
        str(job["ckpt_dir"]),
        "--raw-dir",
        str(job["raw_dir"]),
        "--rows-out",
        str(job["rows_out"]),
        "--env-python",
        args.env_python,
        "--heal-root",
        args.heal_root,
        "--gpu-id",
        str(gpu_id),
        "--num-samples",
        str(job["num_samples"]),
        "--run-id",
        f"true_fp32_original60_ap_{job['label']}_{args.mode}_gpu{gpu_id}_{time.strftime('%Y%m%d_%H%M%S')}",
        "--execute",
    ]
    if args.mode == "smoke":
        command.append("--report-only")
    return command


def run_one(job: dict[str, Any], *, gpu_id: int, args: argparse.Namespace, control_dir: Path) -> dict[str, Any]:
    label = str(job["label"])
    lane_dir = control_dir / f"gpu{gpu_id}" / label
    lane_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(job, gpu_id=gpu_id, args=args)
    write_json(
        lane_dir / "command.json",
        {
            "schema": "stage2_fp32_ap_bulk_worker_command_v1",
            "label": label,
            "gpu_id": gpu_id,
            "mode": args.mode,
            "command": command,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    if args.dry_run:
        return {"label": label, "gpu_id": gpu_id, "status": "dry_run", "rc": None}

    with (lane_dir / "stdout.txt").open("w", encoding="utf-8") as stdout, (
        lane_dir / "stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + str(args.heal_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    result = {
        "label": label,
        "gpu_id": gpu_id,
        "status": "ok" if proc.returncode == 0 else "failed",
        "rc": proc.returncode,
        "raw_dir": str(job["raw_dir"]),
        "control_dir": str(lane_dir),
    }
    write_json(lane_dir / "result.json", result)
    return result


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    queue = (
        Path(args.queue)
        if args.queue
        else output_root / ("jobs/fp32_ap_smoke_queue_v1.jsonl" if args.mode == "smoke" else "jobs/fp32_ap_full_queue_v1.jsonl")
    )
    gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("no GPU ids provided")
    jobs = read_jsonl(queue)
    if args.label:
        allowed = set(args.label)
        jobs = [job for job in jobs if str(job.get("label")) in allowed]
    if args.mode == "full":
        done = measured_labels(output_root / "rows/fp32_true_original60_ap_rows_v1.jsonl")
        jobs = [job for job in jobs if str(job.get("label")) not in done]
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    control_dir = output_root / "control/fp32_ap_backfill" / f"{args.mode}_{run_stamp}"
    control_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        control_dir / "worker_manifest.json",
        {
            "schema": "stage2_fp32_ap_bulk_worker_manifest_v1",
            "mode": args.mode,
            "queue": str(queue),
            "gpu_ids": gpu_ids,
            "job_count": len(jobs),
            "dry_run": args.dry_run,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {}
        for index, job in enumerate(jobs):
            gpu_id = gpu_ids[index % len(gpu_ids)]
            futures[executor.submit(run_one, job, gpu_id=gpu_id, args=args, control_dir=control_dir)] = job
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except subprocess.TimeoutExpired as exc:
                job = futures[future]
                results.append(
                    {
                        "label": str(job.get("label")),
                        "status": "failed",
                        "failure_reason": f"timeout:{exc.timeout}",
                    }
                )

    summary = {
        "schema": "stage2_fp32_ap_bulk_worker_summary_v1",
        "mode": args.mode,
        "queue": str(queue),
        "control_dir": str(control_dir),
        "job_count": len(jobs),
        "ok_count": sum(1 for row in results if row.get("status") in {"ok", "dry_run"}),
        "failed_count": sum(1 for row in results if row.get("status") == "failed"),
        "results": sorted(results, key=lambda row: str(row.get("label"))),
    }
    write_json(control_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
