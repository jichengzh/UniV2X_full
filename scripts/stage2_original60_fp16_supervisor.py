#!/usr/bin/env python3
"""Automate original60 FP16 lane launches and persist low-frequency state."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_original60_fp16_ap_batch_launcher import busy_gpus_from_inputs, nvidia_compute_apps, nvidia_gpu_snapshot

DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-plan",
        default=str(DEFAULT_OUTPUT_ROOT / "exports/fp16_original60_ap_launch_plan_latest.json"),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--phase", choices=("pilot", "bulk", "all"), default="all")
    parser.add_argument("--gpu-snapshot-json", default=None)
    parser.add_argument("--compute-apps-json", default=None)
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--master-port-base", type=int, default=29730)
    parser.add_argument("--poll-seconds", type=int, default=900)
    parser.add_argument("--watcher-poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_gpu_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    return read_json(Path(args.gpu_snapshot_json)) if args.gpu_snapshot_json else nvidia_gpu_snapshot()


def load_compute_apps(args: argparse.Namespace) -> dict[str, Any]:
    return read_json(Path(args.compute_apps_json)) if args.compute_apps_json else nvidia_compute_apps()


def active_lane_gpus() -> list[int]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,cmd"],
        capture_output=True,
        text=True,
        check=False,
    )
    active: set[int] = set()
    for line in proc.stdout.splitlines():
        if "stage2_original60_fp16_lane_runner.py" not in line:
            continue
        parts = line.split()
        for index, token in enumerate(parts):
            if token == "--gpu-id" and index + 1 < len(parts):
                try:
                    active.add(int(parts[index + 1]))
                except ValueError:
                    pass
    return sorted(active)


def eligible_labels(plan: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    labels = plan.get("labels", [])
    if phase == "all":
        return list(labels)
    return [item for item in labels if str(item.get("phase")) == phase]


def build_lane_launches(
    *,
    plan: dict[str, Any],
    phase: str,
    available_gpus: list[int],
    active_lane_gpus: list[int],
    output_root: Path,
    env_python: str,
    master_port_base: int,
    poll_seconds: int,
    watcher_poll_seconds: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    lane_map: dict[int, list[str]] = {}
    deferred: list[str] = []
    for item in eligible_labels(plan, phase):
        label = str(item["label"])
        gpu_id = int(item["gpu_id"])
        if gpu_id not in available_gpus:
            deferred.append(label)
            continue
        if gpu_id in active_lane_gpus:
            deferred.append(label)
            continue
        lane_map.setdefault(gpu_id, []).append(label)

    queue_path = output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
    rows_out = output_root / "rows/fp16_true_original60_ap_rows_v1.jsonl"
    raw_root = output_root / "raw/ap_eval_original60"
    launches: list[dict[str, Any]] = []
    for gpu_id in sorted(lane_map):
        labels = lane_map[gpu_id]
        command = [
            env_python,
            str(ROOT / "scripts/stage2_original60_fp16_lane_runner.py"),
            "--gpu-id",
            str(gpu_id),
            "--queue",
            str(queue_path),
            "--rows-out",
            str(rows_out),
            "--raw-root",
            str(raw_root),
            "--master-port-base",
            str(master_port_base),
            "--poll-seconds",
            str(poll_seconds),
            "--watcher-poll-seconds",
            str(watcher_poll_seconds),
            "--labels",
            ",".join(labels),
        ]
        launches.append({"gpu_id": gpu_id, "labels": labels, "command": command})
    return launches, deferred


def launch_lanes(launches: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    raw_root = output_root / "raw/ap_eval_original60"
    started: list[dict[str, Any]] = []
    for item in launches:
        gpu_id = int(item["gpu_id"])
        stdout_path = raw_root / f"fp16_lane_gpu{gpu_id}_runner.stdout"
        stderr_path = raw_root / f"fp16_lane_gpu{gpu_id}_runner.stderr"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
            proc = subprocess.Popen(item["command"], cwd=str(ROOT), stdout=stdout, stderr=stderr, start_new_session=True)
        pid_path = raw_root / f"fp16_lane_gpu{gpu_id}_runner.pid"
        pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
        started.append({**item, "pid": proc.pid, "pid_path": str(pid_path)})
    return started


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    launch_plan_path = Path(args.launch_plan).resolve()
    state_path = output_root / "exports/fp16_original60_supervisor_state_latest.json"
    stop_path = output_root / "exports/fp16_original60_supervisor.stop"

    while True:
        plan = read_json(launch_plan_path)
        gpu_snapshot = load_gpu_snapshot(args)
        compute_apps = load_compute_apps(args)
        busy_gpus = busy_gpus_from_inputs(gpu_snapshot, compute_apps)
        running_lane_gpus = active_lane_gpus()
        plan_gpus = [int(item) for item in plan.get("summary", {}).get("gpus", [])]
        available_gpus = [gpu for gpu in plan_gpus if gpu not in busy_gpus]
        lane_launches, deferred_labels = build_lane_launches(
            plan=plan,
            phase=args.phase,
            available_gpus=available_gpus,
            active_lane_gpus=running_lane_gpus,
            output_root=output_root,
            env_python=args.env_python,
            master_port_base=args.master_port_base,
            poll_seconds=args.poll_seconds,
            watcher_poll_seconds=args.watcher_poll_seconds,
        )

        status = "dry_run" if args.dry_run else "planned_only"
        started_launches = lane_launches
        if args.execute and not args.dry_run and lane_launches:
            started_launches = launch_lanes(lane_launches, output_root)
            status = "started"

        payload = {
            "status": status,
            "phase": args.phase,
            "launch_plan": str(launch_plan_path),
            "busy_gpus": busy_gpus,
            "active_lane_gpus": running_lane_gpus,
            "available_gpus": available_gpus,
            "deferred_labels": deferred_labels,
            "lane_launches": started_launches,
            "gpu_snapshot": gpu_snapshot,
            "compute_apps": compute_apps,
        }
        write_json(state_path, payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        sys.stdout.flush()

        if args.dry_run or not args.execute:
            return 0
        if stop_path.exists():
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
