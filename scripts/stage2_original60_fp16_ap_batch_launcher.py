#!/usr/bin/env python3
"""Preflight and select one original60 FP16 AP launch batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-plan",
        default=str(DEFAULT_OUTPUT_ROOT / "exports/fp16_original60_ap_launch_plan_latest.json"),
    )
    parser.add_argument("--phase", choices=("pilot", "bulk", "all"), default="pilot")
    parser.add_argument("--gpu-snapshot-json", default=None)
    parser.add_argument("--compute-apps-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nvidia_gpu_snapshot() -> dict[str, Any]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "memory_used": parts[2],
                    "memory_total": parts[3],
                }
            )
    return {"gpus": gpus, "returncode": proc.returncode}


def nvidia_compute_apps() -> dict[str, Any]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    apps = []
    for line in proc.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 4:
            apps.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory": parts[3],
                }
            )
    return {"compute_apps": apps, "returncode": proc.returncode}


def busy_gpus_from_inputs(gpu_snapshot: dict[str, Any], compute_apps: dict[str, Any]) -> list[int]:
    uuid_to_index = {
        str(item.get("uuid")): int(item["index"])
        for item in gpu_snapshot.get("gpus", [])
        if item.get("uuid") is not None and item.get("index") is not None
    }
    busy: set[int] = set()
    for item in compute_apps.get("compute_apps", []):
        if "gpu_index" in item:
            busy.add(int(item["gpu_index"]))
            continue
        uuid = str(item.get("gpu_uuid") or "")
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
    return sorted(busy)


def main() -> int:
    args = parse_args()
    plan = read_json(Path(args.launch_plan))
    gpu_snapshot = read_json(Path(args.gpu_snapshot_json)) if args.gpu_snapshot_json else nvidia_gpu_snapshot()
    compute_apps = read_json(Path(args.compute_apps_json)) if args.compute_apps_json else nvidia_compute_apps()
    busy_gpus = busy_gpus_from_inputs(gpu_snapshot, compute_apps)

    labels = [
        item
        for item in plan.get("labels", [])
        if args.phase == "all" or str(item.get("phase")) == args.phase
    ]
    selected: list[str] = []
    deferred: list[str] = []
    selected_items: list[dict[str, Any]] = []
    for item in labels:
        gpu_id = int(item["gpu_id"])
        if gpu_id in busy_gpus:
            deferred.append(str(item["label"]))
            continue
        selected.append(str(item["label"]))
        selected_items.append(item)

    payload = {
        "status": "dry_run" if args.dry_run else "planned_only",
        "phase": args.phase,
        "launch_plan": str(Path(args.launch_plan).resolve()),
        "busy_gpus": busy_gpus,
        "selected_labels": selected,
        "deferred_labels": deferred,
        "selected": selected_items,
        "gpu_snapshot": gpu_snapshot,
        "compute_apps": compute_apps,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
