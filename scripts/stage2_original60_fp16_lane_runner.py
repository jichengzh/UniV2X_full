#!/usr/bin/env python3
"""Sequentially consume original60 FP16 AP jobs on one GPU lane."""

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
DEFAULT_QUEUE = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
)
DEFAULT_ROWS_OUT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "rows/fp16_true_original60_ap_rows_v1.jsonl"
)
DEFAULT_CKPT_ROOT = Path("/exdata/jichengzhi/heal_research/checkpoints/stage1")
DEFAULT_RAW_ROOT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "raw/ap_eval_original60"
)
DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT))
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--master-port-base", type=int, default=29730)
    parser.add_argument("--labels", default="", help="comma-separated labels for this lane")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--watcher-poll-seconds", type=int, default=60)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ps_alive(pid: int | None) -> bool:
    if not pid:
        return False
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,stat="],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        if parts[0] != str(pid):
            continue
        stat = parts[1]
        return "Z" not in stat
    return False


def load_measured_labels(rows_out: Path) -> set[str]:
    labels: set[str] = set()
    for row in read_jsonl(rows_out):
        if str(row.get("measurement_status") or "") == "measured":
            label = str(row.get("label") or "")
            if label:
                labels.add(label)
    return labels


def completion_jobs(queue_path: Path) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(queue_path):
        if str(row.get("precision") or "") != "fp16":
            continue
        if str(row.get("ap_status") or "") != "no_claim":
            continue
        label = str(row.get("label") or "")
        if label:
            jobs[label] = row
    return jobs


def raw_dir_for(raw_root: Path, label: str, gpu_id: int) -> Path:
    return raw_root / f"fp16_ckpt_gen_{label}_gpu{gpu_id}_v1"


def ckpt_dir_for(ckpt_root: Path, label: str) -> Path:
    return ckpt_root / f"Pyramid_DAIR_m1_stage2_ap_{label}_2026_06_28"


def master_port_for(base: int, *, gpu_id: int, label_index: int) -> int:
    return base + gpu_id + (label_index * 8)


def launch_train(
    args: argparse.Namespace,
    label: str,
    width: list[int],
    ckpt_dir: Path,
    raw_dir: Path,
    label_index: int,
) -> int:
    command = [
        args.env_python,
        str(ROOT / "scripts/stage2_original60_fp16_train_launcher.py"),
        "--label",
        label,
        "--width",
        ",".join(str(x) for x in width),
        "--gpu-id",
        str(args.gpu_id),
        "--master-port",
        str(master_port_for(args.master_port_base, gpu_id=args.gpu_id, label_index=label_index)),
        "--ckpt-dir",
        str(ckpt_dir),
        "--raw-dir",
        str(raw_dir),
    ]
    proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
    write_json(raw_dir / "lane_train_launch.json", {"command": command, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    if proc.returncode != 0:
        raise RuntimeError(f"train launcher failed for {label}: rc={proc.returncode}")
    train_pid_path = raw_dir / "train_runner.pid"
    if train_pid_path.exists():
        return int(train_pid_path.read_text(encoding="utf-8").strip())
    if (raw_dir / "train_skip.json").exists():
        return 0
    raise RuntimeError(f"train launcher returned success without train_runner.pid or train_skip.json for {label}")


def ensure_watcher(args: argparse.Namespace, label: str, width: list[int], ckpt_dir: Path, raw_dir: Path, train_pid: int) -> int:
    pid_path = raw_dir / "eval_watcher.pid"
    if pid_path.exists():
        try:
            watcher_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            watcher_pid = 0
        if ps_alive(watcher_pid):
            return watcher_pid
    stdout_path = raw_dir / "eval_watcher.stdout"
    stderr_path = raw_dir / "eval_watcher.stderr"
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    command = [
        args.env_python,
        str(ROOT / "scripts/stage2_original60_fp16_eval_when_ready.py"),
        "--label",
        label,
        "--width",
        ",".join(str(x) for x in width),
        "--gpu-id",
        str(args.gpu_id),
        "--ckpt-dir",
        str(ckpt_dir),
        "--raw-dir",
        str(raw_dir),
        "--train-pid",
        str(train_pid),
        "--rows-out",
        str(Path(args.rows_out).resolve()),
        "--poll-seconds",
        str(args.watcher_poll_seconds),
    ]
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    write_json(raw_dir / "lane_eval_watcher_launch.json", {"command": command, "pid": proc.pid})
    return proc.pid


def wait_for_label(args: argparse.Namespace, label: str, raw_dir: Path) -> None:
    while True:
        measured = load_measured_labels(Path(args.rows_out))
        if label in measured:
            return
        watcher_status = raw_dir / "eval_watcher_status.json"
        if watcher_status.exists():
            payload = json.loads(watcher_status.read_text(encoding="utf-8"))
            status = str(payload.get("status") or "")
            if status in {"eval_failed", "blocked"}:
                raise RuntimeError(f"watcher ended with status={status} for {label}")
        time.sleep(args.poll_seconds)


def record_label_failure(*, raw_dir: Path, gpu_id: int, label: str, index: int, reason: str) -> None:
    write_json(
        raw_dir / "lane_failure.json",
        {
            "gpu_id": gpu_id,
            "label": label,
            "index": index,
            "status": "label_failed",
            "reason": reason,
        },
    )


def run_lane(args: argparse.Namespace) -> int:
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if not labels:
        raise SystemExit("labels must be non-empty")

    queue = completion_jobs(Path(args.queue))
    rows_out = Path(args.rows_out)
    ckpt_root = Path(args.ckpt_root)
    raw_root = Path(args.raw_root)
    lane_log = raw_root / f"fp16_lane_gpu{args.gpu_id}_status.json"
    failed_labels: list[str] = []

    for index, label in enumerate(labels):
        job = queue.get(label)
        if job is None:
            write_json(lane_log, {"gpu_id": args.gpu_id, "label": label, "status": "missing_from_queue", "index": index})
            failed_labels.append(label)
            continue
        if label in load_measured_labels(rows_out):
            write_json(lane_log, {"gpu_id": args.gpu_id, "label": label, "status": "already_measured", "index": index})
            continue

        width = [int(x) for x in job.get("width") or []]
        ckpt_dir = ckpt_dir_for(ckpt_root, label)
        raw_dir = raw_dir_for(raw_root, label, args.gpu_id)
        raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            train_pid_path = raw_dir / "train_runner.pid"
            train_pid = int(train_pid_path.read_text(encoding="utf-8").strip()) if train_pid_path.exists() else 0
            if not ps_alive(train_pid):
                train_pid = launch_train(args, label, width, ckpt_dir, raw_dir, index)
            watcher_pid = ensure_watcher(args, label, width, ckpt_dir, raw_dir, train_pid)

            write_json(
                lane_log,
                {
                    "gpu_id": args.gpu_id,
                    "label": label,
                    "index": index,
                    "status": "running",
                    "train_pid": train_pid,
                    "watcher_pid": watcher_pid,
                    "width": width,
                    "raw_dir": str(raw_dir),
                    "ckpt_dir": str(ckpt_dir),
                },
            )
            wait_for_label(args, label, raw_dir)
            write_json(
                lane_log,
                {
                    "gpu_id": args.gpu_id,
                    "label": label,
                    "index": index,
                    "status": "measured",
                    "width": width,
                    "raw_dir": str(raw_dir),
                    "ckpt_dir": str(ckpt_dir),
                },
            )
        except Exception as exc:
            failed_labels.append(label)
            record_label_failure(raw_dir=raw_dir, gpu_id=args.gpu_id, label=label, index=index, reason=str(exc))
            write_json(
                lane_log,
                {
                    "gpu_id": args.gpu_id,
                    "label": label,
                    "index": index,
                    "status": "label_failed",
                    "reason": str(exc),
                    "raw_dir": str(raw_dir),
                    "ckpt_dir": str(ckpt_dir),
                },
            )
            continue

    write_json(
        lane_log,
        {
            "gpu_id": args.gpu_id,
            "status": "lane_complete",
            "labels": labels,
            "failed_labels": failed_labels,
        },
    )
    return 0


def main() -> int:
    args = parse_args()
    return run_lane(args)


if __name__ == "__main__":
    raise SystemExit(main())
