#!/usr/bin/env python3
"""Wait for one FP16 train job to reach a usable checkpoint, then run true AP eval."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_ROWS_OUT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "rows/fp16_true_original60_ap_rows_v1.jsonl"
)


def train_stdout_path(raw_dir: Path) -> Path:
    return raw_dir / "train_stdout.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT))
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--min-epoch", type=int, default=25)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ps_alive(pid: int) -> bool:
    proc = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True, check=False)
    return proc.returncode == 0 and str(pid) in proc.stdout


def best_checkpoint_epoch(ckpt_dir: Path) -> tuple[Path | None, int]:
    post_baseline_best_path: Path | None = None
    post_baseline_best_epoch = -1
    for path in ckpt_dir.glob("net_epoch_bestval_at*.pth"):
        match = re.search(r"bestval_at(\d+)", path.name)
        if match:
            epoch = int(match.group(1))
            if epoch > 23 and epoch > post_baseline_best_epoch:
                post_baseline_best_epoch = epoch
                post_baseline_best_path = path
    if post_baseline_best_path is not None:
        return post_baseline_best_path, post_baseline_best_epoch

    best_path: Path | None = None
    best_epoch = -1
    for path in ckpt_dir.glob("net_epoch*.pth"):
        if "bestval" in path.name:
            continue
        match = re.search(r"net_epoch(\d+)", path.name)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = path
    return best_path, best_epoch


def training_finished_marker(raw_dir: Path) -> bool:
    path = train_stdout_path(raw_dir)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "Training Finished, checkpoints saved to" in text


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    watcher_meta = {
        "schema": "stage2_original60_fp16_eval_when_ready_v1",
        "label": args.label,
        "width": args.width,
        "gpu_id": args.gpu_id,
        "train_pid": args.train_pid,
        "ckpt_dir": str(ckpt_dir),
        "raw_dir": str(raw_dir),
        "rows_out": str(Path(args.rows_out).resolve()),
        "poll_seconds": args.poll_seconds,
        "min_epoch": args.min_epoch,
    }
    write_json(raw_dir / "eval_watcher_command.json", watcher_meta)

    status_path = raw_dir / "eval_watcher_status.json"
    while ps_alive(args.train_pid):
        best_path, best_epoch = best_checkpoint_epoch(ckpt_dir)
        finished = training_finished_marker(raw_dir)
        ready_checkpoint = best_epoch >= args.min_epoch
        write_json(
            status_path,
            {
                **watcher_meta,
                "status": "waiting_for_train_exit",
                "best_checkpoint_seen": str(best_path) if best_path else None,
                "best_epoch_seen": best_epoch,
                "ready_checkpoint_seen": ready_checkpoint,
                "training_finished_marker": finished,
                "timestamp": int(time.time()),
            },
        )
        if finished and ready_checkpoint:
            break
        time.sleep(args.poll_seconds)

    best_path, best_epoch = best_checkpoint_epoch(ckpt_dir)
    if best_path is None or best_epoch < args.min_epoch:
        write_json(
            raw_dir / "ap_eval_blocker.json",
            {
                "schema": "stage2_true_fp16_ap_eval_blocker_v1",
                "label": args.label,
                "precision": "fp16",
                "failure_type": "checkpoint_generation_incomplete",
                "failure_reason": f"best checkpoint epoch {best_epoch} is below required min_epoch {args.min_epoch}",
                "gpu_id": args.gpu_id,
                "raw_artifact": str(raw_dir),
                "created_at": int(time.time()),
            },
        )
        write_json(
            status_path,
            {
                **watcher_meta,
                "status": "blocked",
                "best_checkpoint_seen": str(best_path) if best_path else None,
                "best_epoch_seen": best_epoch,
                "ready_checkpoint_seen": best_epoch >= args.min_epoch,
                "training_finished_marker": training_finished_marker(raw_dir),
                "timestamp": int(time.time()),
            },
        )
        return 2

    eval_raw_dir = raw_dir / "true_fp16_ap_eval"
    eval_raw_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONPATH"] = str(Path(args.heal_root).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        args.env_python,
        str(ROOT / "scripts/stage2_h800_true_fp16_ap_eval.py"),
        "--label",
        args.label,
        "--width",
        args.width,
        "--ckpt-dir",
        str(ckpt_dir),
        "--raw-dir",
        str(eval_raw_dir),
        "--rows-out",
        str(Path(args.rows_out).resolve()),
        "--heal-root",
        str(Path(args.heal_root).resolve()),
        "--gpu-id",
        str(args.gpu_id),
        "--precision-mode",
        "amp_fp16",
        "--execute",
    ]
    write_json(
        eval_raw_dir / "eval_dispatch.json",
        {
            "command": command,
            "cwd": str(ROOT),
            "env": {
                "CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"],
                "PYTHONPATH": env["PYTHONPATH"],
            },
            "best_checkpoint_seen": str(best_path),
            "best_epoch_seen": best_epoch,
        },
    )
    stdout_path = eval_raw_dir / "runner_stdout.txt"
    stderr_path = eval_raw_dir / "runner_stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    write_json(
        status_path,
        {
            **watcher_meta,
            "status": "completed" if proc.returncode == 0 else "eval_failed",
            "best_checkpoint_seen": str(best_path),
            "best_epoch_seen": best_epoch,
            "ready_checkpoint_seen": best_epoch >= args.min_epoch,
            "training_finished_marker": training_finished_marker(raw_dir),
            "eval_raw_dir": str(eval_raw_dir),
            "eval_returncode": proc.returncode,
            "timestamp": int(time.time()),
        },
    )
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
