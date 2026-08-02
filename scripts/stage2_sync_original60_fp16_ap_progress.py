#!/usr/bin/env python3
"""Sync remote original60 FP16 AP rows and refresh local coverage artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
LOCAL_ROWS = OUTPUT_ROOT / "rows/fp16_true_original60_ap_rows_v1.jsonl"
REVIEW_JSON = OUTPUT_ROOT / "exports/fp16_int8_original60_completion_review_latest.json"
DEFAULT_REMOTE_HOST = os.environ.get("H800_HOST", "")
DEFAULT_REMOTE_PORT = int(os.environ.get("H800_PORT", "30001"))
DEFAULT_REMOTE_USER = os.environ.get("H800_USER", "")
DEFAULT_REMOTE_ROWS = os.environ.get("H800_REMOTE_ROWS", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-rows", default=DEFAULT_REMOTE_ROWS)
    parser.add_argument("--password-env", default="H800_PASS")
    parser.add_argument("--skip-sync", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(command, cwd=str(ROOT), env=env, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def sync_rows(args: argparse.Namespace) -> None:
    if not args.remote_host or not args.remote_user or not args.remote_rows:
        raise SystemExit(
            "set H800_HOST, H800_USER, and H800_REMOTE_ROWS or pass the matching CLI options"
        )
    password = os.environ.get(args.password_env)
    if not password:
        raise SystemExit(f"missing ${args.password_env} for remote sync")
    env = os.environ.copy()
    env["SSHPASS"] = password
    remote = f"{args.remote_user}@{args.remote_host}:{args.remote_rows}"
    run(
        [
            "sshpass",
            "-e",
            "scp",
            "-P",
            str(args.remote_port),
            remote,
            str(LOCAL_ROWS),
        ],
        env=env,
    )


def refresh_local_views() -> None:
    python = sys.executable
    commands = [
        [python, str(ROOT / "scripts/stage2_generate_original60_quant_state_coverage.py")],
        [python, str(ROOT / "scripts/stage2_generate_fp16_int8_original60_completion_queue.py")],
        [python, str(ROOT / "scripts/stage2_generate_original60_quant_ap_true_eval_queue.py")],
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    for command in commands:
        run(command, env=env)


def print_summary() -> None:
    rows = [
        json.loads(line)
        for line in LOCAL_ROWS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    measured_labels = sorted(
        {
            str(row.get("label") or "")
            for row in rows
            if str(row.get("measurement_status") or row.get("ap_status") or "") == "measured"
        }
    )
    review = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
    fp16_jobs = [job for job in review.get("jobs", []) if str(job.get("precision") or "") == "fp16"]
    fp16_measured = sum(1 for job in fp16_jobs if str(job.get("ap_status") or "") == "measured")
    fp16_no_claim = sum(1 for job in fp16_jobs if str(job.get("ap_status") or "") == "no_claim")
    payload = {
        "local_row_count": len(rows),
        "measured_labels": measured_labels,
        "fp16_ap_measured": fp16_measured,
        "fp16_ap_no_claim": fp16_no_claim,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    if not args.skip_sync:
        sync_rows(args)
    refresh_local_views()
    print_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
