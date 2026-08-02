#!/usr/bin/env python3
"""Watch remote original60 FP16 AP progress and auto-sync on changes."""

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
OUTPUT_ROOT = ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
EXPORT_ROOT = OUTPUT_ROOT / "exports"
DEFAULT_STATE_OUT = EXPORT_ROOT / "fp16_original60_remote_watch_state_latest.json"
DEFAULT_LOG_OUT = EXPORT_ROOT / "fp16_original60_remote_watch_log_latest.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default=os.environ.get("H800_HOST", ""))
    parser.add_argument("--remote-port", type=int, default=int(os.environ.get("H800_PORT", "30001")))
    parser.add_argument("--remote-user", default=os.environ.get("H800_USER", ""))
    parser.add_argument("--password-env", default="H800_PASS")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--state-out", default=str(DEFAULT_STATE_OUT))
    parser.add_argument("--log-out", default=str(DEFAULT_LOG_OUT))
    parser.add_argument("--max-loops", type=int, default=0, help="0 means infinite")
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, env=env, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(command)}\n{proc.stderr}")
    return proc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def remote_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if not args.remote_host or not args.remote_user:
        raise RuntimeError("set H800_HOST and H800_USER or pass --remote-host and --remote-user")
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"missing ${args.password_env}")
    env = os.environ.copy()
    env["SSHPASS"] = password
    script = r"""
python3 - <<'PY'
import json
from pathlib import Path
rows=Path('/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/rows/fp16_true_original60_ap_rows_v1.jsonl')
items=[json.loads(x) for x in rows.read_text().splitlines() if x.strip()]
base=Path('/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/raw/ap_eval_original60')
lanes={}
for gpu in [3,4,5,6]:
    p=base / f'fp16_lane_gpu{gpu}_status.json'
    lanes[str(gpu)] = json.loads(p.read_text()) if p.exists() else None
print(json.dumps({
    'row_count': len(items),
    'labels': sorted({r['label'] for r in items}),
    'lanes': lanes,
}, ensure_ascii=False))
PY
"""
    proc = run(
        [
            "sshpass",
            "-e",
            "ssh",
            "-p",
            str(args.remote_port),
            f"{args.remote_user}@{args.remote_host}",
            script,
        ],
        env=env,
    )
    return json.loads(proc.stdout)


def maybe_sync(args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault(args.password_env, os.environ.get(args.password_env, ""))
    proc = run(
        [sys.executable, str(ROOT / "scripts/stage2_sync_original60_fp16_ap_progress.py")],
        env=env,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def lane_signature(snapshot: dict[str, Any]) -> dict[str, Any]:
    signature: dict[str, Any] = {}
    for gpu, lane in snapshot.get("lanes", {}).items():
        if lane is None:
            signature[gpu] = None
            continue
        signature[gpu] = {
            "label": lane.get("label"),
            "index": lane.get("index"),
            "status": lane.get("status"),
        }
    return signature


def main() -> int:
    args = parse_args()
    state_out = Path(args.state_out)
    log_out = Path(args.log_out)
    last_row_count = None
    last_lane_signature = None
    loop = 0
    while args.max_loops == 0 or loop < args.max_loops:
        loop += 1
        now = int(time.time())
        snapshot = remote_snapshot(args)
        row_count = int(snapshot.get("row_count", 0))
        current_lane_signature = lane_signature(snapshot)
        changed = (row_count != last_row_count) or (current_lane_signature != last_lane_signature)
        sync_result = None
        if changed and last_row_count is not None and row_count > last_row_count:
            sync_result = maybe_sync(args)
        payload = {
            "timestamp": now,
            "loop": loop,
            "row_count": row_count,
            "labels": snapshot.get("labels", []),
            "lane_signature": current_lane_signature,
            "changed": changed,
            "sync_result": sync_result,
        }
        write_json(state_out, payload)
        append_jsonl(log_out, payload)
        last_row_count = row_count
        last_lane_signature = current_lane_signature
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
