#!/usr/bin/env python3
"""Authenticate and pause only the Stage7 takeover orchestrator."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: pause.py ROOT PID AUDIT")
    root = Path(sys.argv[1]).resolve()
    pid = int(sys.argv[2])
    audit_path = Path(sys.argv[3]).resolve()
    proc = Path("/proc") / str(pid)
    command = tuple(
        part.decode("utf-8", errors="replace")
        for part in (proc / "cmdline").read_bytes().split(b"\0")
        if part
    )
    if (
        not command
        or "stage7_speed_priority_takeover_v1.py" not in " ".join(command)
        or str(root) not in command
        or os.stat(proc).st_uid != os.getuid()
    ):
        raise SystemExit("refuse to signal unauthenticated process")
    status_before = (proc / "status").read_text(encoding="utf-8")
    start_time = (proc / "stat").read_text(encoding="utf-8").split()[21]
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while proc.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if proc.exists():
        raise SystemExit("orchestrator did not stop after SIGTERM")
    atomic_json(
        audit_path,
        {
            "schema_version": "stage7_maintenance_orchestrator_pause_v1",
            "wall_time": time.time(),
            "root": str(root),
            "pid": pid,
            "start_time": start_time,
            "command": list(command),
            "signal": "SIGTERM",
            "stopped": True,
            "worker_processes_signaled": 0,
            "selected_event_budget_delta": 0,
            "status_before": status_before,
            "reason": "drain active workers before reviewed deployment",
        },
    )
    print(json.dumps({"stopped_pid": pid, "audit": str(audit_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
