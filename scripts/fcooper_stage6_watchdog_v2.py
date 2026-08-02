#!/usr/bin/env python3
"""Bounded, evidence-preserving watchdog for the F-Cooper Stage6 supervisor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


SUPERVISOR_TOKEN = "fcooper_stage6_five_arm_supervisor_v2.sh"
RUNNER_TOKEN = "fcooper_execute_measurement_row_v2.py"


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    command: tuple[str, ...]
    start_ticks: int = 0


def _read_command(path: Path) -> tuple[str, ...]:
    values = [part.decode() for part in path.read_bytes().split(b"\0") if part]
    return tuple(values)


def process_snapshot(proc_root: Path = Path("/proc")) -> dict[int, ProcessInfo]:
    result: dict[int, ProcessInfo] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text().splitlines()
            ppid_line = next(line for line in status if line.startswith("PPid:"))
            command = _read_command(entry / "cmdline")
            stat_tail = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            start_ticks = int(stat_tail[19])
        except (
            FileNotFoundError,
            PermissionError,
            StopIteration,
            UnicodeDecodeError,
            ValueError,
            IndexError,
        ):
            continue
        if command:
            pid = int(entry.name)
            result[pid] = ProcessInfo(
                pid=pid,
                ppid=int(ppid_line.split()[1]),
                command=command,
                start_ticks=start_ticks,
            )
    return result


def _contains_script_argv(command: Sequence[str], script_name: str) -> bool:
    return any(Path(argument).name == script_name for argument in command)


def root_supervisor_pids(
    processes: Mapping[int, ProcessInfo],
    command: Sequence[str],
) -> list[int]:
    matching = {
        pid
        for pid, process in processes.items()
        if process.command == tuple(command)
    }
    return sorted(
        pid for pid in matching if processes[pid].ppid not in matching
    )


def _argument_is_under(argument: str, root: Path) -> bool:
    if not os.path.isabs(argument):
        return False
    try:
        common = os.path.commonpath(
            [str(Path(argument).resolve()), str(root.resolve())]
        )
    except ValueError:
        return False
    return common == str(root.resolve())


def formal_runner_pids(
    processes: Mapping[int, ProcessInfo],
    formal_root: Path,
) -> list[int]:
    matching = {
        pid
        for pid, process in processes.items()
        if _contains_script_argv(process.command, RUNNER_TOKEN)
        and any(_argument_is_under(argument, formal_root) for argument in process.command)
    }
    return sorted(
        pid for pid in matching if processes[pid].ppid not in matching
    )


def capture_supervisor_command(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[str, ...]:
    command = _read_command(proc_root / str(pid) / "cmdline")
    if not _contains_script_argv(command, SUPERVISOR_TOKEN):
        raise ValueError("captured command is missing the required token")
    return command


def validate_initial_supervisor(
    pid: int,
    processes: Mapping[int, ProcessInfo],
    command: Sequence[str],
) -> None:
    if pid not in root_supervisor_pids(processes, command):
        raise ValueError("initial PID is not the exact root supervisor process")


def tracked_pid_alive(
    processes: Mapping[int, ProcessInfo],
    pid: int | None,
    start_ticks: int | None,
) -> bool:
    if not isinstance(pid, int) or pid not in processes:
        return False
    return start_ticks is None or processes[pid].start_ticks == start_ticks


def discover_live_supervisor_pid(
    processes: Mapping[int, ProcessInfo],
    *,
    pid_file: Path,
    formal_root: Path,
) -> int | None:
    try:
        pid = int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    process = processes.get(pid)
    if process is None:
        return None
    if not _contains_script_argv(process.command, SUPERVISOR_TOKEN):
        return None
    if not any(_argument_is_under(argument, formal_root) for argument in process.command):
        return None
    return pid


def decide_action(
    *,
    complete: bool,
    supervisor_pids: Sequence[int],
    runner_pids: Sequence[int],
    restart_count: int,
    max_restarts: int,
) -> str:
    if complete:
        return "complete"
    if supervisor_pids:
        return "monitor"
    if runner_pids:
        return "wait_runners"
    if restart_count >= max_restarts:
        return "exhausted"
    return "restart"


def restart_guard_action(
    *,
    complete: bool,
    processes: Mapping[int, ProcessInfo],
    formal_root: Path,
    tracked_supervisor_pid: int | None,
    tracked_supervisor_start_ticks: int | None = None,
) -> str:
    if complete:
        return "complete"
    if tracked_pid_alive(
        processes,
        tracked_supervisor_pid,
        tracked_supervisor_start_ticks,
    ):
        return "monitor"
    if formal_runner_pids(processes, formal_root):
        return "wait_runners"
    return "restart"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _state_payload(
    *,
    status: str,
    command: Sequence[str],
    restart_count: int,
    events: Sequence[Mapping[str, object]],
    tracked_supervisor_pid: int | None = None,
    tracked_supervisor_start_ticks: int | None = None,
) -> dict[str, object]:
    command_bytes = b"\0".join(item.encode() for item in command)
    return {
        "schema": "fcooper_stage6_watchdog_v2",
        "status": status,
        "updated_at_utc": _utc_now(),
        "restart_count": restart_count,
        "tracked_supervisor_pid": tracked_supervisor_pid,
        "tracked_supervisor_start_ticks": tracked_supervisor_start_ticks,
        "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
        "events": list(events),
    }


def serialize_state(
    *,
    status: str,
    command: Sequence[str],
    restart_count: int,
    events: Sequence[Mapping[str, object]],
    tracked_supervisor_pid: int | None = None,
    tracked_supervisor_start_ticks: int | None = None,
) -> str:
    return (
        json.dumps(
            _state_payload(
                status=status,
                command=command,
                restart_count=restart_count,
                events=events,
                tracked_supervisor_pid=tracked_supervisor_pid,
                tracked_supervisor_start_ticks=tracked_supervisor_start_ticks,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _command_sha256(command: Sequence[str]) -> str:
    return hashlib.sha256(b"\0".join(item.encode() for item in command)).hexdigest()


def load_restart_history(
    state_path: Path,
    command: Sequence[str],
) -> tuple[int, list[dict[str, object]]]:
    if not state_path.is_file():
        return 0, []
    payload = json.loads(state_path.read_text())
    if payload.get("schema") != "fcooper_stage6_watchdog_v2":
        raise ValueError("existing watchdog state has an unexpected schema")
    if payload.get("command_sha256") != _command_sha256(command):
        raise ValueError("existing watchdog state belongs to a different command")
    restart_count = payload.get("restart_count")
    events = payload.get("events")
    if not isinstance(restart_count, int) or restart_count < 0:
        raise ValueError("existing watchdog restart_count is invalid")
    if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
        raise ValueError("existing watchdog events are invalid")
    return restart_count, events


def _require_exact_path(actual: Path, expected: Path, option: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{option} must be {expected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-supervisor-pid", type=int, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--complete-marker", type=Path, required=True)
    parser.add_argument("--state-json", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    if args.max_restarts < 0:
        raise ValueError("--max-restarts cannot be negative")

    formal_root = args.formal_root.resolve()
    watchdog_root = formal_root / "controls" / "watchdog"
    _require_exact_path(
        args.complete_marker,
        formal_root / "controls" / "five_arm_supervisor_complete.json",
        "--complete-marker",
    )
    _require_exact_path(
        args.state_json, watchdog_root / "state.json", "--state-json"
    )
    _require_exact_path(
        args.lock_file, watchdog_root / "watchdog.lock", "--lock-file"
    )
    _require_exact_path(
        args.log_file,
        watchdog_root / "supervisor_restart.log",
        "--log-file",
    )

    command = capture_supervisor_command(
        args.initial_supervisor_pid,
    )
    initial_processes = process_snapshot()
    validate_initial_supervisor(
        args.initial_supervisor_pid, initial_processes, command
    )
    if not any(_argument_is_under(argument, formal_root) for argument in command):
        raise ValueError("captured supervisor command is not bound to --formal-root")

    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = args.lock_file.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another watchdog instance already owns the lock") from exc

    restart_count, old_events = load_restart_history(args.state_json, command)
    current_supervisor_pid: int | None = args.initial_supervisor_pid
    current_supervisor_start_ticks: int | None = initial_processes[
        args.initial_supervisor_pid
    ].start_ticks
    supervisor_pid_file = (
        formal_root / "controls" / "five_arm_supervisor.pid"
    )
    events: list[dict[str, object]] = [
        *old_events,
        {
            "event": "watchdog_started",
            "at_utc": _utc_now(),
            "initial_supervisor_pid": args.initial_supervisor_pid,
        },
    ]
    with args.log_file.open("ab", buffering=0) as log_handle:
        while True:
            processes = process_snapshot()
            if not tracked_pid_alive(
                processes,
                current_supervisor_pid,
                current_supervisor_start_ticks,
            ):
                current_supervisor_pid = discover_live_supervisor_pid(
                    processes,
                    pid_file=supervisor_pid_file,
                    formal_root=formal_root,
                )
                current_supervisor_start_ticks = (
                    processes[current_supervisor_pid].start_ticks
                    if current_supervisor_pid is not None
                    else None
                )
            supervisor_pids = (
                [current_supervisor_pid]
                if tracked_pid_alive(
                    processes,
                    current_supervisor_pid,
                    current_supervisor_start_ticks,
                )
                else []
            )
            runner_pids = formal_runner_pids(processes, formal_root)
            action = decide_action(
                complete=args.complete_marker.is_file(),
                supervisor_pids=supervisor_pids,
                runner_pids=runner_pids,
                restart_count=restart_count,
                max_restarts=args.max_restarts,
            )
            _atomic_write_json(
                args.state_json,
                _state_payload(
                    status=action,
                    command=command,
                    restart_count=restart_count,
                    events=events,
                    tracked_supervisor_pid=current_supervisor_pid,
                    tracked_supervisor_start_ticks=(
                        current_supervisor_start_ticks
                    ),
                ),
            )
            if action == "complete":
                return 0
            if action == "exhausted":
                return 2
            if action == "restart":
                guarded_processes = process_snapshot()
                discovered_pid = discover_live_supervisor_pid(
                    guarded_processes,
                    pid_file=supervisor_pid_file,
                    formal_root=formal_root,
                )
                if discovered_pid is not None:
                    current_supervisor_pid = discovered_pid
                    time.sleep(args.poll_seconds)
                    continue
                guarded_action = restart_guard_action(
                    complete=args.complete_marker.is_file(),
                    processes=guarded_processes,
                    formal_root=formal_root,
                    tracked_supervisor_pid=current_supervisor_pid,
                    tracked_supervisor_start_ticks=(
                        current_supervisor_start_ticks
                    ),
                )
                if guarded_action != "restart":
                    time.sleep(args.poll_seconds)
                    continue
                restart_count += 1
                events = [
                    *events,
                    {
                        "event": "supervisor_restart_reserved",
                        "at_utc": _utc_now(),
                        "restart_count": restart_count,
                    },
                ]
                _atomic_write_json(
                    args.state_json,
                    _state_payload(
                        status="restart_reserved",
                        command=command,
                        restart_count=restart_count,
                        events=events,
                        tracked_supervisor_pid=current_supervisor_pid,
                        tracked_supervisor_start_ticks=(
                            current_supervisor_start_ticks
                        ),
                    ),
                )
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                time.sleep(2.0)
                return_code = process.poll()
                if return_code is None:
                    current_supervisor_pid = process.pid
                    restarted_processes = process_snapshot()
                    restarted_process = restarted_processes.get(process.pid)
                    current_supervisor_start_ticks = (
                        restarted_process.start_ticks
                        if restarted_process is not None
                        else None
                    )
                    events = [
                        *events,
                        {
                            "event": "supervisor_restarted",
                            "at_utc": _utc_now(),
                            "pid": process.pid,
                            "restart_count": restart_count,
                        },
                    ]
                elif return_code == 75:
                    events = [
                        *events,
                        {
                            "event": "supervisor_restart_lock_contended",
                            "at_utc": _utc_now(),
                            "pid": process.pid,
                            "restart_count": restart_count,
                        },
                    ]
                elif args.complete_marker.is_file():
                    return 0
                else:
                    events = [
                        *events,
                        {
                            "event": "supervisor_restart_failed",
                            "at_utc": _utc_now(),
                            "pid": process.pid,
                            "return_code": return_code,
                            "restart_count": restart_count,
                        },
                    ]
                _atomic_write_json(
                    args.state_json,
                    _state_payload(
                        status=(
                            "monitor"
                            if return_code is None
                            else "restart_lock_contended"
                            if return_code == 75
                            else "restart_failed"
                        ),
                        command=command,
                        restart_count=restart_count,
                        events=events,
                        tracked_supervisor_pid=current_supervisor_pid,
                        tracked_supervisor_start_ticks=(
                            current_supervisor_start_ticks
                        ),
                    ),
                )
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
