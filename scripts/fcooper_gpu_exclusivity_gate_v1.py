#!/usr/bin/env python3
"""Fail-closed physical-GPU exclusivity gate for F-Cooper repeats."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "fcooper_gpu_exclusivity_gate_v1"
Sample = dict[str, Any]
Sampler = Callable[[int], Sample]


class GpuBusyError(RuntimeError):
    def __init__(self, audit: Mapping[str, Any]) -> None:
        self.audit = dict(audit)
        super().__init__(
            f"physical GPU{self.audit.get('gpu_index')} is not exclusive: "
            f"{self.audit.get('status')}"
        )


class TelemetryError(RuntimeError):
    """The physical-GPU state could not be measured without ambiguity."""


class GuardError(RuntimeError):
    def __init__(self, audit: Mapping[str, Any]) -> None:
        self.audit = dict(audit)
        super().__init__(
            f"GPU{self.audit.get('gpu_index')} guarded command failed: "
            f"{self.audit.get('status')}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_nvidia_smi(*arguments: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


def _csv_rows(payload: str) -> list[list[str]]:
    return [
        [field.strip() for field in row]
        for row in csv.reader(payload.splitlines())
        if row
    ]


def _parse_compute_process_rows(
    rows: Sequence[Sequence[str]], *, target_uuid: str
) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue
        if row[0] != target_uuid:
            continue
        if len(row) != 4:
            raise TelemetryError(
                f"malformed compute-app row for {target_uuid}: {list(row)!r}"
            )
        try:
            pid = int(row[1])
        except ValueError as error:
            raise TelemetryError(f"invalid compute-app pid: {row[1]!r}") from error
        try:
            used_memory = int(row[3])
        except ValueError as error:
            raise TelemetryError(
                f"invalid compute-app used_memory: {row[3]!r}"
            ) from error
        processes.append(
            {
                "pid": pid,
                "process_name": row[2],
                "used_memory_mib": used_memory,
            }
        )
    return sorted(processes, key=lambda row: int(row["pid"]))


class NvidiaSmiSampler:
    def __init__(self) -> None:
        self._uuid_by_index: dict[int, str] | None = None

    def _gpu_uuids(self) -> dict[int, str]:
        if self._uuid_by_index is None:
            rows = _csv_rows(
                _run_nvidia_smi(
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader,nounits",
                )
            )
            parsed: dict[int, str] = {}
            for row in rows:
                if len(row) != 2:
                    raise TelemetryError(f"malformed GPU identity row: {row!r}")
                try:
                    index = int(row[0])
                except ValueError as error:
                    raise TelemetryError(
                        f"invalid GPU index in telemetry: {row[0]!r}"
                    ) from error
                if not row[1]:
                    raise TelemetryError(f"empty GPU UUID for index {index}")
                parsed[index] = row[1]
            self._uuid_by_index = parsed
        return self._uuid_by_index

    def __call__(self, gpu_index: int) -> Sample:
        uuid = self._gpu_uuids().get(gpu_index)
        if not uuid:
            raise ValueError(f"physical GPU index not found: {gpu_index}")
        rows = _csv_rows(
            _run_nvidia_smi(
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            )
        )
        processes = _parse_compute_process_rows(rows, target_uuid=uuid)
        return {
            "gpu_index": gpu_index,
            "gpu_uuid": uuid,
            "processes": sorted(processes, key=lambda row: int(row["pid"])),
        }


def _base_audit(gpu_index: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "host": socket.gethostname(),
        "gpu_index": gpu_index,
        "started_at": _utc_now(),
    }


def _observation(
    sample: Mapping[str, Any], *, state: str, elapsed_seconds: float
) -> dict[str, Any]:
    return {
        "state": state,
        "elapsed_seconds": float(elapsed_seconds),
        "processes": [dict(row) for row in sample.get("processes") or []],
    }


def wait_for_gpu_exclusive(
    *,
    gpu_index: int,
    timeout_seconds: float,
    poll_seconds: float,
    quiet_seconds: float,
    sampler: Sampler | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if timeout_seconds < 0 or poll_seconds <= 0 or quiet_seconds < 0:
        raise ValueError("invalid GPU exclusivity timing contract")
    sample_gpu = sampler or NvidiaSmiSampler()
    started = clock()
    deadline = started + timeout_seconds
    audit = {
        **_base_audit(gpu_index),
        "status": "waiting",
        "timeout_seconds": float(timeout_seconds),
        "poll_seconds": float(poll_seconds),
        "quiet_seconds": float(quiet_seconds),
        "quiet_sample_count": 0,
        "observations": [],
    }
    last_state: str | None = None

    def record(current: Mapping[str, Any], state: str) -> None:
        nonlocal last_state
        if state != last_state:
            audit["observations"].append(
                _observation(
                    current,
                    state=state,
                    elapsed_seconds=clock() - started,
                )
            )
            last_state = state

    while True:
        if clock() >= deadline:
            audit.update(
                status="timeout_busy",
                ended_at=_utc_now(),
                wait_seconds=float(clock() - started),
                processes=[],
            )
            raise GpuBusyError(audit)
        current = sample_gpu(gpu_index)
        audit["gpu_uuid"] = current["gpu_uuid"]
        elapsed = clock() - started
        if current.get("processes"):
            record(current, "busy")
            if elapsed >= timeout_seconds:
                audit.update(
                    status="timeout_busy",
                    ended_at=_utc_now(),
                    wait_seconds=float(elapsed),
                    processes=[dict(row) for row in current["processes"]],
                )
                raise GpuBusyError(audit)
            sleeper(min(poll_seconds, max(timeout_seconds - elapsed, 0.0)))
            continue

        record(current, "empty_candidate")
        if quiet_seconds:
            remaining = deadline - clock()
            if quiet_seconds > remaining:
                audit.update(
                    status="timeout_quiet_window",
                    ended_at=_utc_now(),
                    wait_seconds=float(clock() - started),
                    processes=[],
                )
                raise GpuBusyError(audit)
            quiet_started = clock()
            quiet_interrupted = False
            while clock() - quiet_started < quiet_seconds:
                quiet_remaining = quiet_seconds - (clock() - quiet_started)
                deadline_remaining = deadline - clock()
                if deadline_remaining <= 0:
                    audit.update(
                        status="timeout_quiet_window",
                        ended_at=_utc_now(),
                        wait_seconds=float(clock() - started),
                        processes=[],
                    )
                    raise GpuBusyError(audit)
                sleeper(
                    min(
                        0.5,
                        poll_seconds,
                        quiet_remaining,
                        deadline_remaining,
                    )
                )
                confirmed = sample_gpu(gpu_index)
                audit["quiet_sample_count"] += 1
                audit["gpu_uuid"] = confirmed["gpu_uuid"]
                if confirmed.get("processes"):
                    record(confirmed, "busy")
                    quiet_interrupted = True
                    break
                current = confirmed
            if quiet_interrupted:
                continue
        record(current, "acquired")
        elapsed = clock() - started
        audit.update(
            status="acquired",
            ended_at=_utc_now(),
            wait_seconds=float(elapsed),
            processes=[],
        )
        return audit


def check_gpu_exclusive(
    *, gpu_index: int, sampler: Sampler | None = None
) -> dict[str, Any]:
    sample_gpu = sampler or NvidiaSmiSampler()
    current = sample_gpu(gpu_index)
    processes = [dict(row) for row in current.get("processes") or []]
    audit = {
        **_base_audit(gpu_index),
        "gpu_uuid": current["gpu_uuid"],
        "ended_at": _utc_now(),
        "status": "busy" if processes else "exclusive",
        "processes": processes,
    }
    if processes:
        raise GpuBusyError(audit)
    return audit


def write_audit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _descendant_pids(root_pid: int) -> set[int]:
    descendants = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        task_root = Path(f"/proc/{parent}/task")
        try:
            children_paths = list(task_root.glob("*/children"))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        for children_path in children_paths:
            try:
                content = children_path.read_text(encoding="utf-8").strip()
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            for token in content.split():
                try:
                    child = int(token)
                except ValueError:
                    continue
                if child not in descendants:
                    descendants.add(child)
                    pending.append(child)
    return descendants


def _process_group_pids(process_group_id: int) -> set[int]:
    members: set[int] = set()
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            payload = stat_path.read_text(encoding="utf-8")
            fields = payload[payload.rfind(")") + 2 :].split()
            if len(fields) >= 3 and int(fields[2]) == process_group_id:
                members.add(int(stat_path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return members


def _terminate_process_group(
    process_group_id: int, process: subprocess.Popen[Any]
) -> None:
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    if _process_group_pids(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass


def guard_command(
    *,
    gpu_index: int,
    command: Sequence[str],
    timeout_seconds: float,
    poll_seconds: float,
    quiet_seconds: float,
    monitor_seconds: float,
    runtime_timeout_seconds: float,
    cwd: Path | None,
    environment: Mapping[str, str],
    log_path: Path,
    sampler: Sampler | None = None,
) -> dict[str, Any]:
    if not command:
        raise ValueError("guarded command cannot be empty")
    if monitor_seconds <= 0 or runtime_timeout_seconds <= 0:
        raise ValueError("monitor and runtime timeout must be positive")
    sample_gpu = sampler or NvidiaSmiSampler()
    acquired = wait_for_gpu_exclusive(
        gpu_index=gpu_index,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        quiet_seconds=quiet_seconds,
        sampler=sample_gpu,
    )
    audit = {
        **_base_audit(gpu_index),
        "status": "launching",
        "evidence_scope": "sampled_process_exclusivity",
        "gpu_uuid": acquired["gpu_uuid"],
        "preflight": acquired,
        "command": list(command),
        "cwd": str(cwd) if cwd else None,
        "log_path": str(log_path),
        "environment_override_keys": sorted(environment),
        "monitor_seconds": float(monitor_seconds),
        "runtime_timeout_seconds": float(runtime_timeout_seconds),
        "runtime_sample_count": 0,
        "runtime_observations": [],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_environment = {**os.environ, **dict(environment)}
    with log_path.open("xb") as log_stream:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=merged_environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        audit["command_pid"] = process.pid
        process_group_id = process.pid
        audit["process_group_id"] = process_group_id
        audit["status"] = "running"
        runtime_started = time.monotonic()
        try:
            while process.poll() is None:
                runtime_elapsed = time.monotonic() - runtime_started
                if runtime_elapsed >= runtime_timeout_seconds:
                    audit.update(
                        status="runtime_timeout",
                        ended_at=_utc_now(),
                        runtime_seconds=float(runtime_elapsed),
                    )
                    _terminate_process_group(process_group_id, process)
                    raise GuardError(audit)
                current = sample_gpu(gpu_index)
                audit["runtime_sample_count"] += 1
                audit["last_runtime_sample_at"] = _utc_now()
                allowed = _descendant_pids(process.pid) | _process_group_pids(
                    process_group_id
                )
                foreign = [
                    dict(row)
                    for row in current.get("processes") or []
                    if int(row["pid"]) not in allowed
                ]
                if foreign:
                    audit["runtime_observations"].append(
                        {
                            "state": "foreign_process",
                            "timestamp": _utc_now(),
                            "processes": foreign,
                        }
                    )
                    audit.update(
                        status="runtime_interference",
                        ended_at=_utc_now(),
                        foreign_processes=foreign,
                    )
                    _terminate_process_group(process_group_id, process)
                    raise GuardError(audit)
                time.sleep(monitor_seconds)
        except BaseException:
            _terminate_process_group(process_group_id, process)
            raise
        return_code = process.wait()
        final_sample = sample_gpu(gpu_index)
        residual = [dict(row) for row in final_sample.get("processes") or []]
        if return_code != 0:
            _terminate_process_group(process_group_id, process)
            audit.update(
                status="command_failed",
                ended_at=_utc_now(),
                return_code=return_code,
                residual_processes=residual,
            )
            raise GuardError(audit)
        if residual:
            _terminate_process_group(process_group_id, process)
            audit.update(
                status="postcondition_busy",
                ended_at=_utc_now(),
                return_code=return_code,
                residual_processes=residual,
            )
            raise GuardError(audit)
        audit.update(
            status="completed_exclusive",
            ended_at=_utc_now(),
            return_code=return_code,
            runtime_seconds=float(time.monotonic() - runtime_started),
            residual_processes=[],
        )
        return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    wait_parser = subparsers.add_parser("wait")
    wait_parser.add_argument("--gpu-index", type=int, required=True)
    wait_parser.add_argument("--timeout-seconds", type=float, default=43200)
    wait_parser.add_argument("--poll-seconds", type=float, default=30)
    wait_parser.add_argument("--quiet-seconds", type=float, default=15)
    wait_parser.add_argument("--audit-json", type=Path, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--gpu-index", type=int, required=True)
    check_parser.add_argument("--audit-json", type=Path, required=True)
    guard_parser = subparsers.add_parser("guard")
    guard_parser.add_argument("--gpu-index", type=int, required=True)
    guard_parser.add_argument("--timeout-seconds", type=float, default=43200)
    guard_parser.add_argument("--poll-seconds", type=float, default=30)
    guard_parser.add_argument("--quiet-seconds", type=float, default=15)
    guard_parser.add_argument("--monitor-seconds", type=float, default=0.5)
    guard_parser.add_argument("--runtime-timeout-seconds", type=float, default=7200)
    guard_parser.add_argument("--lock-file", type=Path, required=True)
    guard_parser.add_argument("--audit-json", type=Path, required=True)
    guard_parser.add_argument("--log-file", type=Path, required=True)
    guard_parser.add_argument("--cwd", type=Path)
    guard_parser.add_argument("--env", action="append", default=[])
    guard_parser.add_argument("guarded_command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _environment_overrides(values: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env requires KEY=VALUE, got {value!r}")
        key, setting = value.split("=", 1)
        if not key or "\x00" in key or "=" in key:
            raise ValueError(f"invalid environment key: {key!r}")
        overrides[key] = setting
    return overrides


def _error_audit(gpu_index: int, error: BaseException) -> dict[str, Any]:
    return {
        **_base_audit(gpu_index),
        "status": "telemetry_or_gate_error",
        "ended_at": _utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def main() -> int:
    args = parse_args()
    try:
        if args.command == "wait":
            audit = wait_for_gpu_exclusive(
                gpu_index=args.gpu_index,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
                quiet_seconds=args.quiet_seconds,
            )
        elif args.command == "check":
            audit = check_gpu_exclusive(gpu_index=args.gpu_index)
        else:
            guarded_command = list(args.guarded_command)
            if guarded_command and guarded_command[0] == "--":
                guarded_command = guarded_command[1:]
            args.lock_file.parent.mkdir(parents=True, exist_ok=True)
            with args.lock_file.open("a+b") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                audit = guard_command(
                    gpu_index=args.gpu_index,
                    command=guarded_command,
                    timeout_seconds=args.timeout_seconds,
                    poll_seconds=args.poll_seconds,
                    quiet_seconds=args.quiet_seconds,
                    monitor_seconds=args.monitor_seconds,
                    runtime_timeout_seconds=args.runtime_timeout_seconds,
                    cwd=args.cwd,
                    environment=_environment_overrides(args.env),
                    log_path=args.log_file,
                )
                audit["lock_file"] = str(args.lock_file)
    except (GpuBusyError, GuardError) as error:
        write_audit(args.audit_json, error.audit)
        print(str(error), file=sys.stderr)
        return 2
    except BaseException as error:
        write_audit(args.audit_json, _error_audit(args.gpu_index, error))
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    write_audit(args.audit_json, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
