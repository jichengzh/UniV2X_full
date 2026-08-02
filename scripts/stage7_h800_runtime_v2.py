#!/usr/bin/env python3
"""Small, side-effect-free H800 runtime inventory helpers for Stage7."""

from __future__ import annotations

import fcntl
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence


class FileLock:
    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def release(self) -> None:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def acquire_resource_lock(resource_key: str) -> FileLock | None:
    digest = hashlib.sha256(resource_key.encode("utf-8")).hexdigest()
    path = Path("/var/lock") / f"stage7_resource_{digest}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return FileLock(handle)


def query_gpu_models(
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, str]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    records: dict[str, str] = {}
    for line in result.stdout.splitlines():
        uuid, separator, model = line.partition(",")
        if not separator or not uuid.strip() or not model.strip():
            raise ValueError("nvidia-smi GPU model output is malformed")
        records = {**records, uuid.strip(): model.strip()}
    return records


def discover_process_reservations(
    processes: Sequence[Any],
) -> dict[str, tuple[str, ...]]:
    reservations: dict[str, tuple[str, ...]] = {}
    single_flags = {"--gpu", "--gpu-id", "--physical-gpu-id", "--device"}
    pool_flags = {"--gpus", "--gpu-pool", "--gpu-uuids"}
    for process in processes:
        command = tuple(process.command)
        if not command:
            continue
        reason = "active_process_reservation:" f"{process.pid}:{Path(command[0]).name}"
        for index, value in enumerate(command[:-1]):
            if value in single_flags and command[index + 1].isdigit():
                key = f"index:{int(command[index + 1])}"
                reservations[key] = (*reservations.get(key, ()), reason)
            if value not in pool_flags:
                continue
            for raw in command[index + 1].split(","):
                item = raw.strip()
                key = f"index:{int(item)}" if item.isdigit() else item
                if key:
                    reservations[key] = (*reservations.get(key, ()), reason)
    return reservations


def redacted_command(
    command: Sequence[str],
    secret_terms: Sequence[str],
) -> list[str]:
    result: list[str] = []
    redact_next = False
    for value in command:
        lowered = value.lower()
        if redact_next:
            result.append("<redacted>")
            redact_next = False
        elif any(term in lowered for term in secret_terms):
            result.append(value.split("=", 1)[0])
            redact_next = "=" not in value
        else:
            result.append(value)
    return result


__all__ = [
    "FileLock",
    "acquire_resource_lock",
    "discover_process_reservations",
    "query_gpu_models",
    "redacted_command",
]
