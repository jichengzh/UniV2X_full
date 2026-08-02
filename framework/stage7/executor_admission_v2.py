"""Frozen executor, primitive, and process-identity admission for Stage7 v2."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import pwd
from typing import Callable, Mapping, Sequence

from framework.stage7.core_ablation_v2 import (
    FROZEN_ACTUAL_V3_EXECUTORS,
    canonical_sha256,
)


ProcessIdentityProbe = Callable[[int], Mapping[str, object]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verified_primitive_bindings(
    records: object,
    *,
    repo_root: Path,
    primitive_paths: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    if not isinstance(records, Mapping) or set(records) != set(primitive_paths):
        raise ValueError("embedded primitive binding key drift")
    root = Path(repo_root).resolve(strict=True)
    verified: dict[str, dict[str, str]] = {}
    for name, relative_text in primitive_paths.items():
        record = records[name]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"embedded primitive record shape drift: {name}")
        expected_path = (root / relative_text).resolve(strict=False)
        supplied_path = Path(str(record.get("path") or ""))
        expected_sha = record.get("sha256")
        if (
            not supplied_path.is_absolute()
            or supplied_path.resolve(strict=False) != expected_path
            or not expected_path.is_file()
            or not is_sha256(expected_sha)
            or file_sha256(expected_path) != expected_sha
        ):
            raise ValueError(f"embedded primitive path/SHA drift: {name}")
        verified[name] = {
            "path": str(expected_path),
            "repo_relative_path": relative_text,
            "sha256": str(expected_sha),
        }
    return verified


def protocol_hashes(
    primitives: Mapping[str, Mapping[str, str]],
    protocol_primitives: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    return {
        field: canonical_sha256(
            [
                {
                    "name": name,
                    "repo_relative_path": primitives[name]["repo_relative_path"],
                    "sha256": primitives[name]["sha256"],
                }
                for name in names
            ]
        )
        for field, names in protocol_primitives.items()
    }


def core_executor_records() -> list[dict[str, object]]:
    """Return a mutable copy without exposing the frozen core tuple."""
    return [copy.deepcopy(record) for record in FROZEN_ACTUAL_V3_EXECUTORS]


def verify_frozen_actual_v3_executors(
    repo_root: Path,
    *,
    executor_records: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Read and hash every frozen executor before any v2 artifact is written."""
    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("actual-feedback v3 executor repo_root is invalid")
    records = core_executor_records() if executor_records is None else executor_records
    verified: list[dict[str, object]] = []
    for record in records:
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("actual-feedback v3 executor path is unsafe")
        path = (root / relative).resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "actual-feedback v3 executor path escapes repo_root"
            ) from error
        if not path.is_file():
            raise ValueError(f"actual-feedback v3 executor is missing: {relative}")
        actual_sha = file_sha256(path)
        if actual_sha != record["sha256"]:
            raise ValueError(f"actual-feedback v3 executor SHA drift: {relative}")
        verified.append(
            {
                "path": str(relative),
                "resolved_path": str(path),
                "expected_sha256": record["sha256"],
                "actual_sha256": actual_sha,
            }
        )
    return verified


def normalize_process_identity(
    payload: Mapping[str, object], expected_pid: int
) -> dict[str, object]:
    identity = copy.deepcopy(dict(payload))
    expected_fields = {
        "pid",
        "uid",
        "username",
        "start_time_ticks",
        "executable",
        "cmdline",
    }
    if set(identity) != expected_fields:
        raise ValueError("orchestrator process identity schema drift")
    if identity.get("pid") != expected_pid:
        raise ValueError("orchestrator process identity PID drift")
    uid = identity.get("uid")
    start_ticks = identity.get("start_time_ticks")
    if (
        isinstance(uid, bool)
        or not isinstance(uid, int)
        or uid < 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
    ):
        raise ValueError("orchestrator process identity numeric drift")
    username = identity.get("username")
    executable = identity.get("executable")
    cmdline = identity.get("cmdline")
    if not isinstance(username, str) or not username:
        raise ValueError("orchestrator process identity owner drift")
    if (
        not isinstance(executable, str)
        or not Path(executable).is_absolute()
        or not isinstance(cmdline, list)
        or not cmdline
        or not all(isinstance(item, str) and item for item in cmdline)
    ):
        raise ValueError("orchestrator process identity command drift")
    normalized = {
        "pid": expected_pid,
        "uid": uid,
        "username": username,
        "start_time_ticks": start_ticks,
        "executable": str(Path(executable).resolve(strict=False)),
        "cmdline": list(cmdline),
    }
    return {
        **normalized,
        "process_identity_sha256": canonical_sha256(normalized),
    }


def capture_process_identity(
    probe: ProcessIdentityProbe, pid: int
) -> dict[str, object]:
    try:
        return normalize_process_identity(probe(pid), pid)
    except (OSError, ValueError) as error:
        raise ValueError(
            "v2 recovery orchestrator process identity probe failed"
        ) from error


def probe_linux_process_identity(pid: int) -> dict[str, object]:
    """Capture a stable Linux process identity and reject dead/reused PIDs."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("orchestrator PID is invalid")
    proc = Path("/proc") / str(pid)
    try:
        stat_before = (proc / "stat").read_text(encoding="utf-8")
        status = (proc / "status").read_text(encoding="utf-8")
        cmdline_bytes = (proc / "cmdline").read_bytes()
        executable = str((proc / "exe").resolve(strict=True))
        stat_after = (proc / "stat").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError("orchestrator process is not alive") from error

    def start_time(stat: str) -> int:
        closing = stat.rfind(")")
        fields = stat[closing + 2 :].split() if closing >= 0 else []
        if len(fields) <= 19:
            raise ValueError("orchestrator process stat identity is invalid")
        try:
            return int(fields[19])
        except ValueError as error:
            raise ValueError("orchestrator process start time is invalid") from error

    start_time_ticks = start_time(stat_before)
    if start_time_ticks != start_time(stat_after):
        raise ValueError("orchestrator process identity changed during probe")
    uid_line = next(
        (line for line in status.splitlines() if line.startswith("Uid:")), ""
    )
    try:
        uid = int(uid_line.split()[1])
        username = pwd.getpwuid(uid).pw_name
    except (IndexError, KeyError, ValueError) as error:
        raise ValueError("orchestrator process owner identity is invalid") from error
    try:
        cmdline = [item.decode("utf-8") for item in cmdline_bytes.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise ValueError("orchestrator process command identity is invalid") from error
    return {
        "pid": pid,
        "uid": uid,
        "username": username,
        "start_time_ticks": start_time_ticks,
        "executable": executable,
        "cmdline": cmdline,
    }


__all__ = [
    "ProcessIdentityProbe",
    "capture_process_identity",
    "core_executor_records",
    "file_sha256",
    "is_sha256",
    "normalize_process_identity",
    "probe_linux_process_identity",
    "protocol_hashes",
    "verified_primitive_bindings",
    "verify_frozen_actual_v3_executors",
]
