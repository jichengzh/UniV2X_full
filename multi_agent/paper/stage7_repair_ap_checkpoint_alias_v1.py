#!/usr/bin/env python3
"""Add an authenticated AP-compatible alias for a frozen low-epoch source."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping


GROUP_PATTERN = re.compile(r"pyramid\|([1-9][0-9]*)x([1-9][0-9]*)x([1-9][0-9]*)")
BESTVAL_PATTERN = re.compile(r"net_epoch_bestval_at([1-9][0-9]*)\.pth")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def source_paths(v2_root: Path, group_id: str) -> tuple[Path, Path]:
    match = GROUP_PATTERN.fullmatch(group_id)
    if match is None:
        raise ValueError("invalid Pyramid group_id")
    width = "x".join(f"{int(value):03d}" for value in match.groups())
    source_root = Path(v2_root).resolve() / "sources/pyramid" / width
    return source_root / "checkpoint", source_root / "source_ready_evidence.json"


def _require_file(path: Path, expected_sha256: str, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size <= 0
        or file_sha256(path) != expected_sha256
    ):
        raise ValueError(f"{label} identity mismatch")


def _matching_bestval(checkpoint_dir: Path, expected_sha256: str) -> tuple[Path, int]:
    matches: list[tuple[Path, int]] = []
    for path in checkpoint_dir.glob("net_epoch_bestval_at*.pth"):
        match = BESTVAL_PATTERN.fullmatch(path.name)
        if (
            match is not None
            and path.is_file()
            and not path.is_symlink()
            and file_sha256(path) == expected_sha256
        ):
            matches.append((path, int(match.group(1))))
    if len(matches) != 1:
        raise ValueError("expected exactly one matching bestval checkpoint")
    return matches[0]


def _create_hardlink(primary: Path, alias: Path, expected_sha256: str) -> None:
    if alias.exists():
        _require_file(alias, expected_sha256, "existing AP checkpoint alias")
        return
    temporary = alias.with_name(f".{alias.name}.tmp-{os.getpid()}")
    os.link(primary, temporary)
    _require_file(temporary, expected_sha256, "temporary AP checkpoint alias")
    os.replace(temporary, alias)


def repair_alias(
    *,
    v2_root: Path,
    group_id: str,
    expected_checkpoint_sha256: str,
    audit_path: Path,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_sha256):
        raise ValueError("invalid expected checkpoint SHA256")
    checkpoint_dir, evidence_path = source_paths(v2_root, group_id)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "ready"
        or evidence.get("group_id") != group_id
        or evidence.get("checkpoint_sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("source evidence identity mismatch")
    primary = Path(str(evidence.get("checkpoint_path") or "")).resolve()
    if primary.parent != checkpoint_dir.resolve():
        raise ValueError("source evidence checkpoint path escapes group")
    _require_file(primary, expected_checkpoint_sha256, "primary checkpoint")
    bestval, epoch = _matching_bestval(checkpoint_dir, expected_checkpoint_sha256)
    alias = checkpoint_dir / f"net_epoch{epoch}.pth"
    lock_path = checkpoint_dir.parent / "calibration/.stage5_source_group.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _create_hardlink(primary, alias, expected_checkpoint_sha256)
    payload = {
        "schema_version": "stage7_ap_checkpoint_alias_repair_v1",
        "status": "ready",
        "completed_wall_time": time.time(),
        "group_id": group_id,
        "primary_checkpoint_path": str(primary),
        "bestval_checkpoint_path": str(bestval.resolve()),
        "ap_checkpoint_alias_path": str(alias.resolve()),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "alias_sha256": file_sha256(alias),
        "source_evidence_path": str(evidence_path.resolve()),
        "source_evidence_file_sha256": file_sha256(evidence_path),
        "hardlink_identity_preserved": os.stat(primary).st_ino == os.stat(alias).st_ino,
        "selected_event_budget_delta": 0,
        "scientific_contract_changed": False,
        "weights_changed": False,
    }
    atomic_json(Path(audit_path), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--audit-json", required=True)
    args = parser.parse_args()
    repair_alias(
        v2_root=Path(args.v2_root),
        group_id=args.group_id,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        audit_path=Path(args.audit_json),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
