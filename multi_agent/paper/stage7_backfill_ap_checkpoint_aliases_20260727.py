#!/usr/bin/env python3
"""Backfill immutable AP-compatible aliases for verified Stage7 checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path


ALIAS_NAME = "net_epoch1.pth"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_immutable(source: Path, destination: Path) -> str:
    digest = file_sha256(source)
    if destination.exists():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or file_sha256(destination) != digest
        ):
            raise RuntimeError(f"checkpoint alias conflict: {destination}")
        return "already_present"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        if file_sha256(temporary) != digest:
            raise RuntimeError("checkpoint alias copy SHA drift")
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return "created"


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: backfill.py ROOT AUDIT_PATH")
    root = Path(sys.argv[1]).resolve(strict=True)
    audit_path = Path(sys.argv[2]).resolve()
    if audit_path.exists():
        raise SystemExit("audit path already exists")
    source_root = root / "sources" / "pyramid"
    checkpoints = tuple(sorted(source_root.glob("*/checkpoint/stage5_best.pth")))
    if not checkpoints:
        raise SystemExit("no canonical Stage7 checkpoints found")
    records: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise RuntimeError(f"canonical checkpoint is not a plain file: {checkpoint}")
        alias = checkpoint.with_name(ALIAS_NAME)
        action = copy_immutable(checkpoint, alias)
        digest = file_sha256(checkpoint)
        records.append(
            {
                "group_directory": checkpoint.parents[1].name,
                "canonical_checkpoint": str(checkpoint),
                "canonical_sha256": digest,
                "alias_checkpoint": str(alias),
                "alias_sha256": file_sha256(alias),
                "action": action,
            }
        )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        audit_path,
        {
            "schema_version": "stage7_ap_checkpoint_alias_backfill_v1",
            "wall_time": time.time(),
            "root": str(root),
            "alias_name": ALIAS_NAME,
            "records": records,
            "canonical_source_evidence_changed": False,
            "selected_ids_changed": False,
            "measurement_contract_changed": False,
            "selected_event_budget_delta": 0,
        },
    )
    print(
        json.dumps(
            {
                "audit": str(audit_path),
                "alias_count": len(records),
                "created": sum(row["action"] == "created" for row in records),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
