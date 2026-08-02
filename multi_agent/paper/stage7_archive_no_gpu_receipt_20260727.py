#!/usr/bin/env python3
"""Recoverably archive a deployment-bound Stage7 no-GPU receipt tree."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


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
    if len(sys.argv) != 4:
        raise SystemExit("usage: archive.py ROOT TAG EXPECTED_OLD_RELEASE")
    root = Path(sys.argv[1]).resolve(strict=True)
    tag = sys.argv[2]
    expected_release = sys.argv[3]
    source = root / "audits" / "no_gpu_dry_run"
    receipt = source / "dry_run_receipt.json"
    if not source.is_dir() or not receipt.is_file():
        raise SystemExit("no-GPU receipt tree is missing")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if value.get("deployment_release_sha256") != expected_release:
        raise SystemExit("old no-GPU receipt release SHA drift")
    archive = root / "audits" / f"no_gpu_dry_run_archived_{tag}"
    if archive.exists():
        raise SystemExit("no-GPU archive destination exists")
    files = [
        {
            "relative_path": str(path.relative_to(source)),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    os.replace(source, archive)
    fsync_directory(source.parent)
    audit_path = (
        root
        / "audits"
        / "speed_priority_recovery"
        / f"no_gpu_archive_{tag}.json"
    )
    atomic_json(
        audit_path,
        {
            "schema_version": "stage7_no_gpu_receipt_archive_v1",
            "wall_time": time.time(),
            "source": str(source),
            "archive": str(archive),
            "old_release_sha256": expected_release,
            "files": files,
            "formal_gpu_jobs_launched": 0,
            "selected_event_budget_delta": 0,
        },
    )
    print(json.dumps({"archive": str(archive), "audit": str(audit_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
