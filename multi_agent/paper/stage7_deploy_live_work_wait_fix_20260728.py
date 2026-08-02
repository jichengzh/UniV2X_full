#!/usr/bin/env python3
"""Atomically deploy the audited Stage7 live-work retry-wait fix."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_owned_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    if path.stat().st_uid != os.getuid():
        raise ValueError(f"{label} owner mismatch: {path}")


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"temporary audit path already exists: {temporary}")
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: deploy.py TARGET STAGING AUDIT_DIR EXPECTED_OLD EXPECTED_NEW"
        )
    target = Path(sys.argv[1]).resolve(strict=True)
    staging = Path(sys.argv[2]).resolve(strict=True)
    audit_dir = Path(sys.argv[3]).resolve(strict=True)
    expected_old, expected_new = sys.argv[4:6]
    _require_regular_owned_file(target, label="target")
    _require_regular_owned_file(staging, label="staging")
    if not audit_dir.is_dir() or audit_dir.is_symlink():
        raise ValueError(f"audit directory is missing or unsafe: {audit_dir}")
    old_sha256 = _sha256(target)
    new_sha256 = _sha256(staging)
    if old_sha256 != expected_old:
        raise ValueError(f"target SHA drift: {old_sha256}")
    if new_sha256 != expected_new:
        raise ValueError(f"staging SHA drift: {new_sha256}")
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    backup = audit_dir / f"stage7_speed_priority_takeover_v1.{timestamp}.before.py"
    audit = audit_dir / f"live_work_retry_wait_deployment_{timestamp}.json"
    if backup.exists() or backup.is_symlink() or audit.exists() or audit.is_symlink():
        raise ValueError("deployment audit target already exists")
    shutil.copy2(target, backup)
    os.replace(staging, target)
    deployed_sha256 = _sha256(target)
    if deployed_sha256 != expected_new:
        raise ValueError(f"deployed SHA mismatch: {deployed_sha256}")
    _atomic_json(
        audit,
        {
            "action": "atomic_runtime_wrapper_patch",
            "audit_path": str(audit),
            "backup_path": str(backup),
            "deployment_wall_time": time.time(),
            "new_sha256": deployed_sha256,
            "old_sha256": old_sha256,
            "preserves_formal_release": True,
            "scientific_identity_changed": False,
            "target_path": str(target),
        },
    )
    print(json.dumps({"audit": str(audit), "new_sha256": deployed_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
