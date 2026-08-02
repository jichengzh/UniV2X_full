#!/usr/bin/env python3
"""Recoverably archive diagnostic bytecode accidentally written into Stage7."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: archive.py ROOT TIMESTAMP")
    root = Path(sys.argv[1]).resolve()
    timestamp = sys.argv[2]
    code_root = root / "deployment" / "code"
    audit_root = root / "audits" / "speed_priority_recovery"
    archive_root = audit_root / f"diagnostic_pycache_{timestamp}"
    targets = (
        code_root / "framework" / "__pycache__",
        code_root / "framework" / "stage5" / "__pycache__",
        code_root / "framework" / "stage7" / "__pycache__",
    )
    existing = tuple(path for path in targets if path.is_dir())
    if not existing:
        raise SystemExit("no expected diagnostic __pycache__ directories found")
    archive_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for source in existing:
        files = tuple(sorted(path for path in source.rglob("*") if path.is_file()))
        record = {
            "source": str(source),
            "relative_source": str(source.relative_to(code_root)),
            "files": [
                {
                    "relative_path": str(path.relative_to(source)),
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
                for path in files
            ],
        }
        destination = archive_root / source.relative_to(code_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        records.append({**record, "archive": str(destination)})
    audit = {
        "schema_version": "stage7_diagnostic_pycache_archive_v1",
        "wall_time": datetime.now().astimezone().isoformat(),
        "reason": "read_only_diagnostic_import_created_unlisted_bytecode",
        "budget_delta": 0,
        "deployment_sources_modified": False,
        "records": records,
    }
    audit_path = audit_root / f"diagnostic_pycache_archive_{timestamp}.json"
    temporary = audit_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, audit_path)
    directory_fd = os.open(audit_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({"audit": str(audit_path), "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
