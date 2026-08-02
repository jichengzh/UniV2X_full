#!/usr/bin/env python3
"""Atomically replace one failed independent-validation configuration with a repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _backup(path: Path) -> Path:
    backup = path.with_name(path.name + ".pre_quant_contract_repair_v1")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _replace_by_id(
    rows: list[dict[str, Any]], replacement: dict[str, Any], config_id: str
) -> list[dict[str, Any]]:
    matches = [row for row in rows if row.get("manifest_job_id") == config_id]
    if len(matches) != 1 or replacement.get("manifest_job_id") != config_id:
        raise ValueError(f"expected one row for configuration: {config_id}")
    return [replacement if row.get("manifest_job_id") == config_id else row for row in rows]


def apply_repair(task_root: Path, repair_root: Path, config_id: str) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": "stage5_single_config_repair_audit_v1",
        "configuration_id": config_id,
        "files": [],
    }
    for repeat in range(3):
        target = task_root / f"repeat_{repeat}"
        source = repair_root / f"repeat_{repeat}"
        target_jobs = target / "performance_jobs.jsonl"
        source_jobs = _jsonl(source / "performance_jobs.jsonl")
        if len(source_jobs) != 1:
            raise ValueError("repair repeat must contain exactly one job")
        target_states = target / "performance_state.jsonl"
        source_states = _jsonl(source / "performance_state.jsonl")
        successful = [row for row in source_states if row.get("status") == "success"]
        if len(successful) != 1:
            raise ValueError("repair repeat must contain exactly one successful state")
        repair_job_id = str(source_jobs[0]["job_id"])
        merged_jobs = _replace_by_id(_jsonl(target_jobs), source_jobs[0], config_id)
        merged_states = [
            row for row in _jsonl(target_states) if row.get("job_id") != repair_job_id
        ] + source_states
        manifest_path = target / "performance_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        repair_manifest = json.loads((source / "performance_manifest.json").read_text())
        manifest["jobs"] = _replace_by_id(
            manifest["jobs"], repair_manifest["jobs"][0], config_id
        )
        for path, writer, payload in (
            (target_jobs, _write_jsonl, merged_jobs),
            (target_states, _write_jsonl, merged_states),
            (manifest_path, _write_json, manifest),
        ):
            backup = _backup(path)
            old_sha = _sha(path)
            writer(path, payload)
            audit["files"].append(
                {"path": str(path), "backup": str(backup), "old_sha256": old_sha, "new_sha256": _sha(path)}
            )
    target_ap = task_root / "ap/ap_state.jsonl"
    repair_ap_rows = _jsonl(repair_root / "ap/ap_state.jsonl")
    full_success = [
        row for row in repair_ap_rows
        if row.get("record_type") == "job_terminal"
        and row.get("stage") == "full"
        and row.get("status") == "success"
    ]
    if len(full_success) != 1 or full_success[0].get("job_id") != config_id:
        raise ValueError("repair AP must contain one successful full terminal")
    merged_ap = [row for row in _jsonl(target_ap) if row.get("job_id") != config_id]
    merged_ap.extend(repair_ap_rows)
    backup = _backup(target_ap)
    old_sha = _sha(target_ap)
    _write_jsonl(target_ap, merged_ap)
    audit["files"].append(
        {"path": str(target_ap), "backup": str(backup), "old_sha256": old_sha, "new_sha256": _sha(target_ap)}
    )
    audit["performance_repeat_count"] = 3
    audit["full_ap_repair_count"] = 1
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--configuration-id", required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    audit = apply_repair(args.task_root, args.repair_root, args.configuration_id)
    _write_json(args.output_audit, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
