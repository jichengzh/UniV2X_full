"""Byte-stable paper and audit artifact emission for Stage7 v2."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage7.ablation_statistics_v2 import paper_rows
from framework.stage7.formal_evidence_v2 import (
    FORMAL_OUTPUT_PATHS,
    SCHEMA_VERSION,
    TOTAL_EVENTS,
    TOTAL_TRAJECTORIES,
    JSON,
    fail,
)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for field, value in row.items()
            }
        )
    return stream.getvalue()


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(rows[0])
    lines = [
        "# Stage7 core component ablation (scanner deferred)",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(row[field]).replace("|", "/") for field in fields) + " |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def validate_output_root(root: Path, allowed: Sequence[str]) -> None:
    if not root.exists():
        return
    if not root.is_dir():
        fail("finalizer output root must be a directory")
    allowed_files = {Path(path) for path in allowed}
    allowed_dirs = {
        parent
        for path in allowed_files
        for parent in path.parents
        if parent != Path(".")
    }
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink() or (
            relative not in allowed_files and relative not in allowed_dirs
        ):
            fail(f"unmanaged finalizer output path: {relative}")


def emit_formal_outputs(
    root: Path,
    *,
    evidence: Mapping[str, Any],
    events: Sequence[JSON],
    trajectories: Sequence[JSON],
    stats: Mapping[str, Any],
) -> None:
    paper = paper_rows(stats)
    audit_bundle = {
        "schema_version": "stage7_core_audit_bundle_v2",
        "passed": True,
        "audits": copy.deepcopy(dict(evidence["audits"])),
        "bound_audit_artifacts": copy.deepcopy(
            evidence.get("bound_audit_artifacts") or []
        ),
        "trajectory_count": TOTAL_TRAJECTORIES,
        "selected_event_count": TOTAL_EVENTS,
        "miss_count": evidence["miss_count"],
        "actual_v3_miss_evidence_count": evidence["actual_v3_miss_evidence_count"],
        "silent_surrogate_fallback_count": 0,
        "formal_v2_gpu_jobs_launched": evidence["formal_v2_gpu_jobs_launched"],
    }
    status = {
        "schema_version": SCHEMA_VERSION,
        "paper_ready": True,
        "core_ablation_ready": True,
        "full_gear_s7_ready": False,
        "scanner_component_status": "deferred_important_fix",
        "formal_v2_gpu_jobs_launched": evidence["formal_v2_gpu_jobs_launched"],
        "blocking_reason": None,
    }
    writes = {
        FORMAL_OUTPUT_PATHS[0]: json_text(list(events)),
        FORMAL_OUTPUT_PATHS[1]: csv_text(events),
        FORMAL_OUTPUT_PATHS[2]: json_text(list(trajectories)),
        FORMAL_OUTPUT_PATHS[3]: csv_text(trajectories),
        FORMAL_OUTPUT_PATHS[4]: json_text(audit_bundle),
        FORMAL_OUTPUT_PATHS[5]: json_text(stats),
        FORMAL_OUTPUT_PATHS[6]: (
            "# Stage7 core-ablation root-cause summary\n\n"
            "- core_ablation_ready: `true`\n"
            "- full_gear_s7_ready: `false`\n"
            "- scanner_component_status: `deferred_important_fix`\n"
            "- incomplete or invalid formal evidence: none\n"
        ),
        FORMAL_OUTPUT_PATHS[7]: csv_text(paper),
        FORMAL_OUTPUT_PATHS[8]: markdown_table(paper),
        FORMAL_OUTPUT_PATHS[9]: json_text(status),
    }
    for relative, content in writes.items():
        atomic_write(root / relative, content)
