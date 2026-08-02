"""Frozen independent-validation request construction for Stage5."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUEST_SCHEMA = "stage5_independent_validation_request_v1"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def build_validation_request(
    task_root: Path,
    closure: Mapping[str, Any],
    *,
    fallback_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    task_id = str(closure.get("task_id") or "")
    task_sha = str(closure.get("task_sha256") or "")
    selected = [str(value) for value in closure.get("independent_validation_ids") or []]
    if not task_id or len(task_sha) != 64 or not 1 <= len(selected) <= 4:
        raise ValueError("closure independent-validation selection is incomplete")
    if len(set(selected)) != len(selected):
        raise ValueError("closure independent-validation selection contains duplicates")

    formal_rows: dict[str, Mapping[str, Any]] = {}
    for path in sorted(task_root.glob("round_*/measurement_request.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows") or []:
            row_id = _row_id(row)
            if row_id:
                formal_rows[row_id] = row
    for row_id, row in (fallback_rows or {}).items():
        actual_id = _row_id(row)
        if actual_id != row_id:
            raise ValueError(f"fallback row identity mismatch: {row_id}")
        if row_id not in formal_rows:
            formal_rows[row_id] = row
    missing = [row_id for row_id in selected if row_id not in formal_rows]
    if missing:
        raise ValueError(
            f"independent-validation IDs not found in formal or cold-start requests: {missing}"
        )
    rows = [copy.deepcopy(dict(formal_rows[row_id])) for row_id in selected]
    if any(
        row.get("task_id") != task_id or row.get("task_sha256") != task_sha
        for row in rows
    ):
        raise ValueError("independent-validation row task identity drift")
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA,
        "task_id": task_id,
        "task_sha256": task_sha,
        "batch_size": len(rows),
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "real_h800_measurement_required": True,
        "independent_from_search_measurement": True,
        "rows": rows,
        "row_sha256": {
            _row_id(row): _sha(row)
            for row in rows
        },
    }
    request["measurement_request_sha256"] = _sha(request)
    return request


__all__ = ["REQUEST_SCHEMA", "build_validation_request"]
