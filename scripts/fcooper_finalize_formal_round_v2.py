#!/usr/bin/env python3
"""Collect and atomically release one formal F-Cooper T16 feedback round."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.single_target_search_v2 import (  # noqa: E402
    finalize_atomic_batch,
)
from scripts.stage5_advance_fcooper_round_v2 import (  # noqa: E402
    validate_formal_feedback_evidence,
)


TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _request_rows(
    request: Mapping[str, Any], *, task_id: str = TASK_ID
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in request.get("rows") or []]
    identities = [_row_id(row) for row in rows]
    if (
        request.get("schema_version") != "stage5_measurement_request_v2"
        or request.get("task_id") != task_id
        or request.get("batch_size") != 4
        or len(rows) != 4
        or any(not identity for identity in identities)
        or len(set(identities)) != 4
    ):
        raise ValueError("formal round requires exactly four unique requested rows")
    return rows


def collect_requested_feedback(
    request: Mapping[str, Any],
    artifact_root: Path,
    *,
    task_id: str = TASK_ID,
) -> list[dict[str, Any]]:
    """Return only feedback rows in the request, preserving request order."""
    request_rows = _request_rows(request, task_id=task_id)
    requested_ids = [_row_id(row) for row in request_rows]
    requested = set(requested_ids)
    matches: dict[str, dict[str, Any]] = {}
    for path in sorted(artifact_root.glob("execution/*/feedback_row.json")):
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        row = dict(payload)
        identity = _row_id(row)
        if identity not in requested:
            continue
        if identity in matches:
            raise ValueError(f"duplicate feedback for requested row: {identity}")
        matches[identity] = row
    missing = [identity for identity in requested_ids if identity not in matches]
    if missing:
        raise ValueError(f"missing requested feedback rows: {missing}")
    return [matches[identity] for identity in requested_ids]


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def bind_feedback_round(
    rows: Sequence[Mapping[str, Any]], *, round_index: int
) -> list[dict[str, Any]]:
    bound_rows = []
    for source in rows:
        existing = source.get("round_index")
        if existing is not None and int(existing) != round_index:
            raise ValueError("feedback row round index drift")
        row = {**dict(source), "round_index": round_index}
        if "actual_feedback_row_sha256" in row:
            row.pop("actual_feedback_row_sha256")
            row["actual_feedback_row_sha256"] = _sha_payload(row)
        bound_rows.append(row)
    return bound_rows


def _history_rows(
    history_input_json: Path | None,
    *,
    round_index: int,
    current_rows: Sequence[Mapping[str, Any]],
    task_id: str = TASK_ID,
) -> list[dict[str, Any]]:
    previous: list[dict[str, Any]] = []
    if history_input_json is not None:
        payload = _read_json(history_input_json)
        source = payload.get("rows") if isinstance(payload, Mapping) else payload
        if not isinstance(source, list):
            raise ValueError("cumulative feedback history must contain rows")
        previous = [dict(row) for row in source]
    if len(previous) != round_index * 4:
        raise ValueError("cumulative history must contain four rows per prior round")
    combined = [*previous, *[dict(row) for row in current_rows]]
    identities = [_row_id(row) for row in combined]
    if (
        len(combined) != (round_index + 1) * 4
        or any(not identity for identity in identities)
        or len(set(identities)) != len(identities)
        or any(row.get("task_id") != task_id for row in combined)
    ):
        raise ValueError("cumulative feedback history identity or task drift")
    return combined


def _stage_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _publish_json_batch(
    *,
    round_feedback_json: Path,
    feedback_payload: Mapping[str, Any],
    history_output_json: Path,
    history_payload: Mapping[str, Any],
    atomic_audit_json: Path,
    atomic_audit: Mapping[str, Any],
) -> None:
    outputs = (
        (round_feedback_json, feedback_payload),
        (history_output_json, history_payload),
        (atomic_audit_json, atomic_audit),
    )
    existing = [target.exists() for target, _ in outputs]
    if any(existing):
        if not all(existing) or any(
            _read_json(target) != payload for target, payload in outputs
        ):
            raise ValueError("refusing to overwrite drifted formal round release")
        return
    staged: list[tuple[Path, Path]] = []
    try:
        staged = [(target, _stage_json(target, payload)) for target, payload in outputs]
        # The atomic audit is the release marker and is intentionally published last.
        for target, temporary in staged:
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def finalize_round(
    *,
    request_json: Path,
    artifact_root: Path,
    round_feedback_json: Path,
    atomic_audit_json: Path,
    history_output_json: Path,
    history_input_json: Path | None,
    round_index: int,
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    if round_index not in range(4):
        raise ValueError("round index must be 0..3")
    for path in (
        request_json,
        artifact_root,
        round_feedback_json,
        atomic_audit_json,
        history_output_json,
        history_input_json,
    ):
        if path is not None and PILOT_FRAGMENT in str(path):
            raise ValueError(f"formal finalization cannot use pilot path: {path}")

    request = _read_json(request_json)
    if not isinstance(request, Mapping) or request.get("round_index") != round_index:
        raise ValueError("measurement request round mismatch")
    if request.get("task_id") != task_id:
        raise ValueError("measurement request task mismatch")
    rows = collect_requested_feedback(request, artifact_root, task_id=task_id)
    evidence_audit = validate_formal_feedback_evidence(rows)
    atomic_audit = finalize_atomic_batch(request, rows)
    if (
        atomic_audit.get("feedback_released") is not True
        or atomic_audit.get("batch_quarantined") is not False
        or atomic_audit.get("budget_consumed") != 4
    ):
        raise ValueError("formal atomic batch was not released")
    released_rows = bind_feedback_round(rows, round_index=round_index)
    history = _history_rows(
        history_input_json,
        round_index=round_index,
        current_rows=released_rows,
        task_id=task_id,
    )
    feedback_payload = {
        "schema_version": "stage5_fcooper_formal_actual_feedback_batch_v2",
        "task_id": task_id,
        "round_index": round_index,
        "row_count": 4,
        "evidence_audit": evidence_audit,
        "rows": released_rows,
    }
    history_payload = {
        "schema_version": "stage5_fcooper_formal_feedback_history_v2",
        "task_id": task_id,
        "through_round_index": round_index,
        "row_count": len(history),
        "rows": history,
    }
    _publish_json_batch(
        round_feedback_json=round_feedback_json,
        feedback_payload=feedback_payload,
        history_output_json=history_output_json,
        history_payload=history_payload,
        atomic_audit_json=atomic_audit_json,
        atomic_audit=atomic_audit,
    )
    return {
        "schema_version": "stage5_fcooper_formal_round_release_v2",
        "task_id": task_id,
        "round_index": round_index,
        "released_rows": 4,
        "cumulative_rows": len(history),
        "feedback_json": str(round_feedback_json),
        "atomic_audit_json": str(atomic_audit_json),
        "history_json": str(history_output_json),
    }


def _write_failure(
    path: Path,
    error: BaseException,
    started: float,
    *,
    task_id: str = TASK_ID,
) -> None:
    payload = {
        "schema_version": "stage5_fcooper_formal_round_failure_v2",
        "task_id": task_id,
        "status": "failed",
        "failure_class": "infrastructure_or_evidence_failure",
        "classified_as_feasibility": False,
        "budget_consumed": 0,
        "pid": os.getpid(),
        "elapsed_seconds": time.monotonic() - started,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc().splitlines()[-20:],
    }
    temporary = _stage_json(path, payload)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--round-feedback-json", type=Path, required=True)
    parser.add_argument("--atomic-audit-json", type=Path, required=True)
    parser.add_argument("--history-input-json", type=Path)
    parser.add_argument("--history-output-json", type=Path, required=True)
    parser.add_argument("--failure-evidence-json", type=Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    return parser.parse_args()


def _reject_pilot_cli_paths(args: argparse.Namespace) -> None:
    for value in vars(args).values():
        if isinstance(value, Path) and PILOT_FRAGMENT in str(value):
            raise ValueError(f"formal finalization cannot use pilot path: {value}")


def main() -> int:
    args = parse_args()
    try:
        _reject_pilot_cli_paths(args)
    except ValueError as error:
        print(f"ValueError: {error}", file=sys.stderr)
        return 2
    started = time.monotonic()
    try:
        result = finalize_round(
            request_json=args.request_json,
            artifact_root=args.artifact_root,
            round_feedback_json=args.round_feedback_json,
            atomic_audit_json=args.atomic_audit_json,
            history_output_json=args.history_output_json,
            history_input_json=args.history_input_json,
            round_index=args.round_index,
            task_id=args.task_id,
        )
    except Exception as error:
        _write_failure(
            args.failure_evidence_json,
            error,
            started,
            task_id=args.task_id,
        )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
