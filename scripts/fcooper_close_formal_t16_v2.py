#!/usr/bin/env python3
"""Close the formal F-Cooper T16 trajectory and select its unrestricted winner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frozen(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to overwrite drifted T16 closure artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def validate_round_identity(
    request: Mapping[str, Any],
    feedback: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    *,
    round_index: int,
) -> set[str]:
    requested = {_row_id(row) for row in request.get("rows") or []}
    measured = {_row_id(row) for row in feedback}
    released = {_row_id(row) for row in audit.get("released_feedback_rows") or []}
    if (
        request.get("schema_version") != "stage5_measurement_request_v2"
        or request.get("task_id") != TASK_ID
        or int(request.get("round_index", -1)) != round_index
        or len(requested) != 4
        or measured != requested
        or released != requested
    ):
        raise ValueError(f"formal round {round_index} identity drift")
    if (
        audit.get("schema_version") != "stage5_atomic_batch_audit_v2"
        or audit.get("feedback_released") is not True
        or audit.get("batch_quarantined") is not False
        or int(audit.get("budget_consumed") or 0) != 4
    ):
        raise ValueError(f"formal round {round_index} atomic release is invalid")
    if any(
        int(row.get("round_index", -1)) != round_index
        or row.get("task_id") != TASK_ID
        or row.get("training_source") != "online_feedback"
        for row in feedback
    ):
        raise ValueError(f"formal round {round_index} feedback round/task drift")
    return requested


def bind_success_evidence_files(
    rows: Sequence[Mapping[str, Any]], *, root: Path
) -> tuple[dict[str, Any], list[Path]]:
    from scripts.stage6_finalize_fcooper_table1_v2 import verify_success_evidence

    row_audits: list[dict[str, Any]] = []
    evidence_paths: list[Path] = []
    for row in rows:
        if row.get("terminal_status") != "measured_success_gold":
            continue
        audit = verify_success_evidence(row, root=root)
        row_audits.append(audit)
        evidence_paths.extend(Path(path) for path in audit["verified_paths"])
    unique_paths = list(dict.fromkeys(path.resolve() for path in evidence_paths))
    return (
        {
            "schema_version": "fcooper_formal_t16_deep_evidence_audit_v2",
            "passed": True,
            "verified_success_rows": len(row_audits),
            "row_audits": row_audits,
            "file_count": len(unique_paths),
        },
        unique_paths,
    )


def close(
    *,
    formal_root: Path,
    contract_json: Path,
    probe_isolation_json: Path,
) -> dict[str, Any]:
    for path in (formal_root, contract_json, probe_isolation_json):
        if PILOT_FRAGMENT in str(path):
            raise ValueError(f"formal closure cannot use pilot path: {path}")
    from scripts.stage5_advance_fcooper_round_v2 import (
        validate_formal_feedback_evidence,
    )
    from scripts.stage6_finalize_fcooper_table1_v2 import select_gear_candidate

    isolation = _read(probe_isolation_json)
    if (
        isolation.get("status") != "passed"
        or isolation.get("probe_rows_allowed_as_winner") is not False
    ):
        raise ValueError("probe isolation is not valid for formal closure")
    probe_ids = set(isolation.get("probe_row_ids") or [])
    all_rows: list[dict[str, Any]] = []
    evidence_files: list[Path] = [contract_json, probe_isolation_json]
    seen: set[str] = set()
    for round_index in range(4):
        round_root = formal_root / f"round_{round_index:02d}"
        request_path = round_root / "measurement_request.json"
        feedback_path = round_root / "actual_feedback.json"
        audit_path = round_root / "atomic_batch_audit.json"
        request = _read(request_path)
        feedback_payload = _read(feedback_path)
        rows = [dict(row) for row in feedback_payload.get("rows") or []]
        audit = _read(audit_path)
        identities = validate_round_identity(
            request, rows, audit, round_index=round_index
        )
        if identities & seen or identities & probe_ids:
            raise ValueError("T16 rows overlap prior rounds or isolated probes")
        seen.update(identities)
        all_rows.extend(rows)
        evidence_files.extend((request_path, feedback_path, audit_path))
    if len(all_rows) != 16 or len(seen) != 16:
        raise ValueError("formal T16 closure requires exactly 16 unique rows")
    evidence_audit = validate_formal_feedback_evidence(all_rows)
    deep_evidence_audit, deep_evidence_files = bind_success_evidence_files(
        all_rows, root=formal_root
    )
    evidence_files.extend(deep_evidence_files)
    contract = _read(contract_json)
    winner = select_gear_candidate(
        all_rows,
        ap70_ref=float(contract["ap70_ref"]),
        max_ap_drop=0.10,
    )
    final_history = {
        "schema_version": "stage5_fcooper_formal_feedback_history_final_v2",
        "task_id": TASK_ID,
        "round_count": 4,
        "row_count": 16,
        "rows": all_rows,
    }
    history_path = formal_root / "feedback_history_final_t16.json"
    _write_frozen(history_path, final_history)
    evidence_files.append(history_path)
    integrity = {
        "schema_version": "fcooper_formal_t16_file_integrity_audit_v2",
        "passed": True,
        "row_count": 16,
        "deep_evidence_audit": deep_evidence_audit,
        "files": [
            {"path": str(path.resolve()), "sha256": _sha_file(path)}
            for path in evidence_files
        ],
    }
    online_integrity_path = formal_root / "online_file_integrity_audit.json"
    final_integrity_path = formal_root / "final_feedback_integrity_audit.json"
    _write_frozen(online_integrity_path, integrity)
    _write_frozen(final_integrity_path, integrity)
    winner_payload = {
        "schema_version": "fcooper_formal_t16_winner_v2",
        "task_id": TASK_ID,
        "selection_pool": "formal_t16_online_feedback_only",
        "probe_rows_used": False,
        "ap70_ref": float(contract["ap70_ref"]),
        "max_ap_drop": 0.10,
        "winner": winner,
    }
    winner_path = formal_root / "formal_t16_winner.json"
    _write_frozen(winner_path, winner_payload)
    closure = {
        "schema_version": "fcooper_formal_t16_closure_audit_v2",
        "task_id": TASK_ID,
        "status": "closed",
        "rounds": 4,
        "online_rows": 16,
        "probe_overlap_count": 0,
        "evidence_audit": evidence_audit,
        "deep_evidence_audit": deep_evidence_audit,
        "winner_row_id": _row_id(winner),
        "winner_terminal_status": winner["terminal_status"],
        "winner_path": str(winner_path.resolve()),
        "winner_sha256": _sha_file(winner_path),
    }
    _write_frozen(formal_root / "formal_t16_closure_audit.json", closure)
    return closure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--contract-json", type=Path, required=True)
    parser.add_argument("--probe-isolation-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            close(
                formal_root=args.formal_root,
                contract_json=args.contract_json,
                probe_isolation_json=args.probe_isolation_json,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
