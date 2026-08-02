#!/usr/bin/env python3
"""Import immutable Pyramid round-0 evidence into an actual-feedback v3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage5_promote_actual_feedback_v3 import promote_feedback_batch  # noqa: E402


TASKS = ("S5-PYR-TVM", "S5-PYR-TRT")
ROUND_FILES = (
    "predicted_candidates.json",
    "acquisition.json",
    "measurement_request.json",
)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != content:
        raise ValueError(f"refusing to overwrite drifted v3 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _copy_immutable(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ValueError(f"missing v2 round-0 artifact: {source}")
    if target.is_file():
        if target.read_bytes() != source.read_bytes():
            raise ValueError(f"drifted imported artifact: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def initialize_task(v2_root: Path, v3_root: Path, task_id: str) -> dict[str, Any]:
    old_task = v2_root / task_id
    new_task = v3_root / task_id
    for name in ("task_contract.json", "candidate_manifest.json"):
        _copy_immutable(old_task / name, new_task / name)
    for name in ROUND_FILES:
        _copy_immutable(old_task / "round_00" / name, new_task / "round_00" / name)
    old_request = old_task / "round_00/measurement_request.json"
    old_feedback = old_task / "round_00/final/stage5_feedback_v2_final.json"
    result = promote_feedback_batch(old_request, old_feedback)
    actual_dir = new_task / "round_00/actual_feedback"
    actual_path = actual_dir / "stage5_feedback_v3_actual.json"
    audit_path = actual_dir / "actual_feedback_batch_audit_v3.json"
    _write(actual_path, result["rows"])
    _write(audit_path, result["audit"])
    history_path = new_task / "feedback_history_through_round_00.json"
    _write(history_path, result["rows"])
    audit = {
        "schema_version": "stage5_round0_actual_feedback_import_audit_v3",
        "task_id": task_id,
        "source_v2_root": str(v2_root),
        "target_v3_root": str(v3_root),
        "measurement_request_sha256": _sha_file(old_request),
        "historical_feedback_sha256": _sha_file(old_feedback),
        "actual_feedback_sha256": _sha_file(actual_path),
        "actual_feedback_audit_sha256": _sha_file(audit_path),
        "row_count": len(result["rows"]),
        "silent_surrogate_fallback_count": result["audit"][
            "silent_surrogate_fallback_count"
        ],
        "imported_measurements_reexecuted": False,
        "v2_rounds_after_zero_imported": False,
    }
    _write(new_task / "round_00/round0_import_audit_v3.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audits = [initialize_task(args.v2_root, args.v3_root, task) for task in TASKS]
    summary = {
        "schema_version": "stage5_pyramid_actual_v3_initialization_summary",
        "task_count": len(audits),
        "imported_round0_rows": sum(int(row["row_count"]) for row in audits),
        "silent_surrogate_fallback_count": sum(
            int(row["silent_surrogate_fallback_count"]) for row in audits
        ),
        "tasks": audits,
    }
    _write(args.v3_root / "pyramid_actual_v3_initialization_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
