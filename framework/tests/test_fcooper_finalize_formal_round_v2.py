from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import fcooper_finalize_formal_round_v2 as finalizer


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "scripts/fcooper_formal_t16_supervisor_v2.sh"


def _request() -> dict:
    return {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S5-FCO-TRT-V2",
        "round_index": 0,
        "batch_size": 4,
        "rows": [
            {"row_id": f"requested-{index}", "task_id": "S5-FCO-TRT-V2"}
            for index in range(4)
        ],
    }


def _write_feedback(root: Path, directory: str, row_id: str) -> None:
    target = root / "execution" / directory / "feedback_row.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps({"row_id": row_id, "task_id": "S5-FCO-TRT-V2"}),
        encoding="utf-8",
    )


def test_collection_uses_request_identity_and_request_order(tmp_path: Path) -> None:
    request = _request()
    for index in (2, 0, 3, 1):
        _write_feedback(tmp_path, f"match-{index}", f"requested-{index}")
    _write_feedback(tmp_path, "stale-other-round", "not-requested")

    rows = finalizer.collect_requested_feedback(request, tmp_path)

    assert [row["row_id"] for row in rows] == [
        "requested-0",
        "requested-1",
        "requested-2",
        "requested-3",
    ]


def test_collection_rejects_duplicate_matching_identity(tmp_path: Path) -> None:
    request = _request()
    for index in range(4):
        _write_feedback(tmp_path, f"match-{index}", f"requested-{index}")
    _write_feedback(tmp_path, "duplicate", "requested-2")

    with pytest.raises(ValueError, match="duplicate feedback"):
        finalizer.collect_requested_feedback(request, tmp_path)


def test_collection_accepts_explicit_tvm_task_id(tmp_path: Path) -> None:
    request = _request()
    request["task_id"] = "S5-FCO-TVM-V1"
    for row in request["rows"]:
        row["task_id"] = "S5-FCO-TVM-V1"
    for index in range(4):
        target = tmp_path / "execution" / f"match-{index}" / "feedback_row.json"
        target.parent.mkdir(parents=True)
        target.write_text(
            json.dumps(
                {
                    "row_id": f"requested-{index}",
                    "task_id": "S5-FCO-TVM-V1",
                }
            )
        )

    rows = finalizer.collect_requested_feedback(
        request,
        tmp_path,
        task_id="S5-FCO-TVM-V1",
    )

    assert len(rows) == 4
    assert {row["task_id"] for row in rows} == {"S5-FCO-TVM-V1"}


def test_release_validates_before_atomically_publishing_outputs(
    tmp_path: Path,
) -> None:
    request = _request()
    request_path = tmp_path / "measurement_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    for index in range(4):
        _write_feedback(tmp_path, f"match-{index}", f"requested-{index}")
    feedback_path = tmp_path / "round_00" / "actual_feedback.json"
    audit_path = tmp_path / "round_00" / "atomic_batch_audit.json"
    history_path = tmp_path / "feedback_history_through_round_00.json"

    with patch.object(
        finalizer,
        "validate_formal_feedback_evidence",
        side_effect=ValueError("bad formal evidence"),
    ):
        with pytest.raises(ValueError, match="bad formal evidence"):
            finalizer.finalize_round(
                request_json=request_path,
                artifact_root=tmp_path,
                round_feedback_json=feedback_path,
                atomic_audit_json=audit_path,
                history_output_json=history_path,
                history_input_json=None,
                round_index=0,
            )
    assert not feedback_path.exists()
    assert not audit_path.exists()
    assert not history_path.exists()

    released = {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": True,
        "batch_quarantined": False,
        "budget_consumed": 4,
        "released_feedback_rows": request["rows"],
    }
    with (
        patch.object(
            finalizer,
            "validate_formal_feedback_evidence",
            return_value={"schema_version": "formal-evidence-test"},
        ),
        patch.object(finalizer, "finalize_atomic_batch", return_value=released),
    ):
        result = finalizer.finalize_round(
            request_json=request_path,
            artifact_root=tmp_path,
            round_feedback_json=feedback_path,
            atomic_audit_json=audit_path,
            history_output_json=history_path,
            history_input_json=None,
            round_index=0,
        )

    assert result["released_rows"] == 4
    released_rows = json.loads(feedback_path.read_text())["rows"]
    history_rows = json.loads(history_path.read_text())["rows"]
    assert [row["row_id"] for row in released_rows] == [
        row["row_id"] for row in request["rows"]
    ]
    assert [row["round_index"] for row in released_rows] == [0, 0, 0, 0]
    assert history_rows == released_rows
    assert json.loads(audit_path.read_text())["feedback_released"] is True


def test_round_binding_recomputes_actual_feedback_identity() -> None:
    row = {
        "row_id": "requested-0",
        "task_id": "S5-FCO-TRT-V2",
        "actual_feedback_row_sha256": "stale",
    }

    bound = finalizer.bind_feedback_round([row], round_index=2)[0]
    unhashed = {key: value for key, value in bound.items() if key != "actual_feedback_row_sha256"}
    expected = hashlib.sha256(
        json.dumps(
            unhashed, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    assert bound["round_index"] == 2
    assert bound["actual_feedback_row_sha256"] == expected


def _dry_run_args(formal_root: Path) -> list[str]:
    values = {
        "--formal-root": formal_root,
        "--gate-ready": formal_root / "gate.ready",
        "--stage4-dir": formal_root / "stage4",
        "--coldstart-root": formal_root / "coldstart",
        "--profiles-json": formal_root / "profiles.json",
        "--source-registry-json": formal_root / "registry.json",
        "--formal-contract-json": formal_root / "contract.json",
        "--probe-audit-json": formal_root / "probe.json",
        "--probe-isolation-audit-json": formal_root / "isolation.json",
        "--numeric-gate-summary-json": formal_root / "numeric.json",
        "--scanner-execution-json": formal_root / "scanner.json",
        "--artifact-root": formal_root / "artifacts",
        "--heal-root": formal_root / "heal",
        "--python": Path("/usr/bin/python3"),
        "--source-config": formal_root / "config.yaml",
        "--source-checkpoint": formal_root / "checkpoint.pth",
        "--recovery-contract": formal_root / "recovery.json",
        "--calibration-dir": formal_root / "calibration",
        "--calibration-summary": formal_root / "calibration.json",
    }
    args = [str(SUPERVISOR), "--dry-run"]
    for flag, value in values.items():
        args.extend([flag, str(value)])
    return args


def test_supervisor_rejects_pilot_root() -> None:
    pilot = Path("/tmp/fcooper_workpackage_a_20260723/formal")
    completed = subprocess.run(
        _dry_run_args(pilot),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "pilot path" in completed.stderr


def test_supervisor_dry_run_constructs_four_parallel_rows_per_round(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        _dry_run_args(tmp_path / "formal"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    row_lines = [
        line
        for line in completed.stdout.splitlines()
        if "fcooper_execute_measurement_row_v2.py" in line
    ]
    assert len(row_lines) == 16
    for round_index in range(4):
        round_lines = row_lines[round_index * 4 : (round_index + 1) * 4]
        assert [
            line.split("--gpu ", 1)[1].split()[0] for line in round_lines
        ] == ["1", "2", "3", "4"]
    assert completed.stdout.count("stage5_initialize_fcooper_actual_v2.py") == 1
    assert completed.stdout.count("stage5_advance_fcooper_round_v2.py") == 3
    assert completed.stdout.count("fcooper_close_formal_t16_v2.py") == 1
    advance_lines = [
        line
        for line in completed.stdout.splitlines()
        if "stage5_advance_fcooper_round_v2.py" in line
    ]
    for latest_round, line in enumerate(advance_lines):
        expected = (
            tmp_path
            / "formal"
            / f"round_{latest_round:02d}"
            / "atomic_batch_audit.json"
        )
        assert f"--atomic-audit-json {expected}" in line
