from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import hashlib
import json

from scripts.fcooper_close_formal_t16_v2 import (
    bind_success_evidence_files,
    validate_round_identity,
)
from scripts.stage6_finalize_fcooper_table1_v2 import _canonical_sha


ROOT = Path(__file__).resolve().parents[2]


def test_closure_script_resolves_repo_modules_outside_repo_cwd(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts/fcooper_close_formal_t16_v2.py"
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--formal-root",
            str(tmp_path / "formal"),
            "--contract-json",
            str(tmp_path / "contract.json"),
            "--probe-isolation-json",
            str(tmp_path / "isolation.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "No module named 'scripts'" not in completed.stderr
    assert "FileNotFoundError" in completed.stderr


def _request(round_index: int) -> dict:
    rows = [
        {
            "row_id": f"r{round_index}-{index}",
            "manifest_job_id": f"r{round_index}-{index}",
            "task_id": "S5-FCO-TRT-V2",
        }
        for index in range(4)
    ]
    return {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S5-FCO-TRT-V2",
        "round_index": round_index,
        "rows": rows,
    }


def test_round_identity_requires_exact_requested_released_rows() -> None:
    request = _request(2)
    feedback = [
        {**row, "round_index": 2, "training_source": "online_feedback"}
        for row in request["rows"]
    ]
    audit = {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": True,
        "batch_quarantined": False,
        "budget_consumed": 4,
        "released_feedback_rows": feedback,
    }

    assert validate_round_identity(request, feedback, audit, round_index=2) == {
        "r2-0",
        "r2-1",
        "r2-2",
        "r2-3",
    }

    audit["feedback_released"] = False
    with pytest.raises(ValueError, match="atomic"):
        validate_round_identity(request, feedback, audit, round_index=2)


def test_round_identity_rejects_cross_round_feedback() -> None:
    request = _request(1)
    feedback = [
        {**row, "round_index": 0, "training_source": "online_feedback"}
        for row in request["rows"]
    ]
    audit = {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": True,
        "batch_quarantined": False,
        "budget_consumed": 4,
        "released_feedback_rows": feedback,
    }

    with pytest.raises(ValueError, match="round"):
        validate_round_identity(request, feedback, audit, round_index=1)


def test_t16_closure_binds_underlying_success_evidence_bytes(
    tmp_path: Path,
) -> None:
    paths = {
        "performance_result_json": tmp_path / "performance.json",
        "ap_report_path": tmp_path / "ap.json",
        "engine_path": tmp_path / "compiled.engine",
        "checkpoint_path": tmp_path / "checkpoint.pth",
    }
    for field, path in paths.items():
        path.write_bytes(field.encode())
    source_evidence = {
        "checkpoint_path": str(paths["checkpoint_path"]),
        "checkpoint_sha256": hashlib.sha256(
            paths["checkpoint_path"].read_bytes()
        ).hexdigest(),
    }
    source_path = tmp_path / "source_evidence.json"
    source_path.write_text(json.dumps(source_evidence))
    graph = {"onnx_sha256": "a" * 64, "node_count": 1}
    row = {
        "row_id": "original",
        "manifest_job_id": "original",
        "terminal_status": "measured_success_gold",
        "width": [64, 128, 256, 128, 256],
        "q_mode": "fp16",
        "latency_ms": 1.0,
        "energy_j": 0.2,
        "ap30": 0.8,
        "ap50": 0.7,
        "ap70": 0.6,
        "graph_features": graph,
        "materialized_graph_features_sha256": _canonical_sha(graph),
        "materialized_source_evidence_path": str(source_path),
        "materialized_source_evidence_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": source_evidence["checkpoint_sha256"],
        "recovery_training_report_sha256": None,
    }
    for path_field in ("performance_result_json", "ap_report_path", "engine_path"):
        row[path_field] = str(paths[path_field])
        row[path_field.replace("_path", "_sha256").replace("_json", "_sha256")] = (
            hashlib.sha256(paths[path_field].read_bytes()).hexdigest()
        )
    row["performance_result_sha256"] = hashlib.sha256(
        paths["performance_result_json"].read_bytes()
    ).hexdigest()
    row["actual_feedback_row_sha256"] = _canonical_sha(row)

    audit, evidence = bind_success_evidence_files([row], root=tmp_path)

    assert audit["verified_success_rows"] == 1
    assert paths["engine_path"] in evidence
    paths["engine_path"].write_bytes(b"drift")
    with pytest.raises(ValueError, match="SHA"):
        bind_success_evidence_files([row], root=tmp_path)
