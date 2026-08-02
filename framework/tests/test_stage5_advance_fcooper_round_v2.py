from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage5_advance_fcooper_round_v2 import (
    validate_atomic_release,
    validate_formal_feedback_evidence,
)


def _feedback(
    width: list[int],
    recovery_sha: str | None,
    recovery_path: Path | None = None,
    checkpoint_sha: str = "e" * 64,
) -> dict:
    graph = {
        "group_id": "fcooper|test",
        "model": "fcooper",
        "graph_feature_provenance": "materialized_onnx_extracted_v1",
    }
    row = {
        "group_id": "fcooper|test",
        "model": "fcooper",
        "width": width,
        "terminal_status": "measured_success_gold",
        "graph_features": graph,
        "materialized_graph_features_sha256": hashlib.sha256(
            json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "performance_result_sha256": "a" * 64,
        "ap_report_sha256": "b" * 64,
        "materialized_source_evidence_sha256": "c" * 64,
        "engine_sha256": "d" * 64,
        "checkpoint_sha256": checkpoint_sha,
        "recovery_training_report_sha256": recovery_sha,
        "recovery_training_report_path": (
            str(recovery_path) if recovery_path is not None else None
        ),
    }
    row["actual_feedback_row_sha256"] = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return row


def test_pruned_formal_feedback_requires_recovery_training_sha() -> None:
    with pytest.raises(ValueError, match="recovery-training SHA"):
        validate_formal_feedback_evidence(
            [_feedback([32, 64, 128, 64, 128], None)]
        )


def test_original_formal_feedback_does_not_require_recovery_training() -> None:
    audit = validate_formal_feedback_evidence(
        [_feedback([64, 128, 256, 128, 256], None)]
    )

    assert audit["recovered_pruned_rows"] == 0
    assert audit["prefix_only_measurement_rows"] == 0


def test_pruned_formal_feedback_verifies_recovery_report_file(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    config_path = tmp_path / "config.yaml"
    initial_path = tmp_path / "initial.pth"
    recovered_path = tmp_path / "recovered.pth"
    config_path.write_text("model: fcooper\n")
    initial_path.write_bytes(b"initial")
    recovered_path.write_bytes(b"recovered")
    contract = {
        "seed": 20260723,
        "start_epoch": 23,
        "minimum_epochs": 4,
        "recovery_epochs": 8,
        "amp_fp16": True,
    }
    contract_path.write_text(json.dumps(contract))
    epoch_records = []
    for epoch in range(24, 28):
        checkpoint = tmp_path / f"net_epoch{epoch}.pth"
        checkpoint.write_bytes(f"epoch-{epoch}".encode())
        epoch_records.append(
            {
                "epoch": epoch,
                "train_loss": 1.0,
                "validation_loss": 0.9,
                "elapsed_seconds": 1.0,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
        )
    report_path = tmp_path / "recovery.json"
    report = {
        "schema_version": "fcooper_recovery_training_report_v2",
        "status": "success",
        "initialization_policy": "scanner_dependency_l1_v2",
        "recovery_contract_path": str(contract_path),
        "recovery_contract_sha256": hashlib.sha256(
            contract_path.read_bytes()
        ).hexdigest(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "initial_checkpoint_path": str(initial_path),
        "initial_checkpoint_sha256": hashlib.sha256(
            initial_path.read_bytes()
        ).hexdigest(),
        "recovered_checkpoint_path": str(recovered_path),
        "epochs_completed": 4,
        "recovered_checkpoint_sha256": hashlib.sha256(
            recovered_path.read_bytes()
        ).hexdigest(),
        "seed": 20260723,
        "amp_fp16": True,
        "dataset": {
            "train_root": "/data/train",
            "validation_root": "/data/val",
            "train_samples": 10,
            "validation_samples": 5,
            "full_train_split": True,
            "full_validation_split": True,
        },
        "epoch_records": epoch_records,
        "elapsed_seconds": 4.0,
    }
    report_path.write_text(json.dumps(report))
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()

    audit = validate_formal_feedback_evidence(
        [
            _feedback(
                [32, 64, 128, 64, 128],
                report_sha,
                report_path,
                checkpoint_sha=report["recovered_checkpoint_sha256"],
            )
        ]
    )

    assert audit["recovered_pruned_rows"] == 1


def test_atomic_release_requires_released_exact_latest_batch() -> None:
    feedback = [
        {
            "row_id": f"row-{index}",
            "manifest_job_id": f"row-{index}",
            "round_index": 0,
        }
        for index in range(4)
    ]
    audit = {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": False,
        "batch_quarantined": True,
        "released_feedback_rows": [],
    }
    with pytest.raises(ValueError, match="not released"):
        validate_atomic_release(audit, feedback, completed_rounds=1)
