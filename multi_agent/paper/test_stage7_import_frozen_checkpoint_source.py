from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import stage7_import_frozen_checkpoint_source as module


def test_authenticate_request_rebinds_row_and_request_hashes() -> None:
    rows = [{"row_id": f"row-{index}", "value": index} for index in range(4)]
    request = {
        "schema_version": "stage5_measurement_request_v2",
        "rows": rows,
        "row_sha256": {"stale": "value"},
        "measurement_request_sha256": "stale",
    }

    authenticated = module.authenticate_request(request)

    assert authenticated["row_sha256"] == {
        row["row_id"]: module.canonical_sha256(row) for row in rows
    }
    unsigned = dict(authenticated)
    recorded = unsigned.pop("measurement_request_sha256")
    assert recorded == module.canonical_sha256(unsigned)
    assert request["row_sha256"] == {"stale": "value"}


def test_copy_immutable_verifies_source_and_refuses_destination_conflict(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    destination = tmp_path / "nested/destination.bin"

    module.copy_immutable(source, destination, digest)
    module.copy_immutable(source, destination, digest)

    assert destination.read_bytes() == b"trusted"
    destination.write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="destination conflict"):
        module.copy_immutable(source, destination, digest)


def test_evidence_valid_authenticates_all_bound_files(tmp_path: Path) -> None:
    fields = {}
    for name in ("checkpoint", "onnx", "calibration", "summary"):
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        fields[name] = path
    evidence = {
        "schema_version": "stage5_source_materialization_evidence_v1",
        "group_id": "pyramid|24x32x64",
        "source_plan_sha256": "a" * 64,
        "status": "ready",
        "checkpoint_path": str(fields["checkpoint"]),
        "checkpoint_sha256": module.file_sha256(fields["checkpoint"]),
        "onnx_path": str(fields["onnx"]),
        "onnx_sha256": module.file_sha256(fields["onnx"]),
        "calibration_path": str(fields["calibration"]),
        "calibration_sha256": module.file_sha256(fields["calibration"]),
        "calibration_summary_path": str(fields["summary"]),
        "calibration_summary_sha256": module.file_sha256(fields["summary"]),
    }
    evidence_path = tmp_path / "source_ready_evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    assert module.evidence_valid(
        evidence_path, "pyramid|24x32x64", "a" * 64
    )
    fields["onnx"].write_bytes(b"drift")
    assert not module.evidence_valid(
        evidence_path, "pyramid|24x32x64", "a" * 64
    )


def test_source_plan_sha_changes_when_checkpoint_binding_changes() -> None:
    row = {
        "materialization_kind": "pyramid_checkpoint_export",
        "width": [24, 32, 64],
        "source_contract": {
            "checkpoint_path": "/formal/source/stage5_best.pth",
            "checkpoint_sha256": None,
        },
    }

    canonical = module.source_plan_sha256(row)
    rebound = module.source_plan_sha256(
        {
            **row,
            "source_contract": {
                **row["source_contract"],
                "checkpoint_sha256": "b" * 64,
            },
        }
    )

    assert module.is_sha256(canonical)
    assert module.is_sha256(rebound)
    assert canonical != rebound


def test_export_checkpoint_alias_has_parseable_epoch_name() -> None:
    canonical_checkpoint = Path("/formal/source/checkpoint/stage5_best.pth")

    export_checkpoint = canonical_checkpoint.with_name(
        module.EXPORT_CHECKPOINT_NAME
    )

    assert export_checkpoint.name == "net_epoch1.pth"
    assert canonical_checkpoint.name == "stage5_best.pth"


def test_export_checkpoint_alias_is_selected_by_frozen_ap_bridge(
    tmp_path: Path,
) -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "stage2_h800_true_fp16_ap_eval.py"
    )
    spec = importlib.util.spec_from_file_location("_stage7_ap_bridge_test", module_path)
    assert spec is not None and spec.loader is not None
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    alias = tmp_path / module.EXPORT_CHECKPOINT_NAME
    alias.write_bytes(b"checkpoint")

    assert bridge.best_checkpoint(tmp_path) == alias
    assert bridge.checkpoint_epoch(alias) == 1
