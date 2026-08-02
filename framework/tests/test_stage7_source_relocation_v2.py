from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage5.measurement_plan_v2 import _validate_request
from framework.stage7.source_relocation_v2 import (
    relocate_pyramid_request_to_v2_root,
    validate_source_relocation,
)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _source_contract(root: Path, width_tag: str) -> dict[str, Any]:
    source_root = root / width_tag
    return {
        "base_checkpoint_dir": "/frozen/stage1/base",
        "base_checkpoint_path": "/frozen/stage1/base/net_epoch_bestval_at23.pth",
        "base_checkpoint_sha256": "4ccc6fe1f7cc13b5d1294f74014b01e849cc8e90b69cbded14158e1999fa42b2",
        "checkpoint_dir": str(source_root / "checkpoint"),
        "checkpoint_path": str(source_root / "checkpoint/stage5_best.pth"),
        "checkpoint_sha256": "c" * 64,
        "config_path": str(source_root / "checkpoint/config.yaml"),
        "training_done_marker": str(
            source_root / "checkpoint/stage5_training_complete.json"
        ),
        "training_epoches": 31,
        "training_required": True,
        "width_per_group": 4,
        "groups": 32,
        "onnx_path": str(source_root / "onnx/model.onnx"),
        "onnx_sha256": "d" * 64,
        "onnx_report_path": str(source_root / "onnx/report.json"),
        "calibration_root": str(source_root / "calibration"),
        "calibration_npz": str(source_root / "calibration/input.npz"),
        "calibration_summary": str(source_root / "calibration/summary.json"),
        "trt_calibration_dir": str(source_root / "calibration/trt_npy"),
        "source_done_marker": str(source_root / "source.done"),
    }


def _request(source_root: Path) -> dict[str, Any]:
    specs = (
        ([16, 32, 64], "fp16"),
        ([16, 32, 64], "int8"),
        ([40, 32, 96], "int8"),
        ([16, 64, 224], "int8"),
    )
    rows = []
    for index, (width, q_mode) in enumerate(specs):
        width_tag = "x".join(map(str, width))
        candidate_id = (
            f"pyramid|{width_tag}|q={q_mode}|"
            "profile=h800-tvm-probe-conditioned-v3"
        )
        row = {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": candidate_id,
            "manifest_job_id": candidate_id,
            "strategy_id": f"candidate-{index}",
            "model": "pyramid",
            "group_id": f"pyramid|{width_tag}",
            "width": width,
            "genome": [*width, q_mode],
            "q_mode": q_mode,
            "hardware_id": "h800",
            "capability_profile_id": "h800-tvm-probe-conditioned-v3",
            "capability_digest": "b" * 64,
            "dispatch_key": "tvm_auto",
            "source_status": "planned",
            "materialization_kind": "pyramid_checkpoint_export",
            "source_contract": _source_contract(source_root, width_tag),
            "graph_features": {"conv_count": 51},
        }
        row["source_evidence_sha256"] = _source_plan_sha(row)
        rows.append(row)
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "a" * 64,
        "round_index": 0,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": [
            "latency_ms",
            "energy_j",
            "ap30",
            "ap50",
            "ap70",
        ],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _resign_request(request: dict[str, Any]) -> None:
    for row in request["rows"]:
        row["source_evidence_sha256"] = _source_plan_sha(row)
    request["row_sha256"] = {
        row["row_id"]: _sha(row) for row in request["rows"]
    }
    unsigned = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request["measurement_request_sha256"] = _sha(unsigned)


def _resign_audit(result: dict[str, Any]) -> None:
    audit = result["audit"]
    unsigned = {
        key: value
        for key, value in audit.items()
        if key != "source_relocation_sha256"
    }
    audit["source_relocation_sha256"] = _sha(unsigned)


def test_relocation_confines_outputs_and_preserves_selected_identity(
    tmp_path: Path,
) -> None:
    v2_root = (tmp_path / "formal_v2").resolve()
    request = _request(tmp_path / "historical_stage5")
    original = copy.deepcopy(request)
    ordered_ids = [row["row_id"] for row in request["rows"]]

    result = relocate_pyramid_request_to_v2_root(request, v2_root=v2_root)

    relocated = result["request"]
    audit = result["audit"]
    assert request == original
    assert [row["row_id"] for row in relocated["rows"]] == ordered_ids
    assert relocated["measurement_request_sha256"] != original[
        "measurement_request_sha256"
    ]
    assert _validate_request(relocated) == relocated["rows"]
    assert audit["ordered_candidate_ids"] == ordered_ids
    assert audit["ordered_candidate_ids_sha256"] == _sha(ordered_ids)
    assert audit["pre_relocation_request_sha256"] == original[
        "measurement_request_sha256"
    ]
    assert audit["relocated_request_sha256"] == relocated[
        "measurement_request_sha256"
    ]
    assert audit["cache_or_label_fields_observed"] == []
    assert audit["selected_ids_changed"] is False

    expected = (
        v2_root
        / "sources/pyramid/016x032x064"
    )
    first = relocated["rows"][0]["source_contract"]
    assert first["checkpoint_dir"] == str(expected / "checkpoint")
    assert first["checkpoint_path"] == str(
        expected / "checkpoint/stage5_best.pth"
    )
    assert first["config_path"] == str(expected / "checkpoint/config.yaml")
    assert first["training_done_marker"] == str(
        expected / "checkpoint/stage5_training_complete.json"
    )
    assert first["onnx_path"] == str(
        expected / "onnx/pyramid_016x032x064_multiscale.onnx"
    )
    assert first["onnx_report_path"] == str(
        expected / "onnx/onnx_export_report.json"
    )
    assert first["calibration_npz"] == str(
        expected / "calibration/spatial_features_train16.npz"
    )
    assert first["calibration_summary"] == str(
        expected / "calibration/summary.json"
    )
    assert first["trt_calibration_dir"] == str(
        expected / "calibration/trt_npy"
    )
    assert first["source_done_marker"] == str(expected / "source_ready.done")
    assert first["checkpoint_sha256"] is None
    assert first["onnx_sha256"] is None
    assert first["base_checkpoint_path"] == original["rows"][0][
        "source_contract"
    ]["base_checkpoint_path"]
    assert first["training_epoches"] == 31


def test_relocation_deduplicates_same_width_without_cross_width_aliasing(
    tmp_path: Path,
) -> None:
    v2_root = (tmp_path / "formal_v2").resolve()

    relocated = relocate_pyramid_request_to_v2_root(
        _request(tmp_path / "old"),
        v2_root=v2_root,
    )["request"]

    first, second, third, _ = relocated["rows"]
    assert first["source_contract"] == second["source_contract"]
    assert first["source_evidence_sha256"] == second["source_evidence_sha256"]
    assert first["source_contract"] != third["source_contract"]
    assert first["source_evidence_sha256"] != third["source_evidence_sha256"]


def test_relocation_rejects_unsafe_root_and_non_pyramid_request(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path / "old")
    with pytest.raises(ValueError, match="absolute"):
        relocate_pyramid_request_to_v2_root(
            request,
            v2_root=Path("relative-v2-root"),
        )

    request["rows"][0]["model"] = "codriving"
    with pytest.raises(ValueError, match="Pyramid"):
        relocate_pyramid_request_to_v2_root(
            request,
            v2_root=(tmp_path / "formal_v2").resolve(),
        )


def test_relocation_validator_rejects_path_escape_and_identity_drift(
    tmp_path: Path,
) -> None:
    v2_root = (tmp_path / "formal_v2").resolve()
    result = relocate_pyramid_request_to_v2_root(
        _request(tmp_path / "old"),
        v2_root=v2_root,
    )

    escaped = copy.deepcopy(result)
    escaped["request"]["rows"][0]["source_contract"]["onnx_path"] = (
        "/tmp/escaped.onnx"
    )
    _resign_request(escaped["request"])
    escaped["audit"]["relocated_request_sha256"] = escaped["request"][
        "measurement_request_sha256"
    ]
    _resign_audit(escaped)
    with pytest.raises(ValueError, match="v2 source root"):
        validate_source_relocation(escaped, v2_root=v2_root)

    drifted = copy.deepcopy(result)
    drifted["audit"]["ordered_candidate_ids"][0] = "different-candidate"
    _resign_audit(drifted)
    with pytest.raises(ValueError, match="selected"):
        validate_source_relocation(drifted, v2_root=v2_root)
