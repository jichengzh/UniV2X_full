import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage6_finalize_tvm_default_baselines_v1 import (
    build_codriving_audit,
    build_fcooper_audit,
    rebuild_three_model_rows,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_codriving_audit_requires_three_default_repeats_and_full_ap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repair"
    latency_root = tmp_path / "latency"
    artifact = root / "artifact/tvm_fp32_default.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"default-tvm")
    _write_json(
        root / "artifact/artifact_report.json",
        {
            "status": "success",
            "schedule_policy": "tvm_default_zero_trial",
            "tuning_trials": 0,
            "artifact_path": str(artifact),
            "artifact_sha256": _sha(artifact),
            "onnx_path": "/models/codriving.onnx",
            "onnx_sha256": "onnx-sha",
        },
    )
    for index, latency in enumerate((11.57, 11.58, 11.59)):
        _write_jsonl(
            latency_root / f"repeat_{index}/latency_row.jsonl",
            {
                "schedule_policy": "default",
                "measurement_status": "measured",
                "width": [64, 128, 256],
                "latency_p50_us": latency * 1000,
                "source_files": ["/models/codriving.onnx"],
            },
        )
        _write_jsonl(
            root / f"energy_repeats_v2/repeat_{index}/energy_row.jsonl",
            {
                "schedule_policy": "default",
                "measurement_status": "measured",
                "width": [64, 128, 256],
                "joule_per_inference": 0.8 + index * 0.1,
                "source_files": ["/models/codriving.onnx"],
            },
        )
    _write_json(
        root / "full_ap/full_ap_eval_report.json",
        {
            "status": "success_full",
            "processed_samples": 1789,
            "failed_samples": 0,
            "fallback_samples": 0,
            "ap30": 0.8,
            "ap50": 0.6,
            "ap70": 0.4,
            "artifact_sha256": _sha(artifact),
        },
    )

    audit = build_codriving_audit(root=root, latency_root=latency_root)

    assert audit["status"] == "passed"
    assert audit["schedule_policy"] == "tvm_default_zero_trial"
    assert audit["latency_ms"] == pytest.approx(11.58)
    assert audit["energy_j"] == pytest.approx(0.9)
    assert audit["AP70"] == pytest.approx(0.4)
    assert len(audit["performance_repeats"]) == 3


def test_fcooper_audit_rejects_non_zero_trial_build(tmp_path: Path) -> None:
    root = tmp_path / "fcooper"
    artifact = root / "repeat_0/fcooper_default_r0/route_b_fp32_auto.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"compiled")
    for index in range(3):
        repeat_dir = root / f"repeat_{index}/fcooper_default_r{index}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        repeat_artifact = repeat_dir / "route_b_fp32_auto.so"
        repeat_artifact.write_bytes(b"compiled")
        _write_json(
            repeat_dir / "route_b_fp32_auto_build.json",
            {
                "status": "success",
                "build_success": True,
                "max_trials": 0,
                "tuning_policy": (
                    "metaschedule" if index == 1 else "default_compile_no_metaschedule"
                ),
            },
        )
        _write_json(
            repeat_dir / "route_b_fp32_auto_result.json",
            {
                "status": "success",
                "build_success": True,
                "width": [64, 128, 256, 128, 256],
                "artifact_path": str(repeat_artifact),
                "latency": {"latency_ms_p50": 117.0 + index},
                "energy": {"status": "success", "joules_per_inference": 28.0 + index},
            },
        )
    _write_json(
        root / "full_ap/full_ap_eval_report.json",
        {
            "status": "success_full",
            "processed_samples": 2170,
            "failed_samples": 0,
            "fallback_samples": 0,
            "ap30": 0.9,
            "ap50": 0.8,
            "ap70": 0.63,
            "artifact_sha256": _sha(artifact),
        },
    )

    with pytest.raises(ValueError, match="zero-trial"):
        build_fcooper_audit(root=root)


def test_rebuild_rows_normalizes_real_pyramid_repair_shape() -> None:
    source_rows = [
        {"model": "pyramid", "method": "Original/default", "latency_ms": "3.2"},
        {"model": "codriving", "method": "Original/default", "latency_ms": "1.8"},
        {"model": "fcooper", "method": "Original/default", "latency_ms": "11.8"},
    ]
    pyramid_rows = [
        {
            "method": "original_default",
            "backend": "tvm",
            "AP70": "0.63",
            "latency_ms": "56.3",
            "energy_j": "20.2",
        },
        {
            "method": "compression_only",
            "backend": "tvm",
            "AP70": "0.55",
            "latency_ms": "16.7",
            "energy_j": "2.7",
        },
        {"method": "tune_then_compress", "backend": "tvm"},
        {
            "method": "joint_shcosearch",
            "backend": "tvm",
            "AP70": "0.61",
            "latency_ms": "3.3",
            "energy_j": "0.6",
        },
    ]
    audits = {
        "codriving": {
            "AP70": 0.4,
            "latency_ms": 11.58,
            "energy_j": 0.9,
            "evidence_path": "c.json",
        },
        "fcooper": {
            "AP70": 0.63,
            "latency_ms": 117.5,
            "energy_j": 29.0,
            "evidence_path": "f.json",
        },
    }

    rows = rebuild_three_model_rows(
        source_rows=source_rows,
        pyramid_rows=pyramid_rows,
        audits=audits,
        delta_ap=0.10,
    )

    pyramid = [row for row in rows if row.get("model") == "pyramid"]
    assert [row["method"] for row in pyramid] == [
        "Original/default",
        "Compression only",
        "GEAR",
    ]
    assert pyramid[0]["schedule_policy"] == "tvm_default_zero_trial"
    assert all(row.get("model") for row in rows)
    assert not any(row.get("method") == "Tune -> Compress" for row in rows)


def test_rebuild_rows_replaces_only_backend_default_and_recomputes_speedup() -> None:
    source_rows = [
        {"model": "codriving", "method": "Original/default", "latency_ms": "1.8"},
        {"model": "codriving", "method": "GEAR", "latency_ms": "1.5", "AP70": "0.36"},
        {"model": "fcooper", "method": "Original/default", "latency_ms": "11.8"},
        {"model": "fcooper", "method": "GEAR", "latency_ms": "10.2", "AP70": "0.60"},
    ]
    audits = {
        "codriving": {
            "AP70": 0.40,
            "latency_ms": 11.58,
            "energy_j": 0.90,
            "evidence_path": "codriving-audit.json",
        },
        "fcooper": {
            "AP70": 0.63,
            "latency_ms": 117.53,
            "energy_j": 28.98,
            "evidence_path": "fcooper-audit.json",
        },
    }

    rows = rebuild_three_model_rows(
        source_rows=source_rows,
        pyramid_rows=[],
        audits=audits,
        delta_ap=0.10,
    )

    codriving_default = rows[0]
    codriving_gear = rows[1]
    assert codriving_default["latency_ms"] == 11.58
    assert codriving_default["schedule_policy"] == "tvm_default_zero_trial"
    assert codriving_gear["speedup"] == pytest.approx(11.58 / 1.5)
    assert codriving_gear["ap70_floor"] == pytest.approx(0.30)
    assert codriving_gear["HV"] == ""
