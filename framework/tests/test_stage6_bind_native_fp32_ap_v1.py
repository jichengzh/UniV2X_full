import hashlib
import json
from pathlib import Path

import pytest

from scripts.stage6_bind_native_fp32_ap_v1 import bind_ap_report


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bind_ap_report_requires_matching_checkpoint_and_full_eval(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "report.json"
    baseline = {
        "schema_version": "stage6_native_fp32_baseline_v1",
        "precision": "fp32",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    report = {
        "status": "success",
        "precision": "fp32",
        "precision_mode": "model_float32",
        "ckpt_path": str(checkpoint),
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "num_samples": 1789,
        "ap30": 0.8,
        "ap50": 0.7,
        "ap70": 0.6,
    }
    _write(baseline_path, baseline)
    _write(report_path, report)

    bound = bind_ap_report(baseline_path, report_path)

    assert bound["ap70"] == 0.6
    assert bound["ap_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert bound["ap_checkpoint_sha256"] == baseline["checkpoint_sha256"]


def test_bind_ap_report_rejects_checkpoint_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    baseline_path = tmp_path / "baseline.json"
    report_path = tmp_path / "report.json"
    _write(
        baseline_path,
        {
            "schema_version": "stage6_native_fp32_baseline_v1",
            "precision": "fp32",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )
    _write(
        report_path,
        {
            "status": "success",
            "precision": "fp32",
            "precision_mode": "model_float32",
            "ckpt_path": str(tmp_path / "other.pth"),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "num_samples": 1789,
            "ap30": 0.8,
            "ap50": 0.7,
            "ap70": 0.6,
        },
    )

    with pytest.raises(ValueError, match="checkpoint path drift"):
        bind_ap_report(baseline_path, report_path)
