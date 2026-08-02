from __future__ import annotations

import json

import numpy as np
import pytest

from framework.stage6.fcooper_orin_v1 import (
    CONFIGURATIONS,
    energy_joules,
    numeric_metrics,
    parse_tegrastats_power,
    summarize_latency,
    validate_inspector_precision,
)


def test_frozen_five_configuration_contract() -> None:
    assert tuple(CONFIGURATIONS) == (
        "original",
        "compression_only",
        "schedule_only",
        "compress_then_tune",
        "joint_fp16_control",
    )
    assert CONFIGURATIONS["original"]["width"] == (64, 128, 256, 128, 256)
    assert CONFIGURATIONS["compression_only"]["builder_level"] == 0
    assert CONFIGURATIONS["schedule_only"]["precision"] == "fp32"
    assert CONFIGURATIONS["schedule_only"]["builder_level"] == 5
    assert CONFIGURATIONS["joint_fp16_control"]["precision"] == "fp16"


def test_latency_summary_requires_exact_protocol() -> None:
    samples = [list(np.arange(300, dtype=float) + repeat) for repeat in range(5)]
    report = summarize_latency(samples)
    assert report["sample_count"] == 1500
    assert len(report["repeats"]) == 5
    assert report["aggregate"]["median_ms"] == pytest.approx(151.5)
    assert report["protocol"]["warmup"] == 20
    assert report["protocol"]["iterations"] == 300
    with pytest.raises(ValueError, match="five repeats"):
        summarize_latency(samples[:4])
    with pytest.raises(ValueError, match="300 samples"):
        summarize_latency([row[:299] for row in samples])


def test_numeric_metrics_report_expected_fields() -> None:
    reference = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32)
    candidate = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float16)
    report = numeric_metrics(reference, candidate)
    assert report["shape"] == [4]
    assert report["reference_dtype"] == "float32"
    assert report["candidate_dtype"] == "float16"
    assert report["mae"] == pytest.approx(0.25)
    assert report["max_abs"] == pytest.approx(1.0)
    assert report["finite_ratio"] == pytest.approx(1.0)
    assert report["reference_zero_ratio"] == pytest.approx(0.5)
    assert report["candidate_zero_ratio"] == pytest.approx(0.5)


def test_numeric_metrics_reject_shape_drift() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        numeric_metrics(np.zeros((2, 2)), np.zeros((4,)))


def test_parse_tegrastats_vin_sys_5v0_and_auxiliary_rail() -> None:
    text = "\n".join(
        [
            "RAM 100/1000MB CPU [1%@1] VIN_SYS_5V0 7500mW/7200mW VDD_GPU_SOC 3100mW/3000mW",
            "RAM 100/1000MB CPU [1%@1] VIN_SYS_5V0 7700mW/7300mW VDD_GPU_SOC 3300mW/3100mW",
        ]
    )
    report = parse_tegrastats_power(text)
    assert report["sample_count"] == 2
    assert report["vin_sys_5v0_mean_w"] == pytest.approx(7.6)
    assert report["vdd_gpu_soc_mean_w"] == pytest.approx(3.2)
    assert report["rails_combined"] is False
    with pytest.raises(ValueError, match="VIN_SYS_5V0"):
        parse_tegrastats_power("RAM 100/1000MB CPU [1%@1]")


def test_energy_uses_only_active_window_board_rail() -> None:
    assert energy_joules(7.6, 12.5) == pytest.approx(0.095)
    with pytest.raises(ValueError):
        energy_joules(-1.0, 12.5)


def test_strict_fp32_inspector_rejects_reduced_precision() -> None:
    good = json.dumps(
        {
            "Layers": [
                {"Name": "conv", "LayerType": "CaskConvolution", "Precision": "Float"}
            ]
        }
    )
    report = validate_inspector_precision(good, expected="fp32", tf32_disabled=True)
    assert report["valid"] is True
    for forbidden in ("Half", "Int8", "BFloat16", "TF32"):
        bad = json.dumps({"Layers": [{"Name": "conv", "Precision": forbidden}]})
        report = validate_inspector_precision(
            bad, expected="fp32", tf32_disabled=True
        )
        assert report["valid"] is False


def test_fp16_inspector_requires_reduced_precision_evidence() -> None:
    report = validate_inspector_precision(
        json.dumps({"Layers": [{"Format/Datatype": "Row major FP16 format"}]}),
        expected="fp16",
        tf32_disabled=False,
    )
    assert report["valid"] is True
    report = validate_inspector_precision(
        json.dumps({"Layers": [{"Precision": "Float"}]}),
        expected="fp16",
        tf32_disabled=False,
    )
    assert report["valid"] is False
