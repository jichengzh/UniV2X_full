import hashlib
import json

import pytest

from framework.stage6.evidence_bundle_v1 import (
    apply_independent_validation,
    attach_common_hypervolume,
    build_independent_validation_index,
    normalize_closure_point,
    normalize_actual_feedback_row,
    payload_sha256,
)


def _file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_actual_feedback_normalization_verifies_bound_evidence(tmp_path) -> None:
    performance = tmp_path / "performance.json"
    ap = tmp_path / "ap.json"
    source = tmp_path / "source.json"
    for path, payload in (
        (performance, {"latency_ms": 1.0}),
        (ap, {"processed_samples": 1789}),
        (source, {"status": "ready"}),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    graph = {"graph_feature_provenance": "materialized_onnx_extracted_v1", "conv_count": 51}
    candidate_graph = {"graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1"}
    row = {
        "terminal_status": "measured_success_gold",
        "task_id": "S5-PYR-TRT",
        "model": "pyramid",
        "hardware_id": "h800",
        "capability_profile_id": "h800-trt-probe-conditioned-v3",
        "dispatch_key": "trt_engine",
        "manifest_job_id": "pyramid|16x48x64|q=fp16|profile=h800-trt-probe-conditioned-v3",
        "width": [16, 48, 64],
        "q_mode": "fp16",
        "genome": [16, 48, 64, "fp16"],
        "latency_ms": 1.0,
        "energy_j": 0.2,
        "ap70": 0.6,
        "performance_result_json": str(performance),
        "performance_result_sha256": _file_sha(performance),
        "ap_report_path": str(ap),
        "ap_report_sha256": _file_sha(ap),
        "materialized_source_evidence_path": str(source),
        "materialized_source_evidence_sha256": _file_sha(source),
        "graph_features": graph,
        "materialized_graph_features_sha256": payload_sha256(graph),
        "candidate_graph_features": candidate_graph,
        "candidate_graph_features_sha256": payload_sha256(candidate_graph),
    }
    row["actual_feedback_row_sha256"] = payload_sha256(row)

    point = normalize_actual_feedback_row(row, expected_backend="trt")

    assert point["config"] == [16, 48, 64, "fp16"]
    assert point["AP70"] == 0.6
    assert point["evidence_sha_verified"] is True
    assert point["actual_graph_features_verified"] is True

    broken = {**row, "latency_ms": 0.5}
    with pytest.raises(ValueError, match="actual feedback row SHA"):
        normalize_actual_feedback_row(broken, expected_backend="trt")


def test_common_hypervolume_uses_one_backend_normalization_without_mutation() -> None:
    arms = {
        "fast": {
            "points": [
                {
                    "terminal_status": "measured_success_gold",
                    "evidence_sha_verified": True,
                    "latency_ms": 1.0,
                    "energy_j": 1.0,
                    "AP70": 0.6,
                }
            ]
        },
        "slow": {
            "points": [
                {
                    "terminal_status": "measured_success_gold",
                    "evidence_sha_verified": True,
                    "latency_ms": 2.0,
                    "energy_j": 2.0,
                    "AP70": 0.6,
                }
            ]
        },
    }

    result = attach_common_hypervolume(arms)

    assert "HV" not in arms["fast"]
    assert result["arms"]["fast"]["HV"] > result["arms"]["slow"]["HV"]
    assert result["arms"]["fast"]["pareto_count"] == 1
    assert result["normalization"]["derived_from_point_count"] == 2


def test_common_hypervolume_rejects_unverified_or_empty_evidence() -> None:
    with pytest.raises(ValueError, match="no SHA-verified measured points"):
        attach_common_hypervolume(
            {
                "arm": {
                    "points": [
                        {
                            "terminal_status": "measured_success_gold",
                            "evidence_sha_verified": False,
                            "latency_ms": 1.0,
                            "energy_j": 1.0,
                            "AP70": 0.6,
                        }
                    ]
                }
            }
        )


def test_closure_point_and_independent_audit_replace_metrics_with_rerun(tmp_path) -> None:
    performance = tmp_path / "search_performance.json"
    ap = tmp_path / "search_ap.json"
    performance.write_text("{}", encoding="utf-8")
    ap.write_text(json.dumps({"processed_samples": 1789}), encoding="utf-8")
    point = normalize_closure_point(
        {
            "manifest_job_id": "pyramid|16x32x64|q=int8|profile=h800-tvm-probe-conditioned-v3",
            "genome": [16, 32, 64, "int8"],
            "q_mode": "int8",
            "terminal_status": "measured_success_gold",
            "objectives": {"latency_ms": 3.3, "energy_j": 0.7, "ap70": 0.61},
            "performance_result_json": str(performance),
            "performance_result_sha256": _file_sha(performance),
            "ap_report_path": str(ap),
            "ap_report_sha256": _file_sha(ap),
        },
        expected_backend="tvm",
    )
    bound_files = []
    for name in ("repeat0", "repeat1", "repeat2", "full_ap", "source"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        bound_files.append(path)
    audit = {
        "schema_version": "stage5_independent_validation_audit_v1",
        "tasks": [
            {
                "task_id": "S5-PYR-TVM",
                "passed": True,
                "configurations": [
                    {
                        "configuration_id": point["manifest_job_id"],
                        "consistency": {
                            "passed": True,
                            "rerun": {
                                "latency_median_ms": 3.4,
                                "energy_median_j": 0.65,
                                "ap70": 0.612,
                            },
                        },
                        "performance_repeats": [
                            {
                                "performance_result_json": str(path),
                                "performance_result_sha256": _file_sha(path),
                            }
                            for path in bound_files[:3]
                        ],
                        "ap_report_path": str(bound_files[3]),
                        "ap_report_sha256": _file_sha(bound_files[3]),
                        "evidence_path": str(bound_files[4]),
                        "evidence_sha256": _file_sha(bound_files[4]),
                    }
                ],
            }
        ],
    }

    index = build_independent_validation_index([audit])
    validated = apply_independent_validation(
        [point], index, arm_id="joint_shcosearch"
    )[0]

    assert validated["search_latency_ms"] == 3.3
    assert validated["latency_ms"] == 3.4
    assert validated["energy_j"] == 0.65
    assert validated["AP70"] == 0.612
    assert validated["independent_validation_passed"] is True
    assert validated["evidence_sha_verified"] is True
    other_arm = apply_independent_validation(
        [point], index, arm_id="compression_only"
    )[0]
    assert other_arm["independent_validation_passed"] is False
