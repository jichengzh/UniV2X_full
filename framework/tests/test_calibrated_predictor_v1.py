from __future__ import annotations

from framework.stage1.calibrated_predictor import build_calibrated_report, build_predictions_v1


def _by_model(report: dict) -> dict[str, dict]:
    return {item["model"]: item for item in report["predictions"]}


def test_calibrated_predictor_v1_keeps_no_overpromotion_guard():
    report = build_predictions_v1()

    assert report["schema"] == "stage1_coupling_predictions_v1"
    assert report["no_overpromotion"] is True
    assert report["overpromotion"] == []
    assert len(report["predictions"]) == 9
    assert report["policy"]["groups_1_is_not_a_separability_rule"] is True
    assert report["policy"]["blocked_unresolved_partial_bound_never_pass"] is True


def test_calibrated_predictor_v1_evidence_categories_are_explicit():
    by_model = _by_model(build_predictions_v1())

    assert "measured_existing_codriving_anchor" in by_model["codriving"]["evidence_categories"]
    assert "measured_h800_tvm_latency" in by_model["fcooper"]["evidence_categories"]
    assert "s4_three_arm_latency_validation" in by_model["fcooper"]["evidence_categories"]
    assert "historical_trt_evidence" in by_model["pyramid_lidar"]["evidence_categories"]
    assert "true_trt_ap_latency" not in by_model["pyramid_lidar"]["evidence_categories"]
    assert "fake_quant_gate" in by_model["v2xvit"]["evidence_categories"]
    assert "structural_routing_gate" in by_model["v2xvit"]["evidence_categories"]
    assert "missing_architecture_scan_only" in by_model["v2vnet"]["evidence_categories"]
    assert "random_init_fusion_timing_sidecar" in by_model["disconet"]["evidence_categories"]


def test_calibrated_predictor_v1_keeps_trt_historical_and_h800_measured_separate():
    by_model = _by_model(build_predictions_v1())

    fcooper = by_model["fcooper"]
    pyramid = by_model["pyramid_lidar"]
    v2xvit = by_model["v2xvit"]

    assert "measured_h800_tvm_latency" in fcooper["evidence_categories"]
    assert "historical_trt_evidence" not in fcooper["evidence_categories"]
    assert "historical_trt_evidence" in pyramid["evidence_categories"]
    assert "historical_trt_evidence" in pyramid["evidence_level"]
    assert "measured_h800_tvm" not in repr(pyramid.get("historical_evidence_sources", "")).lower()
    assert "true TRT INT8 AP closure" in repr(v2xvit["unsupported_conclusions"])


def test_calibrated_predictor_v1_blocks_known_bad_claims():
    report = build_predictions_v1()
    prohibited = set(report["prohibited_claims"])
    by_model = _by_model(report)

    assert "groups=1 proves separability" in prohibited
    assert "V2X-ViT true TRT INT8 AP closed" in prohibited
    assert "Pyramid hand-forced per-stage mixed precision is Pareto-positive" in prohibited
    assert "Who2com is not implemented in HEAL" in prohibited
    assert "V2X-ViT true TRT INT8 AP closure" in by_model["v2xvit"]["unsupported_conclusions"]
    assert (
        "Pyramid hand-forced per-stage mixed precision Pareto-positive"
        in by_model["pyramid_lidar"]["unsupported_conclusions"]
    )
    assert "v2vnet trained checkpoint scan" in by_model["v2vnet"]["unsupported_conclusions"]


def test_calibrated_report_v1_has_s0_to_s6_status_and_sources():
    predictions = build_predictions_v1()
    report = build_calibrated_report(predictions)

    stages = {item["stage"]: item["status"] for item in report["stage_status"]}
    assert report["schema"] == "stage1_calibrated_predictor_report_v1"
    assert stages["S4 Mini Three-Arm Validation"] == "done_latency_only"
    assert stages["S5 Predictor Rule Update"] == "done_v1_guarded"
    assert stages["S6 Calibration And Reporting"] == "done_current_closure"
    assert report["evidence_chain"]["s5"]["no_overpromotion"] is True
    assert "S4 is latency-only and does not provide AP/HV joint-vs-serial proof." in report[
        "weaker_or_blocked_conclusions"
    ]
