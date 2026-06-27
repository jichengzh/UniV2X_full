from __future__ import annotations

from framework.stage1.model_classifier import build_classification_report


def _by_model(report: dict) -> dict[str, dict]:
    return {item["model"]: item for item in report["models"]}


def _skip_by_name(item: dict) -> dict[str, dict]:
    return {
        skipped["name"]: skipped
        for skipped in item["trace_plan"].get("skipped_subgraphs", [])
    }


def test_nine_model_report_exposes_trace_plan_contract_and_keeps_safe_classes():
    report = build_classification_report()
    by_model = _by_model(report)

    assert len(report["models"]) == 9
    assert report["no_overpromotion"] is True
    for item in report["models"]:
        assert item["trace_plan"]["schema"] == "stage1_trace_plan_v1"
        assert "trace_confidence" in item
        assert "coverage_scope" in item
        assert "manual_override_used" in item
        assert "review_required" in item
        assert isinstance(item["trace_plan"]["included_modules"], list)
        assert isinstance(item["trace_plan"]["ignored_layers"], list)
        assert isinstance(item["trace_plan"]["skipped_subgraphs"], list)
        assert isinstance(item["trace_plan"]["rejected_candidates"], list)

    for model in ("fcooper", "attfuse", "where2comm", "v2vnet", "disconet"):
        item = by_model[model]
        assert item["trace_plan"]["detector"] == "TraceBoundaryDetector.heter_baseline_v1"
        assert item["manual_override_used"] is False
        assert item["coverage_scope"] == "dense_core_only"
        assert item["trace_plan"]["selected_candidate"]["wrapper_kind"] == "heter_baseline_dense_path"
        assert {"backbone_m1", "shrinker_m1", "cls_head", "reg_head", "dir_head"} <= {
            module["name"] for module in item["trace_plan"]["included_modules"]
        }
        assert {"cls_head", "reg_head", "dir_head"} <= {
            module["name"] for module in item["trace_plan"]["ignored_layers"]
        }

    assert _skip_by_name(by_model["fcooper"])["fusion_net"]["blocker_gate"] == "maxfusion_coverage_anchor"
    assert _skip_by_name(by_model["attfuse"])["fusion_net"]["blocker_gate"] == "attention_fusion_coverage_anchor"
    for model in ("where2comm", "v2vnet", "disconet"):
        item = by_model[model]
        assert item["acceleration_class"] == "SCAN_FAILED"
        assert item["ckpt_status"] == "missing_architecture_scan_only"
        assert _skip_by_name(item)["fusion_net"]["blocker_gate"] == "routing_fusion_coverage_anchor"

    codriving = by_model["codriving"]
    assert codriving["acceleration_class"] == "SEPARABLE_ACCELERATION"
    assert codriving["scope"] == "codriving_measured_resnet_backbone_envelope"
    assert any("cross-model" in value for value in codriving["unsupported_conclusions"])

    for model in ("pyramid_lidar", "pyramid_camera"):
        item = by_model[model]
        assert item["acceleration_class"] == "CO_ACCELERATION_REQUIRED"
        assert "historical_trt_evidence" in item["evidence_level"]
        assert item["historical_evidence_sources"]

    v2xvit = by_model["v2xvit"]
    assert v2xvit["acceleration_class"] == "CO_ACCELERATION_REQUIRED"
    assert any("true TRT INT8 AP" in item for item in v2xvit["unsupported_conclusions"])
