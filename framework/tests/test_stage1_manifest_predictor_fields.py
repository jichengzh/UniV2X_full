from __future__ import annotations

from pathlib import Path

import yaml

from framework.stage1.coupling_predictor import find_overpromotions, predict_manifests
from framework.stage1.model_classifier import normalize_manifest_for_predictor


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "results/stage1_model_predict"
MANIFESTS = [
    ROOT / "framework/partitions/codriving_partition.yaml",
    ROOT / "results/autoscan_fcooper_partition.yaml",
    ROOT / "results/autoscan_attfuse_partition.yaml",
    ROOT / "framework/partitions/v2xvit_partition.yaml",
    ROOT / "framework/partitions/pyramid_lidar_partition.yaml",
    ROOT / "framework/partitions/pyramid_camera_partition.yaml",
    ROOT / "results/autoscan_where2comm_partition.yaml",
    ROOT / "results/autoscan_v2vnet_partition.yaml",
    ROOT / "results/autoscan_disconet_partition.yaml",
]


def _by_model(report: dict) -> dict[str, dict]:
    return {item["model"]: item for item in report["predictions"]}


def test_current_manifests_emit_required_predictor_fields():
    report = predict_manifests(MANIFESTS, evidence_dir=EVIDENCE_DIR)

    assert report["schema"] == "stage1_coupling_predictions_v0"
    assert len(report["predictions"]) == 9

    required = {
        "model",
        "manifest",
        "verdict",
        "scope",
        "evidence_level",
        "ckpt_status",
        "blockers",
        "required_next_probe_or_gate",
        "evidence_sources",
    }
    for item in report["predictions"]:
        assert required <= item.keys()
        assert isinstance(item["blockers"], list)
        assert isinstance(item["required_next_probe_or_gate"], list)
        assert isinstance(item["evidence_sources"], list)
        assert item["manifest"] in item["evidence_sources"]
        assert "results/stage1_model_predict/standard_conv_probe_queue_v1.json" in item["evidence_sources"]
        assert "results/stage1_model_predict/standard_conv_census_v1.json" in item["evidence_sources"]
        assert item["verdict"]
        assert item["scope"]
        assert item["evidence_level"]


def test_current_manifests_have_typed_skip_and_predictor_feature_fields_after_normalization():
    for path in MANIFESTS:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = normalize_manifest_for_predictor(raw)
        trace = manifest["trace"]

        assert "skipped_modules" in trace
        assert "skipped_subgraphs" in trace
        assert isinstance(trace["skipped_subgraphs"], list)
        for skipped in trace["skipped_subgraphs"]:
            assert {
                "name",
                "type",
                "full_model_verdict_blocker",
                "blocker_gate",
                "source",
            } <= set(skipped)

        for group in manifest["view_b1_prune_groups"]:
            feature = group.get("feature", {})
            assert {
                "cin",
                "cout",
                "groups",
                "ic_bn",
                "kernel",
                "stride",
                "op_types",
                "fanout_buckets",
            } <= set(feature)
            assert isinstance(feature["op_types"], list)
            assert isinstance(feature["fanout_buckets"], list)

        for group in manifest["view_b1_search_groups"]:
            feature = group.get("feature", {})
            assert {
                "min_ic_bn",
                "max_groups",
                "op_types",
                "fanout_buckets",
            } <= set(feature)
            assert isinstance(feature["op_types"], list)
            assert isinstance(feature["fanout_buckets"], list)

        coverage = manifest["view_latency"]["coverage"]
        assert coverage["coverage_scope"] == "trace_net_only"
        assert coverage["full_model_latency_pct"] is None
        assert coverage["skipped_subgraphs_accounted_separately"] is True
        assert "not full-model coverage" in coverage["note"].lower()


def test_current_model_verdicts_follow_safe_v0_gates():
    by_model = _by_model(predict_manifests(MANIFESTS, evidence_dir=EVIDENCE_DIR))

    assert by_model["codriving"]["verdict"] == "ANCHOR_PROBED_LOW_RISK"
    assert by_model["codriving"]["scope"] == "codriving_measured_resnet_backbone_envelope"
    assert by_model["fcooper"]["verdict"] == "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE"
    assert by_model["attfuse"]["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert by_model["v2xvit"]["verdict"] == "JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND"
    assert by_model["where2comm"]["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert by_model["v2vnet"]["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert by_model["disconet"]["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert by_model["where2comm"]["ckpt_status"] == "missing_architecture_scan_only"
    assert by_model["v2vnet"]["ckpt_status"] == "missing_architecture_scan_only"
    assert by_model["disconet"]["ckpt_status"] == "missing_architecture_scan_only"
    assert (
        by_model["pyramid_lidar"]["verdict"]
        == "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION"
    )
    assert (
        by_model["pyramid_camera"]["verdict"]
        == "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION"
    )


def test_skipped_attention_fusion_and_custom_block_full_model_overpromotion():
    report = predict_manifests(MANIFESTS, evidence_dir=EVIDENCE_DIR)

    by_model = _by_model(report)
    assert "attention_fusion_coverage_anchor" in by_model["attfuse"]["blockers"]
    assert "attention_fusion_coverage_anchor" in by_model["v2xvit"]["blockers"]
    assert "routing_fusion_coverage_anchor" in by_model["v2xvit"]["blockers"]
    assert "attention_fusion_coverage_anchor" in by_model["where2comm"]["blockers"]
    assert "routing_fusion_coverage_anchor" in by_model["v2vnet"]["blockers"]
    assert "routing_fusion_coverage_anchor" in by_model["disconet"]["blockers"]
    assert "pyramid_mixed_p_hub_context_anchor" in by_model["pyramid_camera"]["blockers"]

    bad = []
    for item in report["predictions"]:
        if item["blockers"] and item["scope"] == "full_model" and item["verdict"] in {
            "MEASURED_SEPARABLE",
            "PREDICTED_SEPARABLE_LOW_RISK",
        }:
            bad.append((item["model"], item["verdict"], item["blockers"]))

    assert bad == []


def test_overpromotion_guard_treats_full_model_prefixed_scopes_as_full_model():
    bad = find_overpromotions(
        [
            {
                "model": "v2xvit",
                "scope": "full_model_gate_blocked",
                "verdict": "PREDICTED_SEPARABLE_LOW_RISK",
                "blockers": ["attention_fusion_coverage_anchor"],
            }
        ]
    )

    assert bad == [
        {
            "model": "v2xvit",
            "verdict": "PREDICTED_SEPARABLE_LOW_RISK",
            "scope": "full_model_gate_blocked",
            "blockers": ["attention_fusion_coverage_anchor"],
        }
    ]


def test_overpromotion_guard_treats_full_model_anchor_low_risk_as_full_model():
    bad = find_overpromotions(
        [
            {
                "model": "unsafe_anchor",
                "scope": "full_model",
                "verdict": "ANCHOR_PROBED_LOW_RISK",
                "blockers": ["missing_architecture_scan_only"],
            }
        ]
    )

    assert bad == [
        {
            "model": "unsafe_anchor",
            "verdict": "ANCHOR_PROBED_LOW_RISK",
            "scope": "full_model",
            "blockers": ["missing_architecture_scan_only"],
        }
    ]


def test_static_only_evidence_is_never_promoted_to_predicted_low_risk():
    report = predict_manifests(MANIFESTS, evidence_dir=EVIDENCE_DIR)

    static_promotions = [
        item["model"]
        for item in report["predictions"]
        if item["evidence_level"].startswith("static_only")
        and item["verdict"] == "PREDICTED_SEPARABLE_LOW_RISK"
    ]

    assert static_promotions == []
