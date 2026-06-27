from __future__ import annotations

from pathlib import Path

import yaml

from framework.stage1.coupling_predictor import predict_manifest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "results/stage1_model_predict"


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / f"{payload['model']}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_static_only_standard_conv_never_predicts_low_risk(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path,
        {
            "stage": "stage1_partition",
            "model": "fcooper",
            "scan_status": "ok",
            "search_space_summary": {
                "n_b1_search_knobs": 1,
                "latency_status": "skipped",
            },
            "trace": {
                "entry_shape": [1, 64, 512, 512],
                "skipped_modules": [],
            },
            "view_b1_search_groups": [
                {
                    "search_group_id": "backbone.s0",
                    "bucket": "backbone",
                    "widths": [64],
                    "grouped_conv": False,
                    "int8_buildable_align": 4,
                }
            ],
        },
    )

    report = predict_manifest(manifest, evidence_dir=EVIDENCE_DIR)

    assert report["verdict"] == "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE"
    assert report["verdict"] != "PREDICTED_SEPARABLE_LOW_RISK"
    assert report["evidence_level"] == "static_only_pending_anchor_probe"
    assert "std_basebev_backbone_schedule_anchor" in report["required_next_probe_or_gate"]
    assert report["blockers"]


def test_legacy_attention_fusion_skip_blocks_full_model_low_risk(tmp_path: Path):
    manifest = _write_manifest(
        tmp_path,
        {
            "stage": "stage1_partition",
            "model": "attfuse",
            "scan_status": "ok",
            "search_space_summary": {
                "n_b1_search_knobs": 1,
                "latency_status": "skipped",
            },
            "trace": {
                "entry_shape": [1, 64, 512, 512],
                "skipped_modules": [
                    "fusion_net (AttFusion: multi-agent attention fusion, auto-skip)"
                ],
            },
            "view_b1_search_groups": [
                {
                    "search_group_id": "backbone.s0",
                    "bucket": "backbone",
                    "widths": [64],
                    "grouped_conv": False,
                    "int8_buildable_align": 4,
                }
            ],
        },
    )

    report = predict_manifest(manifest, evidence_dir=EVIDENCE_DIR)

    assert report["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert report["scope"] == "traced_dense_subgraph"
    assert report["evidence_level"] == "uncovered_unknown_until_attention_integration"
    assert "attention_fusion_coverage_anchor" in report["blockers"]
    assert report["verdict"] not in {
        "MEASURED_SEPARABLE",
        "PREDICTED_SEPARABLE_LOW_RISK",
    }


def test_codriving_measured_anchor_stays_scoped_to_codriving_envelope():
    report = predict_manifest(
        ROOT / "framework/partitions/codriving_partition.yaml",
        evidence_dir=EVIDENCE_DIR,
    )

    assert report["verdict"] == "ANCHOR_PROBED_LOW_RISK"
    assert report["scope"] == "codriving_measured_resnet_backbone_envelope"
    assert report["evidence_level"] == "measured_negative_anchor"
    assert "standard_neck_deconv_anchor" in report["required_next_probe_or_gate"]
    assert any("CoDriving" in item for item in report["evidence"])


def test_pyramid_mixed_context_blocks_model_level_standard_conv_promotion():
    report = predict_manifest(
        ROOT / "framework/partitions/pyramid_lidar_partition.yaml",
        evidence_dir=EVIDENCE_DIR,
    )

    assert report["verdict"] == "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION"
    assert report["scope"] == "full_model_p_hub_context"
    assert report["evidence_level"] == "historical_trt_evidence_plus_p_hub_context"
    assert "pyramid_mixed_p_hub_context_anchor" in report["blockers"]
    assert any("historical TRT AP/latency evidence" in item for item in report["evidence"])
