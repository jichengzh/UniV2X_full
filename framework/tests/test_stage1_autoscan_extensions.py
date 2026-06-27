from __future__ import annotations

from pathlib import Path

from framework.stage1.auto_trace import AUTO_REGISTRY, _ensure_disco_fuse_compat
from framework.stage1.coupling_predictor import predict_manifest
from framework.stage1.standard_conv_census import DEFAULT_MANIFESTS


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "results/stage1_model_predict"
EXTENSION_MODELS = {"where2comm", "v2vnet", "disconet"}


def test_a2_extension_models_are_registered_for_autoscan():
    assert EXTENSION_MODELS <= set(AUTO_REGISTRY)
    assert "who2com" not in AUTO_REGISTRY


def test_default_census_manifest_list_includes_extension_outputs():
    manifest_names = {path.name for path in DEFAULT_MANIFESTS}

    assert {
        "autoscan_where2comm_partition.yaml",
        "autoscan_v2vnet_partition.yaml",
        "autoscan_disconet_partition.yaml",
    } <= manifest_names


def test_disconet_compat_loader_exposes_pixel_weight_layer():
    module = _ensure_disco_fuse_compat()

    assert hasattr(module, "PixelWeightLayer")


def test_extension_fusion_models_remain_uncovered_until_fusion_gate_is_bound():
    report = predict_manifest(
        {
            "stage": "stage1_partition",
            "model": "v2vnet",
            "scan_status": "ok",
            "search_space_summary": {
                "n_b1_search_knobs": 1,
                "latency_status": "skipped",
            },
            "trace": {
                "entry_shape": [1, 64, 512, 512],
                "skipped_modules": [
                    "fusion_net (V2VNet: GNN message passing fusion, auto-skip)"
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
        evidence_dir=EVIDENCE_DIR,
    )

    assert report["verdict"] == "FUSION_UNCOVERED_UNKNOWN"
    assert "routing_fusion_coverage_anchor" in report["blockers"]
    assert "routing_fusion_coverage_anchor" in report["required_next_probe_or_gate"]
    assert "std_basebev_backbone_schedule_anchor" in report["required_next_probe_or_gate"]
