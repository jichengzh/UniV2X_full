from __future__ import annotations

from pathlib import Path

from framework.stage1_bridge import load_stage2_search_space


ROOT = Path(__file__).resolve().parents[2]


def _by_id(items: list[dict]) -> dict[str, dict]:
    return {item["id"]: item for item in items}


def test_pyramid_stage2_search_space_has_width_anchors_hierarchy_and_joint_policy():
    space = load_stage2_search_space(
        ROOT / "framework/partitions/pyramid_lidar_partition.yaml"
    )

    assert space["schema"] == "stage2_search_space_v1"
    assert space["model"] == "pyramid_lidar"
    assert space["optimized_scope"] == "rsu_dense_core"
    assert space["claim_scope"]["full_model_claim_allowed"] is False
    assert "ego_fusion_attention_future" in space["claim_scope"]["future_blocks"]

    policy = space["model_search_policy"]
    assert policy["selected"] == "joint"
    assert policy["decision_level"] == "model"
    assert "P_IC_BN_hub" in policy["reasons"]
    assert "per_knob_joint_serial" in policy["unsupported_conclusions"]

    stages = {
        block["dense_stage"]
        for block in space["hierarchical_blocks"]
        if block["execution_block"] == "dense_perception"
    }
    assert {"stage1", "stage2", "stage3"} <= stages

    candidates = _by_id(space["software_candidates"])
    grouped = candidates["bev_encoder.s2"]
    assert grouped["dense_stage"] == "stage3"
    assert grouped["grouped_conv"] is True
    assert 5 <= len(grouped["width_anchors"]) <= 9
    assert grouped["max_prune_rate"]["value"] == 0.875
    assert {"structure", "hardware", "ap", "lut"} <= set(
        grouped["max_prune_rate"]["bounds"]
    )

    widths = {anchor["width"] for anchor in grouped["width_anchors"]}
    assert grouped["base_width"] in widths
    assert any(anchor["anchor_type"] == "diagnostic_trap" for anchor in grouped["width_anchors"])
    assert any(anchor["anchor_type"] == "production" for anchor in grouped["width_anchors"])
    assert any("legacy_50pct_mapped" in anchor["roles"] for anchor in grouped["width_anchors"])

    for candidate in space["software_candidates"]:
        assert candidate["quant_axis_status"] in {
            "active_software_candidate",
            "fixed_quant_policy",
            "evidence_only_attribute",
            "unavailable",
        }
        assert candidate["quant_policies"]
        for policy in candidate["quant_policies"]:
            assert {"policy", "backend_scope", "provenance", "status"} <= policy.keys()


def test_codriving_stage2_search_space_stays_serial_without_groups1_overpromotion():
    space = load_stage2_search_space(
        ROOT / "framework/partitions/codriving_partition.yaml"
    )

    assert space["schema"] == "stage2_search_space_v1"
    assert space["model"] == "codriving"
    assert space["optimized_scope"] == "rsu_dense_core"
    assert space["claim_scope"]["full_model_claim_allowed"] is False

    policy = space["model_search_policy"]
    assert policy["selected"] == "serial"
    assert policy["decision_level"] == "model"
    assert "groups1_not_model_separable_proof" in policy["unsupported_conclusions"]
    assert "per_knob_joint_serial" in policy["unsupported_conclusions"]

    candidates = _by_id(space["software_candidates"])
    assert {"backbone.s0", "backbone.s1", "backbone.s2"} <= set(candidates)
    assert all(candidate["grouped_conv"] is False for candidate in candidates.values())
    assert all(
        anchor["anchor_type"] == "production"
        for candidate in candidates.values()
        for anchor in candidate["width_anchors"]
    )
    assert all(
        2 <= len(candidate["width_anchors"]) <= 9
        for candidate in candidates.values()
    )

    blocks = space["hierarchical_blocks"]
    assert any(
        block["device_scope"] == "RSU"
        and block["execution_block"] == "dense_perception"
        and block["dense_stage"] == "stage1"
        for block in blocks
    )
    assert any(block["execution_block"] == "fusion_attention_future" for block in blocks)
