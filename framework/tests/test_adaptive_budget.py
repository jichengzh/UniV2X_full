from __future__ import annotations

from pathlib import Path

import pytest

from framework.adaptive_budget import (
    NARROW_BUDGET_OK_VERDICTS,
    NEEDS_WIDE_BUDGET_VERDICTS,
    adaptive_inner_loop_budget,
    from_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = ROOT / "framework/partitions"


def test_requires_at_least_one_signal():
    with pytest.raises(ValueError):
        adaptive_inner_loop_budget()


def test_high_coupling_score_selects_wide_tier():
    plan = adaptive_inner_loop_budget(coupling_score=0.69)
    assert plan.tier == "wide"
    assert plan.k_per_round == 12
    assert plan.n_rounds == 3
    assert plan.autotune_candidate_multiplier == 4
    assert any("gap1_grid_corrected" in r for r in plan.rationale)


def test_low_coupling_score_selects_narrow_tier():
    plan = adaptive_inner_loop_budget(coupling_score=0.16)
    assert plan.tier == "narrow"
    assert plan.k_per_round == 4
    assert plan.n_rounds == 1
    assert plan.autotune_candidate_multiplier == 1
    assert any("codriving_dair_grid_8point_clean" in r for r in plan.rationale)


def test_selected_joint_forces_wide_even_with_low_score():
    # OR-conservative: any signal pointing to "high coupling" wins.
    plan = adaptive_inner_loop_budget(coupling_score=0.1, selected="joint")
    assert plan.tier == "wide"


def test_unclassified_verdict_is_conservative_wide():
    plan = adaptive_inner_loop_budget(coupling_score=0.1, verdict="SOME_NEW_UNSEEN_VERDICT")
    assert plan.tier == "wide"


def test_verdict_sets_disjoint_and_nonempty():
    assert NEEDS_WIDE_BUDGET_VERDICTS & NARROW_BUDGET_OK_VERDICTS == set()
    assert NEEDS_WIDE_BUDGET_VERDICTS
    assert NARROW_BUDGET_OK_VERDICTS


def test_narrow_budget_ok_verdict_alone_is_narrow():
    plan = adaptive_inner_loop_budget(verdict="ANCHOR_PROBED_LOW_RISK")
    assert plan.tier == "narrow"


@pytest.mark.parametrize(
    "manifest_name,expected_tier",
    [
        ("pyramid_lidar_partition.yaml", "wide"),
        ("pyramid_camera_partition.yaml", "wide"),
        ("codriving_partition.yaml", "narrow"),
        ("v2xvit_partition.yaml", "wide"),  # attention-fusion gate unresolved -> conservative wide
        ("attfuse_partition.yaml", "wide"),  # fusion-uncovered -> conservative wide
        ("fcooper_partition.yaml", "wide"),  # low-confidence pending anchor -> conservative wide
    ],
)
def test_from_manifest_matches_known_ground_truth(manifest_name, expected_tier):
    plan = from_manifest(PARTITIONS / manifest_name)
    assert plan.tier == expected_tier, (manifest_name, plan.to_dict())


def test_from_manifest_total_measurements_property():
    plan = from_manifest(PARTITIONS / "codriving_partition.yaml")
    assert plan.total_real_measurements == plan.k_per_round * plan.n_rounds
