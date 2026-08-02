from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import replace
from typing import Any, Mapping, Sequence

import pytest

from framework.stage5.single_target_search_v2 import (
    build_measurement_request,
    validate_search_task,
)
from framework.stage7 import actual_v3_selector_v2 as selector
from framework.stage7 import search_policy_v1 as policy
from framework.tests.test_stage7_search_policy_v1 import _profile, _registry


VARIANTS = (
    "full",
    "without_surrogate",
    "without_measured_feedback",
    "backend_blind",
)


def _large_pre_scan_pool() -> list[dict[str, Any]]:
    registry = _registry()
    registry["groups"] = []
    for index in range(24):
        width = [16 + 8 * index, 32 + 16 * index, 64 + 32 * index]
        group_id = "pyramid|" + "x".join(map(str, width))
        registry["groups"].append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "source_status": "ready",
                "source_evidence_sha256": hashlib.sha256(
                    group_id.encode()
                ).hexdigest(),
                "source_contract": {
                    "checkpoint_path": f"/frozen/{group_id}.pth",
                    "onnx_path": f"/frozen/{group_id}.onnx",
                },
                "graph_features": {
                    "group_id": group_id,
                    "model": "pyramid",
                    "width": width,
                    "conv_count": 20 + index,
                },
            }
        )
    return policy.prepare_frozen_candidate_pools(
        registry,
        raw_profile=_profile(),
        measured_row_ids=set(),
    )["pre_scan_candidates"]


def _promoted(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    task: Any,
) -> list[dict[str, Any]]:
    contract = validate_search_task(task)
    result = []
    for index, source in enumerate(selected_rows):
        graph = {
            **copy.deepcopy(dict(source["graph_features"])),
            "graph_feature_provenance": "materialized_onnx_extracted_v1",
            "actual_marker": index,
        }
        row = {
            **copy.deepcopy(dict(source)),
            "task_id": task.task_id,
            "model": task.target_model,
            "hardware_id": task.hardware_id,
            "capability_profile_id": contract["capability_profile_id"],
            "dispatch_key": "tvm_auto",
            "task_sha256": contract["task_sha256"],
            "training_source": "online_feedback",
            "terminal_status": "measured_success_gold",
            "latency_ms": 2.0 + index,
            "energy_j": 0.2 + index,
            "ap30": 0.9,
            "ap50": 0.8,
            "ap70": 0.7 - index / 100.0,
            "graph_features": graph,
            "feedback_feature_contract": "actual_feedback_v3",
            "graph_feature_promotion_schema": (
                "stage5_actual_feedback_promotion_v3"
            ),
            "materialized_graph_features_sha256": policy.canonical_sha256(
                graph
            ),
        }
        row["actual_feedback_row_sha256"] = policy.canonical_sha256(row)
        result.append(row)
    return result


class RecordingRoundSelector:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(copy.deepcopy(kwargs))
        selected = set(kwargs["selected_ids"])
        rows = [
            copy.deepcopy(dict(row))
            for row in kwargs["candidate_pool"]
            if str(row["row_id"]) not in selected
        ][:4]
        request = build_measurement_request(
            task=kwargs["task"],
            selected_rows=rows,
            round_index=kwargs["round_index"],
        )
        result = {
            "acquisition": {
                "policy": (
                    "uniform_random_without_replacement"
                    if kwargs["variant"] == "without_surrogate"
                    else "predicted_frontier_diversity"
                ),
                "selected_row_ids": [str(row["row_id"]) for row in rows],
                "selected_rows": rows,
            },
            "measurement_request": request,
        }
        if kwargs["variant"] == "without_measured_feedback":
            result["a2_frozen"] = (
                kwargs["a2_frozen"]
                if kwargs["a2_frozen"] is not None
                else {
                    "frozen_payload_sha256": "a" * 64,
                    "bundle_sha256": "b" * 64,
                }
            )
        if kwargs["variant"] == "backend_blind":
            result["backend_blind_audit"] = {
                "full_feature_names": ["width:axis0", "cap:operator_coverage"],
                "blind_feature_names": ["width:axis0"],
                "removed_names": ["cap:operator_coverage"],
                "leakage_verdict": "no_forbidden_features",
                "prediction_deltas": [{"row_id": row["row_id"]} for row in rows],
                "selected_id_overlap": {
                    "full_selected_row_ids": [],
                    "blind_selected_row_ids": [row["row_id"] for row in rows],
                    "overlap_row_ids": [],
                    "overlap_count": 0,
                },
            }
        return result


def _dependencies(recorder: RecordingRoundSelector) -> Any:
    return replace(selector.DEFAULT_DEPENDENCIES, select_round=recorder)


def _select(**kwargs: Any) -> Any:
    dependencies = kwargs.pop("dependencies")
    return selector._select_actual_v3_pre_scan_round_with_dependencies(
        **kwargs,
        dependencies=dependencies,
    )


def test_default_dependency_is_existing_stage5_backed_round_selector() -> None:
    assert (
        selector.DEFAULT_DEPENDENCIES.select_round
        is policy.select_stage7_round
    )
    result = selector.select_actual_v3_pre_scan_round(
        variant="without_surrogate",
        seed=20260718,
        round_index=0,
        task=policy.build_stage7_task(_profile()),
        pre_scan_pool=_large_pre_scan_pool(),
        initial_rows=[],
        initial_graph_features=[],
        capability_profiles=[_profile()],
        closure={},
    )
    assert (
        result.selection["acquisition"]["policy"]
        == "uniform_random_without_replacement"
    )
    assert result.audit["formal_dependency_contract"] is True
    assert result.audit["execution_path_claim_allowed"] is True
    assert result.audit["dependency_identities"] == {
        "select_round": {
            "module": "framework.stage7.search_policy_v1",
            "qualname": "select_stage7_round",
        },
        "validate_feedback_history": {
            "module": "framework.stage5.single_target_search_v2",
            "qualname": "validate_task_feedback_history",
        },
    }
    assert result.audit["reviewed_source_sha256"] == {
        "framework/stage7/search_policy_v1.py": (
            "23207c57f609f8f127b2b0fa4ba3d44d16bee817e973f596380fcf8597b0ba7c"
        ),
        "framework/stage5/single_target_search_v2.py": (
            "7d3694a82cee4e9f9265471fd32371d1012279a71a6c150bf8443240b870df35"
        ),
        "framework/stage5/production_search_v1.py": (
            "6d373c312e236222d7630d91deb8cb122d5c928b464e5d1b60f7070ea33db2e2"
        ),
    }


def test_public_signature_has_no_dependencies_and_rejects_injection() -> None:
    assert "dependencies" not in inspect.signature(
        selector.select_actual_v3_pre_scan_round
    ).parameters
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        selector.select_actual_v3_pre_scan_round(
            variant="without_surrogate",
            seed=20260718,
            round_index=0,
            task=policy.build_stage7_task(_profile()),
            pre_scan_pool=_large_pre_scan_pool(),
            initial_rows=[],
            initial_graph_features=[],
            capability_profiles=[_profile()],
            closure={},
            dependencies=_dependencies(RecordingRoundSelector()),
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_four_rounds_freeze_ordered_unique_ids_and_variant_paths(
    variant: str,
) -> None:
    assert "cache" not in inspect.signature(
        selector.select_actual_v3_pre_scan_round
    ).parameters
    task = policy.build_stage7_task(_profile())
    pool = _large_pre_scan_pool()
    recorder = RecordingRoundSelector()
    prior_ids: list[str] = []
    a2_frozen = None

    for round_index in range(4):
        prior_rows = [
            row for row_id in prior_ids for row in pool if row["row_id"] == row_id
        ]
        result = _select(
            variant=variant,
            seed=20260718,
            round_index=round_index,
            task=task,
            pre_scan_pool=pool,
            initial_rows=[],
            initial_graph_features=[],
            capability_profiles=[_profile()],
            closure={},
            prior_selected_ids=prior_ids,
            promoted_feedback=_promoted(prior_rows, task=task),
            a2_frozen=a2_frozen,
            dependencies=_dependencies(recorder),
        )
        chosen = result.selection["acquisition"]["selected_row_ids"]
        assert len(chosen) == len(set(chosen)) == 4
        assert not set(chosen) & set(prior_ids)
        assert result.audit["selected_row_ids"] == chosen
        assert result.audit["selected_row_ids_sha256"] == policy.canonical_sha256(
            chosen
        )
        assert (
            result.audit["logical_request_sha256"]
            == result.selection["measurement_request"][
                "measurement_request_sha256"
            ]
        )
        assert result.audit["cache_visible_during_selection"] is False
        assert result.audit["cache_input_accepted"] is False
        call = recorder.calls[-1]
        if variant in {"full", "backend_blind"}:
            assert [row["row_id"] for row in call["feedback_rows"]] == prior_ids
            assert result.audit["feedback_history_validation"] == {
                "task_id": "S7-PYR-TVM",
                "completed_rounds": round_index,
                "feedback_rows": round_index * 4,
                "task_sha256": validate_search_task(task)["task_sha256"],
            }
        else:
            assert call["feedback_rows"] == []
        if variant == "without_surrogate":
            assert result.audit["surrogate_calls"] is None
            assert result.audit["uncertainty_calls"] is None
            assert result.audit["predicted_frontier_calls"] is None
        if variant == "without_measured_feedback":
            assert result.audit["bundle_refit_calls_after_initial"] is None
            assert result.audit["actual_graph_feedback_rows"] == 0
            assert result.audit["feedback_metrics_consumed"] == 0
            if round_index:
                assert call["a2_frozen"] == a2_frozen
                assert call["a2_frozen"] is not a2_frozen
            a2_frozen = result.selection["a2_frozen"]
        if variant == "backend_blind":
            assert result.audit["fixed_dispatch_key"] == "tvm_auto"
            assert result.audit["removed_feature_names"]
            assert result.audit["leakage_verdict"] == "no_forbidden_features"
            assert result.audit["prediction_delta_count"] == 4
            assert result.audit["selected_id_overlap"]["overlap_count"] == 0
        prior_ids.extend(chosen)

    assert len(prior_ids) == 16


def test_prior_ids_reject_unordered_set() -> None:
    with pytest.raises(TypeError, match="ordered Sequence"):
        _select(
            variant="without_surrogate",
            seed=20260718,
            round_index=1,
            task=policy.build_stage7_task(_profile()),
            pre_scan_pool=_large_pre_scan_pool(),
            initial_rows=[],
            initial_graph_features=[],
            capability_profiles=[_profile()],
            closure={},
            prior_selected_ids={"a", "b", "c", "d"},
            promoted_feedback=[],
            dependencies=_dependencies(RecordingRoundSelector()),
        )


def test_feedback_order_must_exactly_match_chronological_selected_ids() -> None:
    task = policy.build_stage7_task(_profile())
    pool = _large_pre_scan_pool()
    prior_ids = [row["row_id"] for row in pool[:4]]
    feedback = list(reversed(_promoted(pool[:4], task=task)))

    with pytest.raises(ValueError, match="chronological order"):
        _select(
            variant="full",
            seed=20260718,
            round_index=1,
            task=task,
            pre_scan_pool=pool,
            initial_rows=[],
            initial_graph_features=[],
            capability_profiles=[_profile()],
            closure={},
            prior_selected_ids=prior_ids,
            promoted_feedback=feedback,
            dependencies=_dependencies(RecordingRoundSelector()),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("task_id", "S7-WRONG", "task_id drift"),
        ("model", "codriving", "model drift"),
        ("hardware_id", "a100", "hardware_id drift"),
        ("capability_profile_id", "h800-trt", "capability_profile_id drift"),
        ("dispatch_key", "trt_engine", "dispatch_key drift"),
        ("task_sha256", "0" * 64, "task_sha256 drift"),
        ("training_source", "initial_coldstart", "training_source drift"),
        ("terminal_status", "running", "non-terminal"),
    ),
)
def test_full_rejects_cross_task_trt_or_nonterminal_feedback(
    field: str,
    value: str,
    message: str,
) -> None:
    task = policy.build_stage7_task(_profile())
    pool = _large_pre_scan_pool()
    prior_ids = [row["row_id"] for row in pool[:4]]
    feedback = _promoted(pool[:4], task=task)
    feedback[0][field] = value
    feedback[0].pop("actual_feedback_row_sha256")
    feedback[0]["actual_feedback_row_sha256"] = policy.canonical_sha256(
        feedback[0]
    )

    with pytest.raises(ValueError, match=message):
        _select(
            variant="full",
            seed=20260718,
            round_index=1,
            task=task,
            pre_scan_pool=pool,
            initial_rows=[],
            initial_graph_features=[],
            capability_profiles=[_profile()],
            closure={},
            prior_selected_ids=prior_ids,
            promoted_feedback=feedback,
            dependencies=_dependencies(RecordingRoundSelector()),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "feedback_feature_contract",
            "candidate_graph",
            "actual-feedback-v3 contract",
        ),
        (
            "materialized_graph_features_sha256",
            "0" * 64,
            "actual graph feedback",
        ),
        ("actual_feedback_row_sha256", "0" * 64, "promoted feedback SHA"),
    ),
)
def test_backend_blind_rejects_graph_promotion_or_sha_drift(
    field: str,
    value: str,
    message: str,
) -> None:
    task = policy.build_stage7_task(_profile())
    pool = _large_pre_scan_pool()
    prior_ids = [row["row_id"] for row in pool[:4]]
    feedback = _promoted(pool[:4], task=task)
    feedback[0][field] = value
    if field != "actual_feedback_row_sha256":
        feedback[0].pop("actual_feedback_row_sha256")
        feedback[0]["actual_feedback_row_sha256"] = policy.canonical_sha256(
            feedback[0]
        )

    with pytest.raises(ValueError, match=message):
        _select(
            variant="backend_blind",
            seed=20260718,
            round_index=1,
            task=task,
            pre_scan_pool=pool,
            initial_rows=[],
            initial_graph_features=[],
            capability_profiles=[_profile()],
            closure={},
            prior_selected_ids=prior_ids,
            promoted_feedback=feedback,
            dependencies=_dependencies(RecordingRoundSelector()),
        )


def test_full_rejects_non_materialized_graph_provenance() -> None:
    task = policy.build_stage7_task(_profile())
    pool = _large_pre_scan_pool()
    prior_ids = [row["row_id"] for row in pool[:4]]
    feedback = _promoted(pool[:4], task=task)
    feedback[0]["graph_features"]["graph_feature_provenance"] = (
        "coldstart_width_conditioned_surrogate_v1"
    )
    feedback[0]["materialized_graph_features_sha256"] = (
        policy.canonical_sha256(feedback[0]["graph_features"])
    )
    feedback[0].pop("actual_feedback_row_sha256")
    feedback[0]["actual_feedback_row_sha256"] = policy.canonical_sha256(
        feedback[0]
    )

    with pytest.raises(ValueError, match="actual graph feedback"):
        _select(
            variant="full",
            seed=20260718,
            round_index=1,
            task=task,
            pre_scan_pool=pool,
            initial_rows=[],
            initial_graph_features=[],
            capability_profiles=[_profile()],
            closure={},
            prior_selected_ids=prior_ids,
            promoted_feedback=feedback,
            dependencies=_dependencies(RecordingRoundSelector()),
        )
