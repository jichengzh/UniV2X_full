"""Scanner-deferred Stage7 selector over frozen Stage5 search primitives."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framework.stage5 import production_search_v1 as stage5_production
from framework.stage5 import single_target_search_v2 as stage5_search
from framework.stage5.single_target_search_v2 import SearchTask
from framework.stage7 import search_policy_v1 as search_policy
from framework.stage7.core_ablation_v2 import CORE_VARIANTS


RoundSelector = Callable[..., Mapping[str, Any]]
FeedbackHistoryValidator = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class SelectorDependencies:
    """Immutable, explicitly injectable calls into reviewed search code."""

    select_round: RoundSelector
    validate_feedback_history: FeedbackHistoryValidator


DEFAULT_DEPENDENCIES = SelectorDependencies(
    select_round=search_policy.select_stage7_round,
    validate_feedback_history=stage5_search.validate_task_feedback_history,
)
REVIEWED_SOURCE_SHA256 = {
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


def _row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in rows
    ]


def _ordered_prior_ids(
    prior_selected_ids: Sequence[str],
    *,
    round_index: int,
) -> list[str]:
    if isinstance(prior_selected_ids, (str, bytes, set, frozenset)) or not isinstance(
        prior_selected_ids, Sequence
    ):
        raise TypeError("prior_selected_ids must be an ordered Sequence")
    ordered = [str(row_id) for row_id in prior_selected_ids]
    if (
        any(not row_id for row_id in ordered)
        or len(set(ordered)) != len(ordered)
        or len(ordered) != round_index * 4
    ):
        raise ValueError("prior selected identity history drift")
    return ordered


def _ordered_feedback(
    rows: Sequence[Mapping[str, Any]],
    *,
    prior_selected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    copied = [copy.deepcopy(dict(row)) for row in rows]
    identities = _row_ids(copied)
    if identities != list(prior_selected_ids):
        raise ValueError(
            "promoted feedback must match selected IDs in chronological order"
        )
    return copied


def _validate_actual_v3_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validated = [copy.deepcopy(dict(row)) for row in rows]
    for row in validated:
        row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
        if (
            row.get("feedback_feature_contract") != "actual_feedback_v3"
            or row.get("graph_feature_promotion_schema")
            != "stage5_actual_feedback_promotion_v3"
        ):
            raise ValueError(f"actual-feedback-v3 contract drift: {row_id}")
        graph = row.get("graph_features")
        if (
            not isinstance(graph, Mapping)
            or graph.get("graph_feature_provenance")
            != "materialized_onnx_extracted_v1"
            or row.get("materialized_graph_features_sha256")
            != search_policy.canonical_sha256(dict(graph))
        ):
            raise ValueError(f"actual graph feedback drift: {row_id}")
        recorded = row.get("actual_feedback_row_sha256")
        payload = copy.deepcopy(row)
        payload.pop("actual_feedback_row_sha256", None)
        if recorded != search_policy.canonical_sha256(payload):
            raise ValueError(f"promoted feedback SHA drift: {row_id}")
    return validated


def _feedback_view(
    *,
    variant: str,
    task: SearchTask,
    round_index: int,
    ordered_rows: Sequence[Mapping[str, Any]],
    dependencies: SelectorDependencies,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    if variant not in {"full", "backend_blind"}:
        return [], None, "identity_exclusion_only"
    if round_index == 0:
        return (
            [],
            {
                "task_id": task.task_id,
                "completed_rounds": 0,
                "feedback_rows": 0,
                "task_sha256": stage5_search.validate_search_task(task)[
                    "task_sha256"
                ],
            },
            "actual_v3_online_refit",
        )
    history_audit = dict(
        dependencies.validate_feedback_history(
            ordered_rows,
            task=task,
            completed_rounds=round_index,
        )
    )
    return (
        _validate_actual_v3_rows(ordered_rows),
        history_audit,
        "actual_v3_online_refit",
    )


def _declared_path(variant: str, round_index: int) -> dict[str, int]:
    if variant == "without_surrogate":
        return {
            "surrogate_calls": 0,
            "uncertainty_calls": 0,
            "predicted_frontier_calls": 0,
            "bundle_refit_calls_after_initial": 0,
        }
    if variant == "without_measured_feedback" and round_index > 0:
        return {
            "surrogate_calls": 0,
            "uncertainty_calls": 0,
            "predicted_frontier_calls": 1,
            "bundle_refit_calls_after_initial": 0,
        }
    if variant == "backend_blind":
        return {
            "surrogate_calls": 2,
            "uncertainty_calls": 2,
            "predicted_frontier_calls": 2,
            "bundle_refit_calls_after_initial": int(round_index > 0),
        }
    return {
        "surrogate_calls": 1,
        "uncertainty_calls": 1,
        "predicted_frontier_calls": 1,
        "bundle_refit_calls_after_initial": int(
            variant == "full" and round_index > 0
        ),
    }


def _callable_identity(callable_value: Callable[..., Any]) -> dict[str, str]:
    callable_type = type(callable_value)
    return {
        "module": str(
            getattr(callable_value, "__module__", callable_type.__module__)
        ),
        "qualname": str(
            getattr(callable_value, "__qualname__", callable_type.__qualname__)
        ),
    }


def _dependency_audit(
    dependencies: SelectorDependencies,
) -> dict[str, Any]:
    identities = {
        "select_round": _callable_identity(dependencies.select_round),
        "validate_feedback_history": _callable_identity(
            dependencies.validate_feedback_history
        ),
    }
    formal = dependencies is DEFAULT_DEPENDENCIES
    if not formal:
        return {
            "formal_dependency_contract": False,
            "execution_path_claim_allowed": False,
            "dependency_identities": identities,
            "reviewed_source_sha256": None,
        }
    sources = {
        "framework/stage7/search_policy_v1.py": Path(
            str(search_policy.__file__)
        ),
        "framework/stage5/single_target_search_v2.py": Path(
            str(stage5_search.__file__)
        ),
        "framework/stage5/production_search_v1.py": Path(
            str(stage5_production.__file__)
        ),
    }
    actual = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    }
    if actual != REVIEWED_SOURCE_SHA256:
        raise ValueError("reviewed selector dependency SHA drift")
    expected_identities = {
        "select_round": {
            "module": "framework.stage7.search_policy_v1",
            "qualname": "select_stage7_round",
        },
        "validate_feedback_history": {
            "module": "framework.stage5.single_target_search_v2",
            "qualname": "validate_task_feedback_history",
        },
    }
    if identities != expected_identities:
        raise ValueError("reviewed selector callable identity drift")
    return {
        "formal_dependency_contract": True,
        "execution_path_claim_allowed": True,
        "dependency_identities": identities,
        "reviewed_source_sha256": copy.deepcopy(REVIEWED_SOURCE_SHA256),
    }


def _validate_selection(
    selection: Mapping[str, Any],
    *,
    task: SearchTask,
    prior_selected_ids: Sequence[str],
) -> tuple[list[str], str]:
    request = selection.get("measurement_request")
    acquisition = selection.get("acquisition")
    if not isinstance(request, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("selector did not emit acquisition and request")
    selected = [str(value) for value in acquisition.get("selected_row_ids") or ()]
    request_rows = request.get("rows")
    request_ids = _row_ids(request_rows if isinstance(request_rows, list) else [])
    if (
        len(selected) != task.batch_size
        or len(set(selected)) != task.batch_size
        or selected != request_ids
        or set(selected) & set(prior_selected_ids)
    ):
        raise ValueError("selected/request identity drift")
    logical_sha = str(request.get("measurement_request_sha256") or "")
    if len(logical_sha) != 64:
        raise ValueError("logical request SHA missing")
    return selected, logical_sha


def _backend_audit(
    selection: Mapping[str, Any],
    *,
    pool: Sequence[Mapping[str, Any]],
    variant: str,
) -> dict[str, Any]:
    if variant != "backend_blind":
        return {
            "fixed_dispatch_key": None,
            "removed_feature_names": [],
            "model_feature_names": [],
            "forbidden_backend_feature_names": [],
            "leakage_verdict": None,
            "prediction_delta_count": 0,
            "selected_id_overlap": None,
        }
    if {str(row.get("dispatch_key") or "") for row in pool} != {"tvm_auto"}:
        raise ValueError("backend-blind route drift")
    evidence = selection.get("backend_blind_audit")
    if not isinstance(evidence, Mapping):
        raise ValueError("backend-blind audit missing")
    full = [str(name) for name in evidence.get("full_feature_names") or ()]
    blind = [str(name) for name in evidence.get("blind_feature_names") or ()]
    removed = [str(name) for name in evidence.get("removed_names") or ()]
    forbidden = [
        name for name in full if name.startswith(("cap:", "cap_x_q:"))
    ]
    if (
        evidence.get("leakage_verdict") != "no_forbidden_features"
        or not removed
        or set(blind) & set(forbidden)
    ):
        raise ValueError("backend-blind feature leakage")
    return {
        "fixed_dispatch_key": "tvm_auto",
        "removed_feature_names": removed,
        "model_feature_names": blind,
        "forbidden_backend_feature_names": forbidden,
        "leakage_verdict": evidence["leakage_verdict"],
        "prediction_delta_count": len(evidence.get("prediction_deltas") or ()),
        "selected_id_overlap": copy.deepcopy(
            evidence.get("selected_id_overlap")
        ),
    }


def _select_actual_v3_pre_scan_round_with_dependencies(
    *,
    variant: str,
    seed: int,
    round_index: int,
    task: SearchTask,
    pre_scan_pool: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    prior_selected_ids: Sequence[str] = (),
    promoted_feedback: Sequence[Mapping[str, Any]] = (),
    a2_frozen: Mapping[str, Any] | None = None,
    dependencies: SelectorDependencies = DEFAULT_DEPENDENCIES,
) -> search_policy.V2PreScanSelection:
    """Freeze one Stage7 logical request before any cache reveal."""
    dependency_audit = _dependency_audit(dependencies)
    if variant not in CORE_VARIANTS:
        raise ValueError("unknown v2 core-ablation variant")
    task_contract = stage5_search.validate_search_task(task)
    if task_contract["task_id"] != "S7-PYR-TVM" or round_index not in range(4):
        raise ValueError("selector task or round drift")
    prior = _ordered_prior_ids(
        prior_selected_ids,
        round_index=round_index,
    )
    ordered_feedback = _ordered_feedback(
        promoted_feedback,
        prior_selected_ids=prior,
    )
    feedback_for_model, history_audit, feedback_policy = _feedback_view(
        variant=variant,
        task=task,
        round_index=round_index,
        ordered_rows=ordered_feedback,
        dependencies=dependencies,
    )
    pool = [copy.deepcopy(dict(row)) for row in pre_scan_pool]
    selection = dict(
        dependencies.select_round(
            variant=variant,
            seed=seed,
            round_index=round_index,
            task=task,
            candidate_pool=pool,
            initial_rows=copy.deepcopy(list(initial_rows)),
            initial_graph_features=copy.deepcopy(
                list(initial_graph_features)
            ),
            capability_profiles=copy.deepcopy(list(capability_profiles)),
            closure=copy.deepcopy(dict(closure)),
            selected_ids=set(prior),
            feedback_rows=feedback_for_model,
            a2_frozen=copy.deepcopy(a2_frozen),
        )
    )
    selected, logical_sha = _validate_selection(
        selection,
        task=task,
        prior_selected_ids=prior,
    )
    backend = _backend_audit(selection, pool=pool, variant=variant)
    path = (
        _declared_path(variant, round_index)
        if dependency_audit["execution_path_claim_allowed"]
        else {
            "surrogate_calls": None,
            "uncertainty_calls": None,
            "predicted_frontier_calls": None,
            "bundle_refit_calls_after_initial": None,
        }
    )
    audit = {
        "schema_version": "stage7_actual_v3_selector_audit_v2",
        "variant": variant,
        "round_index": round_index,
        "ordered_pre_scan_sha256": search_policy.canonical_sha256(pool),
        "candidate_pool": "pre_scan",
        "scanner_deferred": True,
        "prior_selected_row_count": len(prior),
        "prior_selected_row_ids_sha256": search_policy.canonical_sha256(
            prior
        ),
        "promoted_feedback_row_ids_sha256": search_policy.canonical_sha256(
            _row_ids(ordered_feedback)
        ),
        "selected_row_ids": selected,
        "selected_row_ids_sha256": search_policy.canonical_sha256(selected),
        "logical_request_sha256": logical_sha,
        "cache_visible_during_selection": False,
        "cache_input_accepted": False,
        "feedback_policy": feedback_policy,
        "feedback_metrics_consumed": len(feedback_for_model),
        "actual_graph_feedback_rows": len(feedback_for_model),
        "feedback_history_validation": history_audit,
        **dependency_audit,
        **path,
        **backend,
    }
    return search_policy.V2PreScanSelection(
        selection=copy.deepcopy(selection),
        audit=copy.deepcopy(audit),
    )


def select_actual_v3_pre_scan_round(
    *,
    variant: str,
    seed: int,
    round_index: int,
    task: SearchTask,
    pre_scan_pool: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    prior_selected_ids: Sequence[str] = (),
    promoted_feedback: Sequence[Mapping[str, Any]] = (),
    a2_frozen: Mapping[str, Any] | None = None,
) -> search_policy.V2PreScanSelection:
    """Formal selector entry point bound to frozen reviewed dependencies."""
    return _select_actual_v3_pre_scan_round_with_dependencies(
        variant=variant,
        seed=seed,
        round_index=round_index,
        task=task,
        pre_scan_pool=pre_scan_pool,
        initial_rows=initial_rows,
        initial_graph_features=initial_graph_features,
        capability_profiles=capability_profiles,
        closure=closure,
        prior_selected_ids=prior_selected_ids,
        promoted_feedback=promoted_feedback,
        a2_frozen=a2_frozen,
        dependencies=DEFAULT_DEPENDENCIES,
    )


__all__ = [
    "DEFAULT_DEPENDENCIES",
    "REVIEWED_SOURCE_SHA256",
    "SelectorDependencies",
    "select_actual_v3_pre_scan_round",
]
