"""Production contracts and acquisition for the Stage5 two-model search."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor

from framework.stage2.canonical_search_v3 import validate_capability_profile
from framework.stage4.cost_model_selection_v1 import encode_rows

try:  # pragma: no cover - fallback is covered in lean environments.
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]


SCHEMA_VERSION = "stage5_production_search_v1"
TARGETS = ("latency_ms", "energy_j", "ap70")
EXPECTED_HEADS = {
    "latency_ms": "extra_trees_log",
    "energy_j": "extra_trees_log",
    "ap70": "lgbm_huber_residual",
}
EXPECTED_UNCERTAINTY = "lgbm_quantile_plus_group_conformal"
EXPECTED_ACQUISITION = "predicted_frontier_diversity"
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
ALLOWED_SOURCES = {"initial_coldstart", "online_feedback"}


@dataclass
class ProductionBundle:
    manifest: dict[str, Any]
    feature_names: tuple[str, ...]
    graph_feature_names: tuple[str, ...]
    value_heads: dict[str, Any]
    interval_heads: dict[str, tuple[Any, Any, Any]]
    conformal_corrections: dict[str, dict[str, float]]
    model_anchors: dict[str, float]


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def _validate_profiles(profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated = [validate_capability_profile(profile) for profile in profiles]
    if {str(profile["dispatch_key"]) for profile in validated} != {
        "tvm_auto",
        "trt_engine",
    }:
        raise ValueError("Stage5 requires exactly the frozen TVM-auto/TRT capability pair")
    if len(validated) != 2:
        raise ValueError("Stage5 requires exactly two capability profiles")
    return validated


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        groups[str(source["group_id"])].append(dict(source))
    return dict(groups)


def _validate_complete_groups(rows: Sequence[Mapping[str, Any]]) -> None:
    groups = _group_rows(rows)
    profile_dispatch: dict[str, str] = {}
    for group_id, group_rows in groups.items():
        arms = {
            (str(row.get("dispatch_key")), str(row.get("q_mode")))
            for row in group_rows
        }
        if len(group_rows) != 4 or arms != EXPECTED_ARMS:
            raise ValueError(f"incomplete four-arm group: {group_id}")
        identities = {
            (
                str(row.get("model") or ""),
                tuple(int(value) for value in row.get("width") or []),
            )
            for row in group_rows
        }
        if len(identities) != 1:
            raise ValueError(f"four-arm group identity drift: {group_id}")
        model, width = next(iter(identities))
        if len(width) != 3 or group_id != f"{model}|{'x'.join(map(str, width))}":
            raise ValueError(f"four-arm group identity drift: {group_id}")
        for row in group_rows:
            profile = str(row.get("capability_profile_id") or "")
            dispatch = str(row.get("dispatch_key") or "")
            if not profile or (
                profile in profile_dispatch and profile_dispatch[profile] != dispatch
            ):
                raise ValueError("capability profile/dispatch identity drift")
            profile_dispatch[profile] = dispatch
        ids = [_row_id(row) for row in group_rows]
        if any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"empty or duplicate row identity: {group_id}")


def _value_training_view(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    excluded: list[str] = []
    for group_id, group_rows in sorted(_group_rows(rows).items()):
        if all(
            str(row.get("terminal_status")) == "measured_success_gold"
            and all(_finite(row.get(target)) for target in TARGETS)
            for row in group_rows
        ):
            selected.extend(group_rows)
        else:
            invalid_statuses = {
                str(row.get("terminal_status") or "")
                for row in group_rows
                if str(row.get("terminal_status")) != "measured_success_gold"
            }
            if invalid_statuses - {"feasibility_failure", "numerical_feasibility_failure"}:
                raise ValueError(
                    f"unsupported non-value terminal status in {group_id}: {sorted(invalid_statuses)}"
                )
            excluded.append(group_id)
    if not selected:
        raise ValueError("no complete finite four-arm groups for value-model training")
    return selected, excluded


def validate_stage5_contract(
    closure: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    training_view_policy: str = "inherit_stage4_feedback",
) -> dict[str, Any]:
    if closure.get("schema_version") != "stage4_p1_p3_closure_audit_v1":
        raise ValueError("unexpected Stage4 closure schema")
    if closure.get("stage4_closed") is not True or closure.get("stage5_search_ready") is not True:
        raise ValueError("Stage4 closure does not admit Stage5 search")
    if closure.get("canonical_value_heads") != EXPECTED_HEADS:
        raise ValueError(f"Stage5 requires frozen heads: {EXPECTED_HEADS}")
    if closure.get("uncertainty_policy") != EXPECTED_UNCERTAINTY:
        raise ValueError(f"Stage5 requires {EXPECTED_UNCERTAINTY}")
    if closure.get("selected_acquisition_policy") != EXPECTED_ACQUISITION:
        raise ValueError(f"Stage5 requires {EXPECTED_ACQUISITION}")
    _validate_profiles(capability_profiles)
    source_rows = [dict(row) for row in rows]
    if not source_rows:
        raise ValueError("Stage5 training rows must not be empty")
    _validate_complete_groups(source_rows)
    if any(str(row.get("q_mode")) not in {"fp16", "int8"} for row in source_rows):
        raise ValueError("Stage5 q_mode must be fp16 or int8")
    if any(
        row.get("mixed_policy_id") not in {None, "", "none"}
        for row in source_rows
    ):
        raise ValueError("mixed policy is forbidden in Stage5 production training")
    value_rows, excluded_value_groups = _value_training_view(source_rows)
    sources: dict[str, int] = defaultdict(int)
    for row in source_rows:
        source = str(row.get("training_source") or "")
        if source not in ALLOWED_SOURCES:
            raise ValueError(f"unexpected training source role: {source or '<empty>'}")
        sources[source] += 1
    if training_view_policy not in {"inherit_stage4_feedback", "initial_coldstart_only"}:
        raise ValueError(f"unsupported Stage5 training view policy: {training_view_policy}")
    expected_sources = closure.get("training_source_rows")
    excluded_stage4_feedback_rows = 0
    if isinstance(expected_sources, Mapping):
        expected = {str(key): int(value) for key, value in expected_sources.items()}
        if sources.get("initial_coldstart", 0) != expected.get("initial_coldstart", 0):
            raise ValueError("initial_coldstart count drift from Stage4 closure")
        if training_view_policy == "inherit_stage4_feedback":
            if sources.get("online_feedback", 0) < expected.get("online_feedback", 0):
                raise ValueError("online_feedback cannot remove Stage4 evidence")
        else:
            if sources.get("online_feedback", 0) != 0 or set(sources) != {"initial_coldstart"}:
                raise ValueError("initial_coldstart_only forbids online or diagnostic evidence")
            excluded_stage4_feedback_rows = expected.get("online_feedback", 0)
    graph_by_group = {str(item["group_id"]): item for item in graph_features}
    missing_graphs = set(_group_rows(source_rows)) - set(graph_by_group)
    if missing_graphs:
        raise ValueError(f"training graph features missing: {sorted(missing_graphs)}")
    return {
        "schema_version": "stage5_input_contract_audit_v1",
        "training_row_count": len(source_rows),
        "group_count": len(_group_rows(source_rows)),
        "value_training_row_count": len(value_rows),
        "value_training_group_count": len(_group_rows(value_rows)),
        "excluded_value_groups": excluded_value_groups,
        "training_source_rows": dict(sorted(sources.items())),
        "training_view_policy": training_view_policy,
        "excluded_stage4_feedback_rows": excluded_stage4_feedback_rows,
        "four_arm_groups_complete": True,
        "canonical_value_heads": dict(EXPECTED_HEADS),
        "uncertainty_policy": EXPECTED_UNCERTAINTY,
        "acquisition_policy": EXPECTED_ACQUISITION,
        "training_rows_sha256": _sha(source_rows),
        "graph_features_sha256": _sha(list(graph_features)),
        "capability_profiles_sha256": _sha(list(capability_profiles)),
    }


def build_candidate_manifest(
    source_registry: Mapping[str, Any],
    *,
    measured_group_ids: set[str],
    frozen_holdout: Mapping[str, Any],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if source_registry.get("schema_version") != "stage5_candidate_source_registry_v1":
        raise ValueError("unexpected Stage5 candidate source registry schema")
    profiles = _validate_profiles(capability_profiles)
    holdout_ids = {
        str(group["group_id"])
        for group in (frozen_holdout.get("groups") or [])
    }
    groups = source_registry.get("groups")
    if not isinstance(groups, list):
        raise ValueError("candidate source registry groups must be a list")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    observed_ids: set[str] = set()
    for source in groups:
        group = copy.deepcopy(dict(source))
        group_id = str(group.get("group_id") or "")
        if not group_id or group_id in observed_ids:
            raise ValueError("candidate source registry has empty or duplicate group_id")
        observed_ids.add(group_id)
        model = str(group.get("model") or "")
        width = [int(value) for value in group.get("width") or []]
        if model not in {"pyramid", "codriving"} or len(width) != 3:
            raise ValueError(f"invalid candidate identity: {group_id}")
        if group_id != f"{model}|{'x'.join(map(str, width))}":
            raise ValueError(f"candidate group identity mismatch: {group_id}")
        if group_id in measured_group_ids:
            excluded.append({"group_id": group_id, "reason": "already_measured"})
            continue
        if group_id in holdout_ids:
            excluded.append({"group_id": group_id, "reason": "frozen_independent_holdout"})
            continue
        if group.get("source_status") not in {"ready", "materializable"}:
            excluded.append({"group_id": group_id, "reason": "source_not_ready"})
            continue
        if not isinstance(group.get("graph_features"), Mapping):
            raise ValueError(f"candidate graph features missing: {group_id}")
        evidence_sha = str(group.get("source_evidence_sha256") or "")
        if len(evidence_sha) != 64:
            raise ValueError(f"candidate source evidence SHA missing: {group_id}")
        eligible.append(group)
    rows = []
    for group in eligible:
        for profile in profiles:
            for q_mode in ("fp16", "int8"):
                profile_id = str(profile["capability_profile_id"])
                group_id = str(group["group_id"])
                row_id = f"{group_id}|q={q_mode}|profile={profile_id}"
                rows.append(
                    {
                        "schema_version": "stage5_candidate_row_v1",
                        "row_id": row_id,
                        "manifest_job_id": row_id,
                        "group_id": group_id,
                        "model": group["model"],
                        "width": list(group["width"]),
                        "genome": [*group["width"], q_mode],
                        "strategy_id": f"q={q_mode}",
                        "q_mode": q_mode,
                        "capability_profile_id": profile_id,
                        "dispatch_key": profile["dispatch_key"],
                        "capability_digest": profile["capability_digest"],
                        "source_status": group["source_status"],
                        "source_contract": copy.deepcopy(group["source_contract"]),
                        "source_evidence_sha256": group["source_evidence_sha256"],
                        "graph_features": copy.deepcopy(group["graph_features"]),
                    }
                )
    _validate_complete_groups(rows) if rows else None
    return {
        "schema_version": "stage5_candidate_manifest_v1",
        "registry_group_count": len(groups),
        "eligible_group_count": len(eligible),
        "eligible_row_count": len(rows),
        "excluded_groups": excluded,
        "rows": sorted(rows, key=_row_id),
    }


def _stable_calibration_groups(rows: Sequence[Mapping[str, Any]], seed: int) -> set[str]:
    by_model: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_model[str(row["model"])].add(str(row["group_id"]))
    calibration: set[str] = set()
    for model, groups in sorted(by_model.items()):
        ordered = sorted(
            groups,
            key=lambda group_id: hashlib.sha256(f"{seed}|{model}|{group_id}".encode()).hexdigest(),
        )
        if len(ordered) < 2:
            raise ValueError(f"at least two groups are required for model {model}")
        calibration.update(ordered[: max(1, math.ceil(0.20 * len(ordered)))])
    return calibration


def _extra_trees(seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=seed,
        n_jobs=1,
    )


def _lgbm_huber(seed: int):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="huber",
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=7,
            max_depth=4,
            min_child_samples=4,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbosity=-1,
            random_state=seed,
            n_jobs=1,
        )
    return GradientBoostingRegressor(loss="huber", random_state=seed)


def _quantile(alpha: float, seed: int):
    if LGBMRegressor is not None:
        return LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=8,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=4,
            min_child_samples=2,
            random_state=seed,
            verbosity=-1,
            n_jobs=1,
        )
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=8,
        learning_rate=0.05,
        max_depth=2,
        random_state=seed,
    )


def _predict_model(model: Any, matrix: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        return np.asarray(model.predict(matrix), dtype=float)


def _graph_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name.removeprefix("graph:") for name in feature_names if name.startswith("graph:"))


def fit_production_bundle(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    *,
    seed: int = 20260717,
    training_view_policy: str = "inherit_stage4_feedback",
) -> ProductionBundle:
    contract = validate_stage5_contract(
        closure,
        rows,
        graph_features,
        capability_profiles,
        training_view_policy=training_view_policy,
    )
    all_source_rows = [dict(row) for row in rows]
    source_rows, excluded_value_groups = _value_training_view(all_source_rows)
    encoded = encode_rows(source_rows, graph_features, capability_profiles)
    x = encoded.matrix
    anchors = {
        model: float(
            np.median(
                [float(row["ap70"]) for row in source_rows if str(row["model"]) == model]
            )
        )
        for model in sorted({str(row["model"]) for row in source_rows})
    }
    value_heads: dict[str, Any] = {}
    for target in ("latency_ms", "energy_j"):
        model = _extra_trees(seed)
        model.fit(x, np.log1p([float(row[target]) for row in source_rows]))
        value_heads[target] = model
    ap_model = _lgbm_huber(seed)
    ap_model.fit(
        x,
        np.asarray(
            [float(row["ap70"]) - anchors[str(row["model"])] for row in source_rows],
            dtype=float,
        ),
    )
    value_heads["ap70"] = ap_model

    calibration_groups = _stable_calibration_groups(source_rows, seed)
    fit_indices = [
        index for index, row in enumerate(source_rows) if str(row["group_id"]) not in calibration_groups
    ]
    calibration_indices = [
        index for index, row in enumerate(source_rows) if str(row["group_id"]) in calibration_groups
    ]
    interval_heads: dict[str, tuple[Any, Any, Any]] = {}
    corrections: dict[str, dict[str, float]] = {}
    for target_index, target in enumerate(TARGETS):
        models = tuple(_quantile(alpha, seed + target_index) for alpha in (0.05, 0.50, 0.95))
        y_fit = np.asarray([float(source_rows[index][target]) for index in fit_indices], dtype=float)
        for model in models:
            model.fit(x[fit_indices], y_fit)
        interval_heads[target] = models
        predictions = np.sort(
            np.vstack([_predict_model(model, x[calibration_indices]) for model in models]), axis=0
        )
        scores_by_model_group: dict[tuple[str, str], list[float]] = defaultdict(list)
        for local_index, row_index in enumerate(calibration_indices):
            row = source_rows[row_index]
            truth = float(row[target])
            score = max(
                float(predictions[0, local_index]) - truth,
                truth - float(predictions[2, local_index]),
                0.0,
            )
            scores_by_model_group[(str(row["model"]), str(row["group_id"]))].append(score)
        scores_by_model: dict[str, list[float]] = defaultdict(list)
        for (model_name, _), scores in scores_by_model_group.items():
            scores_by_model[model_name].append(max(scores))
        corrections[target] = {
            model_name: float(max(scores))
            for model_name, scores in scores_by_model.items()
        }

    manifest = {
        "schema_version": "stage5_production_model_bundle_manifest_v1",
        "canonical_value_heads": dict(EXPECTED_HEADS),
        "uncertainty_policy": EXPECTED_UNCERTAINTY,
        "acquisition_policy": EXPECTED_ACQUISITION,
        "seed": seed,
        "feature_names": list(encoded.feature_names),
        "calibration_groups": sorted(calibration_groups),
        "input_contract_row_count": len(all_source_rows),
        "value_training_row_count": len(source_rows),
        "value_training_group_count": len(_group_rows(source_rows)),
        "excluded_value_groups": excluded_value_groups,
        "training_contract": contract,
    }
    manifest["bundle_config_sha256"] = _sha(manifest)
    return ProductionBundle(
        manifest=manifest,
        feature_names=encoded.feature_names,
        graph_feature_names=_graph_feature_names(encoded.feature_names),
        value_heads=value_heads,
        interval_heads=interval_heads,
        conformal_corrections=corrections,
        model_anchors=anchors,
    )


def _candidate_matrix(
    bundle: ProductionBundle,
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    graphs = []
    seen: set[str] = set()
    for row in rows:
        group_id = str(row["group_id"])
        if group_id in seen:
            continue
        seen.add(group_id)
        source = row.get("graph_features") or {}
        graph = {
            "group_id": group_id,
            "model": row["model"],
            "width": list(row["width"]),
            **{
                name: float(source.get(name, 0.0)) if _finite(source.get(name)) else 0.0
                for name in bundle.graph_feature_names
            },
        }
        graphs.append(graph)
    encoded = encode_rows(rows, graphs, profiles)
    if encoded.feature_names != bundle.feature_names:
        raise ValueError("candidate feature schema drift from frozen Stage4 feature schema")
    return encoded.matrix


def predict_candidate_rows(
    bundle: ProductionBundle,
    rows: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [copy.deepcopy(dict(row)) for row in rows]
    matrix = _candidate_matrix(bundle, candidates, capability_profiles)
    value_predictions = {
        target: np.maximum(0.0, np.expm1(_predict_model(bundle.value_heads[target], matrix)))
        for target in ("latency_ms", "energy_j")
    }
    ap_residual = _predict_model(bundle.value_heads["ap70"], matrix)
    value_predictions["ap70"] = np.asarray(
        [
            float(ap_residual[index]) + bundle.model_anchors[str(row["model"])]
            for index, row in enumerate(candidates)
        ],
        dtype=float,
    )
    interval_predictions: dict[str, np.ndarray] = {}
    for target, models in bundle.interval_heads.items():
        interval_predictions[target] = np.sort(
            np.vstack([_predict_model(model, matrix) for model in models]), axis=0
        )
    results = []
    for index, row in enumerate(candidates):
        model_name = str(row["model"])
        intervals = {}
        for target in TARGETS:
            values = interval_predictions[target]
            correction = float(bundle.conformal_corrections[target].get(model_name, 0.0))
            lower = float(values[0, index] - correction)
            median = float(values[1, index])
            upper = float(values[2, index] + correction)
            intervals[target] = {"lower": lower, "median": median, "upper": upper}
        results.append(
            {
                **row,
                "predictions": {
                    target: float(value_predictions[target][index]) for target in TARGETS
                },
                "prediction_intervals": intervals,
                "prediction_bundle_sha256": bundle.manifest["bundle_config_sha256"],
            }
        )
    return results


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def _frontier_indices(vectors: Sequence[Sequence[float]]) -> set[int]:
    return {
        index
        for index, vector in enumerate(vectors)
        if not any(
            other != index and _dominates(vectors[other], vector)
            for other in range(len(vectors))
        )
    }


def _group_feature_vectors(
    rows: Sequence[Mapping[str, Any]], graph_features: Sequence[Mapping[str, Any]]
) -> dict[str, np.ndarray]:
    graph_by_group = {str(item["group_id"]): item for item in graph_features}
    graph_names = sorted(
        {
            str(name)
            for graph in graph_by_group.values()
            for name, value in graph.items()
            if name not in {"group_id", "model", "width", "input_dims"} and _finite(value)
        }
    )
    result = {}
    for row in rows:
        group_id = str(row["group_id"])
        if group_id in result:
            continue
        graph = row.get("graph_features") or graph_by_group.get(group_id) or {}
        result[group_id] = np.asarray(
            [*[float(value) for value in row["width"]], *[float(graph.get(name, 0.0)) for name in graph_names]],
            dtype=float,
        )
    return result


def select_predicted_frontier_diversity(
    predicted_rows: Sequence[Mapping[str, Any]],
    measured_rows: Sequence[Mapping[str, Any]],
    measured_graph_features: Sequence[Mapping[str, Any]],
    *,
    group_budget_by_model: Mapping[str, int],
) -> dict[str, Any]:
    candidates = [copy.deepcopy(dict(row)) for row in predicted_rows]
    _validate_complete_groups(candidates)
    forbidden_tokens = {"latency_ms", "energy_j", "ap30", "ap50", "ap70"}
    for row in candidates:
        leaked = forbidden_tokens & set(row)
        if leaked:
            raise ValueError(f"candidate labels visible before measurement: {sorted(leaked)}")
    groups = _group_rows(candidates)
    all_feature_rows = [*candidates, *[dict(row) for row in measured_rows]]
    vectors = _group_feature_vectors(all_feature_rows, measured_graph_features)
    matrix = np.vstack(list(vectors.values()))
    span = np.maximum(np.ptp(matrix, axis=0), 1e-12)
    normalized = {
        group_id: (vector - np.min(matrix, axis=0)) / span
        for group_id, vector in vectors.items()
    }
    measured_by_model: dict[str, list[str]] = defaultdict(list)
    for row in measured_rows:
        group_id = str(row["group_id"])
        if group_id not in measured_by_model[str(row["model"])]:
            measured_by_model[str(row["model"])].append(group_id)
    frontier_hits = {group_id: 0 for group_id in groups}
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_scope[(str(row["model"]), str(row["capability_profile_id"]))].append(row)
    for scope_rows in by_scope.values():
        objective_vectors = [
            (
                float(row["predictions"]["latency_ms"]),
                float(row["predictions"]["energy_j"]),
                -float(row["predictions"]["ap70"]),
            )
            for row in scope_rows
        ]
        for index in _frontier_indices(objective_vectors):
            frontier_hits[str(scope_rows[index]["group_id"])] += 1
    diagnostics: dict[str, dict[str, Any]] = {}
    for group_id, group_rows in groups.items():
        model = str(group_rows[0]["model"])
        references = measured_by_model.get(model) or []
        diversity = min(
            (float(np.linalg.norm(normalized[group_id] - normalized[item])) for item in references),
            default=0.0,
        )
        uncertainty = max(
            sum(
                max(
                    0.0,
                    float(row["prediction_intervals"][target]["upper"])
                    - float(row["prediction_intervals"][target]["lower"]),
                )
                / max(abs(float(row["predictions"][target])), 1e-9)
                for target in TARGETS
            )
            for row in group_rows
        )
        diagnostics[group_id] = {
            "model": model,
            "predicted_frontier_hits": int(frontier_hits[group_id]),
            "feature_diversity": diversity,
            "uncertainty": uncertainty,
        }
    selected_ids = []
    for model, budget in sorted(group_budget_by_model.items()):
        if budget <= 0:
            raise ValueError("group budget must be positive")
        model_groups = [
            group_id for group_id, payload in diagnostics.items() if payload["model"] == model
        ]
        if len(model_groups) < budget:
            raise ValueError(f"not enough eligible candidate groups for {model}")
        ranked = sorted(
            model_groups,
            key=lambda group_id: (
                -diagnostics[group_id]["predicted_frontier_hits"],
                -diagnostics[group_id]["feature_diversity"],
                -diagnostics[group_id]["uncertainty"],
                group_id,
            ),
        )
        selected_ids.extend(ranked[:budget])
    selected_rows = [row for row in candidates if str(row["group_id"]) in selected_ids]
    selected_groups = [
        {
            "group_id": group_id,
            "diagnostics": diagnostics[group_id],
            "rows": sorted(groups[group_id], key=_row_id),
        }
        for group_id in sorted(selected_ids)
    ]
    return {
        "schema_version": "stage5_predicted_frontier_diversity_selection_v1",
        "policy": EXPECTED_ACQUISITION,
        "candidate_labels_visible_before_measurement": False,
        "selected_group_ids": sorted(selected_ids),
        "selected_rows": sorted(selected_rows, key=_row_id),
        "groups": selected_groups,
        "all_group_diagnostics": diagnostics,
    }


def _success_feedback(row: Mapping[str, Any]) -> bool:
    return str(row.get("terminal_status")) == "measured_success_gold"


def _nested_finite(payload: Mapping[str, Any], paths: Sequence[tuple[str, ...]]) -> float | None:
    for path in paths:
        value: Any = payload
        for key in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(key)
        if _finite(value):
            return float(value)
    return None


def _verify_performance_metric_evidence(row: Mapping[str, Any]) -> None:
    performance_path = Path(str(row.get("performance_result_json") or ""))
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    latency = _nested_finite(
        performance,
        (("lat_p50_ms",), ("latency_ms",), ("latency", "latency_ms_p50"), ("latency", "lat_p50_ms")),
    )
    energy = _nested_finite(
        performance,
        (("energy_j",), ("energy", "energy_j"), ("energy", "joules"), ("energy", "joules_per_inference"), ("energy", "joule_per_inference"), ("energy", "energy_J")),
    )
    for key, value in {"latency_ms": latency, "energy_j": energy}.items():
        if value is None or not math.isclose(
            value, float(row[key]), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"feedback metric evidence mismatch: {_row_id(row)}:{key}")


def _verify_success_metric_evidence(row: Mapping[str, Any]) -> None:
    _verify_performance_metric_evidence(row)
    ap_path = Path(str(row.get("ap_report_path") or ""))
    report = json.loads(ap_path.read_text(encoding="utf-8"))
    ap_source = report.get("ap") if isinstance(report.get("ap"), Mapping) else report
    for key in ("ap30", "ap50", "ap70"):
        value = _nested_finite(ap_source, ((key,),))
        if value is None or not math.isclose(
            value, float(row[key]), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"feedback metric evidence mismatch: {_row_id(row)}:{key}")
    if (
        report.get("status") != "success"
        or int(report.get("processed_samples") or 0) != 1789
        or int(report.get("fallback_samples") or 0) != 0
        or int(report.get("failed_samples") or 0) != 0
    ):
        raise ValueError(f"feedback full-AP contract mismatch: {_row_id(row)}")


def _validate_feedback_rows(
    rows: Sequence[Mapping[str, Any]],
    selected_group_ids: set[str],
    measurement_request_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[dict[str, Any]]]:
    source_rows = [dict(row) for row in rows]
    if set(_group_rows(source_rows)) != selected_group_ids:
        raise ValueError("feedback groups do not exactly match the selected groups")
    _validate_complete_groups(source_rows)
    request_by_id = {_row_id(row): dict(row) for row in measurement_request_rows}
    feedback_by_id = {_row_id(row): row for row in source_rows}
    if not request_by_id or set(feedback_by_id) != set(request_by_id):
        raise ValueError("feedback rows do not exactly match the measurement request")
    identity_fields = (
        "group_id",
        "model",
        "width",
        "q_mode",
        "capability_profile_id",
        "dispatch_key",
        "genome",
        "strategy_id",
        "capability_digest",
        "source_status",
        "source_contract",
        "source_evidence_sha256",
    )
    for row_id, row in feedback_by_id.items():
        request = request_by_id[row_id]
        if any(row.get(field) != request.get(field) for field in identity_fields):
            raise ValueError(f"feedback identity drift from measurement request: {row_id}")
        if row.get("measurement_request_row_sha256") != _sha(request):
            raise ValueError(f"feedback request-row SHA drift: {row_id}")
    evidence = []
    for row in source_rows:
        source_path = Path(str(row.get("materialized_source_evidence_path") or ""))
        source_sha = str(row.get("materialized_source_evidence_sha256") or "")
        if (
            not source_path.is_file()
            or len(source_sha) != 64
            or _file_sha(source_path) != source_sha
        ):
            raise ValueError(f"materialized source evidence SHA mismatch: {_row_id(row)}")
        source_evidence = json.loads(source_path.read_text(encoding="utf-8"))
        if source_evidence.get("source_plan_sha256") != row.get("source_evidence_sha256"):
            raise ValueError(f"materialized source plan drift: {_row_id(row)}")
        evidence.append(
            {"row_id": _row_id(row), "path": str(source_path), "sha256": source_sha}
        )
        if _success_feedback(row):
            if not all(_finite(row.get(target)) for target in (*TARGETS, "ap30", "ap50")):
                raise ValueError(f"successful feedback row lacks finite metrics: {_row_id(row)}")
            pairs = (
                ("performance_result_json", "performance_result_sha256"),
                ("ap_report_path", "ap_report_sha256"),
            )
            for path_field, sha_field in pairs:
                path = Path(str(row.get(path_field) or ""))
                expected = str(row.get(sha_field) or "")
                if not path.is_file() or len(expected) != 64 or _file_sha(path) != expected:
                    raise ValueError(f"feedback evidence SHA mismatch: {_row_id(row)}:{path_field}")
                evidence.append({"row_id": _row_id(row), "path": str(path), "sha256": expected})
            _verify_success_metric_evidence(row)
        else:
            status = str(row.get("terminal_status") or "")
            if "failure" not in status or not row.get("failure_reason"):
                raise ValueError(f"invalid feasibility terminal state: {_row_id(row)}")
            failure_path = Path(str(row.get("failure_evidence_path") or ""))
            failure_sha = str(row.get("failure_evidence_sha256") or "")
            if (
                not failure_path.is_file()
                or len(failure_sha) != 64
                or _file_sha(failure_path) != failure_sha
            ):
                raise ValueError(f"failure evidence SHA mismatch: {_row_id(row)}")
            evidence.append(
                {"row_id": _row_id(row), "path": str(failure_path), "sha256": failure_sha}
            )
            if status == "numerical_feasibility_failure":
                if any(_finite(row.get(key)) for key in ("ap30", "ap50", "ap70")):
                    raise ValueError(f"numerical failure must not contain fake AP: {_row_id(row)}")
                if not all(_finite(row.get(key)) for key in ("latency_ms", "energy_j")):
                    raise ValueError(f"numerical failure must preserve real performance: {_row_id(row)}")
                performance_path = Path(str(row.get("performance_result_json") or ""))
                performance_sha = str(row.get("performance_result_sha256") or "")
                if (
                    not performance_path.is_file()
                    or len(performance_sha) != 64
                    or _file_sha(performance_path) != performance_sha
                ):
                    raise ValueError(f"feedback evidence SHA mismatch: {_row_id(row)}:performance")
                _verify_performance_metric_evidence(row)
            elif any(_finite(row.get(target)) for target in TARGETS):
                raise ValueError(f"feasibility failure must not contain fake metrics: {_row_id(row)}")
    return True, evidence


def _measured_pareto(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _success_feedback(row) and all(_finite(row.get(target)) for target in TARGETS):
            scopes[(str(row["model"]), str(row["capability_profile_id"]))].append(dict(row))
    report = {}
    for (model, profile), scope_rows in sorted(scopes.items()):
        vectors = [
            (float(row["latency_ms"]), float(row["energy_j"]), -float(row["ap70"]))
            for row in scope_rows
        ]
        indices = _frontier_indices(vectors)
        report[f"{model}|{profile}"] = {
            "row_count": len(scope_rows),
            "frontier_row_ids": sorted(_row_id(scope_rows[index]) for index in indices),
        }
    return report


def _area_2d(points: Sequence[tuple[float, float]], reference: tuple[float, float]) -> float:
    if not points:
        return 0.0
    ys = sorted({point[0] for point in points if point[0] < reference[0]})
    area = 0.0
    for index, y in enumerate(ys):
        next_y = ys[index + 1] if index + 1 < len(ys) else reference[0]
        min_z = min(point[1] for point in points if point[0] <= y)
        area += max(0.0, next_y - y) * max(0.0, reference[1] - min_z)
    return area


def _hypervolume_3d(
    points: Sequence[tuple[float, float, float]], reference: tuple[float, float, float]
) -> float:
    xs = sorted({point[0] for point in points if point[0] < reference[0]})
    volume = 0.0
    for index, x in enumerate(xs):
        next_x = xs[index + 1] if index + 1 < len(xs) else reference[0]
        active = [(point[1], point[2]) for point in points if point[0] <= x]
        volume += max(0.0, next_x - x) * _area_2d(active, reference[1:])
    return float(volume)


def _measured_hv(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _success_feedback(row) and all(_finite(row.get(target)) for target in TARGETS):
            scopes[(str(row["model"]), str(row["capability_profile_id"]))].append(dict(row))
    report = {}
    for (model, profile), scope_rows in sorted(scopes.items()):
        raw = np.asarray(
            [
                (float(row["latency_ms"]), float(row["energy_j"]), -float(row["ap70"]))
                for row in scope_rows
            ],
            dtype=float,
        )
        low = np.min(raw, axis=0)
        span = np.maximum(np.max(raw, axis=0) - low, 1e-12)
        normalized = (raw - low) / span
        report[f"{model}|{profile}"] = _hypervolume_3d(
            [tuple(map(float, point)) for point in normalized], (1.1, 1.1, 1.1)
        )
    return report


def build_feedback_round_audit(
    *,
    initial_rows: Sequence[Mapping[str, Any]],
    feedback_rows: Sequence[Mapping[str, Any]],
    selected_group_ids: set[str],
    measurement_request_rows: Sequence[Mapping[str, Any]],
    round_index: int,
) -> dict[str, Any]:
    if round_index <= 0:
        raise ValueError("feedback round index must be positive")
    verified, evidence = _validate_feedback_rows(
        feedback_rows, selected_group_ids, measurement_request_rows
    )
    combined = [*map(dict, initial_rows), *map(dict, feedback_rows)]
    return {
        "schema_version": "stage5_feedback_round_audit_v1",
        "round_index": round_index,
        "feedback_row_count": len(feedback_rows),
        "feedback_group_count": len(selected_group_ids),
        "evidence_sha_verified": verified,
        "evidence": evidence,
        "budget": {
            "new_complete_groups": len(selected_group_ids),
            "new_rows": len(feedback_rows),
            "successful_rows": sum(_success_feedback(row) for row in feedback_rows),
            "feasibility_terminal_rows": sum(not _success_feedback(row) for row in feedback_rows),
        },
        "measured_pareto": _measured_pareto(combined),
        "measured_hv": _measured_hv(combined),
        "feedback_rows_sha256": _sha(list(feedback_rows)),
    }


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json_idempotent(path: Path, payload: Any) -> None:
    content = _json_bytes(payload)
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite drifted Stage5 checkpoint: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def initialize_round_zero(
    *,
    closure: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    source_registry: Mapping[str, Any],
    frozen_holdout: Mapping[str, Any],
    output_dir: Path,
    seed: int = 20260717,
) -> dict[str, Any]:
    contract = validate_stage5_contract(
        closure, training_rows, graph_features, capability_profiles
    )
    measured_group_ids = {str(row["group_id"]) for row in training_rows}
    candidate_manifest = build_candidate_manifest(
        source_registry,
        measured_group_ids=measured_group_ids,
        frozen_holdout=frozen_holdout,
        capability_profiles=capability_profiles,
    )
    if candidate_manifest["eligible_group_count"] < 2:
        raise ValueError("Stage5 round zero requires eligible candidates for both models")
    bundle = fit_production_bundle(
        training_rows,
        graph_features,
        capability_profiles,
        closure,
        seed=seed,
    )
    predicted = predict_candidate_rows(
        bundle, candidate_manifest["rows"], capability_profiles
    )
    selection = select_predicted_frontier_diversity(
        predicted,
        training_rows,
        graph_features,
        group_budget_by_model={"pyramid": 1, "codriving": 1},
    )
    request_rows = []
    for row in selection["selected_rows"]:
        request_rows.append(
            {
                key: copy.deepcopy(row[key])
                for key in (
                    "manifest_job_id",
                    "group_id",
                    "model",
                    "width",
                    "genome",
                    "strategy_id",
                    "q_mode",
                    "capability_profile_id",
                    "capability_digest",
                    "dispatch_key",
                    "source_status",
                    "source_contract",
                    "source_evidence_sha256",
                )
            }
        )
    measurement_request = {
        "schema_version": "stage5_measurement_request_v1",
        "round_index": 0,
        "group_count": len(selection["selected_group_ids"]),
        "row_count": len(request_rows),
        "required_arm_product": sorted([list(arm) for arm in EXPECTED_ARMS]),
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "real_h800_measurement_required": True,
        "offline_replay_allowed": False,
        "rows": request_rows,
    }
    round_state = {
        "schema_version": "stage5_round_state_v1",
        "round_index": 0,
        "status": "awaiting_real_measurement",
        "offline_replay": False,
        "selected_group_ids": selection["selected_group_ids"],
        "selected_group_count": len(selection["selected_group_ids"]),
        "selected_row_count": len(request_rows),
        "source_materialization_required_groups": sorted(
            {
                str(row["group_id"])
                for row in request_rows
                if row["source_status"] == "materializable"
            }
        ),
        "training_contract_sha256": _sha(contract),
        "model_bundle_config_sha256": bundle.manifest["bundle_config_sha256"],
        "model_bundle_manifest_sha256": _sha(bundle.manifest),
        "candidate_manifest_sha256": _sha(candidate_manifest),
        "selection_sha256": _sha(selection),
        "measurement_request_sha256": _sha(measurement_request),
        "resume_contract": {
            "immutable_round_zero": True,
            "feedback_must_match_selected_groups": True,
            "feedback_evidence_sha_required": True,
        },
    }
    paths = {
        "input_contract": output_dir / "coldstart_contract.json",
        "model_bundle_manifest": output_dir / "model_bundle_manifest.json",
        "candidate_manifest": output_dir / "candidate_manifest.json",
        "predictions": output_dir / "round_00/predicted_candidates.json",
        "selection": output_dir / "round_00/acquisition.json",
        "measurement_request": output_dir / "round_00/measurement_request.json",
        "round_state": output_dir / "round_00/round_state.json",
    }
    payloads = {
        "input_contract": contract,
        "model_bundle_manifest": bundle.manifest,
        "candidate_manifest": candidate_manifest,
        "predictions": {
            "schema_version": "stage5_candidate_predictions_v1",
            "row_count": len(predicted),
            "rows": predicted,
        },
        "selection": selection,
        "measurement_request": measurement_request,
        "round_state": round_state,
    }
    for name, path in paths.items():
        _write_json_idempotent(path, payloads[name])
    return {
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output_dir),
        "selected_group_ids": selection["selected_group_ids"],
        "round_state_sha256": _file_sha(paths["round_state"]),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def advance_search_round(
    *,
    closure: Mapping[str, Any],
    initial_training_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    source_registry: Mapping[str, Any],
    frozen_holdout: Mapping[str, Any],
    feedback_rows: Sequence[Mapping[str, Any]],
    measurement_request: Mapping[str, Any],
    selected_group_ids: set[str],
    output_dir: Path,
    round_index: int = 1,
    seed: int = 20260717,
) -> dict[str, Any]:
    """Verify real feedback, refit frozen head families, and emit the next round."""
    if round_index <= 0:
        raise ValueError("advance_search_round requires a positive round index")
    previous_measurement_request = measurement_request
    audit = build_feedback_round_audit(
        initial_rows=initial_training_rows,
        feedback_rows=feedback_rows,
        selected_group_ids=selected_group_ids,
        measurement_request_rows=measurement_request.get("rows") or [],
        round_index=round_index,
    )
    registry_groups = {
        str(group["group_id"]): dict(group)
        for group in source_registry.get("groups") or []
    }
    missing = selected_group_ids - set(registry_groups)
    if missing:
        raise ValueError(f"feedback graph features missing from registry: {sorted(missing)}")
    enriched_feedback = [
        {
            **copy.deepcopy(dict(row)),
            "training_source": "online_feedback",
        }
        for row in feedback_rows
    ]
    combined_rows = [*map(dict, initial_training_rows), *enriched_feedback]
    graph_by_group = {
        str(graph["group_id"]): copy.deepcopy(dict(graph))
        for graph in initial_graph_features
    }
    for group_id in sorted(selected_group_ids):
        source = registry_groups[group_id]
        graph_by_group[group_id] = {
            "group_id": group_id,
            "model": source["model"],
            "width": list(source["width"]),
            **copy.deepcopy(dict(source.get("graph_features") or {})),
        }
    combined_graphs = [graph_by_group[group_id] for group_id in sorted(graph_by_group)]
    bundle = fit_production_bundle(
        combined_rows,
        combined_graphs,
        capability_profiles,
        closure,
        seed=seed + round_index,
    )
    candidate_manifest = build_candidate_manifest(
        source_registry,
        measured_group_ids={str(row["group_id"]) for row in combined_rows},
        frozen_holdout=frozen_holdout,
        capability_profiles=capability_profiles,
    )
    predicted = predict_candidate_rows(
        bundle, candidate_manifest["rows"], capability_profiles
    )
    selection = select_predicted_frontier_diversity(
        predicted,
        combined_rows,
        combined_graphs,
        group_budget_by_model={"pyramid": 1, "codriving": 1},
    )
    request_rows = [
        {
            key: copy.deepcopy(row[key])
            for key in (
                "manifest_job_id",
                "group_id",
                "model",
                "width",
                "genome",
                "strategy_id",
                "q_mode",
                "capability_profile_id",
                "capability_digest",
                "dispatch_key",
                "source_status",
                "source_contract",
                "source_evidence_sha256",
            )
        }
        for row in selection["selected_rows"]
    ]
    next_measurement_request = {
        "schema_version": "stage5_measurement_request_v1",
        "round_index": round_index,
        "group_count": len(selection["selected_group_ids"]),
        "row_count": len(request_rows),
        "required_arm_product": sorted([list(arm) for arm in EXPECTED_ARMS]),
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "real_h800_measurement_required": True,
        "offline_replay_allowed": False,
        "rows": request_rows,
    }
    state = {
        "schema_version": "stage5_round_state_v1",
        "round_index": round_index,
        "status": "awaiting_real_measurement",
        "offline_replay": False,
        "previous_round_feedback_verified": True,
        "selected_group_ids": selection["selected_group_ids"],
        "selected_group_count": len(selection["selected_group_ids"]),
        "selected_row_count": len(request_rows),
        "feedback_audit_sha256": _sha(audit),
        "previous_measurement_request_sha256": _sha(previous_measurement_request),
        "combined_training_rows_sha256": _sha(combined_rows),
        "model_bundle_config_sha256": bundle.manifest["bundle_config_sha256"],
        "model_bundle_manifest_sha256": _sha(bundle.manifest),
        "candidate_manifest_sha256": _sha(candidate_manifest),
        "selection_sha256": _sha(selection),
        "measurement_request_sha256": _sha(next_measurement_request),
        "resume_contract": {
            "immutable_previous_rounds": True,
            "feedback_must_match_selected_groups": True,
            "feedback_evidence_sha_required": True,
        },
    }
    round_dir = output_dir / f"round_{round_index:02d}"
    payloads = {
        "feedback_rows.json": enriched_feedback,
        "feedback_audit.json": audit,
        "model_bundle_manifest.json": bundle.manifest,
        "candidate_manifest.json": candidate_manifest,
        "predicted_candidates.json": {
            "schema_version": "stage5_candidate_predictions_v1",
            "row_count": len(predicted),
            "rows": predicted,
        },
        "acquisition.json": selection,
        "measurement_request.json": next_measurement_request,
        "round_state.json": state,
    }
    for name, payload in payloads.items():
        _write_json_idempotent(round_dir / name, payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "round_index": round_index,
        "feedback_audit": audit,
        "selected_group_ids": selection["selected_group_ids"],
        "round_state_sha256": _file_sha(round_dir / "round_state.json"),
    }


__all__ = [
    "ProductionBundle",
    "advance_search_round",
    "build_candidate_manifest",
    "build_feedback_round_audit",
    "fit_production_bundle",
    "initialize_round_zero",
    "predict_candidate_rows",
    "select_predicted_frontier_diversity",
    "validate_stage5_contract",
]
