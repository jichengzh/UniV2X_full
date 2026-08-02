#!/usr/bin/env python3
"""Fail-closed Stage7 ablation completeness audit and descriptive aggregator.

The caller supplies both roots.  This module has no default formal-result path and
does not discover evidence outside ``input_root``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage7.online_component_ablation_v1 import (
    classify_failure,
    validate_result_row,
)


SCHEMA_VERSION = "stage7_finalize_ablation_v1"
TASK_ID = "S7-PYR-TVM"
VARIANTS = (
    "full",
    "without_surrogate",
    "without_measured_feedback",
    "backend_blind",
    "without_capability_scan",
)
SEEDS = (20260718, 20260719, 20260720)
ROUND_COUNT = 4
BATCH_SIZE = 4
EVENT_COUNT = ROUND_COUNT * BATCH_SIZE
SUCCESS_STATUSES = {"success", "measured_success", "measured_success_gold"}
METRICS = (
    "delta_hv_auc",
    "delta_hv_at_16",
    "frontier_recall",
    "invalid_rate",
    "valid_yield",
    "iso_ap_latency_ratio",
    "iso_ap_energy_ratio",
    "exact_cache_hits",
    "exact_cache_misses",
    "gpu_hours",
    "wall_clock_hours",
)
HIGHER_IS_BETTER = {
    "delta_hv_auc",
    "delta_hv_at_16",
    "frontier_recall",
    "valid_yield",
}
OUTPUT_PATHS = (
    "aggregate/stage7_events_raw.csv",
    "aggregate/stage7_trajectories_raw.csv",
    "aggregate/stage7_aggregate.json",
    "aggregate/stage7_completeness_audit.json",
    "aggregate/stage7_isolation_audit.json",
    "aggregate/stage7_root_cause_summary.md",
    "paper/table_stage7_component_ablation.csv",
    "paper/table_stage7_component_ablation.md",
    "paper/table_stage7_execution_cost.csv",
    "paper/table_stage7_execution_cost.md",
)


class FinalizationError(ValueError):
    """Evidence failed a frozen Stage7 finalization contract."""


def _fail(message: str) -> None:
    raise FinalizationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _embedded_sha256(
    payload: Mapping[str, Any],
    field: str,
    label: str,
) -> str:
    recorded = payload.get(field)
    unsigned = {key: value for key, value in payload.items() if key != field}
    expected = _payload_sha256(unsigned)
    if recorded != expected:
        _fail(f"{label}_canonical_sha_drift")
    return str(recorded)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"missing_{label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"invalid_{label}:{error}")
    if not isinstance(value, dict):
        _fail(f"invalid_{label}:expected_object")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        _fail(f"{label}_drift")


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}_not_measured")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        _fail(f"{label}_not_measured")
    return number


def _contains_surrogate_metric(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key, item in value.items():
        field = str(key).lower()
        is_metric_field = field == "metric_source" or (
            any(metric in field for metric in ("latency", "energy", "ap", "hv", "frontier"))
            and "source" in field
        )
        if is_metric_field and "surrogate" in str(item).lower():
            return True
        if isinstance(item, Mapping) and _contains_surrogate_metric(item):
            return True
        if isinstance(item, (list, tuple)) and any(
            isinstance(nested, Mapping) and _contains_surrogate_metric(nested)
            for nested in item
        ):
            return True
    return False


def _trajectory_dir(root: Path, variant: str, seed: int) -> Path:
    return root / "variants" / variant / f"seed_{seed}"


def _validate_output_root(output_root: Path) -> None:
    if not output_root.exists():
        return
    if not output_root.is_dir():
        raise ValueError("output_root_must_be_directory")
    allowed_files = {Path(path) for path in OUTPUT_PATHS}
    allowed_directories = {
        parent
        for path in allowed_files
        for parent in path.parents
        if parent != Path(".")
    }
    for existing in output_root.rglob("*"):
        relative = existing.relative_to(output_root)
        if (
            existing.is_symlink()
            or relative not in allowed_files
            and relative not in allowed_directories
        ):
            raise ValueError(f"output_root_contains_unmanaged_files:{relative}")


def _audit_global_contracts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    scanner = _load_json(
        root / "audits" / "scanner_admission.json",
        "scanner_admission_audit",
    )
    isolation = _load_json(
        root / "audits" / "single_variable_isolation.json",
        "single_variable_isolation_audit",
    )
    pools = _load_json(
        root / "contracts" / "frozen_candidate_pools.json",
        "frozen_candidate_pools",
    )
    frozen_inputs = _load_json(
        root / "contracts" / "frozen_inputs.json",
        "frozen_inputs",
    )
    _embedded_sha256(
        frozen_inputs,
        "frozen_inputs_sha256",
        "frozen_inputs",
    )
    _require_equal(scanner.get("admission_passed"), True, "scanner_admission")
    if scanner.get("status") not in {
        "scanner_ready_discriminative",
        "scanner_ready_all_pass",
    }:
        _fail("scanner_admission_terminal_status_drift")
    for field in (
        "pre_scan_sha256",
        "scan_pass_sha256",
        "scanner_rule_sha256",
        "scanner_decision_sha256",
    ):
        if not _is_sha256(pools.get(field)):
            _fail(f"scanner_{field}_drift")
    _require_equal(
        pools.get("scanner_rule_sha256"),
        scanner.get("scanner_rule_file_sha256"),
        "scanner_rule_sha",
    )
    _require_equal(
        pools.get("scanner_decision_sha256"),
        scanner.get("scanner_decision_file_sha256"),
        "scanner_decision_sha",
    )
    variant_pool_sha = pools.get("variant_pool_sha256")
    if not isinstance(variant_pool_sha, dict) or set(variant_pool_sha) != set(VARIANTS):
        _fail("variant_pool_sha_matrix_drift")
    if any(not _is_sha256(value) for value in variant_pool_sha.values()):
        _fail("variant_pool_sha_matrix_drift")
    isolation_rows = isolation.get("audits")
    if not isinstance(isolation_rows, list):
        _fail("single_variable_isolation_drift")
    isolation_by_variant = {
        str(row.get("variant")): row
        for row in isolation_rows
        if isinstance(row, dict)
    }
    expected_isolation_variants = set(VARIANTS) - {"full"}
    _require_equal(
        set(isolation_by_variant),
        expected_isolation_variants,
        "isolation_variant_matrix",
    )
    if any(
        row.get("verdict") != "pass" for row in isolation_by_variant.values()
    ):
        _fail("single_variable_isolation_drift")
    normalized = {
        "terminal_status": scanner["status"],
        "all_candidates_pass": scanner["status"] == "scanner_ready_all_pass",
        "pre_scan_sha256": pools["pre_scan_sha256"],
        "scan_pass_sha256": pools["scan_pass_sha256"],
        "scanner_rule_sha256": pools["scanner_rule_sha256"],
        "scanner_decision_sha256": pools["scanner_decision_sha256"],
        "variant_pool_sha256": dict(variant_pool_sha),
        "frozen_input_sha256": frozen_inputs.get("frozen_inputs_sha256"),
        "source_scanner_audit": scanner,
    }
    if not _is_sha256(normalized["frozen_input_sha256"]):
        _fail("frozen_input_sha_drift")
    return normalized, isolation


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                _fail(f"invalid_{label}_line_{line_number}")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"invalid_{label}:{error}")
    return rows


def _load_scheduler_windows(root: Path) -> dict[str, dict[str, Any]]:
    lease_rows = _load_jsonl(
        root / "status" / "gpu_lease_audit.jsonl",
        "gpu_lease_audit",
    )
    occupancy_rows = _load_jsonl(
        root / "status" / "gpu_occupancy_snapshots.jsonl",
        "gpu_occupancy_snapshots",
    )
    for retry in (
        row
        for row in lease_rows
        if row.get("event") == "occupancy_drift_same_request_retry"
    ):
        if not _is_sha256(retry.get("request_sha256")):
            _fail("infrastructure_retry_request_sha_drift")
        _require_equal(
            retry.get("selected_event_budget_consumed"),
            0,
            "infrastructure_retry_budget",
        )
    attempts: dict[str, list[dict[str, Any]]] = {}
    for lease_index, lease in (
        (index, row)
        for index, row in enumerate(lease_rows)
        if row.get("event") == "batch_leased"
    ):
        request_sha = lease.get("request_sha256")
        controller_id = str(lease.get("controller_id") or "")
        gpu_uuids = lease.get("gpu_uuids")
        if not _is_sha256(request_sha) or not controller_id:
            _fail("gpu_lease_request_sha_drift")
        if (
            not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != 4
            or len(set(map(str, gpu_uuids))) != 4
        ):
            _fail("gpu_lease_cardinality_drift")
        start = _finite_number(lease.get("wall_time"), "gpu_lease_start", minimum=0.0)
        terminal: dict[str, Any] | None = None
        for later in lease_rows[lease_index + 1 :]:
            if str(later.get("controller_id") or "") != controller_id:
                continue
            if later.get("event") == "batch_leased":
                break
            if later.get("event") in {
                "controller_finished",
                "occupancy_drift_same_request_retry",
            }:
                terminal = later
                break
        if terminal is None:
            _fail("gpu_lease_missing_terminal_event")
        is_retry = terminal.get("event") == "occupancy_drift_same_request_retry"
        if is_retry:
            _require_equal(
                terminal.get("request_sha256"),
                request_sha,
                "infrastructure_retry_request_sha",
            )
            _require_equal(
                terminal.get("selected_event_budget_consumed"),
                0,
                "infrastructure_retry_budget",
            )
        elif terminal.get("returncode") != 0:
            _fail("gpu_lease_missing_successful_finish")
        end = _finite_number(terminal.get("wall_time"), "gpu_lease_end", minimum=0.0)
        if end <= start:
            _fail("gpu_lease_window_drift")
        covered = False
        for snapshot in occupancy_rows:
            wall_time = snapshot.get("wall_time")
            if isinstance(wall_time, bool) or not isinstance(wall_time, (int, float)):
                continue
            if not start <= float(wall_time) <= end:
                continue
            gpus = snapshot.get("gpus")
            if not isinstance(gpus, list):
                continue
            observed = {
                str(gpu.get("uuid"))
                for gpu in gpus
                if isinstance(gpu, dict) and gpu.get("uuid")
            }
            if set(map(str, gpu_uuids)) <= observed:
                covered = True
                break
        if not covered:
            _fail("gpu_occupancy_window_not_covered")
        attempt = {
            "controller_id": controller_id,
            "gpu_uuids": list(map(str, gpu_uuids)),
            "start_wall_time": start,
            "end_wall_time": end,
            "gpu_hours": len(gpu_uuids) * (end - start) / 3600.0,
            "terminal_event": terminal.get("event"),
        }
        request_attempts = attempts.get(str(request_sha), [])
        attempts[str(request_sha)] = [*request_attempts, attempt]
    windows: dict[str, dict[str, Any]] = {}
    for request_sha, request_attempts in attempts.items():
        successful = [
            attempt
            for attempt in request_attempts
            if attempt["terminal_event"] == "controller_finished"
        ]
        if len(successful) != 1:
            _fail("gpu_lease_missing_successful_finish")
        windows[request_sha] = {
            "controller_id": successful[0]["controller_id"],
            "controller_ids": [
                attempt["controller_id"] for attempt in request_attempts
            ],
            "gpu_uuids": successful[0]["gpu_uuids"],
            "start_wall_time": min(
                attempt["start_wall_time"] for attempt in request_attempts
            ),
            "end_wall_time": max(
                attempt["end_wall_time"] for attempt in request_attempts
            ),
            "gpu_hours": sum(
                attempt["gpu_hours"] for attempt in request_attempts
            ),
            "attempt_count": len(request_attempts),
        }
    return windows


def _audit_a2(trajectory_dir: Path) -> dict[str, str]:
    fields = (
        "bundle_sha256",
        "prediction_view_sha256",
        "training_view_sha256",
        "graph_feature_view_sha256",
    )
    frozen: dict[str, str] = {}
    for round_index in range(ROUND_COUNT):
        row = _load_json(
            trajectory_dir
            / f"round_{round_index:02d}"
            / "a2_frozen_contract.json",
            f"a2_frozen_contract_round_{round_index}",
        )
        _embedded_sha256(row, "frozen_payload_sha256", "a2_frozen_payload")
        _require_equal(
            row.get("bundle_sha256"),
            (row.get("bundle_manifest") or {}).get("bundle_config_sha256"),
            "a2_bundle_sha256",
        )
        _require_equal(
            row.get("prediction_view_sha256"),
            _payload_sha256(row.get("candidate_predictions") or []),
            "a2_prediction_view_sha256",
        )
        for field in fields:
            value = row.get(field)
            if not _is_sha256(value):
                _fail(f"a2_{field}_drift")
            if field in frozen and frozen[field] != value:
                _fail(f"a2_{field}_drift")
            frozen[field] = str(value)
    return frozen


def _audit_a3(trajectory_dir: Path) -> None:
    required_shas = (
        "full_training_matrix_sha256",
        "blind_training_matrix_sha256",
        "full_candidate_matrix_sha256",
        "blind_candidate_matrix_sha256",
    )
    for round_index in range(ROUND_COUNT):
        row = _load_json(
            trajectory_dir
            / f"round_{round_index:02d}"
            / "backend_blind_audit.json",
            f"backend_blind_audit_round_{round_index}",
        )
        _require_equal(row.get("leakage_verdict"), "pass", "a3_leakage")
        if any(not _is_sha256(row.get(field)) for field in required_shas):
            _fail("a3_matrix_sha_drift")
        full_names = row.get("full_feature_names")
        blind_names = row.get("blind_feature_names")
        removed_names = row.get("removed_names")
        if not all(isinstance(value, list) for value in (full_names, blind_names, removed_names)):
            _fail("a3_feature_schema_drift")
        _require_equal(
            set(full_names) - set(blind_names),
            set(removed_names),
            "a3_removed_feature_schema",
        )


def _audit_result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_surrogate_metric(row):
        _fail("surrogate_metric_rejected")
    terminal_status = row.get("terminal_status")
    if terminal_status in SUCCESS_STATUSES:
        _require_equal(row.get("metric_source"), "measured_artifacts", "metric_source")
    else:
        failure_kind = row.get("failure_kind")
        try:
            classification = classify_failure(str(failure_kind))
        except ValueError:
            _fail("unclassified_candidate_failure")
        if classification["consumes_selected_event_budget"] is not True:
            _fail("infrastructure_failure_consumed_event")
        _require_equal(
            row.get("consumes_selected_event_budget"),
            True,
            "candidate_failure_budget",
        )
    try:
        validated = validate_result_row(row)
    except ValueError as error:
        _fail(f"result_row_invalid:{error}")
    return dict(validated["row"])


def _objective(row: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        _finite_number(row.get("latency_ms"), "latency_ms"),
        _finite_number(row.get("energy_j"), "energy_j"),
        -_finite_number(row.get("ap70"), "ap70"),
    )


def _dominates(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(
        a < b for a, b in zip(left, right)
    )


def _frontier_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
        if not row_id:
            _fail("frontier_row_identity_missing")
        prior = unique.get(row_id)
        if prior is not None and _objective(prior) != _objective(row):
            _fail("repeated_row_metric_drift")
        unique = {**unique, row_id: row}
    items = [(row_id, _objective(row)) for row_id, row in unique.items()]
    return {
        row_id
        for index, (row_id, point) in enumerate(items)
        if not any(
            other_index != index and _dominates(other, point)
            for other_index, (_, other) in enumerate(items)
        )
    }


def _hypervolume_2d(
    points: Sequence[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    eligible = sorted(
        {
            point
            for point in points
            if point[0] < reference[0] and point[1] < reference[1]
        }
    )
    if not eligible:
        return 0.0
    xs = sorted({point[0] for point in eligible} | {reference[0]})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        active_y = [y for x, y in eligible if x <= left]
        area += (right - left) * max(0.0, reference[1] - min(active_y))
    return area


def _hypervolume_3d(
    rows: Sequence[Mapping[str, Any]],
    reference: tuple[float, float, float],
) -> float:
    points = list({_objective(row) for row in rows})
    eligible = [
        point
        for point in points
        if all(value < reference[index] for index, value in enumerate(point))
    ]
    frontier = [
        point
        for index, point in enumerate(eligible)
        if not any(
            other_index != index and _dominates(other, point)
            for other_index, other in enumerate(eligible)
        )
    ]
    if not frontier:
        return 0.0
    xs = sorted({point[0] for point in frontier} | {reference[0]})
    volume = 0.0
    for left, right in zip(xs, xs[1:]):
        active = [(y, z) for x, y, z in frontier if x <= left]
        volume += (right - left) * _hypervolume_2d(
            active,
            (reference[1], reference[2]),
        )
    return float(volume)


def _metric_context(
    root: Path,
) -> tuple[list[dict[str, Any]], tuple[float, float, float]]:
    frozen = _load_json(root / "contracts" / "frozen_inputs.json", "frozen_inputs")
    gold_contract = (frozen.get("inputs") or {}).get("gold176")
    if not isinstance(gold_contract, dict):
        _fail("frozen_gold_contract_missing")
    gold_path = Path(str(gold_contract.get("path") or "")).resolve()
    if not gold_path.is_file():
        _fail("frozen_gold_evidence_missing")
    _require_equal(
        _sha256(gold_path),
        gold_contract.get("sha256"),
        "frozen_gold_evidence_sha",
    )
    gold_payload = _load_json(gold_path, "frozen_gold_evidence")
    raw_rows = gold_payload.get("rows")
    if not isinstance(raw_rows, list):
        _fail("frozen_gold_rows_missing")
    initial_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        if (
            row.get("model") == "pyramid"
            and row.get("hardware_id") == "h800"
            and row.get("dispatch_key") == "tvm_auto"
            and row.get("terminal_status") in SUCCESS_STATUSES
        ):
            initial_rows.append(_audit_result_row(row))
    if not initial_rows:
        _fail("frozen_gold_task_scope_empty")
    reference_payload = _load_json(
        root / "contracts" / "raw_objective_reference.json",
        "raw_objective_reference",
    )
    values = reference_payload.get("values")
    if not isinstance(values, dict):
        _fail("raw_objective_reference_missing")
    reference = (
        _finite_number(values.get("latency_ms"), "reference_latency"),
        _finite_number(values.get("energy_j"), "reference_energy"),
        _finite_number(values.get("negative_ap70"), "reference_negative_ap"),
    )
    return initial_rows, reference


def _compute_trajectory_metrics(
    root: Path,
    trajectories: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_rows, reference = _metric_context(root)
    online_successes = [
        row
        for trajectory in trajectories
        for row in trajectory["events"]
        if row.get("terminal_status") in SUCCESS_STATUSES
    ]
    pooled_rows = [*initial_rows, *online_successes]
    pooled_frontier = _frontier_ids(pooled_rows)
    best_pooled_ap70 = max(float(row["ap70"]) for row in pooled_rows)
    iso_ap_floor = best_pooled_ap70 - 0.10
    baseline_hv = _hypervolume_3d(initial_rows, reference)
    computed: list[dict[str, Any]] = []
    for trajectory in trajectories:
        cumulative = list(initial_rows)
        delta_curve = [0.0]
        for round_index in range(ROUND_COUNT):
            cumulative = [
                *cumulative,
                *[
                    row
                    for row in trajectory["events"]
                    if row["round_index"] == round_index
                    and row.get("terminal_status") in SUCCESS_STATUSES
                ],
            ]
            delta_curve.append(
                max(0.0, _hypervolume_3d(cumulative, reference) - baseline_hv)
            )
        delta_hv_auc = sum(
            2.0 * (left + right)
            for left, right in zip(delta_curve, delta_curve[1:])
        )
        success_rows = [
            row
            for row in trajectory["events"]
            if row.get("terminal_status") in SUCCESS_STATUSES
        ]
        comparable = [
            row for row in success_rows if float(row["ap70"]) >= iso_ap_floor
        ]
        selected_success_ids = {
            str(row.get("row_id") or row.get("manifest_job_id"))
            for row in success_rows
        }
        computed.append(
            {
                **trajectory,
                "delta_hv_auc": delta_hv_auc,
                "delta_hv_at_16": delta_curve[-1],
                "delta_hv_curve": delta_curve,
                "frontier_recall": (
                    len(selected_success_ids & pooled_frontier)
                    / len(pooled_frontier)
                ),
                "best_iso_ap_latency_ms": (
                    min(float(row["latency_ms"]) for row in comparable)
                    if comparable
                    else None
                ),
                "best_iso_ap_energy_j": (
                    min(float(row["energy_j"]) for row in comparable)
                    if comparable
                    else None
                ),
                "iso_ap_comparability": (
                    "comparable"
                    if comparable
                    else "no_successful_point_within_0.10_ap70_of_pooled_best"
                ),
            }
        )
    return computed, {
        "hv_reference": {
            "latency_ms": reference[0],
            "energy_j": reference[1],
            "negative_ap70": reference[2],
        },
        "hv_auc_rule": "trapezoidal_integral_at_budgets_0_4_8_12_16",
        "pooled_frontier_row_ids": sorted(pooled_frontier),
        "frontier_recall_note": (
            "secondary post-hoc recall against the pooled measured attainable "
            "frontier from frozen Gold plus all contract-matched online successes"
        ),
        "iso_ap_rule": (
            "successful online points with AP70 no more than 0.10 below the "
            "pooled measured best; ratios are within-seed versus Full"
        ),
    }


def _audit_round(
    round_dir: Path,
    *,
    round_index: int,
    contract_sha: str,
    scheduler_windows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    request_path = round_dir / "measurement_request.json"
    request = _load_json(request_path, f"round_{round_index}_request")
    request_sha = _embedded_sha256(
        request,
        "measurement_request_sha256",
        "measurement_request",
    )
    _require_equal(request.get("round_index"), round_index, "request_round_index")
    if "trajectory_contract_sha256" in request:
        _require_equal(
            request.get("trajectory_contract_sha256"),
            contract_sha,
            "request_contract_sha",
        )
    selected_rows = request.get("rows")
    if not isinstance(selected_rows, list) or len(selected_rows) != BATCH_SIZE:
        _fail(f"selected_event_count_round_{round_index}")
    selected_ids = [
        str(row.get("row_id") or "") if isinstance(row, dict) else ""
        for row in selected_rows
    ]
    if any(not row_id for row_id in selected_ids) or len(set(selected_ids)) != BATCH_SIZE:
        _fail(f"selected_event_identity_round_{round_index}")

    binding_path = round_dir / "stage7_request_binding.json"
    binding = _load_json(binding_path, f"round_{round_index}_request_binding")
    binding_sha = _embedded_sha256(
        binding,
        "request_binding_sha256",
        "request_binding",
    )
    _require_equal(binding.get("round_index"), round_index, "binding_round_index")
    _require_equal(
        binding.get("measurement_request_sha256"),
        request_sha,
        "binding_request_sha",
    )
    _require_equal(
        binding.get("trajectory_contract_sha256"),
        contract_sha,
        "binding_contract_sha",
    )
    _require_equal(binding.get("selected_row_ids"), selected_ids, "binding_selected_ids")
    _require_equal(
        binding.get("selected_row_ids_sha256"),
        _payload_sha256(selected_ids),
        "binding_selected_ids_sha",
    )

    invariance = _load_json(
        round_dir / "cache_selection_invariance_audit.json",
        f"round_{round_index}_cache_selection_invariance_audit",
    )
    _require_equal(invariance.get("verdict"), "pass", "cache_selection_invariance")
    _require_equal(
        invariance.get("empty_cache_selected_ids"),
        selected_ids,
        "cache_empty_selection_ids",
    )
    _require_equal(
        invariance.get("populated_cache_selected_ids"),
        selected_ids,
        "cache_populated_selection_ids",
    )
    _require_equal(
        invariance.get("cache_labels_released_to_selector"),
        False,
        "cache_label_release",
    )

    cache_path = round_dir / "cache_reveal.json"
    cache_reveal = _load_json(cache_path, f"round_{round_index}_cache_reveal")
    cache_sha = _embedded_sha256(cache_reveal, "cache_reveal_sha256", "cache_reveal")
    _require_equal(
        cache_reveal.get("measurement_request_sha256"),
        request_sha,
        "cache_request_sha",
    )
    _require_equal(
        cache_reveal.get("request_binding_sha256"),
        binding_sha,
        "cache_binding_sha",
    )
    _require_equal(
        cache_reveal.get("trajectory_contract_sha256"),
        contract_sha,
        "cache_contract_sha",
    )
    cache_entries = cache_reveal.get("rows")
    if not isinstance(cache_entries, list) or len(cache_entries) != BATCH_SIZE:
        _fail("cache_reveal_event_matrix_drift")
    cache_by_id: dict[str, dict[str, Any]] = {}
    for entry in cache_entries:
        if not isinstance(entry, dict):
            _fail("cache_reveal_entry_drift")
        row_id = str(entry.get("row_id") or "")
        if row_id in cache_by_id or row_id not in selected_ids:
            _fail("cache_reveal_selected_ids_drift")
        if entry.get("disposition") not in {"exact_hit", "miss"}:
            _fail("cache_disposition_drift")
        if entry.get("disposition") == "exact_hit":
            if not _is_sha256(entry.get("source_cache_entry_sha256")):
                _fail("cache_hit_source_entry_sha_drift")
            if not _is_sha256(entry.get("source_evidence_sha256")):
                _fail("cache_hit_source_evidence_sha_drift")
        _require_equal(
            entry.get("revealed_after_request_sha256"),
            request_sha,
            "cache_entry_request_sha",
        )
        _require_equal(
            entry.get("trajectory_contract_sha256"),
            contract_sha,
            "cache_entry_contract_sha",
        )
        cache_by_id[row_id] = entry
    _require_equal(set(cache_by_id), set(selected_ids), "cache_reveal_selected_ids")

    miss_ids = {
        row_id
        for row_id, entry in cache_by_id.items()
        if entry["disposition"] == "miss"
    }
    scheduler_window = scheduler_windows.get(request_sha)
    if miss_ids and scheduler_window is None:
        _fail("gpu_lease_request_sha_missing")
    if not miss_ids and scheduler_window is not None:
        _fail("all_hit_batch_unexpected_gpu_lease")
    gpu_hours = (
        _finite_number(
            scheduler_window.get("gpu_hours"),
            "gpu_hours",
            minimum=0.0,
        )
        if scheduler_window is not None
        else 0.0
    )

    finalization_path = round_dir / "stage7_finalization.json"
    finalization = _load_json(
        finalization_path,
        f"round_{round_index}_stage7_finalization",
    )
    _embedded_sha256(
        finalization,
        "stage7_finalization_sha256",
        "stage7_finalization",
    )
    for field, expected in (
        ("measurement_request_sha256", request_sha),
        ("request_binding_sha256", binding_sha),
        ("cache_reveal_sha256", cache_sha),
    ):
        _require_equal(finalization.get(field), expected, f"feedback_{field}")
    _require_equal(finalization.get("feedback_released"), True, "feedback_release")
    _require_equal(finalization.get("batch_quarantined"), False, "feedback_quarantine")
    _require_equal(finalization.get("same_request_retry"), False, "feedback_retry")
    _require_equal(finalization.get("budget_consumed"), BATCH_SIZE, "feedback_budget")
    finalization_rows = finalization.get("released_feedback_rows")
    if not isinstance(finalization_rows, list) or len(finalization_rows) != BATCH_SIZE:
        _fail(f"selected_event_count_round_{round_index}")

    feedback_path = round_dir / "released_feedback.json"
    feedback = _load_json(feedback_path, f"round_{round_index}_released_feedback")
    feedback_sha = _embedded_sha256(
        feedback,
        "released_feedback_sha256",
        "released_feedback",
    )
    for field, expected in (
        ("round_index", round_index),
        ("measurement_request_sha256", request_sha),
        ("request_binding_sha256", binding_sha),
    ):
        _require_equal(feedback.get(field), expected, f"released_feedback_{field}")
    _require_equal(
        Path(str(feedback.get("feedback_input_path") or "")).resolve(),
        finalization_path.resolve(),
        "released_feedback_input_path",
    )
    _require_equal(
        feedback.get("feedback_input_sha256"),
        _sha256(finalization_path),
        "released_feedback_input_sha",
    )
    feedback_rows = feedback.get("rows")
    if not isinstance(feedback_rows, list) or len(feedback_rows) != BATCH_SIZE:
        _fail(f"selected_event_count_round_{round_index}")
    _require_equal(feedback_rows, finalization_rows, "released_feedback_rows")
    validated_rows = [_audit_result_row(row) for row in feedback_rows if isinstance(row, dict)]
    if len(validated_rows) != BATCH_SIZE:
        _fail(f"selected_event_count_round_{round_index}")
    feedback_ids = [str(row.get("row_id") or "") for row in validated_rows]
    _require_equal(feedback_ids, selected_ids, "feedback_selected_ids")
    enriched_rows: list[dict[str, Any]] = []
    for event_index, row in enumerate(validated_rows):
        row_id = selected_ids[event_index]
        _require_equal(
            row.get("cache_disposition"),
            cache_by_id[row_id]["disposition"],
            "feedback_cache_disposition",
        )
        enriched_rows.append(
            {
                **row,
                "round_index": round_index,
                "event_index": event_index,
                "request_sha256": request_sha,
                "binding_sha256": binding_sha,
                "cache_reveal_sha256": cache_sha,
                "feedback_sha256": feedback_sha,
            }
        )
    return enriched_rows, feedback_sha, {
        "gpu_hours": gpu_hours,
        "start_wall_time": (
            scheduler_window.get("start_wall_time")
            if scheduler_window is not None
            else None
        ),
        "end_wall_time": (
            scheduler_window.get("end_wall_time")
            if scheduler_window is not None
            else None
        ),
        "released_feedback_rows": validated_rows,
    }


def _audit_trajectory(
    root: Path,
    variant: str,
    seed: int,
    scanner: Mapping[str, Any],
    scheduler_windows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    trajectory_dir = _trajectory_dir(root, variant, seed)
    contract_path = trajectory_dir / "trajectory_contract.json"
    contract = _load_json(contract_path, "trajectory_contract")
    contract_sha = _embedded_sha256(
        contract,
        "trajectory_contract_sha256",
        "trajectory_contract",
    )
    for field, expected in (
        ("task_id", TASK_ID),
        ("variant", variant),
        ("seed", seed),
        ("trajectory_dir", str(trajectory_dir.resolve())),
        ("frozen_input_sha256", scanner["frozen_input_sha256"]),
        ("pre_scan_sha256", scanner["pre_scan_sha256"]),
        ("scan_pass_sha256", scanner["scan_pass_sha256"]),
        ("scanner_rule_sha256", scanner["scanner_rule_sha256"]),
        ("scanner_decision_sha256", scanner["scanner_decision_sha256"]),
    ):
        _require_equal(contract.get(field), expected, f"trajectory_{field}")
    expected_pool = scanner["variant_pool_sha256"][variant]
    _require_equal(contract.get("candidate_pool_sha256"), expected_pool, "candidate_pool_sha")
    a2_audit: dict[str, str] | None = None
    if variant == "without_measured_feedback":
        a2_audit = _audit_a2(trajectory_dir)
    if variant == "backend_blind":
        _audit_a3(trajectory_dir)

    events: list[dict[str, Any]] = []
    terminal_feedback_rows: list[dict[str, Any]] = []
    feedback_shas: list[str] = []
    feedback_sha: str | None = None
    gpu_hours = 0.0
    start_wall_times: list[float] = []
    end_wall_times: list[float] = []
    for round_index in range(ROUND_COUNT):
        rows, feedback_sha, costs = _audit_round(
            trajectory_dir / f"round_{round_index:02d}",
            round_index=round_index,
            contract_sha=contract_sha,
            scheduler_windows=scheduler_windows,
        )
        events.extend(rows)
        terminal_feedback_rows.extend(costs["released_feedback_rows"])
        feedback_shas.append(feedback_sha)
        gpu_hours += costs["gpu_hours"]
        if costs["start_wall_time"] is not None:
            start_wall_times.append(float(costs["start_wall_time"]))
            end_wall_times.append(float(costs["end_wall_time"]))
    selected_ids = [str(row["row_id"]) for row in events]
    if len(events) != EVENT_COUNT:
        _fail(f"selected_event_count:{len(events)}")
    if len(set(selected_ids)) != EVENT_COUNT:
        _fail("selected_events_not_unique")
    terminal = _load_json(
        trajectory_dir / "trajectory_terminal.json",
        "trajectory_terminal",
    )
    _embedded_sha256(
        terminal,
        "trajectory_terminal_sha256",
        "trajectory_terminal",
    )
    for field, expected in (
        ("status", "completed_at_T16"),
        ("variant", variant),
        ("seed", seed),
        ("task_id", TASK_ID),
        ("completed_atomic_rounds", ROUND_COUNT),
        ("selected_event_count", EVENT_COUNT),
        ("selected_row_ids", selected_ids),
        ("released_feedback_sha256", _payload_sha256(terminal_feedback_rows)),
        ("trajectory_contract_sha256", contract_sha),
    ):
        _require_equal(terminal.get(field), expected, f"trajectory_terminal_{field}")

    if not start_wall_times or not end_wall_times:
        _fail("trajectory_wall_clock_evidence_missing")
    successes = sum(row["terminal_status"] in SUCCESS_STATUSES for row in events)
    failures = EVENT_COUNT - successes
    q_mode_distribution = {
        q_mode: sum(str(row.get("q_mode")) == q_mode for row in events)
        for q_mode in sorted({str(row.get("q_mode")) for row in events})
    }
    cache_hits = sum(row["cache_disposition"] == "exact_hit" for row in events)
    terminal_note = (
        "capability_scan_non_discriminative_on_frozen_pool"
        if variant == "without_capability_scan" and scanner["all_candidates_pass"]
        else "discriminative"
    )
    return {
        "variant": variant,
        "seed": seed,
        "contract_sha256": contract_sha,
        "events": events,
        "selected_row_ids": selected_ids,
        "feedback_sha256s": feedback_shas,
        "invalid_rate": failures / EVENT_COUNT,
        "valid_yield": successes / EVENT_COUNT,
        "q_mode_distribution": q_mode_distribution,
        "exact_cache_hits": cache_hits,
        "exact_cache_misses": EVENT_COUNT - cache_hits,
        "gpu_hours": gpu_hours,
        "wall_clock_hours": (
            max(end_wall_times) - min(start_wall_times)
        )
        / 3600.0,
        "terminal_note": terminal_note,
        "a2_frozen_audit": a2_audit,
    }


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        _fail("descriptive_statistics_require_three_seeds")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    median = statistics.median(values)
    return {
        "mean": mean,
        "sample_std": sample_std,
        "median": median,
        "min": min(values),
        "max": max(values),
        "mean_sample_std": f"{mean:.6f} ± {sample_std:.6f}",
        "median_range": f"{median:.6f} [{min(values):.6f}, {max(values):.6f}]",
    }


def _format_statistic(summary: Mapping[str, Any]) -> str:
    if summary.get("status") == "not_comparable":
        return "not comparable"
    return f"{summary['mean_sample_std']}; {summary['median_range']}"


def _aggregate_trajectories(
    trajectories: Sequence[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    source_by_key = {(row["variant"], row["seed"]): row for row in trajectories}
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            source = source_by_key[(variant, seed)]
            full = source_by_key[("full", seed)]
            latency_comparable = (
                source["best_iso_ap_latency_ms"] is not None
                and full["best_iso_ap_latency_ms"] is not None
            )
            energy_comparable = (
                source["best_iso_ap_energy_j"] is not None
                and full["best_iso_ap_energy_j"] is not None
            )
            by_key = {
                **by_key,
                (variant, seed): {
                    **source,
                    "iso_ap_latency_ratio": (
                        source["best_iso_ap_latency_ms"]
                        / full["best_iso_ap_latency_ms"]
                        if latency_comparable
                        else None
                    ),
                    "iso_ap_energy_ratio": (
                        source["best_iso_ap_energy_j"]
                        / full["best_iso_ap_energy_j"]
                        if energy_comparable
                        else None
                    ),
                    "iso_ap_latency_ratio_reason": (
                        None
                        if latency_comparable
                        else source["iso_ap_comparability"]
                    ),
                    "iso_ap_energy_ratio_reason": (
                        None
                        if energy_comparable
                        else source["iso_ap_comparability"]
                    ),
                },
            }

    variants: dict[str, Any] = {}
    paper_rows: list[dict[str, str]] = []
    execution_rows: list[dict[str, str]] = []
    for variant in VARIANTS:
        rows = [by_key[(variant, seed)] for seed in SEEDS]
        metric_summaries: dict[str, Any] = {}
        for metric in METRICS:
            seed_values = {
                str(seed): (
                    float(by_key[(variant, seed)][metric])
                    if by_key[(variant, seed)][metric] is not None
                    else None
                )
                for seed in SEEDS
            }
            if any(value is None for value in seed_values.values()):
                metric_summaries[metric] = {
                    "status": "not_comparable",
                    "seeds": seed_values,
                    "reason_by_seed": {
                        str(seed): by_key[(variant, seed)].get(f"{metric}_reason")
                        for seed in SEEDS
                        if by_key[(variant, seed)][metric] is None
                    },
                    "paired_delta_vs_full": {
                        "status": "not_comparable",
                        "seed_deltas": {str(seed): None for seed in SEEDS},
                    },
                }
                continue
            summary = _describe(list(seed_values.values()))
            paired_values = [
                float(by_key[(variant, seed)][metric])
                - float(by_key[("full", seed)][metric])
                for seed in SEEDS
            ]
            if variant == "full":
                paired = {"status": "reference", "seed_deltas": {str(seed): 0.0 for seed in SEEDS}}
            else:
                improved = sum(
                    delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
                    for delta in paired_values
                )
                equal = sum(delta == 0 for delta in paired_values)
                paired = {
                    "seed_deltas": {
                        str(seed): paired_values[index] for index, seed in enumerate(SEEDS)
                    },
                    "mean": statistics.fmean(paired_values),
                    "median": statistics.median(paired_values),
                    "direction_improved_count": improved,
                    "direction_equal_count": equal,
                    "direction_total": len(SEEDS),
                }
            metric_summaries[metric] = {
                "seeds": seed_values,
                **summary,
                "paired_delta_vs_full": paired,
            }
        variants[variant] = {
            "completed_trajectories": len(rows),
            "metrics": metric_summaries,
            "precision_distribution_by_seed": {
                str(row["seed"]): row["q_mode_distribution"] for row in rows
            },
            "terminal_note": rows[0]["terminal_note"],
        }
        hv_paired = metric_summaries["delta_hv_auc"]["paired_delta_vs_full"]
        if variant == "full":
            paired_direction = "reference"
        else:
            paired_direction = (
                f"{hv_paired['mean']:.6f}; "
                f"{hv_paired['direction_improved_count']}/3 seeds improved"
            )
            if hv_paired["direction_improved_count"] == 0:
                paired_direction = f"{hv_paired['mean']:.6f}; Full better in 3/3 seeds"
        paper_rows.append(
            {
                "Variant": variant,
                "DeltaHV-AUC ↑": _format_statistic(metric_summaries["delta_hv_auc"]),
                "DeltaHV@16 ↑": _format_statistic(metric_summaries["delta_hv_at_16"]),
                "Frontier recall† ↑": _format_statistic(metric_summaries["frontier_recall"]),
                "Invalid rate ↓": _format_statistic(metric_summaries["invalid_rate"]),
                "Valid yield ↑": _format_statistic(metric_summaries["valid_yield"]),
                "Iso-AP latency ratio ↓": _format_statistic(
                    metric_summaries["iso_ap_latency_ratio"]
                ),
                "Iso-AP energy ratio ↓": _format_statistic(
                    metric_summaries["iso_ap_energy_ratio"]
                ),
                "Paired Δ / direction": paired_direction,
                "Status": "complete",
            }
        )
        execution_rows.append(
            {
                "Variant": variant,
                "Exact cache hit / 16": _format_statistic(metric_summaries["exact_cache_hits"]),
                "Exact cache miss / 16": _format_statistic(metric_summaries["exact_cache_misses"]),
                "GPU-hours ↓": _format_statistic(metric_summaries["gpu_hours"]),
                "Wall-clock ↓": _format_statistic(metric_summaries["wall_clock_hours"]),
                "Completed trajectories": "3/3",
                "Terminal note": rows[0]["terminal_note"],
            }
        )
    full_selected_sequences = {
        tuple(str(row_id) for row_id in by_key[("full", seed)]["selected_row_ids"])
        for seed in SEEDS
    }
    limited_independence = len(full_selected_sequences) == 1
    normalized_trajectories = [
        by_key[(variant, seed)] for variant in VARIANTS for seed in SEEDS
    ]
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "variants": variants,
            "limited_effective_independence": limited_independence,
            "limited_effective_independence_note": (
                "deterministic Full trajectories are identical; effective independence is limited"
                if limited_independence
                else None
            ),
        },
        paper_rows,
        execution_rows,
        normalized_trajectories,
    )


def _incomplete_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    paper = [
        {
            "Variant": variant,
            "DeltaHV-AUC ↑": "incomplete",
            "DeltaHV@16 ↑": "incomplete",
            "Frontier recall† ↑": "incomplete",
            "Invalid rate ↓": "incomplete",
            "Valid yield ↑": "incomplete",
            "Iso-AP latency ratio ↓": "incomplete",
            "Iso-AP energy ratio ↓": "incomplete",
            "Paired Δ / direction": "incomplete",
            "Status": "incomplete",
        }
        for variant in VARIANTS
    ]
    execution = [
        {
            "Variant": variant,
            "Exact cache hit / 16": "incomplete",
            "Exact cache miss / 16": "incomplete",
            "GPU-hours ↓": "incomplete",
            "Wall-clock ↓": "incomplete",
            "Completed trajectories": "incomplete",
            "Terminal note": "incomplete",
        }
        for variant in VARIANTS
    ]
    return paper, execution


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    json.dumps(row.get(field), sort_keys=True, separators=(",", ":"))
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field, "")
                )
                for field in fieldnames
            }
        )
    return stream.getvalue()


def _markdown_table(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(str(row.get(field, "")).replace("|", "/") for field in fieldnames) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body]) + "\n"


def _emit_outputs(
    output_root: Path,
    *,
    events: Sequence[dict[str, Any]],
    trajectories: Sequence[dict[str, Any]],
    aggregate: Mapping[str, Any],
    completeness: Mapping[str, Any],
    isolation: Mapping[str, Any],
    paper_rows: Sequence[dict[str, str]],
    execution_rows: Sequence[dict[str, str]],
) -> None:
    event_fields = (
        "variant",
        "seed",
        "round_index",
        "event_index",
        "row_id",
        "q_mode",
        "terminal_status",
        "cache_disposition",
        "latency_ms",
        "energy_j",
        "ap70",
        "metric_source",
        "request_sha256",
        "binding_sha256",
        "cache_reveal_sha256",
        "feedback_sha256",
    )
    trajectory_fields = (
        "variant",
        "seed",
        "delta_hv_auc",
        "delta_hv_at_16",
        "frontier_recall",
        "invalid_rate",
        "valid_yield",
        "q_mode_distribution",
        "iso_ap_latency_ratio",
        "iso_ap_energy_ratio",
        "exact_cache_hits",
        "exact_cache_misses",
        "gpu_hours",
        "wall_clock_hours",
        "terminal_note",
    )
    paper_fields = tuple(paper_rows[0])
    execution_fields = tuple(execution_rows[0])
    _atomic_write(
        output_root / OUTPUT_PATHS[0],
        _csv_text(events, event_fields),
    )
    _atomic_write(
        output_root / OUTPUT_PATHS[1],
        _csv_text(trajectories, trajectory_fields),
    )
    aggregate_payload = {
        **aggregate,
        "paper_ready": completeness["paper_ready"],
        "paper_rows": list(paper_rows),
        "execution_cost_rows": list(execution_rows),
    }
    _write_json(output_root / OUTPUT_PATHS[2], aggregate_payload)
    _write_json(output_root / OUTPUT_PATHS[3], completeness)
    _write_json(output_root / OUTPUT_PATHS[4], isolation)
    root_causes = completeness["missing_matrix"]
    summary_lines = [
        "# Stage7 root-cause summary",
        "",
        f"- paper_ready: `{str(completeness['paper_ready']).lower()}`",
        f"- complete trajectories: `{completeness['complete_trajectory_count']}/15`",
        "",
        "## Missing or invalid matrix",
        "",
    ]
    if root_causes:
        summary_lines.extend(
            f"- `{item.get('variant', 'global')}` / `{item.get('seed', 'all')}`: "
            f"`{item['reason']}`"
            for item in root_causes
        )
    else:
        summary_lines.append("- none")
    if aggregate.get("limited_effective_independence_note"):
        summary_lines.extend(
            ["", "## Descriptive caution", "", f"- {aggregate['limited_effective_independence_note']}"]
        )
    _atomic_write(output_root / OUTPUT_PATHS[5], "\n".join(summary_lines) + "\n")
    _atomic_write(output_root / OUTPUT_PATHS[6], _csv_text(paper_rows, paper_fields))
    _atomic_write(output_root / OUTPUT_PATHS[7], _markdown_table(paper_rows, paper_fields))
    _atomic_write(output_root / OUTPUT_PATHS[8], _csv_text(execution_rows, execution_fields))
    _atomic_write(output_root / OUTPUT_PATHS[9], _markdown_table(execution_rows, execution_fields))


def finalize(input_root: Path | str, output_root: Path | str) -> dict[str, Any]:
    """Audit a caller-provided Stage7 tree and emit the ten frozen outputs."""
    source_root = Path(input_root).resolve()
    destination_root = Path(output_root).resolve()
    if source_root == destination_root:
        raise ValueError("input_root and output_root must differ")
    _validate_output_root(destination_root)
    scanner: dict[str, Any] | None = None
    isolation_source: dict[str, Any] | None = None
    scheduler_windows: dict[str, dict[str, Any]] = {}
    missing_matrix: list[dict[str, Any]] = []
    try:
        scanner, isolation_source = _audit_global_contracts(source_root)
    except FinalizationError as error:
        missing_matrix.append({"variant": "global", "seed": None, "reason": str(error)})
    try:
        scheduler_windows = _load_scheduler_windows(source_root)
    except FinalizationError as error:
        missing_matrix.append({"variant": "global", "seed": None, "reason": str(error)})

    trajectories: list[dict[str, Any]] = []
    if scanner is not None and not missing_matrix:
        for variant in VARIANTS:
            for seed in SEEDS:
                trajectory_dir = _trajectory_dir(source_root, variant, seed)
                if not trajectory_dir.is_dir():
                    missing_matrix.append(
                        {"variant": variant, "seed": seed, "reason": "missing_trajectory"}
                    )
                    continue
                try:
                    trajectories.append(
                        _audit_trajectory(
                            source_root,
                            variant,
                            seed,
                            scanner,
                            scheduler_windows,
                        )
                    )
                except (FinalizationError, OSError) as error:
                    missing_matrix.append(
                        {"variant": variant, "seed": seed, "reason": str(error)}
                    )
    if not missing_matrix and len(trajectories) == len(VARIANTS) * len(SEEDS):
        used_scheduler_requests = {
            str(event["request_sha256"])
            for trajectory in trajectories
            for event in trajectory["events"]
            if event["cache_disposition"] == "miss"
        }
        if set(scheduler_windows) != used_scheduler_requests:
            missing_matrix.append(
                {
                    "variant": "global",
                    "seed": None,
                    "reason": "unmatched_gpu_lease_request_sha",
                }
            )
    metric_contract: dict[str, Any] | None = None
    if not missing_matrix and len(trajectories) == len(VARIANTS) * len(SEEDS):
        try:
            trajectories, metric_contract = _compute_trajectory_metrics(
                source_root,
                trajectories,
            )
        except FinalizationError as error:
            missing_matrix.append(
                {"variant": "global", "seed": None, "reason": str(error)}
            )
    paper_ready = not missing_matrix and len(trajectories) == len(VARIANTS) * len(SEEDS)
    if paper_ready:
        aggregate, paper_rows, execution_rows, output_trajectories = (
            _aggregate_trajectories(trajectories)
        )
        aggregate = {**aggregate, "metric_contract": metric_contract}
    else:
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "variants": {},
            "limited_effective_independence": False,
            "limited_effective_independence_note": None,
            "metric_contract": metric_contract,
        }
        paper_rows, execution_rows = _incomplete_rows()
        output_trajectories = trajectories

    flattened_events: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for event in trajectory["events"]:
            flattened_events.append(
                {
                    "variant": trajectory["variant"],
                    "seed": trajectory["seed"],
                    **event,
                }
            )
    completeness = {
        "schema_version": SCHEMA_VERSION,
        "paper_ready": paper_ready,
        "expected_trajectory_count": len(VARIANTS) * len(SEEDS),
        "complete_trajectory_count": len(trajectories),
        "expected_event_count_per_trajectory": EVENT_COUNT,
        "missing_matrix": missing_matrix,
    }
    isolation = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "pass" if isolation_source is not None and not missing_matrix else "fail",
        "source_audit": isolation_source,
        "scanner_admission": scanner,
    }
    _emit_outputs(
        destination_root,
        events=flattened_events,
        trajectories=output_trajectories,
        aggregate=aggregate,
        completeness=completeness,
        isolation=isolation,
        paper_rows=paper_rows,
        execution_rows=execution_rows,
    )
    return {
        "paper_ready": paper_ready,
        "missing_matrix": missing_matrix,
        "aggregate": aggregate,
        "paper_rows": paper_rows,
        "execution_cost_rows": execution_rows,
        "output_root": str(destination_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = finalize(args.input_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["paper_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
