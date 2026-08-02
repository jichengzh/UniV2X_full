"""Thin Stage7 finalization around frozen Stage3 and actual-v3 primitives.

This module classifies authenticated physical outcomes before Stage3 row
finalization, preserves the independent-projection lineage, and builds cache
wrappers only from feedback rows that have already been promoted to actual
graph features.  It does not execute hardware, build TVM programs, or extract
AP/latency/energy evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage5 import single_target_search_v2 as stage5_search
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import core_cache_v2 as cache_v2
from scripts import stage3_finalize_gold96_v3 as stage3_finalizer


JSON = dict[str, Any]
STRUCTURED_FAILURE_SCHEMA = "stage7_structured_physical_failure_v2"
PHYSICAL_TERMINAL_SCHEMA = "stage7_actual_v3_physical_terminal_batch_v2"
EMPTY_TERMINAL_SCHEMA = "stage7_actual_v3_empty_physical_terminal_v2"
PERFORMANCE_ARTIFACT_SCHEMA = "stage7_authenticated_performance_artifacts_v2"
FROZEN_STAGE3_FINALIZER_SHA256 = (
    "21ea511c2f27874c612b85b41e028c63123490433f1d88dd4498f653e04cc238"
)
_FROZEN_STAGE3_FINALIZER_CALLABLE = stage3_finalizer._finalize_row

_CANDIDATE_REASONS = frozenset(
    """backend_capability_failure backend_failure build_failure
    unsupported_precision quantization_failure numerical_failure
    candidate_runtime_capability_failure""".split()
)
_INFRASTRUCTURE_REASONS = frozenset(
    """gpu_occupancy_drift gpu_unavailable unrelated_process_oom contention
    ssh_failure network_failure missing_source_artifact permission_failure
    runner_bug""".split()
)
_EVIDENCE_REASONS = frozenset(
    """terminal_artifact_sha_mismatch terminal_evidence_sha_mismatch
    evidence_schema_mismatch evidence_missing""".split()
)
_REASONS = {
    "candidate": _CANDIDATE_REASONS,
    "infrastructure": _INFRASTRUCTURE_REASONS,
    "evidence": _EVIDENCE_REASONS,
}
_REPORT_FIELDS = frozenset(
    """schema_version candidate_id failure_class failure_reason
    evidence_authenticated primitive_terminal_status primitive_evidence
    structured_failure_report_sha256""".split()
)
_TERMINAL_FIELDS = frozenset(
    """schema_version rows lineage_inputs failures projection_artifact
    performance_artifacts_sha256 execution_attempt barrier_release_allowed
    retry_logical_request_sha256 physical_terminal_batch_sha256""".split()
)
_PROJECTION_ARTIFACT_FIELDS = frozenset(
    """artifact_kind measurement_request_sha256 projection_payload_sha256
    deployment_bundle_sha256""".split()
)
_EMPTY_PROJECTION_LINEAGE_FIELDS = frozenset(
    """logical_request_sha256 selection_binding_sha256 cache_snapshot_sha256
    cache_reveal_sha256 physical_request_sha256 source_resolution_plan_sha256
    source_resolution_result_sha256 resolved_source_lock_sha256
    executor_admission_sha256 deployment_bundle_sha256 rows""".split()
)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _row_id(row: Mapping[str, Any]) -> str:
    candidate_id = str(row.get("manifest_job_id") or row.get("row_id") or "")
    if not candidate_id:
        raise ValueError("physical feedback row identity is missing")
    return candidate_id


def _default_validate_projection(value: Mapping[str, Any]) -> JSON:
    # Local import is deliberate: physical_execution imports scheduler request
    # identities whose source scheduler imports the Stage7 barrier.
    from framework.stage7 import physical_execution_v2

    return physical_execution_v2._validate_projection(value)


def _validate_stage3_finalizer() -> Callable[..., JSON]:
    source_path = Path(str(stage3_finalizer.__file__ or ""))
    if (
        not source_path.is_file()
        or hashlib.sha256(source_path.read_bytes()).hexdigest()
        != FROZEN_STAGE3_FINALIZER_SHA256
        or not callable(getattr(stage3_finalizer, "_finalize_row", None))
        or stage3_finalizer._finalize_row is not _FROZEN_STAGE3_FINALIZER_CALLABLE
    ):
        raise ValueError("Stage3 finalizer deployment primitive SHA drift")
    return stage3_finalizer._finalize_row


def _terminal_state(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    terminal = [
        row for row in rows if row.get("status") in {"success", "confirmed_failure"}
    ]
    return terminal[-1] if terminal else None


def _validate_structured_report(
    report: Mapping[str, Any], *, candidate_id: str
) -> JSON:
    if not isinstance(report, Mapping):
        raise ValueError("structured failure report must be a mapping")
    copied = copy.deepcopy(dict(report))
    recorded = copied.pop("structured_failure_report_sha256", None)
    failure_class = str(copied.get("failure_class") or "")
    reason = str(copied.get("failure_reason") or "")
    primitive_status = str(copied.get("primitive_terminal_status") or "")
    evidence_ref = copied.get("primitive_evidence")
    if (
        set(report) != _REPORT_FIELDS
        or copied.get("schema_version") != STRUCTURED_FAILURE_SCHEMA
        or copied.get("candidate_id") != candidate_id
        or copied.get("evidence_authenticated") is not True
        or failure_class not in _REASONS
        or reason not in _REASONS[failure_class]
        or not primitive_status
        or not isinstance(evidence_ref, Mapping)
        or not _is_sha256(recorded)
        or recorded != _sha(copied)
    ):
        raise ValueError("structured failure report authentication failed")
    evidence_path = Path(str(evidence_ref.get("path") or ""))
    evidence_sha = evidence_ref.get("artifact_sha256")
    if (
        set(evidence_ref) != {"artifact_kind", "path", "artifact_sha256"}
        or evidence_ref.get("artifact_kind") != "stage7_structured_primitive_terminal"
        or not evidence_path.is_file()
        or not _is_sha256(evidence_sha)
        or hashlib.sha256(evidence_path.read_bytes()).hexdigest() != evidence_sha
    ):
        raise ValueError("structured primitive evidence authentication failed")
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("structured primitive evidence is invalid") from error
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("candidate_id") != candidate_id
        or evidence.get("failure_class") != failure_class
        or evidence.get("failure_reason") != reason
        or evidence.get("status") != primitive_status
    ):
        raise ValueError("structured primitive evidence semantic drift")
    if failure_class == "candidate":
        allowed_terminal = {
            "confirmed_failure",
            "numerical_feasibility_failure",
        }
        if primitive_status not in allowed_terminal:
            raise ValueError("candidate failure lacks primitive terminal proof")
        if (
            reason == "numerical_failure"
            and primitive_status != "numerical_feasibility_failure"
        ):
            raise ValueError("numerical failure lacks numerical terminal proof")
    return {**copied, "structured_failure_report_sha256": recorded}


def classify_structured_failure(
    *,
    candidate_id: str,
    performance_rows: Sequence[Mapping[str, Any]],
    ap_rows: Sequence[Mapping[str, Any]],
    structured_report: Mapping[str, Any] | None,
) -> JSON:
    """Classify before ``_finalize_row``; raw return codes never prove a candidate."""
    del ap_rows  # Stage3 owns AP semantics; only authenticated reports classify failure.
    if structured_report is not None:
        report = _validate_structured_report(
            structured_report, candidate_id=candidate_id
        )
        failure_class = str(report["failure_class"])
        return {
            "candidate_id": candidate_id,
            "failure_class": failure_class,
            "failure_reason": report["failure_reason"],
            "consumes_selected_event_budget": failure_class == "candidate",
            "structured_failure_report_sha256": report[
                "structured_failure_report_sha256"
            ],
        }
    performance = _terminal_state(performance_rows)
    if performance is not None and performance.get("status") == "success":
        return {
            "candidate_id": candidate_id,
            "failure_class": "success",
            "failure_reason": None,
            "consumes_selected_event_budget": True,
            "structured_failure_report_sha256": None,
        }
    if performance is not None and performance.get("status") == "confirmed_failure":
        return {
            "candidate_id": candidate_id,
            "failure_class": "candidate",
            "failure_reason": "backend_failure",
            "consumes_selected_event_budget": True,
            "structured_failure_report_sha256": None,
        }
    return {
        "candidate_id": candidate_id,
        "failure_class": "infrastructure",
        "failure_reason": "runner_bug",
        "consumes_selected_event_budget": False,
        "structured_failure_report_sha256": None,
    }


def validate_performance_projection_lineage(
    projection: Mapping[str, Any],
    performance_artifacts: Mapping[str, Any],
    *,
    validate_projection: Callable[[Mapping[str, Any]], JSON] | None = None,
) -> tuple[JSON, JSON]:
    """Authenticate the planner output without rewriting its source-request SHA."""
    projection_validator = validate_projection or _default_validate_projection
    verified_projection = projection_validator(projection)
    if not isinstance(performance_artifacts, Mapping):
        raise ValueError("authenticated performance artifacts are missing")
    artifacts = copy.deepcopy(dict(performance_artifacts))
    recorded = artifacts.pop("performance_artifacts_sha256", None)
    manifest = artifacts.get("manifest")
    projected_ids = [_row_id(row) for row in verified_projection.get("rows") or ()]
    if (
        artifacts.get("schema_version") != PERFORMANCE_ARTIFACT_SCHEMA
        or not _is_sha256(recorded)
        or recorded != _sha(artifacts)
        or artifacts.get("independent_request_sha256")
        != verified_projection["measurement_request_sha256"]
        or artifacts.get("planned_candidate_ids") != projected_ids
        or artifacts.get("manifest_row_count") != len(projected_ids)
        or not isinstance(manifest, Mapping)
        or manifest.get("source_request_sha256")
        != verified_projection["measurement_request_sha256"]
    ):
        raise ValueError("performance manifest projection SHA drift")
    jobs = manifest.get("jobs")
    if (
        not isinstance(jobs, list)
        or len(jobs) != len(projected_ids)
        or [_row_id(row) for row in jobs] != projected_ids
    ):
        raise ValueError("performance manifest row identity drift")
    return verified_projection, {
        **artifacts,
        "performance_artifacts_sha256": recorded,
    }


def _rows_by_id(
    rows: Sequence[Mapping[str, Any]], *, name: str
) -> dict[str, list[JSON]]:
    result: dict[str, list[JSON]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} row must be a mapping")
        row = copy.deepcopy(dict(raw))
        candidate_id = str(row.get("manifest_job_id") or row.get("job_id") or "")
        if not candidate_id:
            raise ValueError(f"{name} row identity is missing")
        result.setdefault(candidate_id, []).append(row)
    return result


def _unique_by_id(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, JSON]:
    result: dict[str, JSON] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{name} row must be a mapping")
        row = copy.deepcopy(dict(raw))
        candidate_id = str(row.get("candidate_id") or _row_id(row))
        if candidate_id in result:
            raise ValueError(f"duplicate {name} row: {candidate_id}")
        result[candidate_id] = row
    return result


def _candidate_failure_performance(
    reason: str, report: Mapping[str, Any]
) -> list[JSON]:
    return [
        {
            "status": "confirmed_failure",
            "failure_reason": reason,
            "structured_failure_report_sha256": report[
                "structured_failure_report_sha256"
            ],
        }
    ]


def _bind_historical_row(
    requested: Mapping[str, Any],
    source: Mapping[str, Any],
    finalized: Mapping[str, Any],
    *,
    logical_row_sha256: str,
) -> JSON:
    candidate_id = _row_id(requested)
    row = {
        **copy.deepcopy(dict(requested)),
        **copy.deepcopy(dict(finalized)),
        "row_id": candidate_id,
        "manifest_job_id": candidate_id,
        "training_source": "online_feedback",
        "measurement_request_row_sha256": logical_row_sha256,
        "materialized_source_evidence_path": str(
            source.get("source_evidence_path") or ""
        ),
        "materialized_source_evidence_sha256": str(
            source.get("source_evidence_sha256") or ""
        ),
    }
    if not row["materialized_source_evidence_path"] or not _is_sha256(
        row["materialized_source_evidence_sha256"]
    ):
        raise ValueError("materialized source evidence binding is missing")
    return row


def _validate_execution_attempt(
    execution_attempt: Mapping[str, Any], *, deployment_bundle_sha256: str
) -> JSON:
    if not isinstance(execution_attempt, Mapping):
        raise ValueError("execution attempt must be a mapping")
    copied = copy.deepcopy(dict(execution_attempt))
    recorded = copied.pop("execution_attempt_sha256", None)
    primitives = copied.get("primitive_sha256")
    zero_miss = copied.get("zero_miss") is True
    if (
        not str(copied.get("attempt_id") or "")
        or copied.get("deployment_bundle_sha256") != deployment_bundle_sha256
        or not isinstance(primitives, Mapping)
        or (not primitives and not zero_miss)
        or any(
            not str(name) or not _is_sha256(digest)
            for name, digest in primitives.items()
        )
        or (
            zero_miss
            and (
                copied.get("gpu_subprocess_count") != 0
                or not _is_sha256(copied.get("empty_physical_terminal_sha256"))
            )
        )
        or not _is_sha256(recorded)
        or recorded != _sha(copied)
    ):
        raise ValueError("execution attempt lineage authentication failed")
    return {**copied, "execution_attempt_sha256": recorded}


def finalize_physical_rows(
    *,
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    projection: Mapping[str, Any],
    performance_artifacts: Mapping[str, Any],
    performance_state_rows: Sequence[Mapping[str, Any]],
    ap_plan_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
    structured_failure_reports: Sequence[Mapping[str, Any]],
    lineage_inputs: Sequence[Mapping[str, Any]],
    execution_attempt: Mapping[str, Any],
    finalize_row: Callable[..., JSON] | None = None,
    validate_projection: Callable[[Mapping[str, Any]], JSON] | None = None,
    validate_physical_plan: Callable[..., tuple[JSON, JSON]] = (
        actual_adapter._validate_physical_plan
    ),
) -> JSON:
    """Finalize only success/proven-candidate rows and quarantine retryable rows."""
    row_finalizer = finalize_row or _validate_stage3_finalizer()
    verified_projection, artifacts = validate_performance_projection_lineage(
        projection,
        performance_artifacts,
        validate_projection=validate_projection,
    )
    verified_selection, verified_plan = validate_physical_plan(
        logical_request, selection_binding, physical_plan
    )
    projected_rows = [copy.deepcopy(dict(row)) for row in verified_projection["rows"]]
    projected_ids = [_row_id(row) for row in projected_rows]
    plan_misses = [
        str(row["candidate_id"])
        for row in verified_plan.get("logical_row_bindings") or ()
        if row.get("disposition") == "miss"
    ]
    if projected_ids != plan_misses:
        raise ValueError("projection differs from physical miss order")
    logical_by_id = {
        _row_id(row): copy.deepcopy(dict(row))
        for row in logical_request.get("rows") or ()
    }
    manifest_by_id = {
        _row_id(row): copy.deepcopy(dict(row)) for row in artifacts["manifest"]["jobs"]
    }
    performance_by_id = _rows_by_id(performance_state_rows, name="performance state")
    ap_by_id = _rows_by_id(ap_state_rows, name="AP state")
    ap_plan = _unique_by_id(ap_plan_rows, name="AP plan")
    reports = _unique_by_id(
        structured_failure_reports, name="structured failure report"
    )
    lineages = _unique_by_id(lineage_inputs, name="raw lineage")
    if set(reports) - set(projected_ids) or set(lineages) - set(projected_ids):
        raise ValueError("physical evidence contains an unprojected candidate")

    finalized_rows: list[JSON] = []
    failures: list[JSON] = []
    successful_ids: list[str] = []
    retryable = False
    for candidate_id in projected_ids:
        plan_row = ap_plan.get(candidate_id)
        if plan_row is None:
            raise ValueError(f"AP plan row is missing: {candidate_id}")
        performance_id = str(plan_row.get("performance_job_id") or candidate_id)
        candidate_performance = performance_by_id.get(performance_id, [])
        candidate_ap = ap_by_id.get(candidate_id, [])
        classification = classify_structured_failure(
            candidate_id=candidate_id,
            performance_rows=candidate_performance,
            ap_rows=candidate_ap,
            structured_report=reports.get(candidate_id),
        )
        failure_class = classification["failure_class"]
        if failure_class in {"infrastructure", "evidence"}:
            retryable = True
            failures.append(
                actual_adapter.finalize_actual_v3_failure(
                    logical_request,
                    selection_binding,
                    physical_plan,
                    candidate_id=candidate_id,
                    failure_class=failure_class,
                    reason=str(classification["failure_reason"]),
                )
            )
            continue

        if failure_class == "candidate":
            report = reports.get(candidate_id)
            terminal_performance = _terminal_state(candidate_performance)
            if report is not None:
                finalizer_performance = _candidate_failure_performance(
                    str(classification["failure_reason"]),
                    _validate_structured_report(report, candidate_id=candidate_id),
                )
            elif (
                terminal_performance is not None
                and terminal_performance.get("status") == "confirmed_failure"
            ):
                finalizer_performance = [terminal_performance]
            else:
                raise ValueError("candidate failure lacks terminal performance proof")
        else:
            finalizer_performance = candidate_performance
        finalized = row_finalizer(
            manifest_by_id[candidate_id],
            finalizer_performance,
            candidate_ap,
            output_schema="stage5_feedback_row_v2",
        )
        terminal_status = str(finalized.get("terminal_status") or "")
        if failure_class == "candidate":
            if terminal_status not in stage5_search.TRUE_FAILURE_STATUSES:
                raise ValueError("candidate failure did not produce a Stage5 terminal")
            finalized = {
                **copy.deepcopy(dict(finalized)),
                **{metric: None for metric in cache_v2.METRIC_FIELDS},
            }
            failures.append(
                actual_adapter.finalize_actual_v3_failure(
                    logical_request,
                    selection_binding,
                    physical_plan,
                    candidate_id=candidate_id,
                    failure_class="candidate",
                    reason=str(classification["failure_reason"]),
                )
            )
        elif terminal_status != stage5_search.SUCCESS_STATUS:
            raise ValueError("successful physical row did not reach Stage5 terminal")
        else:
            successful_ids.append(candidate_id)
        finalized_rows.append(
            _bind_historical_row(
                logical_by_id[candidate_id],
                manifest_by_id[candidate_id],
                finalized,
                logical_row_sha256=logical_request["row_sha256"][candidate_id],
            )
        )

    if set(lineages) != set(successful_ids):
        raise ValueError("raw lineage inputs must equal successful physical misses")
    ordered_lineage = [lineages[candidate_id] for candidate_id in successful_ids]
    projection_lineage = verified_projection["stage7_projection_lineage"]
    verified_attempt = _validate_execution_attempt(
        execution_attempt,
        deployment_bundle_sha256=str(
            projection_lineage.get("deployment_bundle_sha256") or ""
        ),
    )
    projection_artifact = {
        "artifact_kind": "stage5_independent_projection",
        "measurement_request_sha256": verified_projection["measurement_request_sha256"],
        "projection_payload_sha256": _sha(verified_projection),
        "deployment_bundle_sha256": projection_lineage["deployment_bundle_sha256"],
    }
    payload = {
        "schema_version": PHYSICAL_TERMINAL_SCHEMA,
        "rows": finalized_rows,
        "lineage_inputs": ordered_lineage,
        "failures": failures,
        "projection_artifact": projection_artifact,
        "performance_artifacts_sha256": artifacts["performance_artifacts_sha256"],
        "execution_attempt": verified_attempt,
        "barrier_release_allowed": not retryable,
        "retry_logical_request_sha256": (
            logical_request["measurement_request_sha256"] if retryable else None
        ),
    }
    terminal = {**payload, "physical_terminal_batch_sha256": _sha(payload)}
    return validate_physical_terminal_batch(terminal)


def _normalize_empty_physical_terminal(payload: Mapping[str, Any]) -> JSON:
    from framework.stage7 import physical_execution_v2

    performance_artifacts = physical_execution_v2._empty_performance_artifacts(payload)
    lineage = payload.get("stage7_projection_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("empty physical terminal lineage is missing")
    preserved_lineage = copy.deepcopy(dict(lineage))
    logical_request_sha256 = lineage.get("logical_request_sha256")
    deployment_bundle_sha256 = lineage.get("deployment_bundle_sha256")
    if (
        set(lineage) != _EMPTY_PROJECTION_LINEAGE_FIELDS
        or any(not _is_sha256(value) for key, value in lineage.items() if key != "rows")
        or lineage.get("rows") != []
    ):
        raise ValueError("empty physical terminal lineage authentication failed")
    empty_terminal_sha256 = payload.get("empty_physical_terminal_sha256")
    attempt_payload = {
        "attempt_id": "zero_miss",
        "deployment_bundle_sha256": deployment_bundle_sha256,
        "primitive_sha256": {},
        "zero_miss": True,
        "gpu_subprocess_count": 0,
        "empty_physical_terminal_sha256": empty_terminal_sha256,
    }
    attempt = {
        **attempt_payload,
        "execution_attempt_sha256": _sha(attempt_payload),
    }
    normalized_payload = {
        "schema_version": PHYSICAL_TERMINAL_SCHEMA,
        "rows": [],
        "lineage_inputs": [],
        "failures": [],
        "projection_artifact": {
            "artifact_kind": "stage5_independent_projection",
            "measurement_request_sha256": logical_request_sha256,
            "projection_payload_sha256": empty_terminal_sha256,
            "deployment_bundle_sha256": deployment_bundle_sha256,
            "empty_projection_lineage": preserved_lineage,
        },
        "performance_artifacts_sha256": performance_artifacts[
            "performance_artifacts_sha256"
        ],
        "execution_attempt": attempt,
        "barrier_release_allowed": True,
        "retry_logical_request_sha256": None,
    }
    return {
        **normalized_payload,
        "physical_terminal_batch_sha256": _sha(normalized_payload),
    }


def validate_physical_terminal_batch(payload: Mapping[str, Any]) -> JSON:
    """Validate executor output; pre-promotion cache wrappers are forbidden."""
    if not isinstance(payload, Mapping):
        raise ValueError("physical terminal batch must be a mapping")
    if "terminal_wrappers" in payload:
        raise ValueError("terminal wrappers are forbidden before promotion")
    if payload.get("schema_version") == EMPTY_TERMINAL_SCHEMA:
        payload = _normalize_empty_physical_terminal(payload)
    copied = copy.deepcopy(dict(payload))
    recorded = copied.pop("physical_terminal_batch_sha256", None)
    rows = copied.get("rows")
    lineages = copied.get("lineage_inputs")
    failures = copied.get("failures")
    if (
        set(payload) != _TERMINAL_FIELDS
        or copied.get("schema_version") != PHYSICAL_TERMINAL_SCHEMA
        or not _is_sha256(recorded)
        or recorded != _sha(copied)
        or not isinstance(rows, list)
        or not isinstance(lineages, list)
        or not isinstance(failures, list)
        or not all(isinstance(row, Mapping) for row in rows)
        or not all(isinstance(row, Mapping) for row in lineages)
        or not all(isinstance(row, Mapping) for row in failures)
        or not isinstance(copied.get("projection_artifact"), Mapping)
        or not _is_sha256(copied.get("performance_artifacts_sha256"))
        or not isinstance(copied.get("execution_attempt"), Mapping)
    ):
        raise ValueError("physical terminal batch authentication failed")
    projection_artifact = copied["projection_artifact"]
    projection_fields = frozenset(projection_artifact)
    empty_lineage = projection_artifact.get("empty_projection_lineage")
    if (
        projection_fields
        not in {
            frozenset(_PROJECTION_ARTIFACT_FIELDS),
            frozenset(_PROJECTION_ARTIFACT_FIELDS | {"empty_projection_lineage"}),
        }
        or projection_artifact.get("artifact_kind") != "stage5_independent_projection"
        or any(
            not _is_sha256(projection_artifact.get(field))
            for field in (
                "measurement_request_sha256",
                "projection_payload_sha256",
                "deployment_bundle_sha256",
            )
        )
        or (
            empty_lineage is not None
            and (
                not isinstance(empty_lineage, Mapping)
                or set(empty_lineage) != _EMPTY_PROJECTION_LINEAGE_FIELDS
                or any(
                    not _is_sha256(value)
                    for key, value in empty_lineage.items()
                    if key != "rows"
                )
                or empty_lineage.get("rows") != []
                or empty_lineage.get("logical_request_sha256")
                != projection_artifact.get("measurement_request_sha256")
                or empty_lineage.get("deployment_bundle_sha256")
                != projection_artifact.get("deployment_bundle_sha256")
            )
        )
    ):
        raise ValueError("physical projection artifact authentication failed")
    verified_attempt = _validate_execution_attempt(
        copied["execution_attempt"],
        deployment_bundle_sha256=projection_artifact["deployment_bundle_sha256"],
    )
    if empty_lineage is not None and (
        rows != []
        or lineages != []
        or failures != []
        or verified_attempt.get("zero_miss") is not True
        or verified_attempt.get("empty_physical_terminal_sha256")
        != projection_artifact.get("projection_payload_sha256")
    ):
        raise ValueError("empty physical terminal preservation drift")
    row_ids = [_row_id(row) for row in rows]
    lineage_ids = [str(row.get("candidate_id") or "") for row in lineages]
    failure_ids = [str(row.get("candidate_id") or "") for row in failures]
    if (
        len(set(row_ids)) != len(row_ids)
        or len(set(lineage_ids)) != len(lineage_ids)
        or len(set(failure_ids)) != len(failure_ids)
        or "" in lineage_ids
        or "" in failure_ids
    ):
        raise ValueError("physical terminal batch contains duplicate identities")
    successful_ids = {
        _row_id(row)
        for row in rows
        if row.get("terminal_status") == stage5_search.SUCCESS_STATUS
    }
    candidate_failure_ids = {
        _row_id(row)
        for row in rows
        if row.get("terminal_status") in stage5_search.TRUE_FAILURE_STATUSES
    }
    if any(
        row.get("terminal_status")
        not in {stage5_search.SUCCESS_STATUS, *stage5_search.TRUE_FAILURE_STATUSES}
        for row in rows
    ):
        raise ValueError("physical terminal batch contains a nonterminal row")
    reported_candidate_failures = {
        str(row["candidate_id"])
        for row in failures
        if row.get("failure_class") == "candidate"
    }
    retry_failure_ids = {
        str(row["candidate_id"])
        for row in failures
        if row.get("failure_class") in {"infrastructure", "evidence"}
    }
    if (
        set(lineage_ids) != successful_ids
        or reported_candidate_failures != candidate_failure_ids
        or retry_failure_ids & set(row_ids)
    ):
        raise ValueError("physical terminal row/failure/lineage identity drift")
    retryable = any(
        failure.get("failure_class") in {"infrastructure", "evidence"}
        for failure in failures
    )
    if (
        copied.get("barrier_release_allowed") is not (not retryable)
        or (retryable and not _is_sha256(copied.get("retry_logical_request_sha256")))
        or (not retryable and copied.get("retry_logical_request_sha256") is not None)
    ):
        raise ValueError("physical terminal retry/barrier contract drift")
    return {**copied, "physical_terminal_batch_sha256": recorded}


def build_post_promotion_terminal_wrappers(
    *,
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    promoted_rows: Sequence[Mapping[str, Any]],
    lineage_inputs: Sequence[Mapping[str, Any]],
    stage5_terminal_artifacts: Mapping[str, Mapping[str, Any]],
) -> list[JSON]:
    """Build cache wrappers only from successful actual-feedback-v3 rows."""
    promoted_by_id = _unique_by_id(promoted_rows, name="promoted feedback")
    lineage_by_id = _unique_by_id(lineage_inputs, name="raw lineage")
    miss_ids = {
        str(row.get("candidate_id") or "")
        for row in physical_plan.get("logical_row_bindings") or ()
        if row.get("disposition") == "miss"
    }
    success_ids = [
        candidate_id
        for candidate_id, row in promoted_by_id.items()
        if candidate_id in miss_ids
        and row.get("terminal_status") == stage5_search.SUCCESS_STATUS
    ]
    if set(lineage_by_id) != set(success_ids) or set(stage5_terminal_artifacts) != set(
        success_ids
    ):
        raise ValueError("post-promotion wrapper evidence is incomplete")
    wrappers = []
    for candidate_id in success_ids:
        row = promoted_by_id[candidate_id]
        if (
            row.get("feedback_feature_contract") != "actual_feedback_v3"
            or not _is_sha256(row.get("actual_feedback_row_sha256"))
            or not _is_sha256(row.get("materialized_graph_features_sha256"))
            or not isinstance(row.get("graph_features"), Mapping)
        ):
            raise ValueError("historical feedback cannot create a cache wrapper")
        lineage = lineage_by_id[candidate_id]
        if set(lineage) != {
            "candidate_id",
            "stage3_performance_artifact",
            "stage3_ap_artifact",
        }:
            raise ValueError("raw Stage3 lineage shape drift")
        wrappers.append(
            actual_adapter.wrap_actual_v3_terminal(
                logical_request,
                selection_binding,
                physical_plan,
                candidate_id=candidate_id,
                stage5_terminal=row,
                stage5_terminal_artifact=stage5_terminal_artifacts[candidate_id],
                stage3_performance_artifact=lineage["stage3_performance_artifact"],
                stage3_ap_artifact=lineage["stage3_ap_artifact"],
                actual_graph_features=row["graph_features"],
            )
        )
    return wrappers


__all__ = [
    "EMPTY_TERMINAL_SCHEMA",
    "FROZEN_STAGE3_FINALIZER_SHA256",
    "PHYSICAL_TERMINAL_SCHEMA",
    "STRUCTURED_FAILURE_SCHEMA",
    "build_post_promotion_terminal_wrappers",
    "classify_structured_failure",
    "finalize_physical_rows",
    "validate_performance_projection_lineage",
    "validate_physical_terminal_batch",
]
