from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from framework.stage7 import physical_feedback_v2 as feedback
from scripts import stage7_actual_feedback_barrier_v2 as barrier


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _request(count: int = 2) -> dict[str, Any]:
    rows = [
        {
            "row_id": f"c{index}",
            "manifest_job_id": f"c{index}",
            "group_id": f"g{index}",
            "model": "pyramid",
            "width": [16, 32, 64],
            "q_mode": "fp16",
            "dispatch_key": "tvm_auto",
            "graph_features": {"graph_feature_provenance": "candidate"},
        }
        for index in range(count)
    ]
    row_sha = {row["row_id"]: _sha(row) for row in rows}
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "a" * 64,
        "round_index": 0,
        "batch_size": count,
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": row_sha,
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _projection(request: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "stage5_independent_validation_request_v1",
        "task_id": request["task_id"],
        "task_sha256": request["task_sha256"],
        "round_index": 0,
        "batch_size": len(request["rows"]),
        "required_metrics": request["required_metrics"],
        "real_h800_measurement_required": True,
        "independent_from_search_measurement": True,
        "row_sha256": copy.deepcopy(request["row_sha256"]),
        "rows": copy.deepcopy(request["rows"]),
        "stage7_projection_lineage": {
            "physical_request_sha256": "b" * 64,
            "deployment_bundle_sha256": "d" * 64,
            "rows": [],
        },
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _selection(request: dict[str, Any]) -> dict[str, Any]:
    selected = [
        {
            "candidate_id": row["row_id"],
            "logical_row_sha256": request["row_sha256"][row["row_id"]],
            "logical_exact_binding_sha256": f"{index + 1:x}" * 64,
            "exact_cache_key_sha256": f"{index + 3:x}" * 64,
            "exact_key_dimensions": {"candidate_id": row["row_id"]},
        }
        for index, row in enumerate(request["rows"])
    ]
    return {
        "logical_request": {
            "logical_request_sha256": request["measurement_request_sha256"]
        },
        "selection_binding_sha256": "e" * 64,
        "selected_candidates": selected,
    }


def _plan(request: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "physical_request_sha256": "b" * 64,
        "logical_row_bindings": [
            {
                "candidate_id": row["row_id"],
                "disposition": "miss",
                "physical_row_sha256": request["row_sha256"][row["row_id"]],
            }
            for row in request["rows"]
        ],
    }


def _performance_artifacts(
    request: dict[str, Any], projection: dict[str, Any]
) -> dict[str, Any]:
    jobs = [
        {
            **copy.deepcopy(row),
            "job_id": row["row_id"],
            "source_evidence_path": f"/evidence/{row['row_id']}.json",
            "source_evidence_sha256": "f" * 64,
        }
        for row in request["rows"]
    ]
    payload = {
        "schema_version": "stage7_authenticated_performance_artifacts_v2",
        "independent_request_sha256": projection["measurement_request_sha256"],
        "planned_candidate_ids": [row["row_id"] for row in request["rows"]],
        "manifest": {
            "schema_version": "stage5_performance_manifest_v2",
            "source_request_sha256": projection["measurement_request_sha256"],
            "jobs": jobs,
        },
        "manifest_row_count": len(jobs),
        "performance_jobs": [
            {"manifest_job_id": row["row_id"]} for row in request["rows"]
        ],
        "planner_call_count": 1,
        "source_binding_audit": {},
    }
    return {**payload, "performance_artifacts_sha256": _sha(payload)}


def _attempt() -> dict[str, Any]:
    payload = {
        "attempt_id": "attempt_000",
        "deployment_bundle_sha256": "d" * 64,
        "primitive_sha256": {"stage3_finalizer": "7" * 64},
    }
    return {**payload, "execution_attempt_sha256": _sha(payload)}


def _empty_projection_lineage(
    *,
    logical_request_sha256: str = "1" * 64,
    deployment_bundle_sha256: str = "2" * 64,
) -> dict[str, Any]:
    return {
        "logical_request_sha256": logical_request_sha256,
        "selection_binding_sha256": "3" * 64,
        "cache_snapshot_sha256": "4" * 64,
        "cache_reveal_sha256": "5" * 64,
        "physical_request_sha256": "6" * 64,
        "source_resolution_plan_sha256": "7" * 64,
        "source_resolution_result_sha256": "8" * 64,
        "resolved_source_lock_sha256": _sha([]),
        "executor_admission_sha256": "9" * 64,
        "deployment_bundle_sha256": deployment_bundle_sha256,
        "rows": [],
    }


@pytest.mark.parametrize(
    "first_import",
    [
        "framework.stage7.physical_feedback_v2",
        "framework.stage7.physical_execution_v2",
        "framework.stage7.physical_runtime_v2",
        "scripts.stage7_scheduler_requests_v2",
        "scripts.stage7_core_online_ablation_v2",
    ],
)
def test_physical_feedback_import_chain_is_order_independent(
    first_import: str,
) -> None:
    modules = [
        first_import,
        "framework.stage7.physical_feedback_v2",
        "framework.stage7.physical_execution_v2",
        "framework.stage7.physical_runtime_v2",
        "scripts.stage7_scheduler_requests_v2",
        "scripts.stage7_core_online_ablation_v2",
    ]
    command = "; ".join(f"import {module}" for module in modules)
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _report(
    root: Path, candidate_id: str, failure_class: str, reason: str
) -> dict[str, Any]:
    primitive_status = (
        "confirmed_failure" if failure_class == "candidate" else "runner_failed"
    )
    evidence = {
        "candidate_id": candidate_id,
        "failure_class": failure_class,
        "failure_reason": reason,
        "status": primitive_status,
    }
    evidence_path = root / f"{candidate_id}-{failure_class}-primitive.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    payload = {
        "schema_version": feedback.STRUCTURED_FAILURE_SCHEMA,
        "candidate_id": candidate_id,
        "failure_class": failure_class,
        "failure_reason": reason,
        "evidence_authenticated": True,
        "primitive_terminal_status": primitive_status,
        "primitive_evidence": {
            "artifact_kind": "stage7_structured_primitive_terminal",
            "path": str(evidence_path),
            "artifact_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        },
    }
    return {**payload, "structured_failure_report_sha256": _sha(payload)}


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_finalize_calls_stage3_once_per_success_or_candidate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    request = _request(count)
    projection = _projection(request)
    selection = _selection(request)
    plan = _plan(request, selection)
    artifacts = _performance_artifacts(request, projection)
    calls: list[str] = []

    def fake_finalize(
        source: dict[str, Any],
        performance_rows: list[dict[str, Any]],
        _ap_rows: list[dict[str, Any]],
        *,
        output_schema: str,
    ) -> dict[str, Any]:
        calls.append(source["job_id"])
        candidate_failure = performance_rows[-1]["status"] == "confirmed_failure"
        return {
            "schema_version": output_schema,
            "manifest_job_id": source["job_id"],
            "terminal_status": (
                "feasibility_failure" if candidate_failure else "measured_success_gold"
            ),
            "failure_reason": "build_failure" if candidate_failure else None,
            "latency_ms": None if candidate_failure else 1.0,
            "energy_j": None if candidate_failure else 2.0,
            "ap30": None if candidate_failure else 0.3,
            "ap50": None if candidate_failure else 0.2,
            "ap70": None if candidate_failure else 0.1,
        }

    monkeypatch.setattr(
        feedback.actual_adapter,
        "finalize_actual_v3_failure",
        lambda *_args, candidate_id, failure_class, reason, **_kwargs: {
            "candidate_id": candidate_id,
            "failure_class": failure_class,
            "failure_reason": reason,
            "consumes_selected_event_budget": failure_class == "candidate",
            "retry_logical_request_sha256": (
                None
                if failure_class == "candidate"
                else request["measurement_request_sha256"]
            ),
        },
    )
    reports = (
        [_report(tmp_path, "c0", "candidate", "build_failure")] if count > 1 else []
    )
    result = feedback.finalize_physical_rows(
        logical_request=request,
        selection_binding=selection,
        physical_plan=plan,
        projection=projection,
        performance_artifacts=artifacts,
        performance_state_rows=[
            {"manifest_job_id": row["row_id"], "status": "success"}
            for row in request["rows"]
        ],
        ap_plan_rows=[
            {"manifest_job_id": row["row_id"], "performance_job_id": row["row_id"]}
            for row in request["rows"]
        ],
        ap_state_rows=[],
        structured_failure_reports=reports,
        lineage_inputs=[
            {
                "candidate_id": row["row_id"],
                "stage3_performance_artifact": {},
                "stage3_ap_artifact": {},
            }
            for row in request["rows"]
            if row["row_id"] != "c0" or count == 1
        ],
        execution_attempt=_attempt(),
        finalize_row=fake_finalize,
        validate_projection=lambda value: copy.deepcopy(value),
        validate_physical_plan=lambda *_args: (selection, plan),
    )
    assert calls == [row["row_id"] for row in request["rows"]]
    assert len(result["rows"]) == count
    assert "terminal_wrappers" not in result
    if count > 1:
        assert result["rows"][0]["terminal_status"] == "feasibility_failure"
        assert all(
            result["rows"][0][metric] is None
            for metric in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
        )


@pytest.mark.parametrize(
    ("failure_class", "reason"),
    [
        ("infrastructure", "gpu_occupancy_drift"),
        ("evidence", "evidence_missing"),
    ],
)
def test_non_candidate_failure_never_enters_stage3_and_blocks_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_class: str,
    reason: str,
) -> None:
    request = _request(1)
    projection = _projection(request)
    selection = _selection(request)
    plan = _plan(request, selection)
    monkeypatch.setattr(
        feedback.actual_adapter,
        "finalize_actual_v3_failure",
        lambda *_args, candidate_id, failure_class, reason, **_kwargs: {
            "candidate_id": candidate_id,
            "failure_class": failure_class,
            "failure_reason": reason,
            "consumes_selected_event_budget": False,
            "retry_logical_request_sha256": request["measurement_request_sha256"],
        },
    )

    result = feedback.finalize_physical_rows(
        logical_request=request,
        selection_binding=selection,
        physical_plan=plan,
        projection=projection,
        performance_artifacts=_performance_artifacts(request, projection),
        performance_state_rows=[
            {
                "manifest_job_id": "c0",
                "status": "failed",
                "returncode": 1,
            }
        ],
        ap_plan_rows=[{"manifest_job_id": "c0", "performance_job_id": "c0"}],
        ap_state_rows=[],
        structured_failure_reports=[_report(tmp_path, "c0", failure_class, reason)],
        lineage_inputs=[],
        execution_attempt=_attempt(),
        finalize_row=lambda *_args, **_kwargs: pytest.fail(
            "_finalize_row must not run"
        ),
        validate_projection=lambda value: copy.deepcopy(value),
        validate_physical_plan=lambda *_args: (selection, plan),
    )
    assert result["rows"] == []
    assert result["failures"][0]["failure_class"] == failure_class
    assert result["barrier_release_allowed"] is False
    assert (
        result["retry_logical_request_sha256"] == request["measurement_request_sha256"]
    )


def test_unstructured_returncode_cannot_be_candidate_failure() -> None:
    classified = feedback.classify_structured_failure(
        candidate_id="c0",
        performance_rows=[{"status": "failed", "returncode": 1}],
        ap_rows=[],
        structured_report=None,
    )
    assert classified["failure_class"] == "infrastructure"
    assert classified["consumes_selected_event_budget"] is False


def test_confirmed_performance_failure_consumes_budget_without_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(1)
    projection = _projection(request)
    selection = _selection(request)
    plan = _plan(request, selection)

    def fake_finalize(
        source: dict[str, Any],
        performance_rows: list[dict[str, Any]],
        _ap_rows: list[dict[str, Any]],
        *,
        output_schema: str,
    ) -> dict[str, Any]:
        assert source["job_id"] == "c0"
        assert performance_rows[-1]["status"] == "confirmed_failure"
        return {
            "schema_version": output_schema,
            "manifest_job_id": "c0",
            "terminal_status": "feasibility_failure",
            "failure_reason": "backend_failure",
            "latency_ms": None,
            "energy_j": None,
            "ap30": None,
            "ap50": None,
            "ap70": None,
        }

    monkeypatch.setattr(
        feedback.actual_adapter,
        "finalize_actual_v3_failure",
        lambda *_args, candidate_id, failure_class, reason, **_kwargs: {
            "candidate_id": candidate_id,
            "failure_class": failure_class,
            "failure_reason": reason,
            "consumes_selected_event_budget": True,
            "retry_logical_request_sha256": None,
        },
    )
    result = feedback.finalize_physical_rows(
        logical_request=request,
        selection_binding=selection,
        physical_plan=plan,
        projection=projection,
        performance_artifacts=_performance_artifacts(request, projection),
        performance_state_rows=[
            {
                "manifest_job_id": "c0",
                "status": "confirmed_failure",
                "returncode": 1,
            }
        ],
        ap_plan_rows=[{"manifest_job_id": "c0", "performance_job_id": "c0"}],
        ap_state_rows=[],
        structured_failure_reports=[],
        lineage_inputs=[],
        execution_attempt=_attempt(),
        finalize_row=fake_finalize,
        validate_projection=lambda value: copy.deepcopy(value),
        validate_physical_plan=lambda *_args: (selection, plan),
    )
    assert result["barrier_release_allowed"] is True
    assert result["retry_logical_request_sha256"] is None
    assert result["rows"][0]["terminal_status"] == "feasibility_failure"
    assert result["failures"][0]["failure_class"] == "candidate"
    assert result["failures"][0]["failure_reason"] == "backend_failure"


def test_performance_manifest_retains_projection_sha() -> None:
    request = _request(1)
    projection = _projection(request)
    artifacts = _performance_artifacts(request, projection)
    drifted = copy.deepcopy(artifacts)
    drifted["manifest"]["source_request_sha256"] = request["measurement_request_sha256"]
    unsigned = copy.deepcopy(drifted)
    unsigned.pop("performance_artifacts_sha256")
    drifted["performance_artifacts_sha256"] = _sha(unsigned)
    with pytest.raises(ValueError, match="projection SHA"):
        feedback.validate_performance_projection_lineage(
            projection,
            drifted,
            validate_projection=lambda value: copy.deepcopy(value),
        )


def test_physical_terminal_rejects_prepromotion_terminal_wrappers() -> None:
    payload = {
        "schema_version": feedback.PHYSICAL_TERMINAL_SCHEMA,
        "rows": [],
        "lineage_inputs": [],
        "failures": [],
        "terminal_wrappers": [],
    }
    with pytest.raises(ValueError, match="promotion"):
        feedback.validate_physical_terminal_batch(payload)


def test_zero_miss_empty_terminal_normalizes_without_gpu_or_wrappers() -> None:
    payload = {
        "schema_version": "stage7_actual_v3_empty_physical_terminal_v2",
        "stage7_projection_lineage": _empty_projection_lineage(),
        "rows": [],
        "lineage_inputs": [],
        "gpu_subprocess_count": 0,
    }
    empty = {**payload, "empty_physical_terminal_sha256": _sha(payload)}
    normalized = feedback.validate_physical_terminal_batch(empty)
    assert normalized["schema_version"] == feedback.PHYSICAL_TERMINAL_SCHEMA
    assert normalized["rows"] == []
    assert normalized["lineage_inputs"] == []
    assert normalized["failures"] == []
    assert normalized["barrier_release_allowed"] is True
    assert normalized["execution_attempt"]["gpu_subprocess_count"] == 0
    assert (
        normalized["projection_artifact"]["empty_projection_lineage"]
        == payload["stage7_projection_lineage"]
    )
    assert "terminal_wrappers" not in normalized
    assert feedback.validate_physical_terminal_batch(normalized) == normalized


@pytest.mark.parametrize(
    "field",
    [
        "physical_request_sha256",
        "cache_reveal_sha256",
        "executor_admission_sha256",
        "source_resolution_plan_sha256",
        "source_resolution_result_sha256",
    ],
)
def test_zero_miss_empty_terminal_rejects_incomplete_full_lineage(field: str) -> None:
    lineage = _empty_projection_lineage()
    lineage.pop(field)
    payload = {
        "schema_version": feedback.EMPTY_TERMINAL_SCHEMA,
        "stage7_projection_lineage": lineage,
        "rows": [],
        "lineage_inputs": [],
        "gpu_subprocess_count": 0,
    }
    terminal = {**payload, "empty_physical_terminal_sha256": _sha(payload)}
    with pytest.raises(ValueError, match="lineage"):
        feedback.validate_physical_terminal_batch(terminal)


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("physical_request_sha256", "a" * 64),
        ("cache_reveal_sha256", "b" * 64),
        ("executor_admission_sha256", "c" * 64),
        ("source_resolution_plan_sha256", "d" * 64),
        ("source_resolution_result_sha256", "e" * 64),
    ],
)
def test_barrier_rejects_zero_miss_lineage_drift_before_promotion(
    field: str, drifted: str
) -> None:
    lineage = _empty_projection_lineage()
    terminal = {
        "projection_artifact": {"empty_projection_lineage": {**lineage, field: drifted}}
    }
    with pytest.raises(ValueError, match="zero-miss"):
        barrier.validate_zero_miss_round_lineage(
            terminal,
            request={"measurement_request_sha256": lineage["logical_request_sha256"]},
            selection_binding={
                "selection_binding_sha256": lineage["selection_binding_sha256"]
            },
            cache_reveal={
                "cache_snapshot_sha256": lineage["cache_snapshot_sha256"],
                "cache_reveal_sha256": lineage["cache_reveal_sha256"],
            },
            physical_plan={
                "physical_request_sha256": lineage["physical_request_sha256"]
            },
            source_plan={
                "source_resolution_plan_sha256": lineage[
                    "source_resolution_plan_sha256"
                ]
            },
            source_result={
                "source_resolution_result_sha256": lineage[
                    "source_resolution_result_sha256"
                ]
            },
            stored_admission={"admission_sha256": lineage["executor_admission_sha256"]},
        )


def test_post_promotion_wrapper_builder_uses_promoted_row_and_raw_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(1)
    selection = _selection(request)
    plan = _plan(request, selection)
    promoted = {
        **request["rows"][0],
        "terminal_status": "measured_success_gold",
        "actual_feedback_row_sha256": "9" * 64,
        "materialized_graph_features_sha256": "8" * 64,
        "graph_features": {"graph_feature_provenance": "materialized"},
        "feedback_feature_contract": "actual_feedback_v3",
    }
    seen: list[dict[str, Any]] = []

    def fake_wrap(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(copy.deepcopy(kwargs))
        return {"candidate_id": kwargs["candidate_id"], "wrapped": True}

    monkeypatch.setattr(feedback.actual_adapter, "wrap_actual_v3_terminal", fake_wrap)
    wrappers = feedback.build_post_promotion_terminal_wrappers(
        logical_request=request,
        selection_binding=selection,
        physical_plan=plan,
        promoted_rows=[promoted],
        lineage_inputs=[
            {
                "candidate_id": "c0",
                "stage3_performance_artifact": {"raw": "performance"},
                "stage3_ap_artifact": {"raw": "ap"},
            }
        ],
        stage5_terminal_artifacts={
            "c0": {"artifact_kind": "stage5_terminal", "path": "/terminal"}
        },
    )
    assert wrappers == [{"candidate_id": "c0", "wrapped": True}]
    assert seen[0]["stage5_terminal"] == promoted
    assert seen[0]["stage3_performance_artifact"] == {"raw": "performance"}
    assert seen[0]["actual_graph_features"] == promoted["graph_features"]


def test_post_promotion_wrapper_builder_excludes_successful_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(2)
    selection = _selection(request)
    plan = _plan(request, selection)
    plan["logical_row_bindings"][0]["disposition"] = "hit"
    promoted = [
        {
            **row,
            "terminal_status": "measured_success_gold",
            "actual_feedback_row_sha256": f"{index + 7:x}" * 64,
            "materialized_graph_features_sha256": f"{index + 8:x}" * 64,
            "graph_features": {"graph_feature_provenance": "materialized"},
            "feedback_feature_contract": "actual_feedback_v3",
        }
        for index, row in enumerate(request["rows"])
    ]
    monkeypatch.setattr(
        feedback.actual_adapter,
        "wrap_actual_v3_terminal",
        lambda *_args, **kwargs: {"candidate_id": kwargs["candidate_id"]},
    )
    wrappers = feedback.build_post_promotion_terminal_wrappers(
        logical_request=request,
        selection_binding=selection,
        physical_plan=plan,
        promoted_rows=promoted,
        lineage_inputs=[
            {
                "candidate_id": "c1",
                "stage3_performance_artifact": {},
                "stage3_ap_artifact": {},
            }
        ],
        stage5_terminal_artifacts={"c1": {}},
    )
    assert wrappers == [{"candidate_id": "c1"}]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_stage3_finalizer_default_matches_frozen_primitive_for_one_to_four_rows(
    count: int, tmp_path: Path
) -> None:
    request = _request(count)
    projection = _projection(request)
    selection = _selection(request)
    plan = _plan(request, selection)
    artifacts = _performance_artifacts(request, projection)
    performance_rows = []
    ap_rows = []
    lineage_inputs = []
    expected_by_id: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(artifacts["manifest"]["jobs"]):
        candidate_id = source["job_id"]
        performance_path = tmp_path / f"{candidate_id}_performance.json"
        performance_path.write_text(
            json.dumps({"latency_ms": 1.0 + index, "energy_j": 2.0 + index}),
            encoding="utf-8",
        )
        ap_path = tmp_path / f"{candidate_id}_ap.json"
        ap_path.write_text(
            json.dumps(
                {
                    "status": "success",
                    "processed_samples": 1789,
                    "fallback_samples": 0,
                    "failed_samples": 0,
                    "ap_measured": True,
                    "smoke_gate_passed": True,
                    "ap30": 0.8,
                    "ap50": 0.7,
                    "ap70": 0.6,
                }
            ),
            encoding="utf-8",
        )
        candidate_performance = {
            "manifest_job_id": candidate_id,
            "status": "success",
            "result_json": str(performance_path),
        }
        candidate_ap = {
            "manifest_job_id": candidate_id,
            "record_type": "terminal_event",
            "stage": "full",
            "status": "success",
            "report_path": str(ap_path),
        }
        performance_rows.append(candidate_performance)
        ap_rows.append(candidate_ap)
        lineage_inputs.append(
            {
                "candidate_id": candidate_id,
                "stage3_performance_artifact": {
                    "path": str(performance_path),
                    "artifact_sha256": hashlib.sha256(
                        performance_path.read_bytes()
                    ).hexdigest(),
                },
                "stage3_ap_artifact": {
                    "path": str(ap_path),
                    "artifact_sha256": hashlib.sha256(ap_path.read_bytes()).hexdigest(),
                },
            }
        )
        expected_by_id[candidate_id] = feedback.stage3_finalizer._finalize_row(
            source,
            [candidate_performance],
            [candidate_ap],
            output_schema="stage5_feedback_row_v2",
        )

    terminal = feedback.finalize_physical_rows(
        logical_request=request,
        selection_binding=selection,
        physical_plan=plan,
        projection=projection,
        performance_artifacts=artifacts,
        performance_state_rows=performance_rows,
        ap_plan_rows=[
            {
                "manifest_job_id": row["row_id"],
                "performance_job_id": row["row_id"],
            }
            for row in request["rows"]
        ],
        ap_state_rows=ap_rows,
        structured_failure_reports=[],
        lineage_inputs=lineage_inputs,
        execution_attempt=_attempt(),
        validate_projection=lambda value: copy.deepcopy(value),
        validate_physical_plan=lambda *_args: (
            copy.deepcopy(selection),
            copy.deepcopy(plan),
        ),
    )

    assert len(terminal["rows"]) == count
    for actual in terminal["rows"]:
        expected = expected_by_id[actual["manifest_job_id"]]
        assert all(actual[key] == value for key, value in expected.items())


def test_stage3_finalizer_pin_rejects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drifted = tmp_path / "stage3_finalize_gold96_v3.py"
    drifted.write_text("# drift\n", encoding="utf-8")
    monkeypatch.setattr(feedback.stage3_finalizer, "__file__", str(drifted))
    with pytest.raises(ValueError, match="Stage3 finalizer"):
        feedback._validate_stage3_finalizer()


def test_stage3_finalizer_pin_rejects_runtime_callable_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback.stage3_finalizer, "_finalize_row", lambda: None)
    with pytest.raises(ValueError, match="Stage3 finalizer"):
        feedback._validate_stage3_finalizer()


def test_barrier_orders_promotion_before_wrapper_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "round"
    directory.mkdir()
    request = _request(1)
    for name, payload in {
        "logical_request.json": request,
        "selection_binding.json": {},
        "source_resolution_plan.json": {},
        "source_resolution_result.json": {},
        "exact_selection_binding.json": {},
        "cache_snapshot_before_reveal.json": {},
        "cache_reveal.json": {
            "entries": [{"candidate_id": "c0", "disposition": "miss"}]
        },
        "miss_only_physical_request.json": {},
        "executor_admission.json": {},
    }.items():
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    terminal = {
        "schema_version": feedback.PHYSICAL_TERMINAL_SCHEMA,
        "rows": [
            {
                **request["rows"][0],
                "terminal_status": "measured_success_gold",
                "measurement_request_row_sha256": request["row_sha256"]["c0"],
            }
        ],
        "lineage_inputs": [
            {
                "candidate_id": "c0",
                "stage3_performance_artifact": {},
                "stage3_ap_artifact": {},
            }
        ],
        "failures": [],
        "projection_artifact": {"sha256": "1" * 64},
        "performance_artifacts_sha256": "2" * 64,
        "execution_attempt": {},
        "barrier_release_allowed": True,
        "retry_logical_request_sha256": None,
    }
    terminal["physical_terminal_batch_sha256"] = _sha(terminal)
    terminal_path = tmp_path / "terminal.json"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    order: list[str] = []
    monkeypatch.setattr(
        barrier.source_resolution,
        "validate_source_resolution_plan",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        barrier.source_resolution,
        "validate_formal_source_resolution_result",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        barrier.physical_feedback,
        "validate_physical_terminal_batch",
        lambda value: copy.deepcopy(value),
    )
    monkeypatch.setattr(
        barrier.physical_feedback,
        "build_post_promotion_terminal_wrappers",
        lambda **_kwargs: order.append("wrappers") or [{"candidate_id": "c0"}],
    )
    monkeypatch.setattr(
        barrier,
        "_write_promoted_terminal_artifacts",
        lambda *_args, **_kwargs: {"c0": {}},
    )

    def promote_atomic(**_kwargs: Any) -> dict[str, Any]:
        order.append("promote_atomic")
        return {
            "promotion": {
                "rows": [
                    {
                        **terminal["rows"][0],
                        "actual_feedback_row_sha256": "9" * 64,
                    }
                ],
                "audit": {"silent_surrogate_fallback_count": 0},
            },
            "barrier": {
                "feedback_released": True,
                "budget_consumed": 1,
                "successful_rows": 1,
                "feasibility_terminal_rows": 0,
            },
        }

    def append_wrappers(cache: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        order.append("cache")
        return cache

    result = barrier.finalize_round(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        terminal_payload_path=terminal_path,
        repo_root=tmp_path,
        validate_root=lambda *_args, **_kwargs: {"root": tmp_path},
        round_directory=lambda *_args: directory,
        validate_identity=lambda *_args: {
            "logical_request_sha256": request["measurement_request_sha256"]
        },
        merge_logical=lambda _request, rows: list(rows),
        promote_atomic=promote_atomic,
        append_wrappers=append_wrappers,
        validate_admission=lambda *_args, **_kwargs: {},
        write_json=lambda path, payload, **_kwargs: path.write_text(
            json.dumps(payload), encoding="utf-8"
        ),
        validate_precommit=lambda **_kwargs: {},
        validate_selection_binding=lambda *_args: {
            "logical_request": {
                "logical_request_sha256": request["measurement_request_sha256"]
            }
        },
    )
    assert result["feedback_released"] is True
    assert order == ["promote_atomic", "wrappers", "cache"]


def test_barrier_commits_four_cache_hits_from_authenticated_empty_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "round"
    directory.mkdir()
    request = _request(4)
    cached_rows = []
    reveal_entries = []
    for row in request["rows"]:
        candidate_id = row["row_id"]
        cached = {
            **row,
            "terminal_status": "measured_success_gold",
            "feedback_feature_contract": "actual_feedback_v3",
            "actual_feedback_row_sha256": "9" * 64,
        }
        path = tmp_path / f"{candidate_id}_cached.json"
        path.write_text(json.dumps(cached), encoding="utf-8")
        cached_rows.append(cached)
        reveal_entries.append(
            {
                "candidate_id": candidate_id,
                "disposition": "hit",
                "terminal_evidence": {
                    "stage5_terminal_artifact": {
                        "path": str(path),
                        "artifact_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        )
    exact_binding = {"selection_binding_sha256": "3" * 64}
    source_plan = {"source_resolution_plan_sha256": "7" * 64}
    source_result = {"source_resolution_result_sha256": "8" * 64}
    cache_reveal = {
        "cache_snapshot_sha256": "4" * 64,
        "cache_reveal_sha256": "5" * 64,
        "entries": reveal_entries,
    }
    physical_plan = {
        "physical_request_sha256": "6" * 64,
        "physical_row_count": 0,
        "rows": [],
        "logical_row_bindings": [
            {"candidate_id": row["row_id"], "disposition": "hit"}
            for row in request["rows"]
        ],
    }
    for name, payload in {
        "logical_request.json": request,
        "selection_binding.json": {},
        "source_resolution_plan.json": source_plan,
        "source_resolution_result.json": source_result,
        "exact_selection_binding.json": exact_binding,
        "cache_snapshot_before_reveal.json": {"entries": {}},
        "cache_reveal.json": cache_reveal,
        "miss_only_physical_request.json": physical_plan,
        "executor_admission.json": {"admission_sha256": "9" * 64},
    }.items():
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    empty_payload = {
        "schema_version": feedback.EMPTY_TERMINAL_SCHEMA,
        "stage7_projection_lineage": _empty_projection_lineage(
            logical_request_sha256=request["measurement_request_sha256"],
            deployment_bundle_sha256="d" * 64,
        ),
        "rows": [],
        "lineage_inputs": [],
        "gpu_subprocess_count": 0,
    }
    empty_terminal = {
        **empty_payload,
        "empty_physical_terminal_sha256": _sha(empty_payload),
    }
    terminal_path = tmp_path / "empty_terminal.json"
    terminal_path.write_text(json.dumps(empty_terminal), encoding="utf-8")
    monkeypatch.setattr(
        barrier.source_resolution,
        "validate_source_resolution_plan",
        lambda value, **_kwargs: copy.deepcopy(value),
    )
    monkeypatch.setattr(
        barrier.source_resolution,
        "validate_formal_source_resolution_result",
        lambda value, *_args, **_kwargs: copy.deepcopy(value),
    )

    def promote_atomic(**_kwargs: Any) -> dict[str, Any]:
        return {
            "promotion": {
                "rows": copy.deepcopy(cached_rows),
                "audit": {"silent_surrogate_fallback_count": 0},
            },
            "barrier": {
                "feedback_released": True,
                "budget_consumed": 4,
                "successful_rows": 4,
                "feasibility_terminal_rows": 0,
            },
        }

    append_calls = 0

    def append_wrappers(cache: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        nonlocal append_calls
        append_calls += 1
        assert kwargs["terminal_wrappers"] == []
        return copy.deepcopy(cache)

    committed = barrier.finalize_round(
        tmp_path,
        variant="full",
        seed=20260718,
        round_index=0,
        terminal_payload_path=terminal_path,
        repo_root=tmp_path,
        validate_root=lambda *_args, **_kwargs: {"root": tmp_path},
        round_directory=lambda *_args: directory,
        validate_identity=lambda *_args: {
            "logical_request_sha256": request["measurement_request_sha256"]
        },
        merge_logical=lambda _request, rows: list(rows),
        promote_atomic=promote_atomic,
        append_wrappers=append_wrappers,
        validate_admission=lambda *_args, **_kwargs: {},
        write_json=lambda path, payload, **_kwargs: (
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_text(json.dumps(payload), encoding="utf-8"),
        ),
        validate_precommit=lambda **_kwargs: {},
        validate_selection_binding=lambda *_args: {
            "logical_request": {
                "logical_request_sha256": request["measurement_request_sha256"]
            }
        },
    )

    assert committed["feedback_released"] is True
    assert committed["budget_consumed"] == 4
    assert append_calls == 1
    assert not (directory / "promoted_stage5_terminals").exists()


def test_promoted_terminal_sidecars_include_only_successful_misses(
    tmp_path: Path,
) -> None:
    written: list[Path] = []

    def writer(path: Path, payload: dict[str, Any], **_kwargs: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(barrier._json_bytes(payload))
        written.append(path)

    rows = [
        {"row_id": "hit", "terminal_status": "measured_success_gold"},
        {"row_id": "success", "terminal_status": "measured_success_gold"},
        {"row_id": "failure", "terminal_status": "feasibility_failure"},
    ]
    plan = {
        "logical_row_bindings": [
            {
                "logical_row_index": 0,
                "candidate_id": "hit",
                "disposition": "hit",
            },
            {
                "logical_row_index": 1,
                "candidate_id": "success",
                "disposition": "miss",
            },
            {
                "logical_row_index": 2,
                "candidate_id": "failure",
                "disposition": "miss",
            },
        ]
    }
    refs = barrier._write_promoted_terminal_artifacts(
        tmp_path,
        promoted_rows=rows,
        physical_plan=plan,
        write_json=writer,
        validate=lambda: {},
    )
    assert list(refs) == ["success"]
    assert written == [tmp_path / "promoted_stage5_terminals/logical_row_01.json"]
    assert (
        refs["success"]["artifact_sha256"]
        == hashlib.sha256(written[0].read_bytes()).hexdigest()
    )
    bad = copy.deepcopy(plan)
    bad["logical_row_bindings"][1]["logical_row_index"] = 4
    with pytest.raises(ValueError, match="logical row index"):
        barrier._write_promoted_terminal_artifacts(
            tmp_path,
            promoted_rows=rows,
            physical_plan=bad,
            write_json=writer,
            validate=lambda: {},
        )


def test_barrier_rejects_retryable_failure_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "schema_version": feedback.PHYSICAL_TERMINAL_SCHEMA,
        "rows": [],
        "lineage_inputs": [],
        "failures": [{"failure_class": "infrastructure"}],
        "projection_artifact": {"sha256": "1" * 64},
        "performance_artifacts_sha256": "2" * 64,
        "execution_attempt": {},
        "barrier_release_allowed": False,
        "retry_logical_request_sha256": "a" * 64,
    }
    payload["physical_terminal_batch_sha256"] = _sha(payload)
    terminal = tmp_path / "terminal.json"
    terminal.write_text(json.dumps(payload), encoding="utf-8")
    writes: list[Path] = []
    monkeypatch.setattr(
        barrier.physical_feedback,
        "validate_physical_terminal_batch",
        lambda value: copy.deepcopy(value),
    )
    with pytest.raises(ValueError, match="retryable"):
        barrier.finalize_round(
            tmp_path,
            variant="full",
            seed=20260718,
            round_index=0,
            terminal_payload_path=terminal,
            repo_root=tmp_path,
            validate_root=lambda *_args, **_kwargs: {"root": tmp_path},
            round_directory=lambda *_args: tmp_path,
            validate_identity=lambda *_args: {},
            merge_logical=lambda *_args: [],
            promote_atomic=lambda **_kwargs: pytest.fail("promotion ran"),
            append_wrappers=lambda *_args, **_kwargs: {},
            validate_admission=lambda *_args, **_kwargs: {},
            write_json=lambda path, *_args, **_kwargs: writes.append(path),
        )
    assert writes == []
