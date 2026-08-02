from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from framework.stage7 import cache_feedback_v1 as cache_feedback
from framework.stage7.online_component_ablation_v1 import (
    CACHE_KEY_DIMENSIONS,
    build_measurement_cache_key,
)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file(path: Path, content: bytes) -> tuple[str, str]:
    path.write_bytes(content)
    return str(path), hashlib.sha256(content).hexdigest()


def _request() -> dict[str, Any]:
    rows = []
    for index in range(4):
        row_id = f"row-{index}"
        rows.append(
            {
                "schema_version": "stage5_candidate_manifest_row_v2",
                "task_id": "S7-PYR-TVM",
                "task_sha256": "1" * 64,
                "row_id": row_id,
                "manifest_job_id": row_id,
                "group_id": f"group-{index}",
                "model": "pyramid",
                "width": [16 + index * 8, 32, 64],
                "genome": [16 + index * 8, 32, 64, "fp16"],
                "q_mode": "fp16",
                "hardware_id": "h800",
                "capability_profile_id": "h800-tvm-auto-v1",
                "source_evidence_sha256": f"{index + 2:x}" * 64,
                "graph_features": {
                    "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
                    "flops": 10.0 + index,
                },
            }
        )
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "1" * 64,
        "round_index": 2,
        "batch_size": 4,
        "sample_budget": 16,
        "required_metrics": [
            "latency_ms",
            "energy_j",
            "ap30",
            "ap50",
            "ap70",
        ],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _binding(request: dict[str, Any]) -> dict[str, Any]:
    row_ids = [row["row_id"] for row in request["rows"]]
    payload = {
        "schema_version": "stage7_request_binding_v1",
        "round_index": request["round_index"],
        "measurement_request_sha256": request["measurement_request_sha256"],
        "trajectory_contract_sha256": "a" * 64,
        "selected_row_ids": row_ids,
        "selected_row_ids_sha256": _sha(row_ids),
    }
    return {**payload, "request_binding_sha256": _sha(payload)}


def _dimensions(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for index, row in enumerate(request["rows"]):
        result[row["row_id"]] = {
            "model": row["model"],
            "capability_profile_id": row["capability_profile_id"],
            "hardware_id": row["hardware_id"],
            "measurement_scope": "formal_h800_latency_energy_ap",
            "input_protocol_sha256": "b" * 64,
            "batch_size": 1,
            "genome": copy.deepcopy(row["genome"]),
            "q_mode": row["q_mode"],
            "source_checkpoint_sha256": f"{index + 6:x}" * 64,
            "onnx_sha256": f"{index + 10:x}" * 64,
            "build_protocol_sha256": "c" * 64,
            "tuning_protocol_sha256": "d" * 64,
            "measurement_protocol_sha256": "e" * 64,
            "ap_protocol_sha256": "f" * 64,
        }
    return result


def _cache_entry(
    tmp_path: Path,
    request_row: dict[str, Any],
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    row_id = request_row["row_id"]
    metrics = {
        "latency_ms": 1.0,
        "energy_j": 2.0,
        "ap30": 0.8,
        "ap50": 0.7,
        "ap70": 0.6,
    }
    paths_and_shas = {}
    for label in ("latency_artifact", "energy_artifact", "ap_artifact"):
        path, digest = _file(
            tmp_path / f"{row_id}-{label}.json", f"{row_id}:{label}".encode()
        )
        paths_and_shas[f"{label}_path"] = path
        paths_and_shas[f"{label}_sha256"] = digest
    checkpoint_path, checkpoint_sha = _file(
        tmp_path / f"{row_id}.pth", f"{row_id}:checkpoint".encode()
    )
    onnx_path, onnx_sha = _file(
        tmp_path / f"{row_id}.onnx", f"{row_id}:onnx".encode()
    )
    dimensions["source_checkpoint_sha256"] = checkpoint_sha
    dimensions["onnx_sha256"] = onnx_sha
    terminal_payload = {
        "schema_version": "stage5_terminal_measurement_evidence_v1",
        "row_id": f"origin-{row_id}",
        "terminal_status": "measured_success_gold",
        **metrics,
    }
    terminal_path, terminal_sha = _file(
        tmp_path / f"{row_id}-terminal_evidence.json",
        json.dumps(terminal_payload, sort_keys=True).encode(),
    )
    materialized_payload = {
        "schema_version": "stage5_source_materialization_evidence_v1",
        "status": "ready",
        "group_id": request_row["group_id"],
        "model": request_row["model"],
        "width": "x".join(str(value) for value in request_row["width"]),
        "source_plan_sha256": request_row["source_evidence_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "onnx_path": onnx_path,
        "onnx_sha256": onnx_sha,
    }
    materialized_path, materialized_sha = _file(
        tmp_path / f"{row_id}-source.json",
        json.dumps(materialized_payload, sort_keys=True).encode(),
    )
    result = {
        "row_id": f"origin-{row_id}",
        "task_id": "S5-ORIGIN",
        "task_sha256": "9" * 64,
        "round_index": 0,
        "terminal_status": "measured_success_gold",
        **metrics,
        **paths_and_shas,
        "terminal_evidence_path": terminal_path,
        "terminal_evidence_sha256": terminal_sha,
        "source_checkpoint_path": checkpoint_path,
        "source_checkpoint_sha256": checkpoint_sha,
        "onnx_path": onnx_path,
        "onnx_sha256": onnx_sha,
        "materialized_source_evidence_path": materialized_path,
        "materialized_source_evidence_sha256": materialized_sha,
    }
    entry_payload = {
        "schema_version": "stage7_measurement_cache_entry_v1",
        "cache_key_sha256": build_measurement_cache_key(dimensions),
        "origin": {"result_root": "/frozen/origin", "trajectory": "S5-ORIGIN"},
        "result": result,
    }
    return {**entry_payload, "cache_entry_sha256": _sha(entry_payload)}


def _cache_fixture(
    tmp_path: Path, hit_count: int
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    request = _request()
    binding = _binding(request)
    dimensions = _dimensions(request)
    entries = {}
    for row in request["rows"][:hit_count]:
        entry = _cache_entry(tmp_path, row, dimensions[row["row_id"]])
        entries[entry["cache_key_sha256"]] = entry
    return request, binding, dimensions, entries


def _miss_result(request: dict[str, Any], row_id: str) -> dict[str, Any]:
    request_row = next(row for row in request["rows"] if row["row_id"] == row_id)
    return {
        **copy.deepcopy(request_row),
        "measurement_request_row_sha256": request["row_sha256"][row_id],
        "terminal_status": "measured_success_gold",
        "latency_ms": 3.0,
        "energy_j": 4.0,
        "ap30": 0.75,
        "ap50": 0.65,
        "ap70": 0.55,
        "latency_artifact_sha256": "5" * 64,
        "energy_artifact_sha256": "6" * 64,
        "ap_artifact_sha256": "7" * 64,
    }


def _promotion_result(
    request: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    requested = {row["row_id"]: row for row in request["rows"]}
    promoted_rows = []
    audit_rows = []
    for feedback in rows:
        row_id = feedback["row_id"]
        request_row = requested[row_id]
        candidate = copy.deepcopy(request_row["graph_features"])
        actual = {
            "schema": "stage5_actual_graph_features_v1",
            "group_id": request_row["group_id"],
            "model": request_row["model"],
            "width": copy.deepcopy(request_row["width"]),
            "onnx_sha256": feedback.get("onnx_sha256", "8" * 64),
            "operator_count": 17,
            "graph_feature_provenance": "materialized_onnx_extracted_v1",
        }
        promoted = {
            **copy.deepcopy(feedback),
            "candidate_graph_features": candidate,
            "candidate_graph_features_sha256": _sha(candidate),
            "graph_features": actual,
            "materialized_graph_features_sha256": _sha(actual),
            "historical_feedback_row_sha256": _sha(feedback),
            "feedback_feature_contract": "actual_feedback_v3",
            "graph_feature_promotion_schema": "stage5_actual_feedback_promotion_v3",
        }
        promoted["actual_feedback_row_sha256"] = _sha(promoted)
        promoted_rows.append(promoted)
        audit_rows.append(
            {
                "manifest_job_id": row_id,
                "group_id": request_row["group_id"],
                "candidate_graph_features_sha256": promoted[
                    "candidate_graph_features_sha256"
                ],
                "materialized_graph_features_sha256": promoted[
                    "materialized_graph_features_sha256"
                ],
                "actual_feedback_row_sha256": promoted[
                    "actual_feedback_row_sha256"
                ],
            }
        )
    return {
        "rows": promoted_rows,
        "audit": {
            "schema_version": "stage5_actual_feedback_batch_audit_v3",
            "task_id": request["task_id"],
            "round_index": request["round_index"],
            "promoted_row_count": 4,
            "actual_group_count": len(
                {row["group_id"] for row in request["rows"]}
            ),
            "silent_surrogate_fallback_count": 0,
            "rows": audit_rows,
        },
    }


def _reveal(
    request: dict[str, Any],
    binding: dict[str, Any],
    dimensions: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return cache_feedback.reveal_selected_batch_cache(
        request,
        binding,
        trajectory_contract_sha256="a" * 64,
        key_dimensions_by_row=dimensions,
        cache=cache,
    )


def test_reveal_validates_persisted_request_binding_before_examining_cache() -> None:
    request = _request()
    binding = _binding(request)
    binding["measurement_request_sha256"] = "0" * 64

    class ExplodingCache(dict[str, dict[str, Any]]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError(f"cache examined before binding validation: {key}")

    with pytest.raises(ValueError, match="binding"):
        cache_feedback.reveal_selected_batch_cache(
            request,
            binding,
            trajectory_contract_sha256="a" * 64,
            key_dimensions_by_row=_dimensions(request),
            cache=ExplodingCache(),
        )


@pytest.mark.parametrize(
    "binding_sha",
    (None, "not-a-sha", "0" * 64),
)
def test_binding_self_sha_is_required_before_cache_lookup(
    binding_sha: str | None,
) -> None:
    request = _request()
    binding = _binding(request)
    if binding_sha is None:
        del binding["request_binding_sha256"]
    else:
        binding["request_binding_sha256"] = binding_sha

    class ExplodingCache(dict[str, dict[str, Any]]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError(f"cache examined before binding validation: {key}")

    with pytest.raises(ValueError, match="binding.*SHA"):
        cache_feedback.reveal_selected_batch_cache(
            request,
            binding,
            trajectory_contract_sha256="a" * 64,
            key_dimensions_by_row=_dimensions(request),
            cache=ExplodingCache(),
        )


def test_every_exact_key_dimension_changes_the_key_and_near_keys_miss(
    tmp_path: Path,
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 1)
    row_id = request["rows"][0]["row_id"]
    exact = dimensions[row_id]
    assert set(exact) == set(CACHE_KEY_DIMENSIONS)
    assert _reveal(request, binding, dimensions, entries)["rows"][0][
        "disposition"
    ] == "exact_hit"

    for field in CACHE_KEY_DIMENSIONS:
        changed = copy.deepcopy(exact)
        value = changed[field]
        if field.endswith("_sha256"):
            changed[field] = "0" * 64
        elif isinstance(value, int):
            changed[field] = value + 1
        elif isinstance(value, list):
            changed[field] = [*value, "changed"]
        else:
            changed[field] = f"{value}-changed"
        if field == "q_mode":
            changed[field] = "int8"
        near_dimensions = {**dimensions, row_id: changed}
        assert _reveal(request, binding, near_dimensions, entries)["rows"][0][
            "disposition"
        ] == "miss"

        missing = copy.deepcopy(exact)
        del missing[field]
        missing_dimensions = {**dimensions, row_id: missing}
        assert _reveal(request, binding, missing_dimensions, entries)["rows"][0][
            "disposition"
        ] == "miss"


def test_unusable_exact_hit_is_rejected_as_a_miss(tmp_path: Path) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 1)
    entry = next(iter(entries.values()))
    Path(entry["result"]["ap_artifact_path"]).unlink()

    reveal = _reveal(request, binding, dimensions, entries)

    assert reveal["rows"][0]["disposition"] == "miss"
    assert reveal["rows"][0]["reason"] == "cache_entry_unusable_missing_evidence"
    assert "bound_result" not in reveal["rows"][0]


@pytest.mark.parametrize("evidence_kind", ("terminal", "materialized"))
def test_exact_hit_rejects_semantically_drifted_terminal_or_materialization_evidence(
    tmp_path: Path, evidence_kind: str
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 1)
    entry = next(iter(entries.values()))
    result = entry["result"]
    if evidence_kind == "terminal":
        path_field = "terminal_evidence_path"
        sha_field = "terminal_evidence_sha256"
        payload = {
            "schema_version": "stage5_terminal_measurement_evidence_v1",
            "row_id": result["row_id"],
            "terminal_status": "measured_success_gold",
            "latency_ms": 999.0,
            "energy_j": result["energy_j"],
            "ap30": result["ap30"],
            "ap50": result["ap50"],
            "ap70": result["ap70"],
        }
    else:
        path_field = "materialized_source_evidence_path"
        sha_field = "materialized_source_evidence_sha256"
        payload = {
            "schema_version": "stage5_source_materialization_evidence_v1",
            "status": "ready",
            "group_id": request["rows"][0]["group_id"],
            "model": request["rows"][0]["model"],
            "width": "x".join(
                str(value) for value in request["rows"][0]["width"]
            ),
            "source_plan_sha256": "0" * 64,
            "checkpoint_sha256": result["source_checkpoint_sha256"],
            "onnx_path": result["onnx_path"],
            "onnx_sha256": result["onnx_sha256"],
        }
    path = Path(result[path_field])
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    result[sha_field] = hashlib.sha256(path.read_bytes()).hexdigest()
    entry["cache_entry_sha256"] = _sha(
        {key: value for key, value in entry.items() if key != "cache_entry_sha256"}
    )

    reveal = _reveal(request, binding, dimensions, entries)

    assert reveal["rows"][0]["disposition"] == "miss"
    assert reveal["rows"][0]["reason"] == "cache_entry_unusable_invalid_evidence"


@pytest.mark.parametrize("drift", ("width", "onnx_path"))
def test_exact_hit_rejects_stage5_materialization_width_or_onnx_path_drift(
    tmp_path: Path, drift: str
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 1)
    entry = next(iter(entries.values()))
    result = entry["result"]
    evidence_path = Path(result["materialized_source_evidence_path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if drift == "width":
        evidence["width"] = "8x8x8"
    else:
        other_onnx, _ = _file(tmp_path / "other.onnx", b"other-onnx")
        evidence["onnx_path"] = other_onnx
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True), encoding="utf-8"
    )
    result["materialized_source_evidence_sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    entry["cache_entry_sha256"] = _sha(
        {key: value for key, value in entry.items() if key != "cache_entry_sha256"}
    )

    reveal = _reveal(request, binding, dimensions, entries)

    assert reveal["rows"][0]["disposition"] == "miss"
    assert reveal["rows"][0]["reason"] == "cache_entry_unusable_invalid_evidence"


def test_cached_result_rebinds_current_s7_identity_and_retains_origin(
    tmp_path: Path,
) -> None:
    request, _, dimensions, entries = _cache_fixture(tmp_path, 1)
    request_row = request["rows"][0]
    entry = next(iter(entries.values()))
    original_entry = copy.deepcopy(entry)

    bound = cache_feedback.bind_cached_result_to_request(
        request,
        request_row,
        dimensions[request_row["row_id"]],
        entry,
        trajectory_contract_sha256="a" * 64,
    )

    assert bound["task_id"] == "S7-PYR-TVM"
    assert bound["task_sha256"] == request["task_sha256"]
    assert bound["round_index"] == request["round_index"]
    assert bound["row_id"] == request_row["row_id"]
    assert bound["measurement_request_row_sha256"] == request["row_sha256"][
        request_row["row_id"]
    ]
    assert bound["cache_provenance"] == {
        "origin_row_id": f"origin-{request_row['row_id']}",
        "source_cache_entry_sha256": entry["cache_entry_sha256"],
        "origin": entry["origin"],
        "origin_terminal_status": "measured_success_gold",
        "trajectory_contract_sha256": "a" * 64,
    }
    bound["cache_provenance"]["origin"]["trajectory"] = "mutated"
    assert entry == original_entry


@pytest.mark.parametrize("hit_count", range(5))
def test_zero_through_four_hits_merge_to_exactly_four_rows_in_request_order(
    tmp_path: Path, hit_count: int
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, hit_count)
    reveal = _reveal(request, binding, dimensions, entries)
    misses = [
        _miss_result(request, row["row_id"])
        for row in request["rows"][hit_count:]
    ]
    calls = []

    def promote(request_value: dict[str, Any], rows: list[dict[str, Any]]) -> dict:
        calls.append(("promote", [row["row_id"] for row in rows]))
        return _promotion_result(request_value, rows)

    def finalize(request_value: dict[str, Any], rows: list[dict[str, Any]]) -> dict:
        calls.append(("finalize", [row["row_id"] for row in rows]))
        return {
            "schema_version": "stage5_atomic_batch_audit_v2",
            "feedback_released": True,
            "batch_quarantined": False,
            "budget_consumed": 4,
            "released_feedback_rows": copy.deepcopy(rows),
        }

    result = cache_feedback.finalize_stage7_atomic_batch(
        request,
        binding,
        reveal,
        misses,
        promote_feedback_batch=promote,
        finalize_atomic_batch=finalize,
    )

    expected_ids = [row["row_id"] for row in request["rows"]]
    assert [row["row_id"] for row in result["released_feedback_rows"]] == expected_ids
    assert calls == [("promote", expected_ids), ("finalize", expected_ids)]
    assert result["exact_hit_count"] == hit_count
    assert result["miss_count"] == 4 - hit_count


def test_all_hit_has_no_gpu_runner_surface_and_needs_no_miss_results(
    tmp_path: Path,
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 4)
    reveal = _reveal(request, binding, dimensions, entries)
    assert "gpu_runner" not in inspect.signature(
        cache_feedback.finalize_stage7_atomic_batch
    ).parameters

    result = cache_feedback.finalize_stage7_atomic_batch(
        request,
        binding,
        reveal,
        [],
        promote_feedback_batch=_promotion_result,
        finalize_atomic_batch=lambda _, rows: {
            "feedback_released": True,
            "batch_quarantined": False,
            "budget_consumed": 4,
            "released_feedback_rows": rows,
        },
    )

    assert result["exact_hit_count"] == 4
    assert result["miss_count"] == 0


def test_miss_only_plan_filters_jobs_and_binds_original_lineage(
    tmp_path: Path,
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 2)
    reveal = _reveal(request, binding, dimensions, entries)
    jobs = [{**copy.deepcopy(row), "command": ["measure", row["row_id"]]} for row in request["rows"]]
    performance_jobs = [
        {"manifest_job_id": row["row_id"], "command": ["runner", row["row_id"]]}
        for row in request["rows"]
    ]
    full_plan = {
        "manifest": {
            "schema_version": "stage5_performance_manifest_v2",
            "source_request_sha256": request["measurement_request_sha256"],
            "task_id": request["task_id"],
            "task_sha256": request["task_sha256"],
            "genome_count": 4,
            "row_count": 4,
            "group_count": 4,
            "group_ids": [row["group_id"] for row in request["rows"]],
            "jobs": jobs,
        },
        "performance_jobs": performance_jobs,
    }
    original = copy.deepcopy(full_plan)

    miss_plan = cache_feedback.derive_miss_only_plan(
        full_plan, request, binding, reveal
    )

    expected_ids = ["row-2", "row-3"]
    assert [row["row_id"] for row in miss_plan["manifest"]["jobs"]] == expected_ids
    assert [
        row["manifest_job_id"] for row in miss_plan["performance_jobs"]
    ] == expected_ids
    assert miss_plan["manifest"]["row_count"] == 2
    assert miss_plan["manifest"]["genome_count"] == 2
    assert miss_plan["manifest"]["group_count"] == 2
    assert miss_plan["original_measurement_request_sha256"] == request[
        "measurement_request_sha256"
    ]
    assert miss_plan["request_binding_sha256"] == binding[
        "request_binding_sha256"
    ]
    assert set(miss_plan["original_request_row_sha256"]) == set(expected_ids)
    assert miss_plan["miss_plan_sha256"] == _sha(
        {key: value for key, value in miss_plan.items() if key != "miss_plan_sha256"}
    )
    assert full_plan == original


def test_infrastructure_failure_retries_same_request_without_budget_or_callbacks(
    tmp_path: Path,
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 3)
    reveal = _reveal(request, binding, dimensions, entries)
    miss = {
        **request["rows"][3],
        "measurement_request_row_sha256": request["row_sha256"]["row-3"],
        "terminal_status": "public_runner_failure",
        "failure_kind": "gpu_unavailable",
        "failure_reason": "lease contention",
    }
    calls = []

    result = cache_feedback.finalize_stage7_atomic_batch(
        request,
        binding,
        reveal,
        [miss],
        promote_feedback_batch=lambda *_: calls.append("promote"),
        finalize_atomic_batch=lambda *_: calls.append("finalize"),
    )

    assert result["feedback_released"] is False
    assert result["same_request_retry"] is True
    assert result["budget_consumed"] == 0
    assert result["measurement_request_sha256"] == request[
        "measurement_request_sha256"
    ]
    assert result["request"] == request
    assert calls == []


def test_true_candidate_failure_consumes_event_without_fabricated_metrics(
    tmp_path: Path,
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 3)
    reveal = _reveal(request, binding, dimensions, entries)
    failure = {
        **request["rows"][3],
        "measurement_request_row_sha256": request["row_sha256"]["row-3"],
        "terminal_status": "feasibility_failure",
        "failure_kind": "build_failure",
        "failure_reason": "compiler rejected candidate",
    }

    result = cache_feedback.finalize_stage7_atomic_batch(
        request,
        binding,
        reveal,
        [failure],
        promote_feedback_batch=_promotion_result,
        finalize_atomic_batch=lambda _, rows: {
            "feedback_released": True,
            "batch_quarantined": False,
            "budget_consumed": 4,
            "released_feedback_rows": rows,
        },
    )

    released_failure = result["released_feedback_rows"][3]
    assert result["budget_consumed"] == 4
    assert released_failure["terminal_status"] == "feasibility_failure"
    assert not {
        "latency_ms",
        "energy_j",
        "ap30",
        "ap50",
        "ap70",
    } & set(released_failure)


def test_invalid_merge_rejects_duplicates_and_drift_before_callbacks(
    tmp_path: Path,
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 2)
    reveal = _reveal(request, binding, dimensions, entries)
    duplicate = _miss_result(request, "row-2")
    calls = []

    with pytest.raises(ValueError, match="duplicate|omission"):
        cache_feedback.finalize_stage7_atomic_batch(
            request,
            binding,
            reveal,
            [duplicate, copy.deepcopy(duplicate)],
            promote_feedback_batch=lambda *_: calls.append("promote"),
            finalize_atomic_batch=lambda *_: calls.append("finalize"),
        )
    assert calls == []

    drifted = _miss_result(request, "row-3")
    drifted["task_id"] = "S7-OTHER"
    with pytest.raises(ValueError, match="identity"):
        cache_feedback.finalize_stage7_atomic_batch(
            request,
            binding,
            reveal,
            [_miss_result(request, "row-2"), drifted],
            promote_feedback_batch=lambda *_: calls.append("promote"),
            finalize_atomic_batch=lambda *_: calls.append("finalize"),
        )
    assert calls == []


def test_request_row_sha_and_reveal_sha_drift_are_rejected(
    tmp_path: Path,
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 4)
    reveal = _reveal(request, binding, dimensions, entries)
    reveal["cache_reveal_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="reveal SHA"):
        cache_feedback.finalize_stage7_atomic_batch(
            request,
            binding,
            reveal,
            [],
            promote_feedback_batch=_promotion_result,
            finalize_atomic_batch=lambda _, rows: {"released_feedback_rows": rows},
        )

    drifted_request = copy.deepcopy(request)
    drifted_request["rows"][0]["q_mode"] = "int8"
    with pytest.raises(ValueError, match="request"):
        cache_feedback.reveal_selected_batch_cache(
            drifted_request,
            binding,
            trajectory_contract_sha256="a" * 64,
            key_dimensions_by_row=dimensions,
            cache=entries,
        )


@pytest.mark.parametrize(
    "mode",
    ("missing_audit", "surrogate_noop", "zero_audit_surrogate_graph"),
)
def test_noop_or_silent_surrogate_promotion_is_rejected_before_finalizer(
    tmp_path: Path, mode: str
) -> None:
    request, binding, dimensions, entries = _cache_fixture(tmp_path, 4)
    reveal = _reveal(request, binding, dimensions, entries)
    finalizer_calls = []

    def invalid_promotion(
        request_value: dict[str, Any], rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if mode == "missing_audit":
            return {"rows": copy.deepcopy(rows)}
        result = _promotion_result(request_value, rows)
        if mode == "surrogate_noop":
            result["audit"]["silent_surrogate_fallback_count"] = 4
        for promoted, requested in zip(result["rows"], request_value["rows"]):
            promoted["graph_features"] = copy.deepcopy(
                requested["graph_features"]
            )
            promoted["materialized_graph_features_sha256"] = _sha(
                promoted["graph_features"]
            )
            promoted["actual_feedback_row_sha256"] = _sha(
                {
                    key: value
                    for key, value in promoted.items()
                    if key != "actual_feedback_row_sha256"
                }
            )
        for audit_row, promoted in zip(
            result["audit"]["rows"], result["rows"]
        ):
            audit_row["materialized_graph_features_sha256"] = promoted[
                "materialized_graph_features_sha256"
            ]
            audit_row["actual_feedback_row_sha256"] = promoted[
                "actual_feedback_row_sha256"
            ]
        return result

    with pytest.raises(ValueError, match="promotion"):
        cache_feedback.finalize_stage7_atomic_batch(
            request,
            binding,
            reveal,
            [],
            promote_feedback_batch=invalid_promotion,
            finalize_atomic_batch=lambda *_: finalizer_calls.append("finalize"),
        )

    assert finalizer_calls == []


def test_real_stage5_promotion_adapter_then_atomic_finalizer(
    tmp_path: Path,
) -> None:
    from framework.stage5.single_target_search_v2 import (
        finalize_atomic_batch as real_atomic_finalizer,
    )
    from scripts.stage5_promote_actual_feedback_v3 import (
        promote_feedback_batch as real_promoter,
    )

    request, binding, dimensions, entries = _cache_fixture(tmp_path, 4)
    reveal = _reveal(request, binding, dimensions, entries)
    request_path = tmp_path / "request.json"
    feedback_path = tmp_path / "feedback.json"

    def adapter(
        request_value: dict[str, Any], rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        request_path.write_text(
            json.dumps(request_value, sort_keys=True), encoding="utf-8"
        )
        feedback_path.write_text(
            json.dumps({"rows": rows}, sort_keys=True), encoding="utf-8"
        )

        def extractor(onnx_path: Path) -> dict[str, Any]:
            return {
                "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
                "operator_count": 23,
            }

        return real_promoter(
            request_path, feedback_path, extractor=extractor
        )

    result = cache_feedback.finalize_stage7_atomic_batch(
        request,
        binding,
        reveal,
        [],
        promote_feedback_batch=adapter,
        finalize_atomic_batch=real_atomic_finalizer,
    )

    assert result["feedback_released"] is True
    assert result["budget_consumed"] == 4
    assert all(
        row["graph_features"]["graph_feature_provenance"]
        == "materialized_onnx_extracted_v1"
        for row in result["released_feedback_rows"]
    )
