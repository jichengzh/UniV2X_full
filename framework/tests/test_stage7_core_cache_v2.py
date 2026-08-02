from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from framework.stage7 import core_cache_v2 as cache_v2


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _dimensions(candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "model": "pyramid",
        "capability_profile_id": "h800-tvm-auto-v1",
        "hardware_id": "h800",
        "dispatch_key": "tvm_auto",
        "measurement_scope": "formal_h800_latency_energy_ap",
        "input_protocol_sha256": "1" * 64,
        "batch_size": 1,
        "genome": [candidate_id, 32, 64, "fp16"],
        "q_mode": "fp16",
        "source_checkpoint_sha256": "2" * 64,
        "onnx_sha256": "3" * 64,
        "build_protocol_sha256": "4" * 64,
        "tuning_protocol_sha256": "5" * 64,
        "measurement_protocol_sha256": "6" * 64,
        "ap_protocol_sha256": "7" * 64,
        "runtime_contract_sha256": "8" * 64,
    }


def _request(candidate_ids: list[str] | None = None) -> dict[str, Any]:
    candidate_ids = candidate_ids or ["candidate-a"]
    rows = [
        {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": candidate_id,
            "manifest_job_id": candidate_id,
        }
        for candidate_id in candidate_ids
    ]
    payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "a" * 64,
        "round_index": 0,
        "batch_size": len(rows),
        "sample_budget": 16,
        "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "row_sha256": {row["row_id"]: _sha(row) for row in rows},
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _sha(payload)}


def _binding(
    candidate_ids: list[str],
    *,
    frozen: bool,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = request or _request(candidate_ids)
    selections = [
        (
            lambda cross_payload: {
                **cross_payload,
                "logical_exact_binding_sha256": _sha(cross_payload),
            }
        )(
            {
            "candidate_id": candidate_id,
            "logical_row_sha256": request["row_sha256"][candidate_id],
            "exact_key_dimensions": _dimensions(candidate_id),
            "exact_cache_key_sha256": cache_v2.build_v2_exact_cache_key(
                _dimensions(candidate_id)
            ),
            }
        )
        for candidate_id in candidate_ids
    ]
    payload = {
        "schema_version": cache_v2.SELECTION_BINDING_SCHEMA,
        "selection_frozen": frozen,
        "logical_request_sha256": request["measurement_request_sha256"],
        "selected_candidates": selections,
        "selected_candidate_ids_sha256": _sha(candidate_ids),
    }
    return {**payload, "selection_binding_sha256": _sha(payload)}


def _terminal_success(candidate_id: str = "candidate-a") -> dict[str, Any]:
    dimensions = _dimensions(candidate_id)
    key = cache_v2.build_v2_exact_cache_key(dimensions)
    payload = {
        "schema_version": "stage7_v2_terminal_evidence_v1",
        "created_by": "stage7_core_cache_v2",
        "terminal_status": "measured_success",
        "candidate_id": candidate_id,
        "exact_key_dimensions": dimensions,
        "exact_cache_key_sha256": key,
        "request_sha256": _request()["measurement_request_sha256"],
        "latency_ms": 1.0,
        "energy_j": 2.0,
        "ap30": 0.8,
        "ap50": 0.7,
        "ap70": 0.6,
    }
    return {**payload, "terminal_evidence_sha256": _sha(payload)}


def _cache_with_exact(candidate_id: str = "candidate-a") -> dict[str, Any]:
    evidence = _terminal_success(candidate_id)
    return {
        "schema_version": "stage7_core_cache_v2",
        "entries": {evidence["exact_cache_key_sha256"]: evidence},
        "lineage": [],
    }


def test_cache_membership_cannot_be_revealed_before_ordered_selection_is_frozen() -> None:
    class ExplodingCache(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError(f"cache membership inspected before freeze: {key}")

    cache = ExplodingCache(_cache_with_exact())
    request = _request()
    with pytest.raises(ValueError, match="selection binding must be frozen"):
        cache_v2.reveal_v2_cache_after_selection(
            request,
            _binding(["candidate-a"], frozen=False, request=request),
            cache,
        )


def test_formal_contract_shape_gates_fail_closed_before_cache_access() -> None:
    with pytest.raises(ValueError, match="logical request must be a mapping"):
        cache_v2.validate_logical_request(None)  # type: ignore[arg-type]

    malformed_request = _request()
    malformed_request["rows"] = None
    malformed_payload = {
        key: value
        for key, value in malformed_request.items()
        if key != "measurement_request_sha256"
    }
    malformed_request["measurement_request_sha256"] = _sha(malformed_payload)
    with pytest.raises(ValueError, match="logical request rows are invalid"):
        cache_v2.validate_logical_request(malformed_request)

    request = _request()
    binding = _binding(["candidate-a"], frozen=True, request=request)
    binding["schema_version"] = "legacy-selection"
    binding_payload = {
        key: value
        for key, value in binding.items()
        if key != "selection_binding_sha256"
    }
    binding["selection_binding_sha256"] = _sha(binding_payload)
    with pytest.raises(ValueError, match="unexpected selection binding schema"):
        cache_v2.validate_selection_binding(request, binding)

    with pytest.raises(ValueError, match="unexpected v2 cache schema"):
        cache_v2.validate_actual_v3_cache(
            {"schema_version": "legacy-cache", "entries": {}, "lineage": []}
        )


def test_formal_reveal_rejects_legacy_exact_hit() -> None:
    request = _request()
    with pytest.raises(ValueError, match="formal actual-v3 terminal schema"):
        cache_v2.reveal_v2_cache_after_selection(
            request,
            _binding(["candidate-a"], frozen=True, request=request),
            _cache_with_exact(),
        )


def test_selected_miss_still_consumes_one_event_and_requires_measurement() -> None:
    request = _request()
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request,
        _binding(["candidate-a"], frozen=True, request=request),
        {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []},
    )

    assert reveal["entries"][0]["disposition"] == "miss"
    assert reveal["entries"][0]["selected_event_budget_delta"] == 1
    assert reveal["entries"][0]["hardware_measurement_required"] is True


def test_binding_candidate_identity_mismatch_fails_before_cache_lookup() -> None:
    request = _request(["candidate-b"])
    binding = _binding(["candidate-b"], frozen=True, request=request)
    binding["selected_candidates"][0]["exact_key_dimensions"] = _dimensions(
        "candidate-a"
    )
    binding["selection_binding_sha256"] = _sha(
        {key: value for key, value in binding.items() if key != "selection_binding_sha256"}
    )

    class ExplodingCache(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError(f"cache lookup must follow candidate binding: {key}")

    with pytest.raises(ValueError, match="candidate identity mismatch"):
        cache_v2.reveal_v2_cache_after_selection(
            request, binding, ExplodingCache(_cache_with_exact())
        )


@pytest.mark.parametrize(
    "reason",
    (
        "backend_capability_failure",
        "build_failure",
        "quantization_failure",
        "numerical_failure",
    ),
)
def test_candidate_failure_consumes_slot_with_null_objectives(reason: str) -> None:
    terminal = cache_v2.finalize_candidate_failure(
        _request(), reason=reason
    )

    assert terminal["consumes_selected_event_budget"] is True
    assert terminal["selected_event_budget_delta"] == 1
    assert terminal["latency_ms"] is None
    assert terminal["energy_j"] is None
    assert terminal["ap30"] is None
    assert terminal["ap50"] is None
    assert terminal["ap70"] is None


def test_candidate_failure_rejects_an_infrastructure_reason() -> None:
    with pytest.raises(ValueError, match="candidate failure reason"):
        cache_v2.finalize_candidate_failure(
            _request(), reason="gpu_occupancy_drift"
        )


def test_failure_finalizers_reject_legacy_self_hashed_requests() -> None:
    legacy = _request()
    legacy["schema_version"] = "stage7_v2_selected_request_v1"
    legacy["measurement_request_sha256"] = _sha(
        {
            key: value
            for key, value in legacy.items()
            if key != "measurement_request_sha256"
        }
    )

    with pytest.raises(ValueError, match="unexpected logical request schema"):
        cache_v2.finalize_infrastructure_failure(
            legacy, reason="gpu_occupancy_drift"
        )


def test_failure_finalizers_reject_self_hashed_unknown_request_fields() -> None:
    request = _request()
    request["legacy_cache_wrapper"] = "must-not-cross-v2-boundary"
    request["measurement_request_sha256"] = _sha(
        {
            key: value
            for key, value in request.items()
            if key != "measurement_request_sha256"
        }
    )

    with pytest.raises(ValueError, match="logical request"):
        cache_v2.finalize_evidence_failure(
            request, reason="terminal_artifact_sha_mismatch"
        )


def test_infrastructure_failure_retries_same_request_without_consuming_slot() -> None:
    retry = cache_v2.finalize_infrastructure_failure(
        _request(), reason="gpu_occupancy_drift"
    )

    assert retry["consumes_selected_event_budget"] is False
    assert retry["selected_event_budget_delta"] == 0
    assert retry["retry_request_sha256"] == retry["request_sha256"]


def test_evidence_failure_retries_same_request_without_consuming_slot() -> None:
    retry = cache_v2.finalize_evidence_failure(
        _request(), reason="terminal_artifact_sha_mismatch"
    )

    assert retry["consumes_selected_event_budget"] is False
    assert retry["selected_event_budget_delta"] == 0
    assert retry["retry_request_sha256"] == _request()[
        "measurement_request_sha256"
    ]


@pytest.mark.parametrize(
    ("finalizer", "reason"),
    (
        (cache_v2.finalize_candidate_failure, "gpu_occupancy_drift"),
        (cache_v2.finalize_candidate_failure, "terminal_artifact_sha_mismatch"),
        (cache_v2.finalize_infrastructure_failure, "build_failure"),
        (cache_v2.finalize_infrastructure_failure, "terminal_artifact_sha_mismatch"),
        (cache_v2.finalize_evidence_failure, "numerical_failure"),
        (cache_v2.finalize_evidence_failure, "gpu_unavailable"),
    ),
)
def test_failure_finalizers_reject_reasons_owned_by_other_classes(
    finalizer: Any, reason: str
) -> None:
    with pytest.raises(ValueError, match="failure reason"):
        finalizer(_request(), reason=reason)


def test_append_creates_immutable_cache_lineage_for_complete_v2_evidence() -> None:
    original = {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []}
    evidence = _terminal_success()

    appended = cache_v2.append_v2_terminal_evidence(original, evidence)

    assert original == {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []}
    assert appended is not original
    assert appended["entries"] is not original["entries"]
    assert appended["entries"][evidence["exact_cache_key_sha256"]] == evidence
    assert appended["lineage"][-1]["action"] == "append_terminal_evidence"
    assert appended["lineage"][-1]["evidence_sha256"] == evidence["terminal_evidence_sha256"]


def test_append_is_idempotent_for_identical_evidence_but_fails_closed_on_conflict() -> None:
    evidence = _terminal_success()
    cache = cache_v2.append_v2_terminal_evidence(
        {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []}, evidence
    )

    assert cache_v2.append_v2_terminal_evidence(cache, evidence) == cache
    conflicting = copy.deepcopy(evidence)
    conflicting["latency_ms"] = 99.0
    conflicting["terminal_evidence_sha256"] = _sha(
        {key: value for key, value in conflicting.items() if key != "terminal_evidence_sha256"}
    )
    with pytest.raises(ValueError, match="conflicting exact-cache evidence"):
        cache_v2.append_v2_terminal_evidence(cache, conflicting)


def test_append_rejects_historical_wrapper_missing_protocol_sha() -> None:
    historical = _terminal_success()
    historical["schema_version"] = "stage7_measurement_cache_entry_v1"
    historical["created_by"] = "historical_wrapper"
    del historical["exact_key_dimensions"]["ap_protocol_sha256"]
    historical["terminal_evidence_sha256"] = _sha(
        {key: value for key, value in historical.items() if key != "terminal_evidence_sha256"}
    )

    with pytest.raises(ValueError, match="protocol SHA"):
        cache_v2.append_v2_terminal_evidence(
            {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []},
            historical,
        )


@pytest.mark.parametrize("container", ("binding", "evidence"))
def test_exact_contract_rejects_self_hashed_unknown_dimensions(
    container: str,
) -> None:
    if container == "binding":
        request = _request()
        binding = _binding(["candidate-a"], frozen=True, request=request)
        binding["selected_candidates"][0]["exact_key_dimensions"]["legacy_field"] = "x"
        binding["selection_binding_sha256"] = _sha(
            {key: value for key, value in binding.items() if key != "selection_binding_sha256"}
        )
        with pytest.raises(ValueError, match="exact cache key dimensions shape"):
            cache_v2.reveal_v2_cache_after_selection(
                request, binding, _cache_with_exact()
            )
        return

    evidence = _terminal_success()
    evidence["exact_key_dimensions"]["legacy_field"] = "x"
    evidence["terminal_evidence_sha256"] = _sha(
        {key: value for key, value in evidence.items() if key != "terminal_evidence_sha256"}
    )
    with pytest.raises(ValueError, match="exact cache key dimensions shape"):
        cache_v2.append_v2_terminal_evidence(
            {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []},
            evidence,
        )


def test_terminal_evidence_rejects_self_hashed_unknown_fields() -> None:
    evidence = _terminal_success()
    evidence["legacy_wrapper"] = "v1"
    evidence["terminal_evidence_sha256"] = _sha(
        {key: value for key, value in evidence.items() if key != "terminal_evidence_sha256"}
    )

    with pytest.raises(ValueError, match="terminal evidence shape"):
        cache_v2.append_v2_terminal_evidence(
            {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []},
            evidence,
        )


@pytest.mark.parametrize("defect", ("historical", "sha_drift", "non_finite"))
def test_existing_exact_key_with_invalid_evidence_fails_closed(defect: str) -> None:
    cache = _cache_with_exact()
    key = next(iter(cache["entries"]))
    evidence = cache["entries"][key]
    if defect == "historical":
        evidence["schema_version"] = "stage7_measurement_cache_entry_v1"
    elif defect == "sha_drift":
        evidence["terminal_evidence_sha256"] = "0" * 64
    else:
        evidence["ap70"] = float("nan")
        evidence["terminal_evidence_sha256"] = _sha(
            {key: value for key, value in evidence.items() if key != "terminal_evidence_sha256"}
        )

    with pytest.raises(ValueError):
        request = _request()
        cache_v2.reveal_v2_cache_after_selection(
            request,
            _binding(["candidate-a"], frozen=True, request=request),
            cache,
        )


def test_existing_exact_key_with_none_evidence_fails_before_hardware_miss() -> None:
    request = _request()
    binding = _binding(["candidate-a"], frozen=True, request=request)
    key = cache_v2.build_v2_exact_cache_key(
        binding["selected_candidates"][0]["exact_key_dimensions"]
    )
    cache = {
        "schema_version": "stage7_core_cache_v2",
        "entries": {key: None},
        "lineage": [],
    }

    with pytest.raises(ValueError, match="present exact-cache evidence is invalid"):
        cache_v2.reveal_v2_cache_after_selection(request, binding, cache)


def test_append_rejects_existing_exact_key_with_none_evidence() -> None:
    evidence = _terminal_success()
    cache = {
        "schema_version": "stage7_core_cache_v2",
        "entries": {evidence["exact_cache_key_sha256"]: None},
        "lineage": [],
    }

    with pytest.raises(ValueError, match="present exact-cache evidence is invalid"):
        cache_v2.append_v2_terminal_evidence(cache, evidence)


@pytest.mark.parametrize("metric", ("latency_ms", "energy_j", "ap30", "ap50", "ap70"))
@pytest.mark.parametrize("non_finite", (float("nan"), float("inf"), float("-inf")))
def test_append_rejects_non_finite_success_metrics(metric: str, non_finite: float) -> None:
    evidence = _terminal_success()
    evidence[metric] = non_finite
    evidence["terminal_evidence_sha256"] = _sha(
        {key: value for key, value in evidence.items() if key != "terminal_evidence_sha256"}
    )

    with pytest.raises(ValueError, match="successful terminal evidence metric is invalid"):
        cache_v2.append_v2_terminal_evidence(
            {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []},
            evidence,
        )
