"""Terminal feedback/cache commit operations for Stage7 core rounds."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage5 import single_target_search_v2 as stage5_search
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import core_cache_v2 as cache_v2
from framework.stage7 import physical_feedback_v2 as physical_feedback
from framework.stage7 import source_resolution_v2 as source_resolution
from framework.stage7.core_ablation_v2 import canonical_sha256


JSON = dict[str, Any]
BARRIER_SCHEMA = "stage7_actual_v3_atomic_feedback_barrier_v2"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _read_mapping(path: Path) -> JSON:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return copy.deepcopy(dict(payload))


def feedback_for_reveal(
    reveal: Mapping[str, Any], physical_rows: Sequence[Mapping[str, Any]]
) -> list[JSON]:
    physical = {
        str(row.get("row_id") or row.get("manifest_job_id") or ""): copy.deepcopy(
            dict(row)
        )
        for row in physical_rows
    }
    rows: list[JSON] = []
    for entry in reveal["entries"]:
        candidate_id = str(entry["candidate_id"])
        if entry["disposition"] == "miss":
            if candidate_id not in physical:
                raise ValueError("physical terminal feedback is incomplete")
            rows.append(physical.pop(candidate_id))
            continue
        terminal = entry["terminal_evidence"]
        artifact = terminal["stage5_terminal_artifact"]
        path = Path(str(artifact["path"]))
        if _file_sha256(path) != artifact["artifact_sha256"]:
            raise ValueError("cache-hit Stage5 terminal artifact SHA drift")
        rows.append(_read_mapping(path))
    if physical:
        raise ValueError("unexpected physical terminal feedback row")
    return rows


def _read_stage5_terminal_artifact(wrapper: Mapping[str, Any]) -> JSON:
    artifact = wrapper.get("stage5_terminal_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("terminal wrapper Stage5 artifact is missing")
    path = Path(str(artifact.get("path") or ""))
    if not path.is_file() or _file_sha256(path) != artifact.get("artifact_sha256"):
        raise ValueError("terminal wrapper Stage5 artifact SHA drift")
    return _read_mapping(path)


def append_selected_terminal_wrappers(
    cache: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    merged_feedback: Sequence[Mapping[str, Any]],
    terminal_wrappers: Sequence[Mapping[str, Any]],
    artifact_reader: Callable[[Mapping[str, Any]], JSON] = (
        _read_stage5_terminal_artifact
    ),
) -> JSON:
    selected = cache_v2.validate_selection_binding(request, selection_binding)[
        "selected_candidates"
    ]
    selected_by_id = {row["candidate_id"]: row for row in selected}
    miss_ids = {
        str(row["candidate_id"])
        for row in physical_plan.get("logical_row_bindings") or ()
        if row.get("disposition") == "miss"
    }
    success_ids = {
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in merged_feedback
        if row.get("terminal_status") == stage5_search.SUCCESS_STATUS
    }
    feedback_by_id = {
        str(row.get("row_id") or row.get("manifest_job_id") or ""): copy.deepcopy(
            dict(row)
        )
        for row in merged_feedback
    }
    expected = miss_ids & success_ids
    validated: dict[str, JSON] = {}
    for wrapper in terminal_wrappers:
        terminal = actual_adapter.validate_actual_v3_terminal_wrapper(wrapper)
        candidate_id = str(terminal["candidate_id"])
        selected_row = selected_by_id.get(candidate_id)
        if (
            candidate_id in validated
            or candidate_id not in expected
            or selected_row is None
            or terminal["exact_cache_key_sha256"]
            != selected_row["exact_cache_key_sha256"]
            or terminal["exact_key_dimensions"] != selected_row["exact_key_dimensions"]
            or artifact_reader(terminal) != feedback_by_id.get(candidate_id)
        ):
            raise ValueError(
                "terminal wrapper is outside selected successful misses "
                "or differs from promoted feedback row"
            )
        validated[candidate_id] = terminal
    if set(validated) != expected:
        raise ValueError("selected successful miss terminal wrappers are incomplete")
    updated = copy.deepcopy(dict(cache))
    ordered_successes = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in merged_feedback
    ]
    for candidate_id in ordered_successes:
        if candidate_id in expected:
            updated = actual_adapter.append_terminal_wrapper(
                updated, validated[candidate_id]
            )
    return updated


def _write_promoted_terminal_artifacts(
    directory: Path,
    *,
    promoted_rows: Sequence[Mapping[str, Any]],
    physical_plan: Mapping[str, Any],
    write_json: Callable[..., None],
    validate: Callable[[], Mapping[str, Any]],
) -> dict[str, JSON]:
    """Write successful-miss Stage5 rows; these sidecars never release feedback."""
    promoted_by_id = {
        str(row.get("row_id") or row.get("manifest_job_id") or ""): copy.deepcopy(
            dict(row)
        )
        for row in promoted_rows
    }
    references: dict[str, JSON] = {}
    for binding in physical_plan.get("logical_row_bindings") or ():
        if binding.get("disposition") != "miss":
            continue
        candidate_id = str(binding.get("candidate_id") or "")
        row = promoted_by_id.get(candidate_id)
        if row is None or row.get("terminal_status") != stage5_search.SUCCESS_STATUS:
            continue
        logical_index = binding.get("logical_row_index")
        if (
            isinstance(logical_index, bool)
            or not isinstance(logical_index, int)
            or logical_index not in range(4)
        ):
            raise ValueError("physical plan logical row index is invalid")
        path = (
            directory
            / "promoted_stage5_terminals"
            / f"logical_row_{logical_index:02d}.json"
        )
        write_json(path, row, validate=validate)
        references[candidate_id] = {
            "artifact_kind": "stage5_terminal",
            "path": str(path),
            "artifact_sha256": hashlib.sha256(_json_bytes(row)).hexdigest(),
        }
    return references


def validate_precommit_lineage(
    *,
    request: Mapping[str, Any],
    exact_binding: Mapping[str, Any],
    cache_snapshot: Mapping[str, Any],
    cache_reveal: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    stored_admission: Mapping[str, Any],
    validate_admission: Callable[..., JSON],
) -> JSON:
    """Recompute every exact-cache artifact before any round commit is written."""
    try:
        validated_binding = cache_v2.validate_selection_binding(request, exact_binding)
        validated_snapshot = cache_v2.validate_actual_v3_cache(cache_snapshot)
        if (
            validated_binding["binding"] != exact_binding
            or validated_snapshot != cache_snapshot
        ):
            raise ValueError("validated exact binding or cache snapshot changed")
        expected_reveal = cache_v2.reveal_v2_cache_after_selection(
            request, exact_binding, cache_snapshot
        )
        if expected_reveal != cache_reveal:
            raise ValueError("cache reveal differs from exact recomputation")
        expected_plan = actual_adapter.build_miss_only_physical_plan(
            request,
            exact_binding,
            cache_reveal,
            cache_snapshot=cache_snapshot,
        )
        if expected_plan != physical_plan:
            raise ValueError("miss-only physical plan differs from exact recomputation")
        expected_admission = validate_admission(
            physical_plan,
            logical_request=request,
            selection_binding=exact_binding,
            cache_reveal=cache_reveal,
            cache_snapshot=cache_snapshot,
            contract_sha256=str(stored_admission.get("contract_sha256") or ""),
        )
        if expected_admission != stored_admission:
            raise ValueError("stored executor admission differs from recomputation")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("precommit lineage drift") from error
    return {
        "exact_binding": copy.deepcopy(dict(exact_binding)),
        "cache_snapshot": copy.deepcopy(dict(validated_snapshot)),
        "cache_reveal": copy.deepcopy(dict(expected_reveal)),
        "physical_plan": copy.deepcopy(dict(expected_plan)),
        "executor_admission": copy.deepcopy(dict(expected_admission)),
    }


def validate_zero_miss_round_lineage(
    terminal_payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    cache_reveal: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    source_result: Mapping[str, Any],
    stored_admission: Mapping[str, Any],
) -> JSON | None:
    """Cross-check preserved empty-terminal lineage against round artifacts."""
    projection = terminal_payload.get("projection_artifact")
    lineage = (
        projection.get("empty_projection_lineage")
        if isinstance(projection, Mapping)
        else None
    )
    if lineage is None:
        return None
    expected = {
        "logical_request_sha256": request.get("measurement_request_sha256"),
        "selection_binding_sha256": selection_binding.get("selection_binding_sha256"),
        "cache_snapshot_sha256": cache_reveal.get("cache_snapshot_sha256"),
        "cache_reveal_sha256": cache_reveal.get("cache_reveal_sha256"),
        "physical_request_sha256": physical_plan.get("physical_request_sha256"),
        "source_resolution_plan_sha256": source_plan.get(
            "source_resolution_plan_sha256"
        ),
        "source_resolution_result_sha256": source_result.get(
            "source_resolution_result_sha256"
        ),
        "resolved_source_lock_sha256": canonical_sha256([]),
        "executor_admission_sha256": stored_admission.get("admission_sha256"),
        "deployment_bundle_sha256": lineage.get("deployment_bundle_sha256"),
        "rows": [],
    }
    if (
        not isinstance(lineage, Mapping)
        or dict(lineage) != expected
        or physical_plan.get("physical_row_count") not in (None, 0)
        or physical_plan.get("rows") not in (None, [])
    ):
        raise ValueError("zero-miss terminal lineage differs from round artifacts")
    return copy.deepcopy(dict(lineage))


def finalize_round(
    v2_root: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    terminal_payload_path: Path,
    repo_root: Path,
    validate_root: Callable[..., Mapping[str, Any]],
    round_directory: Callable[[Path, str, int, int], Path],
    validate_identity: Callable[[Mapping[str, Any], Mapping[str, Any]], JSON],
    merge_logical: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]]], list[JSON]
    ],
    promote_atomic: Callable[..., JSON],
    append_wrappers: Callable[..., JSON],
    validate_admission: Callable[..., JSON],
    write_json: Callable[..., None],
    validate_precommit: Callable[..., JSON] | None = None,
    validate_selection_binding: Callable[..., JSON] | None = None,
) -> JSON:
    precommit_validator = validate_precommit or validate_precommit_lineage
    selection_validator = (
        validate_selection_binding or cache_v2.validate_selection_binding
    )
    validated = validate_root(v2_root, repo_root=repo_root)
    root = validated["root"]
    directory = round_directory(root, variant, seed, round_index)
    raw_terminal_payload = _read_mapping(terminal_payload_path)
    if raw_terminal_payload.get("barrier_release_allowed") is False:
        physical_feedback.validate_physical_terminal_batch(raw_terminal_payload)
        raise ValueError(
            "retryable infrastructure/evidence failure blocks feedback barrier"
        )
    request_path = directory / "logical_request.json"
    request = _read_mapping(request_path)
    identity = validate_identity(
        request, _read_mapping(directory / "selection_binding.json")
    )
    source_plan = source_resolution.validate_source_resolution_plan(
        _read_mapping(directory / "source_resolution_plan.json"),
        logical_request=request,
    )
    source_result = source_resolution.validate_formal_source_resolution_result(
        _read_mapping(directory / "source_resolution_result.json"), source_plan
    )
    binding = _read_mapping(directory / "exact_selection_binding.json")
    exact_request_sha = selection_validator(request, binding)["logical_request"][
        "logical_request_sha256"
    ]
    if exact_request_sha != identity["logical_request_sha256"]:
        raise ValueError("identity/exact selection binding request drift")
    snapshot = _read_mapping(directory / "cache_snapshot_before_reveal.json")
    reveal = _read_mapping(directory / "cache_reveal.json")
    physical_plan = _read_mapping(directory / "miss_only_physical_request.json")
    stored_admission = _read_mapping(directory / "executor_admission.json")
    precommit_validator(
        request=request,
        exact_binding=binding,
        cache_snapshot=snapshot,
        cache_reveal=reveal,
        physical_plan=physical_plan,
        stored_admission=stored_admission,
        validate_admission=validate_admission,
    )
    terminal_payload = physical_feedback.validate_physical_terminal_batch(
        raw_terminal_payload
    )
    validate_zero_miss_round_lineage(
        terminal_payload,
        request=request,
        selection_binding=binding,
        cache_reveal=reveal,
        physical_plan=physical_plan,
        source_plan=source_plan,
        source_result=source_result,
        stored_admission=stored_admission,
    )
    for failure in terminal_payload["failures"]:
        actual_adapter.validate_actual_v3_failure(
            failure,
            request,
            binding,
            physical_plan,
        )
    physical_rows = terminal_payload.get("rows")
    if not isinstance(physical_rows, list) or not all(
        isinstance(row, Mapping) for row in physical_rows
    ):
        raise ValueError("physical terminal rows are invalid")
    merged = merge_logical(request, feedback_for_reveal(reveal, physical_rows))
    with tempfile.TemporaryDirectory(prefix="stage7-v2-barrier-") as temporary:
        feedback_path = Path(temporary) / "historical_feedback.json"
        feedback_path.write_bytes(_json_bytes(merged))
        result = promote_atomic(
            request_path=request_path,
            historical_feedback_path=feedback_path,
        )
    barrier = result["barrier"]
    expected_budget = len(request.get("rows") or ())
    if (
        barrier.get("feedback_released") is not True
        or barrier.get("budget_consumed") != expected_budget
    ):
        raise ValueError("atomic feedback barrier did not release")
    cache = snapshot
    validator = lambda: validate_root(root, repo_root=repo_root)
    promoted_rows = result["promotion"]["rows"]
    stage5_terminal_artifacts = _write_promoted_terminal_artifacts(
        directory,
        promoted_rows=promoted_rows,
        physical_plan=physical_plan,
        write_json=write_json,
        validate=validator,
    )
    wrappers = physical_feedback.build_post_promotion_terminal_wrappers(
        logical_request=request,
        selection_binding=binding,
        physical_plan=physical_plan,
        promoted_rows=promoted_rows,
        lineage_inputs=terminal_payload["lineage_inputs"],
        stage5_terminal_artifacts=stage5_terminal_artifacts,
    )
    updated_cache = append_wrappers(
        cache,
        request=request,
        selection_binding=binding,
        physical_plan=physical_plan,
        merged_feedback=promoted_rows,
        terminal_wrappers=wrappers,
    )
    artifacts = (
        ("physical_terminal.json", terminal_payload),
        ("historical_feedback.json", merged),
        ("promoted_feedback.json", promoted_rows),
        ("promotion_audit.json", result["promotion"]["audit"]),
        ("stage5_atomic_audit.json", barrier),
        ("cache_after_round.json", updated_cache),
    )
    for filename, artifact in artifacts:
        write_json(directory / filename, artifact, validate=validator)
    payload = {
        "schema_version": BARRIER_SCHEMA,
        "logical_request_sha256": request["measurement_request_sha256"],
        "physical_terminal_batch_sha256": terminal_payload[
            "physical_terminal_batch_sha256"
        ],
        "physical_terminal_file_sha256": hashlib.sha256(
            _json_bytes(terminal_payload)
        ).hexdigest(),
        "feedback_released": True,
        "budget_consumed": 4,
        "successful_rows": barrier.get("successful_rows"),
        "feasibility_terminal_rows": barrier.get("feasibility_terminal_rows"),
        "silent_surrogate_fallback_count": result["promotion"]["audit"][
            "silent_surrogate_fallback_count"
        ],
        "historical_feedback_file_sha256": hashlib.sha256(
            _json_bytes(merged)
        ).hexdigest(),
        "promoted_feedback_file_sha256": hashlib.sha256(
            _json_bytes(result["promotion"]["rows"])
        ).hexdigest(),
        "promotion_audit_file_sha256": hashlib.sha256(
            _json_bytes(result["promotion"]["audit"])
        ).hexdigest(),
        "stage5_atomic_audit_file_sha256": hashlib.sha256(
            _json_bytes(barrier)
        ).hexdigest(),
        "cache_after_round_file_sha256": hashlib.sha256(
            _json_bytes(updated_cache)
        ).hexdigest(),
        "stage5_atomic_audit": copy.deepcopy(dict(barrier)),
    }
    committed = {**payload, "barrier_receipt_sha256": canonical_sha256(payload)}
    write_json(
        directory / "atomic_feedback_barrier.json",
        committed,
        validate=validator,
    )
    return committed
