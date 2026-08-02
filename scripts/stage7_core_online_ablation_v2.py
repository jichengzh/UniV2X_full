#!/usr/bin/env python3
"""Stage7 selection/cache orchestration over frozen actual-feedback v3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5 import single_target_search_v2 as stage5_search
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import actual_v3_selector_v2 as actual_selector
from framework.stage7 import core_cache_v2 as cache_v2
from framework.stage7 import deployment_bundle_v2
from framework.stage7 import search_policy_v1 as search_policy
from framework.stage7 import source_round_orchestration_v2 as source_round
from framework.stage7 import source_resolution_v2 as source_resolution
from framework.stage7.core_ablation_v2 import (
    CORE_VARIANTS,
    V2_ROOT,
    canonical_sha256,
    validate_v2_contract,
)
from framework.stage7.online_component_ablation_v1 import SEEDS
from scripts import stage5_promote_actual_feedback_v3 as stage5_promote
from scripts import stage7_actual_feedback_barrier_v2 as feedback_barrier
from scripts import stage7_prepare_core_ablation_v2 as prepare_v2

BARRIER_SCHEMA = "stage7_actual_v3_atomic_feedback_barrier_v2"
ADMISSION_SCHEMA = "stage7_actual_v3_executor_admission_v2"
RECEIPT_SCHEMA = "stage7_core_online_command_receipt_v2"
JSON = dict[str, Any]
_PINNED_SELECTOR_INPUT_IDENTITIES = {
    key: copy.deepcopy(prepare_v2.FORMAL_INPUT_IDENTITIES[key])
    for key in (
        "gold176",
        "graph_features",
        "capability_profiles",
        "source_registry",
    )
}

class SourceResolutionRequiredBeforeExactCacheReveal(ValueError):
    """Selected structures lack immutable checkpoint/ONNX identities."""

    def __init__(self, audit: Mapping[str, Any]) -> None:
        self.audit = copy.deepcopy(dict(audit))
        super().__init__(
            "round0 selected source contract requires both checkpoint and ONNX "
            "SHA before exact cache reveal"
        )

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

def write_immutable_json(
    path: Path,
    payload: Any,
    *,
    validate: Callable[[], Mapping[str, Any]],
) -> None:
    """Revalidate the formal root immediately before every immutable write."""
    validate()
    encoded = _json_bytes(payload)
    if path.is_file():
        if path.read_bytes() != encoded:
            raise ValueError(f"refusing to overwrite drifted artifact: {path}")
        return
    if path.exists():
        raise ValueError(f"immutable artifact path is not a file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)

def _verify_bound_file(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path") or ""))
    expected = str(record.get("sha256") or "")
    if not path.is_absolute() or len(expected) != 64 or not path.is_file():
        raise ValueError(f"formal immutable input is unavailable: {label}")
    if _file_sha256(path) != expected:
        raise ValueError(f"formal immutable input SHA drift: {label}")
    return path

def validate_formal_v2_root(
    v2_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    expected_release_sha256: str | None = None, expected_manifest_file_sha256: str | None = None,
) -> JSON:
    """Authenticate the complete v2 contract, inputs, executors and 12 roots."""
    deployment_bundle_v2.validate_deployment_bundle(
        v2_root,
        frozen_repo_root=repo_root,
        expected_release_sha256=expected_release_sha256, expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    root = Path(v2_root).resolve(strict=True)
    contract = validate_v2_contract(
        _read_mapping(root / "contracts" / "core_ablation_v2.json")
    )
    if Path(str(contract["v2_root"])).resolve(strict=False) != root:
        raise ValueError("formal v2 root contract drift")
    prepare_v2.verify_frozen_actual_v3_executors(repo_root)
    for label, record in contract["immutable_inputs"].items():
        _verify_bound_file(record, label=str(label))
    registry = _read_mapping(root / "contracts" / "pre_scan_candidate_registry.json")
    rows = registry.get("rows")
    if (
        registry.get("schema_version") != "stage7_core_ablation_v2_pre_scan_registry"
        or not isinstance(rows, list)
        or len(rows) != 686
        or registry.get("ordered_pre_scan_sha256")
        != contract["ordered_pre_scan_sha256"]
        or canonical_sha256(rows) != contract["ordered_pre_scan_sha256"]
    ):
        raise ValueError("formal pre-scan registry drift")
    cache_v2.validate_actual_v3_cache(
        _read_mapping(root / "contracts" / "measurement_cache_initial.json")
    )
    contracts = sorted(root.glob("variants/*/seed_*/trajectory_contract.json"))
    expected = {
        root / "variants" / variant / f"seed_{seed}" / "trajectory_contract.json"
        for variant in CORE_VARIANTS
        for seed in SEEDS
    }
    if set(contracts) != expected:
        raise ValueError("formal v2 requires exactly twelve trajectory contracts")
    for path in contracts:
        trajectory = _read_mapping(path)
        stored = trajectory.pop("trajectory_contract_sha256", None)
        if (
            stored != canonical_sha256(trajectory)
            or trajectory.get("core_ablation_contract_sha256")
            != contract["contract_sha256"]
            or trajectory.get("scanner_claim_allowed") is not False
        ):
            raise ValueError("trajectory contract drift")
    return {
        "root": root,
        "contract": contract,
        "pre_scan_pool": copy.deepcopy(rows),
    }

def initialize_formal_v2(
    v2_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    expected_release_sha256: str | None = None, expected_manifest_file_sha256: str | None = None,
) -> JSON:
    """Initialize exactly the twelve canonical seed roots using Task1."""
    result = prepare_v2.initialize_v2_trajectories(
        Path(v2_root).resolve(),
        repo_root=Path(repo_root).resolve(),
        expected_release_sha256=expected_release_sha256, expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    validated = validate_formal_v2_root(
        v2_root,
        repo_root=repo_root,
        expected_release_sha256=expected_release_sha256, expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    if int(result.get("trajectory_count", -1)) != 12:
        raise ValueError("Stage7 v2 initialization did not create 12 trajectories")
    return {
        **copy.deepcopy(result),
        "contract_sha256": validated["contract"]["contract_sha256"],
    }

freeze_selection_identity = source_round.freeze_selection_identity
validate_selection_identity = source_round.validate_selection_identity
bind_reveal_and_plan = source_round.bind_reveal_and_plan
validate_executor_admission = source_round.validate_executor_admission

_ROUND_ARTIFACTS = {
    "selector_output": "selector_output.json",
    "selector_audit": "selector_audit.json",
    "logical_request": "logical_request.json",
    "selection_binding": "selection_binding.json",
    "cache_snapshot": "cache_snapshot_before_reveal.json",
    "cache_reveal": "cache_reveal.json",
    "physical_plan": "miss_only_physical_request.json",
    "executor_admission": "executor_admission.json",
}

def freeze_request_artifacts(
    round_dir: Path,
    artifacts: Mapping[str, Any],
    *,
    validate: Callable[[], Mapping[str, Any]],
) -> None:
    unknown = set(artifacts) - set(_ROUND_ARTIFACTS)
    required = set(_ROUND_ARTIFACTS) - {"selector_output", "cache_snapshot"}
    if unknown or not required.issubset(artifacts):
        raise ValueError("request artifact set is invalid")
    for key in _ROUND_ARTIFACTS:
        if key in artifacts:
            write_immutable_json(
                round_dir / _ROUND_ARTIFACTS[key],
                artifacts[key],
                validate=validate,
            )

def merge_feedback_in_logical_order(
    request: Mapping[str, Any],
    feedback_rows: Sequence[Mapping[str, Any]],
) -> list[JSON]:
    logical = cache_v2.validate_logical_request(request)
    if len(feedback_rows) != 4:
        raise ValueError("atomic feedback requires all four logical rows")
    by_id = {
        str(row.get("row_id") or row.get("manifest_job_id") or ""): copy.deepcopy(
            dict(row)
        )
        for row in feedback_rows
    }
    if set(by_id) != set(logical["row_ids"]) or len(by_id) != 4:
        raise ValueError("atomic feedback requires all four unique logical rows")
    merged: list[JSON] = []
    for requested in logical["rows"]:
        row_id = str(requested.get("row_id") or requested.get("manifest_job_id"))
        feedback = by_id[row_id]
        if (
            feedback.get("measurement_request_row_sha256")
            != logical["row_sha256"][row_id]
        ):
            raise ValueError("feedback request row SHA drift")
        for field, value in requested.items():
            if field != "graph_features" and feedback.get(field) != value:
                raise ValueError(f"feedback identity field drift: {row_id}:{field}")
        merged.append(feedback)
    return merged

def promote_and_finalize_atomic_barrier(
    *,
    request_path: Path,
    historical_feedback_path: Path,
    extractor: Callable[
        [Path], Mapping[str, Any]
    ] = stage5_promote.extract_graph_features,
) -> JSON:
    """Call the real v3 promoter followed by the real four-row finalizer."""
    request = _read_mapping(request_path)
    promotion = stage5_promote.promote_feedback_batch(
        request_path, historical_feedback_path, extractor=extractor
    )
    if (
        promotion["audit"].get("promoted_row_count") != 4
        or promotion["audit"].get("silent_surrogate_fallback_count") != 0
    ):
        raise ValueError("actual-feedback-v3 promotion did not close")
    barrier = stage5_search.finalize_atomic_batch(request, promotion["rows"])
    return {"promotion": promotion, "barrier": barrier}

def next_round_allowed(round_dir: Path) -> bool:
    try:
        validate_committed_barrier(Path(round_dir))
    except (OSError, ValueError):
        return False
    return True

def _validate_prior_feedback_barrier(round_dir: Path) -> JSON:
    """Authenticate released prior feedback without opening any cache artifact."""
    directory = Path(round_dir)
    barrier = _read_mapping(directory / "atomic_feedback_barrier.json")
    recorded = barrier.pop("barrier_receipt_sha256", None)
    request = _read_mapping(directory / "logical_request.json")
    bound_files = {
        "physical_terminal_file_sha256": directory / "physical_terminal.json",
        "historical_feedback_file_sha256": directory / "historical_feedback.json",
        "promoted_feedback_file_sha256": directory / "promoted_feedback.json",
        "promotion_audit_file_sha256": directory / "promotion_audit.json",
        "stage5_atomic_audit_file_sha256": directory / "stage5_atomic_audit.json",
    }
    if (
        barrier.get("schema_version") != BARRIER_SCHEMA
        or recorded != canonical_sha256(barrier)
        or barrier.get("feedback_released") is not True
        or barrier.get("budget_consumed") != 4
        or barrier.get("logical_request_sha256")
        != request.get("measurement_request_sha256")
        or any(
            barrier.get(field) != _file_sha256(path)
            for field, path in bound_files.items()
        )
    ):
        raise ValueError("prior feedback barrier authentication failed")
    return {**barrier, "barrier_receipt_sha256": recorded}

def validate_committed_barrier(round_dir: Path) -> JSON:
    """Authenticate the complete four-row commit, not merely its marker."""
    directory = Path(round_dir)
    request_path = directory / "logical_request.json"
    binding = _read_mapping(directory / "exact_selection_binding.json")
    snapshot = _read_mapping(directory / "cache_snapshot_before_reveal.json")
    reveal = _read_mapping(directory / "cache_reveal.json")
    physical_plan = _read_mapping(directory / "miss_only_physical_request.json")
    stored_admission = _read_mapping(directory / "executor_admission.json")
    historical_path = directory / "historical_feedback.json"
    physical_terminal_path = directory / "physical_terminal.json"
    promoted_path = directory / "promoted_feedback.json"
    audit_path = directory / "promotion_audit.json"
    atomic_path = directory / "stage5_atomic_audit.json"
    cache_path = directory / "cache_after_round.json"
    request = _read_mapping(request_path)
    logical = cache_v2.validate_logical_request(request)
    historical = _rows(_read_json(historical_path), label="historical feedback")
    promoted = _rows(_read_json(promoted_path), label="promoted feedback")
    audit = _read_mapping(audit_path)
    atomic_artifact = _read_mapping(atomic_path)
    cache_after = cache_v2.validate_actual_v3_cache(_read_mapping(cache_path))
    physical_terminal = (
        feedback_barrier.physical_feedback.validate_physical_terminal_batch(
            _read_mapping(physical_terminal_path)
        )
    )
    feedback_barrier.validate_zero_miss_round_lineage(
        physical_terminal,
        request=request,
        selection_binding=binding,
        cache_reveal=reveal,
        physical_plan=physical_plan,
        source_plan=_read_mapping(directory / "source_resolution_plan.json"),
        source_result=_read_mapping(directory / "source_resolution_result.json"),
        stored_admission=stored_admission,
    )
    admission = validate_executor_admission(
        physical_plan,
        logical_request=request,
        selection_binding=binding,
        cache_reveal=reveal,
        cache_snapshot=snapshot,
        contract_sha256=str(stored_admission.get("contract_sha256") or ""),
    )
    if admission != stored_admission:
        raise ValueError("committed executor admission drift")
    if len(historical) != 4 or len(promoted) != 4:
        raise ValueError("committed barrier does not contain four feedback rows")
    physical_feedback = [
        row
        for row in historical
        if next(
            entry["disposition"]
            for entry in reveal["entries"]
            if entry["candidate_id"]
            == str(row.get("row_id") or row.get("manifest_job_id") or "")
        )
        == "miss"
    ]
    expected_historical = merge_feedback_in_logical_order(
        request, _feedback_for_reveal(reveal, physical_feedback)
    )
    if expected_historical != historical:
        raise ValueError("committed historical feedback lineage drift")
    if physical_terminal["rows"] != physical_feedback:
        raise ValueError("committed physical terminal rows drift")
    promoted_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "") for row in promoted
    ]
    if promoted_ids != logical["row_ids"]:
        raise ValueError("committed promoted feedback order drift")
    for row in promoted:
        copied = copy.deepcopy(row)
        recorded = copied.pop("actual_feedback_row_sha256", None)
        if (
            row.get("feedback_feature_contract") != "actual_feedback_v3"
            or row.get("graph_feature_promotion_schema")
            != "stage5_actual_feedback_promotion_v3"
            or recorded != canonical_sha256(copied)
        ):
            raise ValueError("committed actual-feedback-v3 row drift")
    with tempfile.TemporaryDirectory(
        prefix="stage7-v2-revalidate-promotion-"
    ) as temporary:
        temporary_root = Path(temporary)
        temporary_request = temporary_root / "logical_request.json"
        temporary_historical = temporary_root / "historical_feedback.json"
        temporary_request.write_bytes(request_path.read_bytes())
        temporary_historical.write_bytes(historical_path.read_bytes())
        recomputed_promotion = stage5_promote.promote_feedback_batch(
            temporary_request, temporary_historical
        )
    if (
        recomputed_promotion["rows"] != promoted
        or recomputed_promotion["audit"] != audit
    ):
        raise ValueError("committed promotion cannot be reproduced")
    if (
        audit.get("schema_version") != "stage5_actual_feedback_batch_audit_v3"
        or audit.get("promoted_row_count") != 4
        or audit.get("silent_surrogate_fallback_count") != 0
        or audit.get("measurement_request_file_sha256") != _file_sha256(request_path)
        or audit.get("historical_feedback_file_sha256") != _file_sha256(historical_path)
    ):
        raise ValueError("committed promotion audit drift")
    path = directory / "atomic_feedback_barrier.json"
    barrier = _read_mapping(path)
    recorded = barrier.pop("barrier_receipt_sha256", None)
    atomic = barrier.get("stage5_atomic_audit")
    recomputed_atomic = stage5_search.finalize_atomic_batch(request, promoted)
    if (
        recorded != canonical_sha256(barrier)
        or barrier.get("schema_version") != BARRIER_SCHEMA
        or barrier.get("logical_request_sha256") != logical["logical_request_sha256"]
        or barrier.get("feedback_released") is not True
        or barrier.get("budget_consumed") != 4
        or barrier.get("physical_terminal_batch_sha256")
        != physical_terminal["physical_terminal_batch_sha256"]
        or barrier.get("physical_terminal_file_sha256")
        != _file_sha256(physical_terminal_path)
        or not isinstance(atomic, Mapping)
        or dict(atomic) != atomic_artifact
        or dict(atomic) != recomputed_atomic
        or atomic.get("schema_version") != "stage5_atomic_batch_audit_v2"
        or atomic.get("feedback_released") is not True
        or atomic.get("budget_consumed") != 4
        or atomic.get("released_feedback_rows") != promoted
        or barrier.get("historical_feedback_file_sha256")
        != _file_sha256(historical_path)
        or barrier.get("promoted_feedback_file_sha256") != _file_sha256(promoted_path)
        or barrier.get("promotion_audit_file_sha256") != _file_sha256(audit_path)
        or barrier.get("stage5_atomic_audit_file_sha256") != _file_sha256(atomic_path)
        or barrier.get("cache_after_round_file_sha256") != _file_sha256(cache_path)
    ):
        raise ValueError("atomic feedback barrier authentication failed")
    selected = cache_v2.validate_selection_binding(request, binding)[
        "selected_candidates"
    ]
    selected_by_id = {row["candidate_id"]: row for row in selected}
    expected_cache = copy.deepcopy(snapshot)
    for entry, row in zip(reveal["entries"], promoted):
        if (
            entry["disposition"] != "miss"
            or row.get("terminal_status") != stage5_search.SUCCESS_STATUS
        ):
            continue
        selected_row = selected_by_id[entry["candidate_id"]]
        terminal = cache_after["entries"].get(selected_row["exact_cache_key_sha256"])
        if terminal is None:
            raise ValueError("committed successful miss is absent from cache")
        if _read_stage5_terminal_artifact(terminal) != row:
            raise ValueError("committed cache terminal differs from feedback")
        expected_cache = actual_adapter.append_terminal_wrapper(
            expected_cache, terminal
        )
    if expected_cache != cache_after:
        raise ValueError("committed cache contains non-round lineage")
    return {**barrier, "barrier_receipt_sha256": recorded}

_rows = source_round._rows
_load_bound_payload = source_round._load_bound_payload

def _selector_inputs(validated: Mapping[str, Any]) -> JSON:
    return source_round.selector_inputs(
        validated,
        pinned_identities=_PINNED_SELECTOR_INPUT_IDENTITIES,
        expected_executor_records=prepare_v2.core_executor_records(),
    )

def _source_file_error(
    source: Mapping[str, Any], *, path_field: str, sha_field: str
) -> str | None:
    path_value, sha_value = source.get(path_field), source.get(sha_field)
    if (
        not isinstance(path_value, str)
        or not Path(path_value).is_absolute()
        or not isinstance(sha_value, str)
        or len(sha_value) != 64
        or any(character not in "0123456789abcdef" for character in sha_value)
    ):
        return f"{path_field}/{sha_field}_missing_or_invalid"
    if len(set(sha_value)) == 1:
        return f"{sha_field}_placeholder"
    path = Path(path_value)
    if not path.is_file():
        return f"{path_field}_unavailable"
    if _file_sha256(path) != sha_value:
        return f"{path_field}/{sha_field}_mismatch"
    return None

def _preflight_selected_sources(request: Mapping[str, Any]) -> JSON:
    rows = request.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("round0 selected request rows are invalid")
    unresolved = []
    checks = (
        ("checkpoint_path", "checkpoint_sha256"),
        ("onnx_path", "onnx_sha256"),
        ("materialization_evidence_path", "materialization_evidence_sha256"),
    )
    for row in rows:
        source = row.get("source_contract")
        errors = (
            [
                error
                for path_field, sha_field in checks
                if (
                    error := _source_file_error(
                        source, path_field=path_field, sha_field=sha_field
                    )
                )
            ]
            if isinstance(source, Mapping)
            else ["source_contract_missing"]
        )
        if errors:
            unresolved.append(
                {
                    "candidate_id": str(
                        row.get("row_id") or row.get("manifest_job_id") or ""
                    ),
                    "resolution_errors": errors,
                }
            )
    audit = {
        "schema_version": "stage7_round0_source_resolution_preflight_v2",
        "status": (
            "source_resolution_ready"
            if not unresolved
            else "source_resolution_required_before_exact_cache_reveal"
        ),
        "selected_candidate_count": len(rows),
        "unresolved_candidate_count": len(unresolved),
        "unresolved_candidates": unresolved,
        "placeholder_sha_inserted": False,
        "cache_reveal_allowed": not unresolved,
    }
    if unresolved:
        raise SourceResolutionRequiredBeforeExactCacheReveal(audit)
    return audit

def _exact_dimensions(
    request: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    selector_inputs: Mapping[str, Any],
    resolved_sources_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, JSON]:
    if resolved_sources_by_candidate is None:
        _preflight_selected_sources(request)
    return source_round.exact_dimensions(
        request,
        contract=contract,
        selector_inputs_value=selector_inputs,
        resolved_sources_by_candidate=resolved_sources_by_candidate,
    )

def _round_dir(root: Path, variant: str, seed: int, round_index: int) -> Path:
    return source_round.round_dir(root, variant, seed, round_index)

def _prior_selection_state(
    root: Path, variant: str, seed: int, round_index: int
) -> JSON:
    return source_round.prior_selection_state(
        root,
        variant,
        seed,
        round_index,
        barrier_validator=_validate_prior_feedback_barrier,
    )

def _prior_state(root: Path, variant: str, seed: int, round_index: int) -> JSON:
    """Compatibility helper for post-source callers that explicitly need cache."""
    return {
        **_prior_selection_state(root, variant, seed, round_index),
        "cache": _global_cache_snapshot(root),
    }

def _global_cache_snapshot(root: Path) -> JSON:
    return source_round.global_cache_snapshot(root, round_committed=next_round_allowed)

def round_controller_state(round_dir: Path) -> str:
    return source_round.round_controller_state(round_dir)

def freeze_selection_and_source_plan(
    v2_root: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    repo_root: Path = REPO_ROOT,
    expected_release_sha256: str | None = None, expected_manifest_file_sha256: str | None = None,
) -> JSON:
    return source_round.freeze_selection_and_source_plan(
        v2_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        repo_root=repo_root,
        validate_root=lambda root, *, repo_root: validate_formal_v2_root(
            root,
            repo_root=repo_root,
            expected_release_sha256=expected_release_sha256, expected_manifest_file_sha256=expected_manifest_file_sha256,
        ),
        write_json=write_immutable_json,
        selector_inputs_builder=_selector_inputs,
        prior_state_builder=_prior_selection_state,
    )

def prepare_round(
    v2_root: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    repo_root: Path = REPO_ROOT,
    expected_release_sha256: str | None = None, expected_manifest_file_sha256: str | None = None,
) -> JSON:
    """Fail-closed compatibility entry: freeze selection/source plan only."""
    return freeze_selection_and_source_plan(
        v2_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        repo_root=repo_root,
        expected_release_sha256=expected_release_sha256, expected_manifest_file_sha256=expected_manifest_file_sha256,
    )

_validated_source_result = source_round._validated_source_result
_resolved_source_rows = source_round._resolved_source_rows

def bind_reveal_after_source_ready(
    v2_root: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    source_result_path: Path,
    synthetic_no_gpu_dryrun: bool = False,
    repo_root: Path = REPO_ROOT,
    expected_release_sha256: str | None = None, expected_manifest_file_sha256: str | None = None,
) -> JSON:
    return source_round.bind_reveal_after_source_ready(
        v2_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        source_result_path=source_result_path,
        synthetic_no_gpu_dryrun=synthetic_no_gpu_dryrun,
        repo_root=repo_root,
        validate_root=lambda root, *, repo_root: validate_formal_v2_root(
            root,
            repo_root=repo_root,
            expected_release_sha256=expected_release_sha256, expected_manifest_file_sha256=expected_manifest_file_sha256,
        ),
        write_json=write_immutable_json,
        selector_inputs_builder=_selector_inputs,
        exact_dimensions_builder=lambda request, **kwargs: _exact_dimensions(
            request,
            contract=kwargs["contract"],
            selector_inputs=kwargs["selector_inputs_value"],
            resolved_sources_by_candidate=kwargs["resolved_sources_by_candidate"],
        ),
        cache_snapshot_builder=_global_cache_snapshot,
    )

_feedback_for_reveal = feedback_barrier.feedback_for_reveal
_read_stage5_terminal_artifact = feedback_barrier._read_stage5_terminal_artifact

def append_selected_terminal_wrappers(*args: Any, **kwargs: Any) -> JSON:
    return feedback_barrier.append_selected_terminal_wrappers(
        *args,
        **kwargs,
        artifact_reader=_read_stage5_terminal_artifact,
    )

def finalize_round(
    v2_root: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    terminal_payload_path: Path,
    repo_root: Path = REPO_ROOT,
    expected_release_sha256: str | None = None, expected_manifest_file_sha256: str | None = None,
) -> JSON:
    return feedback_barrier.finalize_round(
        v2_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        terminal_payload_path=terminal_payload_path,
        repo_root=repo_root,
        validate_root=lambda root, *, repo_root: validate_formal_v2_root(
            root,
            repo_root=repo_root,
            expected_release_sha256=expected_release_sha256, expected_manifest_file_sha256=expected_manifest_file_sha256,
        ),
        round_directory=_round_dir,
        validate_identity=validate_selection_identity,
        merge_logical=merge_feedback_in_logical_order,
        promote_atomic=promote_and_finalize_atomic_barrier,
        append_wrappers=append_selected_terminal_wrappers,
        validate_admission=validate_executor_admission,
        write_json=write_immutable_json,
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    initialize.add_argument("--v2-root", type=Path, default=V2_ROOT)
    initialize.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze = commands.add_parser("freeze-selection-and-source-plan")
    bind = commands.add_parser("bind-reveal-after-source-ready")
    prepare = commands.add_parser("prepare-round")
    finalize = commands.add_parser("finalize-round")
    for command in (freeze, bind, prepare, finalize):
        command.add_argument("--v2-root", type=Path, default=V2_ROOT)
        command.add_argument("--repo-root", type=Path, default=REPO_ROOT)
        command.add_argument("--variant", choices=CORE_VARIANTS, required=True)
        command.add_argument("--seed", choices=SEEDS, type=int, required=True)
        command.add_argument("--round-index", choices=range(4), type=int, required=True)
    bind.add_argument("--source-result-json", type=Path, required=True)
    bind.add_argument("--synthetic-no-gpu-dryrun", action="store_true")
    finalize.add_argument("--terminal-payload-json", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--v2-root", type=Path, default=V2_ROOT)
    validate.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "initialize":
        result = initialize_formal_v2(args.v2_root, repo_root=args.repo_root)
    elif args.command in {
        "freeze-selection-and-source-plan",
        "prepare-round",
    }:
        result = freeze_selection_and_source_plan(
            args.v2_root,
            variant=args.variant,
            seed=args.seed,
            round_index=args.round_index,
            repo_root=args.repo_root,
        )
    elif args.command == "bind-reveal-after-source-ready":
        result = bind_reveal_after_source_ready(
            args.v2_root,
            variant=args.variant,
            seed=args.seed,
            round_index=args.round_index,
            source_result_path=args.source_result_json,
            synthetic_no_gpu_dryrun=args.synthetic_no_gpu_dryrun,
            repo_root=args.repo_root,
        )
    elif args.command == "finalize-round":
        result = finalize_round(
            args.v2_root,
            variant=args.variant,
            seed=args.seed,
            round_index=args.round_index,
            terminal_payload_path=args.terminal_payload_json,
            repo_root=args.repo_root,
        )
    else:
        validated = validate_formal_v2_root(args.v2_root, repo_root=args.repo_root)
        result = {
            "schema_version": RECEIPT_SCHEMA,
            "command": "validate",
            "contract_sha256": validated["contract"]["contract_sha256"],
            "trajectory_count": 12,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
