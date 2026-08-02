"""Fail-closed Stage7 selection/source/exact-cache round orchestration.

This module owns only the source-round state machine.  Hardware execution,
measurement, AP, terminal promotion, and cache append remain in the existing
actual-feedback-v3 path.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage5 import single_target_search_v2 as stage5_search
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import actual_v3_selector_v2 as actual_selector
from framework.stage7 import core_cache_v2 as cache_v2
from framework.stage7 import search_policy_v1 as search_policy
from framework.stage7 import source_resolution_v2 as source_resolution
from framework.stage7 import source_relocation_v2 as source_relocation
from framework.stage7.core_ablation_v2 import CORE_VARIANTS, canonical_sha256
from framework.stage7.online_component_ablation_v1 import SEEDS


JSON = dict[str, Any]
SELECTION_IDENTITY_SCHEMA = "stage7_selected_identity_binding_v2"
ADMISSION_SCHEMA = "stage7_actual_v3_executor_admission_v2"
RECEIPT_SCHEMA = "stage7_core_online_command_receipt_v2"
BARRIER_SCHEMA = "stage7_actual_v3_atomic_feedback_barrier_v2"
ValidateRoot = Callable[[Path], Mapping[str, Any]]
WriteJSON = Callable[..., None]


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


def _verify_bound_file(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path") or ""))
    expected = str(record.get("sha256") or "")
    if not path.is_absolute() or len(expected) != 64 or not path.is_file():
        raise ValueError(f"formal immutable input is unavailable: {label}")
    if _file_sha256(path) != expected:
        raise ValueError(f"formal immutable input SHA drift: {label}")
    return path


def _rows(payload: Any, *, label: str) -> list[JSON]:
    value = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{label} rows are invalid")
    return [copy.deepcopy(dict(row)) for row in value]


def _load_bound_payload(record: Mapping[str, Any], *, label: str) -> Any:
    path = _verify_bound_file(record, label=label)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    return _read_json(path)


def selector_inputs(
    validated: Mapping[str, Any],
    *,
    pinned_identities: Mapping[str, Mapping[str, Any]],
    expected_executor_records: Sequence[Mapping[str, Any]],
) -> JSON:
    records = validated["contract"]["immutable_inputs"]
    for label, pinned in pinned_identities.items():
        record = records.get(label)
        if not isinstance(record, Mapping) or set(record) < {"path", "sha256"}:
            raise ValueError(f"frozen selector input identity is missing: {label}")
        normalized = {
            "path": str(Path(str(record.get("path") or "")).resolve(strict=False)),
            "sha256": record.get("sha256"),
        }
        if normalized != pinned:
            raise ValueError(f"frozen selector input identity drift: {label}")
        _verify_bound_file(record, label=label)
    gold = _load_bound_payload(records["gold176"], label="gold176")
    graphs = _load_bound_payload(records["graph_features"], label="graph_features")
    profiles = _load_bound_payload(
        records["capability_profiles"], label="capability_profiles"
    )
    measurement = _load_bound_payload(records["measurement_ap"], label="measurement_ap")
    closure = next(
        (
            copy.deepcopy(dict(payload["closure"]))
            for payload in (gold, measurement)
            if isinstance(payload, Mapping)
            and isinstance(payload.get("closure"), Mapping)
        ),
        None,
    )
    if closure is None:
        raise ValueError("frozen Stage5 closure is missing from formal inputs")
    expected = next(
        (
            record
            for record in expected_executor_records
            if record["path"] == "framework/stage5/single_target_search_v2.py"
        ),
        None,
    )
    actual = next(
        (
            record
            for record in validated["contract"]["actual_feedback_v3_executors"]
            if record.get("path") == "framework/stage5/single_target_search_v2.py"
        ),
        None,
    )
    if actual != expected:
        raise ValueError("frozen Stage5 coldstart callable/source SHA drift")
    initial_rows = stage5_search.freeze_initial_coldstart(_rows(gold, label="Gold176"))
    return {
        "initial_rows": initial_rows,
        "initial_graph_features": _rows(graphs, label="graph features"),
        "capability_profiles": _rows(profiles, label="capability profiles"),
        "closure": closure,
        "measurement_contract": measurement,
        "coldstart_freeze_binding": copy.deepcopy(dict(actual)),
    }


def freeze_selection_identity(selection: Mapping[str, Any]) -> JSON:
    if not isinstance(selection, Mapping):
        raise ValueError("selector output must be a mapping")
    request = selection.get("measurement_request")
    acquisition = selection.get("acquisition")
    if not isinstance(request, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("selector output lacks acquisition or logical request")
    logical = cache_v2.validate_logical_request(request)
    selected_ids = [str(value) for value in acquisition.get("selected_row_ids") or ()]
    if len(selected_ids) != 4 or selected_ids != logical["row_ids"]:
        raise ValueError("selector and four-row logical request order drift")
    payload = {
        "schema_version": SELECTION_IDENTITY_SCHEMA,
        "selection_frozen": True,
        "logical_request_sha256": logical["logical_request_sha256"],
        "ordered_candidate_ids": copy.deepcopy(selected_ids),
        "ordered_candidate_ids_sha256": canonical_sha256(selected_ids),
        "logical_row_sha256": copy.deepcopy(logical["row_sha256"]),
        "exact_binding_frozen": False,
        "cache_membership_observed": False,
    }
    return {**payload, "selection_binding_sha256": canonical_sha256(payload)}


def validate_selection_identity(
    request: Mapping[str, Any], binding: Mapping[str, Any]
) -> JSON:
    logical = cache_v2.validate_logical_request(request)
    if not isinstance(binding, Mapping):
        raise ValueError("selection identity binding is missing")
    copied = copy.deepcopy(dict(binding))
    recorded = copied.pop("selection_binding_sha256", None)
    if (
        copied.get("schema_version") != SELECTION_IDENTITY_SCHEMA
        or copied.get("selection_frozen") is not True
        or copied.get("logical_request_sha256") != logical["logical_request_sha256"]
        or copied.get("ordered_candidate_ids") != logical["row_ids"]
        or copied.get("ordered_candidate_ids_sha256")
        != canonical_sha256(logical["row_ids"])
        or copied.get("logical_row_sha256") != logical["row_sha256"]
        or copied.get("exact_binding_frozen") is not False
        or copied.get("cache_membership_observed") is not False
        or recorded != canonical_sha256(copied)
    ):
        raise ValueError("selection identity binding authentication failed")
    return {**copied, "selection_binding_sha256": recorded}


def bind_reveal_and_plan(
    selection: Mapping[str, Any],
    *,
    exact_dimensions_by_candidate: Mapping[str, Mapping[str, Any]],
    cache_snapshot: Mapping[str, Any],
) -> JSON:
    frozen = actual_adapter.bind_selector_output(
        selection,
        exact_dimensions_by_candidate=exact_dimensions_by_candidate,
    )
    reveal = cache_v2.reveal_v2_cache_after_selection(
        frozen["logical_request"],
        frozen["selection_binding"],
        cache_snapshot,
    )
    plan = actual_adapter.build_miss_only_physical_plan(
        frozen["logical_request"],
        frozen["selection_binding"],
        reveal,
        cache_snapshot=cache_snapshot,
    )
    return {
        **frozen,
        "cache_snapshot": copy.deepcopy(dict(cache_snapshot)),
        "cache_reveal": reveal,
        "physical_plan": plan,
    }


def validate_executor_admission(
    physical_plan: Mapping[str, Any],
    *,
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    cache_reveal: Mapping[str, Any],
    cache_snapshot: Mapping[str, Any],
    contract_sha256: str,
) -> JSON:
    expected = actual_adapter.build_miss_only_physical_plan(
        logical_request,
        selection_binding,
        cache_reveal,
        cache_snapshot=cache_snapshot,
    )
    if copy.deepcopy(dict(physical_plan)) != expected:
        raise ValueError("actual-v3 physical request authentication failed")
    rows = physical_plan.get("rows")
    bindings = physical_plan.get("logical_row_bindings")
    count = physical_plan.get("physical_row_count")
    if (
        physical_plan.get("schema_version") != actual_adapter.PHYSICAL_REQUEST_SCHEMA
        or not isinstance(rows, list)
        or not isinstance(bindings, list)
        or len(bindings) != 4
        or isinstance(count, bool)
        or count != len(rows)
        or count not in range(5)
        or len({str(row.get("row_id") or row.get("manifest_job_id")) for row in rows})
        != len(rows)
    ):
        raise ValueError("actual-v3 miss-only executor admission failed")
    payload = {
        "schema_version": ADMISSION_SCHEMA,
        "admission_passed": True,
        "contract_sha256": contract_sha256,
        "logical_request_sha256": physical_plan["logical_request_sha256"],
        "physical_request_sha256": physical_plan["physical_request_sha256"],
        "logical_row_count": 4,
        "physical_row_count": count,
        "execution_primitive": "existing_stage5_stage3_actual_feedback_v3",
        "gpu_jobs_launched": 0,
    }
    return {**payload, "admission_sha256": canonical_sha256(payload)}


def exact_dimensions(
    request: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    selector_inputs_value: Mapping[str, Any],
    resolved_sources_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, JSON]:
    scope_record = contract["immutable_inputs"]["scope_input_batch"]
    scope = _load_bound_payload(scope_record, label="scope_input_batch")
    measurement = selector_inputs_value["measurement_contract"]
    if not isinstance(scope, Mapping) or not isinstance(measurement, Mapping):
        raise ValueError("formal exact-key protocol contracts are invalid")
    protocol = measurement.get("exact_key_protocol_sha256")
    required = {
        "build_protocol_sha256",
        "tuning_protocol_sha256",
        "measurement_protocol_sha256",
        "ap_protocol_sha256",
        "runtime_contract_sha256",
    }
    if not isinstance(protocol, Mapping) or set(protocol) != required:
        raise ValueError("formal exact-key protocol SHA binding shape drift")
    if resolved_sources_by_candidate is not None:
        logical = cache_v2.validate_logical_request(request)
        if set(resolved_sources_by_candidate) != set(logical["row_ids"]):
            raise ValueError("resolved source candidate coverage/order drift")
    result: dict[str, JSON] = {}
    for row in request["rows"]:
        candidate_id = str(row.get("row_id") or row.get("manifest_job_id"))
        source = (
            resolved_sources_by_candidate[candidate_id]
            if resolved_sources_by_candidate is not None
            else row.get("source_contract")
        )
        if not isinstance(source, Mapping):
            raise ValueError("candidate resolved source identity is missing")
        checkpoint_sha = source.get(
            "checkpoint_sha256", source.get("synthetic_checkpoint_sha256")
        )
        onnx_sha = source.get("onnx_sha256", source.get("synthetic_onnx_sha256"))
        result[candidate_id] = {
            "candidate_id": candidate_id,
            "model": row["model"],
            "capability_profile_id": row["capability_profile_id"],
            "hardware_id": row["hardware_id"],
            "dispatch_key": row["dispatch_key"],
            "measurement_scope": scope.get("measurement_scope"),
            "input_protocol_sha256": scope_record["sha256"],
            "batch_size": scope.get("inference_batch"),
            "genome": copy.deepcopy(row["genome"]),
            "q_mode": row["q_mode"],
            "source_checkpoint_sha256": checkpoint_sha,
            "onnx_sha256": onnx_sha,
            **copy.deepcopy(dict(protocol)),
        }
        cache_v2.validate_formal_exact_dimensions(result[candidate_id])
    return result


def round_dir(root: Path, variant: str, seed: int, round_index: int) -> Path:
    if variant not in CORE_VARIANTS or seed not in SEEDS or round_index not in range(4):
        raise ValueError("variant, seed or round is outside the frozen contract")
    return root / "variants" / variant / f"seed_{seed}" / f"round_{round_index:02d}"


def validate_prior_feedback_barrier(round_directory: Path) -> JSON:
    directory = Path(round_directory)
    barrier = _read_mapping(directory / "atomic_feedback_barrier.json")
    recorded = barrier.pop("barrier_receipt_sha256", None)
    request = _read_mapping(directory / "logical_request.json")
    bound_files = {
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


def prior_selection_state(
    root: Path,
    variant: str,
    seed: int,
    round_index: int,
    *,
    barrier_validator: Callable[[Path], Mapping[str, Any]],
) -> JSON:
    selected: list[str] = []
    promoted: list[JSON] = []
    a2_frozen = None
    for index in range(round_index):
        directory = round_dir(root, variant, seed, index)
        try:
            barrier_validator(directory)
        except (OSError, ValueError) as error:
            raise ValueError(
                "previous atomic feedback barrier is not committed"
            ) from error
        request = _read_mapping(directory / "logical_request.json")
        selected.extend(cache_v2.validate_logical_request(request)["row_ids"])
        promoted.extend(
            _rows(
                _read_json(directory / "promoted_feedback.json"),
                label="promoted",
            )
        )
        if index == 0:
            selector_output = _read_mapping(directory / "selector_output.json")
            a2_frozen = selector_output.get("a2_frozen")
    return {
        "prior_selected_ids": selected,
        "promoted_feedback": promoted,
        "a2_frozen": a2_frozen,
    }


def global_cache_snapshot(
    root: Path,
    *,
    round_committed: Callable[[Path], bool],
) -> JSON:
    merged = _read_mapping(root / "contracts" / "measurement_cache_initial.json")
    cache_v2.validate_actual_v3_cache(merged)
    committed = sorted(
        path
        for path in root.glob("variants/*/seed_*/round_*/cache_after_round.json")
        if round_committed(path.parent)
    )
    for path in committed:
        branch = cache_v2.validate_actual_v3_cache(_read_mapping(path))
        for key in sorted(branch["entries"]):
            merged = cache_v2.append_actual_v3_terminal_evidence(
                merged, branch["entries"][key]
            )
    return merged


def round_controller_state(round_directory: Path) -> str:
    directory = Path(round_directory)
    logical = directory / "logical_request.json"
    identity = directory / "selection_binding.json"
    plan = directory / "source_resolution_plan.json"
    result = directory / "source_resolution_result.json"
    exact = directory / "exact_selection_binding.json"
    reveal = directory / "cache_reveal.json"
    if reveal.is_file():
        if not all(path.is_file() for path in (logical, identity, plan, result, exact)):
            raise ValueError("CACHE_REVEALED artifact lineage is incomplete")
        return "CACHE_REVEALED"
    if exact.is_file():
        if not all(path.is_file() for path in (logical, identity, plan, result)):
            raise ValueError("EXACT_BINDING_FROZEN artifact lineage is incomplete")
        return "EXACT_BINDING_FROZEN"
    if result.is_file():
        if not all(path.is_file() for path in (logical, identity, plan)):
            raise ValueError("SOURCE_READY artifact lineage is incomplete")
        return "SOURCE_READY"
    if plan.is_file():
        if not logical.is_file() or not identity.is_file():
            raise ValueError("SOURCE_PLAN_FROZEN artifact lineage is incomplete")
        return "SOURCE_PLAN_FROZEN"
    if logical.is_file() or identity.is_file():
        if not logical.is_file() or not identity.is_file():
            raise ValueError("SELECTED_FROZEN artifact lineage is incomplete")
        return "SELECTED_FROZEN"
    return "UNINITIALIZED"


def freeze_selection_and_source_plan(
    v2_root: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    repo_root: Path,
    validate_root: Callable[..., Mapping[str, Any]],
    write_json: WriteJSON,
    selector_inputs_builder: Callable[[Mapping[str, Any]], JSON],
    prior_state_builder: Callable[[Path, str, int, int], JSON],
) -> JSON:
    validated = validate_root(v2_root, repo_root=repo_root)
    root = validated["root"]
    prior = prior_state_builder(root, variant, seed, round_index)
    inputs = selector_inputs_builder(validated)
    task = search_policy.build_stage7_task(inputs["capability_profiles"][0])
    selected = actual_selector.select_actual_v3_pre_scan_round(
        variant=variant,
        seed=seed,
        round_index=round_index,
        task=task,
        pre_scan_pool=validated["pre_scan_pool"],
        initial_rows=inputs["initial_rows"],
        initial_graph_features=inputs["initial_graph_features"],
        capability_profiles=inputs["capability_profiles"],
        closure=inputs["closure"],
        prior_selected_ids=prior["prior_selected_ids"],
        promoted_feedback=prior["promoted_feedback"],
        a2_frozen=prior["a2_frozen"],
    )
    selection = copy.deepcopy(dict(selected.selection))
    raw_request = copy.deepcopy(dict(selection["measurement_request"]))
    relocated = source_relocation.relocate_pyramid_request_to_v2_root(
        raw_request,
        v2_root=root,
    )
    request = copy.deepcopy(dict(relocated["request"]))
    relocation_audit = copy.deepcopy(dict(relocated["audit"]))
    selection["measurement_request"] = copy.deepcopy(request)
    identity_binding = freeze_selection_identity(selection)
    plan = source_resolution.build_source_resolution_plan(request)
    source_resolution.validate_source_resolution_plan(plan, logical_request=request)
    directory = round_dir(root, variant, seed, round_index)
    validator = lambda: validate_root(root, repo_root=repo_root)
    artifacts = {
        "selector_output.json": selection,
        "selector_audit.json": {
            **copy.deepcopy(dict(selected.audit)),
            "coldstart_freeze_binding": copy.deepcopy(
                inputs["coldstart_freeze_binding"]
            ),
            "pre_relocation_request_sha256": relocation_audit[
                "pre_relocation_request_sha256"
            ],
            "source_relocation_sha256": relocation_audit[
                "source_relocation_sha256"
            ],
        },
        "logical_request.json": request,
        "selection_binding.json": identity_binding,
        "source_contract_relocation.json": relocation_audit,
        "source_resolution_plan.json": plan,
    }
    for filename, artifact in artifacts.items():
        write_json(directory / filename, artifact, validate=validator)
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "command": "freeze-selection-and-source-plan",
        "variant": variant,
        "seed": seed,
        "round_index": round_index,
        "contract_sha256": validated["contract"]["contract_sha256"],
        "logical_request_sha256": request["measurement_request_sha256"],
        "pre_relocation_request_sha256": relocation_audit[
            "pre_relocation_request_sha256"
        ],
        "source_relocation_sha256": relocation_audit[
            "source_relocation_sha256"
        ],
        "selection_binding_sha256": identity_binding["selection_binding_sha256"],
        "source_resolution_plan_sha256": plan["source_resolution_plan_sha256"],
        "controller_state": "SOURCE_PLAN_FROZEN",
        "cache_membership_observed": False,
        "exact_binding_frozen": False,
        "state_transitions": ["SELECTED_FROZEN", "SOURCE_PLAN_FROZEN"],
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    write_json(
        directory / "source_plan_freeze_receipt.json",
        receipt,
        validate=validator,
    )
    return receipt


def _validated_source_result(
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    synthetic_no_gpu_dryrun: bool,
) -> tuple[str, JSON]:
    schema = payload.get("schema_version")
    if schema == source_resolution.RETRY_SCHEMA:
        return (
            "retry",
            source_resolution.validate_source_resolution_retry_result(payload, plan),
        )
    if schema == source_resolution.SYNTHETIC_SCHEMA:
        if not synthetic_no_gpu_dryrun:
            raise ValueError(
                "synthetic source result requires explicit no-GPU dry-run mode"
            )
        return (
            "synthetic",
            source_resolution.validate_synthetic_dryrun_source_resolution_result(
                payload, plan
            ),
        )
    if synthetic_no_gpu_dryrun:
        raise ValueError(
            "explicit no-GPU dry-run mode accepts only synthetic source result"
        )
    return (
        "formal",
        source_resolution.validate_formal_source_resolution_result(payload, plan),
    )


def _resolved_source_rows(result: Mapping[str, Any], *, kind: str) -> dict[str, JSON]:
    rows = result.get("rows")
    if kind not in {"formal", "synthetic"} or not isinstance(rows, list):
        raise ValueError("source result is not ready for exact binding")
    return {str(row["candidate_id"]): copy.deepcopy(dict(row)) for row in rows}


def bind_reveal_after_source_ready(
    v2_root: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    source_result_path: Path,
    synthetic_no_gpu_dryrun: bool,
    repo_root: Path,
    validate_root: Callable[..., Mapping[str, Any]],
    write_json: WriteJSON,
    selector_inputs_builder: Callable[[Mapping[str, Any]], JSON],
    exact_dimensions_builder: Callable[..., dict[str, JSON]],
    cache_snapshot_builder: Callable[[Path], JSON],
) -> JSON:
    validated = validate_root(v2_root, repo_root=repo_root)
    root = validated["root"]
    directory = round_dir(root, variant, seed, round_index)
    request_path = directory / "logical_request.json"
    request_bytes = request_path.read_bytes()
    request = _read_mapping(request_path)
    identity = validate_selection_identity(
        request, _read_mapping(directory / "selection_binding.json")
    )
    plan = source_resolution.validate_source_resolution_plan(
        _read_mapping(directory / "source_resolution_plan.json"),
        logical_request=request,
    )
    result_kind, source_result = _validated_source_result(
        _read_mapping(source_result_path),
        plan,
        synthetic_no_gpu_dryrun=synthetic_no_gpu_dryrun,
    )
    validator = lambda: validate_root(root, repo_root=repo_root)
    if result_kind == "retry":
        write_json(
            directory / "source_resolution_retry.json",
            source_result,
            validate=validator,
        )
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "command": "bind-reveal-after-source-ready",
            "variant": variant,
            "seed": seed,
            "round_index": round_index,
            "contract_sha256": validated["contract"]["contract_sha256"],
            "logical_request_sha256": request["measurement_request_sha256"],
            "selection_binding_sha256": identity["selection_binding_sha256"],
            "source_resolution_plan_sha256": plan["source_resolution_plan_sha256"],
            "source_resolution_retry_sha256": source_result[
                "source_resolution_retry_sha256"
            ],
            "controller_state": "SOURCE_PLAN_FROZEN",
            "selected_event_budget_delta": 0,
            "cache_membership_observed": False,
            "exact_binding_frozen": False,
        }
        receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
        write_json(
            directory / "source_resolution_retry_receipt.json",
            receipt,
            validate=validator,
        )
        if request_path.read_bytes() != request_bytes:
            raise ValueError("logical request bytes changed during source retry")
        return receipt

    source_rows = _resolved_source_rows(source_result, kind=result_kind)
    inputs = selector_inputs_builder(validated)
    dimensions = exact_dimensions_builder(
        request,
        contract=validated["contract"],
        selector_inputs_value=inputs,
        resolved_sources_by_candidate=source_rows,
    )
    selection = _read_mapping(directory / "selector_output.json")
    exact = actual_adapter.bind_selector_output(
        selection, exact_dimensions_by_candidate=dimensions
    )
    if exact["logical_request"] != request:
        raise ValueError("logical request changed during exact source binding")
    exact_binding = exact["selection_binding"]
    if result_kind == "synthetic":
        request_sha = request["measurement_request_sha256"]
        plan_sha = plan["source_resolution_plan_sha256"]
        result_sha = source_result["synthetic_dryrun_result_sha256"]
        binding_sha = exact_binding["selection_binding_sha256"]
        synthetic_protocol = {
            "schema_version": "stage7_synthetic_no_gpu_physical_protocol_v2",
            "logical_request_sha256": request_sha,
            "ordered_candidate_ids": list(plan["ordered_row_ids"]),
            "source_resolution_plan_sha256": plan_sha,
            "synthetic_source_resolution_result_sha256": result_sha,
            "synthetic_exact_selection_binding_sha256": binding_sha,
            "cache_snapshot_allowed": False,
            "cache_reveal_allowed": False,
            "formal_miss_plan_allowed": False,
            "executor_admission_allowed": False,
            "gpu_launch_allowed": False,
            "eligible_for_cache_append": False,
            "eligible_for_finalization": False,
        }
        synthetic_protocol = {
            **synthetic_protocol,
            "synthetic_physical_protocol_sha256": canonical_sha256(synthetic_protocol),
        }
        for filename, artifact in (
            ("synthetic_source_resolution_result.json", source_result),
            ("synthetic_exact_selection_binding.json", exact_binding),
            ("synthetic_physical_protocol.json", synthetic_protocol),
        ):
            write_json(directory / filename, artifact, validate=validator)
        payload = {
            "schema_version": RECEIPT_SCHEMA,
            "command": "bind-reveal-after-source-ready",
            "variant": variant,
            "seed": seed,
            "round_index": round_index,
            "contract_sha256": validated["contract"]["contract_sha256"],
            "logical_request_sha256": request_sha,
            "selection_binding_sha256": identity["selection_binding_sha256"],
            "source_resolution_plan_sha256": plan_sha,
            "synthetic_source_resolution_result_sha256": result_sha,
            "synthetic_exact_selection_binding_sha256": binding_sha,
            "synthetic_physical_protocol_sha256": synthetic_protocol[
                "synthetic_physical_protocol_sha256"
            ],
            "controller_state": "SYNTHETIC_PROTOCOL_CLOSED",
            "synthetic_nonfinal": True,
            "cache_membership_observed": False,
            "eligible_for_cache_append": False,
            "eligible_for_finalization": False,
            "formal_terminal_evidence_required": False,
            "gpu_launch_allowed": False,
        }
        receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
        write_json(
            directory / "synthetic_protocol_receipt.json",
            receipt,
            validate=validator,
        )
        if request_path.read_bytes() != request_bytes:
            raise ValueError("logical request bytes changed during synthetic dry-run")
        return receipt

    write_json(
        directory / "source_resolution_result.json",
        source_result,
        validate=validator,
    )
    write_json(
        directory / "exact_selection_binding.json",
        exact_binding,
        validate=validator,
    )
    snapshot_path = directory / "cache_snapshot_before_reveal.json"
    cache_snapshot = (
        _read_mapping(snapshot_path)
        if snapshot_path.is_file()
        else cache_snapshot_builder(root)
    )
    reveal = cache_v2.reveal_v2_cache_after_selection(
        request, exact_binding, cache_snapshot
    )
    physical_plan = actual_adapter.build_miss_only_physical_plan(
        request,
        exact_binding,
        reveal,
        cache_snapshot=cache_snapshot,
    )
    admission = validate_executor_admission(
        physical_plan,
        logical_request=request,
        selection_binding=exact_binding,
        cache_reveal=reveal,
        cache_snapshot=cache_snapshot,
        contract_sha256=validated["contract"]["contract_sha256"],
    )
    for filename, artifact in (
        ("cache_snapshot_before_reveal.json", cache_snapshot),
        ("cache_reveal.json", reveal),
        ("miss_only_physical_request.json", physical_plan),
        ("executor_admission.json", admission),
    ):
        write_json(directory / filename, artifact, validate=validator)
    if request_path.read_bytes() != request_bytes:
        raise ValueError("logical request bytes changed after source resolution")
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "command": "bind-reveal-after-source-ready",
        "variant": variant,
        "seed": seed,
        "round_index": round_index,
        "contract_sha256": validated["contract"]["contract_sha256"],
        "logical_request_sha256": request["measurement_request_sha256"],
        "selection_binding_sha256": identity["selection_binding_sha256"],
        "exact_selection_binding_sha256": exact_binding["selection_binding_sha256"],
        "source_resolution_plan_sha256": plan["source_resolution_plan_sha256"],
        "source_resolution_result_sha256": source_result.get(
            "source_resolution_result_sha256",
            source_result.get("synthetic_dryrun_result_sha256"),
        ),
        "physical_request_sha256": physical_plan["physical_request_sha256"],
        "controller_state": "CACHE_REVEALED",
        "state_transitions": [
            "SOURCE_READY",
            "EXACT_BINDING_FROZEN",
            "CACHE_REVEALED",
        ],
        "synthetic_nonfinal": result_kind == "synthetic",
        "eligible_for_cache_append": False,
        "eligible_for_finalization": False,
        "formal_terminal_evidence_required": result_kind == "formal",
    }
    receipt = {**payload, "receipt_sha256": canonical_sha256(payload)}
    write_json(
        directory / "exact_binding_reveal_receipt.json",
        receipt,
        validate=validator,
    )
    return receipt
