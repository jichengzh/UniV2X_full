"""Pure Stage7 contracts for source materialization execution.

The authenticated Stage5 request remains byte-for-byte authoritative.  This
module projects only source-safe execution identity and never reads cache,
objective, terminal, measurement, AP, or feedback state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage5.measurement_plan_v2 import _validate_request
from framework.stage7.source_resolution_v2 import (
    build_source_resolution_retry_result,
    validate_source_resolution_plan,
)
from framework.stage7.source_round_orchestration_v2 import (
    validate_selection_identity,
)


EXECUTION_PLAN_SCHEMA = "stage7_source_execution_plan_v2"
SOURCE_LEASE_SCHEMA = "stage7_source_gpu_lease_v2"
EXPECTED_HOSTNAME = os.environ.get(
    "V2X_FORMAL_H800_HOSTNAME", "zs-nj-tap-gpu18"
).strip()
EXPECTED_GPU_MODELS = frozenset(
    {
        "NVIDIA H800",
        "NVIDIA H800 PCIe",
        "NVIDIA H800 SXM",
        "NVIDIA H800 NVL",
    }
)
EXECUTION_MODES = frozenset({"formal", "no_gpu_dryrun"})


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and len(set(value)) > 1
    )


def _request_from_bytes(request_bytes: bytes) -> dict[str, Any]:
    if not isinstance(request_bytes, bytes) or not request_bytes:
        raise ValueError("authenticated logical request bytes are required")
    try:
        payload = json.loads(request_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authenticated logical request JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError("authenticated logical request must be an object")
    request = copy.deepcopy(dict(payload))
    _validate_request(request)
    return request


def _candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def _group_identity(plan_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": plan_row["model"],
        "group_id": plan_row["group_id"],
        "width": copy.deepcopy(plan_row["width"]),
        "materialization_kind": plan_row["materialization_kind"],
        "source_plan_sha256": plan_row["source_plan_sha256"],
        "source_contract_sha256": plan_row["source_contract_sha256"],
        "source_evidence_path": plan_row["source_evidence_path"],
    }


def _unsigned_execution_plan(
    request_bytes: bytes,
    selection_binding: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> dict[str, Any]:
    request = _request_from_bytes(request_bytes)
    binding = validate_selection_identity(request, selection_binding)
    plan = validate_source_resolution_plan(source_plan, logical_request=request)
    rows = request["rows"]
    if len(rows) != 4 or len(plan["rows"]) != 4:
        raise ValueError("source execution requires four logical rows")
    candidate_bindings: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    group_order: list[str] = []
    identity_by_group: dict[str, dict[str, Any]] = {}
    for index, (logical_row, plan_row) in enumerate(zip(rows, plan["rows"])):
        candidate_id = _candidate_id(logical_row)
        source_contract = logical_row.get("source_contract")
        if (
            candidate_id != plan_row["candidate_id"]
            or plan_row["logical_row_index"] != index
            or logical_row.get("group_id") != plan_row["group_id"]
            or logical_row.get("model") != plan_row["model"]
            or logical_row.get("width") != plan_row["width"]
            or not isinstance(source_contract, Mapping)
            or _sha(dict(source_contract)) != plan_row["source_contract_sha256"]
            or _source_plan_sha(logical_row) != plan_row["source_plan_sha256"]
            or logical_row.get("source_evidence_sha256")
            != plan_row["source_plan_sha256"]
        ):
            raise ValueError("source execution logical/plan row binding drift")
        identity = _group_identity(plan_row)
        group_id = str(plan_row["group_id"])
        prior = identity_by_group.get(group_id)
        if prior is not None and prior != identity:
            raise ValueError("same group has divergent source identity")
        identity_by_group[group_id] = identity
        group_key = _sha(identity)
        binding_row = {
            "candidate_id": candidate_id,
            "logical_row_index": index,
            "logical_row_sha256": plan_row["logical_row_sha256"],
            "group_id": group_id,
            "source_group_execution_key": group_key,
            "source_evidence_path": plan_row["source_evidence_path"],
        }
        candidate_bindings.append(binding_row)
        if group_key not in groups:
            group_order.append(group_key)
            groups[group_key] = {
                **identity,
                "source_group_execution_key": group_key,
                "candidate_ids": [candidate_id],
            }
        else:
            groups[group_key] = {
                **groups[group_key],
                "candidate_ids": [*groups[group_key]["candidate_ids"], candidate_id],
            }
    group_jobs = [groups[key] for key in group_order]
    return {
        "schema_version": EXECUTION_PLAN_SCHEMA,
        "status": "frozen",
        "logical_request_sha256": request["measurement_request_sha256"],
        "logical_request_file_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "selection_binding_sha256": binding["selection_binding_sha256"],
        "source_resolution_plan_sha256": plan["source_resolution_plan_sha256"],
        "materializer": copy.deepcopy(plan["materializer"]),
        "ordered_candidate_ids": copy.deepcopy(plan["ordered_row_ids"]),
        "candidate_evidence_binding_count": 4,
        "candidate_evidence_bindings": candidate_bindings,
        "group_job_count": len(group_jobs),
        "group_jobs": group_jobs,
    }


def build_source_execution_plan(
    request_bytes: bytes,
    selection_binding: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Project source-safe group jobs without rewriting the Stage5 request."""
    payload = _unsigned_execution_plan(request_bytes, selection_binding, source_plan)
    return {**payload, "source_execution_plan_sha256": _sha(payload)}


def validate_source_execution_plan(
    execution_plan: Mapping[str, Any],
    request_bytes: bytes,
    selection_binding: Mapping[str, Any],
    source_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the projection from authenticated inputs and compare exactly."""
    if not isinstance(execution_plan, Mapping):
        raise ValueError("source execution plan must be an object")
    copied = copy.deepcopy(dict(execution_plan))
    recorded = copied.pop("source_execution_plan_sha256", None)
    expected = _unsigned_execution_plan(request_bytes, selection_binding, source_plan)
    if not _is_sha(recorded) or recorded != _sha(copied) or copied != expected:
        raise ValueError("source execution plan authentication failed")
    return {**copied, "source_execution_plan_sha256": recorded}


def candidate_evidence_paths(
    execution_plan: Mapping[str, Any],
) -> dict[str, str]:
    """Return exactly four selected candidate-to-evidence bindings."""
    bindings = execution_plan.get("candidate_evidence_bindings")
    ordered = execution_plan.get("ordered_candidate_ids")
    if (
        execution_plan.get("schema_version") != EXECUTION_PLAN_SCHEMA
        or not isinstance(bindings, list)
        or len(bindings) != 4
        or not isinstance(ordered, list)
        or [row.get("candidate_id") for row in bindings] != ordered
        or len(set(ordered)) != 4
    ):
        raise ValueError("candidate evidence bindings are not atomic four-row data")
    return {
        str(row["candidate_id"]): str(row["source_evidence_path"]) for row in bindings
    }


def _normalized_inventory(
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(inventory, Mapping):
        raise ValueError("H800 lease inventory is missing")
    result: dict[str, dict[str, Any]] = {}
    for uuid, raw in inventory.items():
        if (
            not isinstance(uuid, str)
            or not uuid.startswith("GPU-")
            or not isinstance(raw, Mapping)
        ):
            raise ValueError("H800 lease inventory UUID is invalid")
        index = raw.get("physical_index")
        model = raw.get("model")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or model not in EXPECTED_GPU_MODELS
        ):
            raise ValueError("H800 inventory index/model is invalid")
        result[uuid] = {"physical_index": index, "model": str(model)}
    return result


def _normalized_lock_owner(owner: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(owner, Mapping):
        raise ValueError("source lock owner identity is missing")
    pid = owner.get("pid")
    user = owner.get("owner")
    start = owner.get("start_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or not isinstance(user, str)
        or not user
        or not isinstance(start, str)
        or not start
    ):
        raise ValueError("source lock owner identity is invalid")
    return {"pid": pid, "owner": user, "start_time": start}


def _unsigned_lease(
    execution_plan: Mapping[str, Any],
    *,
    hostname: str,
    inventory: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, str],
    attempt_index: int,
    lock_owner: Mapping[str, Any],
    execution_mode: str,
    source_locks_held: bool,
) -> dict[str, Any]:
    if not hostname or (EXPECTED_HOSTNAME and hostname != EXPECTED_HOSTNAME):
        raise ValueError("formal H800 lease hostname drift")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("source lease execution mode is invalid")
    expected_locks = execution_mode == "formal"
    if source_locks_held is not expected_locks:
        raise ValueError("source lock claim disagrees with execution mode")
    if (
        isinstance(attempt_index, bool)
        or not isinstance(attempt_index, int)
        or attempt_index < 0
    ):
        raise ValueError("source attempt index must be nonnegative")
    if execution_plan.get("schema_version") != EXECUTION_PLAN_SCHEMA or not _is_sha(
        execution_plan.get("source_execution_plan_sha256")
    ):
        raise ValueError("authenticated source execution plan is required")
    jobs = execution_plan.get("group_jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("source execution group jobs are missing")
    group_keys = [str(job.get("source_group_execution_key") or "") for job in jobs]
    if (
        not isinstance(assignments, Mapping)
        or set(assignments) != set(group_keys)
        or any(not isinstance(value, str) for value in assignments.values())
        or len(set(assignments.values())) != len(group_keys)
    ):
        raise ValueError("source group GPU assignment is incomplete or duplicate")
    normalized_inventory = _normalized_inventory(inventory)
    group_bindings = []
    for job in jobs:
        group_key = str(job["source_group_execution_key"])
        uuid = str(assignments[group_key])
        if uuid not in normalized_inventory:
            raise ValueError("source group GPU assignment is outside inventory")
        gpu = normalized_inventory[uuid]
        group_bindings.append(
            {
                "source_group_execution_key": group_key,
                "group_id": job["group_id"],
                "candidate_ids": copy.deepcopy(job["candidate_ids"]),
                "gpu_uuid": uuid,
                "physical_index": gpu["physical_index"],
                "gpu_model": gpu["model"],
            }
        )
    indices = [row["physical_index"] for row in group_bindings]
    if len(set(indices)) != len(indices):
        raise ValueError("source lease physical index assignments are duplicate")
    owner = _normalized_lock_owner(lock_owner)
    attempt_payload = {
        "source_execution_plan_sha256": execution_plan["source_execution_plan_sha256"],
        "attempt_index": attempt_index,
        "execution_mode": execution_mode,
        "group_bindings": group_bindings,
    }
    return {
        "schema_version": SOURCE_LEASE_SCHEMA,
        "status": "frozen",
        "hostname": hostname,
        "execution_mode": execution_mode,
        "logical_request_sha256": execution_plan["logical_request_sha256"],
        "source_resolution_plan_sha256": execution_plan[
            "source_resolution_plan_sha256"
        ],
        "source_execution_plan_sha256": execution_plan["source_execution_plan_sha256"],
        "attempt_index": attempt_index,
        "source_attempt_sha256": _sha(attempt_payload),
        "inventory_snapshot_sha256": _sha(normalized_inventory),
        "inventory": normalized_inventory,
        "source_locks_held": source_locks_held,
        "source_lock_keys": sorted(group_keys),
        "lock_owner": owner,
        "gpu_uuids": [row["gpu_uuid"] for row in group_bindings],
        "group_binding_count": len(group_bindings),
        "group_bindings": group_bindings,
    }


def build_source_gpu_lease(
    execution_plan: Mapping[str, Any],
    *,
    hostname: str,
    inventory: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, str],
    attempt_index: int,
    lock_owner: Mapping[str, Any],
    execution_mode: str,
    source_locks_held: bool,
) -> dict[str, Any]:
    """Freeze UUID-to-index source assignments and lock-owner identity."""
    payload = _unsigned_lease(
        execution_plan,
        hostname=hostname,
        inventory=inventory,
        assignments=assignments,
        attempt_index=attempt_index,
        lock_owner=lock_owner,
        execution_mode=execution_mode,
        source_locks_held=source_locks_held,
    )
    return {**payload, "source_gpu_lease_sha256": _sha(payload)}


def validate_source_gpu_lease(
    lease: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a lease by rebuilding it from its frozen public fields."""
    if not isinstance(lease, Mapping):
        raise ValueError("source GPU lease must be an object")
    copied = copy.deepcopy(dict(lease))
    recorded = copied.pop("source_gpu_lease_sha256", None)
    bindings = copied.get("group_bindings")
    if not isinstance(bindings, list):
        raise ValueError("source GPU lease group bindings are missing")
    inventory = copied.get("inventory")
    assignments = {
        str(row.get("source_group_execution_key") or ""): str(row.get("gpu_uuid") or "")
        for row in bindings
        if isinstance(row, Mapping)
    }
    expected = _unsigned_lease(
        execution_plan,
        hostname=str(copied.get("hostname") or ""),
        inventory=inventory,
        assignments=assignments,
        attempt_index=copied.get("attempt_index"),
        lock_owner=copied.get("lock_owner"),
        execution_mode=str(copied.get("execution_mode") or ""),
        source_locks_held=copied.get("source_locks_held"),
    )
    if not _is_sha(recorded) or recorded != _sha(copied) or copied != expected:
        raise ValueError("source GPU lease authentication failed")
    return {**copied, "source_gpu_lease_sha256": recorded}


def build_zero_budget_source_retry(
    source_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    *,
    failed_group_keys: Sequence[str],
    reason_code: str,
) -> dict[str, Any]:
    """Map failed group jobs back to the frozen candidate order."""
    validate_source_resolution_plan(source_plan)
    jobs = execution_plan.get("group_jobs")
    if not isinstance(jobs, list):
        raise ValueError("source execution group jobs are missing")
    failed = [str(value) for value in failed_group_keys]
    known = {str(job["source_group_execution_key"]) for job in jobs}
    if not failed or len(set(failed)) != len(failed) or not set(failed) <= known:
        raise ValueError("failed source group identity is invalid")
    retry_ids = [
        candidate_id
        for job in jobs
        if job["source_group_execution_key"] in set(failed)
        for candidate_id in job["candidate_ids"]
    ]
    ordered = execution_plan.get("ordered_candidate_ids")
    retry_set = set(retry_ids)
    return build_source_resolution_retry_result(
        source_plan,
        retry_candidate_ids=[
            candidate_id for candidate_id in ordered if candidate_id in retry_set
        ],
        reason_code=reason_code,
    )


__all__ = [
    "build_source_execution_plan",
    "validate_source_execution_plan",
    "build_source_gpu_lease",
    "validate_source_gpu_lease",
    "build_zero_budget_source_retry",
    "candidate_evidence_paths",
]
