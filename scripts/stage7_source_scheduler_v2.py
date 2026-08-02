#!/usr/bin/env python3
"""Source-phase state and lease helpers for the Stage7 v2 scheduler.

This module is deliberately hardware-passive.  It validates scheduler state,
lock ordering, inventory identity, retry classification, and the
source-to-measurement barrier.  The Phase3A resolver remains the sole source
materialization implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage7 import source_resolution_v2
from scripts import stage7_core_online_ablation_v2 as task4_online


MAX_ACTIVE_BATCHES = 2
ACTIVE_STATES = frozenset(
    {
        "SOURCE_LEASE_PENDING",
        "SOURCE_MATERIALIZING",
        "MEASUREMENT_LEASE_PENDING",
        "MEASURING",
    }
)
FORMAL_MEASUREMENT_ARTIFACTS = (
    "logical_request.json",
    "source_resolution_plan.json",
    "source_resolution_result.json",
    "exact_selection_binding.json",
    "cache_snapshot_before_reveal.json",
    "cache_reveal.json",
    "miss_only_physical_request.json",
    "executor_admission.json",
)
H800_MODELS = frozenset(
    {
        "NVIDIA H800",
        "NVIDIA H800 PCIe",
        "NVIDIA H800 SXM",
        "NVIDIA H800 NVL",
    }
)
PHASE3A_REPLACEMENT_PINS = {
    "framework/stage7/source_execution_v2.py": "606282b3be8478ae6f7c8b2d401591bfd40b58e9684936e7695b5766a1b345e8",
    "scripts/stage7_resolve_round_sources_v2.py": "6824892fcc6a51ff51f0a061fb2b4f401008e24bc0512fdeaf1ce8760c9f7b8c",
}
PHASE3B_DEPLOY_FILES = (
    "scripts/stage7_core_ablation_scheduler_v2.py",
    "scripts/stage7_execute_actual_v3_misses_v2.sh",
    "scripts/stage7_scheduler_requests_v2.py",
    "scripts/stage7_h800_runtime_v2.py",
    "scripts/stage7_source_scheduler_v2.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dependency_pins(
    repo_root: Path,
    pins: Mapping[str, str] = PHASE3A_REPLACEMENT_PINS,
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    validated: dict[str, str] = {}
    for relative, expected in pins.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("dependency path escapes repository") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Phase3A dependency SHA drift: {relative}")
        validated[relative] = expected
    return validated


def required_deployment_pins(repo_root: Path) -> dict[str, str]:
    root = Path(repo_root).resolve()
    phase3b = {
        relative: sha256_file(root / relative) for relative in PHASE3B_DEPLOY_FILES
    }
    return {**PHASE3A_REPLACEMENT_PINS, **phase3b}


def validate_deploy_manifest_pins(
    manifest: Mapping[str, Any],
    pins: Mapping[str, str] = PHASE3A_REPLACEMENT_PINS,
) -> dict[str, str]:
    records = manifest.get("files")
    if not isinstance(records, (list, tuple)):
        raise ValueError("deployment pin missing: file records")
    deployed = {
        relative: any(
            isinstance(record, Mapping)
            and str(record.get("destination") or "").endswith("/" + relative)
            and record.get("sha256") == expected
            for record in records
        )
        for relative, expected in pins.items()
    }
    missing = [relative for relative, present in deployed.items() if not present]
    if missing:
        raise ValueError(f"deployment pin missing: {missing}")
    return dict(pins)


def detect_round_state(round_dir: Path) -> str:
    directory = Path(round_dir)
    if (directory / "atomic_feedback_barrier.json").is_file():
        return "FEEDBACK_COMPLETE"
    synthetic_present = any(directory.glob("synthetic_*"))
    formal_complete = all(
        (directory / name).is_file() for name in FORMAL_MEASUREMENT_ARTIFACTS
    )
    if synthetic_present and (
        formal_complete or (directory / "source_resolution_result.json").is_file()
    ):
        return "INVALID_MIXED_SOURCE_STATE"
    if synthetic_present:
        return "SYNTHETIC_NONFINAL"
    if (directory / "source_resolution_retry.json").is_file():
        return "SOURCE_RETRY_REQUIRED"
    if formal_complete:
        return "CACHE_REVEALED"
    if (directory / "source_resolution_result.json").is_file():
        return "SOURCE_READY"
    if (directory / "source_gpu_lease.json").is_file():
        return "SOURCE_MATERIALIZING"
    if (directory / "source_execution_plan.json").is_file():
        return "SOURCE_LEASE_PENDING"
    if (directory / "source_resolution_plan.json").is_file():
        return "SOURCE_PLAN_FROZEN"
    if (directory / "logical_request.json").is_file():
        return "SELECTED_FROZEN"
    return "UNINITIALIZED"


def measurement_ready(round_dir: Path) -> bool:
    return detect_round_state(round_dir) == "CACHE_REVEALED"


def require_measurement_ready(round_dir: Path) -> dict[str, Path]:
    directory = Path(round_dir)
    state = detect_round_state(directory)
    if state == "SYNTHETIC_NONFINAL":
        raise ValueError("synthetic source result cannot admit measurement")
    if state != "CACHE_REVEALED":
        raise ValueError("measurement requires SOURCE_READY plus exact/cache reveal")
    return {name: directory / name for name in FORMAL_MEASUREMENT_ARTIFACTS}


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"measurement artifact must be an object: {path.name}")
    return copy.deepcopy(dict(payload))


def validate_measurement_artifacts(round_dir: Path) -> dict[str, Any]:
    """Revalidate source and exact/cache admission before a measurement lease."""
    paths = require_measurement_ready(round_dir)
    try:
        logical = _load_mapping(paths["logical_request.json"])
        source_plan = _load_mapping(paths["source_resolution_plan.json"])
        source_result = _load_mapping(paths["source_resolution_result.json"])
        exact = _load_mapping(paths["exact_selection_binding.json"])
        snapshot = _load_mapping(paths["cache_snapshot_before_reveal.json"])
        reveal = _load_mapping(paths["cache_reveal.json"])
        physical = _load_mapping(paths["miss_only_physical_request.json"])
        admission = _load_mapping(paths["executor_admission.json"])
        validated_plan = source_resolution_v2.validate_source_resolution_plan(
            source_plan, logical_request=logical
        )
        validated_result = (
            source_resolution_v2.validate_formal_source_resolution_result(
                source_result, validated_plan
            )
        )
        contract_sha = str(admission.get("contract_sha256") or "")
        recomputed = task4_online.validate_executor_admission(
            physical,
            logical_request=logical,
            selection_binding=exact,
            cache_reveal=reveal,
            cache_snapshot=snapshot,
            contract_sha256=contract_sha,
        )
        if recomputed != admission:
            raise ValueError("stored executor admission differs from recomputation")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("measurement artifact authentication failed") from error
    return {
        "logical_request": logical,
        "source_plan": validated_plan,
        "source_result": validated_result,
        "exact_selection_binding": exact,
        "cache_snapshot": snapshot,
        "cache_reveal": reveal,
        "physical_plan": physical,
        "executor_admission": admission,
    }


def validate_zero_budget_retry(
    retry: Mapping[str, Any],
    *,
    logical_request_sha256: str,
    source_resolution_plan_sha256: str,
) -> dict[str, Any]:
    copied = copy.deepcopy(dict(retry))
    if (
        copied.get("logical_request_sha256") != logical_request_sha256
        or copied.get("source_resolution_plan_sha256") != source_resolution_plan_sha256
        or copied.get("selected_event_budget_delta") != 0
        or copied.get("partial_reveal_allowed") is not False
    ):
        raise ValueError("source retry request/plan/zero-budget identity drift")
    return copied


def _release_all(locks: Sequence[Any]) -> None:
    for lock in locks:
        lock.release()


def acquire_source_then_uuid_locks(
    source_keys: Sequence[str],
    gpu_uuids: Sequence[str],
    source_lock_factory: Callable[[str], Any | None],
    uuid_lock_factory: Callable[[str], Any | None],
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    source_locks: list[Any] = []
    uuid_locks: list[Any] = []
    for key in sorted(source_keys):
        lock = source_lock_factory(key)
        if lock is None:
            _release_all(source_locks)
            return None
        source_locks.append(lock)
    for uuid in gpu_uuids:
        lock = uuid_lock_factory(uuid)
        if lock is None:
            _release_all(uuid_locks)
            _release_all(source_locks)
            return None
        uuid_locks.append(lock)
    return tuple(source_locks), tuple(uuid_locks)


def available_batch_slots(controllers: Mapping[str, Mapping[str, Any]]) -> int:
    active = sum(
        str(controller.get("status") or "") in ACTIVE_STATES
        for controller in controllers.values()
    )
    return max(0, MAX_ACTIVE_BATCHES - active)


def recheck_source_lease_inventory(
    lease: Mapping[str, Any],
    current_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    bindings = lease.get("group_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("source lease group bindings are missing")
    indices: list[int] = []
    for binding in bindings:
        if not isinstance(binding, Mapping):
            raise ValueError("source lease binding is invalid")
        uuid = str(binding.get("gpu_uuid") or "")
        current = current_inventory.get(uuid)
        if (
            not isinstance(current, Mapping)
            or current.get("physical_index") != binding.get("physical_index")
            or current.get("model") != binding.get("gpu_model")
            or current.get("model") not in H800_MODELS
        ):
            raise ValueError("source lease UUID/index/model drift")
        indices.append(int(current["physical_index"]))
    if len(set(indices)) != len(indices):
        raise ValueError("source lease physical index is duplicate")
    return copy.deepcopy(dict(lease))


def ready_source_uuids(
    inventory: Mapping[str, Any],
    *,
    models: Mapping[str, str],
    reservations: Mapping[str, Sequence[str]],
    fcooper_gpu7_active: bool,
) -> tuple[str, ...]:
    ready: list[str] = []
    for uuid, snapshot in sorted(inventory.items(), key=lambda item: item[1].index):
        if (
            models.get(uuid) not in H800_MODELS
            or uuid in reservations
            or getattr(snapshot, "memory_used_mib", 1) != 0
            or getattr(snapshot, "utilization_percent", 1) != 0
            or tuple(getattr(snapshot, "compute_pids", ()))
            or getattr(snapshot, "index", -1) == 7
        ):
            continue
        ready.append(uuid)
    return tuple(ready)


def classify_source_failure(reason: str) -> str:
    if reason in {
        "uuid_index_drift",
        "gpu_model_drift",
        "lease_unavailable",
        "foreign_process",
        "filesystem_drift",
    }:
        return "infrastructure_unavailable"
    if reason in {"interrupted", "host_shutdown"}:
        return "source_interrupted"
    if reason in {"child_nonzero", "evidence_missing"}:
        return "evidence_unavailable"
    return "evidence_invalid"


def advance_source_prelease(
    *,
    validate_ready_evidence: Callable[[], Mapping[str, Any]],
    lease_factory: Callable[[], Any | None],
    launch_wrapper: Callable[[Any], Any],
) -> dict[str, Any]:
    """Use complete evidence before acquiring any lease or launching a child."""
    try:
        formal = validate_ready_evidence()
    except (OSError, ValueError):
        formal = None
    if formal is not None:
        result_sha = str(formal.get("source_resolution_result_sha256") or "")
        if (
            formal.get("schema_version") != "stage7_source_resolution_result_v2"
            or formal.get("row_count") != 4
            or formal.get("eligible_for_exact_cache_reveal") is not True
            or len(result_sha) != 64
            or any(character not in "0123456789abcdef" for character in result_sha)
        ):
            raise ValueError("validated ready source evidence is not formal/atomic")
        return {
            "state": "SOURCE_READY",
            "fast_path": True,
            "source_resolution_result_sha256": result_sha,
        }
    lease = lease_factory()
    if lease is None:
        return {
            "state": "SOURCE_LEASE_PENDING",
            "fast_path": False,
            "lease_acquired": False,
        }
    launch_wrapper(lease)
    return {
        "state": "SOURCE_MATERIALIZING",
        "fast_path": False,
        "lease_acquired": True,
    }


__all__ = [
    "acquire_source_then_uuid_locks",
    "advance_source_prelease",
    "available_batch_slots",
    "classify_source_failure",
    "detect_round_state",
    "measurement_ready",
    "required_deployment_pins",
    "sha256_file",
    "validate_dependency_pins",
    "validate_deploy_manifest_pins",
    "validate_measurement_artifacts",
    "ready_source_uuids",
    "recheck_source_lease_inventory",
    "require_measurement_ready",
    "validate_zero_budget_retry",
]
