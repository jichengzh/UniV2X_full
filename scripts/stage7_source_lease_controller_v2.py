#!/usr/bin/env python3
"""Synchronous source-only lease controller for Stage7 core ablation v2.

This module owns neither source materialization nor exact-cache semantics.  It
only authenticates the frozen source projection, acquires resource locks in
source-before-UUID order, binds a live H800 lease to the persistent
orchestrator PID, invokes the existing source resolver, and hands its staged
formal result to the existing bind/reveal API.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from framework.stage7.core_ablation_v2 import CORE_VARIANTS
from framework.stage7 import source_execution_v2 as source_execution
from framework.stage7 import source_resolution_v2 as source_resolution
from scripts import stage7_source_scheduler_v2 as source_scheduler


JSON = dict[str, Any]
EXACT_ARTIFACTS = (
    "exact_selection_binding.json",
    "cache_snapshot_before_reveal.json",
    "cache_reveal.json",
    "miss_only_physical_request.json",
    "executor_admission.json",
)
FROZEN_SEEDS = frozenset({20260718, 20260719, 20260720})
FROZEN_ROUNDS = frozenset(range(4))
PRE_GATE_AUTHORIZATION_SCHEMA = "stage7_pre_gate_source_authorization_v2"
PRE_GATE_AUTHORIZED_ACTION = "source_only_materialization_before_no_gpu_gate"


class SourceLeaseContractError(RuntimeError):
    """A fail-closed source identity, recovery, or lease error."""


SchedulerFactory = Callable[[Path], Any]
Resolver = Callable[..., Mapping[str, Any]]
Binder = Callable[..., Mapping[str, Any]]
FormalProbe = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Optional[Mapping[str, Any]]
]
FormalValidator = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class SourceLeaseDependencies:
    scheduler_factory: SchedulerFactory
    resolver: Resolver
    binder: Binder
    formal_result_probe: FormalProbe
    formal_result_validator: FormalValidator


def _read_mapping(path: Path, *, label: str) -> JSON:
    if not path.is_file() or path.is_symlink():
        raise SourceLeaseContractError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceLeaseContractError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping):
        raise SourceLeaseContractError(f"{label} must be a JSON object")
    return copy.deepcopy(dict(payload))


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.is_file():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise SourceLeaseContractError(
                f"immutable source artifact conflict: {path}"
            )
        return
    if path.exists():
        raise SourceLeaseContractError(f"unsafe source artifact path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _ensure_safe_directory(
    path: Path, *, allowed_root: Path, label: str
) -> Path:
    root = Path(allowed_root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        raise SourceLeaseContractError(f"{label} is not absolute")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise SourceLeaseContractError(f"{label} escapes its allowed root") from error
    if ".." in relative.parts:
        raise SourceLeaseContractError(f"{label} escapes its allowed root")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SourceLeaseContractError(f"{label} contains a symlink")
        if current.exists():
            if not current.is_dir():
                raise SourceLeaseContractError(f"{label} is not a directory")
        else:
            try:
                current.mkdir()
            except OSError as error:
                raise SourceLeaseContractError(
                    f"{label} cannot be created safely"
                ) from error
        if current.is_symlink():
            raise SourceLeaseContractError(f"{label} contains a symlink")
    try:
        current.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise SourceLeaseContractError(f"{label} escapes its allowed root") from error
    return current


def _require_no_synthetic_round_artifacts(directory: Path) -> None:
    if any(directory.glob("synthetic_*")):
        raise SourceLeaseContractError(
            "formal source resolution cannot reuse synthetic round artifacts"
        )


def _sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _round_dir(root: Path, variant: str, seed: int, round_index: int) -> Path:
    if (
        variant not in CORE_VARIANTS
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in FROZEN_SEEDS
        or isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or round_index not in FROZEN_ROUNDS
    ):
        raise SourceLeaseContractError("source round identity is invalid")
    return root / "variants" / variant / f"seed_{seed}" / f"round_{round_index:02d}"


def _require_no_gpu_gate(root: Path) -> JSON:
    receipt = _read_mapping(
        root / "audits/no_gpu_dry_run/dry_run_receipt.json",
        label="no-GPU gate receipt",
    )
    if (
        receipt.get("formal_v2_gpu_jobs_launched") != 0
        or receipt.get("canonical_round0_requests_written") != 4
    ):
        raise SourceLeaseContractError("no-GPU gate has not passed")
    return receipt


def _require_pre_gate_source_authorization(root: Path) -> JSON:
    authorization = _read_mapping(
        root / "contracts/pre_gate_source_authorization.json",
        label="pre-gate source authorization",
    )
    recorded = authorization.pop("authorization_sha256", None)
    if (
        authorization.get("schema_version") != PRE_GATE_AUTHORIZATION_SCHEMA
        or authorization.get("authorized_action") != PRE_GATE_AUTHORIZED_ACTION
        or authorization.get("v2_root") != str(root)
        or authorization.get("gpu7_excluded") is not True
        or authorization.get("performance_measurement_allowed") is not False
        or authorization.get("ap_allowed") is not False
        or authorization.get("cache_reveal_allowed") is not False
        or authorization.get("feedback_allowed") is not False
        or authorization.get("selected_event_budget_delta") != 0
        or not isinstance(recorded, str)
        or recorded != _sha(authorization)
    ):
        raise SourceLeaseContractError("pre-gate source authorization drift")
    return {**authorization, "authorization_sha256": recorded}


def _require_pre_gate_source_paths(root: Path, plan: Mapping[str, Any]) -> None:
    allowed_root = (root / "sources").resolve()
    rows = plan.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SourceLeaseContractError("pre-gate source plan rows are missing")
    path_fields = (
        "source_evidence_path",
        "checkpoint_path",
        "checkpoint_allowed_root",
        "onnx_path",
        "calibration_path",
        "calibration_summary_path",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            raise SourceLeaseContractError("pre-gate source plan row is invalid")
        for field in path_fields:
            value = row.get(field)
            if value is None and field in {"checkpoint_path", "checkpoint_allowed_root"}:
                continue
            if not isinstance(value, str) or not value:
                raise SourceLeaseContractError(
                    f"pre-gate source path is missing: {field}"
                )
            try:
                Path(value).resolve().relative_to(allowed_root)
            except ValueError as error:
                raise SourceLeaseContractError(
                    "pre-gate source path escapes the v2 source root"
                ) from error


def _load_source_projection(directory: Path) -> tuple[bytes, JSON, JSON, JSON]:
    request_path = directory / "logical_request.json"
    if not request_path.is_file() or request_path.is_symlink():
        raise SourceLeaseContractError("logical request is missing or unsafe")
    request_bytes = request_path.read_bytes()
    binding = _read_mapping(
        directory / "selection_binding.json", label="selection binding"
    )
    plan = _read_mapping(
        directory / "source_resolution_plan.json", label="source resolution plan"
    )
    try:
        execution = source_execution.build_source_execution_plan(
            request_bytes, binding, plan
        )
        source_execution.validate_source_execution_plan(
            execution, request_bytes, binding, plan
        )
    except ValueError as error:
        raise SourceLeaseContractError("source execution projection drift") from error
    _write_immutable(directory / "source_execution_plan.json", execution)
    return request_bytes, binding, plan, execution


def _default_formal_probe(
    plan: Mapping[str, Any], execution: Mapping[str, Any]
) -> Optional[JSON]:
    try:
        result = source_resolution.build_formal_source_resolution_result(
            plan,
            evidence_paths_by_candidate=source_execution.candidate_evidence_paths(
                execution
            ),
        )
        return source_resolution.validate_formal_source_resolution_result(result, plan)
    except (OSError, ValueError):
        return None


def _default_scheduler_factory(root: Path) -> Any:
    from scripts.stage7_core_ablation_scheduler_v2 import V2Stage7Scheduler

    return V2Stage7Scheduler(
        status_dir=root / "status/source_scheduler_v2",
        base_env=os.environ,
    )


def _default_dependencies() -> SourceLeaseDependencies:
    from scripts import stage7_core_online_ablation_v2 as online
    from scripts import stage7_resolve_round_sources_v2 as resolver

    return SourceLeaseDependencies(
        scheduler_factory=_default_scheduler_factory,
        resolver=resolver.run_source_resolution,
        binder=online.bind_reveal_after_source_ready,
        formal_result_probe=_default_formal_probe,
        formal_result_validator=source_resolution.validate_formal_source_resolution_result,
    )


def _identity_fields(payload: Mapping[str, Any]) -> tuple[str, str]:
    request_sha = payload.get("logical_request_sha256")
    plan_sha = payload.get("source_resolution_plan_sha256")
    if (
        not isinstance(request_sha, str)
        or len(request_sha) != 64
        or not isinstance(plan_sha, str)
        or len(plan_sha) != 64
    ):
        raise SourceLeaseContractError("source request/plan identity is missing")
    return request_sha, plan_sha


def _retry_payload(execution: Mapping[str, Any], *, status: str, reason: str) -> JSON:
    return {
        "status": status,
        "reason": reason,
        "logical_request_sha256": execution["logical_request_sha256"],
        "source_resolution_plan_sha256": execution["source_resolution_plan_sha256"],
        "selected_event_budget_delta": 0,
        "partial_reveal_allowed": False,
        "cache_membership_observed": False,
    }


def _validated_existing_canonical(
    directory: Path,
    plan: Mapping[str, Any],
    validator: FormalValidator,
) -> Optional[JSON]:
    path = directory / "source_resolution_result.json"
    if not path.is_file():
        return None
    try:
        return copy.deepcopy(
            dict(validator(_read_mapping(path, label="canonical source result"), plan))
        )
    except (TypeError, ValueError) as error:
        raise SourceLeaseContractError(
            "canonical source result authentication failed"
        ) from error


def _validate_retry_receipt(
    receipt: Mapping[str, Any],
    *,
    retry: Mapping[str, Any],
    lease: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> JSON:
    copied = copy.deepcopy(dict(receipt))
    recorded = copied.pop("receipt_sha256", None)
    if not isinstance(recorded, str) or recorded != _sha(copied):
        raise SourceLeaseContractError("source retry receipt signature drift")
    retry_path = copied.get("retry_path")
    if (
        copied.get("schema_version") != "stage7_source_execution_receipt_v2"
        or copied.get("status") != "retry_required"
        or copied.get("logical_request_sha256") != execution["logical_request_sha256"]
        or copied.get("source_resolution_plan_sha256")
        != execution["source_resolution_plan_sha256"]
        or copied.get("source_execution_plan_sha256")
        != execution["source_execution_plan_sha256"]
        or copied.get("source_gpu_lease_sha256") != lease["source_gpu_lease_sha256"]
        or copied.get("source_attempt_sha256") != lease["source_attempt_sha256"]
        or copied.get("source_resolution_retry_sha256")
        != retry["source_resolution_retry_sha256"]
        or copied.get("retry_reason_code") != retry["reason_code"]
        or copied.get("selected_event_budget_delta") != 0
        or copied.get("partial_reveal_allowed") is not False
        or copied.get("formal_result_written") is not False
        or copied.get("source_result_path") is not None
        or copied.get("eligible_for_exact_cache_reveal") is not False
        or copied.get("eligible_for_cache_append") is not False
        or copied.get("eligible_for_finalization") is not False
        or not isinstance(retry_path, str)
        or not retry_path
    ):
        raise SourceLeaseContractError("source retry receipt contract drift")
    return {**copied, "receipt_sha256": recorded}


def _validate_terminal_retry_attempt(
    attempt: Path,
    *,
    attempt_index: int,
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> JSON:
    lease_path = attempt / "source_gpu_lease.json"
    retry_path = attempt / "source_resolution_retry.json"
    receipt_path = attempt / "source_execution_receipt.json"
    required = (lease_path, retry_path, receipt_path)
    if not all(path.is_file() and not path.is_symlink() for path in required):
        raise SourceLeaseContractError(
            "existing live lease/attempt takeover is forbidden without terminal retry"
        )
    try:
        lease = source_execution.validate_source_gpu_lease(
            _read_mapping(lease_path, label="source GPU lease"), execution
        )
        retry = source_resolution.validate_source_resolution_retry_result(
            _read_mapping(retry_path, label="source retry"), plan
        )
    except ValueError as error:
        raise SourceLeaseContractError(
            "existing source retry authentication failed"
        ) from error
    raw_receipt = _read_mapping(receipt_path, label="source execution receipt")
    receipt = _validate_retry_receipt(
        raw_receipt, retry=retry, lease=lease, execution=execution
    )
    recorded_retry = Path(str(receipt["retry_path"])).resolve()
    if (
        lease.get("attempt_index") != attempt_index
        or recorded_retry != retry_path.resolve()
        or any(
            (attempt / name).exists()
            for name in (
                "source_resolution_result.staged.json",
                "source_resolution_result.json",
            )
        )
        or any((attempt.parents[1] / name).exists() for name in EXACT_ARTIFACTS)
    ):
        raise SourceLeaseContractError("existing source retry attempt identity drift")
    return {"lease": lease, "retry": retry, "receipt": receipt}


def _attempt_index(
    directory: Path,
    *,
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> int:
    attempts_root = directory / "source_attempts"
    entries = sorted(attempts_root.glob("attempt_[0-9][0-9][0-9]"))
    if any(path.is_symlink() or not path.is_dir() for path in entries):
        raise SourceLeaseContractError(
            "source attempt directory contains an unsafe entry"
        )
    attempts = entries
    if not attempts:
        return 0
    for index, attempt in enumerate(attempts):
        if attempt.name != f"attempt_{index:03d}":
            raise SourceLeaseContractError("source attempt sequence has a gap")
        _validate_terminal_retry_attempt(
            attempt,
            attempt_index=index,
            plan=plan,
            execution=execution,
        )
    return len(attempts)


def _lock_owner(scheduler: Any, orchestrator_pid: int) -> JSON:
    identity = scheduler.process_inspector(orchestrator_pid)
    if (
        identity is None
        or getattr(identity, "alive", True) is not True
        or getattr(identity, "pid", None) != orchestrator_pid
        or getattr(identity, "owner", None) != scheduler.current_owner
        or not str(getattr(identity, "start_time", ""))
    ):
        raise SourceLeaseContractError("persistent orchestrator lock owner drift")
    return {
        "pid": orchestrator_pid,
        "owner": str(identity.owner),
        "start_time": str(identity.start_time),
    }


def _inventory(scheduler: Any, uuids: tuple[str, ...]) -> JSON:
    result: JSON = {}
    for uuid in uuids:
        snapshot = scheduler._last_snapshot.get(uuid)
        model = scheduler._gpu_models.get(uuid)
        if snapshot is None or not model:
            raise SourceLeaseContractError("post-lock H800 inventory is incomplete")
        result[uuid] = {
            "physical_index": int(snapshot.index),
            "model": str(model),
        }
    return result


def _release_locks(*groups: tuple[Any, ...]) -> None:
    for group in groups:
        for lock in reversed(group):
            lock.release()


def _bind_formal_result(
    *,
    root: Path,
    repo_root: Path,
    directory: Path,
    variant: str,
    seed: int,
    round_index: int,
    staged_path: Path,
    plan: Mapping[str, Any],
    dependencies: SourceLeaseDependencies,
) -> JSON:
    if staged_path.is_symlink():
        raise SourceLeaseContractError("staged source result cannot be a symlink")
    try:
        staged_resolved = staged_path.resolve(strict=True)
        staged_resolved.relative_to(directory.resolve())
    except (OSError, ValueError) as error:
        raise SourceLeaseContractError(
            "staged source result escapes its round"
        ) from error
    try:
        staged = dependencies.formal_result_validator(
            _read_mapping(staged_resolved, label="staged source result"), plan
        )
    except (TypeError, ValueError) as error:
        raise SourceLeaseContractError(
            "staged source result authentication failed"
        ) from error
    receipt = copy.deepcopy(
        dict(
            dependencies.binder(
                root,
                variant=variant,
                seed=seed,
                round_index=round_index,
                source_result_path=staged_resolved,
                synthetic_no_gpu_dryrun=False,
                repo_root=repo_root,
            )
        )
    )
    if receipt.get("controller_state") != "CACHE_REVEALED":
        raise SourceLeaseContractError(
            "staged-to-canonical bind did not close cache reveal"
        )
    canonical = _validated_existing_canonical(
        directory, plan, dependencies.formal_result_validator
    )
    if canonical is None or canonical != staged:
        raise SourceLeaseContractError(
            "canonical source result differs from staged result"
        )
    if not all((directory / name).is_file() for name in EXACT_ARTIFACTS):
        raise SourceLeaseContractError(
            "cache reveal artifacts are incomplete after source bind"
        )
    return receipt


def _commit_pre_gate_source_result(
    *,
    root: Path,
    directory: Path,
    staged_path: Path,
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
    dependencies: SourceLeaseDependencies,
    source_gpu_jobs_launched: int,
    source_attempt_index: int | None,
    source_gpu_lease_sha256: str | None,
    fast_path: bool,
) -> JSON:
    if any((directory / name).exists() for name in EXACT_ARTIFACTS):
        raise SourceLeaseContractError(
            "pre-gate source-only mode found forbidden cache-reveal artifacts"
        )
    if staged_path.is_symlink():
        raise SourceLeaseContractError("staged source result cannot be a symlink")
    try:
        staged_resolved = staged_path.resolve(strict=True)
        staged_resolved.relative_to(directory.resolve())
    except (OSError, ValueError) as error:
        raise SourceLeaseContractError(
            "staged source result escapes its round"
        ) from error
    try:
        formal = copy.deepcopy(
            dict(
                dependencies.formal_result_validator(
                    _read_mapping(staged_resolved, label="staged source result"),
                    plan,
                )
            )
        )
    except (TypeError, ValueError) as error:
        raise SourceLeaseContractError(
            "staged source result authentication failed"
        ) from error
    canonical = directory / "source_resolution_result.json"
    _write_immutable(canonical, formal)
    validated = _validated_existing_canonical(
        directory, plan, dependencies.formal_result_validator
    )
    if validated != formal:
        raise SourceLeaseContractError(
            "canonical source result differs from staged result"
        )
    if any((directory / name).exists() for name in EXACT_ARTIFACTS):
        raise SourceLeaseContractError(
            "pre-gate source-only mode revealed cache artifacts"
        )
    unsigned = {
        "schema_version": "stage7_pre_gate_source_only_receipt_v2",
        "status": "source_ready",
        "logical_request_sha256": execution["logical_request_sha256"],
        "source_resolution_plan_sha256": execution[
            "source_resolution_plan_sha256"
        ],
        "source_resolution_result_sha256": formal.get(
            "source_resolution_result_sha256"
        ),
        "source_attempt_index": source_attempt_index,
        "source_gpu_lease_sha256": source_gpu_lease_sha256,
        "fast_path": fast_path,
        "pre_gate_source_gpu_jobs_launched": source_gpu_jobs_launched,
        "performance_measurement_jobs_launched": 0,
        "ap_jobs_launched": 0,
        "cache_membership_observed": False,
        "cache_reveal_allowed": False,
        "feedback_released": False,
        "selected_event_budget_delta": 0,
        "gpu7_excluded": True,
        "source_root": str((root / "sources").resolve()),
    }
    receipt = {**unsigned, "receipt_sha256": _sha(unsigned)}
    _write_immutable(directory / "pre_gate_source_only_receipt.json", receipt)
    return receipt


def run_source_lease_controller(
    *,
    v2_root: Path,
    repo_root: Path,
    variant: str,
    seed: int,
    round_index: int,
    orchestrator_pid: int,
    dependencies: Optional[SourceLeaseDependencies] = None,
    pre_gate_source_only: bool = False,
) -> JSON:
    """Advance one SOURCE_PLAN_FROZEN round without changing its selected rows."""
    if (
        isinstance(orchestrator_pid, bool)
        or not isinstance(orchestrator_pid, int)
        or orchestrator_pid <= 0
    ):
        raise SourceLeaseContractError("persistent orchestrator PID is invalid")
    root = Path(v2_root).resolve()
    frozen_repo = Path(repo_root).resolve()
    if pre_gate_source_only:
        _require_pre_gate_source_authorization(root)
    else:
        _require_no_gpu_gate(root)
    directory = _round_dir(root, variant, seed, round_index)
    _, _, plan, execution = _load_source_projection(directory)
    _require_no_synthetic_round_artifacts(directory)
    if pre_gate_source_only:
        _require_pre_gate_source_paths(root, plan)
    deps = dependencies or _default_dependencies()
    request_sha, plan_sha = _identity_fields(execution)

    canonical = _validated_existing_canonical(
        directory, plan, deps.formal_result_validator
    )
    if canonical is not None:
        if pre_gate_source_only:
            return _commit_pre_gate_source_result(
                root=root,
                directory=directory,
                staged_path=directory / "source_resolution_result.json",
                plan=plan,
                execution=execution,
                dependencies=deps,
                source_gpu_jobs_launched=0,
                source_attempt_index=None,
                source_gpu_lease_sha256=None,
                fast_path=True,
            )
        if all((directory / name).is_file() for name in EXACT_ARTIFACTS):
            return {
                "status": "cache_revealed",
                "fast_path": True,
                "logical_request_sha256": request_sha,
                "source_resolution_plan_sha256": plan_sha,
                "selected_event_budget_delta": 0,
            }
        receipt = _bind_formal_result(
            root=root,
            repo_root=frozen_repo,
            directory=directory,
            variant=variant,
            seed=seed,
            round_index=round_index,
            staged_path=directory / "source_resolution_result.json",
            plan=plan,
            dependencies=deps,
        )
        return {"status": "cache_revealed", "fast_path": True, **receipt}

    prelease = deps.formal_result_probe(plan, execution)
    if prelease is not None:
        try:
            formal = deps.formal_result_validator(prelease, plan)
        except (TypeError, ValueError) as error:
            raise SourceLeaseContractError(
                "prelease formal source evidence is invalid"
            ) from error
        staged = directory / "source_resolution_result.prelease.staged.json"
        _write_immutable(staged, formal)
        if pre_gate_source_only:
            return _commit_pre_gate_source_result(
                root=root,
                directory=directory,
                staged_path=staged,
                plan=plan,
                execution=execution,
                dependencies=deps,
                source_gpu_jobs_launched=0,
                source_attempt_index=None,
                source_gpu_lease_sha256=None,
                fast_path=True,
            )
        receipt = _bind_formal_result(
            root=root,
            repo_root=frozen_repo,
            directory=directory,
            variant=variant,
            seed=seed,
            round_index=round_index,
            staged_path=staged,
            plan=plan,
            dependencies=deps,
        )
        return {"status": "cache_revealed", "fast_path": True, **receipt}

    attempts_root = _ensure_safe_directory(
        directory / "source_attempts",
        allowed_root=directory,
        label="source attempt directory",
    )
    attempt_index = _attempt_index(directory, plan=plan, execution=execution)
    attempt_target = attempts_root / f"attempt_{attempt_index:03d}"
    if attempt_target.is_symlink() or attempt_target.exists():
        raise SourceLeaseContractError(
            "source attempt directory contains an unsafe entry"
        )
    scheduler = deps.scheduler_factory(root)
    scheduler.observe_gpus()
    ready = tuple(scheduler._ready_uuids())
    jobs = execution.get("group_jobs")
    if not isinstance(jobs, list) or not jobs:
        raise SourceLeaseContractError("source execution group jobs are missing")
    required = len(jobs)
    if len(ready) < required:
        return _retry_payload(
            execution,
            status="waiting_for_source_gpu_capacity",
            reason=f"need {required} eligible unreserved H800 UUIDs; observed {len(ready)}",
        )
    selected = tuple(ready[:required])
    source_keys = tuple(f"source:{job['source_group_execution_key']}" for job in jobs)
    acquired = source_scheduler.acquire_source_then_uuid_locks(
        source_keys,
        selected,
        scheduler.resource_lock_factory,
        scheduler.lock_factory,
    )
    if acquired is None:
        return _retry_payload(
            execution,
            status="waiting_for_source_gpu_capacity",
            reason="source or UUID lock unavailable",
        )
    source_locks, uuid_locks = acquired
    try:
        if not scheduler._post_lock_runtime_admission(selected):
            return _retry_payload(
                execution,
                status="waiting_for_source_gpu_capacity",
                reason="post-lock H800 occupancy/reservation drift",
            )
        owner = _lock_owner(scheduler, orchestrator_pid)
        inventory = _inventory(scheduler, selected)
        assignments = {
            str(job["source_group_execution_key"]): uuid
            for job, uuid in zip(jobs, selected)
        }
        lease = source_execution.build_source_gpu_lease(
            execution,
            hostname=str(scheduler.hostname_probe()),
            inventory=inventory,
            assignments=assignments,
            attempt_index=attempt_index,
            lock_owner=owner,
            execution_mode="formal",
            source_locks_held=True,
        )
        source_execution.validate_source_gpu_lease(lease, execution)
        attempt_dir = _ensure_safe_directory(
            attempt_target,
            allowed_root=attempts_root,
            label="source attempt directory",
        )
        lease_path = attempt_dir / "source_gpu_lease.json"
        _write_immutable(lease_path, lease)
        result = copy.deepcopy(
            dict(
                deps.resolver(
                    logical_request_path=directory / "logical_request.json",
                    selection_binding_path=directory / "selection_binding.json",
                    source_plan_path=directory / "source_resolution_plan.json",
                    source_lease_path=lease_path,
                    output_dir=attempt_dir,
                    no_gpu_dryrun=False,
                    pre_gate_source_only=pre_gate_source_only,
                    advertised_gpu_uuids=selected,
                    environment={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": ",".join(selected),
                    },
                )
            )
        )
        if result.get("status") == "retry_required":
            terminal = _validate_terminal_retry_attempt(
                attempt_dir,
                attempt_index=attempt_index,
                plan=plan,
                execution=execution,
            )
            receipt = terminal["receipt"]
            retry = terminal["retry"]
            result_request, result_plan = _identity_fields(receipt)
            if (
                result_request != request_sha
                or result_plan != plan_sha
                or receipt.get("selected_event_budget_delta") != 0
                or result.get("logical_request_sha256") != request_sha
                or result.get("source_resolution_plan_sha256") != plan_sha
                or result.get("selected_event_budget_delta") != 0
                or result.get("source_resolution_retry_sha256")
                != retry["source_resolution_retry_sha256"]
                or result.get("receipt_sha256") != receipt["receipt_sha256"]
            ):
                raise SourceLeaseContractError(
                    "source retry changed identity or budget"
                )
            return result
        if result.get("status") != "source_ready":
            raise SourceLeaseContractError("source resolver returned an unknown status")
        staged_value = result.get("source_result_path")
        if not isinstance(staged_value, str) or not staged_value:
            raise SourceLeaseContractError("source resolver omitted staged result path")
        if pre_gate_source_only:
            receipt = _commit_pre_gate_source_result(
                root=root,
                directory=directory,
                staged_path=Path(staged_value),
                plan=plan,
                execution=execution,
                dependencies=deps,
                source_gpu_jobs_launched=required,
                source_attempt_index=attempt_index,
                source_gpu_lease_sha256=lease["source_gpu_lease_sha256"],
                fast_path=False,
            )
            return {
                "status": "source_ready",
                **receipt,
            }
        receipt = _bind_formal_result(
            root=root,
            repo_root=frozen_repo,
            directory=directory,
            variant=variant,
            seed=seed,
            round_index=round_index,
            staged_path=Path(staged_value),
            plan=plan,
            dependencies=deps,
        )
        return {
            "status": "cache_revealed",
            "fast_path": False,
            "source_attempt_index": attempt_index,
            "source_gpu_lease_sha256": lease["source_gpu_lease_sha256"],
            **receipt,
        }
    finally:
        _release_locks(uuid_locks, source_locks)


__all__ = [
    "SourceLeaseContractError",
    "SourceLeaseDependencies",
    "run_source_lease_controller",
]
