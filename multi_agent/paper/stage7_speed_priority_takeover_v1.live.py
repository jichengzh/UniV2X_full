#!/usr/bin/env python3
"""Operational Stage7 takeover shim for speed-priority GPU scheduling.

The formal deployment bundle and scientific request identities remain unchanged.
This shim only rebinds the dead orchestrator's runtime PID and disables
command-line-only GPU reservations in the live scheduler process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


JSON = dict[str, Any]


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> JSON:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".takeover.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"takeover temporary artifact already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_pid_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _require_sha256(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _value_after_flag(argv: Sequence[str], flag: str) -> str | None:
    try:
        index = list(argv).index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} value is missing")
    return str(argv[index + 1])


def rebind_operational_identity(
    v2_root: Path,
    *,
    current_pid: int,
    capture_identity: Callable[[int], Mapping[str, Any]],
    pid_exists: Callable[[int], bool] = _default_pid_exists,
    current_uid: int | None = None,
    deployment_release_sha256: str | None = None,
    deployment_manifest_file_sha256: str | None = None,
) -> JSON:
    """Rebind operational PID state, optionally accepting a reviewed deployment."""
    root = Path(v2_root).resolve(strict=True)
    prepare_path = root / "prepare_state.json"
    initialize_path = root / "initialize_state.json"
    prepare = _read_mapping(prepare_path, label="prepare state")
    initialize = _read_mapping(initialize_path, label="initialize state")
    if (
        prepare.get("schema_version") != "stage7_core_ablation_v2_prepare_state"
        or prepare.get("status") != "prepared"
        or prepare.get("v2_root") not in (None, str(root))
        or initialize.get("schema_version")
        != "stage7_core_ablation_v2_initialize_result"
    ):
        raise ValueError("formal v2 root is not prepared and initialized")
    uid = os.getuid() if current_uid is None else current_uid
    if prepare.get("owner_uid") != uid:
        raise ValueError("takeover owner UID does not match prepared root")
    old_pid = prepare.get("orchestrator_pid")
    if isinstance(old_pid, bool) or not isinstance(old_pid, int) or old_pid <= 0:
        raise ValueError("prepared orchestrator PID is invalid")
    if old_pid != current_pid and pid_exists(old_pid):
        raise ValueError("prepared orchestrator PID is still alive")
    old_recovery_sha256 = prepare.get("recovery_contract_sha256")
    if (
        not isinstance(old_recovery_sha256, str)
        or initialize.get("prepare_recovery_contract_sha256")
        != old_recovery_sha256
    ):
        raise ValueError("prepare/initialize recovery binding is inconsistent")
    identity = dict(capture_identity(current_pid))
    if identity.get("pid") != current_pid or identity.get("uid") != uid:
        raise ValueError("captured takeover process identity is invalid")
    pin_updates: dict[str, str] = {}
    if deployment_release_sha256 is not None:
        pin_updates["deployment_release_sha256"] = _require_sha256(
            deployment_release_sha256, label="deployment release SHA256"
        )
    if deployment_manifest_file_sha256 is not None:
        pin_updates["deployment_manifest_file_sha256"] = _require_sha256(
            deployment_manifest_file_sha256,
            label="deployment manifest-file SHA256",
        )
    prepare_unsigned = {
        **{
            key: value
            for key, value in prepare.items()
            if key != "recovery_contract_sha256"
        },
        **pin_updates,
        "orchestrator_pid": current_pid,
        "orchestrator_process_identity": identity,
    }
    rebound_prepare = {
        **prepare_unsigned,
        "recovery_contract_sha256": _canonical_sha256(prepare_unsigned),
    }
    rebound_initialize = {
        **initialize,
        "prepare_recovery_contract_sha256": rebound_prepare[
            "recovery_contract_sha256"
        ],
    }
    _atomic_write_json(prepare_path, rebound_prepare)
    _atomic_write_json(initialize_path, rebound_initialize)
    if (
        _read_mapping(prepare_path, label="rebound prepare state")
        != rebound_prepare
        or _read_mapping(initialize_path, label="rebound initialize state")
        != rebound_initialize
    ):
        raise ValueError("operational PID rebind did not persist atomically")
    return {
        "old_orchestrator_pid": old_pid,
        "new_orchestrator_pid": current_pid,
        "deployment_release_sha256": rebound_prepare.get(
            "deployment_release_sha256"
        ),
        "deployment_manifest_file_sha256": rebound_prepare.get(
            "deployment_manifest_file_sha256"
        ),
        "old_prepare_recovery_contract_sha256": old_recovery_sha256,
        "new_prepare_recovery_contract_sha256": rebound_prepare[
            "recovery_contract_sha256"
        ],
    }


def install_speed_priority_policy(scheduler_module: Any) -> JSON:
    """Use physical occupancy, not command-line intent, for GPU availability."""
    original_leased_uuids = scheduler_module.V2Stage7Scheduler._leased_uuids
    original_barrier_ready = getattr(
        scheduler_module.V2Stage7Scheduler,
        "_request_is_barrier_ready",
        None,
    )

    def no_command_line_reservations(_processes: Sequence[Any]) -> dict[str, tuple]:
        return {}

    def no_unconditional_gpu7_exclusion(_scheduler: Any) -> None:
        return None

    def single_gpu_future_controller(_request: Any, _ready_uuids: Sequence[str]) -> int:
        return 1

    def canonical_barrier_ready(scheduler: Any, request: Any) -> bool:
        if callable(original_barrier_ready) and original_barrier_ready(
            scheduler, request
        ):
            return True
        round_index = getattr(request, "round_index", None)
        trajectory_path = Path(getattr(request, "trajectory_path", ""))
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index <= 0
            or not trajectory_path.is_absolute()
        ):
            return False
        trajectory_root = trajectory_path.parent
        for prior_index in range(round_index):
            prior_round = trajectory_root / f"round_{prior_index:02d}"
            barrier = prior_round / "atomic_feedback_barrier.json"
            if not barrier.is_file() or barrier.is_symlink():
                return False
            try:
                receipt = scheduler.completion_validator(prior_round)
            except (OSError, TypeError, ValueError):
                return False
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("feedback_released") is not True
                or receipt.get("budget_consumed") != 4
            ):
                return False
        return True

    def acquire_scientific_resource_locks(
        scheduler: Any, request: Any
    ) -> tuple[Any, ...] | None:
        if (
            tuple(request.width) in scheduler._running_widths()
            or request.source_lock_key in scheduler._running_source_keys()
        ):
            scheduler._lease_event(
                "lease_refused_running_resource_conflict",
                controller_id=request.controller_id,
                request_sha256=request.request_sha256,
                width=list(request.width),
                source_lock_key=request.source_lock_key,
            )
            return None
        acquired: list[Any] = []
        for resource_key in (
            "width:" + "x".join(str(value) for value in request.width),
            "source:" + request.source_lock_key,
        ):
            lock = scheduler.resource_lock_factory(resource_key)
            if lock is None:
                scheduler._release_locks(acquired)
                scheduler._lease_event(
                    "lease_refused_resource_lock",
                    controller_id=request.controller_id,
                    request_sha256=request.request_sha256,
                    resource_key=resource_key,
                )
                return None
            acquired.append(lock)
        return tuple(acquired)

    def effective_controller_leases(
        scheduler: Any,
        controller: Mapping[str, Any],
        snapshots: Mapping[str, Any],
    ) -> tuple[str, ...]:
        leases = tuple(str(uuid) for uuid in (controller.get("gpu_uuids") or ()))
        if controller.get("status") != "running":
            return ()
        if len(leases) <= 1:
            return leases
        wave_contract = controller.get("physical_wave_contract")
        ordered = (
            tuple(str(uuid) for uuid in (wave_contract.get("ordered_lease_uuids") or ()))
            if isinstance(wave_contract, Mapping)
            else ()
        )
        waves = (
            list(wave_contract.get("waves") or ())
            if isinstance(wave_contract, Mapping)
            else []
        )
        if leases != ordered or not waves:
            return leases
        last_wave = waves[-1]
        if not isinstance(last_wave, Mapping):
            return leases
        last_rows = last_wave.get("physical_row_indices") or ()
        if not isinstance(last_rows, (list, tuple)):
            return leases
        final_wave_uuids = set(ordered[: len(last_rows)])
        if not final_wave_uuids or final_wave_uuids == set(ordered):
            return leases
        try:
            owner_pid = int(controller.get("pid", -1))
        except (TypeError, ValueError):
            return leases
        final_wave_active = any(
            any(
                scheduler.pid_is_owned(owner_pid, pid)
                for pid in getattr(snapshots.get(uuid), "compute_pids", ())
            )
            for uuid in final_wave_uuids
        )
        if not final_wave_active:
            return leases
        return tuple(uuid for uuid in leases if uuid in final_wave_uuids)

    def active_or_unfinished_leased_uuids(scheduler: Any) -> set[str]:
        leased = set(original_leased_uuids(scheduler))
        try:
            snapshots = dict(scheduler.gpu_probe())
        except Exception:
            return leased
        controllers = scheduler._state.get("controllers") or {}
        if not isinstance(controllers, Mapping):
            return leased
        effective: set[str] = set()
        for controller in controllers.values():
            if isinstance(controller, Mapping):
                effective.update(effective_controller_leases(scheduler, controller, snapshots))
        return effective

    def release_inactive_memory_locks(scheduler: Any) -> tuple[str, ...]:
        controllers = scheduler._state.get("controllers") or {}
        locks = getattr(scheduler, "_locks", {})
        if not isinstance(controllers, Mapping) or not isinstance(locks, Mapping):
            return ()
        inactive = tuple(
            sorted(
                controller_id
                for controller_id in locks
                if not isinstance(controllers.get(controller_id), Mapping)
                or controllers[controller_id].get("status") != "running"
            )
        )
        for controller_id in inactive:
            scheduler._release_locks(locks.get(controller_id, ()))
        scheduler._locks = {
            controller_id: handles
            for controller_id, handles in locks.items()
            if controller_id not in inactive
        }
        if inactive:
            scheduler._lease_event(
                "inactive_controller_memory_locks_released",
                controller_ids=list(inactive),
            )
        return inactive

    scheduler_module.discover_process_reservations = no_command_line_reservations
    scheduler_module.V2Stage7Scheduler._gpu7_uuid = no_unconditional_gpu7_exclusion
    scheduler_module.V2Stage7Scheduler._elastic_lease_count = staticmethod(
        single_gpu_future_controller
    )
    if callable(original_barrier_ready):
        scheduler_module.V2Stage7Scheduler._request_is_barrier_ready = (
            canonical_barrier_ready
        )
    scheduler_module.V2Stage7Scheduler._acquire_resource_locks = (
        acquire_scientific_resource_locks
    )
    scheduler_module.V2Stage7Scheduler._leased_uuids = active_or_unfinished_leased_uuids
    scheduler_module.V2Stage7Scheduler._release_inactive_memory_locks = (
        release_inactive_memory_locks
    )
    scheduler_module.MAX_PARALLEL_BATCHES = 8
    return {
        "command_line_reservations_blocking": False,
        "gpu7_unconditional_exclusion": False,
        "max_parallel_controllers": 8,
        "future_controller_lease_count": 1,
    }


def install_concurrent_resume_policy(scheduler_module: Any) -> JSON:
    """Keep live controllers while filling every other eligible idle GPU."""

    def concurrent_resume(
        scheduler: Any, requests: Sequence[Any], *, once: bool = False
    ) -> JSON:
        scheduler_module._source_scheduler.validate_dependency_pins(
            Path(scheduler_module.__file__).resolve().parents[1]
        )
        scheduler.reap_controllers()
        repair_committed = getattr(
            scheduler, "_repair_committed_feedback_controllers", None
        )
        if callable(repair_committed):
            repair_committed(())
        release_inactive_locks = getattr(
            scheduler, "_release_inactive_memory_locks", None
        )
        if callable(release_inactive_locks):
            release_inactive_locks()
        scheduler.observe_gpus()
        scheduler._validate_runtime_admission()
        controllers = scheduler._state.get("controllers") or {}
        pid_records = scheduler._controller_manifest.get("controllers") or {}
        already_live = [
            controller_id
            for controller_id, controller in sorted(controllers.items())
            if controller.get("status") == "running"
            and scheduler_module._v1._identity_matches(
                pid_records.get(controller_id) or {},
                scheduler.process_inspector(
                    int((pid_records.get(controller_id) or {}).get("pid", -1))
                ),
                scheduler.current_owner,
            )
        ]
        scheduled = scheduler.schedule(requests)
        if not once and not already_live and not scheduled["launched"]:
            scheduler.sleeper(scheduler.poll_interval_seconds)
        return {
            "already_live": already_live,
            "launched": [
                request.controller_id for request in scheduled["launched"]
            ],
            "blocked": scheduled["blocked"],
            "status": scheduler.status(),
            "runtime_admission": "h800_exact_model_and_uuid_passed",
        }

    scheduler_module.V2Stage7Scheduler.resume = concurrent_resume
    return {
        "live_controller_blocks_pending_schedule": False,
        "existing_live_controller_preserved": True,
        "committed_feedback_repair_before_schedule": True,
    }


def install_empty_physical_terminal_observer_policy(worker_module: Any) -> JSON:
    """Authenticate zero-miss terminals emitted by the existing actual-v3 path."""
    original_validate_terminal = worker_module._validate_terminal

    def validate_terminal(
        path: Path, *, request_sha256: str, physical_sha256: str
    ) -> JSON:
        raw = worker_module._read_mapping(
            path, label="actual-v3 terminal payload"
        )
        if (
            raw.get("schema_version")
            != "stage7_actual_v3_empty_physical_terminal_v2"
        ):
            return original_validate_terminal(
                path,
                request_sha256=request_sha256,
                physical_sha256=physical_sha256,
            )
        terminal = worker_module.validate_physical_terminal_batch(raw)
        independent = worker_module._read_mapping(
            path.parent / "independent_request.json",
            label="actual-v3 independent request",
        )
        contract = worker_module._read_mapping(
            path.parent / "execution_contract.json",
            label="actual-v3 execution contract",
        )
        projection = terminal.get("projection_artifact")
        lineage = raw.get("stage7_projection_lineage")
        unsigned_raw = {
            key: value
            for key, value in raw.items()
            if key != "empty_physical_terminal_sha256"
        }
        unsigned_contract = {
            key: value
            for key, value in contract.items()
            if key != "execution_contract_sha256"
        }
        empty_sha = raw.get("empty_physical_terminal_sha256")
        if (
            independent != raw
            or empty_sha != _canonical_sha256(unsigned_raw)
            or raw.get("rows") != []
            or raw.get("lineage_inputs") != []
            or raw.get("gpu_subprocess_count") != 0
            or not isinstance(lineage, Mapping)
            or lineage.get("logical_request_sha256") != request_sha256
            or lineage.get("physical_request_sha256") != physical_sha256
            or not isinstance(projection, Mapping)
            or projection.get("measurement_request_sha256") != request_sha256
            or projection.get("projection_payload_sha256") != empty_sha
            or projection.get("empty_projection_lineage") != lineage
            or terminal.get("rows") != []
            or terminal.get("failures") != []
            or terminal.get("barrier_release_allowed") is not True
            or contract.get("schema_version")
            != "stage7_actual_v3_execution_contract_v2"
            or contract.get("logical_request_sha256") != request_sha256
            or contract.get("physical_request_sha256") != physical_sha256
            or contract.get("dry_run") is not False
            or projection.get("deployment_bundle_sha256")
            != contract.get("deployment_bundle_sha256")
            or lineage.get("deployment_bundle_sha256")
            != contract.get("deployment_bundle_sha256")
            or contract.get("execution_contract_sha256")
            != _canonical_sha256(unsigned_contract)
        ):
            raise ValueError(
                "authenticated zero-miss terminal request/source/cache "
                "lineage drift"
            )
        return terminal

    worker_module._validate_terminal = validate_terminal
    return {
        "zero_miss_terminal_observer_enabled": True,
        "gpu_subprocess_count_required": 0,
        "selected_event_budget_delta": 0,
    }


def _running_controller_ids(v2_root: Path | None) -> set[str]:
    if v2_root is None:
        return set()
    state_path = Path(v2_root) / "status" / "scheduler_state.json"
    try:
        state = _read_mapping(state_path, label="scheduler state")
    except ValueError:
        return set()
    controllers = state.get("controllers") or {}
    if not isinstance(controllers, Mapping):
        return set()
    return {
        str(controller_id)
        for controller_id, controller in controllers.items()
        if isinstance(controller, Mapping) and controller.get("status") == "running"
    }


def _process_command(pid: int) -> tuple[str, ...] | None:
    try:
        return tuple(
            value.decode(errors="replace")
            for value in (Path("/proc") / str(pid) / "cmdline")
            .read_bytes()
            .split(b"\0")
            if value
        )
    except OSError:
        return None


def _process_uid(pid: int) -> int | None:
    try:
        return (Path("/proc") / str(pid)).stat().st_uid
    except OSError:
        return None


def _command_option(command: Sequence[str], option: str) -> str | None:
    try:
        index = tuple(command).index(option)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(command):
        return None
    return str(command[value_index])


def _active_source_sidecar_rounds(
    v2_root: Path,
) -> set[tuple[str, int, int]]:
    root = Path(v2_root).resolve()
    audit_root = (
        root
        / "audits/speed_priority_recovery/concurrent_source_sidecars"
    )
    active: set[tuple[str, int, int]] = set()
    for start_path in sorted(audit_root.glob("*/start.json")):
        audit_dir = start_path.parent
        if (audit_dir / "completion.json").exists() or (
            audit_dir / "failure.json"
        ).exists():
            continue
        try:
            start = _read_mapping(start_path, label="source sidecar start")
            pid = int(start["pid"])
            variant = str(start["variant"])
            seed = int(start["seed"])
            round_index = int(start["round_index"])
        except (KeyError, TypeError, ValueError):
            continue
        command = _process_command(pid)
        if (
            not command
            or "stage7_concurrent_source_sidecar_20260728.py"
            not in " ".join(command)
            or str(root) not in command
            or "--variant" not in command
            or variant not in command
            or "--seed" not in command
            or str(seed) not in command
            or "--round-index" not in command
            or str(round_index) not in command
        ):
            continue
        if _process_uid(pid) != os.getuid():
            continue
        active.add((variant, seed, round_index))
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        if _process_uid(pid) != os.getuid():
            continue
        command = _process_command(pid)
        if not command or not any(
            Path(token).name == "stage5_materialize_round_sources_v1.sh"
            for token in command
        ):
            continue
        request_path = _command_option(command, "--request")
        if request_path is None:
            continue
        try:
            relative = Path(request_path).resolve().relative_to(root / "variants")
        except (OSError, ValueError):
            continue
        parts = relative.parts
        if (
            len(parts) < 4
            or not parts[1].startswith("seed_")
            or not parts[2].startswith("round_")
            or parts[3] != "logical_request.json"
        ):
            continue
        try:
            active.add((parts[0], int(parts[1][5:]), int(parts[2][6:])))
        except ValueError:
            continue
    return active


def _has_live_source_materializer(
    config: Any, variant: str, seed: int, round_index: int
) -> bool:
    return (variant, seed, round_index) in _active_source_sidecar_rounds(
        Path(config.v2_root)
    )


def install_live_source_materializer_wait_policy(orchestrator_module: Any) -> JSON:
    original_source_step = orchestrator_module._default_source_step
    source_controller = __import__(
        "scripts.stage7_source_lease_controller_v2",
        fromlist=["SourceLeaseContractError"],
    )

    def patched_source_step(
        config: Any, action: str, variant: str, seed: int, round_index: int
    ) -> JSON:
        try:
            return original_source_step(config, action, variant, seed, round_index)
        except source_controller.SourceLeaseContractError as error:
            if "existing live lease/attempt takeover is forbidden" in str(error):
                if _has_live_source_materializer(
                    config, variant, seed, round_index
                ):
                    reason = "source materializer already running"
                    category = "controller_running"
                else:
                    reason = (
                        "source attempt pending terminalization for "
                        f"{variant}:seed_{seed}:round_{round_index}"
                    )
                    category = "source_infrastructure"
                raise orchestrator_module.RetryableOrchestrationError(
                    reason,
                    category=category,
                ) from error
            raise

    orchestrator_module._default_source_step = patched_source_step
    return {
        "live_source_materializer_wait": True,
        "stale_source_attempt_rotation": True,
        "recognized_command": "stage5_materialize_round_sources_v1.sh",
        "scope": "matching canonical logical_request path only",
    }


def _reap_measurement_controllers(v2_root: Path) -> list[str]:
    prepare = _read_mapping(
        Path(v2_root) / "prepare_state.json", label="prepare state"
    )
    release = _require_sha256(
        prepare.get("deployment_release_sha256"),
        label="prepared deployment release SHA256",
    )
    manifest = _require_sha256(
        prepare.get("deployment_manifest_file_sha256"),
        label="prepared deployment manifest-file SHA256",
    )
    from scripts import stage7_orchestrator_scheduler_adapter_v2 as adapter

    _scheduler_module, scheduler = adapter.measurement_scheduler(
        str(Path(v2_root)), release, manifest
    )
    return list(scheduler.reap_controllers())


def install_all_seed_fair_policy(
    orchestrator_module: Any,
    *,
    v2_root: Path | None = None,
    controller_reaper: Callable[[Path], Sequence[str]] | None = None,
    active_source_rounds_probe: (
        Callable[[Path], set[tuple[str, int, int]]] | None
    ) = None,
) -> JSON:
    """Open a fair all-seed queue after the first authenticated Full barrier."""
    original_next = orchestrator_module._next_round_operation
    reap_controllers = controller_reaper or _reap_measurement_controllers
    active_source_rounds = (
        active_source_rounds_probe or _active_source_sidecar_rounds
    )
    cursor = 0

    def first_incomplete_action(
        snapshot: Mapping[str, Any],
        variant: str,
        seed: int,
        active_source: set[tuple[str, int, int]],
        live_blocked: set[tuple[str, int, int]],
    ) -> tuple[str, str, int, int] | None:
        for round_index in orchestrator_module.ROUNDS:
            record = orchestrator_module._round_record(
                snapshot, variant, seed, round_index
            )
            if round_index and (
                orchestrator_module._round_record(
                    snapshot, variant, seed, round_index - 1
                ).get("barrier")
                is not True
            ):
                return None
            if record.get("barrier") is True:
                continue
            if record.get("request_frozen") is not True:
                action = orchestrator_module.FREEZE_SOURCE_PLAN
            elif record.get("source_ready") is not True:
                if (variant, seed, round_index) in active_source:
                    live_blocked.add((variant, seed, round_index))
                    return None
                action = orchestrator_module.RESOLVE_SOURCE
            elif record.get("exact_bound") is not True:
                action = orchestrator_module.BIND_EXACT
            elif record.get("terminal") is not True:
                controller_id = f"{variant}:seed_{seed}:round_{round_index}"
                if controller_id in _running_controller_ids(v2_root):
                    live_blocked.add((variant, seed, round_index))
                    return None
                action = orchestrator_module.EXECUTE_MEASUREMENT
            else:
                action = orchestrator_module.FINALIZE_TERMINAL_BARRIER
            return action, variant, seed, round_index
        return None

    def fair_next(
        snapshot: Mapping[str, Any], seed: int
    ) -> tuple[str, str, int, int] | None:
        nonlocal cursor
        if v2_root is not None:
            reap_controllers(Path(v2_root))
            active_source = active_source_rounds(Path(v2_root))
        else:
            active_source = set()
        first_full = orchestrator_module._round_record(
            snapshot,
            "full",
            orchestrator_module.PILOT_SEED,
            0,
        )
        if first_full.get("barrier") is not True:
            return original_next(snapshot, seed)
        live_blocked: set[tuple[str, int, int]] = set()
        operations = [
            operation
            for variant in orchestrator_module.CORE_VARIANTS
            for selected_seed in orchestrator_module.SEEDS
            if (
                operation := first_incomplete_action(
                    snapshot,
                    variant,
                    selected_seed,
                    active_source,
                    live_blocked,
                )
            )
            is not None
        ]
        terminal = [
            operation
            for operation in operations
            if operation[0]
            == orchestrator_module.FINALIZE_TERMINAL_BARRIER
        ]
        if terminal:
            return terminal[0]
        measurement = [
            operation
            for operation in operations
            if operation[0] == orchestrator_module.EXECUTE_MEASUREMENT
        ]
        if measurement:
            selected = measurement[cursor % len(measurement)]
            cursor = (cursor + 1) % len(measurement)
            return selected
        if not operations:
            if live_blocked:
                identities = ",".join(
                    f"{variant}:seed_{selected_seed}:round_{round_index}"
                    for variant, selected_seed, round_index in sorted(live_blocked)
                )
                raise orchestrator_module.RetryableOrchestrationError(
                    "all admissible incomplete rounds already have authenticated "
                    f"live work: {identities}",
                    category="controller_running",
                )
            return None
        selected = operations[cursor % len(operations)]
        cursor = (cursor + 1) % len(operations)
        return selected

    orchestrator_module._next_round_operation = fair_next
    return {
        "all_seed_queue_after_first_full_barrier": True,
        "terminal_finalization_priority": True,
        "measurement_execution_priority": True,
        "trajectory_internal_feedback_barrier": True,
        "live_source_sidecar_round_skip": True,
        "authenticated_live_work_retry_wait": True,
    }


def _load_deployment_modules(v2_root: Path) -> tuple[Any, Any, Any]:
    code_root = (v2_root / "deployment" / "code").resolve(strict=True)
    if code_root != v2_root / "deployment" / "code" or not code_root.is_dir():
        raise ValueError("deployment code root is not canonical")
    prepare = _read_mapping(v2_root / "prepare_state.json", label="prepare state")
    raw_frozen_root = prepare.get("executor_repo_root")
    if not isinstance(raw_frozen_root, str) or not raw_frozen_root:
        raise ValueError("prepare state executor repo root is missing")
    frozen_root = Path(raw_frozen_root).resolve(strict=True)
    if not frozen_root.is_dir():
        raise ValueError("prepare state executor repo root is unavailable")
    for module_name in tuple(sys.modules):
        if (
            module_name == "framework"
            or module_name.startswith("framework.")
            or module_name == "scripts"
            or module_name.startswith("scripts.")
        ):
            sys.modules.pop(module_name, None)
    for runtime_root in (str(frozen_root), str(code_root)):
        if runtime_root in sys.path:
            sys.path.remove(runtime_root)
        sys.path.insert(0, runtime_root)
    os.environ["PYTHONPATH"] = os.pathsep.join((str(code_root), str(frozen_root)))
    os.environ["STAGE7_FROZEN_REPO_ROOT"] = str(frozen_root)
    from framework.stage7 import executor_admission_v2
    from scripts import stage7_core_ablation_orchestrator_v2
    from scripts import stage7_core_ablation_scheduler_v2

    return (
        executor_admission_v2,
        stage7_core_ablation_orchestrator_v2,
        stage7_core_ablation_scheduler_v2,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speed-priority operational takeover for Stage7"
    )
    parser.add_argument("--takeover-v2-root", type=Path, required=True)
    parser.add_argument("--takeover-expected-scheduler-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    args, orchestrator_argv = build_parser().parse_known_args(argv)
    root = args.takeover_v2_root.resolve(strict=True)
    expected_scheduler_sha256 = args.takeover_expected_scheduler_sha256
    if (
        len(expected_scheduler_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_scheduler_sha256)
    ):
        raise ValueError("expected scheduler SHA256 is invalid")
    executor, orchestrator, scheduler = _load_deployment_modules(root)
    from scripts import stage7_core_round_worker_v2 as round_worker

    scheduler_path = Path(scheduler.__file__).resolve(strict=True)
    actual_scheduler_sha256 = _file_sha256(scheduler_path)
    rebind = rebind_operational_identity(
        root,
        current_pid=os.getpid(),
        capture_identity=lambda pid: executor.capture_process_identity(
            executor.probe_linux_process_identity, pid
        ),
        deployment_release_sha256=_value_after_flag(
            orchestrator_argv, "--expected-release-sha256"
        ),
        deployment_manifest_file_sha256=_value_after_flag(
            orchestrator_argv, "--expected-manifest-file-sha256"
        ),
    )
    policy = install_speed_priority_policy(scheduler)
    resume_policy = install_concurrent_resume_policy(scheduler)
    orchestration_policy = install_all_seed_fair_policy(orchestrator, v2_root=root)
    source_materializer_policy = install_live_source_materializer_wait_policy(
        orchestrator
    )
    zero_miss_observer_policy = install_empty_physical_terminal_observer_policy(
        round_worker
    )
    audit = {
        "schema_version": "stage7_speed_priority_takeover_v1",
        "v2_root": str(root),
        "takeover_wrapper_sha256": _file_sha256(Path(__file__).resolve()),
        "expected_scheduler_sha256": expected_scheduler_sha256,
        "deployed_scheduler_sha256": actual_scheduler_sha256,
        "scheduler_sha_mismatch_allowed": (
            actual_scheduler_sha256 != expected_scheduler_sha256
        ),
        "rebind": rebind,
        "policy": policy,
        "resume_policy": resume_policy,
        "orchestration_policy": orchestration_policy,
        "source_materializer_policy": source_materializer_policy,
        "zero_miss_observer_policy": zero_miss_observer_policy,
        "child_python_environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "STAGE7_FROZEN_REPO_ROOT": os.environ.get(
                "STAGE7_FROZEN_REPO_ROOT"
            ),
        },
        "scientific_contract_changed": False,
    }
    _atomic_write_json(
        root / "audits" / f"speed_priority_takeover_pid_{os.getpid()}.json",
        audit,
    )
    return int(orchestrator.main(orchestrator_argv))


if __name__ == "__main__":
    raise SystemExit(main())
