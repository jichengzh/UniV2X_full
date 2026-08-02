#!/usr/bin/env python3
"""V2-only adapter around the tested Stage7 UUID lease scheduler.

This module deliberately owns policy admission only.  GPU probing, non-blocking
UUID leases, audit files, and process recovery remain the v1 scheduler's tested
responsibility.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from framework.stage7.core_ablation_v2 import CORE_VARIANTS, V1_ROOT, V2_ROOT
from scripts import stage7_h800_runtime_v2 as _runtime
from scripts import stage7_scheduler_requests_v2 as _requests
from scripts import stage7_source_scheduler_v2 as _source_scheduler


def _load_v1():
    name = "stage7_ablation_scheduler_v1_for_core_v2"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    path = Path(__file__).with_name("stage7_ablation_scheduler_v1.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage7 scheduler v1 primitives")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_v1 = _load_v1()


def _default_completion_validator(round_dir: Path) -> Mapping[str, Any]:
    """Load the validator from this scheduler's active code overlay."""
    path = Path(__file__).resolve().with_name("stage7_core_online_ablation_v2.py")
    source = path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    module_key = hashlib.sha256(
        str(path).encode("utf-8") + b"\0" + source
    ).hexdigest()
    name = (
        "stage7_core_online_ablation_v2_for_scheduler_"
        + module_key
    )
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__file__ = str(path)
        module.__package__ = ""
        sys.modules[name] = module
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_sha256:
            sys.modules.pop(name, None)
            raise RuntimeError("completion validator bytes changed while loading")
    return module.validate_committed_barrier(Path(round_dir))


GpuSnapshot = _v1.GpuSnapshot
ProcessIdentity = _v1.ProcessIdentity
GPU_BATCH_SIZE = _requests.GPU_BATCH_SIZE
MAX_PARALLEL_BATCHES = _v1.MAX_PARALLEL_BATCHES
CORE_CONTROLLER = _requests.CORE_CONTROLLER
V2Trajectory = _requests.V2Trajectory
V2BatchRequest = _requests.V2BatchRequest
build_v2_trajectory_queue = _requests.build_v2_trajectory_queue
EXPECTED_HOSTNAME = os.environ.get(
    "V2X_FORMAL_H800_HOSTNAME", "zs-nj-tap-gpu18"
).strip()
EXPECTED_GPU_MODEL = "NVIDIA H800"
EXPECTED_GPU_MODELS = frozenset(
    {
        "NVIDIA H800",
        "NVIDIA H800 PCIe",
        "NVIDIA H800 SXM",
        "NVIDIA H800 NVL",
    }
)
FCOOPER_GPU7_SCRIPT = "fcooper_tvm_gpu7_validation_v1.py"
RUNTIME_AUDIT_SCHEMA = "stage7_h800_runtime_admission_v2"
PHYSICAL_PLAN_SCHEMA = "stage7_actual_v3_miss_only_physical_request_v2"


query_gpu_models = _runtime.query_gpu_models


def scan_processes() -> tuple[ProcessIdentity, ...]:
    records: list[ProcessIdentity] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = _v1.inspect_process(int(entry.name))
        if identity is not None:
            records.append(identity)
    return tuple(records)


discover_process_reservations = _runtime.discover_process_reservations


acquire_resource_lock = _runtime.acquire_resource_lock


def _redacted_command(command: Sequence[str]) -> list[str]:
    return _runtime.redacted_command(command, _v1.SECRET_TERMS)


class V2Stage7Scheduler(_v1.Stage7Scheduler):
    """Adds v2 request/root/width admission without replacing v1 lease logic."""

    def __init__(
        self,
        *,
        hostname_probe: Callable[[], str] = socket.gethostname,
        gpu_model_probe: Callable[[], Mapping[str, str]] = query_gpu_models,
        reservation_probe: Callable[[], Mapping[str, Sequence[str]]] | None = None,
        process_scan: Callable[[], Sequence[ProcessIdentity]] = scan_processes,
        resource_lock_factory: Callable[[str], Any | None] = acquire_resource_lock,
        completion_validator: Callable[[Path], Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.hostname_probe = hostname_probe
        self.gpu_model_probe = gpu_model_probe
        self.reservation_probe = reservation_probe
        self.process_scan = process_scan
        self.resource_lock_factory = resource_lock_factory
        self.completion_validator = (
            completion_validator or _default_completion_validator
        )
        self._gpu_models: dict[str, str] = {}
        self._reservation_reasons: dict[str, tuple[str, ...]] = {}
        self._runtime_processes: tuple[ProcessIdentity, ...] = ()
        super().__init__(**kwargs)
        self.runtime_audit_path = self.status_dir / "h800_runtime_admission.jsonl"
        self.runtime_audit_path.touch(exist_ok=True)

    def _persist(self, **updates: Any) -> None:
        incoming = updates.get("controllers")
        key = getattr(self, "_prelaunch_controller_id", "")
        if key and isinstance(incoming, Mapping) and key in incoming:
            current = self._state.get("controllers") or {}
            updates = {
                **updates,
                "controllers": {
                    **incoming,
                    key: {**dict(current.get(key) or {}), **dict(incoming[key])},
                },
            }
        super()._persist(**updates)

    def advance_source_prelease(
        self,
        *,
        validate_ready_evidence: Callable[[], Mapping[str, Any]],
        lease_factory: Callable[[], Any | None],
        launch_wrapper: Callable[[Any], Any],
    ) -> dict[str, Any]:
        """Own the source fast-path decision before any H800 lease exists."""
        _source_scheduler.validate_dependency_pins(Path(__file__).resolve().parents[1])
        return _source_scheduler.advance_source_prelease(
            validate_ready_evidence=validate_ready_evidence,
            lease_factory=lease_factory,
            launch_wrapper=launch_wrapper,
        )

    @staticmethod
    def _fcooper_gpu7_active(
        processes: Sequence[ProcessIdentity],
    ) -> bool:
        return any(
            getattr(process, "alive", True)
            and any(
                FCOOPER_GPU7_SCRIPT in part
                or (
                    "fcooper" in part.lower()
                    and ("gpu7" in part.lower() or "recovery" in part.lower())
                )
                for part in process.command
            )
            for process in processes
        )

    def _gpu7_uuid(self) -> str | None:
        for uuid, snapshot in self._last_snapshot.items():
            if snapshot.index == 7:
                return uuid
        return None

    def _refresh_runtime_context(self) -> None:
        models = dict(self.gpu_model_probe())
        processes = tuple(self.process_scan())
        owned_pids = {
            int(controller.get("pid", -1))
            for controller in (self._state.get("controllers") or {}).values()
            if controller.get("status") == "running"
        }
        discovered = discover_process_reservations(
            tuple(process for process in processes if process.pid not in owned_pids)
        )
        supplied = (
            self.reservation_probe() if self.reservation_probe is not None else {}
        )
        raw_reservations = {
            **discovered,
            **{
                str(key): (
                    *discovered.get(str(key), ()),
                    *(str(reason) for reason in reasons),
                )
                for key, reasons in supplied.items()
            },
        }
        reservations: dict[str, tuple[str, ...]] = {}
        by_index = {
            snapshot.index: uuid for uuid, snapshot in self._last_snapshot.items()
        }
        for key, reasons in raw_reservations.items():
            normalized = (
                by_index.get(int(key.split(":", 1)[1]))
                if key.startswith("index:") and key.split(":", 1)[1].isdigit()
                else key
            )
            if normalized in self._last_snapshot:
                reservations[str(normalized)] = (
                    *reservations.get(str(normalized), ()),
                    *reasons,
                )
        gpu7_uuid = self._gpu7_uuid()
        fcooper_active = self._fcooper_gpu7_active(processes)
        if gpu7_uuid:
            reservations = {
                **reservations,
                gpu7_uuid: (
                    *reservations.get(gpu7_uuid, ()),
                    "stage7_gpu7_unconditionally_excluded",
                    *(("fcooper_gpu7_reserved",) if fcooper_active else ()),
                ),
            }
        self._gpu_models = models
        self._reservation_reasons = reservations
        self._runtime_processes = processes
        _v1.append_jsonl(
            self.runtime_audit_path,
            {
                "schema_version": RUNTIME_AUDIT_SCHEMA,
                "event": "runtime_context_sample",
                "wall_time": self.wall_time(),
                "hostname": self.hostname_probe(),
                "gpu_models": dict(sorted(models.items())),
                "reservations": {
                    uuid: list(reasons)
                    for uuid, reasons in sorted(reservations.items())
                },
                "fcooper_gpu7_active": fcooper_active,
                "processes": [
                    {
                        "pid": process.pid,
                        "owner": process.owner,
                        "start_time": process.start_time,
                        "command": _redacted_command(process.command),
                    }
                    for process in processes
                ],
            },
        )

    def _validate_runtime_admission(self) -> None:
        hostname = self.hostname_probe()
        if not hostname or (EXPECTED_HOSTNAME and hostname != EXPECTED_HOSTNAME):
            raise ValueError(
                f"H800 runtime hostname must be {EXPECTED_HOSTNAME}, got {hostname}"
            )
        snapshot_uuids = set(self._last_snapshot)
        if not snapshot_uuids:
            raise ValueError("H800 runtime admission requires a GPU snapshot")
        if set(self._gpu_models) != snapshot_uuids:
            raise ValueError("H800 runtime GPU UUID/model inventory drift")
        wrong = {
            uuid: model
            for uuid, model in self._gpu_models.items()
            if model not in EXPECTED_GPU_MODELS
        }
        if wrong:
            raise ValueError(
                "H800 runtime requires an exact frozen NVIDIA H800 model "
                f"from {sorted(EXPECTED_GPU_MODELS)}: {wrong}"
            )

    def observe_gpus(self) -> dict[str, GpuSnapshot]:
        snapshots = super().observe_gpus()
        self._refresh_runtime_context()
        return snapshots

    def _ready_uuids(self) -> list[str]:
        self._validate_runtime_admission()
        return [
            uuid
            for uuid in super()._ready_uuids()
            if uuid not in self._reservation_reasons
            and self._gpu_models.get(uuid) in EXPECTED_GPU_MODELS
        ]

    def preflight(self) -> dict[str, Any]:
        _source_scheduler.validate_dependency_pins(Path(__file__).resolve().parents[1])
        snapshots = self.observe_gpus()
        self._validate_runtime_admission()
        ready = self._ready_uuids()
        status = (
            "four_or_more_idle_gpus_observed"
            if len(ready) >= GPU_BATCH_SIZE
            else "waiting_for_four_idle_gpus"
        )
        self._lease_event(
            "preflight_read_only",
            idle_gpu_uuids=ready,
            excluded_reservations={
                uuid: list(reasons)
                for uuid, reasons in sorted(self._reservation_reasons.items())
            },
            gpu_models={uuid: self._gpu_models[uuid] for uuid in sorted(snapshots)},
        )
        self._persist(scheduler_status=status)
        return {"scheduler_status": status, "idle_gpu_uuids": ready}

    def _launch(
        self, request: V2BatchRequest, uuids: Sequence[str], locks: tuple[Any, ...]
    ) -> None:
        if (
            len(uuids) != GPU_BATCH_SIZE
            or len(set(uuids)) != GPU_BATCH_SIZE
            or any(uuid in self._leased_uuids() for uuid in uuids)
        ):
            self._release_locks(locks)
            raise ValueError("H800 UUID lease must be immutable and one-job-per-UUID")
        if any(self._gpu_models.get(uuid) not in EXPECTED_GPU_MODELS for uuid in uuids):
            self._release_locks(locks)
            raise ValueError("H800 UUID lease model drift")
        _requests.launch_with_gate(
            self,
            request,
            tuple(uuids),
            locks,
            lambda: super(V2Stage7Scheduler, self)._launch(request, uuids, locks),
        )

    def _controller_request(self, controller: Mapping[str, Any]) -> V2BatchRequest:
        return V2BatchRequest(
            trajectory_id=str(controller["trajectory_id"]),
            trajectory_path=Path(str(controller["trajectory_path"])),
            round_index=int(controller["round_index"]),
            request_sha256=str(controller["request_sha256"]),
            selected_row_ids=tuple(controller["selected_row_ids"]),
            command=tuple(controller["command"]),
            width=tuple(int(value) for value in controller.get("width") or ()),
            expected_release_sha256=str(controller["expected_release_sha256"]),
            expected_manifest_file_sha256=str(controller["expected_manifest_file_sha256"]),
            task_id=str(controller["task_id"]),
            source_lock_key=str(controller.get("source_lock_key") or ""),
            physical_plan_path=Path(str(controller.get("physical_plan_path") or "")),
            physical_request_sha256=str(
                controller.get("physical_request_sha256") or ""
            ),
        )

    def _running_widths(self) -> set[tuple[int, ...]]:
        return {
            tuple(int(value) for value in controller.get("width") or ())
            for controller in (self._state.get("controllers") or {}).values()
            if controller.get("status") == "running" and controller.get("width")
        }

    def _running_source_keys(self) -> set[str]:
        return {
            str(controller.get("source_lock_key") or "")
            for controller in (self._state.get("controllers") or {}).values()
            if controller.get("status") == "running"
            and controller.get("source_lock_key")
        }

    def _acquire_resource_locks(
        self, request: V2BatchRequest
    ) -> tuple[Any, ...] | None:
        acquired: list[Any] = []
        for resource_key in (
            "width:" + "x".join(str(value) for value in request.width),
            "source:" + request.source_lock_key,
        ):
            lock = self.resource_lock_factory(resource_key)
            if lock is None:
                self._release_locks(acquired)
                self._lease_event(
                    "lease_refused_resource_lock",
                    controller_id=request.controller_id,
                    request_sha256=request.request_sha256,
                    resource_key=resource_key,
                )
                return None
            acquired.append(lock)
        return tuple(acquired)

    def _acquire_available_four(
        self, ready_uuids: Sequence[str]
    ) -> tuple[tuple[str, ...], tuple[Any, ...]] | None:
        acquired: list[tuple[str, Any]] = []
        for uuid in ready_uuids:
            lock = self.lock_factory(uuid)
            if lock is None:
                self._lease_event("lease_refused_foreign_lock", gpu_uuid=uuid)
                continue
            acquired.append((uuid, lock))
            if len(acquired) == GPU_BATCH_SIZE:
                break
        if len(acquired) != GPU_BATCH_SIZE:
            self._release_locks(lock for _, lock in acquired)
            return None
        uuids = tuple(uuid for uuid, _ in acquired)
        immediate = dict(self.gpu_probe())
        self._record_occupancy(immediate, "post_lock_immediate_recheck")
        if any(
            uuid not in immediate or not self._is_idle(immediate[uuid])
            for uuid in uuids
        ):
            self._release_locks(lock for _, lock in acquired)
            self._lease_event(
                "lease_abandoned_occupancy_drift",
                gpu_uuids=list(uuids),
            )
            return None
        return uuids, tuple(lock for _, lock in acquired)

    def _post_lock_runtime_admission(self, uuids: Sequence[str]) -> bool:
        immediate = dict(self.gpu_probe())
        self._last_snapshot = immediate
        self._record_occupancy(immediate, "post_all_locks_immediate_recheck")
        self._refresh_runtime_context()
        self._validate_runtime_admission()
        admitted = all(
            uuid in immediate
            and self._is_idle(immediate[uuid])
            and uuid not in self._reservation_reasons
            and self._gpu_models.get(uuid) in EXPECTED_GPU_MODELS
            for uuid in uuids
        )
        if not admitted:
            self._lease_event(
                "lease_abandoned_post_lock_runtime_drift",
                gpu_uuids=list(uuids),
                reservations={
                    uuid: list(self._reservation_reasons.get(uuid, ()))
                    for uuid in uuids
                    if uuid in self._reservation_reasons
                },
            )
        return admitted

    def reap_controllers(self) -> list[str]:
        return _requests.reap_scheduler_controllers(self, _v1._identity_matches)

    def monitor_occupancy(self) -> list[V2BatchRequest]:
        snapshots = dict(self.gpu_probe())
        self._last_snapshot = snapshots
        self._record_occupancy(snapshots, "running_monitor")
        self._refresh_runtime_context()
        self._validate_runtime_admission()
        controllers = dict(self._state.get("controllers") or {})
        deferred_retries: list[V2BatchRequest] = []
        changed = False
        for controller_id, controller in sorted(controllers.items()):
            if controller.get("status") != "running":
                continue
            owned_pid = int(controller["pid"])
            leased = tuple(controller.get("gpu_uuids") or ())
            recorded_models = controller.get("gpu_models") or {}
            drift_reasons: list[str] = []
            if len(leased) != GPU_BATCH_SIZE or len(set(leased)) != GPU_BATCH_SIZE:
                drift_reasons.append("immutable_uuid_lease_drift")
            if any(
                recorded_models.get(uuid) not in EXPECTED_GPU_MODELS
                or self._gpu_models.get(uuid) != recorded_models.get(uuid)
                for uuid in leased
            ):
                drift_reasons.append("gpu_model_drift")
            if any(uuid in self._reservation_reasons for uuid in leased):
                drift_reasons.append("reserved_uuid_conflict")
            for uuid in leased:
                snapshot = snapshots.get(uuid)
                if snapshot is None:
                    drift_reasons.append("uuid_missing")
                    continue
                if any(
                    not self.pid_is_owned(owned_pid, pid)
                    for pid in snapshot.compute_pids
                ):
                    drift_reasons.append("foreign_compute_process")
                elif not snapshot.compute_pids and not self._is_idle(snapshot):
                    drift_reasons.append("occupancy_without_owned_process")
            if not drift_reasons:
                continue
            request = self._controller_request(controller)
            deferred_retries.append(request)
            controllers[controller_id] = {
                **controller,
                "admission_blocked_while_process_alive": True,
                "deferred_retry_request_sha256": request.request_sha256,
                "runtime_drift_reasons": sorted(set(drift_reasons)),
            }
            self._lease_event(
                "runtime_drift_deferred_until_controller_exit",
                controller_id=controller_id,
                request_sha256=request.request_sha256,
                gpu_uuids=list(leased),
                drift_reasons=sorted(set(drift_reasons)),
                process_was_stopped=False,
                locks_released=False,
                duplicate_launch_allowed=False,
            )
            changed = True
        if changed:
            self._persist(
                scheduler_status="running_drift_wait_for_controller_exit",
                controllers=controllers,
            )
        return deferred_retries

    def resume(
        self, requests: Sequence[V2BatchRequest], *, once: bool = False
    ) -> dict[str, Any]:
        _source_scheduler.validate_dependency_pins(Path(__file__).resolve().parents[1])
        self.observe_gpus()
        self._validate_runtime_admission()
        result = super().resume(requests, once=once)
        return {
            **result,
            "runtime_admission": "h800_exact_model_and_uuid_passed",
        }

    def schedule(self, requests: Sequence[V2BatchRequest]) -> dict[str, object]:
        """Launch only whole, barrier-ready, distinct-width batches.

        Result objects make a denied request auditable rather than silently
        dropping it, while v1 remains responsible for acquiring/releasing the
        actual UUID file locks and writing lease/occupancy audit records.
        """
        if any(not isinstance(request, V2BatchRequest) for request in requests):
            raise ValueError("v2 scheduler accepts V2BatchRequest values only")
        _requests.validate_scheduler_status_dir(self)
        ready_uuids = self._ready_uuids()
        slots = max(
            0, MAX_PARALLEL_BATCHES
            - sum(
                controller.get("status") == "running"
                for controller in (self._state.get("controllers") or {}).values()
            ),
        )
        launched: list[V2BatchRequest] = []
        blocked: list[dict[str, object]] = []
        available_uuids = list(ready_uuids)
        used_trajectories: set[str] = set()
        used_widths = self._running_widths()
        used_source_keys = self._running_source_keys()

        for request in requests:
            if request.trajectory_id in used_trajectories:
                blocked.append(
                    {
                        "trajectory_id": request.trajectory_id,
                        "round_index": request.round_index,
                        "reason": "feedback_barrier",
                    }
                )
                continue
            existing = (self._state.get("controllers") or {}).get(request.controller_id)
            if existing:
                if existing.get("request_sha256") != request.request_sha256:
                    raise ValueError("controller_id recovery request_sha256 drift")
                _requests.validate_controller_deployment_pins(existing, request)
                if (
                    not existing.get("physical_request_sha256")
                    or existing.get("physical_request_sha256")
                    != request.physical_request_sha256
                ):
                    raise ValueError(
                        "controller_id recovery physical request SHA drift"
                    )
                status = str(existing.get("status") or "")
                if status != "infrastructure_retry_required":
                    blocked.append(
                        {
                            "trajectory_id": request.trajectory_id,
                            "round_index": request.round_index,
                            "reason": "already_recorded",
                        }
                    )
                    continue
            if not self._request_is_barrier_ready(request):
                blocked.append(
                    {
                        "trajectory_id": request.trajectory_id,
                        "round_index": request.round_index,
                        "reason": "feedback_barrier",
                    }
                )
                continue
            if request.width in used_widths:
                blocked.append(
                    {
                        "trajectory_id": request.trajectory_id,
                        "reason": "same_width_lock",
                    }
                )
                continue
            if request.source_lock_key in used_source_keys:
                blocked.append(
                    {
                        "trajectory_id": request.trajectory_id,
                        "reason": "same_source_lock",
                    }
                )
                continue
            if len(launched) >= slots or (len(launched) + 1) * GPU_BATCH_SIZE > len(
                ready_uuids
            ):
                blocked.append(
                    {"trajectory_id": request.trajectory_id, "reason": "capacity"}
                )
                continue
            resource_locks = self._acquire_resource_locks(request)
            if resource_locks is None:
                blocked.append(
                    {
                        "trajectory_id": request.trajectory_id,
                        "reason": "resource_lock_unavailable",
                    }
                )
                continue
            gpu_lease = self._acquire_available_four(available_uuids)
            if gpu_lease is None:
                self._release_locks(resource_locks)
                blocked.append(
                    {
                        "trajectory_id": request.trajectory_id,
                        "reason": "uuid_lock_unavailable",
                    }
                )
                continue
            uuids, gpu_locks = gpu_lease
            locks = (*gpu_locks, *resource_locks)
            if not self._post_lock_runtime_admission(uuids):
                self._release_locks(locks)
                blocked.append(
                    {
                        "trajectory_id": request.trajectory_id,
                        "reason": "post_lock_runtime_drift",
                    }
                )
                continue
            try:
                self._launch(request, uuids, locks)
            except _requests.LaunchTerminationUnconfirmed:
                raise
            except BaseException:
                self._release_locks(locks)
                raise
            launched.append(request)
            available_uuids = [
                uuid for uuid in available_uuids if uuid not in set(uuids)
            ]
            used_trajectories.add(request.trajectory_id)
            used_widths.add(request.width)
            used_source_keys.add(request.source_lock_key)

        if not launched:
            status = (
                "waiting_for_four_idle_gpus"
                if len(ready_uuids) < GPU_BATCH_SIZE
                else "waiting_for_eligible_batch"
            )
            self._persist(scheduler_status=status)
        else:
            status = str(self.status()["scheduler_status"])
        return {"status": status, "launched": launched, "blocked": blocked}

    def run(
        self,
        requests: Sequence[V2BatchRequest],
        *,
        deploy_manifest: Mapping[str, Any],
        remote_sha_probe: Callable[[str], str],
        once: bool = False,
    ) -> dict[str, Any]:
        _source_scheduler.validate_dependency_pins(Path(__file__).resolve().parents[1])
        required_pins = _source_scheduler.required_deployment_pins(
            Path(__file__).resolve().parents[1]
        )
        _source_scheduler.validate_deploy_manifest_pins(deploy_manifest, required_pins)
        _v1.verify_remote_deploy(deploy_manifest, remote_sha_probe)
        while True:
            self.reap_controllers()
            self.observe_gpus()
            scheduled = self.schedule(requests)
            retries = self.monitor_occupancy() if self._leased_uuids() else []
            if once:
                return {
                    "launched": [item.controller_id for item in scheduled["launched"]],
                    "blocked": scheduled["blocked"],
                    "retries": [item.controller_id for item in retries],
                    "status": self.status(),
                }
            self.sleeper(self.poll_interval_seconds)


Stage7CoreAblationScheduler = V2Stage7Scheduler


def _batch_from_json(payload: Mapping[str, Any]) -> V2BatchRequest:
    return V2BatchRequest(
        trajectory_id=str(payload["trajectory_id"]),
        trajectory_path=Path(str(payload["trajectory_path"])),
        round_index=int(payload["round_index"]),
        request_sha256=str(payload["request_sha256"]),
        selected_row_ids=tuple(payload["selected_row_ids"]),
        command=tuple(payload["command"]),
        width=tuple(int(value) for value in payload["width"]),
        expected_release_sha256=str(payload["expected_release_sha256"]),
        expected_manifest_file_sha256=str(payload["expected_manifest_file_sha256"]),
        task_id=str(payload.get("task_id") or "S7-PYR-TVM"),
        source_lock_key=str(payload["source_lock_key"]),
        physical_plan_path=Path(str(payload["physical_plan_path"])),
        physical_request_sha256=str(payload["physical_request_sha256"]),
    )


def _scheduler_from_config(config: Mapping[str, Any]) -> V2Stage7Scheduler:
    """Production construction intentionally retains every default live probe."""
    return V2Stage7Scheduler(
        status_dir=Path(str(config["status_dir"])),
        idle_samples_required=int(config.get("idle_samples_required", 3)),
        memory_idle_threshold_mib=int(config.get("memory_idle_threshold_mib", 1024)),
        utilization_idle_threshold_percent=int(
            config.get("utilization_idle_threshold_percent", 5)
        ),
        poll_interval_seconds=float(config.get("poll_interval_seconds", 30.0)),
        base_env=config.get("child_environment") or {},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    for name in ("preflight", "run", "resume", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in {"run", "resume"}:
            command.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _v1.read_json(args.config)
    if _v1._contains_secret(config):
        raise ValueError("Stage7 v2 runtime config must not contain credentials")
    scheduler = _scheduler_from_config(config)
    if args.subcommand == "preflight":
        result = scheduler.preflight()
    elif args.subcommand == "status":
        result = scheduler.status()
    else:
        requests = tuple(_batch_from_json(item) for item in config.get("batches", ()))
        if args.subcommand == "resume":
            result = scheduler.resume(requests, once=args.once)
        else:
            deploy_manifest = _v1.read_json(Path(str(config["deploy_manifest_path"])))
            remote_sha256 = config.get("remote_sha256") or {}
            result = scheduler.run(
                requests,
                deploy_manifest=deploy_manifest,
                remote_sha_probe=lambda destination: str(
                    remote_sha256.get(destination, "")
                ),
                once=args.once,
            )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
