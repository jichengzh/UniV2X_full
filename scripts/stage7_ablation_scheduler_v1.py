#!/usr/bin/env python3
"""Injectable UUID-based H800 lease scheduler that owns only Stage7 resources."""
from __future__ import annotations
import argparse
import fcntl
import getpass
import hashlib
import json
import os
import pwd
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
STATE_SCHEMA = "stage7_ablation_scheduler_state_v1"
CONTROLLER_SCHEMA = "stage7_controller_pids_v1"
DEPLOY_SCHEMA = "stage7_deploy_manifest_v1"
SUBCOMMANDS = ("deploy", "preflight", "run", "resume", "status", "stop-own-processes")
VARIANTS = ("full", "without_surrogate", "without_measured_feedback", "backend_blind", "without_capability_scan")
SEEDS = (20260718, 20260719, 20260720)
GPU_BATCH_SIZE, MAX_PARALLEL_BATCHES = 4, 2
SECRET_TERMS = ("password", "token", "private_key", "private-key")
@dataclass(frozen=True)
class GpuSnapshot:
    uuid: str
    index: int
    memory_used_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...]
@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    owner: str
    start_time: str
    command: tuple[str, ...]
    alive: bool
@dataclass(frozen=True)
class Trajectory:
    variant: str
    seed: int
    path: Path
    task_id: str = "S7-PYR-TVM"
    @property
    def trajectory_id(self) -> str:
        return f"{self.variant}:seed_{self.seed}"
    def round_path(self, round_index: int) -> Path:
        if round_index < 0:
            raise ValueError("round_index must be non-negative")
        return self.path / f"round_{round_index:02d}"
@dataclass(frozen=True)
class BatchRequest:
    trajectory_id: str
    trajectory_path: Path
    round_index: int
    request_sha256: str
    selected_row_ids: tuple[str, ...]
    command: tuple[str, ...]
    task_id: str = "S7-PYR-TVM"
    def __post_init__(self) -> None:
        if len(self.selected_row_ids) != GPU_BATCH_SIZE:
            raise ValueError("a Stage7 round must contain exactly four selected rows")
        if len(set(self.selected_row_ids)) != GPU_BATCH_SIZE:
            raise ValueError("a Stage7 round must contain four unique selected rows")
        if len(self.request_sha256) != 64:
            raise ValueError("request_sha256 must be a 64-character digest")
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if self.task_id != "S7-PYR-TVM" or not any(Path(part).name == "stage7_task_round_controller_v1.sh" for part in self.command):
            raise ValueError("Stage7 task_id/controller command mismatch")
        if self.trajectory_path.name != f"round_{self.round_index:02d}" or self.task_id in self.trajectory_path.parts:
            raise ValueError("BatchRequest must use the canonical round directory")
    @property
    def controller_id(self) -> str:
        return f"{self.trajectory_id}:round_{self.round_index}"
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload
def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
def build_trajectory_queue(result_root: Path) -> tuple[Trajectory, ...]:
    root = result_root.resolve()
    queue = tuple(
        Trajectory(variant=variant, seed=seed, path=root / "variants" / variant / f"seed_{seed}")
        for variant in VARIANTS
        for seed in SEEDS
    )
    if len({item.path for item in queue}) != len(VARIANTS) * len(SEEDS):
        raise ValueError("Stage7 trajectory paths must be unique")
    return queue
def _contains_secret(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(term in str(key).lower() for term in SECRET_TERMS) or _contains_secret(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret(item) for item in value)
    return False
def build_deploy_manifest(
    *,
    files: Sequence[tuple[Path, str]],
    python_environment: str,
    git_branch: str,
    git_commit: str,
    frozen_evidence: Mapping[str, str],
) -> dict[str, Any]:
    records = tuple(
        {"source": str(source.resolve()), "destination": destination, "sha256": sha256_file(source)}
        for source, destination in files
    )
    manifest = {
        "schema_version": DEPLOY_SCHEMA,
        "files": records,
        "python_environment": python_environment,
        "git": {"branch": git_branch, "commit": git_commit},
        "frozen_evidence": dict(frozen_evidence),
    }
    if _contains_secret(manifest):
        raise ValueError("deploy manifest must not contain credentials")
    return manifest
def verify_remote_deploy(
    manifest: Mapping[str, Any],
    remote_sha_probe: Callable[[str], str],
) -> None:
    if manifest.get("schema_version") != DEPLOY_SCHEMA:
        raise ValueError("blocked_remote_deploy_sha_mismatch: invalid manifest schema")
    records = manifest.get("files")
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("blocked_remote_deploy_sha_mismatch: empty file manifest")
    for record in records:
        digest = str(record.get("sha256", ""))
        if not record.get("source") or not record.get("destination") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("blocked_remote_deploy_sha_mismatch: malformed file record")
        destination = str(record["destination"])
        if remote_sha_probe(destination) != record.get("sha256"):
            raise ValueError(f"blocked_remote_deploy_sha_mismatch: {destination}")
def _parse_csv(text: str, field_count: int) -> tuple[tuple[str, ...], ...]:
    rows = []
    for line in text.splitlines():
        fields = tuple(field.strip() for field in line.split(","))
        if fields and len(fields) == field_count:
            rows = [*rows, fields]
    return tuple(rows)
def query_nvidia_smi(run_command: Callable[..., Any] = subprocess.run) -> dict[str, GpuSnapshot]:
    gpu_result = run_command(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    process_result = run_command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    pids_by_uuid: dict[str, tuple[int, ...]] = {}
    for uuid, raw_pid in _parse_csv(process_result.stdout, 2):
        pids_by_uuid = {
            **pids_by_uuid,
            uuid: (*pids_by_uuid.get(uuid, ()), int(raw_pid)),
        }
    return {
        uuid: GpuSnapshot(
            uuid=uuid,
            index=int(index),
            memory_used_mib=int(memory_used),
            utilization_percent=int(utilization),
            compute_pids=pids_by_uuid.get(uuid, ()),
        )
        for index, uuid, memory_used, utilization in _parse_csv(gpu_result.stdout, 4)
    }
class FileLock:
    def __init__(self, handle: Any) -> None:
        self._handle = handle
    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
def acquire_uuid_lock(uuid: str) -> FileLock | None:
    safe_uuid = "".join(character for character in uuid if character.isalnum() or character in "-_")
    if safe_uuid != uuid or not uuid:
        raise ValueError("unsafe GPU UUID")
    path = Path("/var/lock") / f"stage7_{safe_uuid}.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return FileLock(handle)
def inspect_process(pid: int) -> ProcessIdentity | None:
    proc = Path("/proc") / str(pid)
    try:
        status = (proc / "status").read_text(encoding="utf-8")
        stat_fields = (proc / "stat").read_text(encoding="utf-8").split()
        command = tuple(
            part.decode(errors="surrogateescape")
            for part in (proc / "cmdline").read_bytes().split(b"\0")
            if part
        )
        uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
        owner = pwd.getpwuid(int(uid_line.split()[1])).pw_name
        return ProcessIdentity(
            pid=pid,
            owner=owner,
            start_time=stat_fields[21],
            command=command,
            alive=True,
        )
    except (FileNotFoundError, IndexError, KeyError, PermissionError, ProcessLookupError):
        return None
def process_is_descendant_or_same(controller_pid: int, candidate_pid: int) -> bool:
    current = candidate_pid
    visited: set[int] = set()
    while current > 1 and current not in visited:
        if current == controller_pid:
            return True
        visited = {*visited, current}
        try:
            tail = (Path("/proc") / str(current) / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1]
            current = int(tail.split()[1])
        except (FileNotFoundError, IndexError, PermissionError, ValueError):
            return False
    return False
def terminate_process(pid: int) -> None:
    os.kill(pid, signal.SIGTERM)
def _identity_matches(
    expected: Mapping[str, Any],
    actual: ProcessIdentity | None,
    current_owner: str,
) -> bool:
    is_alive = bool(
        actual
        and (
            getattr(actual, "alive", None) is True
            or (
                getattr(actual, "alive", None) is None
                and callable(getattr(actual, "poll", None))
                and actual.poll() is None
            )
        )
    )
    return bool(
        actual
        and is_alive
        and expected.get("owner") == current_owner == actual.owner
        and str(expected.get("start_time")) == actual.start_time
        and tuple(expected.get("command") or ()) == actual.command
        and int(expected.get("pid")) == actual.pid
    )
def stop_own_processes(
    manifest_path: Path,
    *,
    process_inspector: Callable[[int], ProcessIdentity | None] = inspect_process,
    terminator: Callable[[int], None] = terminate_process,
    current_owner: str | None = None,
) -> dict[str, list[str]]:
    owner = current_owner or getpass.getuser()
    payload = read_json(manifest_path)
    if payload.get("schema_version") != CONTROLLER_SCHEMA:
        raise ValueError("controller PID manifest schema mismatch")
    stopped: list[str] = []
    refused: list[str] = []
    for controller_id, expected in sorted((payload.get("controllers") or {}).items()):
        pid = int(expected.get("pid", -1))
        command = tuple(expected.get("command") or ())
        is_stage7 = any("stage7_task_round_controller_v1" in part for part in command)
        if not is_stage7 or len(str(expected.get("request_sha256", ""))) != 64:
            refused = [*refused, controller_id]
            continue
        actual = process_inspector(pid)
        if not _identity_matches(expected, actual, owner):
            refused = [*refused, controller_id]
            continue
        terminator(pid)
        stopped = [*stopped, controller_id]
    return {"stopped": stopped, "refused": refused}
class Stage7Scheduler:
    def __init__(
        self,
        *,
        status_dir: Path,
        gpu_probe: Callable[[], Mapping[str, GpuSnapshot]] = query_nvidia_smi,
        lock_factory: Callable[[str], Any | None] = acquire_uuid_lock,
        launcher: Callable[..., Any] = subprocess.Popen,
        process_inspector: Callable[[int], ProcessIdentity | None] = inspect_process,
        pid_is_owned: Callable[[int, int], bool] = process_is_descendant_or_same,
        wall_time: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        current_owner: str | None = None,
        idle_samples_required: int = 3,
        memory_idle_threshold_mib: int = 1024,
        utilization_idle_threshold_percent: int = 5,
        poll_interval_seconds: float = 30.0,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        if idle_samples_required < 1:
            raise ValueError("idle_samples_required must be positive")
        if not 0 < poll_interval_seconds <= 60:
            raise ValueError("poll_interval_seconds must be in (0, 60]")
        self.status_dir = status_dir
        self.gpu_probe = gpu_probe
        self.lock_factory = lock_factory
        self.launcher = launcher
        self.process_inspector = process_inspector
        self.pid_is_owned = pid_is_owned
        self.wall_time = wall_time
        self.sleeper = sleeper
        self.current_owner = current_owner or getpass.getuser()
        self.idle_samples_required = idle_samples_required
        self.memory_idle_threshold_mib = memory_idle_threshold_mib
        self.utilization_idle_threshold_percent = utilization_idle_threshold_percent
        self.poll_interval_seconds = poll_interval_seconds
        self.base_env = dict(base_env or {})
        self.state_path = status_dir / "scheduler_state.json"
        self.controller_path = status_dir / "controller_pids.json"
        self.lease_audit_path = status_dir / "gpu_lease_audit.jsonl"
        self.occupancy_path = status_dir / "gpu_occupancy_snapshots.jsonl"
        self._locks: dict[str, tuple[Any, ...]] = {}
        self._processes: dict[str, Any] = {}
        self._last_snapshot: dict[str, GpuSnapshot] = {}
        self._state = self._load_state()
        self._controller_manifest = self._load_controller_manifest()
        self._initialize_audit_files()
    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if state.get("schema_version") != STATE_SCHEMA:
                raise ValueError("scheduler state schema mismatch")
            return state
        state = {
            "schema_version": STATE_SCHEMA,
            "scheduler_status": "initialized",
            "idle_streaks": {},
            "controllers": {},
            "selected_event_budget_consumed": 0,
            "updated_wall_time": self.wall_time(),
        }
        atomic_write_json(self.state_path, state)
        return state
    def _load_controller_manifest(self) -> dict[str, Any]:
        if self.controller_path.is_file():
            payload = read_json(self.controller_path)
            if payload.get("schema_version") != CONTROLLER_SCHEMA:
                raise ValueError("controller PID manifest schema mismatch")
            return payload
        payload = {"schema_version": CONTROLLER_SCHEMA, "controllers": {}}
        atomic_write_json(self.controller_path, payload)
        return payload
    def _initialize_audit_files(self) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.lease_audit_path.touch(exist_ok=True)
        self.occupancy_path.touch(exist_ok=True)
    def _persist(self, **updates: Any) -> None:
        self._state = {
            **self._state,
            **updates,
            "updated_wall_time": self.wall_time(),
        }
        atomic_write_json(self.state_path, self._state)
        atomic_write_json(self.controller_path, self._controller_manifest)
    def _is_idle(self, snapshot: GpuSnapshot) -> bool:
        return (
            not snapshot.compute_pids
            and snapshot.memory_used_mib < self.memory_idle_threshold_mib
            and snapshot.utilization_percent < self.utilization_idle_threshold_percent
        )
    def _record_occupancy(
        self, snapshots: Mapping[str, GpuSnapshot], event: str
    ) -> None:
        append_jsonl(
            self.occupancy_path,
            {
                "schema_version": "stage7_gpu_occupancy_snapshot_v1",
                "event": event,
                "wall_time": self.wall_time(),
                "gpus": [
                    asdict(snapshots[uuid])
                    for uuid in sorted(snapshots)
                ],
            },
        )
    def _lease_event(self, event: str, **values: Any) -> None:
        append_jsonl(
            self.lease_audit_path,
            {
                "schema_version": "stage7_gpu_lease_audit_v1",
                "event": event,
                "wall_time": self.wall_time(),
                **values,
            },
        )
    def observe_gpus(self) -> dict[str, GpuSnapshot]:
        snapshots = dict(self.gpu_probe())
        if any(uuid != snapshot.uuid for uuid, snapshot in snapshots.items()):
            raise ValueError("GPU probe must be keyed by immutable UUID")
        previous = self._state.get("idle_streaks") or {}
        streaks = {
            uuid: int(previous.get(uuid, 0)) + 1 if self._is_idle(snapshot) else 0
            for uuid, snapshot in snapshots.items()
        }
        self._last_snapshot = snapshots
        self._record_occupancy(snapshots, "sample")
        self._persist(idle_streaks=streaks)
        return snapshots
    def preflight(self) -> dict[str, Any]:
        snapshots = self.observe_gpus()
        idle = sorted(uuid for uuid, item in snapshots.items() if self._is_idle(item))
        status = (
            "four_or_more_idle_gpus_observed"
            if len(idle) >= GPU_BATCH_SIZE
            else "waiting_for_four_idle_gpus"
        )
        self._lease_event("preflight_read_only", idle_gpu_uuids=idle)
        self._persist(scheduler_status=status)
        return {"scheduler_status": status, "idle_gpu_uuids": idle}
    def _leased_uuids(self) -> set[str]:
        return {
            uuid
            for controller in (self._state.get("controllers") or {}).values()
            if controller.get("status") == "running"
            for uuid in controller.get("gpu_uuids", ())
        }
    def _ready_uuids(self) -> list[str]:
        leased = self._leased_uuids()
        streaks = self._state.get("idle_streaks") or {}
        return sorted(
            uuid
            for uuid, count in streaks.items()
            if int(count) >= self.idle_samples_required and uuid not in leased
        )
    def _request_is_barrier_ready(self, request: BatchRequest) -> bool:
        if request.round_index == 0:
            return True
        controllers = self._state.get("controllers") or {}
        earlier = {
            int(controller.get("round_index", -1)): controller
            for controller in controllers.values()
            if controller.get("trajectory_id") == request.trajectory_id
            and int(controller.get("round_index", -1)) < request.round_index
        }
        required = set(range(request.round_index))
        return required == set(earlier) and all(
            earlier[index].get("status") == "feedback_complete" for index in required
        )
    def _eligible_requests(
        self, requests: Sequence[BatchRequest]
    ) -> tuple[BatchRequest, ...]:
        controllers = self._state.get("controllers") or {}
        seen_trajectories: set[str] = set()
        eligible: list[BatchRequest] = []
        for request in requests:
            existing = controllers.get(request.controller_id)
            if existing and existing.get("status") in {
                "running",
                "feedback_complete",
                "succeeded",
            }:
                continue
            if request.trajectory_id in seen_trajectories:
                continue
            if not self._request_is_barrier_ready(request):
                continue
            seen_trajectories = {*seen_trajectories, request.trajectory_id}
            eligible = [*eligible, request]
        return tuple(eligible)
    @staticmethod
    def _release_locks(locks: Iterable[Any]) -> None:
        for lock in locks:
            lock.release()
    def _acquire_four(self, uuids: Sequence[str]) -> tuple[Any, ...] | None:
        acquired: list[Any] = []
        for uuid in uuids:
            lock = self.lock_factory(uuid)
            if lock is None:
                self._release_locks(acquired)
                self._lease_event("lease_refused_foreign_lock", gpu_uuid=uuid)
                return None
            acquired = [*acquired, lock]
        immediate = dict(self.gpu_probe())
        self._record_occupancy(immediate, "post_lock_immediate_recheck")
        if any(
            uuid not in immediate or not self._is_idle(immediate[uuid])
            for uuid in uuids
        ):
            self._release_locks(acquired)
            self._lease_event(
                "lease_abandoned_occupancy_drift",
                gpu_uuids=list(uuids),
            )
            return None
        return tuple(acquired)
    def _launch(
        self, request: BatchRequest, uuids: Sequence[str], locks: tuple[Any, ...]
    ) -> None:
        environment = {
            **self.base_env,
            "CUDA_VISIBLE_DEVICES": ",".join(uuids),
        }
        request.trajectory_path.mkdir(parents=True, exist_ok=True)
        process = self.launcher(
            list(request.command),
            env=environment,
            cwd=request.trajectory_path,
        )
        fallback = ProcessIdentity(
            pid=int(process.pid),
            owner=str(getattr(process, "owner", self.current_owner)),
            start_time=str(getattr(process, "start_time", "")),
            command=tuple(getattr(process, "command", request.command)),
            alive=True,
        )
        identity = self.process_inspector(int(process.pid)) or fallback
        controller = {
            "controller_id": request.controller_id,
            "trajectory_id": request.trajectory_id,
            "trajectory_path": str(request.trajectory_path),
            "task_id": request.task_id,
            "round_index": request.round_index,
            "request_sha256": request.request_sha256,
            "selected_row_ids": list(request.selected_row_ids),
            "command": list(request.command),
            "gpu_uuids": list(uuids),
            "pid": identity.pid,
            "owner": identity.owner,
            "start_time": identity.start_time,
            "status": "running",
            "selected_event_budget_consumed": 0,
        }
        controllers = {
            **(self._state.get("controllers") or {}),
            request.controller_id: controller,
        }
        pid_record = {
            "pid": identity.pid,
            "owner": identity.owner,
            "start_time": identity.start_time,
            "command": list(identity.command),
            "request_sha256": request.request_sha256,
        }
        self._controller_manifest = {**self._controller_manifest, "controllers": {
            **(self._controller_manifest.get("controllers") or {}), request.controller_id: pid_record}}
        self._locks = {**self._locks, request.controller_id: locks}
        self._processes = {**self._processes, request.controller_id: process}
        self._lease_event("batch_leased", controller_id=request.controller_id,
                          request_sha256=request.request_sha256, gpu_uuids=list(uuids))
        self._persist(scheduler_status="running", controllers=controllers)
    def reap_controllers(self) -> list[str]:
        controllers = dict(self._state.get("controllers") or {})
        completed: list[str] = []
        budget = int(self._state.get("selected_event_budget_consumed", 0))
        recovered_retry = False
        for controller_id, controller in tuple(controllers.items()):
            if controller.get("status") != "running" or controller_id in self._processes: continue
            expected = (self._controller_manifest.get("controllers") or {}).get(controller_id) or {}
            actual = self.process_inspector(int(expected.get("pid", -1)))
            if _identity_matches(expected, actual, self.current_owner):
                continue
            controllers = {**controllers, controller_id: {**controller, "status": "infrastructure_retry_required",
                "retry_request_sha256": controller["request_sha256"]}}
            recovered_retry = True
            self._lease_event("resumed_controller_gone_same_request_retry", controller_id=controller_id)
        for controller_id, process in tuple(self._processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            controller = controllers[controller_id]
            status = "feedback_complete" if returncode == 0 else "failed_terminal"
            controllers = {
                **controllers,
                controller_id: {**controller, "status": status, "returncode": returncode},
            }
            self._release_locks(self._locks.get(controller_id, ()))
            self._locks = {key: value for key, value in self._locks.items() if key != controller_id}
            self._processes = {key: value for key, value in self._processes.items() if key != controller_id}
            if returncode == 0:
                budget += GPU_BATCH_SIZE
                completed = [*completed, controller_id]
            self._lease_event("controller_finished", controller_id=controller_id, returncode=returncode)
        if completed:
            self._persist(
                scheduler_status="feedback_barrier_open",
                controllers=controllers,
                selected_event_budget_consumed=budget,
            )
        elif controllers != self._state.get("controllers"):
            status = "infrastructure_retry_required" if recovered_retry else self._state["scheduler_status"]
            self._persist(scheduler_status=status, controllers=controllers)
        return completed
    def schedule(self, requests: Sequence[BatchRequest]) -> list[BatchRequest]:
        ready_uuids = self._ready_uuids()
        eligible = self._eligible_requests(requests)
        available_slots = max(
            0,
            MAX_PARALLEL_BATCHES
            - sum(
                controller.get("status") == "running"
                for controller in (self._state.get("controllers") or {}).values()
            ),
        )
        capacity = min(len(ready_uuids) // GPU_BATCH_SIZE, available_slots, len(eligible))
        launched: list[BatchRequest] = []
        for slot in range(capacity):
            uuids = ready_uuids[
                slot * GPU_BATCH_SIZE : (slot + 1) * GPU_BATCH_SIZE
            ]
            locks = self._acquire_four(uuids)
            if locks is None:
                continue
            request = eligible[slot]
            self._launch(request, uuids, locks)
            launched = [*launched, request]
        if not launched and eligible:
            self._persist(scheduler_status="waiting_for_four_idle_gpus")
        return launched
    def _controller_request(self, controller: Mapping[str, Any]) -> BatchRequest:
        return BatchRequest(
            trajectory_id=str(controller["trajectory_id"]),
            trajectory_path=Path(controller["trajectory_path"]),
            round_index=int(controller["round_index"]),
            request_sha256=str(controller["request_sha256"]),
            selected_row_ids=tuple(controller["selected_row_ids"]),
            command=tuple(controller["command"]),
            task_id=str(controller["task_id"]),
        )
    def monitor_occupancy(self) -> list[BatchRequest]:
        snapshots = dict(self.gpu_probe())
        self._record_occupancy(snapshots, "running_monitor")
        controllers = dict(self._state.get("controllers") or {})
        retries: list[BatchRequest] = []
        for controller_id, controller in sorted(controllers.items()):
            if controller.get("status") != "running":
                continue
            owned_pid = int(controller["pid"])
            drifted = any(
                uuid not in snapshots
                or any(
                    not self.pid_is_owned(owned_pid, pid)
                    for pid in snapshots[uuid].compute_pids
                )
                or (
                    not snapshots[uuid].compute_pids
                    and not self._is_idle(snapshots[uuid])
                )
                for uuid in controller.get("gpu_uuids", ())
            )
            if not drifted:
                continue
            updated = {
                **controller,
                "status": "infrastructure_retry_required",
                "retry_request_sha256": controller["request_sha256"],
            }
            controllers = {**controllers, controller_id: updated}
            self._release_locks(self._locks.get(controller_id, ()))
            self._locks = {
                key: value for key, value in self._locks.items() if key != controller_id
            }
            request = self._controller_request(updated)
            retries = [*retries, request]
            self._lease_event(
                "occupancy_drift_same_request_retry",
                controller_id=controller_id,
                request_sha256=request.request_sha256,
                selected_event_budget_consumed=controller.get(
                    "selected_event_budget_consumed", 0
                ),
            )
        status = "infrastructure_retry_required" if retries else self._state["scheduler_status"]
        self._persist(scheduler_status=status, controllers=controllers)
        return retries
    def run(
        self,
        requests: Sequence[BatchRequest],
        *,
        deploy_manifest: Mapping[str, Any],
        remote_sha_probe: Callable[[str], str],
        once: bool = False,
    ) -> dict[str, Any]:
        verify_remote_deploy(deploy_manifest, remote_sha_probe)
        while True:
            self.reap_controllers()
            self.observe_gpus()
            launched = self.schedule(requests)
            retries = self.monitor_occupancy() if self._leased_uuids() else []
            if once:
                return {
                    "launched": [item.controller_id for item in launched],
                    "retries": [item.controller_id for item in retries],
                    "status": self.status(),
                }
            self.sleeper(self.poll_interval_seconds)
    def resume(
        self, requests: Sequence[BatchRequest], *, once: bool = False
    ) -> dict[str, Any]:
        already_live: list[str] = []
        controllers = self._state.get("controllers") or {}
        pid_records = self._controller_manifest.get("controllers") or {}
        for controller_id, controller in sorted(controllers.items()):
            if controller.get("status") != "running":
                continue
            expected = pid_records.get(controller_id) or {}
            actual = self.process_inspector(int(expected.get("pid", -1)))
            if _identity_matches(expected, actual, self.current_owner):
                already_live = [*already_live, controller_id]
        if not already_live:
            self.observe_gpus()
            self.schedule(requests)
        if not once and not already_live:
            self.sleeper(self.poll_interval_seconds)
        return {"already_live": already_live, "status": self.status()}
    def status(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._state))
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    for name in SUBCOMMANDS:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, required=True)
        if name == "preflight":
            subparser.add_argument("--dry-run", action="store_true")
        if name in {"run", "resume"}:
            subparser.add_argument("--once", action="store_true")
    return parser
def _batch_from_json(payload: Mapping[str, Any]) -> BatchRequest:
    return BatchRequest(
        trajectory_id=str(payload["trajectory_id"]), trajectory_path=Path(payload["trajectory_path"]),
        round_index=int(payload["round_index"]), request_sha256=str(payload["request_sha256"]),
        selected_row_ids=tuple(payload["selected_row_ids"]), command=tuple(payload["command"]),
        task_id=str(payload["task_id"]),
    )
def _scheduler_from_config(config: Mapping[str, Any]) -> Stage7Scheduler:
    return Stage7Scheduler(
        status_dir=Path(config["status_dir"]),
        idle_samples_required=int(config.get("idle_samples_required", 3)),
        memory_idle_threshold_mib=int(config.get("memory_idle_threshold_mib", 1024)),
        utilization_idle_threshold_percent=int(config.get("utilization_idle_threshold_percent", 5)),
        poll_interval_seconds=float(config.get("poll_interval_seconds", 30.0)),
        base_env=config.get("child_environment") or {},
    )
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_json(args.config)
    if _contains_secret(config):
        raise ValueError("Stage7 runtime config must not contain credentials")
    if args.subcommand == "deploy":
        manifest = build_deploy_manifest(
            files=[(Path(record["source"]), str(record["destination"])) for record in config["deploy_files"]],
            python_environment=str(config["python_environment"]),
            git_branch=str(config["git_branch"]),
            git_commit=str(config["git_commit"]),
            frozen_evidence=config["frozen_evidence"],
        )
        destination = Path(config["deploy_manifest_path"])
        atomic_write_json(destination, manifest)
        print(json.dumps(manifest, sort_keys=True))
        return 0
    scheduler = _scheduler_from_config(config)
    if args.subcommand == "preflight":
        print(json.dumps(scheduler.preflight(), sort_keys=True))
        return 0
    if args.subcommand == "status":
        print(json.dumps(scheduler.status(), sort_keys=True))
        return 0
    if args.subcommand == "stop-own-processes":
        result = stop_own_processes(scheduler.controller_path)
        print(json.dumps(result, sort_keys=True))
        return 0 if not result["refused"] else 2
    requests = tuple(_batch_from_json(item) for item in config.get("batches", ()))
    if args.subcommand == "resume":
        print(json.dumps(scheduler.resume(requests, once=args.once), sort_keys=True))
        return 0
    deploy_manifest = read_json(Path(config["deploy_manifest_path"]))
    remote_sha256 = config.get("remote_sha256") or {}
    result = scheduler.run(requests, deploy_manifest=deploy_manifest,
        remote_sha_probe=lambda destination: str(remote_sha256.get(destination, "")),
        once=args.once)
    print(json.dumps(result, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
