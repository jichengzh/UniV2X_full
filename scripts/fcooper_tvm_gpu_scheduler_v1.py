#!/usr/bin/env python3
"""Dynamically schedule immutable F-Cooper TVM measurement jobs on H800 GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

SCHEMA = "fcooper_tvm_gpu_job_manifest_v1"
STATE_SCHEMA = "fcooper_tvm_gpu_scheduler_state_v1"
AUDIT_SCHEMA = "fcooper_tvm_gpu_scheduler_audit_v1"
TERMINAL = {"succeeded", "failed_terminal", "failed_infrastructure", "blocked_dependency"}
RESERVED_RUNNER_ARGS = {"request-json", "row-index", "request-kind", "gpu", "max-trials"}
DEFAULT_INFRASTRUCTURE_EXIT_CODES = {75, 124, 137, 143}
INFRASTRUCTURE_LOG_MARKERS = (
    "connection reset", "connection timed out", "no space left on device",
    "resource temporarily unavailable", "transport endpoint", "worker lost",
)
SCIENTIFIC_LOG_MARKERS = (
    "feasibility failure", "numerical failure", "unsupported operator",
    "contract drift", "shape mismatch",
)

class GpuStatus(NamedTuple):
    index: int
    uuid: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int
    compute_pids: tuple[int, ...]

def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload

def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)

def _parse_csv_rows(text: str, expected_fields: int) -> list[list[str]]:
    result: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != expected_fields:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        result.append(fields)
    return result

def query_nvidia_smi(
    *, run_command: Callable[..., Any] = subprocess.run
) -> dict[int, GpuStatus]:
    common = {"text": True, "capture_output": True, "check": True}
    gpu_result = run_command([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], **common)
    process_result = run_command([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], **common)
    pids_by_uuid: dict[str, list[int]] = {}
    for uuid, raw_pid in _parse_csv_rows(process_result.stdout, 2):
        pids_by_uuid.setdefault(uuid, []).append(int(raw_pid))
    statuses: dict[int, GpuStatus] = {}
    for raw_index, uuid, raw_used, raw_total, raw_util in _parse_csv_rows(
        gpu_result.stdout, 5
    ):
        index = int(raw_index)
        statuses[index] = GpuStatus(
            index=index,
            uuid=uuid,
            memory_used_mib=int(raw_used),
            memory_total_mib=int(raw_total),
            utilization_percent=int(raw_util),
            compute_pids=tuple(sorted(pids_by_uuid.get(uuid, []))),
        )
    return statuses


def gpu_is_free(
    status: GpuStatus, *, memory_threshold_mib: int, utilization_threshold_percent: int
) -> bool:
    return (
        not status.compute_pids
        and status.memory_used_mib <= memory_threshold_mib
        and status.utilization_percent <= utilization_threshold_percent
    )


def _request_snapshot(path: Path, row_index: int) -> dict[str, Any]:
    payload = read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not 0 <= row_index < len(rows):
        raise ValueError(f"request row index is invalid: {path} row={row_index}")
    row = rows[row_index]
    if not isinstance(row, Mapping):
        raise ValueError(f"request row is not an object: {path} row={row_index}")
    row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    if not row_id:
        raise ValueError(f"request row identity is missing: {path} row={row_index}")
    return {
        "request_sha256": sha256_file(path),
        "request_payload_sha256": canonical_sha256(payload),
        "row_id": row_id,
        "row_sha256": canonical_sha256(row),
        "batch_size": int(payload.get("batch_size", len(rows))),
    }


def _validate_scalar_or_sequence(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if all(item is None or isinstance(item, (str, int, float)) for item in value):
            return
    raise ValueError(f"runner argument {name!r} has unsupported value")


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = read_json(path)
    recorded_sha = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("schema_version") != SCHEMA:
        raise ValueError(f"unsupported job manifest schema: {payload.get('schema_version')}")
    if recorded_sha != canonical_sha256(unsigned):
        raise ValueError("job manifest SHA drift")
    gpu_pool = payload.get("gpu_pool")
    if (
        not isinstance(gpu_pool, list)
        or not gpu_pool
        or len(set(gpu_pool)) != len(gpu_pool)
        or any(not isinstance(gpu, int) or not 0 <= gpu <= 7 for gpu in gpu_pool)
    ):
        raise ValueError("gpu_pool must contain unique H800 GPU indices in [0,7]")
    barrier_order = payload.get("barrier_order")
    if (
        not isinstance(barrier_order, list)
        or len(set(barrier_order)) != len(barrier_order)
        or any(not isinstance(item, str) or not item for item in barrier_order)
    ):
        raise ValueError("barrier_order must contain unique non-empty strings")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("job manifest must contain at least one job")

    normalized: dict[str, dict[str, Any]] = {}
    gear_priorities: list[int] = []
    control_priorities: list[int] = []
    for sequence, raw_job in enumerate(jobs):
        if not isinstance(raw_job, Mapping):
            raise ValueError("each manifest job must be an object")
        job = dict(raw_job)
        job_id = str(job.get("job_id") or "")
        if not job_id or job_id in normalized:
            raise ValueError("job_id values must be non-empty and unique")
        kind = str(job.get("request_kind") or "")
        if kind not in {"t16", "stage6-control"}:
            raise ValueError(f"unsupported request_kind for {job_id}: {kind}")
        barrier = job.get("barrier_id")
        priority = job.get("priority")
        if not isinstance(priority, int):
            raise ValueError(f"invalid priority for {job_id}")
        if kind == "t16":
            if barrier not in barrier_order:
                raise ValueError(f"GEAR job {job_id} has an unknown barrier_id")
            gear_priorities.append(priority)
        else:
            if barrier is not None:
                raise ValueError("control jobs must not participate in GEAR barriers")
            control_priorities.append(priority)
        row_index = job.get("row_index")
        max_trials = job.get("max_trials")
        if not isinstance(row_index, int) or row_index < 0:
            raise ValueError(f"invalid row_index for {job_id}")
        if not isinstance(max_trials, int) or max_trials < 0:
            raise ValueError(f"invalid max_trials for {job_id}")
        request_path = Path(str(job.get("request_json") or ""))
        runner_python = Path(str(job.get("runner_python") or ""))
        runner_script = Path(str(job.get("runner_script") or ""))
        if not request_path.is_file():
            raise ValueError(f"request_json does not exist for {job_id}: {request_path}")
        if not str(runner_python) or not str(runner_script):
            raise ValueError(f"runner command is incomplete for {job_id}")
        common = job.get("runner_common_args")
        if not isinstance(common, Mapping):
            raise ValueError(f"runner_common_args is not an object for {job_id}")
        names = {str(name).lstrip("-") for name in common}
        conflict = RESERVED_RUNNER_ARGS.intersection(names)
        if conflict:
            raise ValueError(
                f"runner_common_args overrides immutable scheduler fields: {sorted(conflict)}"
            )
        for name, value in common.items():
            _validate_scalar_or_sequence(value, str(name))
        release_marker = job.get("gpu_release_marker")
        if release_marker is not None and not str(release_marker):
            raise ValueError(f"empty gpu_release_marker for {job_id}")
        normalized[job_id] = {
            **job,
            "job_id": job_id,
            "request_json": str(request_path.resolve()),
            "runner_python": str(runner_python),
            "runner_script": str(runner_script),
            "priority": int(job.get("priority")),
            "sequence": sequence,
            "input_snapshot": _request_snapshot(request_path, row_index),
            "job_sha256": canonical_sha256(job),
        }
    if gear_priorities and control_priorities and min(gear_priorities) <= max(
        control_priorities
    ):
        raise ValueError("all GEAR priorities must be higher than control priorities")
    return payload, normalized


def _runner_arguments(common: Mapping[str, Any]) -> list[str]:
    arguments: list[str] = []
    for raw_name in sorted(common):
        name = str(raw_name).lstrip("-")
        value = common[raw_name]
        flag = f"--{name}"
        if value is True:
            arguments.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                arguments.extend([flag, str(item)])
        else:
            arguments.extend([flag, str(value)])
    return arguments


def build_runner_command(job: Mapping[str, Any], gpu: int) -> list[str]:
    return [
        str(job["runner_python"]),
        str(job["runner_script"]),
        "--request-json",
        str(job["request_json"]),
        "--row-index",
        str(job["row_index"]),
        "--request-kind",
        str(job["request_kind"]),
        "--gpu",
        str(gpu),
        "--max-trials",
        str(job["max_trials"]),
        *_runner_arguments(job["runner_common_args"]),
    ]


def classify_failure(returncode: int, log_tail: str) -> str:
    lowered = log_tail.lower()
    if any(marker in lowered for marker in SCIENTIFIC_LOG_MARKERS):
        return "scientific"
    if returncode in DEFAULT_INFRASTRUCTURE_EXIT_CODES or any(
        marker in lowered for marker in INFRASTRUCTURE_LOG_MARKERS
    ):
        return "infrastructure"
    return "scientific"


class Scheduler:
    def __init__(
        self,
        *,
        manifest_path: Path,
        state_path: Path,
        events_path: Path,
        audit_path: Path,
        gpu_probe: Callable[[], Mapping[int, GpuStatus]] = query_nvidia_smi,
        popen_factory: Callable[..., Any] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
        poll_interval_seconds: float = 30.0,
        memory_threshold_mib: int = 1024,
        utilization_threshold_percent: int = 5,
        max_infrastructure_attempts: int = 3,
        drain_gpu7: bool = False,
        gpu7_exclusive_job_id: str | None = None,
    ):
        if not 0 < poll_interval_seconds <= 60:
            raise ValueError("poll interval must be in (0,60] seconds")
        if max_infrastructure_attempts < 1:
            raise ValueError("max infrastructure attempts must be positive")
        self.manifest_path = manifest_path.resolve()
        self.state_path = state_path
        self.events_path = events_path
        self.audit_path = audit_path
        self.gpu_probe = gpu_probe
        self.popen_factory = popen_factory
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.poll_interval_seconds = poll_interval_seconds
        self.memory_threshold_mib = memory_threshold_mib
        self.utilization_threshold_percent = utilization_threshold_percent
        self.max_infrastructure_attempts = max_infrastructure_attempts
        self.drain_gpu7 = drain_gpu7
        self.gpu7_exclusive_job_id = gpu7_exclusive_job_id
        self.manifest, self.jobs = load_manifest(self.manifest_path)
        if gpu7_exclusive_job_id and gpu7_exclusive_job_id not in self.jobs:
            raise ValueError("GPU7 exclusive job is not present in the manifest")
        self.manifest_file_sha256 = sha256_file(self.manifest_path)
        self.processes: dict[str, Any] = {}
        self.log_handles: dict[str, Any] = {}
        self.state = self._load_or_initialize_state()
        self._verify_immutable_inputs()

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if (
                state.get("schema_version") != STATE_SCHEMA
                or state.get("manifest_sha256") != self.manifest["manifest_sha256"]
                or state.get("manifest_file_sha256") != self.manifest_file_sha256
            ):
                raise ValueError("scheduler state does not match immutable manifest")
            expected = set(self.jobs)
            if set(state.get("jobs") or {}) != expected:
                raise ValueError("scheduler state job set drift")
            return state
        now_wall = self.wall_time()
        state = {
            "schema_version": STATE_SCHEMA,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest["manifest_sha256"],
            "manifest_file_sha256": self.manifest_file_sha256,
            "started_wall_time": now_wall,
            "updated_wall_time": now_wall,
            "peak_running_jobs": 0,
            "gpu_busy_seconds": {str(gpu): 0.0 for gpu in self.manifest["gpu_pool"]},
            "jobs": {
                job_id: {
                    "status": "pending",
                    "attempts": 0,
                    "infrastructure_retries": 0,
                    "gpu": None,
                    "gpu_released": False,
                    "pid": None,
                    "queued_wall_time": now_wall,
                    "started_wall_time": None,
                    "attempt_started_monotonic": None,
                    "finished_wall_time": None,
                    "returncode": None,
                    "attempt_history": [],
                    "input_snapshot": job["input_snapshot"],
                    "job_sha256": job["job_sha256"],
                }
                for job_id, job in self.jobs.items()
            },
        }
        atomic_write_json(self.state_path, state)
        return state

    def _verify_immutable_inputs(self) -> None:
        if sha256_file(self.manifest_path) != self.manifest_file_sha256:
            raise ValueError("job manifest drift")
        for job_id, job in self.jobs.items():
            current = _request_snapshot(Path(job["request_json"]), job["row_index"])
            recorded = self.state["jobs"][job_id]["input_snapshot"]
            if current != recorded:
                raise ValueError(f"request drift for job {job_id}")
            if self.state["jobs"][job_id]["job_sha256"] != job["job_sha256"]:
                raise ValueError(f"job contract drift for job {job_id}")

    def _event(self, event: str, **values: Any) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "fcooper_tvm_gpu_scheduler_event_v1",
            "event": event,
            "wall_time": self.wall_time(),
            **values,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    def leased_gpus(self) -> set[int]:
        return {
            int(status["gpu"])
            for status in self.state["jobs"].values()
            if status["status"] == "running"
            and not status.get("gpu_released")
            and status.get("gpu") is not None
        }

    def _barrier_ready(self, job: Mapping[str, Any]) -> bool:
        if job["request_kind"] != "t16":
            return True
        order = self.manifest["barrier_order"]
        barrier_index = order.index(job["barrier_id"])
        earlier = set(order[:barrier_index])
        return all(
            self.state["jobs"][other_id]["status"] == "succeeded"
            for other_id, other in self.jobs.items()
            if other.get("barrier_id") in earlier
        )

    def _eligible_jobs(self) -> list[dict[str, Any]]:
        candidates = [
            job
            for job_id, job in self.jobs.items()
            if self.state["jobs"][job_id]["status"] == "pending"
            and self._barrier_ready(job)
        ]
        return sorted(
            candidates,
            key=lambda job: (
                -int(job["priority"]),
                0 if job["request_kind"] == "t16" else 1,
                int(job["sequence"]),
            ),
        )

    def _gpu_allowed_for_job(self, gpu: int, job_id: str) -> bool:
        if gpu != 7:
            return True
        if self.drain_gpu7:
            return False
        if self.gpu7_exclusive_job_id is not None:
            return job_id == self.gpu7_exclusive_job_id
        return True

    def _free_gpus(self) -> list[int]:
        statuses = self.gpu_probe()
        leased = self.leased_gpus()
        result = []
        for gpu in sorted(self.manifest["gpu_pool"]):
            status = statuses.get(gpu)
            if (
                status is not None
                and gpu not in leased
                and gpu_is_free(
                    status,
                    memory_threshold_mib=self.memory_threshold_mib,
                    utilization_threshold_percent=self.utilization_threshold_percent,
                )
            ):
                result.append(gpu)
        return result

    def _log_path(self, job_id: str, attempt: int) -> Path:
        return self.state_path.parent / "logs" / f"{job_id}.attempt{attempt:02d}.log"

    def _launch(self, job: Mapping[str, Any], gpu: int) -> None:
        job_id = str(job["job_id"])
        status = self.state["jobs"][job_id]
        attempt = int(status["attempts"]) + 1
        log_path = self._log_path(job_id, attempt)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab")
        command = build_runner_command(job, gpu)
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "FCOOPER_TVM_SCHEDULER_JOB_ID": job_id,
        }
        try:
            process = self.popen_factory(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        except Exception:
            handle.close()
            raise
        now_wall = self.wall_time()
        status.update(
            {
                "status": "running",
                "attempts": attempt,
                "gpu": gpu,
                "gpu_released": False,
                "pid": int(process.pid),
                "started_wall_time": status["started_wall_time"] or now_wall,
                "attempt_started_monotonic": self.monotonic(),
                "last_attempt_wall_time": now_wall,
                "last_log_path": str(log_path),
                "returncode": None,
            }
        )
        status["attempt_history"].append({
            "attempt": attempt, "gpu": gpu, "started_wall_time": now_wall,
            "gpu_seconds": None, "returncode": None,
        })
        self.processes[job_id] = process
        self.log_handles[job_id] = handle
        self._event(
            "job_started",
            job_id=job_id,
            row_id=job["input_snapshot"]["row_id"],
            gpu=gpu,
            attempt=attempt,
            request_sha256=job["input_snapshot"]["request_sha256"],
        )

    def _log_tail(self, job_id: str, limit: int = 16384) -> str:
        raw_path = self.state["jobs"][job_id].get("last_log_path")
        if not raw_path:
            return ""
        path = Path(raw_path)
        if not path.is_file():
            return ""
        with path.open("rb") as handle:
            handle.seek(max(path.stat().st_size - limit, 0))
            return handle.read().decode(errors="replace")

    def _account_gpu_time(self, job_id: str) -> None:
        status = self.state["jobs"][job_id]
        if status.get("gpu_released") or status.get("gpu") is None:
            return
        started = status.get("attempt_started_monotonic")
        elapsed = max(0.0, self.monotonic() - float(started)) if started is not None else 0.0
        gpu_key = str(status["gpu"])
        self.state["gpu_busy_seconds"][gpu_key] = (
            float(self.state["gpu_busy_seconds"].get(gpu_key, 0.0)) + elapsed
        )
        status["gpu_released"] = True
        status["gpu_seconds_last_attempt"] = elapsed
        if status["attempt_history"]:
            status["attempt_history"][-1]["gpu_seconds"] = elapsed

    def _close_process(self, job_id: str) -> None:
        handle = self.log_handles.pop(job_id, None)
        if handle is not None:
            handle.close()
        self.processes.pop(job_id, None)

    def _finish(self, job_id: str, returncode: int) -> None:
        status = self.state["jobs"][job_id]
        self._account_gpu_time(job_id)
        self._close_process(job_id)
        status["pid"] = None
        status["returncode"] = returncode
        status["finished_wall_time"] = self.wall_time()
        if status["attempt_history"]:
            status["attempt_history"][-1]["returncode"] = returncode
        if returncode == 0:
            status["status"] = "succeeded"
            outcome = "success"
        else:
            outcome = classify_failure(returncode, self._log_tail(job_id))
            if (
                outcome == "infrastructure"
                and int(status["attempts"]) < self.max_infrastructure_attempts
            ):
                status["status"] = "pending"
                status["infrastructure_retries"] = int(status["infrastructure_retries"]) + 1
                status["gpu"] = None
                status["gpu_released"] = False
                status["attempt_started_monotonic"] = None
            elif outcome == "infrastructure":
                status["status"] = "failed_infrastructure"
            else:
                status["status"] = "failed_terminal"
        self._event("job_finished", job_id=job_id, returncode=returncode,
                    classification=outcome, status=status["status"],
                    attempts=status["attempts"])

    def _reconcile(self) -> None:
        for job_id, process in list(self.processes.items()):
            returncode = process.poll()
            if returncode is not None:
                self._finish(job_id, int(returncode))
                continue
            status = self.state["jobs"][job_id]
            marker = self.jobs[job_id].get("gpu_release_marker")
            if (
                marker
                and not status.get("gpu_released")
                and Path(str(marker)).is_file()
                and Path(str(marker)).stat().st_mtime
                >= float(status.get("last_attempt_wall_time") or 0.0)
            ):
                gpu = status.get("gpu")
                self._account_gpu_time(job_id)
                self._event("gpu_phase_released", job_id=job_id, gpu=gpu)

    def _recover_orphaned_running_jobs(self) -> None:
        for job_id, status in self.state["jobs"].items():
            if status["status"] != "running" or job_id in self.processes:
                continue
            pid = int(status.get("pid") or 0)
            if pid > 0:
                try:
                    os.kill(pid, 0)
                    continue
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue
            completion = self.jobs[job_id].get("completion_marker")
            if completion and Path(str(completion)).is_file():
                self._finish(job_id, 0)
            else:
                status["status"] = "pending"
                status["infrastructure_retries"] = int(status["infrastructure_retries"]) + 1
                status["gpu"] = None
                status["gpu_released"] = False
                status["pid"] = None
                status["attempt_started_monotonic"] = None
                self._event("orphaned_job_requeued", job_id=job_id)

    def _block_dependent_barriers(self) -> None:
        order = self.manifest["barrier_order"]
        failed_barriers = {
            job.get("barrier_id")
            for job_id, job in self.jobs.items()
            if self.state["jobs"][job_id]["status"]
            in {"failed_terminal", "failed_infrastructure", "blocked_dependency"}
            and job.get("barrier_id") in order
        }
        if not failed_barriers:
            return
        first_failed_index = min(order.index(barrier) for barrier in failed_barriers)
        blocked = set(order[first_failed_index + 1 :])
        for job_id, job in self.jobs.items():
            status = self.state["jobs"][job_id]
            if status["status"] == "pending" and job.get("barrier_id") in blocked:
                status["status"] = "blocked_dependency"
                status["finished_wall_time"] = self.wall_time()
                self._event("job_blocked_by_barrier", job_id=job_id,
                            barrier_id=job.get("barrier_id"))

    def _launch_available(self) -> None:
        while True:
            free_gpus = self._free_gpus()
            if not free_gpus:
                return
            selected: tuple[dict[str, Any], int] | None = None
            for job in self._eligible_jobs():
                allowed = [gpu for gpu in free_gpus
                           if self._gpu_allowed_for_job(gpu, str(job["job_id"]))]
                if allowed:
                    selected = (job, allowed[0])
                    break
            if selected is None:
                return
            self._verify_immutable_inputs()
            self._launch(*selected)

    def _write_audit(self) -> None:
        now = self.wall_time()
        states = self.state["jobs"]
        counts = {
            name: sum(1 for value in states.values() if value["status"] == name)
            for name in sorted({"pending", "running", *TERMINAL})
        }
        running_count = counts["running"]
        self.state["peak_running_jobs"] = max(
            int(self.state.get("peak_running_jobs", 0)), running_count)
        self.state["updated_wall_time"] = now
        wall_seconds = max(0.0, now - float(self.state["started_wall_time"]))
        gpu_seconds = sum(float(value) for value in self.state["gpu_busy_seconds"].values())
        queue_delays = [
            float(value["started_wall_time"]) - float(value["queued_wall_time"])
            for value in states.values()
            if value.get("started_wall_time") is not None
        ]
        audit = {
            "schema_version": AUDIT_SCHEMA,
            "manifest_sha256": self.manifest["manifest_sha256"],
            "wall_clock_seconds": wall_seconds,
            "gpu_hours": gpu_seconds / 3600.0,
            "per_gpu": {
                gpu: {
                    "busy_seconds": float(seconds),
                    "jobs_started": sum(1 for value in states.values()
                                        for attempt in value.get("attempt_history", [])
                                        if attempt.get("gpu") == int(gpu)),
                    "job_attempts": [
                        {"job_id": job_id, **attempt}
                        for job_id, value in states.items()
                        for attempt in value.get("attempt_history", [])
                        if attempt.get("gpu") == int(gpu)
                    ],
                }
                for gpu, seconds in self.state["gpu_busy_seconds"].items()
            },
            "effective_parallelism": {
                "peak_running_jobs": self.state["peak_running_jobs"],
                "average_gpu_jobs": gpu_seconds / wall_seconds if wall_seconds > 0 else 0.0,
            },
            "queue": {
                "pending_jobs": counts["pending"],
                "maximum_start_delay_seconds": max(queue_delays, default=0.0),
            },
            "retries": {
                "infrastructure_retry_count": sum(
                    int(value["infrastructure_retries"]) for value in states.values()),
                "jobs_retried": sum(1 for value in states.values()
                                    if value["infrastructure_retries"]),
            },
            "status_counts": counts,
            "jobs": {
                job_id: {
                    key: value
                    for key, value in status.items()
                    if key in {"status", "attempts", "infrastructure_retries", "gpu",
                               "gpu_released", "started_wall_time",
                               "finished_wall_time", "returncode"}
                }
                for job_id, status in states.items()
            },
        }
        atomic_write_json(self.state_path, self.state)
        atomic_write_json(self.audit_path, audit)

    def tick(self) -> bool:
        self._verify_immutable_inputs()
        self._recover_orphaned_running_jobs()
        self._reconcile()
        self._block_dependent_barriers()
        self._launch_available()
        self._write_audit()
        return all(status["status"] in TERMINAL for status in self.state["jobs"].values())

    def run(self) -> int:
        self._event("scheduler_started", manifest_sha256=self.manifest["manifest_sha256"])
        try:
            while not self.tick():
                time.sleep(self.poll_interval_seconds)
        except KeyboardInterrupt:
            self._event("scheduler_interrupted")
            self._write_audit()
            return 130
        self._event("scheduler_finished")
        return 0 if all(
            status["status"] == "succeeded" for status in self.state["jobs"].values()
        ) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--free-memory-threshold-mib", type=int, default=1024)
    parser.add_argument("--free-utilization-threshold-percent", type=int, default=5)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    parser.add_argument("--drain-gpu7", action="store_true")
    parser.add_argument("--gpu7-exclusive-job-id")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler = Scheduler(
        manifest_path=args.manifest,
        state_path=args.state,
        events_path=args.events,
        audit_path=args.audit,
        poll_interval_seconds=args.poll_interval_seconds,
        memory_threshold_mib=args.free_memory_threshold_mib,
        utilization_threshold_percent=args.free_utilization_threshold_percent,
        max_infrastructure_attempts=args.max_infrastructure_attempts,
        drain_gpu7=args.drain_gpu7,
        gpu7_exclusive_job_id=args.gpu7_exclusive_job_id,
    )
    if args.once:
        scheduler.tick()
        return 0
    return scheduler.run()


if __name__ == "__main__":
    raise SystemExit(main())
