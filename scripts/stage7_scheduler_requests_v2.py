"""Request and authenticated-completion boundaries for the Stage7 scheduler."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import secrets
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from framework.stage7.core_ablation_v2 import CORE_VARIANTS, V1_ROOT, V2_ROOT
from scripts import stage7_source_scheduler_v2 as _source_scheduler


GPU_BATCH_SIZE = 4
SEEDS = (20260718, 20260719, 20260720)
PHYSICAL_PLAN_SCHEMA = "stage7_actual_v3_miss_only_physical_request_v2"
COMPLETION_BARRIER_SCHEMA = "stage7_actual_v3_atomic_feedback_barrier_v2"
LAUNCH_GATE_SCHEMA = "stage7_v2_controller_launch_gate_v1"
LAUNCH_GATE_TIMEOUT_SECONDS = 60
CORE_CONTROLLER = (
    Path(__file__).resolve().with_name("stage7_core_round_controller_v2.sh")
)
validate_measurement_artifacts = _source_scheduler.validate_measurement_artifacts


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class V2Trajectory:
    variant: str
    seed: int
    path: Path
    batch_size: int = GPU_BATCH_SIZE

    @property
    def trajectory_id(self) -> str:
        return f"{self.variant}:seed_{self.seed}"


@dataclass(frozen=True)
class V2LaunchGate:
    path: Path
    token: str

    @property
    def token_sha256(self) -> str:
        return hashlib.sha256(self.token.encode("ascii")).hexdigest()

    @property
    def environment(self) -> dict[str, str]:
        return {
            "STAGE7_V2_LAUNCH_GATE_PATH": str(self.path),
            "STAGE7_V2_LAUNCH_GATE_TOKEN": self.token,
            "STAGE7_V2_LAUNCH_GATE_TIMEOUT_SECONDS": str(
                LAUNCH_GATE_TIMEOUT_SECONDS
            ),
        }


class LaunchTerminationUnconfirmed(RuntimeError):
    """The child may still use its UUIDs, so scheduler locks must remain held."""


@dataclass(frozen=True)
class V2BatchRequest:
    """An indivisible four-row measurement request for the v2 controller."""

    trajectory_id: str
    trajectory_path: Path
    round_index: int
    request_sha256: str
    selected_row_ids: tuple[str, ...]
    command: tuple[str, ...]
    width: tuple[int, ...]
    expected_release_sha256: str = ""
    expected_manifest_file_sha256: str = ""
    task_id: str = "S7-PYR-TVM"
    source_lock_key: str = ""
    physical_plan_path: Path | None = None
    physical_request_sha256: str = ""

    def __post_init__(self) -> None:
        if len(self.selected_row_ids) != 4 or len(set(self.selected_row_ids)) != 4:
            raise ValueError(
                "a v2 Stage7 round must contain exactly four unique selected rows"
            )
        if len(self.request_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.request_sha256
        ):
            raise ValueError("request_sha256 must be a 64-character lowercase digest")
        if not 0 <= self.round_index <= 3:
            raise ValueError("round_index must be in the frozen range 0..3")
        try:
            variant, raw_seed = self.trajectory_id.split(":seed_", 1)
            seed = int(raw_seed)
        except (TypeError, ValueError):
            raise ValueError(
                "trajectory_id must have the canonical variant:seed_<seed> form"
            ) from None
        if variant not in CORE_VARIANTS:
            raise ValueError("trajectory_id must use a v2 core variant")
        if seed not in SEEDS or raw_seed != str(seed):
            raise ValueError("trajectory_id must use a canonical frozen seed")
        expected_path = (
            Path(V2_ROOT).resolve()
            / "variants"
            / variant
            / f"seed_{seed}"
            / f"round_{self.round_index:02d}"
        )
        if self.trajectory_path.resolve() != expected_path:
            raise ValueError(
                "trajectory_path must use the canonical v2 trajectory path"
            )
        if self.trajectory_path.name != f"round_{self.round_index:02d}":
            raise ValueError("V2BatchRequest must use the canonical round directory")
        for label, value in (
            ("release", self.expected_release_sha256),
            ("manifest-file", self.expected_manifest_file_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(
                    f"{label} deployment pin must be a lowercase 64-hex SHA256"
                )
        expected_command = (
            str(CORE_CONTROLLER),
            "--v2-root",
            str(Path(V2_ROOT).resolve()),
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--round-index",
            str(self.round_index),
            "--request-sha256",
            self.request_sha256,
            "--expected-release-sha256",
            self.expected_release_sha256,
            "--expected-manifest-file-sha256",
            self.expected_manifest_file_sha256,
        )
        if self.task_id != "S7-PYR-TVM" or tuple(self.command) != expected_command:
            raise ValueError("V2BatchRequest must use the exact v2 controller argv")
        if not self.width or any(
            not isinstance(value, int) or value <= 0 for value in self.width
        ):
            raise ValueError(
                "v2 request width must be a non-empty positive integer tuple"
            )
        _source_scheduler.require_measurement_ready(self.trajectory_path)
        plan_path = self.physical_plan_path
        if (
            plan_path is None
            or Path(plan_path).resolve()
            != (self.trajectory_path / "miss_only_physical_request.json").resolve()
            or not Path(plan_path).is_file()
        ):
            raise ValueError("physical_plan_path must bind the canonical round plan")
        try:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("physical miss plan is invalid") from error
        if not isinstance(plan, Mapping):
            raise ValueError("physical miss plan must be an object")
        unsigned = {
            key: value
            for key, value in plan.items()
            if key not in {"physical_request_sha256", "logical_row_bindings"}
        }
        recorded = str(plan.get("physical_request_sha256") or "")
        rows = plan.get("rows")
        bindings = plan.get("logical_row_bindings")
        if (
            plan.get("schema_version") != PHYSICAL_PLAN_SCHEMA
            or plan.get("logical_request_sha256") != self.request_sha256
            or plan.get("logical_row_count") != 4
            or not isinstance(rows, list)
            or plan.get("physical_row_count") != len(rows)
            or len(rows) not in range(5)
            or not isinstance(bindings, list)
            or len(bindings) != 4
            or recorded != canonical_sha256(unsigned)
            or self.physical_request_sha256 != recorded
        ):
            raise ValueError("physical miss plan/request identity drift")
        binding_ids = tuple(
            str(binding.get("candidate_id") or "")
            for binding in bindings
            if isinstance(binding, Mapping)
        )
        physical_ids = tuple(
            str(row.get("row_id") or row.get("manifest_job_id") or "")
            for row in rows
            if isinstance(row, Mapping)
        )
        expected_physical_ids = tuple(
            str(binding.get("candidate_id") or "")
            for binding in bindings
            if isinstance(binding, Mapping) and binding.get("disposition") == "miss"
        )
        if (
            binding_ids != self.selected_row_ids
            or physical_ids != expected_physical_ids
            or len(set(binding_ids)) != 4
        ):
            raise ValueError("physical miss plan selected-row binding drift")
        validated = validate_measurement_artifacts(self.trajectory_path)
        formal_result = validated["source_result"]
        if validated["physical_plan"] != plan:
            raise ValueError("validated physical plan differs from request plan")
        expected_source_key = resolved_source_lock_sha256(plan, formal_result)
        if self.source_lock_key != expected_source_key:
            raise ValueError(
                "source_lock_key must equal the authenticated formal "
                "resolved-source identity SHA"
            )

    @property
    def controller_id(self) -> str:
        return f"{self.trajectory_id}:round_{self.round_index}"

    @property
    def source_resolution_result_sha256(self) -> str:
        payload = json.loads(
            (self.trajectory_path / "source_resolution_result.json").read_text(
                encoding="utf-8"
            )
        )
        return str(payload["source_resolution_result_sha256"])


def validate_controller_deployment_pins(
    controller: Mapping[str, Any], request: V2BatchRequest
) -> None:
    if (
        controller.get("expected_release_sha256")
        != request.expected_release_sha256
        or controller.get("expected_manifest_file_sha256")
        != request.expected_manifest_file_sha256
    ):
        raise ValueError("controller_id recovery deployment pin drift")


def validate_scheduler_status_dir(scheduler: Any) -> Path:
    status_dir = Path(scheduler.status_dir).resolve()
    if status_dir != Path(V2_ROOT).resolve() / "status":
        raise ValueError("launch gate must use the canonical v2 status directory")
    return status_dir


def prepare_launch_gate(scheduler: Any, request: V2BatchRequest) -> V2LaunchGate:
    gate_dir = validate_scheduler_status_dir(scheduler) / "launch_gates"
    gate_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = os.lstat(gate_dir)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError("launch gate directory ownership or mode drift")
    nonce_value = secrets.token_hex(32)
    token_sha = hashlib.sha256(nonce_value.encode("ascii")).hexdigest()
    controller_sha = hashlib.sha256(
        request.controller_id.encode("utf-8")
    ).hexdigest()
    name = (
        f"gate_{controller_sha[:16]}_{request.request_sha256[:16]}_"
        f"{token_sha[:16]}.json"
    )
    gate = V2LaunchGate(path=gate_dir / name, token=nonce_value)
    if gate.path.exists() or gate.path.is_symlink():
        raise ValueError("launch gate path already exists")
    return gate


def prelaunch_controller_record(
    scheduler: Any,
    request: V2BatchRequest,
    uuids: tuple[str, ...],
    gate: V2LaunchGate,
) -> dict[str, Any]:
    """Build the complete record that must be durable before worker exec."""
    return {
        "controller_id": request.controller_id,
        "trajectory_id": request.trajectory_id,
        "trajectory_path": str(request.trajectory_path),
        "task_id": request.task_id,
        "round_index": request.round_index,
        "request_sha256": request.request_sha256,
        "selected_row_ids": list(request.selected_row_ids),
        "command": list(request.command),
        "gpu_uuids": list(uuids),
        "pid": -1,
        "owner": scheduler.current_owner,
        "start_time": "",
        "status": "running",
        "selected_event_budget_consumed": 0,
        "width": list(request.width),
        "source_lock_key": request.source_lock_key,
        "resolved_source_lock_sha256": request.source_lock_key,
        "source_resolution_result_sha256": request.source_resolution_result_sha256,
        "physical_plan_path": str(request.physical_plan_path),
        "physical_request_sha256": request.physical_request_sha256,
        "expected_release_sha256": request.expected_release_sha256,
        "expected_manifest_file_sha256": request.expected_manifest_file_sha256,
        "launch_gate_path": str(gate.path),
        "launch_gate_token_sha256": gate.token_sha256,
        "launch_gate_schema": LAUNCH_GATE_SCHEMA,
        "controller_generation": "v2",
        "gpu_models": {uuid: scheduler._gpu_models[uuid] for uuid in uuids},
        "lease_window": {
            "opened_wall_time": scheduler.wall_time(),
            "closed_wall_time": None,
        },
        "reservation_snapshot": {
            uuid: list(reasons)
            for uuid, reasons in sorted(scheduler._reservation_reasons.items())
        },
    }


def _atomic_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def release_launch_gate(
    scheduler: Any,
    request: V2BatchRequest,
    process: Any,
    gate: V2LaunchGate,
) -> dict[str, Any]:
    controller = (scheduler._state.get("controllers") or {}).get(
        request.controller_id
    )
    if (
        not isinstance(controller, Mapping)
        or controller.get("pid") != int(process.pid)
        or controller.get("launch_gate_path") != str(gate.path)
        or controller.get("launch_gate_token_sha256") != gate.token_sha256
    ):
        raise ValueError("real-PID controller record is not durable")
    unsigned = {
        "schema_version": LAUNCH_GATE_SCHEMA,
        "controller_id": request.controller_id,
        "request_sha256": request.request_sha256,
        "child_pid": int(process.pid),
        "owner_uid": os.getuid(),
        "gate_token_sha256": gate.token_sha256,
        "scheduler_controller_record_sha256": canonical_sha256(controller),
        "released_at_wall_time": scheduler.wall_time(),
    }
    payload = {**unsigned, "release_sha256": canonical_sha256(unsigned)}
    _atomic_private_json(gate.path, payload)
    return payload


def terminate_launch_process(process: Any) -> bool:
    try:
        if process.poll() is not None:
            return True
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
        return process.poll() is not None
    except BaseException:
        return False


def cleanup_launch_gate(gate: V2LaunchGate) -> None:
    gate.path.unlink(missing_ok=True)


def cleanup_controller_launch_gate(
    scheduler: Any, controller: Mapping[str, Any]
) -> None:
    path = Path(str(controller.get("launch_gate_path") or ""))
    gate_dir = Path(scheduler.status_dir).resolve() / "launch_gates"
    if (
        not path.is_absolute()
        or path.parent.resolve() != gate_dir
        or not path.name.startswith("gate_")
        or path.suffix != ".json"
        or not path.exists()
    ):
        return
    metadata = os.lstat(path)
    if stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid():
        path.unlink()


def launch_with_gate(
    scheduler: Any,
    request: V2BatchRequest,
    uuids: tuple[str, ...],
    locks: tuple[Any, ...],
    base_launch: Callable[[], None],
) -> None:
    gate = prepare_launch_gate(scheduler, request)
    controller = prelaunch_controller_record(scheduler, request, uuids, gate)
    controllers = {
        **(scheduler._state.get("controllers") or {}),
        request.controller_id: controller,
    }
    scheduler._persist(scheduler_status="running", controllers=controllers)
    original_env = scheduler.base_env
    original_launcher = scheduler.launcher
    captured: list[Any] = []

    def capture_launcher(*args: Any, **kwargs: Any) -> Any:
        process = original_launcher(*args, **kwargs)
        captured.append(process)
        return process

    scheduler.base_env = {**original_env, **gate.environment}
    scheduler.launcher = capture_launcher
    scheduler._locks[request.controller_id] = locks
    scheduler._prelaunch_controller_id = request.controller_id
    try:
        base_launch()
        if len(captured) != 1:
            raise RuntimeError("launch gate did not capture exactly one child")
        release_launch_gate(scheduler, request, captured[0], gate)
    except BaseException as error:
        process = captured[0] if captured else None
        if process is not None and not terminate_launch_process(process):
            raise LaunchTerminationUnconfirmed(
                "launch child death unconfirmed; retaining scheduler locks"
            ) from error
        scheduler._processes.pop(request.controller_id, None)
        scheduler._locks.pop(request.controller_id, None)
        cleanup_launch_gate(gate)
        raise
    finally:
        scheduler.base_env = original_env
        scheduler.launcher = original_launcher
        del scheduler._prelaunch_controller_id


def build_v2_trajectory_queue(result_root: Path) -> tuple[V2Trajectory, ...]:
    root = Path(result_root).resolve()
    if root != Path(V2_ROOT).resolve():
        raise ValueError("v2 result root is required; refusing v1 or non-v2 root")
    queue = tuple(
        V2Trajectory(
            variant=variant,
            seed=seed,
            path=root / "variants" / variant / f"seed_{seed}",
        )
        for variant in CORE_VARIANTS
        for seed in SEEDS
    )
    if len(queue) != 12 or len({item.path for item in queue}) != 12:
        raise ValueError("v2 scheduler requires exactly twelve unique trajectories")
    return queue


def resolved_source_lock_sha256(
    physical_plan: Mapping[str, Any],
    formal_source_result: Mapping[str, Any],
) -> str:
    """Bind the ordered physical misses to their separately resolved sources."""
    bindings = physical_plan.get("logical_row_bindings")
    physical_rows = physical_plan.get("rows")
    result_rows = formal_source_result.get("rows")
    if (
        not isinstance(bindings, list)
        or len(bindings) != 4
        or not isinstance(physical_rows, list)
        or not isinstance(result_rows, list)
        or len(result_rows) != 4
    ):
        raise ValueError("resolved-source lock requires atomic round artifacts")
    result_by_id = {
        str(row.get("candidate_id") or ""): row
        for row in result_rows
        if isinstance(row, Mapping)
    }
    physical_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in physical_rows
        if isinstance(row, Mapping)
    ]
    miss_bindings = [
        binding
        for binding in bindings
        if isinstance(binding, Mapping) and binding.get("disposition") == "miss"
    ]
    if [
        str(binding.get("candidate_id") or "") for binding in miss_bindings
    ] != physical_ids or len(result_by_id) != 4:
        raise ValueError("physical miss order differs from formal source result")
    records = []
    for binding in miss_bindings:
        candidate_id = str(binding["candidate_id"])
        resolved = result_by_id.get(candidate_id)
        if resolved is None:
            raise ValueError("formal source result omits a physical miss")
        record = {
            "resolved_source_sha256": resolved.get("resolved_source_sha256"),
            "checkpoint_sha256": resolved.get("checkpoint_sha256"),
            "onnx_sha256": resolved.get("onnx_sha256"),
        }
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in record.values()
        ):
            raise ValueError("resolved-source lock contains an invalid SHA")
        records.append(record)
    return canonical_sha256(records)


def validate_resolved_source_lock(
    recorded_sha256: str,
    physical_plan: Mapping[str, Any],
    formal_source_result: Mapping[str, Any],
) -> str:
    expected = resolved_source_lock_sha256(physical_plan, formal_source_result)
    if recorded_sha256 != expected:
        raise ValueError("resolved-source lock SHA drift")
    return expected


def resolved_source_bindings(
    physical_plan: Mapping[str, Any],
    formal_source_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return immutable resolved rows only for ordered physical misses."""
    result_by_id = {
        str(row["candidate_id"]): row
        for row in formal_source_result.get("rows") or ()
        if isinstance(row, Mapping)
    }
    bindings = []
    for row in physical_plan.get("rows") or ():
        candidate_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
        if candidate_id not in result_by_id:
            raise ValueError("physical miss has no formal resolved source")
        bindings.append(copy.deepcopy(dict(result_by_id[candidate_id])))
    return bindings


def validate_completion_receipt(
    request: V2BatchRequest,
    validator: Any,
) -> dict[str, Any]:
    """Authenticate the exact committed round before scheduler budget release."""
    receipt = validator(request.trajectory_path)
    if not isinstance(receipt, Mapping):
        raise ValueError("completion validator did not return a barrier receipt")
    required_digests = (
        "physical_terminal_batch_sha256",
        "cache_after_round_file_sha256",
        "barrier_receipt_sha256",
    )
    if (
        receipt.get("schema_version") != COMPLETION_BARRIER_SCHEMA
        or receipt.get("logical_request_sha256") != request.request_sha256
        or receipt.get("feedback_released") is not True
        or receipt.get("budget_consumed") != GPU_BATCH_SIZE
        or any(
            not isinstance(receipt.get(field), str)
            or len(str(receipt[field])) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(receipt[field])
            )
            for field in required_digests
        )
    ):
        raise ValueError("completion barrier identity or lineage admission failed")
    lineage = {
        "logical_request_sha256": request.request_sha256,
        "physical_request_sha256": request.physical_request_sha256,
        "source_resolution_result_sha256": (
            request.source_resolution_result_sha256
        ),
        "selected_row_ids": list(request.selected_row_ids),
        "cache_after_round_file_sha256": receipt[
            "cache_after_round_file_sha256"
        ],
        "barrier_receipt_sha256": receipt["barrier_receipt_sha256"],
    }
    return {
        **copy.deepcopy(dict(receipt)),
        "scheduler_completion_lineage": {
            **lineage,
            "completion_lineage_sha256": canonical_sha256(lineage),
        },
    }


def reap_scheduler_controllers(
    scheduler: Any,
    identity_matches: Callable[[Mapping[str, Any], Any, str], bool],
) -> list[str]:
    """Reap children without treating process success as feedback success."""
    controllers = dict(scheduler._state.get("controllers") or {})
    completed: list[str] = []
    budget = int(scheduler._state.get("selected_event_budget_consumed", 0))
    changed = False
    for controller_id, controller in tuple(controllers.items()):
        if controller.get("status") != "running":
            continue
        process = scheduler._processes.get(controller_id)
        if process is None:
            expected = (
                scheduler._controller_manifest.get("controllers") or {}
            ).get(controller_id) or {}
            actual = scheduler.process_inspector(int(expected.get("pid", -1)))
            if identity_matches(expected, actual, scheduler.current_owner):
                continue
            returncode = None
            admission_error = "controller_process_vanished"
        else:
            returncode = process.poll()
            if returncode is None:
                continue
            admission_error = ""
            completion_receipt: dict[str, Any] | None = None
            if returncode == 0:
                try:
                    request = scheduler._controller_request(controller)
                    if (
                        request.source_resolution_result_sha256
                        != controller.get("source_resolution_result_sha256")
                        or int(controller.get("selected_event_budget_consumed", 0))
                        != 0
                    ):
                        raise ValueError("stored completion lineage drift")
                    completion_receipt = validate_completion_receipt(
                        request, scheduler.completion_validator
                    )
                except Exception as error:
                    admission_error = (
                        f"{type(error).__name__}:completion_admission_failed"
                    )
            else:
                admission_error = "controller_returncode_nonzero"
        if returncode == 0 and not admission_error:
            status, consumed = "feedback_complete", GPU_BATCH_SIZE
            budget += GPU_BATCH_SIZE
            completed.append(controller_id)
        else:
            status, consumed = "infrastructure_retry_required", 0
        window = controller.get("lease_window")
        updates = {
            "status": status,
            "returncode": returncode,
            "selected_event_budget_consumed": consumed,
            "completion_admission_error": admission_error or None,
        }
        if status == "infrastructure_retry_required":
            updates["retry_request_sha256"] = controller["request_sha256"]
        elif completion_receipt is not None:
            updates["completion_lineage"] = completion_receipt[
                "scheduler_completion_lineage"
            ]
        if isinstance(window, Mapping):
            updates["lease_window"] = {
                **dict(window),
                "closed_wall_time": scheduler.wall_time(),
            }
        scheduler._release_locks(scheduler._locks.get(controller_id, ()))
        scheduler._locks = {
            key: value
            for key, value in scheduler._locks.items()
            if key != controller_id
        }
        scheduler._processes = {
            key: value
            for key, value in scheduler._processes.items()
            if key != controller_id
        }
        cleanup_controller_launch_gate(scheduler, controller)
        controllers[controller_id] = {**controller, **updates}
        scheduler._lease_event(
            "controller_completion_checked",
            controller_id=controller_id,
            returncode=returncode,
            completion_status=status,
            request_sha256=controller["request_sha256"],
            budget_consumed=consumed,
        )
        changed = True
    if changed:
        retrying = any(
            item.get("status") == "infrastructure_retry_required"
            for item in controllers.values()
        )
        scheduler_status = (
            "feedback_barrier_open"
            if completed
            else "infrastructure_retry_required"
            if retrying
            else str(scheduler._state.get("scheduler_status") or "initialized")
        )
        scheduler._persist(
            scheduler_status=scheduler_status,
            controllers=controllers,
            selected_event_budget_consumed=budget,
        )
    return completed


__all__ = [
    "CORE_CONTROLLER",
    "COMPLETION_BARRIER_SCHEMA",
    "GPU_BATCH_SIZE",
    "V1_ROOT",
    "V2_ROOT",
    "V2BatchRequest",
    "V2Trajectory",
    "build_v2_trajectory_queue",
    "canonical_sha256",
    "prelaunch_controller_record",
    "resolved_source_bindings",
    "resolved_source_lock_sha256",
    "reap_scheduler_controllers",
    "validate_completion_receipt",
    "validate_resolved_source_lock",
]
