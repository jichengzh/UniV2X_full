from __future__ import annotations

from functools import lru_cache
import importlib
import json
from pathlib import Path
from typing import Any, Mapping


EVENTS_PER_ROUND = 4
JSON = dict[str, Any]


class SchedulerAdapterError(RuntimeError):
    pass


class SchedulerAdapterRetry(RuntimeError):
    def __init__(self, reason: str, *, category: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.category = category


def _read_mapping(path: Path, *, label: str) -> JSON:
    if not path.is_file() or path.is_symlink():
        raise SchedulerAdapterError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchedulerAdapterError(f"{label} is unreadable or unsafe") from error
    if not isinstance(payload, Mapping):
        raise SchedulerAdapterError(f"{label} is non-mapping and unsafe")
    return dict(payload)


@lru_cache(maxsize=4)
def measurement_scheduler(v2_root: str) -> tuple[Any, Any]:
    try:
        module = importlib.import_module("scripts.stage7_core_ablation_scheduler_v2")
    except ImportError as error:
        raise SchedulerAdapterRetry(
            "canonical measurement scheduler deployment is not available",
            category="dependency_wait",
        ) from error
    scheduler_type = getattr(module, "V2Stage7Scheduler", None)
    request_type = getattr(module, "V2BatchRequest", None)
    if not callable(scheduler_type) or not callable(request_type):
        raise SchedulerAdapterError("canonical measurement scheduler API drift")
    status_dir = Path(v2_root) / "status"
    return module, scheduler_type(status_dir=status_dir)


def _batch_request(
    *,
    v2_root: Path,
    variant: str,
    seed: int,
    round_index: int,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    directory: Path,
    logical: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    module, scheduler = measurement_scheduler(str(v2_root))
    rows, bindings = logical.get("rows"), physical_plan.get("logical_row_bindings")
    if (
        not isinstance(rows, list)
        or not isinstance(bindings, list)
        or len(rows) != EVENTS_PER_ROUND
        or len(bindings) != EVENTS_PER_ROUND
    ):
        raise SchedulerAdapterError("measurement request row contract drift")
    selected_ids = tuple(
        str(item.get("candidate_id") or "")
        for item in bindings
        if isinstance(item, Mapping)
    )
    widths = tuple(
        int(value)
        for row in rows
        if isinstance(row, Mapping)
        for value in (row.get("width") or ())
    )
    request_sha = str(logical.get("measurement_request_sha256") or "")
    physical_sha = str(physical_plan.get("physical_request_sha256") or "")
    source_lock = module._requests.resolved_source_lock_sha256(
        physical_plan, source_result
    )
    command = (
        str(module.CORE_CONTROLLER),
        "--v2-root",
        str(v2_root.resolve()),
        "--variant",
        variant,
        "--seed",
        str(seed),
        "--round-index",
        str(round_index),
        "--request-sha256",
        request_sha,
        "--expected-release-sha256",
        expected_release_sha256,
        "--expected-manifest-file-sha256",
        expected_manifest_file_sha256,
    )
    return (
        module,
        scheduler,
        module.V2BatchRequest(
            trajectory_id=f"{variant}:seed_{seed}",
            trajectory_path=directory,
            round_index=round_index,
            request_sha256=request_sha,
            selected_row_ids=selected_ids,
            command=command,
            width=widths,
            expected_release_sha256=expected_release_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
            source_lock_key=source_lock,
            physical_plan_path=directory / "miss_only_physical_request.json",
            physical_request_sha256=physical_sha,
        ),
    )


def _controller(state: Mapping[str, Any], scheduler: Any, request: Any) -> Any:
    controller = (state.get("controllers") or {}).get(request.controller_id)
    if isinstance(controller, Mapping):
        if scheduler._controller_request(controller) != request:
            raise SchedulerAdapterError("scheduler controller identity drift")
        return controller
    return None


def _validate_completion(
    module: Any, scheduler: Any, controller: Mapping[str, Any], request: Any
) -> JSON:
    lineage = controller.get("completion_lineage")
    try:
        receipt = module._requests.validate_completion_receipt(
            request, scheduler.completion_validator
        )
    except Exception as error:
        raise SchedulerAdapterError(str(error)) from error
    if (
        controller.get("selected_event_budget_consumed") != EVENTS_PER_ROUND
        or not isinstance(lineage, Mapping)
        or receipt.get("scheduler_completion_lineage") != lineage
        or lineage.get("logical_request_sha256") != request.request_sha256
        or lineage.get("physical_request_sha256") != request.physical_request_sha256
        or lineage.get("selected_row_ids") != list(request.selected_row_ids)
    ):
        raise SchedulerAdapterError("scheduler completion lineage or budget drift")
    return {"status": "feedback_complete", "controller_id": request.controller_id}


def submit_or_observe(
    *,
    v2_root: Path,
    variant: str,
    seed: int,
    round_index: int,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    directory: Path,
    logical: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> JSON:
    module, scheduler, request = _batch_request(
        v2_root=v2_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        directory=directory,
        logical=logical,
        physical_plan=physical_plan,
        source_result=source_result,
    )
    state = scheduler.status()
    _controller(state, scheduler, request)
    scheduler.reap_controllers()
    state = scheduler.status()
    controller = _controller(state, scheduler, request)
    if isinstance(controller, Mapping):
        status = str(controller.get("status") or "")
        if status == "feedback_complete":
            return _validate_completion(module, scheduler, controller, request)
        if status not in {"running", "infrastructure_retry_required"}:
            raise SchedulerAdapterError("scheduler recovery status drift")
    budget = int(state.get("selected_event_budget_consumed", 0))
    resumed = scheduler.resume([request], once=True)
    state = scheduler.status()
    if int(state.get("selected_event_budget_consumed", 0)) != budget:
        raise SchedulerAdapterError("scheduler submission consumed event budget")
    controller = _controller(state, scheduler, request)
    if isinstance(controller, Mapping) and controller.get("status") == "running":
        scheduler.monitor_occupancy()
        reason = (
            "scheduler controller running"
            if request.controller_id in resumed.get("already_live", ())
            else "scheduler controller launched"
        )
        raise SchedulerAdapterRetry(reason, category="controller_running")
    if isinstance(controller, Mapping):
        if (
            controller.get("status") != "infrastructure_retry_required"
            or controller.get("selected_event_budget_consumed") != 0
            or controller.get("retry_request_sha256") != request.request_sha256
        ):
            raise SchedulerAdapterError("scheduler retry lineage drift")
        reason = str(controller.get("completion_admission_error") or "retry")
        raise SchedulerAdapterRetry(reason, category="infrastructure")
    raise SchedulerAdapterRetry("capacity", category="resource_wait")


def recover_terminal_barrier(
    config: Any,
    variant: str,
    seed: int,
    round_index: int,
    directory: Path,
) -> JSON:
    """Finalize authenticated terminal evidence without GPU work or relaunch."""
    try:
        logical = _read_mapping(
            directory / "logical_request.json", label="logical request"
        )
        physical_plan = _read_mapping(
            directory / "miss_only_physical_request.json",
            label="physical request",
        )
        source_result = _read_mapping(
            directory / "source_resolution_result.json", label="source result"
        )
        worker = importlib.import_module("scripts.stage7_core_round_worker_v2")
        online = importlib.import_module("scripts.stage7_core_online_ablation_v2")
    except ImportError as error:
        raise SchedulerAdapterRetry(
            "barrier-only recovery dependency is unavailable",
            category="dependency_wait",
        ) from error
    _module, scheduler, request = _batch_request(
        v2_root=config.v2_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        expected_release_sha256=config.expected_release_sha256,
        expected_manifest_file_sha256=config.expected_manifest_file_sha256,
        directory=directory,
        logical=logical,
        physical_plan=physical_plan,
        source_result=source_result,
    )
    controller = _controller(scheduler.status(), scheduler, request)
    status = str((controller or {}).get("status") or "")
    if (
        status not in {"running", "infrastructure_retry_required"}
        or controller.get("selected_event_budget_consumed") != 0
        or (
            status == "infrastructure_retry_required"
            and controller.get("retry_request_sha256") != request.request_sha256
        )
    ):
        raise SchedulerAdapterError("barrier-only scheduler lineage admission failed")
    request_sha = str(logical.get("measurement_request_sha256") or "")
    observed = worker.observe_round(
        config.v2_root, variant, seed, round_index, request_sha
    )
    terminal_path = Path(str(observed.get("terminal_path") or ""))
    if (
        observed.get("terminal_authenticated") is not True
        or observed.get("barrier_authenticated") is True
        or not terminal_path.is_file()
        or terminal_path.resolve().parent != directory.resolve()
    ):
        raise SchedulerAdapterError("barrier-only terminal recovery admission failed")
    return online.finalize_round(
        config.v2_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        terminal_payload_path=terminal_path,
        repo_root=config.repo_root,
        expected_release_sha256=config.expected_release_sha256,
        expected_manifest_file_sha256=config.expected_manifest_file_sha256,
    )
