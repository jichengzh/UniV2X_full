#!/usr/bin/env python3
"""Idempotent Stage7 v2 round worker over the existing actual-v3 executor.

This worker owns orchestration and lineage admission only.  It never implements
TVM build, tuning, latency, energy, numerical, AP, or graph-feature semantics.
Those remain exclusively in ``stage7_execute_actual_v3_misses_v2.sh`` and the
frozen Stage5/Stage3 actual-feedback-v3 primitives called by that script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from framework.stage7 import deployment_bundle_v2
from framework.stage7.actual_mode_v2 import validate_no_gpu_receipt
from framework.stage7.core_ablation_v2 import CORE_VARIANTS
from framework.stage7.physical_feedback_v2 import validate_physical_terminal_batch
from scripts import stage7_core_online_ablation_v2 as core_online
from scripts.stage7_core_online_ablation_v2 import (
    validate_committed_barrier,
    validate_formal_v2_root,
)
from scripts.stage7_source_scheduler_v2 import validate_measurement_artifacts

JSON = dict[str, Any]
SEEDS = (20260718, 20260719, 20260720)
H800_MODELS = frozenset({"NVIDIA H800", "NVIDIA H800 PCIe", "NVIDIA H800 SXM", "NVIDIA H800 NVL"})
SCHEDULER_SCHEMA = "stage7_ablation_scheduler_state_v1"
PHYSICAL_SCHEMA = "stage7_actual_v3_miss_only_physical_request_v2"
ADMISSION_SCHEMA = "stage7_actual_v3_executor_admission_v2"
NO_GPU_RECEIPT = Path("audits/no_gpu_dry_run/dry_run_receipt.json")
SCHEDULER_STATE = Path("status/scheduler_state.json")
_UUID = re.compile(r"GPU-[0-9A-Fa-f-]{16,}")
_SHA = re.compile(r"[0-9a-f]{64}")
_GATE_FIELDS = frozenset({
    "schema_version", "controller_id", "request_sha256", "child_pid", "owner_uid", "gate_token_sha256",
    "scheduler_controller_record_sha256", "released_at_wall_time", "release_sha256",
})
_GATE_FACTORY_KEY = object()

class RetryableRoundError(RuntimeError):
    """An infrastructure/evidence failure that must retain the request budget."""

class _AuthenticatedLaunchGate:
    __slots__ = ("payload", "scheduler_state_path")

    def __init__(self, payload: JSON, key: object,
                 scheduler_state_path: Path | None = None) -> None:
        if key is not _GATE_FACTORY_KEY:
            raise ValueError("launch gate receipt is not authenticated")
        self.payload = dict(payload)
        self.scheduler_state_path = scheduler_state_path

def _read_mapping(path: Path, *, label: str) -> JSON:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unavailable or invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)

def _require_sha(value: Any, *, label: str) -> str:
    text = str(value or "")
    if _SHA.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text

def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

def _canonical_round(root: Path, variant: str, seed: int, round_index: int) -> Path:
    if variant not in CORE_VARIANTS:
        raise ValueError("variant is outside the frozen Stage7 core contract")
    if seed not in SEEDS:
        raise ValueError("seed is outside the frozen Stage7 core contract")
    if round_index not in range(4):
        raise ValueError("round-index is outside the frozen range 0..3")
    return root / "variants" / variant / f"seed_{seed}" / f"round_{round_index:02d}"

def _controller_id(variant: str, seed: int, round_index: int) -> str:
    return f"{variant}:seed_{seed}:round_{round_index}"

def _await_launch_gate(
    *, v2_root: Path, variant: str, seed: int, round_index: int,
    request_sha256: str, environ: Mapping[str, str],
    pid_getter: Callable[[], int] = os.getpid,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> _AuthenticatedLaunchGate:
    """Block until the scheduler proves this real PID is durably recorded."""
    root = Path(v2_root).resolve(strict=True)
    request_sha = _require_sha(request_sha256, label="launch gate request SHA")
    controller_id = _controller_id(variant, seed, round_index)
    raw_path = environ.get("STAGE7_V2_LAUNCH_GATE_PATH", "")
    token = environ.get("STAGE7_V2_LAUNCH_GATE_TOKEN", "")
    if not raw_path:
        raise ValueError("launch gate path is required")
    gate_path = Path(raw_path)
    if not gate_path.is_absolute():
        raise ValueError("launch gate path must be absolute")
    token_sha = hashlib.sha256(_require_sha(
        token, label="launch gate token"
    ).encode("utf-8")).hexdigest()
    if environ.get("STAGE7_V2_LAUNCH_GATE_TIMEOUT_SECONDS") != "60":
        raise ValueError("launch gate timeout must equal 60 seconds")
    gate_dir = gate_path.parent
    if gate_dir != root / "status/launch_gates":
        raise ValueError("launch gate canonical directory drift")
    status_metadata, directory_metadata = os.lstat(root / "status"), os.lstat(gate_dir)
    if (
        not stat.S_ISDIR(status_metadata.st_mode)
        or not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
        or directory_metadata.st_uid != os.getuid()
    ):
        raise ValueError("launch gate canonical directory identity/mode drift")
    controller_sha = hashlib.sha256(controller_id.encode("utf-8")).hexdigest()
    expected_name = f"gate_{controller_sha[:16]}_{request_sha[:16]}_{token_sha[:16]}.json"
    if gate_path.name != expected_name:
        raise ValueError("launch gate canonical path drift")
    deadline = monotonic() + 60.0
    while not os.path.lexists(gate_path):
        if monotonic() >= deadline:
            raise TimeoutError("launch gate timed out before scheduler release")
        sleeper(0.05)
    metadata = os.lstat(gate_path)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
    ):
        raise ValueError("launch gate file identity/mode drift")
    payload = _read_mapping(gate_path, label="launch gate release")
    unsigned = {key: value for key, value in payload.items() if key != "release_sha256"}
    if set(payload) != _GATE_FIELDS or payload.get("release_sha256") != _canonical_sha(
        unsigned
    ):
        raise ValueError("launch gate release SHA/shape drift")
    state_path = root / SCHEDULER_STATE
    state_metadata = os.lstat(state_path)
    if not stat.S_ISREG(state_metadata.st_mode) or state_metadata.st_uid != os.getuid():
        raise ValueError("launch gate scheduler state file identity drift")
    state = _read_mapping(state_path, label="scheduler state")
    controllers = state.get("controllers")
    controller = controllers.get(controller_id) if isinstance(controllers, Mapping) else None
    if (
        payload.get("schema_version") != "stage7_v2_controller_launch_gate_v1"
        or payload.get("controller_id") != controller_id
        or payload.get("request_sha256") != request_sha
        or payload.get("child_pid") != pid_getter()
        or payload.get("owner_uid") != os.getuid()
        or payload.get("gate_token_sha256") != token_sha
    ):
        raise ValueError("launch gate process/request identity drift")
    if (
        not isinstance(controller, Mapping)
        or controller.get("controller_id") != controller_id
        or controller.get("request_sha256") != request_sha
        or controller.get("pid") != pid_getter()
        or controller.get("launch_gate_path") != str(gate_path)
        or controller.get("launch_gate_token_sha256") != token_sha
        or controller.get("launch_gate_schema") != "stage7_v2_controller_launch_gate_v1"
        or payload.get("scheduler_controller_record_sha256")
        != _canonical_sha(controller)
    ):
        raise ValueError("launch gate scheduler state binding drift")
    released_at = payload.get("released_at_wall_time")
    if isinstance(released_at, bool) or not isinstance(released_at, (int, float)):
        raise ValueError("launch gate release timestamp drift")
    return _AuthenticatedLaunchGate(payload, _GATE_FACTORY_KEY, state_path)

def _ordered_uuids(environment: Mapping[str, str]) -> tuple[str, ...]:
    raw = environment.get("CUDA_VISIBLE_DEVICES", "")
    values = tuple(part.strip() for part in raw.split(","))
    if (
        len(values) != 4
        or len(set(values)) != 4
        or any(_UUID.fullmatch(value) is None for value in values)
        or raw != ",".join(values)
    ):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be exactly four unique ordered H800 UUIDs"
        )
    return values

def _validate_scheduler_controller(
    *,
    state: Mapping[str, Any],
    controller_id: str,
    round_dir: Path,
    variant: str,
    seed: int,
    round_index: int,
    request_sha256: str,
    physical_request_sha256: str,
    source_result_sha256: str,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    uuids: Sequence[str],
) -> JSON:
    controllers = state.get("controllers")
    if (
        state.get("schema_version") != SCHEDULER_SCHEMA
        or not isinstance(controllers, Mapping)
        or not isinstance(controllers.get(controller_id), Mapping)
    ):
        raise ValueError("scheduler state/controller admission is missing")
    controller = dict(controllers[controller_id])
    models = controller.get("gpu_models")
    expected_trajectory = f"{variant}:seed_{seed}"
    expected_source_lock = str(
        controller.get("resolved_source_lock_sha256")
        or controller.get("source_lock_key")
        or ""
    )
    if (
        controller.get("controller_id", controller_id) != controller_id
        or controller.get("trajectory_id") != expected_trajectory
        or Path(str(controller.get("trajectory_path") or "")).resolve()
        != round_dir
        or controller.get("round_index") != round_index
        or controller.get("request_sha256") != request_sha256
        or controller.get("physical_request_sha256") != physical_request_sha256
        or controller.get("source_resolution_result_sha256")
        != source_result_sha256
        or controller.get("expected_release_sha256")
        != expected_release_sha256
        or controller.get("expected_manifest_file_sha256")
        != expected_manifest_file_sha256
        or tuple(controller.get("gpu_uuids") or ()) != tuple(uuids)
        or controller.get("status") not in {"running", "feedback_complete"}
        or not isinstance(models, Mapping)
        or set(models) != set(uuids)
        or any(models.get(uuid) not in H800_MODELS for uuid in uuids)
    ):
        raise ValueError(
            "scheduler controller/request/deployment pin/H800 UUID lineage drift"
        )
    _require_sha(expected_source_lock, label="resolved source lock")
    if (
        controller.get("source_lock_key")
        and controller.get("source_lock_key") != expected_source_lock
    ):
        raise ValueError("scheduler resolved source lock identity drift")
    return {**controller, "resolved_source_lock_sha256": expected_source_lock}

def _validate_lineage(
    *,
    artifacts: Mapping[str, Any],
    request_sha256: str,
    contract_sha256: str,
) -> JSON:
    logical = artifacts["logical_request"]
    source_plan = artifacts["source_plan"]
    source_result = artifacts["source_result"]
    exact = artifacts["exact_selection_binding"]
    reveal = artifacts["cache_reveal"]
    physical = artifacts["physical_plan"]
    admission = artifacts["executor_admission"]
    exact_logical = exact.get("logical_request")
    logical_rows = logical.get("rows")
    bindings = physical.get("logical_row_bindings")
    physical_rows = physical.get("rows")
    physical_sha = _require_sha(
        physical.get("physical_request_sha256"), label="physical request SHA"
    )
    source_plan_sha = _require_sha(
        source_plan.get("source_resolution_plan_sha256"),
        label="source resolution plan SHA",
    )
    source_result_sha = _require_sha(
        source_result.get("source_resolution_result_sha256"),
        label="source resolution result SHA",
    )
    if (
        logical.get("measurement_request_sha256") != request_sha256
        or source_plan.get("logical_request_sha256") != request_sha256
        or source_result.get("logical_request_sha256") != request_sha256
        or source_result.get("source_resolution_plan_sha256") != source_plan_sha
        or not isinstance(exact_logical, Mapping)
        or exact_logical.get("logical_request_sha256") != request_sha256
        or reveal.get("logical_request_sha256", request_sha256) != request_sha256
        or physical.get("schema_version") != PHYSICAL_SCHEMA
        or physical.get("logical_request_sha256") != request_sha256
        or admission.get("schema_version") != ADMISSION_SCHEMA
        or admission.get("admission_passed") is not True
        or admission.get("contract_sha256") != contract_sha256
        or admission.get("logical_request_sha256") != request_sha256
        or admission.get("physical_request_sha256") != physical_sha
        or admission.get("execution_primitive")
        != "existing_stage5_stage3_actual_feedback_v3"
        or admission.get("gpu_jobs_launched") != 0
    ):
        raise ValueError("request/source/exact-cache/executor admission lineage drift")
    if (
        not isinstance(logical_rows, list)
        or len(logical_rows) != 4
        or not isinstance(bindings, list)
        or len(bindings) != 4
        or not isinstance(physical_rows, list)
        or physical.get("logical_row_count") != 4
        or physical.get("physical_row_count") != len(physical_rows)
        or len(physical_rows) not in range(5)
        or admission.get("logical_row_count") != 4
        or admission.get("physical_row_count") != len(physical_rows)
    ):
        raise ValueError("four-row physical/executor admission shape drift")
    logical_ids = tuple(
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in logical_rows
        if isinstance(row, Mapping)
    )
    binding_ids = tuple(
        str(row.get("candidate_id") or "")
        for row in bindings
        if isinstance(row, Mapping)
    )
    miss_ids = tuple(
        str(row.get("candidate_id") or "")
        for row in bindings
        if isinstance(row, Mapping) and row.get("disposition") == "miss"
    )
    physical_ids = tuple(
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in physical_rows
        if isinstance(row, Mapping)
    )
    if (
        logical_ids != binding_ids
        or len(set(logical_ids)) != 4
        or miss_ids != physical_ids
    ):
        raise ValueError("logical/source/cache physical row lineage drift")
    return {
        "physical_request_sha256": physical_sha,
        "source_resolution_plan_sha256": source_plan_sha,
        "source_resolution_result_sha256": source_result_sha,
        "physical_row_count": len(physical_rows),
    }

def _validate_no_gpu_receipt(
    root: Path,
    *,
    frozen_repo_root: Path,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
) -> JSON:
    receipt_path = root / NO_GPU_RECEIPT
    receipt = _read_mapping(receipt_path, label="no-GPU receipt")
    deployment = deployment_bundle_v2.validate_deployment_bundle(
        root,
        frozen_repo_root=frozen_repo_root,
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    return validate_no_gpu_receipt(
        receipt,
        root=root,
        frozen_repo_root=frozen_repo_root,
        deployment_bundle_sha256=deployment["deployment_bundle_sha256"],
    )

def _terminal_path(round_dir: Path, physical_sha256: str) -> Path:
    return (
        round_dir
        / "actual_v3_execution"
        / physical_sha256
        / "terminal_payload.json"
    )

def _validate_terminal(path: Path, *, request_sha256: str, physical_sha256: str) -> JSON:
    terminal = validate_physical_terminal_batch(
        _read_mapping(path, label="actual-v3 terminal payload")
    )
    projection = terminal.get("projection_artifact")
    execution_contract = _read_mapping(
        path.parent / "execution_contract.json",
        label="actual-v3 execution contract",
    )
    unsigned_contract = {
        key: value
        for key, value in execution_contract.items()
        if key != "execution_contract_sha256"
    }
    if (
        not isinstance(projection, Mapping)
        or projection.get("measurement_request_sha256") != request_sha256
        or execution_contract.get("schema_version")
        != "stage7_actual_v3_execution_contract_v2"
        or execution_contract.get("logical_request_sha256") != request_sha256
        or execution_contract.get("physical_request_sha256") != physical_sha256
        or execution_contract.get("dry_run") is not False
        or execution_contract.get("execution_contract_sha256")
        != _canonical_sha(unsigned_contract)
    ):
        raise ValueError("authenticated terminal request/source/cache lineage drift")
    return terminal

def _executor_argv(
    *,
    code_root: Path,
    root: Path,
    round_dir: Path,
    request_sha256: str,
    contract_sha256: str,
    physical_sha256: str,
    source_plan_sha256: str,
    source_result_sha256: str,
    source_lock_sha256: str,
    uuids: Sequence[str],
    controller_id: str,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
) -> tuple[str, ...]:
    execution = round_dir / "actual_v3_execution" / physical_sha256
    return (
        str(code_root / "scripts/stage7_execute_actual_v3_misses_v2.sh"),
        "--physical-plan-json",
        str(round_dir / "miss_only_physical_request.json"),
        "--output-dir",
        str(execution),
        "--formal-v2-root",
        str(root),
        "--expected-release-sha256",
        expected_release_sha256,
        "--expected-manifest-file-sha256",
        expected_manifest_file_sha256,
        "--logical-request-json",
        str(round_dir / "logical_request.json"),
        "--selection-binding-json",
        str(round_dir / "exact_selection_binding.json"),
        "--cache-reveal-json",
        str(round_dir / "cache_reveal.json"),
        "--cache-snapshot-json",
        str(round_dir / "cache_snapshot_before_reveal.json"),
        "--no-gpu-receipt-json",
        str(root / NO_GPU_RECEIPT),
        "--admission-json",
        str(execution / "miss_execution_admission.json"),
        "--request-sha256",
        request_sha256,
        "--executor-admission-json",
        str(round_dir / "executor_admission.json"),
        "--contract-sha256",
        contract_sha256,
        "--source-plan-json",
        str(round_dir / "source_resolution_plan.json"),
        "--source-result-json",
        str(round_dir / "source_resolution_result.json"),
        "--source-plan-sha256",
        source_plan_sha256,
        "--source-result-sha256",
        source_result_sha256,
        "--resolved-source-lock-sha256",
        source_lock_sha256,
        "--gpu-uuids",
        ",".join(uuids),
        "--scheduler-state-json",
        str(root / SCHEDULER_STATE),
        "--controller-id",
        controller_id,
    )

def _valid_existing_barrier(
    round_dir: Path, *, request_sha256: str
) -> JSON | None:
    path = round_dir / "atomic_feedback_barrier.json"
    if not path.is_file():
        return None
    barrier = validate_committed_barrier(round_dir)
    if (
        barrier.get("logical_request_sha256") != request_sha256
        or barrier.get("feedback_released") is not True
        or barrier.get("budget_consumed") != 4
        or barrier.get("silent_surrogate_fallback_count") != 0
    ):
        raise ValueError("committed feedback barrier request/budget lineage drift")
    return dict(barrier)

def observe_round(
    v2_root: Path,
    variant: str,
    seed: int,
    round_index: int,
    request_sha256: str,
) -> JSON:
    """Read-only observer used by persistent orchestration and recovery."""
    root = Path(v2_root).resolve()
    request_sha = _require_sha(request_sha256, label="logical request SHA")
    round_dir = _canonical_round(root, variant, seed, round_index)
    physical_path = round_dir / "miss_only_physical_request.json"
    physical_sha = None
    terminal_path = None
    terminal_present = False
    terminal_authenticated = False
    barrier_present = (round_dir / "atomic_feedback_barrier.json").is_file()
    barrier_authenticated = False
    if physical_path.is_file():
        physical = _read_mapping(physical_path, label="physical request")
        physical_sha = physical.get("physical_request_sha256")
        if _SHA.fullmatch(str(physical_sha or "")):
            terminal_path = _terminal_path(round_dir, str(physical_sha))
            terminal_present = terminal_path.is_file()
            if terminal_present:
                _validate_terminal(
                    terminal_path,
                    request_sha256=request_sha,
                    physical_sha256=str(physical_sha),
                )
                terminal_authenticated = True
    if barrier_present:
        barrier_authenticated = (
            _valid_existing_barrier(
                round_dir, request_sha256=request_sha
            )
            is not None
        )
    return {
        "controller_id": _controller_id(variant, seed, round_index),
        "request_sha256": request_sha,
        "round_dir": str(round_dir),
        "physical_request_sha256": physical_sha,
        "terminal_path": str(terminal_path) if terminal_path else None,
        "terminal_present": terminal_present,
        "terminal_authenticated": terminal_authenticated,
        "barrier_present": barrier_present,
        "barrier_authenticated": barrier_authenticated,
        "state": (
            "feedback_complete"
            if barrier_authenticated
            else "terminal_ready"
            if terminal_authenticated
            else "execution_pending"
        ),
    }

def execute_round(
    *,
    v2_root: Path,
    variant: str,
    seed: int,
    round_index: int,
    request_sha256: str,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    subprocess_runner: Callable[..., Any] = subprocess.run,
    finalizer: Callable[..., Mapping[str, Any]] = core_online.finalize_round,
    environ: Mapping[str, str] | None = None,
    launch_gate_receipt: _AuthenticatedLaunchGate | None = None,
) -> JSON:
    """Execute or recover one indivisible four-row round and open its barrier."""
    environment = dict(os.environ if environ is None else environ)
    if launch_gate_receipt is None:
        launch_gate_receipt = _await_launch_gate(
            v2_root=v2_root,
            variant=variant,
            seed=seed,
            round_index=round_index,
            request_sha256=request_sha256,
            environ=environment,
        )
    if (
        not isinstance(launch_gate_receipt, _AuthenticatedLaunchGate)
        or launch_gate_receipt.payload.get("controller_id")
        != _controller_id(variant, seed, round_index)
        or launch_gate_receipt.payload.get("request_sha256") != request_sha256
    ):
        raise ValueError("authenticated launch gate receipt identity drift")
    root = Path(v2_root).resolve()
    request_sha = _require_sha(request_sha256, label="logical request SHA")
    release_pin = _require_sha(
        expected_release_sha256, label="external deployment release pin"
    )
    manifest_pin = _require_sha(
        expected_manifest_file_sha256,
        label="external deployment manifest-file pin",
    )
    round_dir = _canonical_round(root, variant, seed, round_index)
    uuids = _ordered_uuids(environment)
    frozen_repo_root = Path(
        environment.get("STAGE7_FROZEN_REPO_ROOT", str(CODE_ROOT))
    ).resolve()
    code_root = Path(
        environment.get("STAGE7_BUNDLE_CODE_ROOT", str(CODE_ROOT))
    ).resolve()
    validated_root = validate_formal_v2_root(root, repo_root=frozen_repo_root)
    if Path(validated_root["root"]).resolve() != root:
        raise ValueError("formal v2 root validator returned a different root")
    contract_sha = _require_sha(
        validated_root["contract"].get("contract_sha256"),
        label="formal contract SHA",
    )
    artifacts = validate_measurement_artifacts(round_dir)
    lineage = _validate_lineage(
        artifacts=artifacts,
        request_sha256=request_sha,
        contract_sha256=contract_sha,
    )
    _validate_no_gpu_receipt(
        root,
        frozen_repo_root=frozen_repo_root,
        expected_release_sha256=release_pin,
        expected_manifest_file_sha256=manifest_pin,
    )
    controller_id = _controller_id(variant, seed, round_index)
    state_path = launch_gate_receipt.scheduler_state_path or root / SCHEDULER_STATE
    state = _read_mapping(state_path, label="scheduler state")
    controller = _validate_scheduler_controller(
        state=state,
        controller_id=controller_id,
        round_dir=round_dir,
        variant=variant,
        seed=seed,
        round_index=round_index,
        request_sha256=request_sha,
        physical_request_sha256=lineage["physical_request_sha256"],
        source_result_sha256=lineage["source_resolution_result_sha256"],
        expected_release_sha256=release_pin,
        expected_manifest_file_sha256=manifest_pin,
        uuids=uuids,
    )
    barrier = _valid_existing_barrier(round_dir, request_sha256=request_sha)
    if barrier is not None:
        return {
            "status": "feedback_complete",
            "controller_id": controller_id,
            "request_sha256": request_sha,
            "physical_row_count": lineage["physical_row_count"],
            "gpu_work_expected": lineage["physical_row_count"] > 0,
            "executor_relaunched": False,
            "terminal_reused": True,
            "barrier": barrier,
        }
    if controller.get("status") == "feedback_complete":
        raise ValueError(
            "scheduler feedback_complete state lacks an authenticated feedback barrier"
        )
    terminal_path = _terminal_path(
        round_dir, lineage["physical_request_sha256"]
    )
    terminal_reused = terminal_path.is_file()
    if terminal_reused:
        terminal = _validate_terminal(
            terminal_path,
            request_sha256=request_sha,
            physical_sha256=lineage["physical_request_sha256"],
        )
    else:
        argv = _executor_argv(
            code_root=code_root,
            root=root,
            round_dir=round_dir,
            request_sha256=request_sha,
            contract_sha256=contract_sha,
            physical_sha256=lineage["physical_request_sha256"],
            source_plan_sha256=lineage["source_resolution_plan_sha256"],
            source_result_sha256=lineage["source_resolution_result_sha256"],
            source_lock_sha256=controller["resolved_source_lock_sha256"],
            uuids=uuids,
            controller_id=controller_id,
            expected_release_sha256=release_pin,
            expected_manifest_file_sha256=manifest_pin,
        )
        try:
            subprocess_runner(
                list(argv),
                env={**environment, "CUDA_VISIBLE_DEVICES": ",".join(uuids)},
                cwd=round_dir,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RetryableRoundError(
                "actual-v3 executor failed; retry the same logical request SHA"
            ) from error
        if not terminal_path.is_file():
            raise RetryableRoundError(
                "actual-v3 terminal evidence is incomplete; no budget was claimed"
            )
        try:
            terminal = _validate_terminal(
                terminal_path,
                request_sha256=request_sha,
                physical_sha256=lineage["physical_request_sha256"],
            )
        except (OSError, ValueError) as error:
            raise RetryableRoundError(
                "actual-v3 terminal evidence is invalid; no budget was claimed"
            ) from error
    if terminal.get("barrier_release_allowed") is False:
        raise RetryableRoundError(
            "retryable infrastructure/evidence terminal blocks the feedback barrier"
        )
    try:
        finalizer(
            root,
            variant=variant,
            seed=seed,
            round_index=round_index,
            terminal_payload_path=terminal_path,
            repo_root=frozen_repo_root,
        )
        barrier = _valid_existing_barrier(round_dir, request_sha256=request_sha)
    except (OSError, ValueError) as error:
        raise RetryableRoundError(
            "atomic feedback barrier finalization failed; no budget was claimed"
        ) from error
    if barrier is None:
        raise RetryableRoundError(
            "atomic feedback barrier is missing; no budget was claimed"
        )
    return {
        "status": "feedback_complete",
        "controller_id": controller_id,
        "request_sha256": request_sha,
        "physical_row_count": lineage["physical_row_count"],
        "gpu_work_expected": lineage["physical_row_count"] > 0,
        "executor_relaunched": not terminal_reused,
        "terminal_reused": terminal_reused,
        "barrier": barrier,
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--variant", choices=CORE_VARIANTS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--round-index", choices=range(4), type=int, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        environment = dict(os.environ)
        launch_gate_receipt = _await_launch_gate(
            v2_root=args.v2_root,
            variant=args.variant,
            seed=args.seed,
            round_index=args.round_index,
            request_sha256=args.request_sha256,
            environ=environment,
        )
        result = execute_round(
            v2_root=args.v2_root,
            variant=args.variant,
            seed=args.seed,
            round_index=args.round_index,
            request_sha256=args.request_sha256,
            expected_release_sha256=args.expected_release_sha256,
            expected_manifest_file_sha256=args.expected_manifest_file_sha256,
            environ=environment,
            launch_gate_receipt=launch_gate_receipt,
        )
    except RetryableRoundError as error:
        print(
            json.dumps(
                {
                    "status": "infrastructure_retry_required",
                    "request_sha256": args.request_sha256,
                    "selected_event_budget_consumed": 0,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as error:
        print(json.dumps({
            "status": "stage7_round_admission_failed", "request_sha256": args.request_sha256,
            "selected_event_budget_consumed": 0, "gpu_jobs_launched": 0, "error": str(error),
        }, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
