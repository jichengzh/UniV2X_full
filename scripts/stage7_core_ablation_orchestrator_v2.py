from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage7.core_ablation_v2 import CORE_VARIANTS, canonical_sha256
from framework.stage7.no_gpu_support_v2 import no_gpu_integration_result_closed, validate_no_gpu_integration_result
from scripts.stage7_ablation_scheduler_v1 import (
    append_jsonl as _append_jsonl,
    atomic_write_json as _atomic_write_json,
)


BOOTSTRAP = "BOOTSTRAP"
PRE_GATE_SELECTION, PRE_GATE_SOURCE = "PRE_GATE_SELECTION", "PRE_GATE_SOURCE"
NO_GPU_GATE = "NO_GPU_GATE"
PILOT_20260718, PILOT_AUDIT = "PILOT_20260718", "PILOT_AUDIT"
SEED_20260719, SEED_20260720 = "SEED_20260719", "SEED_20260720"
FINALIZE, COMPLETE = "FINALIZE", "COMPLETE"

FREEZE_SOURCE_PLAN, RESOLVE_SOURCE = "freeze_source_plan", "resolve_source"
RESOLVE_PRE_GATE_SOURCE = "resolve_pre_gate_source"
BIND_EXACT = "bind_exact"
EXECUTE_MEASUREMENT = "execute_measurement"
FINALIZE_TERMINAL_BARRIER = "finalize_terminal_barrier"

PILOT_SEED = 20260718
SEEDS = (20260718, 20260719, 20260720)
ROUNDS = tuple(range(4))
EVENTS_PER_ROUND = 4
JSON = dict[str, Any]
DESCRIPTION = "Artifact-derived persistent Stage7 core-ablation orchestrator"


OrchestrationContractError = type("OrchestrationContractError", (RuntimeError,), {})


class RetryableOrchestrationError(RuntimeError):
    def __init__(self, reason: str, *, category: str = "resource_wait") -> None:
        super().__init__(reason)
        self.reason = reason
        self.category = category


def _read_mapping(path: Path, *, label: str) -> JSON:
    if not path.is_file() or path.is_symlink():
        raise OrchestrationContractError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OrchestrationContractError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping):
        raise OrchestrationContractError(f"{label} is not a mapping")
    return copy.deepcopy(dict(payload))


@dataclass(frozen=True)
class OrchestratorConfig:
    v1_root: Path
    v2_root: Path
    repo_root: Path
    prepare_inputs: Mapping[str, Any]
    embedded_source_manifest: Mapping[str, Any]
    frozen_onnx: Path
    frozen_onnx_sha256: str
    expected_release_sha256: str
    expected_manifest_file_sha256: str
    poll_seconds: float = 30.0
    pre_gate_source_only: bool = False

    def __post_init__(self) -> None:
        for field in ("v1_root", "v2_root", "repo_root", "frozen_onnx"):
            value = Path(getattr(self, field))
            if not value.is_absolute():
                raise ValueError(f"{field} must be absolute")
            object.__setattr__(self, field, value)
        if not 0 < self.poll_seconds <= 60:
            raise ValueError("poll_seconds must be in (0, 60]")
        if not isinstance(self.pre_gate_source_only, bool):
            raise ValueError("pre_gate_source_only must be boolean")
        for field, label in (
            ("frozen_onnx_sha256", "frozen ONNX SHA256"),
            ("expected_release_sha256", "release SHA256"),
            ("expected_manifest_file_sha256", "manifest-file SHA256"),
        ):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{label} is invalid")
        object.__setattr__(
            self, "prepare_inputs", copy.deepcopy(dict(self.prepare_inputs))
        )
        object.__setattr__(
            self,
            "embedded_source_manifest",
            copy.deepcopy(dict(self.embedded_source_manifest)),
        )


PrepareInitialize = Callable[[OrchestratorConfig, int], Mapping[str, Any]]
NoGpuGate = Callable[[OrchestratorConfig], Mapping[str, Any]]
InspectCanonical = Callable[[OrchestratorConfig], Mapping[str, Any]]
ValidateRuntime = Callable[
    [OrchestratorConfig, int, Mapping[str, Any]], Mapping[str, Any]
]
RoundStep = Callable[[OrchestratorConfig, str, str, int, int], Mapping[str, Any]]
SourceStep = Callable[[OrchestratorConfig, str, str, int, int], Mapping[str, Any]]
UnaryStep = Callable[[OrchestratorConfig], Mapping[str, Any]]


@dataclass(frozen=True)
class OrchestratorDependencies:
    prepare_initialize: PrepareInitialize
    run_no_gpu_gate: NoGpuGate
    inspect_canonical: InspectCanonical
    validate_runtime: ValidateRuntime
    source_step: SourceStep
    round_step: RoundStep
    audit_pilot: UnaryStep
    gate_later_seeds: UnaryStep
    finalize: UnaryStep
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time


def _round_record(
    snapshot: Mapping[str, Any], variant: str, seed: int, round_index: int
) -> Mapping[str, Any]:
    trajectories = snapshot.get("trajectories")
    if not isinstance(trajectories, Mapping):
        return {}
    trajectory = trajectories.get(f"{variant}:{seed}")
    if not isinstance(trajectory, Mapping):
        return {}
    rounds = trajectory.get("rounds")
    if not isinstance(rounds, Mapping):
        return {}
    row = rounds.get(str(round_index))
    return row if isinstance(row, Mapping) else {}


def _seed_complete(snapshot: Mapping[str, Any], seed: int) -> bool:
    for variant in CORE_VARIANTS:
        total = 0
        for round_index in ROUNDS:
            record = _round_record(snapshot, variant, seed, round_index)
            if record.get("barrier") is not True:
                return False
            budget = record.get("selected_events")
            if isinstance(budget, bool) or not isinstance(budget, int):
                return False
            total += budget
        if total != len(ROUNDS) * EVENTS_PER_ROUND:
            return False
    return True


def _derive_phase(
    snapshot: Mapping[str, Any], *, pre_gate_source_only: bool = False
) -> str:
    if snapshot.get("prepared") is not True:
        return BOOTSTRAP
    if snapshot.get("initialized") is not True:
        return BOOTSTRAP
    if pre_gate_source_only and snapshot.get("no_gpu_gate_passed") is not True:
        round_zero = [
            _round_record(snapshot, variant, PILOT_SEED, 0)
            for variant in CORE_VARIANTS
        ]
        if any(record.get("request_frozen") is not True for record in round_zero):
            return PRE_GATE_SELECTION
        if any(record.get("source_ready") is not True for record in round_zero):
            return PRE_GATE_SOURCE
    if snapshot.get("no_gpu_gate_passed") is not True:
        return NO_GPU_GATE
    if not _seed_complete(snapshot, PILOT_SEED):
        return PILOT_20260718
    if (
        snapshot.get("pilot_audit_passed") is not True
        or snapshot.get("later_seed_gate_passed") is not True
    ):
        return PILOT_AUDIT
    if not _seed_complete(snapshot, 20260719):
        return SEED_20260719
    if not _seed_complete(snapshot, 20260720):
        return SEED_20260720
    if snapshot.get("finalization_complete") is not True:
        return FINALIZE
    return COMPLETE


def _validate_no_gpu_result(result: Mapping[str, Any]) -> None:
    try:
        validate_no_gpu_integration_result(
            result, variants=CORE_VARIANTS, events_per_round=EVENTS_PER_ROUND
        )
    except ValueError as error:
        raise OrchestrationContractError(str(error)) from error


def _next_round_operation(
    snapshot: Mapping[str, Any], seed: int
) -> Optional[tuple[str, str, int, int]]:
    for variant in CORE_VARIANTS:
        for round_index in ROUNDS:
            record = _round_record(snapshot, variant, seed, round_index)
            if (
                round_index
                and _round_record(snapshot, variant, seed, round_index - 1).get(
                    "barrier"
                )
                is not True
            ):
                break
            if record.get("barrier") is True:
                continue
            if record.get("request_frozen") is not True:
                return FREEZE_SOURCE_PLAN, variant, seed, round_index
            if record.get("source_ready") is not True:
                return RESOLVE_SOURCE, variant, seed, round_index
            if record.get("exact_bound") is not True:
                return BIND_EXACT, variant, seed, round_index
            if record.get("terminal") is not True:
                return EXECUTE_MEASUREMENT, variant, seed, round_index
            return FINALIZE_TERMINAL_BARRIER, variant, seed, round_index
    return None


class PersistentCoreAblationOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        dependencies: Optional[OrchestratorDependencies] = None,
        pid: Optional[int] = None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies or default_dependencies()
        self.pid = os.getpid() if pid is None else pid
        if isinstance(self.pid, bool) or not isinstance(self.pid, int) or self.pid <= 0:
            raise ValueError("orchestrator PID is invalid")
        self.tick_index = 0

    def _validate_tick_identity_and_pins(self) -> JSON:
        state = _read_mapping(
            self.config.v2_root / "prepare_state.json", label="prepare state"
        )
        if state.get("orchestrator_pid") != self.pid:
            raise OrchestrationContractError("orchestrator PID identity drift")
        if (
            state.get("deployment_release_sha256")
            != self.config.expected_release_sha256
            or state.get("deployment_manifest_file_sha256")
            != self.config.expected_manifest_file_sha256
        ):
            raise OrchestrationContractError("external pin drift")
        validated = copy.deepcopy(
            dict(
                self.dependencies.validate_runtime(
                    self.config, self.pid, copy.deepcopy(state)
                )
            )
        )
        if validated.get("process_identity") != state.get(
            "orchestrator_process_identity"
        ):
            raise OrchestrationContractError("orchestrator process identity drift")
        for field in (
            "deployment_manifest_sha256",
            "deployment_manifest_file_sha256",
            "deployment_bundle_sha256",
            "deployment_release_sha256",
            "executor_verification",
        ):
            if validated.get(field) != state.get(field):
                raise OrchestrationContractError(f"runtime {field} pin drift")
        return validated

    def _persist(
        self,
        *,
        phase: str,
        status: str,
        event: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> JSON:
        now = self.dependencies.clock()
        unsigned = {
            "schema_version": "stage7_core_ablation_orchestrator_state_v2",
            "orchestrator_pid": self.pid,
            "phase": phase,
            "status": status,
            "tick_index": self.tick_index,
            "updated_wall_time": now,
            "canonical_artifacts_are_truth": True,
            "expected_release_sha256": self.config.expected_release_sha256,
            "expected_manifest_file_sha256": (
                self.config.expected_manifest_file_sha256
            ),
        }
        state = {**unsigned, "state_sha256": canonical_sha256(unsigned)}
        _atomic_write_json(
            self.config.v2_root / "status/orchestrator_state.json", state
        )
        _append_jsonl(
            self.config.v2_root / "audits/orchestrator_events.jsonl",
            {
                "schema_version": "stage7_core_ablation_orchestrator_event_v2",
                "orchestrator_pid": self.pid,
                "tick_index": self.tick_index,
                "wall_time": now,
                "phase": phase,
                "status": status,
                "event": event,
                "details": copy.deepcopy(dict(details or {})),
            },
        )
        return state

    def _step_phase(self, phase: str, snapshot: Mapping[str, Any]) -> JSON:
        if phase in (PRE_GATE_SELECTION, PRE_GATE_SOURCE):
            field = (
                "request_frozen"
                if phase == PRE_GATE_SELECTION
                else "source_ready"
            )
            action = (
                FREEZE_SOURCE_PLAN
                if phase == PRE_GATE_SELECTION
                else RESOLVE_PRE_GATE_SOURCE
            )
            for variant in CORE_VARIANTS:
                if _round_record(snapshot, variant, PILOT_SEED, 0).get(field) is True:
                    continue
                step = (
                    self.dependencies.round_step
                    if phase == PRE_GATE_SELECTION
                    else self.dependencies.source_step
                )
                result = step(
                    self.config, action, variant, PILOT_SEED, 0
                )
                return {
                    "event": action,
                    "variant": variant,
                    "seed": PILOT_SEED,
                    "round_index": 0,
                    "result": copy.deepcopy(dict(result)),
                }
            raise OrchestrationContractError(
                "pre-gate phase has no admissible round operation"
            )
        if phase == NO_GPU_GATE:
            result = self.dependencies.run_no_gpu_gate(self.config)
            _validate_no_gpu_result(result)
            return {"event": "no_gpu_gate", "result": copy.deepcopy(dict(result))}
        if phase in (PILOT_20260718, SEED_20260719, SEED_20260720):
            seed = {
                PILOT_20260718: 20260718,
                SEED_20260719: 20260719,
                SEED_20260720: 20260720,
            }[phase]
            operation = _next_round_operation(snapshot, seed)
            if operation is None:
                raise OrchestrationContractError(
                    "seed is incomplete but has no admissible round operation"
                )
            action, variant, selected_seed, round_index = operation
            step = (
                self.dependencies.source_step
                if action == RESOLVE_SOURCE
                else self.dependencies.round_step
            )
            result = step(self.config, action, variant, selected_seed, round_index)
            return {
                "event": action,
                "variant": variant,
                "seed": selected_seed,
                "round_index": round_index,
                "result": copy.deepcopy(dict(result)),
            }
        if phase == PILOT_AUDIT:
            if not _seed_complete(snapshot, PILOT_SEED):
                raise OrchestrationContractError(
                    "later-seed gate requires four pilot trajectories and 64 events"
                )
            if snapshot.get("pilot_audit_passed") is not True:
                result = self.dependencies.audit_pilot(self.config)
                return {"event": "pilot_audit", "result": copy.deepcopy(dict(result))}
            result = self.dependencies.gate_later_seeds(self.config)
            return {
                "event": "later_seed_gate",
                "result": copy.deepcopy(dict(result)),
            }
        if phase == FINALIZE:
            result = self.dependencies.finalize(self.config)
            return {"event": "formal_finalize", "result": copy.deepcopy(dict(result))}
        if phase == COMPLETE:
            return {"event": "complete"}
        raise OrchestrationContractError(f"unsupported orchestration phase: {phase}")

    def tick(self) -> JSON:
        if not (self.config.v2_root / "prepare_state.json").is_file():
            self.dependencies.prepare_initialize(self.config, self.pid)
            if not (self.config.v2_root / "prepare_state.json").is_file():
                raise OrchestrationContractError(
                    "prepare returned without canonical prepare_state"
                )
            self._validate_tick_identity_and_pins()
            self.tick_index += 1
            snapshot = copy.deepcopy(
                dict(self.dependencies.inspect_canonical(self.config))
            )
            phase = _derive_phase(
                snapshot,
                pre_gate_source_only=self.config.pre_gate_source_only,
            )
            self._persist(
                phase=phase,
                status="running",
                event="bootstrap_prepared",
            )
            return {"phase": phase, "status": "running", "event": "bootstrap_prepared"}

        self._validate_tick_identity_and_pins()
        self.tick_index += 1
        snapshot = copy.deepcopy(dict(self.dependencies.inspect_canonical(self.config)))
        phase = _derive_phase(
            snapshot,
            pre_gate_source_only=self.config.pre_gate_source_only,
        )
        try:
            details = self._step_phase(phase, snapshot)
        except RetryableOrchestrationError as error:
            self._persist(
                phase=phase,
                status="waiting_retryable",
                event="retryable_wait",
                details={"category": error.category, "reason": error.reason},
            )
            return {
                "phase": phase,
                "status": "waiting_retryable",
                "reason": error.reason,
                "category": error.category,
            }
        updated = copy.deepcopy(dict(self.dependencies.inspect_canonical(self.config)))
        next_phase = _derive_phase(
            updated,
            pre_gate_source_only=self.config.pre_gate_source_only,
        )
        status = "complete" if next_phase == COMPLETE else "running"
        self._persist(
            phase=next_phase,
            status=status,
            event=str(details["event"]),
            details=details,
        )
        return {
            "phase": next_phase,
            "status": status,
            "event": details["event"],
        }

    def run(self, *, max_ticks: Optional[int] = None) -> JSON:
        ticks = 0
        while True:
            result = self.tick()
            ticks += 1
            if result["phase"] == COMPLETE:
                return result
            if max_ticks is not None and ticks >= max_ticks:
                return result
            self.dependencies.sleep(self.config.poll_seconds)


def _default_prepare(config: OrchestratorConfig, pid: int) -> JSON:
    from scripts import stage7_core_no_gpu_dry_run_v2 as no_gpu

    result = no_gpu.prepare_and_initialize_no_gpu_root(
        v1_root=config.v1_root,
        v2_root=config.v2_root,
        prepare_inputs=config.prepare_inputs,
        embedded_source_manifest=config.embedded_source_manifest,
        orchestrator_pid=pid,
        repo_root=config.repo_root,
        expected_release_sha256=config.expected_release_sha256,
        expected_manifest_file_sha256=config.expected_manifest_file_sha256,
    )
    if config.pre_gate_source_only:
        _ensure_pre_gate_source_authorization(config)
    return result


def _ensure_pre_gate_source_authorization(config: OrchestratorConfig) -> JSON:
    unsigned = {
        "schema_version": "stage7_pre_gate_source_authorization_v2",
        "authorized_action": "source_only_materialization_before_no_gpu_gate",
        "v2_root": str(config.v2_root.resolve()),
        "gpu7_excluded": True,
        "performance_measurement_allowed": False,
        "ap_allowed": False,
        "cache_reveal_allowed": False,
        "feedback_allowed": False,
        "selected_event_budget_delta": 0,
    }
    authorization = {
        **unsigned,
        "authorization_sha256": canonical_sha256(unsigned),
    }
    path = config.v2_root / "contracts/pre_gate_source_authorization.json"
    if path.is_file():
        if _read_mapping(path, label="pre-gate source authorization") != authorization:
            raise OrchestrationContractError(
                "pre-gate source authorization conflict"
            )
        return authorization
    if path.exists():
        raise OrchestrationContractError(
            "pre-gate source authorization path is unsafe"
        )
    _atomic_write_json(path, authorization)
    return authorization


def _default_no_gpu(config: OrchestratorConfig) -> JSON:
    from scripts import stage7_core_no_gpu_dry_run_v2 as no_gpu

    result = no_gpu.run_no_gpu_dry_run(
        config.v2_root,
        repo_root=config.repo_root,
        frozen_onnx=config.frozen_onnx,
        expected_frozen_onnx_sha256=config.frozen_onnx_sha256,
        expected_release_sha256=config.expected_release_sha256,
        expected_manifest_file_sha256=config.expected_manifest_file_sha256,
    )
    _validate_no_gpu_result(result)
    return result


def _default_validate_runtime(
    config: OrchestratorConfig, pid: int, prepare_state: Mapping[str, Any]
) -> JSON:
    from framework.stage7 import actual_mode_v2
    from framework.stage7 import deployment_bundle_v2
    from framework.stage7 import executor_admission_v2

    identity = executor_admission_v2.capture_process_identity(
        executor_admission_v2.probe_linux_process_identity, pid
    )
    executors = executor_admission_v2.verify_frozen_actual_v3_executors(
        config.repo_root
    )
    deployment = deployment_bundle_v2.validate_deployment_bundle(
        config.v2_root,
        frozen_repo_root=config.repo_root,
        expected_release_sha256=config.expected_release_sha256,
        expected_manifest_file_sha256=config.expected_manifest_file_sha256,
        expected_owner=str(prepare_state.get("owner") or ""),
        expected_owner_uid=prepare_state.get("owner_uid"),
    )
    receipt = config.v2_root / "audits/no_gpu_dry_run/dry_run_receipt.json"
    if receipt.is_file():
        actual_mode_v2.validate_no_gpu_receipt(
            _read_mapping(receipt, label="no-GPU receipt"),
            root=config.v2_root,
            frozen_repo_root=config.repo_root,
            deployment_bundle_sha256=deployment["deployment_bundle_sha256"],
        )
    return {
        "process_identity": identity,
        "deployment_manifest_sha256": deployment["deployment_manifest_sha256"],
        "deployment_manifest_file_sha256": deployment[
            "deployment_manifest_file_sha256"
        ],
        "deployment_bundle_sha256": deployment["deployment_bundle_sha256"],
        "deployment_release_sha256": deployment["deployment_release_sha256"],
        "executor_verification": executors,
    }


def _json_if_present(path: Path) -> JSON:
    return _read_mapping(path, label=str(path)) if path.is_file() else {}


def _default_inspect(config: OrchestratorConfig) -> JSON:
    root = config.v2_root
    try:
        worker = importlib.import_module("scripts.stage7_core_round_worker_v2")
    except ImportError:
        worker = None
    prepared = (root / "prepare_state.json").is_file()
    initialized = (root / "initialize_state.json").is_file()
    dry = _json_if_present(root / "audits/no_gpu_dry_run/dry_run_receipt.json")
    no_gpu_passed = no_gpu_integration_result_closed(
        dry, variants=CORE_VARIANTS, events_per_round=EVENTS_PER_ROUND
    )
    trajectories: dict[str, Any] = {}
    for variant in CORE_VARIANTS:
        for seed in SEEDS:
            rounds: dict[str, Any] = {}
            for index in ROUNDS:
                directory = (
                    root / "variants" / variant / f"seed_{seed}" / f"round_{index:02d}"
                )
                barrier = _json_if_present(directory / "atomic_feedback_barrier.json")
                request_path = directory / "logical_request.json"
                observed: Mapping[str, Any] = {}
                if worker is not None and request_path.is_file():
                    request = _read_mapping(request_path, label="logical request")
                    request_sha = request.get("measurement_request_sha256")
                    if isinstance(request_sha, str) and len(request_sha) == 64:
                        observed = worker.observe_round(
                            root, variant, seed, index, request_sha
                        )
                rounds[str(index)] = {
                    "request_frozen": (
                        (directory / "logical_request.json").is_file()
                        and (directory / "source_resolution_plan.json").is_file()
                    ),
                    "source_ready": (
                        directory / "source_resolution_result.json"
                    ).is_file(),
                    "exact_bound": all(
                        (directory / name).is_file()
                        for name in (
                            "exact_selection_binding.json",
                            "cache_snapshot_before_reveal.json",
                            "cache_reveal.json",
                            "miss_only_physical_request.json",
                            "executor_admission.json",
                        )
                    ),
                    "terminal": observed.get("terminal_authenticated") is True
                    or observed.get("barrier_authenticated") is True,
                    "barrier": observed.get("barrier_authenticated") is True,
                    "selected_events": (
                        int(barrier.get("budget_consumed", 0)) if barrier else 0
                    ),
                }
            trajectories[f"{variant}:{seed}"] = {"rounds": rounds}
    pilot = _json_if_present(root / "audits/pilot_audit_v2.json")
    gate = _json_if_present(root / "status/later_seed_gate_v2.json")
    final = _json_if_present(root / "paper_outputs/status/finalization_status.json")
    return {
        "prepared": prepared,
        "initialized": initialized,
        "no_gpu_gate_passed": no_gpu_passed,
        "trajectories": trajectories,
        "pilot_audit_passed": pilot.get("status") == "passed",
        "later_seed_gate_passed": gate.get("status") == "passed",
        "finalization_complete": (
            final.get("paper_ready") is True
            and final.get("core_ablation_ready") is True
        ),
    }


def _default_round_step(
    config: OrchestratorConfig,
    action: str,
    variant: str,
    seed: int,
    round_index: int,
) -> JSON:
    if action == FREEZE_SOURCE_PLAN:
        from scripts import stage7_core_online_ablation_v2 as online

        try:
            return online.prepare_round(
                config.v2_root,
                variant=variant,
                seed=seed,
                round_index=round_index,
                repo_root=config.repo_root,
                expected_release_sha256=config.expected_release_sha256,
                expected_manifest_file_sha256=config.expected_manifest_file_sha256,
            )
        except online.SourceResolutionRequiredBeforeExactCacheReveal as error:
            return {"status": "source_plan_frozen", "audit": copy.deepcopy(error.audit)}
    directory = (
        config.v2_root
        / "variants"
        / variant
        / f"seed_{seed}"
        / f"round_{round_index:02d}"
    )
    if action == BIND_EXACT:
        from scripts import stage7_core_online_ablation_v2 as online

        source_result = directory / "source_resolution_result.json"
        if not source_result.is_file():
            raise RetryableOrchestrationError(
                "formal source result is not yet available",
                category="source_evidence_wait",
            )
        return online.bind_reveal_after_source_ready(
            config.v2_root,
            variant=variant,
            seed=seed,
            round_index=round_index,
            source_result_path=source_result,
            synthetic_no_gpu_dryrun=False,
            repo_root=config.repo_root,
            expected_release_sha256=config.expected_release_sha256,
            expected_manifest_file_sha256=config.expected_manifest_file_sha256,
        )
    from scripts import stage7_orchestrator_scheduler_adapter_v2 as adapter

    try:
        if action == FINALIZE_TERMINAL_BARRIER:
            return adapter.recover_terminal_barrier(
                config, variant, seed, round_index, directory
            )
        return adapter.submit_or_observe(
            v2_root=config.v2_root,
            variant=variant,
            seed=seed,
            round_index=round_index,
            expected_release_sha256=config.expected_release_sha256,
            expected_manifest_file_sha256=config.expected_manifest_file_sha256,
            directory=directory,
            logical=_read_mapping(
                directory / "logical_request.json", label="logical request"
            ),
            physical_plan=_read_mapping(
                directory / "miss_only_physical_request.json",
                label="physical request",
            ),
            source_result=_read_mapping(
                directory / "source_resolution_result.json", label="source result"
            ),
        )
    except adapter.SchedulerAdapterRetry as error:
        raise RetryableOrchestrationError(
            error.reason, category=error.category
        ) from error
    except adapter.SchedulerAdapterError as error:
        raise OrchestrationContractError(str(error)) from error


def _default_source_step(
    config: OrchestratorConfig,
    action: str,
    variant: str,
    seed: int,
    round_index: int,
) -> JSON:
    if action not in (RESOLVE_SOURCE, RESOLVE_PRE_GATE_SOURCE):
        raise OrchestrationContractError("source_step received a non-source action")
    directory = (
        config.v2_root
        / "variants"
        / variant
        / f"seed_{seed}"
        / f"round_{round_index:02d}"
    )
    from scripts import stage7_source_lease_controller_v2 as source_controller

    prepare = _read_mapping(
        config.v2_root / "prepare_state.json", label="prepare state"
    )
    result = source_controller.run_source_lease_controller(
        v2_root=config.v2_root,
        repo_root=config.repo_root,
        variant=variant,
        seed=seed,
        round_index=round_index,
        orchestrator_pid=int(prepare.get("orchestrator_pid", -1)),
        pre_gate_source_only=(action == RESOLVE_PRE_GATE_SOURCE),
    )
    if result.get("status") == "waiting_for_source_gpu_capacity":
        raise RetryableOrchestrationError(
            str(result.get("reason") or "source H800 capacity unavailable"),
            category="source_resource_wait",
        )
    if result.get("status") == "retry_required":
        raise RetryableOrchestrationError(
            str(result.get("retry_reason_code") or "source resolution retry"),
            category="source_infrastructure",
        )
    expected_status = (
        "source_ready"
        if action == RESOLVE_PRE_GATE_SOURCE
        else "cache_revealed"
    )
    if result.get("status") != expected_status:
        raise OrchestrationContractError("source lease controller status drift")
    return copy.deepcopy(dict(result))


def _strict_gate_payload(config: OrchestratorConfig, *, kind: str) -> JSON:
    snapshot = _default_inspect(config)
    if not _seed_complete(snapshot, PILOT_SEED):
        raise OrchestrationContractError("pilot gate requires 64 barrier-bound events")
    unsigned = {
        "schema_version": f"stage7_core_ablation_{kind}_v2",
        "status": "passed",
        "pilot_seed": PILOT_SEED,
        "pilot_trajectory_count": len(CORE_VARIANTS),
        "pilot_selected_event_count": 64,
    }
    return {**unsigned, "audit_sha256": canonical_sha256(unsigned)}


def _default_audit_pilot(config: OrchestratorConfig) -> JSON:
    payload = _strict_gate_payload(config, kind="pilot_audit")
    _atomic_write_json(config.v2_root / "audits/pilot_audit_v2.json", payload)
    return payload


def _default_gate_later(config: OrchestratorConfig) -> JSON:
    payload = _strict_gate_payload(config, kind="later_seed_gate")
    _atomic_write_json(config.v2_root / "status/later_seed_gate_v2.json", payload)
    return payload


def _default_finalize(config: OrchestratorConfig) -> JSON:
    from scripts import stage7_finalize_core_ablation_v2 as finalizer

    return finalizer.finalize_v2(
        config.v2_root,
        config.v2_root / "paper_outputs",
        repo_root=config.repo_root,
    )


def default_dependencies() -> OrchestratorDependencies:
    return OrchestratorDependencies(
        prepare_initialize=_default_prepare,
        run_no_gpu_gate=_default_no_gpu,
        inspect_canonical=_default_inspect,
        validate_runtime=_default_validate_runtime,
        source_step=_default_source_step,
        round_step=_default_round_step,
        audit_pilot=_default_audit_pilot,
        gate_later_seeds=_default_gate_later,
        finalize=_default_finalize,
    )


def _read_cli_mapping(path: Path) -> JSON:
    return _read_mapping(path.resolve(strict=True), label=str(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prepare-inputs-json", type=Path, required=True)
    parser.add_argument("--embedded-source-manifest-json", type=Path, required=True)
    parser.add_argument("--frozen-onnx", type=Path, required=True)
    parser.add_argument("--frozen-onnx-sha256", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--pre-gate-source-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = OrchestratorConfig(
        v1_root=args.v1_root.resolve(),
        v2_root=args.v2_root.resolve(),
        repo_root=args.repo_root.resolve(),
        prepare_inputs=_read_cli_mapping(args.prepare_inputs_json),
        embedded_source_manifest=_read_cli_mapping(args.embedded_source_manifest_json),
        frozen_onnx=args.frozen_onnx.resolve(),
        frozen_onnx_sha256=args.frozen_onnx_sha256,
        expected_release_sha256=args.expected_release_sha256,
        expected_manifest_file_sha256=args.expected_manifest_file_sha256,
        poll_seconds=args.poll_seconds,
        pre_gate_source_only=args.pre_gate_source_only,
    )
    result = PersistentCoreAblationOrchestrator(config).run(
        max_ticks=1 if args.once else None
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
