from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import stage7_core_ablation_orchestrator_v2 as orchestrator
from scripts import stage7_orchestrator_scheduler_adapter_v2 as scheduler_adapter


VARIANTS = (
    "full",
    "without_surrogate",
    "without_measured_feedback",
    "backend_blind",
)
PILOT_SEED = 20260718


def _identity(pid: int, *, start: int = 123) -> dict[str, Any]:
    unsigned = {
        "pid": pid,
        "uid": 1000,
        "username": "runner",
        "start_time_ticks": start,
        "executable": "/usr/bin/python3",
        "cmdline": ["python3", "scripts/stage7_core_ablation_orchestrator_v2.py"],
    }
    return {
        **unsigned,
        "process_identity_sha256": orchestrator.canonical_sha256(unsigned),
    }


class FakeRuntime:
    def __init__(self, config: orchestrator.OrchestratorConfig, pid: int) -> None:
        self.config = config
        self.pid = pid
        self.identity = _identity(pid)
        self.canonical: dict[str, Any] = {
            "prepared": False,
            "initialized": False,
            "no_gpu_gate_passed": False,
            "trajectories": {},
            "pilot_audit_passed": False,
            "later_seed_gate_passed": False,
            "finalization_complete": False,
        }
        self.calls: list[tuple[Any, ...]] = []
        self.retry_action: str | None = None
        self.prepare_error: Exception | None = None

    def dependencies(self) -> orchestrator.OrchestratorDependencies:
        return orchestrator.OrchestratorDependencies(
            prepare_initialize=self.prepare_initialize,
            run_no_gpu_gate=self.run_no_gpu_gate,
            inspect_canonical=self.inspect_canonical,
            validate_runtime=self.validate_runtime,
            source_step=self.round_step,
            round_step=self.round_step,
            audit_pilot=self.audit_pilot,
            gate_later_seeds=self.gate_later_seeds,
            finalize=self.finalize,
            sleep=lambda _seconds: None,
            clock=lambda: 10.0 + len(self.calls),
        )

    def prepare_initialize(
        self, config: orchestrator.OrchestratorConfig, pid: int
    ) -> dict[str, Any]:
        self.calls.append(("prepare_initialize", pid))
        if self.prepare_error is not None:
            raise self.prepare_error
        config.v2_root.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "stage7_core_ablation_v2_prepare_state",
            "status": "prepared",
            "orchestrator_pid": pid,
            "orchestrator_process_identity": copy.deepcopy(self.identity),
            "deployment_manifest_sha256": "d" * 64,
            "deployment_manifest_file_sha256": "c" * 64,
            "deployment_bundle_sha256": "b" * 64,
            "deployment_release_sha256": "a" * 64,
            "executor_verification": [{"actual_sha256": "e" * 64}],
        }
        (config.v2_root / "prepare_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        self.canonical["prepared"] = True
        self.canonical["initialized"] = True
        return {"status": "initialized"}

    def run_no_gpu_gate(
        self, _config: orchestrator.OrchestratorConfig
    ) -> dict[str, Any]:
        self.calls.append(("run_no_gpu_gate",))
        self.canonical["no_gpu_gate_passed"] = True
        return {
            "formal_v2_gpu_jobs_launched": 0,
            "canonical_round0_requests_written": 4,
            "round0_miss_count": 16,
            "integration_closed": True,
            "blocking_reason": None,
            "variant_receipts": [
                {
                    "variant": variant,
                    "round0_miss_count": 4,
                    "miss_admission_passed": True,
                }
                for variant in orchestrator.CORE_VARIANTS
            ],
        }

    def inspect_canonical(
        self, _config: orchestrator.OrchestratorConfig
    ) -> dict[str, Any]:
        return copy.deepcopy(self.canonical)

    def validate_runtime(
        self,
        _config: orchestrator.OrchestratorConfig,
        pid: int,
        _prepare_state: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("validate_runtime", pid))
        return {
            "process_identity": copy.deepcopy(self.identity),
            "deployment_manifest_sha256": "d" * 64,
            "deployment_manifest_file_sha256": "c" * 64,
            "deployment_bundle_sha256": "b" * 64,
            "deployment_release_sha256": "a" * 64,
            "executor_verification": [{"actual_sha256": "e" * 64}],
        }

    def _round(self, variant: str, seed: int, round_index: int) -> dict[str, Any]:
        trajectory = self.canonical["trajectories"].setdefault(
            f"{variant}:{seed}", {"rounds": {}}
        )
        return trajectory["rounds"].setdefault(
            str(round_index),
            {
                "request_frozen": False,
                "source_ready": False,
                "exact_bound": False,
                "terminal": False,
                "barrier": False,
                "selected_events": 0,
            },
        )

    def round_step(
        self,
        _config: orchestrator.OrchestratorConfig,
        action: str,
        variant: str,
        seed: int,
        round_index: int,
    ) -> dict[str, Any]:
        self.calls.append(("round_step", action, variant, seed, round_index))
        if self.retry_action == action:
            self.retry_action = None
            raise orchestrator.RetryableOrchestrationError(
                f"retry-{action}", category="infrastructure"
            )
        row = self._round(variant, seed, round_index)
        field = {
            orchestrator.FREEZE_SOURCE_PLAN: "request_frozen",
            orchestrator.RESOLVE_SOURCE: "source_ready",
            orchestrator.RESOLVE_PRE_GATE_SOURCE: "source_ready",
            orchestrator.BIND_EXACT: "exact_bound",
            orchestrator.EXECUTE_MEASUREMENT: "terminal",
            orchestrator.FINALIZE_TERMINAL_BARRIER: "barrier",
        }[action]
        row[field] = True
        if action == orchestrator.EXECUTE_MEASUREMENT:
            row["terminal"] = True
            row["barrier"] = True
            row["selected_events"] = 4
        elif action == orchestrator.FINALIZE_TERMINAL_BARRIER:
            row["selected_events"] = 4
        return {"status": "completed", "action": action}

    def audit_pilot(self, _config: orchestrator.OrchestratorConfig) -> dict[str, Any]:
        self.calls.append(("audit_pilot",))
        self.canonical["pilot_audit_passed"] = True
        return {"status": "passed"}

    def gate_later_seeds(
        self, _config: orchestrator.OrchestratorConfig
    ) -> dict[str, Any]:
        self.calls.append(("gate_later_seeds",))
        self.canonical["later_seed_gate_passed"] = True
        return {"status": "passed"}

    def finalize(self, _config: orchestrator.OrchestratorConfig) -> dict[str, Any]:
        self.calls.append(("finalize",))
        self.canonical["finalization_complete"] = True
        return {"status": "complete"}


@pytest.fixture
def config(tmp_path: Path) -> orchestrator.OrchestratorConfig:
    frozen = tmp_path / "frozen.onnx"
    frozen.write_bytes(b"onnx")
    return orchestrator.OrchestratorConfig(
        v1_root=tmp_path / "v1",
        v2_root=tmp_path / "v2",
        repo_root=tmp_path / "repo",
        prepare_inputs={},
        embedded_source_manifest={},
        frozen_onnx=frozen,
        frozen_onnx_sha256="f" * 64,
        expected_release_sha256="a" * 64,
        expected_manifest_file_sha256="c" * 64,
        poll_seconds=0.01,
    )


def _machine(
    config: orchestrator.OrchestratorConfig, runtime: FakeRuntime
) -> orchestrator.PersistentCoreAblationOrchestrator:
    return orchestrator.PersistentCoreAblationOrchestrator(
        config,
        dependencies=runtime.dependencies(),
        pid=runtime.pid,
    )


def _run_until(
    machine: orchestrator.PersistentCoreAblationOrchestrator,
    predicate,
    *,
    limit: int = 500,
) -> None:
    for _ in range(limit):
        machine.tick()
        if predicate():
            return
    raise AssertionError("state machine did not reach expected state")


def test_prepare_failure_writes_no_derived_state_or_events(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 700)
    runtime.prepare_error = RuntimeError("prepare failed")
    machine = _machine(config, runtime)

    with pytest.raises(RuntimeError, match="prepare failed"):
        machine.tick()

    assert not (config.v2_root / "status/orchestrator_state.json").exists()
    assert not (config.v2_root / "audits/orchestrator_events.jsonl").exists()


def test_bootstrap_binds_prepare_to_persistent_orchestrator_pid(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 701)
    machine = _machine(config, runtime)

    result = machine.tick()
    prepare = json.loads((config.v2_root / "prepare_state.json").read_text())
    state = json.loads((config.v2_root / "status/orchestrator_state.json").read_text())

    assert prepare["orchestrator_pid"] == 701
    assert state["expected_release_sha256"] == "a" * 64
    assert state["expected_manifest_file_sha256"] == "c" * 64
    assert result["phase"] == orchestrator.NO_GPU_GATE
    assert (config.v2_root / "status/orchestrator_state.json").is_file()
    assert ("prepare_initialize", 701) in runtime.calls


def test_approved_pre_gate_source_sequence_precedes_no_gpu_gate_and_pilot(
    config: orchestrator.OrchestratorConfig,
) -> None:
    enabled = replace(config, pre_gate_source_only=True)
    runtime = FakeRuntime(enabled, 709)
    machine = _machine(enabled, runtime)

    assert machine.tick()["phase"] == orchestrator.PRE_GATE_SELECTION
    for _ in orchestrator.CORE_VARIANTS:
        machine.tick()
    assert not any(call[0] == "run_no_gpu_gate" for call in runtime.calls)
    assert all(
        runtime._round(variant, PILOT_SEED, 0)["request_frozen"]
        for variant in orchestrator.CORE_VARIANTS
    )

    for _ in orchestrator.CORE_VARIANTS:
        machine.tick()
    assert not any(call[0] == "run_no_gpu_gate" for call in runtime.calls)
    assert all(
        runtime._round(variant, PILOT_SEED, 0)["source_ready"]
        for variant in orchestrator.CORE_VARIANTS
    )
    source_actions = [
        call[1] for call in runtime.calls if call[0] == "round_step"
    ]
    assert source_actions == (
        [orchestrator.FREEZE_SOURCE_PLAN] * 4
        + [orchestrator.RESOLVE_PRE_GATE_SOURCE] * 4
    )

    gated = machine.tick()
    assert gated["phase"] == orchestrator.PILOT_20260718
    assert runtime.calls.count(("run_no_gpu_gate",)) == 1


def test_pid_reuse_or_cmdline_drift_is_fatal_on_every_tick(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 702)
    machine = _machine(config, runtime)
    machine.tick()
    runtime.identity = _identity(702, start=999)

    with pytest.raises(orchestrator.OrchestrationContractError, match="identity drift"):
        machine.tick()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("deployment_release_sha256", "x" * 64),
        ("deployment_manifest_file_sha256", "y" * 64),
    ),
)
def test_external_deployment_pin_drift_stops_before_no_gpu_or_source(
    field: str,
    value: str,
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 799)
    machine = _machine(config, runtime)
    machine.tick()
    prepare_path = config.v2_root / "prepare_state.json"
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    prepare[field] = value
    prepare_path.write_text(json.dumps(prepare), encoding="utf-8")
    calls_before = list(runtime.calls)
    tick_before = machine.tick_index
    state_path = config.v2_root / "status" / "orchestrator_state.json"
    events_path = config.v2_root / "audits" / "orchestrator_events.jsonl"
    state_before = state_path.read_bytes()
    events_before = events_path.read_bytes()

    with pytest.raises(orchestrator.OrchestrationContractError, match="external pin"):
        machine.tick()

    assert runtime.calls == calls_before
    assert machine.tick_index == tick_before
    assert state_path.read_bytes() == state_before
    assert events_path.read_bytes() == events_before
    assert not any(call[0] == "run_no_gpu_gate" for call in runtime.calls)
    assert not any(call[0] == "round_step" for call in runtime.calls)


def test_existing_terminal_without_barrier_selects_barrier_only_recovery(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 703)
    machine = _machine(config, runtime)
    machine.tick()
    machine.tick()
    row = runtime._round(VARIANTS[0], PILOT_SEED, 0)
    row.update(
        request_frozen=True,
        source_ready=True,
        exact_bound=True,
        terminal=True,
        barrier=False,
    )

    machine.tick()

    actions = [call[1] for call in runtime.calls if call[0] == "round_step"]
    assert actions == [orchestrator.FINALIZE_TERMINAL_BARRIER]
    assert orchestrator.EXECUTE_MEASUREMENT not in actions


def test_source_and_infrastructure_retry_consume_zero_budget_and_stay_live(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 704)
    machine = _machine(config, runtime)
    machine.tick()
    machine.tick()
    machine.tick()  # freeze
    runtime.retry_action = orchestrator.RESOLVE_SOURCE

    waited = machine.tick()
    row = runtime._round(VARIANTS[0], PILOT_SEED, 0)

    assert waited["status"] == "waiting_retryable"
    assert waited["phase"] == orchestrator.PILOT_20260718
    assert row["selected_events"] == 0
    machine.tick()
    assert row["source_ready"] is True


def test_no_cross_round_prefetch_and_exact_once_next_round(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 705)
    machine = _machine(config, runtime)
    machine.tick()
    machine.tick()

    _run_until(
        machine,
        lambda: runtime._round(VARIANTS[0], PILOT_SEED, 0)["barrier"],
    )
    actions_before = list(runtime.calls)
    machine.tick()

    round1_freezes = [
        call
        for call in runtime.calls
        if call[:5]
        == (
            "round_step",
            orchestrator.FREEZE_SOURCE_PLAN,
            VARIANTS[0],
            PILOT_SEED,
            1,
        )
    ]
    assert len(round1_freezes) == 1
    assert not any(
        call[0] == "round_step" and call[3] == PILOT_SEED and call[4] > 1
        for call in actions_before
    )


def test_idempotent_resume_uses_canonical_artifacts_not_derived_state(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 706)
    first = _machine(config, runtime)
    first.tick()
    first.tick()
    first.tick()
    assert runtime._round(VARIANTS[0], PILOT_SEED, 0)["request_frozen"]
    runtime.calls.clear()

    resumed = _machine(config, runtime)
    resumed.tick()

    assert ("prepare_initialize", 706) not in runtime.calls
    actions = [call[1] for call in runtime.calls if call[0] == "round_step"]
    assert actions == [orchestrator.RESOLVE_SOURCE]


def test_fake_pilot_e2e_closes_four_variants_four_rounds_before_later_seed(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 707)
    machine = _machine(config, runtime)

    _run_until(machine, lambda: runtime.canonical["pilot_audit_passed"])

    pilot_rows = [
        row
        for key, trajectory in runtime.canonical["trajectories"].items()
        if key.endswith(f":{PILOT_SEED}")
        for row in trajectory["rounds"].values()
    ]
    assert len(pilot_rows) == 16
    assert sum(row["selected_events"] for row in pilot_rows) == 64
    assert all(row["barrier"] for row in pilot_rows)
    assert not any(
        key.endswith(":20260719") for key in runtime.canonical["trajectories"]
    )
    assert runtime.calls.count(("audit_pilot",)) == 1


def test_later_seed_gate_requires_all_pilot_barriers_and_64_events(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 708)
    machine = _machine(config, runtime)
    machine.tick()
    machine.tick()
    runtime.canonical["pilot_audit_passed"] = True

    machine.tick()

    assert ("gate_later_seeds",) not in runtime.calls
    assert not any(
        call[0] == "round_step" and call[3] != PILOT_SEED for call in runtime.calls
    )


def test_run_stops_at_max_ticks_and_persistent_complete(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 709)
    machine = _machine(config, runtime)

    limited = machine.run(max_ticks=2)
    assert limited["phase"] == orchestrator.PILOT_20260718

    for seed in (20260718, 20260719, 20260720):
        for variant in VARIANTS:
            for index in range(4):
                runtime._round(variant, seed, index).update(
                    request_frozen=True,
                    source_ready=True,
                    exact_bound=True,
                    terminal=True,
                    barrier=True,
                    selected_events=4,
                )
    runtime.canonical["pilot_audit_passed"] = True
    runtime.canonical["later_seed_gate_passed"] = True
    completed = machine.run(max_ticks=3)
    assert completed == {
        "phase": orchestrator.COMPLETE,
        "status": "complete",
        "event": "formal_finalize",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("deployment_manifest_sha256", "x" * 64),
        ("deployment_manifest_file_sha256", "x" * 64),
        ("deployment_bundle_sha256", "x" * 64),
        ("deployment_release_sha256", "x" * 64),
        ("executor_verification", []),
    ],
)
def test_runtime_pin_drift_is_fatal(
    config: orchestrator.OrchestratorConfig,
    field: str,
    value: Any,
) -> None:
    runtime = FakeRuntime(config, 710)
    machine = _machine(config, runtime)
    machine.tick()
    original = runtime.validate_runtime

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        result[field] = value
        return result

    object.__setattr__(
        machine.dependencies,
        "validate_runtime",
        drift,
    )
    with pytest.raises(orchestrator.OrchestrationContractError, match="pin drift"):
        machine.tick()


def test_no_gpu_gate_rejects_gpu_launch(
    config: orchestrator.OrchestratorConfig,
) -> None:
    runtime = FakeRuntime(config, 711)
    machine = _machine(config, runtime)
    machine.tick()

    def launched(_config):
        return {"formal_v2_gpu_jobs_launched": 1}

    object.__setattr__(machine.dependencies, "run_no_gpu_gate", launched)
    with pytest.raises(orchestrator.OrchestrationContractError, match="launched"):
        machine.tick()


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"poll_seconds": 0}, "poll_seconds"),
        ({"frozen_onnx_sha256": "short"}, "SHA256"),
        ({"expected_release_sha256": None}, "release SHA256"),
        ({"expected_release_sha256": "R" * 64}, "release SHA256"),
        ({"expected_manifest_file_sha256": "short"}, "manifest-file SHA256"),
        ({"expected_manifest_file_sha256": "g" * 64}, "manifest-file SHA256"),
        ({"v2_root": Path("relative")}, "v2_root"),
    ],
)
def test_config_rejects_invalid_boundaries(
    tmp_path: Path, changes: dict[str, Any], error: str
) -> None:
    values = {
        "v1_root": tmp_path / "v1",
        "v2_root": tmp_path / "v2",
        "repo_root": tmp_path / "repo",
        "prepare_inputs": {},
        "embedded_source_manifest": {},
        "frozen_onnx": tmp_path / "model.onnx",
        "frozen_onnx_sha256": "f" * 64,
        "expected_release_sha256": "a" * 64,
        "expected_manifest_file_sha256": "c" * 64,
        "poll_seconds": 1,
    }
    with pytest.raises(ValueError, match=error):
        orchestrator.OrchestratorConfig(**{**values, **changes})


def test_default_inspector_reads_only_canonical_round_artifacts(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = config.v2_root
    (root / "prepare_state.json").parent.mkdir(parents=True)
    (root / "prepare_state.json").write_text("{}", encoding="utf-8")
    (root / "initialize_state.json").write_text("{}", encoding="utf-8")
    dry = root / "audits/no_gpu_dry_run/dry_run_receipt.json"
    dry.parent.mkdir(parents=True)
    dry.write_text(
        json.dumps(
            {
                "formal_v2_gpu_jobs_launched": 0,
                "canonical_round0_requests_written": 4,
                "round0_miss_count": 16,
                "integration_closed": True,
                "blocking_reason": None,
                "variant_receipts": [
                    {
                        "variant": variant,
                        "round0_miss_count": 4,
                        "miss_admission_passed": True,
                    }
                    for variant in orchestrator.CORE_VARIANTS
                ],
            }
        ),
        encoding="utf-8",
    )
    directory = root / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    (directory / "logical_request.json").write_text(
        json.dumps({"measurement_request_sha256": "a" * 64}), encoding="utf-8"
    )
    for name in (
        "source_resolution_plan.json",
        "source_resolution_result.json",
        "exact_selection_binding.json",
        "cache_snapshot_before_reveal.json",
        "cache_reveal.json",
        "miss_only_physical_request.json",
        "executor_admission.json",
        "physical_terminal.json",
    ):
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / "atomic_feedback_barrier.json").write_text(
        json.dumps({"feedback_released": True, "budget_consumed": 4}),
        encoding="utf-8",
    )
    pilot = root / "audits/pilot_audit_v2.json"
    pilot.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    gate = root / "status/later_seed_gate_v2.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    final = root / "paper_outputs/status/finalization_status.json"
    final.parent.mkdir(parents=True)
    final.write_text(
        json.dumps({"paper_ready": True, "core_ablation_ready": True}),
        encoding="utf-8",
    )
    real_import = orchestrator.importlib.import_module
    monkeypatch.setattr(
        orchestrator.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(
                observe_round=lambda *args: {
                    "terminal_authenticated": True,
                    "barrier_authenticated": True,
                }
            )
            if name == "scripts.stage7_core_round_worker_v2"
            else real_import(name)
        ),
    )

    snapshot = orchestrator._default_inspect(config)
    row = snapshot["trajectories"]["full:20260718"]["rounds"]["0"]

    assert snapshot["prepared"] is True
    assert snapshot["initialized"] is True
    assert snapshot["no_gpu_gate_passed"] is True
    assert snapshot["pilot_audit_passed"] is True
    assert snapshot["later_seed_gate_passed"] is True
    assert snapshot["finalization_complete"] is True
    assert row == {
        "request_frozen": True,
        "source_ready": True,
        "exact_bound": True,
        "terminal": True,
        "barrier": True,
        "selected_events": 4,
    }


def test_default_inspector_keeps_persisted_synthetic_receipt_blocked(
    config: orchestrator.OrchestratorConfig,
) -> None:
    root = config.v2_root
    (root / "prepare_state.json").parent.mkdir(parents=True)
    (root / "prepare_state.json").write_text("{}", encoding="utf-8")
    (root / "initialize_state.json").write_text("{}", encoding="utf-8")
    receipt = root / "audits/no_gpu_dry_run/dry_run_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            {
                "formal_v2_gpu_jobs_launched": 0,
                "canonical_round0_requests_written": 4,
                "round0_miss_count": 0,
                "integration_closed": False,
                "blocking_reason": (
                    "stage7_to_actual_feedback_v3_integration_not_yet_closed"
                ),
                "variant_receipts": [
                    {
                        "variant": variant,
                        "round0_miss_count": 0,
                        "miss_admission_passed": False,
                    }
                    for variant in orchestrator.CORE_VARIANTS
                ],
            }
        ),
        encoding="utf-8",
    )

    assert orchestrator._default_inspect(config)["no_gpu_gate_passed"] is False


def test_default_prepare_and_no_gpu_delegate_to_canonical_driver(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import stage7_core_no_gpu_dry_run_v2 as no_gpu

    calls: list[tuple[str, dict[str, Any]]] = []

    def prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return {"status": "initialized"}

    def dry(root, **kwargs):
        calls.append(("dry", {"root": root, **kwargs}))
        return {
            "formal_v2_gpu_jobs_launched": 0,
            "canonical_round0_requests_written": 4,
            "round0_miss_count": 16,
            "integration_closed": True,
            "blocking_reason": None,
            "variant_receipts": [
                {
                    "variant": variant,
                    "round0_miss_count": 4,
                    "miss_admission_passed": True,
                }
                for variant in orchestrator.CORE_VARIANTS
            ],
        }

    monkeypatch.setattr(no_gpu, "prepare_and_initialize_no_gpu_root", prepare)
    monkeypatch.setattr(no_gpu, "run_no_gpu_dry_run", dry)

    assert orchestrator._default_prepare(config, 55)["status"] == "initialized"
    assert (
        orchestrator._default_no_gpu(config)["canonical_round0_requests_written"] == 4
    )
    assert calls[0][1]["orchestrator_pid"] == 55
    assert calls[0][1]["expected_release_sha256"] == "a" * 64
    assert calls[0][1]["expected_manifest_file_sha256"] == "c" * 64
    assert calls[1][1]["expected_release_sha256"] == "a" * 64
    assert calls[1][1]["expected_manifest_file_sha256"] == "c" * 64
    assert calls[1][1]["expected_frozen_onnx_sha256"] == "f" * 64

    monkeypatch.setattr(
        no_gpu,
        "run_no_gpu_dry_run",
        lambda *_args, **_kwargs: {
            "formal_v2_gpu_jobs_launched": 0,
            "canonical_round0_requests_written": 3,
            "round0_miss_count": 12,
            "integration_closed": False,
            "blocking_reason": "stage7_to_actual_feedback_v3_integration_not_yet_closed",
            "variant_receipts": [],
        },
    )
    with pytest.raises(orchestrator.OrchestrationContractError, match="incomplete"):
        orchestrator._default_no_gpu(config)

    monkeypatch.setattr(
        no_gpu,
        "run_no_gpu_dry_run",
        lambda *_args, **_kwargs: {
            "formal_v2_gpu_jobs_launched": 0,
            "canonical_round0_requests_written": 4,
            "round0_miss_count": 0,
            "integration_closed": False,
            "blocking_reason": "stage7_to_actual_feedback_v3_integration_not_yet_closed",
            "variant_receipts": [
                {
                    "variant": variant,
                    "round0_miss_count": 0,
                    "miss_admission_passed": False,
                }
                for variant in orchestrator.CORE_VARIANTS
            ],
        },
    )
    with pytest.raises(orchestrator.OrchestrationContractError, match="incomplete"):
        orchestrator._default_no_gpu(config)


def test_default_runtime_validator_rehashes_identity_deployment_and_executors(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from framework.stage7 import actual_mode_v2
    from framework.stage7 import deployment_bundle_v2
    from framework.stage7 import executor_admission_v2

    monkeypatch.setattr(
        executor_admission_v2,
        "capture_process_identity",
        lambda probe, pid: _identity(pid),
    )
    monkeypatch.setattr(
        executor_admission_v2,
        "verify_frozen_actual_v3_executors",
        lambda root: [{"actual_sha256": "e" * 64}],
    )
    deployment_calls: list[dict[str, Any]] = []

    def validate_deployment(*args, **kwargs):
        deployment_calls.append({"args": args, **kwargs})
        return {
            "deployment_manifest_sha256": "d" * 64,
            "deployment_manifest_file_sha256": "c" * 64,
            "deployment_bundle_sha256": "b" * 64,
            "deployment_release_sha256": "a" * 64,
        }

    monkeypatch.setattr(
        deployment_bundle_v2, "validate_deployment_bundle", validate_deployment
    )
    receipt = config.v2_root / "audits/no_gpu_dry_run/dry_run_receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    validated_receipts: list[dict[str, Any]] = []
    monkeypatch.setattr(
        actual_mode_v2,
        "validate_no_gpu_receipt",
        lambda payload, **kwargs: validated_receipts.append(
            {"payload": payload, **kwargs}
        ),
    )

    result = orchestrator._default_validate_runtime(
        config,
        900,
        {"owner": "runner", "owner_uid": 1000},
    )

    assert result["process_identity"] == _identity(900)
    assert result["executor_verification"] == [{"actual_sha256": "e" * 64}]
    assert deployment_calls[0]["expected_release_sha256"] == "a" * 64
    assert deployment_calls[0]["expected_manifest_file_sha256"] == "c" * 64
    assert validated_receipts[0]["deployment_bundle_sha256"] == "b" * 64


def test_default_round_freeze_treats_source_preflight_as_frozen(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import stage7_core_online_ablation_v2 as online

    monkeypatch.setattr(
        online,
        "prepare_round",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            online.SourceResolutionRequiredBeforeExactCacheReveal(
                {"blocking_reason": "resolve"}
            )
        ),
    )

    result = orchestrator._default_round_step(
        config,
        orchestrator.FREEZE_SOURCE_PLAN,
        "full",
        20260718,
        0,
    )

    assert result == {
        "status": "source_plan_frozen",
        "audit": {"blocking_reason": "resolve"},
    }


def test_default_source_step_delegates_to_synchronous_source_lease_controller(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = config.v2_root / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    (config.v2_root / "prepare_state.json").write_text(
        json.dumps({"orchestrator_pid": 812}), encoding="utf-8"
    )
    from scripts import stage7_source_lease_controller_v2 as source_controller

    captured: dict[str, Any] = {}

    def resolve(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "cache_revealed"}

    monkeypatch.setattr(source_controller, "run_source_lease_controller", resolve)
    result = orchestrator._default_source_step(
        config, orchestrator.RESOLVE_SOURCE, "full", 20260718, 0
    )
    assert result["status"] == "cache_revealed"
    assert captured["v2_root"] == config.v2_root
    assert captured["repo_root"] == config.repo_root
    assert captured["variant"] == "full"
    assert captured["orchestrator_pid"] == 812

    monkeypatch.setattr(
        source_controller,
        "run_source_lease_controller",
        lambda **_kwargs: {
            "status": "waiting_for_source_gpu_capacity",
            "reason": "not enough H800 UUIDs",
            "selected_event_budget_delta": 0,
        },
    )
    with pytest.raises(orchestrator.RetryableOrchestrationError) as retry:
        orchestrator._default_source_step(
            config, orchestrator.RESOLVE_SOURCE, "full", 20260718, 0
        )
    assert retry.value.category == "source_resource_wait"


def test_default_bind_exact_requires_canonical_not_staged_source_result(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = config.v2_root / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    with pytest.raises(orchestrator.RetryableOrchestrationError) as waiting:
        orchestrator._default_round_step(
            config, orchestrator.BIND_EXACT, "full", 20260718, 0
        )
    assert waiting.value.category == "source_evidence_wait"

    staged = directory / "source_resolution_result.staged.json"
    staged.write_text("{}", encoding="utf-8")
    with pytest.raises(orchestrator.RetryableOrchestrationError):
        orchestrator._default_round_step(
            config, orchestrator.BIND_EXACT, "full", 20260718, 0
        )

    canonical = directory / "source_resolution_result.json"
    canonical.write_text("{}", encoding="utf-8")
    from scripts import stage7_core_online_ablation_v2 as online

    monkeypatch.setattr(
        online,
        "bind_reveal_after_source_ready",
        lambda *args, **kwargs: {
            "status": "cache_revealed",
            "source_result_path": str(kwargs["source_result_path"]),
        },
    )
    result = orchestrator._default_round_step(
        config, orchestrator.BIND_EXACT, "full", 20260718, 0
    )
    assert result["status"] == "cache_revealed"
    assert result["source_result_path"] == str(canonical)


def _measurement_artifacts(config: orchestrator.OrchestratorConfig) -> Path:
    directory = config.v2_root / "variants/full/seed_20260718/round_00"
    directory.mkdir(parents=True)
    rows = [
        {"row_id": f"row-{index}", "width": [16 + index, 32, 64]} for index in range(4)
    ]
    (directory / "logical_request.json").write_text(
        json.dumps(
            {
                "measurement_request_sha256": "r" * 64,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    (directory / "miss_only_physical_request.json").write_text(
        json.dumps(
            {
                "physical_request_sha256": "p" * 64,
                "logical_row_bindings": [
                    {"candidate_id": row["row_id"]} for row in rows
                ],
            }
        ),
        encoding="utf-8",
    )
    (directory / "source_resolution_result.json").write_text(
        json.dumps({"source_resolution_result_sha256": "s" * 64}),
        encoding="utf-8",
    )
    return directory


class FakeBatchRequest:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
        self.controller_id = f"{values['trajectory_id']}:round_{values['round_index']}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeBatchRequest) and all(
            getattr(self, field, None) == getattr(other, field, None)
            for field in (
                "controller_id",
                "request_sha256",
                "physical_request_sha256",
                "expected_release_sha256",
                "expected_manifest_file_sha256",
                "selected_row_ids",
            )
        )


class FakeMeasurementScheduler:
    def __init__(self, controller: dict[str, Any] | None = None) -> None:
        self.controller = copy.deepcopy(controller)
        self.schedule_calls: list[FakeBatchRequest] = []
        self.observe_calls = 0
        self.monitor_calls = 0
        self.reap_calls = 0
        self.block_reason: str | None = None
        self.reap_replacement: dict[str, Any] | None = None
        self.completion_error: Exception | None = None
        self.completion_validator = object()
        self.controller_request_calls = 0

    def status(self) -> dict[str, Any]:
        controllers = (
            {self.controller["controller_id"]: copy.deepcopy(self.controller)}
            if self.controller
            else {}
        )
        return {
            "controllers": controllers,
            "selected_event_budget_consumed": int(
                (self.controller or {}).get("selected_event_budget_consumed", 0)
            ),
        }

    def reap_controllers(self) -> list[str]:
        self.reap_calls += 1
        if self.reap_replacement is not None:
            self.controller = copy.deepcopy(self.reap_replacement)
        return []

    def _controller_request(self, controller: dict[str, Any]) -> FakeBatchRequest:
        self.controller_request_calls += 1
        request = copy.copy(self.schedule_calls[-1]) if self.schedule_calls else None
        if request is None:
            request = FakeBatchRequest(
                trajectory_id="full:seed_20260718",
                round_index=0,
                request_sha256=controller["request_sha256"],
                physical_request_sha256=controller["physical_request_sha256"],
                expected_release_sha256=controller["expected_release_sha256"],
                expected_manifest_file_sha256=controller[
                    "expected_manifest_file_sha256"
                ],
                selected_row_ids=("row-0", "row-1", "row-2", "row-3"),
            )
        return request

    def monitor_occupancy(self) -> list[FakeBatchRequest]:
        self.monitor_calls += 1
        return []

    def observe_gpus(self) -> dict[str, Any]:
        self.observe_calls += 1
        return {}

    def schedule(self, requests: list[FakeBatchRequest]) -> dict[str, Any]:
        self.schedule_calls.extend(requests)
        if self.block_reason:
            return {
                "status": "waiting_for_four_idle_gpus",
                "launched": [],
                "blocked": [{"reason": self.block_reason}],
            }
        request = requests[0]
        self.controller = {
            "controller_id": request.controller_id,
            "status": "running",
            "request_sha256": request.request_sha256,
            "physical_request_sha256": request.physical_request_sha256,
            "expected_release_sha256": request.expected_release_sha256,
            "expected_manifest_file_sha256": request.expected_manifest_file_sha256,
            "selected_event_budget_consumed": 0,
        }
        return {"status": "running", "launched": list(requests), "blocked": []}

    def resume(self, requests: list[FakeBatchRequest], *, once: bool) -> dict[str, Any]:
        assert once is True
        if self.controller and self.controller.get("status") == "running":
            return {
                "already_live": [self.controller["controller_id"]],
                "status": self.status(),
            }
        self.observe_gpus()
        self.schedule(requests)
        return {"already_live": [], "status": self.status()}


def _scheduler_module(
    scheduler: FakeMeasurementScheduler,
    execute_calls: list[dict[str, Any]],
) -> SimpleNamespace:
    def validate_completion(
        request: FakeBatchRequest, _validator: object
    ) -> dict[str, Any]:
        if scheduler.completion_error is not None:
            raise scheduler.completion_error
        return {
            "scheduler_completion_lineage": copy.deepcopy(
                (scheduler.controller or {})["completion_lineage"]
            )
        }

    return SimpleNamespace(
        V2BatchRequest=FakeBatchRequest,
        V2Stage7Scheduler=lambda **_kwargs: scheduler,
        CORE_CONTROLLER=Path("/frozen/stage7_core_round_controller_v2.sh"),
        _requests=SimpleNamespace(
            resolved_source_lock_sha256=lambda _plan, _source: "l" * 64,
            validate_completion_receipt=validate_completion,
        ),
        execute_round=lambda **kwargs: execute_calls.append(kwargs),
    )


def _patch_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: FakeMeasurementScheduler,
    execute_calls: list[dict[str, Any]],
) -> None:
    module = _scheduler_module(scheduler, execute_calls)

    monkeypatch.setattr(
        scheduler_adapter.importlib,
        "import_module",
        lambda name: (
            module
            if name == "scripts.stage7_core_ablation_scheduler_v2"
            else (_ for _ in ()).throw(ImportError(name))
        ),
    )
    scheduler_adapter.measurement_scheduler.cache_clear()


def test_measurement_scheduler_uses_canonical_v2_status_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[Path] = []
    module = SimpleNamespace(
        V2BatchRequest=FakeBatchRequest,
        V2Stage7Scheduler=lambda **kwargs: captured.append(kwargs["status_dir"])
        or object(),
    )
    monkeypatch.setattr(
        scheduler_adapter.importlib, "import_module", lambda _name: module
    )
    scheduler_adapter.measurement_scheduler.cache_clear()

    scheduler_adapter.measurement_scheduler(str(tmp_path))

    assert captured == [tmp_path / "status"]


def test_ready_to_measure_submits_exactly_once_through_scheduler_and_never_worker(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measurement_artifacts(config)
    scheduler = FakeMeasurementScheduler()
    execute_calls: list[dict[str, Any]] = []
    _patch_scheduler(monkeypatch, scheduler, execute_calls)

    with pytest.raises(
        orchestrator.RetryableOrchestrationError, match="controller launched"
    ):
        orchestrator._default_round_step(
            config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
        )

    assert scheduler.reap_calls == 1
    assert scheduler.observe_calls == 1
    assert len(scheduler.schedule_calls) == 1
    request = scheduler.schedule_calls[0]
    assert request.request_sha256 == "r" * 64
    assert request.expected_release_sha256 == "a" * 64
    assert request.expected_manifest_file_sha256 == "c" * 64
    assert request.selected_row_ids == ("row-0", "row-1", "row-2", "row-3")
    assert execute_calls == []


def test_terminal_without_barrier_recovers_with_no_hardware_or_relaunch(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _measurement_artifacts(config)
    terminal = directory / "physical_terminal.json"
    terminal.write_text("{}", encoding="utf-8")
    finalize_calls: list[dict[str, Any]] = []
    controller = {
        "controller_id": "full:seed_20260718:round_0",
        "status": "running",
        "request_sha256": "r" * 64,
        "physical_request_sha256": "p" * 64,
        "expected_release_sha256": "a" * 64,
        "expected_manifest_file_sha256": "c" * 64,
        "selected_event_budget_consumed": 0,
    }
    scheduler = FakeMeasurementScheduler(controller)
    scheduler_module = _scheduler_module(scheduler, [])
    worker = SimpleNamespace(
        observe_round=lambda *_args: {
            "terminal_authenticated": True,
            "barrier_authenticated": False,
            "terminal_path": str(terminal),
        }
    )
    online = SimpleNamespace(
        finalize_round=lambda *_args, **kwargs: finalize_calls.append(kwargs)
        or {"status": "barrier_committed"}
    )
    modules = {
        "scripts.stage7_core_ablation_scheduler_v2": scheduler_module,
        "scripts.stage7_core_round_worker_v2": worker,
        "scripts.stage7_core_online_ablation_v2": online,
    }
    monkeypatch.setattr(
        scheduler_adapter.importlib,
        "import_module",
        lambda name: modules[name],
    )
    scheduler_adapter.measurement_scheduler.cache_clear()

    result = orchestrator._default_round_step(
        config, orchestrator.FINALIZE_TERMINAL_BARRIER, "full", 20260718, 0
    )

    assert result["status"] == "barrier_committed"
    assert len(finalize_calls) == 1
    assert finalize_calls[0]["terminal_payload_path"] == terminal
    assert finalize_calls[0]["expected_release_sha256"] == "a" * 64
    assert finalize_calls[0]["expected_manifest_file_sha256"] == "c" * 64
    assert scheduler.controller_request_calls == 1
    assert scheduler.reap_calls == 0
    assert scheduler.observe_calls == 0
    assert scheduler.schedule_calls == []


def test_barrier_only_recovery_missing_scheduler_controller_fails_closed(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _measurement_artifacts(config)
    terminal = directory / "physical_terminal.json"
    terminal.write_text("{}", encoding="utf-8")
    scheduler = FakeMeasurementScheduler()
    modules = {
        "scripts.stage7_core_ablation_scheduler_v2": _scheduler_module(scheduler, []),
        "scripts.stage7_core_round_worker_v2": SimpleNamespace(
            observe_round=lambda *_args: {
                "terminal_authenticated": True,
                "barrier_authenticated": False,
                "terminal_path": str(terminal),
            }
        ),
        "scripts.stage7_core_online_ablation_v2": SimpleNamespace(
            finalize_round=lambda *_args, **_kwargs: {"status": "must_not_finalize"}
        ),
    }
    monkeypatch.setattr(
        scheduler_adapter.importlib, "import_module", lambda name: modules[name]
    )
    scheduler_adapter.measurement_scheduler.cache_clear()

    with pytest.raises(
        orchestrator.OrchestrationContractError,
        match="barrier-only scheduler lineage",
    ):
        orchestrator._default_round_step(
            config, orchestrator.FINALIZE_TERMINAL_BARRIER, "full", 20260718, 0
        )

    assert scheduler.reap_calls == 0
    assert scheduler.schedule_calls == []


@pytest.mark.parametrize("unsafe_kind", ("symlink", "non_mapping"))
def test_barrier_only_mapping_reader_rejects_unsafe_or_non_mapping_json(
    tmp_path: Path, unsafe_kind: str
) -> None:
    path = tmp_path / "artifact.json"
    if unsafe_kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}", encoding="utf-8")
        path.symlink_to(target)
    else:
        path.write_text("[]", encoding="utf-8")

    with pytest.raises(scheduler_adapter.SchedulerAdapterError, match="unsafe"):
        scheduler_adapter._read_mapping(path, label="barrier artifact")


def test_scheduler_resource_wait_is_zero_budget_and_reuses_same_request(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measurement_artifacts(config)
    scheduler = FakeMeasurementScheduler()
    scheduler.block_reason = "capacity"
    execute_calls: list[dict[str, Any]] = []
    _patch_scheduler(monkeypatch, scheduler, execute_calls)

    request_ids: list[str] = []
    for _ in range(2):
        with pytest.raises(orchestrator.RetryableOrchestrationError, match="capacity"):
            orchestrator._default_round_step(
                config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
            )
        request_ids.append(scheduler.schedule_calls[-1].request_sha256)

    assert request_ids == ["r" * 64, "r" * 64]
    assert scheduler.status()["selected_event_budget_consumed"] == 0
    assert len(scheduler.schedule_calls) == 2
    assert execute_calls == []


@pytest.mark.parametrize("status", ("running", "feedback_complete"))
def test_scheduler_recovery_observes_existing_controller_without_resubmission(
    status: str,
    config: orchestrator.OrchestratorConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _measurement_artifacts(config)
    controller = {
        "controller_id": "full:seed_20260718:round_0",
        "status": status,
        "request_sha256": "r" * 64,
        "physical_request_sha256": "p" * 64,
        "expected_release_sha256": "a" * 64,
        "expected_manifest_file_sha256": "c" * 64,
        "selected_event_budget_consumed": 4 if status == "feedback_complete" else 0,
        "completion_lineage": (
            {
                "logical_request_sha256": "r" * 64,
                "physical_request_sha256": "p" * 64,
                "selected_row_ids": ["row-0", "row-1", "row-2", "row-3"],
            }
            if status == "feedback_complete"
            else None
        ),
    }
    scheduler = FakeMeasurementScheduler(controller)
    execute_calls: list[dict[str, Any]] = []
    _patch_scheduler(monkeypatch, scheduler, execute_calls)

    if status == "running":
        with pytest.raises(
            orchestrator.RetryableOrchestrationError, match="controller running"
        ):
            orchestrator._default_round_step(
                config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
            )
    else:
        result = orchestrator._default_round_step(
            config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
        )
        assert result["status"] == "feedback_complete"

    assert scheduler.reap_calls == 1
    assert scheduler.monitor_calls == (1 if status == "running" else 0)
    assert scheduler.schedule_calls == []
    assert execute_calls == []


def test_scheduler_validates_controller_identity_before_mutating_reap(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measurement_artifacts(config)
    valid = {
        "controller_id": "full:seed_20260718:round_0",
        "status": "running",
        "request_sha256": "r" * 64,
        "physical_request_sha256": "p" * 64,
        "expected_release_sha256": "a" * 64,
        "expected_manifest_file_sha256": "c" * 64,
        "selected_event_budget_consumed": 0,
    }
    scheduler = FakeMeasurementScheduler({**valid, "physical_request_sha256": "x" * 64})
    scheduler.reap_replacement = valid
    _patch_scheduler(monkeypatch, scheduler, [])

    with pytest.raises(
        orchestrator.OrchestrationContractError, match="controller identity"
    ):
        orchestrator._default_round_step(
            config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
        )

    assert scheduler.reap_calls == 0
    assert scheduler.schedule_calls == []


def test_scheduler_rejects_feedback_complete_without_authenticated_receipt(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measurement_artifacts(config)
    controller = {
        "controller_id": "full:seed_20260718:round_0",
        "status": "feedback_complete",
        "request_sha256": "r" * 64,
        "physical_request_sha256": "p" * 64,
        "expected_release_sha256": "a" * 64,
        "expected_manifest_file_sha256": "c" * 64,
        "selected_event_budget_consumed": 4,
        "completion_lineage": {
            "logical_request_sha256": "r" * 64,
            "physical_request_sha256": "p" * 64,
            "selected_row_ids": ["row-0", "row-1", "row-2", "row-3"],
        },
    }
    scheduler = FakeMeasurementScheduler(controller)
    scheduler.completion_error = ValueError("barrier receipt invalid")
    _patch_scheduler(monkeypatch, scheduler, [])

    with pytest.raises(
        orchestrator.OrchestrationContractError, match="barrier receipt invalid"
    ):
        orchestrator._default_round_step(
            config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
        )

    assert scheduler.schedule_calls == []


def test_scheduler_infrastructure_retry_resubmits_same_request_at_zero_budget(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measurement_artifacts(config)
    controller = {
        "controller_id": "full:seed_20260718:round_0",
        "status": "infrastructure_retry_required",
        "request_sha256": "r" * 64,
        "retry_request_sha256": "r" * 64,
        "physical_request_sha256": "p" * 64,
        "expected_release_sha256": "a" * 64,
        "expected_manifest_file_sha256": "c" * 64,
        "selected_event_budget_consumed": 0,
    }
    scheduler = FakeMeasurementScheduler(controller)
    execute_calls: list[dict[str, Any]] = []
    _patch_scheduler(monkeypatch, scheduler, execute_calls)

    with pytest.raises(
        orchestrator.RetryableOrchestrationError, match="controller launched"
    ):
        orchestrator._default_round_step(
            config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
        )

    assert len(scheduler.schedule_calls) == 1
    assert (
        scheduler.schedule_calls[0].request_sha256 == controller["retry_request_sha256"]
    )
    assert scheduler.status()["selected_event_budget_consumed"] == 0
    assert execute_calls == []


def test_scheduler_dependency_or_request_drift_fails_without_worker_fallback(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _measurement_artifacts(config)
    execute_calls: list[dict[str, Any]] = []
    real_import = scheduler_adapter.importlib.import_module
    monkeypatch.setattr(
        scheduler_adapter.importlib,
        "import_module",
        lambda name: (
            (_ for _ in ()).throw(ImportError(name))
            if name == "scripts.stage7_core_ablation_scheduler_v2"
            else real_import(name)
        ),
    )
    scheduler_adapter.measurement_scheduler.cache_clear()
    with pytest.raises(
        orchestrator.RetryableOrchestrationError, match="scheduler deployment"
    ):
        orchestrator._default_round_step(
            config, orchestrator.EXECUTE_MEASUREMENT, "full", 20260718, 0
        )
    assert execute_calls == []


def test_scheduler_adapter_builds_current_real_v2_batch_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from framework.tests import test_stage7_core_ablation_scheduler_v2 as fixture

    module = fixture.load_module()
    root = fixture.v2_root(module, tmp_path)
    expected = fixture.request(module, root, "full:seed_20260718")
    directory = expected.trajectory_path
    logical = json.loads((directory / "logical_request.json").read_text())
    logical["rows"] = [
        {"row_id": row_id, "width": [16 + index, 32, 64]}
        for index, row_id in enumerate(expected.selected_row_ids)
    ]
    plan = json.loads(expected.physical_plan_path.read_text())
    source = json.loads((directory / "source_resolution_result.json").read_text())
    fake_scheduler = object()
    monkeypatch.setattr(
        scheduler_adapter,
        "measurement_scheduler",
        lambda _root: (module, fake_scheduler),
    )

    actual_module, scheduler, actual = scheduler_adapter._batch_request(
        v2_root=root,
        variant="full",
        seed=20260718,
        round_index=0,
        expected_release_sha256=expected.expected_release_sha256,
        expected_manifest_file_sha256=expected.expected_manifest_file_sha256,
        directory=directory,
        logical=logical,
        physical_plan=plan,
        source_result=source,
    )

    assert actual_module is module
    assert scheduler is fake_scheduler
    assert isinstance(actual, module.V2BatchRequest)
    assert actual.request_sha256 == expected.request_sha256
    assert actual.selected_row_ids == expected.selected_row_ids
    assert actual.physical_request_sha256 == expected.physical_request_sha256


def _write_complete_seed(root: Path, seed: int) -> None:
    for variant in VARIANTS:
        for index in range(4):
            directory = (
                root / "variants" / variant / f"seed_{seed}" / f"round_{index:02d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            for name in (
                "logical_request.json",
                "source_resolution_plan.json",
                "source_resolution_result.json",
                "exact_selection_binding.json",
                "cache_snapshot_before_reveal.json",
                "cache_reveal.json",
                "miss_only_physical_request.json",
                "executor_admission.json",
                "physical_terminal.json",
            ):
                (directory / name).write_text("{}", encoding="utf-8")
            (directory / "atomic_feedback_barrier.json").write_text(
                json.dumps({"feedback_released": True, "budget_consumed": 4}),
                encoding="utf-8",
            )


def test_default_pilot_audit_and_later_gate_are_strict_immutable_outputs(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = {}
    for variant in VARIANTS:
        trajectories[f"{variant}:{PILOT_SEED}"] = {
            "rounds": {
                str(index): {"barrier": True, "selected_events": 4}
                for index in range(4)
            }
        }
    monkeypatch.setattr(
        orchestrator,
        "_default_inspect",
        lambda _config: {"trajectories": trajectories},
    )

    audit = orchestrator._default_audit_pilot(config)
    gate = orchestrator._default_gate_later(config)

    assert audit["status"] == "passed"
    assert gate["pilot_selected_event_count"] == 64
    assert (config.v2_root / "audits/pilot_audit_v2.json").is_file()
    assert (config.v2_root / "status/later_seed_gate_v2.json").is_file()


def test_default_finalize_calls_formal_finalizer(
    config: orchestrator.OrchestratorConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import stage7_finalize_core_ablation_v2 as finalizer

    monkeypatch.setattr(
        finalizer,
        "finalize_v2",
        lambda source, destination, repo_root: {
            "source": str(source),
            "destination": str(destination),
            "repo": str(repo_root),
        },
    )
    result = orchestrator._default_finalize(config)
    assert result["destination"].endswith("/v2/paper_outputs")


def test_main_once_builds_canonical_config_without_daemonizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    prepare = tmp_path / "prepare.json"
    manifest = tmp_path / "manifest.json"
    frozen = tmp_path / "frozen.onnx"
    prepare.write_text("{}", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    frozen.write_bytes(b"x")
    captured: dict[str, Any] = {}

    class FakeMachine:
        def __init__(self, config):
            captured["config"] = config

        def run(self, *, max_ticks=None):
            captured["max_ticks"] = max_ticks
            return {"phase": "NO_GPU_GATE", "status": "running"}

    monkeypatch.setattr(orchestrator, "PersistentCoreAblationOrchestrator", FakeMachine)
    result = orchestrator.main(
        [
            "--v1-root",
            str(tmp_path / "v1"),
            "--v2-root",
            str(tmp_path / "v2"),
            "--repo-root",
            str(tmp_path),
            "--prepare-inputs-json",
            str(prepare),
            "--embedded-source-manifest-json",
            str(manifest),
            "--frozen-onnx",
            str(frozen),
            "--frozen-onnx-sha256",
            "f" * 64,
            "--expected-release-sha256",
            "a" * 64,
            "--expected-manifest-file-sha256",
            "c" * 64,
            "--once",
        ]
    )

    assert result == 0
    assert captured["max_ticks"] == 1
    assert captured["config"].expected_release_sha256 == "a" * 64
    assert captured["config"].expected_manifest_file_sha256 == "c" * 64
    assert '"phase": "NO_GPU_GATE"' in capsys.readouterr().out
