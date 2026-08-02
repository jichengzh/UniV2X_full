from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage7_core_round_worker_v2.py"
UUIDS = tuple(f"GPU-{index:032x}" for index in range(4))
REQUEST_SHA = "a" * 64
PHYSICAL_SHA = "b" * 64
CONTRACT_SHA = "c" * 64
SOURCE_PLAN_SHA = "d" * 64
SOURCE_RESULT_SHA = "e" * 64
SOURCE_LOCK_SHA = "f" * 64
RELEASE_PIN = "7" * 64
MANIFEST_FILE_PIN = "8" * 64
PINS = {
    "expected_release_sha256": RELEASE_PIN,
    "expected_manifest_file_sha256": MANIFEST_FILE_PIN,
}


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "stage7_core_round_worker_v2_for_test", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _launch_gate(
    module: Any,
    tmp_path: Path,
    *,
    pid: int = 4321,
    token: str = "6" * 64,
    external_status: bool = False,
) -> tuple[Path, dict[str, str]]:
    root = (tmp_path / "v2").resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_dir = (
        (tmp_path / "scheduler-status").resolve()
        if external_status
        else root / "status"
    )
    gate_dir = status_dir / "launch_gates"
    gate_dir.mkdir(parents=True)
    gate_dir.chmod(0o700)
    state_path = status_dir / "scheduler_state.json"
    controller = {
        "controller_id": "full:seed_20260718:round_0",
        "request_sha256": REQUEST_SHA,
        "pid": pid,
    }
    token_sha = hashlib.sha256(token.encode("utf-8")).hexdigest()
    controller_id_sha = hashlib.sha256(
        b"full:seed_20260718:round_0"
    ).hexdigest()
    gate_path = gate_dir / (
        f"gate_{controller_id_sha[:16]}_{REQUEST_SHA[:16]}_{token_sha[:16]}.json"
    )
    controller = {
        **controller,
        "launch_gate_path": str(gate_path),
        "launch_gate_token_sha256": token_sha,
        "launch_gate_schema": "stage7_v2_controller_launch_gate_v1",
    }
    _write(
        state_path,
        {
            "schema_version": "stage7_ablation_scheduler_state_v1",
            "controllers": {controller["controller_id"]: controller},
        },
    )
    controller_sha = module._canonical_sha(controller)
    unsigned = {
        "schema_version": "stage7_v2_controller_launch_gate_v1",
        "controller_id": "full:seed_20260718:round_0",
        "request_sha256": REQUEST_SHA,
        "child_pid": pid,
        "owner_uid": os.getuid(),
        "gate_token_sha256": token_sha,
        "scheduler_controller_record_sha256": controller_sha,
        "released_at_wall_time": 123.5,
    }
    _write(gate_path, {**unsigned, "release_sha256": module._canonical_sha(unsigned)})
    gate_path.chmod(0o600)
    return root, {
        "STAGE7_V2_LAUNCH_GATE_PATH": str(gate_path),
        "STAGE7_V2_LAUNCH_GATE_TOKEN": token,
        "STAGE7_V2_LAUNCH_GATE_TIMEOUT_SECONDS": "60",
    }


def _fixture(tmp_path: Path, *, physical_count: int = 4):
    root = (tmp_path / "v2").resolve()
    round_dir = (
        root / "variants" / "full" / "seed_20260718" / "round_00"
    )
    round_dir.mkdir(parents=True)
    row_ids = [f"candidate-{index}" for index in range(4)]
    physical_ids = row_ids[:physical_count]
    logical = {
        "measurement_request_sha256": REQUEST_SHA,
        "rows": [{"row_id": value} for value in row_ids],
    }
    source_plan = {
        "logical_request_sha256": REQUEST_SHA,
        "source_resolution_plan_sha256": SOURCE_PLAN_SHA,
    }
    source_result = {
        "logical_request_sha256": REQUEST_SHA,
        "source_resolution_plan_sha256": SOURCE_PLAN_SHA,
        "source_resolution_result_sha256": SOURCE_RESULT_SHA,
        "rows": [{"candidate_id": value} for value in row_ids],
    }
    binding = {
        "logical_request": {"logical_request_sha256": REQUEST_SHA},
        "selection_binding_sha256": "1" * 64,
    }
    cache = {"schema_version": "stage7_core_cache_v2", "entries": {}, "lineage": []}
    reveal = {
        "logical_request_sha256": REQUEST_SHA,
        "entries": [
            {
                "candidate_id": value,
                "disposition": "miss" if value in physical_ids else "hit",
            }
            for value in row_ids
        ],
    }
    physical = {
        "schema_version": "stage7_actual_v3_miss_only_physical_request_v2",
        "logical_request_sha256": REQUEST_SHA,
        "physical_request_sha256": PHYSICAL_SHA,
        "logical_row_count": 4,
        "physical_row_count": physical_count,
        "rows": [{"row_id": value} for value in physical_ids],
        "logical_row_bindings": [
            {
                "logical_row_index": index,
                "candidate_id": value,
                "disposition": "miss" if value in physical_ids else "hit",
            }
            for index, value in enumerate(row_ids)
        ],
    }
    admission = {
        "schema_version": "stage7_actual_v3_executor_admission_v2",
        "admission_passed": True,
        "contract_sha256": CONTRACT_SHA,
        "logical_request_sha256": REQUEST_SHA,
        "physical_request_sha256": PHYSICAL_SHA,
        "logical_row_count": 4,
        "physical_row_count": physical_count,
        "execution_primitive": "existing_stage5_stage3_actual_feedback_v3",
        "gpu_jobs_launched": 0,
        "admission_sha256": "2" * 64,
    }
    artifacts = {
        "logical_request.json": logical,
        "selection_binding.json": {"selection_frozen": True},
        "source_resolution_plan.json": source_plan,
        "source_resolution_result.json": source_result,
        "exact_selection_binding.json": binding,
        "cache_snapshot_before_reveal.json": cache,
        "cache_reveal.json": reveal,
        "miss_only_physical_request.json": physical,
        "executor_admission.json": admission,
    }
    for name, value in artifacts.items():
        _write(round_dir / name, value)
    _write(root / "contracts/core_ablation_v2.json", {"contract_sha256": CONTRACT_SHA})
    _write(
        root / "audits/no_gpu_dry_run/dry_run_receipt.json",
        {
            "schema_version": "stage7_core_no_gpu_dry_run_v2",
            "formal_v2_root": str(root),
            "formal_v2_gpu_jobs_launched": 0,
            "cuda_visible_devices": "",
            "synthetic_non_measurement": True,
            "actual_v3_hardware_evidence": False,
            "eligible_for_cache_append": False,
            "eligible_for_formal_finalization": False,
            "dry_run_receipt_sha256": "3" * 64,
        },
    )
    controller_id = "full:seed_20260718:round_0"
    _write(
        root / "status/scheduler_state.json",
        {
            "schema_version": "stage7_ablation_scheduler_state_v1",
            "controllers": {
                controller_id: {
                    "controller_id": controller_id,
                    "trajectory_id": "full:seed_20260718",
                    "trajectory_path": str(round_dir),
                    "round_index": 0,
                    "request_sha256": REQUEST_SHA,
                    "physical_request_sha256": PHYSICAL_SHA,
                    "source_resolution_result_sha256": SOURCE_RESULT_SHA,
                    "resolved_source_lock_sha256": SOURCE_LOCK_SHA,
                    "source_lock_key": SOURCE_LOCK_SHA,
                    "expected_release_sha256": RELEASE_PIN,
                    "expected_manifest_file_sha256": MANIFEST_FILE_PIN,
                    "gpu_uuids": list(UUIDS),
                    "gpu_models": {value: "NVIDIA H800" for value in UUIDS},
                    "status": "running",
                }
            },
        },
    )
    return root, round_dir, artifacts


def _patch_admission(monkeypatch: pytest.MonkeyPatch, module, artifacts):
    _patch_launch_gate(monkeypatch, module)
    monkeypatch.setattr(
        module,
        "validate_formal_v2_root",
        lambda root, **_kwargs: {
            "root": Path(root).resolve(),
            "contract": {"contract_sha256": CONTRACT_SHA},
        },
    )
    monkeypatch.setattr(
        module,
        "validate_measurement_artifacts",
        lambda _round: {
            "logical_request": artifacts["logical_request.json"],
            "source_plan": artifacts["source_resolution_plan.json"],
            "source_result": artifacts["source_resolution_result.json"],
            "exact_selection_binding": artifacts["exact_selection_binding.json"],
            "cache_snapshot": artifacts["cache_snapshot_before_reveal.json"],
            "cache_reveal": artifacts["cache_reveal.json"],
            "physical_plan": artifacts["miss_only_physical_request.json"],
            "executor_admission": artifacts["executor_admission.json"],
        },
    )
    monkeypatch.setattr(
        module,
        "_validate_no_gpu_receipt",
        lambda *_args, **_kwargs: {"dry_run_receipt_sha256": "3" * 64},
    )
    monkeypatch.setattr(module, "validate_physical_terminal_batch", lambda value: value)


def _patch_launch_gate(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    payload = {
        "controller_id": "full:seed_20260718:round_0",
        "request_sha256": REQUEST_SHA,
    }
    monkeypatch.setattr(
        module,
        "_await_launch_gate",
        lambda **_kwargs: module._AuthenticatedLaunchGate(
            payload, module._GATE_FACTORY_KEY
        ),
    )


def _terminal(request_sha: str = REQUEST_SHA) -> dict[str, Any]:
    return {
        "schema_version": "stage7_actual_v3_physical_terminal_batch_v2",
        "barrier_release_allowed": True,
        "retry_logical_request_sha256": None,
        "lineage_inputs": {
            "logical_request_sha256": request_sha,
            "physical_request_sha256": PHYSICAL_SHA,
        },
        "projection_artifact": {
            "measurement_request_sha256": request_sha,
        },
        "rows": [],
        "failures": [],
        "physical_terminal_batch_sha256": "4" * 64,
    }


def _write_execution_terminal(round_dir: Path, terminal: dict[str, Any]) -> Path:
    execution = round_dir / "actual_v3_execution" / PHYSICAL_SHA
    unsigned = {
        "schema_version": "stage7_actual_v3_execution_contract_v2",
        "logical_request_sha256": REQUEST_SHA,
        "physical_request_sha256": PHYSICAL_SHA,
        "deployment_bundle_sha256": "6" * 64,
        "no_gpu_receipt_sha256": "3" * 64,
        "dry_run": False,
    }
    _write(
        execution / "execution_contract.json",
        {
            **unsigned,
            "execution_contract_sha256": hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
    )
    path = execution / "terminal_payload.json"
    _write(path, terminal)
    return path


def _barrier() -> dict[str, Any]:
    return {
        "schema_version": "stage7_actual_v3_atomic_feedback_barrier_v2",
        "logical_request_sha256": REQUEST_SHA,
        "feedback_released": True,
        "budget_consumed": 4,
        "silent_surrogate_fallback_count": 0,
        "barrier_receipt_sha256": "5" * 64,
    }


def test_actual_executor_argv_is_exact_and_uses_scheduler_owned_dynamic_uuids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    calls: list[dict[str, Any]] = []

    def runner(argv, *, env, cwd, check):
        calls.append(
            {"argv": tuple(argv), "env": dict(env), "cwd": Path(cwd), "check": check}
        )
        _write_execution_terminal(round_dir, _terminal())
        return subprocess.CompletedProcess(argv, 0)

    def finalizer(*_args, **_kwargs):
        _write(round_dir / "atomic_feedback_barrier.json", _barrier())
        return _barrier()

    monkeypatch.setattr(module, "validate_committed_barrier", lambda _path: _barrier())
    result = module.execute_round(
        v2_root=root,
        variant="full",
        seed=20260718,
        round_index=0,
        request_sha256=REQUEST_SHA,
        **PINS,
        subprocess_runner=runner,
        finalizer=finalizer,
        environ={
            "CUDA_VISIBLE_DEVICES": ",".join(UUIDS),
            "STAGE7_BUNDLE_CODE_ROOT": str(ROOT),
            "STAGE7_FROZEN_REPO_ROOT": str(ROOT / "frozen"),
        },
    )

    assert result["status"] == "feedback_complete"
    assert len(calls) == 1
    expected = (
        str(ROOT / "scripts/stage7_execute_actual_v3_misses_v2.sh"),
        "--physical-plan-json",
        str(round_dir / "miss_only_physical_request.json"),
        "--output-dir",
        str(round_dir / "actual_v3_execution" / PHYSICAL_SHA),
        "--formal-v2-root",
        str(root),
        "--expected-release-sha256",
        RELEASE_PIN,
        "--expected-manifest-file-sha256",
        MANIFEST_FILE_PIN,
        "--logical-request-json",
        str(round_dir / "logical_request.json"),
        "--selection-binding-json",
        str(round_dir / "exact_selection_binding.json"),
        "--cache-reveal-json",
        str(round_dir / "cache_reveal.json"),
        "--cache-snapshot-json",
        str(round_dir / "cache_snapshot_before_reveal.json"),
        "--no-gpu-receipt-json",
        str(root / "audits/no_gpu_dry_run/dry_run_receipt.json"),
        "--admission-json",
        str(
            round_dir
            / "actual_v3_execution"
            / PHYSICAL_SHA
            / "miss_execution_admission.json"
        ),
        "--request-sha256",
        REQUEST_SHA,
        "--executor-admission-json",
        str(round_dir / "executor_admission.json"),
        "--contract-sha256",
        CONTRACT_SHA,
        "--source-plan-json",
        str(round_dir / "source_resolution_plan.json"),
        "--source-result-json",
        str(round_dir / "source_resolution_result.json"),
        "--source-plan-sha256",
        SOURCE_PLAN_SHA,
        "--source-result-sha256",
        SOURCE_RESULT_SHA,
        "--resolved-source-lock-sha256",
        SOURCE_LOCK_SHA,
        "--gpu-uuids",
        ",".join(UUIDS),
        "--scheduler-state-json",
        str(root / "status/scheduler_state.json"),
        "--controller-id",
        "full:seed_20260718:round_0",
    )
    assert calls[0]["argv"] == expected
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == ",".join(UUIDS)
    assert calls[0]["cwd"] == round_dir
    assert calls[0]["check"] is True


def test_authenticated_existing_terminal_is_not_relaunched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    _write_execution_terminal(round_dir, _terminal())
    calls: list[object] = []

    def finalizer(*_args, **_kwargs):
        _write(round_dir / "atomic_feedback_barrier.json", _barrier())
        return _barrier()

    monkeypatch.setattr(module, "validate_committed_barrier", lambda _path: _barrier())
    result = module.execute_round(
        v2_root=root,
        variant="full",
        seed=20260718,
        round_index=0,
        request_sha256=REQUEST_SHA,
        **PINS,
        subprocess_runner=lambda *_args, **_kwargs: calls.append("launched"),
        finalizer=finalizer,
        environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
    )
    assert calls == []
    assert result["executor_relaunched"] is False


def test_zero_miss_executor_produces_no_gpu_work_but_opens_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, round_dir, artifacts = _fixture(tmp_path, physical_count=0)
    _patch_admission(monkeypatch, module, artifacts)
    calls = 0

    def runner(argv, **_kwargs):
        nonlocal calls
        calls += 1
        _write_execution_terminal(
            round_dir,
            {
                **_terminal(),
                "schema_version": "stage7_actual_v3_empty_physical_terminal_v2",
                "gpu_subprocess_count": 0,
            },
        )
        return subprocess.CompletedProcess(argv, 0)

    def finalizer(*_args, **_kwargs):
        _write(round_dir / "atomic_feedback_barrier.json", _barrier())
        return _barrier()

    monkeypatch.setattr(module, "validate_committed_barrier", lambda _path: _barrier())
    result = module.execute_round(
        v2_root=root,
        variant="full",
        seed=20260718,
        round_index=0,
        request_sha256=REQUEST_SHA,
        **PINS,
        subprocess_runner=runner,
        finalizer=finalizer,
        environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
    )
    assert calls == 1
    assert result["physical_row_count"] == 0
    assert result["gpu_work_expected"] is False
    assert result["status"] == "feedback_complete"


@pytest.mark.parametrize("mode", ("missing", "invalid"))
def test_missing_or_invalid_atomic_barrier_is_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    module = _load_module()
    root, round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    _write_execution_terminal(round_dir, _terminal())

    def finalizer(*_args, **_kwargs):
        if mode == "invalid":
            _write(round_dir / "atomic_feedback_barrier.json", {"bad": True})
        return {}

    monkeypatch.setattr(
        module,
        "validate_committed_barrier",
        lambda _path: (_ for _ in ()).throw(ValueError("barrier invalid")),
    )
    with pytest.raises(module.RetryableRoundError, match="feedback barrier"):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=lambda *_args, **_kwargs: None,
            finalizer=finalizer,
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )


@pytest.mark.parametrize(
    ("filename", "field", "value"),
    (
        ("logical_request.json", "measurement_request_sha256", "9" * 64),
        ("source_resolution_result.json", "logical_request_sha256", "9" * 64),
        ("exact_selection_binding.json", "logical_request", {"logical_request_sha256": "9" * 64}),
        ("executor_admission.json", "physical_request_sha256", "9" * 64),
    ),
)
def test_request_source_cache_lineage_tamper_fails_before_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    field: str,
    value: Any,
) -> None:
    module = _load_module()
    root, _round_dir, artifacts = _fixture(tmp_path)
    artifacts[filename][field] = value
    _patch_admission(monkeypatch, module, artifacts)
    calls: list[str] = []
    with pytest.raises(ValueError, match="lineage|request|admission"):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=lambda *_args, **_kwargs: calls.append("launched"),
            finalizer=lambda *_args, **_kwargs: {},
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )
    assert calls == []


def test_no_gpu_gate_failure_stops_before_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, _round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    monkeypatch.setattr(
        module,
        "_validate_no_gpu_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no-GPU drift")),
    )
    calls: list[str] = []
    with pytest.raises(ValueError, match="no-GPU"):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=lambda *_args, **_kwargs: calls.append("launched"),
            finalizer=lambda *_args, **_kwargs: {},
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )
    assert calls == []


def test_scheduler_uuid_or_request_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, _round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    state_path = root / "status/scheduler_state.json"
    state = json.loads(state_path.read_text())
    controller = state["controllers"]["full:seed_20260718:round_0"]
    controller["request_sha256"] = "9" * 64
    _write(state_path, state)
    with pytest.raises(ValueError, match="scheduler"):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=lambda *_args, **_kwargs: None,
            finalizer=lambda *_args, **_kwargs: {},
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )


def test_existing_authenticated_barrier_is_idempotent_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    _write(round_dir / "atomic_feedback_barrier.json", _barrier())
    monkeypatch.setattr(module, "validate_committed_barrier", lambda _path: _barrier())
    result = module.execute_round(
        v2_root=root,
        variant="full",
        seed=20260718,
        round_index=0,
        request_sha256=REQUEST_SHA,
        **PINS,
        subprocess_runner=lambda *_args, **_kwargs: pytest.fail("must not launch"),
        finalizer=lambda *_args, **_kwargs: pytest.fail("must not finalize"),
        environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
    )
    assert result["status"] == "feedback_complete"
    assert result["terminal_reused"] is True


def test_feedback_complete_scheduler_state_requires_authenticated_barrier_and_never_relaunches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, _round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    state_path = root / "status/scheduler_state.json"
    state = json.loads(state_path.read_text())
    state["controllers"]["full:seed_20260718:round_0"]["status"] = (
        "feedback_complete"
    )
    _write(state_path, state)
    calls: list[str] = []
    with pytest.raises(ValueError, match="feedback_complete.*barrier"):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=lambda *_args, **_kwargs: calls.append("launched"),
            finalizer=lambda *_args, **_kwargs: calls.append("finalized"),
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )
    assert calls == []


@pytest.mark.parametrize("failure", ("subprocess", "missing_terminal", "retry_terminal"))
def test_executor_infrastructure_or_evidence_failure_is_retryable_without_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = _load_module()
    root, round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)

    def runner(argv, **_kwargs):
        if failure == "subprocess":
            raise subprocess.CalledProcessError(1, argv)
        if failure == "retry_terminal":
            _write_execution_terminal(
                round_dir, {**_terminal(), "barrier_release_allowed": False}
            )
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(module.RetryableRoundError):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=runner,
            finalizer=lambda *_args, **_kwargs: pytest.fail("must not finalize"),
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )


def test_observer_reports_authenticated_pending_terminal_and_barrier_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, round_dir, _artifacts = _fixture(tmp_path)
    monkeypatch.setattr(module, "validate_physical_terminal_batch", lambda value: value)
    monkeypatch.setattr(module, "validate_committed_barrier", lambda _path: _barrier())
    pending = module.observe_round(root, "full", 20260718, 0, REQUEST_SHA)
    assert pending["state"] == "execution_pending"
    _write_execution_terminal(round_dir, _terminal())
    terminal = module.observe_round(root, "full", 20260718, 0, REQUEST_SHA)
    assert terminal["state"] == "terminal_ready"
    assert terminal["terminal_authenticated"] is True
    _write(round_dir / "atomic_feedback_barrier.json", _barrier())
    barrier = module.observe_round(root, "full", 20260718, 0, REQUEST_SHA)
    assert barrier["state"] == "feedback_complete"
    assert barrier["barrier_authenticated"] is True


def test_default_no_gpu_gate_delegates_to_authenticated_bundle_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, _round_dir, _artifacts = _fixture(tmp_path)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        module.deployment_bundle_v2,
        "validate_deployment_bundle",
        lambda actual_root, **kwargs: {
            "deployment_bundle_sha256": calls.append(
                ("deployment", (actual_root, kwargs))
            )
            or "6" * 64
        },
    )
    monkeypatch.setattr(
        module,
        "validate_no_gpu_receipt",
        lambda receipt, **kwargs: calls.append(("receipt", (receipt, kwargs)))
        or receipt,
    )
    result = module._validate_no_gpu_receipt(
        root,
        frozen_repo_root=tmp_path,
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
    )
    assert result["schema_version"] == "stage7_core_no_gpu_dry_run_v2"
    assert [name for name, _value in calls] == ["deployment", "receipt"]
    assert calls[0][1][1]["expected_release_sha256"] == RELEASE_PIN
    assert (
        calls[0][1][1]["expected_manifest_file_sha256"] == MANIFEST_FILE_PIN
    )


def test_cli_main_maps_success_retry_and_admission_to_stable_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    argv = [
        "--v2-root",
        str(tmp_path),
        "--variant",
        "full",
        "--seed",
        "20260718",
        "--round-index",
        "0",
        "--request-sha256",
        REQUEST_SHA,
        "--expected-release-sha256",
        RELEASE_PIN,
        "--expected-manifest-file-sha256",
        MANIFEST_FILE_PIN,
    ]
    monkeypatch.setattr(
        module, "execute_round", lambda **_kwargs: {"status": "feedback_complete"}
    )
    _patch_launch_gate(monkeypatch, module)
    assert module.main(argv) == 0
    assert "feedback_complete" in capsys.readouterr().out
    monkeypatch.setattr(
        module,
        "execute_round",
        lambda **_kwargs: (_ for _ in ()).throw(module.RetryableRoundError("retry")),
    )
    assert module.main(argv) == 1
    assert "selected_event_budget_consumed" in capsys.readouterr().err
    monkeypatch.setattr(
        module,
        "execute_round",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("drift")),
    )
    assert module.main(argv) == 2
    assert "stage7_round_admission_failed" in capsys.readouterr().err


def test_launch_gate_authenticates_exact_release_before_work(tmp_path: Path) -> None:
    module = _load_module()
    root, environment = _launch_gate(module, tmp_path)

    receipt = module._await_launch_gate(
        v2_root=root,
        variant="full",
        seed=20260718,
        round_index=0,
        request_sha256=REQUEST_SHA,
        environ=environment,
        pid_getter=lambda: 4321,
    )

    assert receipt.payload["controller_id"] == "full:seed_20260718:round_0"
    assert receipt.payload["gate_token_sha256"] == hashlib.sha256(
        environment["STAGE7_V2_LAUNCH_GATE_TOKEN"].encode("utf-8")
    ).hexdigest()


def test_launch_gate_rejects_same_user_external_status_tree(tmp_path: Path) -> None:
    module = _load_module()
    root, environment = _launch_gate(module, tmp_path, external_status=True)

    with pytest.raises(ValueError, match="canonical"):
        module._await_launch_gate(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            environ=environment,
            pid_getter=lambda: 4321,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("STAGE7_V2_LAUNCH_GATE_PATH", "", "path"),
        ("STAGE7_V2_LAUNCH_GATE_PATH", "relative/gate.json", "absolute"),
        ("STAGE7_V2_LAUNCH_GATE_TOKEN", "A" * 64, "token"),
        ("STAGE7_V2_LAUNCH_GATE_TIMEOUT_SECONDS", "30", "timeout"),
    ),
)
def test_launch_gate_env_fails_closed_before_work(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    module = _load_module()
    root, environment = _launch_gate(module, tmp_path)
    environment[field] = value

    with pytest.raises(ValueError, match=message):
        module._await_launch_gate(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            environ=environment,
            pid_getter=lambda: 4321,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("child_pid", 9999, "identity"),
        ("owner_uid", -1, "identity"),
        ("gate_token_sha256", "9" * 64, "identity"),
        ("scheduler_controller_record_sha256", "9" * 64, "state"),
        ("release_sha256", "9" * 64, "release"),
    ),
)
def test_launch_gate_rejects_forged_release(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    module = _load_module()
    root, environment = _launch_gate(module, tmp_path)
    gate_path = Path(environment["STAGE7_V2_LAUNCH_GATE_PATH"])
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload[field] = value
    if field != "release_sha256":
        unsigned = {
            key: item for key, item in payload.items() if key != "release_sha256"
        }
        payload["release_sha256"] = module._canonical_sha(unsigned)
    _write(gate_path, payload)
    gate_path.chmod(0o600)

    with pytest.raises(ValueError, match=message):
        module._await_launch_gate(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            environ=environment,
            pid_getter=lambda: 4321,
        )


def test_launch_gate_timeout_never_enters_work(tmp_path: Path) -> None:
    module = _load_module()
    root, environment = _launch_gate(module, tmp_path)
    Path(environment["STAGE7_V2_LAUNCH_GATE_PATH"]).unlink()
    ticks = iter((0.0, 61.0))

    with pytest.raises(TimeoutError, match="launch gate"):
        module._await_launch_gate(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            environ=environment,
            pid_getter=lambda: 4321,
            monotonic=lambda: next(ticks),
            sleeper=lambda _seconds: None,
        )


def test_launch_gate_rejects_file_shape_mode_and_symlink(
    tmp_path: Path,
) -> None:
    module = _load_module()
    root, environment = _launch_gate(module, tmp_path)
    gate_path = Path(environment["STAGE7_V2_LAUNCH_GATE_PATH"])
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    payload["release_sha256"] = module._canonical_sha(
        {key: value for key, value in payload.items() if key != "release_sha256"}
    )
    _write(gate_path, payload)
    gate_path.chmod(0o600)
    with pytest.raises(ValueError, match="shape"):
        module._await_launch_gate(
            v2_root=root, variant="full", seed=20260718, round_index=0,
            request_sha256=REQUEST_SHA, environ=environment,
        )
    gate_path.chmod(0o644)
    with pytest.raises(ValueError, match="mode"):
        module._await_launch_gate(
            v2_root=root, variant="full", seed=20260718, round_index=0,
            request_sha256=REQUEST_SHA, environ=environment,
        )
    target = gate_path.with_suffix(".target")
    gate_path.replace(target)
    gate_path.symlink_to(target)
    with pytest.raises(ValueError, match="file identity"):
        module._await_launch_gate(
            v2_root=root, variant="full", seed=20260718, round_index=0,
            request_sha256=REQUEST_SHA, environ=environment,
        )


def test_formal_execute_without_gate_fails_before_executor(tmp_path: Path) -> None:
    module = _load_module()
    calls: list[str] = []
    with pytest.raises(ValueError, match="launch gate path"):
        module.execute_round(
            v2_root=tmp_path,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=lambda *_args, **_kwargs: calls.append("executor"),
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )
    assert calls == []


def test_cli_gate_failure_reports_zero_gpu_and_zero_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    for name in (
        "STAGE7_V2_LAUNCH_GATE_PATH",
        "STAGE7_V2_LAUNCH_GATE_TOKEN",
        "STAGE7_V2_LAUNCH_GATE_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    calls: list[str] = []
    monkeypatch.setattr(
        module, "execute_round", lambda **_kwargs: calls.append("executor")
    )
    result = module.main([
        "--v2-root", str(tmp_path), "--variant", "full", "--seed", "20260718",
        "--round-index", "0", "--request-sha256", REQUEST_SHA,
        "--expected-release-sha256", RELEASE_PIN,
        "--expected-manifest-file-sha256", MANIFEST_FILE_PIN,
    ])
    failure = json.loads(capsys.readouterr().err)
    assert result == 2
    assert failure["gpu_jobs_launched"] == 0
    assert failure["selected_event_budget_consumed"] == 0
    assert calls == []


@pytest.mark.parametrize(
    ("release_pin", "manifest_pin"),
    (
        ("", MANIFEST_FILE_PIN),
        ("A" * 64, MANIFEST_FILE_PIN),
        ("9" * 63, MANIFEST_FILE_PIN),
        (RELEASE_PIN, ""),
        (RELEASE_PIN, "not-a-sha"),
    ),
)
def test_external_deployment_pins_fail_before_cuda_or_executor_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release_pin: str,
    manifest_pin: str,
) -> None:
    module = _load_module()
    _patch_launch_gate(monkeypatch, module)
    calls: list[str] = []
    with pytest.raises(ValueError, match="deployment .*pin"):
        module.execute_round(
            v2_root=tmp_path,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            expected_release_sha256=release_pin,
            expected_manifest_file_sha256=manifest_pin,
            subprocess_runner=lambda *_args, **_kwargs: calls.append("executor"),
            environ={"CUDA_VISIBLE_DEVICES": ""},
        )
    assert calls == []


def test_scheduler_external_deployment_pin_drift_fails_before_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root, _round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    state_path = root / "status/scheduler_state.json"
    state = json.loads(state_path.read_text())
    state["controllers"]["full:seed_20260718:round_0"][
        "expected_release_sha256"
    ] = "9" * 64
    _write(state_path, state)
    calls: list[str] = []
    with pytest.raises(ValueError, match="scheduler.*deployment pin"):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            subprocess_runner=lambda *_args, **_kwargs: calls.append("executor"),
            environ={"CUDA_VISIBLE_DEVICES": ",".join(UUIDS)},
        )
    assert calls == []


@pytest.mark.parametrize(
    "cuda",
    (
        "",
        "0,1,2,3",
        ",".join((UUIDS[0], UUIDS[0], UUIDS[2], UUIDS[3])),
        "GPU-a,GPU-b,GPU-c,GPU-d",
    ),
)
def test_cuda_visible_devices_requires_four_ordered_uuid_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda: str,
) -> None:
    module = _load_module()
    root, _round_dir, artifacts = _fixture(tmp_path)
    _patch_admission(monkeypatch, module, artifacts)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        module.execute_round(
            v2_root=root,
            variant="full",
            seed=20260718,
            round_index=0,
            request_sha256=REQUEST_SHA,
            **PINS,
            environ={"CUDA_VISIBLE_DEVICES": cuda},
        )


def test_cli_rejects_legacy_output_root_and_accepts_stable_argument_names() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            "/tmp/not-v2",
            "--variant",
            "full",
            "--seed",
            "20260718",
            "--round-index",
            "0",
            "--request-sha256",
            REQUEST_SHA,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 2
    assert "--v2-root" in result.stderr
