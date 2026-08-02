from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage7_core_ablation_scheduler_v2.py"
EXPECTED_RELEASE_SHA256 = "1" * 64
EXPECTED_MANIFEST_FILE_SHA256 = "2" * 64


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage7_core_ablation_scheduler_v2", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class FakeProcess:
    pid: int
    owner: str = "stage7-user"
    start_time: str = "2026-07-25T00:00:00Z"
    command: tuple[str, ...] = ()
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


class FakeLauncher:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, command, *, env, cwd):
        self.calls.append(
            {"command": tuple(command), "env": dict(env), "cwd": Path(cwd)}
        )
        return FakeProcess(pid=8100 + len(self.calls), command=tuple(command))


class FakeLock:
    def __init__(self, uuid: str, released: list[str]) -> None:
        self.uuid = uuid
        self.released = released

    def release(self) -> None:
        self.released.append(self.uuid)


class FakeLockFactory:
    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.released: list[str] = []

    def __call__(self, uuid: str) -> FakeLock:
        self.acquired.append(uuid)
        return FakeLock(uuid, self.released)


class SnapshotProbe:
    def __init__(self, snapshots: list[dict[str, object]]) -> None:
        self.snapshots = snapshots
        self.calls = 0

    def __call__(self):
        snapshot = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
        self.calls += 1
        return snapshot


def gpu_snapshots(module, count: int) -> dict[str, object]:
    return {
        f"GPU-{index}": module.GpuSnapshot(
            uuid=f"GPU-{index}",
            index=index,
            memory_used_mib=0,
            utilization_percent=0,
            compute_pids=(),
        )
        for index in range(count)
    }


def v2_root(module, tmp_path: Path) -> Path:
    root = (
        tmp_path / "stage7_pyramid_tvm_online_core_ablation_quick_v2_20260725"
    ).resolve()
    module.V2_ROOT = root
    module._requests.V2_ROOT = root

    def validate_fixture_artifacts(round_dir: Path):
        return {
            "physical_plan": json.loads(
                (round_dir / "miss_only_physical_request.json").read_text()
            ),
            "source_result": json.loads(
                (round_dir / "source_resolution_result.json").read_text()
            ),
        }

    module._requests.validate_measurement_artifacts = validate_fixture_artifacts
    return root


def v1_root(tmp_path: Path) -> Path:
    return (tmp_path / "stage7_pyramid_tvm_online_ablation_quick_v1_20260725").resolve()


def canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def request(
    module,
    root: Path,
    trajectory: str,
    *,
    round_index: int = 0,
    width=(16, 32, 64),
    source_identity: str | None = None,
    request_sha_override: str | None = None,
    expected_release_sha256: str = EXPECTED_RELEASE_SHA256,
    expected_manifest_file_sha256: str = EXPECTED_MANIFEST_FILE_SHA256,
):
    variant, raw_seed = trajectory.split(":seed_", 1)
    source_identity = source_identity or (
        f"{variant}:{raw_seed}:{round_index}:{'x'.join(map(str, width))}"
    )
    request_sha = request_sha_override or (("a" if round_index == 0 else "b") * 64)
    round_dir = (
        root / "variants" / variant / f"seed_{raw_seed}" / f"round_{round_index:02d}"
    )
    rows = [
        {
            "row_id": f"row-{index}",
            "manifest_job_id": f"row-{index}",
            "source_contract": {
                "checkpoint_sha256": hashlib.sha256(
                    f"{source_identity}:checkpoint".encode()
                ).hexdigest(),
                "onnx_sha256": hashlib.sha256(
                    f"{source_identity}:onnx".encode()
                ).hexdigest(),
            },
            "source_evidence_sha256": hashlib.sha256(
                f"{source_identity}:materialization".encode()
            ).hexdigest(),
        }
        for index in range(4)
    ]
    unsigned_plan = {
        "schema_version": "stage7_actual_v3_miss_only_physical_request_v2",
        "logical_request_sha256": request_sha,
        "logical_row_count": 4,
        "physical_row_count": 4,
        "rows": rows,
        "logical_row_bindings": [
            {
                "logical_row_index": index,
                "candidate_id": f"row-{index}",
                "logical_row_sha256": hashlib.sha256(
                    f"{trajectory}:{round_index}:row-{index}".encode()
                ).hexdigest(),
                "disposition": "miss",
            }
            for index in range(4)
        ],
    }
    physical_request_sha = canonical_sha(
        {
            key: value
            for key, value in unsigned_plan.items()
            if key != "logical_row_bindings"
        }
    )
    plan = {
        **unsigned_plan,
        "physical_request_sha256": physical_request_sha,
    }
    round_dir.mkdir(parents=True, exist_ok=True)
    plan_path = round_dir / "miss_only_physical_request.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n")
    source_plan = {
        "schema_version": "stage7_source_resolution_plan_v2",
        "logical_request_sha256": request_sha,
        "source_resolution_plan_sha256": hashlib.sha256(
            f"{trajectory}:{round_index}:source-plan".encode()
        ).hexdigest(),
    }
    formal_rows = [
        {
            "candidate_id": f"row-{index}",
            "logical_row_sha256": plan["logical_row_bindings"][index][
                "logical_row_sha256"
            ],
            "resolved_source_sha256": hashlib.sha256(
                f"{source_identity}:resolved:{index}".encode()
            ).hexdigest(),
            "checkpoint_sha256": rows[index]["source_contract"]["checkpoint_sha256"],
            "onnx_sha256": rows[index]["source_contract"]["onnx_sha256"],
        }
        for index in range(4)
    ]
    formal_unsigned = {
        "schema_version": "stage7_source_resolution_result_v2",
        "logical_request_sha256": request_sha,
        "source_resolution_plan_sha256": source_plan["source_resolution_plan_sha256"],
        "rows": formal_rows,
    }
    formal_result = {
        **formal_unsigned,
        "source_resolution_result_sha256": canonical_sha(formal_unsigned),
    }
    artifacts = {
        "logical_request.json": {"measurement_request_sha256": request_sha},
        "source_resolution_plan.json": source_plan,
        "source_resolution_result.json": formal_result,
        "exact_selection_binding.json": {"status": "frozen"},
        "cache_snapshot_before_reveal.json": {"entries": {}},
        "cache_reveal.json": {"status": "revealed"},
        "executor_admission.json": {"status": "admitted"},
    }
    for name, payload in artifacts.items():
        (round_dir / name).write_text(json.dumps(payload, sort_keys=True) + "\n")
    source_lock_key = module._requests.resolved_source_lock_sha256(plan, formal_result)
    return module.V2BatchRequest(
        trajectory_id=trajectory,
        trajectory_path=round_dir,
        round_index=round_index,
        request_sha256=request_sha,
        selected_row_ids=("row-0", "row-1", "row-2", "row-3"),
        command=(
            str(module.CORE_CONTROLLER),
            "--v2-root",
            str(root),
            "--variant",
            variant,
            "--seed",
            raw_seed,
            "--round-index",
            str(round_index),
            "--request-sha256",
            request_sha,
            "--expected-release-sha256",
            expected_release_sha256,
            "--expected-manifest-file-sha256",
            expected_manifest_file_sha256,
        ),
        width=width,
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        physical_plan_path=plan_path,
        physical_request_sha256=physical_request_sha,
        source_lock_key=source_lock_key,
    )


def test_measurement_request_rejects_source_state_before_formal_cache_barrier(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    complete = request(module, root, "full:seed_20260718")
    (complete.trajectory_path / "cache_reveal.json").unlink()

    with pytest.raises(ValueError, match="SOURCE_READY"):
        replace(complete)


def test_measurement_source_lock_comes_from_formal_resolved_source_rows(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    batch = request(module, root, "full:seed_20260718")
    plan = json.loads(batch.physical_plan_path.read_text())
    formal = json.loads(
        (batch.trajectory_path / "source_resolution_result.json").read_text()
    )

    assert batch.source_lock_key == module._requests.resolved_source_lock_sha256(
        plan, formal
    )


def test_scheduler_source_prelease_fast_path_cannot_touch_lease_or_child(
    tmp_path: Path,
) -> None:
    module = load_module()
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 4)], launcher)
    calls: list[str] = []

    result = subject.advance_source_prelease(
        validate_ready_evidence=lambda: {
            "schema_version": "stage7_source_resolution_result_v2",
            "row_count": 4,
            "eligible_for_exact_cache_reveal": True,
            "source_resolution_result_sha256": "a" * 64,
        },
        lease_factory=lambda: calls.append("lease"),
        launch_wrapper=lambda _lease: calls.append("child"),
    )

    assert result["state"] == "SOURCE_READY"
    assert calls == []
    assert launcher.calls == []


def test_measurement_request_calls_strong_artifact_validator_before_lease(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    complete = request(module, root, "full:seed_20260718")

    def reject(_round_dir):
        raise ValueError("strong formal validation rejected")

    module._requests.validate_measurement_artifacts = reject
    with pytest.raises(ValueError, match="strong formal validation rejected"):
        replace(complete)


def scheduler(
    module,
    tmp_path: Path,
    snapshots: list[dict[str, object]],
    launcher: FakeLauncher,
    *,
    completion_validator=None,
):
    kwargs = {}
    if completion_validator is not None:
        kwargs["completion_validator"] = completion_validator
    return module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe(snapshots),
        lock_factory=FakeLockFactory(),
        launcher=launcher,
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        wall_time=lambda: 1_753_392_000.0,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {
            uuid: module.EXPECTED_GPU_MODEL for uuid in snapshots[0]
        },
        reservation_probe=lambda: {},
        process_scan=lambda: (),
        resource_lock_factory=FakeLockFactory(),
        **kwargs,
    )


def completion_receipt(batch, **updates):
    return {
        "schema_version": "stage7_actual_v3_atomic_feedback_barrier_v2",
        "logical_request_sha256": batch.request_sha256,
        "feedback_released": True,
        "budget_consumed": 4,
        "physical_terminal_batch_sha256": "c" * 64,
        "cache_after_round_file_sha256": "d" * 64,
        "barrier_receipt_sha256": "e" * 64,
        **updates,
    }


def with_deployment_pins(batch):
    return batch


def test_batch_request_requires_external_deployment_pins_in_stable_worker_argv(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    batch = with_deployment_pins(
        request(module, root, "full:seed_20260718")
    )

    assert batch.expected_release_sha256 == EXPECTED_RELEASE_SHA256
    assert batch.expected_manifest_file_sha256 == EXPECTED_MANIFEST_FILE_SHA256
    assert batch.command[-4:] == (
        "--expected-release-sha256",
        EXPECTED_RELEASE_SHA256,
        "--expected-manifest-file-sha256",
        EXPECTED_MANIFEST_FILE_SHA256,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_release_sha256", ""),
        ("expected_release_sha256", "A" * 64),
        ("expected_manifest_file_sha256", "short"),
        ("expected_manifest_file_sha256", "g" * 64),
    ),
)
def test_batch_request_rejects_missing_or_noncanonical_deployment_pin_before_lease(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    batch = with_deployment_pins(
        request(module, root, "full:seed_20260718")
    )

    with pytest.raises(ValueError, match="deployment pin"):
        replace(batch, **{field: value})


def test_scheduler_controller_record_and_recovery_bind_external_deployment_pins(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(
        module, tmp_path, [gpu_snapshots(module, 4)] * 4, launcher
    )
    subject.observe_gpus()
    batch = with_deployment_pins(
        request(module, root, "full:seed_20260718")
    )

    subject.schedule([batch])

    controller = subject.status()["controllers"][batch.controller_id]
    assert controller["expected_release_sha256"] == EXPECTED_RELEASE_SHA256
    assert (
        controller["expected_manifest_file_sha256"]
        == EXPECTED_MANIFEST_FILE_SHA256
    )
    controllers = dict(subject._state["controllers"])
    controllers[batch.controller_id] = {
        **controller,
        "expected_release_sha256": "3" * 64,
        "status": "infrastructure_retry_required",
    }
    subject._persist(controllers=controllers)
    acquired_before = len(subject.lock_factory.acquired)

    with pytest.raises(ValueError, match="deployment pin drift"):
        subject.schedule([batch])
    assert len(subject.lock_factory.acquired) == acquired_before
    assert len(launcher.calls) == 1


def test_complete_v2_controller_record_is_durable_before_worker_launch(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    observed: list[dict[str, object]] = []
    subject = None

    def launcher(command, *, env, cwd):
        state = json.loads(subject.state_path.read_text(encoding="utf-8"))
        observed.append(dict(state["controllers"][batch.controller_id]))
        return FakeProcess(pid=8123, command=tuple(command))

    subject = scheduler(
        module, tmp_path, [gpu_snapshots(module, 4)] * 4, launcher
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")

    subject.schedule([batch])

    assert len(observed) == 1
    assert observed[0]["status"] == "running"
    assert observed[0]["physical_request_sha256"] == batch.physical_request_sha256
    assert observed[0]["source_resolution_result_sha256"] == (
        batch.source_resolution_result_sha256
    )
    assert observed[0]["gpu_models"] == {
        f"GPU-{index}": module.EXPECTED_GPU_MODEL for index in range(4)
    }
    assert observed[0]["expected_release_sha256"] == EXPECTED_RELEASE_SHA256
    assert (
        observed[0]["expected_manifest_file_sha256"]
        == EXPECTED_MANIFEST_FILE_SHA256
    )


def test_launch_gate_releases_only_after_real_pid_record_is_durable(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    observed: list[tuple[Path, bool, dict[str, str]]] = []
    subject = None

    def launcher(command, *, env, cwd):
        gate_path = Path(env["STAGE7_V2_LAUNCH_GATE_PATH"])
        observed.append((gate_path, gate_path.exists(), dict(env)))
        return FakeProcess(pid=8125, command=tuple(command))

    subject = scheduler(
        module, tmp_path, [gpu_snapshots(module, 4)] * 4, launcher
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")

    subject.schedule([batch])

    gate_path, existed_at_launch, child_env = observed[0]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    controller = subject.status()["controllers"][batch.controller_id]
    assert existed_at_launch is False
    assert gate["child_pid"] == 8125
    assert gate["request_sha256"] == batch.request_sha256
    assert gate["scheduler_controller_record_sha256"] == canonical_sha(controller)
    assert controller["pid"] == 8125
    from scripts import stage7_core_round_worker_v2 as worker

    authenticated = worker._await_launch_gate(
        v2_root=root,
        variant="full",
        seed=20260718,
        round_index=0,
        request_sha256=batch.request_sha256,
        environ=child_env,
        pid_getter=lambda: 8125,
    )
    assert authenticated.payload == gate


def test_prelaunch_controller_persist_failure_never_starts_worker(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(
        module, tmp_path, [gpu_snapshots(module, 4)] * 4, launcher
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")

    def fail_persist(**_updates):
        raise OSError("atomic controller persistence failed")

    subject._persist = fail_persist
    with pytest.raises(OSError, match="controller persistence"):
        subject.schedule([batch])

    assert launcher.calls == []
    assert subject.lock_factory.released == [f"GPU-{index}" for index in range(4)]


def test_noncanonical_scheduler_status_root_fails_before_any_lock_or_launch(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = module.V2Stage7Scheduler(
        status_dir=tmp_path / "outside_status",
        gpu_probe=SnapshotProbe([gpu_snapshots(module, 4)] * 4),
        lock_factory=FakeLockFactory(),
        resource_lock_factory=FakeLockFactory(),
        launcher=launcher,
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {
            f"GPU-{index}": module.EXPECTED_GPU_MODEL for index in range(4)
        },
        reservation_probe=lambda: {},
        process_scan=lambda: (),
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")

    with pytest.raises(ValueError, match="canonical v2 status"):
        subject.schedule([batch])

    assert subject.lock_factory.acquired == []
    assert subject.resource_lock_factory.acquired == []
    assert launcher.calls == []


def test_postlaunch_controller_persist_failure_kills_child_before_unlock(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    events: list[str] = []

    class Child(FakeProcess):
        def terminate(self):
            events.append("terminate")
            self.returncode = -15

        def kill(self):
            events.append("kill")
            self.returncode = -9

        def wait(self, timeout=None):
            events.append(f"wait:{timeout}")
            return self.returncode

    class Launcher(FakeLauncher):
        def __call__(self, command, *, env, cwd):
            super().__call__(command, env=env, cwd=cwd)
            events.append("launch")
            self.child = Child(pid=8124, command=tuple(command))
            return self.child

    class OrderedLock(FakeLock):
        def release(self):
            events.append(f"release:{self.uuid}")
            super().release()

    class OrderedLocks(FakeLockFactory):
        def __call__(self, uuid):
            self.acquired.append(uuid)
            return OrderedLock(uuid, self.released)

    launcher = Launcher()
    subject = scheduler(
        module, tmp_path, [gpu_snapshots(module, 4)] * 4, launcher
    )
    subject.lock_factory = OrderedLocks()
    subject.resource_lock_factory = OrderedLocks()
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")
    persist = subject._persist
    persist_calls = 0

    def fail_real_pid_persist(**updates):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            events.append("persist-real-pid-failed")
            raise OSError("real PID controller persistence failed")
        return persist(**updates)

    subject._persist = fail_real_pid_persist
    with pytest.raises(OSError, match="real PID controller persistence"):
        subject.schedule([batch])

    assert launcher.child.poll() is not None
    assert events.index("terminate") < events.index("wait:10.0")
    assert events.index("wait:10.0") < next(
        index for index, event in enumerate(events) if event.startswith("release:")
    )


def test_unconfirmed_postlaunch_child_death_retains_all_locks(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)

    class UnstoppableChild(FakeProcess):
        def terminate(self):
            pass

        def kill(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(self.command, timeout)

    class Launcher(FakeLauncher):
        def __call__(self, command, *, env, cwd):
            super().__call__(command, env=env, cwd=cwd)
            self.child = UnstoppableChild(pid=8126, command=tuple(command))
            return self.child

    launcher = Launcher()
    subject = scheduler(
        module, tmp_path, [gpu_snapshots(module, 4)] * 4, launcher
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")
    persist = subject._persist
    persist_calls = 0

    def fail_real_pid_persist(**updates):
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise OSError("real PID controller persistence failed")
        return persist(**updates)

    subject._persist = fail_real_pid_persist
    with pytest.raises(
        module._requests.LaunchTerminationUnconfirmed,
        match="retaining scheduler locks",
    ):
        subject.schedule([batch])

    assert launcher.child.poll() is None
    assert subject.lock_factory.released == []
    assert subject.resource_lock_factory.released == []


def test_returncode_zero_without_authenticated_barrier_consumes_no_budget_and_retries(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    validator_calls: list[Path] = []

    def reject_missing_barrier(round_dir: Path):
        validator_calls.append(round_dir)
        raise ValueError("atomic feedback barrier is missing")

    subject = scheduler(
        module,
        tmp_path,
        [gpu_snapshots(module, 4)] * 5,
        launcher,
        completion_validator=reject_missing_barrier,
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")
    subject.schedule([batch])
    launcher_process = subject._processes[batch.controller_id]
    launcher_process.returncode = 0

    assert subject.reap_controllers() == []
    status = subject.status()
    controller = status["controllers"][batch.controller_id]
    assert validator_calls == [batch.trajectory_path]
    assert controller["status"] == "infrastructure_retry_required"
    assert controller["retry_request_sha256"] == batch.request_sha256
    assert controller["selected_event_budget_consumed"] == 0
    assert status["selected_event_budget_consumed"] == 0
    assert batch.controller_id not in subject._locks
    assert batch.controller_id not in subject._processes

    subject.observe_gpus()
    assert subject.schedule([batch])["launched"] == [batch]
    assert len(launcher.calls) == 2


def test_completion_validator_runtime_failure_is_recoverable_and_zero_budget(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()

    def fail_runtime(_round_dir: Path):
        raise RuntimeError("overlay validator load drift")

    subject = scheduler(
        module,
        tmp_path,
        [gpu_snapshots(module, 4)] * 4,
        launcher,
        completion_validator=fail_runtime,
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")
    subject.schedule([batch])
    subject._processes[batch.controller_id].returncode = 0

    assert subject.reap_controllers() == []
    state = subject.status()
    assert state["selected_event_budget_consumed"] == 0
    assert (
        state["controllers"][batch.controller_id]["status"]
        == "infrastructure_retry_required"
    )


def test_authenticated_barrier_consumes_exactly_four_once_and_opens_feedback(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    calls: list[Path] = []
    batch = request(module, root, "full:seed_20260718")

    def accept(round_dir: Path):
        calls.append(round_dir)
        return completion_receipt(batch)

    subject = scheduler(
        module,
        tmp_path,
        [gpu_snapshots(module, 4)] * 4,
        launcher,
        completion_validator=accept,
    )
    subject.observe_gpus()
    subject.schedule([batch])
    gate_path = Path(
        subject.status()["controllers"][batch.controller_id]["launch_gate_path"]
    )
    assert gate_path.is_file()
    subject._processes[batch.controller_id].returncode = 0

    assert subject.reap_controllers() == [batch.controller_id]
    assert not gate_path.exists()
    first = subject.status()
    assert first["selected_event_budget_consumed"] == 4
    assert first["controllers"][batch.controller_id]["status"] == "feedback_complete"
    assert (
        first["controllers"][batch.controller_id]["selected_event_budget_consumed"]
        == 4
    )
    completion_lineage = first["controllers"][batch.controller_id][
        "completion_lineage"
    ]
    assert completion_lineage == {
        "logical_request_sha256": batch.request_sha256,
        "physical_request_sha256": batch.physical_request_sha256,
        "source_resolution_result_sha256": (
            batch.source_resolution_result_sha256
        ),
        "selected_row_ids": list(batch.selected_row_ids),
        "cache_after_round_file_sha256": "d" * 64,
        "barrier_receipt_sha256": "e" * 64,
        "completion_lineage_sha256": completion_lineage[
            "completion_lineage_sha256"
        ],
    }
    unsigned_lineage = {
        key: value
        for key, value in completion_lineage.items()
        if key != "completion_lineage_sha256"
    }
    assert completion_lineage["completion_lineage_sha256"] == canonical_sha(
        unsigned_lineage
    )
    assert calls == [batch.trajectory_path]

    assert subject.reap_controllers() == []
    replay = subject.status()
    assert replay["selected_event_budget_consumed"] == 4
    assert calls == [batch.trajectory_path]


@pytest.mark.parametrize(
    "barrier_updates,state_updates",
    (
        ({"logical_request_sha256": "f" * 64}, {}),
        ({}, {"round_index": 1}),
    ),
)
def test_completion_admission_rejects_wrong_request_or_round_without_budget(
    tmp_path: Path,
    barrier_updates: dict[str, object],
    state_updates: dict[str, object],
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    batch = request(module, root, "full:seed_20260718")
    subject = scheduler(
        module,
        tmp_path,
        [gpu_snapshots(module, 4)] * 4,
        launcher,
        completion_validator=lambda _path: completion_receipt(
            batch, **barrier_updates
        ),
    )
    subject.observe_gpus()
    subject.schedule([batch])
    if state_updates:
        controllers = dict(subject._state["controllers"])
        controllers[batch.controller_id] = {
            **controllers[batch.controller_id],
            **state_updates,
        }
        subject._persist(controllers=controllers)
    subject._processes[batch.controller_id].returncode = 0

    assert subject.reap_controllers() == []
    state = subject.status()
    assert state["selected_event_budget_consumed"] == 0
    assert (
        state["controllers"][batch.controller_id]["status"]
        == "infrastructure_retry_required"
    )


def test_vanished_controller_releases_locks_and_recovers_same_request_without_budget(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    validator_calls: list[Path] = []
    subject = scheduler(
        module,
        tmp_path,
        [gpu_snapshots(module, 4)] * 4,
        launcher,
        completion_validator=lambda path: validator_calls.append(path),
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")
    subject.schedule([batch])
    subject._processes = {}

    assert subject.reap_controllers() == []
    state = subject.status()
    controller = state["controllers"][batch.controller_id]
    assert controller["status"] == "infrastructure_retry_required"
    assert controller["retry_request_sha256"] == batch.request_sha256
    assert controller["selected_event_budget_consumed"] == 0
    assert state["selected_event_budget_consumed"] == 0
    assert batch.controller_id not in subject._locks
    assert validator_calls == []


def test_nonzero_controller_exit_releases_locks_and_retries_same_request_without_budget(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    validator_calls: list[Path] = []
    subject = scheduler(
        module,
        tmp_path,
        [gpu_snapshots(module, 4)] * 5,
        launcher,
        completion_validator=lambda path: validator_calls.append(path),
    )
    subject.observe_gpus()
    batch = request(module, root, "full:seed_20260718")
    subject.schedule([batch])
    subject._processes[batch.controller_id].returncode = 23

    assert subject.reap_controllers() == []
    state = subject.status()
    controller = state["controllers"][batch.controller_id]
    assert controller["status"] == "infrastructure_retry_required"
    assert controller["retry_request_sha256"] == batch.request_sha256
    assert controller["selected_event_budget_consumed"] == 0
    assert state["selected_event_budget_consumed"] == 0
    assert batch.controller_id not in subject._locks
    assert validator_calls == []

    subject.observe_gpus()
    assert subject.schedule([batch])["launched"] == [batch]


def test_v2_queue_has_exactly_twelve_core_trajectories_and_no_a4(tmp_path: Path):
    module = load_module()
    root = v2_root(module, tmp_path)
    queue = module.build_v2_trajectory_queue(root)

    assert len(queue) == 12
    assert {item.variant for item in queue} == set(module.CORE_VARIANTS)
    assert "without_capability_scan" not in {item.variant for item in queue}
    assert {item.batch_size for item in queue} == {4}
    assert len({item.path for item in queue}) == 12
    assert queue[0].trajectory_id == "full:seed_20260718"


def test_serialized_batch_loader_preserves_stable_round_controller_identity(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = v2_root(module, tmp_path)
    batch = request(module, root, "backend_blind:seed_20260719", round_index=2)
    payload = {
        "trajectory_id": batch.trajectory_id,
        "trajectory_path": str(batch.trajectory_path),
        "round_index": batch.round_index,
        "request_sha256": batch.request_sha256,
        "selected_row_ids": list(batch.selected_row_ids),
        "command": list(batch.command),
        "width": list(batch.width),
        "expected_release_sha256": batch.expected_release_sha256,
        "expected_manifest_file_sha256": batch.expected_manifest_file_sha256,
        "task_id": batch.task_id,
        "source_lock_key": batch.source_lock_key,
        "physical_plan_path": str(batch.physical_plan_path),
        "physical_request_sha256": batch.physical_request_sha256,
    }

    rebuilt = module._batch_from_json(payload)

    assert rebuilt == batch
    assert rebuilt.command[1] == "--v2-root"


def test_default_completion_validator_resolves_from_active_deployment_overlay(
    tmp_path: Path,
) -> None:
    module = load_module()
    overlay_scripts = tmp_path / "deployment" / "code" / "scripts"
    overlay_scripts.mkdir(parents=True)
    overlay_scheduler = overlay_scripts / "stage7_core_ablation_scheduler_v2.py"
    overlay_scheduler.write_text("# overlay scheduler marker\n", encoding="utf-8")
    overlay_core = overlay_scripts / "stage7_core_online_ablation_v2.py"
    overlay_core.write_text(
        "def validate_committed_barrier(round_dir):\n"
        "    return {'loaded_from': __file__, 'round_dir': str(round_dir)}\n",
        encoding="utf-8",
    )
    module.__file__ = str(overlay_scheduler)

    receipt = module._default_completion_validator(tmp_path / "round_00")

    assert receipt == {
        "loaded_from": str(overlay_core),
        "round_dir": str(tmp_path / "round_00"),
    }


def test_default_completion_validator_reloads_when_overlay_bytes_change(
    tmp_path: Path,
) -> None:
    module = load_module()
    overlay_scripts = tmp_path / "deployment" / "code" / "scripts"
    overlay_scripts.mkdir(parents=True)
    overlay_scheduler = overlay_scripts / "stage7_core_ablation_scheduler_v2.py"
    overlay_scheduler.write_text("# overlay scheduler marker\n", encoding="utf-8")
    overlay_core = overlay_scripts / "stage7_core_online_ablation_v2.py"
    module.__file__ = str(overlay_scheduler)
    overlay_core.write_text(
        "def validate_committed_barrier(round_dir):\n"
        "    return {'generation': 'first-generation'}\n",
        encoding="utf-8",
    )
    first = module._default_completion_validator(tmp_path / "round_00")
    overlay_core.write_text(
        "def validate_committed_barrier(round_dir):\n"
        "    return {'generation': 'next--generation'}\n",
        encoding="utf-8",
    )

    second = module._default_completion_validator(tmp_path / "round_00")

    assert first == {"generation": "first-generation"}
    assert second == {"generation": "next--generation"}


def test_identical_validator_bytes_do_not_cross_deployment_overlay_paths(
    tmp_path: Path,
) -> None:
    module = load_module()
    source = (
        "def validate_committed_barrier(round_dir):\n"
        "    return {'loaded_from': __file__}\n"
    )
    loaded_from: list[str] = []
    for name in ("deployment-a", "deployment-b"):
        scripts = tmp_path / name / "code" / "scripts"
        scripts.mkdir(parents=True)
        scheduler_path = scripts / "stage7_core_ablation_scheduler_v2.py"
        scheduler_path.write_text("# scheduler marker\n", encoding="utf-8")
        validator_path = scripts / "stage7_core_online_ablation_v2.py"
        validator_path.write_text(source, encoding="utf-8")
        module.__file__ = str(scheduler_path)
        loaded_from.append(
            module._default_completion_validator(tmp_path / "round_00")[
                "loaded_from"
            ]
        )

    assert loaded_from == [
        str(
            tmp_path
            / "deployment-a/code/scripts/stage7_core_online_ablation_v2.py"
        ),
        str(
            tmp_path
            / "deployment-b/code/scripts/stage7_core_online_ablation_v2.py"
        ),
    ]


def test_gpu_model_probe_parses_uuid_inventory_and_rejects_malformed_rows():
    module = load_module()

    class Result:
        stdout = "GPU-a, NVIDIA H800\nGPU-b, NVIDIA H800\n"

    assert module.query_gpu_models(lambda *args, **kwargs: Result()) == {
        "GPU-a": "NVIDIA H800",
        "GPU-b": "NVIDIA H800",
    }

    class Malformed:
        stdout = "GPU-a\n"

    with pytest.raises(ValueError, match="malformed"):
        module.query_gpu_models(lambda *args, **kwargs: Malformed())


def test_process_audit_redacts_secret_values_without_hiding_ordinary_argv():
    module = load_module()

    assert module._redacted_command(
        ("python", "job.py", "--token", "secret", "--gpu", "7")
    ) == ["python", "job.py", "--token", "<redacted>", "--gpu", "7"]


def test_production_config_constructor_keeps_live_default_reservation_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = load_module()
    captured: dict[str, object] = {}

    def build(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "V2Stage7Scheduler", build)
    module._scheduler_from_config({"status_dir": str(tmp_path / "status")})

    assert "reservation_probe" not in captured
    assert "process_scan" not in captured
    assert "hostname_probe" not in captured
    assert "gpu_model_probe" not in captured
    assert set(module.build_parser()._subparsers._group_actions[0].choices) == {
        "preflight",
        "run",
        "resume",
        "status",
    }


def test_v2_queue_refuses_v1_or_non_v2_root(tmp_path: Path):
    module = load_module()

    with pytest.raises(ValueError, match="v2 result root"):
        module.build_v2_trajectory_queue(v1_root(tmp_path))
    with pytest.raises(ValueError, match="v2 result root"):
        module.build_v2_trajectory_queue(tmp_path / "other")


def test_v2_scheduler_waits_with_fewer_than_four_idle_uuids(tmp_path: Path):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 3)], launcher)
    subject.observe_gpus()

    result = subject.schedule([request(module, root, "full:seed_20260718")])

    assert result["status"] == "waiting_for_four_idle_gpus"
    assert result["launched"] == []
    assert launcher.calls == []


def test_v2_scheduler_keeps_four_gpu_requests_indivisible_with_gpu7_excluded(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 8)] * 3, launcher)
    subject.observe_gpus()
    requests = [
        request(module, root, "full:seed_20260718", width=(16, 32, 64)),
        request(module, root, "without_surrogate:seed_20260718", width=(24, 48, 96)),
        request(module, root, "backend_blind:seed_20260718", width=(32, 64, 128)),
    ]

    result = subject.schedule(requests)

    assert len(result["launched"]) == 1
    assert len(launcher.calls) == 1
    assert [
        call["env"]["CUDA_VISIBLE_DEVICES"].count(",") for call in launcher.calls
    ] == [3]
    assert result["blocked"] == [
        {
            "trajectory_id": "without_surrogate:seed_20260718",
            "reason": "capacity",
        },
        {"trajectory_id": "backend_blind:seed_20260718", "reason": "capacity"}
    ]


def test_v2_scheduler_never_prefetches_across_feedback_barrier(tmp_path: Path):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 8)] * 3, launcher)
    subject.observe_gpus()
    requests = [
        request(module, root, "full:seed_20260718", round_index=0),
        request(module, root, "full:seed_20260718", round_index=1),
    ]

    result = subject.schedule(requests)

    assert len(result["launched"]) == 1
    assert result["blocked"] == [
        {
            "trajectory_id": "full:seed_20260718",
            "round_index": 1,
            "reason": "feedback_barrier",
        }
    ]


def test_v2_scheduler_same_width_lock_prevents_concurrent_width_collision(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 8)] * 3, launcher)
    subject.observe_gpus()
    first = request(module, root, "full:seed_20260718", width=(16, 32, 64))
    same_width = request(
        module, root, "backend_blind:seed_20260718", width=(16, 32, 64)
    )

    result = subject.schedule([first, same_width])

    assert [item.trajectory_id for item in result["launched"]] == [first.trajectory_id]
    assert result["blocked"] == [
        {"trajectory_id": same_width.trajectory_id, "reason": "same_width_lock"}
    ]


def test_v2_scheduler_resumes_with_the_same_request_sha(tmp_path: Path):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    idle = gpu_snapshots(module, 4)
    busy = {
        uuid: module.GpuSnapshot(
            uuid=uuid,
            index=index,
            memory_used_mib=4096,
            utilization_percent=90,
            compute_pids=(9999,),
        )
        for index, uuid in enumerate(idle)
    }
    subject = scheduler(module, tmp_path, [idle, idle, idle, busy], launcher)
    subject.observe_gpus()
    original = request(module, root, "full:seed_20260718")
    assert len(subject.schedule([original])["launched"]) == 1
    retry = subject.monitor_occupancy()

    assert len(retry) == 1
    rebuilt = retry[0]
    assert rebuilt.request_sha256 == original.request_sha256
    assert rebuilt.selected_row_ids == original.selected_row_ids
    assert (
        subject.status()["controllers"][original.controller_id]["status"] == "running"
    )
    assert (
        subject.status()["controllers"][original.controller_id][
            "admission_blocked_while_process_alive"
        ]
        is True
    )
    assert len(subject.schedule([rebuilt])["launched"]) == 0
    assert len(launcher.calls) == 1


def test_preflight_cannot_bypass_host_admission(tmp_path: Path):
    module = load_module()
    idle = gpu_snapshots(module, 4)
    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle]),
        lock_factory=FakeLockFactory(),
        resource_lock_factory=FakeLockFactory(),
        launcher=FakeLauncher(),
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: "wrong-host",
        gpu_model_probe=lambda: {uuid: module.EXPECTED_GPU_MODEL for uuid in idle},
        reservation_probe=lambda: {},
        process_scan=lambda: (),
    )

    with pytest.raises(ValueError, match="hostname"):
        subject.preflight()


def test_preflight_reports_only_exact_model_unreserved_uuids(
    tmp_path: Path,
):
    module = load_module()
    idle = gpu_snapshots(module, 5)
    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle]),
        lock_factory=FakeLockFactory(),
        resource_lock_factory=FakeLockFactory(),
        launcher=FakeLauncher(),
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {uuid: module.EXPECTED_GPU_MODEL for uuid in idle},
        reservation_probe=lambda: {"GPU-4": ("waiting-task",)},
        process_scan=lambda: (),
    )

    result = subject.preflight()

    assert result == {
        "scheduler_status": "four_or_more_idle_gpus_observed",
        "idle_gpu_uuids": ["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
    }


@pytest.mark.parametrize(
    "invalid_model",
    ("NVIDIA A100-SXM4-80GB", "NVIDIA H800 unreviewed-custom"),
)
def test_runtime_admission_fails_closed_for_wrong_host_or_gpu_model(
    tmp_path: Path, invalid_model: str
):
    module = load_module()
    root = v2_root(module, tmp_path)
    snapshots = [gpu_snapshots(module, 4)]

    wrong_host = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe(snapshots),
        lock_factory=FakeLockFactory(),
        launcher=FakeLauncher(),
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: "not-the-h800-host",
        gpu_model_probe=lambda: {
            uuid: module.EXPECTED_GPU_MODEL for uuid in snapshots[0]
        },
        reservation_probe=lambda: {},
        process_scan=lambda: (),
        resource_lock_factory=FakeLockFactory(),
    )
    wrong_host.observe_gpus()
    with pytest.raises(ValueError, match="hostname"):
        wrong_host.schedule([request(module, root, "full:seed_20260718")])

    wrong_model = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe(snapshots),
        lock_factory=FakeLockFactory(),
        launcher=FakeLauncher(),
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {uuid: invalid_model for uuid in snapshots[0]},
        reservation_probe=lambda: {},
        process_scan=lambda: (),
        resource_lock_factory=FakeLockFactory(),
    )
    wrong_model.observe_gpus()
    with pytest.raises(ValueError, match="NVIDIA H800"):
        wrong_model.schedule([request(module, root, "full:seed_20260718")])


def test_reserved_waiting_and_fcooper_gpu7_are_excluded_even_when_idle(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    idle = gpu_snapshots(module, 8)
    launcher = FakeLauncher()
    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle, idle]),
        lock_factory=FakeLockFactory(),
        launcher=launcher,
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {uuid: module.EXPECTED_GPU_MODEL for uuid in idle},
        reservation_probe=lambda: {
            "GPU-1": ("waiting-job",),
            "GPU-2": ("reserved",),
        },
        process_scan=lambda: (
            module.ProcessIdentity(
                pid=77,
                owner="other-user",
                start_time="100",
                command=(
                    "python",
                    "/home/jichengzhi/V2X/scripts/fcooper_tvm_gpu7_validation_v1.py",
                    "--physical-gpu-id",
                    "0",
                ),
                alive=True,
            ),
        ),
        resource_lock_factory=FakeLockFactory(),
    )
    subject.observe_gpus()

    result = subject.schedule([request(module, root, "full:seed_20260718")])

    assert len(result["launched"]) == 1
    leased = launcher.calls[0]["env"]["CUDA_VISIBLE_DEVICES"].split(",")
    assert "GPU-1" not in leased
    assert "GPU-2" not in leased
    assert "GPU-7" not in leased
    audit = (
        Path(module.V2_ROOT) / "status" / "h800_runtime_admission.jsonl"
    ).read_text()
    assert "fcooper_gpu7_reserved" in audit
    assert "waiting-job" in audit


def test_gpu7_is_unconditionally_excluded_without_fcooper_process(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    idle = gpu_snapshots(module, 8)
    launcher = FakeLauncher()
    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle, idle]),
        lock_factory=FakeLockFactory(),
        launcher=launcher,
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {
            uuid: module.EXPECTED_GPU_MODEL for uuid in idle
        },
        reservation_probe=lambda: {},
        process_scan=lambda: (),
        resource_lock_factory=FakeLockFactory(),
    )
    subject.observe_gpus()

    result = subject.schedule([request(module, root, "full:seed_20260718")])

    assert len(result["launched"]) == 1
    leased = launcher.calls[0]["env"]["CUDA_VISIBLE_DEVICES"].split(",")
    assert "GPU-7" not in leased
    audit = (
        Path(module.V2_ROOT) / "status" / "h800_runtime_admission.jsonl"
    ).read_text()
    assert "stage7_gpu7_unconditionally_excluded" in audit


def test_default_process_reservation_discovery_excludes_waiting_gpu_flag(
    tmp_path: Path,
):
    module = load_module()
    idle = gpu_snapshots(module, 5)
    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle]),
        lock_factory=FakeLockFactory(),
        resource_lock_factory=FakeLockFactory(),
        launcher=FakeLauncher(),
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {uuid: module.EXPECTED_GPU_MODEL for uuid in idle},
        process_scan=lambda: (
            module.ProcessIdentity(
                pid=88,
                owner="other-user",
                start_time="101",
                command=(
                    "python",
                    "waiting_worker.py",
                    "--gpu",
                    "3",
                ),
                alive=True,
            ),
        ),
    )

    result = subject.preflight()

    assert result["idle_gpu_uuids"] == [
        "GPU-0",
        "GPU-1",
        "GPU-2",
        "GPU-4",
    ]


def test_post_lock_reservation_refresh_prevents_stale_launch(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    idle = gpu_snapshots(module, 4)
    launcher = FakeLauncher()

    class Reservations:
        calls = 0

        def __call__(self):
            self.calls += 1
            return {} if self.calls == 1 else {"GPU-0": ("new-waiter",)}

    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle, idle, idle]),
        lock_factory=FakeLockFactory(),
        resource_lock_factory=FakeLockFactory(),
        launcher=launcher,
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {uuid: module.EXPECTED_GPU_MODEL for uuid in idle},
        reservation_probe=Reservations(),
        process_scan=lambda: (),
    )
    subject.observe_gpus()

    result = subject.schedule([request(module, root, "full:seed_20260718")])

    assert result["launched"] == []
    assert result["blocked"] == [
        {
            "trajectory_id": "full:seed_20260718",
            "reason": "post_lock_runtime_drift",
        }
    ]
    assert launcher.calls == []


def test_resolved_source_lock_serializes_same_materialized_source_despite_logical_ids(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 8)] * 3, launcher)
    first = request(
        module,
        root,
        "full:seed_20260718",
        width=(16, 32, 64),
        source_identity="shared-source",
    )
    second = request(
        module,
        root,
        "backend_blind:seed_20260718",
        width=(24, 48, 96),
        source_identity="shared-source",
    )
    subject.observe_gpus()

    result = subject.schedule([first, second])

    assert first.source_lock_key == second.source_lock_key
    assert [item.trajectory_id for item in result["launched"]] == [
        first.trajectory_id,
    ]
    assert result["blocked"] == [
        {
            "trajectory_id": second.trajectory_id,
            "reason": "same_source_lock",
        }
    ]


def test_foreign_uuid_lock_is_skipped_when_four_other_uuids_are_available(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    idle = gpu_snapshots(module, 5)
    launcher = FakeLauncher()

    class ForeignUuidLockFactory(FakeLockFactory):
        def __call__(self, uuid: str):
            if uuid == "GPU-0":
                return None
            return super().__call__(uuid)

    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle, idle]),
        lock_factory=ForeignUuidLockFactory(),
        resource_lock_factory=FakeLockFactory(),
        launcher=launcher,
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {uuid: module.EXPECTED_GPU_MODEL for uuid in idle},
        reservation_probe=lambda: {},
        process_scan=lambda: (),
    )
    subject.observe_gpus()

    result = subject.schedule([request(module, root, "full:seed_20260718")])

    assert len(result["launched"]) == 1
    assert launcher.calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == (
        "GPU-1,GPU-2,GPU-3,GPU-4"
    )


def test_external_source_lock_refuses_batch_before_launcher(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    idle = gpu_snapshots(module, 4)
    launcher = FakeLauncher()

    class RefusingResourceLocks(FakeLockFactory):
        def __call__(self, key: str):
            if key.startswith("source:"):
                return None
            return super().__call__(key)

    subject = module.V2Stage7Scheduler(
        status_dir=Path(module.V2_ROOT) / "status",
        gpu_probe=SnapshotProbe([idle]),
        lock_factory=FakeLockFactory(),
        resource_lock_factory=RefusingResourceLocks(),
        launcher=launcher,
        process_inspector=lambda _pid: None,
        current_owner="stage7-user",
        idle_samples_required=1,
        hostname_probe=lambda: module.EXPECTED_HOSTNAME,
        gpu_model_probe=lambda: {uuid: module.EXPECTED_GPU_MODEL for uuid in idle},
        reservation_probe=lambda: {},
        process_scan=lambda: (),
    )
    subject.observe_gpus()

    result = subject.schedule([request(module, root, "full:seed_20260718")])

    assert result["launched"] == []
    assert result["blocked"] == [
        {
            "trajectory_id": "full:seed_20260718",
            "reason": "resource_lock_unavailable",
        }
    ]
    assert launcher.calls == []


def test_running_lease_audit_binds_uuid_model_request_and_window(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 4)] * 3, launcher)
    subject.observe_gpus()
    original = request(module, root, "full:seed_20260718")

    subject.schedule([original])

    controller = subject.status()["controllers"][original.controller_id]
    assert controller["request_sha256"] == original.request_sha256
    assert controller["resolved_source_lock_sha256"] == original.source_lock_key
    assert (
        controller["source_resolution_result_sha256"]
        == original.source_resolution_result_sha256
    )
    assert controller["gpu_uuids"] == [f"GPU-{index}" for index in range(4)]
    assert controller["gpu_models"] == {
        f"GPU-{index}": module.EXPECTED_GPU_MODEL for index in range(4)
    }
    assert controller["lease_window"]["opened_wall_time"] == 1_753_392_000.0
    assert controller["lease_window"]["closed_wall_time"] is None


@pytest.mark.parametrize(
    "existing_status",
    (
        "running",
        "feedback_complete",
        "succeeded",
        "failed_terminal",
        "infrastructure_retry_required",
    ),
)
def test_existing_controller_identity_rejects_changed_request_sha_in_all_states(
    tmp_path: Path, existing_status: str
):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 4)] * 8, launcher)
    subject.observe_gpus()
    original = request(module, root, "full:seed_20260718")
    subject.schedule([original])
    controllers = dict(subject._state["controllers"])
    controllers[original.controller_id] = {
        **controllers[original.controller_id],
        "status": existing_status,
    }
    subject._persist(controllers=controllers)
    changed = request(
        module,
        root,
        "full:seed_20260718",
        request_sha_override="c" * 64,
    )

    with pytest.raises(ValueError, match="request_sha256 drift"):
        subject.schedule([changed])
    assert len(launcher.calls) == 1


@pytest.mark.parametrize(
    "existing_status",
    (
        "running",
        "feedback_complete",
        "succeeded",
        "failed_terminal",
        "infrastructure_retry_required",
    ),
)
@pytest.mark.parametrize("physical_binding", ("missing", "changed"))
def test_existing_controller_rejects_missing_or_changed_physical_sha_in_all_states(
    tmp_path: Path, existing_status: str, physical_binding: str
):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 4)] * 10, launcher)
    subject.observe_gpus()
    original = request(module, root, "full:seed_20260718")
    subject.schedule([original])
    controllers = dict(subject._state["controllers"])
    stored = {
        **controllers[original.controller_id],
        "status": existing_status,
    }
    if physical_binding == "missing":
        stored.pop("physical_request_sha256", None)
    controllers[original.controller_id] = stored
    subject._persist(controllers=controllers)
    candidate = (
        original
        if physical_binding == "missing"
        else request(
            module,
            root,
            "full:seed_20260718",
            source_identity="changed-physical-source",
        )
    )

    with pytest.raises(ValueError, match="physical request SHA drift"):
        subject.schedule([candidate])
    assert len(launcher.calls) == 1


def test_v2_run_once_adapts_dict_schedule_result_without_bypass(
    tmp_path: Path,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, [gpu_snapshots(module, 4)] * 8, launcher)
    source = tmp_path / "deploy.py"
    source.write_text("pass\n")
    deploy_sources = [
        ROOT / relative
        for relative in module._source_scheduler.required_deployment_pins(ROOT)
    ]
    deploy = module._v1.build_deploy_manifest(
        files=[
            (source, "/remote/deploy.py"),
            *[
                (
                    phase3a_source,
                    "/remote/" + str(phase3a_source.relative_to(ROOT)),
                )
                for phase3a_source in deploy_sources
            ],
        ],
        python_environment="UniV2X_2.0",
        git_branch="test",
        git_commit="0" * 40,
        frozen_evidence={"contract": "9" * 64},
    )

    result = subject.run(
        [request(module, root, "full:seed_20260718")],
        deploy_manifest=deploy,
        remote_sha_probe=lambda destination: next(
            record["sha256"]
            for record in deploy["files"]
            if record["destination"] == destination
        ),
        once=True,
    )

    assert result["launched"] == ["full:seed_20260718:round_0"]
    assert result["blocked"] == []


def test_v2_root_must_equal_the_canonical_resolved_root_not_just_its_basename(
    tmp_path: Path,
):
    module = load_module()
    canonical = v2_root(module, tmp_path / "canonical")
    same_basename_elsewhere = tmp_path / "other" / canonical.name

    assert len(module.build_v2_trajectory_queue(canonical)) == 12
    with pytest.raises(ValueError, match="v2 result root"):
        module.build_v2_trajectory_queue(same_basename_elsewhere)


@pytest.mark.parametrize(
    ("trajectory", "round_index", "path_suffix", "command", "error"),
    [
        (
            "without_capability_scan:seed_20260718",
            0,
            "round_00",
            ("bash", "scripts/stage7_core_round_controller_v2.sh"),
            "core variant",
        ),
        (
            "full:seed_999",
            0,
            "round_00",
            ("bash", "scripts/stage7_core_round_controller_v2.sh"),
            "seed",
        ),
        (
            "full:seed_20260718",
            4,
            "round_04",
            ("bash", "scripts/stage7_core_round_controller_v2.sh"),
            "round_index",
        ),
        (
            "full:seed_20260718",
            0,
            "../../escape/round_00",
            ("bash", "scripts/stage7_core_round_controller_v2.sh"),
            "canonical v2 trajectory path",
        ),
        (
            "full:seed_20260718",
            0,
            "round_00",
            ("bash", "scripts/stage7_task_round_controller_v1.sh"),
            "exact v2 controller argv",
        ),
    ],
)
def test_v2_batch_request_rejects_noncanonical_identity_path_or_controller_before_leasing(
    tmp_path: Path,
    trajectory: str,
    round_index: int,
    path_suffix: str,
    command: tuple[str, ...],
    error: str,
):
    module = load_module()
    root = v2_root(module, tmp_path)
    variant, raw_seed = trajectory.split(":seed_", 1)
    path = root / "variants" / variant / f"seed_{raw_seed}" / path_suffix

    with pytest.raises(ValueError, match=error):
        module.V2BatchRequest(
            trajectory_id=trajectory,
            trajectory_path=path,
            round_index=round_index,
            request_sha256="a" * 64,
            selected_row_ids=("row-0", "row-1", "row-2", "row-3"),
            command=command,
            width=(16, 32, 64),
            expected_release_sha256=EXPECTED_RELEASE_SHA256,
            expected_manifest_file_sha256=EXPECTED_MANIFEST_FILE_SHA256,
        )


@pytest.mark.parametrize(
    "command",
    [
        ("stage7_core_round_controller_v2.sh",),
        ("scripts/stage7_core_round_controller_v2.sh",),
        ("/tmp/stage7_core_round_controller_v2.sh",),
        ("/bin/bash", str(ROOT / "scripts/stage7_core_round_controller_v2.sh")),
        (
            str(ROOT / "scripts/stage7_core_round_controller_v2.sh"),
            "--output-root",
            "/wrong",
        ),
    ],
)
def test_v2_batch_request_requires_exact_absolute_controller_argv(
    tmp_path: Path, command: tuple[str, ...]
):
    module = load_module()
    root = v2_root(module, tmp_path)

    with pytest.raises(ValueError, match="exact v2 controller argv"):
        module.V2BatchRequest(
            trajectory_id="full:seed_20260718",
            trajectory_path=root / "variants/full/seed_20260718/round_00",
            round_index=0,
            request_sha256="a" * 64,
            selected_row_ids=("row-0", "row-1", "row-2", "row-3"),
            command=command,
            width=(16, 32, 64),
            expected_release_sha256=EXPECTED_RELEASE_SHA256,
            expected_manifest_file_sha256=EXPECTED_MANIFEST_FILE_SHA256,
        )
