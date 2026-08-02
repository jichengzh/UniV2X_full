from __future__ import annotations
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage7_ablation_scheduler_v1.py"
def load_module():
    spec = importlib.util.spec_from_file_location("stage7_ablation_scheduler_v1", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
@dataclass
class FakeProcess:
    pid: int
    owner: str
    start_time: str
    command: tuple[str, ...]
    returncode: int | None = None
    def poll(self) -> int | None:
        return self.returncode
class FakeLauncher:
    def __init__(self, *, owner: str = "stage7-user") -> None:
        self.owner = owner
        self.calls: list[dict[str, object]] = []
        self.processes: list[FakeProcess] = []
    def __call__(self, command, *, env, cwd):
        call = {"command": tuple(command), "env": dict(env), "cwd": str(cwd)}
        self.calls.append(call)
        pid = 7000 + len(self.calls)
        process = FakeProcess(
            pid=pid,
            owner=self.owner,
            start_time=f"2026-07-25T00:00:{len(self.calls):02d}Z",
            command=tuple(command),
        )
        self.processes.append(process)
        return process
class FakeLock:
    def __init__(self, uuid: str, released: list[str]) -> None:
        self.uuid = uuid
        self._released = released
    def release(self) -> None:
        self._released.append(self.uuid)
class FakeLockFactory:
    def __init__(self, denied: set[str] | None = None) -> None:
        self.denied = denied or set()
        self.acquired: list[str] = []
        self.released: list[str] = []
    def __call__(self, uuid: str):
        if uuid in self.denied:
            return None
        self.acquired.append(uuid)
        return FakeLock(uuid, self.released)
class SequenceProbe:
    def __init__(self, snapshots: list[dict[str, object]]) -> None:
        self.snapshots = list(snapshots)
        self.calls = 0
    def __call__(self):
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        return self.snapshots[index]
def gpu_set(module, count: int, *, start_index: int = 0):
    return {
        f"GPU-{number}": module.GpuSnapshot(
            uuid=f"GPU-{number}",
            index=start_index + number,
            memory_used_mib=0,
            utilization_percent=0,
            compute_pids=(),
        )
        for number in range(count)
    }
def busy_gpu(module, uuid: str, *, index: int, pid: int = 9001):
    return module.GpuSnapshot(
        uuid=uuid,
        index=index,
        memory_used_mib=4096,
        utilization_percent=90,
        compute_pids=(pid,),
    )
def batch(module, root: Path, trajectory_id: str, *, round_index: int = 0, sha: str = "a" * 64):
    return module.BatchRequest(
        trajectory_id=trajectory_id,
        trajectory_path=root / trajectory_id / f"round_{round_index:02d}",
        round_index=round_index,
        request_sha256=sha,
        selected_row_ids=("r0", "r1", "r2", "r3"),
        command=("python", "scripts/stage7_task_round_controller_v1.sh", trajectory_id),
        task_id="S7-PYR-TVM",
    )
def scheduler(module, tmp_path: Path, probe, locks, launcher, **overrides):
    return module.Stage7Scheduler(
        status_dir=tmp_path / "status",
        gpu_probe=probe,
        lock_factory=locks,
        launcher=launcher,
        process_inspector=overrides.pop("process_inspector", lambda _pid: None),
        wall_time=overrides.pop("wall_time", lambda: 1_753_392_000.0),
        current_owner=overrides.pop("current_owner", "stage7-user"),
        idle_samples_required=overrides.pop("idle_samples_required", 2),
        memory_idle_threshold_mib=overrides.pop("memory_idle_threshold_mib", 1024),
        utilization_idle_threshold_percent=overrides.pop(
            "utilization_idle_threshold_percent", 5
        ),
        **overrides,
    )
def test_leases_only_after_consecutive_idle_samples_and_nonblocking_locks(tmp_path):
    module = load_module()
    probe = SequenceProbe([gpu_set(module, 4)] * 3)
    locks = FakeLockFactory()
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, locks, launcher)
    subject.observe_gpus()
    assert subject.schedule([batch(module, tmp_path, "full-seed-1")]) == []
    subject.observe_gpus()
    launched = subject.schedule([batch(module, tmp_path, "full-seed-1")])
    assert len(launched) == 1
    assert locks.acquired == ["GPU-0", "GPU-1", "GPU-2", "GPU-3"]
    assert launcher.calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-0,GPU-1,GPU-2,GPU-3"
def test_foreign_uuid_lock_prevents_batch_and_releases_partial_lease(tmp_path):
    module = load_module()
    probe = SequenceProbe([gpu_set(module, 4)] * 3)
    locks = FakeLockFactory(denied={"GPU-2"})
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, locks, launcher)
    subject.observe_gpus()
    subject.observe_gpus()
    launched = subject.schedule([batch(module, tmp_path, "full-seed-1")])
    assert launched == []
    assert launcher.calls == []
    assert locks.released == ["GPU-0", "GPU-1"]
    events = [
        json.loads(line)
        for line in (tmp_path / "status" / "gpu_lease_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[-1]["event"] == "lease_refused_foreign_lock"
def test_idle_streak_tracks_uuid_when_index_changes(tmp_path):
    module = load_module()
    first = gpu_set(module, 4)
    second = {
        uuid: module.GpuSnapshot(
            uuid=uuid,
            index=snapshot.index + 4,
            memory_used_mib=0,
            utilization_percent=0,
            compute_pids=(),
        )
        for uuid, snapshot in first.items()
    }
    probe = SequenceProbe([first, second, second])
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, FakeLockFactory(), launcher)
    subject.observe_gpus()
    subject.observe_gpus()
    subject.schedule([batch(module, tmp_path, "full-seed-1")])
    assert len(launcher.calls) == 1
    assert launcher.calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-0,GPU-1,GPU-2,GPU-3"
def test_releases_all_locks_when_immediate_recheck_finds_external_process(tmp_path):
    module = load_module()
    idle = gpu_set(module, 4)
    occupied = {**idle, "GPU-2": busy_gpu(module, "GPU-2", index=2)}
    probe = SequenceProbe([idle, idle, occupied])
    locks = FakeLockFactory()
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, locks, launcher)
    subject.observe_gpus()
    subject.observe_gpus()
    launched = subject.schedule([batch(module, tmp_path, "full-seed-1")])
    assert launched == []
    assert launcher.calls == []
    assert set(locks.released) == {"GPU-0", "GPU-1", "GPU-2", "GPU-3"}
    assert subject.status()["scheduler_status"] == "waiting_for_four_idle_gpus"
def test_fewer_than_four_idle_uuids_never_launches_a_substitute_batch(tmp_path):
    module = load_module()
    probe = SequenceProbe([gpu_set(module, 3)] * 3)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, FakeLockFactory(), launcher)
    subject.observe_gpus()
    subject.observe_gpus()
    launched = subject.schedule([batch(module, tmp_path, "full-seed-1")])
    assert launched == []
    assert launcher.calls == []
    assert subject.status()["scheduler_status"] == "waiting_for_four_idle_gpus"
def test_eight_idle_uuids_launch_at_most_two_indivisible_batches(tmp_path):
    module = load_module()
    probe = SequenceProbe([gpu_set(module, 8)] * 3)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, FakeLockFactory(), launcher)
    requests = [batch(module, tmp_path, f"trajectory-{index}") for index in range(3)]
    subject.observe_gpus()
    subject.observe_gpus()
    launched = subject.schedule(requests)
    assert len(launched) == 2
    assert len(launcher.calls) == 2
    visible = {call["env"]["CUDA_VISIBLE_DEVICES"] for call in launcher.calls}
    assert visible == {
        "GPU-0,GPU-1,GPU-2,GPU-3",
        "GPU-4,GPU-5,GPU-6,GPU-7",
    }
def test_feedback_barrier_blocks_next_round_prefetch(tmp_path):
    module = load_module()
    probe = SequenceProbe([gpu_set(module, 8)] * 4)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, FakeLockFactory(), launcher)
    round_zero = batch(module, tmp_path, "same-trajectory", round_index=0)
    round_one = batch(
        module, tmp_path, "same-trajectory", round_index=1, sha="b" * 64
    )
    subject.observe_gpus()
    subject.observe_gpus()
    subject.schedule([round_zero])
    launched = subject.schedule([round_one])
    assert launched == []
    assert len(launcher.calls) == 1
def test_round_after_zero_cannot_launch_without_all_prior_feedback(tmp_path):
    module = load_module()
    probe = SequenceProbe([gpu_set(module, 4)] * 3)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, FakeLockFactory(), launcher)
    orphan_round = batch(
        module, tmp_path, "same-trajectory", round_index=1, sha="b" * 64
    )
    subject.observe_gpus()
    subject.observe_gpus()
    launched = subject.schedule([orphan_round])
    assert launched == []
    assert launcher.calls == []
def test_successful_controller_completion_opens_feedback_barrier(tmp_path):
    module = load_module()
    probe = SequenceProbe([gpu_set(module, 8)] * 5)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, FakeLockFactory(), launcher)
    round_zero = batch(module, tmp_path, "same-trajectory", round_index=0)
    round_one = batch(
        module, tmp_path, "same-trajectory", round_index=1, sha="b" * 64
    )
    subject.observe_gpus()
    subject.observe_gpus()
    subject.schedule([round_zero])
    launcher.processes[0].returncode = 0
    completed = subject.reap_controllers()
    launched = subject.schedule([round_one])
    assert completed == [round_zero.controller_id]
    assert [item.controller_id for item in launched] == [round_one.controller_id]
    assert subject.status()["selected_event_budget_consumed"] == 4
def test_legitimate_descendant_compute_pid_is_not_occupancy_drift(tmp_path):
    module = load_module()
    idle = gpu_set(module, 4)
    child_busy = {
        uuid: module.GpuSnapshot(
            uuid=uuid,
            index=snapshot.index,
            memory_used_mib=4096,
            utilization_percent=90,
            compute_pids=(7100 + snapshot.index,),
        )
        for uuid, snapshot in idle.items()
    }
    probe = SequenceProbe([idle, idle, idle, child_busy])
    launcher = FakeLauncher()
    subject = scheduler(
        module,
        tmp_path,
        probe,
        FakeLockFactory(),
        launcher,
        pid_is_owned=lambda controller_pid, gpu_pid: (
            controller_pid == 7001 and 7100 <= gpu_pid <= 7103
        ),
    )
    request = batch(module, tmp_path, "full-seed-1")
    subject.observe_gpus()
    subject.observe_gpus()
    subject.schedule([request])
    assert subject.monitor_occupancy() == []
    assert subject.status()["controllers"][request.controller_id]["status"] == "running"
def test_occupancy_drift_retries_the_same_request_sha_without_budget_change(tmp_path):
    module = load_module()
    request_sha = "d" * 64
    idle = gpu_set(module, 4)
    drifted = {**idle, "GPU-1": busy_gpu(module, "GPU-1", index=1, pid=9999)}
    probe = SequenceProbe([idle, idle, idle, drifted])
    locks = FakeLockFactory()
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, locks, launcher)
    request = batch(module, tmp_path, "full-seed-1", sha=request_sha)
    subject.observe_gpus()
    subject.observe_gpus()
    subject.schedule([request])
    before = subject.status()["selected_event_budget_consumed"]
    retries = subject.monitor_occupancy()
    assert [item.request_sha256 for item in retries] == [request_sha]
    assert subject.status()["selected_event_budget_consumed"] == before
    controller = next(iter(subject.status()["controllers"].values()))
    assert controller["status"] == "infrastructure_retry_required"
    assert controller["request_sha256"] == request_sha
    assert set(locks.released) == {"GPU-0", "GPU-1", "GPU-2", "GPU-3"}
def test_trajectory_queue_is_exactly_five_variants_by_three_seeds_with_unique_paths(tmp_path):
    module = load_module()
    queue = module.build_trajectory_queue(tmp_path)
    assert len(queue) == 15
    assert {item.variant for item in queue} == {
        "full",
        "without_surrogate",
        "without_measured_feedback",
        "backend_blind",
        "without_capability_scan",
    }
    assert {item.seed for item in queue} == {20260718, 20260719, 20260720}
    assert len({item.path for item in queue}) == 15
    assert all(item.task_id == "S7-PYR-TVM" for item in queue)
    assert all(item.path.name == f"seed_{item.seed}" for item in queue)
    assert all(item.round_path(0).name == "round_00" for item in queue)
    assert all("S7-PYR-TVM" not in item.path.parts for item in queue)
def test_batch_request_rejects_noncanonical_round_directory(tmp_path):
    module = load_module()

    with pytest.raises(ValueError, match="round directory"):
        module.BatchRequest(
            trajectory_id="full:seed_20260718",
            trajectory_path=tmp_path / "S7-PYR-TVM",
            round_index=0,
            request_sha256="a" * 64,
            selected_row_ids=("r0", "r1", "r2", "r3"),
            command=("python", "scripts/stage7_task_round_controller_v1.sh"),
        )


def test_batch_request_rejects_non_stage7_controller_command(tmp_path):
    module = load_module()

    with pytest.raises(ValueError, match="controller command"):
        module.BatchRequest(
            trajectory_id="full:seed_20260718",
            trajectory_path=tmp_path / "round_00",
            round_index=0,
            request_sha256="a" * 64,
            selected_row_ids=("r0", "r1", "r2", "r3"),
            command=("python", "foreign_program.py"),
        )


def test_deploy_sha_mismatch_blocks_run_before_any_launch(tmp_path):
    module = load_module()
    source = tmp_path / "scheduler.py"
    source.write_text("frozen", encoding="utf-8")
    manifest = module.build_deploy_manifest(
        files=[(source, "/remote/stage7/scheduler.py")],
        python_environment="/opt/conda/envs/stage7/bin/python",
        git_branch="stage7",
        git_commit="1234567",
        frozen_evidence={"document37_sha256": "e" * 64},
    )
    probe = SequenceProbe([gpu_set(module, 4)] * 3)
    launcher = FakeLauncher()
    subject = scheduler(module, tmp_path, probe, FakeLockFactory(), launcher)

    with pytest.raises(ValueError, match="blocked_remote_deploy_sha_mismatch"):
        subject.run(
            [batch(module, tmp_path, "full-seed-1")],
            deploy_manifest=manifest,
            remote_sha_probe=lambda _destination: "0" * 64,
            once=True,
        )

    assert launcher.calls == []


def test_empty_deploy_file_set_cannot_bypass_remote_sha_gate(tmp_path):
    module = load_module()
    subject = scheduler(
        module,
        tmp_path,
        SequenceProbe([gpu_set(module, 4)]),
        FakeLockFactory(),
        FakeLauncher(),
    )

    with pytest.raises(ValueError, match="blocked_remote_deploy_sha_mismatch"):
        subject.run(
            [batch(module, tmp_path, "full-seed-1")],
            deploy_manifest={"schema_version": module.DEPLOY_SCHEMA, "files": []},
            remote_sha_probe=lambda _destination: "",
            once=True,
        )


def test_nonhex_deploy_digest_is_rejected_before_remote_probe():
    module = load_module()
    manifest = {
        "schema_version": module.DEPLOY_SCHEMA,
        "files": [
            {
                "source": "/local/scheduler.py",
                "destination": "/remote/scheduler.py",
                "sha256": "z" * 64,
            }
        ],
    }

    with pytest.raises(ValueError, match="blocked_remote_deploy_sha_mismatch"):
        module.verify_remote_deploy(manifest, lambda _destination: "z" * 64)


def test_fake_nvidia_smi_is_joined_by_uuid_not_index():
    module = load_module()

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_kwargs):
        calls.append(tuple(command))
        if "--query-gpu=index,uuid,memory.used,utilization.gpu" in command:
            return Result("7, GPU-stable, 12, 3\n0, GPU-other, 0, 0\nmalformed\n")
        return Result("GPU-stable, 8123\n")

    snapshots = module.query_nvidia_smi(run_command=fake_run)

    assert len(calls) == 2
    assert snapshots["GPU-stable"] == module.GpuSnapshot(
        uuid="GPU-stable",
        index=7,
        memory_used_mib=12,
        utilization_percent=3,
        compute_pids=(8123,),
    )
    assert snapshots["GPU-other"].compute_pids == ()


def test_local_deploy_and_status_cli_write_only_frozen_manifests(
    tmp_path, capsys
):
    module = load_module()
    source = tmp_path / "scheduler.py"
    source.write_text("frozen", encoding="utf-8")
    deploy_manifest_path = tmp_path / "stage7_deploy_manifest.json"
    status_dir = tmp_path / "status"
    deploy_config = tmp_path / "deploy.json"
    deploy_config.write_text(
        json.dumps(
            {
                "deploy_files": [
                    {"source": str(source), "destination": "/remote/scheduler.py"}
                ],
                "python_environment": "/opt/stage7/python",
                "git_branch": "stage7",
                "git_commit": "1234567",
                "frozen_evidence": {"document37_sha256": "f" * 64},
                "deploy_manifest_path": str(deploy_manifest_path),
            }
        ),
        encoding="utf-8",
    )

    assert module.main(["deploy", "--config", str(deploy_config)]) == 0
    capsys.readouterr()
    manifest = json.loads(deploy_manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"][0]["sha256"] == module.sha256_file(source)

    status_config = tmp_path / "status.json"
    status_config.write_text(
        json.dumps({"status_dir": str(status_dir)}),
        encoding="utf-8",
    )
    assert module.main(["status", "--config", str(status_config)]) == 0
    status_output = json.loads(capsys.readouterr().out)
    assert status_output["scheduler_status"] == "initialized"


def test_deploy_manifest_and_cli_surface_never_accept_secrets(tmp_path):
    module = load_module()
    source = tmp_path / "scheduler.py"
    source.write_text("frozen", encoding="utf-8")

    manifest = module.build_deploy_manifest(
        files=[(source, "/remote/stage7/scheduler.py")],
        python_environment="/opt/conda/envs/stage7/bin/python",
        git_branch="stage7",
        git_commit="1234567",
        frozen_evidence={"document37_sha256": "e" * 64},
    )
    serialized = json.dumps(manifest, sort_keys=True).lower()
    parser = module.build_parser()
    option_names = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert all(word not in serialized for word in ("password", "token", "private_key"))
    assert not {"--password", "--token", "--private-key"} & option_names
    assert set(module.SUBCOMMANDS) == {
        "deploy",
        "preflight",
        "run",
        "resume",
        "status",
        "stop-own-processes",
    }
    subparsers_action = next(
        action
        for action in parser._actions
        if isinstance(action, module.argparse._SubParsersAction)
    )
    for name in ("run", "resume"):
        options = {
            option
            for action in subparsers_action.choices[name]._actions
            for option in action.option_strings
        }
        assert "--dry-run" not in options


def test_resume_does_not_duplicate_an_already_live_matching_controller(tmp_path):
    module = load_module()
    idle = gpu_set(module, 4)
    first_launcher = FakeLauncher()
    first = scheduler(
        module,
        tmp_path,
        SequenceProbe([idle, idle, idle]),
        FakeLockFactory(),
        first_launcher,
    )
    request = batch(module, tmp_path, "full-seed-1")
    first.observe_gpus()
    first.observe_gpus()
    first.schedule([request])
    launched_process = FakeProcess(
        pid=7001,
        owner="stage7-user",
        start_time="2026-07-25T00:00:01Z",
        command=request.command,
    )

    resumed_launcher = FakeLauncher()
    resumed = scheduler(
        module,
        tmp_path,
        SequenceProbe([idle]),
        FakeLockFactory(),
        resumed_launcher,
        process_inspector=lambda pid: launched_process if pid == 7001 else None,
    )
    result = resumed.resume([request], once=True)

    assert result["already_live"] == [request.controller_id]
    assert resumed_launcher.calls == []


def test_resumed_controller_disappearance_retries_same_request_instead_of_stalling(
    tmp_path,
):
    module = load_module()
    idle = gpu_set(module, 4)
    first_launcher = FakeLauncher()
    first = scheduler(
        module,
        tmp_path,
        SequenceProbe([idle, idle, idle]),
        FakeLockFactory(),
        first_launcher,
    )
    request = batch(module, tmp_path, "full-seed-1")
    first.observe_gpus()
    first.observe_gpus()
    first.schedule([request])
    live = {
        7001: module.ProcessIdentity(
            pid=7001,
            owner="stage7-user",
            start_time="2026-07-25T00:00:01Z",
            command=request.command,
            alive=True,
        )
    }
    resumed = scheduler(
        module,
        tmp_path,
        SequenceProbe([idle]),
        FakeLockFactory(),
        FakeLauncher(),
        process_inspector=lambda pid: live.get(pid),
    )

    resumed.resume([request], once=True)
    live.clear()
    later = scheduler(
        module,
        tmp_path,
        SequenceProbe([idle]),
        FakeLockFactory(),
        FakeLauncher(),
        process_inspector=lambda pid: live.get(pid),
    )
    later.reap_controllers()

    controller = later.status()["controllers"][request.controller_id]
    assert controller["status"] == "infrastructure_retry_required"
    assert controller["retry_request_sha256"] == request.request_sha256
    assert later.status()["scheduler_status"] == "infrastructure_retry_required"


def test_stop_own_processes_requires_exact_owner_start_time_and_command(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "status" / "controller_pids.json"
    manifest_path.parent.mkdir(parents=True)
    record = {
        "controller-a": {
            "pid": 8123,
            "owner": "stage7-user",
            "start_time": "2026-07-25T01:02:03Z",
            "command": ["python", "scripts/stage7_task_round_controller_v1.sh", "a"],
            "request_sha256": "a" * 64,
        }
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": module.CONTROLLER_SCHEMA,
                "controllers": record,
            }
        ),
        encoding="utf-8",
    )
    killed: list[int] = []

    mismatch = module.stop_own_processes(
        manifest_path,
        process_inspector=lambda _pid: module.ProcessIdentity(
            pid=8123,
            owner="other-user",
            start_time="2026-07-25T01:02:03Z",
            command=("python", "scripts/stage7_task_round_controller_v1.sh", "a"),
            alive=True,
        ),
        terminator=killed.append,
        current_owner="stage7-user",
    )
    matched = module.stop_own_processes(
        manifest_path,
        process_inspector=lambda _pid: module.ProcessIdentity(
            pid=8123,
            owner="stage7-user",
            start_time="2026-07-25T01:02:03Z",
            command=("python", "scripts/stage7_task_round_controller_v1.sh", "a"),
            alive=True,
        ),
        terminator=killed.append,
        current_owner="stage7-user",
    )

    assert mismatch == {"stopped": [], "refused": ["controller-a"]}
    assert matched == {"stopped": ["controller-a"], "refused": []}
    assert killed == [8123]


def test_stop_refuses_non_stage7_or_wrong_schema_controller_manifest(tmp_path):
    module = load_module()
    manifest_path = tmp_path / "controller_pids.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "untrusted",
                "controllers": {
                    "foreign": {
                        "pid": 8123,
                        "owner": "stage7-user",
                        "start_time": "123",
                        "command": ["python", "foreign.py"],
                        "request_sha256": "a" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    killed: list[int] = []

    with pytest.raises(ValueError, match="controller PID manifest schema"):
        module.stop_own_processes(
            manifest_path,
            process_inspector=lambda _pid: None,
            terminator=killed.append,
            current_owner="stage7-user",
        )

    assert killed == []


def test_real_launcher_identity_is_inspected_and_persisted_for_safe_stop(tmp_path):
    module = load_module()
    idle = gpu_set(module, 4)

    class BareProcess:
        pid = 7001
        returncode = None

        def poll(self):
            return self.returncode

    launcher_calls: list[tuple[str, ...]] = []

    def bare_launcher(command, **_kwargs):
        launcher_calls.append(tuple(command))
        return BareProcess()

    identity = module.ProcessIdentity(
        pid=7001,
        owner="stage7-user",
        start_time="424242",
        command=("python", "scripts/stage7_task_round_controller_v1.sh", "full-seed-1"),
        alive=True,
    )
    subject = scheduler(
        module,
        tmp_path,
        SequenceProbe([idle, idle, idle]),
        FakeLockFactory(),
        bare_launcher,
        process_inspector=lambda pid: identity if pid == 7001 else None,
    )
    request = batch(module, tmp_path, "full-seed-1")
    subject.observe_gpus()
    subject.observe_gpus()
    subject.schedule([request])
    manifest = json.loads(subject.controller_path.read_text(encoding="utf-8"))

    assert launcher_calls == [request.command]
    assert manifest["controllers"][request.controller_id]["start_time"] == "424242"


def test_preflight_is_read_only_and_writes_named_audit_files(tmp_path):
    module = load_module()
    locks = FakeLockFactory()
    subject = scheduler(
        module,
        tmp_path,
        SequenceProbe([gpu_set(module, 4)]),
        locks,
        FakeLauncher(),
        idle_samples_required=1,
    )

    result = subject.preflight()

    assert result["idle_gpu_uuids"] == ["GPU-0", "GPU-1", "GPU-2", "GPU-3"]
    assert locks.acquired == []
    assert (tmp_path / "status" / "gpu_occupancy_snapshots.jsonl").is_file()
    assert (tmp_path / "status" / "gpu_lease_audit.jsonl").is_file()
    assert (tmp_path / "status" / "scheduler_state.json").is_file()
    assert (tmp_path / "status" / "controller_pids.json").is_file()
