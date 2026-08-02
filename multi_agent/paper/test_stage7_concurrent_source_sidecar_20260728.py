from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


SCRIPT = Path(__file__).with_name("stage7_concurrent_source_sidecar_20260728.py")
SPEC = importlib.util.spec_from_file_location("stage7_source_sidecar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_find_canonical_orchestrator_uses_unique_live_replacement(monkeypatch):
    root = Path("/formal/stage7")
    commands = {
        111: None,
        222: [
            "python",
            "stage7_speed_priority_takeover_v1.py",
            "--v2-root",
            str(root),
        ],
        333: ["python", "unrelated.py", str(root)],
    }
    monkeypatch.setattr(MODULE, "_iter_numeric_proc_pids", lambda: (111, 222, 333))
    monkeypatch.setattr(MODULE, "process_command", commands.get)

    assert MODULE.find_canonical_orchestrator_pid(root, 111) == 222


def test_find_canonical_orchestrator_rejects_ambiguous_live_replacements(
    monkeypatch,
):
    root = Path("/formal/stage7")
    command = [
        "python",
        "stage7_speed_priority_takeover_v1.py",
        "--v2-root",
        str(root),
    ]
    monkeypatch.setattr(MODULE, "_iter_numeric_proc_pids", lambda: (222, 444))
    monkeypatch.setattr(
        MODULE,
        "process_command",
        lambda pid: None if pid == 111 else command,
    )

    try:
        MODULE.find_canonical_orchestrator_pid(root, 111)
    except RuntimeError as error:
        assert "not unique" in str(error)
    else:
        raise AssertionError("ambiguous canonical orchestrator was accepted")


def test_limit_ready_uuids_keeps_order_and_caps_one_gpu():
    assert MODULE.limit_ready_uuids(
        ("GPU-a", "GPU-b", "GPU-c"),
        max_gpus=1,
    ) == ("GPU-a",)


def test_limit_ready_uuids_rejects_nonpositive_limit():
    try:
        MODULE.limit_ready_uuids(("GPU-a",), max_gpus=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive sidecar GPU limit was accepted")


def test_install_sidecar_gpu_cap_only_admits_once_per_process():
    locks = []

    class FakeScheduler:
        def _ready_uuids(self):
            return ["GPU-a", "GPU-b", "GPU-c"]

        def lock_factory(self, uuid):
            lock = _FakeLock()
            locks.append((uuid, lock))
            return lock

    MODULE.install_sidecar_gpu_cap(FakeScheduler, max_gpus=1)
    scheduler = FakeScheduler()

    assert scheduler._ready_uuids() == ["GPU-a"]
    assert scheduler._ready_uuids() == []
    assert [uuid for uuid, _ in locks] == ["GPU-a", "GPU-b", "GPU-c"]
    assert all(lock.released for _, lock in locks)


class _FakeLock:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def _run_main_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    controller_result=None,
    controller_error: Exception | None = None,
):
    root = tmp_path / "formal"
    frozen = tmp_path / "frozen"
    code = root / "deployment" / "code"
    round_dir = (
        root
        / "variants"
        / "without_surrogate"
        / "seed_20260718"
        / "round_02"
    )
    for directory in (frozen, code, round_dir, root / "tools"):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "prepare_state.json").write_text(
        json.dumps(
            {
                "executor_repo_root": str(frozen),
                "orchestrator_pid": 123,
                "deployment_release_sha256": "a" * 64,
                "deployment_manifest_file_sha256": "b" * 64,
            }
        )
    )
    (round_dir / "source_resolution_plan.json").write_text("{}")
    (round_dir / "logical_request.json").write_text(
        json.dumps({"measurement_request_sha256": "c" * 64})
    )
    (root / "tools" / "stage7_speed_priority_takeover_v1.py").write_text(
        "def install_speed_priority_policy(module):\n"
        "    return {'max_parallel_controllers': 8}\n"
    )

    locks = []

    class FakeScheduler:
        def _ready_uuids(self):
            return ["GPU-a", "GPU-b"]

        def lock_factory(self, uuid):
            lock = _FakeLock()
            locks.append((uuid, lock))
            return lock

    scheduler_module = ModuleType(
        "scripts.stage7_core_ablation_scheduler_v2"
    )
    scheduler_module.V2Stage7Scheduler = FakeScheduler
    controller_module = ModuleType("scripts.stage7_source_lease_controller_v2")
    observed = {}

    def run_source_lease_controller(**kwargs):
        observed.update(kwargs)
        observed["ready"] = FakeScheduler()._ready_uuids()
        if controller_error is not None:
            raise controller_error
        return controller_result or {"status": "cache_revealed"}

    controller_module.run_source_lease_controller = run_source_lease_controller
    scripts_package = ModuleType("scripts")
    scripts_package.__path__ = []
    scripts_package.stage7_core_ablation_scheduler_v2 = scheduler_module
    scripts_package.stage7_source_lease_controller_v2 = controller_module
    monkeypatch.setitem(sys.modules, "scripts", scripts_package)
    monkeypatch.setitem(
        sys.modules,
        "scripts.stage7_core_ablation_scheduler_v2",
        scheduler_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "scripts.stage7_source_lease_controller_v2",
        controller_module,
    )
    monkeypatch.setattr(
        MODULE, "find_canonical_orchestrator_pid", lambda root, pid: 456
    )
    monkeypatch.setattr(
        MODULE,
        "process_command",
        lambda pid: [
            "python",
            "stage7_speed_priority_takeover_v1.py",
            str(root),
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--root",
            str(root),
            "--variant",
            "without_surrogate",
            "--seed",
            "20260718",
            "--round-index",
            "2",
            "--audit-tag",
            "test_run",
            "--max-gpus",
            "1",
        ],
    )
    return root, round_dir, locks, observed


def test_main_caps_each_sidecar_to_one_gpu_and_writes_completion(
    tmp_path,
    monkeypatch,
):
    root, round_dir, locks, observed = _run_main_fixture(
        tmp_path,
        monkeypatch,
    )

    assert MODULE.main() == 0
    completion = json.loads(
        (
            root
            / "audits/speed_priority_recovery/concurrent_source_sidecars"
            / "test_run"
            / "completion.json"
        ).read_text()
    )
    assert completion["status"] == "completed"
    assert completion["speed_policy"]["max_gpus_per_sidecar"] == 1
    assert completion["sidecar_file_sha256"] == MODULE.sha256(SCRIPT)
    assert completion["orchestrator_pid_rebound"] is True
    assert observed["ready"] == ["GPU-a"]
    assert all(lock.released for _, lock in locks)
    assert observed["variant"] == "without_surrogate"
    assert observed["round_index"] == 2
    assert observed["repo_root"].is_dir()
    assert not (round_dir / "source_resolution_result.json").exists()


def test_main_writes_failure_audit_before_reraising(tmp_path, monkeypatch):
    root, _, _, _ = _run_main_fixture(
        tmp_path,
        monkeypatch,
        controller_error=RuntimeError("source failed"),
    )

    with pytest.raises(RuntimeError, match="source failed"):
        MODULE.main()
    failure = json.loads(
        (
            root
            / "audits/speed_priority_recovery/concurrent_source_sidecars"
            / "test_run"
            / "failure.json"
        ).read_text()
    )
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert "source failed" in failure["traceback"]
