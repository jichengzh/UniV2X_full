from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "fcooper_stage6_watchdog_v2.py"
)
SPEC = importlib.util.spec_from_file_location("fcooper_stage6_watchdog_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_decide_action_stops_after_completion() -> None:
    assert (
        MODULE.decide_action(
            complete=True,
            supervisor_pids=[],
            runner_pids=[],
            restart_count=0,
            max_restarts=3,
        )
        == "complete"
    )


def test_decide_action_monitors_existing_supervisor() -> None:
    assert (
        MODULE.decide_action(
            complete=False,
            supervisor_pids=[123],
            runner_pids=[],
            restart_count=0,
            max_restarts=3,
        )
        == "monitor"
    )


def test_decide_action_waits_for_orphaned_rows() -> None:
    assert (
        MODULE.decide_action(
            complete=False,
            supervisor_pids=[],
            runner_pids=[456],
            restart_count=0,
            max_restarts=3,
        )
        == "wait_runners"
    )


def test_decide_action_restarts_only_within_bound() -> None:
    assert (
        MODULE.decide_action(
            complete=False,
            supervisor_pids=[],
            runner_pids=[],
            restart_count=2,
            max_restarts=3,
        )
        == "restart"
    )
    assert (
        MODULE.decide_action(
            complete=False,
            supervisor_pids=[],
            runner_pids=[],
            restart_count=3,
            max_restarts=3,
        )
        == "exhausted"
    )


def test_root_supervisor_excludes_child_shell_copies() -> None:
    processes = {
        100: MODULE.ProcessInfo(
            pid=100,
            ppid=1,
            command=("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh"),
        ),
        101: MODULE.ProcessInfo(
            pid=101,
            ppid=100,
            command=("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh"),
        ),
        102: MODULE.ProcessInfo(
            pid=102,
            ppid=101,
            command=("python", "/repo/fcooper_execute_measurement_row_v2.py"),
        ),
    }
    command = processes[100].command
    assert MODULE.root_supervisor_pids(processes, command) == [100]


def test_exact_supervisor_matching_rejects_wrapper_substrings() -> None:
    command = ("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh", "--x")
    processes = {
        100: MODULE.ProcessInfo(pid=100, ppid=1, command=command),
        200: MODULE.ProcessInfo(
            pid=200,
            ppid=1,
            command=(
                "bash",
                "-c",
                "inspect fcooper_stage6_five_arm_supervisor_v2.sh --x",
            ),
        ),
    }
    assert MODULE.root_supervisor_pids(processes, command) == [100]


def test_formal_runner_matching_requires_script_argv_and_formal_path() -> None:
    root = Path("/formal/v2")
    processes = {
        100: MODULE.ProcessInfo(
            pid=100,
            ppid=1,
            command=(
                "python",
                "/repo/fcooper_execute_measurement_row_v2.py",
                "--artifact-root",
                "/formal/v2/artifacts",
            ),
        ),
        200: MODULE.ProcessInfo(
            pid=200,
            ppid=1,
            command=(
                "bash",
                "-c",
                "grep fcooper_execute_measurement_row_v2.py /formal/v2",
            ),
        ),
        300: MODULE.ProcessInfo(
            pid=300,
            ppid=1,
            command=(
                "python",
                "/repo/fcooper_execute_measurement_row_v2.py",
                "--artifact-root",
                "/other/run",
            ),
        ),
    }
    assert MODULE.formal_runner_pids(processes, root) == [100]


def test_initial_supervisor_must_be_exact_root_process() -> None:
    command = ("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh")
    processes = {
        100: MODULE.ProcessInfo(pid=100, ppid=1, command=command),
        101: MODULE.ProcessInfo(pid=101, ppid=100, command=command),
    }
    MODULE.validate_initial_supervisor(100, processes, command)
    try:
        MODULE.validate_initial_supervisor(101, processes, command)
    except ValueError as exc:
        assert "root supervisor" in str(exc)
    else:
        raise AssertionError("child shell must not be accepted as initial supervisor")


def test_command_capture_rejects_unrelated_process(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_dir = proc_root / "77"
    proc_dir.mkdir(parents=True)
    (proc_dir / "cmdline").write_bytes(b"python\0other.py\0")

    try:
        MODULE.capture_supervisor_command(
            77,
            proc_root=proc_root,
        )
    except ValueError as exc:
        assert "required token" in str(exc)
    else:
        raise AssertionError("unrelated process command must be rejected")


def test_persisted_restart_count_is_loaded_for_same_command(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    command = ("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh")
    state.write_text(
        MODULE.serialize_state(
            status="monitor",
            command=command,
            restart_count=2,
            events=[{"event": "old"}],
        )
    )
    restart_count, events = MODULE.load_restart_history(state, command)
    assert restart_count == 2
    assert events == [{"event": "old"}]


def test_second_restart_guard_detects_new_supervisor() -> None:
    processes = {
        100: MODULE.ProcessInfo(
            pid=100,
            ppid=1,
            command=("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh"),
        ),
    }
    assert (
        MODULE.restart_guard_action(
            complete=False,
            processes=processes,
            formal_root=Path("/formal/v2"),
            tracked_supervisor_pid=100,
        )
        == "monitor"
    )


def test_second_restart_guard_ignores_reparented_shell_copy() -> None:
    processes = {
        101: MODULE.ProcessInfo(
            pid=101,
            ppid=1,
            command=("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh"),
        ),
    }
    assert (
        MODULE.restart_guard_action(
            complete=False,
            processes=processes,
            formal_root=Path("/formal/v2"),
            tracked_supervisor_pid=100,
        )
        == "restart"
    )


def test_tracked_pid_does_not_confuse_reparented_worker_for_supervisor() -> None:
    command = ("bash", "/repo/fcooper_stage6_five_arm_supervisor_v2.sh")
    processes = {
        101: MODULE.ProcessInfo(pid=101, ppid=1, command=command),
    }
    assert MODULE.tracked_pid_alive(processes, 100, None) is False
    assert MODULE.tracked_pid_alive(processes, 101, 0) is True


def test_tracked_pid_rejects_reused_process_start_time() -> None:
    processes = {
        100: MODULE.ProcessInfo(
            pid=100,
            ppid=1,
            command=("unrelated",),
            start_ticks=200,
        ),
    }
    assert MODULE.tracked_pid_alive(processes, 100, 100) is False
    assert MODULE.tracked_pid_alive(processes, 100, 200) is True


def test_discover_live_pid_uses_formal_pid_file(tmp_path: Path) -> None:
    formal_root = tmp_path / "formal"
    pid_file = formal_root / "controls/five_arm_supervisor.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text("200\n")
    processes = {
        200: MODULE.ProcessInfo(
            pid=200,
            ppid=199,
            command=(
                "bash",
                "/repo/fcooper_stage6_five_arm_supervisor_v2.sh",
                "--formal-root",
                str(formal_root),
            ),
        ),
    }
    assert (
        MODULE.discover_live_supervisor_pid(
            processes, pid_file=pid_file, formal_root=formal_root
        )
        == 200
    )
