from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts.fcooper_gpu_exclusivity_gate_v1 import (
    GuardError,
    GpuBusyError,
    TelemetryError,
    _parse_compute_process_rows,
    check_gpu_exclusive,
    guard_command,
    wait_for_gpu_exclusive,
    write_audit,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def sample(processes: list[dict[str, object]]) -> dict[str, object]:
    return {
        "gpu_index": 7,
        "gpu_uuid": "GPU-test",
        "processes": processes,
    }


def test_waits_for_busy_gpu_and_requires_a_quiet_window() -> None:
    clock = FakeClock()
    calls = 0

    def sampler(_: int) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return (
            sample([{"pid": 11, "used_memory_mib": 1024}])
            if calls == 1
            else sample([])
        )

    audit = wait_for_gpu_exclusive(
        gpu_index=7,
        timeout_seconds=30,
        poll_seconds=2,
        quiet_seconds=5,
        sampler=sampler,
        clock=clock.now,
        sleeper=clock.sleep,
    )

    assert audit["status"] == "acquired"
    assert audit["wait_seconds"] == pytest.approx(7.0)
    assert [row["state"] for row in audit["observations"]] == [
        "busy",
        "empty_candidate",
        "acquired",
    ]
    assert calls > 3


def test_wait_times_out_without_silent_fallback() -> None:
    clock = FakeClock()

    with pytest.raises(GpuBusyError) as error:
        wait_for_gpu_exclusive(
            gpu_index=7,
            timeout_seconds=4,
            poll_seconds=2,
            quiet_seconds=1,
            sampler=lambda _: sample([{"pid": 12, "used_memory_mib": 2048}]),
            clock=clock.now,
            sleeper=clock.sleep,
        )

    assert error.value.audit["status"] == "timeout_busy"
    assert error.value.audit["gpu_index"] == 7


def test_quiet_window_cannot_cross_hard_timeout() -> None:
    clock = FakeClock()

    with pytest.raises(GpuBusyError) as error:
        wait_for_gpu_exclusive(
            gpu_index=7,
            timeout_seconds=4,
            poll_seconds=2,
            quiet_seconds=5,
            sampler=lambda _: sample([]),
            clock=clock.now,
            sleeper=clock.sleep,
        )

    assert error.value.audit["status"] == "timeout_quiet_window"
    assert clock.value == 100.0


def test_empty_busy_flapping_honors_hard_timeout() -> None:
    clock = FakeClock()
    calls = 0

    def sampler(_: int) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return (
            sample([])
            if calls % 2
            else sample([{"pid": 14, "used_memory_mib": 64}])
        )

    with pytest.raises(GpuBusyError) as error:
        wait_for_gpu_exclusive(
            gpu_index=7,
            timeout_seconds=4,
            poll_seconds=1,
            quiet_seconds=2,
            sampler=sampler,
            clock=clock.now,
            sleeper=clock.sleep,
        )

    assert error.value.audit["status"] in {
        "timeout_busy",
        "timeout_quiet_window",
    }
    assert clock.value <= 104.0


def test_post_check_fails_closed_when_foreign_process_is_present() -> None:
    with pytest.raises(GpuBusyError) as error:
        check_gpu_exclusive(
            gpu_index=7,
            sampler=lambda _: sample([{"pid": 13, "used_memory_mib": 512}]),
        )

    assert error.value.audit["status"] == "busy"
    assert error.value.audit["processes"][0]["pid"] == 13


def test_audit_is_written_as_stable_json(tmp_path: Path) -> None:
    output = tmp_path / "gpu7_preflight.json"
    payload = {
        "schema_version": "fcooper_gpu_exclusivity_gate_v1",
        "status": "acquired",
        "gpu_index": 7,
        "observations": [],
    }

    write_audit(output, payload)

    assert json.loads(output.read_text()) == payload
    assert not list(tmp_path.glob("*.tmp"))


def test_audit_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "gpu7_preflight.json"
    write_audit(output, {"status": "first"})

    with pytest.raises(FileExistsError):
        write_audit(output, {"status": "second"})

    assert json.loads(output.read_text()) == {"status": "first"}


def test_compute_process_parser_rejects_malformed_target_rows() -> None:
    with pytest.raises(TelemetryError, match="malformed compute-app row"):
        _parse_compute_process_rows(
            [["GPU-test", "123", "python"]],
            target_uuid="GPU-test",
        )

    with pytest.raises(TelemetryError, match="used_memory"):
        _parse_compute_process_rows(
            [["GPU-test", "123", "python", "N/A"]],
            target_uuid="GPU-test",
        )


def test_compute_process_parser_ignores_well_formed_other_gpu_rows() -> None:
    processes = _parse_compute_process_rows(
        [
            ["GPU-other", "99", "python", "1024"],
            ["GPU-test", "123", "python", "2048"],
        ],
        target_uuid="GPU-test",
    )

    assert processes == [
        {
            "pid": 123,
            "process_name": "python",
            "used_memory_mib": 2048,
        }
    ]


def test_guard_terminates_workload_on_runtime_interference(
    tmp_path: Path,
) -> None:
    samples = iter(
        [
            sample([]),
            sample([{"pid": 999999, "used_memory_mib": 128}]),
        ]
    )

    with pytest.raises(GuardError) as error:
        guard_command(
            gpu_index=7,
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=5,
            poll_seconds=1,
            quiet_seconds=0,
            monitor_seconds=0.01,
            runtime_timeout_seconds=5,
            cwd=None,
            environment={},
            log_path=tmp_path / "guard.log",
            sampler=lambda _: next(samples),
        )

    assert error.value.audit["status"] == "runtime_interference"
    assert error.value.audit["foreign_processes"][0]["pid"] == 999999


def test_guard_records_success_and_command_log(tmp_path: Path) -> None:
    audit = guard_command(
        gpu_index=7,
        command=[sys.executable, "-c", "print('guarded')"],
        timeout_seconds=5,
        poll_seconds=1,
        quiet_seconds=0,
        monitor_seconds=0.01,
        runtime_timeout_seconds=5,
        cwd=None,
        environment={},
        log_path=tmp_path / "guard.log",
        sampler=lambda _: sample([]),
    )

    assert audit["status"] == "completed_exclusive"
    assert audit["runtime_seconds"] > 0
    assert (tmp_path / "guard.log").read_text().strip() == "guarded"


def test_guard_runtime_timeout_kills_hung_workload(tmp_path: Path) -> None:
    with pytest.raises(GuardError) as error:
        guard_command(
            gpu_index=7,
            command=[sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=5,
            poll_seconds=1,
            quiet_seconds=0,
            monitor_seconds=0.01,
            runtime_timeout_seconds=0.05,
            cwd=None,
            environment={},
            log_path=tmp_path / "guard.log",
            sampler=lambda _: sample([]),
        )

    assert error.value.audit["status"] == "runtime_timeout"
