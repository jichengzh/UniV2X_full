from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts import stage7_h800_runtime_v2 as runtime


@dataclass(frozen=True)
class Process:
    pid: int
    command: tuple[str, ...]


def test_query_gpu_models_parses_uuid_model_rows() -> None:
    class Result:
        stdout = "GPU-a, NVIDIA H800\nGPU-b, NVIDIA H800 PCIe\n"

    assert runtime.query_gpu_models(lambda *args, **kwargs: Result()) == {
        "GPU-a": "NVIDIA H800",
        "GPU-b": "NVIDIA H800 PCIe",
    }


def test_query_gpu_models_rejects_malformed_rows() -> None:
    class Result:
        stdout = "not-a-pair\n"

    with pytest.raises(ValueError, match="malformed"):
        runtime.query_gpu_models(lambda *args, **kwargs: Result())


def test_discover_process_reservations_supports_single_and_pool_flags() -> None:
    processes = (
        Process(41, ("python", "worker.py", "--gpu", "7")),
        Process(42, ("python", "worker.py", "--gpu-uuids", "GPU-a,GPU-b")),
        Process(43, ("python", "worker.py", "--gpus", "0,3")),
    )

    result = runtime.discover_process_reservations(processes)

    assert set(result) == {"index:7", "GPU-a", "GPU-b", "index:0", "index:3"}
    assert all(
        value[0].startswith("active_process_reservation:") for value in result.values()
    )


def test_discover_process_reservations_ignores_empty_process_command() -> None:
    assert runtime.discover_process_reservations((Process(44, ()),)) == {}


def test_redacted_command_never_emits_secret_values() -> None:
    assert runtime.redacted_command(
        ("runner", "--password", "secret", "--token=hidden", "--gpu", "1"),
        ("password", "token"),
    ) == ["runner", "--password", "<redacted>", "--token", "--gpu", "1"]


def test_resource_lock_is_nonblocking_and_recoverable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runtime, "Path", lambda _value: tmp_path)
    first = runtime.acquire_resource_lock("source:group-a")
    assert first is not None
    assert runtime.acquire_resource_lock("source:group-a") is None
    first.release()
    recovered = runtime.acquire_resource_lock("source:group-a")
    assert recovered is not None
    recovered.release()
