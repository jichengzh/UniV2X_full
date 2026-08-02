from __future__ import annotations

from pathlib import Path

import pytest

from scripts.fcooper_remeasure_legacy_gpu7_timing_v2 import (
    _validate_old_audit,
    rewrite_command_outputs,
    validate_timing_guard,
)


def test_rewrite_native_repeat_output() -> None:
    command = ["python", "native.py", "--output-json", "/old/repeat.json"]
    rewritten = rewrite_command_outputs(
        command,
        output_root=Path("/new/native_0"),
        replacement_engine=None,
    )
    assert rewritten[-1] == "/new/native_0/repeat.json"


def test_rewrite_trt_repeat_outputs() -> None:
    command = [
        "python",
        "trt.py",
        "--artifact-dir",
        "/old/engine",
        "--out",
        "/old/performance.json",
    ]
    rewritten = rewrite_command_outputs(
        command,
        output_root=Path("/new/trt_0"),
        replacement_engine=None,
    )
    assert rewritten[rewritten.index("--artifact-dir") + 1] == "/new/trt_0/engine"
    assert rewritten[rewritten.index("--out") + 1] == "/new/trt_0/performance.json"


def test_rewrite_ap_output_and_engine() -> None:
    command = [
        "python",
        "ap.py",
        "--engine",
        "/old/compiled.engine",
        "--output-json",
        "/old/ap.json",
    ]
    rewritten = rewrite_command_outputs(
        command,
        output_root=Path("/new/ap"),
        replacement_engine=Path("/new/trt_0/engine/compiled.engine"),
    )
    assert (
        rewritten[rewritten.index("--engine") + 1]
        == "/new/trt_0/engine/compiled.engine"
    )
    assert rewritten[rewritten.index("--output-json") + 1] == "/new/ap/ap.json"


def test_validate_timing_guard_requires_exclusive_gpu7_and_runtime_match() -> None:
    guard = {
        "schema_version": "fcooper_gpu_exclusivity_gate_v1",
        "status": "completed_exclusive",
        "evidence_scope": "sampled_process_exclusivity",
        "gpu_index": 7,
        "return_code": 0,
        "runtime_sample_count": 4,
        "runtime_seconds": 12.5,
        "runtime_observations": [],
        "residual_processes": [],
        "command": ["python", "runner.py"],
    }
    validate_timing_guard(guard, expected_runtime=12.5)
    with pytest.raises(ValueError, match="runtime"):
        validate_timing_guard({**guard, "runtime_seconds": 13.0}, expected_runtime=12.5)
    with pytest.raises(ValueError, match="exclusive"):
        validate_timing_guard(
            {**guard, "residual_processes": [{"pid": 1}]},
            expected_runtime=12.5,
        )
    with pytest.raises(ValueError, match="exclusive"):
        validate_timing_guard(
            {**guard, "runtime_observations": [{"pid": 2}]},
            expected_runtime=12.5,
        )


def test_validate_old_audit_requires_sampled_exclusivity(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        """{
          "schema_version": "fcooper_gpu_exclusivity_gate_v1",
          "status": "completed_exclusive",
          "gpu_index": 7,
          "return_code": 0,
          "command": ["python", "runner.py"]
        }"""
    )
    with pytest.raises(ValueError, match="invalid"):
        _validate_old_audit(path)
