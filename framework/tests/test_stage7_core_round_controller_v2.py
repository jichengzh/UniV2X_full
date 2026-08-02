from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "scripts" / "stage7_core_round_controller_v2.sh"


def _source() -> str:
    return CONTROLLER.read_text(encoding="utf-8")


def test_controller_is_argv_safe_thin_round_worker_entrypoint() -> None:
    source = _source()
    assert source.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in source
    assert "stage7_core_round_worker_v2.py" in source
    assert 'exec "$PYTHON" "$WORKER" "$@"' in source
    for forbidden in (
        "stage7_core_online_ablation_v2.py",
        "stage7_task_round_controller_v1.sh",
        "eval ",
        "bash -c",
        "CUDA_VISIBLE_DEVICES=",
        "latency_ms",
        "energy_j",
        "full AP",
    ):
        assert forbidden not in source


def test_controller_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(CONTROLLER)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
