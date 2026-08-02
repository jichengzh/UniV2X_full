from __future__ import annotations

from pathlib import Path

from framework.stage7 import actual_mode_v2
from framework.tests import test_stage7_actual_mode_integration_v2 as mode_fixture
from framework.tests import test_stage7_actual_pipeline_v2 as pipeline_fixture


def test_fake_actual_pipeline_closes_recovery_layout_without_gpu(
    tmp_path: Path,
) -> None:
    projection = mode_fixture._projection()
    projection["rows"] = [
        {
            "row_id": "candidate-0",
            "candidate_id": "candidate-0",
            "group_id": "group-0",
            "q_mode": "int8",
        }
    ]
    unsigned = {
        key: value
        for key, value in projection.items()
        if key != "measurement_request_sha256"
    }
    projection["measurement_request_sha256"] = mode_fixture._sha(unsigned)
    fakes = pipeline_fixture._PipelineFakes()
    stage_runner = pipeline_fixture._build_runner(
        tmp_path,
        1,
        fakes,
        projection=projection,
    )
    root = tmp_path / "formal"
    round_dir = root / "variants/full/seed_20260718/round_00"
    arguments = {
        "round_dir": round_dir,
        "physical_request_sha256": "5" * 64,
        "logical_request_sha256": "1" * 64,
        "projection": projection,
        "deployment_bundle_sha256": "a" * 64,
        "no_gpu_receipt": mode_fixture._receipt(root),
        "frozen_repo_root": Path(__file__).resolve().parents[2],
        "dry_run": False,
        "stage_runner": stage_runner,
        "terminal_validator": lambda payload: dict(payload),
    }

    first = actual_mode_v2.execute_recoverable_attempt(**arguments)
    terminal = Path(first["terminal_payload_path"])
    original = terminal.read_bytes()
    second = actual_mode_v2.execute_recoverable_attempt(**arguments)

    assert first == second
    assert terminal.read_bytes() == original
    assert [name for name, _candidate in fakes.calls] == [
        "quant",
        "planner",
        "performance",
        "ap_plan",
        "sanity",
        "full",
        "finalizer_evidence",
        "finalizer",
    ]
    assert not list(root.rglob("__pycache__"))
