from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from framework.stage7 import actual_mode_v2
from framework.stage7 import deployment_bundle_v2


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _empty_projection() -> dict[str, object]:
    lineage = {
        "logical_request_sha256": "1" * 64,
        "selection_binding_sha256": "2" * 64,
        "cache_snapshot_sha256": "3" * 64,
        "cache_reveal_sha256": "4" * 64,
        "physical_request_sha256": "5" * 64,
        "source_resolution_plan_sha256": "6" * 64,
        "source_resolution_result_sha256": "7" * 64,
        "resolved_source_lock_sha256": "8" * 64,
        "executor_admission_sha256": "9" * 64,
        "deployment_bundle_sha256": "a" * 64,
        "rows": [],
    }
    unsigned = {
        "schema_version": "stage7_actual_v3_empty_physical_terminal_v2",
        "stage7_projection_lineage": lineage,
        "rows": [],
        "lineage_inputs": [],
        "gpu_subprocess_count": 0,
    }
    return {**unsigned, "empty_physical_terminal_sha256": _sha(unsigned)}


def _receipt(root: Path) -> dict[str, object]:
    unsigned = {
        "schema_version": "stage7_core_no_gpu_dry_run_v2",
        "formal_v2_root": str(root),
        "frozen_repo_root": str(Path(__file__).resolve().parents[2]),
        "formal_v2_gpu_jobs_launched": 0,
        "cuda_visible_devices": "",
        "deployment_bundle_sha256": "a" * 64,
        "deployment_primitive_pins_sha256": (
            deployment_bundle_v2.primitive_pins_sha256()
        ),
        "synthetic_non_measurement": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": False,
    }
    return {**unsigned, "dry_run_receipt_sha256": _sha(unsigned)}


def _projection() -> dict[str, object]:
    lineage = {
        "logical_request_sha256": "1" * 64,
        "selection_binding_sha256": "2" * 64,
        "cache_snapshot_sha256": "3" * 64,
        "cache_reveal_sha256": "4" * 64,
        "physical_request_sha256": "5" * 64,
        "source_resolution_plan_sha256": "6" * 64,
        "source_resolution_result_sha256": "7" * 64,
        "resolved_source_lock_sha256": "8" * 64,
        "executor_admission_sha256": "9" * 64,
        "deployment_bundle_sha256": "a" * 64,
        "rows": [{"candidate_id": "candidate-0"}],
    }
    unsigned = {
        "schema_version": "stage5_independent_validation_request_v1",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "b" * 64,
        "round_index": 0,
        "batch_size": 1,
        "required_metrics": ["latency_ms"],
        "real_h800_measurement_required": True,
        "independent_from_search_measurement": True,
        "row_sha256": {"candidate-0": "c" * 64},
        "rows": [{"row_id": "candidate-0"}],
        "stage7_projection_lineage": lineage,
    }
    return {**unsigned, "measurement_request_sha256": _sha(unsigned)}


def test_zero_miss_dry_run_writes_fixed_layout_and_replays_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    round_dir = root / "variants/full/seed_20260718/round_00"
    receipt = _receipt(root)
    calls: list[str] = []

    first = actual_mode_v2.execute_recoverable_attempt(
        round_dir=round_dir,
        physical_request_sha256="5" * 64,
        logical_request_sha256="1" * 64,
        projection=_empty_projection(),
        deployment_bundle_sha256="a" * 64,
        no_gpu_receipt=receipt,
        frozen_repo_root=Path(__file__).resolve().parents[2],
        dry_run=True,
        stage_runner=lambda stage, _: calls.append(stage),
    )
    terminal = Path(first["terminal_payload_path"])
    original = terminal.read_bytes()
    second = actual_mode_v2.execute_recoverable_attempt(
        round_dir=round_dir,
        physical_request_sha256="5" * 64,
        logical_request_sha256="1" * 64,
        projection=_empty_projection(),
        deployment_bundle_sha256="a" * 64,
        no_gpu_receipt=receipt,
        frozen_repo_root=Path(__file__).resolve().parents[2],
        dry_run=True,
        stage_runner=lambda stage, _: calls.append(stage),
    )

    execution = round_dir / "actual_v3_execution" / ("5" * 64)
    assert (execution / "execution_contract.json").is_file()
    assert (execution / "independent_request.json").is_file()
    assert (execution / "attempt_000/attempt_state.json").is_file()
    assert terminal == execution / "terminal_payload.json"
    assert terminal.read_bytes() == original
    assert first == second
    assert calls == []


def test_actual_mode_requires_authenticated_no_gpu_receipt(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    receipt = _receipt(root)
    receipt["dry_run_receipt_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="no-GPU receipt"):
        actual_mode_v2.execute_recoverable_attempt(
            round_dir=root / "variants/full/seed_20260718/round_00",
            physical_request_sha256="5" * 64,
            logical_request_sha256="1" * 64,
            projection=_empty_projection(),
            deployment_bundle_sha256="a" * 64,
            no_gpu_receipt=receipt,
            frozen_repo_root=Path(__file__).resolve().parents[2],
            dry_run=True,
            stage_runner=lambda *_: None,
        )


@pytest.mark.parametrize("interrupted_stage", actual_mode_v2.STAGES)
def test_each_stage_resumes_without_rerunning_completed_stages(
    tmp_path: Path, interrupted_stage: str
) -> None:
    root = tmp_path / "formal"
    round_dir = root / "variants/full/seed_20260718/round_00"
    calls: list[str] = []
    crashed = False

    def runner(stage: str, _: Path) -> dict[str, object]:
        nonlocal crashed
        calls.append(stage)
        if stage == interrupted_stage and not crashed:
            crashed = True
            raise RuntimeError("simulated process interruption")
        if stage == "final":
            return {
                "status": "success",
                "terminal_payload": {
                    "schema_version": "test_terminal",
                    "physical_terminal_batch_sha256": "d" * 64,
                    "gpu_subprocess_count": 4,
                },
            }
        return {"status": "success", "stage": stage}

    arguments = {
        "round_dir": round_dir,
        "physical_request_sha256": "5" * 64,
        "logical_request_sha256": "1" * 64,
        "projection": _projection(),
        "deployment_bundle_sha256": "a" * 64,
        "no_gpu_receipt": _receipt(root),
        "frozen_repo_root": Path(__file__).resolve().parents[2],
        "dry_run": False,
        "stage_runner": runner,
        "terminal_validator": lambda payload: dict(payload),
    }
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        actual_mode_v2.execute_recoverable_attempt(**arguments)
    result = actual_mode_v2.execute_recoverable_attempt(**arguments)

    interrupted_index = actual_mode_v2.STAGES.index(interrupted_stage)
    assert all(
        calls.count(stage) == 1 for stage in actual_mode_v2.STAGES[:interrupted_index]
    )
    assert calls.count(interrupted_stage) == 2
    assert all(
        calls.count(stage) == 1
        for stage in actual_mode_v2.STAGES[interrupted_index + 1 :]
    )
    assert result["retry_required"] is False


def test_infrastructure_failure_creates_new_attempt_with_parent_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    round_dir = root / "variants/full/seed_20260718/round_00"
    first = True

    def runner(stage: str, _: Path) -> dict[str, object]:
        nonlocal first
        if first:
            first = False
            return {"status": "infrastructure_failure", "reason": "lease drift"}
        if stage == "final":
            return {
                "status": "success",
                "terminal_payload": {
                    "schema_version": "test_terminal",
                    "physical_terminal_batch_sha256": "d" * 64,
                    "gpu_subprocess_count": 1,
                },
            }
        return {"status": "success"}

    arguments = {
        "round_dir": round_dir,
        "physical_request_sha256": "5" * 64,
        "logical_request_sha256": "1" * 64,
        "projection": _projection(),
        "deployment_bundle_sha256": "a" * 64,
        "no_gpu_receipt": _receipt(root),
        "frozen_repo_root": Path(__file__).resolve().parents[2],
        "dry_run": False,
        "stage_runner": runner,
        "terminal_validator": lambda payload: dict(payload),
    }
    failed = actual_mode_v2.execute_recoverable_attempt(**arguments)
    completed = actual_mode_v2.execute_recoverable_attempt(**arguments)
    execution = Path(completed["execution_root"])
    state = json.loads((execution / "attempt_001/attempt_state.json").read_text())

    assert failed["retry_required"] is True
    assert completed["retry_required"] is False
    assert state["parent_failure_sha256"] == failed["attempt_failure_sha256"]


def test_stage_receipt_drift_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "formal"
    round_dir = root / "variants/full/seed_20260718/round_00"

    def runner(stage: str, _: Path) -> dict[str, object]:
        if stage == "performance":
            raise RuntimeError("stop after quant")
        return {"status": "success"}

    arguments = {
        "round_dir": round_dir,
        "physical_request_sha256": "5" * 64,
        "logical_request_sha256": "1" * 64,
        "projection": _projection(),
        "deployment_bundle_sha256": "a" * 64,
        "no_gpu_receipt": _receipt(root),
        "frozen_repo_root": Path(__file__).resolve().parents[2],
        "dry_run": False,
        "stage_runner": runner,
        "terminal_validator": lambda payload: dict(payload),
    }
    with pytest.raises(RuntimeError):
        actual_mode_v2.execute_recoverable_attempt(**arguments)
    receipt_path = (
        round_dir / "actual_v3_execution" / ("5" * 64) / "attempt_000/quant/quant.json"
    )
    receipt_path.write_text("{}\n")

    with pytest.raises(ValueError, match="stage receipt"):
        actual_mode_v2.execute_recoverable_attempt(**arguments)


def test_symlinked_stage_directory_outside_attempt_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal"
    round_dir = root / "variants/full/seed_20260718/round_00"
    attempt = round_dir / "actual_v3_execution" / ("5" * 64) / "attempt_000"
    external = tmp_path / "external"
    external.mkdir()
    attempt.mkdir(parents=True)
    (attempt / "quant").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        actual_mode_v2.execute_recoverable_attempt(
            round_dir=round_dir,
            physical_request_sha256="5" * 64,
            logical_request_sha256="1" * 64,
            projection=_projection(),
            deployment_bundle_sha256="a" * 64,
            no_gpu_receipt=_receipt(root),
            frozen_repo_root=Path(__file__).resolve().parents[2],
            dry_run=False,
            stage_runner=lambda *_: {"status": "success"},
            terminal_validator=lambda payload: dict(payload),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("frozen_repo_root", "/wrong/frozen/repo"),
        ("eligible_for_formal_finalization", True),
        ("eligible_for_cache_append", True),
        ("actual_v3_hardware_evidence", True),
        ("synthetic_non_measurement", False),
    ),
)
def test_no_gpu_receipt_binds_frozen_root_and_truth_flags(
    tmp_path: Path, field: str, value: object
) -> None:
    root = tmp_path / "formal"
    receipt = _receipt(root)
    receipt[field] = value
    unsigned = {
        key: item for key, item in receipt.items() if key != "dry_run_receipt_sha256"
    }
    receipt["dry_run_receipt_sha256"] = _sha(unsigned)

    with pytest.raises(ValueError, match="no-GPU receipt"):
        actual_mode_v2.execute_recoverable_attempt(
            round_dir=root / "variants/full/seed_20260718/round_00",
            physical_request_sha256="5" * 64,
            logical_request_sha256="1" * 64,
            projection=_empty_projection(),
            deployment_bundle_sha256="a" * 64,
            no_gpu_receipt=receipt,
            frozen_repo_root=Path(__file__).resolve().parents[2],
            dry_run=True,
            stage_runner=lambda *_: None,
        )
