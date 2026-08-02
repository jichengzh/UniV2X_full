from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).with_name(
    "stage7_speed_priority_takeover_v1.live.py"
)


class FakeRetryableOrchestrationError(RuntimeError):
    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        self.category = category


def _load_takeover_module():
    spec = importlib.util.spec_from_file_location("stage7_takeover_live", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _orchestrator(
    records: dict[tuple[str, int, int], dict[str, object]],
    *,
    rounds: tuple[int, ...] = (0,),
):
    def round_record(snapshot, variant, seed, round_index):
        del snapshot
        return dict(records.get((variant, seed, round_index), {}))

    return SimpleNamespace(
        _next_round_operation=lambda snapshot, seed: None,
        _round_record=round_record,
        CORE_VARIANTS=("candidate",),
        SEEDS=(2,),
        ROUNDS=rounds,
        PILOT_SEED=1,
        FREEZE_SOURCE_PLAN="freeze_source_plan",
        RESOLVE_SOURCE="resolve_source",
        BIND_EXACT="bind_exact",
        EXECUTE_MEASUREMENT="execute_measurement",
        FINALIZE_TERMINAL_BARRIER="finalize_terminal_barrier",
        RetryableOrchestrationError=FakeRetryableOrchestrationError,
    )


def _first_full_barrier() -> dict[tuple[str, int, int], dict[str, object]]:
    return {("full", 1, 0): {"barrier": True}}


def test_waits_retryably_when_incomplete_measurement_has_live_controller(
    tmp_path: Path,
) -> None:
    takeover = _load_takeover_module()
    records = {
        **_first_full_barrier(),
        (
            "candidate",
            2,
            0,
        ): {
            "request_frozen": True,
            "source_ready": True,
            "exact_bound": True,
            "terminal": False,
        },
    }
    scheduler_state = {
        "controllers": {
            "candidate:seed_2:round_0": {"status": "running"},
        }
    }
    status = tmp_path / "status"
    status.mkdir()
    (status / "scheduler_state.json").write_text(
        json.dumps(scheduler_state), encoding="utf-8"
    )
    orchestrator = _orchestrator(records)
    takeover.install_all_seed_fair_policy(
        orchestrator,
        v2_root=tmp_path,
        controller_reaper=lambda root: (),
        active_source_rounds_probe=lambda root: set(),
    )

    with pytest.raises(FakeRetryableOrchestrationError) as caught:
        orchestrator._next_round_operation({}, 1)

    assert caught.value.category == "controller_running"
    assert "candidate:seed_2:round_0" in str(caught.value)


def test_waits_retryably_when_incomplete_source_has_live_sidecar(
    tmp_path: Path,
) -> None:
    takeover = _load_takeover_module()
    records = {
        **_first_full_barrier(),
        (
            "candidate",
            2,
            0,
        ): {
            "request_frozen": True,
            "source_ready": False,
        },
    }
    orchestrator = _orchestrator(records)
    takeover.install_all_seed_fair_policy(
        orchestrator,
        v2_root=tmp_path,
        controller_reaper=lambda root: (),
        active_source_rounds_probe=lambda root: {("candidate", 2, 0)},
    )

    with pytest.raises(FakeRetryableOrchestrationError) as caught:
        orchestrator._next_round_operation({}, 1)

    assert caught.value.category == "controller_running"
    assert "candidate:seed_2:round_0" in str(caught.value)


def test_preserves_none_for_genuine_no_operation_without_live_work(
    tmp_path: Path,
) -> None:
    takeover = _load_takeover_module()
    records = {
        **_first_full_barrier(),
        ("candidate", 2, 0): {"barrier": False},
        ("candidate", 2, 1): {"barrier": False},
    }
    orchestrator = _orchestrator(records, rounds=(1,))
    takeover.install_all_seed_fair_policy(
        orchestrator,
        v2_root=tmp_path,
        controller_reaper=lambda root: (),
        active_source_rounds_probe=lambda root: set(),
    )

    assert orchestrator._next_round_operation({}, 1) is None


def test_selects_available_operation_while_another_round_has_live_work(
    tmp_path: Path,
) -> None:
    takeover = _load_takeover_module()
    records = {
        **_first_full_barrier(),
        ("blocked", 2, 0): {
            "request_frozen": True,
            "source_ready": False,
        },
        ("ready", 2, 0): {
            "request_frozen": True,
            "source_ready": True,
            "exact_bound": True,
            "terminal": False,
        },
    }
    orchestrator = _orchestrator(records)
    orchestrator.CORE_VARIANTS = ("blocked", "ready")
    takeover.install_all_seed_fair_policy(
        orchestrator,
        v2_root=tmp_path,
        controller_reaper=lambda root: (),
        active_source_rounds_probe=lambda root: {("blocked", 2, 0)},
    )

    assert orchestrator._next_round_operation({}, 1) == (
        "execute_measurement",
        "ready",
        2,
        0,
    )
