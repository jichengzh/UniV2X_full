import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path("scripts/stage5_initialize_codriving_actual_v3.py")


def _module():
    spec = importlib.util.spec_from_file_location("codriving_actual_v3_init", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_initializer_exposes_only_two_codriving_tasks() -> None:
    module = _module()
    assert module.TASK_SPECS == (
        ("S5-COD-TVM", "codriving", "tvm_auto"),
        ("S5-COD-TRT", "codriving", "trt_engine"),
    )
    assert module.TRAINING_VIEW_POLICY == "initial_coldstart_only"


def test_fresh_initializer_rejects_non_gold_training_rows() -> None:
    module = _module()
    rows = [
        {"manifest_job_id": f"gold-{index}", "training_source": "initial_coldstart"}
        for index in range(176)
    ]
    assert len(module.freeze_initial_coldstart(rows)) == 176
    rows[-1] = {"manifest_job_id": "pilot", "training_source": "online_feedback"}
    with pytest.raises(ValueError, match="initial_coldstart"):
        module.freeze_initial_coldstart(rows)


def test_codriving_scheduler_freezes_fresh_four_round_actual_feedback_contract() -> None:
    source = Path("scripts/stage5_codriving_actual_v3_scheduler.sh").read_text()
    assert "for round in 0 1 2 3" in source
    assert "S5-COD-TVM S5-COD-TRT" in source
    assert "FEEDBACK_CONTRACT=actual_v3" in source
    assert "budget_consumed == 16" in source
    assert "stage5_single_target_search_v2_gold176_20260718/S5-COD" not in source
