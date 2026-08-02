from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.stage2_route_b_int8_auto_decomp import (
    audit_metaschedule_database,
    make_tuning_budget_config,
    persist_runtime_weights,
)


def test_zero_budget_preserves_default_tensorization_contract() -> None:
    assert make_tuning_budget_config(0) == {
        "enabled": False,
        "requested_trials": 0,
        "max_trials_global": 0,
        "max_trials_per_task": 0,
        "num_trials_per_iter": 0,
        "allocation_policy": "default_per_primfunc_tensorization_no_metaschedule",
    }


def test_positive_budget_is_passed_as_a_strict_global_cap() -> None:
    assert make_tuning_budget_config(17) == {
        "enabled": True,
        "requested_trials": 17,
        "max_trials_global": 17,
        "max_trials_per_task": 17,
        "num_trials_per_iter": 1,
        "allocation_policy": "metaschedule_global_gradient_with_per_task_cap",
    }
    with pytest.raises(ValueError, match="non-negative"):
        make_tuning_budget_config(-1)


def test_database_audit_counts_attempted_and_valid_trials(tmp_path: Path) -> None:
    work_dir = tmp_path / "database"
    work_dir.mkdir()
    workload = work_dir / "database_workload.json"
    records = work_dir / "database_tuning_record.json"
    workload.write_text("[0, {}]\n", encoding="utf-8")
    rows = [
        [0, [["trace"], [0.001], {"arch": "sm_90"}, []]],
        [0, [["trace"], [10_000_000_000], {"arch": "sm_90"}, []]],
        [0, [["trace"], [0.002, 0.003], {"arch": "sm_90"}, []]],
    ]
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    audit = audit_metaschedule_database(work_dir, requested_trials=3)

    assert audit["attempted_trials"] == 3
    assert audit["valid_trials"] == 2
    assert audit["database_workload_path"] == str(workload)
    assert audit["database_tuning_record_path"] == str(records)
    assert len(audit["database_workload_sha256"]) == 64
    assert len(audit["database_tuning_record_sha256"]) == 64


def test_database_audit_rejects_budget_overrun(tmp_path: Path) -> None:
    (tmp_path / "database_workload.json").write_text("[0, {}]\n", encoding="utf-8")
    rows = [[0, [["trace"], [0.001], {}, []]]] * 2
    (tmp_path / "database_tuning_record.json").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        audit_metaschedule_database(tmp_path, requested_trials=1)


def test_runtime_weight_npz_contains_only_planned_initializers(tmp_path: Path) -> None:
    runtime_arg_plan = [
        {"role": "graph_input", "arg_name": "spatial_features"},
        {"role": "weight_input", "arg_name": "weight_1"},
        {"role": "bias_input", "arg_name": "bias_1"},
        {"role": "graph_output", "arg_name": "output"},
    ]
    values = {
        "weight_1": np.arange(8, dtype=np.int8).reshape(2, 4),
        "bias_1": np.arange(2, dtype=np.int32),
    }
    output = tmp_path / "runtime_weights_int8.npz"

    audit = persist_runtime_weights(
        output,
        runtime_arg_plan=runtime_arg_plan,
        weight_runtime_values=values,
    )

    with np.load(output, allow_pickle=False) as archive:
        assert archive.files == ["bias_1", "weight_1"]
        np.testing.assert_array_equal(archive["weight_1"], values["weight_1"])
        np.testing.assert_array_equal(archive["bias_1"], values["bias_1"])
        assert "spatial_features" not in archive.files
    assert audit["path"] == str(output)
    assert len(audit["sha256"]) == 64
    assert audit["array_names"] == ["bias_1", "weight_1"]
    assert audit["arrays"]["weight_1"] == {
        "shape": [2, 4],
        "dtype": "int8",
        "nbytes": 8,
    }


def test_runtime_weight_npz_rejects_plan_value_drift(tmp_path: Path) -> None:
    plan = [{"role": "weight_input", "arg_name": "expected_weight"}]
    with pytest.raises(ValueError, match="runtime weight contract mismatch"):
        persist_runtime_weights(
            tmp_path / "runtime_weights_int8.npz",
            runtime_arg_plan=plan,
            weight_runtime_values={"unexpected_weight": np.ones((1,), dtype=np.int8)},
        )
