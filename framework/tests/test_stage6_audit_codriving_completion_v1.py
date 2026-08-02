from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.stage6_audit_codriving_completion_v1 import (
    _audit_formal_plan,
    _audit_independent_measurements,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _formal_root(tmp_path: Path, *, cost_policy: str = "parameter_bits_and_bitops") -> Path:
    selected = [[24, 80, 128, "fp16"], [16, 32, 64, "int8"]]
    backend_plans = {
        backend: {
            "compression_only": selected,
            "compress_then_tune": selected[:1],
            "tune_then_compress": selected,
        }
        for backend in ("tvm", "trt")
    }
    _write(
        tmp_path / "stage6_formal_execution_plan_v1.json",
        {
            "schema_version": "stage6_formal_execution_plan_v1",
            "passed": True,
            "target_model": "codriving",
            "effective_candidate_pool_size": 648,
            "hardware_blind_backend_labels_used": False,
            "hardware_blind_cost_policy": cost_policy,
            "plan_sha256": "plan-contract-sha",
            "backend_plans": backend_plans,
        },
    )
    for backend, arms in backend_plans.items():
        for arm, genomes in arms.items():
            _write(
                tmp_path / backend / f"{arm}_candidate_plan.json",
                {
                    "schema_version": "stage6_formal_arm_candidate_plan_v1",
                    "backend": backend,
                    "arm_id": arm,
                    "plan_sha256": "plan-contract-sha",
                    "rows": [
                        {"width": genome[:3], "q_mode": genome[3]}
                        for genome in genomes
                    ],
                },
            )
    return tmp_path


def test_audits_bitcost_policy_and_frozen_candidate_bindings(tmp_path: Path) -> None:
    result = _audit_formal_plan(_formal_root(tmp_path))
    assert result["hardware_blind_cost_policy"] == "parameter_bits_and_bitops"
    assert result["effective_candidate_pool_size"] == 648
    assert result["candidate_plan_count"] == 6
    assert result["selected_q_mode_counts"] == {"fp16": 6, "int8": 4}


def test_rejects_old_flops_only_ranking_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bit-cost policy"):
        _audit_formal_plan(_formal_root(tmp_path, cost_policy="parameter_and_flops"))


def test_requires_all_seven_independent_validation_audits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected seven"):
        _audit_independent_measurements(tmp_path)
