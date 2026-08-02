from framework.stage6.paper_table_v1 import build_backend_tables


def _point(width, ap, latency, energy, *, trusted=True):
    return {
        "config": [*width, "fp16"],
        "AP70": ap,
        "latency_ms": latency,
        "energy_j": energy,
        "terminal_status": "measured_success_gold",
        "independent_validation_passed": trusted,
        "evidence_sha_verified": trusted,
    }


def test_threshold_selection_uses_energy_within_one_percent_latency() -> None:
    baseline = {"AP70": 0.63, "latency_ms": 3.0, "energy_j": 1.0}
    arms = {
        "compression_only": {
            "status": "complete",
            "independent_validation_complete": True,
            "outer_genomes": 16,
            "failure_count": 0,
            "tuning_trials": 0,
            "gpu_hours": 1.0,
            "wallclock_s": 10.0,
            "points": [
                _point([16, 32, 64], 0.59, 1.000, 0.50),
                _point([24, 48, 96], 0.60, 1.009, 0.20),
                _point([32, 64, 128], 0.61, 1.020, 0.10),
            ],
        }
    }
    result = build_backend_tables(
        backend="trt",
        baseline=baseline,
        arms=arms,
        deltas=(0.05, 0.10),
        required_arms=("compression_only",),
    )
    selected = result["tables"]["delta_0.05"][1]
    assert selected["config"] == [24, 48, 96, "fp16"]
    assert selected["latency_ms"] == 1.009
    assert selected["speedup"] == 3.0 / 1.009
    assert selected["outcome"] == "selected"


def test_untrusted_and_below_floor_points_are_not_selected() -> None:
    baseline = {"AP70": 0.63, "latency_ms": 3.0, "energy_j": 1.0}
    arms = {
        "joint_shcosearch": {
            "status": "complete",
            "independent_validation_complete": True,
            "outer_genomes": 16,
            "failure_count": 1,
            "tuning_trials": 64,
            "gpu_hours": 2.0,
            "wallclock_s": 20.0,
            "points": [
                _point([16, 32, 64], 0.60, 0.5, 0.1, trusted=False),
                _point([24, 48, 96], 0.57, 0.7, 0.2),
            ],
        }
    }
    result = build_backend_tables(
        backend="tvm",
        baseline=baseline,
        arms=arms,
        deltas=(0.05,),
        required_arms=("joint_shcosearch",),
    )
    row = result["tables"]["delta_0.05"][1]
    assert row["selection_status"] == "no_feasible_point"
    assert row["outcome"] == "no_point_satisfies_ap_floor"
    assert row["config"] == [24, 48, 96, "fp16"]
    assert row["AP70"] == 0.57
    assert row["ap_constraint_violated"] is True
    assert row["failure_rate"] == 1 / 16


def test_explicit_failure_is_distinct_from_no_feasible_ap_point() -> None:
    result = build_backend_tables(
        backend="tvm",
        baseline={"AP70": 0.63, "latency_ms": 3.0, "energy_j": 1.0},
        arms={
            "schedule_only": {
                "status": "complete_failure",
                "failure_evidence_sha_verified": True,
                "failure_reason": "base_fp32_backend_numerical_contract_failure",
                "outer_genomes": 1,
                "failure_count": 1,
                "points": [],
            }
        },
        deltas=(0.05,),
        required_arms=("schedule_only",),
    )
    row = result["tables"]["delta_0.05"][1]
    assert row["selection_status"] == "feasibility_failure"
    assert row["failure_reason"] == "base_fp32_backend_numerical_contract_failure"
    assert row["outcome"] == "feasibility_failure:base_fp32_backend_numerical_contract_failure"
    assert row["ap_constraint_violated"] is None


def test_table_is_not_paper_ready_until_every_arm_is_terminal() -> None:
    result = build_backend_tables(
        backend="trt",
        baseline={"AP70": 0.63, "latency_ms": 3.0, "energy_j": 1.0},
        arms={},
        deltas=(0.05,),
        required_arms=("compression_only", "joint_shcosearch"),
    )
    assert result["paper_ready"] is False
    assert result["missing_arms"] == ["compression_only", "joint_shcosearch"]


def test_original_default_uses_backend_specific_baseline_contract() -> None:
    result = build_backend_tables(
        backend="tvm",
        baseline={
            "backend": "tvm",
            "schedule_policy": "tvm_default_zero_trial",
            "evidence_origin": "stage6_tvm_default_zero_trial_measurement",
            "AP70": 0.631,
            "latency_ms": 56.3,
            "energy_j": 20.2,
        },
        arms={},
        deltas=(0.05,),
        required_arms=(),
    )

    row = result["tables"]["delta_0.05"][0]
    assert row["backend"] == "tvm"
    assert row["latency_ms"] == 56.3
    assert row["tuning_trials"] == 0
    assert row["schedule_policy"] == "tvm_default_zero_trial"
    assert row["evidence_origin"] == "stage6_tvm_default_zero_trial_measurement"
