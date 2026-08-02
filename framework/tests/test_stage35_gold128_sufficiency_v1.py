from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold128_sufficiency_v1 as stage35  # noqa: E402


class Stage35Gold128SufficiencyV1Tests(unittest.TestCase):
    def test_pre_registered_learning_and_transfer_sizes(self) -> None:
        self.assertEqual(stage35.LEARNING_SIZES, (6, 12, 18, 24, 26))
        self.assertEqual(stage35.K_GROUPS, (2, 4, 6, 8))

    def test_hypervolume_3d_ignores_dominated_points(self) -> None:
        point = (0.2, 0.3, 0.4)
        expected = 0.8 * 0.7 * 0.6
        self.assertAlmostEqual(stage35.hypervolume_3d([point], (1.0, 1.0, 1.0)), expected)
        self.assertAlmostEqual(
            stage35.hypervolume_3d([point, (0.5, 0.6, 0.7)], (1.0, 1.0, 1.0)),
            expected,
        )

    def test_selection_quality_has_zero_regret_for_perfect_predictions(self) -> None:
        truth = [
            {"id": "a", "ap70": 0.8, "latency_ms": 2.0, "energy_j": 2.0},
            {"id": "b", "ap70": 0.7, "latency_ms": 1.0, "energy_j": 1.0},
            {"id": "c", "ap70": 0.6, "latency_ms": 3.0, "energy_j": 3.0},
        ]
        predictions = {row["id"]: {key: row[key] for key in ("ap70", "latency_ms", "energy_j")} for row in truth}
        result = stage35.selection_quality(truth, predictions)
        self.assertEqual(result["pareto_recall"], 1.0)
        self.assertAlmostEqual(result["hv_regret"], 0.0)

    def test_interval_coverage_reports_row_and_complete_group_rates(self) -> None:
        records = [
            {"group_id": "g1", "truth": 1.0, "lower": 0.5, "upper": 1.5},
            {"group_id": "g1", "truth": 2.0, "lower": 1.5, "upper": 2.5},
            {"group_id": "g2", "truth": 3.0, "lower": 2.5, "upper": 3.5},
            {"group_id": "g2", "truth": 4.0, "lower": 4.1, "upper": 4.5},
        ]
        result = stage35.interval_coverage(records)
        self.assertEqual(result["coverage_90"], 0.75)
        self.assertEqual(result["fully_covered_group_rate_90"], 0.5)

    def test_conformalize_intervals_uses_model_specific_calibration_error(self) -> None:
        calibration = [
            {"model": "pyramid", "group_id": "p1", "truth": 3.0, "lower": 1.0, "upper": 2.0},
            {"model": "codriving", "group_id": "c1", "truth": 1.1, "lower": 1.0, "upper": 1.2},
        ]
        test = [
            {"model": "pyramid", "lower": 4.0, "upper": 5.0},
            {"model": "codriving", "lower": 4.0, "upper": 5.0},
        ]

        result = stage35.conformalize_intervals(calibration, test, coverage=0.90)

        self.assertEqual(result[0]["lower"], 3.0)
        self.assertEqual(result[0]["upper"], 6.0)
        self.assertEqual(result[1]["lower"], 4.0)
        self.assertEqual(result[1]["upper"], 5.0)

    def test_conformalize_intervals_calibrates_complete_group_error(self) -> None:
        calibration = [
            {"model": "pyramid", "group_id": "g1", "truth": 1.0, "lower": 1.0, "upper": 1.0},
            {"model": "pyramid", "group_id": "g1", "truth": 2.0, "lower": 1.0, "upper": 1.0},
            {"model": "pyramid", "group_id": "g2", "truth": 1.0, "lower": 1.0, "upper": 1.0},
            {"model": "pyramid", "group_id": "g2", "truth": 1.0, "lower": 1.0, "upper": 1.0},
        ]
        test = [{"model": "pyramid", "lower": 4.0, "upper": 5.0}]

        result = stage35.conformalize_intervals(calibration, test, coverage=0.50)

        self.assertEqual(result[0]["conformal_correction"], 1.0)

    def test_residual_calibrator_learns_feature_conditioned_correction(self) -> None:
        prediction = stage35.residual_calibration_prediction(
            np.asarray([[-1.0], [1.0]]),
            np.asarray([-1.0, 1.0]),
            np.asarray([[-1.0], [1.0]]),
            np.asarray([0.0, 0.0]),
        )

        self.assertLess(prediction[0], 0.0)
        self.assertGreater(prediction[1], 0.0)
        extrapolated = stage35.residual_calibration_prediction(
            np.asarray([[-1.0], [1.0]]),
            np.asarray([-1.0, 1.0]),
            np.asarray([[100.0]]),
            np.asarray([0.0]),
        )
        self.assertGreaterEqual(extrapolated[0], -1.0)
        self.assertLessEqual(extrapolated[0], 1.0)

    def test_validate_dataset_contract_rejects_duplicate_result_id(self) -> None:
        profiles = {
            "tvm_auto": "h800-tvm-probe-conditioned-v3",
            "trt_engine": "h800-trt-probe-conditioned-v3",
        }
        rows = []
        jobs = []
        for model in ("pyramid", "codriving"):
            for group_index in range(16):
                group_id = f"{model}|g{group_index}"
                split = "locked_holdout" if group_index >= 13 else "train"
                for backend in ("tvm_auto", "trt_engine"):
                    for q_mode in ("fp16", "int8"):
                        job_id = f"{group_id}|q={q_mode}|profile={profiles[backend]}"
                        common = {
                            "group_id": group_id,
                            "model": model,
                            "width": [16, 32, 64],
                            "q_mode": q_mode,
                            "dispatch_key": backend,
                            "capability_profile_id": profiles[backend],
                        }
                        jobs.append({"job_id": job_id, "split": split, "width_stratum": "test", **common})
                        rows.append({
                            "manifest_job_id": job_id,
                            "split": split,
                            "terminal_status": "measured_success_gold",
                            "latency_ms": 1.0,
                            "energy_j": 0.2,
                            "ap70": 0.5,
                            **common,
                        })
        rows[-1] = {**rows[-1], "manifest_job_id": rows[0]["manifest_job_id"]}

        with self.assertRaisesRegex(ValueError, "unique"):
            stage35.validate_dataset_contract(rows, {"jobs": jobs})

    def test_validate_repeat_audit_rejects_unbound_qualified_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema"):
            stage35.validate_repeat_audit(
                {"qualified": True}, [], source_gold_sha256="a" * 64
            )

    def test_lock_decision_requires_repeat_audit_and_all_thresholds(self) -> None:
        evidence = {
            "learning": {
                target: {"mae_improvement_24_to_26": 0.04, "spearman_26": 0.92, "topk_recall_26": 0.8}
                for target in ("ap70", "latency_ms", "energy_j")
            },
            "calibrated_beats_global_only_count": 24,
            "calibrated_beats_local_only_count": 16,
            "transfer_task_count": 24,
            "pareto_recall_p10": 0.85,
            "hv_regret_p90": 0.08,
            "coverage_90_min": 0.82,
            "fully_covered_group_rate_90_min": 0.82,
        }
        pending = stage35.lock_decision(evidence, repeat_audit_qualified=None)
        self.assertEqual(pending["decision"], "repeat_audit_required")
        locked = stage35.lock_decision(evidence, repeat_audit_qualified=True)
        self.assertEqual(locked["decision"], "lock_gold128_for_stage4")
        evidence["learning"]["latency_ms"]["mae_improvement_24_to_26"] = 0.08
        rejected = stage35.lock_decision(evidence, repeat_audit_qualified=True)
        self.assertEqual(rejected["decision"], "targeted_supplement_required")

    def test_lock_decision_uses_pareto_tail_not_mean(self) -> None:
        evidence = {
            "learning": {
                target: {"mae_improvement_24_to_26": 0.0, "spearman_26": 0.95, "topk_recall_26": 0.9}
                for target in ("ap70", "latency_ms", "energy_j")
            },
            "calibrated_beats_global_only_count": 24,
            "calibrated_beats_local_only_count": 16,
            "transfer_task_count": 24,
            "pareto_recall_p10": 0.33,
            "hv_regret_p90": 0.30,
            "coverage_90_min": 0.9,
            "fully_covered_group_rate_90_min": 0.9,
        }

        result = stage35.lock_decision(evidence, repeat_audit_qualified=True)

        self.assertFalse(result["checks"]["pareto_recall"])
        self.assertFalse(result["checks"]["hv_regret"])


if __name__ == "__main__":
    unittest.main()
