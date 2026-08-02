from __future__ import annotations

import math
import unittest

from framework.stage2 import s1_probe_metrics_v3 as metrics


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "probe_id": "conv",
        "q_mode": "int8",
        "build_success": True,
        "int8_propagated_ops": 8,
        "precision_eligible_ops": 10,
        "qdq_folded_pairs": 6,
        "qdq_pairs": 8,
        "reformat_ops": 2,
        "total_ops": 20,
        "fused_ops": 9,
        "fusible_ops": 12,
        "probe_seconds": 0.4,
        "build_seconds": 1.2,
    }
    return {**row, **overrides}


class S1ProbeMetricsV3Tests(unittest.TestCase):
    def test_summarize_uses_pooled_structural_counts_and_never_emits_perf_metrics(self) -> None:
        summary = metrics.summarize_probe_records(
            [_row(), _row(int8_propagated_ops=1, precision_eligible_ops=2, build_success=False)]
        )
        group = summary["groups"][0]
        self.assertEqual(group["build_success_rate"], 0.5)
        self.assertEqual(group["int8_precision_propagation_ratio"], 0.75)
        self.assertEqual(group["qdq_fold_ratio"], 0.75)
        self.assertEqual(group["reformat_rate"], 0.1)
        self.assertEqual(group["fusion_coverage"], 0.75)
        self.assertAlmostEqual(group["probe_cost_seconds"], 0.8)
        self.assertAlmostEqual(group["build_seconds"], 2.4)
        self.assertFalse(any("latency" in key or "energy" in key for key in group))

    def test_summary_reports_completeness_and_repeat_stability(self) -> None:
        stable = metrics.summarize_probe_records([_row(), _row()])
        incomplete = metrics.summarize_probe_records([_row(qdq_pairs=None, qdq_folded_pairs=None)])
        self.assertTrue(stable["completeness"]["complete"])
        self.assertTrue(stable["stability"]["stable"])
        self.assertEqual(stable["stability"]["max_abs_spread"], 0.0)
        self.assertFalse(incomplete["completeness"]["complete"])
        self.assertIn("conv:int8:qdq_fold_ratio", incomplete["completeness"]["missing_metrics"])

    def test_invalid_or_forbidden_records_are_rejected(self) -> None:
        cases = [
            (_row(latency_ms=1.0), "latency/energy"),
            (_row(ap70=0.5), "latency/energy"),
            (_row(total_ops=0), "denominator"),
            (_row(fused_ops=13), "cannot exceed"),
            (_row(probe_seconds=math.inf), "finite"),
        ]
        for bad_row, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                metrics.summarize_probe_records([bad_row])

    def test_merge_c0_c1_is_immutable_and_rejects_conflicts_or_forbidden_features(self) -> None:
        c0 = {"supports_int8_tensorcore": 1.0, "vector_width": 16}
        c1 = {"build_success_rate": 1.0, "qdq_fold_ratio": 0.75}
        merged = metrics.merge_c0_c1_capability_features(c0, c1)
        self.assertEqual(merged, {**c0, **c1})
        self.assertEqual(c0, {"supports_int8_tensorcore": 1.0, "vector_width": 16})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            metrics.merge_c0_c1_capability_features(c0, {"vector_width": 32})
        with self.assertRaisesRegex(ValueError, "latency/energy"):
            metrics.merge_c0_c1_capability_features(c0, {"latency_ms": 1.0})

    def test_admission_requires_complete_stable_structural_evidence(self) -> None:
        admitted = metrics.check_capability_admission(metrics.summarize_probe_records([_row(), _row()]))
        unstable = metrics.summarize_probe_records([_row(), _row(fused_ops=1)])
        self.assertEqual(admitted, {"admitted": True, "reasons": []})
        denied = metrics.check_capability_admission(unstable, max_abs_spread=0.2)
        self.assertFalse(denied["admitted"])
        self.assertIn("unstable_metrics", denied["reasons"])


if __name__ == "__main__":
    unittest.main()
