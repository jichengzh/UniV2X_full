import unittest
import json
import tempfile
from pathlib import Path

from scripts.stage6_finalize_fcooper_table1_v1 import (
    _median_performance,
    select_arm_candidate,
)


def row(row_id: str, ap70: float, latency: float, energy: float) -> dict:
    return {
        "row_id": row_id,
        "terminal_status": "measured_success_gold",
        "ap70": ap70,
        "latency_ms": latency,
        "energy_j": energy,
    }


class FCooperTableSelectionTest(unittest.TestCase):
    def test_filters_by_ap_before_minimizing_latency(self) -> None:
        selected = select_arm_candidate(
            [
                row("fast-invalid", 0.40, 0.5, 0.1),
                row("valid", 0.55, 1.0, 0.3),
            ],
            ap70_ref=0.63,
            max_ap_drop=0.10,
        )

        self.assertEqual(selected["row_id"], "valid")
        self.assertTrue(selected["ap_constraint_satisfied"])

    def test_energy_breaks_only_one_percent_latency_tie(self) -> None:
        selected = select_arm_candidate(
            [
                row("minimum", 0.60, 1.0, 0.4),
                row("tie", 0.60, 1.009, 0.2),
                row("too-slow", 0.60, 1.02, 0.1),
            ],
            ap70_ref=0.63,
            max_ap_drop=0.10,
        )

        self.assertEqual(selected["row_id"], "tie")

    def test_no_feasible_point_returns_fastest_with_violation(self) -> None:
        selected = select_arm_candidate(
            [row("a", 0.0, 0.8, 0.2), row("b", 0.1, 1.0, 0.1)],
            ap70_ref=0.63,
            max_ap_drop=0.10,
        )

        self.assertEqual(selected["row_id"], "a")
        self.assertFalse(selected["ap_constraint_satisfied"])

    def test_repeats_must_match_selected_onnx_precision_and_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(3):
                path = Path(tmp) / f"repeat_{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "lat_p50_ms": 1.0 + index / 100,
                            "energy_j": 0.2,
                            "onnx": "fcooper_dense_64x128x256x128x256.onnx",
                            "precision": "int8" if index < 2 else "fp16",
                            "builder_optimization_level": 5,
                            "gpu_abs": 7,
                            "artifact_sha256": {
                                "source_onnx": "a" * 64,
                                "compiled_engine": "b" * 64,
                            },
                        }
                    )
                )
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "precision drift"):
                _median_performance(
                    paths,
                    expected_onnx_sha256="a" * 64,
                    expected_precision="int8",
                    expected_builder_level=5,
                    expected_gpu=7,
                )


if __name__ == "__main__":
    unittest.main()
