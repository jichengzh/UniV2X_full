from __future__ import annotations

import unittest

from framework.stage6.protocol_smoke_v1 import (
    lock_compress_then_tune,
    record_reverse_transfer_attempts,
    select_hardware_blind_batch,
)


class Stage6ProtocolSmokeV1Tests(unittest.TestCase):
    def test_hardware_blind_selection_ignores_backend_labels(self) -> None:
        candidates = [
            {
                "candidate_id": str(index),
                "width": [16 + index, 32, 64],
                "q_mode": "fp16" if index % 2 else "int8",
                "parameter_count": 100 - index,
                "flops": 200 - index,
                "ap_surrogate": 0.5 + index / 100,
                "latency_ms": -1000 * index,
                "energy_j": -1000 * index,
            }
            for index in range(8)
        ]
        selected = select_hardware_blind_batch(candidates, batch_size=4)
        perturbed = [
            {**row, "latency_ms": 10**9 + i, "energy_j": 10**9 - i}
            for i, row in enumerate(candidates)
        ]
        selected_perturbed = select_hardware_blind_batch(perturbed, batch_size=4)

        self.assertEqual(
            [row["candidate_id"] for row in selected],
            [row["candidate_id"] for row in selected_perturbed],
        )
        self.assertTrue(all("latency_ms" not in row and "energy_j" not in row for row in selected))

    def test_forward_serial_locks_four_after_exactly_twelve(self) -> None:
        screened = [{"candidate_id": str(index), "screen_rank": index} for index in range(12)]
        result = lock_compress_then_tune(screened)

        self.assertEqual(result["screen_count"], 12)
        self.assertEqual(result["locked_count"], 4)
        self.assertTrue(result["lock_precedes_tuning"])

    def test_reverse_transfer_keeps_failures_without_fallback(self) -> None:
        result = record_reverse_transfer_attempts(
            [
                {"candidate_id": "a", "applicable": False, "reason": "shape_mismatch"},
                {"candidate_id": "b", "applicable": True},
            ]
        )

        self.assertEqual(result["failure_count"], 1)
        self.assertEqual(result["fallback_count"], 0)
        self.assertEqual(result["retune_count"], 0)

    def test_reverse_transfer_preserves_fallback_and_retune_violations(self) -> None:
        result = record_reverse_transfer_attempts(
            [{
                "candidate_id": "bad",
                "applicable": False,
                "fallback_used": True,
                "compressed_shape_retuned": True,
            }]
        )

        self.assertEqual(result["fallback_count"], 1)
        self.assertEqual(result["retune_count"], 1)


if __name__ == "__main__":
    unittest.main()
