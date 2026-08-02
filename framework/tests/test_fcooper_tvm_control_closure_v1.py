from __future__ import annotations

import unittest

from scripts import fcooper_tvm_control_closure_v1 as closure


class FCooperTvmControlClosureTests(unittest.TestCase):
    def test_selects_four_feasible_screen_rows_by_latency_then_energy(self) -> None:
        rows = [
            {
                "row_id": f"row-{index}",
                "terminal_status": "measured_success_gold",
                "ap70": 0.65,
                "latency_ms": float(20 - index),
                "energy_j": float(index),
            }
            for index in range(12)
        ]

        selected = closure.select_tuned_screen_rows(
            rows,
            ap70_ref=0.70,
            max_ap_drop=0.10,
        )

        self.assertEqual(
            [row["row_id"] for row in selected],
            ["row-11", "row-10", "row-9", "row-8"],
        )

    def test_tuned_request_preserves_candidate_identity_and_changes_only_phase(self) -> None:
        screen_rows = [
            {
                "row_id": f"screen-{index}",
                "manifest_job_id": f"screen-{index}",
                "task_id": "S6-FCO-TVM-COMPRESS-THEN-TUNE-SCREEN-V1",
                "task_sha256": "s" * 64,
                "arm_id": "compress_then_tune",
                "phase": "screen",
                "model": "fcooper",
                "hardware_id": "h800",
                "backend": "tvm",
                "dispatch_key": "tvm_auto",
                "capability_profile_id": "profile",
                "capability_digest": "c" * 64,
                "width": [32, 64, 64, 32, 64],
                "q_mode": "fp16",
                "source_contract": {"checkpoint": "x"},
                "tvm_trials": 0,
            }
            for index in range(4)
        ]

        request = closure.build_tuned_request(screen_rows)

        self.assertEqual(request["batch_size"], 4)
        self.assertEqual(request["tvm_trials"], 64)
        self.assertEqual(
            [row["row_id"] for row in request["rows"]],
            [row["row_id"] for row in screen_rows],
        )
        self.assertTrue(
            all(row["phase"] == "tuned_remeasurement" for row in request["rows"])
        )
        self.assertTrue(all(row["tvm_trials"] == 64 for row in request["rows"]))
        self.assertTrue(
            all(
                row["source_contract"] == screen["source_contract"]
                for row, screen in zip(request["rows"], screen_rows)
            )
        )


if __name__ == "__main__":
    unittest.main()
