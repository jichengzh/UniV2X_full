from __future__ import annotations

import unittest

from framework.stage6.contracts_v1 import (
    build_stage6_manifest,
    validate_stage6_manifest,
)


class Stage6ContractsV1Tests(unittest.TestCase):
    def test_frozen_six_arm_manifest_passes(self) -> None:
        manifest = build_stage6_manifest()

        audit = validate_stage6_manifest(manifest)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["arm_count"], 6)
        self.assertEqual(manifest["common_contract"]["genome"], ["w0", "w1", "w2", "q_mode"])
        self.assertEqual(manifest["common_contract"]["theoretical_width_grid_size"], 343)
        self.assertEqual(manifest["common_contract"]["registered_width_count"], 60)
        self.assertEqual(manifest["common_contract"]["effective_candidate_pool_size"], 100)
        self.assertEqual(manifest["arms"][1]["q_modes"], ["fp16", "int8"])
        self.assertEqual(manifest["arms"][3]["outer_budget"], {"screen": 12, "locked": 4})

    def test_hardware_blind_arm_rejects_backend_labels(self) -> None:
        manifest = build_stage6_manifest()
        arm = manifest["arms"][1]
        arm["acquisition_features"] = [*arm["acquisition_features"], "latency_ms"]

        audit = validate_stage6_manifest(manifest)

        self.assertFalse(audit["passed"])
        self.assertIn("compression_only_backend_label_leakage", audit["failures"])

    def test_reverse_serial_rejects_retune_and_fallback(self) -> None:
        manifest = build_stage6_manifest()
        arm = manifest["arms"][4]
        arm["compressed_shape_retune_allowed"] = True

        audit = validate_stage6_manifest(manifest)

        self.assertFalse(audit["passed"])
        self.assertIn("tune_then_compress_contract_violation", audit["failures"])

    def test_manifest_rejects_theoretical_grid_as_effective_pool(self) -> None:
        manifest = build_stage6_manifest()
        manifest["common_contract"]["effective_candidate_pool_size"] = 686

        audit = validate_stage6_manifest(manifest)

        self.assertFalse(audit["passed"])
        self.assertIn("effective_candidate_pool_contract_mismatch", audit["failures"])


if __name__ == "__main__":
    unittest.main()
