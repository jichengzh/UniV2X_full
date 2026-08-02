from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_canonical_search_smoke_v3 as smoke  # noqa: E402


class Stage2CanonicalSearchSmokeV3Tests(unittest.TestCase):
    def test_diagnostic_smoke_learns_profile_condition_and_exposes_blind_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profiles = root / "profiles.json"
            decision = root / "decision.json"
            out = root / "smoke.json"
            profiles.write_text(json.dumps(smoke.MINIMAL_PROFILES_FIXTURE), encoding="utf-8")
            decision.write_text(json.dumps(smoke.MINIMAL_DECISION_FIXTURE), encoding="utf-8")

            report = smoke.run_smoke(profiles_path=profiles, decision_path=decision)
            smoke.write_report(report, out)
            self.assertTrue(out.is_file())

        self.assertEqual(report["evidence_kind"], "historical_label_mechanism_test")
        self.assertFalse(report["valid_as_cold_start_framework_evidence"])
        self.assertEqual(report["conditioned_decisions"]["h800-tvm-auto-v3"]["q_mode"], "fp16")
        self.assertEqual(report["conditioned_decisions"]["h800-trt-v3"]["q_mode"], "int8")
        blind_modes = {
            decision["q_mode"] for decision in report["backend_blind_decisions"].values()
        }
        self.assertEqual(len(blind_modes), 1)
        self.assertTrue(
            any(
                report["backend_blind_decisions"][profile_id]["q_mode"]
                != report["conditioned_decisions"][profile_id]["q_mode"]
                for profile_id in report["conditioned_decisions"]
            )
        )
        self.assertGreater(report["unconditional_int8_regret"]["h800-tvm-auto-v3"]["relative_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
