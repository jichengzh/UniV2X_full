import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_fp16_h800_gate_check as gate_check  # noqa: E402


class Stage2Fp16H800GateCheckTests(unittest.TestCase):
    def test_h800_preflight_check_reports_failed_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            preflight = tmp_path / "fp16_lhc07_h800_suite_preflight_latest.json"
            preflight.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "failure_reasons": [
                            "python_bin_not_found",
                            "h800_sm90_gpu_not_available",
                        ],
                        "checks": {
                            "python_bin_exists": False,
                            "selected_gpu_name": "NVIDIA GeForce RTX 4090",
                            "selected_gpu_compute_cap": "8.9",
                            "sm90_gate": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = gate_check._check_h800_preflight(tmp_path)

            self.assertEqual(result["status"], "fail")
            self.assertTrue(result["required_for_goal"])
            self.assertEqual(
                result["failure_reasons"],
                ["python_bin_not_found", "h800_sm90_gpu_not_available"],
            )
            self.assertEqual(result["selected_gpu_compute_cap"], "8.9")

    def test_fp16_rewritten_ap_gate_rejects_generic_ap_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generic = tmp_path / "fp16_original60_ap_launch_plan_latest.json"
            generic.write_text(json.dumps({"status": "success"}), encoding="utf-8")

            result = gate_check._check_ap(tmp_path)

            self.assertEqual(result["status"], "missing")
            self.assertEqual(
                result["missing_reason"],
                "missing_tvm_rewritten_backbone_to_head_postprocess_bridge",
            )
            self.assertIn(str(generic), result["ignored_generic_ap_paths"])
            self.assertEqual(result["candidate_paths"], [])

    def test_fp16_rewritten_ap_gate_accepts_strict_lhc_rewrite_ap_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidate = tmp_path / "fp16_lhc07_rewritten_full_engine_ap_smoke_latest.json"
            candidate.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "ap30": 0.2,
                        "ap50": 0.1,
                        "ap70": 0.02,
                        "num_predictions": 12,
                    }
                ),
                encoding="utf-8",
            )

            result = gate_check._check_ap(tmp_path)

            self.assertEqual(result["status"], "manual_review")
            self.assertEqual(result["candidate_paths"], [str(candidate)])
            self.assertIsNone(result["missing_reason"])


if __name__ == "__main__":
    unittest.main()
