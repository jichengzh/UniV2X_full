import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_fp16_h800_rewrite_suite_runner as runner  # noqa: E402


class Stage2Fp16H800RewriteSuiteRunnerTests(unittest.TestCase):
    def test_preflight_fails_for_missing_python_and_non_sm90_gpu(self):
        report = runner._preflight_environment(
            python_bin=Path("/tmp/definitely_missing_h800_python_for_test"),
            gpu=0,
            nvidia_smi_output="0, NVIDIA GeForce RTX 4090, 8.9\n",
        )

        self.assertEqual(report["status"], "failed")
        self.assertIn("python_bin_not_found", report["failure_reasons"])
        self.assertIn("h800_sm90_gpu_not_available", report["failure_reasons"])
        self.assertFalse(report["checks"]["python_bin_exists"])
        self.assertEqual(report["checks"]["selected_gpu_compute_cap"], "8.9")

    def test_preflight_passes_for_executable_python_and_sm90_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            python_bin = Path(tmp) / "python"
            python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_bin.chmod(python_bin.stat().st_mode | stat.S_IXUSR)

            report = runner._preflight_environment(
                python_bin=python_bin,
                gpu=0,
                nvidia_smi_output="0, NVIDIA H800, 9.0\n",
            )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failure_reasons"], [])
        self.assertTrue(report["checks"]["python_bin_executable"])
        self.assertEqual(report["checks"]["selected_gpu_compute_cap"], "9.0")


if __name__ == "__main__":
    unittest.main()
