from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96_codriving_trt_collab_builder as builder  # noqa: E402


class V2GoldColdstart96CoDrivingTrtCollabBuilderTests(unittest.TestCase):
    def test_engine_path_uses_width_and_precision(self) -> None:
        path = builder.engine_path(Path("/tmp/out"), "16x32x64", "fp16")

        self.assertEqual(path, Path("/tmp/out/16x32x64/collab_16x32x64_fp16.engine"))

    def test_build_plan_requires_calibration_for_int8(self) -> None:
        jobs = builder.build_plan(
            onnx=Path("/tmp/model.onnx"),
            out_root=Path("/tmp/out"),
            width="16x32x64",
            modes=["fp16", "int8"],
            calib_data=Path("/tmp/calib.npz"),
            workspace_mb=2048,
        )

        self.assertEqual([job["mode"] for job in jobs], ["fp16", "int8"])
        self.assertIsNone(jobs[0]["calib_data"])
        self.assertEqual(jobs[1]["calib_data"], "/tmp/calib.npz")
        self.assertEqual(jobs[1]["workspace_mb"], 2048)


if __name__ == "__main__":
    unittest.main()
