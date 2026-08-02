from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_gold32_performance_plan_v1.py"
PLAN_INDEX = REPO_ROOT / "results/stage35_gold32_supplement_v1_20260713/plan/gold32_supplement_plan_v1.json"


def _manifest() -> dict:
    index = json.loads(PLAN_INDEX.read_text(encoding="utf-8"))
    return json.loads(Path(index["manifest_json"]).read_text(encoding="utf-8"))


def _module():
    spec = importlib.util.spec_from_file_location("stage35_gold32_performance_plan_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage35Gold32PerformancePlanV1Tests(unittest.TestCase):
    def test_two_batches_cover_all_groups_and_use_row_source_contracts(self) -> None:
        module = _module()
        manifest = _manifest()
        batches = module.build_all_batches(
            manifest,
            remote_artifact_root="/remote/gold32",
            gpus=[7],
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual([batch["group_count"] for batch in batches], [4, 4])
        jobs = [job for batch in batches for job in batch["jobs"]]
        self.assertEqual(len(jobs), 32)
        self.assertEqual({job["runner_key"] for job in jobs}, {"tvm_fp16", "tvm_int8", "trt_fp16", "trt_int8"})

        pyramid = next(job for job in jobs if job["job_id"] == "pyramid|24x64x128|tvm_fp16")
        self.assertTrue(pyramid["onnx_path"].endswith("pyramid_024x064x128_multiscale.onnx"))
        codriving = next(job for job in jobs if job["job_id"] == "codriving|24x56x128|trt_int8")
        self.assertTrue(codriving["onnx_path"].endswith("resnet_multiscale_24x56x128_final_fp32.onnx"))
        self.assertIn("--calib-dir", codriving["command"])

        for job in jobs:
            command = " ".join(job["command"]).lower()
            self.assertNotIn("hand_rewrite", command)
            self.assertNotIn("hand-rewrite", command)
            self.assertNotIn("mixed", command)


if __name__ == "__main__":
    unittest.main()
