from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import stage35_gold176_ap_plan_v1 as ap_plan


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results/stage35_gold144_targeted_supplement_v2_20260714"
MANIFEST = RESULT_ROOT / "plan/targeted_supplement_manifest.json"
PERFORMANCE_JOBS = RESULT_ROOT / "plan/targeted_performance_jobs.jsonl"


class Stage35Gold176ApPlanTests(unittest.TestCase):
    def test_builds_32_ready_rows_bound_to_performance_artifacts(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        performance_jobs = [
            json.loads(line)
            for line in PERFORMANCE_JOBS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            states = []
            for index, job in enumerate(performance_jobs):
                if job["runner_key"].startswith("trt_"):
                    artifact = root / f"artifact-{index}.engine"
                elif job["runner_key"] == "tvm_fp16":
                    artifact = root / f"artifact-{index}.so"
                else:
                    artifact = root / f"route-{index}/route_b_int8_auto_decomp.vmexec"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"artifact")
                result = root / f"result-{index}.json"
                result.write_text(json.dumps({"artifact_path": str(artifact)}), encoding="utf-8")
                states.append(
                    {"job_id": job["job_id"], "status": "success", "result_json": str(result)}
                )

            rows = ap_plan.build_gold176_ap_plan(
                manifest,
                performance_jobs=performance_jobs,
                performance_state_rows=states,
                output_root=root / "ap_execution",
            )

        self.assertEqual(len(rows), 32)
        self.assertEqual({row["ap_terminal"] for row in rows}, {"ready"})
        self.assertEqual(
            {row["runner_key"] for row in rows},
            {
                "pyramid_trt_multiscale",
                "pyramid_tvm_fp16_bridge",
                "pyramid_tvm_int8_numeric_gate",
                "codriving_trt_multiscale",
                "codriving_tvm_fp16_bridge",
                "codriving_tvm_int8_numeric_gate",
            },
        )
        pyramid = next(row for row in rows if row["model"] == "pyramid")
        checkpoint_dir = pyramid["source_contract"]["checkpoint_dir"]
        self.assertIn(checkpoint_dir, pyramid["sanity_command"])


if __name__ == "__main__":
    unittest.main()
