from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold128_ap_repeat_plan_v1 as plan  # noqa: E402


class Stage35Gold128ApRepeatPlanV1Tests(unittest.TestCase):
    def test_builds_four_category_plan_with_new_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            gold = []
            for category, model, width, runner in plan.AP_ANCHORS:
                out_dir = root / category / model / width / runner
                label = f"{model}_{width}" + ("_scaleaware" if runner == "tvm_int8" else "")
                command = ["python3", "runner.py"]
                if runner.startswith("tvm_"):
                    command += ["--out-dir", str(out_dir), "--label", label]
                    artifact = out_dir / label / (
                        "route_b_fp16_auto.so" if runner == "tvm_fp16" else "route_b_int8_auto_decomp.vmexec"
                    )
                else:
                    artifact_dir = out_dir / "artifacts"
                    command += ["--artifact-dir", str(artifact_dir)]
                    artifact = artifact_dir / "compiled.engine"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(f"artifact-{category}".encode())
                q_mode = "int8" if runner.endswith("int8") else "fp16"
                profile = "h800-tvm-probe-conditioned-v3" if runner.startswith("tvm_") else "h800-trt-probe-conditioned-v3"
                manifest_id = f"{model}|{width}|q={q_mode}|profile={profile}"
                jobs.append({
                    "job_id": f"repeat|{category}|{model}|{width}|{runner}",
                    "manifest_job_id": manifest_id,
                    "repeat_category": category,
                    "model": model,
                    "width_key": width,
                    "runner_key": runner,
                    "command": command,
                })
                gold.append({
                    "manifest_job_id": manifest_id,
                    "ap30": 0.8,
                    "ap50": 0.7,
                    "ap70": 0.5,
                    "ap_report_sha256": "a" * 64,
                })

            rows, seeds = plan.build_ap_repeat_plan(jobs, gold, output_root=str(root / "ap"))

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(seeds), 0)
        self.assertEqual({row["repeat_category"] for row in rows}, {item[0] for item in plan.AP_ANCHORS})
        self.assertTrue(all(len(row["compiled_artifact_digest"]) == 64 for row in rows))
        self.assertTrue(all(row["ap_terminal"] == "ready" for row in rows))
        self.assertTrue(all("1789" in row["full_command"] for row in rows))
        self.assertTrue(all("16" in row["sanity_command"] for row in rows))

    def test_missing_selected_artifact_is_rejected(self) -> None:
        category, model, width, runner = plan.AP_ANCHORS[0]
        job = {
            "job_id": f"repeat|{category}|{model}|{width}|{runner}",
            "manifest_job_id": "manifest",
            "repeat_category": category,
            "model": model,
            "width_key": width,
            "runner_key": runner,
            "command": ["python3", "runner.py", "--out-dir", "/missing", "--label", "missing"],
        }
        with self.assertRaisesRegex(FileNotFoundError, "compiled artifact"):
            plan.build_ap_repeat_plan([job], [{"manifest_job_id": "manifest"}], output_root="/tmp/ap")


if __name__ == "__main__":
    unittest.main()
