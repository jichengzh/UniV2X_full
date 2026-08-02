from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage5_task_round_controller_v3.sh"


class Stage5TaskRoundControllerV3Tests(unittest.TestCase):
    def test_controller_accepts_valid_numerical_failure_as_terminal(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('failure_reason == "numerical_feasibility_failure"', source)
        self.assertIn('status == "skipped_numerical_feasibility"', source)
        self.assertIn('wait "$pid" || rc=1', source)

    def test_controller_supports_a_smaller_gpu_pool_without_changing_batch_size(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn("GPU_POOL must contain at least four GPUs", source)
        self.assertIn("GPU_COUNT=${#GPUS[@]}", source)
        self.assertIn("gpu=${GPUS[$((index % GPU_COUNT))]}", source)
        self.assertIn('--max-workers "$GPU_COUNT"', source)
        self.assertIn('budget_consumed == 4', source)

    def _run(self, task: str, model: str, dispatch: str, *, request_task: str | None = None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            round_dir = root / task / "round_00"
            round_dir.mkdir(parents=True)
            rows = [
                {
                    "task_id": request_task or task,
                    "model": model,
                    "dispatch_key": dispatch,
                    "manifest_job_id": f"row-{index}",
                    "group_id": f"{model}|16x32x64",
                    "width": [16, 32, 64],
                    "q_mode": "fp16" if index == 0 else "int8",
                }
                for index in range(4)
            ]
            (round_dir / "measurement_request.json").write_text(
                json.dumps(
                    {
                        "schema_version": "stage5_measurement_request_v2",
                        "task_id": request_task or task,
                        "round_index": 0,
                        "batch_size": 4,
                        "rows": rows,
                    }
                ),
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "REPO": str(REPO),
                "ROOT": str(root),
                "TASK": task,
                "ROUND_INDEX": "0",
            }
            return subprocess.run(
                ["bash", str(SCRIPT), "--validate-only"],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_validate_only_accepts_all_four_fixed_task_contexts(self) -> None:
        for task, model, dispatch in (
            ("S5-PYR-TVM", "pyramid", "tvm_auto"),
            ("S5-PYR-TRT", "pyramid", "trt_engine"),
            ("S5-COD-TVM", "codriving", "tvm_auto"),
            ("S5-COD-TRT", "codriving", "trt_engine"),
        ):
            with self.subTest(task=task):
                result = self._run(task, model, dispatch)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("REQUEST_VALID", result.stdout)

    def test_validate_only_rejects_cross_task_request(self) -> None:
        result = self._run(
            "S5-PYR-TVM",
            "pyramid",
            "tvm_auto",
            request_task="S5-PYR-TRT",
        )

        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
