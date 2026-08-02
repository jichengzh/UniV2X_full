from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage5_run_independent_validation_v1.sh"


class Stage5RunIndependentValidationV1Tests(unittest.TestCase):
    def test_ap_validation_runs_in_gpu_bounded_waves(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('GPU_COUNT=${#GPUS[@]}', source)
        self.assertIn('if (( ${#pids[@]} == GPU_COUNT )); then', source)
        self.assertIn('pids=()', source)

    def test_plan_contains_three_repeats_and_full_ap_for_four_tasks(self) -> None:
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--plan-only"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for task in ("S5-PYR-TVM", "S5-PYR-TRT", "S5-COD-TVM", "S5-COD-TRT"):
            self.assertEqual(completed.stdout.count(f"PERFORMANCE {task}"), 3)
            self.assertIn(f"FULL_AP {task}", completed.stdout)
        self.assertIn("FINAL_CLOSURE", completed.stdout)


if __name__ == "__main__":
    unittest.main()
