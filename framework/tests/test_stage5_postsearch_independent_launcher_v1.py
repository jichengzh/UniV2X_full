from pathlib import Path
import subprocess
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage5_postsearch_independent_launcher_v1.sh"


class Stage5PostsearchIndependentLauncherV1Tests(unittest.TestCase):
    def test_launcher_waits_for_cpu_closure_and_stable_idle_gpus(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("stage5_postsearch_cpu_closure_v1.done", source)
        self.assertIn("GPU_POOL=${GPU_POOL:-0,3,6,7}", source)
        self.assertIn("STABLE_IDLE_CHECKS=${STABLE_IDLE_CHECKS:-3}", source)
        self.assertIn("memory > 100 || utilization > 10", source)
        self.assertIn("stage5_run_independent_validation_v1.sh", source)
        self.assertIn("stage5_postsearch_independent_launcher_v1.done", source)
        self.assertIn("stage5_postsearch_independent_launcher_v1.failed", source)


if __name__ == "__main__":
    unittest.main()
