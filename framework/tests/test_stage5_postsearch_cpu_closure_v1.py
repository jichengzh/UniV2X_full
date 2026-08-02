from pathlib import Path
import subprocess
import unittest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage5_postsearch_cpu_closure_v1.sh"


class Stage5PostsearchCpuClosureV1Tests(unittest.TestCase):
    def test_script_is_valid_and_waits_for_full_budget_terminal(self) -> None:
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        text = SCRIPT.read_text()
        self.assertIn("full_budget_scheduler_terminal.json", text)
        self.assertIn("stage5_finalize_full_budget_v3.py", text)
        self.assertIn("stage5_backfill_actual_graph_features_v1.py", text)
        self.assertIn("stage5_graph_feature_replay_v1.py", text)
        self.assertNotIn("stage5_run_independent_validation_v1.sh", text)
        self.assertIn("stage5_postsearch_cpu_closure_v1.failed", text)
        self.assertIn("stage5_postsearch_cpu_closure_v1.done", text)


if __name__ == "__main__":
    unittest.main()
