from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from framework.tests.test_stage4_selection_completion_v1 import (
    _graph,
    _perfect_cost_report,
    _profiles,
    _rows,
)
from scripts import stage4_complete_selection_eval_v1 as cli


class Stage4CompleteSelectionEvalV1Tests(unittest.TestCase):
    def test_cli_writes_bound_stage4_artifacts(self) -> None:
        rows = _rows()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {}
            for name, payload in (
                ("gold", rows),
                ("graph", _graph(rows)),
                ("profiles", _profiles()),
                ("cost", _perfect_cost_report(rows)),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            output = root / "out"

            return_code = cli.main(
                [
                    "--gold-json",
                    str(paths["gold"]),
                    "--graph-features-json",
                    str(paths["graph"]),
                    "--capability-profiles-json",
                    str(paths["profiles"]),
                    "--cost-model-report-json",
                    str(paths["cost"]),
                    "--output-dir",
                    str(output),
                    "--outer-splits",
                    "3",
                    "--uncertainty-splits",
                    "3",
                    "--replay-initial-groups",
                    "2",
                    "--replay-budget-groups",
                    "6",
                    "--replay-seeds",
                    "3",
                ]
            )

            self.assertEqual(return_code, 0)
            summary = json.loads((output / "stage4_completion_summary.json").read_text())
            self.assertEqual(summary["evaluation_row_count"], 24)
            self.assertEqual(summary["evaluation_group_count"], 6)
            self.assertEqual(set(summary["input_sha256"]), {"gold", "graph_features", "capability_profiles", "cost_model_report"})
            for name in (
                "stage4_ranking_pareto_report.json",
                "stage4_uncertainty_report.json",
                "stage4_closed_loop_replay_report.json",
                "stage4_completion_report.json",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
