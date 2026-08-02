from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "multi_agent/figure/cost_model_selection"
MODULE_PATH = FIGURE_DIR / "make_cost_model_selection_alternatives.py"
SOURCE_PATH = FIGURE_DIR / "cost_model_selection_source.csv"


def load_alternatives_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"alternative plotting module is missing: {MODULE_PATH}")
    sys.path.insert(0, str(FIGURE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "cost_model_selection_alternatives",
            MODULE_PATH,
        )
        if spec is None or spec.loader is None:
            raise AssertionError(f"cannot import plotting module: {MODULE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class CostModelSelectionAlternativeTests(unittest.TestCase):
    def test_normalized_mae_spans_each_target(self) -> None:
        module = load_alternatives_module()
        rows = module.load_rows(SOURCE_PATH)
        normalized = module.normalized_mae_by_identity(rows)

        self.assertEqual(len(normalized), 24)
        for target in module.TARGET_ORDER:
            target_rows = [row for row in rows if row["target"] == target]
            target_values = {
                str(row["candidate"]): normalized[
                    (target, str(row["candidate"]))
                ]
                for row in target_rows
            }
            self.assertAlmostEqual(min(target_values.values()), 0.0)
            self.assertAlmostEqual(max(target_values.values()), 1.0)
            best_mae = min(target_rows, key=lambda row: float(row["oof_mae"]))
            worst_mae = max(target_rows, key=lambda row: float(row["oof_mae"]))
            self.assertAlmostEqual(target_values[str(best_mae["candidate"])], 0.0)
            self.assertAlmostEqual(target_values[str(worst_mae["candidate"])], 1.0)

    def test_heatmap_desirability_directions_and_precision(self) -> None:
        module = load_alternatives_module()
        rows = module.load_rows(SOURCE_PATH)
        target_rows = sorted(
            (dict(row) for row in rows if row["target"] == "energy_j"),
            key=lambda row: float(row["inner_score"]),
        )
        matrix, labels = module.heatmap_matrix(target_rows)

        best_indices = [
            max(range(len(target_rows)), key=lambda index: matrix[index][column])
            for column in range(4)
        ]
        self.assertEqual(
            [target_rows[index]["candidate"] for index in best_indices],
            [
                "lgbm_huber_raw",
                "extra_trees_log",
                "extra_trees_log",
                "extra_trees_log",
            ],
        )
        extra_trees_log_index = next(
            index
            for index, row in enumerate(target_rows)
            if row["candidate"] == "extra_trees_log"
        )
        self.assertEqual(labels[extra_trees_log_index][0], "0.2083")

    def test_builds_scatter_and_heatmap_with_three_target_panels(self) -> None:
        module = load_alternatives_module()
        rows = module.load_rows(SOURCE_PATH)

        scatter = module.build_scatter_figure(rows)
        heatmap = module.build_heatmap_figure(rows)

        self.assertEqual(len(scatter.axes), 3)
        self.assertEqual(len(heatmap.axes), 3)
        self.assertEqual(
            sum(len(axis.texts) for axis in scatter.axes),
            5,
            "scatter directly labels only candidates selected in at least one outer fold",
        )
        scatter_notes = " ".join(text.get_text() for text in scatter.texts)
        heatmap_notes = " ".join(text.get_text() for text in heatmap.texts)
        self.assertIn("within-target normalized MAE", scatter_notes)
        self.assertIn("selected by the inner joint score", scatter_notes)
        self.assertIn("5 outer × 3 inner grouped CV", scatter_notes)
        self.assertIn("MAE/(P90–P10)", heatmap_notes)
        self.assertIn("darker cells indicate better", heatmap_notes)
        plt.close(scatter)
        plt.close(heatmap)

    def test_scatter_labels_point_inward_at_plot_boundaries(self) -> None:
        module = load_alternatives_module()

        self.assertEqual(module.annotation_offset(0.1, 1.0, 0.1, 0.9, 0), (5, -7))
        self.assertEqual(module.annotation_offset(0.9, 0.0, 0.1, 0.9, 1), (-5, 6))

    def test_exports_complete_bundle_for_both_alternatives(self) -> None:
        module = load_alternatives_module()
        rows = module.load_rows(SOURCE_PATH)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            figures = {
                "scatter": module.build_scatter_figure(rows),
                "heatmap": module.build_heatmap_figure(rows),
            }
            for name, figure in figures.items():
                module.export_figure(figure, output_dir / name)
                for suffix in (".png", ".svg", ".pdf", ".tiff"):
                    artifact = (output_dir / name).with_suffix(suffix)
                    self.assertTrue(artifact.is_file(), artifact)
                    self.assertGreater(artifact.stat().st_size, 0)
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()
