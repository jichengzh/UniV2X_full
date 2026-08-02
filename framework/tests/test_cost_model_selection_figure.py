from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "multi_agent/figure/cost_model_selection"
MODULE_PATH = FIGURE_DIR / "make_cost_model_selection_figure.py"
SOURCE_PATH = FIGURE_DIR / "cost_model_selection_source.csv"


def load_plot_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"plotting module is missing: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("cost_model_selection_figure", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import plotting module: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CostModelSelectionFigureTests(unittest.TestCase):
    def test_source_contract_and_retained_heads(self) -> None:
        module = load_plot_module()
        rows = module.load_rows(SOURCE_PATH)
        module.validate_rows(rows)

        self.assertEqual(len(rows), 24)
        self.assertEqual(Counter(row["target"] for row in rows), {
            "latency_ms": 8,
            "energy_j": 8,
            "ap70": 8,
        })
        self.assertEqual(
            {
                row["target"]: (
                    row["candidate"],
                    round(float(row["inner_score"]), 10),
                    int(row["selected_folds"]),
                )
                for row in module.retained_rows(rows)
            },
            {
                "latency_ms": ("extra_trees_log", 0.1024188961, 5),
                "energy_j": ("extra_trees_log", 0.1412040529, 4),
                "ap70": ("lgbm_huber_residual", 0.1835866354, 3),
            },
        )

    def test_validation_rejects_duplicate_candidate(self) -> None:
        module = load_plot_module()
        rows = module.load_rows(SOURCE_PATH)

        with self.assertRaisesRegex(ValueError, "unique"):
            module.validate_rows([*rows, dict(rows[0])])

    def test_validation_rejects_unknown_candidate_schema(self) -> None:
        module = load_plot_module()
        rows = module.load_rows(SOURCE_PATH)
        invalid_rows = [
            {**row, "objective": "typo"} if index == 0 else row
            for index, row in enumerate(rows)
        ]

        with self.assertRaisesRegex(ValueError, "candidate schema"):
            module.validate_rows(invalid_rows)

    def test_build_and_export_complete_bundle(self) -> None:
        module = load_plot_module()
        rows = module.load_rows(SOURCE_PATH)
        figure = module.build_figure(rows)
        self.assertEqual(len(figure.axes), 3)
        self.assertEqual(
            len(figure.legends),
            0,
            "direct candidate labels make a shared legend redundant",
        )
        figure_notes = " ".join(text.get_text() for text in figure.texts)
        self.assertIn("five-fold means", figure_notes)
        self.assertIn("not comparable across panels", figure_notes)

        with tempfile.TemporaryDirectory() as temporary:
            output_stem = Path(temporary) / "cost_model_selection"
            module.export_figure(figure, output_stem)
            for suffix in (".png", ".svg", ".pdf", ".tiff"):
                artifact = output_stem.with_suffix(suffix)
                self.assertTrue(artifact.is_file(), artifact)
                self.assertGreater(artifact.stat().st_size, 0)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
