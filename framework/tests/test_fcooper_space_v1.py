from __future__ import annotations

import unittest
from pathlib import Path

from framework.stage5.fcooper_space_v1 import build_fcooper_source_registry


ROOT = Path(__file__).resolve().parents[2]


class FCooperSpaceV1Tests(unittest.TestCase):
    def test_registry_is_derived_from_four_scanner_groups(self) -> None:
        registry = build_fcooper_source_registry(
            ROOT / "framework/partitions/fcooper_partition.yaml",
            artifact_root=Path("/remote/fcooper"),
        )

        self.assertEqual(registry["model"], "fcooper")
        self.assertEqual(
            registry["width_schema"],
            [
                "backbone.s0",
                "backbone.s1",
                "backbone.s2",
                "neck.deblock",
                "neck.output",
            ],
        )
        self.assertEqual(registry["structure_group_count"], 5)
        self.assertEqual(registry["structure_candidate_count"], 1792)
        self.assertEqual(len(registry["groups"]), 1792)
        self.assertEqual(len({row["group_id"] for row in registry["groups"]}), 1792)
        self.assertTrue(all(len(row["width"]) == 5 for row in registry["groups"]))
        self.assertTrue(
            all(
                row["materialization_kind"] == "fcooper_scanner_materialize_export"
                for row in registry["groups"]
            )
        )

    def test_registry_contains_original_and_boundary_widths(self) -> None:
        registry = build_fcooper_source_registry(
            ROOT / "framework/partitions/fcooper_partition.yaml",
            artifact_root=Path("/remote/fcooper"),
        )
        widths = {tuple(row["width"]) for row in registry["groups"]}

        self.assertIn((64, 128, 256, 128, 256), widths)
        self.assertIn((32, 32, 32, 32, 64), widths)


if __name__ == "__main__":
    unittest.main()
