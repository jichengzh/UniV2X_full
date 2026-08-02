from __future__ import annotations

import unittest

from framework.stage5.genome_contract_v1 import (
    canonical_group_id,
    validate_structure_identity,
    width_schema_for_model,
)


class Stage5GenomeContractV1Tests(unittest.TestCase):
    def test_legacy_models_keep_triplet_identity(self) -> None:
        row = {
            "model": "pyramid",
            "width": [16, 32, 64],
            "group_id": "pyramid|16x32x64",
        }

        identity = validate_structure_identity(row)

        self.assertEqual(identity.width_schema, ("w0", "w1", "w2"))
        self.assertEqual(identity.group_id, "pyramid|16x32x64")

    def test_fcooper_identity_is_named_and_order_stable(self) -> None:
        schema = width_schema_for_model("fcooper")
        widths = [64, 128, 256, 128, 256]
        row = {
            "model": "fcooper",
            "width": widths,
            "width_schema": list(schema),
            "structure_widths": dict(zip(schema, widths)),
            "group_id": canonical_group_id("fcooper", widths, schema),
        }

        identity = validate_structure_identity(row)

        self.assertEqual(identity.width_schema, schema)
        self.assertEqual(identity.structure_widths["neck.output"], 256)
        self.assertEqual(
            identity.group_id,
            "fcooper|backbone.s0=64|backbone.s1=128|backbone.s2=256|neck.deblock=128|neck.output=256",
        )

    def test_fcooper_rejects_missing_or_reordered_scanner_axes(self) -> None:
        row = {
            "model": "fcooper",
            "width": [256, 128, 256, 128, 64],
            "width_schema": [
                "neck.output",
                "neck.deblock",
                "backbone.s2",
                "backbone.s1",
                "backbone.s0",
            ],
            "group_id": "fcooper|neck.output=256|neck.deblock=128|backbone.s2=256|backbone.s1=128|backbone.s0=64",
        }

        with self.assertRaisesRegex(ValueError, "width_schema"):
            validate_structure_identity(row)


if __name__ == "__main__":
    unittest.main()
