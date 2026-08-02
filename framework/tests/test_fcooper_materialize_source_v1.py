from __future__ import annotations

import unittest

import torch

from scripts.fcooper_materialize_source_v1 import (
    apply_structure_to_hypes,
    project_state_dict,
)


class FCooperMaterializeSourceV1Tests(unittest.TestCase):
    def test_structure_widths_update_backbone_neck_and_head_contracts(self) -> None:
        source = {
            "name": "fcooper",
            "model": {
                "args": {
                    "in_head": 256,
                    "m1": {
                        "backbone_args": {
                            "num_filters": [64, 128, 256],
                            "num_upsample_filter": [128, 128, 128],
                        },
                        "shrink_header": {"input_dim": 384, "dim": [256]},
                    },
                }
            },
        }

        target = apply_structure_to_hypes(source, [32, 64, 128, 96, 64])

        args = target["model"]["args"]
        self.assertEqual(args["m1"]["backbone_args"]["num_filters"], [32, 64, 128])
        self.assertEqual(args["m1"]["backbone_args"]["num_upsample_filter"], [96, 96, 96])
        self.assertEqual(args["m1"]["shrink_header"]["input_dim"], 288)
        self.assertEqual(args["m1"]["shrink_header"]["dim"], [64])
        self.assertEqual(args["in_head"], 64)
        self.assertEqual(source["model"]["args"]["in_head"], 256)

    def test_state_projection_is_deterministic_and_shape_exact(self) -> None:
        source = {
            "weight": torch.arange(6 * 4 * 3 * 3).reshape(6, 4, 3, 3),
            "bias": torch.arange(6),
        }
        target = {
            "weight": torch.empty(3, 2, 3, 3),
            "bias": torch.empty(3),
        }

        projected, audit = project_state_dict(source, target)

        self.assertEqual(projected["weight"].shape, target["weight"].shape)
        self.assertTrue(torch.equal(projected["bias"], torch.tensor([0, 1, 2])))
        self.assertEqual(audit["changed_tensor_count"], 2)


if __name__ == "__main__":
    unittest.main()
