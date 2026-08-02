import unittest
from unittest import mock

import torch

from tools.orin_deploy.m1_standalone.voxelizer_torch import _stable_argsort


class LaneCVoxelizerCompatTests(unittest.TestCase):
    def test_numpy_fallback_preserves_equal_value_order(self):
        values = torch.tensor([2, 1, 2, 1], dtype=torch.int64)
        with mock.patch(
            "tools.orin_deploy.m1_standalone.voxelizer_torch.torch.argsort",
            side_effect=TypeError("stable keyword unavailable"),
        ):
            order = _stable_argsort(values)

        self.assertEqual(order.tolist(), [1, 3, 0, 2])


if __name__ == "__main__":
    unittest.main()
