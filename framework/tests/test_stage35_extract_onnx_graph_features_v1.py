import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_extract_onnx_graph_features_v1 as extractor  # noqa: E402


class Stage35ExtractOnnxGraphFeaturesV1Tests(unittest.TestCase):
    def test_product_ignores_unknown_dimensions(self) -> None:
        self.assertEqual(extractor.product([2, 64, 0, 256]), 32768)

    def test_attribute_ints_preserves_lists_and_scalars(self) -> None:
        node = SimpleNamespace(attribute=[
            SimpleNamespace(name="group", ints=[], i=4),
            SimpleNamespace(name="strides", ints=[2, 2], i=0),
        ])
        self.assertEqual(extractor.attribute_ints(node), {"group": 4, "strides": [2, 2]})

    def test_conv_macs_use_output_spatial(self) -> None:
        macs = extractor.estimate_conv_macs(
            "Conv", weight_dims=[32, 16, 3, 3], input_dims=[2, 16, 64, 64],
            output_dims=[2, 32, 32, 32],
        )
        self.assertEqual(macs, 2 * 32 * 32 * 32 * 16 * 3 * 3)

    def test_conv_transpose_macs_use_input_spatial_and_survive_dynamic_batch(self) -> None:
        macs = extractor.estimate_conv_macs(
            "ConvTranspose", weight_dims=[16, 32, 3, 3], input_dims=[0, 16, 32, 32],
            output_dims=[0, 32, 64, 64],
        )
        self.assertEqual(macs, 32 * 32 * 16 * 32 * 3 * 3)


if __name__ == "__main__":
    unittest.main()
