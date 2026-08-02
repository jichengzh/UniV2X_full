import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_fp16_tensorcore_convblock_and_engine_probe as probe  # noqa: E402


class Stage2Fp16FullEngineGroupConvMatcherTests(unittest.TestCase):
    def test_classifies_speed_gate_group_conv_shapes(self):
        downsample_text = """
        T.Buffer((T.int64(256), T.int64(8), T.int64(3), T.int64(3)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(64), T.int64(128)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(32), T.int64(64)), "float16")
        """
        repeated_text = """
        T.Buffer((T.int64(256), T.int64(8), T.int64(3), T.int64(3)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(32), T.int64(64)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(32), T.int64(64)), "float16")
        """

        downsample = probe._classify_full_engine_group_conv_primfunc_text(downsample_text)
        repeated = probe._classify_full_engine_group_conv_primfunc_text(repeated_text)

        self.assertEqual(downsample["candidate"], "fused_conv2d4_add10_relu6")
        self.assertEqual(downsample["spec"]["input_nchw"], (2, 256, 64, 128))
        self.assertEqual(repeated["candidate"], "fused_conv2d6_add10_relu6")
        self.assertEqual(repeated["spec"]["input_nchw"], (2, 256, 32, 64))

    def test_classifies_ap_shape_group_conv_shapes(self):
        downsample_text = """
        T.Buffer((T.int64(256), T.int64(8), T.int64(3), T.int64(3)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(128), T.int64(128)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(64), T.int64(64)), "float16")
        """
        repeated_text = """
        T.Buffer((T.int64(256), T.int64(8), T.int64(3), T.int64(3)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(64), T.int64(64)), "float16")
        T.Buffer((T.int64(2), T.int64(256), T.int64(64), T.int64(64)), "float16")
        """

        downsample = probe._classify_full_engine_group_conv_primfunc_text(downsample_text)
        repeated = probe._classify_full_engine_group_conv_primfunc_text(repeated_text)

        self.assertEqual(downsample["candidate"], "fused_conv2d4_add10_relu6")
        self.assertEqual(downsample["spec"]["input_nchw"], (2, 256, 128, 128))
        self.assertEqual(repeated["candidate"], "fused_conv2d6_add10_relu6")
        self.assertEqual(repeated["spec"]["input_nchw"], (2, 256, 64, 64))

    def test_rejects_non_target_group_conv_shape(self):
        text = """
        T.Buffer((T.int64(96), T.int64(3), T.int64(3), T.int64(3)), "float16")
        T.Buffer((T.int64(2), T.int64(96), T.int64(128), T.int64(128)), "float16")
        """

        self.assertIsNone(probe._classify_full_engine_group_conv_primfunc_text(text))


if __name__ == "__main__":
    unittest.main()
