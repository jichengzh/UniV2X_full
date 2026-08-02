from __future__ import annotations

import unittest

from scripts import stage2_fp16_tensorcore_convblock_and_engine_probe as probe


class Stage2Fp16TensorcoreGroupConvClassifierTest(unittest.TestCase):
    def test_classifier_accepts_dynamic_group_conv_channels(self) -> None:
        text = """
        def fused_conv2d16_add8_relu6(
            lv129: T.Buffer((T.int64(2), T.int64(384), T.int64(64), T.int64(64)), "float16"),
            param_0: T.Buffer((T.int64(384), T.int64(12), T.int64(3), T.int64(3)), "float16"),
            lv131: T.Buffer((T.int64(1), T.int64(384), T.int64(1), T.int64(1)), "float16"),
            compute_intermediate: T.Buffer((T.int64(2), T.int64(384), T.int64(64), T.int64(64)), "float16"),
        ):
            pass
        """
        match = probe._classify_full_engine_group_conv_primfunc_text(text)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match["candidate"], "fused_conv2d6_add10_relu6")
        self.assertEqual(match["spec"]["input_nchw"], (2, 384, 64, 64))
        self.assertEqual(match["spec"]["weight_oihw"], (384, 12, 3, 3))
        self.assertEqual(match["spec"]["bias"], (1, 384, 1, 1))

    def test_classifier_rejects_non_group32_weight(self) -> None:
        text = """
        def fused_conv2d(
            x: T.Buffer((T.int64(2), T.int64(384), T.int64(64), T.int64(64)), "float16"),
            w: T.Buffer((T.int64(384), T.int64(8), T.int64(3), T.int64(3)), "float16"),
            y: T.Buffer((T.int64(2), T.int64(384), T.int64(64), T.int64(64)), "float16"),
        ):
            pass
        """
        self.assertIsNone(probe._classify_full_engine_group_conv_primfunc_text(text))


if __name__ == "__main__":
    unittest.main()
