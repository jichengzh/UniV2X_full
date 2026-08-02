from __future__ import annotations

import unittest

import numpy as np

from tools.orin_deploy.lane_c_backbone_rpc import (
    decode_array_bundle,
    encode_array_bundle,
    validate_bind_host,
)


class LaneCBackboneRpcTest(unittest.TestCase):
    def test_array_bundle_round_trip_preserves_names_shapes_and_values(self) -> None:
        arrays = {
            "pyramid_level0": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "pyramid_level1": np.zeros((2, 2, 2), dtype=np.float32),
        }
        decoded = decode_array_bundle(encode_array_bundle(arrays))
        self.assertEqual(list(decoded), list(arrays))
        for name in arrays:
            np.testing.assert_array_equal(decoded[name], arrays[name])

    def test_bundle_decoder_rejects_oversized_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload"):
            decode_array_bundle(b"x" * 64, max_uncompressed_bytes=32)

    def test_bind_policy_allows_only_loopback(self) -> None:
        validate_bind_host("127.0.0.1")
        with self.assertRaisesRegex(ValueError, "loopback"):
            validate_bind_host("192.0.2.1")
        with self.assertRaisesRegex(ValueError, "loopback"):
            validate_bind_host("8.8.8.8")


if __name__ == "__main__":
    unittest.main()
