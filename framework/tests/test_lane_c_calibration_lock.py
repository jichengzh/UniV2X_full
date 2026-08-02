from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.orin_deploy.lane_c_calibration_lock import (
    build_canonical_manifest,
    calibration_payload_id,
)
from tools.orin_deploy.lane_c_backbone_parity_runner import sha256_file


class LaneCCalibrationLockTest(unittest.TestCase):
    def test_manifest_is_path_stable_and_payload_id_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(15):
                np.save(
                    root / f"batch2_{index:03d}.npy",
                    np.full((2, 64, 2, 2), index, dtype=np.float32),
                )

            first = build_canonical_manifest(
                calibration_dir=root,
                checkpoint_sha256="a" * 64,
                onnx_sha256="b" * 64,
                expected_shape=(2, 64, 2, 2),
            )
            second = build_canonical_manifest(
                calibration_dir=root,
                checkpoint_sha256="a" * 64,
                onnx_sha256="b" * 64,
                expected_shape=(2, 64, 2, 2),
            )

            self.assertEqual(first, second)
            self.assertNotIn(str(root), json.dumps(first))
            self.assertEqual(first["calibration"]["file_count"], 15)
            self.assertEqual(
                first["calibration"]["payload_id"],
                calibration_payload_id(first["calibration"]["files"]),
            )
            self.assertEqual(
                first["calibration"]["files"][0]["sha256"],
                sha256_file(root / "batch2_000.npy"),
            )

    def test_manifest_rejects_extra_npy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(16):
                np.save(
                    root / f"batch2_{index:03d}.npy",
                    np.zeros((2, 64, 2, 2), dtype=np.float32),
                )
            with self.assertRaisesRegex(ValueError, "exactly 15"):
                build_canonical_manifest(
                    calibration_dir=root,
                    checkpoint_sha256="a" * 64,
                    onnx_sha256="b" * 64,
                    expected_shape=(2, 64, 2, 2),
                )


if __name__ == "__main__":
    unittest.main()
