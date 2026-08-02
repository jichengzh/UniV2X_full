from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/stage35_prepare_pyramid_trt_calibration_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("stage35_prepare_pyramid_trt_calibration_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage35PreparePyramidTrtCalibrationV1Tests(unittest.TestCase):
    def test_pairs_real_activations_and_repeats_last_odd_sample(self) -> None:
        module = _module()
        source = np.arange(5 * 3 * 2 * 4, dtype=np.float16).reshape(5, 3, 2, 4)

        batches = module.build_batch2_calibration(source, expected_chw=(3, 2, 4))

        self.assertEqual(batches.shape, (3, 2, 3, 2, 4))
        self.assertEqual(batches.dtype, np.float32)
        np.testing.assert_array_equal(batches[0], source[:2].astype(np.float32))
        np.testing.assert_array_equal(batches[-1, 0], source[-1].astype(np.float32))
        np.testing.assert_array_equal(batches[-1, 1], source[-1].astype(np.float32))

    def test_writes_one_npy_per_batch_with_stable_names(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "calibration.npz"
            output = root / "trt_npy"
            np.savez_compressed(source, spatial_features=np.zeros((5, 3, 2, 4), dtype=np.float16))

            output.mkdir()
            np.save(output / "batch2_999.npy", np.ones((2, 3, 2, 4), dtype=np.float32))

            report = module.write_trt_calibration(source, output, expected_chw=(3, 2, 4))

            self.assertEqual(report["input_count"], 5)
            self.assertEqual(report["batch_count"], 3)
            self.assertEqual(report["sample_shape"], [2, 3, 2, 4])
            self.assertEqual([path.name for path in sorted(output.glob("*.npy"))], [
                "batch2_000.npy", "batch2_001.npy", "batch2_002.npy"
            ])

    def test_default_contract_rejects_non_pyramid_feature_shape(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "calibration.npz"
            np.savez_compressed(source, spatial_features=np.zeros((4, 3, 2, 4), dtype=np.float16))

            with self.assertRaisesRegex(ValueError, "expected activation CHW"):
                module.write_trt_calibration(source, root / "trt_npy")


if __name__ == "__main__":
    unittest.main()
