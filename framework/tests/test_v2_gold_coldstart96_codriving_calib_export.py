from __future__ import annotations

import sys
import inspect
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96_codriving_calib_export as calib_export  # noqa: E402


class V2GoldColdstart96CoDrivingCalibExportTests(unittest.TestCase):
    def test_validate_calib_arrays_accepts_expected_shapes(self) -> None:
        spatial = np.zeros((2, 2, 64, 256, 512), dtype=np.float16)
        tmat = np.zeros((2, 1, 2, 2, 4, 4), dtype=np.float32)

        shapes = calib_export.validate_calib_arrays(spatial, tmat)

        self.assertEqual(shapes["spatial_features"], [2, 2, 64, 256, 512])
        self.assertEqual(shapes["pairwise_t_matrix"], [2, 1, 2, 2, 4, 4])

    def test_validate_calib_arrays_rejects_mismatched_sample_count(self) -> None:
        spatial = np.zeros((2, 2, 64, 256, 512), dtype=np.float16)
        tmat = np.zeros((1, 1, 2, 2, 4, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "same N"):
            calib_export.validate_calib_arrays(spatial, tmat)

    def test_build_report_records_dataset_and_output_metadata(self) -> None:
        report = calib_export.build_report(
            width="16x32x64",
            model_dir=Path("/tmp/model"),
            output=Path("/tmp/calib.npz"),
            requested_samples=4,
            collected_samples=3,
            skipped_non_2_agent=2,
            calibration_split="train",
            split_source_file=Path("/data/train.json"),
            output_sha256="a" * 64,
            shapes={
                "spatial_features": [3, 2, 64, 256, 512],
                "pairwise_t_matrix": [3, 1, 2, 2, 4, 4],
            },
            elapsed_secs=1.5,
        )

        self.assertEqual(report["schema"], "v2_gold_coldstart_96_codriving_calib_export_v1")
        self.assertEqual(report["width"], "16x32x64")
        self.assertEqual(report["model_dir"], "/tmp/model")
        self.assertEqual(report["output"], "/tmp/calib.npz")
        self.assertEqual(report["requested_samples"], 4)
        self.assertEqual(report["collected_samples"], 3)
        self.assertEqual(report["skipped_non_2_agent"], 2)
        self.assertEqual(report["calibration_split"], "train")
        self.assertEqual(report["split_source_file"], "/data/train.json")
        self.assertEqual(report["output_sha256"], "a" * 64)

    def test_select_calibration_hypes_uses_train_split_without_mutating_input(self) -> None:
        original = {
            "root_dir": "/data/train.json",
            "validate_dir": "/data/val.json",
            "test_dir": "/data/val.json",
        }

        selected = calib_export.select_calibration_hypes(original, "train")

        self.assertEqual(selected["validate_dir"], "/data/train.json")
        self.assertEqual(selected["test_dir"], "/data/val.json")
        self.assertEqual(original["validate_dir"], "/data/val.json")

    def test_collect_calib_data_does_not_override_selected_split(self) -> None:
        source = inspect.getsource(calib_export.collect_calib_data)

        self.assertNotIn('hypes["validate_dir"] =', source)


if __name__ == "__main__":
    unittest.main()
