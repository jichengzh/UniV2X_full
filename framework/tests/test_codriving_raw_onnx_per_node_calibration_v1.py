from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_codriving_raw_onnx_per_node_calibration_v1 as calibration  # noqa: E402


class CoDrivingRawOnnxPerNodeCalibrationTests(unittest.TestCase):
    def test_build_manifest_keeps_same_shape_nodes_independent(self) -> None:
        records = [
            {
                "node_id": "node_a",
                "node_index": 1,
                "input_name": "input_a",
                "weight_name": "weight_a",
                "output_name": "output_a",
                "input_shape": [2, 32, 8, 8],
                "weight_shape": [32, 32, 3, 3],
                "output_shape": [2, 32, 8, 8],
            },
            {
                "node_id": "node_b",
                "node_index": 2,
                "input_name": "input_b",
                "weight_name": "weight_b",
                "output_name": "output_b",
                "input_shape": [2, 32, 8, 8],
                "weight_shape": [32, 32, 3, 3],
                "output_shape": [2, 32, 8, 8],
            },
        ]
        manifest = calibration.build_manifest(
            records=records,
            input_absmax={"input_a": 12.7, "input_b": 25.4},
            weights={
                "weight_a": np.asarray([-0.127, 0.254], dtype="float32"),
                "weight_b": np.asarray([-0.508, 0.127], dtype="float32"),
            },
            onnx_path=Path("/tmp/model.onnx"),
            onnx_sha256="a" * 64,
            calibration_source=Path("/tmp/calib.npz"),
            calibration_source_sha256="b" * 64,
            calibration_summary=Path("/tmp/calib_summary.json"),
            calibration_summary_sha256="c" * 64,
            calibration_split_source=Path("/data/train.json"),
            sample_count=16,
        )
        self.assertAlmostEqual(manifest["nodes"]["node_a"]["input_scale"], 0.1)
        self.assertAlmostEqual(manifest["nodes"]["node_b"]["input_scale"], 0.2)
        self.assertAlmostEqual(manifest["nodes"]["node_a"]["weight_scale"], 0.002)
        self.assertAlmostEqual(manifest["nodes"]["node_b"]["weight_scale"], 0.004)
        calibration.validate_manifest(manifest, expected_node_ids={"node_a", "node_b"})

    def test_validate_manifest_rejects_missing_node(self) -> None:
        with self.assertRaisesRegex(ValueError, "node coverage"):
            calibration.validate_manifest(
                {
                    "schema": calibration.SCHEMA,
                    "onnx_sha256": "a" * 64,
                    "sample_count": 16,
                    "calibration_split": "train",
                    "calibration_source_sha256": "b" * 64,
                    "calibration_summary_sha256": "c" * 64,
                    "nodes": {},
                },
                expected_node_ids={"node_a"},
            )

    def test_validate_source_summary_requires_exact_train_16(self) -> None:
        with self.assertRaisesRegex(ValueError, "collected_samples"):
            calibration.validate_source_summary(
                {
                    "schema": "v2_gold_coldstart_96_codriving_calib_export_v1",
                    "calibration_split": "train",
                    "collected_samples": 16.0,
                    "shapes": {"spatial_features": [16, 2, 64, 256, 512]},
                    "split_source_file": "/data/train.json",
                    "output_sha256": "a" * 64,
                }
            )


if __name__ == "__main__":
    unittest.main()
