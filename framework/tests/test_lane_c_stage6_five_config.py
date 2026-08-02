from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.orin_deploy.lane_c_stage6_five_config import (
    CONFIG_SPECS,
    materialize_calibration,
    validate_config_specs,
)
from tools.orin_deploy.lane_c_native_backbone_orin import select_native_input


class LaneCStage6FiveConfigTest(unittest.TestCase):
    def test_manifest_has_exactly_five_executable_arms_per_model(self) -> None:
        validate_config_specs(CONFIG_SPECS)
        self.assertEqual(len(CONFIG_SPECS), 10)
        for model in ("pyramid", "codriving"):
            rows = [row for row in CONFIG_SPECS if row["model"] == model]
            self.assertEqual(
                [row["arm"] for row in rows],
                [
                    "original_default",
                    "compression_only",
                    "schedule_only",
                    "compress_then_tune",
                    "joint_shcosearch",
                ],
            )

    def test_pyramid_calibration_pairs_agents_and_repeats_odd_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            activations = np.arange(
                5 * 64 * 2 * 2, dtype=np.float16
            ).reshape(5, 64, 2, 2)
            np.savez(source, spatial_features=activations)
            manifest_path = materialize_calibration(
                source_npz=source,
                output_dir=root / "out",
                expected_input_shape=(2, 64, 2, 2),
                source_layout="agent_rows",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["calibration"]["file_count"], 3)
            tail = np.load(root / "out" / "batch2_002.npy", allow_pickle=False)
            self.assertEqual(tail.dtype, np.float32)
            np.testing.assert_array_equal(tail[0], activations[-1])
            np.testing.assert_array_equal(tail[1], activations[-1])

    def test_codriving_calibration_preserves_all_batch2_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            batches = np.arange(
                3 * 2 * 64 * 2 * 2, dtype=np.float16
            ).reshape(3, 2, 64, 2, 2)
            np.savez(source, spatial_features=batches)
            manifest_path = materialize_calibration(
                source_npz=source,
                output_dir=root / "out",
                expected_input_shape=(2, 64, 2, 2),
                source_layout="batch2_rows",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["calibration"]["file_count"], 3)
            for index in range(3):
                actual = np.load(
                    root / "out" / f"batch2_{index:03d}.npy",
                    allow_pickle=False,
                )
                np.testing.assert_array_equal(actual, batches[index])

    def test_native_latency_selects_the_only_batch_from_trt_input_file(self) -> None:
        inputs = np.zeros((1, 2, 64, 2, 2), dtype=np.float32)
        selected = select_native_input(inputs)
        self.assertEqual(selected.shape, (2, 64, 2, 2))
        with self.assertRaisesRegex(ValueError, "exactly one batch"):
            select_native_input(np.zeros((2, 2, 64, 2, 2), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
