import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold96_sufficiency_transfer_v1 as stage35  # noqa: E402


class Stage35Gold96SufficiencyTransferV1Tests(unittest.TestCase):
    def test_manifest_group_split_keeps_locked_holdout_and_reference_fixed(self) -> None:
        train_pools = {
            "pyramid": [f"p{i}" for i in range(10)],
            "codriving": [f"c{i}" for i in range(10)],
        }
        references = {"pyramid": "p0", "codriving": "c0"}
        locked = ["p10", "p11", "c10", "c11"]
        train, test = stage35.manifest_group_split(
            train_pools,
            locked_groups=locked,
            reference_groups=references,
            train_count=12,
            seed=7,
        )
        self.assertEqual(len(train), 12)
        self.assertEqual(len(test), 4)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(test), set(locked))
        self.assertIn("p0", train)
        self.assertIn("c0", train)

    def test_manifest_group_split_obeys_requested_training_size(self) -> None:
        pools = {"pyramid": [f"p{i}" for i in range(10)], "codriving": [f"c{i}" for i in range(10)]}
        train, _ = stage35.manifest_group_split(
            pools,
            locked_groups=["p10", "p11", "c10", "c11"],
            reference_groups={"pyramid": "p0", "codriving": "c0"},
            train_count=6,
            seed=3,
        )
        self.assertEqual(len(train), 6)
        self.assertEqual(sum(item.startswith("p") for item in train), 3)
        self.assertEqual(sum(item.startswith("c") for item in train), 3)

    def test_affine_calibrator_recovers_scale_and_intercept(self) -> None:
        prediction = np.asarray([-1.0, 0.0, 1.0, 2.0])
        truth = 0.25 + 1.5 * prediction
        intercept, scale = stage35.fit_affine_calibrator(prediction, truth)
        self.assertAlmostEqual(intercept, 0.25)
        self.assertAlmostEqual(scale, 1.5)

    def test_graph_features_are_structural_and_model_id_free(self) -> None:
        values = stage35.graph_feature_values({
            "conv_count": 10,
            "node_count": 25,
            "group_conv_count": 2,
            "stride2_conv_count": 3,
            "kernel1_conv_count": 4,
            "kernel3_conv_count": 6,
            "conv_macs": 1024,
            "parameter_elements": 256,
            "conv_output_elements": 512,
            "input_elements": 128,
            "arithmetic_intensity_proxy": 8,
        })
        self.assertAlmostEqual(values["group_conv_ratio"], 0.2)
        self.assertAlmostEqual(values["conv_node_ratio"], 0.4)
        self.assertAlmostEqual(values["log_conv_macs"], np.log1p(1024))
        self.assertNotIn("model_id", values)

    def test_feature_contract_excludes_model_id_and_result_path_metadata(self) -> None:
        names = stage35.feature_names([], {"p": {"features": {"probe": 1.0}}}, include_graph=True)
        self.assertNotIn("model_id", names)
        self.assertNotIn("calib_percentile", names)
        self.assertTrue(any(name.startswith("graph:") for name in names))

    def test_relative_targets_use_supplied_training_centers(self) -> None:
        values = np.asarray([0.5, 0.7, 0.2])
        models = np.asarray(["p", "p", "c"])
        centers = {"p": 0.6, "c": 0.25}
        encoded = stage35.encode_targets(values, models, target="ap70", centers=centers)
        np.testing.assert_allclose(encoded, [-0.1, 0.1, -0.05])
        decoded = stage35.decode_targets(encoded, models, target="ap70", centers=centers)
        np.testing.assert_allclose(decoded, values)

    def test_log_relative_targets_round_trip(self) -> None:
        values = np.asarray([2.0, 8.0])
        models = np.asarray(["p", "c"])
        centers = {"p": np.log(4.0), "c": np.log(2.0)}
        encoded = stage35.encode_targets(values, models, target="latency_ms", centers=centers)
        decoded = stage35.decode_targets(encoded, models, target="latency_ms", centers=centers)
        np.testing.assert_allclose(decoded, values)

    def test_pareto_metrics_are_computed_within_one_model(self) -> None:
        truth = [
            {"id": "a", "ap70": 0.8, "latency_ms": 2.0, "energy_j": 2.0},
            {"id": "b", "ap70": 0.7, "latency_ms": 1.0, "energy_j": 1.0},
            {"id": "c", "ap70": 0.6, "latency_ms": 3.0, "energy_j": 3.0},
        ]
        pred = {row["id"]: dict(row) for row in truth}
        result = stage35.pareto_precision_recall(truth, pred)
        self.assertEqual(result["true_frontier"], ["a", "b"])
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
