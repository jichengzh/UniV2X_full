from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.stage2 import canonical_search_v3 as canonical
from framework.stage2 import cost_model_bundle_v3 as cost_model


def _profiles() -> list[dict]:
    common = {
        "supports_fp16_tensorcore": 1.0,
        "supports_int8_tensorcore": 1.0,
        "build_success_rate": 1.0,
    }
    return [
        canonical.build_capability_profile(
            capability_profile_id="tvm-profile",
            hardware_target="h800",
            compiler_fingerprint="a" * 64,
            dispatch_key="tvm_auto",
            features={
                **common,
                "int8_precision_propagation_ratio": 0.25,
                "qdq_fold_ratio": 0.1,
                "reformat_rate": 0.4,
            },
        ),
        canonical.build_capability_profile(
            capability_profile_id="trt-profile",
            hardware_target="h800",
            compiler_fingerprint="b" * 64,
            dispatch_key="trt_engine",
            features={
                **common,
                "int8_precision_propagation_ratio": 0.95,
                "qdq_fold_ratio": 0.9,
                "reformat_rate": 0.03,
            },
        ),
    ]


def _rows() -> list[dict]:
    rows = []
    for width, scale in [([16, 32, 64], 1.0), ([32, 64, 128], 2.0), ([48, 96, 192], 3.0)]:
        for profile_id, fp16_ratio, int8_ratio in [
            ("tvm-profile", 1.0, 1.3),
            ("trt-profile", 1.0, 0.7),
        ]:
            for q_mode, ratio in [("fp16", fp16_ratio), ("int8", int8_ratio)]:
                rows.append(
                    {
                        "row_id": f"{width}-{profile_id}-{q_mode}",
                        "width": width,
                        "q_mode": q_mode,
                        "capability_profile_id": profile_id,
                        "graph_features": {"flops_norm": scale},
                        "build_status": "success",
                        "numerical_status": "pass",
                        "latency_ms": scale * ratio,
                        "energy_j": scale * ratio * 0.1,
                        "ap70": 0.6 - scale * 0.01 - (0.01 if q_mode == "int8" else 0.0),
                    }
                )
    return rows


class Stage2CostModelBundleV3Tests(unittest.TestCase):
    def test_fit_predict_uses_numeric_capability_context_without_backend_feature(self) -> None:
        bundle = cost_model.fit_model_bundle(_rows(), _profiles(), ridge=1e-6)

        predictions = cost_model.predict_rows(bundle, _rows(), _profiles())

        self.assertIn("latency_ms", bundle["heads"])
        self.assertIn("energy_j", bundle["heads"])
        self.assertIn("ap70", bundle["heads"])
        self.assertIn("feasibility", bundle["heads"])
        self.assertNotIn("backend", " ".join(bundle["feature_schema"]["feature_order"]))
        by_id = {row["row_id"]: row for row in predictions}
        self.assertGreater(
            by_id["[32, 64, 128]-tvm-profile-int8"]["predictions"]["latency_ms"],
            by_id["[32, 64, 128]-tvm-profile-fp16"]["predictions"]["latency_ms"],
        )
        self.assertLess(
            by_id["[32, 64, 128]-trt-profile-int8"]["predictions"]["latency_ms"],
            by_id["[32, 64, 128]-trt-profile-fp16"]["predictions"]["latency_ms"],
        )

    def test_update_returns_new_bundle_and_save_load_round_trips(self) -> None:
        rows = _rows()
        first = cost_model.fit_model_bundle(rows[:8], _profiles(), ridge=1e-6)

        updated = cost_model.update_model_bundle(first, rows[8:], _profiles())

        self.assertEqual(first["training_row_count"], 8)
        self.assertEqual(updated["training_row_count"], 12)
        self.assertNotEqual(first["training_data_sha256"], updated["training_data_sha256"])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bundle.json"
            cost_model.save_model_bundle(updated, path)
            loaded = cost_model.load_model_bundle(path)
        self.assertEqual(loaded, updated)

    def test_backend_blind_projection_removes_capability_features(self) -> None:
        bundle = cost_model.fit_model_bundle(_rows(), _profiles(), ridge=1e-6)

        blind = cost_model.backend_blind_bundle(bundle)

        self.assertTrue(all(not name.startswith("cap:") for name in blind["feature_schema"]["feature_order"]))


if __name__ == "__main__":
    unittest.main()
