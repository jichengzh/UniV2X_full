from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_smbo_qxs_entry_v2 as qxs_v2  # noqa: E402


def _backend_scan() -> dict:
    return {
        "schema_version": "h800_compiler_backend_capability_scan_v1",
        "hardware_target": "h800",
        "compiler_backend_profiles": [
            {
                "compiler_profile_id": "h800_tvm_routeb_20260708",
                "backend": "tvm",
                "supports_fp16": True,
                "supports_int8": True,
                "supports_mixed_int8": True,
                "supports_cutlass_byoc": False,
                "supports_requant_fusion": "partial",
                "supports_cast_fusion": "partial",
                "measurement_confidence": "real_measurement_summary",
            },
            {
                "compiler_profile_id": "h800_trt_20260708",
                "backend": "tensorrt",
                "supports_fp16": True,
                "supports_int8": True,
                "supports_mixed_int8": "unknown",
                "supports_cutlass_byoc": False,
                "supports_requant_fusion": "backend_managed_qdq",
                "supports_cast_fusion": "backend_managed_qdq",
                "measurement_confidence": "real_measurement_summary",
            },
        ],
    }


class Stage2QxSSearchEntryV2Tests(unittest.TestCase):
    def test_strategy_manifest_keeps_backend_and_graph_boundary_out_of_strategy_id(self) -> None:
        manifest = qxs_v2.build_strategy_manifest(
            widths=["16x32x64", "32x32x128"],
            backend_scan=_backend_scan(),
        )

        self.assertEqual(manifest["genome_schema"], ["w0", "w1", "w2", "strategy_id"])
        self.assertEqual(manifest["strategy_space"]["q_mode"], ["fp16", "int8"])
        self.assertNotIn("graph_boundary", manifest["strategy_space"])
        self.assertEqual(
            [item["strategy_id"] for item in manifest["strategies"]],
            [
                "q=fp16|mixed=none",
                "q=int8|mixed=top10_flops",
                "q=int8|mixed=top25_flops",
                "q=int8|mixed=top50_flops",
                "q=int8|mixed=top75_flops",
                "q=int8|mixed=all_eligible_conv",
            ],
        )
        for item in manifest["strategies"]:
            self.assertNotIn("tvm", item["strategy_id"])
            self.assertNotIn("trt", item["strategy_id"])
            self.assertNotIn("cutlass", item["strategy_id"])
            self.assertNotIn("boundary", item["strategy_id"])

    def test_feature_encoding_uses_backend_as_context_not_genome(self) -> None:
        rows = [
            {
                "width": "16x32x64",
                "q_mode": "int8",
                "mixed_policy_id": "top25_flops",
                "compiler_profile_id": "h800_tvm_routeb_20260708",
                "backend": "tvm",
                "latency_ms": 2.48,
                "trusted_for_final_frontier": True,
                "derived_graph_features": {
                    "dtype_boundary_count": None,
                    "requant_count": None,
                    "cast_count": None,
                    "fusion_break_count": None,
                    "mixed_int8_conv_count": 3,
                    "mixed_fp16_conv_count": 9,
                    "feature_status": "partial_raw_counts_only",
                },
            }
        ]

        spec = qxs_v2.build_feature_spec(_backend_scan(), rows)
        encoded = qxs_v2.encode_training_row(rows[0], spec)

        self.assertEqual(encoded["genome"], [16, 32, 64, "q=int8|mixed=top25_flops"])
        self.assertEqual(encoded["source"]["strategy_id"], "q=int8|mixed=top25_flops")
        self.assertEqual(encoded["source"]["compiler_profile_id"], "h800_tvm_routeb_20260708")
        self.assertNotIn("backend", spec["genome_schema"])
        self.assertIn("backend=tvm", spec["feature_order"])
        self.assertIn("supports_int8", spec["feature_order"])
        self.assertEqual(encoded["derived_graph_features"]["mixed_int8_conv_count"], 3)

    def test_context_training_rows_from_summary_include_fp16_int8_and_mixed_strategies(self) -> None:
        summary_rows = [
            {
                "width_key": "16x32x64",
                "fp16_ms": 2.5,
                "int8_ms": 2.9,
                "mixed_top25_flops_ms": 2.4,
                "mixed_top25_flops_int8_convs": 3,
                "mixed_top25_flops_fp16_convs": 9,
            }
        ]

        rows = qxs_v2.training_rows_from_tvm_mixed_summary_rows(summary_rows)

        self.assertEqual(
            [(row["q_mode"], row["mixed_policy_id"], row["latency_ms"]) for row in rows],
            [
                ("fp16", "none", 2.5),
                ("int8", "all_eligible_conv", 2.9),
                ("int8", "top25_flops", 2.4),
            ],
        )
        self.assertTrue(all(row["compiler_profile_id"] == "h800_tvm_routeb_20260708" for row in rows))
        self.assertEqual(rows[-1]["derived_graph_features"]["mixed_int8_conv_count"], 3)

    def test_latency_learning_smoke_consumes_v2_feature_rows(self) -> None:
        training_rows = [
            {
                "width": "16x32x64",
                "q_mode": "fp16",
                "mixed_policy_id": "none",
                "compiler_profile_id": "h800_tvm_routeb_20260708",
                "backend": "tvm",
                "latency_ms": 2.0,
                "trusted_for_final_frontier": True,
            },
            {
                "width": "16x32x64",
                "q_mode": "int8",
                "mixed_policy_id": "all_eligible_conv",
                "compiler_profile_id": "h800_trt_20260708",
                "backend": "tensorrt",
                "latency_ms": 1.0,
                "trusted_for_final_frontier": True,
            },
            {
                "width": "32x32x128",
                "q_mode": "fp16",
                "mixed_policy_id": "none",
                "compiler_profile_id": "h800_tvm_routeb_20260708",
                "backend": "tvm",
                "latency_ms": 4.0,
                "trusted_for_final_frontier": True,
            },
            {
                "width": "32x32x128",
                "q_mode": "int8",
                "mixed_policy_id": "all_eligible_conv",
                "compiler_profile_id": "h800_trt_20260708",
                "backend": "tensorrt",
                "latency_ms": 2.0,
                "trusted_for_final_frontier": True,
            },
        ]
        manifest = qxs_v2.build_strategy_manifest(["16x32x64", "32x32x128"], _backend_scan())
        entry = qxs_v2.build_entry(
            backend_scan=_backend_scan(),
            strategy_manifest=manifest,
            training_rows=training_rows,
        )

        metrics = qxs_v2.evaluate_latency_learning_smoke(entry)

        self.assertEqual(metrics["n_rows"], 4)
        self.assertEqual(metrics["cv_group"], "width")
        self.assertIn("backend=tensorrt", metrics["feature_order"])
        self.assertLessEqual(metrics["train_mae_ms"], metrics["mean_baseline_mae_ms"])


if __name__ == "__main__":
    unittest.main()
