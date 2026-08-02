from __future__ import annotations

import unittest

from scripts.stage5_finalize_pyramid_actual_v3 import validate_actual_feedback_rows


class FinalizePyramidActualV3Tests(unittest.TestCase):
    def test_requires_materialized_actual_features_without_fallback(self) -> None:
        rows = [{
            "manifest_job_id": "pyramid|16x32x64|q=fp16|profile=tvm",
            "feedback_feature_contract": "actual_feedback_v3",
            "materialized_graph_features_sha256": "a" * 64,
            "graph_features": {
                "graph_feature_provenance": "materialized_onnx_extracted_v1",
            },
        }]

        self.assertEqual(validate_actual_feedback_rows(rows), 1)

        rows[0]["graph_features"] = {
            "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
        }
        with self.assertRaisesRegex(ValueError, "actual graph feature provenance"):
            validate_actual_feedback_rows(rows)


if __name__ == "__main__":
    unittest.main()
