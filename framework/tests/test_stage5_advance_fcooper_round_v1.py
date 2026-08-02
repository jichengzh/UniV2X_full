import unittest
import hashlib
import json

from scripts.stage5_advance_fcooper_round_v1 import (
    merge_actual_graph_features,
    validate_fcooper_feedback_evidence,
    validate_fcooper_registry,
)


class FCooperRoundAdvanceTest(unittest.TestCase):
    def test_actual_feedback_overrides_scanner_graph_prior(self) -> None:
        base = [
            {"group_id": "gold-a", "model": "pyramid", "conv_count": 1},
            {"group_id": "fcooper-a", "model": "fcooper", "conv_count": 31},
        ]
        feedback = [
            {
                "group_id": "fcooper-a",
                "model": "fcooper",
                "graph_features": {
                    "group_id": "fcooper-a",
                    "model": "fcooper",
                    "conv_count": 29,
                    "graph_feature_provenance": "materialized_onnx_extracted_v1",
                },
            }
        ]

        merged = merge_actual_graph_features(base, feedback)

        by_group = {row["group_id"]: row for row in merged}
        self.assertEqual(by_group["gold-a"]["conv_count"], 1)
        self.assertEqual(by_group["fcooper-a"]["conv_count"], 29)
        self.assertEqual(
            by_group["fcooper-a"]["graph_feature_provenance"],
            "materialized_onnx_extracted_v1",
        )

    def test_registry_requires_five_width_schema_and_unique_groups(self) -> None:
        registry = {
            "model": "fcooper",
            "width_schema": [
                "backbone.s0",
                "backbone.s1",
                "backbone.s2",
                "neck.deblock",
                "neck.output",
            ],
            "groups": [
                {
                    "group_id": "a",
                    "model": "fcooper",
                    "width": [64, 128, 256, 128, 256],
                    "width_schema": [
                        "backbone.s0",
                        "backbone.s1",
                        "backbone.s2",
                        "neck.deblock",
                        "neck.output",
                    ],
                }
            ],
        }

        audit = validate_fcooper_registry(registry)

        self.assertEqual(audit["group_count"], 1)
        self.assertEqual(audit["genome_count"], 2)

    def test_registry_rejects_duplicate_group_ids(self) -> None:
        group = {
            "group_id": "a",
            "model": "fcooper",
            "width": [64, 128, 256, 128, 256],
            "width_schema": [
                "backbone.s0",
                "backbone.s1",
                "backbone.s2",
                "neck.deblock",
                "neck.output",
            ],
        }
        registry = {
            "model": "fcooper",
            "width_schema": group["width_schema"],
            "groups": [group, dict(group)],
        }

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_fcooper_registry(registry)

    def test_feedback_evidence_rejects_mutated_actual_graph(self) -> None:
        feedback = {
            "row_id": "row-a",
            "model": "fcooper",
            "group_id": "group-a",
            "graph_features": {
                "group_id": "group-a",
                "model": "fcooper",
                "graph_feature_provenance": "materialized_onnx_extracted_v1",
                "conv_count": 21,
            },
            "materialized_graph_features_sha256": "0" * 64,
            "actual_feedback_row_sha256": "1" * 64,
        }

        with self.assertRaisesRegex(ValueError, "graph feature SHA drift"):
            validate_fcooper_feedback_evidence([feedback])

    def test_tvm_feedback_accepts_tvm_artifact_and_actual_prepared_graph(self) -> None:
        graph = {
            "group_id": "group-a",
            "model": "fcooper",
            "graph_feature_provenance": "materialized_tvm_prepared_onnx_extracted_v1",
            "conv_count": 24,
        }
        digest = lambda value: hashlib.sha256(
            json.dumps(
                value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        feedback = {
            "row_id": "row-a",
            "model": "fcooper",
            "group_id": "group-a",
            "dispatch_key": "tvm_auto",
            "terminal_status": "measured_success_gold",
            "graph_features": graph,
            "materialized_graph_features_sha256": digest(graph),
            "performance_result_sha256": "a" * 64,
            "ap_report_sha256": "b" * 64,
            "materialized_source_evidence_sha256": "c" * 64,
            "tvm_artifact_sha256": "d" * 64,
        }
        feedback["actual_feedback_row_sha256"] = digest(feedback)

        audit = validate_fcooper_feedback_evidence([feedback])
        merged = merge_actual_graph_features([], [feedback])

        self.assertEqual(audit["verified_actual_feedback_rows"], 1)
        self.assertEqual(merged[0]["graph_feature_provenance"], graph["graph_feature_provenance"])


if __name__ == "__main__":
    unittest.main()
