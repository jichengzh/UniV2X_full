from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.stage2 import measurement_contract_v3 as contract  # noqa: E402


def _base_row() -> dict:
    row = {
        "schema_version": "stage2_measurement_row_v3",
        "row_id": "codriving-32x64x128-fp16",
        "model": "codriving",
        "dataset": "dair-v2x-c",
        "checkpoint_id": "codriving-gold",
        "width": [32, 64, 128],
        "q_mode": "fp16",
        "mixed_policy_id": "none",
        "capability_profile_id": "h800-tvm-profile",
        "pipeline": {
            "onnx_sha256": "a" * 64,
            "calibration_sha256": None,
            "calibration_kind": "none",
            "lowering_origin": "source_ir_automatic",
            "source_ir_sha256": "c" * 64,
            "schedule_trace_sha256": "d" * 64,
            "tuning_database_sha256": "e" * 64,
            "compiler_fingerprint": "b" * 64,
            "optimized_scope": "backbone_only",
            "input_shape": [2, 64, 256, 512],
            "batch_size": 2,
            "tuning_budget": {"trials": 64, "repeat": 3},
        },
        "statuses": {"build": "success", "numerical": "pass"},
        "provenance": {
            "evidence_kind": "measured",
            "source_artifacts": [
                {
                    "role": "measurement_report",
                    "path": str(Path(__file__).resolve()),
                    "sha256": contract.sha256_file(Path(__file__).resolve()),
                }
            ],
        },
        "trusted_for_final_frontier": True,
    }
    fingerprint = contract.compute_pipeline_fingerprint(row)
    row["pipeline_fingerprint"] = fingerprint
    row["metrics"] = {
        "latency": {
            "status": "measured",
            "p50_ms": 1.8,
            "p90_ms": 1.9,
            "pipeline_fingerprint": fingerprint,
        },
        "energy": {
            "status": "measured",
            "joules_per_inference": 0.4,
            "pipeline_fingerprint": fingerprint,
        },
        "ap": {
            "status": "measured",
            "ap30": 0.72,
            "ap50": 0.64,
            "ap70": 0.55,
            "pipeline_fingerprint": fingerprint,
        },
    }
    return row


def _engine_row() -> dict:
    row = _base_row()
    artifact_path = Path(__file__).resolve()
    artifact_sha = contract.sha256_file(artifact_path)
    row["row_id"] = "codriving-32x64x128-trt-int8"
    row["q_mode"] = "int8"
    row["mixed_policy_id"] = "none"
    row["capability_profile_id"] = "h800-trt-profile"
    row["pipeline"].update(
        {
            "calibration_sha256": "f" * 64,
            "calibration_kind": "formal",
            "lowering_origin": "compiler_engine_automatic",
            "source_ir_sha256": None,
            "schedule_trace_sha256": None,
            "tuning_database_sha256": None,
            "engine_build_config_sha256": artifact_sha,
            "engine_inspector_sha256": artifact_sha,
            "compiled_engine_sha256": artifact_sha,
        }
    )
    row["provenance"]["source_artifacts"] = [
        {"role": role, "path": str(artifact_path), "sha256": artifact_sha}
        for role in ("engine_build_config", "engine_inspector", "compiled_engine")
    ]
    row["pipeline_fingerprint"] = contract.compute_pipeline_fingerprint(row)
    for metric in row["metrics"].values():
        metric["pipeline_fingerprint"] = row["pipeline_fingerprint"]
    return row


class Stage2MeasurementContractV3Tests(unittest.TestCase):
    def test_compiler_engine_automatic_is_valid_gold_provenance(self) -> None:
        row = _engine_row()

        contract.validate_measurement_row(row)

        self.assertTrue(contract.is_search_training_eligible(row))
        self.assertTrue(contract.is_final_frontier_eligible(row))

    def test_compiler_engine_automatic_requires_build_inspector_and_engine_digests(self) -> None:
        for field in (
            "engine_build_config_sha256",
            "engine_inspector_sha256",
            "compiled_engine_sha256",
        ):
            with self.subTest(field=field):
                row = _engine_row()
                row["pipeline"][field] = None
                row["pipeline_fingerprint"] = contract.compute_pipeline_fingerprint(row)
                for metric in row["metrics"].values():
                    metric["pipeline_fingerprint"] = row["pipeline_fingerprint"]
                with self.assertRaisesRegex(ValueError, field):
                    contract.validate_measurement_row(row)

    def test_compiler_engine_automatic_requires_matching_engine_artifact_roles(self) -> None:
        row = _engine_row()
        row["provenance"]["source_artifacts"] = row["provenance"]["source_artifacts"][:-1]

        with self.assertRaisesRegex(ValueError, "compiled_engine"):
            contract.validate_measurement_row(row)

    def test_evidence_views_are_mutually_exclusive(self) -> None:
        frontier = _base_row()
        training = _base_row()
        training["row_id"] = "training"
        training["trusted_for_final_frontier"] = False
        for metric_name in ("energy", "ap"):
            training["metrics"][metric_name] = {
                "status": "not_measured",
                "pipeline_fingerprint": training["pipeline_fingerprint"],
            }
        failed = _base_row()
        failed["row_id"] = "failed"
        failed["trusted_for_final_frontier"] = False
        failed["statuses"]["build"] = "failed"
        historical = _base_row()
        historical["row_id"] = "historical"
        historical["trusted_for_final_frontier"] = False
        historical["provenance"]["evidence_kind"] = "historical"

        views = contract.partition_measurement_rows([frontier, training, failed, historical])

        self.assertEqual([row["row_id"] for row in views["final_frontier"]], [frontier["row_id"]])
        self.assertEqual([row["row_id"] for row in views["gold_training"]], [training["row_id"]])
        self.assertEqual([row["row_id"] for row in views["feasibility_failure"]], [failed["row_id"]])
        self.assertEqual([row["row_id"] for row in views["historical_prior"]], [historical["row_id"]])
        self.assertEqual(sum(len(rows) for rows in views.values()), 4)
    def test_valid_gold_row_is_final_frontier_eligible(self) -> None:
        row = _base_row()

        validated = contract.validate_measurement_row(row)

        self.assertEqual(validated, row)
        self.assertTrue(contract.is_final_frontier_eligible(row))
        self.assertEqual(contract.final_frontier_rejection_reasons(row), [])

    def test_metric_pipeline_mismatch_is_rejected(self) -> None:
        row = _base_row()
        row["metrics"]["energy"]["pipeline_fingerprint"] = "c" * 64

        with self.assertRaisesRegex(ValueError, "energy.*pipeline_fingerprint"):
            contract.validate_measurement_row(row)

    def test_historical_row_cannot_claim_final_frontier_trust(self) -> None:
        row = _base_row()
        row["provenance"]["evidence_kind"] = "historical"

        with self.assertRaisesRegex(ValueError, "historical.*trusted_for_final_frontier"):
            contract.validate_measurement_row(row)

    def test_diagnostic_calibration_cannot_enter_final_frontier(self) -> None:
        row = _base_row()
        row["q_mode"] = "mixed_int8"
        row["mixed_policy_id"] = "top25_flops"
        row["pipeline"]["calibration_sha256"] = "c" * 64
        row["pipeline"]["calibration_kind"] = "diagnostic"
        row["pipeline_fingerprint"] = contract.compute_pipeline_fingerprint(row)
        for metric in row["metrics"].values():
            metric["pipeline_fingerprint"] = row["pipeline_fingerprint"]

        with self.assertRaisesRegex(ValueError, "diagnostic calibration"):
            contract.validate_measurement_row(row)

    def test_diagnostic_calibration_cannot_enter_search_training(self) -> None:
        row = _base_row()
        row["q_mode"] = "mixed_int8"
        row["mixed_policy_id"] = "top25_flops"
        row["trusted_for_final_frontier"] = False
        row["pipeline"]["calibration_sha256"] = "c" * 64
        row["pipeline"]["calibration_kind"] = "diagnostic"
        row["pipeline_fingerprint"] = contract.compute_pipeline_fingerprint(row)
        for metric in row["metrics"].values():
            metric["pipeline_fingerprint"] = row["pipeline_fingerprint"]

        contract.validate_measurement_row(row)
        self.assertFalse(contract.is_search_training_eligible(row))

    def test_diagnostic_metric_cannot_enter_gold_training(self) -> None:
        for metric_name in ("energy", "ap"):
            with self.subTest(metric_name=metric_name):
                row = _base_row()
                row["trusted_for_final_frontier"] = False
                row["metrics"][metric_name]["status"] = "diagnostic"

                self.assertFalse(contract.is_search_training_eligible(row))
                self.assertEqual(contract.measurement_evidence_view(row), "historical_prior")

    def test_historical_calibration_cannot_enter_final_frontier(self) -> None:
        row = _base_row()
        row["q_mode"] = "int8"
        row["mixed_policy_id"] = "all"
        row["pipeline"]["calibration_sha256"] = "c" * 64
        row["pipeline"]["calibration_kind"] = "historical"
        row["pipeline_fingerprint"] = contract.compute_pipeline_fingerprint(row)
        for metric in row["metrics"].values():
            metric["pipeline_fingerprint"] = row["pipeline_fingerprint"]

        with self.assertRaisesRegex(ValueError, "historical calibration"):
            contract.validate_measurement_row(row)

    def test_explicit_te_lowering_is_diagnostic_only(self) -> None:
        row = _base_row()
        row["pipeline"]["lowering_origin"] = "explicit_te_constructed"
        row["pipeline_fingerprint"] = contract.compute_pipeline_fingerprint(row)
        for metric in row["metrics"].values():
            metric["pipeline_fingerprint"] = row["pipeline_fingerprint"]

        with self.assertRaisesRegex(ValueError, "explicit TE lowering"):
            contract.validate_measurement_row(row)

        row["trusted_for_final_frontier"] = False
        validated = contract.validate_measurement_row(row)
        self.assertEqual(validated["pipeline"]["lowering_origin"], "explicit_te_constructed")
        self.assertIn("lowering_not_automatic", contract.final_frontier_rejection_reasons(row))
        self.assertFalse(contract.is_search_training_eligible(row))

    def test_source_ir_automatic_latency_row_can_train_cost_model_without_ap(self) -> None:
        row = _base_row()
        row["trusted_for_final_frontier"] = False
        row["metrics"]["energy"]["status"] = "not_measured"
        row["metrics"]["energy"].pop("joules_per_inference")
        row["metrics"]["ap"]["status"] = "not_measured"
        for field in ("ap30", "ap50", "ap70"):
            row["metrics"]["ap"].pop(field)

        self.assertTrue(contract.is_search_training_eligible(row))

    def test_artifact_sha_mismatch_is_rejected(self) -> None:
        row = _base_row()
        row["provenance"]["source_artifacts"][0]["sha256"] = "f" * 64

        with self.assertRaisesRegex(ValueError, "artifact.*SHA256 mismatch"):
            contract.validate_measurement_row(row)

    def test_numerical_failure_is_preserved_but_not_final_gold(self) -> None:
        row = _base_row()
        row["statuses"]["numerical"] = "fail"
        row["trusted_for_final_frontier"] = False

        validated = contract.validate_measurement_row(row)

        self.assertEqual(validated["statuses"]["numerical"], "fail")
        self.assertFalse(contract.is_final_frontier_eligible(row))
        self.assertIn("numerical_not_pass", contract.final_frontier_rejection_reasons(row))

    def test_fingerprint_changes_with_pipeline_but_not_metrics(self) -> None:
        row = _base_row()
        original = contract.compute_pipeline_fingerprint(row)
        changed_metric = copy.deepcopy(row)
        changed_metric["metrics"]["latency"]["p50_ms"] = 99.0
        changed_pipeline = copy.deepcopy(row)
        changed_pipeline["pipeline"]["compiler_fingerprint"] = "d" * 64

        self.assertEqual(contract.compute_pipeline_fingerprint(changed_metric), original)
        self.assertNotEqual(contract.compute_pipeline_fingerprint(changed_pipeline), original)

    def test_summary_counts_gold_and_rejection_reasons(self) -> None:
        gold = _base_row()
        failed = _base_row()
        failed["row_id"] = "failed"
        failed["statuses"]["build"] = "failed"
        failed["trusted_for_final_frontier"] = False

        summary = contract.summarize_measurement_registry([gold, failed])

        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["final_frontier_eligible_rows"], 1)
        self.assertEqual(summary["rejection_reason_counts"]["build_not_success"], 1)

    def test_validation_cli_writes_only_gold_frontier_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "rows.jsonl"
            summary = root / "summary.json"
            frontier = root / "frontier.jsonl"
            registry = root / "artifacts.json"
            views = root / "evidence_views.json"
            gold = _base_row()
            failed = _base_row()
            failed["row_id"] = "failed"
            failed["statuses"]["numerical"] = "fail"
            failed["trusted_for_final_frontier"] = False
            source.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in (gold, failed)) + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "stage2_validate_measurement_rows_v3.py"),
                    "--rows-jsonl",
                    str(source),
                    "--summary-json",
                    str(summary),
                    "--frontier-jsonl",
                    str(frontier),
                    "--artifact-registry-json",
                    str(registry),
                    "--evidence-views-json",
                    str(views),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(summary.read_text())["final_frontier_eligible_rows"], 1)
            frontier_rows = [json.loads(line) for line in frontier.read_text().splitlines()]
            self.assertEqual([row["row_id"] for row in frontier_rows], [gold["row_id"]])
            self.assertEqual(json.loads(registry.read_text())["artifact_count"], 1)
            view_rows = json.loads(views.read_text())
            self.assertEqual([row["row_id"] for row in view_rows["final_frontier"]], [gold["row_id"]])
            self.assertEqual([row["row_id"] for row in view_rows["feasibility_failure"]], [failed["row_id"]])


if __name__ == "__main__":
    unittest.main()
