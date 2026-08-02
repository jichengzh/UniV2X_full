from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage6_aggregate_pyramid_six_arm_v1 import (
    _load_terminal_summaries,
    aggregate_six_arm_results,
    write_aggregate_outputs,
)


def _manifest() -> dict:
    return {
        "schema_version": "stage6_pyramid_arm_manifest_v2",
        "experiment_id": "stage6-test",
        "independent_backends": ["tvm", "trt"],
        "arms": [
            {
                "arm_id": "original_default",
                "backend": "pytorch_eager",
                "width": [64, 128, 256],
                "q_modes": ["fp32"],
            },
            {"arm_id": "compression_only"},
            {"arm_id": "schedule_only"},
            {"arm_id": "compress_then_tune"},
            {"arm_id": "tune_then_compress"},
            {"arm_id": "joint_shcosearch"},
        ],
    }


def _baseline() -> dict:
    return {
        "schema_version": "stage6_native_fp32_baseline_v1",
        "backend": "pytorch_eager",
        "width": [64, 128, 256],
        "precision": "fp32",
        "latency_p50_ms": 3.2,
        "energy_j": 1.1,
    }


class Stage6AggregatePyramidSixArmV1Tests(unittest.TestCase):
    def test_preserves_all_slots_and_never_backfills_missing_measurements(self) -> None:
        complete = {
            "arm_id": "joint_shcosearch",
            "backend": "trt",
            "status": "complete",
            "representative_config": [16, 32, 64, "int8"],
            "objectives": {"ap70": 0.62, "latency_ms": 1.7, "energy_j": 0.4},
            "hypervolume": 1.14,
            "frontier_count": 8,
            "outer_measured_genomes": 16,
            "backend_tuning_trials": 64,
            "gpu_hours": 2.5,
            "wallclock_s": 5400,
            "failure_count": 1,
            "evidence": ["result.json", "rows.csv"],
        }
        aggregate = aggregate_six_arm_results(
            _manifest(),
            _baseline(),
            [(complete, "/tmp/joint-trt.json")],
            manifest_evidence="/tmp/manifest.json",
            baseline_evidence="/tmp/baseline.json",
        )

        self.assertEqual(len(aggregate["rows"]), 11)
        by_key = {(row["arm"], row["backend"]): row for row in aggregate["rows"]}

        original = by_key[("original_default", "pytorch_eager")]
        self.assertEqual(original["latency"], 3.2)
        self.assertEqual(original["energy"], 1.1)
        self.assertIsNone(original["AP70"])
        self.assertEqual(original["status"], "provisional")

        joint = by_key[("joint_shcosearch", "trt")]
        self.assertEqual(joint["status"], "complete")
        self.assertEqual(joint["config"], [16, 32, 64, "int8"])
        self.assertEqual(joint["AP70"], 0.62)
        self.assertEqual(joint["latency"], 1.7)
        self.assertEqual(joint["energy"], 0.4)
        self.assertEqual(joint["HV"], 1.14)
        self.assertEqual(joint["frontier_count"], 8)
        self.assertEqual(joint["outer_genomes"], 16)
        self.assertEqual(joint["tuning_trials"], 64)
        self.assertEqual(joint["gpu_hours"], 2.5)
        self.assertEqual(joint["wallclock"], 5400)
        self.assertEqual(joint["failures"], 1)
        self.assertIn("/tmp/joint-trt.json", joint["evidence"])

        missing = by_key[("compression_only", "tvm")]
        self.assertEqual(missing["status"], "provisional")
        for field in ("config", "AP70", "latency", "energy", "HV"):
            self.assertIsNone(missing[field])

        self.assertFalse(aggregate["readiness"]["ready"])
        self.assertEqual(aggregate["readiness"]["expected_rows"], 11)
        self.assertEqual(aggregate["readiness"]["complete_rows"], 1)

    def test_ignores_surrogate_and_legacy_proxy_values(self) -> None:
        incomplete = {
            "arm_id": "compression_only",
            "backend": "tvm",
            "terminal_status": "running",
            "surrogate": {"ap70": 0.99, "latency_ms": 0.1, "energy_j": 0.01},
            "legacy_proxy": {"hypervolume": 9.9},
            "predicted_objectives": {"ap70": 0.98, "latency_ms": 0.2},
        }

        aggregate = aggregate_six_arm_results(
            _manifest(),
            _baseline(),
            [(incomplete, "/tmp/incomplete.json")],
        )
        row = next(
            row
            for row in aggregate["rows"]
            if row["arm"] == "compression_only" and row["backend"] == "tvm"
        )

        self.assertEqual(row["status"], "provisional")
        self.assertIsNone(row["AP70"])
        self.assertIsNone(row["latency"])
        self.assertIsNone(row["energy"])
        self.assertIsNone(row["HV"])

    def test_writes_csv_markdown_and_json_with_blank_unmeasured_cells(self) -> None:
        aggregate = aggregate_six_arm_results(_manifest(), _baseline(), [])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_aggregate_outputs(aggregate, Path(tmp))

            self.assertEqual(set(paths), {"csv", "markdown", "json"})
            with paths["csv"].open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            original = csv_rows[0]
            self.assertEqual(original["AP70"], "")
            self.assertEqual(original["latency"], "3.2")

            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("| arm | backend | status |", markdown)
            self.assertIn("provisional", markdown)

            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertFalse(payload["readiness"]["ready"])
            self.assertEqual(payload["readiness"]["provisional_rows"], 11)

    def test_rejects_duplicate_terminal_summary_for_same_slot(self) -> None:
        summary = {
            "arm": "schedule_only",
            "backend": "tvm",
            "status": "running",
        }
        with self.assertRaisesRegex(ValueError, "duplicate terminal summary"):
            aggregate_six_arm_results(
                _manifest(),
                _baseline(),
                [(summary, "one.json"), (summary, "two.json")],
            )

    def test_accepts_empty_incomplete_summary_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "still-running.json"
            path.write_text(json.dumps({"rows": []}), encoding="utf-8")

            summaries = _load_terminal_summaries([path])

        self.assertEqual(summaries, [])


if __name__ == "__main__":
    unittest.main()
