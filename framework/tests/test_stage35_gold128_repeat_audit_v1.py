from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold128_repeat_audit_v1 as audit  # noqa: E402


class Stage35Gold128RepeatAuditV1Tests(unittest.TestCase):
    def test_two_point_cv_is_zero_for_identical_measurements(self) -> None:
        self.assertEqual(audit.coefficient_of_variation([2.0, 2.0]), 0.0)

    def test_stable_32_row_repeat_is_qualified_for_performance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps({"latency_ms": 1.02, "energy_j": 2.04}), encoding="utf-8")
            jobs = []
            states = []
            gold = []
            for index in range(32):
                job_id = f"repeat-{index}"
                manifest_id = f"manifest-{index}"
                jobs.append({
                    "job_id": job_id, "manifest_job_id": manifest_id,
                    "repeat_category": ("base", "small_channel", "alignment_trap", "large_model")[index % 4],
                    "group_id": f"group-{index // 4}", "runner_key": "tvm_fp16",
                    "command": ["runner", "--out", str(result_path)],
                })
                states.append({
                    "job_id": job_id, "status": "success", "result_json": str(result_path),
                    "result_sha256": audit.sha256_file(result_path),
                })
                gold.append({
                    "manifest_job_id": manifest_id, "latency_ms": 1.0, "energy_j": 2.0,
                    "terminal_status": "measured_success_gold",
                })

            result = audit.build_repeat_audit(jobs, states, gold)

        self.assertEqual(result["terminal_rows"], 32)
        self.assertTrue(result["performance_repeat_qualified"])
        self.assertFalse(result["qualified"])
        self.assertEqual(result["ap_repeat_status"], "not_measured")
        self.assertEqual(len(result["rows"]), 32)

    def test_missing_repeat_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 successful"):
            audit.build_repeat_audit([], [], [])

    def test_compact_state_reads_metrics_from_bound_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.json"
            result_path.write_text(json.dumps({
                "latency": {"latency_ms_p50": 1.02},
                "energy": {"joules_per_inference": 2.04},
            }), encoding="utf-8")
            jobs = []
            states = []
            gold = []
            for index in range(32):
                job_id = f"compact-{index}"
                manifest_id = f"compact-manifest-{index}"
                jobs.append({
                    "job_id": job_id, "manifest_job_id": manifest_id,
                    "repeat_category": "base", "group_id": f"group-{index // 4}",
                    "runner_key": "trt_fp16", "command": ["runner", "--out", str(result_path)],
                })
                states.append({
                    "job_id": job_id, "status": "success", "result_json": str(result_path),
                    "result_sha256": audit.sha256_file(result_path),
                })
                gold.append({
                    "manifest_job_id": manifest_id, "latency_ms": 1.0, "energy_j": 2.0,
                    "terminal_status": "measured_success_gold",
                })

            result = audit.build_repeat_audit(jobs, states, gold)

        self.assertEqual(result["terminal_rows"], 32)
        self.assertAlmostEqual(result["rows"][0]["repeat_latency_ms"], 1.02)
        self.assertAlmostEqual(result["rows"][0]["repeat_energy_j"], 2.04)

    def test_later_corrected_success_overrides_earlier_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale_path = Path(tmp) / "stale.json"
            corrected_path = Path(tmp) / "corrected.json"
            stale_path.write_text(json.dumps({"latency_ms": 9.0, "energy_j": 9.0}), encoding="utf-8")
            corrected_path.write_text(json.dumps({"latency_ms": 1.0, "energy_j": 2.0}), encoding="utf-8")
            jobs = []
            states = []
            gold = []
            for index in range(32):
                job_id = f"repeat-{index}"
                manifest_id = f"manifest-{index}"
                jobs.append({
                    "job_id": job_id, "manifest_job_id": manifest_id,
                    "repeat_category": "base", "group_id": f"group-{index // 4}",
                    "runner_key": "tvm_int8" if index == 0 else "tvm_fp16",
                    "command": [
                        "runner", "--out", str(corrected_path if index == 0 else stale_path)
                    ],
                })
                states.append({
                    "job_id": job_id, "status": "success", "result_json": str(stale_path),
                    "result_sha256": audit.sha256_file(stale_path),
                })
                gold.append({
                    "manifest_job_id": manifest_id, "latency_ms": 1.0, "energy_j": 2.0,
                    "terminal_status": "measured_success_gold",
                })
            states.append({
                "job_id": "repeat-0", "status": "success", "result_json": str(corrected_path),
                "result_sha256": audit.sha256_file(corrected_path),
            })

            result = audit.build_repeat_audit(jobs, states, gold)

        corrected = next(row for row in result["rows"] if row["job_id"] == "repeat-0")
        self.assertEqual(corrected["repeat_result_json"], str(corrected_path))
        self.assertEqual(corrected["latency_cv_pct"], 0.0)

    def test_out_dir_binding_rejects_another_result_in_the_same_tree(self) -> None:
        job = {
            "command": [
                "python3", "scripts/stage2_route_b_int8_auto_decomp.py",
                "--label", "expected_label", "--out-dir", "/tmp/repeat-output",
            ]
        }

        self.assertFalse(
            audit._result_is_bound_to_job(
                job, "/tmp/repeat-output/another_label/route_b_int8_auto_decomp_result.json"
            )
        )
        self.assertTrue(
            audit._result_is_bound_to_job(
                job, "/tmp/repeat-output/expected_label/route_b_int8_auto_decomp_result.json"
            )
        )


if __name__ == "__main__":
    unittest.main()
