from __future__ import annotations

import sys
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold128_ap_repeat_audit_v1 as audit  # noqa: E402


class Stage35Gold128ApRepeatAuditV1Tests(unittest.TestCase):
    def test_four_stable_categories_are_qualified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plans = []
            states = []
            for index, category in enumerate(("base", "small_channel", "alignment_trap", "large_model")):
                job_id = f"ap-repeat-{index}"
                artifact = Path(tmp) / f"artifact-{index}.bin"
                full_report = Path(tmp) / f"full-{index}.json"
                sanity_report = Path(tmp) / f"sanity-{index}.json"
                artifact.write_bytes(f"artifact-{index}".encode())
                report_payload = {"ap30": 0.801, "ap50": 0.701, "ap70": 0.501}
                full_report.write_text(json.dumps(report_payload), encoding="utf-8")
                sanity_report.write_text(json.dumps(report_payload), encoding="utf-8")
                plan = {
                    "job_id": job_id,
                    "source_manifest_job_id": f"gold-{index}",
                    "repeat_category": category,
                    "runner_key": "pyramid_tvm_fp16_bridge",
                    "compiled_artifact_path": str(artifact),
                    "compiled_artifact_digest": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "sanity_command": ["python", "x.py", "--report-json", str(sanity_report)],
                    "full_command": ["python", "x.py", "--report-json", str(full_report)],
                    "baseline_ap30": 0.8,
                    "baseline_ap50": 0.7,
                    "baseline_ap70": 0.5,
                    "baseline_ap_report_sha256": "a" * 64,
                }
                plans.append(plan)
                for stage, report in (("sanity", sanity_report), ("full", full_report)):
                    states.append({
                        "job_id": job_id, "stage": stage, "status": "success",
                        "ap": report_payload,
                        "report_path": str(report),
                        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                        "plan_fingerprint": audit.plan_fingerprint(plan, stage),
                    })
                if index == 0:
                    states.append({
                        "job_id": job_id,
                        "stage": "sanity",
                        "status": "success",
                        "plan_fingerprint": None,
                    })

            result = audit.build_ap_repeat_audit(plans, states)

        self.assertEqual(result["terminal_rows"], 4)
        self.assertTrue(result["qualified"])
        self.assertEqual(set(result["category_summary"]), {row["repeat_category"] for row in plans})
        self.assertEqual(result["rows"][0]["source_manifest_job_id"], "gold-0")
        self.assertEqual(result["rows"][0]["baseline_ap_report_sha256"], "a" * 64)

    def test_missing_full_success_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "4 successful"):
            audit.build_ap_repeat_audit([], [])

    def test_report_path_must_match_plan_command(self) -> None:
        plan = {"full_command": ["python", "runner.py", "--report-json", "/tmp/expected.json"]}

        self.assertTrue(
            audit._report_is_bound_to_plan(plan, {"report_path": "/tmp/expected.json"}, "full")
        )
        self.assertFalse(
            audit._report_is_bound_to_plan(plan, {"report_path": "/tmp/substituted.json"}, "full")
        )
        out_json_plan = {
            "full_command": ["python", "runner.py", "--out-json", "/tmp/expected.json"]
        }
        self.assertTrue(
            audit._report_is_bound_to_plan(
                out_json_plan, {"report_path": "/tmp/expected.json"}, "full"
            )
        )


if __name__ == "__main__":
    unittest.main()
