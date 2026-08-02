from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import stage35_finalize_gold176_supplement_v1 as finalizer


def _manifest() -> dict:
    jobs = []
    for group_index in range(8):
        model = "pyramid" if group_index < 4 else "codriving"
        width = [24 + group_index, 32, 64]
        width_key = "x".join(map(str, width))
        group_id = f"{model}|{width_key}"
        for dispatch, profile in (
            ("tvm_auto", "h800-tvm-probe-conditioned-v3"),
            ("trt_engine", "h800-trt-probe-conditioned-v3"),
        ):
            for q_mode in ("fp16", "int8"):
                jobs.append(
                    {
                        "job_id": f"{group_id}|q={q_mode}|profile={profile}",
                        "group_id": group_id,
                        "model": model,
                        "width": width,
                        "q_mode": q_mode,
                        "capability_profile_id": profile,
                        "dispatch_key": dispatch,
                        "split": "train",
                    }
                )
    return {"schema_version": finalizer.MANIFEST_SCHEMA, "jobs": jobs}


class Stage35FinalizeGold176SupplementTests(unittest.TestCase):
    def test_enforces_32_rows_and_writes_targeted32_outputs(self) -> None:
        manifest = _manifest()
        plan = [
            {"manifest_job_id": row["job_id"], "performance_job_id": f"perf-{index}"}
            for index, row in enumerate(manifest["jobs"])
        ]
        result = finalizer.finalize_targeted32(
            manifest,
            ap_plan_rows=plan,
            performance_state_rows=[],
            ap_state_rows=[],
        )
        self.assertEqual(result["summary"]["total"], 32)
        self.assertEqual(len(result["group_audit"]), 8)
        self.assertTrue(all(row["schema_version"] == finalizer.OUTPUT_SCHEMA for row in result["rows"]))
        with tempfile.TemporaryDirectory() as directory:
            finalizer.write_targeted32_outputs(result, directory)
            output = Path(directory)
            self.assertTrue((output / "targeted32_final.json").is_file())
            audit = json.loads((output / "targeted32_audit.json").read_text())
        self.assertEqual(audit["schema_version"], finalizer.OUTPUT_SCHEMA)

    def test_rejects_incomplete_manifest(self) -> None:
        manifest = _manifest()
        manifest["jobs"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 32 jobs"):
            finalizer.finalize_targeted32(
                manifest,
                ap_plan_rows=[],
                performance_state_rows=[],
                ap_state_rows=[],
            )

    def test_rejects_group_without_all_four_arms(self) -> None:
        manifest = _manifest()
        manifest["jobs"][1]["q_mode"] = "fp16"
        manifest["jobs"][1]["job_id"] += "-duplicate-arm"
        plan = [
            {"manifest_job_id": row["job_id"], "performance_job_id": f"perf-{index}"}
            for index, row in enumerate(manifest["jobs"])
        ]
        with self.assertRaisesRegex(ValueError, "four-arm"):
            finalizer.finalize_targeted32(
                manifest,
                ap_plan_rows=plan,
                performance_state_rows=[],
                ap_state_rows=[],
            )

    def test_repair_overlay_is_selected_independent_of_input_order(self) -> None:
        plan = [{
            "manifest_job_id": "m",
            "performance_job_id": "p",
            "q": "int8",
            "profile": "h800-tvm-probe-conditioned-v3",
        }]
        base_performance = {"job_id": "p", "status": "success", "result_json": "/base.json"}
        repair_performance = {
            "job_id": "p", "status": "success", "result_json": "/repair.json",
            "source": "stage3_tvm_int8_repair_v3",
        }
        base_ap = {"job_id": "m", "stage": "full", "status": "success", "report_path": "/base-ap.json"}
        repair_ap = {
            "job_id": "m", "stage": "full", "status": "success", "report_path": "/repair-ap.json",
            "source": "stage3_tvm_int8_repair_v3",
        }
        performance, ap = finalizer.select_repair_overlays(
            plan,
            [repair_performance, base_performance],
            [repair_ap, base_ap],
        )
        self.assertEqual(performance, [repair_performance])
        self.assertEqual(ap, [repair_ap])


if __name__ == "__main__":
    unittest.main()
