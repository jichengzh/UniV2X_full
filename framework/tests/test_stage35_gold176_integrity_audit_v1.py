from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts import stage35_gold176_integrity_audit_v1 as audit


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/stage35_gold128_targeted_supplement_v1_20260714/final_gold144_v1"
TARGET = ROOT / "results/stage35_gold144_targeted_supplement_v2_20260714"


class Stage35Gold176IntegrityAuditTests(unittest.TestCase):
    def _inputs(self):
        return (
            json.loads((TARGET / "final_gold176_v1/gold176_final.json").read_text()),
            json.loads((TARGET / "final_gold176_v1/gold176_manifest.json").read_text()),
            json.loads((BASE / "gold144_final.json").read_text()),
            json.loads((TARGET / "final_targeted32_v1/targeted32_final.json").read_text()),
            [json.loads(line) for line in (TARGET / "plan/targeted_performance_jobs.jsonl").read_text().splitlines()],
            [json.loads(line) for line in (TARGET / "ap_plan/gold176_ap_plan.jsonl").read_text().splitlines()],
        )

    def test_qualifies_current_gold176_structure(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        result = audit.audit_gold176(
            rows,
            manifest,
            gold144_rows=gold144,
            targeted32_rows=targeted32,
            performance_job_rows=performance_jobs,
            ap_plan_rows=ap_plan,
            verify_evidence_files=False,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["rows"], 176)
        self.assertEqual(result["groups"], 44)
        self.assertEqual(result["targeted32_repaired_tvm_int8_rows"], 8)

    def test_rejects_duplicate_manifest_identity(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        changed = copy.deepcopy(rows)
        changed[-1]["manifest_job_id"] = changed[0]["manifest_job_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            audit.audit_gold176(
                changed,
                manifest,
                gold144_rows=gold144,
                targeted32_rows=targeted32,
                performance_job_rows=performance_jobs,
                ap_plan_rows=ap_plan,
                verify_evidence_files=False,
            )

    def test_rejects_changed_gold144_train_evidence(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        changed = copy.deepcopy(rows)
        source = next(row for row in changed if row["source_pool"] == "gold144" and row["split"] == "train")
        source["latency_ms"] = float(source["latency_ms"] or 0.0) + 1.0
        with self.assertRaisesRegex(ValueError, "Gold144 source evidence changed"):
            audit.audit_gold176(
                changed,
                manifest,
                gold144_rows=gold144,
                targeted32_rows=targeted32,
                performance_job_rows=performance_jobs,
                ap_plan_rows=ap_plan,
                verify_evidence_files=False,
            )

    def test_rejects_relabelled_plan_arm_identity(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        changed_rows = copy.deepcopy(rows)
        changed_targeted = copy.deepcopy(targeted32)
        changed_performance = copy.deepcopy(performance_jobs)
        changed_ap_plan = copy.deepcopy(ap_plan)
        evidence_fields = (
            "latency_ms", "energy_j", "ap30", "ap50", "ap70",
            "performance_result_json", "performance_result_sha256",
            "ap_report_path", "ap_report_sha256",
        )
        for pool in (changed_rows, changed_targeted):
            pair_rows = [
                row for row in pool
                if row["group_id"] == "codriving|48x32x128"
                and row["dispatch_key"] == "trt_engine"
            ]
            self.assertEqual(len(pair_rows), 2)
            values = [tuple(row[field] for field in evidence_fields) for row in pair_rows]
            for field, value in zip(evidence_fields, values[1]):
                pair_rows[0][field] = value
            for field, value in zip(evidence_fields, values[0]):
                pair_rows[1][field] = value
        pair = [
            row for row in changed_performance
            if row["group_id"] == "codriving|48x32x128"
            and row["dispatch_key"] == "trt_engine"
        ]
        self.assertEqual(len(pair), 2)
        pair[0]["manifest_job_id"], pair[1]["manifest_job_id"] = (
            pair[1]["manifest_job_id"], pair[0]["manifest_job_id"]
        )
        plan_pair = [
            row for row in changed_ap_plan
            if row["model"] == "codriving" and row["width"] == [48, 32, 128]
            and row["profile"] == "h800-trt-probe-conditioned-v3"
        ]
        plan_pair[0]["manifest_job_id"], plan_pair[1]["manifest_job_id"] = (
            plan_pair[1]["manifest_job_id"], plan_pair[0]["manifest_job_id"]
        )
        with self.assertRaisesRegex(ValueError, "arm identity mismatch"):
            audit.audit_gold176(
                changed_rows,
                manifest,
                gold144_rows=gold144,
                targeted32_rows=changed_targeted,
                performance_job_rows=changed_performance,
                ap_plan_rows=changed_ap_plan,
                verify_evidence_files=False,
            )

    def test_rejects_performance_command_evidence_path_swap(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        changed_performance = copy.deepcopy(performance_jobs)
        pair = [
            row for row in changed_performance
            if row["group_id"] == "codriving|48x32x128"
            and row["dispatch_key"] == "trt_engine"
        ]
        self.assertEqual(len(pair), 2)
        out_values = [audit._command_option(row["command"], "--out") for row in pair]
        self.assertNotEqual(out_values[0], out_values[1])
        for row, swapped_out in zip(pair, reversed(out_values)):
            option_index = row["command"].index("--out")
            row["command"][option_index + 1] = swapped_out
        with self.assertRaisesRegex(ValueError, "TRT result identity mismatch"):
            audit.audit_gold176(
                rows,
                manifest,
                gold144_rows=gold144,
                targeted32_rows=targeted32,
                performance_job_rows=changed_performance,
                ap_plan_rows=ap_plan,
                verify_evidence_files=False,
            )

    def test_rejects_tvm_sibling_performance_result(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        changed_rows = copy.deepcopy(rows)
        changed_targeted = copy.deepcopy(targeted32)
        manifest_id = "codriving|48x32x128|q=fp16|profile=h800-tvm-probe-conditioned-v3"
        for pool in (changed_rows, changed_targeted):
            row = next(item for item in pool if item["manifest_job_id"] == manifest_id)
            row["performance_result_json"] = str(
                Path(row["performance_result_json"]).with_name("stale_or_wrong_result.json")
            )
        with self.assertRaisesRegex(ValueError, "TVM result identity mismatch"):
            audit.audit_gold176(
                changed_rows,
                manifest,
                gold144_rows=gold144,
                targeted32_rows=changed_targeted,
                performance_job_rows=performance_jobs,
                ap_plan_rows=ap_plan,
                verify_evidence_files=False,
            )

    def test_rejects_repair_evidence_under_wrong_root(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        changed_rows = copy.deepcopy(rows)
        changed_targeted = copy.deepcopy(targeted32)
        manifest_id = "codriving|48x32x128|q=int8|profile=h800-tvm-probe-conditioned-v3"
        for pool in (changed_rows, changed_targeted):
            row = next(item for item in pool if item["manifest_job_id"] == manifest_id)
            suffix = Path(*Path(row["performance_result_json"]).parts[-7:])
            row["performance_result_json"] = str(Path("/tmp/evil") / suffix)
        with self.assertRaisesRegex(ValueError, "repair performance identity mismatch"):
            audit.audit_gold176(
                changed_rows,
                manifest,
                gold144_rows=gold144,
                targeted32_rows=changed_targeted,
                performance_job_rows=performance_jobs,
                ap_plan_rows=ap_plan,
                verify_evidence_files=False,
            )

    def test_rejects_ap_report_under_wrong_root(self) -> None:
        rows, manifest, gold144, targeted32, performance_jobs, ap_plan = self._inputs()
        changed_rows = copy.deepcopy(rows)
        changed_targeted = copy.deepcopy(targeted32)
        changed_ap_plan = copy.deepcopy(ap_plan)
        manifest_id = "codriving|48x32x128|q=fp16|profile=h800-trt-probe-conditioned-v3"
        wrong_path = Path(
            "/tmp/evil/ap/codriving/48x32x128/fp16/"
            "h800-trt-probe-conditioned-v3/full_1789/full_ap_eval_report.json"
        )
        for pool in (changed_rows, changed_targeted):
            row = next(item for item in pool if item["manifest_job_id"] == manifest_id)
            row["ap_report_path"] = str(wrong_path)
        plan = next(item for item in changed_ap_plan if item["manifest_job_id"] == manifest_id)
        option_index = plan["full_command"].index("--out-json")
        plan["full_command"][option_index + 1] = str(wrong_path)
        with self.assertRaisesRegex(ValueError, "AP report identity mismatch"):
            audit.audit_gold176(
                changed_rows,
                manifest,
                gold144_rows=gold144,
                targeted32_rows=changed_targeted,
                performance_job_rows=performance_jobs,
                ap_plan_rows=changed_ap_plan,
                verify_evidence_files=False,
            )


if __name__ == "__main__":
    unittest.main()
