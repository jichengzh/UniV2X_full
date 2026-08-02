from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold128_ap_label_repair_v2 as repair  # noqa: E402


class Stage35Gold128ApLabelRepairV2Tests(unittest.TestCase):
    def test_repairs_only_declared_rows_with_matching_checkpoint_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(json.dumps({
                "status": "success", "processed_samples": 1789,
                "checkpoint_epoch": 23, "failed_samples": 0, "fallback_samples": 0,
                "ap30": 0.8, "ap50": 0.7, "ap70": 0.6,
            }), encoding="utf-8")
            report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            rows = [{
                "manifest_job_id": "pyramid|64x128x256|q=fp16|profile=h800-trt-probe-conditioned-v3",
                "ap30": 0.0, "ap50": 0.0, "ap70": 0.0,
                "ap_report_path": "old.json", "ap_report_sha256": "a" * 64,
            }, {"manifest_job_id": "untouched", "ap70": 0.2}]
            manifest = {"jobs": [{
                "job_id": rows[0]["manifest_job_id"],
                "source_contract": {"checkpoint_path": "/ckpt/net_epoch_bestval_at23.pth"},
            }]}

            result = repair.repair_ap_labels(
                rows, manifest, {rows[0]["manifest_job_id"]: report_path}
            )

        repaired = result["rows"][0]
        self.assertEqual(repaired["ap70"], 0.6)
        self.assertEqual(repaired["ap_report_sha256"], report_digest)
        self.assertEqual(result["rows"][1], rows[1])
        self.assertEqual(result["audit"]["repaired_rows"], 1)

    def test_rejects_report_from_wrong_checkpoint_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            report_path.write_text(json.dumps({
                "status": "success", "processed_samples": 1789,
                "resume_epoch": 1, "ap30": 0.8, "ap50": 0.7, "ap70": 0.6,
            }), encoding="utf-8")
            row = {"manifest_job_id": "job"}
            manifest = {"jobs": [{
                "job_id": "job", "source_contract": {"checkpoint_path": "/ckpt/net_epoch23.pth"},
            }]}

            with self.assertRaisesRegex(ValueError, "checkpoint epoch mismatch"):
                repair.repair_ap_labels([row], manifest, {"job": report_path})


if __name__ == "__main__":
    unittest.main()
