import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stage3_run_tvm_int8_repair_batch_v3 import (  # noqa: E402
    provenance_sha256,
    resolve_calibration_source,
)


class ProvenanceSha256Test(unittest.TestCase):
    def test_calibration_source_prefers_manifest_bound_gold32_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "spatial_features_train16.npz"
            summary = root / "summary.json"
            calibration.write_bytes(b"npz")
            summary.write_text("{}")
            plan = {
                "source_contract": {
                    "calibration_npz": str(calibration),
                    "calibration_summary": str(summary),
                }
            }

            resolved = resolve_calibration_source(plan, model="pyramid", width="16x32x64", job_dir=root / "job")

            self.assertEqual(resolved, (calibration, summary, False))

    def test_digest_binds_filename_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "full_ap_eval_report.json"
            report.write_bytes(b'{"status":"success"}\n')
            expected = hashlib.sha256(
                report.name.encode("utf-8") + b"\0" + report.read_bytes()
            ).hexdigest()
            self.assertEqual(provenance_sha256(report), expected)

    def test_digest_differs_from_content_only_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_bytes(b"payload")
            self.assertNotEqual(
                provenance_sha256(report), hashlib.sha256(report.read_bytes()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
