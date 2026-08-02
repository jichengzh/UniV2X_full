import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from stage3_import_tvm_int8_repair_v3 import (  # noqa: E402
    evidence_paths,
    import_rows,
    validate_evidence,
)


class Stage3ImportTvmInt8RepairV3Tests(unittest.TestCase):
    def test_import_can_bind_gold32_subset_as_hashed_full_ap_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = root / "pyramid" / "1x2x3"
            build = job / "build" / "pyramid_1x2x3_scaleaware" / "route_b_int8_auto_decomp_result.json"
            ap = job / "ap_full" / "full_ap_eval_report.json"
            build.parent.mkdir(parents=True)
            ap.parent.mkdir(parents=True)
            build.write_text(json.dumps({
                "correctness_all_exact": True,
                "lat_p50_ms": 1.0,
                "energy_j": 0.2,
            }))
            ap.write_text(json.dumps({
                "status": "success",
                "processed_samples": 1789,
                "fallback_samples": 0,
                "failed_samples": 0,
                "ap30": 0.8,
                "ap50": 0.7,
                "ap70": 0.5,
                "ap_measured": True,
                "ap_row_allowed": True,
                "feasibility_blockers": [],
                "gates": {"full_1789": True},
            }))
            plan = [{
                "model": "pyramid",
                "width": [1, 2, 3],
                "q": "int8",
                "profile": "h800-tvm-probe-conditioned-v3",
                "performance_job_id": "perf-1",
                "manifest_job_id": "manifest-1",
            }]

            performance, ap_state = import_rows(plan, root, expected_rows=1)

            self.assertEqual(performance[0]["job_id"], "perf-1")
            self.assertEqual(ap_state[0]["schema_version"], "stage3_gold96_ap_seed_state_v3")
            self.assertEqual(ap_state[0]["ap"], {"ap30": 0.8, "ap50": 0.7, "ap70": 0.5})
            self.assertEqual(len(ap_state[0]["report_sha256"]), 64)

    def test_percentile_evidence_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ap = root / "codriving" / "1x2x3" / "ap_full_percentile_99_99" / "full_ap_eval_report.json"
            ap.parent.mkdir(parents=True)
            ap.write_text("{}")
            build, selected_ap = evidence_paths(root, "codriving", "1x2x3")
            self.assertIn("build_percentile_99_99", str(build))
            self.assertEqual(selected_ap, ap)

    def test_validation_requires_energy_and_complete_ap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build.json"
            ap = Path(directory) / "ap.json"
            build.write_text(json.dumps({"correctness_all_exact": True, "lat_p50_ms": 1.0, "energy_j": 0.2}))
            ap.write_text(json.dumps({
                "status": "success", "processed_samples": 1789, "fallback_samples": 0,
                "failed_samples": 0, "ap30": 0.8, "ap50": 0.7, "ap70": 0.5,
                "ap_measured": True, "smoke_gate_passed": True,
            }))
            validate_evidence(build, ap, model="pyramid")
            build.write_text(json.dumps({"correctness_all_exact": True, "lat_p50_ms": 1.0}))
            with self.assertRaises(ValueError):
                validate_evidence(build, ap, model="pyramid")

    def test_validation_accepts_route_b_nested_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build.json"
            ap = Path(directory) / "ap.json"
            build.write_text(json.dumps({
                "correctness_all_exact": True,
                "latency": {"latency_ms_p50": 2.3}, "energy": {"energy_J": 0.27},
            }))
            ap.write_text(json.dumps({
                "status": "success", "processed_samples": 1789,
                "ap30": 0.8, "ap50": 0.7, "ap70": 0.5,
                "gates": {"full_1789": True},
            }))
            validate_evidence(build, ap, model="codriving")

    def test_validation_rejects_report_missing_finalizer_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build.json"
            ap = Path(directory) / "ap.json"
            build.write_text(json.dumps({
                "correctness_all_exact": True, "lat_p50_ms": 1.0, "energy_j": 0.2,
            }))
            ap.write_text(json.dumps({
                "status": "success", "processed_samples": 1789,
                "ap30": 0.8, "ap50": 0.7, "ap70": 0.5,
            }))
            with self.assertRaisesRegex(ValueError, "Gold gate"):
                validate_evidence(build, ap, model="pyramid")


if __name__ == "__main__":
    unittest.main()
