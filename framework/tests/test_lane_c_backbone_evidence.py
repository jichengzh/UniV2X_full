from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools.orin_deploy.lane_c_backbone_evidence import (
    compute_output_metrics,
    main,
    parse_args,
    prepare_heldout_batches,
)


class LaneCBackboneEvidenceTest(unittest.TestCase):
    def test_prepare_heldout_batches_excludes_calibration_scenes(self) -> None:
        scene_lens = [1, 2, 1, 2]
        features = np.arange(6 * 2, dtype=np.float32).reshape(6, 1, 1, 2)
        batches, audit = prepare_heldout_batches(
            features, scene_lens, calibration_scene_count=2
        )
        self.assertEqual(list(batches.shape), [1, 2, 1, 1, 2])
        np.testing.assert_array_equal(batches.reshape(2, 2), features[3:5].reshape(2, 2))
        self.assertEqual(audit["excluded_agent_instances"], 3)
        self.assertEqual(audit["heldout_agent_instances_used"], 2)
        self.assertEqual(audit["heldout_agent_instances_dropped"], 1)

    def test_compute_output_metrics_reports_exact_match(self) -> None:
        reference = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        metrics = compute_output_metrics(reference, reference.copy())
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertEqual(metrics["nrmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["candidate_below_reference_min_fraction"], 0.0)
        self.assertEqual(metrics["candidate_above_reference_max_fraction"], 0.0)
        self.assertEqual(
            metrics["saturation_clipping_semantics"],
            "output_distribution_proxy_not_internal_quantizer_counter",
        )

    def test_compare_parser_accepts_one_fp32_reference_argument(self) -> None:
        args = parse_args(
            [
                "compare",
                "--artifact-root",
                "/tmp/lane-c-evidence",
                "--reference-npz",
                "/tmp/lane-c-evidence/reference.npz",
                "--candidate-npz",
                "/tmp/lane-c-evidence/candidate.npz",
                "--precision",
                "fp32",
                "--report-json",
                "/tmp/lane-c-evidence/report.json",
            ]
        )
        self.assertEqual(args.precision, "fp32")
        self.assertEqual(
            args.reference_npz, Path("/tmp/lane-c-evidence/reference.npz")
        )

    def test_fp32_compare_preserves_dtype_and_reports_each_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.npz"
            candidate_path = root / "candidate.npz"
            report_path = root / "report.json"
            reference = {
                f"level_{level}": np.array(
                    [-1.0, 0.0, float(level), 2.0], dtype=np.float32
                )
                for level in (16, 32, 64)
            }
            candidate = {
                name: value + np.float32(0.01)
                for name, value in reference.items()
            }
            np.savez(reference_path, **reference)
            np.savez(candidate_path, **candidate)

            with redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "compare",
                        "--artifact-root",
                        str(root),
                        "--reference-npz",
                        str(reference_path),
                        "--candidate-npz",
                        str(candidate_path),
                        "--precision",
                        "fp32",
                        "--report-json",
                        str(report_path),
                    ]
                )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(report["dtype_contract"]["required_dtype"], "float32")
            self.assertTrue(report["dtype_contract"]["validated"])
            self.assertEqual(set(report["per_output"]), set(reference))
            for metrics in report["per_output"].values():
                self.assertEqual(metrics["reference_dtype"], "float32")
                self.assertEqual(metrics["candidate_dtype"], "float32")
                self.assertIn("cosine", metrics)
                self.assertIn("nrmse", metrics)
                self.assertIn("mae", metrics)
                self.assertIn("candidate_at_observed_min_fraction", metrics)
                self.assertIn("candidate_at_observed_max_fraction", metrics)

    def test_fp32_compare_rejects_non_float32_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.npz"
            candidate_path = root / "candidate.npz"
            report_path = root / "report.json"
            np.savez(reference_path, level_16=np.ones((2, 2), dtype=np.float32))
            np.savez(candidate_path, level_16=np.ones((2, 2), dtype=np.float16))

            with self.assertRaisesRegex(
                ValueError, "fp32 comparison requires float32.*candidate.*level_16"
            ):
                with redirect_stdout(io.StringIO()):
                    main(
                        [
                            "compare",
                            "--artifact-root",
                            str(root),
                            "--reference-npz",
                            str(reference_path),
                            "--candidate-npz",
                            str(candidate_path),
                            "--precision",
                            "fp32",
                            "--report-json",
                            str(report_path),
                        ]
                    )

    def test_existing_fp16_and_int8_compare_paths_remain_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.npz"
            candidate_path = root / "candidate.npz"
            np.savez(reference_path, level_16=np.ones((2, 2), dtype=np.float32))
            np.savez(candidate_path, level_16=np.ones((2, 2), dtype=np.float16))

            for precision in ("fp16", "int8"):
                with self.subTest(precision=precision):
                    report_path = root / f"{precision}.json"
                    with redirect_stdout(io.StringIO()):
                        result = main(
                            [
                                "compare",
                                "--artifact-root",
                                str(root),
                                "--reference-npz",
                                str(reference_path),
                                "--candidate-npz",
                                str(candidate_path),
                                "--precision",
                                precision,
                                "--report-json",
                                str(report_path),
                            ]
                        )
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertEqual(result, 0)
                    self.assertEqual(report["precision"], precision)
                    self.assertEqual(
                        report["per_output"]["level_16"]["candidate_dtype"], "float16"
                    )


if __name__ == "__main__":
    unittest.main()
