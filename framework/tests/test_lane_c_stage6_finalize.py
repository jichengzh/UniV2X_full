from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.orin_deploy.lane_c_stage6_finalize import (
    _load_ap_terminal,
    apply_manifest_overrides,
    artifact_dir_for_spec,
    collect_result_row,
    latency_path_for_arm,
    validate_manifest_rows,
    write_outputs,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_engine_evidence(arm_dir: Path) -> str:
    engine = arm_dir / "backbone.engine"
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(b"test engine")
    engine_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    write_json(arm_dir / "build_report.json", {"engine_sha256": engine_sha})
    return engine_sha


class LaneCStage6FinalizeTest(unittest.TestCase):
    def test_fp16_joint_override_is_immutable_and_keeps_row_identity(self) -> None:
        manifest = {
            "rows": [
                {
                    "id": "pyramid:joint_shcosearch",
                    "model": "pyramid",
                    "arm": "joint_shcosearch",
                    "precision": "int8",
                }
            ]
        }
        overridden = apply_manifest_overrides(
            manifest,
            {
                "rows": [
                    {
                        "id": "pyramid:joint_shcosearch",
                        "precision": "fp16",
                        "artifact_arm": "joint_shcosearch_fp16",
                    }
                ]
            },
        )

        self.assertEqual(manifest["rows"][0]["precision"], "int8")
        self.assertEqual(overridden["rows"][0]["precision"], "fp16")
        self.assertEqual(
            overridden["rows"][0]["artifact_arm"],
            "joint_shcosearch_fp16",
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            apply_manifest_overrides(
                manifest,
                {
                    "rows": [
                        {
                            "id": "pyramid:joint_shcosearch",
                            "arm": "other",
                        }
                    ]
                },
            )

    def test_artifact_arm_and_sudo_latency_report_are_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = {
                "model": "codriving",
                "arm": "joint_shcosearch",
                "artifact_arm": "joint_shcosearch_fp16",
            }
            arm_dir = artifact_dir_for_spec(spec, root)
            arm_dir.mkdir(parents=True)
            default = arm_dir / "primary_latency_power.json"
            privileged = arm_dir / "primary_latency_power_sudo.json"
            default.write_text("{}", encoding="utf-8")
            privileged.write_text("{}", encoding="utf-8")

            self.assertEqual(
                arm_dir,
                root / "codriving" / "joint_shcosearch_fp16",
            )
            self.assertEqual(latency_path_for_arm(arm_dir), privileged)

    def test_manifest_requires_exact_unique_two_by_five_ids(self) -> None:
        duplicate_rows = [
            {
                "id": "pyramid:original_default",
                "model": "pyramid",
                "arm": "original_default",
            }
            for _ in range(10)
        ]
        with self.assertRaisesRegex(ValueError, "exact ordered 2x5"):
            validate_manifest_rows({"rows": duplicate_rows})

    def test_collects_native_and_tensorrt_rows_without_inventing_energy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "rows": [
                    {
                        "id": "pyramid:original_default",
                        "model": "pyramid",
                        "arm": "original_default",
                        "widths": [64, 128, 256],
                        "precision": "fp32",
                        "runtime": "native_pytorch",
                        "source_sha256": {"checkpoint": "a" * 64},
                        "h800_source": {
                            "ap70": 0.63,
                            "latency_ms": 3.2,
                            "energy_j": 1.1,
                        },
                    },
                    {
                        "id": "pyramid:schedule_only",
                        "model": "pyramid",
                        "arm": "schedule_only",
                        "widths": [64, 128, 256],
                        "precision": "fp32",
                        "runtime": "tensorrt",
                        "source_sha256": {"checkpoint": "a" * 64},
                        "h800_source": {
                            "ap70": 0.63,
                            "latency_ms": 1.1,
                            "energy_j": 0.4,
                        },
                    },
                ]
            }
            for arm in ("original_default", "schedule_only"):
                engine_sha = (
                    write_engine_evidence(root / "pyramid" / arm)
                    if arm == "schedule_only"
                    else None
                )
                write_json(
                    root / "pyramid" / arm / "primary_latency_power.json",
                    {
                        "sample_count": 1500,
                        "median_ms": 100.0,
                        "p90_ms": 101.0,
                        "p99_ms": 102.0,
                        "mean_ms": 100.5,
                        "scope": (
                            "pyramid_multiscale_backbone_compute_no_data_transfer"
                            if arm == "original_default"
                            else
                            "pyramid_get_multiscale_feature_64x128x256_"
                            "engine_compute_no_data_transfer"
                        ),
                        "engine_sha256": engine_sha,
                        "protocol": {
                            "warmup": 20,
                            "iters": 300,
                            "repeat": 5,
                            "timing": "CUDA_event",
                            "data_transfer_inside_timed_region": False,
                        },
                        "power_measurement": {
                            "measurement_status": "unavailable",
                            "blocker": "missing_tegrastats_power_rails",
                        },
                    },
                )
                write_json(
                    root / "pyramid" / arm / "ap_full" / "report.json",
                    {
                        "status": "success",
                        "num_samples": 1789,
                        "processed_samples": 1789,
                        "failed_samples": 0,
                        "fallback_samples": 0,
                        "engine_ap_claim": True,
                        "ap50": 0.79,
                        "ap70": 0.63,
                    },
                )
            rows = [
                collect_result_row(spec, root)
                for spec in manifest["rows"]
            ]

            self.assertEqual(len(rows), 2)
            self.assertIsNone(rows[0]["orin"]["energy_j"])
            self.assertEqual(
                rows[0]["orin"]["energy_status"],
                "missing_tegrastats_power_rails",
            )
            self.assertIsNone(rows[0]["evidence_sha256"]["engine"])
            self.assertEqual(
                rows[1]["evidence_sha256"]["engine"],
                engine_sha,
            )

    def test_rejects_incomplete_full_1789_ap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "rows": [
                    {
                        "id": "codriving:joint_shcosearch",
                        "model": "codriving",
                        "arm": "joint_shcosearch",
                        "widths": [16, 32, 64],
                        "precision": "int8",
                        "runtime": "tensorrt",
                        "source_sha256": {"checkpoint": "a" * 64},
                        "h800_source": {
                            "ap70": 0.35,
                            "latency_ms": 0.3,
                            "energy_j": 0.1,
                        },
                    }
                ]
            }
            write_json(
                root
                / "codriving"
                / "joint_shcosearch"
                / "primary_latency_power.json",
                {
                    "sample_count": 1500,
                    "median_ms": 4.7,
                    "p90_ms": 4.8,
                    "p99_ms": 4.9,
                    "mean_ms": 4.7,
                    "scope": (
                        "codriving_backbone_resnet_16x32x64_"
                        "engine_compute_no_data_transfer"
                    ),
                    "protocol": {
                        "warmup": 20,
                        "iters": 300,
                        "repeat": 5,
                        "timing": "CUDA_event",
                        "data_transfer_inside_timed_region": False,
                    },
                    "power_measurement": {
                        "measurement_status": "unavailable",
                        "blocker": "missing_tegrastats_power_rails",
                    },
                },
            )
            write_json(
                root
                / "codriving"
                / "joint_shcosearch"
                / "ap_full"
                / "report.json",
                {
                    "status": "success",
                    "processed_samples": 16,
                    "failed_samples": 0,
                    "fallback_samples": 0,
                    "engine_ap_claim": True,
                    "ap50": 0.5,
                    "ap70": 0.3,
                },
            )
            write_json(
                root / "codriving" / "joint_shcosearch" / "build_report.json",
                {"engine_sha256": "b" * 64},
            )

            with self.assertRaisesRegex(ValueError, "full_1789"):
                collect_result_row(manifest["rows"][0], root)

    def test_accepts_zero_prediction_sanity_as_explicit_ap_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_sha = write_engine_evidence(
                root / "pyramid" / "joint_shcosearch"
            )
            manifest = {
                "rows": [
                    {
                        "id": "pyramid:joint_shcosearch",
                        "model": "pyramid",
                        "arm": "joint_shcosearch",
                        "widths": [16, 32, 64],
                        "precision": "int8",
                        "runtime": "tensorrt",
                        "source_sha256": {"checkpoint": "a" * 64},
                        "h800_source": {
                            "ap70": 0.61,
                            "latency_ms": 0.45,
                            "energy_j": 0.11,
                        },
                    }
                ]
            }
            write_json(
                root
                / "pyramid"
                / "joint_shcosearch"
                / "primary_latency_power.json",
                {
                    "sample_count": 1500,
                    "median_ms": 9.1,
                    "p90_ms": 9.2,
                    "p99_ms": 9.3,
                    "mean_ms": 9.1,
                    "scope": (
                        "pyramid_get_multiscale_feature_16x32x64_"
                        "engine_compute_no_data_transfer"
                    ),
                    "engine_sha256": engine_sha,
                    "protocol": {
                        "warmup": 20,
                        "iters": 300,
                        "repeat": 5,
                        "timing": "CUDA_event",
                        "data_transfer_inside_timed_region": False,
                    },
                    "power_measurement": {
                        "measurement_status": "unavailable",
                        "blocker": "missing_tegrastats_power_rails",
                    },
                },
            )
            write_json(
                root
                / "pyramid"
                / "joint_shcosearch"
                / "ap_sanity"
                / "report.json",
                {
                    "status": "success",
                    "processed_samples": 16,
                    "failed_samples": 0,
                    "fallback_samples": 0,
                    "pred_nonempty_count": 0,
                    "ap50": 0.0,
                    "ap70": 0.0,
                    "output_error_summary": {
                        "all_finite": True,
                        "max_abs_err": 20.7,
                    },
                },
            )
            rows = [collect_result_row(manifest["rows"][0], root)]

            self.assertIsNone(rows[0]["orin"]["ap50"])
            self.assertIsNone(rows[0]["orin"]["ap70"])
            self.assertEqual(
                rows[0]["orin"]["ap_status"],
                "sanity_failed_zero_predictions",
            )
            self.assertEqual(rows[0]["orin"]["sanity_ap70"], 0.0)

    def test_rejects_zero_prediction_sanity_for_any_other_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arm_dir = Path(temporary)
            write_json(
                arm_dir / "ap_sanity" / "report.json",
                {
                    "status": "success",
                    "processed_samples": 16,
                    "failed_samples": 0,
                    "fallback_samples": 0,
                    "pred_nonempty_count": 0,
                    "ap50": 0.0,
                    "ap70": 0.0,
                    "output_error_summary": {"all_finite": True},
                },
            )
            with self.assertRaisesRegex(ValueError, "only permitted"):
                _load_ap_terminal(
                    arm_dir=arm_dir,
                    identifier="codriving:compression_only",
                    runtime="tensorrt",
                )

    def test_accepts_codriving_full_1789_gate_without_pyramid_claim_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_sha = write_engine_evidence(
                root / "codriving" / "compression_only"
            )
            manifest = {
                "rows": [
                    {
                        "id": "codriving:compression_only",
                        "model": "codriving",
                        "arm": "compression_only",
                        "widths": [32, 64, 96],
                        "precision": "fp16",
                        "runtime": "tensorrt",
                        "source_sha256": {"checkpoint": "a" * 64},
                        "h800_source": {
                            "ap70": 0.41,
                            "latency_ms": 0.35,
                            "energy_j": 0.10,
                        },
                    }
                ]
            }
            write_json(
                root
                / "codriving"
                / "compression_only"
                / "primary_latency_power.json",
                {
                    "sample_count": 1500,
                    "median_ms": 8.4,
                    "p90_ms": 8.5,
                    "p99_ms": 8.6,
                    "mean_ms": 8.4,
                    "scope": (
                        "codriving_backbone_resnet_32x64x96_"
                        "engine_compute_no_data_transfer"
                    ),
                    "engine_sha256": engine_sha,
                    "protocol": {
                        "warmup": 20,
                        "iters": 300,
                        "repeat": 5,
                        "timing": "CUDA_event",
                        "data_transfer_inside_timed_region": False,
                    },
                    "power_measurement": {
                        "measurement_status": "unavailable",
                        "blocker": "missing_tegrastats_power_rails",
                    },
                },
            )
            write_json(
                root
                / "codriving"
                / "compression_only"
                / "ap_full"
                / "report.json",
                {
                    "status": "success",
                    "processed_samples": 1789,
                    "failed_samples": 0,
                    "fallback_samples": 0,
                    "gates": {"full_1789": True},
                    "ap50": 0.42,
                    "ap70": 0.31,
                    "output_vs_reference_error": {
                        "all_finite": True,
                        "shape_mismatch_count": 0,
                        "nonfinite_count": 0,
                        "num_records": 1789 * 6,
                        "num_compared": 1789 * 6,
                    },
                },
            )

            rows = [collect_result_row(manifest["rows"][0], root)]

            self.assertEqual(rows[0]["orin"]["ap_status"], "full_1789_success")
            self.assertEqual(rows[0]["orin"]["ap70"], 0.31)

    def test_writes_null_energy_and_explicit_blocker_to_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text('{"rows":[]}\n', encoding="utf-8")
            rows = [
                {
                    "id": "pyramid:original_default",
                    "model": "pyramid",
                    "arm": "original_default",
                    "widths": [64, 128, 256],
                    "precision": "fp32",
                    "runtime": "native_pytorch",
                    "h800": {
                        "ap70": 0.6311,
                        "latency_ms": 3.2187,
                        "energy_j": 1.1636,
                    },
                    "orin": {
                        "ap50": 0.7907,
                        "ap70": 0.6313,
                        "latency_median_ms": 126.5,
                        "latency_p90_ms": 126.8,
                        "latency_p99_ms": 127.1,
                        "latency_mean_ms": 126.6,
                        "energy_j": None,
                        "energy_status": "missing_tegrastats_power_rails",
                    },
                    "evidence_sha256": {
                        "checkpoint": "a" * 64,
                        "engine": None,
                        "latency_report": "b" * 64,
                        "ap_report": "c" * 64,
                        "build_report": None,
                    },
                }
            ]

            outputs = write_outputs(
                rows=rows,
                output_dir=root / "out",
                manifest_path=manifest_path,
            )

            summary = json.loads(
                outputs["summary"].read_text(encoding="utf-8")
            )
            self.assertIsNone(summary["rows"][0]["orin"]["energy_j"])
            self.assertIn(
                "missing_tegrastats_power_rails",
                outputs["csv"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "missing_tegrastats_power_rails",
                outputs["markdown"].read_text(encoding="utf-8"),
            )
            sha_manifest = json.loads(
                outputs["sha_manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(sha_manifest["files"]),
                {
                    "final_summary.json",
                    "orin_stage6_five_config.csv",
                    "summary.md",
                },
            )


if __name__ == "__main__":
    unittest.main()
