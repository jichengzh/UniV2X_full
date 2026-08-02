from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from tools.orin_deploy.lane_c_backbone_parity_runner import (
    build_power_measurement,
    builder_flags_for_precision,
    configure_builder_precision,
    deployment_measurement_scope,
    multiscale_backbone_scope,
    multiscale_measurement_scope,
    measure_primary,
    parse_args,
    parse_nvml_power_log,
    summarize_latency_samples,
    strict_fp32_inspector_evidence,
    run_numerical,
    validate_calibration_manifest,
    validate_fresh_build_outputs,
    validate_output_channel_signature,
    validate_engine_output_channel_signature,
    validate_primary_protocol,
    validate_heldout_inputs,
    validate_output_paths,
)


class LaneCBackboneParityRunnerTest(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_builder_flags_match_h800_contract(self) -> None:
        self.assertEqual(
            builder_flags_for_precision("fp32"), ["fp32", "tf32_disabled"]
        )
        self.assertEqual(builder_flags_for_precision("fp16"), ["fp16"])
        self.assertEqual(
            builder_flags_for_precision("int8"), ["int8", "fp16_fallback"]
        )

    def test_fp32_precision_clears_tf32_without_enabling_reduced_precision(self) -> None:
        class FakeBuilderFlag:
            TF32 = "tf32"
            FP16 = "fp16"
            INT8 = "int8"

        class FakeTensorRT:
            BuilderFlag = FakeBuilderFlag

        config = mock.Mock()
        configure_builder_precision(
            FakeTensorRT(), mock.Mock(platform_has_fast_int8=True), config, "fp32"
        )
        config.clear_flag.assert_called_once_with(FakeBuilderFlag.TF32)
        config.set_flag.assert_not_called()

    def test_output_channel_signature_requires_expected_ordered_channels(self) -> None:
        self.assertEqual(
            validate_output_channel_signature(
                [[2, 64, 128, 256], [2, 128, 64, 128], [2, 256, 32, 64]],
                expected=(64, 128, 256),
            ),
            [64, 128, 256],
        )
        with self.assertRaisesRegex(ValueError, "output channel signature"):
            validate_output_channel_signature(
                [[2, 64, 128, 256], [2, 64, 64, 128], [2, 256, 32, 64]],
                expected=(64, 128, 256),
            )

    def test_multiscale_scope_is_bound_to_the_required_channel_signature(self) -> None:
        self.assertEqual(
            multiscale_backbone_scope((64, 128, 256)),
            "pyramid_get_multiscale_feature_64x128x256_only",
        )

    def test_measurement_scope_is_bound_to_the_required_channel_signature(self) -> None:
        self.assertEqual(
            multiscale_measurement_scope((64, 128, 256)),
            "pyramid_get_multiscale_feature_64x128x256_engine_compute_no_data_transfer",
        )
        self.assertEqual(
            deployment_measurement_scope(
                "codriving", (32, 64, 96)
            ),
            "codriving_backbone_resnet_32x64x96_engine_compute_no_data_transfer",
        )

    def test_power_measurement_integrates_named_orin_rail_over_median_latency(
        self,
    ) -> None:
        measurement = build_power_measurement(
            power={
                "rail_semantics": "Jetson rail telemetry",
                "vin_sys_5v0_mean_w": 8.0,
                "vdd_gpu_soc_mean_w": 7.5,
            },
            power_source="tegrastats",
            latency_summary={"median_ms": 10.0},
            raw_power_sha256="a" * 64,
            used_sudo=True,
        )

        self.assertEqual(measurement["measurement_status"], "available")
        self.assertEqual(measurement["energy_rail"], "VIN_SYS_5V0")
        self.assertAlmostEqual(measurement["energy_j"], 0.08)
        self.assertEqual(
            measurement["energy_method"],
            "mean_named_rail_power_w_x_median_engine_latency_s",
        )
        self.assertTrue(measurement["used_sudo"])

    def test_strict_fp32_inspector_evidence_rejects_reduced_precision_or_tf32(self) -> None:
        for precision in ("FP16", "INT8", "TF32", "BF16"):
            evidence = strict_fp32_inspector_evidence(
                {"Layers": [{"Name": "conv", "Precision": precision}]}
            )
            self.assertFalse(evidence["strict_fp32"])
            self.assertFalse(evidence["tf32_allowed"])
            self.assertEqual(
                evidence["forbidden_inspector_matches"][0]["precision"],
                precision.lower(),
            )

    def test_strict_fp32_inspector_evidence_accepts_untyped_tensorrt_structural_layers(self) -> None:
        evidence = strict_fp32_inspector_evidence(
            {
                "Layers": [
                    {
                        "Name": "conv",
                        "LayerType": "Convolution",
                        "Outputs": [{"Format/Datatype": "Float"}],
                    },
                    {"Name": "shuffle", "LayerType": "Shuffle"},
                    {"Name": "constant", "LayerType": "Constant"},
                    {"Name": "shape", "LayerType": "Shape"},
                    {"Name": "resize", "LayerType": "Resize"},
                ]
            }
        )
        self.assertTrue(evidence["strict_fp32"])
        self.assertEqual(evidence["forbidden_inspector_matches"], [])
        self.assertEqual(
            [row["layer_name"] for row in evidence["untyped_structural_layers"]],
            ["shuffle", "constant", "shape", "resize"],
        )

    def test_strict_fp32_inspector_evidence_rejects_untyped_compute_layers(self) -> None:
        evidence = strict_fp32_inspector_evidence(
            {"Layers": [{"Name": "conv", "LayerType": "Convolution"}]}
        )
        self.assertFalse(evidence["strict_fp32"])
        self.assertEqual(
            evidence["forbidden_inspector_matches"][0]["precision"], "unavailable"
        )

    def test_build_parser_keeps_legacy_precision_channel_defaults(self) -> None:
        base = [
            "build",
            "--artifact-root",
            "/tmp/artifacts",
            "--onnx",
            "/tmp/model.onnx",
            "--engine",
            "/tmp/model.engine",
            "--inspector-json",
            "/tmp/inspector.json",
            "--artifact-json",
            "/tmp/build.json",
        ]
        for precision in ("fp16", "int8"):
            args = parse_args(base + ["--precision", precision])
            self.assertEqual(args.expected_output_channels, (16, 32, 64))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(base + ["--precision", "fp32"])
        args = parse_args(
            base
            + [
                "--precision",
                "fp32",
                "--expected-output-channels",
                "64,128,256",
            ]
        )
        self.assertEqual(args.expected_output_channels, (64, 128, 256))

    def test_numerical_and_measurement_parsers_accept_dynamic_channel_contracts(self) -> None:
        numerical = [
            "numerical",
            "--artifact-root",
            "/tmp/artifacts",
            "--engine",
            "/tmp/model.engine",
            "--inputs-npy",
            "/tmp/inputs.npy",
            "--output-npz",
            "/tmp/outputs.npz",
            "--report-json",
            "/tmp/report.json",
        ]
        self.assertEqual(parse_args(numerical).expected_output_channels, (16, 32, 64))
        self.assertEqual(
            parse_args(numerical + ["--expected-output-channels", "64,128,256"])
            .expected_output_channels,
            (64, 128, 256),
        )
        measurement = [
            "measure-primary",
            "--artifact-root",
            "/tmp/artifacts",
            "--engine",
            "/tmp/model.engine",
            "--inputs-npy",
            "/tmp/inputs.npy",
            "--power-log",
            "/tmp/power.log",
            "--output-json",
            "/tmp/measurement.json",
        ]
        self.assertEqual(
            parse_args(measurement + ["--expected-output-channels", "64,128,256"])
            .expected_output_channels,
            (64, 128, 256),
        )
        self.assertFalse(parse_args(measurement).allow_missing_power_rails)
        self.assertTrue(
            parse_args(measurement + ["--allow-missing-power-rails"])
            .allow_missing_power_rails
        )

    def test_parsers_accept_codriving_input_and_calibration_contracts(self) -> None:
        build = [
            "build",
            "--artifact-root",
            "/tmp/artifacts",
            "--onnx",
            "/tmp/model.onnx",
            "--engine",
            "/tmp/model.engine",
            "--precision",
            "int8",
            "--expected-output-channels",
            "16,32,64",
            "--expected-input-shape",
            "2,64,256,512",
            "--expected-calibration-count",
            "16",
            "--calibration-dir",
            "/tmp/calibration",
            "--calibration-manifest",
            "/tmp/calibration.json",
            "--calibration-cache",
            "/tmp/calibration.cache",
            "--inspector-json",
            "/tmp/inspector.json",
            "--artifact-json",
            "/tmp/build.json",
        ]
        build_args = parse_args(build)
        self.assertEqual(build_args.expected_input_shape, (2, 64, 256, 512))
        self.assertEqual(build_args.expected_calibration_count, 16)

        numerical = [
            "numerical",
            "--artifact-root",
            "/tmp/artifacts",
            "--engine",
            "/tmp/model.engine",
            "--inputs-npy",
            "/tmp/inputs.npy",
            "--output-npz",
            "/tmp/outputs.npz",
            "--report-json",
            "/tmp/report.json",
            "--expected-input-shape",
            "2,64,256,512",
        ]
        self.assertEqual(
            parse_args(numerical).expected_input_shape, (2, 64, 256, 512)
        )

    def test_validate_heldout_inputs_accepts_codriving_shape(self) -> None:
        inputs = np.zeros((3, 2, 64, 256, 512), dtype=np.float32)
        self.assertEqual(
            validate_heldout_inputs(
                inputs, expected_input_shape=(2, 64, 256, 512)
            ),
            [3, 2, 64, 256, 512],
        )

    def test_calibration_manifest_count_is_explicit_not_globally_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(2):
                filename = f"batch2_{index:03d}.npy"
                path = root / filename
                np.save(path, np.zeros((2, 64, 256, 512), dtype=np.float32))
                rows.append(
                    {
                        "filename": filename,
                        "bytes": path.stat().st_size,
                        "sha256": self._sha256(path),
                    }
                )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "lane_c_shared_calibration_manifest_v1",
                        "calibration": {
                            "file_count": 2,
                            "batch_size": 2,
                            "files": rows,
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit = validate_calibration_manifest(
                manifest_path=manifest,
                calibration_dir=root,
                expected_shape=(2, 64, 256, 512),
                expected_file_count=2,
            )
            self.assertEqual(audit["verified_file_count"], 2)

    def test_measure_primary_records_missing_tegrastats_rails_only_when_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_path = root / "model.engine"
            inputs_path = root / "inputs.npy"
            power_log = root / "tegrastats.log"
            output_json = root / "measurement.json"
            engine_path.write_bytes(b"engine")
            inputs_path.write_bytes(b"inputs")
            power_log.write_text("GR3D_FREQ 99% GPU@61C\n", encoding="utf-8")
            args = argparse.Namespace(
                warmup=20,
                iters=300,
                repeat=5,
                engine=engine_path,
                inputs_npy=inputs_path,
                expected_output_channels=(64, 128, 256),
                power_source="tegrastats",
                power_log=power_log,
                nvml_gpu_index=0,
                output_json=output_json,
                allow_missing_power_rails=True,
            )
            stream = mock.Mock()

            def execute(timed: bool) -> float | None:
                return 1.0 if timed else None

            with (
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner._deserialize_engine",
                    return_value=("fake-trt", mock.Mock()),
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner.validate_engine_output_channel_signature",
                    return_value=[64, 128, 256],
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner.np.load",
                    return_value=[object()],
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner.validate_heldout_inputs"
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner._prepare_execution",
                    return_value=(execute, stream, {}, []),
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner._start_tegrastats",
                    return_value=(mock.Mock(), mock.Mock(), False),
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner._stop_tegrastats"
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner.time.sleep"
                ),
                mock.patch(
                    "tools.orin_deploy.lane_c_backbone_parity_runner.parse_tegrastats",
                    side_effect=ValueError("tegrastats contains no VIN_SYS_5V0 samples"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(measure_primary(args), 0)

            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["sample_count"], 1500)
            self.assertEqual(
                report["power_measurement"],
                {
                    "measurement_status": "unavailable",
                    "blocker": "missing_tegrastats_power_rails",
                    "source": "Orin tegrastats unavailable_without_privileged_rail_access",
                    "raw_log_sha256": self._sha256(power_log),
                    "used_sudo": False,
                },
            )

    def test_numerical_validates_engine_channels_before_execution(self) -> None:
        args = argparse.Namespace(
            engine=Path("/tmp/model.engine"),
            inputs_npy=Path("/tmp/missing-inputs.npy"),
            output_npz=Path("/tmp/outputs.npz"),
            report_json=Path("/tmp/report.json"),
            expected_output_channels=(64, 128, 256),
        )
        engine = mock.Mock()
        with (
            mock.patch(
                "tools.orin_deploy.lane_c_backbone_parity_runner._deserialize_engine",
                return_value=("fake-trt", engine),
            ),
            mock.patch(
                "tools.orin_deploy.lane_c_backbone_parity_runner.validate_engine_output_channel_signature",
                side_effect=ValueError("output channel signature mismatch"),
            ) as validate_channels,
            mock.patch(
                "tools.orin_deploy.lane_c_backbone_parity_runner._prepare_execution"
            ) as prepare_execution,
        ):
            with self.assertRaisesRegex(ValueError, "output channel signature mismatch"):
                run_numerical(args)
        validate_channels.assert_called_once_with(
            "fake-trt", engine, expected=(64, 128, 256)
        )
        prepare_execution.assert_not_called()

    def test_engine_output_channel_boundary_rejects_mismatched_or_reordered_outputs(self) -> None:
        class FakeTensorIOMode:
            INPUT = "input"
            OUTPUT = "output"

        class FakeTensorRT:
            TensorIOMode = FakeTensorIOMode

        class FakeEngine:
            num_io_tensors = 4

            def __init__(self, output_shapes: list[list[int]]) -> None:
                self._names = ["input", "level0", "level1", "level2"]
                self._shapes = [[2, 64, 128, 256], *output_shapes]

            def get_tensor_name(self, index: int) -> str:
                return self._names[index]

            def get_tensor_mode(self, name: str) -> str:
                return (
                    FakeTensorIOMode.INPUT
                    if name == "input"
                    else FakeTensorIOMode.OUTPUT
                )

            def get_tensor_shape(self, name: str) -> list[int]:
                return self._shapes[self._names.index(name)]

        expected = (64, 128, 256)
        self.assertEqual(
            validate_output_channel_signature(
                [[2, 64, 128, 256], [2, 128, 64, 128], [2, 256, 32, 64]],
                expected=expected,
            ),
            [64, 128, 256],
        )
        self.assertEqual(
            validate_engine_output_channel_signature(
                FakeTensorRT(),
                FakeEngine(
                    [[2, 64, 128, 256], [2, 128, 64, 128], [2, 256, 32, 64]]
                ),
                expected=expected,
            ),
            [64, 128, 256],
        )
        for output_shapes in (
            [[2, 16, 128, 256], [2, 32, 64, 128], [2, 64, 32, 64]],
            [[2, 128, 64, 128], [2, 64, 128, 256], [2, 256, 32, 64]],
        ):
            with self.assertRaisesRegex(ValueError, "output channel signature mismatch"):
                validate_engine_output_channel_signature(
                    FakeTensorRT(), FakeEngine(output_shapes), expected=expected
                )

    def test_validate_heldout_inputs_requires_batch2_backbone_shape(self) -> None:
        valid = np.zeros((3, 2, 64, 128, 256), dtype=np.float16)
        self.assertEqual(validate_heldout_inputs(valid), [3, 2, 64, 128, 256])
        with self.assertRaisesRegex(ValueError, "expected"):
            validate_heldout_inputs(np.zeros((3, 1, 64, 128, 256), dtype=np.float32))

    def test_latency_summary_uses_h800_order_statistic_contract(self) -> None:
        samples = [float(value) for value in range(1, 101)]
        summary = summarize_latency_samples(samples)
        self.assertEqual(summary["median_ms"], 50.5)
        self.assertEqual(summary["p90_ms"], 90.0)
        self.assertEqual(summary["p99_ms"], 99.0)
        self.assertEqual(summary["mean_ms"], 50.5)

    def test_primary_protocol_is_locked_to_h800_contract(self) -> None:
        validate_primary_protocol(warmup=20, iters=300, repeat=5)
        with self.assertRaisesRegex(ValueError, "warmup=20"):
            validate_primary_protocol(warmup=0, iters=300, repeat=5)

    def test_output_paths_are_confined_to_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            valid = argparse.Namespace(
                command="numerical",
                artifact_root=root,
                output_npz=root / "numerical" / "outputs.npz",
                report_json=root / "numerical" / "report.json",
            )
            validate_output_paths(valid)
            escaped = argparse.Namespace(
                command="numerical",
                artifact_root=root,
                output_npz=root.parent / "escaped.npz",
                report_json=root / "numerical" / "report.json",
            )
            with self.assertRaisesRegex(ValueError, "outside artifact root"):
                validate_output_paths(escaped)

    def test_calibration_manifest_requires_15_byte_identical_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_dir = root / "calibration"
            calibration_dir.mkdir()
            files = []
            for index in range(15):
                path = calibration_dir / f"batch2_{index:03d}.npy"
                np.save(path, np.full((2, 1, 1, 1), index, dtype=np.float32))
                files.append(
                    {
                        "filename": path.name,
                        "sha256": self._sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
            manifest = {
                "schema_version": "lane_c_shared_calibration_manifest_v1",
                "calibration": {
                    "file_count": 15,
                    "batch_size": 2,
                    "dtype": "float32",
                    "files": files,
                },
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = validate_calibration_manifest(
                manifest_path=manifest_path,
                calibration_dir=calibration_dir,
                expected_shape=None,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["verified_file_count"], 15)

            (calibration_dir / "batch2_007.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                validate_calibration_manifest(
                    manifest_path=manifest_path,
                    calibration_dir=calibration_dir,
                    expected_shape=None,
                )

    def test_fresh_build_rejects_any_preexisting_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                root / "backbone.engine",
                root / "calibration.cache",
                root / "inspector.json",
                root / "build_report.json",
            ]
            validate_fresh_build_outputs(paths)
            paths[1].write_bytes(b"old-cache")
            with self.assertRaisesRegex(FileExistsError, "fresh build"):
                validate_fresh_build_outputs(paths)

    def test_nvml_power_parser_keeps_board_power_semantics(self) -> None:
        report = parse_nvml_power_log(
            "\n".join(
                [
                    "2026/07/24 01:02:03.000, 71.25",
                    "2026/07/24 01:02:03.050, 75.75",
                ]
            )
        )
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["board_power_w"], [71.25, 75.75])
        self.assertEqual(report["board_power_mean_w"], 73.5)
        self.assertEqual(report["physical_quantity"], "NVML board power")

    @mock.patch(
        "tools.orin_deploy.lane_c_backbone_parity_runner.subprocess.run"
    )
    def test_privileged_tegrastats_stop_uses_stop_command(
        self, run: mock.Mock
    ) -> None:
        from tools.orin_deploy.lane_c_backbone_parity_runner import _stop_tegrastats

        process = mock.Mock()
        handle = mock.Mock()
        with mock.patch.dict("os.environ", {"ORIN_SUDO_PW": "temporary"}):
            _stop_tegrastats(process, handle, privileged=True)
        run.assert_called_once_with(
            ["sudo", "-S", "-p", "", "tegrastats", "--stop"],
            input="temporary\n",
            text=True,
            check=True,
            timeout=5,
            env=mock.ANY,
        )
        self.assertNotIn("ORIN_SUDO_PW", run.call_args.kwargs["env"])
        process.wait.assert_called_once_with(timeout=5)
        handle.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
