from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.native_int8_route import (
    NATIVE_INT8_QUANT_METHOD,
    NativeInt8RouteError,
    expected_label_output_paths,
    validate_blocker_attempt_summary,
    validate_native_route_record,
)
from framework.stage2.native_int8_full_onnx import (
    REQUIRED_FULL_ONNX_BLOCKER_FIELDS,
    REQUIRED_FULL_ONNX_OPS,
    build_runtime_arg_plan,
    build_tvm_worker_request,
    build_conv_op_spec,
    expected_full_onnx_row_paths,
    parse_input_shape_overrides,
    quantize_activation_uint8,
    quantize_activation_uint8_static,
    resolve_initializer_name,
    validate_full_onnx_blocker,
    validate_tvm_worker_response,
)
from framework.stage2.lut_productization import (
    LutProductizationError,
    latency_lut_row,
    validate_lut_row,
)


FULL_ONNX_ROUTE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "raw/int8_native_route/stage2_h800_native_int8_full_onnx_route.py"
)


def load_full_onnx_route_module():
    module_dir = str(FULL_ONNX_ROUTE_SCRIPT.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(
        "stage2_h800_native_int8_full_onnx_route_under_test",
        FULL_ONNX_ROUTE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import {FULL_ONNX_ROUTE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_int8_latency_row(engine_kind: str) -> dict[str, object]:
    return latency_lut_row(
        config_id="native_int8_latency_smoke_s0_024",
        model="pyramid_lidar",
        manifest_digest="native_int8_manifest",
        candidate_id="s0_024",
        software_point_id="native_int8:s0_024:24x128x256:latency",
        dense_stage="backbone",
        optimized_scope="backbone_only",
        width=[24, 128, 256],
        quant_policy="int8_native",
        schedule_policy="native_int8_qnn_direct",
        backend="h800_tvm",
        measurement_status="measured",
        run_id="native_int8_latency_smoke_s0_024_h800",
        created_at="2026-06-27T00:00:00Z",
        latency_p50_us=50.0,
        source_files=["raw/int8_native_route/run/s0_024/latency_result.json"],
        raw_artifact="raw/int8_native_route/run/s0_024",
        provenance="H800 TVM native INT8 latency smoke",
        notes="native INT8 graph executor route",
        precision="int8",
        quant_scheme="tvm_native_int8_qnn_conv_chain",
        quant_method=NATIVE_INT8_QUANT_METHOD,
        quant_scope="backbone_subnet_native_int8",
        calibration_source="none_direct_native_int8_synthetic_inputs",
        calibration_digest="none",
        calibrator="none",
        fallback_policy="none",
        layer_precision_summary="raw/int8_native_route/run/s0_024/tvm_operator_inventory.json",
        full_network_claim=False,
        engine_kind=engine_kind,
        engine_digest="native-int8-engine-digest",
        measurement_source="true_measurement_smoke",
        claim_status="claimable_true_measurement_smoke",
        quality_gate_status="native_int8_tvm_latency_smoke_only",
        schedule_profile="native_int8_qnn_direct",
        tune_budget="no_metaschedule_native_int8_probe",
    )


class Stage2NativeInt8RouteTest(unittest.TestCase):
    def test_expected_label_output_paths_include_latency_and_energy_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = expected_label_output_paths(Path(tmp), "s0_024")

        self.assertEqual(
            [path.name for path in paths],
            [
                "latency_result.json",
                "energy_result.json",
                "idle_power_samples.csv",
                "active_power_samples.csv",
                "telemetry_payload.json",
            ],
        )

    def test_native_route_rejects_static_qdq_method(self) -> None:
        record = {
            "label": "s0_024",
            "quant_method": "static_qdq_synthetic_minmax",
            "quant_scope": "backbone_subnet_native_int8",
            "full_network_claim": False,
        }

        with self.assertRaisesRegex(NativeInt8RouteError, "static_qdq"):
            validate_native_route_record(record)

    def test_native_route_requires_energy_artifact_paths(self) -> None:
        record = {
            "label": "s0_024",
            "quant_method": NATIVE_INT8_QUANT_METHOD,
            "quant_scope": "backbone_subnet_native_int8",
            "full_network_claim": False,
            "latency_result_path": "raw/int8_native_route/run/s0_024/latency_result.json",
            "energy_result_path": "raw/int8_native_route/run/s0_024/energy_result.json",
            "idle_power_samples_path": "raw/int8_native_route/run/s0_024/idle_power_samples.csv",
            "active_power_samples_path": "raw/int8_native_route/run/s0_024/active_power_samples.csv",
            "telemetry_payload_path": "raw/int8_native_route/run/s0_024/telemetry_payload.json",
        }

        validate_native_route_record(record)

    def test_blocker_summary_requires_all_three_attempt_classes(self) -> None:
        summary = {
            "attempts": {
                "tiny_int8_conv": {
                    "attempt_json": "tiny_int8_conv_attempt.json",
                    "stdout_path": "tiny_int8_conv_stdout.txt",
                    "stderr_path": "tiny_int8_conv_stderr.txt",
                    "traceback": "Traceback...",
                    "build_status": "failed",
                    "run_status": "not_run",
                    "failure_reason": "missing tvm",
                },
                "resnet_style_int8_block": {
                    "attempt_json": "resnet_style_int8_block_attempt.json",
                    "stdout_path": "resnet_style_int8_block_stdout.txt",
                    "stderr_path": "resnet_style_int8_block_stderr.txt",
                    "traceback": "Traceback...",
                    "build_status": "failed",
                    "run_status": "not_run",
                    "failure_reason": "missing tvm",
                },
                "target_backbone_subnet_route": {
                    "attempt_json": "target_backbone_subnet_route_attempt.json",
                    "stdout_path": "target_backbone_subnet_route_stdout.txt",
                    "stderr_path": "target_backbone_subnet_route_stderr.txt",
                    "traceback": "Traceback...",
                    "build_status": "failed",
                    "run_status": "not_run",
                    "failure_reason": "missing tvm",
                },
            }
        }

        validate_blocker_attempt_summary(summary)

    def test_blocker_summary_rejects_missing_target_attempt(self) -> None:
        summary = {
            "attempts": {
                "tiny_int8_conv": {
                    "attempt_json": "tiny_int8_conv_attempt.json",
                    "stdout_path": "tiny_int8_conv_stdout.txt",
                    "stderr_path": "tiny_int8_conv_stderr.txt",
                    "traceback": "Traceback...",
                    "build_status": "failed",
                    "run_status": "not_run",
                    "failure_reason": "missing tvm",
                },
                "resnet_style_int8_block": {
                    "attempt_json": "resnet_style_int8_block_attempt.json",
                    "stdout_path": "resnet_style_int8_block_stdout.txt",
                    "stderr_path": "resnet_style_int8_block_stderr.txt",
                    "traceback": "Traceback...",
                    "build_status": "failed",
                    "run_status": "not_run",
                    "failure_reason": "missing tvm",
                },
            }
        }

        with self.assertRaisesRegex(NativeInt8RouteError, "target_backbone_subnet_route"):
            validate_blocker_attempt_summary(summary)

    def test_native_int8_latency_row_allows_graph_executor(self) -> None:
        row = native_int8_latency_row("tvm_graph_executor")

        validate_lut_row(row)

    def test_non_native_int8_latency_row_still_requires_tvm_vm(self) -> None:
        row = native_int8_latency_row("tvm_graph_executor")
        row["quant_method"] = "h800_tvm_int8_backbone_subnet_experimental"
        row["quant_scope"] = "backbone_only_requested"

        with self.assertRaisesRegex(LutProductizationError, "engine_kind=tvm_vm"):
            validate_lut_row(row)

    def test_full_onnx_row_paths_are_separate_from_smoke_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            latency_rows, energy_rows = expected_full_onnx_row_paths(Path(tmp))

        self.assertEqual(latency_rows.name, "native_int8_full_onnx_latency_rows_v1.jsonl")
        self.assertEqual(energy_rows.name, "native_int8_full_onnx_energy_rows_v1.jsonl")

    def test_full_onnx_conv_spec_records_group_stride_pad_and_shapes(self) -> None:
        spec = build_conv_op_spec(
            name="/resnet/layer1/layer1.0/conv2/Conv",
            input_name="x",
            output_name="y",
            input_shape=[2, 256, 64, 128],
            output_shape=[2, 256, 32, 64],
            weight_shape=[256, 8, 3, 3],
            group=32,
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        )

        self.assertEqual(spec["op_type"], "Conv")
        self.assertEqual(spec["group"], 32)
        self.assertEqual(spec["strides"], [2, 2])
        self.assertEqual(spec["pads"], [1, 1, 1, 1])
        self.assertEqual(spec["input_shape"], [2, 256, 64, 128])
        self.assertEqual(spec["output_shape"], [2, 256, 32, 64])

    def test_full_onnx_blocker_requires_op_level_fields(self) -> None:
        self.assertEqual(REQUIRED_FULL_ONNX_OPS, ("Conv", "Relu", "Add", "Identity"))
        blocker = {
            "op_name": "/resnet/layer1/layer1.0/conv2/Conv",
            "op_type": "Conv",
            "shape": [2, 256, 32, 64],
            "group": 32,
            "stride": [2, 2],
            "pad": [1, 1, 1, 1],
            "stdout_path": "stdout.txt",
            "stderr_path": "stderr.txt",
            "traceback": "Traceback...",
            "build_status": "failed",
            "run_status": "not_run",
            "failure_reason": "unsupported grouped conv schedule",
        }

        validate_full_onnx_blocker(blocker)
        self.assertEqual(
            tuple(blocker.keys()),
            REQUIRED_FULL_ONNX_BLOCKER_FIELDS,
        )

    def test_full_onnx_resolves_chained_initializer_identity_aliases(self) -> None:
        aliases = {
            "identity_weight_1": "onnx::Conv_548",
            "identity_weight_2": "identity_weight_1",
        }

        self.assertEqual(
            resolve_initializer_name("identity_weight_2", aliases),
            "onnx::Conv_548",
        )

    def test_full_onnx_parses_ap_shape_override_for_graph_input(self) -> None:
        overrides = parse_input_shape_overrides(["spatial_features=1,64,256,256"])

        self.assertEqual(overrides, {"spatial_features": [1, 64, 256, 256]})

    def test_full_onnx_rejects_non_nchw_shape_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "NCHW"):
            parse_input_shape_overrides(["spatial_features=1,64,256"])

    def test_runtime_arg_plan_marks_real_weight_inputs_before_outputs(self) -> None:
        plan = build_runtime_arg_plan(
            graph_input_shapes={"spatial_features": [1, 64, 256, 256]},
            weight_inputs=[
                {
                    "arg_name": "weight_1_resnet_layer0_conv1",
                    "initializer_name": "resnet.layer0.conv1.weight",
                    "shape": [24, 64, 3, 3],
                    "dtype": "int8",
                    "quantization": {
                        "scheme": "symmetric_absmax_int8",
                        "scale": 0.0125,
                        "zero_point": 0,
                    },
                },
            ],
            output_shapes={
                "/resnet/layer0/relu/Relu_output_0": [1, 24, 256, 256],
            },
        )

        self.assertEqual(
            [item["role"] for item in plan],
            ["graph_input", "weight_input", "graph_output"],
        )
        self.assertEqual(plan[1]["source"], "onnx_initializer_quantized_int8")
        self.assertEqual(plan[1]["initializer_name"], "resnet.layer0.conv1.weight")

    def test_runtime_arg_plan_preserves_bias_initializer_role(self) -> None:
        plan = build_runtime_arg_plan(
            graph_input_shapes={"spatial_features": [1, 64, 256, 256]},
            weight_inputs=[
                {
                    "role": "weight_input",
                    "arg_name": "weight_1_resnet_layer0_conv1",
                    "initializer_name": "resnet.layer0.conv1.weight",
                    "shape": [24, 64, 3, 3],
                    "dtype": "int8",
                    "quantization": {"scheme": "symmetric_absmax_int8", "scale": 0.0125},
                },
                {
                    "role": "bias_input",
                    "arg_name": "bias_1_resnet_layer0_conv1",
                    "initializer_name": "resnet.layer0.conv1.bias",
                    "shape": [24],
                    "dtype": "int32",
                    "quantization": {
                        "scheme": "float_bias_to_int32_accumulator",
                        "scale": 0.003125,
                    },
                },
            ],
            output_shapes={"pyramid_level0": [1, 24, 256, 256]},
        )

        self.assertEqual(
            [item["role"] for item in plan],
            ["graph_input", "weight_input", "bias_input", "graph_output"],
        )
        self.assertEqual(plan[2]["dtype"], "int32")
        self.assertEqual(plan[2]["source"], "onnx_initializer_quantized_int32_bias")

    def test_tvm_worker_request_records_artifacts_runtime_plan_and_expected_outputs(self) -> None:
        plan = build_runtime_arg_plan(
            graph_input_shapes={"spatial_features": [1, 64, 256, 256]},
            weight_inputs=[
                {
                    "arg_name": "weight_1_resnet_layer0_conv1",
                    "initializer_name": "resnet.layer0.conv1.weight",
                    "shape": [24, 64, 3, 3],
                    "dtype": "int8",
                    "quantization": {
                        "scheme": "symmetric_absmax_int8",
                        "scale": 0.0125,
                        "zero_point": 0,
                    },
                },
            ],
            output_shapes={
                "pyramid_level0": [1, 24, 256, 256],
                "pyramid_level1": [1, 128, 128, 128],
            },
        )

        request = build_tvm_worker_request(
            label="s0_024",
            run_id="native_int8_worker_smoke",
            artifact_path=Path("s0_024_tvm_graph.so"),
            inventory_path=Path("tvm_operator_inventory.json"),
            runtime_weight_archive_path=Path("runtime_weights_int8.npz"),
            activation_npy_path=Path("activation_uint8.npy"),
            output_dir=Path("worker_outputs"),
            gpu=0,
            runtime_arg_plan=plan,
        )

        self.assertEqual(request["schema"], "native_int8_tvm_worker_request_v1")
        self.assertEqual(request["label"], "s0_024")
        self.assertEqual(request["gpu"], 0)
        self.assertEqual(request["artifact_path"], "s0_024_tvm_graph.so")
        self.assertEqual(request["runtime_weight_archive_path"], "runtime_weights_int8.npz")
        self.assertEqual(
            request["expected_output_shapes"],
            {
                "pyramid_level0": [1, 24, 256, 256],
                "pyramid_level1": [1, 128, 128, 128],
            },
        )
        self.assertEqual([item["role"] for item in request["runtime_arg_plan"]], ["graph_input", "weight_input", "graph_output", "graph_output"])

    def test_tvm_worker_response_rejects_missing_expected_output(self) -> None:
        response = {
            "schema": "native_int8_tvm_worker_response_v1",
            "status": "success",
            "outputs": [
                {
                    "arg_name": "pyramid_level0",
                    "path": "worker_outputs/pyramid_level0.npy",
                    "shape": [1, 24, 256, 256],
                    "dtype": "uint8",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "missing worker output"):
            validate_tvm_worker_response(
                response,
                expected_output_shapes={
                    "pyramid_level0": [1, 24, 256, 256],
                    "pyramid_level1": [1, 128, 128, 128],
                },
            )

    def test_tvm_worker_script_uses_stable_output_filename_for_onnx_tensor_name(self) -> None:
        from scripts.stage2_native_int8_tvm_worker import output_filename_for_arg

        self.assertEqual(
            output_filename_for_arg("/resnet/layer0/layer0.2/relu_2/Relu_output_0"),
            "out_resnet_layer0_layer0_2_relu_2_Relu_output_0.npy",
        )

    def test_tvm_worker_script_builds_response_record_for_output_array(self) -> None:
        import numpy as np

        from scripts.stage2_native_int8_tvm_worker import build_output_record

        record = build_output_record(
            arg_name="pyramid_level0",
            output_path=Path("worker_outputs/pyramid_level0.npy"),
            array=np.zeros((1, 24, 256, 256), dtype=np.uint8),
        )

        self.assertEqual(
            record,
            {
                "arg_name": "pyramid_level0",
                "path": "worker_outputs/pyramid_level0.npy",
                "shape": [1, 24, 256, 256],
                "dtype": "uint8",
            },
        )

    def test_tvm_worker_script_uses_runtime_tensor_fallback_when_tvm_nd_is_missing(self) -> None:
        import numpy as np

        from scripts.stage2_native_int8_tvm_worker import make_tvm_array

        class FakeRuntime:
            @staticmethod
            def tensor(value: np.ndarray, *, device: object) -> tuple[str, tuple[int, ...], object]:
                return ("runtime.tensor", tuple(value.shape), device)

        class FakeTvm:
            runtime = FakeRuntime()

        result = make_tvm_array(FakeTvm(), np.zeros((1, 2), dtype=np.uint8), "cuda0")

        self.assertEqual(result, ("runtime.tensor", (1, 2), "cuda0"))

    def test_quantize_activation_uint8_records_asymmetric_quant_params(self) -> None:
        import numpy as np

        values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

        quantized, summary = quantize_activation_uint8(values, tensor_name="spatial_features")

        self.assertEqual(quantized.dtype, np.uint8)
        self.assertEqual(quantized.tolist(), [0, 128, 255])
        self.assertEqual(summary["tensor_name"], "spatial_features")
        self.assertEqual(summary["scheme"], "asymmetric_minmax_uint8")
        self.assertAlmostEqual(summary["scale"], 2.0 / 255.0)
        self.assertEqual(summary["zero_point"], 128)
        self.assertEqual(summary["shape"], [3])

    def test_quantize_activation_uint8_handles_constant_tensor(self) -> None:
        import numpy as np

        quantized, summary = quantize_activation_uint8(
            np.full((2, 2), 3.5, dtype=np.float32),
            tensor_name="constant",
        )

        self.assertEqual(quantized.tolist(), [[0, 0], [0, 0]])
        self.assertEqual(summary["scale"], 1.0)
        self.assertEqual(summary["zero_point"], 0)
        self.assertEqual(summary["min"], 3.5)
        self.assertEqual(summary["max"], 3.5)

    def test_quantize_activation_uint8_static_uses_calibration_scale(self) -> None:
        import numpy as np

        quantized, summary = quantize_activation_uint8_static(
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),
            tensor_name="spatial_features",
            scale=0.5,
            zero_point=128,
            source="tensor_quant_params_v2",
        )

        self.assertEqual(quantized.dtype, np.uint8)
        self.assertEqual(quantized.tolist(), [126, 128, 130])
        self.assertEqual(summary["scheme"], "calibration_static_uint8")
        self.assertEqual(summary["source"], "tensor_quant_params_v2")
        self.assertEqual(summary["scale"], 0.5)
        self.assertEqual(summary["zero_point"], 128)

    def test_real_activation_bridge_summarizes_numpy_array(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import summarize_numpy_array

        summary = summarize_numpy_array(
            np.array([[0, 2], [4, 6]], dtype=np.uint8),
            tensor_name="pyramid_level0",
            path=Path("pyramid_level0.npy"),
        )

        self.assertEqual(
            summary,
            {
                "tensor_name": "pyramid_level0",
                "path": "pyramid_level0.npy",
                "shape": [2, 2],
                "dtype": "uint8",
                "size": 4,
                "min": 0.0,
                "max": 6.0,
                "mean": 3.0,
                "std": 2.23606797749979,
            },
        )

    def test_real_activation_bridge_summarizes_empty_numpy_array(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import summarize_numpy_array

        summary = summarize_numpy_array(
            np.empty((0, 7), dtype=np.float32),
            tensor_name="pred_box_tensor",
        )

        self.assertEqual(summary["tensor_name"], "pred_box_tensor")
        self.assertEqual(summary["shape"], [0, 7])
        self.assertEqual(summary["dtype"], "float32")
        self.assertEqual(summary["size"], 0)
        self.assertIsNone(summary["min"])
        self.assertIsNone(summary["max"])
        self.assertIsNone(summary["mean"])
        self.assertIsNone(summary["std"])

    def test_native_int8_full_ap_gate_rejects_partial_or_empty_prediction_report(self) -> None:
        from scripts.stage2_h800_native_int8_real_activation_bridge import (
            build_full_ap_report_gate_fields,
            native_int8_full_ap_gate,
        )

        partial = {
            "processed_samples": 3,
            "pred_nonempty_count": 1,
            "ap30": 0.1,
            "ap50": 0.1,
            "ap70": 0.1,
        }
        empty = {
            "processed_samples": 1789,
            "pred_nonempty_count": 0,
            "ap30": 0.0,
            "ap50": 0.0,
            "ap70": 0.0,
        }
        ok = {
            "processed_samples": 1789,
            "pred_nonempty_count": 25,
            "ap30": 0.1,
            "ap50": 0.1,
            "ap70": 0.1,
        }

        self.assertEqual(native_int8_full_ap_gate(partial)["status"], "blocked")
        self.assertIn("partial", native_int8_full_ap_gate(partial)["reason"])
        self.assertEqual(native_int8_full_ap_gate(empty)["status"], "blocked")
        self.assertIn("empty", native_int8_full_ap_gate(empty)["reason"])
        self.assertEqual(native_int8_full_ap_gate(ok)["status"], "measured_row_allowed")
        smoke_fields = build_full_ap_report_gate_fields(
            partial,
            min_samples=1,
            row_min_samples=1789,
        )
        self.assertEqual(smoke_fields["gate"]["status"], "measured_row_allowed")
        self.assertTrue(smoke_fields["smoke_gate_passed"])
        self.assertFalse(smoke_fields["ap_row_allowed"])
        self.assertIn("full_eval", smoke_fields["ap_row_block_reason"])

    def test_native_int8_full_ap_blocker_records_counts_and_reason(self) -> None:
        from scripts.stage2_h800_native_int8_real_activation_bridge import build_full_ap_blocker

        blocker = build_full_ap_blocker(
            label="s0_024",
            raw_artifact="raw/int8/full_ap_smoke",
            report={
                "processed_samples": 5,
                "pred_nonempty_count": 0,
                "ap30": 0.0,
                "ap50": 0.0,
                "ap70": 0.0,
            },
            reason="empty_predictions",
        )

        self.assertEqual(blocker["schema"], "native_int8_full_ap_gate_blocker_v1")
        self.assertEqual(blocker["label"], "s0_024")
        self.assertEqual(blocker["sample_count"], 5)
        self.assertEqual(blocker["pred_nonempty_count"], 0)
        self.assertEqual(blocker["failure_reason"], "empty_predictions")
        self.assertFalse(blocker["ap_measured"])

    def test_output_dequant_sanity_record_aligns_uint8_range_to_reference_range(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import build_output_dequant_sanity_record

        record = build_output_dequant_sanity_record(
            tensor_name="level0",
            reference=np.array([0.0, 2.0, 4.0], dtype=np.float32),
            tvm_uint8=np.array([10, 20, 30], dtype=np.uint8),
            sample_index=7,
        )

        self.assertEqual(record["tensor_name"], "level0")
        self.assertEqual(record["sample_index"], 7)
        self.assertEqual(record["reference"]["shape"], [3])
        self.assertEqual(record["tvm_uint8"]["min"], 10.0)
        self.assertEqual(record["candidate_dequant"]["scheme"], "minmax_align_uint8_to_reference")
        self.assertAlmostEqual(record["candidate_dequant"]["scale"], 0.2)
        self.assertAlmostEqual(record["candidate_dequant"]["zero_point"], 10.0)
        self.assertAlmostEqual(record["alignment_error"]["mae"], 0.0)
        self.assertAlmostEqual(record["alignment_error"]["rmse"], 0.0)
        self.assertAlmostEqual(record["alignment_error"]["corrcoef"], 1.0)

    def test_output_dequant_sanity_record_uses_actual_dequant_details_when_provided(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import build_output_dequant_sanity_record

        record = build_output_dequant_sanity_record(
            tensor_name="pyramid_level0",
            reference=np.array([0.0, 2.0, 4.0], dtype=np.float32),
            tvm_uint8=np.array([128, 138, 148], dtype=np.uint8),
            sample_index=3,
            candidate_dequant=np.array([0.0, 2.0, 4.0], dtype=np.float32),
            dequant_details={
                "scheme": "tensor_quant_params_v2",
                "source": "observed_reference_range",
                "scale": 0.2,
                "zero_point": 128,
            },
        )

        self.assertEqual(record["candidate_dequant"]["scheme"], "tensor_quant_params_v2")
        self.assertEqual(record["candidate_dequant"]["source"], "observed_reference_range")
        self.assertAlmostEqual(record["candidate_dequant"]["scale"], 0.2)
        self.assertEqual(record["candidate_dequant"]["zero_point"], 128)
        self.assertAlmostEqual(record["candidate_dequant"]["dequant_min"], 0.0)
        self.assertAlmostEqual(record["candidate_dequant"]["dequant_max"], 4.0)
        self.assertAlmostEqual(record["candidate_dequant"]["dynamic_range"], 4.0)
        self.assertAlmostEqual(record["candidate_dequant"]["nonzero_ratio"], 2.0 / 3.0)

    def test_summarize_dequant_sanity_reports_scheme_dynamic_range_and_nonzero_ratio(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import (
            build_output_dequant_sanity_record,
            summarize_dequant_sanity,
        )

        records = [
            build_output_dequant_sanity_record(
                tensor_name="pyramid_level0",
                reference=np.array([0.0, 2.0, 4.0], dtype=np.float32),
                tvm_uint8=np.array([128, 138, 148], dtype=np.uint8),
                sample_index=0,
                candidate_dequant=np.array([0.0, 2.0, 4.0], dtype=np.float32),
                dequant_details={
                    "scheme": "tensor_quant_params_v2",
                    "source": "observed_reference_range",
                    "scale": 0.2,
                    "zero_point": 128,
                },
            ),
            build_output_dequant_sanity_record(
                tensor_name="pyramid_level0",
                reference=np.array([0.0, 1.0, 2.0], dtype=np.float32),
                tvm_uint8=np.array([128, 133, 138], dtype=np.uint8),
                sample_index=1,
                candidate_dequant=np.array([0.0, 1.0, 2.0], dtype=np.float32),
                dequant_details={
                    "scheme": "tensor_quant_params_v2",
                    "source": "observed_reference_range",
                    "scale": 0.2,
                    "zero_point": 128,
                },
            ),
        ]

        summary = summarize_dequant_sanity(records)
        output = summary["outputs"]["pyramid_level0"]

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(output["dequant_scheme"], "tensor_quant_params_v2")
        self.assertEqual(output["dequant_source"], "observed_reference_range")
        self.assertAlmostEqual(output["dequant_dynamic_range_mean"], 3.0)
        self.assertAlmostEqual(output["dequant_nonzero_ratio_mean"], 2.0 / 3.0)
        self.assertAlmostEqual(output["tvm_uint8_dynamic_range_mean"], 15.0)

    def test_dequantize_tvm_output_uses_tensor_quant_params_v2(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import dequantize_tvm_output_uint8

        dequant, details = dequantize_tvm_output_uint8(
            np.array([128, 138], dtype=np.uint8),
            tensor_name="pyramid_level0",
            activation_quant={"scale": 0.01, "zero_point": 0},
            tensor_quant_params={
                "pyramid_level0": {
                    "scale": 0.2,
                    "zero_point": 128,
                    "source": "observed_reference_range",
                }
            },
        )

        self.assertEqual(dequant.dtype, np.float32)
        self.assertEqual(dequant.tolist(), [0.0, 2.0])
        self.assertEqual(details["scheme"], "tensor_quant_params_v2")
        self.assertEqual(details["tensor_name"], "pyramid_level0")
        self.assertAlmostEqual(details["scale"], 0.2)
        self.assertEqual(details["zero_point"], 128)

    def test_calibrated_worker_request_records_tensor_quant_params_source(self) -> None:
        from scripts.stage2_h800_native_int8_real_activation_bridge import (
            attach_tensor_quant_params_to_worker_request,
        )

        request = {"schema": "native_int8_tvm_worker_request_v1", "label": "s0_024"}
        updated = attach_tensor_quant_params_to_worker_request(
            request,
            tensor_quant_params_path="/raw/tensor_quant_params_calibration_v2.json",
            tensor_quant_params_digest="abc123",
        )

        self.assertIs(updated, request)
        self.assertEqual(
            updated["tensor_quant_params_path"],
            "/raw/tensor_quant_params_calibration_v2.json",
        )
        self.assertEqual(updated["tensor_quant_params_digest"], "abc123")
        self.assertEqual(updated["output_dequant_policy"], "per_output_tensor_quant_params_v2")

    def test_op_level_alignment_record_reports_error_and_pass_flag(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import build_op_level_alignment_record

        record = build_op_level_alignment_record(
            tensor_name="/resnet/layer0/layer0.0/conv1/Conv_output_0",
            reference=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            candidate=np.array([0.1, 1.1, 2.1], dtype=np.float32),
            sample_index=3,
            candidate_kind="native_int8_formula_sim",
            candidate_details={"requant": "x//256+128"},
            pass_rel_rmse=0.2,
            pass_corrcoef=0.95,
        )

        self.assertEqual(record["schema"], "native_int8_op_level_alignment_record_v1")
        self.assertEqual(record["tensor_name"], "/resnet/layer0/layer0.0/conv1/Conv_output_0")
        self.assertEqual(record["candidate_kind"], "native_int8_formula_sim")
        self.assertEqual(record["candidate_details"], {"requant": "x//256+128"})
        self.assertEqual(record["reference"]["shape"], [3])
        self.assertAlmostEqual(record["alignment_error"]["mae"], 0.1, places=6)
        self.assertAlmostEqual(record["alignment_error"]["corrcoef"], 1.0, places=6)
        self.assertTrue(record["passed"])

    def test_op_level_alignment_record_rejects_shape_mismatch(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_real_activation_bridge import build_op_level_alignment_record

        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            build_op_level_alignment_record(
                tensor_name="bad",
                reference=np.zeros((1, 2), dtype=np.float32),
                candidate=np.zeros((2, 1), dtype=np.float32),
                sample_index=0,
                candidate_kind="native_int8_formula_sim",
            )

    def test_op_alignment_resolves_onnx_downsample_module_suffix(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import resolve_module_name

        module_names = [
            "resnet.layer0.0.conv1",
            "resnet.layer0.0.downsample.0",
        ]

        self.assertEqual(
            resolve_module_name(
                module_names,
                "/resnet/layer0/layer0.0/downsample/downsample.0/Conv",
            ),
            "resnet.layer0.0.downsample.0",
        )

    def test_op_alignment_resolves_exported_pyramid_backbone_prefix(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import resolve_module_name

        module_names = ["resnet.layer0.0.conv1"]

        self.assertEqual(
            resolve_module_name(
                module_names,
                "/pyramid_backbone/resnet/layer0/layer0.0/conv1/Conv",
            ),
            "resnet.layer0.0.conv1",
        )

    def test_op_alignment_selects_deeper_convs_by_regex(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import select_conv_records_from_op_records

        records = [
            {
                "op_type": "Conv",
                "op_name": "/resnet/layer0/layer0.0/conv1/Conv",
                "input_name": "spatial_features",
            },
            {
                "op_type": "Relu",
                "op_name": "/resnet/layer0/layer0.0/relu/Relu",
            },
            {
                "op_type": "Conv",
                "op_name": "/resnet/layer0/layer0.1/conv1/Conv",
                "input_name": "/resnet/layer0/layer0.0/relu_2/Relu_output_0",
            },
            {
                "op_type": "Conv",
                "op_name": "/resnet/layer1/layer1.0/conv1/Conv",
                "input_name": "pyramid_level0",
            },
        ]

        selected = select_conv_records_from_op_records(
            records,
            limit=10,
            op_name_regexes=[r"layer0\.1/conv1", r"layer1\.0/conv1"],
        )

        self.assertEqual([idx for idx, _ in selected], [1, 2])
        self.assertEqual([item["op_name"] for _, item in selected], [records[2]["op_name"], records[3]["op_name"]])

    def test_native_prefix_simulator_records_relu_zero_point_and_centered_add(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_op_alignment import simulate_native_int8_graph_prefix

        tensors, trace = simulate_native_int8_graph_prefix(
            activation_uint8=np.array([[[[240, 10]]]], dtype=np.uint8),
            op_records=[
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu_2/Relu",
                    "input_name": "spatial_features",
                    "output_name": "branch_a",
                },
                {
                    "op_type": "Add",
                    "op_name": "/resnet/layer0/layer0.0/Add",
                    "input_names": ["branch_a", "spatial_features"],
                    "output_name": "add_out",
                },
            ],
            runtime_weights={},
            weights_by_index={},
            graph_input_zero_point=128,
        )

        np.testing.assert_array_equal(tensors["branch_a"], np.array([[[[240, 128]]]], dtype=np.uint8))
        np.testing.assert_array_equal(tensors["add_out"], np.array([[[[255, 10]]]], dtype=np.uint8))
        self.assertEqual(trace[0]["op_type"], "Relu")
        self.assertEqual(trace[0]["tensor"]["min"], 128.0)
        self.assertEqual(trace[0]["tensor"]["max_fraction"], 0.0)
        self.assertEqual(trace[1]["op_type"], "Add")
        self.assertEqual(trace[1]["tensor"]["max_fraction"], 0.5)

    def test_native_prefix_simulator_centers_intermediate_conv_input_zero_point(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_op_alignment import simulate_native_int8_graph_prefix

        tensors, trace = simulate_native_int8_graph_prefix(
            activation_uint8=np.array([[[[255]]]], dtype=np.uint8),
            op_records=[
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu_2/Relu",
                    "input_name": "spatial_features",
                    "output_name": "relu_out",
                },
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer0/layer0.1/conv1/Conv",
                    "input_name": "relu_out",
                    "output_name": "conv_out",
                    "group": 1,
                    "strides": [1, 1],
                    "pads": [0, 0, 0, 0],
                },
            ],
            runtime_weights={"weight_1": np.array([[[[2]]]], dtype=np.int8)},
            weights_by_index={
                0: {
                    "arg_name": "weight_1",
                    "quantization": {"scale": 1.0, "zero_point": 0},
                }
            },
            graph_input_zero_point=128,
        )

        np.testing.assert_array_equal(tensors["relu_out"], np.array([[[[255]]]], dtype=np.uint8))
        np.testing.assert_array_equal(tensors["conv_out"], np.array([[[[128]]]], dtype=np.uint8))
        self.assertEqual(trace[1]["input_zero_point"], 128)

    def test_scale_aware_requant_uses_input_weight_and_output_scale(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_op_alignment import requantize_int32_scale_aware

        requantized = requantize_int32_scale_aware(
            accumulator=np.array([-20, 0, 20, 40], dtype=np.int32),
            input_scale=0.5,
            weight_scale=0.25,
            output_scale=0.25,
            output_zero_point=128,
        )

        np.testing.assert_array_equal(
            requantized,
            np.array([118, 128, 138, 148], dtype=np.uint8),
        )

    def test_scale_aware_add_rescales_branches_to_common_output_scale(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_op_alignment import scale_aware_add_uint8

        added = scale_aware_add_uint8(
            lhs_uint8=np.array([130, 132], dtype=np.uint8),
            lhs_scale=0.5,
            lhs_zero_point=128,
            rhs_uint8=np.array([126, 124], dtype=np.uint8),
            rhs_scale=0.25,
            rhs_zero_point=128,
            output_scale=0.25,
            output_zero_point=128,
        )

        np.testing.assert_array_equal(
            added,
            np.array([130, 132], dtype=np.uint8),
        )

    def test_scale_aware_prefix_simulator_uses_tensor_quant_params(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_op_alignment import (
            QuantTensor,
            simulate_scale_aware_int8_graph_prefix,
        )

        tensors, trace = simulate_scale_aware_int8_graph_prefix(
            graph_input=QuantTensor(
                values_uint8=np.array([[[[132]]]], dtype=np.uint8),
                scale=0.5,
                zero_point=128,
                tensor_name="spatial_features",
            ),
            op_records=[
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer0/layer0.0/conv1/Conv",
                    "input_name": "spatial_features",
                    "output_name": "conv_out",
                    "group": 1,
                    "strides": [1, 1],
                    "pads": [0, 0, 0, 0],
                },
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu/Relu",
                    "input_name": "conv_out",
                    "output_name": "relu_out",
                },
            ],
            runtime_weights={"weight_1": np.array([[[[2]]]], dtype=np.int8)},
            weights_by_index={
                0: {
                    "arg_name": "weight_1",
                    "quantization": {"scale": 0.25, "zero_point": 0},
                }
            },
            tensor_quant_params={
                "conv_out": {"scale": 0.25, "zero_point": 128},
                "relu_out": {"scale": 0.25, "zero_point": 128},
            },
        )

        np.testing.assert_array_equal(tensors["conv_out"].values_uint8, np.array([[[[132]]]], dtype=np.uint8))
        np.testing.assert_array_equal(tensors["relu_out"].values_uint8, np.array([[[[132]]]], dtype=np.uint8))
        self.assertEqual(trace[0]["scale"], 0.25)
        self.assertEqual(trace[0]["zero_point"], 128)

    def test_weight_plan_by_conv_index_ignores_bias_inputs(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import weight_plan_by_conv_index

        plan = weight_plan_by_conv_index(
            {
                "runtime_arg_plan": [
                    {"role": "weight_input", "arg_name": "weight_1", "dtype": "int8"},
                    {"role": "bias_input", "arg_name": "bias_1", "dtype": "int32"},
                    {"role": "weight_input", "arg_name": "weight_2", "dtype": "int8"},
                ]
            }
        )

        self.assertEqual(plan[0]["arg_name"], "weight_1")
        self.assertEqual(plan[1]["arg_name"], "weight_2")
        self.assertEqual(set(plan), {0, 1})

    def test_full_onnx_route_builder_accepts_tensor_quant_params_for_scale_aware_lowering(self) -> None:
        route = load_full_onnx_route_module()

        signature = inspect.signature(route.build_full_onnx_te_spec)
        self.assertIn("tensor_quant_params", signature.parameters)
        self.assertIn("graph_input_quant_params", signature.parameters)

        self.assertEqual(
            route.tensor_quant_params_for(
                {"conv_out": {"scale": 0.25, "zero_point": 127}},
                "conv_out",
                default_scale=1.0,
                default_zero_point=128,
            ),
            (0.25, 127),
        )
        self.assertEqual(
            route.tensor_quant_params_for(
                {},
                "missing",
                default_scale=0.5,
                default_zero_point=0,
            ),
            (0.5, 0),
        )
        with self.assertRaisesRegex(ValueError, "positive scale"):
            route.tensor_quant_params_for(
                {"bad": {"scale": 0.0, "zero_point": 128}},
                "bad",
                default_scale=1.0,
                default_zero_point=128,
            )

    def test_full_onnx_route_builder_quantizes_conv_bias_for_int32_accumulator(self) -> None:
        import numpy as np

        route = load_full_onnx_route_module()

        signature = inspect.signature(route.conv2d_native_int8)
        self.assertIn("bias", signature.parameters)

        quantized, quantization = route.quantize_bias_int32(
            np.array([0.25, -0.5], dtype=np.float32),
            input_scale=0.5,
            weight_scale=0.25,
        )

        np.testing.assert_array_equal(quantized, np.array([2, -4], dtype=np.int32))
        self.assertEqual(quantization["scheme"], "float_bias_to_int32_accumulator")
        self.assertAlmostEqual(quantization["scale"], 0.125)
        self.assertAlmostEqual(quantization["input_scale"], 0.5)
        self.assertAlmostEqual(quantization["weight_scale"], 0.25)

    def test_reference_ranges_build_tensor_quant_params_without_scale_collapse(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import (
            build_tensor_quant_params_from_reference_ranges,
        )

        calibration = build_tensor_quant_params_from_reference_ranges(
            label="s0_024",
            reference_ranges=[
                {
                    "tensor_name": "/resnet/layer0/layer0.1/Add_output_0",
                    "shape": [1, 24, 256, 256],
                    "min": 0.0,
                    "max": 12.7,
                },
                {
                    "tensor_name": "pyramid_level0",
                    "shape": [1, 24, 256, 256],
                    "min": 0.0,
                    "max": 25.4,
                },
            ],
            graph_input_quant={
                "tensor_name": "spatial_features",
                "scale": 0.05,
                "zero_point": 0,
            },
            route_dir="raw/int8_native_route/s0_024",
            raw_artifact="raw/int8_native_route/s0_024/tensor_reference_ranges_layer0_v1",
        )

        self.assertEqual(calibration["schema"], "native_int8_tensor_quant_params_calibration_v2")
        self.assertEqual(calibration["status"], "ready_for_scale_aware_simulator")
        self.assertEqual(calibration["params"]["spatial_features"]["zero_point"], 0)
        self.assertEqual(calibration["params"]["/resnet/layer0/layer0.1/Add_output_0"]["zero_point"], 128)
        self.assertAlmostEqual(
            calibration["params"]["/resnet/layer0/layer0.1/Add_output_0"]["scale"],
            0.1,
        )
        self.assertAlmostEqual(calibration["params"]["pyramid_level0"]["scale"], 0.2)
        self.assertGreater(
            calibration["params"]["/resnet/layer0/layer0.1/Add_output_0"]["scale"],
            1e-6,
        )
        self.assertEqual(calibration["range_count"], 2)

    def test_prefix_reference_range_targets_stop_at_requested_output(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import prefix_reference_range_targets

        targets = prefix_reference_range_targets(
            [
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer0/layer0.0/conv1/Conv",
                    "output_name": "conv1_out",
                },
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu/Relu",
                    "output_name": "relu_out",
                },
                {
                    "op_type": "Add",
                    "op_name": "/resnet/layer0/layer0.0/Add",
                    "output_name": "add_out",
                },
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.2/relu_2/Relu",
                    "output_name": "pyramid_level0",
                },
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer1/layer1.0/conv1/Conv",
                    "output_name": "layer1_out",
                },
            ],
            stop_output_names=["pyramid_level0"],
        )

        self.assertEqual(
            [target["output_name"] for target in targets],
            ["conv1_out", "relu_out", "add_out", "pyramid_level0"],
        )
        self.assertEqual(targets[-1]["stop_matched"], True)
        self.assertTrue(all(target["requires_reference_range"] for target in targets))

    def test_reference_range_capture_plan_maps_conv_relu_and_blocks_add(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import reference_range_capture_plan

        plan = reference_range_capture_plan(
            targets=[
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer0/layer0.0/conv1/Conv",
                    "output_name": "conv1_out",
                },
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu/Relu",
                    "output_name": "relu_out",
                },
                {
                    "op_type": "Add",
                    "op_name": "/resnet/layer0/layer0.0/Add",
                    "output_name": "add_out",
                },
            ],
            module_names=["resnet.layer0.0.conv1", "resnet.layer0.0.relu"],
        )

        self.assertEqual(plan["summary"]["hook_ready"], 2)
        self.assertEqual(plan["summary"]["blocked"], 1)
        self.assertEqual(plan["items"][0]["capture_status"], "hook_ready")
        self.assertEqual(plan["items"][0]["source_module_name"], "resnet.layer0.0.conv1")
        self.assertEqual(plan["items"][1]["capture_status"], "hook_ready")
        self.assertEqual(plan["items"][1]["source_module_name"], "resnet.layer0.0.relu")
        self.assertEqual(plan["items"][2]["capture_status"], "blocked")
        self.assertEqual(
            plan["items"][2]["failure_reason"],
            "add_output_requires_following_relu_pre_hook_capture",
        )

    def test_reference_range_capture_plan_maps_fused_conv_outputs_to_batchnorm(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import reference_range_capture_plan

        plan = reference_range_capture_plan(
            targets=[
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer1/layer1.1/conv2/Conv",
                    "output_name": "conv2_out",
                },
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer0/layer0.0/downsample/downsample.0/Conv",
                    "output_name": "downsample_out",
                },
                {
                    "op_type": "Conv",
                    "op_name": "/resnet/layer2/layer2.0/conv1/Conv",
                    "output_name": "conv1_without_bn_out",
                },
            ],
            module_names=[
                "resnet.layer1.1.conv2",
                "resnet.layer1.1.bn2",
                "resnet.layer0.0.downsample.0",
                "resnet.layer0.0.downsample.1",
                "resnet.layer2.0.conv1",
            ],
        )

        self.assertEqual(plan["summary"]["hook_ready"], 3)
        self.assertEqual(plan["summary"]["blocked"], 0)
        self.assertEqual(plan["items"][0]["source_module_name"], "resnet.layer1.1.bn2")
        self.assertEqual(plan["items"][1]["source_module_name"], "resnet.layer0.0.downsample.1")
        self.assertEqual(plan["items"][2]["source_module_name"], "resnet.layer2.0.conv1")

    def test_reference_range_capture_plan_maps_reused_relu_and_add_by_call_index(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import reference_range_capture_plan

        plan = reference_range_capture_plan(
            targets=[
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu_1/Relu",
                    "output_name": "relu_1_out",
                },
                {
                    "op_type": "Add",
                    "op_name": "/resnet/layer0/layer0.0/Add",
                    "output_name": "add_out",
                },
                {
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu_2/Relu",
                    "output_name": "relu_2_out",
                },
            ],
            module_names=["resnet.layer0.0.relu"],
        )

        self.assertEqual(plan["summary"]["hook_ready"], 3)
        self.assertEqual(plan["summary"]["blocked"], 0)
        self.assertEqual(plan["items"][0]["source_module_name"], "resnet.layer0.0.relu")
        self.assertEqual(plan["items"][0]["source_call_index"], 1)
        self.assertEqual(plan["items"][0]["source_capture"], "module_forward_hook_call_index")
        self.assertEqual(plan["items"][1]["source_module_name"], "resnet.layer0.0.relu")
        self.assertEqual(plan["items"][1]["source_call_index"], 2)
        self.assertEqual(plan["items"][1]["source_capture"], "module_forward_pre_hook_call_index")
        self.assertEqual(plan["items"][2]["source_module_name"], "resnet.layer0.0.relu")
        self.assertEqual(plan["items"][2]["source_call_index"], 2)
        self.assertEqual(plan["items"][2]["source_capture"], "module_forward_hook_call_index")

    def test_build_reference_range_capture_payload_summarizes_ranges(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_op_alignment import (
            build_reference_range_capture_payload,
        )

        plan = {
            "schema": "native_int8_reference_range_capture_plan_v1",
            "summary": {"total": 1, "hook_ready": 1, "blocked": 0},
            "items": [
                {
                    "op_index": 7,
                    "op_type": "Relu",
                    "op_name": "/resnet/layer0/layer0.0/relu_2/Relu",
                    "output_name": "relu_2_out",
                    "capture_status": "hook_ready",
                    "source_capture": "module_forward_hook_call_index",
                    "source_module_name": "resnet.layer0.0.relu",
                    "source_call_index": 2,
                }
            ],
        }

        payload = build_reference_range_capture_payload(
            label="s0_024",
            raw_dir="raw/int8_native_route/s0_024",
            route_dir="raw/int8_native_route/route",
            plan=plan,
            captured_tensors={
                "relu_2_out": np.array([[-1.0, 0.0, 2.0]], dtype=np.float32),
            },
            sample_count=1,
        )

        self.assertEqual(payload["schema"], "native_int8_reference_ranges_v1")
        self.assertEqual(payload["status"], "ready_for_calibration")
        self.assertEqual(payload["range_count"], 1)
        item = payload["reference_ranges"][0]
        self.assertEqual(item["tensor_name"], "relu_2_out")
        self.assertEqual(item["source_module_name"], "resnet.layer0.0.relu")
        self.assertEqual(item["source_call_index"], 2)
        self.assertEqual(item["min"], -1.0)
        self.assertEqual(item["max"], 2.0)
        self.assertAlmostEqual(item["recommended_uint8_scale"], 2.0 / 127.0)
        self.assertEqual(item["recommended_zero_point"], 128)

    def test_build_pytorch_module_inventory_payload_records_module_classes(self) -> None:
        from scripts.stage2_h800_native_int8_op_alignment import (
            build_pytorch_module_inventory_payload,
        )

        class DummyConv:
            pass

        payload = build_pytorch_module_inventory_payload(
            label="s0_024",
            raw_dir="raw/int8_native_route/s0_024",
            route_dir="raw/int8_native_route/route",
            module_map={"": object(), "resnet.layer0.0.conv1": DummyConv()},
            scope="pyramid_backbone",
        )

        self.assertEqual(payload["schema"], "native_int8_pytorch_module_inventory_v1")
        self.assertEqual(payload["module_count"], 2)
        self.assertEqual(payload["module_names"], ["", "resnet.layer0.0.conv1"])
        self.assertEqual(payload["modules"][1]["module_name"], "resnet.layer0.0.conv1")
        self.assertEqual(payload["modules"][1]["class_name"], "DummyConv")
        self.assertEqual(payload["scope"], "pyramid_backbone")

    def test_checkpoint_weight_audit_record_passes_for_matching_weights(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_checkpoint_weight_audit import build_weight_alignment_record

        weight = np.array([[[[0.0]], [[1.0]]], [[[2.0]], [[3.0]]]], dtype=np.float32)

        record = build_weight_alignment_record(
            op_name="/resnet/layer0/layer0.0/conv1/Conv",
            initializer_name="onnx::Conv_474",
            module_name="resnet.layer0.0.conv1",
            initializer_weight=weight,
            module_weight=weight.copy(),
        )

        self.assertTrue(record["passed"])
        self.assertAlmostEqual(record["alignment_error"]["rmse"], 0.0)
        self.assertAlmostEqual(record["alignment_error"]["corrcoef"], 1.0)
        self.assertEqual(record["initializer"]["shape"], [2, 2, 1, 1])

    def test_checkpoint_weight_audit_record_fails_for_uncorrelated_weights(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_checkpoint_weight_audit import build_weight_alignment_record

        record = build_weight_alignment_record(
            op_name="/resnet/layer0/layer0.0/conv1/Conv",
            initializer_name="onnx::Conv_474",
            module_name="resnet.layer0.0.conv1",
            initializer_weight=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            module_weight=np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32),
        )

        self.assertFalse(record["passed"])
        self.assertLess(record["alignment_error"]["corrcoef"], 0.0)

    def test_checkpoint_weight_audit_fuses_conv_bn_weight_per_output_channel(self) -> None:
        import numpy as np

        from scripts.stage2_h800_native_int8_checkpoint_weight_audit import fuse_conv_bn_weight

        conv = np.ones((2, 1, 1, 1), dtype=np.float32)
        fused = fuse_conv_bn_weight(
            conv_weight=conv,
            bn_weight=np.array([2.0, 3.0], dtype=np.float32),
            bn_running_var=np.array([3.0, 8.0], dtype=np.float32),
            bn_eps=1.0,
        )

        self.assertEqual(fused.shape, (2, 1, 1, 1))
        self.assertAlmostEqual(float(fused[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(fused[1, 0, 0, 0]), 1.0)

    def test_checkpoint_weight_audit_maps_conv_to_following_bn(self) -> None:
        from scripts.stage2_h800_native_int8_checkpoint_weight_audit import batch_norm_name_for_conv_module

        self.assertEqual(batch_norm_name_for_conv_module("resnet.layer0.0.conv1"), "resnet.layer0.0.bn1")
        self.assertEqual(
            batch_norm_name_for_conv_module("resnet.layer0.0.downsample.0"),
            "resnet.layer0.0.downsample.1",
        )

    def test_multiscale_export_output_names_are_stable(self) -> None:
        from scripts.stage2_h800_export_checkpoint_multiscale_onnx import multiscale_output_names

        self.assertEqual(
            multiscale_output_names(3),
            ["pyramid_level0", "pyramid_level1", "pyramid_level2"],
        )


if __name__ == "__main__":
    unittest.main()
