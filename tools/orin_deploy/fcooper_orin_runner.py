#!/usr/bin/env python3
"""Fail-closed F-Cooper TensorRT and native measurement runner for Orin."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


INPUT_NAME = "spatial_features"
OUTPUT_NAME = "encoded_features"
INPUT_SHAPE = [5, 64, 512, 512]
SCOPE = "post_scatter_backbone_shrinker"
PROTOCOL = {"warmup": 20, "iters": 300, "repeat": 5, "sample_count": 1500}
TOLERANCES = {
    "fp32": {"cosine_min": 0.99999, "nrmse_max": 0.01, "mae_max": 0.001, "max_abs_max": 0.05},
    "fp16": {"cosine_min": 0.999, "nrmse_max": 0.05, "mae_max": 0.05, "max_abs_max": 0.5},
}


class RunnerError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path | str, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path | str) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"invalid manifest: {error}") from error
    if manifest.get("status") != "ready" or not isinstance(manifest.get("arms"), dict):
        raise RunnerError("manifest is not ready")
    return manifest


def get_arm(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        arm = manifest["arms"][name]
    except KeyError as error:
        raise RunnerError(f"unknown frozen arm: {name}") from error
    return arm


def path_is_within(path: Path | str, root: Path | str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def orin_platform_probe() -> tuple[str, str, str]:
    model_path = Path("/proc/device-tree/model")
    try:
        board = model_path.read_text(encoding="utf-8", errors="replace").replace("\x00", " ").strip()
    except OSError:
        board = ""
    return platform.system(), platform.machine(), f"{board} {platform.platform()}".strip()


def require_orin(probe: Callable[[], tuple[str, str, str]] = orin_platform_probe) -> tuple[str, str, str]:
    system, machine, description = probe()
    if system != "Linux" or machine.lower() not in {"aarch64", "arm64"} or "orin" not in description.lower():
        raise RunnerError("TensorRT build requires Linux aarch64/Orin")
    return system, machine, description


def validate_builder_policy(arm: dict[str, Any]) -> None:
    if (
        arm.get("runtime") != "trt"
        or arm.get("builder_policy") != "trt85_default"
        or arm.get("builder_optimization_level") is not None
    ):
        raise RunnerError(
            "F-Cooper TRT arm must use the Pyramid TRT 8.5 default builder"
        )


def validate_build_preflight(manifest: dict[str, Any], arm_name: str, paths: dict[str, Path], probe: Callable[[], tuple[str, str, str]], onnx_contract: dict[str, Any]) -> dict[str, Any]:
    arm = get_arm(manifest, arm_name)
    if arm_name == "original_default" or arm.get("runtime") != "trt":
        raise RunnerError("build refuses original_default/native arm")
    validate_builder_policy(arm)
    require_orin(probe)
    for key in ("engine", "timing_cache"):
        if Path(paths[key]).exists():
            raise RunnerError(f"pre-existing {key} output is refused")
    if paths.get("artifact_root") is not None:
        artifact_root = Path(paths["artifact_root"]).resolve()
        for key in ("engine", "timing_cache", "inspector_json", "artifact_json"):
            if key in paths and not path_is_within(paths[key], artifact_root):
                raise RunnerError(f"{key} must be under --artifact-root")
    inputs, outputs = onnx_contract.get("inputs", []), onnx_contract.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        raise RunnerError("ONNX must expose exactly one input and one output")
    input_spec, output_spec = inputs[0], outputs[0]
    expected_shape = arm.get("input_shape", manifest.get("input_shape", INPUT_SHAPE))
    if input_spec.get("name") != arm.get("input_name", INPUT_NAME) or list(input_spec.get("shape", [])) != expected_shape or input_spec.get("dtype", "float32") != "float32":
        raise RunnerError("ONNX input contract mismatch")
    if output_spec.get("name") != arm.get("output_name", OUTPUT_NAME) or output_spec.get("dtype") != arm.get("output_dtype", "float32"):
        raise RunnerError("ONNX output contract mismatch")
    return arm


def is_trt_float32(dtype: Any, trt: Any = None) -> bool:
    if trt is not None and hasattr(trt, "float32"):
        return dtype == trt.float32
    return str(dtype).upper() in {"DATATYPE.FLOAT", "FLOAT", "KFLOAT", "FLOAT32"}


def validate_trt_network_contract(arm: dict[str, Any], network: Any, trt: Any = None) -> None:
    if getattr(network, "num_inputs", 0) != 1 or getattr(network, "num_outputs", 0) != 1:
        raise RunnerError("TensorRT network must expose exactly one input and one output")
    input_tensor, output_tensor = network.get_input(0), network.get_output(0)
    input_shape, output_shape = tuple(input_tensor.shape), tuple(output_tensor.shape)
    expected_input, expected_output = tuple(arm.get("input_shape", INPUT_SHAPE)), tuple(arm["output_shape"])
    if input_tensor.name != arm.get("input_name", INPUT_NAME) or input_shape != expected_input or not is_trt_float32(input_tensor.dtype, trt):
        raise RunnerError("TensorRT input contract mismatch")
    if output_tensor.name != arm.get("output_name", OUTPUT_NAME) or output_shape != expected_output or any(not isinstance(dim, int) or dim <= 0 for dim in output_shape) or not is_trt_float32(output_tensor.dtype, trt):
        raise RunnerError("TensorRT output contract mismatch")


def _onnx_contract(onnx_path: Path) -> dict[str, Any]:
    try:
        import onnx  # lazy: not installed on the validation server
    except ImportError as error:
        raise RunnerError("build requires the onnx package on Orin") from error
    model = onnx.load(str(onnx_path))
    def spec(value: Any) -> dict[str, Any]:
        dims = value.type.tensor_type.shape.dim
        shape = [dimension.dim_value if dimension.dim_value > 0 else None for dimension in dims]
        dtype = "float32" if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT else str(value.type.tensor_type.elem_type)
        return {"name": value.name, "shape": shape, "dtype": dtype}
    return {"inputs": [spec(value) for value in model.graph.input], "outputs": [spec(value) for value in model.graph.output]}


def _layer_precision_evidence(layer: dict[str, Any]) -> str:
    values = [
        str(layer.get(key, ""))
        for key in ("precision", "compute_precision", "output_type", "input_type")
    ]
    for group in ("Inputs", "Outputs"):
        tensors = layer.get(group, [])
        if isinstance(tensors, list):
            values.extend(
                str(tensor.get("Format/Datatype", ""))
                for tensor in tensors
                if isinstance(tensor, dict)
            )
    return " ".join(values).upper()


def validate_schedule_inspector(inspector: dict[str, Any]) -> None:
    unsupported = inspector.get("unsupported_fields", [])
    if unsupported:
        raise RunnerError("strict FP32 inspection unsupported: " + ", ".join(map(str, unsupported)))
    layers = inspector.get("layers", [])
    if not layers:
        raise RunnerError("strict FP32 inspector has no layer evidence")
    for layer in layers:
        evidence = _layer_precision_evidence(layer)
        name = layer.get("name", layer.get("Name", "<unnamed>"))
        if any(
            token in evidence
            for token in ("FP16", "INT8", "BF16", "TF32", "HALF", "KHALF")
        ):
            raise RunnerError(
                f"strict FP32 inspector found reduced precision: {name}"
            )
        if "FP32" not in evidence and "FLOAT" not in evidence:
            raise RunnerError(
                f"strict FP32 inspector lacks compute precision: {name}"
            )


def timing_cache_attached(result: Any) -> bool:
    return result is not False


def is_trt85_version(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("8.5.")


def validate_build_receipt(
    receipt: dict[str, Any],
    arm_name: str,
    manifest_sha: str,
    arm: dict[str, Any],
    engine_sha: str,
) -> dict[str, Any]:
    expected_arguments = {
        "builder_policy": "trt85_default",
        "builder_optimization_level": None,
        "q_mode": arm["q_mode"],
    }
    if (
        receipt.get("status") != "built"
        or receipt.get("arm") != arm_name
        or receipt.get("manifest_sha256") != manifest_sha
        or receipt.get("onnx_sha256") != arm["onnx_sha256"]
        or receipt.get("config_sha256") != arm["config_sha256"]
        or receipt.get("checkpoint_sha256") != arm["checkpoint_sha256"]
        or receipt.get("engine_sha256") != engine_sha
        or not is_trt85_version(receipt.get("tensorrt_version"))
        or receipt.get("build_arguments") != expected_arguments
        or receipt.get("h800_engine_or_cache_read") is not False
        or receipt.get("calibration_cache_read") is not False
    ):
        raise RunnerError("build receipt identity mismatch")
    platform_value = receipt.get("platform")
    if (
        not isinstance(platform_value, list)
        or len(platform_value) < 3
        or platform_value[0] != "Linux"
        or str(platform_value[1]).lower() not in {"aarch64", "arm64"}
        or "orin" not in " ".join(map(str, platform_value[2:])).lower()
    ):
        raise RunnerError("build receipt platform mismatch")
    return receipt


def commit_build_artifacts(paths: dict[str, Path], engine_bytes: bytes, cache_bytes: bytes, inspector: dict[str, Any], receipt: dict[str, Any], strict_fp32: bool) -> None:
    if strict_fp32:
        validate_schedule_inspector(inspector)
    temporary: list[Path] = []
    try:
        content = {"engine": engine_bytes, "timing_cache": cache_bytes, "inspector_json": json.dumps(inspector, sort_keys=True).encode(), "artifact_json": json.dumps(receipt, sort_keys=True).encode()}
        for key, data in content.items():
            target = Path(paths[key]); target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".tmp")
            temp.write_bytes(data); temporary.append(temp)
        for key in content:
            temporary.pop(0).replace(paths[key])
    except Exception:
        for path in temporary:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _inspector_payload(engine: Any, trt: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"layers": [], "unsupported_fields": []}
    if not hasattr(engine, "create_engine_inspector"):
        payload["unsupported_fields"].append("create_engine_inspector")
        return payload
    inspector = engine.create_engine_inspector()
    if inspector is None or not hasattr(inspector, "get_layer_information"):
        payload["unsupported_fields"].append("get_layer_information")
        return payload
    fmt = getattr(getattr(trt, "LayerInformationFormat", None), "JSON", None)
    if fmt is None:
        payload["unsupported_fields"].append("LayerInformationFormat.JSON")
        return payload
    for index in range(getattr(engine, "num_layers", 0)):
        try:
            info = inspector.get_layer_information(index, fmt)
            payload["layers"].append(json.loads(info) if isinstance(info, str) and info.startswith("{") else {"name": str(index), "information": str(info)})
        except Exception as error:  # TensorRT version-dependent surface
            payload["unsupported_fields"].append(f"layer_{index}:{type(error).__name__}")
    return payload


def build_engine(manifest: dict[str, Any], arm_name: str, paths: dict[str, Path], probe: Callable[[], tuple[str, str, str]] = orin_platform_probe) -> dict[str, Any]:
    arm = get_arm(manifest, arm_name)
    onnx_path = Path(arm["onnx_path"])
    if not onnx_path.is_file() or sha256_file(onnx_path) != arm.get("onnx_sha256"):
        raise RunnerError("ONNX identity mismatch")
    validate_build_preflight(manifest, arm_name, paths, probe, _onnx_contract(onnx_path))
    try:
        import tensorrt as trt  # lazy
    except ImportError as error:
        raise RunnerError("build requires TensorRT on Orin") from error
    if not is_trt85_version(getattr(trt, "__version__", None)):
        raise RunnerError("trt85_default requires TensorRT 8.5.x")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RunnerError("TensorRT ONNX parse failed: " + "; ".join(errors))
    validate_trt_network_contract(arm, network, trt)
    config = builder.create_builder_config()
    validate_builder_policy(arm)
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    for flag_name in ("TF32", "FP16", "INT8", "BF16"):
        flag = getattr(trt.BuilderFlag, flag_name, None)
        if flag is not None and hasattr(config, "clear_flag"):
            config.clear_flag(flag)
    if arm["q_mode"] == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    if arm.get("strict_fp32") and not hasattr(config, "clear_flag"):
        raise RunnerError("strict FP32 requires TensorRT clear_flag support")
    if arm.get("strict_fp32"):
        verbosity = getattr(getattr(trt, "ProfilingVerbosity", None), "DETAILED", None)
        if verbosity is None or not hasattr(config, "profiling_verbosity"):
            raise RunnerError("strict FP32 requires detailed TensorRT inspector verbosity")
        config.profiling_verbosity = verbosity
    empty_cache = config.create_timing_cache(b"")
    if empty_cache is None:
        raise RunnerError("could not create an empty local timing cache")
    if not hasattr(config, "set_timing_cache"):
        raise RunnerError("TensorRT API lacks set_timing_cache")
    if not timing_cache_attached(config.set_timing_cache(empty_cache, False)):
        raise RunnerError("TensorRT rejected the empty local timing cache")
    serialized = builder.build_serialized_network(network, config) if hasattr(builder, "build_serialized_network") else None
    if serialized is None:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RunnerError("TensorRT engine build failed")
        serialized = engine.serialize()
    else:
        engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    cache = config.get_timing_cache() if hasattr(config, "get_timing_cache") else None
    if cache is None or not hasattr(cache, "serialize"):
        raise RunnerError("TensorRT did not expose locally generated timing cache")
    inspector = _inspector_payload(engine, trt)
    engine_bytes, cache_bytes = bytes(serialized), bytes(cache.serialize())
    receipt = {"status": "built", "scope": SCOPE, "arm": arm_name, "manifest_sha256": manifest.get("_manifest_sha256"), "engine": str(Path(paths["engine"]).resolve()), "timing_cache": str(Path(paths["timing_cache"]).resolve()), "engine_sha256": hashlib.sha256(engine_bytes).hexdigest(), "timing_cache_sha256": hashlib.sha256(cache_bytes).hexdigest(), "onnx_sha256": arm["onnx_sha256"], "config_sha256": arm["config_sha256"], "config_path": arm.get("config_path"), "checkpoint_sha256": arm["checkpoint_sha256"], "tensorrt_version": getattr(trt, "__version__", "unknown"), "platform": list(require_orin(probe)), "build_command": list(sys.argv), "build_arguments": {"builder_policy": arm["builder_policy"], "builder_optimization_level": None, "q_mode": arm["q_mode"]}, "build_log": "TensorRT 8.5 default builder; TensorRT Python logger emitted to the invoking process; no external cache was read.", "calibration_cache_read": False, "h800_engine_or_cache_read": False}
    commit_build_artifacts(paths, engine_bytes, cache_bytes, inspector, receipt, bool(arm.get("strict_fp32")))
    return receipt


def _stats(value: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(value)
    return {"shape": list(flat.shape), "dtype": str(flat.dtype), "finite": bool(np.isfinite(flat).all()), "min": float(np.min(flat)) if flat.size else None, "max": float(np.max(flat)) if flat.size else None, "zero_ratio": float(np.count_nonzero(flat == 0) / flat.size) if flat.size else None}


def _heldout_identity_path(inputs_path: Path | str) -> Path:
    return Path(inputs_path).with_suffix(".heldout.json")


def _load_heldout_identity(inputs_path: Path | str) -> dict[str, Any]:
    path = _heldout_identity_path(inputs_path)
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"missing or invalid held-out provenance: {path}") from error
    required = ("opv2v_test_manifest_sha256", "capture_provenance_sha256", "input_sha256")
    if any(not identity.get(key) for key in required) or identity["input_sha256"] != sha256_file(inputs_path):
        raise RunnerError("held-out provenance identity mismatch")
    return identity


def validate_heldout_provenance(manifest: dict[str, Any], inputs_path: Path | str) -> dict[str, Any]:
    identity = _load_heldout_identity(inputs_path)
    try:
        frozen = manifest["global_identities"]["contracts/opv2v_test_manifest_fresh.txt"]["sha256"]
    except KeyError as error:
        raise RunnerError("canonical manifest lacks frozen OPV2V identity") from error
    if identity["opv2v_test_manifest_sha256"] != frozen:
        raise RunnerError("OPV2V test manifest SHA mismatch")
    return identity


def measurement_identity(manifest: dict[str, Any], inputs_path: Path | str) -> dict[str, Any]:
    heldout = validate_heldout_provenance(manifest, inputs_path)
    return {"input_sha256": heldout["input_sha256"], "opv2v_test_manifest_sha256": heldout["opv2v_test_manifest_sha256"], "capture_provenance_sha256": heldout["capture_provenance_sha256"]}


def _load_inputs(path: Path | str, require_provenance: bool = False) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 5 or list(values.shape[1:]) != INPUT_SHAPE:
        raise RunnerError("inputs must be float32 [N,5,64,512,512]")
    if values.shape[0] <= 0:
        raise RunnerError("held-out input set is empty")
    if require_provenance:
        _load_heldout_identity(path)
    return values


def _identity_path(npz_path: Path | str) -> Path:
    return Path(npz_path).with_suffix(".identity.json")


def _write_identity(npz_path: Path | str, identity: dict[str, Any]) -> None:
    write_json(_identity_path(npz_path), identity)


def _load_identity(npz_path: Path | str) -> dict[str, Any]:
    path = _identity_path(npz_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"missing or invalid output identity: {path}") from error


def _native_encoder(arm: dict[str, Any], input_array: np.ndarray) -> np.ndarray:
    try:
        import torch
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.tools import train_utils
    except ImportError as error:
        raise RunnerError("reference/native measurement requires OpenCOOD and PyTorch on Orin") from error
    config = load_yaml(arm["config_path"])
    model = train_utils.create_model(config).cuda().eval()
    state = torch.load(arm["checkpoint_path"], map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    with torch.no_grad():
        tensor = torch.from_numpy(input_array).cuda(non_blocking=True)
        result = _run_native_scope(model, tensor)
    return result.detach().cpu().numpy()


def _run_native_scope(model: Any, tensor: Any) -> Any:
    result = model.backbone_m1({"spatial_features": tensor})
    if not isinstance(result, dict) or "spatial_features_2d" not in result:
        raise RunnerError("F-Cooper backbone did not return spatial_features_2d")
    return model.shrinker_m1(result["spatial_features_2d"])


def run_reference(manifest: dict[str, Any], arm_name: str, inputs_path: Path, output_npz: Path, report_path: Path, manifest_sha256: str | None = None) -> dict[str, Any]:
    arm, inputs = get_arm(manifest, arm_name), _load_inputs(inputs_path, require_provenance=True)
    heldout = validate_heldout_provenance(manifest, inputs_path)
    outputs = {f"item_{index}": _native_encoder(arm, item) for index, item in enumerate(inputs)}
    np.savez(output_npz, **outputs)
    _write_identity(output_npz, {"kind": "reference", "manifest_sha256": manifest_sha256, "input_sha256": sha256_file(inputs_path), "opv2v_test_manifest_sha256": heldout["opv2v_test_manifest_sha256"], "capture_provenance_sha256": heldout["capture_provenance_sha256"], "checkpoint_sha256": arm["checkpoint_sha256"], "config_sha256": arm["config_sha256"], "onnx_sha256": arm.get("onnx_sha256")})
    report = {"status": "complete", "kind": "reference", "arm": arm_name, "scope": SCOPE, "input_sha256": sha256_file(inputs_path), "output_sha256": sha256_file(output_npz), "checkpoint_sha256": arm["checkpoint_sha256"], "config_sha256": arm["config_sha256"], "outputs": {name: _stats(value) for name, value in outputs.items()}, "device": "cuda", "host": platform.platform()}
    write_json(report_path, report)
    return report


def _trt_outputs(engine_path: Path, inputs: np.ndarray, arm: dict[str, Any]) -> dict[str, np.ndarray]:
    try:
        import torch
        import tensorrt as trt
    except ImportError as error:
        raise RunnerError("numerical requires PyTorch CUDA and TensorRT on Orin") from error
    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RunnerError("invalid TensorRT engine")
    context = engine.create_execution_context()
    if engine.num_bindings != 2:
        raise RunnerError("engine must have exactly one input and one output binding")
    input_index = next((i for i in range(engine.num_bindings) if engine.binding_is_input(i)), None)
    output_index = next((i for i in range(engine.num_bindings) if not engine.binding_is_input(i)), None)
    if input_index is None or output_index is None or engine.get_binding_name(input_index) != arm.get("input_name", INPUT_NAME) or engine.get_binding_name(output_index) != arm.get("output_name", OUTPUT_NAME):
        raise RunnerError("engine binding identity mismatch")
    expected = tuple(arm.get("input_shape", INPUT_SHAPE))
    if tuple(engine.get_binding_shape(input_index)) != expected:
        raise RunnerError("engine static binding shape mismatch")
    output_shape = tuple(engine.get_binding_shape(output_index))
    if output_shape != tuple(arm["output_shape"]) or any(dim <= 0 for dim in output_shape):
        raise RunnerError("engine output binding contract mismatch")
    outputs = {}
    for index, item in enumerate(inputs):
        input_tensor = torch.from_numpy(item).cuda()
        output_tensor = torch.empty(output_shape, device="cuda", dtype=torch.float32)
        bindings = [0, 0]
        bindings[input_index], bindings[output_index] = input_tensor.data_ptr(), output_tensor.data_ptr()
        if not context.execute_v2(bindings):
            raise RunnerError("TensorRT execute_v2 failed")
        outputs[f"item_{index}"] = output_tensor.detach().cpu().numpy()
    return outputs


def run_numerical(manifest: dict[str, Any], arm_name: str, engine_path: Path, build_receipt_path: Path, inputs_path: Path, output_npz: Path, report_path: Path, manifest_sha256: str | None = None) -> dict[str, Any]:
    arm, inputs = get_arm(manifest, arm_name), _load_inputs(inputs_path, require_provenance=True)
    heldout = validate_heldout_provenance(manifest, inputs_path)
    if arm.get("runtime") != "trt" or not engine_path.is_file():
        raise RunnerError("numerical requires a local TRT arm and engine")
    if not build_receipt_path.is_file():
        raise RunnerError("missing build receipt")
    receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    validate_build_receipt(
        receipt,
        arm_name,
        manifest_sha256 or "",
        arm,
        sha256_file(engine_path),
    )
    outputs = _trt_outputs(engine_path, inputs, arm)
    np.savez(output_npz, **outputs)
    _write_identity(output_npz, {"kind": "numerical", "manifest_sha256": manifest_sha256, "input_sha256": sha256_file(inputs_path), "opv2v_test_manifest_sha256": heldout["opv2v_test_manifest_sha256"], "capture_provenance_sha256": heldout["capture_provenance_sha256"], "checkpoint_sha256": arm["checkpoint_sha256"], "config_sha256": arm["config_sha256"], "onnx_sha256": arm["onnx_sha256"], "engine_sha256": sha256_file(engine_path)})
    report = {"status": "complete", "kind": "numerical", "arm": arm_name, "scope": SCOPE, "input_sha256": sha256_file(inputs_path), "output_sha256": sha256_file(output_npz), "engine_sha256": sha256_file(engine_path), "onnx_sha256": arm["onnx_sha256"], "checkpoint_sha256": arm["checkpoint_sha256"], "outputs": {name: _stats(value) for name, value in outputs.items()}, "device": "cuda", "host": platform.platform()}
    write_json(report_path, report)
    return report


def numeric_gate(reference_npz: Path | str, actual_npz: Path | str, output_json: Path | str, arm: dict[str, Any]) -> dict[str, Any]:
    try:
        reference, actual = np.load(reference_npz, allow_pickle=False), np.load(actual_npz, allow_pickle=False)
        reference_identity, actual_identity = _load_identity(reference_npz), _load_identity(actual_npz)
        for key in ("manifest_sha256", "input_sha256", "opv2v_test_manifest_sha256", "capture_provenance_sha256", "checkpoint_sha256", "config_sha256", "onnx_sha256"):
            if reference_identity.get(key) != actual_identity.get(key) or reference_identity.get(key) is None:
                raise RunnerError(f"identity mismatch for {key}")
        if not actual_identity.get("engine_sha256"):
            raise RunnerError("identity mismatch for engine_sha256")
        if actual_identity["checkpoint_sha256"] != arm.get("checkpoint_sha256") or actual_identity["onnx_sha256"] != arm.get("onnx_sha256"):
            raise RunnerError("identity mismatch against frozen arm")
        if set(reference.files) != set(actual.files) or not reference.files:
            raise RunnerError("missing arrays or held-out identity mismatch")
        metrics: dict[str, Any] = {}
        for name in sorted(reference.files):
            expected, observed = reference[name], actual[name]
            if expected.shape != observed.shape:
                raise RunnerError(f"shape mismatch for {name}")
            if not np.isfinite(expected).all() or not np.isfinite(observed).all():
                raise RunnerError(f"nonfinite values for {name}")
            first, second = expected.astype(np.float64).ravel(), observed.astype(np.float64).ravel()
            norm = float(np.linalg.norm(first) * np.linalg.norm(second))
            if norm == 0:
                raise RunnerError(f"degenerate cosine for {name}")
            difference = second - first
            metrics[name] = {"cosine": float(np.dot(first, second) / norm), "nRMSE": float(np.sqrt(np.mean(difference ** 2)) / max(float(np.sqrt(np.mean(first ** 2))), np.finfo(np.float64).eps)), "MAE": float(np.mean(np.abs(difference))), "max_abs": float(np.max(np.abs(difference))), "finite_ratio": 1.0, "shape": list(expected.shape), "reference_dtype": str(expected.dtype), "actual_dtype": str(observed.dtype), "reference_min": float(np.min(expected)), "reference_max": float(np.max(expected)), "reference_zero_ratio": float(np.mean(expected == 0)), "actual_min": float(np.min(observed)), "actual_max": float(np.max(observed)), "actual_zero_ratio": float(np.mean(observed == 0))}
        tolerance = TOLERANCES[arm.get("q_mode", "fp32")]
        passed = all(item["cosine"] >= tolerance["cosine_min"] and item["nRMSE"] <= tolerance["nrmse_max"] and item["MAE"] <= tolerance["mae_max"] and item["max_abs"] <= tolerance["max_abs_max"] for item in metrics.values())
        report = {"status": "pass" if passed else "fail", "items": metrics, "tolerances": tolerance, **actual_identity, "reference_sha256": sha256_file(reference_npz), "actual_sha256": sha256_file(actual_npz)}
        write_json(output_json, report)
        if not passed:
            raise RunnerError("numeric gate tolerance failure")
        return report
    finally:
        # NPZ files own descriptors; close them even if a gate fails.
        for value in (locals().get("reference"), locals().get("actual")):
            if hasattr(value, "close"):
                value.close()


def validate_measurement_protocol(warmup: int, iters: int, repeat: int) -> None:
    if (warmup, iters, repeat) != (20, 300, 5):
        raise RunnerError("measurement protocol is immutable: warmup=20, iters=300, repeat=5")


def require_sample_count(sample_count: int) -> None:
    if sample_count != PROTOCOL["sample_count"]:
        raise RunnerError("measurement requires exactly 1500 CUDA-event samples")


def secure_tegrastats_command(environment: dict[str, str] | None = None) -> tuple[list[str], dict[str, str]]:
    child_environment = {key: value for key, value in (environment or os.environ).items() if key != "ORIN_SUDO_PW"}
    return [
        "sudo",
        "-n",
        "/usr/bin/tegrastats",
        "--interval",
        "500",
    ], child_environment


def tegrastats_stop_command() -> list[str]:
    return ["sudo", "-n", "/usr/bin/tegrastats", "--stop"]


def parse_tegrastats(path: Path | str) -> dict[str, dict[str, list[Any]]]:
    rails: dict[str, dict[str, list[Any]]] = {}
    rail_pattern = re.compile(r"\b(VIN_SYS_5V0|VDD_GPU_SOC)\s+(\d+(?:\.\d+)?)mW\b")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        timestamp = line.split(maxsplit=1)[0] if line else ""
        for rail, milliwatts in rail_pattern.findall(line):
            record = rails.setdefault(rail, {"watts": [], "timestamps": [], "sample_count": 0})
            record["watts"].append(float(milliwatts) / 1000.0)
            record["timestamps"].append(timestamp)
            record["sample_count"] += 1
    return rails


def compute_energy_j(rails: dict[str, dict[str, list[Any]]], median_latency_ms: float) -> float:
    main = rails.get("VIN_SYS_5V0", {}).get("watts", [])
    if not main:
        raise RunnerError("VIN_SYS_5V0 main-rail energy evidence is missing")
    return float(np.mean(main) * median_latency_ms / 1000.0)


def measurement_receipt(output_json: Path | str, samples: list[float], rails: dict[str, dict[str, list[Any]]], identity: dict[str, Any], power_log: Path | str) -> dict[str, Any]:
    require_sample_count(len(samples))
    try:
        pooled = _summaries(samples)
        receipt = {**identity, "status": "complete", "sample_count": len(samples), "pooled": pooled, "rails": rails, "energy_j": compute_energy_j(rails, pooled["median_ms"]), "raw_power_log_sha256": sha256_file(power_log) if Path(power_log).is_file() else None}
    except RunnerError as error:
        receipt = {**identity, "status": "energy_evidence_missing", "sample_count": len(samples), "error": str(error), "rails": rails}
        write_json(output_json, receipt)
        raise
    write_json(output_json, receipt)
    return receipt


def _summaries(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {"median_ms": float(np.median(values)), "p90_ms": float(np.percentile(values, 90)), "p99_ms": float(np.percentile(values, 99)), "mean_ms": float(np.mean(values))}


def _run_cuda_events(run_once: Callable[[], None], torch: Any) -> tuple[list[float], list[dict[str, float]]]:
    for _ in range(PROTOCOL["warmup"]):
        run_once()
    torch.cuda.synchronize()
    all_samples: list[float] = []
    repeats = []
    for _ in range(PROTOCOL["repeat"]):
        current = []
        for _ in range(PROTOCOL["iters"]):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); run_once(); end.record(); end.synchronize()
            current.append(float(start.elapsed_time(end)))
        all_samples.extend(current)
        repeats.append(_summaries(current))
    require_sample_count(len(all_samples))
    return all_samples, repeats


def _write_samples_csv(path: Path, samples: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "cuda_event_ms"])
        writer.writerows(enumerate(samples))


def _prepare_native_callable(arm: dict[str, Any], input_item: np.ndarray) -> tuple[Callable[[], None], Any]:
    try:
        import torch
        from opencood.hypes_yaml.yaml_utils import load_yaml
        from opencood.tools import train_utils
    except ImportError as error:
        raise RunnerError("measure-native requires OpenCOOD and PyTorch on Orin") from error
    config = load_yaml(arm["config_path"])
    model = train_utils.create_model(config).cuda().eval()
    state = torch.load(arm["checkpoint_path"], map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=True)
    tensor = torch.from_numpy(input_item).cuda(non_blocking=True)
    def run_once() -> None:
        with torch.no_grad():
            _run_native_scope(model, tensor)
    return run_once, torch


def _prepare_trt_callable(engine_path: Path, arm: dict[str, Any], input_item: np.ndarray) -> tuple[Callable[[], None], Any]:
    try:
        import torch
        import tensorrt as trt
    except ImportError as error:
        raise RunnerError("measure-trt requires PyTorch CUDA and TensorRT on Orin") from error
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None or engine.num_bindings != 2:
        raise RunnerError("invalid TensorRT engine binding contract")
    context = engine.create_execution_context()
    input_index = next((index for index in range(engine.num_bindings) if engine.binding_is_input(index)), None)
    output_index = next((index for index in range(engine.num_bindings) if not engine.binding_is_input(index)), None)
    if input_index is None or output_index is None or engine.get_binding_name(input_index) != arm.get("input_name", INPUT_NAME) or engine.get_binding_name(output_index) != arm.get("output_name", OUTPUT_NAME):
        raise RunnerError("engine binding identity mismatch")
    if tuple(engine.get_binding_shape(input_index)) != tuple(arm.get("input_shape", INPUT_SHAPE)):
        raise RunnerError("engine static binding shape mismatch")
    output_shape = tuple(engine.get_binding_shape(output_index))
    if output_shape != tuple(arm["output_shape"]) or any(dimension <= 0 for dimension in output_shape):
        raise RunnerError("engine output binding contract mismatch")
    input_tensor = torch.from_numpy(input_item).cuda(non_blocking=True)
    output_tensor = torch.empty(output_shape, device="cuda", dtype=torch.float32)
    bindings = [0, 0]
    bindings[input_index], bindings[output_index] = input_tensor.data_ptr(), output_tensor.data_ptr()
    def run_once() -> None:
        if not context.execute_v2(bindings):
            raise RunnerError("TensorRT execute_v2 failed")
    return run_once, torch


def _start_tegrastats(power_log: Path) -> tuple[subprocess.Popen[bytes], bytes | None]:
    command, environment = secure_tegrastats_command()
    os.environ.pop("ORIN_SUDO_PW", None)
    try:
        subprocess.run(
            tegrastats_stop_command(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RunnerError("failed to clear stale tegrastats before measurement") from error
    power_log.parent.mkdir(parents=True, exist_ok=True)
    handle = power_log.open("wb")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    # Keep the handle reachable through the process, so it can be closed with it.
    setattr(process, "_fcooper_power_log_handle", handle)
    return process, None


def wait_for_tegrastats_main_rail(process: Any, power_log: Path, timeout_seconds: float = 10.0, sleep: Callable[[float], None] = time.sleep) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if process.poll() is not None:
            raise RunnerError("tegrastats exited before VIN_SYS_5V0 evidence")
        if power_log.is_file() and parse_tegrastats(power_log).get("VIN_SYS_5V0", {}).get("watts"):
            return
        if time.monotonic() >= deadline:
            raise RunnerError("tegrastats did not produce VIN_SYS_5V0 evidence before measurement")
        sleep(0.1)


def _stop_tegrastats(process: subprocess.Popen[bytes], password: bytes | None) -> None:
    try:
        subprocess.run(
            tegrastats_stop_command(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10,
        )
        process.wait(timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RunnerError("tegrastats native stop failed") from error
    finally:
        handle = getattr(process, "_fcooper_power_log_handle", None)
        if handle is not None:
            handle.close()


def _measure(manifest: dict[str, Any], arm_name: str, inputs_path: Path, power_log: Path, raw_samples: Path, output_json: Path, warmup: int, iters: int, repeat: int, engine: Path | None = None) -> dict[str, Any]:
    validate_measurement_protocol(warmup, iters, repeat)
    arm, inputs = get_arm(manifest, arm_name), _load_inputs(inputs_path, require_provenance=True)
    validate_heldout_provenance(manifest, inputs_path)
    is_trt = engine is not None
    if is_trt and (arm_name == "original_default" or arm.get("runtime") != "trt"):
        raise RunnerError("measure-trt rejects original_default")
    if not is_trt and arm_name != "original_default":
        raise RunnerError("measure-native accepts only original_default")
    if is_trt:
        if not engine.is_file():
            raise RunnerError("measure-trt requires a local engine")
        run_once, torch = _prepare_trt_callable(engine, arm, inputs[0])
    else:
        run_once, torch = _prepare_native_callable(arm, inputs[0])
    if not torch.cuda.is_available():
        raise RunnerError("measurement requires CUDA")
    # Inputs, output buffers, and weights were allocated before this point; the
    # event loop measures only the frozen dense scope, never transfer/setup.
    for _ in range(PROTOCOL["warmup"]):
        run_once()
    torch.cuda.synchronize()
    process, password = _start_tegrastats(power_log)
    samples: list[float] = []
    repeat_summaries: list[dict[str, float]] = []
    try:
        wait_for_tegrastats_main_rail(process, power_log)
        for _ in range(PROTOCOL["repeat"]):
            current: list[float] = []
            for _ in range(PROTOCOL["iters"]):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(); run_once(); end.record(); end.synchronize()
                current.append(float(start.elapsed_time(end)))
            samples.extend(current)
            repeat_summaries.append(_summaries(current))
    finally:
        _stop_tegrastats(process, password)
    require_sample_count(len(samples))
    _write_samples_csv(raw_samples, samples)
    identity = {"arm": arm_name, "scope": SCOPE, "protocol": PROTOCOL, "repeat_summaries": repeat_summaries, **measurement_identity(manifest, inputs_path), "checkpoint_sha256": arm.get("checkpoint_sha256"), "config_sha256": arm.get("config_sha256"), "onnx_sha256": arm.get("onnx_sha256"), "engine_sha256": sha256_file(engine) if engine is not None else None, "measurement_platform": list(orin_platform_probe())}
    return measurement_receipt(output_json, samples, parse_tegrastats(power_log), identity, power_log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    for name in ("manifest", "arm", "artifact-root", "engine", "timing-cache", "inspector-json", "artifact-json"):
        build.add_argument(f"--{name}", required=True)
    reference = sub.add_parser("reference")
    for name in ("manifest", "arm", "inputs-npy", "output-npz", "report-json"):
        reference.add_argument(f"--{name}", required=True)
    numerical = sub.add_parser("numerical")
    for name in ("manifest", "arm", "engine", "build-receipt", "inputs-npy", "output-npz", "report-json"):
        numerical.add_argument(f"--{name}", required=True)
    gate = sub.add_parser("numeric-gate")
    for name in ("manifest", "arm", "reference-npz", "actual-npz", "output-json"):
        gate.add_argument(f"--{name}", required=True)
    for command in ("measure-trt", "measure-native"):
        measure = sub.add_parser(command)
        for name in ("manifest", "arm", "inputs-npy", "power-log", "raw-samples-csv", "output-json"):
            measure.add_argument(f"--{name}", required=True)
        measure.add_argument("--engine", required=command == "measure-trt")
        measure.add_argument("--warmup", type=int, default=20)
        measure.add_argument("--iters", type=int, default=300)
        measure.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        manifest["_manifest_sha256"] = sha256_file(args.manifest)
        if args.command == "build":
            paths = {"artifact_root": Path(args.artifact_root), "engine": Path(args.engine), "timing_cache": Path(args.timing_cache), "inspector_json": Path(args.inspector_json), "artifact_json": Path(args.artifact_json)}
            build_engine(manifest, args.arm, paths)
        elif args.command == "reference":
            run_reference(manifest, args.arm, Path(args.inputs_npy), Path(args.output_npz), Path(args.report_json), sha256_file(args.manifest))
        elif args.command == "numerical":
            run_numerical(manifest, args.arm, Path(args.engine), Path(args.build_receipt), Path(args.inputs_npy), Path(args.output_npz), Path(args.report_json), sha256_file(args.manifest))
        elif args.command == "numeric-gate":
            numeric_gate(args.reference_npz, args.actual_npz, args.output_json, get_arm(manifest, args.arm))
        else:
            _measure(manifest, args.arm, Path(args.inputs_npy), Path(args.power_log), Path(args.raw_samples_csv), Path(args.output_json), args.warmup, args.iters, args.repeat, Path(args.engine) if args.command == "measure-trt" else None)
    except RunnerError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
