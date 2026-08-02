#!/usr/bin/env python3
"""Build neutral S1 probes and record compiler structural evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_compiler_fingerprint(version: str, file_paths: list[Path] | tuple[Path, ...]) -> str:
    normalized_version = str(version or "").strip()
    if not normalized_version:
        raise ValueError("compiler version is required")
    fingerprints = []
    for raw_path in sorted((Path(path).resolve() for path in file_paths), key=str):
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        fingerprints.append({"path": str(raw_path), "sha256": _sha256(raw_path)})
    if not fingerprints:
        raise ValueError("at least one compiler runtime file is required")
    payload = {
        "version": normalized_version,
        "files": fingerprints,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _loaded_module_files(prefix: str) -> list[Path]:
    files: set[Path] = set()
    for name, module in sys.modules.items():
        if name != prefix and not name.startswith(f"{prefix}."):
            continue
        location = getattr(module, "__file__", None)
        if not location:
            continue
        path = Path(location).resolve()
        if path.is_file():
            files.add(path)
    return sorted(files)


def _existing_files(paths: list[Path] | tuple[Path, ...]) -> list[Path]:
    unique = {Path(path).resolve() for path in paths}
    return sorted(path for path in unique if path.is_file())


def _collect_tvm_provenance() -> dict[str, Any]:
    import tvm
    from tvm import relax  # noqa: F401
    from tvm.relax.frontend.onnx import from_onnx  # noqa: F401

    files = _loaded_module_files("tvm")
    try:
        from tvm._ffi import libinfo

        files = _existing_files([*files, *(Path(path) for path in libinfo.find_lib_path())])
    except Exception:
        files = _existing_files(files)
    fingerprint = build_compiler_fingerprint(tvm.__version__, files)
    return {
        "backend": "tvm",
        "compiler_version": str(tvm.__version__),
        "compiler_files": [str(path) for path in files],
        "compiler_file_sha256": {str(path): _sha256(path) for path in files},
        "compiler_fingerprint": fingerprint,
    }


def _collect_trt_provenance() -> dict[str, Any]:
    import tensorrt as trt

    files = _existing_files(_loaded_module_files("tensorrt"))
    fingerprint = build_compiler_fingerprint(trt.__version__, files)
    return {
        "backend": "trt",
        "compiler_version": str(trt.__version__),
        "compiler_files": [str(path) for path in files],
        "compiler_file_sha256": {str(path): _sha256(path) for path in files},
        "compiler_fingerprint": fingerprint,
    }


def _collect_backend_provenance(backend: str) -> dict[str, Any]:
    if backend == "tvm":
        return _collect_tvm_provenance()
    if backend == "trt":
        return _collect_trt_provenance()
    raise ValueError(f"unsupported backend: {backend}")


def _onnx_counts(path: Path) -> dict[str, int]:
    import onnx

    model = onnx.load(str(path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _empty_counts() -> dict[str, None]:
    return {
        "int8_propagated_ops": None,
        "precision_eligible_ops": None,
        "qdq_folded_pairs": None,
        "qdq_pairs": None,
        "reformat_ops": None,
        "total_ops": None,
        "fused_ops": None,
        "fusible_ops": None,
    }


def _tvm_build(path: Path, artifact_dir: Path) -> tuple[dict[str, Any], str]:
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    import onnx

    model = onnx.load(str(path))
    shapes = {
        value.name: tuple(int(dim.dim_value) for dim in value.type.tensor_type.shape.dim)
        for value in model.graph.input
    }
    mod = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("TVM CUDA device is unavailable")
    target = tvm.target.Target.from_device(dev)
    seq = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    with target, tvm.transform.PassContext(opt_level=3):
        lowered = seq(mod)
        tvm.compile(lowered, target=target)
    script = lowered.script(show_meta=True)
    (artifact_dir / "lowered_tir.py").write_text(script, encoding="utf-8")
    counts = _onnx_counts(path)
    conv_count = counts.get("Conv", 0)
    qdq_pairs = min(counts.get("QuantizeLinear", 0), counts.get("DequantizeLinear", 0))
    conv_sections = []
    parts = script.split("    @T.prim_func")
    for part in parts:
        header = part[:300].lower()
        if "def conv" in header:
            conv_sections.append(part)
    int8_conv_count = sum("int8" in section.lower() for section in conv_sections)
    primfunc_count = script.count("@T.prim_func")
    reformat_count = script.lower().count("layout_transform")
    return {
        "int8_propagated_ops": min(conv_count, int8_conv_count) if qdq_pairs else 0,
        "precision_eligible_ops": conv_count,
        "qdq_folded_pairs": qdq_pairs if qdq_pairs and "quantize" not in script.lower() else 0,
        "qdq_pairs": qdq_pairs,
        "reformat_ops": reformat_count,
        "total_ops": max(1, primfunc_count),
        "fused_ops": 0,
        "fusible_ops": conv_count,
    }, script


def _trt_layers(info: Any) -> list[dict[str, Any]]:
    if isinstance(info, list):
        return [item for item in info if isinstance(item, dict)]
    if isinstance(info, dict):
        values = info.get("Layers", [])
        return [item for item in values if isinstance(item, dict)]
    return []


def _normalize_precision_metrics(metrics: dict[str, Any], q_mode: str) -> dict[str, Any]:
    normalized = dict(metrics)
    if q_mode == "fp16":
        for field in (
            "int8_propagated_ops",
            "precision_eligible_ops",
            "qdq_folded_pairs",
            "qdq_pairs",
        ):
            normalized[field] = None
    return normalized


def _trt_build(path: Path, artifact_dir: Path, q_mode: str) -> tuple[dict[str, Any], str]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(path.read_bytes()):
        raise RuntimeError("ONNX parse failed: " + " | ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    if q_mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build returned None")
    engine_path = artifact_dir / "engine.plan"
    engine_path.write_bytes(bytes(serialized))
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("TensorRT deserialize failed")
    text = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    (artifact_dir / "engine_inspector.json").write_text(text, encoding="utf-8")
    info = json.loads(text)
    layers = _trt_layers(info)
    counts = _onnx_counts(path)
    conv_count = counts.get("Conv", 0)
    qdq_pairs = min(counts.get("QuantizeLinear", 0), counts.get("DequantizeLinear", 0))
    conv_layers = [
        layer for layer in layers if str(layer.get("ParameterType", "")).lower() == "convolution"
    ]
    int8_conv = [
        layer
        for layer in conv_layers
        if "int8" in json.dumps(layer.get("Inputs", []), sort_keys=True).lower()
        or str(layer.get("Weights", {}).get("Type", "")).lower() == "int8"
    ]
    reformats = [
        layer for layer in layers if str(layer.get("LayerType", "")).lower() == "reformat"
    ]
    qdq_layers = [
        layer for layer in reformats if str(layer.get("Origin", "")).upper() == "QDQ"
    ]
    return {
        "int8_propagated_ops": min(conv_count, len(int8_conv)),
        "precision_eligible_ops": conv_count,
        "qdq_folded_pairs": max(0, qdq_pairs - len(qdq_layers)),
        "qdq_pairs": qdq_pairs,
        "reformat_ops": len(reformats),
        "total_ops": max(1, len(layers)),
        "fused_ops": max(0, conv_count - len(conv_layers)),
        "fusible_ops": conv_count,
    }, text


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = _collect_backend_provenance(args.backend)
    records = []
    started = time.monotonic()
    for item in manifest["probes"]:
        artifact = item.get("artifact") or item
        path = root / artifact["path"]
        artifact_sha256 = artifact["sha256"]
        if _sha256(path) != artifact_sha256:
            raise ValueError(f"probe SHA mismatch: {path}")
        q_mode = "int8" if item["precision"] in {"int8", "int8_qdq"} else "fp16"
        artifact_dir = output / args.backend / item["probe_id"] / q_mode
        artifact_dir.mkdir(parents=True, exist_ok=True)
        begin = time.monotonic()
        success = False
        error = None
        counts: dict[str, Any] = _empty_counts()
        try:
            if args.backend == "tvm":
                counts, _ = _tvm_build(path, artifact_dir)
            else:
                counts, _ = _trt_build(path, artifact_dir, q_mode)
            counts = _normalize_precision_metrics(counts, q_mode)
            success = True
        except Exception as exc:  # each failure remains useful feasibility evidence
            error = f"{type(exc).__name__}: {exc}"
            (artifact_dir / "error.txt").write_text(error + "\n", encoding="utf-8")
        elapsed = time.monotonic() - begin
        records.append(
            {
                "schema_version": "stage2_s1_structural_probe_record_v3",
                "backend_runner": args.backend,
                "probe_id": item["probe_id"],
                "q_mode": q_mode,
                "onnx_sha256": artifact_sha256,
                "build_success": success,
                "build_seconds": elapsed,
                "probe_seconds": elapsed,
                "error": error,
                **counts,
            }
        )
    report = {
        "schema_version": "stage2_s1_structural_probe_run_v3",
        "backend_runner": args.backend,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "manifest_sha256": _sha256(manifest_path),
        "wall_seconds": time.monotonic() - started,
        "contains_latency_or_energy_measurement": False,
        "provenance": provenance,
        "records": records,
    }
    (output / f"{args.backend}_records.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tvm", "trt"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"backend": result["backend_runner"], "records": len(result["records"]), "wall_seconds": result["wall_seconds"]}))
