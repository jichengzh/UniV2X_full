#!/usr/bin/env python3
"""Fail-closed F-Cooper Orin five-arm Table 2 evidence finalizer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARMS = (
    "original_default",
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "joint_fp16_control",
)
DISPLAY_LABELS = {
    "original_default": "Original/default",
    "compression_only": "Compression only",
    "schedule_only": "Schedule only",
    "compress_then_tune": "Compress -> Tune",
    "joint_fp16_control": "Joint FP16 control",
}
SCOPE = "post_scatter_backbone_shrinker"
INPUT_SHAPE = [5, 64, 512, 512]
PROTOCOL = {"warmup": 20, "iters": 300, "repeat": 5, "sample_count": 1500}
EXPECTED_ARM_CONTRACT = {
    "original_default": ("native", "fp32", None, None),
    "compression_only": ("trt", "fp16", "trt85_default", None),
    "schedule_only": ("trt", "fp32", "trt85_default", None),
    "compress_then_tune": ("trt", "fp16", "trt85_default", None),
    "joint_fp16_control": ("trt", "fp16", "trt85_default", None),
}
NUMERIC_TOLERANCES = {
    "fp32": {
        "cosine_min": 0.99999,
        "nrmse_max": 0.01,
        "mae_max": 0.001,
        "max_abs_max": 0.05,
    },
    "fp16": {
        "cosine_min": 0.999,
        "nrmse_max": 0.05,
        "mae_max": 0.05,
        "max_abs_max": 0.5,
    },
}
CSV_FIELDS = (
    "arm",
    "display_label",
    "ap70",
    "latency_median_ms",
    "energy_j",
)


class FinalizationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code: str, message: str) -> None:
    raise FinalizationError(code, message)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, code: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(code, f"missing evidence: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(code, f"invalid JSON evidence {path}: {error}")
    if not isinstance(value, dict):
        _fail(code, f"expected JSON object: {path}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _resolved_within(path: Path | str, root: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail("source_sha_mismatch", f"evidence escapes result root: {resolved}")
    return resolved


def _finite(value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a measurement")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"non-finite or out-of-range measurement: {value}")
    return number


def _equal_float(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-9, abs_tol=1e-12)


def _is_orin_platform(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 3
        and value[0] == "Linux"
        and str(value[1]).lower() in {"aarch64", "arm64"}
        and "orin" in " ".join(map(str, value[2:])).lower()
    )


def _artifact(
    path: Path,
    *,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
    code: str = "source_artifacts_not_restored",
) -> Path:
    resolved = _resolved_within(path, root)
    if not resolved.is_file():
        _fail(code, f"missing evidence: {resolved}")
    relative = resolved.relative_to(root).as_posix()
    artifacts[relative] = {
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    return resolved


def _identity(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> None:
    for key, frozen in expected.items():
        if value.get(key) != frozen:
            _fail(
                "source_sha_mismatch",
                f"{context} identity mismatch for {key}",
            )


def _validate_manifest(
    manifest_path: Path,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    path = _artifact(manifest_path, root=root, artifacts=artifacts)
    manifest = _load_json(path, "source_artifacts_not_restored")
    if manifest.get("status") != "ready":
        _fail("source_artifacts_not_restored", "canonical manifest is not ready")
    if manifest.get("scope") != SCOPE:
        _fail("scope_mismatch", "canonical manifest scope mismatch")
    if manifest.get("input_shape") != INPUT_SHAPE:
        _fail("scope_mismatch", "canonical manifest input shape mismatch")
    arms = manifest.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(ARMS) or len(arms) != 5:
        _fail(
            "source_artifacts_not_restored",
            "canonical manifest must contain exactly five frozen arms",
        )
    for name in ARMS:
        arm = arms[name]
        runtime, precision, builder_policy, builder_level = EXPECTED_ARM_CONTRACT[
            name
        ]
        if (
            arm.get("runtime") != runtime
            or arm.get("q_mode") != precision
            or arm.get("builder_policy") != builder_policy
            or arm.get("builder_optimization_level") != builder_level
        ):
            code = (
                "joint_cross_precision_control"
                if name == "joint_fp16_control"
                else "source_sha_mismatch"
            )
            _fail(code, f"{name} frozen runtime/precision/builder drift")
        if name == "schedule_only" and arm.get("strict_fp32") is not True:
            _fail(
                "strict_fp32_inspector_failed",
                "schedule_only lacks strict_fp32 contract",
            )
        if name == "joint_fp16_control" and (
            arm.get("joint_cross_precision_control") is not True
            or arm.get("same_precision_h800_int8_reproduction") is True
        ):
            _fail(
                "joint_cross_precision_control",
                "Joint must be explicit FP16 cross-precision control",
            )
        for label in ("checkpoint", "config", "onnx"):
            source_path = _artifact(
                Path(str(arm.get(f"{label}_path", ""))),
                root=root,
                artifacts=artifacts,
            )
            if sha256_file(source_path) != arm.get(f"{label}_sha256"):
                _fail("source_sha_mismatch", f"{name} {label} SHA mismatch")
    return manifest, sha256_file(path)


def _validate_heldout(
    manifest: dict[str, Any],
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, str | int]:
    input_path = _artifact(
        root / "01_numeric_gate/heldout_inputs.npy",
        root=root,
        artifacts=artifacts,
    )
    capture_path = _artifact(
        root / "01_numeric_gate/heldout_manifest.json",
        root=root,
        artifacts=artifacts,
    )
    sidecar_path = _artifact(
        input_path.with_suffix(".heldout.json"),
        root=root,
        artifacts=artifacts,
    )
    capture = _load_json(capture_path, "real_heldout_numeric_gate_failed")
    sidecar = _load_json(sidecar_path, "real_heldout_numeric_gate_failed")
    contract = capture.get("tensor_contract", {})
    shape = contract.get("shape")
    valid_shape = (
        isinstance(shape, list)
        and len(shape) == 5
        and isinstance(shape[0], int)
        and 1 <= shape[0] <= 4
        and shape[1:] == INPUT_SHAPE
    )
    items = capture.get("items")
    if (
        capture.get("status") != "ready"
        or capture.get("purpose") != "heldout_numeric_only_not_calibration"
        or capture.get("calibration_eligible") is not False
        or capture.get("scope") != SCOPE
        or capture.get("dataset_samples") != 2170
        or not valid_shape
        or contract.get("dtype") != "float32"
        or contract.get("engine_agent_batch") != 5
        or not isinstance(items, list)
        or len(items) != shape[0]
    ):
        _fail(
            "real_heldout_numeric_gate_failed",
            "held-out capture contract is incomplete",
        )
    if any(not isinstance(item, dict) for item in items):
        _fail(
            "real_heldout_numeric_gate_failed",
            "held-out item provenance is incomplete",
        )
    stable_ids: list[str] = []
    for item in items:
        stable_id = item.get("stable_id")
        tensor_sha = item.get("tensor_sha256")
        agents = item.get("agent_count")
        if (
            not isinstance(stable_id, str)
            or not stable_id
            or not isinstance(tensor_sha, str)
            or len(tensor_sha) != 64
            or isinstance(agents, bool)
            or not isinstance(agents, int)
            or not 1 <= agents <= 5
        ):
            _fail(
                "real_heldout_numeric_gate_failed",
                "held-out item provenance is incomplete",
            )
        stable_ids.append(stable_id)
    if len(set(stable_ids)) != len(stable_ids):
        _fail(
            "real_heldout_numeric_gate_failed",
            "held-out stable IDs are not unique",
        )
    frozen_test_sha = (
        manifest.get("global_identities", {})
        .get("contracts/opv2v_test_manifest_fresh.txt", {})
        .get("sha256")
    )
    expected = {
        "input_sha256": sha256_file(input_path),
        "capture_provenance_sha256": sha256_file(capture_path),
        "opv2v_test_manifest_sha256": frozen_test_sha,
    }
    if capture.get("npy_sha256") != expected["input_sha256"]:
        _fail("source_sha_mismatch", "held-out NPY SHA mismatch")
    if capture.get("opv2v_test_manifest_sha256") != frozen_test_sha:
        _fail("source_sha_mismatch", "held-out test manifest SHA mismatch")
    _identity(sidecar, expected, context="held-out sidecar")
    if (
        sidecar.get("purpose") != "heldout_numeric_only_not_calibration"
        or sidecar.get("calibration_eligible") is not False
        or sidecar.get("scope") != SCOPE
    ):
        _fail(
            "real_heldout_numeric_gate_failed",
            "held-out sidecar purpose mismatch",
        )
    return {**expected, "item_count": shape[0]}


def _validate_original_reference(
    manifest: dict[str, Any],
    heldout: Mapping[str, Any],
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> None:
    path = _artifact(
        root / "01_numeric_gate/original_default/reference.json",
        root=root,
        artifacts=artifacts,
    )
    report = _load_json(path, "real_heldout_numeric_gate_failed")
    arm = manifest["arms"]["original_default"]
    if (
        report.get("status") != "complete"
        or report.get("kind") != "reference"
        or report.get("arm") != "original_default"
        or report.get("scope") != SCOPE
    ):
        _fail(
            "real_heldout_numeric_gate_failed",
            "Original native reference is incomplete",
        )
    _identity(
        report,
        {
            "input_sha256": heldout["input_sha256"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "config_sha256": arm["config_sha256"],
        },
        context="Original reference",
    )
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or len(outputs) != heldout["item_count"]:
        _fail(
            "real_heldout_numeric_gate_failed",
            "Original reference outputs are incomplete",
        )
    for output in outputs.values():
        if (
            output.get("finite") is not True
            or output.get("dtype") != "float32"
            or not isinstance(output.get("shape"), list)
        ):
            _fail(
                "real_heldout_numeric_gate_failed",
                "Original reference output contract failed",
            )


def _layer_precision_evidence(layer: Mapping[str, Any]) -> str:
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
                if isinstance(tensor, Mapping)
            )
    return " ".join(values).upper()


def _validate_inspector(inspector: dict[str, Any]) -> None:
    if inspector.get("unsupported_fields"):
        _fail(
            "strict_fp32_inspector_failed",
            "schedule inspector has unsupported fields",
        )
    layers = inspector.get("layers")
    if not isinstance(layers, list) or not layers:
        _fail("strict_fp32_inspector_failed", "schedule inspector has no layers")
    for layer in layers:
        evidence = _layer_precision_evidence(layer)
        if any(
            token in evidence
            for token in ("FP16", "INT8", "BF16", "TF32", "HALF", "KHALF")
        ) or ("FP32" not in evidence and "FLOAT" not in evidence):
            _fail(
                "strict_fp32_inspector_failed",
                "schedule layer is not proven FP32: "
                f"{layer.get('name', layer.get('Name'))}",
            )


def _validate_build(
    name: str,
    arm: dict[str, Any],
    manifest_sha: str,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, str]:
    directory = root / f"02_engines/{name}"
    engine = _artifact(directory / "model.engine", root=root, artifacts=artifacts)
    cache = _artifact(directory / "timing.cache", root=root, artifacts=artifacts)
    inspector_path = _artifact(directory / "inspector.json", root=root, artifacts=artifacts)
    build_path = _artifact(directory / "build.json", root=root, artifacts=artifacts)
    receipt = _load_json(build_path, "source_artifacts_not_restored")
    engine_sha, cache_sha = sha256_file(engine), sha256_file(cache)
    expected = {
        "status": "built",
        "scope": SCOPE,
        "arm": name,
        "manifest_sha256": manifest_sha,
        "engine_sha256": engine_sha,
        "timing_cache_sha256": cache_sha,
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "config_sha256": arm["config_sha256"],
        "onnx_sha256": arm["onnx_sha256"],
    }
    _identity(receipt, expected, context=f"{name} build")
    if not _is_orin_platform(receipt.get("platform")):
        _fail("source_sha_mismatch", f"{name} build is not Orin/aarch64")
    if not str(receipt.get("tensorrt_version", "")).startswith("8.5."):
        _fail("source_sha_mismatch", f"{name} build is not TensorRT 8.5.x")
    if (
        receipt.get("h800_engine_or_cache_read") is not False
        or receipt.get("calibration_cache_read") is not False
    ):
        _fail("source_sha_mismatch", f"{name} reused forbidden build evidence")
    if Path(str(receipt.get("engine", ""))).resolve() != engine:
        _fail("source_sha_mismatch", f"{name} engine path mismatch")
    if Path(str(receipt.get("timing_cache", ""))).resolve() != cache:
        _fail("source_sha_mismatch", f"{name} timing-cache path mismatch")
    arguments = receipt.get("build_arguments", {})
    if arguments != {
        "builder_policy": arm["builder_policy"],
        "builder_optimization_level": arm["builder_optimization_level"],
        "q_mode": arm["q_mode"],
    }:
        _fail("source_sha_mismatch", f"{name} builder contract mismatch")
    inspector = _load_json(inspector_path, "strict_fp32_inspector_failed")
    if name == "schedule_only":
        _validate_inspector(inspector)
    return {
        "engine_sha256": engine_sha,
        "build_receipt_sha256": sha256_file(build_path),
    }


def _validate_gate(
    name: str,
    arm: dict[str, Any],
    build: Mapping[str, str],
    manifest_sha: str,
    heldout: Mapping[str, Any],
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> None:
    path = _artifact(
        root / f"01_numeric_gate/{name}/gate.json",
        root=root,
        artifacts=artifacts,
    )
    gate = _load_json(path, "real_heldout_numeric_gate_failed")
    if gate.get("status") != "pass":
        _fail("real_heldout_numeric_gate_failed", f"{name} numeric gate failed")
    _identity(
        gate,
        {
            "manifest_sha256": manifest_sha,
            "input_sha256": heldout["input_sha256"],
            "capture_provenance_sha256": heldout["capture_provenance_sha256"],
            "opv2v_test_manifest_sha256": heldout["opv2v_test_manifest_sha256"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "config_sha256": arm["config_sha256"],
            "onnx_sha256": arm["onnx_sha256"],
            "engine_sha256": build["engine_sha256"],
        },
        context=f"{name} numeric gate",
    )
    items = gate.get("items")
    required_metrics = {
        "cosine",
        "nRMSE",
        "MAE",
        "max_abs",
        "finite_ratio",
        "shape",
        "reference_dtype",
        "actual_dtype",
        "reference_min",
        "reference_max",
        "reference_zero_ratio",
        "actual_min",
        "actual_max",
        "actual_zero_ratio",
    }
    if not isinstance(items, dict) or len(items) != heldout["item_count"]:
        _fail(
            "real_heldout_numeric_gate_failed",
            f"{name} numeric metrics are incomplete",
        )
    tolerance = NUMERIC_TOLERANCES[arm["q_mode"]]
    if gate.get("tolerances") != tolerance:
        _fail(
            "real_heldout_numeric_gate_failed",
            f"{name} numeric tolerance contract mismatch",
        )
    for metrics in items.values():
        if not required_metrics.issubset(metrics):
            _fail(
                "real_heldout_numeric_gate_failed",
                f"{name} numeric metric fields are incomplete",
            )
        numeric_keys = required_metrics - {
            "shape",
            "reference_dtype",
            "actual_dtype",
        }
        try:
            numeric = {
                key: _finite(metrics[key])
                for key in numeric_keys
                if isinstance(metrics[key], (int, float)) and not isinstance(metrics[key], bool)
            }
            if len(numeric) != len(numeric_keys):
                raise TypeError("metric is not numeric")
        except (TypeError, ValueError):
            _fail(
                "real_heldout_numeric_gate_failed",
                f"{name} numeric metric is non-finite",
            )
        if (
            numeric["finite_ratio"] != 1.0
            or metrics["reference_dtype"] != "float32"
            or metrics["actual_dtype"] != "float32"
            or not isinstance(metrics["shape"], list)
            or numeric["cosine"] < tolerance["cosine_min"]
            or numeric["nRMSE"] > tolerance["nrmse_max"]
            or numeric["MAE"] > tolerance["mae_max"]
            or numeric["max_abs"] > tolerance["max_abs_max"]
        ):
            _fail(
                "real_heldout_numeric_gate_failed",
                f"{name} numeric output contract mismatch",
            )


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _load_latency_samples(path: Path) -> list[float]:
    values: list[float] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["sample_index", "cuda_event_ms"]:
                _fail("latency_protocol_mismatch", "latency CSV header mismatch")
            for expected_index, row in enumerate(reader):
                if int(row["sample_index"]) != expected_index:
                    _fail(
                        "latency_protocol_mismatch",
                        "latency sample index drift",
                    )
                values.append(_finite(row["cuda_event_ms"], minimum=0.0))
    except (OSError, ValueError, TypeError):
        _fail("latency_protocol_mismatch", f"invalid latency CSV: {path}")
    if len(values) != 1500:
        _fail("latency_protocol_mismatch", "latency CSV is not 1500 samples")
    return values


def _validate_latency_energy(
    name: str,
    arm: dict[str, Any],
    heldout: Mapping[str, Any],
    engine_sha: str | None,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, float]:
    directory = root / f"03_latency/{name}"
    report_path = _artifact(directory / "report.json", root=root, artifacts=artifacts)
    samples_path = _artifact(directory / "samples.csv", root=root, artifacts=artifacts)
    report = _load_json(report_path, "latency_protocol_mismatch")
    if report.get("status") != "complete":
        _fail("latency_protocol_mismatch", f"{name} latency is not complete")
    if report.get("scope") != SCOPE:
        _fail("scope_mismatch", f"{name} latency scope mismatch")
    if not _is_orin_platform(report.get("measurement_platform")):
        _fail("source_sha_mismatch", f"{name} measurement is not Orin/aarch64")
    if report.get("sample_count") != 1500:
        _fail("latency_protocol_mismatch", f"{name} sample count mismatch")
    protocol = report.get("protocol", {})
    if any(protocol.get(key) != value for key, value in PROTOCOL.items()):
        _fail("latency_protocol_mismatch", f"{name} protocol mismatch")
    _identity(
        report,
        {
            "input_sha256": heldout["input_sha256"],
            "capture_provenance_sha256": heldout["capture_provenance_sha256"],
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "config_sha256": arm["config_sha256"],
            "engine_sha256": engine_sha,
        },
        context=f"{name} latency",
    )
    if engine_sha is not None and report.get("onnx_sha256") != arm["onnx_sha256"]:
        _fail("source_sha_mismatch", f"{name} latency ONNX mismatch")
    samples = _load_latency_samples(samples_path)
    calculated = {
        "median_ms": statistics.median(samples),
        "p90_ms": _percentile(samples, 0.90),
        "p99_ms": _percentile(samples, 0.99),
        "mean_ms": statistics.fmean(samples),
    }
    pooled = report.get("pooled", {})
    try:
        for key, value in calculated.items():
            if not _equal_float(_finite(pooled.get(key)), value):
                _fail(
                    "latency_protocol_mismatch",
                    f"{name} pooled {key} does not match raw samples",
                )
    except (TypeError, ValueError):
        _fail("latency_protocol_mismatch", f"{name} pooled latency is non-finite")

    rails = report.get("rails", {})
    main = rails.get("VIN_SYS_5V0", {}) if isinstance(rails, dict) else {}
    watts = main.get("watts")
    timestamps = main.get("timestamps")
    if (
        not isinstance(watts, list)
        or not watts
        or not isinstance(timestamps, list)
        or len(timestamps) != len(watts)
        or main.get("sample_count") != len(watts)
    ):
        _fail("energy_evidence_missing", f"{name} lacks VIN_SYS_5V0 samples")
    try:
        main_watts = [_finite(value, minimum=0.0) for value in watts]
        energy = _finite(report.get("energy_j"), minimum=0.0)
    except (TypeError, ValueError):
        _fail("energy_evidence_missing", f"{name} energy is non-finite")
    expected_energy = statistics.fmean(main_watts) * calculated["median_ms"] / 1000.0
    if not _equal_float(energy, expected_energy):
        _fail(
            "energy_evidence_missing",
            f"{name} energy does not use VIN_SYS_5V0-only formula",
        )
    power_path = _artifact(
        root / f"04_energy/raw_tegrastats/{name}.log",
        root=root,
        artifacts=artifacts,
        code="energy_evidence_missing",
    )
    if report.get("raw_power_log_sha256") != sha256_file(power_path):
        _fail("energy_evidence_missing", f"{name} raw power log SHA mismatch")
    return {
        "latency_median_ms": calculated["median_ms"],
        "latency_p90_ms": calculated["p90_ms"],
        "latency_p99_ms": calculated["p99_ms"],
        "latency_mean_ms": calculated["mean_ms"],
        "mean_vin_sys_5v0_w": statistics.fmean(main_watts),
        "energy_j": energy,
    }


def _canonical_id_sha(sample_ids: Sequence[str]) -> str:
    content = json.dumps(list(sample_ids), ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(content).hexdigest()


def _validate_ap(
    name: str,
    arm: dict[str, Any],
    manifest_sha: str,
    engine_sha: str | None,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> tuple[dict[str, float], str]:
    directory = root / f"05_full2170_ap/{name}"
    metrics_path = _artifact(directory / "metrics.json", root=root, artifacts=artifacts)
    prediction_path = _artifact(
        directory / "prediction_manifest.json", root=root, artifacts=artifacts
    )
    run_path = _artifact(directory / "run_manifest.json", root=root, artifacts=artifacts)
    metrics = _load_json(metrics_path, "full2170_ap_incomplete")
    prediction = _load_json(prediction_path, "full2170_ap_incomplete")
    run_log = _load_json(run_path, "full2170_ap_incomplete")
    if metrics.get("fallback_samples") not in (None, 0):
        _fail("engine_fallback_detected", f"{name} AP used fallback")
    expected_calls = 0 if name == "original_default" else 2170
    expected_runtime = "native_fp32" if name == "original_default" else "trt"
    expected_requested = "native" if name == "original_default" else "trt"
    if (
        metrics.get("status") != "success_full"
        or metrics.get("dataset") != "OPV2V"
        or metrics.get("split") != "test"
        or metrics.get("dataset_samples") != 2170
        or metrics.get("processed_samples") != 2170
        or metrics.get("failed_samples") != 0
        or metrics.get("fallback_samples") != 0
        or metrics.get("engine_calls") != expected_calls
        or metrics.get("runtime") != expected_runtime
    ):
        _fail("full2170_ap_incomplete", f"{name} AP is not full-2170")
    if metrics.get("scope") != SCOPE:
        _fail("scope_mismatch", f"{name} AP scope mismatch")
    if metrics.get("requested_runtime") != expected_requested:
        _fail("source_sha_mismatch", f"{name} requested runtime mismatch")
    if name == "original_default":
        if metrics.get("native_identity") != "exact_checkpoint_pytorch_cuda_fp32_unchanged":
            _fail("source_sha_mismatch", "Original native AP identity mismatch")
    elif metrics.get("native_identity") is not None:
        _fail("source_sha_mismatch", f"{name} unexpectedly claims native AP")
    numerical = metrics.get("numerical_contract")
    if (
        not isinstance(numerical, dict)
        or numerical.get("dense_scope_engine_execution") is not True
        or numerical.get("silent_fallback_forbidden") is not True
        or numerical.get("fallback_samples") != 0
    ):
        _fail("engine_fallback_detected", f"{name} AP scope execution failed")
    _identity(
        metrics,
        {
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "config_sha256": arm["config_sha256"],
            "engine_sha256": engine_sha,
        },
        context=f"{name} AP",
    )
    if name != "original_default":
        build_identity = metrics.get("build_identity", {})
        _identity(
            build_identity,
            {
                "manifest_sha256": manifest_sha,
                "engine_sha256": engine_sha,
                "checkpoint_sha256": arm["checkpoint_sha256"],
                "config_sha256": arm["config_sha256"],
                "onnx_sha256": arm["onnx_sha256"],
            },
            context=f"{name} AP build",
        )
        if metrics.get("arm") != name:
            _fail("source_sha_mismatch", f"{name} AP arm mismatch")
    sample_ids = prediction.get("sample_ids")
    if (
        prediction.get("dataset") != "OPV2V"
        or prediction.get("split") != "test"
        or prediction.get("dataset_samples") != 2170
        or prediction.get("processed_samples") != 2170
        or not isinstance(sample_ids, list)
        or len(sample_ids) != 2170
        or len(set(sample_ids)) != 2170
        or prediction.get("sample_ids_sha256") != _canonical_id_sha(sample_ids)
    ):
        _fail("full2170_ap_incomplete", f"{name} prediction manifest incomplete")
    prediction_sha = sha256_file(prediction_path)
    if metrics.get("prediction_manifest_sha256") != prediction_sha:
        _fail("source_sha_mismatch", f"{name} prediction SHA link mismatch")
    if (
        run_log.get("status") != "success"
        or run_log.get("scope") != SCOPE
        or run_log.get("runtime") != expected_runtime
        or run_log.get("prediction_manifest_sha256") != prediction_sha
        or run_log.get("output_json_sha256") != sha256_file(metrics_path)
    ):
        _fail("full2170_ap_incomplete", f"{name} run manifest incomplete")
    _identity(
        run_log,
        {
            "checkpoint_sha256": arm["checkpoint_sha256"],
            "config_sha256": arm["config_sha256"],
            "engine_sha256": engine_sha,
        },
        context=f"{name} AP run",
    )
    try:
        result = {key: _finite(metrics.get(key), minimum=0.0) for key in ("ap30", "ap50", "ap70")}
    except (TypeError, ValueError):
        _fail("full2170_ap_incomplete", f"{name} AP metrics are non-finite")
    if any(value > 1.0 for value in result.values()):
        _fail("full2170_ap_incomplete", f"{name} AP is out of range")
    return result, prediction["sample_ids_sha256"]


def _collect(
    manifest_path: Path,
    root: Path,
    artifacts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    blocker = root / "02_engines/build_contract_blocker.json"
    if blocker.is_file():
        _artifact(blocker, root=root, artifacts=artifacts)
        value = _load_json(blocker, "build_contract_unsupported")
        if value.get("status") == "build_contract_unsupported":
            _fail(
                "build_contract_unsupported",
                "Orin TensorRT builder contract is blocked",
            )
    manifest, manifest_sha = _validate_manifest(manifest_path, root, artifacts)
    heldout = _validate_heldout(manifest, root, artifacts)
    _validate_original_reference(manifest, heldout, root, artifacts)
    rows: list[dict[str, Any]] = []
    prediction_ids_sha: str | None = None
    for name in ARMS:
        arm = manifest["arms"][name]
        build = (
            None
            if name == "original_default"
            else _validate_build(name, arm, manifest_sha, root, artifacts)
        )
        if build is not None:
            _validate_gate(name, arm, build, manifest_sha, heldout, root, artifacts)
        engine_sha = None if build is None else build["engine_sha256"]
        performance = _validate_latency_energy(name, arm, heldout, engine_sha, root, artifacts)
        ap, arm_prediction_ids_sha = _validate_ap(
            name, arm, manifest_sha, engine_sha, root, artifacts
        )
        if prediction_ids_sha is None:
            prediction_ids_sha = arm_prediction_ids_sha
        elif arm_prediction_ids_sha != prediction_ids_sha:
            _fail(
                "source_sha_mismatch",
                f"{name} prediction sample sequence differs across arms",
            )
        rows.append(
            {
                "arm": name,
                "display_label": DISPLAY_LABELS[name],
                **ap,
                **performance,
                "runtime": arm["runtime"],
                "precision": arm["q_mode"],
                "joint_fp16_control": name == "joint_fp16_control",
            }
        )
    return rows, manifest_sha


def _table_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "arm": row["arm"],
            "display_label": row["display_label"],
            "ap70": row["ap70"],
            "latency_median_ms": row["latency_median_ms"],
            "energy_j": row["energy_j"],
        }
        for row in rows
    ]


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(CSV_FIELDS))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def finalize(
    manifest_path: Path | str,
    result_root: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    root = Path(result_root).resolve()
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    _resolved_within(manifest, root)
    if output != root / "06_final":
        _fail("source_sha_mismatch", "output directory must be result-root/06_final")
    artifacts: dict[str, dict[str, Any]] = {}
    evidence_path = output / "evidence_manifest.json"
    try:
        rows, manifest_sha = _collect(manifest, root, artifacts)
        table_rows = _table_rows(rows)
        table_json = {
            "schema_version": "fcooper_orin_table2_values_v1",
            "status": "complete",
            "evidence_grade": "A",
            "row_count": 5,
            "rows": rows,
        }
        json_bytes = (json.dumps(table_json, indent=2, sort_keys=True) + "\n").encode("utf-8")
        csv_bytes = _csv_bytes(table_rows)
        output_hashes = {
            "table2_values.json": hashlib.sha256(json_bytes).hexdigest(),
            "table2_values.csv": hashlib.sha256(csv_bytes).hexdigest(),
        }
        evidence = {
            "schema_version": "fcooper_orin_evidence_manifest_v1",
            "status": "complete",
            "evidence_grade": "A",
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_sha,
            "scope": SCOPE,
            "arms": list(ARMS),
            "joint_disclosure": (
                "Orin FP16 control of the H800-selected structure; "
                "not a same-precision H800 INT8 reproduction"
            ),
            "artifacts": dict(sorted(artifacts.items())),
            "outputs": output_hashes,
        }
        _atomic_json(
            evidence_path,
            {
                "schema_version": "fcooper_orin_evidence_manifest_v1",
                "status": "writing",
                "evidence_grade": "Invalid",
                "manifest_path": str(manifest),
                "manifest_sha256": manifest_sha,
            },
        )
        _atomic_write(output / "table2_values.json", json_bytes)
        _atomic_write(output / "table2_values.csv", csv_bytes)
        _atomic_json(evidence_path, evidence)
        return evidence
    except (FinalizationError, OSError) as error:
        code = error.code if isinstance(error, FinalizationError) else "finalization_io_failed"
        failure = {
            "schema_version": "fcooper_orin_evidence_manifest_v1",
            "status": "failure",
            "evidence_grade": "Invalid",
            "failure_code": code,
            "error": str(error),
            "manifest_path": str(manifest),
            "artifacts": dict(sorted(artifacts.items())),
        }
        try:
            _atomic_json(evidence_path, failure)
        except OSError:
            pass
        if isinstance(error, FinalizationError):
            raise
        raise FinalizationError(code, str(error)) from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evidence = finalize(args.manifest, args.result_root, args.output_dir)
    except FinalizationError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
