"""Pure contract helpers for Lane C Orin build and measurement evidence."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Mapping, Sequence

import numpy as np


_TRT_PATTERNS = {
    "median_ms": r"GPU Compute Time:.*?\bmedian\s*=\s*([\d.]+)\s*ms",
    "p90_ms": r"GPU Compute Time:.*?percentile\(90%\)\s*=\s*([\d.]+)\s*ms",
    "p99_ms": r"GPU Compute Time:.*?percentile\(99%\)\s*=\s*([\d.]+)\s*ms",
    "mean_ms": r"GPU Compute Time:.*?\bmean\s*=\s*([\d.]+)\s*ms",
    "throughput_qps": r"Throughput:\s*([\d.]+)\s*qps",
    "host_walltime_s": r"Total Host Walltime:\s*([\d.]+)\s*s",
    "gpu_compute_total_s": r"Total GPU Compute Time:\s*([\d.]+)\s*s",
}


def parse_trtexec(output: str) -> dict[str, float]:
    """Parse the complete per-process trtexec summary required by Lane C."""
    parsed: dict[str, float] = {}
    for field, pattern in _TRT_PATTERNS.items():
        match = re.search(pattern, output, flags=re.IGNORECASE | re.DOTALL)
        if match is not None:
            parsed[field] = float(match.group(1))
    missing = sorted(set(_TRT_PATTERNS) - set(parsed))
    if missing:
        raise ValueError(f"missing trtexec metrics: {', '.join(missing)}")
    return parsed


def parse_tegrastats(output: str) -> dict[str, Any]:
    """Parse instantaneous power/utilization samples from tegrastats output."""
    patterns = {
        "vin_sys_5v0_w": re.compile(r"VIN_SYS_5V0\s+(\d+)mW"),
        "vdd_gpu_soc_w": re.compile(r"VDD_GPU_SOC\s+(\d+)mW"),
        "gr3d_percent": re.compile(r"GR3D_FREQ\s+([\d.]+)%"),
        "gpu_temperature_c": re.compile(r"GPU@([-\d.]+)C"),
    }
    samples: dict[str, list[float]] = {name: [] for name in patterns}
    for line in output.splitlines():
        for name, pattern in patterns.items():
            match = pattern.search(line)
            if match is None:
                continue
            value = float(match.group(1))
            if name.endswith("_w"):
                value /= 1000.0
            samples[name].append(value)
    count = len(samples["vin_sys_5v0_w"])
    if count == 0:
        raise ValueError("tegrastats contains no VIN_SYS_5V0 samples")
    return {
        **samples,
        "sample_count": count,
        "vin_sys_5v0_mean_w": statistics.fmean(samples["vin_sys_5v0_w"]),
        "vdd_gpu_soc_mean_w": (
            statistics.fmean(samples["vdd_gpu_soc_w"])
            if samples["vdd_gpu_soc_w"]
            else None
        ),
    }


def _cosine_similarity(reference: np.ndarray, actual: np.ndarray) -> float:
    ref = reference.astype(np.float64, copy=False).reshape(-1)
    got = actual.astype(np.float64, copy=False).reshape(-1)
    denominator = float(np.linalg.norm(ref) * np.linalg.norm(got))
    if denominator == 0.0:
        return 1.0 if np.array_equal(ref, got) else 0.0
    return float(np.dot(ref, got) / denominator)


def compare_outputs(
    reference: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    precision: str,
) -> dict[str, Any]:
    """Compare engine outputs with explicit, precision-specific sanity limits."""
    thresholds_by_precision = {
        "fp32": {
            "cosine_similarity_min": 0.999,
            "normalized_rmse_max": 0.01,
            "mean_abs_error_max": 0.005,
        },
        "fp16": {
            "cosine_similarity_min": 0.998,
            "normalized_rmse_max": 0.02,
            "mean_abs_error_max": 0.02,
        },
        "int8": {
            "cosine_similarity_min": 0.99,
            "normalized_rmse_max": 0.1,
            "mean_abs_error_max": 0.2,
        },
    }
    normalized_precision = precision.lower()
    if normalized_precision not in thresholds_by_precision:
        raise ValueError(f"unsupported precision: {precision}")
    if set(reference) != set(actual):
        raise ValueError(
            f"output names differ: reference={sorted(reference)} actual={sorted(actual)}"
        )
    thresholds = thresholds_by_precision[normalized_precision]
    per_output: dict[str, dict[str, Any]] = {}
    all_passed = True
    for name in sorted(reference):
        ref = np.asarray(reference[name])
        got = np.asarray(actual[name])
        if ref.shape != got.shape:
            raise ValueError(f"shape mismatch for {name}: {ref.shape} != {got.shape}")
        finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
        if finite:
            difference = got.astype(np.float64) - ref.astype(np.float64)
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            reference_rms = float(np.sqrt(np.mean(np.square(ref.astype(np.float64)))))
            normalized_rmse = rmse / max(reference_rms, 1e-12)
            mean_abs_error = float(np.mean(np.abs(difference)))
            max_abs_error = float(np.max(np.abs(difference)))
            cosine = _cosine_similarity(ref, got)
        else:
            normalized_rmse = math.inf
            mean_abs_error = math.inf
            max_abs_error = math.inf
            cosine = -1.0
        output_passed = (
            finite
            and cosine >= thresholds["cosine_similarity_min"]
            and normalized_rmse <= thresholds["normalized_rmse_max"]
            and mean_abs_error <= thresholds["mean_abs_error_max"]
        )
        all_passed = all_passed and output_passed
        per_output[name] = {
            "finite": finite,
            "cosine_similarity": cosine,
            "normalized_rmse": normalized_rmse,
            "mean_abs_error": mean_abs_error,
            "max_abs_error": max_abs_error,
            "passed": output_passed,
        }
    return {
        "precision": normalized_precision,
        "thresholds": thresholds,
        "per_output": per_output,
        "passed": all_passed,
    }


def assess_server_reference_sanity(
    reference: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    precision: str,
) -> dict[str, Any]:
    """Apply scale-independent sanity limits to target-vs-server outputs."""
    thresholds_by_precision = {
        "fp32": {"cosine_similarity_min": 0.999, "normalized_rmse_max": 0.02},
        "fp16": {"cosine_similarity_min": 0.998, "normalized_rmse_max": 0.03},
        "int8": {"cosine_similarity_min": 0.99, "normalized_rmse_max": 0.1},
    }
    normalized_precision = precision.lower()
    if normalized_precision not in thresholds_by_precision:
        raise ValueError(f"unsupported precision: {precision}")
    diagnostic = compare_outputs(reference, actual, precision=normalized_precision)
    thresholds = thresholds_by_precision[normalized_precision]
    per_output: dict[str, dict[str, Any]] = {}
    for name, metrics in diagnostic["per_output"].items():
        output_passed = bool(
            metrics["finite"]
            and metrics["cosine_similarity"] >= thresholds["cosine_similarity_min"]
            and metrics["normalized_rmse"] <= thresholds["normalized_rmse_max"]
        )
        per_output[name] = {**metrics, "passed": output_passed}
    return {
        "precision": normalized_precision,
        "thresholds": thresholds,
        "per_output": per_output,
        "passed": all(metrics["passed"] for metrics in per_output.values()),
        "note": (
            "Sanity uses scale-independent cosine and normalized RMSE. Absolute "
            "error remains available in the strict diagnostic report."
        ),
    }


def assess_deployment_numerical_gate(
    *,
    cross_backend_report: Mapping[str, Any],
    same_device_report: Mapping[str, Any] | None,
    current_onnx_sha256: str,
    reference_onnx_sha256: str | None,
) -> dict[str, Any]:
    """Require server-reference sanity; same-device evidence is corroborative."""
    if bool(cross_backend_report.get("passed")):
        return {
            "passed": True,
            "basis": "server_reference_sanity",
            "same_onnx_source": (
                reference_onnx_sha256 == current_onnx_sha256
                if reference_onnx_sha256
                else None
            ),
        }

    same_onnx_source = bool(
        reference_onnx_sha256
        and current_onnx_sha256
        and reference_onnx_sha256 == current_onnx_sha256
    )
    reproducibility_limits = {
        "cosine_similarity_min": 0.999999,
        "normalized_rmse_max": 1e-6,
    }
    outputs = (
        same_device_report.get("per_output", {})
        if isinstance(same_device_report, Mapping)
        else {}
    )
    reproducible = bool(outputs) and all(
        bool(metrics.get("finite"))
        and float(metrics.get("cosine_similarity", -1.0))
        >= reproducibility_limits["cosine_similarity_min"]
        and float(metrics.get("normalized_rmse", math.inf))
        <= reproducibility_limits["normalized_rmse_max"]
        for metrics in outputs.values()
    )
    return {
        "passed": False,
        "basis": "server_reference_sanity_failed",
        "same_onnx_source": same_onnx_source,
        "same_device_reproducible": reproducible,
        "reproducibility_limits": reproducibility_limits,
    }


def assess_no_execution_fallback(
    *,
    requested_precision: str,
    builder_flags: Sequence[str],
    execution_provider: str,
    external_fallback_count: int,
    layer_precision_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Audit whole-engine execution without hiding legal TRT mixed tactics."""
    precision = requested_precision.lower()
    normalized_flags = {flag.lower() for flag in builder_flags}
    requested_flag_present = precision == "fp32" or precision in normalized_flags
    native_engine = execution_provider == "tensorrt_engine"
    no_external_fallback = external_fallback_count == 0
    mixed_inside_engine = sum(
        1 for value in layer_precision_counts.values() if int(value) > 0
    ) > 1
    return {
        "requested_precision": precision,
        "builder_flags": sorted(normalized_flags),
        "execution_provider": execution_provider,
        "external_fallback_count": int(external_fallback_count),
        "layer_precision_counts": {
            name: int(value) for name, value in layer_precision_counts.items()
        },
        "mixed_precision_inside_engine": mixed_inside_engine,
        "contract_note": (
            "No execution may leave the TensorRT engine. Internal TensorRT mixed "
            "tactics are reported and are not relabeled as pure INT8 coverage."
        ),
        "passed": requested_flag_present and native_engine and no_external_fallback,
    }


def summarize_repeat_medians(medians_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize drift across independent trtexec process medians."""
    values = [float(value) for value in medians_ms]
    if not values:
        raise ValueError("at least one repeat median is required")
    mean_value = statistics.fmean(values)
    cv_percent = (
        statistics.pstdev(values) / mean_value * 100.0 if mean_value else math.inf
    )
    return {
        "repeat_count": len(values),
        "median_of_medians_ms": statistics.median(values),
        "mean_of_medians_ms": mean_value,
        "cv_percent": cv_percent,
    }
