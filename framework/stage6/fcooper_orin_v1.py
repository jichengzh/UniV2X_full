"""Pure contracts and metric helpers for the F-Cooper Orin evidence lane."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CONFIGURATIONS: Mapping[str, Mapping[str, Any]] = {
    "original": {
        "width": (64, 128, 256, 128, 256),
        "precision": "fp32",
        "backend": "pytorch_cuda",
        "builder_level": None,
    },
    "compression_only": {
        "width": (32, 64, 64, 32, 64),
        "precision": "fp16",
        "backend": "tensorrt",
        "builder_level": 0,
    },
    "schedule_only": {
        "width": (64, 128, 256, 128, 256),
        "precision": "fp32",
        "backend": "tensorrt",
        "builder_level": 5,
    },
    "compress_then_tune": {
        "width": (64, 64, 64, 32, 64),
        "precision": "fp16",
        "backend": "tensorrt",
        "builder_level": 5,
    },
    "joint_fp16_control": {
        "width": (32, 32, 64, 32, 64),
        "precision": "fp16",
        "backend": "tensorrt",
        "builder_level": 5,
    },
}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def summarize_latency(repeats: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Validate and summarize the frozen 5 x 300 latency protocol."""
    if len(repeats) != 5:
        raise ValueError("latency protocol requires exactly five repeats")
    arrays = []
    per_repeat = []
    for repeat_index, values in enumerate(repeats):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (300,):
            raise ValueError("each latency repeat must contain exactly 300 samples")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("latency samples must be finite and non-negative")
        arrays.append(array)
        per_repeat.append({"repeat": repeat_index, **_summary(array)})
    aggregate = np.concatenate(arrays)
    return {
        "protocol": {
            "warmup": 20,
            "iterations": 300,
            "repeat": 5,
            "timing": "cuda_event",
            "scope": "compute_only_no_data_transfer",
            "sample_batch": 1,
            "dense_agent_batch": 5,
        },
        "sample_count": int(aggregate.size),
        "repeats": per_repeat,
        "aggregate": _summary(aggregate),
    }


def numeric_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Compute the frozen numerical-comparison fields without altering dtype."""
    reference_array = np.asarray(reference)
    candidate_array = np.asarray(candidate)
    if reference_array.shape != candidate_array.shape:
        raise ValueError(
            f"shape mismatch: {reference_array.shape} != {candidate_array.shape}"
        )
    ref = reference_array.astype(np.float64, copy=False).ravel()
    cand = candidate_array.astype(np.float64, copy=False).ravel()
    finite = np.isfinite(ref) & np.isfinite(cand)
    finite_ratio = float(np.mean(finite)) if finite.size else 1.0
    if not finite.all():
        ref = ref[finite]
        cand = cand[finite]
    difference = cand - ref
    ref_norm = float(np.linalg.norm(ref))
    candidate_norm = float(np.linalg.norm(cand))
    denominator = ref_norm * candidate_norm
    cosine = float(np.dot(ref, cand) / denominator) if denominator else 1.0
    rmse = float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0
    reference_rms = float(np.sqrt(np.mean(np.square(ref)))) if ref.size else 0.0
    nrmse = rmse / max(reference_rms, np.finfo(np.float64).eps)
    return {
        "shape": list(reference_array.shape),
        "reference_dtype": str(reference_array.dtype),
        "candidate_dtype": str(candidate_array.dtype),
        "cosine": cosine,
        "nrmse": float(nrmse),
        "mae": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "finite_ratio": finite_ratio,
        "reference_min": float(np.min(ref)) if ref.size else 0.0,
        "reference_max": float(np.max(ref)) if ref.size else 0.0,
        "candidate_min": float(np.min(cand)) if cand.size else 0.0,
        "candidate_max": float(np.max(cand)) if cand.size else 0.0,
        "reference_zero_ratio": float(np.mean(ref == 0)) if ref.size else 1.0,
        "candidate_zero_ratio": float(np.mean(cand == 0)) if cand.size else 1.0,
    }


_POWER_PATTERN = re.compile(
    r"(?P<rail>VIN_SYS_5V0|VDD_GPU_SOC)\s+(?P<current>\d+)mW/\d+mW"
)


def parse_tegrastats_power(text: str) -> dict[str, Any]:
    """Parse active-window rail samples; never combine physical rails."""
    samples: dict[str, list[float]] = {"VIN_SYS_5V0": [], "VDD_GPU_SOC": []}
    for line in text.splitlines():
        for match in _POWER_PATTERN.finditer(line):
            samples[match.group("rail")].append(float(match.group("current")) / 1000)
    if not samples["VIN_SYS_5V0"]:
        raise ValueError("VIN_SYS_5V0 rail evidence is absent")
    return {
        "sample_count": len(samples["VIN_SYS_5V0"]),
        "vin_sys_5v0_mean_w": float(np.mean(samples["VIN_SYS_5V0"])),
        "vdd_gpu_soc_mean_w": (
            float(np.mean(samples["VDD_GPU_SOC"]))
            if samples["VDD_GPU_SOC"]
            else None
        ),
        "rails_combined": False,
    }


def energy_joules(vin_sys_5v0_mean_w: float, median_latency_ms: float) -> float:
    if vin_sys_5v0_mean_w < 0 or median_latency_ms < 0:
        raise ValueError("power and latency must be non-negative")
    return float(vin_sys_5v0_mean_w * median_latency_ms / 1000)


def validate_inspector_precision(
    inspector_text: str, *, expected: str, tf32_disabled: bool
) -> dict[str, Any]:
    """Fail closed on precision evidence extracted from a TRT inspector dump."""
    normalized = inspector_text.lower()
    forbidden_fp32 = {
        "half": "fp16",
        "fp16": "fp16",
        "int8": "int8",
        "bfloat16": "bf16",
        "bf16": "bf16",
        "tf32": "tf32",
    }
    findings = sorted(
        label for token, label in forbidden_fp32.items() if token in normalized
    )
    if expected == "fp32":
        valid = (
            tf32_disabled
            and not findings
            and ("float" in normalized or "fp32" in normalized)
        )
    elif expected == "fp16":
        valid = (
            ("half" in normalized or "fp16" in normalized)
            and "int8" not in normalized
        )
    else:
        raise ValueError(f"unsupported expected precision: {expected}")
    return {
        "expected": expected,
        "tf32_disabled": bool(tf32_disabled),
        "reduced_precision_findings": findings,
        "valid": bool(valid),
    }
