#!/usr/bin/env python3
"""Fail-closed provenance validation for CoDriving INT8 AP evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Set
from typing import Any


INT8_CALIBRATION_SCHEMA = "codriving_routeb_int8_scale_manifest_v1"
INT8_QUANTIZATION_SEMANTICS = "symmetric_absmax_int8_dequant_fp32"
EXPECTED_CALIBRATION_SAMPLES = 16
EXPECTED_SPATIAL_SAMPLE_SHAPE = (2, 64, 256, 512)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _exact_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def validate_calibration_tensor_shape(shape: Any) -> tuple[int, ...]:
    normalized = tuple(
        _exact_integer(value, f"spatial_features_shape[{index}]")
        for index, value in enumerate(shape)
    )
    if not normalized or normalized[0] != EXPECTED_CALIBRATION_SAMPLES:
        raise ValueError(
            f"INT8 calibration requires exactly {EXPECTED_CALIBRATION_SAMPLES} samples, got {normalized}"
        )
    expected = (EXPECTED_CALIBRATION_SAMPLES, *EXPECTED_SPATIAL_SAMPLE_SHAPE)
    if normalized != expected:
        raise ValueError(f"unexpected spatial_features calibration shape: {normalized} != {expected}")
    return normalized


def validate_int8_calibration_manifest(
    manifest: Any,
    *,
    required_signatures: Set[str] | set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("INT8 calibration manifest must be a mapping")
    if manifest.get("schema") != INT8_CALIBRATION_SCHEMA:
        raise ValueError(f"unexpected INT8 calibration schema: {manifest.get('schema')!r}")
    if manifest.get("quantization_semantics") != INT8_QUANTIZATION_SEMANTICS:
        raise ValueError("unexpected INT8 quantization semantics")
    if manifest.get("calibration_split") != "train":
        raise ValueError("INT8 calibration_split must be train")
    calibration_samples = _exact_integer(
        manifest.get("calibration_samples"),
        "calibration_samples",
    )
    if calibration_samples != EXPECTED_CALIBRATION_SAMPLES:
        raise ValueError(f"INT8 calibration_samples must be {EXPECTED_CALIBRATION_SAMPLES}")
    validate_calibration_tensor_shape(manifest.get("spatial_features_shape") or ())
    if not str(manifest.get("calibration_source") or ""):
        raise ValueError("INT8 calibration_source is required")
    if not str(manifest.get("calibration_summary") or ""):
        raise ValueError("INT8 calibration_summary is required")
    if not str(manifest.get("calibration_split_source") or ""):
        raise ValueError("INT8 calibration_split_source is required")
    for field in ("calibration_source_sha256", "calibration_summary_sha256"):
        value = str(manifest.get(field) or "")
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field} must be a lowercase SHA256 digest")
    qmin = _exact_integer(manifest.get("qmin"), "qmin")
    qmax = _exact_integer(manifest.get("qmax"), "qmax")
    if qmin != -127 or qmax != 127:
        raise ValueError("INT8 calibration qrange must be [-127, 127]")

    scales = manifest.get("scales_by_signature")
    if not isinstance(scales, Mapping) or not scales:
        raise ValueError("INT8 calibration scales_by_signature must be nonempty")
    for signature, values in scales.items():
        if not str(signature) or not isinstance(values, Mapping):
            raise ValueError(f"invalid INT8 calibration scale record: {signature!r}")
        for field in ("input_scale", "weight_scale"):
            value = float(values.get(field) or 0.0)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{signature} has invalid {field}: {value}")

    missing = sorted(set(required_signatures) - set(scales))
    if missing:
        raise ValueError(f"missing calibrated signatures: {missing}")
    return dict(manifest)


def report_has_valid_int8_calibration(payload: Mapping[str, Any]) -> bool:
    precision = str(payload.get("precision") or "")
    if precision == "fp16":
        return True
    if precision not in {"int8", "mixed"}:
        return False
    compile_summary = payload.get("compile_summary")
    if not isinstance(compile_summary, Mapping):
        return False
    if compile_summary.get("quantization_semantics") != INT8_QUANTIZATION_SEMANTICS:
        return False
    precision_plan = compile_summary.get("conv_precision_plan")
    signature_plan = compile_summary.get("int8_signature_plan")
    if not isinstance(precision_plan, Mapping) or not isinstance(signature_plan, Mapping):
        return False
    int8_names = {str(name) for name, value in precision_plan.items() if value == "int8"}
    if not int8_names or not int8_names.issubset(set(signature_plan)):
        return False
    required_signatures = {str(signature_plan[name]) for name in int8_names}
    try:
        validate_int8_calibration_manifest(
            compile_summary.get("int8_calibration"),
            required_signatures=required_signatures,
        )
    except (TypeError, ValueError):
        return False
    return True
