"""Helpers for Stage2 native INT8 route evidence contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


NATIVE_INT8_QUANT_METHOD = "h800_tvm_native_int8_backbone_subnet"
NATIVE_INT8_QUANT_SCOPE = "backbone_subnet_native_int8"
REQUIRED_RESULT_FILES = (
    "latency_result.json",
    "energy_result.json",
    "idle_power_samples.csv",
    "active_power_samples.csv",
    "telemetry_payload.json",
)
REQUIRED_BLOCKER_ATTEMPTS = (
    "tiny_int8_conv",
    "resnet_style_int8_block",
    "target_backbone_subnet_route",
)
REQUIRED_BLOCKER_FIELDS = (
    "attempt_json",
    "stdout_path",
    "stderr_path",
    "traceback",
    "build_status",
    "run_status",
    "failure_reason",
)


class NativeInt8RouteError(ValueError):
    """Raised when native INT8 route evidence is incomplete or overclaimed."""


def expected_label_output_paths(run_root: Path, label: str) -> list[Path]:
    label_dir = Path(run_root) / label
    return [label_dir / name for name in REQUIRED_RESULT_FILES]


def _require_text(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise NativeInt8RouteError(f"missing required field: {field}")
    return value


def validate_native_route_record(record: dict[str, Any]) -> None:
    quant_method = _require_text(record, "quant_method")
    if quant_method == "static_qdq_synthetic_minmax" or "qdq" in quant_method.lower():
        raise NativeInt8RouteError(f"static_qdq route is not native INT8: {quant_method}")
    if quant_method != NATIVE_INT8_QUANT_METHOD:
        raise NativeInt8RouteError(
            f"unexpected native INT8 quant_method: {quant_method}; "
            f"expected {NATIVE_INT8_QUANT_METHOD}"
        )
    quant_scope = _require_text(record, "quant_scope")
    if quant_scope != NATIVE_INT8_QUANT_SCOPE:
        raise NativeInt8RouteError(
            f"unexpected native INT8 quant_scope: {quant_scope}; expected {NATIVE_INT8_QUANT_SCOPE}"
        )
    if bool(record.get("full_network_claim")):
        raise NativeInt8RouteError("native INT8 smoke rows must keep full_network_claim=false")
    for field in (
        "latency_result_path",
        "energy_result_path",
        "idle_power_samples_path",
        "active_power_samples_path",
        "telemetry_payload_path",
    ):
        _require_text(record, field)


def validate_blocker_attempt_summary(summary: dict[str, Any]) -> None:
    attempts = summary.get("attempts")
    if not isinstance(attempts, dict):
        raise NativeInt8RouteError("blocker summary must contain attempts object")
    for attempt_name in REQUIRED_BLOCKER_ATTEMPTS:
        attempt = attempts.get(attempt_name)
        if not isinstance(attempt, dict):
            raise NativeInt8RouteError(f"missing blocker attempt: {attempt_name}")
        for field in REQUIRED_BLOCKER_FIELDS:
            _require_text(attempt, field)
