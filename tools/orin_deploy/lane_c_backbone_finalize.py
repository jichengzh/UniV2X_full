#!/usr/bin/env python3
"""Assemble the Lane C backbone parity table without invalid speedup claims."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CHECKPOINT_SHA256 = (
    "d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279"
)
EXPECTED_SCOPE = "pyramid_get_multiscale_feature_engine_compute_no_data_transfer"
EXPECTED_AP_BRIDGE = "stage3_trt_multiscale_ap_bridge_v3"
EXPECTED_AP_ROUTE = (
    "HEAL spatial_features -> TensorRT multiscale outputs -> "
    "PyTorch fusion/head/postprocess"
)


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _validate_latency_report(report: dict[str, Any], *, h800: bool) -> None:
    if h800:
        _require_equal("H800 input_shape", report.get("input_shape"), [2, 64, 128, 256])
        for field, expected in (("warmup", 20), ("iters", 300), ("repeat", 5)):
            _require_equal(f"H800 {field}", report.get(field), expected)
        caliber = str(report.get("caliber", ""))
        for marker in ("backbone-subnet", "batch=2", "warmup20/iters300/repeat5", "CUDA-event"):
            if marker not in caliber:
                raise ValueError(f"H800 caliber is missing {marker!r}")
        return

    _require_equal("Orin scope", report.get("scope"), EXPECTED_SCOPE)
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Orin protocol must be present")
    expected_protocol = {
        "agent_batch": 2,
        "warmup": 20,
        "iters": 300,
        "repeat": 5,
        "timing": "CUDA_event",
        "data_transfer_inside_timed_region": False,
    }
    for field, expected in expected_protocol.items():
        _require_equal(f"Orin {field}", protocol.get(field), expected)


def _latency_row(report: dict[str, Any], *, h800: bool) -> dict[str, Any]:
    _validate_latency_report(report, h800=h800)
    if h800:
        return {
            "median_ms": report["lat_p50_ms"],
            "p90_ms": report.get("lat_p90_ms"),
            "p99_ms": report.get("lat_p99_ms"),
            "mean_ms": report.get("lat_mean_ms"),
            "warmup": report.get("warmup"),
            "iters": report.get("iters"),
            "repeat": report.get("repeat"),
        }
    protocol = report["protocol"]
    return {
        "median_ms": report["median_ms"],
        "p90_ms": report.get("p90_ms"),
        "p99_ms": report.get("p99_ms"),
        "mean_ms": report.get("mean_ms"),
        "warmup": protocol["warmup"],
        "iters": protocol["iters"],
        "repeat": protocol["repeat"],
    }


def _validate_ap_report(
    report: dict[str, Any],
    *,
    precision: str,
    expected_engine_sha256: str,
) -> None:
    required = {
        "schema": EXPECTED_AP_BRIDGE,
        "status": "success",
        "precision_tag": precision,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "engine_sha256": expected_engine_sha256,
        "processed_samples": 1789,
        "failed_samples": 0,
        "fallback_samples": 0,
        "full_ap_min_samples": 1789,
        "engine_ap_claim": True,
    }
    for field, expected in required.items():
        _require_equal(f"AP {precision} {field}", report.get(field), expected)
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"AP {precision} protocol must be present")
    for field, expected in {
        "bridge": EXPECTED_AP_BRIDGE,
        "engine_batch": 2,
        "full_ap_min_samples": 1789,
        "input_route": EXPECTED_AP_ROUTE,
    }.items():
        _require_equal(f"AP {precision} protocol.{field}", protocol.get(field), expected)


def _validate_execution_audit(
    audit: dict[str, Any] | None,
    *,
    fp16_ap: dict[str, Any],
    int8_ap: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(audit, dict):
        raise ValueError("direct Orin execution audit is required for AP promotion")
    required = {
        "schema": "lane_c_direct_orin_ap_execution_audit_v1",
        "execution_mode": "direct_on_orin_same_process",
        "replacement_scope": "get_multiscale_feature_only",
        "engine_file_role": "local_orin_execution",
        "rpc_used": False,
    }
    for field, expected in required.items():
        _require_equal(f"execution audit {field}", audit.get(field), expected)
    host = audit.get("execution_host")
    if not isinstance(host, dict):
        raise ValueError("execution audit execution_host must be present")
    address = str(host.get("address") or "").strip()
    hostname = str(host.get("hostname") or "").strip()
    _require_equal("execution audit host.architecture", host.get("architecture"), "aarch64")
    if not address or not hostname:
        raise ValueError("execution audit host address and hostname must be present")
    _require_equal(
        "execution audit backbone_execution_location",
        audit.get("backbone_execution_location"),
        f"{address}_orin_tensorrt",
    )
    _require_equal(
        "execution audit downstream_execution_location",
        audit.get("downstream_execution_location"),
        f"{address}_same_process_pytorch",
    )
    report_audits = audit.get("ap_reports")
    if not isinstance(report_audits, dict):
        raise ValueError("execution audit ap_reports must be present")
    for precision, report in (("fp16", fp16_ap), ("int8", int8_ap)):
        item = report_audits.get(precision)
        if not isinstance(item, dict):
            raise ValueError(f"execution audit ap_reports.{precision} must be present")
        _require_equal(
            f"execution audit {precision} report_sha256",
            item.get("report_sha256"),
            report.get("_report_sha256"),
        )
        _require_equal(
            f"execution audit {precision} engine_sha256",
            item.get("engine_sha256"),
            report.get("engine_sha256"),
        )
    if not audit.get("_audit_sha256"):
        raise ValueError("execution audit SHA must be present")
    return {
        "audit_sha256": audit["_audit_sha256"],
        "execution_mode": audit["execution_mode"],
        "execution_host": dict(host),
        "replacement_scope": audit["replacement_scope"],
        "engine_file_role": audit["engine_file_role"],
        "backbone_execution_location": audit["backbone_execution_location"],
        "downstream_execution_location": audit["downstream_execution_location"],
        "rpc_used": audit["rpc_used"],
    }


def _ap_pair(
    fp16_ap: dict[str, Any] | None,
    int8_ap: dict[str, Any] | None,
    *,
    fp16_engine_sha256: str,
    int8_engine_sha256: str,
    execution_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    if fp16_ap is None or int8_ap is None:
        return {
            "status": "pending_or_blocked",
            "fp16": fp16_ap,
            "int8": int8_ap,
            "delta_int8_minus_fp16": None,
        }
    _validate_ap_report(
        fp16_ap,
        precision="fp16",
        expected_engine_sha256=fp16_engine_sha256,
    )
    _validate_ap_report(
        int8_ap,
        precision="int8",
        expected_engine_sha256=int8_engine_sha256,
    )
    if fp16_ap.get("dataset_split_sha256") != int8_ap.get("dataset_split_sha256"):
        raise ValueError("AP dataset_split_sha256 must match between FP16 and INT8")
    execution = _validate_execution_audit(
        execution_audit,
        fp16_ap=fp16_ap,
        int8_ap=int8_ap,
    )
    fp16_values = {"ap50": float(fp16_ap["ap50"]), "ap70": float(fp16_ap["ap70"])}
    int8_values = {"ap50": float(int8_ap["ap50"]), "ap70": float(int8_ap["ap70"])}
    return {
        "status": "paired_orin_full_1789",
        "fp16": fp16_values,
        "int8": int8_values,
        "delta_int8_minus_fp16": {
            "ap50": round(int8_values["ap50"] - fp16_values["ap50"], 12),
            "ap70": round(int8_values["ap70"] - fp16_values["ap70"], 12),
        },
        "provenance": {
            "bridge": EXPECTED_AP_BRIDGE,
            "processed_samples": 1789,
            "fp16_engine_sha256": fp16_engine_sha256,
            "int8_engine_sha256": int8_engine_sha256,
            "dataset_split_sha256": fp16_ap["dataset_split_sha256"],
            **execution,
        },
    }


def build_comparison(
    *,
    calibration_status: str,
    h800_fp16: dict[str, Any],
    h800_int8: dict[str, Any],
    orin_fp16: dict[str, Any],
    orin_int8: dict[str, Any],
    fp16_ap: dict[str, float] | None,
    int8_ap: dict[str, float] | None,
    execution_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return raw observations while keeping non-equivalent power rails separate."""
    identical = calibration_status == "identical"
    return {
        "schema": "lane_c_backbone_cross_hardware_comparison_v1",
        "scope": "Pyramid (16,32,64) get_multiscale_feature/backbone subnet",
        "calibration_status": calibration_status,
        "evidence_grade": "complete_parity" if identical else "degraded",
        "latency": {
            "protocol": {
                "batch": 2,
                "warmup": 20,
                "iters": 300,
                "repeat": 5,
                "timer": "CUDA event",
                "boundary": "engine-compute/no-data-transfer",
            },
            "h800_fp16": _latency_row(h800_fp16, h800=True),
            "h800_int8": _latency_row(h800_int8, h800=True),
            "orin_fp16": _latency_row(orin_fp16, h800=False),
            "orin_int8": _latency_row(orin_int8, h800=False),
            "cross_hardware_ratio_policy": "not_reported",
        },
        "power": {
            "comparison_policy": "interfaces are not the same physical quantity",
            "h800": {
                "measurement_interface": "NVML board power",
                "fp16_mean_w": h800_fp16.get("watt_avg"),
                "int8_mean_w": h800_int8.get("watt_avg"),
            },
            "orin": {
                "measurement_interface": "tegrastats VIN_SYS_5V0 rail",
                "fp16_mean_w": orin_fp16.get("power_measurement", {}).get(
                    "vin_sys_5v0_mean_w"
                ),
                "int8_mean_w": orin_int8.get("power_measurement", {}).get(
                    "vin_sys_5v0_mean_w"
                ),
            },
        },
        "ap": _ap_pair(
            fp16_ap,
            int8_ap,
            fp16_engine_sha256=orin_fp16["engine_sha256"],
            int8_engine_sha256=orin_int8["engine_sha256"],
            execution_audit=execution_audit,
        ),
        "conclusion_policy": (
            "No cross-hardware acceleration claim because calibration payloads "
            "are not byte-identical."
            if not identical
            else "Raw device observations remain subject to TensorRT-version audit."
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_within(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"{label} must remain under artifact root {resolved_root}")
    return resolved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fp16-ap-report", type=Path)
    parser.add_argument("--int8-ap-report", type=Path)
    parser.add_argument("--execution-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = _require_within(root, args.output, label="output")
    if bool(args.fp16_ap_report) != bool(args.int8_ap_report):
        raise ValueError(
            "--fp16-ap-report and --int8-ap-report must be provided together"
        )
    if bool(args.fp16_ap_report) != bool(args.execution_audit):
        raise ValueError(
            "--execution-audit is required exactly when AP reports are provided"
        )
    fp16_ap_path = (
        _require_within(root, args.fp16_ap_report, label="fp16-ap-report")
        if args.fp16_ap_report
        else None
    )
    int8_ap_path = (
        _require_within(root, args.int8_ap_report, label="int8-ap-report")
        if args.int8_ap_report
        else None
    )
    execution_audit_path = (
        _require_within(root, args.execution_audit, label="execution-audit")
        if args.execution_audit
        else None
    )
    fp16_ap = (
        {
            **_read_json(fp16_ap_path),
            "_report_sha256": _sha256_file(fp16_ap_path),
        }
        if fp16_ap_path
        else None
    )
    int8_ap = (
        {
            **_read_json(int8_ap_path),
            "_report_sha256": _sha256_file(int8_ap_path),
        }
        if int8_ap_path
        else None
    )
    execution_audit = (
        {
            **_read_json(execution_audit_path),
            "_audit_sha256": _sha256_file(execution_audit_path),
        }
        if execution_audit_path
        else None
    )
    comparison = build_comparison(
        calibration_status=_read_json(
            root / "audits/calibration_identity_audit.json"
        )["status"],
        h800_fp16=_read_json(root / "h800_reference/fp16_trt_profile_result.json"),
        h800_int8=_read_json(root / "h800_reference/int8_trt_profile_result.json"),
        orin_fp16=_read_json(root / "orin/fp16/primary_latency_power.json"),
        orin_int8=_read_json(root / "orin/int8/primary_latency_power.json"),
        fp16_ap=fp16_ap,
        int8_ap=int8_ap,
        execution_audit=execution_audit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
