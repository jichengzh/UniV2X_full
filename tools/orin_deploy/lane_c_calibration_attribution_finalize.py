#!/usr/bin/env python3
"""Validate and summarize the Lane C calibration-locked attribution run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ONNX_SHA256 = (
    "8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279"
)
EXPECTED_SPLIT_SHA256 = (
    "f6805e26aec6af0f994395ad8c74105e65c88770bc7261bd7e995e30d47e2814"
)
EXPECTED_SCOPE = "pyramid_get_multiscale_feature_engine_compute_no_data_transfer"
EXPECTED_AP_BRIDGE = "stage3_trt_multiscale_ap_bridge_v3"


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, got {actual!r}")


def classify_device_ap(
    *,
    fp16: dict[str, float],
    int8: dict[str, float],
) -> str:
    fp50, fp70 = float(fp16["ap50"]), float(fp16["ap70"])
    i50, i70 = float(int8["ap50"]), float(int8["ap70"])
    if i50 - fp50 >= -0.01 and i70 - fp70 >= -0.01:
        return "maintained"
    ratios = (
        i50 / fp50 if fp50 > 0.0 else float("inf"),
        i70 / fp70 if fp70 > 0.0 else float("inf"),
    )
    if i50 <= 0.05 and i70 <= 0.05 and max(ratios) <= 0.10:
        return "collapsed"
    return "mixed"


def attribute_ap_outcomes(h800: str, orin: str) -> str:
    matrix = {
        ("collapsed", "collapsed"): (
            "shared_rebuilt_calibration_supported_as_common_trigger"
        ),
        ("maintained", "collapsed"): (
            "tensorrt_runtime_build_difference_audit_required"
        ),
        ("collapsed", "maintained"): (
            "calibration_runtime_interaction_unresolved"
        ),
        ("maintained", "maintained"): (
            "prior_orin_collapse_not_reproduced"
        ),
    }
    return matrix.get((h800, orin), "mixed_or_ambiguous")


def _validate_receipts(receipts: dict[str, dict[str, Any]]) -> tuple[str, str]:
    _require_equal("receipt names", set(receipts), {"local", "h800", "orin"})
    manifests = set()
    payloads = set()
    for name, receipt in receipts.items():
        _require_equal(
            f"{name} receipt schema",
            receipt.get("schema_version"),
            "lane_c_calibration_receive_receipt_v1",
        )
        _require_equal(
            f"{name} receipt scope",
            receipt.get("scope"),
            "pyramid_get_multiscale_feature_16x32x64_only",
        )
        _require_equal(
            f"{name} calibration status",
            receipt.get("status"),
            "byte_identical_15_of_15",
        )
        _require_equal(
            f"{name} matched calibration files",
            receipt.get("matched_file_count"),
            15,
        )
        _require_equal(
            f"{name} expected calibration files",
            receipt.get("expected_file_count"),
            15,
        )
        _require_equal(f"{name} missing calibration files", receipt.get("missing_files"), [])
        _require_equal(f"{name} extra calibration files", receipt.get("extra_files"), [])
        rows = receipt.get("files")
        if not isinstance(rows, list) or len(rows) != 15:
            raise ValueError(f"{name} receipt files must contain exactly 15 rows")
        expected_names = [f"batch2_{index:03d}.npy" for index in range(15)]
        _require_equal(
            f"{name} receipt filenames",
            [row.get("filename") for row in rows],
            expected_names,
        )
        for index, row in enumerate(rows):
            _require_equal(
                f"{name} receipt file {index} bytes",
                row.get("bytes"),
                16777344,
            )
            _require_equal(
                f"{name} receipt file {index} shape",
                row.get("shape"),
                [2, 64, 128, 256],
            )
            _require_equal(
                f"{name} receipt file {index} dtype",
                row.get("dtype"),
                "float32",
            )
            _require_equal(
                f"{name} receipt file {index} C contiguous",
                row.get("c_contiguous"),
                True,
            )
            digest = str(row.get("sha256", ""))
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"{name} receipt file {index} has invalid SHA-256")
        host = receipt.get("host", {})
        expected_host = {
            "h800": ("zs-nj-tap-gpu18", "x86_64", "10.13.0.35"),
            "orin": ("ubuntu", "aarch64", "8.5.2.2"),
        }.get(name)
        if expected_host is not None:
            _require_equal(f"{name} receipt hostname", host.get("hostname"), expected_host[0])
            _require_equal(
                f"{name} receipt architecture", host.get("architecture"), expected_host[1]
            )
            _require_equal(
                f"{name} receipt TensorRT", host.get("tensorrt_version"), expected_host[2]
            )
        manifests.add(str(receipt.get("manifest_sha256")))
        payloads.add(str(receipt.get("payload_id")))
    if len(manifests) != 1 or len(payloads) != 1:
        raise ValueError("calibration receipts are not byte-identical")
    return manifests.pop(), payloads.pop()


def _validate_builds(
    *,
    device: str,
    builds: dict[str, dict[str, Any]],
    manifest_sha256: str,
) -> None:
    _require_equal(f"{device} build precisions", set(builds), {"fp16", "int8"})
    for precision, report in builds.items():
        _require_equal(
            f"{device} {precision} TensorRT version",
            report.get("tensorrt_version"),
            {"h800": "10.13.0.35", "orin": "8.5.2.2"}[device],
        )
        _require_equal(
            f"{device} {precision} ONNX SHA",
            report.get("onnx_sha256"),
            EXPECTED_ONNX_SHA256,
        )
        _require_equal(
            f"{device} {precision} precision",
            report.get("precision"),
            precision,
        )
        fresh = report.get("fresh_build", {})
        _require_equal(
            f"{device} {precision} preexisting outputs",
            fresh.get("preexisting_output_count"),
            0,
        )
        _require_equal(
            f"{device} {precision} calibration cache read",
            fresh.get("calibration_cache_read"),
            False,
        )
        if precision == "int8":
            _require_equal(
                f"{device} INT8 calibration manifest",
                report.get("calibration_manifest_sha256"),
                manifest_sha256,
            )
            _require_equal(
                f"{device} INT8 calibration verified files",
                report.get("calibration_verified_file_count"),
                15,
            )
            _require_equal(
                f"{device} INT8 calibration consumed files",
                report.get("calibration_consumed_file_count"),
                15,
            )
            if int(report.get("layer_precision_counts", {}).get("int8", 0)) <= 0:
                raise ValueError(f"{device} INT8 engine has no INT8 layers")


def _validate_latency(
    *,
    device: str,
    precision: str,
    report: dict[str, Any],
    engine_sha256: str,
) -> None:
    _require_equal(
        f"{device} {precision} latency scope",
        report.get("scope"),
        EXPECTED_SCOPE,
    )
    _require_equal(
        f"{device} {precision} latency engine",
        report.get("engine_sha256"),
        engine_sha256,
    )
    expected_protocol = {
        "agent_batch": 2,
        "warmup": 20,
        "iters": 300,
        "repeat": 5,
        "timing": "CUDA_event",
        "data_transfer_inside_timed_region": False,
    }
    protocol = report.get("protocol", {})
    for field, expected in expected_protocol.items():
        _require_equal(
            f"{device} {precision} latency {field}",
            protocol.get(field),
            expected,
        )
    _require_equal(
        f"{device} {precision} latency sample count",
        report.get("sample_count"),
        1500,
    )


def _validate_ap(
    *,
    device: str,
    precision: str,
    report: dict[str, Any],
    engine_sha256: str,
) -> dict[str, float]:
    expected = {
        "schema": EXPECTED_AP_BRIDGE,
        "status": "success",
        "precision_tag": precision,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "engine_sha256": engine_sha256,
        "dataset_split_sha256": EXPECTED_SPLIT_SHA256,
        "processed_samples": 1789,
        "failed_samples": 0,
        "fallback_samples": 0,
        "full_ap_min_samples": 1789,
        "engine_ap_claim": True,
    }
    for field, value in expected.items():
        _require_equal(f"{device} {precision} AP {field}", report.get(field), value)
    protocol = report.get("protocol", {})
    for field, value in {
        "bridge": EXPECTED_AP_BRIDGE,
        "engine_batch": 2,
        "eval_range": "102.4,51.2",
        "full_ap_min_samples": 1789,
        "fallback_policy": "forbidden_for_engine_ap_claim",
        "input_route": (
            "HEAL spatial_features -> TensorRT multiscale outputs -> "
            "PyTorch fusion/head/postprocess"
        ),
    }.items():
        _require_equal(
            f"{device} {precision} AP protocol.{field}",
            protocol.get(field),
            value,
        )
    return {"ap50": float(report["ap50"]), "ap70": float(report["ap70"])}


def _validate_execution_audit(
    *,
    device: str,
    audit: dict[str, Any],
    ap_reports: dict[str, dict[str, Any]],
    builds: dict[str, dict[str, Any]],
) -> None:
    expected = {
        "schema_version": "lane_c_direct_device_ap_execution_audit_v1",
        "device": device,
        "execution_mode": "direct_same_process",
        "replacement_scope": "get_multiscale_feature_only",
        "rpc_used": False,
    }
    for field, value in expected.items():
        _require_equal(f"{device} execution audit {field}", audit.get(field), value)
    host = audit.get("host", {})
    expected_host = {
        "h800": ("zs-nj-tap-gpu18", "x86_64"),
        "orin": ("ubuntu", "aarch64"),
    }[device]
    _require_equal(f"{device} execution hostname", host.get("hostname"), expected_host[0])
    _require_equal(
        f"{device} execution architecture", host.get("architecture"), expected_host[1]
    )
    audited_reports = audit.get("ap_reports", {})
    for precision in ("fp16", "int8"):
        row = audited_reports.get(precision, {})
        _require_equal(
            f"{device} {precision} audited AP report SHA",
            row.get("report_sha256"),
            ap_reports[precision].get("_report_sha256"),
        )
        _require_equal(
            f"{device} {precision} audited engine SHA",
            row.get("engine_sha256"),
            builds[precision].get("engine_sha256"),
        )


def _numerical_complete(report: dict[str, Any]) -> bool:
    if "all_required_metrics_present" in report:
        return bool(report["all_required_metrics_present"])
    required = {
        "cosine",
        "nrmse",
        "mae",
        "reference_min",
        "reference_max",
        "candidate_min",
        "candidate_max",
        "saturation_clipping_semantics",
    }
    outputs = report.get("per_output", {})
    return len(outputs) == 3 and all(required <= set(row) for row in outputs.values())


def _power_summary(device: str, latency: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {
        precision: report.get("power_measurement", {})
        for precision, report in latency.items()
    }
    if device == "h800":
        for precision, row in rows.items():
            if "NVML" not in str(row.get("source", "")):
                raise ValueError(f"H800 {precision} power is not NVML board power")
        return {
            "physical_quantity": "NVML board power",
            "measurement_interface": "nvidia-smi/NVML",
            "fp16_mean_w": rows["fp16"].get("board_power_mean_w"),
            "int8_mean_w": rows["int8"].get("board_power_mean_w"),
            "comparison_policy": "not_physically_equivalent_to_orin_rails",
        }
    for precision, row in rows.items():
        if "tegrastats" not in str(row.get("source", "")):
            raise ValueError(f"Orin {precision} power is not tegrastats")
    return {
        "physical_quantity": "Jetson tegrastats rails",
        "measurement_interface": "tegrastats",
        "fp16_vin_sys_5v0_mean_w": rows["fp16"].get("vin_sys_5v0_mean_w"),
        "int8_vin_sys_5v0_mean_w": rows["int8"].get("vin_sys_5v0_mean_w"),
        "fp16_vdd_gpu_soc_mean_w": rows["fp16"].get("vdd_gpu_soc_mean_w"),
        "int8_vdd_gpu_soc_mean_w": rows["int8"].get("vdd_gpu_soc_mean_w"),
        "comparison_policy": "not_physically_equivalent_to_h800_board_power",
    }


def build_attribution_summary(
    *,
    receipts: dict[str, dict[str, Any]],
    devices: dict[str, dict[str, Any]],
    execution_audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_sha256, payload_id = _validate_receipts(receipts)
    _require_equal("device names", set(devices), {"h800", "orin"})
    _require_equal(
        "execution audit device names",
        set(execution_audits),
        {"h800", "orin"},
    )

    summarized: dict[str, Any] = {}
    outcomes: dict[str, str] = {}
    for device, evidence in devices.items():
        builds = evidence["build"]
        latency = evidence["latency"]
        ap_reports = evidence["ap"]
        numerical = evidence["numerical"]
        _validate_builds(
            device=device,
            builds=builds,
            manifest_sha256=manifest_sha256,
        )
        for precision in ("fp16", "int8"):
            _validate_latency(
                device=device,
                precision=precision,
                report=latency[precision],
                engine_sha256=builds[precision]["engine_sha256"],
            )
            if not _numerical_complete(numerical[precision]):
                raise ValueError(f"{device} {precision} numerical report incomplete")
        _validate_execution_audit(
            device=device,
            audit=execution_audits[device],
            ap_reports=ap_reports,
            builds=builds,
        )
        ap = {
            precision: _validate_ap(
                device=device,
                precision=precision,
                report=ap_reports[precision],
                engine_sha256=builds[precision]["engine_sha256"],
            )
            for precision in ("fp16", "int8")
        }
        outcome = classify_device_ap(fp16=ap["fp16"], int8=ap["int8"])
        outcomes[device] = outcome
        summarized[device] = {
            "runtime": {
                "tensorrt_version": builds["int8"]["tensorrt_version"],
            },
            "engines": {
                precision: {
                    "sha256": builds[precision]["engine_sha256"],
                    "layer_precision_counts": builds[precision][
                        "layer_precision_counts"
                    ],
                }
                for precision in ("fp16", "int8")
            },
            "latency": {
                precision: {
                    field: latency[precision][field]
                    for field in ("median_ms", "p90_ms", "p99_ms", "mean_ms")
                }
                for precision in ("fp16", "int8")
            },
            "power": _power_summary(device, latency),
            "ap": {
                **ap,
                "delta_int8_minus_fp16": {
                    "ap50": ap["int8"]["ap50"] - ap["fp16"]["ap50"],
                    "ap70": ap["int8"]["ap70"] - ap["fp16"]["ap70"],
                },
                "classification": outcome,
            },
            "numerical_status": "three_outputs_complete",
        }

    decision = attribute_ap_outcomes(outcomes["h800"], outcomes["orin"])
    return {
        "schema_version": "lane_c_calibration_locked_attribution_summary_v1",
        "scope": "Pyramid (16,32,64) get_multiscale_feature/backbone subnet only",
        "calibration": {
            "status": "byte_identical_15_of_15",
            "canonical_manifest_sha256": manifest_sha256,
            "payload_id": payload_id,
        },
        "devices": summarized,
        "attribution": {
            "h800_ap_classification": outcomes["h800"],
            "orin_ap_classification": outcomes["orin"],
            "decision": decision,
            "fine_tuning_performed": False,
        },
        "execution_evidence_grade": "calibration_locked_complete",
        "causal_evidence_grade": "degraded_for_strict_binary_causality",
        "preprocessing": {
            "h800": "native_spconv_voxelizer",
            "orin": "pure_pytorch_compatibility_voxelizer",
            "full_1789_byte_identity_proven": False,
            "prior_exact_audit_population": 60,
            "task_level_corroboration": (
                "H800 and Orin FP16 full_1789 AP are nearly identical"
            ),
        },
        "reporting_policy": {
            "cross_device_latency_comparison": "raw_values_only",
            "power_comparison": "separate_physical_quantities",
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if root != output and root not in output.parents:
        raise ValueError("output must remain inside result root")
    receipts = {
        "local": _read_json(root / "receipts/local_source_receipt.json"),
        "h800": _read_json(root / "receipts/h800_calibration_receipt.json"),
        "orin": _read_json(root / "receipts/orin_calibration_receipt.json"),
    }
    devices = {}
    execution_audits = {}
    for device in ("h800", "orin"):
        device_root = root / device
        devices[device] = {
            "build": {
                precision: _read_json(device_root / precision / "build_report.json")
                for precision in ("fp16", "int8")
            },
            "latency": {
                precision: _read_json(
                    device_root / precision / "primary_latency_power.json"
                )
                for precision in ("fp16", "int8")
            },
            "ap": {
                precision: {
                    **_read_json(
                        root
                        / "ap_full"
                        / device
                        / precision
                        / "stage3_trt_multiscale_ap_bridge_report.json"
                    ),
                    "_report_sha256": _sha256_file(
                        root
                        / "ap_full"
                        / device
                        / precision
                        / "stage3_trt_multiscale_ap_bridge_report.json"
                    ),
                }
                for precision in ("fp16", "int8")
            },
            "numerical": {
                precision: _read_json(
                    device_root / precision / "numerical_comparison.json"
                )
                for precision in ("fp16", "int8")
            },
        }
        execution_audits[device] = _read_json(
            root / "audits" / f"{device}_ap_execution_audit.json"
        )
    summary = build_attribution_summary(
        receipts=receipts,
        devices=devices,
        execution_audits=execution_audits,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
