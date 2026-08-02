#!/usr/bin/env python3
"""Fail-closed finalizer for Lane C Original strict-FP32 Orin evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from tools.orin_deploy.lane_c_backbone_parity_runner import (
    classify_inspector_precisions,
    strict_fp32_inspector_evidence,
)


LOCKED_SOURCE = {
    "checkpoint_sha256": (
        "4ccc6fe1f7cc13b5d1294f74014b01e849cc8e90b69cbded14158e1999fa42b2"
    ),
    "config_sha256": (
        "f55a6ad8cff9fa522ce74fe9ef4d85d682314455278240ce7d5ec998d567f540"
    ),
    "onnx_sha256": (
        "3b125b3c77f7484d8b3a46ae674446f30e663c53fa38431c3f2cb96ae9cbb0ff"
    ),
}
LOCKED_AP_CONFIG_SHA256 = (
    "143831e75b041f9cd571c0e5a73c58bdfcbe092f7162c0130f3dcc146bb98088"
)
COMPACT_CHECKPOINT_SHA256 = (
    "d08fb16e778c6701aef7172e8d9609f1e4d73f20c419ee27bb8f158acc24a279"
)
COMPACT_ONNX_SHA256 = (
    "8f09b5256f1856cc79ebbebf0d6c26e6dc63fba2ac3994552a690eaacd3be2a5"
)
ORIGINAL_CHANNELS = (64, 128, 256)
COMPACT_CHANNELS = (16, 32, 64)
OUTPUT_NAMES = ("pyramid_level0", "pyramid_level1", "pyramid_level2")
FORBIDDEN_CLAIM_KEYS = {
    "speedup",
    "speedup_ratio",
    "cross_hardware_speedup",
    "isolated_pruning_claim",
    "isolated_dtype_claim",
    "single_factor_causal_claim",
}
FORBIDDEN_POWER_STRING_PATTERNS = (
    re.compile(
        r"(?<![a-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*w\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwatts?\b", re.IGNORECASE),
    re.compile(r"\b[a-z][a-z0-9_]*_w\b", re.IGNORECASE),
    re.compile(r"\bw\b", re.IGNORECASE),
)


class EvidenceValidationError(ValueError):
    """Raised when a required evidence contract is absent or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing required evidence: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"invalid JSON evidence {path}: {error}") from error
    _require(isinstance(payload, dict), f"JSON evidence must be an object: {path}")
    return payload


def _require_file_sha(path: Path, expected: str, label: str) -> str:
    _require(path.is_file(), f"missing {label}: {path}")
    observed = sha256_file(path)
    _require(observed == expected, f"{label} SHA mismatch: {observed} != {expected}")
    return observed


def _reject_forbidden_claims(value: Any, location: str = "evidence") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_CLAIM_KEYS:
                raise EvidenceValidationError(
                    f"forbidden claim field {key!r} in {location}"
                )
            _reject_forbidden_claims(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_claims(child, f"{location}[{index}]")


def _validate_metric_outputs(
    comparison: dict[str, Any], *, label: str
) -> dict[str, dict[str, Any]]:
    per_output = comparison.get("per_output")
    _require(isinstance(per_output, dict), f"{label} per_output is missing")
    _require(
        set(per_output) == set(OUTPUT_NAMES),
        f"{label} must contain exactly three numerical output levels",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        metrics = per_output[name]
        _require(isinstance(metrics, dict), f"{label} metrics missing for {name}")
        _require(metrics.get("finite") is True, f"{label} {name} is not finite")
        row: dict[str, Any] = {"finite": True}
        for metric in ("cosine", "nrmse", "mae"):
            _require(metric in metrics, f"{label} {name} missing {metric}")
            number = float(metrics[metric])
            _require(math.isfinite(number), f"{label} {name} {metric} is non-finite")
            row[metric] = number
        result[name] = row
    return result


def _validate_protocol(report: dict[str, Any], *, label: str) -> None:
    protocol = report.get("protocol")
    _require(isinstance(protocol, dict), f"{label} latency protocol is missing")
    expected = {
        "agent_batch": 2,
        "warmup": 20,
        "iters": 300,
        "repeat": 5,
        "timing": "CUDA_event",
        "data_transfer_inside_timed_region": False,
    }
    for key, value in expected.items():
        _require(
            protocol.get(key) == value,
            f"{label} latency protocol {key} must be {value!r}",
        )
    boundary = protocol.get("boundary", "engine_compute_no_data_transfer")
    _require(
        boundary == "engine_compute_no_data_transfer",
        f"{label} latency boundary must exclude data transfer",
    )
    _require(report.get("sample_count") == 1500, f"{label} sample_count must be 1500")
    per_repeat = report.get("per_repeat")
    if per_repeat is not None:
        _require(
            isinstance(per_repeat, list) and len(per_repeat) == 5,
            f"{label} must contain five latency repeats",
        )
        _require(
            all(item.get("sample_count") == 300 for item in per_repeat),
            f"{label} each latency repeat must contain 300 samples",
        )


def _raw_latency(report: dict[str, Any], *, label: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("median_ms", "p90_ms", "p99_ms", "mean_ms"):
        _require(key in report, f"{label} latency missing {key}")
        number = float(report[key])
        _require(math.isfinite(number) and number > 0.0, f"{label} invalid {key}")
        result[key] = number
    return result


def _validate_ap(
    report: dict[str, Any],
    *,
    label: str,
    engine_sha256: str,
    checkpoint_sha256: str | None = None,
    config_sha256: str | None = None,
    require_detailed_output_counts: bool = True,
) -> dict[str, float]:
    expected_counts = {
        "processed_samples": 1789,
        "failed_samples": 0,
        "fallback_samples": 0,
    }
    for key, value in expected_counts.items():
        _require(report.get(key) == value, f"{label} AP {key} must be {value}")
    _require(report.get("engine_ap_claim") is True, f"{label} engine_ap_claim must be true")
    blockers = report.get("engine_ap_claim_blockers", [])
    _require(blockers == [], f"{label} AP claim blockers must be empty")
    _require(
        report.get("engine_sha256") == engine_sha256,
        f"{label} AP engine SHA mismatch",
    )
    if checkpoint_sha256 is not None:
        _require(
            report.get("checkpoint_sha256") == checkpoint_sha256,
            f"{label} AP checkpoint SHA mismatch",
        )
    if config_sha256 is not None:
        _require(
            report.get("config_sha256") == config_sha256,
            f"{label} AP config SHA mismatch",
        )
    output = report.get("output_error_summary")
    _require(isinstance(output, dict), f"{label} AP output summary is missing")
    _require(output.get("num_records") == 5367, f"{label} AP num_records must be 5367")
    _require(
        output.get("num_compared") == 5367,
        f"{label} AP num_compared must be 5367",
    )
    for key, description in (
        ("shape_mismatch_count", "shape mismatch count"),
        ("nonfinite_count", "nonfinite count"),
    ):
        if require_detailed_output_counts or key in output:
            _require(output.get(key) == 0, f"{label} AP {description} must be zero")
    _require(output.get("all_finite") is True, f"{label} AP outputs are not finite")
    result = {"ap50": float(report["ap50"]), "ap70": float(report["ap70"])}
    _require(
        all(math.isfinite(value) for value in result.values()),
        f"{label} AP values are non-finite",
    )
    return result


def _validate_power_unavailable(report: dict[str, Any]) -> dict[str, str]:
    power = report.get("power_measurement")
    _require(isinstance(power, dict), "Original power measurement evidence is missing")
    _require(
        power.get("measurement_status") == "unavailable",
        "Original power measurement_status must be unavailable",
    )
    _require(
        power.get("blocker") == "missing_tegrastats_power_rails",
        "Original unavailable power blocker must be missing_tegrastats_power_rails",
    )
    _require(power.get("used_sudo") is False, "Original power used_sudo must be false")
    allowed_metadata = {
        "measurement_status",
        "blocker",
        "source",
        "raw_log_sha256",
        "used_sudo",
        "rail_semantics",
    }
    for key, value in power.items():
        normalized = str(key).lower()
        _require(
            "watt" not in normalized
            and not normalized.endswith("_w")
            and "_w_" not in normalized,
            "unavailable power evidence must not contain watts",
        )
        _require(
            key in allowed_metadata,
            "unavailable power measurement payload contains unsupported fields",
        )
        _require(
            not isinstance(value, (dict, list, tuple))
            and not (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ),
            "unavailable power measurement payload must not contain numeric samples",
        )
        _require(
            not (
                isinstance(value, str)
                and any(
                    pattern.search(value)
                    for pattern in FORBIDDEN_POWER_STRING_PATTERNS
                )
            ),
            "unavailable power metadata contains a power value or unit",
        )
    return {
        "measurement_status": "unavailable",
        "blocker": "missing_tegrastats_power_rails",
        "semantics": str(power.get("rail_semantics", "Orin tegrastats named rails")),
    }


def _validate_original(
    root: Path,
    expected_source: dict[str, str],
    expected_ap_config_sha256: str,
) -> dict[str, Any]:
    contract = _load_json(root / "contracts/experiment_contract.json")
    _reject_forbidden_claims(contract)
    source_contract = contract.get("source")
    _require(isinstance(source_contract, dict), "source contract is missing")
    for key, expected_sha256 in expected_source.items():
        _require(
            source_contract.get(key) == expected_sha256,
            f"locked source contract {key} mismatch",
        )
    expected_input = {
        "name": "spatial_features",
        "dtype": "float32",
        "shape": [2, 64, 128, 256],
    }
    _require(
        source_contract.get("input") == expected_input,
        "source ONNX input contract mismatch",
    )
    expected_outputs = [
        {
            "name": name,
            "dtype": "float32",
            "shape": shape,
        }
        for name, shape in zip(
            OUTPUT_NAMES,
            (
                [2, 64, 128, 256],
                [2, 128, 64, 128],
                [2, 256, 32, 64],
            ),
        )
    ]
    _require(
        source_contract.get("outputs") == expected_outputs,
        "source ONNX output contract mismatch",
    )
    scope = contract.get("scope", {})
    _require(scope.get("structure") == list(ORIGINAL_CHANNELS), "scope channels mismatch")
    _require(
        contract.get("reporting", {}).get("comparison")
        == "combined_structure_and_precision_deployment_comparison",
        "comparison must be structure-plus-precision deployment evidence",
    )

    source_paths = {
        "checkpoint_sha256": root / "source/net_epoch_bestval_at23.pth",
        "config_sha256": root / "source/original_config.yaml",
        "onnx_sha256": root / "source/pyramid_064x128x256_multiscale.onnx",
    }
    for key, path in source_paths.items():
        _require_file_sha(path, expected_source[key], key.removesuffix("_sha256"))
    export_report_path = root / "source/onnx_export_report.json"
    _require_file_sha(
        export_report_path,
        str(source_contract.get("onnx_export_report_sha256")),
        "ONNX export report",
    )
    export_report = _load_json(export_report_path)
    _require(
        export_report.get("onnx_digest") == expected_source["onnx_sha256"],
        "ONNX export report digest mismatch",
    )
    _require(
        export_report.get("input_shape") == expected_input["shape"],
        "ONNX export report input shape mismatch",
    )
    _require(
        export_report.get("output_names") == list(OUTPUT_NAMES),
        "ONNX export report output names mismatch",
    )
    _require(
        export_report.get("output_shapes")
        == {
            item["name"]: item["shape"]
            for item in expected_outputs
        },
        "ONNX export report output shapes mismatch",
    )

    heldout = _load_json(root / "heldout/original_heldout_batch2_audit.json")
    _reject_forbidden_claims(heldout)
    _require(
        heldout.get("checkpoint_sha256") == expected_source["checkpoint_sha256"],
        "held-out checkpoint source SHA mismatch",
    )
    _require(
        heldout.get("config_sha256") == expected_source["config_sha256"],
        "held-out config source SHA mismatch",
    )
    _require(
        heldout.get("output_dtype") == "float32"
        and heldout.get("source_dtype") == "float32",
        "held-out audit must declare float32 source and output",
    )
    _require(
        heldout.get("output_all_finite") is True
        and heldout.get("source_all_finite") is True,
        "held-out audit must be finite",
    )
    _require(
        heldout.get("calibration_used_for_engine") is False,
        "held-out data must not be used for calibration",
    )
    heldout_path = root / "heldout/original_heldout_batch2_fp32.npy"
    heldout_sha = _require_file_sha(
        heldout_path, str(heldout.get("output_npy_sha256")), "held-out NPY"
    )
    heldout_array = np.load(heldout_path, mmap_mode="r", allow_pickle=False)
    _require(
        heldout_array.dtype == np.dtype(np.float32),
        "held-out NPY must preserve float32 dtype",
    )
    source_npz = root / "heldout/original_train32_spatial_features_fp32.npz"
    _require_file_sha(
        source_npz,
        str(heldout.get("source_export_npz_sha256")),
        "held-out source export",
    )
    _require_file_sha(
        root / "heldout/original_train32_export_summary.json",
        str(heldout.get("source_export_summary_sha256")),
        "held-out source summary",
    )
    split_sha = str(heldout.get("split_source_sha256", ""))
    _require(len(split_sha) == 64, "held-out split source SHA is missing")

    build = _load_json(root / "orin/fp32/build_receipt.json")
    _reject_forbidden_claims(build)
    _require(build.get("precision") == "fp32", "build precision must be fp32")
    _require(build.get("strict_fp32") is True, "build strict_fp32 must be true")
    _require(build.get("tf32_allowed") is False, "build must disable TF32")
    _require(
        build.get("builder_flags") == ["fp32", "tf32_disabled"],
        "build flags must explicitly disable TF32",
    )
    counts = build.get("layer_precision_counts", {})
    for precision, display in (("fp16", "FP16"), ("int8", "INT8")):
        _require(counts.get(precision) == 0, f"build {display} count must be zero")
    _require(counts.get("tf32", 0) == 0, "build TF32 count must be zero")
    _require(
        build.get("forbidden_inspector_matches") == [],
        "build inspector contains forbidden precision evidence",
    )
    _require(
        build.get("external_fallback_count") == 0,
        "build external fallback count must be zero",
    )
    _require(
        build.get("onnx_sha256") == expected_source["onnx_sha256"],
        "build ONNX SHA mismatch",
    )
    _require(
        build.get("output_channel_signature") == list(ORIGINAL_CHANNELS),
        "build output channels must be 64,128,256",
    )
    engine_sha = _require_file_sha(
        root / "orin/fp32/original_strict_fp32.engine",
        str(build.get("engine_sha256")),
        "strict-FP32 engine",
    )
    inspector_path = root / "orin/fp32/engine_inspector.json"
    _require_file_sha(
        inspector_path,
        str(build.get("inspector_json_sha256")),
        "engine inspector",
    )
    inspector = _load_json(inspector_path)
    strict_inspector = strict_fp32_inspector_evidence(inspector)
    _require(
        strict_inspector["strict_fp32"] is True,
        "strict FP32 inspector validation failed: "
        + json.dumps(
            strict_inspector["forbidden_inspector_matches"],
            sort_keys=True,
        ),
    )
    computed_counts = classify_inspector_precisions(inspector)
    _require(counts.get("other") == 0, "build other count must be zero")
    for precision in ("int8", "fp16", "fp32", "other"):
        _require(
            counts.get(precision) == computed_counts[precision],
            f"build receipt {precision} count does not match engine inspector",
        )
    _require(
        computed_counts["int8"] == 0
        and computed_counts["fp16"] == 0
        and computed_counts["other"] == 0,
        "strict FP32 inspector contains reduced, other, or untyped compute precision",
    )
    _require(
        build.get("forbidden_inspector_matches")
        == strict_inspector["forbidden_inspector_matches"],
        "build receipt forbidden inspector evidence mismatch",
    )

    numerical = _load_json(root / "orin/fp32/numerical_report.json")
    _reject_forbidden_claims(numerical)
    _require(numerical.get("all_finite") is True, "numerical outputs are not finite")
    _require(
        numerical.get("engine_sha256") == engine_sha,
        "numerical engine SHA mismatch",
    )
    _require(
        numerical.get("inputs_sha256") == heldout_sha,
        "numerical input SHA mismatch",
    )
    _require(
        numerical.get("expected_output_channels") == list(ORIGINAL_CHANNELS)
        and numerical.get("output_channel_signature") == list(ORIGINAL_CHANNELS),
        "numerical output channels must be 64,128,256",
    )
    _require(
        numerical.get("external_fallback_count") == 0,
        "numerical external fallback count must be zero",
    )
    candidate_sha = _require_file_sha(
        root / "orin/fp32/heldout_outputs_fp32.npz",
        str(numerical.get("output_sha256")),
        "numerical candidate output",
    )
    reference = _load_json(root / "reference/server_ort_fp32_reference_report.json")
    _require(reference.get("all_finite") is True, "reference outputs are not finite")
    _require(reference.get("inputs_sha256") == heldout_sha, "reference input SHA mismatch")
    _require(
        reference.get("onnx_sha256") == expected_source["onnx_sha256"],
        "reference ONNX SHA mismatch",
    )
    reference_sha = _require_file_sha(
        root / "reference/server_ort_fp32_outputs.npz",
        str(reference.get("output_npz_sha256")),
        "numerical reference output",
    )
    comparison = _load_json(root / "orin/fp32/numerical_comparison.json")
    _reject_forbidden_claims(comparison)
    _require(comparison.get("precision") == "fp32", "comparison precision must be fp32")
    _require(
        comparison.get("candidate_npz_sha256") == candidate_sha,
        "comparison candidate SHA mismatch",
    )
    _require(
        comparison.get("reference_npz_sha256") == reference_sha,
        "comparison reference SHA mismatch",
    )
    numerical_metrics = _validate_metric_outputs(comparison, label="Original FP32")

    latency = _load_json(root / "orin/fp32/primary_latency_power.json")
    _reject_forbidden_claims(latency)
    _require(latency.get("engine_sha256") == engine_sha, "latency engine SHA mismatch")
    _require(latency.get("inputs_sha256") == heldout_sha, "latency input SHA mismatch")
    _validate_protocol(latency, label="Original FP32")
    _require(
        latency.get("expected_output_channels") == list(ORIGINAL_CHANNELS)
        and latency.get("output_channel_signature") == list(ORIGINAL_CHANNELS),
        "latency output channels must be 64,128,256",
    )
    latency_metrics = _raw_latency(latency, label="Original FP32")
    power = _validate_power_unavailable(latency)

    ap_config = (
        root
        / "ap_source/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml"
    )
    derived_config_sha256 = _require_file_sha(
        ap_config,
        expected_ap_config_sha256,
        "derived AP config",
    )
    derivation = _load_json(root / "ap_source/config_derivation_audit.json")
    _require(
        derivation.get("original_config_sha256")
        == expected_source["config_sha256"],
        "AP config derivation original config SHA mismatch",
    )
    _require(
        derivation.get("derived_config_sha256") == derived_config_sha256,
        "AP config derivation derived config SHA mismatch",
    )
    allowed_changes = derivation.get("allowed_changes")
    _require(
        isinstance(allowed_changes, list)
        and len(allowed_changes) == 4
        and set(allowed_changes)
        == {"data_dir", "root_dir", "test_dir", "validate_dir"},
        "AP config derivation allowed changes mismatch",
    )
    _require(
        derivation.get("unexpected_changes", []) == [],
        "AP config derivation has unexpected changes",
    )
    ap = _load_json(
        root / "ap_full/orin/fp32/stage3_trt_multiscale_ap_bridge_report.json"
    )
    _reject_forbidden_claims(ap)
    ap_metrics = _validate_ap(
        ap,
        label="Original FP32",
        engine_sha256=engine_sha,
        checkpoint_sha256=expected_source["checkpoint_sha256"],
        config_sha256=derived_config_sha256,
    )
    _require(ap.get("precision_tag") == "fp32", "Original AP precision must be fp32")
    _require(
        ap.get("protocol", {}).get("expected_output_channels")
        == list(ORIGINAL_CHANNELS),
        "Original AP output channels must be 64,128,256",
    )
    _require(
        ap.get("protocol", {}).get("eval_range") == "102.4,51.2",
        "Original AP eval_range must be 102.4,51.2",
    )
    return {
        "variant": "Original FP32",
        "structure": list(ORIGINAL_CHANNELS),
        "precision": "fp32",
        "engine_sha256": engine_sha,
        "heldout_sha256": heldout_sha,
        "ap_config_sha256": derived_config_sha256,
        "strict_precision_counts": {
            "tf32": 0,
            "bf16": 0,
            **computed_counts,
        },
        "latency": latency_metrics,
        "ap": ap_metrics,
        "numerical": numerical_metrics,
        "power": power,
    }


def _validate_baseline(root: Path) -> dict[str, Any]:
    build = _load_json(root / "orin/fp16/build_report.json")
    numerical = _load_json(root / "orin/fp16/numerical_report.json")
    comparison = _load_json(root / "orin/fp16/numerical_comparison.json")
    latency = _load_json(root / "orin/fp16/primary_latency_power.json")
    ap = _load_json(
        root / "ap_full/orin/fp16/stage3_trt_multiscale_ap_bridge_report.json"
    )
    for payload in (build, numerical, comparison, latency, ap):
        _reject_forbidden_claims(payload, "compact FP16 baseline")
    _require(build.get("precision") == "fp16", "baseline precision must be fp16")
    _require(
        build.get("scope") == "pyramid_get_multiscale_feature_16x32x64_only",
        "baseline scope must be compact 16,32,64 backbone",
    )
    _require(
        build.get("onnx_sha256") == COMPACT_ONNX_SHA256,
        "baseline build ONNX SHA mismatch",
    )
    _require_file_sha(
        root / "source/pyramid_016x032x064_multiscale.onnx",
        COMPACT_ONNX_SHA256,
        "compact FP16 ONNX",
    )
    engine_sha = _require_file_sha(
        root / "orin/fp16/backbone_fp16.engine",
        str(build.get("engine_sha256")),
        "compact FP16 engine",
    )
    _require(
        numerical.get("engine_sha256") == engine_sha,
        "baseline numerical engine SHA mismatch",
    )
    _require(
        numerical.get("all_finite") is True
        and numerical.get("external_fallback_count") == 0,
        "baseline numerical report is not finite or used fallback",
    )
    shapes = numerical.get("output_shapes", {})
    channels = [shapes.get(name, [None, None, None])[2] for name in OUTPUT_NAMES]
    _require(channels == list(COMPACT_CHANNELS), "baseline output channels mismatch")
    numerical_metrics = _validate_metric_outputs(comparison, label="compact FP16")
    _require(comparison.get("precision") == "fp16", "baseline comparison precision mismatch")
    _require(latency.get("engine_sha256") == engine_sha, "baseline latency engine SHA mismatch")
    _validate_protocol(latency, label="compact FP16")
    latency_metrics = _raw_latency(latency, label="compact FP16")
    _require(
        ap.get("checkpoint_sha256") == COMPACT_CHECKPOINT_SHA256,
        "baseline AP checkpoint SHA mismatch",
    )
    ap_metrics = _validate_ap(
        ap,
        label="compact FP16",
        engine_sha256=engine_sha,
        require_detailed_output_counts=False,
    )
    return {
        "variant": "compact FP16",
        "structure": list(COMPACT_CHANNELS),
        "precision": "fp16",
        "engine_sha256": engine_sha,
        "checkpoint_sha256": COMPACT_CHECKPOINT_SHA256,
        "onnx_sha256": COMPACT_ONNX_SHA256,
        "latency": latency_metrics,
        "ap": ap_metrics,
        "numerical": numerical_metrics,
        "power": {
            "measurement_status": "prior_result",
            "semantics": "Orin tegrastats named rails; not cross-device board power",
        },
    }


def _comparison_row(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": evidence["variant"],
        "structure": ",".join(str(value) for value in evidence["structure"]),
        "precision": evidence["precision"],
        **evidence["latency"],
        **evidence["ap"],
        "power_status": evidence["power"]["measurement_status"],
        "power_semantics": evidence["power"]["semantics"],
    }


def _write_outputs(root: Path, summary: dict[str, Any]) -> None:
    final_path = root / "final_summary.json"
    final_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = [_comparison_row(item) for item in summary["rows"]]
    csv_path = root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original, baseline = summary["rows"]
    original_row = (
        "| Original FP32 | 64,128,256 | FP32 "
        f"| {original['latency']['median_ms']} "
        f"| {original['latency']['p90_ms']} "
        f"| {original['latency']['p99_ms']} "
        f"| {original['latency']['mean_ms']} "
        f"| {original['ap']['ap50']} | {original['ap']['ap70']} |"
    )
    baseline_row = (
        "| compact FP16 | 16,32,64 | FP16 "
        f"| {baseline['latency']['median_ms']} "
        f"| {baseline['latency']['p90_ms']} "
        f"| {baseline['latency']['p99_ms']} "
        f"| {baseline['latency']['mean_ms']} "
        f"| {baseline['ap']['ap50']} | {baseline['ap']['ap70']} |"
    )
    markdown = f"""# Lane C Original FP32 final evidence

Status: complete.

Original FP32 vs compact FP16 compares deployed model variants (checkpoint +
structure) together with precision. This structure + precision result is not
an isolated pruning, channel-count, or dtype causal claim. Even a future
same-precision comparison would not establish pure channel-count causality,
because the deployed checkpoints differ.

| Variant | Structure | Precision | Median ms | P90 ms | P99 ms | Mean ms | AP50 | AP70 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{original_row}
{baseline_row}

Only original measured values are listed; no derived performance ratio is
reported.

Power is not comparable across physical measurement semantics, and the
Original FP32 power measurement is unavailable in this run because named
tegrastats power rails were missing. No watt value is imputed.

AP caveat: the full_1789 bridge uses the pure-torch voxelizer while replacing
only `get_multiscale_feature`; fusion, shrinker, and heads remain in PyTorch.
"""
    (root / "summary.md").write_text(markdown, encoding="utf-8")


def finalize_evidence(
    artifact_root: Path,
    baseline_root: Path,
    *,
    expected_source: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = Path(artifact_root).expanduser().resolve()
    baseline = Path(baseline_root).expanduser().resolve()
    _require(root.is_dir(), f"artifact root does not exist: {root}")
    _require(baseline.is_dir(), f"baseline root does not exist: {baseline}")
    locked_source = dict(LOCKED_SOURCE if expected_source is None else expected_source)
    original_evidence = _validate_original(
        root,
        locked_source,
        LOCKED_AP_CONFIG_SHA256,
    )
    baseline_evidence = _validate_baseline(baseline)
    summary = {
        "schema_version": "lane_c_original_fp32_final_summary_v1",
        "status": "complete",
        "comparison_semantics": (
            "deployed_model_variant_checkpoint_and_structure_plus_precision"
        ),
        "source_sha256": locked_source,
        "derived_ap_config_sha256": LOCKED_AP_CONFIG_SHA256,
        "rows": [original_evidence, baseline_evidence],
        "reporting_caveats": {
            "power": (
                "power is not comparable; Original FP32 is unavailable this run"
            ),
            "ap": "full_1789 AP uses the pure-torch voxelizer",
            "causality": (
                "deployed model variant (checkpoint + structure) + precision; "
                "even same-precision evidence would not isolate channel-count "
                "causality"
            ),
        },
    }
    _write_outputs(root, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = finalize_evidence(args.artifact_root, args.baseline_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
