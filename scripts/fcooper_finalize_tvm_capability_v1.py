#!/usr/bin/env python3
"""Finalize label-free F-Cooper/H800/TVM capability evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage2.canonical_search_v3 import (
    build_capability_profile,
    validate_capability_profile,
)


PROFILE_ID = "h800-tvm-fcooper-probe-conditioned-v1"
DISPATCH_KEY = "tvm_auto"
HARDWARE_TARGET = "h800"
TARGET_LABEL_TOKENS = ("latency", "energy", "ap30", "ap50", "ap70", "accuracy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def resolve_path(raw: Any, owner: Path) -> Path:
    path = Path(str(raw or ""))
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def require_file_sha(
    path: Path, expected: Any, *, label: str, allow_missing: bool = False
) -> str:
    if not path.is_file():
        if allow_missing:
            return ""
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256_file(path)
    if str(expected or "") != actual:
        raise ValueError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}"
        )
    return actual


def _validate_optional_pair(
    payload: Mapping[str, Any],
    owner: Path,
    path_key: str,
    sha_key: str,
    *,
    label: str,
) -> str | None:
    raw_path = payload.get(path_key)
    expected = payload.get(sha_key)
    if raw_path in (None, "") and expected in (None, ""):
        return None
    if raw_path in (None, "") or expected in (None, ""):
        raise ValueError(f"{label} must bind both {path_key} and {sha_key}")
    path = resolve_path(raw_path, owner)
    return require_file_sha(path, expected, label=label)


def _correctness_rows(payload: Mapping[str, Any], precision: str) -> list[dict[str, Any]]:
    rows = payload.get(f"correctness_vs_default_{precision}")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{precision} probe lacks correctness rows")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def validate_fp_probe(path: Path, *, precision: str, role: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "success" or payload.get("build_success") is not True:
        raise ValueError(f"{role} {precision} probe did not build and run successfully")
    declared_precision = str(payload.get("precision") or precision)
    if declared_precision != precision:
        raise ValueError(f"{role} precision drift: {declared_precision} != {precision}")
    artifact = resolve_path(payload.get("artifact_path"), path)
    require_file_sha(
        artifact, payload.get("artifact_digest"), label=f"{role} artifact"
    )
    onnx = resolve_path(payload.get("onnx_path"), path)
    if not onnx.is_file():
        raise FileNotFoundError(f"{role} ONNX not found: {onnx}")
    _validate_optional_pair(
        payload, path, "ref_so", "ref_digest", label=f"{role} reference artifact"
    )
    rows = _correctness_rows(payload, precision)
    if len(rows) == 0:
        raise ValueError(f"{role} correctness rows are malformed")
    for index, row in enumerate(rows):
        if row.get("shape_match") is not True:
            raise ValueError(f"{role} correctness shape mismatch at output {index}")
        for key in ("max_abs_err", "p99_rel_err"):
            value = float(row.get(key, math.inf))
            if not math.isfinite(value):
                raise ValueError(f"{role} non-finite correctness metric {key}")
        if float(row.get("p99_rel_err", math.inf)) > 0.02:
            raise ValueError(f"{role} p99 relative correctness gate failed")
    return {
        "role": role,
        "precision": precision,
        "width": list(payload.get("width") or []),
        "result_path": str(path.resolve()),
        "result_sha256": sha256_file(path),
        "artifact_sha256": sha256_file(artifact),
        "onnx_sha256": sha256_file(onnx),
        "build_run_passed": True,
        "correctness_passed": True,
        "correctness_output_count": len(rows),
        "schema": payload.get("schema"),
        "method": payload.get("method"),
        "host": payload.get("host"),
    }


def validate_quant_contract(path: Path, *, role: str) -> dict[str, Any]:
    payload = read_json(path)
    if (
        payload.get("schema_version") != "fcooper_tvm_int8_quant_contract_v1"
        or payload.get("status") != "success"
    ):
        raise ValueError(f"{role} quant contract is not successful v1 evidence")
    quantization = payload.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ValueError(f"{role} quantization contract is missing")
    if (
        quantization.get("dtype") != "uint8"
        or quantization.get("semantics")
        != "static_symmetric_uint8_centered_128"
    ):
        raise ValueError(f"{role} quantization must use the frozen static uint8 contract")

    onnx = payload.get("onnx")
    calibration = payload.get("calibration")
    coverage = payload.get("coverage")
    if not all(isinstance(item, Mapping) for item in (onnx, calibration, coverage)):
        raise ValueError(f"{role} quant contract sections are incomplete")
    onnx_path = resolve_path(onnx.get("path"), path)
    require_file_sha(onnx_path, onnx.get("sha256"), label=f"{role} quant ONNX")
    summary_path = resolve_path(calibration.get("summary_path"), path)
    require_file_sha(
        summary_path,
        calibration.get("summary_sha256"),
        label=f"{role} calibration summary",
    )
    samples = calibration.get("samples") or []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"{role} calibration sample {index} is malformed")
        require_file_sha(
            resolve_path(sample.get("path"), path),
            sample.get("sha256"),
            label=f"{role} calibration sample {index}",
        )
    if int(payload.get("sample_count", -1)) != 16 or int(
        calibration.get("sample_count", -1)
    ) != 16:
        raise ValueError(f"{role} quant contract requires exactly 16 samples")
    required = int(coverage.get("required_count", -1))
    observed = int(coverage.get("observed_count", -1))
    missing = coverage.get("missing")
    if required <= 0 or observed != required or missing != []:
        raise ValueError(
            f"{role} quant coverage incomplete: observed={observed}, "
            f"required={required}, missing={missing}"
        )
    params = payload.get("params")
    if not isinstance(params, Mapping) or len(params) != observed:
        raise ValueError(f"{role} quant coverage and params count disagree")
    for name, param in params.items():
        if not isinstance(param, Mapping):
            raise ValueError(f"{role} quant parameter {name} is malformed")
        scale = float(param.get("scale", 0.0))
        if not math.isfinite(scale) or scale <= 0.0 or int(param.get("zero_point", -1)) != 128:
            raise ValueError(f"{role} quant parameter {name} violates static contract")
    for group in payload.get("concat_scale_normalization") or []:
        members = group.get("members") if isinstance(group, Mapping) else None
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"{role} Concat normalization group is malformed")
        scales = {float(params[name]["scale"]) for name in members}
        if len(scales) != 1:
            raise ValueError(f"{role} Concat members do not share one quant scale")
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "sample_count": 16,
        "required_tensor_count": required,
        "observed_tensor_count": observed,
        "concat_group_count": len(payload.get("concat_scale_normalization") or []),
        "coverage_passed": True,
    }


def _load_block_reports(payload: Mapping[str, Any], owner: Path) -> tuple[list[dict[str, Any]], str | None]:
    inline = payload.get("block_reports")
    if isinstance(inline, list):
        return [dict(item) for item in inline if isinstance(item, Mapping)], None
    raw_path = payload.get("block_reports_path")
    if not raw_path:
        raise ValueError("INT8 probe must expose block reports")
    path = resolve_path(raw_path, owner)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"block report root must be a list: {path}")
    return [dict(item) for item in raw if isinstance(item, Mapping)], sha256_file(path)


def validate_int8_probe(
    path: Path, *, role: str, quant: Mapping[str, Any]
) -> dict[str, Any]:
    payload = read_json(path)
    route_spec = str(payload.get("route_spec") or "")
    method = str(payload.get("method") or "")
    if (
        "auto_decomp" not in route_spec
        or "tensorization" not in route_spec
        or "hand_written" in route_spec.lower()
        or "hand-written" in method.lower()
        or "hand rewrite" in method.lower()
    ):
        raise ValueError(f"{role} is not the required automatic Route B path")
    if payload.get("status") != "success" or payload.get("build_success") is not True:
        raise ValueError(f"{role} INT8 probe did not build and run successfully")
    if payload.get("correctness_all_exact") is not True:
        raise ValueError(f"{role} INT8 correctness is not exact")
    correctness = payload.get("correctness_vs_native_direct")
    if not isinstance(correctness, list) or not correctness or not all(
        isinstance(row, Mapping) and row.get("exact_equal") is True
        for row in correctness
    ):
        raise ValueError(f"{role} INT8 correctness rows are incomplete")

    artifact = resolve_path(payload.get("artifact_path"), path)
    require_file_sha(
        artifact, payload.get("artifact_digest"), label=f"{role} INT8 artifact"
    )
    onnx = resolve_path(payload.get("onnx_path"), path)
    if sha256_file(onnx) != quant["onnx_sha256"]:
        raise ValueError(f"{role} INT8 ONNX does not match quant contract")
    quant_path = resolve_path(payload.get("tensor_quant_params_path"), path)
    if quant_path != Path(str(quant["path"])).resolve():
        raise ValueError(f"{role} INT8 quant contract path drift")
    require_file_sha(
        quant_path,
        payload.get("tensor_quant_params_sha256"),
        label=f"{role} INT8 quant contract",
    )
    if int(payload.get("tensor_quant_params_count", -1)) != int(
        quant["observed_tensor_count"]
    ):
        raise ValueError(f"{role} INT8 quant parameter count drift")

    blocks, block_report_sha = _load_block_reports(payload, path)
    conv_count = int(payload.get("n_conv_blocks", -1))
    tensorized = int(payload.get("n_tensorized_conv_blocks", -1))
    failures = int(payload.get("n_schedule_failures", -1))
    if conv_count <= 0 or not 0 <= tensorized <= conv_count or failures != 0:
        raise ValueError(f"{role} automatic tensorization coverage is invalid")
    block_conv = [item for item in blocks if item.get("op_type") == "Conv"]
    if len(block_conv) != conv_count:
        raise ValueError(f"{role} block report Conv count drift")
    tensorized_from_blocks = sum(
        item.get("tensorization_status") == "tensorized" for item in block_conv
    )
    if tensorized_from_blocks != tensorized:
        raise ValueError(f"{role} tensorized block count drift")
    fallback = sum(
        "fallback" in str(item.get("tensorization_status", "")).lower()
        for item in block_conv
    )
    unclassified = conv_count - tensorized - fallback
    if unclassified:
        raise ValueError(f"{role} has {unclassified} unclassified Conv blocks")
    op_counts = payload.get("op_counts")
    if not isinstance(op_counts, Mapping):
        raise ValueError(f"{role} lacks operation coverage")
    depth_to_space = int(op_counts.get("DepthToSpace", 0))
    concat = int(op_counts.get("Concat", 0))
    if depth_to_space <= 0 or concat <= 0:
        raise ValueError(f"{role} lacks F-Cooper DepthToSpace/Concat coverage")
    return {
        "role": role,
        "precision": "int8",
        "width": list(payload.get("width") or []),
        "result_path": str(path.resolve()),
        "result_sha256": sha256_file(path),
        "artifact_sha256": sha256_file(artifact),
        "onnx_sha256": sha256_file(onnx),
        "quant_contract_sha256": sha256_file(quant_path),
        "block_report_sha256": block_report_sha,
        "build_run_passed": True,
        "correctness_passed": True,
        "automatic_route_passed": True,
        "conv_count": conv_count,
        "tensorized_conv_count": tensorized,
        "fallback_conv_count": fallback,
        "schedule_failure_count": failures,
        "depth_to_space_count": depth_to_space,
        "concat_count": concat,
        "materialization_boundary_count": depth_to_space + concat,
        "schema": payload.get("schema"),
        "route_spec": route_spec,
        "method": method,
        "host": payload.get("host"),
    }


def validate_formal_inputs(
    formal_path: Path, scanner_path: Path, recovery_reports: Sequence[Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    formal = read_json(formal_path)
    if formal.get("schema_version") != "fcooper_formal_v2_contract":
        raise ValueError("unexpected F-Cooper formal contract schema")
    if formal.get("model") != "fcooper":
        raise ValueError("formal contract model must be fcooper")
    declared_scanner = resolve_path(formal.get("partition_path"), formal_path)
    if declared_scanner != scanner_path.resolve():
        raise ValueError("formal contract scanner path drift")
    require_file_sha(
        scanner_path,
        formal.get("partition_sha256"),
        label="scanner manifest",
    )
    if formal.get("probe_labels_allowed_in_training") is not False:
        raise ValueError("formal contract must forbid probe labels in training")
    if formal.get("probe_rows_allowed_as_winner") is not False:
        raise ValueError("formal contract must forbid probe rows as winner")
    recovery_contract = resolve_path(formal.get("recovery_contract_path"), formal_path)
    recovery_contract_sha = require_file_sha(
        recovery_contract,
        formal.get("recovery_contract_sha256"),
        label="recovery contract",
    )
    if not recovery_reports:
        raise ValueError("at least one backend-neutral recovery report is required")
    reports = []
    for report_path in recovery_reports:
        report = read_json(report_path)
        if (
            report.get("schema_version") != "fcooper_recovery_training_report_v2"
            or report.get("status") != "success"
        ):
            raise ValueError(f"recovery report is not successful v2 evidence: {report_path}")
        if report.get("recovery_contract_sha256") != recovery_contract_sha:
            raise ValueError(f"recovery contract SHA drift: {report_path}")
        _validate_optional_pair(
            report,
            report_path,
            "recovery_contract_path",
            "recovery_contract_sha256",
            label="recovery contract",
        )
        checkpoint_sha = _validate_optional_pair(
            report,
            report_path,
            "recovered_checkpoint_path",
            "recovered_checkpoint_sha256",
            label="recovered checkpoint",
        )
        config_sha = _validate_optional_pair(
            report,
            report_path,
            "config_path",
            "config_sha256",
            label="recovery config",
        )
        reports.append(
            {
                "report_path": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "schema_version": report["schema_version"],
                "status": "success",
                "recovered_checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "recovery_contract_sha256": recovery_contract_sha,
            }
        )
    formal_audit = {
        "formal_contract_path": str(formal_path.resolve()),
        "formal_contract_sha256": sha256_file(formal_path),
        "scanner_manifest_path": str(scanner_path.resolve()),
        "scanner_manifest_sha256": sha256_file(scanner_path),
        "recovery_contract_sha256": recovery_contract_sha,
        "width_schema_count": len(formal.get("width_schema") or []),
        "search_budget": formal.get("search_budget"),
    }
    recovery = {
        "schema_version": "fcooper_backend_neutral_recovery_numeric_gate_v1",
        "passed": True,
        "status": "passed",
        "t16_search_allowed": True,
        "scope": "backend_neutral_source_recovery_only",
        "excludes": [
            "backend_performance",
            "backend_accuracy",
            "TRT_artifacts",
            "probe_labels",
        ],
        "report_count": len(reports),
        "recovery_contract_sha256": recovery_contract_sha,
        "reports": reports,
    }
    return formal_audit, recovery


def _compiler_fingerprint(probes: Sequence[Mapping[str, Any]]) -> str:
    identity = {
        "dispatch_key": DISPATCH_KEY,
        "hardware_target": HARDWARE_TARGET,
        "hosts": sorted({str(item.get("host") or "") for item in probes}),
        "route_schemas": sorted({str(item.get("schema") or "") for item in probes}),
        "route_specs": sorted(
            {
                str(item.get("route_spec") or item.get("method") or "")
                for item in probes
            }
        ),
        "artifact_sha256": sorted(str(item["artifact_sha256"]) for item in probes),
    }
    return canonical_sha256(identity)


def _assert_label_free_profile(profile: Mapping[str, Any]) -> None:
    serialized = json.dumps(profile, ensure_ascii=True, sort_keys=True).lower()
    leaked = [token for token in TARGET_LABEL_TOKENS if token in serialized]
    if leaked:
        raise ValueError(f"capability profile contains target labels: {leaked}")


def finalize_capability(
    *,
    base_fp16_result: Path,
    boundary_fp16_result: Path,
    base_int8_result: Path,
    boundary_int8_result: Path,
    fp32_schedule_result: Path,
    base_int8_quant_contract: Path,
    boundary_int8_quant_contract: Path,
    recovery_reports: Sequence[Path],
    scanner_manifest: Path,
    formal_contract: Path,
    base_capability_profile: Path | None = None,
) -> dict[str, dict[str, Any]]:
    formal_audit, recovery = validate_formal_inputs(
        formal_contract, scanner_manifest, recovery_reports
    )
    base_quant = validate_quant_contract(base_int8_quant_contract, role="base")
    boundary_quant = validate_quant_contract(
        boundary_int8_quant_contract, role="boundary"
    )
    probes = [
        validate_fp_probe(base_fp16_result, precision="fp16", role="base"),
        validate_fp_probe(boundary_fp16_result, precision="fp16", role="boundary"),
        validate_fp_probe(fp32_schedule_result, precision="fp32", role="original"),
        validate_int8_probe(base_int8_result, role="base", quant=base_quant),
        validate_int8_probe(
            boundary_int8_result, role="boundary", quant=boundary_quant
        ),
    ]
    int8 = [item for item in probes if item["precision"] == "int8"]
    total_conv = sum(int(item["conv_count"]) for item in int8)
    tensorized = sum(int(item["tensorized_conv_count"]) for item in int8)
    inherited_features: dict[str, Any] = {}
    inherited_profile_digest = None
    if base_capability_profile is not None:
        payload = json.loads(base_capability_profile.read_text())
        candidates = payload if isinstance(payload, list) else (
            payload.get("capability_profiles")
            or payload.get("profiles")
            or [payload]
        )
        matches = [
            dict(item)
            for item in candidates
            if item.get("dispatch_key") == DISPATCH_KEY
            and item.get("hardware_target") == HARDWARE_TARGET
        ]
        if len(matches) != 1:
            raise ValueError("expected one inherited H800 TVM capability profile")
        inherited = validate_capability_profile(matches[0])
        inherited_features = dict(inherited.get("features") or {})
        inherited_profile_digest = inherited["capability_digest"]
    features: dict[str, float | int | None] = {
        **inherited_features,
        "probe_count": len(probes),
        "fp16_probe_build_coverage": 1.0,
        "fp32_schedule_probe_build_coverage": 1.0,
        "int8_probe_build_coverage": 1.0,
        "fp_probe_correctness_coverage": 1.0,
        "int8_exact_correctness_coverage": 1.0,
        "int8_automatic_route_coverage": 1.0,
        "int8_tensorized_conv_coverage": tensorized / total_conv,
        "int8_conv_count": total_conv,
        "int8_tensorized_conv_count": tensorized,
        "int8_fallback_conv_count": sum(
            int(item["fallback_conv_count"]) for item in int8
        ),
        "int8_schedule_failure_count": sum(
            int(item["schedule_failure_count"]) for item in int8
        ),
        "depth_to_space_count": sum(
            int(item["depth_to_space_count"]) for item in int8
        ),
        "concat_count": sum(int(item["concat_count"]) for item in int8),
        "materialization_boundary_count": sum(
            int(item["materialization_boundary_count"]) for item in int8
        ),
        "int8_quant_tensor_coverage": 1.0,
        "int8_quant_contract_count": 2,
        "int8_concat_quant_group_count": sum(
            int(item["concat_group_count"]) for item in (base_quant, boundary_quant)
        ),
        "recovery_source_coverage": 1.0,
        "recovery_report_count": recovery["report_count"],
        "structure_axis_count": formal_audit["width_schema_count"] or 5,
    }
    profile = build_capability_profile(
        capability_profile_id=PROFILE_ID,
        hardware_target=HARDWARE_TARGET,
        compiler_fingerprint=_compiler_fingerprint(probes),
        dispatch_key=DISPATCH_KEY,
        features=features,
    )
    _assert_label_free_profile(profile)
    input_evidence = {
        "formal_contract": formal_audit,
        "quant_contracts": [base_quant, boundary_quant],
        "probe_result_sha256": [item["result_sha256"] for item in probes],
        "recovery_report_sha256": [
            item["report_sha256"] for item in recovery["reports"]
        ],
        "inherited_capability_profile_digest": inherited_profile_digest,
    }
    probe_audit = {
        "schema_version": "fcooper_tvm_capability_probe_audit_v1",
        "passed": True,
        "all_probes_terminal": True,
        "profile_id": PROFILE_ID,
        "dispatch_key": DISPATCH_KEY,
        "label_free": True,
        "probe_count": len(probes),
        "probes": probes,
        "quant_contracts": [base_quant, boundary_quant],
        "input_evidence_digest": canonical_sha256(input_evidence),
    }
    probe_row_ids = []
    for item in probes:
        if item["precision"] not in {"fp16", "int8"}:
            continue
        width = item.get("width")
        if not isinstance(width, list) or len(width) != 5:
            probe_row_ids.append(
                f"probe::{item['role']}::{item['precision']}::{PROFILE_ID}"
            )
            continue
        axes = (
            "backbone.s0",
            "backbone.s1",
            "backbone.s2",
            "neck.deblock",
            "neck.output",
        )
        group = "fcooper|" + "|".join(
            f"{axis}={value}" for axis, value in zip(axes, width)
        )
        probe_row_ids.append(
            f"{group}|q={item['precision']}|profile={PROFILE_ID}"
        )
    isolation = {
        "schema_version": "fcooper_tvm_probe_isolation_audit_v1",
        "passed": True,
        "status": "passed",
        "profile_id": PROFILE_ID,
        "probe_row_ids": sorted(set(probe_row_ids)),
        "probe_metrics_allowed_as_cost_model_labels": False,
        "probe_rows_allowed_in_t16_budget": False,
        "probe_rows_allowed_as_winner": False,
        "policy": {
            "probe_labels_allowed_in_training": False,
            "probe_rows_allowed_in_t16": False,
            "probe_rows_allowed_as_winner": False,
            "probe_performance_fields_exported_to_profile": False,
        },
        "probe_result_sha256": [item["result_sha256"] for item in probes],
        "formal_contract_sha256": formal_audit["formal_contract_sha256"],
    }
    return {
        "capability_profiles": {
            "schema_version": "fcooper_tvm_capability_profiles_v1",
            "capability_profiles": [profile],
        },
        "probe_audit": probe_audit,
        "probe_isolation_audit": isolation,
        "recovery_numeric_gate_summary": recovery,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_outputs(output_dir: Path, outputs: Mapping[str, Mapping[str, Any]]) -> None:
    names = {
        "capability_profiles": "capability_profiles.json",
        "probe_audit": "probe_audit.json",
        "probe_isolation_audit": "probe_isolation_audit.json",
        "recovery_numeric_gate_summary": "recovery_numeric_gate_summary.json",
    }
    for key, filename in names.items():
        _write_json_atomic(output_dir / filename, outputs[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fp16-result", type=Path, required=True)
    parser.add_argument("--boundary-fp16-result", type=Path, required=True)
    parser.add_argument("--base-int8-result", type=Path, required=True)
    parser.add_argument("--boundary-int8-result", type=Path, required=True)
    parser.add_argument("--fp32-schedule-result", type=Path, required=True)
    parser.add_argument("--base-int8-quant-contract", type=Path, required=True)
    parser.add_argument("--boundary-int8-quant-contract", type=Path, required=True)
    parser.add_argument(
        "--recovery-report", type=Path, action="append", required=True
    )
    parser.add_argument("--scanner-manifest", type=Path, required=True)
    parser.add_argument("--formal-contract", type=Path, required=True)
    parser.add_argument("--base-capability-profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = finalize_capability(
        base_fp16_result=args.base_fp16_result,
        boundary_fp16_result=args.boundary_fp16_result,
        base_int8_result=args.base_int8_result,
        boundary_int8_result=args.boundary_int8_result,
        fp32_schedule_result=args.fp32_schedule_result,
        base_int8_quant_contract=args.base_int8_quant_contract,
        boundary_int8_quant_contract=args.boundary_int8_quant_contract,
        recovery_reports=args.recovery_report,
        scanner_manifest=args.scanner_manifest,
        formal_contract=args.formal_contract,
        base_capability_profile=args.base_capability_profile,
    )
    write_outputs(args.output_dir, outputs)
    print(
        json.dumps(
            {
                "status": "success",
                "profile_id": PROFILE_ID,
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
