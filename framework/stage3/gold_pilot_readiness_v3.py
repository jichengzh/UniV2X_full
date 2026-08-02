"""Stage3 pilot readiness gate for the Gold Cold-start v3 pilot groups."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "stage3_gold_pilot_readiness_v3"
SOURCE_MANIFEST_SCHEMA = "stage3_gold_coldstart96_manifest_v3"
ALLOWED_RUNNER_KINDS = {"tvm_auto", "trt_engine"}
FORBIDDEN_TVM_AUTO_SOURCE_REALIZATIONS = {
    "hand-rewrite",
    "hand_rewrite",
    "mixed",
    "byoc-trt",
    "byoc_trt",
    "byoctrt",
}


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError(f"expected {SOURCE_MANIFEST_SCHEMA} manifest")
    jobs = manifest.get("jobs")
    profiles = manifest.get("capability_profiles")
    pilot_group_ids = manifest.get("pilot_group_ids")
    if not isinstance(jobs, list):
        raise ValueError("manifest.jobs must be a list")
    if not isinstance(profiles, list):
        raise ValueError("manifest.capability_profiles must be a list")
    if not isinstance(pilot_group_ids, list) or len(pilot_group_ids) != 2:
        raise ValueError("manifest.pilot_group_ids must be a two-item list")
    if len({str(group_id) for group_id in pilot_group_ids}) != len(pilot_group_ids):
        raise ValueError("manifest.pilot_group_ids must be unique")


def _pilot_group_ids(manifest: Mapping[str, Any]) -> tuple[str, str]:
    return tuple(str(group_id) for group_id in manifest["pilot_group_ids"])


def _profile_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for source in manifest["capability_profiles"]:
        profile = copy.deepcopy(dict(source))
        profile_id = str(profile.get("capability_profile_id") or "")
        if not profile_id:
            raise ValueError("capability profile missing capability_profile_id")
        mapping[profile_id] = profile
    return mapping


def _pilot_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    allowed_group_ids = set(_pilot_group_ids(manifest))
    rows = [copy.deepcopy(dict(row)) for row in manifest["jobs"] if str(row.get("group_id")) in allowed_group_ids]
    rows.sort(
        key=lambda row: (
            str(row.get("group_id") or ""),
            str(row.get("q_mode") or ""),
            str(row.get("capability_profile_id") or ""),
        )
    )
    if len(rows) != 8:
        raise ValueError(f"expected exactly 8 pilot rows, got {len(rows)}")
    if {str(row.get("group_id")) for row in rows} != allowed_group_ids:
        raise ValueError("pilot rows must cover both pilot groups exactly once per q/profile combination")
    return rows


def _dispatch_to_runner_kind(dispatch_key: str) -> str | None:
    dispatch = str(dispatch_key or "")
    if dispatch in ALLOWED_RUNNER_KINDS:
        return dispatch
    return None


def _staging_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    staging = row.get("staging_contract")
    if staging is None:
        return {}
    if not isinstance(staging, Mapping):
        raise ValueError(f"row {row.get('job_id')} staging_contract must be a mapping")
    return dict(staging)


def _record_check(
    checks: dict[str, dict[str, Any]],
    gaps: list[dict[str, str]],
    *,
    field: str,
    value: Any,
    gap_code_missing: str,
    gap_code_invalid: str | None = None,
    required: bool = True,
) -> None:
    if value is None or value == "":
        checks[field] = {"status": "missing", "value": None, "required": required}
        if required:
            gaps.append({"code": gap_code_missing, "field": field})
        return
    text = str(value)
    if not _is_sha256(text):
        checks[field] = {"status": "invalid", "value": text, "required": required}
        gaps.append({"code": gap_code_invalid or gap_code_missing, "field": field})
        return
    checks[field] = {"status": "present", "value": text, "required": required}


def _source_realization_text(row: Mapping[str, Any], staging: Mapping[str, Any]) -> str:
    for key in ("source_realization", "realization_origin"):
        value = staging.get(key)
        if value:
            return str(value)
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _evaluate_row(row: Mapping[str, Any], profiles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    profile_id = str(row.get("capability_profile_id") or "")
    if profile_id not in profiles:
        raise ValueError(f"row {row.get('job_id')} references unknown capability_profile_id {profile_id!r}")
    profile = profiles[profile_id]
    staging = _staging_contract(row)
    checks: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, str]] = []

    _record_check(
        checks,
        gaps,
        field="source_checkpoint_sha256",
        value=staging.get("source_checkpoint_sha256"),
        gap_code_missing="missing_source_checkpoint_sha256",
        gap_code_invalid="invalid_source_checkpoint_sha256",
    )
    _record_check(
        checks,
        gaps,
        field="source_onnx_sha256",
        value=staging.get("source_onnx_sha256"),
        gap_code_missing="missing_source_onnx_sha256",
        gap_code_invalid="invalid_source_onnx_sha256",
    )

    expected_compiler = str(profile.get("compiler_fingerprint") or "")
    actual_compiler = staging.get("profile_compiler_fingerprint", expected_compiler)
    if actual_compiler is None or actual_compiler == "":
        checks["profile_compiler_fingerprint"] = {"status": "missing", "value": None, "required": True}
        gaps.append({"code": "missing_profile_compiler_fingerprint", "field": "profile_compiler_fingerprint"})
    else:
        actual_compiler_text = str(actual_compiler)
        if not _is_sha256(actual_compiler_text):
            checks["profile_compiler_fingerprint"] = {
                "status": "invalid",
                "value": actual_compiler_text,
                "required": True,
            }
            gaps.append({"code": "invalid_profile_compiler_fingerprint", "field": "profile_compiler_fingerprint"})
        elif expected_compiler and actual_compiler_text != expected_compiler:
            checks["profile_compiler_fingerprint"] = {
                "status": "mismatch",
                "value": actual_compiler_text,
                "required": True,
                "expected": expected_compiler,
            }
            gaps.append({"code": "profile_compiler_fingerprint_mismatch", "field": "profile_compiler_fingerprint"})
        else:
            checks["profile_compiler_fingerprint"] = {
                "status": "present",
                "value": actual_compiler_text,
                "required": True,
            }

    dispatch_key = str(row.get("dispatch_key") or "")
    derived_runner_kind = _dispatch_to_runner_kind(dispatch_key)
    configured_runner_kind = str(staging.get("runner_kind") or derived_runner_kind or "")
    if not configured_runner_kind:
        checks["runner_kind"] = {"status": "missing", "value": None, "required": True}
        gaps.append({"code": "missing_runner_kind", "field": "runner_kind"})
    elif configured_runner_kind not in ALLOWED_RUNNER_KINDS:
        checks["runner_kind"] = {"status": "invalid", "value": configured_runner_kind, "required": True}
        gaps.append({"code": "invalid_runner_kind", "field": "runner_kind"})
    elif derived_runner_kind is None:
        checks["runner_kind"] = {
            "status": "unsupported_dispatch_key",
            "value": configured_runner_kind,
            "required": True,
            "dispatch_key": dispatch_key,
        }
        gaps.append({"code": "unsupported_dispatch_key", "field": "dispatch_key"})
    elif configured_runner_kind != derived_runner_kind:
        checks["runner_kind"] = {
            "status": "mismatch",
            "value": configured_runner_kind,
            "required": True,
            "expected": derived_runner_kind,
        }
        gaps.append({"code": "runner_kind_dispatch_mismatch", "field": "runner_kind"})
    else:
        checks["runner_kind"] = {
            "status": "present",
            "value": configured_runner_kind,
            "required": True,
        }

    source_realization = _source_realization_text(row, staging).strip().lower()
    if configured_runner_kind == "tvm_auto" and source_realization in FORBIDDEN_TVM_AUTO_SOURCE_REALIZATIONS:
        gaps.append({"code": "forbidden_tvm_auto_source_realization", "field": "source_realization"})
        checks["source_realization"] = {
            "status": "forbidden",
            "value": source_realization,
            "required": False,
        }
    elif source_realization:
        checks["source_realization"] = {
            "status": "informational",
            "value": source_realization,
            "required": False,
        }

    requires_calibration = str(row.get("q_mode") or "") == "int8"
    _record_check(
        checks,
        gaps,
        field="calibration_sha256",
        value=staging.get("calibration_sha256"),
        gap_code_missing="missing_calibration_sha256",
        gap_code_invalid="invalid_calibration_sha256",
        required=requires_calibration,
    )
    _record_check(
        checks,
        gaps,
        field="ap_eval_protocol_sha256",
        value=staging.get("ap_eval_protocol_sha256"),
        gap_code_missing="missing_ap_eval_protocol_sha256",
        gap_code_invalid="invalid_ap_eval_protocol_sha256",
    )
    _record_check(
        checks,
        gaps,
        field="latency_energy_protocol_sha256",
        value=staging.get("latency_energy_protocol_sha256"),
        gap_code_missing="missing_latency_energy_protocol_sha256",
        gap_code_invalid="invalid_latency_energy_protocol_sha256",
    )

    return {
        "job_id": str(row.get("job_id") or ""),
        "group_id": str(row.get("group_id") or ""),
        "model": str(row.get("model") or ""),
        "width_key": str(row.get("width_key") or ""),
        "q_mode": str(row.get("q_mode") or ""),
        "capability_profile_id": profile_id,
        "dispatch_key": dispatch_key,
        "checks": checks,
        "gaps": gaps,
        "status": "ready" if not gaps else "blocked",
    }


def build_gold_pilot_readiness(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    _require_manifest(manifest)
    profiles = _profile_map(manifest)
    rows = [_evaluate_row(row, profiles) for row in _pilot_rows(manifest)]
    blocked_gap_counts: dict[str, int] = {}
    for row in rows:
        for gap in row["gaps"]:
            blocked_gap_counts[gap["code"]] = blocked_gap_counts.get(gap["code"], 0) + 1
    manifest_sha256 = (
        sha256_file(manifest_path) if manifest_path is not None else _sha256_payload(manifest)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_schema_version": str(manifest.get("schema_version") or ""),
        "source_manifest_sha256": manifest_sha256,
        "source_manifest_path": str(manifest_path) if manifest_path is not None else "",
        "pilot_group_ids": list(_pilot_group_ids(manifest)),
        "pilot_row_count": len(rows),
        "status": "ready" if not blocked_gap_counts else "blocked",
        "dispatch_execution_mode": "staging_contract_only_no_h800_execution",
        "fills_from_measurement_history": False,
        "blocked_gap_counts": blocked_gap_counts,
        "rows": rows,
    }


def write_gold_pilot_readiness(
    readiness: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(dict(readiness))
    digest = _sha256_payload(payload)
    path = destination / f"gold_pilot_readiness_v3-{digest}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA",
    "build_gold_pilot_readiness",
    "sha256_file",
    "write_gold_pilot_readiness",
]
