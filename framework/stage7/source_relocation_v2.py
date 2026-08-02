"""Deterministic Stage7 source-output relocation into the formal v2 root."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage5.measurement_plan_v2 import _validate_request


SCHEMA_VERSION = "stage7_source_contract_relocation_v2"
_OUTPUT_PATH_FIELDS = (
    "checkpoint_dir",
    "checkpoint_path",
    "config_path",
    "training_done_marker",
    "onnx_path",
    "onnx_report_path",
    "calibration_root",
    "calibration_npz",
    "calibration_summary",
    "trt_calibration_dir",
    "source_done_marker",
)
_RESET_SHA_FIELDS = (
    "checkpoint_sha256",
    "onnx_sha256",
)
_REMOVE_BOUND_FIELDS = (
    "calibration_npz_sha256",
    "calibration_summary_sha256",
    "materialization_evidence_path",
    "materialization_evidence_sha256",
    "trt_calibration_manifest_sha256",
    "trt_calibration_sample_count",
)
_LABEL_FIELDS = {
    "ap",
    "ap30",
    "ap50",
    "ap70",
    "cache",
    "cache_hit",
    "energy",
    "energy_j",
    "failure_reason",
    "latency",
    "latency_ms",
    "objective",
    "objectives",
    "terminal_status",
}


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_root(v2_root: Path) -> Path:
    raw = Path(v2_root)
    if not raw.is_absolute():
        raise ValueError("formal v2 root must be absolute")
    if ".." in raw.parts:
        raise ValueError("formal v2 root cannot contain parent traversal")
    return raw.resolve(strict=False)


def _candidate_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def _width_slug(row: Mapping[str, Any]) -> str:
    width = row.get("width")
    if (
        not isinstance(width, list)
        or len(width) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in width
        )
    ):
        raise ValueError("Pyramid source width must contain three positive integers")
    return "x".join(f"{value:03d}" for value in width)


def _label_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for raw_key, nested in value.items():
            key = re.sub(r"[^a-z0-9]+", "_", str(raw_key).lower()).strip("_")
            path = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            if key in _LABEL_FIELDS or key.startswith(
                ("cache_", "cached_", "failure_", "objective_", "terminal_")
            ):
                result.append(path)
            result.extend(_label_paths(nested, path))
        return result
    if isinstance(value, (list, tuple)):
        return [
            path
            for index, nested in enumerate(value)
            for path in _label_paths(nested, f"{prefix}[{index}]")
        ]
    return []


def _source_root(v2_root: Path, row: Mapping[str, Any]) -> Path:
    return v2_root / "sources" / "pyramid" / _width_slug(row)


def _relocated_contract(
    source: Mapping[str, Any],
    *,
    destination: Path,
    width_slug: str,
) -> dict[str, Any]:
    relocated = copy.deepcopy(dict(source))
    checkpoint = destination / "checkpoint"
    onnx = destination / "onnx"
    calibration = destination / "calibration"
    relocated.update(
        {
            "checkpoint_dir": str(checkpoint),
            "checkpoint_path": str(checkpoint / "stage5_best.pth"),
            "checkpoint_sha256": None,
            "config_path": str(checkpoint / "config.yaml"),
            "training_done_marker": str(
                checkpoint / "stage5_training_complete.json"
            ),
            "onnx_path": str(
                onnx / f"pyramid_{width_slug}_multiscale.onnx"
            ),
            "onnx_sha256": None,
            "onnx_report_path": str(onnx / "onnx_export_report.json"),
            "calibration_root": str(calibration),
            "calibration_npz": str(
                calibration / "spatial_features_train16.npz"
            ),
            "calibration_summary": str(calibration / "summary.json"),
            "trt_calibration_dir": str(calibration / "trt_npy"),
            "source_done_marker": str(destination / "source_ready.done"),
        }
    )
    for field in _REMOVE_BOUND_FIELDS:
        relocated.pop(field, None)
    return relocated


def _resign_request(request: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    unsigned["rows"] = copy.deepcopy(rows)
    unsigned["row_sha256"] = {
        _candidate_id(row): _sha(row) for row in rows
    }
    return {
        **unsigned,
        "measurement_request_sha256": _sha(unsigned),
    }


def relocate_pyramid_request_to_v2_root(
    request: Mapping[str, Any],
    *,
    v2_root: Path,
) -> dict[str, Any]:
    """Relocate only generated Pyramid source outputs, then re-sign the request."""
    root = _canonical_root(v2_root)
    copied = copy.deepcopy(dict(request))
    raw_rows = copied.get("rows")
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != 4
        or any(not isinstance(row, Mapping) for row in raw_rows)
    ):
        raise ValueError("source relocation requires one four-row request")
    if {str(row.get("model") or "") for row in raw_rows} != {"pyramid"}:
        raise ValueError("source relocation supports only Pyramid requests")
    _validate_request(copied)
    leaks = _label_paths(raw_rows)
    if leaks:
        raise ValueError("source relocation request contains cache or label fields")

    ordered_ids = [_candidate_id(row) for row in raw_rows]
    relocated_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row = copy.deepcopy(dict(raw_row))
        source = row.get("source_contract")
        if not isinstance(source, Mapping):
            raise ValueError("Pyramid row source contract is missing")
        width_slug = _width_slug(row)
        destination = _source_root(root, row)
        relocated_source = _relocated_contract(
            source,
            destination=destination,
            width_slug=width_slug,
        )
        before_source_plan = str(row.get("source_evidence_sha256") or "")
        row["source_contract"] = relocated_source
        row["source_evidence_sha256"] = _source_plan_sha(row)
        changed = sorted(
            {
                key
                for key in set(source) | set(relocated_source)
                if source.get(key) != relocated_source.get(key)
            }
        )
        relocated_rows.append(row)
        audit_rows.append(
            {
                "candidate_id": _candidate_id(row),
                "width": copy.deepcopy(row["width"]),
                "source_root": str(destination),
                "pre_source_contract_sha256": _sha(dict(source)),
                "relocated_source_contract_sha256": _sha(relocated_source),
                "pre_source_plan_sha256": before_source_plan,
                "relocated_source_plan_sha256": row[
                    "source_evidence_sha256"
                ],
                "changed_fields": changed,
            }
        )

    relocated_request = _resign_request(copied, relocated_rows)
    _validate_request(relocated_request)
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "relocated_and_frozen",
        "v2_root": str(root),
        "source_root": str(root / "sources"),
        "pre_relocation_request_sha256": copied[
            "measurement_request_sha256"
        ],
        "relocated_request_sha256": relocated_request[
            "measurement_request_sha256"
        ],
        "ordered_candidate_ids": ordered_ids,
        "ordered_candidate_ids_sha256": _sha(ordered_ids),
        "selected_ids_changed": False,
        "cache_or_label_fields_observed": [],
        "cache_membership_observed": False,
        "selected_event_budget_delta": 0,
        "rows": audit_rows,
    }
    result = {
        "request": relocated_request,
        "audit": {
            **audit_payload,
            "source_relocation_sha256": _sha(audit_payload),
        },
    }
    return validate_source_relocation(result, v2_root=root)


def _require_under(path_value: Any, root: Path) -> Path:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("relocated source path is not absolute")
    resolved = Path(path_value).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("relocated path escapes the v2 source root") from error
    return resolved


def validate_source_relocation(
    result: Mapping[str, Any],
    *,
    v2_root: Path,
) -> dict[str, Any]:
    """Authenticate a relocation receipt and its re-signed logical request."""
    root = _canonical_root(v2_root)
    if not isinstance(result, Mapping) or set(result) != {"request", "audit"}:
        raise ValueError("source relocation result shape drift")
    request = copy.deepcopy(dict(result["request"]))
    audit = copy.deepcopy(dict(result["audit"]))
    _validate_request(request)
    recorded = audit.pop("source_relocation_sha256", None)
    if recorded != _sha(audit):
        raise ValueError("source relocation audit SHA drift")
    ordered_ids = [_candidate_id(row) for row in request["rows"]]
    if (
        audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("status") != "relocated_and_frozen"
        or audit.get("v2_root") != str(root)
        or audit.get("source_root") != str(root / "sources")
        or audit.get("relocated_request_sha256")
        != request["measurement_request_sha256"]
        or audit.get("ordered_candidate_ids") != ordered_ids
        or audit.get("ordered_candidate_ids_sha256") != _sha(ordered_ids)
        or audit.get("selected_ids_changed") is not False
        or audit.get("cache_or_label_fields_observed") != []
        or audit.get("cache_membership_observed") is not False
        or audit.get("selected_event_budget_delta") != 0
    ):
        raise ValueError("source relocation selected identity or contract drift")
    if _label_paths(request["rows"]):
        raise ValueError("relocated request contains cache or label fields")

    audit_rows = audit.get("rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != 4:
        raise ValueError("source relocation row audit coverage drift")
    audit_by_id = {
        str(row.get("candidate_id") or ""): row for row in audit_rows
    }
    if set(audit_by_id) != set(ordered_ids):
        raise ValueError("source relocation selected row audit drift")
    for row in request["rows"]:
        candidate_id = _candidate_id(row)
        source = row.get("source_contract")
        if not isinstance(source, Mapping):
            raise ValueError("relocated source contract is missing")
        expected_root = _source_root(root, row)
        for field in _OUTPUT_PATH_FIELDS:
            _require_under(source.get(field), expected_root)
        if any(source.get(field) is not None for field in _RESET_SHA_FIELDS):
            raise ValueError("unmaterialized relocated source carries artifact SHA")
        if _source_plan_sha(row) != row.get("source_evidence_sha256"):
            raise ValueError("relocated source plan SHA drift")
        row_audit = audit_by_id[candidate_id]
        if (
            row_audit.get("source_root") != str(expected_root)
            or row_audit.get("relocated_source_contract_sha256")
            != _sha(dict(source))
            or row_audit.get("relocated_source_plan_sha256")
            != row.get("source_evidence_sha256")
        ):
            raise ValueError("source relocation row binding drift")
    return {
        "request": request,
        "audit": {
            **audit,
            "source_relocation_sha256": recorded,
        },
    }


__all__ = [
    "relocate_pyramid_request_to_v2_root",
    "validate_source_relocation",
]
