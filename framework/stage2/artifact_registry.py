"""Artifact registry v1 for Stage2 LUT production.

The artifact registry links productized latency/AP/energy rows back to the
model/build/eval artifacts that made them possible. Evidence registry answers
"what can we claim"; artifact registry answers "what produced this row".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .lut_productization import LutProductizationError, utc_timestamp


ARTIFACT_REGISTRY_SCHEMA_VERSION = "stage2_artifact_registry_v1"
VALID_ARTIFACT_STATUSES = {"ready", "missing", "blocked", "quarantined"}


def _require(row: dict[str, Any], fields: list[str]) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise LutProductizationError(f"missing artifact registry fields: {missing}")


def _safe(value: object) -> str:
    return str(value).replace("/", "-").replace(":", "-").replace(" ", "_")


def _short_hash(parts: list[object]) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8"))
    return digest.hexdigest()[:10]


def _label_from_row(row: dict[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id") or "")
    if ":" in candidate_id:
        label = candidate_id.split(":")[-1]
        if label:
            return label
    config_id = str(row.get("config_id") or "")
    for token in (
        "base",
        "p50b2",
        "p50",
        "p75",
        "trap25",
        "pad64",
        "mix_a",
        "mix_b",
        "mix_c",
        "mix_d",
        "mix_e",
        "iso_s0",
        "iso_s1",
        "iso_s2",
    ):
        if token in config_id:
            return token
    return config_id or "unknown"


def _arm_from_row(row: dict[str, Any], label: str) -> str:
    quant_policy = str(row.get("quant_policy") or "").lower()
    schedule_policy = str(row.get("schedule_policy") or "").lower()
    if "int8" in quant_policy or "quant" in quant_policy:
        return "Q"
    if label.startswith(("iso_", "mix_")) or "schedule" in schedule_policy:
        return "S"
    if label.startswith(("p", "trap")) or "prune" in label:
        return "P"
    if label == "base":
        return "baseline"
    return "mixed"


def _source_files(rows: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    for row in rows:
        for item in row.get("source_files", []) or []:
            text = str(item)
            if text and text not in files:
                files.append(text)
    return files


def _first_source_with(rows: list[dict[str, Any]], predicate: Any) -> str | None:
    for item in _source_files(rows):
        if predicate(item):
            return item
    return None


def _raw_artifact(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        value = row.get("raw_artifact")
        if value:
            return str(value)
    return None


def _group_by_config(*row_groups: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    names = ["latency", "ap", "energy"]
    for name, rows in zip(names, row_groups, strict=True):
        for row in rows:
            config_id = str(row.get("config_id") or "")
            if not config_id:
                continue
            grouped.setdefault(config_id, {"latency": [], "ap": [], "energy": []})
            grouped[config_id][name].append(row)
    return grouped


def artifact_id_for_row(row: dict[str, Any]) -> str:
    config_id = str(row.get("config_id") or "unknown")
    model = str(row.get("model") or row.get("model_name") or "unknown")
    width = row.get("width") or []
    suffix = _short_hash(
        [
            model,
            config_id,
            width,
            row.get("quant_policy"),
            row.get("schedule_policy"),
            row.get("optimized_scope"),
        ]
    )
    return f"artifact:{_safe(model)}:{_safe(config_id)}:{suffix}"


def build_artifact_registry_rows(
    *,
    latency_rows: list[dict[str, Any]] | None = None,
    ap_rows: list[dict[str, Any]] | None = None,
    energy_rows: list[dict[str, Any]] | None = None,
    quarantine_rows: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> list[dict[str, Any]]:
    latency_rows = list(latency_rows or [])
    ap_rows = list(ap_rows or [])
    energy_rows = list(energy_rows or [])
    quarantine_rows = list(quarantine_rows or [])
    grouped = _group_by_config(latency_rows, ap_rows, energy_rows)
    now = created_at or utc_timestamp()
    registry_rows: list[dict[str, Any]] = []

    for config_id in sorted(grouped):
        groups = grouped[config_id]
        all_rows = [*groups["latency"], *groups["ap"], *groups["energy"]]
        if not all_rows:
            continue
        sample = all_rows[0]
        label = _label_from_row(sample)
        onnx_path = _first_source_with(all_rows, lambda value: value.endswith(".onnx"))
        tvm_work_dir = _first_source_with(all_rows, _looks_like_tvm_work_dir)
        database_path = tvm_work_dir
        ap_source_path = _raw_artifact(groups["ap"])
        energy_raw_dir = _raw_artifact(groups["energy"])
        quarantine_refs = [
            str(row.get("job_id") or row.get("config_id"))
            for row in quarantine_rows
            if row.get("config_id") == config_id and row.get("status") == "active"
        ]

        missing_artifacts: list[str] = []
        if (groups["latency"] or groups["energy"]) and not onnx_path:
            missing_artifacts.append("onnx_path")
        if groups["latency"] and not tvm_work_dir:
            missing_artifacts.append("tvm_work_dir")
        if groups["ap"] and not ap_source_path:
            missing_artifacts.append("ap_source_path")
        if groups["energy"] and not energy_raw_dir:
            missing_artifacts.append("energy_raw_dir")

        status = "ready"
        if quarantine_refs:
            status = "quarantined"
        elif missing_artifacts:
            status = "missing"

        row = {
            "schema_version": ARTIFACT_REGISTRY_SCHEMA_VERSION,
            "artifact_id": artifact_id_for_row(sample),
            "config_id": config_id,
            "model_name": str(sample.get("model") or "unknown"),
            "label": label,
            "width": list(sample.get("width") or []),
            "arm": _arm_from_row(sample, label),
            "optimized_scope": str(sample.get("optimized_scope") or "unknown"),
            "onnx_path": onnx_path,
            "tvm_work_dir": tvm_work_dir,
            "database_path": database_path,
            "ap_source_path": ap_source_path,
            "energy_raw_dir": energy_raw_dir,
            "latency_row_ids": [str(row["row_id"]) for row in groups["latency"]],
            "ap_row_ids": [str(row["row_id"]) for row in groups["ap"]],
            "energy_row_ids": [str(row["row_id"]) for row in groups["energy"]],
            "artifact_status": status,
            "missing_artifacts": missing_artifacts,
            "quarantine_refs": quarantine_refs,
            "source_files": _source_files(all_rows),
            "created_at": now,
            "updated_at": now,
        }
        validate_artifact_registry_row(row)
        registry_rows.append(row)

    return registry_rows


def validate_artifact_registry_row(row: dict[str, Any]) -> None:
    _require(
        row,
        [
            "schema_version",
            "artifact_id",
            "config_id",
            "model_name",
            "label",
            "width",
            "arm",
            "optimized_scope",
            "onnx_path",
            "tvm_work_dir",
            "database_path",
            "ap_source_path",
            "energy_raw_dir",
            "latency_row_ids",
            "ap_row_ids",
            "energy_row_ids",
            "artifact_status",
            "missing_artifacts",
            "quarantine_refs",
            "created_at",
            "updated_at",
        ],
    )
    if row["schema_version"] != ARTIFACT_REGISTRY_SCHEMA_VERSION:
        raise LutProductizationError(
            f"unexpected artifact registry schema_version: {row['schema_version']}"
        )
    if row["artifact_status"] not in VALID_ARTIFACT_STATUSES:
        raise LutProductizationError(
            f"unknown artifact_status: {row['artifact_status']}"
        )
    for field in (
        "width",
        "latency_row_ids",
        "ap_row_ids",
        "energy_row_ids",
        "missing_artifacts",
        "quarantine_refs",
    ):
        if not isinstance(row[field], list):
            raise LutProductizationError(f"{field} must be a list")
    if not row["artifact_id"]:
        raise LutProductizationError("artifact_id must be non-empty")
    if not row["config_id"]:
        raise LutProductizationError("config_id must be non-empty")
    if row["artifact_status"] == "ready" and row["missing_artifacts"]:
        raise LutProductizationError("ready artifact cannot list missing_artifacts")


def summarize_artifact_registry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        validate_artifact_registry_row(row)
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row["artifact_status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "schema": "stage2_artifact_registry_summary_v1",
        "total_artifacts": len(rows),
        "ready_artifacts": statuses.get("ready", 0),
        "missing_artifacts": statuses.get("missing", 0),
        "blocked_artifacts": statuses.get("blocked", 0),
        "quarantined_artifacts": statuses.get("quarantined", 0),
        "status_counts": statuses,
    }


def artifact_ready_for_config(rows: list[dict[str, Any]], config_id: str) -> bool:
    for row in rows:
        if row.get("config_id") == config_id and row.get("artifact_status") == "ready":
            return True
    return False


def existing_path_status(path: str | None) -> str:
    if not path:
        return "missing"
    return "exists" if Path(path).exists() else "unverified"


def _looks_like_tvm_work_dir(value: str) -> bool:
    path = Path(value)
    name = path.name
    return (
        name.startswith("ms_")
        or "ms_work" in value
        or "/database" in value
        or value.endswith(".db")
    )
