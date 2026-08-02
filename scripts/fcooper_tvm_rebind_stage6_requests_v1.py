#!/usr/bin/env python3
"""Rebind frozen F-Cooper TRT Stage6 controls to a TVM capability profile."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage2.canonical_search_v3 import validate_capability_profile


REQUEST_SCHEMA = "stage6_fcooper_tvm_control_measurement_request_v1"
MANIFEST_SCHEMA = "stage6_fcooper_tvm_control_request_manifest_v1"
TASK_SCHEMA = "stage6_fcooper_tvm_control_task_contract_v1"
AUDIT_SCHEMA = "stage6_fcooper_tvm_request_rebind_audit_v1"

ARM_CONTRACTS = {
    "compression_only": {
        "phase": "measure",
        "expected_count": 16,
        "tvm_trials": 0,
        "task_id": "S6-FCO-TVM-COMPRESSION-ONLY-MEASURE-V1",
    },
    "compress_then_tune": {
        "phase": "screen",
        "expected_count": 12,
        "tvm_trials": 0,
        "task_id": "S6-FCO-TVM-COMPRESS-THEN-TUNE-SCREEN-V1",
    },
    "schedule_only": {
        "phase": "measure",
        "expected_count": 1,
        "tvm_trials": 64,
        "task_id": "S6-FCO-TVM-SCHEDULE-ONLY-MEASURE-V1",
    },
}

DERIVED_ROW_FIELDS = {
    "row_id",
    "manifest_job_id",
    "task_id",
    "task_sha256",
    "arm_id",
    "phase",
    "backend",
    "dispatch_key",
    "capability_profile_id",
    "capability_digest",
    "builder_optimization_level",
    "tvm_trials",
    "control_source_task_id",
    "control_source_task_sha256",
}

PERFORMANCE_ARTIFACT_FIELDS = {
    "latency_ms",
    "latency_p50_ms",
    "latency_p99_ms",
    "energy_j",
    "ap30",
    "ap50",
    "ap70",
    "compiled_engine",
    "compiled_engine_path",
    "compiled_engine_sha256",
    "engine_path",
    "engine_sha256",
    "engine_build_config",
    "engine_inspector",
    "tactic",
    "tactics",
    "tactic_sources",
    "prediction_path",
    "prediction_sha256",
    "ap_report_path",
    "ap_report_sha256",
    "actual_feedback_row_sha256",
    "terminal_status",
    "winner",
}


def _is_performance_artifact_field(name: str) -> bool:
    lowered = name.lower()
    if lowered == "engine_agent_batch":
        return False
    return (
        lowered in PERFORMANCE_ARTIFACT_FIELDS
        or lowered.startswith(("latency_", "energy_", "predicted_latency"))
        or lowered.startswith(("ap30", "ap50", "ap70"))
        or "compiled_artifact" in lowered
        or "prediction_output" in lowered
        or "engine" in lowered
        or "tactic" in lowered
    )


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_immutable(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
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


def _artifact_fields(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            field_path = f"{prefix}.{name}" if prefix else name
            if _is_performance_artifact_field(name):
                found.append(field_path)
            found.extend(_artifact_fields(item, field_path))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            found.extend(_artifact_fields(item, f"{prefix}[{index}]"))
    return found


def _request_path(manifest_path: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else manifest_path.parent / path


def _load_frozen_requests(
    manifest_path: Path,
    *,
    expected_arm: str,
    expected_phase: str,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_payload = _read_json(manifest_path)
    if not isinstance(manifest_payload, Mapping):
        raise ValueError(f"request manifest must be an object: {manifest_path}")
    manifest = copy.deepcopy(dict(manifest_payload))
    recorded_manifest_sha = manifest.pop("manifest_sha256", None)
    if recorded_manifest_sha != canonical_sha256(manifest):
        raise ValueError(f"source manifest SHA drift: {manifest_path}")
    if (
        manifest.get("arm_id") != expected_arm
        or manifest.get("phase") != expected_phase
    ):
        raise ValueError(f"source manifest arm/phase drift: {manifest_path}")

    requests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    source_request_shas: list[str] = []
    for expected_index, entry in enumerate(manifest.get("requests") or []):
        if int(entry.get("request_index", -1)) != expected_index:
            raise ValueError("source request ordering is not contiguous")
        path = _request_path(manifest_path, entry.get("path"))
        request_payload = _read_json(path)
        if not isinstance(request_payload, Mapping):
            raise ValueError(f"source request must be an object: {path}")
        request = copy.deepcopy(dict(request_payload))
        recorded_request_sha = request.pop("measurement_request_sha256", None)
        if (
            recorded_request_sha != canonical_sha256(request)
            or recorded_request_sha != entry.get("measurement_request_sha256")
        ):
            raise ValueError(f"source request SHA drift: {path}")
        task_contract = request.get("task_contract")
        if (
            not isinstance(task_contract, Mapping)
            or request.get("task_sha256") != canonical_sha256(task_contract)
            or request.get("task_id") != task_contract.get("task_id")
        ):
            raise ValueError(f"source task contract SHA drift: {path}")
        if (
            request.get("arm_id") != expected_arm
            or request.get("phase") != expected_phase
        ):
            raise ValueError(f"source request arm/phase drift: {path}")
        if task_contract.get("backend") != "trt":
            raise ValueError(f"source request is not a TRT contract: {path}")
        request_rows = request.get("rows")
        if not isinstance(request_rows, list):
            raise ValueError(f"source request rows are invalid: {path}")
        if (
            int(request.get("batch_size", -1)) != len(request_rows)
            or int(entry.get("batch_size", -1)) != len(request_rows)
        ):
            raise ValueError(f"source request batch size drift: {path}")
        for row in request_rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"source request row is invalid: {path}")
            row_copy = copy.deepcopy(dict(row))
            row_id = str(
                row_copy.get("row_id") or row_copy.get("manifest_job_id") or ""
            )
            if (
                not row_id
                or row_copy.get("task_id") != request.get("task_id")
                or row_copy.get("task_sha256") != request.get("task_sha256")
                or (request.get("row_sha256") or {}).get(row_id)
                != canonical_sha256(row_copy)
            ):
                raise ValueError(f"source row identity/SHA drift: {path}")
            forbidden = _artifact_fields(row_copy)
            if forbidden:
                raise ValueError(
                    "source row contains TRT performance artifact fields: "
                    + ", ".join(forbidden)
                )
            rows.append(row_copy)
        requests.append(dict(request_payload))
        source_request_shas.append(str(recorded_request_sha))

    if (
        len(rows) != expected_count
        or int(manifest.get("row_count", -1)) != expected_count
        or int(manifest.get("request_count", -1)) != len(requests)
    ):
        raise ValueError(
            f"{expected_arm} expected {expected_count} frozen rows, got {len(rows)}"
        )
    identities = [
        str(row.get("row_id") or row.get("manifest_job_id") or "") for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{expected_arm} source row identities are not unique")
    provenance = {
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": str(recorded_manifest_sha),
        "source_request_sha256": source_request_shas,
        "source_row_ids": identities,
    }
    return rows, provenance


def _genome(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "width": copy.deepcopy(row.get("width")),
        "q_mode": row.get("q_mode"),
    }


def _source_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_contract": copy.deepcopy(row.get("source_contract")),
        "source_evidence": copy.deepcopy(row.get("source_evidence")),
        "source_evidence_sha256": row.get("source_evidence_sha256"),
    }


def _task_contract(
    *, arm: str, profile: Mapping[str, Any]
) -> dict[str, Any]:
    contract = ARM_CONTRACTS[arm]
    return {
        "schema_version": TASK_SCHEMA,
        "task_id": contract["task_id"],
        "arm_id": arm,
        "phase": contract["phase"],
        "model": "fcooper",
        "hardware_id": "h800",
        "backend": "tvm",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": profile["capability_profile_id"],
        "capability_digest": profile["capability_digest"],
        "tvm_trials": contract["tvm_trials"],
        "runner_request_kind": "stage6-tvm-control",
        "recovery_contract_required_for_pruned_rows": True,
    }


def _rebind_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    profile: Mapping[str, Any],
    task_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    contract = ARM_CONTRACTS[arm]
    task_sha = canonical_sha256(task_contract)
    rebound: list[dict[str, Any]] = []
    mappings: list[dict[str, str]] = []
    for source in rows:
        source_copy = copy.deepcopy(dict(source))
        source_row_id = str(
            source_copy.get("row_id") or source_copy.get("manifest_job_id")
        )
        base = {
            key: value
            for key, value in source_copy.items()
            if key not in DERIVED_ROW_FIELDS
        }
        identity = {
            "schema_version": "stage6_fcooper_tvm_row_identity_v1",
            "task_id": contract["task_id"],
            "task_sha256": task_sha,
            "arm_id": arm,
            "phase": contract["phase"],
            "model": base.get("model"),
            "hardware_id": base.get("hardware_id"),
            "genome": _genome(base),
            "source_binding": _source_binding(base),
            "capability_profile_id": profile["capability_profile_id"],
            "capability_digest": profile["capability_digest"],
            "dispatch_key": "tvm_auto",
            "tvm_trials": contract["tvm_trials"],
        }
        row_id = f"{contract['task_id']}-ROW-{canonical_sha256(identity)[:24]}"
        row = {
            **base,
            "row_id": row_id,
            "manifest_job_id": row_id,
            "task_id": contract["task_id"],
            "task_sha256": task_sha,
            "arm_id": arm,
            "phase": contract["phase"],
            "backend": "tvm",
            "dispatch_key": "tvm_auto",
            "capability_profile_id": profile["capability_profile_id"],
            "capability_digest": profile["capability_digest"],
            "tvm_trials": contract["tvm_trials"],
        }
        rebound.append(row)
        mappings.append(
            {
                "source_row_id": source_row_id,
                "rebound_row_id": row_id,
                "rebound_row_sha256": canonical_sha256(row),
            }
        )
    if len({row["row_id"] for row in rebound}) != len(rebound):
        raise ValueError(f"{arm} rebound row identities are not unique")
    return rebound, mappings


def _write_requests(
    rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    profile: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    contract = ARM_CONTRACTS[arm]
    task_contract = _task_contract(arm=arm, profile=profile)
    task_sha = canonical_sha256(task_contract)
    rebound, mappings = _rebind_rows(
        rows, arm=arm, profile=profile, task_contract=task_contract
    )
    entries = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for request_index, offset in enumerate(range(0, len(rebound), 4)):
        batch = rebound[offset : offset + 4]
        request = {
            "schema_version": REQUEST_SCHEMA,
            "task_id": contract["task_id"],
            "task_sha256": task_sha,
            "task_contract": task_contract,
            "arm_id": arm,
            "phase": contract["phase"],
            "request_index": request_index,
            "batch_size": len(batch),
            "atomic_feedback": True,
            "real_h800_measurement_required": True,
            "backend": "tvm",
            "dispatch_key": "tvm_auto",
            "capability_profile_id": profile["capability_profile_id"],
            "capability_digest": profile["capability_digest"],
            "tvm_trials": contract["tvm_trials"],
            "row_sha256": {
                str(row["row_id"]): canonical_sha256(row) for row in batch
            },
            "rows": batch,
        }
        request["measurement_request_sha256"] = canonical_sha256(request)
        path = (
            output_dir
            / f"request_{request_index:02d}_"
            f"{request['measurement_request_sha256'][:16]}.json"
        )
        _write_json_immutable(path, request)
        entries.append(
            {
                "request_index": request_index,
                "path": str(path.resolve()),
                "measurement_request_sha256": request[
                    "measurement_request_sha256"
                ],
                "batch_size": len(batch),
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "task_id": contract["task_id"],
        "arm_id": arm,
        "phase": contract["phase"],
        "backend": "tvm",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": profile["capability_profile_id"],
        "capability_digest": profile["capability_digest"],
        "tvm_trials": contract["tvm_trials"],
        "request_count": len(entries),
        "row_count": len(rebound),
        "requests": entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write_json_immutable(output_dir / "request_manifest.json", manifest)
    return manifest, mappings


def rebind_stage6_requests(
    *,
    compression_manifest: Path,
    compress_tune_manifest: Path,
    schedule_manifest: Path,
    capability_profile: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    profile_copy = copy.deepcopy(dict(capability_profile))
    if profile_copy.get("dispatch_key") != "tvm_auto":
        raise ValueError("capability profile dispatch_key must be tvm_auto")
    profile = validate_capability_profile(profile_copy)
    if str(profile["hardware_target"]).lower() != "h800":
        raise ValueError("capability profile hardware_target must be h800")

    source_paths = {
        "compression_only": Path(compression_manifest),
        "compress_then_tune": Path(compress_tune_manifest),
        "schedule_only": Path(schedule_manifest),
    }
    source_rows: dict[str, list[dict[str, Any]]] = {}
    source_provenance: dict[str, dict[str, Any]] = {}
    for arm, path in source_paths.items():
        contract = ARM_CONTRACTS[arm]
        rows, provenance = _load_frozen_requests(
            path,
            expected_arm=arm,
            expected_phase=str(contract["phase"]),
            expected_count=int(contract["expected_count"]),
        )
        source_rows[arm] = rows
        source_provenance[arm] = provenance

    manifests: dict[str, dict[str, Any]] = {}
    mappings: dict[str, list[dict[str, str]]] = {}
    for arm, rows in source_rows.items():
        manifest, row_mappings = _write_requests(
            rows,
            arm=arm,
            profile=profile,
            output_dir=Path(output_dir) / arm,
        )
        manifests[arm] = manifest
        mappings[arm] = row_mappings

    output_rows = {
        arm: [
            row
            for entry in manifest["requests"]
            for row in _read_json(Path(entry["path"]))["rows"]
        ]
        for arm, manifest in manifests.items()
    }
    genome_equal = all(
        [_genome(row) for row in source_rows[arm]]
        == [_genome(row) for row in output_rows[arm]]
        for arm in source_rows
    )
    source_contract_equal = all(
        [copy.deepcopy(row.get("source_contract")) for row in source_rows[arm]]
        == [copy.deepcopy(row.get("source_contract")) for row in output_rows[arm]]
        for arm in source_rows
    )
    source_evidence_equal = all(
        [_source_binding(row) for row in source_rows[arm]]
        == [_source_binding(row) for row in output_rows[arm]]
        for arm in source_rows
    )
    forbidden = sorted(
        {
            field
            for rows in output_rows.values()
            for field in _artifact_fields(rows)
        }
    )
    if not genome_equal or not source_contract_equal or not source_evidence_equal:
        raise RuntimeError("immutable candidate/source rebind audit failed")
    if forbidden:
        raise RuntimeError(
            "rebound rows contain TRT performance artifact fields: "
            + ", ".join(forbidden)
        )

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "backend": "tvm",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": profile["capability_profile_id"],
        "capability_digest": profile["capability_digest"],
        "candidate_counts": {
            arm: len(rows) for arm, rows in source_rows.items()
        },
        "tvm_trials": {
            arm: int(contract["tvm_trials"])
            for arm, contract in ARM_CONTRACTS.items()
        },
        "source_provenance": source_provenance,
        "row_identity_mappings": mappings,
        "genome_sequence_equal": genome_equal,
        "source_contract_sequence_equal": source_contract_equal,
        "source_evidence_sequence_equal": source_evidence_equal,
        "trt_performance_artifact_fields": forbidden,
        "old_measurement_results_read": False,
        "input_manifests_modified": False,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    audit_path = Path(output_dir) / "provenance_audit.json"
    _write_json_immutable(audit_path, audit)
    return {
        "manifests": manifests,
        "audit": audit,
        "audit_path": str(audit_path.resolve()),
    }


def _profile_from_payload(payload: Any, profile_id: str | None) -> dict[str, Any]:
    if isinstance(payload, Mapping) and "capability_profiles" in payload:
        profiles = payload["capability_profiles"]
    elif isinstance(payload, list):
        profiles = payload
    elif isinstance(payload, Mapping):
        profiles = [payload]
    else:
        raise ValueError("capability profile JSON has an unsupported shape")
    matches = [
        dict(profile)
        for profile in profiles
        if profile.get("dispatch_key") == "tvm_auto"
        and (
            profile_id is None
            or profile.get("capability_profile_id") == profile_id
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one matching tvm_auto capability profile, got {len(matches)}"
        )
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compression-manifest", type=Path, required=True)
    parser.add_argument("--compress-tune-screen-manifest", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--capability-profile-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = _profile_from_payload(
        _read_json(args.capability_profiles_json), args.capability_profile_id
    )
    result = rebind_stage6_requests(
        compression_manifest=args.compression_manifest,
        compress_tune_manifest=args.compress_tune_screen_manifest,
        schedule_manifest=args.schedule_manifest,
        capability_profile=profile,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": "success",
                "audit_path": result["audit_path"],
                "candidate_counts": result["audit"]["candidate_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
