"""Build Stage5 performance jobs from verified source materialization evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stage35_gold32_performance_plan_v1 import _build_job


SOURCE_EVIDENCE_SCHEMA = "stage5_source_materialization_evidence_v1"
REQUEST_SCHEMA = "stage5_measurement_request_v1"
MANIFEST_SCHEMA = "stage5_performance_manifest_v1"
JOB_SCHEMA = "stage5_performance_job_v1"
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
ARTIFACT_FIELDS = (
    ("checkpoint_path", "checkpoint_sha256"),
    ("onnx_path", "onnx_sha256"),
    ("calibration_path", "calibration_sha256"),
    ("calibration_summary_path", "calibration_summary_sha256"),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest(path: Path) -> tuple[int, str]:
    if not path.is_dir():
        raise ValueError(f"TRT calibration directory missing: {path}")
    entries = [
        {"name": item.name, "sha256": _file_sha256(item)}
        for item in sorted(path.glob("*.npy"))
        if item.is_file()
    ]
    if not entries:
        raise ValueError(f"TRT calibration directory has no NPY samples: {path}")
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return len(entries), digest


def _source_plan_sha(row: Mapping[str, Any]) -> str:
    model = str(row.get("model") or "")
    allowed_kinds = {
        "pyramid": {"pyramid_checkpoint_export", "pyramid_prepare_train_export"},
        "codriving": {"codriving_prepare_train_export"},
        "fcooper": {"fcooper_scanner_materialize_export"},
    }
    if model not in allowed_kinds:
        raise ValueError(f"unsupported source model: {model or '<empty>'}")
    kind = str(row.get("materialization_kind") or "")
    if not kind:
        default_kinds = {
            "pyramid": "pyramid_checkpoint_export",
            "codriving": "codriving_prepare_train_export",
            "fcooper": "fcooper_scanner_materialize_export",
        }
        kind = default_kinds[model]
    if kind not in allowed_kinds[model]:
        raise ValueError(f"unsupported materialization kind for {model}: {kind}")
    payload = {
        "kind": kind,
        "width": [int(value) for value in row.get("width") or []],
        "contract": row.get("source_contract") or {},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_source_evidence(
    path: Path, expected_group_id: str, expected_plan_sha: str
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"source evidence missing for {expected_group_id}: {path}")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != SOURCE_EVIDENCE_SCHEMA:
        raise ValueError(f"unexpected source evidence schema for {expected_group_id}")
    if evidence.get("group_id") != expected_group_id:
        raise ValueError(f"source evidence group mismatch for {expected_group_id}")
    if evidence.get("source_plan_sha256") != expected_plan_sha:
        raise ValueError(f"source plan SHA mismatch for {expected_group_id}")
    if evidence.get("status") != "ready":
        raise ValueError(f"source evidence is not ready for {expected_group_id}")
    for path_key, sha_key in ARTIFACT_FIELDS:
        artifact = Path(str(evidence.get(path_key) or ""))
        expected_sha = str(evidence.get(sha_key) or "")
        if not artifact.is_file():
            raise ValueError(f"source artifact missing for {expected_group_id}: {artifact}")
        actual_sha = _file_sha256(artifact)
        if len(expected_sha) != 64 or actual_sha != expected_sha:
            raise ValueError(
                f"SHA mismatch for {expected_group_id} {path_key}: "
                f"expected={expected_sha}, actual={actual_sha}"
            )
    return evidence


def _validate_request(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(f"expected {REQUEST_SCHEMA}")
    rows = request.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("measurement request rows must be a non-empty list")
    copied = [copy.deepcopy(dict(row)) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in copied:
        groups.setdefault(str(row.get("group_id") or ""), []).append(row)
    if any(not group_id for group_id in groups):
        raise ValueError("measurement request has an empty group_id")
    for group_id, group_rows in groups.items():
        arms = {
            (str(row.get("dispatch_key") or ""), str(row.get("q_mode") or ""))
            for row in group_rows
        }
        if len(group_rows) != 4 or arms != EXPECTED_ARMS:
            raise ValueError(f"incomplete four-arm measurement group: {group_id}")
        identities = {
            json.dumps(
                {
                    "group_id": row.get("group_id"),
                    "model": row.get("model"),
                    "width": row.get("width"),
                    "source_evidence_sha256": row.get("source_evidence_sha256"),
                    "source_contract": row.get("source_contract"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in group_rows
        }
        if len(identities) != 1:
            raise ValueError(f"four-arm group identity drift: {group_id}")
        model = str(group_rows[0].get("model") or "")
        width = [int(value) for value in group_rows[0].get("width") or []]
        if len(width) != 3 or group_id != f"{model}|{'x'.join(map(str, width))}":
            raise ValueError(f"four-arm group identity drift: {group_id}")
        expected_plan = str(group_rows[0].get("source_evidence_sha256") or "")
        if len(expected_plan) != 64 or _source_plan_sha(group_rows[0]) != expected_plan:
            raise ValueError(f"source plan SHA mismatch for {group_id}")
    profile_dispatch: dict[str, str] = {}
    dispatch_profile: dict[str, str] = {}
    for row in copied:
        profile = str(row.get("capability_profile_id") or "")
        dispatch = str(row.get("dispatch_key") or "")
        if not profile or (
            profile in profile_dispatch and profile_dispatch[profile] != dispatch
        ) or (
            dispatch in dispatch_profile and dispatch_profile[dispatch] != profile
        ):
            raise ValueError("capability profile/dispatch identity drift")
        profile_dispatch[profile] = dispatch
        dispatch_profile[dispatch] = profile
    if int(request.get("group_count", -1)) != len(groups):
        raise ValueError("measurement request group_count mismatch")
    if int(request.get("row_count", -1)) != len(copied):
        raise ValueError("measurement request row_count mismatch")
    return copied


def _bind_verified_source(
    row: Mapping[str, Any], evidence: Mapping[str, Any], evidence_path: Path
) -> dict[str, Any]:
    source_contract = copy.deepcopy(dict(row.get("source_contract") or {}))
    trt_calibration_dir = Path(str(source_contract.get("trt_calibration_dir") or ""))
    trt_sample_count, trt_manifest_sha = _directory_manifest(trt_calibration_dir)
    source_contract.update(
        {
            "checkpoint_path": str(evidence["checkpoint_path"]),
            "checkpoint_sha256": str(evidence["checkpoint_sha256"]),
            "onnx_path": str(evidence["onnx_path"]),
            "onnx_sha256": str(evidence["onnx_sha256"]),
            "calibration_npz": str(evidence["calibration_path"]),
            "calibration_npz_sha256": str(evidence["calibration_sha256"]),
            "calibration_summary": str(evidence["calibration_summary_path"]),
            "calibration_summary_sha256": str(
                evidence["calibration_summary_sha256"]
            ),
            "trt_calibration_sample_count": trt_sample_count,
            "trt_calibration_manifest_sha256": trt_manifest_sha,
        }
    )
    row_id = str(row.get("manifest_job_id") or row.get("row_id") or "")
    if not row_id:
        raise ValueError(f"measurement row identity missing: {row.get('group_id')}")
    return {
        **copy.deepcopy(dict(row)),
        "schema_version": "stage5_performance_manifest_row_v1",
        "job_id": row_id,
        "manifest_job_id": row_id,
        "split": "online_feedback",
        "source_pool": "stage5_online_feedback",
        "required_metrics": ["latency", "energy", "ap"],
        "source_status": "ready",
        "source_evidence_path": str(evidence_path),
        "source_evidence_sha256": _file_sha256(evidence_path),
        "source_contract": source_contract,
        "terminal_status": "pending",
    }


def build_performance_plan(
    request: Mapping[str, Any],
    *,
    source_evidence_paths: Mapping[str, Path],
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    """Validate sources and emit one automatic performance job per requested arm."""
    rows = _validate_request(request)
    if not gpus:
        raise ValueError("gpus must not be empty")
    group_ids = sorted({str(row["group_id"]) for row in rows})
    if set(source_evidence_paths) != set(group_ids):
        raise ValueError("source evidence paths must exactly match requested groups")

    evidence_by_group = {
        group_id: _load_source_evidence(
            Path(source_evidence_paths[group_id]),
            group_id,
            str(next(row for row in rows if row["group_id"] == group_id)["source_evidence_sha256"]),
        )
        for group_id in group_ids
    }
    bound_rows = [
        _bind_verified_source(
            row,
            evidence_by_group[str(row["group_id"])],
            Path(source_evidence_paths[str(row["group_id"])]),
        )
        for row in rows
    ]
    bound_rows.sort(
        key=lambda row: (
            str(row["group_id"]),
            str(row["dispatch_key"]),
            str(row["q_mode"]),
        )
    )
    jobs = []
    for index, row in enumerate(bound_rows):
        job = _build_job(
            row,
            batch_index=1,
            row_index=index,
            remote_artifact_root=remote_artifact_root,
            gpus=gpus,
        )
        jobs.append({**job, "schema_version": JOB_SCHEMA})

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_request_schema": REQUEST_SCHEMA,
        "source_request_sha256": hashlib.sha256(
            json.dumps(
                request, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "source_pool": "online_feedback",
        "group_count": len(group_ids),
        "row_count": len(bound_rows),
        "group_ids": group_ids,
        "source_evidence": {
            group_id: {
                "path": str(source_evidence_paths[group_id]),
                "sha256": _file_sha256(Path(source_evidence_paths[group_id])),
            }
            for group_id in group_ids
        },
        "jobs": bound_rows,
    }
    return {"manifest": manifest, "performance_jobs": jobs}
