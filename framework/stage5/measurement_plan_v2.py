"""Verified performance jobs for a Stage5 v2 single-task atomic batch."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from framework.stage5.measurement_plan_v1 import (
    JOB_SCHEMA,
    _bind_verified_source,
    _build_job,
    _load_source_evidence,
    _source_plan_sha,
)
from framework.stage5.genome_contract_v1 import validate_structure_identity


REQUEST_SCHEMA = "stage5_measurement_request_v2"
INDEPENDENT_REQUEST_SCHEMA = "stage5_independent_validation_request_v1"
MANIFEST_SCHEMA = "stage5_performance_manifest_v2"


def _validate_request(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    schema = str(request.get("schema_version") or "")
    if schema not in {REQUEST_SCHEMA, INDEPENDENT_REQUEST_SCHEMA}:
        raise ValueError(f"expected {REQUEST_SCHEMA} or {INDEPENDENT_REQUEST_SCHEMA}")
    if request.get("real_h800_measurement_required") is not True or request.get(
        "required_metrics"
    ) != ["latency_ms", "energy_j", "ap30", "ap50", "ap70"]:
        raise ValueError("Stage5 H800 metric contract is incomplete")
    if schema == REQUEST_SCHEMA and (
        request.get("atomic_feedback") is not True
        or int(request.get("sample_budget", -1)) != 16
    ):
        raise ValueError("online atomic-feedback contract is incomplete")
    if schema == INDEPENDENT_REQUEST_SCHEMA and request.get(
        "independent_from_search_measurement"
    ) is not True:
        raise ValueError("independent-validation contract is incomplete")
    rows = request.get("rows")
    expected_count = 4 if schema == REQUEST_SCHEMA else int(request.get("batch_size", -1))
    if (
        not isinstance(rows, list)
        or expected_count not in {1, 2, 3, 4}
        or len(rows) != expected_count
        or int(request.get("batch_size", -1)) != expected_count
    ):
        raise ValueError("Stage5 request must contain one verified batch of one to four genomes")
    copied = [copy.deepcopy(dict(row)) for row in rows]
    request_payload = {
        key: copy.deepcopy(value)
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    expected_request_sha = hashlib.sha256(
        json.dumps(
            request_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if request.get("measurement_request_sha256") != expected_request_sha:
        raise ValueError("measurement request SHA mismatch")
    row_ids = [str(row.get("manifest_job_id") or row.get("row_id") or "") for row in copied]
    if any(not row_id for row_id in row_ids) or len(set(row_ids)) != expected_count:
        raise ValueError("v2 measurement request has empty or duplicate genome identity")
    row_sha = request.get("row_sha256")
    if not isinstance(row_sha, Mapping) or set(row_sha) != set(row_ids):
        raise ValueError("measurement request row SHA map mismatch")
    for row_id, row in zip(row_ids, copied):
        actual = hashlib.sha256(
            json.dumps(
                row, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if row_sha.get(row_id) != actual:
            raise ValueError(f"measurement request row SHA mismatch: {row_id}")
    task_ids = {str(row.get("task_id") or "") for row in copied}
    task_shas = {str(row.get("task_sha256") or "") for row in copied}
    models = {str(row.get("model") or "") for row in copied}
    profiles = {str(row.get("capability_profile_id") or "") for row in copied}
    dispatches = {str(row.get("dispatch_key") or "") for row in copied}
    hardware_ids = {str(row.get("hardware_id") or "") for row in copied}
    capability_digests = {str(row.get("capability_digest") or "") for row in copied}
    if (
        task_ids != {str(request.get("task_id") or "")}
        or task_shas != {str(request.get("task_sha256") or "")}
        or len(models) != 1
        or len(profiles) != 1
        or len(dispatches) != 1
        or hardware_ids != {"h800"}
        or len(capability_digests) != 1
        or len(next(iter(capability_digests), "")) != 64
    ):
        raise ValueError("measurement rows drift from fixed task/profile")
    for row in copied:
        identity = validate_structure_identity(row)
        width = list(identity.width)
        group_id = identity.group_id
        q_mode = str(row.get("q_mode") or "")
        if row.get("genome") != [*width, q_mode] or q_mode not in {"fp16", "int8"}:
            raise ValueError(f"genome payload drift: {row_ids[copied.index(row)]}")
        plan_sha = str(row.get("source_evidence_sha256") or "")
        if len(plan_sha) != 64 or _source_plan_sha(row) != plan_sha:
            raise ValueError(f"source plan SHA mismatch for {group_id}")
    return copied


def build_performance_plan(
    request: Mapping[str, Any],
    *,
    source_evidence_paths: Mapping[str, Path],
    quant_contract_paths: Mapping[str, Path],
    remote_artifact_root: str | Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    rows = _validate_request(request)
    if not gpus:
        raise ValueError("gpus must not be empty")
    group_ids = sorted({str(row["group_id"]) for row in rows})
    if set(source_evidence_paths) != set(group_ids):
        raise ValueError("source evidence paths must exactly match requested source groups")
    evidence_by_group = {
        group_id: _load_source_evidence(
            Path(source_evidence_paths[group_id]),
            group_id,
            str(next(row for row in rows if row["group_id"] == group_id)["source_evidence_sha256"]),
        )
        for group_id in group_ids
    }
    bound_rows = [
        {
            **_bind_verified_source(
                row,
                evidence_by_group[str(row["group_id"])],
                Path(source_evidence_paths[str(row["group_id"])]),
            ),
            "source_plan_sha256": str(row["source_evidence_sha256"]),
        }
        for row in rows
    ]
    required_quant_ids = {
        str(row["manifest_job_id"])
        for row in bound_rows
        if row["dispatch_key"] == "tvm_auto" and row["q_mode"] == "int8"
    }
    if set(quant_contract_paths) != required_quant_ids:
        raise ValueError("quant contract paths must exactly match TVM INT8 genomes")
    prepared_rows = []
    jobs = []
    for index, source in enumerate(bound_rows):
        row = copy.deepcopy(source)
        row_id = str(row["manifest_job_id"])
        quant_path = quant_contract_paths.get(row_id)
        if quant_path is not None:
            path = Path(quant_path)
            if not path.is_file():
                raise ValueError(f"quant contract missing: {row_id}")
            contract = json.loads(path.read_text(encoding="utf-8"))
            if (
                contract.get("schema") != "stage3_tvm_int8_quant_contract_v3"
                or not isinstance(contract.get("params"), Mapping)
                or not contract["params"]
            ):
                raise ValueError(f"invalid quant contract: {row_id}")
            row["source_contract"] = {
                **row["source_contract"],
                "tensor_quant_params_json": str(path),
                "tensor_quant_params_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        job = _build_job(
            row,
            batch_index=1,
            row_index=index,
            remote_artifact_root=remote_artifact_root,
            gpus=gpus,
        )
        if quant_path is not None:
            command = list(job["command"])
            if "--tensor-quant-params-json" in command:
                raise ValueError(f"duplicate quant contract argument: {row_id}")
            job = {
                **job,
                "command": [*command, "--tensor-quant-params-json", str(quant_path)],
            }
        prepared_rows.append(row)
        jobs.append({**job, "schema_version": JOB_SCHEMA})
    request_sha = str(request["measurement_request_sha256"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_request_schema": request["schema_version"],
        "source_request_sha256": request_sha,
        "task_id": request["task_id"],
        "task_sha256": request["task_sha256"],
        "source_pool": "stage5_online_feedback",
        "genome_count": len(bound_rows),
        "row_count": len(bound_rows),
        "group_count": len(group_ids),
        "group_ids": group_ids,
        "jobs": prepared_rows,
    }
    return {"manifest": manifest, "performance_jobs": jobs}
