"""Concrete dependency-injected Stage7 actual-mode physical pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage7.physical_execution_v2 import (
    build_authenticated_performance_artifacts,
)
from framework.stage7.physical_feedback_v2 import finalize_physical_rows
from framework.stage7.physical_runtime_v2 import (
    build_uuid_assignments,
    execute_bound_ap_row,
    execute_bound_performance_row,
    execute_bound_quant_row,
    verify_runtime_primitives,
)


JSON = dict[str, Any]
STAGES = ("quant", "performance", "sanity", "full", "final")
EXPECTED_HOST = os.environ.get(
    "V2X_FORMAL_H800_HOSTNAME", "zs-nj-tap-gpu18"
).strip()
ADMISSION_SCHEMA = "stage7_actual_runtime_admission_v2"
OUTPUT_SCHEMA = "stage7_actual_pipeline_stage_output_v2"
_ATTEMPT_NAME = re.compile(r"attempt_[0-9]{3,}")
_NO_GPU_TRUTH = {
    "formal_v2_gpu_jobs_launched": 0,
    "cuda_visible_devices": "",
    "synthetic_non_measurement": True,
    "actual_v3_hardware_evidence": False,
    "eligible_for_cache_append": False,
    "eligible_for_formal_finalization": False,
}
_ACTUAL_TRUTH = {
    "synthetic_non_measurement": False,
    "actual_v3_hardware_evidence": True,
    "eligible_for_cache_append": False,
    "eligible_for_formal_finalization": True,
}


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    return copy.deepcopy(value)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"path contains a symlink: {candidate}")


def _plain_directory(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    _reject_symlink_components(absolute)
    if not absolute.is_dir() or absolute.is_symlink():
        raise ValueError(f"{label} must be a plain directory")
    if absolute.resolve() != absolute:
        raise ValueError(f"{label} path resolution drift")
    return absolute


def _plain_file(path, *, label, allowed_roots=()):
    absolute = path.absolute()
    _reject_symlink_components(absolute)
    if not absolute.is_file() or absolute.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = absolute.resolve()
    if resolved != absolute:
        raise ValueError(f"{label} path resolution drift")
    details = absolute.stat()
    if details.st_nlink != 1:
        raise ValueError(f"{label} hardlink count is not one")
    if allowed_roots and not any(_inside(resolved, root) for root in allowed_roots):
        raise ValueError(f"{label} escapes authenticated roots")
    return absolute


def _safe_output_path(path: Path, *, attempt: Path, label: str) -> Path:
    absolute = path.absolute()
    if not _inside(absolute, attempt):
        raise ValueError(f"{label} output path escapes the attempt")
    _reject_symlink_components(absolute)
    if absolute.exists():
        _plain_file(absolute, label=label, allowed_roots=(attempt,))
    return absolute


def _write_stage_output(path: Path, payload: Mapping[str, Any], attempt: Path) -> JSON:
    unsigned = {
        "schema_version": OUTPUT_SCHEMA,
        **_jsonable(dict(payload)),
    }
    output = {**unsigned, "pipeline_output_sha256": _sha(unsigned)}
    content = (
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    destination = _safe_output_path(path, attempt=attempt, label="stage artifact")
    if destination.is_file():
        if destination.read_bytes() != content:
            raise ValueError("immutable stage artifact drift")
        return output
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    _plain_file(destination, label="stage artifact", allowed_roots=(attempt,))
    return output


def _read_stage_output(path: Path, *, attempt: Path, expected_stage: str) -> JSON:
    source = _plain_file(path, label="stage artifact", allowed_roots=(attempt,))
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("stage artifact is invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError("stage artifact must be a mapping")
    copied = dict(value)
    recorded = copied.pop("pipeline_output_sha256", None)
    if (
        copied.get("schema_version") != OUTPUT_SCHEMA
        or copied.get("stage") != expected_stage
        or recorded != _sha(copied)
    ):
        raise ValueError("stage artifact authentication failed")
    return {**copied, "pipeline_output_sha256": recorded}


def _stage_output_path(attempt: Path, stage: str) -> Path:
    if stage == "sanity":
        return attempt / "ap" / "sanity_pipeline_output.json"
    if stage == "full":
        return attempt / "ap" / "full_pipeline_output.json"
    return attempt / stage / "pipeline_output.json"


def _validated_attempt(path: Path) -> Path:
    attempt = path.absolute()
    _reject_symlink_components(attempt)
    if not _ATTEMPT_NAME.fullmatch(attempt.name):
        raise ValueError("stage attempt path has an invalid name")
    attempt.mkdir(parents=True, exist_ok=True)
    attempt = _plain_directory(attempt, label="stage attempt")
    if any(child.name == "__pycache__" for child in attempt.rglob("__pycache__")):
        raise ValueError("stage attempt contains a forbidden __pycache__ directory")
    return attempt


def _validate_no_gpu_receipt(
    receipt, *, formal_v2_root, frozen_repo_root, deployment_bundle_sha256
):
    if not isinstance(receipt, Mapping):
        raise ValueError("authenticated no-GPU receipt is required")
    copied = _jsonable(dict(receipt))
    recorded = copied.pop("dry_run_receipt_sha256", None)
    truth = {key: copied.get(key) for key in _NO_GPU_TRUTH}
    if (
        copied.get("schema_version") != "stage7_core_no_gpu_dry_run_v2"
        or copied.get("formal_v2_root") != str(formal_v2_root)
        or copied.get("frozen_repo_root") != str(frozen_repo_root)
        or copied.get("deployment_bundle_sha256") != deployment_bundle_sha256
        or truth != _NO_GPU_TRUTH
        or recorded != _sha(copied)
    ):
        raise ValueError("authenticated no-GPU truth drift")
    return {**copied, "dry_run_receipt_sha256": recorded}


def validate_actual_runtime_admission(
    *,
    host_name: str,
    formal_v2_root: Path,
    frozen_repo_root: Path,
    deployment_bundle_sha256: str,
    no_gpu_receipt: Mapping[str, Any],
    physical_bindings: Sequence[Mapping[str, Any]],
    ordered_lease_uuids: Sequence[str],
    parent_cuda_visible_devices: str,
    expected_lock_owner: str,
    inventory_probe: Callable[[], Sequence[Mapping[str, Any]]],
    lock_probe: Callable[[str], str | None],
    process_probe: Callable[[], Sequence[Mapping[str, Any]]],
    primitive_sha_probe: Callable[[str], str],
    uuid_assignment_builder: Callable[..., Mapping[str, Any]] = (
        build_uuid_assignments
    ),
) -> JSON:
    if not host_name or (EXPECTED_HOST and host_name != EXPECTED_HOST):
        raise ValueError("actual runtime host does not match V2X_FORMAL_H800_HOSTNAME")
    if len(deployment_bundle_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in deployment_bundle_sha256
    ):
        raise ValueError("deployment bundle SHA256 is invalid")
    formal_root = _plain_directory(Path(formal_v2_root), label="formal v2 root")
    frozen_root = _plain_directory(Path(frozen_repo_root), label="frozen repo root")
    receipt = _validate_no_gpu_receipt(
        no_gpu_receipt,
        formal_v2_root=formal_root,
        frozen_repo_root=frozen_root,
        deployment_bundle_sha256=deployment_bundle_sha256,
    )
    leases = tuple(str(value) for value in ordered_lease_uuids)
    if not parent_cuda_visible_devices:
        raise ValueError("actual CUDA mask must be nonempty")
    if (
        len(leases) != 4
        or len(set(leases)) != 4
        or parent_cuda_visible_devices != ",".join(leases)
    ):
        raise ValueError("actual CUDA mask must exactly match four ordered UUID leases")
    pins = verify_runtime_primitives(primitive_sha_probe)
    inventory = [_jsonable(row) for row in inventory_probe()]
    lock_owners = {uuid: lock_probe(uuid) for uuid in leases}
    related_processes = [_jsonable(row) for row in process_probe()]
    assignments = dict(
        uuid_assignment_builder(
            physical_bindings=physical_bindings,
            ordered_lease_uuids=leases,
            parent_cuda_visible_devices=parent_cuda_visible_devices,
            inventory=inventory,
            lock_owners=lock_owners,
            expected_lock_owner=expected_lock_owner,
            related_processes=related_processes,
        )
    )
    assignment_rows = assignments.get("assignments")
    if (
        not isinstance(assignment_rows, list)
        or len(assignment_rows) not in range(1, 5)
        or len(assignment_rows) != len(physical_bindings)
    ):
        raise ValueError("actual admission requires one through four physical rows")
    unsigned = {
        "schema_version": ADMISSION_SCHEMA,
        "host_name": host_name,
        "formal_v2_root": str(formal_root),
        "frozen_repo_root": str(frozen_root),
        "deployment_bundle_sha256": deployment_bundle_sha256,
        "no_gpu_receipt_sha256": receipt["dry_run_receipt_sha256"],
        "no_gpu_receipt": receipt,
        "no_gpu_truth": {key: receipt[key] for key in _NO_GPU_TRUTH},
        "actual_execution_truth": dict(_ACTUAL_TRUTH),
        "actual_cuda_visible_devices": parent_cuda_visible_devices,
        "expected_lock_owner": expected_lock_owner,
        "physical_row_count": len(assignment_rows),
        "primitive_sha256": pins,
        "uuid_assignments": assignments,
    }
    return {**unsigned, "actual_runtime_admission_sha256": _sha(unsigned)}


def _rows(projection: Mapping[str, Any]) -> list[JSON]:
    raw = projection.get("rows")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("independent projection rows are invalid")
    rows = [_jsonable(dict(row)) for row in raw if isinstance(row, Mapping)]
    ids = [str(row.get("candidate_id") or row.get("row_id") or "") for row in rows]
    if (
        len(rows) != len(raw)
        or len(rows) not in range(5)
        or any(not item for item in ids)
    ):
        raise ValueError("actual pipeline requires zero through four identified rows")
    if len(set(ids)) != len(ids):
        raise ValueError("actual pipeline candidate identities are duplicated")
    if any(
        Path(item).name != item or item in {".", "..", "__pycache__"} or "\\" in item
        for item in ids
    ):
        raise ValueError("actual pipeline candidate identity is not path-safe")
    return rows


def _assignments(admission: Mapping[str, Any]) -> dict[str, JSON]:
    batch = admission["uuid_assignments"]
    return {str(row["candidate_id"]): _jsonable(row) for row in batch["assignments"]}


def _validate_source_inputs(source_result, *, allowed_roots):
    rows = source_result.get("rows") or ()
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("source result rows are invalid")
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("source_evidence_path"):
            continue
        path = _plain_file(
            Path(str(row["source_evidence_path"])),
            label="source evidence",
            allowed_roots=allowed_roots,
        )
        expected = str(row.get("source_evidence_file_sha256") or "")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("source evidence SHA drift")


def _quant_contract(path: Path, *, attempt: Path) -> None:
    source = _plain_file(path, label="quant contract", allowed_roots=(attempt,))
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("quant contract is invalid") from error
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "stage3_tvm_int8_quant_contract_v3"
        or not isinstance(value.get("params"), Mapping)
        or not value["params"]
    ):
        raise ValueError("quant contract is invalid")


def _validate_quant_plan(plan, *, output_path, attempt, allowed_roots):
    if not isinstance(plan, Mapping):
        raise ValueError("quant plan must be a mapping")
    arguments = plan.get("arguments")
    if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
        raise ValueError("quant arguments must be an argv array")
    argv = [str(value) for value in arguments]
    if argv.count("--output-json") != 1:
        raise ValueError("quant output path is missing")
    index = argv.index("--output-json")
    if index + 1 >= len(argv) or Path(argv[index + 1]).absolute() != output_path:
        raise ValueError("quant output path escapes isolated row artifacts")
    for raw in plan.get("input_paths") or ():
        _plain_file(Path(str(raw)), label="quant input", allowed_roots=allowed_roots)
    _safe_output_path(output_path, attempt=attempt, label="quant contract")
    return argv


def _validate_ap_plan(value, *, candidate_ids, attempt):
    if not isinstance(value, Mapping):
        raise ValueError("AP plan builder returned an invalid result")
    raw_rows = value.get("rows")
    shards = value.get("shards")
    if not isinstance(raw_rows, Sequence) or not isinstance(shards, Mapping):
        raise ValueError("AP plan rows or shards are missing")
    rows = [_jsonable(dict(row)) for row in raw_rows if isinstance(row, Mapping)]
    observed = [str(row.get("manifest_job_id") or "") for row in rows]
    if observed != list(candidate_ids) or set(shards) != set(candidate_ids):
        raise ValueError("AP plan candidate order drift")
    normalized: dict[str, str] = {}
    for candidate in candidate_ids:
        path = _plain_file(
            Path(str(shards[candidate])),
            label="AP plan shard",
            allowed_roots=(attempt,),
        )
        normalized[candidate] = str(path)
    return rows, normalized


def build_concrete_stage_runner(
    *,
    projection: Mapping[str, Any],
    logical_request: Mapping[str, Any],
    selection_binding: Mapping[str, Any],
    physical_plan: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    source_result: Mapping[str, Any],
    runtime_admission: Mapping[str, Any],
    python_executable: Path,
    remote_artifact_root: str | Path,
    quant_plan_builder: Callable[..., Mapping[str, Any]],
    ap_plan_builder: Callable[..., Mapping[str, Any]],
    finalizer_evidence_builder: Callable[..., Mapping[str, Any]],
    inventory_probe: Callable[[], Sequence[Mapping[str, Any]]],
    lock_probe: Callable[[str], str | None],
    process_probe: Callable[[], Sequence[Mapping[str, Any]]],
    primitive_sha_probe: Callable[[str], str],
    performance_artifact_builder: Callable[..., Mapping[str, Any]] = (
        build_authenticated_performance_artifacts
    ),
    uuid_assignment_builder: Callable[..., Mapping[str, Any]] = build_uuid_assignments,
    quant_executor: Callable[..., Mapping[str, Any]] = execute_bound_quant_row,
    performance_executor: Callable[..., Mapping[str, Any]] = (
        execute_bound_performance_row
    ),
    ap_executor: Callable[..., Mapping[str, Any]] = execute_bound_ap_row,
    physical_finalizer: Callable[..., Mapping[str, Any]] = finalize_physical_rows,
    run_single_job: Callable[..., Mapping[str, Any]] | None = None,
    run_argv: Callable[[list[str], dict[str, str]], Mapping[str, Any]] | None = None,
    finalize_row: Callable[..., Mapping[str, Any]] | None = None,
    forbidden_components: Mapping[str, Callable[..., Any]] | None = None,
) -> Callable[[str, Path], Mapping[str, Any]]:
    del forbidden_components
    authenticated_projection = _jsonable(dict(projection))
    projected_rows = _rows(authenticated_projection)
    candidate_ids = [
        str(row.get("candidate_id") or row.get("row_id")) for row in projected_rows
    ]
    if projected_rows:
        supplied = _jsonable(dict(runtime_admission))
        batch = supplied.get("uuid_assignments", {})
        try:
            admission = validate_actual_runtime_admission(
                host_name=str(supplied["host_name"]),
                formal_v2_root=supplied["formal_v2_root"],
                frozen_repo_root=supplied["frozen_repo_root"],
                deployment_bundle_sha256=str(supplied["deployment_bundle_sha256"]),
                no_gpu_receipt=supplied["no_gpu_receipt"],
                physical_bindings=physical_plan["logical_row_bindings"],
                ordered_lease_uuids=batch["ordered_lease_uuids"],
                parent_cuda_visible_devices=supplied["actual_cuda_visible_devices"],
                expected_lock_owner=str(supplied["expected_lock_owner"]),
                inventory_probe=inventory_probe,
                lock_probe=lock_probe,
                process_probe=process_probe,
                primitive_sha_probe=primitive_sha_probe,
                uuid_assignment_builder=uuid_assignment_builder,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                "actual runtime admission authentication failed"
            ) from error
        if admission != supplied:
            raise ValueError("actual runtime admission authentication failed")
    else:
        admission = {}
    assignment_by_id = _assignments(admission) if projected_rows else {}
    if list(assignment_by_id) != candidate_ids:
        raise ValueError("actual runtime admission candidate order drift")
    frozen_root = (
        Path(str(admission["frozen_repo_root"])) if projected_rows else Path(".")
    )
    formal_root = (
        Path(str(admission["formal_v2_root"])) if projected_rows else Path(".")
    )
    python_path = Path(python_executable).absolute()

    def load(attempt: Path, stage: str) -> JSON:
        return _read_stage_output(
            _stage_output_path(attempt, stage),
            attempt=attempt,
            expected_stage=stage,
        )

    def finish(attempt: Path, stage: str, payload: Mapping[str, Any]) -> JSON:
        output = _write_stage_output(
            _stage_output_path(attempt, stage),
            {"stage": stage, **dict(payload)},
            attempt,
        )
        return {
            key: copy.deepcopy(value)
            for key, value in output.items()
            if key not in {"schema_version", "pipeline_output_sha256", "stage"}
        }

    def run_quant(attempt: Path) -> JSON:
        evidence: list[JSON] = []
        contracts: dict[str, str] = {}
        for row in projected_rows:
            candidate = str(row.get("candidate_id") or row.get("row_id"))
            row_root = attempt / "quant" / "rows" / candidate
            row_root.mkdir(parents=True, exist_ok=True)
            output_path = (row_root / "quant_contract.json").absolute()
            plan = quant_plan_builder(
                row=copy.deepcopy(row),
                output_path=output_path,
                row_root=row_root,
            )
            arguments = _validate_quant_plan(
                plan,
                output_path=output_path,
                attempt=attempt,
                allowed_roots=(formal_root, frozen_root, attempt),
            )
            kwargs: JSON = {
                "candidate_id": candidate,
                "q_mode": str(row.get("q_mode") or ""),
                "assignment": assignment_by_id[candidate],
                "frozen_repo_root": frozen_root,
                "python_executable": python_path,
                "quant_arguments": arguments,
                "inventory_probe": inventory_probe,
                "lock_probe": lock_probe,
                "process_probe": process_probe,
                "primitive_sha_probe": primitive_sha_probe,
            }
            if run_argv is not None:
                kwargs["run_argv"] = run_argv
            result = _jsonable(dict(quant_executor(**kwargs)))
            evidence.append(result)
            if result.get("status") == "failed":
                return finish(
                    attempt,
                    "quant",
                    {
                        "status": "infrastructure_failure",
                        "reason": "quant primitive returned failure",
                        "rows": evidence,
                    },
                )
            if str(row.get("q_mode")) == "int8":
                _quant_contract(output_path, attempt=attempt)
                contracts[candidate] = str(output_path)
        return finish(
            attempt,
            "quant",
            {
                "status": "success",
                "physical_row_count": len(projected_rows),
                "quant_contract_paths": contracts,
                "rows": evidence,
            },
        )

    def run_performance(attempt: Path) -> JSON:
        quant = load(attempt, "quant")
        quant_paths = {
            candidate: _plain_file(
                Path(str(path)),
                label="quant contract",
                allowed_roots=(attempt,),
            )
            for candidate, path in quant.get("quant_contract_paths", {}).items()
        }
        _validate_source_inputs(
            source_result, allowed_roots=(formal_root, frozen_root, attempt)
        )
        remote_root = Path(remote_artifact_root).absolute()
        if not _inside(remote_root, formal_root) and not _inside(remote_root, attempt):
            raise ValueError("remote artifact root escapes authenticated roots")
        _reject_symlink_components(remote_root)
        if remote_root.exists():
            _plain_directory(remote_root, label="remote artifact root")
        artifacts = dict(
            performance_artifact_builder(
                projection=copy.deepcopy(authenticated_projection),
                source_plan=copy.deepcopy(dict(source_plan)),
                source_result=copy.deepcopy(dict(source_result)),
                quant_contract_paths=quant_paths,
                remote_artifact_root=remote_root,
                gpus=[
                    int(assignment_by_id[candidate]["global_index"])
                    for candidate in candidate_ids
                ],
            )
        )
        jobs = artifacts.get("performance_jobs")
        if not isinstance(jobs, Sequence) or len(jobs) != len(candidate_ids):
            raise ValueError("performance artifacts job count drift")
        job_ids = [
            [str(job.get(field) or "") for job in jobs if isinstance(job, Mapping)]
            for field in ("candidate_id", "manifest_job_id")
        ]
        if job_ids != [candidate_ids, candidate_ids]:
            raise ValueError("performance job identity/order drift")
        manifest = artifacts.get("manifest")
        manifest_jobs = manifest.get("jobs") if isinstance(manifest, Mapping) else ()
        manifest_ids = [
            str(job.get("manifest_job_id") or "")
            for job in manifest_jobs
            if isinstance(job, Mapping)
        ]
        if manifest_ids != candidate_ids:
            raise ValueError("performance manifest identity/order drift")
        evidence: list[JSON] = []
        for candidate, job in zip(candidate_ids, jobs):
            if not isinstance(job, Mapping):
                raise ValueError("performance job is malformed")
            kwargs: JSON = {
                "authenticated_job": copy.deepcopy(dict(job)),
                "assignment": assignment_by_id[candidate],
                "state_jsonl": attempt / "performance" / "stage3_state.jsonl",
                "attempt_root": attempt / "performance" / "rows" / candidate,
                "inventory_probe": inventory_probe,
                "lock_probe": lock_probe,
                "process_probe": process_probe,
                "primitive_sha_probe": primitive_sha_probe,
                "frozen_repo_root": frozen_root,
            }
            if run_single_job is not None:
                kwargs["run_single_job"] = run_single_job
            result = _jsonable(dict(performance_executor(**kwargs)))
            evidence.append(result)
            if result.get("status") == "failed":
                return finish(
                    attempt,
                    "performance",
                    {
                        "status": "infrastructure_failure",
                        "reason": "performance primitive returned failure",
                        "rows": evidence,
                    },
                )
        return finish(
            attempt,
            "performance",
            {
                "status": "success",
                "physical_row_count": len(evidence),
                "performance_artifacts": artifacts,
                "performance_executions": evidence,
            },
        )

    def run_ap(attempt: Path, stage: str) -> JSON:
        performance = load(attempt, "performance")
        if stage == "sanity":
            plan_value = ap_plan_builder(
                performance_artifacts=copy.deepcopy(
                    performance["performance_artifacts"]
                ),
                performance_executions=copy.deepcopy(
                    performance["performance_executions"]
                ),
                output_root=attempt / "ap" / "plan",
            )
            plan_rows, shard_paths = _validate_ap_plan(
                plan_value,
                candidate_ids=candidate_ids,
                attempt=attempt,
            )
        else:
            sanity = load(attempt, "sanity")
            plan_rows = copy.deepcopy(sanity["ap_plan_rows"])
            shard_paths = copy.deepcopy(sanity["ap_plan_shards"])
            _validate_ap_plan(
                {"rows": plan_rows, "shards": shard_paths},
                candidate_ids=candidate_ids,
                attempt=attempt,
            )
        evidence: list[JSON] = []
        for candidate in candidate_ids:
            kwargs: JSON = {
                "candidate_id": candidate,
                "stage": stage,
                "assignment": assignment_by_id[candidate],
                "frozen_repo_root": frozen_root,
                "python_executable": python_path,
                "ap_plan_jsonl": Path(shard_paths[candidate]),
                "expected_manifest_job_id": candidate,
                "state_jsonl": attempt / "ap" / f"{stage}_state.jsonl",
                "artifact_root": attempt / "ap" / stage / candidate,
                "inventory_probe": inventory_probe,
                "lock_probe": lock_probe,
                "process_probe": process_probe,
                "primitive_sha_probe": primitive_sha_probe,
            }
            if run_argv is not None:
                kwargs["run_argv"] = run_argv
            result = _jsonable(dict(ap_executor(**kwargs)))
            evidence.append(result)
            if result.get("status") == "failed":
                return finish(
                    attempt,
                    stage,
                    {
                        "status": "infrastructure_failure",
                        "reason": f"AP {stage} primitive returned failure",
                        "rows": evidence,
                        "ap_plan_rows": plan_rows,
                        "ap_plan_shards": shard_paths,
                    },
                )
        return finish(
            attempt,
            stage,
            {
                "status": "success",
                "physical_row_count": len(evidence),
                "rows": evidence,
                "ap_plan_rows": plan_rows,
                "ap_plan_shards": shard_paths,
            },
        )

    def run_final(attempt: Path) -> JSON:
        performance = load(attempt, "performance")
        sanity = load(attempt, "sanity")
        full = load(attempt, "full")
        evidence = finalizer_evidence_builder(
            performance_artifacts=copy.deepcopy(performance["performance_artifacts"]),
            performance_executions=copy.deepcopy(performance["performance_executions"]),
            ap_plan_rows=copy.deepcopy(full["ap_plan_rows"]),
            sanity_executions=copy.deepcopy(sanity["rows"]),
            full_executions=copy.deepcopy(full["rows"]),
            output_root=attempt / "final",
        )
        required = {
            "performance_state_rows",
            "ap_plan_rows",
            "ap_state_rows",
            "structured_failure_reports",
            "lineage_inputs",
            "execution_attempt",
        }
        if not isinstance(evidence, Mapping) or set(evidence) != required:
            raise ValueError("finalizer evidence mapping is incomplete")
        kwargs: JSON = {
            "logical_request": copy.deepcopy(dict(logical_request)),
            "selection_binding": copy.deepcopy(dict(selection_binding)),
            "physical_plan": copy.deepcopy(dict(physical_plan)),
            "projection": copy.deepcopy(authenticated_projection),
            "performance_artifacts": copy.deepcopy(
                performance["performance_artifacts"]
            ),
            **copy.deepcopy(dict(evidence)),
        }
        if finalize_row is not None:
            kwargs["finalize_row"] = finalize_row
        terminal = _jsonable(dict(physical_finalizer(**kwargs)))
        if (
            terminal.get("schema_version")
            != "stage7_actual_v3_physical_terminal_batch_v2"
        ):
            raise ValueError("Phase5D physical terminal schema drift")
        return finish(
            attempt,
            "final",
            {
                "status": "success",
                "terminal_payload": terminal,
                "physical_row_count": len(projected_rows),
            },
        )

    def stage_runner(stage: str, attempt_path: Path) -> Mapping[str, Any]:
        if stage not in STAGES:
            raise ValueError("actual pipeline stage is invalid")
        attempt = _validated_attempt(Path(attempt_path))
        output_path = _stage_output_path(attempt, stage)
        if output_path.is_file():
            output = load(attempt, stage)
            return {
                key: copy.deepcopy(value)
                for key, value in output.items()
                if key not in {"schema_version", "pipeline_output_sha256", "stage"}
            }
        if not projected_rows:
            payload: JSON = {
                "status": "success",
                "physical_row_count": 0,
                "gpu_subprocess_count": 0,
            }
            if stage == "final":
                payload["terminal_payload"] = copy.deepcopy(authenticated_projection)
            return finish(attempt, stage, payload)
        prior = {
            "performance": "quant",
            "sanity": "performance",
            "full": "sanity",
            "final": "full",
        }.get(stage)
        if prior and (
            not _stage_output_path(attempt, prior).is_file()
            or load(attempt, prior).get("status") != "success"
        ):
            raise ValueError(f"{stage} requires a successful prior stage")
        if stage == "quant":
            return run_quant(attempt)
        if stage == "performance":
            return run_performance(attempt)
        if stage in {"sanity", "full"}:
            return run_ap(attempt, stage)
        return run_final(attempt)

    return stage_runner
