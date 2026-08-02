"""UUID-bound Stage7 wrappers around frozen Stage3 execution primitives."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


RUNTIME_PRIMITIVE_SHA256 = {
    "scripts/stage3_tvm_int8_quant_contract_v3.py": (
        "788556b775a1378e75458623c9c3186a417d30609200979694079ec628536fe5"
    ),
    "scripts/stage3_execute_performance_plan_v3.py": (
        "3c0af3d857570a2bdba6b6ff5fe50a532abb5e079370119923ff0dae16f57335"
    ),
    "scripts/stage3_execute_ap_plan_v3.py": (
        "da71d1ba94480b9034c9dc3b16011ba0bba83166e478f865934b80b50dbecaff"
    ),
}


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_sha_probe(frozen_repo_root: Path) -> Callable[[str], str]:
    root = Path(frozen_repo_root).resolve()
    return lambda relative: _file_sha(root / relative)


def verify_runtime_primitives(
    primitive_sha_probe: Callable[[str], str],
) -> dict[str, str]:
    observed = {
        relative: str(primitive_sha_probe(relative))
        for relative in RUNTIME_PRIMITIVE_SHA256
    }
    if observed != RUNTIME_PRIMITIVE_SHA256:
        raise RuntimeError("runtime primitive SHA drift")
    return observed


def _normalized_inventory(
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise ValueError("GPU inventory must be a sequence")
    normalized: list[dict[str, Any]] = []
    by_uuid: dict[str, dict[str, Any]] = {}
    indices: set[int] = set()
    for raw in inventory:
        if not isinstance(raw, Mapping):
            raise ValueError("GPU inventory row is malformed")
        uuid = str(raw.get("uuid") or "")
        name = str(raw.get("name") or "")
        try:
            global_index = int(raw["global_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("GPU global index is malformed") from exc
        if (
            not uuid.startswith("GPU-")
            or uuid in by_uuid
            or global_index in indices
            or "H800" not in name.upper()
        ):
            raise ValueError("GPU UUID/index/model inventory is invalid")
        row = {
            "global_index": global_index,
            "uuid": uuid,
            "name": name,
        }
        normalized.append(row)
        by_uuid[uuid] = row
        indices.add(global_index)
    return normalized, by_uuid


def _validate_related_processes(
    related_processes: Sequence[Mapping[str, Any]],
    leased_uuids: Sequence[str],
    *,
    expected_lock_owner: str,
) -> list[dict[str, Any]]:
    if isinstance(related_processes, (str, bytes)) or not isinstance(
        related_processes, Sequence
    ):
        raise ValueError("related process inventory must be a sequence")
    leased = set(leased_uuids)
    normalized: list[dict[str, Any]] = []
    formal_jobs_by_uuid: dict[str, set[str]] = {}
    for raw in related_processes:
        if not isinstance(raw, Mapping):
            raise ValueError("related process row is malformed")
        row = copy.deepcopy(dict(raw))
        if row.get("related") is True:
            gpu_uuid = str(row.get("gpu_uuid") or "")
            if gpu_uuid not in leased:
                raise ValueError("related process is using an unleased GPU UUID")
            if (
                row.get("controller_owned") is not True
                or row.get("owner") != expected_lock_owner
            ):
                raise ValueError("foreign related process is using a leased GPU UUID")
            formal_job_id = str(row.get("formal_job_id") or "")
            if formal_job_id:
                formal_jobs_by_uuid.setdefault(gpu_uuid, set()).add(formal_job_id)
        normalized.append(row)
    if any(len(job_ids) > 1 for job_ids in formal_jobs_by_uuid.values()):
        raise ValueError("multiple formal jobs share one leased GPU UUID")
    return normalized


def _assignment_payload(
    *,
    candidate_id: str,
    logical_row_index: int,
    leased_uuid: str,
    global_index: int,
    lock_owner: str,
    ordered_lease_uuids: Sequence[str],
    parent_cuda_visible_devices: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "logical_row_index": logical_row_index,
        "lease_slot": logical_row_index,
        "leased_uuid": leased_uuid,
        "global_index": global_index,
        "runtime_visible_ordinal": 0,
        "lock_owner": lock_owner,
        "ordered_lease_uuids": list(ordered_lease_uuids),
        "parent_cuda_visible_devices": parent_cuda_visible_devices,
    }


def build_uuid_assignments(
    *,
    physical_bindings: Sequence[Mapping[str, Any]],
    ordered_lease_uuids: Sequence[str],
    parent_cuda_visible_devices: str,
    inventory: Sequence[Mapping[str, Any]],
    lock_owners: Mapping[str, str | None],
    expected_lock_owner: str,
    related_processes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    leases = tuple(str(value) for value in ordered_lease_uuids)
    if (
        len(leases) != 4
        or len(set(leases)) != 4
        or any(not value.startswith("GPU-") for value in leases)
    ):
        raise ValueError("exactly four unique GPU UUID leases are required")
    if parent_cuda_visible_devices != ",".join(leases):
        raise ValueError("parent CUDA_VISIBLE_DEVICES order differs from leases")
    normalized_inventory, by_uuid = _normalized_inventory(inventory)
    if any(uuid not in by_uuid for uuid in leases):
        raise ValueError("leased GPU UUID is absent from inventory")
    if not expected_lock_owner or any(
        lock_owners.get(uuid) != expected_lock_owner for uuid in leases
    ):
        raise ValueError("GPU UUID lock ownership is incomplete")
    normalized_processes = _validate_related_processes(
        related_processes,
        leases,
        expected_lock_owner=expected_lock_owner,
    )
    assignments: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for binding in physical_bindings:
        if not isinstance(binding, Mapping) or binding.get("disposition") != "miss":
            raise ValueError("physical binding must be a cache miss")
        try:
            logical_index = int(binding["logical_row_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("logical row index is invalid") from exc
        candidate_id = str(binding.get("candidate_id") or "")
        if (
            logical_index not in range(4)
            or logical_index in seen_indices
            or not candidate_id
        ):
            raise ValueError("physical logical-row binding is invalid")
        leased_uuid = leases[logical_index]
        payload = _assignment_payload(
            candidate_id=candidate_id,
            logical_row_index=logical_index,
            leased_uuid=leased_uuid,
            global_index=int(by_uuid[leased_uuid]["global_index"]),
            lock_owner=expected_lock_owner,
            ordered_lease_uuids=leases,
            parent_cuda_visible_devices=parent_cuda_visible_devices,
        )
        assignments.append({**payload, "assignment_sha256": _sha(payload)})
        seen_indices.add(logical_index)
    batch = {
        "schema_version": "stage7_uuid_assignment_batch_v2",
        "ordered_lease_uuids": list(leases),
        "parent_cuda_visible_devices": parent_cuda_visible_devices,
        "expected_lock_owner": expected_lock_owner,
        "pre_inventory_sha256": _sha(normalized_inventory),
        "pre_related_processes_sha256": _sha(normalized_processes),
        "assignments": assignments,
    }
    return {**batch, "assignment_batch_sha256": _sha(batch)}


def _validated_assignment(assignment: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(assignment, Mapping):
        raise ValueError("runtime assignment is missing")
    copied = copy.deepcopy(dict(assignment))
    recorded = copied.pop("assignment_sha256", None)
    leases = copied.get("ordered_lease_uuids")
    logical_index = copied.get("logical_row_index")
    if (
        recorded != _sha(copied)
        or not isinstance(leases, list)
        or len(leases) != 4
        or len(set(leases)) != 4
        or logical_index not in range(4)
        or copied.get("lease_slot") != logical_index
        or copied.get("leased_uuid") != leases[logical_index]
        or copied.get("runtime_visible_ordinal") != 0
        or copied.get("parent_cuda_visible_devices") != ",".join(leases)
        or not isinstance(copied.get("global_index"), int)
        or not copied.get("candidate_id")
        or not copied.get("lock_owner")
    ):
        raise ValueError("runtime assignment authentication failed")
    return {**copied, "assignment_sha256": recorded}


def _runtime_snapshot(
    *,
    assignment: Mapping[str, Any],
    inventory_probe: Callable[[], Sequence[Mapping[str, Any]]],
    lock_probe: Callable[[str], str | None],
    process_probe: Callable[[], Sequence[Mapping[str, Any]]],
    phase: str,
) -> dict[str, Any]:
    leases = assignment["ordered_lease_uuids"]
    normalized_inventory, by_uuid = _normalized_inventory(inventory_probe())
    if any(uuid not in by_uuid for uuid in leases):
        raise RuntimeError(f"{phase} leased UUID missing")
    leased_uuid = assignment["leased_uuid"]
    if by_uuid[leased_uuid]["global_index"] != assignment["global_index"]:
        raise RuntimeError(f"{phase} UUID to global-index mapping drift")
    lock_owners = {uuid: lock_probe(uuid) for uuid in leases}
    if any(owner != assignment["lock_owner"] for owner in lock_owners.values()):
        raise RuntimeError(f"{phase} UUID lock ownership drift")
    try:
        processes = _validate_related_processes(
            process_probe(),
            leases,
            expected_lock_owner=str(assignment["lock_owner"]),
        )
    except ValueError as exc:
        raise RuntimeError(f"{phase} unleased related process") from exc
    payload = {
        "phase": phase,
        "inventory": normalized_inventory,
        "lock_owners": lock_owners,
        "related_processes": processes,
    }
    return {**payload, "runtime_snapshot_sha256": _sha(payload)}


def _replace_gpu_token(command: Any, leased_uuid: str) -> list[str]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise TypeError("performance command must be an argv array")
    argv = [str(value) for value in command]
    if "--gpus" in argv or argv.count("--gpu") != 1:
        raise ValueError("performance command must contain exactly one --gpu")
    index = argv.index("--gpu")
    if index + 1 >= len(argv):
        raise ValueError("performance --gpu is missing its value")
    return [*argv[: index + 1], leased_uuid, *argv[index + 2 :]]


@contextmanager
def _single_uuid_environment(leased_uuid: str) -> Any:
    sentinel = object()
    previous: object | str = os.environ.get("CUDA_VISIBLE_DEVICES", sentinel)
    os.environ["CUDA_VISIBLE_DEVICES"] = leased_uuid
    try:
        yield
    finally:
        if previous is sentinel:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(previous)


def _load_run_single_job(frozen_repo_root: Path) -> Callable[..., dict[str, Any]]:
    path = (
        Path(frozen_repo_root).resolve()
        / "scripts/stage3_execute_performance_plan_v3.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_stage7_pinned_stage3_performance_v3", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Stage3 performance primitive")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_single_job


def _primitive_probe(
    *,
    frozen_repo_root: Path | None,
    primitive_sha_probe: Callable[[str], str] | None,
) -> Callable[[str], str]:
    if primitive_sha_probe is not None:
        return primitive_sha_probe
    if frozen_repo_root is None:
        raise ValueError("frozen repo root is required for primitive verification")
    return _default_sha_probe(frozen_repo_root)


def execute_bound_performance_row(
    *,
    authenticated_job: Mapping[str, Any],
    assignment: Mapping[str, Any],
    state_jsonl: Path,
    attempt_root: Path,
    inventory_probe: Callable[[], Sequence[Mapping[str, Any]]],
    lock_probe: Callable[[str], str | None],
    process_probe: Callable[[], Sequence[Mapping[str, Any]]],
    primitive_sha_probe: Callable[[str], str] | None = None,
    run_single_job: Callable[..., dict[str, Any]] | None = None,
    frozen_repo_root: Path | None = None,
) -> dict[str, Any]:
    pins = verify_runtime_primitives(
        _primitive_probe(
            frozen_repo_root=frozen_repo_root,
            primitive_sha_probe=primitive_sha_probe,
        )
    )
    bound_assignment = _validated_assignment(assignment)
    original = copy.deepcopy(dict(authenticated_job))
    if original.get("candidate_id") != bound_assignment["candidate_id"]:
        raise ValueError("performance job candidate differs from assignment")
    overlay = copy.deepcopy(original)
    overlay.pop("runtime_gpu", None)
    overlay["command"] = _replace_gpu_token(
        overlay.get("command"), bound_assignment["leased_uuid"]
    )
    pre = _runtime_snapshot(
        assignment=bound_assignment,
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        phase="pre-runtime",
    )
    primitive = run_single_job
    if primitive is None:
        if frozen_repo_root is None:
            raise ValueError("frozen repo root is required for Stage3 import")
        primitive = _load_run_single_job(frozen_repo_root)
    with _single_uuid_environment(bound_assignment["leased_uuid"]):
        state_row = primitive(
            job=copy.deepcopy(overlay),
            state_jsonl=Path(state_jsonl),
            attempt_root=Path(attempt_root),
        )
    post = _runtime_snapshot(
        assignment=bound_assignment,
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        phase="post-runtime",
    )
    if dict(authenticated_job) != original:
        raise RuntimeError("authenticated performance job was mutated")
    payload = {
        "schema_version": "stage7_bound_performance_execution_v2",
        "candidate_id": bound_assignment["candidate_id"],
        "job_id": original.get("job_id"),
        "assignment": bound_assignment,
        "authenticated_job_sha256": _sha(original),
        "execution_overlay_sha256": _sha(overlay),
        "child_environment": {
            "CUDA_VISIBLE_DEVICES": bound_assignment["leased_uuid"],
            "runtime_visible_ordinal": 0,
        },
        "primitive_sha256": pins,
        "pre_runtime_audit": pre,
        "post_runtime_audit": post,
        "stage3_state_row": copy.deepcopy(state_row),
    }
    return {**payload, "execution_evidence_sha256": _sha(payload)}


def _validated_argv(arguments: Sequence[str], *, label: str) -> list[str]:
    if isinstance(arguments, (str, bytes)) or not isinstance(arguments, Sequence):
        raise TypeError(f"{label} arguments must be an argv array")
    return [str(value) for value in arguments]


def _run_subprocess_argv(argv: list[str], env: dict[str, str]) -> dict[str, Any]:
    completed = subprocess.run(
        argv, env=env, text=True, capture_output=True, check=False
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _bound_primitive_call(
    *,
    assignment: Mapping[str, Any],
    inventory_probe: Callable[[], Sequence[Mapping[str, Any]]],
    lock_probe: Callable[[str], str | None],
    process_probe: Callable[[], Sequence[Mapping[str, Any]]],
    run: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pre = _runtime_snapshot(
        assignment=assignment,
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        phase="pre-runtime",
    )
    result = run()
    post = _runtime_snapshot(
        assignment=assignment,
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        phase="post-runtime",
    )
    return pre, copy.deepcopy(result), post


def execute_bound_quant_row(
    *,
    candidate_id: str,
    q_mode: str,
    assignment: Mapping[str, Any],
    frozen_repo_root: Path,
    python_executable: Path,
    quant_arguments: Sequence[str],
    inventory_probe: Callable[[], Sequence[Mapping[str, Any]]],
    lock_probe: Callable[[str], str | None],
    process_probe: Callable[[], Sequence[Mapping[str, Any]]],
    primitive_sha_probe: Callable[[str], str] | None = None,
    run_argv: Callable[[list[str], dict[str, str]], dict[str, Any]] = (
        _run_subprocess_argv
    ),
) -> dict[str, Any]:
    if q_mode not in {"fp16", "int8"}:
        raise ValueError("runtime quant mode must be fp16 or int8")
    pins = verify_runtime_primitives(
        _primitive_probe(
            frozen_repo_root=frozen_repo_root,
            primitive_sha_probe=primitive_sha_probe,
        )
    )
    bound_assignment = _validated_assignment(assignment)
    if candidate_id != bound_assignment["candidate_id"]:
        raise ValueError("quant candidate differs from assignment")
    if q_mode == "fp16":
        pre = _runtime_snapshot(
            assignment=bound_assignment,
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            phase="pre-runtime",
        )
        payload = {
            "schema_version": "stage7_bound_quant_execution_v2",
            "candidate_id": candidate_id,
            "q_mode": q_mode,
            "status": "skipped_fp16",
            "gpu_subprocess_count": 0,
            "assignment": bound_assignment,
            "primitive_sha256": pins,
            "pre_runtime_audit": pre,
        }
        return {**payload, "execution_evidence_sha256": _sha(payload)}
    arguments = _validated_argv(quant_arguments, label="quant")
    if any(flag in arguments for flag in ("--gpu", "--gpus")):
        raise ValueError("quant GPU selection is controlled by the UUID wrapper")
    argv = [
        str(python_executable),
        str(
            Path(frozen_repo_root).resolve()
            / "scripts/stage3_tvm_int8_quant_contract_v3.py"
        ),
        *arguments,
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": bound_assignment["leased_uuid"]}
    pre, result, post = _bound_primitive_call(
        assignment=bound_assignment,
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        run=lambda: run_argv(argv, env),
    )
    payload = {
        "schema_version": "stage7_bound_quant_execution_v2",
        "candidate_id": candidate_id,
        "q_mode": q_mode,
        "status": "success" if int(result.get("returncode", 1)) == 0 else "failed",
        "assignment": bound_assignment,
        "argv": argv,
        "child_environment": {
            "CUDA_VISIBLE_DEVICES": bound_assignment["leased_uuid"],
            "runtime_visible_ordinal": 0,
        },
        "primitive_sha256": pins,
        "pre_runtime_audit": pre,
        "post_runtime_audit": post,
        "primitive_result": result,
    }
    return {**payload, "execution_evidence_sha256": _sha(payload)}


def _validated_ap_shard(
    ap_plan_jsonl: Path,
    *,
    expected_manifest_job_id: str,
) -> tuple[dict[str, Any], str]:
    path = Path(ap_plan_jsonl)
    if not expected_manifest_job_id or not path.is_file() or path.is_symlink():
        raise ValueError("AP plan shard is missing or unauthenticated")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("AP plan shard is malformed") from exc
    if (
        len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or rows[0].get("schema_version") != "stage5_ap_plan_v2"
        or rows[0].get("manifest_job_id") != expected_manifest_job_id
    ):
        raise ValueError("AP plan must be one authenticated candidate shard")
    return copy.deepcopy(dict(rows[0])), _file_sha(path)


def execute_bound_ap_row(
    *,
    candidate_id: str,
    stage: str,
    assignment: Mapping[str, Any],
    frozen_repo_root: Path,
    python_executable: Path,
    ap_plan_jsonl: Path,
    expected_manifest_job_id: str,
    state_jsonl: Path,
    artifact_root: Path,
    inventory_probe: Callable[[], Sequence[Mapping[str, Any]]],
    lock_probe: Callable[[str], str | None],
    process_probe: Callable[[], Sequence[Mapping[str, Any]]],
    primitive_sha_probe: Callable[[str], str] | None = None,
    run_argv: Callable[[list[str], dict[str, str]], dict[str, Any]] = (
        _run_subprocess_argv
    ),
    extra_arguments: Sequence[str] = (),
) -> dict[str, Any]:
    if stage not in {"sanity", "full"}:
        raise ValueError("AP stage must be sanity or full")
    extras = _validated_argv(extra_arguments, label="AP")
    if any(flag in extras for flag in ("--gpu", "--gpus")):
        raise ValueError("AP GPU selection must be a single bound --gpu")
    pins = verify_runtime_primitives(
        _primitive_probe(
            frozen_repo_root=frozen_repo_root,
            primitive_sha_probe=primitive_sha_probe,
        )
    )
    bound_assignment = _validated_assignment(assignment)
    if candidate_id != bound_assignment["candidate_id"]:
        raise ValueError("AP candidate differs from assignment")
    ap_plan_row, ap_plan_file_sha256 = _validated_ap_shard(
        ap_plan_jsonl,
        expected_manifest_job_id=expected_manifest_job_id,
    )
    argv = [
        str(python_executable),
        str(Path(frozen_repo_root).resolve() / "scripts/stage3_execute_ap_plan_v3.py"),
        "--ap-plan-jsonl",
        str(ap_plan_jsonl),
        "--stage",
        stage,
        "--state-jsonl",
        str(state_jsonl),
        "--gpu",
        str(bound_assignment["global_index"]),
        "--artifact-root",
        str(artifact_root),
        *extras,
    ]
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    pre, result, post = _bound_primitive_call(
        assignment=bound_assignment,
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        run=lambda: run_argv(argv, env),
    )
    payload = {
        "schema_version": "stage7_bound_ap_execution_v2",
        "candidate_id": candidate_id,
        "manifest_job_id": expected_manifest_job_id,
        "stage": stage,
        "status": "success" if int(result.get("returncode", 1)) == 0 else "failed",
        "assignment": bound_assignment,
        "argv": argv,
        "ap_plan_row_sha256": _sha(ap_plan_row),
        "ap_plan_file_sha256": ap_plan_file_sha256,
        "child_environment": {"CUDA_VISIBLE_DEVICES": None},
        "primitive_sha256": pins,
        "pre_runtime_audit": pre,
        "post_runtime_audit": post,
        "primitive_result": result,
    }
    return {**payload, "execution_evidence_sha256": _sha(payload)}


def execute_runtime_rows(
    *,
    runtime_rows: Sequence[Mapping[str, Any]],
    inventory_probe: Callable[[], Any],
    primitive_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    forbidden_components: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    del forbidden_components
    rows = [copy.deepcopy(dict(row)) for row in runtime_rows]
    if not rows:
        return {
            "schema_version": "stage7_empty_physical_runtime_v2",
            "physical_row_count": 0,
            "gpu_subprocess_count": 0,
            "rows": [],
        }
    inventory_probe()
    results = [copy.deepcopy(dict(primitive_runner(row))) for row in rows]
    return {
        "schema_version": "stage7_physical_runtime_batch_v2",
        "physical_row_count": len(results),
        "gpu_subprocess_count": 0,
        "rows": results,
    }


__all__ = [
    "RUNTIME_PRIMITIVE_SHA256",
    "build_uuid_assignments",
    "execute_bound_ap_row",
    "execute_bound_performance_row",
    "execute_bound_quant_row",
    "execute_runtime_rows",
    "verify_runtime_primitives",
]
