"""Concrete support adapters for the reviewed Stage7 actual-mode pipeline.

This module only prepares argv/contracts and reads runtime evidence.  Frozen
Stage3/Stage5 code remains the owner of quantization, measurement, AP, and row
finalization semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
from typing import Any, Callable, Mapping, Sequence

from scripts import stage3_gold96_ap_plan_v3


JSON = dict[str, Any]
SCHEDULER_SCHEMA = "stage7_ablation_scheduler_state_v1"


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def canonical_sha256(payload: object) -> str:
    """Public canonical JSON SHA used by execution-attempt contracts."""
    return _sha(payload)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> JSON:
    source = Path(path).absolute()
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve() != source
        or source.stat().st_nlink != 1
    ):
        raise ValueError(f"{label} must be a regular canonical file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return copy.deepcopy(dict(payload))


def _write_immutable(path: Path, content: str, *, root: Path) -> None:
    destination = Path(path).absolute()
    boundary = Path(root).absolute()
    try:
        relative = destination.relative_to(boundary)
    except ValueError as error:
        raise ValueError("immutable output escapes its root") from error
    current = boundary
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("immutable output parent contains a symlink")
        if current.exists() and not current.is_dir():
            raise ValueError("immutable output parent is not a directory")
        current.mkdir(exist_ok=True)
    encoded = content.encode("utf-8")
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_nlink != 1
            or destination.read_bytes() != encoded
        ):
            raise ValueError("immutable output drift")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
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


def build_quant_plan(
    *,
    row: Mapping[str, Any],
    output_path: Path,
    row_root: Path,
) -> JSON:
    """Build the frozen Stage3 quant CLI arguments without running it."""
    del row_root
    output = Path(output_path).absolute()
    if str(row.get("q_mode") or "") == "fp16":
        return {
            "arguments": ["--output-json", str(output)],
            "input_paths": [],
        }
    if str(row.get("q_mode") or "") != "int8":
        raise ValueError("quant plan requires fp16 or int8")
    source = row.get("source_contract")
    if not isinstance(source, Mapping):
        raise ValueError("INT8 quant plan requires a source contract")
    onnx = str(source.get("onnx_path") or "")
    calibration = str(source.get("calibration_npz") or "")
    summary = str(source.get("calibration_summary") or "")
    if not all((onnx, calibration, summary)):
        raise ValueError("INT8 quant plan inputs are incomplete")
    return {
        "arguments": [
            "--onnx",
            onnx,
            "--calibration-npz",
            calibration,
            "--calibration-summary",
            summary,
            "--output-json",
            str(output),
        ],
        "input_paths": [onnx, calibration, summary],
    }


def query_h800_inventory(
    *,
    run_command: Callable[..., Any] = subprocess.run,
) -> list[JSON]:
    """Query index/UUID/model without initializing CUDA."""
    completed = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if int(completed.returncode) != 0:
        detail = str(completed.stderr).strip()
        raise RuntimeError(f"nvidia-smi inventory failed: {detail}")
    rows: list[JSON] = []
    indices: set[int] = set()
    uuids: set[str] = set()
    for line in str(completed.stdout).splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise ValueError("nvidia-smi inventory row is malformed")
        try:
            index = int(parts[0])
        except ValueError as error:
            raise ValueError("nvidia-smi inventory row is malformed") from error
        uuid, name = parts[1:]
        if (
            index in indices
            or uuid in uuids
            or not uuid.startswith("GPU-")
            or "H800" not in name.upper()
        ):
            raise ValueError("nvidia-smi inventory row is malformed")
        indices.add(index)
        uuids.add(uuid)
        rows.append({"global_index": index, "uuid": uuid, "name": name})
    if not rows:
        raise ValueError("nvidia-smi inventory is empty")
    return rows


def validate_scheduler_lease(
    *,
    scheduler_state_path: Path,
    controller_id: str,
    logical_request_sha256: str,
    ordered_lease_uuids: Sequence[str],
    ancestor_pids: Sequence[int],
) -> JSON:
    """Bind actual execution to a live scheduler-owned controller lease."""
    path = Path(scheduler_state_path).absolute()
    state = _read_mapping(path, label="scheduler lease state")
    controllers = state.get("controllers")
    controller = (
        controllers.get(controller_id) if isinstance(controllers, Mapping) else None
    )
    leases = [str(value) for value in ordered_lease_uuids]
    if (
        state.get("schema_version") != SCHEDULER_SCHEMA
        or not isinstance(controller, Mapping)
        or controller.get("controller_id") != controller_id
        or controller.get("status") != "running"
        or controller.get("request_sha256") != logical_request_sha256
        or controller.get("gpu_uuids") != leases
        or int(controller.get("pid", -1)) not in set(ancestor_pids)
    ):
        raise ValueError("scheduler lease does not bind this live controller")
    unsigned = {
        "schema_version": "stage7_actual_scheduler_lease_binding_v2",
        "scheduler_state_path": str(path),
        "scheduler_state_file_sha256": _file_sha(path),
        "controller_id": controller_id,
        "controller_pid": int(controller["pid"]),
        "logical_request_sha256": logical_request_sha256,
        "ordered_lease_uuids": leases,
        "expected_lock_owner": controller_id,
    }
    return {**unsigned, "scheduler_lease_binding_sha256": _sha(unsigned)}


def build_scheduler_lock_probe(
    *,
    scheduler_state_path: Path,
    controller_id: str,
    logical_request_sha256: str,
    ordered_lease_uuids: Sequence[str],
    ancestor_pids: Sequence[int],
) -> Callable[[str], str | None]:
    """Revalidate scheduler ownership at every Phase5C runtime boundary."""
    leases = tuple(str(value) for value in ordered_lease_uuids)

    def probe(uuid: str) -> str | None:
        binding = validate_scheduler_lease(
            scheduler_state_path=scheduler_state_path,
            controller_id=controller_id,
            logical_request_sha256=logical_request_sha256,
            ordered_lease_uuids=leases,
            ancestor_pids=ancestor_pids,
        )
        return binding["expected_lock_owner"] if uuid in leases else None

    return probe


def build_ap_plan(
    *,
    performance_artifacts: Mapping[str, Any],
    performance_executions: Sequence[Mapping[str, Any]],
    output_root: Path,
    ap_plan_function: Callable[..., list[JSON]] = (
        stage3_gold96_ap_plan_v3.build_ap_plan
    ),
) -> JSON:
    """Call the existing Stage5-compatible AP plan function and shard by row."""
    manifest = performance_artifacts.get("manifest")
    jobs = performance_artifacts.get("performance_jobs")
    if not isinstance(manifest, Mapping) or not isinstance(jobs, Sequence):
        raise ValueError("authenticated performance artifacts are incomplete")
    count = int(manifest.get("row_count", -1))
    state_rows = [
        copy.deepcopy(dict(execution["stage3_state_row"]))
        for execution in performance_executions
        if isinstance(execution, Mapping)
        and isinstance(execution.get("stage3_state_row"), Mapping)
    ]
    if count not in range(1, 5) or len(jobs) != count or len(state_rows) != count:
        raise ValueError("AP planning requires one through four physical rows")
    root = Path(output_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    rows = ap_plan_function(
        copy.deepcopy(dict(manifest)),
        performance_jobs=copy.deepcopy(list(jobs)),
        performance_state_rows=state_rows,
        pilot_root=root,
        manifest_schema="stage5_performance_manifest_v2",
        expected_row_count=count,
    )
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Stage5 AP planner row count drift")
    shards: dict[str, str] = {}
    normalized: list[JSON] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("Stage5 AP planner row is malformed")
        row = {**copy.deepcopy(dict(raw)), "schema_version": "stage5_ap_plan_v2"}
        candidate = str(row.get("manifest_job_id") or "")
        if (
            not candidate
            or Path(candidate).name != candidate
            or candidate in {".", "..", "__pycache__"}
            or candidate in shards
        ):
            raise ValueError("Stage5 AP planner candidate identity drift")
        shard = root / "shards" / f"{candidate}.jsonl"
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        _write_immutable(shard, encoded, root=root)
        shards[candidate] = str(shard)
        normalized.append(row)
    return {"rows": normalized, "shards": shards}


def run_argv(
    argv: list[str],
    environment: Mapping[str, str],
    *,
    run_command: Callable[..., Any] = subprocess.run,
) -> JSON:
    """Execute a validated argv array; shell parsing is never enabled."""
    if not argv or not all(isinstance(value, str) and value for value in argv):
        raise ValueError("subprocess argv is invalid")
    completed = run_command(
        list(argv),
        env=dict(environment),
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    return {
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout),
        "stderr": str(completed.stderr),
    }


def _jsonl_path_from_argv(execution: Mapping[str, Any]) -> Path:
    argv = execution.get("argv")
    if (
        isinstance(argv, (str, bytes))
        or not isinstance(argv, Sequence)
        or argv.count("--state-jsonl") != 1
    ):
        raise ValueError("AP execution state argv is invalid")
    index = argv.index("--state-jsonl")
    if index + 1 >= len(argv):
        raise ValueError("AP execution state argv is invalid")
    return Path(str(argv[index + 1])).absolute()


def _read_jsonl(path: Path, *, label: str) -> list[JSON]:
    source = Path(path).absolute()
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve() != source
        or source.stat().st_nlink != 1
    ):
        raise ValueError(f"{label} must be a regular canonical file")
    rows: list[JSON] = []
    try:
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{label} row must be a mapping")
            rows.append(copy.deepcopy(dict(value)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid") from error
    return rows


def _artifact_reference(
    path_value: object, digest_value: object, *, label: str
) -> JSON:
    path = Path(str(path_value or "")).absolute()
    digest = str(digest_value or "")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve() != path
        or path.stat().st_nlink != 1
        or _file_sha(path) != digest
    ):
        raise ValueError(f"{label} artifact evidence drift")
    return {"path": str(path), "artifact_sha256": digest}


def build_finalizer_evidence(
    *,
    performance_artifacts: Mapping[str, Any],
    performance_executions: Sequence[Mapping[str, Any]],
    ap_plan_rows: Sequence[Mapping[str, Any]],
    sanity_executions: Sequence[Mapping[str, Any]],
    full_executions: Sequence[Mapping[str, Any]],
    output_root: Path,
    deployment_bundle_sha256: str,
    primitive_sha256: Mapping[str, str],
) -> JSON:
    """Rehydrate frozen Stage3 state into the thin Phase5D finalizer contract."""
    manifest = performance_artifacts.get("manifest")
    jobs = manifest.get("jobs") if isinstance(manifest, Mapping) else None
    if not isinstance(jobs, Sequence) or not jobs:
        raise ValueError("finalizer performance manifest is incomplete")
    candidate_ids = [
        str(job.get("manifest_job_id") or job.get("job_id") or "")
        for job in jobs
        if isinstance(job, Mapping)
    ]
    if len(candidate_ids) != len(jobs) or any(not value for value in candidate_ids):
        raise ValueError("finalizer manifest candidate identity drift")
    performance_rows: list[JSON] = []
    performance_by_candidate: dict[str, JSON] = {}
    for execution in performance_executions:
        if not isinstance(execution, Mapping) or not isinstance(
            execution.get("stage3_state_row"), Mapping
        ):
            raise ValueError("finalizer performance execution evidence is invalid")
        candidate = str(execution.get("candidate_id") or "")
        state = copy.deepcopy(dict(execution["stage3_state_row"]))
        if candidate not in candidate_ids or candidate in performance_by_candidate:
            raise ValueError("finalizer performance candidate identity drift")
        performance_by_candidate[candidate] = state
        performance_rows.append(state)
    if set(performance_by_candidate) != set(candidate_ids):
        raise ValueError("finalizer performance state coverage drift")

    ap_rows: list[JSON] = []
    for executions, expected_stage in (
        (sanity_executions, "sanity"),
        (full_executions, "full"),
    ):
        paths = {
            _jsonl_path_from_argv(execution)
            for execution in executions
            if isinstance(execution, Mapping)
        }
        for path in sorted(paths):
            stage_rows = _read_jsonl(path, label=f"AP {expected_stage} state")
            if any(row.get("stage") != expected_stage for row in stage_rows):
                raise ValueError("AP state stage drift")
            ap_rows.extend(stage_rows)

    full_by_candidate = {
        str(row.get("manifest_job_id") or ""): row
        for row in ap_rows
        if row.get("stage") == "full" and row.get("status") == "success"
    }
    lineages: list[JSON] = []
    for candidate in candidate_ids:
        performance = performance_by_candidate[candidate]
        if performance.get("status") != "success":
            continue
        full = full_by_candidate.get(candidate)
        if full is None:
            continue
        lineages.append(
            {
                "candidate_id": candidate,
                "stage3_performance_artifact": _artifact_reference(
                    performance.get("result_json"),
                    performance.get("result_sha256"),
                    label="performance",
                ),
                "stage3_ap_artifact": _artifact_reference(
                    full.get("report_path"),
                    full.get("report_sha256"),
                    label="AP",
                ),
            }
        )
    attempt_unsigned = {
        "attempt_id": Path(output_root).absolute().parent.name,
        "deployment_bundle_sha256": deployment_bundle_sha256,
        "primitive_sha256": dict(primitive_sha256),
    }
    attempt = {
        **attempt_unsigned,
        "execution_attempt_sha256": _sha(attempt_unsigned),
    }
    return {
        "performance_state_rows": performance_rows,
        "ap_plan_rows": copy.deepcopy(list(ap_plan_rows)),
        "ap_state_rows": ap_rows,
        "structured_failure_reports": [],
        "lineage_inputs": lineages,
        "execution_attempt": attempt,
    }


def query_related_processes(
    *,
    leased_uuids: Sequence[str],
    controller_pid: int,
    formal_v2_root: Path,
    expected_lock_owner: str = "stage7-controller",
    run_command: Callable[..., Any] = subprocess.run,
    process_probe: Callable[[int], Mapping[str, Any] | None],
) -> list[JSON]:
    """Classify active compute PIDs for the Phase5C exclusivity gate."""
    completed = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )
    if int(completed.returncode) != 0:
        raise RuntimeError(
            "nvidia-smi process query failed: " + str(completed.stderr).strip()
        )
    leased = set(str(value) for value in leased_uuids)
    root_text = str(Path(formal_v2_root).absolute())
    result: list[JSON] = []
    for line in str(completed.stdout).splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0].startswith("GPU-"):
            raise ValueError("nvidia-smi process row is malformed")
        try:
            pid = int(parts[1])
        except ValueError as error:
            raise ValueError("nvidia-smi process row is malformed") from error
        identity = process_probe(pid)
        if not isinstance(identity, Mapping):
            raise ValueError("compute process identity is unavailable")
        command = tuple(str(value) for value in identity.get("command") or ())
        ancestors = tuple(int(value) for value in identity.get("ancestor_pids") or ())
        owned = controller_pid in ancestors
        related = parts[0] in leased or any(root_text in value for value in command)
        result.append(
            {
                "pid": pid,
                "gpu_uuid": parts[0],
                "related": related,
                "controller_owned": owned,
                "owner": (
                    expected_lock_owner if owned else str(identity.get("owner") or "")
                ),
                "formal_job_id": (f"controller-{controller_pid}" if owned else ""),
            }
        )
    return result


def inspect_linux_process(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    uid_name_probe: Callable[[int], str] = lambda uid: pwd.getpwuid(uid).pw_name,
) -> JSON | None:
    """Read a process identity and its parent chain without mutating procfs."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process PID is invalid")
    root = Path(proc_root)

    def status_fields(current: int) -> tuple[int, int]:
        path = root / str(current) / "status"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise ValueError("process status is unavailable") from error
        values = {
            key: value.strip()
            for line in lines
            if ":" in line
            for key, value in (line.split(":", 1),)
        }
        try:
            parent = int(values["PPid"].split()[0])
            uid = int(values["Uid"].split()[0])
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError("process status is malformed") from error
        return parent, uid

    try:
        raw_command = (root / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    command = tuple(
        value.decode("utf-8", errors="replace")
        for value in raw_command.split(b"\0")
        if value
    )
    parent, uid = status_fields(pid)
    ancestors = [pid]
    seen = {pid}
    current = parent
    while current > 0:
        if current in seen:
            raise ValueError("process ancestry cycle detected")
        ancestors.append(current)
        seen.add(current)
        current, _ = status_fields(current)
    return {
        "pid": pid,
        "owner": str(uid_name_probe(uid)),
        "command": command,
        "ancestor_pids": tuple(ancestors),
    }


__all__ = [
    "build_ap_plan",
    "build_finalizer_evidence",
    "build_quant_plan",
    "build_scheduler_lock_probe",
    "canonical_sha256",
    "inspect_linux_process",
    "query_h800_inventory",
    "query_related_processes",
    "run_argv",
    "validate_scheduler_lease",
]
