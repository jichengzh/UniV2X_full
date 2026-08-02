#!/usr/bin/env python3
"""Advance formal F-Cooper TVM GEAR rounds after atomic four-row closure."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


TERMINAL_STATUS_FIELDS = (
    "pending",
    "running",
    "failed_infrastructure",
    "failed_terminal",
    "blocked_dependency",
)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_manifest_sha(manifest: Mapping[str, Any]) -> None:
    payload = copy.deepcopy(dict(manifest))
    recorded = payload.pop("manifest_sha256", None)
    if recorded != canonical_sha256(payload):
        raise ValueError("scheduler manifest SHA drift")


def build_next_manifest(
    template: Mapping[str, Any],
    *,
    round_index: int,
    request_path: Path,
) -> dict[str, Any]:
    validate_manifest_sha(template)
    if len(template.get("jobs") or []) != 4:
        raise ValueError("GEAR scheduler template must contain four jobs")
    if any(
        job.get("request_kind") != "t16"
        or int(job.get("max_trials", -1)) != 64
        or not job.get("runner_python")
        or not job.get("runner_script")
        or not isinstance(job.get("runner_common_args"), Mapping)
        for job in template["jobs"]
    ):
        raise ValueError("GEAR scheduler template is not a valid 64-trial t16 contract")
    barrier_id = f"round_{round_index:02d}"
    jobs = []
    for row_index, source in enumerate(template["jobs"]):
        job = copy.deepcopy(dict(source))
        job.update(
            {
                "job_id": f"gear-r{round_index:02d}-{row_index}",
                "request_json": str(request_path.resolve()),
                "row_index": row_index,
                "barrier_id": barrier_id,
            }
        )
        jobs.append(job)
    result = {
        "schema_version": template.get("schema_version"),
        "barrier_order": [barrier_id],
        "gpu_pool": copy.deepcopy(template.get("gpu_pool")),
        "jobs": jobs,
    }
    if (
        result["schema_version"] != "fcooper_tvm_gpu_job_manifest_v1"
        or result["gpu_pool"] != list(range(8))
    ):
        raise ValueError("scheduler template contract drift")
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def write_manifest_idempotently(path: Path, manifest: Mapping[str, Any]) -> None:
    validate_manifest_sha(manifest)
    if path.exists():
        existing = read_json(path)
        validate_manifest_sha(existing)
        if existing != manifest:
            raise ValueError(f"scheduler manifest drift: {path}")
        return
    write_json_atomic(path, manifest)


def scheduler_audit_state(audit: Mapping[str, Any]) -> str:
    if audit.get("schema_version") != "fcooper_tvm_gpu_scheduler_audit_v1":
        raise ValueError("unexpected scheduler audit schema")
    counts = audit.get("status_counts")
    if not isinstance(counts, Mapping):
        raise ValueError("scheduler audit status counts missing")
    if int(counts.get("pending", -1)) > 0 or int(counts.get("running", -1)) > 0:
        return "running"
    if int(counts.get("succeeded", -1)) == 4 and all(
        int(counts.get(field, -1)) == 0 for field in TERMINAL_STATUS_FIELDS
    ):
        return "succeeded"
    return "failed"


def validate_scheduler_audit(audit: Mapping[str, Any]) -> None:
    if scheduler_audit_state(audit) != "succeeded":
        raise ValueError("GEAR round did not close with exactly four successful jobs")


def transition_outputs_complete(paths: Sequence[Path]) -> bool:
    existence = [path.is_file() for path in paths]
    if any(existence) and not all(existence):
        raise ValueError("partial round transition outputs require manual audit")
    return all(existence)


def validate_advanced_round(
    round_dir: Path,
    *,
    round_index: int,
    task_id: str,
) -> Path:
    required = [
        round_dir / "measurement_request.json",
        round_dir / "round_state.json",
        round_dir / "acquisition.json",
        round_dir / "candidate_manifest.json",
        round_dir / "predicted_candidates.json",
    ]
    existence = [path.is_file() for path in required]
    if any(existence) and not all(existence):
        raise ValueError(f"partial next-round output requires manual audit: {round_dir}")
    if not all(existence):
        raise FileNotFoundError(f"advanced round is absent: {round_dir}")
    request = read_json(required[0])
    state = read_json(required[1])
    request_without_sha = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request_sha = canonical_sha256(request_without_sha)
    rows = request.get("rows")
    row_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in (rows if isinstance(rows, list) else [])
    ]
    if (
        request.get("schema_version") != "stage5_measurement_request_v2"
        or request.get("task_id") != task_id
        or int(request.get("round_index", -1)) != round_index
        or int(request.get("batch_size", -1)) != 4
        or len(row_ids) != 4
        or len(set(row_ids)) != 4
        or request.get("measurement_request_sha256") != request_sha
    ):
        raise ValueError("advanced measurement request contract drift")
    if (
        state.get("schema_version") != "stage5_fcooper_formal_round_state_v2"
        or state.get("task_id") != task_id
        or int(state.get("round_index", -1)) != round_index
        or state.get("status") != "awaiting_recovered_source_measurement"
        or list(state.get("selected_row_ids") or []) != row_ids
        or state.get("measurement_request_sha256") != request_sha
    ):
        raise ValueError("advanced round state/request binding drift")
    return required[0]


def run_checked(command: Sequence[str], *, code_root: Path) -> None:
    environment = {**os.environ, "PYTHONPATH": str(code_root.resolve())}
    subprocess.run(
        list(command),
        cwd=code_root,
        env=environment,
        check=True,
    )


def wait_for_scheduler_audit(
    path: Path,
    *,
    poll_seconds: int,
    deadline_monotonic: float,
) -> dict[str, Any]:
    while True:
        if time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"scheduler audit deadline exceeded: {path}")
        if not path.is_file():
            time.sleep(poll_seconds)
            continue
        try:
            audit = read_json(path)
        except json.JSONDecodeError:
            time.sleep(poll_seconds)
            continue
        state = scheduler_audit_state(audit)
        if state == "succeeded":
            return audit
        if state == "failed":
            validate_scheduler_audit(audit)
        time.sleep(poll_seconds)


def finalize_round(args: argparse.Namespace, round_index: int) -> None:
    round_dir = args.search_root / f"round_{round_index:02d}"
    outputs = [
        round_dir / "actual_feedback.json",
        round_dir / "atomic_batch_audit.json",
        round_dir / f"feedback_history_through_round_{round_index:02d}.json",
    ]
    if transition_outputs_complete(outputs):
        return
    history_input = (
        args.search_root
        / f"round_{round_index - 1:02d}"
        / f"feedback_history_through_round_{round_index - 1:02d}.json"
    )
    if not history_input.is_file():
        raise FileNotFoundError(f"previous feedback history missing: {history_input}")
    run_checked(
        [
            str(args.python),
            str(args.code_root / "scripts/fcooper_finalize_formal_round_v2.py"),
            "--request-json",
            str(round_dir / "measurement_request.json"),
            "--artifact-root",
            str(args.artifact_root),
            "--round-feedback-json",
            str(outputs[0]),
            "--atomic-audit-json",
            str(outputs[1]),
            "--history-input-json",
            str(history_input),
            "--history-output-json",
            str(outputs[2]),
            "--failure-evidence-json",
            str(round_dir / "finalization_failure.json"),
            "--round-index",
            str(round_index),
            "--task-id",
            args.task_id,
        ],
        code_root=args.code_root,
    )
    if not transition_outputs_complete(outputs):
        raise RuntimeError(f"round {round_index} finalization did not close atomically")


def advance_round(args: argparse.Namespace, completed_round: int) -> Path:
    next_round = completed_round + 1
    current_dir = args.search_root / f"round_{completed_round:02d}"
    next_dir = args.search_root / f"round_{next_round:02d}"
    request_path = next_dir / "measurement_request.json"
    if next_dir.exists() and any(next_dir.iterdir()):
        return validate_advanced_round(
            next_dir,
            round_index=next_round,
            task_id=args.task_id,
        )
    run_checked(
        [
            str(args.python),
            str(args.code_root / "scripts/stage5_advance_fcooper_round_v2.py"),
            "--feedback-json",
            str(current_dir / f"feedback_history_through_round_{completed_round:02d}.json"),
            "--atomic-audit-json",
            str(current_dir / "atomic_batch_audit.json"),
            "--output-dir",
            str(args.search_root),
            "--round-index",
            str(next_round),
            "--source-registry-json",
            str(args.source_registry),
            "--probe-isolation-audit-json",
            str(args.probe_isolation_audit),
            "--coldstart-rows-json",
            str(args.coldstart_rows),
            "--coldstart-graph-features-json",
            str(args.coldstart_graph_features),
            "--profiles-json",
            str(args.profiles),
            "--task-id",
            args.task_id,
            "--dispatch-key",
            args.dispatch_key,
            "--capability-profile-id",
            args.capability_profile_id,
        ],
        code_root=args.code_root,
    )
    return validate_advanced_round(
        next_dir,
        round_index=next_round,
        task_id=args.task_id,
    )


def process_matches_scheduler(pid: int, *, manifest_path: Path) -> bool:
    command_path = Path(f"/proc/{pid}/cmdline")
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return (
        "fcooper_tvm_gpu_scheduler_v1.py" in command
        and str(manifest_path.resolve()) in command
    )


def find_running_scheduler(
    manifest_path: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> int | None:
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        command_path = process_dir / "cmdline"
        try:
            command = command_path.read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if (
            "fcooper_tvm_gpu_scheduler_v1.py" in command
            and str(manifest_path.resolve()) in command
        ):
            return pid
    return None


def run_scheduler(
    args: argparse.Namespace,
    *,
    round_index: int,
    request_path: Path,
    template_manifest: Mapping[str, Any],
) -> None:
    scheduler_dir = args.scheduler_root / f"round_{round_index:02d}_formal_v1"
    manifest_path = scheduler_dir / "manifest.json"
    manifest = build_next_manifest(
        template_manifest,
        round_index=round_index,
        request_path=request_path,
    )
    write_manifest_idempotently(manifest_path, manifest)
    audit_path = scheduler_dir / "audit.json"
    if audit_path.is_file():
        state = scheduler_audit_state(read_json(audit_path))
        if state == "succeeded":
            return
        if state == "failed":
            validate_scheduler_audit(read_json(audit_path))
    pid_path = scheduler_dir / "scheduler_process.json"
    discovered_pid = find_running_scheduler(manifest_path)
    if discovered_pid is not None:
        write_json_atomic(
            pid_path,
            {
                "schema_version": "fcooper_tvm_scheduler_process_v1",
                "pid": discovered_pid,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "discovered_after_watcher_restart": True,
                "started_wall_time": None,
            },
        )
        wait_for_scheduler_audit(
            audit_path,
            poll_seconds=args.scheduler_poll_seconds,
            deadline_monotonic=args.deadline_monotonic,
        )
        return
    if pid_path.is_file():
        process_record = read_json(pid_path)
        pid = int(process_record.get("pid", -1))
        if (
            process_record.get("manifest_sha256") == manifest["manifest_sha256"]
            and process_matches_scheduler(pid, manifest_path=manifest_path)
        ):
            wait_for_scheduler_audit(
                audit_path,
                poll_seconds=args.scheduler_poll_seconds,
                deadline_monotonic=args.deadline_monotonic,
            )
            return
    command = [
            str(args.python),
            str(args.code_root / "scripts/fcooper_tvm_gpu_scheduler_v1.py"),
            "--manifest",
            str(manifest_path),
            "--state",
            str(scheduler_dir / "state.json"),
            "--events",
            str(scheduler_dir / "events.jsonl"),
            "--audit",
            str(audit_path),
            "--poll-interval-seconds",
            str(args.scheduler_poll_seconds),
        ]
    environment = {**os.environ, "PYTHONPATH": str(args.code_root.resolve())}
    process = subprocess.Popen(
        command,
        cwd=args.code_root,
        env=environment,
    )
    write_json_atomic(
        pid_path,
        {
            "schema_version": "fcooper_tvm_scheduler_process_v1",
            "pid": process.pid,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest["manifest_sha256"],
            "started_wall_time": time.time(),
        },
    )
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    validate_scheduler_audit(read_json(audit_path))


def acquire_single_flight_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another GEAR round watcher holds the lock: {path}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--scheduler-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--template-manifest", type=Path, required=True)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--probe-isolation-audit", type=Path, required=True)
    parser.add_argument("--coldstart-rows", type=Path, required=True)
    parser.add_argument("--coldstart-graph-features", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--task-id", default="S5-FCO-TVM-V1")
    parser.add_argument("--dispatch-key", default="tvm_auto")
    parser.add_argument(
        "--capability-profile-id",
        default="h800-tvm-fcooper-probe-conditioned-v1",
    )
    parser.add_argument("--start-round", type=int, default=2)
    parser.add_argument("--final-round", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--scheduler-poll-seconds", type=int, default=30)
    parser.add_argument("--deadline-seconds", type=int, default=172800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_round < 1 or args.final_round < args.start_round:
        raise ValueError("invalid round interval")
    lock_handle = acquire_single_flight_lock(
        args.scheduler_root / "gear_round_transition_v1.lock"
    )
    try:
        template_manifest = read_json(args.template_manifest)
        deadline = time.monotonic() + args.deadline_seconds
        args.deadline_monotonic = deadline
        for round_index in range(args.start_round, args.final_round + 1):
            scheduler_audit = (
                args.scheduler_root
                / f"round_{round_index:02d}_formal_v1"
                / "audit.json"
            )
            wait_for_scheduler_audit(
                scheduler_audit,
                poll_seconds=args.poll_seconds,
                deadline_monotonic=deadline,
            )
            finalize_round(args, round_index)
            if round_index == args.final_round:
                break
            request_path = advance_round(args, round_index)
            run_scheduler(
                args,
                round_index=round_index + 1,
                request_path=request_path,
                template_manifest=template_manifest,
            )
    finally:
        lock_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
