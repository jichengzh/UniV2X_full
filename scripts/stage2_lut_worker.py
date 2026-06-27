#!/usr/bin/env python3
"""Run Stage2 LUT jobs from a resumable JSONL queue."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    append_jsonl,
    build_tvm_runtime_env,
    failure_requires_quarantine,
    is_job_quarantined,
    latest_job_status,
    next_queued_jobs,
    quarantine_row,
    read_jsonl,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-plan", required=True)
    parser.add_argument("--job-state", required=True)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-dir")
    parser.add_argument("--quarantine-db")
    parser.add_argument("--require-gpu-idle", action="store_true")
    parser.add_argument("--no-tvm-runtime-env", action="store_true")
    return parser.parse_args()


def _worker_id() -> str:
    return f"{socket.gethostname()}:pid{os.getpid()}"


def _next_attempt(state_rows: list[dict[str, Any]], job_id: str) -> int:
    attempts = [
        int(row.get("attempt", 0))
        for row in state_rows
        if row.get("job_id") == job_id
    ]
    return (max(attempts) if attempts else 0) + 1


def _write_state(
    path: Path,
    *,
    job_id: str,
    status: str,
    attempt: int,
    started_at: str | None = None,
    finished_at: str | None = None,
    log_path: str | None = None,
    failure_reason: str | None = None,
    returncode: int | None = None,
    preflight_path: str | None = None,
) -> None:
    append_jsonl(
        path,
        {
            "schema": "lut_job_state_row_v1",
            "job_id": job_id,
            "status": status,
            "attempt": attempt,
            "worker_id": _worker_id(),
            "started_at": started_at,
            "finished_at": finished_at,
            "log_path": log_path,
            "output_row_id": None,
            "failure_reason": failure_reason,
            "returncode": returncode,
            "preflight_path": preflight_path,
        },
    )


def _job_gpu(job: dict[str, Any]) -> str | None:
    resource = job.get("resource", {})
    if not isinstance(resource, dict):
        return None
    gpu = resource.get("gpu")
    if gpu is None:
        return None
    value = str(gpu)
    return value if value.isdigit() else None


def _job_requires_gpu_idle(job: dict[str, Any]) -> bool:
    resource = job.get("resource", {})
    if not isinstance(resource, dict):
        return False
    if not resource.get("exclusive"):
        return False
    if job.get("lut_kind") not in {"latency", "energy"}:
        return False
    return _job_gpu(job) is not None


def _parse_gpu_query(stdout: str, target_gpu: str) -> tuple[bool, str]:
    def first_number(value: str) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match is None:
            raise ValueError(value)
        return float(match.group(0))

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("index"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts or parts[0] != target_gpu:
            continue
        if len(parts) < 4:
            return False, "gpu_query_malformed"
        try:
            util_pct = int(first_number(parts[2]))
            memory_mib = int(first_number(parts[3]))
        except ValueError:
            return False, "gpu_query_malformed"
        if util_pct > 5:
            return False, f"gpu_not_idle:utilization={util_pct}%"
        if memory_mib > 1024:
            return False, f"gpu_not_idle:memory_used={memory_mib}MiB"
        return True, "gpu_idle"
    return False, f"gpu_{target_gpu}_not_found"


def _parse_pmon(stdout: str, target_gpu: str) -> tuple[bool, str]:
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0] != target_gpu:
            continue
        pid = parts[1]
        if pid != "-" and pid.isdigit():
            return False, f"gpu_not_idle:pmon_pid={pid}"
    return True, "pmon_idle"


def _gpu_preflight(
    job: dict[str, Any],
    *,
    log_dir: Path,
) -> tuple[bool, str | None, str]:
    target_gpu = _job_gpu(job)
    if target_gpu is None:
        return True, None, "gpu_preflight_not_applicable"
    safe_job_id = str(job["job_id"]).replace("/", "_").replace(":", "_")
    preflight_path = log_dir / f"{safe_job_id}_preflight.json"
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate",
        "--format=csv",
    ]
    pmon_cmd = ["nvidia-smi", "pmon", "-c", "1"]
    try:
        query = subprocess.run(query_cmd, capture_output=True, text=True, check=False)
        pmon = subprocess.run(pmon_cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        payload = {
            "schema": "lut_gpu_preflight_v1",
            "job_id": job["job_id"],
            "target_gpu": target_gpu,
            "query_command": query_cmd,
            "pmon_command": pmon_cmd,
            "ok": False,
            "failure_reason": f"gpu_preflight_unavailable:{exc}",
        }
        preflight_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return False, str(preflight_path), str(payload["failure_reason"])

    query_ok, query_reason = _parse_gpu_query(query.stdout, target_gpu)
    pmon_ok, pmon_reason = _parse_pmon(pmon.stdout, target_gpu)
    ok = query.returncode == 0 and pmon.returncode == 0 and query_ok and pmon_ok
    failure_reason = None
    if query.returncode != 0:
        failure_reason = query.stderr.strip() or f"gpu_query_failed:{query.returncode}"
    elif pmon.returncode != 0:
        failure_reason = pmon.stderr.strip() or f"gpu_pmon_failed:{pmon.returncode}"
    elif not query_ok:
        failure_reason = query_reason
    elif not pmon_ok:
        failure_reason = pmon_reason

    payload = {
        "schema": "lut_gpu_preflight_v1",
        "job_id": job["job_id"],
        "target_gpu": target_gpu,
        "query_command": query_cmd,
        "pmon_command": pmon_cmd,
        "query_returncode": query.returncode,
        "pmon_returncode": pmon.returncode,
        "query_stdout": query.stdout,
        "query_stderr": query.stderr,
        "pmon_stdout": pmon.stdout,
        "pmon_stderr": pmon.stderr,
        "ok": ok,
        "failure_reason": failure_reason,
    }
    preflight_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return ok, str(preflight_path), failure_reason or "gpu_idle"


def _run_job(
    job: dict[str, Any],
    *,
    state_path: Path,
    state_rows: list[dict[str, Any]],
    log_dir: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    job_id = str(job["job_id"])
    attempt = _next_attempt(state_rows, job_id)
    started_at = utc_timestamp()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id.replace('/', '_').replace(':', '_')}_attempt{attempt}.log"
    _write_state(
        state_path,
        job_id=job_id,
        status="running",
        attempt=attempt,
        started_at=started_at,
        log_path=str(log_path),
    )

    proc = subprocess.run(
        list(job["command"]),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=int(job["timeout_s"]),
        env=env,
        check=False,
    )
    log_path.write_text(
        json.dumps(
            {
                "schema": "lut_job_log_v1",
                "job_id": job_id,
                "attempt": attempt,
                "command": job["command"],
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = "succeeded" if proc.returncode == 0 else "failed"
    failure_reason = None if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())
    _write_state(
        state_path,
        job_id=job_id,
        status=status,
        attempt=attempt,
        started_at=started_at,
        finished_at=utc_timestamp(),
        log_path=str(log_path),
        failure_reason=failure_reason,
        returncode=proc.returncode,
    )
    return {"job_id": job_id, "status": status, "returncode": proc.returncode}


def _append_quarantine_if_needed(
    *,
    quarantine_path: Path | None,
    quarantine_rows: list[dict[str, Any]],
    job: dict[str, Any],
    failure_reason: str | None,
) -> list[dict[str, Any]]:
    if quarantine_path is None or not failure_requires_quarantine(failure_reason):
        return quarantine_rows
    if is_job_quarantined(quarantine_rows, job):
        return quarantine_rows
    append_jsonl(
        quarantine_path,
        quarantine_row(
            job_id=str(job["job_id"]),
            config_id=str(job["config_id"]),
            model=str(job["model"]),
            lut_kind=str(job["lut_kind"]),
            job_type=str(job["job_type"]),
            failure_reason=str(failure_reason),
        ),
    )
    return read_jsonl(quarantine_path)


def _energy_salvage_config(job: dict[str, Any]) -> dict[str, Any] | None:
    if job.get("lut_kind") != "energy":
        return None
    resource = job.get("resource", {})
    if not isinstance(resource, dict):
        return None
    config = resource.get("energy_salvage")
    return config if isinstance(config, dict) else None


def _try_energy_salvage(
    job: dict[str, Any],
    *,
    state_path: Path,
    log_dir: Path,
    env: dict[str, str],
) -> dict[str, Any] | None:
    config = _energy_salvage_config(job)
    if config is None:
        return None
    payload_root = config.get("payload_root")
    latency_rows = config.get("latency_rows")
    out_jsonl = config.get("out_jsonl") or job.get("expected_output")
    if not payload_root or not latency_rows or not out_jsonl:
        return None
    if isinstance(latency_rows, str):
        latency_rows = [latency_rows]
    if not isinstance(latency_rows, list):
        return None

    command = [
        sys.executable,
        "scripts/stage2_salvage_energy_payloads.py",
        "--payload-root",
        str(payload_root),
        "--out-jsonl",
        str(out_jsonl),
        "--job-state",
        str(state_path),
    ]
    for path in latency_rows:
        command.extend(["--latency-rows", str(path)])
    existing_rows = config.get("existing_energy_rows", [])
    if isinstance(existing_rows, str):
        existing_rows = [existing_rows]
    if isinstance(existing_rows, list):
        for path in existing_rows:
            command.extend(["--existing-energy-rows", str(path)])

    safe_job_id = str(job["job_id"]).replace("/", "_").replace(":", "_")
    log_path = log_dir / f"{safe_job_id}_energy_salvage.log"
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    log_path.write_text(
        json.dumps(
            {
                "schema": "lut_energy_salvage_log_v1",
                "job_id": job["job_id"],
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return None
    try:
        summary = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return None
    if int(summary.get("salvaged_rows", 0)) < 1:
        return None
    return {
        "job_id": str(job["job_id"]),
        "status": "succeeded",
        "returncode": 0,
        "salvaged_rows": int(summary["salvaged_rows"]),
    }


def _max_attempts_reached(job: dict[str, Any], state_rows: list[dict[str, Any]]) -> bool:
    job_id = str(job["job_id"])
    attempts = [
        int(row.get("attempt", 0))
        for row in state_rows
        if row.get("job_id") == job_id and row.get("status") in {"failed", "running"}
    ]
    return bool(attempts) and max(attempts) >= int(job["max_attempts"])


def main() -> int:
    args = parse_args()
    plan_rows = read_jsonl(args.job_plan)
    state_path = Path(args.job_state)
    state_rows = read_jsonl(state_path) if args.resume else []
    log_dir = Path(args.log_dir) if args.log_dir else state_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    quarantine_path = Path(args.quarantine_db) if args.quarantine_db else None
    quarantine_rows = read_jsonl(quarantine_path) if quarantine_path else []
    job_env = (
        dict(os.environ)
        if args.no_tvm_runtime_env
        else build_tvm_runtime_env(dict(os.environ))
    )
    deadline = None
    if args.max_hours is not None:
        deadline = time.monotonic() + float(args.max_hours) * 3600.0

    ran = []
    for job in next_queued_jobs(plan_rows, state_rows):
        if args.max_jobs is not None and len(ran) >= args.max_jobs:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        if latest_job_status(state_rows, str(job["job_id"])) in {"succeeded", "skipped"}:
            continue
        if quarantine_path is not None and is_job_quarantined(quarantine_rows, job):
            _write_state(
                state_path,
                job_id=str(job["job_id"]),
                status="skipped",
                attempt=_next_attempt(state_rows, str(job["job_id"])),
                failure_reason="quarantined_bad_db",
            )
            ran.append({"job_id": str(job["job_id"]), "status": "skipped", "returncode": None})
            state_rows = read_jsonl(state_path)
            continue
        if _max_attempts_reached(job, state_rows):
            _write_state(
                state_path,
                job_id=str(job["job_id"]),
                status="skipped",
                attempt=_next_attempt(state_rows, str(job["job_id"])),
                failure_reason="max_attempts_reached",
            )
            state_rows = read_jsonl(state_path)
            continue
        if args.require_gpu_idle and _job_requires_gpu_idle(job):
            ok, preflight_path, failure_reason = _gpu_preflight(job, log_dir=log_dir)
            if not ok:
                _write_state(
                    state_path,
                    job_id=str(job["job_id"]),
                    status="preflight_blocked",
                    attempt=0,
                    finished_at=utc_timestamp(),
                    failure_reason=failure_reason,
                    preflight_path=preflight_path,
                )
                ran.append(
                    {
                        "job_id": str(job["job_id"]),
                        "status": "preflight_blocked",
                        "returncode": None,
                    }
                )
                state_rows = read_jsonl(state_path)
                continue
        try:
            result = _run_job(
                job,
                state_path=state_path,
                state_rows=state_rows,
                log_dir=log_dir,
                env=job_env,
            )
            if result["status"] == "failed":
                salvaged_result = _try_energy_salvage(
                    job,
                    state_path=state_path,
                    log_dir=log_dir,
                    env=job_env,
                )
                if salvaged_result is not None:
                    result = salvaged_result
                    state_rows = read_jsonl(state_path)
                    ran.append(result)
                    continue
                failed_rows = read_jsonl(state_path)
                latest_failure = next(
                    (
                        row
                        for row in reversed(failed_rows)
                        if row.get("job_id") == job.get("job_id")
                        and row.get("status") == "failed"
                    ),
                    {},
                )
                quarantine_rows = _append_quarantine_if_needed(
                    quarantine_path=quarantine_path,
                    quarantine_rows=quarantine_rows,
                    job=job,
                    failure_reason=latest_failure.get("failure_reason"),
                )
        except subprocess.TimeoutExpired as exc:
            attempt = _next_attempt(state_rows, str(job["job_id"]))
            _write_state(
                state_path,
                job_id=str(job["job_id"]),
                status="failed",
                attempt=attempt,
                finished_at=utc_timestamp(),
                failure_reason=f"timeout_after_{job['timeout_s']}s:{exc}",
            )
            result = {"job_id": str(job["job_id"]), "status": "failed", "returncode": None}
        ran.append(result)
        state_rows = read_jsonl(state_path)

    print(
        json.dumps(
            {
                "schema": "lut_worker_summary_v1",
                "jobs_run": len(ran),
                "results": ran,
                "state": str(state_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
