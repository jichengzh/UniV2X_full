#!/usr/bin/env python3
"""Execute ready Stage3 AP plan jobs with GPU gating and resumable state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIV2X_PYTHON = Path("/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python")
MAX_ATTEMPTS = 3
CommandResult = namedtuple("CommandResult", "returncode stdout stderr")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.is_file():
        return []
    return [json.loads(line) for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path_from_command(command: Sequence[str]) -> Path:
    for index, token in enumerate(command):
        for option in ("--report-json", "--out-json", "--export-report-json"):
            if token == option and index + 1 < len(command):
                return Path(command[index + 1])
            if token.startswith(option + "="):
                return Path(token.split("=", 1)[1])
    raise ValueError("command must contain --report-json, --out-json, or --export-report-json")


def plan_fingerprint(job: Mapping[str, Any], stage: str) -> str:
    raw_command = job.get(f"{stage}_command")
    command = list(map(str, raw_command)) if isinstance(raw_command, list) else None
    try:
        report_path = str(report_path_from_command(command or []))
    except ValueError:
        report_path = None
    binding = {
        "runner_key": job.get("runner_key"),
        "compiled_artifact_digest": job.get("compiled_artifact_digest"),
        "compiled_artifact_path": job.get("compiled_artifact_path") or job.get("compiled_artifact"),
        "stage": stage,
        "stage_command": command,
        "report_path": report_path,
    }
    encoded = json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def bind_physical_gpu(command: Sequence[str], runner_key: str, gpu: str) -> list[str]:
    bound = list(command)
    if not (runner_key.startswith("tvm_") or "_tvm_" in runner_key):
        return bound
    if "--gpu-id" in bound:
        index = bound.index("--gpu-id")
        if index + 1 < len(bound):
            return [*bound[: index + 1], str(gpu), *bound[index + 2 :]]
    return [*bound, "--gpu-id", str(gpu)]


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _extract_ap(report: Mapping[str, Any]) -> dict[str, float]:
    nested = report.get("ap")
    sources = [report, nested] if isinstance(nested, Mapping) else [report]
    result: dict[str, float] = {}
    for key in ("ap30", "ap50", "ap70"):
        for source in sources:
            value = source.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = float(value)
                break
    return result


def _ap_fields_are_finite_numbers(report: Mapping[str, Any]) -> bool:
    nested = report.get("ap")
    sources = [report, nested] if isinstance(nested, Mapping) else [report]
    return all(
        isinstance(source[key], (int, float))
        and not isinstance(source[key], bool)
        and math.isfinite(float(source[key]))
        for source in sources
        for key in ("ap30", "ap50", "ap70")
        if key in source
    )


def validate_report(
    model: str,
    stage: str,
    report: Mapping[str, Any],
    *,
    runner_key: str | None = None,
) -> tuple[bool, str | None, dict[str, float]]:
    ap = _extract_ap(report)
    if not _all_finite(report) or not _ap_fields_are_finite_numbers(report):
        return False, "report_contains_non_finite_output", ap
    if model == "pyramid":
        if report.get("status") != "success":
            return False, "status_not_success", ap
        if int(report.get("processed_samples") or 0) < (16 if stage == "sanity" else 1789):
            return False, "processed_samples_below_stage_minimum", ap
        if int(report.get("fallback_samples") or 0) != 0:
            return False, "fallback_samples_nonzero", ap
        if int(report.get("failed_samples") or 0) != 0:
            return False, "failed_samples_nonzero", ap
        if stage == "full":
            if runner_key == "pyramid_tvm_fp16_bridge":
                if report.get("ap_measured") is not True or report.get("smoke_gate_passed") is not True:
                    return False, "tvm_fp16_full_ap_gate_not_true", ap
            elif runner_key == "pyramid_tvm_int8_numeric_gate":
                gates = report.get("gates")
                if (
                    report.get("ap_measured") is not True
                    or report.get("ap_row_allowed") is not True
                    or not isinstance(gates, Mapping)
                    or gates.get("full_1789") is not True
                ):
                    return False, "tvm_int8_full_ap_gate_not_true", ap
            elif report.get("engine_ap_claim") is not True:
                return False, "engine_ap_claim_not_true", ap
            if set(ap) != {"ap30", "ap50", "ap70"}:
                return False, "ap30_ap50_ap70_required", ap
        return True, None, ap
    if model == "codriving":
        expected_samples = 16 if stage == "sanity" else 1789
        allowed_statuses = {"success"}
        if stage == "sanity" and runner_key == "codriving_tvm_int8_numeric_gate":
            allowed_statuses.add("numerical_sanity_passed")
        if report.get("status") not in allowed_statuses:
            return False, "status_not_success", ap
        if report.get("processed_samples") != expected_samples:
            return False, "processed_samples_not_stage_count", ap
        if report.get("engine_samples") != expected_samples:
            return False, "engine_samples_not_processed", ap
        if report.get("fallback_samples") != 0:
            return False, "fallback_samples_nonzero", ap
        if report.get("failed_samples") != 0:
            return False, "failed_samples_nonzero", ap
        gate = "sanity_16" if stage == "sanity" else "full_1789"
        gates = report.get("gates")
        if not isinstance(gates, Mapping) or gates.get(gate) is not True:
            return False, f"gates.{gate}_not_true", ap
        if stage == "full" and set(ap) != {"ap30", "ap50", "ap70"}:
            return False, "ap30_ap50_ap70_required", ap
        return True, None, ap
    return False, f"unsupported_model:{model}", ap


def is_valid_numerical_feasibility_report(
    model: str,
    stage: str,
    report: Mapping[str, Any],
    *,
    runner_key: str | None = None,
) -> bool:
    if (
        model != "codriving"
        or stage != "sanity"
        or runner_key != "codriving_tvm_int8_numeric_gate"
        or report.get("status") != "numerical_feasibility_failure"
        or report.get("processed_samples") != 16
        or report.get("engine_samples") != 16
        or report.get("engine_accounting_valid") is not True
        or report.get("fallback_samples") != 0
        or report.get("failed_samples") != 0
        or report.get("ap_measured") is not False
        or report.get("gates", {}).get("sanity_16") is not False
    ):
        return False
    reasons = report.get("failure_reasons")
    outputs = report.get("numeric_outputs")
    return (
        isinstance(reasons, list)
        and bool(reasons)
        and all(isinstance(reason, str) and reason for reason in reasons)
        and isinstance(outputs, Mapping)
        and len(outputs) == 3
        and any(isinstance(value, Mapping) and value.get("passed") is False for value in outputs.values())
        and not any(key in report for key in ("ap", "ap30", "ap50", "ap70"))
    )


def _job_id(job: Mapping[str, Any]) -> str:
    value = job.get("manifest_job_id") or job.get("job_id")
    if not value:
        raise ValueError("AP plan row missing manifest_job_id/job_id")
    return str(value)


def _has_success(state_rows: Sequence[Mapping[str, Any]], job: Mapping[str, Any], stage: str) -> bool:
    job_id = _job_id(job)

    def evidence_valid(row: Mapping[str, Any]) -> bool:
        if row.get("record_type") is None:
            return True
        if row.get("record_type") != "job_terminal":
            return False
        report_path = Path(str(row.get("report_path") or ""))
        report_sha = str(row.get("report_sha256") or "")
        return report_path.is_file() and bool(report_sha) and sha256_file(report_path) == report_sha

    return any(
        str(row.get("job_id") or row.get("manifest_job_id") or "") == job_id
        and (row.get("stage") == stage or (stage == "sanity" and row.get("stage") == "full"))
        and row.get("status") == "success"
        and evidence_valid(row)
        and (
            row.get("plan_fingerprint") == plan_fingerprint(job, str(row.get("stage")))
        )
        for row in state_rows
    )


def _numerical_terminal_row(
    state_rows: Sequence[Mapping[str, Any]], job: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    job_id = _job_id(job)
    candidates = [
        row
        for row in state_rows
        if str(row.get("job_id") or row.get("manifest_job_id") or "") == job_id
        and row.get("record_type") == "job_terminal"
        and row.get("stage") == "sanity"
        and row.get("status") == "failed"
        and row.get("failure_reason") == "numerical_feasibility_failure"
        and row.get("plan_fingerprint") == plan_fingerprint(job, "sanity")
    ]
    return candidates[-1] if candidates else None


def _has_numerical_terminal(
    state_rows: Sequence[Mapping[str, Any]], job: Mapping[str, Any]
) -> bool:
    return _numerical_terminal_row(state_rows, job) is not None


def _has_full_numerical_skip(
    state_rows: Sequence[Mapping[str, Any]], job: Mapping[str, Any]
) -> bool:
    job_id = _job_id(job)
    return any(
        str(row.get("job_id") or row.get("manifest_job_id") or "") == job_id
        and row.get("record_type") == "job_terminal"
        and row.get("stage") == "full"
        and row.get("status") == "skipped_numerical_feasibility"
        and row.get("failure_reason") == "numerical_feasibility_failure"
        and row.get("plan_fingerprint") == plan_fingerprint(job, "full")
        for row in state_rows
    )


def _replace_or_append_option(command: Sequence[str], option: str, value: str) -> list[str]:
    result = list(command)
    if option in result:
        index = result.index(option)
        if index + 1 >= len(result):
            raise ValueError(f"command option missing value: {option}")
        return [*result[: index + 1], value, *result[index + 2 :]]
    return [*result, option, value]


def bind_full_command_state(job: Mapping[str, Any], state_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bound = dict(job)
    bindings = job.get("full_command_state_bindings")
    if not isinstance(bindings, Mapping):
        return bound
    command = list(job.get("full_command") or [])
    job_id = _job_id(job)
    for spec in bindings.values():
        if not isinstance(spec, Mapping):
            raise ValueError("full command state binding must be an object")
        candidates = [
            row for row in state_rows
            if str(row.get("job_id") or row.get("manifest_job_id") or "") == job_id
            and row.get("stage") == spec.get("state_stage")
            and row.get("status") == spec.get("state_status")
            and row.get("plan_fingerprint") == plan_fingerprint(job, str(spec.get("state_stage")))
        ]
        if not candidates:
            raise ValueError("required sanity success state is missing or stale")
        state = candidates[-1]
        path = Path(str(state.get(str(spec.get("path_field"))) or ""))
        digest = str(state.get(str(spec.get("sha256_field"))) or "")
        if not path.is_file() or not digest:
            raise ValueError("sanity report path/SHA is missing")
        if spec.get("verify_sha256") and sha256_file(path) != digest:
            raise ValueError("sanity report SHA256 mismatch")
        command = _replace_or_append_option(command, str(spec["command_option"]), str(path))
        sha_option = spec.get("sha256_command_option")
        if sha_option:
            command = _replace_or_append_option(command, str(sha_option), digest)
    bound["full_command"] = command
    return bound


def select_jobs(
    jobs: Sequence[Mapping[str, Any]], *, stage: str, state_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    selected: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for source in jobs:
        if source.get("ap_terminal") != "ready":
            continue
        job = dict(source)
        job_id = _job_id(job)
        if stage == "sanity":
            if _has_success(state_rows, job, stage):
                skipped[job_id] = "already_successful"
                continue
            if _has_numerical_terminal(state_rows, job):
                skipped[job_id] = "numerical_feasibility_terminal"
                continue
        if stage == "full" and _has_full_numerical_skip(state_rows, job):
            skipped[job_id] = "already_numerical_feasibility_skip"
            continue
        if stage == "full" and _has_numerical_terminal(state_rows, job):
            skipped[job_id] = "numerical_feasibility_terminal"
            continue
        if stage == "full" and job.get("full_command_state_bindings"):
            try:
                job = bind_full_command_state(job, state_rows)
            except ValueError:
                skipped[job_id] = "sanity_evidence_binding_failed"
                continue
        if _has_success(state_rows, job, stage):
            skipped[job_id] = "already_successful"
            continue
        if stage == "full" and not _has_success(state_rows, job, "sanity"):
            skipped[job_id] = "sanity_success_required"
        else:
            selected.append(job)
    return selected, skipped


def gpu_idle(gpu: str) -> tuple[bool, str]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
        text=True,
        capture_output=True,
        check=False,
    )
    if query.returncode != 0:
        return False, f"gpu_probe_failed:{query.stderr.strip() or query.returncode}"
    try:
        utilization, memory_mib = [int(part.strip()) for part in query.stdout.strip().split(",")[:2]]
    except (TypeError, ValueError):
        return False, f"gpu_probe_invalid:{query.stdout.strip()}"
    if utilization > 2:
        return False, f"gpu_busy:utilization={utilization}"
    if memory_mib > 100:
        return False, f"gpu_busy:memory_used_mib={memory_mib}"
    return True, f"gpu_idle:utilization={utilization},memory_used_mib={memory_mib}"


def first_idle_gpu(
    candidates: Sequence[str],
    *,
    probe: Callable[[str], tuple[bool, str]] = gpu_idle,
) -> tuple[str | None, str]:
    evidence: list[str] = []
    for gpu in candidates:
        idle, detail = probe(str(gpu))
        evidence.append(f"gpu{gpu}:{detail}")
        if idle:
            return str(gpu), detail
    return None, ";".join(evidence)


def wait_for_available_gpu(candidates: Sequence[str], *, poll_seconds: float) -> str:
    while True:
        gpu, _ = first_idle_gpu(candidates)
        if gpu is not None:
            return gpu
        time.sleep(poll_seconds)


def run_subprocess(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> CommandResult:
    completed = subprocess.run(command, cwd=cwd, env=dict(env), text=True, capture_output=True, check=False)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def execute_job(
    job: Mapping[str, Any],
    *,
    stage: str,
    gpu: str,
    python: Path,
    artifact_root: Path,
    append_state: Callable[[dict[str, Any]], None],
    wait_for_gpu: Callable[[str], tuple[bool, str]] = gpu_idle,
    run_command: Callable[..., CommandResult] = run_subprocess,
    busy_poll_seconds: float = 30.0,
) -> dict[str, Any]:
    job_id = _job_id(job)
    fingerprint = plan_fingerprint(job, stage)
    raw_command = job.get(f"{stage}_command")
    if not isinstance(raw_command, list) or not raw_command:
        terminal = {"record_type": "job_terminal", "job_id": job_id, "stage": stage, "status": "failed", "failure_reason": f"missing_{stage}_command", "plan_fingerprint": fingerprint, "timestamp": utc_now()}
        append_state(terminal)
        return terminal
    command = bind_physical_gpu(
        [str(python), *map(str, raw_command[1:])],
        str(job.get("runner_key") or ""),
        str(gpu),
    )
    report_path = report_path_from_command(command)
    if not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    attempt_dir = artifact_root / job_id.replace("/", "_") / stage
    attempt_dir.mkdir(parents=True, exist_ok=True)

    while True:
        idle, evidence = wait_for_gpu(str(gpu))
        if idle:
            break
        append_state({"record_type": "scheduler_wait", "job_id": job_id, "stage": stage, "status": "waiting_gpu_idle", "gpu": str(gpu), "reason": evidence, "timestamp": utc_now()})
        time.sleep(busy_poll_seconds)

    last_attempt: dict[str, Any] | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if report_path.exists():
            report_path.unlink()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        try:
            result = run_command(command, cwd=REPO_ROOT, env=env)
        except OSError as exc:
            result = CommandResult(127, "", f"{type(exc).__name__}:{exc}")
        stdout_path = attempt_dir / f"attempt_{attempt}.stdout.log"
        stderr_path = attempt_dir / f"attempt_{attempt}.stderr.log"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")
        failure_reason: str | None = None
        report_sha: str | None = None
        ap: dict[str, float] = {}
        terminal_numerical_failure = False
        if result.returncode != 0:
            failure_reason = f"command_exit_{result.returncode}"
        if report_path.is_file():
            report_sha = sha256_file(report_path)
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                valid, report_failure, ap = validate_report(
                    str(job.get("model") or ""),
                    stage,
                    report,
                    runner_key=str(job.get("runner_key") or ""),
                )
                if result.returncode == 0:
                    failure_reason = None if valid else report_failure
                elif result.returncode == 2 and is_valid_numerical_feasibility_report(
                    str(job.get("model") or ""),
                    stage,
                    report,
                    runner_key=str(job.get("runner_key") or ""),
                ):
                    failure_reason = "numerical_feasibility_failure"
                    terminal_numerical_failure = True
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                if result.returncode == 0:
                    failure_reason = f"report_invalid:{exc}"
        elif result.returncode == 0:
            failure_reason = "report_missing"
        last_attempt = {
            "record_type": "attempt",
            "job_id": job_id,
            "model": job.get("model"),
            "stage": stage,
            "attempt": attempt,
            "status": "success" if failure_reason is None else "failed",
            "command": command,
            "cuda_visible_devices": str(gpu),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "report_path": str(report_path),
            "report_sha256": report_sha,
            "ap": ap,
            "failure_reason": failure_reason,
            "plan_fingerprint": fingerprint,
            "timestamp": utc_now(),
        }
        append_state(last_attempt)
        if failure_reason is None or terminal_numerical_failure:
            break

    assert last_attempt is not None
    terminal = {
        "record_type": "job_terminal",
        "job_id": job_id,
        "model": job.get("model"),
        "stage": stage,
        "status": last_attempt["status"],
        "attempts": last_attempt["attempt"],
        "report_path": last_attempt["report_path"],
        "report_sha256": last_attempt["report_sha256"],
        "ap": last_attempt["ap"],
        "failure_reason": last_attempt["failure_reason"],
        "plan_fingerprint": fingerprint,
        "timestamp": utc_now(),
    }
    append_state(terminal)
    return terminal


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--stage", choices=("sanity", "full"), required=True)
    parser.add_argument("--state-jsonl", type=Path, required=True)
    gpu_group = parser.add_mutually_exclusive_group(required=True)
    gpu_group.add_argument("--gpu")
    gpu_group.add_argument("--gpus")
    parser.add_argument("--univ2x-python", type=Path, default=DEFAULT_UNIV2X_PYTHON)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--busy-poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = read_jsonl(args.ap_plan_jsonl)
    state_rows = read_jsonl(args.state_jsonl)
    selected, skipped = select_jobs(jobs, stage=args.stage, state_rows=state_rows)
    artifact_root = args.artifact_root or args.state_jsonl.parent / "stage3_ap_attempts"
    gpu_candidates = [str(args.gpu)] if args.gpu is not None else [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    results = []
    if args.stage == "full":
        for job in jobs:
            if skipped.get(_job_id(job)) != "numerical_feasibility_terminal":
                continue
            sanity = _numerical_terminal_row(state_rows, job)
            assert sanity is not None
            append_jsonl(args.state_jsonl, {
                "record_type": "job_terminal",
                "job_id": _job_id(job),
                "model": job.get("model"),
                "stage": "full",
                "status": "skipped_numerical_feasibility",
                "attempts": 0,
                "report_path": sanity.get("report_path"),
                "report_sha256": sanity.get("report_sha256"),
                "ap": {},
                "failure_reason": "numerical_feasibility_failure",
                "plan_fingerprint": plan_fingerprint(job, "full"),
                "timestamp": utc_now(),
            })
    for job in selected:
        selected_gpu = wait_for_available_gpu(gpu_candidates, poll_seconds=args.busy_poll_seconds)
        results.append(execute_job(
            job,
            stage=args.stage,
            gpu=selected_gpu,
            python=args.univ2x_python,
            artifact_root=artifact_root,
            append_state=lambda row: append_jsonl(args.state_jsonl, row),
            busy_poll_seconds=args.busy_poll_seconds,
        ))
    failures = sum(
        result["status"] != "success"
        and result.get("failure_reason") != "numerical_feasibility_failure"
        for result in results
    )
    print(json.dumps({"stage": args.stage, "selected": len(selected), "success": len(results) - failures, "failed": failures, "skipped": skipped}, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
