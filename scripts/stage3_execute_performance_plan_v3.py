#!/usr/bin/env python3
"""Execute stage3 performance jobs on local H800 GPUs with resumable state."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


TVM_PYTHON = Path("/exdata/jichengzhi/tvm310/bin/python")
TRT_PYTHON = Path("/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python")
STATE_SCHEMA = "stage3_execute_performance_plan_v3_state"
GPU_IDLE_MEMORY_USED_MIB = 100
GPU_IDLE_UTILIZATION_PCT = 10
DEFAULT_MAX_WORKERS = 3
REPO_ROOT = Path(__file__).resolve().parents[1]
TVM_SITE = Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages")
TVM_CUDA_RUNTIME_LIB = TVM_SITE / "nvidia" / "cuda_runtime" / "lib"
TVM_LIBRARY_DIR = TVM_SITE / "tvm" / "lib"
TVM_NVLIBS_FILE = Path("/exdata/jichengzhi/tvm_nvlibs.path")

_STATE_WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-jsonl", required=True)
    parser.add_argument("--state-jsonl", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    return parser.parse_args()


def parse_gpu_list(value: str) -> list[int]:
    gpus = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    return gpus


def load_jobs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_state_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def append_state_row(path: Path, row: dict[str, Any]) -> None:
    payload = dict(row)
    payload.setdefault("schema_version", STATE_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def job_slug(job_id: str) -> str:
    return job_id.replace("|", "__").replace("/", "_")


def _replace_flag_value(command: list[str], flag: str, value: str) -> list[str]:
    updated = list(command)
    if flag in updated:
        index = updated.index(flag)
        if index + 1 >= len(updated):
            raise ValueError(f"flag missing value: {flag}")
        updated[index + 1] = value
    else:
        updated.extend([flag, value])
    return updated


def _derive_trt_calibration_dir(job: dict[str, Any]) -> str:
    calibration_root = str(job.get("calibration_root") or "").rstrip("/")
    if not calibration_root:
        raise ValueError(f"job {job.get('job_id')} missing calibration_root")
    if str(job.get("model")) == "codriving" or "/codriving_" in calibration_root:
        parent = Path(calibration_root).parent
        return str(parent / "trt_calibration_npy")
    return calibration_root + "/trt_npy"


def prepare_job_command(job: dict[str, Any]) -> list[str]:
    command = list(job.get("command") or [])
    if not command:
        raise ValueError(f"job {job.get('job_id')} missing command")
    runner_key = str(job.get("runner_key") or "")
    if runner_key.startswith("tvm_"):
        command[0] = str(TVM_PYTHON)
    elif runner_key.startswith("trt_"):
        command[0] = str(TRT_PYTHON)
    else:
        raise ValueError(f"unsupported runner_key: {runner_key}")
    if runner_key == "trt_int8":
        command = _replace_flag_value(command, "--calib-dir", _derive_trt_calibration_dir(job))
    if job.get("runtime_gpu") is not None:
        command = _replace_flag_value(command, "--gpu", str(int(job["runtime_gpu"])))
    return command


def subprocess_env_for_job(job: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    if not str(job.get("runner_key") or "").startswith("tvm_"):
        return env
    nvlibs = TVM_NVLIBS_FILE.read_text(encoding="utf-8").strip() if TVM_NVLIBS_FILE.is_file() else ""
    components = [str(TVM_CUDA_RUNTIME_LIB), str(TVM_LIBRARY_DIR)]
    if nvlibs:
        components.append(nvlibs)
    if env.get("LD_LIBRARY_PATH"):
        components.append(env["LD_LIBRARY_PATH"])
    return {**env, "LD_LIBRARY_PATH": ":".join(components)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _extract_latency(payload: dict[str, Any]) -> float | None:
    candidates = [
        payload.get("lat_p50_ms"),
        payload.get("latency_ms"),
        (payload.get("latency") or {}).get("latency_ms_p50") if isinstance(payload.get("latency"), dict) else None,
        (payload.get("latency") or {}).get("lat_p50_ms") if isinstance(payload.get("latency"), dict) else None,
    ]
    for candidate in candidates:
        if _is_finite_number(candidate):
            return float(candidate)
    return None


def _extract_energy(payload: dict[str, Any]) -> float | None:
    candidates = [
        payload.get("energy_j"),
        (payload.get("energy") or {}).get("energy_j") if isinstance(payload.get("energy"), dict) else None,
        (payload.get("energy") or {}).get("joules") if isinstance(payload.get("energy"), dict) else None,
        (payload.get("energy") or {}).get("joules_per_inference") if isinstance(payload.get("energy"), dict) else None,
        (payload.get("energy") or {}).get("joule_per_inference") if isinstance(payload.get("energy"), dict) else None,
        (payload.get("energy") or {}).get("energy_J") if isinstance(payload.get("energy"), dict) else None,
    ]
    for candidate in candidates:
        if _is_finite_number(candidate):
            return float(candidate)
    return None


def _extract_success_and_correctness(payload: dict[str, Any]) -> tuple[bool, bool]:
    success = False
    correctness = False

    if payload.get("status") == "success" or payload.get("build_success") is True:
        success = True
    if payload.get("numerical_finite") is True:
        correctness = True
    if payload.get("correctness_all_exact") is True:
        correctness = True
    if isinstance(payload.get("correctness_vs_default_fp16"), list):
        correctness = len(payload["correctness_vs_default_fp16"]) > 0
    if isinstance(payload.get("correctness_vs_native_direct"), list):
        correctness = len(payload["correctness_vs_native_direct"]) > 0

    return success, correctness


def _candidate_result_paths(job: dict[str, Any], work_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    expected = job.get("expected_result_json")
    if expected:
        candidates.append(Path(str(expected)))
        candidates.append(work_dir / Path(str(expected)).name)
    command = list(job.get("command") or [])
    if "--out" in command:
        output_path = Path(command[command.index("--out") + 1])
        candidates.append(output_path)
        candidates.append(work_dir / output_path.name)
    if "--out-dir" in command:
        out_dir = Path(command[command.index("--out-dir") + 1])
        candidates.append(out_dir / "route_b_fp16_auto_result.json")
        candidates.append(out_dir / "route_b_int8_auto_decomp_result.json")
        if "--label" in command:
            label = command[command.index("--label") + 1]
            candidates.append(out_dir / label / "route_b_fp16_auto_result.json")
            candidates.append(out_dir / label / "route_b_int8_auto_decomp_result.json")
        candidates.append(work_dir / out_dir.name / "route_b_fp16_auto_result.json")
        candidates.append(work_dir / out_dir.name / "route_b_int8_auto_decomp_result.json")
    runner_key = str(job.get("runner_key") or "")
    if runner_key == "tvm_fp16":
        candidates.append(work_dir / "route_b_fp16_auto_result.json")
    elif runner_key == "tvm_int8":
        candidates.append(work_dir / "route_b_int8_auto_decomp_result.json")
    else:
        candidates.append(work_dir / "trt_profile_result.json")
    seen: set[str] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def validate_job_result(job: dict[str, Any], *, returncode: int, work_dir: Path) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "job_id": job.get("job_id"),
        "success": False,
        "failure_reasons": [],
        "result_json": None,
        "result_sha256": None,
    }
    if returncode != 0:
        verdict["failure_reasons"].append(f"returncode={returncode}")
        return verdict

    result_path = next((path for path in _candidate_result_paths(job, work_dir) if path.is_file()), None)
    if result_path is None:
        for candidate in sorted(work_dir.rglob("*.json")):
            if candidate.name in {
                "route_b_fp16_auto_result.json",
                "route_b_int8_auto_decomp_result.json",
                "trt_profile_result.json",
            }:
                result_path = candidate
                break
    if result_path is None:
        verdict["failure_reasons"].append("result_json_missing")
        return verdict

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    latency = _extract_latency(payload)
    energy = _extract_energy(payload)
    success, correctness = _extract_success_and_correctness(payload)
    if latency is None:
        verdict["failure_reasons"].append("latency_missing_or_nonfinite")
    if energy is None:
        verdict["failure_reasons"].append("energy_missing_or_nonfinite")
    if not success:
        verdict["failure_reasons"].append("success_flag_missing")
    if not correctness:
        verdict["failure_reasons"].append("correctness_flag_missing")
    if verdict["failure_reasons"]:
        verdict["result_json"] = str(result_path)
        verdict["result_sha256"] = _sha256_file(result_path)
        return verdict

    verdict.update(
        {
            "success": True,
            "latency_ms": latency,
            "energy_j": energy,
            "result_json": str(result_path),
            "result_sha256": _sha256_file(result_path),
        }
    )
    return verdict


def aggregate_state_rows(jobs: list[dict[str, Any]], state_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_job: dict[str, dict[str, Any]] = {}
    indexed_rows: dict[str, list[dict[str, Any]]] = {}
    for row in state_rows:
        indexed_rows.setdefault(str(row.get("job_id")), []).append(row)

    ready_job_ids: list[str] = []
    success_job_ids: list[str] = []
    confirmed_failure_job_ids: list[str] = []
    for job in jobs:
        job_id = str(job["job_id"])
        max_attempts = int(job.get("max_attempts", 2))
        rows = sorted(indexed_rows.get(job_id, []), key=lambda item: (int(item.get("attempt", 0)), str(item.get("status", ""))))
        status = "pending"
        if any(row.get("status") == "success" for row in rows):
            status = "success"
        elif any(row.get("status") == "confirmed_failure" for row in rows):
            status = "confirmed_failure"
        else:
            failed_attempts = {int(row.get("attempt", 0)) for row in rows if row.get("status") == "failed"}
            if len(failed_attempts) >= max_attempts:
                status = "confirmed_failure"
        failed_attempts = [row for row in rows if row.get("status") == "failed"]
        success_attempts = [row for row in rows if row.get("status") == "success"]
        attempts_used = len({int(row.get("attempt", 0)) for row in rows if row.get("status") in {"failed", "success"}})
        by_job[job_id] = {
            "job_id": job_id,
            "latest_status": rows[-1]["status"] if rows else "pending",
            "terminal_status": status,
            "attempts_used": attempts_used,
            "failed_attempts": failed_attempts,
            "success_attempts": success_attempts,
        }
        if status == "pending":
            ready_job_ids.append(job_id)
        elif status == "success":
            success_job_ids.append(job_id)
        else:
            confirmed_failure_job_ids.append(job_id)
    return {
        "schema_version": STATE_SCHEMA,
        "job_count": len(jobs),
        "by_job": by_job,
        "ready_job_ids": ready_job_ids,
        "success_job_ids": success_job_ids,
        "confirmed_failure_job_ids": confirmed_failure_job_ids,
    }


def _query_gpu_idle_state(gpus: list[int]) -> dict[int, dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"nvidia-smi failed: rc={proc.returncode}")
    rows: dict[int, dict[str, float]] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        gpu = int(parts[0])
        rows[gpu] = {"memory_used_mib": float(parts[1]), "utilization_pct": float(parts[2])}
    return {gpu: rows[gpu] for gpu in gpus if gpu in rows}


def preflight_gpu_check(gpus: list[int]) -> None:
    states = _query_gpu_idle_state(gpus)
    missing = [gpu for gpu in gpus if gpu not in states]
    if missing:
        raise RuntimeError(f"missing_gpu_rows:{missing}")
    busy = [
        gpu
        for gpu, state in states.items()
        if state["memory_used_mib"] > GPU_IDLE_MEMORY_USED_MIB or state["utilization_pct"] > GPU_IDLE_UTILIZATION_PCT
    ]
    if busy:
        raise RuntimeError(f"gpus_not_idle:{busy}")


def _attempt_dir(attempt_root: Path, job: dict[str, Any], attempt: int) -> Path:
    return attempt_root / job_slug(str(job["job_id"])) / f"attempt_{attempt:02d}"


def run_single_job(*, job: dict[str, Any], state_jsonl: Path, attempt_root: Path) -> dict[str, Any]:
    command = prepare_job_command(job)
    max_attempts = int(job.get("max_attempts", 2))
    last_failure: dict[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        work_dir = _attempt_dir(attempt_root, job, attempt)
        work_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = work_dir / "stdout.txt"
        stderr_path = work_dir / "stderr.txt"
        started = time.time()
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=subprocess_env_for_job(job),
            capture_output=True,
            text=True,
            check=False,
        )
        ended = time.time()
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        verdict = validate_job_result(job, returncode=proc.returncode, work_dir=work_dir)
        row = {
            "job_id": job["job_id"],
            "attempt": attempt,
            "status": "success" if verdict["success"] else "failed",
            "returncode": proc.returncode,
            "start_time_unix": started,
            "end_time_unix": ended,
            "elapsed_s": round(ended - started, 6),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "result_json": verdict.get("result_json"),
            "result_sha256": verdict.get("result_sha256"),
            "failure_reasons": verdict.get("failure_reasons", []),
        }
        append_state_row(state_jsonl, row)
        if verdict["success"]:
            return row
        last_failure = row

    assert last_failure is not None
    final_row = {
        "job_id": job["job_id"],
        "attempt": last_failure["attempt"],
        "status": "confirmed_failure",
        "returncode": last_failure["returncode"],
        "start_time_unix": last_failure["start_time_unix"],
        "end_time_unix": last_failure["end_time_unix"],
        "elapsed_s": last_failure["elapsed_s"],
        "stdout_path": last_failure["stdout_path"],
        "stderr_path": last_failure["stderr_path"],
        "result_json": last_failure["result_json"],
        "result_sha256": last_failure["result_sha256"],
        "failure_reasons": last_failure["failure_reasons"],
    }
    append_state_row(state_jsonl, final_row)
    return final_row


def _job_by_id(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(job["job_id"]): job for job in jobs}


def execute_jobs(
    *,
    jobs: list[dict[str, Any]],
    state_jsonl: Path,
    gpus: list[int],
    max_workers: int,
    attempt_root: Path,
) -> list[dict[str, Any]]:
    preflight_gpu_check(gpus)
    state = aggregate_state_rows(jobs, load_state_rows(state_jsonl))
    jobs_by_id = {
        str(job["job_id"]): {**job, "runtime_gpu": int(gpus[index % len(gpus)])}
        for index, job in enumerate(jobs)
    }
    gpu_locks = {gpu: threading.Lock() for gpu in gpus}
    results: list[dict[str, Any]] = []

    def worker(job_id: str) -> dict[str, Any]:
        job = jobs_by_id[job_id]
        gpu = int(job["runtime_gpu"])
        if gpu not in gpu_locks:
            raise RuntimeError(f"job {job_id} assigned_gpu={gpu} not in gpu pool {gpus}")
        with gpu_locks[gpu]:
            preflight_gpu_check([gpu])
            return run_single_job(job=job, state_jsonl=state_jsonl, attempt_root=attempt_root)

    ready_ids = list(state["ready_job_ids"])
    if not ready_ids:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(ready_ids))) as executor:
        futures = [executor.submit(worker, job_id) for job_id in ready_ids]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def main() -> int:
    args = parse_args()
    jobs_path = Path(args.jobs_jsonl).resolve()
    state_path = Path(args.state_jsonl).resolve()
    jobs = load_jobs(jobs_path)
    gpus = parse_gpu_list(args.gpus)
    attempt_root = state_path.parent / "attempts"
    results = execute_jobs(
        jobs=jobs,
        state_jsonl=state_path,
        gpus=gpus,
        max_workers=int(args.max_workers),
        attempt_root=attempt_root,
    )
    summary = aggregate_state_rows(jobs, load_state_rows(state_path))
    print(
        json.dumps(
            {
                "state_jsonl": str(state_path),
                "attempt_root": str(attempt_root),
                "executed_jobs": len(results),
                "success_job_ids": summary["success_job_ids"],
                "confirmed_failure_job_ids": summary["confirmed_failure_job_ids"],
                "ready_job_ids": summary["ready_job_ids"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not summary["ready_job_ids"] else 1 if summary["confirmed_failure_job_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
