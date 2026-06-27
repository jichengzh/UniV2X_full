#!/usr/bin/env python3
"""Build and validate original60 H800 TVM artifact workdirs.

This worker is artifact-only. It does not emit latency/AP/energy measured rows.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TVM_SITE = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"
TVM_LD_PREFIX = [
    f"{TVM_SITE}/nvidia/cuda_runtime/lib",
    f"{TVM_SITE}/tvm/lib",
]
CUDA_BIN = "/usr/local/cuda-12.2/bin"


def utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.exists():
        return []
    return [
        json.loads(line)
        for line in item.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_tvm_runtime_env(gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [*TVM_LD_PREFIX]
    nvlibs_path = Path("/exdata/jichengzhi/tvm_nvlibs.path")
    if nvlibs_path.is_file():
        nvlibs = nvlibs_path.read_text(encoding="utf-8").strip()
        if nvlibs:
            ld_parts.append(nvlibs)
    if existing_ld:
        ld_parts.append(existing_ld)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    existing_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{CUDA_BIN}:{existing_path}" if existing_path else CUDA_BIN

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    python_parts = [str(ROOT), TVM_SITE]
    if existing_pythonpath:
        python_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = ":".join(python_parts)


def _safe(value: object) -> str:
    return str(value).replace("/", "-").replace(":", "-").replace(" ", "_")


def latest_status_by_job(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        job_id = row.get("job_id")
        if job_id:
            latest[str(job_id)] = row
    return latest


def _first_number(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if match is None:
        raise ValueError(value)
    return float(match.group(0))


def gpu_preflight(gpu: int, *, log_dir: Path, job_id: str) -> tuple[bool, str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{_safe(job_id)}_preflight.json"
    query_cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate",
        "--format=csv",
    ]
    pmon_cmd = ["nvidia-smi", "pmon", "-c", "1"]
    query = subprocess.run(query_cmd, capture_output=True, text=True, check=False)
    pmon = subprocess.run(pmon_cmd, capture_output=True, text=True, check=False)
    ok = query.returncode == 0 and pmon.returncode == 0
    reason = "gpu_idle"
    if not ok:
        reason = (query.stderr or pmon.stderr or "nvidia_smi_failed").strip()
    util_pct = None
    memory_mib = None
    if ok:
        found = False
        for raw_line in query.stdout.splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("index"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if not parts or parts[0] != str(gpu):
                continue
            found = True
            util_pct = int(_first_number(parts[2]))
            memory_mib = int(_first_number(parts[3]))
            if util_pct > 5:
                ok = False
                reason = f"gpu_not_idle:utilization={util_pct}%"
            if memory_mib > 1024:
                ok = False
                reason = f"gpu_not_idle:memory_used={memory_mib}MiB"
            break
        if not found:
            ok = False
            reason = f"gpu_{gpu}_not_found"
    if ok:
        for raw_line in pmon.stdout.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0] == str(gpu) and parts[1] != "-":
                ok = False
                reason = f"gpu_not_idle:pmon_pid={parts[1]}"
                break
    payload = {
        "schema": "stage2_original60_gpu_preflight_v1",
        "job_id": job_id,
        "gpu": gpu,
        "ok": ok,
        "reason": reason,
        "utilization_gpu_pct": util_pct,
        "memory_used_mib": memory_mib,
        "query_command": query_cmd,
        "pmon_command": pmon_cmd,
        "query_returncode": query.returncode,
        "pmon_returncode": pmon.returncode,
        "query_stdout": query.stdout,
        "query_stderr": query.stderr,
        "pmon_stdout": pmon.stdout,
        "pmon_stderr": pmon.stderr,
        "created_at": utc_timestamp(),
    }
    write_json(out_path, payload)
    return ok, str(out_path), reason


def json_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        if line.strip():
            json.loads(line)
    return True


def read_inputs(onnx_model: Any) -> dict[str, tuple[int, ...]]:
    init = {item.name for item in onnx_model.graph.initializer}
    return {
        item.name: tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim)
        for item in onnx_model.graph.input
        if item.name not in init
    }


def build_relax_module(onnx_path: Path) -> tuple[Any, dict[str, tuple[int, ...]]]:
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    model = onnx.load(str(onnx_path))
    shapes = read_inputs(model)
    return from_onnx(model, shape_dict=shapes, keep_params_in_input=False), shapes


def lower_relax(mod0: Any, target: Any) -> Any:
    import tvm
    from tvm import relax

    seq = tvm.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
    )
    with target, tvm.transform.PassContext(opt_level=3):
        return seq(mod0)


def build_one(job: dict[str, Any], *, args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    configure_tvm_runtime_env(int(job["gpu"]))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if TVM_SITE not in sys.path:
        sys.path.insert(0, TVM_SITE)
    import tvm
    import tvm.s_tir.tensor_intrin.cuda  # noqa: F401 - registers CUDA tensor intrinsics for MetaSchedule.
    from tvm.s_tir.meta_schedule import relax_integration as ri

    onnx_path = Path(job["onnx_path"])
    work_dir = Path(job["tvm_work_dir"])
    workload_path = Path(job["database_workload_path"])
    tuning_record_path = Path(job["database_tuning_record_path"])
    if not onnx_path.is_file() or onnx_path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing ONNX: {onnx_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    job_log_dir = log_dir / _safe(job["label"])
    job_log_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError(f"CUDA device not visible for physical GPU {job['gpu']}")
    target = tvm.target.Target.from_device(dev)
    mod0, shapes = build_relax_module(onnx_path)
    modt = lower_relax(mod0, target)

    tuned = False
    if not workload_path.is_file() or not tuning_record_path.is_file() or args.force_tune:
        ri.tune_relax(
            mod=modt,
            params={},
            target=target,
            work_dir=str(work_dir),
            max_trials_global=int(job.get("max_trials") or args.max_trials),
            seed=args.seed,
        )
        tuned = True
    if not json_readable(workload_path):
        raise RuntimeError(f"database workload missing or unreadable: {workload_path}")
    if not json_readable(tuning_record_path):
        raise RuntimeError(f"database tuning record missing or unreadable: {tuning_record_path}")
    if args.apply_database_smoke:
        from tvm import relax

        with target, tvm.transform.PassContext(opt_level=3):
            scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=str(work_dir))(modt)
            tvm.compile(scheduled, target=target)
    elapsed_s = round(time.time() - start, 3)
    manifest = {
        "schema": "stage2_original60_artifact_build_manifest_row_v1",
        "candidate_id": job["candidate_id"],
        "job_id": job["job_id"],
        "label": job["label"],
        "width": job["width"],
        "gpu": job["gpu"],
        "onnx_path": str(onnx_path),
        "tvm_work_dir": str(work_dir),
        "database_workload_path": str(workload_path),
        "database_tuning_record_path": str(tuning_record_path),
        "input_shapes": {key: list(value) for key, value in shapes.items()},
        "max_trials": int(job.get("max_trials") or args.max_trials),
        "tuned_this_run": tuned,
        "apply_database_smoke": bool(args.apply_database_smoke),
        "elapsed_s": elapsed_s,
        "hostname": socket.gethostname(),
        "created_at": utc_timestamp(),
    }
    write_json(job_log_dir / "artifact_manifest.json", manifest)
    return manifest


def state_row(
    job: dict[str, Any],
    *,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    failure_reason: str | None = None,
    preflight_path: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "schema": "stage2_original60_artifact_build_state_v1",
        "job_id": job["job_id"],
        "candidate_id": job["candidate_id"],
        "label": job["label"],
        "width": job["width"],
        "gpu": job["gpu"],
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "failure_reason": failure_reason,
        "preflight_path": preflight_path,
        "onnx_path": job["onnx_path"],
        "tvm_work_dir": job["tvm_work_dir"],
        "database_workload_path": job["database_workload_path"],
        "database_tuning_record_path": job["database_tuning_record_path"],
    }
    if manifest is not None:
        row["manifest"] = manifest
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-plan", required=True)
    parser.add_argument("--job-state", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--max-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-tune", action="store_true")
    parser.add_argument("--require-gpu-idle", action="store_true")
    parser.add_argument("--apply-database-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_rows = read_jsonl(args.job_plan)
    state_path = Path(args.job_state)
    state_rows = read_jsonl(state_path) if args.resume else []
    latest = latest_status_by_job(state_rows)
    log_dir = Path(args.log_dir)
    ran = 0
    failed = 0
    for job in plan_rows:
        if args.max_jobs is not None and ran >= args.max_jobs:
            break
        if latest.get(str(job["job_id"]), {}).get("status") == "succeeded":
            continue
        started_at = utc_timestamp()
        preflight_path = None
        if args.require_gpu_idle:
            ok, preflight_path, reason = gpu_preflight(int(job["gpu"]), log_dir=log_dir, job_id=str(job["job_id"]))
            if not ok:
                append_jsonl(
                    state_path,
                    state_row(
                        job,
                        status="preflight_blocked",
                        finished_at=utc_timestamp(),
                        failure_reason=reason,
                        preflight_path=preflight_path,
                    ),
                )
                latest = latest_status_by_job(read_jsonl(state_path))
                ran += 1
                continue
        append_jsonl(
            state_path,
            state_row(job, status="running", started_at=started_at, preflight_path=preflight_path),
        )
        try:
            manifest = build_one(job, args=args, log_dir=log_dir)
            append_jsonl(args.manifest_out, manifest)
            append_jsonl(
                state_path,
                state_row(
                    job,
                    status="succeeded",
                    started_at=started_at,
                    finished_at=utc_timestamp(),
                    preflight_path=preflight_path,
                    manifest=manifest,
                ),
            )
        except Exception as exc:
            failed += 1
            failure = {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(log_dir / f"{_safe(job['job_id'])}_failure.json", failure)
            append_jsonl(
                state_path,
                state_row(
                    job,
                    status="failed",
                    started_at=started_at,
                    finished_at=utc_timestamp(),
                    failure_reason=repr(exc),
                    preflight_path=preflight_path,
                ),
            )
        finally:
            gc.collect()
            latest = latest_status_by_job(read_jsonl(state_path))
            ran += 1
    print(
        json.dumps(
            {
                "schema": "stage2_original60_tvm_artifact_worker_summary_v1",
                "job_plan": args.job_plan,
                "job_state": args.job_state,
                "ran": ran,
                "failed": failed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
