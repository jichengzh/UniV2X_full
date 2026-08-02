#!/usr/bin/env python3
"""Launch TVM latency+energy补点 jobs for the v2 96-point cold-start plan.

This runner intentionally handles only the TVM whole-engine path that can be
closed with the current ``stage2_codriving_whole_engine_tc_v1.py`` entry.
It writes per-job raw JSON and failure JSON, so the gold collector can import
successful latency+energy rows without treating failures as missing evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/jichengzhi/V2X")
TVM_PYTHON = Path("/exdata/jichengzhi/tvm310/bin/python")
ONNX_ROOT = Path("/exdata/jichengzhi/codriving_onnx/qxs8_backboneonly_20260708")
OUT_ROOT = REPO / "results/v2_gold_coldstart_96_20260708/tvm_energy_raw"
READY_WIDTHS = (
    "16x32x64",
    "24x32x96",
    "32x32x128",
    "32x64x128",
    "48x96x192",
    "56x112x224",
    "64x96x192",
    "64x128x256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tvm_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = "/usr/local/cuda-12.2/bin:" + env.get("PATH", "")
    ld_parts = [
        "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
        "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib",
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def build_jobs(widths: list[str], gpu: int, reps: int, energy_iters: int, energy_min_active_s: float) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for width in widths:
        onnx = ONNX_ROOT / width / "backbone_only.onnx"
        specs = [
            {
                "job_kind": "pure_both",
                "precision": "both",
                "mixed_policy": "none",
                "out_json": str(OUT_ROOT / width / f"pure_both_energy_gpu{gpu}.json"),
            },
            {
                "job_kind": "mixed_top25_flops",
                "precision": "mixed",
                "mixed_policy": "top25_flops",
                "out_json": str(OUT_ROOT / width / f"mixed_top25_flops_energy_gpu{gpu}.json"),
            },
            {
                "job_kind": "mixed_top50_flops",
                "precision": "mixed",
                "mixed_policy": "top50_flops",
                "out_json": str(OUT_ROOT / width / f"mixed_top50_flops_energy_gpu{gpu}.json"),
            },
        ]
        for spec in specs:
            jobs.append(
                {
                    **spec,
                    "width": width,
                    "onnx": str(onnx),
                    "gpu": gpu,
                    "reps": reps,
                    "energy_iters": energy_iters,
                    "energy_min_active_s": energy_min_active_s,
                }
            )
    return jobs


def command_for(job: dict[str, Any]) -> list[str]:
    cmd = [
        str(TVM_PYTHON),
        "scripts/stage2_codriving_whole_engine_tc_v1.py",
        "--gpu",
        str(job["gpu"]),
        "--onnx",
        str(job["onnx"]),
        "--precision",
        str(job["precision"]),
        "--reps",
        str(job["reps"]),
        "--measure-energy",
        "--energy-iters",
        str(job["energy_iters"]),
        "--energy-min-active-s",
        str(job["energy_min_active_s"]),
        "--out-json",
        str(job["out_json"]),
    ]
    if job["precision"] == "mixed":
        cmd.extend(["--mixed-policy", str(job["mixed_policy"])])
    return cmd


def run_job(job: dict[str, Any], force: bool) -> dict[str, Any]:
    out_json = Path(job["out_json"])
    log_path = out_json.with_suffix(".log")
    fail_json = out_json.with_suffix(".failed.json")
    if out_json.is_file() and not force:
        return {"status": "skipped_existing", "job": job, "out_json": str(out_json)}
    if not Path(job["onnx"]).is_file():
        failure = {"status": "failed", "reason": "missing_onnx", "job": job, "created_at_utc": utc_now()}
        write_json(fail_json, failure)
        return failure

    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = command_for(job)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=tvm_env(), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.time() - started
    if proc.returncode != 0:
        failure = {
            "status": "failed",
            "reason": "nonzero_returncode",
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "job": job,
            "command": cmd,
            "log": str(log_path),
            "created_at_utc": utc_now(),
        }
        write_json(fail_json, failure)
        return failure
    return {"status": "success", "elapsed_s": elapsed, "job": job, "out_json": str(out_json), "log": str(log_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--widths", nargs="*", default=list(READY_WIDTHS))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--energy-iters", type=int, default=300)
    parser.add_argument("--energy-min-active-s", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.widths, args.gpu, args.reps, args.energy_iters, args.energy_min_active_s)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard arguments")
    jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]

    status_path = OUT_ROOT / f"queue_status_gpu{args.gpu}_shard{args.shard_index}_of_{args.num_shards}.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        status = run_job(job, force=args.force)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_at_utc": utc_now(), **status}, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"width": job["width"], "job_kind": job["job_kind"], "status": status["status"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
