#!/usr/bin/env python3
"""Launch TensorRT FP16/INT8 latency+energy jobs for v2 cold-start 96补点."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/home/jichengzhi/V2X")
TRT_PYTHON = Path("/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python")
ONNX_ROOT = Path("/exdata/jichengzhi/codriving_onnx/qxs8_backboneonly_20260708")
CALIB_DIR = Path("/exdata/jichengzhi/calib/codriving_spatial_features_b2_64_256_512_synthetic_20260708")
OUT_ROOT = REPO / "results/codriving_trt_qxs8_20260708"
READY_WIDTHS = (
    "16x32x64",
    "24x32x96",
    "32x32x128",
    "32x64x128",
    "48x96x192",
    "56x112x224",
    "64x96x192",
    "64x128x256",
    "24x64x128",
    "40x64x128",
    "48x64x128",
    "64x64x128",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def trt_env() -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def build_jobs(
    widths: list[str],
    gpu: int,
    warmup: int,
    iters: int,
    repeat: int,
    energy_secs: float,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for width in widths:
        for precision in ("fp16", "int8"):
            jobs.append(
                {
                    "width": width,
                    "precision": precision,
                    "onnx": str(ONNX_ROOT / width / "backbone_only.onnx"),
                    "calib_dir": str(CALIB_DIR),
                    "gpu": gpu,
                    "warmup": warmup,
                    "iters": iters,
                    "repeat": repeat,
                    "energy_secs": energy_secs,
                    "out_json": str(OUT_ROOT / f"codriving_trt_{width}_{precision}_20260708.json"),
                }
            )
    return jobs


def command_for(job: dict[str, Any]) -> list[str]:
    cmd = [
        str(TRT_PYTHON),
        "framework/trt_baseline/trt_profile_v1.py",
        "--gpu",
        str(job["gpu"]),
        "--onnx",
        str(job["onnx"]),
        "--precision",
        str(job["precision"]),
        "--warmup",
        str(job["warmup"]),
        "--iters",
        str(job["iters"]),
        "--repeat",
        str(job["repeat"]),
        "--energy-secs",
        str(job["energy_secs"]),
        "--out",
        str(job["out_json"]),
    ]
    if job["precision"] == "int8":
        cmd.extend(["--calib-dir", str(job["calib_dir"])])
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
    if job["precision"] == "int8" and not any(Path(job["calib_dir"]).glob("*.npy")):
        failure = {"status": "failed", "reason": "missing_calib_npy", "job": job, "created_at_utc": utc_now()}
        write_json(fail_json, failure)
        return failure

    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = command_for(job)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO, env=trt_env(), stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--energy-secs", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.widths, args.gpu, args.warmup, args.iters, args.repeat, args.energy_secs)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard arguments")
    jobs = [job for idx, job in enumerate(jobs) if idx % args.num_shards == args.shard_index]

    status_path = OUT_ROOT / f"queue_status_gpu{args.gpu}_shard{args.shard_index}_of_{args.num_shards}.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        status = run_job(job, force=args.force)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"created_at_utc": utc_now(), **status}, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"width": job["width"], "precision": job["precision"], "status": status["status"]},
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
