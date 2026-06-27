#!/usr/bin/env python3
"""Adapter from H800 B1 TVM smoke output to Stage2 latency generator JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PYTHON = "/exdata/jichengzhi/tvm310/bin/python"
DEFAULT_B1_SCRIPT = "/exdata/jichengzhi/s2_tvm/b1_single_width.py"
DEFAULT_MODELS_DIR = "/exdata/jichengzhi/s2_tvm/models"


class AdapterError(RuntimeError):
    """Raised when the B1 latency result cannot become a generator payload."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--onnx-file", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument(
        "--schedule-policy",
        choices=("default", "metaschedule_tuned", "tuned"),
        default="metaschedule_tuned",
    )
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--b1-script", default=DEFAULT_B1_SCRIPT)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def _float_field(result: dict[str, Any], key: str) -> float:
    try:
        return float(result[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError(f"B1 result missing numeric field: {key}") from exc


def _int_field(result: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(result.get(key, default))
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"B1 result has invalid integer field: {key}") from exc


def _selected_latency_us(result: dict[str, Any], schedule_policy: str) -> float:
    key = "default_us" if schedule_policy == "default" else "tuned_us"
    value = _float_field(result, key)
    if value <= 0:
        raise AdapterError(f"B1 result does not contain a positive latency for {key}")
    return value


def payload_from_b1_result(
    result: dict[str, Any],
    *,
    schedule_policy: str,
    raw_artifact: str,
    log_artifact: str,
    b1_script: str,
    onnx_path: str,
) -> dict[str, Any]:
    if result.get("status") != "DONE":
        raise AdapterError(f"B1 result is not DONE: {result.get('status')}")

    latency_us = _selected_latency_us(result, schedule_policy)
    strategy = "relax_default" if schedule_policy == "default" else "relax_metaschedule"
    reps = _int_field(result, "reps", 50)
    trials = _int_field(result, "trials", 0)
    tune_s = _float_field(result, "tune_s") if result.get("tune_s") is not None else None

    return {
        "latency_p50_us": latency_us,
        "latency_mean_us": latency_us,
        "latency_min_us": latency_us,
        "warmup_iters": 1,
        "measure_iters": reps,
        "repeat": 5,
        "batch_size": 1,
        "input_shape": {},
        "tvm_target": "cuda",
        "tvm_strategy": strategy,
        "build_status": "success",
        "build_time_s": tune_s,
        "provenance": "H800 TVM B1 single-width real smoke adapter",
        "source_files": [b1_script, onnx_path, log_artifact],
        "raw_artifact": raw_artifact,
        "notes": (
            "H800 dense-core/backbone-only smoke; "
            f"schedule_policy={schedule_policy}; trials={trials}; reps={reps}"
        ),
    }


def _run_b1(args: argparse.Namespace) -> None:
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_json.with_suffix(out_json.suffix + ".log.json")
    command = [
        args.python,
        args.b1_script,
        args.label,
        args.onnx_file,
        str(out_json),
        str(args.trials),
        str(args.reps),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    proc = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    log_path.write_text(
        json.dumps(
            {
                "schema": "stage2_h800_b1_latency_adapter_log_v1",
                "command": command,
                "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise AdapterError(detail or f"B1 command failed with exit code {proc.returncode}")


def main() -> int:
    args = parse_args()
    try:
        if not args.skip_run:
            _run_b1(args)
        raw_artifact = str(Path(args.out_json))
        log_artifact = str(Path(args.out_json).with_suffix(Path(args.out_json).suffix + ".log.json"))
        result = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
        onnx_path = str(Path(args.models_dir) / args.onnx_file)
        payload = payload_from_b1_result(
            result,
            schedule_policy=args.schedule_policy,
            raw_artifact=raw_artifact,
            log_artifact=log_artifact,
            b1_script=args.b1_script,
            onnx_path=onnx_path,
        )
    except (AdapterError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
