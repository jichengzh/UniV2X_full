#!/usr/bin/env python3
"""Measure latency or energy for an exported FP16 rewritten Relax VM artifact."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TVM_SITE = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"
CUDA_BIN = "/usr/local/cuda-12.2/bin"
TVM_NVLIBS_PATH = "/exdata/jichengzhi/tvm_nvlibs.path"

from framework.stage2.lut_productization import (  # noqa: E402
    append_jsonl,
    energy_lut_row,
    latency_lut_row,
    utc_timestamp,
    validate_lut_row,
)


def _now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"{cmd} failed")
    return proc.stdout


def configure_tvm_env(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    nvlibs = Path(TVM_NVLIBS_PATH).read_text(encoding="utf-8").strip() if Path(TVM_NVLIBS_PATH).is_file() else ""
    ld_parts = [
        f"{TVM_SITE}/nvidia/cuda_runtime/lib",
        f"{TVM_SITE}/tvm/lib",
    ]
    if nvlibs:
        ld_parts.append(nvlibs)
    if os.environ.get("LD_LIBRARY_PATH"):
        ld_parts.append(os.environ["LD_LIBRARY_PATH"])
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    os.environ["PATH"] = f"{CUDA_BIN}:{os.environ.get('PATH', '')}"
    os.environ["PYTHONPATH"] = f"{ROOT}:{TVM_SITE}:{os.environ.get('PYTHONPATH', '')}"


def gpu_snapshot() -> dict[str, tuple[int, int]]:
    out = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    result: dict[str, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            result[parts[0]] = (int(float(parts[1])), int(float(parts[2])))
    return result


def target_has_pmon(gpu: str) -> bool:
    proc = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (proc.stdout or "").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) >= 2 and parts[0] == str(gpu) and parts[1] != "-":
            return True
    return False


def wait_gpu_idle(gpu: str, raw: Path, samples: int = 3, interval_s: int = 5, timeout_s: int = 900) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_s
    clean = 0
    history: list[dict[str, Any]] = []
    while time.time() < deadline:
        util, mem = gpu_snapshot().get(str(gpu), (999, 999999))
        has_pmon = target_has_pmon(str(gpu))
        item = {"ts": _now_local(), "gpu": str(gpu), "util": util, "mem_mib": mem, "has_pmon": has_pmon}
        history.append(item)
        print(json.dumps({"event": "idle_check", **item}), flush=True)
        if util <= 5 and mem <= 1024 and not has_pmon:
            clean += 1
            if clean >= samples:
                _write_text(
                    raw / "nvidia_smi_preflight.csv",
                    _run_text(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate",
                            "--format=csv",
                        ]
                    ),
                )
                with (raw / "nvidia_smi_pmon_preflight.txt").open("w", encoding="utf-8") as handle:
                    subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "um"], stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
                return history
        else:
            clean = 0
        time.sleep(interval_s)
    raise RuntimeError(f"GPU{gpu} not idle before timeout; tail={history[-5:]}")


def query_power_w(gpu: str) -> float:
    out = _run_text(["nvidia-smi", f"--id={gpu}", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    return float(out.strip().splitlines()[0].strip())


def sample_power(gpu: str, seconds: float, interval_s: float) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    end = time.time() + seconds
    while time.time() < end:
        try:
            rows.append((time.time(), query_power_w(gpu)))
        except Exception:
            pass
        time.sleep(interval_s)
    return rows


def power_stats(rows: list[tuple[float, float]]) -> dict[str, float | None]:
    vals = [item[1] for item in rows]
    if not vals:
        return {"avg": None, "p50": None, "p90": None}
    vals_sorted = sorted(vals)
    return {
        "avg": sum(vals) / len(vals),
        "p50": vals_sorted[len(vals_sorted) // 2],
        "p90": vals_sorted[min(len(vals_sorted) - 1, int(0.9 * (len(vals_sorted) - 1)))],
    }


def write_power_csv(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_s", "power_w"])
        for ts, watt in rows:
            writer.writerow([f"{ts:.6f}", f"{watt:.3f}"])


def load_shape(report: dict[str, Any]) -> tuple[str, tuple[int, ...], str]:
    shape_dict = report.get("shape_dict") or {}
    input_dtypes = report.get("input_dtypes") or {}
    if not shape_dict:
        raise ValueError("rewrite report missing shape_dict")
    name = next(iter(shape_dict.keys()))
    shape = tuple(int(item) for item in shape_dict[name])
    dtype = str(input_dtypes.get(name) or "float16")
    return name, shape, dtype


def make_vm_args(tvm: Any, shape: tuple[int, ...], dtype: str, dev: Any) -> list[Any]:
    rng = np.random.RandomState(0)
    value = rng.rand(*shape).astype(dtype)
    tensor_ctor = getattr(tvm.runtime, "tensor", None)
    arr = tensor_ctor(value, device=dev) if tensor_ctor is not None else tvm.nd.array(value, dev)
    return [arr]


def summarize_latency(results_s: list[float]) -> dict[str, Any]:
    vals = [float(item) * 1e6 for item in results_s]
    ordered = sorted(vals)
    mid = len(ordered) // 2
    p50 = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "latency_p50_us": round(p50, 3),
        "latency_mean_us": round(sum(vals) / len(vals), 3),
        "latency_min_us": round(min(vals), 3),
        "latency_max_us": round(max(vals), 3),
        "latency_repeats_us": [round(item, 3) for item in vals],
    }


def common_row_args(args: argparse.Namespace, report: dict[str, Any], raw: Path) -> dict[str, Any]:
    return {
        "model": "pyramid_lidar",
        "manifest_digest": args.manifest_digest,
        "candidate_id": args.label,
        "software_point_id": f"original60:{args.label}:fp16_rewritten_tensorcore:{args.route}:artifact",
        "dense_stage": "backbone",
        "optimized_scope": "backbone_only",
        "width": [int(item) for item in args.width.split(",")],
        "quant_policy": "fp16",
        "schedule_policy": args.route,
        "measurement_status": "measured",
        "provenance": "H800 TVM FP16 rewritten TensorCore Relax VM artifact direct measurement",
        "created_at": utc_timestamp(),
        "source_files": [str(args.rewrite_report), str(args.artifact), str(raw)],
        "raw_artifact": str(raw),
        "precision": "fp16",
        "quant_scheme": "fp16_cast",
        "quant_method": "h800_tvm_fp16_rewritten_tensorcore",
        "quant_scope": "backbone_only",
        "calibration_source": "none",
        "calibration_digest": "none",
        "calibrator": "none",
        "calibration_inputs": [],
        "fallback_policy": "none",
        "layer_precision_summary": str(args.rewrite_report),
        "full_network_claim": False,
        "engine_kind": "tvm_vm",
        "engine_digest": "unknown",
        "measurement_source": "true_measurement",
        "claim_status": "claimable_true_measurement",
        "schedule_profile": args.route,
        "tune_budget": "rewrite_existing_artifact_no_fresh_tune",
        "input_shape": report.get("shape_dict"),
        "notes": f"FP16 rewritten TensorCore artifact; label={args.label}; route={args.route}; full_network_claim=false",
    }


def run_latency(args: argparse.Namespace) -> int:
    raw = Path(args.raw_root) / args.run_id
    raw.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema": "fp16_rewritten_artifact_latency_result_v1", "status": "started", "run_id": args.run_id}
    try:
        configure_tvm_env(args.gpu)
        wait_gpu_idle(args.gpu, raw)
        _write_text(raw / "hostname.txt", socket.gethostname() + "\n")
        _write_text(raw / "command.json", json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n")
        report = json.loads(Path(args.rewrite_report).read_text(encoding="utf-8"))
        input_name, shape, dtype = load_shape(report)
        import tvm
        from tvm import relax

        dev = tvm.cuda(0)
        lib = tvm.runtime.load_module(str(args.artifact))
        vm = relax.VirtualMachine(lib, dev)
        vm_args = make_vm_args(tvm, shape, dtype, dev)
        for _ in range(args.warmup_iters):
            vm["main"](*vm_args)
            dev.sync()
        timed = vm.time_evaluator("main", dev, number=args.measure_iters, repeat=args.repeat)(*vm_args)
        stats = summarize_latency(list(timed.results))
        result.update(
            {
                "status": "success",
                "artifact": str(args.artifact),
                "input_name": input_name,
                "input_shape": list(shape),
                "input_dtype": dtype,
                "warmup_iters": args.warmup_iters,
                "measure_iters": args.measure_iters,
                "repeat": args.repeat,
                **stats,
            }
        )
        _write_text(raw / "latency_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        row = latency_lut_row(
            config_id=args.config_id,
            backend="h800_tvm",
            run_id=args.run_id,
            batch_size=1,
            build_status="loaded_exported_artifact",
            tvm_strategy="load_exported_rewritten_tensorcore_relax_vm",
            tvm_target="cuda",
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            repeat=args.repeat,
            quality_gate_status="fp16_rewritten_tensorcore_latency_h800",
            **common_row_args(args, report, raw),
            **stats,
        )
        validate_lut_row(row)
        append_jsonl(args.out_jsonl, row)
        print(json.dumps({"event": "latency_success", "run_id": args.run_id, **stats}))
        return 0
    except Exception as exc:
        result.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        _write_text(raw / "latency_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "latency_failed", "run_id": args.run_id, "error": repr(exc)}), file=sys.stderr)
        return 1


def run_energy(args: argparse.Namespace) -> int:
    raw = Path(args.raw_root) / args.run_id
    raw.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"schema": "fp16_rewritten_artifact_energy_result_v1", "status": "started", "run_id": args.run_id}
    try:
        configure_tvm_env(args.gpu)
        wait_gpu_idle(args.gpu, raw)
        _write_text(raw / "hostname.txt", socket.gethostname() + "\n")
        _write_text(raw / "command.json", json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n")
        report = json.loads(Path(args.rewrite_report).read_text(encoding="utf-8"))
        input_name, shape, dtype = load_shape(report)
        import tvm
        from tvm import relax

        dev = tvm.cuda(0)
        lib = tvm.runtime.load_module(str(args.artifact))
        vm = relax.VirtualMachine(lib, dev)
        vm_args = make_vm_args(tvm, shape, dtype, dev)
        for _ in range(args.energy_warmup_iters):
            vm["main"](*vm_args)
            dev.sync()
        idle_samples = sample_power(args.gpu, 5.0, 0.05)
        active_samples: list[tuple[float, float]] = []
        completed = 0
        stop_sampling = threading.Event()
        start = time.time()

        def poll_power() -> None:
            while not stop_sampling.is_set():
                try:
                    active_samples.append((time.time(), query_power_w(args.gpu)))
                except Exception:
                    pass
                time.sleep(0.05)

        sampler = threading.Thread(target=poll_power, name="fp16_rewritten_energy_power_sampler", daemon=True)
        sampler.start()
        try:
            while completed < args.energy_measure_iters or (time.time() - start) < args.energy_min_active_s:
                vm["main"](*vm_args)
                completed += 1
                if completed % args.energy_sync_interval_iters == 0:
                    dev.sync()
            dev.sync()
        finally:
            stop_sampling.set()
            sampler.join(timeout=1.0)
        elapsed = max(time.time() - start, 1e-9)
        idle = power_stats(idle_samples)
        active = power_stats(active_samples)
        idle_avg = float(idle["avg"] or 0.0)
        watt_avg = float(active["avg"] or 0.0)
        net_watt = max(watt_avg - idle_avg, 0.0)
        joule = net_watt * elapsed / float(max(completed, 1))
        write_power_csv(raw / "idle_power_samples.csv", idle_samples)
        write_power_csv(raw / "active_power_samples.csv", active_samples)
        result.update(
            {
                "status": "success",
                "artifact": str(args.artifact),
                "input_name": input_name,
                "input_shape": list(shape),
                "input_dtype": dtype,
                "energy_sampling_mode": "threaded_window",
                "energy_warmup_iters": args.energy_warmup_iters,
                "completed_measure_iters": completed,
                "requested_measure_iters": args.energy_measure_iters,
                "energy_sync_interval_iters": args.energy_sync_interval_iters,
                "elapsed_s": elapsed,
                "sample_window_ms": int(round(elapsed * 1000.0)),
                "idle_watt_avg": idle_avg,
                "watt_avg": watt_avg,
                "watt_p50": active["p50"],
                "watt_p90": active["p90"],
                "dynamic_watt_avg": net_watt,
                "joule_per_inference": joule,
                "idle_sample_count": len(idle_samples),
                "active_sample_count": len(active_samples),
            }
        )
        _write_text(raw / "energy_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        row = energy_lut_row(
            config_id=args.config_id,
            backend="h800_tvm_power_telemetry",
            run_id=args.run_id,
            measurement_run_id=args.run_id,
            latency_run_id=args.latency_run_id,
            latency_config_id=args.config_id,
            joule_per_inference=joule,
            watt_avg=watt_avg,
            watt_p50=active["p50"],
            watt_p90=active["p90"],
            idle_watt_avg=idle_avg,
            idle_baseline_policy="subtract_idle_avg_5s_pre_window",
            sample_window_ms=int(round(elapsed * 1000.0)),
            telemetry_source="nvidia-smi power.draw polling 50ms",
            power_cap_watt=None,
            clock_policy="default",
            warmup_iters=args.energy_warmup_iters,
            measure_iters=completed,
            requested_measure_iters=args.energy_measure_iters,
            completed_measure_iters=completed,
            min_active_s=args.energy_min_active_s,
            energy_sampling_mode="threaded_window",
            energy_sync_interval_iters=args.energy_sync_interval_iters,
            repeat=1,
            quality_gate_status="fp16_rewritten_tensorcore_energy_threaded_window_h800",
            **common_row_args(args, report, raw),
        )
        validate_lut_row(row)
        append_jsonl(args.out_jsonl, row)
        print(json.dumps({"event": "energy_success", "run_id": args.run_id, "joule_per_inference": joule, "dynamic_watt_avg": net_watt}))
        return 0
    except Exception as exc:
        result.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
        _write_text(raw / "energy_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "energy_failed", "run_id": args.run_id, "error": repr(exc)}), file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("latency", "energy"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--rewrite-report", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--latency-run-id", default="same_config_pending")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--manifest-digest", default="original60_quant_20260701_fp16_rewritten_tensorcore")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--energy-warmup-iters", type=int, default=20)
    parser.add_argument("--energy-measure-iters", type=int, default=300)
    parser.add_argument("--energy-min-active-s", type=float, default=5.0)
    parser.add_argument("--energy-sync-interval-iters", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_latency(args) if args.kind == "latency" else run_energy(args)


if __name__ == "__main__":
    raise SystemExit(main())
