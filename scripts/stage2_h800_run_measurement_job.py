#!/usr/bin/env python3
"""Run one real H800 TVM latency or energy job and append Stage2 LUT rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TVM_SITE = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"
LD_PREFIX = [
    f"{TVM_SITE}/nvidia/cuda_runtime/lib",
    f"{TVM_SITE}/tvm/lib",
]
CUDA_BIN = "/usr/local/cuda-12.2/bin"
TVM_NVLIBS_PATH = "/exdata/jichengzhi/tvm_nvlibs.path"


def now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(detail or f"{cmd} failed")
    return proc.stdout


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def configure_tvm_env(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ld_existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [*LD_PREFIX]
    nvlibs_path = Path(TVM_NVLIBS_PATH)
    if nvlibs_path.is_file():
        nvlibs = nvlibs_path.read_text(encoding="utf-8").strip()
        if nvlibs:
            ld_parts.append(nvlibs)
    if ld_existing:
        ld_parts.append(ld_existing)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    path_existing = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{CUDA_BIN}:{path_existing}" if path_existing else CUDA_BIN
    python_existing = os.environ.get("PYTHONPATH", "")
    python_parts = [str(ROOT), TVM_SITE]
    if python_existing:
        python_parts.append(python_existing)
    os.environ["PYTHONPATH"] = ":".join(python_parts)


def gpu_snapshot() -> dict[str, tuple[int, int]]:
    out = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    snap: dict[str, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            snap[parts[0]] = (int(float(parts[1])), int(float(parts[2])))
    return snap


def target_has_pmon(gpu: str) -> bool:
    proc = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[0] == gpu and parts[1] != "-":
            return True
    return False


def wait_gpu_idle(
    gpu: str,
    raw: Path,
    *,
    required_samples: int = 3,
    interval_s: int = 5,
    timeout_s: int = 900,
) -> list[dict[str, object]]:
    deadline = time.time() + timeout_s
    clean = 0
    history: list[dict[str, object]] = []
    while time.time() < deadline:
        snap = gpu_snapshot()
        util, mem = snap.get(gpu, (999, 999999))
        has_pmon = target_has_pmon(gpu)
        item = {"ts": now_local(), "gpu": gpu, "util": util, "mem_mib": mem, "has_pmon": has_pmon}
        history.append(item)
        print(json.dumps({"event": "idle_check", **item}), flush=True)
        if util <= 5 and mem <= 1024 and not has_pmon:
            clean += 1
            if clean >= required_samples:
                write_text(
                    raw / "nvidia_smi_preflight.csv",
                    run_text(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate",
                            "--format=csv",
                        ]
                    ),
                )
                with (raw / "nvidia_smi_pmon_preflight.txt").open("w", encoding="utf-8") as handle:
                    subprocess.run(
                        ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                return history
        else:
            clean = 0
        time.sleep(interval_s)
    raise RuntimeError(f"GPU{gpu} not idle before timeout; tail={history[-5:]}")


def read_inputs(onnx_model: Any) -> dict[str, tuple[int, ...]]:
    init = {item.name for item in onnx_model.graph.initializer}
    return {
        item.name: tuple(dim.dim_value for dim in item.type.tensor_type.shape.dim)
        for item in onnx_model.graph.input
        if item.name not in init
    }


def summarize_times(results_s: list[float]) -> dict[str, object]:
    vals = [float(item) * 1e6 for item in results_s]
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    p50 = vals_sorted[mid] if len(vals_sorted) % 2 else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0
    return {
        "us": round(p50, 3),
        "mean_us": round(sum(vals) / len(vals), 3),
        "min_us": round(min(vals), 3),
        "max_us": round(max(vals), 3),
        "repeats_us": [round(item, 3) for item in vals],
    }


def time_vm(vm: Any, args: list[Any], dev: Any, *, warmup: int, number: int, repeat: int) -> dict[str, object]:
    for _ in range(warmup):
        vm["main"](*args)
        dev.sync()
    result = vm.time_evaluator("main", dev, number=number, repeat=repeat)(*args)
    return summarize_times(list(result.results))


def load_relax_module(onnx_path: Path) -> tuple[Any, dict[str, tuple[int, ...]]]:
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    model = onnx.load(str(onnx_path))
    shapes = read_inputs(model)
    return from_onnx(model, shape_dict=shapes, keep_params_in_input=False), shapes


def make_args(shapes: dict[str, tuple[int, ...]], dev: Any) -> list[Any]:
    import numpy as np
    import tvm

    rng = np.random.RandomState(0)
    feeds_np = {key: rng.rand(*shape).astype("float32") for key, shape in shapes.items()}
    return [tvm.runtime.tensor(feeds_np[key], device=dev) for key in shapes]


def latency_payload(result: dict[str, object], raw: Path, schedule: str, args: argparse.Namespace) -> dict[str, object]:
    prefix = "default" if schedule == "default" else "tuned"
    strategy = "relax_default" if schedule == "default" else "relax_metaschedule_reuse_existing_ms_db"
    return {
        "batch_size": 1,
        "build_status": "success",
        "build_time_s": result.get("build_default_s" if schedule == "default" else "build_tuned_s"),
        "input_shape": result.get("input_shape", {}),
        "latency_max_us": result.get(f"{prefix}_max_us"),
        "latency_mean_us": result.get(f"{prefix}_mean_us"),
        "latency_min_us": result.get(f"{prefix}_min_us"),
        "latency_p50_us": result.get(f"{prefix}_us"),
        "measure_iters": args.measure_iters,
        "notes": f"{args.phase}; {args.label}; {schedule}; backbone-only; no fresh tune",
        "provenance": f"H800 TVM {args.phase} measured on clean target GPU; {schedule}; reuse existing artifacts",
        "raw_artifact": str(raw),
        "repeat": args.repeat,
        "run_id": result.get("run_id"),
        "source_files": [str(args.onnx), str(args.work_dir), str(raw / "latency_result.json")],
        "tvm_strategy": strategy,
        "tvm_target": "cuda",
        "warmup_iters": args.warmup_iters,
    }


def call_generator(command: list[str]) -> None:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(detail or f"generator failed: {command}")


def run_latency(args: argparse.Namespace) -> int:
    raw = Path(args.raw_root) / args.run_id
    raw.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema": "stage2_h800_overnight_latency_result_v1",
        "status": "started",
        "phase": args.phase,
        "label": args.label,
        "gpu": args.gpu,
        "run_id": args.run_id,
        "onnx_path": args.onnx,
        "work_dir": args.work_dir,
        "width": [int(item) for item in args.width.split(",")],
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "repeat": args.repeat,
        "batch_size": 1,
    }
    try:
        configure_tvm_env(args.gpu)
        wait_gpu_idle(args.gpu, raw)
        write_text(raw / "hostname.txt", socket.gethostname() + "\n")
        write_text(raw / "env.json", json.dumps({"CUDA_VISIBLE_DEVICES": args.gpu, "python": sys.executable}, indent=2) + "\n")
        write_text(raw / "command.json", json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

        import tvm
        from tvm import relax
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        mod0, shapes = load_relax_module(Path(args.onnx))
        result["input_shape"] = {key: list(value) for key, value in shapes.items()}
        vm_args = make_args(shapes, dev)

        start = time.time()
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target="cuda")
        result["build_default_s"] = round(time.time() - start, 3)
        vm = relax.VirtualMachine(ex, dev)
        default_stats = time_vm(vm, vm_args, dev, warmup=args.warmup_iters, number=args.measure_iters, repeat=args.repeat)
        result.update(
            {
                "default_us": default_stats["us"],
                "default_mean_us": default_stats["mean_us"],
                "default_min_us": default_stats["min_us"],
                "default_max_us": default_stats["max_us"],
                "default_repeats_us": default_stats["repeats_us"],
            }
        )
        del vm, ex
        dev.sync()

        seq = tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        start = time.time()
        with target, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
            scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=str(args.work_dir))(modt)
            ex2 = tvm.compile(scheduled, target=target)
        result["build_tuned_s"] = round(time.time() - start, 3)
        vm2 = relax.VirtualMachine(ex2, dev)
        tuned_stats = time_vm(vm2, vm_args, dev, warmup=args.warmup_iters, number=args.measure_iters, repeat=args.repeat)
        result.update(
            {
                "tuned_us": tuned_stats["us"],
                "tuned_mean_us": tuned_stats["mean_us"],
                "tuned_min_us": tuned_stats["min_us"],
                "tuned_max_us": tuned_stats["max_us"],
                "tuned_repeats_us": tuned_stats["repeats_us"],
            }
        )
        result["ratio"] = round(float(result["default_us"]) / float(result["tuned_us"]), 6)
        result["status"] = "success"
        write_text(raw / "latency_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")

        for schedule, config_id in (("default", args.config_id_default), ("metaschedule_tuned", args.config_id_tuned)):
            payload_path = raw / f"measurement_payload_{schedule}.json"
            write_text(payload_path, json.dumps(latency_payload(result, raw, schedule, args), indent=2, sort_keys=True) + "\n")
            call_generator(
                [
                    sys.executable,
                    "scripts/stage2_generate_latency_lut.py",
                    "--job-id",
                    f"latency:{args.label}:{schedule}:{args.run_id}",
                    "--model",
                    args.model,
                    "--config-id",
                    config_id,
                    "--candidate-id",
                    args.candidate_id,
                    "--software-point-id",
                    args.software_point_id,
                    "--dense-stage",
                    "backbone",
                    "--optimized-scope",
                    args.optimized_scope,
                    "--width",
                    args.width,
                    "--quant-policy",
                    args.quant_policy,
                    "--schedule-policy",
                    schedule,
                    "--backend",
                    "h800_tvm",
                    "--manifest-digest",
                    args.manifest_digest,
                    "--run-id",
                    f"{args.run_id}_{schedule}",
                    "--warmup-iters",
                    str(args.warmup_iters),
                    "--measure-iters",
                    str(args.measure_iters),
                    "--repeat",
                    str(args.repeat),
                    "--batch-size",
                    "1",
                    "--measurement-command-json",
                    json.dumps(["/bin/cat", str(payload_path)]),
                    "--out-jsonl",
                    args.out_jsonl,
                ]
            )
        print(json.dumps({"event": "latency_success", "label": args.label, "run_id": args.run_id, "tuned_us": result["tuned_us"]}))
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        write_text(raw / "latency_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "latency_failed", "label": args.label, "run_id": args.run_id, "error": repr(exc)}), file=sys.stderr)
        return 1


def query_power_w(gpu: str) -> float:
    out = run_text(["nvidia-smi", f"--id={gpu}", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    return float(out.strip().splitlines()[0].strip())


def sample_power(gpu: str, seconds: float, interval: float) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    end = time.time() + seconds
    while time.time() < end:
        ts = time.time()
        try:
            rows.append((ts, query_power_w(gpu)))
        except Exception:
            pass
        time.sleep(interval)
    return rows


def write_power_csv(path: Path, rows: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_s", "power_w"])
        for ts, watt in rows:
            writer.writerow([f"{ts:.6f}", f"{watt:.3f}"])


def power_stats(rows: list[tuple[float, float]]) -> dict[str, float | None]:
    vals = [watt for _, watt in rows]
    if not vals:
        return {"avg": None, "p50": None, "p90": None}
    sorted_vals = sorted(vals)
    return {
        "avg": sum(vals) / len(vals),
        "p50": sorted_vals[len(sorted_vals) // 2],
        "p90": sorted_vals[min(len(sorted_vals) - 1, int(0.9 * (len(sorted_vals) - 1)))],
    }


def energy_payload(result: dict[str, object], raw: Path, args: argparse.Namespace) -> dict[str, object]:
    return {
        "run_id": args.run_id,
        "latency_run_id": args.latency_run_id,
        "latency_config_id": args.config_id_tuned,
        "joule_per_inference": result["joule_per_inference"],
        "watt_avg": result["watt_avg"],
        "watt_p50": result["watt_p50"],
        "watt_p90": result["watt_p90"],
        "idle_watt_avg": result["idle_watt_avg"],
        "idle_baseline_policy": "subtract_idle_avg_5s_pre_window",
        "sample_window_ms": result["sample_window_ms"],
        "telemetry_source": "nvidia-smi power.draw polling 50ms",
        "power_cap_watt": None,
        "clock_policy": "default",
        "warmup_iters": args.energy_warmup_iters,
        "measure_iters": args.energy_measure_iters,
        "repeat": 1,
        "provenance": f"H800 power telemetry {args.phase} aligned with tuned TVM VM",
        "source_files": [
            str(args.onnx),
            str(args.work_dir),
            str(raw / "energy_result.json"),
            str(raw / "idle_power_samples.csv"),
            str(raw / "active_power_samples.csv"),
        ],
        "raw_artifact": str(raw),
        "notes": f"{args.phase} energy telemetry; clean target GPU preflight; backbone-only",
    }


def run_energy(args: argparse.Namespace) -> int:
    raw = Path(args.raw_root) / args.run_id
    raw.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema": "stage2_h800_overnight_energy_result_v1",
        "status": "started",
        "phase": args.phase,
        "label": args.label,
        "gpu": args.gpu,
        "run_id": args.run_id,
    }
    try:
        configure_tvm_env(args.gpu)
        wait_gpu_idle(args.gpu, raw)
        write_text(raw / "hostname.txt", socket.gethostname() + "\n")
        write_text(raw / "env.json", json.dumps({"CUDA_VISIBLE_DEVICES": args.gpu, "python": sys.executable}, indent=2) + "\n")
        write_text(raw / "command.json", json.dumps(vars(args), indent=2, sort_keys=True) + "\n")

        import tvm
        from tvm import relax
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        mod0, shapes = load_relax_module(Path(args.onnx))
        vm_args = make_args(shapes, dev)
        seq = tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        with target, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
            scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=str(args.work_dir))(modt)
            ex = tvm.compile(scheduled, target=target)
        vm = relax.VirtualMachine(ex, dev)
        for _ in range(args.energy_warmup_iters):
            vm["main"](*vm_args)
            dev.sync()
        idle_samples = sample_power(args.gpu, 5.0, 0.05)
        active_samples: list[tuple[float, float]] = []
        start = time.time()
        for _ in range(args.energy_measure_iters):
            sample_start = time.time()
            vm["main"](*vm_args)
            dev.sync()
            try:
                active_samples.append((sample_start, query_power_w(args.gpu)))
            except Exception:
                pass
        elapsed = max(time.time() - start, 1e-9)
        idle = power_stats(idle_samples)
        active = power_stats(active_samples)
        idle_avg = float(idle["avg"] or 0.0)
        watt_avg = float(active["avg"] or 0.0)
        net_watt = max(watt_avg - idle_avg, 0.0)
        result.update(
            {
                "idle_watt_avg": idle_avg,
                "watt_avg": watt_avg,
                "watt_p50": active["p50"],
                "watt_p90": active["p90"],
                "sample_window_ms": int(round(elapsed * 1000.0)),
                "joule_per_inference": net_watt * elapsed / float(args.energy_measure_iters),
                "elapsed_s": elapsed,
                "status": "success",
            }
        )
        write_power_csv(raw / "idle_power_samples.csv", idle_samples)
        write_power_csv(raw / "active_power_samples.csv", active_samples)
        write_text(raw / "energy_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        payload_path = raw / "telemetry_payload.json"
        write_text(payload_path, json.dumps(energy_payload(result, raw, args), indent=2, sort_keys=True) + "\n")
        call_generator(
            [
                sys.executable,
                "scripts/stage2_generate_energy_lut.py",
                "--job-id",
                f"energy:{args.label}:{args.run_id}",
                "--model",
                args.model,
                "--config-id",
                args.config_id_tuned,
                "--candidate-id",
                args.candidate_id,
                "--software-point-id",
                args.software_point_id,
                "--dense-stage",
                "backbone",
                "--optimized-scope",
                args.optimized_scope,
                "--width",
                args.width,
                "--quant-policy",
                args.quant_policy,
                "--schedule-policy",
                "metaschedule_tuned",
                "--backend",
                "h800_tvm_power_telemetry",
                "--manifest-digest",
                args.manifest_digest,
                "--latency-run-id",
                args.latency_run_id,
                "--run-id",
                args.run_id,
                "--warmup-iters",
                str(args.energy_warmup_iters),
                "--measure-iters",
                str(args.energy_measure_iters),
                "--repeat",
                "1",
                "--telemetry-command-json",
                json.dumps(["/bin/cat", str(payload_path)]),
                "--out-jsonl",
                args.out_jsonl,
            ]
        )
        print(json.dumps({"event": "energy_success", "label": args.label, "run_id": args.run_id, "joule_per_inference": result["joule_per_inference"]}))
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        write_text(raw / "energy_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "energy_failed", "label": args.label, "run_id": args.run_id, "error": repr(exc)}), file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("latency", "energy"), required=True)
    parser.add_argument("--model", default="pyramid_lidar")
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", default="overnight_6h")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--software-point-id", required=True)
    parser.add_argument("--config-id-tuned", required=True)
    parser.add_argument("--config-id-default", default="")
    parser.add_argument("--latency-run-id", default="same_config_pending")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--manifest-digest", default="overnight_6h_20260626")
    parser.add_argument("--optimized-scope", default="backbone_only")
    parser.add_argument("--quant-policy", default="fp16")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--measure-iters", type=int, default=500)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--energy-warmup-iters", type=int, default=50)
    parser.add_argument("--energy-measure-iters", type=int, default=1500)
    args = parser.parse_args()
    if args.kind == "latency" and not args.config_id_default:
        parser.error("--config-id-default is required for latency")
    return args


def main() -> int:
    args = parse_args()
    if args.kind == "latency":
        return run_latency(args)
    return run_energy(args)


if __name__ == "__main__":
    raise SystemExit(main())
