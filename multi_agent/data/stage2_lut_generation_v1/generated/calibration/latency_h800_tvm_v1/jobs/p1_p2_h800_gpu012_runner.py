from __future__ import annotations

import csv
import argparse
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

ROOT = Path("/home/jichengzhi/V2X")
CALIB_ROOT = ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1"
SMOKE_ROOT = ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1"
STATE_PATH = CALIB_ROOT / "jobs/job_state_p1_p2_gpu012.jsonl"
QUARANTINE_PATH = ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/bad_db_quarantine_v1.jsonl"
GEN_PY = sys.executable

TVM_SITE = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"
LD_PREFIX = [
    f"{TVM_SITE}/nvidia/cuda_runtime/lib",
    f"{TVM_SITE}/tvm/lib",
]
CUDA_BIN = "/usr/local/cuda-12.2/bin"

WARMUP_LATENCY = 1
MEASURE_LATENCY = 500
REPEAT_LATENCY = 5
WARMUP_ENERGY = 50
MEASURE_ENERGY = 1500
REPEAT_ENERGY = 1

ENERGY_JOBS = [
    {
        "phase": "p1_energy",
        "label": "base",
        "gpu": "2",
        "width": [64, 128, 256],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_work_2e_base",
        "config_id": "calibration_h800_tvm_pyramid_base_fp16_metaschedule_tuned",
        "candidate_id": "calibration:pyramid_lidar:base",
        "software_point_id": "pyramid_lidar:backbone:w64x128x256:fp16",
        "latency_run_id": "calib_h800_tvm_pyramid_base_fp16_tuned_clean_20260625_203000",
    },
    {
        "phase": "p1_energy",
        "label": "p50",
        "gpu": "2",
        "width": [32, 64, 128],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/p50_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_work_2e_p50",
        "config_id": "calibration_h800_tvm_pyramid_p50_fp16_metaschedule_tuned",
        "candidate_id": "calibration:pyramid_lidar:p50",
        "software_point_id": "pyramid_lidar:backbone:w32x64x128:fp16",
        "latency_run_id": "calib_h800_tvm_pyramid_p50_fp16_tuned_clean_20260625_203000",
    },
    {
        "phase": "p1_energy",
        "label": "p75",
        "gpu": "2",
        "width": [16, 32, 64],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/p75_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_work_2e_p75_retest",
        "config_id": "calibration_h800_tvm_pyramid_p75_fp16_metaschedule_tuned",
        "candidate_id": "calibration:pyramid_lidar:p75",
        "software_point_id": "pyramid_lidar:backbone:w16x32x64:fp16",
        "latency_run_id": "calib_h800_tvm_pyramid_p75_fp16_tuned_clean_20260625_203000",
    },
    {
        "phase": "p1_energy",
        "label": "mix_b",
        "gpu": "2",
        "width": [48, 64, 256],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/mix_b_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_b",
        "config_id": "calibration_h800_tvm_pyramid_mix_b_fp16_metaschedule_tuned",
        "candidate_id": "calibration:pyramid_lidar:mix_b",
        "software_point_id": "pyramid_lidar:backbone:w48x64x256:fp16",
        "latency_run_id": "calib_h800_tvm_pyramid_mix_b_fp16_medium_clean_retry_g1_20260625_205310",
    },
    {
        "phase": "p1_energy",
        "label": "mix_d",
        "gpu": "2",
        "width": [48, 128, 128],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/mix_d_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_d",
        "config_id": "calibration_h800_tvm_pyramid_mix_d_fp16_metaschedule_tuned",
        "candidate_id": "calibration:pyramid_lidar:mix_d",
        "software_point_id": "pyramid_lidar:backbone:w48x128x128:fp16",
        "latency_run_id": "calib_h800_tvm_pyramid_mix_d_fp16_medium_clean_g012_20260625_202323",
    },
]

LATENCY_REPEAT_JOBS = [
    {
        "phase": "p2_latency_repeat",
        "label": "base",
        "gpu": "0",
        "width": [64, 128, 256],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_work_2e_base",
        "config_id_tuned": "calibration_h800_tvm_pyramid_base_fp16_metaschedule_tuned",
        "config_id_default": "calibration_h800_tvm_pyramid_base_fp16_default",
        "candidate_id": "calibration:pyramid_lidar:base",
        "software_point_id": "pyramid_lidar:backbone:w64x128x256:fp16",
    },
    {
        "phase": "p2_latency_repeat",
        "label": "p50",
        "gpu": "0",
        "width": [32, 64, 128],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/p50_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_work_2e_p50",
        "config_id_tuned": "calibration_h800_tvm_pyramid_p50_fp16_metaschedule_tuned",
        "config_id_default": "calibration_h800_tvm_pyramid_p50_fp16_default",
        "candidate_id": "calibration:pyramid_lidar:p50",
        "software_point_id": "pyramid_lidar:backbone:w32x64x128:fp16",
    },
    {
        "phase": "p2_latency_repeat",
        "label": "trap25",
        "gpu": "0",
        "width": [48, 96, 192],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/trap25_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_work_2e_trap25",
        "config_id_tuned": "calibration_h800_tvm_pyramid_trap25_fp16_metaschedule_tuned",
        "config_id_default": "calibration_h800_tvm_pyramid_trap25_fp16_default",
        "candidate_id": "calibration:pyramid_lidar:trap25",
        "software_point_id": "pyramid_lidar:backbone:w48x96x192:fp16",
    },
    {
        "phase": "p2_latency_repeat",
        "label": "mix_b",
        "gpu": "1",
        "width": [48, 64, 256],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/mix_b_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_b",
        "config_id_tuned": "calibration_h800_tvm_pyramid_mix_b_fp16_metaschedule_tuned",
        "config_id_default": "calibration_h800_tvm_pyramid_mix_b_fp16_default",
        "candidate_id": "calibration:pyramid_lidar:mix_b",
        "software_point_id": "pyramid_lidar:backbone:w48x64x256:fp16",
    },
    {
        "phase": "p2_latency_repeat",
        "label": "mix_d",
        "gpu": "1",
        "width": [48, 128, 128],
        "onnx": "/exdata/jichengzhi/s2_tvm/models/mix_d_backbone.onnx",
        "work_dir": "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_d",
        "config_id_tuned": "calibration_h800_tvm_pyramid_mix_d_fp16_metaschedule_tuned",
        "config_id_default": "calibration_h800_tvm_pyramid_mix_d_fp16_default",
        "candidate_id": "calibration:pyramid_lidar:mix_d",
        "software_point_id": "pyramid_lidar:backbone:w48x128x128:fp16",
    },
]

LATENCY_NEW_JOBS = [
    {
        "phase": "p1_latency_new",
        "label": "p50b2_136",
        "gpu": "0",
        "width": [32, 64, 136],
        "onnx_candidates": [
            "/exdata/jichengzhi/s2_tvm/models/p50b2_136_backbone.onnx",
            "/exdata/jichengzhi/s2_tvm/models/p50b2_136.onnx",
            "/home/jichengzhi/UniV2X/models/stage_a_cache/p50b2_136.onnx",
        ],
        "work_dir_candidates": [
            "/exdata/jichengzhi/s2_tvm/ms_work_2e_p50b2_136",
            "/exdata/jichengzhi/s2_tvm/ms_bumped_p50b2_136",
            "/exdata/jichengzhi/s2_tvm/ms_p50b2_136",
        ],
        "config_id_tuned": "calibration_h800_tvm_pyramid_p50b2_136_fp16_metaschedule_tuned",
        "config_id_default": "calibration_h800_tvm_pyramid_p50b2_136_fp16_default",
        "candidate_id": "calibration:pyramid_lidar:p50b2_136",
        "software_point_id": "pyramid_lidar:backbone:w32x64x136:fp16",
    },
    {
        "phase": "p1_latency_new",
        "label": "mix_e_64_64_128",
        "gpu": "1",
        "width": [64, 64, 128],
        "onnx_candidates": [
            "/exdata/jichengzhi/s2_tvm/models/mix_e_64_64_128_backbone.onnx",
            "/exdata/jichengzhi/s2_tvm/models/64_64_128_backbone.onnx",
            "/exdata/jichengzhi/s2_tvm/models/s1s2_64_64_128_backbone.onnx",
        ],
        "work_dir_candidates": [
            "/exdata/jichengzhi/s2_tvm/ms_bumped_mix_e_64_64_128",
            "/exdata/jichengzhi/s2_tvm/ms_work_2e_64_64_128",
            "/exdata/jichengzhi/s2_tvm/ms_64_64_128",
        ],
        "config_id_tuned": "calibration_h800_tvm_pyramid_mix_e_64_64_128_fp16_metaschedule_tuned",
        "config_id_default": "calibration_h800_tvm_pyramid_mix_e_64_64_128_fp16_default",
        "candidate_id": "calibration:pyramid_lidar:mix_e_64_64_128",
        "software_point_id": "pyramid_lidar:backbone:w64x64x128:fp16",
    },
]


def now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dirs() -> None:
    for root in (CALIB_ROOT, SMOKE_ROOT):
        for sub in ("jobs", "raw", "logs", "compare", "latency", "energy"):
            (root / sub).mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"{cmd} failed")
    return proc.stdout


def configure_tvm_env(gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ld_existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*LD_PREFIX, ld_existing] if ld_existing else LD_PREFIX)
    path_existing = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{CUDA_BIN}:{path_existing}" if path_existing else CUDA_BIN


def gpu_snapshot() -> dict[str, tuple[int, int]]:
    out = run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    snap = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            snap[parts[0]] = (int(float(parts[1])), int(float(parts[2])))
    return snap


def target_has_pmon(gpu: str) -> bool:
    proc = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "um"], capture_output=True, text=True, check=False)
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[0] == gpu and parts[1] != "-":
            return True
    return False


def wait_gpu_idle(gpu: str, raw: Path, *, required_samples: int = 3, interval_s: int = 5, timeout_s: int = 900) -> list[dict[str, object]]:
    deadline = time.time() + timeout_s
    clean = 0
    history = []
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
                with (raw / "nvidia_smi_pmon_preflight.txt").open("w", encoding="utf-8") as fh:
                    subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "um"], stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)
                return history
        else:
            clean = 0
        time.sleep(interval_s)
    raise RuntimeError(f"GPU{gpu} not idle before timeout; tail={history[-5:]}")


def resolve_first(paths: list[str]) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    raise RuntimeError(f"none of the candidate paths exists: {paths}")


def read_inputs(onnx_model):
    init = {i.name for i in onnx_model.graph.initializer}
    return {
        i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
        for i in onnx_model.graph.input
        if i.name not in init
    }


def summarize_times(results_s: list[float]) -> dict[str, object]:
    vals = [float(x) * 1e6 for x in results_s]
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    p50 = vals_sorted[mid] if len(vals_sorted) % 2 else (vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0
    return {
        "us": round(p50, 3),
        "mean_us": round(sum(vals) / len(vals), 3),
        "min_us": round(min(vals), 3),
        "max_us": round(max(vals), 3),
        "repeats_us": [round(v, 3) for v in vals],
    }


def time_vm(vm, args, dev, number: int, repeat: int) -> dict[str, object]:
    for _ in range(WARMUP_LATENCY):
        vm["main"](*args)
        dev.sync()
    vf = vm.time_evaluator("main", dev, number=number, repeat=repeat)
    result = vf(*args)
    return summarize_times(list(result.results))


def latency_payload(result: dict[str, object], raw: Path, schedule: str) -> dict[str, object]:
    prefix = "default" if schedule == "default" else "tuned"
    strategy = "relax_default" if schedule == "default" else "relax_metaschedule_reuse_existing_ms_db"
    build_key = "build_default_s" if schedule == "default" else "build_tuned_s"
    return {
        "batch_size": 1,
        "build_status": "success",
        "build_time_s": result.get(build_key),
        "input_shape": result.get("input_shape", {}),
        "latency_max_us": result.get(f"{prefix}_max_us"),
        "latency_mean_us": result.get(f"{prefix}_mean_us"),
        "latency_min_us": result.get(f"{prefix}_min_us"),
        "latency_p50_us": result.get(f"{prefix}_us"),
        "measure_iters": MEASURE_LATENCY,
        "notes": f"{result['phase']}; {result['label']}; {schedule}; backbone-only; no fresh tune",
        "provenance": f"H800 TVM {result['phase']} measured on clean target GPU; {schedule}; reuse existing artifacts",
        "raw_artifact": str(raw),
        "repeat": REPEAT_LATENCY,
        "run_id": result.get("run_id"),
        "source_files": [result["onnx_path"], result["work_dir"], str(raw / "latency_result.json")],
        "tvm_strategy": strategy,
        "tvm_target": "cuda",
        "warmup_iters": WARMUP_LATENCY,
    }


def call_generator(cmd: list[str]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"generator failed: {cmd}")


def append_latency_rows(job: dict[str, object], raw: Path, result: dict[str, object]) -> None:
    for schedule, config_key in (("default", "config_id_default"), ("metaschedule_tuned", "config_id_tuned")):
        payload_path = raw / f"measurement_payload_{schedule}.json"
        write_text(payload_path, json.dumps(latency_payload(result, raw, schedule), indent=2, sort_keys=True) + "\n")
        call_generator(
            [
                GEN_PY,
                "scripts/stage2_generate_latency_lut.py",
                "--job-id",
                f"latency:{job['phase']}_{job['label']}_{schedule}",
                "--model",
                "pyramid_lidar",
                "--config-id",
                str(job[config_key]),
                "--candidate-id",
                str(job["candidate_id"]),
                "--software-point-id",
                str(job["software_point_id"]),
                "--dense-stage",
                "backbone",
                "--optimized-scope",
                "backbone_only",
                "--width",
                ",".join(str(x) for x in job["width"]),
                "--quant-policy",
                "fp16",
                "--schedule-policy",
                schedule,
                "--backend",
                "h800_tvm",
                "--manifest-digest",
                f"{job['phase']}_20260625",
                "--run-id",
                f"{result['run_id']}_{schedule}",
                "--warmup-iters",
                str(WARMUP_LATENCY),
                "--measure-iters",
                str(MEASURE_LATENCY),
                "--repeat",
                str(REPEAT_LATENCY),
                "--batch-size",
                "1",
                "--measurement-command-json",
                json.dumps(["/bin/cat", str(payload_path)]),
                "--out-jsonl",
                str(CALIB_ROOT / "latency/latency_lut_rows_v1.jsonl"),
            ]
        )


def measure_latency(job: dict[str, object]) -> dict[str, object]:
    gpu = str(job["gpu"])
    label = str(job["label"])
    run_id = f"calib_h800_tvm_pyramid_{label}_{job['phase']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw = CALIB_ROOT / "raw" / run_id
    raw.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": "lut_job_state_row_v1",
        "job_id": f"latency:{job['phase']}:{label}",
        "status": "running",
        "attempt": 1,
        "started_at": now_local(),
        "finished_at": None,
        "log_path": str(CALIB_ROOT / "logs" / f"{label}.{job['phase']}.log.json"),
        "output_row_id": None,
    }
    append_jsonl(STATE_PATH, state)
    result: dict[str, object] = {
        "schema": "stage2_h800_calibration_latency_result_v1",
        "status": "started",
        "phase": job["phase"],
        "model": "pyramid_lidar",
        "label": label,
        "gpu": gpu,
        "run_id": run_id,
        "onnx_path": job.get("onnx"),
        "work_dir": job.get("work_dir"),
        "width": job["width"],
        "warmup_iters": WARMUP_LATENCY,
        "measure_iters": MEASURE_LATENCY,
        "repeat": REPEAT_LATENCY,
        "batch_size": 1,
    }
    try:
        result["onnx_path"] = result["onnx_path"] or resolve_first(job["onnx_candidates"])
        result["work_dir"] = result["work_dir"] or resolve_first(job["work_dir_candidates"])
        configure_tvm_env(gpu)
        idle_history = wait_gpu_idle(gpu, raw)
        result["idle_history_tail"] = idle_history[-3:]
        write_text(raw / "hostname.txt", socket.gethostname() + "\n")
        write_text(raw / "env.json", json.dumps({"CUDA_VISIBLE_DEVICES": gpu, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""), "python": sys.executable}, indent=2) + "\n")
        write_text(raw / "command.json", json.dumps({"schema": "stage2_h800_p1_p2_latency_command_v1", **{k: result[k] for k in ("phase", "label", "gpu", "run_id", "onnx_path", "work_dir")}}, indent=2) + "\n")

        import numpy as np
        import onnx
        import tvm
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        model = onnx.load(str(result["onnx_path"]))
        shapes = read_inputs(model)
        result["input_shape"] = {k: list(v) for k, v in shapes.items()}
        rng = np.random.RandomState(0)
        feeds_np = {k: rng.rand(*v).astype("float32") for k, v in shapes.items()}
        mod0 = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in shapes]

        t0 = time.time()
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target="cuda")
        result["build_default_s"] = round(time.time() - t0, 3)
        vm = relax.VirtualMachine(ex, dev)
        default_stats = time_vm(vm, args, dev, MEASURE_LATENCY, REPEAT_LATENCY)
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
        t1 = time.time()
        with target, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
            scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=str(result["work_dir"]))(modt)
            ex2 = tvm.compile(scheduled, target=target)
        result["build_tuned_s"] = round(time.time() - t1, 3)
        vm2 = relax.VirtualMachine(ex2, dev)
        tuned_stats = time_vm(vm2, args, dev, MEASURE_LATENCY, REPEAT_LATENCY)
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
        append_latency_rows(job, raw, result)
        print(json.dumps({"event": "latency_success", "label": label, "phase": job["phase"], "default_us": result["default_us"], "tuned_us": result["tuned_us"], "run_id": run_id}), flush=True)
        final_state = dict(state, status="succeeded", finished_at=now_local())
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        write_text(raw / "latency_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "latency_failed", "label": label, "phase": job["phase"], "error": repr(exc), "run_id": run_id}), flush=True)
        final_state = dict(state, status="failed", finished_at=now_local(), error=repr(exc))
        maybe_quarantine(job, state["job_id"], repr(exc))
    append_jsonl(STATE_PATH, final_state)
    write_text(CALIB_ROOT / "logs" / f"{label}.{job['phase']}.log.json", json.dumps({"schema": "lut_job_log_v1", "job_id": state["job_id"], "raw": str(raw), "result": result, "returncode": 0 if result.get("status") == "success" else 1}, indent=2, sort_keys=True) + "\n")
    return result


def query_power_w(gpu: str) -> float:
    out = run_text(["nvidia-smi", f"--id={gpu}", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    return float(out.strip().splitlines()[0].strip())


def sample_power(gpu: str, seconds: float, interval: float) -> list[tuple[float, float]]:
    rows = []
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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_s", "power_w"])
        for ts, watt in rows:
            writer.writerow([f"{ts:.6f}", f"{watt:.3f}"])


def power_stats(rows: list[tuple[float, float]]) -> dict[str, float | None]:
    vals = [w for _, w in rows]
    if not vals:
        return {"avg": None, "p50": None, "p90": None}
    vals_sorted = sorted(vals)
    return {
        "avg": sum(vals) / len(vals),
        "p50": vals_sorted[len(vals_sorted) // 2],
        "p90": vals_sorted[min(len(vals_sorted) - 1, int(0.9 * (len(vals_sorted) - 1)))],
    }


def append_energy_row(job: dict[str, object], raw: Path, result: dict[str, object]) -> None:
    telemetry = {
        "run_id": result["run_id"],
        "latency_run_id": job["latency_run_id"],
        "latency_config_id": job["config_id"],
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
        "warmup_iters": WARMUP_ENERGY,
        "measure_iters": MEASURE_ENERGY,
        "repeat": REPEAT_ENERGY,
        "provenance": f"H800 power telemetry {job['phase']} aligned with tuned TVM VM",
        "source_files": [
            job["onnx"],
            job["work_dir"],
            str(raw / "energy_result.json"),
            str(raw / "idle_power_samples.csv"),
            str(raw / "active_power_samples.csv"),
        ],
        "raw_artifact": str(raw),
        "notes": f"{job['phase']} energy telemetry; clean target GPU preflight; not full-model energy",
    }
    payload_path = raw / "telemetry_payload.json"
    write_text(payload_path, json.dumps(telemetry, indent=2, sort_keys=True) + "\n")
    call_generator(
        [
            GEN_PY,
            "scripts/stage2_generate_energy_lut.py",
            "--job-id",
            f"energy:{job['phase']}:{job['label']}",
            "--model",
            "pyramid_lidar",
            "--config-id",
            str(job["config_id"]),
            "--candidate-id",
            str(job["candidate_id"]),
            "--software-point-id",
            str(job["software_point_id"]),
            "--dense-stage",
            "backbone",
            "--optimized-scope",
            "backbone_only",
            "--width",
            ",".join(str(x) for x in job["width"]),
            "--quant-policy",
            "fp16",
            "--schedule-policy",
            "metaschedule_tuned",
            "--backend",
            "h800_tvm_power_telemetry",
            "--manifest-digest",
            f"{job['phase']}_20260625",
            "--run-id",
            str(result["run_id"]),
            "--latency-run-id",
            str(job["latency_run_id"]),
            "--warmup-iters",
            str(WARMUP_ENERGY),
            "--measure-iters",
            str(MEASURE_ENERGY),
            "--repeat",
            str(REPEAT_ENERGY),
            "--telemetry-command-json",
            json.dumps(["/bin/cat", str(payload_path)]),
            "--out-jsonl",
            str(CALIB_ROOT / "energy/energy_lut_rows_v1.jsonl"),
        ]
    )


def measure_energy(job: dict[str, object]) -> dict[str, object]:
    gpu = str(job["gpu"])
    label = str(job["label"])
    run_id = f"calib_h800_tvm_energy_pyramid_{label}_{job['phase']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    raw = CALIB_ROOT / "raw" / run_id
    raw.mkdir(parents=True, exist_ok=True)
    state = {
        "schema": "lut_job_state_row_v1",
        "job_id": f"energy:{job['phase']}:{label}",
        "status": "running",
        "attempt": 1,
        "started_at": now_local(),
        "finished_at": None,
        "log_path": str(CALIB_ROOT / "logs" / f"energy.{label}.{job['phase']}.log.json"),
        "output_row_id": None,
    }
    append_jsonl(STATE_PATH, state)
    result: dict[str, object] = {
        "schema": "stage2_h800_energy_result_v1",
        "status": "started",
        "phase": job["phase"],
        "gpu": gpu,
        "model": "pyramid_lidar",
        "label": label,
        "width": job["width"],
        "onnx_path": job["onnx"],
        "work_dir": job["work_dir"],
        "run_id": run_id,
        "latency_run_id": job["latency_run_id"],
        "warmup_iters": WARMUP_ENERGY,
        "measure_iters": MEASURE_ENERGY,
        "repeat": REPEAT_ENERGY,
        "batch_size": 1,
    }
    try:
        configure_tvm_env(gpu)
        idle_history = wait_gpu_idle(gpu, raw)
        result["idle_history_tail"] = idle_history[-3:]
        write_text(raw / "hostname.txt", socket.gethostname() + "\n")
        write_text(raw / "env.json", json.dumps({"CUDA_VISIBLE_DEVICES": gpu, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""), "python": sys.executable}, indent=2) + "\n")
        write_text(raw / "command.json", json.dumps({"schema": "stage2_h800_p1_energy_command_v1", **{k: result[k] for k in ("phase", "label", "gpu", "run_id", "onnx_path", "work_dir")}}, indent=2) + "\n")
        idle_rows = sample_power(gpu, 5.0, 0.2)
        write_power_csv(raw / "idle_power_samples.csv", idle_rows)
        idle = power_stats(idle_rows)

        import numpy as np
        import onnx
        import tvm
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        model = onnx.load(str(job["onnx"]))
        shapes = read_inputs(model)
        result["input_shape"] = {k: list(v) for k, v in shapes.items()}
        rng = np.random.RandomState(0)
        feeds_np = {k: rng.rand(*v).astype("float32") for k, v in shapes.items()}
        mod0 = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
        seq = tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        t_build = time.time()
        with target, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
            scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=str(job["work_dir"]))(modt)
            ex = tvm.compile(scheduled, target=target)
        result["build_time_s"] = round(time.time() - t_build, 3)
        vm = relax.VirtualMachine(ex, dev)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in shapes]
        for _ in range(WARMUP_ENERGY):
            vm["main"](*args)
        dev.sync()

        samples: list[tuple[float, float]] = []
        stop = {"value": False}

        def poller() -> None:
            while not stop["value"]:
                try:
                    samples.append((time.time(), query_power_w(gpu)))
                except Exception:
                    pass
                time.sleep(0.05)

        thread = threading.Thread(target=poller, daemon=True)
        thread.start()
        t0 = time.time()
        vm["main"](*args)
        vf = vm.time_evaluator("main", dev, number=MEASURE_ENERGY, repeat=1)
        timing = vf(*args)
        dev.sync()
        active_duration = time.time() - t0
        stop["value"] = True
        thread.join(timeout=1.0)
        write_power_csv(raw / "active_power_samples.csv", samples)
        active = power_stats(samples)
        time_eval_s = float(timing.results[0])
        joule = max(0.0, ((active["avg"] or 0.0) - (idle["avg"] or 0.0)) * active_duration / MEASURE_ENERGY)
        result.update(
            {
                "status": "success",
                "active_duration_s": round(active_duration, 6),
                "active_samples": len(samples),
                "idle_samples": len(idle_rows),
                "idle_watt_avg": round(float(idle["avg"] or 0.0), 4),
                "watt_avg": round(float(active["avg"] or 0.0), 4),
                "watt_p50": round(float(active["p50"] or 0.0), 4),
                "watt_p90": round(float(active["p90"] or 0.0), 4),
                "sample_window_ms": int(round(active_duration * 1000)),
                "time_evaluator_s": round(time_eval_s, 6),
                "joule_per_inference": round(joule, 8),
            }
        )
        write_text(raw / "energy_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        append_energy_row(job, raw, result)
        print(json.dumps({"event": "energy_success", "label": label, "joule_per_inference": result["joule_per_inference"], "run_id": run_id}), flush=True)
        final_state = dict(state, status="succeeded", finished_at=now_local())
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        write_text(raw / "energy_result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "energy_failed", "label": label, "error": repr(exc), "run_id": run_id}), flush=True)
        final_state = dict(state, status="failed", finished_at=now_local(), error=repr(exc))
        maybe_quarantine(job, state["job_id"], repr(exc))
    append_jsonl(STATE_PATH, final_state)
    write_text(CALIB_ROOT / "logs" / f"energy.{label}.{job['phase']}.log.json", json.dumps({"schema": "lut_job_log_v1", "job_id": state["job_id"], "raw": str(raw), "result": result, "returncode": 0 if result.get("status") == "success" else 1}, indent=2, sort_keys=True) + "\n")
    return result


def maybe_quarantine(job: dict[str, object], job_id: str, reason: str) -> None:
    text = reason.lower()
    if "illegal memory access" not in text and "cuda_error_illegal_address" not in text:
        return
    row = {
        "schema": "lut_bad_db_quarantine_row_v1",
        "job_id": job_id,
        "config_id": str(job.get("config_id_tuned") or job.get("config_id") or ""),
        "model": "pyramid_lidar",
        "lut_kind": "energy" if job_id.startswith("energy:") else "latency",
        "job_type": "generate_energy_lut" if job_id.startswith("energy:") else "generate_latency_lut",
        "status": "active",
        "failure_reason": reason,
        "created_at": now_utc(),
    }
    append_jsonl(QUARANTINE_PATH, row)


def run_child(kind: str, index: int) -> int:
    if kind == "energy":
        result = measure_energy(ENERGY_JOBS[index])
    elif kind == "latency_repeat":
        result = measure_latency(LATENCY_REPEAT_JOBS[index])
    elif kind == "latency_new":
        result = measure_latency(LATENCY_NEW_JOBS[index])
    else:
        raise RuntimeError(f"unknown child kind: {kind}")
    return 0 if result.get("status") == "success" else 1


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    ld_existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([*LD_PREFIX, ld_existing] if ld_existing else LD_PREFIX)
    path_existing = env.get("PATH", "")
    env["PATH"] = f"{CUDA_BIN}:{path_existing}" if path_existing else CUDA_BIN
    env["PYTHONPATH"] = str(ROOT)
    return env


def run_subprocess_job(kind: str, index: int, label: str) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-kind",
        kind,
        "--child-index",
        str(index),
    ]
    proc = subprocess.run(command, cwd=ROOT, env=child_env(), capture_output=True, text=True, check=False)
    log_path = CALIB_ROOT / "logs" / f"p1_p2_child_{kind}_{label}.log"
    write_text(
        log_path,
        json.dumps(
            {
                "schema": "stage2_h800_p1_p2_child_log_v1",
                "kind": kind,
                "index": index,
                "label": label,
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return {
        "kind": kind,
        "index": index,
        "label": label,
        "returncode": proc.returncode,
        "log_path": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-kind", choices=("energy", "latency_repeat", "latency_new"))
    parser.add_argument("--child-index", type=int)
    args = parser.parse_args()

    ensure_dirs()
    if args.child_kind is not None:
        if args.child_index is None:
            raise RuntimeError("--child-index is required with --child-kind")
        return run_child(args.child_kind, args.child_index)

    results = []
    for idx, job in enumerate(ENERGY_JOBS):
        results.append(run_subprocess_job("energy", idx, str(job["label"])))
        time.sleep(10)
    for idx, job in enumerate(LATENCY_REPEAT_JOBS):
        results.append(run_subprocess_job("latency_repeat", idx, str(job["label"])))
        time.sleep(10)
    for idx, job in enumerate(LATENCY_NEW_JOBS):
        results.append(run_subprocess_job("latency_new", idx, str(job["label"])))
        time.sleep(10)
    summary = {
        "schema": "stage2_h800_p1_p2_gpu012_summary_v1",
        "created_at": now_local(),
        "results": results,
    }
    write_text(CALIB_ROOT / "compare/p1_p2_h800_gpu012_summary_v1.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if all(int(item.get("returncode", 1)) == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
