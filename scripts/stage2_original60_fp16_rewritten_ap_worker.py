#!/usr/bin/env python3
"""Run original60 FP16 AP-shape rewritten-engine smoke/full-val jobs on one GPU lane."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
)
DEFAULT_RAW_ROOT = Path("/exdata/jichengzhi/s2_tvm/fp16_rewritten_ap_original60_20260630")
DEFAULT_CKPT_ROOT = Path("/exdata/jichengzhi/heal_research/checkpoints/stage1")
DEFAULT_ROWS_OUT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "rows/fp16_rewritten_original60_ap_rows_v1.jsonl"
)
DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_TVM_PYTHON = "/exdata/jichengzhi/tvm310/bin/python"
DEFAULT_HEAL_ROOT = "/exdata/jichengzhi/heal_research/HEAL"
DEFAULT_TVM_LD_LIBRARY_PATH = (
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--labels", required=True, help="comma-separated labels for this lane")
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT))
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--tvm-python", default=DEFAULT_TVM_PYTHON)
    parser.add_argument("--tvm-ld-library-path", default=DEFAULT_TVM_LD_LIBRARY_PATH)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--master-port-base", type=int, default=29830)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-epoch", type=int, default=25)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--full-reps", type=int, default=30)
    parser.add_argument("--input-shape", default="2,64,256,256")
    parser.add_argument("--mode", choices=["smoke", "full-val"], default="smoke")
    return parser.parse_args()


def safe_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("_") or "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def completion_jobs(queue_path: Path) -> dict[str, dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(queue_path):
        if str(row.get("precision") or "") != "fp16":
            continue
        label = str(row.get("label") or "")
        if label:
            jobs[label] = row
    return jobs


def ckpt_dir_for(ckpt_root: Path, label: str) -> Path:
    return ckpt_root / f"Pyramid_DAIR_m1_stage2_ap_{label}_2026_06_28"


def label_raw_dir(raw_root: Path, label: str, gpu_id: int) -> Path:
    return raw_root / f"{safe_label(label)}_gpu{gpu_id}"


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=str(cwd), env=env, stdout=stdout, stderr=stderr, text=True, check=False)
    return int(proc.returncode)


def ps_alive(pid: int | None) -> bool:
    if not pid:
        return False
    proc = subprocess.run(["ps", "-p", str(pid), "-o", "pid=,stat="], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return False
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == str(pid):
            return "Z" not in parts[1]
    return False


def best_checkpoint_epoch(ckpt_dir: Path) -> tuple[Path | None, int]:
    best_path: Path | None = None
    best_epoch = -1
    for path in list(ckpt_dir.glob("net_epoch_bestval_at*.pth")) + list(ckpt_dir.glob("net_epoch*.pth")):
        match = re.search(r"(?:bestval_at|net_epoch)(\d+)", path.name)
        if match and int(match.group(1)) > best_epoch:
            best_epoch = int(match.group(1))
            best_path = path
    return best_path, best_epoch


def build_train_command(args: argparse.Namespace, label: str, width: list[int], ckpt_dir: Path, raw_dir: Path, index: int) -> list[str]:
    return [
        args.env_python,
        str(ROOT / "scripts/stage2_original60_fp16_train_launcher.py"),
        "--label",
        label,
        "--width",
        ",".join(str(x) for x in width),
        "--gpu-id",
        str(args.gpu_id),
        "--master-port",
        str(int(args.master_port_base) + int(args.gpu_id) + index * 8),
        "--ckpt-dir",
        str(ckpt_dir),
        "--raw-dir",
        str(raw_dir / "checkpoint_generation"),
        "--env-python",
        args.env_python,
        "--heal-root",
        args.heal_root,
    ]


def ensure_checkpoint(args: argparse.Namespace, label: str, width: list[int], ckpt_dir: Path, raw_dir: Path, index: int) -> tuple[Path, int]:
    best_path, best_epoch = best_checkpoint_epoch(ckpt_dir)
    if best_path is not None and best_epoch >= int(args.min_epoch):
        return best_path, best_epoch

    train_raw = raw_dir / "checkpoint_generation"
    command = build_train_command(args, label, width, ckpt_dir, raw_dir, index)
    env = build_env(args)
    rc = run_logged(
        command,
        cwd=ROOT,
        env=env,
        stdout_path=train_raw / "launcher_stdout.txt",
        stderr_path=train_raw / "launcher_stderr.txt",
    )
    write_json(train_raw / "launcher_dispatch.json", {"command": command, "returncode": rc})
    if rc != 0:
        raise RuntimeError(f"checkpoint launcher failed rc={rc}")

    pid_path = train_raw / "train_runner.pid"
    train_pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.exists() else 0
    while ps_alive(train_pid):
        best_path, best_epoch = best_checkpoint_epoch(ckpt_dir)
        write_json(
            train_raw / "checkpoint_wait_status.json",
            {"label": label, "train_pid": train_pid, "best_checkpoint": str(best_path) if best_path else None, "best_epoch": best_epoch},
        )
        if best_path is not None and best_epoch >= int(args.min_epoch):
            return best_path, best_epoch
        time.sleep(int(args.poll_seconds))

    best_path, best_epoch = best_checkpoint_epoch(ckpt_dir)
    if best_path is None or best_epoch < int(args.min_epoch):
        raise RuntimeError(f"checkpoint incomplete: best_epoch={best_epoch}, min_epoch={args.min_epoch}")
    return best_path, best_epoch


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    # Child scripts already accept physical --gpu-id/--gpu and manage CUDA_VISIBLE_DEVICES
    # themselves. Masking here would turn a physical id such as 3 into an invalid local
    # ordinal for scripts that call torch.cuda.set_device(args.gpu_id).
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(Path(args.heal_root).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def build_tvm_env(args: argparse.Namespace) -> dict[str, str]:
    env = build_env(args)
    parts = [str(args.tvm_ld_library_path)]
    nvlibs = Path("/exdata/jichengzhi/tvm_nvlibs.path")
    if nvlibs.exists():
        parts.append(nvlibs.read_text(encoding="utf-8").strip())
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(item for item in parts if item)
    return env


def build_export_command(args: argparse.Namespace, label: str, ckpt_dir: Path, raw_dir: Path) -> tuple[list[str], Path]:
    onnx_dir = raw_dir / "apshape_onnx"
    onnx_path = onnx_dir / f"{safe_label(label)}_apshape_multiscale.onnx"
    report_path = onnx_dir / f"{safe_label(label)}_apshape_multiscale_export_report.json"
    return [
        args.env_python,
        str(ROOT / "scripts/stage2_h800_export_checkpoint_multiscale_onnx.py"),
        "--label",
        label,
        "--ckpt-dir",
        str(ckpt_dir),
        "--out",
        str(onnx_path),
        "--report-json",
        str(report_path),
        "--heal-root",
        args.heal_root,
        "--gpu-id",
        str(args.gpu_id),
        "--input-shape",
        args.input_shape,
    ], onnx_path


def rewrite_report_path(raw_dir: Path, label: str) -> Path:
    return raw_dir / "rewrite_exports" / f"fp16_{safe_label(label)}_full_engine_group_conv_rewrite_latest.json"


def build_rewrite_command(args: argparse.Namespace, label: str, onnx_path: Path, raw_dir: Path) -> tuple[list[str], Path]:
    export_dir = raw_dir / "rewrite_exports"
    report_path = rewrite_report_path(raw_dir, label)
    command = [
        args.tvm_python,
        str(ROOT / "scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py"),
        "--mode",
        "full-engine-group-conv-rewrite",
        "--label",
        label,
        "--gpu",
        str(args.gpu_id),
        "--full-reps",
        str(args.full_reps),
        "--raw-dir",
        str(raw_dir / "rewrite_raw"),
        "--export-dir",
        str(export_dir),
        "--onnx",
        str(onnx_path),
        "--cast-fp16-source",
    ]
    return command, report_path


def assert_rewrite_gate(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rewritten = report.get("rewritten_full_engine") or {}
    export = rewritten.get("export_library") or {}
    if report.get("status") != "success":
        raise RuntimeError(f"rewrite status is {report.get('status')}")
    if not rewritten.get("tensorcore_gate"):
        raise RuntimeError("rewrite tensorcore_gate is false")
    if export.get("status") != "success" or not export.get("path"):
        raise RuntimeError("rewrite export library missing")
    return report


def build_bridge_command(args: argparse.Namespace, label: str, ckpt_dir: Path, rewrite_report: Path, raw_dir: Path) -> tuple[list[str], Path]:
    samples = None if args.mode == "full-val" else int(args.num_samples)
    bridge_raw = raw_dir / ("ap_fullval" if args.mode == "full-val" else f"ap_smoke_{args.num_samples}")
    export_report = raw_dir / "bridge_exports" / (
        f"fp16_{safe_label(label)}_rewritten_ap_fullval_latest.json"
        if args.mode == "full-val"
        else f"fp16_{safe_label(label)}_rewritten_ap_smoke_latest.json"
    )
    command = [
        args.env_python,
        str(ROOT / "scripts/stage2_h800_fp16_rewritten_activation_bridge.py"),
        "--label",
        label,
        "--ckpt-dir",
        str(ckpt_dir),
        "--raw-dir",
        str(bridge_raw),
        "--heal-root",
        args.heal_root,
        "--gpu-id",
        str(args.gpu_id),
        "--rewrite-report",
        str(rewrite_report),
        "--persistent-worker",
        "--tvm-python",
        args.tvm_python,
        "--full-ap-min-samples",
        "1" if args.mode == "smoke" else "1789",
        "--keep-detailed-samples",
        "1",
        "--export-report-json",
        str(export_report),
    ]
    if samples is not None:
        command += ["--num-samples", str(samples)]
    return command, export_report


def assert_bridge_gate(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise RuntimeError(f"bridge status is {report.get('status')}")
    if int(report.get("processed_samples") or 0) <= 0:
        raise RuntimeError("bridge processed no samples")
    if not report.get("smoke_gate_passed"):
        raise RuntimeError("bridge smoke gate failed")
    return report


def run_label(args: argparse.Namespace, label: str, job: dict[str, Any], index: int) -> dict[str, Any]:
    width = [int(x) for x in job.get("width") or []]
    if not width:
        raise RuntimeError(f"missing width for {label}")
    raw_dir = label_raw_dir(Path(args.raw_root), label, args.gpu_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_dir_for(Path(args.ckpt_root), label)
    ckpt_path, ckpt_epoch = ensure_checkpoint(args, label, width, ckpt_dir, raw_dir, index)

    env = build_env(args)
    export_cmd, onnx_path = build_export_command(args, label, ckpt_dir, raw_dir)
    rc = run_logged(export_cmd, cwd=ROOT, env=env, stdout_path=raw_dir / "apshape_onnx/export_stdout.txt", stderr_path=raw_dir / "apshape_onnx/export_stderr.txt")
    if rc != 0 or not onnx_path.exists():
        raise RuntimeError(f"ONNX export failed rc={rc}")

    rewrite_cmd, rewrite_report = build_rewrite_command(args, label, onnx_path, raw_dir)
    rc = run_logged(rewrite_cmd, cwd=ROOT, env=build_tvm_env(args), stdout_path=raw_dir / "rewrite_stdout.txt", stderr_path=raw_dir / "rewrite_stderr.txt")
    if rc != 0 or not rewrite_report.exists():
        raise RuntimeError(f"rewrite failed rc={rc}")
    rewrite_payload = assert_rewrite_gate(rewrite_report)

    bridge_cmd, bridge_report = build_bridge_command(args, label, ckpt_dir, rewrite_report, raw_dir)
    rc = run_logged(bridge_cmd, cwd=ROOT, env=env, stdout_path=raw_dir / "bridge_stdout.txt", stderr_path=raw_dir / "bridge_stderr.txt")
    if rc != 0 or not bridge_report.exists():
        raise RuntimeError(f"bridge failed rc={rc}")
    bridge_payload = assert_bridge_gate(bridge_report)

    return {
        "schema": "fp16_rewritten_original60_ap_row_v1",
        "label": label,
        "precision": "fp16_rewritten",
        "measurement_status": "measured",
        "mode": args.mode,
        "gpu_id": int(args.gpu_id),
        "width": width,
        "ckpt_dir": str(ckpt_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_epoch": int(ckpt_epoch),
        "raw_artifact": str(raw_dir),
        "onnx_path": str(onnx_path),
        "rewrite_report": str(rewrite_report),
        "bridge_report": str(bridge_report),
        "rewritten_latency_ms": ((rewrite_payload.get("rewritten_full_engine") or {}).get("latency_mean_ms")),
        "default_latency_ms": ((rewrite_payload.get("default_full_engine") or {}).get("latency_mean_ms")),
        "ap30": bridge_payload.get("ap30"),
        "ap50": bridge_payload.get("ap50"),
        "ap70": bridge_payload.get("ap70"),
        "processed_samples": bridge_payload.get("processed_samples"),
        "pred_nonempty_count": bridge_payload.get("pred_nonempty_count"),
        "created_at": int(time.time()),
    }


def run_lane(args: argparse.Namespace) -> int:
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    jobs = completion_jobs(Path(args.queue))
    raw_root = Path(args.raw_root)
    status_path = raw_root / f"fp16_rewritten_ap_gpu{args.gpu_id}_lane_status.json"
    failures: list[dict[str, Any]] = []
    measured: list[str] = []
    for index, label in enumerate(labels):
        raw_dir = label_raw_dir(raw_root, label, args.gpu_id)
        try:
            job = jobs[label]
            write_json(status_path, {"gpu_id": args.gpu_id, "label": label, "status": "running", "index": index})
            row = run_label(args, label, job, index)
            append_jsonl(Path(args.rows_out), row)
            measured.append(label)
            write_json(status_path, {"gpu_id": args.gpu_id, "label": label, "status": "measured", "index": index, "row": row})
        except Exception as exc:
            failure = {"label": label, "status": "failed", "reason": str(exc), "raw_artifact": str(raw_dir), "index": index}
            failures.append(failure)
            write_json(raw_dir / "fp16_rewritten_ap_worker_blocker.json", failure)
            write_json(status_path, {"gpu_id": args.gpu_id, **failure})
            continue
    write_json(status_path, {"gpu_id": args.gpu_id, "status": "lane_complete", "measured_labels": measured, "failures": failures})
    return 0


def main() -> int:
    return run_lane(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
