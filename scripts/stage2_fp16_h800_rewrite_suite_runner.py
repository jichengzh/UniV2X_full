#!/usr/bin/env python3
"""Run the Stage2 FP16 H800 rewrite diagnostics with auditable logs.

This wrapper intentionally does not import TVM. It launches
stage2_fp16_tensorcore_convblock_and_engine_probe.py with the H800 TVM Python
environment and records command, stdout, stderr, return code, and environment
for each sub-run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/jichengzhi/V2X")
DEFAULT_TVM_PYTHON = Path("/exdata/jichengzhi/tvm310/bin/python")
DEFAULT_EXPORT_DIR = (
    REPO_ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/"
    / "original60_quant_20260627/exports"
)
DEFAULT_RAW_DIR = Path("/exdata/jichengzhi/s2_tvm/fp16_rewrite_suite_20260629")
PROBE_SCRIPT = REPO_ROOT / "scripts/stage2_fp16_tensorcore_convblock_and_engine_probe.py"
GATE_CHECK_SCRIPT = REPO_ROOT / "scripts/stage2_fp16_h800_gate_check.py"


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _build_ld_library_path(existing: str) -> str:
    parts = [
        "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
        "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib",
    ]
    nvlibs = _read_text_if_exists(Path("/exdata/jichengzhi/tvm_nvlibs.path"))
    if nvlibs:
        parts.append(nvlibs)
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _base_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = "/usr/local/cuda-12.2/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = _build_ld_library_path(env.get("LD_LIBRARY_PATH", ""))
    return env


def _command(
    python_bin: Path,
    mode: str,
    gpu: int,
    raw_dir: Path,
    export_dir: Path,
    reps: int,
    full_reps: int,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        str(python_bin),
        str(PROBE_SCRIPT),
        "--mode",
        mode,
        "--gpu",
        str(gpu),
        "--reps",
        str(reps),
        "--full-reps",
        str(full_reps),
        "--raw-dir",
        str(raw_dir),
        "--export-dir",
        str(export_dir),
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def _suite_steps(
    python_bin: Path,
    gpu: int,
    raw_dir: Path,
    export_dir: Path,
    reps: int,
    full_reps: int,
    suite: str,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "name": "self_test_no_tvm",
            "claim": "no_tvm_numpy_mapping_self_test",
            "cmd": _command(python_bin, "self-test-no-tvm", gpu, raw_dir, export_dir, reps, full_reps),
        },
        {
            "name": "rewrite_onnx_1x1",
            "claim": "real_rewritten_1x1_full_engine",
            "cmd": _command(
                python_bin,
                "rewrite-onnx-1x1",
                gpu,
                raw_dir,
                export_dir,
                reps,
                full_reps,
                ["--cast-fp16-source"],
            ),
        },
        {
            "name": "group_conv_accum_compare_repeated",
            "claim": "accumulation_order_diagnostic_not_full_engine",
            "cmd": _command(
                python_bin,
                "group-conv-accum-compare",
                gpu,
                raw_dir,
                export_dir,
                reps,
                full_reps,
                ["--conv-candidate", "fused_conv2d6_add10_relu6"],
            ),
        },
        {
            "name": "group_conv_full_im2col_downsample",
            "claim": "single_layer_full_im2col_tensorcore_positive",
            "cmd": _command(
                python_bin,
                "group-conv-full-im2col",
                gpu,
                raw_dir,
                export_dir,
                reps,
                full_reps,
                ["--conv-candidate", "fused_conv2d4_add10_relu6"],
            ),
        },
        {
            "name": "group_conv_full_im2col_repeated",
            "claim": "single_layer_full_im2col_tensorcore_positive",
            "cmd": _command(
                python_bin,
                "group-conv-full-im2col",
                gpu,
                raw_dir,
                export_dir,
                reps,
                full_reps,
                ["--conv-candidate", "fused_conv2d6_add10_relu6"],
            ),
        },
        {
            "name": "full_engine_group_conv_rewrite_all",
            "claim": "real_rewritten_full_engine_latency",
            "cmd": _command(
                python_bin,
                "full-engine-group-conv-rewrite",
                gpu,
                raw_dir,
                export_dir,
                reps,
                full_reps,
                ["--cast-fp16-source", "--group-conv-rewrite-filter", "all"],
            ),
        },
    ]
    if suite == "full":
        steps.extend(
            [
                {
                    "name": "full_engine_group_conv_rewrite_downsample",
                    "claim": "one_primfunc_rewrite_diagnostic",
                    "cmd": _command(
                        python_bin,
                        "full-engine-group-conv-rewrite",
                        gpu,
                        raw_dir,
                        export_dir,
                        reps,
                        full_reps,
                        ["--cast-fp16-source", "--group-conv-rewrite-filter", "downsample"],
                    ),
                },
                {
                    "name": "full_engine_group_conv_rewrite_repeated",
                    "claim": "one_primfunc_rewrite_diagnostic",
                    "cmd": _command(
                        python_bin,
                        "full-engine-group-conv-rewrite",
                        gpu,
                        raw_dir,
                        export_dir,
                        reps,
                        full_reps,
                        ["--cast-fp16-source", "--group-conv-rewrite-filter", "repeated"],
                    ),
                },
                {
                    "name": "full_engine_group_conv_intermediate_debug",
                    "claim": "intermediate_output_drift_diagnostic",
                    "cmd": _command(
                        python_bin,
                        "full-engine-group-conv-intermediate-debug",
                        gpu,
                        raw_dir,
                        export_dir,
                        reps,
                        full_reps,
                        ["--cast-fp16-source", "--group-conv-rewrite-filter", "repeated"],
                    ),
                },
            ]
        )
    return steps


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _query_nvidia_smi() -> tuple[str, str | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,compute_cap",
        "--format=csv,noheader",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return "", repr(exc)
    if proc.returncode != 0:
        return proc.stdout, proc.stderr.strip() or f"returncode={proc.returncode}"
    return proc.stdout, None


def _parse_gpu_rows(nvidia_smi_output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in nvidia_smi_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            rows.append({"raw": line, "parse_error": "expected:index,name,compute_cap"})
            continue
        try:
            index: int | None = int(parts[0])
        except ValueError:
            index = None
        rows.append(
            {
                "index": index,
                "name": parts[1],
                "compute_cap": parts[2],
                "raw": line,
            }
        )
    return rows


def _preflight_environment(
    *,
    python_bin: Path,
    gpu: int,
    nvidia_smi_output: str | None = None,
    nvidia_smi_error: str | None = None,
) -> dict[str, Any]:
    if nvidia_smi_output is None:
        nvidia_smi_output, nvidia_smi_error = _query_nvidia_smi()
    gpu_rows = _parse_gpu_rows(nvidia_smi_output)
    selected = next((row for row in gpu_rows if row.get("index") == int(gpu)), None)
    selected_compute_cap = selected.get("compute_cap") if selected else None
    python_exists = python_bin.exists()
    python_executable = python_exists and os.access(python_bin, os.X_OK)
    sm90_ok = selected_compute_cap == "9.0"
    failure_reasons: list[str] = []
    if not python_exists:
        failure_reasons.append("python_bin_not_found")
    elif not python_executable:
        failure_reasons.append("python_bin_not_executable")
    if nvidia_smi_error:
        failure_reasons.append("nvidia_smi_failed")
    if not sm90_ok:
        failure_reasons.append("h800_sm90_gpu_not_available")
    return {
        "schema": "stage2_fp16_h800_rewrite_suite_preflight_v1",
        "status": "failed" if failure_reasons else "pass",
        "failure_reasons": failure_reasons,
        "checks": {
            "python_bin": str(python_bin),
            "python_bin_exists": python_exists,
            "python_bin_executable": python_executable,
            "selected_gpu": int(gpu),
            "selected_gpu_name": selected.get("name") if selected else None,
            "selected_gpu_compute_cap": selected_compute_cap,
            "sm90_required": True,
            "sm90_gate": sm90_ok,
            "nvidia_smi_error": nvidia_smi_error,
            "gpu_rows": gpu_rows,
        },
        "interpretation": (
            "This suite is for H800/sm90 final evidence. Non-sm90 runs are allowed only "
            "as structure diagnostics and must not be written as H800 measured rows."
        ),
    }


def _write_preflight_artifacts(report: dict[str, Any], log_dir: Path, export_dir: Path) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    _write_text(log_dir / "h800_suite_preflight.json", payload)
    lines = [
        "# Stage2 FP16 H800 rewrite suite preflight",
        "",
        f"- status: `{report.get('status')}`",
        f"- failure_reasons: `{', '.join(report.get('failure_reasons') or [])}`",
        "",
        "| check | value |",
        "|---|---|",
    ]
    for key, value in (report.get("checks") or {}).items():
        if key == "gpu_rows":
            continue
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        str(report.get("interpretation") or ""),
        "",
    ]
    _write_text(log_dir / "h800_suite_preflight.md", "\n".join(lines))
    _write_text(export_dir / "fp16_lhc07_h800_suite_preflight_latest.json", payload)
    _write_text(export_dir / "fp16_lhc07_h800_suite_preflight_latest.md", "\n".join(lines))


def _run_step(step: dict[str, Any], env: dict[str, str], cwd: Path, log_dir: Path, dry_run: bool) -> dict[str, Any]:
    name = str(step["name"])
    step_dir = log_dir / name
    step_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(part) for part in step["cmd"]]
    command_text = " ".join(cmd)
    _write_text(step_dir / "command.txt", command_text + "\n")
    record: dict[str, Any] = {
        "name": name,
        "claim": step.get("claim"),
        "command": cmd,
        "command_path": str(step_dir / "command.txt"),
        "stdout_path": str(step_dir / "stdout.txt"),
        "stderr_path": str(step_dir / "stderr.txt"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": dry_run,
    }
    if dry_run:
        _write_text(step_dir / "stdout.txt", "")
        _write_text(step_dir / "stderr.txt", "")
        record.update({"status": "dry_run", "returncode": None})
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return record

    proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=False)
    _write_text(step_dir / "stdout.txt", proc.stdout)
    _write_text(step_dir / "stderr.txt", proc.stderr)
    record.update(
        {
            "status": "success" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--full-reps", type=int, default=20)
    parser.add_argument("--suite", choices=["core", "full"], default="core")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--skip-gate-check", action="store_true")
    parser.add_argument("--python-bin", default=str(DEFAULT_TVM_PYTHON))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    export_dir = Path(args.export_dir)
    python_bin = Path(args.python_bin)
    log_dir = raw_dir / "suite_logs" / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_dir.mkdir(parents=True, exist_ok=True)
    env = _base_env(int(args.gpu))
    steps = _suite_steps(
        python_bin,
        int(args.gpu),
        raw_dir,
        export_dir,
        int(args.reps),
        int(args.full_reps),
        str(args.suite),
    )
    manifest: dict[str, Any] = {
        "schema": "stage2_fp16_h800_rewrite_suite_runner_v1",
        "status": "started",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(REPO_ROOT),
        "probe_script": str(PROBE_SCRIPT),
        "gate_check_script": str(GATE_CHECK_SCRIPT),
        "python_bin": str(python_bin),
        "python_bin_exists": python_bin.exists(),
        "gpu": int(args.gpu),
        "suite": str(args.suite),
        "dry_run": bool(args.dry_run),
        "raw_dir": str(raw_dir),
        "export_dir": str(export_dir),
        "log_dir": str(log_dir),
        "env_audit": {
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "PATH_prefix": env.get("PATH", "").split(":")[:3],
            "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH", ""),
        },
        "steps": [],
        "gate_check": None,
    }
    preflight = _preflight_environment(python_bin=python_bin, gpu=int(args.gpu))
    manifest["preflight"] = preflight
    _write_preflight_artifacts(preflight, log_dir, export_dir)
    if preflight.get("status") != "pass" and not args.dry_run:
        manifest.update(
            {
                "status": "failed",
                "error": "preflight_failed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _write_text(log_dir / "suite_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"[failed] manifest={log_dir / 'suite_manifest.json'}")
        return 2

    failed = False
    for step in steps:
        record = _run_step(step, env, REPO_ROOT, log_dir, bool(args.dry_run))
        manifest["steps"].append(record)
        print(f"[{record['status']}] {record['name']} -> {record['command_path']}")
        if record.get("status") == "failed":
            failed = True
            if args.stop_on_failure:
                break

    if not args.skip_gate_check:
        gate_record = _run_step(
            {
                "name": "h800_gate_check",
                "claim": "final_h800_sm90_latency_tensorcore_ap_gate",
                "cmd": [
                    sys.executable,
                    str(GATE_CHECK_SCRIPT),
                    "--export-dir",
                    str(export_dir),
                ],
            },
            env,
            REPO_ROOT,
            log_dir,
            bool(args.dry_run),
        )
        manifest["gate_check"] = gate_record
        print(f"[{gate_record['status']}] {gate_record['name']} -> {gate_record['command_path']}")
        if gate_record.get("status") == "failed":
            failed = True

    manifest["status"] = "failed" if failed else ("dry_run" if args.dry_run else "success")
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_text(log_dir / "suite_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] manifest={log_dir / 'suite_manifest.json'}")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
