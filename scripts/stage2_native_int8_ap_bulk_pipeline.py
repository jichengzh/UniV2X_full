#!/usr/bin/env python3
"""Run native INT8 AP bulk pipeline for original60 labels on one H800 GPU lane.

Pipeline per label:
checkpoint-consistent ONNX export -> bootstrap route -> BN-aware range capture
-> scale-aware calibrated route -> numeric sanity -> AP smoke -> persistent
full-val AP -> gated AP row import.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_TVMPY = "/exdata/jichengzhi/tvm310/bin/python"
DEFAULT_HEAL = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_CKPT_ROOT = "/home/jichengzhi/heal_research/checkpoints/stage1"
LD_LIBRARY_PATH = (
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib"
)
TIMEOUT_RC = 124
KILL_GRACE_SECONDS = 20
DEFAULT_QUARANTINE_LABELS = {
    "frontier_16",
    "frontier_18",
    "frontier_25",
    "frontier_26",
    "frontier_27",
    "frontier_01",
    "s1_112",
    "s2_096",
}
STEP_TIMEOUT_SECONDS = {
    "export": 20 * 60,
    "bootstrap_route": 20 * 60,
    "range_capture": 30 * 60,
    "calibrated_route": 30 * 60,
    "numeric_sanity": 15 * 60,
    "ap_smoke": 20 * 60,
    "ap_fullval": 180 * 60,
    "import_row": 5 * 60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--labels", default="", help="Comma-separated labels. Empty means all missing INT8 AP labels.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--env-python", default=DEFAULT_PY)
    parser.add_argument("--tvm-python", default=DEFAULT_TVMPY)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL)
    parser.add_argument("--ckpt-root", default=DEFAULT_CKPT_ROOT)
    parser.add_argument("--run-stamp", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-full-ap", action="store_true")
    parser.add_argument(
        "--full-ap-raw-root",
        default="",
        help=(
            "Optional directory for large full-val AP raw outputs. Route, range, "
            "and calibrated artifacts still use output-root; only ap_fullval raw "
            "dirs are placed under this root."
        ),
    )
    parser.add_argument("--quarantine-labels", default=",".join(sorted(DEFAULT_QUARANTINE_LABELS)))
    parser.add_argument("--quarantine-jsonl", default=None)
    parser.add_argument("--disable-default-quarantine", action="store_true")
    parser.add_argument(
        "--rerun-measured",
        action="store_true",
        help="Allow explicitly requested measured labels to run again for anomaly recheck.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(control_dir: Path, fields: list[Any]) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    with (control_dir / "events.tsv").open("a", encoding="utf-8") as handle:
        handle.write("\t".join(str(item) for item in [time.strftime("%Y-%m-%dT%H:%M:%S%z"), *fields]) + "\n")


def _public_env(env: dict[str, str]) -> dict[str, str | None]:
    return {
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONPATH": env.get("PYTHONPATH"),
        "LD_LIBRARY_PATH": env.get("LD_LIBRARY_PATH"),
        "STAGE2_NATIVE_INT8_MODEL_ROOT": env.get("STAGE2_NATIVE_INT8_MODEL_ROOT"),
    }


def _terminate_process_group(proc: subprocess.Popen[Any], *, step_dir: Path) -> dict[str, Any]:
    outcome: dict[str, Any] = {
        "pid": proc.pid,
        "sigterm_sent": False,
        "sigkill_sent": False,
        "terminated": False,
        "final_returncode": None,
    }
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        outcome["sigterm_sent"] = True
    except ProcessLookupError:
        outcome["terminated"] = True
        outcome["final_returncode"] = proc.poll()
        return outcome
    try:
        proc.wait(timeout=KILL_GRACE_SECONDS)
        outcome["terminated"] = True
        outcome["final_returncode"] = proc.returncode
        return outcome
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        outcome["sigkill_sent"] = True
    except ProcessLookupError:
        outcome["terminated"] = True
        outcome["final_returncode"] = proc.poll()
        return outcome
    try:
        proc.wait(timeout=KILL_GRACE_SECONDS)
        outcome["terminated"] = True
        outcome["final_returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        outcome["terminated"] = False
        outcome["final_returncode"] = proc.poll()
        write_json(step_dir / "process_group_kill_incomplete.json", outcome)
    return outcome


def _write_step_command(name: str, command: list[str], *, cwd: Path, env: dict[str, str], step_dir: Path, timeout_seconds: int | None) -> None:
    write_json(
        step_dir / "command.json",
        {
            "name": name,
            "command": command,
            "cwd": str(cwd),
            "timeout_seconds": timeout_seconds,
            "env": _public_env(env),
        },
    )


def _write_failure_report(
    *,
    step_dir: Path,
    name: str,
    command: list[str],
    cwd: Path,
    rc: int,
    failure_reason: str,
    timeout_seconds: int | None = None,
    process_outcome: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    write_json(
        step_dir / "failure_report.json",
        {
            "schema": "stage2_native_int8_ap_bulk_step_failure_v1",
            "name": name,
            "command": command,
            "cwd": str(cwd),
            "rc": rc,
            "failure_reason": failure_reason,
            "timeout_seconds": timeout_seconds,
            "process_outcome": process_outcome,
            "extra": extra or {},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def run_step(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    step_dir: Path,
    control_dir: Path,
    timeout_seconds: int | None,
) -> int:
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_step_command(name, command, cwd=cwd, env=env, step_dir=step_dir, timeout_seconds=timeout_seconds)
    (step_dir / "started_at.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    with (step_dir / "stdout.txt").open("w", encoding="utf-8") as stdout, (step_dir / "stderr.txt").open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        process_outcome: dict[str, Any] | None = None
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process_outcome = _terminate_process_group(proc, step_dir=step_dir)
    rc = TIMEOUT_RC if timed_out else int(proc.returncode)
    (step_dir / "rc.txt").write_text(f"{rc}\n", encoding="utf-8")
    if timed_out:
        _write_failure_report(
            step_dir=step_dir,
            name=name,
            command=command,
            cwd=cwd,
            rc=rc,
            failure_reason="timeout",
            timeout_seconds=timeout_seconds,
            process_outcome=process_outcome,
        )
    elif rc != 0:
        _write_failure_report(
            step_dir=step_dir,
            name=name,
            command=command,
            cwd=cwd,
            rc=rc,
            failure_reason="nonzero_returncode",
            timeout_seconds=timeout_seconds,
        )
    append_event(control_dir, [name, rc, step_dir, "timeout" if timed_out else "done"])
    return rc


def run_step_until_report(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    step_dir: Path,
    control_dir: Path,
    report_path: Path,
    min_samples: int,
    timeout_seconds: int | None,
    poll_seconds: int = 30,
    grace_seconds: int = 120,
) -> int:
    """Run a command but treat a completed gated report as success.

    Some PyTorch/CUDA full-val processes finish writing AP reports but keep a
    non-Python worker thread alive during shutdown. Waiting only on process
    exit can stall the lane forever, so the report gate is the source of truth.
    """
    step_dir.mkdir(parents=True, exist_ok=True)
    _write_step_command(name, command, cwd=cwd, env=env, step_dir=step_dir, timeout_seconds=timeout_seconds)
    (step_dir / "started_at.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n", encoding="utf-8")
    stdout = (step_dir / "stdout.txt").open("w", encoding="utf-8")
    stderr = (step_dir / "stderr.txt").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )
    try:
        report_seen_at: float | None = None
        started_at = time.time()
        while True:
            rc = proc.poll()
            if rc is not None:
                (step_dir / "rc.txt").write_text(f"{rc}\n", encoding="utf-8")
                if rc != 0:
                    _write_failure_report(
                        step_dir=step_dir,
                        name=name,
                        command=command,
                        cwd=cwd,
                        rc=int(rc),
                        failure_reason="nonzero_returncode",
                        timeout_seconds=timeout_seconds,
                    )
                append_event(control_dir, [name, rc, step_dir])
                return int(rc)
            if report_success(report_path, min_samples=min_samples):
                if report_seen_at is None:
                    report_seen_at = time.time()
                    write_json(
                        step_dir / "report_success_before_process_exit.json",
                        {
                            "status": "report_success_waiting_for_process_exit",
                            "pid": proc.pid,
                            "report_path": str(report_path),
                            "grace_seconds": grace_seconds,
                        },
                    )
                elif time.time() - report_seen_at >= grace_seconds:
                    process_outcome = _terminate_process_group(proc, step_dir=step_dir)
                    (step_dir / "rc.txt").write_text("0\n", encoding="utf-8")
                    write_json(
                        step_dir / "process_terminated_after_report_success.json",
                        {
                            "status": "success",
                            "pid": proc.pid,
                            "reason": "full_ap_report_gate_passed_but_process_did_not_exit",
                            "report_path": str(report_path),
                            "process_outcome": process_outcome,
                        },
                    )
                    append_event(control_dir, [name, 0, step_dir, "report_success_killed_stale_process"])
                    return 0
            if timeout_seconds is not None and time.time() - started_at >= timeout_seconds:
                process_outcome = _terminate_process_group(proc, step_dir=step_dir)
                (step_dir / "rc.txt").write_text(f"{TIMEOUT_RC}\n", encoding="utf-8")
                _write_failure_report(
                    step_dir=step_dir,
                    name=name,
                    command=command,
                    cwd=cwd,
                    rc=TIMEOUT_RC,
                    failure_reason="timeout_without_success_report",
                    timeout_seconds=timeout_seconds,
                    process_outcome=process_outcome,
                    extra={"report_path": str(report_path), "min_samples": min_samples},
                )
                append_event(control_dir, [name, TIMEOUT_RC, step_dir, "timeout_without_success_report"])
                return TIMEOUT_RC
            time.sleep(poll_seconds)
    finally:
        stdout.close()
        stderr.close()


def stage_from_name(name: str) -> str:
    return name.split(":", 1)[-1] if ":" in name else name


def timeout_for(stage: str) -> int:
    return STEP_TIMEOUT_SECONDS[stage]


def quarantine_labels(args: argparse.Namespace) -> set[str]:
    labels: set[str] = set()
    if not args.disable_default_quarantine:
        labels.update(DEFAULT_QUARANTINE_LABELS)
    labels.update(item.strip() for item in str(args.quarantine_labels or "").split(",") if item.strip())
    if args.quarantine_jsonl:
        for row in read_jsonl(Path(args.quarantine_jsonl)):
            status = str(row.get("quarantine_status") or row.get("status") or "active")
            if status and status not in {"active", "quarantined"}:
                continue
            label = str(row.get("label") or row.get("config_label") or row.get("candidate_label") or "")
            if label:
                labels.add(label)
    return labels


def write_label_attempt(control_dir: Path, payload: dict[str, Any]) -> None:
    attempts_dir = control_dir / "label_attempts"
    label = str(payload.get("label") or "unknown")
    write_json(attempts_dir / f"{label}.json", payload)


def measured_int8_ap_labels(output_root: Path) -> set[str]:
    rows = output_root / "rows/native_int8_original60_ap_rows_v1.jsonl"
    labels: set[str] = set()
    for row in read_jsonl(rows):
        if row.get("measurement_status") == "measured" or row.get("metric_value") is not None:
            label = str(row.get("label") or "")
            if label:
                labels.add(label)
    return labels


def completion_widths(output_root: Path) -> dict[str, list[int]]:
    queue = output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
    widths: dict[str, list[int]] = {}
    for row in read_jsonl(queue):
        if row.get("precision") != "int8":
            continue
        label = str(row.get("label") or "")
        width = row.get("width")
        if label and isinstance(width, list):
            widths[label] = [int(item) for item in width]
    return widths


def resolve_labels(args: argparse.Namespace, output_root: Path) -> list[str]:
    widths = completion_widths(output_root)
    measured = measured_int8_ap_labels(output_root)
    if args.labels.strip():
        return [item.strip() for item in args.labels.split(",") if item.strip()]
    return [label for label in widths if label not in measured]


def resolve_ckpt_dir(ckpt_root: Path, label: str) -> Path | None:
    candidates = sorted(ckpt_root.glob(f"Pyramid_DAIR_m1_stage2_ap_{label}_2026_06_*"))
    usable = [p for p in candidates if (p / "config.yaml").exists() and list(p.glob("net_epoch*.pth"))]
    return usable[-1] if usable else None


def report_success(path: Path, *, min_samples: int | None = None) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") not in {"success", "passed"}:
        return False
    if min_samples is not None:
        processed = int(payload.get("processed_samples") or payload.get("num_samples") or 0)
        if processed < min_samples:
            return False
    return True


def run_label(args: argparse.Namespace, *, label: str, width: list[int], output_root: Path, control_dir: Path) -> dict[str, Any]:
    gpu = int(args.gpu_id)
    raw_parent = output_root / "raw/int8_native_route"
    route_script = raw_parent / "stage2_h800_native_int8_full_onnx_route.py"
    queue = output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
    ckpt_dir = resolve_ckpt_dir(Path(args.ckpt_root), label)
    if ckpt_dir is None:
        append_event(control_dir, [label, "blocked", "missing_checkpoint"])
        return {"label": label, "status": "blocked", "reason": "missing_checkpoint"}

    model_root = raw_parent / f"checkpoint_consistent_{label}_multiscale_export_v2"
    onnx_path = model_root / f"{label}_backbone.onnx"
    export_report = model_root / "export_report.json"
    env_base = dict(os.environ)
    env_base["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    env_base["PYTHONPATH"] = f"{ROOT}:{args.heal_root}:{env_base.get('PYTHONPATH', '')}"

    append_event(control_dir, [label, "LABEL_START", ckpt_dir, width])
    if not report_success(export_report) or not onnx_path.exists():
        rc = run_step(
            f"{label}:export",
            [
                args.env_python,
                "scripts/stage2_h800_export_checkpoint_multiscale_onnx.py",
                "--label", label,
                "--ckpt-dir", str(ckpt_dir),
                "--out", str(onnx_path),
                "--report-json", str(export_report),
                "--gpu-id", str(gpu),
            ],
            cwd=ROOT,
            env=env_base,
            step_dir=model_root / f"export_gpu{gpu}",
            control_dir=control_dir,
            timeout_seconds=timeout_for("export"),
        )
        if rc != 0:
            return {"label": label, "status": "failed", "stage": "export", "rc": rc}
    else:
        append_event(control_dir, [f"{label}:export_skip", 0, model_root])

    boot_run = f"20260629_{label}_bootstrap_bnaware_bulk_v1"
    boot_root = raw_parent / boot_run
    boot_route = boot_root / label
    tvm_env = dict(env_base)
    tvm_env["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    tvm_env["STAGE2_NATIVE_INT8_MODEL_ROOT"] = str(model_root)
    if not (boot_route / "native_int8_route_manifest.json").exists():
        rc = run_step(
            f"{label}:bootstrap_route",
            [
                args.tvm_python,
                str(route_script),
                "--run-id", boot_run,
                "--gpu", str(gpu),
                "--labels", label,
                "--completion-queue", str(queue),
                "--allow-non-h800-debug",
            ],
            cwd=ROOT,
            env=tvm_env,
            step_dir=boot_root / "launch",
            control_dir=control_dir,
            timeout_seconds=timeout_for("bootstrap_route"),
        )
        if rc != 0:
            return {"label": label, "status": "failed", "stage": "bootstrap_route", "rc": rc}
    else:
        append_event(control_dir, [f"{label}:bootstrap_skip", 0, boot_route])

    capture_dir = raw_parent / f"20260629_{label}_bnaware_tensor_range_capture_bulk_v1_gpu{gpu}"
    params = capture_dir / "tensor_quant_params_calibration_bnaware_v1_to_pyramid_level2.json"
    if not params.exists():
        rc = run_step(
            f"{label}:range_capture",
            [
                args.env_python,
                "scripts/stage2_h800_native_int8_op_alignment.py",
                "--label", label,
                "--ckpt-dir", str(ckpt_dir),
                "--route-dir", str(boot_route),
                "--raw-dir", str(capture_dir),
                "--gpu-id", str(gpu),
                "--collect-reference-ranges",
                "--execute-reference-range-capture",
                "--reference-range-stop-output", "pyramid_level2",
                "--reference-range-out", "tensor_reference_range_targets_to_pyramid_level2_bnaware_v1.json",
                "--module-inventory-out", "pytorch_module_inventory_to_pyramid_level2_bnaware_v1.json",
                "--reference-range-plan-out", "tensor_reference_range_capture_plan_to_pyramid_level2_bnaware_v1.json",
                "--reference-ranges-out", "tensor_reference_ranges_to_pyramid_level2_bnaware_v1.json",
                "--reference-range-calibration-out", params.name,
            ],
            cwd=ROOT,
            env=env_base,
            step_dir=capture_dir,
            control_dir=control_dir,
            timeout_seconds=timeout_for("range_capture"),
        )
        if rc != 0 or not params.exists():
            return {"label": label, "status": "failed", "stage": "range_capture", "rc": rc}
    else:
        append_event(control_dir, [f"{label}:range_capture_skip", 0, capture_dir])

    cal_run = f"20260629_{label}_scaleaware_bnaware_bias_bulk_v1"
    cal_root = raw_parent / cal_run
    route = cal_root / label
    artifact = route / f"{label}_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so"
    inventory = route / "tvm_operator_inventory.json"
    weights = route / "runtime_weights_int8.npz"
    if not (route / "native_int8_route_manifest.json").exists() or not artifact.exists():
        cal_env = dict(tvm_env)
        cal_env["STAGE2_NATIVE_INT8_MODEL_ROOT"] = str(model_root)
        rc = run_step(
            f"{label}:calibrated_route",
            [
                args.tvm_python,
                str(route_script),
                "--run-id", cal_run,
                "--gpu", str(gpu),
                "--labels", label,
                "--completion-queue", str(queue),
                "--tensor-quant-params-path", str(params),
                "--allow-non-h800-debug",
            ],
            cwd=ROOT,
            env=cal_env,
            step_dir=cal_root / "launch",
            control_dir=control_dir,
            timeout_seconds=timeout_for("calibrated_route"),
        )
        if rc != 0 or not artifact.exists():
            return {"label": label, "status": "failed", "stage": "calibrated_route", "rc": rc}
    else:
        append_event(control_dir, [f"{label}:calibrated_route_skip", 0, route])

    numeric_dir = route / f"numeric_sanity_bnaware_bias_bulk_v1_gpu{gpu}"
    numeric_report = numeric_dir / "real_activation_bridge_report.json"
    if not report_success(numeric_report):
        rc = run_step(
            f"{label}:numeric_sanity",
            [
                args.env_python,
                "scripts/stage2_h800_native_int8_real_activation_bridge.py",
                "--label", label,
                "--ckpt-dir", str(ckpt_dir),
                "--raw-dir", str(numeric_dir),
                "--route-dir", str(route),
                "--artifact-path", str(artifact),
                "--inventory-path", str(inventory),
                "--runtime-weight-archive-path", str(weights),
                "--gpu-id", str(gpu),
                "--num-samples", "1",
                "--full-ap-min-samples", "1",
                "--ap-row-min-samples", "1789",
                "--keep-detailed-samples", "1",
                "--numeric-sanity-only",
                "--tensor-quant-params-path", str(params),
                "--persistent-worker",
            ],
            cwd=ROOT,
            env=env_base,
            step_dir=numeric_dir,
            control_dir=control_dir,
            timeout_seconds=timeout_for("numeric_sanity"),
        )
        if rc != 0 or not report_success(numeric_report):
            return {"label": label, "status": "failed", "stage": "numeric_sanity", "rc": rc}
    else:
        append_event(control_dir, [f"{label}:numeric_sanity_skip", 0, numeric_dir])

    if not args.skip_smoke:
        smoke_dir = route / f"ap_smoke_5samples_bnaware_bias_bulk_v1_gpu{gpu}"
        smoke_report = smoke_dir / "full_ap_eval_report.json"
        if not report_success(smoke_report, min_samples=5):
            rc = run_step(
                f"{label}:ap_smoke",
                [
                    args.env_python,
                    "scripts/stage2_h800_native_int8_real_activation_bridge.py",
                    "--label", label,
                    "--ckpt-dir", str(ckpt_dir),
                    "--raw-dir", str(smoke_dir),
                    "--route-dir", str(route),
                    "--artifact-path", str(artifact),
                    "--inventory-path", str(inventory),
                    "--runtime-weight-archive-path", str(weights),
                    "--gpu-id", str(gpu),
                    "--num-samples", "5",
                    "--full-ap-min-samples", "5",
                    "--ap-row-min-samples", "1789",
                    "--keep-detailed-samples", "5",
                    "--tensor-quant-params-path", str(params),
                    "--persistent-worker",
                ],
                cwd=ROOT,
                env=env_base,
                step_dir=smoke_dir,
                control_dir=control_dir,
                timeout_seconds=timeout_for("ap_smoke"),
            )
            if rc != 0 or not report_success(smoke_report, min_samples=5):
                return {"label": label, "status": "failed", "stage": "ap_smoke", "rc": rc}
        else:
            append_event(control_dir, [f"{label}:ap_smoke_skip", 0, smoke_dir])

    if args.skip_full_ap:
        append_event(control_dir, [label, "LABEL_READY_FULL_AP_SKIPPED"])
        return {"label": label, "status": "ready_full_ap_skipped"}

    if str(args.full_ap_raw_root or "").strip():
        full_parent = Path(args.full_ap_raw_root) / label
    else:
        full_parent = route
    full_dir = full_parent / f"ap_fullval_1789_bnaware_bias_bulk_v1_gpu{gpu}_{args.run_stamp}"
    full_report = full_dir / "full_ap_eval_report.json"
    if not report_success(full_report, min_samples=1789):
        rc = run_step_until_report(
            f"{label}:ap_fullval",
            [
                args.env_python,
                "scripts/stage2_h800_native_int8_real_activation_bridge.py",
                "--label", label,
                "--ckpt-dir", str(ckpt_dir),
                "--raw-dir", str(full_dir),
                "--route-dir", str(route),
                "--artifact-path", str(artifact),
                "--inventory-path", str(inventory),
                "--runtime-weight-archive-path", str(weights),
                "--gpu-id", str(gpu),
                "--num-samples", "1789",
                "--full-ap-min-samples", "1789",
                "--ap-row-min-samples", "1789",
                "--keep-detailed-samples", "20",
                "--tensor-quant-params-path", str(params),
                "--persistent-worker",
            ],
            cwd=ROOT,
            env=env_base,
            step_dir=full_dir,
            control_dir=control_dir,
            report_path=full_report,
            min_samples=1789,
            timeout_seconds=timeout_for("ap_fullval"),
        )
        if rc != 0 or not report_success(full_report, min_samples=1789):
            return {"label": label, "status": "failed", "stage": "ap_fullval", "rc": rc}
    else:
        append_event(control_dir, [f"{label}:ap_fullval_skip", 0, full_dir])

    rc = run_step(
        f"{label}:import_row",
        [
            args.env_python,
            "scripts/stage2_import_native_int8_full_ap_row.py",
            "--label", label,
            "--width", ",".join(str(item) for item in width),
            "--raw-dir", str(full_dir),
            "--route-dir", str(route),
            "--tensor-quant-params-path", str(params),
            "--report-json", str(full_report),
            "--run-id", f"20260629_native_int8_{label}_bnaware_bias_bulk_fullval_v1",
        ],
        cwd=ROOT,
        env=env_base,
        step_dir=full_dir / "import_row",
        control_dir=control_dir,
        timeout_seconds=timeout_for("import_row"),
    )
    if rc != 0:
        return {"label": label, "status": "failed", "stage": "import_row", "rc": rc}
    append_event(control_dir, [label, "LABEL_DONE", full_dir])
    return {"label": label, "status": "succeeded", "full_dir": str(full_dir)}


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    control_dir = output_root / "raw/int8_native_route" / f"20260629_native_int8_ap_bulk_pipeline_gpu{args.gpu_id}_{args.run_stamp}"
    active_quarantine = quarantine_labels(args)
    labels = [label for label in resolve_labels(args, output_root) if label not in active_quarantine]
    widths = completion_widths(output_root)
    write_json(
        control_dir / "launch_manifest.json",
        {
            "gpu_id": args.gpu_id,
            "labels": labels,
            "run_stamp": args.run_stamp,
            "quarantine_labels": sorted(active_quarantine),
            "step_timeout_seconds": STEP_TIMEOUT_SECONDS,
        },
    )
    results: list[dict[str, Any]] = []
    for label in labels:
        try:
            width = widths.get(label)
            if not width:
                result = {"label": label, "status": "blocked", "reason": "missing_width"}
            elif label in measured_int8_ap_labels(output_root) and not args.rerun_measured:
                result = {"label": label, "status": "skipped_measured"}
            else:
                result = run_label(args, label=label, width=width, output_root=output_root, control_dir=control_dir)
        except Exception as exc:
            result = {
                "label": label,
                "status": "failed",
                "stage": "pipeline_exception",
                "reason": type(exc).__name__,
                "message": str(exc),
            }
            append_event(control_dir, [label, "PIPELINE_EXCEPTION", type(exc).__name__, str(exc)])
        results.append(result)
        write_label_attempt(control_dir, {**result, "gpu_id": args.gpu_id, "run_stamp": args.run_stamp})
        write_json(control_dir / "latest_results.json", {"results": results})
    write_json(control_dir / "final_results.json", {"results": results})
    ok = all(item.get("status") in {"succeeded", "skipped_measured", "ready_full_ap_skipped"} for item in results)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
