#!/usr/bin/env python3
"""Run the resumable Stage3 scale-aware TVM INT8 repair batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("/home/jichengzhi/V2X")
UNIV2X = Path("/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python")
TVM_PYTHON = Path("/exdata/jichengzhi/tvm310/bin/python")
CODRIVING_REPO = Path("/exdata/jichengzhi/V2Xverse_pyramid")
TVM_LD = ":".join([
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib",
])


def provenance_sha256(path: Path) -> str:
    """Match the AP gate's path-bound digest contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def command_option(command: list[str], option: str) -> str | None:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def idle_gpu(candidates: list[int]) -> int | None:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    busy_uuids = {line.strip() for line in query if line.strip()}
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    for row in rows:
        index, uuid = [item.strip() for item in row.split(",", 1)]
        if int(index) in candidates and uuid not in busy_uuids:
            return int(index)
    return None


def wait_gpu(candidates: list[int], poll_seconds: int) -> int:
    while True:
        gpu = idle_gpu(candidates)
        if gpu is not None:
            return gpu
        time.sleep(poll_seconds)


def run_step(job_dir: Path, state: Path, job_id: str, stage: str, command: list[str], *, env: dict[str, str] | None = None) -> None:
    marker = job_dir / f"{stage}.success.json"
    if marker.is_file():
        return
    log = job_dir / f"{stage}.log"
    job_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(state, {"timestamp": utc_now(), "job_id": job_id, "stage": stage, "status": "started", "command": command})
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        append_jsonl(state, {"timestamp": utc_now(), "job_id": job_id, "stage": stage, "status": "failed", "exit_code": completed.returncode, "log": str(log)})
        raise RuntimeError(f"{job_id} {stage} failed with exit {completed.returncode}")
    marker.write_text(json.dumps({"timestamp": utc_now(), "command": command}, indent=2) + "\n")
    append_jsonl(state, {"timestamp": utc_now(), "job_id": job_id, "stage": stage, "status": "success", "log": str(log)})


def report_passed(path: Path, *, full: bool) -> bool:
    if not path.is_file():
        return False
    report = read_json(path)
    gate = "full_1789" if full else "sanity_16"
    return report.get("gates", {}).get(gate) is True


def resolve_calibration_source(
    plan: Mapping[str, Any], *, model: str, width: str, job_dir: Path
) -> tuple[Path, Path, bool]:
    source_contract = plan.get("source_contract")
    source_contract = source_contract if isinstance(source_contract, Mapping) else {}
    calibration_value = source_contract.get("calibration_npz")
    summary_value = source_contract.get("calibration_summary")
    if calibration_value and summary_value:
        calibration = Path(str(calibration_value)).resolve()
        summary = Path(str(summary_value)).resolve()
        if not calibration.is_file() or not summary.is_file():
            raise FileNotFoundError(f"manifest calibration evidence missing: {calibration} / {summary}")
        return calibration, summary, False
    if model == "pyramid":
        key = "".join(f"{int(value):03d}x" for value in plan["width"]).rstrip("x")
        source = ROOT / "results/stage3_gold96_v3_20260711/pyramid_calibration" / key
        calibration = source / "spatial_features_train16.npz"
        summary = source / "summary.json"
        if not calibration.is_file() or not summary.is_file():
            raise FileNotFoundError(f"legacy Pyramid calibration evidence missing: {calibration} / {summary}")
        return calibration, summary, False
    calibration_dir = job_dir / "calibration"
    return (
        calibration_dir / "spatial_features_train16.npz",
        calibration_dir / "summary.json",
        True,
    )


def run_job(plan: dict[str, Any], root: Path, state: Path, candidates: list[int], poll: int,
            *, allow_shared_gpu: bool = False) -> None:
    model = str(plan["model"])
    width = "x".join(str(value) for value in plan["width"])
    job_id = str(plan["manifest_job_id"])
    job_dir = root / model / width
    full_report = job_dir / "ap_full" / "full_ap_eval_report.json"
    if report_passed(full_report, full=True):
        append_jsonl(state, {"timestamp": utc_now(), "job_id": job_id, "stage": "job", "status": "already_complete"})
        return

    result = read_json(Path(plan["performance_result_json"]))
    onnx = Path(result["onnx_path"])
    model_dir_value = command_option(plan["sanity_command"], "--model-dir") or command_option(plan["sanity_command"], "--ckpt-dir")
    if not model_dir_value:
        raise ValueError(f"model directory missing from plan: {job_id}")
    model_dir = Path(model_dir_value)
    calibration_npz, calibration_summary, calibration_export_required = resolve_calibration_source(
        plan, model=model, width=width, job_dir=job_dir
    )

    gpu = candidates[0] if allow_shared_gpu else wait_gpu(candidates, poll)
    if calibration_export_required:
        run_step(job_dir, state, job_id, "calibration", [
            str(UNIV2X), "scripts/stage2_v2_gold_coldstart96_codriving_calib_export.py",
            "--repo-root", str(CODRIVING_REPO), "--width", width,
            "--model-dir", str(model_dir), "--output", str(calibration_npz.resolve()),
            "--summary", str(calibration_summary.resolve()), "--n-samples", "16",
            "--num-workers", "0", "--storage-dtype", "float32", "--split", "train",
        ], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})

    contract = job_dir / "tensor_quant_params.json"
    run_step(job_dir, state, job_id, "quant_contract", [
        str(UNIV2X), "scripts/stage3_tvm_int8_quant_contract_v3.py",
        "--onnx", str(onnx), "--calibration-npz", str(calibration_npz),
        "--calibration-summary", str(calibration_summary), "--output-json", str(contract),
    ], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})

    label = f"{model}_{width}_scaleaware"
    build_root = job_dir / "build"
    run_step(job_dir, state, job_id, "build_performance_energy", [
        str(TVM_PYTHON), "scripts/stage2_route_b_int8_auto_decomp.py",
        "--label", label, "--width", ",".join(str(value) for value in plan["width"]),
        "--onnx", str(onnx), "--out-dir", str(build_root), "--gpu", str(gpu),
        "--tensor-quant-params-json", str(contract), "--warmup", "20", "--number", "20",
        "--repeat", "5", "--measure-energy", "--energy-iters", "300",
    ], env={**os.environ, "LD_LIBRARY_PATH": TVM_LD})
    artifact = build_root / label / "route_b_int8_auto_decomp.vmexec"
    build_report = read_json(build_root / label / "route_b_int8_auto_decomp_result.json")
    if build_report.get("correctness_all_exact") is not True:
        raise RuntimeError(f"{job_id} candidate/native exact comparison failed")

    sanity_report = job_dir / "ap_sanity" / "full_ap_eval_report.json"
    if model == "pyramid":
        sanity = [str(UNIV2X), "scripts/stage3_pyramid_tvm_int8_ap_numeric_gate_v3.py",
                  "--compiled-artifact", str(artifact), "--precision-tag", "int8", "--num-samples", "16",
                  "--full-ap-min-samples", "1789", "--output-dir", str(job_dir / "ap_sanity"),
                  "--model-dir", str(model_dir), "--eval-range", "102.4,51.2",
                  "--report-json", str(sanity_report), "--gpu-id", str(gpu)]
    else:
        sanity = [str(UNIV2X), "scripts/stage3_codriving_tvm_int8_ap_numeric_gate_v3.py",
                  "--compiled-artifact", str(artifact), "--precision-tag", "int8", "--num-samples", "16",
                  "--full-ap-min-samples", "1789", "--output-dir", str(job_dir / "ap_sanity"),
                  "--model-dir", str(model_dir), "--repo-root", str(CODRIVING_REPO), "--num-workers", "0",
                  "--report-json", str(sanity_report), "--gpu-id", str(gpu)]
    try:
        run_step(job_dir, state, job_id, "sanity", sanity)
    except RuntimeError:
        if not sanity_report.is_file():
            raise
    if not report_passed(sanity_report, full=False):
        if model != "codriving":
            raise RuntimeError(f"{job_id} numerical sanity gate failed")
        contract = job_dir / "tensor_quant_params_percentile_99_99.json"
        run_step(job_dir, state, job_id, "quant_contract_percentile", [
            str(UNIV2X), "scripts/stage3_tvm_int8_quant_contract_v3.py",
            "--onnx", str(onnx), "--calibration-npz", str(calibration_npz),
            "--calibration-summary", str(calibration_summary), "--output-json", str(contract),
            "--calibration-method", "percentile_99_99",
        ], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
        label = f"{model}_{width}_percentile_99_99"
        build_root = job_dir / "build_percentile_99_99"
        run_step(job_dir, state, job_id, "build_performance_energy_percentile", [
            str(TVM_PYTHON), "scripts/stage2_route_b_int8_auto_decomp.py",
            "--label", label, "--width", ",".join(str(value) for value in plan["width"]),
            "--onnx", str(onnx), "--out-dir", str(build_root), "--gpu", str(gpu),
            "--tensor-quant-params-json", str(contract), "--warmup", "20", "--number", "20",
            "--repeat", "5", "--measure-energy", "--energy-iters", "300",
        ], env={**os.environ, "LD_LIBRARY_PATH": TVM_LD})
        artifact = build_root / label / "route_b_int8_auto_decomp.vmexec"
        percentile_report = read_json(build_root / label / "route_b_int8_auto_decomp_result.json")
        if percentile_report.get("correctness_all_exact") is not True:
            raise RuntimeError(f"{job_id} percentile candidate/native exact comparison failed")
        sanity_report = job_dir / "ap_sanity_percentile_99_99" / "full_ap_eval_report.json"
        sanity = [str(UNIV2X), "scripts/stage3_codriving_tvm_int8_ap_numeric_gate_v3.py",
                  "--compiled-artifact", str(artifact), "--precision-tag", "int8", "--num-samples", "16",
                  "--full-ap-min-samples", "1789", "--output-dir", str(sanity_report.parent),
                  "--model-dir", str(model_dir), "--repo-root", str(CODRIVING_REPO), "--num-workers", "0",
                  "--report-json", str(sanity_report), "--gpu-id", str(gpu)]
        run_step(job_dir, state, job_id, "sanity_percentile", sanity)
        if not report_passed(sanity_report, full=False):
            raise RuntimeError(f"{job_id} percentile numerical sanity gate failed")

    full_dir_name = "ap_full_percentile_99_99" if "percentile" in sanity_report.parent.name else "ap_full"
    full_report = job_dir / full_dir_name / "full_ap_eval_report.json"
    full = list(sanity)
    full[full.index("16")] = "1789"
    full[full.index(str(sanity_report.parent))] = str(full_report.parent)
    full[full.index(str(sanity_report))] = str(full_report)
    if model == "codriving":
        digest = provenance_sha256(sanity_report)
        full.extend(["--sanity-report-json", str(sanity_report), "--sanity-report-sha256", digest])
    run_step(job_dir, state, job_id, "full_ap", full)
    if not report_passed(full_report, full=True):
        raise RuntimeError(f"{job_id} full AP gate failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-jsonl", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--widths", help="Optional comma-separated width triples, e.g. 16x32x64,32x64x128")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.ap_plan_jsonl.read_text().splitlines() if line.strip()]
    jobs = [row for row in rows if row.get("q") == "int8" and "h800-tvm" in str(row.get("profile"))]
    if args.widths:
        selected = {value.strip() for value in args.widths.split(",") if value.strip()}
        jobs = [row for row in jobs if "x".join(str(value) for value in row["width"]) in selected]
    for job in jobs:
        try:
            run_job(job, args.output_root.resolve(), args.state_jsonl.resolve(),
                    [int(value) for value in args.gpus.split(",")], args.poll_seconds,
                    allow_shared_gpu=args.allow_shared_gpu)
        except Exception as exc:
            append_jsonl(args.state_jsonl, {"timestamp": utc_now(), "job_id": job["manifest_job_id"],
                         "stage": "job", "status": "failed", "error": f"{type(exc).__name__}:{exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
