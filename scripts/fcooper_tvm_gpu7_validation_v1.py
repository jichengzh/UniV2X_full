#!/usr/bin/env python3
"""Run three performance repeats and one full-AP repeat on physical GPU7."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.fcooper_tvm_evidence_pools_v1 import (
    binding,
    file_sha256,
    normalize_success_row,
    write_immutable,
)


REPO = Path("/home/jichengzhi/V2X")


def _append_tvm_libs(command: list[str], tvm_lib_dirs: Sequence[Path]) -> None:
    for path in tvm_lib_dirs:
        command.extend(["--tvm-lib-dir", str(path)])


def build_ap_command(
    feedback: Mapping[str, Any],
    *,
    output_dir: Path,
    python: Path,
    tvm_python: Path,
    tvm_site: Path,
    tvm_lib_dirs: Sequence[Path],
    execution_dir: Path | None = None,
) -> list[str]:
    output_dir = Path(output_dir)
    q_mode = str(feedback["q_mode"])
    source = feedback["source_contract"]
    common = [
        str(python),
        "",
        "--config",
        str(source["config_path"]),
        "--checkpoint-dir",
        str(Path(str(source["checkpoint_path"])).parent),
        "--checkpoint",
        str(source["checkpoint_path"]),
        "--artifact",
        str(feedback["tvm_artifact_path"]),
        "--output-json",
        str(output_dir / "full_ap_report.json"),
        "--prediction-path",
        str(output_dir / "predictions.jsonl"),
        "--gpu-id",
        "0",
        "--num-workers",
        "4",
        "--tvm-worker-python",
        str(tvm_python),
        "--tvm-site",
        str(tvm_site),
    ]
    if q_mode == "int8":
        label_dir = Path(str(feedback["tvm_artifact_path"])).parent
        execution_dir = execution_dir or label_dir.parents[1]
        sanity = execution_dir / "numeric_sanity.json"
        common[1] = str(REPO / "scripts/fcooper_tvm_int8_ap_bridge_v1.py")
        common.extend(
            [
                "--route-result",
                str(label_dir / "route_b_int8_auto_decomp_result.json"),
                "--runtime-weights",
                str(label_dir / "runtime_weights_int8.npz"),
                "--quant-contract",
                str(feedback["quant_contract_path"]),
                "--mode",
                "full",
                "--sanity-report",
                str(sanity),
                "--sanity-report-sha256",
                file_sha256(sanity),
            ]
        )
    else:
        common[1] = str(REPO / "scripts/fcooper_tvm_fp16_ap_bridge_v1.py")
        common.extend(
            [
                "--input-shape",
                "5,64,512,512",
                "--output-shape",
                f"5,{int(feedback['width'][-1])},256,256",
                "--precision",
                q_mode,
                "--artifact-output-dtype",
                "float32",
            ]
        )
    _append_tvm_libs(common, tvm_lib_dirs)
    return common


def _gpu_processes(index: int) -> list[int]:
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    uuid = None
    for row in gpu_rows:
        gpu_index, gpu_uuid = (part.strip() for part in row.split(",", 1))
        if int(gpu_index) == index:
            uuid = gpu_uuid
            break
    if uuid is None:
        raise RuntimeError(f"GPU{index} does not exist")
    process_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return [
        int(pid.strip())
        for gpu_uuid, pid in (
            (part.strip() for part in row.split(",", 1))
            for row in process_rows
            if "," in row
        )
        if gpu_uuid == uuid
    ]


def wait_for_exclusive_gpu(
    physical_gpu_id: int,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    process_probe=_gpu_processes,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> dict[str, Any]:
    if timeout_seconds < 0:
        raise ValueError("exclusive wait timeout must be non-negative")
    if poll_interval_seconds <= 0:
        raise ValueError("exclusive poll interval must be positive")
    started = monotonic()
    checks = 0
    while True:
        checks += 1
        if not process_probe(physical_gpu_id):
            return {
                "physical_gpu_id": int(physical_gpu_id),
                "checks": checks,
                "waited_seconds": float(monotonic() - started),
                "exclusive": True,
            }
        elapsed = monotonic() - started
        if elapsed >= timeout_seconds:
            raise TimeoutError(
                f"GPU{physical_gpu_id} did not become exclusive within "
                f"{timeout_seconds} seconds"
            )
        sleep(min(poll_interval_seconds, timeout_seconds - elapsed))


def _environment(
    *,
    physical_gpu: int,
    heal_root: Path,
    python_site: Path,
    tvm_site: Path,
    tvm_lib_dirs: Sequence[Path],
) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["PYTHONPATH"] = ":".join(
        [
            str(REPO),
            str(heal_root),
            str(python_site),
            str(tvm_site),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in tvm_lib_dirs), env.get("LD_LIBRARY_PATH", "")]
    )
    return env


def _tvm_environment(
    *,
    physical_gpu: int,
    tvm_site: Path,
    tvm_lib_dirs: Sequence[Path],
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_environment is None else base_environment)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["PYTHONPATH"] = ":".join([str(REPO), str(tvm_site)])
    env["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in tvm_lib_dirs), env.get("LD_LIBRARY_PATH", "")]
    ).rstrip(":")
    return env


def feedback_with_resolved_source(
    feedback: Mapping[str, Any], *, execution_dir: Path
) -> dict[str, Any]:
    provenance_path = Path(execution_dir) / "source_reuse_audit.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    resolved = provenance.get("resolved_source_contract")
    if provenance.get("passed") is not True or not isinstance(resolved, Mapping):
        raise ValueError("source reuse audit lacks a passed resolved source contract")
    checkpoint = Path(str(resolved.get("checkpoint_path") or ""))
    config = Path(str(resolved.get("config_path") or ""))
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("resolved checkpoint/config is missing")
    return {
        **dict(feedback),
        "source_contract": {
            **dict(feedback.get("source_contract") or {}),
            "checkpoint_path": str(checkpoint),
            "config_path": str(config),
        },
    }


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with {completed.returncode}: {log}")


def _sha_fields(artifacts: Mapping[str, Any]) -> dict[str, str]:
    fields = {
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "tvm_module_sha256": artifacts["tvm_module"]["sha256"],
        "tvm_database_sha256": artifacts["tvm_database"]["sha256"],
    }
    if "quant_contract" in artifacts:
        fields["quant_contract_sha256"] = artifacts["quant_contract"]["sha256"]
    return fields


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if _gpu_processes(args.physical_gpu_id):
        raise RuntimeError(f"GPU{args.physical_gpu_id} is not exclusive")
    feedback_path = args.feedback_row.resolve()
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    execution_dir = feedback_path.parent
    normalized = normalize_success_row(feedback, execution_dir=execution_dir)
    artifacts = normalized["artifacts"]
    sha_fields = _sha_fields(artifacts)
    ap_env = _environment(
        physical_gpu=args.physical_gpu_id,
        heal_root=args.heal_root,
        python_site=args.python_site,
        tvm_site=args.tvm_site,
        tvm_lib_dirs=args.tvm_lib_dir,
    )
    performance_env = _tvm_environment(
        physical_gpu=args.physical_gpu_id,
        tvm_site=args.tvm_site,
        tvm_lib_dirs=args.tvm_lib_dir,
    )
    repeats = []
    for index in range(3):
        report_path = args.output_dir / f"performance_repeat_{index}.json"
        command = [
            str(args.tvm_python),
            str(REPO / "scripts/fcooper_tvm_independent_performance_v1.py"),
            "--feedback-row",
            str(feedback_path),
            "--output-json",
            str(report_path),
            "--gpu-id",
            "0",
            "--physical-gpu-id",
            str(args.physical_gpu_id),
            "--seed",
            str(args.seed + index),
        ]
        _run(
            command,
            cwd=REPO,
            env=performance_env,
            log=args.output_dir / f"performance_repeat_{index}.log",
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        repeats.append(
            {
                "gpu_index": args.physical_gpu_id,
                "latency_ms": report["latency_ms"],
                "energy_j": report["energy_j"],
                "report": binding(report_path),
                **sha_fields,
            }
        )
    ap_feedback = feedback_with_resolved_source(
        feedback, execution_dir=execution_dir
    )
    ap_command = build_ap_command(
        ap_feedback,
        output_dir=args.output_dir,
        python=args.python,
        tvm_python=args.tvm_python,
        tvm_site=args.tvm_site,
        tvm_lib_dirs=args.tvm_lib_dir,
        execution_dir=execution_dir,
    )
    _run(
        ap_command,
        cwd=args.heal_root,
        env=ap_env,
        log=args.output_dir / "full_ap.log",
    )
    ap_path = args.output_dir / "full_ap_report.json"
    prediction_path = args.output_dir / "predictions.jsonl"
    ap = json.loads(ap_path.read_text(encoding="utf-8"))
    prediction = binding(prediction_path)
    manifest = {
        "schema_version": "fcooper_tvm_gpu7_winner_validation_v1",
        "row_id": feedback["row_id"],
        "gpu_index": args.physical_gpu_id,
        "exclusive_wait": dict(args.exclusive_wait_audit),
        "performance_repeats": repeats,
        "full_ap": {
            "gpu_index": args.physical_gpu_id,
            "ap30": ap["ap30"],
            "ap50": ap["ap50"],
            "ap70": ap["ap70"],
            "report": binding(ap_path),
            "prediction": prediction,
            "prediction_sha256": prediction["sha256"],
            **sha_fields,
        },
    }
    write_immutable(args.output_dir / "validation_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-row", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu-id", type=int, default=7)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"),
    )
    parser.add_argument(
        "--python-site",
        type=Path,
        default=Path(
            "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/lib/python3.9/site-packages"
        ),
    )
    parser.add_argument(
        "--tvm-python",
        type=Path,
        default=Path("/exdata/jichengzhi/tvm310/bin/python"),
    )
    parser.add_argument(
        "--tvm-site",
        type=Path,
        default=Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"),
    )
    parser.add_argument(
        "--tvm-lib-dir",
        type=Path,
        action="append",
        default=[
            Path(
                "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/"
                "nvidia/cuda_runtime/lib"
            ),
            Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib"),
        ],
    )
    parser.add_argument(
        "--heal-root",
        type=Path,
        default=Path("/exdata/jichengzhi/heal_research/HEAL"),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--wait-for-exclusive-seconds", type=float, default=0.0)
    parser.add_argument("--exclusive-poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_path = Path("/exdata/jichengzhi/results/S5-FCO-TVM-V1/gpu7.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        args.exclusive_wait_audit = wait_for_exclusive_gpu(
            args.physical_gpu_id,
            timeout_seconds=args.wait_for_exclusive_seconds,
            poll_interval_seconds=args.exclusive_poll_seconds,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = validate(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
