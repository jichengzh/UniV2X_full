#!/usr/bin/env python3
"""Execute one frozen F-Cooper measurement row on H800."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    log: Path,
    cwd: Path | None = None,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = json.loads(args.request_json.read_text())
    row = dict(request["rows"][args.row_index])
    row_id = str(row["row_id"])
    width = [int(value) for value in row["width"]]
    tag = "x".join(map(str, width))
    row_tag = hashlib.sha256(row_id.encode()).hexdigest()[:16]
    execution = args.artifact_root / "execution" / row_tag
    source_dir = args.artifact_root / "sources" / tag
    execution.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": f"{args.code_root}:{args.heal_root}",
    }
    onnx_path = source_dir / f"fcooper_dense_{tag}.onnx"
    lock_dir = args.artifact_root / "source_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{tag}.lock").open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        ready_files = (
            source_dir / "source.ready",
            source_dir / "config.yaml",
            source_dir / "net_epoch_bestval_at23.pth",
            onnx_path,
            source_dir / "materialization_report.json",
        )
        if not all(path.is_file() for path in ready_files):
            run(
                [
                    str(args.python),
                    str(args.code_root / "scripts/fcooper_materialize_source_v1.py"),
                    "--source-config",
                    str(args.source_config),
                    "--source-checkpoint",
                    str(args.source_checkpoint),
                    "--width",
                    ",".join(map(str, width)),
                    "--output-dir",
                    str(source_dir),
                ],
                env=env,
                log=execution / "materialize.log",
            )
        calibration_link = source_dir / "calibration"
        summary_link = source_dir / "calibration_summary.json"
        if calibration_link.is_symlink() and calibration_link.resolve() != args.calibration_dir.resolve():
            calibration_link.unlink()
        if not calibration_link.exists():
            calibration_link.symlink_to(args.calibration_dir, target_is_directory=True)
        if summary_link.is_symlink() and summary_link.resolve() != args.calibration_summary.resolve():
            summary_link.unlink()
        if not summary_link.exists():
            summary_link.symlink_to(args.calibration_summary)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)

    performance_path = execution / "performance.json"
    artifact_dir = execution / "engine"
    precision = str(row["q_mode"])
    run(
        [
            str(args.python),
            str(args.code_root / "framework/trt_baseline/trt_profile_v1.py"),
            "--onnx",
            str(onnx_path),
            "--precision",
            precision,
            "--gpu",
            str(args.gpu),
            "--calib-dir",
            str(args.calibration_dir),
            "--calibration-dataset",
            "OPV2V-validate",
            "--builder-optimization-level",
            str(args.builder_optimization_level),
            "--warmup",
            "20",
            "--iters",
            "300",
            "--repeat",
            "5",
            "--energy-secs",
            "5",
            "--artifact-dir",
            str(artifact_dir),
            "--out",
            str(performance_path),
        ],
        env=env,
        log=execution / "performance.log",
    )
    ap_path = execution / "ap_report.json"
    ap_env = {**env, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    run(
        [
            str(args.python),
            str(args.code_root / "scripts/fcooper_trt_ap_bridge_v1.py"),
            "--config",
            str(source_dir / "config.yaml"),
            "--checkpoint-dir",
            str(source_dir),
            "--engine",
            str(artifact_dir / "compiled.engine"),
            "--output-json",
            str(ap_path),
            "--num-workers",
            "4",
        ],
        env=ap_env,
        log=execution / "ap.log",
        cwd=args.heal_root,
    )
    performance = json.loads(performance_path.read_text())
    ap = json.loads(ap_path.read_text())
    if ap.get("status") != "success_full" or ap.get("fallback_samples") != 0:
        raise ValueError(f"AP contract failed for {row_id}")

    sys.path.insert(0, str(args.code_root / "scripts"))
    from stage35_extract_onnx_graph_features_v1 import graph_features

    actual_graph = {
        "schema": "stage35_actual_onnx_graph_features_v1",
        "group_id": row["group_id"],
        "model": "fcooper",
        "width": width,
        "width_schema": row["width_schema"],
        "graph_feature_provenance": "materialized_onnx_extracted_v1",
        **graph_features(onnx_path),
    }
    graph_path = execution / "actual_graph_features.json"
    graph_path.write_text(json.dumps(actual_graph, indent=2, sort_keys=True) + "\n")
    calibration_file = sorted(args.calibration_dir.glob("*.npy"))[0]
    source_evidence = {
        "schema_version": "stage5_source_materialization_evidence_v1",
        "group_id": row["group_id"],
        "source_plan_sha256": row["source_evidence_sha256"],
        "status": "ready",
        "checkpoint_path": str(source_dir / "net_epoch_bestval_at23.pth"),
        "checkpoint_sha256": sha256_file(source_dir / "net_epoch_bestval_at23.pth"),
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "calibration_path": str(calibration_file),
        "calibration_sha256": sha256_file(calibration_file),
        "calibration_summary_path": str(args.calibration_summary),
        "calibration_summary_sha256": sha256_file(args.calibration_summary),
        "actual_graph_features_path": str(graph_path),
        "actual_graph_features_sha256": sha256_file(graph_path),
    }
    source_evidence_path = execution / "source_evidence.json"
    source_evidence_path.write_text(
        json.dumps(source_evidence, indent=2, sort_keys=True) + "\n"
    )
    feedback = {
        **row,
        "training_source": "online_feedback",
        "terminal_status": "measured_success_gold",
        "latency_ms": float(performance["lat_p50_ms"]),
        "energy_j": float(performance["energy_j"]),
        "ap30": float(ap["ap30"]),
        "ap50": float(ap["ap50"]),
        "ap70": float(ap["ap70"]),
        "measurement_request_row_sha256": request["row_sha256"][row_id],
        "performance_result_json": str(performance_path),
        "performance_result_sha256": sha256_file(performance_path),
        "ap_report_path": str(ap_path),
        "ap_report_sha256": sha256_file(ap_path),
        "materialized_source_evidence_path": str(source_evidence_path),
        "materialized_source_evidence_sha256": sha256_file(source_evidence_path),
        "materialized_graph_features_sha256": sha256_payload(actual_graph),
        "graph_features": actual_graph,
        "engine_path": str(artifact_dir / "compiled.engine"),
        "engine_sha256": sha256_file(artifact_dir / "compiled.engine"),
        "builder_optimization_level": args.builder_optimization_level,
    }
    feedback["actual_feedback_row_sha256"] = sha256_payload(feedback)
    feedback_path = execution / "feedback_row.json"
    feedback_path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "success", "row_id": row_id, "feedback": str(feedback_path)}))


if __name__ == "__main__":
    main()
