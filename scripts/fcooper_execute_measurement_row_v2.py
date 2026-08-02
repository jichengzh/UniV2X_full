#!/usr/bin/env python3
"""Execute one formal recovered F-Cooper measurement row on H800."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


BASE_WIDTH = (64, 128, 256, 128, 256)
FORMAL_TASK_ID = "S5-FCO-TRT-V2"
CONTROL_TASK_PREFIX = "S6-FCO-TRT-"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def validate_recovery_training_evidence(
    *,
    report_path: Path,
    recovery_contract_path: Path,
    config_path: Path,
    initial_checkpoint_path: Path,
    recovered_checkpoint_path: Path,
) -> dict[str, Any]:
    required_files = (
        report_path,
        recovery_contract_path,
        config_path,
        initial_checkpoint_path,
        recovered_checkpoint_path,
    )
    if any(not path.is_file() for path in required_files):
        raise ValueError("recovery evidence is missing a required file")
    report = json.loads(report_path.read_text())
    contract = json.loads(recovery_contract_path.read_text())
    dataset = report.get("dataset")
    epoch_records = report.get("epoch_records")
    epochs_completed = int(report.get("epochs_completed") or 0)
    minimum_epochs = int(contract.get("minimum_epochs") or 0)
    maximum_epochs = int(contract.get("recovery_epochs") or 0)
    start_epoch = int(contract.get("start_epoch") or -1)
    if (
        report.get("schema_version") != "fcooper_recovery_training_report_v2"
        or report.get("status") != "success"
        or report.get("initialization_policy") != "scanner_dependency_l1_v2"
        or report.get("recovery_contract_sha256")
        != sha256_file(recovery_contract_path)
        or report.get("config_sha256") != sha256_file(config_path)
        or report.get("initial_checkpoint_sha256")
        != sha256_file(initial_checkpoint_path)
        or report.get("recovered_checkpoint_sha256")
        != sha256_file(recovered_checkpoint_path)
        or int(report.get("seed") or -1) != int(contract.get("seed") or -2)
        or bool(report.get("amp_fp16")) != bool(contract.get("amp_fp16"))
        or not isinstance(dataset, dict)
        or dataset.get("full_train_split") is not True
        or dataset.get("full_validation_split") is not True
        or int(dataset.get("train_samples") or 0) <= 0
        or int(dataset.get("validation_samples") or 0) <= 0
        or not dataset.get("train_root")
        or not dataset.get("validation_root")
        or not isinstance(epoch_records, list)
        or not minimum_epochs <= epochs_completed <= maximum_epochs
        or len(epoch_records) != epochs_completed
        or float(report.get("elapsed_seconds") or 0.0) <= 0.0
    ):
        raise ValueError("recovery training report contract or provenance drift")
    expected_epochs = list(
        range(start_epoch + 1, start_epoch + epochs_completed + 1)
    )
    if [int(record.get("epoch") or -1) for record in epoch_records] != expected_epochs:
        raise ValueError("recovery epoch sequence drift")
    for record in epoch_records:
        checkpoint = Path(str(record.get("checkpoint_path") or ""))
        if (
            not checkpoint.is_file()
            or record.get("checkpoint_sha256") != sha256_file(checkpoint)
            or not math.isfinite(float(record.get("train_loss", math.nan)))
            or not math.isfinite(float(record.get("validation_loss", math.nan)))
            or float(record.get("elapsed_seconds") or 0.0) <= 0.0
        ):
            raise ValueError("recovery epoch evidence drift")
    return {
        "schema_version": "fcooper_recovery_training_evidence_audit_v2",
        "passed": True,
        "report_sha256": sha256_file(report_path),
        "recovery_contract_sha256": sha256_file(recovery_contract_path),
        "epochs_completed": epochs_completed,
        "recovered_checkpoint_sha256": sha256_file(recovered_checkpoint_path),
    }


def validate_formal_request(
    request: dict[str, Any], *, row_index: int
) -> dict[str, Any]:
    if request.get("schema_version") != "stage5_measurement_request_v2":
        raise ValueError("unexpected formal request schema")
    if request.get("task_id") != FORMAL_TASK_ID:
        raise ValueError("formal request task identity drift")
    if (
        request.get("atomic_feedback") is not True
        or request.get("real_h800_measurement_required") is not True
        or int(request.get("batch_size") or 0) != 4
    ):
        raise ValueError("formal request execution contract drift")
    recorded = request.get("measurement_request_sha256")
    payload = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    if recorded != sha256_payload(payload):
        raise ValueError("formal measurement request SHA drift")
    rows = request.get("rows") or []
    if len(rows) != 4 or not 0 <= row_index < len(rows):
        raise ValueError("formal request row index or batch size drift")
    row = dict(rows[row_index])
    row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    if (
        row.get("task_id") != FORMAL_TASK_ID
        or row.get("model") != "fcooper"
        or row.get("hardware_id") != "h800"
        or row.get("task_sha256") != request.get("task_sha256")
        or request.get("row_sha256", {}).get(row_id) != sha256_payload(row)
    ):
        raise ValueError("formal request row task or identity SHA drift")
    return row


def validate_control_request(
    request: dict[str, Any], *, row_index: int
) -> dict[str, Any]:
    if (
        request.get("schema_version")
        != "stage6_fcooper_control_measurement_request_v2"
    ):
        raise ValueError("unexpected Stage6 control request schema")
    task_id = str(request.get("task_id") or "")
    rows = request.get("rows") or []
    if (
        not task_id.startswith(CONTROL_TASK_PREFIX)
        or request.get("real_h800_measurement_required") is not True
        or int(request.get("batch_size") or 0) != len(rows)
        or not 1 <= len(rows) <= 4
        or not 0 <= row_index < len(rows)
    ):
        raise ValueError("Stage6 control request execution contract drift")
    recorded = request.get("measurement_request_sha256")
    payload = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    if recorded != sha256_payload(payload):
        raise ValueError("Stage6 control measurement request SHA drift")
    row = dict(rows[row_index])
    row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    if (
        row.get("task_id") != task_id
        or row.get("model") != "fcooper"
        or row.get("hardware_id") != "h800"
        or row.get("task_sha256") != request.get("task_sha256")
        or request.get("row_sha256", {}).get(row_id) != sha256_payload(row)
    ):
        raise ValueError("Stage6 control row task or identity SHA drift")
    return row


def run(
    command: list[str],
    *,
    env: dict[str, str],
    log: Path,
    cwd: Path | None = None,
) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w") as handle:
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")
    return elapsed


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
    parser.add_argument("--recovery-contract", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    parser.add_argument(
        "--request-kind",
        choices=("t16", "stage6-control"),
        default="t16",
    )
    return parser.parse_args()


def _prepare_source(
    args: argparse.Namespace,
    *,
    width: tuple[int, ...],
    source_dir: Path,
    execution: Path,
    base_env: dict[str, str],
    timings: dict[str, float],
) -> tuple[Path, Path, Path | None]:
    tag = "x".join(map(str, width))
    config = source_dir / "config.yaml"
    checkpoint = source_dir / "recovered_checkpoint.pth"
    training_report = source_dir / "recovery_training_report.json"
    onnx_path = source_dir / f"fcooper_dense_{tag}.onnx"
    export_report = source_dir / "source_export_report.json"
    ready = source_dir / "formal_source.ready"
    if ready.is_file():
        report = json.loads(export_report.read_text())
        if (
            report.get("formal_measurement_eligible") is not True
            or report.get("checkpoint_sha256") != sha256_file(checkpoint)
            or report.get("onnx_sha256") != sha256_file(onnx_path)
        ):
            raise ValueError(f"formal source evidence drift for {tag}")
        if width != BASE_WIDTH:
            validate_recovery_training_evidence(
                report_path=training_report,
                recovery_contract_path=args.recovery_contract,
                config_path=config,
                initial_checkpoint_path=source_dir / "net_epoch_bestval_at23.pth",
                recovered_checkpoint_path=checkpoint,
            )
        timings["recovery_initialization_seconds"] = 0.0
        timings["recovery_training_seconds"] = 0.0
        timings["onnx_export_seconds"] = 0.0
        return checkpoint, onnx_path, training_report if training_report.exists() else None

    if width == BASE_WIDTH:
        shutil.copy2(args.source_config, config)
        shutil.copy2(args.source_checkpoint, checkpoint)
        timings["recovery_initialization_seconds"] = 0.0
        timings["recovery_training_seconds"] = 0.0
    else:
        initialization = source_dir / "recovery_initialization_report.json"
        if not initialization.is_file():
            timings["recovery_initialization_seconds"] = run(
                [
                    str(args.python),
                    str(args.code_root / "scripts/fcooper_recovery_v2.py"),
                    "--source-config",
                    str(args.source_config),
                    "--source-checkpoint",
                    str(args.source_checkpoint),
                    "--width",
                    ",".join(map(str, width)),
                    "--recovery-contract",
                    str(args.recovery_contract),
                    "--output-dir",
                    str(source_dir),
                ],
                env=base_env,
                log=execution / "recovery_initialization.log",
            )
        else:
            timings["recovery_initialization_seconds"] = 0.0
        initial_checkpoint = source_dir / "net_epoch_bestval_at23.pth"
        if not training_report.is_file():
            train_env = {
                **base_env,
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "PYTHONUNBUFFERED": "1",
            }
            timings["recovery_training_seconds"] = run(
                [
                    str(args.python),
                    str(args.code_root / "scripts/fcooper_recovery_train_v2.py"),
                    "--heal-root",
                    str(args.heal_root),
                    "--config",
                    str(config),
                    "--initial-checkpoint",
                    str(initial_checkpoint),
                    "--recovery-contract",
                    str(args.recovery_contract),
                    "--model-dir",
                    str(source_dir),
                    "--report",
                    str(training_report),
                ],
                env=train_env,
                log=execution / "recovery_training.log",
                cwd=args.heal_root,
            )
        else:
            timings["recovery_training_seconds"] = 0.0
        validate_recovery_training_evidence(
            report_path=training_report,
            recovery_contract_path=args.recovery_contract,
            config_path=config,
            initial_checkpoint_path=initial_checkpoint,
            recovered_checkpoint_path=checkpoint,
        )

    export_command = [
        str(args.python),
        str(args.code_root / "scripts/fcooper_export_source_v2.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--width",
        ",".join(map(str, width)),
        "--onnx",
        str(onnx_path),
        "--report",
        str(export_report),
        "--formal",
    ]
    if width == BASE_WIDTH:
        export_command.append("--original-unpruned")
    else:
        export_command.extend(["--training-report", str(training_report)])
    timings["onnx_export_seconds"] = run(
        export_command,
        env=base_env,
        log=execution / "source_export.log",
    )
    ready.touch()
    return checkpoint, onnx_path, training_report if training_report.exists() else None


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if PILOT_FRAGMENT in str(args.request_json) or PILOT_FRAGMENT in str(
        args.artifact_root
    ):
        raise ValueError("formal runner cannot read or write the pilot result root")
    request = json.loads(args.request_json.read_text())
    row = (
        validate_formal_request(request, row_index=args.row_index)
        if args.request_kind == "t16"
        else validate_control_request(request, row_index=args.row_index)
    )
    row_id = str(row["row_id"])
    width = tuple(int(value) for value in row["width"])
    tag = "x".join(map(str, width))
    row_tag = hashlib.sha256(
        f"{request['task_id']}:{row_id}".encode()
    ).hexdigest()[:16]
    execution = args.artifact_root / "execution" / row_tag
    source_dir = args.artifact_root / "sources" / tag
    execution.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    base_env = {
        **os.environ,
        "PYTHONPATH": f"{args.code_root}:{args.heal_root}",
    }
    timings: dict[str, float] = {}

    lock_dir = args.artifact_root / "source_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{tag}.lock").open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        checkpoint, onnx_path, training_report = _prepare_source(
            args,
            width=width,
            source_dir=source_dir,
            execution=execution,
            base_env=base_env,
            timings=timings,
        )
        fcntl.flock(lock_handle, fcntl.LOCK_UN)

    performance_path = execution / "performance.json"
    artifact_dir = execution / "engine"
    timings["trt_build_performance_energy_seconds"] = run(
        [
            str(args.python),
            str(args.code_root / "framework/trt_baseline/trt_profile_v1.py"),
            "--onnx",
            str(onnx_path),
            "--precision",
            str(row["q_mode"]),
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
        env=base_env,
        log=execution / "performance.log",
    )

    ap_path = execution / "ap_report.json"
    ap_env = {
        **base_env,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONUNBUFFERED": "1",
    }
    timings["full_ap_seconds"] = run(
        [
            str(args.python),
            str(args.code_root / "scripts/fcooper_trt_ap_bridge_v1.py"),
            "--config",
            str(source_dir / "config.yaml"),
            "--checkpoint-dir",
            str(source_dir),
            "--checkpoint",
            str(checkpoint),
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
        raise ValueError(f"formal AP contract failed for {row_id}")

    sys.path.insert(0, str(args.code_root / "scripts"))
    from stage35_extract_onnx_graph_features_v1 import graph_features

    actual_graph = {
        "schema": "stage35_actual_onnx_graph_features_v1",
        "group_id": row["group_id"],
        "model": "fcooper",
        "width": list(width),
        "width_schema": row["width_schema"],
        "graph_feature_provenance": "materialized_onnx_extracted_v1",
        "onnx_sha256": sha256_file(onnx_path),
        **graph_features(onnx_path),
    }
    graph_path = execution / "actual_graph_features.json"
    graph_path.write_text(json.dumps(actual_graph, indent=2, sort_keys=True) + "\n")
    calibration_file = sorted(args.calibration_dir.glob("*.npy"))[0]
    source_evidence = {
        "schema_version": "stage5_source_materialization_evidence_v2",
        "group_id": row["group_id"],
        "source_plan_sha256": row["source_evidence_sha256"],
        "status": "ready",
        "initialization_policy": (
            "original_unpruned_checkpoint"
            if width == BASE_WIDTH
            else "scanner_dependency_l1_v2"
        ),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_path": str(source_dir / "config.yaml"),
        "config_sha256": sha256_file(source_dir / "config.yaml"),
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256_file(onnx_path),
        "calibration_path": str(calibration_file),
        "calibration_sha256": sha256_file(calibration_file),
        "calibration_summary_path": str(args.calibration_summary),
        "calibration_summary_sha256": sha256_file(args.calibration_summary),
        "actual_graph_features_path": str(graph_path),
        "actual_graph_features_sha256": sha256_file(graph_path),
        "recovery_training_report_path": (
            str(training_report) if training_report is not None else None
        ),
        "recovery_training_report_sha256": (
            sha256_file(training_report) if training_report is not None else None
        ),
        "phase_timings_seconds": timings,
        "physical_gpu": args.gpu,
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
        "checkpoint_sha256": sha256_file(checkpoint),
        "recovery_training_report_sha256": source_evidence[
            "recovery_training_report_sha256"
        ],
        "recovery_training_report_path": source_evidence[
            "recovery_training_report_path"
        ],
        "phase_timings_seconds": timings,
        "builder_optimization_level": args.builder_optimization_level,
    }
    feedback["actual_feedback_row_sha256"] = sha256_payload(feedback)
    feedback_path = execution / "feedback_row.json"
    feedback_path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n")
    return {"status": "success", "row_id": row_id, "feedback": str(feedback_path)}


def main() -> int:
    args = parse_args()
    try:
        print(json.dumps(execute(args), sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "schema_version": "fcooper_formal_measurement_failure_v2",
            "status": "failure",
            "failure_class": "unclassified_pending_retry",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc().splitlines()[-20:],
        }
        failure_path = args.artifact_root / "failures"
        failure_path.mkdir(parents=True, exist_ok=True)
        target = failure_path / f"row_{args.row_index:04d}.json"
        target.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
