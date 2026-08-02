#!/usr/bin/env python3
"""Execute one SHA-bound F-Cooper TVM/H800 formal measurement row."""

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
from typing import Any, Mapping, Sequence


FORMAL_TASK_ID = "S5-FCO-TVM-V1"
CONTROL_TASK_PREFIX = "S6-FCO-TVM-"
BASE_WIDTH = (64, 128, 256, 128, 256)
SUCCESS = "measured_success_gold"
TRT_PERFORMANCE_FIELDS = {
    "engine_path",
    "engine_sha256",
    "tactic",
    "tactics",
    "trt_latency_ms",
    "trt_energy_j",
    "trt_ap70",
    "formal_incumbent",
    "winner",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def reject_cross_backend_performance(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = TRT_PERFORMANCE_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(
                "TRT performance evidence is forbidden in TVM input: "
                + ", ".join(sorted(forbidden))
            )
        for child in value.values():
            reject_cross_backend_performance(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            reject_cross_backend_performance(child)


def validate_tvm_request(
    request: Mapping[str, Any],
    *,
    row_index: int,
    request_kind: str,
) -> dict[str, Any]:
    if request_kind not in {"t16", "stage6-control"}:
        raise ValueError(f"unsupported request kind: {request_kind}")
    expected_schema = (
        "stage5_measurement_request_v2"
        if request_kind == "t16"
        else "stage6_fcooper_tvm_control_measurement_request_v1"
    )
    task_id = str(request.get("task_id") or "")
    rows = request.get("rows")
    if request.get("schema_version") != expected_schema or not isinstance(rows, list):
        raise ValueError("unexpected TVM measurement request schema")
    if request_kind == "t16":
        valid_identity = task_id == FORMAL_TASK_ID and len(rows) == 4
    else:
        valid_identity = task_id.startswith(CONTROL_TASK_PREFIX) and 1 <= len(rows) <= 4
    if (
        not valid_identity
        or request.get("atomic_feedback") is not True
        or request.get("real_h800_measurement_required") is not True
        or int(request.get("batch_size") or 0) != len(rows)
        or not 0 <= row_index < len(rows)
    ):
        raise ValueError("TVM measurement request execution contract drift")
    recorded = request.get("measurement_request_sha256")
    unsigned = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    if recorded != sha256_payload(unsigned):
        raise ValueError("TVM measurement request SHA drift")
    row = dict(rows[row_index])
    row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
    if (
        row.get("task_id") != task_id
        or row.get("task_sha256") != request.get("task_sha256")
        or row.get("model") != "fcooper"
        or row.get("hardware_id") != "h800"
        or row.get("dispatch_key") != "tvm_auto"
        or not str(row.get("capability_profile_id") or "").startswith("h800-tvm")
    ):
        raise ValueError("row is not bound to the fixed F-Cooper H800 TVM profile")
    if request.get("row_sha256", {}).get(row_id) != sha256_payload(row):
        raise ValueError("TVM measurement row SHA drift")
    reject_cross_backend_performance(row)
    return row


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except (json.JSONDecodeError, ValueError):
        return {}


def audit_reused_source(
    row: Mapping[str, Any],
    *,
    recovery_contract: Path | None = None,
    width: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if width is None:
        width = tuple(int(value) for value in row.get("width") or ())
    contract = row.get("source_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("source contract is missing")
    paths = {
        "checkpoint": Path(str(contract.get("checkpoint_path") or "")),
        "config": Path(str(contract.get("config_path") or "")),
        "onnx": Path(str(contract.get("onnx_path") or "")),
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("backend-neutral source is missing checkpoint/config/ONNX")
    source_report = Path(
        str(
            contract.get("source_export_report")
            or paths["onnx"].parent / "source_export_report.json"
        )
    )
    if not source_report.is_file():
        raise ValueError("formal source export report is missing")
    report = read_json(source_report)
    if (
        report.get("status") != "success"
        or report.get("formal_measurement_eligible") is not True
        or report.get("onnx_sha256") != sha256_file(paths["onnx"])
    ):
        raise ValueError("formal source export provenance drift")
    declared_paths = {
        "checkpoint": Path(str(report.get("checkpoint_path") or paths["checkpoint"])),
        "config": Path(str(report.get("config_path") or paths["config"])),
        "onnx": Path(str(report.get("onnx_path") or paths["onnx"])),
    }
    if any(not path.is_file() for path in declared_paths.values()):
        raise ValueError("formal source export declares a missing source artifact")
    for name, sha_field in (
        ("checkpoint", "checkpoint_sha256"),
        ("config", "config_sha256"),
        ("onnx", "onnx_sha256"),
    ):
        expected = report.get(sha_field)
        if expected and expected != sha256_file(declared_paths[name]):
            raise ValueError(f"formal source {name} SHA drift")
    paths = declared_paths
    recovery_report = Path(
        str(
            report.get("training_report_path")
            or contract.get("recovery_training_report_path")
            or paths["onnx"].parent / "recovery_training_report.json"
        )
    )
    recovery = _optional_json(recovery_report)
    if width != (64, 128, 256, 128, 256):
        if recovery_contract is None or not recovery_report.is_file():
            raise ValueError("pruned source lacks frozen recovery-contract evidence")
        from fcooper_execute_measurement_row_v2 import (
            validate_recovery_training_evidence,
        )

        validate_recovery_training_evidence(
            report_path=recovery_report,
            recovery_contract_path=recovery_contract,
            config_path=paths["config"],
            initial_checkpoint_path=paths["onnx"].parent / "net_epoch_bestval_at23.pth",
            recovered_checkpoint_path=paths["checkpoint"],
        )
    saved = float(recovery.get("elapsed_seconds") or 0.0)
    evidence = {
        name: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    evidence["source_export_report"] = {
        "path": str(source_report.resolve()),
        "sha256": sha256_file(source_report),
    }
    if recovery_report.is_file():
        evidence["recovery_training_report"] = {
            "path": str(recovery_report.resolve()),
            "sha256": sha256_file(recovery_report),
        }
    return {
        "schema_version": "fcooper_tvm_backend_neutral_source_reuse_audit_v1",
        "passed": True,
        "backend_neutral_only": True,
        "source_profile": "fcooper_trt_v2_recovery_contract_only",
        "reused_artifacts": evidence,
        "recovery_training_seconds_saved": max(saved, 0.0),
        "trt_performance_labels_reused": False,
        "trt_compiled_artifacts_reused": False,
        "trt_predictions_or_ap_reused": False,
        "resolved_source_contract": {
            "checkpoint_path": str(paths["checkpoint"].resolve()),
            "config_path": str(paths["config"].resolve()),
            "onnx_path": str(paths["onnx"].resolve()),
            "source_export_report": str(source_report.resolve()),
            "recovery_training_report_path": (
                str(recovery_report.resolve()) if recovery_report.is_file() else None
            ),
        },
    }


def ensure_backend_neutral_source(
    args: argparse.Namespace,
    *,
    row: Mapping[str, Any],
    width: tuple[int, ...],
    execution: Path,
    base_env: Mapping[str, str],
    timings: dict[str, float],
    prepare_source: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        audit = audit_reused_source(
            row, recovery_contract=args.recovery_contract, width=width
        )
        resolved_row = {
            **dict(row),
            "source_contract": audit["resolved_source_contract"],
        }
        return resolved_row, {
            **audit,
            "source_reused_from_trt_v2": True,
            "source_generated_in_tvm_v1": False,
        }
    except (FileNotFoundError, ValueError) as reuse_error:
        if prepare_source is None:
            from fcooper_execute_measurement_row_v2 import (
                _prepare_source as prepare_source,
            )

        tag = "x".join(map(str, width))
        source_dir = args.artifact_root / "sources" / tag
        source_dir.mkdir(parents=True, exist_ok=True)
        lock_dir = args.artifact_root / "source_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / f"{tag}.lock").open("w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            checkpoint, onnx_path, training_report = prepare_source(
                args,
                width=width,
                source_dir=source_dir,
                execution=execution,
                base_env=dict(base_env),
                timings=timings,
            )
        source_contract = {
            **dict(row.get("source_contract") or {}),
            "checkpoint_path": str(checkpoint),
            "config_path": str(source_dir / "config.yaml"),
            "onnx_path": str(onnx_path),
            "source_export_report": str(source_dir / "source_export_report.json"),
            "recovery_training_report_path": (
                str(training_report) if training_report is not None else None
            ),
        }
        prepared_row = {**dict(row), "source_contract": source_contract}
        audit = audit_reused_source(
            prepared_row, recovery_contract=args.recovery_contract, width=width
        )
        return prepared_row, {
            **audit,
            "source_profile": "fcooper_tvm_v1_frozen_recovery_contract",
            "source_reused_from_trt_v2": False,
            "source_generated_in_tvm_v1": True,
            "recovery_training_seconds_saved": 0.0,
            "reuse_rejection": f"{type(reuse_error).__name__}: {reuse_error}",
        }


def build_route_command(
    *,
    python: Path,
    code_root: Path,
    q_mode: str,
    onnx: Path,
    out_dir: Path,
    label: str,
    width: tuple[int, ...],
    gpu: int,
    max_trials: int,
    quant_contract: Path | None = None,
) -> list[str]:
    common = [
        "--label",
        label,
        "--width",
        ",".join(map(str, width)),
        "--onnx",
        str(onnx),
        "--out-dir",
        str(out_dir),
        "--gpu",
        str(gpu),
    ]
    if q_mode in {"fp16", "fp32"}:
        return [
            str(python),
            str(code_root / "scripts/stage2_route_b_fp16_auto_runner.py"),
            *common,
            "--precision",
            q_mode,
            "--fix",
            "none",
            "--max-trials",
            str(max_trials),
            "--measure-energy",
        ]
    if q_mode != "int8":
        raise ValueError(f"unsupported q_mode: {q_mode}")
    if quant_contract is None:
        raise ValueError("INT8 route requires a TVM quantization contract")
    return [
        str(python),
        str(code_root / "scripts/stage2_route_b_int8_auto_decomp.py"),
        *common,
        "--tensor-quant-params-json",
        str(quant_contract),
        "--max-trials",
        str(max_trials),
        "--tuning-work-dir",
        str(out_dir / label / "tuning_database"),
        "--measure-energy",
    ]


def build_quant_contract_command(
    *,
    python: Path,
    code_root: Path,
    onnx: Path,
    calibration_summary: Path,
    calibration_dir: Path,
    output_json: Path,
) -> list[str]:
    return [
        str(python),
        str(code_root / "scripts/fcooper_tvm_int8_quant_contract_v1.py"),
        "--onnx",
        str(onnx),
        "--calibration-summary",
        str(calibration_summary),
        "--calibration-dir",
        str(calibration_dir),
        "--output-json",
        str(output_json),
        "--max-outputs-per-run",
        "4",
    ]


def tvm_worker_options(
    *,
    tvm_python: Path,
    tvm_site: Path,
    tvm_lib_dirs: Sequence[Path],
) -> list[str]:
    options = [
        "--tvm-worker-python",
        str(tvm_python),
        "--tvm-site",
        str(tvm_site),
    ]
    for path in tvm_lib_dirs:
        options.extend(["--tvm-lib-dir", str(path)])
    return options


def build_fp16_ap_command(
    *,
    python: Path,
    tvm_python: Path,
    code_root: Path,
    config: Path,
    checkpoint: Path,
    artifact: Path,
    input_shape: str,
    output_shape: str,
    precision: str,
    artifact_output_dtype: str,
    output_json: Path,
    gpu_id: int,
    tvm_site: Path,
    tvm_lib_dirs: Sequence[Path],
) -> list[str]:
    return [
        str(python),
        str(code_root / "scripts/fcooper_tvm_fp16_ap_bridge_v1.py"),
        "--config",
        str(config),
        "--checkpoint-dir",
        str(checkpoint.parent),
        "--checkpoint",
        str(checkpoint),
        "--artifact",
        str(artifact),
        "--input-shape",
        input_shape,
        "--output-shape",
        output_shape,
        "--precision",
        precision,
        "--artifact-output-dtype",
        artifact_output_dtype,
        "--output-json",
        str(output_json),
        "--gpu-id",
        str(gpu_id),
        *tvm_worker_options(
            tvm_python=tvm_python,
            tvm_site=tvm_site,
            tvm_lib_dirs=tvm_lib_dirs,
        ),
    ]


def route_metrics(route: Mapping[str, Any]) -> tuple[float, float]:
    if route.get("status") != "success" or route.get("build_success") is not True:
        raise ValueError("TVM route did not build and run successfully")
    latency = float((route.get("latency") or {}).get("latency_ms_p50", math.nan))
    energy = route.get("energy") or {}
    if energy.get("status") not in {None, "success"}:
        raise ValueError("TVM energy measurement failed")
    energy_j = float(
        energy.get("joules_per_inference", energy.get("energy_J", math.nan))
    )
    if not math.isfinite(latency) or not math.isfinite(energy_j):
        raise ValueError("TVM latency or energy is non-finite")
    return latency, energy_j


def first_route_shape(raw: Any, fallback: list[int]) -> list[int]:
    if isinstance(raw, Mapping):
        raw = next(iter(raw.values()), fallback)
    elif isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
        raw = raw[0]
    if not isinstance(raw, (list, tuple)):
        return fallback
    return [int(value) for value in raw]


def run(
    command: list[str],
    *,
    env: Mapping[str, str],
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
            env=dict(env),
            cwd=cwd,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log}"
        )
    return elapsed


def _route_paths(route_root: Path, label: str, q_mode: str) -> tuple[Path, Path]:
    label_dir = route_root / label
    if q_mode == "int8":
        return (
            label_dir / "route_b_int8_auto_decomp_result.json",
            label_dir / "route_b_int8_auto_decomp.vmexec",
        )
    return (
        label_dir / f"route_b_{q_mode}_auto_result.json",
        label_dir / f"route_b_{q_mode}_auto.so",
    )


def archive_stale_route_attempt(*, route_root: Path, label: str) -> Path | None:
    label_dir = Path(route_root) / label
    if not label_dir.exists():
        return None
    archive_root = Path(route_root) / "failed_attempt_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    sequence = 1
    while True:
        target = archive_root / f"{label}.attempt{sequence:02d}"
        if not target.exists():
            break
        sequence += 1
    shutil.move(str(label_dir), str(target))
    return target


def _extract_graph_features(code_root: Path, onnx_path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(code_root / "scripts"))
    from stage35_extract_onnx_graph_features_v1 import graph_features

    return {
        "schema": "stage35_actual_onnx_graph_features_v1",
        "group_id": row["group_id"],
        "model": "fcooper",
        "width": list(row["width"]),
        "width_schema": list(row["width_schema"]),
        "graph_feature_provenance": "materialized_tvm_prepared_onnx_extracted_v1",
        "onnx_sha256": sha256_file(onnx_path),
        **graph_features(onnx_path),
    }


def onnx_io_shapes(path: Path) -> tuple[list[int], list[int]]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)

    def shape(value: Any) -> list[int]:
        dimensions = value.type.tensor_type.shape.dim
        result = [int(item.dim_value) for item in dimensions]
        if not result or any(item <= 0 for item in result):
            raise ValueError("F-Cooper TVM measurement requires static ONNX IO shapes")
        return result

    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("F-Cooper TVM measurement requires one ONNX input and output")
    return shape(model.graph.input[0]), shape(model.graph.output[0])


def onnx_io_dtypes(path: Path) -> tuple[str, str]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)

    def dtype(value: Any) -> str:
        return str(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type))

    return dtype(model.graph.input[0]), dtype(model.graph.output[0])


def _ap_values(report: Mapping[str, Any]) -> tuple[float, float, float]:
    if report.get("status") not in {"success_full", "success"}:
        raise ValueError("TVM full AP execution did not pass")
    if int(report.get("fallback_samples") or 0):
        raise ValueError("TVM full AP used forbidden fallback")
    values = tuple(float(report[name]) for name in ("ap30", "ap50", "ap70"))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("TVM AP report contains non-finite values")
    return values


def execute(args: argparse.Namespace) -> dict[str, Any]:
    request = read_json(args.request_json)
    row = validate_tvm_request(
        request,
        row_index=args.row_index,
        request_kind=args.request_kind,
    )
    request_row = dict(row)
    row_id = str(row["row_id"])
    width = tuple(int(value) for value in row["width"])
    q_mode = str(row["q_mode"])
    row_tag = hashlib.sha256(
        f"{request['task_id']}:{row_id}".encode()
    ).hexdigest()[:16]
    execution = args.artifact_root / "execution" / row_tag
    execution.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {
        "recovery_training_seconds": 0.0,
    }
    source_env = {
        **os.environ,
        "PYTHONPATH": f"{args.code_root}:{args.heal_root}",
        "PYTHONUNBUFFERED": "1",
    }
    tvm_env = {
        **os.environ,
        "PYTHONPATH": f"{args.code_root}:{args.tvm_site}",
        "LD_LIBRARY_PATH": ":".join(
            [*(str(path) for path in args.tvm_lib_dir), os.environ.get("LD_LIBRARY_PATH", "")]
        ).rstrip(":"),
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONUNBUFFERED": "1",
    }
    bridge_env = {
        **os.environ,
        "PYTHONPATH": f"{args.code_root}:{args.heal_root}:{args.python_site}",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONUNBUFFERED": "1",
    }
    row, source_audit = ensure_backend_neutral_source(
        args,
        row=row,
        width=width,
        execution=execution,
        base_env=source_env,
        timings=timings,
    )
    timings["recovery_training_seconds_saved"] = float(
        source_audit["recovery_training_seconds_saved"]
    )
    source_audit_path = execution / "source_reuse_audit.json"
    write_json(source_audit_path, source_audit)
    source = row["source_contract"]
    source_onnx = Path(str(source["onnx_path"]))
    source_report = Path(
        str(source.get("source_export_report") or source_onnx.parent / "source_export_report.json")
    )
    prepared = execution / "prepared" / "fcooper_tvm.onnx"
    prepare_report = execution / "prepared" / "prepare_report.json"
    timings["tvm_source_preparation_seconds"] = run(
        [
            str(args.python),
            str(args.code_root / "scripts/fcooper_prepare_tvm_source_v1.py"),
            "--source-onnx",
            str(source_onnx),
            "--source-export-report",
            str(source_report),
            "--source-export-report-sha256",
            sha256_file(source_report),
            "--output-onnx",
            str(prepared),
            "--report",
            str(prepare_report),
            "--decompose-convtranspose",
        ],
        env=source_env,
        log=execution / "prepare.log",
    )
    graph = _extract_graph_features(args.code_root, prepared, row)
    graph_path = execution / "actual_graph_features.json"
    write_json(graph_path, graph)

    quant_contract = None
    if q_mode == "int8":
        quant_contract = execution / "quant" / "quant_contract.json"
        timings["tvm_quant_calibration_seconds"] = run(
            build_quant_contract_command(
                python=args.python,
                code_root=args.code_root,
                onnx=prepared,
                calibration_summary=args.calibration_summary,
                calibration_dir=args.calibration_dir,
                output_json=quant_contract,
            ),
            env=source_env,
            log=execution / "quant_calibration.log",
        )
    label = f"{row_tag}_{q_mode}"
    route_root = execution / "route"
    stale_route_archive = archive_stale_route_attempt(
        route_root=route_root,
        label=label,
    )
    route_command = build_route_command(
        python=args.tvm_python,
        code_root=args.code_root,
        q_mode=q_mode,
        onnx=prepared,
        out_dir=route_root,
        label=label,
        width=width,
        gpu=args.gpu,
        max_trials=args.max_trials,
        quant_contract=quant_contract,
    )
    timings["tvm_build_tune_performance_energy_seconds"] = run(
        route_command,
        env=tvm_env,
        log=execution / "route.log",
    )
    route_result_path, artifact_path = _route_paths(route_root, label, q_mode)
    route_result = read_json(route_result_path)
    latency_ms, energy_j = route_metrics(route_result)

    config = Path(str(source["config_path"]))
    checkpoint = Path(str(source["checkpoint_path"]))
    recovery_entry = source_audit["reused_artifacts"].get(
        "recovery_training_report"
    )
    ap_path = execution / "ap_report.json"
    onnx_input_shape, onnx_output_shape = onnx_io_shapes(prepared)
    onnx_input_dtype, onnx_output_dtype = onnx_io_dtypes(prepared)
    if onnx_input_dtype != "float32":
        raise ValueError(f"prepared F-Cooper ONNX input dtype must be float32, got {onnx_input_dtype}")
    input_shape = ",".join(
        map(str, first_route_shape(route_result.get("input_shapes"), onnx_input_shape))
    )
    output_shape = ",".join(
        map(str, first_route_shape(route_result.get("output_shapes"), onnx_output_shape))
    )
    if q_mode == "int8":
        runtime_weights_payload = route_result.get("runtime_weights")
        if not isinstance(runtime_weights_payload, Mapping):
            raise ValueError("INT8 route result is missing runtime weight evidence")
        runtime_weights = Path(str(runtime_weights_payload.get("path") or ""))
        sanity_path = execution / "numeric_sanity.json"
        int8_bridge_base = [
            str(args.python),
            str(args.code_root / "scripts/fcooper_tvm_int8_ap_bridge_v1.py"),
            "--config", str(config),
            "--checkpoint-dir", str(checkpoint.parent),
            "--checkpoint", str(checkpoint),
            "--artifact", str(artifact_path),
            "--route-result", str(route_result_path),
            "--runtime-weights", str(runtime_weights),
            "--quant-contract", str(quant_contract),
            *tvm_worker_options(
                tvm_python=args.tvm_python,
                tvm_site=args.tvm_site,
                tvm_lib_dirs=args.tvm_lib_dir,
            ),
        ]
        sanity_command = [
            *int8_bridge_base,
            "--mode", "sanity",
            "--output-json", str(sanity_path),
            "--gpu-id", "0",
        ]
        timings["numeric_sanity_seconds"] = run(
            sanity_command,
            env=bridge_env,
            log=execution / "numeric_sanity.log",
            cwd=args.heal_root,
        )
        timings["full_ap_seconds"] = run(
            [
                *int8_bridge_base,
                "--mode", "full",
                "--output-json", str(ap_path),
                "--sanity-report", str(sanity_path),
                "--sanity-report-sha256", sha256_file(sanity_path),
                "--gpu-id", "0",
            ],
            env=bridge_env,
            log=execution / "ap.log",
            cwd=args.heal_root,
        )
    else:
        timings["full_ap_seconds"] = run(
            build_fp16_ap_command(
                python=args.python,
                tvm_python=args.tvm_python,
                code_root=args.code_root,
                config=config,
                checkpoint=checkpoint,
                artifact=artifact_path,
                input_shape=input_shape,
                output_shape=output_shape,
                precision=q_mode,
                artifact_output_dtype=onnx_output_dtype,
                output_json=ap_path,
                gpu_id=0,
                tvm_site=args.tvm_site,
                tvm_lib_dirs=args.tvm_lib_dir,
            ),
            env=bridge_env,
            log=execution / "ap.log",
            cwd=args.heal_root,
        )
    ap_report = read_json(ap_path)
    ap30, ap50, ap70 = _ap_values(ap_report)
    evidence = {
        "schema_version": "fcooper_tvm_materialization_evidence_v1",
        "row_id": row_id,
        "source_plan_sha256": request_row.get("source_evidence_sha256"),
        "recovery_contract_path": str(args.recovery_contract.resolve()),
        "recovery_contract_sha256": sha256_file(args.recovery_contract),
        "resolved_source_contract": source,
        "source_reuse_audit_path": str(source_audit_path),
        "source_reuse_audit_sha256": sha256_file(source_audit_path),
        "prepared_onnx_path": str(prepared),
        "prepared_onnx_sha256": sha256_file(prepared),
        "prepare_report_path": str(prepare_report),
        "prepare_report_sha256": sha256_file(prepare_report),
        "actual_graph_features_path": str(graph_path),
        "actual_graph_features_sha256": sha256_file(graph_path),
        "quant_contract_path": str(quant_contract) if quant_contract else None,
        "quant_contract_sha256": sha256_file(quant_contract) if quant_contract else None,
        "route_result_path": str(route_result_path),
        "route_result_sha256": sha256_file(route_result_path),
        "tvm_artifact_path": str(artifact_path),
        "tvm_artifact_sha256": sha256_file(artifact_path),
        "phase_timings_seconds": timings,
        "stale_route_archive_path": (
            str(stale_route_archive) if stale_route_archive is not None else None
        ),
        "physical_gpu": args.gpu,
    }
    evidence_path = execution / "tvm_evidence.json"
    write_json(evidence_path, evidence)
    feedback = {
        **request_row,
        "backend": "tvm_auto",
        "training_source": "online_feedback",
        "terminal_status": SUCCESS,
        "latency_ms": latency_ms,
        "energy_j": energy_j,
        "ap30": ap30,
        "ap50": ap50,
        "ap70": ap70,
        "measurement_request_row_sha256": request["row_sha256"][row_id],
        "performance_result_json": str(route_result_path),
        "performance_result_sha256": sha256_file(route_result_path),
        "ap_report_path": str(ap_path),
        "ap_report_sha256": sha256_file(ap_path),
        "materialized_source_evidence_path": str(evidence_path),
        "materialized_source_evidence_sha256": sha256_file(evidence_path),
        "materialized_graph_features_sha256": sha256_payload(graph),
        "graph_features": graph,
        "tvm_artifact_path": str(artifact_path),
        "tvm_artifact_sha256": sha256_file(artifact_path),
        "quant_contract_path": str(quant_contract) if quant_contract else None,
        "quant_contract_sha256": sha256_file(quant_contract) if quant_contract else None,
        "checkpoint_sha256": sha256_file(checkpoint),
        "resolved_source_evidence_sha256": sha256_file(source_audit_path),
        "recovery_training_report_path": (
            recovery_entry["path"] if recovery_entry else None
        ),
        "recovery_training_report_sha256": (
            recovery_entry["sha256"] if recovery_entry else None
        ),
        "phase_timings_seconds": timings,
        "tvm_trials": args.max_trials,
        "tvm_max_trials": args.max_trials,
    }
    feedback["actual_feedback_row_sha256"] = sha256_payload(feedback)
    feedback_path = execution / "feedback_row.json"
    write_json(feedback_path, feedback)
    return {"status": "success", "row_id": row_id, "feedback": str(feedback_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--request-kind", choices=("t16", "stage6-control"), default="t16")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-trials", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--python-site", type=Path, required=True)
    parser.add_argument("--tvm-python", type=Path, required=True)
    parser.add_argument("--tvm-site", type=Path, required=True)
    parser.add_argument("--tvm-lib-dir", type=Path, action="append", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-contract", type=Path, required=True)
    args = parser.parse_args()
    if args.max_trials < 0:
        parser.error("--max-trials must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        print(json.dumps(execute(args), sort_keys=True))
        return 0
    except Exception as error:
        failure = {
            "schema_version": "fcooper_tvm_formal_measurement_failure_v1",
            "status": "failure",
            "failure_class": "unclassified_pending_retry",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc().splitlines()[-30:],
        }
        target = args.artifact_root / "failures" / (
            f"{args.request_json.stem}_row_{args.row_index:02d}.json"
        )
        write_json(target, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
