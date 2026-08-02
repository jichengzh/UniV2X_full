#!/usr/bin/env python3
"""Run Pyramid TVM INT8 numeric sanity or gated full HEAL AP evaluation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage2_h800_native_int8_real_activation_bridge as stage2_bridge  # noqa: E402


SCHEMA = "stage3_pyramid_tvm_int8_ap_numeric_gate_v3"
SANITY_SAMPLES = 16
FULL_SAMPLES = 1789
EXECUTION_ABI = "relax_vm_return"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_compiled_artifact(compiled_artifact: Path, output_dir: Path) -> Path:
    source = Path(compiled_artifact).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if source.suffix != ".vmexec":
        return source
    destination = output_dir / f"{source.stem}.so"
    shutil.copy2(source, destination)
    return destination


def _archive_key_for_arg(item: dict[str, Any], archive_keys: set[str]) -> str | None:
    candidates = (item.get("arg_name"), item.get("initializer_name"))
    return next((str(value) for value in candidates if value is not None and str(value) in archive_keys), None)


def build_audit_inventory(
    *, runtime_arg_plan: list[dict[str, Any]], runtime_weight_archive_path: Path,
    route_report_path: Path,
) -> dict[str, Any]:
    if not runtime_arg_plan:
        raise ValueError("route report runtime_arg_plan must be non-empty")
    archive_path = Path(runtime_weight_archive_path).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        archive_keys = set(archive.files)
    key_plan: dict[str, str] = {}
    missing: list[str] = []
    for item in runtime_arg_plan:
        if str(item.get("role")) not in {"weight_input", "bias_input"}:
            continue
        arg_name = str(item.get("arg_name") or "")
        archive_key = _archive_key_for_arg(item, archive_keys)
        if not arg_name or archive_key is None:
            missing.append(arg_name or "<missing_arg_name>")
        else:
            key_plan[arg_name] = archive_key
    if missing:
        raise ValueError(f"runtime weight archive missing keys for: {','.join(missing)}")
    return {
        "schema": f"{SCHEMA}_audit_inventory_v1",
        "source_route_report": str(Path(route_report_path).resolve()),
        "runtime_weight_archive_path": str(archive_path),
        "runtime_weight_archive_keys": key_plan,
        "runtime_arg_plan": runtime_arg_plan,
        "execution_abi": EXECUTION_ABI,
    }


def build_tensor_quant_params(runtime_arg_plan: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    params: dict[str, Any] = {}
    blockers: list[str] = []
    quantized_items = [
        item for item in runtime_arg_plan
        if str(item.get("role")) in {"graph_input", "graph_output"}
    ]
    if not quantized_items:
        return {}, ["missing_graph_input_or_output_quant_params"]
    for item in quantized_items:
        name = str(item.get("arg_name") or "")
        quant = item.get("quantization")
        if not name or not isinstance(quant, dict):
            blockers.append(f"missing_tensor_quant_params:{name or '<missing_arg_name>'}")
            continue
        try:
            scale = float(quant["scale"])
            zero_point = int(quant.get("zero_point", 0))
        except (KeyError, TypeError, ValueError, OverflowError):
            blockers.append(f"invalid_tensor_quant_params:{name}")
            continue
        if not math.isfinite(scale) or scale <= 0.0 or not 0 <= zero_point <= 255:
            blockers.append(f"invalid_tensor_quant_params:{name}")
            continue
        params[name] = {
            "scale": scale,
            "zero_point": zero_point,
            "source": str(quant.get("source") or "route_b_runtime_arg_plan"),
        }
    return params, blockers


def evaluate_result_gates(
    stage2_report: dict[str, Any], *, tensor_quant_params_valid: bool,
    requested_samples: int, full_ap_min_samples: int,
) -> dict[str, Any]:
    processed = int(stage2_report.get("processed_samples") or 0)
    failed = int(stage2_report.get("failed_samples") or 0)
    blockers: list[str] = []
    if not tensor_quant_params_valid:
        blockers.append("invalid_tensor_quant_params")
    if failed:
        blockers.append("failed_samples_present")
    if processed < min(SANITY_SAMPLES, requested_samples):
        blockers.append("numeric_sanity_samples_incomplete")
    sanity_allowed = not blockers and processed >= min(SANITY_SAMPLES, requested_samples)
    full_requested = requested_samples >= full_ap_min_samples
    if not full_requested:
        blockers.append(f"full_eval_not_requested:{requested_samples}_lt_{full_ap_min_samples}")
    if processed < full_ap_min_samples:
        blockers.append(f"full_samples_below_{full_ap_min_samples}")
    if full_requested and not bool(stage2_report.get("ap_row_allowed")):
        blockers.append("stage2_ap_row_not_allowed")
    full_allowed = bool(
        tensor_quant_params_valid
        and failed == 0
        and full_requested
        and processed >= full_ap_min_samples
        and stage2_report.get("ap_row_allowed")
    )
    return {
        "sanity_16": sanity_allowed,
        "full_1789": full_allowed,
        "ap_row_allowed": full_allowed,
        "feasibility_blockers": list(dict.fromkeys(blockers)),
    }


def _stage2_args(
    args: argparse.Namespace, *, artifact_path: Path, inventory_path: Path,
    weights_path: Path, quant_params_path: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        label=f"pyramid_{args.precision_tag}", ckpt_dir=str(args.model_dir),
        raw_dir=str(args.output_dir / "heal_activation"), heal_root=str(args.heal_root),
        gpu_id=int(args.gpu_id), eval_range=str(args.eval_range), route_dir=str(args.compiled_artifact.parent),
        artifact_path=str(artifact_path), inventory_path=str(inventory_path),
        runtime_weight_archive_path=str(weights_path),
        worker_script=str(args.worker_script), persistent_worker=True,
        tvm_python=str(args.tvm_python), tvm_ld_library_path=str(args.tvm_ld_library_path),
        num_samples=int(args.num_samples), full_ap_min_samples=int(args.full_ap_min_samples),
        ap_row_min_samples=int(args.full_ap_min_samples),
        keep_detailed_samples=int(args.keep_detailed_samples),
        numeric_sanity_only=bool(args.numeric_sanity_only),
        tensor_quant_params_path=str(quant_params_path), execution_abi=EXECUTION_ABI,
    )


def _blocked_report(args: argparse.Namespace, blockers: list[str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "status": "blocked", "precision_tag": str(args.precision_tag),
        "processed_samples": 0, "requested_num_samples": int(args.num_samples),
        "ap_measured": False, "ap_row_allowed": False,
        "gates": {"sanity_16": False, "full_1789": False},
        "feasibility_blockers": list(dict.fromkeys(blockers)),
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    args.output_dir = output_dir
    args.report_json = Path(args.report_json).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_source = Path(args.compiled_artifact).resolve()
    route_report_path = artifact_source.parent / "route_b_int8_auto_decomp_result.json"
    route_report = _read_json(route_report_path)
    runtime_arg_plan = route_report.get("runtime_arg_plan")
    if not isinstance(runtime_arg_plan, list):
        raise ValueError("route_b_int8_auto_decomp_result.json missing runtime_arg_plan")
    weights_path = artifact_source.parent / "native_direct_reference" / "runtime_weights_int8.npz"
    inventory = build_audit_inventory(
        runtime_arg_plan=runtime_arg_plan, runtime_weight_archive_path=weights_path,
        route_report_path=route_report_path,
    )
    inventory_path = output_dir / "tvm_operator_inventory.json"
    _write_json(inventory_path, inventory)
    params, quant_blockers = build_tensor_quant_params(runtime_arg_plan)
    quant_params_path = output_dir / "tensor_quant_params.json"
    _write_json(quant_params_path, {"schema": f"{SCHEMA}_tensor_quant_params_v1", "params": params})
    prepared_artifact = prepare_compiled_artifact(artifact_source, output_dir)

    if args.num_samples >= args.full_ap_min_samples and quant_blockers:
        report = _blocked_report(args, ["invalid_tensor_quant_params", *quant_blockers])
        _write_json(output_dir / "feasibility_blocker.json", report)
        _write_json(Path(args.report_json), report)
        return report

    stage2_report = stage2_bridge.run_bridge(_stage2_args(
        args, artifact_path=prepared_artifact, inventory_path=inventory_path,
        weights_path=weights_path, quant_params_path=quant_params_path,
    ))
    gates = evaluate_result_gates(
        stage2_report, tensor_quant_params_valid=not quant_blockers,
        requested_samples=int(args.num_samples), full_ap_min_samples=int(args.full_ap_min_samples),
    )
    report = {
        "schema": SCHEMA, "status": "success" if gates["sanity_16"] else "blocked",
        "precision_tag": str(args.precision_tag), "compiled_artifact": str(prepared_artifact),
        "source_compiled_artifact": str(artifact_source), "route_report": str(route_report_path),
        "inventory_path": str(inventory_path), "runtime_weight_archive_path": str(weights_path),
        "tensor_quant_params_path": str(quant_params_path), "execution_abi": EXECUTION_ABI,
        "requested_num_samples": int(args.num_samples),
        "processed_samples": int(stage2_report.get("processed_samples") or 0),
        "ap_measured": bool(gates["full_1789"]), "ap_row_allowed": bool(gates["ap_row_allowed"]),
        "gates": {"sanity_16": gates["sanity_16"], "full_1789": gates["full_1789"]},
        "feasibility_blockers": gates["feasibility_blockers"], "stage2_report": stage2_report,
    }
    for key in ("ap30", "ap50", "ap70"):
        if key in stage2_report:
            report[key] = stage2_report[key]
    if not report["ap_row_allowed"]:
        _write_json(output_dir / "feasibility_blocker.json", report)
    _write_json(Path(args.report_json), report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-artifact", type=Path, required=True)
    parser.add_argument("--precision-tag", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--full-ap-min-samples", type=int, default=FULL_SAMPLES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--heal-root", type=Path, default=Path(stage2_bridge.DEFAULT_HEAL_ROOT))
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--worker-script", type=Path, default=ROOT / "scripts/stage2_native_int8_tvm_worker.py")
    parser.add_argument("--tvm-python", type=Path, default=Path(stage2_bridge.DEFAULT_TVM_PYTHON))
    parser.add_argument("--tvm-ld-library-path", default=stage2_bridge.DEFAULT_TVM_LD_LIBRARY_PATH)
    parser.add_argument("--keep-detailed-samples", type=int, default=1)
    args = parser.parse_args(argv)
    if args.num_samples <= 0 or args.full_ap_min_samples <= 0:
        parser.error("sample counts must be positive")
    args.numeric_sanity_only = args.num_samples < args.full_ap_min_samples
    if args.report_json is None:
        args.report_json = args.output_dir / "full_ap_eval_report.json"
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_gate(args)
    except Exception as exc:
        report = _blocked_report(args, [f"feasibility_exception:{type(exc).__name__}:{exc}"])
        _write_json(Path(args.output_dir) / "feasibility_blocker.json", report)
        _write_json(Path(args.report_json), report)
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
