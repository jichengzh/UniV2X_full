#!/usr/bin/env python3
"""Gate CoDriving TVM INT8 AP on a 16-sample numerical comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_native_int8_real_activation_bridge import (  # noqa: E402
    DEFAULT_TVM_LD_LIBRARY_PATH,
    DEFAULT_TVM_PYTHON,
    NativeInt8BackboneBridge,
)
from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, write_json  # noqa: E402
from scripts.stage3_codriving_tvm_fp16_ap_bridge_v3 import sha256_path  # noqa: E402


SCHEMA = "stage3_codriving_tvm_int8_ap_numeric_gate_v3"
DEFAULT_REPO_ROOT = Path("/exdata/jichengzhi/V2Xverse_pyramid")
SANITY_SAMPLES = 16
FULL_SAMPLES = 1789
BOUNDARY_INPUT = "spatial_features"
DYNAMIC_QUANT_KEYS = ("dynamic", "dynamic_per_chunk", "per_chunk", "allow_fallback")


@dataclass(frozen=True)
class RuntimeBundle:
    artifact_path: Path
    inventory_path: Path
    runtime_weights_path: Path
    native_direct_reference_path: Path
    tensor_quant_params_path: Path
    source_tensor_quant_params_path: Path
    source_result_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file_content(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_path(value: Any, *, base: Path, label: str) -> Path:
    if not value:
        raise FileNotFoundError(f"{label} path missing")
    path = Path(str(value))
    candidates = [path, base / path, base / path.name, base / label / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} not found: {path}")


def _result_path(route_dir: Path) -> Path:
    candidates = [
        route_dir / "route_b_int8_auto_decomp_result.json",
        route_dir / "result.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Route-B result JSON missing beside artifact: {route_dir}")


def _output_names(runtime_arg_plan: list[dict[str, Any]]) -> list[str]:
    return [str(item["arg_name"]) for item in runtime_arg_plan if item.get("role") == "graph_output"]


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_payload(*, precision_tag: str, full_ap_min_samples: int) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_protocol_v1",
        "precision_tag": str(precision_tag),
        "artifact_input_dtype": "uint8",
        "graph_input": BOUNDARY_INPUT,
        "graph_outputs": 3,
        "quant_params_policy": "static_scale_and_zero_point_required",
        "fallback_policy": "forbidden",
        "patch_target": "model.backbone.resnet",
        "sanity_samples": SANITY_SAMPLES,
        "full_samples": int(full_ap_min_samples),
    }


def validate_boundary_quant_params(
    runtime_arg_plan: list[dict[str, Any]], params_payload: Mapping[str, Any]
) -> list[str]:
    params = params_payload.get("params", params_payload)
    failures: list[str] = []
    input_names = [
        str(item.get("arg_name") or "")
        for item in runtime_arg_plan if item.get("role") == "graph_input"
    ]
    output_names = _output_names(runtime_arg_plan)
    if input_names != [BOUNDARY_INPUT]:
        failures.append(f"expected_spatial_features_graph_input:found_{','.join(input_names) or 'none'}")
    if len(output_names) != 3:
        failures.append(f"expected_3_graph_outputs:found_{len(output_names)}")
    for name in [BOUNDARY_INPUT, *output_names]:
        item = params.get(name) if isinstance(params, Mapping) else None
        if isinstance(item, Mapping):
            source = str(item.get("source") or "").lower().replace("-", "_")
            dynamic_source = any(token in source for token in ("dynamic", "per_chunk", "fallback"))
            if dynamic_source or any(bool(item.get(key)) for key in DYNAMIC_QUANT_KEYS):
                failures.append(f"dynamic_quant_fallback_forbidden:{name}")
        try:
            scale = float(item["scale"])
            zero_point = int(item["zero_point"])
        except (KeyError, TypeError, ValueError, OverflowError):
            failures.append(f"missing_static_quant_params:{name}")
            if name in output_names:
                failures.append(f"missing_output_dequant_params:{name}")
            continue
        if not np.isfinite(scale) or scale <= 0.0 or not 0 <= zero_point <= 255:
            failures.append(f"invalid_static_quant_params:{name}")
    return list(dict.fromkeys(failures))


def validate_output_quant_params(
    runtime_arg_plan: list[dict[str, Any]], params_payload: Mapping[str, Any]
) -> list[str]:
    return validate_boundary_quant_params(runtime_arg_plan, params_payload)


def build_run_bindings(
    *, source_compiled_artifact: Path, model_dir: Path, checkpoint_path: Path,
    tensor_quant_params_path: Path, protocol: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = Path(source_compiled_artifact).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    params = Path(tensor_quant_params_path).resolve()
    protocol_value = dict(protocol)
    return {
        "source_compiled_artifact": str(artifact),
        "source_compiled_artifact_sha256": sha256_path(artifact),
        "model_dir": str(Path(model_dir).resolve()),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "tensor_quant_params_path": str(params),
        "tensor_quant_params_sha256": sha256_path(params),
        "protocol": protocol_value,
        "protocol_sha256": sha256_json(protocol_value),
    }


def inspect_route_artifact(compiled_artifact: Path) -> tuple[Path, dict[str, Any], Path, Path, Path]:
    source = Path(compiled_artifact).resolve()
    if source.name != "route_b_int8_auto_decomp.vmexec" or not source.is_file():
        raise FileNotFoundError(f"expected existing route_b_int8_auto_decomp.vmexec: {source}")
    result_path = _result_path(source.parent)
    result = _load_json(result_path)
    if result.get("status") != "success":
        raise ValueError("Route-B result status is not success")
    runtime_plan = result.get("runtime_arg_plan")
    if not isinstance(runtime_plan, list) or not runtime_plan:
        raise ValueError("Route-B result missing runtime_arg_plan")
    weight_value = result.get("runtime_weight_archive_path") or result.get("runtime_weights")
    if isinstance(weight_value, Mapping):
        weight_value = weight_value.get("path") or weight_value.get("archive_path")
    weights = _existing_path(
        weight_value or source.parent / "native_direct_reference" / "runtime_weights_int8.npz",
        base=source.parent,
        label="runtime_weights",
    )
    direct_value = result.get("native_direct_reference") or {}
    if isinstance(direct_value, Mapping):
        direct_value = direct_value.get("artifact_path") or direct_value.get("path")
    direct = _existing_path(direct_value, base=source.parent, label="native_direct_reference")
    params_value = result.get("tensor_quant_params_path") or source.parent / "tensor_quant_params.json"
    params = _existing_path(params_value, base=source.parent, label="tensor_quant_params")
    return result_path, result, weights, direct, params


def materialize_runtime_bundle(compiled_artifact: Path, output_dir: Path) -> RuntimeBundle:
    source = Path(compiled_artifact).resolve()
    result_path, result, weights, direct, params = inspect_route_artifact(source)
    destination = Path(output_dir).resolve() / "runtime_bundle"
    destination.mkdir(parents=True, exist_ok=True)
    artifact_copy = destination / "route_b_int8_auto_decomp.so"
    weights_copy = destination / "runtime_weights_int8.npz"
    direct_copy = destination / "native_direct_reference.so"
    params_copy = destination / "tensor_quant_params.json"
    for src, dst in ((source, artifact_copy), (weights, weights_copy), (direct, direct_copy), (params, params_copy)):
        shutil.copy2(src, dst)
    inventory = {
        "schema": f"{SCHEMA}_inventory_v1",
        "execution_abi": "relax_vm_return",
        "runtime_arg_plan": result["runtime_arg_plan"],
        "runtime_weight_archive_keys": result.get("runtime_weight_archive_keys", {}),
        "source_result_path": str(result_path),
        "native_direct_reference": str(direct_copy),
    }
    inventory_path = destination / "tvm_operator_inventory.json"
    write_json(inventory_path, inventory)
    return RuntimeBundle(
        artifact_copy, inventory_path, weights_copy, direct_copy, params_copy, params, result_path,
    )


def preflight_numerical_feasibility(compiled_artifact: Path) -> dict[str, Any]:
    try:
        result_path, result, weights, direct, params_path = inspect_route_artifact(compiled_artifact)
        reasons = validate_boundary_quant_params(result["runtime_arg_plan"], _load_json(params_path))
        evidence = {
            "source_result": str(result_path), "runtime_weights": str(weights),
            "native_direct_reference": str(direct), "tensor_quant_params": str(params_path),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        reasons = [f"artifact_preflight_failed:{type(exc).__name__}:{exc}"]
        evidence = {}
    return {
        "schema": SCHEMA, "status": "ready_for_numerical_sanity" if not reasons else "numerical_feasibility_failure",
        "failure_reasons": reasons, "ap_measured": False, "full_network_claim": False,
        "evidence": evidence,
    }


def build_numerical_gate_report(
    records: list[dict[str, Any]], *, processed_samples: int, engine_calls: int
) -> dict[str, Any]:
    reasons: list[str] = []
    accounting = engine_accounting(
        processed_samples=processed_samples,
        engine_calls=engine_calls,
    )
    if not accounting["engine_accounting_valid"]:
        reasons.append(str(accounting["engine_accounting_failure_reason"]))
    by_output: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_output.setdefault(str(record.get("tensor_name")), []).append(record)
    if processed_samples < SANITY_SAMPLES:
        reasons.append(f"numeric_sanity_samples_{processed_samples}_lt_{SANITY_SAMPLES}")
    if len(by_output) != 3:
        reasons.append(f"numeric_sanity_expected_3_outputs:found_{len(by_output)}")
    outputs: dict[str, Any] = {}
    for name, items in sorted(by_output.items()):
        correlations = [item.get("alignment_error", {}).get("corrcoef") for item in items]
        valid = [float(value) for value in correlations if value is not None and np.isfinite(float(value))]
        mean_corr = float(np.mean(valid)) if valid else None
        observed_call_ids = {
            int(item["sample_index"])
            for item in items
            if isinstance(item.get("sample_index"), int)
        }
        expected_call_ids = set(range(int(engine_calls)))
        missing_call_ids = sorted(expected_call_ids - observed_call_ids)
        unexpected_call_ids = sorted(observed_call_ids - expected_call_ids)
        coverage_ok = not missing_call_ids and not unexpected_call_ids
        passed = coverage_ok and mean_corr is not None and mean_corr >= 0.5
        if not coverage_ok:
            reasons.append(
                f"output_record_coverage_failed:{name}:records={len(items)}:engine_calls={engine_calls}:"
                f"missing_call_ids={','.join(map(str, missing_call_ids)) or 'none'}:"
                f"unexpected_call_ids={','.join(map(str, unexpected_call_ids)) or 'none'}"
            )
        if not passed:
            reasons.append(f"output_correlation_failed:{name}")
        outputs[name] = {"records": len(items), "corrcoef_mean": mean_corr, "passed": passed}
    passed = not reasons
    return {
        "schema": SCHEMA,
        "status": "numerical_sanity_passed" if passed else "numerical_feasibility_failure",
        "processed_samples": int(processed_samples), "numeric_outputs": outputs,
        "failure_reasons": reasons, "ap_measured": False, "full_network_claim": False,
        "gates": {"sanity_16": passed, "full_1789": False},
        **accounting,
    }


def build_full_gate_decision(
    *, processed_samples: int, minimum_samples: int, accounting: dict[str, Any]
) -> dict[str, Any]:
    reasons: list[str] = []
    if int(processed_samples) < int(minimum_samples):
        reasons.append(f"full_ap_samples_{processed_samples}_lt_{minimum_samples}")
    if not accounting["engine_accounting_valid"]:
        reasons.append(str(accounting["engine_accounting_failure_reason"]))
    full_gate = not reasons
    return {
        "full_gate": full_gate,
        "ap_measured": full_gate,
        "failure_reasons": reasons,
    }


def engine_accounting(
    *,
    processed_samples: int,
    engine_calls: int,
    expected_calls_per_sample: float | None = None,
) -> dict[str, Any]:
    samples = int(processed_samples)
    calls = int(engine_calls)
    failure_reason: str | None = None
    if samples <= 0 or calls <= 0:
        calls_per_sample = 0.0 if samples == 0 else calls / samples
        failure_reason = "engine_accounting_non_positive_count"
    elif calls % samples != 0:
        calls_per_sample = calls / samples
        failure_reason = "engine_calls_not_divisible_by_processed_samples"
    else:
        calls_per_sample = calls / samples
        if (
            expected_calls_per_sample is not None
            and not math.isclose(calls_per_sample, float(expected_calls_per_sample), rel_tol=0.0, abs_tol=0.0)
        ):
            failure_reason = (
                "engine_call_pattern_mismatch:"
                f"expected={float(expected_calls_per_sample)}:actual={calls_per_sample}"
            )
    return {
        "engine_samples": samples,
        "engine_calls": calls,
        "engine_calls_per_sample": calls_per_sample,
        "engine_accounting_valid": failure_reason is None,
        "engine_accounting_failure_reason": failure_reason,
    }


def require_full_run_sanity(
    num_samples: int, sanity_report: Path | None, expected_sha256: str | None,
    expected_bindings: Mapping[str, Any],
) -> float | None:
    if int(num_samples) < FULL_SAMPLES:
        return None
    if sanity_report is None or not Path(sanity_report).is_file():
        raise ValueError("passing 16-sample numerical sanity report is required before full AP")
    report_path = Path(sanity_report).resolve()
    if not expected_sha256:
        raise ValueError("sanity report SHA256 is required before full AP")
    actual_sha256 = sha256_file_content(report_path)
    if actual_sha256.lower() != str(expected_sha256).lower():
        raise ValueError("sanity report SHA256 mismatch")
    report = _load_json(report_path)
    if report.get("gates", {}).get("sanity_16") is not True or int(report.get("processed_samples") or 0) < SANITY_SAMPLES:
        raise ValueError("passing 16-sample numerical sanity report is required before full AP")
    calls_per_sample = report.get("engine_calls_per_sample")
    if (
        report.get("engine_accounting_valid") is not True
        or not isinstance(calls_per_sample, (int, float))
        or isinstance(calls_per_sample, bool)
        or not math.isfinite(float(calls_per_sample))
        or float(calls_per_sample) <= 0.0
    ):
        raise ValueError("sanity engine call accounting is missing or invalid")
    for key, expected in expected_bindings.items():
        if report.get(key) != expected:
            raise ValueError(f"sanity binding mismatch:{key}")
    return float(calls_per_sample)


def _sanity_report_sha256(args: argparse.Namespace) -> str | None:
    if args.sanity_report_sha256:
        return str(args.sanity_report_sha256)
    if args.plan_json is None:
        return None
    plan = _load_json(Path(args.plan_json))
    value = plan.get("sanity_report_sha256")
    if value is None and isinstance(plan.get("sanity"), Mapping):
        value = plan["sanity"].get("report_sha256")
    return str(value) if value else None


class CoDrivingTvmInt8ResnetModule(torch.nn.Module):
    def __init__(self, *, reference_resnet: Any, shared_bridge: Any) -> None:
        super().__init__()
        self.reference_resnet = reference_resnet
        self.shared_bridge = shared_bridge
        shared_bridge.reference_get_multiscale = reference_resnet

    def forward(self, spatial_features: Any) -> tuple[Any, ...]:
        return self.shared_bridge(spatial_features)

    def close(self) -> None:
        self.shared_bridge.close()


def patch_model_resnet(model: Any, shared_bridge: Any) -> CoDrivingTvmInt8ResnetModule:
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "resnet"):
        raise AttributeError("CoDriving model must expose model.backbone.resnet")
    module = CoDrivingTvmInt8ResnetModule(reference_resnet=model.backbone.resnet, shared_bridge=shared_bridge)
    model.backbone.resnet = module
    return module


def logical_cuda_index(gpu_id: int, visible_devices: str | None = None) -> int:
    requested = int(gpu_id)
    raw = os.environ.get("CUDA_VISIBLE_DEVICES") if visible_devices is None else visible_devices
    if not raw:
        return requested
    devices = [item.strip() for item in str(raw).split(",") if item.strip()]
    if str(requested) in devices:
        return devices.index(str(requested))
    if 0 <= requested < len(devices):
        return requested
    raise ValueError(f"gpu {requested} is not visible in CUDA_VISIBLE_DEVICES={raw}")


def place_model_on_cuda(model: Any, gpu_id: int) -> Any:
    device = torch.device("cuda", logical_cuda_index(gpu_id))
    torch.cuda.set_device(device)
    return model.to(device).eval()


def _bridge_args(args: argparse.Namespace, bundle: RuntimeBundle) -> argparse.Namespace:
    return argparse.Namespace(
        label=f"stage3_codriving_{args.precision_tag}", artifact_path=str(bundle.artifact_path),
        inventory_path=str(bundle.inventory_path), runtime_weight_archive_path=str(bundle.runtime_weights_path),
        tensor_quant_params_path=str(bundle.tensor_quant_params_path), persistent_worker=True,
        numeric_sanity_only=int(args.num_samples) == SANITY_SAMPLES, worker_script=str(args.worker_script),
        tvm_python=str(args.tvm_python), tvm_ld_library_path=str(args.tvm_ld_library_path),
        gpu_id=int(args.gpu_id), keep_detailed_samples=int(args.keep_detailed_samples),
    )


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    args.compiled_artifact = Path(args.compiled_artifact).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.report_json = Path(args.report_json).resolve()
    args.model_dir = Path(args.model_dir).resolve()
    if args.sanity_report_json is not None:
        args.sanity_report_json = Path(args.sanity_report_json).resolve()
    preflight = preflight_numerical_feasibility(args.compiled_artifact)
    if preflight["status"] != "ready_for_numerical_sanity":
        write_json(args.report_json, preflight)
        return preflight
    bundle = materialize_runtime_bundle(args.compiled_artifact, args.output_dir)
    from torch.utils.data import DataLoader
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    model_dir = Path(args.model_dir).resolve()
    hypes = load_yaml(str(model_dir / "config.yaml"))
    hypes["validate_dir"] = hypes["test_dir"]
    model = train_utils.create_model(hypes)
    checkpoint = (
        Path(args.checkpoint_path).resolve()
        if args.checkpoint_path is not None
        else best_checkpoint(model_dir)
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    protocol = protocol_payload(
        precision_tag=args.precision_tag, full_ap_min_samples=args.full_ap_min_samples,
    )
    bindings = build_run_bindings(
        source_compiled_artifact=Path(args.compiled_artifact), model_dir=model_dir,
        checkpoint_path=checkpoint,
        tensor_quant_params_path=bundle.source_tensor_quant_params_path,
        protocol=protocol,
    )
    expected_calls_per_sample = require_full_run_sanity(
        args.num_samples, args.sanity_report_json, _sanity_report_sha256(args), bindings,
    )
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model = place_model_on_cuda(model, args.gpu_id)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                        collate_fn=dataset.collate_batch_test, shuffle=False, drop_last=False)
    raw_dir = Path(args.output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shared = NativeInt8BackboneBridge(_bridge_args(args, bundle), raw_dir, _load_json(bundle.inventory_path))
    module = patch_model_resnet(model, shared)
    result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}
    processed = 0
    started = time.time()
    try:
        with torch.inference_mode():
            for batch_data in loader:
                if processed >= args.num_samples:
                    break
                if batch_data is None:
                    continue
                batch_data = train_utils.to_device(batch_data, "cuda")
                output = model(batch_data["ego"])
                pred_box, pred_score, gt_box = dataset.post_process(batch_data, {"ego": output})
                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
                processed += 1
    finally:
        module.close()
    if args.num_samples == SANITY_SAMPLES:
        report = build_numerical_gate_report(
            shared.numeric_sanity_records,
            processed_samples=processed,
            engine_calls=shared.call_index,
        )
        report.update({
            "fallback_samples": 0,
            "failed_samples": 0,
        })
    else:
        eval_dir = Path(args.output_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(eval_dir), args.precision_tag)
        report = {
            "schema": SCHEMA, "processed_samples": processed,
            "fallback_samples": 0, "failed_samples": 0,
        }
        accounting = engine_accounting(
            processed_samples=processed,
            engine_calls=shared.call_index,
            expected_calls_per_sample=expected_calls_per_sample,
        )
        decision = build_full_gate_decision(
            processed_samples=processed,
            minimum_samples=args.full_ap_min_samples,
            accounting=accounting,
        )
        full_gate = decision["full_gate"]
        if full_gate:
            report.update({
                "ap": {"ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70)},
                "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
            })
        report.update(accounting)
        report.update({
            "status": "success" if full_gate else "numerical_feasibility_failure",
            "failure_reasons": decision["failure_reasons"],
            "ap_measured": decision["ap_measured"],
            "gates": {"sanity_16": True, "full_1789": full_gate},
        })
    report.update(bindings)
    report.update({
        "repo_root": str(repo_root), "compiled_artifact": str(bundle.artifact_path),
        "artifact_sha256": sha256_path(bundle.artifact_path),
        "elapsed_secs": time.time() - started,
    })
    if int(args.num_samples) >= FULL_SAMPLES:
        report["sanity_report_path"] = str(Path(args.sanity_report_json).resolve())
        report["sanity_report_sha256"] = sha256_file_content(Path(args.sanity_report_json))
    write_json(args.report_json, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-artifact", type=Path, required=True)
    parser.add_argument("--precision-tag", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--full-ap-min-samples", type=int, default=FULL_SAMPLES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--sanity-report-json", type=Path)
    parser.add_argument("--sanity-report-sha256")
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--worker-script", type=Path, default=ROOT / "scripts/stage2_native_int8_tvm_worker.py")
    parser.add_argument("--tvm-python", type=Path, default=DEFAULT_TVM_PYTHON)
    parser.add_argument("--tvm-ld-library-path", default=DEFAULT_TVM_LD_LIBRARY_PATH)
    parser.add_argument("--keep-detailed-samples", type=int, default=16)
    args = parser.parse_args(argv)
    args.report_json = args.report_json or args.output_dir / "full_ap_eval_report.json"
    args.persistent_worker = True
    args.numeric_sanity_only = args.num_samples == SANITY_SAMPLES
    if args.precision_tag != "int8":
        parser.error("--precision-tag must be int8")
    if args.num_samples not in (SANITY_SAMPLES, FULL_SAMPLES):
        parser.error("--num-samples must be 16 or 1789")
    return args


def main() -> int:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report = run_bridge(args)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status") != "numerical_feasibility_failure" else 2


if __name__ == "__main__":
    raise SystemExit(main())
