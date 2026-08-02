#!/usr/bin/env python3
"""One-sample HEAL real-activation bridge for native INT8 TVM backbone route."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
import shutil
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.native_int8_full_onnx import (  # noqa: E402
    build_tvm_worker_request,
    quantize_activation_uint8,
    quantize_activation_uint8_static,
)
from scripts.stage2_h800_true_fp16_ap_eval import (  # noqa: E402
    best_checkpoint,
    cast_floating_tensors,
    checkpoint_epoch,
    dtype_counts,
    write_json,
)


DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_TVM_PYTHON = "/exdata/jichengzhi/tvm310/bin/python"
DEFAULT_TVM_LD_LIBRARY_PATH = (
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
    "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib"
)
DEFAULT_ROUTE_DIR = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "raw/int8_native_route/20260628_native_int8_apshape_s0_024_realweight_probe_v2/s0_024"
)


def load_exact_checkpoint(
    model: Any,
    checkpoint_path: Path,
    *,
    torch_module: Any,
    train_utils_module: Any | None,
) -> int:
    state_dict = torch_module.load(checkpoint_path, map_location="cpu")
    if train_utils_module is not None:
        train_utils_module.check_missing_key(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=False)
    return checkpoint_epoch(checkpoint_path)


def summarize_numpy_array(array: np.ndarray, *, tensor_name: str, path: Path | None = None) -> dict[str, Any]:
    arr = np.asarray(array)
    base = {
        "tensor_name": str(tensor_name),
        "path": str(path) if path is not None else "",
        "shape": [int(dim) for dim in arr.shape],
        "dtype": str(arr.dtype),
        "size": int(arr.size),
    }
    if arr.size == 0:
        return {
            **base,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        **base,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def attach_tensor_quant_params_to_worker_request(
    request: dict[str, Any],
    *,
    tensor_quant_params_path: str,
    tensor_quant_params_digest: str,
) -> dict[str, Any]:
    request["tensor_quant_params_path"] = str(tensor_quant_params_path)
    request["tensor_quant_params_digest"] = str(tensor_quant_params_digest)
    request["output_dequant_policy"] = "per_output_tensor_quant_params_v2"
    return request


def dequantize_tvm_output_uint8(
    values: np.ndarray,
    *,
    tensor_name: str,
    activation_quant: dict[str, Any],
    tensor_quant_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = (tensor_quant_params or {}).get(str(tensor_name)) or {}
    if params:
        scale = float(params.get("scale") or 0.0)
        if scale <= 0.0:
            raise ValueError(f"invalid tensor quant scale for {tensor_name}: {params}")
        zero_point = int(params.get("zero_point", 0))
        scheme = "tensor_quant_params_v2"
        source = str(params.get("source") or "tensor_quant_params")
    else:
        scale = float(activation_quant["scale"])
        zero_point = int(activation_quant["zero_point"])
        scheme = "activation_quant_fallback"
        source = "per_agent_activation_quant"
    arr = np.asarray(values, dtype=np.uint8)
    dequant = ((arr.astype(np.float32) - float(zero_point)) * float(scale)).astype(np.float32)
    return dequant, {
        "schema": "native_int8_output_dequant_details_v1",
        "tensor_name": str(tensor_name),
        "scheme": scheme,
        "source": source,
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


def native_int8_full_ap_gate(report: dict[str, Any], *, min_samples: int = 1789) -> dict[str, str]:
    processed = int(report.get("processed_samples") or 0)
    if processed < int(min_samples):
        return {
            "status": "blocked",
            "reason": f"partial_eval_num_samples_{processed}_lt_{int(min_samples)}",
        }
    pred_nonempty_count = int(report.get("pred_nonempty_count") or 0)
    if pred_nonempty_count <= 0:
        return {"status": "blocked", "reason": "empty_predictions_all_samples"}
    for key in ("ap30", "ap50", "ap70"):
        try:
            value = float(report[key])
        except Exception:
            return {"status": "blocked", "reason": f"missing_or_invalid_{key}"}
        if not np.isfinite(value):
            return {"status": "blocked", "reason": f"non_finite_{key}"}
    return {"status": "measured_row_allowed", "reason": "full_eval_ap_metrics_present"}


def build_full_ap_report_gate_fields(
    report: dict[str, Any],
    *,
    min_samples: int,
    row_min_samples: int = 1789,
) -> dict[str, Any]:
    gate = native_int8_full_ap_gate(report, min_samples=int(min_samples))
    processed = int(report.get("processed_samples") or 0)
    smoke_gate_passed = gate["status"] == "measured_row_allowed"
    row_sample_gate = processed >= int(row_min_samples)
    ap_row_allowed = bool(smoke_gate_passed and row_sample_gate)
    if ap_row_allowed:
        block_reason = ""
    elif not smoke_gate_passed:
        block_reason = str(gate["reason"])
    else:
        block_reason = f"full_eval_num_samples_{processed}_lt_{int(row_min_samples)}"
    return {
        "gate": gate,
        "smoke_gate_passed": smoke_gate_passed,
        "ap_row_allowed": ap_row_allowed,
        "ap_row_min_samples": int(row_min_samples),
        "ap_row_block_reason": block_reason,
    }


def build_full_ap_blocker(
    *,
    label: str,
    raw_artifact: str,
    report: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": "native_int8_full_ap_gate_blocker_v1",
        "status": "blocked",
        "label": str(label),
        "failure_reason": str(reason),
        "sample_count": int(report.get("processed_samples") or 0),
        "pred_nonempty_count": int(report.get("pred_nonempty_count") or 0),
        "ap30": report.get("ap30"),
        "ap50": report.get("ap50"),
        "ap70": report.get("ap70"),
        "raw_artifact": str(raw_artifact),
        "full_network_claim": False,
        "ap_measured": False,
    }


def _finite_ratio(array: np.ndarray) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0
    return float(np.isfinite(arr).sum() / arr.size)


def _nonzero_ratio(array: np.ndarray) -> float:
    arr = np.asarray(array)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr) / arr.size)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float | None:
    lhs = np.asarray(a, dtype=np.float64).reshape(-1)
    rhs = np.asarray(b, dtype=np.float64).reshape(-1)
    if lhs.size == 0 or rhs.size == 0 or lhs.size != rhs.size:
        return None
    if float(np.std(lhs)) == 0.0 or float(np.std(rhs)) == 0.0:
        return None
    return float(np.corrcoef(lhs, rhs)[0, 1])


def build_output_dequant_sanity_record(
    *,
    tensor_name: str,
    reference: np.ndarray,
    tvm_uint8: np.ndarray,
    sample_index: int,
    candidate_dequant: np.ndarray | None = None,
    dequant_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32)
    u8 = np.asarray(tvm_uint8, dtype=np.uint8)
    if ref.shape != u8.shape:
        raise ValueError(f"reference/TVM shape mismatch for {tensor_name}: {ref.shape} != {u8.shape}")
    ref_min = float(np.min(ref))
    ref_max = float(np.max(ref))
    u8_min = float(np.min(u8))
    u8_max = float(np.max(u8))
    details = dict(dequant_details or {})
    if candidate_dequant is not None:
        dequant = np.asarray(candidate_dequant, dtype=np.float32)
        if dequant.shape != ref.shape:
            raise ValueError(f"reference/dequant shape mismatch for {tensor_name}: {ref.shape} != {dequant.shape}")
        scale = float(details.get("scale", 1.0))
        zero_point = float(details.get("zero_point", u8_min))
        scheme = str(details.get("scheme") or "provided_candidate_dequant")
        source = str(details.get("source") or "provided_candidate_dequant")
    elif u8_max == u8_min:
        scale = 1.0
        zero_point = u8_min
        dequant = np.full(ref.shape, ref_min, dtype=np.float32)
        scheme = "minmax_align_uint8_to_reference"
        source = "reference_range_fit"
    else:
        scale = float((ref_max - ref_min) / (u8_max - u8_min)) if ref_max != ref_min else 1.0
        zero_point = float(u8_min - (ref_min / scale)) if scale != 0 else u8_min
        dequant = ((u8.astype(np.float32) - u8_min) * scale + ref_min).astype(np.float32)
        scheme = "minmax_align_uint8_to_reference"
        source = "reference_range_fit"
    diff = dequant.astype(np.float64) - ref.astype(np.float64)
    return {
        "schema": "native_int8_output_dequant_sanity_record_v1",
        "tensor_name": str(tensor_name),
        "sample_index": int(sample_index),
        "reference": summarize_numpy_array(ref, tensor_name="reference"),
        "tvm_uint8": summarize_numpy_array(u8, tensor_name="tvm_uint8"),
        "candidate_dequant": {
            "scheme": scheme,
            "source": source,
            "scale": scale,
            "zero_point": zero_point,
            "dequant_min": float(np.min(dequant)),
            "dequant_max": float(np.max(dequant)),
            "dynamic_range": float(np.max(dequant) - np.min(dequant)),
            "finite_ratio": _finite_ratio(dequant),
            "nonzero_ratio": _nonzero_ratio(dequant),
            "uint8_min": u8_min,
            "uint8_max": u8_max,
            "uint8_dynamic_range": float(u8_max - u8_min),
            "uint8_nonzero_ratio": _nonzero_ratio(u8),
        },
        "alignment_error": {
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "max_abs": float(np.max(np.abs(diff))),
            "corrcoef": _corrcoef(ref, dequant),
        },
    }


def build_op_level_alignment_record(
    *,
    tensor_name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_index: int,
    candidate_kind: str,
    candidate_details: dict[str, Any] | None = None,
    pass_rel_rmse: float = 0.25,
    pass_corrcoef: float = 0.5,
) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"reference/candidate shape mismatch for {tensor_name}: {ref.shape} != {cand.shape}")
    diff = cand.astype(np.float64) - ref.astype(np.float64)
    ref_range = float(np.max(ref) - np.min(ref)) if ref.size else 0.0
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else float("inf")
    corrcoef = _corrcoef(ref, cand)
    passed = bool(ref_range > 0.0 and rmse <= max(1e-4, float(pass_rel_rmse) * ref_range))
    if corrcoef is not None:
        passed = passed and corrcoef >= float(pass_corrcoef)
    return {
        "schema": "native_int8_op_level_alignment_record_v1",
        "tensor_name": str(tensor_name),
        "sample_index": int(sample_index),
        "candidate_kind": str(candidate_kind),
        "candidate_details": dict(candidate_details or {}),
        "reference": summarize_numpy_array(ref, tensor_name="reference"),
        "candidate": summarize_numpy_array(cand, tensor_name="candidate"),
        "alignment_error": {
            "mae": float(np.mean(np.abs(diff))) if diff.size else float("inf"),
            "rmse": rmse,
            "max_abs": float(np.max(np.abs(diff))) if diff.size else float("inf"),
            "corrcoef": corrcoef,
            "reference_range": ref_range,
            "rmse_over_reference_range": float(rmse / ref_range) if ref_range > 0.0 else None,
        },
        "pass_criteria": {
            "pass_rel_rmse": float(pass_rel_rmse),
            "pass_corrcoef": float(pass_corrcoef),
        },
        "passed": passed,
    }


def summarize_dequant_sanity(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_output: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_output.setdefault(str(record["tensor_name"]), []).append(record)
    outputs: dict[str, Any] = {}
    pass_flags: list[bool] = []
    for name, items in sorted(by_output.items()):
        maes = [float(item["alignment_error"]["mae"]) for item in items]
        rmses = [float(item["alignment_error"]["rmse"]) for item in items]
        corrs = [
            float(item["alignment_error"]["corrcoef"])
            for item in items
            if item["alignment_error"]["corrcoef"] is not None
        ]
        ref_ranges = [
            float(item["reference"]["max"]) - float(item["reference"]["min"])
            for item in items
        ]
        mean_ref_range = float(np.mean(ref_ranges)) if ref_ranges else 0.0
        mean_rmse = float(np.mean(rmses)) if rmses else float("inf")
        mean_corr = float(np.mean(corrs)) if corrs else None
        schemes = sorted({str(item["candidate_dequant"].get("scheme") or "") for item in items})
        sources = sorted({str(item["candidate_dequant"].get("source") or "") for item in items})
        passed = bool(mean_ref_range > 0 and mean_rmse <= max(1e-4, 0.25 * mean_ref_range))
        if mean_corr is not None:
            passed = passed and mean_corr >= 0.5
        pass_flags.append(passed)
        outputs[name] = {
            "samples": len(items),
            "mae_mean": float(np.mean(maes)) if maes else None,
            "rmse_mean": mean_rmse,
            "max_abs_max": float(max(float(item["alignment_error"]["max_abs"]) for item in items)) if items else None,
            "corrcoef_mean": mean_corr,
            "reference_range_mean": mean_ref_range,
            "candidate_scale_mean": float(
                np.mean([float(item["candidate_dequant"]["scale"]) for item in items])
            )
            if items
            else None,
            "dequant_scheme": schemes[0] if len(schemes) == 1 else "mixed",
            "dequant_source": sources[0] if len(sources) == 1 else "mixed",
            "dequant_dynamic_range_mean": float(
                np.mean([float(item["candidate_dequant"].get("dynamic_range") or 0.0) for item in items])
            )
            if items
            else None,
            "dequant_nonzero_ratio_mean": float(
                np.mean([float(item["candidate_dequant"].get("nonzero_ratio") or 0.0) for item in items])
            )
            if items
            else None,
            "tvm_uint8_dynamic_range_mean": float(
                np.mean([float(item["candidate_dequant"].get("uint8_dynamic_range") or 0.0) for item in items])
            )
            if items
            else None,
            "tvm_uint8_nonzero_ratio_mean": float(
                np.mean([float(item["candidate_dequant"].get("uint8_nonzero_ratio") or 0.0) for item in items])
            )
            if items
            else None,
            "passed": passed,
        }
    return {
        "schema": "native_int8_output_dequant_calibration_summary_v1",
        "status": "passed" if pass_flags and all(pass_flags) else "blocked",
        "output_count": len(outputs),
        "outputs": outputs,
        "full_network_claim": False,
        "ap_measured": False,
    }


def tensor_summary(value: Any, *, tensor_name: str) -> dict[str, Any]:
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().to(torch.float32).cpu().numpy()
        return summarize_numpy_array(array, tensor_name=tensor_name)
    if isinstance(value, dict):
        return {
            "tensor_name": tensor_name,
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys()),
        }
    if isinstance(value, (list, tuple)):
        return {
            "tensor_name": tensor_name,
            "type": type(value).__name__,
            "length": len(value),
        }
    return {"tensor_name": tensor_name, "type": type(value).__name__}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="s0_024")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--route-dir", default=str(DEFAULT_ROUTE_DIR))
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--inventory-path", default=None)
    parser.add_argument("--runtime-weight-archive-path", default=None)
    parser.add_argument("--worker-script", default=str(ROOT / "scripts/stage2_native_int8_tvm_worker.py"))
    parser.add_argument(
        "--persistent-worker",
        action="store_true",
        help="reuse a single long-lived TVM worker process (load .so/weights once) "
        "instead of spawning a fresh process per frame; large speedup for full-val.",
    )
    parser.add_argument("--tvm-python", default=DEFAULT_TVM_PYTHON)
    parser.add_argument("--tvm-ld-library-path", default=DEFAULT_TVM_LD_LIBRARY_PATH)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--full-ap-min-samples", type=int, default=1789)
    parser.add_argument("--ap-row-min-samples", type=int, default=1789)
    parser.add_argument("--keep-detailed-samples", type=int, default=1)
    parser.add_argument("--numeric-sanity-only", action="store_true")
    parser.add_argument(
        "--tensor-quant-params-path",
        default=None,
        help="Optional calibration v2 JSON whose params are used to dequantize TVM uint8 graph outputs by tensor name.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> None:
    route_dir = Path(args.route_dir)
    args.route_dir = str(route_dir)
    args.artifact_path = str(
        Path(args.artifact_path)
        if args.artifact_path
        else route_dir / "s0_024_native_int8_full_onnx_native_int8_full_onnx_tvm_graph.so"
    )
    args.inventory_path = str(
        Path(args.inventory_path) if args.inventory_path else route_dir / "tvm_operator_inventory.json"
    )
    args.runtime_weight_archive_path = str(
        Path(args.runtime_weight_archive_path)
        if args.runtime_weight_archive_path
        else route_dir / "runtime_weights_int8.npz"
    )
    if getattr(args, "tensor_quant_params_path", None):
        args.tensor_quant_params_path = str(Path(args.tensor_quant_params_path))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _tvm_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(args.tvm_ld_library_path) + (os.pathsep + existing if existing else "")
    return env


def _run_worker(args: argparse.Namespace, request_path: Path, agent_dir: Path) -> dict[str, Any]:
    stdout_path = agent_dir / "worker_stdout.txt"
    stderr_path = agent_dir / "worker_stderr.txt"
    command = [
        str(args.tvm_python),
        str(args.worker_script),
        "--request-json",
        str(request_path),
    ]
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        env=_tvm_env(args),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    response_path = agent_dir / "native_int8_worker_response.json"
    response = _load_json(response_path) if response_path.exists() else {}
    response["worker_command"] = command
    response["worker_returncode"] = proc.returncode
    response["worker_stdout_path"] = str(stdout_path)
    response["worker_stderr_path"] = str(stderr_path)
    if proc.returncode != 0:
        raise RuntimeError(f"TVM worker failed for {request_path}: {response.get('failure_reason') or proc.stderr}")
    return response


def engine_batch_from_runtime_arg_plan(runtime_arg_plan: list[dict[str, Any]]) -> int:
    for item in runtime_arg_plan:
        if str(item.get("role")) != "graph_input":
            continue
        shape = item.get("shape")
        if not isinstance(shape, (list, tuple)) or not shape:
            raise ValueError("graph_input runtime_arg_plan item must include non-empty shape")
        engine_batch = int(shape[0])
        if engine_batch <= 0:
            raise ValueError(f"graph_input batch must be positive, got {engine_batch}")
        return engine_batch
    raise ValueError("runtime_arg_plan missing graph_input item")


def prepare_worker_activation_uint8(
    activation: np.ndarray,
    *,
    engine_batch: int,
) -> tuple[np.ndarray, dict[str, int]]:
    arr = np.asarray(activation, dtype=np.uint8)
    if arr.ndim != 4:
        raise ValueError(f"expected quantized activation rank 4, got shape {arr.shape}")
    original_agents = int(arr.shape[0])
    if original_agents <= 0:
        raise ValueError("quantized activation batch must be positive")
    if original_agents > int(engine_batch):
        raise ValueError(f"activation batch {original_agents} exceeds engine batch {int(engine_batch)}")
    padded = np.zeros((int(engine_batch), *arr.shape[1:]), dtype=np.uint8)
    padded[:original_agents] = arr
    return np.ascontiguousarray(padded), {
        "original_agents": original_agents,
        "engine_batch": int(engine_batch),
        "padding_agents": int(engine_batch) - original_agents,
    }


def slice_worker_output_to_original_agents(output: np.ndarray, *, original_agents: int) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim < 1:
        raise ValueError("worker output must have a batch dimension")
    if int(arr.shape[0]) < int(original_agents):
        raise ValueError(f"worker output batch {arr.shape[0]} is smaller than original_agents {int(original_agents)}")
    return np.ascontiguousarray(arr[: int(original_agents)])


class NativeInt8BackboneBridge:
    def __init__(self, args: argparse.Namespace, raw_dir: Path, inventory: dict[str, Any]) -> None:
        self.args = args
        self.raw_dir = raw_dir
        self.inventory = inventory
        self.call_index = 0
        self.activation_summaries: list[dict[str, Any]] = []
        self.activation_quant_summaries: list[dict[str, Any]] = []
        self.output_summaries: list[dict[str, Any]] = []
        self.worker_responses: list[dict[str, Any]] = []
        self.reference_get_multiscale: Any = None
        self.numeric_sanity_records: list[dict[str, Any]] = []
        self.output_dequant_summaries: list[dict[str, Any]] = []
        self.padding_records: list[dict[str, Any]] = []
        self.tensor_quant_params_path: str | None = None
        self.tensor_quant_params_digest: str | None = None
        self._worker_proc = None
        self._worker_stderr = None
        self._worker_command: list[str] | None = None
        self.tensor_quant_params: dict[str, Any] = {}
        if getattr(args, "tensor_quant_params_path", None):
            path = Path(str(args.tensor_quant_params_path))
            payload = _load_json(path)
            self.tensor_quant_params = dict(payload.get("params") or payload)
            self.tensor_quant_params_path = str(path)
            self.tensor_quant_params_digest = sha256_file(path)

    def _ensure_persistent_worker(self):
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return self._worker_proc
        server_log = self.raw_dir / "persistent_worker_stderr.txt"
        self._worker_stderr = open(server_log, "a")
        command = [str(self.args.tvm_python), str(self.args.worker_script), "--server"]
        self._worker_command = command
        self._worker_proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=_tvm_env(self.args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._worker_stderr,
            text=True,
            bufsize=1,
        )
        ready = self._worker_proc.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(
                f"persistent TVM worker failed to start (got {ready!r}); see {server_log}"
            )
        return self._worker_proc

    def _run_worker_persistent(self, request_path: Path, agent_dir: Path) -> dict[str, Any]:
        proc = self._ensure_persistent_worker()
        proc.stdin.write(f"{request_path}\n")
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        response_path = agent_dir / "native_int8_worker_response.json"
        if not line:
            raise RuntimeError(
                f"persistent TVM worker produced no output for {request_path} "
                f"(worker may have died); see {self.raw_dir / 'persistent_worker_stderr.txt'}"
            )
        status, _, _payload = line.partition(" ")
        response = _load_json(response_path) if response_path.exists() else {}
        response["worker_command"] = self._worker_command
        response["worker_mode"] = "persistent"
        response["worker_server_line"] = line
        if status != "DONE":
            raise RuntimeError(
                f"persistent TVM worker failed for {request_path}: "
                f"{response.get('failure_reason') or line}"
            )
        return response

    def _dispatch_worker(self, request_path: Path, agent_dir: Path) -> dict[str, Any]:
        if getattr(self.args, "persistent_worker", False):
            return self._run_worker_persistent(request_path, agent_dir)
        return _run_worker(self.args, request_path, agent_dir)

    def close(self) -> None:
        proc = self._worker_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.stdin.write("STOP\n")
                proc.stdin.flush()
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
        if self._worker_stderr is not None:
            try:
                self._worker_stderr.close()
            except Exception:
                pass
        self._worker_proc = None

    def __call__(self, x: Any) -> tuple[Any, ...]:
        import torch

        call_index = self.call_index
        keep_detail = call_index < int(getattr(self.args, "keep_detailed_samples", 1))
        call_dir = (
            self.raw_dir / f"bridge_call_{call_index:03d}"
            if keep_detail
            else self.raw_dir / "worker_tmp" / f"bridge_call_{call_index:06d}"
        )
        call_dir.mkdir(parents=True, exist_ok=True)
        self.call_index += 1
        full_activation = x.detach().to(torch.float32).cpu().numpy()
        full_activation_path = call_dir / "activation_float32.npy"
        if keep_detail:
            np.save(full_activation_path, full_activation)
        self.activation_summaries.append(
            summarize_numpy_array(
                full_activation,
                tensor_name="spatial_features_full_batch",
                path=full_activation_path if keep_detail else None,
            )
            | {"call_index": call_index, "detail_persisted": keep_detail}
        )

        collected: list[list[Any]] | None = None
        reference_outputs: tuple[Any, ...] | None = None
        engine_batch = engine_batch_from_runtime_arg_plan(list(self.inventory["runtime_arg_plan"]))
        if bool(getattr(self.args, "numeric_sanity_only", False)):
            if self.reference_get_multiscale is None:
                raise RuntimeError("numeric sanity mode requires reference_get_multiscale")
            reference_outputs = tuple(self.reference_get_multiscale(x))
        success = False
        try:
            for chunk_start in range(0, int(x.shape[0]), int(engine_batch)):
                chunk_end = min(chunk_start + int(engine_batch), int(x.shape[0]))
                chunk_agents = chunk_end - chunk_start
                chunk_index = chunk_start // int(engine_batch)
                agent_dir = call_dir / f"chunk_{chunk_index:03d}"
                agent_dir.mkdir(parents=True, exist_ok=True)
                activation = x[chunk_start:chunk_end].detach().to(torch.float32).cpu().numpy()
                activation_float_path = agent_dir / "activation_float32.npy"
                if keep_detail:
                    np.save(activation_float_path, activation)
                input_quant_params = self.tensor_quant_params.get("spatial_features") or {}
                if input_quant_params:
                    activation_uint8, quant_summary = quantize_activation_uint8_static(
                        activation,
                        tensor_name=f"spatial_features_chunk_{chunk_index:03d}",
                        scale=float(input_quant_params["scale"]),
                        zero_point=int(input_quant_params.get("zero_point", 0)),
                        source=str(input_quant_params.get("source") or "tensor_quant_params_v2"),
                    )
                else:
                    activation_uint8, quant_summary = quantize_activation_uint8(
                        activation,
                        tensor_name=f"spatial_features_chunk_{chunk_index:03d}",
                    )
                worker_activation, batch_meta = prepare_worker_activation_uint8(
                    activation_uint8,
                    engine_batch=int(engine_batch),
                )
                activation_uint8_path = agent_dir / "activation_uint8.npy"
                np.save(activation_uint8_path, worker_activation)
                quant_summary.update(
                    {
                        "activation_float32_path": str(activation_float_path) if keep_detail else "",
                        "activation_uint8_path": str(activation_uint8_path) if keep_detail else "",
                        "chunk_index": chunk_index,
                        "agent_start_index": chunk_start,
                        "agent_end_index_exclusive": chunk_end,
                        **batch_meta,
                        "call_index": call_index,
                        "detail_persisted": keep_detail,
                    }
                )
                self.activation_quant_summaries.append(quant_summary)
                self.padding_records.append(
                    {
                        "call_index": call_index,
                        "chunk_index": chunk_index,
                        "agent_start_index": chunk_start,
                        "agent_end_index_exclusive": chunk_end,
                        **batch_meta,
                    }
                )
                write_json(agent_dir / "activation_quant_summary.json", quant_summary)
                request = build_tvm_worker_request(
                    label=str(self.args.label),
                    run_id=f"{self.args.label}_real_activation_bridge_call{call_index:03d}_chunk{chunk_index:03d}",
                    artifact_path=Path(self.args.artifact_path),
                    inventory_path=Path(self.args.inventory_path),
                    runtime_weight_archive_path=Path(self.args.runtime_weight_archive_path),
                    activation_npy_path=activation_uint8_path,
                    output_dir=agent_dir,
                    gpu=0,
                    runtime_arg_plan=list(self.inventory["runtime_arg_plan"]),
                )
                request["runtime_weight_archive_keys"] = self.inventory.get("runtime_weight_archive_keys", {})
                if self.inventory.get("execution_abi"):
                    request["execution_abi"] = str(self.inventory["execution_abi"])
                if self.tensor_quant_params_path and self.tensor_quant_params_digest:
                    attach_tensor_quant_params_to_worker_request(
                        request,
                        tensor_quant_params_path=self.tensor_quant_params_path,
                        tensor_quant_params_digest=self.tensor_quant_params_digest,
                    )
                request_path = agent_dir / "native_int8_worker_request.json"
                write_json(request_path, request)
                response = self._dispatch_worker(request_path, agent_dir)
                response["call_index"] = call_index
                response["chunk_index"] = chunk_index
                response["agent_start_index"] = chunk_start
                response["agent_end_index_exclusive"] = chunk_end
                response.update(batch_meta)
                response["detail_persisted"] = keep_detail
                self.worker_responses.append(response)
                outputs = []
                for record in response.get("outputs", []):
                    output_path = Path(str(record["path"]))
                    if not output_path.exists():
                        output_path = agent_dir / output_path.name
                    arr_u8 = slice_worker_output_to_original_agents(
                        np.load(output_path),
                        original_agents=chunk_agents,
                    )
                    output_idx = len(outputs)
                    dequant, dequant_details = dequantize_tvm_output_uint8(
                        arr_u8,
                        tensor_name=str(record["arg_name"]),
                        activation_quant=quant_summary,
                        tensor_quant_params=self.tensor_quant_params,
                    )
                    if reference_outputs is not None:
                        ref_np = (
                            reference_outputs[output_idx][chunk_start:chunk_end]
                            .detach()
                            .to(torch.float32)
                            .cpu()
                            .numpy()
                        )
                        self.numeric_sanity_records.append(
                            build_output_dequant_sanity_record(
                                tensor_name=str(record["arg_name"]),
                                reference=ref_np,
                                tvm_uint8=arr_u8,
                                sample_index=call_index,
                                candidate_dequant=dequant,
                                dequant_details=dequant_details,
                            )
                            | {
                                "chunk_index": chunk_index,
                                "agent_start_index": chunk_start,
                                "agent_end_index_exclusive": chunk_end,
                                "output_index": output_idx,
                            }
                        )
                    self.output_summaries.append(
                        summarize_numpy_array(
                            arr_u8,
                            tensor_name=str(record["arg_name"]),
                            path=output_path if keep_detail else None,
                        )
                        | {
                            "chunk_index": chunk_index,
                            "agent_start_index": chunk_start,
                            "agent_end_index_exclusive": chunk_end,
                            **batch_meta,
                            "call_index": call_index,
                            "detail_persisted": keep_detail,
                        }
                    )
                    dequant_details.update(
                        {
                            "chunk_index": chunk_index,
                            "agent_start_index": chunk_start,
                            "agent_end_index_exclusive": chunk_end,
                            **batch_meta,
                            "call_index": call_index,
                            "detail_persisted": keep_detail,
                        }
                    )
                    self.output_dequant_summaries.append(dequant_details)
                    outputs.append(torch.from_numpy(dequant).to(device=x.device, dtype=torch.float32))
                if collected is None:
                    collected = [[] for _ in outputs]
                for idx, tensor in enumerate(outputs):
                    collected[idx].append(tensor)
            success = True
        finally:
            if success and not keep_detail and call_dir.exists():
                shutil.rmtree(call_dir, ignore_errors=True)

        if collected is None:
            raise RuntimeError("native INT8 bridge did not collect any worker outputs")
        if bool(getattr(self.args, "numeric_sanity_only", False)) and reference_outputs is not None:
            return reference_outputs
        return tuple(torch.cat(items, dim=0) for items in collected)


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    heal_root = Path(args.heal_root)
    sys.path.insert(0, str(heal_root))
    os.chdir(heal_root)

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.utils import eval_utils
    from opencood.utils.common_utils import update_dict

    ckpt_dir = Path(args.ckpt_dir)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    opt = argparse.Namespace(
        model_dir=str(ckpt_dir),
        fusion_method="intermediate",
        save_vis_interval=10**9,
        save_npy=False,
        range=args.eval_range,
        no_score=True,
        note="stage2_native_int8_real_activation_bridge",
    )
    hypes = yaml_utils.load_yaml(None, opt)
    if "heter" in hypes:
        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        new_cav_range = [
            x_min,
            y_min,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max,
            y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(
            hypes,
            {
                "cav_lidar_range": new_cav_range,
                "lidar_range": new_cav_range,
                "gt_range": new_cav_range,
            },
        )
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        parser_func = getattr(yaml_utils_lib, hypes["yaml_parser"])
        hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    ckpt_path = best_checkpoint(ckpt_dir)
    resume_epoch = load_exact_checkpoint(
        model, ckpt_path, torch_module=torch, train_utils_module=train_utils
    )
    model = model.to(device).eval()
    model_dtype_summary = dtype_counts(model)

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    inventory = _load_json(Path(args.inventory_path))
    bridge = NativeInt8BackboneBridge(args, raw_dir, inventory)
    original_get_multiscale = model.pyramid_backbone.get_multiscale_feature
    bridge.reference_get_multiscale = original_get_multiscale
    model.pyramid_backbone.get_multiscale_feature = bridge

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    processed = 0
    failed = 0
    pred_nonempty_count = 0
    pred_total_count = 0
    head_summary_items: list[dict[str, Any]] = []
    postprocess_summary_items: list[dict[str, Any]] = []
    t0 = time.time()
    for sample_index, batch_data in enumerate(loader):
        if args.num_samples is not None and processed >= int(args.num_samples):
            break
        if batch_data is None:
            continue
        try:
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                output_dict: OrderedDict[str, Any] = OrderedDict()
                with nullcontext():
                    output_dict["ego"] = model(batch_data["ego"])
                head_summary = {
                    key: tensor_summary(value, tensor_name=key)
                    for key, value in output_dict["ego"].items()
                }
                head_summary["sample_index"] = sample_index
                output_dict["ego"] = cast_floating_tensors(output_dict["ego"], torch.float32)
                pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
                for threshold in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(
                        pred_box_tensor,
                        pred_score,
                        gt_box_tensor,
                        result_stat,
                        threshold,
                    )
                pred_count = int(pred_score.shape[0]) if pred_score is not None else 0
                pred_total_count += pred_count
                if pred_count > 0:
                    pred_nonempty_count += 1
                postprocess_summary = {
                    "status": "success",
                    "sample_index": sample_index,
                    "pred_count": pred_count,
                    "pred_box_tensor": tensor_summary(pred_box_tensor, tensor_name="pred_box_tensor"),
                    "pred_score": tensor_summary(pred_score, tensor_name="pred_score"),
                    "gt_box_tensor": tensor_summary(gt_box_tensor, tensor_name="gt_box_tensor"),
                }
                if len(head_summary_items) < int(args.keep_detailed_samples):
                    head_summary_items.append(head_summary)
                if len(postprocess_summary_items) < int(args.keep_detailed_samples):
                    postprocess_summary_items.append(postprocess_summary)
                processed += 1
        except Exception as exc:
            failed += 1
            sample_blocker = {
                "schema": "native_int8_real_activation_bridge_sample_blocker_v1",
                "status": "failed",
                "sample_index": sample_index,
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(),
            }
            write_json(raw_dir / f"sample_{sample_index:06d}_blocker.json", sample_blocker)
            if processed == 0:
                raise
    if processed == 0:
        raise RuntimeError("no non-empty batch processed for native INT8 real activation bridge")

    ap30, ap50, ap70 = eval_utils.eval_final_results(
        result_stat,
        str(raw_dir),
        f"{args.label}_native_int8_bridge",
    )
    write_json(
        raw_dir / "activation_quant_summary.json",
        {
            "full_batch_items": bridge.activation_summaries,
            "per_agent_quant_items": bridge.activation_quant_summaries,
            "padding_items": bridge.padding_records,
        },
    )
    write_json(raw_dir / "multiscale_output_summary.json", {"items": bridge.output_summaries})
    write_json(
        raw_dir / "output_dequant_summary.json",
        {
            "items": bridge.output_dequant_summaries,
            "tensor_quant_params_path": bridge.tensor_quant_params_path,
            "tensor_quant_params_digest": bridge.tensor_quant_params_digest,
            "full_network_claim": False,
            "ap_measured": False,
        },
    )
    bridge.close()
    write_json(raw_dir / "worker_response_summary.json", {"items": bridge.worker_responses})
    write_json(raw_dir / "head_output_summary.json", {"items": head_summary_items})
    write_json(raw_dir / "postprocess_summary.json", {"items": postprocess_summary_items})
    if bool(args.numeric_sanity_only):
        sanity_summary = summarize_dequant_sanity(bridge.numeric_sanity_records)
        sanity_summary.update(
            {
                "label": str(args.label),
                "processed_samples": processed,
                "records": len(bridge.numeric_sanity_records),
                "raw_artifact": str(raw_dir),
            }
        )
        write_json(
            raw_dir / "numeric_sanity_summary.json",
            {"items": bridge.numeric_sanity_records},
        )
        write_json(raw_dir / "output_dequant_calibration_summary.json", sanity_summary)
        report = {
            "schema": "native_int8_output_dequant_numeric_sanity_report_v1",
            "status": sanity_summary["status"],
            "label": str(args.label),
            "processed_samples": processed,
            "failed_samples": failed,
            "raw_artifact": str(raw_dir),
            "output_dequant_calibration_summary": str(raw_dir / "output_dequant_calibration_summary.json"),
            "numeric_sanity_summary": str(raw_dir / "numeric_sanity_summary.json"),
            "full_network_claim": False,
            "ap_measured": False,
        }
        write_json(raw_dir / "real_activation_bridge_report.json", report)
        return report
    report = {
        "schema": "native_int8_full_ap_gate_report_v1",
        "status": "success",
        "label": str(args.label),
        "processed_samples": processed,
        "failed_samples": failed,
        "requested_num_samples": int(args.num_samples) if args.num_samples is not None else None,
        "pred_nonempty_count": pred_nonempty_count,
        "pred_total_count": pred_total_count,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_secs": time.time() - t0,
        "resume_epoch": int(resume_epoch),
        "ckpt_path": str(best_checkpoint(ckpt_dir)),
        "config_path": str(ckpt_dir / "config.yaml"),
        "raw_artifact": str(raw_dir),
        "model_dtype_counts": model_dtype_summary,
        "input_route": "HEAL spatial_features real activation -> TVM native INT8 worker -> PyTorch head/postprocess",
        "full_network_claim": False,
        "ap_measured": False,
        "output_dequant_policy": (
            "per_output_tensor_quant_params_v2_with_activation_quant_fallback"
            if bridge.tensor_quant_params
            else "temporary activation minmax scale applied to uint8 TVM outputs for one-sample postprocess smoke; not valid AP measured evidence"
        ),
        "tensor_quant_params_path": bridge.tensor_quant_params_path,
        "tensor_quant_params_digest": bridge.tensor_quant_params_digest,
        "padding_summary": {
            "calls": len(bridge.padding_records),
            "padded_calls": sum(1 for item in bridge.padding_records if int(item.get("padding_agents") or 0) > 0),
            "padded_agents": sum(int(item.get("padding_agents") or 0) for item in bridge.padding_records),
            "engine_batch": (
                int(bridge.padding_records[0]["engine_batch"])
                if bridge.padding_records
                else engine_batch_from_runtime_arg_plan(list(bridge.inventory["runtime_arg_plan"]))
            ),
        },
    }
    report.update(
        build_full_ap_report_gate_fields(
            report,
            min_samples=int(args.full_ap_min_samples),
            row_min_samples=int(args.ap_row_min_samples),
        )
    )
    write_json(raw_dir / "real_activation_bridge_report.json", report)
    write_json(raw_dir / "full_ap_eval_report.json", report)
    if not bool(report["ap_row_allowed"]):
        blocker = build_full_ap_blocker(
            label=str(args.label),
            raw_artifact=str(raw_dir),
            report=report,
            reason=str(report["ap_row_block_reason"]),
        )
        write_json(raw_dir / "native_int8_s0_024_full_ap_blocker.json", blocker)
    return report


def classify_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__}:{exc}".lower()
    if "worker failed" in text or "tvm" in text:
        return "tvm_worker_failure"
    if "shape" in text:
        return "shape_mismatch"
    if "post" in text or "nms" in text:
        return "postprocess_failure"
    if isinstance(exc, FileNotFoundError):
        return "missing_artifact"
    return "real_activation_bridge_failure"


def main() -> int:
    args = parse_args()
    _resolve_paths(args)
    raw_dir = Path(args.raw_dir)
    try:
        report = run_bridge(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        raw_dir.mkdir(parents=True, exist_ok=True)
        blocker = {
            "schema": "native_int8_real_activation_bridge_blocker_v1",
            "status": "failed",
            "label": str(args.label),
            "failure_type": classify_failure(exc),
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
            "raw_artifact": str(raw_dir),
            "full_network_claim": False,
            "ap_measured": False,
        }
        write_json(raw_dir / "real_activation_bridge_blocker.json", blocker)
        print(json.dumps(blocker, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
