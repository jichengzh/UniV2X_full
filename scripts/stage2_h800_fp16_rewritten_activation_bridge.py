#!/usr/bin/env python3
"""HEAL real-activation bridge for FP16 rewritten TVM backbone/subnet."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_native_int8_real_activation_bridge import (  # noqa: E402
    summarize_numpy_array,
    tensor_summary,
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
DEFAULT_REWRITE_REPORT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "exports/fp16_lhc07_full_engine_group_conv_rewrite_latest.json"
)
DEFAULT_EXPORT_DIR = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/exports"
)
SUPPORTED_ARTIFACT_INPUT_DTYPES = ("float16", "float32")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_artifact_input_dtype(value: str) -> str:
    dtype = str(value).strip().lower()
    if dtype not in SUPPORTED_ARTIFACT_INPUT_DTYPES:
        raise ValueError(
            f"unsupported artifact input dtype {value!r}; expected one of {SUPPORTED_ARTIFACT_INPUT_DTYPES}"
        )
    return dtype


def _tvm_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    extra_ld = ""
    nvlibs = Path("/exdata/jichengzhi/tvm_nvlibs.path")
    if nvlibs.exists():
        extra_ld = nvlibs.read_text(encoding="utf-8").strip()
    parts = [str(args.tvm_ld_library_path)]
    if extra_ld:
        parts.append(extra_ld)
    if env.get("LD_LIBRARY_PATH"):
        parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(item for item in parts if item)
    return env


def build_fp16_worker_request(
    *,
    label: str,
    run_id: str,
    artifact_path: Path,
    activation_npy_path: Path,
    output_dir: Path,
    gpu: int,
    activation_dtype: str = "float16",
    expected_output_shapes: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    normalized_dtype = normalize_artifact_input_dtype(activation_dtype)
    shapes = expected_output_shapes or {
        "output0": [2, 24, 128, 256],
        "output1": [2, 48, 64, 128],
        "output2": [2, 128, 32, 64],
    }
    return {
        "schema": "stage2_fp16_tvm_worker_request_v1",
        "label": str(label),
        "run_id": str(run_id),
        "artifact_path": str(artifact_path),
        "activation_npy_path": str(activation_npy_path),
        "activation_dtype": normalized_dtype,
        "output_dir": str(output_dir),
        "gpu": int(gpu),
        "output_names": ["output0", "output1", "output2"],
        "expected_output_shapes": shapes,
    }


def expected_output_shapes_from_rewrite_report(report: dict[str, Any]) -> dict[str, list[int]]:
    items = (report.get("rewritten_full_engine") or {}).get("output_compare") or []
    shapes: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        output_idx = int(item.get("output_index", idx))
        shape = item.get("rewritten_shape")
        if not shape:
            continue
        shapes[f"output{output_idx}"] = [int(dim) for dim in shape]
    if len(shapes) != 3:
        raise ValueError(f"rewrite report must contain 3 rewritten output shapes, got {len(shapes)}")
    return shapes


def expected_output_shapes_from_reference_outputs(
    reference_outputs: tuple[Any, ...] | list[Any],
    *,
    engine_batch: int,
) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for idx, value in enumerate(reference_outputs):
        arr = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
        if arr.ndim == 0:
            raise ValueError(f"reference output {idx} must have a batch dimension")
        shape = [int(dim) for dim in arr.shape]
        shape[0] = int(engine_batch)
        shapes[f"output{idx}"] = shape
    if len(shapes) != 3:
        raise ValueError(f"reference outputs must contain 3 tensors, got {len(shapes)}")
    return shapes


def load_configured_expected_output_shapes(args: argparse.Namespace) -> tuple[dict[str, list[int]] | None, str]:
    if bool(getattr(args, "rewrite_report_explicit", False)):
        shapes = expected_output_shapes_from_rewrite_report(_load_json(Path(args.rewrite_report)))
        return shapes, "explicit_rewrite_report"
    return None, "reference_outputs_dynamic"


def attach_expected_output_shape_metadata(
    report: dict[str, Any],
    *,
    source: str,
    shapes: dict[str, list[int]],
) -> dict[str, Any]:
    report["expected_output_shapes_source"] = str(source)
    report["expected_output_shapes"] = {
        str(name): [int(dim) for dim in dims]
        for name, dims in dict(shapes).items()
    }
    return report


def prepare_worker_activation(
    activation: np.ndarray,
    *,
    engine_batch: int = 2,
    activation_dtype: str = "float16",
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(activation)
    normalized_dtype = normalize_artifact_input_dtype(activation_dtype)
    if arr.ndim != 4:
        raise ValueError(f"expected NCHW activation rank 4, got shape {arr.shape}")
    original_batch = int(arr.shape[0])
    if original_batch <= 0:
        raise ValueError(f"invalid activation batch: {original_batch}")
    if original_batch > int(engine_batch):
        raise ValueError(f"activation batch {original_batch} exceeds fixed engine batch {int(engine_batch)}")
    worker = np.zeros((int(engine_batch), *arr.shape[1:]), dtype=np.dtype(normalized_dtype))
    worker[:original_batch] = arr.astype(np.dtype(normalized_dtype))
    return np.ascontiguousarray(worker), {
        "original_batch": original_batch,
        "engine_batch": int(engine_batch),
        "padding_batch": int(engine_batch) - original_batch,
        "requested_activation_dtype": normalized_dtype,
    }


def slice_worker_output_to_original_batch(output: np.ndarray, *, original_batch: int) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 0:
        raise ValueError("worker output must have a batch dimension")
    if int(arr.shape[0]) < int(original_batch):
        raise ValueError(f"worker output batch {arr.shape[0]} is smaller than original batch {int(original_batch)}")
    return np.ascontiguousarray(arr[: int(original_batch)])


def output_error_record(name: str, reference: Any, candidate: np.ndarray) -> dict[str, Any]:
    ref = reference.detach().to("cpu").to(dtype=reference.dtype).numpy().astype(np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        return {
            "tensor_name": name,
            "status": "shape_mismatch",
            "reference_shape": list(ref.shape),
            "candidate_shape": list(cand.shape),
        }
    diff = cand - ref
    abs_diff = np.abs(diff)
    denom = float(np.mean(np.abs(ref))) if ref.size else 0.0
    return {
        "tensor_name": name,
        "status": "compared",
        "shape": [int(dim) for dim in ref.shape],
        "max_abs_err": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_err": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "mean_err": float(np.mean(diff)) if diff.size else 0.0,
        "reference_abs_mean": denom,
        "mean_abs_err_over_reference_abs_mean": float(np.mean(abs_diff) / denom) if denom else None,
    }


def _response_path(call_dir: Path) -> Path:
    return call_dir / "fp16_tvm_worker_response.json"


def _run_worker(args: argparse.Namespace, request_path: Path, call_dir: Path) -> dict[str, Any]:
    stdout_path = call_dir / "worker_stdout.txt"
    stderr_path = call_dir / "worker_stderr.txt"
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
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    response = _load_json(_response_path(call_dir)) if _response_path(call_dir).exists() else {}
    response["worker_command"] = command
    response["worker_returncode"] = proc.returncode
    response["worker_stdout_path"] = str(stdout_path)
    response["worker_stderr_path"] = str(stderr_path)
    if proc.returncode != 0:
        raise RuntimeError(f"FP16 TVM worker failed for {request_path}: {response.get('failure_reason') or proc.stderr}")
    return response


class Fp16RewrittenBackboneBridge:
    def __init__(self, args: argparse.Namespace, raw_dir: Path) -> None:
        self.args = args
        self.raw_dir = raw_dir
        self.configured_expected_output_shapes, self.configured_expected_output_shapes_source = (
            load_configured_expected_output_shapes(args)
        )
        self.last_expected_output_shapes: dict[str, list[int]] = (
            dict(self.configured_expected_output_shapes) if self.configured_expected_output_shapes else {}
        )
        self.last_expected_output_shapes_source = self.configured_expected_output_shapes_source
        self.call_index = 0
        self.reference_get_multiscale: Any = None
        self.activation_summaries: list[dict[str, Any]] = []
        self.output_summaries: list[dict[str, Any]] = []
        self.output_error_records: list[dict[str, Any]] = []
        self.worker_responses: list[dict[str, Any]] = []
        self._worker_proc = None
        self._worker_stderr = None
        self._worker_command: list[str] | None = None

    def _ensure_persistent_worker(self):
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return self._worker_proc
        server_log = self.raw_dir / "persistent_worker_stderr.txt"
        self._worker_stderr = open(server_log, "a", encoding="utf-8")
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
            raise RuntimeError(f"persistent FP16 TVM worker failed to start (got {ready!r}); see {server_log}")
        return self._worker_proc

    def _run_worker_persistent(self, request_path: Path, call_dir: Path) -> dict[str, Any]:
        proc = self._ensure_persistent_worker()
        proc.stdin.write(f"{request_path}\n")
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        response = _load_json(_response_path(call_dir)) if _response_path(call_dir).exists() else {}
        response["worker_command"] = self._worker_command
        response["worker_mode"] = "persistent"
        response["worker_server_line"] = line
        if not line or not line.startswith("DONE "):
            raise RuntimeError(f"persistent FP16 TVM worker failed for {request_path}: {response.get('failure_reason') or line}")
        return response

    def _dispatch_worker(self, request_path: Path, call_dir: Path) -> dict[str, Any]:
        if bool(getattr(self.args, "persistent_worker", False)):
            return self._run_worker_persistent(request_path, call_dir)
        return _run_worker(self.args, request_path, call_dir)

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

        if self.reference_get_multiscale is None:
            raise RuntimeError("reference_get_multiscale is required for output error audit")
        call_index = self.call_index
        keep_detail = call_index < int(getattr(self.args, "keep_detailed_samples", 1))
        call_dir = (
            self.raw_dir / f"bridge_call_{call_index:03d}"
            if keep_detail
            else self.raw_dir / "worker_tmp" / f"bridge_call_{call_index:06d}"
        )
        call_dir.mkdir(parents=True, exist_ok=True)
        self.call_index += 1

        reference_outputs = tuple(self.reference_get_multiscale(x))
        original_activation = x.detach().to(torch.float32).cpu().numpy()
        requested_activation_dtype = normalize_artifact_input_dtype(self.args.artifact_input_dtype)
        activation, batch_meta = prepare_worker_activation(
            original_activation,
            engine_batch=2,
            activation_dtype=requested_activation_dtype,
        )
        if self.configured_expected_output_shapes is not None:
            expected_output_shapes = dict(self.configured_expected_output_shapes)
            expected_output_shapes_source = self.configured_expected_output_shapes_source
        else:
            expected_output_shapes = expected_output_shapes_from_reference_outputs(
                reference_outputs,
                engine_batch=int(batch_meta["engine_batch"]),
            )
            expected_output_shapes_source = "reference_outputs_dynamic"
        self.last_expected_output_shapes = dict(expected_output_shapes)
        self.last_expected_output_shapes_source = expected_output_shapes_source
        activation_path = call_dir / f"activation_{requested_activation_dtype}.npy"
        np.save(activation_path, activation)
        self.activation_summaries.append(
            summarize_numpy_array(
                original_activation,
                tensor_name="spatial_features_full_batch",
                path=activation_path if keep_detail else None,
            )
            | {
                "call_index": call_index,
                "detail_persisted": keep_detail,
                "worker_batch_meta": batch_meta,
                "saved_activation_dtype": requested_activation_dtype,
                "saved_activation_path": str(activation_path),
                "expected_output_shapes_source": expected_output_shapes_source,
                "expected_output_shapes": expected_output_shapes,
            }
        )

        request = build_fp16_worker_request(
            label=str(self.args.label),
            run_id=f"{self.args.label}_fp16_rewritten_bridge_call{call_index:03d}",
            artifact_path=Path(self.args.artifact_path),
            activation_npy_path=activation_path,
            output_dir=call_dir,
            gpu=0,
            activation_dtype=requested_activation_dtype,
            expected_output_shapes=expected_output_shapes,
        )
        request_path = call_dir / "fp16_tvm_worker_request.json"
        write_json(request_path, request)
        response = self._dispatch_worker(request_path, call_dir)
        response["call_index"] = call_index
        response["detail_persisted"] = keep_detail
        response["expected_output_shapes_source"] = expected_output_shapes_source
        response["expected_output_shapes"] = expected_output_shapes
        self.worker_responses.append(response)

        outputs = []
        for idx, record in enumerate(response.get("outputs", [])):
            output_path = Path(str(record["path"]))
            if not output_path.exists():
                output_path = call_dir / output_path.name
            raw_array = np.load(output_path)
            array = slice_worker_output_to_original_batch(
                raw_array,
                original_batch=int(batch_meta["original_batch"]),
            )
            name = str(record.get("arg_name") or f"output{idx}")
            self.output_summaries.append(
                summarize_numpy_array(array, tensor_name=name, path=output_path if keep_detail else None)
                | {
                    "call_index": call_index,
                    "output_index": idx,
                    "detail_persisted": keep_detail,
                    "raw_worker_shape": [int(dim) for dim in raw_array.shape],
                    "sliced_to_original_batch": int(batch_meta["original_batch"]),
                }
            )
            self.output_error_records.append(
                output_error_record(name, reference_outputs[idx], array)
                | {"call_index": call_index, "output_index": idx}
            )
            outputs.append(torch.from_numpy(array.astype(np.float32)).to(device=x.device, dtype=torch.float32))

        if len(outputs) != 3:
            raise RuntimeError(f"FP16 TVM bridge expected 3 outputs, got {len(outputs)}")
        if not keep_detail and call_dir.exists():
            shutil.rmtree(call_dir, ignore_errors=True)
        return tuple(outputs)


def _prediction_distribution(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(item.get("pred_count") or 0) for item in items if item.get("status") == "success"]
    return {
        "samples": len(counts),
        "nonempty": int(sum(1 for item in counts if item > 0)),
        "total_predictions": int(sum(counts)),
        "min_predictions": int(min(counts)) if counts else None,
        "max_predictions": int(max(counts)) if counts else None,
        "mean_predictions": float(np.mean(counts)) if counts else None,
    }


class StreamingTensorDistribution:
    def __init__(self, tensor_name: str):
        self.tensor_name = str(tensor_name)
        self.samples = 0
        self.size = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_value: float | None = None
        self.max_value: float | None = None

    def update(self, value: Any) -> None:
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float64)
        self.samples += 1
        if arr.size == 0:
            return
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return
        current_min = float(np.min(finite))
        current_max = float(np.max(finite))
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)
        self.size += int(finite.size)
        self.sum += float(np.sum(finite))
        self.sum_sq += float(np.sum(finite * finite))

    def to_summary(self) -> dict[str, Any]:
        if self.size == 0:
            return {
                "tensor_name": self.tensor_name,
                "samples": int(self.samples),
                "size": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        mean = self.sum / float(self.size)
        variance = max(self.sum_sq / float(self.size) - mean * mean, 0.0)
        return {
            "tensor_name": self.tensor_name,
            "samples": int(self.samples),
            "size": int(self.size),
            "min": self.min_value,
            "max": self.max_value,
            "mean": float(mean),
            "std": float(np.sqrt(variance)),
        }


class FullPostprocessDistribution:
    def __init__(self):
        self.pred_counts: list[int] = []
        self.pred_score = StreamingTensorDistribution("pred_score")
        self.pred_box_tensor = StreamingTensorDistribution("pred_box_tensor")
        self.gt_box_tensor = StreamingTensorDistribution("gt_box_tensor")

    def update(
        self,
        *,
        pred_count: int,
        pred_score: Any,
        pred_box_tensor: Any,
        gt_box_tensor: Any,
    ) -> None:
        self.pred_counts.append(int(pred_count))
        self.pred_score.update(pred_score)
        self.pred_box_tensor.update(pred_box_tensor)
        self.gt_box_tensor.update(gt_box_tensor)

    def prediction_distribution(self) -> dict[str, Any]:
        counts = self.pred_counts
        return {
            "samples": len(counts),
            "nonempty": int(sum(1 for item in counts if item > 0)),
            "total_predictions": int(sum(counts)),
            "min_predictions": int(min(counts)) if counts else None,
            "max_predictions": int(max(counts)) if counts else None,
            "mean_predictions": float(np.mean(counts)) if counts else None,
        }

    def to_summary(self) -> dict[str, Any]:
        return {
            "prediction_distribution": self.prediction_distribution(),
            "pred_score": self.pred_score.to_summary(),
            "pred_box_tensor": self.pred_box_tensor.to_summary(),
            "gt_box_tensor": self.gt_box_tensor.to_summary(),
        }


def smoke_gate_status(*, pred_nonempty_count: int, ap_values: list[float]) -> bool:
    return bool(int(pred_nonempty_count) > 0 and np.isfinite(ap_values).all())


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
    ckpt_path = resolve_checkpoint_path(
        ckpt_dir,
        Path(args.checkpoint_path) if args.checkpoint_path else None,
    )
    resume_epoch = checkpoint_epoch(ckpt_path)

    opt = argparse.Namespace(
        model_dir=str(ckpt_dir),
        fusion_method="intermediate",
        save_vis_interval=10**9,
        save_npy=False,
        range=args.eval_range,
        no_score=True,
        note="stage2_fp16_rewritten_activation_bridge",
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
            {"cav_lidar_range": new_cav_range, "lidar_range": new_cav_range, "gt_range": new_cav_range},
        )
        parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"), hypes["yaml_parser"])
        hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
    train_utils.check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)
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
    bridge = Fp16RewrittenBackboneBridge(args, raw_dir)
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
    postprocess_distribution = FullPostprocessDistribution()
    t0 = time.time()

    try:
        for sample_index, batch_data in enumerate(loader):
            if args.num_samples is not None and processed >= int(args.num_samples):
                break
            if batch_data is None:
                continue
            try:
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict: OrderedDict[str, Any] = OrderedDict()
                    output_dict["ego"] = model(batch_data["ego"])
                    head_summary = {
                        key: tensor_summary(value, tensor_name=key)
                        for key, value in output_dict["ego"].items()
                    }
                    head_summary["sample_index"] = sample_index
                    output_dict["ego"] = cast_floating_tensors(output_dict["ego"], torch.float32)
                    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
                    for threshold in (0.3, 0.5, 0.7):
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, threshold)
                    pred_count = int(pred_score.shape[0]) if pred_score is not None else 0
                    pred_total_count += pred_count
                    if pred_count > 0:
                        pred_nonempty_count += 1
                    postprocess_distribution.update(
                        pred_count=pred_count,
                        pred_score=pred_score.detach().cpu().numpy() if pred_score is not None else None,
                        pred_box_tensor=pred_box_tensor.detach().cpu().numpy() if pred_box_tensor is not None else None,
                        gt_box_tensor=gt_box_tensor.detach().cpu().numpy() if gt_box_tensor is not None else None,
                    )
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
                write_json(
                    raw_dir / f"sample_{sample_index:06d}_blocker.json",
                    {
                        "schema": "fp16_rewritten_activation_bridge_sample_blocker_v1",
                        "status": "failed",
                        "sample_index": sample_index,
                        "failure_reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                if processed == 0:
                    raise
    finally:
        bridge.close()

    if processed == 0:
        raise RuntimeError("no non-empty batch processed for FP16 rewritten activation bridge")

    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(raw_dir), f"{args.label}_fp16_rewritten")
    postprocess_distribution_summary = postprocess_distribution.to_summary()
    report = {
        "schema": "fp16_rewritten_activation_bridge_report_v1",
        "status": "success",
        "label": str(args.label),
        "processed_samples": processed,
        "failed_samples": failed,
        "requested_num_samples": int(args.num_samples) if args.num_samples is not None else None,
        "pred_nonempty_count": pred_nonempty_count,
        "pred_total_count": pred_total_count,
        "prediction_distribution": postprocess_distribution_summary["prediction_distribution"],
        "postprocess_distribution": postprocess_distribution_summary,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_secs": time.time() - t0,
        "resume_epoch": int(resume_epoch),
        "ckpt_path": str(ckpt_path),
        "config_path": str(ckpt_dir / "config.yaml"),
        "artifact_path": str(args.artifact_path),
        "artifact_input_dtype": normalize_artifact_input_dtype(args.artifact_input_dtype),
        "rewrite_report": str(args.rewrite_report),
        "raw_artifact": str(raw_dir),
        "model_dtype_counts": model_dtype_summary,
        "input_route": "HEAL spatial_features real activation -> TVM FP16 rewritten worker -> PyTorch head/postprocess",
        "full_network_claim": False,
        "ap_measured": processed >= int(args.full_ap_min_samples),
        "smoke_gate_passed": smoke_gate_status(
            pred_nonempty_count=pred_nonempty_count,
            ap_values=[float(ap30), float(ap50), float(ap70)],
        ),
    }
    attach_expected_output_shape_metadata(
        report,
        source=bridge.last_expected_output_shapes_source,
        shapes=bridge.last_expected_output_shapes,
    )
    write_json(raw_dir / "activation_summary.json", {"items": bridge.activation_summaries})
    write_json(raw_dir / "multiscale_output_summary.json", {"items": bridge.output_summaries})
    write_json(raw_dir / "output_error_summary.json", {"items": bridge.output_error_records})
    write_json(raw_dir / "worker_response_summary.json", {"items": bridge.worker_responses})
    write_json(raw_dir / "head_output_summary.json", {"items": head_summary_items})
    write_json(raw_dir / "postprocess_summary.json", {"items": postprocess_summary_items})
    write_json(raw_dir / "postprocess_distribution_summary.json", postprocess_distribution_summary)
    write_json(raw_dir / "fp16_rewritten_activation_bridge_report.json", report)
    write_json(Path(args.export_report_json), report)
    return report


def _artifact_path_from_report(report_path: Path) -> Path:
    report = _load_json(report_path)
    path = (((report.get("rewritten_full_engine") or {}).get("export_library") or {}).get("path"))
    if not path:
        raise ValueError(f"rewrite report has no rewritten_full_engine.export_library.path: {report_path}")
    return Path(str(path))


def resolve_checkpoint_path(
    checkpoint_dir: Path,
    checkpoint_path: Path | None,
) -> Path:
    if checkpoint_path is None:
        return best_checkpoint(checkpoint_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="lhc_07")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--gpu-id", type=int, default=4)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--rewrite-report", default=str(DEFAULT_REWRITE_REPORT))
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--worker-script", default=str(ROOT / "scripts/stage2_fp16_tvm_worker.py"))
    parser.add_argument("--persistent-worker", action="store_true")
    parser.add_argument("--tvm-python", default=DEFAULT_TVM_PYTHON)
    parser.add_argument("--tvm-ld-library-path", default=DEFAULT_TVM_LD_LIBRARY_PATH)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--full-ap-min-samples", type=int, default=1789)
    parser.add_argument("--keep-detailed-samples", type=int, default=1)
    parser.add_argument(
        "--artifact-input-dtype",
        choices=SUPPORTED_ARTIFACT_INPUT_DTYPES,
        default="float16",
    )
    parser.add_argument(
        "--export-report-json",
        default=str(DEFAULT_EXPORT_DIR / "fp16_lhc07_rewritten_full_engine_ap_smoke_latest.json"),
    )
    args = parser.parse_args()
    argv = sys.argv[1:]
    args.rewrite_report_explicit = "--rewrite-report" in argv
    return args


def _resolve_paths(args: argparse.Namespace) -> None:
    args.rewrite_report = str(Path(args.rewrite_report).resolve())
    args.artifact_path = str(
        Path(args.artifact_path).resolve()
        if args.artifact_path
        else _artifact_path_from_report(Path(args.rewrite_report)).resolve()
    )
    args.worker_script = str(Path(args.worker_script).resolve())
    args.ckpt_dir = str(Path(args.ckpt_dir).resolve())
    args.checkpoint_path = (
        str(Path(getattr(args, "checkpoint_path")).resolve())
        if getattr(args, "checkpoint_path", None)
        else None
    )
    args.raw_dir = str(Path(args.raw_dir).resolve())
    args.heal_root = str(Path(args.heal_root).resolve())
    args.export_report_json = str(Path(args.export_report_json).resolve())


def classify_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__}:{exc}".lower()
    if isinstance(exc, FileNotFoundError) or "checkpoint" in text or "config.yaml" in text:
        return "missing_checkpoint_or_config"
    if "worker" in text or "tvm" in text or "relax" in text:
        return "tvm_worker_failure"
    if "shape" in text:
        return "shape_mismatch"
    if "post" in text or "nms" in text:
        return "postprocess_failure"
    return "fp16_rewritten_bridge_failure"


def main() -> int:
    args = parse_args()
    _resolve_paths(args)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        raw_dir / "runner_command.json",
        {
            "command": sys.argv,
            "cwd": os.getcwd(),
            "gpu_id": args.gpu_id,
            "label": args.label,
            "checkpoint_path": args.checkpoint_path,
            "artifact_path": args.artifact_path,
            "artifact_input_dtype": args.artifact_input_dtype,
        },
    )
    try:
        report = run_bridge(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        blocker = {
            "schema": "fp16_rewritten_activation_bridge_blocker_v1",
            "status": "failed",
            "label": str(args.label),
            "failure_type": classify_failure(exc),
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
            "raw_artifact": str(raw_dir),
            "artifact_path": str(getattr(args, "artifact_path", "")),
            "rewrite_report": str(getattr(args, "rewrite_report", "")),
            "full_network_claim": False,
            "ap_measured": False,
        }
        write_json(raw_dir / "fp16_rewritten_activation_bridge_blocker.json", blocker)
        print(json.dumps(blocker, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
