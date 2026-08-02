#!/usr/bin/env python3
"""HEAL/Pyramid multiscale TensorRT AP bridge with fixed-batch padding."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
import sys
import time
import traceback
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_native_int8_real_activation_bridge import summarize_numpy_array, tensor_summary  # noqa: E402
from scripts.stage2_h800_true_fp16_ap_eval import (  # noqa: E402
    best_checkpoint,
    cast_floating_tensors,
    checkpoint_epoch,
    dtype_counts,
    write_json,
)


SCHEMA = "stage3_trt_multiscale_ap_bridge_v3"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_REPORT_JSON = "stage3_trt_multiscale_ap_bridge_report.json"
SHA256_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_expected_output_channels(value: str) -> tuple[int, int, int]:
    try:
        channels = tuple(int(item.strip()) for item in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected output channels must be comma-separated integers") from exc
    if len(channels) != 3 or any(channel <= 0 for channel in channels):
        raise argparse.ArgumentTypeError("expected output channels must contain exactly three positive integers")
    return channels


def protocol_payload(
    *,
    precision_tag: str,
    eval_range: str,
    full_ap_min_samples: int,
    expected_output_channels: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    return {
        "bridge": SCHEMA,
        "precision_tag": str(precision_tag),
        "eval_range": str(eval_range),
        "engine_batch": 2,
        "expected_output_channels": (
            [int(channel) for channel in expected_output_channels]
            if expected_output_channels is not None
            else None
        ),
        "fallback_policy": "forbidden_for_engine_ap_claim",
        "input_route": "HEAL spatial_features -> TensorRT multiscale outputs -> PyTorch fusion/head/postprocess",
        "full_ap_min_samples": int(full_ap_min_samples),
    }


def sort_output_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [{**item, "shape": [int(dim) for dim in item["shape"]]} for item in specs],
        key=lambda item: (-int(item["shape"][2]), -int(item["shape"][3]), str(item["name"])),
    )


def prepare_spatial_features_for_engine(
    spatial_features: np.ndarray,
    *,
    record_len: int,
    engine_batch: int = 2,
) -> tuple[np.ndarray, dict[str, int]]:
    array = np.asarray(spatial_features, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"spatial_features must be rank-4 NCHW, got {list(array.shape)}")
    original_agents = int(record_len)
    if original_agents <= 0:
        raise ValueError("record_len must be positive")
    if array.shape[0] != original_agents:
        raise ValueError(f"record_len {original_agents} does not match spatial_features batch {array.shape[0]}")
    if original_agents > int(engine_batch):
        raise ValueError(f"record_len {original_agents} exceeds fixed engine batch {int(engine_batch)}")
    padded = np.zeros((int(engine_batch), *array.shape[1:]), dtype=np.float32)
    padded[:original_agents] = array
    return np.ascontiguousarray(padded), {
        "original_agents": original_agents,
        "engine_batch": int(engine_batch),
        "padding_agents": int(engine_batch) - original_agents,
    }


def slice_output_to_record_len(output: np.ndarray, *, record_len: int) -> np.ndarray:
    array = np.asarray(output, dtype=np.float32)
    if array.ndim < 1:
        raise ValueError("TRT output must have a batch dimension")
    if array.shape[0] < int(record_len):
        raise ValueError(f"TRT output batch {array.shape[0]} is smaller than record_len {int(record_len)}")
    return np.ascontiguousarray(array[: int(record_len)])


def output_error_record(name: str, reference: Any, candidate: np.ndarray) -> dict[str, Any]:
    ref = reference.detach().to("cpu", dtype=reference.dtype).numpy().astype(np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    record = {
        "tensor_name": str(name),
        "reference_shape": [int(dim) for dim in ref.shape],
        "candidate_shape": [int(dim) for dim in cand.shape],
        "reference_all_finite": bool(np.isfinite(ref).all()),
        "candidate_all_finite": bool(np.isfinite(cand).all()),
    }
    if ref.shape != cand.shape:
        return {**record, "status": "shape_mismatch"}
    diff = cand - ref
    abs_diff = np.abs(diff)
    return {
        **record,
        "status": "compared",
        "diff_all_finite": bool(np.isfinite(diff).all()),
        "max_abs_err": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_err": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "mean_err": float(np.mean(diff)) if diff.size else 0.0,
    }


def summarize_error_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    compared = [item for item in records if item.get("status") == "compared"]
    shape_mismatch_count = sum(item.get("status") == "shape_mismatch" for item in records)
    nonfinite_count = sum(
        not (
            bool(item.get("reference_all_finite"))
            and bool(item.get("candidate_all_finite"))
            and (item.get("status") != "compared" or bool(item.get("diff_all_finite")))
        )
        for item in records
    )
    return {
        "num_records": len(records),
        "num_compared": len(compared),
        "shape_mismatch_count": int(shape_mismatch_count),
        "nonfinite_count": int(nonfinite_count),
        "max_abs_err": float(max(item["max_abs_err"] for item in compared)) if compared else None,
        "mean_abs_err": float(np.mean([item["mean_abs_err"] for item in compared])) if compared else None,
        "all_finite": nonfinite_count == 0,
    }


def engine_ap_gate(
    *,
    processed_samples: int,
    min_samples: int,
    fallback_samples: int,
    failed_samples: int,
    output_error_summary: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected_records = int(processed_samples) * 3
    if int(processed_samples) < int(min_samples):
        reasons.append("processed_samples_below_min")
    if int(fallback_samples) != 0:
        reasons.append("fallback_samples_present")
    if int(failed_samples) != 0:
        reasons.append("failed_samples_present")
    if int(output_error_summary.get("num_records", -1)) != expected_records:
        reasons.append("output_num_records_mismatch")
    if int(output_error_summary.get("num_compared", -1)) != expected_records:
        reasons.append("output_num_compared_mismatch")
    if int(output_error_summary.get("shape_mismatch_count", -1)) != 0:
        reasons.append("output_shape_mismatch_present")
    if output_error_summary.get("all_finite") is not True:
        reasons.append("output_nonfinite_present")
    return len(reasons) == 0, reasons


def build_report(
    *,
    label: str,
    precision_tag: str,
    ckpt_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    config_path: Path,
    config_sha256: str,
    engine_path: Path,
    engine_sha256: str,
    dataset_split_file: Path,
    dataset_split_sha256: str,
    protocol: dict[str, Any],
    processed_samples: int,
    failed_samples: int,
    fallback_samples: int,
    ap30: float,
    ap50: float,
    ap70: float,
    pred_nonempty_count: int,
    pred_total_count: int,
    model_dtype_counts: dict[str, int],
    output_error_summary: dict[str, Any],
    elapsed_secs: float,
) -> dict[str, Any]:
    claim_ok, claim_blockers = engine_ap_gate(
        processed_samples=processed_samples,
        min_samples=int(protocol["full_ap_min_samples"]),
        fallback_samples=fallback_samples,
        failed_samples=failed_samples,
        output_error_summary=output_error_summary,
    )
    return {
        "schema": SCHEMA,
        "status": "success",
        "label": str(label),
        "precision_tag": str(precision_tag),
        "ckpt_dir": str(ckpt_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_sha256": str(checkpoint_sha256),
        "config_path": str(config_path),
        "config_sha256": str(config_sha256),
        "engine_path": str(engine_path),
        "engine_sha256": str(engine_sha256),
        "dataset_split_file": str(dataset_split_file),
        "dataset_split_sha256": str(dataset_split_sha256),
        "protocol": protocol,
        "protocol_sha256": sha256_json(protocol),
        "processed_samples": int(processed_samples),
        "failed_samples": int(failed_samples),
        "fallback_samples": int(fallback_samples),
        "pred_nonempty_count": int(pred_nonempty_count),
        "pred_total_count": int(pred_total_count),
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "model_dtype_counts": dict(model_dtype_counts),
        "output_error_summary": dict(output_error_summary),
        "elapsed_secs": float(elapsed_secs),
        "full_ap_min_samples": int(protocol["full_ap_min_samples"]),
        "ap_measured": int(processed_samples) >= int(protocol["full_ap_min_samples"]),
        "engine_ap_claim": bool(claim_ok),
        "engine_ap_claim_blockers": claim_blockers,
    }


class TrtMultiscaleRunner:
    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(f"failed to create TensorRT execution context: {engine_path}")
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        if len(self.input_names) != 1:
            raise ValueError(f"expected exactly one spatial input, got {self.input_names}")
        if len(self.output_names) != 3:
            raise ValueError(f"expected exactly three multiscale outputs, got {self.output_names}")
        self.input_name = self.input_names[0]
        self.stream = torch.cuda.Stream()

    def output_specs(self, input_shape: tuple[int, ...]) -> list[dict[str, Any]]:
        self.ctx.set_input_shape(self.input_name, input_shape)
        specs = [{"name": name, "shape": list(self.ctx.get_tensor_shape(name))} for name in self.output_names]
        return sort_output_specs(specs)

    def __call__(self, spatial_features: np.ndarray) -> list[tuple[str, np.ndarray]]:
        spatial = self.torch.from_numpy(np.ascontiguousarray(spatial_features)).to(device="cuda", dtype=self.torch.float32)
        self.ctx.set_input_shape(self.input_name, tuple(spatial.shape))
        specs = self.output_specs(tuple(spatial.shape))
        buffers = {self.input_name: spatial}
        for item in specs:
            buffers[item["name"]] = self.torch.empty(tuple(item["shape"]), dtype=self.torch.float32, device="cuda")
        for name, tensor in buffers.items():
            self.ctx.set_tensor_address(name, int(tensor.data_ptr()))
        with self.torch.cuda.stream(self.stream):
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        return [
            (item["name"], buffers[item["name"]].detach().cpu().numpy().astype(np.float32, copy=False))
            for item in specs
        ]


class TrtMultiscaleBackboneBridge:
    def __init__(
        self,
        *,
        raw_dir: Path,
        precision_tag: str,
        engine_path: Path,
        trt_runner: Callable[[np.ndarray], list[tuple[str, np.ndarray]]] | None = None,
        expected_output_channels: tuple[int, int, int] | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.precision_tag = str(precision_tag)
        self.engine_path = Path(engine_path)
        self.trt_runner = trt_runner or TrtMultiscaleRunner(self.engine_path)
        self.expected_output_channels = (
            tuple(int(channel) for channel in expected_output_channels)
            if expected_output_channels is not None
            else None
        )
        if self.expected_output_channels is not None and (
            len(self.expected_output_channels) != 3
            or any(channel <= 0 for channel in self.expected_output_channels)
        ):
            raise ValueError("expected_output_channels must contain exactly three positive integers")
        self.reference_get_multiscale: Callable[[Any], tuple[Any, ...]] | None = None
        self.current_record_len: int | None = None
        self.call_index = 0
        self.activation_summaries: list[dict[str, Any]] = []
        self.output_summaries: list[dict[str, Any]] = []
        self.output_error_records: list[dict[str, Any]] = []

    def __call__(self, x: Any) -> tuple[Any, ...]:
        import torch

        if self.reference_get_multiscale is None:
            raise RuntimeError("reference_get_multiscale must be set before bridge use")
        record_len = int(self.current_record_len if self.current_record_len is not None else x.shape[0])
        call_index = self.call_index
        self.call_index += 1
        reference_outputs = tuple(self.reference_get_multiscale(x))
        original = x.detach().to(torch.float32).cpu().numpy()
        engine_input, batch_meta = prepare_spatial_features_for_engine(original, record_len=record_len, engine_batch=2)
        self.activation_summaries.append(
            {
                **summarize_numpy_array(
                    original,
                    tensor_name="spatial_features",
                ),
                "call_index": call_index,
                "batch_meta": batch_meta,
            }
        )
        runner_outputs = self.trt_runner(engine_input)
        if len(runner_outputs) != 3:
            raise RuntimeError(f"expected 3 multiscale outputs, got {len(runner_outputs)}")
        name_to_output = {name: output for name, output in runner_outputs}
        if len(name_to_output) != 3:
            raise RuntimeError("expected 3 uniquely named multiscale outputs")
        ordered_specs = sort_output_specs(
            [{"name": name, "shape": list(np.asarray(output).shape)} for name, output in runner_outputs]
        )
        if len(reference_outputs) != 3:
            raise RuntimeError(f"expected 3 reference multiscale outputs, got {len(reference_outputs)}")
        ordered_channels = tuple(int(item["shape"][1]) for item in ordered_specs)
        if self.expected_output_channels is not None and ordered_channels != self.expected_output_channels:
            raise RuntimeError(
                "ordered TensorRT output channel contract mismatch: "
                f"expected {list(self.expected_output_channels)}, got {list(ordered_channels)}"
            )
        validated_outputs: list[tuple[str, np.ndarray]] = []
        for output_index, item in enumerate(ordered_specs):
            name = str(item["name"])
            sliced = slice_output_to_record_len(name_to_output[name], record_len=record_len)
            reference_shape = tuple(int(dim) for dim in reference_outputs[output_index].shape)
            candidate_shape = tuple(int(dim) for dim in sliced.shape)
            if len(reference_shape) < 2 or len(candidate_shape) < 2:
                raise RuntimeError(
                    f"reference and TensorRT output {output_index} must expose NCHW shape"
                )
            if reference_shape[1] != candidate_shape[1]:
                raise RuntimeError(
                    f"reference output channel mismatch at index {output_index}: "
                    f"reference {reference_shape[1]}, TensorRT {candidate_shape[1]}"
                )
            if reference_shape != candidate_shape:
                raise RuntimeError(
                    f"reference output shape mismatch at index {output_index}: "
                    f"reference {list(reference_shape)}, TensorRT {list(candidate_shape)}"
                )
            validated_outputs.append((name, sliced))
        outputs: list[torch.Tensor] = []
        for output_index, (name, sliced) in enumerate(validated_outputs):
            raw_output = name_to_output[name]
            self.output_summaries.append(
                {
                    **summarize_numpy_array(sliced, tensor_name=str(name)),
                    "call_index": call_index,
                    "output_index": output_index,
                    "raw_worker_shape": [int(dim) for dim in np.asarray(raw_output).shape],
                    "record_len": record_len,
                }
            )
            self.output_error_records.append(
                {
                    **output_error_record(
                        str(name),
                        reference_outputs[output_index],
                        sliced,
                    ),
                    "call_index": call_index,
                    "output_index": output_index,
                }
            )
            outputs.append(torch.from_numpy(sliced.astype(np.float32)).to(device=x.device, dtype=torch.float32))
        return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="pyramid")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--precision-tag", required=True)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--full-ap-min-samples", type=int, default=1789)
    parser.add_argument("--expected-output-channels", type=parse_expected_output_channels, default=None)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> None:
    for attr in ("ckpt_dir", "engine", "heal_root", "raw_dir", "report_json"):
        value = getattr(args, attr, None)
        if not value:
            continue
        setattr(args, attr, str(Path(str(value)).resolve()))
    if not args.report_json:
        args.report_json = str((Path(args.raw_dir) / DEFAULT_REPORT_JSON).resolve())


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    heal_root = Path(args.heal_root)
    if str(heal_root) not in sys.path:
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
    checkpoint_path = best_checkpoint(ckpt_dir)
    protocol = protocol_payload(
        precision_tag=args.precision_tag,
        eval_range=args.eval_range,
        full_ap_min_samples=args.full_ap_min_samples,
        expected_output_channels=getattr(args, "expected_output_channels", None),
    )
    resume_epoch = checkpoint_epoch(checkpoint_path)

    opt = argparse.Namespace(
        model_dir=str(ckpt_dir),
        fusion_method="intermediate",
        save_vis_interval=10**9,
        save_npy=False,
        range=args.eval_range,
        no_score=True,
        note=SCHEMA,
    )
    hypes = yaml_utils.load_yaml(None, opt)
    if "heter" in hypes:
        x_max, y_max = [float(item) for item in str(args.eval_range).split(",")]
        new_range = [
            -x_max,
            -y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max,
            y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {"cav_lidar_range": new_range, "lidar_range": new_range, "gt_range": new_range})
        parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"), hypes["yaml_parser"])
        hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_utils.create_model(hypes)
    loaded_state_dict = torch.load(checkpoint_path, map_location="cpu")
    train_utils.check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)
    model = model.to(device).eval()
    model_dtype_summary = dtype_counts(model)

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    bridge = TrtMultiscaleBackboneBridge(
        raw_dir=raw_dir,
        precision_tag=args.precision_tag,
        engine_path=Path(args.engine),
        expected_output_channels=getattr(args, "expected_output_channels", None),
    )
    original_get_multiscale = model.pyramid_backbone.get_multiscale_feature
    bridge.reference_get_multiscale = original_get_multiscale
    model.pyramid_backbone.get_multiscale_feature = bridge

    result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}
    processed = 0
    failed = 0
    fallback_samples = 0
    pred_nonempty_count = 0
    pred_total_count = 0
    head_summary_items: list[dict[str, Any]] = []
    started = time.time()
    try:
        selected_loader = (
            itertools.islice(loader, int(args.num_samples))
            if args.num_samples is not None
            else loader
        )
        for sample_index, batch_data in enumerate(selected_loader):
            if batch_data is None:
                continue
            try:
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    ego = batch_data["ego"]
                    record_len = int(ego["record_len"][0].item())
                    if record_len > 2:
                        raise RuntimeError(f"record_len {record_len} exceeds fixed TensorRT engine batch 2")
                    bridge.current_record_len = record_len
                    output_dict: OrderedDict[str, Any] = OrderedDict()
                    output_dict["ego"] = model(ego)
                    head_summary = {key: tensor_summary(value, tensor_name=key) for key, value in output_dict["ego"].items()}
                    head_summary["sample_index"] = sample_index
                    output_dict["ego"] = cast_floating_tensors(output_dict["ego"], torch.float32)
                    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
                    for threshold in (0.3, 0.5, 0.7):
                        eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, threshold)
                    pred_count = int(pred_score.shape[0]) if pred_score is not None else 0
                    pred_total_count += pred_count
                    if pred_count > 0:
                        pred_nonempty_count += 1
                    if len(head_summary_items) < 1:
                        head_summary_items.append(head_summary)
                    processed += 1
            except Exception as exc:
                failed += 1
                write_json(
                    raw_dir / f"sample_{sample_index:06d}_blocker.json",
                    {
                        "schema": f"{SCHEMA}_sample_blocker_v1",
                        "status": "failed",
                        "sample_index": sample_index,
                        "failure_reason": f"{type(exc).__name__}:{exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                if processed == 0:
                    raise
    finally:
        model.pyramid_backbone.get_multiscale_feature = original_get_multiscale

    if processed == 0:
        raise RuntimeError("no non-empty batch processed for TensorRT multiscale AP bridge")

    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(raw_dir), f"{args.label}_{args.precision_tag}")
    config_path = ckpt_dir / "config.yaml"
    dataset_split_file = Path(str(hypes["validate_dir"]))
    output_error_summary = summarize_error_records(bridge.output_error_records)
    report = build_report(
        label=args.label,
        precision_tag=args.precision_tag,
        ckpt_dir=ckpt_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_epoch=resume_epoch,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        engine_path=Path(args.engine),
        engine_sha256=sha256_file(Path(args.engine)),
        dataset_split_file=dataset_split_file,
        dataset_split_sha256=sha256_file(dataset_split_file),
        protocol=protocol,
        processed_samples=processed,
        failed_samples=failed,
        fallback_samples=fallback_samples,
        ap30=float(ap30),
        ap50=float(ap50),
        ap70=float(ap70),
        pred_nonempty_count=pred_nonempty_count,
        pred_total_count=pred_total_count,
        model_dtype_counts=model_dtype_summary,
        output_error_summary=output_error_summary,
        elapsed_secs=time.time() - started,
    )
    report["resume_epoch"] = int(resume_epoch)
    report["head_output_samples"] = head_summary_items
    write_json(raw_dir / "activation_summary.json", {"items": bridge.activation_summaries})
    write_json(raw_dir / "multiscale_output_summary.json", {"items": bridge.output_summaries})
    write_json(raw_dir / "output_error_summary.json", {"items": bridge.output_error_records, "summary": output_error_summary})
    write_json(Path(args.report_json), report)
    return report


def classify_failure(exc: Exception) -> str:
    text = f"{type(exc).__name__}:{exc}".lower()
    if "record_len" in text or "batch" in text or "shape" in text:
        return "shape_or_batch_mismatch"
    if "tensorrt" in text or "engine" in text:
        return "tensorrt_execution_failure"
    if "checkpoint" in text or "config" in text:
        return "missing_checkpoint_or_config"
    return "stage3_trt_multiscale_ap_bridge_failure"


def main() -> int:
    args = parse_args()
    _resolve_paths(args)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "runner_command.json", {"command": sys.argv, "cwd": os.getcwd(), "gpu_id": args.gpu_id})
    try:
        report = run_bridge(args)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        blocker = {
            "schema": f"{SCHEMA}_blocker_v1",
            "status": "failed",
            "failure_type": classify_failure(exc),
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
            "raw_dir": str(raw_dir),
            "engine_path": str(args.engine),
            "ckpt_dir": str(args.ckpt_dir),
        }
        write_json(raw_dir / "stage3_trt_multiscale_ap_bridge_blocker.json", blocker)
        print(json.dumps(blocker, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
