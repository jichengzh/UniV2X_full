#!/usr/bin/env python3
"""Export checkpoint-consistent INT8 calibration inputs at pyramid_backbone boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, checkpoint_epoch  # noqa: E402


SCHEMA = "stage3_pyramid_calibration_export_v3"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
SHA256_LENGTH = 64
OUTPUT_DTYPES = ("float16", "float32")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an exact JSON integer")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be a lowercase SHA256 hex digest")
    return text


def parse_eval_range(value: str) -> tuple[float, float]:
    parts = [item.strip() for item in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError("eval range must be 'x,y'")
    try:
        x_max = float(parts[0])
        y_max = float(parts[1])
    except ValueError as exc:
        raise ValueError("eval range must contain numeric values") from exc
    if not math.isfinite(x_max) or not math.isfinite(y_max) or x_max <= 0.0 or y_max <= 0.0:
        raise ValueError("eval range values must be positive finite numbers")
    return x_max, y_max


def select_frozen_train_hypes(raw_hypes: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    root_dir = str(raw_hypes.get("root_dir") or "").strip()
    if not root_dir:
        raise ValueError("train calibration requires root_dir in config")
    selected = dict(raw_hypes)
    selected["validate_dir"] = root_dir
    return selected, Path(root_dir)


def apply_eval_range_to_hypes(hypes: dict[str, Any], eval_range: str) -> dict[str, Any]:
    if "heter" not in hypes:
        return dict(hypes)
    x_max, y_max = parse_eval_range(eval_range)
    anchor_args = (((hypes.get("postprocess") or {}).get("anchor_args") or {}).get("cav_lidar_range"))
    if not isinstance(anchor_args, (list, tuple)) or len(anchor_args) != 6:
        raise ValueError("heter hypes missing postprocess.anchor_args.cav_lidar_range")
    new_range = [-x_max, -y_max, anchor_args[2], x_max, y_max, anchor_args[5]]
    updated = dict(hypes)
    updated["cav_lidar_range"] = list(new_range)
    updated["lidar_range"] = list(new_range)
    updated["gt_range"] = list(new_range)
    yaml_parser_name = str(updated.get("yaml_parser") or "").strip()
    if not yaml_parser_name:
        raise ValueError("heter hypes missing yaml_parser")
    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = getattr(yaml_utils_lib, yaml_parser_name)
    parsed = parser_func(updated)
    parsed["validate_dir"] = str(hypes["validate_dir"])
    return parsed


def normalize_output_dtype(value: Any) -> str:
    dtype = str(value or "")
    if dtype not in OUTPUT_DTYPES:
        raise ValueError(f"output_dtype must be one of {OUTPUT_DTYPES}, got {dtype!r}")
    return dtype


def validate_spatial_features(
    spatial_features: np.ndarray,
    *,
    expected_dtype: str | None = None,
) -> list[int]:
    array = np.asarray(spatial_features)
    if array.ndim != 4:
        raise ValueError(f"spatial_features must have shape [N,C,H,W], got {list(array.shape)}")
    if array.shape[0] <= 0:
        raise ValueError("spatial_features must contain at least one agent instance")
    if any(int(dim) <= 0 for dim in array.shape[1:]):
        raise ValueError(f"spatial_features has non-positive dimensions: {list(array.shape)}")
    actual_dtype = array.dtype.name
    if actual_dtype not in OUTPUT_DTYPES:
        raise ValueError(f"spatial_features actual dtype {actual_dtype} is unsupported")
    if expected_dtype is not None:
        normalized_expected = normalize_output_dtype(expected_dtype)
        if actual_dtype != normalized_expected:
            raise ValueError(
                f"spatial_features actual dtype {actual_dtype} does not match expected {normalized_expected}"
            )
    return [int(dim) for dim in array.shape]


def validate_scene_capture(
    scene_features: np.ndarray,
    *,
    record_len: int,
    scene_index: int,
    expected_dtype: str | None = None,
) -> list[int]:
    shape = validate_spatial_features(scene_features, expected_dtype=expected_dtype)
    if int(record_len) <= 0:
        raise ValueError(f"scene {scene_index} record_len must be positive")
    if shape[0] != int(record_len):
        raise ValueError(
            f"scene {scene_index} record_len {int(record_len)} does not match captured agent axis {shape[0]}"
        )
    return shape


def build_summary(
    *,
    ckpt_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    config_path: Path,
    config_sha256: str,
    heal_root: Path,
    split_source_file: Path,
    split_source_sha256: str,
    output_npz: Path,
    output_sha256: str,
    summary_json: Path,
    num_scenes_requested: int,
    scene_sample_count: int,
    agent_instance_count: int,
    scene_record_lens: list[int],
    scene_record_lens_sha256: str,
    spatial_shape: list[int],
    eval_range: str,
    gpu_id: int,
    elapsed_secs: float,
    output_dtype: str = "float16",
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "calibration_split": "train",
        "ckpt_dir": str(ckpt_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_sha256": checkpoint_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "heal_root": str(heal_root),
        "split_source_file": str(split_source_file),
        "split_source_sha256": split_source_sha256,
        "output_npz": str(output_npz),
        "output_sha256": output_sha256,
        "summary_json": str(summary_json),
        "num_scenes_requested": int(num_scenes_requested),
        "scene_sample_count": int(scene_sample_count),
        "agent_instance_count": int(agent_instance_count),
        "scene_record_lens": [int(value) for value in scene_record_lens],
        "scene_record_lens_sha256": scene_record_lens_sha256,
        "spatial_features_shape": [int(value) for value in spatial_shape],
        "output_dtype": normalize_output_dtype(output_dtype),
        "eval_range": str(eval_range),
        "gpu_id": int(gpu_id),
        "elapsed_secs": float(elapsed_secs),
    }


def validate_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise ValueError("summary must be a mapping")
    if summary.get("schema") != SCHEMA:
        raise ValueError("unexpected summary schema")
    if summary.get("calibration_split") != "train":
        raise ValueError("summary must bind a train split export")
    normalize_output_dtype(summary.get("output_dtype"))
    scene_sample_count = _exact_int(summary.get("scene_sample_count"), "scene_sample_count")
    num_scenes_requested = _exact_int(summary.get("num_scenes_requested"), "num_scenes_requested")
    agent_instance_count = _exact_int(summary.get("agent_instance_count"), "agent_instance_count")
    checkpoint_epoch = _exact_int(summary.get("checkpoint_epoch"), "checkpoint_epoch")
    _exact_int(summary.get("gpu_id"), "gpu_id")
    if scene_sample_count <= 0:
        raise ValueError("scene_sample_count must be positive")
    if num_scenes_requested <= 0:
        raise ValueError("num_scenes_requested must be positive")
    if scene_sample_count > num_scenes_requested:
        raise ValueError("scene_sample_count cannot exceed num_scenes_requested")
    if checkpoint_epoch < 0:
        raise ValueError("checkpoint_epoch must be non-negative")
    if agent_instance_count <= 0:
        raise ValueError("agent_instance_count must be positive")
    for field in ("checkpoint_sha256", "config_sha256", "split_source_sha256", "output_sha256", "scene_record_lens_sha256"):
        _require_sha256(summary.get(field), field)
    scene_record_lens = summary.get("scene_record_lens")
    if not isinstance(scene_record_lens, list) or len(scene_record_lens) != scene_sample_count:
        raise ValueError("scene_record_lens must match scene_sample_count")
    normalized_record_lens = [_exact_int(value, f"scene_record_lens[{index}]") for index, value in enumerate(scene_record_lens)]
    if any(value <= 0 for value in normalized_record_lens):
        raise ValueError("scene_record_lens must contain positive integers")
    if sum(normalized_record_lens) != agent_instance_count:
        raise ValueError("scene_record_lens sum must equal agent_instance_count")
    if sha256_json(normalized_record_lens) != summary["scene_record_lens_sha256"]:
        raise ValueError("scene_record_lens_sha256 does not match scene_record_lens")
    spatial_shape_raw = summary.get("spatial_features_shape")
    if not isinstance(spatial_shape_raw, list) or len(spatial_shape_raw) != 4:
        raise ValueError("spatial_features_shape must be a 4-element list")
    spatial_shape = [_exact_int(value, f"spatial_features_shape[{index}]") for index, value in enumerate(spatial_shape_raw)]
    if spatial_shape[0] != agent_instance_count:
        raise ValueError("spatial_features_shape[0] must equal agent_instance_count")
    if any(value <= 0 for value in spatial_shape[1:]):
        raise ValueError("spatial_features_shape tail must be positive")
    parse_eval_range(str(summary.get("eval_range") or ""))
    for field in ("ckpt_dir", "checkpoint_path", "config_path", "heal_root", "split_source_file", "output_npz", "summary_json"):
        if not str(summary.get(field) or ""):
            raise ValueError(f"{field} is required")
    elapsed_secs = float(summary.get("elapsed_secs") or 0.0)
    if not math.isfinite(elapsed_secs) or elapsed_secs < 0.0:
        raise ValueError("elapsed_secs must be a non-negative finite float")
    return dict(summary)


def write_export_artifacts(
    *,
    spatial_features: np.ndarray,
    output_npz: Path,
    summary_json: Path,
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    output_dtype = normalize_output_dtype(summary_payload.get("output_dtype"))
    actual_shape = validate_spatial_features(spatial_features, expected_dtype=output_dtype)
    declared_shape = summary_payload.get("spatial_features_shape")
    if actual_shape != declared_shape:
        raise ValueError(
            f"spatial_features actual shape {actual_shape} does not match declared shape {declared_shape}"
        )
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, spatial_features=np.asarray(spatial_features))
    updated_summary = dict(summary_payload)
    updated_summary["output_npz"] = str(output_npz)
    updated_summary["summary_json"] = str(summary_json)
    updated_summary["output_sha256"] = sha256_file(output_npz)
    validated = validate_summary(updated_summary)
    summary_json.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validated


class PyramidInputCapture:
    def __init__(self, model: Any) -> None:
        self.model = model
        self._originals: list[tuple[Any, str, Any]] = []
        self._current_scene_capture: np.ndarray | None = None

    def __enter__(self) -> "PyramidInputCapture":
        backbone = getattr(self.model, "pyramid_backbone", None)
        if backbone is None:
            raise ValueError("model is missing pyramid_backbone")
        for method_name in ("forward_collab", "forward", "get_multiscale_feature"):
            if not hasattr(backbone, method_name):
                continue
            original = getattr(backbone, method_name)
            self._originals.append((backbone, method_name, original))
            setattr(backbone, method_name, MethodType(self._wrap_method(method_name, original), backbone))
        if not self._originals:
            raise ValueError("pyramid_backbone exposes no captureable entrypoint")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for owner, method_name, original in reversed(self._originals):
            setattr(owner, method_name, original)
        self._originals.clear()

    def start_scene(self) -> None:
        self._current_scene_capture = None

    def finish_scene(self) -> np.ndarray:
        if self._current_scene_capture is None:
            raise RuntimeError("failed to capture pyramid_backbone spatial_features for scene")
        return self._current_scene_capture

    def _wrap_method(self, method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(backbone_self: Any, *args: Any, **kwargs: Any) -> Any:
            self._maybe_capture(method_name, args, kwargs)
            return original(*args, **kwargs)

        return wrapped

    def _maybe_capture(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if self._current_scene_capture is not None:
            return
        candidate: Any = None
        if method_name == "forward_collab":
            if args:
                candidate = args[0]
        elif method_name == "forward":
            if args:
                candidate = args[0]
                if isinstance(candidate, dict):
                    candidate = candidate.get("spatial_features")
        elif method_name == "get_multiscale_feature":
            if args:
                candidate = args[0]
        if candidate is None:
            candidate = kwargs.get("spatial_features")
        self._current_scene_capture = tensor_to_numpy(candidate)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("expected tensor-like spatial_features, got None")
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    raise TypeError(f"unsupported spatial_features type: {type(value)!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--heal-root", type=Path, default=Path(DEFAULT_HEAL_ROOT))
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16, help="Number of frozen train scenes to export")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--eval-range", default="102.4,102.4")
    parser.add_argument("--output-dtype", choices=OUTPUT_DTYPES, default="float16")
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    parse_eval_range(args.eval_range)
    return args


def load_explicit_checkpoint(model: Any, checkpoint_path: Path) -> tuple[int, Any]:
    import torch

    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"explicit checkpoint not found: {path}")
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state must be dict: {path}")
    model.load_state_dict(state, strict=False)
    return int(checkpoint_epoch(path)), model


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    ckpt_dir = Path(args.ckpt_dir).resolve()
    heal_root = Path(args.heal_root).resolve()
    output_npz = Path(args.output_npz).resolve()
    summary_json = Path(args.summary_json).resolve()
    output_dtype = normalize_output_dtype(getattr(args, "output_dtype", "float16"))
    config_path = ckpt_dir / "config.yaml"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config not found: {config_path}")
    if not heal_root.is_dir():
        raise FileNotFoundError(f"HEAL root not found: {heal_root}")
    checkpoint_path = (
        Path(args.checkpoint_path).resolve()
        if args.checkpoint_path is not None
        else best_checkpoint(ckpt_dir)
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"resolved checkpoint not found: {checkpoint_path}")

    started = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if str(heal_root) not in sys.path:
        sys.path.insert(0, str(heal_root))
    os.chdir(str(heal_root))

    import torch
    from torch.utils.data import DataLoader
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    raw_hypes = yaml_utils.load_yaml(str(config_path))
    selected_hypes, split_source_file = select_frozen_train_hypes(raw_hypes)
    if not split_source_file.is_file():
        raise FileNotFoundError(f"frozen train split file not found: {split_source_file}")
    hypes = apply_eval_range_to_hypes(selected_hypes, args.eval_range)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for checkpoint-consistent calibration export")
    model = train_utils.create_model(hypes)
    if args.checkpoint_path is not None:
        resume_epoch, model = load_explicit_checkpoint(model, checkpoint_path)
    else:
        resume_epoch, model = train_utils.load_saved_model(str(ckpt_dir), model)
    model = model.to(device).eval()
    checkpoint_epoch_value = checkpoint_epoch(checkpoint_path)
    if int(resume_epoch) != checkpoint_epoch_value:
        raise RuntimeError(
            f"loaded checkpoint epoch {int(resume_epoch)} does not match resolved checkpoint {checkpoint_path.name}"
        )
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    captured_scenes: list[np.ndarray] = []
    scene_record_lens: list[int] = []
    with torch.inference_mode(), PyramidInputCapture(model) as capture:
        for batch_data in loader:
            if len(captured_scenes) >= int(args.num_samples):
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            ego = batch_data["ego"]
            record_len_tensor = ego.get("record_len")
            if record_len_tensor is None:
                raise ValueError("batch is missing ego.record_len")
            record_len = int(record_len_tensor[0].item())
            if record_len <= 0:
                raise ValueError(f"invalid record_len: {record_len}")
            capture.start_scene()
            _ = model(ego)
            scene_features = np.asarray(capture.finish_scene(), dtype=np.dtype(output_dtype))
            validate_scene_capture(
                scene_features,
                record_len=record_len,
                scene_index=len(captured_scenes),
                expected_dtype=output_dtype,
            )
            captured_scenes.append(scene_features)
            scene_record_lens.append(record_len)

    if not captured_scenes:
        raise RuntimeError("no frozen train scenes produced a pyramid_backbone capture")
    if len(captured_scenes) != int(args.num_samples):
        raise RuntimeError(
            f"requested {int(args.num_samples)} scenes but captured only {len(captured_scenes)} frozen train scenes"
        )
    spatial_features = np.concatenate(captured_scenes, axis=0)
    spatial_shape = validate_spatial_features(spatial_features, expected_dtype=output_dtype)
    scene_record_lens_sha256 = sha256_json(scene_record_lens)
    summary_payload = build_summary(
        ckpt_dir=ckpt_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_epoch=checkpoint_epoch_value,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        heal_root=heal_root,
        split_source_file=split_source_file,
        split_source_sha256=sha256_file(split_source_file),
        output_npz=output_npz,
        output_sha256="0" * SHA256_LENGTH,
        summary_json=summary_json,
        num_scenes_requested=int(args.num_samples),
        scene_sample_count=len(captured_scenes),
        agent_instance_count=int(spatial_shape[0]),
        scene_record_lens=scene_record_lens,
        scene_record_lens_sha256=scene_record_lens_sha256,
        spatial_shape=spatial_shape,
        eval_range=str(args.eval_range),
        gpu_id=int(args.gpu_id),
        elapsed_secs=time.time() - started,
        output_dtype=output_dtype,
    )
    return write_export_artifacts(
        spatial_features=spatial_features,
        output_npz=output_npz,
        summary_json=summary_json,
        summary_payload=summary_payload,
    )


def main() -> int:
    summary = run_export(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
