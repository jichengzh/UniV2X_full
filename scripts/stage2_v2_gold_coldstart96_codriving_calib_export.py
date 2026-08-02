#!/usr/bin/env python3
"""Export real CoDriving collab calibration tensors for v2 gold INT8 builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REMOTE_REPO = Path("/exdata/jichengzhi/V2Xverse_pyramid")
SPATIAL_SHAPE = (2, 64, 256, 512)
TMAT_SHAPE = (1, 2, 2, 4, 4)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_calib_arrays(spatial: np.ndarray, tmat: np.ndarray) -> dict[str, list[int]]:
    if spatial.ndim != 5 or tuple(spatial.shape[1:]) != SPATIAL_SHAPE:
        raise ValueError(f"spatial_features must have shape (N, {SPATIAL_SHAPE}), got {spatial.shape}")
    if tmat.ndim != 6 or tuple(tmat.shape[1:]) != TMAT_SHAPE:
        raise ValueError(f"pairwise_t_matrix must have shape (N, {TMAT_SHAPE}), got {tmat.shape}")
    if spatial.shape[0] != tmat.shape[0]:
        raise ValueError(
            "spatial_features and pairwise_t_matrix must have the same N, "
            f"got {spatial.shape[0]} and {tmat.shape[0]}"
        )
    if spatial.shape[0] <= 0:
        raise ValueError("calibration data must contain at least one sample")
    return {
        "spatial_features": list(spatial.shape),
        "pairwise_t_matrix": list(tmat.shape),
    }


def build_report(
    *,
    width: str,
    model_dir: Path,
    output: Path,
    requested_samples: int,
    collected_samples: int,
    skipped_non_2_agent: int,
    calibration_split: str,
    split_source_file: Path,
    output_sha256: str,
    shapes: dict[str, list[int]],
    elapsed_secs: float,
) -> dict[str, Any]:
    return {
        "schema": "v2_gold_coldstart_96_codriving_calib_export_v1",
        "created_at_utc": utc_now(),
        "width": width,
        "model_dir": str(model_dir),
        "output": str(output),
        "requested_samples": int(requested_samples),
        "collected_samples": int(collected_samples),
        "skipped_non_2_agent": int(skipped_non_2_agent),
        "calibration_split": calibration_split,
        "split_source_file": str(split_source_file),
        "output_sha256": output_sha256,
        "shapes": shapes,
        "elapsed_secs": float(elapsed_secs),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_calibration_hypes(hypes: dict[str, Any], split: str) -> dict[str, Any]:
    if split == "val":
        return dict(hypes)
    if split == "train":
        root_dir = hypes.get("root_dir")
        if not root_dir:
            raise ValueError("train calibration requires root_dir")
        return {**hypes, "validate_dir": root_dir}
    raise ValueError(f"unsupported calibration split: {split}")


def _storage_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"unsupported storage dtype: {name}")


def collect_calib_data(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    repo_root = Path(args.repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))

    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset

    started = time.time()
    raw_hypes = load_yaml(str(args.model_dir / "config.yaml"))
    split_source_file = Path(raw_hypes["root_dir"] if args.split == "train" else raw_hypes["validate_dir"])
    hypes = select_calibration_hypes(raw_hypes, args.split)
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.model_dir), model)
    model = model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    spatial_list: list[np.ndarray] = []
    tmat_list: list[np.ndarray] = []
    skipped_non_2_agent = 0
    storage_dtype = _storage_dtype(args.storage_dtype)
    with torch.inference_mode():
        for batch_data in loader:
            if len(spatial_list) >= args.n_samples:
                break
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            record_len = int(ego["record_len"][0].item())
            if record_len != 2:
                skipped_non_2_agent += 1
                continue
            voxel_batch = {
                "voxel_features": ego["processed_lidar"]["voxel_features"],
                "voxel_coords": ego["processed_lidar"]["voxel_coords"],
                "voxel_num_points": ego["processed_lidar"]["voxel_num_points"],
                "record_len": ego["record_len"],
            }
            voxel_batch = model.pillar_vfe(voxel_batch)
            voxel_batch = model.scatter(voxel_batch)
            spatial = voxel_batch["spatial_features"].detach().cpu().float().numpy().astype(storage_dtype, copy=False)
            tmat = ego["pairwise_t_matrix"].detach().cpu().float().numpy()
            spatial_list.append(spatial)
            tmat_list.append(tmat)
            if len(spatial_list) % args.progress_every == 0:
                print(
                    f"[calib] {args.width} collected={len(spatial_list)}/{args.n_samples} "
                    f"skipped_non_2_agent={skipped_non_2_agent}",
                    flush=True,
                )

    spatial_arr = np.stack(spatial_list, axis=0)
    tmat_arr = np.stack(tmat_list, axis=0)
    shapes = validate_calib_arrays(spatial_arr, tmat_arr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, spatial_features=spatial_arr, pairwise_t_matrix=tmat_arr)
    output_sha256 = sha256_file(args.output)
    report = build_report(
        width=args.width,
        model_dir=args.model_dir,
        output=args.output,
        requested_samples=args.n_samples,
        collected_samples=spatial_arr.shape[0],
        skipped_non_2_agent=skipped_non_2_agent,
        calibration_split=args.split,
        split_source_file=split_source_file,
        output_sha256=output_sha256,
        shapes=shapes,
        elapsed_secs=time.time() - started,
    )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REMOTE_REPO)
    parser.add_argument("--width", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument("--storage-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()
    if args.n_samples <= 0:
        parser.error("--n-samples must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main() -> int:
    collect_calib_data(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
