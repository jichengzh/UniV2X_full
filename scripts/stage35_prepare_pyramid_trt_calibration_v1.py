#!/usr/bin/env python3
"""Convert frozen Pyramid activations into static batch-2 TRT calibration tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ACTIVATION_CHW = (64, 128, 256)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_batch2_calibration(
    spatial_features: np.ndarray,
    *,
    expected_chw: tuple[int, int, int] = EXPECTED_ACTIVATION_CHW,
) -> np.ndarray:
    source = np.asarray(spatial_features)
    if source.ndim != 4 or source.shape[0] <= 0:
        raise ValueError(f"expected non-empty [N,C,H,W], got {list(source.shape)}")
    if tuple(source.shape[1:]) != tuple(expected_chw):
        raise ValueError(
            f"expected activation CHW {list(expected_chw)}, got {list(source.shape[1:])}"
        )
    converted = source.astype(np.float32, copy=True)
    if converted.shape[0] % 2:
        converted = np.concatenate((converted, converted[-1:]), axis=0)
    return converted.reshape((-1, 2, *converted.shape[1:]))


def write_trt_calibration(
    source_npz: Path,
    output_dir: Path,
    *,
    expected_chw: tuple[int, int, int] = EXPECTED_ACTIVATION_CHW,
) -> dict[str, Any]:
    source_npz = Path(source_npz).resolve()
    output_dir = Path(output_dir).resolve()
    with np.load(source_npz, allow_pickle=False) as payload:
        if "spatial_features" not in payload.files:
            raise ValueError(f"missing spatial_features in {source_npz}")
        source = np.asarray(payload["spatial_features"])
    batches = build_batch2_calibration(source, expected_chw=expected_chw)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    sample_paths: list[Path] = []
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup")
    try:
        staged_paths: list[Path] = []
        for index, batch in enumerate(batches):
            staged_path = staging_dir / f"batch2_{index:03d}.npy"
            np.save(staged_path, batch, allow_pickle=False)
            staged_paths.append(staged_path)
            sample_paths.append(output_dir / staged_path.name)
        report = {
            "schema_version": "stage35_pyramid_trt_calibration_v1",
            "source_npz": str(source_npz),
            "source_sha256": _sha256_file(source_npz),
            "input_count": int(source.shape[0]),
            "batch_count": int(batches.shape[0]),
            "sample_shape": [int(value) for value in batches.shape[1:]],
            "sample_dtype": str(batches.dtype),
            "sample_files": [str(path) for path in sample_paths],
            "sample_sha256": {
                staged.name: _sha256_file(staged) for staged in staged_paths
            },
            "odd_tail_policy": "repeat_last_activation",
        }
        (staging_dir / "trt_npy_manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if output_dir.exists():
            output_dir.rename(backup_dir)
        try:
            staging_dir.rename(output_dir)
        except Exception:
            if backup_dir.exists() and not output_dir.exists():
                backup_dir.rename(output_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        return report
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_trt_calibration(args.source_npz, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
