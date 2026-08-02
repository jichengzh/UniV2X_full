#!/usr/bin/env python3
"""Create and verify the Lane C shared calibration byte contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .lane_c_backbone_parity_runner import (
        EXPECTED_CALIBRATION_FILE_COUNT,
        EXPECTED_INPUT_TAIL,
        sha256_file,
        validate_calibration_manifest,
    )
except ImportError:
    from lane_c_backbone_parity_runner import (
        EXPECTED_CALIBRATION_FILE_COUNT,
        EXPECTED_INPUT_TAIL,
        sha256_file,
        validate_calibration_manifest,
    )


SCHEMA = "lane_c_shared_calibration_manifest_v1"


def calibration_payload_id(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: int(item["index"])):
        record = (
            f"{row['filename']}\0{int(row['bytes'])}\0{row['sha256']}\n"
        )
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def _validate_sha256(value: str, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def build_canonical_manifest(
    *,
    calibration_dir: Path,
    checkpoint_sha256: str,
    onnx_sha256: str,
    expected_shape: tuple[int, ...] = EXPECTED_INPUT_TAIL,
) -> dict[str, Any]:
    calibration_dir = Path(calibration_dir)
    files = sorted(calibration_dir.glob("batch2_*.npy"))
    if len(files) != EXPECTED_CALIBRATION_FILE_COUNT:
        raise ValueError(
            f"calibration directory must contain exactly 15 batch2 NPY files, got {len(files)}"
        )
    expected_names = [
        f"batch2_{index:03d}.npy"
        for index in range(EXPECTED_CALIBRATION_FILE_COUNT)
    ]
    if [path.name for path in files] != expected_names:
        raise ValueError("calibration filenames must be contiguous batch2_000..014")

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files):
        array = np.load(path, allow_pickle=False)
        if tuple(array.shape) != tuple(expected_shape):
            raise ValueError(f"invalid shape in {path.name}: {list(array.shape)}")
        if array.dtype != np.dtype(np.float32):
            raise ValueError(f"invalid dtype in {path.name}: {array.dtype}")
        if not array.flags.c_contiguous:
            raise ValueError(f"calibration array is not C-contiguous: {path.name}")
        rows.append(
            {
                "index": index,
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "dtype": str(array.dtype),
                "shape": [int(value) for value in array.shape],
                "c_contiguous": True,
                "fortran_order": False,
            }
        )
    payload_id = calibration_payload_id(rows)
    return {
        "schema_version": SCHEMA,
        "scope": "pyramid_get_multiscale_feature_16x32x64_only",
        "checkpoint_sha256": _validate_sha256(
            checkpoint_sha256, "checkpoint_sha256"
        ),
        "onnx_sha256": _validate_sha256(onnx_sha256, "onnx_sha256"),
        "calibration": {
            "source_lineage": "rebuilt_exact_command_gpu7",
            "file_count": len(rows),
            "batch_size": 2,
            "sample_count": 30,
            "dtype": "float32",
            "shape_per_file": [int(value) for value in expected_shape],
            "sample_order": expected_names,
            "odd_tail_policy": "not_applicable_exact_30_samples",
            "payload_id": payload_id,
            "files": rows,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gpu_identity() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        rows = [
            [field.strip() for field in line.split(",")]
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        return {"query": command, "rows": rows}
    except (OSError, subprocess.SubprocessError) as error:
        return {"query": command, "error": f"{type(error).__name__}:{error}"}


def build_receipt(*, manifest_path: Path, calibration_dir: Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    audit = validate_calibration_manifest(
        manifest_path=manifest_path,
        calibration_dir=calibration_dir,
    )
    observed_payload_id = calibration_payload_id(
        [
            {**row, "index": index}
            for index, row in enumerate(audit["files"])
        ]
    )
    expected_payload_id = str(manifest["calibration"]["payload_id"])
    if observed_payload_id != expected_payload_id:
        raise ValueError("calibration payload ID mismatch")
    try:
        import tensorrt as trt

        tensorrt_version = trt.__version__
    except Exception as error:
        tensorrt_version = f"unavailable:{type(error).__name__}"
    return {
        "schema_version": "lane_c_calibration_receive_receipt_v1",
        "status": "byte_identical_15_of_15",
        "scope": manifest["scope"],
        "host": {
            "hostname": platform.node(),
            "architecture": platform.machine(),
            "gpu_identity": _gpu_identity(),
            "tensorrt_version": tensorrt_version,
        },
        "manifest_sha256": audit["manifest_sha256"],
        "payload_id": observed_payload_id,
        "received_directory": str(Path(calibration_dir).resolve()),
        "expected_file_count": EXPECTED_CALIBRATION_FILE_COUNT,
        "matched_file_count": audit["verified_file_count"],
        "missing_files": [],
        "extra_files": [],
        "files": audit["files"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    create.add_argument("--calibration-dir", type=Path, required=True)
    create.add_argument("--checkpoint-sha256", required=True)
    create.add_argument("--onnx-sha256", required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--sha-output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--calibration-dir", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "create":
        manifest = build_canonical_manifest(
            calibration_dir=args.calibration_dir,
            checkpoint_sha256=args.checkpoint_sha256,
            onnx_sha256=args.onnx_sha256,
        )
        write_json(args.output, manifest)
        args.sha_output.parent.mkdir(parents=True, exist_ok=True)
        args.sha_output.write_text(
            f"{sha256_file(args.output)}  {args.output.name}\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    receipt = build_receipt(
        manifest_path=args.manifest,
        calibration_dir=args.calibration_dir,
    )
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
