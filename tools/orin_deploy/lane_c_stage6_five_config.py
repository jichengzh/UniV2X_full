#!/usr/bin/env python3
"""Freeze and materialize the Lane C Stage6 two-model, five-arm contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ARM_ORDER = (
    "original_default",
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "joint_shcosearch",
)


def _row(
    model: str,
    arm: str,
    widths: tuple[int, int, int],
    precision: str,
    runtime: str,
    source_dir: str,
    checkpoint: str,
    onnx: str,
    input_shape: tuple[int, int, int, int],
    h800_ap70: float,
    h800_latency_ms: float,
    h800_energy_j: float,
) -> dict[str, Any]:
    return {
        "id": f"{model}:{arm}",
        "model": model,
        "arm": arm,
        "widths": list(widths),
        "precision": precision,
        "runtime": runtime,
        "source_dir": source_dir,
        "checkpoint": checkpoint,
        "config": "config.yaml",
        "onnx": onnx,
        "expected_input_shape": list(input_shape),
        "expected_output_channels": list(widths),
        "h800_source": {
            "ap70": h800_ap70,
            "latency_ms": h800_latency_ms,
            "energy_j": h800_energy_j,
        },
    }


CONFIG_SPECS: tuple[dict[str, Any], ...] = (
    _row("pyramid", "original_default", (64, 128, 256), "fp32", "native_pytorch",
         "pyramid/064x128x256", "net_epoch_bestval_at23.pth",
         "pyramid_064x128x256_multiscale.onnx", (2, 64, 128, 256),
         0.6311, 3.2187, 1.1636),
    _row("pyramid", "compression_only", (24, 128, 64), "fp16", "tensorrt",
         "pyramid/024x128x064", "net_epoch_bestval_at1.pth",
         "pyramid_024x128x064_multiscale.onnx", (2, 64, 128, 256),
         0.5377, 0.5173, 0.1318),
    _row("pyramid", "schedule_only", (64, 128, 256), "fp32", "tensorrt",
         "pyramid/064x128x256", "net_epoch_bestval_at23.pth",
         "pyramid_064x128x256_multiscale.onnx", (2, 64, 128, 256),
         0.6314, 1.1038, 0.4305),
    _row("pyramid", "compress_then_tune", (32, 48, 128), "fp16", "tensorrt",
         "pyramid/032x048x128", "net_epoch_bestval_at1.pth",
         "pyramid_032x048x128_multiscale.onnx", (2, 64, 128, 256),
         0.5332, 1.6008, 0.4677),
    _row("pyramid", "joint_shcosearch", (16, 32, 64), "int8", "tensorrt",
         "pyramid/016x032x064", "net_epoch_bestval_at31.pth",
         "pyramid_016x032x064_multiscale.onnx", (2, 64, 128, 256),
         0.6189, 0.4524, 0.1112),
    _row("codriving", "original_default", (64, 128, 256), "fp32", "native_pytorch",
         "codriving/064x128x256", "net_epoch29.pth",
         "resnet_multiscale_64x128x256_final_fp32.onnx", (2, 64, 256, 512),
         0.3975, 1.8500, 0.6999),
    _row("codriving", "compression_only", (32, 64, 96), "fp16", "tensorrt",
         "codriving/032x064x096", "net_epoch_bestval_at11.pth",
         "resnet_multiscale_32x64x96_final_fp32.onnx", (2, 64, 256, 512),
         0.4190, 0.3579, 0.1013),
    _row("codriving", "schedule_only", (64, 128, 256), "fp32", "tensorrt",
         "codriving/064x128x256", "net_epoch29.pth",
         "resnet_multiscale_64x128x256_final_fp32.onnx", (2, 64, 256, 512),
         0.3973, 0.7686, 0.3510),
    _row("codriving", "compress_then_tune", (32, 80, 128), "fp16", "tensorrt",
         "codriving/032x080x128", "net_epoch_bestval_at5.pth",
         "resnet_multiscale_32x80x128_final_fp32.onnx", (2, 64, 256, 512),
         0.3984, 0.3892, 0.1391),
    _row("codriving", "joint_shcosearch", (16, 32, 64), "int8", "tensorrt",
         "codriving/016x032x064", "net_epoch_bestval_at9.pth",
         "resnet_multiscale_16x32x64_final_fp32.onnx", (2, 64, 256, 512),
         0.3565, 0.2976, 0.0708),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config_specs(specs: tuple[dict[str, Any], ...]) -> None:
    if len(specs) != 10:
        raise ValueError("contract must contain exactly ten executable rows")
    identifiers = [str(row["id"]) for row in specs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("configuration identifiers must be unique")
    for model in ("pyramid", "codriving"):
        rows = [row for row in specs if row["model"] == model]
        if tuple(row["arm"] for row in rows) != ARM_ORDER:
            raise ValueError(f"{model} arm order does not match Stage6 contract")
        for row in rows:
            if len(row["widths"]) != 3 or len(row["expected_input_shape"]) != 4:
                raise ValueError(f"invalid shape contract in {row['id']}")
            if row["runtime"] not in {"native_pytorch", "tensorrt"}:
                raise ValueError(f"invalid runtime in {row['id']}")


def build_source_manifest(source_root: Path) -> dict[str, Any]:
    validate_config_specs(CONFIG_SPECS)
    file_records: dict[str, dict[str, Any]] = {
        str(path.relative_to(source_root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(source_root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    rows: list[dict[str, Any]] = []
    for spec in CONFIG_SPECS:
        source_dir = source_root / spec["source_dir"]
        required = {
            "checkpoint": source_dir / spec["checkpoint"],
            "config": source_dir / spec["config"],
            "onnx": source_dir / spec["onnx"],
        }
        for path in required.values():
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = str(path.relative_to(source_root))
            if relative not in file_records:
                raise RuntimeError(f"source inventory omitted required file: {relative}")
        rows.append(
            {
                **spec,
                "source_files": {
                    key: str(path.relative_to(source_root))
                    for key, path in required.items()
                },
                "source_sha256": {
                    key: file_records[str(path.relative_to(source_root))]["sha256"]
                    for key, path in required.items()
                },
            }
        )
    return {
        "schema_version": "lane_c_stage6_five_config_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root.resolve()),
        "scope": "two_models_x_five_executable_arms",
        "excluded": {
            "tune_then_compress": "16/16 policy-transfer failures; no executable config",
            "f_cooper": "not present in source sections 11.3 or 14.3",
        },
        "latency_contract": {
            "batch": 2,
            "warmup": 20,
            "iters": 300,
            "repeat": 5,
            "timing": "CUDA_event",
            "scope": "multiscale_backbone_compute_no_data_transfer",
        },
        "ap_contract": "Orin full_1789; only replace multiscale backbone/resnet",
        "energy_contract": (
            "same latency window; named Orin tegrastats rail or "
            "missing_tegrastats_power_rails"
        ),
        "rows": rows,
        "files": file_records,
    }


def materialize_calibration(
    *,
    source_npz: Path,
    output_dir: Path,
    expected_input_shape: tuple[int, ...],
    source_layout: str,
) -> Path:
    with np.load(source_npz, allow_pickle=False) as archive:
        activations = np.asarray(archive["spatial_features"])
    if source_layout == "agent_rows":
        if tuple(activations.shape[1:]) != tuple(expected_input_shape[1:]):
            raise ValueError("agent-row calibration tail does not match input contract")
        rows = []
        for index in range(0, len(activations), 2):
            second = min(index + 1, len(activations) - 1)
            rows.append(np.stack((activations[index], activations[second]), axis=0))
        odd_tail_policy = "repeat_last_activation"
    elif source_layout == "batch2_rows":
        if tuple(activations.shape[1:]) != tuple(expected_input_shape):
            raise ValueError("batch2 calibration rows do not match input contract")
        rows = [activations[index] for index in range(len(activations))]
        odd_tail_policy = "not_applicable"
    else:
        raise ValueError(f"unsupported source layout: {source_layout}")
    if not rows:
        raise ValueError("calibration source contains no activation rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(rows):
        target = output_dir / f"batch2_{index:03d}.npy"
        np.save(target, np.ascontiguousarray(row, dtype=np.float32))
        records.append(
            {
                "filename": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    manifest = {
        "schema_version": "lane_c_shared_calibration_manifest_v1",
        "source": {
            "path": str(source_npz),
            "sha256": sha256_file(source_npz),
            "layout": source_layout,
            "odd_tail_policy": odd_tail_policy,
        },
        "calibration": {
            "file_count": len(records),
            "batch_size": 2,
            "input_shape": list(expected_input_shape),
            "dtype": "float32",
            "files": records,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(item) for item in value.split(","))
    if not shape or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError("shape must contain positive integers")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--source-root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--sha-sums", type=Path)
    calibration = commands.add_parser("calibration")
    calibration.add_argument("--source-npz", type=Path, required=True)
    calibration.add_argument("--output-dir", type=Path, required=True)
    calibration.add_argument("--expected-input-shape", type=parse_shape, required=True)
    calibration.add_argument(
        "--source-layout",
        choices=("agent_rows", "batch2_rows"),
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "manifest":
        payload = build_source_manifest(args.source_root.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if args.sha_sums is not None:
            args.sha_sums.parent.mkdir(parents=True, exist_ok=True)
            args.sha_sums.write_text(
                "".join(
                    f"{record['sha256']}  {relative}\n"
                    for relative, record in sorted(payload["files"].items())
                ),
                encoding="utf-8",
            )
        print(args.output)
        return 0
    manifest = materialize_calibration(
        source_npz=args.source_npz.resolve(),
        output_dir=args.output_dir.resolve(),
        expected_input_shape=args.expected_input_shape,
        source_layout=args.source_layout,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
