#!/usr/bin/env python3
"""Export a strict F-Cooper dense source from a recovered checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fcooper_materialize_source_v1 import (  # noqa: E402
    FCooperDenseEncoder,
    parse_width,
    sha256_file,
)


def _load_hypes(config: Path) -> dict[str, Any]:
    from opencood.hypes_yaml import yaml_utils

    return yaml_utils.load_yaml(str(config), SimpleNamespace(model_dir=None))


def _create_model(hypes: Mapping[str, Any]) -> nn.Module:
    from opencood.tools import train_utils

    return train_utils.create_model(dict(hypes))


def _validate_training_report(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.formal:
        return None
    if args.original_unpruned:
        return {
            "initialization_policy": "original_unpruned_checkpoint",
            "status": "success",
        }
    if args.training_report is None or not args.training_report.is_file():
        raise ValueError("formal export requires a recovery training report")
    report = json.loads(args.training_report.read_text())
    if (
        report.get("schema_version") != "fcooper_recovery_training_report_v2"
        or report.get("status") != "success"
        or report.get("initialization_policy") != "scanner_dependency_l1_v2"
        or int(report.get("epochs_completed") or 0) < 4
    ):
        raise ValueError("recovery training report does not satisfy formal export")
    if report.get("recovered_checkpoint_sha256") != sha256_file(args.checkpoint):
        raise ValueError("formal checkpoint SHA does not match recovery report")
    return report


def export(args: argparse.Namespace) -> dict[str, Any]:
    width = parse_width(args.width)
    training = _validate_training_report(args)
    hypes = _load_hypes(args.config)
    model = _create_model(hypes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    dense = FCooperDenseEncoder(model).eval()
    sample = torch.zeros(*args.input_shape, dtype=torch.float32)
    with torch.no_grad():
        output = dense(sample)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        dense,
        sample,
        str(args.onnx),
        input_names=["spatial_features"],
        output_names=["encoded_features"],
        opset_version=17,
        do_constant_folding=True,
    )
    import onnx

    graph = onnx.load(str(args.onnx))
    onnx.checker.check_model(graph)
    report = {
        "schema_version": "fcooper_source_export_v2",
        "status": "success",
        "source_kind": "formal_recovered" if args.formal else "capability_probe_only",
        "formal_measurement_eligible": bool(args.formal),
        "width": list(width),
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "numerical_finite": bool(torch.isfinite(output).all()),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "onnx_path": str(args.onnx.resolve()),
        "onnx_sha256": sha256_file(args.onnx),
        "training_report_path": (
            str(args.training_report.resolve()) if args.training_report else None
        ),
        "training_report_sha256": (
            sha256_file(args.training_report) if args.training_report else None
        ),
        "initialization_policy": (
            training["initialization_policy"] if training else "probe_only_not_a_label"
        ),
    }
    if not report["numerical_finite"]:
        raise ValueError("F-Cooper exported source produced non-finite values")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--width", required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--original-unpruned", action="store_true")
    parser.add_argument(
        "--input-shape", type=int, nargs=4, default=[5, 64, 512, 512]
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(export(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
