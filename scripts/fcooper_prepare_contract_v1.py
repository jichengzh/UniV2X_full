#!/usr/bin/env python3
"""Freeze F-Cooper Work Package A contracts from scanner and H800 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage1.hardware_scan import HwCapability
from framework.stage5.fcooper_space_v1 import build_fcooper_source_registry


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-artifact-root", type=Path, required=True)
    parser.add_argument("--dataset-samples", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    partition = yaml.safe_load(args.partition.read_text())
    capability = HwCapability.from_yaml(args.hardware)
    partition["hw_capability"] = capability.summary()
    partition["hw_capability"]["runtime_evidence"] = {
        "gpu": "NVIDIA H800",
        "cuda_capability": [9, 0],
        "tensorrt": "10.13.0.35",
        "torch": "2.0.1+cu118",
        "cuda_runtime": "11.8",
        "cudnn": 8700,
    }
    partition["config"] = str(args.config.resolve())
    partition["ckpt"] = str(args.checkpoint.resolve())
    partition_path = output / "fcooper_partition_h800.yaml"
    partition_path.write_text(yaml.safe_dump(partition, sort_keys=False))

    registry = build_fcooper_source_registry(
        partition_path, artifact_root=args.remote_artifact_root
    )
    write_json(output / "candidate_source_registry.json", registry)
    reference = yaml.safe_load(args.reference_eval.read_text())
    contract = {
        "schema_version": "fcooper_workpackage_a_contract_v1",
        "model": "fcooper",
        "dataset": "OPV2V",
        "split": "test",
        "dataset_samples": args.dataset_samples,
        "test_manifest_path": str(args.test_manifest.resolve()),
        "test_manifest_sha256": sha256_file(args.test_manifest),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "partition_path": str(partition_path),
        "partition_sha256": sha256_file(partition_path),
        "ap70_ref": float(reference["ap_70"]),
        "ap50_ref": float(reference["ap_50"]),
        "ap30_ref": float(reference["ap30"]),
        "measurement_scope": "post_scatter_backbone_shrinker_plus_native_maxfusion_heads",
        "engine_agent_batch": 5,
        "inference_batch": 1,
        "input_shape": [5, 64, 512, 512],
        "width_schema": registry["width_schema"],
        "structure_candidate_count": registry["structure_candidate_count"],
        "precision_genome_count": 2 * registry["structure_candidate_count"],
        "search_budget": {"batch_size": 4, "rounds": 4, "total": 16},
    }
    write_json(output / "frozen_contract.json", contract)
    probe_plan = {
        "schema_version": "fcooper_probe_plan_v1",
        "rows": [
            {"probe_id": "fp16_base_default", "width": [64, 128, 256, 128, 256]},
            {"probe_id": "fp16_base_tuned", "width": [64, 128, 256, 128, 256]},
            {"probe_id": "fp16_boundary_default", "width": [32, 32, 32, 32, 64]},
            {"probe_id": "fp16_boundary_tuned", "width": [32, 32, 32, 32, 64]},
            {"probe_id": "int8_base", "width": [64, 128, 256, 128, 256]},
            {"probe_id": "int8_boundary", "width": [32, 32, 32, 32, 64]},
            {"probe_id": "maxfusion_base", "width": [64, 128, 256, 128, 256]},
            {"probe_id": "maxfusion_boundary", "width": [32, 32, 32, 32, 64]},
        ],
    }
    write_json(output / "probe_plan.json", probe_plan)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
