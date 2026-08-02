#!/usr/bin/env python3
"""Collect real OPV2V post-scatter tensors for F-Cooper TRT calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--engine-agent-batch", type=int, default=5)
    parser.add_argument("--split", choices=["validate", "test"], default="validate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    if args.sample_count < 1 or args.engine_agent_batch < 1:
        raise ValueError("sample-count and engine-agent-batch must be positive")
    hypes = yaml_utils.load_yaml(
        str(args.config), SimpleNamespace(model_dir=None)
    )
    if args.split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
    dataset = build_dataset(hypes, visualize=False, train=False)
    indices = np.linspace(
        0, len(dataset) - 1, num=min(args.sample_count, len(dataset)), dtype=int
    ).tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
    )
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.checkpoint_dir), model)
    model.cuda().eval()

    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        captured.append(inputs[0]["spatial_features"].detach().cpu())

    handle = model.backbone_m1.register_forward_pre_hook(hook)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for sample_index, batch in zip(indices, loader):
            if batch is None:
                continue
            captured.clear()
            batch = train_utils.to_device(batch, torch.device("cuda"))
            with torch.no_grad():
                model(batch["ego"])
            if len(captured) != 1:
                raise RuntimeError("expected exactly one backbone input capture")
            tensor = captured[0].numpy().astype(np.float32, copy=False)
            agents = int(tensor.shape[0])
            if agents > args.engine_agent_batch:
                raise ValueError(
                    f"sample {sample_index} has {agents} agents, exceeds engine batch"
                )
            padded = np.zeros(
                (args.engine_agent_batch, *tensor.shape[1:]), dtype=np.float32
            )
            padded[:agents] = tensor
            path = output / f"sample_{sample_index:05d}.npy"
            np.save(path, padded)
            records.append(
                {
                    "sample_index": int(sample_index),
                    "agents": agents,
                    "shape": list(padded.shape),
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
    finally:
        handle.remove()
    summary = {
        "schema_version": "fcooper_calibration_manifest_v1",
        "dataset": "OPV2V",
        "split": args.split,
        "dataset_samples": len(dataset),
        "sample_count": len(records),
        "engine_agent_batch": args.engine_agent_batch,
        "config_sha256": sha256_file(args.config),
        "checkpoint_sha256": sha256_file(
            args.checkpoint_dir / "net_epoch_bestval_at23.pth"
        ),
        "records": records,
    }
    summary_path = output.parent / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
