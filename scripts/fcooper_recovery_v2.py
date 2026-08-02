#!/usr/bin/env python3
"""Prepare an importance-initialized F-Cooper subnet for recovery training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fcooper_materialize_source_v1 import (  # noqa: E402
    FCooperDenseEncoder,
    apply_structure_to_hypes,
    parse_width,
    sha256_file,
)


def build_recovery_contract(seed: int = 20260723) -> dict[str, Any]:
    return {
        "schema_version": "fcooper_recovery_training_contract_v2",
        "initialization_policy": "scanner_dependency_l1_v2",
        "importance_criterion": "conv_filter_l1",
        "prefix_projection_is_final_measurement_source": False,
        "start_epoch": 23,
        "recovery_epochs": 8,
        "minimum_epochs": 4,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 1.0e-4,
        "optimizer_source": "frozen_model_config",
        "scheduler_source": "frozen_model_config",
        "amp_fp16": True,
        "seed": int(seed),
        "full_train_split": True,
        "full_validation_split": True,
        "train_shuffle": True,
        "validation_shuffle": False,
        "num_workers": 4,
    }


def select_l1_channels(
    weight: torch.Tensor,
    *,
    count: int,
    output_axis: int,
) -> torch.Tensor:
    if count <= 0 or count > weight.shape[output_axis]:
        raise ValueError("selected channel count is outside the source tensor")
    reduce_axes = tuple(axis for axis in range(weight.ndim) if axis != output_axis)
    scores = weight.detach().abs().sum(dim=reduce_axes).cpu().numpy()
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return torch.tensor(sorted(ranked[:count]), dtype=torch.long)


def project_conv2d_weight(
    weight: torch.Tensor,
    *,
    output_indices: torch.Tensor,
    input_indices: torch.Tensor,
) -> torch.Tensor:
    return weight.index_select(0, output_indices).index_select(1, input_indices).clone()


def project_conv_transpose2d_weight(
    weight: torch.Tensor,
    *,
    output_indices: torch.Tensor,
    input_indices: torch.Tensor,
) -> torch.Tensor:
    return weight.index_select(0, input_indices).index_select(1, output_indices).clone()


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _load_hypes(config: Path) -> dict[str, Any]:
    from opencood.hypes_yaml import yaml_utils

    return yaml_utils.load_yaml(str(config), SimpleNamespace(model_dir=None))


def _create_model(hypes: Mapping[str, Any]) -> nn.Module:
    from opencood.tools import train_utils

    return train_utils.create_model(dict(hypes))


def _conv_names(model: nn.Module, prefix: str, kind: type[nn.Module]) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if name.startswith(prefix) and isinstance(module, kind)
    ]


def _copy_batch_norm(
    source: Mapping[str, torch.Tensor],
    projected: dict[str, torch.Tensor],
    assigned: set[str],
    name: str,
    indices: torch.Tensor,
) -> None:
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        key = f"{name}.{suffix}"
        projected[key] = source[key].index_select(0, indices).clone()
        assigned.add(key)
    tracked = f"{name}.num_batches_tracked"
    if tracked in source:
        projected[tracked] = source[tracked].clone()
        assigned.add(tracked)


def _following_batch_norm(model: nn.Module, conv_name: str) -> str | None:
    parent_name, child_name = conv_name.rsplit(".", 1)
    parent = model.get_submodule(parent_name)
    children = list(parent._modules)
    start = children.index(child_name)
    for candidate in children[start + 1 :]:
        module = parent._modules[candidate]
        if isinstance(module, nn.BatchNorm2d):
            return f"{parent_name}.{candidate}"
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            break
    return None


def _copy_conv2d(
    source_model: nn.Module,
    source: Mapping[str, torch.Tensor],
    target_model: nn.Module,
    projected: dict[str, torch.Tensor],
    assigned: set[str],
    name: str,
    input_indices: torch.Tensor,
) -> torch.Tensor:
    source_module = source_model.get_submodule(name)
    target_module = target_model.get_submodule(name)
    if not isinstance(source_module, nn.Conv2d) or not isinstance(
        target_module, nn.Conv2d
    ):
        raise TypeError(f"{name} is not a Conv2d pair")
    output_indices = select_l1_channels(
        source[f"{name}.weight"],
        count=target_module.out_channels,
        output_axis=0,
    )
    projected[f"{name}.weight"] = project_conv2d_weight(
        source[f"{name}.weight"],
        output_indices=output_indices,
        input_indices=input_indices,
    )
    assigned.add(f"{name}.weight")
    if source_module.bias is not None:
        projected[f"{name}.bias"] = source[f"{name}.bias"].index_select(
            0, output_indices
        )
        assigned.add(f"{name}.bias")
    batch_norm = _following_batch_norm(source_model, name)
    if batch_norm is not None:
        _copy_batch_norm(
            source,
            projected,
            assigned,
            batch_norm,
            output_indices,
        )
    return output_indices


def _copy_deblock(
    source_model: nn.Module,
    source: Mapping[str, torch.Tensor],
    target_model: nn.Module,
    projected: dict[str, torch.Tensor],
    assigned: set[str],
    name: str,
    input_indices: torch.Tensor,
) -> torch.Tensor:
    target_module = target_model.get_submodule(name)
    if not isinstance(target_module, nn.ConvTranspose2d):
        raise TypeError(f"{name} is not a ConvTranspose2d")
    output_indices = select_l1_channels(
        source[f"{name}.weight"],
        count=target_module.out_channels,
        output_axis=1,
    )
    projected[f"{name}.weight"] = project_conv_transpose2d_weight(
        source[f"{name}.weight"],
        output_indices=output_indices,
        input_indices=input_indices,
    )
    assigned.add(f"{name}.weight")
    batch_norm = _following_batch_norm(source_model, name)
    if batch_norm is not None:
        _copy_batch_norm(
            source,
            projected,
            assigned,
            batch_norm,
            output_indices,
        )
    return output_indices


def project_fcooper_state_dict(
    source_model: nn.Module,
    target_model: nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    projected = {
        name: tensor.detach().cpu().clone()
        for name, tensor in target_model.state_dict().items()
    }
    source = {name: tensor.detach().cpu() for name, tensor in source_state.items()}
    assigned: set[str] = set()
    selections: dict[str, list[int]] = {}

    current = torch.arange(64, dtype=torch.long)
    stage_outputs: list[torch.Tensor] = []
    for stage in range(3):
        names = _conv_names(source_model, f"backbone_m1.blocks.{stage}.", nn.Conv2d)
        if not names:
            raise ValueError(f"scanner dependency stage {stage} has no Conv2d")
        for name in names:
            current = _copy_conv2d(
                source_model,
                source,
                target_model,
                projected,
                assigned,
                name,
                current,
            )
            selections[name] = current.tolist()
        stage_outputs.append(current)

    deblock_outputs = []
    for stage, input_indices in enumerate(stage_outputs):
        name = f"backbone_m1.deblocks.{stage}.0"
        selected = _copy_deblock(
            source_model,
            source,
            target_model,
            projected,
            assigned,
            name,
            input_indices,
        )
        selections[name] = selected.tolist()
        deblock_outputs.append(selected)

    source_deblock_width = int(
        source_model.get_submodule("backbone_m1.deblocks.0.0").out_channels
    )
    concat_input = torch.cat(
        [indices + branch * source_deblock_width for branch, indices in enumerate(deblock_outputs)]
    )
    shrink0 = "shrinker_m1.layers.0.double_conv.0"
    shrink0_output = _copy_conv2d(
        source_model,
        source,
        target_model,
        projected,
        assigned,
        shrink0,
        concat_input,
    )
    selections[shrink0] = shrink0_output.tolist()
    shrink2 = "shrinker_m1.layers.0.double_conv.2"
    shrink2_output = _copy_conv2d(
        source_model,
        source,
        target_model,
        projected,
        assigned,
        shrink2,
        shrink0_output,
    )
    selections[shrink2] = shrink2_output.tolist()

    for head in ("cls_head", "reg_head", "dir_head"):
        weight_key = f"{head}.weight"
        projected[weight_key] = source[weight_key].index_select(
            1, shrink2_output
        ).clone()
        assigned.add(weight_key)
        bias_key = f"{head}.bias"
        projected[bias_key] = source[bias_key].clone()
        assigned.add(bias_key)

    for name, target_tensor in target_model.state_dict().items():
        if name in assigned:
            continue
        source_tensor = source.get(name)
        if source_tensor is None or source_tensor.shape != target_tensor.shape:
            raise ValueError(f"unmapped F-Cooper dependency tensor: {name}")
        projected[name] = source_tensor.clone()
        assigned.add(name)

    target_model.load_state_dict(projected, strict=True)
    audit = {
        "schema_version": "fcooper_importance_projection_audit_v2",
        "policy": "scanner_dependency_l1_v2",
        "assigned_tensor_count": len(assigned),
        "target_tensor_count": len(projected),
        "selected_source_channels": selections,
    }
    audit["audit_sha256"] = _sha_payload(audit)
    return projected, audit


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    contract = json.loads(args.recovery_contract.read_text())
    expected = build_recovery_contract(seed=int(contract["seed"]))
    if contract != expected:
        raise ValueError("recovery training contract drift")
    width = parse_width(args.width)
    random.seed(contract["seed"])
    np.random.seed(contract["seed"])
    torch.manual_seed(contract["seed"])

    source_hypes = _load_hypes(args.source_config)
    target_hypes = apply_structure_to_hypes(source_hypes, width)
    source_model = _create_model(source_hypes)
    target_model = _create_model(target_hypes)
    source_state = torch.load(args.source_checkpoint, map_location="cpu")
    projected, projection_audit = project_fcooper_state_dict(
        source_model, target_model, source_state
    )
    dense = FCooperDenseEncoder(target_model).eval()
    sample = torch.zeros(1, 64, 64, 64, dtype=torch.float32)
    with torch.no_grad():
        forward_output = dense(sample)

    from opencood.hypes_yaml import yaml_utils

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = output / "config.yaml"
    checkpoint = output / "net_epoch_bestval_at23.pth"
    yaml_utils.save_yaml(target_hypes, str(config))
    torch.save(projected, checkpoint)
    report = {
        "schema_version": "fcooper_recovery_initialization_v2",
        "status": "ready_for_recovery_training",
        "width": list(width),
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "initial_checkpoint_path": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "source_config_sha256": sha256_file(args.source_config),
        "source_checkpoint_sha256": sha256_file(args.source_checkpoint),
        "recovery_contract_path": str(args.recovery_contract.resolve()),
        "recovery_contract_sha256": sha256_file(args.recovery_contract),
        "projection": projection_audit,
        "forward_sanity": {
            "status": "success",
            "input_shape": list(sample.shape),
            "output_shape": list(forward_output.shape),
            "finite": bool(torch.isfinite(forward_output).all()),
        },
    }
    if not report["forward_sanity"]["finite"]:
        raise ValueError("importance-initialized F-Cooper forward produced non-finite values")
    report_path = output / "recovery_initialization_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--recovery-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
