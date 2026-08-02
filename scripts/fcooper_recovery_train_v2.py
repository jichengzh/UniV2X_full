#!/usr/bin/env python3
"""Run the frozen F-Cooper recovery-training contract on one H800 GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def validate_recovery_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    from scripts.fcooper_recovery_v2 import build_recovery_contract

    expected = build_recovery_contract(seed=int(contract.get("seed", -1)))
    if dict(contract) != expected:
        drift = sorted(
            key
            for key in set(contract) | set(expected)
            if contract.get(key) != expected.get(key)
        )
        raise ValueError(f"F-Cooper recovery contract drift: {drift}")
    if int(contract["recovery_epochs"]) < int(contract["minimum_epochs"]):
        raise ValueError("F-Cooper recovery contract drift: recovery_epochs")
    return dict(contract)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_early_stopping(
    state: Mapping[str, Any],
    *,
    validation_loss: float,
    epoch: int,
    min_delta: float,
) -> dict[str, Any]:
    if validation_loss < float(state["best_loss"]) - min_delta:
        return {
            "best_loss": float(validation_loss),
            "bad_epochs": 0,
            "best_epoch": int(epoch),
        }
    return {
        "best_loss": float(state["best_loss"]),
        "bad_epochs": int(state["bad_epochs"]) + 1,
        "best_epoch": int(state["best_epoch"]),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    workers: int,
    train: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=dataset.collate_batch_train,
        shuffle=train,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if workers else None,
        generator=generator,
    )


def _run_epoch(
    model: torch.nn.Module,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    model.train()
    if hasattr(model, "model_train_init"):
        model.model_train_init()
    losses = []
    for batch_data in loader:
        if batch_data is None or batch_data["ego"]["object_bbx_mask"].sum() == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        from opencood.tools import train_utils

        batch_data = train_utils.to_device(batch_data, device)
        batch_data["ego"]["epoch"] = epoch
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            output = model(batch_data["ego"])
            loss = criterion(output, batch_data["ego"]["label_dict"])
            if getattr(loader.dataset, "supervise_single", False):
                loss = loss + criterion(
                    output,
                    batch_data["ego"]["label_dict_single"],
                    suffix="_single",
                )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("recovery train epoch produced no valid batches")
    return float(np.mean(losses))


def _validate(
    model: torch.nn.Module,
    criterion: Any,
    loader: DataLoader,
    device: torch.device,
    *,
    epoch: int,
    amp_enabled: bool,
) -> float:
    from opencood.tools import train_utils

    model.eval()
    losses = []
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = model(batch_data["ego"])
                loss = criterion(output, batch_data["ego"]["label_dict"])
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("recovery validation produced no valid batches")
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> dict[str, Any]:
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    heal_root = args.heal_root.expanduser().resolve()
    if not (heal_root / "opencood").is_dir():
        raise FileNotFoundError(f"invalid HEAL root: {heal_root}")
    os.chdir(heal_root)
    contract = validate_recovery_contract(
        json.loads(args.recovery_contract.read_text())
    )
    seed = int(contract["seed"])
    _set_seed(seed)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("F-Cooper recovery training requires CUDA")

    hypes = yaml_utils.load_yaml(
        str(args.config), SimpleNamespace(model_dir=str(args.model_dir))
    )
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    validation_dataset = build_dataset(hypes, visualize=False, train=False)
    batch_size = int(hypes["train_params"]["batch_size"])
    workers = int(contract["num_workers"])
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        workers=workers,
        train=True,
        seed=seed,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=batch_size,
        workers=workers,
        train=False,
        seed=seed,
    )

    model = train_utils.create_model(hypes)
    initial_state = torch.load(args.initial_checkpoint, map_location="cpu")
    model.load_state_dict(initial_state, strict=True)
    model.to(device)
    criterion = train_utils.create_loss(hypes)
    optimizer = train_utils.setup_optimizer(hypes, model)
    start_epoch = int(contract["start_epoch"])
    scheduler = train_utils.setup_lr_schedular(
        hypes, optimizer, init_epoch=start_epoch
    )
    amp_enabled = bool(contract["amp_fp16"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    writer = SummaryWriter(str(args.model_dir / "recovery_tensorboard"))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    initial_validation = _validate(
        model,
        criterion,
        validation_loader,
        device,
        epoch=start_epoch,
        amp_enabled=amp_enabled,
    )
    state: dict[str, Any] = {
        "best_loss": initial_validation,
        "bad_epochs": 0,
        "best_epoch": start_epoch,
    }
    best_source = args.initial_checkpoint
    epoch_records = []
    training_started = time.monotonic()
    stop_reason = "fixed_budget_complete"
    for offset in range(int(contract["recovery_epochs"])):
        epoch = start_epoch + offset + 1
        epoch_started = time.monotonic()
        train_loss = _run_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch=epoch,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        validation_loss = _validate(
            model,
            criterion,
            validation_loader,
            device,
            epoch=epoch,
            amp_enabled=amp_enabled,
        )
        previous = state
        state = update_early_stopping(
            state,
            validation_loss=validation_loss,
            epoch=epoch,
            min_delta=float(contract["early_stopping_min_delta"]),
        )
        checkpoint = args.model_dir / f"net_epoch{epoch}.pth"
        torch.save(model.state_dict(), checkpoint)
        if state["best_epoch"] != previous["best_epoch"]:
            best_source = checkpoint
        scheduler.step(epoch)
        train_dataset.reinitialize()
        writer.add_scalar("Recovery/TrainLoss", train_loss, epoch)
        writer.add_scalar("Recovery/ValidationLoss", validation_loss, epoch)
        epoch_records.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": time.monotonic() - epoch_started,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        completed = offset + 1
        if (
            completed >= int(contract["minimum_epochs"])
            and state["bad_epochs"] >= int(contract["early_stopping_patience"])
        ):
            stop_reason = "early_stopping_patience_reached"
            break
    writer.close()
    if len(epoch_records) < int(contract["minimum_epochs"]):
        raise RuntimeError("recovery training did not satisfy the minimum epoch contract")

    recovered = args.model_dir / "recovered_checkpoint.pth"
    shutil.copy2(best_source, recovered)
    gpu = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "fcooper_recovery_training_report_v2",
        "status": "success",
        "initialization_policy": contract["initialization_policy"],
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "initial_checkpoint_path": str(args.initial_checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
        "recovery_contract_path": str(args.recovery_contract.resolve()),
        "recovery_contract_sha256": sha256_file(args.recovery_contract),
        "dataset": {
            "train_root": hypes["root_dir"],
            "validation_root": hypes["validate_dir"],
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "full_train_split": True,
            "full_validation_split": True,
        },
        "seed": seed,
        "amp_fp16": amp_enabled,
        "optimizer": hypes["optimizer"],
        "lr_scheduler": hypes["lr_scheduler"],
        "initial_validation_loss": initial_validation,
        "best_validation_loss": state["best_loss"],
        "best_epoch": state["best_epoch"],
        "epochs_completed": len(epoch_records),
        "stop_reason": stop_reason,
        "epoch_records": epoch_records,
        "recovered_checkpoint_path": str(recovered),
        "recovered_checkpoint_sha256": sha256_file(recovered),
        "elapsed_seconds": time.monotonic() - training_started,
        "gpu": {
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": gpu.name,
            "total_memory": gpu.total_memory,
        },
        "heal_root": str(heal_root),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-contract", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
