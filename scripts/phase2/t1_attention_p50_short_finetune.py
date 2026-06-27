"""Short finetune runner for T1 attention-p50 recovery checks.

This stays in T1. It reuses the same manual HMSA/MSwin p50 surgery from
``t1_attention_e2e_pq.py`` and records enough metadata to audit whether the
no-finetune AP drop can recover under a controlled short finetune.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from t1_attention_e2e_pq import (
    ARTIFACT_DIR,
    LOG_DIR,
    REPO_ROOT,
    _command_string,
    _round,
    apply_attention_p50_surgery,
    build_dataset_from_hypes,
    evaluate_ap,
    load_model_and_hypes,
    measure_model_forward_latency,
    save_attention_pruned_artifact,
)


def make_short_finetune_manifest(
    *,
    steps: int,
    lr: float,
    seed: int,
    batch_size: int,
    train_scope: str,
    amp: bool,
    max_train_samples: int | None,
) -> dict[str, Any]:
    return {
        "method": "attention_p50_short_finetune",
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "train_scope": train_scope,
        "amp": bool(amp),
        "max_train_samples": max_train_samples,
        "notes": [
            "Starts from official epoch17 checkpoint and applies the same manual attention-p50 surgery.",
            "This is a recovery check for the no-finetune Stop-C risk, not a final full-val result by itself.",
        ],
    }


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_train_scope(model, train_scope: str) -> dict[str, Any]:
    if train_scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
        return {"train_scope": "all", "trainable_params": sum(p.numel() for p in model.parameters())}
    if train_scope != "attention_only":
        raise ValueError(f"unsupported train_scope={train_scope}")

    keywords = ("hmsa", "mswin", "cav_att", "pwindow", "fusion_net")
    trainable = 0
    matched_names: list[str] = []
    for name, param in model.named_parameters():
        enabled = any(key in name.lower() for key in keywords)
        param.requires_grad_(enabled)
        if enabled:
            trainable += param.numel()
            matched_names.append(name)
    return {
        "train_scope": "attention_only",
        "trainable_params": trainable,
        "matched_param_count": len(matched_names),
        "matched_param_examples": matched_names[:20],
    }


def _make_train_loader(hypes: dict[str, Any], batch_size: int, num_workers: int, max_train_samples: int | None, seed: int):
    import torch
    from torch.utils.data import DataLoader, Subset
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=False, train=True)
    if max_train_samples is not None and max_train_samples > 0:
        dataset = Subset(dataset, list(range(min(max_train_samples, len(dataset)))))
        collate_fn = dataset.dataset.collate_batch_train
    else:
        collate_fn = dataset.collate_batch_train
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        generator=generator,
    )


def short_finetune_attention_p50(
    *,
    device: str,
    steps: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_scope: str,
    amp: bool,
    max_train_samples: int | None,
    eval_samples: int,
    latency_samples: int,
    latency_warmup: int,
    eval_precision: str,
) -> dict[str, Any]:
    import torch

    started = time.time()
    _set_seed(seed)
    manifest = make_short_finetune_manifest(
        steps=steps,
        lr=lr,
        seed=seed,
        batch_size=batch_size,
        train_scope=train_scope,
        amp=amp,
        max_train_samples=max_train_samples,
    )
    train_log_path = LOG_DIR / "attention_p50_shortft_train_v1.json"
    result: dict[str, Any]
    try:
        model, hypes = load_model_and_hypes(device)
        from opencood.tools import train_utils

        prune_manifest = apply_attention_p50_surgery(model)
        scope_info = _set_train_scope(model, train_scope)
        model.train()
        try:
            model.model_train_init()
        except Exception:
            pass

        criterion = train_utils.create_loss(hypes)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=float(hypes.get("optimizer", {}).get("args", {}).get("weight_decay", 0.0)),
            eps=float(hypes.get("optimizer", {}).get("args", {}).get("eps", 1e-10)),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        loader = _make_train_loader(hypes, batch_size, num_workers, max_train_samples, seed)
        device_obj = torch.device(device)
        losses: list[float] = []
        step = 0
        while step < steps:
            for batch in loader:
                if step >= steps:
                    break
                if batch is None or batch["ego"]["object_bbx_mask"].sum() == 0:
                    continue
                batch = train_utils.to_device(batch, device_obj)
                batch["ego"]["epoch"] = 0
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(batch["ego"])
                    loss = criterion(output, batch["ego"]["label_dict"])
                    if hasattr(loader.dataset, "supervise_single") and loader.dataset.supervise_single:
                        loss = loss + criterion(output, batch["ego"]["label_dict_single"], suffix="_single")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                step += 1
            if len(loader) == 0:
                raise RuntimeError("empty train loader")

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = ARTIFACT_DIR / f"attention_p50_shortft_steps{steps}_lr{lr:g}_seed{seed}.pth"
        manifest_path = ARTIFACT_DIR / f"attention_p50_shortft_steps{steps}_manifest_v1.json"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "attention_prune_manifest": prune_manifest,
                "short_finetune_manifest": manifest,
            },
            ckpt_path,
        )
        combined_manifest = {
            "attention_prune_manifest": prune_manifest,
            "short_finetune_manifest": manifest,
            "train_scope_info": scope_info,
        }
        manifest_path.write_text(json.dumps(combined_manifest, indent=2, ensure_ascii=False))

        eval_info: dict[str, Any] = {}
        if eval_samples > 0 or latency_samples > 0:
            val_dataset = build_dataset_from_hypes(hypes)
            if latency_samples > 0:
                eval_info["latency"] = measure_model_forward_latency(
                    model.eval(),
                    val_dataset,
                    device,
                    eval_precision,
                    latency_warmup,
                    latency_samples,
                    LOG_DIR / "attention_p50_shortft_latency_v1.json",
                    num_workers,
                )
            if eval_samples > 0:
                eval_info["ap"] = evaluate_ap(
                    model.eval(),
                    val_dataset,
                    device,
                    eval_precision,
                    eval_samples,
                    LOG_DIR / "attention_p50_shortft_ap_v1.json",
                    num_workers,
                )

        result = {
            "status": "OK",
            "checkpoint_path": str(ckpt_path),
            "manifest_path": str(manifest_path),
            "short_finetune_manifest": manifest,
            "train_scope_info": scope_info,
            "steps_completed": step,
            "loss_first": _round(losses[0], 6) if losses else None,
            "loss_last": _round(losses[-1], 6) if losses else None,
            "loss_mean": _round(np.mean(losses), 6) if losses else None,
            "losses": [_round(x, 6) for x in losses[-20:]],
            "eval": eval_info,
            "elapsed_secs": _round(time.time() - started, 3),
            "command": _command_string(),
        }
    except Exception as exc:
        result = {
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-5000:],
            "short_finetune_manifest": manifest,
            "elapsed_secs": _round(time.time() - started, 3),
            "command": _command_string(),
        }
    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    train_log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (REPO_ROOT / "results" / "attention_p50_shortft_v1.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--train-scope", choices=["all", "attention_only"], default="all")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=0)
    parser.add_argument("--latency-samples", type=int, default=0)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--eval-precision", choices=["fp16", "fp32"], default="fp16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = short_finetune_attention_p50(
        device=args.device,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_scope=args.train_scope,
        amp=args.amp,
        max_train_samples=args.max_train_samples or None,
        eval_samples=args.eval_samples,
        latency_samples=args.latency_samples,
        latency_warmup=args.latency_warmup,
        eval_precision=args.eval_precision,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
