from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "t1_attention_p50_short_finetune.py"
    spec = importlib.util.spec_from_file_location("t1_attention_p50_short_finetune", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_short_finetune_manifest_records_protocol_fields():
    mod = _load_module()

    manifest = mod.make_short_finetune_manifest(
        steps=100,
        lr=1e-4,
        seed=20260623,
        batch_size=1,
        train_scope="all",
        amp=True,
        max_train_samples=512,
    )

    assert manifest["method"] == "attention_p50_short_finetune"
    assert manifest["steps"] == 100
    assert manifest["lr"] == 1e-4
    assert manifest["seed"] == 20260623
    assert manifest["batch_size"] == 1
    assert manifest["train_scope"] == "all"
    assert manifest["amp"] is True
    assert manifest["max_train_samples"] == 512
    assert "no-finetune Stop-C risk" in " ".join(manifest["notes"])


def test_attention_only_scope_freezes_non_attention_parameters():
    mod = _load_module()

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.hmsa_block = torch.nn.Linear(2, 2)
            self.fusion_net = torch.nn.Linear(2, 2)

    model = Tiny()

    info = mod._set_train_scope(model, "attention_only")

    assert info["train_scope"] == "attention_only"
    assert model.backbone.weight.requires_grad is False
    assert model.hmsa_block.weight.requires_grad is True
    assert model.fusion_net.weight.requires_grad is True
    assert info["matched_param_count"] == 4


def test_all_scope_enables_all_parameters():
    mod = _load_module()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for param in model.parameters():
        param.requires_grad_(False)

    info = mod._set_train_scope(model, "all")

    assert info["train_scope"] == "all"
    assert all(param.requires_grad for param in model.parameters())
    assert info["trainable_params"] == sum(param.numel() for param in model.parameters())
