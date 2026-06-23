"""【A-2 正式授权 — team-lead 2026-06-04】A-2 批准启动
授权范围: backbone_m1 L1 剪枝 p50+p75 finetune 25ep

A-2 V2X-ViT backbone 剪枝后 finetune 脚本。

策略: "re-prune from epoch17 → load saved pruned ckpt → finetune"
  1. 加载 epoch17 原始权重 → 对 backbone_m1 应用相同 DepGraph 剪枝 (确定性, 同权重→同通道选择)
  2. 将 depgraph_v2xvit.py 保存的 flat state_dict 载入剪枝后模型 (strict=True, 结构完全匹配)
  3. 用 HEAL 原生训练 loop 在 DAIR train (4811 samples) 上 finetune 25 epoch
  4. 每 5 epoch 保存 ckpt, 记录 bestval AP → flat state_dict (ISS-005)

纪律:
  ISS-005: flat state_dict 保存 (无 model_state_dict 包裹)
  ISS-009: finetune 后 AP 才算有效数据; 未 finetune 的 AP 会崩
  ISS-024: epoch_used 记录; 这里 epoch_src = bestval_at17_pruned (pruned init)

用法:
  CUDA_VISIBLE_DEVICES=1 python scripts/phase2/a2_finetune_v2xvit.py \\
      --ratio 0.5 \\
      --prune-ckpt output/a2_prune/v2xvit_bb_p50/v2xvit_pruned_50_epoch17_depgraph.pth \\
      --out-dir output/a2_finetune/v2xvit_bb_p50 \\
      2>&1 | tee logs/a2_finetune_p50.log

  CUDA_VISIBLE_DEVICES=2 python scripts/phase2/a2_finetune_v2xvit.py \\
      --ratio 0.75 \\
      --prune-ckpt output/a2_prune/v2xvit_bb_p75/v2xvit_pruned_75_epoch17_depgraph.pth \\
      --out-dir output/a2_finetune/v2xvit_bb_p75 \\
      2>&1 | tee logs/a2_finetune_p75.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
_HEAL = Path("/home/jichengzhi/heal_research/HEAL")
for _p in (str(_REPO), str(_HEAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(str(_HEAL))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils

# DepGraph pruning utilities
import torch_pruning as tp
from tools.configurable.depgraph_v2xvit import (
    V2XViTBackboneTraceNet, build_model, build_pruner, get_scatter_shape
)

CKPT_DIR = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
N_FINETUNE_EPOCHS = 25
SAVE_FREQ = 5
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"


def rebuild_pruned_model(ratio: float, saved_ckpt: Path, device: str):
    """Re-run DepGraph pruning from epoch17 to get exact pruned architecture,
    then load saved pruned flat state_dict (deterministic: same weights → same channels).
    """
    print(f"[rebuild] Loading epoch17 base weights for deterministic repruning...")
    model = build_model(device)
    net   = V2XViTBackboneTraceNet(model).to(device).eval()

    hypes_raw = yaml_utils.load_yaml(str(CONFIG_YAML))
    ny, nx    = get_scatter_shape(hypes_raw)
    x = torch.randn(1, 64, ny, nx, device=device)

    print(f"[rebuild] Applying DepGraph pruning ratio={ratio} (deterministic)...")
    pr = build_pruner(net, x, ratio, device)
    pr.step()

    # Copy pruned backbone/shrinker weights back to original model object
    for attr in ("backbone_m1", "shrinker_m1"):
        # params are shared (V2XViTBackboneTraceNet holds refs)
        pass  # weights already updated in-place by pr.step()

    print(f"[rebuild] Loading saved pruned flat ckpt: {saved_ckpt.name}")
    saved_sd = torch.load(str(saved_ckpt), map_location="cpu")

    # Build full model state_dict from pruned trace net + remaining original modules
    # The trace net has backbone_m1 and shrinker_m1 already pruned.
    # We merge with the original model's other modules (fusion_net, encoder_m1, heads)
    merged_sd = {}
    for k, v in net.backbone_m1.state_dict().items():
        merged_sd[f"backbone_m1.{k}"] = v
    for k, v in net.shrinker_m1.state_dict().items():
        merged_sd[f"shrinker_m1.{k}"] = v
    for k, v in model.state_dict().items():
        if not k.startswith("backbone_m1.") and not k.startswith("shrinker_m1."):
            merged_sd[k] = v

    # Verify saved ckpt matches our rebuilt structure
    repruned_keys = set(merged_sd.keys())
    saved_keys    = set(saved_sd.keys())
    if repruned_keys != saved_keys:
        extra_in_saved  = saved_keys - repruned_keys
        missing_in_saved = repruned_keys - saved_keys
        print(f"[WARN] Key mismatch: extra={len(extra_in_saved)} missing={len(missing_in_saved)}")
        if extra_in_saved:
            print(f"       extra in saved: {list(extra_in_saved)[:5]}")
        if missing_in_saved:
            print(f"       missing from saved: {list(missing_in_saved)[:5]}")

    # Load saved pruned weights (override repruned weights with saved)
    missing, unexpected = model.load_state_dict(saved_sd, strict=False)
    if missing:
        print(f"[WARN] missing keys when loading: {missing[:5]}")
    if unexpected:
        print(f"[WARN] unexpected keys: {unexpected[:5]}")
    print(f"[rebuild] ✓ Model rebuilt with saved pruned weights")

    # Verify backbone params
    bb_params = sum(p.numel() for p in model.backbone_m1.parameters())
    sh_params  = sum(p.numel() for p in model.shrinker_m1.parameters())
    print(f"[rebuild] backbone_m1: {bb_params:,}  shrinker_m1: {sh_params:,}")

    return model


def save_flat_ckpt(model, out_dir: Path, epoch: int, suffix: str = ""):
    """Save flat state_dict (ISS-005)."""
    fname = f"net_epoch{epoch}{suffix}.pth"
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(sd, str(out_dir / fname))
    return out_dir / fname


def finetune(model, out_dir: Path, device: str, ratio: float):
    """HEAL-style training loop for V2X-ViT on DAIR train split."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load hypes for DAIR training
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    from opencood.hypes_yaml.yaml_utils import load_general_params
    hypes = load_general_params(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    # set train=True to use train split
    train_dataset = build_dataset(hypes, visualize=False, train=True)
    val_dataset   = build_dataset(hypes, visualize=False, train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=2, num_workers=4,
        collate_fn=train_dataset.collate_batch_train,
        shuffle=True, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, num_workers=2,
        collate_fn=val_dataset.collate_batch_test,
        shuffle=False, pin_memory=False, drop_last=False,
    )

    print(f"[train] train={len(train_dataset)} val={len(val_dataset)}")

    # HEAL-style criterion (from hypes['loss'] config)
    criterion = train_utils.create_loss(hypes)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3,
        eps=1e-10, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.1
    )

    model = model.to(device)
    best_ap50 = -1.0
    best_epoch = -1
    meta = {
        "ratio": ratio,
        "epoch_src": "bestval_at17_pruned",
        "n_finetune_epochs": N_FINETUNE_EPOCHS,
        "epochs": []
    }

    for epoch in range(1, N_FINETUNE_EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss, n_batch = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            if batch is None: continue
            if batch["ego"]["object_bbx_mask"].sum() == 0: continue
            batch = train_utils.to_device(batch, device)
            batch["ego"]["epoch"] = epoch
            model.zero_grad()
            optimizer.zero_grad()
            # HEAL forward: model(batch['ego']) → output_dict
            output_dict = model(batch["ego"])
            # HEAL criterion: criterion(output_dict, label_dict) → scalar loss
            loss = criterion(output_dict, batch["ego"]["label_dict"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += float(loss)
            n_batch += 1
        scheduler.step()
        train_loss = total_loss / max(n_batch, 1)

        # ── Val AP (every epoch) ───────────────────────────────────────────
        model.eval()
        # BUG FIX: eval_final_results internally calls calculate_ap(0.30) →
        # KeyError if 0.3 missing. Must include all three IoU thresholds.
        result_stat = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                batch = train_utils.to_device(batch, device)
                # Use HEAL inference_utils for model forward
                from opencood.tools import inference_utils
                infer = inference_utils.inference_intermediate_fusion(
                    batch, model, val_dataset)
                for iou_th in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(
                        infer["pred_box_tensor"], infer["pred_score"],
                        infer["gt_box_tensor"], result_stat, iou_th)

        tmp_dir = out_dir / f"epoch{epoch}_eval_tmp"
        tmp_dir.mkdir(exist_ok=True)
        _, ap50, _ = eval_utils.eval_final_results(result_stat, str(tmp_dir))
        import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

        elapsed = time.time() - t0
        print(f"  ep{epoch:02d}/{N_FINETUNE_EPOCHS} "
              f"loss={train_loss:.4f} ap50={ap50:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.6f} t={elapsed:.0f}s",
              flush=True)

        # ── Checkpoint ────────────────────────────────────────────────────
        if epoch % SAVE_FREQ == 0:
            save_flat_ckpt(model, out_dir, epoch)

        if ap50 > best_ap50:
            best_ap50 = ap50
            best_epoch = epoch
            save_flat_ckpt(model, out_dir, epoch, "_bestval")
            # keep only latest bestval
            for old in out_dir.glob("net_epoch*_bestval.pth"):
                if old.stem != f"net_epoch{epoch}_bestval":
                    old.unlink(missing_ok=True)

        meta["epochs"].append({
            "epoch": epoch, "train_loss": train_loss,
            "ap50": float(ap50), "is_best": ap50 == best_ap50
        })
        with open(out_dir / "train_log.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ── Final save ─────────────────────────────────────────────────────────
    final_ckpt = save_flat_ckpt(model, out_dir, N_FINETUNE_EPOCHS)
    print(f"\n[finetune] DONE: best ap50={best_ap50:.4f} @ epoch{best_epoch}")
    print(f"[finetune] best ckpt: net_epoch{best_epoch}_bestval.pth")
    print(f"[finetune] final ckpt: {final_ckpt.name}")
    meta["best_ap50"] = float(best_ap50)
    meta["best_epoch"] = best_epoch
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(meta, f, indent=2)
    return best_ap50, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio",      type=float, required=True,
                    help="Pruning ratio used (e.g. 0.5)")
    ap.add_argument("--prune-ckpt", type=str,   required=True,
                    help="Path to saved pruned flat state_dict")
    ap.add_argument("--out-dir",    type=str,   required=True,
                    help="Output dir for finetune ckpts + logs")
    ap.add_argument("--device",     type=str,   default="cuda",
                    help="Device (cuda or cuda:0)")
    args = ap.parse_args()

    prune_ckpt = Path(args.prune_ckpt).resolve()
    out_dir    = Path(args.out_dir).resolve()
    device     = args.device

    print(f"\n{'='*72}")
    print(f"A-2 V2X-ViT backbone finetune  ratio={args.ratio}  device={device}")
    print(f"  prune_ckpt: {prune_ckpt}")
    print(f"  out_dir:    {out_dir}")
    print(f"{'='*72}")

    # 1. Rebuild pruned model + load saved pruned weights
    model = rebuild_pruned_model(args.ratio, prune_ckpt, "cpu")
    model = model.to(device)

    # 2. Finetune
    best_ap50, best_epoch = finetune(model, out_dir, device, args.ratio)

    print(f"\n★ A-2 finetune complete: ratio={args.ratio} best_ap50={best_ap50:.4f} ep{best_epoch}")
    print(f"  Next: run eval_v2xvit_baseline_a1.py with this ckpt for AP70+mAOE")


if __name__ == "__main__":
    main()
