#!/usr/bin/env python3
"""
Standalone AP evaluator for Pyramid pruned models.
Evaluates a specific checkpoint file (flat state_dict) on DAIR-V2X val_1789.

Usage:
    PYTHONPATH=/data/jichengzhi_v2x/t2lib:/exdata/jichengzhi/heal_research/HEAL \
    CUDA_VISIBLE_DEVICES=5 python3 tools/eval_pyramid_ap_by_ckpt.py \
        --config-yaml /path/to/config.yaml \
        --ckpt-path /path/to/net_epochN.pth \
        --label epoch_N_description \
        --out /home/jichengzhi/V2X/results/ap_epoch_curve.json \
        [--n-samples 1789]

Requirements: opencood available in PYTHONPATH (via HEAL PYTHONPATH)
Key: loads ckpt with strict=False and verifies missing_keys == 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# HEAL path bootstrap (needed if not in PYTHONPATH already)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_HEAL = "/exdata/jichengzhi/heal_research/HEAL"
_HEAL_PATH = os.environ.get("HEAL_ROOT", _DEFAULT_HEAL)
for _p in (_HEAL_PATH,):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_model(config_yaml: str, ckpt_path: str, device: str) -> tuple:
    """Load pruned Pyramid model from config + flat-state-dict ckpt.

    Returns (model, hypes, missing_keys, unexpected_keys).
    Raises ValueError if missing_keys != 0 (signals wrong architecture/format).
    """
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.models.heter_pyramid_collab import HeterPyramidCollab

    hypes = load_yaml(config_yaml)
    model = HeterPyramidCollab(hypes["model"]["args"])

    raw = torch.load(ckpt_path, map_location="cpu")
    # Handle wrapped format {"model_state_dict": ...}
    if isinstance(raw, dict) and "model_state_dict" in raw and len(raw) == 1:
        raw = raw["model_state_dict"]
    if isinstance(raw, dict) and "state_dict" in raw and len(raw) == 1:
        raw = raw["state_dict"]

    missing, unexpected = model.load_state_dict(raw, strict=False)
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing[:5]: {missing[:5]}")
    if len(missing) > 0:
        raise ValueError(
            f"ckpt has {len(missing)} missing keys — "
            "architecture mismatch or wrong config. Abort."
        )

    model = model.to(device).eval()
    if device.startswith("cuda"):
        model = model.half()  # FP16 inference
    return model, hypes


def run_eval(
    model: nn.Module,
    hypes: dict,
    n_samples: int,
    device: str,
) -> dict:
    """Run DAIR-V2X val eval and return AP30/50/70."""
    from opencood.data_utils.datasets.intermediate_fusion_dataset import (
        IntermediateFusionDataset,
    )
    from torch.utils.data import DataLoader
    from opencood.tools.inference_utils import get_cav_box
    from opencood.utils.eval_utils import caluclate_tp_fp, eval_final_results

    # Build val dataset
    hypes_tmp = dict(hypes)
    hypes_tmp["validate"] = True  # ensure we use val split
    dataset = IntermediateFusionDataset(hypes_tmp, visualize=False, train=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_batch_test,
        pin_memory=False,
    )

    total = 0
    iou_30 = []
    iou_50 = []
    iou_70 = []

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            if n_samples > 0 and total >= n_samples:
                break

            batch_data = dataset.collate_batch_test([batch_data[0]])
            # Move to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(device)
                    if device.startswith("cuda"):
                        batch_data[key] = batch_data[key].half()

            output_dict = model(batch_data)

            pred_box_tensor, pred_score, gt_box_tensor = get_cav_box(
                output_dict, batch_data, dataset
            )

            if (pred_box_tensor is not None and gt_box_tensor is not None
                    and gt_box_tensor.shape[0] > 0):
                for iou_thresh, iou_list in zip(
                    [0.30, 0.50, 0.70], [iou_30, iou_50, iou_70]
                ):
                    tp, fp, gt = caluclate_tp_fp(
                        pred_box_tensor, pred_score, gt_box_tensor,
                        iou_thresh
                    )
                    iou_list.append((tp, fp, gt))

            total += 1
            if total % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{total}/{n_samples}] elapsed={elapsed:.1f}s")

    # Compute final AP
    ap30 = eval_final_results(iou_30) if iou_30 else 0.0
    ap50 = eval_final_results(iou_50) if iou_50 else 0.0
    ap70 = eval_final_results(iou_70) if iou_70 else 0.0

    return {
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "n_samples": total,
        "elapsed_secs": round(time.time() - t0, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Eval pyramid pruned model AP")
    parser.add_argument("--config-yaml", required=True, help="Path to config.yaml")
    parser.add_argument("--ckpt-path", required=True, help="Path to flat state_dict .pth")
    parser.add_argument("--label", default="", help="Label for this eval point")
    parser.add_argument("--out", required=True, help="Output JSON path (appended)")
    parser.add_argument("--n-samples", type=int, default=1789,
                        help="Max samples to eval (0=all, 1789=full DAIR val)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device: cuda:0 or cuda:5 etc.")
    args = parser.parse_args()

    print(f"=== Pyramid AP eval: {args.label} ===")
    print(f"  config: {args.config_yaml}")
    print(f"  ckpt:   {args.ckpt_path}")
    print(f"  device: {args.device}")
    print(f"  n_samples: {args.n_samples}")

    t_start = time.time()

    # Load model
    try:
        model, hypes = load_model(args.config_yaml, args.ckpt_path, args.device)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Run eval
    result = run_eval(model, hypes, args.n_samples, args.device)
    result["label"] = args.label
    result["ckpt_path"] = args.ckpt_path
    result["config_yaml"] = args.config_yaml
    result["total_secs"] = round(time.time() - t_start, 2)

    print(f"\n=== RESULT: {args.label} ===")
    print(f"  AP30={result['ap30']:.6f}  AP50={result['ap50']:.6f}  "
          f"AP70={result['ap70']:.6f}  n={result['n_samples']}  "
          f"elapsed={result['elapsed_secs']:.1f}s")

    # Append to output JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if isinstance(existing, list):
            existing.append(result)
        else:
            existing = [existing, result]
    else:
        existing = [result]
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"  → saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
