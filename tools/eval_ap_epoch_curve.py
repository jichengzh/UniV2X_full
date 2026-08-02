#!/usr/bin/env python3
"""
Zero-shot + AP-vs-epoch evaluator for Pyramid p50 pruning.
Uses HEAL's standard intermediate-fusion inference pipeline.

Usage on H800 (GPU5):
    CUDA_VISIBLE_DEVICES=5 \
    PYTHONPATH=/data/jichengzhi_v2x/t2lib:/exdata/jichengzhi/heal_research/HEAL \
    python3 /home/jichengzhi/V2X/tools/eval_ap_epoch_curve.py \
        --config-yaml /exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/config.yaml \
        --evals \
          "zeroshot_ep0:/home/jichengzhi/V2X/results/zeroshot_p50_exp/net_epoch_bestval_at23.pth" \
          "finetune_ep1:/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch24.pth" \
          "finetune_ep2:/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch25.pth" \
          "finetune_ep4:/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch27.pth" \
          "finetune_ep8:/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch31.pth" \
          "converged_ep29bv:/exdata/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at29.pth" \
        --out /home/jichengzhi/V2X/results/ap_epoch_curve_p50.json \
        --out-png /home/jichengzhi/V2X/results/figure/ap_epoch_curve_p50.png \
        --base-ap70 0.630864

Eval discipline:
- Loads ckpt with strict=False, verifies missing_keys == 0 (printed to stdout)
- Runs on DAIR val_1789 (n_samples=1789)
- FP16 inference (same as trained)
- Results appended to JSON incrementally
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# ── HEAL path bootstrap ───────────────────────────���───────────────────────────
_HEAL = "/exdata/jichengzhi/heal_research/HEAL"
if _HEAL not in sys.path:
    sys.path.insert(0, _HEAL)


def load_model(config_yaml: str, ckpt_path: str, device: str, half: bool = False):
    """Build HEAL HeterPyramidCollab from pruned config + load flat ckpt.

    Uses FP32 by default (half=False) for standard HEAL inference.py behavior.
    Prints '[ckpt] missing=0 unexpected=N' and RAISES ValueError if missing != 0.
    """
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.models.heter_pyramid_collab import HeterPyramidCollab

    hypes = load_yaml(config_yaml)
    model = HeterPyramidCollab(hypes["model"]["args"])

    raw = torch.load(ckpt_path, map_location="cpu")
    # Unwrap wrapped format
    if isinstance(raw, dict) and len(raw) == 1:
        first_key = next(iter(raw))
        if first_key in ("model_state_dict", "state_dict"):
            raw = raw[first_key]
            print(f"  [ckpt] unwrapped '{first_key}' wrapper")

    # If weights were saved in fp16 (from --half training), cast to fp32 for eval
    raw_fp32 = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
            raw_fp32[k] = v.float()
        else:
            raw_fp32[k] = v

    missing, unexpected = model.load_state_dict(raw_fp32, strict=False)
    print(f"  [ckpt] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  [ckpt] missing_modules: {sorted({k.split('.')[0] for k in missing})}")
    if len(missing) > 0:
        raise ValueError(
            f"[FATAL] {len(missing)} missing keys — architecture mismatch! "
            f"Ensure config num_filters matches the pruned ckpt. Abort."
        )

    model = model.to(device).eval()
    return model, hypes


def run_eval_on_dataset(
    model,
    hypes: dict,
    device: str,
    n_samples: int,
    half: bool,
    tmp_dir: str,
) -> dict:
    """Run HEAL intermediate-fusion eval on DAIR val.

    Returns dict with ap30, ap50, ap70, n_samples, elapsed_secs.
    Uses train_utils.to_device() to recursively move batch_data to device
    (handles nested dict/list of tensors, skips non-tensor scalars).
    """
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import inference_utils
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils
    from torch.utils.data import DataLoader

    # Build val dataset (train=False → uses test_dir / validate_dir from config)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_batch_test,
        pin_memory=False,
        drop_last=False,
    )

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    total = 0
    t0 = time.time()

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if n_samples > 0 and total >= n_samples:
                break
            if batch_data is None:
                continue

            # Recursively move all tensors to device
            # (handles nested dict/list, skips int/float/str)
            batch_data = to_device(batch_data, device)

            ret = inference_utils.inference_intermediate_fusion(
                batch_data, model, dataset
            )
            # Handle both tuple and dict return formats
            if isinstance(ret, dict):
                pred_box_tensor = ret.get("pred_box_tensor")
                pred_score = ret.get("pred_score")
                gt_box_tensor = ret.get("gt_box_tensor")
            else:
                pred_box_tensor, pred_score, gt_box_tensor = ret

            if pred_box_tensor is not None and gt_box_tensor is not None:
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.3
                )
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5
                )
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7
                )

            total += 1
            if total % 300 == 0:
                elapsed = time.time() - t0
                eta_min = (elapsed / total * n_samples / 60) if total > 0 else 0
                print(f"  [{total}/{n_samples}] elapsed={elapsed:.1f}s  "
                      f"~{eta_min:.1f}min total")

    # Final AP
    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, tmp_dir)
    elapsed = time.time() - t0
    return {
        "ap30": float(ap_30),
        "ap50": float(ap_50),
        "ap70": float(ap_70),
        "n_samples": total,
        "elapsed_secs": round(elapsed, 2),
    }


def eval_one_ckpt(
    label: str,
    ckpt_path: str,
    config_yaml: str,
    device: str,
    n_samples: int,
    half: bool,
    results: list,
    out_path: Path,
) -> Optional[dict]:
    """Evaluate one checkpoint and append result to results list."""
    print(f"\n{'=' * 70}")
    print(f"EVAL: {label}")
    print(f"  ckpt:   {ckpt_path}")
    print(f"  device: {device}  half={half}  n={n_samples}")

    if not Path(ckpt_path).exists():
        print(f"  [SKIP] ckpt not found: {ckpt_path}")
        return None

    t0 = time.time()

    # Load model (verifies missing=0)
    try:
        model, hypes = load_model(config_yaml, ckpt_path, device, half)
    except ValueError as e:
        print(f"  [ERROR] {e}")
        return None

    # Eval
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            result = run_eval_on_dataset(model, hypes, device, n_samples, half, tmp_dir)
        except Exception as e:
            print(f"  [ERROR during eval] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

    result["label"] = label
    result["ckpt_path"] = ckpt_path
    result["config_yaml"] = config_yaml
    result["total_secs"] = round(time.time() - t0, 2)

    print(f"\n  ✓ {label}:")
    print(f"    AP30={result['ap30']:.6f}  AP50={result['ap50']:.6f}  "
          f"AP70={result['ap70']:.6f}")
    print(f"    n={result['n_samples']}  elapsed={result['elapsed_secs']:.1f}s")

    results.append(result)

    # Save incrementally (in case of crash)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  → saved to {out_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


def make_ap_curve_plot(results: list, out_png: Path, base_ap70: float, base_ap50: float):
    """Plot AP50 + AP70 vs finetune epoch with base reference lines."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot")
        return

    # Parse epoch numbers
    epoch_map = []
    for r in results:
        lbl = r["label"]
        if "zeroshot" in lbl or "ep0" in lbl:
            ep = 0
        elif "ep" in lbl:
            # e.g. finetune_ep4 → 4
            try:
                ep = int(lbl.rstrip("bv").split("ep")[-1])
            except ValueError:
                ep = None
        else:
            ep = None
        epoch_map.append((ep, r))

    # Separate zero-shot + finetune points from converged (bestval) points
    finetune_pts = [(ep, r) for ep, r in epoch_map if ep is not None]
    finetune_pts.sort(key=lambda x: x[0])
    other_pts = [(ep, r) for ep, r in epoch_map if ep is None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["ap50", "ap70"]):
        base_ref = base_ap50 if metric == "ap50" else base_ap70
        base_name = "base AP50" if metric == "ap50" else "base AP70"
        base_val = 0.791 if metric == "ap50" else base_ap70

        x = [ep for ep, _ in finetune_pts]
        y = [r[metric] for _, r in finetune_pts]
        ax.plot(x, y, "o-", color="steelblue" if metric == "ap50" else "crimson",
                linewidth=2, markersize=8, label=metric.upper())

        ax.axhline(y=base_val, linestyle="--", color="gray", alpha=0.7,
                   label=f"{base_name}={base_val:.3f}")

        # Mark zero-shot
        zs = [(ep, r) for ep, r in finetune_pts if ep == 0]
        if zs:
            ax.axvline(x=0, linestyle=":", color="orange", alpha=0.5, label="zero-shot")

        ax.set_xlabel("Finetune epochs from zero-shot")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Pyramid p50: {metric.upper()} vs finetune epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

    plt.suptitle("A/B/C hypothesis: zero-shot vs converged AP (Pyramid p50 pruning, DAIR val 1789)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"\n→ Plot saved: {out_png}")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="AP-vs-epoch curve for Pyramid p50 pruning")
    p.add_argument("--config-yaml", required=True, help="Pruned config.yaml (num_filters=[32,64,128])")
    p.add_argument("--evals", nargs="+", required=True,
                   help="'label:ckpt_path' pairs, evaluated in order")
    p.add_argument("--out", required=True, help="Output JSON results path")
    p.add_argument("--out-png", default="", help="Output AP curve plot PNG")
    p.add_argument("--n-samples", type=int, default=1789,
                   help="DAIR val samples (1789 = full val set)")
    p.add_argument("--device", default="cuda:0",
                   help="Device: 'cuda:0' after CUDA_VISIBLE_DEVICES is set")
    p.add_argument("--no-half", action="store_true",
                   help="Use FP32 instead of FP16")
    p.add_argument("--base-ap70", type=float, default=0.630864,
                   help="Base model AP70 for plot reference line")
    p.add_argument("--base-ap50", type=float, default=0.791041,
                   help="Base model AP50 for plot reference line")
    args = p.parse_args()

    device = args.device
    half = not args.no_half
    out_path = Path(args.out)

    print(f"=== Pyramid p50 pruning: AP-vs-epoch experiment ===")
    print(f"  device={device}  half={half}  n_samples={args.n_samples}")
    print(f"  config: {args.config_yaml}")
    print(f"  output: {args.out}")
    print(f"  n_evals: {len(args.evals)}")
    print()

    # Load existing results if file exists (for resuming)
    results = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            existing_labels = {r["label"] for r in results}
            print(f"  [resume] Found {len(results)} existing results: {existing_labels}")
        except Exception:
            results = []
            existing_labels = set()
    else:
        existing_labels = set()

    # Run evals
    for eval_spec in args.evals:
        label, ckpt_path = eval_spec.split(":", 1)
        if label in existing_labels:
            print(f"  [skip] Already computed: {label}")
            continue
        eval_one_ckpt(
            label=label,
            ckpt_path=ckpt_path,
            config_yaml=args.config_yaml,
            device=device,
            n_samples=args.n_samples,
            half=half,
            results=results,
            out_path=out_path,
        )

    # Final summary table
    print(f"\n{'=' * 70}")
    print("=== FINAL AP TABLE ===")
    print(f"{'Label':<30} {'AP30':>10} {'AP50':>10} {'AP70':>10}  {'N':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<30} {r['ap30']:>10.6f} {r['ap50']:>10.6f} {r['ap70']:>10.6f}  {r['n_samples']:>6}")

    # Base reference
    print(f"{'base_fp16 (reference)':30} {'0.833159':>10} {'0.791041':>10} {'0.630864':>10}  {'1789':>6}")

    # Plot
    if args.out_png:
        make_ap_curve_plot(results, Path(args.out_png), args.base_ap70, args.base_ap50)

    # A/B/C analysis
    zs = next((r for r in results if "zeroshot" in r["label"]), None)
    conv = next((r for r in results if "converged" in r["label"] or "bestval" in r["label"]), None)
    if zs and conv:
        delta_ap70 = abs(conv["ap70"] - zs["ap70"])
        delta_ap50 = abs(conv["ap50"] - zs["ap50"])
        base_ap70 = args.base_ap70
        print(f"\n=== A/B/C ANALYSIS ===")
        print(f"  zero-shot AP70 = {zs['ap70']:.6f}")
        print(f"  converged AP70 = {conv['ap70']:.6f}")
        print(f"  delta AP70     = {delta_ap70:.6f}")
        print(f"  base AP70      = {base_ap70:.6f}")
        print()
        if zs["ap70"] > 0.60:
            print("  → STRONG B support: zero-shot AP70 >> 0 (model is near-loss after pruning without finetune)")
        elif zs["ap70"] > 0.40:
            print("  → MODERATE B support: zero-shot AP70 meaningfully high")
        elif zs["ap70"] < 0.10:
            print("  → REFUTES B: zero-shot crashes (needs finetune for recovery)")
        else:
            print("  → Ambiguous: zero-shot AP is non-trivial but significantly below converged")

    return 0


if __name__ == "__main__":
    sys.exit(main())
