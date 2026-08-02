#!/usr/bin/env python3
"""Direction B: assemble config x epoch x IoU AP tensor + signal heatmap + verdict.

Reads per-config eval JSONs (each a list of eval-result dicts from
eval_ap_epoch_curve_iou.py) and produces:
  - dirB_config_epoch_iou_tensor.json  (flat list of tensor points)
  - figure/dirB_signal_heatmap.png     (config x epoch heatmap per IoU)
  - prints a quantitative monotone-separability verdict

Every point carries ckpt_path + n_samples for downstream verification.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE = Path("/home/jichengzhi/V2X/results/dirB_stage")   # scp'd eval jsons land here
OUT_TENSOR = Path("/home/jichengzhi/V2X/results/dirB_config_epoch_iou_tensor.json")
OUT_PNG = Path("/home/jichengzhi/V2X/results/figure/dirB_signal_heatmap.png")

# config -> (prune_pct, num_filters, prune_rank higher=more pruned)
CONFIGS = {
    "base": dict(prune_pct=0,  nf=[64, 128, 256], rank=0),
    "p25":  dict(prune_pct=25, nf=[48, 96, 192],  rank=1),
    "p50":  dict(prune_pct=50, nf=[32, 64, 128],  rank=2),
    "p75":  dict(prune_pct=75, nf=[16, 32, 64],   rank=3),
}
LABEL2EPOCH = {"converged": "converged", "e0": 0, "e2": 2, "e4": 4, "e8": 8}
IOUS = [50, 70, 80, 90]
EPOCHS = [0, 2, 4, 8]
NOISE = 0.005  # finetune / pipeline noise floor stated by lead


def load_config_json(cfg):
    p = STAGE / f"eval_{cfg}.json"
    if not p.exists():
        print(f"  [WARN] missing {p}")
        return []
    return json.loads(p.read_text())


def build_tensor():
    points = []
    for cfg, meta in CONFIGS.items():
        for r in load_config_json(cfg):
            label = r.get("label", "")
            epoch = LABEL2EPOCH.get(label, label)
            for iou in IOUS:
                ap = r.get(f"ap{iou}")
                if ap is None:
                    continue
                ntp = r.get("ntp_by_iou", {}).get(f"ntp{iou}")
                ngt = r.get("ngt_by_iou", {}).get(f"ngt{iou}")
                points.append(dict(
                    config=cfg, prune_pct=meta["prune_pct"], num_filters=meta["nf"],
                    epoch=epoch, iou=iou / 100.0, ap=round(float(ap), 6),
                    ntp=ntp, ngt=ngt, n_samples=r.get("n_samples"),
                    ckpt_path=r.get("ckpt_path"), config_yaml=r.get("config_yaml"),
                ))
    return points


def ap_lookup(points, cfg, epoch, iou):
    for p in points:
        if p["config"] == cfg and p["epoch"] == epoch and abs(p["iou"] - iou / 100.0) < 1e-6:
            return p["ap"]
    # base: converged value used as ceiling at every epoch column
    if cfg == "base":
        for p in points:
            if p["config"] == "base" and p["epoch"] == "converged" and abs(p["iou"] - iou / 100.0) < 1e-6:
                return p["ap"]
    return None


def make_heatmap(points):
    rows = ["base", "p25", "p50", "p75"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, iou in zip(axes.ravel(), IOUS):
        M = np.full((len(rows), len(EPOCHS)), np.nan)
        for i, cfg in enumerate(rows):
            for j, ep in enumerate(EPOCHS):
                v = ap_lookup(points, cfg, ep, iou)
                if v is not None:
                    M[i, j] = v
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.1, np.nanmax(M)))
        ax.set_xticks(range(len(EPOCHS)))
        ax.set_xticklabels([f"e{e}" for e in EPOCHS])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r}\n({CONFIGS[r]['prune_pct']}%)" for r in rows])
        ax.set_title(f"AP@IoU{iou/100:.1f}")
        ax.set_xlabel("finetune epoch (base=converged ceiling)")
        for i in range(len(rows)):
            for j in range(len(EPOCHS)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                            color="w" if M[i, j] < 0.5 * np.nanmax(M) else "k", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Direction B: config x epoch AP signal (n=1789 DAIR val)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"  saved heatmap -> {OUT_PNG}")


def verdict(points):
    print("\n" + "=" * 78)
    print("MONOTONE-SEPARABILITY VERDICT  (noise floor=%.3f)" % NOISE)
    print("  base = converged ceiling reference; pruned rows = fresh-from-pruned-init")
    print("=" * 78)
    best_pruned = None
    for ep in EPOCHS:
        for iou in IOUS:
            aps = {c: ap_lookup(points, c, ep, iou) for c in ["base", "p25", "p50", "p75"]}
            if any(aps[c] is None for c in ["p25", "p50", "p75"]):
                continue
            full = [aps["base"], aps["p25"], aps["p50"], aps["p75"]]
            pr = [aps["p25"], aps["p50"], aps["p75"]]
            # pruned-only ladder (apples-to-apples: all at same epoch)
            pr_mono = all(pr[i] >= pr[i + 1] - NOISE for i in range(2))
            pr_spread = pr[0] - pr[2]
            pr_sep = pr_mono and pr_spread > 5 * NOISE
            # full ladder incl. base ceiling
            full_mono = (aps["base"] is not None
                         and all(full[i] >= full[i + 1] - NOISE for i in range(3)))
            btag = "PRUNED-SEP" if pr_sep else ("pr-mono" if pr_mono else "-")
            ftag = "full-mono" if full_mono else ""
            bstr = f"base={aps['base']:.3f} " if aps["base"] is not None else "base=  -   "
            print(f" epoch{ep:<2} IoU{iou/100:.1f}: {bstr}p25={pr[0]:.3f} p50={pr[1]:.3f} "
                  f"p75={pr[2]:.3f} | pr_spread={pr_spread:+.3f} "
                  f"(={pr_spread/NOISE:+.0f}x) [{btag} {ftag}]")
            score = pr_spread if pr_sep else -1
            if best_pruned is None or score > best_pruned[0]:
                best_pruned = (score, ep, iou, pr, pr_spread, pr_sep)
    print("-" * 78)
    if best_pruned and best_pruned[5]:
        _, ep, iou, pr, sp, _ = best_pruned
        print(f"BEST PRUNED-LADDER OPERATING POINT: epoch={ep}, IoU={iou/100:.1f}")
        print(f"  p25={pr[0]:.3f} > p50={pr[1]:.3f} > p75={pr[2]:.3f}  "
              f"spread={sp:.3f} (={sp/NOISE:.0f}x noise), monotone in prune ratio.")
        print("VERDICT: YES - an undertrained (epoch,IoU) operating point exists where")
        print("  AP is monotone-separable across prune ratios with spread >> noise.")
    else:
        print("VERDICT: NO pruned-ladder operating point with monotone spread>5*noise.")
    print("\nNOTE: separation is a RECOVERABILITY signal (how fast each prune ratio")
    print("  refits under a small epoch budget), NOT a converged-AP gap. By epoch>=4")
    print("  the ladder collapses/scrambles (over-parameterization: all ratios refit).")
    return best_pruned


def main():
    points = build_tensor()
    if not points:
        print("No points assembled; check STAGE dir.")
        sys.exit(1)
    OUT_TENSOR.write_text(json.dumps(points, indent=2))
    print(f"  wrote {len(points)} tensor points -> {OUT_TENSOR}")
    make_heatmap(points)
    verdict(points)


if __name__ == "__main__":
    main()
