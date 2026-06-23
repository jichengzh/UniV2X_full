"""Plan v5 Phase A.1 — Structural channel prune g8 baseline.

Goal: produce 4 pruned ckpts (plane_factor=0.75/0.50/0.25/0.125), each ready
for FT=8 finetune. The pruner walks pyramid_backbone.resnet (ResNeXt Bottleneck,
expansion=1, groups=8, wpg=16) and truncates each conv along channel dim by
L1-norm top-N selection, propagating selected channels across cross-layer
boundaries to keep shapes consistent.

Run:
    python scripts/phase2/plan5_phaseA_prune_g8.py
Output:
    /tmp/plan5_phaseA_ckpts/g8_p{48,32,16,8}_pruned_unfinetuned.pth
    plan5_phaseA_prune_log.json (shape diff per layer per plane)

Limitations (documented):
- We DO NOT re-prune encoder_m1 / backbone_m1 / heads — only pyramid_backbone
  (the main computational hotspot per plan v4 §LAT analysis).
- deblocks (upsample to 128 ch fixed) input dim is reduced; output unchanged.
- shrink_conv (3*128 → 384) is unaffected (deblock outputs are still 128).
- single_head_{i} input dim is reduced from new_planes[i].
- Cross-layer channel alignment is enforced within a stage; first-block input
  dim follows previous-stage's last-block output (selected channels).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
OUT_DIR = Path("/tmp/plan5_phaseA_ckpts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

G8_CKPT = Path(
    "/home/jichengzhi/heal_research/HEAL/opencood/logs/"
    "Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/net_epoch_bestval_at19.pth"
)

PLANE_FACTORS = [
    ("p48", 48 / 64),
    ("p32", 32 / 64),
    ("p16", 16 / 64),
    ("p8", 8 / 64),
]

GROUPS = 8
WPG = 16
NUM_FILTERS_BASE = [64, 128, 256]
LAYER_NUMS = [3, 5, 8]
DEBLOCK_OUTPUT = 128


def compute_width(plane: int) -> int:
    """ResNeXt width = int(plane * wpg / 64) * groups."""
    return max(GROUPS, int(plane * WPG / 64) * GROUPS)


def select_topn_by_l1(weight: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    """Return indices of top-n channels by L1 norm along given dim."""
    other_dims = tuple(d for d in range(weight.ndim) if d != dim)
    l1 = weight.abs().sum(dim=other_dims) if other_dims else weight.abs()
    n = min(n, l1.numel())
    return torch.topk(l1, k=n).indices.sort().values


def slice_tensor(t: torch.Tensor, idx_per_dim: Dict[int, torch.Tensor]) -> torch.Tensor:
    """Slice tensor selecting given indices per dim."""
    out = t
    for dim in sorted(idx_per_dim.keys()):
        out = torch.index_select(out, dim, idx_per_dim[dim])
    return out


def prune_block(
    sd: Dict[str, torch.Tensor],
    prefix: str,
    new_plane: int,
    new_width: int,
    in_plane_idx: torch.Tensor,
    out_plane_idx: torch.Tensor,
    is_first_block: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Prune one Bottleneck block. Returns (new_sub_sd, width_idx for downstream).

    Block layout (ResNeXt Bottleneck, expansion=1):
        conv1: 1x1 (width, plane_in, 1, 1) → bn1 (width)
        conv2: 3x3 grouped (width, width/groups, 3, 3) → bn2 (width)
        conv3: 1x1 (plane_out, width, 1, 1)              → bn3 (plane_out)
      downsample (only on first block of stage): conv 1x1 (plane_out, plane_in)
    """
    new_sub: Dict[str, torch.Tensor] = {}
    conv1_w = sd[f"{prefix}.conv1.weight"]
    width_idx = select_topn_by_l1(conv1_w, new_width, dim=0)
    new_sub[f"{prefix}.conv1.weight"] = slice_tensor(conv1_w, {0: width_idx, 1: in_plane_idx})

    for bn_key in ("bn1.weight", "bn1.bias", "bn1.running_mean", "bn1.running_var"):
        full = f"{prefix}.{bn_key}"
        new_sub[full] = sd[full][width_idx]
    nbtk = f"{prefix}.bn1.num_batches_tracked"
    if nbtk in sd:
        new_sub[nbtk] = sd[nbtk].clone()

    conv2_w = sd[f"{prefix}.conv2.weight"]
    per_group_in = new_width // GROUPS
    full_per_group_in = conv2_w.shape[1]
    assert full_per_group_in == sd[f"{prefix}.conv1.weight"].shape[0] // GROUPS, (
        f"conv2 grouped in mismatch at {prefix}"
    )
    out_per_group = new_width // GROUPS
    new_conv2 = torch.empty(
        (new_width, per_group_in, conv2_w.shape[2], conv2_w.shape[3]),
        dtype=conv2_w.dtype,
    )
    full_width = conv2_w.shape[0]
    full_out_per_group = full_width // GROUPS
    width_idx_sorted = width_idx.tolist()
    for g in range(GROUPS):
        new_out_lo = g * out_per_group
        new_out_hi = (g + 1) * out_per_group
        new_out_idx = [i for i in width_idx_sorted
                       if g * full_out_per_group <= i < (g + 1) * full_out_per_group][:out_per_group]
        while len(new_out_idx) < out_per_group:
            new_out_idx.append(g * full_out_per_group)
        new_out_idx_t = torch.tensor(new_out_idx, dtype=torch.long)
        new_conv2[new_out_lo:new_out_hi] = conv2_w[new_out_idx_t, :per_group_in]
    new_sub[f"{prefix}.conv2.weight"] = new_conv2

    for bn_key in ("bn2.weight", "bn2.bias", "bn2.running_mean", "bn2.running_var"):
        full = f"{prefix}.{bn_key}"
        new_sub[full] = sd[full][width_idx]
    nbtk = f"{prefix}.bn2.num_batches_tracked"
    if nbtk in sd:
        new_sub[nbtk] = sd[nbtk].clone()

    conv3_w = sd[f"{prefix}.conv3.weight"]
    new_sub[f"{prefix}.conv3.weight"] = slice_tensor(conv3_w, {0: out_plane_idx, 1: width_idx})

    for bn_key in ("bn3.weight", "bn3.bias", "bn3.running_mean", "bn3.running_var"):
        full = f"{prefix}.{bn_key}"
        new_sub[full] = sd[full][out_plane_idx]
    nbtk = f"{prefix}.bn3.num_batches_tracked"
    if nbtk in sd:
        new_sub[nbtk] = sd[nbtk].clone()

    if is_first_block and f"{prefix}.downsample.0.weight" in sd:
        ds_w = sd[f"{prefix}.downsample.0.weight"]
        new_sub[f"{prefix}.downsample.0.weight"] = slice_tensor(ds_w, {0: out_plane_idx, 1: in_plane_idx})
        for bn_key in ("downsample.1.weight", "downsample.1.bias",
                       "downsample.1.running_mean", "downsample.1.running_var"):
            full = f"{prefix}.{bn_key}"
            if full in sd:
                new_sub[full] = sd[full][out_plane_idx]
        nbtk = f"{prefix}.downsample.1.num_batches_tracked"
        if nbtk in sd:
            new_sub[nbtk] = sd[nbtk].clone()

    return new_sub, width_idx


def prune_pyramid(sd: Dict[str, torch.Tensor], plane_factor: float) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Walk pyramid_backbone.resnet and truncate channels."""
    new_planes = [max(GROUPS, int(p * plane_factor)) for p in NUM_FILTERS_BASE]
    new_widths = [compute_width(p) for p in new_planes]
    new_sd: Dict[str, torch.Tensor] = {}
    log = {"plane_factor": plane_factor, "new_planes": new_planes, "new_widths": new_widths, "layers": {}}

    pyramid_keys = [k for k in sd.keys() if k.startswith("pyramid_backbone")]
    non_pyramid_keys = [k for k in sd.keys() if not k.startswith("pyramid_backbone")]

    for k in non_pyramid_keys:
        new_sd[k] = sd[k].clone()

    in_plane_idx = torch.arange(NUM_FILTERS_BASE[0], dtype=torch.long)
    stage_final_out_idx: Dict[int, torch.Tensor] = {}

    for stage_idx in range(3):
        plane_full = NUM_FILTERS_BASE[stage_idx]
        plane_new = new_planes[stage_idx]
        width_new = new_widths[stage_idx]
        n_blocks = LAYER_NUMS[stage_idx]

        for block_idx in range(n_blocks):
            prefix = f"pyramid_backbone.resnet.layer{stage_idx}.{block_idx}"
            block_keys = [k for k in pyramid_keys if k.startswith(prefix + ".")]
            if not block_keys:
                continue

            conv3_w = sd[f"{prefix}.conv3.weight"]
            out_plane_idx = select_topn_by_l1(conv3_w, plane_new, dim=0)

            block_sd, width_idx = prune_block(
                sd, prefix, plane_new, width_new,
                in_plane_idx=in_plane_idx,
                out_plane_idx=out_plane_idx,
                is_first_block=(block_idx == 0),
            )
            new_sd.update(block_sd)
            log["layers"][prefix] = {
                "plane_full": plane_full, "plane_new": plane_new,
                "width_full": compute_width(plane_full), "width_new": width_new,
                "selected_out_planes": out_plane_idx.tolist()[:10] + ["..."] if plane_new > 10 else out_plane_idx.tolist(),
            }
            in_plane_idx = out_plane_idx
            if block_idx == n_blocks - 1:
                stage_final_out_idx[stage_idx] = out_plane_idx

    for level_idx in range(3):
        prefix = f"pyramid_backbone.deblocks.{level_idx}"
        deblock_conv_key = f"{prefix}.0.weight"
        if deblock_conv_key in sd:
            full_w = sd[deblock_conv_key]
            sel = stage_final_out_idx.get(level_idx, torch.arange(new_planes[level_idx], dtype=torch.long))
            new_sd[deblock_conv_key] = slice_tensor(full_w, {0: sel})
            for bn_key in ("1.weight", "1.bias", "1.running_mean", "1.running_var"):
                full = f"{prefix}.{bn_key}"
                if full in sd:
                    new_sd[full] = sd[full].clone()
            nbtk = f"{prefix}.1.num_batches_tracked"
            if nbtk in sd:
                new_sd[nbtk] = sd[nbtk].clone()

    for level_idx in range(3):
        head_key = f"pyramid_backbone.single_head_{level_idx}.weight"
        head_bias = f"pyramid_backbone.single_head_{level_idx}.bias"
        if head_key in sd:
            full_w = sd[head_key]
            sel = stage_final_out_idx.get(level_idx, torch.arange(new_planes[level_idx], dtype=torch.long))
            new_sd[head_key] = slice_tensor(full_w, {1: sel})
            if head_bias in sd:
                new_sd[head_bias] = sd[head_bias].clone()

    return new_sd, log


def main() -> int:
    print(f"[Plan v5 Phase A.1] Loading baseline ckpt: {G8_CKPT}")
    baseline_sd = torch.load(G8_CKPT, map_location="cpu", weights_only=False)
    print(f"  baseline ckpt: {len(baseline_sd)} tensors, total params: "
          f"{sum(v.numel() for v in baseline_sd.values() if hasattr(v, 'numel'))/1e6:.2f}M")

    full_log: Dict[str, dict] = {}
    for tag, pf in PLANE_FACTORS:
        print(f"\n[plane_factor={pf:.4f} ({tag})] pruning ...")
        pruned_sd, log = prune_pyramid(baseline_sd, pf)
        n_params = sum(v.numel() for v in pruned_sd.values() if hasattr(v, "numel"))
        log["total_params"] = n_params
        log["total_params_M"] = round(n_params / 1e6, 3)
        out_path = OUT_DIR / f"g8_{tag}_pruned_unfinetuned.pth"
        torch.save(pruned_sd, out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  saved {out_path} ({size_mb:.2f} MB, {n_params/1e6:.2f}M params)")
        log["output_path"] = str(out_path)
        log["output_size_mb"] = round(size_mb, 2)
        full_log[tag] = log

    log_path = DATA_DIR / "plan5_phaseA_prune_log.json"
    log_path.write_text(json.dumps(full_log, ensure_ascii=False, indent=2))
    print(f"\n[Plan v5 Phase A.1] wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
