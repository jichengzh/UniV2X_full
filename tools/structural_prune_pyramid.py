"""Phase B.1 — L1-norm structural channel pruning for Pyramid_DAIR_m1_base.

Reduces ``fusion_backbone.num_filters`` from [64, 128, 256] to [32, 64, 128]
(50% reduction). This is REAL structural pruning — output ckpt has smaller
weights, NOT mask-based. ResNeXt grouped-conv (groups=32) constraint preserved.

Pipeline
--------
1. Load original Pyramid_DAIR_m1_base ckpt + config
2. Build NEW model with smaller fusion_backbone.num_filters
3. For each stage i in pyramid_backbone.resnet:
   - Compute L1 norm over output channels for last conv3 of each Bottleneck
     within the stage; sum L1 across all blocks → top-half indices = stage_keep
   - Per-block: select conv1 output (= width_new) by L1 of conv1.weight
   - Per-block: select conv2 output (grouped, top 2 per group of 4) by L1
   - conv3 output = stage_keep
4. Apply selections to copy weights from old ckpt to new model
5. Save new ckpt + new config yaml

Outputs
-------
    checkpoints/Pyramid_DAIR_m1_pruned50/
        net_epoch_bestval_at23.pth   (pruned ckpt, ~5.5 MB)
        config.yaml                  (smaller num_filters)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))

sys.path.insert(0, str(REPO_ROOT))
from tools.export_onnx_pyramid import build_pyramid_from_ckpt  # noqa: E402
from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.models.heter_pyramid_collab import HeterPyramidCollab  # noqa: E402


# ---------------------------------------------------------------------------
# L1 selection helpers
# ---------------------------------------------------------------------------

def l1_along_dim(weight: torch.Tensor, dim: int) -> torch.Tensor:
    """Sum |weight| over all dims except `dim`. Returns 1D tensor of length weight.shape[dim]."""
    other = [d for d in range(weight.ndim) if d != dim]
    return weight.detach().abs().sum(dim=tuple(other))


def top_indices(scores: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Return sorted indices of top-N scores."""
    return torch.topk(scores, n_keep).indices.sort().values


def top_per_group(scores: torch.Tensor, n_groups: int, keep_per_group: int) -> torch.Tensor:
    """Per-group top-K selection. scores: (n_groups * old_per_group,)."""
    g = scores.view(n_groups, -1)
    keep = []
    for gi in range(n_groups):
        idx = torch.topk(g[gi], keep_per_group).indices
        keep.append(idx + gi * g.shape[1])
    return torch.cat(keep).sort().values


# ---------------------------------------------------------------------------
# Per-stage selection
# ---------------------------------------------------------------------------

def select_stage_output(layer: torch.nn.Module, n_keep: int) -> torch.Tensor:
    """Sum L1 across all conv3 weights within a stage's Bottlenecks → top-N.

    layer: nn.Sequential of Bottleneck blocks.
    Each Bottleneck has conv3 (1x1, width → planes); we pick top-N planes channels.
    """
    accum = None
    for block in layer:
        c3 = block.conv3.weight   # (planes, width, 1, 1)
        s = l1_along_dim(c3, dim=0)
        accum = s if accum is None else accum + s
    return top_indices(accum, n_keep)


def select_block_widths(block, width_new_per_group, groups):
    """For one Bottleneck, return:
       - conv1_out_idx: indices to keep for conv1 output (= width_new)
       - conv2_out_idx: indices to keep for conv2 output (per-group selection)
    """
    c1 = block.conv1.weight   # (width_old, in_old, 1, 1)
    width_old = c1.shape[0]
    n_keep_w = width_new_per_group * groups
    # conv1 output: pick top width_new by conv1 L1
    s1 = l1_along_dim(c1, dim=0)
    conv1_out_idx = top_indices(s1, n_keep_w)

    c2 = block.conv2.weight   # (width_old, width_old/groups, 3, 3) for grouped conv
    # For grouped conv, weight shape is (out_ch, out_ch/groups, kh, kw)
    # output channel L1: sum over (in_per_group, kh, kw)
    s2 = l1_along_dim(c2, dim=0)   # (width_old,)
    conv2_out_idx = top_per_group(s2, groups, width_new_per_group)
    return conv1_out_idx, conv2_out_idx


# ---------------------------------------------------------------------------
# Weight transfer: old → new with selections
# ---------------------------------------------------------------------------

def slice_conv_weight(w_old: torch.Tensor, out_idx=None, in_idx=None) -> torch.Tensor:
    """Slice conv weight (out_ch, in_ch, kh, kw) by selecting out_idx / in_idx."""
    w = w_old
    if out_idx is not None:
        w = w[out_idx]
    if in_idx is not None:
        if w.ndim >= 2 and len(in_idx) <= w.shape[1]:
            w = w[:, in_idx]
    return w.contiguous().clone()


def slice_grouped_conv2(w_old: torch.Tensor, out_idx: torch.Tensor,
                        groups: int, width_new_per_group: int) -> torch.Tensor:
    """Slice conv2 grouped weight (width_old, width_old/groups, kh, kw) preserving group structure.

    out_idx is global ch indices; we map them per-group and slice both
    output channels and the per-group input slot (which group input ch are kept).
    For groups=32 with width_old=128, each group has 4 input ch (in_per_group=4).
    Going to width_new=64, each group keeps 2 input ch.

    Since input channels of conv2 within group g correspond to output channels of
    conv1 within group g — we keep the input slot indices that align with conv1
    output selection per-group.
    """
    width_old = w_old.shape[0]
    in_per_group_old = w_old.shape[1]
    in_per_group_new = width_new_per_group   # Same as out per group for this prune
    new_w = torch.zeros((groups * width_new_per_group, in_per_group_new, *w_old.shape[2:]),
                        dtype=w_old.dtype)
    out_idx_sorted, _ = torch.sort(out_idx)
    # For each new output ch j, find which group g and slot s_old it came from
    for new_j, old_j in enumerate(out_idx_sorted):
        g_old = old_j // in_per_group_old
        # All output ch in same group share the same input ch slots
        # We just select top in_per_group_new of in_per_group_old by L1
        in_l1 = w_old[old_j].abs().sum(dim=(-2, -1))  # (in_per_group_old,)
        top_in = torch.topk(in_l1, in_per_group_new).indices.sort().values
        new_w[new_j] = w_old[old_j, top_in]
    return new_w.contiguous()


def transfer_weights(old_model, new_model,
                     num_filters_old: list, num_filters_new: list,
                     groups: int = 32, width_per_group: int = 4,
                     skip_shrink_heads: bool = False):
    """Transfer weights from old PyramidFusion to new (smaller) one with L1 selection."""
    pb_old = old_model.pyramid_backbone
    pb_new = new_model.pyramid_backbone

    # Compute width_per_group_new for each stage
    # Original: width = int(planes * 4 / 64) * groups  (per HEAL Bottleneck formula)
    def width_for(planes):
        return int(planes * width_per_group / 64) * groups
    width_old = [width_for(p) for p in num_filters_old]
    width_new = [width_for(p) for p in num_filters_new]
    width_per_group_new = [w // groups for w in width_new]

    print(f"  Old num_filters {num_filters_old}  widths {width_old}")
    print(f"  New num_filters {num_filters_new}  widths {width_new}  (per-group {width_per_group_new})")

    # 1. Stage output selections (top-N planes channels per stage)
    stage_keeps = []
    for i, n_keep in enumerate(num_filters_new):
        layer_old = getattr(pb_old.resnet, f"layer{i}")
        keep = select_stage_output(layer_old, n_keep)
        stage_keeps.append(keep)
        print(f"  stage{i} output keep: {len(keep)} of {num_filters_old[i]}")

    # 2. Per-stage per-block transfer
    prev_keep = None  # input to layer0 is spatial_features (64 ch, fixed)
    for i in range(len(num_filters_new)):
        layer_old = getattr(pb_old.resnet, f"layer{i}")
        layer_new = getattr(pb_new.resnet, f"layer{i}")
        stage_out = stage_keeps[i]
        for j, (b_old, b_new) in enumerate(zip(layer_old, layer_new)):
            # conv1 output selection
            c1_out, c2_out = select_block_widths(b_old, width_per_group_new[i], groups)
            # conv1: in = inplanes_for_this_block, out = width_new
            #   inplanes = 64 (input to first block of stage 0) or stage_out_prev
            #   For block 0 of stage > 0: inplanes = stage_keeps[i-1]
            #   For block 0 of stage 0: inplanes = 64 (full, no select)
            #   For block j>0: inplanes = stage_keeps[i] (since prev block ended at planes=stage_out)
            if j == 0:
                if i == 0:
                    in_idx = torch.arange(b_old.conv1.weight.shape[1])
                else:
                    in_idx = stage_keeps[i - 1]
            else:
                in_idx = stage_out
            b_new.conv1.weight.data = slice_conv_weight(b_old.conv1.weight.data,
                                                        out_idx=c1_out, in_idx=in_idx)
            # bn1 (per output)
            b_new.bn1.weight.data = b_old.bn1.weight.data[c1_out].clone()
            b_new.bn1.bias.data = b_old.bn1.bias.data[c1_out].clone()
            b_new.bn1.running_mean.data = b_old.bn1.running_mean.data[c1_out].clone()
            b_new.bn1.running_var.data = b_old.bn1.running_var.data[c1_out].clone()

            # conv2: grouped, width → width
            b_new.conv2.weight.data = slice_grouped_conv2(
                b_old.conv2.weight.data, c2_out, groups, width_per_group_new[i],
            )
            b_new.bn2.weight.data = b_old.bn2.weight.data[c2_out].clone()
            b_new.bn2.bias.data = b_old.bn2.bias.data[c2_out].clone()
            b_new.bn2.running_mean.data = b_old.bn2.running_mean.data[c2_out].clone()
            b_new.bn2.running_var.data = b_old.bn2.running_var.data[c2_out].clone()

            # conv3: width → planes; in = c2_out, out = stage_out
            b_new.conv3.weight.data = slice_conv_weight(b_old.conv3.weight.data,
                                                        out_idx=stage_out, in_idx=c2_out)
            b_new.bn3.weight.data = b_old.bn3.weight.data[stage_out].clone()
            b_new.bn3.bias.data = b_old.bn3.bias.data[stage_out].clone()
            b_new.bn3.running_mean.data = b_old.bn3.running_mean.data[stage_out].clone()
            b_new.bn3.running_var.data = b_old.bn3.running_var.data[stage_out].clone()

            # downsample (only block 0 has it)
            if b_old.downsample is not None and b_new.downsample is not None:
                # downsample[0] is conv1x1 (in = inplanes, out = planes)
                ds_in = (torch.arange(b_old.downsample[0].weight.shape[1])
                         if (i == 0) else stage_keeps[i - 1])
                b_new.downsample[0].weight.data = slice_conv_weight(
                    b_old.downsample[0].weight.data, out_idx=stage_out, in_idx=ds_in,
                )
                # downsample[1] is BN (per planes channel)
                ds_bn_old = b_old.downsample[1]
                ds_bn_new = b_new.downsample[1]
                ds_bn_new.weight.data = ds_bn_old.weight.data[stage_out].clone()
                ds_bn_new.bias.data = ds_bn_old.bias.data[stage_out].clone()
                ds_bn_new.running_mean.data = ds_bn_old.running_mean.data[stage_out].clone()
                ds_bn_new.running_var.data = ds_bn_old.running_var.data[stage_out].clone()

        prev_keep = stage_out

    # 3. Deblocks: input ch goes from num_filters_old[i] to num_filters_new[i],
    #    output stays 128. Select input by stage_keeps[i].
    for i in range(len(num_filters_new)):
        deblock_old = pb_old.deblocks[i]
        deblock_new = pb_new.deblocks[i]
        # deblock layout: ConvTranspose2d (or Conv2d) → BN → ReLU
        old_conv = deblock_old[0]
        new_conv = deblock_new[0]
        # ConvTranspose2d weight shape: (in_ch, out_ch, kh, kw)
        # Conv2d weight shape: (out_ch, in_ch, kh, kw)
        if isinstance(old_conv, torch.nn.ConvTranspose2d):
            new_conv.weight.data = old_conv.weight.data[stage_keeps[i]].contiguous().clone()
        else:
            new_conv.weight.data = old_conv.weight.data[:, stage_keeps[i]].contiguous().clone()
        # BN output ch unchanged (still 128)
        deblock_new[1].weight.data = deblock_old[1].weight.data.clone()
        deblock_new[1].bias.data = deblock_old[1].bias.data.clone()
        deblock_new[1].running_mean.data = deblock_old[1].running_mean.data.clone()
        deblock_new[1].running_var.data = deblock_old[1].running_var.data.clone()

    # 4. single_head_i (1x1 conv: in = num_filters[i], out = 1)
    for i in range(len(num_filters_new)):
        h_old = getattr(pb_old, f"single_head_{i}")
        h_new = getattr(pb_new, f"single_head_{i}")
        h_new.weight.data = h_old.weight.data[:, stage_keeps[i]].contiguous().clone()
        if h_old.bias is not None:
            h_new.bias.data = h_old.bias.data.clone()

    # 5+6. shrink_conv + cls/reg/dir heads: direct copy (unchanged shapes).
    #   skip_shrink_heads=True 时由 whole-net caller (wholenet_prune_pyramid) 自行重切,
    #   因为整网剪枝会改 deblocks 输出 / shrink in_dim / in_head, 直接 copy 会 shape mismatch.
    if not skip_shrink_heads:
        # 5. shrink_conv: input = sum(num_upsample_filter) = 384 (unchanged),
        #    output = 256 (unchanged). Direct copy.
        for k in old_model.shrink_conv.state_dict().keys():
            new_model.shrink_conv.state_dict()[k].data.copy_(
                old_model.shrink_conv.state_dict()[k].data
            )

        # 6. cls/reg/dir heads (unchanged shapes)
        for h_name in ("cls_head", "reg_head", "dir_head"):
            for k in getattr(old_model, h_name).state_dict().keys():
                getattr(new_model, h_name).state_dict()[k].data.copy_(
                    getattr(old_model, h_name).state_dict()[k].data
                )

    # 7. encoder_m1 + backbone_m1 + aligner_m1: unchanged (we only prune pyramid_backbone)
    for mod in ("encoder_m1", "backbone_m1", "aligner_m1"):
        for k, v in getattr(old_model, mod).state_dict().items():
            getattr(new_model, mod).state_dict()[k].data.copy_(v)

    print(f"  weight transfer done")


# ---------------------------------------------------------------------------
# Build new HeterPyramidCollab with smaller num_filters
# ---------------------------------------------------------------------------

def build_smaller_model(orig_hypes_path: str, num_filters_new: list,
                        groups: int = 32, width_per_group: int = 4):
    hypes = load_yaml(orig_hypes_path)
    args = hypes["model"]["args"]
    args["fusion_backbone"]["num_filters"] = num_filters_new
    if groups != 32:
        args["fusion_backbone"]["resnext_groups"] = groups
    if width_per_group != 4:
        args["fusion_backbone"]["width_per_group"] = width_per_group
    # num_upsample_filter stays [128, 128, 128]
    model = HeterPyramidCollab(args)
    return model, hypes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-dir", default="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29")
    p.add_argument("--out-dir", default="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10")
    p.add_argument("--num-filters-new", default="32,64,128",
                   help="new num_filters [N0, N1, N2]; original is [64,128,256]")
    p.add_argument("--groups", type=int, default=32,
                   help="ResNeXt groups (default 32; use 8 for extreme prune)")
    p.add_argument("--width-per-group", type=int, default=4,
                   help="ResNeXt width_per_group (default 4; HEAL: width = int(p*wpg/64)*groups)")
    args = p.parse_args()

    nf_new = [int(x) for x in args.num_filters_new.split(",")]
    groups = args.groups
    wpg = args.width_per_group
    # Validate: width = int(planes * wpg / 64) * groups; must be > 0 and
    # width // groups (in_per_group) must be >= 1
    for p_size in nf_new:
        w = int(p_size * wpg / 64) * groups
        ipg = w // groups
        if w <= 0 or ipg < 1:
            print(f"  ERROR: planes={p_size} + groups={groups} + wpg={wpg} → width={w} ipg={ipg}; infeasible")
            sys.exit(1)
        print(f"  validate: planes={p_size} groups={groups} wpg={wpg} → width={w} ipg={ipg}")
    orig_hypes = Path(args.orig_dir) / "config.yaml"
    # Auto-detect bestval ckpt (don't hardcode epoch 23 — baselines may differ)
    bestvals = sorted(Path(args.orig_dir).glob("net_epoch_bestval_at*.pth"))
    if not bestvals:
        print(f"  ERROR: no net_epoch_bestval_at*.pth in {args.orig_dir}")
        sys.exit(1)
    # Pick highest epoch
    def _ep(p):
        return int(p.stem.split("_at")[-1])
    bestvals.sort(key=_ep)
    orig_ckpt = bestvals[-1]
    src_epoch = _ep(orig_ckpt)
    print(f"  using bestval ckpt at epoch {src_epoch}: {orig_ckpt.name}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Build original model + load ckpt")
    old_model = build_pyramid_from_ckpt(str(orig_hypes), str(orig_ckpt), device="cpu")
    nf_old = old_model.pyramid_backbone.model_cfg["num_filters"]
    print(f"  Original num_filters: {nf_old}")

    print(f"[2/4] Build smaller model with num_filters={nf_new} groups={groups} wpg={wpg}")
    new_model, hypes = build_smaller_model(str(orig_hypes), nf_new,
                                           groups=groups, width_per_group=wpg)
    new_model.eval()

    print(f"[3/4] Transfer weights with L1 selection (groups={groups} wpg={wpg})")
    transfer_weights(old_model, new_model, nf_old, nf_new,
                     groups=groups, width_per_group=wpg)

    n_old = sum(p.numel() for p in old_model.parameters())
    n_new = sum(p.numel() for p in new_model.parameters())
    print(f"  params {n_old:,} -> {n_new:,}  (-{(1 - n_new/n_old)*100:.1f}%)")
    n_pb_old = sum(p.numel() for p in old_model.pyramid_backbone.parameters())
    n_pb_new = sum(p.numel() for p in new_model.pyramid_backbone.parameters())
    print(f"  pyramid_backbone {n_pb_old:,} -> {n_pb_new:,}  (-{(1 - n_pb_new/n_pb_old)*100:.1f}%)")

    # Quick numerical sanity: forward both models on random input
    print(f"\n  forward sanity (pyramid sub-module only)")
    from tools.export_onnx_pyramid import PyramidSubnet
    sub_old = PyramidSubnet(old_model).eval()
    sub_new = PyramidSubnet(new_model).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 64, 128, 256)
    with torch.no_grad():
        cls_o, reg_o, dir_o = sub_old(x)
        cls_n, reg_n, dir_n = sub_new(x)
    rel_cls = (cls_o - cls_n).abs().max() / max(cls_o.abs().max().item(), 1e-6)
    rel_reg = (reg_o - reg_n).abs().max() / max(reg_o.abs().max().item(), 1e-6)
    print(f"  cls rel={rel_cls:.2%}  reg rel={rel_reg:.2%}  "
          f"(expected large drift: pruned model needs finetune)")

    print(f"\n[4/4] Save pruned ckpt + smaller config")
    out_ckpt = out_dir / f"net_epoch_bestval_at{src_epoch}.pth"
    torch.save({"model_state_dict": new_model.state_dict()}, out_ckpt)
    print(f"  saved {out_ckpt}  ({out_ckpt.stat().st_size / 1e6:.2f} MB)")

    # Patch hypes config → new num_filters, save as out_dir/config.yaml
    hypes_dict = load_yaml(str(orig_hypes))
    hypes_dict["model"]["args"]["fusion_backbone"]["num_filters"] = nf_new
    if groups != 32:
        hypes_dict["model"]["args"]["fusion_backbone"]["resnext_groups"] = groups
    if wpg != 4:
        hypes_dict["model"]["args"]["fusion_backbone"]["width_per_group"] = wpg
    hypes_dict["name"] = f"Pyramid_DAIR_m1_pruned_g{groups}_" + "_".join(f"{n:03d}" for n in nf_new)
    out_yaml = out_dir / "config.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(hypes_dict, f, default_flow_style=False, allow_unicode=True)
    print(f"  saved {out_yaml}")
    # Patch input_source LiDAR-only (same as we did for orig)
    with open(out_yaml) as f:
        s = f.read()
    if "- camera" in s:
        import re
        s = re.sub(r"(input_source:\n)- lidar\n- camera\n", r"\1- lidar\n", s)
        with open(out_yaml, "w") as f:
            f.write(s)
        print(f"  patched input_source -> LiDAR-only")
    print(f"\n  ready for finetune at {out_dir}")


if __name__ == "__main__":
    main()
