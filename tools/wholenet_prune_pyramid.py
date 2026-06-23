"""全网剪枝 (含 deblocks 输出 + shrink + heads 输入) — 任务 B 新维度.

在 `tools/structural_prune_pyramid.py` (只剪 backbone num_filters) 基础上扩展:

  新维度 1 — deblocks 输出 (num_upsample_filter [128,128,128] → 更小):
      每个 deblock 是 ConvTranspose2d(num_filters[i] → num_upsample_filter[i]) + BN + ReLU.
      之前只剪了 deblock 输入 (跟 stage 输出), 输出固定 128. 这里把输出也按 L1 剪。

  新维度 2 — shrink_conv + cls/reg/dir heads 输入 (in_head 384→256 路径):
      shrink_conv = DownsampleConv → DoubleConv(input=sum(num_upsample_filter), out=in_head).
      deblocks 输出剪小后, shrink 输入 = sum(nuf_new) 自动变小 (concat); 同时把 shrink
      输出 (in_head) 也按 L1 剪, cls/reg/dir head 输入跟着剪。

依赖一致性 (本工具保证):
  deblock_out[i] = nuf_new[i]
     → shrink double_conv[0].in = sum(nuf_new) (按 concat 顺序拼 deblock 输出 keep)
  shrink double_conv[0].out = shrink_dim_new (= in_head_new)
     → shrink double_conv[2] in=out=shrink_dim_new
     → cls/reg/dir head in = shrink_dim_new

backbone (num_filters) 剪枝完全复用 structural_prune_pyramid.transfer_weights 的逻辑;
本工具在其之后追加 deblocks-out + shrink 重切, 并重建带新 num_upsample_filter / in_head 的模型。

输出: flat state_dict ckpt (HEAL load_saved_model 要求 flat, 不能 {"model_state_dict":...})
      + config.yaml (patch 后的 num_filters / num_upsample_filter / shrink_header / in_head)

用法:
    CUDA_VISIBLE_DEVICES="" python tools/wholenet_prune_pyramid.py \
        --orig-dir .../Pyramid_DAIR_m1_base_2023_08_14_11_42_29 \
        --out-dir  .../Pyramid_DAIR_m1_wholenet_p50 \
        --num-filters-new 32,64,128 \
        --num-upsample-new 96,96,96 \
        --shrink-dim-new 192
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from opencood.hypes_yaml.yaml_utils import load_yaml  # noqa: E402
from opencood.models.heter_pyramid_collab import HeterPyramidCollab  # noqa: E402
from tools.export_onnx_pyramid import build_pyramid_from_ckpt, PyramidSubnet  # noqa: E402
from tools.structural_prune_pyramid import (  # noqa: E402
    transfer_weights,
    l1_along_dim,
    top_indices,
)


# ---------------------------------------------------------------------------
# Build model with smaller deblocks-out + shrink
# ---------------------------------------------------------------------------

def build_wholenet_smaller(orig_hypes_path, nf_new, nuf_new, shrink_dim_new,
                           groups=32, wpg=4):
    hypes = load_yaml(orig_hypes_path)
    args = hypes["model"]["args"]
    args["fusion_backbone"]["num_filters"] = nf_new
    args["fusion_backbone"]["num_upsample_filter"] = nuf_new
    if groups != 32:
        args["fusion_backbone"]["resnext_groups"] = groups
    if wpg != 4:
        args["fusion_backbone"]["width_per_group"] = wpg
    # shrink: input_dim = sum(nuf_new), output dim = shrink_dim_new
    args["shrink_header"]["input_dim"] = int(sum(nuf_new))
    args["shrink_header"]["dim"] = [int(shrink_dim_new)]
    args["in_head"] = int(shrink_dim_new)
    model = HeterPyramidCollab(args)
    return model, hypes


# ---------------------------------------------------------------------------
# deblocks-out + shrink + heads reslice (新维度核心)
# ---------------------------------------------------------------------------

def transfer_deblocks_shrink_heads(old_model, new_model, nuf_old, nuf_new,
                                   shrink_dim_old, shrink_dim_new):
    """在 backbone transfer 之后, 重切 deblocks 输出 / shrink / cls/reg/dir heads。

    注意: backbone transfer (structural_prune_pyramid.transfer_weights) 已经把
    deblock **输入** 通道 (跟 stage 输出) 切好了, 但它把 deblock 输出 / shrink / heads
    都当成"形状不变直接 copy"。这里覆盖那部分, 真正剪 deblock 输出 + shrink + heads 输入。
    """
    pb_old = old_model.pyramid_backbone
    pb_new = new_model.pyramid_backbone
    num_levels = len(nuf_new)

    # ---- 1. deblock 输出剪枝: 每个 deblock[0] = ConvTranspose2d(nf[i] -> nuf[i]) ----
    # ConvTranspose2d weight shape = (in_ch, out_ch, kh, kw); 输出在 dim=1.
    # backbone transfer 已按 stage_keep 切了输入 (dim=0); 这里再按 L1 切输出 (dim=1)。
    deblock_out_keeps = []
    for i in range(num_levels):
        deb_old = pb_old.deblocks[i]
        deb_new = pb_new.deblocks[i]
        conv_old = deb_old[0]
        conv_new = deb_new[0]
        is_convT = isinstance(conv_old, torch.nn.ConvTranspose2d)
        out_dim = 1 if is_convT else 0  # ConvT 输出在 dim1, Conv 在 dim0
        # 在原始权重上算输出通道 L1 (dim = out_dim)
        scores = l1_along_dim(conv_old.weight.data, dim=out_dim)
        keep = top_indices(scores, nuf_new[i])
        deblock_out_keeps.append(keep)
        # conv_new.weight 已被 backbone transfer 按"输入 keep"切好 (它现在 shape 应是
        # ConvT: (in_new, nuf_old, ...) 或 Conv: (nuf_old, in_new, ...))。我们再切输出。
        w = conv_new.weight.data
        if is_convT:
            conv_new.weight.data = w[:, keep].contiguous().clone()
        else:
            conv_new.weight.data = w[keep].contiguous().clone()
        if conv_new.bias is not None and conv_old.bias is not None:
            conv_new.bias.data = conv_old.bias.data[keep].contiguous().clone()
        # BN (deb[1]): 输出通道 = nuf, 按 keep 切
        bn_old, bn_new = deb_old[1], deb_new[1]
        bn_new.weight.data = bn_old.weight.data[keep].clone()
        bn_new.bias.data = bn_old.bias.data[keep].clone()
        bn_new.running_mean.data = bn_old.running_mean.data[keep].clone()
        bn_new.running_var.data = bn_old.running_var.data[keep].clone()

    # 是否有额外的 deblock (len(deblocks) > num_levels) — DAIR Pyramid upsample_strides
    # 长度 == num_levels, 无额外 deblock; 若有需另处理。
    assert len(pb_new.deblocks) == num_levels, (
        f"额外 deblock 未处理: {len(pb_new.deblocks)} vs {num_levels}")

    # ---- 2. shrink_conv 输入 (concat 顺序) + 输出剪枝 ----
    # shrink = DownsampleConv.layers[0] = DoubleConv:
    #   double_conv[0] = Conv2d(sum(nuf), shrink_dim, k=3) ; [1]=ReLU
    #   double_conv[2] = Conv2d(shrink_dim, shrink_dim, k=3); [3]=ReLU
    # concat 顺序 = deblock0_out ++ deblock1_out ++ deblock2_out (按 forward 中 cat 顺序)
    shrink_old = old_model.shrink_conv.layers[0].double_conv
    shrink_new = new_model.shrink_conv.layers[0].double_conv
    c0_old = shrink_old[0]
    c0_new = shrink_new[0]
    # 输入索引: 把每个 deblock 的输出 keep 映射到 concat 后的全局 index
    in_idx = []
    offset = 0
    for i in range(num_levels):
        in_idx.append(deblock_out_keeps[i] + offset)
        offset += nuf_old[i]
    in_idx = torch.cat(in_idx)
    # 输出索引: shrink_dim L1 (在 c0_old 输出维 dim0)
    s_out_scores = l1_along_dim(c0_old.weight.data, dim=0)
    shrink_out_keep = top_indices(s_out_scores, shrink_dim_new)
    # c0: (shrink_dim, sum(nuf), k, k) -> 切 out (shrink_out_keep) + in (in_idx)
    w0 = c0_old.weight.data[shrink_out_keep][:, in_idx]
    c0_new.weight.data = w0.contiguous().clone()
    if c0_new.bias is not None:
        c0_new.bias.data = c0_old.bias.data[shrink_out_keep].contiguous().clone()
    # c2 (double_conv[2]): (shrink_dim, shrink_dim, 3,3) -> 切 out+in 都用 shrink_out_keep
    c2_old = shrink_old[2]
    c2_new = shrink_new[2]
    w2 = c2_old.weight.data[shrink_out_keep][:, shrink_out_keep]
    c2_new.weight.data = w2.contiguous().clone()
    if c2_new.bias is not None:
        c2_new.bias.data = c2_old.bias.data[shrink_out_keep].contiguous().clone()

    # ---- 3. cls/reg/dir heads 输入 = shrink_out_keep ----
    for h_name in ("cls_head", "reg_head", "dir_head"):
        h_old = getattr(old_model, h_name)
        h_new = getattr(new_model, h_name)
        h_new.weight.data = h_old.weight.data[:, shrink_out_keep].contiguous().clone()
        if h_new.bias is not None and h_old.bias is not None:
            h_new.bias.data = h_old.bias.data.clone()

    print(f"  deblocks out keeps: {[len(k) for k in deblock_out_keeps]} (was {nuf_old})")
    print(f"  shrink in: {len(in_idx)} (was {sum(nuf_old)}), "
          f"out: {len(shrink_out_keep)} (was {shrink_dim_old})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-dir",
                   default="/home/jichengzhi/heal_research/checkpoints/stage1/"
                           "Pyramid_DAIR_m1_base_2023_08_14_11_42_29")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-filters-new", default="32,64,128",
                   help="backbone num_filters; orig [64,128,256]")
    p.add_argument("--num-upsample-new", default="96,96,96",
                   help="deblocks output (num_upsample_filter); orig [128,128,128]")
    p.add_argument("--shrink-dim-new", type=int, default=192,
                   help="shrink_conv output (= in_head); orig 256")
    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--width-per-group", type=int, default=4)
    p.add_argument("--epoches", type=int, default=None,
                   help="patch train_params.epoches for finetune (resume from bestval)")
    args = p.parse_args()

    nf_new = [int(x) for x in args.num_filters_new.split(",")]
    nuf_new = [int(x) for x in args.num_upsample_new.split(",")]
    groups, wpg = args.groups, args.width_per_group
    shrink_dim_new = args.shrink_dim_new

    orig_hypes = str(Path(args.orig_dir) / "config.yaml")
    bestvals = sorted(Path(args.orig_dir).glob("net_epoch_bestval_at*.pth"),
                      key=lambda q: int(q.stem.split("_at")[-1]))
    assert bestvals, f"no bestval ckpt in {args.orig_dir}"
    orig_ckpt = bestvals[-1]
    src_epoch = int(orig_ckpt.stem.split("_at")[-1])
    print(f"[load] {orig_ckpt} (epoch {src_epoch})")

    print("[1/5] build original + load")
    old_model = build_pyramid_from_ckpt(orig_hypes, str(orig_ckpt), device="cpu")
    nf_old = list(old_model.pyramid_backbone.model_cfg["num_filters"])
    nuf_old = list(old_model.pyramid_backbone.model_cfg["num_upsample_filter"])
    shrink_dim_old = old_model.cls_head.in_channels
    print(f"  num_filters {nf_old} -> {nf_new}")
    print(f"  num_upsample_filter {nuf_old} -> {nuf_new}")
    print(f"  shrink/in_head {shrink_dim_old} -> {shrink_dim_new}")

    print(f"[2/5] build smaller wholenet model")
    new_model, hypes = build_wholenet_smaller(
        orig_hypes, nf_new, nuf_new, shrink_dim_new, groups=groups, wpg=wpg)
    new_model.eval()

    print(f"[3/5] backbone transfer (reuse structural_prune)")
    transfer_weights(old_model, new_model, nf_old, nf_new, groups=groups,
                     width_per_group=wpg, skip_shrink_heads=True)
    print(f"[3.5/5] deblocks-out + shrink + heads reslice (NEW dimension)")
    transfer_deblocks_shrink_heads(old_model, new_model, nuf_old, nuf_new,
                                   shrink_dim_old, shrink_dim_new)

    n_old = sum(p.numel() for p in old_model.parameters())
    n_new = sum(p.numel() for p in new_model.parameters())
    print(f"  params {n_old:,} -> {n_new:,}  (-{(1 - n_new / n_old) * 100:.1f}%)")

    print(f"[4/5] forward sanity (pruned model needs finetune; drift expected)")
    sub_old = PyramidSubnet(old_model).eval()
    sub_new = PyramidSubnet(new_model).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 64, 128, 256)
    with torch.no_grad():
        cls_o, reg_o, dir_o = sub_old(x)
        cls_n, reg_n, dir_n = sub_new(x)
    print(f"  old out {tuple(cls_o.shape)} new out {tuple(cls_n.shape)} (shapes match expected)")
    assert cls_o.shape == cls_n.shape and reg_o.shape == reg_n.shape, "head out shape mismatch!"
    print(f"  FORWARD OK (no shape error) — whole-net prune structurally valid")

    print(f"[5/5] save FLAT state_dict + config")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / f"net_epoch_bestval_at{src_epoch}.pth"
    # FLAT (HEAL load_saved_model 要求), NOT {"model_state_dict":...}
    torch.save(new_model.state_dict(), out_ckpt)
    print(f"  saved {out_ckpt} ({out_ckpt.stat().st_size / 1e6:.2f} MB, flat)")

    # patch config
    hypes["model"]["args"]["fusion_backbone"]["num_filters"] = nf_new
    hypes["model"]["args"]["fusion_backbone"]["num_upsample_filter"] = nuf_new
    hypes["model"]["args"]["shrink_header"]["input_dim"] = int(sum(nuf_new))
    hypes["model"]["args"]["shrink_header"]["dim"] = [int(shrink_dim_new)]
    hypes["model"]["args"]["in_head"] = int(shrink_dim_new)
    if groups != 32:
        hypes["model"]["args"]["fusion_backbone"]["resnext_groups"] = groups
    if wpg != 4:
        hypes["model"]["args"]["fusion_backbone"]["width_per_group"] = wpg
    hypes["name"] = ("Pyramid_DAIR_m1_wholenet_nf" + "_".join(f"{n}" for n in nf_new)
                     + "_nuf" + "_".join(f"{n}" for n in nuf_new)
                     + f"_sh{shrink_dim_new}")
    if args.epoches is not None:
        hypes.setdefault("train_params", {})["epoches"] = int(args.epoches)
        hypes["train_params"]["save_freq"] = 1
        hypes["train_params"]["eval_freq"] = 1
    out_yaml = out_dir / "config.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(hypes, f, default_flow_style=False, allow_unicode=True)
    print(f"  saved {out_yaml}")
    print(f"\n  READY for finetune: --model_dir {out_dir} --hypes_yaml {out_yaml}")
    print(f"  resume epoch = {src_epoch}; train to epoches={args.epoches}")


if __name__ == "__main__":
    main()
