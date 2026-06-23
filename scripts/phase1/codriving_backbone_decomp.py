#!/usr/bin/env python3
"""Refine: why does RSU backbone (4.57ms) != vehicle fusion-backbone (2.53ms)?

The full backbone module = resnet(3 stages) + deblocks(ConvTranspose upsample) + concat
(base_bev_backbone_resnet.py L90-102). My earlier 'fusion_backbone' timed ONLY backbone.resnet
(L264 of codriving_attn), so the deblocks fusion ALSO runs (L309/318) got bundled into
'fusion_attn'. This script separates resnet / deblocks / (warp+attention) cleanly so the
genuinely-hard residual (grid_sample warp + where2comm attention, the only NON-conv part) is
measured, not conflated with TVM-tunable ConvTranspose deblocks.

Usage: python codriving_backbone_decomp.py --gpu 0 --tag base --n 30
"""
import os, sys, json, argparse, subprocess, types
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/jichengzhi/V2X"); V2XVERSE = Path("/home/jichengzhi/V2Xverse")
sys.path.insert(0, str(REPO / "scripts/phase1"))


def _util(gpu):
    try:
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "-i", str(gpu)], timeout=2).decode().strip())
    except Exception:
        return 0.0


def ev():
    return torch.cuda.Event(enable_timing=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--gate", type=float, default=60)
    args = ap.parse_args()
    sys.path.insert(0, str(V2XVERSE)); os.chdir(V2XVERSE); torch.cuda.set_device(args.gpu)

    from codriving_wholenet_decodefix import load_model, get_frames, make_decode_cached
    from opencood.tools.train_utils import to_device

    model, hypes = load_model(args.tag)
    model.generate_predicted_boxes = types.MethodType(make_decode_cached(model), model)
    bb = model.backbone
    frames = get_frames(hypes, args.n)

    keys = ["resnet", "deblocks", "fusion_total"]
    segs = {k: [] for k in keys}; nc = 0
    with torch.inference_mode():
        for f in frames[:5]:
            model(to_device(f, "cuda"))
        torch.cuda.synchronize()
        for f in frames[5:5 + args.n]:
            d = to_device(f, "cuda")
            u0 = _util(args.gpu)
            bd = {"voxel_features": d["processed_lidar"]["voxel_features"],
                  "voxel_coords": d["processed_lidar"]["voxel_coords"],
                  "voxel_num_points": d["processed_lidar"]["voxel_num_points"],
                  "record_len": d["record_len"]}
            bd = model.pillar_vfe(bd); bd = model.scatter(bd)
            sf = bd["spatial_features"]
            e = {k: (ev(), ev()) for k in keys}
            torch.cuda.synchronize()
            # 1) resnet stages only
            e["resnet"][0].record(); feats = bb.resnet(sf); e["resnet"][1].record()
            # 2) deblocks + concat (replicate backbone.forward L94-102 exactly)
            e["deblocks"][0].record()
            ups = [bb.deblocks[i](feats[i]) for i in range(len(feats))] if len(bb.deblocks) > 0 else list(feats)
            xx = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
            if len(bb.deblocks) > len(feats):
                xx = bb.deblocks[-1](xx)
            e["deblocks"][1].record()
            # 3) whole fusion (resnet + deblocks + warp + attention) for reference
            bd = model.backbone(bd); sf2d = bd["spatial_features_2d"]
            if model.shrink_flag: sf2d = model.shrink_conv(sf2d)
            psm = model.cls_head(sf2d)
            e["fusion_total"][0].record()
            _ = model.fusion_net(sf, psm, d["record_len"], d["pairwise_t_matrix"], model.backbone, None)
            e["fusion_total"][1].record()
            torch.cuda.synchronize()
            u1 = _util(args.gpu)
            ms = {k: e[k][0].elapsed_time(e[k][1]) for k in keys}
            if max(u0, u1) > args.gate: nc += 1; continue
            for k in keys: segs[k].append(ms[k])
            del d; torch.cuda.empty_cache()

    def p10(v): return round(float(np.percentile(v, 10)), 3)
    r, db, ft = p10(segs["resnet"]), p10(segs["deblocks"]), p10(segs["fusion_total"])
    warp_attn = round(ft - r - db, 3)
    out = {"tag": args.tag, "n_clean": len(segs["resnet"]), "n_contended": nc,
           "resnet_ms": r, "deblocks_concat_ms": db, "fusion_total_ms": ft,
           "warp_attn_ms": warp_attn,
           "full_backbone_recompute_ms": round(r + db, 3),
           "note": "p10 floor; full backbone=resnet+deblocks+concat; fusion=resnet+deblocks+warp+attn; "
                   "warp_attn=fusion_total-resnet-deblocks = the ONLY non-conv (TVM-unschedulable) residual"}
    print(json.dumps(out, indent=2))
    json.dump(out, open(REPO / f"output/codriving_pilot/logs/bb_decomp_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
