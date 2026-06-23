#!/usr/bin/env python3
"""
VERIFY (user challenged the numbers):
  (A) isolated backbone eager latency base/p50/p75 on FIXED input -> is it really flat under pruning?
  (B) decode generate_predicted_boxes: confirm CPU-meshgrid artifact by monkeypatching a cached-grid
      version and re-timing -> does 62ms collapse to ~1ms?
All real CUDA-event timing, fixed synthetic input (no data loader noise).
Usage: python codriving_verify_bottleneck.py --gpu <idle>
"""
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/jichengzhi/V2X")
V2XVERSE = Path("/home/jichengzhi/V2Xverse")
CFG = {
    "base": (REPO / "output/codriving_pilot/collab_export/dair_centerpoint_codriving_4090.yaml", None),
    "p50":  (REPO / "output/codriving_pilot/p50/config_finetune.yaml",
             REPO / "output/codriving_pilot/p50/net_epoch_bestval_at5_p50.pth"),
    "p75":  (REPO / "output/codriving_pilot/p75/config_finetune.yaml", None),
}


def load_model(tag):
    from opencood.tools import train_utils
    from opencood.hypes_yaml.yaml_utils import load_yaml
    cfg_path, ckpt = CFG[tag]
    hypes = load_yaml(str(cfg_path))
    model = train_utils.create_model(hypes)
    if ckpt is not None:
        sd = torch.load(str(ckpt), map_location="cpu")
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=False)
    else:
        _, model = train_utils.load_saved_model(str(cfg_path.parent), model)
    return model.cuda().eval(), hypes["model"]["args"]["base_bev_backbone"]["num_filters"]


def time_module(fn, warmup=50, runs=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lat = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        lat.append(s.elapsed_time(e))
    return float(np.percentile(lat, 50)), float(np.mean(lat))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(V2XVERSE)); os.chdir(V2XVERSE)
    torch.cuda.set_device(args.gpu)

    print("=== (A) ISOLATED BACKBONE eager latency, fixed input (2,64,256,512) ===")
    bb_rows = []
    for tag in ["base", "p50", "p75"]:
        model, nf = load_model(tag)
        bb = model.backbone
        x = {"spatial_features": torch.randn(2, 64, 256, 512, device="cuda")}
        with torch.inference_mode():
            p50, mean = time_module(lambda: bb(x))
        n_params = sum(p.numel() for p in bb.parameters())
        print(f"  {tag:5s} num_filters={nf}  backbone_params={n_params/1e6:.3f}M  p50={p50:.3f}ms mean={mean:.3f}ms")
        bb_rows.append({"tag": tag, "num_filters": nf, "backbone_params_M": round(n_params/1e6, 3),
                        "backbone_p50_ms": round(p50, 4), "backbone_mean_ms": round(mean, 4)})
        del model, bb; torch.cuda.empty_cache()

    print("\n=== (B) DECODE generate_predicted_boxes: original vs cached-grid monkeypatch ===")
    model, _ = load_model("base")
    H, W = 128, 256
    cls = torch.randn(1, 1, H, W, device="cuda")
    box = torch.randn(1, 8, H, W, device="cuda")

    # original
    with torch.inference_mode():
        p50_o, mean_o = time_module(lambda: model.generate_predicted_boxes(cls, box))

    # cached-grid version (precompute meshgrid on GPU once)
    osf, vs, clr = model.out_size_factor, model.voxel_size, model.cav_lidar_range
    ys0, xs0 = torch.meshgrid(torch.arange(0, H, device="cuda"), torch.arange(0, W, device="cuda"))
    xs0 = xs0.reshape(1, -1, 1).float(); ys0 = ys0.reshape(1, -1, 1).float()

    def decode_cached(cls_preds, box_preds):
        bp = box_preds.permute(0, 2, 3, 1).contiguous()
        b, Hh, Ww, cs = bp.size()
        bp = bp.reshape(b, Hh * Ww, cs)
        h = bp[..., 3:4] * osf * vs[0]; w = bp[..., 4:5] * osf * vs[1]; l = bp[..., 5:6] * osf * vs[2]
        dim = torch.cat([h, w, l], dim=-1)
        hei = bp[..., 2:3] * osf * vs[2] + clr[2]
        rot = torch.atan2(bp[..., 6:7], bp[..., 7:8])
        xs = (xs0 + bp[:, :, 0:1]) * osf * vs[0] + clr[0]
        ys = (ys0 + bp[:, :, 1:2]) * osf * vs[1] + clr[1]
        return cls_preds, torch.cat([xs, ys, hei, dim, rot], dim=2)

    with torch.inference_mode():
        p50_c, mean_c = time_module(decode_cached, cls, box) if False else (None, None)
        # time_module takes no-arg fn:
        p50_c, mean_c = time_module(lambda: decode_cached(cls, box))

    print(f"  original (CPU meshgrid+H2D each call): p50={p50_o:.3f}ms mean={mean_o:.3f}ms")
    print(f"  cached-grid (GPU precomputed once):    p50={p50_c:.3f}ms mean={mean_c:.3f}ms")
    print(f"  => speedup {mean_o/mean_c:.0f}x  (per-call; forward calls it ~4x/frame)")

    out = {"backbone_isolated": bb_rows,
           "decode_original_ms": round(mean_o, 4), "decode_cached_ms": round(mean_c, 4),
           "decode_speedup": round(mean_o / mean_c, 1),
           "note": "isolated backbone eager fixed-input; decode original vs cached-grid monkeypatch"}
    json.dump(out, open(REPO / "output/codriving_pilot/logs/verify_bottleneck.json", "w"), indent=2)
    print("\nsaved logs/verify_bottleneck.json")


if __name__ == "__main__":
    main()
