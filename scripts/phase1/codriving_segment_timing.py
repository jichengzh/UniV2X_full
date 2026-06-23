#!/usr/bin/env python3
"""RSU vs 车端 (vehicle) segment timing + fusion_net decomposition.

Answers the user's correction: fusion_net's cost is dominated by a SECOND backbone
pass (center_point_codriving.py L140-145 passes self.backbone into fusion; codriving_attn.py
L264 `feats = backbone.resnet(x)` re-runs the whole ResNet). So the TVM-tunable backbone
compute is NOT 19% of the pipeline — it appears in BOTH the standalone pass AND inside fusion.

Splits the pipeline by physical deployment:
  RSU  (路端): pillar_vfe + scatter + backbone(1x) [+ single cls/reg for confidence]  -> features out
  车端 (vehicle): own perception (= RSU compute) + fusion_net(= backbone re-run + warp/attn) + heads + decode
fusion_net is decomposed into: backbone.resnet re-run  vs  warp+comm+attention.

Per-module CUDA-event timing over real DAIR 2-agent frames; decode uses the cached-grid
fix so it doesn't pollute. Per-frame util gate (report min/p10 as contention-robust).
Usage: python codriving_segment_timing.py --gpu 0 --tag base --n 30
"""
import os, sys, time, json, argparse, subprocess, types
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
    frames = get_frames(hypes, args.n)
    print(f"[{args.tag}] {len(frames)} frames, multi_scale={getattr(model,'multi_scale',None)}")

    # accumulators: list of per-frame ms per segment
    segs = {k: [] for k in ["vfe", "scatter", "backbone_std", "cls_single", "reg_single",
                            "fusion_total", "fusion_backbone", "fusion_attn",
                            "cls_final", "reg_final", "decode"]}
    n_contended = 0

    with torch.inference_mode():
        # warmup
        for f in frames[:5]:
            model(to_device(f, "cuda"))
        torch.cuda.synchronize()

        for f in frames[5:5 + args.n]:
            d = to_device(f, "cuda")
            u0 = _util(args.gpu)
            vf = d["processed_lidar"]["voxel_features"]; vc = d["processed_lidar"]["voxel_coords"]
            vn = d["processed_lidar"]["voxel_num_points"]; rl = d["record_len"]
            ptm = d["pairwise_t_matrix"]
            bd = {"voxel_features": vf, "voxel_coords": vc, "voxel_num_points": vn, "record_len": rl}

            timed = [k for k in segs if k != "fusion_attn"]  # fusion_attn is derived, not recorded
            e = {k: (ev(), ev()) for k in timed}
            torch.cuda.synchronize()
            e["vfe"][0].record();        bd = model.pillar_vfe(bd);  e["vfe"][1].record()
            e["scatter"][0].record();    bd = model.scatter(bd);     e["scatter"][1].record()
            sf = bd["spatial_features"]  # pre-backbone scattered [BN,64,H,W] for fusion re-run
            e["backbone_std"][0].record(); bd = model.backbone(bd);  e["backbone_std"][1].record()
            sf2d = bd["spatial_features_2d"]
            if model.shrink_flag:
                sf2d = model.shrink_conv(sf2d)
            e["cls_single"][0].record(); psm = model.cls_head(sf2d); e["cls_single"][1].record()
            e["reg_single"][0].record(); rm = model.reg_head(sf2d);  e["reg_single"][1].record()

            # fusion total (multi-scale, re-runs backbone internally)
            e["fusion_total"][0].record()
            fused, comm, rd = model.fusion_net(sf, psm, rl, ptm, model.backbone, None)
            e["fusion_total"][1].record()
            # isolated: just the backbone.resnet re-run that fusion does on sf
            e["fusion_backbone"][0].record()
            _ = model.backbone.resnet(sf)
            e["fusion_backbone"][1].record()
            if model.shrink_flag:
                fused = model.shrink_conv(fused)
            e["cls_final"][0].record(); clf = model.cls_head(fused); e["cls_final"][1].record()
            e["reg_final"][0].record(); rgf = model.reg_head(fused); e["reg_final"][1].record()
            e["decode"][0].record(); _ = model.generate_predicted_boxes(clf, rgf); e["decode"][1].record()
            torch.cuda.synchronize()

            u1 = _util(args.gpu)
            ms = {k: e[k][0].elapsed_time(e[k][1]) for k in e}
            del bd
            ms["fusion_attn"] = ms["fusion_total"] - ms["fusion_backbone"]
            if max(u0, u1) > args.gate:
                n_contended += 1; continue
            for k in segs:
                segs[k].append(ms[k])
            del d; torch.cuda.empty_cache()

    def stat(v):
        return {"p50": round(float(np.percentile(v, 50)), 3),
                "p10": round(float(np.percentile(v, 10)), 3),
                "min": round(float(np.min(v)), 3)} if v else None

    out = {k: stat(v) for k, v in segs.items()}
    n_clean = len(segs["vfe"])
    # segment rollups (use p10 = contention-robust floor)
    def p10(k): return out[k]["p10"] if out[k] else 0.0
    rsu = p10("vfe") + p10("scatter") + p10("backbone_std") + p10("cls_single") + p10("reg_single")
    veh_extra = p10("fusion_total") + p10("cls_final") + p10("reg_final") + p10("decode")
    veh_total = rsu + veh_extra
    tvm_backbone_total = p10("backbone_std") + p10("fusion_backbone")  # all TVM-tunable backbone
    summary = {
        "tag": args.tag, "n_clean": n_clean, "n_contended": n_contended,
        "per_module_ms": out,
        "RSU_segment_ms": round(rsu, 3),
        "vehicle_segment_ms": round(veh_total, 3),
        "fusion_backbone_frac_of_fusion": round(p10("fusion_backbone") / p10("fusion_total"), 3) if p10("fusion_total") else None,
        "tvm_backbone_total_ms": round(tvm_backbone_total, 3),
        "tvm_backbone_frac_of_vehicle": round(tvm_backbone_total / veh_total, 3) if veh_total else None,
        "tvm_backbone_frac_of_RSU": round(p10("backbone_std") / rsu, 3) if rsu else None,
        "note": "p10 floor (contention-robust); CUDA-event per-module; decode=cached-grid fixed",
    }
    print(json.dumps(summary, indent=2))
    outp = REPO / f"output/codriving_pilot/logs/segment_timing_{args.tag}.json"
    json.dump(summary, open(outp, "w"), indent=2)
    print("saved", outp)


if __name__ == "__main__":
    main()
