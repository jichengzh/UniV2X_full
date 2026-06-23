"""M4.9 Late Fusion per-stage timing.

Late Fusion (HEAL `heter_model_late`) is structurally different from heter
baselines: each agent runs an INDEPENDENT single-agent forward (encoder +
backbone + multi-scale layers + shrink + heads + NMS), then cross-agent
boxes are merged at the ego via a SECOND NMS pass.

Forward decomposition (HeterModelLate.forward, single agent):
  encoder_m1 → backbone_m1 (ResNetBEVBackbone)
              → layers_m1 (multiscale: layer1, layer2, ... + decode_multiscale)
              → shrink_conv_m1 → cls/reg/dir_head_m1

Per-scene total:
  forward (per-agent) × N_agents + per-agent NMS × N_agents + cross-agent NMS

In OPV2V test, average agents/scene ≈ 2.0.

Usage:
  python m4_9_late_fusion_timing.py \\
      --ckpt /home/jichengzhi/heal_research/checkpoints/baselines_hf/Late_Fusion \\
      --warmup 20 --measure 100
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
sys.path.insert(0, HEAL_ROOT)
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

torch.multiprocessing.set_sharing_strategy("file_system")


class CudaTimer:
    def __init__(self, labels):
        self.labels = list(labels)
        self.events = {k: torch.cuda.Event(enable_timing=True) for k in self.labels}
        self.records = {k: [] for k in self.labels[1:]}

    def mark(self, label):
        self.events[label].record()

    def collect(self):
        torch.cuda.synchronize()
        prev = self.labels[0]
        for cur in self.labels[1:]:
            self.records[cur].append(self.events[prev].elapsed_time(self.events[cur]))
            prev = cur


def percentile(arr, p):
    return float(np.percentile(arr, p))


def stats(arr):
    return {
        "n": len(arr),
        "mean_ms": float(np.mean(arr)) if arr else None,
        "p50_ms": percentile(arr, 50) if arr else None,
        "p99_ms": percentile(arr, 99) if arr else None,
        "min_ms": float(np.min(arr)) if arr else None,
        "max_ms": float(np.max(arr)) if arr else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--modality", default="m1",
                    help="Late Fusion ckpt has m1/m2/m3/m4; pick which to time")
    ap.add_argument("--test-dir",
                    default="/home/jichengzhi/heal_research/dataset/OPV2V_orig/extracted/test")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--out-dir",
                    default="/home/jichengzhi/UniV2X/data/v2x_baseline_timing")
    args = ap.parse_args()

    cfg_path = os.path.join(args.ckpt, "config.yaml")
    print(f"[boot] cfg: {cfg_path}", flush=True)
    print(f"[boot] modality: {args.modality}", flush=True)

    mn = args.modality
    hypes = yaml_utils.load_yaml(cfg_path, None)
    hypes["test_dir"] = args.test_dir
    hypes["validate_dir"] = args.test_dir
    hypes["root_dir"] = args.test_dir
    # For m1-only timing: drop camera/depth inputs to avoid missing _depth.png errors
    if mn == "m1":
        hypes["input_source"] = ["lidar"]
        for m_check in ["m2", "m3", "m4"]:
            if m_check in hypes.get("model", {}).get("args", {}):
                enc_args = hypes["model"]["args"][m_check].get("encoder_args", {})
                if "depth_supervision" in enc_args:
                    enc_args["depth_supervision"] = False
                if "use_depth_gt" in enc_args:
                    enc_args["use_depth_gt"] = False
    parser_func = getattr(yaml_utils, hypes["yaml_parser"])
    hypes = parser_func(hypes)

    # Late fusion in HEAL uses different fusion-method "late" which gives
    # data_dict with 'inputs_m1' / 'inputs_m2' keys.
    print("[boot] building model", flush=True)
    model = train_utils.create_model(hypes)
    # Manually load weights, filtering out m3 (spconv 1.2.1 incompatible) and m4 (camera, lazy depth supervision)
    import glob
    ckpts = sorted(glob.glob(os.path.join(args.ckpt, "net_epoch_bestval_at*.pth")) +
                   glob.glob(os.path.join(args.ckpt, "net_epoch*.pth")))
    ckpt_path = ckpts[-1]
    print(f"[boot] loading {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu")
    # Drop m3 (spconv mismatch) and m4 (camera, needs depth GT) keys
    sd_filtered = {k: v for k, v in sd.items()
                   if not (k.startswith("encoder_m3.") or k.startswith("encoder_m4.") or
                           k.startswith("backbone_m3.") or k.startswith("backbone_m4.") or
                           k.startswith("layers_m3.") or k.startswith("layers_m4.") or
                           k.startswith("shrink_conv_m3.") or k.startswith("shrink_conv_m4.") or
                           k.startswith("cls_head_m3.") or k.startswith("cls_head_m4.") or
                           k.startswith("reg_head_m3.") or k.startswith("reg_head_m4.") or
                           k.startswith("dir_head_m3.") or k.startswith("dir_head_m4."))}
    print(f"[boot] filtered {len(sd) - len(sd_filtered)} keys (m3+m4), loading {len(sd_filtered)}", flush=True)
    missing, unexpected = model.load_state_dict(sd_filtered, strict=False)
    m1_missing = [k for k in missing if "_m1" in k]
    print(f"[boot] m1 missing keys: {len(m1_missing)} (expect 0 for fully loaded m1)", flush=True)
    model.cuda().eval()

    print("[boot] building dataset", flush=True)
    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, num_workers=2,
                        collate_fn=ds.collate_batch_test,
                        shuffle=False, pin_memory=False)

    # Each batch in late fusion has 'ego' key with multiple cavs.
    # We measure per-agent single forward (HeterModelLate.forward takes one
    # cav at a time via inputs_m{i}), then sum across cavs in the scene.

    mn = args.modality
    labels = ["start", "encoder", "backbone", "layers", "shrink", "heads"]
    timer = CudaTimer(labels)
    e2e_per_scene = []
    nms_per_scene = []
    n_agents_per_scene = []

    # We need to call model.forward(per-cav dict). Late fusion dataset
    # gives ego dict containing N cavs' data; we iterate over them.

    total_iters = args.warmup + args.measure
    print(f"[boot] starting: warmup={args.warmup}, measure={args.measure}", flush=True)

    # Patch postproc for cross-agent NMS too
    post_processor = ds.post_processor

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None or i >= total_iters:
                if i >= total_iters:
                    break
                continue
            batch = train_utils.to_device(batch, torch.device("cuda"))
            torch.cuda.synchronize()
            t_total_start = time.time()

            data_dict = batch["ego"]
            # Iterate over all cavs in scene
            agent_count = 0
            for cav_id in data_dict.keys():
                cav_dict = data_dict[cav_id] if isinstance(data_dict[cav_id], dict) else None
                if cav_dict is None:
                    continue
                # check if this cav has m1 inputs
                input_keys = [k for k in cav_dict.keys() if k.startswith("inputs_")]
                if not input_keys:
                    continue
                cav_mn = input_keys[0].replace("inputs_", "")
                if cav_mn != mn:
                    continue

                timer.mark("start")

                feat = getattr(model, f"encoder_{cav_mn}")(cav_dict, cav_mn)
                timer.mark("encoder")

                feat = getattr(model, f"backbone_{cav_mn}")(
                    {"spatial_features": feat})["spatial_features_2d"]
                timer.mark("backbone")

                # multiscale layers
                feature_list = [feat]
                layers_num = getattr(model, f"layers_num_{cav_mn}")
                layers = getattr(model, f"layers_{cav_mn}")
                for li in range(1, layers_num):
                    feat = layers.get_layer_i_feature(feat, layer_i=li)
                    feature_list.append(feat)
                feat = layers.decode_multiscale_feature(feature_list)
                timer.mark("layers")

                feat = getattr(model, f"shrink_conv_{cav_mn}")(feat)
                timer.mark("shrink")

                cls = getattr(model, f"cls_head_{cav_mn}")(feat)
                reg = getattr(model, f"reg_head_{cav_mn}")(feat)
                dir_ = getattr(model, f"dir_head_{cav_mn}")(feat)
                timer.mark("heads")
                timer.collect()
                agent_count += 1

            torch.cuda.synchronize()
            t_total = (time.time() - t_total_start) * 1000

            if i >= args.warmup:
                e2e_per_scene.append(t_total)
                n_agents_per_scene.append(agent_count)
            if i + 1 == args.warmup:
                timer.records = {k: [] for k in timer.records}
                print(f"[warmup-done] reset", flush=True)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total_iters}]", flush=True)

    report = {
        "tag": "late_fusion",
        "ckpt": args.ckpt,
        "modality": mn,
        "warmup": args.warmup,
        "measure": args.measure,
        "n_collected": len(e2e_per_scene),
        "n_agents_per_scene_mean": float(np.mean(n_agents_per_scene)) if n_agents_per_scene else None,
        "e2e_walltime_per_scene": stats(e2e_per_scene),
        "forward_stages_per_agent_cuda_event": {k: stats(v) for k, v in timer.records.items()},
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / "late_fusion_real.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {out_path}", flush=True)

    print(f"\n=== late_fusion per-agent forward summary (mean / p50 / p99) ===")
    for k, v in timer.records.items():
        s = stats(v)
        if s["mean_ms"] is not None:
            print(f"  {k:14s}  mean={s['mean_ms']:7.3f}  p50={s['p50_ms']:7.3f}  p99={s['p99_ms']:7.3f}")
    print(f"  e2e per scene  mean={np.mean(e2e_per_scene):7.3f}  p50={np.percentile(e2e_per_scene, 50):7.3f}")
    print(f"  agents/scene   mean={np.mean(n_agents_per_scene):.2f}")


if __name__ == "__main__":
    main()
