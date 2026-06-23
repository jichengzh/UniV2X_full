#!/usr/bin/env python3
"""
Whole-network (deployment口径) latency: original decode vs cached-grid decode monkeypatch.
Quantifies §4-step1: how much the decode CPU-meshgrid fix moves the FULL-network wall,
and whether pruning (base/p50/p75) becomes more visible at whole-net once decode is fixed.

- Times complete model(ego) forward over real DAIR 2-agent frames (eager, all-PyTorch).
- monkeypatch: replaces generate_predicted_boxes with a (H,W)-keyed GPU-cached-grid version
  (zero edits to framework files; method rebound per instance).
- contention-gate: samples GPU util/power throughout the timed loop in a bg thread;
  if max util from contention is high, flags the run (per HANDOFF v2 §7 lesson).
Usage: python codriving_wholenet_decodefix.py --gpu 5
"""
import os, sys, time, json, argparse, threading, subprocess, types
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/jichengzhi/V2X"); V2XVERSE = Path("/home/jichengzhi/V2Xverse")
SPLIT = "/data/jichengzhi_dair/dair_eval/split_json"
DAIR_DATA = "/data/jichengzhi_dair/dair_eval/cooperative-vehicle-infrastructure"
CFG = {
    "base": (REPO / "output/codriving_pilot/collab_export/dair_centerpoint_codriving_4090.yaml", None),
    "p50":  (REPO / "output/codriving_pilot/p50/config_finetune.yaml",
             REPO / "output/codriving_pilot/p50/net_epoch_bestval_at5_p50.pth"),
    "p75":  (REPO / "output/codriving_pilot/p75/config_finetune.yaml", None),
}


class GpuMon:
    """Background sampler of one GPU's util/power; reports max util + power range during run."""
    def __init__(self, gpu, period=0.3):
        self.gpu, self.period, self.stop = gpu, period, False
        self.utils, self.powers = [], []
    def _loop(self):
        while not self.stop:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits", "-i", str(self.gpu)], timeout=2).decode().strip()
                u, p = out.split(", "); self.utils.append(float(u)); self.powers.append(float(p))
            except Exception:
                pass
            time.sleep(self.period)
    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True); self.t.start(); return self
    def __exit__(self, *a):
        self.stop = True; self.t.join(timeout=2)
    def report(self):
        return {"util_max": max(self.utils) if self.utils else None,
                "util_mean": round(float(np.mean(self.utils)), 1) if self.utils else None,
                "power_min": min(self.powers) if self.powers else None,
                "power_max": max(self.powers) if self.powers else None,
                "n_samples": len(self.utils)}


def make_decode_cached(model):
    osf, vs, clr = model.out_size_factor, model.voxel_size, model.cav_lidar_range
    cache = {}
    def decode(self, cls_preds, box_preds, dir_cls_preds=None):
        bp = box_preds.permute(0, 2, 3, 1).contiguous()
        b, H, W, cs = bp.size()
        bp = bp.reshape(b, H * W, cs)
        key = (H, W, cls_preds.device)
        if key not in cache:
            ys0, xs0 = torch.meshgrid(torch.arange(0, H, device=cls_preds.device),
                                      torch.arange(0, W, device=cls_preds.device))
            cache[key] = (xs0.reshape(1, -1, 1).float(), ys0.reshape(1, -1, 1).float())
        xs0, ys0 = cache[key]
        h = bp[..., 3:4] * osf * vs[0]; w = bp[..., 4:5] * osf * vs[1]; l = bp[..., 5:6] * osf * vs[2]
        dim = torch.cat([h, w, l], dim=-1)
        hei = bp[..., 2:3] * osf * vs[2] + clr[2]
        rot = torch.atan2(bp[..., 6:7], bp[..., 7:8])
        xs = (xs0 + bp[:, :, 0:1]) * osf * vs[0] + clr[0]
        ys = (ys0 + bp[:, :, 1:2]) * osf * vs[1] + clr[1]
        return cls_preds, torch.cat([xs, ys, hei, dim, rot], dim=2)
    return decode


def load_model(tag):
    from opencood.tools import train_utils
    from opencood.hypes_yaml.yaml_utils import load_yaml
    cfg_path, ckpt = CFG[tag]
    hypes = load_yaml(str(cfg_path))
    hypes["data_dir"] = DAIR_DATA; hypes["root_dir"] = f"{SPLIT}/train.json"
    hypes["validate_dir"] = f"{SPLIT}/val.json"; hypes["test_dir"] = f"{SPLIT}/val.json"
    model = train_utils.create_model(hypes)
    if ckpt is not None:
        sd = torch.load(str(ckpt), map_location="cpu")
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=False)
    else:
        _, model = train_utils.load_saved_model(str(cfg_path.parent), model)
    return model.cuda().eval(), hypes


def get_frames(hypes, n):
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader
    ds = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=ds.collate_batch_test, shuffle=False)
    frames = []
    for b in loader:
        if int(b["ego"]["record_len"][0]) == 2:
            frames.append(b["ego"])
        if len(frames) >= n + 5:
            break
    return frames


def _util(gpu):
    try:
        return float(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "-i", str(gpu)], timeout=2).decode().strip())
    except Exception:
        return 0.0


def time_wholenet(model, frames, gpu, warmup=5, runs=35, gate=50):
    """Per-frame contention gating: sample util right before+after each frame; if either
    exceeds `gate` (well above own eager ~21% ceiling), mark frame contended and exclude.
    Returns clean (gated) and raw distributions."""
    from opencood.tools.train_utils import to_device
    walls, contended = [], []
    with torch.inference_mode():
        for f in frames[:warmup]:
            model(to_device(f, "cuda"))
        torch.cuda.synchronize()
        for f in frames[warmup:warmup + runs]:
            g = to_device(f, "cuda")
            u0 = _util(gpu)
            torch.cuda.synchronize(); t0 = time.time(); model(g); torch.cuda.synchronize()
            w = (time.time() - t0) * 1000
            u1 = _util(gpu)
            if max(u0, u1) > gate:
                contended.append(w)
            else:
                walls.append(w)
            del g; torch.cuda.empty_cache()
    clean = walls if walls else contended  # fall back if all contended
    return {"p50": round(float(np.percentile(clean, 50)), 2),
            "p10": round(float(np.percentile(clean, 10)), 2),
            "p90": round(float(np.percentile(clean, 90)), 2),
            "mean": round(float(np.mean(clean)), 2),
            "n_clean": len(walls), "n_contended": len(contended)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()
    sys.path.insert(0, str(V2XVERSE)); os.chdir(V2XVERSE); torch.cuda.set_device(args.gpu)

    results = {}
    for tag in ["base", "p50", "p75"]:
        model, hypes = load_model(tag)
        frames = get_frames(hypes, args.n)
        orig_method = model.generate_predicted_boxes

        # (1) original decode
        o = time_wholenet(model, frames, args.gpu)
        # (2) cached-grid decode (monkeypatch, instance-bound)
        model.generate_predicted_boxes = types.MethodType(make_decode_cached(model), model)
        c = time_wholenet(model, frames, args.gpu)
        model.generate_predicted_boxes = orig_method  # restore (hygiene)

        results[tag] = {
            "n_frames": len(frames),
            "orig": o, "cached": c,
            "decodefix_saving_ms": round(o["p50"] - c["p50"], 2),
            "decodefix_speedup": round(o["p50"] / c["p50"], 3),
        }
        print(f"=== {tag} === orig p50={o['p50']} [{o['p10']}-{o['p90']}] (clean {o['n_clean']}/{o['n_clean']+o['n_contended']}) | "
              f"cached p50={c['p50']} [{c['p10']}-{c['p90']}] (clean {c['n_clean']}/{c['n_clean']+c['n_contended']}) | "
              f"decode-fix {o['p50']-c['p50']:.1f}ms ({o['p50']/c['p50']:.2f}x)")
        del model; torch.cuda.empty_cache()

    # cross-tag: pruning visibility at whole-net (cached-decode口径, median)
    if "base" in results and "p50" in results and "p75" in results:
        b = results["base"]["cached"]["p50"]; p5 = results["p50"]["cached"]["p50"]; p7 = results["p75"]["cached"]["p50"]
        print(f"\n=== pruning at whole-net (cached decode, p50 median): "
              f"base {b}ms -> p50 {p5}ms ({b/p5:.2f}x) -> p75 {p7}ms ({b/p7:.2f}x) "
              f"[vs isolated backbone eager 1.56x] ===")

    out_path = REPO / "output/codriving_pilot/logs/wholenet_decodefix.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
