#!/usr/bin/env python3
"""
CoDriving WHOLE-NETWORK per-stage timing (all-PyTorch, no TRT).
Reconciles "backbone is the bottleneck" vs e2e claims, and measures
whole-network speedup from pruning (base vs p50 vs p75) in a CONSISTENT regime.

Uses forward hooks (CUDA events) -> captures EVERY invocation of each module
(backbone runs TWICE: standalone + inside multi_scale fusion).

Usage:
  python codriving_perstage_timing.py --tag base
  python codriving_perstage_timing.py --tag p50
  python codriving_perstage_timing.py --tag p75
"""
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/jichengzhi/V2X")
V2XVERSE = Path("/home/jichengzhi/V2Xverse")
DAIR_DATA = "/data/jichengzhi_dair/dair_eval/cooperative-vehicle-infrastructure"

CFG = {
    "base": (REPO / "output/codriving_pilot/collab_export/dair_centerpoint_codriving_4090.yaml", None),  # dir-load bestval
    "p50":  (REPO / "output/codriving_pilot/p50/config_finetune.yaml",
             REPO / "output/codriving_pilot/p50/net_epoch_bestval_at5_p50.pth"),
    "p75":  (REPO / "output/codriving_pilot/p75/config_finetune.yaml", None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(CFG.keys()))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=15)
    args = ap.parse_args()

    sys.path.insert(0, str(V2XVERSE))
    os.chdir(V2XVERSE)
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from torch.utils.data import DataLoader

    cfg_path, ckpt_path = CFG[args.tag]
    hypes = load_yaml(str(cfg_path))
    hypes["data_dir"] = DAIR_DATA
    SPLIT = "/data/jichengzhi_dair/dair_eval/split_json"
    hypes["root_dir"] = f"{SPLIT}/train.json"
    hypes["test_dir"] = f"{SPLIT}/val.json"
    hypes["validate_dir"] = f"{SPLIT}/val.json"

    model = train_utils.create_model(hypes)
    if ckpt_path is not None:
        sd = torch.load(str(ckpt_path), map_location="cpu")
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        m, u = model.load_state_dict(sd, strict=False)
        print(f"[load] {args.tag} ckpt direct, missing={len(m)} unexpected={len(u)}")
    else:
        _, model = train_utils.load_saved_model(str(cfg_path.parent), model)
        print(f"[load] {args.tag} dir-bestval from {cfg_path.parent}")
    model = model.cuda().eval()
    nfilt = hypes["model"]["args"]["base_bev_backbone"]["num_filters"]
    print(f"[{args.tag}] num_filters={nfilt}")

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    # ---- hook timing (CUDA events), accumulate per module across ALL calls ----
    stage_modules = {
        "pillar_vfe": model.pillar_vfe,
        "scatter": model.scatter,
        "backbone": model.backbone,        # fires TWICE (standalone + fusion multi-scale)
        "fusion_net": model.fusion_net,    # includes 2nd backbone + where2comm attention
        "cls_head": model.cls_head,
        "reg_head": model.reg_head,
    }
    acc = {k: 0.0 for k in stage_modules}
    cnt = {k: 0 for k in stage_modules}
    ev = {k: [] for k in stage_modules}  # (start,end) events per active call
    handles = []

    def mk_pre(name):
        def pre(mod, inp):
            s = torch.cuda.Event(enable_timing=True); s.record()
            ev[name].append(s)
        return pre

    def mk_post(name):
        def post(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True); e.record()
            s = ev[name].pop()
            # defer elapsed to after sync; store pair
            post._pairs.append((name, s, e))
        post._pairs = []
        return post

    posts = {}
    for name, mod in stage_modules.items():
        handles.append(mod.register_forward_pre_hook(mk_pre(name)))
        p = mk_post(name)
        posts[name] = p
        handles.append(mod.register_forward_hook(p))

    # wrap generate_predicted_boxes (a method, not a module) to time decode
    decode_acc = {"t": 0.0, "n": 0}
    _orig_gen = model.generate_predicted_boxes
    def _timed_gen(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = _orig_gen(*a, **k); e.record()
        _timed_gen._pairs.append((s, e))
        return r
    _timed_gen._pairs = []
    model.generate_predicted_boxes = _timed_gen

    totals = []
    n_done = 0
    with torch.inference_mode():
        for batch in loader:
            ego = batch["ego"]
            rec = ego.get("record_len")
            if rec is None or int(rec[0]) != 2:
                continue
            # move to cuda
            from opencood.tools.train_utils import to_device
            ego = to_device(ego, "cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model(ego)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000.0
            # accumulate stage times (skip during warmup)
            if n_done >= args.warmup:
                totals.append(dt)
                for name, s, e in sum([p._pairs for p in posts.values()], []):
                    acc[name] += s.elapsed_time(e)
                    cnt[name] += 1
                for s, e in _timed_gen._pairs:
                    decode_acc["t"] += s.elapsed_time(e); decode_acc["n"] += 1
            for p in posts.values():
                p._pairs.clear()
            _timed_gen._pairs.clear()
            n_done += 1
            if n_done >= args.n:
                break

    for h in handles:
        h.remove()

    nmeas = len(totals)
    total_p50 = float(np.percentile(totals, 50))
    total_mean = float(np.mean(totals))
    print(f"\n=== {args.tag} WHOLE-NETWORK (all-PyTorch) over {nmeas} 2-agent frames ===")
    print(f"whole_net p50={total_p50:.2f}ms  mean={total_mean:.2f}ms")
    print(f"{'stage':<12} {'ms/frame':>10} {'calls/frame':>12} {'%of_module_sum':>14}")
    per_frame = {k: acc[k] / nmeas for k in acc}
    msum = sum(per_frame.values())
    rows = []
    for k in stage_modules:
        pct = 100.0 * per_frame[k] / msum if msum else 0
        print(f"{k:<12} {per_frame[k]:>10.3f} {cnt[k]/nmeas:>12.1f} {pct:>13.1f}%")
        rows.append({"stage": k, "ms_per_frame": round(per_frame[k], 4),
                     "calls_per_frame": round(cnt[k] / nmeas, 2), "pct_of_module_sum": round(pct, 1)})
    decode_pf = decode_acc["t"] / nmeas
    print(f"{'(module sum)':<12} {msum:>10.3f}")
    print(f"{'decode(genbox)':<12} {decode_pf:>10.3f} {decode_acc['n']/nmeas:>12.1f}")
    print(f"residual overhead (whole - modulesum - decode) ~= {total_mean - msum - decode_pf:.2f}ms")

    out = {"tag": args.tag, "num_filters": nfilt, "n_frames": nmeas,
           "whole_net_p50_ms": round(total_p50, 3), "whole_net_mean_ms": round(total_mean, 3),
           "module_sum_ms": round(msum, 3),
           "decode_genbox_ms": round(decode_pf, 3), "decode_calls_per_frame": round(decode_acc["n"]/nmeas, 2),
           "residual_overhead_ms": round(total_mean - msum - decode_pf, 3), "stages": rows,
           "note": "all-PyTorch whole-network forward; backbone fires twice (standalone+multiscale fusion)"}
    outp = REPO / f"output/codriving_pilot/logs/perstage_{args.tag}.json"
    json.dump(out, open(outp, "w"), indent=2)
    print(f"saved {outp}")


if __name__ == "__main__":
    main()
