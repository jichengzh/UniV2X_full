#!/usr/bin/env python3
"""
Reliable whole-network breakdown via torch.profiler (hook-based timing was contaminated).
Profiles full model(ego) forward over real DAIR 2-agent frames; reports top ops by CUDA time
and total CUDA vs wall time (reveals GPU-bound vs CPU/sync/Python-bound eager overhead).
Usage: python codriving_profile.py --tag base --gpu <idle>
"""
import os, sys, time, argparse
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/jichengzhi/V2X"); V2XVERSE = Path("/home/jichengzhi/V2Xverse")
SPLIT = "/data/jichengzhi_dair/dair_eval/split_json"
DAIR_DATA = "/data/jichengzhi_dair/dair_eval/cooperative-vehicle-infrastructure"
CFG = {"base": (REPO/"output/codriving_pilot/collab_export/dair_centerpoint_codriving_4090.yaml", None),
       "p50": (REPO/"output/codriving_pilot/p50/config_finetune.yaml", REPO/"output/codriving_pilot/p50/net_epoch_bestval_at5_p50.pth")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="base"); ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--n", type=int, default=20); args = ap.parse_args()
    sys.path.insert(0, str(V2XVERSE)); os.chdir(V2XVERSE); torch.cuda.set_device(args.gpu)
    from opencood.tools import train_utils
    from opencood.tools.train_utils import to_device
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from torch.utils.data import DataLoader

    cfg, ckpt = CFG[args.tag]; hypes = load_yaml(str(cfg))
    hypes["data_dir"] = DAIR_DATA; hypes["root_dir"] = f"{SPLIT}/train.json"
    hypes["validate_dir"] = f"{SPLIT}/val.json"; hypes["test_dir"] = f"{SPLIT}/val.json"
    model = train_utils.create_model(hypes)
    if ckpt is not None:
        sd = torch.load(str(ckpt), map_location="cpu"); model.load_state_dict(sd, strict=False)
    else:
        _, model = train_utils.load_saved_model(str(cfg.parent), model)
    model = model.cuda().eval()
    ds = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=ds.collate_batch_test, shuffle=False)

    frames = []  # keep on CPU, move per-iter
    for b in loader:
        if int(b["ego"]["record_len"][0]) == 2:
            frames.append(b["ego"])
        if len(frames) >= args.n + 5: break

    with torch.inference_mode():
        for f in frames[:5]:
            model(to_device(f, "cuda"))
        torch.cuda.synchronize()
        # wall time
        walls = []
        for f in frames[5:]:
            g = to_device(f, "cuda")
            torch.cuda.synchronize(); t0 = time.time(); model(g); torch.cuda.synchronize()
            walls.append((time.time()-t0)*1000)
            del g; torch.cuda.empty_cache()
        wall_mean = float(np.mean(walls))
        # profiler
        from torch.profiler import profile, ProfilerActivity
        nprof = min(10, len(frames)-5)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for f in frames[5:5+nprof]:
                model(to_device(f, "cuda"))
            torch.cuda.synchronize()
    ka = prof.key_averages()
    tot_cuda = sum(k.self_cuda_time_total for k in ka) / 1e3 / nprof
    tot_cpu = sum(k.self_cpu_time_total for k in ka) / 1e3 / nprof
    print(f"\n=== {args.tag} whole-net: wall_mean={wall_mean:.1f}ms | sum_self_CUDA={tot_cuda:.1f}ms/frame | sum_self_CPU={tot_cpu:.1f}ms/frame ===")
    print(f"  (wall >> CUDA => CPU/sync/Python-bound eager overhead; wall ~ CUDA => GPU-bound)")
    print("\n--- top 15 ops by self CUDA time (ms/frame) ---")
    for k in sorted(ka, key=lambda x: x.self_cuda_time_total, reverse=True)[:15]:
        print(f"  {k.key[:45]:45s} cuda={k.self_cuda_time_total/1e3/nprof:7.3f}  cpu={k.self_cpu_time_total/1e3/nprof:7.3f}  n={k.count}")
    print("\n--- top 10 ops by self CPU time (ms/frame) ---")
    for k in sorted(ka, key=lambda x: x.self_cpu_time_total, reverse=True)[:10]:
        print(f"  {k.key[:45]:45s} cpu={k.self_cpu_time_total/1e3/nprof:7.3f}  cuda={k.self_cuda_time_total/1e3/nprof:7.3f}  n={k.count}")

if __name__ == "__main__":
    main()
