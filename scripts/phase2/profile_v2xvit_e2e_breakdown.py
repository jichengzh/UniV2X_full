"""V2X-ViT e2e module-breakdown profiler (Q1 gating experiment).

Tests the UNTESTED hypothesis "V2X-ViT's bottleneck is the transformer fusion,
not the conv backbone".  Reuses eval_v2xvit_baseline_a1 loading; adds CUDA-event
timing hooks on the model's top-level submodules + drills into the fusion
(V2XTransformer: HMSA+MSwin).  Reports per-module mean ms + % of model forward,
and model-forward vs postprocess(NMS) split of the full inference call.

Run:  CUDA_VISIBLE_DEVICES=5 python scripts/phase2/profile_v2xvit_e2e_breakdown.py
Output: results/v2xvit_e2e_breakdown.json
"""
import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset

CKPT_DIR = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE = CKPT_DIR / "net_epoch_bestval_at17.pth"
N_WARMUP = 10
N_SAMPLES = 60
OUT = REPO_ROOT / "results" / "v2xvit_e2e_breakdown.json"


def main():
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
    model = train_utils.create_model(hypes)
    state = torch.load(CKPT_FILE, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    # ---- show top-level structure (class names) so the breakdown is auditable ----
    print("=== model top-level named_children ===", flush=True)
    top = list(model.named_children())
    for name, mod in top:
        n_params = sum(p.numel() for p in mod.parameters())
        print(f"  {name:24s} {type(mod).__name__:28s} params={n_params/1e6:.3f}M", flush=True)

    # ---- identify the fusion / transformer module(s) by class or name ----
    def is_fusion(name, mod):
        cn = type(mod).__name__.lower()
        return ("transformer" in cn or "fusion" in cn or "v2xvit" in cn
                or "fusion" in name.lower() or name.lower() in ("fusion_net", "pyramid_backbone"))

    # Hook every top-level child + flag fusion ones; one accumulator per child.
    timed = list(top)
    acc_ms = defaultdict(float)
    events = {}   # name -> (start_event, end_event)

    def mk_pre(name):
        def hook(_m, _inp):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            events[name] = (s, e)
        return hook

    def mk_post(name):
        def hook(_m, _inp, _out):
            s, e = events[name]
            e.record()
        return hook

    handles = []
    for name, mod in timed:
        handles.append(mod.register_forward_pre_hook(mk_pre(name)))
        handles.append(mod.register_forward_hook(mk_post(name)))

    # full model forward timer
    model_events = {}

    def model_pre(_m, _inp):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); model_events["m"] = (s, e)

    def model_post(_m, _inp, _out):
        model_events["m"][1].record()
    handles.append(model.register_forward_pre_hook(model_pre))
    handles.append(model.register_forward_hook(model_post))

    dataset = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    print(f"[profile] val size={len(dataset)}", flush=True)

    device = torch.device("cuda")
    fusion_names = {name for name, mod in timed if is_fusion(name, mod)}
    print(f"[profile] fusion/transformer modules = {sorted(fusion_names)}", flush=True)

    model_fwd_ms = 0.0
    total_infer_ms = 0.0
    n_done = 0
    n_seen = 0
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_done >= N_WARMUP + N_SAMPLES:
                break
            batch_data = train_utils.to_device(batch_data, device)
            torch.cuda.synchronize()
            t_inf0 = time.time()
            inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
            torch.cuda.synchronize()
            t_inf = (time.time() - t_inf0) * 1000.0

            warm = n_seen < N_WARMUP
            n_seen += 1
            if warm:
                continue
            n_done_eff = n_seen - N_WARMUP
            # accumulate per-module elapsed
            for name in list(events):
                s, e = events[name]
                acc_ms[name] += s.elapsed_time(e)
            if "m" in model_events:
                ms, me = model_events["m"]
                model_fwd_ms += ms.elapsed_time(me)
            total_infer_ms += t_inf
            n_done = n_done_eff + N_WARMUP

    for h in handles:
        h.remove()

    N = max(1, n_seen - N_WARMUP)
    per_mod = {k: round(v / N, 4) for k, v in sorted(acc_ms.items(), key=lambda x: -x[1])}
    model_fwd = round(model_fwd_ms / N, 4)
    total_inf = round(total_infer_ms / N, 4)
    postproc = round(total_inf - model_fwd, 4)
    fusion_ms = round(sum(acc_ms[n] for n in fusion_names) / N, 4)

    report = {
        "model": "v2xvit (HeterBaseline_DAIR_lidar_v2xvit)",
        "n_samples": N, "device": torch.cuda.get_device_name(0),
        "per_module_ms_mean": per_mod,
        "fusion_modules": sorted(fusion_names),
        "fusion_ms_mean": fusion_ms,
        "model_forward_ms_mean": model_fwd,
        "postprocess_ms_mean": postproc,             # NMS + decode (CPU/GPU)
        "total_inference_ms_mean": total_inf,
        "fusion_pct_of_model_forward": round(100 * fusion_ms / model_fwd, 1) if model_fwd else None,
        "fusion_pct_of_total_inference": round(100 * fusion_ms / total_inf, 1) if total_inf else None,
        "model_forward_pct_of_total": round(100 * model_fwd / total_inf, 1) if total_inf else None,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== V2X-ViT e2e breakdown ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[written] {OUT}", flush=True)


if __name__ == "__main__":
    main()
