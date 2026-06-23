"""
L4 Master Runner - runs ALL 5 finetune+eval chains sequentially
Designed to be run ONCE from this script only.

GPU 0: cliff_a_wpg8 -> cliff_b_wpg8
GPU 1: wg_pair4 -> wg_pair5 -> wg_pair6

Writes: /home/jichengzhi/V2X/results/l4_b2_ap_finetune.json

Safety: checks for existing bestval ckpts before finetuning (resume if interrupted)
"""
import json
import os
import re
import subprocess
import sys
import yaml
from pathlib import Path
import torch

HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
REPO = Path("/home/jichengzhi/V2X")
RESULT_JSON = REPO / "results/l4_b2_ap_finetune.json"

GPU0_CONFIGS = [
    ("cliff_a_wpg8", [8, 16, 32], 8, 0),
    ("cliff_b_wpg8", [8,  8, 16], 8, 0),
]
GPU1_CONFIGS = [
    ("wg_pair4", [48, 96, 256], 4, 1),
    ("wg_pair5", [48, 32, 128], 4, 1),
    ("wg_pair6", [48, 64, 192], 4, 1),
]

def ckpt_dir(tag):
    return CKROOT / f"Pyramid_DAIR_m1_l4_{tag}_2026_06_21"

def check_finetune_done(d):
    """Check if finetune produced ckpts beyond epoch 23."""
    bests = [p for p in d.glob("net_epoch_bestval_at*.pth")
             if int(re.search(r"at(\d+)\.pth", p.name).group(1)) > 23]
    return len(bests) > 0

def finetune(tag, gpu):
    d = ckpt_dir(tag)
    if check_finetune_done(d):
        print(f"[{tag}] finetune already done (post-23 ckpt found), skipping")
        return 0

    log = REPO / f"results/l4_{tag}_finetune.log"
    cmd = [PY, "opencood/tools/train.py",
           "--hypes_yaml", str(d / "config.yaml"),
           "--model_dir", str(d)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] finetune on GPU{gpu}, log={log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(HEAL), env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] finetune done rc={rc}")
    return rc

def eval_ap(tag, gpu):
    d = ckpt_dir(tag)
    log = REPO / f"results/l4_{tag}_eval.log"

    # Remove stale eval yamls
    for ev in d.glob("eval*.yaml"):
        ev.unlink()
        print(f"[{tag}] removed stale {ev.name}")

    cmd = [PY, "opencood/tools/inference.py",
           "--model_dir", str(d),
           "--fusion_method", "intermediate"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] eval on GPU{gpu}, log={log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=str(HEAL), env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] eval done rc={rc}")

    # Parse AP
    ap = None
    for ev in sorted(d.glob("eval*.yaml")):
        try:
            y = yaml.safe_load(ev.read_text()) or {}
            ap30 = float(y.get("ap30") or y.get("ap_30") or 0)
            ap50 = float(y.get("ap_50") or y.get("ap50") or 0)
            ap70 = float(y.get("ap_70") or y.get("ap70") or 0)
            ap = {"ap30": round(ap30, 4), "ap50": round(ap50, 4), "ap70": round(ap70, 4)}
            print(f"[{tag}] AP from {ev.name}: {ap}")
            break
        except Exception as e:
            print(f"[{tag}] yaml parse err: {e}")

    if ap is None:
        try:
            txt = log.read_text()
            m = re.search(r"0\.3 is ([0-9.]+).*?0\.5 is ([0-9.]+).*?0\.7 is ([0-9.]+)", txt, re.DOTALL)
            if m:
                ap = {"ap30": round(float(m.group(1)), 4),
                      "ap50": round(float(m.group(2)), 4),
                      "ap70": round(float(m.group(3)), 4)}
                print(f"[{tag}] AP from log: {ap}")
        except Exception as e:
            print(f"[{tag}] log parse err: {e}")

    return ap

def get_best_ckpt(d):
    bests = [p for p in d.glob("net_epoch_bestval_at*.pth")]
    if not bests:
        return None
    return max(bests, key=lambda p: int(re.search(r"at(\d+)\.pth", p.name).group(1))).name

def run_config(tag, nf, wpg, gpu, results):
    print(f"\n{'='*60}")
    print(f"[{tag}] nf={nf} wpg={wpg} GPU={gpu}")
    print(f"{'='*60}")

    # Skip if already complete with valid AP
    if tag in results and results[tag].get("status") == "ok" and results[tag].get("ap70", 0) and results[tag]["ap70"] > 0.01:
        print(f"[{tag}] already done with valid AP, skipping")
        return

    rc = finetune(tag, gpu)

    d = ckpt_dir(tag)
    best = get_best_ckpt(d)

    ap = eval_ap(tag, gpu)

    results[tag] = {
        "num_filters": nf,
        "wpg": wpg,
        "groups": 32,
        "ap70": ap["ap70"] if ap else None,
        "ap50": ap["ap50"] if ap else None,
        "ap30": ap["ap30"] if ap else None,
        "bestval_epoch": best,
        "ckpt": str(d),
        "finetune_rc": rc,
        "status": "ok" if (ap and ap.get("ap70", 0) is not None and (ap.get("ap70", 0) > 0.01 or rc == 0)) else f"failed:rc={rc}",
    }
    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[{tag}] DONE: AP70={results[tag]['ap70']}")

def gpu_chain(configs, results):
    for tag, nf, wpg, gpu in configs:
        run_config(tag, nf, wpg, gpu, results)

def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else -1

    results = json.loads(RESULT_JSON.read_text()) if RESULT_JSON.exists() else {}

    if gpu == 0:
        print("=== GPU 0 CHAIN ===")
        gpu_chain(GPU0_CONFIGS, results)
    elif gpu == 1:
        print("=== GPU 1 CHAIN ===")
        gpu_chain(GPU1_CONFIGS, results)
    else:
        print("Usage: python l4_master_runner.py [0|1]")
        sys.exit(1)

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
