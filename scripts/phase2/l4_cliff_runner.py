"""L4 DELIVERABLE A: cliff search on GPU 0.

Configs (wpg=8 to unlock extreme pruning):
  cliff_a_wpg8: [8,16,32] wpg=8 groups=32  (~87.5% channel reduction)
  cliff_b_wpg8: [8,8,16]  wpg=8 groups=32  (~93.75% channel reduction)

Runs serially on GPU 0. Updates results/l4_b2_ap_finetune.json.

Usage:
  cd /home/jichengzhi/V2X
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/jichengzhi/heal_research/HEAL \
  /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \
    scripts/phase2/l4_cliff_runner.py \
    > results/l4_cliff_chain.log 2>&1 &
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import torch
import yaml

REPO = Path("/home/jichengzhi/V2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
EPOCHES = 31
GROUPS = 32
GPU = 0
MASTER_PORT = 29810
RESULT_JSON = REPO / "results/l4_b2_ap_finetune.json"

CONFIGS = [
    ("cliff_a_wpg8", [8, 16, 32], 8),
    ("cliff_b_wpg8", [8,  8, 16], 8),
]


def ckpt_dir_for(tag):
    return CKROOT / f"Pyramid_DAIR_m1_l4_{tag}_2026_06_21"


def validate_config(nf, wpg):
    for p in nf:
        w = int(p * wpg / 64) * GROUPS
        ipg = w // GROUPS
        if w <= 0 or ipg < 1:
            print(f"  INFEASIBLE: planes={p} wpg={wpg} groups={GROUPS} → width={w} ipg={ipg}")
            return False
        print(f"  validate: planes={p} wpg={wpg} groups={GROUPS} → width={w} ipg={ipg} [OK]")
    return True


def prune(tag, nf, wpg):
    out_dir = ckpt_dir_for(tag)
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"

    post_epochs = [
        p for p in out_dir.glob("net_epoch*.pth")
        if re.search(r"epoch(?:_bestval_at)?(\d+)", p.name) and
        int(re.search(r"epoch(?:_bestval_at)?(\d+)", p.name).group(1)) > 23
    ]
    if post_epochs:
        print(f"[{tag}] post-init epochs exist → skip prune")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "tools/structural_prune_pyramid.py",
           "--orig-dir", BASELINE,
           "--out-dir", str(out_dir),
           "--num-filters-new", ",".join(str(x) for x in nf),
           "--width-per-group", str(wpg),
           "--groups", str(GROUPS)]
    env = {**os.environ, "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] pruning {nf} wpg={wpg} ...")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    print(r.stdout[-3000:])
    if r.returncode != 0 or not init_ckpt.exists():
        print(f"[{tag}] prune FAILED rc={r.returncode}\nSTDERR: {r.stderr[-500:]}")
        return None
    print(f"[{tag}] prune OK → {init_ckpt}")
    return out_dir


def flatten_ckpt(out_dir, tag):
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    if not init_ckpt.exists():
        return
    sd = torch.load(init_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        torch.save(sd["model_state_dict"], init_ckpt)
        print(f"[{tag}] flattened init ckpt (unwrapped model_state_dict)")
    else:
        print(f"[{tag}] init ckpt already flat")


def patch_config(out_dir, tag):
    cfg = out_dir / "config.yaml"
    if not cfg.exists():
        return
    s = cfg.read_text()
    s2 = re.sub(r"epoches:\s*\d+", f"epoches: {EPOCHES}", s)
    if s2 != s:
        cfg.write_text(s2)
        print(f"[{tag}] patched config epoches → {EPOCHES}")


def finetune(out_dir, tag):
    cfg = out_dir / "config.yaml"
    log = REPO / f"results/l4_{tag}_finetune.log"
    cmd = [PY, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={MASTER_PORT}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(out_dir),
           "--half"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] finetune GPU{GPU} log={log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=HEAL, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] finetune done rc={rc}")
    return rc


def find_best_ckpt(ft_dir):
    bests = list(ft_dir.glob("net_epoch_bestval_at*.pth"))
    if bests:
        return max(bests, key=lambda p: int(re.search(r"at(\d+)\.pth", p.name).group(1)))
    all_pths = [p for p in ft_dir.glob("net_epoch*.pth") if "bestval" not in p.name]
    if all_pths:
        return max(all_pths, key=lambda p: int(re.search(r"epoch(\d+)", p.name).group(1)))
    return None


def eval_ap(tag, out_dir):
    log = REPO / f"results/l4_{tag}_eval.log"
    for ev in out_dir.glob("eval*.yaml"):
        ev.unlink()
        print(f"[{tag}] removed stale {ev.name}")

    cmd = [PY, "opencood/tools/inference.py",
           "--model_dir", str(out_dir),
           "--fusion_method", "intermediate"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU), "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] eval AP GPU{GPU} → log={log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=HEAL, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] eval rc={rc}")

    ap = None
    for ev in out_dir.glob("eval*.yaml"):
        try:
            y = yaml.safe_load(ev.read_text()) or {}
            ap30 = float(y.get("ap30") or y.get("ap_30") or 0)
            ap50 = float(y.get("ap_50") or y.get("ap50") or 0)
            ap70 = float(y.get("ap_70") or y.get("ap70") or 0)
            ap = {"ap30": round(ap30, 4), "ap50": round(ap50, 4), "ap70": round(ap70, 4)}
            print(f"[{tag}] AP from {ev.name}: {ap}")
            break
        except Exception as e:
            print(f"[{tag}] yaml parse error: {e}")

    if ap is None:
        try:
            txt = (REPO / f"results/l4_{tag}_eval.log").read_text()
            m = re.search(
                r"Average Precision at IOU 0\.3 is ([0-9.]+).*?0\.5 is ([0-9.]+).*?0\.7 is ([0-9.]+)",
                txt, re.DOTALL)
            if m:
                ap = {"ap30": round(float(m.group(1)), 4),
                      "ap50": round(float(m.group(2)), 4),
                      "ap70": round(float(m.group(3)), 4)}
                print(f"[{tag}] AP from stdout: {ap}")
        except Exception as e:
            print(f"[{tag}] stdout parse error: {e}")

    return ap


def load_results():
    if RESULT_JSON.exists():
        return json.loads(RESULT_JSON.read_text())
    return {}


def save_results(results):
    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved to {RESULT_JSON}")


def run_one(tag, nf, wpg, results):
    print(f"\n{'='*60}")
    print(f"[{tag}] nf={nf} wpg={wpg} groups={GROUPS} GPU={GPU}")
    print(f"{'='*60}")

    # Skip if already done
    if tag in results and results[tag].get("status") == "ok":
        print(f"[{tag}] already done (status=ok), skipping")
        return

    if not validate_config(nf, wpg):
        results[tag] = {"num_filters": nf, "wpg": wpg, "groups": GROUPS,
                        "ap70": None, "ap50": None,
                        "status": f"infeasible:width=0 for nf={nf} wpg={wpg} g={GROUPS}"}
        save_results(results)
        return

    out_dir = prune(tag, nf, wpg)
    if out_dir is None:
        results[tag] = {"num_filters": nf, "wpg": wpg, "groups": GROUPS,
                        "ap70": None, "ap50": None, "status": "prune_failed"}
        save_results(results)
        return

    flatten_ckpt(out_dir, tag)
    patch_config(out_dir, tag)

    rc = finetune(out_dir, tag)

    best = find_best_ckpt(out_dir)
    bestval_name = best.name if best else None

    ap = eval_ap(tag, out_dir)

    results[tag] = {
        "num_filters": nf,
        "wpg": wpg,
        "groups": GROUPS,
        "ap70": ap["ap70"] if ap else None,
        "ap50": ap["ap50"] if ap else None,
        "ap30": ap["ap30"] if ap else None,
        "bestval_epoch": bestval_name,
        "ckpt": str(out_dir),
        "finetune_rc": rc,
        "status": "ok" if ap else f"eval_failed:rc={rc}",
    }
    save_results(results)
    print(f"[{tag}] COMPLETE: AP70={results[tag]['ap70']}")


def main():
    print("=== L4 Cliff Runner (GPU 0) ===")
    results = load_results()
    for tag, nf, wpg in CONFIGS:
        run_one(tag, nf, wpg, results)
    print("\n=== DONE ===")
    print(json.dumps({k: results[k] for k in results if k.startswith("cliff")}, indent=2))


if __name__ == "__main__":
    main()
