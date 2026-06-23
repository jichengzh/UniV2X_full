"""L4 — DELIVERABLE A (cliff) + DELIVERABLE B2 (wg_pair) finetune launcher.

Deliverable A: AP cliff search with wpg=8 to unlock extreme pruning.
  - cliff_a_wpg8: [8,16,32] wpg=8 groups=32 → ~87.5% channel reduction
  - cliff_b_wpg8: [8,8,16]  wpg=8 groups=32 → ~93.75% channel reduction

Deliverable B2: W_g AP70 for rank-flip pairs 4/5/6.
  - wg_pair4: [48,96,256]  → calibration
  - wg_pair5: [48,32,128]  → calibration
  - wg_pair6: [48,64,192]  → calibration

GPU assignment:
  - GPU 0: cliff_a_wpg8 finetune (then cliff_b_wpg8 serially)
  - GPU 1: wg_pair4 finetune (then pair5, then pair6 serially)

Protocol (stage_a):
  - init from baseline at epoch23, finetune to epoch31 (8 epochs)
  - wpg=4 for B2 configs (matches base model), wpg=8 for cliff configs
  - flatten ckpt (unwrap model_state_dict) before finetune
  - HEAL train_ddp.py with --half

Output: results/l4_b2_ap_finetune.json
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import torch
import yaml

REPO = Path("/home/jichengzhi/V2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
CKROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
EPOCHES = 31   # init@23 → epoch 31 = 8 finetune epochs
GROUPS = 32
MASTER_PORT_BASE = 29800  # offset to avoid clash

# All configs: (tag, num_filters, wpg, gpu)
CONFIGS = [
    # DELIVERABLE A: cliff search
    ("cliff_a_wpg8", [8, 16, 32], 8, 0),
    ("cliff_b_wpg8", [8,  8, 16], 8, 0),   # serial after cliff_a on GPU 0
    # DELIVERABLE B2: W_g pairs
    ("wg_pair4", [48, 96, 256], 4, 1),
    ("wg_pair5", [48, 32, 128], 4, 1),      # serial after pair4 on GPU 1
    ("wg_pair6", [48, 64, 192], 4, 1),      # serial after pair5 on GPU 1
]

RESULT_JSON = REPO / "results/l4_b2_ap_finetune.json"


def ckpt_dir_for(tag: str) -> Path:
    return CKROOT / f"Pyramid_DAIR_m1_l4_{tag}_2026_06_21"


def validate_config(nf: list, wpg: int, groups: int) -> bool:
    """Check HEAL width formula: width = int(planes * wpg / 64) * groups >= 1 per group."""
    for p in nf:
        w = int(p * wpg / 64) * groups
        ipg = w // groups
        if w <= 0 or ipg < 1:
            return False
    return True


def prune(tag: str, nf: list, wpg: int) -> Path | None:
    """Run structural_prune_pyramid.py. Returns out_dir or None on failure."""
    out_dir = ckpt_dir_for(tag)
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"

    # Skip if already done (post-init epochs exist)
    post_epochs = [
        p for p in out_dir.glob("net_epoch*.pth")
        if re.search(r"epoch(?:_bestval_at)?(\d+)", p.name) and
        int(re.search(r"epoch(?:_bestval_at)?(\d+)", p.name).group(1)) > 23
    ]
    if post_epochs:
        print(f"[{tag}] post-init ckpt exists → skip prune (will resume)")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "tools/structural_prune_pyramid.py",
           "--orig-dir", BASELINE,
           "--out-dir", str(out_dir),
           "--num-filters-new", ",".join(str(x) for x in nf),
           "--width-per-group", str(wpg),
           "--groups", str(GROUPS)]
    env = {**os.environ, "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] pruning {nf} wpg={wpg} groups={GROUPS} ...")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    print(f"[{tag}] prune stdout:\n{r.stdout[-2000:]}")
    if r.returncode != 0:
        print(f"[{tag}] prune FAILED rc={r.returncode} stderr:\n{r.stderr[-1000:]}")
        return None
    if not init_ckpt.exists():
        print(f"[{tag}] prune FAILED: init_ckpt not created at {init_ckpt}")
        return None
    print(f"[{tag}] prune OK → {init_ckpt}")
    return out_dir


def flatten_ckpt(out_dir: Path, tag: str):
    """Unwrap {'model_state_dict': ...} → flat state_dict (CLAUDE.md ckpt trap)."""
    init_ckpt = out_dir / "net_epoch_bestval_at23.pth"
    if not init_ckpt.exists():
        print(f"[{tag}] no init ckpt to flatten (already replaced by training)")
        return
    sd = torch.load(init_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        torch.save(sd["model_state_dict"], init_ckpt)
        print(f"[{tag}] flattened init ckpt (unwrapped model_state_dict key)")
    else:
        print(f"[{tag}] init ckpt already flat (no unwrap needed)")


def patch_config(out_dir: Path, tag: str):
    """Patch epoches in config.yaml to EPOCHES."""
    cfg = out_dir / "config.yaml"
    if not cfg.exists():
        print(f"[{tag}] WARNING: config.yaml not found in {out_dir}")
        return
    s = cfg.read_text()
    s2 = re.sub(r"epoches:\s*\d+", f"epoches: {EPOCHES}", s)
    if s2 != s:
        cfg.write_text(s2)
        print(f"[{tag}] patched config epoches → {EPOCHES}")
    else:
        print(f"[{tag}] config epoches already {EPOCHES}")


def find_best_ckpt(ft_dir: Path) -> Path | None:
    bests = list(ft_dir.glob("net_epoch_bestval_at*.pth"))
    if bests:
        return max(bests, key=lambda p: int(re.search(r"at(\d+)\.pth", p.name).group(1)))
    # fallback
    all_pths = [p for p in ft_dir.glob("net_epoch*.pth") if "bestval" not in p.name]
    if all_pths:
        return max(all_pths, key=lambda p: int(re.search(r"epoch(\d+)", p.name).group(1)))
    return None


def finetune(out_dir: Path, tag: str, gpu: int) -> int:
    """Launch HEAL train_ddp.py (blocking). Returns rc."""
    cfg = out_dir / "config.yaml"
    log = REPO / f"results/l4_{tag}_finetune.log"
    port = MASTER_PORT_BASE + gpu * 10
    cmd = [PY, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={port}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(out_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] finetune → GPU{gpu} port={port} log={log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=HEAL, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] finetune done rc={rc}")
    return rc


def eval_ap(tag: str, out_dir: Path, gpu: int) -> dict | None:
    """Run HEAL inference.py and parse AP from eval.yaml."""
    log = REPO / f"results/l4_{tag}_eval.log"
    # Remove stale eval yaml
    eval_yaml = out_dir / "eval_intermediate_102.4_102.4.yaml"
    # look for any eval yaml
    for ev in out_dir.glob("eval*.yaml"):
        ev.unlink()
        print(f"[{tag}] removed stale {ev.name}")

    cmd = [PY, "opencood/tools/inference.py",
           "--model_dir", str(out_dir),
           "--fusion_method", "intermediate"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    print(f"[{tag}] eval AP on GPU{gpu} → log={log.name}")
    with open(log, "w") as lf:
        rc = subprocess.call(cmd, cwd=HEAL, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"[{tag}] eval rc={rc}")

    # Parse eval yaml (HEAL writes eval_<infer_info>.yaml or eval.yaml)
    ap = None
    for ev in out_dir.glob("eval*.yaml"):
        try:
            y = yaml.safe_load(ev.read_text()) or {}
            ap30 = y.get("ap30") or y.get("ap_30", 0)
            ap50 = y.get("ap_50") or y.get("ap50", 0)
            ap70 = y.get("ap_70") or y.get("ap70", 0)
            ap = {"ap30": round(float(ap30), 4),
                  "ap50": round(float(ap50), 4),
                  "ap70": round(float(ap70), 4)}
            print(f"[{tag}] AP from {ev.name}: AP30={ap['ap30']} AP50={ap['ap50']} AP70={ap['ap70']}")
            break
        except Exception as e:
            print(f"[{tag}] failed to parse {ev}: {e}")

    if ap is None:
        # Try parsing stdout
        try:
            txt = log.read_text()
            m = re.search(
                r"Average Precision at IOU 0\.3 is ([0-9.]+), .*?0\.5 is ([0-9.]+), .*?0\.7 is ([0-9.]+)",
                txt)
            if m:
                ap = {"ap30": round(float(m.group(1)), 4),
                      "ap50": round(float(m.group(2)), 4),
                      "ap70": round(float(m.group(3)), 4)}
                print(f"[{tag}] AP from stdout: AP30={ap['ap30']} AP50={ap['ap50']} AP70={ap['ap70']}")
        except Exception as e:
            print(f"[{tag}] stdout parse failed: {e}")

    return ap


def run_one(tag: str, nf: list, wpg: int, gpu: int, results: dict):
    """Full pipeline: validate → prune → flatten → patch → finetune → eval → record."""
    print(f"\n{'='*60}")
    print(f"[{tag}] START  nf={nf} wpg={wpg} gpu={gpu}")
    print(f"{'='*60}")

    # 0. Validate feasibility
    if not validate_config(nf, wpg, GROUPS):
        msg = f"INFEASIBLE: wpg={wpg} groups={GROUPS} nf={nf}"
        print(f"[{tag}] {msg}")
        results[tag] = {"num_filters": nf, "wpg": wpg, "groups": GROUPS,
                        "ap70": None, "ap50": None, "status": f"infeasible:{msg}"}
        return

    # 1. Prune
    out_dir = prune(tag, nf, wpg)
    if out_dir is None:
        results[tag] = {"num_filters": nf, "wpg": wpg, "groups": GROUPS,
                        "ap70": None, "ap50": None, "status": "prune_failed"}
        return

    # 2. Flatten init ckpt
    flatten_ckpt(out_dir, tag)

    # 3. Patch config epoches
    patch_config(out_dir, tag)

    # 4. Finetune
    rc = finetune(out_dir, tag, gpu)
    if rc != 0:
        print(f"[{tag}] WARNING: finetune rc={rc} (may have partial ckpts)")

    # 5. Find bestval ckpt
    best = find_best_ckpt(out_dir)
    bestval_name = best.name if best else None
    print(f"[{tag}] bestval ckpt: {bestval_name}")

    # 6. Eval AP
    ap = eval_ap(tag, out_dir, gpu)

    # 7. Record
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

    # Save interim results
    RESULT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[{tag}] DONE → saved to {RESULT_JSON}")


def run_gpu_serial(gpu_configs: list, results: dict):
    """Run a list of (tag, nf, wpg, gpu) serially on a single GPU."""
    for tag, nf, wpg, gpu in gpu_configs:
        run_one(tag, nf, wpg, gpu, results)


def main():
    print("=== L4 Finetune Launcher: DELIVERABLE A (cliff) + B2 (wg_pair) ===")
    print(f"REPO: {REPO}")
    print(f"EPOCHES: {EPOCHES} (init@23 → epoch {EPOCHES})")

    # Load existing results if any
    results = {}
    if RESULT_JSON.exists():
        results = json.loads(RESULT_JSON.read_text())
        print(f"Loaded existing results: {list(results.keys())}")

    # Group by GPU
    gpu_groups: dict[int, list] = {}
    for tag, nf, wpg, gpu in CONFIGS:
        gpu_groups.setdefault(gpu, []).append((tag, nf, wpg, gpu))

    # GPU 0: cliff_a → cliff_b (serial)
    # GPU 1: pair4 → pair5 → pair6 (serial)
    # Run in main thread: GPU 1 first in background, then GPU 0 in foreground
    # Since we're single-process, run GPU 0 then GPU 1 or use subprocess for parallel.
    # Actually: launch GPU 1 as subprocess, run GPU 0 here.

    gpu1_configs = gpu_groups.get(1, [])
    gpu0_configs = gpu_groups.get(0, [])

    if gpu1_configs:
        print(f"\nLaunching GPU 1 chain ({[c[0] for c in gpu1_configs]}) as subprocess...")
        # Write a sub-launcher script for GPU1
        sub_script = REPO / "scripts/phase2/l4_gpu1_chain.py"
        sub_script.write_text(f"""
import sys
sys.path.insert(0, '/home/jichengzhi/V2X/scripts/phase2')
from l4_finetune_launcher import run_gpu_serial, results, gpu_groups, RESULT_JSON
import json
from pathlib import Path
results = {{}}
r = Path('{RESULT_JSON}')
if r.exists():
    results = json.loads(r.read_text())
run_gpu_serial({repr(gpu1_configs)}, results)
""")
        env_gpu1 = {**os.environ, "PYTHONPATH": str(HEAL)}
        log_gpu1 = REPO / "results/l4_gpu1_chain.log"
        with open(log_gpu1, "w") as lf:
            proc_gpu1 = subprocess.Popen(
                [PY, str(sub_script)],
                cwd=REPO,
                env=env_gpu1,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"GPU 1 chain launched: pid={proc_gpu1.pid} log={log_gpu1}")

    # Run GPU 0 serially in this process
    if gpu0_configs:
        print(f"\nRunning GPU 0 chain ({[c[0] for c in gpu0_configs]}) in main process...")
        run_gpu_serial(gpu0_configs, results)

    # Wait for GPU 1 subprocess
    if gpu1_configs:
        print(f"\nWaiting for GPU 1 chain (pid={proc_gpu1.pid})...")
        proc_gpu1.wait()
        print(f"GPU 1 chain finished rc={proc_gpu1.returncode}")
        # Merge GPU1 results
        if RESULT_JSON.exists():
            merged = json.loads(RESULT_JSON.read_text())
            results.update(merged)
            RESULT_JSON.write_text(json.dumps(results, indent=2))

    print(f"\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"Written to {RESULT_JSON}")


if __name__ == "__main__":
    main()
