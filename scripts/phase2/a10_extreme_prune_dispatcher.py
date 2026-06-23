"""Phase 2 dispatcher — Extreme prune (groups=8) + 15 epoch finetune.

Two stages:
  Stage A: Structural prune baseline_g8 → 3 raw pruned ckpts
           T_g8_p87 (8,8,8), T_g8_p93 (8,4,4), T_g8_p97 (4,4,4)
  Stage B: DDP finetune each, save every epoch (epoch baseline+1 .. baseline+15)

Triplets are dispatched in parallel (1 GPU each).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
LOG = Path("/tmp/a10_phase2")
LOG.mkdir(exist_ok=True)

TRIPLETS = [
    # (tag, num_filters [list], gpu_default)
    ("T_g8_p87", [8, 8, 8], 5),
    ("T_g8_p93", [8, 4, 4], 7),
    ("T_g8_p97", [4, 4, 4], 3),
]


def find_baseline_bestval(baseline_dir: Path) -> tuple[Path, int]:
    """Find net_epoch_bestval_at*.pth and return (path, epoch)."""
    ckpts = list(baseline_dir.glob("net_epoch_bestval_at*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"no bestval ckpt in {baseline_dir}")
    # parse "_at{N}.pth"
    def epoch_of(p):
        s = p.stem  # net_epoch_bestval_at27
        return int(s.split("_at")[-1])
    ckpts.sort(key=epoch_of)
    best = ckpts[-1]
    return best, epoch_of(best)


def prune_one(baseline_dir: Path, triplet_tag: str, nf: list[int],
              out_root: Path) -> tuple[Path, Path]:
    """Run structural_prune_pyramid.py on baseline_g8 for given triplet."""
    out_dir = out_root / f"ft_{triplet_tag}_raw"
    if (out_dir / "net_epoch_bestval_at0.pth").exists() or \
       any(out_dir.glob("net_epoch_bestval_at*.pth")):
        print(f"[prune {triplet_tag}] already exists, skip")
        return out_dir, out_dir / "config.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)
    nf_str = ",".join(str(n) for n in nf)
    cmd = [PYTHON, str(REPO / "tools/structural_prune_pyramid.py"),
           "--orig-dir", str(baseline_dir),
           "--out-dir", str(out_dir),
           "--num-filters-new", nf_str,
           "--groups", "8", "--width-per-group", "16"]
    log = LOG / f"prune_{triplet_tag}.log"
    print(f"[prune {triplet_tag}] {nf_str} → {out_dir}")
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": str(HEAL)},
                           stdout=f, stderr=subprocess.STDOUT,
                           timeout=600)
    if r.returncode != 0:
        print(f"  FAIL — see {log}")
        sys.exit(1)
    return out_dir, out_dir / "config.yaml"


def patch_ft_config(cfg_path: Path, finetune_epochs: int, baseline_epoch: int):
    """Patch HEAL config: save_freq=1, eval_freq=1, epoches = baseline+ft.

    HEAL config has Python OrderedDict + numpy ndarray tags — use yaml.UnsafeLoader.
    """
    import yaml
    with open(cfg_path) as f:
        hypes = yaml.load(f, Loader=yaml.UnsafeLoader)
    target = baseline_epoch + finetune_epochs
    hypes["train_params"]["save_freq"] = 1
    hypes["train_params"]["eval_freq"] = 1
    hypes["train_params"]["epoches"] = target
    with open(cfg_path, "w") as f:
        yaml.dump(hypes, f, default_flow_style=False, allow_unicode=True)
    print(f"  config patched: epoches={target} save/eval=1")


def rename_bestval_to_continue(out_dir: Path, baseline_ckpt_src: Path,
                               baseline_epoch: int):
    """HEAL load_saved_model finds bestval ckpt + uses its epoch as resume.
    structural_prune_pyramid.py already saves as net_epoch_bestval_at23.pth
    (hardcoded). We rename to baseline_epoch so HEAL resumes correctly.
    """
    fixed = out_dir / "net_epoch_bestval_at23.pth"
    target = out_dir / f"net_epoch_bestval_at{baseline_epoch}.pth"
    if fixed.exists() and not target.exists():
        fixed.rename(target)
        print(f"  rename bestval ckpt: at23 → at{baseline_epoch}")
    elif target.exists():
        print(f"  bestval ckpt already at{baseline_epoch}")
    else:
        raise FileNotFoundError(f"neither at23 nor at{baseline_epoch} exists in {out_dir}")


def finetune_one(spec):
    """spec = (tag, out_dir, gpu, baseline_epoch, finetune_epochs)"""
    tag, out_dir, gpu, baseline_ep, ft_n = spec
    cfg = out_dir / "config.yaml"
    log = LOG / f"finetune_{tag}.log"
    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29700 + gpu}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(out_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    t0 = time.time()
    print(f"[ft {tag}] GPU {gpu} start, target epoch {baseline_ep + ft_n}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=HEAL, env=env, stdout=f,
                           stderr=subprocess.STDOUT, start_new_session=True,
                           timeout=4 * 3600)
    elapsed = time.time() - t0
    return (tag, r.returncode, elapsed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True,
                   help="path to baseline_g8 ckpt dir (with bestval + config)")
    p.add_argument("--out-root", default=str(REPO / "models/dataset_a_cache_g8"),
                   help="where to put pruned + finetuned ckpts")
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--gpus", default="3,5,7",
                   help="comma-sep GPUs for the 3 triplet finetunes")
    args = p.parse_args()

    baseline_dir = Path(args.baseline_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) >= len(TRIPLETS), f"need {len(TRIPLETS)} GPUs, got {len(gpus)}"

    print(f"[1/3] Find baseline_g8 bestval ckpt in {baseline_dir}")
    bestval_ckpt, baseline_ep = find_baseline_bestval(baseline_dir)
    print(f"  bestval: {bestval_ckpt.name}  epoch={baseline_ep}")

    print(f"\n[2/3] Prune 3 triplets (sequential, ~1 min each)")
    triplet_dirs = {}
    for (tag, nf, _gpu_default), gpu in zip(TRIPLETS, gpus):
        out_dir, _cfg = prune_one(baseline_dir, tag, nf, out_root)
        rename_bestval_to_continue(out_dir, bestval_ckpt, baseline_ep)
        patch_ft_config(out_dir / "config.yaml", args.finetune_epochs, baseline_ep)
        triplet_dirs[tag] = (out_dir, gpu)

    print(f"\n[3/3] Parallel finetune ({args.finetune_epochs} epoch each)")
    specs = [(tag, d, g, baseline_ep, args.finetune_epochs)
             for tag, (d, g) in triplet_dirs.items()]
    t0 = time.time()
    with Pool(processes=len(specs)) as pool:
        for tag, rc, secs in pool.imap_unordered(finetune_one, specs):
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"[done] {tag} {status} ({secs/60:.1f} min)", flush=True)
    print(f"\nfinetune wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
