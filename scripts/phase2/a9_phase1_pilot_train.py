"""Phase 1 Pilot — Finetune T11/T17/T22 raw → +15 epoch, save every epoch.

3 个 triplet 并行训 (T11 GPU 5, T17 GPU 3, T22 GPU 7).
HEAL save_freq 已改为 1 epoch, train 38 epoch (baseline 23 + 15 FT).

输出 ckpts: ft_{sig}_raw/net_epoch24.pth ... net_epoch38.pth
"""
from __future__ import annotations
import os, subprocess, sys, time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
LOG = Path("/tmp/a9_pilot")
LOG.mkdir(exist_ok=True)

JOBS = [
    # (tag, ft_dir, gpu)
    ("T11_p14", REPO / "models/dataset_a_cache/ft_064_064_256_raw", 5),
    ("T17_p57", REPO / "models/dataset_a_cache/ft_032_032_128_raw", 3),
    ("T22_p89", REPO / "models/dataset_a_cache/ft_016_016_016_raw", 7),
]


def train_one(spec):
    tag, ft_dir, gpu = spec
    cfg = ft_dir / "config.yaml"
    log = LOG / f"train_{tag}.log"
    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29600 + gpu}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(ft_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    t0 = time.time()
    print(f"[{tag}] GPU {gpu} start", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=HEAL, env=env, stdout=f,
                           stderr=subprocess.STDOUT, start_new_session=True,
                           timeout=4 * 3600)
    elapsed = time.time() - t0
    return (tag, r.returncode, elapsed)


def main():
    print(f"[a9 phase 1] {len(JOBS)} finetunes, ~25 min each")
    t0 = time.time()
    with Pool(processes=len(JOBS)) as pool:
        for tag, rc, secs in pool.imap_unordered(train_one, JOBS):
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(f"[done] {tag} {status} ({secs/60:.1f} min)", flush=True)
    print(f"\nwall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
