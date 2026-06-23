"""Plan 4 Phase A pre-work — 补训 T2_p25 / T4_p50 / T6_p75 至 FT=8 (epoch 31).

HEAL 官方 pruned ckpt 只到 epoch 25 (= FT 2), 续训 6 epoch 到 epoch 31 (= FT 8).
3 GPU 并行, 期望 wall ~30-40 min.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"

TARGET_EPOCH = 31  # = g32 baseline 23 + FT 8

TRIPLETS = [
    ("T2_p25",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10"),
    ("T4_p50",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10"),
    ("T6_p75",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10"),
]

OUT = Path("/tmp/plan4_phaseA_finetune_t246")
OUT.mkdir(parents=True, exist_ok=True)


def patch_config(cfg_path: Path):
    import yaml
    with open(cfg_path) as f:
        hypes = yaml.load(f, Loader=yaml.UnsafeLoader)
    hypes["train_params"]["save_freq"] = 1
    hypes["train_params"]["eval_freq"] = 1
    hypes["train_params"]["epoches"] = TARGET_EPOCH
    with open(cfg_path, "w") as f:
        yaml.dump(hypes, f, default_flow_style=False, allow_unicode=True)


_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def finetune_one(spec):
    name, src_dir = spec
    gpu = _GPU
    src = Path(src_dir)
    target_ckpt = src / f"net_epoch{TARGET_EPOCH}.pth"
    log = OUT / f"ft_{name}.log"

    if target_ckpt.exists():
        return (name, 0, "cached")

    # Patch config to epoches=31 (HEAL will resume from epoch 25 → 31)
    cfg = src / "config.yaml"
    patch_config(cfg)

    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29900 + hash(name) % 100}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(src),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL),
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    t0 = time.time()
    print(f"[ft {name}] GPU {gpu} resume from epoch 25 → {TARGET_EPOCH}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=HEAL, env=env, stdout=f,
                           stderr=subprocess.STDOUT,
                           start_new_session=True, timeout=2 * 3600)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return (name, elapsed, f"FAIL rc={r.returncode}")
    if not target_ckpt.exists():
        return (name, elapsed, "FAIL no target ckpt")
    return (name, elapsed, "OK")


def main():
    gpus = [4, 5, 7]  # 3 GPU 并行 (避开 GPU 0-3,6 给 Phase A AP eval)
    # Actually GPU 5 is busy, use 4, 7, and try 6 if free
    import subprocess as sp
    util_check = sp.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader"],
                       capture_output=True, text=True)
    free_gpus = []
    for line in util_check.stdout.strip().split("\n"):
        idx, util = line.split(",")
        if int(util.strip().rstrip(" %")) < 50:
            free_gpus.append(int(idx.strip()))
    gpus = free_gpus[:3]
    print(f"[t246] using GPUs: {gpus}")

    specs = [(name, src) for name, src in TRIPLETS]
    n_par = min(len(specs), len(gpus))
    with Pool(processes=n_par, initializer=_init,
              initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(finetune_one, specs))

    for name, secs, status in sorted(results):
        print(f"  {name}: {status} ({secs/60:.1f} min)")


if __name__ == "__main__":
    main()
