"""Track A v2 — train 13 NEW unseen Pyramid triplets (B 维度扩 8→21).

设计源: paper_learning/2. AAAI最终故事/data/triplet_design_v2.json
排除 T9 (64, 256, 128): stage1 grow 不可行 (L1 prune 不能 grow channel)

13 NEW triplets (满足 pow-2 in_per_group 约束):
    T10 (16,128,256)  10.7% prune
    T11 (64, 64,256)  14.3%
    T12 (32, 64,256)  21.4%
    T13 (32, 32,256)  28.6%
    T14 (16, 16,256)  35.7%
    T15 (16,128,128)  39.3%
    T16 (16, 64,128)  53.6%
    T17 (32, 32,128)  57.1%
    T18 (16, 16,128)  64.3%
    T19 (32, 32, 64)  71.4%
    T20 (16, 16, 64)  78.6%
    T21 (16, 32, 32)  82.1%
    T22 (16, 16, 16)  89.3%

Pipeline per triplet (从 dataset_a_prepare_ckpts.py 复用):
    1. structural_prune_pyramid → init ckpt
    2. HEAL train_ddp.py --half (AMP) 25 epoch finetune
    3. AP eval verify ≥ baseline - 5pp (P4 convergence gate)

GPU 分配: 6 GPU pool, 13 triplets 分批 batch_size=6.
预期 wall: 3 batch × 3h ≈ 9h
"""
from __future__ import annotations
import json, os, re, subprocess, time, multiprocessing as mp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE_CKPT_DIR = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"

A_CACHE = REPO_ROOT / "models/dataset_a_cache"
A_CACHE.mkdir(parents=True, exist_ok=True)
A_LOG = REPO_ROOT / "logs/dataset_a_v2"
A_LOG.mkdir(parents=True, exist_ok=True)

# 13 NEW triplets — 排除 T9 (stage1 grow)
NEW_TRIPLETS = [
    (16, 128, 256, "T10_p11"),
    (64,  64, 256, "T11_p14"),
    (32,  64, 256, "T12_p21"),
    (32,  32, 256, "T13_p29"),
    (16,  16, 256, "T14_p36"),
    (16, 128, 128, "T15_p39"),
    (16,  64, 128, "T16_p54"),
    (32,  32, 128, "T17_p57"),
    (16,  16, 128, "T18_p64"),
    (32,  32,  64, "T19_p71"),
    (16,  16,  64, "T20_p79"),
    (16,  32,  32, "T21_p82"),
    (16,  16,  16, "T22_p89"),
]

N_GPUS = 6
BATCH_SIZE = 6


def signature(s0, s1, s2) -> str:
    return f"{s0:03d}_{s1:03d}_{s2:03d}"


def structural_prune(s0, s1, s2, gpu) -> Path | None:
    sig = signature(s0, s1, s2)
    out_dir = A_CACHE / f"ft_{sig}"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_pths = list(out_dir.glob("*.pth"))
    if existing_pths:
        print(f"[prune GPU{gpu}] {sig} skip — {len(existing_pths)} .pth exist")
        return out_dir
    cmd = [PYTHON, "tools/structural_prune_pyramid.py",
           "--orig-dir", BASELINE_CKPT_DIR,
           "--out-dir", str(out_dir),
           "--num-filters-new", f"{s0},{s1},{s2}"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[prune GPU{gpu}] {sig} structural pruning...")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"[prune GPU{gpu}] {sig} FAILED: {r.stderr[-300:]}")
        return None
    print(f"[prune GPU{gpu}] {sig} prune done ({time.time()-t0:.0f}s)")
    return out_dir


def patch_config_epoches(ft_dir: Path) -> Path:
    cfg_path = ft_dir / "config.yaml"
    if not cfg_path.exists():
        raise RuntimeError(f"config.yaml missing in {ft_dir}")
    cfg = cfg_path.read_text()
    cfg_patched = re.sub(r"epoches:\s*\d+", "epoches: 48", cfg)
    if cfg_patched != cfg:
        cfg_path.write_text(cfg_patched)
    return cfg_path


def has_full_finetuned_ckpt(ft_dir: Path) -> bool:
    for p in ft_dir.glob("net_epoch_bestval_at*.pth"):
        m = re.search(r"at(\d+)", p.name)
        if m and int(m.group(1)) >= 40:
            return True
    return False


def has_converged_ckpt(ft_dir: Path) -> bool:
    for p in ft_dir.glob("net_epoch_bestval_at*.pth"):
        m = re.search(r"at(\d+)", p.name)
        if m and int(m.group(1)) > 23:
            return True
    return False


def train_one(args):
    s0, s1, s2, tag, gpu = args
    sig = signature(s0, s1, s2)
    print(f"\n{'='*72}\n[GPU{gpu}] {tag} ({sig}) starting\n{'='*72}")
    t_total = time.time()

    ft_dir = structural_prune(s0, s1, s2, gpu)
    if ft_dir is None:
        return {"sig": sig, "tag": tag, "status": "prune_failed"}
    cfg_path = patch_config_epoches(ft_dir)

    if has_full_finetuned_ckpt(ft_dir):
        print(f"[GPU{gpu}] {sig} fully-finetuned ckpt exists, skip")
        return {"sig": sig, "tag": tag, "status": "pickup",
                "elapsed_secs": time.time() - t_total}

    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29500 + gpu}",
           str(HEAL_ROOT / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg_path),
           "--model_dir", str(ft_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL_ROOT)}
    log_path = A_LOG / f"train_{sig}.log"
    print(f"[GPU{gpu}] {sig} train.py launched → {log_path}")
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env,
                               stdout=lf, stderr=subprocess.STDOUT,
                               text=True, timeout=12 * 3600,
                               start_new_session=True)
        except subprocess.TimeoutExpired:
            print(f"[GPU{gpu}] {sig} timeout > 12h")
            return {"sig": sig, "tag": tag,
                    "status": "timeout_with_partial" if has_converged_ckpt(ft_dir) else "timeout_no_ckpt",
                    "elapsed_secs": time.time() - t_total}

    elapsed = time.time() - t0
    if r.returncode != 0:
        tail = log_path.read_text()[-500:]
        print(f"[GPU{gpu}] {sig} train FAIL ({elapsed:.0f}s)")
        return {"sig": sig, "tag": tag, "status": "train_failed",
                "elapsed_secs": elapsed, "log_tail": tail}
    if not has_converged_ckpt(ft_dir):
        return {"sig": sig, "tag": tag, "status": "no_ckpt_after_train",
                "elapsed_secs": elapsed}
    print(f"[GPU{gpu}] {sig} TRAIN OK ({elapsed/3600:.2f}h)")
    return {"sig": sig, "tag": tag, "status": "converged",
            "elapsed_secs": elapsed, "ft_dir": str(ft_dir)}


def main():
    print("=" * 78)
    print(f"Track A v2: train {len(NEW_TRIPLETS)} NEW Pyramid triplets")
    print(f"  GPUs: {N_GPUS}, batch_size: {BATCH_SIZE}")
    print("=" * 78)

    t_start = time.time()
    all_results = []

    # 分批: 13 triplet / 6 GPU = 3 批 (6 + 6 + 1)
    for batch_i in range(0, len(NEW_TRIPLETS), BATCH_SIZE):
        batch = NEW_TRIPLETS[batch_i:batch_i + BATCH_SIZE]
        print(f"\n=== Batch {batch_i//BATCH_SIZE + 1}/"
              f"{(len(NEW_TRIPLETS)+BATCH_SIZE-1)//BATCH_SIZE}: {len(batch)} triplets ===")
        # GPU 分配: round-robin 0..N_GPUS-1
        args_batch = [(s0, s1, s2, tag, i % N_GPUS)
                       for i, (s0, s1, s2, tag) in enumerate(batch)]
        for s0, s1, s2, tag, gpu in args_batch:
            print(f"  GPU{gpu}: {tag} ({s0},{s1},{s2})")

        t_batch = time.time()
        with mp.Pool(processes=len(args_batch)) as pool:
            results = pool.map(train_one, args_batch)
        all_results.extend(results)
        print(f"\nBatch {batch_i//BATCH_SIZE + 1} done in {(time.time()-t_batch)/3600:.2f}h")
        for r in results:
            print(f"  {r['tag']:10s} {r['sig']:18s} {r['status']:18s} "
                  f"{r.get('elapsed_secs',0)/3600:.2f}h")

    print("\n" + "=" * 78)
    print(f"Track A v2 done in {(time.time()-t_start)/3600:.2f}h")
    print("=" * 78)
    summary = REPO_ROOT / "results/track_a_v2_summary.json"
    summary.parent.mkdir(exist_ok=True)
    summary.write_text(json.dumps({"triplets": all_results,
                                    "total_wall_secs": time.time() - t_start},
                                   indent=2, default=str))
    print(f"summary → {summary}")


if __name__ == "__main__":
    main()
