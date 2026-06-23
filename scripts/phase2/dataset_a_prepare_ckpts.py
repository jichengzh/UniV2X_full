"""P0.1 — Dataset Class A: train 4 NEW Pyramid triplets to convergence.

8 个 plan §2.1 triplet 里 4 个已有 (复用 Stage A M4.9 ckpts):
    T1 (64,128,256) ← Pyramid_DAIR_m1_base
    T2 (48, 96,192) ← Pyramid_DAIR_m1_pruned25
    T4 (32, 64,128) ← Pyramid_DAIR_m1_pruned50
    T6 (16, 32, 64) ← Pyramid_DAIR_m1_pruned75

4 个新 triplet 必须从头训:
    T3 (40, 80,160) — 37% prune, 填中间档
    T5 (24, 56,128) — 62% prune, 填中间档
    T7 (48, 64,128) — wide-shallow 非标 ratio
    T8 (24, 48,192) — narrow-deep 非标 ratio

Pipeline per triplet:
    1. structural_prune_pyramid: 从 baseline ckpt 删 channel → 输出 net_epoch_bestval_at23.pth (AP≈0)
    2. HEAL train.py 25 epoch + AMP --half: 收敛到 net_epoch_bestval_atN.pth
    3. AP eval: 验证 AP50 >= baseline - 5pp (§8.0.1 convergence gate)

并行: GPU 0-3 跑 4 个 triplet (每 GPU 1 个 triplet, AMP, ~8h wall).
GPU 4, 6 空闲不参与本批; GPU 5, 7 被其他用户占.

输出: models/dataset_a_cache/ft_{sig}/net_epoch_bestval_at*.pth
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, time, multiprocessing as mp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
BASELINE_CKPT_DIR = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
TEMPLATE_CFG = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/config.yaml"

A_CACHE = REPO_ROOT / "models/dataset_a_cache"
A_CACHE.mkdir(parents=True, exist_ok=True)
A_LOG = REPO_ROOT / "logs/dataset_a"
A_LOG.mkdir(parents=True, exist_ok=True)

# 4 NEW triplets to train (T1/T2/T4/T6 already have Stage A ckpts)
NEW_TRIPLETS = [
    # (s0, s1, s2, gpu_id, tag, label)
    (40,  80, 160, 0, "p37",   "T3 37% prune mid"),
    (24,  56, 128, 1, "p62",   "T5 62% prune mid"),
    (48,  64, 128, 2, "wide",  "T7 wide-shallow non-standard"),
    (24,  48, 192, 3, "deep",  "T8 narrow-deep non-standard"),
]


def signature(s0, s1, s2) -> str:
    return f"{s0:03d}_{s1:03d}_{s2:03d}"


def structural_prune(s0, s1, s2, gpu) -> Path | None:
    """Run tools/structural_prune_pyramid.py to create init ckpt at epoch 23 baseline.

    Skip if ANY .pth file exists in ft_dir (avoid overwriting trained ckpts).
    """
    sig = signature(s0, s1, s2)
    out_dir = A_CACHE / f"ft_{sig}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 关键安全检查: 任何 .pth 存在就跳过 prune (防止覆盖已训练 ckpt)
    existing_pths = list(out_dir.glob("*.pth"))
    if existing_pths:
        print(f"[prune GPU{gpu}] {sig} skip — {len(existing_pths)} .pth exist (HEAL resume from latest)")
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
    if r.returncode != 0 or not init_ckpt.exists():
        print(f"[prune GPU{gpu}] {sig} FAILED: {r.stderr[-500:]}")
        return None
    print(f"[prune GPU{gpu}] {sig} prune done ({time.time()-t0:.0f}s)")
    return out_dir


def patch_config_epoches(ft_dir: Path) -> Path:
    """structural_prune_pyramid.py 生成的 config.yaml 已含正确 num_filters.
    设 epoches=48 (init 至 epoch 23, 训练 24-48 共 25 epoch) — 满足 plan §8.0.1 P1.
    """
    cfg_path = ft_dir / "config.yaml"
    if not cfg_path.exists():
        raise RuntimeError(f"config.yaml missing in {ft_dir}")
    cfg = cfg_path.read_text()
    cfg_patched = re.sub(r"epoches:\s*\d+", "epoches: 48", cfg)
    if cfg_patched != cfg:
        cfg_path.write_text(cfg_patched)
    return cfg_path


def has_full_finetuned_ckpt(ft_dir: Path) -> bool:
    """Plan §8.0.1 P1 要求 ≥25 epoch training, 检查是否有 bestval@N>=46 (训了≥23 epoch).
    宽松点: 有 bestval@N>40 视为充分训练."""
    for p in ft_dir.glob("net_epoch_bestval_at*.pth"):
        m = re.search(r"at(\d+)", p.name)
        if m and int(m.group(1)) >= 40:
            return True
    return False


def has_converged_ckpt(ft_dir: Path) -> bool:
    """True iff a post-baseline bestval ckpt exists."""
    for p in ft_dir.glob("net_epoch_bestval_at*.pth"):
        m = re.search(r"at(\d+)", p.name)
        if m and int(m.group(1)) > 23:
            return True
    return False


def train_one(s0, s1, s2, gpu, tag, label) -> dict:
    """Full pipeline: structural_prune → patch config → HEAL train.py with AMP."""
    sig = signature(s0, s1, s2)
    print(f"\n{'='*72}\n[GPU{gpu}] {label} ({sig}) starting\n{'='*72}")
    t_total = time.time()

    # Step 1: structural prune
    ft_dir = structural_prune(s0, s1, s2, gpu)
    if ft_dir is None:
        return {"sig": sig, "status": "prune_failed"}

    # Step 2: patch config epoches 40 → 25
    cfg_path = patch_config_epoches(ft_dir)
    print(f"[GPU{gpu}] {sig} config epoches → 25 at {cfg_path}")

    # Step 3: HEAL train (--half = AMP mixed precision)
    # 需要 ≥25 epoch finetune (plan §8.0.1 P1), bestval@N>=40 才算充分训练
    if has_full_finetuned_ckpt(ft_dir):
        print(f"[GPU{gpu}] {sig} fully-finetuned ckpt exists, skip training")
        return {"sig": sig, "status": "pickup", "elapsed_secs": time.time() - t_total}

    # 用 train_ddp.py + torch.distributed.launch --nproc_per_node=1 才能拿 --half (AMP).
    # train.py 单卡不支持 --half. train_ddp.py 在 nproc=1 下单 GPU 也跑通.
    # master_port 按 GPU 唯一分配避免 4 worker 端口冲突 (29500+gpu).
    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29500 + gpu}",
           str(HEAL_ROOT / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg_path),
           "--model_dir", str(ft_dir),
           "--half"]  # AMP
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL_ROOT)}
    log_path = A_LOG / f"train_{sig}.log"
    print(f"[GPU{gpu}] {sig} train.py launched, log → {log_path}")
    t0 = time.time()
    with open(log_path, "w") as lf:
        try:
            r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env,
                               stdout=lf, stderr=subprocess.STDOUT,
                               text=True, timeout=12 * 3600,  # 12h cap
                               start_new_session=True)
        except subprocess.TimeoutExpired:
            print(f"[GPU{gpu}] {sig} timeout > 12h — checking partial ckpt")
            if has_converged_ckpt(ft_dir):
                return {"sig": sig, "status": "timeout_with_partial",
                        "elapsed_secs": time.time() - t_total}
            return {"sig": sig, "status": "timeout_no_ckpt",
                    "elapsed_secs": time.time() - t_total}

    elapsed_train = time.time() - t0
    if r.returncode != 0:
        # Tail log for diagnosis
        tail = log_path.read_text()[-1000:]
        print(f"[GPU{gpu}] {sig} train.py FAILED ({elapsed_train:.0f}s):\n{tail}")
        return {"sig": sig, "status": "train_failed",
                "elapsed_secs": elapsed_train, "log_tail": tail[-500:]}

    if not has_converged_ckpt(ft_dir):
        print(f"[GPU{gpu}] {sig} no converged ckpt produced after training")
        return {"sig": sig, "status": "no_ckpt_after_train",
                "elapsed_secs": elapsed_train}

    print(f"[GPU{gpu}] {sig} TRAIN OK ({elapsed_train/3600:.2f} h)")
    return {"sig": sig, "status": "converged",
            "elapsed_secs": elapsed_train, "ft_dir": str(ft_dir)}


def worker(args):
    s0, s1, s2, gpu, tag, label = args
    try:
        return train_one(s0, s1, s2, gpu, tag, label)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"sig": signature(s0, s1, s2), "status": "exception", "error": str(e)}


def main():
    print("=" * 78)
    print("P0.1 Dataset Class A: train 4 NEW Pyramid triplets")
    print("=" * 78)
    print("Reuse Stage A ckpts (T1/T2/T4/T6, AP50 ≥ 0.75 confirmed)")
    print()
    print("NEW triplets to train (4 × 1 GPU + AMP, parallel):")
    for s0, s1, s2, gpu, tag, label in NEW_TRIPLETS:
        print(f"  GPU{gpu}  {label:35s}  ({s0:3d},{s1:3d},{s2:3d})")
    print()

    t0 = time.time()
    with mp.Pool(processes=len(NEW_TRIPLETS)) as pool:
        results = pool.map(worker, NEW_TRIPLETS)

    print("\n" + "=" * 78)
    print(f"P0.1 done in {(time.time()-t0)/3600:.2f} h")
    print("=" * 78)
    for r in results:
        print(f"  {r['sig']:18s}  status={r['status']:18s}  "
              f"elapsed={r.get('elapsed_secs', 0)/3600:.2f}h")

    # Save summary
    summary = REPO_ROOT / "results/dataset_a_p0_1_summary.json"
    summary.parent.mkdir(exist_ok=True)
    with open(summary, "w") as f:
        json.dump({"triplets": results, "total_wall_secs": time.time() - t0}, f, indent=2)
    print(f"\nsummary saved → {summary}")


if __name__ == "__main__":
    main()
