"""Plan 4 Pre-Phase 0 — σ_FT8 实测 (5-seed noise floor at FT=8 锁定 anchor).

跟 a11_noise_seed_dispatcher_ft6.py 的差别:
  - FT_EPOCHS = 8 (vs 6)
  - 起点不是 raw, 而是 FT=6 seed dir 的 epoch25 — 续训 2 epoch → epoch27
  - 输出 dir 名 noise_ft8_p97_seed{N}

阈值 (plan v4 §4.6):
  σ_FT8 < 0.02 → threshold_signal = 0.06 (Phase A 阈值)
  0.02 ≤ σ_FT8 < 0.04 → threshold_signal = 0.10 (放宽)
  σ_FT8 ≥ 0.04 → abort

Output:
  /tmp/plan4_phase0/noise.json  (5 AP + σ + verdict)
  paper_learning/2. AAAI最终故事/data/plan4_state.json  (更新 phase_0_done + sigma + threshold)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"

BASELINE_EPOCH = 19  # g8 baseline epoch
FT_EPOCHS = 8        # 锁定 FT=8 (plan v4 §0.1)
TARGET_EPOCH = BASELINE_EPOCH + FT_EPOCHS  # = 27
SRC_EPOCH = BASELINE_EPOCH + 6             # = 25 (FT=6 起点)
TRIPLET = "T_g8_p97"

OUT = Path("/tmp/plan4_phase0")
OUT.mkdir(parents=True, exist_ok=True)
STATE_FILE = REPO / "paper_learning/2. AAAI最终故事/data/plan4_state.json"


def patch_config(cfg_path: Path):
    import yaml
    with open(cfg_path) as f:
        hypes = yaml.load(f, Loader=yaml.UnsafeLoader)
    hypes["train_params"]["save_freq"] = 1
    hypes["train_params"]["eval_freq"] = 1
    hypes["train_params"]["epoches"] = TARGET_EPOCH
    with open(cfg_path, "w") as f:
        yaml.dump(hypes, f, default_flow_style=False, allow_unicode=True)


def ensure_seed_dir(seed: int) -> Path:
    """Copy FT=6 seed dir → FT=8 seed dir, patch config to epoches=27.

    HEAL train_ddp.py auto-resumes from largest existing net_epoch{N}.pth in
    model_dir, so all FT=6 epoch20-25 ckpts must come along.
    """
    src = REPO / f"models/dataset_a_cache_g8/noise_ft6_p97_seed{seed}"
    dst = REPO / f"models/dataset_a_cache_g8/noise_ft8_p97_seed{seed}"

    target_ckpt = dst / f"net_epoch{TARGET_EPOCH}.pth"
    if target_ckpt.exists() and (dst / "config.yaml").exists():
        return dst  # cached

    dst.mkdir(parents=True, exist_ok=True)
    src_ckpt = src / f"net_epoch{SRC_EPOCH}.pth"
    if not src_ckpt.exists():
        raise FileNotFoundError(f"FT=6 epoch{SRC_EPOCH} ckpt missing: {src_ckpt}")

    # Copy net_epoch20..25 + bestval + config.yaml
    for epoch in range(BASELINE_EPOCH + 1, SRC_EPOCH + 1):
        src_pth = src / f"net_epoch{epoch}.pth"
        if src_pth.exists():
            dst_pth = dst / src_pth.name
            if not dst_pth.exists():
                shutil.copy(src_pth, dst_pth)
    bestval = list(src.glob("net_epoch_bestval_at*.pth"))
    if bestval and not (dst / bestval[0].name).exists():
        shutil.copy(bestval[0], dst / bestval[0].name)
    shutil.copy(src / "config.yaml", dst / "config.yaml")
    patch_config(dst / "config.yaml")
    print(f"[seed{seed}] dir prepared, will resume from epoch{SRC_EPOCH}", flush=True)
    return dst


_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def finetune_one(spec):
    """spec = (seed, ft_dir)"""
    seed, ft_dir = spec
    gpu = _GPU
    cfg = ft_dir / "config.yaml"
    target_ckpt = ft_dir / f"net_epoch{TARGET_EPOCH}.pth"
    log = OUT / f"ft_seed{seed}.log"
    if target_ckpt.exists():
        print(f"[ft seed{seed}] target ckpt cached", flush=True)
        return (seed, ft_dir, 0, "cached")
    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29800 + 100 + seed}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(ft_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL),
           "PYTHONHASHSEED": str(1000 + seed * 17),
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    t0 = time.time()
    print(f"[ft seed{seed}] GPU {gpu} resume from epoch{SRC_EPOCH} → "
          f"epoch{TARGET_EPOCH}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=HEAL, env=env, stdout=f,
                           stderr=subprocess.STDOUT,
                           start_new_session=True, timeout=2 * 3600)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return (seed, ft_dir, elapsed, f"FAIL rc={r.returncode}")
    if not target_ckpt.exists():
        return (seed, ft_dir, elapsed, "FAIL no target ckpt")
    return (seed, ft_dir, elapsed, "OK")


def eval_one(spec):
    """eval AP_FP32 for seed's epoch{TARGET}.pth."""
    seed, ft_dir = spec
    gpu = _GPU
    tag = f"plan4_ph0_seed{seed}"
    ckpt = ft_dir / f"net_epoch{TARGET_EPOCH}.pth"
    cfg = ft_dir / "config.yaml"
    onnx = OUT / f"{tag}.onnx"
    engine = OUT / f"{tag}_fp32.engine"
    build_rep = OUT / f"{tag}_fp32.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (seed, d.get("ap50") or d.get("ap_50"), 0, "cached")
    if not ckpt.exists():
        return (seed, None, 0, "no ckpt")

    if not onnx.exists():
        cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_e2e.py"),
               "--ckpt", str(ckpt), "--hypes", str(cfg),
               "--out", str(onnx), "--max-voxels", "32000"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=300)
        if r.returncode != 0:
            (OUT / f"{tag}.onnx.err").write_text(r.stdout + r.stderr)
            return (seed, None, 0, "onnx fail")

    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx), "--precision", "fp32",
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=600)
        if r.returncode != 0:
            (OUT / f"{tag}.engine.err").write_text(r.stdout + r.stderr)
            return (seed, None, 0, "engine fail")

    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", str(ft_dir),
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", "1789", "--tag", tag,
           "--report", str(ap_rep)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=900)
    if r.returncode != 0:
        (OUT / f"{tag}.ap.err").write_text(r.stdout + r.stderr)
        return (seed, None, time.time()-t0, "AP fail")
    d = json.loads(ap_rep.read_text())
    ap50 = d.get("ap50") or d.get("ap_50")
    return (seed, ap50, time.time()-t0, "OK")


def update_state(sigma, threshold, mean_ap, aps, verdict):
    """Write plan4_state.json incrementally."""
    state = {}
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
    state["phase_0_done"] = True
    state["phase_0_progress"] = {"seeds_done": len(aps)}
    state["sigma_ft8"] = sigma
    state["threshold_signal"] = threshold
    state["phase_0_mean_ap"] = mean_ap
    state["phase_0_aps"] = aps
    state["phase_0_verdict"] = verdict
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--gpus", default="0,1,2,3,6",
                   help="comma-sep GPUs (1 per seed parallel; 5 needed)")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]

    # Step 1: prepare seed dirs (copy FT=6 → FT=8 + patch config)
    print(f"[Ph0] Step 1: prepare {len(seeds)} seed dirs from FT=6 ckpts")
    seed_specs = []
    for s in seeds:
        ft_dir = ensure_seed_dir(s)
        seed_specs.append((s, ft_dir))

    # Step 2: parallel finetune 2 epoch
    n_parallel = min(len(seeds), len(gpus))
    print(f"[Ph0] Step 2: parallel finetune {n_parallel} seed × 2 epoch")
    t0 = time.time()
    with Pool(processes=n_parallel, initializer=_init,
              initargs=(gpus,)) as pool:
        ft_results = list(pool.imap_unordered(finetune_one, seed_specs))
    print(f"[Ph0] finetune wall: {(time.time()-t0)/60:.1f} min")
    for seed, _, secs, status in sorted(ft_results):
        print(f"  seed{seed}: {status} ({secs/60:.1f} min)")

    # Step 3: parallel AP eval (n=1789)
    print(f"[Ph0] Step 3: parallel AP eval (n=1789)")
    t0 = time.time()
    with Pool(processes=n_parallel, initializer=_init,
              initargs=(gpus,)) as pool:
        ap_results = list(pool.imap_unordered(eval_one, seed_specs))
    print(f"[Ph0] eval wall: {(time.time()-t0)/60:.1f} min")

    aps = []
    per_seed = {}
    for seed, ap, secs, status in sorted(ap_results):
        print(f"  seed{seed}: AP={ap}  ({status}, {secs:.0f}s)")
        per_seed[seed] = {"ap": ap, "status": status}
        if ap is not None:
            aps.append(ap)

    if len(aps) < 4:
        print(f"[Ph0] FAIL: only {len(aps)}/5 successful, cannot compute σ")
        update_state(None, None, None, aps, "FAIL_INCOMPLETE")
        return

    mean_ap = sum(aps) / len(aps)
    var_ap = sum((a - mean_ap) ** 2 for a in aps) / (len(aps) - 1)
    sigma = math.sqrt(var_ap)

    # Threshold decision (plan v4 §4.6)
    if sigma < 0.02:
        threshold = 0.06
        verdict = "PASS_TIGHT"
    elif sigma < 0.04:
        threshold = 0.10
        verdict = "PASS_RELAXED"
    else:
        threshold = None
        verdict = "ABORT_FT8_NOISE_TOO_HIGH"

    out = {
        "seeds": seeds,
        "aps": aps,
        "n_aps": len(aps),
        "per_seed": per_seed,
        "mean_ap": mean_ap,
        "sigma_noise": sigma,
        "threshold_signal": threshold,
        "verdict": verdict,
        "phase_a_signal_threshold": threshold,
    }
    (OUT / "noise.json").write_text(json.dumps(out, indent=2))
    update_state(sigma, threshold, mean_ap, aps, verdict)

    print(f"\n[Ph0 RESULT]")
    print(f"  mean AP: {mean_ap:.4f}")
    print(f"  σ_noise: {sigma:.4f}")
    print(f"  threshold_signal: {threshold}")
    print(f"  verdict: {verdict}")
    print(f"  → {OUT/'noise.json'}")
    print(f"  → {STATE_FILE}")


if __name__ == "__main__":
    main()
