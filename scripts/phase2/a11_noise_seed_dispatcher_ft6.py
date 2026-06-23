"""Phase 1.3 (plan v3) — 5-seed noise floor estimation.

Run 5 independent FT=4 trainings of T_g8_p97 (raw → epoch 23) with different
PYTHONHASHSEED to inject DataLoader shuffle variance. Measure AP_FP32 each.
Compute σ_noise = std(AP across 5 seeds).

Success criteria per plan v3 §1.3:
  σ_noise ≤ 0.02 → R²_ceiling ≥ 0.96 (excellent)
  σ_noise ≤ 0.03 → R²_ceiling ≥ 0.91 (healthy)
  σ_noise ≤ 0.05 → R²_ceiling ≥ 0.75 (marginal)
  σ_noise >  0.05 → R²_ceiling < 0.75 (FAIL, need broader noise estimation)

Output: /tmp/a11_noise/noise.json (5 AP + σ + R²_ceiling) + 5 ft dirs.
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
CALIB_DIR = REPO / "calibration/pyramid_dair_e2e_32k"
BASELINE_DIR = Path("/home/jichengzhi/heal_research/HEAL/opencood/logs/"
                    "Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45")
BASELINE_EPOCH = 19
TRIPLET = "T_g8_p97"
NUM_FILTERS = [4, 4, 4]
FT_EPOCHS = 6
TARGET_EPOCH = BASELINE_EPOCH + FT_EPOCHS

OUT = Path("/tmp/a11_noise_ft6")
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


def rename_bestval(out_dir: Path):
    """structural_prune saves as net_epoch_bestval_at{src_epoch}.pth. Already
    matches baseline_epoch=19 for our case, so usually no-op."""
    target = out_dir / f"net_epoch_bestval_at{BASELINE_EPOCH}.pth"
    if target.exists():
        return
    existing = list(out_dir.glob("net_epoch_bestval_at*.pth"))
    if existing:
        existing[0].rename(target)
        print(f"  rename bestval: {existing[0].name} → {target.name}")


def ensure_raw_seed_dir(seed: int, raw_template: Path) -> Path:
    """Create models/dataset_a_cache_g8/noise_ft6_p97_seed{N}/ from raw template."""
    seed_dir = REPO / f"models/dataset_a_cache_g8/noise_ft6_p97_seed{seed}"
    if (seed_dir / f"net_epoch_bestval_at{BASELINE_EPOCH}.pth").exists() \
       and (seed_dir / "config.yaml").exists():
        return seed_dir
    seed_dir.mkdir(parents=True, exist_ok=True)
    # Copy raw template's bestval ckpt + config
    raw_ckpt_glob = list(raw_template.glob("net_epoch_bestval_at*.pth"))
    if not raw_ckpt_glob:
        raise FileNotFoundError(f"no raw bestval in {raw_template}")
    shutil.copy(raw_ckpt_glob[0],
                seed_dir / raw_ckpt_glob[0].name)
    shutil.copy(raw_template / "config.yaml", seed_dir / "config.yaml")
    rename_bestval(seed_dir)
    patch_config(seed_dir / "config.yaml")
    return seed_dir


def ensure_raw_template(gpu: int) -> Path:
    """Make sure raw_T_g8_p97 exists (regen via structural_prune if absent)."""
    raw = REPO / "models/dataset_a_cache_g8/raw_T_g8_p97"
    bestval_glob = list(raw.glob("net_epoch_bestval_at*.pth"))
    if bestval_glob and (raw / "config.yaml").exists():
        return raw
    raw.mkdir(parents=True, exist_ok=True)
    nf_str = ",".join(str(n) for n in NUM_FILTERS)
    cmd = [PYTHON, str(REPO / "tools/structural_prune_pyramid.py"),
           "--orig-dir", str(BASELINE_DIR),
           "--out-dir", str(raw),
           "--num-filters-new", nf_str,
           "--groups", "8", "--width-per-group", "16"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    log = OUT / "prune_raw.log"
    print(f"[raw] regen raw_T_g8_p97 → {raw}")
    with open(log, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"raw prune failed, see {log}")
    rename_bestval(raw)
    return raw


_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def finetune_one(spec):
    """spec = (seed, ft_dir, gpu)"""
    seed, ft_dir = spec
    gpu = _GPU
    cfg = ft_dir / "config.yaml"
    target_ckpt = ft_dir / f"net_epoch{TARGET_EPOCH}.pth"
    log = OUT / f"ft_seed{seed}.log"
    if target_ckpt.exists():
        print(f"[ft seed{seed}] cached target ckpt", flush=True)
        return (seed, ft_dir, 0, "cached")
    cmd = [PYTHON, "-m", "torch.distributed.launch",
           "--nproc_per_node=1", "--use_env",
           f"--master_port={29700 + 100 + seed}",
           str(HEAL / "opencood/tools/train_ddp.py"),
           "--hypes_yaml", str(cfg),
           "--model_dir", str(ft_dir),
           "--half"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL),
           "PYTHONHASHSEED": str(1000 + seed * 17),  # variance injection
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    t0 = time.time()
    print(f"[ft seed{seed}] GPU {gpu} start → epoch {TARGET_EPOCH}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(cmd, cwd=HEAL, env=env, stdout=f,
                           stderr=subprocess.STDOUT,
                           start_new_session=True, timeout=4 * 3600)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return (seed, ft_dir, elapsed, f"FAIL rc={r.returncode}")
    if not target_ckpt.exists():
        return (seed, ft_dir, elapsed, f"FAIL no target ckpt")
    return (seed, ft_dir, elapsed, "OK")


def eval_one(spec):
    """Eval AP_FP32 for seed's epoch23 ckpt. Build engine + run AP."""
    seed, ft_dir = spec
    gpu = _GPU
    tag = f"noise_seed{seed}"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--gpus", default="0,2,3,6,7",
                   help="comma-sep GPUs (1 per seed parallel)")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]

    # Step 1: ensure raw_T_g8_p97 (single gpu)
    print("[1.3] Step 1: ensure raw_T_g8_p97 template")
    raw_template = ensure_raw_template(gpu=gpus[0])

    # Step 2: copy raw to noise_ft6_p97_seed{N} per seed
    print(f"[1.3] Step 2: prepare {len(seeds)} seed dirs")
    seed_specs = []
    for s in seeds:
        ft_dir = ensure_raw_seed_dir(s, raw_template)
        seed_specs.append((s, ft_dir))

    # Step 3: parallel finetune
    n_parallel = min(len(seeds), len(gpus))
    print(f"[1.3] Step 3: parallel finetune ({n_parallel} concurrent)")
    t0 = time.time()
    with Pool(processes=n_parallel, initializer=_init,
              initargs=(gpus,)) as pool:
        ft_results = list(pool.imap_unordered(finetune_one, seed_specs))
    print(f"[1.3] finetune wall: {(time.time()-t0)/60:.1f} min")
    for seed, _, secs, status in sorted(ft_results):
        print(f"  seed{seed}: {status} ({secs/60:.1f} min)")

    # Step 4: parallel eval
    print(f"[1.3] Step 4: parallel AP eval (n=1789)")
    t0 = time.time()
    with Pool(processes=n_parallel, initializer=_init,
              initargs=(gpus,)) as pool:
        ap_results = list(pool.imap_unordered(eval_one, seed_specs))
    print(f"[1.3] eval wall: {(time.time()-t0)/60:.1f} min")

    aps = []
    for seed, ap, secs, status in sorted(ap_results):
        print(f"  seed{seed}: AP={ap}  ({status}, {secs:.0f}s)")
        if ap is not None:
            aps.append(ap)

    if len(aps) < 2:
        print("[1.3] FAIL: <2 successful AP evals, cannot compute σ_noise")
        return

    mean_ap = sum(aps) / len(aps)
    var_ap = sum((a - mean_ap) ** 2 for a in aps) / (len(aps) - 1)
    sigma = math.sqrt(var_ap)

    # Per plan v3: assume σ_total ≈ 0.1 (rough estimate of full dataset AP std)
    sigma_total_est = 0.10
    r2_ceiling = 1.0 - (sigma ** 2) / (sigma_total_est ** 2) if sigma < sigma_total_est else 0.0

    out = {
        "seeds": seeds,
        "aps": aps,
        "n_aps": len(aps),
        "mean_ap": mean_ap,
        "sigma_noise": sigma,
        "sigma_total_estimate": sigma_total_est,
        "r2_ceiling": r2_ceiling,
        "verdict": "PASS" if sigma <= 0.03 else ("MARGINAL" if sigma <= 0.05 else "FAIL"),
    }
    (OUT / "noise.json").write_text(json.dumps(out, indent=2))
    print(f"\n[1.3 RESULT]")
    print(f"  mean AP: {mean_ap:.4f}")
    print(f"  σ_noise: {sigma:.4f}")
    print(f"  R²_ceiling (assuming σ_total={sigma_total_est}): {r2_ceiling:.3f}")
    print(f"  verdict: {out['verdict']}")
    print(f"  → /tmp/a11_noise/noise.json")


if __name__ == "__main__":
    main()
