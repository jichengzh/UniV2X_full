"""Stage B — 20 strategic random_bench triplets finetune + AP eval, 6-GPU parallel.

For each triplet:
    1. Setup epoches=24 config (1 epoch finetune from epoch 23)
    2. HEAL train.py --model_dir prune_{sig} → finetuned ckpt at epoch 24
    3. Export ONNX from finetuned ckpt
    4. Build FP16 + INT8 engines
    5. AP eval each engine

Parallelism: 6 GPUs (1-6) run different triplets concurrently.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, time, traceback, multiprocessing as mp
from pathlib import Path
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
RANDOM_CACHE = REPO_ROOT / "models/p0_random_cache"
B_CACHE = REPO_ROOT / "models/stage_b_cache"
B_CACHE.mkdir(parents=True, exist_ok=True)
B_OUT = REPO_ROOT / "results/stage_b"
B_OUT.mkdir(parents=True, exist_ok=True)

CALIB_SPATIAL = REPO_ROOT / "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = REPO_ROOT / "calibration/pyramid_dair_collab_tego.npy"

# Selected 20 triplets (from /tmp/stage_b_triplets.csv)
TRIPLETS = [
    "064_128_256","032_072_128","024_072_136","032_040_136","032_040_128",
    "032_032_128","024_040_128","032_040_064","016_064_064","024_040_072",
    "032_032_032","024_024_072","024_040_032","016_032_064","016_024_072",
    "016_016_072","016_016_064","024_024_016","016_016_040","016_016_016",
]


def setup_finetune_config(sig: str) -> Path:
    """Copy prune_{sig} into stage_b_cache + set epoches=24 (1 finetune epoch)."""
    src = RANDOM_CACHE / f"prune_{sig}"
    dst = B_CACHE / f"ft_{sig}"
    if not (src / "net_epoch_bestval_at23.pth").exists():
        return None
    dst.mkdir(parents=True, exist_ok=True)
    # Copy ckpt
    if not (dst / "net_epoch_bestval_at23.pth").exists():
        shutil.copy(src / "net_epoch_bestval_at23.pth", dst / "net_epoch_bestval_at23.pth")
    # Copy + patch config: any "epoches: N" → "epoches: 24"
    # (original prune_*/config.yaml uses epoches:40 — replace blanket via regex)
    if not (dst / "config.yaml").exists():
        with open(src / "config.yaml") as f:
            cfg = f.read()
        cfg = re.sub(r"epoches:\s*\d+", "epoches: 24", cfg)
        with open(dst / "config.yaml", "w") as f:
            f.write(cfg)
    return dst


def _has_finetuned_ckpt(ft_dir: Path) -> bool:
    """True iff a finetuned ckpt past epoch 23 exists (bestval@N>23 or epoch>=24)."""
    for p in ft_dir.glob("net_epoch_bestval_at*.pth"):
        m = re.search(r"at(\d+)", p.name)
        if m and int(m.group(1)) > 23:
            return True
    for p in ft_dir.glob("net_epoch*.pth"):
        if "bestval" in p.name: continue
        m = re.search(r"net_epoch(\d+)", p.name)
        if m and int(m.group(1)) >= 24:
            return True
    return False


def finetune(sig: str, gpu: int) -> Path | None:
    """Run HEAL train.py finetune (~1 epoch wall, bounded by timeout). Returns ft_dir."""
    ft_dir = setup_finetune_config(sig)
    if ft_dir is None:
        return None
    # Pickup: if any post-baseline ckpt exists, skip retraining
    if _has_finetuned_ckpt(ft_dir):
        print(f"[ft GPU{gpu}] {sig} pickup — finetuned ckpt already present")
        return ft_dir

    cmd = [PYTHON, str(HEAL_ROOT / "opencood/tools/train.py"),
           "--hypes_yaml", str(ft_dir / "config.yaml"),
           "--model_dir", str(ft_dir)]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL_ROOT),
           "PATH": os.environ.get("PATH", "")}
    print(f"[ft GPU{gpu}] {sig} finetuning...")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env,
                           capture_output=True, text=True, timeout=5400)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[ft GPU{gpu}] {sig} timeout after {elapsed:.0f}s — checking for partial ckpt")
        if _has_finetuned_ckpt(ft_dir):
            print(f"[ft GPU{gpu}] {sig} partial ckpt present, continuing")
            return ft_dir
        return None
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"[ft GPU{gpu}] {sig} FAILED ({elapsed:.0f}s): {r.stderr[-300:]}")
        return None
    print(f"[ft GPU{gpu}] {sig} done ({elapsed:.0f}s)")
    return ft_dir


def export_onnx(sig: str, ft_dir: Path, gpu: int) -> Path | None:
    out_onnx = B_CACHE / f"{sig}_ft.onnx"
    if out_onnx.exists(): return out_onnx
    # Find latest epoch ckpt
    ckpts = sorted(ft_dir.glob("net_epoch*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        print(f"[export GPU{gpu}] {sig}: no ckpt")
        return None
    ckpt = ckpts[-1]
    cmd = [PYTHON, str(REPO_ROOT / "tools/export_onnx_pyramid_collab.py"),
           "--ckpt", str(ckpt),
           "--hypes", str(ft_dir / "config.yaml"),
           "--out", str(out_onnx),
           "--feat-h", "128"]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": os.environ.get("PATH", "")}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0 or not out_onnx.exists():
        print(f"[export GPU{gpu}] {sig} FAILED: {r.stderr[-300:]}")
        return None
    return out_onnx


def build_engine(sig: str, prec: str, onnx_path: Path, gpu: int) -> Path | None:
    eng = B_CACHE / f"{sig}_ft_{prec}.engine"
    if eng.exists(): return eng
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx_path),
           "--precision", prec,
           "--engine", str(eng),
           "--report", str(B_CACHE / f"{sig}_ft_{prec}_build.json"),
           "--input-shape", "2,64,128,256",
           "--extra-input-shape", "t_ego:2,2,3",
           "--n-warmup", "30", "--n-measure", "100"]
    if prec == "int8":
        cmd += ["--calib-multi", f"spatial_features:{CALIB_SPATIAL}",
                "--calib-multi", f"t_ego:{CALIB_TEGO}",
                "--calib-cache", str(B_CACHE / f"{sig}_ft_int8_calib.cache")]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": os.environ.get("PATH", "")}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=REPO_ROOT)
    if r.returncode != 0 or not eng.exists():
        print(f"[build GPU{gpu}] {sig} {prec} FAILED: {r.stderr[-200:]}")
        return None
    return eng


def ap_eval(sig: str, prec: str, engine: Path, ft_dir: Path, gpu: int) -> dict | None:
    tag = f"sB_{sig}_{prec}"
    report = B_OUT / f"{tag}.json"
    if report.exists(): return json.loads(report.read_text())
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
           "--engine-collab", str(engine),
           "--tag", tag,
           "--model-dir", str(ft_dir),
           "--n-samples", "1789", "--dataset", "dair", "--range", "102.4,51.2",
           "--collab-spatial-shape", "2,64,128,256",
           "--collab-tego-shape", "2,2,3",
           "--report", str(report)]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": os.environ.get("PATH", "")}
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env, capture_output=True, text=True, timeout=1500)
    elapsed = time.time() - t0
    if r.returncode != 0 or not report.exists():
        print(f"[AP GPU{gpu}] {tag} FAILED: {r.stderr[-200:]}")
        return None
    rep = json.loads(report.read_text())
    print(f"[AP GPU{gpu}] {tag} AP50={rep['ap50']:.4f} AP70={rep['ap70']:.4f} ({elapsed:.0f}s)")
    return rep


def process_one_anchor(args):
    """Per-triplet worker: finetune → export → build (FP16+INT8) → AP eval."""
    sig, gpu = args
    print(f"\n[GPU{gpu}] starting {sig}")
    t0 = time.time()

    ft_dir = finetune(sig, gpu)
    if ft_dir is None: return []

    onnx = export_onnx(sig, ft_dir, gpu)
    if onnx is None: return []

    results = []
    for prec in ("fp16", "int8"):
        eng = build_engine(sig, prec, onnx, gpu)
        if eng is None: continue
        rep = ap_eval(sig, prec, eng, ft_dir, gpu)
        if rep is None: continue
        parts = sig.split("_")
        results.append({
            "triplet_sig": sig, "precision": prec,
            "stage0_planes": int(parts[0]), "stage1_planes": int(parts[1]), "stage2_planes": int(parts[2]),
            "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
            "n_samples": rep["n_samples"], "n_trt_path": rep["n_trt_path"],
            "elapsed_secs": rep["elapsed_secs"],
        })

    print(f"[GPU{gpu}] {sig} all done ({(time.time()-t0)/60:.1f} min)")
    return results


def gpu_worker(gpu_anchors):
    """Process all anchors assigned to one GPU sequentially. Top-level for pickle.
    Isolates per-anchor failures so the whole pool doesn't die on one timeout.
    """
    gpu, anchors = gpu_anchors
    rows = []
    for sig in anchors:
        try:
            rows.extend(process_one_anchor((sig, gpu)))
        except Exception as e:
            print(f"[GPU{gpu}] {sig} EXCEPTION: {type(e).__name__}: {e}")
            traceback.print_exc()
    return rows


def main():
    t0 = time.time()
    gpu_ids = [1, 2, 3, 4, 5, 6]
    tasks = [(sig, gpu_ids[i % len(gpu_ids)]) for i, sig in enumerate(TRIPLETS)]
    by_gpu = {}
    for sig, gpu in tasks:
        by_gpu.setdefault(gpu, []).append(sig)
    print(f"Task distribution: {by_gpu}")

    with mp.Pool(processes=6) as pool:
        all_rows = pool.map(gpu_worker, list(by_gpu.items()))

    rows = [r for lst in all_rows for r in lst]
    df = pd.DataFrame(rows)
    out = REPO_ROOT / "data/stage_b_ap_real.parquet"
    df.to_parquet(out); df.to_csv(out.with_suffix(".csv"), index=False)
    elapsed = (time.time() - t0) / 60
    print(f"\n[done] {len(df)} rows in {elapsed:.1f} min -> {out}")
    print(df[["triplet_sig","precision","ap50","ap70"]].to_string(index=False))


if __name__ == "__main__":
    main()
