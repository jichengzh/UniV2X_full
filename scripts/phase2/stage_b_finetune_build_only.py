"""Stage B Phase-1 — finetune + ONNX export + TRT engine build (no AP eval).

Companion to stage_b_ap_eval_seq.py.  Runs the GPU-bound prep stage in parallel
across GPUs 1-6, leaves AP eval to the sequential script on GPU 0.

For each triplet:
    1. setup_finetune_config (epoches=24, regex-patched)
    2. HEAL train.py 1 epoch (skipped if bestval@N>23 ckpt already present)
    3. Export ONNX
    4. Build FP16 + INT8 engines (skipped if exists)
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, time, traceback, multiprocessing as mp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
RANDOM_CACHE = REPO_ROOT / "models/p0_random_cache"
B_CACHE = REPO_ROOT / "models/stage_b_cache"
B_CACHE.mkdir(parents=True, exist_ok=True)

CALIB_SPATIAL = REPO_ROOT / "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = REPO_ROOT / "calibration/pyramid_dair_collab_tego.npy"

TRIPLETS = [
    "064_128_256","032_072_128","024_072_136","032_040_136","032_040_128",
    "032_032_128","024_040_128","032_040_064","016_064_064","024_040_072",
    "032_032_032","024_024_072","024_040_032","016_032_064","016_024_072",
    "016_016_072","016_016_064","024_024_016","016_016_040","016_016_016",
]


def setup_ft_dir(sig: str) -> Path | None:
    src = RANDOM_CACHE / f"prune_{sig}"
    dst = B_CACHE / f"ft_{sig}"
    if not (src / "net_epoch_bestval_at23.pth").exists():
        return None
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "net_epoch_bestval_at23.pth").exists():
        shutil.copy(src / "net_epoch_bestval_at23.pth", dst / "net_epoch_bestval_at23.pth")
    if not (dst / "config.yaml").exists():
        with open(src / "config.yaml") as f:
            cfg = f.read()
        cfg = re.sub(r"epoches:\s*\d+", "epoches: 24", cfg)
        with open(dst / "config.yaml", "w") as f:
            f.write(cfg)
    return dst


def _has_finetuned_ckpt(ft_dir: Path) -> bool:
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
    ft_dir = setup_ft_dir(sig)
    if ft_dir is None: return None
    if _has_finetuned_ckpt(ft_dir):
        print(f"[ft GPU{gpu}] {sig} pickup")
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
                           capture_output=True, text=True, timeout=5400,
                           start_new_session=True)
    except subprocess.TimeoutExpired:
        if _has_finetuned_ckpt(ft_dir):
            print(f"[ft GPU{gpu}] {sig} timeout but partial ckpt OK")
            return ft_dir
        return None
    if r.returncode != 0:
        print(f"[ft GPU{gpu}] {sig} FAILED: {r.stderr[-300:]}")
        return None
    print(f"[ft GPU{gpu}] {sig} done ({time.time()-t0:.0f}s)")
    return ft_dir


def export_onnx(sig: str, ft_dir: Path, gpu: int) -> Path | None:
    out_onnx = B_CACHE / f"{sig}_ft.onnx"
    if out_onnx.exists(): return out_onnx
    ckpts = sorted(ft_dir.glob("net_epoch*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts: return None
    ckpt = ckpts[-1]
    cmd = [PYTHON, str(REPO_ROOT / "tools/export_onnx_pyramid_collab.py"),
           "--ckpt", str(ckpt), "--hypes", str(ft_dir / "config.yaml"),
           "--out", str(out_onnx), "--feat-h", "128"]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": os.environ.get("PATH", "")}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env,
                           start_new_session=True)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not out_onnx.exists():
        print(f"[onnx GPU{gpu}] {sig} FAILED: {r.stderr[-200:]}")
        return None
    return out_onnx


def build_engine(sig: str, prec: str, onnx_path: Path, gpu: int) -> Path | None:
    eng = B_CACHE / f"{sig}_ft_{prec}.engine"
    if eng.exists(): return eng
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx_path), "--precision", prec, "--engine", str(eng),
           "--report", str(B_CACHE / f"{sig}_ft_{prec}_build.json"),
           "--input-shape", "2,64,128,256",
           "--extra-input-shape", "t_ego:2,2,3",
           "--n-warmup", "30", "--n-measure", "100"]
    if prec == "int8":
        cmd += ["--calib-multi", f"spatial_features:{CALIB_SPATIAL}",
                "--calib-multi", f"t_ego:{CALIB_TEGO}",
                "--calib-cache", str(B_CACHE / f"{sig}_ft_int8_calib.cache")]
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": os.environ.get("PATH", "")}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env,
                           cwd=REPO_ROOT, start_new_session=True)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not eng.exists():
        print(f"[build GPU{gpu}] {sig} {prec} FAILED: {r.stderr[-200:]}")
        return None
    print(f"[build GPU{gpu}] {sig} {prec} OK")
    return eng


def process_one(args):
    sig, gpu = args
    try:
        print(f"\n[GPU{gpu}] {sig} starting")
        t0 = time.time()
        ft_dir = finetune(sig, gpu)
        if ft_dir is None: return
        onnx = export_onnx(sig, ft_dir, gpu)
        if onnx is None: return
        for prec in ("fp16", "int8"):
            build_engine(sig, prec, onnx, gpu)
        print(f"[GPU{gpu}] {sig} all artifacts ready ({(time.time()-t0)/60:.1f} min)")
    except Exception as e:
        print(f"[GPU{gpu}] {sig} EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()


def gpu_worker(gpu_anchors):
    gpu, anchors = gpu_anchors
    for sig in anchors:
        process_one((sig, gpu))


def main():
    t0 = time.time()
    gpu_ids = [1, 2, 3, 4, 5, 6]
    tasks = [(sig, gpu_ids[i % len(gpu_ids)]) for i, sig in enumerate(TRIPLETS)]
    by_gpu = {}
    for sig, gpu in tasks:
        by_gpu.setdefault(gpu, []).append(sig)
    print(f"Task distribution: {by_gpu}")
    with mp.Pool(processes=6) as pool:
        pool.map(gpu_worker, list(by_gpu.items()))
    print(f"\n[done] all triplets in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
