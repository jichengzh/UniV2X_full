"""Plan v3 §7.11 follow-up — FT=4 / FT=6 / FT=8 sweep at full Q × triplet grid.

Goal: empirically resolve the "FT=8 too large, FT=4 too noisy" dilemma.

For each (triplet ∈ 3) × (Q ∈ 5: fp32, fp16, int8_mm, int8_pc_wo, int8_ent) ×
    (FT ∈ {4, 6, 8}) = 45 anchors. Skip those already cached (from plan v2/v3):
  - FT=4 FP32: 3 (plan v2 §3a)
  - FT=8 all 5 Q: 15 (plan v2 §3a + §3b)
  - new evals needed: 45 - 18 = ~27

Reuses ft_T_g8_p{87,93,97}_raw/ ckpts at epoch:
  FT=4 → epoch 23 (baseline_g8 19 + 4)
  FT=6 → epoch 25 (baseline_g8 19 + 6)
  FT=8 → epoch 27 (baseline_g8 19 + 8) — cached from plan v3 §1.1 pre-export

Output: /tmp/a11_ft_sweep/ft_sweep.csv (45 row table: triplet × Q × FT)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
CALIB_DIR = REPO / "calibration/pyramid_dair_e2e_32k"
OUT = Path("/tmp/a11_ft_sweep")
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_EPOCH = 19
TRIPLETS = [
    ("T_g8_p87", "ft_T_g8_p87_raw"),
    ("T_g8_p93", "ft_T_g8_p93_raw"),
    ("T_g8_p97", "ft_T_g8_p97_raw"),
]
FT_LEVELS = [4, 6, 8]

# Q variants — fp32 is precision="fp32" (no INT8 calib needed)
def make_calib_args() -> list[str]:
    return ["--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
            "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
            "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
            "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
            "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy"]


Q_VARIANTS = [
    ("fp32", ["--precision", "fp32"]),
    ("fp16", ["--precision", "fp16"]),
    ("int8_mm", ["--precision", "int8", "--calibrator", "minmax"] + make_calib_args()),
    ("int8_pc_wo", ["--precision", "int8", "--calibrator", "minmax", "--w-only"]
                    + make_calib_args()),
    ("int8_ent", ["--precision", "int8", "--calibrator", "entropy"] + make_calib_args()),
]

_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def run_one(spec):
    triplet, ft_dir_name, ft, qvar, q_args, ft_cache_root = spec
    tag = f"{triplet}_ft{ft:02d}_{qvar}"
    ft_dir = Path(ft_cache_root) / ft_dir_name
    ckpt = ft_dir / f"net_epoch{BASELINE_EPOCH + ft}.pth"
    cfg = ft_dir / "config.yaml"
    onnx = OUT / f"{triplet}_ft{ft:02d}.onnx"
    engine = OUT / f"{tag}.engine"
    cache = OUT / f"{tag}.cache"
    build_rep = OUT / f"{tag}.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, triplet, ft, qvar, d.get("ap50") or d.get("ap_50"),
                0, "cached")
    if not ckpt.exists():
        return (tag, triplet, ft, qvar, None, 0, f"ckpt missing: {ckpt.name}")
    if not onnx.exists():
        return (tag, triplet, ft, qvar, None, 0,
                f"onnx missing (pre-export step?): {onnx.name}")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(_GPU),
           "PYTHONPATH": str(HEAL)}
    t0 = time.time()

    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx),
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench",
               "--calib-cache", str(cache)] + q_args
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           timeout=900)
        if r.returncode != 0:
            (OUT / f"{tag}.engine.err").write_text(r.stdout + r.stderr)
            return (tag, triplet, ft, qvar, None,
                    time.time()-t0, "engine fail")

    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", str(ft_dir),
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", "1789", "--tag", tag,
           "--report", str(ap_rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=1800)
    if r.returncode != 0:
        (OUT / f"{tag}.ap.err").write_text(r.stdout + r.stderr)
        return (tag, triplet, ft, qvar, None, time.time()-t0, "AP fail")

    d = json.loads(ap_rep.read_text())
    ap50 = d.get("ap50") or d.get("ap_50")
    return (tag, triplet, ft, qvar, ap50, time.time()-t0, "OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft-cache-root",
                   default=str(REPO / "models/dataset_a_cache_g8"))
    p.add_argument("--gpus", default="2,4,5,6,7")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    ft_cache_root = Path(args.ft_cache_root)

    # Step 1: pre-export ONNX for each (triplet, ft) — sequential to avoid races
    print(f"[ft_sweep] Step 1: pre-export ONNX "
          f"({len(TRIPLETS) * len(FT_LEVELS)} files)")
    for triplet, ft_dir_name in TRIPLETS:
        ft_dir = ft_cache_root / ft_dir_name
        for ft in FT_LEVELS:
            onnx = OUT / f"{triplet}_ft{ft:02d}.onnx"
            if onnx.exists():
                print(f"  [{triplet} ft{ft}] cached")
                continue
            ckpt = ft_dir / f"net_epoch{BASELINE_EPOCH + ft}.pth"
            cfg = ft_dir / "config.yaml"
            if not ckpt.exists():
                print(f"  [{triplet} ft{ft}] CKPT MISSING: {ckpt.name}")
                continue
            cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_e2e.py"),
                   "--ckpt", str(ckpt), "--hypes", str(cfg),
                   "--out", str(onnx), "--max-voxels", "32000"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus[0]),
                   "PYTHONPATH": str(HEAL)}
            r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               timeout=300)
            if r.returncode != 0:
                (OUT / f"{triplet}_ft{ft:02d}.onnx.err").write_text(
                    r.stdout + r.stderr)
                print(f"  [{triplet} ft{ft}] ONNX FAIL")
            else:
                print(f"  [{triplet} ft{ft}] onnx exported "
                      f"({onnx.stat().st_size / 1e6:.1f} MB)")

    # Step 2: dispatch 45 evals on GPU pool
    specs = []
    for triplet, ft_dir_name in TRIPLETS:
        for ft in FT_LEVELS:
            for qvar, q_args in Q_VARIANTS:
                specs.append((triplet, ft_dir_name, ft, qvar, q_args,
                              str(ft_cache_root)))

    print(f"[ft_sweep] Step 2: {len(specs)} (triplet × FT × Q) AP eval "
          f"on {len(gpus)} GPU")

    with Pool(processes=len(gpus), initializer=_init,
              initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, specs))

    print(f"\n{'tag':<36}{'AP50':>9}{'secs':>7}  status")
    by_tup = {}  # (triplet, ft) → {qvar: ap50}
    for tag, triplet, ft, qvar, ap50, secs, status in sorted(
            results, key=lambda r: r[0]):
        ap_str = f'{ap50:.3f}' if ap50 is not None else 'FAIL'
        print(f'{tag:<36}{ap_str:>9}{secs:>7.0f}  {status}')
        by_tup.setdefault((triplet, ft), {})[qvar] = ap50

    csv_path = OUT / "ft_sweep.csv"
    with open(csv_path, "w") as f:
        cols = ["triplet", "ft"] + [q for q, _ in Q_VARIANTS]
        f.write(",".join(cols) + "\n")
        for triplet, _ in TRIPLETS:
            for ft in FT_LEVELS:
                row = [triplet, str(ft)]
                for q, _ in Q_VARIANTS:
                    v = by_tup.get((triplet, ft), {}).get(q)
                    row.append(f"{v:.4f}" if v is not None else "NA")
                f.write(",".join(row) + "\n")
    print(f"\nft_sweep → {csv_path}")

    # Per-FT statistics
    print(f"\n[ft_sweep SUMMARY]")
    print(f"{'FT':>4}{'N':>5}{'mean':>9}{'std':>9}{'range':>9}{'min':>9}{'max':>9}")
    import numpy as np
    for ft in FT_LEVELS:
        vals = [v for triplet, _ in TRIPLETS
                for v in by_tup.get((triplet, ft), {}).values()
                if v is not None]
        if not vals:
            continue
        arr = np.array(vals)
        print(f'{ft:>4}{len(arr):>5}{arr.mean():>9.4f}{arr.std():>9.4f}'
              f'{arr.max()-arr.min():>9.4f}{arr.min():>9.4f}{arr.max():>9.4f}')


if __name__ == "__main__":
    main()
