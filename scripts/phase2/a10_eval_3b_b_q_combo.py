"""Phase 3b — B × Q combo AP eval (FT=8 fixed) for g8 extreme triplets.

For each triplet ∈ {T_g8_p87, T_g8_p93, T_g8_p97}, using FT=8 ckpt:
  build 4 engines: fp16, int8_minmax, int8_pc_wo, int8_entropy
  run AP eval each
Total: 3 × 4 = 12 AP eval.

Output: /tmp/a10_phase3b/{tag}_{qvar}_ap.json + combo_3b.csv
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
OUT = Path("/tmp/a10_phase3b")
OUT.mkdir(parents=True, exist_ok=True)

TRIPLETS = [
    ("T_g8_p87", "ft_T_g8_p87_raw"),
    ("T_g8_p93", "ft_T_g8_p93_raw"),
    ("T_g8_p97", "ft_T_g8_p97_raw"),
]
# (variant_tag, build_args_extra)
Q_VARIANTS = [
    ("fp16", ["--precision", "fp16"]),
    ("int8_mm", ["--precision", "int8", "--calibrator", "minmax",
                 "--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
                 "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
                 "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
                 "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
                 "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy"]),
    ("int8_pc_wo", ["--precision", "int8", "--calibrator", "minmax", "--w-only",
                     "--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
                     "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
                     "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
                     "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
                     "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy"]),
    ("int8_ent", ["--precision", "int8", "--calibrator", "entropy",
                   "--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
                   "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
                   "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
                   "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
                   "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy"]),
]
FT_TARGET = 8

_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def run_one(spec):
    triplet, ft_dir_name, qvar, q_args, ft_cache_root, baseline_ep = spec
    tag = f"{triplet}_ft{FT_TARGET:02d}_{qvar}"
    ft_dir = Path(ft_cache_root) / ft_dir_name
    ckpt = ft_dir / f"net_epoch{baseline_ep + FT_TARGET}.pth"
    cfg = ft_dir / "config.yaml"
    onnx = OUT / f"{triplet}_ft{FT_TARGET:02d}.onnx"  # pre-exported by main
    engine = OUT / f"{tag}.engine"
    cache = OUT / f"{tag}.cache"
    build_rep = OUT / f"{tag}.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, triplet, qvar, d.get("ap50") or d.get("ap_50"), 0, "cached")
    if not ckpt.exists():
        return (tag, triplet, qvar, None, 0, f"ckpt missing: {ckpt.name}")
    if not onnx.exists():
        return (tag, triplet, qvar, None, 0, f"onnx missing (pre-export step?): {onnx.name}")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(_GPU),
           "PYTHONPATH": str(HEAL)}
    t0 = time.time()

    # 2. TRT build for this Q variant
    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx),
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench",
               "--calib-cache", str(cache)] + q_args
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        if r.returncode != 0:
            (OUT / f"{tag}.engine.err").write_text(r.stdout + r.stderr)
            return (tag, triplet, qvar, None, time.time()-t0, "engine fail")

    # 3. AP eval n=1789
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", str(ft_dir),
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", "1789", "--tag", tag,
           "--report", str(ap_rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        (OUT / f"{tag}.ap.err").write_text(r.stdout + r.stderr)
        return (tag, triplet, qvar, None, time.time()-t0, "AP fail")

    d = json.loads(ap_rep.read_text())
    ap50 = d.get("ap50") or d.get("ap_50")
    return (tag, triplet, qvar, ap50, time.time()-t0, "OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft-cache-root", default=str(REPO / "models/dataset_a_cache_g8"))
    p.add_argument("--baseline-epoch", type=int, required=True)
    p.add_argument("--gpus", default="2,3,4,5,6,7")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]

    # PRE-EXPORT ONNX sequentially per triplet (avoids race condition when 4 Q
    # variants for same triplet write to the shared onnx path concurrently).
    print(f"[3b] pre-exporting ONNX for 3 triplets (sequential, ~30s each)")
    for triplet, ft_dir_name in TRIPLETS:
        ft_dir = Path(args.ft_cache_root) / ft_dir_name
        ckpt = ft_dir / f"net_epoch{args.baseline_epoch + FT_TARGET}.pth"
        cfg = ft_dir / "config.yaml"
        onnx = OUT / f"{triplet}_ft{FT_TARGET:02d}.onnx"
        if onnx.exists():
            print(f"  [{triplet}] onnx cached")
            continue
        if not ckpt.exists():
            print(f"  [{triplet}] ERROR: ckpt missing {ckpt}")
            continue
        cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_e2e.py"),
               "--ckpt", str(ckpt), "--hypes", str(cfg),
               "--out", str(onnx), "--max-voxels", "32000"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpus[0]),
               "PYTHONPATH": str(HEAL)}
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if r.returncode != 0:
            (OUT / f"{triplet}.onnx.err").write_text(r.stdout + r.stderr)
            print(f"  [{triplet}] ONNX EXPORT FAIL — see err file")
        else:
            print(f"  [{triplet}] onnx exported ({onnx.stat().st_size / 1e6:.1f} MB)")

    specs = []
    for triplet, ft_dir_name in TRIPLETS:
        for qvar, q_args in Q_VARIANTS:
            specs.append((triplet, ft_dir_name, qvar, q_args,
                          args.ft_cache_root, args.baseline_epoch))
    print(f"[3b] {len(specs)} (triplet × Q) AP evals on {len(gpus)} GPU")

    with Pool(processes=len(gpus), initializer=_init, initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, specs))

    print(f"\n{'tag':<32}{'AP50':>9}{'secs':>7}  status")
    by_tup = {}
    for tag, triplet, qvar, ap50, secs, status in sorted(results, key=lambda r: r[0]):
        ap_str = f'{ap50:.3f}' if ap50 is not None else 'FAIL'
        print(f'{tag:<32}{ap_str:>9}{secs:>7.0f}  {status}')
        by_tup.setdefault(triplet, {})[qvar] = ap50

    csv_path = OUT / "combo_3b.csv"
    with open(csv_path, "w") as f:
        cols = ["triplet"] + [q for q, _ in Q_VARIANTS]
        f.write(",".join(cols) + "\n")
        for triplet, _ in TRIPLETS:
            row = [triplet]
            for q, _ in Q_VARIANTS:
                v = by_tup.get(triplet, {}).get(q)
                row.append(f"{v:.4f}" if v is not None else "NA")
            f.write(",".join(row) + "\n")
    print(f"\ncombo_3b → {csv_path}")


if __name__ == "__main__":
    main()
