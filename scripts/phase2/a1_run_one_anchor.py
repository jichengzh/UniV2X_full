"""Run one A.1 phase1 anchor: build + lat + AP, write row JSON.

Usage:
  python a1_run_one_anchor.py --triplet T1_base --q-tag Q_mix_s0 --gpu 0

写出: results/a1_q_expand/{triplet}_{q_tag}.row.json
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
DATA_OUT = REPO_ROOT / "paper_learning/2. AAAI最终故事/data"
CACHE = REPO_ROOT / "models/e2e_cache_a1"
AP_DIR = REPO_ROOT / "results/a1_q_expand"
CALIB_DIR = REPO_ROOT / "calibration/pyramid_dair_e2e_32k"
DAIR_ROOT = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
MAX_VOX = 32000
CACHE.mkdir(parents=True, exist_ok=True)
AP_DIR.mkdir(parents=True, exist_ok=True)

TRIPLET_MAP = {
    "T1_base": ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth", (64, 128, 256)),
    "T2_p25":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch_bestval_at25.pth", (48, 96, 192)),
    "T3_p37":  (str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/net_epoch_bestval_at33.pth"), (40, 80, 160)),
    "T4_p50":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at23.pth", (32, 64, 128)),
    "T5_p62":  (str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/net_epoch_bestval_at33.pth"), (24, 56, 128)),
    "T6_p75":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at25.pth", (16, 32, 64)),
    "T7_wide_shallow": (str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/net_epoch_bestval_at33.pth"), (48, 64, 128)),
    "T8_narrow_deep":  (str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/net_epoch_bestval_at33.pth"), (24, 48, 192)),
}

# (q_tag) -> mixed_int8_substr (correct interpretation A semantic)
Q_MAP = {
    "Q_mix_s0": "layer0",
    "Q_mix_s2": "layer2",
}


def run(cmd, env, timeout=1800):
    print(f"[run ] {' '.join(cmd[:4])} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"cmd failed rc={r.returncode}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplet", required=True, choices=list(TRIPLET_MAP))
    ap.add_argument("--q-tag", required=True, choices=list(Q_MAP))
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    ckpt, planes = TRIPLET_MAP[args.triplet]
    ckpt_dir = str(Path(ckpt).parent)
    mixed_substr = Q_MAP[args.q_tag]

    onnx = REPO_ROOT / "models/e2e_cache" / f"{args.triplet}.onnx"
    if not onnx.exists():
        sys.exit(f"ERR: ONNX missing: {onnx}")

    engine = CACHE / f"{args.triplet}_{args.q_tag}.engine"
    build_rep = CACHE / f"{args.triplet}_{args.q_tag}.build.json"
    bench_rep = CACHE / f"{args.triplet}_{args.q_tag}_bench.json"
    calib_cache = CACHE / f"{args.triplet}_{args.q_tag}_calib.cache"
    ap_rep = AP_DIR / f"{args.triplet}_{args.q_tag}_ap.json"
    row_out = AP_DIR / f"{args.triplet}_{args.q_tag}.row.json"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    print(f"[gpu {args.gpu}] {args.triplet}/{args.q_tag} start")

    # 1. Build
    if engine.exists() and build_rep.exists():
        print(f"[build] {args.triplet}/{args.q_tag} cached")
        bld = json.loads(build_rep.read_text())
    else:
        t0 = time.time()
        cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx),
               "--precision", "mixed",
               "--engine", str(engine),
               "--report", str(build_rep),
               "--workspace-mb", "4096",
               "--skip-bench",
               "--calibrator", "minmax",
               "--mixed-int8-substr", mixed_substr,
               "--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
               "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
               "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
               "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
               "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy",
               "--calib-cache", str(calib_cache)]
        run(cmd, env, timeout=1200)
        bld = json.loads(build_rep.read_text())
        print(f"[build] {args.triplet}/{args.q_tag}: {time.time()-t0:.0f}s, {bld.get('engine_size_mb', 0):.1f}MB")

    # 2. Bench
    if bench_rep.exists():
        print(f"[bench] {args.triplet}/{args.q_tag} cached")
        bench = json.loads(bench_rep.read_text())
    else:
        t0 = time.time()
        cmd = [PYTHON, str(REPO_ROOT / "scripts/phase2/e2e_bench_pyramid.py"),
               "--engine", str(engine),
               "--ckpt-dir", ckpt_dir,
               "--dair-root", DAIR_ROOT,
               "--max-voxels", str(MAX_VOX),
               "--n-warmup", "10",
               "--n-measure", "80",
               "--report", str(bench_rep)]
        run(cmd, env, timeout=1800)
        bench = json.loads(bench_rep.read_text())
        print(f"[bench] {args.triplet}/{args.q_tag}: e2e={bench['lat_e2e_ms']['mean']:.2f}ms ({time.time()-t0:.0f}s)")

    # 3. AP
    if ap_rep.exists():
        print(f"[ap   ] {args.triplet}/{args.q_tag} cached")
        ap_data = json.loads(ap_rep.read_text())
    else:
        t0 = time.time()
        anchor_id = f"{args.triplet}_{args.q_tag}_e2e"
        cmd = [PYTHON, str(REPO_ROOT / "scripts/phase2/e2e_eval_ap.py"),
               "--engine", str(engine),
               "--ckpt-dir", ckpt_dir,
               "--dair-root", DAIR_ROOT,
               "--max-voxels", str(MAX_VOX),
               "--n-samples", "1789",
               "--tag", anchor_id,
               "--report", str(ap_rep)]
        run(cmd, env, timeout=3600)
        ap_data = json.loads(ap_rep.read_text())
        ap50 = ap_data.get("ap_50") or ap_data.get("ap50") or 0.0
        print(f"[ap   ] {args.triplet}/{args.q_tag}: AP50={ap50:.4f} ({time.time()-t0:.0f}s)")

    # 4. Compose row (schema v1.3 精简后只需 throughput, 不再依赖 baseline)
    lat_e2e_mean = bench["lat_e2e_ms"]["mean"]
    ap30 = ap_data.get("ap_30") or ap_data.get("ap30")
    ap50 = ap_data.get("ap_50") or ap_data.get("ap50")
    ap70 = ap_data.get("ap_70") or ap_data.get("ap70")

    # Schema 精简 (2026-05-15): lat 仅保留 throughput_fps. 详细 lat breakdown 见
    # bench_rep ({tag}_{q_tag}_bench.json) — 需要时按需读取, 不进 dataset table.
    row = {
        "triplet": args.triplet,
        "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
        "prune_object": "channel" if planes != (64, 128, 256) else "none",
        "sparse_mask": "dense",
        "q_tag": args.q_tag, "prec_flag": "mixed",
        "q_bits": "mixed",
        "q_bits_per_stage": "INT8/FP16/FP16" if args.q_tag == "Q_mix_s0" else "FP16/FP16/INT8",
        "q_granularity": "per-tensor", "q_object": "W+A",
        "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "hardware": "rtx4090",
        "device": bench.get("device", "cuda:0"),
        "max_voxels": MAX_VOX,
        "n_collected": bench["n_collected"], "n_skipped": bench["n_skipped"],
        "real_voxels_mean": bench["real_voxels"]["mean"],
        "real_voxels_p99": bench["real_voxels"]["p99"],
        "throughput_fps": round(1000.0 / lat_e2e_mean, 4),
        "ap30": ap30, "ap50": ap50, "ap70": ap70,
        "engine_size_mb": bld.get("engine_size_mb"),
        "build_secs": bld.get("build_secs"),
        "build_success": True, "fail_reason": "",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    row_out.write_text(json.dumps(row, indent=2, default=str))
    print(f"[done] {args.triplet}/{args.q_tag} -> {row_out.name}")


if __name__ == "__main__":
    main()
