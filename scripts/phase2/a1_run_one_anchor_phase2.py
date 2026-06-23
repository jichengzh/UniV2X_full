"""A.1 phase 2: 3 granularity variants on 8 triplets.

支持 3 个 Q 配置:
  Q_int8_pc_wo: per-channel weight (默认), W-only (activation FP16)
                → m4_8_trt_build_bench.py --precision int8 --w-only
  Q_int8_pt_wa: per-tensor weight, W+A
                → 预处理 ONNX 加 per-tensor Q/DQ → TRT build with explicit Q/DQ
                (no calibrator needed because Q/DQ scales are in ONNX)
  Q_int8_pt_wo: per-tensor weight, W-only
                → 同 pt_wa 的预处理 ONNX + --w-only flag

写出: results/a1_q_expand_phase2/{triplet}_{q_tag}.row.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
DATA = REPO / "paper_learning/2. AAAI最终故事/data"
CACHE = REPO / "models/e2e_cache_a1_phase2"
AP_DIR = REPO / "results/a1_q_expand_phase2"
CALIB = REPO / "calibration/pyramid_dair_e2e_32k"
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
MAX_VOX = 32000

CACHE.mkdir(parents=True, exist_ok=True)
AP_DIR.mkdir(parents=True, exist_ok=True)

TRIPLET_MAP = {
    "T1_base": ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth", (64, 128, 256)),
    "T2_p25":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch_bestval_at25.pth", (48, 96, 192)),
    "T3_p37":  (str(REPO / "models/dataset_a_cache/ft_040_080_160/net_epoch_bestval_at33.pth"), (40, 80, 160)),
    "T4_p50":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at23.pth", (32, 64, 128)),
    "T5_p62":  (str(REPO / "models/dataset_a_cache/ft_024_056_128/net_epoch_bestval_at33.pth"), (24, 56, 128)),
    "T6_p75":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at25.pth", (16, 32, 64)),
    "T7_wide_shallow": (str(REPO / "models/dataset_a_cache/ft_048_064_128/net_epoch_bestval_at33.pth"), (48, 64, 128)),
    "T8_narrow_deep":  (str(REPO / "models/dataset_a_cache/ft_024_048_192/net_epoch_bestval_at33.pth"), (24, 48, 192)),
}


def get_build_args(q_tag: str, triplet: str) -> dict:
    """Return dict with onnx path, extra flags for given Q variant."""
    base_onnx = REPO / "models/e2e_cache" / f"{triplet}.onnx"
    pt_onnx = REPO / "models/e2e_cache" / f"{triplet}_pt_weight.onnx"  # preprocessed
    if q_tag == "Q_int8_pc_wo":
        return {"onnx": base_onnx, "extra": ["--w-only"], "uses_calibrator": True,
                "q_bits_per_stage": "INT8/INT8/INT8", "q_granularity": "per-channel",
                "q_object": "W-only"}
    if q_tag == "Q_int8_pt_wa":
        return {"onnx": pt_onnx, "extra": [], "uses_calibrator": True,
                "q_bits_per_stage": "INT8/INT8/INT8", "q_granularity": "per-tensor",
                "q_object": "W+A"}
    if q_tag == "Q_int8_pt_wo":
        return {"onnx": pt_onnx, "extra": ["--w-only"], "uses_calibrator": True,
                "q_bits_per_stage": "INT8/INT8/INT8", "q_granularity": "per-tensor",
                "q_object": "W-only"}
    raise ValueError(f"unknown q_tag: {q_tag}")


def run(cmd, env, timeout=1800):
    print(f"[run ] {' '.join(cmd[:4])} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"cmd failed rc={r.returncode}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplet", required=True, choices=list(TRIPLET_MAP))
    ap.add_argument("--q-tag", required=True,
                    choices=["Q_int8_pc_wo", "Q_int8_pt_wa", "Q_int8_pt_wo"])
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    ckpt, planes = TRIPLET_MAP[args.triplet]
    ckpt_dir = str(Path(ckpt).parent)
    cfg = get_build_args(args.q_tag, args.triplet)
    if not cfg["onnx"].exists():
        sys.exit(f"ERR: ONNX missing: {cfg['onnx']}")

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
        bld = json.loads(build_rep.read_text())
        print(f"[build] {args.triplet}/{args.q_tag} cached engine={bld.get('engine_size_mb',0):.1f}MB")
    else:
        t0 = time.time()
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(cfg["onnx"]),
               "--precision", "int8",
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", "4096", "--skip-bench",
               "--calibrator", "minmax",
               "--calib-multi", f"voxel_features:{CALIB}/voxel_features.npy",
               "--calib-multi", f"voxel_num_points:{CALIB}/voxel_num_points.npy",
               "--calib-multi", f"voxel_coords:{CALIB}/voxel_coords.npy",
               "--calib-multi", f"voxel_mask:{CALIB}/voxel_mask.npy",
               "--calib-multi", f"t_ego:{CALIB}/t_ego.npy",
               "--calib-cache", str(calib_cache)] + cfg["extra"]
        run(cmd, env, timeout=1200)
        bld = json.loads(build_rep.read_text())
        print(f"[build] {args.triplet}/{args.q_tag}: {time.time()-t0:.0f}s, {bld.get('engine_size_mb',0):.1f}MB")

    # 2. Bench
    if bench_rep.exists():
        bench = json.loads(bench_rep.read_text())
        print(f"[bench] {args.triplet}/{args.q_tag} cached")
    else:
        t0 = time.time()
        cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_bench_pyramid.py"),
               "--engine", str(engine), "--ckpt-dir", ckpt_dir,
               "--dair-root", DAIR, "--max-voxels", str(MAX_VOX),
               "--n-warmup", "10", "--n-measure", "80", "--report", str(bench_rep)]
        run(cmd, env, timeout=1800)
        bench = json.loads(bench_rep.read_text())
        print(f"[bench] {args.triplet}/{args.q_tag}: e2e={bench['lat_e2e_ms']['mean']:.2f}ms ({time.time()-t0:.0f}s)")

    # 3. AP
    if ap_rep.exists():
        ap_data = json.loads(ap_rep.read_text())
        print(f"[ap   ] {args.triplet}/{args.q_tag} cached")
    else:
        t0 = time.time()
        cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
               "--engine", str(engine), "--ckpt-dir", ckpt_dir,
               "--dair-root", DAIR, "--max-voxels", str(MAX_VOX),
               "--n-samples", "1789",
               "--tag", f"{args.triplet}_{args.q_tag}_e2e",
               "--report", str(ap_rep)]
        run(cmd, env, timeout=3600)
        ap_data = json.loads(ap_rep.read_text())
        print(f"[ap   ] {args.triplet}/{args.q_tag}: AP50={ap_data['ap50']:.4f} ({time.time()-t0:.0f}s)")

    # 4. Row JSON
    lat_e2e_mean = bench["lat_e2e_ms"]["mean"]
    row = {
        "triplet": args.triplet,
        "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
        "prune_object": "channel" if planes != (64, 128, 256) else "none",
        "sparse_mask": "dense",
        "q_tag": args.q_tag, "prec_flag": "int8",
        "q_bits": "INT8", "q_bits_per_stage": cfg["q_bits_per_stage"],
        "q_granularity": cfg["q_granularity"], "q_object": cfg["q_object"],
        "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "hardware": "rtx4090",
        "device": bench.get("device", "cuda:0"),
        "max_voxels": MAX_VOX,
        "n_collected": bench["n_collected"], "n_skipped": bench["n_skipped"],
        "real_voxels_mean": bench["real_voxels"]["mean"],
        "real_voxels_p99": bench["real_voxels"]["p99"],
        "throughput_fps": round(1000.0 / lat_e2e_mean, 4),
        "ap30": ap_data.get("ap_30") or ap_data.get("ap30"),
        "ap50": ap_data.get("ap_50") or ap_data.get("ap50"),
        "ap70": ap_data.get("ap_70") or ap_data.get("ap70"),
        "engine_size_mb": bld.get("engine_size_mb"),
        "build_secs": bld.get("build_secs"),
        "build_success": True, "fail_reason": "",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    row_out.write_text(json.dumps(row, indent=2, default=str))
    print(f"[done] {args.triplet}/{args.q_tag} -> {row_out.name}")


if __name__ == "__main__":
    main()
