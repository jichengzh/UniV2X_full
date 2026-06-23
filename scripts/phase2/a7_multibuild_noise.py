"""Multi-build noise floor 实验 — 验证 TRT autotuner 噪声是 IID 还是 systematic.

对 K 个代表 anchor 各跑 N 次 fresh build, 测每次 fps. 看:
  1. mean-of-N 是否比 single build 更可预测 (IID 噪声 → R² 升)
  2. 还是 mean 也漂移 ~28% (systematic 噪声 → R² 不变)

每次 build 用唯一 engine path 避免 cache hit. 同一 (T, Q, tactic, ws, BL) 重复 N 次.

用法:
  python a7_multibuild_noise.py --K 32 --N 5 --gpus 0,1,2,3,4,5
"""
from __future__ import annotations
import argparse, json, os, random, subprocess, time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CACHE = REPO / "models/e2e_cache_a7"
CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO / "results/a7_multibuild"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/tmp/a7_parallel")
LOG_DIR.mkdir(exist_ok=True)
CALIB = REPO / "calibration/pyramid_dair_e2e_32k"
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"

# Reuse a2 TRIPLET_MAP via lazy import
import sys
sys.path.insert(0, str(REPO / "scripts/phase2"))
from a2_run_one_anchor import TRIPLET_MAP, q_build_config

# 固定 D-config (default tactic, 4GB ws, BL=3 = TRT 默认) 测 build-to-build 噪声
TACTIC = "default"
WS_MB = 4096
BL = 3

_WORKER_GPU = None


def _init_worker(gpus):
    global _WORKER_GPU
    wid = current_process()._identity[0]
    _WORKER_GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] bound to GPU {_WORKER_GPU}", flush=True)


def run_one_build(spec):
    """Build + bench one (T, Q, build_idx). Returns (T, Q, build_idx, fps, build_secs)."""
    triplet, q_tag, build_idx = spec
    gpu = _WORKER_GPU
    ckpt, planes = TRIPLET_MAP[triplet]
    ckpt_dir = str(Path(ckpt).parent)
    qcfg = q_build_config(q_tag, triplet)
    if not qcfg["onnx"].exists():
        return (triplet, q_tag, build_idx, None, None, "ONNX missing")

    tag = f"{triplet}_{q_tag}_v{build_idx}"
    engine = CACHE / f"{tag}.engine"
    build_rep = CACHE / f"{tag}.build.json"
    bench_rep = CACHE / f"{tag}_bench.json"
    # calib cache shared across all versions of same (T, Q) —
    # calibration scales are deterministic given ONNX + calibrator class;
    # only the per-build tactic/kernel selection varies. 避免 6 worker 并行 OOM.
    calib_cache = CACHE / f"{triplet}_{q_tag}_calib.cache"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    t0 = time.time()

    # Build (no cache reuse — unique tag)
    cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(qcfg["onnx"]),
           "--precision", qcfg["precision"],
           "--engine", str(engine), "--report", str(build_rep),
           "--workspace-mb", str(WS_MB),
           "--tactic", TACTIC,
           "--builder-opt-level", str(BL),
           "--skip-bench"]
    if qcfg["uses_calib"]:
        cmd += ["--calib-multi", f"voxel_features:{CALIB}/voxel_features.npy",
                "--calib-multi", f"voxel_num_points:{CALIB}/voxel_num_points.npy",
                "--calib-multi", f"voxel_coords:{CALIB}/voxel_coords.npy",
                "--calib-multi", f"voxel_mask:{CALIB}/voxel_mask.npy",
                "--calib-multi", f"t_ego:{CALIB}/t_ego.npy",
                "--calib-cache", str(calib_cache)]
    cmd += qcfg["extras"]
    log = LOG_DIR / f"{tag}.log"
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
    if r.returncode != 0:
        return (triplet, q_tag, build_idx, None, time.time() - t0, "build failed")
    bld = json.loads(build_rep.read_text())

    # Bench
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_bench_pyramid.py"),
           "--engine", str(engine), "--ckpt-dir", ckpt_dir,
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-warmup", "10", "--n-measure", "80", "--report", str(bench_rep)]
    with open(log, "a") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
    if r.returncode != 0:
        return (triplet, q_tag, build_idx, None, time.time() - t0, "bench failed")
    bench = json.loads(bench_rep.read_text())
    fps = 1000.0 / bench["lat_e2e_ms"]["mean"]
    return (triplet, q_tag, build_idx, round(fps, 4), round(time.time() - t0, 1), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=32, help="anchor count")
    ap.add_argument("--N", type=int, default=5, help="builds per anchor")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    triplets = list(TRIPLET_MAP.keys())  # 21 T
    q_list = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]
    all_pairs = [(t, q) for t in triplets for q in q_list]  # 21 × 6 = 126

    random.seed(args.seed)
    sampled = random.sample(all_pairs, min(args.K, len(all_pairs)))
    print(f"[a7] sampled {len(sampled)} (T, Q) pairs, N={args.N} builds each "
          f"= {len(sampled) * args.N} total builds")
    print(f"     (tactic={TACTIC}, ws={WS_MB}MB, BL={BL})")

    specs = [(t, q, idx) for (t, q) in sampled for idx in range(args.N)]
    gpus = [int(x) for x in args.gpus.split(",")]
    print(f"[a7] dispatching {len(specs)} builds on {len(gpus)} GPUs")

    results = []
    t0 = time.time()
    with Pool(processes=len(gpus), initializer=_init_worker, initargs=(gpus,)) as pool:
        for r in pool.imap_unordered(run_one_build, specs):
            results.append(r)
            n = len(results)
            elapsed = time.time() - t0
            eta = elapsed / n * (len(specs) - n) if n else 0
            err = r[5] if r[5] else "OK"
            print(f"[{n:3d}/{len(specs)}] {r[0]}_{r[1]}_v{r[2]} fps={r[3]} "
                  f"({r[4]}s) {err} | elapsed {elapsed/60:.1f}min eta {eta/60:.1f}min",
                  flush=True)

    # Save raw
    raw_path = OUT_DIR / f"multibuild_K{args.K}_N{args.N}_raw.json"
    raw_path.write_text(json.dumps(
        [{"triplet": r[0], "q_tag": r[1], "build_idx": r[2],
          "fps": r[3], "build_secs": r[4], "error": r[5]} for r in results],
        indent=2, default=str))
    print(f"\nraw → {raw_path}")
    print(f"wall: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
