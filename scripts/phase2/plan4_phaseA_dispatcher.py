"""Plan 4 Phase A dispatcher — B × Q 因素归因 (120 anchor @ FT=8 锁定).

24 triplet × 5 Q × FT=8 × D=D1_default_4gb = 120 anchor.

24 triplet:
  T1_base (no FT, baseline) — Pyramid_DAIR_m1_base bestval_at23
  T2_p25 / T4_p50 / T6_p75 (HEAL 官方, 补训 epoch 25→31)
  T3/T5/T7/T8/T10-T22 (17 Track A v2, ft_*/net_epoch31)
  T_g8_p87/p93/p97 (3 g8, ft_T_g8_p*/net_epoch27)

5 Q: fp32, fp16, int8_pt_wa_mm, int8_pc_wo, int8_pt_wa_ent

D = D1_default_4gb (固定)

Output:
  /tmp/plan4_phaseA/{anchor}.engine + bench.json + ap.json
  /tmp/plan4_phaseA/phase_a_anchors.csv (120 行)
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

OUT = Path("/tmp/plan4_phaseA")
OUT.mkdir(parents=True, exist_ok=True)


# (triplet_name, ckpt_path, planes (s0,s1,s2), groups, prune_pct)
TRIPLETS = [
    ("T1_base", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth", (64, 128, 256), 32, 0.0),
    ("T2_p25", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch31.pth", (48, 96, 192), 32, 0.25),
    ("T3_p37", str(REPO / "models/dataset_a_cache/ft_040_080_160/net_epoch31.pth"), (40, 80, 160), 32, 0.37),
    ("T4_p50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch31.pth", (32, 64, 128), 32, 0.50),
    ("T5_p62", str(REPO / "models/dataset_a_cache/ft_024_056_128/net_epoch31.pth"), (24, 56, 128), 32, 0.62),
    ("T6_p75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch31.pth", (16, 32, 64), 32, 0.75),
    ("T7_wide_shallow", str(REPO / "models/dataset_a_cache/ft_048_064_128/net_epoch31.pth"), (48, 64, 128), 32, 0.40),
    ("T8_narrow_deep", str(REPO / "models/dataset_a_cache/ft_024_048_192/net_epoch31.pth"), (24, 48, 192), 32, 0.30),
    ("T10_p11", str(REPO / "models/dataset_a_cache/ft_016_128_256/net_epoch31.pth"), (16, 128, 256), 32, 0.11),
    ("T11_p14", str(REPO / "models/dataset_a_cache/ft_064_064_256/net_epoch31.pth"), (64, 64, 256), 32, 0.14),
    ("T12_p21", str(REPO / "models/dataset_a_cache/ft_032_064_256/net_epoch31.pth"), (32, 64, 256), 32, 0.21),
    ("T13_p29", str(REPO / "models/dataset_a_cache/ft_032_032_256/net_epoch31.pth"), (32, 32, 256), 32, 0.29),
    ("T14_p36", str(REPO / "models/dataset_a_cache/ft_016_016_256/net_epoch31.pth"), (16, 16, 256), 32, 0.36),
    ("T15_p39", str(REPO / "models/dataset_a_cache/ft_016_128_128/net_epoch31.pth"), (16, 128, 128), 32, 0.39),
    ("T16_p54", str(REPO / "models/dataset_a_cache/ft_016_064_128/net_epoch31.pth"), (16, 64, 128), 32, 0.54),
    ("T17_p57", str(REPO / "models/dataset_a_cache/ft_032_032_128/net_epoch31.pth"), (32, 32, 128), 32, 0.57),
    ("T18_p64", str(REPO / "models/dataset_a_cache/ft_016_016_128/net_epoch31.pth"), (16, 16, 128), 32, 0.64),
    ("T19_p71", str(REPO / "models/dataset_a_cache/ft_032_032_064/net_epoch31.pth"), (32, 32, 64), 32, 0.71),
    ("T20_p79", str(REPO / "models/dataset_a_cache/ft_016_016_064/net_epoch31.pth"), (16, 16, 64), 32, 0.79),
    ("T21_p82", str(REPO / "models/dataset_a_cache/ft_016_032_032/net_epoch31.pth"), (16, 32, 32), 32, 0.82),
    ("T22_p89", str(REPO / "models/dataset_a_cache/ft_016_016_016/net_epoch31.pth"), (16, 16, 16), 32, 0.89),
    ("T_g8_p87", str(REPO / "models/dataset_a_cache_g8/ft_T_g8_p87_raw/net_epoch27.pth"), (8, 8, 8), 8, 0.87),
    ("T_g8_p93", str(REPO / "models/dataset_a_cache_g8/ft_T_g8_p93_raw/net_epoch27.pth"), (8, 4, 4), 8, 0.93),
    ("T_g8_p97", str(REPO / "models/dataset_a_cache_g8/ft_T_g8_p97_raw/net_epoch27.pth"), (4, 4, 4), 8, 0.97),
]


def make_calib_args() -> list[str]:
    return ["--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
            "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
            "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
            "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
            "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy"]


Q_VARIANTS = [
    ("fp32", ["--precision", "fp32"]),
    ("fp16", ["--precision", "fp16"]),
    ("int8_pt_wa_mm", ["--precision", "int8", "--calibrator", "minmax"] + make_calib_args()),
    ("int8_pc_wo", ["--precision", "int8", "--calibrator", "minmax", "--w-only"] + make_calib_args()),
    ("int8_pt_wa_ent", ["--precision", "int8", "--calibrator", "entropy"] + make_calib_args()),
]

_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def export_onnx_if_needed(trip_name: str, ckpt_path: str, onnx_path: Path, gpu: int) -> bool:
    if onnx_path.exists():
        return True
    cfg_path = Path(ckpt_path).parent / "config.yaml"
    if not cfg_path.exists():
        print(f"[{trip_name}] config.yaml missing at {cfg_path}", flush=True)
        return False
    if not Path(ckpt_path).exists():
        print(f"[{trip_name}] ckpt missing: {ckpt_path}", flush=True)
        return False
    cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_e2e.py"),
           "--ckpt", ckpt_path, "--hypes", str(cfg_path),
           "--out", str(onnx_path), "--max-voxels", "32000"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        (OUT / f"{trip_name}.onnx.err").write_text(r.stdout + r.stderr)
        return False
    return True


def run_one(spec):
    """spec = (trip_name, ckpt, planes, groups, prune_pct, q_name, q_args)"""
    trip_name, ckpt, planes, groups, prune, q_name, q_args = spec
    gpu = _GPU
    tag = f"{trip_name}_Q_{q_name}"
    onnx = OUT / f"{trip_name}.onnx"
    engine = OUT / f"{tag}.engine"
    cache = OUT / f"{tag}.cache"
    build_rep = OUT / f"{tag}.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, trip_name, q_name, d.get("ap50") or d.get("ap_50"),
                0, "cached")

    # Export ONNX
    if not export_onnx_if_needed(trip_name, ckpt, onnx, gpu):
        return (tag, trip_name, q_name, None, 0, "onnx fail")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}

    # Build engine
    t0 = time.time()
    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx), "--engine", str(engine),
               "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench",
               "--calib-cache", str(cache)] + q_args
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        if r.returncode != 0:
            (OUT / f"{tag}.engine.err").write_text(r.stdout + r.stderr)
            return (tag, trip_name, q_name, None, time.time()-t0, "engine fail")

    # AP eval
    ckpt_dir = str(Path(ckpt).parent)
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", ckpt_dir,
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", "1789", "--tag", tag,
           "--report", str(ap_rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode != 0:
        (OUT / f"{tag}.ap.err").write_text(r.stdout + r.stderr)
        return (tag, trip_name, q_name, None, time.time()-t0, "AP fail")

    d = json.loads(ap_rep.read_text())
    ap50 = d.get("ap50") or d.get("ap_50")
    return (tag, trip_name, q_name, ap50, time.time()-t0, "OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1,2,3,6,7",
                   help="comma-sep GPUs (typically 6 parallel)")
    p.add_argument("--triplets", default="all",
                   help="comma-sep triplet names or 'all'")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    if args.triplets == "all":
        triplets = TRIPLETS
    else:
        keep = set(args.triplets.split(","))
        triplets = [t for t in TRIPLETS if t[0] in keep]

    # Build specs (triplet × Q)
    specs = []
    for trip in triplets:
        trip_name, ckpt, planes, groups, prune = trip
        for q_name, q_args in Q_VARIANTS:
            specs.append((trip_name, ckpt, planes, groups, prune, q_name, q_args))
    print(f"[Phase A] {len(triplets)} triplet × {len(Q_VARIANTS)} Q = {len(specs)} anchor, "
          f"on {len(gpus)} GPU")

    # Step 1: pre-export ONNX (sequential, 1 per triplet)
    print(f"[Phase A] Step 1: pre-export ONNX ({len(triplets)} files)")
    for trip in triplets:
        trip_name, ckpt, planes, groups, prune = trip
        onnx = OUT / f"{trip_name}.onnx"
        if onnx.exists():
            print(f"  [{trip_name}] cached")
            continue
        # Use first GPU for serial export
        if export_onnx_if_needed(trip_name, ckpt, onnx, gpus[0]):
            print(f"  [{trip_name}] exported ({onnx.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  [{trip_name}] EXPORT FAIL")

    # Step 2: parallel dispatch 120 anchor
    print(f"[Phase A] Step 2: dispatch {len(specs)} (triplet × Q) eval on {len(gpus)} GPU")
    t0 = time.time()
    with Pool(processes=len(gpus), initializer=_init,
              initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, specs))
    print(f"[Phase A] dispatch wall: {(time.time()-t0)/60:.1f} min")

    # Step 3: write csv
    csv_path = OUT / "phase_a_anchors.csv"
    by_anchor = {}
    for tag, trip, q, ap, secs, status in sorted(results):
        by_anchor[(trip, q)] = (ap, status)
        ap_str = f'{ap:.4f}' if ap is not None else 'FAIL'
        print(f"  {tag:<32} AP={ap_str:>9}  {secs:>5.0f}s  {status}")

    with open(csv_path, "w") as f:
        cols = ["triplet", "planes_s0", "planes_s1", "planes_s2",
                "groups", "prune_pct", "q", "ap50", "ft", "d", "status"]
        f.write(",".join(cols) + "\n")
        for trip in triplets:
            trip_name, ckpt, planes, groups, prune = trip
            for q_name, _ in Q_VARIANTS:
                ap, status = by_anchor.get((trip_name, q_name), (None, "missing"))
                row = [trip_name, str(planes[0]), str(planes[1]), str(planes[2]),
                       str(groups), f"{prune:.2f}", q_name,
                       f"{ap:.4f}" if ap is not None else "NA",
                       "8", "D1_default_4gb", status]
                f.write(",".join(row) + "\n")
    print(f"\n[Phase A] csv → {csv_path}")

    # Summary stats
    aps = [v[0] for v in by_anchor.values() if v[0] is not None]
    print(f"\n[Phase A SUMMARY]")
    print(f"  N anchor: {len(aps)}/{len(specs)}")
    if aps:
        import numpy as np
        a = np.array(aps)
        print(f"  AP50 mean={a.mean():.4f} std={a.std():.4f}")
        print(f"  range=[{a.min():.4f}, {a.max():.4f}]")


if __name__ == "__main__":
    main()
