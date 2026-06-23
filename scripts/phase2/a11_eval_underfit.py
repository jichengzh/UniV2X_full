"""Phase 1.1 (plan v3) — FT-underfit AP eval.

Goal: anchor AP collapse 下限 by evaluating raw (FT=0) and lightly-finetuned (FT=2)
checkpoints across 4 Q variants for each of 3 extreme triplets = 24 anchors.

FT=0 raw ckpts (at epoch 19) were overwritten by HEAL bestval saves during plan v2
finetune — regenerate via structural_prune into raw_T_g8_p{87,93,97}/ dirs.
FT=2 ckpts (epoch 21) exist in plan v2's ft_T_g8_p*_raw/ dirs.

Expected per plan v3 §1.1 success criteria:
  - 24 AP eval completed
  - ≥6 anchors with AP<0.30 (strict path A)
  - ≥3 anchors with AP<0.10 (random AP region anchored)

Output: /tmp/a11_underfit/{tag}_ap.json + underfit.csv (24 rows: triplet/ft/qvar/ap50).
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
OUT = Path("/tmp/a11_underfit")
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_DIR = Path("/home/jichengzhi/heal_research/HEAL/opencood/logs/"
                    "Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45")
BASELINE_EPOCH = 19  # per phase3_finding.md baseline_g8 bestval

# (triplet_tag, num_filters_new, ft_dir_name_in_plan_v2_cache)
TRIPLETS = [
    ("T_g8_p87", [8, 8, 8], "ft_T_g8_p87_raw"),
    ("T_g8_p93", [8, 4, 4], "ft_T_g8_p93_raw"),
    ("T_g8_p97", [4, 4, 4], "ft_T_g8_p97_raw"),
]
FT_LEVELS = [0, 2]

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

_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def ensure_raw_ckpt(triplet: str, nf: list[int], raw_root: Path,
                    gpu: int) -> tuple[Path, Path] | None:
    """Regenerate raw (FT=0) pruned ckpt into raw_{triplet}/ dir if absent.

    Returns (ckpt_path, config_path) or None on failure.
    """
    out_dir = raw_root / f"raw_{triplet}"
    bestval_glob = list(out_dir.glob("net_epoch_bestval_at*.pth"))
    cfg = out_dir / "config.yaml"
    if bestval_glob and cfg.exists():
        print(f"[raw {triplet}] cached: {bestval_glob[0].name}")
        return bestval_glob[0], cfg
    out_dir.mkdir(parents=True, exist_ok=True)
    nf_str = ",".join(str(n) for n in nf)
    cmd = [PYTHON, str(REPO / "tools/structural_prune_pyramid.py"),
           "--orig-dir", str(BASELINE_DIR),
           "--out-dir", str(out_dir),
           "--num-filters-new", nf_str,
           "--groups", "8", "--width-per-group", "16"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}
    log = OUT / f"prune_{triplet}.log"
    print(f"[raw {triplet}] regen {nf_str} → {out_dir}")
    with open(log, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=600)
    if r.returncode != 0:
        print(f"[raw {triplet}] FAIL — see {log}")
        return None
    bestval_glob = list(out_dir.glob("net_epoch_bestval_at*.pth"))
    if not bestval_glob:
        print(f"[raw {triplet}] no bestval ckpt after prune")
        return None
    return bestval_glob[0], cfg


def run_one(spec):
    triplet, ft, qvar, q_args, ckpt_path, cfg_path, ckpt_dir = spec
    tag = f"{triplet}_ft{ft:02d}_{qvar}"
    onnx = OUT / f"{triplet}_ft{ft:02d}.onnx"  # pre-exported by main()
    engine = OUT / f"{tag}.engine"
    cache = OUT / f"{tag}.cache"
    build_rep = OUT / f"{tag}.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, triplet, ft, qvar,
                d.get("ap50") or d.get("ap_50"), 0, "cached")
    if not onnx.exists():
        return (tag, triplet, ft, qvar, None, 0,
                f"onnx missing: {onnx.name}")

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

    # FT=0 raw is just confirming AP~0 collapse — 500 samples is enough
    # FT=2 needs full 1789 for accurate AP measurement
    n_samples = "500" if ft == 0 else "1789"
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", str(ckpt_dir),
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", n_samples, "--tag", tag,
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
    p.add_argument("--raw-root",
                   default=str(REPO / "models/dataset_a_cache_g8"))
    p.add_argument("--gpus", default="2,3,4,5,6,7")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    ft_cache_root = Path(args.ft_cache_root)
    raw_root = Path(args.raw_root)

    # Step 1: regen raw ckpts (FT=0) sequentially
    print("[1.1] Step 1: regen 3 raw ckpts (sequential, ~2 min each)")
    raw_info = {}
    for triplet, nf, _ in TRIPLETS:
        res = ensure_raw_ckpt(triplet, nf, raw_root, gpu=gpus[0])
        if res is None:
            print(f"[1.1] ABORT: raw {triplet} regen failed")
            return
        raw_info[triplet] = res  # (ckpt, cfg)

    # Step 2: pre-export ONNX per (triplet, ft) — sequential to avoid races
    print(f"[1.1] Step 2: pre-export {len(TRIPLETS) * len(FT_LEVELS)} ONNX")
    onnx_specs = []  # (triplet, ft, ckpt, cfg, ckpt_dir)
    for triplet, _, ft_dir_name in TRIPLETS:
        for ft in FT_LEVELS:
            if ft == 0:
                ckpt, cfg = raw_info[triplet]
                ckpt_dir = ckpt.parent
            else:
                ckpt_dir = ft_cache_root / ft_dir_name
                ckpt = ckpt_dir / f"net_epoch{BASELINE_EPOCH + ft}.pth"
                cfg = ckpt_dir / "config.yaml"
            onnx_specs.append((triplet, ft, ckpt, cfg, ckpt_dir))

    for triplet, ft, ckpt, cfg, _ in onnx_specs:
        onnx = OUT / f"{triplet}_ft{ft:02d}.onnx"
        if onnx.exists():
            print(f"  [{triplet} ft{ft}] cached")
            continue
        if not ckpt.exists():
            print(f"  [{triplet} ft{ft}] CKPT MISSING: {ckpt}")
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

    # Step 3: dispatch 24 (triplet × ft × qvar) parallel AP eval
    specs = []
    for triplet, ft, ckpt, cfg, ckpt_dir in onnx_specs:
        for qvar, q_args in Q_VARIANTS:
            specs.append((triplet, ft, qvar, q_args, ckpt, cfg, ckpt_dir))

    print(f"[1.1] Step 3: {len(specs)} (triplet × FT × Q) AP eval "
          f"on {len(gpus)} GPU")

    with Pool(processes=len(gpus), initializer=_init, initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, specs))

    # Report
    print(f"\n{'tag':<36}{'AP50':>9}{'secs':>7}  status")
    by_tup = {}  # (triplet, ft) → {qvar: ap50}
    for tag, triplet, ft, qvar, ap50, secs, status in sorted(
            results, key=lambda r: r[0]):
        ap_str = f'{ap50:.3f}' if ap50 is not None else 'FAIL'
        print(f'{tag:<36}{ap_str:>9}{secs:>7.0f}  {status}')
        by_tup.setdefault((triplet, ft), {})[qvar] = ap50

    # CSV
    csv_path = OUT / "underfit.csv"
    with open(csv_path, "w") as f:
        cols = ["triplet", "ft"] + [q for q, _ in Q_VARIANTS]
        f.write(",".join(cols) + "\n")
        for triplet, _, _ in TRIPLETS:
            for ft in FT_LEVELS:
                row = [triplet, str(ft)]
                for q, _ in Q_VARIANTS:
                    v = by_tup.get((triplet, ft), {}).get(q)
                    row.append(f"{v:.4f}" if v is not None else "NA")
                f.write(",".join(row) + "\n")
    print(f"\nunderfit → {csv_path}")

    # Success-criteria summary
    aps = [ap for _, _, _, _, ap, _, _ in results if ap is not None]
    n_below_30 = sum(1 for a in aps if a < 0.30)
    n_below_10 = sum(1 for a in aps if a < 0.10)
    print(f"\n[1.1 SUCCESS CRITERIA]")
    print(f"  Total anchors with AP: {len(aps)}/{len(specs)}")
    print(f"  AP < 0.30: {n_below_30} (target ≥6 for strict path A)")
    print(f"  AP < 0.10: {n_below_10} (target ≥3 for random region anchor)")


if __name__ == "__main__":
    main()
