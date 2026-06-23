"""Phase 3a — B × FT (FP32 only) AP eval for g8 extreme triplets.

For each (triplet ∈ {T_g8_p87, T_g8_p93, T_g8_p97}, FT ∈ {0, 4, 8, 15}):
  1. Pick ckpt at epoch (baseline_g8_epoch + FT) — FT=0 is raw pruned
  2. ONNX export
  3. TRT FP32 build
  4. e2e_eval_ap n=1789
Total: 3 × 4 = 12 AP eval. 6 GPU parallel ~30 min.

Output: /tmp/a10_phase3a/{tag}_ap.json + spread_3a.csv
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
OUT = Path("/tmp/a10_phase3a")
OUT.mkdir(parents=True, exist_ok=True)

TRIPLETS = [
    ("T_g8_p87", "ft_T_g8_p87_raw"),
    ("T_g8_p93", "ft_T_g8_p93_raw"),
    ("T_g8_p97", "ft_T_g8_p97_raw"),
]
FT_LEVELS = [4, 8, 15]  # raw (FT=0) ckpt overwritten by HEAL save; AP=0 known

_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def run_eval(spec):
    triplet, ft_dir_name, ft, ft_cache_root, baseline_epoch = spec
    tag = f"{triplet}_ft{ft:02d}"
    ft_dir = Path(ft_cache_root) / ft_dir_name
    if ft == 0:
        ckpt = ft_dir / f"net_epoch_bestval_at{baseline_epoch}.pth"
    else:
        ckpt = ft_dir / f"net_epoch{baseline_epoch + ft}.pth"
    cfg = ft_dir / "config.yaml"
    onnx = OUT / f"{tag}.onnx"
    engine = OUT / f"{tag}_fp32.engine"
    build_rep = OUT / f"{tag}_fp32.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, ft, d.get("ap50") or d.get("ap_50"), 0, "cached")
    if not ckpt.exists():
        return (tag, ft, None, 0, f"ckpt missing: {ckpt.name}")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(_GPU),
           "PYTHONPATH": str(HEAL)}
    t0 = time.time()

    # 1. ONNX export
    if not onnx.exists():
        cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_e2e.py"),
               "--ckpt", str(ckpt), "--hypes", str(cfg),
               "--out", str(onnx), "--max-voxels", "32000"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if r.returncode != 0:
            (OUT / f"{tag}.onnx.err").write_text(r.stdout + r.stderr)
            return (tag, ft, None, time.time()-t0, "ONNX fail")

    # 2. TRT FP32 engine
    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx), "--precision", "fp32",
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if r.returncode != 0:
            (OUT / f"{tag}.engine.err").write_text(r.stdout + r.stderr)
            return (tag, ft, None, time.time()-t0, "engine fail")

    # 3. AP eval n=1789
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine), "--ckpt-dir", str(ft_dir),
           "--dair-root", DAIR, "--max-voxels", "32000",
           "--n-samples", "1789", "--tag", tag,
           "--report", str(ap_rep)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if r.returncode != 0:
        (OUT / f"{tag}.ap.err").write_text(r.stdout + r.stderr)
        return (tag, ft, None, time.time()-t0, "AP fail")

    d = json.loads(ap_rep.read_text())
    ap50 = d.get("ap50") or d.get("ap_50")
    return (tag, ft, ap50, time.time()-t0, "OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft-cache-root", default=str(REPO / "models/dataset_a_cache_g8"))
    p.add_argument("--baseline-epoch", type=int, required=True,
                   help="baseline_g8 epoch at start of finetune (e.g., 27)")
    p.add_argument("--gpus", default="2,3,4,5,6,7")
    args = p.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    specs = []
    for triplet, ft_dir_name in TRIPLETS:
        for ft in FT_LEVELS:
            specs.append((triplet, ft_dir_name, ft, args.ft_cache_root, args.baseline_epoch))
    print(f"[3a] {len(specs)} (triplet × FT) AP evals on {len(gpus)} GPU")

    with Pool(processes=len(gpus), initializer=_init, initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_eval, specs))

    # Pivot table
    print(f"\n{'tag':<18}{'FT':>4}{'AP50':>9}{'secs':>7}  status")
    by_tup = {}
    for tag, ft, ap50, secs, status in sorted(results, key=lambda r: (r[0], r[1])):
        ap_str = f'{ap50:.3f}' if ap50 is not None else 'FAIL'
        print(f'{tag:<18}{ft:>4}{ap_str:>9}{secs:>7.0f}  {status}')
        triplet = "_".join(tag.split("_")[:-1])
        by_tup.setdefault(triplet, {})[ft] = ap50

    # Save CSV
    csv_path = OUT / "spread_3a.csv"
    with open(csv_path, "w") as f:
        cols = ["triplet"] + [f"FT{ft}" for ft in FT_LEVELS]
        f.write(",".join(cols) + "\n")
        for triplet in [t[0] for t in TRIPLETS]:
            row = [triplet]
            for ft in FT_LEVELS:
                v = by_tup.get(triplet, {}).get(ft)
                row.append(f"{v:.4f}" if v is not None else "NA")
            f.write(",".join(row) + "\n")
    print(f"\nspread_3a → {csv_path}")


if __name__ == "__main__":
    main()
