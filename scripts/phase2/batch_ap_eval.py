"""Batch real-AP eval for all unique (B, Q) anchors.

Iterates over:
    1. 100 random_bench anchors (50 triplet × 2 prec) — use prune_{sig}/ ckpt
    2. 18 per-stage anchors — use baseline ckpt + per-stage mixed engine

For each anchor: call m4_8_hybrid_infer_ap.py via subprocess on full DAIR val (1789).
Records AP30/50/70 + cache JSON, append to data/pyramid_ap_real.parquet.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
CACHE_ROOT = REPO_ROOT / "models/p0_random_cache"
PERSTAGE_CACHE = REPO_ROOT / "models/perstage_quant_cache"
BASELINE_CKPT = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
OUT_DIR = REPO_ROOT / "results/batch_ap_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_pruned_ckpt_dir(sig: str) -> str:
    """Return ckpt dir for given triplet signature."""
    prune_dir = CACHE_ROOT / f"prune_{sig}"
    if (prune_dir / "config.yaml").exists():
        return str(prune_dir)
    return BASELINE_CKPT  # fallback to baseline (only for 064_128_256)


def run_ap_eval(engine_path: str, tag: str, ckpt_dir: str, n_samples: int) -> dict | None:
    """Run m4_8_hybrid_infer_ap.py via subprocess. Return parsed report."""
    report_path = OUT_DIR / f"{tag}.json"
    if report_path.exists():
        return json.loads(report_path.read_text())

    cmd = [
        PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
        "--engine-collab", engine_path,
        "--tag", tag,
        "--model-dir", ckpt_dir,
        "--n-samples", str(n_samples),
        "--dataset", "dair",
        "--range", "102.4,51.2",
        "--collab-spatial-shape", "2,64,128,256",
        "--collab-tego-shape", "2,2,3",
        "--report", str(report_path),
    ]
    env = {"CUDA_VISIBLE_DEVICES": "0", "PATH": os.environ.get("PATH", "")}
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env,
                           capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        print(f"  {tag}: TIMEOUT after {time.time()-t0:.0f}s")
        return None
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  {tag}: FAILED ({elapsed:.0f}s)")
        print(f"    stderr tail: {r.stderr[-500:]}")
        return None
    if not report_path.exists():
        print(f"  {tag}: no report file (elapsed {elapsed:.0f}s)")
        return None
    rep = json.loads(report_path.read_text())
    print(f"  {tag}: AP30={rep['ap30']:.4f} AP50={rep['ap50']:.4f} AP70={rep['ap70']:.4f} ({elapsed:.0f}s)")
    return rep


def collect_random_bench(n_samples: int):
    """100 anchor: triplet × prec, use engine_{sig}_{prec}.engine."""
    df = pd.read_parquet(REPO_ROOT / "data/pyramid_random_bench.parquet")
    unique = df[['stage0_planes','stage1_planes','stage2_planes','precision']].drop_duplicates()
    print(f"[random_bench] {len(unique)} unique (B,Q) anchors")
    rows = []
    for i, row in enumerate(unique.itertuples(index=False), 1):
        s0, s1, s2, prec = row
        sig = f"{s0:03d}_{s1:03d}_{s2:03d}"
        engine = CACHE_ROOT / f"engine_{sig}_{prec}.engine"
        if not engine.exists():
            print(f"  [{i}/{len(unique)}] MISSING engine: {engine.name}")
            continue
        ckpt_dir = get_pruned_ckpt_dir(sig)
        tag = f"rb_{sig}_{prec}"
        print(f"  [{i}/{len(unique)}] {tag}")
        rep = run_ap_eval(str(engine), tag, ckpt_dir, n_samples)
        if rep is None: continue
        rows.append({
            "source": "random_bench",
            "stage0_planes": s0, "stage1_planes": s1, "stage2_planes": s2,
            "precision": prec, "config_label": tag,
            "n_samples": rep["n_samples"],
            "n_trt_path": rep["n_trt_path"],
            "n_pytorch_fallback": rep["n_pytorch_fallback"],
            "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
            "elapsed_secs": rep["elapsed_secs"],
        })
    return rows


def collect_perstage(n_samples: int):
    """18 per-stage anchor: 3 triplet × 6 config, use perstage engine."""
    df = pd.read_parquet(REPO_ROOT / "data/perstage_quant_bench.parquet")
    print(f"[perstage] {len(df)} anchors")
    rows = []
    for i, row in enumerate(df.itertuples(index=False), 1):
        sig = row.triplet_sig
        cfg = row.config_label
        engine = PERSTAGE_CACHE / f"engine_{sig}_{cfg}.engine"
        if not engine.exists():
            print(f"  [{i}/{len(df)}] MISSING: {engine.name}")
            continue
        ckpt_dir = get_pruned_ckpt_dir(sig)
        tag = f"ps_{sig}_{cfg}"
        print(f"  [{i}/{len(df)}] {tag}")
        rep = run_ap_eval(str(engine), tag, ckpt_dir, n_samples)
        if rep is None: continue
        rows.append({
            "source": "perstage",
            "stage0_planes": row.stage0_planes, "stage1_planes": row.stage1_planes, "stage2_planes": row.stage2_planes,
            "config_label": tag, "perstage": cfg,
            "stage0_prec": row.stage0_prec, "stage1_prec": row.stage1_prec, "stage2_prec": row.stage2_prec,
            "precision": "mixed",
            "n_samples": rep["n_samples"],
            "n_trt_path": rep["n_trt_path"],
            "n_pytorch_fallback": rep["n_pytorch_fallback"],
            "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
            "elapsed_secs": rep["elapsed_secs"],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=1789,
                   help="DAIR val size (1789 full). Smaller for smoke test.")
    p.add_argument("--scope", choices=["all", "random_bench", "perstage"], default="all")
    p.add_argument("--output", default="data/pyramid_ap_real.parquet")
    args = p.parse_args()

    t0 = time.time()
    rows = []
    if args.scope in ("all", "random_bench"):
        rows.extend(collect_random_bench(args.n_samples))
    if args.scope in ("all", "perstage"):
        rows.extend(collect_perstage(args.n_samples))

    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    elapsed = time.time() - t0
    print(f"\n[done] {len(df)} rows in {elapsed/60:.1f} min -> {out}")
    print(df[["config_label","precision","ap30","ap50","ap70"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
