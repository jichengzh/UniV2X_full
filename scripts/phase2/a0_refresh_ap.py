"""A.0 baseline AP refresh: 32 anchor (8T × 4 baseline Q) e2e_eval_ap.py n=1789.

修复 问题 5 — csv A.0 AP 列来自 dataset_a_main subnet 管线, 应改用 e2e single-engine n=1789.

复用 models/e2e_cache/ 既有 engine (不重 build). 仅跑 e2e_eval_ap.py + 回填 csv.

Usage:
  python a0_refresh_ap.py --dispatch          # 32 anchor 6-GPU 并行
  python a0_refresh_ap.py --backfill          # 跑完后回填 csv (A.0 + A.2 inherited)
  python a0_refresh_ap.py --one T1_base Q_fp16 --gpu 0   # 单个 anchor (smoke)
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
ENG_DIR = REPO / "models/e2e_cache"
AP_DIR = REPO / "results/a0_refresh_ap"
LOG_DIR = Path("/tmp/a0_refresh")
CSV = REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv"
PQ = REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.parquet"
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
MAX_VOX = 32000

AP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

TRIPLET_MAP = {
    "T1_base": "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",
    "T2_p25":  "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10",
    "T3_p37":  str(REPO / "models/dataset_a_cache/ft_040_080_160"),
    "T4_p50":  "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",
    "T5_p62":  str(REPO / "models/dataset_a_cache/ft_024_056_128"),
    "T6_p75":  "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10"
,
    "T7_wide_shallow": str(REPO / "models/dataset_a_cache/ft_048_064_128"),
    "T8_narrow_deep":  str(REPO / "models/dataset_a_cache/ft_024_048_192"),
}
Q_BASELINE = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent"]


def eval_one(spec):
    triplet, q_tag, gpu = spec
    tag = f"{triplet}_{q_tag}"
    engine = ENG_DIR / f"{tag}.engine"
    ap_rep = AP_DIR / f"{tag}_ap.json"
    log = LOG_DIR / f"{tag}.log"

    if not engine.exists():
        return (tag, "NO_ENGINE", 0)
    if ap_rep.exists():
        return (tag, "CACHED", 0)

    ckpt_dir = TRIPLET_MAP[triplet]
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine),
           "--ckpt-dir", ckpt_dir,
           "--dair-root", DAIR,
           "--max-voxels", str(MAX_VOX),
           "--n-samples", "1789",
           "--tag", f"{tag}_e2e_refresh",
           "--report", str(ap_rep)]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return (tag, "OK" if r.returncode == 0 else f"FAIL(rc={r.returncode})", time.time() - t0)


def cmd_dispatch(gpus):
    anchors = [(t, q) for t in TRIPLET_MAP for q in Q_BASELINE]
    n_gpu = len(gpus)
    specs = [(t, q, gpus[i % n_gpu]) for i, (t, q) in enumerate(anchors)]
    print(f"[dispatch] {len(specs)} anchors on {n_gpu} GPUs ({gpus})")
    t0 = time.time()
    n_ok = n_fail = n_cached = 0
    with Pool(processes=n_gpu) as pool:
        for tag, status, secs in pool.imap_unordered(eval_one, specs):
            if status == "OK": n_ok += 1
            elif status == "CACHED": n_cached += 1
            else: n_fail += 1
            done = n_ok + n_fail + n_cached
            print(f"[{done:2d}/{len(specs)}] {status} {tag} ({secs:.0f}s)", flush=True)
    print(f"\n[dispatch] {n_ok} OK, {n_cached} CACHED, {n_fail} FAIL, "
          f"wall {time.time()-t0:.0f}s")


def cmd_backfill():
    """Update csv: 32 A.0 rows from refresh AP, then 96 A.2 inherited rows."""
    import pandas as pd
    df = pd.read_csv(CSV)
    print(f"[backfill] csv: {len(df)} rows")

    # 加载 32 refresh AP
    refresh_ap = {}
    for jf in AP_DIR.glob("*_ap.json"):
        tag = jf.stem.replace("_ap", "")
        # tag = "T1_base_Q_fp16"
        for t in TRIPLET_MAP:
            if tag.startswith(t + "_"):
                q = tag[len(t)+1:]
                d = json.loads(jf.read_text())
                refresh_ap[(t, q)] = {
                    "ap30": d.get("ap_30") or d.get("ap30"),
                    "ap50": d.get("ap_50") or d.get("ap50"),
                    "ap70": d.get("ap_70") or d.get("ap70"),
                }
                break
    print(f"[backfill] refresh AP files loaded: {len(refresh_ap)}")

    # 修 A.0 行 (d_tactic=default, ws=4) 和 A.2 继承行 (d_tactic in D2/3/4) for baseline Q
    n_updated = 0
    for (t, q), ap in refresh_ap.items():
        mask = (df["triplet"] == t) & (df["q_tag"] == q) & (df["q_tag"].isin(Q_BASELINE))
        n = mask.sum()
        df.loc[mask, "ap30"] = ap["ap30"]
        df.loc[mask, "ap50"] = ap["ap50"]
        df.loc[mask, "ap70"] = ap["ap70"]
        n_updated += n
        print(f"  {t}/{q}: updated {n} rows, ap50={ap['ap50']:.4f}")

    print(f"[backfill] total rows updated: {n_updated} (expect 32 A.0 + 96 A.2 = 128)")
    df.to_csv(CSV, index=False)
    df.to_parquet(PQ, index=False)
    print(f"[backfill] saved -> {CSV.name}, {PQ.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--one", nargs=2, metavar=("TRIPLET", "Q"))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.one:
        t, q = args.one
        tag, status, secs = eval_one((t, q, args.gpu))
        print(f"{status} {tag} ({secs:.0f}s)")
    elif args.dispatch:
        gpus = [int(x) for x in args.gpus.split(",")]
        cmd_dispatch(gpus)
    elif args.backfill:
        cmd_backfill()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
