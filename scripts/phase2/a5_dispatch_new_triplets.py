"""Track A 采集 — 13 new triplets × 7 Q × 12 D = 1092 anchor.

2-phase 设计避免 AP eval 重复 (D-invariance + csv 复用):
  Phase 1: 91 (T, Q, D1=default/4gb) anchor 跑 AP eval + 写 row.json
  Phase 2: merge row.json → e2e_bench_v1.csv (csv 含新 AP)
  Phase 3: 1001 (T, Q, D2..D12) anchor 跑 lat (AP 从 csv 复用)

总耗时估 ~10h on 6 GPU (vs 24h naive 全 AP eval).

用法:
  python a5_dispatch_new_triplets.py --phase 1       # AP eval batch (~2h)
  python a5_dispatch_new_triplets.py --merge         # 合并 phase 1 row.json → csv
  python a5_dispatch_new_triplets.py --phase 3       # lat-only batch (~8h)
  python a5_dispatch_new_triplets.py --resume        # 默认 --resume 跳过已存在
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
LOG_DIR = Path("/tmp/a5_parallel")
LOG_DIR.mkdir(exist_ok=True)
ROW_DIR = REPO / "results/a2_d_expand"

# 13 new triplets (T9 excluded)
NEW_TRIPLETS = [
    "T10_p11", "T11_p14", "T12_p21", "T13_p29", "T14_p36", "T15_p39",
    "T16_p54", "T17_p57", "T18_p64", "T19_p71", "T20_p79", "T21_p82", "T22_p89",
]
Q_LIST = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent",
          "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]
D_PHASE1 = ["D1_default_4gb"]  # 唯一 AP eval D
D_PHASE3 = ["D2_with_cudnn_8gb", "D3_cublas_lt_16gb", "D4_all_enabled_1gb",
            "D5_default_1gb", "D6_default_8gb", "D7_default_16gb",
            "D8_with_cudnn_4gb", "D9_cublas_lt_4gb", "D10_cublas_lt_8gb",
            "D11_all_enabled_4gb", "D12_edge_only_4gb"]


def run_anchor(spec):
    triplet, q_tag, d_tag, gpu = spec
    tag = f"{triplet}_{q_tag}_{d_tag}"
    log = LOG_DIR / f"{tag}.log"
    cmd = [PYTHON, str(REPO / "scripts/phase2/a2_run_one_anchor.py"),
           "--triplet", triplet, "--q-tag", q_tag, "--d-tag", d_tag,
           "--gpu", str(gpu)]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return (tag, r.returncode, time.time() - t0)


def dispatch_anchors(specs, n_gpu):
    t0 = time.time()
    n_ok = n_fail = 0
    with Pool(processes=n_gpu) as pool:
        for tag, rc, secs in pool.imap_unordered(run_anchor, specs):
            if rc == 0: n_ok += 1
            else: n_fail += 1
            done = n_ok + n_fail
            elapsed = time.time() - t0
            eta = elapsed / done * (len(specs) - done) if done else 0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[{done:4d}/{len(specs)}] {status} {tag} ({secs:.0f}s) "
                  f"| elapsed {elapsed/60:.1f}min eta {eta/60:.1f}min", flush=True)
    print(f"\n[a5] {n_ok} OK / {n_fail} FAIL, wall {time.time()-t0:.0f}s")


def _filter_triplets_with_onnx(triplets):
    """Drop triplets whose e2e ONNX doesn't exist yet (still training)."""
    onnx_dir = REPO / "models/e2e_cache"
    have = [t for t in triplets if (onnx_dir / f"{t}.onnx").exists()]
    miss = [t for t in triplets if t not in have]
    if miss:
        print(f"[skip] ONNX missing → defer: {miss}")
    return have


def cmd_phase1(args):
    """Run 13 × 7 = 91 (T, Q, D1) anchors → fresh AP eval."""
    triplets = NEW_TRIPLETS if not args.only_t else [args.only_t]
    triplets = _filter_triplets_with_onnx(triplets)
    anchors = [(t, q, d) for t in triplets for q in Q_LIST for d in D_PHASE1]
    print(f"[phase 1] {len(anchors)} anchor (D1 only, AP eval)")
    if args.resume:
        before = len(anchors)
        anchors = [(t, q, d) for (t, q, d) in anchors
                   if not (ROW_DIR / f"{t}_{q}_{d}.row.json").exists()]
        print(f"[phase 1] --resume 后剩余 {len(anchors)} (skipped {before - len(anchors)})")

    gpus = [int(x) for x in args.gpus.split(",")]
    specs = [(t, q, d, gpus[i % len(gpus)]) for i, (t, q, d) in enumerate(anchors)]
    dispatch_anchors(specs, len(gpus))


def cmd_phase3(args):
    """Run 13 × 7 × 11 = 1001 (T, Q, D2-D12) anchors → lat only, AP from csv."""
    triplets = NEW_TRIPLETS if not args.only_t else [args.only_t]
    triplets = _filter_triplets_with_onnx(triplets)
    anchors = [(t, q, d) for t in triplets for q in Q_LIST for d in D_PHASE3]
    print(f"[phase 3] {len(anchors)} anchor (D2-D12, AP reused from csv)")
    if args.resume:
        before = len(anchors)
        anchors = [(t, q, d) for (t, q, d) in anchors
                   if not (ROW_DIR / f"{t}_{q}_{d}.row.json").exists()]
        print(f"[phase 3] --resume 后剩余 {len(anchors)} (skipped {before - len(anchors)})")

    gpus = [int(x) for x in args.gpus.split(",")]
    specs = [(t, q, d, gpus[i % len(gpus)]) for i, (t, q, d) in enumerate(anchors)]
    dispatch_anchors(specs, len(gpus))


def cmd_merge(args):
    r = subprocess.run([PYTHON, str(REPO / "scripts/phase2/a2_merge_rows.py")])
    sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "3"], help="1=AP eval batch, 3=lat-only batch")
    ap.add_argument("--merge", action="store_true", help="merge row.json → csv")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--only-t", help="单 triplet 测试 (e.g. T10_p11)")
    args = ap.parse_args()

    if args.merge:
        cmd_merge(args)
    elif args.phase == "1":
        cmd_phase1(args)
    elif args.phase == "3":
        cmd_phase3(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
