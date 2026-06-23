"""A.2 parallel dispatcher: 168 anchors (8T × 7Q × 3D) on 6 GPUs.

每个 GPU 一个队列, 串行处理 28 个 anchor (build + bench only,
AP 复用 A.0/A.1 cache). 预计单 anchor ~3 min → 6 GPU 并行 ~84 min wall.

用法:
  python a2_dispatch_parallel.py                # 全 168 anchor
  python a2_dispatch_parallel.py --smoke        # 仅 8 anchor (1/T × Q=fp16 × D2)
  python a2_dispatch_parallel.py --gpus 0,1,2   # 限定 GPU 子集
  python a2_dispatch_parallel.py --resume       # 跳过已存在 row.json 的 anchor
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
LOG_DIR = Path("/tmp/a2_parallel")
LOG_DIR.mkdir(exist_ok=True)
ROW_DIR = REPO / "results/a2_d_expand"

TRIPLETS = ["T1_base", "T2_p25", "T3_p37", "T4_p50",
            "T5_p62", "T6_p75", "T7_wide_shallow", "T8_narrow_deep"]
Q_LIST = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent",
          "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]
D_LIST = ["D2_with_cudnn_8gb", "D3_cublas_lt_16gb", "D4_all_enabled_1gb"]


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
    secs = time.time() - t0
    return (tag, r.returncode, secs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="8 anchor 烟雾测试 (Q_fp16 × D2 only)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5",
                    help="可用 GPU 列表 (逗号分隔)")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已存在 row.json 的 anchor")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    n_gpu = len(gpus)
    print(f"[dispatch] using GPUs: {gpus} ({n_gpu} workers)")

    # 构造 anchor 列表
    if args.smoke:
        anchors = [(t, "Q_fp16", "D2_with_cudnn_8gb") for t in TRIPLETS]
    else:
        anchors = [(t, q, d) for t in TRIPLETS for q in Q_LIST for d in D_LIST]
    print(f"[dispatch] total anchors: {len(anchors)}")

    # Resume: 过滤已完成
    if args.resume:
        before = len(anchors)
        anchors = [(t, q, d) for (t, q, d) in anchors
                   if not (ROW_DIR / f"{t}_{q}_{d}.row.json").exists()]
        print(f"[dispatch] after --resume filter: {len(anchors)} (skipped {before - len(anchors)})")

    # round-robin GPU 分配
    specs = [(t, q, d, gpus[i % n_gpu]) for i, (t, q, d) in enumerate(anchors)]

    # 每 GPU 1 worker (串行处理自己队列). multiprocessing.Pool 处理 round-robin
    t_start = time.time()
    n_ok = 0; n_fail = 0
    with Pool(processes=n_gpu) as pool:
        for tag, rc, secs in pool.imap_unordered(run_anchor, specs):
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            n_ok += (rc == 0); n_fail += (rc != 0)
            elapsed = time.time() - t_start
            done = n_ok + n_fail
            eta = elapsed / done * (len(specs) - done) if done else 0
            print(f"[{done:3d}/{len(specs)}] {status} {tag} ({secs:.0f}s) | "
                  f"elapsed {elapsed/60:.1f}min eta {eta/60:.1f}min", flush=True)

    print(f"\n[dispatch] done: {n_ok} OK, {n_fail} FAIL, "
          f"wall {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
