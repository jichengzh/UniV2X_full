"""Stage 2e dispatcher — D 维度精细化 (步骤 1: 补完 5×4 网格; 步骤 2: 加 BL).

复用 a2_run_one_anchor.py (已扩 D_MAP 到 32, 含 --builder-opt-level).
全部 lat-only (AP 从 csv 复用), 但 BL=5 build 比 BL=3 慢 ~5×.

总 anchor: 1176 (step 1) + 1764 (step 2) = 2940
预算: ~20h on 6 GPU (BL=5 是瓶颈)

用法:
  python a6_dispatch_d_fill.py --step 1               # 仅步骤 1 (5h)
  python a6_dispatch_d_fill.py --step 2               # 仅步骤 2 (15h)
  python a6_dispatch_d_fill.py --step all             # 步骤 1+2 (20h)
  python a6_dispatch_d_fill.py --step all --only-t T22_p89   # 单 triplet 测试
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from multiprocessing import Pool, current_process
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
LOG_DIR = Path("/tmp/a6_parallel")
LOG_DIR.mkdir(exist_ok=True)
ROW_DIR = REPO / "results/a2_d_expand"

# 21 triplet (Track A v2 全部完成)
ALL_TRIPLETS = [
    "T1_base", "T2_p25", "T3_p37", "T4_p50", "T5_p62", "T6_p75",
    "T7_wide_shallow", "T8_narrow_deep",
    "T10_p11", "T11_p14", "T12_p21", "T13_p29", "T14_p36", "T15_p39",
    "T16_p54", "T17_p57", "T18_p64", "T19_p71", "T20_p79", "T21_p82", "T22_p89",
]
Q_LIST = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent",
          "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]

# 步骤 1: 8 新 D-config 补完 5×4 网格 (BL=3)
STEP1_D = ["D13_with_cudnn_1gb", "D14_with_cudnn_16gb",
           "D15_cublas_lt_1gb",
           "D16_all_enabled_8gb", "D17_all_enabled_16gb",
           "D18_edge_only_1gb", "D19_edge_only_8gb", "D20_edge_only_16gb"]

# 步骤 2: 4 代表 D × 3 BL = 12 新 D-config
STEP2_D = ["D21_default_4gb_BL0", "D22_cublas_lt_8gb_BL0",
           "D23_all_enabled_4gb_BL0", "D24_edge_only_4gb_BL0",
           "D25_default_4gb_BL3", "D26_cublas_lt_8gb_BL3",
           "D27_all_enabled_4gb_BL3", "D28_edge_only_4gb_BL3",
           "D29_default_4gb_BL5", "D30_cublas_lt_8gb_BL5",
           "D31_all_enabled_4gb_BL5", "D32_edge_only_4gb_BL5"]


_WORKER_GPU = None  # 每 worker initializer 设这个 (worker-bound GPU)


def _init_worker(gpus_avail):
    """每个 Pool worker 启动时调用一次, 绑定本 worker 到固定 GPU.

    worker._identity[0] 是 1-based worker index (1..n_gpu). 用它选 GPU.
    这样每个 worker 在整个 dispatch 过程中只用一个 GPU, 避免多 worker 撞同 GPU.
    """
    global _WORKER_GPU
    wid = current_process()._identity[0]  # 1..n_gpu
    _WORKER_GPU = gpus_avail[(wid - 1) % len(gpus_avail)]
    print(f"[worker {wid} pid {os.getpid()}] bound to GPU {_WORKER_GPU}", flush=True)


def run_anchor(spec):
    triplet, q_tag, d_tag = spec
    gpu = _WORKER_GPU
    tag = f"{triplet}_{q_tag}_{d_tag}"
    log = LOG_DIR / f"{tag}.log"
    cmd = [PYTHON, str(REPO / "scripts/phase2/a2_run_one_anchor.py"),
           "--triplet", triplet, "--q-tag", q_tag, "--d-tag", d_tag,
           "--gpu", str(gpu)]
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return (tag, gpu, r.returncode, time.time() - t0)


def dispatch_anchors(specs, gpus):
    t0 = time.time()
    n_ok = n_fail = 0
    # 长任务 (BL=5) 先 dispatch, 让 BL=0/3 短任务后期填补
    specs_sorted = sorted(specs, key=lambda s: ("BL5" not in s[2],
                                                  "BL3" not in s[2],
                                                  "BL0" not in s[2]))
    with Pool(processes=len(gpus), initializer=_init_worker,
              initargs=(gpus,)) as pool:
        for tag, gpu, rc, secs in pool.imap_unordered(run_anchor, specs_sorted):
            if rc == 0: n_ok += 1
            else: n_fail += 1
            done = n_ok + n_fail
            elapsed = time.time() - t0
            eta = elapsed / done * (len(specs) - done) if done else 0
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[{done:4d}/{len(specs)}] {status} g{gpu} {tag} ({secs:.0f}s) "
                  f"| elapsed {elapsed/60:.1f}min eta {eta/60:.1f}min", flush=True)
    print(f"\n[a6] {n_ok} OK / {n_fail} FAIL, wall {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["1", "2", "all"], default="all")
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--only-t", help="单 triplet 测试")
    args = ap.parse_args()

    triplets = ALL_TRIPLETS if not args.only_t else [args.only_t]
    d_list = []
    if args.step in ("1", "all"):
        d_list += STEP1_D
    if args.step in ("2", "all"):
        d_list += STEP2_D

    gpus = [int(x) for x in args.gpus.split(",")]
    raw = [(t, q, d) for t in triplets for q in Q_LIST for d in d_list]
    print(f"[step {args.step}] {len(raw)} anchor candidates "
          f"({len(triplets)} T × {len(Q_LIST)} Q × {len(d_list)} D)")

    if args.resume:
        before = len(raw)
        raw = [(t, q, d) for (t, q, d) in raw
               if not (ROW_DIR / f"{t}_{q}_{d}.row.json").exists()]
        print(f"--resume 后剩余 {len(raw)} (skipped {before - len(raw)})")

    if not raw:
        print("nothing to do")
        return

    # GPU 不再随 spec 携带 — 由 Pool initializer 给 worker 绑定固定 GPU
    dispatch_anchors(raw, gpus)


if __name__ == "__main__":
    main()
