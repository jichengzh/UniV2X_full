"""Track C / Method 3 — D 维度扩档采集.

现有 D-config (224 anchor 已覆盖):
  D1 default/4GB, D2 with_cudnn/8GB, D3 cublas_lt/16GB, D4 all_enabled/1GB

新增 8 个 D-config (本脚本采集):
  D5  default/1GB        — small ws default baseline
  D6  default/8GB        — medium ws default
  D7  default/16GB       — large ws default
  D8  with_cudnn/4GB     — default-ws with_cudnn
  D9  cublas_lt/4GB      — default-ws cublas_lt
  D10 cublas_lt/8GB      — medium ws cublas_lt
  D11 all_enabled/4GB    — default-ws all_enabled
  D12 edge_only/4GB      — new tactic (B class prep)

8 T × 7 Q × 8 NEW D = 448 anchor. AP 全复用 csv (D 数学等价).

复用 scripts/phase2/a2_run_one_anchor.py — 只需新加 D_MAP entry.

用法:
  python a3_dspace_extend.py --dispatch          # 全 448 anchor
  python a3_dspace_extend.py --smoke             # 8 anchor (1/T, Q=fp16, D=D5)
  python a3_dspace_extend.py --resume            # 跳过已存在 row.json
  python a3_dspace_extend.py --merge             # row JSON → e2e_bench_v1.csv
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
LOG_DIR = Path("/tmp/a3_parallel")
LOG_DIR.mkdir(exist_ok=True)
ROW_DIR = REPO / "results/a2_d_expand"   # 同 a2 目录 (统一 D-extension row)

TRIPLETS = ["T1_base", "T2_p25", "T3_p37", "T4_p50",
            "T5_p62", "T6_p75", "T7_wide_shallow", "T8_narrow_deep"]
Q_LIST = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent",
          "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]

# 新增 8 个 D-config (扩档 + 新 tactic edge_only)
NEW_D_MAP = {
    "D5_default_1gb":      ("default",     1024),
    "D6_default_8gb":      ("default",     8192),
    "D7_default_16gb":     ("default",     16384),
    "D8_with_cudnn_4gb":   ("with_cudnn",  4096),
    "D9_cublas_lt_4gb":    ("cublas_lt",   4096),
    "D10_cublas_lt_8gb":   ("cublas_lt",   8192),
    "D11_all_enabled_4gb": ("all_enabled", 4096),
    "D12_edge_only_4gb":   ("edge_only",   4096),
}


def run_anchor(spec):
    """复用 a2_run_one_anchor.py — 已支持 任意 d_tag + tactic + workspace."""
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


def patch_a2_d_map():
    """Inject NEW_D_MAP into a2_run_one_anchor.py 的 D_MAP (运行时方式).

    更优雅: 改 a2_run_one_anchor.py 让它接受 --d-tactic + --d-workspace-mb
    参数. 但为了不动 a2 主脚本, 改用 sys.path hack: a3 启动前先 import + 注入.
    实际本脚本通过 subprocess 调 a2, 所以必须**永久** patch a2 的 D_MAP.

    简单做法: 直接编辑 a2_run_one_anchor.py 加 D5-D12 (一次性).
    """
    a2_path = REPO / "scripts/phase2/a2_run_one_anchor.py"
    code = a2_path.read_text()
    if "D5_default_1gb" in code:
        print("[patch] a2_run_one_anchor.py already has D5-D12")
        return
    # Insert after D4 line
    marker = '    "D4_all_enabled_1gb": ("all_enabled", 1024),'
    insertion = (marker + "\n"
                 + '    "D5_default_1gb":      ("default",     1024),\n'
                 + '    "D6_default_8gb":      ("default",     8192),\n'
                 + '    "D7_default_16gb":     ("default",     16384),\n'
                 + '    "D8_with_cudnn_4gb":   ("with_cudnn",  4096),\n'
                 + '    "D9_cublas_lt_4gb":    ("cublas_lt",   4096),\n'
                 + '    "D10_cublas_lt_8gb":   ("cublas_lt",   8192),\n'
                 + '    "D11_all_enabled_4gb": ("all_enabled", 4096),\n'
                 + '    "D12_edge_only_4gb":   ("edge_only",   4096),')
    new_code = code.replace(marker, insertion)
    a2_path.write_text(new_code)
    print(f"[patch] a2_run_one_anchor.py D_MAP +8 entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    args = ap.parse_args()

    if args.merge:
        # 复用 a2_merge_rows.py
        r = subprocess.run([PYTHON, str(REPO / "scripts/phase2/a2_merge_rows.py")])
        sys.exit(r.returncode)

    # 1. patch a2 D_MAP (一次性)
    patch_a2_d_map()

    # 2. 构造 anchor list
    if args.smoke:
        anchors = [(t, "Q_fp16", "D5_default_1gb") for t in TRIPLETS]
    else:
        anchors = [(t, q, d) for t in TRIPLETS for q in Q_LIST for d in NEW_D_MAP]
    print(f"[a3] {len(anchors)} anchor 待跑")

    # 3. resume
    if args.resume:
        before = len(anchors)
        anchors = [(t, q, d) for (t, q, d) in anchors
                   if not (ROW_DIR / f"{t}_{q}_{d}.row.json").exists()]
        print(f"[a3] --resume 后剩余 {len(anchors)} (跳过 {before - len(anchors)})")

    gpus = [int(x) for x in args.gpus.split(",")]
    n_gpu = len(gpus)
    specs = [(t, q, d, gpus[i % n_gpu]) for i, (t, q, d) in enumerate(anchors)]

    # 4. parallel run
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
            print(f"[{done:3d}/{len(specs)}] {status} {tag} ({secs:.0f}s) "
                  f"| elapsed {elapsed/60:.1f}min eta {eta/60:.1f}min", flush=True)
    print(f"\n[a3] done: {n_ok} OK, {n_fail} FAIL, wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
