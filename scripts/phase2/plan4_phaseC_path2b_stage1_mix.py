"""Plan 4 Phase C path 2b Stage 1 — 4 new mix Q variants × 24 triplet @ FT=8.

预注册触发后, 先扩 mix 4 个 (复用现成 ONNX Q/DQ + --mixed-int8-substr 工具):
  mix_s1   = stage 1 单独 INT8 (其他 FP16)
  mix_s01  = stage 0+1 INT8
  mix_s02  = stage 0+2 INT8
  mix_s12  = stage 1+2 INT8

24 triplet × 4 Q = 96 new anchor, FT=8 锁定, D=D1.

成功信号 (升级到 Stage 2 与否):
  合并 Phase A 119 + Stage 1 96 = 215 anchor, 重训 LGB v9_ap with GroupKFold
  若 R² ≥ 0.55 → 路径成功, 跳 Phase D
  若 R² < 0.55 但 mix Q 有 1+ 个 triplet 在某 mix 触发 partial collapse (AP<0.4) → 升级 Stage 2
  若 mix Q 全 plateau (无新 collapse) → 触发 path 4 reframe

Output:
  /tmp/plan4_phaseC_stage1/{anchor}.engine + ap.json
  /tmp/plan4_phaseC_stage1/stage1_mix.csv (24T × 4 mix)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool, current_process
from pathlib import Path

# 复用 Phase A dispatcher 的 TRIPLETS
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan4_phaseA_dispatcher import (
    TRIPLETS, REPO, HEAL, PYTHON, DAIR, CALIB_DIR, make_calib_args
)

OUT = Path("/tmp/plan4_phaseC_stage1")
OUT.mkdir(parents=True, exist_ok=True)

# 4 mix Q variants — Q_mix_s0/s2 在 Phase A 没跑 (Phase A 是 5 standard Q),
# 这里 mix_s0/s2 也加上, 总 6 个补 (24×6=144). 但 plan 说 Stage 1 加 4 个新.
# 严格按预注册: 加 4 个 NEW (s1, s01, s02, s12), 不重复 s0/s2.
MIX_Q_VARIANTS = [
    ("mix_s1", "layer1"),
    ("mix_s01", "layer0,layer1"),
    ("mix_s02", "layer0,layer2"),
    ("mix_s12", "layer1,layer2"),
]

_GPU = None


def _init(gpus):
    global _GPU
    wid = current_process()._identity[0]
    _GPU = gpus[(wid - 1) % len(gpus)]
    print(f"[worker {wid}] GPU {_GPU}", flush=True)


def run_one(spec):
    """spec = (trip_name, ckpt, planes, groups, prune_pct, q_name, mix_substr)"""
    trip_name, ckpt, planes, groups, prune, q_name, mix_substr = spec
    gpu = _GPU
    tag = f"{trip_name}_Q_{q_name}"

    # 复用 Phase A 的 ONNX (不需重新 export)
    onnx_src = Path("/tmp/plan4_phaseA") / f"{trip_name}.onnx"
    if not onnx_src.exists():
        return (tag, trip_name, q_name, None, 0, f"onnx missing: {onnx_src.name}")

    engine = OUT / f"{tag}.engine"
    cache = OUT / f"{tag}.cache"
    build_rep = OUT / f"{tag}.build.json"
    ap_rep = OUT / f"{tag}_ap.json"

    if ap_rep.exists():
        d = json.loads(ap_rep.read_text())
        return (tag, trip_name, q_name, d.get("ap50") or d.get("ap_50"),
                0, "cached")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL)}

    t0 = time.time()
    # Build engine
    if not engine.exists():
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(onnx_src), "--engine", str(engine),
               "--report", str(build_rep),
               "--workspace-mb", "4096", "--tactic", "default",
               "--builder-opt-level", "3", "--skip-bench",
               "--precision", "mixed",
               "--mixed-int8-substr", mix_substr,
               "--calibrator", "minmax",
               "--calib-cache", str(cache)] + make_calib_args()
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
    p.add_argument("--gpus", default="0,1,2,3,4,6,7")
    args = p.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    specs = []
    for trip in TRIPLETS:
        trip_name, ckpt, planes, groups, prune = trip
        for q_name, mix_substr in MIX_Q_VARIANTS:
            specs.append((trip_name, ckpt, planes, groups, prune,
                          q_name, mix_substr))
    print(f"[Phase C Stage 1] {len(TRIPLETS)} triplet × {len(MIX_Q_VARIANTS)} mix Q "
          f"= {len(specs)} anchor on {len(gpus)} GPU")

    t0 = time.time()
    with Pool(processes=len(gpus), initializer=_init,
              initargs=(gpus,)) as pool:
        results = list(pool.imap_unordered(run_one, specs))
    print(f"[Phase C Stage 1] dispatch wall: {(time.time()-t0)/60:.1f} min")

    # Write csv
    by_anchor = {}
    n_ok = n_fail = 0
    for tag, trip, q, ap, secs, status in sorted(results):
        by_anchor[(trip, q)] = (ap, status)
        ap_str = f'{ap:.4f}' if ap is not None else 'FAIL'
        print(f"  {tag:<36} AP={ap_str:>9}  {secs:>5.0f}s  {status}")
        if ap is not None:
            n_ok += 1
        else:
            n_fail += 1

    csv_path = OUT / "stage1_mix.csv"
    with open(csv_path, "w") as f:
        f.write("triplet,planes_s0,planes_s1,planes_s2,groups,prune_pct,q,ap50,ft,d,status\n")
        for trip in TRIPLETS:
            trip_name, ckpt, planes, groups, prune = trip
            for q_name, _ in MIX_Q_VARIANTS:
                ap, status = by_anchor.get((trip_name, q_name), (None, "missing"))
                row = [trip_name, str(planes[0]), str(planes[1]), str(planes[2]),
                       str(groups), f"{prune:.2f}", q_name,
                       f"{ap:.4f}" if ap is not None else "NA",
                       "8", "D1_default_4gb", status]
                f.write(",".join(row) + "\n")
    print(f"\n[Stage 1] csv → {csv_path}, OK={n_ok}/{len(specs)}, FAIL={n_fail}")

    # Decision check (升级 Stage 2 or 跳 Phase D)
    import numpy as np
    aps = np.array([v[0] for v in by_anchor.values() if v[0] is not None])
    print(f"\n[Stage 1 SUMMARY]")
    print(f"  Mix Q std: {aps.std():.4f}")
    print(f"  Mix Q range: [{aps.min():.4f}, {aps.max():.4f}]")
    n_collapse = (aps < 0.40).sum()
    n_degrade = ((aps >= 0.40) & (aps < 0.50)).sum()
    print(f"  Mix Q collapse (<0.40) count: {n_collapse}")
    print(f"  Mix Q degrade (0.40-0.50) count: {n_degrade}")

    # Decision: signal triggers if ≥1 mix triplet < 0.50 in non-trivial way
    signal_found = bool(n_collapse + n_degrade > 5)
    print(f"  signal_found (collapse+degrade > 5): {signal_found}")
    print(f"  decision: {'CONTINUE to LGB v9_ap fit + Phase D' if signal_found else 'ESCALATE to Stage 2 OR path 4 reframe'}")


if __name__ == "__main__":
    main()
