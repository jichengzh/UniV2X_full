"""P0.2.b — Class A 主网格 4090 补充: 4 NEW Q × 4 D + 6 OLD Q × 1 NEW D = 176 anchor.

把 P0.2 (8 × 6 Q × 3 D = 144) 补到 §1.1 计划 (8 × 10 Q × 4 D = 320), 缺 176:
  - 4 NEW Q × 4 D × 8 triplet = 128
  - 6 OLD Q × 1 NEW D (D_ws16) × 8 triplet = 48

NEW Q cells (4): 填 §2.2 plan Q9/Q10 之外的 mixed 排列 + FP32 baseline
    Q_fp32:    precision=fp32 (FP32 lat 上限 baseline)
    Q_mix_s1:  precision=int8, mixed_fp16_substr=\"stage_0,stage_2\" (只 s1 INT8)
    Q_mix_s12: precision=int8, mixed_fp16_substr=\"stage_0\" (s1+s2 INT8, s0 FP16)
    Q_mix_s02: precision=int8, mixed_fp16_substr=\"stage_1\" (s0+s2 INT8, s1 FP16)

NEW D cell (1):
    D_ws16: workspace_mb=16384 (16GB)

复用 P0.2 的 ONNX (models/dataset_a_main_grid/T{1-8}_*.onnx 已存在).
复用 calibration npy 同 P0.2.

合并产出: 把本 supplement 144→176 row append 到 P0.2 的 class_a_pyramid_full.parquet,
总 144+176=320 完整 Class A 4090 anchor.
"""
from __future__ import annotations
import json, os, subprocess, time, multiprocessing as mp
from pathlib import Path
import pandas as pd

# Reuse all constants from P0.2 main_grid
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_a_main_grid_4090 import (
    REPO_ROOT, PYTHON, HEAL_ROOT,
    CACHE, LOG_DIR, OUT_DIR,
    CALIB_SPATIAL, CALIB_TEGO,
    CKPTS,
    export_onnx, build_and_bench, ap_eval, process_anchor,
)

# 4 NEW Q cells (filling Q1/Q2 perm + FP32 baseline)
NEW_Q_CELLS = [
    # (label, precision, mixed_substr, calibrator)
    ("Q_fp32",     "fp32",  "",                       "entropy"),  # FP32 baseline (calib unused)
    ("Q_mix_s1",   "int8",  "stage_0,stage_2",        "entropy"),  # 只 s1 INT8
    ("Q_mix_s12",  "int8",  "stage_0",                "entropy"),  # s1+s2 INT8, s0 FP16
    ("Q_mix_s02",  "int8",  "stage_1",                "entropy"),  # s0+s2 INT8, s1 FP16
]

# 6 OLD Q cells (re-imported from P0.2 to extend D)
OLD_Q_CELLS = [
    ("Q_fp16",     "fp16",  "",                       "entropy"),
    ("Q_int8_ent", "int8",  "",                       "entropy"),
    ("Q_int8_mm",  "int8",  "",                       "minmax"),
    ("Q_mix_s0",   "int8",  "stage_1,stage_2",        "entropy"),
    ("Q_mix_s2",   "int8",  "stage_0,stage_1",        "entropy"),
    ("Q_mix_s01",  "int8",  "stage_2",                "entropy"),
]

# D cells: NEW Q × 4 D vs OLD Q × 1 D
ALL_D_CELLS = [
    ("D_ws1",  1024),
    ("D_ws4",  4096),
    ("D_ws8",  8192),
    ("D_ws16", 16384),
]
NEW_D_CELLS = [("D_ws16", 16384)]  # only the new ws=16GB for OLD Q


def build_task_list() -> list:
    """176 anchor:
       - 4 NEW Q × 4 D × 8 triplet = 128
       - 6 OLD Q × 1 NEW D × 8 triplet = 48
    Distributed across GPU 0-6 (skip GPU 7 due to external occupant)."""
    tasks = []
    gpu_ids = [0, 1, 2, 3, 4, 5, 6]  # 7 GPUs, skip 7
    idx = 0
    for tag, planes, ckpt, hypes in CKPTS:
        # 4 NEW Q × 4 D = 16 anchor per triplet
        for q_label, prec, mixed, calib in NEW_Q_CELLS:
            for d_label, ws_mb in ALL_D_CELLS:
                gpu = gpu_ids[idx % len(gpu_ids)]
                tasks.append((tag, planes, ckpt, hypes,
                              q_label, prec, mixed, calib,
                              d_label, ws_mb, gpu))
                idx += 1
        # 6 OLD Q × 1 NEW D = 6 anchor per triplet
        for q_label, prec, mixed, calib in OLD_Q_CELLS:
            for d_label, ws_mb in NEW_D_CELLS:
                gpu = gpu_ids[idx % len(gpu_ids)]
                tasks.append((tag, planes, ckpt, hypes,
                              q_label, prec, mixed, calib,
                              d_label, ws_mb, gpu))
                idx += 1
    return tasks


def main():
    print("=" * 78)
    print("P0.2.b Dataset Class A Supplement (4090): 4 NEW Q × 4 D + 6 OLD Q × 1 NEW D")
    print("                                          = 8 × (16 + 6) = 176 anchor")
    print("=" * 78)
    tasks = build_task_list()
    print(f"Total: {len(tasks)} anchor, distributed across 7 GPUs (0-6, skip 7)")
    print(f"Expected wall: ~3-4h (P0.2 ran 144 anchor in 60.7 min on 8 GPU)")
    print()

    t0 = time.time()
    with mp.Pool(processes=7) as pool:
        results = pool.imap_unordered(process_anchor, tasks)
        rows = []
        for i, r in enumerate(results, 1):
            rows.append(r)
            if i % 10 == 0:
                ok_cnt = sum(1 for x in rows if x.get("status") == "OK")
                print(f"\n[progress] {i}/{len(tasks)} anchors done "
                      f"(OK={ok_cnt}, wall={(time.time()-t0)/60:.1f} min)\n")

    elapsed = (time.time() - t0) / 60
    print(f"\n[done] {len(rows)} anchors in {elapsed:.1f} min")

    df_new = pd.DataFrame([r for r in rows if r.get("status") == "OK"])
    failed = [r for r in rows if r.get("status") != "OK"]

    # Append to existing P0.2 parquet (144 rows already there)
    out = REPO_ROOT / "data/_by_class/class_a_pyramid_full.parquet"
    if out.exists():
        df_old = pd.read_parquet(out)
        # Drop dup by anchor_id (in case of restart)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["anchor_id"], keep="last")
        print(f"\nMerged: old {len(df_old)} + new {len(df_new)} → "
              f"combined {len(df_combined)} unique anchor")
    else:
        df_combined = df_new
        print(f"\n[warn] no existing parquet, saving fresh {len(df_combined)} rows")

    df_combined.to_parquet(out)
    df_combined.to_csv(out.with_suffix(".csv"), index=False)
    print(f"✅ saved {len(df_combined)} total anchors → {out}")
    print(f"   {len(failed)} failed in this batch")
    if failed:
        fail_path = REPO_ROOT / "results/dataset_a_main_grid_supplement_failed.json"
        fail_path.write_text(json.dumps(failed, indent=2))
        print(f"   failures → {fail_path}")

    if not df_combined.empty and "ap50" in df_combined.columns:
        print("\n=== AP50 真实分布 (合并后 320 anchor) ===")
        print(f"  min:    {df_combined['ap50'].min():.4f}")
        print(f"  max:    {df_combined['ap50'].max():.4f}")
        print(f"  spread: {(df_combined['ap50'].max() - df_combined['ap50'].min())*100:.2f} pp")
        print(f"  std:    {df_combined['ap50'].std()*100:.2f} pp")


if __name__ == "__main__":
    main()
