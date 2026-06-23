"""A.1 phase 1: Q_mix_s0 + Q_mix_s2 on 8 triplets → e2e_bench_v1 row schema.

工程修正 (vs class_a 旧脚本):
- e2e ONNX 层命名是 `/resnet/layerX/...` 与 `/backbone_m1/resnet/layerX/...`, 不是 stage_X
- m4_8_trt_build_bench.py 仅当 --precision mixed 时才执行 per-layer 设置 (precision=int8 时
  --mixed-fp16-substr 是死参数, 旧 class_a 那 6 个 Q_mix_* 实际是 entropy-INT8 重命名)
- 校准器必须 minmax (entropy 已禁用, 参见 framework/constraints.py)

每个 (T, Q) anchor 输出 3 件:
  1. engine: models/e2e_cache_a1/{T}_{Q}.engine
  2. lat bench: models/e2e_cache_a1/{T}_{Q}_bench.json
  3. AP eval (DAIR 1789 sample): results/a1_q_expand/{T}_{Q}_ap.json

最后聚合 16 行 → paper_learning/2. AAAI最终故事/data/e2e_bench_v1.{csv,parquet} 追加.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
DATA_OUT = REPO_ROOT / "paper_learning/2. AAAI最终故事/data"
CACHE = REPO_ROOT / "models/e2e_cache_a1"
AP_DIR = REPO_ROOT / "results/a1_q_expand"
CALIB_DIR = REPO_ROOT / "calibration/pyramid_dair_e2e_32k"
DAIR_ROOT = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
MAX_VOX = 32000

CACHE.mkdir(parents=True, exist_ok=True)
AP_DIR.mkdir(parents=True, exist_ok=True)

# 复用 e2e_bench_v1_orchestrator.py 的 triplet 定义
TRIPLETS = [
    ("T1_base", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth",
     (64, 128, 256)),
    ("T2_p25", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch_bestval_at25.pth",
     (48, 96, 192)),
    ("T3_p37", str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/net_epoch_bestval_at33.pth"),
     (40, 80, 160)),
    ("T4_p50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at23.pth",
     (32, 64, 128)),
    ("T5_p62", str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/net_epoch_bestval_at33.pth"),
     (24, 56, 128)),
    ("T6_p75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at25.pth",
     (16, 32, 64)),
    ("T7_wide_shallow", str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/net_epoch_bestval_at33.pth"),
     (48, 64, 128)),
    ("T8_narrow_deep", str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/net_epoch_bestval_at33.pth"),
     (24, 48, 192)),
]

# (q_tag, mixed_int8_substr) — precision 固定 mixed, calibrator 固定 minmax.
# 语义 (按 class_a 命名约定): Q_mix_sX = "只 backbone stage X INT8, 其余 (含 VFE/collab/heads) 全 FP16".
# 用 --mixed-int8-substr 指定唯一 INT8 子集, 其余全 FP16 (interpretation A).
# 注: substring "layer0" 同时匹配 /backbone_m1/resnet/layer0/... 与 /resnet/layer0/...,
# 两个 backbone 实例的 stage 0 都 INT8, 与 paper 单 stage 量化语义一致.
Q_CONFIGS = [
    ("Q_mix_s0", "layer0"),
    ("Q_mix_s2", "layer2"),
]


def run(cmd, timeout=1800, env=None, cwd=None):
    print(f"[run ] {' '.join(cmd[:4])} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=timeout, env=env, cwd=cwd)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"cmd failed rc={r.returncode}: {' '.join(cmd[:3])}")
    return r


def export_onnx(tag: str, ckpt: str) -> Path:
    """Reuse e2e_cache/{tag}.onnx if exists."""
    onnx = REPO_ROOT / "models/e2e_cache" / f"{tag}.onnx"
    if not onnx.exists():
        raise FileNotFoundError(f"ONNX 不存在, 先跑 e2e_bench_v1_orchestrator.py: {onnx}")
    return onnx


def build_engine(tag: str, q_tag: str, mixed_substr: str, onnx: Path,
                 engine: Path, calib_cache: Path, report: Path) -> dict:
    if engine.exists() and report.exists():
        print(f"[build] {tag}/{q_tag} cached")
        return json.loads(report.read_text())
    t0 = time.time()
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx),
           "--precision", "mixed",
           "--engine", str(engine),
           "--report", str(report),
           "--workspace-mb", "4096",
           "--skip-bench",
           "--calibrator", "minmax",
           "--mixed-int8-substr", mixed_substr,
           "--calib-multi", f"voxel_features:{CALIB_DIR}/voxel_features.npy",
           "--calib-multi", f"voxel_num_points:{CALIB_DIR}/voxel_num_points.npy",
           "--calib-multi", f"voxel_coords:{CALIB_DIR}/voxel_coords.npy",
           "--calib-multi", f"voxel_mask:{CALIB_DIR}/voxel_mask.npy",
           "--calib-multi", f"t_ego:{CALIB_DIR}/t_ego.npy",
           "--calib-cache", str(calib_cache)]
    run(cmd, timeout=1200)
    rep = json.loads(report.read_text())
    print(f"[build] {tag}/{q_tag}: {time.time()-t0:.0f}s, "
          f"engine={rep.get('engine_size_mb', 0):.1f}MB")
    return rep


def bench_engine(tag: str, q_tag: str, ckpt_dir: str, engine: Path,
                 report: Path) -> dict:
    if report.exists():
        print(f"[bench] {tag}/{q_tag} cached")
        return json.loads(report.read_text())
    t0 = time.time()
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase2/e2e_bench_pyramid.py"),
           "--engine", str(engine),
           "--ckpt-dir", ckpt_dir,
           "--dair-root", DAIR_ROOT,
           "--max-voxels", str(MAX_VOX),
           "--n-warmup", "10",
           "--n-measure", "80",
           "--report", str(report)]
    run(cmd, timeout=1800)
    rep = json.loads(report.read_text())
    print(f"[bench] {tag}/{q_tag}: e2e={rep['lat_e2e_ms']['mean']:.2f}ms "
          f"({time.time()-t0:.0f}s)")
    return rep


def eval_ap(tag: str, q_tag: str, ckpt_dir: str, engine: Path,
            report: Path) -> dict:
    if report.exists():
        print(f"[ap   ] {tag}/{q_tag} cached")
        return json.loads(report.read_text())
    t0 = time.time()
    anchor_id = f"{tag}_{q_tag}_e2e"
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase2/e2e_eval_ap.py"),
           "--engine", str(engine),
           "--ckpt-dir", ckpt_dir,
           "--dair-root", DAIR_ROOT,
           "--max-voxels", str(MAX_VOX),
           "--n-samples", "1789",
           "--tag", anchor_id,
           "--report", str(report)]
    run(cmd, timeout=3600)
    rep = json.loads(report.read_text())
    ap50 = rep.get("ap_50") or rep.get("ap50") or 0.0
    print(f"[ap   ] {tag}/{q_tag}: AP50={ap50:.4f} ({time.time()-t0:.0f}s)")
    return rep


def make_row(tag: str, q_tag: str, planes: tuple,
             bld: dict, bench: dict, ap: dict, baseline_e2e: float,
             baseline_trt: float) -> dict:
    lat_e2e_mean = bench["lat_e2e_ms"]["mean"]
    lat_trt_mean = bench["lat_trt_ms"]["mean"]
    ap30 = ap.get("ap_30") or ap.get("ap30") or np.nan
    ap50 = ap.get("ap_50") or ap.get("ap50") or np.nan
    ap70 = ap.get("ap_70") or ap.get("ap70") or np.nan
    return {
        "triplet": tag,
        "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
        "prune_object": "channel" if planes != (64, 128, 256) else "none",
        "sparse_mask": "dense",
        "q_tag": q_tag, "prec_flag": "mixed",
        "q_bits": "mixed", "q_bits_per_stage": "INT8/FP16/FP16" if q_tag == "Q_mix_s0" else "FP16/FP16/INT8",
        "q_granularity": "per-tensor", "q_object": "W+A",
        "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "hardware": "rtx4090",
        "device": bench.get("device", "cuda:0"),
        "max_voxels": MAX_VOX,
        "n_collected": bench["n_collected"], "n_skipped": bench["n_skipped"],
        "real_voxels_mean": bench["real_voxels"]["mean"],
        "real_voxels_p99": bench["real_voxels"]["p99"],
        "lat_e2e_mean_ms": round(lat_e2e_mean, 4),
        "lat_e2e_p50_ms": round(bench["lat_e2e_ms"]["p50"], 4),
        "lat_e2e_p99_ms": round(bench["lat_e2e_ms"]["p99"], 4),
        "lat_trt_mean_ms": round(lat_trt_mean, 4),
        "lat_trt_p50_ms": round(bench["lat_trt_ms"]["p50"], 4),
        "lat_trt_p99_ms": round(bench["lat_trt_ms"]["p99"], 4),
        "lat_postproc_mean_ms": round(bench["lat_postproc_ms"]["mean"], 4),
        "lat_postproc_p50_ms": round(bench["lat_postproc_ms"]["p50"], 4),
        "speedup_e2e_vs_baseline": round(baseline_e2e / lat_e2e_mean, 4),
        "speedup_trt_vs_baseline": round(baseline_trt / lat_trt_mean, 4),
        "throughput_fps": round(1000.0 / lat_e2e_mean, 4),
        "ap30": ap30, "ap50": ap50, "ap70": ap70,
        "engine_size_mb": bld.get("engine_size_mb"),
        "build_secs": bld.get("build_secs"),
        "build_success": True, "fail_reason": "",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }


def load_baseline() -> tuple[float, float]:
    """Baseline = T1_base Q_fp32 from existing e2e_bench_v1."""
    csv_path = DATA_OUT / "e2e_bench_v1.csv"
    df = pd.read_csv(csv_path)
    base = df[(df["triplet"] == "T1_base") & (df["q_tag"] == "Q_fp32")].iloc[0]
    return float(base["lat_e2e_mean_ms"]), float(base["lat_trt_mean_ms"])


def main():
    ap_arg = argparse.ArgumentParser()
    ap_arg.add_argument("--skip-ap", action="store_true",
                        help="skip AP eval (build + lat only)")
    args = ap_arg.parse_args()

    baseline_e2e, baseline_trt = load_baseline()
    print(f"[baseline] T1_base+Q_fp32 e2e={baseline_e2e:.3f}ms trt={baseline_trt:.3f}ms")

    rows: list[dict] = []
    failures: list[dict] = []

    for tag, ckpt, planes in TRIPLETS:
        ckpt_dir = str(Path(ckpt).parent)
        try:
            onnx = export_onnx(tag, ckpt)
        except Exception as e:
            print(f"[err  ] {tag} onnx: {e}")
            continue

        for q_tag, mixed in Q_CONFIGS:
            engine = CACHE / f"{tag}_{q_tag}.engine"
            calib_cache = CACHE / f"{tag}_{q_tag}_calib.cache"
            build_rep = CACHE / f"{tag}_{q_tag}.build.json"
            bench_rep = CACHE / f"{tag}_{q_tag}_bench.json"
            ap_rep = AP_DIR / f"{tag}_{q_tag}_ap.json"

            try:
                bld = build_engine(tag, q_tag, mixed, onnx, engine,
                                   calib_cache, build_rep)
                bench = bench_engine(tag, q_tag, ckpt_dir, engine, bench_rep)
                if args.skip_ap:
                    ap = {"ap_30": np.nan, "ap_50": np.nan, "ap_70": np.nan}
                else:
                    ap = eval_ap(tag, q_tag, ckpt_dir, engine, ap_rep)
                row = make_row(tag, q_tag, planes, bld, bench, ap,
                               baseline_e2e, baseline_trt)
                rows.append(row)

                # Persist incrementally to staging file
                staging = DATA_OUT / "e2e_bench_v1_a1_staging.csv"
                pd.DataFrame(rows).to_csv(staging, index=False)
                print(f"  [persist] {len(rows)} rows -> {staging.name}")
            except Exception as e:
                print(f"[err  ] {tag}/{q_tag}: {e}")
                failures.append({"triplet": tag, "q_tag": q_tag,
                                 "fail_reason": str(e)[:300]})

    if not rows:
        print("\n=== no successful anchors ===")
        sys.exit(1)

    # Merge staging into e2e_bench_v1
    main_csv = DATA_OUT / "e2e_bench_v1.csv"
    main_pq = DATA_OUT / "e2e_bench_v1.parquet"
    existing = pd.read_csv(main_csv)
    print(f"\n[merge] existing rows: {len(existing)}")
    new_df = pd.DataFrame(rows)
    print(f"[merge] new rows: {len(new_df)}")
    combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    # Dedup on (triplet, q_tag, d_tactic, d_workspace_gb, hardware) — keep latest
    combined = combined.drop_duplicates(
        subset=["triplet", "q_tag", "d_tactic", "d_workspace_gb", "hardware"],
        keep="last")
    combined.to_csv(main_csv, index=False)
    combined.to_parquet(main_pq, index=False)
    print(f"[merge] wrote {len(combined)} rows to e2e_bench_v1")

    if failures:
        print(f"\n[FAIL] {len(failures)} anchors failed:")
        for f in failures:
            print(f"  {f['triplet']}/{f['q_tag']}: {f['fail_reason']}")

    print("\n=== A.1 phase1 done ===")


if __name__ == "__main__":
    main()
