"""P0.2 — Dataset Class A 主网格 4090: 8 triplet × 6 Q × 3 D = 144 anchor.

Inputs:
    - 4 Stage A ckpts (T1 base, T2 p25, T4 p50, T6 p75)
    - 4 P0.1 NEW ckpts (T3 p37, T5 p62, T7 wide, T8 deep)

Per-anchor pipeline:
    1. Export collab ONNX (一次性/ckpt, 复用 stage_a_cache/* + dataset_a_cache/*)
    2. TRT engine build (per Q × D combo)
    3. Lat bench (CUDA events, n_warmup=100, n_measure=200) — included in build script
    4. AP eval on DAIR-V2X val 500-sample sweep (plan §8.0.4)
    5. Resource capture (engine_size_mb, build_secs, params_kb)

Q cells (6):
    Q_fp16:       --precision fp16
    Q_int8_ent:   --precision int8 --calibrator entropy
    Q_int8_mm:    --precision int8 --calibrator minmax (~per-tensor flavor)
    Q_mix_s0:     --precision int8 --mixed-fp16-substr "stage_1,stage_2"   (只 s0 INT8)
    Q_mix_s2:     --precision int8 --mixed-fp16-substr "stage_0,stage_1"   (只 s2 INT8)
    Q_mix_s01:    --precision int8 --mixed-fp16-substr "stage_2"           (s0+s1 INT8, s2 FP16)

D cells (3):
    D_1GB, D_4GB, D_8GB — workspace_mb varies

Parallel strategy:
    Phase 1: ONNX export 8 ckpt (1 GPU shared, sequential ~5 min)
    Phase 2: TRT build × lat bench, 144 anchor / 8 GPU parallel = ~30 min
    Phase 3: AP eval, 144 anchor / 8 GPU parallel (each GPU sequential) = ~60 min
    Total wall: ~95 min

Output:
    data/_by_class/class_a_pyramid_full.parquet  (144 rows)
"""
from __future__ import annotations
import json, os, re, subprocess, time, multiprocessing as mp
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")

CACHE = REPO_ROOT / "models/dataset_a_main_grid"
CACHE.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO_ROOT / "logs/dataset_a_main"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO_ROOT / "results/dataset_a_main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_SPATIAL = REPO_ROOT / "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = REPO_ROOT / "calibration/pyramid_dair_collab_tego.npy"

# 8 ckpts (T1-T8 per plan §2.1)
CKPTS = [
    # (tag, planes, ckpt_path, hypes_path)
    ("T1_base",  (64, 128, 256),
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml"),
    ("T2_p25",   (48, 96, 192),
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch_bestval_at25.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/config.yaml"),
    ("T3_p37",   (40, 80, 160),
     str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_040_080_160/config.yaml")),
    ("T4_p50",   (32, 64, 128),
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at23.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/config.yaml"),
    ("T5_p62",   (24, 56, 128),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_056_128/config.yaml")),
    ("T6_p75",   (16, 32, 64),
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at25.pth",
     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/config.yaml"),
    ("T7_wide",  (48, 64, 128),
     str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_048_064_128/config.yaml")),
    ("T8_deep",  (24, 48, 192),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/net_epoch_bestval_at33.pth"),
     str(REPO_ROOT / "models/dataset_a_cache/ft_024_048_192/config.yaml")),
]

# 6 Q cells
Q_CELLS = [
    # (label, precision, mixed_substr, calibrator)
    ("Q_fp16",     "fp16",  "",                       "entropy"),
    ("Q_int8_ent", "int8",  "",                       "entropy"),
    ("Q_int8_mm",  "int8",  "",                       "minmax"),
    ("Q_mix_s0",   "int8",  "stage_1,stage_2",        "entropy"),
    ("Q_mix_s2",   "int8",  "stage_0,stage_1",        "entropy"),
    ("Q_mix_s01",  "int8",  "stage_2",                "entropy"),
]

# 3 D cells (workspace as resource Pareto axis)
D_CELLS = [
    ("D_ws1",  1024),
    ("D_ws4",  4096),
    ("D_ws8",  8192),
]


def export_onnx(tag, planes, ckpt, hypes, gpu) -> Path | None:
    """Export collab ONNX. Reuse from existing caches if possible."""
    # Check stage_a_cache existing ONNX
    sig_short = {"T1_base": "base", "T2_p25": "pruned25",
                 "T4_p50": "pruned50", "T6_p75": "pruned75"}.get(tag)
    if sig_short:
        existing = REPO_ROOT / f"models/stage_a_cache/{sig_short}.onnx"
        if existing.exists():
            target = CACHE / f"{tag}.onnx"
            if not target.exists():
                target.symlink_to(existing)
            return target

    target = CACHE / f"{tag}.onnx"
    if target.exists():
        return target

    cmd = [PYTHON, str(REPO_ROOT / "tools/export_onnx_pyramid_collab.py"),
           "--ckpt", ckpt, "--hypes", hypes,
           "--out", str(target), "--feat-h", "128"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    print(f"[onnx GPU{gpu}] {tag} exporting...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0 or not target.exists():
        print(f"[onnx GPU{gpu}] {tag} FAILED: {r.stderr[-300:]}")
        return None
    print(f"[onnx GPU{gpu}] {tag} done ({time.time()-t0:.0f}s)")
    return target


def build_and_bench(tag, onnx_path, q_label, prec, mixed, calib, d_label, ws_mb, gpu) -> dict | None:
    """TRT build + lat bench in one call (m4_8_trt_build_bench.py outputs both)."""
    anchor_id = f"{tag}__{q_label}__{d_label}"
    engine = CACHE / f"{anchor_id}.engine"
    report = CACHE / f"{anchor_id}_build.json"
    if report.exists():
        return json.loads(report.read_text())

    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx_path),
           "--precision", prec,
           "--engine", str(engine),
           "--report", str(report),
           "--workspace-mb", str(ws_mb),
           "--input-shape", "2,64,128,256",
           "--extra-input-shape", "t_ego:2,2,3",
           "--n-warmup", "100", "--n-measure", "200",
           "--calibrator", calib]
    if mixed:
        cmd += ["--mixed-fp16-substr", mixed]
    if prec == "int8":
        cmd += ["--calib-multi", f"spatial_features:{CALIB_SPATIAL}",
                "--calib-multi", f"t_ego:{CALIB_TEGO}",
                "--calib-cache", str(CACHE / f"{anchor_id}_calib.cache")]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       env=env, cwd=REPO_ROOT)
    if r.returncode != 0 or not report.exists():
        print(f"[build GPU{gpu}] {anchor_id} FAILED: {r.stderr[-200:]}")
        return None
    rep = json.loads(report.read_text())
    print(f"[build GPU{gpu}] {anchor_id} OK lat_p50={rep.get('lat_p50_ms',0):.3f}ms ({time.time()-t0:.0f}s)")
    return rep


def ap_eval(tag, anchor_id, engine, hypes_dir, gpu, n_samples=500) -> dict | None:
    """AP eval on DAIR-V2X val sweep (n_samples=500 per plan §8.0.4)."""
    out_json = OUT_DIR / f"{anchor_id}.json"
    if out_json.exists():
        return json.loads(out_json.read_text())

    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
           "--engine-collab", str(engine),
           "--tag", anchor_id,
           "--model-dir", str(hypes_dir),
           "--n-samples", str(n_samples),
           "--dataset", "dair", "--range", "102.4,51.2",
           "--collab-spatial-shape", "2,64,128,256",
           "--collab-tego-shape", "2,2,3",
           "--report", str(out_json)]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
           "PYTHONPATH": str(HEAL_ROOT)}
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env,
                           capture_output=True, text=True, timeout=1200,
                           start_new_session=True)
    except subprocess.TimeoutExpired:
        print(f"[ap GPU{gpu}] {anchor_id} timeout 1200s")
        return None
    if r.returncode != 0 or not out_json.exists():
        print(f"[ap GPU{gpu}] {anchor_id} FAILED: {r.stderr[-200:]}")
        return None
    rep = json.loads(out_json.read_text())
    print(f"[ap GPU{gpu}] {anchor_id} AP50={rep['ap50']:.4f} AP70={rep['ap70']:.4f} ({time.time()-t0:.0f}s)")
    return rep


def process_anchor(args):
    """One anchor: build + lat + AP. Called via mp.Pool."""
    tag, planes, ckpt, hypes, q_label, prec, mixed, calib, d_label, ws_mb, gpu = args
    anchor_id = f"{tag}__{q_label}__{d_label}"
    hypes_dir = str(Path(hypes).parent)

    try:
        onnx = export_onnx(tag, planes, ckpt, hypes, gpu)
        if onnx is None:
            return {"anchor_id": anchor_id, "status": "onnx_failed"}

        build_rep = build_and_bench(tag, onnx, q_label, prec, mixed, calib,
                                    d_label, ws_mb, gpu)
        if build_rep is None:
            return {"anchor_id": anchor_id, "status": "build_failed"}

        engine = CACHE / f"{anchor_id}.engine"
        ap_rep = ap_eval(tag, anchor_id, engine, hypes_dir, gpu)
        if ap_rep is None:
            return {"anchor_id": anchor_id, "status": "ap_failed",
                    **build_rep}

        # Resource: engine size in MB
        eng_size_mb = engine.stat().st_size / 1024 / 1024 if engine.exists() else 0
        return {
            "anchor_id": anchor_id,
            "status": "OK",
            "triplet_tag": tag,
            "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
            "q_label": q_label, "precision": prec, "mixed_substr": mixed, "calibrator": calib,
            "d_label": d_label, "workspace_mb": ws_mb,
            "lat_p50_ms": build_rep.get("lat_p50_ms"),
            "lat_p99_ms": build_rep.get("lat_p99_ms"),
            "lat_mean_ms": build_rep.get("lat_mean_ms"),
            "build_secs": build_rep.get("build_secs"),
            "engine_size_mb": eng_size_mb,
            "ap30": ap_rep.get("ap30"),
            "ap50": ap_rep.get("ap50"),
            "ap70": ap_rep.get("ap70"),
            "n_samples": ap_rep.get("n_samples"),
            "n_trt_path": ap_rep.get("n_trt_path"),
            "n_pytorch_fallback": ap_rep.get("n_pytorch_fallback"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"anchor_id": anchor_id, "status": "exception", "error": str(e)}


def build_task_list() -> list:
    """144 anchor tasks distributed to 8 GPUs (round-robin)."""
    tasks = []
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    idx = 0
    for tag, planes, ckpt, hypes in CKPTS:
        for q_label, prec, mixed, calib in Q_CELLS:
            for d_label, ws_mb in D_CELLS:
                gpu = gpu_ids[idx % len(gpu_ids)]
                tasks.append((tag, planes, ckpt, hypes,
                              q_label, prec, mixed, calib,
                              d_label, ws_mb, gpu))
                idx += 1
    return tasks


def main():
    print("=" * 78)
    print("P0.2 Dataset Class A Main Grid (4090): 8 × 6 × 3 = 144 anchor")
    print("=" * 78)
    tasks = build_task_list()
    print(f"Total: {len(tasks)} anchor, distributed across 8 GPUs round-robin")
    print()

    t0 = time.time()
    with mp.Pool(processes=8) as pool:
        results = pool.imap_unordered(process_anchor, tasks)
        rows = []
        for i, r in enumerate(results, 1):
            rows.append(r)
            if i % 10 == 0:
                print(f"\n[progress] {i}/{len(tasks)} anchors done ({(time.time()-t0)/60:.1f} min wall)\n")

    elapsed = (time.time() - t0) / 60
    print(f"\n[done] {len(rows)} anchors in {elapsed:.1f} min")

    # Save parquet
    df = pd.DataFrame([r for r in rows if r.get("status") == "OK"])
    failed = [r for r in rows if r.get("status") != "OK"]
    out = REPO_ROOT / "data/_by_class/class_a_pyramid_full.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n✅ saved {len(df)} OK anchors → {out}")
    print(f"   {len(failed)} failed (see results/dataset_a_main_grid_failed.json)")
    if failed:
        (REPO_ROOT / "results/dataset_a_main_grid_failed.json").write_text(
            json.dumps(failed, indent=2))

    # AP spread summary
    if not df.empty and "ap50" in df.columns:
        print("\n=== AP50 真实分布 ===")
        print(f"  min:    {df['ap50'].min():.4f}")
        print(f"  max:    {df['ap50'].max():.4f}")
        print(f"  spread: {(df['ap50'].max() - df['ap50'].min())*100:.2f} pp")
        print(f"  std:    {df['ap50'].std()*100:.2f} pp")
        print("\n=== AP50 按 triplet × Q 分组 (mean) ===")
        print(df.groupby(['triplet_tag', 'q_label'])['ap50'].mean().unstack().round(4))


if __name__ == "__main__":
    main()
