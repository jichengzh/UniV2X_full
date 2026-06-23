"""Stage A — 4 M4.9 finetuned ckpts × {FP16, INT8} = 8 anchor 真实 AP.

Pipeline per anchor:
    1. Export ONNX from finetuned ckpt (if not cached)
    2. Build TRT engine at precision (FP16 / INT8)
    3. Run hybrid AP eval on DAIR val (1789 samples)

Output: data/stage_a_ap_real.parquet
"""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
CACHE = REPO_ROOT / "models/stage_a_cache"
CACHE.mkdir(parents=True, exist_ok=True)
OUT = REPO_ROOT / "results/stage_a"
OUT.mkdir(parents=True, exist_ok=True)

ANCHORS = [
    # (tag, ckpt_dir, (s0,s1,s2))
    ("base",      "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",  (64,128,256)),
    ("pruned25",  "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10",       (48,96,192)),
    ("pruned50",  "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",       (32,64,128)),
    ("pruned75",  "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10",       (16,32,64)),
]

CALIB_SPATIAL = REPO_ROOT / "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = REPO_ROOT / "calibration/pyramid_dair_collab_tego.npy"


def export_onnx(tag, ckpt_dir):
    out_onnx = CACHE / f"{tag}.onnx"
    if out_onnx.exists(): return out_onnx
    ckpt_pth = Path(ckpt_dir) / "net_epoch_bestval_at23.pth"
    if not ckpt_pth.exists():
        ckpt_pth = next(Path(ckpt_dir).glob("net_epoch*.pth"))
    print(f"[export] {tag} from {ckpt_pth.name}")
    cmd = [PYTHON, str(REPO_ROOT / "tools/export_onnx_pyramid_collab.py"),
           "--ckpt", str(ckpt_pth),
           "--hypes", str(Path(ckpt_dir) / "config.yaml"),
           "--out", str(out_onnx),
           "--feat-h", "128"]
    env = {"CUDA_VISIBLE_DEVICES": "0", "PATH": os.environ.get("PATH", "")}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if r.returncode != 0 or not out_onnx.exists():
        print(f"  FAILED: {r.stderr[-500:]}")
        return None
    return out_onnx


def build_engine(onnx_path, tag, precision, gpu="0"):
    eng = CACHE / f"{tag}_{precision}.engine"
    if eng.exists(): return eng
    print(f"[build] {tag} {precision}")
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx_path),
           "--precision", precision,
           "--engine", str(eng),
           "--report", str(CACHE / f"{tag}_{precision}_build.json"),
           "--input-shape", "2,64,128,256",
           "--extra-input-shape", "t_ego:2,2,3",
           "--n-warmup", "100", "--n-measure", "200"]
    if precision == "int8":
        cmd += ["--calib-multi", f"spatial_features:{CALIB_SPATIAL}",
                "--calib-multi", f"t_ego:{CALIB_TEGO}",
                "--calib-cache", str(CACHE / f"{tag}_int8_calib.cache")]
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PATH": os.environ.get("PATH", "")}
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, cwd=REPO_ROOT)
    if r.returncode != 0 or not eng.exists():
        print(f"  FAILED: {r.stderr[-500:]}")
        return None
    return eng


def ap_eval(engine_path, tag, ckpt_dir, n_samples=1789, gpu="0"):
    report = OUT / f"{tag}.json"
    if report.exists(): return json.loads(report.read_text())
    cmd = [PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
           "--engine-collab", str(engine_path),
           "--tag", tag,
           "--model-dir", ckpt_dir,
           "--n-samples", str(n_samples),
           "--dataset", "dair", "--range", "102.4,51.2",
           "--collab-spatial-shape", "2,64,128,256",
           "--collab-tego-shape", "2,2,3",
           "--report", str(report)]
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PATH": os.environ.get("PATH", "")}
    print(f"[AP eval] {tag} on {n_samples} samples")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env, capture_output=True, text=True, timeout=1500)
    elapsed = time.time() - t0
    if r.returncode != 0 or not report.exists():
        print(f"  FAILED ({elapsed:.0f}s): {r.stderr[-500:]}")
        return None
    rep = json.loads(report.read_text())
    print(f"  AP30={rep['ap30']:.4f} AP50={rep['ap50']:.4f} AP70={rep['ap70']:.4f} ({elapsed:.0f}s)")
    return rep


def main():
    t0 = time.time()
    # Build phase: ensure all 8 engines exist
    engines = {}
    for tag, ckpt_dir, planes in ANCHORS:
        onnx = export_onnx(tag, ckpt_dir)
        if onnx is None: continue
        for prec in ("fp16", "int8"):
            eng = build_engine(onnx, tag, prec)
            if eng is None: continue
            engines[(tag, prec)] = (eng, ckpt_dir, planes)
    print(f"\n[builds done] {len(engines)}/8 engines ready ({(time.time()-t0)/60:.1f} min)")

    # AP eval phase
    rows = []
    for (tag, prec), (eng, ckpt_dir, planes) in engines.items():
        full_tag = f"sA_{tag}_{prec}"
        rep = ap_eval(eng, full_tag, ckpt_dir, n_samples=1789)
        if rep is None: continue
        rows.append({
            "anchor": tag, "precision": prec,
            "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
            "ckpt": ckpt_dir,
            "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
            "n_samples": rep["n_samples"], "n_trt_path": rep["n_trt_path"],
            "elapsed_secs": rep["elapsed_secs"],
        })

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "data/stage_a_ap_real.parquet"
    df.to_parquet(out); df.to_csv(out.with_suffix(".csv"), index=False)
    elapsed_total = (time.time() - t0) / 60
    print(f"\n[done] {len(df)} rows in {elapsed_total:.1f} min -> {out}")
    print(df[["anchor","precision","stage0_planes","stage1_planes","stage2_planes","ap50","ap70"]].to_string(index=False))


if __name__ == "__main__":
    main()
