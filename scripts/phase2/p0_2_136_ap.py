"""P0-2 — [32,64,136] (prune50b2, stage2=136 非32倍数=非对齐对照) FP16+auto-INT8 AP.

收敛模型 = bestval_at35 (finetune 48ep, bestval 定格 epoch35, 已验 flat 388 keys)。
补 backbone 系耦合矩阵的 [32,64,136] 档(与 p25 stage0=48 互为非对齐对照)。
口径: DAIR val 1789, body_subnet_collab2, INT8=TRT-auto(--precision int8), minmax.
GPU: CUDA_VISIBLE_DEVICES(默认 0)。输出: results/p0_2_136_ap.json
"""
from __future__ import annotations
import os, json, subprocess, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
REPO = Path("/home/jichengzhi/UniV2X")
PY = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL = Path("/home/jichengzhi/heal_research/HEAL")
CACHE = REPO / "models/stage_a_cache"
OUT = REPO / "results/p0_2_136"; OUT.mkdir(parents=True, exist_ok=True)
CALIB_SP = REPO / "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TE = REPO / "calibration/pyramid_dair_collab_tego.npy"

CKPT_DIR = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_prune50b2_032_064_136_2026_06_03"
CKPT = f"{CKPT_DIR}/net_epoch_bestval_at35.pth"
TAG = "p50b2_136"
ONNX = CACHE / f"{TAG}.onnx"


def sh(cmd, timeout, cwd=REPO):
    env = {**os.environ}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)


def export():
    if ONNX.exists(): print("[export] cache hit"); return True
    r = sh([PY, str(REPO/"tools/export_onnx_pyramid_collab.py"), "--ckpt", CKPT,
            "--hypes", f"{CKPT_DIR}/config.yaml", "--out", str(ONNX), "--feat-h", "128"], 400)
    if r.returncode != 0 or not ONNX.exists():
        print("EXPORT FAIL:", r.stderr[-600:]); return False
    return True


def build(prec):
    eng = CACHE / f"{TAG}_{prec}.engine"
    if eng.exists(): print(f"[build {prec}] cache hit"); return eng
    cmd = [PY, str(REPO/"scripts/phase1/m4_8_trt_build_bench.py"), "--onnx", str(ONNX),
           "--precision", prec, "--engine", str(eng),
           "--report", str(CACHE/f"{TAG}_{prec}_build.json"),
           "--input-shape", "2,64,128,256", "--extra-input-shape", "t_ego:2,2,3",
           "--n-warmup", "50", "--n-measure", "100"]
    if prec == "int8":
        cmd += ["--calib-multi", f"spatial_features:{CALIB_SP}", "--calib-multi", f"t_ego:{CALIB_TE}",
                "--calib-cache", str(CACHE/f"{TAG}_int8_calib.cache")]
    r = sh(cmd, 1200)
    if r.returncode != 0 or not eng.exists():
        print(f"BUILD {prec} FAIL:", r.stderr[-600:]); return None
    return eng


def ap_eval(eng, prec):
    rep_f = OUT / f"{TAG}_{prec}.json"
    cmd = [PY, str(REPO/"scripts/phase1/m4_8_hybrid_infer_ap.py"), "--engine-collab", str(eng),
           "--tag", f"p0_2_{TAG}_{prec}", "--model-dir", CKPT_DIR, "--n-samples", "1789",
           "--dataset", "dair", "--range", "102.4,51.2",
           "--collab-spatial-shape", "2,64,128,256", "--collab-tego-shape", "2,2,3",
           "--report", str(rep_f)]
    r = sh(cmd, 1800, cwd=HEAL)
    if r.returncode != 0 or not rep_f.exists():
        print(f"AP {prec} FAIL:", r.stderr[-600:]); return None
    return json.loads(rep_f.read_text())


def main():
    assert Path(CKPT).exists(), CKPT
    if not export(): return 1
    rows = []
    for prec in ("fp16", "int8"):
        eng = build(prec)
        if eng is None: rows.append({"prec": prec, "status": "build_fail"}); continue
        rep = ap_eval(eng, prec)
        if rep is None: rows.append({"prec": prec, "status": "ap_fail", "engine": str(eng)}); continue
        rows.append({"prec": prec, "status": "ok", "engine": str(eng),
                     "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
                     "n_samples": rep["n_samples"], "n_trt_path": rep.get("n_trt_collab_path") or rep.get("n_trt_path")})
        print(f"  {prec}: ap50={rep['ap50']:.4f} ap70={rep['ap70']:.4f}")
    out = REPO / "results/p0_2_136_ap.json"
    out.write_text(json.dumps({"anchor": "prune50b2_136", "planes": [32, 64, 136],
                               "note": "stage2=136 非32倍数 非对齐; INT8=TRT-auto; bestval_at35 converged",
                               "ckpt": CKPT, "onnx": str(ONNX),
                               "calibrator": "minmax", "latency_kind": "body_subnet_collab2 (by hw)",
                               "rows": rows}, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
