"""Path A (AP 激活, 免训) — 剪枝×forced-all-INT8 三档 DAIR val 1789 AP.

3 个不同 pyramid_backbone 档各出 forced-all-int8 AP (calibrator=minmax):
  0.364M cliff2_c [16,32,64] / 0.270M prune90 [~6,13,26] / 0.088M prune95 [~4,6,13]
FP16 AP 已有 (ap_cliff json) 不重跑; forced engine 存 cache 供 hw 同引擎测 latency.

forced-all-int8 = --precision mixed --mixed-fp16-substr "__NO_SUCH_LAYER__"
口径: body_subnet_collab2 (2x64x128x256 + t_ego 2x2x3). regime=ablation_guardrail_off (被支配点).
GPU: CUDA_VISIBLE_DEVICES 在 import p0 前设好.
输出: results/pathA_forced_int8_ap.json
"""
from __future__ import annotations
import os, json, subprocess, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import importlib.util
REPO = Path("/home/jichengzhi/UniV2X")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
spec = importlib.util.spec_from_file_location(
    "p0_perstage", str(REPO / "scripts/phase2/p0_perstage_quant_AP_real_v2.py"))
p0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p0)

CKPT_BASE = "/home/jichengzhi/heal_research/checkpoints/stage1"
# (tag, ckpt_dir, bestval_file, planes, pb_params_M)
TIERS = [
    ("cliff2_c", f"{CKPT_BASE}/Pyramid_DAIR_m1_cliff2_c_2026_06_02", "net_epoch_bestval_at39.pth", (16, 32, 64), 0.364),
    ("prune90",  f"{CKPT_BASE}/Pyramid_DAIR_m1_prune90_2026_06_02",  "net_epoch_bestval_at45.pth", (6, 13, 26),  0.270),
    ("prune95",  f"{CKPT_BASE}/Pyramid_DAIR_m1_prune95_2026_06_02",  "net_epoch_bestval_at39.pth", (4, 6, 13),   0.088),
]
C_LABEL = "c_all_int8_forced"
FP16_SUBSTR = "__NO_SUCH_LAYER__"


def export_onnx(tag, ckpt_dir, bestval):
    out_onnx = p0.SA_CACHE / f"{tag}.onnx"
    if out_onnx.exists():
        print(f"  [export {tag}] cache hit"); return out_onnx
    ckpt = Path(ckpt_dir) / bestval
    cmd = [PYTHON, str(REPO / "tools/export_onnx_pyramid_collab.py"),
           "--ckpt", str(ckpt), "--hypes", str(Path(ckpt_dir) / "config.yaml"),
           "--out", str(out_onnx), "--feat-h", "128"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}
    print(f"  [export {tag}] from {bestval}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=400, env=env, cwd=REPO)
    if r.returncode != 0 or not out_onnx.exists():
        print(f"    EXPORT FAILED: {r.stderr[-500:]}"); return None
    return out_onnx


def main():
    results = []
    t0 = time.time()
    for tag, ckpt_dir, bestval, planes, pb in TIERS:
        print(f"\n=== {tag} planes={planes} pb={pb}M ===")
        onnx = export_onnx(tag, ckpt_dir, bestval)
        if onnx is None:
            results.append({"tag": tag, "status": "export_failed"}); continue
        eng, summary = p0.build_mixed_engine(onnx, tag, C_LABEL, FP16_SUBSTR)
        if eng is None:
            results.append({"tag": tag, "status": "build_failed"}); continue
        rep = p0.ap_eval(eng, f"pathA_{tag}_{C_LABEL}", ckpt_dir, n_samples=1789)
        if rep is None:
            results.append({"tag": tag, "status": "ap_failed", "engine": str(eng)}); continue
        results.append({
            "tag": tag, "status": "ok", "planes": list(planes), "pb_params_M": pb,
            "config": "forced_all_int8", "calibrator": "minmax",
            "regime": "ablation_guardrail_off",
            "ckpt": str(Path(ckpt_dir) / bestval), "onnx": str(onnx), "engine": str(eng),
            "fp16_int8_summary": summary,
            "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
            "n_samples": rep["n_samples"], "n_trt_path": rep["n_trt_path"],
        })
        print(f"  -> AP30={rep['ap30']:.4f} AP50={rep['ap50']:.4f} AP70={rep['ap70']:.4f}")
    out = REPO / "results/pathA_forced_int8_ap.json"
    out.write_text(json.dumps({"elapsed_min": round((time.time()-t0)/60, 1), "tiers": results}, indent=2))
    print(f"\n[done] -> {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
