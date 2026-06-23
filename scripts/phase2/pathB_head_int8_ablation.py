"""Path B (AP 激活, 免训, 定位=护栏消融) — 拆 cls/reg/dir head FP16 护栏强压 INT8.

可行性 (已验): collab 子网引擎含 cls_head/reg_head/dir_head/single_head_{0,1,2}
(输出 cls_preds/reg_preds/dir_preds), 但 **不含 encoder/VFE**(在 PyTorch 侧产 spatial_features)。
⇒ 本脚本只做 head 护栏消融; encoder W+A INT8 无法经此引擎做(已报 data)。

配置: --precision mixed --mixed-int8-substr "head"
  → 所有 head conv (9 个) 强制 INT8, 其余层强制 FP16 (m4_8 match_int8=True 语义)。
  对照基线 = global FP16 (ap70: base 0.6309 / p50 0.5641, 金标准)。
  delta = 量化被保护检测头的纯 AP 代价 = "拆护栏" 的损失。
口径: DAIR val 1789, body_subnet_collab2, calibrator=minmax. regime=ablation_guardrail_off.
GPU: CUDA_VISIBLE_DEVICES 在 import p0 前设好 (默认 7)。
输出: results/pathB_head_int8_ablation.json
"""
from __future__ import annotations
import os, json, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
import importlib.util
REPO = Path("/home/jichengzhi/UniV2X")
spec = importlib.util.spec_from_file_location(
    "p0_perstage", str(REPO / "scripts/phase2/p0_perstage_quant_AP_real_v2.py"))
p0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p0)

CKPT_BASE = "/home/jichengzhi/heal_research/checkpoints/stage1"
# (tag, onnx_in_cache, ckpt_dir, planes, fp16_ap70_ref)
TIERS = [
    ("base",     "base",     f"{CKPT_BASE}/Pyramid_DAIR_m1_base_2023_08_14_11_42_29", (64, 128, 256), 0.6309),
    ("pruned50", "pruned50", f"{CKPT_BASE}/Pyramid_DAIR_m1_pruned50_2026_05_10",      (32, 64, 128),  0.5641),
]
C_LABEL = "headINT8_restFP16"
INT8_SUBSTR = "head"   # 命中 cls_head/reg_head/dir_head/single_head_{0,1,2}; 不命中 resnet/shrink/deblock


def build_head_int8(onnx_path, tag):
    """复用 m4_8 mixed + --mixed-int8-substr 'head' (head→INT8, rest→FP16)."""
    import subprocess
    eng = p0.ENG_CACHE / f"{tag}_{C_LABEL}.engine"
    report = p0.ENG_CACHE / f"{tag}_{C_LABEL}_build.json"
    if eng.exists() and report.exists():
        print(f"  [build {tag}] cache hit"); return eng, json.loads(report.read_text()).get("head_int8_summary")
    calib_cache = p0.ENG_CACHE / f"{tag}_{C_LABEL}_calib.cache"
    cmd = [p0.PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
           "--onnx", str(onnx_path), "--precision", "mixed",
           "--mixed-int8-substr", INT8_SUBSTR,
           "--engine", str(eng), "--report", str(report),
           "--input-shape", "2,64,128,256", "--extra-input-shape", "t_ego:2,2,3",
           "--n-warmup", "50", "--n-measure", "100",
           "--calib-multi", f"spatial_features:{p0.CALIB_SPATIAL}",
           "--calib-multi", f"t_ego:{p0.CALIB_TEGO}",
           "--calib-cache", str(calib_cache)]
    env = {**os.environ}
    print(f"  [build {tag}] mixed INT8-match='{INT8_SUBSTR}' (head→INT8, rest→FP16)")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200, env=env)
    import re
    m = re.search(r"\[build\] mixed \(INT8-match\):\s*(\d+)\s*INT8\s*/\s*(\d+)\s*FP16", r.stdout or "")
    summary = f"{m.group(1)} INT8 / {m.group(2)} FP16" if m else None
    if r.returncode != 0 or not eng.exists():
        print(f"    BUILD FAILED: {r.stderr[-600:]}"); return None, summary
    if report.exists():
        try:
            rep = json.loads(report.read_text()); rep["head_int8_summary"] = summary
            report.write_text(json.dumps(rep, indent=2))
        except Exception: pass
    print(f"    built  head_int8_summary={summary}")
    return eng, summary


def main():
    results = []
    t0 = time.time()
    for tag, onnx_tag, ckpt_dir, planes, fp16_ap70 in TIERS:
        onnx = p0.SA_CACHE / f"{onnx_tag}.onnx"
        if not onnx.exists():
            print(f"[{tag}] missing {onnx} — SKIP"); results.append({"tag": tag, "status": "no_onnx"}); continue
        print(f"\n=== Path B {tag} planes={planes} (head→INT8, rest→FP16) ===")
        eng, summary = build_head_int8(onnx, tag)
        if eng is None:
            results.append({"tag": tag, "status": "build_failed"}); continue
        rep = p0.ap_eval(eng, f"pathB_{tag}_{C_LABEL}", ckpt_dir, n_samples=1789)
        if rep is None:
            results.append({"tag": tag, "status": "ap_failed", "engine": str(eng)}); continue
        d70 = rep["ap70"] - fp16_ap70
        results.append({
            "tag": tag, "status": "ok", "planes": list(planes),
            "config": "head_INT8_rest_FP16 (guardrail OFF on detection heads)",
            "forced_int8_layers": "cls_head/reg_head/dir_head/single_head_{0,1,2} (substr 'head')",
            "calibrator": "minmax", "regime": "ablation_guardrail_off",
            "ckpt": ckpt_dir, "onnx": str(onnx), "engine": str(eng),
            "head_int8_summary": summary,
            "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
            "fp16_ap70_ref": fp16_ap70, "delta_ap70_vs_fp16": round(d70, 4),
            "n_samples": rep["n_samples"], "n_trt_path": rep["n_trt_path"],
        })
        print(f"  -> ap70={rep['ap70']:.4f}  ΔAP70 vs FP16={d70:+.4f}")
    out = REPO / "results/pathB_head_int8_ablation.json"
    out.write_text(json.dumps({"elapsed_min": round((time.time()-t0)/60, 1),
                               "note": "encoder/VFE not in subnet engine — head-only ablation",
                               "tiers": results}, indent=2))
    print(f"\n[done] -> {out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
