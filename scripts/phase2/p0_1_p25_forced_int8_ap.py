"""P0-1 ② — p25(48/96/192) forced-all-INT8 真实 AP (耦合矩阵陷阱对照).

复用 p0_perstage_quant_AP_real_v2.py 的既证 build/eval 路径:
  forced-all-INT8 = --precision mixed --mixed-fp16-substr "__NO_SUCH_LAYER__"
  (FP16-keep 子串不匹配任何层 → 所有 quantizable 层强制 INT8, PREFER_PRECISION_CONSTRAINTS)

口径: DAIR val 1789 全集, body_subnet_collab2 (输入 2x64x128x256 + t_ego 2x2x3), calibrator=minmax.
任务①(p25 INT8-all = TRT-auto) AP 已在 data/stage_a_ap_real.parquet, 本脚本只补②forced 对照点。

GPU: CUDA_VISIBLE_DEVICES 必须在 import p0 模块前设好 (该模块 import 期读 GPU 全局)。
输出: results/p0_1_p25_forced_int8_ap.json
"""
from __future__ import annotations
import os, json, time
from pathlib import Path

# 锁 GPU (默认 1, 可被环境覆盖) — 必须在 import p0 模块前
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import importlib.util
REPO = Path("/home/jichengzhi/UniV2X")
spec = importlib.util.spec_from_file_location(
    "p0_perstage", str(REPO / "scripts/phase2/p0_perstage_quant_AP_real_v2.py"))
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)

SA_TAG = "pruned25"
ONNX = p0.SA_CACHE / "pruned25.onnx"
CKPT_DIR = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10"
PLANES = (48, 96, 192)
C_LABEL = "c_all_int8_forced"
FP16_SUBSTR = "__NO_SUCH_LAYER__"


def main():
    assert ONNX.exists(), f"missing {ONNX}"
    print(f"[P0-1②] GPU={os.environ['CUDA_VISIBLE_DEVICES']} onnx={ONNX.name}")
    t0 = time.time()
    eng, summary = p0.build_mixed_engine(ONNX, SA_TAG, C_LABEL, FP16_SUBSTR)
    if eng is None:
        print("BUILD FAILED"); return 1
    full_tag = f"P0_1_{SA_TAG}_{C_LABEL}"
    rep = p0.ap_eval(eng, full_tag, CKPT_DIR, n_samples=1789)
    if rep is None:
        print("AP EVAL FAILED"); return 1

    out = {
        "task": "P0-1 ② p25 forced-all-int8",
        "anchor": "pruned25", "planes": list(PLANES),
        "config": "forced_all_int8 (mixed + PREFER, all quantizable layers INT8)",
        "calibrator": "minmax",
        "latency_kind_note": "AP only; latency=body_subnet_collab2 by hw",
        "ckpt": CKPT_DIR, "onnx": str(ONNX), "engine": str(eng),
        "fp16_int8_summary": summary,
        "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
        "n_samples": rep["n_samples"], "n_trt_path": rep["n_trt_path"],
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    out_json = REPO / "results/p0_1_p25_forced_int8_ap.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[P0-1② done] AP30={rep['ap30']:.4f} AP50={rep['ap50']:.4f} "
          f"AP70={rep['ap70']:.4f} -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
