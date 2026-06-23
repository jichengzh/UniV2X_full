"""Part A 续 — Per-stage 混合精度 **真实 AP** 验证.

旧结论 (perstage_quant_pareto.md) 只测了 18 行延迟、未跑 AP, 就断言整个
per-stage 维度无 Pareto 价值。本脚本对每个 triplet 的 6 个 per-stage 混精配置
(c3-c8) **真实 build mixed engine + 真测 DAIR val AP**, 与同 triplet 的全 FP16 /
全 INT8 AP (data/stage_a_ap_real.parquet) 对照, 判定是否存在 AP 正向数据点
(混精 AP > 全 INT8)。

复用 scripts/phase2/stage_a_ap_real.py 的 TRT-path AP 评测流程 (m4_8_hybrid_infer_ap),
build 命令与 p0_perstage_quant_bench.py 完全一致 (mixed + --mixed-fp16-substr),
但 ONNX 用 **真实 finetuned ckpt** (stage_a_cache/*.onnx), 否则 AP 无意义。

GPU: 默认 6 (空闲卡). 输出: results/perstage_quant_AP_real_v1.csv
"""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
SA_CACHE = REPO_ROOT / "models/stage_a_cache"          # 真 ckpt ONNX (base/pruned50/pruned75)
ENG_CACHE = REPO_ROOT / "models/perstage_quant_ap_cache"
ENG_CACHE.mkdir(parents=True, exist_ok=True)
OUT = REPO_ROOT / "results/perstage_quant_ap"
OUT.mkdir(parents=True, exist_ok=True)

CALIB_SPATIAL = "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = "calibration/pyramid_dair_collab_tego.npy"

# triplet -> (real-ckpt tag in stage_a_cache, ckpt_dir, planes)
# 注意: stage_a pruned50 ckpt 是 (32,64,128), bench T_prune50 标 (32,64,136);
#       AP 用真 ckpt, 故此 triplet 标 T_prune50p (32,64,128), 延迟 provenance 不同已注明.
TRIPLETS = [
    ("T_baseline",  "base",     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29", (64, 128, 256)),
    ("T_prune50p",  "pruned50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",      (32, 64, 128)),
    ("T_prune75",   "pruned75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10",      (16, 32, 64)),
]

# (label, s0_prec, s1_prec, s2_prec, fp16_substrs)  —— 与 p0_perstage_quant_bench.py 完全一致
PERSTAGE_CONFIGS = [
    ("c3_I_F_F", "INT8", "FP16", "FP16", "/resnet/layer1/;/resnet/layer2/"),
    ("c4_F_I_F", "FP16", "INT8", "FP16", "/resnet/layer0/;/resnet/layer2/"),
    ("c5_F_F_I", "FP16", "FP16", "INT8", "/resnet/layer0/;/resnet/layer1/"),
    ("c6_I_I_F", "INT8", "INT8", "FP16", "/resnet/layer2/"),
    ("c7_I_F_I", "INT8", "FP16", "INT8", "/resnet/layer1/"),
    ("c8_F_I_I", "FP16", "INT8", "INT8", "/resnet/layer0/"),
]

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "6")


def build_mixed_engine(onnx_path: Path, sa_tag: str, c_label: str, fp16_substrs: str):
    eng = ENG_CACHE / f"{sa_tag}_{c_label}.engine"
    report = ENG_CACHE / f"{sa_tag}_{c_label}_build.json"
    if eng.exists():
        print(f"  [build {sa_tag} {c_label}] cache hit")
        return eng
    calib_cache = ENG_CACHE / f"{sa_tag}_{c_label}_calib.cache"
    cmd = [
        PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
        "--onnx", str(onnx_path),
        "--precision", "mixed",
        "--mixed-fp16-substr", fp16_substrs,
        "--engine", str(eng),
        "--report", str(report),
        "--input-shape", "2,64,128,256",
        "--extra-input-shape", "t_ego:2,2,3",
        "--n-warmup", "50", "--n-measure", "100",
        "--calib-multi", f"spatial_features:{CALIB_SPATIAL}",
        "--calib-multi", f"t_ego:{CALIB_TEGO}",
        "--calib-cache", str(calib_cache),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    print(f"  [build {sa_tag} {c_label}] mixed FP16-keep={fp16_substrs}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200, env=env)
    if r.returncode != 0 or not eng.exists():
        print(f"    BUILD FAILED ({time.time()-t0:.0f}s): {r.stderr[-600:]}")
        return None
    print(f"    built in {time.time()-t0:.0f}s")
    return eng


def ap_eval(engine_path: Path, full_tag: str, ckpt_dir: str, n_samples=1789):
    report = OUT / f"{full_tag}.json"
    if report.exists():
        rep = json.loads(report.read_text())
        print(f"  [AP {full_tag}] cache hit  AP50={rep['ap50']:.4f}")
        return rep
    cmd = [
        PYTHON, str(REPO_ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
        "--engine-collab", str(engine_path),
        "--tag", full_tag,
        "--model-dir", ckpt_dir,
        "--n-samples", str(n_samples),
        "--dataset", "dair", "--range", "102.4,51.2",
        "--collab-spatial-shape", "2,64,128,256",
        "--collab-tego-shape", "2,2,3",
        "--report", str(report),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    print(f"  [AP {full_tag}] eval on {n_samples} samples ...")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HEAL_ROOT, env=env, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    if r.returncode != 0 or not report.exists():
        print(f"    AP FAILED ({elapsed:.0f}s): {r.stderr[-600:]}")
        return None
    rep = json.loads(report.read_text())
    print(f"    AP30={rep['ap30']:.4f} AP50={rep['ap50']:.4f} AP70={rep['ap70']:.4f} "
          f"n_trt_path={rep['n_trt_path']} ({elapsed:.0f}s)")
    return rep


def main():
    # 全 FP16 / 全 INT8 AP 参考 (复用 stage_a)
    sa = pd.read_parquet(REPO_ROOT / "data/stage_a_ap_real.parquet")
    ref = {}  # sa_tag -> {'fp16': {...}, 'int8': {...}}
    for _, row in sa.iterrows():
        ref.setdefault(row["anchor"], {})[row["precision"]] = row

    t_total = time.time()
    rows = []
    for tri_label, sa_tag, ckpt_dir, planes in TRIPLETS:
        onnx_path = SA_CACHE / f"{sa_tag}.onnx"
        if not onnx_path.exists():
            print(f"[{tri_label}] missing real-ckpt ONNX {onnx_path} — SKIP")
            continue
        int8_ap50 = ref[sa_tag]["int8"]["ap50"]
        fp16_ap50 = ref[sa_tag]["fp16"]["ap50"]
        print(f"\n=== {tri_label} ({sa_tag}) planes={planes} "
              f"| ref fp16_ap50={fp16_ap50:.4f} int8_ap50={int8_ap50:.4f} ===")

        # global references as rows (provenance: reused from stage_a)
        for prec in ("fp16", "int8"):
            rr = ref[sa_tag][prec]
            rows.append({
                "triplet": tri_label, "config_label": f"global_{prec}",
                "stage0_prec": prec.upper(), "stage1_prec": prec.upper(), "stage2_prec": prec.upper(),
                "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                "ap30": rr["ap30"], "ap50": rr["ap50"], "ap70": rr["ap70"],
                "n_trt_path": int(rr["n_trt_path"]), "n_samples": int(rr["n_samples"]),
                "source": "reuse:stage_a_ap_real",
                "delta_ap50_vs_int8": rr["ap50"] - int8_ap50,
            })

        for c_label, p0, p1, p2, fp16_substrs in PERSTAGE_CONFIGS:
            eng = build_mixed_engine(onnx_path, sa_tag, c_label, fp16_substrs)
            if eng is None:
                rows.append({
                    "triplet": tri_label, "config_label": c_label,
                    "stage0_prec": p0, "stage1_prec": p1, "stage2_prec": p2,
                    "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                    "ap30": None, "ap50": None, "ap70": None,
                    "n_trt_path": None, "n_samples": None,
                    "source": "BUILD_FAILED", "delta_ap50_vs_int8": None,
                })
                continue
            full_tag = f"psAP_{sa_tag}_{c_label}"
            rep = ap_eval(eng, full_tag, ckpt_dir, n_samples=1789)
            if rep is None:
                rows.append({
                    "triplet": tri_label, "config_label": c_label,
                    "stage0_prec": p0, "stage1_prec": p1, "stage2_prec": p2,
                    "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                    "ap30": None, "ap50": None, "ap70": None,
                    "n_trt_path": None, "n_samples": None,
                    "source": "AP_FAILED", "delta_ap50_vs_int8": None,
                })
                continue
            rows.append({
                "triplet": tri_label, "config_label": c_label,
                "stage0_prec": p0, "stage1_prec": p1, "stage2_prec": p2,
                "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
                "n_trt_path": int(rep["n_trt_path"]), "n_samples": int(rep["n_samples"]),
                "source": "real_mixed_engine",
                "delta_ap50_vs_int8": rep["ap50"] - int8_ap50,
            })
            print(f"  cumulative {(time.time()-t_total)/60:.1f} min")

    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "results/perstage_quant_AP_real_v1.csv"
    df.to_csv(out_csv, index=False)
    df.to_parquet(REPO_ROOT / "data/perstage_quant_AP_real.parquet")
    print(f"\n[done] {len(df)} rows in {(time.time()-t_total)/60:.1f} min -> {out_csv}")
    cols = ["triplet", "config_label", "stage0_prec", "stage1_prec", "stage2_prec",
            "ap50", "delta_ap50_vs_int8", "source"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
