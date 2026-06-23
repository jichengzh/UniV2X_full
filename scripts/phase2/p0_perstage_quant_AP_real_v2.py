"""Part A 续 v2 — Per-stage 混合精度 **真实 AP** 验证 (修两个 v1 bug).

v1 (results/perstage_quant_AP_real_v1.csv) 被验收判 FAIL, 原因是两个 bug:

  Bug 1 — separator 不匹配:
    v1 的 c3/c4/c5 用 ';' 连接两段 FP16-keep 子串 (如 "/resnet/layer1/;/resnet/layer2/"),
    但 m4_8_trt_build_bench.py:544 按 ',' split → 整串当成一个含字面分号的子串, 匹配 0 层,
    c3/c4/c5 实际 build 成全 INT8 (0 FP16 / 155 INT8), 却被标成混精。
    修法: 改用 ',' 连接两段 (层路径里没有逗号), 让两段都被正确 split 匹配。

  Bug 2 — 基线不可比:
    v1 的 delta 对 stage_a 的 global_int8 (--precision int8, 不设逐层约束, TRT 自由保
    敏感层 FP16 = 自动混精) 算; 而混精路径用 PREFER_PRECISION_CONSTRAINTS 强制每层精度。
    两者不可比。
    修法: 新增 c_all_int8_forced 配置 —— 走和混精**完全相同**的 build 路径
    (mixed + PREFER_PRECISION_CONSTRAINTS), 但 FP16-keep 子串故意不匹配任何层 →
    所有 quantizable 层强制 INT8。delta_ap50_vs_forced_int8 对它算。
    (保留 global_int8/global_fp16 作参考列, 但有效 delta 用 forced 基线。)

GPU: 默认 6 (空闲卡). 输出 (不覆盖 v1):
    results/perstage_quant_AP_real_v2.csv
    data/perstage_quant_AP_real_v2.parquet  (含 delta_ap50_vs_forced_int8)

build stdout 抓取 `[build] mixed (FP16-match): X FP16 / Y INT8` 行, 落盘到
build report json 的 fp16_int8_summary 字段, 供验证 c3/c4/c5 现在真有 ~2 stage FP16。
"""
from __future__ import annotations
import json, os, re, subprocess, time
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
SA_CACHE = REPO_ROOT / "models/stage_a_cache"          # 真 ckpt ONNX (base/pruned50/pruned75)
ENG_CACHE = REPO_ROOT / "models/perstage_quant_ap_cache_v2"
ENG_CACHE.mkdir(parents=True, exist_ok=True)
OUT = REPO_ROOT / "results/perstage_quant_ap_v2"
OUT.mkdir(parents=True, exist_ok=True)

CALIB_SPATIAL = "calibration/pyramid_dair_collab_spatial.npy"
CALIB_TEGO = "calibration/pyramid_dair_collab_tego.npy"

# triplet -> (real-ckpt tag in stage_a_cache, ckpt_dir, planes)
TRIPLETS = [
    ("T_baseline",  "base",     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29", (64, 128, 256)),
    ("T_prune50p",  "pruned50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",      (32, 64, 128)),
    ("T_prune75",   "pruned75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10",      (16, 32, 64)),
]

# (label, s0_prec, s1_prec, s2_prec, fp16_substrs)
# Bug1 修复: 两段用 ',' 连接 (m4_8 按逗号 split). 层路径里无逗号, 安全。
# c_all_int8_forced: FP16-keep 子串故意写一个绝不匹配任何层名的串 →
#   _set_layer_precision(match_int8=False) 让所有 quantizable 层走 INT8,
#   仍设 PREFER_PRECISION_CONSTRAINTS, 与混精完全同口径 (Bug2 修复的同口径基线)。
PERSTAGE_CONFIGS = [
    ("c_all_int8_forced", "INT8", "INT8", "INT8", "__NO_SUCH_LAYER__"),
    ("c3_I_F_F", "INT8", "FP16", "FP16", "/resnet/layer1/,/resnet/layer2/"),
    ("c4_F_I_F", "FP16", "INT8", "FP16", "/resnet/layer0/,/resnet/layer2/"),
    ("c5_F_F_I", "FP16", "FP16", "INT8", "/resnet/layer0/,/resnet/layer1/"),
    ("c6_I_I_F", "INT8", "INT8", "FP16", "/resnet/layer2/"),
    ("c7_I_F_I", "INT8", "FP16", "INT8", "/resnet/layer1/"),
    ("c8_F_I_I", "FP16", "INT8", "INT8", "/resnet/layer0/"),
]

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "6")

# 抓 build stdout 的混精摘要行: "[build] mixed (FP16-match): 57 FP16 / 98 INT8 / ..."
_SUMMARY_RE = re.compile(r"\[build\] mixed \(FP16-match\):\s*(\d+)\s*FP16\s*/\s*(\d+)\s*INT8")


def build_mixed_engine(onnx_path: Path, sa_tag: str, c_label: str, fp16_substrs: str):
    eng = ENG_CACHE / f"{sa_tag}_{c_label}.engine"
    report = ENG_CACHE / f"{sa_tag}_{c_label}_build.json"
    if eng.exists() and report.exists():
        rep = json.loads(report.read_text())
        print(f"  [build {sa_tag} {c_label}] cache hit  fp16/int8={rep.get('fp16_int8_summary')}")
        return eng, rep.get("fp16_int8_summary")
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
    print(f"  [build {sa_tag} {c_label}] mixed FP16-keep={fp16_substrs!r}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=1200, env=env)
    summary = None
    m = _SUMMARY_RE.search(r.stdout or "")
    if m:
        summary = f"{m.group(1)} FP16 / {m.group(2)} INT8"
    if r.returncode != 0 or not eng.exists():
        print(f"    BUILD FAILED ({time.time()-t0:.0f}s): {r.stderr[-600:]}")
        return None, summary
    # patch summary into report json for later verification / cache reuse
    if report.exists():
        try:
            rep = json.loads(report.read_text())
            rep["fp16_int8_summary"] = summary
            report.write_text(json.dumps(rep, indent=2))
        except Exception:
            pass
    print(f"    built in {time.time()-t0:.0f}s  fp16/int8={summary}")
    return eng, summary


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
    # 全 FP16 / 全 INT8 (auto-mix) AP 参考 (复用 stage_a) — 仅作参考列
    sa = pd.read_parquet(REPO_ROOT / "data/stage_a_ap_real.parquet")
    ref = {}  # sa_tag -> {'fp16': row, 'int8': row}
    for _, row in sa.iterrows():
        ref.setdefault(row["anchor"], {})[row["precision"]] = row

    t_total = time.time()
    rows = []
    for tri_label, sa_tag, ckpt_dir, planes in TRIPLETS:
        onnx_path = SA_CACHE / f"{sa_tag}.onnx"
        if not onnx_path.exists():
            print(f"[{tri_label}] missing real-ckpt ONNX {onnx_path} — SKIP")
            continue
        auto_int8_ap50 = ref[sa_tag]["int8"]["ap50"]   # global_int8 = TRT auto-mix (参考)
        fp16_ap50 = ref[sa_tag]["fp16"]["ap50"]
        print(f"\n=== {tri_label} ({sa_tag}) planes={planes} "
              f"| ref fp16_ap50={fp16_ap50:.4f} auto-int8_ap50={auto_int8_ap50:.4f} ===")

        # --- 先 build+测 forced-all-INT8 (同口径有效基线), 拿它的 ap50 算 delta ---
        forced_ap50 = None
        cfg_results = {}  # c_label -> rep
        for c_label, p0, p1, p2, fp16_substrs in PERSTAGE_CONFIGS:
            eng, summary = build_mixed_engine(onnx_path, sa_tag, c_label, fp16_substrs)
            if eng is None:
                cfg_results[c_label] = (None, None, summary)
                continue
            full_tag = f"psAPv2_{sa_tag}_{c_label}"
            rep = ap_eval(eng, full_tag, ckpt_dir, n_samples=1789)
            cfg_results[c_label] = (rep, (p0, p1, p2), summary)
            if c_label == "c_all_int8_forced" and rep is not None:
                forced_ap50 = rep["ap50"]
            print(f"  cumulative {(time.time()-t_total)/60:.1f} min")

        # --- 参考行: global_fp16 / global_int8 (auto-mix) ---
        for prec in ("fp16", "int8"):
            rr = ref[sa_tag][prec]
            rows.append({
                "triplet": tri_label, "config_label": f"global_{prec}_automix",
                "stage0_prec": prec.upper(), "stage1_prec": prec.upper(), "stage2_prec": prec.upper(),
                "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                "ap30": rr["ap30"], "ap50": rr["ap50"], "ap70": rr["ap70"],
                "n_trt_path": int(rr["n_trt_path"]), "n_samples": int(rr["n_samples"]),
                "fp16_int8_summary": "TRT-auto (no per-layer constraint)",
                "source": "reuse:stage_a_ap_real(reference_only)",
                "delta_ap50_vs_forced_int8": (rr["ap50"] - forced_ap50) if forced_ap50 is not None else None,
                "delta_ap50_vs_automix_int8": rr["ap50"] - auto_int8_ap50,
            })

        # --- forced + 混精行, delta 对 forced-all-INT8 算 (Bug2 修复) ---
        for c_label, p0, p1, p2, fp16_substrs in PERSTAGE_CONFIGS:
            rep, precs, summary = cfg_results.get(c_label, (None, None, None))
            if rep is None:
                rows.append({
                    "triplet": tri_label, "config_label": c_label,
                    "stage0_prec": p0, "stage1_prec": p1, "stage2_prec": p2,
                    "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                    "ap30": None, "ap50": None, "ap70": None,
                    "n_trt_path": None, "n_samples": None,
                    "fp16_int8_summary": summary,
                    "source": "BUILD_OR_AP_FAILED",
                    "delta_ap50_vs_forced_int8": None,
                    "delta_ap50_vs_automix_int8": None,
                })
                continue
            is_baseline = (c_label == "c_all_int8_forced")
            rows.append({
                "triplet": tri_label, "config_label": c_label,
                "stage0_prec": p0, "stage1_prec": p1, "stage2_prec": p2,
                "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
                "ap30": rep["ap30"], "ap50": rep["ap50"], "ap70": rep["ap70"],
                "n_trt_path": int(rep["n_trt_path"]), "n_samples": int(rep["n_samples"]),
                "fp16_int8_summary": summary,
                "source": "forced_all_int8_baseline" if is_baseline else "real_mixed_engine",
                "delta_ap50_vs_forced_int8": (rep["ap50"] - forced_ap50) if forced_ap50 is not None else None,
                "delta_ap50_vs_automix_int8": rep["ap50"] - auto_int8_ap50,
            })

    df = pd.DataFrame(rows)
    out_csv = REPO_ROOT / "results/perstage_quant_AP_real_v2.csv"
    df.to_csv(out_csv, index=False)
    df.to_parquet(REPO_ROOT / "data/perstage_quant_AP_real_v2.parquet")
    print(f"\n[done] {len(df)} rows in {(time.time()-t_total)/60:.1f} min -> {out_csv}")
    cols = ["triplet", "config_label", "stage0_prec", "stage1_prec", "stage2_prec",
            "ap50", "delta_ap50_vs_forced_int8", "fp16_int8_summary", "source"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
