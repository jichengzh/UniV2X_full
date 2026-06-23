"""A.2 D-dim sweep: build + bench for one (T, Q, D) anchor.

D-dim 3 new configs (复用 m4_8 --tactic + --workspace-mb):
  D2_with_cudnn_8gb:   tactic=with_cudnn,  workspace=8192MB
  D3_cublas_lt_16gb:   tactic=cublas_lt,   workspace=16384MB
  D4_all_enabled_1gb:  tactic=all_enabled, workspace=1024MB

D-dim 只影响 kernel 选型, 不影响 engine 数学输出 (same Q/DQ scales, same weights).
→ AP 复用 A.0/A.1 phase1+2 既有 cache (D=default 路径) — 见 reuse_ap_from_a1().
   抽样验证 AP 不变 (问题 5 候选).

Q 维 7 个:
  Q_fp32, Q_fp16, Q_int8_mm, Q_int8_ent, Q_mix_s0, Q_mix_s2, Q_int8_pc_wo

写出: results/a2_d_expand/{triplet}_{q_tag}_{d_tag}.row.json
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

REPO = Path("/home/jichengzhi/UniV2X")
CACHE = REPO / "models/e2e_cache_a2"
AP_DIR = REPO / "results/a2_d_expand"
CALIB = REPO / "calibration/pyramid_dair_e2e_32k"
DAIR = "/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure"
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
MAX_VOX = 32000

# 既有 (T, Q) AP 数据源 (D-dim 数学不变, 直接复用 A.0/A.1 既测 56 行)
E2E_CSV = REPO / "paper_learning/2. AAAI最终故事/data/e2e_bench_v1.csv"

CACHE.mkdir(parents=True, exist_ok=True)
AP_DIR.mkdir(parents=True, exist_ok=True)

TRIPLET_MAP = {
    "T1_base": ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth", (64, 128, 256)),
    "T2_p25":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10/net_epoch_bestval_at25.pth", (48, 96, 192)),
    "T3_p37":  (str(REPO / "models/dataset_a_cache/ft_040_080_160/net_epoch_bestval_at33.pth"), (40, 80, 160)),
    "T4_p50":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10/net_epoch_bestval_at23.pth", (32, 64, 128)),
    "T5_p62":  (str(REPO / "models/dataset_a_cache/ft_024_056_128/net_epoch_bestval_at33.pth"), (24, 56, 128)),
    "T6_p75":  ("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at25.pth", (16, 32, 64)),
    "T7_wide_shallow": (str(REPO / "models/dataset_a_cache/ft_048_064_128/net_epoch_bestval_at33.pth"), (48, 64, 128)),
    "T8_narrow_deep":  (str(REPO / "models/dataset_a_cache/ft_024_048_192/net_epoch_bestval_at33.pth"), (24, 48, 192)),
    # Track A v2 — 13 NEW triplets (训练 by dataset_a_prepare_ckpts_v2.py)
    "T10_p11": (str(REPO / "models/dataset_a_cache/ft_016_128_256/net_epoch_bestval_at33.pth"), (16, 128, 256)),
    "T11_p14": (str(REPO / "models/dataset_a_cache/ft_064_064_256/net_epoch_bestval_at33.pth"), (64,  64, 256)),
    "T12_p21": (str(REPO / "models/dataset_a_cache/ft_032_064_256/net_epoch_bestval_at33.pth"), (32,  64, 256)),
    "T13_p29": (str(REPO / "models/dataset_a_cache/ft_032_032_256/net_epoch_bestval_at33.pth"), (32,  32, 256)),
    "T14_p36": (str(REPO / "models/dataset_a_cache/ft_016_016_256/net_epoch_bestval_at33.pth"), (16,  16, 256)),
    "T15_p39": (str(REPO / "models/dataset_a_cache/ft_016_128_128/net_epoch_bestval_at33.pth"), (16, 128, 128)),
    "T16_p54": (str(REPO / "models/dataset_a_cache/ft_016_064_128/net_epoch_bestval_at33.pth"), (16,  64, 128)),
    "T17_p57": (str(REPO / "models/dataset_a_cache/ft_032_032_128/net_epoch_bestval_at33.pth"), (32,  32, 128)),
    "T18_p64": (str(REPO / "models/dataset_a_cache/ft_016_016_128/net_epoch_bestval_at33.pth"), (16,  16, 128)),
    "T19_p71": (str(REPO / "models/dataset_a_cache/ft_032_032_064/net_epoch_bestval_at33.pth"), (32,  32,  64)),
    "T20_p79": (str(REPO / "models/dataset_a_cache/ft_016_016_064/net_epoch_bestval_at33.pth"), (16,  16,  64)),
    "T21_p82": (str(REPO / "models/dataset_a_cache/ft_016_032_032/net_epoch_bestval_at33.pth"), (16,  32,  32)),
    "T22_p89": (str(REPO / "models/dataset_a_cache/ft_016_016_016/net_epoch_bestval_at33.pth"), (16,  16,  16)),
}

# D-dim configs: (tactic, workspace_mb, builder_opt_level)
# BL=3 = TRT default (现有 csv 1764 anchor 全是隐式 BL=3). v1.4 加 BL 维度后,
# 显式标 BL=3 保持向后兼容; BL=0/5 是步骤 2 噪声地板控制实验.
D_MAP = {
    # 现有 12 D-config (BL=3 implicit, v1.4 显式标 3)
    "D1_default_4gb":     ("default",     4096, 3),  # A.0 baseline parity
    "D2_with_cudnn_8gb":  ("with_cudnn",  8192, 3),
    "D3_cublas_lt_16gb":  ("cublas_lt",   16384, 3),
    "D4_all_enabled_1gb": ("all_enabled", 1024, 3),
    "D5_default_1gb":      ("default",     1024, 3),
    "D6_default_8gb":      ("default",     8192, 3),
    "D7_default_16gb":     ("default",     16384, 3),
    "D8_with_cudnn_4gb":   ("with_cudnn",  4096, 3),
    "D9_cublas_lt_4gb":    ("cublas_lt",   4096, 3),
    "D10_cublas_lt_8gb":   ("cublas_lt",   8192, 3),
    "D11_all_enabled_4gb": ("all_enabled", 4096, 3),
    "D12_edge_only_4gb":   ("edge_only",   4096, 3),
    # v1.4 Stage 2e step 1: 补完 5×4 网格 (8 新)
    "D13_with_cudnn_1gb":   ("with_cudnn",  1024, 3),
    "D14_with_cudnn_16gb":  ("with_cudnn",  16384, 3),
    "D15_cublas_lt_1gb":    ("cublas_lt",   1024, 3),
    "D16_all_enabled_8gb":  ("all_enabled", 8192, 3),
    "D17_all_enabled_16gb": ("all_enabled", 16384, 3),
    "D18_edge_only_1gb":    ("edge_only",   1024, 3),
    "D19_edge_only_8gb":    ("edge_only",   8192, 3),
    "D20_edge_only_16gb":   ("edge_only",   16384, 3),
    # v1.4 Stage 2e step 2: builderOptimizationLevel (4 代表 D × 3 BL = 12 新)
    # BL=0 (最噪)
    "D21_default_4gb_BL0":     ("default",     4096, 0),
    "D22_cublas_lt_8gb_BL0":   ("cublas_lt",   8192, 0),
    "D23_all_enabled_4gb_BL0": ("all_enabled", 4096, 0),
    "D24_edge_only_4gb_BL0":   ("edge_only",   4096, 0),
    # BL=3 (重测 D1/D10/D11/D12 同 (tactic, ws) → 噪声地板控制)
    "D25_default_4gb_BL3":     ("default",     4096, 3),
    "D26_cublas_lt_8gb_BL3":   ("cublas_lt",   8192, 3),
    "D27_all_enabled_4gb_BL3": ("all_enabled", 4096, 3),
    "D28_edge_only_4gb_BL3":   ("edge_only",   4096, 3),
    # BL=5 (最稳)
    "D29_default_4gb_BL5":     ("default",     4096, 5),
    "D30_cublas_lt_8gb_BL5":   ("cublas_lt",   8192, 5),
    "D31_all_enabled_4gb_BL5": ("all_enabled", 4096, 5),
    "D32_edge_only_4gb_BL5":   ("edge_only",   4096, 5),
}

Q_LIST = ["Q_fp32", "Q_fp16", "Q_int8_mm", "Q_int8_ent",
          "Q_mix_s0", "Q_mix_s2", "Q_int8_pc_wo"]


def q_build_config(q_tag: str, triplet: str) -> dict:
    """Per-Q build config: onnx + precision + extras + meta."""
    base_onnx = REPO / "models/e2e_cache" / f"{triplet}.onnx"
    if q_tag == "Q_fp32":
        return {"onnx": base_onnx, "precision": "fp32", "extras": [],
                "uses_calib": False,
                "q_bits": "FP32", "q_bits_per_stage": "FP32/FP32/FP32",
                "q_granularity": "n/a", "q_object": "n/a"}
    if q_tag == "Q_fp16":
        return {"onnx": base_onnx, "precision": "fp16", "extras": [],
                "uses_calib": False,
                "q_bits": "FP16", "q_bits_per_stage": "FP16/FP16/FP16",
                "q_granularity": "n/a", "q_object": "n/a"}
    if q_tag == "Q_int8_mm":
        return {"onnx": base_onnx, "precision": "int8",
                "extras": ["--calibrator", "minmax"],
                "uses_calib": True,
                "q_bits": "INT8", "q_bits_per_stage": "INT8/INT8/INT8",
                "q_granularity": "per-tensor", "q_object": "W+A"}
    if q_tag == "Q_int8_ent":
        return {"onnx": base_onnx, "precision": "int8",
                "extras": ["--calibrator", "entropy"],
                "uses_calib": True,
                "q_bits": "INT8", "q_bits_per_stage": "INT8/INT8/INT8",
                "q_granularity": "per-tensor", "q_object": "W+A"}
    if q_tag == "Q_mix_s0":
        return {"onnx": base_onnx, "precision": "mixed",
                "extras": ["--mixed-int8-substr", "layer0", "--calibrator", "minmax"],
                "uses_calib": True,
                "q_bits": "mixed", "q_bits_per_stage": "INT8/FP16/FP16",
                "q_granularity": "per-tensor", "q_object": "W+A"}
    if q_tag == "Q_mix_s2":
        return {"onnx": base_onnx, "precision": "mixed",
                "extras": ["--mixed-int8-substr", "layer2", "--calibrator", "minmax"],
                "uses_calib": True,
                "q_bits": "mixed", "q_bits_per_stage": "FP16/FP16/INT8",
                "q_granularity": "per-tensor", "q_object": "W+A"}
    if q_tag == "Q_int8_pc_wo":
        return {"onnx": base_onnx, "precision": "int8",
                "extras": ["--calibrator", "minmax", "--w-only"],
                "uses_calib": True,
                "q_bits": "INT8", "q_bits_per_stage": "INT8/INT8/INT8",
                "q_granularity": "per-channel", "q_object": "W-only"}
    raise ValueError(f"unknown q_tag: {q_tag}")


def reuse_ap_from_csv(triplet: str, q_tag: str) -> dict | None:
    """Read AP30/50/70 for (T, Q) from canonical e2e_bench_v1.csv (D-dim 数学等价)."""
    import csv
    if not E2E_CSV.exists():
        return None
    with open(E2E_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["triplet"] == triplet and row["q_tag"] == q_tag:
                return {"ap30": float(row["ap30"]) if row.get("ap30") else None,
                        "ap50": float(row["ap50"]) if row.get("ap50") else None,
                        "ap70": float(row["ap70"]) if row.get("ap70") else None}
    return None


def run(cmd, env, timeout=1800):
    print(f"[run ] {' '.join(cmd[:4])} ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise RuntimeError(f"cmd failed rc={r.returncode}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplet", required=True, choices=list(TRIPLET_MAP))
    ap.add_argument("--q-tag", required=True, choices=Q_LIST)
    ap.add_argument("--d-tag", required=True, choices=list(D_MAP))
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--force-ap", action="store_true",
                    help="重新跑 AP eval (默认复用 A.0/A.1 cache, D-dim 不影响 AP)")
    args = ap.parse_args()

    ckpt, planes = TRIPLET_MAP[args.triplet]
    ckpt_dir = str(Path(ckpt).parent)
    qcfg = q_build_config(args.q_tag, args.triplet)
    tactic, ws_mb, bl_level = D_MAP[args.d_tag]
    if not qcfg["onnx"].exists():
        sys.exit(f"ERR: ONNX missing: {qcfg['onnx']}")

    tag = f"{args.triplet}_{args.q_tag}_{args.d_tag}"
    engine = CACHE / f"{tag}.engine"
    build_rep = CACHE / f"{tag}.build.json"
    bench_rep = CACHE / f"{tag}_bench.json"
    calib_cache = CACHE / f"{tag}_calib.cache"
    ap_rep = AP_DIR / f"{tag}_ap.json"
    row_out = AP_DIR / f"{tag}.row.json"

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    print(f"[gpu {args.gpu}] {tag} start (tactic={tactic} ws={ws_mb}MB BL={bl_level})")

    # 1. Build with --tactic + --workspace-mb + --builder-opt-level
    if engine.exists() and build_rep.exists():
        bld = json.loads(build_rep.read_text())
        print(f"[build] {tag} cached engine={bld.get('engine_size_mb',0):.1f}MB")
    else:
        t0 = time.time()
        cmd = [PYTHON, str(REPO / "scripts/phase1/m4_8_trt_build_bench.py"),
               "--onnx", str(qcfg["onnx"]),
               "--precision", qcfg["precision"],
               "--engine", str(engine), "--report", str(build_rep),
               "--workspace-mb", str(ws_mb),
               "--tactic", tactic,
               "--builder-opt-level", str(bl_level),
               "--skip-bench"]
        if qcfg["uses_calib"]:
            cmd += ["--calib-multi", f"voxel_features:{CALIB}/voxel_features.npy",
                    "--calib-multi", f"voxel_num_points:{CALIB}/voxel_num_points.npy",
                    "--calib-multi", f"voxel_coords:{CALIB}/voxel_coords.npy",
                    "--calib-multi", f"voxel_mask:{CALIB}/voxel_mask.npy",
                    "--calib-multi", f"t_ego:{CALIB}/t_ego.npy",
                    "--calib-cache", str(calib_cache)]
        cmd += qcfg["extras"]
        try:
            run(cmd, env, timeout=1800)
        except Exception as e:
            # 写 fail row, 不抛
            fail_row = {"triplet": args.triplet, "q_tag": args.q_tag, "d_tag": args.d_tag,
                        "build_success": False, "fail_reason": str(e)[:500],
                        "ts": datetime.now().isoformat(timespec="seconds")}
            row_out.write_text(json.dumps(fail_row, indent=2, default=str))
            print(f"[fail] {tag}: {e}")
            return
        bld = json.loads(build_rep.read_text())
        print(f"[build] {tag}: {time.time()-t0:.0f}s, {bld.get('engine_size_mb',0):.1f}MB")

    # 2. Bench
    if bench_rep.exists():
        bench = json.loads(bench_rep.read_text())
        print(f"[bench] {tag} cached")
    else:
        t0 = time.time()
        cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_bench_pyramid.py"),
               "--engine", str(engine), "--ckpt-dir", ckpt_dir,
               "--dair-root", DAIR, "--max-voxels", str(MAX_VOX),
               "--n-warmup", "10", "--n-measure", "80", "--report", str(bench_rep)]
        run(cmd, env, timeout=1800)
        bench = json.loads(bench_rep.read_text())
        print(f"[bench] {tag}: e2e={bench['lat_e2e_ms']['mean']:.2f}ms ({time.time()-t0:.0f}s)")

    # 3. AP — 复用 e2e_bench_v1.csv (D-dim 数学等价)
    ap_data = None
    ap_source = "reused_csv"
    if not args.force_ap:
        ap_data = reuse_ap_from_csv(args.triplet, args.q_tag)
        if ap_data:
            print(f"[ap   ] {tag} reused from e2e_bench_v1.csv "
                  f"(ap50={ap_data['ap50']:.4f})")
    if ap_data is None:
        ap_source = "a2_run"
        if ap_rep.exists():
            ap_data = json.loads(ap_rep.read_text())
            print(f"[ap   ] {tag} cached (a2)")
        else:
            t0 = time.time()
            cmd = [PYTHON, str(REPO / "scripts/phase2/e2e_eval_ap.py"),
                   "--engine", str(engine), "--ckpt-dir", ckpt_dir,
                   "--dair-root", DAIR, "--max-voxels", str(MAX_VOX),
                   "--n-samples", "1789",
                   "--tag", f"{tag}_e2e",
                   "--report", str(ap_rep)]
            run(cmd, env, timeout=3600)
            ap_data = json.loads(ap_rep.read_text())
            ap50_disp = ap_data.get('ap_50') or ap_data.get('ap50') or 0.0
            print(f"[ap   ] {tag}: AP50={ap50_disp:.4f} ({time.time()-t0:.0f}s)")

    # 4. Compose row
    lat_e2e_mean = bench["lat_e2e_ms"]["mean"]
    row = {
        "triplet": args.triplet,
        "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
        "prune_object": "channel" if planes != (64, 128, 256) else "none",
        "sparse_mask": "dense",
        "q_tag": args.q_tag, "prec_flag": qcfg["precision"],
        "q_bits": qcfg["q_bits"], "q_bits_per_stage": qcfg["q_bits_per_stage"],
        "q_granularity": qcfg["q_granularity"], "q_object": qcfg["q_object"],
        "d_scheme": "GPU", "d_tactic": tactic, "d_workspace_gb": ws_mb // 1024,
        "d_builder_opt_level": bl_level,
        "d_tag": args.d_tag,
        "hardware": "rtx4090",
        "device": bench.get("device", "cuda:0"),
        "max_voxels": MAX_VOX,
        "n_collected": bench["n_collected"], "n_skipped": bench["n_skipped"],
        "real_voxels_mean": bench["real_voxels"]["mean"],
        "real_voxels_p99": bench["real_voxels"]["p99"],
        "throughput_fps": round(1000.0 / lat_e2e_mean, 4),
        "ap30": ap_data.get("ap_30") or ap_data.get("ap30"),
        "ap50": ap_data.get("ap_50") or ap_data.get("ap50"),
        "ap70": ap_data.get("ap_70") or ap_data.get("ap70"),
        "ap_source": ap_source,
        "engine_size_mb": bld.get("engine_size_mb"),
        "build_secs": bld.get("build_secs"),
        "build_success": True, "fail_reason": "",
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    row_out.write_text(json.dumps(row, indent=2, default=str))
    print(f"[done] {tag} -> {row_out.name}")


if __name__ == "__main__":
    main()
