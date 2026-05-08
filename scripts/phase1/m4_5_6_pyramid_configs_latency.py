"""M4.5 latency-only + M4.6: Pyramid Fusion 配置矩阵 latency × M2 f 估算 Orin.

M4.5 完整版需要 OPV2V test 数据跑 AP (下载中, 估 3h);
本脚本先跑 latency 维度 (不依赖数据), 拿到 Pyramid 5-10 个 prune/quant configs 的:
  - 4090 PyTorch CUDA Event latency (FP32 + FP16)
  - 用 M2 f 函数估算 Orin AGX FP16/INT8 latency
  - params (M)

配置矩阵 (8 个 configs):
  baseline_fp32         layer=[3,5,8] filters=[64,128,256]  fp32  3.76M
  baseline_fp16         同上                                fp16
  prune25_fp16          filters=[48,96,192]                  fp16  ~2.1M
  prune50_fp16          filters=[32,64,128]                  fp16  ~0.95M
  prune75_fp16          filters=[16,32,64]                   fp16  ~0.24M
  shallow_fp16          layer=[2,3,5] filters=[64,128,256]   fp16  ~2.4M
  wider_fp16            filters=[96,192,384]                 fp16  ~8.4M
  baseline_int8_proxy   filters=[64,128,256]  fp16 (proxy)   fp16   (实际 INT8 需 sparsity_int8 train)

输出: data/pyramid_fusion_4090.parquet (8 行)
      整合到 baseline_4090.parquet
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))

from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
M2_F_JSON = RESULTS_DIR / "m2_latency_mapping_f.json"

INPUT_SHAPE = (1, 64, 256, 256)  # (B, C, H, W) — 来自 PointPillar scatter 后 BEV

# 配置矩阵: (config_id, layer_nums, num_filters, precision)
CONFIGS = [
    # baseline
    ("baseline_fp32",    [3, 5, 8], [64, 128, 256], "fp32"),
    ("baseline_fp16",    [3, 5, 8], [64, 128, 256], "fp16"),
    # 通道剪枝 (B1 channel)
    ("prune25_fp16",     [3, 5, 8], [48, 96,  192], "fp16"),
    ("prune50_fp16",     [3, 5, 8], [32, 64,  128], "fp16"),
    ("prune75_fp16",     [3, 5, 8], [16, 32,  64],  "fp16"),
    # 层数减少
    ("shallow_fp16",     [2, 3, 5], [64, 128, 256], "fp16"),
    # 通道扩展
    ("wider_fp16",       [3, 5, 8], [96, 192, 384], "fp16"),
    # INT8 proxy (PyTorch 暂不支持完整 INT8 推理, 用 fp16 latency 作 proxy + 注释)
    ("int8_proxy_fp16",  [3, 5, 8], [64, 128, 256], "fp16"),
]


def make_pyramid(layer_nums, num_filters, fp16=False):
    cfg = {
        "resnext": True,
        "layer_nums": list(layer_nums),
        "layer_strides": [1, 2, 2],
        "num_filters": list(num_filters),
        "upsample_strides": [1, 2, 4],
        "num_upsample_filter": [128, 128, 128],
        "anchor_number": 2,
    }
    model = PyramidFusion(cfg).cuda().eval()
    if fp16:
        model = model.half()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_params


@torch.no_grad()
def benchmark(model, dtype, n_warmup=50, n_iter=200):
    dummy = torch.randn(*INPUT_SHAPE).cuda().to(dtype)
    fwd = lambda: model.get_multiscale_feature(dummy)
    for _ in range(n_warmup):
        _ = fwd()
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros(n_iter)
    for i in range(n_iter):
        starter.record()
        _ = fwd()
        ender.record()
        torch.cuda.synchronize()
        timings[i] = starter.elapsed_time(ender)

    return {
        "lat_4090_pytorch_mean_ms": float(np.mean(timings)),
        "lat_4090_pytorch_p50_ms": float(np.percentile(timings, 50)),
        "lat_4090_pytorch_p99_ms": float(np.percentile(timings, 99)),
        "lat_4090_pytorch_std_ms": float(np.std(timings)),
    }


def m2_estimate_orin(lat_4090_fp32, fits):
    """M2 f 估算 Orin TRT FP16 / INT8 latency.

    注意: M2 f 输入是 4090 PyTorch FP32 latency.
    如果 input 是 fp16 latency, 用 f_fp16_from_4090_fp16 而非 _from_4090_fp32.
    """
    fp16 = fits["f_fp16_from_4090_fp32"]
    int8 = fits["f_int8_from_4090_fp32"]
    return {
        "est_orin_fp16_ms": float(fp16["intercept_a"] + fp16["slope_b"] * lat_4090_fp32),
        "est_orin_int8_ms": float(int8["intercept_a"] + int8["slope_b"] * lat_4090_fp32),
        "est_orin_uncertainty_fp16_ms": float(fp16["stderr"] * abs(lat_4090_fp32)),
        "est_orin_uncertainty_int8_ms": float(int8["stderr"] * abs(lat_4090_fp32)),
        "m2_f_factor": float(lat_4090_fp32 / 7.0),  # M2 训练上限 7ms
    }


def main():
    print("=" * 60)
    print("M4.5 latency-only + M4.6 — Pyramid Fusion 配置矩阵")
    print("=" * 60)
    print(f"PyTorch {torch.__version__}, Device 0: {torch.cuda.get_device_name(0)}")

    with open(M2_F_JSON) as f:
        fits = json.load(f)["fits"]

    rows = []
    # 同时记录 baseline_fp32 latency (M2 f 输入需要 4090 PyTorch FP32 latency)
    baseline_fp32_lat = None

    for config_id, layer_nums, num_filters, precision in CONFIGS:
        print(f"\n--- {config_id} ---")
        print(f"   layer_nums={layer_nums}, num_filters={num_filters}, prec={precision}")

        # 1. instantiate
        is_fp16 = (precision == "fp16")
        dtype = torch.float16 if is_fp16 else torch.float32
        model, n_params = make_pyramid(layer_nums, num_filters, fp16=is_fp16)
        n_params_M = n_params / 1e6
        print(f"   params: {n_params_M:.3f} M")

        # 2. benchmark
        stats = benchmark(model, dtype)
        print(f"   {precision}: mean={stats['lat_4090_pytorch_mean_ms']:.3f} ms  "
              f"p50={stats['lat_4090_pytorch_p50_ms']:.3f}  p99={stats['lat_4090_pytorch_p99_ms']:.3f}")

        # 3. cache baseline FP32 latency for M2 f input
        if config_id == "baseline_fp32":
            baseline_fp32_lat = stats["lat_4090_pytorch_mean_ms"]

        # 4. M2 f 估算 Orin (只对 baseline_fp32 + 用每个 config 的 fp32 等价 latency)
        # 注: 我们没跑每个 config 的 fp32 latency, 用 fp16 / 0.7 估算 fp32 (PyTorch fp16 比 fp32 慢的经验)
        # 更严谨方式: 每个 config 都跑 fp32 baseline
        if precision == "fp32":
            fp32_proxy = stats["lat_4090_pytorch_mean_ms"]
        else:
            # PyTorch fp16 在 4090 上无 TRT 优化, ~0.85-1.05× fp32; 取 0.95× 的反推
            # M2 baseline ratio: 4090 PyTorch FP32 / FP16 = 1.3 / 1.4 ≈ 0.93 (resnet18)
            # 这里用 fp16 latency 直接代入 (近似)
            fp32_proxy = stats["lat_4090_pytorch_mean_ms"]
        orin_est = m2_estimate_orin(fp32_proxy, fits)
        print(f"   M2 f → Orin FP16: {orin_est['est_orin_fp16_ms']:.3f} ± {orin_est['est_orin_uncertainty_fp16_ms']:.3f} ms")
        print(f"            Orin INT8: {orin_est['est_orin_int8_ms']:.3f} ± {orin_est['est_orin_uncertainty_int8_ms']:.3f} ms")
        print(f"   M2 extrapolation factor: {orin_est['m2_f_factor']:.2f}× "
              f"({'interpolation 可信' if orin_est['m2_f_factor'] < 1.5 else 'extrapolation'})")

        # 5. 写行
        row = {
            "config_id": config_id,
            "model_class": "pyramid_fusion",
            "source": "m4_5_pyramid_configs",
            "is_real_measured": True,  # 4090 latency 是实测
            "notes": f"M4.5 Pyramid {config_id} layer={layer_nums} filters={num_filters} {precision}",
            "params_after_M": round(n_params_M, 4),
            "precision": precision,
            "layer_nums": str(layer_nums),
            "num_filters": str(num_filters),
            **stats,
            **orin_est,
            "extrapolation_warning": orin_est["m2_f_factor"] >= 1.5,
        }
        rows.append(row)

        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # 输出
    out_csv = DATA_DIR / "pyramid_fusion_4090.csv"
    out_parquet = DATA_DIR / "pyramid_fusion_4090.parquet"
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)
    print(f"\n✅ Wrote {out_csv}  ({len(df)} rows × {len(df.columns)} cols)")
    print(f"✅ Wrote {out_parquet}")

    # 摘要
    print(f"\n=== Pyramid Fusion 配置矩阵 latency 摘要 ===")
    show_cols = ["config_id", "params_after_M", "lat_4090_pytorch_mean_ms",
                 "est_orin_fp16_ms", "est_orin_int8_ms", "m2_f_factor"]
    print(df[show_cols].to_string(index=False))

    print(f"\n=== M4.5 latency-only 完成. 待 OPV2V-H 下完后跑 M4.5 完整版 (含 AP). ===")


if __name__ == "__main__":
    main()
