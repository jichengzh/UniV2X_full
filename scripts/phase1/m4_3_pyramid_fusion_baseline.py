"""M4.3: HEAL PyramidFusion PyTorch baseline 4090 latency.

Minimal viable 路径 (不依赖 OPV2V 真实数据):
  - 直接用 lidar_pyramid.yaml fusion_backbone 配置 instantiate PyramidFusion
  - 喂 dummy BEV input tensor (1, 64, 256, 256) 测 4090 PyTorch CUDA Event latency
  - 输出: results/m4_3_pyramid_fusion_4090_latency.csv

PyramidFusion 配置 (来自 opencood/hypes_yaml/opv2v/LiDAROnly/lidar_pyramid.yaml):
  resnext: True
  layer_nums: [3, 5, 8]      # 3 stages
  layer_strides: [1, 2, 2]   # 多尺度
  num_filters: [64, 128, 256]
  upsample_strides: [1, 2, 4]
  num_upsample_filter: [128, 128, 128]
  anchor_number: 2
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 让 import opencood 能找到
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))

from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# 来自 lidar_pyramid.yaml fusion_backbone 节
PYRAMID_CONFIG = {
    "resnext": True,
    "layer_nums": [3, 5, 8],
    "layer_strides": [1, 2, 2],
    "num_filters": [64, 128, 256],
    "upsample_strides": [1, 2, 4],
    "num_upsample_filter": [128, 128, 128],
    "anchor_number": 2,
}

# 输入: BEV feature map (来自 PointPillar encoder 后)
# OPV2V LiDAR range cav_lidar = [-102.4, -102.4, -3, 102.4, 102.4, 1]
# voxel_size [0.4, 0.4, 4] → BEV size = 512 × 512
# 但这里是经过 backbone 下采样后的输入: 64 channels @ 256x256 (stride 2 from PointPillar scatter)
INPUT_SHAPE = (1, 64, 256, 256)


@torch.no_grad()
def benchmark_cuda_event(
    model: torch.nn.Module, dummy: torch.Tensor,
    n_warmup: int = 50, n_iter: int = 200,
) -> dict:
    """4090 PyTorch CUDA Event timing."""
    for _ in range(n_warmup):
        _ = model(dummy)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros(n_iter)
    for i in range(n_iter):
        starter.record()
        _ = model(dummy)
        ender.record()
        torch.cuda.synchronize()
        timings[i] = starter.elapsed_time(ender)

    return {
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p99_ms": float(np.percentile(timings, 99)),
        "std_ms": float(np.std(timings)),
        "n_iter": n_iter,
    }


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    print("=" * 60)
    print("M4.3 — Pyramid Fusion PyTorch baseline (4090)")
    print("=" * 60)

    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"Device 0: {torch.cuda.get_device_name(0)}")
    print(f"Pyramid config: {PYRAMID_CONFIG}")
    print(f"Input shape: {INPUT_SHAPE}")

    # 1. instantiate
    model = PyramidFusion(PYRAMID_CONFIG).cuda().eval()
    n_params = count_params(model)
    n_params_M = n_params / 1e6
    print(f"\n✅ PyramidFusion instantiated, params = {n_params_M:.2f} M")

    # 检查模型签名 (forward args)
    print(f"   forward signature: {list(model.forward.__code__.co_varnames[:5])}")

    # 2. dummy input + dry run
    dummy = torch.randn(*INPUT_SHAPE).cuda()
    # PyramidFusion.forward 签名: (spatial_features, record_len, affine_matrix, agent_modality_list, cam_crop_info)
    # 单 agent 简化: record_len=[1], affine_matrix 单位阵
    record_len = torch.tensor([1]).cuda()
    affine_matrix = torch.eye(2, 3).unsqueeze(0).unsqueeze(0).cuda()  # (1,1,2,3)
    agent_modality_list = ["m1"]

    try:
        out = model(dummy, record_len, affine_matrix, agent_modality_list, {})
        print(f"   forward dry-run OK, output type: {type(out).__name__}")
        if isinstance(out, (list, tuple)):
            print(f"   output[0].shape = {out[0].shape if hasattr(out[0], 'shape') else type(out[0])}")
    except Exception as e:
        print(f"   ⚠️ forward dry-run failed: {type(e).__name__}: {e}")
        print("   尝试使用 PyramidFusion.get_multiscale_feature (内部接口)")
        try:
            out = model.get_multiscale_feature(dummy)
            print(f"   get_multiscale_feature OK, returns {len(out)} scales")
        except Exception as e2:
            print(f"   ⚠️ inner API also failed: {e2}")
            return

    # 3. benchmark FP32
    rows = []
    for prec_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        m = PyramidFusion(PYRAMID_CONFIG).cuda().eval()
        if dtype == torch.float16:
            m = m.half()
        d = dummy.to(dtype)

        # 用 get_multiscale_feature (单 tensor 输入, 跳过 fusion 复杂度)
        @torch.no_grad()
        def _fwd():
            return m.get_multiscale_feature(d)

        # warmup
        for _ in range(50):
            _ = _fwd()
        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = np.zeros(200)
        for i in range(200):
            starter.record()
            _ = _fwd()
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender)

        stats = {
            "mean_ms": float(np.mean(timings)),
            "p50_ms": float(np.percentile(timings, 50)),
            "p99_ms": float(np.percentile(timings, 99)),
            "std_ms": float(np.std(timings)),
            "n_iter": 200,
        }
        print(f"\n  {prec_name}: mean={stats['mean_ms']:.3f} ms  "
              f"p50={stats['p50_ms']:.3f}  p99={stats['p99_ms']:.3f}  "
              f"std={stats['std_ms']:.3f}")
        rows.append({
            "model": "pyramid_fusion_lidar_pyramid",
            "platform": "rtx4090",
            "precision": prec_name,
            "framework": "pytorch_cuda_event",
            "n_params_M": round(n_params_M, 4),
            **stats,
        })
        del m
        torch.cuda.empty_cache()

    # 4. apply M2 f for Orin estimate
    M2_F = REPO_ROOT / "results/m2_latency_mapping_f.json"
    with open(M2_F, "r") as f:
        fits = json.load(f)["fits"]
    fp32_lat = rows[0]["mean_ms"]
    fp16_fit = fits["f_fp16_from_4090_fp32"]
    int8_fit = fits["f_int8_from_4090_fp32"]
    est_fp16 = fp16_fit["intercept_a"] + fp16_fit["slope_b"] * fp32_lat
    est_int8 = int8_fit["intercept_a"] + int8_fit["slope_b"] * fp32_lat
    print(f"\n=== M2 f-mapped Orin latency estimates ===")
    print(f"  4090 PyTorch FP32: {fp32_lat:.3f} ms")
    print(f"  est Orin TRT FP16: {est_fp16:.3f} ms (extrapolation factor {fp32_lat/7:.1f}× of M2 training)")
    print(f"  est Orin TRT INT8: {est_int8:.3f} ms")

    # 5. save
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "m4_3_pyramid_fusion_4090_latency.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Wrote {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
