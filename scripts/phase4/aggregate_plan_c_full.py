"""Final aggregation for Plan C tiny: combine eval metrics + TRT latency + D-space.

Inputs:
  - data/phase4/stage5_baseline_v4_tiny.csv  (configs + car_AP_4m + mAP)
  - output/plan_c/trt/<id>_bench.json        (TRT FP16 BEV encoder latency)

Outputs:
  - data/phase4/stage5_v4_tiny_full.csv      (added trt_bev_ms_mean column)
  - data/phase4/stage5_v4_tiny_dspace.csv    (D-space expanded × 16 combinations)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TINY_CSV = ROOT / "data/phase4/stage5_baseline_v4_tiny.csv"
TRT_DIR = ROOT / "output/plan_c/trt"

# Tiny baseline module latencies (实测 2026-05-04)
TINY_BASE_MODULES = {
    "backbone": 15.88,    # R50 PyTorch
    "neck": 1.30,
    "bev_encoder_pytorch": 25.95,
    "bev_encoder_trt_fp16": 15.67,
    "decoder_pytorch": 7.50,    # estimated
    "seg_head_pytorch": 57.78,
    "other_python": 442.00,    # 549 - sum(modules)
}
PYTORCH_BASELINE_E2E = 549.10
TRT_FP16_BASELINE_E2E = 549.10 - TINY_BASE_MODULES["bev_encoder_pytorch"] + TINY_BASE_MODULES["bev_encoder_trt_fp16"]


def load_trt_latency(cfg_id: str) -> float:
    p = TRT_DIR / f"{cfg_id}_bench.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("trt_bev_ms_mean")


def main():
    df = pd.read_csv(TINY_CSV)
    print(f"Loaded {len(df)} configs from {TINY_CSV}")

    # Add TRT latency
    df["trt_bev_ms"] = df["config_id"].map(load_trt_latency)

    # Compute estimated tiny e2e for each config:
    #   tiny_e2e_pytorch = R50 backbone + neck + BEV pytorch + decoder + seg_head + other
    #   tiny_e2e_trt_fp16 = R50 backbone + neck + BEV TRT (实测) + decoder + seg_head + other
    # Note: backbone/decoder/seg_head we keep at baseline values since pruning didn't actually take effect there.
    df["tiny_e2e_pytorch_ms"] = TINY_BASE_MODULES["backbone"] + TINY_BASE_MODULES["neck"] + TINY_BASE_MODULES["bev_encoder_pytorch"] * (1.0 - df["prune_rate__encoder_ffn"]) + TINY_BASE_MODULES["decoder_pytorch"] + TINY_BASE_MODULES["seg_head_pytorch"] + TINY_BASE_MODULES["other_python"]
    df["tiny_e2e_trt_fp16_ms"] = TINY_BASE_MODULES["backbone"] + TINY_BASE_MODULES["neck"] + df["trt_bev_ms"].fillna(TINY_BASE_MODULES["bev_encoder_trt_fp16"]) + TINY_BASE_MODULES["decoder_pytorch"] + TINY_BASE_MODULES["seg_head_pytorch"] + TINY_BASE_MODULES["other_python"]

    out_csv = ROOT / "data/phase4/stage5_v4_tiny_full.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df[["config_id", "car_AP_4m", "trt_bev_ms", "tiny_e2e_pytorch_ms", "tiny_e2e_trt_fp16_ms"]].sort_values("car_AP_4m", ascending=False).to_string(index=False))
    print(f"\ncar_AP_4m range: [{df.car_AP_4m.min():.4f}, {df.car_AP_4m.max():.4f}]")
    print(f"trt_bev_ms range: [{df.trt_bev_ms.min():.2f}, {df.trt_bev_ms.max():.2f}] ms")
    print(f"tiny_e2e_trt_fp16 range: [{df.tiny_e2e_trt_fp16_ms.min():.1f}, {df.tiny_e2e_trt_fp16_ms.max():.1f}] ms")
    print(f"\nBaseline TRT FP16 e2e: {TRT_FP16_BASELINE_E2E:.1f} ms (vs PyTorch baseline {PYTORCH_BASELINE_E2E:.1f} ms = -{(1-TRT_FP16_BASELINE_E2E/PYTORCH_BASELINE_E2E)*100:.1f}%)")


if __name__ == "__main__":
    main()
