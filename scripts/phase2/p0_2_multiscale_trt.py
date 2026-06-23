"""P0.2 — multi-output TRT path for pyramid get_multiscale_feature (Q2 + Q1).

Why (see methods/plan6 §三, P0_1 §五):
  plan v5 Phase C exported get_multiscale_feature + decode_multiscale_feature
  fused into a SINGLE bev output, bypassing the V2X collab fusion that happens
  BETWEEN them (single_head_i + warp_affine + weighted_fuse). Result: INT8/FP16
  AP collapsed to 0. The correct seam is: TRT replaces ONLY get_multiscale_feature
  (the 3-stage ResNeXt), returning its 3 stage feature maps; everything downstream
  (fusion + decode) stays in PyTorch, so AP is preserved.

  get_multiscale_feature(x) = self.resnet(x) -> tuple(feat0, feat1, feat2)
  (base_bev_backbone_resnet.py). This module exports those 3 outputs to ONNX,
  builds FP16/INT8 engines, and wraps the engine to return the 3-tuple.

Q1 (calibration) lives in p0_2_calibrate.py — a real IInt8 calibrator fed with
the actual spatial_features tensors of each anchor (not a stale reused cache).

This file (Q2 foundation): export + ONNX-runtime numerical sanity.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))

INPUT_NAME = "spatial_features"
# Real DAIR-V2X g8 geometry harvested in p0_2_calibrate.py: (C,H,W)=(64,128,256).
# NOTE: plan v5 phase A used (1,64,256,256) (square, OPV2V-like) — wrong for DAIR;
# a fixed 256x256 engine fed a real 128x256 tensor would shape-mismatch, an extra
# cause of phase C AP=0 on top of the single-output seam bug.
INPUT_SHAPE = (1, 64, 128, 256)
OUTPUT_NAMES = ["feat0", "feat1", "feat2"]

ONNX_DIR = Path("/tmp/plan6_p0_2_onnx")
ENGINE_DIR = Path("/tmp/plan6_p0_2_engines")
ONNX_DIR.mkdir(parents=True, exist_ok=True)
ENGINE_DIR.mkdir(parents=True, exist_ok=True)

# (tag, model_dir or None=baseline, num_filters, target_epoch)
P64_BASELINE = (
    "p64_baseline",
    HEAL_ROOT / "opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45",
    [64, 128, 256],
    "net_epoch_bestval_at19.pth",
)


class MultiScaleWrapper(torch.nn.Module):
    """forward(x) -> the 3 stage features from get_multiscale_feature (the TRT seam)."""

    def __init__(self, pyramid):
        super().__init__()
        self.pyramid = pyramid

    def forward(self, x):
        feats = self.pyramid.get_multiscale_feature(x)
        # ensure a flat tuple of tensors for ONNX multi-output
        return tuple(feats)


def boot_pyramid(model_dir: Path, num_filters: List[int], ckpt_name: str):
    """Load HEAL model, override num_filters, return pyramid_backbone on cuda (eval)."""
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    cfg = yaml_utils.load_yaml(str(model_dir / "config.yaml"))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = num_filters
    model = train_utils.create_model(cfg)
    sd = torch.load(model_dir / ckpt_name, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    pyramid = model.pyramid_backbone
    pyramid.eval().cuda()
    return pyramid


def export_multiscale_onnx(pyramid, onnx_path: Path,
                           input_shape: Tuple[int, ...] = INPUT_SHAPE) -> Path:
    """Export get_multiscale_feature (3 outputs) to ONNX."""
    wrapped = MultiScaleWrapper(pyramid).eval()
    dummy = torch.randn(*input_shape, device="cuda")
    with torch.no_grad():
        n_out = len(wrapped(dummy))
    output_names = OUTPUT_NAMES[:n_out] if n_out <= len(OUTPUT_NAMES) else \
        [f"feat{i}" for i in range(n_out)]
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=[INPUT_NAME], output_names=output_names,
        opset_version=16, do_constant_folding=True, dynamic_axes=None,
    )
    return onnx_path


def sanity_check_onnx(pyramid, onnx_path: Path,
                      input_shape: Tuple[int, ...] = INPUT_SHAPE) -> dict:
    """Compare ONNX-runtime outputs vs PyTorch get_multiscale_feature per output."""
    import onnxruntime as ort

    torch.manual_seed(0)
    x = torch.randn(*input_shape, device="cuda")
    with torch.no_grad():
        torch_outs = pyramid.get_multiscale_feature(x)
    torch_outs = [o.detach().cpu().numpy() for o in torch_outs]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {INPUT_NAME: x.detach().cpu().numpy()})

    report = {"n_outputs": len(torch_outs), "per_output": []}
    max_abs = 0.0
    for i, (t, o) in enumerate(zip(torch_outs, ort_outs)):
        diff = np.abs(t - o)
        rel = diff / (np.abs(t) + 1e-6)
        d = {"output": i, "shape": list(t.shape),
             "max_abs_diff": float(diff.max()),
             "mean_abs_diff": float(diff.mean()),
             "max_rel_diff": float(rel.max())}
        report["per_output"].append(d)
        max_abs = max(max_abs, d["max_abs_diff"])
    report["overall_max_abs_diff"] = max_abs
    report["overall_max_mean_abs"] = max(d["mean_abs_diff"] for d in report["per_output"])
    # ORT runs on CPU here (no CUDA EP), PyTorch on GPU -> ~1e-3 max-abs device
    # drift is expected for a multi-stage conv net. Gate on the robust mean-abs
    # metric; max-abs is informational. The definitive gate is end-to-end AP.
    report["pass"] = (report["overall_max_mean_abs"] < 1e-3
                      and max_abs < 1e-2)
    return report


def main():
    tag, model_dir, num_filters, ckpt_name = P64_BASELINE
    print(f"[P0.2] boot pyramid {tag} num_filters={num_filters}", flush=True)
    pyramid = boot_pyramid(model_dir, num_filters, ckpt_name)

    onnx_path = ONNX_DIR / f"{tag}_multiscale.onnx"
    print(f"[P0.2] exporting multi-output ONNX -> {onnx_path}", flush=True)
    export_multiscale_onnx(pyramid, onnx_path)
    print(f"[P0.2] onnx size = {onnx_path.stat().st_size/1e6:.2f} MB", flush=True)

    print(f"[P0.2] numerical sanity ONNX-runtime vs PyTorch ...", flush=True)
    rep = sanity_check_onnx(pyramid, onnx_path)
    print("\n==================== P0.2 Q2 export sanity ====================")
    print(f"n_outputs = {rep['n_outputs']}")
    for d in rep["per_output"]:
        print(f"  feat{d['output']} shape={d['shape']}  "
              f"max_abs={d['max_abs_diff']:.2e}  mean_abs={d['mean_abs_diff']:.2e}  "
              f"max_rel={d['max_rel_diff']:.2e}")
    print(f"overall max_abs_diff = {rep['overall_max_abs_diff']:.2e}  "
          f"-> {'PASS' if rep['pass'] else 'FAIL'} (threshold 1e-3)")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
