"""
export_onnx_lut_widths.py
Export ONNX for ALL new width configs needed for B1 Latency LUT.

New single-stage sweep variants (others held at base=[64,128,256]):
  s0_16  [16, 128, 256]   s0 axis, width=16
  s0_32  [32, 128, 256]   s0 axis, width=32
  s1_32  [64,  32, 256]   s1 axis, width=32
  s1_64  [64,  64, 256]   s1 axis, width=64
  s2_64  [64, 128,  64]   s2 axis, width=64
  s2_128 [64, 128, 128]   s2 axis, width=128

Validation combos (multi-stage, for additivity check):
  mix_a  [32,  96, 192]
  mix_b  [48,  64, 256]
  mix_c  [16, 128, 128]

Usage:
  python tools/export_onnx_lut_widths.py [--targets s0_16,s0_32,...]
  (default: export all 9 new configs)

Outputs to /home/jichengzhi/V2X/models/stage_a_cache/<name>_backbone.onnx
(Same dir + convention as existing iso_s0_backbone.onnx etc.)
"""
from __future__ import annotations
import sys, os, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))

from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion  # noqa

OUT_DIR = REPO_ROOT / "models/stage_a_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SHAPE = (2, 64, 128, 256)
OPSET = 17

# All new targets for B1 LUT
ALL_TARGETS = {
    # Single-stage sweep (others held at base)
    "s0_16":   [16,  128, 256],
    "s0_32":   [32,  128, 256],
    "s1_32":   [64,   32, 256],
    "s1_64":   [64,   64, 256],
    "s2_64":   [64,  128,  64],
    "s2_128":  [64,  128, 128],
    # Validation combos (multi-stage, for additivity check)
    "mix_a":   [32,   96, 192],
    "mix_b":   [48,   64, 256],
    "mix_c":   [16,  128, 128],
    # Extra combos for rank-flip coverage (alignment × size matrix)
    "mix_d":   [48,  128, 128],   # s0 misaligned, s2 medium aligned
    "mix_e":   [64,   96, 128],   # s0 aligned, s1 misaligned, s2 medium
    "mix_f":   [32,   64,  64],   # all aligned, all small (below p50)
}


class BackboneOnly(nn.Module):
    def __init__(self, pyramid_fusion: PyramidFusion):
        super().__init__()
        self.resnet = pyramid_fusion.resnet

    def forward(self, x: torch.Tensor):
        feats = self.resnet(x)
        return feats[0], feats[1], feats[2]


def build_backbone(num_filters: list[int]) -> BackboneOnly:
    cfg = {
        "anchor_number": 2,
        "layer_nums": [3, 5, 8],
        "layer_strides": [1, 2, 2],
        "num_filters": num_filters,
        "num_upsample_filter": [128, 128, 128],
        "resnext": True,
        "resnext_groups": 32,
        "width_per_group": 4,
        "upsample_strides": [1, 2, 4],
        "align_corners": False,
    }
    pf = PyramidFusion(cfg, input_channels=64)
    return BackboneOnly(pf)


def export_backbone(name: str, num_filters: list[int]) -> str:
    print(f"\n{'='*60}")
    print(f"[{name}] num_filters={num_filters}")
    print(f"{'='*60}")

    out_path = OUT_DIR / f"{name}_backbone.onnx"
    if out_path.exists():
        print(f"  EXISTS -> {out_path} ({out_path.stat().st_size/1e6:.2f} MB), skipping re-export")
        return str(out_path)

    bb = build_backbone(num_filters).eval()
    n_params = sum(p.numel() for p in bb.parameters())
    print(f"  params: {n_params:,} ({n_params * 4 / 1e6:.3f} MB FP32)")

    # Validate conv2 groups (HEAL formula: width = int(plane * wpg/64) * g)
    wpg = 4; g = 32
    for si, ln in enumerate(["layer0","layer1","layer2"]):
        layer = getattr(bb.resnet, ln)
        blk = layer[0]
        plane = num_filters[si]
        width = int(plane * wpg / 64) * g
        if width == 0:
            raise ValueError(f"[{name}] {ln}: width=0 for plane={plane} (plane<16 with wpg=4,g=32). "
                             f"Need plane>=16 for nonzero grouped conv.")
        expected = [width, width // g, 3, 3]
        actual = list(blk.conv2.weight.shape)
        assert actual == expected, f"{ln}.conv2 shape {actual} != expected {expected}"
    print(f"  conv2 group checks: PASS")

    torch.manual_seed(42)
    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        o0, o1, o2 = bb(dummy)
    print(f"  PT outputs: {list(o0.shape)}, {list(o1.shape)}, {list(o2.shape)}")
    for i, (actual_shape, exp_ch) in enumerate(zip([o0,o1,o2], num_filters)):
        assert list(actual_shape.shape)[1] == exp_ch, f"out[{i}] ch mismatch"

    print(f"  Exporting ONNX -> {out_path} (opset={OPSET})")
    with torch.no_grad():
        torch.onnx.export(
            bb, (dummy,), str(out_path),
            opset_version=OPSET,
            input_names=["spatial_features"],
            output_names=[
                f"/resnet/layer0/layer0.{2}/relu_2/Relu_output_0",
                f"/resnet/layer1/layer1.{4}/relu_2/Relu_output_0",
                f"/resnet/layer2/layer2.{7}/relu_2/Relu_output_0",
            ],
            dynamic_axes=None,
            do_constant_folding=True,
            verbose=False,
        )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  ONNX size: {size_mb:.2f} MB")

    import onnx, onnxruntime as ort
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    print(f"  ONNX checker: PASS")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"spatial_features": dummy.numpy()})
    pt_outs = [o0.numpy(), o1.numpy(), o2.numpy()]
    for i, (pt, ox) in enumerate(zip(pt_outs, ort_outs)):
        rel = np.abs(pt - ox).max() / max(np.abs(pt).max(), 1e-6)
        assert rel < 0.01, f"ORT numerical diff rel={rel:.2%}"
    print(f"  ORT numerical check: PASS")

    print(f"  DONE -> {out_path}  ({size_mb:.2f} MB)")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default="",
                        help="Comma-separated subset of target names (default: all)")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()] if args.targets else list(ALL_TARGETS.keys())

    results = []
    for name in targets:
        if name not in ALL_TARGETS:
            print(f"  UNKNOWN target '{name}', skip")
            results.append((name, None, f"unknown target"))
            continue
        nf = ALL_TARGETS[name]
        try:
            path = export_backbone(name, nf)
            results.append((name, nf, path))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append((name, nf, f"ERROR: {e}"))

    print(f"\n{'='*60}")
    print("SUMMARY:")
    all_ok = True
    for name, nf, status in results:
        ok = isinstance(status, str) and (status.endswith(".onnx") or "ERROR" not in status)
        if "ERROR" in str(status):
            all_ok = False
        print(f"  {'OK' if ok else 'FAIL'} {name} {nf}: {status}")
    print(f"\nAll OK: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
