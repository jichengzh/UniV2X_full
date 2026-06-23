"""
export_l4_new_widths.py
Export 2 new Pyramid backbone ONNX files for L4 rank-flip pair expansion.

New configs:
  l4_wg7: [48, 128, 192]  -- partner of iso_s2=[64,128,192] (tuned 7.18x)
  l4_wg8: [48, 96,  128]  -- partner of mix_e=[64,96,128] (tuned 2.91x)

Both have s0=48 -> in_per_g=48/16=3 (NOT divisible by 4 = dp4a misaligned).

Output: /home/jichengzhi/V2X/models/stage_a_cache/{name}_backbone.onnx
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))

from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion  # noqa: E402

OUT_DIR = REPO_ROOT / "models/stage_a_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SHAPE = (2, 64, 128, 256)
OPSET = 17


class BackboneOnly(nn.Module):
    def __init__(self, pyramid_fusion: PyramidFusion):
        super().__init__()
        self.resnet = pyramid_fusion.resnet

    def forward(self, x: torch.Tensor):
        feats = self.resnet(x)
        return feats[0], feats[1], feats[2]


def build_backbone(num_filters: list) -> BackboneOnly:
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


def export_backbone(name, num_filters):
    print(f"\n{'='*60}")
    print(f"[{name}] num_filters={num_filters}")
    print(f"{'='*60}")

    bb = build_backbone(num_filters).eval()
    n_params = sum(p.numel() for p in bb.parameters())
    print(f"  params: {n_params:,} ({n_params * 4 / 1e6:.3f} MB FP32)")

    # Verify conv2 groups and s0 alignment
    s0 = num_filters[0]
    in_per_g = s0 // 16  # group size=16 in the ResNeXt conv2
    dp4a_aligned = (in_per_g % 4 == 0)
    print(f"  s0={s0} in_per_g={in_per_g} dp4a_aligned={dp4a_aligned}")

    for si, layer_name in enumerate(["layer0", "layer1", "layer2"]):
        layer = getattr(bb.resnet, layer_name)
        blk = layer[0]
        plane = num_filters[si]
        width = int(plane * 4 / 64) * 32
        print(f"  {layer_name}: plane={plane}, conv2 groups={blk.conv2.groups}, "
              f"weight={list(blk.conv2.weight.shape)}")

    torch.manual_seed(42)
    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        o0, o1, o2 = bb(dummy)
    print(f"  PT outputs: {list(o0.shape)}, {list(o1.shape)}, {list(o2.shape)}")

    # Verify output shapes
    assert list(o0.shape) == [2, num_filters[0], 128, 256], f"out0 shape error: {list(o0.shape)}"
    assert list(o1.shape) == [2, num_filters[1], 64, 128], f"out1 shape error: {list(o1.shape)}"
    assert list(o2.shape) == [2, num_filters[2], 32, 64], f"out2 shape error: {list(o2.shape)}"

    out_path = OUT_DIR / f"{name}_backbone.onnx"
    print(f"  Exporting ONNX -> {out_path} (opset={OPSET})")
    with torch.no_grad():
        torch.onnx.export(
            bb,
            (dummy,),
            str(out_path),
            opset_version=OPSET,
            input_names=["spatial_features"],
            output_names=[
                "/resnet/layer0/layer0.2/relu_2/Relu_output_0",
                "/resnet/layer1/layer1.4/relu_2/Relu_output_0",
                "/resnet/layer2/layer2.7/relu_2/Relu_output_0",
            ],
            dynamic_axes=None,
            do_constant_folding=True,
            verbose=False,
        )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  ONNX size: {size_mb:.2f} MB  path: {out_path}")

    import onnx
    import onnxruntime as ort

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    print(f"  ONNX checker: PASS")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    np_in = dummy.numpy()
    ort_outs = sess.run(None, {"spatial_features": np_in})
    pt_outs = [o0.numpy(), o1.numpy(), o2.numpy()]
    for i, (pt, ox) in enumerate(zip(pt_outs, ort_outs)):
        d = np.abs(pt - ox)
        rel = d.max() / max(np.abs(pt).max(), 1e-6)
        print(f"  ORT vs PT out[{i}]: max|delta|={d.max():.3e} rel={rel:.2%}")
        assert rel < 0.01, f"ORT numerical diff too large: rel={rel:.2%}"
    print(f"  ORT check: PASS")
    print(f"  [DONE] {name}  path={out_path}  size={size_mb:.2f}MB")
    return str(out_path)


TARGETS = [
    ("l4_wg7", [48, 128, 192]),   # partner of iso_s2=[64,128,192] (LUT: tuned=7243us, ratio=7.18x)
    ("l4_wg8", [48,  96, 128]),   # partner of mix_e=[64,96,128]   (LUT: tuned=14669us, ratio=2.91x)
    ("l4_pg4", [64,  96, 256]),   # re-export for clean iso_s1 confirmation (LUT: tuned=6910us, ratio=7.47x)
]


if __name__ == "__main__":
    print(f"Exporting {len(TARGETS)} L4 new pair ONNX models...")
    results = []
    for name, nf in TARGETS:
        try:
            path = export_backbone(name, nf)
            results.append((name, nf, path, "OK"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((name, nf, None, str(e)))

    print(f"\n{'='*60}")
    print("SUMMARY:")
    all_ok = True
    for name, nf, path, status in results:
        ok = status == "OK"
        if not ok:
            all_ok = False
        print(f"  {'OK' if ok else 'FAIL'} {name} {nf}: {path or status}")
    print(f"\nAll OK: {all_ok}")
    import sys
    sys.exit(0 if all_ok else 1)
