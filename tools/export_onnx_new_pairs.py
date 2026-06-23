"""
export_onnx_new_pairs.py
导出 5 个新宽度 PyramidFusion ResNeXt backbone ONNX，用于 L4 rank-flip pair 扩展。

新配置:
  wg_pair4: [48, 96, 256]
  wg_pair5: [48, 32, 128]
  pg_pair5: [64, 32, 128]
  wg_pair6: [48, 64, 192]
  pg_pair6: [64, 64, 192]

输出: /home/jichengzhi/V2X/models/stage_a_cache/{name}_backbone.onnx
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


def export_backbone(name: str, num_filters: list[int]):
    print(f"\n{'='*60}")
    print(f"[{name}] num_filters={num_filters}")
    print(f"{'='*60}")

    bb = build_backbone(num_filters).eval()
    n_params = sum(p.numel() for p in bb.parameters())
    print(f"  params: {n_params:,} ({n_params * 4 / 1e6:.3f} MB FP32)")

    # Verify conv2 groups formula: width = int(plane * 4/64) * 32
    for si, layer_name in enumerate(["layer0", "layer1", "layer2"]):
        layer = getattr(bb.resnet, layer_name)
        blk = layer[0]
        plane = num_filters[si]
        width = int(plane * 4 / 64) * 32
        print(f"  [{name}] {layer_name}: plane={plane}, width={width}, conv2.groups={blk.conv2.groups}")
        assert blk.conv2.groups == 32
        if width > 0:
            assert list(blk.conv2.weight.shape) == [width, width // 32, 3, 3], \
                f"  shape mismatch: {list(blk.conv2.weight.shape)} vs [{width},{width//32},3,3]"
        else:
            print(f"  WARNING: width=0 for plane={plane} (wpg=4, g=32)")

    torch.manual_seed(42)
    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        o0, o1, o2 = bb(dummy)
    print(f"  PT outputs: {list(o0.shape)}, {list(o1.shape)}, {list(o2.shape)}")

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
                f"/resnet/layer0/layer0.{2}/relu_2/Relu_output_0",
                f"/resnet/layer1/layer1.{4}/relu_2/Relu_output_0",
                f"/resnet/layer2/layer2.{7}/relu_2/Relu_output_0",
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
        print(f"  ORT vs PT out[{i}]: shape={list(ox.shape)} max|delta|={d.max():.3e} rel={rel:.2%}")
        assert rel < 0.01

    print(f"  ORT check: PASS")
    print(f"  [DONE] {name}  path={out_path}  size={size_mb:.2f}MB")
    return str(out_path)


# 5 new pair configurations
TARGETS = [
    ("wg_pair4", [48, 96, 256]),
    ("wg_pair5", [48, 32, 128]),
    ("pg_pair5", [64, 32, 128]),
    ("wg_pair6", [48, 64, 192]),
    ("pg_pair6", [64, 64, 192]),
]


if __name__ == "__main__":
    print(f"Exporting {len(TARGETS)} new pair ONNX models...")
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
    sys.exit(0 if all_ok else 1)
