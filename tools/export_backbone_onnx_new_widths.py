"""
export_backbone_onnx_new_widths.py
生成 4 个新宽度 PyramidFusion ResNeXt backbone 子图 ONNX, 用于 W_g 探针 Phase 1 TVM latency 测量.

仅需 latency-correct 架构 (随机权重), 不需要 finetune.

输出位置: /home/jichengzhi/V2X/models/stage_a_cache/{p75,iso_s0,iso_s1,iso_s2}_backbone.onnx

格式 (与 base/p50/trap25_backbone.onnx 完全一致):
  input:  spatial_features [2, 64, 128, 256]
  output: 3 stage relu features
          - layer0 output [2, s0, 128, 256]
          - layer1 output [2, s1,  64, 128]
          - layer2 output [2, s2,  32,  64]
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

# 在 import PyramidFusion 之前, 确保 Bottleneck.expansion 会被 PyramidFusion.__init__ 设为 1
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion  # noqa: E402

OUT_DIR = REPO_ROOT / "models/stage_a_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SHAPE = (2, 64, 128, 256)  # [agents, channels, H, W]
OPSET = 17


class BackboneOnly(nn.Module):
    """wrapper: only runs resnet.get_multiscale_feature, returns 3 stage outputs.

    input:  spatial_features [2, 64, 128, 256]
    output: (layer0_relu, layer1_relu, layer2_relu)
    """
    def __init__(self, pyramid_fusion: PyramidFusion):
        super().__init__()
        self.resnet = pyramid_fusion.resnet

    def forward(self, x: torch.Tensor):
        # ResNetModified._forward_impl 返回 [layer0_out, layer1_out, layer2_out]
        feats = self.resnet(x)   # list of 3 tensors
        return feats[0], feats[1], feats[2]


def build_backbone(num_filters: list[int]) -> BackboneOnly:
    """直接实例化 PyramidFusion (设 Bottleneck.expansion=1), 取其 resnet, 包装为 BackboneOnly."""
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


def dump_layer_shapes(bb: BackboneOnly, label: str):
    """Print layer0/1/2 block0 conv shapes for verification."""
    resnet = bb.resnet
    for si, layer_name in enumerate(["layer0", "layer1", "layer2"]):
        layer = getattr(resnet, layer_name)
        blk = layer[0]  # block0
        print(f"  [{label}] {layer_name}.0:")
        print(f"    conv1: W{list(blk.conv1.weight.shape)} groups={blk.conv1.groups}")
        print(f"    conv2: W{list(blk.conv2.weight.shape)} groups={blk.conv2.groups}")
        print(f"    conv3: W{list(blk.conv3.weight.shape)} groups={blk.conv3.groups}")
        ds = blk.downsample
        if ds is None:
            print(f"    downsample: None (identity)")
        else:
            ds_conv = ds[0]
            print(f"    downsample: Conv W{list(ds_conv.weight.shape)}")


def export_backbone(name: str, num_filters: list[int]):
    print(f"\n{'='*60}")
    print(f"[{name}] num_filters={num_filters}")
    print(f"{'='*60}")

    # 1. Build
    bb = build_backbone(num_filters).eval()
    n_params = sum(p.numel() for p in bb.parameters())
    print(f"  params: {n_params:,} ({n_params * 4 / 1e6:.3f} MB FP32)")

    # 2. Dump layer shapes for verification
    dump_layer_shapes(bb, name)

    # 3. Check downsample logic
    s0 = num_filters[0]
    blk0 = bb.resnet.layer0[0]
    has_ds = blk0.downsample is not None
    expected_ds = (s0 != 64)
    assert has_ds == expected_ds, (
        f"stage0 downsample logic wrong: s0={s0}, has_ds={has_ds}, expected_ds={expected_ds}"
    )
    print(f"  stage0 downsample: {'exists (s0={})'.format(s0) if has_ds else 'identity (s0=64)'} [OK]")

    # 4. Check conv2 groups
    for si, layer_name in enumerate(["layer0", "layer1", "layer2"]):
        layer = getattr(bb.resnet, layer_name)
        blk = layer[0]
        assert blk.conv2.groups == 32, f"{layer_name}.0.conv2 groups={blk.conv2.groups} != 32"
        plane = num_filters[si]
        # width = int(plane * wpg/64) * g = int(plane * 4/64) * 32
        width = int(plane * 4 / 64) * 32
        assert list(blk.conv2.weight.shape) == [width, width // 32, 3, 3], (
            f"{layer_name}.0.conv2 W{list(blk.conv2.weight.shape)} != [{width},{width//32},3,3]"
        )
    print(f"  conv2 group/shape assertions: PASS")

    # 5. PyTorch forward
    torch.manual_seed(42)
    dummy = torch.randn(*INPUT_SHAPE)
    with torch.no_grad():
        o0, o1, o2 = bb(dummy)
    print(f"  PT outputs: {list(o0.shape)}, {list(o1.shape)}, {list(o2.shape)}")
    expected_shapes = [
        [2, num_filters[0], 128, 256],
        [2, num_filters[1], 64, 128],
        [2, num_filters[2], 32, 64],
    ]
    for i, (actual, expected) in enumerate(zip([list(o0.shape), list(o1.shape), list(o2.shape)], expected_shapes)):
        assert actual == expected, f"output[{i}] shape {actual} != expected {expected}"
    print(f"  output shape assertions: PASS")

    # 6. ONNX export
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

    # 7. ORT verification
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    print(f"  ONNX checker: PASS")

    # Verify output shapes from ONNX graph
    onnx_out_shapes = []
    for o in m.graph.output:
        shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
        onnx_out_shapes.append(shape)
        print(f"  ONNX output: {o.name[:60]} shape={shape}")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    np_in = dummy.numpy()
    ort_outs = sess.run(None, {"spatial_features": np_in})

    # Numerical check vs PyTorch
    pt_outs = [o0.numpy(), o1.numpy(), o2.numpy()]
    for i, (pt, ox) in enumerate(zip(pt_outs, ort_outs)):
        d = np.abs(pt - ox)
        rel = d.max() / max(np.abs(pt).max(), 1e-6)
        print(f"  ORT vs PT out[{i}]: shape={list(ox.shape)} max|Δ|={d.max():.3e} rel={rel:.2%}")
        assert rel < 0.01, f"ORT numerical diff too large: rel={rel:.2%}"
    print(f"  ORT numerical check: PASS")

    # Assert channel dims
    for i, (ort_out, exp_ch) in enumerate(zip(ort_outs, num_filters)):
        assert ort_out.shape[1] == exp_ch, (
            f"ORT out[{i}] channel {ort_out.shape[1]} != expected {exp_ch}"
        )
    print(f"  channel assertions: PASS")

    print(f"  ✅ [{name}] DONE  path={out_path}  size={size_mb:.2f}MB")
    return str(out_path)


TARGETS = [
    ("p75",    [16, 32, 64]),
    ("iso_s0", [48, 128, 256]),
    ("iso_s1", [64, 96, 256]),
    ("iso_s2", [64, 128, 192]),
]


if __name__ == "__main__":
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
        print(f"  {'✅' if ok else '❌'} {name} {nf}: {path or status}")
    print(f"\nAll OK: {all_ok}")
    sys.exit(0 if all_ok else 1)
