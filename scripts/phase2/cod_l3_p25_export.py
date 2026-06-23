"""L3 Step-1: Export CoDriving p25 [48,96,192] backbone ONNX for TVM tuning.

p25 = 25% pruned backbone (75% of base channels per stage).
Base [64,128,256] -> p25 [48,96,192].

Key INT8 alignment fact (for downstream L3 Step-3):
  stage0 inner conv: Cin=48, K=48*9=432, 432÷32=13.5 → NOT ÷32 (INT8 WMMA FAIL)
  stage1 inner conv: Cin=96, K=96*9=864, 864÷32=27 ✓
  stage2 inner conv: Cin=192, K=192*9=1728, 1728÷32=54 ✓
  → p25 has stage0 INT8 WMMA misalignment; same as p75 (Cin=16, K=144, ÷32=4.5 ✗)
"""
import sys, os
sys.path.insert(0, "/data/jichengzhi_v2x/t2lib")
sys.path.insert(0, "/exdata/jichengzhi/V2Xverse_pyramid")

import torch

BATCH = 2
IN_CH = 64
H, W = 256, 512  # proxy shape (consistent with existing cod backbone ONNXes)
OUT_DIR = "/exdata/jichengzhi/s2_tvm/models/codriving_cache"


class BackboneWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.bb = backbone

    def forward(self, spatial_features):
        data_dict = {"spatial_features": spatial_features}
        out = self.bb(data_dict)
        if isinstance(out, dict):
            return out.get("spatial_features_2d", list(out.values())[0])
        return out


def export_backbone(s0, s1, s2, label, out_path):
    from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
    cfg = {
        "layer_nums": [3, 4, 5],
        "layer_strides": [2, 2, 2],
        "num_filters": [s0, s1, s2],
        "upsample_strides": [1, 2, 4],
        "num_upsample_filter": [128, 128, 128],
        "inplanes": IN_CH,
    }
    backbone = ResNetBEVBackbone(cfg, input_channels=IN_CH).float().eval()
    model = BackboneWrapper(backbone).float().eval()
    dummy = torch.zeros(BATCH, IN_CH, H, W)

    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["spatial_features"],
            output_names=["backbone_output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={"spatial_features": {0: "batch"}},
        )
    print(f"[EXPORT] {label} [{s0},{s1},{s2}] -> {out_path}")

    import onnx
    m = onnx.load(out_path)
    inits = {n.name for n in m.graph.initializer}
    inputs = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim]
              for i in m.graph.input if i.name not in inits}
    print(f"[VERIFY] {label}: inputs={inputs} n_ops={len(m.graph.node)}")

    # Check INT8 alignment for each stage
    print(f"[INT8 ALIGNMENT CHECK] {label}:")
    for stage_i, cin in enumerate([s0, s1, s2]):
        k = cin * 9  # 3x3 conv inner dim
        aligned = (k % 32 == 0)
        print(f"  stage{stage_i}: Cin={cin}, K={k}, K÷32={'OK ✓' if aligned else f'FAIL ✗ (mod={k%32})'}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    # p25 = 25% pruning = 75% of base channels
    out_path = f"{OUT_DIR}/p25_backbone.onnx"
    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists")
    else:
        export_backbone(48, 96, 192, "p25", out_path)
    print("[EXPORT] ALL DONE")
