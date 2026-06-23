"""
轨C Step 1 — V2X-ViT ONNX export feasibility test (fixed)
目的: 测试 backbone_m1 + fusion_net 分别能否导出 ONNX (DAIR config)
输出: /home/jichengzhi/V2X/results/coupling_map/c4_onnx_export_diag.json

fixes:
  - backbone_m1 takes data_dict, not tensor → use wrapper
  - V2XTransformer input dim must be 259 (256 features + 3 prior encoding)
  - CUDA_VISIBLE_DEVICES=1 → device cuda:0 mapping
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils

CKPT_DIR  = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                 "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE   = "net_epoch_bestval_at17.pth"

OUT_DIR = REPO_ROOT / "results" / "coupling_map"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "c4_onnx_export_diag.json"
ONNX_BB  = OUT_DIR / "v2xvit_backbone_m1.onnx"
ONNX_FUS = OUT_DIR / "v2xvit_fusion_transformer.onnx"

DEVICE = "cuda:0"

print(f"[C4-diag] Using device {DEVICE}")
hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
model = train_utils.create_model(hypes)

ckpt_path = CKPT_DIR / CKPT_FILE
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt_sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(ckpt_sd, strict=False)
model.eval()
model.to(DEVICE)
print("[C4-diag] Model loaded OK")

results = {"exports": {}, "notes": []}


# ─────────────────────────────────────────────────────────────────────
# Test 1: backbone_m1 (BaseBEVBackbone) — wrapper to accept tensor
# ─────────────────────────────────────────────────────────────────────
print("\n[C4-diag] === Test 1: backbone_m1 ONNX export ===")


class BackboneWrapper(nn.Module):
    """Wrap BaseBEVBackbone dict-interface to accept tensor."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        data_dict = {"spatial_features": spatial_features}
        out = self.backbone(data_dict)
        return out["spatial_features_2d"]


backbone = BackboneWrapper(model.backbone_m1)
backbone.eval()

# DAIR range: [-102.4, -51.2, ..., 102.4, 51.2], voxel 0.4 → H=256, W=512
dummy_bev = torch.randn(1, 64, 256, 512, device=DEVICE)

try:
    with torch.no_grad():
        out_bb = backbone(dummy_bev)
        print(f"[C4-diag] backbone_m1 forward OK, output shape: {out_bb.shape}")

    t0 = time.time()
    torch.onnx.export(
        backbone,
        dummy_bev,
        str(ONNX_BB),
        opset_version=16,
        input_names=["spatial_features"],
        output_names=["spatial_features_2d"],
        dynamic_axes={"spatial_features": {0: "batch_size"}, "spatial_features_2d": {0: "batch_size"}},
        do_constant_folding=True,
        verbose=False,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(str(ONNX_BB)) / 1e6
    print(f"[C4-diag] backbone_m1 ONNX export OK in {elapsed:.1f}s, size={size_mb:.1f}MB")
    results["exports"]["backbone_m1"] = {
        "status": "OK",
        "onnx_path": str(ONNX_BB),
        "size_mb": round(size_mb, 2),
        "export_time_s": round(elapsed, 1),
        "input_shape": list(dummy_bev.shape),
        "output_shape": list(out_bb.shape),
    }
except Exception as e:
    tb = traceback.format_exc()
    print(f"[C4-diag] backbone_m1 ONNX export FAILED: {e}")
    print(f"[C4-diag] Traceback:\n{tb[:2000]}")
    results["exports"]["backbone_m1"] = {"status": "FAILED", "error": str(e)[:500], "traceback": tb[:1000]}

# ─────────────────────────────────────────────────────────────────────
# Test 2: V2XTransformer (fusion_net inner module)
# Input: (B, L, H, W, C+3) where C=256, so total 259
# ─────────────────────────────────────────────────────────────────────
print("\n[C4-diag] === Test 2: V2XTransformer ONNX export (C=259) ===")

fusion_net = model.fusion_net.fusion_net  # V2XTransformer
fusion_net.eval()

# C=256 (feature dim) + 3 (prior encoding) = 259 total
B, L, H, W, C_total = 1, 2, 32, 32, 259
dummy_x     = torch.randn(B, L, H, W, C_total, device=DEVICE)
dummy_mask  = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
dummy_scm   = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, L, 4, 4).to(DEVICE).contiguous()

try:
    with torch.no_grad():
        out_fus = fusion_net(dummy_x, dummy_mask, dummy_scm)
        print(f"[C4-diag] V2XTransformer forward OK, output shape: {out_fus.shape}")

    t0 = time.time()
    torch.onnx.export(
        fusion_net,
        (dummy_x, dummy_mask, dummy_scm),
        str(ONNX_FUS),
        opset_version=16,
        input_names=["x", "mask", "spatial_correction_matrix"],
        output_names=["fused_feature"],
        # NOTE: dynamic axes with variable L is tricky; use fixed shapes for now
        do_constant_folding=True,
        verbose=False,
    )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(str(ONNX_FUS)) / 1e6
    print(f"[C4-diag] V2XTransformer ONNX export OK in {elapsed:.1f}s, size={size_mb:.1f}MB")
    results["exports"]["v2xtransformer"] = {
        "status": "OK",
        "onnx_path": str(ONNX_FUS),
        "size_mb": round(size_mb, 2),
        "export_time_s": round(elapsed, 1),
        "input_shape": {"x": [B, L, H, W, C_total], "mask": [B, L], "scm": [B, L, 4, 4]},
        "output_shape": list(out_fus.shape),
        "notes": "Fixed shapes (B=1, L=2, H=32, W=32). Dynamic L not supported without tracing changes.",
    }
except Exception as e:
    tb = traceback.format_exc()
    print(f"[C4-diag] V2XTransformer ONNX export FAILED: {e}")
    print(f"[C4-diag] Traceback:\n{tb[:3000]}")
    results["exports"]["v2xtransformer"] = {"status": "FAILED", "error": str(e)[:500], "traceback": tb[:2000]}

# ─────────────────────────────────────────────────────────────────────
# Test 3: ONNX verification (if fusion export succeeded)
# ─────────────────────────────────────────────────────────────────────
if results["exports"].get("v2xtransformer", {}).get("status") == "OK":
    print("\n[C4-diag] === Test 3: ONNX vs PyTorch numerical match ===")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ONNX_FUS), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        x_np   = dummy_x.cpu().numpy()
        m_np   = dummy_mask.cpu().numpy()
        scm_np = dummy_scm.cpu().numpy()
        ort_out = sess.run(None, {"x": x_np, "mask": m_np, "spatial_correction_matrix": scm_np})[0]
        with torch.no_grad():
            pt_out = fusion_net(dummy_x, dummy_mask, dummy_scm).cpu().numpy()
        max_diff = float(np.abs(pt_out - ort_out).max())
        rel_diff = float(np.abs(pt_out - ort_out).max() / (np.abs(pt_out).max() + 1e-6))
        print(f"[C4-diag] ONNX vs PyTorch: max_abs_diff={max_diff:.6f}, max_rel_diff={rel_diff:.6f}")
        results["exports"]["v2xtransformer"]["onnxrt_match"] = {
            "max_abs_diff": round(max_diff, 8),
            "max_rel_diff": round(rel_diff, 8),
            "pass": bool(max_diff < 1e-2),
        }
    except ImportError:
        print("[C4-diag] onnxruntime not available")
        results["notes"].append("onnxruntime not installed - skip ORT match check")
    except Exception as e:
        print(f"[C4-diag] ORT check failed: {e}")
        results["notes"].append(f"ORT check failed: {str(e)[:200]}")

# Also check if backbone exported successfully, try ORT match
if results["exports"].get("backbone_m1", {}).get("status") == "OK":
    print("\n[C4-diag] === Test 1b: backbone ONNX vs PyTorch match ===")
    try:
        import onnxruntime as ort
        sess2 = ort.InferenceSession(str(ONNX_BB), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        bb_np = dummy_bev.cpu().numpy()
        ort_bb = sess2.run(None, {"spatial_features": bb_np})[0]
        with torch.no_grad():
            pt_bb = backbone(dummy_bev).cpu().numpy()
        max_diff_bb = float(np.abs(pt_bb - ort_bb).max())
        print(f"[C4-diag] backbone ONNX vs PyTorch: max_abs_diff={max_diff_bb:.6f}")
        results["exports"]["backbone_m1"]["onnxrt_match"] = {
            "max_abs_diff": round(max_diff_bb, 8),
            "pass": bool(max_diff_bb < 1e-3),
        }
    except Exception as e:
        print(f"[C4-diag] backbone ORT check failed: {e}")

# Save
results["timestamp"] = "2026-06-22"
results["model_class"] = "v2x_vit"
results["ckpt"] = str(CKPT_DIR / CKPT_FILE)
results["device"] = DEVICE

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[C4-diag] Results saved to {OUT_JSON}")
print("[C4-diag] Done.")
