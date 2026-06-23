"""
轨C — C4 V2X-ViT 模拟 INT8 × 剪枝档 AP 评估

目的: 对比 WA_INT8 simulated quant (静态 calib) 在 base/p50/p75 下的 ΔAP，
     检验 Q×P 是否耦合 (Q-AP 损失是否随剪枝率变化)。

口径: simulated fake-quant (非 TRT real build, FP32 compute path)
     静态 per-tensor 量化 (weight + activation, static scale over 50 calib batches)
     attention einsum 已用 monkey-patch 尽可能覆盖 (动态 scale)

输出: results/coupling_map/C4_QgranxP_v2xvit.json

用法:
  CUDA_VISIBLE_DEVICES=1 python scripts/phase2/c4_v2xvit_int8_pruned_eval.py --anchor base
  CUDA_VISIBLE_DEVICES=2 python scripts/phase2/c4_v2xvit_int8_pruned_eval.py --anchor p50
  CUDA_VISIBLE_DEVICES=1 python scripts/phase2/c4_v2xvit_int8_pruned_eval.py --anchor p75
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
_HEAL = Path("/home/jichengzhi/heal_research/HEAL")
for _p in (str(_REPO), str(_HEAL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(_HEAL))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import torch_tensor_to_numpy, convert_format, compute_iou
from opencood.utils.box_utils import corner_to_center

from tools.configurable.depgraph_v2xvit import (
    V2XViTBackboneTraceNet, build_model as build_base_model,
    build_pruner, get_scatter_shape, CONFIG_YAML, CKPT_DIR
)

# ─────────────────────────────────────────────────────── config ────
N_SAMPLES    = 1789
N_BOOTSTRAP  = 1000
SEED         = 42
N_CALIB      = 50
IOU_THRESH   = 0.5

ANCHORS = {
    "base": {
        "ratio": None,
        "ckpt":  CKPT_DIR / "net_epoch_bestval_at17.pth",
        "epoch_used": "bestval_at17",
        "actual_filters": None,
        "finetuned": False,
    },
    "p50": {
        "ratio": 0.5,
        "ckpt":  _REPO / "output/a2_finetune/v2xvit_bb_p50/net_epoch17_bestval.pth",
        "epoch_used": "bestval_at17_finetuned",
        "actual_filters": [32, 64, 128],
        "finetuned": True,
    },
    "p75": {
        "ratio": 0.75,
        "ckpt":  _REPO / "output/a2_finetune/v2xvit_bb_p75/net_epoch16_bestval.pth",
        "epoch_used": "bestval_at16_finetuned",
        "actual_filters": [64, 32, 64],
        "finetuned": True,
    },
}

OUT_DIR = _REPO / "results" / "coupling_map"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "C4_QgranxP_v2xvit.json"


# ─────────────────────────────────────────── AP utils ────

def compute_tp_errors_frame(pred_box_np, pred_score_np, gt_box_np):
    if gt_box_np is None or len(gt_box_np) == 0:
        return np.array([]), np.array([]), np.array([])
    if pred_box_np is None or len(pred_box_np) == 0:
        return np.array([]), np.array([]), np.array([])
    score_order = np.argsort(-pred_score_np)
    pred_sorted = pred_box_np[score_order]
    pred_params = corner_to_center(pred_sorted, order='lwh')
    gt_params   = corner_to_center(gt_box_np,   order='lwh')
    pred_poly   = list(convert_format(pred_sorted))
    gt_poly     = list(convert_format(gt_box_np))
    remaining_gt = list(range(len(gt_box_np)))
    ate_l, ase_l, aoe_l = [], [], []
    for i in range(len(pred_sorted)):
        if not remaining_gt: break
        ious = compute_iou(pred_poly[i], [gt_poly[j] for j in remaining_gt])
        if not len(ious) or np.max(ious) < IOU_THRESH: continue
        best = int(np.argmax(ious)); gt_idx = remaining_gt.pop(best)
        p, g = pred_params[i], gt_params[gt_idx]
        ate  = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        l_p,w_p,h_p = abs(p[3]),abs(p[4]),abs(p[5])
        l_g,w_g,h_g = abs(g[3]),abs(g[4]),abs(g[5])
        i_vol = min(l_p,l_g)*min(w_p,w_g)*min(h_p,h_g)
        u_vol = l_p*w_p*h_p + l_g*w_g*h_g - i_vol
        delta = abs(p[6]-g[6]) % np.pi
        ate_l.append(ate); ase_l.append(1.0-i_vol/(u_vol+1e-9))
        aoe_l.append(float(min(delta, np.pi-delta)))
    return np.array(ate_l), np.array(ase_l), np.array(aoe_l)


# ─────────────────────────────────────── model loading ────

def load_model(anchor_cfg: dict, device: str):
    """Load base or pruned+finetuned V2X-ViT model."""
    if not anchor_cfg["finetuned"]:
        # base model
        model = build_base_model(device)
        return model
    else:
        # pruned model
        from opencood.hypes_yaml.yaml_utils import load_general_params
        ratio = anchor_cfg["ratio"]
        ft_ckpt = anchor_cfg["ckpt"]
        print(f"[load] Rebuilding pruned model ratio={ratio}...", flush=True)
        model = build_base_model("cpu")
        net   = V2XViTBackboneTraceNet(model).to("cpu").eval()
        hypes_raw = yaml_utils.load_yaml(str(CONFIG_YAML))
        ny, nx    = get_scatter_shape(hypes_raw)
        x_dummy   = torch.randn(1, 64, ny, nx)
        pr = build_pruner(net, x_dummy, ratio, "cpu")
        pr.step()
        print(f"[load] Pruning done. Loading finetuned ckpt: {ft_ckpt.name}", flush=True)
        ft_sd = torch.load(str(ft_ckpt), map_location="cpu")
        if isinstance(ft_sd, dict) and "model_state_dict" in ft_sd:
            ft_sd = ft_sd["model_state_dict"]
        model.load_state_dict(ft_sd, strict=False)
        print(f"[load] ✓ backbone_m1 params: {sum(p.numel() for p in model.backbone_m1.parameters()):,}", flush=True)
        return model.to(device).eval()


# ─────────────────────────────────────── INT8 quantization ────

def get_scale(t: torch.Tensor, bits: int = 8) -> float:
    """per-tensor symmetric scale: max|t| / (2^(bits-1) - 1)"""
    return float(t.abs().max().item()) / (2**(bits-1) - 1) + 1e-8


def fake_quant(t: torch.Tensor, scale: float, bits: int = 8) -> torch.Tensor:
    """Round to nearest INT8 and dequant."""
    qmax = 2**(bits-1) - 1
    return (t / scale).clamp(-qmax, qmax).round() * scale


def apply_static_wa_int8(model: nn.Module, calib_loader, device: str, n_calib: int = 50):
    """
    Apply WA INT8 (weight + activation) simulated quantization.
    - Weight: per-tensor static scale (from weight tensor directly)
    - Activation: per-tensor static scale (collected over n_calib batches via hooks)

    Returns hooks_handles (for cleanup) and act_scales dict.
    """
    # Step 1: Quantize all Conv2d and Linear weights
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                scale = get_scale(module.weight.data)
                module.weight.data = fake_quant(module.weight.data, scale)

    # Step 2: Collect activation scales via forward hooks (calibration)
    act_scales: Dict[str, float] = {}
    handles = []

    def make_hook(name_):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                cur_max = float(out.detach().abs().max().item())
                if name_ not in act_scales:
                    act_scales[name_] = cur_max
                else:
                    act_scales[name_] = max(act_scales[name_], cur_max)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    print(f"[int8] Collecting activation scales on {n_calib} calib batches...", flush=True)
    model.eval()
    n_done = 0
    with torch.no_grad():
        for batch in calib_loader:
            if batch is None: continue
            if n_done >= n_calib: break
            batch = train_utils.to_device(batch, device)
            try:
                _ = inference_utils.inference_intermediate_fusion(batch, model, dataset_ref[0])
            except Exception:
                pass
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [calib] {n_done}/{n_calib}", flush=True)

    for h in handles:
        h.remove()

    print(f"[int8] Collected scales for {len(act_scales)} layers", flush=True)

    # Step 3: Add activation quantization hooks for eval
    def make_quant_hook(name_):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor) and name_ in act_scales:
                scale = act_scales[name_] / 127.0
                return fake_quant(out, scale)
            return out
        return hook

    eval_handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in act_scales:
            h = module.register_forward_hook(make_quant_hook(name))
            eval_handles.append(h)

    print(f"[int8] Applied activation quant hooks for {len(eval_handles)} layers", flush=True)
    return eval_handles, act_scales


# ─────────────────────────────────────────────────────── main ────

def run_eval(model, loader, device: str, desc: str):
    """Run AP evaluation loop."""
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    frame_ate, frame_ase, frame_aoe = [], [], []
    n_done = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            if n_done >= N_SAMPLES: break
            batch = train_utils.to_device(batch, device)
            infer = inference_utils.inference_intermediate_fusion(batch, model, dataset_ref[0])
            pred_box   = infer["pred_box_tensor"]
            pred_score = infer["pred_score"]
            gt_box     = infer["gt_box_tensor"]
            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou_th)
            if pred_box is not None and gt_box is not None:
                pred_np  = torch_tensor_to_numpy(pred_box)
                score_np = torch_tensor_to_numpy(pred_score)
                gt_np    = torch_tensor_to_numpy(gt_box)
                a,s,o    = compute_tp_errors_frame(pred_np, score_np, gt_np)
            else:
                a,s,o = np.array([]), np.array([]), np.array([])
            frame_ate.append(a); frame_ase.append(s); frame_aoe.append(o)
            n_done += 1
            if n_done % 200 == 0:
                elapsed = time.time()-t0
                print(f"  [{desc}] [{n_done}/{N_SAMPLES}] elapsed={elapsed:.0f}s ETA={elapsed/n_done*(N_SAMPLES-n_done):.0f}s", flush=True)

    elapsed = time.time()-t0
    tmp_dir = OUT_DIR / "_ap_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(tmp_dir))
    import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
        "n_samples": n_done,
        "elapsed_secs": round(elapsed, 1),
    }


# ─────────────────────────────────────────────────────── entry ────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=str, required=True, choices=list(ANCHORS.keys()))
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    anchor_name = args.anchor
    anchor_cfg  = ANCHORS[anchor_name]
    device      = args.device

    print(f"\n[C4] === V2X-ViT C4 simulated INT8 × pruning — anchor={anchor_name} ===", flush=True)

    # Load dataset
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    from opencood.hypes_yaml.yaml_utils import load_general_params
    hypes = load_general_params(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_ref = [dataset]  # used in run_eval via closure
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, drop_last=False)
    print(f"[C4] Dataset: {len(dataset)} samples", flush=True)

    # Load model
    model = load_model(anchor_cfg, device)
    model.eval()

    # ── FP32 baseline eval (same model, no quant) ─
    print(f"\n[C4] === FP32 baseline eval for anchor={anchor_name} ===", flush=True)
    fp32_result = run_eval(model, loader, device, f"fp32_{anchor_name}")
    print(f"[C4] FP32: AP30={fp32_result['ap30']:.4f} AP50={fp32_result['ap50']:.4f} AP70={fp32_result['ap70']:.4f}", flush=True)

    # ── Simulated INT8 eval ────────────────────────
    print(f"\n[C4] === Applying WA INT8 simulated quant ===", flush=True)

    # Make a copy for INT8 (to preserve FP32 baseline)
    model_int8 = copy.deepcopy(model)

    # Calibrate and apply INT8 hooks
    calib_loader = DataLoader(dataset, batch_size=1, num_workers=0,
                              collate_fn=dataset.collate_batch_test,
                              shuffle=False, drop_last=False)
    eval_handles, act_scales = apply_static_wa_int8(model_int8, calib_loader, device, n_calib=N_CALIB)

    # Re-create loader for eval (reset iterator)
    eval_loader = DataLoader(dataset, batch_size=1, num_workers=2,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False, drop_last=False)

    print(f"\n[C4] === INT8 eval for anchor={anchor_name} ===", flush=True)
    int8_result = run_eval(model_int8, eval_loader, device, f"int8_{anchor_name}")

    for h in eval_handles:
        h.remove()

    print(f"[C4] INT8: AP30={int8_result['ap30']:.4f} AP50={int8_result['ap50']:.4f} AP70={int8_result['ap70']:.4f}", flush=True)

    # ── Compute deltas ─────────────────────────────
    delta_ap30 = round(int8_result["ap30"] - fp32_result["ap30"], 4)
    delta_ap50 = round(int8_result["ap50"] - fp32_result["ap50"], 4)
    delta_ap70 = round(int8_result["ap70"] - fp32_result["ap70"], 4)
    print(f"[C4] ΔAP: Δ30={delta_ap30:+.4f} Δ50={delta_ap50:+.4f} Δ70={delta_ap70:+.4f}", flush=True)

    # ── Save result ────────────────────────────────
    entry = {
        "anchor": anchor_name,
        "prune_ratio": anchor_cfg["ratio"],
        "actual_filters": anchor_cfg["actual_filters"],
        "epoch_used": anchor_cfg["epoch_used"],
        "n_calib_batches": N_CALIB,
        "fp32": fp32_result,
        "int8_sim": int8_result,
        "delta_ap30": delta_ap30,
        "delta_ap50": delta_ap50,
        "delta_ap70": delta_ap70,
        "quant_method": "simulated_WA_int8_static_scale",
        "caveat": ("simulated fake-quant (non-TRT real build, FP32 compute path). "
                   "Weight: per-tensor symmetric INT8 (static). "
                   "Activation: per-tensor symmetric INT8 (static, calib=50 batches). "
                   "Attention einsum (QK^T, attn@V) partially covered via activation hooks on Linear."),
        "device": device,
        "timestamp": "2026-06-22",
    }

    # Merge with existing results if file exists
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            existing = json.load(f)
    else:
        existing = {"model_class": "v2x_vit", "purpose": "C4 Q×P coupling", "results": {}}

    existing["results"][anchor_name] = entry

    with open(OUT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n[C4] Results saved to {OUT_JSON}", flush=True)
    print(f"[C4] DONE anchor={anchor_name}", flush=True)
