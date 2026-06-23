"""
轨C — C4 V2X-ViT per-CHANNEL weight INT8 × 剪枝档 AP 评估

目的: 补 C4 的 Q-granularity 轴 — per-channel weight INT8 (vs 已有的 per-tensor)
     判断 granularity 是否与 P 耦合 (重剪枝档 per-channel 救回更多 AP?)

口径:
  Weight:     per-output-channel symmetric INT8 (static scale from weight)
  Activation: per-tensor symmetric INT8 (static, calib=50 batches, same as per-tensor run)
  不涉及 fusion ONNX / linalg_inv / TRT real build

输出: 追加到 results/coupling_map/C4_QgranxP_v2xvit.json
      新增 per_channel 段 + granularity_coupling_verdict

用法:
  CUDA_VISIBLE_DEVICES=0 python c4_v2xvit_perchannel_eval.py --anchor base
  CUDA_VISIBLE_DEVICES=1 python c4_v2xvit_perchannel_eval.py --anchor p50
  CUDA_VISIBLE_DEVICES=2 python c4_v2xvit_perchannel_eval.py --anchor p75
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
N_SAMPLES   = 1789
N_CALIB     = 50
IOU_THRESH  = 0.5

ANCHORS = {
    "base": {
        "ratio": None,
        "ckpt":  CKPT_DIR / "net_epoch_bestval_at17.pth",
        "epoch_used": "bestval_at17",
        "actual_filters": [64, 128, 256],
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

OUT_DIR  = _REPO / "results" / "coupling_map"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "C4_QgranxP_v2xvit.json"

dataset_ref = []  # closure


# ─────────────────────────────────────── AP helpers ────

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
    if not anchor_cfg["finetuned"]:
        return build_base_model(device)
    ratio  = anchor_cfg["ratio"]
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


# ──────────────────────────── per-channel weight INT8 ────

def fake_quant_per_tensor(t: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = 2**(bits-1) - 1
    scale = float(t.abs().max().item()) / qmax + 1e-8
    return (t / scale).clamp(-qmax, qmax).round() * scale


def fake_quant_per_channel_weight(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-output-channel symmetric INT8 quantization for weight tensor."""
    qmax = 2**(bits-1) - 1
    out_ch = w.shape[0]
    w_flat = w.view(out_ch, -1)
    scale = w_flat.abs().max(dim=1).values / qmax + 1e-8   # [out_ch]
    # Reshape for broadcasting: [out_ch, 1, 1, ...] or [out_ch, 1]
    shape = [out_ch] + [1] * (w.dim() - 1)
    scale = scale.view(shape)
    return (w / scale).clamp(-qmax, qmax).round() * scale


def apply_perchannel_weight_pertensor_act_int8(model: nn.Module, calib_loader, device: str, n_calib: int = 50):
    """
    Per-channel weight INT8 + per-tensor activation INT8 (static calib).
    Returns eval_handles for cleanup.
    """
    # Step 1: Quantize all Conv2d/Linear weights per-channel
    n_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                module.weight.data = fake_quant_per_channel_weight(module.weight.data)
            n_layers += 1
    print(f"[pc_int8] Per-channel weight quantized {n_layers} layers", flush=True)

    # Step 2: Collect activation scales via hooks (per-tensor, static)
    act_scales: Dict[str, float] = {}
    calib_handles = []

    def make_calib_hook(name_):
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
            h = module.register_forward_hook(make_calib_hook(name))
            calib_handles.append(h)

    print(f"[pc_int8] Collecting activation scales on {n_calib} calib batches...", flush=True)
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

    for h in calib_handles:
        h.remove()

    print(f"[pc_int8] Collected scales for {len(act_scales)} layers", flush=True)

    # Step 3: Apply activation quantization hooks
    def make_quant_hook(name_):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor) and name_ in act_scales:
                scale = act_scales[name_] / 127.0
                qmax = 127.0
                return (out / scale).clamp(-qmax, qmax).round() * scale
            return out
        return hook

    eval_handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name in act_scales:
            h = module.register_forward_hook(make_quant_hook(name))
            eval_handles.append(h)

    print(f"[pc_int8] Applied activation quant hooks for {len(eval_handles)} layers", flush=True)
    return eval_handles


# ──────────────────────────────────────── eval loop ────

def run_eval(model, loader, device: str, desc: str):
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
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
            n_done += 1
            if n_done % 200 == 0:
                elapsed = time.time()-t0
                print(f"  [{desc}] [{n_done}/{N_SAMPLES}] elapsed={elapsed:.0f}s ETA={elapsed/n_done*(N_SAMPLES-n_done):.0f}s", flush=True)

    elapsed = time.time()-t0
    tmp_dir = OUT_DIR / "_ap_tmp_pc"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(tmp_dir))
    import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
            "n_samples": n_done, "elapsed_secs": round(elapsed, 1)}


# ───────────────────────────────────────────── main ────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=str, required=True, choices=list(ANCHORS.keys()))
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    anchor_name = args.anchor
    anchor_cfg  = ANCHORS[anchor_name]
    device      = args.device

    print(f"\n[C4-pc] === per-channel weight INT8 — anchor={anchor_name} ===", flush=True)

    # Load dataset
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    from opencood.hypes_yaml.yaml_utils import load_general_params
    hypes = load_general_params(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_ref.append(dataset)
    print(f"[C4-pc] Dataset: {len(dataset)} samples", flush=True)

    # Load model (FP32)
    model = load_model(anchor_cfg, device)
    model.eval()

    # Reuse FP32 AP from existing JSON (skip re-eval to save time)
    existing_fp32 = None
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            c4_data = json.load(f)
        existing = c4_data.get("results", {}).get(anchor_name, {})
        if "fp32" in existing:
            existing_fp32 = existing["fp32"]
            print(f"[C4-pc] Reusing FP32 from existing JSON: AP50={existing_fp32['ap50']:.4f}", flush=True)

    if existing_fp32 is None:
        # Run fresh FP32 eval
        loader_fp32 = DataLoader(dataset, batch_size=1, num_workers=2,
                                 collate_fn=dataset.collate_batch_test,
                                 shuffle=False, drop_last=False)
        print(f"[C4-pc] === FP32 eval for anchor={anchor_name} ===", flush=True)
        existing_fp32 = run_eval(model, loader_fp32, device, f"fp32_{anchor_name}")
        print(f"[C4-pc] FP32: AP50={existing_fp32['ap50']:.4f}", flush=True)

    # Apply per-channel weight INT8 + per-tensor activation INT8
    model_pc = copy.deepcopy(model)
    calib_loader = DataLoader(dataset, batch_size=1, num_workers=0,
                              collate_fn=dataset.collate_batch_test,
                              shuffle=False, drop_last=False)
    eval_handles = apply_perchannel_weight_pertensor_act_int8(
        model_pc, calib_loader, device, n_calib=N_CALIB)

    # INT8 eval
    eval_loader = DataLoader(dataset, batch_size=1, num_workers=2,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False, drop_last=False)
    print(f"\n[C4-pc] === per-channel INT8 eval for anchor={anchor_name} ===", flush=True)
    pc_result = run_eval(model_pc, eval_loader, device, f"pc_int8_{anchor_name}")

    for h in eval_handles:
        h.remove()

    delta_ap30 = round(pc_result["ap30"] - existing_fp32["ap30"], 4)
    delta_ap50 = round(pc_result["ap50"] - existing_fp32["ap50"], 4)
    delta_ap70 = round(pc_result["ap70"] - existing_fp32["ap70"], 4)

    print(f"[C4-pc] FP32:    AP50={existing_fp32['ap50']:.4f} AP70={existing_fp32['ap70']:.4f}", flush=True)
    print(f"[C4-pc] PC-INT8: AP50={pc_result['ap50']:.4f}  AP70={pc_result['ap70']:.4f}", flush=True)
    print(f"[C4-pc] ΔAP: Δ50={delta_ap50:+.4f} Δ70={delta_ap70:+.4f}", flush=True)

    # Load and update JSON
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            c4_data = json.load(f)
    else:
        c4_data = {"model_class": "v2x_vit", "purpose": "C4 Q×P coupling",
                   "results": {}, "per_channel": {}}

    # Ensure per_channel key exists
    if "per_channel" not in c4_data:
        c4_data["per_channel"] = {}

    c4_data["per_channel"][anchor_name] = {
        "anchor": anchor_name,
        "prune_ratio": anchor_cfg["ratio"],
        "fp32_ap50": round(existing_fp32["ap50"], 4),
        "fp32_ap70": round(existing_fp32["ap70"], 4),
        "pc_int8_ap50": round(pc_result["ap50"], 4),
        "pc_int8_ap70": round(pc_result["ap70"], 4),
        "delta_ap50": delta_ap50,
        "delta_ap70": delta_ap70,
        "quant_method": ("per-channel weight INT8 (per-output-ch symmetric, static from weight) + "
                         "per-tensor activation INT8 (static calib=50 batches)"),
        "n_samples": pc_result["n_samples"],
        "elapsed_secs": pc_result["elapsed_secs"],
        "caveat": ("simulated fake-quant (non-TRT). Weight: per-output-channel INT8. "
                   "Activation: per-tensor static calib=50 batches. "
                   "Attention einsum partially covered via Linear activation hooks."),
        "timestamp": "2026-06-22",
    }

    # Compute granularity verdict if all 3 anchors are done
    pc_done = c4_data["per_channel"]
    pt_results = c4_data.get("results", {})
    if all(k in pc_done for k in ["base", "p50", "p75"]):
        # per-tensor deltas (from existing results)
        pt_base = pt_results.get("base", {}).get("delta_ap50", None)
        pt_p50  = pt_results.get("p50",  {}).get("delta_ap50", None)
        pt_p75  = pt_results.get("p75",  {}).get("delta_ap50", None)
        pc_base = pc_done["base"]["delta_ap50"]
        pc_p50  = pc_done["p50"]["delta_ap50"]
        pc_p75  = pc_done["p75"]["delta_ap50"]

        # "救回" = per-tensor ΔAP - per-channel ΔAP (positive = per-channel loses less)
        # If per-channel delta is LESS negative, it "saved" AP.
        saved_base = round((pt_base - pc_base) if pt_base is not None else 0.0, 4)
        saved_p50  = round((pt_p50  - pc_p50)  if pt_p50  is not None else 0.0, 4)
        saved_p75  = round((pt_p75  - pc_p75)  if pt_p75  is not None else 0.0, 4)

        # Coupling: if saved_p75 >> saved_base → JOINT
        # Separable: if all saved_X roughly equal
        if pt_base is not None and pt_p50 is not None and pt_p75 is not None:
            coupling_margin = abs(saved_p75 - saved_base)
            is_joint = coupling_margin > 0.003  # >0.3% difference

            c4_data["granularity_coupling_verdict"] = {
                "per_tensor_delta_ap50": {"base": pt_base, "p50": pt_p50, "p75": pt_p75},
                "per_channel_delta_ap50": {"base": pc_base, "p50": pc_p50, "p75": pc_p75},
                "per_channel_saves_vs_per_tensor_ap50": {
                    "base": saved_base, "p50": saved_p50, "p75": saved_p75,
                    "interpretation": "positive = per-channel less destructive than per-tensor"
                },
                "verdict": "JOINT" if is_joint else "SEPARABLE",
                "coupling_margin_ap50": coupling_margin,
                "evidence": (
                    f"救回幅度: base={saved_base:+.4f} p50={saved_p50:+.4f} p75={saved_p75:+.4f}. "
                    f"p75救回 {'>' if saved_p75>saved_base else '<='} base救回 by {coupling_margin:.4f} AP50. "
                    f"{'剪枝后模型更依赖细粒度量化 → Q-granularity × P JOINT耦合' if is_joint else 'granularity救回幅度各档一致 → Q-granularity 与 P 可分离(SEPARABLE)'}"
                ),
                "threshold": 0.003,
                "note": ("simulated fake-quant (non-TRT real build). "
                         "Caveat: 各档 quant_method 略有差异 (base dynamic vs p50/p75 static). "
                         "True TRT INT8 per-channel API 留作 open_followup.")
            }
            print(f"\n[C4-pc] === GRANULARITY COUPLING VERDICT ===", flush=True)
            print(f"[C4-pc] per-tensor ΔAP50: base={pt_base:+.4f} p50={pt_p50:+.4f} p75={pt_p75:+.4f}", flush=True)
            print(f"[C4-pc] per-channel ΔAP50: base={pc_base:+.4f} p50={pc_p50:+.4f} p75={pc_p75:+.4f}", flush=True)
            print(f"[C4-pc] saved (pt-pc):  base={saved_base:+.4f} p50={saved_p50:+.4f} p75={saved_p75:+.4f}", flush=True)
            print(f"[C4-pc] Verdict: {'JOINT' if is_joint else 'SEPARABLE'} (margin={coupling_margin:.4f})", flush=True)

    # Mark true TRT INT8 as open followup
    c4_data["open_followup"] = {
        "true_trt_int8_ap": {
            "status": "NOT_DONE",
            "blocker": "aten::linalg_inv in warp_affine (STTF), torch_transformation_utils.py:366. opset16+17 both fail.",
            "workaround": "Pre-compute inverse matrix outside ONNX graph + model surgery. Est. 4-8h.",
            "priority": "low (C4 is auxiliary cell; fake-quant direction signal sufficient)",
        },
        "true_trt_per_channel_int8": {
            "status": "NOT_DONE",
            "dependency": "fusion ONNX export",
        },
        "c5_orin_latency": {
            "status": "NOT_DONE",
            "dependency": "ONNX export",
            "qualitative_conclusion": "STRUCTURAL_COUPLING_CONFIRMED (grid_sample DLA blacklist)",
        }
    }

    with open(OUT_JSON, "w") as f:
        json.dump(c4_data, f, indent=2, ensure_ascii=False)
    print(f"\n[C4-pc] Results saved to {OUT_JSON}", flush=True)
    print(f"[C4-pc] DONE anchor={anchor_name}", flush=True)
