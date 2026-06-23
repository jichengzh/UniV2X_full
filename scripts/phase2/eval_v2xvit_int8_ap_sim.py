"""eval_v2xvit_int8_ap_sim.py
Simulated INT8 PTQ AP eval for V2X-ViT on DAIR val 1789.

目的: 测量 simulated fake-quant INT8 对 V2X-ViT AP30/50/70 的影响，
      判断 INT8 是否给 DAIR 上的 V2X-ViT 一个真 AP trade-off (AP 悬崖)。

量化口径:
  - (a) W-only INT8: 对 Conv2d/Linear 权重做 per-tensor symmetric fake-quant
        (scale = max(|W|)/127, clamp to [-127,127], de-quant 回 fp32)
  - (b) W+A INT8: 权重同 (a) + 激活 per-tensor symmetric fake-quant
        (用前 N_CALIB 个 batch 在 val 上标定 scale = max(|A_abs|)/127)

注意: 这是 simulated fake-quant (纯 PyTorch FP32 推理路径, 权重 de-quant 后精度有损但不改变 dtype)。
      ★ 这不是 TRT INT8 真 build; 报告里必须标注 simulated_int8_fakequant。

量化覆盖范围:
  - backbone_m1 的所有 Conv2d (权重)
  - shrinker_m1 的所有 Conv2d (权重)
  - fusion_net 的所有 Linear (权重) — 覆盖 HMSA/MSwin/FFN/prior_feed
  - (W+A 模式) 以上所有层的输入激活

跳过规则 (记录):
  - LayerNorm: 只有 weight/bias(element-wise scale), 非 matrix op; 量化意义不大且结构不同, 跳过
  - BatchNorm: folded into Conv (weight已含), 跳过单独 BN weight
  - 检测到 NaN/Inf 则跳过该层量化 (记录到 skipped_layers)

用法:
  CUDA_VISIBLE_DEVICES=7 python scripts/phase2/eval_v2xvit_int8_ap_sim.py 2>&1 | tee logs/v2xvit_int8_ap_sim.log

输出:
  results/v2xvit_int8_ap_sim.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(HEAL_ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import torch_tensor_to_numpy, convert_format, compute_iou
from opencood.utils.box_utils import corner_to_center

# ------------------------------------------------------------------ config --

CKPT_DIR  = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                 "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE   = "net_epoch_bestval_at17.pth"
EPOCH_USED  = "bestval_at17"

N_SAMPLES    = 1789
N_CALIB      = 50       # W+A: 用 val 前 50 个 batch 标定激活 scale
N_BOOTSTRAP  = 1000
SEED         = 42

OUT_DIR  = REPO_ROOT / "results"
OUT_JSON = OUT_DIR / "v2xvit_int8_ap_sim.json"

# ----------------------------------------------------------------- helpers --

def fake_quant_tensor(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor fake-quant: round-to-nearest, clamp, de-quant back to fp32."""
    n_lvl = 2 ** (n_bits - 1) - 1   # 127 for int8
    abs_max = t.abs().max().item()
    if abs_max < 1e-10:
        return t
    scale = abs_max / n_lvl
    q = torch.clamp(torch.round(t / scale), -n_lvl, n_lvl)
    return q * scale


def quant_model_weights(model: nn.Module, n_bits: int = 8) -> Tuple[nn.Module, List[str]]:
    """
    In-place weight fake-quant for all Conv2d and Linear layers.
    Returns (modified model, list of quantized layer names).
    """
    quantized = []
    skipped   = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            # Safety check: skip if NaN/Inf
            if not torch.isfinite(w).all():
                skipped.append(f"{name}:non-finite-weight")
                continue
            module.weight.data = fake_quant_tensor(w, n_bits)
            quantized.append(name)
    print(f"[W-quant] Quantized {len(quantized)} layers; skipped {len(skipped)}", flush=True)
    if skipped:
        print(f"[W-quant] Skipped: {skipped[:5]}", flush=True)
    return model, skipped


# ---------------------------------------------------------------- W+A mode --

class ActFakeQuantWrapper(nn.Module):
    """Wrap a Conv2d or Linear to also fake-quant its input activation."""
    def __init__(self, orig: nn.Module, scale: float, n_bits: int = 8):
        super().__init__()
        self.orig    = orig
        self.scale   = scale
        self.n_bits  = n_bits
        self._is_conv = isinstance(orig, nn.Conv2d)

    def forward(self, x, *args, **kwargs):
        n_lvl = 2 ** (self.n_bits - 1) - 1
        xq = torch.clamp(torch.round(x / self.scale), -n_lvl, n_lvl) * self.scale
        return self.orig(xq, *args, **kwargs)


def calibrate_activation_scales(model: nn.Module, loader: DataLoader, device: str,
                                 n_calib: int = N_CALIB) -> Dict[str, float]:
    """
    Run forward on n_calib batches, collect max |activation| per layer.
    Uses forward hooks on Conv2d/Linear layers.
    Returns dict: layer_name -> scale (= max_abs / 127).
    """
    print(f"[A-calib] Collecting activation scales from {n_calib} batches...", flush=True)
    act_max: Dict[str, float] = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            val = inp[0].detach().abs().max().item()
            if not np.isfinite(val):
                return
            if name not in act_max:
                act_max[name] = val
            else:
                act_max[name] = max(act_max[name], val)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            if n_done >= n_calib:
                break
            batch = train_utils.to_device(batch, device)
            try:
                inference_utils.inference_intermediate_fusion(batch, model, dataset_ref[0])
            except Exception as e:
                print(f"[A-calib] batch {n_done} error: {e}", flush=True)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [A-calib] {n_done}/{n_calib}", flush=True)

    for h in hooks:
        h.remove()

    n_lvl = 127
    scales = {k: max(v / n_lvl, 1e-10) for k, v in act_max.items()}
    print(f"[A-calib] Collected scales for {len(scales)} layers", flush=True)
    return scales


def wrap_model_wa_quant(model: nn.Module, act_scales: Dict[str, float],
                        n_bits: int = 8) -> Tuple[nn.Module, List[str]]:
    """
    Wrap Conv2d/Linear with ActFakeQuantWrapper (activation fake-quant).
    Weights already fake-quantized in-place before calling this.
    """
    wrapped   = []
    not_found = []
    # We need to replace submodules by name path
    def _set_attr(root, path, new_mod):
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name in act_scales:
                scale = act_scales[name]
                _set_attr(model, name, ActFakeQuantWrapper(module, scale, n_bits))
                wrapped.append(name)
            else:
                not_found.append(name)

    print(f"[WA-wrap] Wrapped {len(wrapped)} layers; no-scale {len(not_found)}", flush=True)
    return model, not_found


# ---------------------------------------------------------------- AP eval  --

def compute_tp_errors_frame(pred_box_np, pred_score_np, gt_box_np):
    if gt_box_np is None or len(gt_box_np) == 0:
        return np.array([]), np.array([]), np.array([])
    if pred_box_np is None or len(pred_box_np) == 0:
        return np.array([]), np.array([]), np.array([])
    IOU_THRESH = 0.5
    score_order = np.argsort(-pred_score_np)
    pred_sorted = pred_box_np[score_order]
    pred_params = corner_to_center(pred_sorted, order='lwh')
    gt_params   = corner_to_center(gt_box_np,   order='lwh')
    pred_poly   = list(convert_format(pred_sorted))
    gt_poly     = list(convert_format(gt_box_np))
    remaining_gt = list(range(len(gt_box_np)))
    ate_l, ase_l, aoe_l = [], [], []
    for i in range(len(pred_sorted)):
        if not remaining_gt:
            break
        ious = compute_iou(pred_poly[i], [gt_poly[j] for j in remaining_gt])
        if not len(ious) or np.max(ious) < IOU_THRESH:
            continue
        best   = int(np.argmax(ious))
        gt_idx = remaining_gt.pop(best)
        p, g   = pred_params[i], gt_params[gt_idx]
        ate    = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        l_p, w_p, h_p = abs(p[3]), abs(p[4]), abs(p[5])
        l_g, w_g, h_g = abs(g[3]), abs(g[4]), abs(g[5])
        i_vol  = min(l_p, l_g)*min(w_p, w_g)*min(h_p, h_g)
        u_vol  = l_p*w_p*h_p + l_g*w_g*h_g - i_vol
        size_iou = i_vol / u_vol if u_vol > 1e-9 else 0.0
        delta  = abs(p[6]-g[6]) % np.pi
        ate_l.append(ate)
        ase_l.append(1.0 - size_iou)
        aoe_l.append(float(min(delta, np.pi - delta)))
    return np.array(ate_l), np.array(ase_l), np.array(aoe_l)


def bootstrap_ci(frame_ate, frame_ase, frame_aoe, B=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n   = len(frame_aoe)
    boot_ate, boot_ase, boot_aoe = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        def cat(lst):
            parts = [lst[i] for i in idx if len(lst[i]) > 0]
            return np.concatenate(parts) if parts else np.array([])
        a, s, o = cat(frame_ate), cat(frame_ase), cat(frame_aoe)
        boot_ate.append(float(np.mean(a)) if len(a) else float("nan"))
        boot_ase.append(float(np.mean(s)) if len(s) else float("nan"))
        boot_aoe.append(float(np.mean(o)) if len(o) else float("nan"))
    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    all_ate = np.concatenate([a for a in frame_ate if len(a) > 0])
    all_ase = np.concatenate([a for a in frame_ase if len(a) > 0])
    all_aoe = np.concatenate([a for a in frame_aoe if len(a) > 0])
    return {
        "mATE": float(np.mean(all_ate)) if len(all_ate) else float("nan"),
        "mATE_ci_lo": ci(boot_ate)[0], "mATE_ci_hi": ci(boot_ate)[1],
        "mASE": float(np.mean(all_ase)) if len(all_ase) else float("nan"),
        "mASE_ci_lo": ci(boot_ase)[0], "mASE_ci_hi": ci(boot_ase)[1],
        "mAOE": float(np.mean(all_aoe)) if len(all_aoe) else float("nan"),
        "mAOE_ci_lo": ci(boot_aoe)[0], "mAOE_ci_hi": ci(boot_aoe)[1],
        "n_tp": len(all_aoe),
    }


def run_eval(model: nn.Module, loader: DataLoader, device: str,
             label: str) -> Tuple[float, float, float, dict, float]:
    """Run AP eval loop. Returns (ap30, ap50, ap70, ci_dict, elapsed_secs)."""
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    frame_ate, frame_ase, frame_aoe = [], [], []
    n_done = 0
    t0 = time.time()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            if n_done >= N_SAMPLES:
                break
            batch = train_utils.to_device(batch, device)
            try:
                infer = inference_utils.inference_intermediate_fusion(
                    batch, model, dataset_ref[0])
            except Exception as e:
                print(f"[{label}] ERROR at sample {n_done}: {e}", flush=True)
                # Still count — add empty
                frame_ate.append(np.array([]))
                frame_ase.append(np.array([]))
                frame_aoe.append(np.array([]))
                n_done += 1
                continue

            pred_box   = infer["pred_box_tensor"]
            pred_score = infer["pred_score"]
            gt_box     = infer["gt_box_tensor"]

            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box,
                                           result_stat, iou_th)
            if pred_box is not None and gt_box is not None:
                a, s, o = compute_tp_errors_frame(
                    torch_tensor_to_numpy(pred_box),
                    torch_tensor_to_numpy(pred_score),
                    torch_tensor_to_numpy(gt_box))
            else:
                a, s, o = np.array([]), np.array([]), np.array([])
            frame_ate.append(a); frame_ase.append(s); frame_aoe.append(o)
            n_done += 1
            if n_done % 300 == 0:
                el = time.time() - t0
                print(f"  [{label}] {n_done}/{N_SAMPLES}  elapsed={el:.0f}s "
                      f"ETA={el/n_done*(N_SAMPLES-n_done):.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[{label}] Inference done {elapsed:.1f}s ({n_done} samples)", flush=True)

    tmp_dir = OUT_DIR / f"_ap_tmp_{label}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(tmp_dir))
    import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[{label}] Bootstrap CI...", flush=True)
    ci = bootstrap_ci(frame_ate, frame_ase, frame_aoe)
    print(f"[{label}] AP30={ap30:.4f} AP50={ap50:.4f} AP70={ap70:.4f}", flush=True)
    return float(ap30), float(ap50), float(ap70), ci, elapsed


# -------------------------------------------------------------------  main --

# Global dataset ref (used inside run_eval / calibrate as closure)
dataset_ref: List = []


def load_base_model(device="cpu") -> nn.Module:
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    parser_func = getattr(yaml_utils, hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    model = train_utils.create_model(hypes)
    epoch_path = CKPT_DIR / CKPT_FILE
    state = torch.load(str(epoch_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
        print("[load] WARNING: ckpt wrapped, unwrapped", flush=True)
    model.load_state_dict(state)
    return model, hypes


def main():
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("/home/jichengzhi/V2X/logs").mkdir(parents=True, exist_ok=True)

    print("[INT8-sim] === V2X-ViT simulated INT8 AP eval ===", flush=True)
    print(f"[INT8-sim] ckpt={CKPT_FILE}  epoch={EPOCH_USED}", flush=True)
    print(f"[INT8-sim] device={device}", flush=True)

    # ---- Build base model & dataset (reused across all 3 evals) ----
    print("[INT8-sim] Loading base model...", flush=True)
    model_base, hypes = load_base_model("cpu")
    model_base = model_base.to(device).eval()
    print(f"[INT8-sim] Model params: {sum(p.numel() for p in model_base.parameters()):,}", flush=True)

    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_ref.append(dataset)  # global closure for run_eval/calib
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, drop_last=False)

    # ============================================================
    # Pass 1: base (fp32, re-verify) — also verifies eval harness
    # ============================================================
    print("\n[INT8-sim] === Pass 1: base fp32 (re-verify) ===", flush=True)
    ap30_base, ap50_base, ap70_base, ci_base, t_base = run_eval(
        model_base, loader, device, "base_fp32")

    # ============================================================
    # Pass 2: W-only INT8 (weights fake-quant, activation fp32)
    # ============================================================
    print("\n[INT8-sim] === Pass 2: W-only INT8 ===", flush=True)
    # Deep copy to avoid polluting base model
    model_w = copy.deepcopy(model_base)
    model_w, skipped_w = quant_model_weights(model_w, n_bits=8)
    model_w = model_w.to(device).eval()

    # Reload DataLoader for fresh iteration
    loader2 = DataLoader(dataset, batch_size=1, num_workers=2,
                         collate_fn=dataset.collate_batch_test,
                         shuffle=False, drop_last=False)
    ap30_w, ap50_w, ap70_w, ci_w, t_w = run_eval(
        model_w, loader2, device, "W_only_INT8")

    # ============================================================
    # Pass 3: W+A INT8 (calibrate on val first)
    # ============================================================
    print("\n[INT8-sim] === Pass 3: W+A INT8 (calibrate...) ===", flush=True)
    model_wa = copy.deepcopy(model_base)
    # Step 1: weight fake-quant
    model_wa, skipped_wa_w = quant_model_weights(model_wa, n_bits=8)
    model_wa = model_wa.to(device).eval()

    # Step 2: calibrate activation scales from val (use fresh loader)
    loader_calib = DataLoader(dataset, batch_size=1, num_workers=2,
                              collate_fn=dataset.collate_batch_test,
                              shuffle=False, drop_last=False)
    act_scales = calibrate_activation_scales(model_wa, loader_calib, device, N_CALIB)

    # Step 3: wrap activations
    model_wa, skipped_wa_a = wrap_model_wa_quant(model_wa, act_scales, n_bits=8)
    model_wa = model_wa.to(device).eval()

    loader3 = DataLoader(dataset, batch_size=1, num_workers=2,
                         collate_fn=dataset.collate_batch_test,
                         shuffle=False, drop_last=False)
    ap30_wa, ap50_wa, ap70_wa, ci_wa, t_wa = run_eval(
        model_wa, loader3, device, "WA_INT8")

    # ============================================================
    # Assemble result
    # ============================================================
    result = {
        "experiment":     "v2xvit_simulated_int8_ap",
        "quant_method":   "simulated_int8_fakequant_pytorch",
        "ckpt":           str(CKPT_DIR / CKPT_FILE),
        "epoch_used":     EPOCH_USED,
        "n_samples":      N_SAMPLES,
        "n_calib_batches": N_CALIB,
        "gpu_used":       "RTX4090 GPU7 (CUDA_VISIBLE_DEVICES=7)",
        "caveat":         ("simulated fake-quant (PyTorch FP32 compute path, weight/activation "
                           "de-quantized before matmul); NOT TRT INT8 real build. "
                           "Latency NOT measured here."),

        # Pass 1
        "base_fp32": {
            "ap30": ap30_base, "ap50": ap50_base, "ap70": ap70_base,
            "mAOE": ci_base["mAOE"],
            "mAOE_ci_lo": ci_base["mAOE_ci_lo"], "mAOE_ci_hi": ci_base["mAOE_ci_hi"],
            "n_tp": ci_base["n_tp"],
            "elapsed_secs": t_base,
        },

        # Pass 2
        "W_only_INT8": {
            "ap30": ap30_w, "ap50": ap50_w, "ap70": ap70_w,
            "mAOE": ci_w["mAOE"],
            "mAOE_ci_lo": ci_w["mAOE_ci_lo"], "mAOE_ci_hi": ci_w["mAOE_ci_hi"],
            "n_tp": ci_w["n_tp"],
            "delta_ap30": round(ap30_w - ap30_base, 4),
            "delta_ap50": round(ap50_w - ap50_base, 4),
            "delta_ap70": round(ap70_w - ap70_base, 4),
            "elapsed_secs": t_w,
            "skipped_layers": skipped_w,
            "layers_quantized_info": "all Conv2d+Linear weights, per-tensor symmetric int8",
        },

        # Pass 3
        "WA_INT8": {
            "ap30": ap30_wa, "ap50": ap50_wa, "ap70": ap70_wa,
            "mAOE": ci_wa["mAOE"],
            "mAOE_ci_lo": ci_wa["mAOE_ci_lo"], "mAOE_ci_hi": ci_wa["mAOE_ci_hi"],
            "n_tp": ci_wa["n_tp"],
            "delta_ap30": round(ap30_wa - ap30_base, 4),
            "delta_ap50": round(ap50_wa - ap50_base, 4),
            "delta_ap70": round(ap70_wa - ap70_base, 4),
            "elapsed_secs": t_wa,
            "skipped_w_layers": skipped_wa_w,
            "skipped_a_layers_no_scale": skipped_wa_a,
            "layers_quantized_info": (
                "all Conv2d+Linear weights (per-tensor symmetric int8) + "
                f"activations (per-tensor symmetric int8, calib={N_CALIB} batches min-max)"),
        },

        # Quick verdict helper
        "_verdict_note": (
            "崩 threshold = ΔAP50 < -0.03 (远超 finetune 噪声 ±0.01); "
            "近无损 threshold = |ΔAP50| <= 0.02"
        ),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    # ---- Print summary ----
    print("\n[INT8-sim] ====== SUMMARY ======", flush=True)
    print(f"{'Config':<25} {'AP30':>8} {'AP50':>8} {'AP70':>8} {'ΔAP50':>8} {'ΔAP70':>8}", flush=True)
    print(f"{'base_fp32':<25} {ap30_base:>8.4f} {ap50_base:>8.4f} {ap70_base:>8.4f} {'—':>8} {'—':>8}", flush=True)
    print(f"{'W-only INT8':<25} {ap30_w:>8.4f} {ap50_w:>8.4f} {ap70_w:>8.4f} {ap50_w-ap50_base:>+8.4f} {ap70_w-ap70_base:>+8.4f}", flush=True)
    print(f"{'W+A INT8':<25} {ap30_wa:>8.4f} {ap50_wa:>8.4f} {ap70_wa:>8.4f} {ap50_wa-ap50_base:>+8.4f} {ap70_wa-ap70_base:>+8.4f}", flush=True)
    print(f"\n[INT8-sim] Saved: {OUT_JSON}", flush=True)

    # Verdict
    thresh_cliff = -0.03
    da50_w  = ap50_w  - ap50_base
    da50_wa = ap50_wa - ap50_base
    for tag, da50 in [("W-only INT8", da50_w), ("W+A INT8", da50_wa)]:
        if da50 < thresh_cliff:
            verdict = f"[崩] ΔAP50={da50:+.4f} < {thresh_cliff}"
        elif abs(da50) <= 0.02:
            verdict = f"[近无损] ΔAP50={da50:+.4f}"
        else:
            verdict = f"[中度降] ΔAP50={da50:+.4f} (between -0.03 and -0.02)"
        print(f"[INT8-sim] {tag}: {verdict}", flush=True)


if __name__ == "__main__":
    main()
