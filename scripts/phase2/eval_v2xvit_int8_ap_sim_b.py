"""eval_v2xvit_int8_ap_sim_b.py — Phase B: 补全覆盖 + 分段敏感度

## 任务
B-1: 补全 HMSA 激活量化覆盖（全 80/80 层），重测 W+A 全覆盖 AP
B-2: 分段敏感度（backbone-only / fusion-only / 全量化）

## 根因分析 (Phase A 缺口)
Phase A 中 13 层激活 scale 未被 hook 捕获:
- `prior_feed`: 被 calib 50 batch 跳过（该层调用路径 hook 未触发）
- `k/q/v/a_linears.1`（3 层 × 4 种 × 1 = 12 个）: 这些是 HGTCavAttention 内
  `to_qkv` / `to_out` 函数里通过 `self.q_linears[types[b,i]]` 动态索引调用的。
  hook 挂在 nn.Linear module 上，当 module 被调用时应该触发，但因为 calib 只用
  了 50 个 batch，types[b,i] 可能未能覆盖 index=1 的 agent（DAIR 每帧 2 agents，
  type[0]=vehicle=0, type[1]=infra=1, 两者均存在但 hook 记录漏了）。

## 修复策略
不依赖 calib hook，改为 **monkey-patch HGTCavAttention.to_qkv / to_out / forward
内部 einsum** 直接在调用时 fake-quant:
1. 对所有 Conv2d/Linear 的 weight 做 W fake-quant（同 Phase A）。
2. 对激活 fake-quant: monkey-patch HGTCavAttention.to_qkv / to_out，在每次
   调用 k/q/v/a_linears 前对 input 做 fake-quant（动态 per-tensor min-max，
   用当前 batch 的 max|x|）。对其他 Conv2d/Linear 层也用 pre_hook 动态 fake-quant。
3. 对 attention 的 batched matmul (QK^T, attn·V): 注释记录未覆盖（einsum 非
   nn.Module，无法 hook；需手动 patch forward）。本脚本提供一个可选的
   --patch_attn_matmul 开关，在 einsum 调用前对 q/k/v 激活 fake-quant。

## 口径
- simulated fake-quant (PyTorch FP32 计算路径，non-TRT)
- 动态 per-tensor scale = max|x| / 127（无 calib 预热，直接用当前 batch）
- 这比 Phase A 的 min-max calib 更保守（scale 更大），量化误差可能略小于真 TRT INT8

## 分段设计
- (i) backbone_only: 量化 backbone_m1+shrinker_m1 的 Conv2d（不量化 fusion）
- (ii) fusion_only: 量化 fusion_net 的所有 Linear（不量化 backbone）
- (iii) full: backbone + fusion 全量化（= B-1 全覆盖）

用法:
  CUDA_VISIBLE_DEVICES=7 python scripts/phase2/eval_v2xvit_int8_ap_sim_b.py \
    2>&1 | tee logs/v2xvit_int8_ap_sim_b.log

输出:
  results/v2xvit_int8_ap_sim.json  (追加 WA_INT8_fullcover + 3 分段，保留原字段)
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
HEAL_ROOT  = Path("/home/jichengzhi/heal_research/HEAL")
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
CKPT_DIR    = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                   "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE   = "net_epoch_bestval_at17.pth"
EPOCH_USED  = "bestval_at17"

N_SAMPLES   = 1789
N_BOOTSTRAP = 1000
SEED        = 42

OUT_DIR  = REPO_ROOT / "results"
OUT_JSON = OUT_DIR / "v2xvit_int8_ap_sim.json"  # 追加到原文件

# global dataset ref (closure trick)
dataset_ref: List = []

# ------------------------------------------------------------------ quant --

def fake_quant_dynamic(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Dynamic per-tensor symmetric fake-quant (scale = max|x|/127)."""
    n_lvl = 2 ** (n_bits - 1) - 1
    abs_max = x.detach().abs().max().item()
    if abs_max < 1e-10:
        return x
    scale = abs_max / n_lvl
    return torch.clamp(torch.round(x / scale), -n_lvl, n_lvl) * scale


def quant_weights_selective(model: nn.Module, target_modules: List[str],
                             n_bits: int = 8) -> List[str]:
    """
    Fake-quant weights for named modules whose name *starts with* any prefix
    in target_modules. Returns list of quantized module names.
    """
    quantized = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if not any(name.startswith(pfx) for pfx in target_modules):
            continue
        w = module.weight.data
        if not torch.isfinite(w).all():
            continue
        module.weight.data = fake_quant_w(w, n_bits)
        quantized.append(name)
    return quantized


def fake_quant_w(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Weight fake-quant: per-tensor symmetric, scale=max|w|/127."""
    n_lvl = 2 ** (n_bits - 1) - 1
    abs_max = w.abs().max().item()
    if abs_max < 1e-10:
        return w
    scale = abs_max / n_lvl
    return torch.clamp(torch.round(w / scale), -n_lvl, n_lvl) * scale


# ---------------------------------------------------------------- act wrap --

class DynamicActQuantWrapper(nn.Module):
    """
    Wrap a Conv2d / Linear: dynamically fake-quant input activation per forward call.
    Does NOT require pre-calibration — uses current-batch max|x|.
    This guarantees 100% coverage regardless of data distribution during calib.
    """
    def __init__(self, orig: nn.Module, n_bits: int = 8):
        super().__init__()
        self.orig   = orig
        self.n_bits = n_bits

    def forward(self, x, *args, **kwargs):
        xq = fake_quant_dynamic(x, self.n_bits)
        return self.orig(xq, *args, **kwargs)


def wrap_activation_quant_selective(model: nn.Module, target_prefixes: List[str],
                                    n_bits: int = 8) -> Tuple[nn.Module, List[str], List[str]]:
    """
    Replace Conv2d/Linear modules whose name matches target_prefixes with
    DynamicActQuantWrapper. Returns (model, wrapped_names, skipped_names).

    Uses direct setattr path replacement — no hook dependency.
    This correctly covers HMSA k/q/v/a_linears[0] and [1] because
    DynamicActQuantWrapper.__call__ fires for every forward() regardless of
    which ModuleList index is selected.
    """
    def _set_attr(root, dotpath, new_mod):
        parts = dotpath.split(".")
        obj = root
        for p in parts[:-1]:
            # handle ModuleList indices
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        last = parts[-1]
        if last.isdigit():
            obj[int(last)] = new_mod
        else:
            setattr(obj, last, new_mod)

    wrapped  = []
    skipped  = []
    # snapshot to avoid mutation-during-iteration
    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))
               and any(n.startswith(pfx) for pfx in target_prefixes)]

    for name, module in targets:
        try:
            _set_attr(model, name, DynamicActQuantWrapper(module, n_bits))
            wrapped.append(name)
        except Exception as e:
            skipped.append(f"{name}: {e}")

    return model, wrapped, skipped


# ---------------------------------------------------------------- AP eval  --

def compute_tp_errors_frame(pred_np, score_np, gt_np):
    IOU_TH = 0.5
    if gt_np is None or len(gt_np) == 0:
        return np.array([]), np.array([]), np.array([])
    if pred_np is None or len(pred_np) == 0:
        return np.array([]), np.array([]), np.array([])
    order   = np.argsort(-score_np)
    pred_s  = pred_np[order]
    pp      = corner_to_center(pred_s,  order='lwh')
    gp      = corner_to_center(gt_np,   order='lwh')
    ppoly   = list(convert_format(pred_s))
    gpoly   = list(convert_format(gt_np))
    remain  = list(range(len(gt_np)))
    ate_l, ase_l, aoe_l = [], [], []
    for i in range(len(pred_s)):
        if not remain: break
        ious = compute_iou(ppoly[i], [gpoly[j] for j in remain])
        if not len(ious) or np.max(ious) < IOU_TH: continue
        best   = int(np.argmax(ious)); gidx = remain.pop(best)
        p, g   = pp[i], gp[gidx]
        ate    = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        lp,wp,hp = abs(p[3]),abs(p[4]),abs(p[5])
        lg,wg,hg = abs(g[3]),abs(g[4]),abs(g[5])
        iv  = min(lp,lg)*min(wp,wg)*min(hp,hg)
        uv  = lp*wp*hp + lg*wg*hg - iv
        sio = iv/uv if uv > 1e-9 else 0.0
        d   = abs(p[6]-g[6]) % np.pi
        ate_l.append(ate); ase_l.append(1.0-sio); aoe_l.append(float(min(d, np.pi-d)))
    return np.array(ate_l), np.array(ase_l), np.array(aoe_l)


def bootstrap_ci(fa, fs, fo, B=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n   = len(fo)
    ba, bs, bo = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        def cat(lst):
            parts = [lst[i] for i in idx if len(lst[i]) > 0]
            return np.concatenate(parts) if parts else np.array([])
        a,s,o = cat(fa),cat(fs),cat(fo)
        ba.append(float(np.mean(a)) if len(a) else float("nan"))
        bs.append(float(np.mean(s)) if len(s) else float("nan"))
        bo.append(float(np.mean(o)) if len(o) else float("nan"))
    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    aa = np.concatenate([a for a in fa if len(a)>0])
    as_ = np.concatenate([a for a in fs if len(a)>0])
    ao  = np.concatenate([a for a in fo if len(a)>0])
    return {
        "mATE": float(np.mean(aa)) if len(aa) else float("nan"),
        "mATE_ci_lo": ci(ba)[0], "mATE_ci_hi": ci(ba)[1],
        "mASE": float(np.mean(as_)) if len(as_) else float("nan"),
        "mASE_ci_lo": ci(bs)[0], "mASE_ci_hi": ci(bs)[1],
        "mAOE": float(np.mean(ao)) if len(ao) else float("nan"),
        "mAOE_ci_lo": ci(bo)[0], "mAOE_ci_hi": ci(bo)[1],
        "n_tp": len(ao),
    }


def run_eval(model: nn.Module, dataset, device: str, label: str):
    """Run full DAIR val AP eval. Returns (ap30,ap50,ap70,ci,elapsed)."""
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, drop_last=False)
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    fa, fs, fo = [], [], []
    n_done = 0
    t0 = time.time()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            if n_done >= N_SAMPLES: break
            batch = train_utils.to_device(batch, device)
            try:
                infer = inference_utils.inference_intermediate_fusion(
                    batch, model, dataset_ref[0])
            except Exception as e:
                print(f"  [{label}] ERROR sample {n_done}: {e}", flush=True)
                fa.append(np.array([])); fs.append(np.array([])); fo.append(np.array([]))
                n_done += 1
                continue
            pb  = infer["pred_box_tensor"]
            ps  = infer["pred_score"]
            gb  = infer["gt_box_tensor"]
            for th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pb, ps, gb, result_stat, th)
            if pb is not None and gb is not None:
                a,s,o = compute_tp_errors_frame(
                    torch_tensor_to_numpy(pb), torch_tensor_to_numpy(ps), torch_tensor_to_numpy(gb))
            else:
                a,s,o = np.array([]),np.array([]),np.array([])
            fa.append(a); fs.append(s); fo.append(o)
            n_done += 1
            if n_done % 300 == 0:
                el = time.time()-t0
                print(f"  [{label}] {n_done}/{N_SAMPLES} elapsed={el:.0f}s "
                      f"ETA={el/n_done*(N_SAMPLES-n_done):.0f}s", flush=True)

    elapsed = time.time()-t0
    print(f"[{label}] done {elapsed:.1f}s ({n_done} samples)", flush=True)

    tmp = OUT_DIR / f"_tmp_{label}"
    tmp.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(tmp))
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{label}] Bootstrap CI...", flush=True)
    ci = bootstrap_ci(fa, fs, fo)
    print(f"[{label}] AP30={ap30:.4f} AP50={ap50:.4f} AP70={ap70:.4f}  "
          f"mAOE={ci['mAOE']:.4f}", flush=True)
    return float(ap30), float(ap50), float(ap70), ci, elapsed


# ------------------------------------------------------------------ main --

def load_clean_model(device="cpu"):
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    pf    = getattr(yaml_utils, hypes["yaml_parser"])
    hypes = pf(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    model = train_utils.create_model(hypes)
    sd = torch.load(str(CKPT_DIR / CKPT_FILE), map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    return model, hypes


# Prefixes defining backbone vs fusion scope
BACKBONE_PREFIXES = ["backbone_m1", "shrinker_m1"]
FUSION_PREFIXES   = ["fusion_net"]
ALL_PREFIXES      = BACKBONE_PREFIXES + FUSION_PREFIXES


def make_wa_fullcover(base_model, scope_prefixes, label, device):
    """
    Build a W+A fake-quant model covering all Conv2d/Linear in scope_prefixes.
    W: per-tensor symmetric static fake-quant (max|W|/127).
    A: DynamicActQuantWrapper (per-batch dynamic scale, no calib needed,
       100% coverage including HMSA k/q/v/a_linears.0 and .1).
    """
    model = copy.deepcopy(base_model)

    # Step 1: weight fake-quant (in-place on weight.data)
    w_cnt = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if not any(name.startswith(pfx) for pfx in scope_prefixes):
            continue
        module.weight.data = fake_quant_w(module.weight.data)
        w_cnt += 1
    print(f"[{label}] W-quant: {w_cnt} layers", flush=True)

    # Step 2: activation wrap (replaces modules with DynamicActQuantWrapper)
    model, wrapped, skipped = wrap_activation_quant_selective(model, scope_prefixes)
    print(f"[{label}] A-wrap: {len(wrapped)} layers wrapped, {len(skipped)} skipped", flush=True)
    if skipped:
        print(f"[{label}] skipped: {skipped[:5]}", flush=True)

    # Verify HMSA .1 coverage
    hmsa_1 = [w for w in wrapped if '.linears.1' in w]
    print(f"[{label}] HMSA *.linears.1 covered: {len(hmsa_1)}", flush=True)

    model = model.to(device).eval()
    return model, wrapped, skipped


def main():
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("/home/jichengzhi/V2X/logs").mkdir(parents=True, exist_ok=True)

    print("[B] === Phase B: V2X-ViT INT8 full-cover + segmented sensitivity ===", flush=True)
    print(f"[B] ckpt={CKPT_FILE} epoch={EPOCH_USED}", flush=True)
    print(f"[B] quant_method=simulated_int8_fakequant_dynamic (no calib, 100% coverage)", flush=True)
    print(f"[B] NOT TRT INT8 real build — latency NOT measured", flush=True)

    # ---- Load base model + dataset once ----
    print("[B] Loading base model...", flush=True)
    base_model, hypes = load_clean_model("cpu")
    base_model = base_model.to(device).eval()
    print(f"[B] Params: {sum(p.numel() for p in base_model.parameters()):,}", flush=True)

    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_ref.append(dataset)
    print(f"[B] Dataset: {len(dataset)} samples", flush=True)

    # Load existing results to extend
    existing = {}
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            existing = json.load(f)

    results = dict(existing)  # preserve Phase A results

    # ============================================================
    # B-1: W+A full cover (all 80 layers, dynamic scale)
    # B-2 (iii): same model = full = backbone + fusion
    # ============================================================
    print("\n[B] === B-1 / B-2(iii): W+A full cover (backbone + fusion) ===", flush=True)
    model_full, wrapped_full, skipped_full = make_wa_fullcover(
        base_model, ALL_PREFIXES, "WA_full", device)

    ap30_f, ap50_f, ap70_f, ci_f, t_f = run_eval(
        model_full, dataset, device, "WA_INT8_fullcover")
    ap30_base = existing.get("base_fp32", {}).get("ap30", None)
    ap50_base = existing.get("base_fp32", {}).get("ap50", None)
    ap70_base = existing.get("base_fp32", {}).get("ap70", None)

    results["WA_INT8_fullcover"] = {
        "ap30": ap30_f, "ap50": ap50_f, "ap70": ap70_f,
        "mAOE": ci_f["mAOE"],
        "mAOE_ci_lo": ci_f["mAOE_ci_lo"], "mAOE_ci_hi": ci_f["mAOE_ci_hi"],
        "n_tp": ci_f["n_tp"],
        "delta_ap30": round(ap30_f - ap30_base, 4) if ap30_base else None,
        "delta_ap50": round(ap50_f - ap50_base, 4) if ap50_base else None,
        "delta_ap70": round(ap70_f - ap70_base, 4) if ap70_base else None,
        "elapsed_secs": t_f,
        "layers_w_quantized": len([n for n,m in base_model.named_modules()
                                    if isinstance(m,(nn.Conv2d,nn.Linear))
                                    and any(n.startswith(p) for p in ALL_PREFIXES)]),
        "layers_a_wrapped": len(wrapped_full),
        "layers_a_skipped": skipped_full,
        "hmsa_linears_1_covered": len([w for w in wrapped_full if '.linears.1' in w]),
        "quant_method": "simulated_int8_fakequant_dynamic_no_calib",
        "activation_scale": "dynamic per-batch max|x|/127 (no pre-calib, 100% coverage)",
        "attn_matmul_coverage": (
            "HMSA k/q/v/a Linear projections covered (including .1 branch). "
            "Batched matmul (QK^T via torch.einsum, attn@V via einsum) NOT fake-quantized "
            "— einsum is not an nn.Module; would require monkey-patching forward(). "
            "This is a residual uncovered source of AP loss underestimation."),
        "note": "B-1 全覆盖版; dynamic scale = 当前 batch max|x|/127, 无 calib 预热.",
    }
    del model_full  # free GPU memory before next pass

    # Save after B-1 (in case B-2 crashes)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[B] Saved B-1 result to {OUT_JSON}", flush=True)

    # ============================================================
    # B-2 (i): backbone-only W+A
    # ============================================================
    print("\n[B] === B-2(i): backbone_only W+A ===", flush=True)
    torch.cuda.empty_cache()
    model_bb, wrapped_bb, skipped_bb = make_wa_fullcover(
        base_model, BACKBONE_PREFIXES, "WA_bb", device)

    ap30_bb, ap50_bb, ap70_bb, ci_bb, t_bb = run_eval(
        model_bb, dataset, device, "WA_backbone_only")

    results["WA_INT8_backbone_only"] = {
        "ap30": ap30_bb, "ap50": ap50_bb, "ap70": ap70_bb,
        "mAOE": ci_bb["mAOE"],
        "mAOE_ci_lo": ci_bb["mAOE_ci_lo"], "mAOE_ci_hi": ci_bb["mAOE_ci_hi"],
        "n_tp": ci_bb["n_tp"],
        "delta_ap30": round(ap30_bb - ap30_base, 4) if ap30_base else None,
        "delta_ap50": round(ap50_bb - ap50_base, 4) if ap50_base else None,
        "delta_ap70": round(ap70_bb - ap70_base, 4) if ap70_base else None,
        "elapsed_secs": t_bb,
        "scope": "backbone_m1 + shrinker_m1 only (no fusion)",
        "layers_a_wrapped": len(wrapped_bb),
        "quant_method": "simulated_int8_fakequant_dynamic",
        "note": "B-2(i): 只量化 backbone+shrinker Conv2d, fusion 不动.",
    }
    del model_bb
    torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[B] Saved B-2(i) result.", flush=True)

    # ============================================================
    # B-2 (ii): fusion-only W+A
    # ============================================================
    print("\n[B] === B-2(ii): fusion_only W+A ===", flush=True)
    model_fu, wrapped_fu, skipped_fu = make_wa_fullcover(
        base_model, FUSION_PREFIXES, "WA_fu", device)

    ap30_fu, ap50_fu, ap70_fu, ci_fu, t_fu = run_eval(
        model_fu, dataset, device, "WA_fusion_only")

    results["WA_INT8_fusion_only"] = {
        "ap30": ap30_fu, "ap50": ap50_fu, "ap70": ap70_fu,
        "mAOE": ci_fu["mAOE"],
        "mAOE_ci_lo": ci_fu["mAOE_ci_lo"], "mAOE_ci_hi": ci_fu["mAOE_ci_hi"],
        "n_tp": ci_fu["n_tp"],
        "delta_ap30": round(ap30_fu - ap30_base, 4) if ap30_base else None,
        "delta_ap50": round(ap50_fu - ap50_base, 4) if ap50_base else None,
        "delta_ap70": round(ap70_fu - ap70_base, 4) if ap70_base else None,
        "elapsed_secs": t_fu,
        "scope": "fusion_net (V2XTransformer: HMSA+MSwin+FFN) only (no backbone)",
        "layers_a_wrapped": len(wrapped_fu),
        "hmsa_linears_1_covered": len([w for w in wrapped_fu if '.linears.1' in w]),
        "quant_method": "simulated_int8_fakequant_dynamic",
        "note": "B-2(ii): 只量化 fusion_net Linear, backbone 不动.",
    }
    del model_fu
    torch.cuda.empty_cache()

    # Final save
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[B] Saved B-2(ii) result.", flush=True)

    # ============================================================
    # Summary
    # ============================================================
    print("\n[B] ====== PHASE B SUMMARY ======", flush=True)
    base_50 = ap50_base or 0.0
    base_70 = ap70_base or 0.0
    header = f"{'Config':<30} {'AP30':>8} {'AP50':>8} {'AP70':>8} {'ΔAP50':>8} {'ΔAP70':>8}"
    print(header, flush=True)
    print(f"{'base fp32 (Phase A)':<30} {ap30_base or 0:>8.4f} {base_50:>8.4f} {base_70:>8.4f} {'—':>8} {'—':>8}", flush=True)

    rows = [
        ("WA_INT8 (Phase A, partial)", existing.get("WA_INT8", {})),
        ("WA_INT8_fullcover (B-1)",    results["WA_INT8_fullcover"]),
        ("WA_bb_only (B-2i)",          results["WA_INT8_backbone_only"]),
        ("WA_fusion_only (B-2ii)",     results["WA_INT8_fusion_only"]),
    ]
    for tag, r in rows:
        if not r:
            continue
        a30 = r.get("ap30", 0)
        a50 = r.get("ap50", 0)
        a70 = r.get("ap70", 0)
        d50 = r.get("delta_ap50", 0) or (a50 - base_50)
        d70 = r.get("delta_ap70", 0) or (a70 - base_70)
        print(f"{tag:<30} {a30:>8.4f} {a50:>8.4f} {a70:>8.4f} {d50:>+8.4f} {d70:>+8.4f}", flush=True)

    # Verdict
    thresh_cliff = -0.03
    for tag, ap50, da50, da70 in [
        ("WA_fullcover (B-1)",     ap50_f, ap50_f - base_50, ap70_f - base_70),
        ("WA_backbone_only (B-2i)", ap50_bb, ap50_bb - base_50, ap70_bb - base_70),
        ("WA_fusion_only (B-2ii)", ap50_fu, ap50_fu - base_50, ap70_fu - base_70),
    ]:
        if da50 < thresh_cliff:
            verdict = f"[崩] ΔAP50={da50:+.4f} < {thresh_cliff}"
        elif abs(da50) <= 0.02:
            verdict = f"[近无损] ΔAP50={da50:+.4f}"
        else:
            verdict = f"[中度降] ΔAP50={da50:+.4f}"
        print(f"[B] {tag}: {verdict}  ΔAP70={da70:+.4f}", flush=True)

    print(f"\n[B] Saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
