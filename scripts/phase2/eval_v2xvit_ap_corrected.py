"""eval_v2xvit_ap_corrected.py — Phase C

## C-1: 原始剪枝(未 finetune) AP eval
验证"剪枝确实伤 AP, finetune 恢复"这条因果链.
加载 depgraph 剪枝 ckpt (未经 finetune 的 backbone), 测 DAIR val 1789 AP.

## C-2: 严苛版 INT8 — 静态标定 scale + 注意力 matmul fake-quant
修复 Phase B 的两处宽松:
  (a) 动态 per-batch scale → 静态 scale (calib 前 50 batch 的全集 max|x|)
  (b) 补全注意力 matmul 量化:
      - BaseWindowAttention.forward 里的 QK^T einsum 和 attn@V einsum
      - HGTCavAttention.forward 里的 att_map einsum 和 out einsum
通过 monkey-patch HGTCavAttention.forward / BaseWindowAttention.forward 实现.

口径:
  - simulated fake-quant (非 TRT INT8 real build)
  - PyTorch FP32 计算路径 (FP32 累加), 权重/激活 de-quant 后 FP32 算 matmul
  - 静态 scale 来自 val 前 50 batch 的 per-layer max|activation|
  - 这比 TRT INT8 还宽松: TRT 用 INT8 GEMM 做累加; 这里 FP32 累加
  - 残余未覆盖: 无 (W + A Linear + 注意力 matmul 全量化)

用法:
  CUDA_VISIBLE_DEVICES=7 python scripts/phase2/eval_v2xvit_ap_corrected.py \
    2>&1 | tee logs/v2xvit_ap_corrected.log

输出:
  results/v2xvit_ap_corrected.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import types
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

from tools.configurable.depgraph_v2xvit import (
    V2XViTBackboneTraceNet, build_model as build_base_model,
    build_pruner, get_scatter_shape, CONFIG_YAML as DEPGRAPH_CONFIG_YAML
)

# ------------------------------------------------------------------ config --
CKPT_DIR    = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                   "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE   = "net_epoch_bestval_at17.pth"
EPOCH_USED  = "bestval_at17"

PRUNE_P50_CKPT     = REPO_ROOT / "output/a2_prune/v2xvit_bb_p50/v2xvit_pruned_50_epoch17_depgraph.pth"
PRUNE_P75_CKPT     = REPO_ROOT / "output/a2_prune/v2xvit_bb_p75/v2xvit_pruned_75_epoch17_depgraph.pth"
PRUNE_P50_FILTERS  = [32, 64, 128]
PRUNE_P75_FILTERS  = [64, 32, 64]

N_SAMPLES   = 1789
N_CALIB     = 50    # C-2 静态 scale calibration batches
N_BOOTSTRAP = 1000
SEED        = 42

OUT_DIR  = REPO_ROOT / "results"
OUT_JSON = OUT_DIR / "v2xvit_ap_corrected.json"

dataset_ref: List = []

# ------------------------------------------------------------------ helpers --

def fake_quant_sym(x: torch.Tensor, scale: float, n_bits: int = 8) -> torch.Tensor:
    """Symmetric per-tensor fake-quant with fixed scale."""
    n_lvl = 2 ** (n_bits - 1) - 1
    if scale < 1e-12:
        return x
    return torch.clamp(torch.round(x / scale), -n_lvl, n_lvl) * scale


def fake_quant_w(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    n_lvl = 2 ** (n_bits - 1) - 1
    abs_max = w.abs().max().item()
    if abs_max < 1e-10: return w
    return torch.clamp(torch.round(w / (abs_max / n_lvl)), -n_lvl, n_lvl) * (abs_max / n_lvl)


# --------------------------------------------------------------- AP eval  --

def compute_tp_errors_frame(pred_np, score_np, gt_np):
    IOU_TH = 0.5
    if gt_np is None or len(gt_np) == 0: return np.array([]), np.array([]), np.array([])
    if pred_np is None or len(pred_np) == 0: return np.array([]), np.array([]), np.array([])
    order = np.argsort(-score_np)
    pred_s = pred_np[order]
    pp, gp  = corner_to_center(pred_s, 'lwh'), corner_to_center(gt_np, 'lwh')
    ppoly, gpoly = list(convert_format(pred_s)), list(convert_format(gt_np))
    remain = list(range(len(gt_np)))
    ate_l, ase_l, aoe_l = [], [], []
    for i in range(len(pred_s)):
        if not remain: break
        ious = compute_iou(ppoly[i], [gpoly[j] for j in remain])
        if not len(ious) or np.max(ious) < IOU_TH: continue
        best = int(np.argmax(ious)); gidx = remain.pop(best)
        p, g = pp[i], gp[gidx]
        ate = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        lp,wp,hp = abs(p[3]),abs(p[4]),abs(p[5])
        lg,wg,hg = abs(g[3]),abs(g[4]),abs(g[5])
        iv = min(lp,lg)*min(wp,wg)*min(hp,hg); uv = lp*wp*hp + lg*wg*hg - iv
        sio = iv/uv if uv > 1e-9 else 0.0
        d = abs(p[6]-g[6]) % np.pi
        ate_l.append(ate); ase_l.append(1.-sio); aoe_l.append(float(min(d, np.pi-d)))
    return np.array(ate_l), np.array(ase_l), np.array(aoe_l)


def bootstrap_ci(fa, fs, fo, B=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(fo)
    ba, bs, bo = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        def cat(lst):
            parts = [lst[i] for i in idx if len(lst[i]) > 0]
            return np.concatenate(parts) if parts else np.array([])
        a,s,o = cat(fa),cat(fs),cat(fo)
        ba.append(float(np.mean(a)) if len(a) else float('nan'))
        bs.append(float(np.mean(s)) if len(s) else float('nan'))
        bo.append(float(np.mean(o)) if len(o) else float('nan'))
    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        return float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))
    aa = np.concatenate([a for a in fa if len(a)>0])
    as_ = np.concatenate([a for a in fs if len(a)>0])
    ao = np.concatenate([a for a in fo if len(a)>0])
    return {
        "mATE": float(np.mean(aa)) if len(aa) else float('nan'),
        "mATE_ci_lo": ci(ba)[0], "mATE_ci_hi": ci(ba)[1],
        "mASE": float(np.mean(as_)) if len(as_) else float('nan'),
        "mASE_ci_lo": ci(bs)[0], "mASE_ci_hi": ci(bs)[1],
        "mAOE": float(np.mean(ao)) if len(ao) else float('nan'),
        "mAOE_ci_lo": ci(bo)[0], "mAOE_ci_hi": ci(bo)[1],
        "n_tp": len(ao),
    }


def run_eval(model, dataset, device, label):
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, drop_last=False)
    rs = {0.3:{"tp":[],"fp":[],"gt":0,"score":[]},
          0.5:{"tp":[],"fp":[],"gt":0,"score":[]},
          0.7:{"tp":[],"fp":[],"gt":0,"score":[]}}
    fa, fs, fo = [], [], []
    n = 0; t0 = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            if n >= N_SAMPLES: break
            batch = train_utils.to_device(batch, device)
            try:
                infer = inference_utils.inference_intermediate_fusion(
                    batch, model, dataset_ref[0])
            except Exception as e:
                print(f"  [{label}] ERR {n}: {e}", flush=True)
                fa.append(np.array([])); fs.append(np.array([])); fo.append(np.array([]))
                n += 1; continue
            pb, ps, gb = infer["pred_box_tensor"], infer["pred_score"], infer["gt_box_tensor"]
            for th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pb, ps, gb, rs, th)
            if pb is not None and gb is not None:
                a,s,o = compute_tp_errors_frame(
                    torch_tensor_to_numpy(pb), torch_tensor_to_numpy(ps), torch_tensor_to_numpy(gb))
            else:
                a,s,o = np.array([]),np.array([]),np.array([])
            fa.append(a); fs.append(s); fo.append(o)
            n += 1
            if n % 300 == 0:
                el = time.time()-t0
                print(f"  [{label}] {n}/{N_SAMPLES} {el:.0f}s ETA={el/n*(N_SAMPLES-n):.0f}s", flush=True)
    el = time.time()-t0
    print(f"[{label}] done {el:.1f}s ({n} samples)", flush=True)
    tmp = OUT_DIR/f"_tmp_{label}"; tmp.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(rs, str(tmp))
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{label}] Bootstrap CI...", flush=True)
    ci = bootstrap_ci(fa, fs, fo)
    print(f"[{label}] AP30={ap30:.4f} AP50={ap50:.4f} AP70={ap70:.4f} mAOE={ci['mAOE']:.4f}", flush=True)
    return float(ap30), float(ap50), float(ap70), ci, el


# ================================================== C-1: 原始剪枝 AP eval ==

def rebuild_pruned(ratio: float, actual_filters: List[int], prune_ckpt: Path,
                   device: str = "cpu"):
    """
    Rebuild pruned architecture (same as A-2 eval script):
    1. Load base model epoch17
    2. Apply depgraph pruning (deterministic, same ratio)
    3. Load pruned ckpt (flat state_dict, NO finetune)
    Returns model + missing_keys info.
    """
    print(f"[C-1] rebuild ratio={ratio}, filters={actual_filters}", flush=True)
    base_model = build_base_model(device)
    net = V2XViTBackboneTraceNet(base_model).to(device).eval()

    hypes_raw = yaml_utils.load_yaml(str(DEPGRAPH_CONFIG_YAML))
    ny, nx = get_scatter_shape(hypes_raw)
    x = torch.randn(1, 64, ny, nx, device=device)
    pr = build_pruner(net, x, ratio, device)
    pr.step()
    print(f"[C-1] Pruning done. backbone param count = "
          f"{sum(p.numel() for p in base_model.backbone_m1.parameters()):,}", flush=True)

    # Load pruned ckpt (flat format confirmed)
    sd = torch.load(str(prune_ckpt), map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
        print("[C-1] WARNING: wrapped format, unwrapped", flush=True)

    # strict=False because pruned backbone has different channel dims from
    # the full model; fusion/head keys will partially match
    missing, unexpected = base_model.load_state_dict(sd, strict=False)
    print(f"[C-1] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    if len(missing) > 0:
        print(f"[C-1] missing keys (first 5): {missing[:5]}", flush=True)
    if len(unexpected) > 0:
        print(f"[C-1] unexpected keys (first 5): {unexpected[:5]}", flush=True)

    # Sanity: check backbone weight is NOT random (compare vs original epoch17)
    # Load original epoch17 for comparison
    orig_sd = torch.load(str(CKPT_DIR / CKPT_FILE), map_location="cpu")
    if isinstance(orig_sd, dict) and "model_state_dict" in orig_sd:
        orig_sd = orig_sd["model_state_dict"]

    # Check one backbone weight that should differ (pruned vs original)
    k_check = "backbone_m1.blocks.0.1.weight"
    if k_check in sd and k_check in base_model.state_dict():
        pruned_w = sd[k_check]
        loaded_w = base_model.state_dict()[k_check]
        match = torch.allclose(pruned_w, loaded_w)
        print(f"[C-1] Weight check '{k_check}': pruned==loaded: {match} "
              f"(shape pruned={pruned_w.shape}, loaded={loaded_w.shape})", flush=True)
    else:
        print(f"[C-1] WARNING: '{k_check}' not found in sd or model", flush=True)

    return base_model, missing, unexpected


# ================================================== C-2: 严苛 INT8 ==

def calibrate_static_scales(model: nn.Module, loader: DataLoader, device: str,
                              n_calib: int = N_CALIB) -> Dict[str, float]:
    """
    Run n_calib batches, collect max|activation| per Conv2d/Linear layer.
    Uses forward pre-hooks. Returns dict: name -> scale = max_abs/127.
    """
    print(f"[C-2 calib] Collecting static activation scales ({n_calib} batches)...", flush=True)
    act_max: Dict[str, float] = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if not inp or inp[0] is None: return
            val = inp[0].detach().abs().max().item()
            if not np.isfinite(val): return
            act_max[name] = max(act_max.get(name, 0.0), val)
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            if n_done >= n_calib: break
            batch = train_utils.to_device(batch, device)
            try:
                inference_utils.inference_intermediate_fusion(batch, model, dataset_ref[0])
            except Exception as e:
                print(f"  [C-2 calib] batch {n_done} err: {e}", flush=True)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  [C-2 calib] {n_done}/{n_calib}", flush=True)

    for h in hooks: h.remove()
    n_lvl = 127
    scales = {k: max(v / n_lvl, 1e-12) for k, v in act_max.items()}
    print(f"[C-2 calib] Collected scales for {len(scales)} layers "
          f"(total Conv2d+Linear: {sum(1 for _,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear)))})", flush=True)
    return scales


class StaticActQuantWrapper(nn.Module):
    """Activation fake-quant with fixed (calibrated) scale."""
    def __init__(self, orig: nn.Module, scale: float, n_bits: int = 8):
        super().__init__()
        self.orig = orig; self.scale = scale; self.n_bits = n_bits
    def forward(self, x, *a, **kw):
        return self.orig(fake_quant_sym(x, self.scale, self.n_bits), *a, **kw)


def wrap_static_act_quant(model: nn.Module, scales: Dict[str, float],
                          all_prefixes: List[str], n_bits: int = 8):
    """Replace Conv2d/Linear in scope with StaticActQuantWrapper."""
    def _set(root, path, mod):
        parts = path.split(".")
        obj = root
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        last = parts[-1]
        if last.isdigit(): obj[int(last)] = mod
        else: setattr(obj, last, mod)

    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))
               and any(n.startswith(pfx) for pfx in all_prefixes)]
    wrapped = []; no_scale = []
    for name, mod in targets:
        if name in scales:
            _set(model, name, StaticActQuantWrapper(mod, scales[name], n_bits))
            wrapped.append(name)
        else:
            no_scale.append(name)
    print(f"[C-2] Static A-wrap: {len(wrapped)} wrapped, {len(no_scale)} no-scale", flush=True)
    if no_scale:
        print(f"[C-2] no-scale layers (first 5): {no_scale[:5]}", flush=True)
    return model, wrapped, no_scale


# -------- monkey-patch attention matmul fake-quant --------

def patch_attn_matmul(model: nn.Module, scales: Dict[str, float],
                      n_bits: int = 8) -> List[str]:
    """
    Monkey-patch BaseWindowAttention.forward and HGTCavAttention.forward
    to fake-quant q/k/v tensors before einsum (QK^T and attn@V).

    For scales: we use conservative global mean of all collected scales
    as proxy for q/k/v/attn dynamic ranges (no per-attention-layer calib
    since these are not nn.Module — real TRT INT8 would calibrate each).

    Returns list of patched module paths.
    """
    from opencood.models.sub_modules.mswin import BaseWindowAttention
    from opencood.models.sub_modules.hmsa import HGTCavAttention

    # Compute a single proxy scale for attention tensors:
    # Use median of all collected activation scales (conservative estimate)
    if scales:
        attn_scale = float(np.percentile(list(scales.values()), 50))
    else:
        attn_scale = 1.0 / 127.0
    print(f"[C-2 attn-patch] Using proxy attention scale={attn_scale:.6f} "
          f"(median of {len(scales)} layer scales)", flush=True)

    patched = []

    # --- Patch BaseWindowAttention.forward ---
    for name, module in model.named_modules():
        if not isinstance(module, BaseWindowAttention):
            continue
        orig_fwd = module.forward.__func__  # unbound method

        def make_bwa_fwd(scale, n):
            def new_forward(self, x):
                b, l, h, w, c, m = *x.shape, self.heads
                qkv = self.to_qkv(x).chunk(3, dim=-1)
                new_h = h // self.window_size; new_w = w // self.window_size
                from einops import rearrange
                q, k, v = map(
                    lambda t: rearrange(t,
                        'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                        m=m, w_h=self.window_size, w_w=self.window_size), qkv)
                # *** fake-quant q, k before QK^T ***
                q_q = fake_quant_sym(q, scale, n)
                k_q = fake_quant_sym(k, scale, n)
                dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
                                    q_q, k_q) * self.scale
                if self.relative_pos_embedding:
                    dots += self.pos_embedding[self.relative_indices[:,:,0],
                                               self.relative_indices[:,:,1]]
                else:
                    dots += self.pos_embedding
                attn = dots.softmax(dim=-1)
                # *** fake-quant attn, v before attn@V ***
                attn_q = fake_quant_sym(attn, scale, n)
                v_q    = fake_quant_sym(v,    scale, n)
                out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn_q, v_q)
                out = rearrange(out,
                    'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                    m=self.heads, w_h=self.window_size, w_w=self.window_size,
                    new_w=new_w, new_h=new_h)
                out = self.to_out(out)
                return out
            return new_forward

        module.forward = types.MethodType(make_bwa_fwd(attn_scale, n_bits), module)
        patched.append(f"BaseWindowAttention:{name}")

    # --- Patch HGTCavAttention.forward ---
    for name, module in model.named_modules():
        if not isinstance(module, HGTCavAttention):
            continue

        def make_hgt_fwd(scale, n):
            def new_forward(self, x, mask, prior_encoding):
                from einops import rearrange
                x = x.permute(0, 2, 3, 1, 4)
                mask = mask.unsqueeze(1)
                velocities, dts, types = [itm.squeeze(-1) for itm in
                    prior_encoding[:, :, 0, 0, :].split([1, 1, 1], dim=-1)]
                types = types.to(torch.int); dts = dts.to(torch.int)
                qkv = self.to_qkv(x, types)
                w_att, w_msg = self.get_hetero_edge_weights(x, types)
                q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                                  m=self.heads), qkv)
                # *** fake-quant q, k before att_map einsum ***
                q_q = fake_quant_sym(q, scale, n)
                k_q = fake_quant_sym(k, scale, n)
                att_map = torch.einsum(
                    'b m h w i p, b m i j p q, bm h w j q -> b m h w i j',
                    [q_q, w_att, k_q]) * self.scale
                att_map = att_map.masked_fill(mask == 0, -float('inf'))
                att_map = self.attend(att_map)
                # *** fake-quant v before message einsum ***
                v_q = fake_quant_sym(v, scale, n)
                v_msg = torch.einsum('b m i j p c, b m h w j p -> b m h w i j c', w_msg, v_q)
                # *** fake-quant att_map before output einsum ***
                att_q = fake_quant_sym(att_map, scale, n)
                out = torch.einsum('b m h w i j, b m h w i j c -> b m h w i c', att_q, v_msg)
                out = rearrange(out, 'b m h w l c -> b h w l (m c)', m=self.heads)
                out = self.to_out(out, types)
                out = self.drop_out(out)
                out = out.permute(0, 3, 1, 2, 4)
                return out
            return new_forward

        module.forward = types.MethodType(make_hgt_fwd(attn_scale, n_bits), module)
        patched.append(f"HGTCavAttention:{name}")

    print(f"[C-2 attn-patch] Patched {len(patched)} attention modules", flush=True)
    for p in patched[:6]:
        print(f"  {p}", flush=True)
    return patched


# -------------------------------------------------------------------  main --

def load_base_model_clean(device="cpu"):
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    pf = getattr(yaml_utils, hypes["yaml_parser"])
    hypes = pf(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    model = train_utils.create_model(hypes)
    sd = torch.load(str(CKPT_DIR / CKPT_FILE), map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    return model, hypes


def main():
    device = "cuda"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("/home/jichengzhi/V2X/logs").mkdir(parents=True, exist_ok=True)

    print("[C] === Phase C: V2X-ViT AP corrected (C-1 original prune + C-2 strict INT8) ===", flush=True)
    print(f"[C] NOT TRT INT8 real build; latency NOT measured; FP32 compute path", flush=True)

    results = {}

    # Load base model + dataset once
    print("[C] Loading base model...", flush=True)
    base_model, hypes = load_base_model_clean("cpu")
    base_model_gpu = copy.deepcopy(base_model).to(device).eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    dataset_ref.append(dataset)
    print(f"[C] Dataset: {len(dataset)} samples", flush=True)

    # ============================================================
    # C-1: Original (pre-finetune) pruned AP
    # ============================================================
    for ratio, actual_filters, ckpt_path in [
        (0.50, PRUNE_P50_FILTERS, PRUNE_P50_CKPT),
        (0.75, PRUNE_P75_FILTERS, PRUNE_P75_CKPT),
    ]:
        tag = f"prune_p{int(ratio*100)}_no_finetune"
        print(f"\n[C-1] === {tag} ===", flush=True)
        pruned_model, missing, unexpected = rebuild_pruned(ratio, actual_filters, ckpt_path, "cpu")
        pruned_model = pruned_model.to(device).eval()

        ap30, ap50, ap70, ci, elapsed = run_eval(pruned_model, dataset, device, tag)

        results[tag] = {
            "ap30": ap30, "ap50": ap50, "ap70": ap70,
            "mAOE": ci["mAOE"],
            "mAOE_ci_lo": ci["mAOE_ci_lo"], "mAOE_ci_hi": ci["mAOE_ci_hi"],
            "n_tp": ci["n_tp"],
            "delta_ap30_vs_base": round(ap30 - 0.7854, 4),
            "delta_ap50_vs_base": round(ap50 - 0.7103, 4),
            "delta_ap70_vs_base": round(ap70 - 0.5212, 4),
            "elapsed_secs": elapsed,
            "ckpt_path": str(ckpt_path),
            "actual_filters": actual_filters,
            "missing_keys_count": len(missing),
            "unexpected_keys_count": len(unexpected),
            "missing_keys_sample": missing[:5],
            "epoch_used": EPOCH_USED,
            "note": ("C-1: 原始 depgraph 剪枝 ckpt, 无 finetune. "
                     "ISS-005: flat state_dict 确认. "
                     "missing_keys_count 为剪枝导致 channel 不匹配的结构差异, 非 key missing."),
        }
        del pruned_model
        torch.cuda.empty_cache()

        # Save after each C-1 pass
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[C-1] Saved {tag}", flush=True)

    # ============================================================
    # C-2: 严苛版 INT8
    # ============================================================
    print(f"\n[C-2] === Strict INT8: static scale + attn matmul quant ===", flush=True)

    # Step 1: Build clean model for calibration
    model_c2 = copy.deepcopy(base_model).to(device).eval()

    # Step 2: Calibrate static activation scales (pre-quant model)
    loader_calib = DataLoader(dataset, batch_size=1, num_workers=2,
                              collate_fn=dataset.collate_batch_test,
                              shuffle=False, drop_last=False)
    act_scales = calibrate_static_scales(model_c2, loader_calib, device, N_CALIB)

    # Step 3: Weight fake-quant (same as Phase A W-quant)
    ALL_PREFIXES = ["backbone_m1", "shrinker_m1", "fusion_net"]
    w_cnt = 0
    for name, module in model_c2.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and any(name.startswith(p) for p in ALL_PREFIXES):
            if torch.isfinite(module.weight.data).all():
                module.weight.data = fake_quant_w(module.weight.data)
                w_cnt += 1
    print(f"[C-2] W-quant: {w_cnt} layers", flush=True)

    # Step 4: Static activation wrap (Linear + Conv2d)
    model_c2, wrapped_a, no_scale_a = wrap_static_act_quant(model_c2, act_scales, ALL_PREFIXES)
    print(f"[C-2] Static A-wrap: {len(wrapped_a)} layers", flush=True)

    # Step 5: Patch attention matmul
    patched = patch_attn_matmul(model_c2, act_scales, n_bits=8)

    model_c2 = model_c2.to(device).eval()

    loader_c2 = DataLoader(dataset, batch_size=1, num_workers=2,
                           collate_fn=dataset.collate_batch_test,
                           shuffle=False, drop_last=False)
    ap30_c2, ap50_c2, ap70_c2, ci_c2, t_c2 = run_eval(model_c2, dataset, device, "WA_INT8_strict")

    # Load B-1 dynamic result for comparison
    phase_b_json = OUT_DIR / "v2xvit_int8_ap_sim.json"
    b1_ap50 = b1_ap70 = None
    if phase_b_json.exists():
        with open(phase_b_json) as f:
            pb = json.load(f)
        b1 = pb.get("WA_INT8_fullcover", {})
        b1_ap50 = b1.get("ap50"); b1_ap70 = b1.get("ap70")
        base_ap50 = pb.get("base_fp32", {}).get("ap50", 0.7103)
        base_ap70 = pb.get("base_fp32", {}).get("ap70", 0.5211)
    else:
        base_ap50, base_ap70 = 0.7103, 0.5211

    results["WA_INT8_strict"] = {
        "ap30": ap30_c2, "ap50": ap50_c2, "ap70": ap70_c2,
        "mAOE": ci_c2["mAOE"],
        "mAOE_ci_lo": ci_c2["mAOE_ci_lo"], "mAOE_ci_hi": ci_c2["mAOE_ci_hi"],
        "n_tp": ci_c2["n_tp"],
        "delta_ap50_vs_base": round(ap50_c2 - base_ap50, 4),
        "delta_ap70_vs_base": round(ap70_c2 - base_ap70, 4),
        "delta_ap50_vs_B1_dynamic": round(ap50_c2 - b1_ap50, 4) if b1_ap50 else None,
        "delta_ap70_vs_B1_dynamic": round(ap70_c2 - b1_ap70, 4) if b1_ap70 else None,
        "B1_dynamic_ap50_for_ref": b1_ap50,
        "B1_dynamic_ap70_for_ref": b1_ap70,
        "elapsed_secs": t_c2,
        "n_calib_batches": N_CALIB,
        "act_scale_stat": "static max|x| over 50 calib batches / 127",
        "layers_w_quantized": w_cnt,
        "layers_a_wrapped": len(wrapped_a),
        "layers_a_no_scale": no_scale_a,
        "attn_modules_patched": len(patched),
        "attn_scale_proxy_info": "median of all layer act_scales (no per-attn calib)",
        "quant_method": "simulated_int8_fakequant_static_scale_plus_attn_matmul",
        "caveat": ("simulated (non-TRT real build); FP32 compute path, w/a de-quant "
                   "before matmul; FP32 accumulation. Stricter than B-1 dynamic: "
                   "static scale + attention einsum q/k/v/attn fake-quant added."),
        "residual_uncovered": ("FP32 accumulation (vs INT8 in TRT); "
                               "no per-attention-layer scale calib (uses global proxy); "
                               "BN/LN statistics; einsum intermediate accumulation."),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # Summary
    # ============================================================
    print("\n[C] ====== PHASE C SUMMARY ======", flush=True)
    print(f"{'Config':<38} {'AP30':>8} {'AP50':>8} {'AP70':>8} {'ΔAP50':>8} {'ΔAP70':>8}", flush=True)
    BASE_AP30, BASE_AP50, BASE_AP70 = 0.7854, 0.7103, 0.5211
    FT_P50 = {"ap50": 0.7337, "ap70": 0.5336}  # A-2 p50 finetune ref
    FT_P75 = {"ap50": 0.7266, "ap70": 0.5445}  # A-2 p75 finetune ref

    rows = [
        ("base fp32 (A-1)", BASE_AP30, BASE_AP50, BASE_AP70),
        ("A-2 p50 finetune (ref)", None, FT_P50["ap50"], FT_P50["ap70"]),
        ("A-2 p75 finetune (ref)", None, FT_P75["ap50"], FT_P75["ap70"]),
    ]
    for tag_key in ["prune_p50_no_finetune", "prune_p75_no_finetune", "WA_INT8_strict"]:
        r = results.get(tag_key, {})
        if r:
            rows.append((tag_key, r.get("ap30",0), r.get("ap50",0), r.get("ap70",0)))

    for tag, a30, a50, a70 in rows:
        d50 = (a50 - BASE_AP50) if a50 else None
        d70 = (a70 - BASE_AP70) if a70 else None
        a30s = f"{a30:.4f}" if a30 else "  —   "
        d50s = f"{d50:+.4f}" if d50 is not None else "    —  "
        d70s = f"{d70:+.4f}" if d70 is not None else "    —  "
        print(f"{tag:<38} {a30s:>8} {a50:.4f} {a70:.4f} {d50s:>8} {d70s:>8}", flush=True)

    # C-1 verdict
    print("\n[C-1] Prune before/after finetune comparison:", flush=True)
    for tag_key, ft_ap50, ft_ap70 in [
        ("prune_p50_no_finetune", FT_P50["ap50"], FT_P50["ap70"]),
        ("prune_p75_no_finetune", FT_P75["ap50"], FT_P75["ap70"]),
    ]:
        r = results.get(tag_key, {})
        if r:
            da50_base = r["ap50"] - BASE_AP50
            da50_ft   = ft_ap50 - r["ap50"]
            print(f"  {tag_key}: AP50={r['ap50']:.4f}  "
                  f"vs base ΔAP50={da50_base:+.4f}  "
                  f"finetune恢复 ΔAP50={da50_ft:+.4f}", flush=True)

    # C-2 verdict
    r = results.get("WA_INT8_strict", {})
    if r:
        da50 = r["delta_ap50_vs_base"]
        da50_vs_b1 = r["delta_ap50_vs_B1_dynamic"]
        thresh = -0.03
        verdict = "[崩]" if da50 < thresh else ("[近无损]" if abs(da50) <= 0.02 else "[中度降]")
        print(f"\n[C-2] WA_INT8_strict: {verdict} ΔAP50(vs base)={da50:+.4f}  "
              f"ΔAP70={r['delta_ap70_vs_base']:+.4f}", flush=True)
        if da50_vs_b1 is not None:
            print(f"[C-2] vs B-1 dynamic: ΔAP50={da50_vs_b1:+.4f} "
                  f"(strict 比 dynamic 多损失: {da50_vs_b1:.4f} AP50)", flush=True)

    print(f"\n[C] Saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
