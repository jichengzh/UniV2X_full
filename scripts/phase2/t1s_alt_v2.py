"""T1-S-alt v2: HMSA batched patching + per-submodule wall-clock breakdown.

Previous attempt: BatchedHMSA.__init__ failed (to_norm missing).
Fix: directly monkey-patch to_qkv/to_out on HMSA instance, no wrapper class.

Also adds per-submodule CUDA-event timing UNDER CURRENT CONDITIONS (not old profile)
to find where the 202ms actually goes:
  - STTF, HMSA×3, MSwin×3, FFN×3, get_roi_mask, rte, etc.

Run:
  CUDA_VISIBLE_DEVICES=5 python scripts/phase2/t1s_alt_v2.py

Output: results/t1s_alt_v2.json
"""
import os
import sys
import json
import types
import statistics
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset

CKPT_DIR = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE = CKPT_DIR / "net_epoch_bestval_at17.pth"
N_COLLECT = 30
N_WARMUP  = 5
N_MEASURE = 20
OUT = REPO_ROOT / "results" / "t1s_alt_v2.json"


# ── collect transformer inputs ────────────────────────────────────────────────

def collect_inputs(model, loader, dataset, device, n=N_COLLECT):
    from opencood.models.fuse_modules.fusion_in_one import Regroup
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    saved = []

    def patched_fwd(self_outer, x, record_len, affine_matrix):
        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]
        rf, mask = Regroup(x, record_len, L)
        pe = torch.zeros(len(record_len), L, 3, 1, 1).to(record_len.device)
        pe = pe.repeat(1, 1, 1, rf.shape[3], rf.shape[4])
        rf = torch.cat([rf, pe], dim=2)
        rf_new = []
        for b in range(B):
            rf_new.append(warp_affine_simple(rf[b], affine_matrix[b, 0], (H, W)))
        rf = torch.stack(rf_new).permute(0, 1, 3, 4, 2)
        scm = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        if len(saved) < n:
            saved.append((rf.detach().clone(), mask.detach().clone(), scm.detach().clone()))
        fused = self_outer.fusion_net(rf, mask, scm)
        return fused.permute(0, 3, 1, 2)

    model.fusion_net.forward = types.MethodType(patched_fwd, model.fusion_net)
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None or len(saved) >= n: break
            batch_data = train_utils.to_device(batch_data, device)
            torch.cuda.synchronize()
            try: inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
            except Exception: pass
            torch.cuda.synchronize()
    try: del model.fusion_net.forward
    except AttributeError: pass
    print(f"[collect] {len(saved)} inputs shape={saved[0][0].shape}", flush=True)
    return saved


# ── timing helpers ────────────────────────────────────────────────────────────

def cuda_time_fn(fn, inputs, n_warmup=N_WARMUP, n_measure=N_MEASURE, label=""):
    all_inp = inputs * ((n_warmup + n_measure) // len(inputs) + 1)
    events, errors, first_err = [], 0, None
    with torch.no_grad():
        for i, inp in enumerate(all_inp[:n_warmup + n_measure]):
            torch.cuda.synchronize()
            try:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(*inp); e.record()
                torch.cuda.synchronize()
                if i >= n_warmup: events.append(s.elapsed_time(e))
            except Exception as ex:
                errors += 1
                if first_err is None:
                    import traceback; first_err = traceback.format_exc()[:300]
    if not events:
        return {"label": label, "error": first_err or "no events", "n_errors": errors}
    ev = sorted(events)
    return {"label": label, "n": len(events), "n_errors": errors,
            "mean_ms": round(statistics.mean(events), 3),
            "p50_ms": round(ev[len(ev)//2], 3),
            "min_ms": round(min(events), 3), "max_ms": round(max(events), 3),
            "first_error": first_err}


# ── per-submodule CUDA-event hooks ────────────────────────────────────────────

def attach_submodule_hooks(model):
    """Attach CUDA-event hooks to measure each named sub-module."""
    enc = model.fusion_net.fusion_net.encoder
    acc = defaultdict(list)

    def make_hooks(tag):
        def pre(m, inp):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            m._evt_s = s; m._evt_e = e
        def post(m, inp, out):
            m._evt_e.record()
            acc[tag].append((m._evt_s, m._evt_e))
        return pre, post

    handles = []
    # STTF
    pre, post = make_hooks("sttf")
    handles += [enc.sttf.register_forward_pre_hook(pre),
                enc.sttf.register_forward_hook(post)]
    # prior_feed (Linear replacing RTE)
    pre, post = make_hooks("prior_feed")
    handles += [enc.prior_feed.register_forward_pre_hook(pre),
                enc.prior_feed.register_forward_hook(post)]
    # 3 depth layers: HMSA, MSwin, FFN
    # enc.layers[d] is ModuleList([V2XFusionBlock, FeedForward])
    for d in range(len(enc.layers)):
        fb = enc.layers[d][0]   # V2XFusionBlock
        ff = enc.layers[d][1]   # FeedForward
        # fb.layers[0] is ModuleList([PreNorm(HMSA), PreNorm(MSwin)])
        cav_prenorm  = fb.layers[0][0]  # PreNorm wrapping HMSA
        pwin_prenorm = fb.layers[0][1]  # PreNorm wrapping MSwin
        for tag, mod in [(f"L{d}_hmsa_prenorm", cav_prenorm),
                         (f"L{d}_mswin_prenorm", pwin_prenorm),
                         (f"L{d}_ff", ff)]:
            pre, post = make_hooks(tag)
            handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
        # Also hook the inner fn (HMSA / MSwin) to see prenorm overhead
        hmsa_inner = getattr(cav_prenorm, 'fn', cav_prenorm)
        mswin_inner = getattr(pwin_prenorm, 'fn', pwin_prenorm)
        for tag, mod in [(f"L{d}_hmsa_inner", hmsa_inner), (f"L{d}_mswin_inner", mswin_inner)]:
            pre, post = make_hooks(tag)
            handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    # Full V2XTransformer
    pre, post = make_hooks("v2xtransformer_total")
    handles += [model.fusion_net.fusion_net.register_forward_pre_hook(pre),
                model.fusion_net.fusion_net.register_forward_hook(post)]
    return handles, acc


# ── HMSA batched patching ─────────────────────────────────────────────────────

def make_batched_to_qkv(hmsa):
    """Pre-stack weights, return batched to_qkv replacing per-agent loop."""
    n_types = len(hmsa.q_linears)
    # Stack: (n_types, C_out, C_in)
    q_w = torch.stack([m.weight.data for m in hmsa.q_linears]).clone()  # (n_types, C_out, C_in)
    k_w = torch.stack([m.weight.data for m in hmsa.k_linears]).clone()
    v_w = torch.stack([m.weight.data for m in hmsa.v_linears]).clone()
    q_b = torch.stack([m.bias.data for m in hmsa.q_linears]).clone()    # (n_types, C_out)
    k_b = torch.stack([m.bias.data for m in hmsa.k_linears]).clone()
    v_b = torch.stack([m.bias.data for m in hmsa.v_linears]).clone()

    def batched_to_qkv(x, type_ids):
        # x: (B, H, W, L, C_in)  type_ids: (B, L)
        B, H, W, L, C_in = x.shape
        C_out = q_w.shape[1]
        # Per-agent weight lookup: (B*L, C_out, C_in)
        idx = type_ids.reshape(-1)  # (B*L,)
        aw_q = q_w[idx]; ab_q = q_b[idx]  # (B*L, C_out, C_in), (B*L, C_out)
        aw_k = k_w[idx]; ab_k = k_b[idx]
        aw_v = v_w[idx]; ab_v = v_b[idx]
        # x: (B, H, W, L, C_in) → (B*L, H*W, C_in) for batched GEMM
        x_flat = x.permute(0, 3, 1, 2, 4).reshape(B*L, H*W, C_in)
        # (B*L, H*W, C_in) @ (B*L, C_in, C_out) = (B*L, H*W, C_out)
        def bgemm(xf, w, b):
            return torch.bmm(xf, w.permute(0, 2, 1)) + b.unsqueeze(1)
        q_flat = bgemm(x_flat, aw_q, ab_q)
        k_flat = bgemm(x_flat, aw_k, ab_k)
        v_flat = bgemm(x_flat, aw_v, ab_v)
        # Reshape back: (B*L, H*W, C_out) → (B, L, H, W, C_out) → (B, H, W, L, C_out)
        q = q_flat.reshape(B, L, H, W, C_out).permute(0, 2, 3, 1, 4)
        k = k_flat.reshape(B, L, H, W, C_out).permute(0, 2, 3, 1, 4)
        v = v_flat.reshape(B, L, H, W, C_out).permute(0, 2, 3, 1, 4)
        return q, k, v
    return batched_to_qkv, (q_w, k_w, v_w, q_b, k_b, v_b)


def make_batched_to_out(hmsa):
    n_types = len(hmsa.a_linears)
    a_w = torch.stack([m.weight.data for m in hmsa.a_linears]).clone()
    a_b = torch.stack([m.bias.data for m in hmsa.a_linears]).clone()

    def batched_to_out(x, type_ids):
        B, H, W, L, C_in = x.shape
        C_out = a_w.shape[1]
        idx = type_ids.reshape(-1)
        aw = a_w[idx]; ab = a_b[idx]
        x_flat = x.permute(0, 3, 1, 2, 4).reshape(B*L, H*W, C_in)
        out_flat = torch.bmm(x_flat, aw.permute(0, 2, 1)) + ab.unsqueeze(1)
        return out_flat.reshape(B, L, H, W, C_out).permute(0, 2, 3, 1, 4)
    return batched_to_out, (a_w, a_b)


def patch_all_hmsa(model, device):
    """Find all HGTCavAttention modules, replace to_qkv/to_out with batched."""
    enc = model.fusion_net.fusion_net.encoder
    saved = []
    for d in range(len(enc.layers)):
        fb = enc.layers[d][0]
        cav_prenorm = fb.layers[0][0]
        hmsa = getattr(cav_prenorm, 'fn', cav_prenorm)
        if type(hmsa).__name__ != 'HGTCavAttention':
            print(f"[patch] L{d} cav_attn is {type(hmsa).__name__}, skipping", flush=True)
            continue
        orig_qkv = hmsa.to_qkv
        orig_out = hmsa.to_out
        batched_qkv, qkv_tensors = make_batched_to_qkv(hmsa)
        batched_out, out_tensors = make_batched_to_out(hmsa)
        # Move stacked weights to device
        qkv_tensors = tuple(t.to(device) for t in qkv_tensors)
        out_tensors = tuple(t.to(device) for t in out_tensors)
        # Rebind closures with device-moved tensors
        batched_qkv, _ = make_batched_to_qkv(hmsa)
        batched_out, _ = make_batched_to_out(hmsa)
        # Make device-aware versions
        qw, kw, vw, qb, kb, vb = qkv_tensors
        aw, ab = out_tensors

        def _to_qkv_d(x, t, _qw=qw, _kw=kw, _vw=vw, _qb=qb, _kb=kb, _vb=vb):
            B, H, W, L, Ci = x.shape
            Co = _qw.shape[1]
            idx = t.reshape(-1)
            def bgemm(xf, w, b): return torch.bmm(xf, w[idx].permute(0,2,1)) + b[idx].unsqueeze(1)
            xf = x.permute(0,3,1,2,4).reshape(B*L, H*W, Ci)
            q = bgemm(xf, _qw, _qb).reshape(B,L,H,W,Co).permute(0,2,3,1,4)
            k = bgemm(xf, _kw, _kb).reshape(B,L,H,W,Co).permute(0,2,3,1,4)
            v = bgemm(xf, _vw, _vb).reshape(B,L,H,W,Co).permute(0,2,3,1,4)
            return q, k, v

        def _to_out_d(x, t, _aw=aw, _ab=ab):
            B, H, W, L, Ci = x.shape
            Co = _aw.shape[1]
            idx = t.reshape(-1)
            xf = x.permute(0,3,1,2,4).reshape(B*L, H*W, Ci)
            out = torch.bmm(xf, _aw[idx].permute(0,2,1)) + _ab[idx].unsqueeze(1)
            return out.reshape(B,L,H,W,Co).permute(0,2,3,1,4)

        hmsa.to_qkv = _to_qkv_d
        hmsa.to_out = _to_out_d
        saved.append((hmsa, orig_qkv, orig_out))
        print(f"[patch] L{d} HMSA patched; q_linears: {len(hmsa.q_linears)} types, "
              f"weight shape: {qw.shape}", flush=True)
    return saved


def restore_hmsa(saved):
    for hmsa, orig_qkv, orig_out in saved:
        hmsa.to_qkv = orig_qkv
        hmsa.to_out = orig_out


# ── HMSA to_qkv wall-clock micro-bench ───────────────────────────────────────

def hmsa_inner_wallclock(model, inputs):
    """Measure wall-clock time of HMSA to_qkv loop vs batched replacement on a single input."""
    enc = model.fusion_net.fusion_net.encoder
    cav_prenorm = enc.layers[0][0].layers[0][0]
    hmsa = getattr(cav_prenorm, 'fn', cav_prenorm)

    # We need to get the actual input to HMSA (after encoder/sttf/rte/com_mask).
    # Capture it with a hook.
    captured = {}

    def pre_hook(m, inp):
        if 'x' not in captured and inp[0].shape[-2] > 0:
            # inp[0] is x after norm, shape (B,H,W,L,C)
            captured['x'] = inp[0].detach().clone()
            captured['types'] = getattr(m, '_types_cache', None)

    # We need types from hmsa.forward, not available directly.
    # Patch hmsa.forward to cache types.
    orig_forward = hmsa.forward
    hmsa_input = {}

    def patched_hmsa_forward(x, mask, prior_encoding):
        if 'x_captured' not in hmsa_input:
            # Compute types like original code does (B,L) tensor of ints from mask)
            hmsa_input['x_captured'] = x.detach().clone()
            # types are derived from position: in DAIR, types[b,0]=0 (ego), types[b,1]=1 (RSU)
            # Get from mask: mask shape is (B,H,W,1,L)
            # Actually types come from prior_encoding in the HMSA code
            # Let's just use 0 for ego, 1 for RSU for DAIR
            B = x.shape[0]; L = x.shape[-2]
            types_fake = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            hmsa_input['types_fake'] = types_fake.clamp(0, len(hmsa.q_linears)-1)
        return orig_forward(x, mask, prior_encoding)

    hmsa.forward = patched_hmsa_forward

    # Run one inference pass to capture the input
    with torch.no_grad():
        transformer = model.fusion_net.fusion_net
        inp = inputs[0]
        transformer(*inp)
        torch.cuda.synchronize()

    hmsa.forward = orig_forward

    if 'x_captured' not in hmsa_input:
        print("[hmsa_inner] WARNING: failed to capture HMSA input", flush=True)
        return {}

    x_cap = hmsa_input['x_captured']
    types_cap = hmsa_input['types_fake']
    print(f"[hmsa_inner] captured x shape: {x_cap.shape}, types: {types_cap}", flush=True)

    # Bench original to_qkv
    N = 200
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            hmsa.to_qkv(x_cap, types_cap)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    orig_ms = round((t1 - t0) * 1000 / N, 3)

    # Build batched replacement and bench
    _, qkv_t = make_batched_to_qkv(hmsa)
    qw, kw, vw, qb, kb, vb = tuple(t.to(x_cap.device) for t in qkv_t)

    def _bqkv(x, t):
        B, H, W, L, Ci = x.shape
        Co = qw.shape[1]
        idx = t.reshape(-1)
        def bg(xf, w, b): return torch.bmm(xf, w[idx].permute(0,2,1)) + b[idx].unsqueeze(1)
        xf = x.permute(0,3,1,2,4).reshape(B*L, H*W, Ci)
        q = bg(xf, qw, qb).reshape(B,L,H,W,Co).permute(0,2,3,1,4)
        k = bg(xf, kw, kb).reshape(B,L,H,W,Co).permute(0,2,3,1,4)
        v = bg(xf, vw, vb).reshape(B,L,H,W,Co).permute(0,2,3,1,4)
        return q, k, v

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _bqkv(x_cap, types_cap)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    batched_ms = round((t1 - t0) * 1000 / N, 3)

    speedup = round(orig_ms / batched_ms, 2) if batched_ms > 0 else None
    print(f"[hmsa_inner] to_qkv: orig={orig_ms}ms  batched={batched_ms}ms  speedup={speedup}x", flush=True)
    return {"to_qkv_orig_ms": orig_ms, "to_qkv_batched_ms": batched_ms, "to_qkv_speedup": speedup}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
    model = train_utils.create_model(hypes)
    state = torch.load(CKPT_FILE, map_location="cpu")
    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()

    dataset = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    print("\n[t1s-alt-v2] Collecting transformer inputs ...", flush=True)
    inputs = collect_inputs(model, loader, dataset, device, n=N_COLLECT)
    if not inputs: return

    transformer = model.fusion_net.fusion_net

    # ── 1. Baseline eager ────────────────────────────────────────────────────
    print("\n=== 1. Baseline eager ===", flush=True)
    r_base = cuda_time_fn(transformer, inputs, label="eager_baseline")
    print(f"baseline: {r_base}", flush=True)

    # ── 2. Sub-module breakdown (current conditions) ─────────────────────────
    print("\n=== 2. Sub-module CUDA-event hooks ===", flush=True)
    handles, acc = attach_submodule_hooks(model)
    all_inp = inputs * ((N_WARMUP + N_MEASURE) // len(inputs) + 1)
    with torch.no_grad():
        for i, inp in enumerate(all_inp[:N_WARMUP + N_MEASURE]):
            torch.cuda.synchronize()
            transformer(*inp)
            torch.cuda.synchronize()
    for h in handles: h.remove()
    torch.cuda.synchronize()

    def mean_ms_acc(key):
        evts = acc.get(key, [])
        evts_warm = evts[N_WARMUP:] if len(evts) > N_WARMUP else evts
        if not evts_warm: return None
        return round(statistics.mean(s.elapsed_time(e) for s, e in evts_warm), 3)

    submod_keys = (["prior_feed", "sttf"] +
                   [f"L{d}_{m}" for d in range(3)
                    for m in ["hmsa_prenorm", "hmsa_inner", "mswin_prenorm", "mswin_inner", "ff"]])
    submod_ms = {k: mean_ms_acc(k) for k in submod_keys}
    submod_ms["v2xtransformer_total_hooks"] = mean_ms_acc("v2xtransformer_total")
    print(f"sub-module breakdown: {json.dumps(submod_ms, indent=2)}", flush=True)

    # ── 3. HMSA to_qkv micro-bench (wall-clock) ──────────────────────────────
    print("\n=== 3. HMSA to_qkv micro-bench ===", flush=True)
    hmsa_bench = hmsa_inner_wallclock(model, inputs)

    # ── 4. Patched (batched HMSA) ────────────────────────────────────────────
    print("\n=== 4. Patched batched HMSA ===", flush=True)
    try:
        saved = patch_all_hmsa(model, device)
        r_patched = cuda_time_fn(transformer, inputs, label="batched_hmsa")
        print(f"patched: {r_patched}", flush=True)
        restore_hmsa(saved)
    except Exception:
        import traceback
        r_patched = {"label": "batched_hmsa", "error": traceback.format_exc()[:400]}
        print(f"patch failed: {r_patched['error'][:200]}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    base_ms = r_base.get("mean_ms")
    patched_ms = r_patched.get("mean_ms")
    speedup = round(base_ms / patched_ms, 2) if (base_ms and patched_ms) else None

    total_submod = sum(v or 0 for k, v in submod_ms.items()
                       if k != "v2xtransformer_total_hooks" and v is not None)
    unaccounted_ms = round((submod_ms.get("v2xtransformer_total_hooks") or 0) - total_submod, 1)

    report = {
        "device": torch.cuda.get_device_name(0),
        "input_shape": list(inputs[0][0].shape),
        "eager_baseline_ms": r_base,
        "submodule_breakdown_ms": submod_ms,
        "total_submod_ms": round(total_submod, 3),
        "unaccounted_ms": unaccounted_ms,
        "hmsa_to_qkv_microbench": hmsa_bench,
        "batched_hmsa_ms": r_patched,
        "speedup_batched_vs_eager": speedup,
        "gate": _gate(base_ms, patched_ms, speedup, submod_ms),
    }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== T1-S-alt v2 result ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[written] {OUT}", flush=True)


def _gate(base_ms, patched_ms, speedup, submod_ms):
    lines = []
    total_h = submod_ms.get("v2xtransformer_total_hooks") or base_ms or 1
    hmsa_sum  = sum(submod_ms.get(f"L{d}_hmsa_prenorm") or 0 for d in range(3))
    mswin_sum = sum(submod_ms.get(f"L{d}_mswin_prenorm") or 0 for d in range(3))
    ff_sum    = sum(submod_ms.get(f"L{d}_ff") or 0 for d in range(3))
    total_sub = hmsa_sum + mswin_sum + ff_sum + (submod_ms.get("prior_feed") or 0) + (submod_ms.get("sttf") or 0)
    unaccounted = round(total_h - total_sub, 1)

    lines.append(f"HMSA×3={hmsa_sum:.1f}ms  MSwin×3={mswin_sum:.1f}ms  FF×3={ff_sum:.1f}ms  "
                 f"total_sub={total_sub:.1f}ms  unaccounted={unaccounted:.1f}ms (of {total_h:.1f}ms total)")

    if speedup is not None:
        if speedup >= 3.0:
            lines.append(f"S_axis_ALT_VIABLE: {speedup:.1f}x via batched HMSA → Python-GIL is dominant")
        elif speedup >= 1.5:
            lines.append(f"S_axis_ALT_PARTIAL: {speedup:.1f}x batched HMSA; check MSwin for remaining overhead")
        else:
            lines.append(f"S_axis_WEAK: {speedup:.1f}x; HMSA loop NOT dominant; "
                         f"look at unaccounted={unaccounted:.1f}ms")

    if unaccounted > 0.5 * total_h:
        lines.append(f"WARNING: {unaccounted:.0f}ms ({100*unaccounted/total_h:.0f}%) unaccounted "
                     f"— likely in SplitAttn inside MSwin or get_roi_mask; needs deeper profiling")
    return " | ".join(lines)


if __name__ == "__main__":
    main()
