"""T1-S-alt MSwin: test torch.compile + SDPA on BaseWindowAttention.

True bottleneck: MSwin×3 = 214ms = 92% of V2XTransformer (not HMSA).
Root cause: 3 pyramid window sizes (ws=4,8,16), ws=16 creates 256×256 attention
matrices → memory-bandwidth bound on 4090 (vs 3.3× more BW on H800).

BaseWindowAttention has NO Python loops (only einsum+rearrange) → Dynamo-friendly.

Tests:
  A. torch.compile(BaseWindowAttention, mode='default') — Triton fused kernels
  B. SDPA replacement: F.scaled_dot_product_attention replaces
     einsum('b l m h i c, b l m h j c -> b l m h i j', q, k) + softmax +
     einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
     This activates FlashAttention-2 on 4090, never materializes full attn matrix.
  C. Both: compile + SDPA

Gate:
  ≥2× speedup → S_axis_VIABLE for MSwin → proceed T2
  <1.5× → S_axis_WEAK; look at reducing pyramid levels (P-axis depth equivalent)

Run:
  CUDA_VISIBLE_DEVICES=5 python scripts/phase2/t1s_alt_mswin.py

Output: results/t1s_alt_mswin_v1.json
"""
import os
import sys
import json
import types
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
CKPT_FILE   = CKPT_DIR / "net_epoch_bestval_at17.pth"
N_COLLECT   = 30
N_WARMUP    = 8
N_MEASURE   = 20
OUT = REPO_ROOT / "results" / "t1s_alt_mswin_v1.json"


# ── collect inputs ────────────────────────────────────────────────────────────

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
        rf_new = [warp_affine_simple(rf[b], affine_matrix[b, 0], (H, W)) for b in range(B)]
        rf = torch.stack(rf_new).permute(0, 1, 3, 4, 2)
        scm = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        if len(saved) < n:
            saved.append((rf.detach().clone(), mask.detach().clone(), scm.detach().clone()))
        return self_outer.fusion_net(rf, mask, scm).permute(0, 3, 1, 2)

    model.fusion_net.forward = types.MethodType(patched_fwd, model.fusion_net)
    with torch.no_grad():
        for bd in loader:
            if bd is None or len(saved) >= n: break
            bd = train_utils.to_device(bd, device)
            torch.cuda.synchronize()
            try: inference_utils.inference_intermediate_fusion(bd, model, dataset)
            except: pass
            torch.cuda.synchronize()
    try: del model.fusion_net.forward
    except AttributeError: pass
    print(f"[collect] {len(saved)} inputs, shape={saved[0][0].shape}", flush=True)
    return saved


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
                    import traceback; first_err = traceback.format_exc()[:400]
    if not events:
        return {"label": label, "error": first_err or "no events", "n_errors": errors}
    ev = sorted(events)
    return {"label": label, "n": len(events), "n_errors": errors,
            "mean_ms": round(statistics.mean(events), 3),
            "p50_ms": round(ev[len(ev)//2], 3),
            "min_ms": round(min(events), 3), "max_ms": round(max(events), 3)}


# ── SDPA-patched BaseWindowAttention ─────────────────────────────────────────

def make_sdpa_forward(orig_bwa):
    """Replaces the two einsum calls with F.scaled_dot_product_attention.

    Original:
      dots = einsum('b l m h i c, b l m h j c -> b l m h i j', q, k) * scale
      out  = einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)

    SDPA-equivalent:
      Reshape q,k,v to (B_flat, heads, seq, dim_head) so SDPA can use
      FlashAttention-2 backend (never materializes n²×d attention matrix).
    """
    scale = orig_bwa.scale
    rel_pos = orig_bwa.relative_pos_embedding
    if rel_pos:
        rel_idx = orig_bwa.relative_indices
        pos_emb = orig_bwa.pos_embedding  # (2*ws-1, 2*ws-1) parameter

    def sdpa_forward(x):
        b, l, h, w, c = x.shape
        m = orig_bwa.heads
        # 1. QKV projection
        qkv = orig_bwa.to_qkv(x).chunk(3, dim=-1)
        ws = orig_bwa.window_size
        new_h = h // ws
        new_w = w // ws
        from einops import rearrange
        # q: (b, l, m, new_h*new_w, ws², c_head)
        q, k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=ws, w_w=ws), qkv)
        # 2. SDPA instead of manual einsum+softmax
        # q: (b, l, m, n_windows, ws², c_head) → (b*l*m*n_windows, 1, ws², c_head)
        # We use (batch, heads=1, seq=ws², dim=c_head) since heads already split
        b2, l2, m2, n_win, ws2, c_h = q.shape
        # Flatten to (B_flat, ws², c_head), then add a fake "heads" dim for SDPA
        B_flat = b2 * l2 * m2 * n_win
        q_flat = q.reshape(B_flat, 1, ws2, c_h)  # (B_flat, 1, ws², c_h)
        k_flat = k.reshape(B_flat, 1, ws2, c_h)
        v_flat = v.reshape(B_flat, 1, ws2, c_h)

        if rel_pos:
            # Build attn_bias from relative position embedding
            # pos_emb: (2*ws-1, 2*ws-1), rel_idx: (ws², ws², 2)
            bias = pos_emb[rel_idx[:, :, 0], rel_idx[:, :, 1]]  # (ws², ws²)
            # SDPA requires attn_bias shape (B_flat, 1, ws², ws²)
            bias = bias.unsqueeze(0).unsqueeze(0).expand(B_flat, 1, ws2, ws2)
            out_flat = F.scaled_dot_product_attention(
                q_flat, k_flat, v_flat,
                attn_mask=bias,
                scale=scale,
            )
        else:
            out_flat = F.scaled_dot_product_attention(
                q_flat, k_flat, v_flat,
                scale=scale,
            )
        # out_flat: (B_flat, 1, ws², c_h) → (b, l, m, n_win, ws², c_h)
        out = out_flat.reshape(b2, l2, m2, n_win, ws2, c_h)
        # Rearrange back
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=m2, w_h=ws, w_w=ws, new_w=new_w, new_h=new_h)
        return orig_bwa.to_out(out)

    return sdpa_forward


def patch_mswin_with_sdpa(model):
    """Replace all BaseWindowAttention.forward with SDPA version."""
    enc = model.fusion_net.fusion_net.encoder
    saved = []
    for d in range(len(enc.layers)):
        pwin_prenorm = enc.layers[d][0].layers[0][1]  # PreNorm(MSwin)
        pwa = getattr(pwin_prenorm, 'fn', pwin_prenorm)  # PyramidWindowAttention
        for i, bwa in enumerate(pwa.pwmsa):
            orig_fwd = bwa.forward
            sdpa_fwd = make_sdpa_forward(bwa)
            bwa.forward = sdpa_fwd
            saved.append((bwa, orig_fwd))
    print(f"[sdpa] patched {len(saved)} BaseWindowAttention modules", flush=True)
    return saved


def restore_mswin(saved):
    for bwa, orig_fwd in saved:
        bwa.forward = orig_fwd


# ── compile BaseWindowAttention ───────────────────────────────────────────────

def compile_mswin(model):
    """torch.compile each BaseWindowAttention independently."""
    enc = model.fusion_net.fusion_net.encoder
    saved = []
    for d in range(len(enc.layers)):
        pwin_prenorm = enc.layers[d][0].layers[0][1]
        pwa = getattr(pwin_prenorm, 'fn', pwin_prenorm)
        for i, bwa in enumerate(pwa.pwmsa):
            orig_cls_fwd = bwa.forward.__func__ if hasattr(bwa.forward, '__func__') else None
            try:
                compiled = torch.compile(bwa, mode='default', dynamic=True, fullgraph=False)
                pwa.pwmsa[i] = compiled
                saved.append((pwa, i, bwa))
                print(f"  L{d}/ws{i} compiled", flush=True)
            except Exception as ex:
                print(f"  L{d}/ws{i} compile FAILED: {ex}", flush=True)
    return saved


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
    model = train_utils.create_model(hypes)
    state = torch.load(CKPT_FILE, map_location="cpu")
    if "model_state_dict" in state: state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()

    # Print MSwin config
    enc = model.fusion_net.fusion_net.encoder
    pwa = enc.layers[0][0].layers[0][1].fn  # PyramidWindowAttention
    print(f"\n[mswin] PyramidWindowAttention: {len(pwa.pwmsa)} pyramid levels", flush=True)
    for i, bwa in enumerate(pwa.pwmsa):
        print(f"  level {i}: ws={bwa.window_size}, heads={bwa.heads}, "
              f"scale={bwa.scale:.4f}, rel_pos={bwa.relative_pos_embedding}", flush=True)

    dataset = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)

    print("\n[mswin] Collecting inputs ...", flush=True)
    inputs = collect_inputs(model, loader, dataset, device, n=N_COLLECT)
    if not inputs: return

    transformer = model.fusion_net.fusion_net

    # ── A. Baseline eager ────────────────────────────────────────────────────
    print("\n=== A. Baseline eager V2XTransformer ===", flush=True)
    r_base = cuda_time_fn(transformer, inputs, label="eager")
    print(f"baseline: {r_base}", flush=True)

    # ── B. SDPA replacement ──────────────────────────────────────────────────
    print("\n=== B. SDPA (FlashAttn-2) replacement ===", flush=True)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        saved_sdpa = patch_mswin_with_sdpa(model)
        r_sdpa = cuda_time_fn(transformer, inputs, label="sdpa")
        print(f"sdpa: {r_sdpa}", flush=True)
        restore_mswin(saved_sdpa)
    except Exception:
        import traceback
        r_sdpa = {"label": "sdpa", "error": traceback.format_exc()[:400]}
        print(f"sdpa failed: {r_sdpa['error'][:200]}", flush=True)

    # ── C. torch.compile on BaseWindowAttention ──────────────────────────────
    print("\n=== C. torch.compile BaseWindowAttention ===", flush=True)
    torch._dynamo.reset()
    try:
        saved_compile = compile_mswin(model)
        r_compile = cuda_time_fn(transformer, inputs, n_warmup=12, n_measure=N_MEASURE,
                                 label="compile_bwa")
        print(f"compile: {r_compile}", flush=True)
        # Restore: replace compiled back with original
        for pwa, i, orig_bwa in saved_compile:
            pwa.pwmsa[i] = orig_bwa
    except Exception:
        import traceback
        r_compile = {"label": "compile_bwa", "error": traceback.format_exc()[:400]}
        print(f"compile failed: {r_compile['error'][:200]}", flush=True)

    # ── D. SDPA + compile ────────────────────────────────────────────────────
    print("\n=== D. SDPA + compile ===", flush=True)
    torch._dynamo.reset()
    try:
        saved_sdpa2 = patch_mswin_with_sdpa(model)
        # Compile the SDPA version
        for d in range(len(enc.layers)):
            pwin_prenorm = enc.layers[d][0].layers[0][1]
            pwa_d = getattr(pwin_prenorm, 'fn', pwin_prenorm)
            for i in range(len(pwa_d.pwmsa)):
                pwa_d.pwmsa[i] = torch.compile(pwa_d.pwmsa[i], mode='default', dynamic=True)
        r_both = cuda_time_fn(transformer, inputs, n_warmup=12, n_measure=N_MEASURE,
                              label="sdpa_compile")
        print(f"sdpa+compile: {r_both}", flush=True)
        restore_mswin(saved_sdpa2)
    except Exception:
        import traceback
        r_both = {"label": "sdpa_compile", "error": traceback.format_exc()[:400]}
        print(f"sdpa+compile failed: {r_both['error'][:200]}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    base_ms = r_base.get("mean_ms")

    def sp(r):
        v = r.get("mean_ms")
        return round(base_ms / v, 2) if (v and base_ms) else None

    report = {
        "device": torch.cuda.get_device_name(0),
        "input_shape": list(inputs[0][0].shape),
        "pyramid_levels": len(pwa.pwmsa),
        "window_sizes": [bwa.window_size for bwa in pwa.pwmsa],
        "fuse_method": pwa.fuse_mehod,
        "eager_ms": r_base,
        "sdpa_ms": r_sdpa,
        "compile_ms": r_compile,
        "sdpa_compile_ms": r_both,
        "speedups": {
            "sdpa_vs_eager": sp(r_sdpa),
            "compile_vs_eager": sp(r_compile),
            "sdpa_compile_vs_eager": sp(r_both),
        },
        "gate": _gate(base_ms, r_sdpa, r_compile, r_both),
    }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== T1-S-alt MSwin result ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[written] {OUT}", flush=True)


def _gate(base_ms, r_sdpa, r_compile, r_both):
    best = None; best_label = ""; best_ms = None
    for r in [r_sdpa, r_compile, r_both]:
        v = r.get("mean_ms")
        if v and (best_ms is None or v < best_ms):
            best_ms = v; best_label = r.get("label", "?")
    if best_ms is None:
        errors = {r.get('label',''): r.get('error','?')[:80]
                  for r in [r_sdpa, r_compile, r_both] if 'error' in r}
        return f"S_axis_MSWIN_BLOCKED: all methods failed — {errors}"
    speedup = round(base_ms / best_ms, 2)
    if speedup >= 2.0:
        return (f"S_axis_VIABLE: {speedup:.1f}x via {best_label}; "
                "MSwin attention is memory-BW bound, FlashAttn/compile helps → T2")
    elif speedup >= 1.3:
        return (f"S_axis_PARTIAL: {speedup:.1f}x via {best_label}; "
                "significant but not dominant gain; also consider P-axis (reduce pyramid levels)")
    else:
        return (f"S_axis_WEAK: {speedup:.1f}x; attention already near-optimal on 4090; "
                "bottleneck may be inherent compute — consider P-axis pyramid level reduction")


if __name__ == "__main__":
    main()
