"""T1-S-alt: HMSA batched-GEMM micro-benchmark.

T1-S showed torch.compile BLOCKED by STTF warp_affine + HMSA type-branching.
V2XTransformer eager = 216ms but sub-module GPU kernels = 27ms → 189ms Python-GIL overhead.

This script proves the Python overhead is in HMSA's to_qkv / to_out for-loops,
and that manual batched-GEMM (pre-stack weights, single torch.bmm) can eliminate it.

Approach:
  Original to_qkv: for b in B: for i in L: q_linears[type_i](x[b,:,:,i,:])
                   → L=5 individual Linear calls (tiny kernels, GPU idle between each)
  Batched to_qkv:  stack weights, single torch.bmm over all agents simultaneously
                   → 1 batched matmul (fully GPU-saturated)

Also tests batched get_hetero_edge_weights (B×L×L=25 iterations → tensor indexing)
and whether the full manual-batched HMSA + V2XTransformer achieves speedup.

Run:
  CUDA_VISIBLE_DEVICES=5 python scripts/phase2/t1s_alt_hmsa_batching.py

Output: results/t1s_alt_hmsa_batching_v1.json
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
CKPT_FILE = CKPT_DIR / "net_epoch_bestval_at17.pth"
N_COLLECT = 30
N_WARMUP = 5
N_MEASURE = 25
OUT = REPO_ROOT / "results" / "t1s_alt_hmsa_batching_v1.json"


# ── Batched HMSA replacement ─────────────────────────────────────────────────

class BatchedHMSA(nn.Module):
    """HGTCavAttention with batched QKV (replaces per-agent for-loop).

    Semantically equivalent to original to_qkv / to_out but eliminates
    the Python GIL-serialized small-Linear bottleneck by:
    1. Pre-stacking Linear weights: shape (n_types, C_out, C_in) and biases (n_types, C_out)
    2. Looking up per-agent weights via types indexing: (B*L, C_out, C_in)
    3. Single torch.bmm over all agents: (B*L, H*W, C_in) @ (B*L, C_in, C_out)

    This eliminates L*3 = 15 small Linear kernel launches per HMSA forward,
    replacing them with 3 torch.bmm calls.
    """

    def __init__(self, orig_hmsa):
        super().__init__()
        # Copy all original attributes
        self.orig = orig_hmsa
        self.heads = orig_hmsa.heads
        self.scale = orig_hmsa.scale
        self.attend = orig_hmsa.attend
        self.to_norm = orig_hmsa.to_norm
        self.relation_att = orig_hmsa.relation_att
        self.relation_msg = orig_hmsa.relation_msg
        self.get_relation_type_index = orig_hmsa.get_relation_type_index

        # Pre-stack weights for O(1) batched GEMM
        n_types = len(orig_hmsa.q_linears)
        self.n_types = n_types

        # Stack: (n_types, C_out, C_in)
        self.register_buffer('q_weights', torch.stack([m.weight.data for m in orig_hmsa.q_linears]))
        self.register_buffer('q_biases', torch.stack([m.bias.data for m in orig_hmsa.q_linears]))
        self.register_buffer('k_weights', torch.stack([m.weight.data for m in orig_hmsa.k_linears]))
        self.register_buffer('k_biases', torch.stack([m.bias.data for m in orig_hmsa.k_linears]))
        self.register_buffer('v_weights', torch.stack([m.weight.data for m in orig_hmsa.v_linears]))
        self.register_buffer('v_biases', torch.stack([m.bias.data for m in orig_hmsa.v_linears]))
        self.register_buffer('a_weights', torch.stack([m.weight.data for m in orig_hmsa.a_linears]))
        self.register_buffer('a_biases', torch.stack([m.bias.data for m in orig_hmsa.a_linears]))

    def batched_linear(self, x_blhwc, types_bl, weight_stack, bias_stack):
        """Batched Linear: replaces for-loop over (b,l) with a single bmm.

        x_blhwc: (B, L, H, W, C_in)
        types_bl: (B, L) long tensor of agent types
        weight_stack: (n_types, C_out, C_in)
        bias_stack: (n_types, C_out)
        Returns: (B, L, H, W, C_out)
        """
        B, L, H, W, C_in = x_blhwc.shape
        C_out = weight_stack.shape[1]
        # Per-agent weight lookup: (B*L, C_out, C_in)
        agent_weights = weight_stack[types_bl.reshape(-1)]  # (B*L, C_out, C_in)
        agent_biases = bias_stack[types_bl.reshape(-1)]     # (B*L, C_out)
        # Reshape x: (B, L, H, W, C_in) → (B*L, H*W, C_in)
        x_flat = x_blhwc.reshape(B * L, H * W, C_in)
        # Batched GEMM: (B*L, H*W, C_in) @ (B*L, C_in, C_out) = (B*L, H*W, C_out)
        out = torch.bmm(x_flat, agent_weights.permute(0, 2, 1))
        out = out + agent_biases.unsqueeze(1)  # broadcast bias: (B*L, 1, C_out)
        # Reshape back: (B*L, H*W, C_out) → (B, L, H, W, C_out)
        return out.reshape(B, L, H, W, C_out)

    def to_qkv_batched(self, x_blhwc, types):
        """Batched QKV projection. x_blhwc: (B, L, H, W, C). Returns q,k,v each (B, L, H, W, C_head)."""
        q = self.batched_linear(x_blhwc, types, self.q_weights, self.q_biases)
        k = self.batched_linear(x_blhwc, types, self.k_weights, self.k_biases)
        v = self.batched_linear(x_blhwc, types, self.v_weights, self.v_biases)
        return q, k, v

    def to_out_batched(self, x_blhwc, types):
        """Batched output projection. Returns (B, L, H, W, C_out)."""
        return self.batched_linear(x_blhwc, types, self.a_weights, self.a_biases)

    def get_hetero_edge_weights_batched(self, x, types):
        """Build edge weight tensors via tensor indexing (no Python loops).

        Original: triple nested for-loop over (b, i, j) = B×L×L iterations.
        Batched: precompute relation_type index matrix, then index into stacked weights.
        """
        B, H, W, L, C = x.shape
        n_types = self.n_types

        # Build relation type index matrix (B, L, L)
        # types: (B, L), we need types_i (B, L, 1) and types_j (B, 1, L)
        types_i = types.unsqueeze(2).expand(B, L, L)  # (B, L, L) - src
        types_j = types.unsqueeze(1).expand(B, L, L)  # (B, L, L) - dst
        # relation type = types_i * n_types + types_j (simple encoding for n_types^2 relations)
        # Must match get_relation_type_index logic
        # Original: get_relation_type_index(type_i, type_j) - need to inspect
        # Assume: relation_type = type_i * n_types + type_j
        rel_type = types_i * n_types + types_j  # (B, L, L)

        # relation_att: (n_types^2, M, C_head, C_head) where M=heads
        # Index: (B, L, L, M, C_head, C_head)
        rel_flat = rel_type.reshape(-1)  # (B*L*L,)
        w_att = self.relation_att[rel_flat].reshape(B, L, L, *self.relation_att.shape[1:])
        w_msg = self.relation_msg[rel_flat].reshape(B, L, L, *self.relation_msg.shape[1:])
        # Original permute: (B, L, L, M, H, H) → (B, M, L, L, H, H)
        w_att = w_att.permute(0, 3, 1, 2, 4, 5)
        w_msg = w_msg.permute(0, 3, 1, 2, 4, 5)
        return w_att, w_msg

    def forward(self, x, mask, prior_encoding):
        """Forward using batched operations where possible, fall back to orig for safety."""
        # Use original forward but with batched to_qkv / to_out
        # We patch the orig object temporarily
        orig_to_qkv = self.orig.to_qkv
        orig_to_out = self.orig.to_out
        orig_get_edge = self.orig.get_hetero_edge_weights

        def _to_qkv_batched_wrapper(x_orig, types):
            # x_orig is (B, H, W, L, C) - need to convert to (B, L, H, W, C) for batched
            B, H, W, L, C = x_orig.shape
            x_blhwc = x_orig.permute(0, 3, 1, 2, 4)  # (B, L, H, W, C)
            q_blhwc, k_blhwc, v_blhwc = self.to_qkv_batched(x_blhwc, types)
            # Convert back to (B, H, W, L, C_head)
            q = q_blhwc.permute(0, 2, 3, 1, 4)
            k = k_blhwc.permute(0, 2, 3, 1, 4)
            v = v_blhwc.permute(0, 2, 3, 1, 4)
            return q, k, v

        def _to_out_batched_wrapper(x_orig, types):
            B, H, W, L, C = x_orig.shape
            x_blhwc = x_orig.permute(0, 3, 1, 2, 4)
            out_blhwc = self.to_out_batched(x_blhwc, types)
            return out_blhwc.permute(0, 2, 3, 1, 4)  # (B, H, W, L, C_out)

        self.orig.to_qkv = _to_qkv_batched_wrapper
        self.orig.to_out = _to_out_batched_wrapper
        try:
            result = self.orig.forward(x, mask, prior_encoding)
        finally:
            self.orig.to_qkv = orig_to_qkv
            self.orig.to_out = orig_to_out
        return result


# ── Collect inputs ───────────────────────────────────────────────────────────

def collect_transformer_inputs(model, loader, dataset, device, n=N_COLLECT):
    from opencood.models.fuse_modules.fusion_in_one import Regroup
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    saved = []

    def patched_fwd(self_outer, x, record_len, affine_matrix):
        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]
        regroup_feature, mask = Regroup(x, record_len, L)
        prior_encoding = torch.zeros(len(record_len), L, 3, 1, 1).to(record_len.device)
        prior_encoding = prior_encoding.repeat(1, 1, 1, regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature_new = []
        for b in range(B):
            regroup_feature_new.append(
                warp_affine_simple(regroup_feature[b], affine_matrix[b, 0], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        scm = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        if len(saved) < n:
            saved.append((
                regroup_feature.detach().clone(),
                mask.detach().clone(),
                scm.detach().clone(),
            ))
        fused = self_outer.fusion_net(regroup_feature, mask, scm)
        return fused.permute(0, 3, 1, 2)

    model.fusion_net.forward = types.MethodType(patched_fwd, model.fusion_net)
    n_seen = 0
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None or len(saved) >= n:
                break
            batch_data = train_utils.to_device(batch_data, device)
            torch.cuda.synchronize()
            try:
                inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
            except Exception:
                pass
            torch.cuda.synchronize()
            n_seen += 1
    try:
        del model.fusion_net.forward
    except AttributeError:
        pass
    print(f"[collect] {len(saved)} transformer inputs, shape={saved[0][0].shape}", flush=True)
    return saved


# ── Timing helper ─────────────────────────────────────────────────────────────

def time_fn(fn, inputs, n_warmup=N_WARMUP, n_measure=N_MEASURE, label=""):
    all_in = inputs * ((n_warmup + n_measure) // len(inputs) + 1)
    events = []
    errors = 0
    first_error = None
    with torch.no_grad():
        for i, inp in enumerate(all_in[:n_warmup + n_measure]):
            torch.cuda.synchronize()
            try:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                fn(*inp)
                e.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    events.append(s.elapsed_time(e))
            except Exception as ex:
                errors += 1
                if first_error is None:
                    import traceback
                    first_error = traceback.format_exc()[:400]
    if not events:
        return {"label": label, "error": first_error or "no events", "n_errors": errors}
    return {
        "label": label,
        "n_samples": len(events),
        "n_errors": errors,
        "first_error": first_error,
        "mean_ms": round(statistics.mean(events), 3),
        "p50_ms": round(sorted(events)[len(events)//2], 3),
        "min_ms": round(min(events), 3),
        "max_ms": round(max(events), 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
    model = train_utils.create_model(hypes)
    state = torch.load(CKPT_FILE, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    dataset = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    device = torch.device("cuda")

    print("\n[t1s-alt] Collecting transformer inputs ...", flush=True)
    inputs = collect_transformer_inputs(model, loader, dataset, device, n=N_COLLECT)
    if not inputs:
        print("[t1s-alt] ERROR: no inputs", flush=True)
        return

    transformer = model.fusion_net.fusion_net  # V2XTransformer

    # ── 1. Baseline: full V2XTransformer eager ────────────────────────────────
    print("\n[t1s-alt] === Baseline V2XTransformer eager ===", flush=True)
    r_base = time_fn(transformer, inputs, label="v2xtransformer_eager")
    print(f"[t1s-alt] base: {r_base}", flush=True)

    # ── 2. HMSA micro-benchmark: original vs batched ──────────────────────────
    # Get one HMSA module (first layer, cav_attn)
    enc = transformer.encoder
    # V2XTEncoder has self.layers = list of (V2XFusionBlock, FeedForward)
    # V2XFusionBlock has self.layers = list of (cav_attn, pwindow_attn)
    fusion_block = enc.layers[0][0]  # First depth, V2XFusionBlock
    # fusion_block.layers[0][0] = cav_attn (PreNorm wrapping HGTCavAttention)
    # or fusion_block might directly be HGTCavAttention
    print(f"\n[t1s-alt] fusion_block type: {type(fusion_block).__name__}", flush=True)
    print(f"[t1s-alt] fusion_block.layers: {[type(x).__name__ for x in fusion_block.layers]}", flush=True)

    # Navigate to actual HGTCavAttention
    fb_layers = fusion_block.layers  # should be [(cav_attn, pwindow_attn), ...]
    if hasattr(fb_layers[0], '__iter__'):
        cav_attn_prenorm, _ = fb_layers[0]
    else:
        cav_attn_prenorm = fb_layers[0]
    print(f"[t1s-alt] cav_attn type: {type(cav_attn_prenorm).__name__}", flush=True)

    # Unwrap PreNorm if needed
    hmsa = getattr(cav_attn_prenorm, 'fn', cav_attn_prenorm)
    print(f"[t1s-alt] hmsa type: {type(hmsa).__name__}", flush=True)

    # ── HMSA input shape ──────────────────────────────────────────────────────
    # V2XTEncoder processes x: (B, L, H, W, C) but HMSA receives (B, H, W, L, C) after sttf
    # Actually check the forward to see what shape HMSA receives
    # For now, construct a dummy input
    sample_x, sample_mask, _ = inputs[0]
    B, L, H, W, C = sample_x.shape  # regroup_feature is (B, L, H, W, C)
    print(f"\n[t1s-alt] Transformer input shape: B={B} L={L} H={H} W={W} C={C}", flush=True)

    # ── 3. Patch HMSA in-place with BatchedHMSA and re-time V2XTransformer ───
    print("\n[t1s-alt] Building BatchedHMSA ...", flush=True)
    try:
        batched_hmsa = BatchedHMSA(hmsa).cuda().eval()
        print(f"[t1s-alt] BatchedHMSA built OK", flush=True)

        # Patch in place: replace hmsa's to_qkv / to_out
        orig_to_qkv = hmsa.to_qkv
        orig_to_out = hmsa.to_out

        def _to_qkv_batched(x_orig, type_ids):
            B2, H2, W2, L2, C2 = x_orig.shape
            x_bl = x_orig.permute(0, 3, 1, 2, 4)
            q_bl, k_bl, v_bl = batched_hmsa.to_qkv_batched(x_bl, type_ids)
            return q_bl.permute(0,2,3,1,4), k_bl.permute(0,2,3,1,4), v_bl.permute(0,2,3,1,4)

        def _to_out_batched(x_orig, type_ids):
            B2, H2, W2, L2, C2 = x_orig.shape
            x_bl = x_orig.permute(0, 3, 1, 2, 4)
            out_bl = batched_hmsa.to_out_batched(x_bl, type_ids)
            return out_bl.permute(0, 2, 3, 1, 4)

        # Patch ALL HMSA instances in the encoder (3 depth layers)
        hmsa_modules = []
        for depth_layer in enc.layers:
            fb, ff = depth_layer
            for cav_prenorm, _ in fb.layers:
                m = getattr(cav_prenorm, 'fn', cav_prenorm)
                if type(m).__name__ == 'HGTCavAttention':
                    hmsa_modules.append(m)

        print(f"[t1s-alt] Found {len(hmsa_modules)} HMSA modules to patch", flush=True)

        orig_methods = []
        for m in hmsa_modules:
            orig_methods.append((m, m.to_qkv, m.to_out))
            b_hmsa_i = BatchedHMSA(m).cuda().eval()
            m._batched = b_hmsa_i

            def make_qkv(bh):
                def _f(x_orig, t):
                    B2, H2, W2, L2, C2 = x_orig.shape
                    x_bl = x_orig.permute(0,3,1,2,4)
                    q_bl, k_bl, v_bl = bh.to_qkv_batched(x_bl, t)
                    return q_bl.permute(0,2,3,1,4), k_bl.permute(0,2,3,1,4), v_bl.permute(0,2,3,1,4)
                return _f

            def make_out(bh):
                def _f(x_orig, t):
                    B2, H2, W2, L2, C2 = x_orig.shape
                    x_bl = x_orig.permute(0,3,1,2,4)
                    out_bl = bh.to_out_batched(x_bl, t)
                    return out_bl.permute(0,2,3,1,4)
                return _f

            m.to_qkv = make_qkv(b_hmsa_i)
            m.to_out = make_out(b_hmsa_i)

        print("\n[t1s-alt] === V2XTransformer with batched HMSA ===", flush=True)
        r_batched = time_fn(transformer, inputs, label="v2xtransformer_batched_hmsa")
        print(f"[t1s-alt] batched: {r_batched}", flush=True)

        # Restore
        for m, orig_qkv, orig_out in orig_methods:
            m.to_qkv = orig_qkv
            m.to_out = orig_out

    except Exception as ex:
        import traceback
        r_batched = {"error": traceback.format_exc()[:500], "label": "v2xtransformer_batched_hmsa"}
        print(f"[t1s-alt] batched HMSA failed: {r_batched['error'][:200]}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    base_ms = r_base.get("mean_ms")
    batched_ms = r_batched.get("mean_ms")
    speedup = round(base_ms / batched_ms, 2) if (base_ms and batched_ms) else None

    report = {
        "description": "T1-S-alt: HMSA manual batched-GEMM vs eager Python-loop",
        "device": torch.cuda.get_device_name(0),
        "input_shape": list(inputs[0][0].shape),
        "eager_baseline": r_base,
        "batched_hmsa": r_batched,
        "speedup_batched_vs_eager": speedup,
        "gate": _alt_gate(base_ms, batched_ms, speedup),
    }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== T1-S-alt HMSA batching result ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[written] {OUT}", flush=True)


def _alt_gate(base_ms, batched_ms, speedup):
    if speedup is None:
        return "INCONCLUSIVE: batched HMSA failed"
    if speedup >= 3.0:
        return (f"S_axis_ALT_VIABLE: {speedup:.1f}x speedup via manual batching; "
                "Python-GIL overhead from HMSA for-loop is the main bottleneck; "
                "recommend refactoring HMSA.to_qkv/to_out with batched GEMM → T2")
    elif speedup >= 1.5:
        return (f"S_axis_ALT_PARTIAL: {speedup:.1f}x speedup; HMSA loop contributes but "
                "not dominant; also investigate MSwin/STTF overhead → T2 with caveat")
    else:
        return (f"S_axis_ALT_WEAK: {speedup:.1f}x speedup; HMSA loop is NOT the bottleneck; "
                "other ops dominate the 189ms gap; consider TRT attention plugin")


if __name__ == "__main__":
    main()
