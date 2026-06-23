"""T1-S: Attention backend feasibility — torch.compile on V2XTransformer.

T0 showed: fusion_net_inner (V2XTransformer) = 204ms = 98.2% of outer wrapper.
Sub-module hooks only captured 27ms GPU kernel time. The ~177ms gap is Python-GIL
overhead from HMSA's per-agent for-loop serializing many small Linear kernels.

This script tests whether torch.compile eliminates the Python overhead:
  eager  : model.fusion_net.fusion_net (V2XTransformer)
  compile: torch.compile(model.fusion_net.fusion_net, mode=...)

Strategy: collect N input tensors first (by patching V2XViTFusion.forward
to save (regroup_feature, mask, scm) tuples), then time each mode in isolation.

Gate criteria from plan §1 Phase T1:
  ≥2× speedup → S axis viable → proceed to T2
  compile fails → investigate flash-attn / TRT attention plugin

Run:
  CUDA_VISIBLE_DEVICES=5 python scripts/phase2/t1s_v2xvit_compile_feasibility.py

Output: results/attention_axis_feasibility_v1.json
"""
import os
import sys
import json
import types
from pathlib import Path

import torch

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
N_COLLECT = 50   # number of input samples to collect
N_WARMUP = 5
N_MEASURE = 40
OUT = REPO_ROOT / "results" / "attention_axis_feasibility_v1.json"


# ── Step 1: Collect inputs to V2XTransformer ────────────────────────────────

def collect_inputs(model, loader, dataset, device, n=N_COLLECT):
    """Run V2XViTFusion.forward up to the inner fusion_net call, save inputs."""
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
            ego = 0
            regroup_feature_new.append(
                warp_affine_simple(regroup_feature[b], affine_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        scm = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        # Save copies for later timing (detach to avoid autograd issues)
        if len(saved) < n:
            saved.append((
                regroup_feature.detach().clone(),
                mask.detach().clone(),
                scm.detach().clone(),
            ))
        # Still run original to keep model state consistent (not strictly needed for eval)
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
            if n_seen % 10 == 0:
                print(f"[collect] {len(saved)}/{n}", flush=True)

    # Restore by deleting instance attribute
    try:
        del model.fusion_net.forward
    except AttributeError:
        pass

    print(f"[collect] collected {len(saved)} input tuples, "
          f"regroup shape={saved[0][0].shape if saved else 'N/A'}", flush=True)
    return saved


# ── Step 2: Time a callable on pre-collected inputs ─────────────────────────

def time_callable(fn, inputs, n_warmup=N_WARMUP, n_measure=N_MEASURE, label=""):
    """CUDA-event time fn(*inputs[i]) averaged over n_measure samples."""
    device = inputs[0][0].device
    events = []
    errors = 0
    first_error = None

    all_inputs = inputs * ((n_warmup + n_measure) // len(inputs) + 1)
    total = n_warmup + n_measure

    with torch.no_grad():
        for i, (x, mask, scm) in enumerate(all_inputs[:total]):
            torch.cuda.synchronize()
            try:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = fn(x, mask, scm)
                e.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    events.append(s.elapsed_time(e))
            except Exception as ex:
                errors += 1
                if first_error is None:
                    first_error = str(ex)[:200]

    if not events:
        return {"error": first_error or "no events", "n_errors": errors, "label": label}

    import statistics
    mean_ms = statistics.mean(events)
    p50 = sorted(events)[len(events) // 2]
    return {
        "label": label,
        "n_samples": len(events),
        "n_errors": errors,
        "first_error": first_error,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(p50, 3),
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

    print(f"[t1s] torch={torch.__version__}  compile={hasattr(torch,'compile')}", flush=True)
    print(f"[t1s] fusion_net type: {type(model.fusion_net).__name__}", flush=True)
    print(f"[t1s] fusion_net.fusion_net type: {type(model.fusion_net.fusion_net).__name__}", flush=True)

    dataset = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    device = torch.device("cuda")

    # ── Collect inputs ────────────────────────────────────────────────────────
    print("\n[t1s] Collecting inputs to V2XTransformer ...", flush=True)
    inputs = collect_inputs(model, loader, dataset, device, n=N_COLLECT)
    if not inputs:
        print("[t1s] ERROR: no inputs collected", flush=True)
        return

    transformer = model.fusion_net.fusion_net  # V2XTransformer

    # ── Time eager ────────────────────────────────────────────────────────────
    print("\n[t1s] === Timing eager ===", flush=True)
    r_eager = time_callable(transformer, inputs, label="eager")
    print(f"[t1s] eager: {r_eager}", flush=True)

    results = {"eager": r_eager}

    # ── torch.compile default (no CUDA Graphs) ────────────────────────────────
    # reduce-overhead uses CUDA Graphs which fails on data-dependent control flow
    # default mode uses Inductor without CUDA Graphs
    print("\n[t1s] === torch.compile(mode='default', dynamic=True) ===", flush=True)
    torch._dynamo.reset()
    try:
        compiled_df = torch.compile(transformer, mode="default",
                                    fullgraph=False, dynamic=True)
        r_df = time_callable(compiled_df, inputs, n_warmup=8, n_measure=N_MEASURE,
                             label="compile_default_dynamic")
    except Exception as ex:
        r_df = {"error": str(ex)[:300], "label": "compile_default_dynamic"}
    print(f"[t1s] default/dynamic: {r_df}", flush=True)
    results["compile_default_dynamic"] = r_df

    # ── torch.compile default static ─────────────────────────────────────────
    print("\n[t1s] === torch.compile(mode='default', dynamic=False) ===", flush=True)
    torch._dynamo.reset()
    try:
        compiled_st = torch.compile(transformer, mode="default",
                                    fullgraph=False, dynamic=False)
        r_st = time_callable(compiled_st, inputs, n_warmup=8, n_measure=N_MEASURE,
                             label="compile_default_static")
    except Exception as ex:
        r_st = {"error": str(ex)[:300], "label": "compile_default_static"}
    print(f"[t1s] default/static: {r_st}", flush=True)
    results["compile_default_static"] = r_st

    # ── torch.compile reduce-overhead (for record only) ──────────────────────
    r_ro = results.get("compile_reduce_overhead",
                       {"label": "compile_reduce_overhead", "note": "skipped"})
    r_ma = results.get("compile_max_autotune",
                       {"label": "compile_max_autotune", "note": "skipped"})

    # ── Speedup summary ───────────────────────────────────────────────────────
    eager_ms = r_eager.get("mean_ms")

    def speedup(r):
        v = r.get("mean_ms")
        if v and eager_ms:
            return round(eager_ms / v, 2)
        return None

    speedups = {
        "default_dynamic_vs_eager": speedup(r_df),
        "default_static_vs_eager": speedup(r_st),
    }

    gate = _s_gate(eager_ms, r_df, r_st)

    report = {
        "description": "T1-S: V2XTransformer torch.compile S-axis feasibility",
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "n_collect": len(inputs),
        "input_shape": list(inputs[0][0].shape),
        "axis": "S (compile/schedule)",
        "results": results,
        "speedups": speedups,
        "gate": gate,
        "note": ("reduce-overhead skipped: uses CUDA Graphs which fail on "
                 "data-dependent control flow in HMSA/STTF"),
    }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== T1-S Attention S-axis feasibility ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[written] {OUT}", flush=True)


def _s_gate(eager_ms, r_df, r_st):
    if eager_ms is None:
        return "inconclusive: eager baseline failed"
    best_ms = None
    best_label = None
    for r in [r_df, r_st]:
        v = r.get("mean_ms")
        if v and (best_ms is None or v < best_ms):
            best_ms = v
            best_label = r.get("label", "?")

    if best_ms is None:
        r_df_err = r_df.get("error") or r_df.get("first_error") or "unknown"
        return (f"S_axis_BLOCKED: torch.compile blocked by Dynamo incompatibilities "
                f"(STTF warp_affine, data-dependent branching). "
                f"Last error: {r_df_err[:120]}. "
                "Next: try manual HMSA batching (T1-S-alt) or TRT attention plugin.")

    speedup = eager_ms / best_ms
    if speedup >= 3.0:
        return (f"S_axis_VIABLE: {speedup:.1f}x speedup via {best_label}; "
                "Python-GIL overhead eliminated → proceed T2 (stage1 fusion adapter)")
    elif speedup >= 1.5:
        return (f"S_axis_PARTIAL: {speedup:.1f}x speedup via {best_label}; "
                "some Python overhead; T2 worth exploring alongside P/Q axes")
    else:
        return (f"S_axis_WEAK: {speedup:.1f}x via {best_label} (eager={eager_ms:.1f}ms); "
                "Python-GIL not dominant; GEMM may already be fast; "
                "consider flash-attn plugin for further S gain")


if __name__ == "__main__":
    main()
