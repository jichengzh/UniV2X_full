"""T0: V2X-ViT fusion operator-level breakdown (Phase T0 of plan_transformer_into_framework_v1).

Drills into V2XViTFusion.forward() to account for all 216ms.
The e2e_breakdown.json reports fusion_net=216ms, but the existing fusion_profile.json
only captured the inner V2XTransformer (self.fusion_net) = ~27ms.
This script finds where the remaining ~189ms goes by timing each operation
inside V2XViTFusion.forward().

Operations timed:
  regroup          - Regroup(x, record_len, L)
  prior_enc        - zeros+repeat for prior encoding
  cat              - torch.cat([regroup_feature, prior_encoding], dim=2)
  warp_affine      - per-batch warp_affine_simple loop (B iterations)
  permute_in       - regroup_feature.permute(0,1,3,4,2)
  fusion_net       - V2XTransformer forward (inner self.fusion_net)
  permute_out      - fused_feature.permute(0,3,1,2)
  outer_total      - full V2XViTFusion.forward() (should sum to ~216ms)

Gate criteria (per plan §1 Phase T0):
  ① GEMM hotspot (QKV/FFN matmul) → P/Q/S all 3 axes viable
  ② window partition / Python overhead → S axis (compile/fuse) first
  ③ compile reduces bottleneck → fall back to conv framework

Run:
  CUDA_VISIBLE_DEVICES=5 python scripts/phase2/t0_v2xvit_fusion_oplevel.py

Output: results/v2xvit_fusion_oplevel_breakdown.json
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
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
N_WARMUP = 10
N_SAMPLES = 60
OUT = REPO_ROOT / "results" / "v2xvit_fusion_oplevel_breakdown.json"


# ── Patched V2XViTFusion.forward ────────────────────────────────────────────

def _make_timed_forward(acc: dict):
    """Return a replacement forward() that inserts CUDA-event timers."""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.models.fuse_modules.fusion_in_one import Regroup

    def _rec():
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        return s, e

    def forward(self_module, x, record_len, affine_matrix):
        # ── outer total ──────────────────────────────────────────────────────
        s_tot, e_tot = _rec(); s_tot.record()

        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]

        # ── regroup ──────────────────────────────────────────────────────────
        s, e = _rec(); s.record()
        regroup_feature, mask = Regroup(x, record_len, L)
        e.record()
        acc["regroup"].append((s, e))

        # ── prior_enc (zeros + repeat) ────────────────────────────────────────
        s, e = _rec(); s.record()
        prior_encoding = torch.zeros(len(record_len), L, 3, 1, 1).to(record_len.device)
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])
        e.record()
        acc["prior_enc"].append((s, e))

        # ── cat ──────────────────────────────────────────────────────────────
        s, e = _rec(); s.record()
        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        e.record()
        acc["cat"].append((s, e))

        # ── warp_affine (Python for-loop over B) ─────────────────────────────
        s, e = _rec(); s.record()
        regroup_feature_new = []
        for b in range(B):
            ego = 0
            regroup_feature_new.append(
                warp_affine_simple(regroup_feature[b], affine_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)
        e.record()
        acc["warp_affine"].append((s, e))

        # ── permute_in (b l c h w -> b l h w c) ──────────────────────────────
        s, e = _rec(); s.record()
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        e.record()
        acc["permute_in"].append((s, e))

        # ── spatial_correction creation ───────────────────────────────────────
        s, e = _rec(); s.record()
        spatial_correction_matrix = (
            torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device))
        e.record()
        acc["spatial_corr"].append((s, e))

        # ── fusion_net inner (V2XTransformer) ────────────────────────────────
        s, e = _rec(); s.record()
        fused_feature = self_module.fusion_net(
            regroup_feature, mask, spatial_correction_matrix)
        e.record()
        acc["fusion_net_inner"].append((s, e))

        # ── permute_out (b h w c -> b c h w) ──────────────────────────────────
        s, e = _rec(); s.record()
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        e.record()
        acc["permute_out"].append((s, e))

        # ── outer total end ───────────────────────────────────────────────────
        e_tot.record()
        acc["outer_total"].append((s_tot, e_tot))

        return fused_feature

    return forward


def main():
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
    model = train_utils.create_model(hypes)
    state = torch.load(CKPT_FILE, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    # Show tensor shape info
    print("=== model.fusion_net type ===", type(model.fusion_net).__name__, flush=True)
    print("=== model.fusion_net.fusion_net type ===",
          type(model.fusion_net.fusion_net).__name__, flush=True)

    # Patch model.fusion_net.forward with the timed version
    acc = defaultdict(list)
    import types
    model.fusion_net.forward = types.MethodType(_make_timed_forward(acc), model.fusion_net)

    # Also hook outer model.fusion_net to measure pre-/post-patch overhead
    outer_acc_ms = []
    outer_events = []

    def _pre(_m, _inp):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        outer_events.append((s, e))

    def _post(_m, _inp, _out):
        outer_events[-1][1].record()

    h1 = model.fusion_net.register_forward_pre_hook(_pre)
    h2 = model.fusion_net.register_forward_hook(_post)

    dataset = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test,
                        shuffle=False, pin_memory=False, drop_last=False)
    print(f"[t0] val size={len(dataset)}", flush=True)

    device = torch.device("cuda")
    n_seen = 0
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_seen >= N_WARMUP + N_SAMPLES:
                break
            batch_data = train_utils.to_device(batch_data, device)
            torch.cuda.synchronize()
            inference_utils.inference_intermediate_fusion(batch_data, model, dataset)
            torch.cuda.synchronize()
            n_seen += 1
            if n_seen % 10 == 0:
                print(f"[t0] {n_seen}/{N_WARMUP + N_SAMPLES}", flush=True)

    h1.remove()
    h2.remove()

    torch.cuda.synchronize()
    N = max(1, n_seen - N_WARMUP)

    def mean_ms(event_list):
        if not event_list:
            return 0.0
        total = sum(s.elapsed_time(e) for s, e in event_list[N_WARMUP:])
        return round(total / N, 3)

    ops = ["regroup", "prior_enc", "cat", "warp_affine",
           "permute_in", "spatial_corr", "fusion_net_inner",
           "permute_out", "outer_total"]

    per_op = {k: mean_ms(acc[k]) for k in ops}
    hook_outer = round(sum(s.elapsed_time(e)
                           for s, e in outer_events[N_WARMUP:]) / N, 3)

    # ── summary ─────────────────────────────────────────────────────────────
    ref_ms = per_op["outer_total"]
    accounted = sum(per_op[k] for k in ops if k != "outer_total")
    unaccounted = round(ref_ms - accounted, 3)

    report = {
        "model": "v2xvit (V2XViTFusion outer + V2XTransformer inner)",
        "n_samples": N,
        "device": torch.cuda.get_device_name(0),
        "per_op_ms_mean": per_op,
        "hook_outer_ms_mean": hook_outer,
        "accounted_ms": round(accounted, 3),
        "unaccounted_ms": unaccounted,
        "pct_of_outer": {
            k: round(100 * per_op[k] / ref_ms, 1) if ref_ms else None
            for k in ops if k != "outer_total"
        },
        "gate_decision": _gate(per_op),
    }

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== T0 V2X-ViT fusion operator-level breakdown ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"\n[written] {OUT}", flush=True)


def _gate(per_op):
    """Apply go/no-go gate criteria from plan §1 Phase T0."""
    total = per_op.get("outer_total", 1) or 1
    fusion_inner = per_op.get("fusion_net_inner", 0)
    warp = per_op.get("warp_affine", 0)
    regroup = per_op.get("regroup", 0)
    gemm_pct = round(100 * fusion_inner / total, 1)
    warp_pct = round(100 * warp / total, 1)
    regroup_pct = round(100 * regroup / total, 1)

    if gemm_pct >= 60:
        decision = "①_GEMM_dominant: P/Q/S all axes viable; proceed to T1"
    elif warp_pct + regroup_pct >= 60:
        decision = (
            f"②_Python_ops_dominant (warp={warp_pct}%+regroup={regroup_pct}%): "
            "bottleneck is warp_affine/Regroup not transformer GEMM; "
            "S-axis kernel fusion may help but P/Q axes less impactful; "
            "evaluate batching warp_affine or torch.compile first"
        )
    else:
        decision = (
            f"③_mixed (GEMM={gemm_pct}% warp+regroup={warp_pct+regroup_pct}%): "
            "both fusion inner and preprocessing contribute; proceed to T1 for both axes"
        )
    return {
        "gemm_inner_pct": gemm_pct,
        "warp_pct": warp_pct,
        "regroup_pct": regroup_pct,
        "verdict": decision,
    }


if __name__ == "__main__":
    main()
