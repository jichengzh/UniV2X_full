"""【A-2 正式授权 — team-lead 2026-06-04】A-2 批准启动
A-2 eval — V2X-ViT 剪枝档 AP+mAOE eval (A-1 口径).

与 A-1 区别: 先 rebuild 剪枝模型结构, 再加载 finetuned bestval ckpt.
口径: fp32_pytorch (同 A-1), DAIR val 1789, 相同 bootstrap CI.

用法:
  CUDA_VISIBLE_DEVICES=2 python scripts/phase2/eval_v2xvit_a2_pruned.py \\
      --ratio 0.75 \\
      --finetune-ckpt output/a2_finetune/v2xvit_bb_p75/net_epoch16_bestval.pth \\
      --epoch-used "bestval_at16_finetuned" \\
      --actual-filters "64,32,64" \\
      --out-csv results/v2xvit_a2_p75.csv \\
      --out-json results/v2xvit_a2_p75.json \\
      2>&1 | tee logs/a2_eval_p75.log

  CUDA_VISIBLE_DEVICES=1 python scripts/phase2/eval_v2xvit_a2_pruned.py \\
      --ratio 0.5 \\
      --finetune-ckpt output/a2_finetune/v2xvit_bb_p50/net_epoch<N>_bestval.pth \\
      --epoch-used "bestval_at<N>_finetuned" \\
      --actual-filters "32,64,128" \\
      --out-csv results/v2xvit_a2_p50.csv \\
      --out-json results/v2xvit_a2_p50.json \\
      2>&1 | tee logs/a2_eval_p50.log
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch
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
    build_pruner, get_scatter_shape, CONFIG_YAML
)

N_SAMPLES   = 1789
IOU_THRESH  = 0.5
N_BOOTSTRAP = 1000
SEED        = 42


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
        best   = int(np.argmax(ious)); gt_idx = remaining_gt.pop(best)
        p, g   = pred_params[i], gt_params[gt_idx]
        ate    = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        l_p,w_p,h_p = abs(p[3]),abs(p[4]),abs(p[5])
        l_g,w_g,h_g = abs(g[3]),abs(g[4]),abs(g[5])
        i_vol  = min(l_p,l_g)*min(w_p,w_g)*min(h_p,h_g)
        u_vol  = l_p*w_p*h_p + l_g*w_g*h_g - i_vol
        size_iou = i_vol/u_vol if u_vol > 1e-9 else 0.0
        delta  = abs(p[6]-g[6]) % np.pi
        ate_l.append(ate); ase_l.append(1.0-size_iou)
        aoe_l.append(float(min(delta, np.pi-delta)))
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
        a,s,o = cat(frame_ate), cat(frame_ase), cat(frame_aoe)
        boot_ate.append(float(np.mean(a)) if len(a) else float("nan"))
        boot_ase.append(float(np.mean(s)) if len(s) else float("nan"))
        boot_aoe.append(float(np.mean(o)) if len(o) else float("nan"))
    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        return float(np.percentile(arr,2.5)), float(np.percentile(arr,97.5))
    all_ate = np.concatenate([a for a in frame_ate if len(a)>0])
    all_ase = np.concatenate([a for a in frame_ase if len(a)>0])
    all_aoe = np.concatenate([a for a in frame_aoe if len(a)>0])
    return {
        "mATE": float(np.mean(all_ate)) if len(all_ate) else float("nan"),
        "mATE_ci_lo": ci(boot_ate)[0], "mATE_ci_hi": ci(boot_ate)[1],
        "mASE": float(np.mean(all_ase)) if len(all_ase) else float("nan"),
        "mASE_ci_lo": ci(boot_ase)[0], "mASE_ci_hi": ci(boot_ase)[1],
        "mAOE": float(np.mean(all_aoe)) if len(all_aoe) else float("nan"),
        "mAOE_ci_lo": ci(boot_aoe)[0], "mAOE_ci_hi": ci(boot_aoe)[1],
        "n_tp": len(all_aoe),
    }


def rebuild_and_load(ratio: float, finetune_ckpt: Path, device: str):
    """Rebuild pruned architecture from epoch17, then load finetuned bestval ckpt."""
    print(f"[rebuild] ratio={ratio}, loading epoch17 for deterministic repruning...", flush=True)
    model = build_base_model(device)
    net   = V2XViTBackboneTraceNet(model).to(device).eval()
    hypes_raw = yaml_utils.load_yaml(str(CONFIG_YAML))
    ny, nx    = get_scatter_shape(hypes_raw)
    x = torch.randn(1, 64, ny, nx, device=device)
    pr = build_pruner(net, x, ratio, device)
    pr.step()
    print(f"[rebuild] Pruning done. Loading finetuned ckpt: {finetune_ckpt.name}", flush=True)
    ft_sd = torch.load(str(finetune_ckpt), map_location="cpu")
    if isinstance(ft_sd, dict) and "model_state_dict" in ft_sd:
        ft_sd = ft_sd["model_state_dict"]  # ISS-005 guard
        print("[rebuild] WARNING: wrapped format, unwrapped", flush=True)
    missing, unexpected = model.load_state_dict(ft_sd, strict=False)
    if missing: print(f"[rebuild] missing keys: {missing[:3]}", flush=True)
    print(f"[rebuild] ✓ backbone={sum(p.numel() for p in model.backbone_m1.parameters()):,}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio",          type=float, required=True)
    ap.add_argument("--finetune-ckpt",  type=str,   required=True)
    ap.add_argument("--epoch-used",     type=str,   required=True,
                    help='e.g. "bestval_at16_finetuned"')
    ap.add_argument("--actual-filters", type=str,   required=True,
                    help='comma-separated, e.g. "64,32,64"')
    ap.add_argument("--out-csv",  type=str, required=True)
    ap.add_argument("--out-json", type=str, required=True)
    ap.add_argument("--device",   type=str, default="cuda")
    args = ap.parse_args()

    ft_ckpt     = Path(args.finetune_ckpt).resolve()
    out_csv     = Path(args.out_csv).resolve()
    out_json    = Path(args.out_json).resolve()
    actual_filt = [int(x) for x in args.actual_filters.split(",")]
    device      = args.device

    print(f"[A-2 eval] ratio={args.ratio} epoch_used={args.epoch_used}", flush=True)
    print(f"[A-2 eval] ckpt: {ft_ckpt.name}", flush=True)
    print(f"[A-2 eval] actual_filters: {actual_filt}", flush=True)

    model = rebuild_and_load(args.ratio, ft_ckpt, "cpu")
    model = model.to(device).eval()

    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    from opencood.hypes_yaml.yaml_utils import load_general_params
    hypes = load_general_params(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader  = DataLoader(dataset, batch_size=1, num_workers=2,
                         collate_fn=dataset.collate_batch_test,
                         shuffle=False, drop_last=False)
    print(f"[A-2 eval] dataset={len(dataset)} samples", flush=True)

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
            infer = inference_utils.inference_intermediate_fusion(batch, model, dataset)
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
                print(f"  [{n_done}/{N_SAMPLES}] elapsed={elapsed:.0f}s ETA={elapsed/n_done*(N_SAMPLES-n_done):.0f}s", flush=True)

    elapsed = time.time()-t0
    print(f"[A-2 eval] Inference done in {elapsed:.1f}s", flush=True)

    tmp_dir = out_json.parent / "_ap_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(tmp_dir))
    import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[A-2 eval] Bootstrap CI (B={N_BOOTSTRAP})...", flush=True)
    ci = bootstrap_ci(frame_ate, frame_ase, frame_aoe)

    result = {
        "model_class":     "v2x_vit",
        "anchor":          f"pruned_{int(args.ratio*100)}",
        "precision":       "fp32_pytorch",
        "epoch_used":      args.epoch_used,
        "actual_filters":  actual_filt,   # ISS: use actual not nominal
        "prune_ratio":     args.ratio,
        "ckpt_path":       str(ft_ckpt),
        "n_samples":       n_done,
        "n_tp":            ci["n_tp"],
        "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
        "mATE": ci["mATE"], "mATE_ci_lo": ci["mATE_ci_lo"], "mATE_ci_hi": ci["mATE_ci_hi"],
        "mASE": ci["mASE"], "mASE_ci_lo": ci["mASE_ci_lo"], "mASE_ci_hi": ci["mASE_ci_hi"],
        "mAOE": ci["mAOE"], "mAOE_ci_lo": ci["mAOE_ci_lo"], "mAOE_ci_hi": ci["mAOE_ci_hi"],
        "elapsed_secs":    elapsed,
        "latency_kind":    "NOT_MEASURED",
        "note": (f"A-2 pruned eval. actual_filters={actual_filt} (not nominal). "
                 f"ISS-024: epoch_used={args.epoch_used}. "
                 f"Authorization: A-2 正式授权 — team-lead 2026-06-04."),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    with open(out_csv, "w") as f:
        f.write(",".join(result.keys()) + "\n")
        f.write(",".join(str(v) for v in result.values()) + "\n")

    print(f"\n[A-2 eval] ===== RESULTS =====", flush=True)
    print(f"  AP30={ap30:.4f}  AP50={ap50:.4f}  AP70={ap70:.4f}", flush=True)
    print(f"  mATE={ci['mATE']:.4f} [{ci['mATE_ci_lo']:.4f},{ci['mATE_ci_hi']:.4f}]", flush=True)
    print(f"  mASE={ci['mASE']:.4f} [{ci['mASE_ci_lo']:.4f},{ci['mASE_ci_hi']:.4f}]", flush=True)
    print(f"  mAOE={ci['mAOE']:.4f} [{ci['mAOE_ci_lo']:.4f},{ci['mAOE_ci_hi']:.4f}]", flush=True)
    print(f"  n_samples={n_done}  n_tp={ci['n_tp']}", flush=True)
    print(f"  Saved: {out_csv}", flush=True)
    print(f"         {out_json}", flush=True)


if __name__ == "__main__":
    main()
