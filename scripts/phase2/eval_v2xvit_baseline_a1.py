"""【A-1 正式授权 — team-lead 2026-06-04】A-1 批准启动
授权原文逐字引用 (回执协议 ISS-030 落到文件层):
  "双闸已开(Phase M 全收口 + 用户显式授权), supervisor 对
   scripts/phase2/eval_v2xvit_baseline_a1.py(13:13版)全文核验 6/6 PASS。
   授权范围 = 仅 A-1: V2X-ViT DAIR val 1789 baseline AP+mAOE eval,
   纯 PyTorch 推理, ckpt 锁 net_epoch_bestval_at17.pth, 指定 GPU1"
ISS-024: epoch_used = bestval_at17, A-3 TRT build 必须显式锁定同 epoch。

A-1 — V2X-ViT DAIR baseline AP + mAOE eval (纯 PyTorch, 无 TRT).

目的: 获取 V2X-ViT DAIR val 1789 baseline AP30/50/70 + mAOE (with bootstrap CI).
这是 Task#5 A-1 步骤: 不动 TRT, 只跑 PyTorch FP32 推理.

口径标注:
  - precision = fp32_pytorch (非 TRT body_subnet)
  - latency_kind = NOT_MEASURED (本脚本不测延迟)
  - 与 Pyramid stage_a 口径不同, 绝不混表

纪律 (ISS-024):
  - 使用 net_epoch_bestval_at17.pth (ckpt 无续训污染)
  - epoch_used = "bestval_at17", 须与未来 A-3 TRT build 锁定同 epoch

输出:
  results/v2xvit_baseline_a1.csv
  results/v2xvit_baseline_a1.json

用法:
  CUDA_VISIBLE_DEVICES=7 python scripts/phase2/eval_v2xvit_baseline_a1.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)   # resolve relative data paths in yaml

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.tools import inference_utils
from opencood.utils.common_utils import torch_tensor_to_numpy, convert_format, compute_iou
from opencood.utils.box_utils import corner_to_center

# ---------------------------------------------------------------------------
# Config (ISS-024: epoch_used 与 A-3 TRT build 必须一致)
# ---------------------------------------------------------------------------

CKPT_DIR = Path("/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
                "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26")
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE   = "net_epoch_bestval_at17.pth"   # epoch17 bestval; lock此 epoch 供 A-3
EPOCH_USED  = "bestval_at17"                  # 人读标签

N_SAMPLES   = 1789   # DAIR val 全集
IOU_THRESH  = 0.5    # mAOE TP matching threshold (同 Pyramid)
N_BOOTSTRAP = 1000
SEED        = 42

OUT_DIR = REPO_ROOT / "results"
OUT_CSV  = OUT_DIR / "v2xvit_baseline_a1.csv"
OUT_JSON = OUT_DIR / "v2xvit_baseline_a1.json"

# ---------------------------------------------------------------------------
# mAOE 计算 (复用自 eval_tp_errors_corrected_full.py)
# ---------------------------------------------------------------------------

def compute_tp_errors_frame(pred_box_np, pred_score_np, gt_box_np):
    """Per-frame TP error arrays. ISS-020 angle normalization: min(delta%pi, pi-delta%pi)."""
    if gt_box_np is None or len(gt_box_np) == 0:
        return np.array([]), np.array([]), np.array([])
    if pred_box_np is None or len(pred_box_np) == 0:
        return np.array([]), np.array([]), np.array([])
    score_order  = np.argsort(-pred_score_np)
    pred_sorted  = pred_box_np[score_order]
    pred_params  = corner_to_center(pred_sorted, order='lwh')
    gt_params    = corner_to_center(gt_box_np,   order='lwh')
    pred_poly    = list(convert_format(pred_sorted))
    gt_poly      = list(convert_format(gt_box_np))
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
        # ATE
        ate = float(np.sqrt((p[0] - g[0])**2 + (p[1] - g[1])**2))
        # ASE (1 - 3D IoU)
        l_p, w_p, h_p = abs(p[3]), abs(p[4]), abs(p[5])
        l_g, w_g, h_g = abs(g[3]), abs(g[4]), abs(g[5])
        i_vol = min(l_p, l_g) * min(w_p, w_g) * min(h_p, h_g)
        u_vol = l_p*w_p*h_p + l_g*w_g*h_g - i_vol
        size_iou = i_vol / u_vol if u_vol > 1e-9 else 0.0
        # AOE — angle normalization (ISS-020 §3: nuScenes 180° symmetry)
        delta = abs(p[6] - g[6]) % np.pi
        aoe   = float(min(delta, np.pi - delta))
        ate_l.append(ate); ase_l.append(1.0 - size_iou); aoe_l.append(aoe)
    return np.array(ate_l), np.array(ase_l), np.array(aoe_l)


def bootstrap_ci(frame_ate_list, frame_ase_list, frame_aoe_list,
                 B: int = N_BOOTSTRAP, seed: int = SEED):
    """Frame-level bootstrap 95% CI (同 Pyramid eval 口径)."""
    rng = np.random.default_rng(seed)
    n_frames = len(frame_aoe_list)
    boot_ate, boot_ase, boot_aoe = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n_frames, size=n_frames)
        def cat(lst):
            parts = [lst[i] for i in idx if len(lst[i]) > 0]
            return np.concatenate(parts) if parts else np.array([])
        a, s, o = cat(frame_ate_list), cat(frame_ase_list), cat(frame_aoe_list)
        boot_ate.append(float(np.mean(a)) if len(a) > 0 else float("nan"))
        boot_ase.append(float(np.mean(s)) if len(s) > 0 else float("nan"))
        boot_aoe.append(float(np.mean(o)) if len(o) > 0 else float("nan"))

    def ci_from(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    all_ate = np.concatenate([a for a in frame_ate_list if len(a) > 0])
    all_ase = np.concatenate([a for a in frame_ase_list if len(a) > 0])
    all_aoe = np.concatenate([a for a in frame_aoe_list if len(a) > 0])
    return {
        "mATE":     float(np.mean(all_ate)) if len(all_ate) > 0 else float("nan"),
        "mATE_ci_lo": ci_from(boot_ate)[0], "mATE_ci_hi": ci_from(boot_ate)[1],
        "mASE":     float(np.mean(all_ase)) if len(all_ase) > 0 else float("nan"),
        "mASE_ci_lo": ci_from(boot_ase)[0], "mASE_ci_hi": ci_from(boot_ase)[1],
        "mAOE":     float(np.mean(all_aoe)) if len(all_aoe) > 0 else float("nan"),
        "mAOE_ci_lo": ci_from(boot_aoe)[0], "mAOE_ci_hi": ci_from(boot_aoe)[1],
        "n_tp": len(all_aoe),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"[A-1] V2X-ViT DAIR baseline eval", flush=True)
    print(f"  ckpt : {CKPT_DIR.name}", flush=True)
    print(f"  epoch: {CKPT_FILE}  →  epoch_used={EPOCH_USED}", flush=True)
    print(f"  out  : {OUT_CSV}", flush=True)

    # ---- Load model ----
    hypes = yaml_utils.load_yaml(str(CONFIG_YAML))
    # Apply yaml parser (load_general_params for V2X-ViT)
    parser_func = getattr(yaml_utils, hypes["yaml_parser"])
    hypes = parser_func(hypes)
    # Ensure validate_dir = test_dir (DAIR val)
    hypes["validate_dir"] = hypes["test_dir"]

    print(f"[A-1] Building model: {hypes['model']['core_method']}", flush=True)
    model = train_utils.create_model(hypes)

    # Load epoch17 bestval weights (ISS-024: explicit epoch lock)
    epoch_path = CKPT_DIR / CKPT_FILE
    print(f"[A-1] Loading {CKPT_FILE}", flush=True)
    state = torch.load(str(epoch_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        # ISS-005 guard: unwrap if accidentally wrapped
        state = state["model_state_dict"]
        print("  [WARNING] ckpt was wrapped — unwrapped model_state_dict", flush=True)
    model.load_state_dict(state)
    model.cuda().eval()
    print("[A-1] Model loaded OK", flush=True)

    # ---- Build dataset ----
    print("[A-1] Building DAIR val dataset...", flush=True)
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader  = DataLoader(
        dataset, batch_size=1, num_workers=2,
        collate_fn=dataset.collate_batch_test,
        shuffle=False, pin_memory=False, drop_last=False,
    )
    print(f"[A-1] Dataset built. Val set size = {len(dataset)}", flush=True)

    # ---- AP accumulators ----
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    frame_ate_list, frame_ase_list, frame_aoe_list = [], [], []
    n_done = 0
    t0 = time.time()

    device = torch.device("cuda")
    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_done >= N_SAMPLES:
                break

            batch_data = train_utils.to_device(batch_data, device)

            # Pure PyTorch forward via HEAL inference_utils
            infer_result = inference_utils.inference_intermediate_fusion(
                batch_data, model, dataset)

            pred_box   = infer_result["pred_box_tensor"]
            pred_score = infer_result["pred_score"]
            gt_box     = infer_result["gt_box_tensor"]

            # AP
            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box,
                                           result_stat, iou_th)

            # mAOE
            if pred_box is not None and gt_box is not None:
                pred_np  = torch_tensor_to_numpy(pred_box)
                score_np = torch_tensor_to_numpy(pred_score)
                gt_np    = torch_tensor_to_numpy(gt_box)
                ate_a, ase_a, aoe_a = compute_tp_errors_frame(pred_np, score_np, gt_np)
            else:
                ate_a, ase_a, aoe_a = np.array([]), np.array([]), np.array([])

            frame_ate_list.append(ate_a)
            frame_ase_list.append(ase_a)
            frame_aoe_list.append(aoe_a)
            n_done += 1

            if n_done % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{n_done}/{N_SAMPLES}]  elapsed={elapsed:.0f}s  "
                      f"ETA={elapsed/n_done*(N_SAMPLES-n_done):.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[A-1] Inference done in {elapsed:.1f}s", flush=True)

    # ---- AP ----
    tmp_ap_dir = OUT_DIR / "v2xvit_a1_ap_tmp"
    tmp_ap_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(tmp_ap_dir))

    # ---- Bootstrap CI ----
    print(f"[A-1] Bootstrap CI (B={N_BOOTSTRAP})...", flush=True)
    ci = bootstrap_ci(frame_ate_list, frame_ase_list, frame_aoe_list)

    # ---- Assemble result ----
    result = {
        # Identity
        "model_class":   "v2x_vit",
        "anchor":        "base",
        "precision":     "fp32_pytorch",
        "epoch_used":    EPOCH_USED,
        "ckpt_path":     str(CKPT_DIR / CKPT_FILE),
        # Counts
        "n_samples": n_done,
        "n_tp":      ci["n_tp"],
        # AP
        "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
        # TP errors
        "mATE":        ci["mATE"],
        "mATE_ci_lo":  ci["mATE_ci_lo"],  "mATE_ci_hi": ci["mATE_ci_hi"],
        "mASE":        ci["mASE"],
        "mASE_ci_lo":  ci["mASE_ci_lo"],  "mASE_ci_hi": ci["mASE_ci_hi"],
        "mAOE":        ci["mAOE"],
        "mAOE_ci_lo":  ci["mAOE_ci_lo"],  "mAOE_ci_hi": ci["mAOE_ci_hi"],
        # Meta
        "elapsed_secs": elapsed,
        "latency_kind": "NOT_MEASURED_see_v2xvit_dair_real.json",
        "note":         "A-1 PyTorch FP32 baseline. Latency NOT tested here (see existing timing data). "
                        "epoch17=bestval lock for future A-3 TRT build (ISS-024).",
    }

    # ---- Save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    with open(OUT_CSV, "w") as f:
        f.write(",".join(result.keys()) + "\n")
        f.write(",".join(str(v) for v in result.values()) + "\n")

    print(f"\n[A-1] ===== RESULTS =====", flush=True)
    print(f"  AP30={ap30:.4f}  AP50={ap50:.4f}  AP70={ap70:.4f}", flush=True)
    print(f"  mATE={ci['mATE']:.4f} [{ci['mATE_ci_lo']:.4f},{ci['mATE_ci_hi']:.4f}]", flush=True)
    print(f"  mASE={ci['mASE']:.4f} [{ci['mASE_ci_lo']:.4f},{ci['mASE_ci_hi']:.4f}]", flush=True)
    print(f"  mAOE={ci['mAOE']:.4f} [{ci['mAOE_ci_lo']:.4f},{ci['mAOE_ci_hi']:.4f}]", flush=True)
    print(f"  n_samples={n_done}  n_tp={ci['n_tp']}", flush=True)
    print(f"  elapsed={elapsed:.1f}s", flush=True)
    print(f"\n  Saved: {OUT_CSV}", flush=True)
    print(f"         {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
