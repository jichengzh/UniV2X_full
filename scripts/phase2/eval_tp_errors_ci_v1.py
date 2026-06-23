"""Phase M — mATE/mASE/mAOE bootstrap 95% CI.

策略:
  每个 config 只跑 N_FRAMES 帧(默认 300),保存 per-frame TP 误差列表。
  Bootstrap 在帧级别(不是 TP 级别):以帧为单位有放回重采样 B=1000 次,
  每次计算重采样帧里所有 TP 的均值 → 95% CI = [2.5%, 97.5%] 分位点。

  帧级 bootstrap 比 TP 级 bootstrap 更保守(保留帧内相关性),更严格。

  N_FRAMES=300 → ~300×15=4500 TPs/config, SE ≈ σ/√4500.
  对 mAOE σ≈0.07: SE≈0.001, CI宽≈0.004 rad。
  vs base→p25 mAOE Δ=0.0069 rad → SNR≈1.7 (base/p25 步骤)。
  vs base→p75 mAOE Δ=0.025 rad  → SNR≈6.3 (全程, 清晰信号)。

Output:
    results/tp_errors_ci_v1.csv        — mean ± CI per config
    results/tp_errors_ci_v1.json       — 完整 bootstrap 分布
    results/tp_frame_arrays_{tag}_{prec}.npz — per-frame 数组

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/phase2/eval_tp_errors_ci_v1.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import update_dict, torch_tensor_to_numpy, convert_format, compute_iou
from opencood.tools import inference_utils
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.box_utils import corner_to_center

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CACHE = REPO_ROOT / "models/stage_a_cache"
OUT_DIR = REPO_ROOT / "results"

ANCHORS = [
    ("base",     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",  (64, 128, 256)),
    ("pruned25", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10",       (48, 96, 192)),
    ("pruned50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",       (32, 64, 128)),
    ("pruned75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10",       (16, 32, 64)),
]
PRECISIONS = ["fp16", "int8"]

DAIR_RANGE = "102.4,51.2"
COLLAB_SPATIAL_SHAPE = (2, 64, 128, 256)
COLLAB_TEGO_SHAPE = (2, 2, 3)
N_FRAMES = 350       # 每 config 帧数(速度/精度折中); 全部1789改成None
IOU_THRESH = 0.5
N_BOOTSTRAP = 1000   # bootstrap 重采样次数


# ---------------------------------------------------------------------------
# TRT engine wrapper
# ---------------------------------------------------------------------------

class TrtCollabN2:
    def __init__(self, engine_path, spatial_shape=COLLAB_SPATIAL_SHAPE,
                 tego_shape=COLLAB_TEGO_SHAPE):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.output_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                             if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        self.spatial_name = next(n for n in self.input_names if "spatial" in n.lower())
        self.tego_name = next(n for n in self.input_names if "ego" in n.lower())
        self.spatial_shape = spatial_shape
        self.tego_shape = tego_shape

    def __call__(self, spatial, t_ego):
        ctx = self.engine.create_execution_context()
        ctx.set_input_shape(self.spatial_name, self.spatial_shape)
        ctx.set_input_shape(self.tego_name, self.tego_shape)
        bufs = {self.spatial_name: spatial.float().contiguous(),
                self.tego_name: t_ego.float().contiguous()}
        for n in self.output_names:
            shape = tuple(ctx.get_tensor_shape(n))
            bufs[n] = torch.empty(shape, dtype=torch.float32, device="cuda")
        for n in self.input_names + self.output_names:
            ctx.set_tensor_address(n, int(bufs[n].data_ptr()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return tuple(bufs[n] for n in self.output_names)


# ---------------------------------------------------------------------------
# Hybrid forward
# ---------------------------------------------------------------------------

def hybrid_forward(model, batch_data, trt_collab=None):
    ego = batch_data["ego"]
    agent_modality_list = ego["agent_modality_list"]
    affine_matrix = normalize_pairwise_tfm(
        ego["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size)
    modality_count = Counter(agent_modality_list)
    modality_feature_dict = {}
    for m in model.modality_name_list:
        if m not in modality_count:
            continue
        feat = getattr(model, f"encoder_{m}")(ego, m)
        feat = getattr(model, f"backbone_{m}")({"spatial_features": feat})["spatial_features_2d"]
        feat = getattr(model, f"aligner_{m}")(feat)
        modality_feature_dict[m] = feat
    counting = {m: 0 for m in model.modality_name_list}
    heter_list = []
    for m in agent_modality_list:
        heter_list.append(modality_feature_dict[m][counting[m]])
        counting[m] += 1
    heter_feat_2d = torch.stack(heter_list)
    n_agents = heter_feat_2d.shape[0]
    if n_agents == 2 and trt_collab is not None:
        t_ego = affine_matrix[0, 0, :2, :, :].contiguous()
        cls_p, reg_p, dir_p = trt_collab(heter_feat_2d.contiguous(), t_ego)
        return {"cls_preds": cls_p, "reg_preds": reg_p, "dir_preds": dir_p,
                "occ_single_list": [], "_path": "trt_collab"}
    else:
        record_len = ego["record_len"]
        fused, occ_outputs = model.pyramid_backbone.forward_collab(
            heter_feat_2d, record_len, affine_matrix, agent_modality_list, model.cam_crop_info)
        if model.shrink_flag:
            fused = model.shrink_conv(fused)
        return {"cls_preds": model.cls_head(fused), "reg_preds": model.reg_head(fused),
                "dir_preds": model.dir_head(fused), "occ_single_list": occ_outputs,
                "_path": "pytorch_fallback"}


# ---------------------------------------------------------------------------
# Per-frame TP geometric errors
# ---------------------------------------------------------------------------

def compute_tp_errors_frame(pred_box_np, pred_score_np, gt_box_np):
    """Return arrays of (ate, ase, aoe) for TP pairs in ONE frame."""
    if gt_box_np is None or len(gt_box_np) == 0:
        return np.array([]), np.array([]), np.array([])
    if pred_box_np is None or len(pred_box_np) == 0:
        return np.array([]), np.array([]), np.array([])

    score_order = np.argsort(-pred_score_np)
    pred_sorted = pred_box_np[score_order]
    pred_params = corner_to_center(pred_sorted, order='lwh')
    gt_params = corner_to_center(gt_box_np, order='lwh')
    pred_polygon_list = list(convert_format(pred_sorted))
    gt_polygon_list = list(convert_format(gt_box_np))
    remaining_gt_idx = list(range(len(gt_box_np)))

    ate_l, ase_l, aoe_l = [], [], []
    for i in range(len(pred_sorted)):
        if not remaining_gt_idx:
            break
        det_poly = pred_polygon_list[i]
        ious = compute_iou(det_poly, [gt_polygon_list[j] for j in remaining_gt_idx])
        if not len(ious) or np.max(ious) < IOU_THRESH:
            continue
        best_local_idx = int(np.argmax(ious))
        gt_idx = remaining_gt_idx.pop(best_local_idx)
        p, g = pred_params[i], gt_params[gt_idx]
        ate = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        l_p, w_p, h_p = abs(p[3]), abs(p[4]), abs(p[5])
        l_g, w_g, h_g = abs(g[3]), abs(g[4]), abs(g[5])
        i_vol = min(l_p,l_g)*min(w_p,w_g)*min(h_p,h_g)
        vol_p, vol_g = l_p*w_p*h_p, l_g*w_g*h_g
        u_vol = vol_p + vol_g - i_vol
        size_iou = i_vol/u_vol if u_vol > 1e-9 else 0.0
        delta = abs(p[6]-g[6]) % np.pi
        yaw_err = float(min(delta, np.pi-delta))
        ate_l.append(ate); ase_l.append(1.0-size_iou); aoe_l.append(yaw_err)

    return np.array(ate_l), np.array(ase_l), np.array(aoe_l)


# ---------------------------------------------------------------------------
# Bootstrap CI (frame-level resampling)
# ---------------------------------------------------------------------------

def bootstrap_ci(frame_ate_list, frame_ase_list, frame_aoe_list,
                  B=N_BOOTSTRAP, seed=42):
    """
    frame_*_list: list of np.ndarray, one per frame.
    Bootstrap 以帧为单位重采样 B 次,每次算全部 TP 均值。
    返回 (mean, ci_lo, ci_hi) for ate/ase/aoe。
    """
    rng = np.random.default_rng(seed)
    n_frames = len(frame_ate_list)
    boot_ate, boot_ase, boot_aoe = [], [], []

    for _ in range(B):
        idx = rng.integers(0, n_frames, size=n_frames)
        ate_flat = np.concatenate([frame_ate_list[i] for i in idx]) if any(len(frame_ate_list[i]) > 0 for i in idx) else np.array([])
        ase_flat = np.concatenate([frame_ase_list[i] for i in idx]) if any(len(frame_ase_list[i]) > 0 for i in idx) else np.array([])
        aoe_flat = np.concatenate([frame_aoe_list[i] for i in idx]) if any(len(frame_aoe_list[i]) > 0 for i in idx) else np.array([])
        boot_ate.append(np.mean(ate_flat) if len(ate_flat) > 0 else float("nan"))
        boot_ase.append(np.mean(ase_flat) if len(ase_flat) > 0 else float("nan"))
        boot_aoe.append(np.mean(aoe_flat) if len(aoe_flat) > 0 else float("nan"))

    boot_ate = np.array(boot_ate)
    boot_ase = np.array(boot_ase)
    boot_aoe = np.array(boot_aoe)

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    all_ate = np.concatenate([a for a in frame_ate_list if len(a) > 0]) if any(len(a) > 0 for a in frame_ate_list) else np.array([])
    all_ase = np.concatenate([a for a in frame_ase_list if len(a) > 0]) if any(len(a) > 0 for a in frame_ase_list) else np.array([])
    all_aoe = np.concatenate([a for a in frame_aoe_list if len(a) > 0]) if any(len(a) > 0 for a in frame_aoe_list) else np.array([])

    return {
        "mATE": float(np.mean(all_ate)) if len(all_ate) > 0 else float("nan"),
        "mATE_ci_lo": ci(boot_ate)[0], "mATE_ci_hi": ci(boot_ate)[1],
        "mATE_std": float(np.std(all_ate)) if len(all_ate) > 0 else float("nan"),
        "n_tp": len(all_ate),
        "mASE": float(np.mean(all_ase)) if len(all_ase) > 0 else float("nan"),
        "mASE_ci_lo": ci(boot_ase)[0], "mASE_ci_hi": ci(boot_ase)[1],
        "mASE_std": float(np.std(all_ase)) if len(all_ase) > 0 else float("nan"),
        "mAOE": float(np.mean(all_aoe)) if len(all_aoe) > 0 else float("nan"),
        "mAOE_ci_lo": ci(boot_aoe)[0], "mAOE_ci_hi": ci(boot_aoe)[1],
        "mAOE_std": float(np.std(all_aoe)) if len(all_aoe) > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-anchor evaluation
# ---------------------------------------------------------------------------

def run_eval_ci(tag, ckpt_dir, precision, n_frames=N_FRAMES):
    engine_path = CACHE / f"{tag}_{precision}.engine"
    if not engine_path.exists():
        print(f"  [SKIP] engine not found: {engine_path}")
        return None

    hypes = yaml_utils.load_yaml(str(Path(ckpt_dir) / "config.yaml"))
    if "heter" in hypes:
        x_max, y_max = float(DAIR_RANGE.split(",")[0]), float(DAIR_RANGE.split(",")[1])
        new_range = [-x_max, -y_max,
                     hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
                     x_max, y_max,
                     hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5]]
        hypes = update_dict(hypes, {"cav_lidar_range": new_range, "lidar_range": new_range,
                                     "gt_range": new_range})
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    print(f"\n{'='*60}")
    print(f"  anchor={tag}  precision={precision}  n_frames={n_frames}")
    print(f"{'='*60}")

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(ckpt_dir, model)
    model.cuda().eval()

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    trt_collab = TrtCollabN2(str(engine_path))

    frame_ate_list, frame_ase_list, frame_aoe_list = [], [], []
    n_done = 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_frames is not None and n_done >= n_frames:
                break
            batch_data = train_utils.to_device(batch_data, "cuda")
            out = hybrid_forward(model, batch_data, trt_collab)
            out_wrapped = {"ego": out}
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, out_wrapped)

            if pred_box is not None and gt_box is not None:
                pred_np = torch_tensor_to_numpy(pred_box)
                score_np = torch_tensor_to_numpy(pred_score)
                gt_np = torch_tensor_to_numpy(gt_box)
                ate_a, ase_a, aoe_a = compute_tp_errors_frame(pred_np, score_np, gt_np)
            else:
                ate_a, ase_a, aoe_a = np.array([]), np.array([]), np.array([])

            frame_ate_list.append(ate_a)
            frame_ase_list.append(ase_a)
            frame_aoe_list.append(aoe_a)
            n_done += 1

    elapsed = time.time() - t0
    n_tp_total = sum(len(a) for a in frame_aoe_list)
    print(f"  Collected {n_done} frames, {n_tp_total} TP pairs in {elapsed:.0f}s")
    print(f"  Running bootstrap (B={N_BOOTSTRAP})...")

    ci_stats = bootstrap_ci(frame_ate_list, frame_ase_list, frame_aoe_list, B=N_BOOTSTRAP)

    result = {
        "anchor": tag,
        "precision": precision,
        "n_frames": n_done,
        "elapsed_secs": elapsed,
        **ci_stats,
    }

    print(f"  mATE={ci_stats['mATE']:.4f}m  [{ci_stats['mATE_ci_lo']:.4f}, {ci_stats['mATE_ci_hi']:.4f}]")
    print(f"  mASE={ci_stats['mASE']:.4f}   [{ci_stats['mASE_ci_lo']:.4f}, {ci_stats['mASE_ci_hi']:.4f}]")
    print(f"  mAOE={ci_stats['mAOE']:.4f}rad [{ci_stats['mAOE_ci_lo']:.4f}, {ci_stats['mAOE_ci_hi']:.4f}]")

    # Save per-frame arrays for later analysis
    # Use object dtype to handle ragged (different-length) arrays per frame
    arr_path = OUT_DIR / f"tp_frame_arrays_{tag}_{precision}.npz"
    ate_obj = np.empty(len(frame_ate_list), dtype=object)
    ase_obj = np.empty(len(frame_ase_list), dtype=object)
    aoe_obj = np.empty(len(frame_aoe_list), dtype=object)
    for i, a in enumerate(frame_ate_list): ate_obj[i] = a
    for i, a in enumerate(frame_ase_list): ase_obj[i] = a
    for i, a in enumerate(frame_aoe_list): aoe_obj[i] = a
    np.savez_compressed(str(arr_path), ate=ate_obj, ase=ase_obj, aoe=aoe_obj)
    print(f"  Saved frame arrays → {arr_path.name}")

    del model, trt_collab
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Signal / noise analysis
# ---------------------------------------------------------------------------

def analyze_snr(results):
    """Compute SNR for pruning axis (FP16) and INT8 step."""
    from collections import defaultdict

    by_anchor = defaultdict(dict)
    for r in results:
        by_anchor[r["anchor"]][r["precision"]] = r

    anchor_order = ["base", "pruned25", "pruned50", "pruned75"]

    print("\n" + "="*70)
    print("SIGNAL / NOISE ANALYSIS")
    print("="*70)

    # 1. Pruning axis (FP16) SNR
    print("\n[FP16 pruning axis — mAOE]")
    fp16_vals = []
    fp16_ci_width = []
    for anch in anchor_order:
        if anch not in by_anchor or "fp16" not in by_anchor[anch]:
            continue
        r = by_anchor[anch]["fp16"]
        ci_w = r["mAOE_ci_hi"] - r["mAOE_ci_lo"]
        fp16_vals.append((anch, r["mAOE"], r["mAOE_ci_lo"], r["mAOE_ci_hi"], ci_w))
        print(f"  {anch:<12} mAOE={r['mAOE']:.4f}  95%CI=[{r['mAOE_ci_lo']:.4f}, {r['mAOE_ci_hi']:.4f}]  CI_width={ci_w:.4f}")

    if len(fp16_vals) >= 2:
        total_delta = fp16_vals[-1][1] - fp16_vals[0][1]
        avg_ci_w = sum(x[4] for x in fp16_vals) / len(fp16_vals)
        snr = abs(total_delta) / (avg_ci_w / 2)
        print(f"  Total Δ(base→p75): {total_delta:+.4f}  avg CI half-width: {avg_ci_w/2:.4f}  SNR: {snr:.1f}×")
        print(f"  → {'SIGNAL CLEAR (>5×noise)' if snr > 5 else 'SIGNAL PRESENT (>2×noise)' if snr > 2 else 'SIGNAL WEAK (<2×noise)'}")

    # AP70 pruning axis for comparison
    print("\n[FP16 pruning axis — AP70 (from full run tp_errors_v1.csv)]")
    ap70_full = {
        "base": 0.6311, "pruned25": 0.5858, "pruned50": 0.5686, "pruned75": 0.5078
    }
    for anch in anchor_order:
        if anch not in by_anchor or "fp16" not in by_anchor[anch]:
            continue
        r = by_anchor[anch]["fp16"]
        ci_w_aoe = r["mAOE_ci_hi"] - r["mAOE_ci_lo"]
        ap70 = ap70_full.get(anch, None)
        print(f"  {anch:<12} AP70={ap70}  mAOE={r['mAOE']:.4f}  CI_width(mAOE)={ci_w_aoe:.4f}")

    # 2. INT8 step (base only, smallest survivor bias)
    print("\n[INT8 effect at base — all metrics]")
    if "base" in by_anchor and "fp16" in by_anchor["base"] and "int8" in by_anchor["base"]:
        fp16 = by_anchor["base"]["fp16"]
        int8 = by_anchor["base"]["int8"]
        for col in ["mATE", "mASE", "mAOE"]:
            delta = int8[col] - fp16[col]
            noise = (fp16[f"{col}_ci_hi"] - fp16[f"{col}_ci_lo"]) / 2
            snr = abs(delta) / noise if noise > 0 else float("inf")
            print(f"  {col}: fp16={fp16[col]:.4f}  int8={int8[col]:.4f}  Δ={delta:+.5f}  half-CI={noise:.5f}  SNR={snr:.2f}×")
            print(f"    → {'SIGNAL (>2×noise)' if snr > 2 else 'NOISE (<2×noise)'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import csv
    results = []

    for tag, ckpt_dir, _ in ANCHORS:
        for precision in PRECISIONS:
            r = run_eval_ci(tag, ckpt_dir, precision)
            if r is not None:
                results.append(r)

    if not results:
        print("No results.")
        return

    # Save CSV
    csv_path = OUT_DIR / "tp_errors_ci_v1.csv"
    fieldnames = ["anchor", "precision", "n_frames", "n_tp",
                  "mATE", "mATE_ci_lo", "mATE_ci_hi", "mATE_std",
                  "mASE", "mASE_ci_lo", "mASE_ci_hi", "mASE_std",
                  "mAOE", "mAOE_ci_lo", "mAOE_ci_hi", "mAOE_std",
                  "elapsed_secs"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\n✓ Saved: {csv_path}")

    json_path = OUT_DIR / "tp_errors_ci_v1.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'anchor':<12} {'prec':<6} {'mAOE':<8} {'CI_lo':<8} {'CI_hi':<8} {'CI_width':<10} {'n_tp'}")
    print("-"*80)
    for r in results:
        ci_w = r["mAOE_ci_hi"] - r["mAOE_ci_lo"]
        print(f"{r['anchor']:<12} {r['precision']:<6} {r['mAOE']:.4f}   {r['mAOE_ci_lo']:.4f}   {r['mAOE_ci_hi']:.4f}   {ci_w:.4f}      {r['n_tp']}")

    # SNR analysis
    analyze_snr(results)

    # Save analysis summary
    analysis = {
        "n_configs": len(results),
        "n_frames_per_config": N_FRAMES,
        "n_bootstrap": N_BOOTSTRAP,
        "results": results,
    }
    json_path2 = OUT_DIR / "tp_errors_ci_v1_analysis.json"
    with open(json_path2, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✓ Analysis saved: {json_path2}")


if __name__ == "__main__":
    main()
