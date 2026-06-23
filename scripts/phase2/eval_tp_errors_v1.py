"""Phase M — mATE/mASE/mAOE evaluation on DAIR val.

Hypothesis: AP70 is flat across pruning rates (span 0.107), but nuScenes-style
TP geometric errors (translation / scale / orientation) may show monotonic
degradation where AP is blind.

Uses the existing stage_a TRT engines (already built in models/stage_a_cache/).
Runs hybrid TRT+PyTorch inference (same as m4_8_hybrid_infer_ap.py) and adds
per-frame TP matching + geometric error accumulation.

Metrics definition (nuScenes NDS TP convention, at IoU_2D=0.5):
  mATE: mean L2 center distance (m) for TP-matched pairs
  mASE: mean (1 - 3D_size_IoU) for TP-matched pairs
         where 3D_size_IoU = intersection_vol / union_vol assuming boxes
         centered at same point with same orientation (size-only comparison)
  mAOE: mean min angular yaw error (rad) for TP-matched pairs,
         using 180-degree symmetry: err = min(|Δyaw|%π, π - |Δyaw|%π)

Output:
    results/tp_errors_v1.csv      — one row per (anchor, precision)
    results/tp_errors_v1.json     — same + per-frame stats

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/phase2/eval_tp_errors_v1.py
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

# (tag, ckpt_dir, (s0,s1,s2))
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
N_SAMPLES = 1789
IOU_THRESH = 0.5  # TP matching threshold (nuScenes uses 0.5)


# ---------------------------------------------------------------------------
# TRT engine wrapper (verbatim from m4_8_hybrid_infer_ap.py)
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
# Hybrid forward (verbatim from m4_8_hybrid_infer_ap.py)
# ---------------------------------------------------------------------------

def hybrid_forward(model, batch_data, trt_collab=None):
    ego = batch_data["ego"]
    record_len = ego["record_len"]
    agent_modality_list = ego["agent_modality_list"]
    affine_matrix = normalize_pairwise_tfm(
        ego["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size,
    )
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
        idx = counting[m]
        heter_list.append(modality_feature_dict[m][idx])
        counting[m] += 1
    heter_feat_2d = torch.stack(heter_list)

    n_agents = heter_feat_2d.shape[0]
    if n_agents == 2 and trt_collab is not None:
        t_ego = affine_matrix[0, 0, :2, :, :].contiguous()
        cls_p, reg_p, dir_p = trt_collab(heter_feat_2d.contiguous(), t_ego)
        return {"cls_preds": cls_p, "reg_preds": reg_p, "dir_preds": dir_p,
                "occ_single_list": [], "_path": "trt_collab"}
    else:
        fused, occ_outputs = model.pyramid_backbone.forward_collab(
            heter_feat_2d, record_len, affine_matrix, agent_modality_list, model.cam_crop_info,
        )
        if model.shrink_flag:
            fused = model.shrink_conv(fused)
        return {
            "cls_preds": model.cls_head(fused),
            "reg_preds": model.reg_head(fused),
            "dir_preds": model.dir_head(fused),
            "occ_single_list": occ_outputs,
            "_path": "pytorch_fallback",
        }


# ---------------------------------------------------------------------------
# TP geometric error accumulator
# ---------------------------------------------------------------------------

def compute_tp_errors(pred_box_np, pred_score_np, gt_box_np):
    """Compute per-TP geometric errors for one frame.

    Parameters
    ----------
    pred_box_np : np.ndarray (N_pred, 8, 3) corners
    pred_score_np : np.ndarray (N_pred,)
    gt_box_np : np.ndarray (N_gt, 8, 3) corners

    Returns
    -------
    ate_list : list[float]   L2 center distance per TP (m)
    ase_list : list[float]   1 - size_iou_3d per TP
    aoe_list : list[float]   min yaw error per TP (rad)
    n_gt : int               ground truth count
    """
    if gt_box_np is None or len(gt_box_np) == 0:
        return [], [], [], 0
    if pred_box_np is None or len(pred_box_np) == 0:
        return [], [], [], len(gt_box_np)

    n_gt = len(gt_box_np)

    # Sort by score descending
    score_order = np.argsort(-pred_score_np)
    pred_sorted = pred_box_np[score_order]

    # Convert corners (N,8,3) → center params (N,7): [x,y,z,l,w,h,yaw]
    pred_params = corner_to_center(pred_sorted, order='lwh')   # (N,7)
    gt_params = corner_to_center(gt_box_np, order='lwh')       # (M,7)

    # 2D IoU matching using HEAL's polygon IoU (same as caluclate_tp_fp)
    pred_polygon_list = list(convert_format(pred_sorted))
    gt_polygon_list = list(convert_format(gt_box_np))
    remaining_gt_idx = list(range(n_gt))

    ate_list = []
    ase_list = []
    aoe_list = []

    for i in range(len(pred_sorted)):
        if len(remaining_gt_idx) == 0:
            break
        det_poly = pred_polygon_list[i]
        ious = compute_iou(det_poly, [gt_polygon_list[j] for j in remaining_gt_idx])

        if len(ious) == 0 or np.max(ious) < IOU_THRESH:
            continue  # FP

        # TP: matched to best-overlap GT
        best_local_idx = int(np.argmax(ious))
        gt_idx = remaining_gt_idx[best_local_idx]
        remaining_gt_idx.pop(best_local_idx)

        p = pred_params[i]   # [x,y,z,l,w,h,yaw]
        g = gt_params[gt_idx]

        # mATE: L2 center (x,y) distance in meters
        ate = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        ate_list.append(ate)

        # mASE: 1 - 3D_size_IoU (size-only, no rotation, both centered at origin)
        # Intersection = prod of min dims; Union = vol_p + vol_g - Intersection
        l_p, w_p, h_p = abs(p[3]), abs(p[4]), abs(p[5])
        l_g, w_g, h_g = abs(g[3]), abs(g[4]), abs(g[5])
        i_vol = min(l_p, l_g) * min(w_p, w_g) * min(h_p, h_g)
        vol_p = l_p * w_p * h_p
        vol_g = l_g * w_g * h_g
        u_vol = vol_p + vol_g - i_vol
        size_iou = i_vol / u_vol if u_vol > 1e-9 else 0.0
        ase = float(1.0 - size_iou)
        ase_list.append(ase)

        # mAOE: min yaw error accounting for 180° symmetry
        delta = abs(p[6] - g[6])
        delta = delta % np.pi
        yaw_err = float(min(delta, np.pi - delta))
        aoe_list.append(yaw_err)

    return ate_list, ase_list, aoe_list, n_gt


# ---------------------------------------------------------------------------
# Per-anchor evaluation runner
# ---------------------------------------------------------------------------

def run_eval_one_anchor(tag, ckpt_dir, precision, n_samples=N_SAMPLES):
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
    print(f"  anchor={tag}  precision={precision}  engine={engine_path.name}")
    print(f"{'='*60}")

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(ckpt_dir, model)
    model.cuda().eval()

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    trt_collab = TrtCollabN2(str(engine_path))

    # AP accumulators (to verify we reproduce stage_a results)
    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}

    # TP error accumulators
    all_ate, all_ase, all_aoe = [], [], []
    n_gt_total = 0
    n_trt_collab, n_fallback = 0, 0
    t0 = time.time()

    with torch.inference_mode():
        n_done = 0
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_done >= n_samples:
                break

            batch_data = train_utils.to_device(batch_data, "cuda")
            output_dict = hybrid_forward(model, batch_data, trt_collab)

            path = output_dict["_path"]
            if path == "trt_collab":
                n_trt_collab += 1
            else:
                n_fallback += 1

            output_wrapped = {"ego": output_dict}
            pred_box_tensor, pred_score, gt_box_tensor = \
                dataset.post_process(batch_data, output_wrapped)

            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor,
                                           result_stat, iou_th)

            # TP geometric errors
            if pred_box_tensor is not None and gt_box_tensor is not None:
                pred_np = torch_tensor_to_numpy(pred_box_tensor)
                score_np = torch_tensor_to_numpy(pred_score)
                gt_np = torch_tensor_to_numpy(gt_box_tensor)
                ate_l, ase_l, aoe_l, n_gt = compute_tp_errors(pred_np, score_np, gt_np)
            elif gt_box_tensor is not None:
                ate_l, ase_l, aoe_l, n_gt = [], [], [], int(gt_box_tensor.shape[0])
            else:
                ate_l, ase_l, aoe_l, n_gt = [], [], [], 0

            all_ate.extend(ate_l)
            all_ase.extend(ase_l)
            all_aoe.extend(aoe_l)
            n_gt_total += n_gt

            n_done += 1
            if n_done % 200 == 0:
                print(f"  {n_done}/{n_samples}  trt_collab={n_trt_collab} fb={n_fallback} "
                      f"n_tp_so_far={len(all_ate)} elapsed={time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    eval_out_dir = REPO_ROOT / f"results/m_eval_{tag}_{precision}_tperr"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(eval_out_dir))

    n_tp = len(all_ate)
    mATE = float(np.mean(all_ate)) if all_ate else float("nan")
    mASE = float(np.mean(all_ase)) if all_ase else float("nan")
    mAOE = float(np.mean(all_aoe)) if all_aoe else float("nan")

    result = {
        "anchor": tag,
        "precision": precision,
        "n_samples": n_done,
        "n_trt_collab": n_trt_collab,
        "n_fallback": n_fallback,
        "n_gt_total": n_gt_total,
        "n_tp": n_tp,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "mATE": mATE,
        "mASE": mASE,
        "mAOE": mAOE,
        "elapsed_secs": elapsed,
    }
    print(f"\n  RESULT {tag} {precision}:")
    print(f"  AP30={ap30:.4f}  AP50={ap50:.4f}  AP70={ap70:.4f}")
    print(f"  mATE={mATE:.4f}m  mASE={mASE:.4f}  mAOE={mAOE:.4f}rad")
    print(f"  n_tp={n_tp}  n_gt={n_gt_total}  elapsed={elapsed:.0f}s")

    # Clean up GPU memory
    del model
    del trt_collab
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import csv
    results = []

    for tag, ckpt_dir, _ in ANCHORS:
        for precision in PRECISIONS:
            r = run_eval_one_anchor(tag, ckpt_dir, precision)
            if r is not None:
                results.append(r)

    if not results:
        print("No results collected.")
        return

    # Save CSV
    csv_path = OUT_DIR / "tp_errors_v1.csv"
    fieldnames = ["anchor", "precision", "n_samples", "n_gt_total", "n_tp",
                  "ap30", "ap50", "ap70", "mATE", "mASE", "mAOE", "elapsed_secs",
                  "n_trt_collab", "n_fallback"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"\n✓ Saved: {csv_path}")

    # Save JSON
    json_path = OUT_DIR / "tp_errors_v1.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'anchor':<12} {'prec':<6} {'AP50':<8} {'AP70':<8} {'mATE(m)':<10} {'mASE':<8} {'mAOE(rad)':<10} {'n_tp'}")
    print("-"*80)
    for r in results:
        print(f"{r['anchor']:<12} {r['precision']:<6} {r['ap50']:.4f}   {r['ap70']:.4f}   "
              f"{r['mATE']:.4f}     {r['mASE']:.4f}   {r['mAOE']:.4f}      {r['n_tp']}")


if __name__ == "__main__":
    main()
