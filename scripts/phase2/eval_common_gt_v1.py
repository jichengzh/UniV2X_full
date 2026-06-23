"""Phase M — 共同 GT 子集 mAOE 分析 (消除幸存者偏差).

方法:
  对 N_FRAMES 帧,先跑全部 8 configs 的推理,保存每帧的预测 boxes。
  然后对每帧: 找出在 **所有 8 configs 中都被 IoU≥0.5 匹配为 TP** 的 GT 框集合
  (= 共同 GT 子集 Common-GT)。在 Common-GT 上算各 config 的 mAOE。

  这样每个 config 的 mAOE 都在同一组目标上算,跨配置可比。

幸存者偏差分析 (来自 supervisor/team-lead 洞见):
  偏差方向保守: 剪枝丢掉难检/远/小目标(高误差) → Common-GT 比 all-TP 小且更易。
  如果 all-TP 下 mAOE 仍单调上升,Common-GT 下只会更强或持平(退化不被低估)。
  因此补这个分析是为了准确估计幅度,而不是担心信号反向。

Pipeline:
  Stage-1: 8 configs × N_FRAMES → 保存每帧预测 (pred_boxes, pred_scores, gt_boxes)
  Stage-2: 对每帧找共同 GT 集合 → 计算 per-frame Common-GT mAOE
  Stage-3: Bootstrap CI 在帧级别重采样

输出:
  results/common_gt_v1.json    — 每 config 在共同子集上的 mAOE + CI
  results/common_gt_v1.csv     — 摘要表
  results/common_gt_stats.json — 共同子集大小统计

用法:
  CUDA_VISIBLE_DEVICES=3 python scripts/phase2/eval_common_gt_v1.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
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
from opencood.utils.common_utils import update_dict, torch_tensor_to_numpy, convert_format, compute_iou
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
CONFIGS = [(tag, ckpt, prec) for tag, ckpt, _ in ANCHORS for prec in PRECISIONS]
CONFIG_KEYS = [f"{tag}_{prec}" for tag, _, prec in CONFIGS]

DAIR_RANGE = "102.4,51.2"
COLLAB_SPATIAL_SHAPE = (2, 64, 128, 256)
COLLAB_TEGO_SHAPE = (2, 2, 3)
N_FRAMES = 250       # per-run 帧数 (共同子集分析用 250 足够)
IOU_THRESH = 0.5
N_BOOTSTRAP = 1000


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
                "occ_single_list": []}
    else:
        record_len = ego["record_len"]
        fused, occ_outputs = model.pyramid_backbone.forward_collab(
            heter_feat_2d, record_len, affine_matrix, agent_modality_list, model.cam_crop_info)
        if model.shrink_flag:
            fused = model.shrink_conv(fused)
        return {"cls_preds": model.cls_head(fused), "reg_preds": model.reg_head(fused),
                "dir_preds": model.dir_head(fused), "occ_single_list": occ_outputs}


# ---------------------------------------------------------------------------
# TP matching WITH GT index tracking
# ---------------------------------------------------------------------------

def match_tp_with_gt_indices(pred_box_np, pred_score_np, gt_box_np):
    """Return matched GT indices and per-GT errors.

    Returns:
        matched_gt_indices: list[int] — GT box index for each TP
        ate_list: list[float]
        ase_list: list[float]
        aoe_list: list[float]
    """
    if gt_box_np is None or len(gt_box_np) == 0:
        return [], [], [], []
    if pred_box_np is None or len(pred_box_np) == 0:
        return [], [], [], []

    score_order = np.argsort(-pred_score_np)
    pred_sorted = pred_box_np[score_order]
    pred_params = corner_to_center(pred_sorted, order='lwh')
    gt_params = corner_to_center(gt_box_np, order='lwh')
    pred_polygon_list = list(convert_format(pred_sorted))
    gt_polygon_list = list(convert_format(gt_box_np))
    remaining_gt_idx = list(range(len(gt_box_np)))

    matched_gt_indices = []
    ate_list, ase_list, aoe_list = [], [], []

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

        matched_gt_indices.append(gt_idx)
        ate_list.append(ate)
        ase_list.append(1.0-size_iou)
        aoe_list.append(yaw_err)

    return matched_gt_indices, ate_list, ase_list, aoe_list


# ---------------------------------------------------------------------------
# Bootstrap CI (frame-level)
# ---------------------------------------------------------------------------

def bootstrap_ci_frames(frame_aoe_lists, B=N_BOOTSTRAP, seed=42):
    """Bootstrap over frames. Returns (mean, ci_lo, ci_hi, std)."""
    rng = np.random.default_rng(seed)
    n_frames = len(frame_aoe_lists)
    valid_frames = [aoe for aoe in frame_aoe_lists if len(aoe) > 0]
    if not valid_frames:
        return float("nan"), float("nan"), float("nan"), float("nan")

    boot_means = []
    for _ in range(B):
        idx = rng.integers(0, n_frames, size=n_frames)
        all_aoe = np.concatenate([frame_aoe_lists[i] for i in idx
                                   if len(frame_aoe_lists[i]) > 0]) if any(
            len(frame_aoe_lists[i]) > 0 for i in idx) else np.array([])
        boot_means.append(float(np.mean(all_aoe)) if len(all_aoe) > 0 else float("nan"))

    boot_means = np.array([x for x in boot_means if not np.isnan(x)])
    all_aoe_flat = np.concatenate([a for a in frame_aoe_lists if len(a) > 0])
    return (float(np.mean(all_aoe_flat)),
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
            float(np.std(all_aoe_flat)))


# ---------------------------------------------------------------------------
# Stage 1: Collect per-frame predictions for all configs
# ---------------------------------------------------------------------------

def collect_preds_one_config(tag, ckpt_dir, precision, n_frames=N_FRAMES):
    """Run inference for one config on N_FRAMES frames, return per-frame preds."""
    engine_path = CACHE / f"{tag}_{precision}.engine"
    if not engine_path.exists():
        return None

    hypes = yaml_utils.load_yaml(str(Path(ckpt_dir) / "config.yaml"))
    if "heter" in hypes:
        x_max, y_max = float(DAIR_RANGE.split(",")[0]), float(DAIR_RANGE.split(",")[1])
        new_range = [-x_max, -y_max,
                     hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
                     x_max, y_max,
                     hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5]]
        hypes = update_dict(hypes, {"cav_lidar_range": new_range,
                                     "lidar_range": new_range, "gt_range": new_range})
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(ckpt_dir, model)
    model.cuda().eval()

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)
    trt_collab = TrtCollabN2(str(engine_path))

    frame_preds = []  # list of (pred_box_np, score_np, gt_box_np)
    n_done = 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_data in loader:
            if batch_data is None:
                continue
            if n_done >= n_frames:
                break
            batch_data = train_utils.to_device(batch_data, "cuda")
            out = hybrid_forward(model, batch_data, trt_collab)
            out_wrapped = {"ego": out}
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, out_wrapped)

            pred_np = torch_tensor_to_numpy(pred_box) if pred_box is not None else None
            score_np = torch_tensor_to_numpy(pred_score) if pred_score is not None else None
            gt_np = torch_tensor_to_numpy(gt_box) if gt_box is not None else None
            frame_preds.append((pred_np, score_np, gt_np))
            n_done += 1

    elapsed = time.time() - t0
    print(f"  [{tag}_{precision}] {n_done} frames in {elapsed:.0f}s")

    del model, trt_collab
    torch.cuda.empty_cache()
    return frame_preds


# ---------------------------------------------------------------------------
# Stage 2: Common GT subset analysis
# ---------------------------------------------------------------------------

def analyze_common_gt(all_preds: dict[str, list]):
    """
    all_preds: {config_key: [(pred_np, score_np, gt_np), ...]} for N_FRAMES each

    Returns:
        per_config_common_gt_aoe: {config_key: [per_frame_aoe_array]}
        common_gt_stats: {frame_i: {n_gt_total, n_common_gt}}
    """
    config_keys = list(all_preds.keys())
    n_frames = len(all_preds[config_keys[0]])

    per_config_common_aoe = defaultdict(list)  # {key: [aoe_per_frame_array]}
    common_gt_stats = []

    for frame_i in range(n_frames):
        # Get GT boxes (same for all configs - same dataset, same frame)
        gt_np = all_preds[config_keys[0]][frame_i][2]
        if gt_np is None or len(gt_np) == 0:
            for key in config_keys:
                per_config_common_aoe[key].append(np.array([]))
            common_gt_stats.append({"frame": frame_i, "n_gt": 0, "n_common_gt": 0})
            continue

        n_gt = len(gt_np)

        # For each config: get matched GT indices
        config_matched_sets = {}
        config_errors = {}  # {config_key: {gt_idx: aoe}}

        for key in config_keys:
            pred_np, score_np, _ = all_preds[key][frame_i]
            if pred_np is None or score_np is None:
                config_matched_sets[key] = set()
                config_errors[key] = {}
                continue
            gt_indices, ate_l, ase_l, aoe_l = match_tp_with_gt_indices(pred_np, score_np, gt_np)
            config_matched_sets[key] = set(gt_indices)
            config_errors[key] = {idx: aoe for idx, aoe in zip(gt_indices, aoe_l)}

        # Common GT = intersection of all matched sets
        common_gt = set.intersection(*config_matched_sets.values()) if config_matched_sets else set()
        n_common = len(common_gt)

        common_gt_stats.append({
            "frame": frame_i, "n_gt": n_gt, "n_common_gt": n_common,
            "frac": n_common / n_gt if n_gt > 0 else 0.0
        })

        # Compute per-config mAOE on common GT
        for key in config_keys:
            aoe_vals = [config_errors[key][gt_idx] for gt_idx in common_gt
                        if gt_idx in config_errors[key]]
            per_config_common_aoe[key].append(np.array(aoe_vals))

    return per_config_common_aoe, common_gt_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import csv

    # Stage 1: Collect predictions for all 8 configs
    print("Stage 1: Collecting predictions for all 8 configs")
    print(f"  N_FRAMES={N_FRAMES} per config")
    all_preds = {}
    for tag, ckpt_dir, precision in CONFIGS:
        key = f"{tag}_{precision}"
        print(f"\n  [{key}]")
        preds = collect_preds_one_config(tag, ckpt_dir, precision, N_FRAMES)
        if preds is not None:
            all_preds[key] = preds

    if len(all_preds) < 2:
        print("ERROR: need at least 2 configs")
        return

    print(f"\nStage 2: Common GT analysis ({len(all_preds)} configs, {N_FRAMES} frames)")
    per_config_common_aoe, common_gt_stats = analyze_common_gt(all_preds)

    # Summary of common GT size
    n_gt_total_arr = [s["n_gt"] for s in common_gt_stats]
    n_common_arr = [s["n_common_gt"] for s in common_gt_stats]
    frac_arr = [s["frac"] for s in common_gt_stats if s["n_gt"] > 0]
    print(f"\n  Common GT stats over {N_FRAMES} frames:")
    print(f"  avg n_gt={np.mean(n_gt_total_arr):.1f}, avg n_common={np.mean(n_common_arr):.1f}")
    print(f"  common/total = {np.mean(frac_arr):.3f} ({100*np.mean(frac_arr):.1f}%)")
    total_common_tp = sum(sum(len(a) for a in per_config_common_aoe[k]) for k in list(all_preds.keys())[:1])
    print(f"  total common TPs (one config) = {sum(len(a) for a in per_config_common_aoe[list(all_preds.keys())[0]])}")

    # Stage 3: Bootstrap CI for each config on common GT
    print("\nStage 3: Bootstrap CI on common GT")
    results = []
    for key in CONFIG_KEYS:
        if key not in per_config_common_aoe:
            continue
        frame_aoe_list = per_config_common_aoe[key]
        mean_aoe, ci_lo, ci_hi, std_aoe = bootstrap_ci_frames(frame_aoe_list)
        all_aoe = np.concatenate([a for a in frame_aoe_list if len(a) > 0])
        n_tp_common = len(all_aoe)

        tag, precision = key.rsplit("_", 1)
        r = {
            "anchor": tag, "precision": precision,
            "n_frames": N_FRAMES, "n_tp_common": n_tp_common,
            "mAOE_common": float(mean_aoe),
            "mAOE_ci_lo": float(ci_lo),
            "mAOE_ci_hi": float(ci_hi),
            "mAOE_std": float(std_aoe),
            "ci_width": float(ci_hi - ci_lo),
        }
        results.append(r)
        print(f"  {key:<20} mAOE={mean_aoe:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"CI_w={ci_hi-ci_lo:.4f}  n_tp={n_tp_common}")

    # Compute SNR for FP16 pruning axis (common GT)
    fp16_results = [r for r in results if r["precision"] == "fp16"]
    fp16_results.sort(key=lambda x: ["base", "pruned25", "pruned50", "pruned75"].index(x["anchor"]))

    print("\n" + "="*70)
    print("COMMON GT SIGNAL ANALYSIS (幸存者偏差消除后)")
    print("="*70)
    print("\n[FP16 剪枝轴 mAOE — Common GT]")
    for r in fp16_results:
        print(f"  {r['anchor']:<12} mAOE={r['mAOE_common']:.4f}  "
              f"[{r['mAOE_ci_lo']:.4f}, {r['mAOE_ci_hi']:.4f}]  "
              f"CI_width={r['ci_width']:.4f}  n_tp={r['n_tp_common']}")

    if len(fp16_results) >= 2:
        total_delta = fp16_results[-1]["mAOE_common"] - fp16_results[0]["mAOE_common"]
        avg_ci_half = sum(r["ci_width"] for r in fp16_results) / len(fp16_results) / 2
        snr = abs(total_delta) / avg_ci_half if avg_ci_half > 0 else float("inf")
        print(f"\n  Δ(base→p75) = {total_delta:+.4f} rad")
        print(f"  avg CI half-width = {avg_ci_half:.4f} rad")
        print(f"  SNR = {snr:.1f}×")
        print(f"  → {'SIGNAL CLEAR (>5×noise)' if snr > 5 else 'SIGNAL PRESENT (>2×noise)' if snr > 2 else 'SIGNAL WEAK'}")

        # CI overlap check
        print("\n  CI 不重叠检查 (判断趋势是否真实):")
        for i in range(len(fp16_results)-1):
            a, b = fp16_results[i], fp16_results[i+1]
            overlap = a["mAOE_ci_hi"] > b["mAOE_ci_lo"]
            print(f"  {a['anchor']} [{a['mAOE_ci_lo']:.4f},{a['mAOE_ci_hi']:.4f}] "
                  f"vs {b['anchor']} [{b['mAOE_ci_lo']:.4f},{b['mAOE_ci_hi']:.4f}] "
                  f"→ {'OVERLAPPING' if overlap else 'NON-OVERLAPPING ✅'}")

    # INT8 step comparison
    print("\n[INT8 步骤 mAOE — Common GT]")
    for tag_name in ["base", "pruned25", "pruned50", "pruned75"]:
        fp16_r = next((r for r in results if r["anchor"] == tag_name and r["precision"] == "fp16"), None)
        int8_r = next((r for r in results if r["anchor"] == tag_name and r["precision"] == "int8"), None)
        if fp16_r and int8_r:
            delta = int8_r["mAOE_common"] - fp16_r["mAOE_common"]
            ci_half = (fp16_r["ci_width"] + int8_r["ci_width"]) / 4
            snr_int8 = abs(delta) / ci_half if ci_half > 0 else float("inf")
            print(f"  {tag_name:<12}: fp16={fp16_r['mAOE_common']:.4f}  int8={int8_r['mAOE_common']:.4f}  "
                  f"Δ={delta:+.4f}  CI_half={ci_half:.4f}  SNR={snr_int8:.2f}×  "
                  f"→ {'SIGNAL' if snr_int8 > 2 else 'NOISE'}")

    # Common GT size stats
    print(f"\n[共同 GT 子集大小]")
    print(f"  avg fraction = {np.mean(frac_arr)*100:.1f}%  "
          f"(每帧 ~{np.mean(n_common_arr):.0f}/{np.mean(n_gt_total_arr):.0f} GT 框在全 8 configs 中共同匹配)")

    # Save results
    json_path = OUT_DIR / "common_gt_v1.json"
    output = {
        "method": "common_GT_subset",
        "n_frames": N_FRAMES,
        "n_configs": len(all_preds),
        "iou_thresh": IOU_THRESH,
        "n_bootstrap": N_BOOTSTRAP,
        "avg_common_gt_frac": float(np.mean(frac_arr)),
        "avg_n_common_gt": float(np.mean(n_common_arr)),
        "avg_n_gt_total": float(np.mean(n_gt_total_arr)),
        "results": results,
        "frame_stats": common_gt_stats[:20],  # save first 20 for inspection
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved: {json_path}")

    csv_path = OUT_DIR / "common_gt_v1.csv"
    fields = ["anchor", "precision", "n_tp_common", "mAOE_common",
              "mAOE_ci_lo", "mAOE_ci_hi", "ci_width"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"✓ Saved: {csv_path}")


if __name__ == "__main__":
    main()
