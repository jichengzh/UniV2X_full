"""Phase M — 修正版共同 GT 子集分析 (强制 epoch25 消除权重不匹配).

问题: pruned25/50/75 在 stage_a (May 12) 后继续训练 (May 27-28), 生成了
  net_epoch_bestval_at{26,29,31}.pth. load_saved_model 会优先加载 bestval 文件.
  但 TRT engines 用的是 May 12 时的 ONNX (基于 epoch25 权重).
  → PyTorch 预处理用 epoch31, TRT engine 用 epoch25 → **权重不匹配** → AP 偏低.

修复: 对所有 pruned 模型强制加载 net_epoch25.pth (与 ONNX/engine 构建时一致).
  base 模型 bestval_at23.pth 与 stage_a 时相同, 无需修复.

输出:
  results/common_gt_corrected.csv  — 修正后的共同子集 mAOE + CI
  results/common_gt_corrected.json
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

# Force epoch25 for all pruned models to match ONNX/engine export time (May 12)
# Base uses bestval_at23 which was already stable at May 12
ANCHORS = [
    ("base",     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",  None),    # use bestval (epoch23)
    ("pruned25", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10",       "net_epoch25.pth"),  # force epoch25
    ("pruned50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",       "net_epoch25.pth"),  # force epoch25
    ("pruned75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10",       "net_epoch25.pth"),  # force epoch25
]
PRECISIONS = ["fp16", "int8"]
CONFIGS = [(tag, ckpt, epoch_file, prec) for tag, ckpt, epoch_file in ANCHORS for prec in PRECISIONS]
CONFIG_KEYS = [f"{tag}_{prec}" for tag, _, _, prec in CONFIGS]

DAIR_RANGE = "102.4,51.2"
COLLAB_SPATIAL_SHAPE = (2, 64, 128, 256)
COLLAB_TEGO_SHAPE = (2, 2, 3)
N_FRAMES = 250
IOU_THRESH = 0.5
N_BOOTSTRAP = 1000


class TrtCollabN2:
    def __init__(self, engine_path, spatial_shape=COLLAB_SPATIAL_SHAPE, tego_shape=COLLAB_TEGO_SHAPE):
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
        bufs = {self.spatial_name: spatial.float().contiguous(), self.tego_name: t_ego.float().contiguous()}
        for n in self.output_names:
            bufs[n] = torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=torch.float32, device="cuda")
        for n in self.input_names + self.output_names:
            ctx.set_tensor_address(n, int(bufs[n].data_ptr()))
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return tuple(bufs[n] for n in self.output_names)


def hybrid_forward(model, batch_data, trt_collab=None):
    ego = batch_data["ego"]
    al = ego["agent_modality_list"]
    affine = normalize_pairwise_tfm(ego["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size)
    cnt = Counter(al)
    feat_dict = {}
    for m in model.modality_name_list:
        if m not in cnt: continue
        f = getattr(model, f"encoder_{m}")(ego, m)
        f = getattr(model, f"backbone_{m}")({"spatial_features": f})["spatial_features_2d"]
        f = getattr(model, f"aligner_{m}")(f)
        feat_dict[m] = f
    counting = {m: 0 for m in model.modality_name_list}
    hl = []
    for m in al: hl.append(feat_dict[m][counting[m]]); counting[m] += 1
    hf = torch.stack(hl)
    n = hf.shape[0]
    if n == 2 and trt_collab is not None:
        t_ego = affine[0, 0, :2, :, :].contiguous()
        cls_p, reg_p, dir_p = trt_collab(hf.contiguous(), t_ego)
        return {"cls_preds": cls_p, "reg_preds": reg_p, "dir_preds": dir_p, "occ_single_list": []}
    else:
        record_len = ego["record_len"]
        fused, occ = model.pyramid_backbone.forward_collab(hf, record_len, affine, al, model.cam_crop_info)
        if model.shrink_flag: fused = model.shrink_conv(fused)
        return {"cls_preds": model.cls_head(fused), "reg_preds": model.reg_head(fused),
                "dir_preds": model.dir_head(fused), "occ_single_list": occ}


def match_tp_with_gt_indices(pred_box_np, pred_score_np, gt_box_np):
    if gt_box_np is None or len(gt_box_np) == 0: return [], [], [], []
    if pred_box_np is None or len(pred_box_np) == 0: return [], [], [], []
    score_order = np.argsort(-pred_score_np)
    pred_sorted = pred_box_np[score_order]
    pred_params = corner_to_center(pred_sorted, order='lwh')
    gt_params = corner_to_center(gt_box_np, order='lwh')
    pred_polygon_list = list(convert_format(pred_sorted))
    gt_polygon_list = list(convert_format(gt_box_np))
    remaining_gt_idx = list(range(len(gt_box_np)))
    matched_gt_indices, ate_list, ase_list, aoe_list = [], [], [], []
    for i in range(len(pred_sorted)):
        if not remaining_gt_idx: break
        ious = compute_iou(pred_polygon_list[i], [gt_polygon_list[j] for j in remaining_gt_idx])
        if not len(ious) or np.max(ious) < IOU_THRESH: continue
        best_local = int(np.argmax(ious))
        gt_idx = remaining_gt_idx.pop(best_local)
        p, g = pred_params[i], gt_params[gt_idx]
        ate = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        l_p, w_p, h_p = abs(p[3]), abs(p[4]), abs(p[5])
        l_g, w_g, h_g = abs(g[3]), abs(g[4]), abs(g[5])
        i_vol = min(l_p,l_g)*min(w_p,w_g)*min(h_p,h_g)
        u_vol = l_p*w_p*h_p + l_g*w_g*h_g - i_vol
        size_iou = i_vol/u_vol if u_vol > 1e-9 else 0.0
        delta = abs(p[6]-g[6]) % np.pi
        yaw_err = float(min(delta, np.pi-delta))
        matched_gt_indices.append(gt_idx); ate_list.append(ate)
        ase_list.append(1.0-size_iou); aoe_list.append(yaw_err)
    return matched_gt_indices, ate_list, ase_list, aoe_list


def bootstrap_ci_frames(frame_aoe_lists, B=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n_frames = len(frame_aoe_lists)
    boot_means = []
    for _ in range(B):
        idx = rng.integers(0, n_frames, size=n_frames)
        flat = np.concatenate([frame_aoe_lists[i] for i in idx if len(frame_aoe_lists[i]) > 0])
        boot_means.append(float(np.mean(flat)) if len(flat) > 0 else float("nan"))
    boot_means = np.array([x for x in boot_means if not np.isnan(x)])
    all_aoe = np.concatenate([a for a in frame_aoe_lists if len(a) > 0])
    return (float(np.mean(all_aoe)),
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)),
            float(np.std(all_aoe)))


def collect_preds_corrected(tag, ckpt_dir, epoch_file, precision, n_frames=N_FRAMES):
    """Collect predictions with corrected epoch loading."""
    engine_path = CACHE / f"{tag}_{precision}.engine"
    if not engine_path.exists():
        return None

    hypes = yaml_utils.load_yaml(str(Path(ckpt_dir) / "config.yaml"))
    if "heter" in hypes:
        x_max, y_max = float(DAIR_RANGE.split(",")[0]), float(DAIR_RANGE.split(",")[1])
        new_range = [-x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
                     x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5]]
        hypes = update_dict(hypes, {"cav_lidar_range": new_range, "lidar_range": new_range,
                                     "gt_range": new_range})
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    model = train_utils.create_model(hypes)

    if epoch_file is not None:
        # Force load specific epoch to match engine
        epoch_path = Path(ckpt_dir) / epoch_file
        print(f"  FORCE LOAD {epoch_file} (match ONNX/engine epoch)")
        state = torch.load(str(epoch_path), map_location="cpu")
        model.load_state_dict(state)
    else:
        _, model = train_utils.load_saved_model(ckpt_dir, model)

    model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)
    trt_collab = TrtCollabN2(str(engine_path))

    frame_preds = []
    n_done = 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_data in loader:
            if batch_data is None: continue
            if n_done >= n_frames: break
            batch_data = train_utils.to_device(batch_data, "cuda")
            out = hybrid_forward(model, batch_data, trt_collab)
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, {"ego": out})
            pred_np = torch_tensor_to_numpy(pred_box) if pred_box is not None else None
            score_np = torch_tensor_to_numpy(pred_score) if pred_score is not None else None
            gt_np = torch_tensor_to_numpy(gt_box) if gt_box is not None else None
            frame_preds.append((pred_np, score_np, gt_np))
            n_done += 1

    elapsed = time.time() - t0
    epoch_info = epoch_file if epoch_file else "bestval"
    print(f"  [{tag}_{precision}] {n_done} frames in {elapsed:.0f}s (epoch={epoch_info})")
    del model, trt_collab
    torch.cuda.empty_cache()
    return frame_preds


def analyze_common_gt(all_preds):
    config_keys = list(all_preds.keys())
    n_frames = len(all_preds[config_keys[0]])
    per_config_common_aoe = defaultdict(list)
    common_gt_stats = []

    for frame_i in range(n_frames):
        gt_np = all_preds[config_keys[0]][frame_i][2]
        if gt_np is None or len(gt_np) == 0:
            for k in config_keys: per_config_common_aoe[k].append(np.array([]))
            common_gt_stats.append({"frame": frame_i, "n_gt": 0, "n_common_gt": 0, "frac": 0.0})
            continue

        n_gt = len(gt_np)
        config_matched_sets = {}
        config_errors = {}

        for key in config_keys:
            pred_np, score_np, _ = all_preds[key][frame_i]
            if pred_np is None or score_np is None:
                config_matched_sets[key] = set(); config_errors[key] = {}; continue
            gt_indices, _, _, aoe_l = match_tp_with_gt_indices(pred_np, score_np, gt_np)
            config_matched_sets[key] = set(gt_indices)
            config_errors[key] = {idx: aoe for idx, aoe in zip(gt_indices, aoe_l)}

        common_gt = set.intersection(*config_matched_sets.values()) if config_matched_sets else set()
        n_common = len(common_gt)
        common_gt_stats.append({"frame": frame_i, "n_gt": n_gt, "n_common_gt": n_common,
                                  "frac": n_common/n_gt if n_gt > 0 else 0.0})

        for key in config_keys:
            aoe_vals = [config_errors[key][i] for i in common_gt if i in config_errors[key]]
            per_config_common_aoe[key].append(np.array(aoe_vals))

    return per_config_common_aoe, common_gt_stats


def main():
    import csv

    print("=== Phase M 修正版共同 GT 子集分析 ===")
    print("权重修正: 所有 pruned 模型强制 epoch25 (与 TRT engine/ONNX 导出时一致)")
    print()

    all_preds = {}
    for tag, ckpt_dir, epoch_file, precision in CONFIGS:
        key = f"{tag}_{precision}"
        print(f"\n  [{key}]")
        preds = collect_preds_corrected(tag, ckpt_dir, epoch_file, precision, N_FRAMES)
        if preds is not None:
            all_preds[key] = preds

    print(f"\nStage 2: Common GT analysis")
    per_config_common_aoe, common_gt_stats = analyze_common_gt(all_preds)

    n_gt_arr = [s["n_gt"] for s in common_gt_stats]
    n_common_arr = [s["n_common_gt"] for s in common_gt_stats]
    frac_arr = [s["frac"] for s in common_gt_stats if s["n_gt"] > 0]
    print(f"\n  Common GT: avg {np.mean(n_common_arr):.1f}/{np.mean(n_gt_arr):.1f} = {np.mean(frac_arr)*100:.1f}%")

    print("\nStage 3: Bootstrap CI (B=1000, frame-level)")
    results = []
    for key in CONFIG_KEYS:
        if key not in per_config_common_aoe: continue
        frame_aoe_list = per_config_common_aoe[key]
        mean_aoe, ci_lo, ci_hi, std_aoe = bootstrap_ci_frames(frame_aoe_list)
        all_aoe = np.concatenate([a for a in frame_aoe_list if len(a) > 0])
        tag_name, prec = key.rsplit("_", 1)
        r = {"anchor": tag_name, "precision": prec, "n_frames": N_FRAMES,
             "n_tp_common": len(all_aoe),
             "mAOE_common": float(mean_aoe), "mAOE_ci_lo": float(ci_lo),
             "mAOE_ci_hi": float(ci_hi), "mAOE_std": float(std_aoe),
             "ci_width": float(ci_hi - ci_lo),
             "epoch_used": "bestval" if tag_name == "base" else "epoch25_forced"}
        results.append(r)
        print(f"  {key:<20} mAOE={mean_aoe:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"CI_w={ci_hi-ci_lo:.4f}  n_tp={len(all_aoe)}  epoch={r['epoch_used']}")

    # SNR analysis
    fp16 = [r for r in results if r["precision"] == "fp16"]
    fp16.sort(key=lambda x: ["base","pruned25","pruned50","pruned75"].index(x["anchor"]))

    print("\n" + "="*70)
    print("共同 GT + 修正权重 — 剪枝轴信号分析")
    print("="*70)
    print("\n[FP16 mAOE — Common GT, epoch25-corrected]")
    for r in fp16:
        print(f"  {r['anchor']:<12} mAOE={r['mAOE_common']:.4f}  [{r['mAOE_ci_lo']:.4f},{r['mAOE_ci_hi']:.4f}]  "
              f"CI_width={r['ci_width']:.4f}")

    if len(fp16) >= 2:
        delta = fp16[-1]["mAOE_common"] - fp16[0]["mAOE_common"]
        avg_ci_half = sum(r["ci_width"] for r in fp16) / len(fp16) / 2
        snr = abs(delta) / avg_ci_half if avg_ci_half > 0 else float("inf")
        print(f"\n  Δ(base→p75) = {delta:+.4f} rad")
        print(f"  avg CI half-width = {avg_ci_half:.4f} rad")
        print(f"  SNR = {snr:.1f}×")
        print(f"  → {'SIGNAL CLEAR (>5×noise)' if snr > 5 else 'SIGNAL PRESENT' if snr > 2 else 'WEAK'}")

        print("\n  CI 不重叠检查:")
        for i in range(len(fp16)-1):
            a, b = fp16[i], fp16[i+1]
            overlap = a["mAOE_ci_hi"] > b["mAOE_ci_lo"]
            print(f"  {a['anchor']} [{a['mAOE_ci_lo']:.4f},{a['mAOE_ci_hi']:.4f}] "
                  f"vs {b['anchor']} [{b['mAOE_ci_lo']:.4f},{b['mAOE_ci_hi']:.4f}] "
                  f"→ {'OVERLAPPING' if overlap else 'NON-OVERLAPPING ✅'}")

    print("\n[INT8 — Common GT, epoch25-corrected]")
    for tag_name in ["base","pruned25","pruned50","pruned75"]:
        f = next((r for r in results if r["anchor"] == tag_name and r["precision"] == "fp16"), None)
        i = next((r for r in results if r["anchor"] == tag_name and r["precision"] == "int8"), None)
        if f and i:
            delta = i["mAOE_common"] - f["mAOE_common"]
            ci_half = (f["ci_width"] + i["ci_width"]) / 4
            snr = abs(delta) / ci_half if ci_half > 0 else float("inf")
            print(f"  {tag_name:<12}: Δ={delta:+.5f}  CI_half={ci_half:.5f}  SNR={snr:.2f}× "
                  f"→ {'SIGNAL' if snr > 2 else 'NOISE'}")

    print(f"\n  Common GT fraction = {np.mean(frac_arr)*100:.1f}%")

    # Save
    out = {
        "method": "common_GT_subset_epoch25_corrected",
        "n_frames": N_FRAMES, "iou_thresh": IOU_THRESH, "n_bootstrap": N_BOOTSTRAP,
        "epoch_fix": "pruned models force-load net_epoch25.pth to match ONNX/engine",
        "avg_common_gt_frac": float(np.mean(frac_arr)),
        "results": results,
    }
    json_path = OUT_DIR / "common_gt_corrected.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved: {json_path}")

    csv_path = OUT_DIR / "common_gt_corrected.csv"
    fields = ["anchor", "precision", "n_tp_common", "mAOE_common",
              "mAOE_ci_lo", "mAOE_ci_hi", "ci_width", "epoch_used"]
    import csv as csv_mod
    with open(csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in fields})
    print(f"✓ Saved: {csv_path}")


if __name__ == "__main__":
    main()
