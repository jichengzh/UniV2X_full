"""Phase M — 全集 1789 帧 mAOE 评估 (epoch25 修正 + bootstrap CI).

目的: 给 dataset_v2 补充干净的 mAOE 均值 + 全集 bootstrap 95% CI.

修正说明:
  pruned25/50/75 在 stage_a 后继续训练，load_saved_model 会加载新 bestval_at*.pth。
  但 TRT engine 仍用 epoch25 权重 → 权重不匹配。
  修复: 所有 pruned 模型强制加载 net_epoch25.pth（与 ONNX/engine 构建时一致）。
  base 用 bestval_at23（stage_a 时已稳定，无需修复）。

输出:
  results/tp_errors_corrected_full.csv  — 8 configs 均值 + 95% CI
  results/tp_errors_corrected_full.json — 完整数据

用法:
  CUDA_VISIBLE_DEVICES=3 python scripts/phase2/eval_tp_errors_corrected_full.py
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

# Force epoch25 for pruned models to match ONNX/engine (built May 12 from epoch25)
ANCHORS = [
    ("base",     "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",  None),           # bestval = ep23 (stable)
    ("pruned25", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned25_2026_05_10",       "net_epoch25.pth"),
    ("pruned50", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned50_2026_05_10",       "net_epoch25.pth"),
    ("pruned75", "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10",       "net_epoch25.pth"),
]
PRECISIONS = ["fp16", "int8"]

DAIR_RANGE = "102.4,51.2"
COLLAB_SPATIAL_SHAPE = (2, 64, 128, 256)
COLLAB_TEGO_SHAPE = (2, 2, 3)
N_SAMPLES = 1789   # full DAIR val
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
        return {"cls_preds": cls_p, "reg_preds": reg_p, "dir_preds": dir_p, "occ_single_list": [], "_path": "trt"}
    else:
        record_len = ego["record_len"]
        fused, occ = model.pyramid_backbone.forward_collab(hf, record_len, affine, al, model.cam_crop_info)
        if model.shrink_flag: fused = model.shrink_conv(fused)
        return {"cls_preds": model.cls_head(fused), "reg_preds": model.reg_head(fused),
                "dir_preds": model.dir_head(fused), "occ_single_list": occ, "_path": "fallback"}


def compute_tp_errors_frame(pred_box_np, pred_score_np, gt_box_np):
    """Per-frame TP error arrays. Returns (ate_arr, ase_arr, aoe_arr)."""
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
        if not remaining_gt_idx: break
        ious = compute_iou(pred_polygon_list[i], [gt_polygon_list[j] for j in remaining_gt_idx])
        if not len(ious) or np.max(ious) < IOU_THRESH: continue
        best = int(np.argmax(ious)); gt_idx = remaining_gt_idx.pop(best)
        p, g = pred_params[i], gt_params[gt_idx]
        ate = float(np.sqrt((p[0]-g[0])**2 + (p[1]-g[1])**2))
        l_p, w_p, h_p = abs(p[3]), abs(p[4]), abs(p[5])
        l_g, w_g, h_g = abs(g[3]), abs(g[4]), abs(g[5])
        i_vol = min(l_p,l_g)*min(w_p,w_g)*min(h_p,h_g)
        u_vol = l_p*w_p*h_p + l_g*w_g*h_g - i_vol
        size_iou = i_vol/u_vol if u_vol > 1e-9 else 0.0
        delta = abs(p[6]-g[6]) % np.pi
        ate_l.append(ate); ase_l.append(1.0-size_iou); aoe_l.append(float(min(delta, np.pi-delta)))
    return np.array(ate_l), np.array(ase_l), np.array(aoe_l)


def bootstrap_ci(frame_ate_list, frame_ase_list, frame_aoe_list, B=N_BOOTSTRAP, seed=42):
    """Frame-level bootstrap CI for mATE, mASE, mAOE."""
    rng = np.random.default_rng(seed)
    n_frames = len(frame_aoe_list)
    boot_ate, boot_ase, boot_aoe = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n_frames, size=n_frames)
        def cat(lst): return np.concatenate([lst[i] for i in idx if len(lst[i]) > 0]) if any(len(lst[i]) > 0 for i in idx) else np.array([])
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
        "mATE": float(np.mean(all_ate)) if len(all_ate) > 0 else float("nan"),
        "mATE_ci_lo": ci_from(boot_ate)[0], "mATE_ci_hi": ci_from(boot_ate)[1],
        "mASE": float(np.mean(all_ase)) if len(all_ase) > 0 else float("nan"),
        "mASE_ci_lo": ci_from(boot_ase)[0], "mASE_ci_hi": ci_from(boot_ase)[1],
        "mAOE": float(np.mean(all_aoe)) if len(all_aoe) > 0 else float("nan"),
        "mAOE_ci_lo": ci_from(boot_aoe)[0], "mAOE_ci_hi": ci_from(boot_aoe)[1],
        "n_tp": len(all_aoe),
    }


def run_one(tag, ckpt_dir, epoch_file, precision):
    engine_path = CACHE / f"{tag}_{precision}.engine"
    if not engine_path.exists():
        print(f"  [SKIP] {tag}_{precision}: engine not found")
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
        epoch_path = Path(ckpt_dir) / epoch_file
        print(f"  FORCE LOAD {epoch_file}")
        state = torch.load(str(epoch_path), map_location="cpu")
        model.load_state_dict(state)
    else:
        _, model = train_utils.load_saved_model(ckpt_dir, model)

    model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)
    trt_collab = TrtCollabN2(str(engine_path))

    # AP accumulators (cross-validate with stage_a)
    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}

    frame_ate_list, frame_ase_list, frame_aoe_list = [], [], []
    n_done, n_trt, n_fb = 0, 0, 0
    t0 = time.time()

    with torch.inference_mode():
        for batch_data in loader:
            if batch_data is None: continue
            if n_done >= N_SAMPLES: break
            batch_data = train_utils.to_device(batch_data, "cuda")
            out = hybrid_forward(model, batch_data, trt_collab)
            out_wrapped = {"ego": out}
            pred_box, pred_score, gt_box = dataset.post_process(batch_data, out_wrapped)

            if out["_path"] == "trt": n_trt += 1
            else: n_fb += 1

            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou_th)

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

            if n_done % 200 == 0:
                print(f"  {n_done}/{N_SAMPLES}  trt={n_trt} fb={n_fb}  "
                      f"elapsed={time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    out_dir = OUT_DIR / f"m_eval_corrected_{tag}_{precision}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(out_dir))

    print(f"  Bootstrap CI (B={N_BOOTSTRAP})...")
    ci_stats = bootstrap_ci(frame_ate_list, frame_ase_list, frame_aoe_list)

    result = {
        "anchor": tag, "precision": precision,
        "epoch_used": "bestval" if epoch_file is None else epoch_file,
        "n_samples": n_done, "n_trt": n_trt, "n_fallback": n_fb,
        "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
        "elapsed_secs": elapsed,
        **ci_stats,
    }
    print(f"  AP30={ap30:.4f} AP50={ap50:.4f} AP70={ap70:.4f}")
    print(f"  mATE={ci_stats['mATE']:.4f}m [{ci_stats['mATE_ci_lo']:.4f},{ci_stats['mATE_ci_hi']:.4f}]")
    print(f"  mAOE={ci_stats['mAOE']:.4f}rad [{ci_stats['mAOE_ci_lo']:.4f},{ci_stats['mAOE_ci_hi']:.4f}]")
    print(f"  n_tp={ci_stats['n_tp']}  elapsed={elapsed:.0f}s")

    del model, trt_collab
    torch.cuda.empty_cache()
    return result


def main():
    import csv
    results = []

    for tag, ckpt_dir, epoch_file in ANCHORS:
        for precision in PRECISIONS:
            print(f"\n{'='*60}")
            print(f"  {tag}  {precision}  (epoch={epoch_file or 'bestval'})")
            print(f"{'='*60}")
            r = run_one(tag, ckpt_dir, epoch_file, precision)
            if r is not None:
                results.append(r)

    if not results:
        print("No results.")
        return

    # Save CSV
    csv_path = OUT_DIR / "tp_errors_corrected_full.csv"
    fields = ["anchor", "precision", "epoch_used", "n_samples", "n_tp",
              "ap30", "ap50", "ap70",
              "mATE", "mATE_ci_lo", "mATE_ci_hi",
              "mASE", "mASE_ci_lo", "mASE_ci_hi",
              "mAOE", "mAOE_ci_lo", "mAOE_ci_hi",
              "elapsed_secs"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\n✓ Saved: {csv_path}")

    json_path = OUT_DIR / "tp_errors_corrected_full.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {json_path}")

    # Summary
    print("\n" + "="*70)
    print(f"{'anchor':<12} {'prec':<6} {'AP70':<8} {'mAOE':<8} {'CI_lo':<8} {'CI_hi':<8} {'epoch'}")
    print("-"*70)
    for r in results:
        print(f"{r['anchor']:<12} {r['precision']:<6} {r['ap70']:.4f}   "
              f"{r['mAOE']:.4f}   {r['mAOE_ci_lo']:.4f}   {r['mAOE_ci_hi']:.4f}   {r['epoch_used']}")

    # SNR for FP16 pruning axis
    fp16 = [r for r in results if r["precision"] == "fp16"]
    fp16.sort(key=lambda x: ["base","pruned25","pruned50","pruned75"].index(x["anchor"]))
    if len(fp16) >= 2:
        delta = fp16[-1]["mAOE"] - fp16[0]["mAOE"]
        avg_ci_half = sum((r["mAOE_ci_hi"]-r["mAOE_ci_lo"])/2 for r in fp16) / len(fp16)
        snr = delta / avg_ci_half if avg_ci_half > 0 else float("inf")
        print(f"\nFP16 pruning axis SNR: Δ={delta:.4f} / CI_half={avg_ci_half:.4f} = {snr:.1f}×")


if __name__ == "__main__":
    main()
