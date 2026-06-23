"""M4.9 V2X baseline per-stage timing — HEAL heter baselines.

Profiles 5 HEAL OPV2V LiDAR baselines that share identical
encoder/backbone/shrinker/heads/postproc but differ only in fusion_net:

  fcooper  → MaxFusion       (max-pool over agents)
  attfuse  → AttFusion       (self-attention)
  v2vnet   → V2VNetFusion    (GNN message passing)
  v2xvit   → V2XViTFusion    (heterogeneous transformer)
  where2comm → Where2commFusion (spatial confidence + attention)

Forward decomposition (HeterModelBaseline.forward):
  encoder_m1 → backbone_m1 → shrinker_m1 (per-modality)
              → fusion_net (VARIABLE)
              → shrink_conv (top-level) → cls/reg/dir heads
              → postproc (decode/dir/corners/NMS/range_mask)

Differs from m4_8_pyramid_per_stage_timing.py:
  - No aligner (baselines don't have one)
  - Extra per-modality shrinker_m1 stage
  - fusion_net replaces pyramid_backbone.forward_collab

Usage:
  python m4_9_v2x_baselines_timing.py \\
      --config /home/jichengzhi/heal_research/HEAL/opencood/hypes_yaml/opv2v/LiDAROnly/lidar_fcooper.yaml \\
      --tag fcooper \\
      --warmup 20 --measure 100
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
sys.path.insert(0, HEAL_ROOT)
os.chdir(HEAL_ROOT)  # so relative dataset paths in yaml resolve

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.common_utils import update_dict

torch.multiprocessing.set_sharing_strategy("file_system")


class CudaTimer:
    def __init__(self, labels):
        self.labels = list(labels)
        self.events = {k: torch.cuda.Event(enable_timing=True) for k in self.labels}
        self.records = {k: [] for k in self.labels[1:]}

    def mark(self, label):
        self.events[label].record()

    def collect(self):
        torch.cuda.synchronize()
        prev = self.labels[0]
        for cur in self.labels[1:]:
            self.records[cur].append(self.events[prev].elapsed_time(self.events[cur]))
            prev = cur


def run_postproc_with_timing(post_processor, data_dict, output_dict, sub_timers):
    """Re-implements VoxelPostprocessor.post_process with sub-stage timing."""
    import torch.nn.functional as F
    from opencood.utils.box_utils import (
        boxes_to_corners_3d, project_box3d, corner_to_standup_box_torch,
        remove_large_pred_bbx, remove_bbx_abnormal_z,
        mask_boxes_outside_range_numpy, nms_rotated,
    )
    from opencood.utils.common_utils import limit_period

    params = post_processor.params
    e = {k: torch.cuda.Event(enable_timing=True) for k in
         ["s", "decode", "dir", "corners", "nms_pre", "nms_done", "mask"]}
    e["s"].record()

    pred_box3d_list = []
    pred_box2d_list = []
    cav_keys = list(output_dict.keys())

    for cav_id in cav_keys:
        cav_content = data_dict[cav_id]
        transformation_matrix = cav_content["transformation_matrix"]
        anchor_box = cav_content["anchor_box"]
        prob = output_dict[cav_id]["cls_preds"]
        prob = F.sigmoid(prob.permute(0, 2, 3, 1)).reshape(1, -1)
        reg = output_dict[cav_id]["reg_preds"]
        if len(reg.shape) == 4:
            batch_box3d = post_processor.delta_to_boxes3d(reg, anchor_box)
        else:
            batch_box3d = reg.view(1, -1, 7)

        mask = torch.gt(prob, params["target_args"]["score_threshold"])
        mask = mask.view(1, -1)
        mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
        boxes3d = torch.masked_select(batch_box3d[0], mask_reg[0]).view(-1, 7)
        scores = torch.masked_select(prob[0], mask[0])

        if cav_id == cav_keys[-1]:
            e["decode"].record()

        if "dir_preds" in output_dict[cav_id] and len(boxes3d) != 0:
            dir_offset = params["dir_args"]["dir_offset"]
            num_bins = params["dir_args"]["num_bins"]
            dm = output_dict[cav_id]["dir_preds"]
            dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
            dir_cls_preds = dir_cls_preds[mask]
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
            period = 2 * np.pi / num_bins
            dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)
            boxes3d[..., 6] = dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)
            boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)

        if cav_id == cav_keys[-1]:
            e["dir"].record()

        if len(boxes3d) != 0:
            boxes3d_corner = boxes_to_corners_3d(boxes3d, order=params["order"])
            projected_boxes3d = project_box3d(boxes3d_corner, transformation_matrix)
            projected_boxes2d = corner_to_standup_box_torch(projected_boxes3d)
            boxes2d_score = torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)
            pred_box2d_list.append(boxes2d_score)
            pred_box3d_list.append(projected_boxes3d)

    e["corners"].record()

    if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
        e["nms_pre"].record()
        e["nms_done"].record()
        e["mask"].record()
        torch.cuda.synchronize()
        for k, ev_pair in [("decode", ("s", "decode")), ("dir", ("decode", "dir")),
                           ("corners", ("dir", "corners")), ("nms", ("corners", "nms_done")),
                           ("range_mask", ("nms_done", "mask"))]:
            sub_timers[k].append(e[ev_pair[0]].elapsed_time(e[ev_pair[1]]))
        return None, None

    pred_box2d_list = torch.vstack(pred_box2d_list)
    scores = pred_box2d_list[:, -1]
    pred_box3d_tensor = torch.vstack(pred_box3d_list)
    keep_index_1 = remove_large_pred_bbx(pred_box3d_tensor)
    keep_index_2 = remove_bbx_abnormal_z(pred_box3d_tensor)
    keep_index = torch.logical_and(keep_index_1, keep_index_2)
    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    scores = scores[keep_index]

    e["nms_pre"].record()
    keep_index = nms_rotated(pred_box3d_tensor, scores, params["nms_thresh"])
    pred_box3d_tensor = pred_box3d_tensor[keep_index]
    scores = scores[keep_index]
    e["nms_done"].record()

    pred_box3d_np = pred_box3d_tensor.cpu().numpy()
    pred_box3d_np, mask_out = mask_boxes_outside_range_numpy(
        pred_box3d_np, params["gt_range"], order=None, return_mask=True)
    pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(device=pred_box3d_tensor.device)
    scores = scores[mask_out]
    e["mask"].record()
    torch.cuda.synchronize()

    sub_timers["decode"].append(e["s"].elapsed_time(e["decode"]))
    sub_timers["dir"].append(e["decode"].elapsed_time(e["dir"]))
    sub_timers["corners"].append(e["dir"].elapsed_time(e["corners"]))
    sub_timers["nms"].append(e["corners"].elapsed_time(e["nms_done"]))
    sub_timers["range_mask"].append(e["nms_done"].elapsed_time(e["mask"]))
    return pred_box3d_tensor, scores


def replay_baseline_forward(model, batch, post_processor, fwd_timer, sub_timers):
    """Replay HeterModelBaseline.forward with per-stage CUDA events."""
    import torchvision
    data_dict = batch["ego"]
    agent_modality_list = data_dict["agent_modality_list"]
    affine_matrix = normalize_pairwise_tfm(
        data_dict["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size)
    record_len = data_dict["record_len"]
    modality_count_dict = Counter(agent_modality_list)
    modality_feature_dict = {}

    fwd_timer.mark("start")

    # encoder
    for modality_name in model.modality_name_list:
        if modality_name not in modality_count_dict:
            continue
        feat_enc = getattr(model, f"encoder_{modality_name}")(data_dict, modality_name)
    fwd_timer.mark("encoder")

    # backbone (per-modality)
    feat_bb = getattr(model, f"backbone_{modality_name}")(
        {"spatial_features": feat_enc})["spatial_features_2d"]
    fwd_timer.mark("backbone")

    # shrinker_m1 (per-modality)
    feat_sh = getattr(model, f"shrinker_{modality_name}")(feat_bb)
    modality_feature_dict[modality_name] = feat_sh
    fwd_timer.mark("shrinker_m1")

    # Camera crop (skip for LiDAR-only)
    for mn in model.modality_name_list:
        if mn in modality_count_dict and model.sensor_type_dict[mn] == "camera":
            f = modality_feature_dict[mn]
            _, _, H, W = f.shape
            tH = int(H * getattr(model, f"crop_ratio_H_{mn}"))
            tW = int(W * getattr(model, f"crop_ratio_W_{mn}"))
            modality_feature_dict[mn] = torchvision.transforms.CenterCrop((tH, tW))(f)

    # Assemble heter features
    counting_dict = {m: 0 for m in model.modality_name_list}
    heter_feature_2d_list = []
    for mn in agent_modality_list:
        idx = counting_dict[mn]
        heter_feature_2d_list.append(modality_feature_dict[mn][idx])
        counting_dict[mn] += 1
    heter_feature_2d = torch.stack(heter_feature_2d_list)
    if model.compress:
        heter_feature_2d = model.compressor(heter_feature_2d)

    # Fusion (VARIABLE module - the part we care about)
    fused_feature = model.fusion_net(heter_feature_2d, record_len, affine_matrix)
    fwd_timer.mark("fusion_net")

    # Top-level shrink
    if model.shrink_flag:
        fused_feature = model.shrink_conv(fused_feature)
    fwd_timer.mark("shrink_conv")

    # Heads
    cls_preds = model.cls_head(fused_feature)
    reg_preds = model.reg_head(fused_feature)
    dir_preds = model.dir_head(fused_feature)
    fwd_timer.mark("heads")

    output_dict = {"ego": {"cls_preds": cls_preds, "reg_preds": reg_preds, "dir_preds": dir_preds}}

    pred_box, scores = run_postproc_with_timing(
        post_processor, batch, output_dict, sub_timers)
    fwd_timer.mark("postproc")
    fwd_timer.collect()
    return pred_box, scores


def percentile(arr, p):
    return float(np.percentile(arr, p))


def stats(arr):
    return {
        "n": len(arr),
        "mean_ms": float(np.mean(arr)) if arr else None,
        "p50_ms": percentile(arr, 50) if arr else None,
        "p99_ms": percentile(arr, 99) if arr else None,
        "min_ms": float(np.min(arr)) if arr else None,
        "max_ms": float(np.max(arr)) if arr else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to lidar_*.yaml")
    ap.add_argument("--tag", required=True, help="short name e.g. fcooper")
    ap.add_argument("--ckpt", default=None, help="optional pretrained ckpt dir")
    ap.add_argument("--test-dir",
                    default="/home/jichengzhi/heal_research/dataset/OPV2V_orig/extracted/test")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--score-threshold", type=float, default=None,
                    help="Override score_threshold in postproc. For random-init runs, "
                         "use 0.95 to filter most false-positives so NMS workload is realistic. "
                         "Real-weight runs use default (0.2 typically).")
    ap.add_argument("--opt", choices=["none", "p0"], default="none",
                    help="Optional CUDA-kernel optimization: p0 = replace Shapely "
                         "nms_rotated with mmcv CUDA nms_rotated (drop-in via "
                         "m4_8_cuda_kernel_replacements.nms_rotated_mmcv)")
    ap.add_argument("--out-dir",
                    default="/home/jichengzhi/UniV2X/data/v2x_baseline_timing")
    args = ap.parse_args()

    print(f"[boot] config: {args.config}", flush=True)
    print(f"[boot] tag:    {args.tag}", flush=True)
    print(f"[boot] ckpt:   {args.ckpt or '(random init)'}", flush=True)

    hypes = yaml_utils.load_yaml(args.config, None)

    # Override dataset path to absolute
    hypes["test_dir"] = args.test_dir
    hypes["validate_dir"] = args.test_dir
    hypes["root_dir"] = args.test_dir

    # Apply yaml parser
    parser_func = getattr(yaml_utils, hypes["yaml_parser"])
    hypes = parser_func(hypes)

    print("[boot] building model", flush=True)
    model = train_utils.create_model(hypes)
    if args.ckpt:
        _, model = train_utils.load_saved_model(args.ckpt, model)
        print(f"[boot] loaded ckpt from {args.ckpt}", flush=True)
    else:
        print(f"[boot] *** random init, AP measurements MEANINGLESS, latency only ***", flush=True)
    model.cuda().eval()

    # P0: monkey-patch nms_rotated to CUDA mmcv version
    if args.opt == "p0":
        sys.path.insert(0, "/home/jichengzhi/UniV2X/scripts/phase2")
        from m4_8_cuda_kernel_replacements import nms_rotated_mmcv
        from opencood.utils import box_utils as _bu
        _orig_nms = _bu.nms_rotated
        def _patched_nms(boxes_corners, scores, threshold):
            return nms_rotated_mmcv(boxes_corners, scores, threshold)
        _bu.nms_rotated = _patched_nms
        print(f"[boot] *** P0 applied: box_utils.nms_rotated → mmcv CUDA nms_rotated ***", flush=True)

    print("[boot] building dataset", flush=True)
    ds = build_dataset(hypes, visualize=False, train=False)

    if args.score_threshold is not None:
        orig_thr = ds.post_processor.params["target_args"]["score_threshold"]
        ds.post_processor.params["target_args"]["score_threshold"] = args.score_threshold
        print(f"[boot] *** score_threshold override: {orig_thr} -> {args.score_threshold} "
              f"(realistic-NMS-load hack for random init) ***", flush=True)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, num_workers=2,
                        collate_fn=ds.collate_batch_test,
                        shuffle=False, pin_memory=False)

    fwd_labels = ["start", "encoder", "backbone", "shrinker_m1",
                  "fusion_net", "shrink_conv", "heads", "postproc"]
    fwd_timer = CudaTimer(fwd_labels)
    sub_timers = {"decode": [], "dir": [], "corners": [], "nms": [], "range_mask": []}
    e2e_ms = []
    record_len_list = []

    post_processor = ds.post_processor
    total_iters = args.warmup + args.measure
    print(f"[boot] starting: warmup={args.warmup}, measure={args.measure}", flush=True)

    device = torch.device("cuda")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= total_iters:
                break
            batch = train_utils.to_device(batch, device)
            torch.cuda.synchronize()
            t0 = time.time()
            _ = replay_baseline_forward(model, batch, post_processor, fwd_timer, sub_timers)
            torch.cuda.synchronize()
            if i >= args.warmup:
                e2e_ms.append((time.time() - t0) * 1000)
                record_len_list.append(int(batch["ego"]["record_len"][0]))
            if i + 1 == args.warmup:
                fwd_timer.records = {k: [] for k in fwd_timer.records}
                sub_timers = {k: [] for k in sub_timers}
                print(f"[warmup-done] records reset", flush=True)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total_iters}]", flush=True)

    report = {
        "tag": args.tag,
        "config": args.config,
        "ckpt": args.ckpt,
        "random_init": args.ckpt is None,
        "warmup": args.warmup,
        "measure": args.measure,
        "n_collected": len(e2e_ms),
        "record_len_mean": float(np.mean(record_len_list)) if record_len_list else None,
        "e2e_walltime": stats(e2e_ms),
        "forward_stages_cuda_event": {k: stats(v) for k, v in fwd_timer.records.items()},
        "postproc_substages_cuda_event": {k: stats(v) for k, v in sub_timers.items()},
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / f"{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {out_path}", flush=True)

    print(f"\n=== {args.tag} per-stage summary (mean ms / p50 / p99) ===")
    for k, v in fwd_timer.records.items():
        s = stats(v)
        if s["mean_ms"] is not None:
            print(f"  {k:14s}  mean={s['mean_ms']:7.3f}  p50={s['p50_ms']:7.3f}  p99={s['p99_ms']:7.3f}")
    print("--- postproc substages ---")
    for k, v in sub_timers.items():
        s = stats(v)
        if s["mean_ms"] is not None:
            print(f"  {k:14s}  mean={s['mean_ms']:7.3f}  p50={s['p50_ms']:7.3f}  p99={s['p99_ms']:7.3f}")
    if e2e_ms:
        print(f"  e2e walltime    mean={np.mean(e2e_ms):7.3f}  p50={np.percentile(e2e_ms, 50):7.3f}")
        print(f"  record_len_mean = {np.mean(record_len_list):.2f} agents/scene")


if __name__ == "__main__":
    main()
