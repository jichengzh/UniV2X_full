"""M4.8 per-stage timing — Pyramid Fusion forward + postproc decomposed.

What it does:
  Loads Pyramid_m1_base baseline, runs N samples from OPV2V test set,
  measures *each* sub-module with CUDA Event timing (sync'd), reports
  mean/p50/p99 for:
    encoder_m1 / backbone_m1 / aligner_m1
    pyramid_backbone.forward_collab  (multi-scale + warp + weighted_fuse + decode)
    shrink_conv / heads(cls+reg+dir)
    postproc: decode (delta_to_boxes3d) / project_box / NMS (nms_rotated) / mask

The purpose: verify whether the previously-claimed "NMS 60% of e2e" holds, and
expose the real per-stage breakdown for the audit doc.

We deliberately do NOT modify HEAL source; we re-call HeterPyramidCollab's
modules in their original order while inserting Events between them.

Important caveat — encoder/backbone/aligner are run on the *batched tensor of all
agents at once* (ego + collaborators), so the reported number is total work,
not per-agent. Per-agent ≈ total / record_len[0] for the typical OPV2V scene
(2 cavs).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# put HEAL on sys.path
HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
sys.path.insert(0, HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.common_utils import update_dict
from opencood.utils import box_utils

torch.multiprocessing.set_sharing_strategy("file_system")


# ─────────────────────────── Event timer helper ───────────────────────────


class CudaTimer:
    """Records a sequence of CUDA Events and reports inter-event ms."""

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
            ms = self.events[prev].elapsed_time(self.events[cur])
            self.records[cur].append(ms)
            prev = cur


# ─────────────────────────── Patched postproc ───────────────────────────


def run_postproc_with_timing(post_processor, data_dict, output_dict, sub_timers):
    """Re-implements VoxelPostprocessor.post_process with sub-stage timing.

    Sub-stages: decode_reg, dir_classifier, corners+project, remove_large+nms, range_mask
    """
    import torch.nn.functional as F
    from opencood.utils.box_utils import (
        boxes_to_corners_3d,
        project_box3d,
        corner_to_standup_box_torch,
        remove_large_pred_bbx,
        remove_bbx_abnormal_z,
        mask_boxes_outside_range_numpy,
        nms_rotated,
    )
    from opencood.utils.common_utils import limit_period

    params = post_processor.params
    anchor_num = post_processor.anchor_num

    # CUDA events for sub-stages
    e = {
        k: torch.cuda.Event(enable_timing=True)
        for k in ["s", "decode", "dir", "corners", "nms_pre", "nms_done", "mask"]
    }

    e["s"].record()

    pred_box3d_list = []
    pred_box2d_list = []

    for cav_id in output_dict.keys():
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

        # mark decode end (incl. mask)
        if cav_id == list(output_dict.keys())[-1]:
            e["decode"].record()

        # dir classifier
        if "dir_preds" in output_dict[cav_id] and len(boxes3d) != 0:
            dir_offset = params["dir_args"]["dir_offset"]
            num_bins = params["dir_args"]["num_bins"]
            dm = output_dict[cav_id]["dir_preds"]
            dir_cls_preds = dm.permute(0, 2, 3, 1).contiguous().reshape(1, -1, num_bins)
            dir_cls_preds = dir_cls_preds[mask]
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
            period = 2 * np.pi / num_bins
            dir_rot = limit_period(boxes3d[..., 6] - dir_offset, 0, period)
            boxes3d[..., 6] = (
                dir_rot + dir_offset + period * dir_labels.to(dir_cls_preds.dtype)
            )
            boxes3d[..., 6] = limit_period(boxes3d[..., 6], 0.5, 2 * np.pi)

        if cav_id == list(output_dict.keys())[-1]:
            e["dir"].record()

        if len(boxes3d) != 0:
            boxes3d_corner = boxes_to_corners_3d(boxes3d, order=params["order"])
            projected_boxes3d = project_box3d(boxes3d_corner, transformation_matrix)
            projected_boxes2d = corner_to_standup_box_torch(projected_boxes3d)
            boxes2d_score = torch.cat(
                (projected_boxes2d, scores.unsqueeze(1)), dim=1
            )
            pred_box2d_list.append(boxes2d_score)
            pred_box3d_list.append(projected_boxes3d)

    e["corners"].record()

    if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
        e["nms_pre"].record()
        e["nms_done"].record()
        e["mask"].record()
        torch.cuda.synchronize()
        for k, ev_pair in [
            ("decode", ("s", "decode")),
            ("dir", ("decode", "dir")),
            ("corners", ("dir", "corners")),
            ("nms", ("corners", "nms_done")),
            ("range_mask", ("nms_done", "mask")),
        ]:
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
        pred_box3d_np, params["gt_range"], order=None, return_mask=True
    )
    pred_box3d_tensor = torch.from_numpy(pred_box3d_np).to(
        device=pred_box3d_tensor.device
    )
    scores = scores[mask_out]

    e["mask"].record()
    torch.cuda.synchronize()

    sub_timers["decode"].append(e["s"].elapsed_time(e["decode"]))
    sub_timers["dir"].append(e["decode"].elapsed_time(e["dir"]))
    sub_timers["corners"].append(e["dir"].elapsed_time(e["corners"]))
    sub_timers["nms"].append(e["corners"].elapsed_time(e["nms_done"]))
    sub_timers["range_mask"].append(e["nms_done"].elapsed_time(e["mask"]))

    return pred_box3d_tensor, scores


# ─────────────────────────── Main forward replay ───────────────────────────


def replay_forward_timed(model, batch, post_processor, fwd_timer, sub_timers):
    """Re-runs HeterPyramidCollab.forward() with per-stage events."""
    data_dict = batch["ego"]
    agent_modality_list = data_dict["agent_modality_list"]
    affine_matrix = normalize_pairwise_tfm(
        data_dict["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size
    )
    record_len = data_dict["record_len"]
    from collections import Counter

    modality_count_dict = Counter(agent_modality_list)
    modality_feature_dict = {}

    fwd_timer.mark("start")

    for modality_name in model.modality_name_list:
        if modality_name not in modality_count_dict:
            continue
        feature = getattr(model, f"encoder_{modality_name}")(data_dict, modality_name)
    fwd_timer.mark("encoder")

    feature = getattr(model, f"backbone_{modality_name}")(
        {"spatial_features": feature}
    )["spatial_features_2d"]
    fwd_timer.mark("backbone")

    feature = getattr(model, f"aligner_{modality_name}")(feature)
    modality_feature_dict[modality_name] = feature
    fwd_timer.mark("aligner")

    # Assemble heter features
    counting_dict = {m: 0 for m in model.modality_name_list}
    heter_feature_2d_list = []
    for modality_name in agent_modality_list:
        feat_idx = counting_dict[modality_name]
        heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
        counting_dict[modality_name] += 1
    heter_feature_2d = torch.stack(heter_feature_2d_list)

    if model.compress:
        heter_feature_2d = model.compressor(heter_feature_2d)

    fused_feature, occ_outputs = model.pyramid_backbone.forward_collab(
        heter_feature_2d,
        record_len,
        affine_matrix,
        agent_modality_list,
        model.cam_crop_info,
    )
    fwd_timer.mark("pyramid")

    if model.shrink_flag:
        fused_feature = model.shrink_conv(fused_feature)
    fwd_timer.mark("shrink")

    cls_preds = model.cls_head(fused_feature)
    reg_preds = model.reg_head(fused_feature)
    dir_preds = model.dir_head(fused_feature)
    fwd_timer.mark("heads")

    output_dict = {
        "ego": {
            "cls_preds": cls_preds,
            "reg_preds": reg_preds,
            "dir_preds": dir_preds,
        }
    }

    pred_box, scores = run_postproc_with_timing(
        post_processor, batch, output_dict, sub_timers
    )
    fwd_timer.mark("postproc")
    fwd_timer.collect()
    return pred_box, scores


# ─────────────────────────── Main ───────────────────────────


def percentile(arr, p):
    return float(np.percentile(arr, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument(
        "--out",
        default="/home/jichengzhi/UniV2X/data/pyramid_per_stage_timing.json",
    )
    args = ap.parse_args()

    # Load yaml & override range to match published eval (102.4 m)
    class _Opt:
        model_dir = args.ckpt

    opt = _Opt()
    hypes = yaml_utils.load_yaml(None, opt)
    x_min, x_max = -102.4, 102.4
    new_cav_range = [
        x_min,
        x_min,
        hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
        x_max,
        x_max,
        hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
    ]
    hypes = update_dict(
        hypes,
        {
            "cav_lidar_range": new_cav_range,
            "lidar_range": new_cav_range,
            "gt_range": new_cav_range,
        },
    )
    import importlib

    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = getattr(yaml_utils_lib, hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    print("[boot] building model", flush=True)
    model = train_utils.create_model(hypes)
    device = torch.device("cuda")
    _, model = train_utils.load_saved_model(args.ckpt, model)
    model.cuda().eval()

    print("[boot] building dataset", flush=True)
    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=2,
        collate_fn=ds.collate_batch_test,
        shuffle=False,
        pin_memory=False,
    )

    fwd_labels = ["start", "encoder", "backbone", "aligner", "pyramid", "shrink", "heads", "postproc"]
    fwd_timer = CudaTimer(fwd_labels)
    sub_timers = {"decode": [], "dir": [], "corners": [], "nms": [], "range_mask": []}
    e2e_ms = []
    record_len_list = []

    post_processor = ds.post_processor
    total_iters = args.warmup + args.measure
    print(
        f"[boot] starting: warmup={args.warmup}, measure={args.measure}", flush=True
    )

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= total_iters:
                break
            batch = train_utils.to_device(batch, device)
            torch.cuda.synchronize()
            t0 = time.time()
            if i < args.warmup:
                # warmup: just call replay forward, drop records appended this round
                # (we reset records dict after warmup)
                _ = replay_forward_timed(model, batch, post_processor, fwd_timer, sub_timers)
            else:
                _ = replay_forward_timed(model, batch, post_processor, fwd_timer, sub_timers)
                torch.cuda.synchronize()
                e2e_ms.append((time.time() - t0) * 1000)
                record_len_list.append(int(batch["ego"]["record_len"][0]))
            if i + 1 == args.warmup:
                fwd_timer.records = {k: [] for k in fwd_timer.records}
                sub_timers = {k: [] for k in sub_timers}
                post_processor_sub = sub_timers  # alias
                print(f"[warmup-done] reset records", flush=True)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total_iters}]", flush=True)

    def stats(arr):
        return {
            "n": len(arr),
            "mean_ms": float(np.mean(arr)) if arr else None,
            "p50_ms": percentile(arr, 50) if arr else None,
            "p99_ms": percentile(arr, 99) if arr else None,
            "min_ms": float(np.min(arr)) if arr else None,
            "max_ms": float(np.max(arr)) if arr else None,
        }

    report = {
        "ckpt": args.ckpt,
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {args.out}", flush=True)

    # Pretty print summary
    print("\n=== summary (mean ms / p50 / p99) ===")
    for k, v in fwd_timer.records.items():
        s = stats(v)
        print(f"  {k:12s}  mean={s['mean_ms']:7.3f}  p50={s['p50_ms']:7.3f}  p99={s['p99_ms']:7.3f}")
    print("--- postproc substages ---")
    for k, v in sub_timers.items():
        s = stats(v)
        print(f"  {k:12s}  mean={s['mean_ms']:7.3f}  p50={s['p50_ms']:7.3f}  p99={s['p99_ms']:7.3f}")
    print(f"  e2e walltime  mean={np.mean(e2e_ms):7.3f}  p50={np.percentile(e2e_ms, 50):7.3f}")
    print(f"  record_len mean = {np.mean(record_len_list):.2f} agents/scene")


if __name__ == "__main__":
    main()
