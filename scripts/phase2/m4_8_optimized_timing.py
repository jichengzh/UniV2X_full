"""M4.8 Optimized per-stage timing.

Runs Pyramid Fusion with optional CUDA kernel replacements:
  --opt none           : baseline (Shapely NMS + QuickCumsum)
  --opt p0             : CUDA NMS only (mmcv.ops.nms_rotated)
  --opt p0prime        : index_add scatter only (P0' for camera modality)
  --opt p0_both        : both
  --modality m1 | m2

Reports per-stage timing and lets us tabulate before/after for §八 of audit doc.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch

HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
sys.path.insert(0, HEAL_ROOT)
sys.path.insert(0, "/home/jichengzhi/UniV2X/scripts/phase2")
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.common_utils import update_dict
from opencood.utils import box_utils, camera_utils

from m4_8_cuda_kernel_replacements import nms_rotated_mmcv, quickcumsum_native

torch.multiprocessing.set_sharing_strategy("file_system")


# ─────────────────────────── Stage records ───────────────────────────


def stats(arr):
    if not arr:
        return {"n": 0, "mean": None, "p50": None, "p99": None}
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ─────────────────────────── Apply optimizations ───────────────────────────


def install_optimizations(opt: str, encoder_m2_or_none):
    """Monkey-patch HEAL utilities according to --opt."""
    # Save originals
    orig_nms = box_utils.nms_rotated
    orig_quickcumsum = camera_utils.QuickCumsum

    if opt in ("p0", "p0_both"):
        box_utils.nms_rotated = nms_rotated_mmcv
        # Need to also patch the import in voxel_postprocessor (it did `from ... import nms_rotated`)
        import opencood.data_utils.post_processor.voxel_postprocessor as vp_mod
        vp_mod.box_utils.nms_rotated = nms_rotated_mmcv
        print(f"  [opt] P0 enabled: nms_rotated → mmcv.ops.nms_rotated", flush=True)

    if opt in ("p0prime", "p0_both") and encoder_m2_or_none is not None:
        # Patch the encoder's voxel_pooling to use quickcumsum_native
        enc = encoder_m2_or_none
        orig_voxel_pool = enc.voxel_pooling

        def patched_vp(self, geom_feats, x):
            # Replicate original logic but swap QuickCumsum.apply → quickcumsum_native
            B, N, D, H, W, C = x.shape
            Nprime = B * N * D * H * W
            x = x.reshape(Nprime, C)
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                             device=x.device, dtype=torch.long)
                                  for ix in range(B)])
            geom_feats = torch.cat((geom_feats, batch_ix), 1)
            kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
                & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
                & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
            x = x[kept]
            geom_feats = geom_feats[kept]
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

            # ───── P0' replacement ─────
            x, geom_feats = quickcumsum_native(x, geom_feats, ranks)

            final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]),
                                device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1],
                  geom_feats[:, 0]] = x
            final = torch.cat(final.unbind(dim=2), 1)
            return final

        enc.voxel_pooling = patched_vp.__get__(enc, type(enc))
        print(f"  [opt] P0' enabled: QuickCumsum.apply → quickcumsum_native", flush=True)


# ─────────────────────────── Forward replay with timing ───────────────────────────


def time_forward_collab(model, batch, post_processor):
    """m1 collab forward + postproc, returns dict of stage_name -> ms."""
    e = {k: torch.cuda.Event(enable_timing=True) for k in
         ["s", "enc", "back", "align", "pyr", "shrink", "heads", "pp_pre", "pp_nms", "pp_done"]}

    data_dict = batch["ego"]
    agent_modality_list = data_dict["agent_modality_list"]
    affine_matrix = normalize_pairwise_tfm(
        data_dict["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size
    )
    record_len = data_dict["record_len"]
    modality_count_dict = Counter(agent_modality_list)
    modality_feature_dict = {}

    e["s"].record()
    for modality_name in model.modality_name_list:
        if modality_name not in modality_count_dict:
            continue
        feature = getattr(model, f"encoder_{modality_name}")(data_dict, modality_name)
    e["enc"].record()
    feature = getattr(model, f"backbone_{modality_name}")(
        {"spatial_features": feature}
    )["spatial_features_2d"]
    e["back"].record()
    feature = getattr(model, f"aligner_{modality_name}")(feature)
    modality_feature_dict[modality_name] = feature
    e["align"].record()

    counting_dict = {m: 0 for m in model.modality_name_list}
    heter_feature_2d_list = []
    for modality_name in agent_modality_list:
        feat_idx = counting_dict[modality_name]
        heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
        counting_dict[modality_name] += 1
    heter_feature_2d = torch.stack(heter_feature_2d_list)
    fused_feature, occ_outputs = model.pyramid_backbone.forward_collab(
        heter_feature_2d, record_len, affine_matrix, agent_modality_list,
        model.cam_crop_info,
    )
    e["pyr"].record()
    if model.shrink_flag:
        fused_feature = model.shrink_conv(fused_feature)
    e["shrink"].record()
    cls_preds = model.cls_head(fused_feature)
    reg_preds = model.reg_head(fused_feature)
    dir_preds = model.dir_head(fused_feature)
    e["heads"].record()

    # Run full post_process (decode+dir+corners+NMS+range_mask) but split NMS out
    # Easiest: call the post_processor.post_process and time the whole thing
    output_dict = {"ego": {
        "cls_preds": cls_preds,
        "reg_preds": reg_preds,
        "dir_preds": dir_preds,
    }}
    e["pp_pre"].record()
    pred_box, scores = post_processor.post_process(batch, output_dict)
    e["pp_done"].record()
    torch.cuda.synchronize()

    return {
        "encoder": e["s"].elapsed_time(e["enc"]),
        "backbone": e["enc"].elapsed_time(e["back"]),
        "aligner": e["back"].elapsed_time(e["align"]),
        "pyramid": e["align"].elapsed_time(e["pyr"]),
        "shrink": e["pyr"].elapsed_time(e["shrink"]),
        "heads": e["shrink"].elapsed_time(e["heads"]),
        "postproc": e["pp_pre"].elapsed_time(e["pp_done"]),
    }


def time_forward_single(model, batch, post_processor):
    """m2 single forward (no collab) + postproc."""
    import torchvision
    e = {k: torch.cuda.Event(enable_timing=True) for k in
         ["s", "enc", "back", "align", "crop", "pyr", "shrink", "heads", "pp_pre", "pp_done"]}

    data_dict = batch["ego"]
    modality_inputs = [k for k in data_dict.keys() if k.startswith("inputs_")]
    modality_name = modality_inputs[0].replace("inputs_", "")

    e["s"].record()
    feature = getattr(model, f"encoder_{modality_name}")(data_dict, modality_name)
    e["enc"].record()
    feature = getattr(model, f"backbone_{modality_name}")(
        {"spatial_features": feature}
    )["spatial_features_2d"]
    e["back"].record()
    feature = getattr(model, f"aligner_{modality_name}")(feature)
    e["align"].record()

    if model.sensor_type_dict[modality_name] == "camera":
        _, _, H_, W_ = feature.shape
        ch = eval(f"model.crop_ratio_H_{modality_name}")
        cw = eval(f"model.crop_ratio_W_{modality_name}")
        feature = torchvision.transforms.CenterCrop(
            (int(H_ * ch), int(W_ * cw))
        )(feature)
    e["crop"].record()

    feature, occ_map_list = model.pyramid_backbone.forward_single(feature)
    e["pyr"].record()
    if model.shrink_flag:
        feature = model.shrink_conv(feature)
    e["shrink"].record()
    cls_preds = model.cls_head(feature)
    reg_preds = model.reg_head(feature)
    dir_preds = model.dir_head(feature)
    e["heads"].record()

    output_dict = {"ego": {
        "cls_preds": cls_preds,
        "reg_preds": reg_preds,
        "dir_preds": dir_preds,
    }}
    e["pp_pre"].record()
    pred_box, scores = post_processor.post_process(batch, output_dict)
    e["pp_done"].record()
    torch.cuda.synchronize()

    return {
        "encoder": e["s"].elapsed_time(e["enc"]),
        "backbone": e["enc"].elapsed_time(e["back"]),
        "aligner": e["back"].elapsed_time(e["align"]),
        "pyramid": e["crop"].elapsed_time(e["pyr"]),
        "shrink": e["pyr"].elapsed_time(e["shrink"]),
        "heads": e["shrink"].elapsed_time(e["heads"]),
        "postproc": e["pp_pre"].elapsed_time(e["pp_done"]),
    }


# ─────────────────────────── Main ───────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["m1", "m2"], required=True)
    ap.add_argument("--opt", choices=["none", "p0", "p0prime", "p0_both"], default="none")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.modality == "m1":
        ckpt = "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12"
    else:
        ckpt = "/home/jichengzhi/heal_research/checkpoints/stage2/m2_alignto_m1"
    test_dir = "/home/jichengzhi/heal_research/dataset/OPV2V_orig/extracted/test"

    class _Opt:
        model_dir = ckpt
    opt_obj = _Opt()
    hypes = yaml_utils.load_yaml(None, opt_obj)
    x_min, x_max = -102.4, 102.4
    new_cav_range = [
        x_min, x_min, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
        x_max, x_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
    ]
    hypes = update_dict(hypes, {
        "cav_lidar_range": new_cav_range,
        "lidar_range": new_cav_range,
        "gt_range": new_cav_range,
    })

    # m2: disable depth GT loading
    if args.modality == "m2":
        if "input_source" in hypes:
            hypes["input_source"] = [s for s in hypes["input_source"] if s != "depth"]
        if hypes.get("label_type") == "camera":
            hypes["label_type"] = "lidar"
        model_args = hypes.get("model", {}).get("args", {})
        for m in ["m1", "m2", "m3", "m4"]:
            if m in model_args and "encoder_args" in model_args[m]:
                ea = model_args[m]["encoder_args"]
                if "depth_supervision" in ea:
                    ea["depth_supervision"] = False
                if "use_depth_gt" in ea:
                    ea["use_depth_gt"] = False
        if "heter" in hypes and "assignment_path" in hypes["heter"]:
            ap_path = hypes["heter"]["assignment_path"]
            if not os.path.isabs(ap_path):
                hypes["heter"]["assignment_path"] = os.path.join(HEAL_ROOT, ap_path)

    hypes["test_dir"] = test_dir
    hypes["validate_dir"] = test_dir
    hypes["root_dir"] = test_dir

    # re-run yaml parser (range changed)
    import importlib
    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = getattr(yaml_utils_lib, hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["test_dir"] = test_dir
    hypes["validate_dir"] = test_dir

    print(f"[boot] modality={args.modality}, opt={args.opt}", flush=True)
    model = train_utils.create_model(hypes)
    device = torch.device("cuda")
    _, model = train_utils.load_saved_model(ckpt, model)
    model.cuda().eval()

    # camera modality: disable depth_supervision on instance
    encoder_m2 = None
    if args.modality == "m2":
        encoder_m2 = getattr(model, "encoder_m2")
        encoder_m2.depth_supervision = False
        encoder_m2.camencode.depth_supervision = False
        encoder_m2.camencode.use_gt_depth = False

    install_optimizations(args.opt, encoder_m2)

    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=1, num_workers=2,
        collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False,
    )
    post_processor = ds.post_processor

    total = args.warmup + args.measure
    print(f"[boot] starting: warmup={args.warmup}, measure={args.measure}", flush=True)
    records = {k: [] for k in
               ["encoder", "backbone", "aligner", "pyramid", "shrink", "heads", "postproc"]}
    e2e_walltime = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= total:
                break
            batch = train_utils.to_device(batch, device)
            torch.cuda.synchronize()
            t0 = time.time()
            if args.modality == "m1":
                stage_ms = time_forward_collab(model, batch, post_processor)
            else:
                stage_ms = time_forward_single(model, batch, post_processor)
            torch.cuda.synchronize()
            wall = (time.time() - t0) * 1000
            if i >= args.warmup:
                for k, v in stage_ms.items():
                    records[k].append(v)
                e2e_walltime.append(wall)
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total}]", flush=True)

    report = {
        "modality": args.modality,
        "opt": args.opt,
        "ckpt": ckpt,
        "warmup": args.warmup,
        "measure": args.measure,
        "n_collected": len(e2e_walltime),
        "stages_ms": {k: stats(v) for k, v in records.items()},
        "e2e_walltime_ms": stats(e2e_walltime),
        "device": torch.cuda.get_device_name(0),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {args.out}", flush=True)

    print("\n=== summary (mean ms) ===")
    for k, v in records.items():
        s = stats(v)
        print(f"  {k:10s}  mean={s['mean']:7.3f}  p50={s['p50']:7.3f}  p99={s['p99']:7.3f}")
    s_e2e = stats(e2e_walltime)
    print(f"  e2e wall  mean={s_e2e['mean']:7.3f}  p50={s_e2e['p50']:7.3f}")


if __name__ == "__main__":
    main()
