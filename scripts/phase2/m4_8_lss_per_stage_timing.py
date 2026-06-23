"""M4.8 LSS per-stage timing — camera modality Pyramid encoder (m2_alignto_m1 ckpt).

Profiles HEAL LiftSplatShoot encoder's sub-stages:
  CamEncode (EfficientNet img → depth-weighted features)
  get_geometry (pixel → 3D coord via intrins/rots/trans)
  voxel_pooling internals:
    flatten + quantize geom
    range filter
    rank + argsort
    QuickCumsum.apply
    scatter to BEV grid + collapse Z

Plus the rest of the single-modality forward (backbone_m1 / aligner_m1 /
pyramid_backbone.forward_single / shrink / heads).

NMS not profiled here — already known to be 70ms+ (Shapely python loop).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
sys.path.insert(0, HEAL_ROOT)
os.chdir(HEAL_ROOT)  # so "dataset/OPV2V/test" relative paths can resolve via override

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.common_utils import update_dict
from opencood.utils.camera_utils import cumsum_trick, QuickCumsum

torch.multiprocessing.set_sharing_strategy("file_system")


# Storage of LSS sub-stage timings
LSS_STAGE_RECORDS = {
    "cam_encode": [],
    "get_geometry": [],
    "vp_flatten_quantize": [],
    "vp_range_filter": [],
    "vp_rank_argsort": [],
    "vp_quickcumsum": [],
    "vp_scatter_collapse": [],
    "voxel_pool_total": [],
    "Nprime": [],
    "Nkept": [],
}


def patched_voxel_pooling(self, geom_feats, x):
    """Drop-in replacement for LiftSplatShoot.voxel_pooling with CUDA Event timing."""
    e = {k: torch.cuda.Event(enable_timing=True) for k in
         ["s", "flat", "filt", "sort", "cumsum", "scatter"]}

    e["s"].record()
    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W
    x = x.reshape(Nprime, C)
    geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
    geom_feats = geom_feats.view(Nprime, 3)
    batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                     device=x.device, dtype=torch.long)
                          for ix in range(B)])
    geom_feats = torch.cat((geom_feats, batch_ix), 1)
    e["flat"].record()

    kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
        & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
        & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
    x = x[kept]
    geom_feats = geom_feats[kept]
    e["filt"].record()
    Nkept = x.shape[0]

    ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
        + geom_feats[:, 1] * (self.nx[2] * B) \
        + geom_feats[:, 2] * B \
        + geom_feats[:, 3]
    sorts = ranks.argsort()
    x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
    e["sort"].record()

    if not self.use_quickcumsum:
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)
    else:
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
    e["cumsum"].record()

    final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]),
                        device=x.device)
    final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1],
          geom_feats[:, 0]] = x
    final = torch.cat(final.unbind(dim=2), 1)
    e["scatter"].record()

    torch.cuda.synchronize()
    LSS_STAGE_RECORDS["vp_flatten_quantize"].append(e["s"].elapsed_time(e["flat"]))
    LSS_STAGE_RECORDS["vp_range_filter"].append(e["flat"].elapsed_time(e["filt"]))
    LSS_STAGE_RECORDS["vp_rank_argsort"].append(e["filt"].elapsed_time(e["sort"]))
    LSS_STAGE_RECORDS["vp_quickcumsum"].append(e["sort"].elapsed_time(e["cumsum"]))
    LSS_STAGE_RECORDS["vp_scatter_collapse"].append(
        e["cumsum"].elapsed_time(e["scatter"])
    )
    LSS_STAGE_RECORDS["voxel_pool_total"].append(
        e["s"].elapsed_time(e["scatter"])
    )
    LSS_STAGE_RECORDS["Nprime"].append(Nprime)
    LSS_STAGE_RECORDS["Nkept"].append(Nkept)
    return final


def patched_get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
    e_geo_s = torch.cuda.Event(enable_timing=True)
    e_geo_e = torch.cuda.Event(enable_timing=True)
    e_cam_s = torch.cuda.Event(enable_timing=True)
    e_cam_e = torch.cuda.Event(enable_timing=True)

    e_geo_s.record()
    geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
    e_geo_e.record()

    e_cam_s.record()
    x_img, depth_items = self.get_cam_feats(x)
    e_cam_e.record()

    out = self.voxel_pooling(geom, x_img)

    torch.cuda.synchronize()
    LSS_STAGE_RECORDS["get_geometry"].append(e_geo_s.elapsed_time(e_geo_e))
    LSS_STAGE_RECORDS["cam_encode"].append(e_cam_s.elapsed_time(e_cam_e))
    return out, depth_items


def percentile(arr, p):
    return float(np.percentile(arr, p)) if arr else None


def stats(arr):
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)) if arr else None,
        "p50": percentile(arr, 50),
        "p99": percentile(arr, 99),
        "min": float(np.min(arr)) if arr else None,
        "max": float(np.max(arr)) if arr else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="/home/jichengzhi/heal_research/checkpoints/stage2/m2_alignto_m1",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--measure", type=int, default=60)
    ap.add_argument(
        "--out",
        default="/home/jichengzhi/UniV2X/data/pyramid_lss_per_stage_timing.json",
    )
    ap.add_argument(
        "--test_dir",
        default="/home/jichengzhi/heal_research/dataset/OPV2V_orig/extracted/test",
    )
    args = ap.parse_args()

    class _Opt:
        model_dir = args.ckpt

    opt = _Opt()
    hypes = yaml_utils.load_yaml(None, opt)

    # Override dataset path to absolute (yaml has relative "dataset/OPV2V/test")
    hypes["test_dir"] = args.test_dir
    hypes["validate_dir"] = args.test_dir
    hypes["root_dir"] = args.test_dir

    # disable depth GT loading (OPV2V_Hetero depth pngs not on disk)
    if "input_source" in hypes:
        hypes["input_source"] = [s for s in hypes["input_source"] if s != "depth"]
    # use lidar-style label generation (avoids missing bev_visibility.png)
    if hypes.get("label_type") == "camera":
        hypes["label_type"] = "lidar"
    # also turn off depth_supervision on m2 encoder to avoid asking dataset for depth
    model_args = hypes.get("model", {}).get("args", {})
    for m in ["m1", "m2", "m3", "m4"]:
        if m in model_args and "encoder_args" in model_args[m]:
            ea = model_args[m]["encoder_args"]
            if "depth_supervision" in ea:
                ea["depth_supervision"] = False
            if "use_depth_gt" in ea:
                ea["use_depth_gt"] = False

    # ensure heter assignment file path absolute
    if "heter" in hypes and "assignment_path" in hypes["heter"]:
        ap_path = hypes["heter"]["assignment_path"]
        if not os.path.isabs(ap_path):
            hypes["heter"]["assignment_path"] = os.path.join(HEAL_ROOT, ap_path)

    print("[boot] building model", flush=True)
    model = train_utils.create_model(hypes)
    device = torch.device("cuda")
    _, model = train_utils.load_saved_model(args.ckpt, model)
    model.cuda().eval()

    # Monkey-patch the LSS encoder
    encoder_m2 = getattr(model, "encoder_m2")
    encoder_m2.depth_supervision = False
    encoder_m2.camencode.depth_supervision = False
    encoder_m2.camencode.use_gt_depth = False
    encoder_m2.voxel_pooling = patched_voxel_pooling.__get__(
        encoder_m2, type(encoder_m2)
    )
    encoder_m2.get_voxels = patched_get_voxels.__get__(
        encoder_m2, type(encoder_m2)
    )
    print(f"[boot] patched LSS encoder: type={type(encoder_m2).__name__}", flush=True)
    print(
        f"[boot] LSS config: cams=4 (Ncams), final_dim={encoder_m2.data_aug_conf['final_dim']}, "
        f"downsample={encoder_m2.downsample}, D(depth bins)={encoder_m2.D}, "
        f"camC={encoder_m2.camC}, nx={encoder_m2.nx.tolist()}",
        flush=True,
    )

    print("[boot] building dataset", flush=True)
    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds, batch_size=1, num_workers=2,
        collate_fn=ds.collate_batch_test, shuffle=False,
        pin_memory=False,
    )

    # Outer-stage timings (backbone_m2 / aligner / pyramid_backbone forward_single / shrink / heads)
    outer_records = {k: [] for k in ["backbone", "aligner", "pyramid_single", "shrink", "heads"]}
    e2e_walltime = []

    total = args.warmup + args.measure
    print(f"[boot] starting warmup={args.warmup}, measure={args.measure}", flush=True)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if i >= total:
                break
            batch = train_utils.to_device(batch, device)

            # data_dict for heter_pyramid_single: only ego
            data_dict = batch["ego"]
            # find modality name
            modality_inputs = [k for k in data_dict.keys() if k.startswith("inputs_")]
            if not modality_inputs:
                continue
            modality_name = modality_inputs[0].replace("inputs_", "")

            torch.cuda.synchronize()
            t0 = time.time()

            # encoder forward (this triggers all LSS sub-stage recordings)
            feature = getattr(model, f"encoder_{modality_name}")(data_dict, modality_name)

            e_back_s = torch.cuda.Event(enable_timing=True)
            e_back_e = torch.cuda.Event(enable_timing=True)
            e_align_e = torch.cuda.Event(enable_timing=True)
            e_pyr_e = torch.cuda.Event(enable_timing=True)
            e_shrink_e = torch.cuda.Event(enable_timing=True)
            e_heads_e = torch.cuda.Event(enable_timing=True)

            e_back_s.record()
            feature = getattr(model, f"backbone_{modality_name}")(
                {"spatial_features": feature}
            )["spatial_features_2d"]
            e_back_e.record()
            feature = getattr(model, f"aligner_{modality_name}")(feature)
            e_align_e.record()

            # camera path: center-crop to BEV crop ratio
            import torchvision
            if model.sensor_type_dict[modality_name] == "camera":
                _, _, H_, W_ = feature.shape
                ch = eval(f"model.crop_ratio_H_{modality_name}")
                cw = eval(f"model.crop_ratio_W_{modality_name}")
                feature = torchvision.transforms.CenterCrop(
                    (int(H_ * ch), int(W_ * cw))
                )(feature)

            feature, occ_map_list = model.pyramid_backbone.forward_single(feature)
            e_pyr_e.record()
            if model.shrink_flag:
                feature = model.shrink_conv(feature)
            e_shrink_e.record()
            _ = model.cls_head(feature)
            _ = model.reg_head(feature)
            _ = model.dir_head(feature)
            e_heads_e.record()
            torch.cuda.synchronize()

            wall = (time.time() - t0) * 1000

            if i >= args.warmup:
                outer_records["backbone"].append(e_back_s.elapsed_time(e_back_e))
                outer_records["aligner"].append(e_back_e.elapsed_time(e_align_e))
                outer_records["pyramid_single"].append(e_align_e.elapsed_time(e_pyr_e))
                outer_records["shrink"].append(e_pyr_e.elapsed_time(e_shrink_e))
                outer_records["heads"].append(e_shrink_e.elapsed_time(e_heads_e))
                e2e_walltime.append(wall)

            if i + 1 == args.warmup:
                # reset LSS records
                for k in LSS_STAGE_RECORDS:
                    LSS_STAGE_RECORDS[k] = []
                print(f"[warmup-done] reset", flush=True)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{total}]", flush=True)

    # report
    report = {
        "ckpt": args.ckpt,
        "warmup": args.warmup,
        "measure": args.measure,
        "n_collected": len(e2e_walltime),
        "lss_config": {
            "Ncams": 4,
            "final_dim": list(encoder_m2.data_aug_conf["final_dim"]),
            "downsample": int(encoder_m2.downsample),
            "D_depth_bins": int(encoder_m2.D),
            "camC": int(encoder_m2.camC),
            "nx": encoder_m2.nx.tolist(),
        },
        "Nprime_stats": stats(LSS_STAGE_RECORDS["Nprime"]),
        "Nkept_stats": stats(LSS_STAGE_RECORDS["Nkept"]),
        "lss_internal_ms": {k: stats(LSS_STAGE_RECORDS[k]) for k in
                            ["cam_encode", "get_geometry",
                             "vp_flatten_quantize", "vp_range_filter",
                             "vp_rank_argsort", "vp_quickcumsum",
                             "vp_scatter_collapse", "voxel_pool_total"]},
        "outer_ms": {k: stats(v) for k, v in outer_records.items()},
        "e2e_forward_walltime_ms": stats(e2e_walltime),
        "device": torch.cuda.get_device_name(0),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {args.out}\n", flush=True)

    # Pretty print
    print("=== LSS internal (mean / p50 / p99 ms) ===")
    for k in ["cam_encode", "get_geometry", "vp_flatten_quantize", "vp_range_filter",
              "vp_rank_argsort", "vp_quickcumsum", "vp_scatter_collapse", "voxel_pool_total"]:
        s = stats(LSS_STAGE_RECORDS[k])
        if s["mean"] is None:
            print(f"  {k:24s}  no data")
        else:
            print(f"  {k:24s}  mean={s['mean']:7.3f}  p50={s['p50']:7.3f}  p99={s['p99']:7.3f}")
    print("\n=== outer stages (mean / p50 / p99 ms) ===")
    for k, v in outer_records.items():
        s = stats(v)
        print(f"  {k:18s}  mean={s['mean']:7.3f}  p50={s['p50']:7.3f}  p99={s['p99']:7.3f}")
    s_e2e = stats(e2e_walltime)
    print(f"\n  e2e forward walltime (no NMS)  mean={s_e2e['mean']:7.3f}  p50={s_e2e['p50']:7.3f}")
    s_np = stats(LSS_STAGE_RECORDS["Nprime"])
    s_nk = stats(LSS_STAGE_RECORDS["Nkept"])
    print(f"\n  Nprime (frustum points) mean={s_np['mean']:.0f}  Nkept (in-range) mean={s_nk['mean']:.0f}")


if __name__ == "__main__":
    main()
