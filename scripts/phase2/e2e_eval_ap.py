"""E2E TRT engine → AP30/50/70 eval on DAIR-V2X val.

Forwards N samples through e2e TRT engine, runs post_process + CUDA NMS,
accumulates tp/fp, calls eval_final_results.

Usage:
  e2e_eval_ap.py --engine X.engine --ckpt-dir <heal ckpt> --n-samples 500 --report Y.json
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
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/phase2"))
os.chdir(str(HEAL_ROOT))

# Patch read_json
import opencood.utils.common_utils as _cu
_orig_read_json = _cu.read_json


def _safe_read_json(p):
    if not os.path.exists(p) and ("backup" in str(p) or "label" in str(p).split("/")[-2:]):
        return []
    return _orig_read_json(p)


_cu.read_json = _safe_read_json
import opencood.data_utils.datasets.basedataset.dairv2x_basedataset as _dair_mod
_dair_mod.read_json = _safe_read_json

from opencood.hypes_yaml import yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.common_utils import update_dict
from opencood.utils import box_utils, eval_utils

# Reuse mmcv CUDA NMS
from m4_8_cuda_kernel_replacements import nms_rotated_mmcv  # noqa: E402

# Reuse TrtEngine + pad_voxels from e2e_bench_pyramid
from e2e_bench_pyramid import TrtEngine, pad_voxels  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def install_cuda_nms():
    box_utils.nms_rotated = nms_rotated_mmcv
    import opencood.data_utils.post_processor.voxel_postprocessor as vp_mod
    vp_mod.box_utils.nms_rotated = nms_rotated_mmcv


def load_dataset(ckpt_dir, dair_root):
    class _Opt:
        model_dir = ckpt_dir

    hypes = yaml_utils.load_yaml(None, _Opt())
    new_range = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
    hypes = update_dict(hypes, {
        "cav_lidar_range": new_range, "lidar_range": new_range, "gt_range": new_range
    })
    val_split = f"{dair_root}/val.json"
    hypes["data_dir"] = dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    import importlib
    parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"),
                          hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["data_dir"] = dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    # We need model just to get H, W, fake_voxel_size for normalize_pairwise_tfm
    model = train_utils.create_model(hypes)
    H, W, fake_vs = model.H, model.W, model.fake_voxel_size
    del model
    torch.cuda.empty_cache()

    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, num_workers=2,
                        collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False)
    return ds, loader, hypes, H, W, fake_vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--dair-root",
                    default="/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure")
    ap.add_argument("--max-voxels", type=int, default=32000)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    install_cuda_nms()

    print(f"[1/3] load dataset + meta")
    ds, loader, hypes, H, W, fake_vs = load_dataset(args.ckpt_dir, args.dair_root)
    print(f"  H={H}, W={W}, fake_vs={fake_vs}, dataset size={len(ds)}")

    print(f"[2/3] load TRT engine: {args.engine}")
    engine = TrtEngine(args.engine, args.max_voxels)
    stream = torch.cuda.Stream()
    device = torch.device("cuda")

    print(f"[3/3] inference + AP eval, target {args.n_samples} samples")
    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}
    n_done = 0
    n_skipped = 0
    lat_e2e = []
    t_global = time.time()

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if n_done >= args.n_samples:
                break
            batch = train_utils.to_device(batch, device)
            data_dict = batch["ego"]

            voxel_data = data_dict.get("inputs_m1") or data_dict.get("processed_lidar")
            if voxel_data is None:
                n_skipped += 1; continue
            vf = voxel_data["voxel_features"]
            vnp = voxel_data["voxel_num_points"]
            vc = voxel_data["voxel_coords"].int()
            pairwise_t = data_dict["pairwise_t_matrix"]
            record_len = data_dict["record_len"]

            if vf.shape[0] > args.max_voxels:
                n_skipped += 1; continue
            N_b = int(record_len[0].item()) if isinstance(record_len, torch.Tensor) else int(record_len[0])
            if N_b != 2:
                n_skipped += 1; continue

            affine = normalize_pairwise_tfm(pairwise_t, H, W, fake_vs)
            t_ego = affine[0][0][:2].contiguous()

            vf_p, vnp_p, vc_p, vmask, m_real = pad_voxels(vf, vnp, vc, args.max_voxels)

            torch.cuda.synchronize()
            t_a = time.time()
            with torch.cuda.stream(stream):
                cls, reg, dir_ = engine.run(vf_p, vnp_p, vc_p, vmask, t_ego, stream)
            stream.synchronize()
            output_dict = {"ego": {
                "cls_preds": cls.clone(),
                "reg_preds": reg.clone(),
                "dir_preds": dir_.clone(),
            }}
            pred_box, pred_score = ds.post_processor.post_process(batch, output_dict)
            torch.cuda.synchronize()
            lat_e2e.append((time.time() - t_a) * 1000)

            # gt_box from data_dict
            gt_box_tensor = ds.post_processor.generate_gt_bbx(batch)

            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box_tensor,
                                           result_stat, iou_th)
            n_done += 1
            if n_done % 50 == 0:
                el = time.time() - t_global
                print(f"  {n_done}/{args.n_samples}  elapsed={el:.0f}s "
                      f"avg_lat={np.mean(lat_e2e):.1f}ms")

    print(f"\n[done] {n_done} samples ({n_skipped} skipped) in {time.time()-t_global:.0f}s")

    out_dir = Path(args.report).parent / f"ap_eval_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(out_dir))

    rep = {
        "tag": args.tag,
        "engine": args.engine,
        "ckpt_dir": args.ckpt_dir,
        "n_samples": n_done,
        "n_skipped": n_skipped,
        "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
        "lat_mean_ms": float(np.mean(lat_e2e)),
        "lat_p50_ms": float(np.percentile(lat_e2e, 50)),
        "lat_p99_ms": float(np.percentile(lat_e2e, 99)),
        "elapsed_secs": time.time() - t_global,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(rep, indent=2))

    print(f"\n=== AP eval: {args.tag} ===")
    print(f"  AP30 = {rep['ap30']:.4f}")
    print(f"  AP50 = {rep['ap50']:.4f}")
    print(f"  AP70 = {rep['ap70']:.4f}")
    print(f"  lat_e2e = {rep['lat_mean_ms']:.2f} ms (mean)")
    print(f"  report → {args.report}")


if __name__ == "__main__":
    main()
