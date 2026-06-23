"""Dump DAIR test voxel batches as padded calibration data for INT8.

Outputs 5 separate .npy files (one per ONNX input):
  voxel_features      (N_samp, MAX_VOX, 32, 4)  float32
  voxel_num_points    (N_samp, MAX_VOX)         int32
  voxel_coords        (N_samp, MAX_VOX, 4)      int32
  voxel_mask          (N_samp, MAX_VOX)         float32
  t_ego               (N_samp, 2, 2, 3)         float32
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path("/home/jichengzhi/UniV2X")
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/phase2"))
os.chdir(str(HEAL_ROOT))

# Patch read_json (label_world_backup is missing)
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
from opencood.utils.common_utils import update_dict
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--dair-root",
                    default="/home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure")
    ap.add_argument("--max-voxels", type=int, default=32000)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    class _Opt:
        model_dir = args.ckpt_dir

    hypes = yaml_utils.load_yaml(None, _Opt())
    new_range = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
    hypes = update_dict(hypes, {
        "cav_lidar_range": new_range,
        "lidar_range": new_range,
        "gt_range": new_range,
    })
    val_split = f"{args.dair_root}/val.json"
    hypes["data_dir"] = args.dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    import importlib
    parser_func = getattr(importlib.import_module("opencood.hypes_yaml.yaml_utils"),
                          hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["data_dir"] = args.dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    # Build light model for H/W/fake_voxel_size (we only need params)
    model = train_utils.create_model(hypes)
    H, W, fake_vs = model.H, model.W, model.fake_voxel_size
    del model
    torch.cuda.empty_cache()

    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, num_workers=4,
                        collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False)

    n_samp = args.n_samples
    MV = args.max_voxels
    P = 32

    vf_arr = np.zeros((n_samp, MV, P, 4), dtype=np.float32)
    np_arr = np.zeros((n_samp, MV), dtype=np.int32)
    vc_arr = np.zeros((n_samp, MV, 4), dtype=np.int32)
    vm_arr = np.zeros((n_samp, MV), dtype=np.float32)
    te_arr = np.zeros((n_samp, 2, 2, 3), dtype=np.float32)

    n_collected = 0
    for i, batch in enumerate(loader):
        if n_collected >= n_samp:
            break
        if batch is None:
            continue
        data = batch["ego"]
        vf = data["inputs_m1"]["voxel_features"]
        npts = data["inputs_m1"]["voxel_num_points"]
        vc = data["inputs_m1"]["voxel_coords"].int()
        M = vf.shape[0]
        if M > MV:
            continue

        affine = normalize_pairwise_tfm(data["pairwise_t_matrix"], H, W, fake_vs)
        if int(data["record_len"][0]) != 2:
            continue
        t_ego = affine[0][0][:2]  # (2, 2, 3)

        vf_arr[n_collected, :M] = vf.cpu().numpy()
        np_arr[n_collected, :M] = npts.cpu().numpy().astype(np.int32)
        vc_arr[n_collected, :M] = vc.cpu().numpy()
        vm_arr[n_collected, :M] = 1.0
        te_arr[n_collected] = t_ego.cpu().numpy()
        n_collected += 1
        if n_collected % 50 == 0:
            print(f"  collected {n_collected}/{n_samp}", flush=True)

    if n_collected < n_samp:
        print(f"[warn] only collected {n_collected} of {n_samp}; trimming arrays")
        vf_arr = vf_arr[:n_collected]
        np_arr = np_arr[:n_collected]
        vc_arr = vc_arr[:n_collected]
        vm_arr = vm_arr[:n_collected]
        te_arr = te_arr[:n_collected]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "voxel_features.npy", vf_arr)
    np.save(out_dir / "voxel_num_points.npy", np_arr)
    np.save(out_dir / "voxel_coords.npy", vc_arr)
    np.save(out_dir / "voxel_mask.npy", vm_arr)
    np.save(out_dir / "t_ego.npy", te_arr)
    print(f"\nwrote {n_collected} calibration samples to {out_dir}")


if __name__ == "__main__":
    main()
