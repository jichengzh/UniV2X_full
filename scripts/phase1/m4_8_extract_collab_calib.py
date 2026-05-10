"""Phase A.5 — Extract DAIR-V2X N=2 calibration data for collab INT8 engine.

Captures (spatial_features, t_ego) pairs from DAIR val 100 samples where
N=2 (cooperative vehicle + RSU). For multi-input INT8 calibration.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.utils.transformation_utils import normalize_pairwise_tfm  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "calibration"))
    p.add_argument("--range", default="102.4,51.2")
    return p.parse_args()


def main():
    args = parse_args()
    hypes = yaml_utils.load_yaml(str(Path(args.model_dir) / "config.yaml"))
    if "heter" in hypes:
        x_max, y_max = [float(x) for x in args.range.split(",")]
        new_range = [
            -x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {"cav_lidar_range": new_range,
                                    "lidar_range": new_range,
                                    "gt_range": new_range})
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    print("[1] build model + load")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.cuda().eval()

    print("[2] build dataset")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)
    print(f"  size {len(dataset)}")

    print(f"[3] capture {args.n_samples} N=2 (spatial, t_ego) pairs")
    spatials = []
    t_egos = []
    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            if len(spatials) >= args.n_samples:
                break
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            aml = ego["agent_modality_list"]
            mc = Counter(aml)
            mf = {}
            for m in model.modality_name_list:
                if m not in mc:
                    continue
                f = getattr(model, f"encoder_{m}")(ego, m)
                f = getattr(model, f"backbone_{m}")({"spatial_features": f})["spatial_features_2d"]
                f = getattr(model, f"aligner_{m}")(f)
                mf[m] = f
            cnt = {m: 0 for m in model.modality_name_list}
            hl = []
            for m in aml:
                hl.append(mf[m][cnt[m]])
                cnt[m] += 1
            h2d = torch.stack(hl)
            if h2d.shape[0] != 2:
                continue
            af = normalize_pairwise_tfm(ego["pairwise_t_matrix"], model.H, model.W,
                                         model.fake_voxel_size)
            t_ego = af[0, 0, :2, :, :].contiguous()
            spatials.append(h2d.cpu().float().numpy())
            t_egos.append(t_ego.cpu().float().numpy())
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}: captured {len(spatials)}/{args.n_samples}")

    sp = np.stack(spatials[:args.n_samples], axis=0)   # (N, 2, 64, 128, 256)
    te = np.stack(t_egos[:args.n_samples], axis=0)     # (N, 2, 2, 3)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spatial_p = out_dir / "pyramid_dair_collab_spatial.npy"
    tego_p = out_dir / "pyramid_dair_collab_tego.npy"
    np.save(spatial_p, sp)
    np.save(tego_p, te)
    print(f"  saved spatial -> {spatial_p}  shape={sp.shape}  size={spatial_p.stat().st_size/1e6:.1f} MB")
    print(f"  saved t_ego   -> {tego_p}  shape={te.shape}")
    print(f"  spatial range [{sp.min():.3f}, {sp.max():.3f}]")
    print(f"  t_ego   range [{te.min():.3f}, {te.max():.3f}]")


if __name__ == "__main__":
    main()
