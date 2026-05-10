"""Phase A.3a — Extract OPV2V calibration features for TRT INT8.

Runs HEAL Pyramid_m1 inference pipeline on the first N OPV2V test samples
and dumps the per-agent spatial_features (output of backbone_m1 + aligner_m1,
== input to pyramid_backbone) as a numpy array, used by TRT IInt8MinMaxCalibrator.

For multi-agent samples, all agents from a sample are kept (so a 5-CAV sample
contributes 5 calib feature maps). Stops once we have ``--n-samples`` features.

Output:
    calibration/pyramid_calib.npy   shape (N, 64, 256, 256) float32
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import OrderedDict, Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)  # HEAL data builders use relative paths

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12")
    p.add_argument("--n-samples", type=int, default=100, help="N feature maps to capture")
    p.add_argument("--out", default=str(REPO_ROOT / "calibration/pyramid_calib.npy"))
    p.add_argument("--range", default="102.4,102.4")
    return p.parse_args()


def main():
    args = parse_args()
    hypes_path = Path(args.model_dir) / "config.yaml"
    hypes = yaml_utils.load_yaml(str(hypes_path))

    # Re-apply the `--range` cav_lidar_range patch (from HEAL inference.py).
    if "heter" in hypes:
        x_max = float(args.range.split(",")[0])
        y_max = float(args.range.split(",")[1])
        new_range = [
            -x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_range,
            "lidar_range": new_range,
            "gt_range": new_range,
        })
        import importlib
        yu_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        parser_func = getattr(yu_lib, hypes["yaml_parser"])
        hypes = parser_func(hypes)

    hypes["validate_dir"] = hypes["test_dir"]

    print("[1/4] build model + load ckpt")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.cuda().eval()

    print("[2/4] build dataset")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset, batch_size=1, num_workers=2,
        collate_fn=dataset.collate_batch_test, shuffle=False,
        pin_memory=False, drop_last=False,
    )
    print(f"  dataset size: {len(dataset)}")

    print(f"[3/4] capture {args.n_samples} spatial_features ...")
    captured: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]

            agent_modality_list = ego["agent_modality_list"]
            modality_count = Counter(agent_modality_list)
            modality_feature_dict: dict[str, torch.Tensor] = {}

            # mirror HeterPyramidCollab.forward up to assemble heter_feature_2d
            for modality_name in model.modality_name_list:
                if modality_name not in modality_count:
                    continue
                feat = getattr(model, f"encoder_{modality_name}")(ego, modality_name)
                feat = getattr(model, f"backbone_{modality_name}")({"spatial_features": feat})["spatial_features_2d"]
                feat = getattr(model, f"aligner_{modality_name}")(feat)
                modality_feature_dict[modality_name] = feat  # (N_modal, 64, 256, 256)

            # assemble per-agent
            counting = {m: 0 for m in model.modality_name_list}
            heter_list = []
            for m in agent_modality_list:
                idx = counting[m]
                heter_list.append(modality_feature_dict[m][idx])  # (64, 256, 256)
                counting[m] += 1
            heter_feat_2d = torch.stack(heter_list)  # (sum_cav, 64, 256, 256)

            assert heter_feat_2d.shape[1:] == (64, 256, 256), \
                f"unexpected shape {heter_feat_2d.shape}"
            for k in range(heter_feat_2d.shape[0]):
                captured.append(heter_feat_2d[k].cpu().float().numpy())
                if len(captured) >= args.n_samples:
                    break

            if batch_idx % 10 == 0:
                print(f"  batch {batch_idx:3d} | captured {len(captured)}/{args.n_samples}")
            if len(captured) >= args.n_samples:
                break

    arr = np.stack(captured[:args.n_samples], axis=0).astype(np.float32)
    print(f"[4/4] captured array shape={arr.shape} "
          f"min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.4f} std={arr.std():.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)
    print(f"  saved -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
