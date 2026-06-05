"""
Convert DAIR-V2X val .pcd files → .npy files for Orin deployment.

Converts N val frames (both infra-side + vehicle-side point clouds),
outputting  dair_val_npy/infra/<frame_id>.npy  and
            dair_val_npy/veh/<frame_id>.npy

Each .npy is float32 [N_pts, 4]: (x, y, z, intensity)
intensity is normalized to [0, 1] by HEAL's read_pcd (divide by 256).

Usage (on 4090 with HEAL conda env, pypcd installed):
  conda activate UniV2X_2.0
  cd /home/jichengzhi/UniV2X
  python tools/orin_deploy/m1_standalone/convert_dair_to_npy.py --n_frames 50

Author: sw-optimizer (Task #12, 2026-06-06)
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

HEAL_ROOT = os.environ.get("HEAL_ROOT", "/home/jichengzhi/heal_research/HEAL")
if HEAL_ROOT not in sys.path:
    sys.path.insert(0, HEAL_ROOT)

from opencood.utils import pcd_utils

DAIR_ROOT = ("/home/jichengzhi/heal_research/dataset/my_dair_v2x/"
             "v2x_c/cooperative-vehicle-infrastructure")
OUT_DIR   = Path(__file__).parent / "dair_val_npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames", type=int, default=50,
                        help="Number of val frames to convert")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    out_infra = Path(args.out_dir) / "infra"
    out_veh   = Path(args.out_dir) / "veh"
    out_infra.mkdir(parents=True, exist_ok=True)
    out_veh.mkdir(parents=True, exist_ok=True)

    val_ids  = json.load(open(os.path.join(DAIR_ROOT, "val.json")))
    co_info  = json.load(open(os.path.join(DAIR_ROOT, "cooperative/data_info.json")))
    co_by_veh = {
        item["vehicle_image_path"].split("/")[-1].replace(".jpg", ""): item
        for item in co_info
    }
    val_frames = [co_by_veh[v] for v in val_ids if v in co_by_veh][:args.n_frames]

    print(f"[convert_dair_to_npy] Converting {len(val_frames)} frames → {args.out_dir}")
    ok, err = 0, 0
    for idx, frame in enumerate(val_frames):
        # Infra side
        infra_pcd = os.path.join(DAIR_ROOT, frame["infrastructure_pointcloud_path"])
        infra_npy = out_infra / (os.path.basename(infra_pcd).replace(".pcd", ".npy"))
        # Vehicle side
        veh_pcd   = os.path.join(DAIR_ROOT, frame["vehicle_pointcloud_path"])
        veh_npy   = out_veh / (os.path.basename(veh_pcd).replace(".pcd", ".npy"))

        try:
            if not infra_npy.exists():
                pts, _ = pcd_utils.read_pcd(infra_pcd)
                np.save(str(infra_npy), pts.astype(np.float32))
            if not veh_npy.exists():
                pts, _ = pcd_utils.read_pcd(veh_pcd)
                np.save(str(veh_npy), pts.astype(np.float32))
            ok += 1
        except Exception as exc:
            print(f"  [ERR] frame {idx}: {exc}")
            err += 1

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(val_frames)} done")

    print(f"\n[convert_dair_to_npy] Done: {ok} OK, {err} errors")
    print(f"  infra npy count: {len(list(out_infra.glob('*.npy')))}")
    print(f"  veh   npy count: {len(list(out_veh.glob('*.npy')))}")

    # Write manifest
    manifest_path = Path(args.out_dir) / "manifest.json"
    manifest = []
    for frame in val_frames:
        infra_name = os.path.basename(frame["infrastructure_pointcloud_path"]).replace(".pcd", ".npy")
        veh_name   = os.path.basename(frame["vehicle_pointcloud_path"]).replace(".pcd", ".npy")
        manifest.append({
            "infra_npy": f"infra/{infra_name}",
            "veh_npy":   f"veh/{veh_name}",
            "veh_id":    frame["vehicle_image_path"].split("/")[-1].replace(".jpg", ""),
        })
    import json as _json
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)
    print(f"  manifest → {manifest_path}")


if __name__ == "__main__":
    main()
