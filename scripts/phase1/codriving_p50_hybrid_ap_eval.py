"""CoDriving p50 hybrid AP eval harness — DAIR val 1789.

Architecture: PyTorch VFE + scatter (p50 ckpt) -> TRT p50 collab engine -> PyTorch postprocess.

p50 model: backbone pruned 74.8%, backbone+heads 1.91M params.
p50 collab TRT engine: processes (2,64,256,512) spatial_features + pairwise_t_matrix.
Outputs cls_preds (1,1,128,256) + reg_preds (1,8,128,256).

FP16 sanity target: AP50 ≈ 0.615 (PyTorch p50 baseline from HANDOFF).
FP16 sanity gate: |ΔAP50| < 0.02 vs target.

Usage:
    # FP16 sanity gate
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/codriving_p50_hybrid_ap_eval.py \
        --engine output/codriving_pilot/collab_engines/codriving_collab_p50_fp16.engine \
        --tag p50_fp16 --n-samples 1789

    # INT8 eval
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/codriving_p50_hybrid_ap_eval.py \
        --engine output/codriving_pilot/collab_engines/codriving_collab_p50_int8.engine \
        --tag p50_int8 --n-samples 1789
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import yaml
import tensorrt as trt
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
V2XVERSE_ROOT = Path("/home/jichengzhi/V2Xverse")

# p50 config and ckpt
HYPES_PATH = REPO_ROOT / "output/codriving_pilot/p50/config_finetune.yaml"
CKPT_PATH = REPO_ROOT / "output/codriving_pilot/p50/net_epoch_bestval_at5_p50.pth"
RESULTS_CSV = REPO_ROOT / "results/E_codriving_collab_ap_4090.csv"

SPATIAL_SHAPE = (2, 64, 256, 512)
TMAT_SHAPE = (1, 2, 2, 4, 4)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# TRT collab engine wrapper (identical to base harness)
# ---------------------------------------------------------------------------

class TrtCollabCodriving:
    def __init__(self, engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]
        self.spatial_name = next(n for n in self.input_names if "spatial" in n.lower())
        self.tmat_name = next(n for n in self.input_names
                              if "pairwise" in n.lower() or "t_matrix" in n.lower())
        self.stream = torch.cuda.Stream()
        print(f"  [TRT] inputs={self.input_names}  outputs={self.output_names}")
        # Create context once at init (not per-call) to avoid None on allocation failure
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(
                f"TRT create_execution_context() returned None for {engine_path}. "
                "Check GPU memory and engine validity.")
        self.ctx.set_input_shape(self.spatial_name, SPATIAL_SHAPE)
        self.ctx.set_input_shape(self.tmat_name, TMAT_SHAPE)
        print(f"  [TRT] execution context created OK")

    def __call__(self, spatial: torch.Tensor,
                 pairwise_t_matrix: torch.Tensor) -> dict:
        ctx = self.ctx

        spatial_in = spatial.float().contiguous()
        tmat_in = pairwise_t_matrix.float().contiguous()
        bufs = {self.spatial_name: spatial_in, self.tmat_name: tmat_in}

        for n in self.output_names:
            shape = tuple(ctx.get_tensor_shape(n))
            bufs[n] = torch.empty(shape, dtype=torch.float32, device="cuda")

        for n in list(bufs.keys()):
            ctx.set_tensor_address(n, int(bufs[n].data_ptr()))

        with torch.cuda.stream(self.stream):
            ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        cls_name = next(n for n in self.output_names if "cls" in n.lower())
        reg_name = next(n for n in self.output_names if "reg" in n.lower())
        return {
            "cls_preds_raw": bufs[cls_name],
            "reg_preds_raw": bufs[reg_name],
        }


# ---------------------------------------------------------------------------
# CenterPoint box decoder
# ---------------------------------------------------------------------------

def decode_centerpoint_boxes(cls_preds_raw, reg_preds_raw,
                              out_size_factor, voxel_size, cav_lidar_range):
    box_preds = reg_preds_raw.permute(0, 2, 3, 1).contiguous()
    batch, H, W, code_size = box_preds.size()
    box_preds = box_preds.reshape(batch, H * W, code_size)

    batch_reg = box_preds[..., 0:2]
    h = box_preds[..., 3:4] * out_size_factor * voxel_size[0]
    w = box_preds[..., 4:5] * out_size_factor * voxel_size[1]
    l = box_preds[..., 5:6] * out_size_factor * voxel_size[2]
    batch_dim = torch.cat([h, w, l], dim=-1)
    batch_hei = (box_preds[..., 2:3] * out_size_factor * voxel_size[2]
                 + cav_lidar_range[2])
    batch_rots = box_preds[..., 6:7]
    batch_rotc = box_preds[..., 7:8]
    rot = torch.atan2(batch_rots, batch_rotc)

    ys, xs = torch.meshgrid(
        [torch.arange(0, H), torch.arange(0, W)], indexing="ij")
    ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds_raw.device)
    xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds_raw.device)
    xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
    ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]
    xs = xs * out_size_factor * voxel_size[0] + cav_lidar_range[0]
    ys = ys * out_size_factor * voxel_size[1] + cav_lidar_range[1]

    batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
    return cls_preds_raw, batch_box_preds


# ---------------------------------------------------------------------------
# Load hypes
# ---------------------------------------------------------------------------

def load_hypes(hypes_path: str) -> dict:
    loader_cls = yaml.Loader
    loader_cls.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.([0-9_]*)(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(hypes_path, "r") as f:
        return yaml.load(f, Loader=loader_cls)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(args):
    device = "cuda"
    sys.path.insert(0, str(V2XVERSE_ROOT))
    os.chdir(str(V2XVERSE_ROOT))

    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils

    print(f"[p50 eval] Loading p50 config: {HYPES_PATH}")
    hypes = load_yaml(str(HYPES_PATH))

    # Override data paths to 4090 local DAIR paths (p50 config has H800 paths)
    DAIR_DATA_4090 = "/data/jichengzhi_dair/dair_eval"
    hypes["data_dir"] = f"{DAIR_DATA_4090}/cooperative-vehicle-infrastructure"
    hypes["root_dir"] = f"{DAIR_DATA_4090}/split_json/train.json"
    hypes["validate_dir"] = f"{DAIR_DATA_4090}/split_json/val.json"
    hypes["test_dir"] = f"{DAIR_DATA_4090}/split_json/val.json"
    print(f"  data_dir overridden to: {hypes['data_dir']}")

    print(f"[p50 eval] Creating model (p50 arch: num_filters=[32,64,128])")
    model = train_utils.create_model(hypes)

    print(f"[p50 eval] Loading p50 ckpt: {CKPT_PATH}")
    raw = torch.load(str(CKPT_PATH), map_location="cpu")
    # Handle flat state_dict (190 keys: backbone+pillar_vfe+cls_head+reg_head)
    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif any(k.startswith("module.") for k in raw.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in raw.items()}
    else:
        state_dict = raw
    miss, unexp = model.load_state_dict(state_dict, strict=False)
    print(f"  load: missing={len(miss)}, unexpected={len(unexp)}")
    if miss:
        # fusion_net has no params -> scatter missing is OK
        non_fusion = [k for k in miss if "fusion_net" not in k and "scatter" not in k]
        if non_fusion:
            print(f"  WARNING: non-trivial missing keys: {non_fusion[:5]}")
    model = model.to(device).eval()

    print(f"[p50 eval] Building dataset (DAIR val)...")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
    )

    # TRT engine or fallback
    trt_engine = None
    if not args.force_fallback:
        print(f"[p50 eval] Loading TRT engine: {args.engine}")
        trt_engine = TrtCollabCodriving(args.engine)

    voxel_size = hypes["preprocess"]["args"]["voxel_size"]
    cav_lidar_range = list(hypes["preprocess"]["cav_lidar_range"])
    out_size_factor = hypes["model"]["args"]["out_size_factor"]

    # AP eval accumulators — using same result_stat structure as base harness
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    n_done = 0
    n_trt_path = 0
    n_fallback_path = 0
    n_skipped = 0
    t_start = time.time()

    print(f"[p50 eval] Running {args.n_samples} samples (tag={args.tag})...")
    with torch.inference_mode():
        for batch_data in loader:
            if n_done >= args.n_samples:
                break
            if batch_data is None:
                n_skipped += 1
                continue

            batch_data = train_utils.to_device(batch_data, device)
            ego = batch_data["ego"]
            record_len = ego["record_len"]
            n_agents = int(record_len[0].item())

            if args.force_fallback or trt_engine is None or n_agents != 2:
                # PyTorch fallback for non-2-agent or force-fallback
                output_dict_raw = model(ego)
                output_dict = {"ego": {
                    "cls_preds": output_dict_raw["cls_preds"],
                    "reg_preds": output_dict_raw["reg_preds"],
                }}
                n_fallback_path += 1
            else:
                # VFE + scatter (PyTorch p50 weights)
                voxel_batch = {
                    "voxel_features": ego["processed_lidar"]["voxel_features"],
                    "voxel_coords": ego["processed_lidar"]["voxel_coords"],
                    "voxel_num_points": ego["processed_lidar"]["voxel_num_points"],
                    "record_len": ego["record_len"],
                }
                voxel_batch = model.pillar_vfe(voxel_batch)
                voxel_batch = model.scatter(voxel_batch)
                spatial_features = voxel_batch["spatial_features"]  # (2,64,256,512)
                pairwise_t_matrix = ego["pairwise_t_matrix"]         # (1,2,2,4,4)

                # TRT collab inference
                trt_out = trt_engine(spatial_features, pairwise_t_matrix)
                cls_raw = trt_out["cls_preds_raw"]   # (1,1,128,256)
                reg_raw = trt_out["reg_preds_raw"]   # (1,8,128,256)

                # Decode CenterPoint boxes (same as base harness)
                cls_decoded, reg_decoded = decode_centerpoint_boxes(
                    cls_raw, reg_raw, out_size_factor, voxel_size, cav_lidar_range)

                output_dict = {"ego": {
                    "cls_preds": cls_decoded,
                    "reg_preds": reg_decoded,
                }}
                n_trt_path += 1

            # Post-process via dataset (NMS etc) -- same as base harness
            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
                batch_data, output_dict)

            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor,
                    result_stat, iou_th)

            n_done += 1
            if n_done % 100 == 0:
                elapsed = time.time() - t_start
                print(f"  [{n_done}/{args.n_samples}]  trt={n_trt_path}  "
                      f"fb={n_fallback_path}  elapsed={elapsed:.1f}s  skipped={n_skipped}")

    elapsed = time.time() - t_start
    print(f"\n[p50 eval] Inference done: n={n_done} trt={n_trt_path} fb={n_fallback_path} skip={n_skipped}")

    # Compute AP
    out_dir = REPO_ROOT / f"results/E_codriving_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ap_30, ap_50, ap_70 = eval_utils.eval_final_results(result_stat, str(out_dir))

    print(f"  AP30={ap_30:.3f}  AP50={ap_50:.3f}  AP70={ap_70:.3f}")
    print(f"  (target p50 FP16: AP50≈0.615)")

    # Sanity gate check
    sanity_target = 0.615
    if args.tag == "p50_fp16":
        delta = abs(ap_50 - sanity_target)
        if delta < 0.02:
            print(f"  FP16 SANITY GATE: PASS (|ΔAP50|={delta:.4f} < 0.02)")
        else:
            print(f"  FP16 SANITY GATE: FAIL (|ΔAP50|={delta:.4f} >= 0.02, target={sanity_target})")

    # Append result to CSV
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    import csv
    cols = ["model", "precision", "device", "gpu", "n_samples", "n_done",
            "ap30", "ap50", "ap70", "eval_kind", "eval_method", "note"]
    row = {
        "model": "codriving_collab_p50",
        "precision": args.tag.replace("p50_", ""),
        "device": "RTX_4090",
        "gpu": f"GPU{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}",
        "n_samples": args.n_samples,
        "n_done": n_done,
        "ap30": round(float(ap_30), 4),
        "ap50": round(float(ap_50), 4),
        "ap70": round(float(ap_70), 4),
        "eval_kind": "collab_n2_trt",
        "eval_method": "hybrid_pytorch_trt",
        "note": f"DAIR_val_1789_p50_{args.tag}",
    }
    write_header = not RESULTS_CSV.exists()
    with open(str(RESULTS_CSV), "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n[p50 eval] Result appended to: {RESULTS_CSV}")
    print(f"  Row: {row}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", type=str, default=None)
    p.add_argument("--tag", type=str, required=True, help="e.g. p50_fp16, p50_int8")
    p.add_argument("--n-samples", type=int, default=1789)
    p.add_argument("--force-fallback", action="store_true",
                   help="Use PyTorch model (no TRT) for baseline check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.force_fallback and args.engine is None:
        raise ValueError("--engine required unless --force-fallback")
    run_eval(args)
