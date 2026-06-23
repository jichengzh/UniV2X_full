"""CoDriving hybrid AP eval harness — DAIR val 1789.

Architecture: PyTorch VFE + scatter → TRT collab engine → PyTorch postprocess.

The collab TRT engine replaces: backbone + CoDriving fusion + cls/reg heads.
PyTorch handles: PillarVFE + PointPillarScatter (light ops, output spatial_features).
Post-decode: generate_predicted_boxes (runs in PyTorch on TRT outputs).
Post-process: VoxelPostprocessor.post_process (NMS etc).

Target (PyTorch baseline): AP30/50/70 ≈ 0.674/0.561/0.369
FP16 sanity gate: |ΔAP50| < 0.01 vs target.

Usage:
    # FP16 sanity gate
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/codriving_hybrid_ap_eval.py \
        --engine output/codriving_pilot/collab_engines/codriving_collab_base_fp16.engine \
        --tag fp16 --n-samples 1789

    # INT8 eval (only after FP16 passes)
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/codriving_hybrid_ap_eval.py \
        --engine output/codriving_pilot/collab_engines/codriving_collab_base_int8.engine \
        --tag int8 --n-samples 1789

    # PyTorch baseline (force-fallback, no TRT)
    CUDA_VISIBLE_DEVICES=7 python scripts/phase1/codriving_hybrid_ap_eval.py \
        --force-fallback --tag pytorch_baseline --n-samples 1789
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

CKPT_DIR = REPO_ROOT / "output/codriving_pilot/collab_export"
HYPES_PATH = CKPT_DIR / "dair_centerpoint_codriving_4090.yaml"
SPATIAL_SHAPE = (2, 64, 256, 512)
TMAT_SHAPE = (1, 2, 2, 4, 4)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# TRT collab engine wrapper
# ---------------------------------------------------------------------------

class TrtCollabCodriving:
    """CoDriving collab N=2 TRT engine wrapper.

    Inputs:
        spatial_features  (2, 64, 256, 512)  float32 CUDA
        pairwise_t_matrix (1, 2, 2, 4, 4)    float32 CUDA
    Outputs:
        cls_preds (1, 1, 128, 256)
        reg_preds (1, 8, 128, 256)
    """

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

    def __call__(self, spatial: torch.Tensor,
                 pairwise_t_matrix: torch.Tensor) -> dict:
        """
        spatial: (2, 64, 256, 512) float32 CUDA
        pairwise_t_matrix: (1, 2, 2, 4, 4) float32 CUDA
        Returns dict with cls_preds (1,1,128,256) and reg_preds (1,8,128,256) raw logits.
        """
        ctx = self.engine.create_execution_context()
        ctx.set_input_shape(self.spatial_name, SPATIAL_SHAPE)
        ctx.set_input_shape(self.tmat_name, TMAT_SHAPE)

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

        # Map by output name
        cls_name = next(n for n in self.output_names if "cls" in n.lower())
        reg_name = next(n for n in self.output_names if "reg" in n.lower())
        return {
            "cls_preds_raw": bufs[cls_name],   # (1,1,128,256) raw logits
            "reg_preds_raw": bufs[reg_name],   # (1,8,128,256) raw box encoding
        }


# ---------------------------------------------------------------------------
# CenterPoint box decoder (matches centerpointcodriving.generate_predicted_boxes)
# ---------------------------------------------------------------------------

def decode_centerpoint_boxes(cls_preds_raw: torch.Tensor,
                              reg_preds_raw: torch.Tensor,
                              out_size_factor: int,
                              voxel_size: list,
                              cav_lidar_range: list):
    """Decode CenterPoint raw outputs to (batch, N_boxes, 7) boxes.

    Args:
        cls_preds_raw: (1, 1, H, W) logits
        reg_preds_raw: (1, 8, H, W) regression

    Returns:
        cls_preds_decoded: (1, 1, H, W) — passed as-is to postprocessor
        reg_preds_decoded: (1, H*W, 7) — decoded box coords
    """
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
    # (1, H*W, 7)

    return cls_preds_raw, batch_box_preds


# ---------------------------------------------------------------------------
# Load hypes with numpy tags
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
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default=None,
                   help="TRT collab engine path")
    p.add_argument("--force-fallback", action="store_true",
                   help="Skip TRT, run pure PyTorch (for baseline validation)")
    p.add_argument("--tag", required=True,
                   help="Tag for output files (e.g. fp16, int8, pytorch_baseline)")
    p.add_argument("--n-samples", type=int, default=1789)
    p.add_argument("--report", default=None,
                   help="Output JSON path (default: results/E_codriving_collab_ap_4090.csv appended)")
    p.add_argument("--log", default=None,
                   help="Log file path")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.force_fallback and args.engine is None:
        raise SystemExit("Either --engine or --force-fallback required")

    # Log file
    log_dir = REPO_ROOT / "output/codriving_pilot/collab_engines/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log or str(log_dir / f"ap_eval_{args.tag}.log")

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    log = logging.getLogger(__name__)
    log.info(f"=== CoDriving hybrid AP eval: tag={args.tag} ===")

    # GPU snapshot
    import subprocess
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    log.info(f"GPU snapshot:\n{smi.stdout.strip()}")

    # Setup paths
    sys.path.insert(0, str(V2XVERSE_ROOT))
    os.chdir(V2XVERSE_ROOT)

    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils

    log.info(f"[1/4] Loading hypes from {HYPES_PATH}")
    hypes = load_hypes(str(HYPES_PATH))
    hypes["validate_dir"] = hypes["test_dir"]

    log.info(f"[2/4] Building model + loading checkpoint from {CKPT_DIR}")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(CKPT_DIR), model)
    model = model.cuda().eval()

    # Extract decoder params from model
    out_size_factor = model.out_size_factor        # 2
    voxel_size = model.voxel_size                  # [0.4, 0.4, 5]
    cav_lidar_range = model.cav_lidar_range        # [-102.4, -51.2, -3.5, ...]
    log.info(f"  out_size_factor={out_size_factor}, voxel_size={voxel_size}, "
             f"range={cav_lidar_range}")

    log.info(f"[3/4] Building dataset (DAIR val {args.n_samples} samples)")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    # Load TRT engine (or use fallback)
    trt_engine = None
    if args.force_fallback:
        log.info("[3b] FORCE FALLBACK — all samples through PyTorch")
    else:
        engine_path = args.engine
        if not os.path.isabs(engine_path):
            engine_path = str(REPO_ROOT / engine_path)
        log.info(f"[3b] Loading TRT engine: {engine_path}")
        trt_engine = TrtCollabCodriving(engine_path)

    log.info(f"[4/4] Inference + eval ({args.n_samples} samples)")
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    n_done = 0
    n_trt_path = 0
    n_fallback_path = 0
    n_skipped = 0
    lat_acc = []
    t0 = time.time()

    with torch.inference_mode():
        for batch_data in loader:
            if batch_data is None:
                n_skipped += 1
                continue
            if n_done >= args.n_samples:
                break

            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]

            torch.cuda.synchronize()
            t_a = time.time()

            if args.force_fallback:
                # Pure PyTorch: run full model forward
                output_dict_raw = model(ego)
                # model returns {'cls_preds': ..., 'reg_preds': decoded_boxes, ...}
                # VoxelPostprocessor expects: cls_preds (N,1,H,W), reg_preds (N,H*W,7) or (N,7,H,W)
                # centerpointcodriving returns reg_preds = batch_box_preds (N, H*W, 7)
                output_dict = {"ego": {
                    "cls_preds": output_dict_raw["cls_preds"],
                    "reg_preds": output_dict_raw["reg_preds"],
                }}
                n_fallback_path += 1
            else:
                record_len = ego["record_len"]
                n_agents = int(record_len[0].item())
                if n_agents != 2:
                    # Only 2-agent collab scenarios handled by collab engine
                    # For safety, fall back to PyTorch for edge cases
                    output_dict_raw = model(ego)
                    output_dict = {"ego": {
                        "cls_preds": output_dict_raw["cls_preds"],
                        "reg_preds": output_dict_raw["reg_preds"],
                    }}
                    n_fallback_path += 1
                else:
                    # VFE + scatter
                    voxel_batch = {
                        "voxel_features": ego["processed_lidar"]["voxel_features"],
                        "voxel_coords": ego["processed_lidar"]["voxel_coords"],
                        "voxel_num_points": ego["processed_lidar"]["voxel_num_points"],
                        "record_len": ego["record_len"],
                    }
                    voxel_batch = model.pillar_vfe(voxel_batch)
                    voxel_batch = model.scatter(voxel_batch)
                    spatial_features = voxel_batch["spatial_features"]  # (2,64,256,512)
                    pairwise_t_matrix = ego["pairwise_t_matrix"]        # (1,2,2,4,4)

                    # TRT collab
                    trt_out = trt_engine(spatial_features, pairwise_t_matrix)
                    cls_raw = trt_out["cls_preds_raw"]   # (1,1,128,256)
                    reg_raw = trt_out["reg_preds_raw"]   # (1,8,128,256)

                    # Decode CenterPoint boxes
                    cls_decoded, reg_decoded = decode_centerpoint_boxes(
                        cls_raw, reg_raw, out_size_factor, voxel_size, cav_lidar_range)
                    # reg_decoded: (1, H*W, 7)

                    output_dict = {"ego": {
                        "cls_preds": cls_decoded,
                        "reg_preds": reg_decoded,
                    }}
                    n_trt_path += 1

            torch.cuda.synchronize()
            lat_acc.append((time.time() - t_a) * 1000)

            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
                batch_data, output_dict)

            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor, pred_score, gt_box_tensor,
                    result_stat, iou_th)

            n_done += 1
            if n_done % 100 == 0:
                elapsed = time.time() - t0
                log.info(f"  {n_done:4d}/{args.n_samples}  trt={n_trt_path}  "
                         f"fb={n_fallback_path}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0

    # Compute AP
    out_dir = REPO_ROOT / f"results/E_codriving_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(out_dir))

    log.info(f"\n{'='*60}")
    log.info(f"Tag: {args.tag}")
    log.info(f"Samples: {n_done}  (trt={n_trt_path} fb={n_fallback_path} skip={n_skipped})")
    log.info(f"AP30={ap30:.3f}  AP50={ap50:.3f}  AP70={ap70:.3f}")
    log.info(f"Elapsed: {elapsed:.1f}s  mean_lat={np.mean(lat_acc):.1f}ms  "
             f"p50_lat={np.percentile(lat_acc, 50):.1f}ms")

    # Write result JSON
    rep = {
        "tag": args.tag,
        "engine": args.engine,
        "force_fallback": args.force_fallback,
        "n_samples": n_done,
        "n_trt_path": n_trt_path,
        "n_pytorch_fallback": n_fallback_path,
        "n_skipped": n_skipped,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_secs": elapsed,
        "mean_lat_ms": float(np.mean(lat_acc)),
        "p50_lat_ms": float(np.percentile(lat_acc, 50)),
        "p99_lat_ms": float(np.percentile(lat_acc, 99)),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
    }
    rep_path = REPO_ROOT / f"results/E_codriving_collab_ap_{args.tag}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w") as f:
        json.dump(rep, f, indent=2)
    log.info(f"Report -> {rep_path}")

    # Append to CSV
    csv_path = REPO_ROOT / "results/E_codriving_collab_ap_4090.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("tag,engine,n_samples,n_trt,n_fb,ap30,ap50,ap70,"
                    "mean_lat_ms,p50_lat_ms,p99_lat_ms,elapsed_secs,gpu\n")
        f.write(f"{args.tag},{args.engine},{n_done},{n_trt_path},{n_fallback_path},"
                f"{ap30:.4f},{ap50:.4f},{ap70:.4f},"
                f"{np.mean(lat_acc):.2f},{np.percentile(lat_acc, 50):.2f},"
                f"{np.percentile(lat_acc, 99):.2f},{elapsed:.1f},"
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}\n")
    log.info(f"CSV -> {csv_path}")


if __name__ == "__main__":
    main()
