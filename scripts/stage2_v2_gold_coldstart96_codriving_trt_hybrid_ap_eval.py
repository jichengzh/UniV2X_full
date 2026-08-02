#!/usr/bin/env python3
"""Hybrid AP evaluation for v2 CoDriving TensorRT collab engines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REMOTE_REPO = Path("/exdata/jichengzhi/V2Xverse_pyramid")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_report(
    *,
    width: str,
    tag: str,
    model_dir: Path,
    engine: Path | None,
    ap30: float,
    ap50: float,
    ap70: float,
    n_done: int,
    n_trt_path: int,
    n_fallback_path: int,
    n_skipped: int,
    elapsed_secs: float,
) -> dict[str, Any]:
    return {
        "schema": "v2_gold_coldstart_96_codriving_trt_hybrid_ap_eval_v1",
        "created_at_utc": utc_now(),
        "width": width,
        "tag": tag,
        "model_dir": str(model_dir),
        "engine": str(engine) if engine is not None else None,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "n_done": int(n_done),
        "n_trt_path": int(n_trt_path),
        "n_fallback_path": int(n_fallback_path),
        "n_skipped": int(n_skipped),
        "elapsed_secs": float(elapsed_secs),
    }


class TrtCollabCodriving:
    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(f"failed to create TRT execution context: {engine_path}")
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        self.spatial_name = next(name for name in self.input_names if "spatial" in name.lower())
        self.tmat_name = next(
            name for name in self.input_names if "pairwise" in name.lower() or "t_matrix" in name.lower()
        )
        self.stream = torch.cuda.Stream()

    def __call__(self, spatial, pairwise_t_matrix) -> dict[str, Any]:
        spatial_in = spatial.float().contiguous()
        tmat_in = pairwise_t_matrix.float().contiguous()
        self.ctx.set_input_shape(self.spatial_name, tuple(spatial_in.shape))
        self.ctx.set_input_shape(self.tmat_name, tuple(tmat_in.shape))
        buffers = {
            self.spatial_name: spatial_in,
            self.tmat_name: tmat_in,
        }
        for name in self.output_names:
            buffers[name] = self.torch.empty(
                tuple(self.ctx.get_tensor_shape(name)), dtype=self.torch.float32, device="cuda"
            )
        for name, tensor in buffers.items():
            self.ctx.set_tensor_address(name, int(tensor.data_ptr()))
        with self.torch.cuda.stream(self.stream):
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        cls_name = next(name for name in self.output_names if "cls" in name.lower())
        reg_name = next(name for name in self.output_names if "reg" in name.lower())
        return {"cls_preds_raw": buffers[cls_name], "reg_preds_raw": buffers[reg_name]}


def decode_centerpoint_boxes(cls_preds_raw, reg_preds_raw, out_size_factor, voxel_size, cav_lidar_range):
    import torch

    box_preds = reg_preds_raw.permute(0, 2, 3, 1).contiguous()
    batch, height, width, code_size = box_preds.size()
    box_preds = box_preds.reshape(batch, height * width, code_size)
    batch_reg = box_preds[..., 0:2]
    h = box_preds[..., 3:4] * out_size_factor * voxel_size[0]
    w = box_preds[..., 4:5] * out_size_factor * voxel_size[1]
    l = box_preds[..., 5:6] * out_size_factor * voxel_size[2]
    batch_dim = torch.cat([h, w, l], dim=-1)
    batch_hei = box_preds[..., 2:3] * out_size_factor * voxel_size[2] + cav_lidar_range[2]
    rot = torch.atan2(box_preds[..., 6:7], box_preds[..., 7:8])
    ys, xs = torch.meshgrid(
        torch.arange(0, height, device=cls_preds_raw.device),
        torch.arange(0, width, device=cls_preds_raw.device),
        indexing="ij",
    )
    ys = ys.view(1, height, width).repeat(batch, 1, 1).view(batch, -1, 1)
    xs = xs.view(1, height, width).repeat(batch, 1, 1).view(batch, -1, 1)
    xs = (xs + batch_reg[:, :, 0:1]) * out_size_factor * voxel_size[0] + cav_lidar_range[0]
    ys = (ys + batch_reg[:, :, 1:2]) * out_size_factor * voxel_size[1] + cav_lidar_range[1]
    batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
    return cls_preds_raw, batch_box_preds


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    repo_root = Path(args.repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))

    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils

    hypes = load_yaml(str(args.model_dir / "config.yaml"))
    hypes["validate_dir"] = hypes["test_dir"]
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.model_dir), model)
    model = model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    trt_engine = None if args.force_fallback else TrtCollabCodriving(args.engine)
    voxel_size = hypes["preprocess"]["args"]["voxel_size"]
    cav_lidar_range = list(hypes["preprocess"]["cav_lidar_range"])
    out_size_factor = hypes["model"]["args"]["out_size_factor"]
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    n_done = 0
    n_trt_path = 0
    n_fallback_path = 0
    n_skipped = 0
    started = time.time()
    with torch.inference_mode():
        for batch_data in loader:
            if args.n_samples > 0 and n_done >= args.n_samples:
                break
            if batch_data is None:
                n_skipped += 1
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            n_agents = int(ego["record_len"][0].item())
            if trt_engine is None or n_agents != 2:
                output_raw = model(ego)
                output_dict = {"ego": {"cls_preds": output_raw["cls_preds"], "reg_preds": output_raw["reg_preds"]}}
                n_fallback_path += 1
            else:
                voxel_batch = {
                    "voxel_features": ego["processed_lidar"]["voxel_features"],
                    "voxel_coords": ego["processed_lidar"]["voxel_coords"],
                    "voxel_num_points": ego["processed_lidar"]["voxel_num_points"],
                    "record_len": ego["record_len"],
                }
                voxel_batch = model.pillar_vfe(voxel_batch)
                voxel_batch = model.scatter(voxel_batch)
                trt_out = trt_engine(voxel_batch["spatial_features"], ego["pairwise_t_matrix"])
                cls_decoded, reg_decoded = decode_centerpoint_boxes(
                    trt_out["cls_preds_raw"],
                    trt_out["reg_preds_raw"],
                    out_size_factor,
                    voxel_size,
                    cav_lidar_range,
                )
                output_dict = {"ego": {"cls_preds": cls_decoded, "reg_preds": reg_decoded}}
                n_trt_path += 1

            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
            for iou in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou)
            n_done += 1
            if n_done % args.progress_every == 0:
                print(
                    f"[progress] {args.width} {args.tag} n={n_done} trt={n_trt_path} "
                    f"fallback={n_fallback_path} skipped={n_skipped} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    args.eval_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(args.eval_dir), args.tag)
    report = build_report(
        width=args.width,
        tag=args.tag,
        model_dir=args.model_dir,
        engine=args.engine if not args.force_fallback else None,
        ap30=ap30,
        ap50=ap50,
        ap70=ap70,
        n_done=n_done,
        n_trt_path=n_trt_path,
        n_fallback_path=n_fallback_path,
        n_skipped=n_skipped,
        elapsed_secs=time.time() - started,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REMOTE_REPO)
    parser.add_argument("--width", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--n-samples", type=int, default=1789)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if not args.force_fallback and args.engine is None:
        parser.error("--engine is required unless --force-fallback is set")
    return args


def main() -> int:
    run_eval(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
