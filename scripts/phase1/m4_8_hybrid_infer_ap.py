"""Phase A.3/A.4 — Hybrid PyTorch+TRT inference + AP eval on OPV2V test.

Replaces ``pyramid_backbone + shrink_conv + cls/reg/dir heads`` with a TRT
engine, runs the rest in PyTorch, and computes AP50 on OPV2V test.

This is the *honest* AP measurement that resolves whether INT8 PTQ keeps
detection quality (the M4.8 critical question).

Outputs (one per --engine):
    results/m4_8_hybrid_ap_<tag>.json   { ap30, ap50, ap70, lat_*, n_samples }

Usage::
    python scripts/phase1/m4_8_hybrid_infer_ap.py \
        --engine models/pyramid_m1_subnet_int8.engine \
        --tag trt_int8 \
        --n-samples 2170    # full OPV2V test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(HEAL_ROOT))
os.chdir(HEAL_ROOT)

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.utils import eval_utils  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402
from opencood.tools import inference_utils  # noqa: E402
from opencood.utils.transformation_utils import normalize_pairwise_tfm  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# TRT engine wrapper
# ---------------------------------------------------------------------------

class TrtCollabN2:
    """Phase A.5 collab N=2 TRT engine wrapper.

    Inputs:
        spatial_features (2, 64, H, W) torch float on cuda
        t_ego            (2, 2, 3)     torch float on cuda
    Outputs: cls_preds, reg_preds, dir_preds (each (1, *, H, W))
    """

    def __init__(self, engine_path: str, spatial_shape: tuple = (2, 64, 128, 256),
                 tego_shape: tuple = (2, 2, 3)):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.input_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.output_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
                             if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        self.spatial_name = next(n for n in self.input_names if "spatial" in n.lower())
        self.tego_name = next(n for n in self.input_names if "ego" in n.lower())
        self.spatial_shape = spatial_shape
        self.tego_shape = tego_shape

    def __call__(self, spatial: torch.Tensor, t_ego: torch.Tensor):
        # Fresh context per call — matches the numerically-verified
        # trt_run_collab in m4_8_collab_numerical_check.py. Avoid stale
        # state in re-used context that produced wrong outputs.
        ctx = self.engine.create_execution_context()
        ctx.set_input_shape(self.spatial_name, self.spatial_shape)
        ctx.set_input_shape(self.tego_name, self.tego_shape)

        spatial_in = spatial.float().contiguous()
        tego_in = t_ego.float().contiguous()
        bufs = {self.spatial_name: spatial_in, self.tego_name: tego_in}
        for n in self.output_names:
            shape = tuple(ctx.get_tensor_shape(n))
            bufs[n] = torch.empty(shape, dtype=torch.float32, device="cuda")
        for n in self.input_names + self.output_names:
            ctx.set_tensor_address(n, int(bufs[n].data_ptr()))

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return tuple(bufs[n] for n in self.output_names)


class TrtSubnet:
    """Run pyramid_backbone+shrink+heads via TRT engine, returns torch tensors.

    Input
        x : (1, 64, 256, 256) torch float on CUDA
    Output
        cls_preds (1, 2, 256, 256), reg_preds (1, 14, ...), dir_preds (1, 4, ...)
    """

    def __init__(self, engine_path: str, input_shape: tuple = (1, 64, 256, 256)):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = next(
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        )
        self.output_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        # alloc buffers (fixed shape engine)
        self.context.set_input_shape(self.input_name, input_shape)
        self.bufs: dict[str, torch.Tensor] = {}
        for name in [self.input_name, *self.output_names]:
            shape = tuple(self.context.get_tensor_shape(name))
            self.bufs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name, int(self.bufs[name].data_ptr()))
        self.stream = torch.cuda.Stream()

    def __call__(self, x: torch.Tensor):
        # x: (1, 64, 256, 256) float32 cuda
        self.bufs[self.input_name].copy_(x)
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return tuple(self.bufs[n].clone() for n in self.output_names)


# ---------------------------------------------------------------------------
# Hybrid forward
# ---------------------------------------------------------------------------

def hybrid_forward(model, batch_data, trt_subnet=None, trt_collab=None,
                   tile_n_to_one: bool = True):
    """Run HeterPyramidCollab.forward but with sub-module replaced by TRT.

    PyTorch parts kept:
      encoder_m1, backbone_m1, aligner_m1, weighted_fuse (multi-agent fusion).

    TRT engine replaces:
      pyramid_backbone.forward_collab + shrink + cls/reg/dir heads.

    To keep things simple, we run TRT on the *fused per-batch feature map*
    after PyTorch-side weighted_fuse, i.e. exactly the input shape the engine
    was built for: (1, 64, 256, 256). This means multi-agent fusion happens
    in PyTorch (cheap) and the heavy ResNeXt body runs in TRT.

    Wait — that doesn't match: pyramid_backbone takes per-agent features and
    fuses INSIDE its forward_collab via multiscale weighted fuse. Replicating
    that with a single-agent TRT engine requires:
       (a) doing weighted_fuse on the *input* (64-channel) feature map first,
           then running TRT on the fused single-tensor.
       (b) accepting that this changes the algorithm slightly (single-scale
           fusion at backbone-input rather than multiscale fusion inside
           pyramid_backbone).

    For an honest INT8 baseline we want to keep the same algorithm. The
    simplest path: run TRT once per agent, use the cls_preds as occ score
    (last output channel), then weighted-fuse outputs in PyTorch.

    But cls_preds is post-head. The native code uses single_head_{i} (per-scale
    occ heads) BEFORE deblock+head. Those heads aren't in our TRT engine
    (we dropped them since they're train-time supervision).

    Compromise: when the whole sample is single-agent (record_len == [1]),
    TRT runs as-is and matches PyTorch exactly. For multi-agent samples, we
    fall back to PyTorch (since the algorithm requires multi-scale weighted
    fuse). We report what % of test samples are single-agent so the AP gap
    is interpretable.
    """
    ego = batch_data["ego"]
    record_len = ego["record_len"]
    agent_modality_list = ego["agent_modality_list"]
    affine_matrix = normalize_pairwise_tfm(
        ego["pairwise_t_matrix"], model.H, model.W, model.fake_voxel_size,
    )
    modality_count = Counter(agent_modality_list)

    # Mirror HeterPyramidCollab.forward: encoder + backbone + aligner per modality
    modality_feature_dict: dict[str, torch.Tensor] = {}
    for m in model.modality_name_list:
        if m not in modality_count:
            continue
        feat = getattr(model, f"encoder_{m}")(ego, m)
        feat = getattr(model, f"backbone_{m}")({"spatial_features": feat})["spatial_features_2d"]
        feat = getattr(model, f"aligner_{m}")(feat)
        modality_feature_dict[m] = feat

    counting = {m: 0 for m in model.modality_name_list}
    heter_list = []
    for m in agent_modality_list:
        idx = counting[m]
        heter_list.append(modality_feature_dict[m][idx])
        counting[m] += 1
    heter_feat_2d = torch.stack(heter_list)  # (sum_cav, 64, 256, 256)

    # Routing:
    #   N=1 + trt_subnet → single-agent TRT engine
    #   N=2 + trt_collab → collab N=2 TRT engine (Phase A.5)
    #   else            → PyTorch fallback (forward_collab)
    n_agents = heter_feat_2d.shape[0]
    if n_agents == 1 and trt_subnet is not None:
        cls_p, reg_p, dir_p = trt_subnet(heter_feat_2d.contiguous())
        return {"cls_preds": cls_p, "reg_preds": reg_p, "dir_preds": dir_p,
                "occ_single_list": [], "_path": "trt_single"}
    elif n_agents == 2 and trt_collab is not None:
        # affine_matrix: (B, L, L, 2, 3); pick t for ego (b=0, ego_idx=0, both agents)
        # equivalent to weighted_fuse: t_matrix[ego_idx, :N, :, :]
        t_ego = affine_matrix[0, 0, :2, :, :].contiguous()  # (2, 2, 3)
        cls_p, reg_p, dir_p = trt_collab(heter_feat_2d.contiguous(), t_ego)
        return {"cls_preds": cls_p, "reg_preds": reg_p, "dir_preds": dir_p,
                "occ_single_list": [], "_path": "trt_collab"}
    else:
        # PyTorch fallback for multi-agent (true forward_collab)
        fused, occ_outputs = model.pyramid_backbone.forward_collab(
            heter_feat_2d, record_len, affine_matrix, agent_modality_list, model.cam_crop_info,
        )
        if model.shrink_flag:
            fused = model.shrink_conv(fused)
        return {
            "cls_preds": model.cls_head(fused),
            "reg_preds": model.reg_head(fused),
            "dir_preds": model.dir_head(fused),
            "occ_single_list": occ_outputs,
            "_path": "pytorch_fallback",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default=None,
                   help="single-agent TRT engine path (record_len=[1] samples)")
    p.add_argument("--engine-collab", default=None,
                   help="collab N=2 TRT engine path (record_len=[2] samples, Phase A.5)")
    p.add_argument("--wrapper-pytorch", action="store_true",
                   help="Use PyramidCollabSubnetN2 PyTorch wrapper for N=2 (sanity isolation)")
    p.add_argument("--force-fallback", action="store_true",
                   help="Skip TRT engine, force all samples through PyTorch — sanity baseline")
    p.add_argument("--input-shape", default="1,64,256,256",
                   help="TRT engine input shape (B,C,H,W). OPV2V: 1,64,256,256. DAIR: 1,64,128,256")
    p.add_argument("--collab-spatial-shape", default="2,64,128,256",
                   help="collab engine spatial input shape (DAIR default)")
    p.add_argument("--collab-tego-shape", default="2,2,3",
                   help="collab engine t_ego input shape")
    p.add_argument("--tag", required=True)
    p.add_argument("--model-dir", default="/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_m1_base_2023_08_14_04_28_12")
    p.add_argument("--n-samples", type=int, default=2170)
    p.add_argument("--range", default="102.4,102.4")
    p.add_argument("--dataset", choices=["opv2v", "dair"], default="opv2v",
                   help="affects calibrator behaviour and TRT subnet input shape")
    p.add_argument("--report", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.force_fallback and not args.wrapper_pytorch and args.engine is None and args.engine_collab is None:
        raise SystemExit("--engine, --engine-collab, --wrapper-pytorch, or --force-fallback required")
    # Resolve relative paths BEFORE the cwd switch above (which already happened
    # at module import). Re-anchor against REPO_ROOT.
    if args.engine and not os.path.isabs(args.engine):
        args.engine = str(REPO_ROOT / args.engine)
    if args.engine_collab and not os.path.isabs(args.engine_collab):
        args.engine_collab = str(REPO_ROOT / args.engine_collab)
    if args.report is None:
        args.report = str(REPO_ROOT / f"results/m4_8_hybrid_ap_{args.tag}.json")
    elif not os.path.isabs(args.report):
        args.report = str(REPO_ROOT / args.report)

    hypes = yaml_utils.load_yaml(str(Path(args.model_dir) / "config.yaml"))
    if "heter" in hypes:
        x_max = float(args.range.split(",")[0])
        y_max = float(args.range.split(",")[1])
        new_range = [
            -x_max, -y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max, y_max, hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(hypes, {
            "cav_lidar_range": new_range, "lidar_range": new_range, "gt_range": new_range,
        })
        import importlib
        yu = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        hypes = getattr(yu, hypes["yaml_parser"])(hypes)
    hypes["validate_dir"] = hypes["test_dir"]

    print(f"[1/4] build model + load ckpt")
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(args.model_dir, model)
    model.cuda().eval()

    print(f"[2/4] build dataset")
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=2,
                        collate_fn=dataset.collate_batch_test, shuffle=False)

    trt_subnet = None
    trt_collab = None
    if args.force_fallback:
        print(f"[3/4] FORCE FALLBACK — no TRT engine, all samples go through PyTorch")
    elif args.wrapper_pytorch:
        print(f"[3/4 SANITY] using PyramidCollabSubnetN2 PyTorch wrapper (no TRT)")
        sys.path.insert(0, str(REPO_ROOT))
        from tools.export_onnx_pyramid_collab import PyramidCollabSubnetN2
        wrapper_module = PyramidCollabSubnetN2(model, align_corners=False).cuda().eval()
        class WrapperAsTrt:
            def __call__(self, spatial, t_ego):
                return wrapper_module(spatial, t_ego)
        trt_collab = WrapperAsTrt()
    else:
        if args.engine:
            print(f"[3/4a] load single-agent TRT engine: {args.engine}")
            in_shape = tuple(int(x) for x in args.input_shape.split(","))
            trt_subnet = TrtSubnet(args.engine, input_shape=in_shape)
        if args.engine_collab:
            print(f"[3/4b] load collab N=2 TRT engine: {args.engine_collab}")
            sshape = tuple(int(x) for x in args.collab_spatial_shape.split(","))
            tshape = tuple(int(x) for x in args.collab_tego_shape.split(","))
            trt_collab = TrtCollabN2(args.engine_collab, spatial_shape=sshape, tego_shape=tshape)

    print(f"[4/4] run inference + eval ({args.n_samples} samples)")
    result_stat = {0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
                   0.7: {"tp": [], "fp": [], "gt": 0, "score": []}}
    n_done = 0
    n_trt_single = 0
    n_trt_collab = 0
    n_fb = 0
    lat_acc = []
    t0 = time.time()
    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None:
                continue
            if n_done >= args.n_samples:
                break
            batch_data = train_utils.to_device(batch_data, "cuda")

            torch.cuda.synchronize()
            t_a = time.time()
            output_dict = hybrid_forward(model, batch_data, trt_subnet, trt_collab)
            torch.cuda.synchronize()
            lat_acc.append((time.time() - t_a) * 1000)

            path = output_dict["_path"]
            if path == "trt_single":
                n_trt_single += 1
            elif path == "trt_collab":
                n_trt_collab += 1
            else:
                n_fb += 1

            # Match HEAL inference_utils.inference_intermediate_fusion contract
            # for post-process — we replicate the post-process logic here.
            output_wrapped = {"ego": output_dict}
            pred_box_tensor, pred_score, gt_box_tensor = \
                dataset.post_process(batch_data, output_wrapped)

            for iou_th in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor,
                                           result_stat, iou_th)
            n_done += 1
            if (n_done % 100) == 0:
                elapsed = time.time() - t0
                print(f"  {n_done:4d}/{args.n_samples}  trt_single={n_trt_single} "
                      f"trt_collab={n_trt_collab} fb={n_fb}  elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    out_dir = REPO_ROOT / f"results/m4_8_eval_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(out_dir))

    rep = {
        "tag": args.tag,
        "engine": args.engine,
        "engine_collab": args.engine_collab,
        "n_samples": n_done,
        "n_trt_single_path": n_trt_single,
        "n_trt_collab_path": n_trt_collab,
        "n_trt_path": n_trt_single + n_trt_collab,
        "n_pytorch_fallback": n_fb,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_secs": elapsed,
        "mean_per_sample_lat_ms": float(np.mean(lat_acc)),
        "p50_per_sample_lat_ms": float(np.percentile(lat_acc, 50)),
        "p99_per_sample_lat_ms": float(np.percentile(lat_acc, 99)),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\n{rep}\n  -> {args.report}")


if __name__ == "__main__":
    main()
