"""E2E TRT latency bench for HEAL Pyramid Fusion (DAIR-V2X).

Chain (CUDA Event wraps the whole thing — TRT-affected layers ≈ 100% forward):

    voxelize done by HEAL DataLoader (CPU, not timed) ←─ shared across configs
        ↓
    [CUDA Event start]
        ↓
    PyTorch: pad voxel_features/num_points/coords to MAX_VOX
        + compute t_ego from pairwise_t_matrix
        ↓
    TRT engine: VFE + Scatter + backbone_m1 + aligner_m1
                + pyramid_backbone (collab N=2) + shrink_conv + heads
        ↓
    PyTorch: post_processor.post_process
        (delta_to_boxes3d + corners + mmcv CUDA NMS + range_mask)
        ↓
    [CUDA Event end]

Output: lat_e2e_ms (mean/p50/p99) + lat_trt_ms (engine alone)
        + lat_postproc_ms (postproc alone)
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
if str(HEAL_ROOT) not in sys.path:
    sys.path.insert(0, str(HEAL_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/phase2"))
os.chdir(str(HEAL_ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils  # noqa: E402
from opencood.tools import train_utils  # noqa: E402
from opencood.data_utils.datasets import build_dataset  # noqa: E402
from opencood.utils.transformation_utils import normalize_pairwise_tfm  # noqa: E402
from opencood.utils.common_utils import update_dict  # noqa: E402
from opencood.utils import box_utils  # noqa: E402

from m4_8_cuda_kernel_replacements import nms_rotated_mmcv  # noqa: E402

# Monkey-patch read_json to no-op on missing label_world_backup files
# (HEAL DAIR loader tries to load these but they're not required for inference)
import opencood.utils.common_utils as _cu  # noqa: E402
_orig_read_json = _cu.read_json


def _safe_read_json(path):
    # HEAL DAIR loader tries to read fallback "*_backup*" label files that
    # don't exist in our split; safely return empty list since we're doing
    # latency bench only (no AP eval, no GT needed).
    if not os.path.exists(path) and ("backup" in str(path) or "label" in str(path).split("/")[-2:]):
        return []
    return _orig_read_json(path)


_cu.read_json = _safe_read_json
# Patch in dair basedataset's namespace too
import opencood.data_utils.datasets.basedataset.dairv2x_basedataset as _dair_mod  # noqa: E402
_dair_mod.read_json = _safe_read_json

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ---------------------------------------------------------------------------
# Patch NMS to mmcv CUDA
# ---------------------------------------------------------------------------


def install_cuda_nms():
    box_utils.nms_rotated = nms_rotated_mmcv
    import opencood.data_utils.post_processor.voxel_postprocessor as vp_mod
    vp_mod.box_utils.nms_rotated = nms_rotated_mmcv
    print("[opt] nms_rotated -> mmcv CUDA NMS")


# ---------------------------------------------------------------------------
# TRT engine wrapper
# ---------------------------------------------------------------------------


class TrtEngine:
    """Minimal TRT 10 engine wrapper for the e2e Pyramid forward."""

    def __init__(self, engine_path: str, max_voxels: int):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.max_voxels = max_voxels

        n_io = self.engine.num_io_tensors
        names = [self.engine.get_tensor_name(i) for i in range(n_io)]
        self.input_names = [n for n in names
                            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        # Set input shapes (all static for this engine)
        for n in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(n))
            self.context.set_input_shape(n, shape)

        # Pre-allocate buffers (will fill in run())
        self.bufs: dict[str, torch.Tensor] = {}
        for n in names:
            shape = tuple(self.context.get_tensor_shape(n))
            dtype = {
                trt.float32: torch.float32, trt.float16: torch.float16,
                trt.int32: torch.int32, trt.int64: torch.int64,
                trt.int8: torch.int8, trt.bool: torch.bool,
            }.get(self.engine.get_tensor_dtype(n), torch.float32)
            self.bufs[n] = torch.empty(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(n, int(self.bufs[n].data_ptr()))

        self.input_shapes = {n: tuple(self.bufs[n].shape) for n in self.input_names}
        print(f"[trt] engine loaded: max_voxels={max_voxels}, "
              f"inputs={[(n, tuple(self.bufs[n].shape)) for n in self.input_names]}")

    def run(self, voxel_features, voxel_num_points, voxel_coords,
            voxel_mask, t_ego, stream: torch.cuda.Stream):
        """Async run on stream. Caller manages timing.

        All inputs must already be on cuda; will be cast to engine dtypes.
        """
        feeds = {
            "voxel_features": voxel_features,
            "voxel_num_points": voxel_num_points,
            "voxel_coords": voxel_coords,
            "voxel_mask": voxel_mask,
            "t_ego": t_ego,
        }
        for n, x in feeds.items():
            buf = self.bufs[n]
            if buf.dtype != x.dtype:
                x = x.to(buf.dtype)
            buf.copy_(x, non_blocking=True)
        self.context.execute_async_v3(stream.cuda_stream)
        return (self.bufs["cls_preds"], self.bufs["reg_preds"], self.bufs["dir_preds"])


# ---------------------------------------------------------------------------
# Voxel input padding
# ---------------------------------------------------------------------------


def pad_voxels(voxel_features, voxel_num_points, voxel_coords, max_voxels):
    """Pad/truncate voxel inputs to fixed MAX_VOX. Returns padded + mask."""
    M_real = voxel_features.shape[0]
    P = voxel_features.shape[1]
    device = voxel_features.device

    if M_real == max_voxels:
        mask = torch.ones(M_real, device=device, dtype=torch.float32)
        return voxel_features, voxel_num_points, voxel_coords, mask, M_real

    if M_real > max_voxels:
        # Truncate (rare; max_voxel_test=70000 in DAIR config is conservative)
        return (voxel_features[:max_voxels].contiguous(),
                voxel_num_points[:max_voxels].contiguous(),
                voxel_coords[:max_voxels].contiguous(),
                torch.ones(max_voxels, device=device, dtype=torch.float32),
                max_voxels)

    pad = max_voxels - M_real
    voxel_features = torch.cat([
        voxel_features,
        torch.zeros(pad, P, voxel_features.shape[2],
                    device=device, dtype=voxel_features.dtype)
    ], dim=0)
    voxel_num_points = torch.cat([
        voxel_num_points,
        torch.zeros(pad, device=device, dtype=voxel_num_points.dtype)
    ], dim=0)
    voxel_coords = torch.cat([
        voxel_coords,
        torch.zeros(pad, 4, device=device, dtype=voxel_coords.dtype)
    ], dim=0)
    mask = torch.zeros(max_voxels, device=device, dtype=torch.float32)
    mask[:M_real] = 1.0
    return voxel_features, voxel_num_points, voxel_coords, mask, M_real


# ---------------------------------------------------------------------------
# Light model loading (for post_processor + H/W/fake_voxel_size)
# ---------------------------------------------------------------------------


def load_meta(ckpt_dir: str, dair_root: str):
    class _Opt:
        model_dir = ckpt_dir
    opt_obj = _Opt()
    hypes = yaml_utils.load_yaml(None, opt_obj)
    new_range = [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]  # DAIR matches config
    hypes = update_dict(hypes, {
        "cav_lidar_range": new_range,
        "lidar_range": new_range,
        "gt_range": new_range,
    })
    # HEAL DAIR expects:
    #   data_dir = root with cooperative/ vehicle-side/ infrastructure-side/
    #   validate_dir = path to val.json (list of frame IDs)
    val_split = str(Path(dair_root) / "val.json")
    hypes["data_dir"] = dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    import importlib
    yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
    parser_func = getattr(yaml_utils_lib, hypes["yaml_parser"])
    hypes = parser_func(hypes)
    # Re-assert val.json paths after parser (it may reset)
    hypes["data_dir"] = dair_root
    hypes["test_dir"] = val_split
    hypes["validate_dir"] = val_split
    hypes["root_dir"] = val_split

    # Build light model just for post_processor / H / W / fake_voxel_size
    model = train_utils.create_model(hypes)
    H = model.H
    W = model.W
    fake_vs = model.fake_voxel_size
    del model
    torch.cuda.empty_cache()
    return hypes, H, W, fake_vs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def stats(arr):
    if not arr:
        return {"n": 0}
    a = np.asarray(arr)
    return {
        "n": len(arr),
        "mean": float(a.mean()), "std": float(a.std()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()), "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt-dir", required=True,
                    help="HEAL ckpt directory (for post_processor + H/W/fake_voxel_size)")
    ap.add_argument("--dair-root",
                    default="/data/DAIR-V2X/DAIR-V2X-C/cooperative-vehicle-infrastructure")
    ap.add_argument("--max-voxels", type=int, default=70000)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-measure", type=int, default=80)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    # Setup
    install_cuda_nms()
    print(f"[boot] loading meta: ckpt-dir={args.ckpt_dir}")
    hypes, H, W, fake_vs = load_meta(args.ckpt_dir, args.dair_root)
    print(f"[boot] H={H}, W={W}, fake_voxel_size={fake_vs}")

    ds = build_dataset(hypes, visualize=False, train=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds, batch_size=1, num_workers=2,
        collate_fn=ds.collate_batch_test, shuffle=False, pin_memory=False,
    )
    post_processor = ds.post_processor
    print(f"[boot] dataset size: {len(ds)}")

    engine = TrtEngine(args.engine, args.max_voxels)
    stream = torch.cuda.Stream()

    e_start = torch.cuda.Event(enable_timing=True)
    e_post = torch.cuda.Event(enable_timing=True)
    e_end = torch.cuda.Event(enable_timing=True)
    e_trt_start = torch.cuda.Event(enable_timing=True)
    e_trt_end = torch.cuda.Event(enable_timing=True)

    lats_e2e: list[float] = []
    lats_trt: list[float] = []
    lats_postproc: list[float] = []
    real_voxel_counts: list[int] = []

    total = args.n_warmup + args.n_measure
    device = torch.device("cuda")
    n_done = 0
    n_skip = 0

    print(f"[bench] target {args.n_warmup} warmup + {args.n_measure} measure samples")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if n_done >= total:
                break
            if batch is None:
                n_skip += 1
                continue
            batch = train_utils.to_device(batch, device)
            data_dict = batch["ego"]

            # HEAL intermediate heter fusion stores voxel data under inputs_m1
            if n_done == 0:
                print(f"[bench] batch keys: {list(data_dict.keys())[:20]}")
                if "inputs_m1" in data_dict:
                    print(f"[bench] inputs_m1 keys: {list(data_dict['inputs_m1'].keys())}")
            voxel_data = data_dict.get("inputs_m1") or data_dict.get("processed_lidar")
            if voxel_data is None:
                raise KeyError(f"no voxel data found in batch; keys={list(data_dict.keys())}")
            voxel_features = voxel_data["voxel_features"]
            voxel_num_points = voxel_data["voxel_num_points"]
            voxel_coords = voxel_data["voxel_coords"].int()
            pairwise_t = data_dict["pairwise_t_matrix"]
            record_len = data_dict["record_len"]

            if voxel_features.shape[0] > args.max_voxels:
                # Truncation reduces accuracy of timing comparison; skip
                n_skip += 1
                continue

            # Compute t_ego: affine_matrix[0, 0, :N=2]
            affine_matrix = normalize_pairwise_tfm(pairwise_t, H, W, fake_vs)
            N_b = int(record_len[0].item()) if isinstance(record_len, torch.Tensor) \
                else int(record_len[0])
            if N_b != 2:
                # Engine fixed N=2 (ego + 1 collab). DAIR should always have 2.
                n_skip += 1
                continue
            t_ego = affine_matrix[0][0][:2].contiguous()  # (2, 2, 3)

            # Pad voxel inputs
            (vf, vnp, vc, vmask, m_real) = pad_voxels(
                voxel_features, voxel_num_points, voxel_coords, args.max_voxels
            )

            torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                e_start.record(stream)
                e_trt_start.record(stream)
                cls, reg, dir_ = engine.run(vf, vnp, vc, vmask, t_ego, stream)
                e_trt_end.record(stream)

                output_dict = {"ego": {
                    "cls_preds": cls,
                    "reg_preds": reg,
                    "dir_preds": dir_,
                }}
                e_post.record(stream)
                pred_box, scores = post_processor.post_process(batch, output_dict)
                e_end.record(stream)
            stream.synchronize()

            lat_e2e_ms = e_start.elapsed_time(e_end)
            lat_trt_ms = e_trt_start.elapsed_time(e_trt_end)
            lat_post_ms = e_post.elapsed_time(e_end)

            if n_done >= args.n_warmup:
                lats_e2e.append(lat_e2e_ms)
                lats_trt.append(lat_trt_ms)
                lats_postproc.append(lat_post_ms)
                real_voxel_counts.append(m_real)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  [{n_done}/{total}] e2e={lat_e2e_ms:.2f}ms "
                      f"trt={lat_trt_ms:.2f}ms post={lat_post_ms:.2f}ms "
                      f"M={m_real}", flush=True)

    report = {
        "engine": args.engine,
        "ckpt_dir": args.ckpt_dir,
        "max_voxels": args.max_voxels,
        "n_warmup": args.n_warmup,
        "n_measure": args.n_measure,
        "n_collected": len(lats_e2e),
        "n_skipped": n_skip,
        "real_voxels": stats(real_voxel_counts),
        "lat_e2e_ms": stats(lats_e2e),
        "lat_trt_ms": stats(lats_trt),
        "lat_postproc_ms": stats(lats_postproc),
        "device": torch.cuda.get_device_name(0),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print("\n=== e2e bench summary ===")
    print(f"  collected {len(lats_e2e)} samples (skipped {n_skip})")
    print(f"  lat_e2e:      mean={report['lat_e2e_ms']['mean']:.3f}ms  "
          f"p50={report['lat_e2e_ms']['p50']:.3f}  p99={report['lat_e2e_ms']['p99']:.3f}")
    print(f"  lat_trt:      mean={report['lat_trt_ms']['mean']:.3f}ms  "
          f"p50={report['lat_trt_ms']['p50']:.3f}  p99={report['lat_trt_ms']['p99']:.3f}")
    print(f"  lat_postproc: mean={report['lat_postproc_ms']['mean']:.3f}ms  "
          f"p50={report['lat_postproc_ms']['p50']:.3f}")
    print(f"  real voxels:  mean={report['real_voxels']['mean']:.0f}  "
          f"p99={report['real_voxels']['p99']:.0f}")
    print(f"\nreport: {args.report}")


if __name__ == "__main__":
    main()
