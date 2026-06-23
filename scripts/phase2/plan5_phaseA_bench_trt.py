"""Plan v5 Phase A.3 — TRT engine build + lat bench via Python API.

trtexec binary is not installed; we wrap IBuilderConfig + IExecutionContext
to (1) build engine per Q (fp16, int8_mm, int8_pc_wo) per ckpt,
(2) run avgRuns=200 warmUp=200 inference, (3) record p50/p99 latency.

For Phase A: bench 54 anchor = 6 plane (g8 baseline + 4 finetuned + g8 extreme)
                              × 3 Q variant
                              × 3 D variant (collapsed from plan v4)
But Phase A.3 focuses on 5 plane × 3 Q × 3 D = 45 cells (extreme plane=3 was
already measured in plan v3; baseline plane=64 is also already in bench v1).

Run after finetune completes:
    python scripts/phase2/plan5_phaseA_bench_trt.py
Output:
    paper_learning/2. AAAI最终故事/data/plan5_phaseA_anchors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import tensorrt as trt

GPU_UTIL_THRESHOLD_PCT = 1
GPU_MEM_USED_THRESHOLD_MB = 500
GPU_STABILITY_SAMPLES = 3
GPU_STABILITY_INTERVAL_SEC = 2.0
DEFAULT_BENCH_GPU = 7


def query_gpu_stats(gpu_id: int) -> Tuple[int, int, int]:
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", f"--id={gpu_id}"],
        text=True, stderr=subprocess.STDOUT,
    )
    parts = [p.strip() for p in out.strip().split(",")]
    return int(parts[0]), int(parts[1]), int(parts[2])


def assert_gpu_isolated(gpu_id: int) -> dict:
    """Strict guard: refuse to bench unless target GPU is FULLY idle.
    util <= 1% AND mem_used < 500 MB, sampled GPU_STABILITY_SAMPLES times
    over GPU_STABILITY_INTERVAL_SEC*N seconds; ALL samples must pass.
    Raises RuntimeError if any sample fails."""
    samples = []
    for i in range(GPU_STABILITY_SAMPLES):
        try:
            idx, util, mem_used = query_gpu_stats(gpu_id)
        except Exception as e:
            raise RuntimeError(f"nvidia-smi query failed for gpu {gpu_id}: {e}")
        samples.append({"util_pct": util, "mem_used_mb": mem_used})
        if util > GPU_UTIL_THRESHOLD_PCT or mem_used > GPU_MEM_USED_THRESHOLD_MB:
            raise RuntimeError(
                f"GPU {gpu_id} NOT fully idle on sample {i+1}/{GPU_STABILITY_SAMPLES} "
                f"(util={util}% > {GPU_UTIL_THRESHOLD_PCT}% or "
                f"mem_used={mem_used} MB > {GPU_MEM_USED_THRESHOLD_MB} MB). "
                f"Latency bench REFUSED — would contaminate results."
            )
        if i < GPU_STABILITY_SAMPLES - 1:
            time.sleep(GPU_STABILITY_INTERVAL_SEC)
    stats = {
        "gpu_id": gpu_id,
        "samples": samples,
        "max_util_pct": max(s["util_pct"] for s in samples),
        "max_mem_used_mb": max(s["mem_used_mb"] for s in samples),
        "stability_window_sec": GPU_STABILITY_INTERVAL_SEC * (GPU_STABILITY_SAMPLES - 1),
    }
    return stats


def auto_find_isolated_gpu(exclude_ids: List[int] = None) -> Optional[int]:
    """Scan all visible GPUs, return id of first one meeting strict isolation
    on a single quick check (full N-sample stability check still happens later)."""
    exclude_ids = exclude_ids or []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        idx, util, mem_used = int(parts[0]), int(parts[1]), int(parts[2])
        if idx in exclude_ids:
            continue
        if util <= GPU_UTIL_THRESHOLD_PCT and mem_used < GPU_MEM_USED_THRESHOLD_MB:
            return idx
    return None

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
FINETUNE_ROOT = Path("/tmp/plan5_phaseA_finetune")
ONNX_DIR = Path("/tmp/plan5_phaseA_onnx")
ENGINE_DIR = Path("/tmp/plan5_phaseA_engines")
CALIB_DIR = REPO / "calibration"

ONNX_DIR.mkdir(parents=True, exist_ok=True)
ENGINE_DIR.mkdir(parents=True, exist_ok=True)

# 4 finetuned + baseline (epoch 19 reference) — extreme plane=3 read from plan v3
ANCHORS_BACKBONE = [
    ("p64_baseline", None,             [64, 128, 256], 19),
    ("p48",         "/tmp/plan5_phaseA_finetune/p48", [48, 96, 192], 27),
    ("p32",         "/tmp/plan5_phaseA_finetune/p32", [32, 64, 128], 27),
    ("p16",         "/tmp/plan5_phaseA_finetune/p16", [16, 32, 64],  27),
    ("p8",          "/tmp/plan5_phaseA_finetune/p8",  [8, 16, 32],   27),
]

Q_VARIANTS = ["fp16", "int8_mm", "int8_pc_wo"]
D_VARIANTS = [
    ("D1_default",        {"opt_level": 3, "workspace_mb": 4096,  "tactics": "default"}),
    ("D2_BL0_default",    {"opt_level": 0, "workspace_mb": 4096,  "tactics": "default"}),
    ("D3_BL0_enableall",  {"opt_level": 0, "workspace_mb": 16384, "tactics": "enable_all"}),
]

INPUT_NAME = "voxel_features"
INPUT_SHAPE = (1, 64, 256, 256)
N_WARMUP = 200
N_RUNS = 200

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, calib_cache: Path, batch_size: int = 1, shape: Tuple[int, ...] = INPUT_SHAPE):
        super().__init__()
        self.calib_cache = calib_cache
        self.batch_size = batch_size
        self.shape = shape
        self.device_input = None
        self.read_only = calib_cache.exists() and calib_cache.stat().st_size > 0

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        return None  # Use cache only — no batch generation

    def read_calibration_cache(self):
        if self.read_only:
            return self.calib_cache.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.calib_cache.write_bytes(cache)


def find_latest_ckpt(model_dir: Path, target_epoch: int) -> Optional[Path]:
    """Find net_epoch_bestval_at<N>.pth or net_epoch<N>.pth closest to target."""
    bestvals = sorted(model_dir.glob("net_epoch_bestval_at*.pth"))
    if bestvals:
        return bestvals[-1]
    direct = model_dir / f"net_epoch{target_epoch}.pth"
    if direct.exists():
        return direct
    epoch_ckpts = sorted(model_dir.glob("net_epoch*.pth"))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    return None


def export_pyramid_backbone_onnx(model_dir: Path, num_filters: List[int], tag: str, target_epoch: int) -> Optional[Path]:
    """Export pyramid_backbone subnet to ONNX. Returns path to onnx file."""
    sys.path.insert(0, str(HEAL_ROOT))
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils

    onnx_path = ONNX_DIR / f"g8_{tag}_pyramid.onnx"
    if onnx_path.exists():
        return onnx_path

    if model_dir is None:
        cfg_path = HEAL_ROOT / "opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/config.yaml"
        cfg = yaml_utils.load_yaml(str(cfg_path))
        ckpt_path = Path("/home/jichengzhi/heal_research/HEAL/opencood/logs/Pyramid_DAIR_m1_base_g8_2026_05_21_22_07_45/net_epoch_bestval_at19.pth")
    else:
        cfg_path = model_dir / "config.yaml"
        cfg = yaml_utils.load_yaml(str(cfg_path))
        ckpt_path = find_latest_ckpt(model_dir, target_epoch)
        if ckpt_path is None:
            print(f"  WARN: no ckpt found for {tag}, skipping ONNX export")
            return None

    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = num_filters
    model = train_utils.create_model(cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    pyramid = model.pyramid_backbone
    pyramid.eval().cuda()

    class BackboneOnly(torch.nn.Module):
        def __init__(self, pyramid):
            super().__init__()
            self.pyramid = pyramid
        def forward(self, x):
            feats = self.pyramid.get_multiscale_feature(x)
            return self.pyramid.decode_multiscale_feature(feats)

    wrapped = BackboneOnly(pyramid).eval()
    dummy = torch.randn(*INPUT_SHAPE, device="cuda")
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=[INPUT_NAME], output_names=["bev_out"],
        opset_version=16, do_constant_folding=True,
        dynamic_axes=None,
    )
    return onnx_path


def build_engine(onnx_path: Path, engine_path: Path, q_variant: str, d_variant: dict,
                 calib_cache: Optional[Path] = None) -> Tuple[bool, dict]:
    """Build TRT engine with given precision+D config."""
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            return False, {"error": "onnx parse failed", "details": errs}

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, d_variant["workspace_mb"] * (1 << 20))
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = d_variant["opt_level"]

    if q_variant == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif q_variant == "int8_mm":
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_cache is None:
            calib_cache = CALIB_DIR / "pyramid_dair_calib_minmax.cache"
        config.int8_calibrator = MinMaxCalibrator(calib_cache=calib_cache)
    elif q_variant == "int8_pc_wo":
        config.set_flag(trt.BuilderFlag.INT8)
        for layer_idx in range(network.num_layers):
            layer = network.get_layer(layer_idx)
            if layer.type == trt.LayerType.CONVOLUTION:
                layer.precision = trt.DataType.INT8
                if hasattr(layer, "get_output"):
                    out = layer.get_output(0)
                    out.dynamic_range = (-127.0, 127.0)

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_secs = time.time() - t0
    if serialized is None:
        return False, {"error": "engine build failed", "build_secs": build_secs}

    engine_path.write_bytes(bytes(serialized))
    return True, {
        "engine_size_mb": round(engine_path.stat().st_size / 1e6, 3),
        "build_secs": round(build_secs, 1),
    }


def bench_engine(engine_path: Path) -> dict:
    """Load engine + run N_WARMUP + N_RUNS inferences, return p50/p99 lat in ms."""
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    h_in = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    d_in = torch.from_numpy(h_in).cuda()
    out_shape = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            out_shape = tuple(context.get_tensor_shape(name))
            break
    if out_shape is None:
        out_shape = (1, 384, 128, 128)
    d_out = torch.zeros(*out_shape, dtype=torch.float32, device="cuda")

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_tensor_address(name, d_in.data_ptr())
        else:
            context.set_tensor_address(name, d_out.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(N_WARMUP):
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
    for i in range(N_RUNS):
        starts[i].record()
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        ends[i].record()
    torch.cuda.synchronize()

    lats_ms = np.array([starts[i].elapsed_time(ends[i]) for i in range(N_RUNS)])
    return {
        "lat_mean": round(float(lats_ms.mean()), 3),
        "lat_p50":  round(float(np.percentile(lats_ms, 50)), 3),
        "lat_p99":  round(float(np.percentile(lats_ms, 99)), 3),
        "lat_std":  round(float(lats_ms.std()), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", nargs="*", default=None,
                        help="Subset of anchor tags to bench (default all)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip engine build (use cached engines)")
    parser.add_argument("--bench-gpu", type=int, default=None,
                        help=f"GPU id for bench. If None, auto-scan for fully idle GPU.")
    parser.add_argument("--no-isolation-check", action="store_true",
                        help="DANGEROUS — skip isolation guard. Only use if you understand contamination risk.")
    args = parser.parse_args()

    if args.no_isolation_check:
        print(f"WARNING: --no-isolation-check set. Latency results may be contaminated.")
        bench_gpu = args.bench_gpu if args.bench_gpu is not None else DEFAULT_BENCH_GPU
    else:
        if args.bench_gpu is None:
            bench_gpu = auto_find_isolated_gpu()
            if bench_gpu is None:
                print(f"ERROR: no fully idle GPU found (util<={GPU_UTIL_THRESHOLD_PCT}%, "
                      f"mem<{GPU_MEM_USED_THRESHOLD_MB}MB). Bench REFUSED.")
                print(f"Suggestion: wait for finetune to complete, or explicitly pass "
                      f"--bench-gpu N to a known idle device.")
                return 2
            print(f"[Plan v5 Phase A.3] auto-selected GPU {bench_gpu}")
        else:
            bench_gpu = args.bench_gpu
        stats = assert_gpu_isolated(bench_gpu)
        print(f"[Plan v5 Phase A.3] GPU {bench_gpu} STRICT isolation PASS: "
              f"max_util={stats['max_util_pct']}% max_mem={stats['max_mem_used_mb']} MB "
              f"({GPU_STABILITY_SAMPLES} samples over {stats['stability_window_sec']:.0f}s)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(bench_gpu)
    print(f"[Plan v5 Phase A.3] pinned to CUDA_VISIBLE_DEVICES={bench_gpu}")

    rows = []
    selected_anchors = [a for a in ANCHORS_BACKBONE if args.anchors is None or a[0] in args.anchors]
    print(f"[Plan v5 Phase A.3] benching {len(selected_anchors)} anchors × {len(Q_VARIANTS)} Q × {len(D_VARIANTS)} D = "
          f"{len(selected_anchors)*len(Q_VARIANTS)*len(D_VARIANTS)} cells")

    for tag, model_dir_str, num_filters, target_epoch in selected_anchors:
        model_dir = Path(model_dir_str) if model_dir_str else None
        if model_dir is not None and not (model_dir / "config.yaml").exists():
            print(f"\n[{tag}] model_dir not ready yet, skipping")
            continue
        print(f"\n[{tag}] num_filters={num_filters}, exporting ONNX ...")
        onnx_path = export_pyramid_backbone_onnx(model_dir, num_filters, tag, target_epoch)
        if onnx_path is None:
            continue

        for q in Q_VARIANTS:
            for d_tag, d_cfg in D_VARIANTS:
                engine_path = ENGINE_DIR / f"g8_{tag}_{q}_{d_tag}.engine"
                if not engine_path.exists() and not args.skip_build:
                    print(f"  [{tag}|{q}|{d_tag}] building engine ...")
                    ok, build_info = build_engine(onnx_path, engine_path, q, d_cfg)
                    if not ok:
                        print(f"    BUILD FAIL: {build_info}")
                        rows.append({
                            "tag": tag, "q": q, "d": d_tag, "num_filters": str(num_filters),
                            "status": "build_failed", **build_info,
                        })
                        continue
                    print(f"    built {build_info['engine_size_mb']} MB in {build_info['build_secs']}s")
                else:
                    build_info = {"engine_size_mb": round(engine_path.stat().st_size / 1e6, 3),
                                  "build_secs": -1}

                lat_stats = bench_engine(engine_path)
                row = {
                    "tag": tag, "q": q, "d": d_tag, "num_filters": str(num_filters),
                    "target_epoch": target_epoch, "status": "ok",
                    **build_info, **lat_stats,
                }
                rows.append(row)
                print(f"  [{tag}|{q}|{d_tag}] p50={lat_stats['lat_p50']} ms p99={lat_stats['lat_p99']} ms")

    out_csv = DATA_DIR / "plan5_phaseA_anchors.csv"
    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[Plan v5 Phase A.3] wrote {len(rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
