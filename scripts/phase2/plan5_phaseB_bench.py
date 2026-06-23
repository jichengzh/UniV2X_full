"""Plan v5 Phase B.2-3 — Re-mask + TRT --sparsity bench.

For each finetuned sparsity ckpt:
  1. Re-apply 2:4 mask (recover from FT drift)
  2. ONNX export pyramid_backbone subnet
  3. Build TRT engine with BuilderFlag.SPARSE_WEIGHTS + INT8/FP16
  4. Bench lat with strict GPU isolation
  5. Compare against Phase A dense lat (lookup plan5_phaseA_anchors.csv)

Run:
    python scripts/phase2/plan5_phaseB_bench.py --bench-gpu 2
Output:
    paper_learning/2. AAAI最终故事/data/plan5_phaseB_anchors.csv
    paper_learning/2. AAAI最终故事/data/plan5_phaseB_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tensorrt as trt

REPO = Path("/home/jichengzhi/UniV2X")
DATA_DIR = REPO / "paper_learning" / "2. AAAI最终故事" / "data"
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
PHASE_B_ROOT = Path("/tmp/plan5_phaseB_sparse")
ONNX_DIR = Path("/tmp/plan5_phaseB_onnx")
ENGINE_DIR = Path("/tmp/plan5_phaseB_engines")
CALIB_DIR = REPO / "calibration"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
ENGINE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HEAL_ROOT))

GPU_UTIL_THRESHOLD_PCT = 1
GPU_MEM_USED_THRESHOLD_MB = 500
INPUT_NAME = "voxel_features"
INPUT_SHAPE = (1, 64, 256, 256)
N_WARMUP = 200
N_RUNS = 200
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

SPARSE_ANCHORS = [
    ("p64", "/tmp/plan5_phaseB_sparse/p64", "net_epoch20.pth", [64, 128, 256]),
    ("p48", "/tmp/plan5_phaseB_sparse/p48", "net_epoch28.pth", [48, 96, 192]),
    ("p32", "/tmp/plan5_phaseB_sparse/p32", "net_epoch24.pth", [32, 64, 128]),
    ("p16", "/tmp/plan5_phaseB_sparse/p16", "net_epoch28.pth", [16, 32, 64]),
    ("p8",  "/tmp/plan5_phaseB_sparse/p8",  "net_epoch28.pth", [8, 16, 32]),
]
Q_VARIANTS = ["fp16", "int8_mm"]


def assert_gpu_isolated(gpu_id: int) -> dict:
    samples = []
    for i in range(3):
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            text=True,
        )
        parts = [p.strip() for p in out.strip().split(",")]
        util, mem = int(parts[1]), int(parts[2])
        samples.append({"util": util, "mem_mb": mem})
        if util > GPU_UTIL_THRESHOLD_PCT or mem > GPU_MEM_USED_THRESHOLD_MB:
            raise RuntimeError(
                f"GPU {gpu_id} NOT idle on sample {i+1}/3 "
                f"(util={util}% mem={mem} MB). Bench REFUSED.")
        if i < 2:
            time.sleep(2.0)
    return {"gpu_id": gpu_id, "samples": samples}


def make_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 4:
        return torch.ones_like(weight)
    out_c, in_c, kH, kW = weight.shape
    if in_c % 4 != 0 or in_c < 4:
        return torch.ones_like(weight)
    flat = weight.permute(0, 2, 3, 1).reshape(-1, in_c)
    mask = torch.zeros_like(flat)
    abs_w = flat.abs()
    n_groups = in_c // 4
    for g in range(n_groups):
        s, e = g * 4, (g + 1) * 4
        topk = abs_w[:, s:e].topk(2, dim=1).indices
        for col in range(2):
            mask[torch.arange(mask.size(0)), s + topk[:, col]] = 1.0
    return mask.reshape(out_c, kH, kW, in_c).permute(0, 3, 1, 2).contiguous()


def remask_ckpt(in_ckpt: Path, out_ckpt: Path) -> dict:
    sd = torch.load(in_ckpt, map_location="cpu", weights_only=False)
    n_masked = 0
    n_zeroed_total = 0
    for k, v in sd.items():
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] >= 4 and v.shape[1] % 4 == 0:
            mask = make_2_4_mask(v)
            sd[k] = v * mask
            n_masked += 1
            n_zeroed_total += int((mask == 0).sum())
    torch.save(sd, out_ckpt)
    return {"n_layers_masked": n_masked, "n_zeros_total": n_zeroed_total, "size_mb": round(out_ckpt.stat().st_size / 1e6, 2)}


def export_onnx(model_dir: Path, ckpt_name: str, num_filters: List[int], tag: str, remasked_ckpt: Path) -> Optional[Path]:
    from opencood.hypes_yaml import yaml_utils as heal_yaml
    from opencood.tools import train_utils
    onnx_path = ONNX_DIR / f"g8_{tag}_sparse_pyramid.onnx"
    if onnx_path.exists():
        return onnx_path
    cfg = heal_yaml.load_yaml(str(model_dir / "config.yaml"))
    cfg["model"]["args"]["fusion_backbone"]["num_filters"] = num_filters
    model = train_utils.create_model(cfg)
    sd = torch.load(remasked_ckpt, map_location="cpu", weights_only=False)
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
    torch.onnx.export(wrapped, dummy, str(onnx_path),
                       input_names=[INPUT_NAME], output_names=["bev_out"],
                       opset_version=16, do_constant_folding=True)
    return onnx_path


class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, calib_cache: Path):
        super().__init__()
        self.calib_cache = calib_cache
    def get_batch_size(self): return 1
    def get_batch(self, names): return None
    def read_calibration_cache(self):
        if self.calib_cache.exists():
            return self.calib_cache.read_bytes()
        return None
    def write_calibration_cache(self, cache):
        self.calib_cache.write_bytes(cache)


def build_engine(onnx_path: Path, engine_path: Path, q_variant: str) -> Tuple[bool, dict]:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            return False, {"error": "onnx parse failed"}

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 20))
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    if q_variant == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif q_variant == "int8_mm":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = MinMaxCalibrator(CALIB_DIR / "pyramid_dair_calib_minmax.cache")

    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_secs = time.time() - t0
    if serialized is None:
        return False, {"error": "engine build failed", "build_secs": build_secs}
    engine_path.write_bytes(bytes(serialized))
    return True, {"engine_size_mb": round(engine_path.stat().st_size / 1e6, 3),
                   "build_secs": round(build_secs, 1)}


def bench_engine(engine_path: Path) -> dict:
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    d_in = torch.randn(*INPUT_SHAPE, device="cuda")
    out_shape = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            out_shape = tuple(context.get_tensor_shape(name))
            break
    d_out = torch.zeros(*(out_shape or (1, 384, 128, 128)), dtype=torch.float32, device="cuda")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_tensor_address(name, d_in.data_ptr())
        else:
            context.set_tensor_address(name, d_out.data_ptr())
    stream = torch.cuda.Stream()
    for _ in range(N_WARMUP):
        context.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(N_RUNS)]
    for i in range(N_RUNS):
        starts[i].record()
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        ends[i].record()
    torch.cuda.synchronize()
    lats = np.array([starts[i].elapsed_time(ends[i]) for i in range(N_RUNS)])
    return {"lat_p50": round(float(np.percentile(lats, 50)), 3),
             "lat_p99": round(float(np.percentile(lats, 99)), 3),
             "lat_std": round(float(lats.std()), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-gpu", type=int, default=2)
    args = parser.parse_args()

    stats = assert_gpu_isolated(args.bench_gpu)
    print(f"[Plan v5 Phase B.2-3] GPU {args.bench_gpu} STRICT isolation PASS")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.bench_gpu)

    rows = []
    for tag, model_dir_str, ckpt_name, nf in SPARSE_ANCHORS:
        model_dir = Path(model_dir_str)
        src_ckpt = model_dir / ckpt_name
        remasked = model_dir / f"{ckpt_name.replace('.pth', '')}_remasked.pth"
        info = remask_ckpt(src_ckpt, remasked)
        print(f"\n[{tag}] re-masked: {info['n_layers_masked']} layers, "
              f"{info['n_zeros_total']:,} zeros, size={info['size_mb']} MB")

        onnx_path = export_onnx(model_dir, ckpt_name, nf, tag, remasked)
        if onnx_path is None:
            continue

        for q in Q_VARIANTS:
            engine_path = ENGINE_DIR / f"g8_{tag}_sparse_{q}.engine"
            if not engine_path.exists():
                ok, build_info = build_engine(onnx_path, engine_path, q)
                if not ok:
                    print(f"  [{tag}|{q}] BUILD FAIL: {build_info}")
                    rows.append({"tag": tag, "q": q, "num_filters": str(nf),
                                  "status": "build_failed", **build_info})
                    continue
                print(f"  [{tag}|{q}] built {build_info['engine_size_mb']} MB in {build_info['build_secs']}s")
            else:
                build_info = {"engine_size_mb": round(engine_path.stat().st_size / 1e6, 3),
                              "build_secs": -1}
            lat = bench_engine(engine_path)
            print(f"  [{tag}|{q}] p50={lat['lat_p50']} ms p99={lat['lat_p99']} ms")
            rows.append({
                "tag": tag, "q": q, "d": "D1_default", "num_filters": str(nf),
                "sparsity": "n2_m4", "status": "ok",
                **build_info, **lat,
            })

    out_csv = DATA_DIR / "plan5_phaseB_anchors.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\n[Plan v5 Phase B.2-3] wrote {len(rows)} rows -> {out_csv}")

    dense_df = pd.read_csv(DATA_DIR / "plan5_phaseA_anchors.csv")
    dense_df = dense_df[dense_df["d"] == "D1_default"]
    sparse_df = pd.DataFrame(rows)
    if "lat_p50" not in sparse_df.columns:
        sparse_df["lat_p50"] = np.nan
    merged = sparse_df.merge(
        dense_df[["tag", "q", "lat_p50"]].rename(columns={"lat_p50": "lat_dense_p50"}),
        on=["tag", "q"], how="left")
    merged["reduction_pct"] = (1 - merged["lat_p50"] / merged["lat_dense_p50"]) * 100
    print("\n--- B.4 gate G_B sparsity reduction ---")
    print(merged[["tag", "q", "lat_p50", "lat_dense_p50", "reduction_pct"]].to_string(index=False))

    pass_planes_per_q = {}
    for q in Q_VARIANTS:
        sub = merged[merged["q"] == q]
        n_pass = int((sub["reduction_pct"] >= 30).sum())
        pass_planes_per_q[q] = n_pass
        print(f"  Q={q}: {n_pass}/5 plane PASS (reduction >=30%)")
    max_pass = max(pass_planes_per_q.values()) if pass_planes_per_q else 0
    gate_verdict = "PASS" if max_pass >= 3 else "FAIL"
    print(f"\nGate G_B: max pass_planes across Q = {max_pass}/5, threshold=3 → {gate_verdict}")

    summary = [
        "# Plan v5 Phase B summary",
        "",
        f"Gate G_B threshold: ≥3/5 plane with sparsity reduction ≥30%",
        f"Result: max_pass={max_pass}/5 across Q → **{gate_verdict}**",
        "",
        "## Per-Q per-plane lat reduction",
        "",
        merged[["tag", "q", "lat_p50", "lat_dense_p50", "reduction_pct"]].to_markdown(index=False),
    ]
    (DATA_DIR / "plan5_phaseB_summary.md").write_text("\n".join(summary))
    print(f"\n[Plan v5 Phase B] wrote {DATA_DIR / 'plan5_phaseB_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
