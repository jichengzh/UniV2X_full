"""Plan2-Step2a — 4090 D 维度扩展 bench.

3 triplet × 2 prec × 4 D_tactic × 2 D_workspace = 48 points.
4090 D_scheme 退化为 {单 IP GPU},不变.

Tactic 组合 (TRT 10):
    default            : CUBLAS + CUBLAS_LT + EDGE_MASK + JIT
    default+CUDNN      : default + CUDNN
    default+CUBLAS_LT  : 等价 default (CUBLAS_LT 已在 default,这里测排除其他后仅 LT)
                          → 改成 EDGE_MASK_ONLY 作为对照
    all_enabled        : default + CUDNN(TRT 10 已无更多 source)

Workspace 档位: {4GB, 8GB}

输出: data/4090_dspace_bench.parquet
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "models/p0_random_cache"
OUT_DIR = REPO_ROOT / "models/4090_dspace_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRIPLETS = [
    ("T_baseline", 64, 128, 256),
    ("T_prune50",  32,  64, 136),
    ("T_prune75",  16,  32,  64),
]

# Tactic source combos (TRT 10)
DEFAULT_TC = (
    1 << int(trt.TacticSource.CUBLAS)
    | 1 << int(trt.TacticSource.CUBLAS_LT)
    | 1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
    | 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
)
WITH_CUDNN = DEFAULT_TC | (1 << int(trt.TacticSource.CUDNN))
EDGE_ONLY = (
    1 << int(trt.TacticSource.EDGE_MASK_CONVOLUTIONS)
    | 1 << int(trt.TacticSource.JIT_CONVOLUTIONS)
)
ALL_ENABLED = WITH_CUDNN  # TRT 10 has no more sources beyond default+CUDNN

TACTIC_CONFIGS = [
    ("default",          DEFAULT_TC),
    ("with_cudnn",       WITH_CUDNN),
    ("edge_only",        EDGE_ONLY),
    ("all_enabled",      ALL_ENABLED),
]

WORKSPACE_CONFIGS_GB = [4, 8]


class MinMaxCalib(trt.IInt8MinMaxCalibrator):
    def __init__(self, cache_path, shapes):
        super().__init__()
        self.cache_path = Path(cache_path)
        self.bufs = {n: torch.randn(*s, device="cuda").contiguous() for n, s in shapes.items()}
        self.served = False

    def get_batch_size(self): return 1

    def get_batch(self, names):
        if self.served: return None
        self.served = True
        return [int(self.bufs[n].data_ptr()) for n in names]

    def read_calibration_cache(self):
        if self.cache_path.exists(): return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, blob):
        self.cache_path.write_bytes(bytes(blob))


def build_engine(onnx_path, precision, tactic_flag, workspace_bytes, engine_path, calib_cache):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    config.set_tactic_sources(tactic_flag)

    spatial = (2, 64, 128, 256); tego = (2, 2, 3)
    profile = builder.create_optimization_profile()
    profile.set_shape("spatial_features", spatial, spatial, spatial)
    profile.set_shape("t_ego", tego, tego, tego)
    config.add_optimization_profile(profile)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8); config.set_flag(trt.BuilderFlag.FP16)
        calibrator = MinMaxCalib(calib_cache, {"spatial_features": spatial, "t_ego": tego})
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)

    t0 = time.time()
    try:
        serialized = builder.build_serialized_network(network, config)
    except Exception as e:
        return {"build_success": False, "fail_reason": str(e)[:200], "build_secs": time.time()-t0}
    build_secs = time.time() - t0
    if serialized is None:
        return {"build_success": False, "fail_reason": "build_serialized_network returned None",
                "build_secs": build_secs}
    Path(engine_path).write_bytes(bytes(serialized))
    del calibrator, builder, network, parser, config
    gc.collect(); torch.cuda.empty_cache()
    return {"build_success": True, "build_secs": build_secs}


def bench(engine_path, n_warmup=50, n_measure=100):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_names = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]

    shapes = {"spatial_features": (2, 64, 128, 256), "t_ego": (2, 2, 3)}
    for n in input_names: context.set_input_shape(n, shapes[n])
    bufs = {}
    for name in names:
        dtype = engine.get_tensor_dtype(name)
        shape = tuple(context.get_tensor_shape(name))
        torch_dtype = {trt.float32: torch.float32, trt.float16: torch.float16,
                        trt.int32: torch.int32, trt.int8: torch.int8,
                        trt.int64: torch.int64}.get(dtype, torch.float32)
        bufs[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))
    torch.manual_seed(42)
    for n in input_names:
        bufs[n].copy_(torch.randn_like(bufs[n].float()).to(bufs[n].dtype))

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    times = np.empty(n_measure, dtype=np.float64)
    with torch.cuda.stream(stream):
        for _ in range(n_warmup): context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        for k in range(n_measure):
            start.record(stream); context.execute_async_v3(stream.cuda_stream)
            end.record(stream); end.synchronize()
            times[k] = start.elapsed_time(end)
    return {"p50_ms": float(np.percentile(times, 50)),
            "p99_ms": float(np.percentile(times, 99)),
            "mean_ms": float(times.mean()),
            "engine_size_mb": Path(engine_path).stat().st_size / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/4090_dspace_bench.parquet")
    args = ap.parse_args()

    rows = []
    total = len(TRIPLETS) * 2 * len(TACTIC_CONFIGS) * len(WORKSPACE_CONFIGS_GB)
    idx = 0
    t0_total = time.time()

    for tri_label, s0, s1, s2 in TRIPLETS:
        sig = f"{s0:03d}_{s1:03d}_{s2:03d}"
        onnx_path = CACHE_ROOT / f"onnx_{sig}.onnx"
        for prec in ("fp16", "int8"):
            for tc_label, tc_flag in TACTIC_CONFIGS:
                for ws_gb in WORKSPACE_CONFIGS_GB:
                    idx += 1
                    cfg_id = f"{sig}_{prec}_{tc_label}_{ws_gb}GB"
                    engine_path = OUT_DIR / f"engine_{cfg_id}.engine"
                    calib_cache = OUT_DIR / f"calib_{cfg_id}.cache"
                    print(f"\n[{idx}/{total}] {cfg_id}")
                    t0 = time.time()
                    bres = build_engine(onnx_path, prec, tc_flag, ws_gb*1024**3, engine_path, calib_cache)
                    if not bres["build_success"]:
                        print(f"  BUILD FAILED: {bres.get('fail_reason','?')[:100]}")
                        rows.append({
                            "triplet": tri_label, "triplet_sig": sig,
                            "stage0_planes": s0, "stage1_planes": s1, "stage2_planes": s2,
                            "precision": prec, "d_tactic": tc_label, "d_workspace_gb": ws_gb,
                            "build_success": False, "fail_reason": bres.get("fail_reason"),
                            "build_secs": bres["build_secs"],
                            "lat_p50_ms": None, "lat_p99_ms": None, "lat_mean_ms": None,
                            "engine_size_mb": None,
                        })
                        continue
                    try:
                        bench_res = bench(engine_path)
                    except Exception as e:
                        print(f"  BENCH FAILED: {e}")
                        continue
                    elapsed = time.time() - t0
                    print(f"  build={bres['build_secs']:.0f}s p50={bench_res['p50_ms']:.3f}"
                          f" engine={bench_res['engine_size_mb']:.2f}MB ({elapsed:.0f}s)")
                    rows.append({
                        "triplet": tri_label, "triplet_sig": sig,
                        "stage0_planes": s0, "stage1_planes": s1, "stage2_planes": s2,
                        "precision": prec, "d_tactic": tc_label, "d_workspace_gb": ws_gb,
                        "build_success": True, "fail_reason": None,
                        "build_secs": bres["build_secs"],
                        "lat_p50_ms": bench_res["p50_ms"],
                        "lat_p99_ms": bench_res["p99_ms"],
                        "lat_mean_ms": bench_res["mean_ms"],
                        "engine_size_mb": bench_res["engine_size_mb"],
                    })
                    cum = time.time() - t0_total
                    print(f"  cumulative: {cum:.0f}s ({cum/60:.1f} min)")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out); df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    print(df[["triplet","precision","d_tactic","d_workspace_gb","build_success","lat_p50_ms","engine_size_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()
