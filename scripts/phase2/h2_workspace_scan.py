"""H2 — workspace/opt-level scan for 3 configs × 4 workspace sizes.

口径: body_subnet_collab2 (2×64×128×256 + t_ego 2×2×3), CUDA-Event, warmup=200/measure=200.
Builds 12 TRT engines (3 configs × 4 workspace) + benches each.
Records fp16_layer_count / int8_layer_count via IEngineInspector.

Output: results/H2_workspace_scan_4090.csv

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/phase2/h2_workspace_scan.py

纪律:
 - 锁频: GPU3 已 persistence_mode=on + 2520MHz 最大时钟 (sudo 不可用, 等效锁频)
 - 空闲确认: util<=2% / other_mem<=50MiB (脚本内 gate)
 - 真测就是真测: INT8 走真 calib cache, 非 proxy
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "phase1"))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────
PHYS_GPU = int(os.environ.get("CUDA_VISIBLE_DEVICES", "3").split(",")[0])

STAGE_A = REPO / "models" / "stage_a_cache"
H2_ENGINE_DIR = REPO / "models" / "h2_workspace_cache"
H2_ENGINE_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {
        "name": "base_fp16",
        "planes": "64_128_256",
        "onnx": str(STAGE_A / "base.onnx"),
        "precision": "fp16",
        "calib_cache": None,
    },
    {
        "name": "p50_int8",
        "planes": "32_64_128",
        "onnx": str(STAGE_A / "pruned50.onnx"),
        "precision": "int8",
        "calib_cache": str(STAGE_A / "pruned50_int8_calib.cache"),
    },
    {
        "name": "p25_trap_int8",
        "planes": "48_96_192",
        "onnx": str(STAGE_A / "pruned25.onnx"),
        "precision": "int8",
        "calib_cache": str(STAGE_A / "pruned25_int8_calib.cache"),
    },
]

WORKSPACES_MB = [256, 1024, 2048, 4096]   # 256MB / 1GB / 2GB / 4GB

# Also test base_fp16 with opt_level=5 at 4GB workspace (1 extra point)
OPT5_POINT = {
    "name": "base_fp16_opt5",
    "planes": "64_128_256",
    "onnx": str(STAGE_A / "base.onnx"),
    "precision": "fp16",
    "calib_cache": None,
    "workspace_mb": 4096,
    "opt_level": 5,
}

# body_subnet_collab2 口径: 2-agent batch
COLLAB2_INPUT_SHAPE = (2, 64, 128, 256)
COLLAB2_EXTRA_INPUTS = {"t_ego": (2, 2, 3)}

N_WARMUP = 200
N_MEASURE = 200


# ── GPU gate ───────────────────────────────────────────────────────────────

def gpu_idle_check(phys_id: int) -> tuple[int, int]:
    """Return (util%, other_proc_mem_MiB). Raises if not idle."""
    out = subprocess.check_output([
        "nvidia-smi", f"--id={phys_id}",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    util, mem = [int(x.strip()) for x in out.split(",")]
    return util, mem


def foreign_mem(phys_id: int) -> int:
    """Memory used by non-self processes on this GPU (MiB)."""
    out = subprocess.check_output([
        "nvidia-smi", f"--id={phys_id}",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    me = os.getpid()
    total = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, mem_s = line.split(",")
        if int(pid_s.strip()) != me:
            total += int(mem_s.strip())
    return total


def assert_gpu_idle(phys_id: int):
    util, mem = gpu_idle_check(phys_id)
    other = foreign_mem(phys_id)
    if util > 2 or other > 50:
        raise RuntimeError(
            f"GPU {phys_id} not idle: util={util}% other_mem={other}MiB. Abort."
        )
    print(f"[gate] GPU{phys_id} idle: util={util}% mem={mem}MiB (other={other}MiB) ✓")
    return util, other


# ── Build engine ───────────────────────────────────────────────────────────

class CacheCalib(trt.IInt8MinMaxCalibrator):
    """Read-only calibrator that returns a pre-built cache."""
    def __init__(self, cache_path: str):
        super().__init__()
        self._path = cache_path

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        return None

    def read_calibration_cache(self):
        with open(self._path, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache):
        pass  # read-only


def build_engine(
    onnx_path: str,
    precision: str,
    engine_path: str,
    workspace_mb: int = 4096,
    calib_cache: str | None = None,
    opt_level: int = 3,
) -> str:
    """Build TRT engine from ONNX. Returns engine_path."""
    ep = Path(engine_path)
    if ep.exists():
        print(f"[build] CACHE HIT: {ep.name}")
        return engine_path

    print(f"[build] {ep.name} | workspace={workspace_mb}MB | precision={precision} | opt={opt_level}")
    t0 = time.time()

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse failed: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = opt_level

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow FP16 fallback for unsupported INT8 layers
        if calib_cache:
            config.int8_calibrator = CacheCalib(calib_cache)
        else:
            raise ValueError("INT8 requires calib_cache")

    # Use DETAILED verbosity so IEngineInspector can count layer precisions
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"build_serialized_network returned None for {onnx_path}")

    ep.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = ep.stat().st_size / (1024 ** 2)
    print(f"[build] done in {elapsed:.1f}s | size={size_mb:.2f}MB → {ep.name}")
    return engine_path


# ── Layer count ────────────────────────────────────────────────────────────

def count_layer_precisions(engine_path: str) -> dict:
    """Use IEngineInspector to count fp16 / int8 / fp32 layers."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    insp = engine.create_engine_inspector()
    raw = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    data = json.loads(raw)

    counts = {"fp32": 0, "fp16": 0, "int8": 0, "other": 0}
    layers = data.get("Layers", [])
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        # OutputType or InputDataType may indicate precision
        layer_type = str(layer.get("LayerType", "")).lower()
        if "reformat" in layer_type or "noop" in layer_type:
            continue  # skip reformat nodes
        # Try to find precision from output data type
        outputs = layer.get("Outputs", [])
        if outputs and isinstance(outputs[0], dict):
            dtype = str(outputs[0].get("DataType", "")).lower()
        else:
            dtype = str(layer.get("OutputType", "")).lower()

        if "int8" in dtype:
            counts["int8"] += 1
        elif "float16" in dtype or "fp16" in dtype or "half" in dtype:
            counts["fp16"] += 1
        elif "float32" in dtype or "fp32" in dtype or "float" in dtype:
            counts["fp32"] += 1
        else:
            counts["other"] += 1

    total = sum(counts.values())
    print(f"  [layers] total={total} fp16={counts['fp16']} int8={counts['int8']} "
          f"fp32={counts['fp32']} other={counts['other']}")
    return counts


# ── Benchmark ──────────────────────────────────────────────────────────────

def benchmark_engine(engine_path: str, n_warmup: int = 200, n_measure: int = 200) -> dict:
    """Benchmark engine in body_subnet_collab2 口径."""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    n_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_names = [n for n in tensor_names
                   if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    output_names = [n for n in tensor_names
                    if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    # Set input shapes: spatial_features + optional t_ego
    context.set_input_shape(input_names[0], COLLAB2_INPUT_SHAPE)
    for nm in input_names[1:]:
        if nm in COLLAB2_EXTRA_INPUTS:
            context.set_input_shape(nm, COLLAB2_EXTRA_INPUTS[nm])
        else:
            print(f"  [warn] unknown extra input '{nm}', skipping shape set")

    # Allocate buffers
    bufs = {}
    for name in tensor_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype_trt = engine.get_tensor_dtype(name)
        dtype_torch = {
            trt.float32: torch.float32, trt.float16: torch.float16,
            trt.int32: torch.int32, trt.int8: torch.int8,
            trt.int64: torch.int64,
        }.get(dtype_trt, torch.float32)
        bufs[name] = torch.empty(shape, dtype=dtype_torch, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))

    torch.manual_seed(42)
    for nm in input_names:
        bufs[nm].copy_(torch.randn_like(bufs[nm].float()).to(bufs[nm].dtype))

    stream = torch.cuda.Stream()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Warmup
    with torch.cuda.stream(stream):
        for _ in range(n_warmup):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    # Measure
    times_ms = np.empty(n_measure, dtype=np.float64)
    with torch.cuda.stream(stream):
        for k in range(n_measure):
            start_evt.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            end_evt.record(stream)
            end_evt.synchronize()
            times_ms[k] = start_evt.elapsed_time(end_evt)

    engine_size_mb = Path(engine_path).stat().st_size / (1024 ** 2)

    return {
        "lat_mean_ms": float(np.mean(times_ms)),
        "lat_p50_ms": float(np.percentile(times_ms, 50)),
        "lat_p99_ms": float(np.percentile(times_ms, 99)),
        "lat_std_ms": float(np.std(times_ms)),
        "engine_size_mb": round(engine_size_mb, 3),
        "n_warmup": n_warmup,
        "n_measure": n_measure,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES handles physical mapping

    print(f"\n{'='*60}")
    print(f"H2 workspace scan | GPU={PHYS_GPU} | {len(CONFIGS)} configs × {len(WORKSPACES_MB)} ws = {len(CONFIGS)*len(WORKSPACES_MB)} points")
    print(f"{'='*60}\n")

    # Pre-flight: GPU idle check
    util, other = assert_gpu_idle(PHYS_GPU)

    # Get current clock
    clock_info = subprocess.check_output([
        "nvidia-smi", f"--id={PHYS_GPU}",
        "--query-gpu=clocks.current.graphics,clocks.max.graphics",
        "--format=csv,noheader,nounits",
    ]).decode().strip()
    cur_clk, max_clk = [x.strip() for x in clock_info.split(",")]
    clock_locked = (cur_clk == max_clk)
    print(f"[clock] current={cur_clk}MHz max={max_clk}MHz locked_equivalent={clock_locked}")

    rows = []

    # ── Main matrix: 3 configs × 4 workspace ──────────────────────────────
    for cfg in CONFIGS:
        for ws_mb in WORKSPACES_MB:
            engine_name = f"{cfg['name']}_ws{ws_mb}mb.engine"
            engine_path = str(H2_ENGINE_DIR / engine_name)

            print(f"\n── {cfg['name']} | ws={ws_mb}MB ──")

            # Re-check GPU idle before each build+bench cycle
            try:
                util2, other2 = assert_gpu_idle(PHYS_GPU)
            except RuntimeError as e:
                print(f"[ABORT] {e}")
                rows.append({
                    "config": cfg["name"], "planes": cfg["planes"],
                    "workspace_mb": ws_mb, "precision": cfg["precision"],
                    "status": "ABORT_NOT_IDLE", "gpu": PHYS_GPU,
                })
                continue

            try:
                # Build
                build_engine(
                    onnx_path=cfg["onnx"],
                    precision=cfg["precision"],
                    engine_path=engine_path,
                    workspace_mb=ws_mb,
                    calib_cache=cfg.get("calib_cache"),
                    opt_level=3,
                )

                # Layer count
                layer_counts = count_layer_precisions(engine_path)

                # Benchmark
                print(f"  [bench] warmup={N_WARMUP} measure={N_MEASURE} ...")
                lat = benchmark_engine(engine_path, N_WARMUP, N_MEASURE)
                print(f"  [result] p50={lat['lat_p50_ms']:.4f}ms p99={lat['lat_p99_ms']:.4f}ms "
                      f"mean={lat['lat_mean_ms']:.4f}ms size={lat['engine_size_mb']:.2f}MB")

                rows.append({
                    "config": cfg["name"],
                    "planes": cfg["planes"],
                    "workspace_mb": ws_mb,
                    "opt_level": 3,
                    "precision": cfg["precision"],
                    "lat_p50_ms": round(lat["lat_p50_ms"], 4),
                    "lat_p99_ms": round(lat["lat_p99_ms"], 4),
                    "lat_mean_ms": round(lat["lat_mean_ms"], 4),
                    "lat_std_ms": round(lat["lat_std_ms"], 4),
                    "fp16_layer_count": layer_counts["fp16"],
                    "int8_layer_count": layer_counts["int8"],
                    "fp32_layer_count": layer_counts["fp32"],
                    "engine_size_mb": lat["engine_size_mb"],
                    "gpu": PHYS_GPU,
                    "gpu_util_pct": util2,
                    "other_mem_mib": other2,
                    "clock_mhz": cur_clk,
                    "clock_locked": clock_locked,
                    "gpu_idle_verified": True,
                    "latency_kind": "body_subnet_collab2",
                    "regime": "d_space",
                    "ap_reuse_basis": "stage_a_gold_exact_reuse",
                    "engine_path": engine_path,
                    "source": "H2_workspace_scan;lat:CUDA-Event;build:TRT-python-api;gpu_idle_pct_verified",
                    "status": "OK",
                })

            except Exception as e:
                print(f"  [ERROR] {e}")
                rows.append({
                    "config": cfg["name"], "planes": cfg["planes"],
                    "workspace_mb": ws_mb, "precision": cfg["precision"],
                    "status": f"ERROR:{e}", "gpu": PHYS_GPU,
                })

    # ── opt=5 extra point ──────────────────────────────────────────────────
    print(f"\n── base_fp16_opt5 | ws=4096MB | opt_level=5 ──")
    try:
        assert_gpu_idle(PHYS_GPU)
        engine_path_opt5 = str(H2_ENGINE_DIR / "base_fp16_opt5_ws4096mb.engine")
        build_engine(
            onnx_path=OPT5_POINT["onnx"],
            precision=OPT5_POINT["precision"],
            engine_path=engine_path_opt5,
            workspace_mb=OPT5_POINT["workspace_mb"],
            calib_cache=None,
            opt_level=OPT5_POINT["opt_level"],
        )
        layer_counts_opt5 = count_layer_precisions(engine_path_opt5)
        lat_opt5 = benchmark_engine(engine_path_opt5, N_WARMUP, N_MEASURE)
        print(f"  [result] opt5 p50={lat_opt5['lat_p50_ms']:.4f}ms")

        rows.append({
            "config": OPT5_POINT["name"],
            "planes": OPT5_POINT["planes"],
            "workspace_mb": OPT5_POINT["workspace_mb"],
            "opt_level": OPT5_POINT["opt_level"],
            "precision": OPT5_POINT["precision"],
            "lat_p50_ms": round(lat_opt5["lat_p50_ms"], 4),
            "lat_p99_ms": round(lat_opt5["lat_p99_ms"], 4),
            "lat_mean_ms": round(lat_opt5["lat_mean_ms"], 4),
            "lat_std_ms": round(lat_opt5["lat_std_ms"], 4),
            "fp16_layer_count": layer_counts_opt5["fp16"],
            "int8_layer_count": layer_counts_opt5["int8"],
            "fp32_layer_count": layer_counts_opt5["fp32"],
            "engine_size_mb": lat_opt5["engine_size_mb"],
            "gpu": PHYS_GPU,
            "clock_mhz": cur_clk,
            "clock_locked": clock_locked,
            "gpu_idle_verified": True,
            "latency_kind": "body_subnet_collab2",
            "regime": "d_space",
            "ap_reuse_basis": "stage_a_gold_exact_reuse",
            "engine_path": engine_path_opt5,
            "source": "H2_workspace_scan;lat:CUDA-Event;build:TRT-python-api;opt5_vs_opt3",
            "status": "OK",
        })
    except Exception as e:
        print(f"  [ERROR] opt5: {e}")

    # ── Write CSV ──────────────────────────────────────────────────────────
    out_path = REPO / "results" / "H2_workspace_scan_4090.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        all_keys = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        ok_count = sum(1 for r in rows if r.get("status") == "OK")
        print(f"\n[done] {ok_count}/{len(rows)} OK → {out_path}")
    else:
        print("[warn] no rows to write")

    return rows


if __name__ == "__main__":
    main()
