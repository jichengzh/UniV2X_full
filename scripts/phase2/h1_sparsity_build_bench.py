"""H1 — TRT sparse build + AP eval + latency bench after 2:4 finetune.

Steps:
1. Export collab2 ONNX from 2:4-sparse ckpt (net_epoch_bestval_at24.pth)
2. Build TRT FP16+SPARSE_WEIGHTS engine
3. Build TRT INT8+SPARSE_WEIGHTS engine (with existing DAIR calibration cache)
4. Benchmark both (body_subnet_collab2 口径, warmup=200/measure=200, locked GPU)
5. AP eval on DAIR val 1789 using hybrid TRT+PyTorch pipeline

Output:
  models/h1_sparse_cache/base_sparse_fp16.engine
  models/h1_sparse_cache/base_sparse_int8.engine
  results/H1_sparsity_4090.csv
  results/H1_sparsity_ap.json

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/phase2/h1_sparsity_build_bench.py

纪律:
 - 这是真正 2:4 sparse 权重 (不是 SPARSE_WEIGHTS flag noop)
 - AP 必须真测 (不复用 stage_a), independent_measured
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
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
sys.path.insert(0, str(REPO / "scripts" / "phase1"))
sys.path.insert(0, str(REPO / "tools"))

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
PYTHON_BIN = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"

# H1 finetune output
SPARSE_CKPT_DIR = Path("/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_sparse24_2026_06_03")
SPARSE_CKPT = SPARSE_CKPT_DIR / "net_epoch_bestval_at25.pth"
SPARSE_HYPES = SPARSE_CKPT_DIR / "config.yaml"

# Engine cache
H1_ENGINE_DIR = REPO / "models" / "h1_sparse_cache"
H1_ENGINE_DIR.mkdir(parents=True, exist_ok=True)

# Use existing INT8 calibration (DAIR base calib)
INT8_CALIB_CACHE = REPO / "models" / "stage_a_cache" / "base_int8_calib.cache"

COLLAB2_INPUT_SHAPE = (2, 64, 128, 256)
COLLAB2_EXTRA_INPUTS = {"t_ego": (2, 2, 3)}
N_WARMUP = 200
N_MEASURE = 200


# ── GPU gate ──────────────────────────────────────────────────────────────

def assert_gpu_idle(phys_id: int):
    out = subprocess.check_output([
        "nvidia-smi", f"--id={phys_id}",
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
    ]).decode().strip()
    me = os.getpid()
    other = sum(int(l.split(",")[1].strip()) for l in out.splitlines()
                if l.strip() and int(l.split(",")[0].strip()) != me)
    if other > 50:
        raise RuntimeError(f"GPU {phys_id} not idle: other_mem={other}MiB")
    print(f"[gate] GPU{phys_id} idle (other_mem={other}MiB) ✓")


# ── ONNX export ───────────────────────────────────────────────────────────

def export_sparse_onnx(out_path: Path) -> Path:
    """Export collab2 ONNX from sparse ckpt."""
    if out_path.exists():
        print(f"[onnx] CACHE HIT: {out_path.name}")
        return out_path

    print(f"[onnx] exporting sparse collab2 ONNX from {SPARSE_CKPT.name} ...")
    cmd = [
        PYTHON_BIN, str(REPO / "tools" / "export_onnx_pyramid_collab.py"),
        "--ckpt", str(SPARSE_CKPT),
        "--hypes", str(SPARSE_HYPES),
        "--out", str(out_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HEAL_ROOT}:{REPO}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(cmd, cwd=str(REPO), env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[onnx] STDERR: {result.stderr[-500:]}")
        raise RuntimeError("ONNX export failed")

    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"[onnx] done → {out_path.name} ({size_mb:.1f}MB)")
    return out_path


# ── Build engine ──────────────────────────────────────────────────────────

class CacheCalib(trt.IInt8MinMaxCalibrator):
    def __init__(self, p): super().__init__(); self._p = p
    def get_batch_size(self): return 1
    def get_batch(self, names): return None
    def read_calibration_cache(self):
        with open(self._p, "rb") as f: return f.read()
    def write_calibration_cache(self, c): pass


def build_sparse_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,  # "fp16" or "int8"
    workspace_mb: int = 4096,
) -> Path:
    """Build TRT engine with SPARSE_WEIGHTS flag."""
    if engine_path.exists():
        print(f"[build] CACHE HIT: {engine_path.name}")
        return engine_path

    print(f"[build] {engine_path.name} | precision={precision} | ws={workspace_mb}MB | SPARSE_WEIGHTS=True")
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

    # Sparse weights flag - REQUIRED for actual 2:4 speedup
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] precision=FP16 + SPARSE_WEIGHTS")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        if not INT8_CALIB_CACHE.exists():
            raise FileNotFoundError(f"Calib cache not found: {INT8_CALIB_CACHE}")
        config.int8_calibrator = CacheCalib(str(INT8_CALIB_CACHE))
        print("[build] precision=INT8 + SPARSE_WEIGHTS")

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = engine_path.stat().st_size / (1024 ** 2)
    print(f"[build] done in {elapsed:.1f}s | size={size_mb:.2f}MB")
    return engine_path


# ── Layer count ───────────────────────────────────────────────────────────

def count_layer_precisions(engine_path: Path) -> dict:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    insp = engine.create_engine_inspector()
    data = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    counts = {"fp32": 0, "fp16": 0, "int8": 0, "other": 0}
    for layer in data.get("Layers", []):
        if not isinstance(layer, dict): continue
        lt = str(layer.get("LayerType", "")).lower()
        if "reformat" in lt or "noop" in lt: continue
        outputs = layer.get("Outputs", [])
        dtype = ""
        if outputs and isinstance(outputs[0], dict):
            fd = outputs[0].get("Format/Datatype", "")
            dtype = fd.lower() if fd else str(outputs[0].get("DataType", "")).lower()
        if "int8" in dtype: counts["int8"] += 1
        elif "half" in dtype: counts["fp16"] += 1
        elif "float" in dtype: counts["fp32"] += 1
        else: counts["other"] += 1
    return counts


# ── Benchmark ─────────────────────────────────────────────────────────────

def benchmark_engine(engine_path: Path) -> dict:
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    n_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(n_io)]
    input_names = [n for n in tensor_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]

    context.set_input_shape(input_names[0], COLLAB2_INPUT_SHAPE)
    for nm in input_names[1:]:
        if nm in COLLAB2_EXTRA_INPUTS:
            context.set_input_shape(nm, COLLAB2_EXTRA_INPUTS[nm])

    bufs = {}
    for name in tensor_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype_trt = engine.get_tensor_dtype(name)
        dtype_torch = {trt.float32: torch.float32, trt.float16: torch.float16,
                       trt.int32: torch.int32, trt.int8: torch.int8}.get(dtype_trt, torch.float32)
        bufs[name] = torch.empty(shape, dtype=dtype_torch, device="cuda")
        context.set_tensor_address(name, int(bufs[name].data_ptr()))

    torch.manual_seed(42)
    for nm in input_names:
        bufs[nm].copy_(torch.randn_like(bufs[nm].float()).to(bufs[nm].dtype))

    stream = torch.cuda.Stream()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        for _ in range(N_WARMUP):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

    times_ms = np.empty(N_MEASURE, dtype=np.float64)
    with torch.cuda.stream(stream):
        for k in range(N_MEASURE):
            start_evt.record(stream)
            context.execute_async_v3(stream.cuda_stream)
            end_evt.record(stream)
            end_evt.synchronize()
            times_ms[k] = start_evt.elapsed_time(end_evt)

    return {
        "lat_mean_ms": float(np.mean(times_ms)),
        "lat_p50_ms": float(np.percentile(times_ms, 50)),
        "lat_p99_ms": float(np.percentile(times_ms, 99)),
        "lat_std_ms": float(np.std(times_ms)),
        "engine_size_mb": round(engine_path.stat().st_size / (1024 ** 2), 3),
    }


# ── AP eval ───────────────────────────────────────────────────────────────

def run_ap_eval(engine_path: Path, cuda_device: int) -> dict:
    """Run hybrid AP eval using m4_8_hybrid_infer_ap.py on DAIR val 1789."""
    ap_eval_script = REPO / "scripts" / "phase1" / "m4_8_hybrid_infer_ap.py"
    tag = f"H1_{engine_path.stem}"
    out_json = REPO / "results" / f"m4_8_hybrid_ap_{tag}.json"

    cmd = [
        PYTHON_BIN, str(ap_eval_script),
        "--engine-collab", str(engine_path),   # collab2 engine (N=2)
        "--model-dir", str(SPARSE_CKPT_DIR),   # sparse ckpt dir (for PyTorch fallback)
        "--dataset", "dair",
        "--n-samples", "1789",
        "--tag", tag,
        "--report", str(out_json),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["PYTHONPATH"] = f"{HEAL_ROOT}:{REPO}:{env.get('PYTHONPATH', '')}"

    print(f"[ap_eval] running on engine={engine_path.name} ...")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[ap_eval] FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        return {"ap_eval_status": "FAILED", "elapsed_s": round(elapsed, 1)}

    if out_json.exists():
        with open(out_json) as f:
            ap_data = json.load(f)
        ap_data["ap_eval_status"] = "OK"
        ap_data["elapsed_s"] = round(elapsed, 1)
        return ap_data
    else:
        return {"ap_eval_status": "MISSING_OUTPUT", "elapsed_s": round(elapsed, 1)}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    phys_gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "5").split(",")[0])
    torch.cuda.set_device(0)

    print(f"\n{'='*60}")
    print(f"H1 — 2:4 Sparse TRT Build + AP Eval + Latency")
    print(f"GPU: {phys_gpu} | ckpt: {SPARSE_CKPT.name}")
    print(f"{'='*60}\n")

    assert_gpu_idle(phys_gpu)

    # Step 1: Export ONNX from sparse ckpt
    sparse_onnx_path = H1_ENGINE_DIR / "base_sparse_collab2_fp32.onnx"
    export_sparse_onnx(sparse_onnx_path)

    rows = []

    for precision in ["fp16", "int8"]:
        engine_path = H1_ENGINE_DIR / f"base_sparse_{precision}.engine"

        assert_gpu_idle(phys_gpu)

        # Build TRT engine with SPARSE_WEIGHTS
        try:
            build_sparse_engine(sparse_onnx_path, engine_path, precision, workspace_mb=4096)
        except Exception as e:
            print(f"[ERROR] build {precision}: {e}")
            rows.append({"config": f"base_sparse_{precision}", "status": f"BUILD_ERROR:{e}"})
            continue

        # Layer count
        layer_counts = count_layer_precisions(engine_path)

        # Benchmark
        print(f"[bench] {engine_path.name} warmup={N_WARMUP} measure={N_MEASURE} ...")
        assert_gpu_idle(phys_gpu)
        lat = benchmark_engine(engine_path)
        print(f"  p50={lat['lat_p50_ms']:.4f}ms p99={lat['lat_p99_ms']:.4f}ms "
              f"size={lat['engine_size_mb']:.2f}MB")

        # AP eval (DAIR val 1789)
        ap_data = run_ap_eval(engine_path, phys_gpu)
        ap50 = ap_data.get("ap50") or ap_data.get("ap_50") or ap_data.get("AP50")
        ap70 = ap_data.get("ap70") or ap_data.get("ap_70") or ap_data.get("AP70")
        print(f"  AP50={ap50} AP70={ap70} ({ap_data['ap_eval_status']})")

        rows.append({
            "config": f"base_sparse_{precision}",
            "planes": "64_128_256",
            "precision": precision,
            "workspace_mb": 4096,
            "is_sparse": True,
            "sparse_kind": "2_4_magnitude",
            "finetune_epochs": 2,
            "ckpt": str(SPARSE_CKPT),
            "lat_p50_ms": round(lat["lat_p50_ms"], 4),
            "lat_p99_ms": round(lat["lat_p99_ms"], 4),
            "lat_mean_ms": round(lat["lat_mean_ms"], 4),
            "lat_std_ms": round(lat["lat_std_ms"], 4),
            "fp16_layer_count": layer_counts["fp16"],
            "int8_layer_count": layer_counts["int8"],
            "fp32_layer_count": layer_counts["fp32"],
            "engine_size_mb": lat["engine_size_mb"],
            "ap50": ap50,
            "ap70": ap70,
            "ap_eval_status": ap_data.get("ap_eval_status"),
            "ap_eval_elapsed_s": ap_data.get("elapsed_s"),
            "gpu": phys_gpu,
            "gpu_idle_verified": True,
            "latency_kind": "body_subnet_collab2",
            "regime": "front",
            "ap_reuse_basis": "independent_measured",
            "engine_path": str(engine_path),
            "source": "H1_sparsity;lat:CUDA-Event;ap:DAIR_val_1789;is_sparse=True",
            "status": "OK",
        })

    # Write CSV
    out_csv = REPO / "results" / "H1_sparsity_4090.csv"
    if rows:
        all_keys = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\n[done] {sum(1 for r in rows if r.get('status')=='OK')}/{len(rows)} OK → {out_csv}")

    # Also write AP JSON
    ap_json = REPO / "results" / "H1_sparsity_ap.json"
    ap_rows = {r["config"]: {k: r.get(k) for k in ["ap50", "ap70", "ap_eval_status"]}
               for r in rows}
    with open(ap_json, "w") as f:
        json.dump(ap_rows, f, indent=2)
    print(f"[done] AP summary → {ap_json}")


if __name__ == "__main__":
    main()
