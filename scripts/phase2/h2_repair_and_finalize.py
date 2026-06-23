"""H2 repair & finalize: re-benchmark any ERROR/missing rows after main H2 run.

Run after h2_workspace_scan.py completes to:
1. Read H2_workspace_scan_4090.csv
2. Find ERROR/missing rows (typically ws1024 for base_fp16 and ws256 for p50_int8)
3. Re-benchmark those engines (they exist on disk from repair builds)
4. Write final merged CSV

Usage:
    CUDA_VISIBLE_DEVICES=5 python scripts/phase2/h2_repair_and_finalize.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

REPO = Path(__file__).resolve().parents[2]
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

H2_CSV = REPO / "results" / "H2_workspace_scan_4090.csv"
H2_ENGINE_DIR = REPO / "models" / "h2_workspace_cache"

COLLAB2_INPUT_SHAPE = (2, 64, 128, 256)
COLLAB2_EXTRA_INPUTS = {"t_ego": (2, 2, 3)}
N_WARMUP = 200
N_MEASURE = 200


def foreign_mem(phys_id: int) -> int:
    out = subprocess.check_output([
        "nvidia-smi", f"--id={phys_id}",
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
    ]).decode().strip()
    me = os.getpid()
    total = 0
    for line in out.splitlines():
        line = line.strip()
        if not line: continue
        pid_s, mem_s = line.split(",")
        if int(pid_s.strip()) != me:
            total += int(mem_s.strip())
    return total


def count_layer_precisions(engine_path: str) -> dict:
    """Count layers by precision via IEngineInspector.

    TRT 10.x JSON structure:
    - Layers[i]["Outputs"][j]["Format/Datatype"] = "Float" | "Half" | "Int8" | etc.
    """
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    insp = engine.create_engine_inspector()
    raw = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    data = json.loads(raw)
    counts = {"fp32": 0, "fp16": 0, "int8": 0, "other": 0}
    for layer in data.get("Layers", []):
        if not isinstance(layer, dict): continue
        lt = str(layer.get("LayerType", "")).lower()
        if "reformat" in lt or "noop" in lt: continue
        # TRT 10.x: "Format/Datatype" in output dicts
        outputs = layer.get("Outputs", [])
        dtype = ""
        if outputs and isinstance(outputs[0], dict):
            # TRT 10.x key is "Format/Datatype"
            fd = outputs[0].get("Format/Datatype", "")
            if fd:
                dtype = fd.lower()
            else:
                # Fallback to older key names
                dtype = str(outputs[0].get("DataType", outputs[0].get("Datatype", ""))).lower()
        if not dtype:
            dtype = str(layer.get("OutputType", "")).lower()
        # Map TRT type names to categories
        if "int8" in dtype:
            counts["int8"] += 1
        elif "half" in dtype or "fp16" in dtype or "float16" in dtype:
            counts["fp16"] += 1
        elif "float" in dtype or "fp32" in dtype or "float32" in dtype:
            counts["fp32"] += 1
        else:
            counts["other"] += 1
    return counts


def benchmark_engine(engine_path: str) -> dict:
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
        "engine_size_mb": round(Path(engine_path).stat().st_size / (1024**2), 3),
    }


def main():
    phys_gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "5").split(",")[0])
    torch.cuda.set_device(0)

    # Check GPU idle
    other = foreign_mem(phys_gpu)
    if other > 50:
        print(f"[abort] GPU{phys_gpu} not idle: other_mem={other}MiB")
        return

    print(f"[gate] GPU{phys_gpu} idle (other_mem={other}MiB)")

    if not H2_CSV.exists():
        print(f"[abort] {H2_CSV} does not exist yet. Run h2_workspace_scan.py first.")
        return

    # Read CSV
    with open(H2_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[read] {H2_CSV}: {len(rows)} rows")
    # Handle both ERROR and ABORT_NOT_IDLE rows
    error_rows = [r for r in rows if r.get("status", "") not in ("OK", "OK_repaired")]
    ok_rows = [r for r in rows if r.get("status", "").startswith("OK")]
    print(f"  OK: {len(ok_rows)}, ERROR: {len(error_rows)}")

    if not error_rows:
        print("[ok] No error rows found. H2 is complete!")
        return

    # Re-benchmark error rows
    repaired = []
    for row in error_rows:
        cfg = row.get("config", "")
        ws_mb = int(row.get("workspace_mb", 0))
        engine_name = f"{cfg}_ws{ws_mb}mb.engine"
        engine_path = str(H2_ENGINE_DIR / engine_name)

        if not Path(engine_path).exists():
            print(f"[skip] {engine_name}: engine still missing, cannot repair")
            repaired.append(row)
            continue

        print(f"[repair] benchmarking {engine_name} ...")
        try:
            other2 = foreign_mem(phys_gpu)
            if other2 > 50:
                print(f"  [abort] GPU not idle: other={other2}MiB")
                repaired.append(row)
                continue

            layer_counts = count_layer_precisions(engine_path)
            lat = benchmark_engine(engine_path)
            print(f"  p50={lat['lat_p50_ms']:.4f}ms p99={lat['lat_p99_ms']:.4f}ms "
                  f"size={lat['engine_size_mb']:.2f}MB")

            # Build repaired row
            repaired_row = dict(row)
            repaired_row.update({
                "lat_p50_ms": round(lat["lat_p50_ms"], 4),
                "lat_p99_ms": round(lat["lat_p99_ms"], 4),
                "lat_mean_ms": round(lat["lat_mean_ms"], 4),
                "lat_std_ms": round(lat["lat_std_ms"], 4),
                "fp16_layer_count": layer_counts["fp16"],
                "int8_layer_count": layer_counts["int8"],
                "fp32_layer_count": layer_counts["fp32"],
                "engine_size_mb": lat["engine_size_mb"],
                "gpu_idle_verified": True,
                "status": "OK_repaired",
                "source": "H2_repair;lat:CUDA-Event;engine_built_separately",
            })
            repaired.append(repaired_row)

        except Exception as e:
            print(f"  [error] {e}")
            repaired.append(row)

    # Write merged CSV (ok_rows + repaired rows)
    final_rows = ok_rows + repaired
    all_keys = []
    seen = set()
    for r in final_rows:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    final_csv = REPO / "results" / "H2_workspace_scan_4090_final.csv"
    with open(final_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in final_rows:
            writer.writerow(r)

    ok_final = sum(1 for r in final_rows if "OK" in r.get("status", ""))
    print(f"\n[done] {ok_final}/{len(final_rows)} OK → {final_csv}")


if __name__ == "__main__":
    main()
