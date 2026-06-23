"""P0.3 — Orin AGX Class A lat-only bench.

Runs ON ORIN (TRT 8.5.2.2). Builds engines per (T × Q × D) and measures lat via
TRT execute_v2. 432 anchor target = 8 triplet × 6 Q × 9 single-IP D. AP eval NOT
done on Orin (plan §4.1.3 cross-HW AP equivalence assumption).

Multi-IP D cells (D7 dual_A, D8 dual_B, D9 triple_C) deferred to P1.b (need
sub-ONNX export).

Output: /home/jichengzhi/orin_class_a/results/{anchor_id}.json per anchor +
        /home/jichengzhi/orin_class_a/results/_summary.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import tensorrt as trt

ORIN_ROOT = Path("/home/jichengzhi/orin_class_a")
ONNX_DIR = ORIN_ROOT / "onnx"
ENGINE_DIR = ORIN_ROOT / "engines"
RESULT_DIR = ORIN_ROOT / "results"
CALIB_DIR = ORIN_ROOT / "calibration"
ENGINE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# 8 triplets (ONNX must already be deployed to ONNX_DIR)
TRIPLETS = [
    ("T1_base",  (64, 128, 256)),
    ("T2_p25",   (48,  96, 192)),
    ("T3_p37",   (40,  80, 160)),
    ("T4_p50",   (32,  64, 128)),
    ("T5_p62",   (24,  56, 128)),
    ("T6_p75",   (16,  32,  64)),
    ("T7_wide",  (48,  64, 128)),
    ("T8_deep",  (24,  48, 192)),
]

# 6 Q cells (Orin TRT 8.5 capable subset)
Q_CELLS = [
    # (label, precision, calibrator)
    ("Q_fp32",         "fp32", "none"),
    ("Q_fp16",         "fp16", "none"),
    ("Q_int8_mm",      "int8", "minmax"),
    ("Q_int8_ent",     "int8", "entropy"),
    ("Q_best",         "best", "minmax"),  # trtexec --best: auto FP16+INT8 fallback
    ("Q_fp16_strict",  "fp16_strict", "none"),  # FP16 only, no fallback
]

# 9 single-IP D cells (skip D7/D8/D9 multi-IP for P1.b)
D_CELLS = [
    # (label, scheme, tactic, workspace_mb, use_dla, dla_core, gpu_fallback)
    ("D1_gpu_def_2gb",     "GPU",  "default",     2048,  False, -1, False),
    ("D2_gpu_def_8gb",     "GPU",  "default",     8192,  False, -1, False),
    ("D3_gpu_nocudnn_4gb", "GPU",  "no_cudnn",    4096,  False, -1, False),
    ("D4_dla0_def_1gb",    "DLA0", "default",     1024,  True,  0,  True),
    ("D5_dla1_def_1gb",    "DLA1", "default",     1024,  True,  1,  True),
    ("D6_dla0_nocudnn_2gb","DLA0", "no_cudnn",    2048,  True,  0,  True),
    ("D10_gpu_def_1gb",    "GPU",  "default",     1024,  False, -1, False),
    ("D11_gpu_def_8gb_2",  "GPU",  "default",     8192,  False, -1, False),  # repeat for spread reliability
    ("D12_dla0_strict_2gb","DLA0", "default",     2048,  True,  0,  False),  # strict (no fallback)
]


def make_minmax_calibrator(spatial_npy, tego_npy):
    """Tiny INT8 calibrator using pre-computed npy from 4090."""
    class _Calib(trt.IInt8MinMaxCalibrator):
        def __init__(self, spatial, tego):
            trt.IInt8MinMaxCalibrator.__init__(self)
            self.spatial = np.load(spatial)
            self.tego = np.load(tego)
            self.batch = 0
            self.n = min(self.spatial.shape[0] if self.spatial.ndim == 5 else 1, 32)
            import pycuda.autoinit  # noqa
            import pycuda.driver as cuda
            self.cuda = cuda
            self.d_spatial = cuda.mem_alloc(self.spatial[0].nbytes)
            self.d_tego = cuda.mem_alloc(self.tego[0].nbytes if self.tego.ndim == 4 else self.tego.nbytes)
        def get_batch_size(self): return 1
        def get_batch(self, names):
            if self.batch >= self.n: return None
            sp = self.spatial[self.batch] if self.spatial.ndim == 5 else self.spatial
            tg = self.tego[self.batch] if self.tego.ndim == 4 else self.tego
            self.cuda.memcpy_htod(self.d_spatial, np.ascontiguousarray(sp))
            self.cuda.memcpy_htod(self.d_tego, np.ascontiguousarray(tg))
            self.batch += 1
            return [int(self.d_spatial), int(self.d_tego)]
        def read_calibration_cache(self): return None
        def write_calibration_cache(self, cache): pass
    return _Calib(spatial_npy, tego_npy)


def build_engine(onnx_path, q, d, calib_npys):
    """Build TRT engine for (Q, D) combination. Returns (engine, build_secs, err)."""
    q_label, prec, calib_kind = q
    d_label, scheme, tactic, ws_mb, use_dla, dla_core, gpu_fb = d

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            return None, 0, f"onnx_parse_fail: {parser.get_error(0)}"

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_mb * 1024 * 1024)

    if prec == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif prec == "fp16_strict":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    elif prec == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        try:
            calib = make_minmax_calibrator(*calib_npys)
            config.int8_calibrator = calib
        except Exception as e:
            return None, 0, f"calib_fail: {e}"
    elif prec == "best":
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        try:
            calib = make_minmax_calibrator(*calib_npys)
            config.int8_calibrator = calib
        except Exception as e:
            return None, 0, f"calib_fail: {e}"
    # fp32: no flags

    # Tactic filtering
    if tactic == "no_cudnn":
        config.set_tactic_sources(
            config.get_tactic_sources() & ~(1 << int(trt.TacticSource.CUDNN))
        )

    # DLA
    if use_dla:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        if gpu_fb:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    t0 = time.time()
    try:
        serialized = builder.build_serialized_network(network, config)
    except Exception as e:
        return None, time.time() - t0, f"build_exception: {e}"
    elapsed = time.time() - t0
    if serialized is None:
        return None, elapsed, f"build_returned_none"
    return bytes(serialized), elapsed, None


def measure_lat(engine_bytes, n_warmup=100, n_measure=200):
    """Run engine n_measure times, return lat stats."""
    import pycuda.autoinit
    import pycuda.driver as cuda

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        return None, "deserialize_fail"
    context = engine.create_execution_context()

    bindings = []
    host_inputs, dev_inputs, dev_outputs, host_outputs = [], [], [], []
    output_shapes = []
    for i in range(engine.num_bindings):
        binding = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        if engine.binding_is_input(i):
            host = np.zeros(shape, dtype=dtype)
            dev = cuda.mem_alloc(size)
            cuda.memcpy_htod(dev, host)
            host_inputs.append(host); dev_inputs.append(dev)
            bindings.append(int(dev))
        else:
            host = np.zeros(shape, dtype=dtype)
            dev = cuda.mem_alloc(size)
            host_outputs.append(host); dev_outputs.append(dev)
            output_shapes.append(shape)
            bindings.append(int(dev))

    stream = cuda.Stream()
    for _ in range(n_warmup):
        context.execute_async_v2(bindings, stream.handle)
    stream.synchronize()

    starts, ends = [], []
    for _ in range(n_measure):
        s = cuda.Event(); e = cuda.Event()
        s.record(stream)
        context.execute_async_v2(bindings, stream.handle)
        e.record(stream)
        starts.append(s); ends.append(e)
    stream.synchronize()

    lats = np.array([s.time_till(e) for s, e in zip(starts, ends)])
    return {
        "lat_mean_ms": float(lats.mean()),
        "lat_p50_ms":  float(np.percentile(lats, 50)),
        "lat_p99_ms":  float(np.percentile(lats, 99)),
        "lat_min_ms":  float(lats.min()),
        "lat_max_ms":  float(lats.max()),
        "n_measure":   n_measure,
    }, None


def run_anchor(tag, planes, q, d, calib_npys):
    q_label, prec, calib_kind = q
    d_label, scheme, tactic, ws_mb, use_dla, dla_core, gpu_fb = d
    anchor_id = f"{tag}__{q_label}__{d_label}"
    report_path = RESULT_DIR / f"{anchor_id}.json"
    if report_path.exists():
        return json.loads(report_path.read_text())

    onnx = ONNX_DIR / f"{tag}.onnx"
    if not onnx.exists():
        return {"anchor_id": anchor_id, "status": "onnx_missing"}

    rep = {
        "anchor_id": anchor_id,
        "triplet_tag": tag,
        "stage0_planes": planes[0], "stage1_planes": planes[1], "stage2_planes": planes[2],
        "q_label": q_label, "precision": prec, "calibrator": calib_kind,
        "d_label": d_label, "d_scheme": scheme, "d_tactic": tactic,
        "d_workspace_gb": ws_mb / 1024,
        "use_dla": use_dla, "dla_core": dla_core, "gpu_fallback": gpu_fb,
        "hardware": "orin_agx_64gb",
    }
    t0 = time.time()
    engine_bytes, build_secs, build_err = build_engine(onnx, q, d, calib_npys)
    rep["build_secs"] = build_secs
    if engine_bytes is None:
        rep["build_success"] = False
        rep["status"] = "build_failed"
        rep["fail_reason"] = build_err
        rep["elapsed_secs"] = time.time() - t0
        report_path.write_text(json.dumps(rep, indent=2))
        return rep
    rep["build_success"] = True
    rep["engine_size_mb"] = len(engine_bytes) / 1024 / 1024

    lat_stats, lat_err = measure_lat(engine_bytes)
    if lat_stats is None:
        rep["status"] = "lat_failed"; rep["fail_reason"] = lat_err
    else:
        rep.update(lat_stats)
        rep["throughput_fps"] = 1000.0 / lat_stats["lat_mean_ms"] if lat_stats["lat_mean_ms"] > 0 else None
        rep["status"] = "OK"

    rep["elapsed_secs"] = time.time() - t0
    report_path.write_text(json.dumps(rep, indent=2))
    return rep


def main():
    spatial = CALIB_DIR / "pyramid_dair_collab_spatial.npy"
    tego = CALIB_DIR / "pyramid_dair_collab_tego.npy"
    calib_npys = (spatial, tego)

    total_anchors = len(TRIPLETS) * len(Q_CELLS) * len(D_CELLS)
    print(f"=" * 72)
    print(f"P0.3 Orin Class A lat-only: {len(TRIPLETS)} T × {len(Q_CELLS)} Q × {len(D_CELLS)} D = {total_anchors} anchor")
    print(f"=" * 72)

    t_start = time.time()
    results = []
    idx = 0
    for tag, planes in TRIPLETS:
        for q in Q_CELLS:
            for d in D_CELLS:
                idx += 1
                t0 = time.time()
                rep = run_anchor(tag, planes, q, d, calib_npys)
                results.append(rep)
                status = rep.get("status", "?")
                lat = rep.get("lat_p50_ms")
                lat_str = f"lat_p50={lat:.3f}ms" if lat else f"({rep.get('fail_reason', '')[:40]})"
                print(f"[{idx}/{total_anchors}] {rep['anchor_id']:38s} {status:12s} {lat_str}  ({time.time()-t0:.0f}s)")

                # Save incremental summary every 20 anchor
                if idx % 20 == 0:
                    summary = RESULT_DIR / "_summary.json"
                    summary.write_text(json.dumps({
                        "completed": idx, "total": total_anchors,
                        "wall_secs": time.time() - t_start,
                        "ok_count": sum(1 for r in results if r.get("status") == "OK"),
                        "fail_count": sum(1 for r in results if r.get("status") != "OK"),
                    }, indent=2))

    elapsed = time.time() - t_start
    ok_cnt = sum(1 for r in results if r.get("status") == "OK")
    print(f"\n[done] {idx}/{total_anchors} anchors in {elapsed/3600:.2f}h, OK={ok_cnt}, fail={idx - ok_cnt}")

    summary = RESULT_DIR / "_summary.json"
    summary.write_text(json.dumps({
        "completed": idx, "total": total_anchors,
        "wall_secs": elapsed,
        "ok_count": ok_cnt, "fail_count": idx - ok_cnt,
        "results": results,
    }, indent=2))
    print(f"summary → {summary}")


if __name__ == "__main__":
    main()
