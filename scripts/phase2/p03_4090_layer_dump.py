"""P0-3 ISS-017 gate-1 — dump 4090 INT8 per-layer precision set (DETAILED build).

The validated 4090 stage_a_cache/*_int8.engine were built WITHOUT DETAILED
verbosity (no TacticName), so we rebuild base/p50/p75 INT8 with DETAILED + the
SAME calib cache, and dump per-layer precision (int8/fp16/fp32 via tactic token)
for cross-platform comparison vs Orin (gate 1: layer precision set == 4090?).

Builds to memory only (no engine file written). 4090 GPU, not timing-sensitive.
Output: results/P03_4090_int8_layers.json
"""
import os, sys, json
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HW_GPU", "7"))
import tensorrt as trt

REPO = Path(__file__).resolve().parents[2]
SA = REPO / "models/stage_a_cache"
LOG = trt.Logger(trt.Logger.ERROR)

CONFIGS = [
    ("base", "base.onnx", "base_int8_calib.cache"),
    ("p50", "pruned50.onnx", "pruned50_int8_calib.cache"),
    ("p75", "pruned75.onnx", "pruned75_int8_calib.cache"),
]


class CacheCalib(trt.IInt8MinMaxCalibrator):
    def __init__(self, c):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._c = Path(c).read_bytes()
    def get_batch_size(self): return 1
    def get_batch(self, n): return None
    def read_calibration_cache(self): return self._c
    def write_calibration_cache(self, c): return None


def prec_token(t):
    t = (t or "").lower()
    if "i8i8" in t or "imma" in t or "_i8_" in t: return "int8"
    if "h884" in t or "h1688" in t or "hmma" in t or "f16f16" in t: return "fp16"
    if "f32f32" in t or "f32" in t: return "fp32"
    return "other"


def build_dump(onnx, cache):
    b = trt.Builder(LOG)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    trt.OnnxParser(net, LOG).parse(Path(onnx).read_bytes())
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator = CacheCalib(cache)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    ser = b.build_serialized_network(net, cfg)
    eng = trt.Runtime(LOG).deserialize_cuda_engine(ser)
    insp = eng.create_engine_inspector()
    d = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    counts = {"int8": 0, "fp16": 0, "fp32": 0, "other": 0}
    per_conv = {}
    for l in d.get("Layers", []):
        if not isinstance(l, dict): continue
        p = prec_token(str(l.get("TacticName", "")))
        counts[p] += 1
        nm = str(l.get("Name", ""))
        if "conv" in nm.lower():
            per_conv[nm[:70]] = p
    return counts, per_conv


def main():
    out = {}
    for tag, onnx, cache in CONFIGS:
        counts, per_conv = build_dump(str(SA / onnx), str(SA / cache))
        out[tag] = {"counts": counts, "conv_precision": per_conv}
        print(f"[{tag}] int8={counts['int8']} fp16={counts['fp16']} "
              f"fp32={counts['fp32']} other={counts['other']}  convs={len(per_conv)}")
    (REPO / "results/P03_4090_int8_layers.json").write_text(json.dumps(out, indent=2))
    print("done -> results/P03_4090_int8_layers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
