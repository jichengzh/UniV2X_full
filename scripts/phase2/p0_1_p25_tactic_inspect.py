"""P0-1·hw mechanism (ISS-014) — rebuild p25/p50 INT8 with DETAILED profiling
verbosity, extract the ACTUAL per-layer tactic/kernel name for the stage0
grouped-conv layers, to give POSITIVE evidence that p25(48) lands on a different
(slower) TRT tactic than aligned p50(32) — i.e. a kernel-selection cliff, NOT
IMMA padding (already falsified) and NOT reformat (ruled out by IProfiler).

Rebuilds to TEMP engines (does NOT overwrite the validated stage_a_cache ones).
Reuses the existing INT8 MinMax calib cache (no re-calibration). 4090 host, GPU7.
Output: results/P0_1_p25_tactic_inspect.json + console diff of stage0 conv2 tactic.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HW_GPU", "7"))
import tensorrt as trt  # noqa: E402

LOG = trt.Logger(trt.Logger.ERROR)

TARGETS = [
    ("p25_int8", "models/stage_a_cache/pruned25.onnx",
     "models/stage_a_cache/pruned25_int8_calib.cache"),
    ("p50_int8", "models/stage_a_cache/pruned50.onnx",
     "models/stage_a_cache/pruned50_int8_calib.cache"),
]


class CacheCalib(trt.IInt8MinMaxCalibrator):
    """Cache-only calibrator: returns the prebuilt MinMax cache, no fresh data."""
    def __init__(self, cache_path):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._cache = Path(cache_path).read_bytes()

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        return None  # no data -> TRT must use the cache

    def read_calibration_cache(self):
        return self._cache

    def write_calibration_cache(self, cache):
        return None


def build_detailed(onnx_path, calib_cache):
    builder = trt.Builder(LOG)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOG)
    ok = parser.parse(Path(onnx_path).read_bytes())
    if not ok:
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"onnx parse failed: {errs}")
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator = CacheCalib(calib_cache)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # <- key
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        raise RuntimeError("build returned None")
    eng = trt.Runtime(LOG).deserialize_cuda_engine(ser)
    return eng


def tactic_of(eng, name_substrs):
    insp = eng.create_engine_inspector()
    d = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    out = []
    for l in d.get("Layers", []):
        if not isinstance(l, dict):
            continue
        nm = str(l.get("Name", ""))
        if all(s in nm for s in name_substrs):
            out.append({k: l.get(k) for k in l
                        if k in ("Name", "LayerType", "TacticValue", "TacticName",
                                 "Kernel", "ParameterType", "InputDataType",
                                 "OutputDataType", "OutputName")})
    return out, list(d["Layers"][0].keys()) if d.get("Layers") else []


def main():
    results = {}
    for tag, onnx, cache in TARGETS:
        print(f"\n=== building {tag} (DETAILED) from {onnx} ===")
        eng = build_detailed(str(REPO / onnx), str(REPO / cache))
        # stage0 conv2 = the cliff layers; also grab a conv from layer1 for contrast
        s0, keys = tactic_of(eng, ["layer0", "conv2"])
        print(f"  available layer-info keys: {keys}")
        print(f"  stage0 conv2 layers found: {len(s0)}")
        for L in s0[:3]:
            print(f"    Name={str(L.get('Name'))[:60]}")
            for k in ("LayerType", "TacticName", "TacticValue", "Kernel",
                      "InputDataType", "OutputDataType"):
                if L.get(k) is not None:
                    print(f"      {k}: {str(L[k])[:80]}")
        results[tag] = {"available_keys": keys, "stage0_conv2": s0}
    out = REPO / "results/P0_1_p25_tactic_inspect.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
