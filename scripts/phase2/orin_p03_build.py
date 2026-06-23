"""P0-3 Orin engine builder (runs ON Orin, TRT 8.5.2.2, uniad_venv).

Builds collab2 {base,pruned50,pruned75} x {fp16,int8} engines for GPU route.
INT8 uses a cache-only IInt8MinMaxCalibrator (returns the 4090-exported MinMax
calib cache) so algorithm type matches (MinMax) and TRT reuses cached scales
without re-calibration data. (trtexec --calib failed: TRT10 cache header didn't
match trtexec's default Entropy calibrator -> tried to recalibrate w/o data.)

Run on Orin:
  cd /home/jichengzhi/m4_8_orin/p03_build
  /home/jichengzhi/uniad_venv/bin/python orin_p03_build.py
"""
import sys
from pathlib import Path
import tensorrt as trt

HERE = Path("/home/jichengzhi/m4_8_orin/p03_build")
LOG = trt.Logger(trt.Logger.WARNING)

CONFIGS = [
    ("base", "base.onnx", "base_int8_calib.cache"),
    ("p50", "pruned50.onnx", "pruned50_int8_calib.cache"),
    ("p75", "pruned75.onnx", "pruned75_int8_calib.cache"),
]


class CacheCalib(trt.IInt8MinMaxCalibrator):
    def __init__(self, cache_path):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self._c = Path(cache_path).read_bytes()

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        return None

    def read_calibration_cache(self):
        return self._c

    def write_calibration_cache(self, cache):
        return None


def build(onnx, precision, calib_cache=None):
    b = trt.Builder(LOG)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, LOG)
    if not parser.parse(Path(onnx).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"parse fail {onnx}: {errs}")
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        cfg.set_flag(trt.BuilderFlag.FP16)  # allow fp16 fallback
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = CacheCalib(calib_cache)
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        raise RuntimeError(f"build returned None ({onnx},{precision})")
    return bytes(ser)


def main():
    for tag, onnx, cache in CONFIGS:
        for prec in ("fp16", "int8"):
            out = HERE / f"{tag}_{prec}_orin.engine"
            if out.exists():
                print(f"[skip] {out.name} exists ({out.stat().st_size/1e6:.2f}MB)")
                continue
            try:
                eng = build(str(HERE / onnx), prec,
                            str(HERE / cache) if prec == "int8" else None)
                out.write_bytes(eng)
                print(f"[ok] {out.name}  {len(eng)/1e6:.2f}MB")
            except Exception as e:
                print(f"[FAIL] {tag}_{prec}: {str(e)[:200]}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
