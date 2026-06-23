"""Build pure-FP32 TRT engine from stage_a_cache/base.onnx (no fp16/int8 flags).
For quantization-axis table: FP32 -> FP16 -> INT8 at base anchor."""
import json, time
from pathlib import Path
import tensorrt as trt

ROOT = Path("/home/jichengzhi/UniV2X/models/stage_a_cache")
ONNX, ENGINE = ROOT / "base.onnx", ROOT / "base_fp32.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
assert parser.parse(ONNX.read_bytes()), [parser.get_error(i) for i in range(parser.num_errors)]

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
# NO fp16/int8 flags -> pure FP32

t0 = time.time()
serialized = builder.build_serialized_network(network, config)
assert serialized is not None, "build failed"
ENGINE.write_bytes(serialized)
meta = {"precision": "fp32", "onnx": str(ONNX), "engine": str(ENGINE),
        "engine_size_mb": ENGINE.stat().st_size / 1e6, "build_secs": time.time() - t0,
        "trt_version": trt.__version__}
(ROOT / "base_fp32_build.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
