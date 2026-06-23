"""
CD-A2: CoDriving backbone.resnet ONNX export + TRT FP16 build + benchmark
GPU: locked to CUDA_VISIBLE_DEVICES (set before calling this script)
Outputs: models/cd_backbone_resnet_fp32.onnx
         models/cd_backbone_resnet_fp16.engine
         results/CD_A2_trt_fp16_4090.csv

Oral 口径说明:
  - pytorch_fp32: CUDA Event, backbone.resnet only, input [2,64,192,576], warmup 200, runs 500
  - pytorch_fp16: same, model.half()
  - trt_fp16_internal: TRT engine INetworkDefinition profiler / polygraphy-style avg from trt runner
  - trt_fp16_cudaevent: TRT Python runtime + CUDA Event, same input shape, warmup 200, runs 500
"""

import sys
import os
import time
import csv
import json
import torch
import torch.nn as nn
import numpy as np

# ---- ensure V2Xverse is importable -----------------------------------------
REPO = "/home/jichengzhi/V2Xverse"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
DEVICE = torch.device(f"cuda:{GPU_ID}")

CKPT = "/home/jichengzhi/V2Xverse/checkpoints/codriving/perception/net_epoch_bestval_at16.pth"
ONNX_PATH = "/home/jichengzhi/V2X/models/cd_backbone_resnet_fp32.onnx"
ENGINE_PATH = "/home/jichengzhi/V2X/models/cd_backbone_resnet_fp16.engine"
CSV_PATH = "/home/jichengzhi/V2X/results/CD_A2_trt_fp16_4090.csv"

WARMUP = 200
RUNS = 500
# input: B=2 (max_cav=2), C=64, H=192, W=576
INPUT_SHAPE = (2, 64, 192, 576)


# ============================================================
# 1. Build backbone.resnet sub-module wrapper
# ============================================================

from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock

class ResNetWrapper(nn.Module):
    """Wrap ResNetModified so ONNX export sees fixed-size tuple outputs."""
    def __init__(self, resnet: ResNetModified):
        super().__init__()
        self.resnet = resnet

    def forward(self, x):
        feats = self.resnet(x)   # list of 3 tensors
        return feats[0], feats[1], feats[2]


def load_backbone_resnet():
    """Load backbone.resnet weights from full centerpointcodriving ckpt."""
    # config.yaml uses Python-specific YAML tags (spconv grid_size) that
    # safe_load cannot parse. Use hardcoded params verified from audit doc.
    # From codriving_structure_audit_v1.md §1.2 + config.yaml §model.args:
    layer_nums = [3, 4, 5]
    layer_strides = [2, 2, 2]
    num_filters = [64, 128, 256]

    resnet = ResNetModified(
        BasicBlock,
        layers=layer_nums,
        layer_strides=layer_strides,
        num_filters=num_filters,
        inplanes=64,
    )

    # Load weights from full ckpt
    ckpt = torch.load(CKPT, map_location="cpu")
    # Support both {'model': state_dict} and flat state_dict
    if isinstance(ckpt, dict) and 'model' in ckpt:
        full_sd = ckpt['model']
    elif isinstance(ckpt, dict) and all(not k.startswith('{') for k in ckpt):
        full_sd = ckpt
    else:
        full_sd = ckpt

    # Filter keys for backbone.resnet
    prefix = "backbone.resnet."
    resnet_sd = {}
    for k, v in full_sd.items():
        if k.startswith(prefix):
            resnet_sd[k[len(prefix):]] = v

    print(f"  Loaded {len(resnet_sd)} keys for backbone.resnet (out of {len(full_sd)} total)")
    missing, unexpected = resnet.load_state_dict(resnet_sd, strict=True)
    if missing:
        print(f"  WARNING: Missing keys: {missing[:5]}")
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected[:5]}")

    return resnet


# ============================================================
# 2. PyTorch FP32 / FP16 benchmark (CUDA Event)
# ============================================================

def cuda_event_benchmark(model, x_in, warmup=WARMUP, runs=RUNS, label=""):
    """Returns mean_ms, p50_ms, p99_ms."""
    model.eval()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_in)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start_ev.record()
            _ = model(x_in)
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))

    times = np.array(times)
    mean_ms = float(np.mean(times))
    p50_ms = float(np.percentile(times, 50))
    p99_ms = float(np.percentile(times, 99))
    print(f"  [{label}] mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
    return mean_ms, p50_ms, p99_ms


# ============================================================
# 3. ONNX Export + Sanity Check
# ============================================================

def export_onnx(wrapper, x_dummy):
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    print(f"\n[ONNX] Exporting to {ONNX_PATH} ...")
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            x_dummy.cpu(),
            ONNX_PATH,
            opset_version=13,
            input_names=["bev_features"],
            output_names=["feat_stage0", "feat_stage1", "feat_stage2"],
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(ONNX_PATH) / 1e6
    print(f"  ONNX saved: {size_mb:.1f} MB")
    return size_mb


def onnx_sanity(wrapper, x_dummy):
    """Compare ONNX Runtime output vs PyTorch output. Report maxdiff."""
    import onnxruntime as ort
    print("\n[ONNX Sanity] Comparing ORT vs PyTorch FP32 ...")
    wrapper.eval()
    x_cpu = x_dummy.cpu()
    with torch.no_grad():
        pt_outs = wrapper(x_cpu)  # tuple of 3 tensors (cpu)

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"bev_features": x_cpu.numpy()})

    maxdiffs = []
    for i, (pt, ort_o) in enumerate(zip(pt_outs, ort_outs)):
        diff = float(np.abs(pt.numpy() - ort_o).max())
        maxdiffs.append(diff)
        print(f"  stage{i}: maxdiff={diff:.6f}  pt_shape={tuple(pt.shape)}")

    overall_maxdiff = max(maxdiffs)
    print(f"  Overall maxdiff: {overall_maxdiff:.6f}  {'PASS' if overall_maxdiff < 1e-3 else 'FAIL'}")
    return overall_maxdiff, maxdiffs


# ============================================================
# 4. TRT FP16 Build
# ============================================================

def build_trt_engine():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"\n[TRT] Parsing ONNX: {ONNX_PATH} ...")
    with open(ONNX_PATH, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  Parser error: {parser.get_error(i).desc()}")
        raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    # workspace 4 GiB
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1 << 30)
    # FP16
    config.set_flag(trt.BuilderFlag.FP16)

    print(f"  Building FP16 engine ... (this may take 2-5 min)")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build failed: serialized is None")
    elapsed = time.time() - t0
    print(f"  Build done in {elapsed:.1f}s")

    os.makedirs(os.path.dirname(ENGINE_PATH), exist_ok=True)
    with open(ENGINE_PATH, "wb") as f:
        f.write(serialized)
    engine_mb = os.path.getsize(ENGINE_PATH) / 1e6
    print(f"  Engine saved: {ENGINE_PATH} ({engine_mb:.1f} MB)")
    return engine_mb


# ============================================================
# 5. TRT FP16 Benchmark (CUDA Event via TRT Python runtime)
# ============================================================

def trt_cudaevent_benchmark(warmup=WARMUP, runs=RUNS):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    print(f"\n[TRT Benchmark] Loading engine: {ENGINE_PATH}")
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate IO bindings
    import ctypes
    # TRT 10.x API
    num_io = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io)]
    print(f"  TRT IO tensors: {tensor_names}")

    # Prepare input tensor
    x_in = torch.randn(INPUT_SHAPE, dtype=torch.float16, device=DEVICE).contiguous()

    # Set tensor addresses
    context.set_input_shape(tensor_names[0], INPUT_SHAPE)

    # Pre-allocate output tensors
    output_tensors = []
    for name in tensor_names[1:]:
        shape = tuple(context.get_tensor_shape(name))
        out = torch.empty(shape, dtype=torch.float16, device=DEVICE)
        output_tensors.append((name, out))

    # Set all tensor pointers
    context.set_tensor_address(tensor_names[0], x_in.data_ptr())
    for name, out in output_tensors:
        context.set_tensor_address(name, out.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream

    # Warmup
    print(f"  Warmup {warmup} iters ...")
    for _ in range(warmup):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    # Benchmark
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(runs):
        start_ev.record()
        context.execute_async_v3(stream)
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev))

    times = np.array(times)
    mean_ms = float(np.mean(times))
    p50_ms = float(np.percentile(times, 50))
    p99_ms = float(np.percentile(times, 99))
    print(f"  [TRT FP16 CUDA Event] mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
    return mean_ms, p50_ms, p99_ms


# ============================================================
# 6. TRT internal profiler (measures kernel-only latency)
# ============================================================

def trt_internal_profiler_benchmark(warmup=50, runs=100):
    """Use TRT IProfiler to get kernel-level timing (analogous to trtexec --iterations)."""
    import tensorrt as trt

    class SimpleProfiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.layer_times = {}

        def report_layer_time(self, layer_name, ms):
            self.layer_times[layer_name] = self.layer_times.get(layer_name, 0) + ms

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    profiler = SimpleProfiler()
    context.profiler = profiler

    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    x_in = torch.randn(INPUT_SHAPE, dtype=torch.float16, device=DEVICE).contiguous()
    context.set_input_shape(tensor_names[0], INPUT_SHAPE)

    output_tensors = []
    for name in tensor_names[1:]:
        shape = tuple(context.get_tensor_shape(name))
        out = torch.empty(shape, dtype=torch.float16, device=DEVICE)
        output_tensors.append((name, out))

    context.set_tensor_address(tensor_names[0], x_in.data_ptr())
    for name, out in output_tensors:
        context.set_tensor_address(name, out.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream

    # Warmup (profiler active from start)
    for _ in range(warmup):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    profiler.layer_times = {}
    for _ in range(runs):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()

    total_ms = sum(profiler.layer_times.values()) / runs
    print(f"  [TRT Internal Profiler] total kernel time per iter: {total_ms:.3f}ms ({len(profiler.layer_times)} layers)")
    return total_ms


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CD-A2: CoDriving backbone.resnet TRT FP16 benchmark")
    print(f"GPU: cuda:{GPU_ID} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    print("=" * 60)

    # --- Verify GPU is idle ---
    import subprocess
    smi = subprocess.run(
        ["nvidia-smi", f"--id={GPU_ID}", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    util, mem = smi.stdout.strip().split(",")
    util, mem = int(util.strip()), int(mem.strip())
    print(f"\nGPU {GPU_ID} status: util={util}% mem={mem}MiB")
    if util > 5 or mem > 200:
        raise RuntimeError(f"GPU {GPU_ID} not idle (util={util}% mem={mem}MiB). Abort.")

    results = []

    # ---- Load model ----
    print("\n[Step 1] Loading backbone.resnet weights ...")
    resnet = load_backbone_resnet()
    wrapper = ResNetWrapper(resnet)

    # ---- PyTorch FP32 baseline ----
    print("\n[Step 2] PyTorch FP32 benchmark ...")
    x_fp32 = torch.randn(INPUT_SHAPE, dtype=torch.float32, device=DEVICE)
    wrapper.to(DEVICE).eval()
    pt_fp32_mean, pt_fp32_p50, pt_fp32_p99 = cuda_event_benchmark(
        wrapper, x_fp32, label="PyTorch FP32"
    )
    results.append({
        "module": "backbone.resnet",
        "precision": "fp32",
        "tool": "cudaevent",
        "mean_ms": pt_fp32_mean,
        "p50_ms": pt_fp32_p50,
        "p99_ms": pt_fp32_p99,
        "speedup_vs": "1.00x (baseline)",
        "caveat": f"PyTorch FP32 CUDA Event, input {INPUT_SHAPE}, warmup={WARMUP} runs={RUNS}, GPU{GPU_ID}",
    })

    # ---- ONNX Export (must come before FP16 benchmark which mutates resnet) ----
    print("\n[Step 3-pre] ONNX Export (before FP16 to avoid dtype conflict) ...")
    # Load a fresh FP32 copy of resnet for ONNX export
    resnet_fp32_for_onnx = load_backbone_resnet()
    wrapper_cpu = ResNetWrapper(resnet_fp32_for_onnx).cpu().eval()
    x_dummy = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    onnx_size_mb = export_onnx(wrapper_cpu, x_dummy)

    # ---- ONNX Sanity ----
    print("\n[Step 3-sanity] ONNX Runtime sanity check ...")
    maxdiff, per_stage_diffs = onnx_sanity(wrapper_cpu, x_dummy)

    # ---- PyTorch FP16 ----
    print("\n[Step 3] PyTorch FP16 benchmark ...")
    x_fp16 = x_fp32.half()
    wrapper_fp16 = ResNetWrapper(resnet).to(DEVICE).half().eval()
    pt_fp16_mean, pt_fp16_p50, pt_fp16_p99 = cuda_event_benchmark(
        wrapper_fp16, x_fp16, label="PyTorch FP16"
    )
    speedup_fp16 = pt_fp32_p50 / pt_fp16_p50
    results.append({
        "module": "backbone.resnet",
        "precision": "fp16",
        "tool": "cudaevent",
        "mean_ms": pt_fp16_mean,
        "p50_ms": pt_fp16_p50,
        "p99_ms": pt_fp16_p99,
        "speedup_vs": f"{speedup_fp16:.2f}x vs PyTorch FP32 p50",
        "caveat": f"PyTorch FP16 CUDA Event, input {INPUT_SHAPE}, warmup={WARMUP} runs={RUNS}, GPU{GPU_ID}",
    })

    # ---- TRT FP16 Build ----
    print("\n[Step 6] TRT FP16 Build ...")
    engine_size_mb = build_trt_engine()

    # ---- TRT FP16 CUDA Event benchmark ----
    print("\n[Step 7] TRT FP16 CUDA Event benchmark ...")
    trt_mean, trt_p50, trt_p99 = trt_cudaevent_benchmark()
    speedup_vs_fp32 = pt_fp32_p50 / trt_p50
    speedup_vs_fp16 = pt_fp16_p50 / trt_p50
    results.append({
        "module": "backbone.resnet",
        "precision": "fp16_trt",
        "tool": "cudaevent",
        "mean_ms": trt_mean,
        "p50_ms": trt_p50,
        "p99_ms": trt_p99,
        "speedup_vs": f"{speedup_vs_fp32:.2f}x vs PyTorch FP32 p50 / {speedup_vs_fp16:.2f}x vs PyTorch FP16 p50",
        "caveat": f"TRT FP16 engine CUDA Event, input {INPUT_SHAPE}, warmup={WARMUP} runs={RUNS}, GPU{GPU_ID}, engine={engine_size_mb:.1f}MB",
    })

    # ---- TRT Internal Profiler ----
    print("\n[Step 8] TRT internal profiler (kernel-only) ...")
    try:
        trt_internal_ms = trt_internal_profiler_benchmark()
        results.append({
            "module": "backbone.resnet",
            "precision": "fp16_trt",
            "tool": "trt_internal_profiler",
            "mean_ms": trt_internal_ms,
            "p50_ms": "N/A",
            "p99_ms": "N/A",
            "speedup_vs": f"{pt_fp32_mean / trt_internal_ms:.2f}x vs PyTorch FP32 mean",
            "caveat": f"TRT IProfiler sum-of-kernel-ms/iter, warmup=50 runs=100, GPU{GPU_ID}",
        })
    except Exception as e:
        print(f"  TRT internal profiler failed: {e}")

    # ---- e2e synthesis estimate ----
    print("\n[Step 9] E2E synthesis estimate ...")
    # backbone.resnet is called TWICE: once standalone + once inside fusion_net
    # From CD_A1: perception_e2e FP32 = 10.166ms
    # backbone.resnet FP32 p50 = 2.756ms (known), called 2x = 5.512ms
    # Other modules (deblocks 0.965, shrink 1.270, fusion_attn 0.81ms, heads 0.082) FP32 = ~3.127ms
    # TRT FP16 for backbone: trt_p50 * 2 (called twice)
    # Other modules remain PyTorch FP32 (not TRT-ized)
    backbone_saved_per_call = pt_fp32_p50 - trt_p50
    e2e_fp32_known = 10.031  # p50 from CD_A1
    e2e_trt_estimate = e2e_fp32_known - backbone_saved_per_call * 2  # 2 calls
    speedup_e2e = e2e_fp32_known / e2e_trt_estimate if e2e_trt_estimate > 0 else float('nan')

    e2e_note = (
        f"[合成估算-非真测] "
        f"e2e FP32 p50={e2e_fp32_known:.3f}ms (CD_A1真测); "
        f"backbone.resnet TRT FP16 p50={trt_p50:.3f}ms, saves {backbone_saved_per_call:.3f}ms per call x2 calls = {backbone_saved_per_call*2:.3f}ms; "
        f"其余模块保持PyTorch FP32; "
        f"估e2e={e2e_trt_estimate:.3f}ms speedup={speedup_e2e:.2f}x"
    )
    print(f"  {e2e_note}")
    results.append({
        "module": "perception_e2e",
        "precision": "fp16_trt_resnet_only",
        "tool": "synthesis_estimate",
        "mean_ms": "N/A",
        "p50_ms": f"{e2e_trt_estimate:.3f}",
        "p99_ms": "N/A",
        "speedup_vs": f"{speedup_e2e:.2f}x vs e2e FP32",
        "caveat": e2e_note,
    })

    # ---- Write CSV ----
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fieldnames = ["module", "precision", "tool", "mean_ms", "p50_ms", "p99_ms", "speedup_vs", "caveat"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Done] Results written to {CSV_PATH}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ONNX size: {onnx_size_mb:.1f} MB")
    print(f"TRT FP16 engine size: {engine_size_mb:.1f} MB")
    print(f"ONNX sanity maxdiff: {maxdiff:.6f} ({'PASS' if maxdiff < 1e-3 else 'FAIL'})")
    print(f"  Per-stage maxdiffs: {[f'{d:.6f}' for d in per_stage_diffs]}")
    print(f"PyTorch FP32: mean={pt_fp32_mean:.3f}ms  p50={pt_fp32_p50:.3f}ms  p99={pt_fp32_p99:.3f}ms")
    print(f"PyTorch FP16: mean={pt_fp16_mean:.3f}ms  p50={pt_fp16_p50:.3f}ms  p99={pt_fp16_p99:.3f}ms  speedup={speedup_fp16:.2f}x")
    print(f"TRT FP16:     mean={trt_mean:.3f}ms  p50={trt_p50:.3f}ms  p99={trt_p99:.3f}ms  speedup_vs_fp32={speedup_vs_fp32:.2f}x  vs_fp16={speedup_vs_fp16:.2f}x")
    print(f"E2E estimate (合成): {e2e_trt_estimate:.3f}ms  speedup={speedup_e2e:.2f}x [合成估算-非真测]")

    return results


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    main()
