"""
CD-A2 TRT Benchmark only (engine already built).
Tests TRT FP16 CUDA Event latency for backbone.resnet.
Also runs ONNX sanity check.
"""
import sys, os, time
import numpy as np
import torch
import csv

GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
DEVICE = torch.device(f"cuda:{GPU_ID}")
ENGINE_PATH = "/home/jichengzhi/V2X/models/cd_backbone_resnet_fp16.engine"
ONNX_PATH = "/home/jichengzhi/V2X/models/cd_backbone_resnet_fp32.onnx"
CSV_PATH = "/home/jichengzhi/V2X/results/CD_A2_trt_fp16_4090.csv"
INPUT_SHAPE = (2, 64, 192, 576)
WARMUP = 200
RUNS = 500

REPO = "/home/jichengzhi/V2Xverse"
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def gpu_check():
    import subprocess
    smi = subprocess.run(
        ["nvidia-smi", f"--id={GPU_ID}",
         "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    util, mem = smi.stdout.strip().split(",")
    util, mem = int(util.strip()), int(mem.strip())
    print(f"GPU {GPU_ID}: util={util}% mem={mem}MiB")
    if util > 5 or mem > 300:
        raise RuntimeError(f"GPU {GPU_ID} busy (util={util}% mem={mem}MiB)")
    return util, mem


def onnx_sanity():
    """Check ONNX output vs PyTorch on CPU."""
    import onnxruntime as ort
    from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock

    class ResNetWrapper(torch.nn.Module):
        def __init__(self, r): super().__init__(); self.resnet = r
        def forward(self, x): f = self.resnet(x); return f[0], f[1], f[2]

    resnet = ResNetModified(BasicBlock, [3,4,5], [2,2,2], [64,128,256], inplanes=64)
    ckpt = torch.load("/home/jichengzhi/V2Xverse/checkpoints/codriving/perception/net_epoch_bestval_at16.pth", map_location="cpu")
    prefix = "backbone.resnet."
    sd = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    resnet.load_state_dict(sd, strict=True)
    wrapper = ResNetWrapper(resnet).eval()

    x = torch.randn(INPUT_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        pt_outs = wrapper(x)

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"bev_features": x.numpy()})

    maxdiffs = []
    for i, (pt, ort_o) in enumerate(zip(pt_outs, ort_outs)):
        d = float(np.abs(pt.numpy() - ort_o).max())
        maxdiffs.append(d)
        print(f"  stage{i}: pt_shape={tuple(pt.shape)}  maxdiff={d:.6f}")
    overall = max(maxdiffs)
    print(f"  Overall maxdiff: {overall:.6f}  {'PASS' if overall < 1e-3 else 'WARN (>1e-3)'}")
    return overall, maxdiffs


def pytorch_bench(precision="fp32"):
    from opencood.models.sub_modules.resblock import ResNetModified, BasicBlock

    class ResNetWrapper(torch.nn.Module):
        def __init__(self, r): super().__init__(); self.resnet = r
        def forward(self, x): f = self.resnet(x); return f[0], f[1], f[2]

    resnet = ResNetModified(BasicBlock, [3,4,5], [2,2,2], [64,128,256], inplanes=64)
    ckpt = torch.load("/home/jichengzhi/V2Xverse/checkpoints/codriving/perception/net_epoch_bestval_at16.pth", map_location="cpu")
    prefix = "backbone.resnet."
    sd = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    resnet.load_state_dict(sd, strict=True)

    if precision == "fp16":
        model = ResNetWrapper(resnet).to(DEVICE).half().eval()
        x = torch.randn(INPUT_SHAPE, dtype=torch.float16, device=DEVICE)
    else:
        model = ResNetWrapper(resnet).to(DEVICE).eval()
        x = torch.randn(INPUT_SHAPE, dtype=torch.float32, device=DEVICE)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(RUNS):
            start_ev.record()
            _ = model(x)
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))

    times = np.array(times)
    mean_ms = float(np.mean(times))
    p50_ms = float(np.percentile(times, 50))
    p99_ms = float(np.percentile(times, 99))
    print(f"  [PyTorch {precision.upper()}] mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
    return mean_ms, p50_ms, p99_ms


def trt_cudaevent_bench():
    """TRT FP16 CUDA Event benchmark. Uses execute_async_v3 with proper stream."""
    import tensorrt as trt
    print(f"  TRT version: {trt.__version__}")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"  Engine deserialized OK, num_io_tensors={engine.num_io_tensors}")

    context = engine.create_execution_context()

    # Get tensor names
    n = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(n)]
    input_names = [nm for nm in tensor_names if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT]
    output_names = [nm for nm in tensor_names if engine.get_tensor_mode(nm) == trt.TensorIOMode.OUTPUT]
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")

    # Set input shape
    assert len(input_names) == 1
    context.set_input_shape(input_names[0], INPUT_SHAPE)

    # Allocate buffers
    x_in = torch.randn(INPUT_SHAPE, dtype=torch.float16, device=DEVICE).contiguous()
    outputs = {}
    for nm in output_names:
        shape = tuple(context.get_tensor_shape(nm))
        print(f"  Output {nm}: shape={shape}")
        outputs[nm] = torch.empty(shape, dtype=torch.float16, device=DEVICE).contiguous()

    # Set tensor addresses
    context.set_tensor_address(input_names[0], x_in.data_ptr())
    for nm, buf in outputs.items():
        context.set_tensor_address(nm, buf.data_ptr())

    # Use a dedicated CUDA stream (not default stream per TRT warning)
    cuda_stream = torch.cuda.Stream(device=DEVICE)

    # Warmup
    print(f"  Warming up {WARMUP} iters ...")
    with torch.cuda.stream(cuda_stream):
        for _ in range(WARMUP):
            context.execute_async_v3(cuda_stream.cuda_stream)
    torch.cuda.synchronize()

    # Benchmark
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    times = []

    with torch.cuda.stream(cuda_stream):
        for _ in range(RUNS):
            start_ev.record(cuda_stream)
            context.execute_async_v3(cuda_stream.cuda_stream)
            end_ev.record(cuda_stream)
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))

    times = np.array(times)
    mean_ms = float(np.mean(times))
    p50_ms = float(np.percentile(times, 50))
    p99_ms = float(np.percentile(times, 99))
    print(f"  [TRT FP16 CUDA Event] mean={mean_ms:.3f}ms  p50={p50_ms:.3f}ms  p99={p99_ms:.3f}ms")
    return mean_ms, p50_ms, p99_ms


def trt_internal_profiler_bench():
    """TRT IProfiler to get kernel-level timing."""
    import tensorrt as trt

    class SimpleProfiler(trt.IProfiler):
        def __init__(self):
            super().__init__()
            self.layer_times = {}
            self.call_count = 0
        def report_layer_time(self, layer_name, ms):
            self.layer_times[layer_name] = self.layer_times.get(layer_name, 0.0) + ms
            self.call_count += 1

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    profiler = SimpleProfiler()
    context.profiler = profiler

    n = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(n)]
    input_names = [nm for nm in tensor_names if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT]
    output_names = [nm for nm in tensor_names if engine.get_tensor_mode(nm) == trt.TensorIOMode.OUTPUT]

    context.set_input_shape(input_names[0], INPUT_SHAPE)
    x_in = torch.randn(INPUT_SHAPE, dtype=torch.float16, device=DEVICE).contiguous()
    outputs = {nm: torch.empty(tuple(context.get_tensor_shape(nm)), dtype=torch.float16, device=DEVICE).contiguous()
               for nm in output_names}

    context.set_tensor_address(input_names[0], x_in.data_ptr())
    for nm, buf in outputs.items():
        context.set_tensor_address(nm, buf.data_ptr())

    cuda_stream = torch.cuda.Stream(device=DEVICE)
    PROF_RUNS = 100

    # Warmup 50
    with torch.cuda.stream(cuda_stream):
        for _ in range(50):
            context.execute_async_v3(cuda_stream.cuda_stream)
    torch.cuda.synchronize()

    profiler.layer_times = {}
    profiler.call_count = 0

    with torch.cuda.stream(cuda_stream):
        for _ in range(PROF_RUNS):
            context.execute_async_v3(cuda_stream.cuda_stream)
    torch.cuda.synchronize()

    # When profiler is active, TRT runs synchronously (1 call = 1 report per layer)
    n_layers = len(profiler.layer_times)
    total_per_iter = sum(profiler.layer_times.values()) / PROF_RUNS
    print(f"  [TRT IProfiler] {n_layers} layers, total kernel={total_per_iter:.3f}ms/iter")
    return total_per_iter


def main():
    print("=" * 60)
    print("CD-A2: TRT FP16 Benchmark (engine already built)")
    print(f"GPU: cuda:{GPU_ID}")
    print("=" * 60)

    gpu_check()

    results = []

    # ONNX sanity
    print("\n[ONNX Sanity]")
    maxdiff, per_stage = onnx_sanity()

    # PyTorch FP32
    print("\n[PyTorch FP32]")
    fp32_mean, fp32_p50, fp32_p99 = pytorch_bench("fp32")
    results.append({
        "module": "backbone.resnet", "precision": "fp32", "tool": "cudaevent",
        "mean_ms": round(fp32_mean, 3), "p50_ms": round(fp32_p50, 3), "p99_ms": round(fp32_p99, 3),
        "speedup_vs": "1.00x (baseline)",
        "caveat": f"PyTorch FP32 CUDA Event input{INPUT_SHAPE} warmup={WARMUP} runs={RUNS} GPU{GPU_ID}",
    })

    # PyTorch FP16
    print("\n[PyTorch FP16]")
    fp16_mean, fp16_p50, fp16_p99 = pytorch_bench("fp16")
    spd_fp16 = fp32_p50 / fp16_p50
    results.append({
        "module": "backbone.resnet", "precision": "fp16_pytorch", "tool": "cudaevent",
        "mean_ms": round(fp16_mean, 3), "p50_ms": round(fp16_p50, 3), "p99_ms": round(fp16_p99, 3),
        "speedup_vs": f"{spd_fp16:.2f}x vs PyTorch FP32 p50",
        "caveat": f"PyTorch FP16 CUDA Event input{INPUT_SHAPE} warmup={WARMUP} runs={RUNS} GPU{GPU_ID}",
    })

    # TRT FP16 CUDA Event
    print("\n[TRT FP16 CUDA Event]")
    trt_mean, trt_p50, trt_p99 = trt_cudaevent_bench()
    spd_trt_vs_fp32 = fp32_p50 / trt_p50
    spd_trt_vs_fp16 = fp16_p50 / trt_p50
    engine_mb = os.path.getsize(ENGINE_PATH) / 1e6
    results.append({
        "module": "backbone.resnet", "precision": "fp16_trt", "tool": "cudaevent",
        "mean_ms": round(trt_mean, 3), "p50_ms": round(trt_p50, 3), "p99_ms": round(trt_p99, 3),
        "speedup_vs": f"{spd_trt_vs_fp32:.2f}x vs PT FP32 p50 / {spd_trt_vs_fp16:.2f}x vs PT FP16 p50",
        "caveat": f"TRT FP16 engine CUDA Event input{INPUT_SHAPE} warmup={WARMUP} runs={RUNS} GPU{GPU_ID} engine={engine_mb:.1f}MB",
    })

    # TRT IProfiler
    print("\n[TRT IProfiler]")
    try:
        prof_ms = trt_internal_profiler_bench()
        results.append({
            "module": "backbone.resnet", "precision": "fp16_trt", "tool": "trt_iprofiler",
            "mean_ms": round(prof_ms, 3), "p50_ms": "N/A", "p99_ms": "N/A",
            "speedup_vs": f"{fp32_mean / prof_ms:.2f}x vs PT FP32 mean",
            "caveat": f"TRT IProfiler sum-kernel-ms/iter warmup=50 runs=100 GPU{GPU_ID}",
        })
    except Exception as e:
        print(f"  IProfiler failed: {e}")

    # E2E synthesis estimate
    # backbone.resnet called TWICE in CoDriving: main path + fusion_net internal
    # CD_A1 e2e p50=10.031ms (FP32); backbone.resnet FP32 p50=2.756ms (CD_A1 true measurement)
    # With TRT FP16, save (2.756 - trt_p50) * 2 per frame
    backbone_fp32_cd_a1 = 2.756  # p50 from CD_A1, true measurement, same call pattern
    saved_per_call = backbone_fp32_cd_a1 - trt_p50
    e2e_fp32_p50 = 10.031  # from CD_A1
    e2e_trt_estimate = e2e_fp32_p50 - saved_per_call * 2
    spd_e2e = e2e_fp32_p50 / e2e_trt_estimate if e2e_trt_estimate > 0 else float('nan')

    note = (
        f"[合成估算-非真测] "
        f"CD_A1 e2e FP32 p50={e2e_fp32_p50}ms; "
        f"backbone.resnet CD_A1 FP32 p50={backbone_fp32_cd_a1}ms (真测); "
        f"本次TRT FP16 p50={trt_p50:.3f}ms (真测); "
        f"saves {saved_per_call:.3f}ms x2calls={saved_per_call*2:.3f}ms; "
        f"其余模块PyTorch FP32不变; "
        f"est e2e={e2e_trt_estimate:.3f}ms speedup={spd_e2e:.2f}x"
    )
    print(f"\n[E2E Synthesis Estimate] {note}")
    results.append({
        "module": "perception_e2e", "precision": "fp16_trt_resnet_only", "tool": "synthesis_estimate",
        "mean_ms": "N/A", "p50_ms": round(e2e_trt_estimate, 3), "p99_ms": "N/A",
        "speedup_vs": f"{spd_e2e:.2f}x vs e2e FP32 [合成估算]",
        "caveat": note,
    })

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    fields = ["module", "precision", "tool", "mean_ms", "p50_ms", "p99_ms", "speedup_vs", "caveat"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nResults written to {CSV_PATH}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"GPU used: {GPU_ID}")
    print(f"ONNX: {os.path.getsize(ONNX_PATH)/1e6:.1f}MB  Engine: {engine_mb:.1f}MB")
    print(f"ONNX sanity: maxdiff={maxdiff:.6f} {'PASS' if maxdiff < 1e-3 else 'WARN'}")
    print(f"  per-stage diffs: {[f'{d:.6f}' for d in per_stage]}")
    print(f"PyTorch FP32: mean={fp32_mean:.3f}  p50={fp32_p50:.3f}  p99={fp32_p99:.3f}")
    print(f"PyTorch FP16: mean={fp16_mean:.3f}  p50={fp16_p50:.3f}  p99={fp16_p99:.3f}  speedup={spd_fp16:.2f}x")
    print(f"TRT FP16:     mean={trt_mean:.3f}  p50={trt_p50:.3f}  p99={trt_p99:.3f}")
    print(f"  vs PT FP32: {spd_trt_vs_fp32:.2f}x  vs PT FP16: {spd_trt_vs_fp16:.2f}x")
    print(f"E2E estimate: {e2e_trt_estimate:.3f}ms  speedup={spd_e2e:.2f}x  [合成估算-非真测]")


if __name__ == "__main__":
    torch.cuda.set_device(GPU_ID)
    main()
