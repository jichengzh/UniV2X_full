"""Full-network (encoder->heads) TRT engine speed bench — T1_base_Q_{fp32,fp16,int8_mm}.
Inputs: voxel_features randn / voxel_num_points in [1,32] / voxel_coords valid
in-grid indices / mask ones / t_ego identity. Values don't change kernel
selection; coords kept valid to avoid OOB in scatter.
CUDA-Event p50, 200 warmup / 300 measure, single stream, idle GPU."""
import sys, json
from pathlib import Path
import numpy as np, torch, tensorrt as trt, pynvml

REPO = Path("/home/jichengzhi/UniV2X")
sys.path.insert(0, str(REPO / "scripts" / "phase2"))
from e5_collab2_energy_bench import reverify_idle

GPU = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ENGINES = {
    "fp32": "models/e2e_cache/T1_base_Q_fp32.engine",
    "fp16": "models/e2e_cache/T1_base_Q_fp16.engine",
    "int8_mm": "models/e2e_cache/T1_base_Q_int8_mm.engine",
}
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
_T2T = {trt.float32: torch.float32, trt.float16: torch.float16,
        trt.int32: torch.int32, trt.int8: torch.int8, trt.bool: torch.bool}

pynvml.nvmlInit(); handle = pynvml.nvmlDeviceGetHandleByIndex(GPU)
dirty, util, others = reverify_idle(handle, GPU, 50.0)
print(f"[GPU{GPU}] util={util}% other_mem={others:.0f}MiB"); assert not dirty
torch.cuda.set_device(GPU)
rt = trt.Runtime(TRT_LOGGER)
torch.manual_seed(42)

results = {}
for tag, rel in ENGINES.items():
    path = REPO / rel
    eng = rt.deserialize_cuda_engine(path.read_bytes())
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        nm = eng.get_tensor_name(i)
        shape = tuple(eng.get_tensor_shape(nm))
        dt = _T2T.get(eng.get_tensor_dtype(nm), torch.float32)
        if eng.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
            if nm == "voxel_features":
                t = torch.randn(shape, device="cuda").to(dt)
            elif nm == "voxel_num_points":
                t = torch.randint(1, 33, shape, device="cuda").to(dt)
            elif nm == "voxel_coords":   # (N,4) keep all indices small & valid
                t = torch.randint(0, 64, shape, device="cuda").to(dt)
                t[:, 0] = torch.randint(0, 2, (shape[0],), device="cuda").to(dt)  # agent idx
                t[:, 1] = 0  # z
            elif nm == "voxel_mask":
                t = torch.ones(shape, device="cuda").to(dt)
            elif nm == "t_ego":          # identity 2x3 affine per agent
                base = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device="cuda")
                t = base.unsqueeze(0).repeat(shape[0],1,1).to(dt)
            else:
                t = torch.zeros(shape, device="cuda").to(dt)
        else:
            t = torch.empty(shape, dtype=dt, device="cuda")
        bufs[nm] = t; ctx.set_tensor_address(nm, int(t.data_ptr()))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(200): ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    lats = []
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(300):
        with torch.cuda.stream(stream):
            s.record(stream); ctx.execute_async_v3(stream.cuda_stream); e.record(stream)
        stream.synchronize(); lats.append(s.elapsed_time(e))
    a = np.array(lats)
    results[tag] = {"lat_p50_ms": round(float(np.percentile(a,50)),4),
                    "lat_mean_ms": round(float(a.mean()),4),
                    "lat_p99_ms": round(float(np.percentile(a,99)),4),
                    "engine_size_mb": round(path.stat().st_size/1e6,3),
                    "engine": rel}
    print(tag, results[tag])
    del ctx, eng, bufs; torch.cuda.empty_cache()

out = REPO / "results/e2e_engine_speed_bench.json"
out.write_text(json.dumps({"gpu": GPU, "n_warmup": 200, "n_measure": 300,
    "input_caps": "32k voxels (static shape, engine processes cap regardless of fill)",
    "results": results}, indent=2))
print("saved", out)
pynvml.nvmlShutdown()
