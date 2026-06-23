"""Q0-mini: Just the DEFAULT benchmark for base INT8 (no tuning).
Runs fast (~30s total). Used to gate the TVM INT8 pipeline.
Requires: existing base_backbone_int8_qdq.onnx
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.2")

GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "4"))
ONNX_INT8 = "/exdata/jichengzhi/s2_tvm/models/base_backbone_int8_qdq.onnx"
OUT_JSON = "/exdata/jichengzhi/s2_tvm/q0_int8_default_gate.json"

REPS = 200
REPEAT = 5
FP16_TUNED_US = 6319.98

print(f"[Q0-mini] GPU={GPU_ID} checking nvidia-smi...", flush=True)
os.system("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader")

result = {"gpu": GPU_ID, "status": "STARTED", "fp16_tuned_baseline_us": FP16_TUNED_US}

import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
import onnx

print(f"[Q0-mini] TVM {tvm.__version__}", flush=True)

dev = tvm.cuda(0)
target = tvm.target.Target.from_device(dev)
print(f"[Q0-mini] Target: {target}", flush=True)

m_int8 = onnx.load(ONNX_INT8)
init_names = {i.name for i in m_int8.graph.initializer}
shape_dict = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
              for i in m_int8.graph.input if i.name not in init_names}
print(f"[Q0-mini] QDQ inputs: {shape_dict}", flush=True)

mod_int8 = from_onnx(m_int8, shape_dict=shape_dict, keep_params_in_input=False)

seq = tvm.transform.Sequential([
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),
])
with target, tvm.transform.PassContext(opt_level=3):
    mod_legalized = seq(mod_int8)
print(f"[Q0-mini] Legalized OK", flush=True)

# DEFAULT benchmark
with tvm.transform.PassContext(opt_level=3):
    ex_def = relax.build(mod_legalized, target="cuda")
vm_def = relax.VirtualMachine(ex_def, dev)

rng = np.random.RandomState(42)
feeds = {k: rng.uniform(-2.0, 2.0, v).astype("float32") for k, v in shape_dict.items()}
args = [tvm.runtime.tensor(feeds[k], device=dev) for k in shape_dict]

# Check GPU immediately before benchmark
print(f"[Q0-mini] GPU check immediately before benchmark:", flush=True)
os.system("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader")

# Warmup
for _ in range(5):
    vm_def["main"](*args)
dev.sync()

# Benchmark
vf = vm_def.time_evaluator("main", dev, number=REPS, repeat=REPEAT)
r = vf(*args)
def_us = r.mean * 1e6
def_mn = min(r.results) * 1e6

print(f"[Q0-mini] DEFAULT int8: mean={def_us:.1f}us min={def_mn:.1f}us", flush=True)
print(f"[Q0-mini] vs FP16 tuned {FP16_TUNED_US:.1f}us: ratio={FP16_TUNED_US/def_us:.3f}x (expected >1 since INT8 default is untuned)", flush=True)

result["int8_default_us"] = round(def_us, 2)
result["int8_default_min_us"] = round(def_mn, 2)
result["fp16_tuned_us"] = FP16_TUNED_US
result["int8_default_vs_fp16_tuned"] = round(def_us / FP16_TUNED_US, 3)
result["status"] = "DEFAULT_ONLY_COMPLETE"
result["note"] = "Default schedule only; tuning skipped due to GPU contention from sw-optimizer experiments"

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"[Q0-mini] Saved to {OUT_JSON}", flush=True)
print(f"[Q0-mini] DONE", flush=True)
