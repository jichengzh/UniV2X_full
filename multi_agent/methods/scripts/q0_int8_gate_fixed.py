"""Q0 TVM INT8 GATE test - FIXED version (PATH + calibration method + reuse existing QDQ)

Changes vs original:
- Reuses existing base_backbone_int8_qdq.onnx if valid (skip re-quantization)
- Uses CalibrationMethod.MinMax directly (no string)
- Sets CUDA_HOME env so tvm.contrib.nvcc finds nvcc (PATH must be pre-set externally)
- Properly handles TIR compilation step
- 300 trials MetaSchedule tuning

Must run with:
    export PATH=/usr/local/cuda-12.2/bin:$PATH
    export LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path)
    CUDA_VISIBLE_DEVICES=<idle_gpu> /exdata/jichengzhi/tvm310/bin/python q0_int8_gate_fixed.py
"""
from __future__ import annotations
import os, sys, time, traceback, json
import numpy as np

# Set CUDA_HOME for TVM nvcc discovery
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.2")

GPU_ID = int(os.environ.get("CUDA_VISIBLE_DEVICES", "4"))
WORK_DIR = f"/exdata/jichengzhi/s2_tvm/ms_q0_int8_fixed_gpu{GPU_ID}"
ONNX_FP32 = "/exdata/jichengzhi/s2_tvm/models/base_backbone.onnx"
ONNX_INT8 = "/exdata/jichengzhi/s2_tvm/models/base_backbone_int8_qdq.onnx"
OUT_JSON  = "/exdata/jichengzhi/s2_tvm/q0_int8_gate_fixed_result.json"

REPS = 200
REPEAT = 5
TRIALS_TUNE = 300  # Quick check

FP32_TUNED_US = 6319.98  # from latency_lut_pyramid.json base tuned (H800 TVM FP16)

result = {
    "label": "q0_int8_gate_fixed",
    "onnx": ONNX_FP32,
    "onnx_int8": ONNX_INT8,
    "gpu": GPU_ID,
    "status": "STARTED",
    "fp32_tuned_us_baseline": FP32_TUNED_US,
    "steps": {},
}


def save_json():
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)


def time_vm(vm, args, dev, reps=REPS, repeat=REPEAT):
    vm["main"](*args)
    dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=repeat)
    r = vf(*args)
    return r.mean * 1e6, min(r.results) * 1e6


print(f"[Q0-FIXED] === TVM INT8 GATE FIXED ===", flush=True)
print(f"[Q0-FIXED] GPU={GPU_ID} WORK={WORK_DIR}", flush=True)
print(f"[Q0-FIXED] nvcc path: {os.popen('which nvcc 2>/dev/null').read().strip()}", flush=True)

# ======== STEP 1: Check or create INT8 QDQ ONNX ========
print(f"\n[Q0-FIXED] STEP 1: INT8 QDQ ONNX (reuse if valid)", flush=True)
t_step1 = time.time()
qdq_size_mb = -1.0
try:
    import onnx
    # Check if existing QDQ is valid
    reused = False
    if os.path.exists(ONNX_INT8) and os.path.getsize(ONNX_INT8) > 100000:
        try:
            m_check = onnx.load(ONNX_INT8)
            init_chk = {i.name for i in m_check.graph.initializer}
            inputs_chk = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                          for i in m_check.graph.input if i.name not in init_chk}
            ops_chk = [n.op_type for n in m_check.graph.node]
            has_qdq = any(op in ops_chk for op in ['QuantizeLinear', 'DequantizeLinear'])
            if has_qdq and inputs_chk:
                qdq_size_mb = os.path.getsize(ONNX_INT8) / 1024 / 1024
                print(f"[Q0-FIXED] Reusing existing QDQ ONNX ({qdq_size_mb:.1f}MB) inputs={inputs_chk}", flush=True)
                reused = True
        except Exception as e:
            print(f"[Q0-FIXED] Existing QDQ invalid: {e}, will regenerate", flush=True)

    if not reused:
        print(f"[Q0-FIXED] Generating INT8 QDQ from {ONNX_FP32}", flush=True)
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
        )
        import onnxruntime as ort

        class RandomCalibReader(CalibrationDataReader):
            def __init__(self, onnx_path, n_samples=32):
                m = onnx.load(onnx_path)
                init_names = {i.name for i in m.graph.initializer}
                self.inputs = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                               for i in m.graph.input if i.name not in init_names}
                self.n = n_samples
                self.i = 0

            def get_next(self):
                if self.i >= self.n:
                    return None
                rng = np.random.RandomState(self.i)
                data = {k: rng.uniform(-2.0, 2.0, v).astype("float32") for k, v in self.inputs.items()}
                self.i += 1
                return data

        reader = RandomCalibReader(ONNX_FP32, n_samples=32)
        print(f"[Q0-FIXED] Calibration inputs: {reader.inputs}", flush=True)

        quantize_static(
            ONNX_FP32,
            ONNX_INT8,
            reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": True},
        )
        qdq_size_mb = os.path.getsize(ONNX_INT8) / 1024 / 1024
        print(f"[Q0-FIXED] INT8 QDQ generated: {qdq_size_mb:.1f}MB", flush=True)

    result["steps"]["step1_qdq"] = {
        "status": "PASS",
        "qdq_mb": round(qdq_size_mb, 2),
        "reused": reused,
    }
except Exception as e:
    traceback.print_exc()
    result["steps"]["step1_qdq"] = {"status": "FAIL", "error": str(e)[:400]}
    result["status"] = "FAIL_STEP1"
    save_json()
    sys.exit(1)
print(f"[Q0-FIXED] Step 1 done in {time.time()-t_step1:.1f}s", flush=True)


# ======== STEP 2: Import QDQ ONNX to TVM relax ========
print(f"\n[Q0-FIXED] STEP 2: Import QDQ ONNX to TVM relax", flush=True)
t_step2 = time.time()
try:
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    try:
        import tvm.s_tir.tensor_intrin.cuda
    except ImportError:
        print("[Q0-FIXED] WARNING: tvm.s_tir.tensor_intrin.cuda not found", flush=True)

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    print(f"[Q0-FIXED] TVM version: {tvm.__version__}", flush=True)
    print(f"[Q0-FIXED] Target: {target}", flush=True)

    import onnx
    m_int8 = onnx.load(ONNX_INT8)
    init_names = {i.name for i in m_int8.graph.initializer}
    shape_dict = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                  for i in m_int8.graph.input if i.name not in init_names}
    print(f"[Q0-FIXED] QDQ ONNX inputs: {shape_dict}", flush=True)

    mod_int8 = from_onnx(m_int8, shape_dict=shape_dict, keep_params_in_input=False)
    print(f"[Q0-FIXED] TVM relax module imported", flush=True)

    # Apply transforms
    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod_legalized = seq(mod_int8)
    print(f"[Q0-FIXED] LegalizeOps + FuseTIR done", flush=True)

    result["steps"]["step2_import"] = {
        "status": "PASS",
        "shape_dict": str(shape_dict),
    }
except Exception as e:
    traceback.print_exc()
    result["steps"]["step2_import"] = {"status": "FAIL", "error": str(e)[:400]}
    result["status"] = "FAIL_STEP2"
    save_json()
    sys.exit(1)
print(f"[Q0-FIXED] Step 2 done in {time.time()-t_step2:.1f}s", flush=True)


# ======== STEP 3: Default compile + benchmark ========
print(f"\n[Q0-FIXED] STEP 3: Default compile INT8 module", flush=True)
t_step3 = time.time()
def_us = -1.0
feeds_fp32 = {}
try:
    import numpy as np
    rng = np.random.RandomState(42)
    feeds_fp32 = {k: rng.uniform(-2.0, 2.0, v).astype("float32") for k, v in shape_dict.items()}

    with tvm.transform.PassContext(opt_level=3):
        ex_def = relax.build(mod_legalized, target="cuda")

    vm_def = relax.VirtualMachine(ex_def, dev)
    args_def = [tvm.runtime.tensor(feeds_fp32[k], device=dev) for k in shape_dict]

    # Warmup
    for _ in range(5):
        vm_def["main"](*args_def)
    dev.sync()

    def_us, def_mn = time_vm(vm_def, args_def, dev)
    print(f"[Q0-FIXED] DEFAULT int8 mean_us={def_us:.1f} min_us={def_mn:.1f}", flush=True)
    result["steps"]["step3_default"] = {
        "status": "PASS",
        "default_us": round(def_us, 2),
        "default_min_us": round(def_mn, 2),
        "vs_fp16_fp32_tuned": round(FP32_TUNED_US / def_us, 3) if def_us > 0 else -1,
    }
except Exception as e:
    traceback.print_exc()
    result["steps"]["step3_default"] = {"status": "FAIL", "error": str(e)[:400]}
    print(f"[Q0-FIXED] Step 3 FAILED: {e}", flush=True)
print(f"[Q0-FIXED] Step 3 done in {time.time()-t_step3:.1f}s", flush=True)


# ======== STEP 4: MetaSchedule tune + benchmark ========
print(f"\n[Q0-FIXED] STEP 4: MetaSchedule INT8 tuning (trials={TRIALS_TUNE})", flush=True)
t_step4 = time.time()
tun_us = -1.0
tc_path = "UNKNOWN"
try:
    os.makedirs(WORK_DIR, exist_ok=True)
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir.meta_schedule import relax_integration as ri

    t0 = time.time()
    db = ri.tune_relax(
        mod=mod_legalized, params={}, target=target,
        work_dir=WORK_DIR,
        max_trials_global=TRIALS_TUNE,
        seed=42,
        builder=ms.builder.LocalBuilder(timeout_sec=300.0),
    )
    tune_s = time.time() - t0
    print(f"[Q0-FIXED] MetaSchedule tuning done in {tune_s:.0f}s", flush=True)

    with target, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK_DIR)(mod_legalized)

    # Check TC usage in scheduled TIR
    tir_text = sched.script()
    wmma_builtins = {
        "tvm_mma_sync": tir_text.count("tvm_mma_sync"),
        "dp4a": tir_text.lower().count("dp4a"),
        "wmma_total": tir_text.count("wmma"),
    }
    used_wmma = wmma_builtins["tvm_mma_sync"] > 0
    used_dp4a = wmma_builtins["dp4a"] > 0
    tc_path = "WMMA" if used_wmma else ("DP4A" if used_dp4a else "SCALAR")
    print(f"[Q0-FIXED] TC path: {tc_path} wmma_builtins={wmma_builtins}", flush=True)

    # Compile the tuned schedule
    ex_tun = tvm.compile(sched, target=target)
    vm_tun = relax.VirtualMachine(ex_tun, dev)

    args_tun = [tvm.runtime.tensor(feeds_fp32[k], device=dev) for k in shape_dict]
    for _ in range(5):
        vm_tun["main"](*args_tun)
    dev.sync()

    tun_us, tun_mn = time_vm(vm_tun, args_tun, dev)
    speedup = FP32_TUNED_US / tun_us if tun_us > 0 else -1
    print(f"[Q0-FIXED] TUNED int8 mean_us={tun_us:.1f} min_us={tun_mn:.1f}", flush=True)
    print(f"[Q0-FIXED] INT8 tuned / FP16 tuned = {speedup:.3f}x", flush=True)

    result["steps"]["step4_tune"] = {
        "status": "PASS",
        "tc_path": tc_path,
        "wmma_builtins": wmma_builtins,
        "tuned_us": round(tun_us, 2),
        "tuned_min_us": round(tun_mn, 2),
        "tune_s": round(tune_s, 0),
        "speedup_vs_fp16_tuned": round(speedup, 3),
    }
except Exception as e:
    traceback.print_exc()
    result["steps"]["step4_tune"] = {"status": "FAIL", "error": str(e)[:600]}
    print(f"[Q0-FIXED] Step 4 FAILED: {e}", flush=True)
print(f"[Q0-FIXED] Step 4 done in {time.time()-t_step4:.1f}s", flush=True)


# ======== Final summary ========
result["int8_default_us"] = round(def_us, 2)
result["int8_tuned_us"] = round(tun_us, 2)
result["speedup_int8_tuned_vs_fp16_tuned"] = round(FP32_TUNED_US / tun_us, 3) if tun_us > 0 else -1
result["tc_path"] = tc_path
result["status"] = "COMPLETE"

print(f"\n[Q0-FIXED] ===== GATE RESULTS =====", flush=True)
print(f"[Q0-FIXED] FP16 tuned baseline: {FP32_TUNED_US:.1f} us", flush=True)
print(f"[Q0-FIXED] INT8 default: {def_us:.1f} us", flush=True)
print(f"[Q0-FIXED] INT8 tuned:  {tun_us:.1f} us", flush=True)
if tun_us > 0:
    print(f"[Q0-FIXED] INT8 tuned / FP16 tuned speedup: {FP32_TUNED_US/tun_us:.3f}x", flush=True)
print(f"[Q0-FIXED] TC path: {tc_path}", flush=True)

save_json()
print(f"[Q0-FIXED] DONE — Result JSON: {OUT_JSON}", flush=True)
