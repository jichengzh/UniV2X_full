"""Q1 INT8 key-pair worker — runs ONE width in its own process.

Usage:
    CUDA_VISIBLE_DEVICES=<gpu> python q1_int8_pairs_worker.py <label> <onnx_fp32> <out_csv> [trials] [reps]

The script:
1. Quantizes fp32 ONNX to INT8 QDQ (reuses existing file if valid)
2. Imports to TVM relax
3. Benchmarks DEFAULT schedule (no tuning)
4. Tunes with MetaSchedule (trials=500 default)
5. Benchmarks TUNED schedule
6. Appends one row to out_csv per measurement (default/tuned)

CRITICAL: run each label in its own SUBPROCESS (not loop) to avoid:
- Stale TVM db giving fake ~1.0x ratios
- CUDA illegal-access crashes from GPU state pollution
"""
from __future__ import annotations
import os, sys, time, traceback, json
import numpy as np

# PATH fix
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.2")
os.environ["PATH"] = "/usr/local/cuda-12.2/bin:" + os.environ.get("PATH", "")

LABEL  = sys.argv[1]
ONNX_FP32 = sys.argv[2]
OUT_CSV = sys.argv[3]
TRIALS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
REPS   = int(sys.argv[5]) if len(sys.argv) > 5 else 200
REPEAT = 5
SEED   = 42

ONNX_INT8 = ONNX_FP32.replace(".onnx", "_int8_qdq.onnx")
WORK_DIR  = f"/exdata/jichengzhi/tvm_int8/{LABEL}_{int(time.time())}"

print(f"[{LABEL}] === Q1 INT8 pair worker ===", flush=True)
print(f"[{LABEL}] ONNX_FP32={ONNX_FP32}", flush=True)
print(f"[{LABEL}] ONNX_INT8={ONNX_INT8}", flush=True)
print(f"[{LABEL}] WORK_DIR={WORK_DIR}", flush=True)
print(f"[{LABEL}] trials={TRIALS} reps={REPS}", flush=True)
print(f"[{LABEL}] nvcc={os.popen('which nvcc 2>/dev/null').read().strip()}", flush=True)


def write_row(label, sched_type, lat_us):
    """Append one row to out_csv."""
    header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a") as f:
        if header:
            f.write("label,prec,sched,lat_us,source,notes\n")
        f.write(f"{label},int8,{sched_type},{lat_us:.2f},H800_TVM_int8,fresh_workdir\n")
    print(f"[{LABEL}] wrote row: {sched_type} lat={lat_us:.1f}us", flush=True)


def time_vm(vm, args, dev, reps, repeat):
    vm["main"](*args)
    dev.sync()
    vf = vm.time_evaluator("main", dev, number=reps, repeat=repeat)
    r = vf(*args)
    return r.mean * 1e6, min(r.results) * 1e6


# ---- STEP 1: Quantize FP32 ONNX to INT8 QDQ ----
print(f"\n[{LABEL}] STEP 1: INT8 QDQ quantization", flush=True)
try:
    import onnx

    reused = False
    if os.path.exists(ONNX_INT8) and os.path.getsize(ONNX_INT8) > 50000:
        try:
            m_chk = onnx.load(ONNX_INT8)
            ops_chk = [n.op_type for n in m_chk.graph.node]
            if any(op in ops_chk for op in ['QuantizeLinear', 'DequantizeLinear']):
                init_chk = {i.name for i in m_chk.graph.initializer}
                sh_chk = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                          for i in m_chk.graph.input if i.name not in init_chk}
                if sh_chk:
                    print(f"[{LABEL}] Reusing existing QDQ ({os.path.getsize(ONNX_INT8)/1024/1024:.1f}MB) inputs={sh_chk}", flush=True)
                    reused = True
        except Exception as e_chk:
            print(f"[{LABEL}] Existing QDQ invalid ({e_chk}), regenerating", flush=True)

    if not reused:
        from onnxruntime.quantization import (
            quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
        )

        class RandomCalibReader(CalibrationDataReader):
            def __init__(self, path, n=32):
                m = onnx.load(path)
                init_names = {i.name for i in m.graph.initializer}
                self.inps = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                             for i in m.graph.input if i.name not in init_names}
                self.n = n; self.i = 0
            def get_next(self):
                if self.i >= self.n: return None
                rng = np.random.RandomState(self.i)
                d = {k: rng.uniform(-2.0, 2.0, v).astype("float32") for k, v in self.inps.items()}
                self.i += 1
                return d

        reader = RandomCalibReader(ONNX_FP32)
        print(f"[{LABEL}] Calibration inputs: {reader.inps}", flush=True)
        quantize_static(
            ONNX_FP32, ONNX_INT8, reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            extra_options={"ActivationSymmetric": True},
        )
        print(f"[{LABEL}] QDQ generated: {os.path.getsize(ONNX_INT8)/1024/1024:.1f}MB", flush=True)
except Exception as e:
    traceback.print_exc()
    print(f"[{LABEL}] STEP 1 FAILED: {e}", flush=True)
    sys.exit(1)


# ---- STEP 2: Import + legalize ----
print(f"\n[{LABEL}] STEP 2: TVM import + legalize", flush=True)
try:
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    try:
        import tvm.s_tir.tensor_intrin.cuda
    except ImportError:
        print(f"[{LABEL}] WARNING: tensor_intrin.cuda not found", flush=True)

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    print(f"[{LABEL}] TVM {tvm.__version__} target={target}", flush=True)

    import onnx
    m_int8 = onnx.load(ONNX_INT8)
    init_names = {i.name for i in m_int8.graph.initializer}
    shape_dict = {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
                  for i in m_int8.graph.input if i.name not in init_names}
    print(f"[{LABEL}] QDQ inputs: {shape_dict}", flush=True)

    mod_int8 = from_onnx(m_int8, shape_dict=shape_dict, keep_params_in_input=False)

    seq = tvm.transform.Sequential([
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])
    with target, tvm.transform.PassContext(opt_level=3):
        mod_legalized = seq(mod_int8)
    print(f"[{LABEL}] Legalized OK", flush=True)
except Exception as e:
    traceback.print_exc()
    print(f"[{LABEL}] STEP 2 FAILED: {e}", flush=True)
    sys.exit(1)


# ---- STEP 3: Default benchmark ----
print(f"\n[{LABEL}] STEP 3: Default schedule benchmark", flush=True)
def_us = -1.0
feeds_fp32 = {}
try:
    rng = np.random.RandomState(SEED)
    feeds_fp32 = {k: rng.uniform(-2.0, 2.0, v).astype("float32") for k, v in shape_dict.items()}

    with tvm.transform.PassContext(opt_level=3):
        ex_def = relax.build(mod_legalized, target="cuda")
    vm_def = relax.VirtualMachine(ex_def, dev)
    args_def = [tvm.runtime.tensor(feeds_fp32[k], device=dev) for k in shape_dict]

    for _ in range(5):
        vm_def["main"](*args_def)
    dev.sync()

    def_us, def_mn = time_vm(vm_def, args_def, dev, REPS, REPEAT)
    print(f"[{LABEL}] DEFAULT int8: mean={def_us:.1f}us min={def_mn:.1f}us", flush=True)
    write_row(LABEL, "default", def_us)
except Exception as e:
    traceback.print_exc()
    print(f"[{LABEL}] STEP 3 FAILED: {e}", flush=True)
    write_row(LABEL, "default", -1.0)


# ---- STEP 4: MetaSchedule tuning + benchmark ----
print(f"\n[{LABEL}] STEP 4: MetaSchedule tune (trials={TRIALS}) + benchmark", flush=True)
tun_us = -1.0
try:
    os.makedirs(WORK_DIR, exist_ok=True)
    from tvm.s_tir import meta_schedule as ms
    from tvm.s_tir.meta_schedule import relax_integration as ri

    t0 = time.time()
    ri.tune_relax(
        mod=mod_legalized, params={}, target=target,
        work_dir=WORK_DIR,
        max_trials_global=TRIALS,
        seed=SEED,
        builder=ms.builder.LocalBuilder(timeout_sec=300.0),
    )
    tune_s = time.time() - t0
    print(f"[{LABEL}] Tuning done in {tune_s:.0f}s", flush=True)

    with target, tvm.transform.PassContext(opt_level=3):
        sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK_DIR)(mod_legalized)

    # Check TC path
    tir_text = sched.script()
    tc_path = ("WMMA" if "tvm_mma_sync" in tir_text
               else ("DP4A" if "dp4a" in tir_text.lower()
               else "SCALAR"))
    print(f"[{LABEL}] TC path: {tc_path}", flush=True)

    ex_tun = tvm.compile(sched, target=target)
    vm_tun = relax.VirtualMachine(ex_tun, dev)
    args_tun = [tvm.runtime.tensor(feeds_fp32[k], device=dev) for k in shape_dict]

    for _ in range(5):
        vm_tun["main"](*args_tun)
    dev.sync()

    tun_us, tun_mn = time_vm(vm_tun, args_tun, dev, REPS, REPEAT)
    print(f"[{LABEL}] TUNED int8: mean={tun_us:.1f}us min={tun_mn:.1f}us tc_path={tc_path}", flush=True)
    write_row(LABEL, "tuned", tun_us)
except Exception as e:
    traceback.print_exc()
    print(f"[{LABEL}] STEP 4 FAILED: {e}", flush=True)
    write_row(LABEL, "tuned", -1.0)


# ---- Summary ----
print(f"\n[{LABEL}] ===== SUMMARY =====", flush=True)
print(f"[{LABEL}] int8 default: {def_us:.1f}us", flush=True)
print(f"[{LABEL}] int8 tuned:  {tun_us:.1f}us", flush=True)
if def_us > 0 and tun_us > 0:
    print(f"[{LABEL}] tune speedup: {def_us/tun_us:.3f}x", flush=True)
print(f"[{LABEL}] DONE", flush=True)
