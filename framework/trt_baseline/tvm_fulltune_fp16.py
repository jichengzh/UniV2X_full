#!/usr/bin/env python3
"""Decisive full-budget TVM fp16 tune of the Pyramid backbone ONNX (Phase 2.A L-A).

Answers: "how fast can TVM REALLY go on this grouped-conv backbone with a proper
metaschedule budget + tensor-core intrinsics?" — vs the framework's no-fresh-tune
hand-rewrite (6.52ms) and TRT-FP16 (0.88ms).

Pipeline: ONNX -> relax -> ToMixedPrecision(fp16) -> lower -> metaschedule
tune_relax(max_trials_global=N, fresh workdir, CUDA tensor intrinsics registered)
-> apply DB -> compile -> measure p50 (warmup + repeat) + energy (NVML watt_avg*lat).

Runs in the tvm310 env (/exdata/jichengzhi/tvm310/bin/python).
"""
import argparse
import json
import os
import statistics
import sys
import time

TVM_SITE = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"
NVLIBS_PATH_FILE = "/exdata/jichengzhi/tvm_nvlibs.path"  # colon-joined bundled nvidia libs
CUDA_BIN = "/usr/local/cuda-12.2/bin"


def setup_env(gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    parts = []
    if os.path.exists(NVLIBS_PATH_FILE):
        with open(NVLIBS_PATH_FILE) as f:
            parts.append(f.read().strip())
    parts.append(f"{TVM_SITE}/tvm/lib")
    old = os.environ.get("LD_LIBRARY_PATH", "")
    if old:
        parts.append(old)
    os.environ["LD_LIBRARY_PATH"] = ":".join(p for p in parts if p)
    os.environ["PATH"] = CUDA_BIN + ":" + os.environ.get("PATH", "")
    if TVM_SITE not in sys.path:
        sys.path.insert(0, TVM_SITE)


def build_relax_fp16(onnx_path, target):
    import onnx
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx

    model = onnx.load(onnx_path)
    shapes = {
        item.name: tuple(d.dim_value for d in item.type.tensor_type.shape.dim)
        for item in model.graph.input
    }
    mod = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
    # fp16 mixed precision at the relax op level (before legalize).
    try:
        mod = relax.transform.ToMixedPrecision(out_dtype="float16")(mod)
        mixed = True
    except Exception as e:  # keep going in fp32 if the pass signature differs
        print(f"[warn] ToMixedPrecision failed ({e!r}); falling back to fp32 tune", flush=True)
        mixed = False
    with target, tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ])
        modt = seq(mod)
    return modt, shapes, mixed


def measure(vm, inputs, dev, warmup, iters, repeat, gpu_abs, energy_secs):
    import pynvml
    for _ in range(warmup):
        vm["main"](*inputs)
    dev.sync()
    # p50 latency via TVM time_evaluator (same tool the framework's measure uses).
    # number=1, repeat=iters*repeat -> per-call samples comparable to the TRT harness.
    tf = vm.time_evaluator("main", dev, number=1, repeat=iters * repeat)
    res = tf(*inputs)
    samples_ms = sorted(t * 1000.0 for t in res.results)
    lat_p50 = statistics.median(samples_ms)
    lat_mean = statistics.mean(samples_ms)
    # energy: NVML power over >= energy_secs active window
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(gpu_abs)
    watts, t_end, n = [], time.time() + energy_secs, 0
    while time.time() < t_end or n < iters:
        vm["main"](*inputs)
        n += 1
        if n % 10 == 0:
            dev.sync()
            watts.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
    dev.sync()
    watt_avg = statistics.mean(watts) if watts else float("nan")
    pynvml.nvmlShutdown()
    return {
        "lat_p50_ms": lat_p50, "lat_mean_ms": lat_mean,
        "lat_p99_ms": samples_ms[int(0.99 * len(samples_ms)) - 1],
        "watt_avg": watt_avg, "energy_j": watt_avg * lat_p50 / 1000.0,
        "n_lat_samples": len(samples_ms), "n_watt_samples": len(watts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--max-trials", type=int, default=4000)
    ap.add_argument("--work-dir", required=True, help="FRESH workdir (per discipline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--energy-secs", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["tune", "measure", "both"], default="both",
                    help="tune and measure MUST run in separate processes to avoid "
                         "CUDA_ERROR_ILLEGAL_ADDRESS (see feedback-tvm-tune-apply-fresh-workdir)")
    args = ap.parse_args()

    setup_env(args.gpu)
    import numpy as np
    import tvm
    import tvm.s_tir.tensor_intrin.cuda  # noqa: F401  registers CUDA WMMA intrinsics
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri

    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError(f"CUDA device not visible for physical GPU {args.gpu}")
    target = tvm.target.Target.from_device(dev)
    so_path = os.path.join(args.work_dir, "tuned_fp16.so")
    specs_path = os.path.join(args.work_dir, "input_specs.json")

    # ---- tune + BUILD + export .so (must NOT run in the same process as measure:
    # in-process build+execute triggers CUDA_ERROR_ILLEGAL_ADDRESS. The framework's
    # working path exports a .so here and load_module()s it fresh in measure). ----
    if args.mode in ("tune", "both"):
        os.makedirs(args.work_dir, exist_ok=True)
        modt, shapes, mixed = build_relax_fp16(args.onnx, target)
        t_tune = time.time()
        ri.tune_relax(mod=modt, params={}, target=target, work_dir=args.work_dir,
                      max_trials_global=args.max_trials, seed=args.seed)
        tune_s = round(time.time() - t_tune, 1)
        with target, tvm.transform.PassContext(opt_level=3):
            scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=args.work_dir)(modt)
            ex = tvm.compile(scheduled, target=target)
        ex.export_library(so_path)
        in_specs = [([int(x) for x in p.struct_info.shape], str(p.struct_info.dtype))
                    for p in modt["main"].params]
        with open(specs_path, "w") as f:
            json.dump({"in_specs": in_specs, "mixed_precision_applied": mixed,
                       "max_trials": args.max_trials, "tune_s": tune_s}, f, indent=1)
        print(f"[tune+build done] max_trials={args.max_trials} tune_s={tune_s} -> {so_path}", flush=True)
        if args.mode == "tune":
            return

    # ---- measure: load the pre-built .so fresh (framework's working pattern) ----
    meta = json.load(open(specs_path))
    in_specs = meta["in_specs"]
    mixed = meta.get("mixed_precision_applied")
    print(f"[main input specs] {in_specs}", flush=True)
    lib = tvm.runtime.load_module(so_path)
    vm = relax.VirtualMachine(lib, dev)
    inputs = [tvm.runtime.tensor(np.random.randn(*shp).astype(dt), device=dev)
              for shp, dt in in_specs]
    prof = measure(vm, inputs, dev, args.warmup, args.iters, args.repeat, args.gpu, args.energy_secs)
    shapes = {f"in{i}": spc[0] for i, spc in enumerate(in_specs)}
    tune_meta = meta
    rec = {
        "framework": "tvm_fulltune", "precision": "fp16", "mixed_precision_applied": mixed,
        "onnx": os.path.basename(args.onnx), "input_shapes": {k: list(v) for k, v in shapes.items()},
        "max_trials": args.max_trials, "tune_s": tune_meta.get("tune_s"),
        "work_dir": args.work_dir, "gpu_abs": args.gpu,
        "caliber": "backbone-subnet 128x256 batch2; p50 time_evaluator number=1 x(iters*repeat); "
                   "NVML watt_avg*lat (same as measure_config & trt_profile_v1)",
        **prof,
    }
    print(json.dumps(rec, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[out] {args.out}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
