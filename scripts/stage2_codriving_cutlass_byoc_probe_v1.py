#!/usr/bin/env python3
"""CoDriving CUTLASS BYOC smoke probe.

This is a G4-S5 diagnostic entry. It tests whether the real CoDriving
backbone-only ONNX can be converted to a CUTLASS-backed Relax module through
TVM BYOC. It records partition/build/run failures as first-class results.
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import traceback
from typing import Any


def _power_stats(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"avg": None, "p50": None, "p90": None}
    vals = sorted(float(v) for v in samples)
    return {
        "avg": sum(vals) / len(vals),
        "p50": vals[len(vals) // 2],
        "p90": vals[min(len(vals) - 1, int(round((len(vals) - 1) * 0.9)))],
    }


def _query_power_w(gpu: int) -> float:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
    return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0


def _sample_power(gpu: int, duration_s: float, interval_s: float = 0.05) -> list[float]:
    samples: list[float] = []
    deadline = time.time() + max(float(duration_s), 0.0)
    while time.time() < deadline:
        try:
            samples.append(_query_power_w(gpu))
        except Exception:
            pass
        time.sleep(interval_s)
    return samples


def _measure_energy_loop(
    vm: Any,
    vm_args: list[Any],
    dev: Any,
    gpu: int,
    *,
    energy_iters: int,
    min_active_s: float,
) -> dict[str, Any]:
    idle_samples = _sample_power(gpu, 5.0)
    active_samples: list[float] = []
    stop_sampling = threading.Event()

    def poll_power() -> None:
        while not stop_sampling.is_set():
            try:
                active_samples.append(_query_power_w(gpu))
            except Exception:
                pass
            time.sleep(0.05)

    completed_iters = 0
    start = time.time()
    sampler = threading.Thread(target=poll_power, name="codriving_cutlass_energy_sampler", daemon=True)
    sampler.start()
    try:
        while completed_iters < energy_iters or (time.time() - start) < min_active_s:
            vm["main"](*vm_args)
            completed_iters += 1
            if completed_iters % 20 == 0:
                dev.sync()
        dev.sync()
    finally:
        stop_sampling.set()
        sampler.join(timeout=1.0)

    elapsed = max(time.time() - start, 1e-9)
    idle = _power_stats(idle_samples)
    active = _power_stats(active_samples)
    idle_avg = float(idle["avg"] or 0.0)
    watt_avg = float(active["avg"] or 0.0)
    dynamic_watt_avg = max(watt_avg - idle_avg, 0.0)
    completed = max(completed_iters, 1)
    return {
        "status": "success" if active_samples else "no_power_samples",
        "joule_per_inference": dynamic_watt_avg * elapsed / float(completed),
        "idle_watt_avg": idle_avg,
        "watt_avg": watt_avg,
        "dynamic_watt_avg": dynamic_watt_avg,
        "watt_p50": active["p50"],
        "watt_p90": active["p90"],
        "elapsed_s": elapsed,
        "completed_measure_iters": completed_iters,
        "requested_measure_iters": energy_iters,
        "min_active_s": min_active_s,
        "idle_sample_count": len(idle_samples),
        "active_sample_count": len(active_samples),
    }


def _read_onnx_inputs(onnx_model: Any, batch: int) -> dict[str, tuple[int, ...]]:
    init = {i.name for i in onnx_model.graph.initializer}
    shapes: dict[str, tuple[int, ...]] = {}
    for item in onnx_model.graph.input:
        if item.name in init:
            continue
        dims = [int(d.dim_value) if int(d.dim_value) > 0 else int(batch) for d in item.type.tensor_type.shape.dim]
        shapes[item.name] = tuple(dims)
    return shapes


def _main_input_specs(mod: Any, fallback_shapes: dict[str, tuple[int, ...]]) -> list[dict[str, Any]]:
    fallback_items = list(fallback_shapes.items())
    specs: list[dict[str, Any]] = []
    try:
        params = list(mod["main"].params)
    except Exception:
        params = []
    for idx, param in enumerate(params):
        name = str(getattr(param, "name_hint", f"input_{idx}"))
        source_name, source_shape = fallback_items[idx] if idx < len(fallback_items) else (name, fallback_shapes.get(name, ()))
        sinfo = getattr(param, "struct_info", None)
        dtype = str(getattr(sinfo, "dtype", "float32"))
        shape_obj = getattr(sinfo, "shape", None)
        dims = getattr(shape_obj, "values", shape_obj)
        try:
            shape = tuple(int(dim) for dim in dims)
        except Exception:
            shape = tuple(int(dim) for dim in source_shape)
        specs.append(
            {
                "name": name,
                "source_name": source_name,
                "shape": list(shape),
                "source_shape": list(source_shape),
                "dtype": dtype,
            }
        )
    if specs:
        return specs
    return [
        {
            "name": name,
            "source_name": name,
            "shape": list(shape),
            "source_shape": list(shape),
            "dtype": "float32",
        }
        for name, shape in fallback_items
    ]


def _numpy_dtype(dtype: str) -> str:
    if dtype in {"float16", "float32", "int8", "uint8", "int32"}:
        return dtype
    return "float32"


def _attrs_to_dict(func: Any) -> dict[str, str]:
    attrs = getattr(func, "attrs", None)
    if attrs is None:
        return {}
    out: dict[str, str] = {}
    try:
        keys = list(attrs.keys())
    except Exception:
        keys = []
    for key in keys:
        try:
            out[str(key)] = str(attrs[key])
        except Exception:
            pass
    return out


def _summarize_cutlass_functions(mod: Any) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    n_codegen = 0
    n_composite = 0
    for gv, func in mod.functions_items():
        attrs = _attrs_to_dict(func)
        text = " ".join(attrs.values()).lower()
        is_cutlass = "cutlass" in text
        if is_cutlass:
            if "codegen" in {k.lower() for k in attrs}:
                n_codegen += 1
            if "composite" in {k.lower() for k in attrs}:
                n_composite += 1
            items.append({"name": gv.name_hint, "attrs": attrs})
    return {
        "n_cutlass_functions": len(items),
        "n_cutlass_codegen_functions": n_codegen,
        "n_cutlass_composite_functions": n_composite,
        "functions": items[:80],
    }


def _write_json(path: str, payload: dict[str, Any]) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnx
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.relax.backend.cuda import cutlass

    dev = tvm.cuda(args.gpu)
    target = tvm.target.Target.from_device(dev)
    model = onnx.load(args.onnx)
    shapes = _read_onnx_inputs(model, args.batch)

    result: dict[str, Any] = {
        "schema": "codriving_cutlass_byoc_probe_v1",
        "status": "started",
        "host": socket.gethostname(),
        "onnx": args.onnx,
        "gpu": args.gpu,
        "batch": args.batch,
        "graph_io_dtype": args.graph_io_dtype,
        "convert_layout": args.convert_layout,
        "reps": args.reps,
        "stages": [],
    }

    try:
        mod = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
        result["stages"].append("from_onnx_ok")
        if args.graph_io_dtype == "fp16":
            mod = relax.transform.ToMixedPrecision(out_dtype="float16")(mod)
            result["stages"].append("to_mixed_precision_fp16_ok")
        if args.convert_layout:
            desired_layouts = {
                "relax.nn.conv2d": ["NHWC", "OHWI"],
                # TOPI legalization for deconv in this TVM fork only supports
                # NCHW data with IOHW kernels. Keeping this layout avoids leaving
                # high-level conv2d_transpose in the VM codegen path after BYOC.
                "relax.nn.conv2d_transpose": ["NCHW", "IOHW"],
            }
            mod = relax.transform.ConvertLayout(desired_layouts)(mod)
            result["stages"].append("convert_layout_nhwc_ohwi_ok")

        before = _summarize_cutlass_functions(mod)
        result["cutlass_summary_before_partition"] = before
        with target, tvm.transform.PassContext(opt_level=3):
            partitioned = cutlass.partition_for_cutlass(mod, annotate_codegen=True)
        result["stages"].append("partition_for_cutlass_ok")
        after = _summarize_cutlass_functions(partitioned)
        result["cutlass_summary_after_partition"] = after
        partitioned = relax.transform.LambdaLift()(partitioned)
        result["stages"].append("lambda_lift_ok")
        result["cutlass_summary_after_lambda_lift"] = _summarize_cutlass_functions(partitioned)
        with target, tvm.transform.PassContext(opt_level=3):
            partitioned = relax.transform.LegalizeOps()(partitioned)
        result["stages"].append("legalize_remaining_ops_ok")
        result["cutlass_summary_after_legalize"] = _summarize_cutlass_functions(partitioned)
        result["input_specs"] = _main_input_specs(partitioned, shapes)

        if after["n_cutlass_functions"] == 0:
            result["status"] = "no_cutlass_partition"
            return result

        build_start = time.time()
        with target, tvm.transform.PassContext(opt_level=3):
            ex = relax.build(partitioned, target=target)
        result["build_s"] = time.time() - build_start
        result["stages"].append("relax_build_ok")

        vm = relax.VirtualMachine(ex, dev)
        rng = np.random.RandomState(20260708)
        np_feeds: dict[str, np.ndarray] = {}
        for spec in result["input_specs"]:
            source_name = str(spec["source_name"])
            source_shape = tuple(int(v) for v in spec["source_shape"])
            if source_name not in np_feeds:
                np_feeds[source_name] = rng.rand(*source_shape).astype("float32")
        vm_args = []
        for spec in result["input_specs"]:
            value = np.asarray(np_feeds[str(spec["source_name"])], dtype=_numpy_dtype(str(spec["dtype"])))
            vm_args.append(tvm.runtime.tensor(value, device=dev))

        out = vm["main"](*vm_args)
        outs = [out] if hasattr(out, "shape") else list(out)
        dev.sync()
        result["out_shapes"] = [list(o.shape) for o in outs]
        result["stages"].append("vm_run_ok")

        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            ref = sess.run(None, np_feeds)
            worst_abs = 0.0
            for idx, item in enumerate(outs):
                arr = item.numpy() if hasattr(item, "numpy") else np.from_dlpack(item)
                if idx < len(ref) and tuple(arr.shape) == tuple(ref[idx].shape):
                    worst_abs = max(worst_abs, float(np.max(np.abs(arr.astype("float32") - ref[idx].astype("float32")))))
            result["numerical_vs_ort_fp32"] = {"max_abs_err": worst_abs, "n_outputs": len(outs)}
        except Exception as exc:
            result["numerical_vs_ort_fp32"] = {"error": repr(exc)[:500]}

        for _ in range(max(3, args.reps // 10)):
            vm["main"](*vm_args)
        dev.sync()
        lats = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            vm["main"](*vm_args)
            dev.sync()
            lats.append((time.perf_counter() - t0) * 1000.0)
        lats.sort()
        result["latency_ms_p50"] = lats[len(lats) // 2]
        result["latency_ms_min"] = lats[0]
        result["latency_ms_max"] = lats[-1]
        if getattr(args, "measure_energy", False):
            energy = _measure_energy_loop(
                vm,
                vm_args,
                dev,
                args.gpu,
                energy_iters=args.energy_iters,
                min_active_s=args.energy_min_active_s,
            )
            result["energy"] = energy
            result["energy_j"] = energy["joule_per_inference"]
        result["status"] = "success"
        return result
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = repr(exc)[:1000]
        result["traceback"] = traceback.format_exc()[-5000:]
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, choices=[4, 5, 6])
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--graph-io-dtype", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--convert-layout", action="store_true")
    ap.add_argument("--max-preflight-mem-mib", type=int, default=1024)
    ap.add_argument("--measure-energy", action="store_true")
    ap.add_argument("--energy-iters", type=int, default=300)
    ap.add_argument("--energy-min-active-s", type=float, default=5.0)
    args = ap.parse_args()

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used_mib = mem.used / 1024 / 1024
        assert util.gpu <= 5 and mem_used_mib <= args.max_preflight_mem_mib, (
            f"GPU{args.gpu} not idle: util={util.gpu}% mem={mem_used_mib:.0f}MiB "
            f"(limit={args.max_preflight_mem_mib}MiB)"
        )
    except ImportError:
        pass

    result = run_probe(args)
    _write_json(args.out_json, result)
    print(f"STATUS {result.get('status')}")
    print(f"WROTE {args.out_json}")


if __name__ == "__main__":
    main()
