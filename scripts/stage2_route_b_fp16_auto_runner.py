#!/usr/bin/env python3
"""Route B fp16 automatic per-width runner.

The old Route B fp16 fix was a one-off H800 script for 64x128x256.  This
runner exposes the same measurement level as Route B-int8: label/width/ONNX are
CLI inputs, grouped-conv split sizes are detected from each PrimFunc, and
build/measure can be run in separate processes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(os.environ.get("STAGE2_V2X_ROOT", "/home/jichengzhi/V2X"))
DEFAULT_OUT_DIR = Path("/exdata/jichengzhi/s2_tvm/route_b_fp16_auto_20260707")
DEFAULT_TVM_SITE = Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages")
DEFAULT_ONNX = Path("/exdata/jichengzhi/s2_tvm/models/smbo_64x128x256_backbone.onnx")
NVLIBS_PATH = Path("/exdata/jichengzhi/tvm_nvlibs.path")
CAPABILITY_PATH = REPO / (
    "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "raw/int8_native_route/stage2_h800_native_int8_capability_probe.py"
)

_BUFFER4_RE = re.compile(
    r'T\.Buffer\(\((?:T\.int64\()?(\d+)\)?,\s*'
    r'(?:T\.int64\()?(\d+)\)?,\s*'
    r'(?:T\.int64\()?(\d+)\)?,\s*'
    r'(?:T\.int64\()?(\d+)\)?\),\s*"([^"]+)"\)'
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_width(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parts = [int(item) for item in value]
    else:
        parts = [int(item.strip()) for item in str(value).replace("x", ",").split(",") if item.strip()]
    if not parts:
        raise ValueError(f"width must contain at least one value, got {value!r}")
    if any(item <= 0 for item in parts):
        raise ValueError(f"width values must be positive, got {value!r}")
    return tuple(parts)


def width_csv(value: Any) -> str:
    return ",".join(str(item) for item in parse_width(value))


def safe_label(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def precision_of(args: argparse.Namespace) -> str:
    return str(getattr(args, "precision", "fp16"))


def _read_nvlibs() -> str:
    try:
        return NVLIBS_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_h800_env(
    *,
    gpu: str,
    tvm_site: Path,
    existing_env: dict[str, str] | None = None,
    nvlibs_text: str | None = None,
) -> dict[str, str]:
    env = dict(existing_env or os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = "/usr/local/cuda-12.2/bin:" + env.get("PATH", "")
    ld_parts = [
        str(Path(tvm_site) / "nvidia/cuda_runtime/lib"),
        str(Path(tvm_site) / "tvm/lib"),
    ]
    nvlibs = _read_nvlibs() if nvlibs_text is None else nvlibs_text
    if nvlibs:
        ld_parts.append(nvlibs)
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    return env


def plan_paths(args: argparse.Namespace) -> dict[str, Path]:
    label = safe_label(args.label)
    label_dir = Path(args.out_dir) / label
    precision = precision_of(args)
    stem = f"route_b_{precision}_auto"
    return {
        "label_dir": label_dir,
        "work_dir": label_dir / "ms_work_dir",
        "artifact_so": label_dir / f"{stem}.so",
        "ref_so": label_dir / f"route_b_{precision}_default_ref.so",
        "input_specs": label_dir / "input_specs.json",
        "build_json": label_dir / f"{stem}_build.json",
        "result_json": label_dir / f"{stem}_result.json",
        "latest_json": Path(args.out_dir) / f"{stem}_latest.json",
    }


def self_phase_command(python_bin: Path, args: argparse.Namespace, phase: str) -> list[str]:
    cmd = [
        str(python_bin),
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--label",
        str(args.label),
        "--width",
        width_csv(args.width),
        "--onnx",
        str(args.onnx),
        "--out-dir",
        str(args.out_dir),
        "--gpu",
        str(args.gpu),
        "--tvm-site",
        str(args.tvm_site),
        "--max-trials",
        str(args.max_trials),
        "--seed",
        str(args.seed),
        "--fix",
        str(args.fix),
        "--precision",
        precision_of(args),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--repeat",
        str(args.repeat),
        "--energy-iters",
        str(args.energy_iters),
        "--idle-timeout-s",
        str(args.idle_timeout_s),
    ]
    if args.measure_energy:
        cmd.append("--measure-energy")
    if args.ref_so:
        cmd.extend(["--ref-so", str(args.ref_so)])
    if args.wait_idle:
        cmd.append("--wait-idle")
    return cmd


def detect_group_conv_split_size(primfunc_text: str) -> int | None:
    """Return channels-per-group for 32-group fp16 group conv PrimFunc text."""
    for match in _BUFFER4_RE.findall(primfunc_text):
        o, i, kh, kw, dtype = match
        out_channels = int(o)
        in_per_group = int(i)
        if dtype != "float16" or (int(kh), int(kw)) != (3, 3):
            continue
        if in_per_group <= 0:
            continue
        if out_channels == in_per_group * 32:
            return in_per_group
    return None


def _configure_python_imports(tvm_site: Path) -> None:
    tvm_site_text = str(tvm_site)
    if tvm_site_text not in sys.path:
        sys.path.insert(0, tvm_site_text)
    loaded = sys.modules.get("onnx")
    if loaded is not None and not hasattr(loaded, "load"):
        del sys.modules["onnx"]


def _extract_onnx_shape_dict(model: Any) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for item in model.graph.input:
        dims = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(int(dim.dim_value) if dim.dim_value else 1)
        shapes[item.name] = tuple(dims)
    return shapes


def _extract_onnx_input_specs(model: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in model.graph.input:
        shape = []
        for dim in item.type.tensor_type.shape.dim:
            shape.append(int(dim.dim_value) if dim.dim_value else 1)
        specs.append({"name": item.name, "shape": shape, "dtype": "float32"})
    return specs


def _build_relax_fp16_module(
    tvm: Any,
    relax: Any,
    onnx: Any,
    onnx_path: Path,
    target: Any,
    *,
    precision: str = "fp16",
) -> tuple[Any, list[dict[str, Any]]]:
    from tvm.relax.frontend.onnx import from_onnx

    model = onnx.load(str(onnx_path))
    shapes = _extract_onnx_shape_dict(model)
    mod = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
    if precision == "fp16":
        mod = relax.transform.ToMixedPrecision(out_dtype="float16")(mod)
    elif precision != "fp32":
        raise ValueError(f"unsupported precision: {precision}")
    with target, tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential(
            [
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )
        modt = seq(mod)
    return modt, _extract_onnx_input_specs(model)


def _split_ff_loop(tvm: Any, func: Any, group_size: int) -> Any:
    single_mod = tvm.IRModule({"main": func})
    sch = tvm.s_tir.Schedule(single_mod)
    block = sch.get_sblock("group_conv2d_nchw")
    loops = sch.get_loops(block)
    sch.split(loops[1], factors=[None, int(group_size)])
    return sch.mod["main"]


def _apply_auto_group_split(tvm: Any, modt: Any) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    patched: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for gv in list(modt.get_global_vars()):
        name = gv.name_hint
        func = modt[gv]
        try:
            group_size = detect_group_conv_split_size(func.script())
        except Exception as exc:  # pragma: no cover - defensive for TVM script printer
            skipped.append({"name": name, "reason": f"script_failed:{exc!r}"})
            continue
        if group_size is None:
            continue
        try:
            new_func = _split_ff_loop(tvm, func, group_size)
            modt.update_func(gv, new_func)
            patched.append({"name": name, "group_size": int(group_size)})
        except Exception as exc:
            skipped.append({"name": name, "group_size": int(group_size), "reason": repr(exc)})
    return modt, patched, skipped


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    env = build_h800_env(gpu=str(args.gpu), tvm_site=Path(args.tvm_site))
    os.environ.update(env)
    _configure_python_imports(Path(args.tvm_site))

    import onnx
    import tvm
    import tvm.tirx  # noqa: F401
    import tvm.s_tir.tensor_intrin.cuda  # noqa: F401
    from tvm import relax
    from tvm.s_tir.meta_schedule import relax_integration as ri

    paths = plan_paths(args)
    paths["label_dir"].mkdir(parents=True, exist_ok=True)
    started = time.time()
    result: dict[str, Any] = {
        "schema": "route_b_fp16_auto_build_v1",
        "status": "started",
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "label": args.label,
        "width": list(parse_width(args.width)),
        "gpu": str(args.gpu),
        "precision": precision_of(args),
        "onnx_path": str(args.onnx),
        "method": (
            "Relax ToMixedPrecision fp16 + auto group-conv split-fix + whole-graph MetaSchedule"
            if precision_of(args) == "fp16"
            else "Relax fp32 + whole-graph MetaSchedule"
        ),
        "not_used": ["historical hand-written im2col+MMA as final route", "fixed 64x128x256-only split constants"],
    }
    try:
        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        modt, input_specs = _build_relax_fp16_module(
            tvm,
            relax,
            onnx,
            Path(args.onnx),
            target,
            precision=precision_of(args),
        )
        n_funcs_before = len(list(modt.get_global_vars()))
        patched: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        if args.fix == "split":
            modt, patched, skipped = _apply_auto_group_split(tvm, modt)
            if not patched:
                raise RuntimeError("auto split-fix found no 32-group conv PrimFuncs")

        if int(args.max_trials) == 0:
            tune_s = 0.0
            scheduled = modt
            build_start = time.time()
            with target, tvm.transform.PassContext(opt_level=3):
                ex = tvm.compile(scheduled, target=target)
        else:
            tune_start = time.time()
            ri.tune_relax(
                mod=modt,
                params={},
                target=target,
                work_dir=str(paths["work_dir"]),
                max_trials_global=int(args.max_trials),
                seed=int(args.seed),
            )
            tune_s = time.time() - tune_start
            build_start = time.time()
            with target, tvm.transform.PassContext(opt_level=3):
                scheduled = relax.transform.MetaScheduleApplyDatabase(str(paths["work_dir"]))(modt)
                ex = tvm.compile(scheduled, target=target)
        ex.export_library(str(paths["artifact_so"]))
        build_s = time.time() - build_start

        ref_so = Path(args.ref_so) if args.ref_so else paths["ref_so"]
        if not args.ref_so:
            ref_mod, _ = _build_relax_fp16_module(
                tvm,
                relax,
                onnx,
                Path(args.onnx),
                target,
                precision=precision_of(args),
            )
            with target, tvm.transform.PassContext(opt_level=3):
                ref_ex = tvm.compile(ref_mod, target=target)
            ref_ex.export_library(str(ref_so))

        write_json(paths["input_specs"], input_specs)
        result.update(
            {
                "status": "built",
                "build_success": True,
                "n_global_funcs_before": n_funcs_before,
                "patched_group_conv_funcs": patched,
                "split_skipped": skipped,
                "max_trials": int(args.max_trials),
                "tuning_policy": "default_compile_no_metaschedule" if int(args.max_trials) == 0 else "metaschedule",
                "work_dir": str(paths["work_dir"]),
                "artifact_path": str(paths["artifact_so"]),
                "artifact_digest": sha256(paths["artifact_so"]),
                "ref_so": str(ref_so),
                "ref_digest": sha256(ref_so),
                "input_specs_path": str(paths["input_specs"]),
                "tune_s": tune_s,
                "compile_s": build_s,
            }
        )
    except Exception as exc:
        result.update({"status": "failed", "build_success": False, "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        result["elapsed_s"] = round(time.time() - started, 6)
        write_json(paths["build_json"], result)
    return result


def _load_vm(tvm: Any, relax: Any, so_path: Path, dev: Any) -> Any:
    ex = tvm.runtime.load_module(str(so_path))
    return relax.VirtualMachine(ex, dev)


def _unpack_outputs(obj: Any) -> list[Any]:
    if isinstance(obj, (list, tuple)):
        return list(obj)
    try:
        return [obj[i] for i in range(len(obj))]
    except TypeError:
        return [obj]


def _make_runtime_args(tvm: Any, dev: Any, input_specs: list[dict[str, Any]], seed: int) -> list[Any]:
    import numpy as np

    np.random.seed(int(seed))
    args: list[Any] = []
    for spec in input_specs:
        dtype = str(spec.get("dtype", "float32"))
        shape = [int(item) for item in spec["shape"]]
        data = (np.random.randn(*shape) * 0.5).astype(dtype)
        args.append(tvm.runtime.tensor(data, device=dev))
    return args


def _to_numpy(value: Any) -> Any:
    return value.numpy()


def _compare_outputs(ref_outputs: list[Any], cand_outputs: list[Any]) -> list[dict[str, Any]]:
    import numpy as np

    reports: list[dict[str, Any]] = []
    for idx, (ref, cand) in enumerate(zip(ref_outputs, cand_outputs)):
        r = _to_numpy(ref).astype("float32")
        c = _to_numpy(cand).astype("float32")
        if r.shape != c.shape:
            reports.append({"output": idx, "shape_match": False, "ref_shape": list(r.shape), "cand_shape": list(c.shape)})
            continue
        abs_err = np.abs(r - c)
        rel = abs_err / np.maximum(np.abs(r), 1e-2)
        reports.append(
            {
                "output": idx,
                "shape_match": True,
                "shape": list(r.shape),
                "mean_rel_err": float(rel.mean()),
                "p99_rel_err": float(np.percentile(rel, 99)),
                "max_abs_err": float(abs_err.max()),
            }
        )
    return reports


class VmCallable:
    def __init__(self, vm: Any) -> None:
        self._vm = vm

    def __call__(self, *args: Any) -> Any:
        return self._vm["main"](*args)


def _count_power_samples(path: Path) -> int:
    if not path.is_file():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(len(lines) - 1, 0)


def _load_energy_measurement_impl() -> Any:
    return load_module(CAPABILITY_PATH, "route_b_fp16_energy_capability_probe").measure_energy


def measure_compiled_vm_energy(
    *,
    label_dir: Path,
    vm: Any,
    dev: Any,
    runtime_args: list[Any],
    gpu: str,
    measure_iters: int,
    energy_impl: Any | None = None,
) -> dict[str, Any]:
    active_csv = label_dir / "active_power_samples.csv"
    idle_csv = label_dir / "idle_power_samples.csv"
    impl = energy_impl or _load_energy_measurement_impl()
    try:
        raw = impl(
            label_dir,
            VmCallable(vm),
            dev,
            runtime_args,
            gpu=str(gpu),
            measure_iters=int(measure_iters),
        )
        requested_iters = int(raw.get("requested_measure_iters") or measure_iters)
        completed_iters = int(raw.get("measure_iters") or raw.get("completed_measure_iters") or 0)
        joules = raw.get("joules_per_inference")
        if joules is None:
            joules = raw.get("joule_per_inference")
        if joules is None:
            joules = raw.get("energy_J")
        return {
            "status": "success",
            "joules_per_inference": joules,
            "joule_per_inference": joules,
            "requested_measure_iters": requested_iters,
            "completed_measure_iters": completed_iters,
            "warmup_iters": raw.get("warmup_iters"),
            "min_active_s": raw.get("min_active_s"),
            "sample_window_ms": raw.get("sample_window_ms"),
            "idle_watt_avg": raw.get("idle_watt_avg"),
            "watt_avg": raw.get("watt_avg"),
            "watt_delta_avg": raw.get("watt_delta_avg"),
            "watt_p50": raw.get("watt_p50"),
            "watt_p90": raw.get("watt_p90"),
            "idle_sample_count": _count_power_samples(idle_csv),
            "active_sample_count": _count_power_samples(active_csv),
            "telemetry_source": "nvidia-smi power.draw polling 50ms",
            "power_samples_csv": {
                "idle": str(idle_csv),
                "active": str(active_csv),
            },
            "raw": raw,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": repr(exc),
            "requested_measure_iters": int(measure_iters),
            "completed_measure_iters": 0,
            "joules_per_inference": None,
            "joule_per_inference": None,
            "idle_sample_count": _count_power_samples(idle_csv),
            "active_sample_count": _count_power_samples(active_csv),
            "telemetry_source": "nvidia-smi power.draw polling 50ms",
            "power_samples_csv": {
                "idle": str(idle_csv),
                "active": str(active_csv),
            },
        }


def run_measure(args: argparse.Namespace) -> dict[str, Any]:
    env = build_h800_env(gpu=str(args.gpu), tvm_site=Path(args.tvm_site))
    os.environ.update(env)
    _configure_python_imports(Path(args.tvm_site))

    import numpy as np
    import tvm
    import tvm.tirx  # noqa: F401
    from tvm import relax

    paths = plan_paths(args)
    ref_so = Path(args.ref_so) if args.ref_so else paths["ref_so"]
    started = time.time()
    result: dict[str, Any] = {
        "schema": "route_b_fp16_auto_result_v1",
        "status": "started",
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "label": args.label,
        "width": list(parse_width(args.width)),
        "gpu": str(args.gpu),
        "precision": precision_of(args),
        "onnx_path": str(args.onnx),
        "artifact_path": str(paths["artifact_so"]),
        "ref_so": str(ref_so),
        "method": f"Route B {precision_of(args)} automatic measurement",
    }
    try:
        if not paths["artifact_so"].is_file():
            raise FileNotFoundError(paths["artifact_so"])
        if not ref_so.is_file():
            raise FileNotFoundError(ref_so)
        input_specs = read_json(paths["input_specs"])
        dev = tvm.cuda(0)
        vm = _load_vm(tvm, relax, paths["artifact_so"], dev)
        ref_vm = _load_vm(tvm, relax, ref_so, dev)
        runtime_args = _make_runtime_args(tvm, dev, input_specs, seed=int(args.seed))

        cand = _unpack_outputs(vm["main"](*runtime_args))
        dev.sync()
        ref = _unpack_outputs(ref_vm["main"](*runtime_args))
        dev.sync()
        correctness = _compare_outputs(ref, cand)

        for _ in range(int(args.warmup)):
            vm["main"](*runtime_args)
        dev.sync()
        evaluator = vm.time_evaluator("main", dev, number=1, repeat=int(args.iters) * int(args.repeat))
        times_ms = np.array(evaluator(*runtime_args).results) * 1e3
        result.update(
            {
                "status": "success",
                "build_success": True,
                f"correctness_vs_default_{precision_of(args)}": correctness,
                "latency": {
                    "latency_ms_p50": float(np.median(times_ms)),
                    "latency_ms_mean": float(np.mean(times_ms)),
                    "latency_ms_p99": float(np.percentile(times_ms, 99)),
                    "latency_ms_min": float(np.min(times_ms)),
                    "latency_ms_max": float(np.max(times_ms)),
                    "repeat_count": int(len(times_ms)),
                },
                "artifact_digest": sha256(paths["artifact_so"]),
                "ref_digest": sha256(ref_so),
            }
        )
        if args.measure_energy:
            energy = measure_compiled_vm_energy(
                label_dir=paths["label_dir"],
                vm=vm,
                dev=dev,
                runtime_args=runtime_args,
                gpu=str(args.gpu),
                measure_iters=int(args.energy_iters),
            )
            result["energy"] = energy
            if energy.get("status") == "success":
                result["gold_measurement_complete"] = True
            else:
                result["gold_measurement_complete"] = False
                result["gold_measurement_gaps"] = ["energy_failed"]
    except Exception as exc:
        result.update({"status": "failed", "build_success": False, "error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        result["elapsed_s"] = round(time.time() - started, 6)
        write_json(paths["result_json"], result)
        write_json(paths["latest_json"], result)
    return result


def _run_phase_subprocess(args: argparse.Namespace, phase: str) -> dict[str, Any]:
    paths = plan_paths(args)
    paths["label_dir"].mkdir(parents=True, exist_ok=True)
    cmd = self_phase_command(Path(sys.executable), args, phase)
    env = build_h800_env(gpu=str(args.gpu), tvm_site=Path(args.tvm_site))
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO), timeout=None)
    (paths["label_dir"] / f"{phase}_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (paths["label_dir"] / f"{phase}_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return {"phase": phase, "cmd": cmd, "returncode": proc.returncode, "elapsed_s": round(time.time() - started, 6)}


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    paths = plan_paths(args)
    started = time.time()
    build = _run_phase_subprocess(args, "build")
    measure: dict[str, Any] | None = None
    if build["returncode"] == 0:
        measure = _run_phase_subprocess(args, "measure")
    result = {
        "schema": "route_b_fp16_auto_run_wrapper_v1",
        "status": "success" if build["returncode"] == 0 and measure and measure["returncode"] == 0 else "failed",
        "created_at_utc": utc_now(),
        "label": args.label,
        "width": list(parse_width(args.width)),
        "onnx_path": str(args.onnx),
        "precision": precision_of(args),
        "build_phase": build,
        "measure_phase": measure,
        "result_json": str(paths["result_json"]),
        "elapsed_s": round(time.time() - started, 6),
    }
    write_json(paths["label_dir"] / "route_b_fp16_auto_run_wrapper.json", result)
    return result


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    paths = plan_paths(args)
    return {
        "schema": "route_b_fp16_auto_plan_v1",
        "status": "planned",
        "label": args.label,
        "width": list(parse_width(args.width)),
        "onnx_path": str(args.onnx),
        "precision": precision_of(args),
        "paths": {key: str(value) for key, value in paths.items()},
        "build_cmd": self_phase_command(Path(sys.executable), args, "build"),
        "measure_cmd": self_phase_command(Path(sys.executable), args, "measure"),
        "method": "Relax fp16 auto split-fix then whole-graph MetaSchedule",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["plan", "build", "measure", "run"], default="run")
    parser.add_argument("--label", default="smbo_64x128x256")
    parser.add_argument("--width", default="64,128,256")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gpu", default="6")
    parser.add_argument("--tvm-site", type=Path, default=DEFAULT_TVM_SITE)
    parser.add_argument("--max-trials", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fix", choices=["split", "none"], default="split")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--measure-energy", action="store_true")
    parser.add_argument("--energy-iters", type=int, default=300)
    parser.add_argument("--ref-so", type=Path)
    parser.add_argument("--wait-idle", action="store_true")
    parser.add_argument("--idle-timeout-s", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "plan":
        result = run_plan(args)
    elif args.phase == "build":
        result = run_build(args)
    elif args.phase == "measure":
        result = run_measure(args)
    else:
        result = run_all(args)
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"success", "built", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
