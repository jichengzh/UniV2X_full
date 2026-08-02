#!/usr/bin/env python3
"""Hybrid AP evaluation with TVM RouteB replacing CoDriving backbone.resnet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage2_codriving_int8_provenance import (
    EXPECTED_CALIBRATION_SAMPLES,
    INT8_QUANTIZATION_SEMANTICS,
    report_has_valid_int8_calibration,
    validate_calibration_tensor_shape,
    validate_int8_calibration_manifest,
)


REMOTE_REPO = Path("/exdata/jichengzhi/V2Xverse_pyramid")
PIPELINE_SCOPE = "tvm_routeb_resnet_in_full_pytorch_eval"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def routeb_mode_spec(mode: str) -> dict[str, str]:
    specs = {
        "fp16": {"precision": "fp16", "mixed_policy": "none", "tag": "tvm_routeb_fp16"},
        "int8_all": {"precision": "int8", "mixed_policy": "all", "tag": "tvm_routeb_int8_all"},
        "mixed_top25_flops": {
            "precision": "mixed",
            "mixed_policy": "top25_flops",
            "tag": "tvm_routeb_int8_top25",
        },
        "mixed_top50_flops": {
            "precision": "mixed",
            "mixed_policy": "top50_flops",
            "tag": "tvm_routeb_int8_top50",
        },
    }
    if mode not in specs:
        raise ValueError(f"unsupported RouteB AP mode: {mode}")
    return dict(specs[mode])


def build_report(
    *,
    width: str,
    mode: str,
    model_dir: Path,
    onnx: Path,
    onnx_sha256: str,
    ap30: float,
    ap50: float,
    ap70: float,
    n_done: int,
    n_tvm_path: int,
    n_fallback_path: int,
    n_skipped: int,
    elapsed_secs: float,
    compile_summary: dict[str, Any],
) -> dict[str, Any]:
    spec = routeb_mode_spec(mode)
    return {
        "schema": "v2_gold_coldstart_96_codriving_tvm_resnet_ap_eval_v1",
        "created_at_utc": utc_now(),
        "width": width,
        "mode": mode,
        "tag": spec["tag"],
        "precision": spec["precision"],
        "mixed_policy": spec["mixed_policy"],
        "pipeline_scope": PIPELINE_SCOPE,
        "model_dir": str(model_dir),
        "onnx": str(onnx),
        "onnx_sha256": onnx_sha256,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "n_done": int(n_done),
        "n_tvm_path": int(n_tvm_path),
        "n_fallback_path": int(n_fallback_path),
        "n_skipped": int(n_skipped),
        "elapsed_secs": float(elapsed_secs),
        "compile_summary": compile_summary,
    }


def report_complete(path: Path, min_samples: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    n_done = int(payload.get("n_done") or 0)
    n_tvm_path = int(payload.get("n_tvm_path") or 0)
    n_fallback_path = int(payload.get("n_fallback_path") or 0)
    return (
        payload.get("ap70") is not None
        and payload.get("pipeline_scope") == PIPELINE_SCOPE
        and n_done >= min_samples
        and n_tvm_path == n_done
        and n_fallback_path == 0
        and report_has_valid_int8_calibration(payload)
    )


def conv_signature(input_shape: Any, weight_shape: Any) -> str:
    input_key = "x".join(str(int(value)) for value in input_shape)
    weight_key = "x".join(str(int(value)) for value in weight_shape)
    return f"input={input_key}|weight={weight_key}"


def validate_calibration_summary_sample_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"calibration summary collected_samples must be an integer, got {value!r}")
    if value != EXPECTED_CALIBRATION_SAMPLES:
        raise ValueError(
            f"INT8 AP calibration summary must contain {EXPECTED_CALIBRATION_SAMPLES} samples"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_int8_scale_manifest(
    layer_records: list[dict[str, Any]],
    weight_absmax_by_shape: dict[tuple[int, ...], float],
    *,
    calibration_source: Path,
    calibration_summary: Path,
    calibration_split: str,
    calibration_split_source: Path,
    calibration_source_sha256: str,
    calibration_summary_sha256: str,
    calibration_samples: int,
) -> dict[str, Any]:
    if calibration_samples <= 0:
        raise ValueError("calibration_samples must be positive")
    input_absmax_by_signature: dict[str, float] = {}
    weight_shape_by_signature: dict[str, tuple[int, ...]] = {}
    for record in layer_records:
        input_shape = tuple(int(value) for value in record["input_shape"])
        weight_shape = tuple(int(value) for value in record["weight_shape"])
        input_absmax = float(record["input_absmax"])
        if input_absmax < 0.0:
            raise ValueError(f"input_absmax must be nonnegative, got {input_absmax}")
        key = conv_signature(input_shape, weight_shape)
        input_absmax_by_signature[key] = max(input_absmax_by_signature.get(key, 0.0), input_absmax)
        weight_shape_by_signature[key] = weight_shape

    scales_by_signature: dict[str, dict[str, Any]] = {}
    for key, input_absmax in sorted(input_absmax_by_signature.items()):
        weight_shape = weight_shape_by_signature[key]
        if weight_shape not in weight_absmax_by_shape:
            raise ValueError(f"missing ONNX weight calibration for shape {weight_shape}")
        weight_absmax = float(weight_absmax_by_shape[weight_shape])
        if weight_absmax <= 0.0:
            raise ValueError(f"weight_absmax must be positive for shape {weight_shape}")
        scales_by_signature[key] = {
            "input_absmax": input_absmax,
            "input_scale": max(input_absmax / 127.0, 1e-12),
            "weight_absmax": weight_absmax,
            "weight_scale": weight_absmax / 127.0,
        }
    if not scales_by_signature:
        raise ValueError("INT8 calibration produced no convolution signatures")
    return {
        "schema": "codriving_routeb_int8_scale_manifest_v1",
        "quantization_semantics": INT8_QUANTIZATION_SEMANTICS,
        "calibration_source": str(calibration_source),
        "calibration_summary": str(calibration_summary),
        "calibration_split": calibration_split,
        "calibration_split_source": str(calibration_split_source),
        "calibration_source_sha256": calibration_source_sha256,
        "calibration_summary_sha256": calibration_summary_sha256,
        "calibration_samples": int(calibration_samples),
        "spatial_features_shape": [
            EXPECTED_CALIBRATION_SAMPLES,
            2,
            64,
            256,
            512,
        ],
        "qmin": -127,
        "qmax": 127,
        "scales_by_signature": scales_by_signature,
    }


def collect_int8_scale_manifest(
    model: Any,
    onnx_path: Path,
    calibration_source: Path,
    calibration_summary: Path,
) -> dict[str, Any]:
    import numpy as np
    import onnx
    import torch
    from onnx import numpy_helper

    if not calibration_source.is_file():
        raise FileNotFoundError(calibration_source)
    if not calibration_summary.is_file():
        raise FileNotFoundError(calibration_summary)
    summary = json.loads(calibration_summary.read_text(encoding="utf-8"))
    if summary.get("schema") != "v2_gold_coldstart_96_codriving_calib_export_v1":
        raise ValueError(f"unexpected calibration summary schema: {summary.get('schema')!r}")
    if summary.get("calibration_split") != "train":
        raise ValueError("INT8 AP calibration summary must use train split")
    validate_calibration_summary_sample_count(summary.get("collected_samples"))
    summary_output = Path(str(summary.get("output") or ""))
    if summary_output.resolve() != calibration_source.resolve():
        raise ValueError("calibration summary output does not match --calib-npz")
    summary_shape = validate_calibration_tensor_shape(
        (summary.get("shapes") or {}).get("spatial_features") or ()
    )
    calibration_source_sha256 = sha256_file(calibration_source)
    if summary.get("output_sha256") != calibration_source_sha256:
        raise ValueError("calibration NPZ SHA256 does not match calibration summary")
    calibration_summary_sha256 = sha256_file(calibration_summary)
    calibration_split_source_text = str(summary.get("split_source_file") or "")
    if not calibration_split_source_text:
        raise ValueError("calibration summary lacks split_source_file")
    calibration_split_source = Path(calibration_split_source_text)
    records_by_module: dict[str, dict[str, Any]] = {}
    handles = []
    for module_name, module in model.backbone.resnet.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        records_by_module[module_name] = {
            "input_shape": None,
            "weight_shape": list(module.weight.shape),
            "input_absmax": 0.0,
        }

        def pre_hook(_module: Any, inputs: tuple[Any, ...], *, name: str = module_name) -> None:
            tensor = inputs[0].detach()
            previous = float(records_by_module[name]["input_absmax"])
            records_by_module[name] = {
                **records_by_module[name],
                "input_shape": list(tensor.shape),
                "input_absmax": max(previous, float(tensor.abs().max().item())),
            }

        handles.append(module.register_forward_pre_hook(pre_hook))

    try:
        with np.load(calibration_source, allow_pickle=False) as calibration:
            if "spatial_features" not in calibration:
                raise ValueError(f"calibration file lacks spatial_features: {calibration_source}")
            spatial_features = calibration["spatial_features"]
            actual_shape = validate_calibration_tensor_shape(spatial_features.shape)
            if actual_shape != summary_shape:
                raise ValueError("calibration NPZ shape does not match calibration summary")
            device = next(model.parameters()).device
            with torch.inference_mode():
                for sample in spatial_features:
                    tensor = torch.from_numpy(np.asarray(sample, dtype=np.float32)).to(device=device)
                    model.backbone.resnet(tensor)
            calibration_samples = int(spatial_features.shape[0])
    finally:
        for handle in handles:
            handle.remove()

    layer_records = [record for record in records_by_module.values() if record["input_shape"] is not None]
    onnx_model = onnx.load(str(onnx_path))
    weight_absmax_by_shape: dict[tuple[int, ...], float] = {}
    for initializer in onnx_model.graph.initializer:
        if len(initializer.dims) != 4:
            continue
        value = numpy_helper.to_array(initializer)
        shape = tuple(int(dim) for dim in value.shape)
        absmax = float(np.max(np.abs(value)))
        weight_absmax_by_shape[shape] = max(weight_absmax_by_shape.get(shape, 0.0), absmax)
    return build_int8_scale_manifest(
        layer_records,
        weight_absmax_by_shape,
        calibration_source=calibration_source,
        calibration_summary=calibration_summary,
        calibration_split="train",
        calibration_split_source=calibration_split_source,
        calibration_source_sha256=calibration_source_sha256,
        calibration_summary_sha256=calibration_summary_sha256,
        calibration_samples=calibration_samples,
    )


def export_resnet_onnx(model: Any, output: Path) -> None:
    import torch

    def module_device(module: torch.nn.Module) -> torch.device:
        for tensor in list(module.parameters()) + list(module.buffers()):
            return tensor.device
        return torch.device("cpu")

    class ResnetOnly(torch.nn.Module):
        def __init__(self, backbone: torch.nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, spatial_features: torch.Tensor) -> Any:
            return self.backbone.resnet(spatial_features)

    wrapper = ResnetOnly(model.backbone).eval()
    dummy = torch.zeros(2, 64, 256, 512, dtype=torch.float32, device=module_device(wrapper))
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output),
            opset_version=17,
            input_names=["spatial_features"],
            output_names=["res0", "res1", "res2"],
            do_constant_folding=True,
        )


class TvmRouteBResnetRuntime:
    def __init__(
        self,
        onnx: Path,
        *,
        mode: str,
        gpu: int,
        graph_io_dtype: str = "fp32",
        int8_calibration: dict[str, Any] | None = None,
    ) -> None:
        import tvm
        from tvm import relax
        import tvm.s_tir.dlight as dl
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401
        from tvm.s_tir.dlight.gpu.matmul import MatmulInt8Tensorization, MatmulTensorization

        from stage2_codriving_whole_engine_tc_v1 import (
            _counts,
            _full_engine_rules,
            _select_mixed_int8_convs,
            build_and_legalize,
            classify_conv_funcs,
            make_std_conv_im2col_same_signature_primfunc,
        )

        spec = routeb_mode_spec(mode)
        precision = spec["precision"]
        if precision != "fp16" and (
            int8_calibration is None
            or int8_calibration.get("quantization_semantics") != INT8_QUANTIZATION_SEMANTICS
        ):
            raise ValueError("INT8/mixed RouteB requires a calibrated INT8 scale manifest")
        if int8_calibration is not None:
            validate_int8_calibration_manifest(int8_calibration)
        dev = tvm.cuda(gpu)
        target = tvm.target.Target.from_device(dev)
        legal, input_specs = build_and_legalize(str(onnx), 2, target, graph_io_dtype=graph_io_dtype)
        conv_specs = classify_conv_funcs(legal)
        mixed_int8_names = _select_mixed_int8_convs(conv_specs, spec["mixed_policy"]) if precision == "mixed" else set()
        conv_precision_plan: dict[str, str] = {}
        int8_signature_plan: dict[str, str] = {}
        rewritten = legal
        for name, conv_spec in conv_specs.items():
            gv = rewritten.get_global_var(name)
            accum_dtype = "int32" if precision == "int8" or name in mixed_int8_names else "float16"
            conv_precision_plan[name] = "int8" if accum_dtype == "int32" else "fp16"
            scales = None
            if accum_dtype == "int32":
                key = conv_signature(conv_spec["input_nchw"], conv_spec["weight_oihw"])
                int8_signature_plan[name] = key
                scales = int8_calibration["scales_by_signature"].get(key)
                if scales is None:
                    raise ValueError(f"missing INT8 calibration for {name}: {key}")
            new_func = make_std_conv_im2col_same_signature_primfunc(
                conv_spec,
                name,
                accum_dtype,
                io_dtype=conv_spec["io_dtype"],
                input_scale=None if scales is None else float(scales["input_scale"]),
                weight_scale=None if scales is None else float(scales["weight_scale"]),
            )
            rewritten.update_func(gv, new_func)
        schedule_dtype = "int32" if precision == "int8" else "mixed" if precision == "mixed" else "float16"
        with target, tvm.transform.PassContext(opt_level=3):
            scheduled = dl.ApplyDefaultSchedule(*_full_engine_rules(schedule_dtype))(rewritten)
            ex = tvm.compile(scheduled, target=target)
        self.tvm = tvm
        self.relax = relax
        self.dev = dev
        self.vm = relax.VirtualMachine(ex, dev)
        self.compile_summary = {
            "input_specs": input_specs,
            "n_conv_replaced_tensorized": len(conv_specs),
            "conv_precision_plan": conv_precision_plan,
            "int8_signature_plan": int8_signature_plan,
            "counts_whole_module": _counts("\n".join(f.script() for _, f in scheduled.functions_items())),
        }
        if int8_calibration is not None:
            validate_int8_calibration_manifest(
                int8_calibration,
                required_signatures=set(int8_signature_plan.values()),
            )
            self.compile_summary = {
                **self.compile_summary,
                "quantization_semantics": INT8_QUANTIZATION_SEMANTICS,
                "int8_calibration": int8_calibration,
            }

    def __call__(self, spatial_features: Any) -> tuple[Any, ...]:
        import numpy as np
        import torch

        x = spatial_features.detach().float().contiguous()
        try:
            arg = self.tvm.runtime.from_dlpack(torch.utils.dlpack.to_dlpack(x))
            out = self.vm["main"](arg)
            outs = [out] if hasattr(out, "shape") else list(out)
            return tuple(torch.utils.dlpack.from_dlpack(item).to(device=x.device, dtype=x.dtype) for item in outs)
        except Exception:
            value = x.cpu().numpy().astype("float32", copy=False)
            arg = self.tvm.runtime.tensor(np.asarray(value), device=self.dev)
            out = self.vm["main"](arg)
            outs = [out] if hasattr(out, "shape") else list(out)
            return tuple(torch.from_numpy(item.numpy()).to(device=x.device, dtype=x.dtype) for item in outs)


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    repo_root = Path(args.repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(str(repo_root))

    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.utils import eval_utils

    hypes = load_yaml(str(args.model_dir / "config.yaml"))
    hypes["validate_dir"] = hypes["test_dir"]
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(args.model_dir), model)
    model = model.cuda().eval()
    if args.onnx is None:
        args.onnx = args.model_dir / f"resnet_multiscale_{args.width}_final_fp32.onnx"
    if not args.onnx.is_file():
        export_resnet_onnx(model, args.onnx)
    mode_spec = routeb_mode_spec(args.mode)
    int8_calibration = None
    if mode_spec["precision"] != "fp16":
        if args.calib_npz is None:
            raise ValueError("--calib-npz is required for INT8/mixed RouteB AP evaluation")
        if args.calib_summary is None:
            raise ValueError("--calib-summary is required for INT8/mixed RouteB AP evaluation")
        int8_calibration = collect_int8_scale_manifest(
            model,
            args.onnx,
            args.calib_npz,
            args.calib_summary,
        )
    tvm_runtime = None if args.force_fallback else TvmRouteBResnetRuntime(
        args.onnx,
        mode=args.mode,
        gpu=args.tvm_gpu,
        int8_calibration=int8_calibration,
    )

    class TvmResnetModule(torch.nn.Module):
        def __init__(self, runtime: TvmRouteBResnetRuntime) -> None:
            super().__init__()
            self.runtime = runtime

        def forward(self, spatial_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self.runtime(spatial_features)

    if tvm_runtime is not None:
        model.backbone.resnet = TvmResnetModule(tvm_runtime)

    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    n_done = 0
    n_tvm_path = 0
    n_fallback_path = 0
    n_skipped = 0
    started = time.time()
    with torch.inference_mode():
        for batch_data in loader:
            if args.n_samples > 0 and n_done >= args.n_samples:
                break
            if batch_data is None:
                n_skipped += 1
                continue
            batch_data = train_utils.to_device(batch_data, "cuda")
            ego = batch_data["ego"]
            n_agents = int(ego["record_len"][0].item())
            use_tvm = tvm_runtime is not None and n_agents == 2
            if not use_tvm:
                n_fallback_path += 1
            output_raw = model(ego)
            if use_tvm:
                n_tvm_path += 1
            output_dict = {"ego": output_raw}
            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
            for iou in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, gt_box_tensor, result_stat, iou)
            n_done += 1
            if n_done % args.progress_every == 0:
                print(
                    f"[progress] {args.width} {routeb_mode_spec(args.mode)['tag']} n={n_done} "
                    f"tvm={n_tvm_path} fallback={n_fallback_path} skipped={n_skipped} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    args.eval_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(args.eval_dir), routeb_mode_spec(args.mode)["tag"])
    report = build_report(
        width=args.width,
        mode=args.mode,
        model_dir=args.model_dir,
        onnx=args.onnx,
        onnx_sha256=sha256_file(args.onnx),
        ap30=ap30,
        ap50=ap50,
        ap70=ap70,
        n_done=n_done,
        n_tvm_path=n_tvm_path,
        n_fallback_path=n_fallback_path,
        n_skipped=n_skipped,
        elapsed_secs=time.time() - started,
        compile_summary={} if tvm_runtime is None else tvm_runtime.compile_summary,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REMOTE_REPO)
    parser.add_argument("--width", required=True)
    parser.add_argument("--mode", choices=["fp16", "int8_all", "mixed_top25_flops", "mixed_top50_flops"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--calib-npz", type=Path, default=None)
    parser.add_argument("--calib-summary", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=1789)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--tvm-gpu", type=int, default=0)
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    run_eval(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
