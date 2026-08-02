#!/usr/bin/env python3
"""Run or import a Stage2 true-FP32 AP eval for one original60 label."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    ap_anchor_row,
    append_jsonl,
    parse_width_csv,
    stable_config_id,
    utc_timestamp,
    validate_lut_row,
)


DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_ROWS_OUT = (
    ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/"
    "rows/fp32_true_original60_ap_rows_v1.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--width", required=True, help="Comma-separated stage widths, e.g. 24,128,256")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--rows-out", default=str(DEFAULT_ROWS_OUT))
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--fusion-method", default="intermediate")
    parser.add_argument("--range", dest="eval_range", default="102.4,102.4")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--dataset", default="DAIR-V2X")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--emit-row-only", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Run/load the AP report but do not append an AP row. Use for smoke tests.",
    )
    return parser.parse_args()


def resolve_cli_paths(args: argparse.Namespace, launch_cwd: Path) -> None:
    for attr in ("ckpt_dir", "checkpoint_path", "raw_dir", "rows_out", "report_json", "heal_root"):
        value = getattr(args, attr, None)
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            setattr(args, attr, str((launch_cwd / path).resolve()))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def best_checkpoint(ckpt_dir: Path) -> Path:
    bestvals: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("net_epoch_bestval_at*.pth"):
        stem = path.stem
        try:
            epoch = int(stem.split("bestval_at", 1)[1])
        except (IndexError, ValueError):
            continue
        if epoch > 23:
            bestvals.append((epoch, path))
    if bestvals:
        bestvals.sort(key=lambda item: item[0])
        return bestvals[-1][1]

    epochs: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("net_epoch*.pth"):
        if "bestval" in path.name:
            continue
        stem = path.stem
        try:
            epoch = int(stem.split("net_epoch", 1)[1])
        except (IndexError, ValueError):
            continue
        epochs.append((epoch, path))
    if epochs:
        epochs.sort(key=lambda item: item[0])
        return epochs[-1][1]
    raise FileNotFoundError(f"no checkpoint found in {ckpt_dir}")


def selected_checkpoint(args: argparse.Namespace) -> Path:
    ckpt_dir = Path(args.ckpt_dir).resolve()
    explicit_value = getattr(args, "checkpoint_path", None)
    if not explicit_value:
        return best_checkpoint(ckpt_dir)
    explicit = Path(str(explicit_value)).resolve()
    if explicit.parent != ckpt_dir:
        raise ValueError("explicit checkpoint must belong to the checkpoint directory")
    if not explicit.is_file():
        raise FileNotFoundError(f"missing explicit checkpoint: {explicit}")
    return explicit


def validate_execute_inputs(args: argparse.Namespace) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"missing checkpoint directory: {ckpt_dir}")
    config_path = ckpt_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing checkpoint config: {config_path}")
    selected_checkpoint(args)
    heal_root = Path(args.heal_root)
    if not heal_root.exists():
        raise FileNotFoundError(f"missing HEAL root: {heal_root}")


def checkpoint_epoch(ckpt_path: Path) -> int:
    name = ckpt_path.name
    for pattern in (r"bestval_at(\d+)", r"net_epoch(\d+)"):
        import re

        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    raise ValueError(f"unable to parse checkpoint epoch from {ckpt_path}")


def command_env(gpu_id: int, heal_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(heal_root) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def record_preflight(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate",
            "--format=csv",
        ],
        ["nvidia-smi", "pmon", "-c", "1"],
    ]
    records: list[dict[str, Any]] = []
    for command in commands:
        proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        records.append(
            {
                "command": command,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
    write_json(raw_dir / "gpu_preflight.json", records)


def cast_floating_tensors(value: Any, dtype: Any) -> Any:
    import torch

    if isinstance(value, dict):
        return {key: cast_floating_tensors(item, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_floating_tensors(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_floating_tensors(item, dtype) for item in value)
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=dtype)
    return value


def dtype_counts(model: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in model.parameters():
        counts[str(parameter.dtype)] += int(parameter.numel())
    for buffer in model.buffers():
        counts[str(buffer.dtype)] += int(buffer.numel())
    return dict(sorted(counts.items()))


def inference_intermediate_fusion_fp32_postprocess(
    batch_data: dict[str, Any],
    model: Any,
    dataset: Any,
    model_context: Any = None,
) -> dict[str, Any]:
    import torch

    output_dict: OrderedDict[str, Any] = OrderedDict()
    context = model_context if model_context is not None else nullcontext()
    with context:
        output_dict["ego"] = model(batch_data["ego"])
    output_dict["ego"] = cast_floating_tensors(output_dict["ego"], torch.float32)
    pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(batch_data, output_dict)
    return_dict = {
        "pred_box_tensor": pred_box_tensor,
        "pred_score": pred_score,
        "gt_box_tensor": gt_box_tensor,
        "_path": "pytorch_model_output_fp32_postprocess",
    }
    if "depth_items" in output_dict["ego"]:
        return_dict.update({"depth_items": output_dict["ego"]["depth_items"]})
    return return_dict


def run_true_fp32_eval(args: argparse.Namespace, report_path: Path) -> dict[str, Any]:
    import importlib

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    heal_root = Path(args.heal_root)
    sys.path.insert(0, str(heal_root))
    os.chdir(heal_root)

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils
    from opencood.utils import eval_utils
    from opencood.utils.common_utils import update_dict

    ckpt_dir = Path(args.ckpt_dir)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    opt = argparse.Namespace(
        model_dir=str(ckpt_dir),
        fusion_method=args.fusion_method,
        save_vis_interval=10**9,
        save_npy=False,
        range=args.eval_range,
        no_score=True,
        note="stage2_true_fp32",
    )
    hypes = yaml_utils.load_yaml(None, opt)
    if "heter" in hypes:
        x_min, x_max = -eval(opt.range.split(",")[0]), eval(opt.range.split(",")[0])
        y_min, y_max = -eval(opt.range.split(",")[1]), eval(opt.range.split(",")[1])
        new_cav_range = [
            x_min,
            y_min,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][2],
            x_max,
            y_max,
            hypes["postprocess"]["anchor_args"]["cav_lidar_range"][5],
        ]
        hypes = update_dict(
            hypes,
            {
                "cav_lidar_range": new_cav_range,
                "lidar_range": new_cav_range,
                "gt_range": new_cav_range,
            },
        )
        yaml_utils_lib = importlib.import_module("opencood.hypes_yaml.yaml_utils")
        parser_func = getattr(yaml_utils_lib, hypes["yaml_parser"])
        hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    if "box_align" in hypes.keys():
        hypes["box_align"]["val_result"] = hypes["box_align"]["test_result"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(303)
    model = train_utils.create_model(hypes)
    ckpt_path = selected_checkpoint(args)
    resume_epoch = checkpoint_epoch(ckpt_path)
    print(f"resuming selected checkpoint at epoch {resume_epoch}: {ckpt_path}")
    loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
    train_utils.check_missing_key(model.state_dict(), loaded_state_dict)
    model.load_state_dict(loaded_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    before_counts = dtype_counts(model)

    dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
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
    t0 = time.time()
    processed = 0
    path_counts: Counter[str] = Counter()
    for i, batch_data in enumerate(data_loader):
        if args.num_samples is not None and processed >= int(args.num_samples):
            break
        if batch_data is None:
            continue
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            infer_result = inference_intermediate_fusion_fp32_postprocess(
                batch_data,
                model,
                dataset,
            )
            pred_box_tensor = infer_result["pred_box_tensor"]
            gt_box_tensor = infer_result["gt_box_tensor"]
            pred_score = infer_result["pred_score"]
            for threshold in (0.3, 0.5, 0.7):
                eval_utils.caluclate_tp_fp(
                    pred_box_tensor,
                    pred_score,
                    gt_box_tensor,
                    result_stat,
                    threshold,
                )
            path_counts[str(infer_result.get("_path") or "pytorch")] += 1
            processed += 1
        torch.cuda.empty_cache()

    ap30, ap50, ap70 = eval_utils.eval_final_results(
        result_stat,
        str(raw_dir),
        f"{args.label}_model_float32",
    )
    report = {
        "status": "success",
        "label": args.label,
        "precision": "fp32",
        "precision_mode": "model_float32",
        "dataset": args.dataset,
        "eval_split": args.eval_split,
        "num_samples": processed,
        "ap30": float(ap30),
        "ap50": float(ap50),
        "ap70": float(ap70),
        "elapsed_secs": time.time() - t0,
        "path_counts": dict(sorted(path_counts.items())),
        "resume_epoch": int(resume_epoch),
        "ckpt_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "config_path": str(ckpt_dir / "config.yaml"),
        "eval_command": " ".join(sys.argv),
        "raw_artifact": str(raw_dir),
    }
    write_json(report_path, report)
    write_json(
        raw_dir / "layer_precision_summary.json",
        {
            "schema": "stage2_true_fp32_ap_precision_summary_v1",
            "precision_mode": "model_float32",
            "model_dtype_counts": before_counts,
            "runner": "model.float32 with floating batch tensors left in FP32/default dtype",
            "postprocess_dtype_policy": "model outputs cast to float32 before dataset.post_process",
            "full_network_claim": False,
        },
    )
    return report


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(args: argparse.Namespace, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    label = str(args.label)
    width = parse_width_csv(args.width)
    ckpt_dir = Path(args.ckpt_dir)
    raw_dir = Path(args.raw_dir)
    ckpt_path = Path(str(report.get("ckpt_path") or selected_checkpoint(args)))
    config_path = Path(str(report.get("config_path") or ckpt_dir / "config.yaml"))
    layer_summary = raw_dir / "layer_precision_summary.json"
    if not layer_summary.exists():
        write_json(
            layer_summary,
            {
                "schema": "stage2_true_fp32_ap_precision_summary_v1",
                "precision_mode": report.get("precision_mode", "model_float32"),
                "runner": "true_fp32 report import",
                "full_network_claim": False,
            },
        )
    config_id = stable_config_id(
        model="pyramid_lidar",
        candidate_id=f"original60:{label}:fp32:ap",
        software_point_id=f"true_fp32_original60:{label}:{'x'.join(str(item) for item in width)}:fp32:ap",
        quant_policy="fp32",
        schedule_policy="not_applicable",
    )
    source_files = [
        str(report_path),
        str(layer_summary),
    ]
    if ckpt_path.exists():
        source_files.append(str(ckpt_path))
    if config_path.exists():
        source_files.append(str(config_path))
    row = ap_anchor_row(
        config_id=config_id,
        model="pyramid_lidar",
        manifest_digest=sha256_file(ckpt_path) if ckpt_path.exists() else "unknown",
        candidate_id=f"original60:pyramid_lidar:{label}:fp32",
        software_point_id=f"true_fp32_original60:{label}:{'x'.join(str(item) for item in width)}:fp32:ap",
        dense_stage="model",
        optimized_scope="full_model_ap_eval",
        width=width,
        quant_policy="fp32",
        schedule_policy="not_applicable",
        backend="model_eval",
        measurement_status="measured",
        metric="AP70",
        metric_value=float(report["ap70"]),
        secondary_metrics={"AP30": float(report["ap30"]), "AP50": float(report["ap50"])},
        dataset=str(report.get("dataset") or args.dataset),
        eval_split=str(report.get("eval_split") or args.eval_split),
        num_samples=int(report.get("num_samples") or 0),
        ckpt_path=str(ckpt_path),
        ckpt_digest=sha256_file(ckpt_path) if ckpt_path.exists() else "unknown",
        config_path=str(config_path),
        config_digest=sha256_file(config_path) if config_path.exists() else "unknown",
        finetune_protocol="structural_prune_pyramid_train_ddp_fp32_model_eval",
        training_budget="existing_stage2_original60_s0_024_or_label_checkpoint",
        eval_command=str(report.get("eval_command") or "stage2_h800_true_fp32_ap_eval.py"),
        provenance="stage2_original60_true_fp32_h800_model_eval",
        run_id=args.run_id or f"true_fp32_original60_ap_{label}",
        created_at=args.created_at or utc_timestamp(),
        source_files=source_files,
        raw_artifact=str(raw_dir),
        notes="true_fp32 model_eval AP row from measured report",
        precision="fp32",
        quant_scheme="none",
        quant_method="h800_tvm_true_fp32_model_eval",
        quant_scope="full_model_ap_eval_true_fp32",
        calibration_source="none",
        calibration_digest="none",
        calibrator="none",
        calibration_inputs=[],
        fallback_policy="none",
        layer_precision_summary=str(layer_summary),
        full_network_claim=False,
        engine_kind="model_eval",
        engine_digest=sha256_file(report_path),
        measurement_source="true_eval",
        claim_status="claimable_true_eval",
        quality_gate_status="true_fp32_original60_ap_eval",
        schedule_profile="not_applicable",
        tune_budget="not_applicable",
        label=label,
        original60_candidate_id=f"original60:{label}",
    )
    validate_lut_row(row)
    return row


def classify_failure(exc: Exception) -> str:
    reason = f"{type(exc).__name__}:{exc}".lower()
    if isinstance(exc, FileNotFoundError) or "missing checkpoint" in reason or "no checkpoint" in reason:
        return "configuration_error"
    if "refusing to append measured ap row for partial eval" in reason:
        return "partial_eval_not_importable"
    if "cuda" in reason or "nvidia-smi" in reason:
        return "environment_error"
    return "runtime_error"


def write_blocker(
    raw_dir: Path,
    args: argparse.Namespace,
    reason: str,
    traceback_text: str = "",
    failure_type: str = "runtime_error",
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = raw_dir / "stdout.txt"
    stderr_path = raw_dir / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text(reason + "\n" + traceback_text, encoding="utf-8")
    write_json(
        raw_dir / "ap_eval_blocker.json",
        {
            "schema": "stage2_true_fp32_ap_eval_blocker_v1",
            "label": args.label,
            "precision": "fp32",
            "failure_type": failure_type,
            "failure_reason": reason,
            "traceback": traceback_text,
            "gpu_id": args.gpu_id,
            "raw_artifact": str(raw_dir),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "command": sys.argv,
            "created_at": args.created_at or utc_timestamp(),
        },
    )


def main() -> int:
    args = parse_args()
    resolve_cli_paths(args, Path.cwd())
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_json) if args.report_json else raw_dir / "ap_eval_report.json"
    write_json(
        raw_dir / "runner_command.json",
        {
            "command": sys.argv,
            "cwd": os.getcwd(),
            "gpu_id": args.gpu_id,
            "precision_mode": "model_float32",
        },
    )
    try:
        if args.execute:
            validate_execute_inputs(args)
            record_preflight(raw_dir)
            report = run_true_fp32_eval(args, report_path)
        else:
            report = load_report(report_path)
        if args.report_only:
            print(
                json.dumps(
                    {
                        "status": "ok_report_only",
                        "report_json": str(report_path),
                        "num_samples": int(report.get("num_samples") or 0),
                        "ap70": report.get("ap70"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if not args.emit_row_only and int(report.get("num_samples") or 0) < 1789:
            raise RuntimeError(
                f"refusing to append measured AP row for partial eval: num_samples={report.get('num_samples')}"
            )
        row = build_row(args, report, report_path)
        append_jsonl(args.rows_out, row)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "row_id": row["row_id"],
                    "rows_out": str(args.rows_out),
                    "report_json": str(report_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        reason = f"{type(exc).__name__}:{exc}"
        write_blocker(raw_dir, args, reason, traceback.format_exc(), classify_failure(exc))
        print(json.dumps({"status": "failed", "failure_reason": reason}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
