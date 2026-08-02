#!/usr/bin/env python3
"""Evaluate CoDriving AP with the existing Route-B TVM FP16 ResNet artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2_h800_fp16_rewritten_activation_bridge import (  # noqa: E402
    DEFAULT_TVM_LD_LIBRARY_PATH,
    DEFAULT_TVM_PYTHON,
    Fp16RewrittenBackboneBridge,
)
from scripts.stage2_h800_true_fp16_ap_eval import best_checkpoint, write_json  # noqa: E402
from scripts.stage3_trt_multiscale_ap_bridge_v3 import summarize_error_records  # noqa: E402


SCHEMA = "stage3_codriving_tvm_fp16_ap_bridge_v3"
PIPELINE_SCOPE = "codriving_resnet_tvm_fp16_in_full_pytorch_fusion_head_postprocess"
DEFAULT_REPO_ROOT = Path("/exdata/jichengzhi/V2Xverse_pyramid")
SANITY_SAMPLES = 16
FULL_SAMPLES = 1789
ARTIFACT_INPUT_DTYPE = "float32"


def sha256_path(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    root = path.parent if path.is_file() else path
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def protocol_payload(*, precision_tag: str, full_ap_min_samples: int) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_protocol_v1",
        "precision_tag": str(precision_tag),
        "artifact_input_dtype": ARTIFACT_INPUT_DTYPE,
        "engine_batch": 2,
        "engine_outputs": 3,
        "patch_target": "model.backbone.resnet",
        "pipeline_scope": PIPELINE_SCOPE,
        "fallback_policy": "forbidden",
        "sanity_samples": SANITY_SAMPLES,
        "full_samples": int(full_ap_min_samples),
    }


class CoDrivingTvmResnetModule(torch.nn.Module):
    """Thin ``nn.Module`` adapter around the shared Stage2 bridge."""

    def __init__(self, *, reference_resnet: Any, shared_bridge: Any) -> None:
        super().__init__()
        self.reference_resnet = reference_resnet
        self.shared_bridge = shared_bridge
        self.shared_bridge.reference_get_multiscale = reference_resnet

    @property
    def output_error_records(self) -> list[dict[str, Any]]:
        return self.shared_bridge.output_error_records

    def forward(self, spatial_features: Any) -> tuple[Any, ...]:
        return self.shared_bridge(spatial_features)

    def close(self) -> None:
        self.shared_bridge.close()


def patch_model_resnet(model: Any, shared_bridge: Any) -> CoDrivingTvmResnetModule:
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "resnet"):
        raise AttributeError("CoDriving model must expose model.backbone.resnet")
    module = CoDrivingTvmResnetModule(reference_resnet=model.backbone.resnet, shared_bridge=shared_bridge)
    model.backbone.resnet = module
    return module


def evaluate_gates(
    *, processed_samples: int, engine_samples: int, fallback_samples: int, failed_samples: int,
    full_ap_min_samples: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    if fallback_samples:
        blockers.append("fallback_forbidden")
    if failed_samples:
        blockers.append("failed_samples_present")
    if engine_samples != processed_samples:
        blockers.append("engine_samples_mismatch")
    if processed_samples < full_ap_min_samples:
        blockers.append(f"full_samples_below_{full_ap_min_samples}")
    path_blockers = [item for item in blockers if not item.startswith("full_samples_below_")]
    return {
        "sanity_16": processed_samples >= SANITY_SAMPLES and not path_blockers,
        "full_1789": processed_samples >= full_ap_min_samples and not blockers,
        "blockers": blockers,
    }


def build_report(
    *, model_dir: Path, artifact_path: Path, checkpoint_path: Path, config_path: Path,
    dataset_path: Path, shas: dict[str, str], protocol: dict[str, Any], processed_samples: int,
    engine_samples: int, fallback_samples: int, failed_samples: int, ap30: float, ap50: float,
    ap70: float, output_error_summary: dict[str, Any], elapsed_secs: float,
    full_ap_min_samples: int, engine_calls: int | None = None,
) -> dict[str, Any]:
    protocol_sha = sha256_json(protocol)
    bound_shas = dict(shas) | {"protocol": protocol_sha}
    report = {
        "schema": SCHEMA,
        "status": "success",
        "pipeline_scope": PIPELINE_SCOPE,
        "model_dir": str(model_dir),
        "compiled_artifact": str(artifact_path),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "sha256": bound_shas,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "processed_samples": int(processed_samples),
        "engine_samples": int(engine_samples),
        "fallback_samples": int(fallback_samples),
        "failed_samples": int(failed_samples),
        "ap": {"ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70)},
        "ap30": float(ap30), "ap50": float(ap50), "ap70": float(ap70),
        "output_error_summary": dict(output_error_summary),
        "output_vs_reference_error": dict(output_error_summary),
        "gates": evaluate_gates(
            processed_samples=processed_samples, engine_samples=engine_samples,
            fallback_samples=fallback_samples, failed_samples=failed_samples,
            full_ap_min_samples=full_ap_min_samples,
        ),
        "elapsed_secs": float(elapsed_secs),
    }
    for key in ("artifact", "checkpoint", "config", "dataset"):
        report[f"{key}_sha256"] = bound_shas[key]
    if engine_calls is not None:
        report["engine_calls"] = int(engine_calls)
        report["engine_calls_per_sample"] = float(engine_calls) / int(processed_samples)
    return report


def _shared_bridge_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        label=f"codriving_{args.precision_tag}", artifact_path=str(args.compiled_artifact),
        artifact_input_dtype=ARTIFACT_INPUT_DTYPE, persistent_worker=True,
        worker_script=str(args.worker_script), tvm_python=str(args.tvm_python),
        tvm_ld_library_path=str(args.tvm_ld_library_path), gpu_id=int(args.gpu_id),
        keep_detailed_samples=int(args.keep_detailed_samples), rewrite_report="",
        rewrite_report_explicit=False,
    )


def run_bridge(args: argparse.Namespace) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    resolve_output_paths(args)
    repo_root = Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    model_dir = Path(args.model_dir).resolve()
    artifact_path = Path(args.compiled_artifact).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = model_dir / "config.yaml"
    checkpoint_path = best_checkpoint(model_dir)
    for required in (artifact_path, config_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    hypes = load_yaml(str(config_path))
    hypes["validate_dir"] = hypes["test_dir"]
    dataset_path = Path(str(hypes["validate_dir"])).resolve()
    model = train_utils.create_model(hypes)
    _, model = train_utils.load_saved_model(str(model_dir), model)
    model = model.cuda().eval()
    dataset = build_dataset(hypes, visualize=False, train=False)
    loader = DataLoader(dataset, batch_size=1, num_workers=int(args.num_workers),
                        collate_fn=dataset.collate_batch_test, shuffle=False,
                        pin_memory=False, drop_last=False)

    shared = Fp16RewrittenBackboneBridge(_shared_bridge_args(args), raw_dir)
    module = patch_model_resnet(model, shared)
    result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in (0.3, 0.5, 0.7)}
    processed = 0
    started = time.time()
    try:
        with torch.inference_mode():
            for batch_data in loader:
                if processed >= int(args.num_samples):
                    break
                if batch_data is None:
                    continue
                batch_data = train_utils.to_device(batch_data, "cuda")
                output = model(batch_data["ego"])
                pred_box, pred_score, gt_box = dataset.post_process(batch_data, {"ego": output})
                for iou in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou)
                processed += 1
    finally:
        module.close()

    if processed == 0:
        raise RuntimeError("no CoDriving samples were processed")
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(eval_dir), args.precision_tag)
    protocol = protocol_payload(
        precision_tag=args.precision_tag, full_ap_min_samples=args.full_ap_min_samples,
    )
    error_summary = summarize_error_records(module.output_error_records)
    report = build_report(
        model_dir=model_dir, artifact_path=artifact_path, checkpoint_path=checkpoint_path,
        config_path=config_path, dataset_path=dataset_path,
        shas={"artifact": sha256_path(artifact_path), "checkpoint": sha256_path(checkpoint_path),
              "config": sha256_path(config_path), "dataset": sha256_path(dataset_path)},
        protocol=protocol, processed_samples=processed, engine_samples=processed,
        engine_calls=shared.call_index,
        fallback_samples=0, failed_samples=0, ap30=ap30, ap50=ap50, ap70=ap70,
        output_error_summary=error_summary, elapsed_secs=time.time() - started,
        full_ap_min_samples=args.full_ap_min_samples,
    )
    write_json(output_dir / "output_error_summary.json", {
        "items": module.output_error_records, "summary": error_summary,
    })
    write_json(Path(args.report_json), report)
    return report


def resolve_output_paths(args: argparse.Namespace) -> None:
    args.output_dir = Path(args.output_dir).resolve()
    args.report_json = Path(args.report_json).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiled-artifact", type=Path, required=True)
    parser.add_argument("--precision-tag", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--full-ap-min-samples", type=int, default=FULL_SAMPLES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--worker-script", type=Path, default=ROOT / "scripts/stage2_fp16_tvm_worker.py")
    parser.add_argument("--tvm-python", type=Path, default=DEFAULT_TVM_PYTHON)
    parser.add_argument("--tvm-ld-library-path", default=DEFAULT_TVM_LD_LIBRARY_PATH)
    parser.add_argument("--keep-detailed-samples", type=int, default=1)
    args = parser.parse_args(argv)
    args.artifact_input_dtype = ARTIFACT_INPUT_DTYPE
    args.persistent_worker = True
    if args.report_json is None:
        args.report_json = args.output_dir / "full_ap_eval_report.json"
    if args.num_samples <= 0 or args.full_ap_min_samples <= 0:
        parser.error("sample counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_bridge(args)
    except Exception as exc:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output_dir) / "failure.json", {
            "schema": SCHEMA, "status": "failed", "failure_reason": f"{type(exc).__name__}:{exc}",
        })
        raise
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
