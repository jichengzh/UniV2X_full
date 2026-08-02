#!/usr/bin/env python3
"""Admit the SHA-identical native F-Cooper FP32 baseline into the TVM table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

from scripts.fcooper_tvm_evidence_pools_v1 import binding, write_immutable


SUCCESS = "measured_success_gold"
BASE_WIDTH = [64, 128, 256, 128, 256]
SCOPE = "post_scatter_backbone_shrinker"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _original_row(audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in audit.get("rows", [])
        if str(row.get("method")) == "Original/default"
    ]
    if len(rows) != 1:
        raise ValueError("TRT v2 audit must contain one Original/default row")
    row = rows[0]
    if (
        row.get("terminal_status") != SUCCESS
        or row.get("configuration") != "(64,128,256,128,256,fp32)"
    ):
        raise ValueError("TRT v2 original row does not match the native scope")
    return row


def _verify_sha(path: Path, expected: Any, label: str) -> str:
    actual = _file_sha(path)
    if not isinstance(expected, str) or actual != expected:
        raise ValueError(f"{label} SHA drift")
    return actual


def admit_original(
    *,
    trt_audit_path: Path,
    preflight_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    ap_report_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    audit = _read(trt_audit_path)
    preflight = _read(preflight_path)
    ap_report = _read(ap_report_path)
    if (
        audit.get("paper_ready") is not True
        or audit.get("backend") != "trt"
        or int(audit.get("performance_gpu_index", -1)) != 7
        or preflight.get("passed") is not True
    ):
        raise ValueError("source TRT v2 run is not a paper-ready GPU7 baseline")
    row = _original_row(audit)
    checkpoint_sha = _verify_sha(
        checkpoint_path, preflight.get("checkpoint_sha256"), "checkpoint"
    )
    config_sha = _verify_sha(config_path, preflight.get("config_sha256"), "config")
    ap_sha = _verify_sha(
        ap_report_path, preflight.get("ap_reference_report_sha256"), "AP report"
    )
    selected = (audit.get("selected_evidence") or {}).get("Original/default")
    if not isinstance(selected, Mapping):
        raise ValueError("TRT v2 audit lacks selected native evidence")
    if (
        selected.get("checkpoint_sha256") != checkpoint_sha
        or selected.get("ap_report_sha256") != ap_sha
    ):
        raise ValueError("selected native evidence SHA drift")
    if (
        ap_report.get("status") != "success_full"
        or int(ap_report.get("dataset_samples", -1)) != 2170
        or ap_report.get("checkpoint_sha256") != checkpoint_sha
        or ap_report.get("config_sha256") != config_sha
    ):
        raise ValueError("native AP report contract is invalid")
    evaluation_path = Path(str(ap_report.get("raw_report_path") or ""))
    evaluation_sha = _verify_sha(
        evaluation_path, ap_report.get("raw_report_sha256"), "evaluation report"
    )
    for metric in ("ap30", "ap50", "ap70"):
        if not math.isclose(
            _finite(ap_report.get(metric), f"AP report {metric}"),
            _finite(row.get(metric), f"audit {metric}"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("native AP metric drift")

    repeat_paths = [Path(path) for path in selected.get("repeat_paths", [])]
    repeat_shas = list(selected.get("repeat_sha256", []))
    if len(repeat_paths) != 3 or len(repeat_shas) != 3:
        raise ValueError("native baseline requires three repeat reports")
    repeats: list[dict[str, Any]] = []
    latencies: list[float] = []
    energies: list[float] = []
    for index, (path, expected_sha) in enumerate(zip(repeat_paths, repeat_shas)):
        _verify_sha(path, expected_sha, f"repeat {index}")
        report = _read(path)
        if (
            report.get("backend") != "pytorch_cuda_cudnn"
            or int(report.get("gpu_abs", -1)) != 7
            or report.get("optimized_scope") != SCOPE
            or report.get("precision") != "fp32"
            or report.get("checkpoint_sha256") != checkpoint_sha
            or report.get("config_sha256") != config_sha
        ):
            raise ValueError(f"native repeat {index} contract drift")
        latency = _finite(report.get("latency_ms"), "repeat latency")
        energy = _finite(report.get("energy_j"), "repeat energy")
        latencies.append(latency)
        energies.append(energy)
        repeats.append(
            {
                "gpu_index": 7,
                "latency_ms": latency,
                "energy_j": energy,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "report": binding(path, str(expected_sha)),
            }
        )
    if not (
        math.isclose(statistics.median(latencies), float(row["latency_ms"]))
        and math.isclose(statistics.median(energies), float(row["energy_j"]))
    ):
        raise ValueError("native repeat medians drift from the paper-ready audit")

    artifacts = {
        "checkpoint": binding(checkpoint_path, checkpoint_sha),
        "config": binding(config_path, config_sha),
        "ap_report": binding(ap_report_path, ap_sha),
        "evaluation_report": binding(evaluation_path, evaluation_sha),
    }
    original_contract = {
        "schema_version": "fcooper_tvm_native_original_admission_v1",
        "same_scope_sha_admission": True,
        "source_policy": (
            "reuse backend-neutral native FP32 baseline only; no TRT compiled "
            "artifact, latency label, or winner label enters TVM search"
        ),
        "source_trt_v2_audit": binding(trt_audit_path),
        "source_preflight": binding(preflight_path),
        "row": {
            "row_id": "fcooper-original-default",
            "terminal_status": SUCCESS,
            "backend": "pytorch_cuda_cudnn",
            "optimized_scope": SCOPE,
            "width": BASE_WIDTH,
            "q_mode": "fp32",
            **{metric: row[metric] for metric in ("ap30", "ap50", "ap70")},
            "latency_ms": statistics.median(latencies),
            "energy_j": statistics.median(energies),
            "artifacts": artifacts,
        },
    }
    validation = {
        "schema_version": "fcooper_native_gpu7_validation_admission_v1",
        "row_id": "fcooper-original-default",
        "gpu_index": 7,
        "performance_repeats": repeats,
        "full_ap": {
            "gpu_index": 7,
            **{metric: row[metric] for metric in ("ap30", "ap50", "ap70")},
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha,
            "report": artifacts["ap_report"],
            "evaluation_report": artifacts["evaluation_report"],
        },
    }
    admission_audit = {
        "schema_version": "fcooper_tvm_native_original_admission_audit_v1",
        "passed": True,
        "same_scope_sha_admission": True,
        "performance_repeat_count": 3,
        "full_ap_sample_count": 2170,
        "trt_compiled_artifacts_reused": False,
        "native_baseline_only": True,
    }
    output_dir = Path(output_dir)
    outputs = {
        "original_contract": output_dir / "original_contract.json",
        "validation_manifest": output_dir / "validation_manifest.json",
        "admission_audit": output_dir / "admission_audit.json",
    }
    write_immutable(outputs["original_contract"], original_contract)
    write_immutable(outputs["validation_manifest"], validation)
    write_immutable(outputs["admission_audit"], admission_audit)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-audit", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ap-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = admit_original(
        trt_audit_path=args.trt_audit,
        preflight_path=args.preflight,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        ap_report_path=args.ap_report,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {name: binding(path) for name, path in outputs.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
