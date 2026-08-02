#!/usr/bin/env python3
"""Audit TVM zero-trial baselines and rebuild the three-model paper CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


CODRIVING_WIDTH = [64, 128, 256]
FCOOPER_WIDTH = [64, 128, 256, 128, 256]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return rows


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_named_file_sha(path: Path) -> str:
    """Match the historical CoDriving bridge's filename-prefixed file digest."""
    path = path.resolve()
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validate_full_ap(
    path: Path,
    *,
    expected_samples: int,
    artifact_sha256: str | Iterable[str],
) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("status") not in {"success", "success_full"}:
        raise ValueError(f"full AP did not succeed: {path}")
    if int(report.get("processed_samples", report.get("dataset_samples", -1))) != expected_samples:
        raise ValueError(f"full AP sample count mismatch: {path}")
    if int(report.get("failed_samples", 0)) or int(report.get("fallback_samples", 0)):
        raise ValueError(f"full AP contains failed/fallback samples: {path}")
    recorded_artifact = report.get("artifact_sha256") or (
        report.get("sha256") or {}
    ).get("artifact")
    accepted_artifacts = (
        {artifact_sha256}
        if isinstance(artifact_sha256, str)
        else set(artifact_sha256)
    )
    if recorded_artifact not in accepted_artifacts:
        raise ValueError(f"full AP artifact SHA mismatch: {path}")
    for field in ("ap30", "ap50", "ap70"):
        _finite(report.get(field), field)
    return report


def build_codriving_audit(*, root: Path, latency_root: Path) -> dict[str, Any]:
    root = root.resolve()
    latency_root = latency_root.resolve()
    artifact_report_path = root / "artifact/artifact_report.json"
    artifact_report = _read_json(artifact_report_path)
    if (
        artifact_report.get("status") != "success"
        or artifact_report.get("schedule_policy") != "tvm_default_zero_trial"
        or int(artifact_report.get("tuning_trials", -1)) != 0
    ):
        raise ValueError("CoDriving artifact is not a successful TVM zero-trial build")
    artifact = Path(str(artifact_report["artifact_path"]))
    if not artifact.is_file() or _sha(artifact) != artifact_report.get("artifact_sha256"):
        raise ValueError("CoDriving artifact binding mismatch")

    repeats = []
    for index in range(3):
        latency_path = latency_root / f"repeat_{index}/latency_row.jsonl"
        energy_path = root / f"energy_repeats_v2/repeat_{index}/energy_row.jsonl"
        latency_rows = _read_jsonl(latency_path)
        latency_matches = [
            row
            for row in latency_rows
            if row.get("schedule_policy") == "default"
            and row.get("measurement_status") == "measured"
        ]
        energy_rows = _read_jsonl(energy_path)
        energy_matches = [
            row
            for row in energy_rows
            if row.get("schedule_policy") == "default"
            and row.get("measurement_status") == "measured"
        ]
        if len(latency_matches) != 1 or len(energy_matches) != 1:
            raise ValueError(
                f"CoDriving repeat {index} must contain exactly one default row"
            )
        latency = latency_matches[0]
        energy = energy_matches[0]
        if latency.get("width") != CODRIVING_WIDTH or energy.get("width") != CODRIVING_WIDTH:
            raise ValueError(f"CoDriving repeat {index} width mismatch")
        onnx_path = str(artifact_report.get("onnx_path") or "")
        if not onnx_path or any(
            onnx_path not in (row.get("source_files") or ())
            for row in (latency, energy)
        ):
            raise ValueError(f"CoDriving repeat {index} ONNX binding mismatch")
        repeats.append(
            {
                "repeat": index,
                "latency_ms": _finite(latency["latency_p50_us"], "latency_p50_us")
                / 1000.0,
                "energy_j": _finite(
                    energy["joule_per_inference"], "joule_per_inference"
                ),
                "latency_evidence": _binding(latency_path),
                "energy_evidence": _binding(energy_path),
            }
        )

    ap_path = root / "full_ap/full_ap_eval_report.json"
    ap = _validate_full_ap(
        ap_path,
        expected_samples=1789,
        artifact_sha256={
            str(artifact_report["artifact_sha256"]),
            _legacy_named_file_sha(artifact),
        },
    )
    return {
        "schema_version": "stage6_tvm_default_baseline_audit_v1",
        "status": "passed",
        "model": "codriving",
        "backend": "tvm",
        "schedule_policy": "tvm_default_zero_trial",
        "tuning_trials": 0,
        "config": [*CODRIVING_WIDTH, "fp32"],
        "latency_ms": statistics.median(row["latency_ms"] for row in repeats),
        "energy_j": statistics.median(row["energy_j"] for row in repeats),
        "AP30": float(ap["ap30"]),
        "AP50": float(ap["ap50"]),
        "AP70": float(ap["ap70"]),
        "performance_repeats": repeats,
        "artifact": _binding(artifact),
        "artifact_legacy_named_sha256": _legacy_named_file_sha(artifact),
        "artifact_report": _binding(artifact_report_path),
        "ap_report": _binding(ap_path),
        "onnx_sha256": artifact_report.get("onnx_sha256"),
        "evidence_origin": "stage6_tvm_default_zero_trial_measurement",
    }


def build_fcooper_audit(*, root: Path) -> dict[str, Any]:
    root = root.resolve()
    repeats = []
    artifact_sha256 = None
    artifact = None
    for index in range(3):
        repeat_dir = root / f"repeat_{index}/fcooper_default_r{index}"
        build_path = repeat_dir / "route_b_fp32_auto_build.json"
        result_path = repeat_dir / "route_b_fp32_auto_result.json"
        build = _read_json(build_path)
        result = _read_json(result_path)
        if (
            build.get("status") not in {"built", "success"}
            or build.get("build_success") is not True
            or "max_trials" not in build
            or int(build["max_trials"]) != 0
            or build.get("tuning_policy") != "default_compile_no_metaschedule"
        ):
            raise ValueError(f"F-Cooper repeat {index} is not a zero-trial build")
        if result.get("status") != "success" or result.get("build_success") is not True:
            raise ValueError(f"F-Cooper repeat {index} did not complete")
        if result.get("width") != FCOOPER_WIDTH:
            raise ValueError(f"F-Cooper repeat {index} width mismatch")
        repeat_artifact = Path(str(result["artifact_path"]))
        repeat_sha = _sha(repeat_artifact)
        if artifact_sha256 is None:
            artifact_sha256 = repeat_sha
            artifact = repeat_artifact
        elif repeat_sha != artifact_sha256:
            raise ValueError("F-Cooper zero-trial artifact SHA differs across repeats")
        energy = result.get("energy") or {}
        if energy.get("status") not in {None, "success"}:
            raise ValueError(f"F-Cooper repeat {index} energy failed")
        repeats.append(
            {
                "repeat": index,
                "latency_ms": _finite(
                    (result.get("latency") or {}).get("latency_ms_p50"),
                    "latency_ms_p50",
                ),
                "energy_j": _finite(
                    energy.get("joules_per_inference"), "joules_per_inference"
                ),
                "build_evidence": _binding(build_path),
                "result_evidence": _binding(result_path),
            }
        )
    if artifact is None or artifact_sha256 is None:
        raise ValueError("F-Cooper repeats are missing")
    ap_path = root / "full_ap/full_ap_eval_report.json"
    ap = _validate_full_ap(
        ap_path,
        expected_samples=2170,
        artifact_sha256=artifact_sha256,
    )
    return {
        "schema_version": "stage6_tvm_default_baseline_audit_v1",
        "status": "passed",
        "model": "fcooper",
        "backend": "tvm",
        "schedule_policy": "tvm_default_zero_trial",
        "tuning_trials": 0,
        "config": [*FCOOPER_WIDTH, "fp32"],
        "latency_ms": statistics.median(row["latency_ms"] for row in repeats),
        "energy_j": statistics.median(row["energy_j"] for row in repeats),
        "AP30": float(ap["ap30"]),
        "AP50": float(ap["ap50"]),
        "AP70": float(ap["ap70"]),
        "performance_repeats": repeats,
        "artifact": _binding(artifact),
        "ap_report": _binding(ap_path),
        "checkpoint_sha256": ap.get("checkpoint_sha256"),
        "config_sha256": ap.get("config_sha256"),
        "evidence_origin": "stage6_tvm_default_zero_trial_measurement",
    }


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rebuild_three_model_rows(
    *,
    source_rows: Iterable[Mapping[str, Any]],
    pyramid_rows: Iterable[Mapping[str, Any]],
    audits: Mapping[str, Mapping[str, Any]],
    delta_ap: float,
) -> list[dict[str, Any]]:
    original = [dict(row) for row in source_rows]
    pyramid_method_names = {
        "original_default": "Original/default",
        "compression_only": "Compression only",
        "schedule_only": "Schedule only",
        "compress_then_tune": "Compress -> Tune",
        "joint_shcosearch": "GEAR",
    }
    pyramid_schedule_policies = {
        "original_default": "tvm_default_zero_trial",
        "schedule_only": "metaschedule_tuned",
    }
    replacement_pyramid = [
        {
            **dict(row),
            "model": "pyramid",
            "method": pyramid_method_names[str(row["method"])],
            "schedule_policy": pyramid_schedule_policies.get(
                str(row["method"]), row.get("schedule_policy", "")
            ),
            "evidence_origin": (
                "stage6_tvm_default_zero_trial_measurement"
                if str(row["method"]) == "original_default"
                else row.get("evidence_origin", "")
            ),
        }
        for row in pyramid_rows
        if str(row.get("method")) in pyramid_method_names
    ]
    rows = [
        *(replacement_pyramid or [row for row in original if row.get("model") == "pyramid"]),
        *[row for row in original if row.get("model") != "pyramid"],
    ]
    for model, audit in audits.items():
        matches = [
            row
            for row in rows
            if row.get("model") == model and row.get("method") == "Original/default"
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {model} Original/default row")
        row = matches[0]
        row.update(
            {
                "AP70": audit["AP70"],
                "latency_ms": audit["latency_ms"],
                "energy_j": audit["energy_j"],
                "backend": "tvm",
                "config": json.dumps(audit.get("config", [])),
                "configuration": str(tuple(audit.get("config", []))),
                "schedule_policy": "tvm_default_zero_trial",
                "tuning_trials": 0,
                "evidence_origin": audit.get("evidence_origin"),
                "evidence_path": audit.get("evidence_path"),
                "terminal_status": "measured_success_gold",
                "selection_status": "fixed_original_default",
            }
        )

    for model in ("pyramid", "codriving", "fcooper"):
        model_rows = [row for row in rows if row.get("model") == model]
        default = next(
            (row for row in model_rows if row.get("method") == "Original/default"),
            None,
        )
        if default is None:
            continue
        baseline_latency = _as_float(default.get("latency_ms"))
        baseline_ap = _as_float(default.get("AP70"))
        if baseline_latency is None or baseline_ap is None:
            raise ValueError(f"{model} baseline lacks measured metrics")
        floor = baseline_ap - delta_ap
        ranked = sorted(
            (
                (_as_float(row.get("latency_ms")), row)
                for row in model_rows
                if _as_float(row.get("latency_ms")) is not None
            ),
            key=lambda pair: pair[0],
        )
        rank_by_id = {id(row): rank + 1 for rank, (_, row) in enumerate(ranked)}
        for row in model_rows:
            latency = _as_float(row.get("latency_ms"))
            ap70 = _as_float(row.get("AP70"))
            row["delta_ap_max"] = delta_ap
            row["ap70_floor"] = floor
            row["ap_constraint_violated"] = ap70 is not None and ap70 < floor
            row["speedup"] = (
                baseline_latency / latency if latency is not None and latency > 0 else ""
            )
            row["latency_rank"] = rank_by_id.get(id(row), "")
            if model != "pyramid":
                row["HV"] = ""
                row["hv_status"] = "not_recomputed_after_backend_baseline_repair"
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_audit_for_table(model: str, audit: Mapping[str, Any]) -> None:
    if (
        audit.get("schema_version") != "stage6_tvm_default_baseline_audit_v1"
        or audit.get("status") != "passed"
        or audit.get("model") != model
        or audit.get("backend") != "tvm"
        or audit.get("schedule_policy") != "tvm_default_zero_trial"
        or int(audit.get("tuning_trials", -1)) != 0
        or len(audit.get("performance_repeats") or ()) != 3
    ):
        raise ValueError(f"invalid {model} default audit")
    for key in ("artifact", "ap_report"):
        binding = audit.get(key)
        if not isinstance(binding, Mapping) or not binding.get("sha256"):
            raise ValueError(f"{model} audit lacks {key} binding")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    codriving = subparsers.add_parser("codriving")
    codriving.add_argument("--root", type=Path, required=True)
    codriving.add_argument("--latency-root", type=Path, required=True)
    codriving.add_argument("--output-json", type=Path, required=True)

    fcooper = subparsers.add_parser("fcooper")
    fcooper.add_argument("--root", type=Path, required=True)
    fcooper.add_argument("--output-json", type=Path, required=True)

    table = subparsers.add_parser("table")
    table.add_argument("--source-csv", type=Path, required=True)
    table.add_argument("--pyramid-csv", type=Path, required=True)
    table.add_argument("--codriving-audit", type=Path, required=True)
    table.add_argument("--fcooper-audit", type=Path, required=True)
    table.add_argument("--output-csv", type=Path, required=True)
    table.add_argument("--delta-ap", type=float, default=0.10)

    args = parser.parse_args()
    if args.command == "codriving":
        audit = build_codriving_audit(root=args.root, latency_root=args.latency_root)
        audit["evidence_path"] = str(args.output_json.resolve())
        _write_json(args.output_json, audit)
    elif args.command == "fcooper":
        audit = build_fcooper_audit(root=args.root)
        audit["evidence_path"] = str(args.output_json.resolve())
        _write_json(args.output_json, audit)
    else:
        audits = {
            "codriving": _read_json(args.codriving_audit),
            "fcooper": _read_json(args.fcooper_audit),
        }
        for model, audit in audits.items():
            _validate_audit_for_table(model, audit)
            audit["evidence_path"] = str(
                (args.codriving_audit if model == "codriving" else args.fcooper_audit)
                .resolve()
            )
        rows = rebuild_three_model_rows(
            source_rows=_read_csv(args.source_csv),
            pyramid_rows=_read_csv(args.pyramid_csv),
            audits=audits,
            delta_ap=args.delta_ap,
        )
        _write_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
