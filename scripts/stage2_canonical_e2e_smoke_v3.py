#!/usr/bin/env python3
"""Assemble measured TRT rows and close the canonical Stage2 feedback smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage2.cost_model_bundle_v3 import (
    fit_model_bundle,
    predict_rows,
    save_model_bundle,
)
from framework.stage2.measurement_contract_v3 import (
    compute_pipeline_fingerprint,
    partition_measurement_rows,
    sha256_file,
    validate_measurement_row,
)
from framework.stage2.search_loop_v3 import (
    apply_measurement_feedback,
    build_measurement_request,
    nondominated_ranks,
    select_candidates,
)


def _sha_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path.resolve()), "sha256": sha256_file(path)}


def _profile(s1_report: Path, profiler_path: Path) -> dict[str, Any]:
    report = json.loads(s1_report.read_text(encoding="utf-8"))
    features = report["s1_features"]["trt-profile"]
    return build_capability_profile(
        capability_profile_id="h800-trt-canonical-smoke-v3",
        hardware_target="h800",
        compiler_fingerprint=_sha_payload(
            {"trt_version": "10.13.0.35", "profiler_sha256": sha256_file(profiler_path)}
        ),
        dispatch_key="trt_engine",
        features={
            "supports_fp16_tensorcore": 1.0,
            "supports_int8_tensorcore": 1.0,
            "int8_channel_alignment": 16.0,
            "fp16_channel_alignment": 8.0,
            **features,
        },
    )


def _measurement_row(root: Path, q_mode: str, profile: dict[str, Any]) -> dict[str, Any]:
    measurement_path = root / f"{q_mode}_measurement.json"
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    artifact_dir = root / q_mode
    build_config = artifact_dir / "engine_build_config.json"
    inspector = artifact_dir / "engine_inspector.json"
    engine = artifact_dir / "compiled.engine"
    onnx = root / "source" / "smbo_32x64x128_backbone.onnx"
    artifacts = [
        _artifact(build_config, "engine_build_config"),
        _artifact(inspector, "engine_inspector"),
        _artifact(engine, "compiled_engine"),
        _artifact(measurement_path, "measurement_report"),
        _artifact(onnx, "source_onnx"),
    ]
    calibration_sha = None
    calibration_kind = "none"
    if q_mode == "int8":
        calibration = artifact_dir / "calibration.cache"
        calibration_manifest = artifact_dir / "calibration_manifest.json"
        manifest_payload = json.loads(calibration_manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("schema_version") != "stage2_formal_calibration_manifest_v3":
            raise ValueError("unexpected formal calibration manifest schema")
        if manifest_payload.get("sample_count") != 123:
            raise ValueError("formal smoke calibration manifest must contain 123 samples")
        if manifest_payload.get("input_name") != measurement["input_name"]:
            raise ValueError("calibration input_name does not match measured engine input")
        if manifest_payload.get("input_shape") != measurement["input_shape"]:
            raise ValueError("calibration input_shape does not match measured engine input")
        if manifest_payload.get("source_onnx_sha256") != sha256_file(onnx):
            raise ValueError("calibration manifest source ONNX mismatch")
        calibration_files = manifest_payload.get("files")
        if not isinstance(calibration_files, list) or len(calibration_files) != 123:
            raise ValueError("calibration manifest file list is incomplete")
        if any(
            not item.get("name")
            or int(item.get("size_bytes", 0)) <= 0
            or len(str(item.get("sha256", ""))) != 64
            for item in calibration_files
        ):
            raise ValueError("calibration manifest contains invalid file evidence")
        artifacts.append(_artifact(calibration, "calibration_cache"))
        artifacts.append(_artifact(calibration_manifest, "calibration_manifest"))
        calibration_sha = sha256_file(calibration_manifest)
        calibration_kind = "formal"
    expected_hashes = {
        "source_onnx": sha256_file(onnx),
        "compiled_engine": sha256_file(engine),
        "engine_build_config": sha256_file(build_config),
        "engine_inspector": sha256_file(inspector),
    }
    if q_mode == "int8":
        expected_hashes["calibration_manifest"] = calibration_sha
        expected_hashes["calibration_cache"] = sha256_file(calibration)
    if measurement.get("artifact_sha256") != expected_hashes:
        raise ValueError(f"measurement/artifact provenance mismatch for {q_mode}")
    build_config_payload = json.loads(build_config.read_text(encoding="utf-8"))
    if build_config_payload.get("source_onnx_sha256") != sha256_file(onnx):
        raise ValueError("engine build config source ONNX mismatch")
    if q_mode == "int8" and build_config_payload.get("calibration_manifest_sha256") != calibration_sha:
        raise ValueError("engine build config calibration manifest mismatch")
    row = {
        "schema_version": "stage2_measurement_row_v3",
        "row_id": f"pyramid|32x64x128|q={q_mode}|profile=h800-trt-canonical-smoke-v3",
        "model": "pyramid",
        "dataset": "dair-v2x-c",
        "checkpoint_id": "smbo-32x64x128-backbone",
        "width": [32, 64, 128],
        "q_mode": q_mode,
        "mixed_policy_id": "none",
        "capability_profile_id": profile["capability_profile_id"],
        "pipeline": {
            "onnx_sha256": sha256_file(onnx),
            "calibration_sha256": calibration_sha,
            "calibration_kind": calibration_kind,
            "lowering_origin": "compiler_engine_automatic",
            "source_ir_sha256": None,
            "schedule_trace_sha256": None,
            "tuning_database_sha256": None,
            "compiler_fingerprint": profile["compiler_fingerprint"],
            "optimized_scope": "backbone_only",
            "input_shape": list(measurement["input_shape"]),
            "batch_size": int(measurement["input_shape"][0]),
            "tuning_budget": {
                "workspace_gb": 4,
                "warmup": measurement["warmup"],
                "iters": measurement["iters"],
                "repeat": measurement["repeat"],
            },
            "engine_build_config_sha256": sha256_file(build_config),
            "engine_inspector_sha256": sha256_file(inspector),
            "compiled_engine_sha256": sha256_file(engine),
        },
        "statuses": {
            "build": "success" if measurement["build_success"] else "failed",
            "numerical": "pass" if measurement["numerical_finite"] else "fail",
        },
        "provenance": {"evidence_kind": "measured", "source_artifacts": artifacts},
        "trusted_for_final_frontier": False,
    }
    fingerprint = compute_pipeline_fingerprint(row)
    row["pipeline_fingerprint"] = fingerprint
    row["metrics"] = {
        "latency": {
            "status": "measured",
            "p50_ms": measurement["lat_p50_ms"],
            "p90_ms": measurement["lat_p90_ms"],
            "pipeline_fingerprint": fingerprint,
        },
        "energy": {
            "status": "measured",
            "joules_per_inference": measurement["energy_j"],
            "pipeline_fingerprint": fingerprint,
        },
        "ap": {"status": "not_measured", "pipeline_fingerprint": fingerprint},
    }
    return validate_measurement_row(row)


def _cost_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "width": row["width"],
        "q_mode": row["q_mode"],
        "capability_profile_id": row["capability_profile_id"],
        "graph_features": {},
        "build_status": row["statuses"]["build"],
        "numerical_status": row["statuses"]["numerical"],
        "latency_ms": row["metrics"]["latency"]["p50_ms"],
        "energy_j": row["metrics"]["energy"]["joules_per_inference"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--s1-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile = _profile(args.s1_report, REPO / "framework/trt_baseline/trt_profile_v1.py")
    rows = [_measurement_row(args.root, q_mode, profile) for q_mode in ("fp16", "int8")]
    views = partition_measurement_rows(rows)
    if len(views["gold_training"]) != 2:
        raise RuntimeError("both measured rows must enter metric-specific gold_training")
    cost_rows = [_cost_row(row) for row in rows]
    initial = fit_model_bundle(cost_rows[:1], [profile], ridge=1e-4)
    before = predict_rows(initial, cost_rows, [profile])
    selected = select_candidates(
        before,
        measured_row_ids={cost_rows[0]["row_id"]},
        budget=1,
        objectives=("latency_ms", "energy_j"),
    )
    request = build_measurement_request(selected[0], [profile])
    updated = apply_measurement_feedback(initial, cost_rows[1:], [profile])
    after = predict_rows(updated, cost_rows, [profile])
    actual_vectors = [(row["latency_ms"], row["energy_j"]) for row in cost_rows]
    actual_ranks = nondominated_ranks(actual_vectors)
    report = {
        "schema_version": "stage2_canonical_e2e_smoke_v3",
        "stage2_interface_smoke_closed": True,
        "stage2_method_validated": False,
        "scope": "metric-specific latency_energy smoke; AP and final frontier remain Stage3",
        "acquisition_mode": "replay_wiring_smoke_not_causal_dispatch_experiment",
        "s2_performance_probe_enabled": False,
        "capability_profile": profile,
        "measurement_rows": rows,
        "evidence_view_counts": {name: len(value) for name, value in views.items()},
        "initial_training_rows": initial["training_row_count"],
        "updated_training_rows": updated["training_row_count"],
        "measurement_request": request,
        "predictions_before_feedback": before,
        "predictions_after_feedback": after,
        "actual_performance_pareto": [
            {**row, "pareto_rank": actual_ranks[index]} for index, row in enumerate(cost_rows)
        ],
        "closure_gates": {
            "s0_plus_s1_default": True,
            "s2_default_disabled": True,
            "measured_rows_contract_valid": True,
            "metric_specific_gold_training": True,
            "acquisition_replay_emitted_dispatch_request": request["dispatch_key"] == "trt_engine",
            "feedback_increased_training_rows": updated["training_row_count"] == 2,
            "performance_pareto_emitted": len(actual_ranks) == 2,
        },
    }
    if not all(report["closure_gates"].values()):
        raise RuntimeError("Stage2 closure gate failed")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "measurement_rows_v3.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "capability_profile_v3.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "stage2_closure_report_v3.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_model_bundle(updated, args.out / "updated_cost_model_bundle_v3.json")
    print(
        json.dumps(
            {"stage2_interface_smoke_closed": True, "gates": report["closure_gates"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
