#!/usr/bin/env python3
"""Index the fixed Stage3 Gold96 pilot evidence for downstream planners."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "stage3_gold_coldstart96_manifest_v3"
PILOT_GROUPS = (("pyramid", "32x32x128"), ("codriving", "32x64x128"))
RUNNERS = (
    ("trt_fp16", "trt_engine", "fp16"),
    ("trt_int8", "trt_engine", "int8"),
    ("tvm_fp16", "tvm_auto", "fp16"),
    ("tvm_int8", "tvm_auto", "int8"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _manifest_pilot_rows(manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    jobs = manifest.get("jobs")
    expected_groups = [f"{model}|{width}" for model, width in PILOT_GROUPS]
    if manifest.get("schema_version") != MANIFEST_SCHEMA or not isinstance(jobs, list) or len(jobs) != 96:
        raise ValueError(f"expected {MANIFEST_SCHEMA} with exactly 96 jobs")
    if manifest.get("pilot_group_ids") != expected_groups:
        raise ValueError(f"manifest.pilot_group_ids must equal {expected_groups}")

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    expected = {
        (model, width, dispatch, q_mode)
        for model, width in PILOT_GROUPS
        for _, dispatch, q_mode in RUNNERS
    }
    for source in jobs:
        if not isinstance(source, Mapping):
            raise ValueError("manifest.jobs entries must be objects")
        key = (
            str(source.get("model") or ""),
            str(source.get("width_key") or ""),
            str(source.get("dispatch_key") or ""),
            str(source.get("q_mode") or ""),
        )
        if key not in expected:
            continue
        runner = next(name for name, dispatch, q_mode in RUNNERS if (dispatch, q_mode) == key[2:])
        selected_key = (f"{key[0]}|{key[1]}", runner)
        if selected_key in selected:
            raise ValueError(f"duplicate pilot manifest row: {selected_key[0]}/{runner}")
        if str(source.get("group_id") or selected_key[0]) != selected_key[0]:
            raise ValueError(f"pilot manifest row has mismatched group_id: {source.get('job_id')}")
        selected[selected_key] = dict(source)
    if len(selected) != 8:
        missing = sorted(
            f"{model}|{width}/{runner}"
            for model, width in PILOT_GROUPS
            for runner, _, _ in RUNNERS
            if (f"{model}|{width}", runner) not in selected
        )
        raise ValueError(f"manifest is missing pilot rows: {missing}")
    return selected


def _performance_paths(root: Path, model: str, width: str, runner: str) -> tuple[Path, Path]:
    fixed = {
        ("pyramid", "trt_fp16"): (
            "trt/pyramid_fp16_gold.json",
            "trt/pyramid_fp16_gold_artifacts/compiled.engine",
        ),
        ("pyramid", "trt_int8"): (
            "trt/pyramid_int8_gold/result.json",
            "trt/pyramid_int8_gold/artifacts/compiled.engine",
        ),
        ("pyramid", "tvm_fp16"): (
            "tvm_fp16/stage3_pyramid_32x32x128_fp16_gold/route_b_fp16_auto_result.json",
            "tvm_fp16/stage3_pyramid_32x32x128_fp16_gold/route_b_fp16_auto.so",
        ),
        ("pyramid", "tvm_int8"): (
            "tvm_int8/stage3_pyramid_32x32x128_int8_gold/route_b_int8_auto_decomp_result.json",
            "tvm_int8/stage3_pyramid_32x32x128_int8_gold/route_b_int8_auto_decomp.vmexec",
        ),
        ("codriving", "trt_fp16"): (
            "trt/codriving_fp16/result.json",
            "trt/codriving_fp16/artifacts/compiled.engine",
        ),
        ("codriving", "trt_int8"): (
            "trt/codriving_int8/result.json",
            "trt/codriving_int8/artifacts/compiled.engine",
        ),
        ("codriving", "tvm_fp16"): (
            "tvm_fp16/stage3_codriving_32x64x128_fp16_gold/route_b_fp16_auto_result.json",
            "tvm_fp16/stage3_codriving_32x64x128_fp16_gold/route_b_fp16_auto.so",
        ),
        ("codriving", "tvm_int8"): (
            "tvm_int8/stage3_codriving_32x64x128_int8/route_b_int8_auto_decomp_result.json",
            "tvm_int8/stage3_codriving_32x64x128_int8/route_b_int8_auto_decomp.vmexec",
        ),
    }
    result, artifact = fixed[(model, runner)]
    return root / result, root / artifact


def _validate_fixed_artifact(payload: Mapping[str, Any], result: Path, artifact: Path, runner: str) -> None:
    if not artifact.is_file():
        raise ValueError(f"missing compiled artifact: {artifact}")
    if runner.startswith("tvm_"):
        declared = payload.get("artifact_path") or payload.get("compiled_artifact_path")
        if not isinstance(declared, str) or Path(declared).resolve() != artifact.resolve():
            raise ValueError(f"performance result does not bind the fixed compiled artifact: {result}")
    else:
        artifact_shas = payload.get("artifact_sha256")
        declared = artifact_shas.get("compiled_engine") if isinstance(artifact_shas, Mapping) else None
        if declared != _sha256(artifact):
            raise ValueError(f"performance result compiled_engine SHA256 does not match fixed engine: {result}")


def _build_performance_indexes(
    manifest_rows: Mapping[tuple[str, str], Mapping[str, Any]], root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for model, width in PILOT_GROUPS:
        group_id = f"{model}|{width}"
        for runner, dispatch, q_mode in RUNNERS:
            source = manifest_rows[(group_id, runner)]
            result, artifact = _performance_paths(root, model, width, runner)
            payload = _read_object(result, "performance result")
            _validate_fixed_artifact(payload, result, artifact, runner)
            job_id = f"{group_id}|{runner}"
            jobs.append({
                "schema_version": "stage3_gold96_pilot_job_v3",
                "job_id": job_id,
                "manifest_job_id": str(source["job_id"]),
                "group_id": group_id,
                "model": model,
                "width_key": width,
                "q_mode": q_mode,
                "dispatch_key": dispatch,
                "runner_key": runner,
                "capability_profile_id": str(source.get("capability_profile_id") or ""),
                "expected_result_json": str(result),
                "compiled_artifact": str(artifact),
            })
            states.append({
                "schema_version": "stage3_gold96_pilot_state_v3",
                "job_id": job_id,
                "group_id": group_id,
                "runner_key": runner,
                "status": "success",
                "attempt": 1,
                "result_json": str(result),
                "result_sha256": _sha256(result),
                "compiled_artifact": str(artifact),
                "artifact_sha256": _sha256(artifact),
            })
    return jobs, states


def _report_ap(payload: Mapping[str, Any], path: Path) -> dict[str, float]:
    source = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else payload
    ap: dict[str, float] = {}
    for key in ("ap30", "ap50", "ap70"):
        value = source.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"full AP report missing numeric {key}: {path}")
        ap[key] = float(value)
    return ap


def _sample_count(payload: Mapping[str, Any]) -> int:
    for key in ("processed_samples", "num_samples", "sample_count", "evaluated_samples"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _build_ap_seeds(root: Path, rows: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    group_id = "pyramid|32x32x128"
    specs = (
        ("trt_fp16", "fp16", "ap/pyramid_trt_fp16_full/stage3_trt_multiscale_ap_bridge_report.json", "success"),
        ("trt_int8", "int8", "ap/pyramid_trt_int8_full/stage3_trt_multiscale_ap_bridge_report.json", "success"),
        ("tvm_fp16", "fp16", "ap/pyramid_tvm_fp16_full_v4/fp16_rewritten_activation_bridge_report.json", "success"),
        ("tvm_int8", "int8", "ap/pyramid_tvm_int8_sanity_v4/output_dequant_calibration_summary.json", "failed"),
    )
    seeds: list[dict[str, Any]] = []
    for runner, q_mode, relative_report, status in specs:
        source = rows[(group_id, runner)]
        profile = str(source.get("capability_profile_id") or "")
        report = root / relative_report
        payload = _read_object(report, "AP report")
        base: dict[str, Any] = {
            "schema_version": "stage3_gold96_ap_seed_state_v3",
            "record_type": "terminal_event",
            "job_id": str(source["job_id"]),
            "manifest_job_id": str(source["job_id"]),
            "performance_job_id": f"{group_id}|{runner}",
            "group_id": group_id,
            "model": "pyramid",
            "width_key": "32x32x128",
            "runner_key": runner,
            "q_mode": q_mode,
            "stage": "full" if status == "success" else "sanity",
            "status": status,
            "report_json": str(report),
            "report_sha256": _sha256(report),
        }
        if status == "success":
            claim_ok = payload.get("engine_ap_claim") is True or (
                payload.get("ap_measured") is True and payload.get("smoke_gate_passed") is True
            )
            if not claim_ok or _sample_count(payload) != 1789:
                raise ValueError(f"expected a successful 1789-sample full AP report: {report}")
            base["ap"] = _report_ap(payload, report)
        else:
            blockers = [name for name, item in (payload.get("outputs") or {}).items() if not item.get("passed")]
            if payload.get("status") != "blocked" or _sample_count(payload) != 16 or not blockers:
                raise ValueError(f"sanity failure report must contain 16 samples and blockers: {report}")
            base["failure_class"] = "feasibility_failure"
            base["blockers"] = [str(item) for item in blockers]
        seeds.append(base)
    return seeds


def build_indexes(
    manifest: Mapping[str, Any], pilot_root: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(pilot_root)
    if not root.is_dir():
        raise ValueError(f"pilot root does not exist: {root}")
    rows = _manifest_pilot_rows(manifest)
    jobs, states = _build_performance_indexes(rows, root)
    return jobs, states, _build_ap_seeds(root, rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    manifest = _read_object(Path(args.manifest_json), "v3 manifest")
    jobs, states, seeds = build_indexes(manifest, args.pilot_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "pilot_jobs_jsonl": output / "pilot_jobs.jsonl",
        "pilot_state_jsonl": output / "pilot_state.jsonl",
        "ap_seed_state_jsonl": output / "ap_seed_state.jsonl",
    }
    _write_jsonl(paths["pilot_jobs_jsonl"], jobs)
    _write_jsonl(paths["pilot_state_jsonl"], states)
    _write_jsonl(paths["ap_seed_state_jsonl"], seeds)
    print(json.dumps({key: str(value) for key, value in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
