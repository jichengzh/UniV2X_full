#!/usr/bin/env python3
"""Build the Stage3 Gold96 AP execution plan from current performance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "stage3_gold96_ap_plan_v3"
MANIFEST_SCHEMA = "stage3_gold_coldstart96_manifest_v3"
PERFORMANCE_RUNNERS = {
    ("tvm_auto", "fp16"): "tvm_fp16",
    ("tvm_auto", "int8"): "tvm_int8",
    ("trt_engine", "fp16"): "trt_fp16",
    ("trt_engine", "int8"): "trt_int8",
}
AP_RUNNERS = {
    ("pyramid", "trt_fp16"): "pyramid_trt_multiscale",
    ("pyramid", "trt_int8"): "pyramid_trt_multiscale",
    ("pyramid", "tvm_fp16"): "pyramid_tvm_fp16_bridge",
    ("pyramid", "tvm_int8"): "pyramid_tvm_int8_numeric_gate",
    ("codriving", "trt_fp16"): "codriving_trt_multiscale",
    ("codriving", "trt_int8"): "codriving_trt_multiscale",
    ("codriving", "tvm_fp16"): "codriving_tvm_fp16_bridge",
    ("codriving", "tvm_int8"): "codriving_tvm_int8_numeric_gate",
}
RUNNER_SCRIPTS = {
    "pyramid_trt_multiscale": "scripts/stage3_trt_multiscale_ap_bridge_v3.py",
    "pyramid_tvm_fp16_bridge": "scripts/stage2_h800_fp16_rewritten_activation_bridge.py",
    "pyramid_tvm_int8_numeric_gate": "scripts/stage3_pyramid_tvm_int8_ap_numeric_gate_v3.py",
    "codriving_trt_multiscale": "scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py",
    "codriving_tvm_fp16_bridge": "scripts/stage3_codriving_tvm_fp16_ap_bridge_v3.py",
    "codriving_tvm_int8_numeric_gate": "scripts/stage3_codriving_tvm_int8_ap_numeric_gate_v3.py",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    item = Path(path)
    if not item.is_file():
        return []
    return [json.loads(line) for line in item.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pilot_rows(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    item = Path(root)
    if not item.is_dir():
        return [], []
    jobs: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for path in sorted(item.rglob("*.jsonl")):
        name = path.name.lower()
        if "job" in name:
            jobs.extend(read_jsonl(path))
        elif "state" in name:
            states.extend(read_jsonl(path))
    return jobs, states


def _expected_performance_runner(row: Mapping[str, Any]) -> str:
    key = (str(row.get("dispatch_key") or ""), str(row.get("q_mode") or row.get("q") or ""))
    try:
        return PERFORMANCE_RUNNERS[key]
    except KeyError as exc:
        raise ValueError(f"unsupported manifest dispatch/q: {key[0]}/{key[1]}") from exc


def _latest_terminal(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    terminal = [row for row in rows if str(row.get("status")) in {"success", "confirmed_failure"}]
    if not terminal:
        return None
    successes = [row for row in terminal if str(row.get("status")) == "success"]
    return successes[-1] if successes else terminal[-1]


def _compiled_artifact(
    result_json: str | Path,
    *,
    expected_runner: str,
    state_bound_artifact: str | None = None,
    state_bound_digest: str | None = None,
) -> tuple[str, str | None]:
    path = Path(result_json)
    if not path.is_file():
        raise ValueError(f"performance result_json does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("engine_path", "artifact_path", "compiled_artifact", "compiled_artifact_path", "library_path", "module_path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            digest = payload.get("engine_sha256") or payload.get("artifact_digest") or payload.get("artifact_sha256")
            return _validated_artifact(value, expected_runner), str(digest) if digest else None
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, Mapping):
        for key in ("engine_path", "artifact_path", "compiled_artifact_path"):
            value = artifacts.get(key)
            if isinstance(value, str) and value:
                return _validated_artifact(value, expected_runner), None
    trt_engine = path.parent / "artifacts" / "compiled.engine"
    if path.name == "trt_profile_result.json" and trt_engine.is_file():
        digest_map = payload.get("artifact_sha256")
        digest = digest_map.get("compiled_engine") if isinstance(digest_map, Mapping) else None
        return _validated_artifact(trt_engine, expected_runner), str(digest) if digest else None
    if state_bound_artifact:
        return _validated_artifact(state_bound_artifact, expected_runner, source="state-bound"), state_bound_digest
    raise ValueError(f"compiled artifact missing from performance result_json: {path}")


def _validated_artifact(
    artifact: str | Path,
    expected_runner: str,
    *,
    source: str = "performance result",
) -> str:
    path = Path(artifact)
    if not path.is_file():
        raise ValueError(f"{source} compiled artifact does not exist: {path}")
    if expected_runner.startswith("trt_"):
        valid = path.suffix == ".engine"
        expected = "a .engine file"
    elif expected_runner == "tvm_fp16":
        valid = path.suffix == ".so"
        expected = "a .so file"
    elif expected_runner == "tvm_int8":
        valid = path.name == "route_b_int8_auto_decomp.vmexec"
        expected = "route_b_int8_auto_decomp.vmexec"
    else:
        raise ValueError(f"unsupported performance runner for compiled artifact: {expected_runner}")
    if not valid:
        raise ValueError(f"{source} compiled artifact for {expected_runner} expected {expected}: {path}")
    return str(path)


def _command(runner_script: str, row: Mapping[str, Any], artifact: str, pilot_root: str | Path, samples: int) -> list[str]:
    width_key = str(row.get("width_key") or "x".join(map(str, row["width"])))
    q_mode = str(row.get("q_mode") or row.get("q"))
    profile = str(row.get("capability_profile_id") or row.get("profile") or "profile_missing")
    source_contract = row.get("source_contract")
    source_contract = source_contract if isinstance(source_contract, Mapping) else {}
    output_dir = Path(pilot_root) / "ap" / str(row["model"]) / width_key / q_mode / profile
    run_dir = output_dir / ("sanity_16" if samples == 16 else "full_1789")
    if runner_script == "scripts/stage3_trt_multiscale_ap_bridge_v3.py":
        padded_width = "_".join(f"{int(value):03d}" for value in row["width"])
        checkpoint_dir = str(
            source_contract.get("checkpoint_dir")
            or f"/home/jichengzhi/V2X/models/dataset_a_cache/ft_{padded_width}"
        )
        return [
            "python3", runner_script,
            "--label", f"{row['model']}_{width_key}",
            "--ckpt-dir", checkpoint_dir,
            "--engine", artifact,
            "--precision-tag", q_mode,
            "--num-samples", str(samples),
            "--full-ap-min-samples", "1789",
            "--eval-range", "102.4,51.2",
            "--raw-dir", str(run_dir),
            "--report-json", str(run_dir / "full_ap_eval_report.json"),
        ]
    if runner_script == "scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py":
        gate = "sanity" if samples == 16 else "full"
        model_dir = str(
            source_contract.get("model_dir")
            or f"/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/{width_key}"
        )
        return [
            "python3", runner_script,
            "--repo-root", "/exdata/jichengzhi/V2Xverse_pyramid",
            "--model-dir", model_dir,
            "--engine", artifact,
            "--precision-tag", q_mode,
            "--gate", gate,
            "--n-samples", str(samples),
            "--num-workers", "0",
            "--eval-dir", str(run_dir / "eval"),
            "--out-json", str(run_dir / "full_ap_eval_report.json"),
        ]
    if runner_script == "scripts/stage2_h800_fp16_rewritten_activation_bridge.py":
        padded_width = "_".join(f"{int(value):03d}" for value in row["width"])
        checkpoint_dir = str(
            source_contract.get("checkpoint_dir")
            or f"/home/jichengzhi/V2X/models/dataset_a_cache/ft_{padded_width}"
        )
        return [
            "python3", runner_script,
            "--label", f"stage3_{row['model']}_{width_key}_{q_mode}",
            "--ckpt-dir", checkpoint_dir,
            "--raw-dir", str(run_dir),
            "--eval-range", "102.4,51.2",
            "--artifact-path", artifact,
            "--artifact-input-dtype", "float32",
            "--persistent-worker",
            "--num-samples", str(samples),
            "--full-ap-min-samples", "1789",
            "--export-report-json", str(run_dir / "full_ap_eval_report.json"),
        ]
    if runner_script == "scripts/stage3_codriving_tvm_fp16_ap_bridge_v3.py":
        model_dir = str(
            source_contract.get("model_dir")
            or f"/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/{width_key}"
        )
        return [
            "python3", runner_script,
            "--compiled-artifact", artifact,
            "--precision-tag", q_mode,
            "--num-samples", str(samples),
            "--full-ap-min-samples", "1789",
            "--output-dir", str(run_dir),
            "--model-dir", model_dir,
            "--repo-root", "/exdata/jichengzhi/V2Xverse_pyramid",
            "--num-workers", "0",
            "--report-json", str(run_dir / "full_ap_eval_report.json"),
        ]
    if runner_script == "scripts/stage3_pyramid_tvm_int8_ap_numeric_gate_v3.py":
        padded_width = "_".join(f"{int(value):03d}" for value in row["width"])
        checkpoint_dir = str(
            source_contract.get("checkpoint_dir")
            or f"/home/jichengzhi/V2X/models/dataset_a_cache/ft_{padded_width}"
        )
        return [
            "python3", runner_script,
            "--compiled-artifact", artifact,
            "--precision-tag", q_mode,
            "--num-samples", str(samples),
            "--full-ap-min-samples", "1789",
            "--output-dir", str(run_dir),
            "--model-dir", checkpoint_dir,
            "--eval-range", "102.4,51.2",
            "--report-json", str(run_dir / "full_ap_eval_report.json"),
        ]
    if runner_script == "scripts/stage3_codriving_tvm_int8_ap_numeric_gate_v3.py":
        model_dir = str(
            source_contract.get("model_dir")
            or f"/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/{width_key}"
        )
        command = [
            "python3", runner_script,
            "--compiled-artifact", artifact,
            "--precision-tag", q_mode,
            "--num-samples", str(samples),
            "--full-ap-min-samples", "1789",
            "--output-dir", str(run_dir),
            "--model-dir", model_dir,
            "--repo-root", "/exdata/jichengzhi/V2Xverse_pyramid",
            "--num-workers", "0",
            "--report-json", str(run_dir / "full_ap_eval_report.json"),
        ]
        checkpoint_path = str(source_contract.get("checkpoint_path") or "")
        if checkpoint_path:
            command.extend(["--checkpoint-path", checkpoint_path])
        return command
    return [
        "python3", runner_script,
        "--compiled-artifact", artifact,
        "--precision-tag", str(row.get("q_mode") or row.get("q")),
        "--num-samples", str(samples),
        "--full-ap-min-samples", "1789",
        "--output-dir", str(run_dir),
    ]


def build_ap_plan(
    manifest: Mapping[str, Any],
    *,
    performance_jobs: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    pilot_root: str | Path,
    manifest_schema: str = MANIFEST_SCHEMA,
    expected_row_count: int = 96,
) -> list[dict[str, Any]]:
    jobs = manifest.get("jobs")
    if (
        manifest.get("schema_version") != manifest_schema
        or not isinstance(jobs, list)
        or len(jobs) != expected_row_count
    ):
        raise ValueError(f"expected {manifest_schema} with exactly {expected_row_count} jobs")

    pilot_jobs, pilot_states = _pilot_rows(pilot_root)
    all_jobs = [dict(job) for job in performance_jobs] + pilot_jobs
    all_states = [dict(row) for row in performance_state_rows] + pilot_states
    state_by_job: dict[str, list[dict[str, Any]]] = {}
    for state in all_states:
        state_by_job.setdefault(str(state.get("job_id") or ""), []).append(state)

    plan: list[dict[str, Any]] = []
    for source in jobs:
        manifest_id = str(source["job_id"])
        expected_runner = _expected_performance_runner(source)
        matches = [
            job for job in all_jobs
            if str(job.get("manifest_job_id") or "") == manifest_id
            and str(job.get("runner_key") or "") == expected_runner
            and str(job.get("model") or source.get("model")) == str(source.get("model"))
        ]
        terminal_pairs = [(job, _latest_terminal(state_by_job.get(str(job.get("job_id") or ""), []))) for job in matches]
        terminal_pairs = [(job, state) for job, state in terminal_pairs if state is not None]
        job, state = terminal_pairs[-1] if terminal_pairs else (matches[-1] if matches else None, None)
        performance_terminal = str(state.get("status")) if state else "pending"
        ap_runner = AP_RUNNERS[(str(source["model"]), expected_runner)]
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "manifest_job_id": manifest_id,
            "model": source["model"],
            "width": list(source["width"]),
            "q": source.get("q_mode") or source.get("q"),
            "profile": source.get("capability_profile_id") or source.get("profile"),
            "required_metrics": list(source.get("required_metrics") or []),
            "source_contract": dict(source.get("source_contract") or {}),
            "performance_terminal": performance_terminal,
            "performance_job_id": job.get("job_id") if job else None,
            "performance_result_json": state.get("result_json") if state else None,
            "runner_key": ap_runner,
            "compiled_artifact": None,
            "compiled_artifact_path": None,
            "compiled_artifact_digest": None,
            "sanity_command": None,
            "full_command": None,
            "full_command_state_bindings": None,
        }
        if performance_terminal == "confirmed_failure":
            row["ap_terminal"] = "feasibility_failure"
        elif performance_terminal != "success":
            row["ap_terminal"] = "blocked_performance_not_success"
        elif not state.get("result_json"):
            row["ap_terminal"] = "blocked_result_json_missing"
        else:
            try:
                artifact, digest = _compiled_artifact(
                    str(state["result_json"]),
                    expected_runner=expected_runner,
                    state_bound_artifact=str(state.get("compiled_artifact") or "") or None,
                    state_bound_digest=str(state.get("artifact_sha256") or "") or None,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                row["ap_terminal"] = "blocked_compiled_artifact_missing"
                row["block_reason"] = str(exc)
            else:
                row["compiled_artifact"] = artifact
                row["compiled_artifact_path"] = artifact
                row["compiled_artifact_digest"] = digest
                script = RUNNER_SCRIPTS[ap_runner]
                if not (REPO_ROOT / script).is_file():
                    row["ap_terminal"] = "blocked_runner_missing"
                    row["block_reason"] = script
                else:
                    row["ap_terminal"] = "ready"
                    row["sanity_command"] = _command(script, source, artifact, pilot_root, 16)
                    row["full_command"] = _command(script, source, artifact, pilot_root, 1789)
                    if ap_runner == "codriving_tvm_int8_numeric_gate":
                        row["full_command_state_bindings"] = {
                            "sanity_report": {
                                "command_option": "--sanity-report-json",
                                "sha256_command_option": "--sanity-report-sha256",
                                "state_stage": "sanity",
                                "state_status": "success",
                                "path_field": "report_path",
                                "sha256_field": "report_sha256",
                                "verify_sha256": True,
                            }
                        }
        plan.append(row)
    return plan


def write_outputs(rows: Sequence[Mapping[str, Any]], output_json: str | Path, output_jsonl: str | Path) -> tuple[Path, Path]:
    json_path, jsonl_path = Path(output_json), Path(output_jsonl)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "row_count": len(rows), "jobs": list(rows)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jsonl_path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return json_path, jsonl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--performance-jobs-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--performance-state-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_ap_plan(
        json.loads(args.manifest_json.read_text(encoding="utf-8")),
        performance_jobs=[row for path in args.performance_jobs_jsonl for row in read_jsonl(path)],
        performance_state_rows=[row for path in args.performance_state_jsonl for row in read_jsonl(path)],
        pilot_root=args.pilot_root,
    )
    output_json, output_jsonl = write_outputs(rows, args.output_json, args.output_jsonl)
    print(json.dumps({"schema_version": SCHEMA_VERSION, "row_count": len(rows), "output_json": str(output_json), "output_jsonl": str(output_jsonl)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
