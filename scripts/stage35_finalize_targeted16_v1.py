#!/usr/bin/env python3
"""Finalize the 16-row Stage3.5 targeted supplement with strict Gold gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage3_finalize_gold96_v3 import finalize_gold_dataset, read_jsonl


MANIFEST_SCHEMA = "stage35_gold128_targeted_supplement_manifest_v1"
OUTPUT_SCHEMA = "stage35_targeted16_final_v1"
FEEDBACK_MANIFEST_SCHEMA = "stage4_feedback16_manifest_v1"
FEEDBACK_OUTPUT_SCHEMA = "stage4_feedback16_final_v1"
SCHEMA_REQUIRED_SPLITS = {
    MANIFEST_SCHEMA: "train",
    FEEDBACK_MANIFEST_SCHEMA: "online_feedback",
}
SCHEMA_REQUIRED_OUTPUTS = {
    MANIFEST_SCHEMA: OUTPUT_SCHEMA,
    FEEDBACK_MANIFEST_SCHEMA: FEEDBACK_OUTPUT_SCHEMA,
}
ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}
PERFORMANCE_RUNNERS = {
    ("tvm_auto", "fp16"): "tvm_fp16",
    ("tvm_auto", "int8"): "tvm_int8",
    ("trt_engine", "fp16"): "trt_fp16",
    ("trt_engine", "int8"): "trt_int8",
}
AP_RUNNERS = {
    ("pyramid", "tvm_fp16"): "pyramid_tvm_fp16_bridge",
    ("pyramid", "tvm_int8"): "pyramid_tvm_int8_numeric_gate",
    ("pyramid", "trt_fp16"): "pyramid_trt_multiscale",
    ("pyramid", "trt_int8"): "pyramid_trt_multiscale",
    ("codriving", "tvm_fp16"): "codriving_tvm_fp16_bridge",
    ("codriving", "tvm_int8"): "codriving_tvm_int8_numeric_gate",
    ("codriving", "trt_fp16"): "codriving_trt_multiscale",
    ("codriving", "trt_int8"): "codriving_trt_multiscale",
}
METRIC_FIELDS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
HASH_FIELDS = ("performance_result_sha256", "ap_report_sha256")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _resolve_required_split(manifest_schema: str, required_split: str | None) -> str:
    schema_split = SCHEMA_REQUIRED_SPLITS.get(manifest_schema)
    if schema_split is not None and required_split not in (None, schema_split):
        raise ValueError(
            f"schema {manifest_schema} requires split={schema_split}, got {required_split}"
        )
    return required_split or schema_split or "train"


def _resolve_output_schema(manifest_schema: str, output_schema: str | None) -> str:
    schema_output = SCHEMA_REQUIRED_OUTPUTS.get(manifest_schema)
    if schema_output is not None and output_schema not in (None, schema_output):
        raise ValueError(
            f"manifest schema {manifest_schema} requires output schema {schema_output}, "
            f"got {output_schema}"
        )
    return output_schema or schema_output or OUTPUT_SCHEMA


def _normalized_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _command_report_path(command: Any) -> Path | None:
    if not isinstance(command, list):
        return None
    for index, token in enumerate(map(str, command)):
        for option in ("--report-json", "--out-json", "--export-report-json"):
            if token == option and index + 1 < len(command):
                return _normalized_path(command[index + 1])
            if token.startswith(option + "="):
                return _normalized_path(token.split("=", 1)[1])
    return None


def _is_bound_repair_path(path: Any, source: Mapping[str, Any], *, kind: str) -> bool:
    if source.get("dispatch_key") != "tvm_auto" or (source.get("q_mode") or source.get("q")) != "int8":
        return False
    item = _normalized_path(path)
    parts = item.parts
    width = "x".join(map(str, source.get("width") or []))
    needle = ("tvm_int8_repair", str(source.get("model") or ""), width)
    positions = [
        index for index in range(len(parts) - len(needle) + 1)
        if tuple(parts[index : index + len(needle)]) == needle
    ]
    if not positions:
        return False
    tail = parts[positions[-1] + len(needle) :]
    if kind == "performance":
        return (
            len(tail) >= 3
            and tail[0].startswith("build")
            and item.name == "route_b_int8_auto_decomp_result.json"
        )
    if kind == "ap":
        return (
            len(tail) == 2
            and tail[0].startswith("ap_full")
            and item.name == "full_ap_eval_report.json"
        )
    return False


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_schema: str = MANIFEST_SCHEMA,
    required_split: str | None = None,
) -> list[Mapping[str, Any]]:
    required_split = _resolve_required_split(manifest_schema, required_split)
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != manifest_schema:
        raise ValueError(f"expected {manifest_schema}")
    if not isinstance(jobs, list) or len(jobs) != 16:
        raise ValueError("targeted supplement must contain exactly 16 jobs")
    if any(not isinstance(job, Mapping) for job in jobs):
        raise ValueError("targeted supplement jobs must be objects")
    invalid_models = sorted(
        {str(job.get("model") or "") for job in jobs} - {"codriving", "pyramid"}
    )
    if invalid_models:
        raise ValueError(
            "manifest jobs must use canonical model values codriving/pyramid: "
            f"{invalid_models}"
        )

    manifest_ids = [str(job.get("job_id") or "") for job in jobs]
    if any(not manifest_id for manifest_id in manifest_ids) or len(set(manifest_ids)) != 16:
        raise ValueError("targeted supplement job_id values must be non-empty and unique")
    if any(job.get("split") != required_split for job in jobs):
        raise ValueError(
            f"all targeted supplement jobs must use split={required_split}"
        )

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for job in jobs:
        group_id = str(job.get("group_id") or "")
        if not group_id:
            raise ValueError("targeted supplement job is missing group_id")
        grouped.setdefault(group_id, []).append(job)
    if len(grouped) != 4:
        raise ValueError("targeted supplement must contain exactly 4 groups")
    for group_id, group_jobs in grouped.items():
        arms = {
            (str(job.get("dispatch_key") or ""), str(job.get("q_mode") or job.get("q") or ""))
            for job in group_jobs
        }
        if len(group_jobs) != 4 or arms != ARMS:
            raise ValueError(f"targeted supplement group {group_id} is not a complete four-arm group")
    return jobs


def _validate_final_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    invalid_status = [
        str(row.get("manifest_job_id"))
        for row in rows
        if row.get("terminal_status") != "measured_success_gold"
    ]
    if invalid_status:
        raise ValueError(
            "targeted supplement contains non-final rows; all rows must be "
            f"measured_success_gold: {invalid_status}"
        )
    for row in rows:
        manifest_id = str(row.get("manifest_job_id"))
        if not all(_finite(row.get(field)) for field in METRIC_FIELDS):
            raise ValueError(f"targeted row lacks finite latency/energy/AP metrics: {manifest_id}")
        if not all(
            isinstance(row.get(field), str) and SHA256_PATTERN.fullmatch(str(row[field]))
            for field in HASH_FIELDS
        ):
            raise ValueError(f"targeted row lacks valid evidence SHA256 values: {manifest_id}")


def _validate_performance_bindings(
    jobs: Sequence[Mapping[str, Any]],
    *,
    performance_job_rows: Sequence[Mapping[str, Any]],
    ap_plan_rows: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {str(job["job_id"]): job for job in jobs}
    performance_by_manifest: dict[str, Mapping[str, Any]] = {}
    performance_by_id: dict[str, Mapping[str, Any]] = {}
    for performance_job in performance_job_rows:
        if not isinstance(performance_job, Mapping):
            raise ValueError("performance jobs must be objects")
        performance_id = str(performance_job.get("job_id") or "")
        manifest_id = str(performance_job.get("manifest_job_id") or "")
        if manifest_id not in manifest_by_id or not performance_id:
            raise ValueError("performance job mapping must target a manifest row")
        if manifest_id in performance_by_manifest or performance_id in performance_by_id:
            raise ValueError("expected exactly one unique performance job per manifest row")
        source = manifest_by_id[manifest_id]
        runner = PERFORMANCE_RUNNERS[
            (str(source.get("dispatch_key") or ""), str(source.get("q_mode") or ""))
        ]
        if (
            str(performance_job.get("runner_key") or "") != runner
            or str(performance_job.get("model") or "") != str(source.get("model") or "")
        ):
            raise ValueError(f"performance job runner binding mismatch: {manifest_id}")
        performance_by_manifest[manifest_id] = performance_job
        performance_by_id[performance_id] = performance_job
    if set(performance_by_manifest) != set(manifest_by_id):
        raise ValueError("expected exactly one performance job per targeted manifest row")

    plan_by_manifest = {
        str(row.get("manifest_job_id") or ""): row for row in ap_plan_rows
    }
    for manifest_id, source in manifest_by_id.items():
        plan = plan_by_manifest.get(manifest_id)
        performance_job = performance_by_manifest[manifest_id]
        if plan is None or str(plan.get("performance_job_id") or "") != str(
            performance_job["job_id"]
        ):
            raise ValueError(f"AP performance job binding mismatch: {manifest_id}")
        expected_ap_runner = AP_RUNNERS[
            (str(source.get("model") or ""), str(performance_job["runner_key"]))
        ]
        if str(plan.get("runner_key") or "") != expected_ap_runner:
            raise ValueError(f"AP runner binding mismatch: {manifest_id}")

    normalized_states = []
    for state in performance_state_rows:
        if not isinstance(state, Mapping):
            raise ValueError("performance state rows must be objects")
        state_job_id = str(state.get("job_id") or "")
        explicit_manifest_id = str(state.get("manifest_job_id") or "")
        mapped_job = performance_by_id.get(state_job_id)
        mapped_manifest_id = (
            str(mapped_job["manifest_job_id"]) if mapped_job is not None else ""
        )
        if explicit_manifest_id:
            if explicit_manifest_id == state_job_id and mapped_manifest_id:
                # The scale-aware repair importer historically stores the
                # performance job ID in both ID fields. The performance plan
                # remains the authoritative manifest binding.
                manifest_id = mapped_manifest_id
            elif explicit_manifest_id in manifest_by_id and (
                not mapped_manifest_id or mapped_manifest_id == explicit_manifest_id
            ):
                manifest_id = explicit_manifest_id
            else:
                raise ValueError("performance state binding conflicts with manifest mapping")
        elif mapped_manifest_id:
            manifest_id = mapped_manifest_id
        else:
            raise ValueError("performance state binding requires manifest_job_id or job mapping")
        expected_job = performance_by_manifest[manifest_id]
        source = manifest_by_id[manifest_id]
        state_runner = state.get("runner_key")
        if state_runner is not None and str(state_runner) != str(expected_job["runner_key"]):
            raise ValueError(f"performance state runner binding mismatch: {manifest_id}")
        result_json = state.get("result_json")
        if state.get("status") == "success" and result_json:
            if state.get("source") == "stage3_tvm_int8_repair_v3":
                valid_path = _is_bound_repair_path(result_json, source, kind="performance")
            else:
                expected_path = plan_by_manifest[manifest_id].get("performance_result_json")
                valid_path = bool(expected_path) and _normalized_path(result_json) == _normalized_path(expected_path)
            if not valid_path:
                raise ValueError(f"performance evidence path binding mismatch: {manifest_id}")
        normalized_states.append({
            **dict(state),
            "manifest_job_id": str(expected_job["job_id"]),
        })
    return normalized_states


def _validate_ap_bindings(
    jobs: Sequence[Mapping[str, Any]],
    *,
    ap_plan_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {str(job["job_id"]): job for job in jobs}
    plan_by_id = {str(row.get("manifest_job_id") or ""): row for row in ap_plan_rows}
    normalized: list[dict[str, Any]] = []
    for state in ap_state_rows:
        if not isinstance(state, Mapping):
            raise ValueError("AP state rows must be objects")
        manifest_id = str(state.get("manifest_job_id") or state.get("job_id") or "")
        source = manifest_by_id.get(manifest_id)
        if source is None:
            raise ValueError(f"AP state binding targets unknown manifest row: {manifest_id}")
        report_path = state.get("report_path")
        stage = str(state.get("stage") or "")
        if report_path and stage in {"sanity", "full"}:
            if state.get("source") == "stage3_tvm_int8_repair_v3":
                valid_path = stage == "full" and _is_bound_repair_path(
                    report_path, source, kind="ap"
                )
            else:
                expected_path = _command_report_path(plan_by_id[manifest_id].get(f"{stage}_command"))
                valid_path = expected_path is not None and _normalized_path(report_path) == expected_path
            if not valid_path:
                raise ValueError(f"AP evidence path binding mismatch: {manifest_id}")
        normalized.append(dict(state))
    return normalized


def finalize_targeted16(
    manifest: Mapping[str, Any],
    *,
    performance_job_rows: Sequence[Mapping[str, Any]],
    ap_plan_rows: Sequence[Mapping[str, Any]],
    performance_state_rows: Sequence[Mapping[str, Any]],
    ap_state_rows: Sequence[Mapping[str, Any]],
    manifest_schema: str = MANIFEST_SCHEMA,
    output_schema: str | None = None,
    required_split: str | None = None,
) -> dict[str, Any]:
    required_split = _resolve_required_split(manifest_schema, required_split)
    output_schema = _resolve_output_schema(manifest_schema, output_schema)
    jobs = _validate_manifest(
        manifest,
        manifest_schema=manifest_schema,
        required_split=required_split,
    )
    normalized_performance_states = _validate_performance_bindings(
        jobs,
        performance_job_rows=performance_job_rows,
        ap_plan_rows=ap_plan_rows,
        performance_state_rows=performance_state_rows,
    )
    normalized_ap_states = _validate_ap_bindings(
        jobs,
        ap_plan_rows=ap_plan_rows,
        ap_state_rows=ap_state_rows,
    )
    finalized = finalize_gold_dataset(
        manifest,
        ap_plan_rows=ap_plan_rows,
        performance_state_rows=normalized_performance_states,
        ap_state_rows=normalized_ap_states,
        manifest_schema=manifest_schema,
        output_schema=output_schema,
        expected_rows=16,
        expected_groups=4,
    )
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    rows = [
        {
            **dict(row),
            "split": required_split,
            "width_stratum": str(
                jobs_by_id[str(row["manifest_job_id"])].get("width_stratum") or "targeted"
            ),
        }
        for row in finalized["rows"]
    ]
    _validate_final_rows(rows)
    return {**finalized, "rows": rows, "manifest": manifest}


def _csv_value(value: Any) -> Any:
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value


def write_targeted16_outputs(
    result: Mapping[str, Any],
    output_dir: str | Path,
    *,
    file_prefix: str = "targeted16",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in result["rows"]]
    (output / f"{file_prefix}_final.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / f"{file_prefix}_final.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f"{file_prefix}_final.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(row.get(key)) for key in fieldnames}
            for row in rows
        )
    (output / f"{file_prefix}_manifest.json").write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_schema = (
        str(rows[0].get("schema_version")) if rows else OUTPUT_SCHEMA
    )
    audit = {
        "schema_version": output_schema,
        "groups": result["group_audit"],
        "summary": result["summary"],
    }
    (output / f"{file_prefix}_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--performance-jobs-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--performance-state-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--ap-state-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-schema", default=MANIFEST_SCHEMA)
    parser.add_argument("--output-schema")
    parser.add_argument("--required-split")
    parser.add_argument("--file-prefix", default="targeted16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    result = finalize_targeted16(
        manifest,
        performance_job_rows=[
            row for path in args.performance_jobs_jsonl for row in read_jsonl(path)
        ],
        ap_plan_rows=read_jsonl(args.ap_plan_jsonl),
        performance_state_rows=[
            row for path in args.performance_state_jsonl for row in read_jsonl(path)
        ],
        ap_state_rows=[row for path in args.ap_state_jsonl for row in read_jsonl(path)],
        manifest_schema=args.manifest_schema,
        output_schema=args.output_schema,
        required_split=args.required_split,
    )
    write_targeted16_outputs(result, args.output_dir, file_prefix=args.file_prefix)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
