#!/usr/bin/env python3
"""Run an independent structural and evidence audit for Gold176."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
ROW_SCHEMA = "stage35_gold176_final_v1"
MANIFEST_SCHEMA = "stage35_gold176_manifest_v1"
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
    ("pyramid", "trt_fp16"): "pyramid_trt_multiscale",
    ("pyramid", "trt_int8"): "pyramid_trt_multiscale",
    ("pyramid", "tvm_fp16"): "pyramid_tvm_fp16_bridge",
    ("pyramid", "tvm_int8"): "pyramid_tvm_int8_numeric_gate",
    ("codriving", "trt_fp16"): "codriving_trt_multiscale",
    ("codriving", "trt_int8"): "codriving_trt_multiscale",
    ("codriving", "tvm_fp16"): "codriving_tvm_fp16_bridge",
    ("codriving", "tvm_int8"): "codriving_tvm_int8_numeric_gate",
}
PERFORMANCE_SCRIPTS = {
    "trt_fp16": "framework/trt_baseline/trt_profile_v1.py",
    "trt_int8": "framework/trt_baseline/trt_profile_v1.py",
    "tvm_fp16": "scripts/stage2_route_b_fp16_auto_runner.py",
    "tvm_int8": "scripts/stage2_route_b_int8_auto_decomp.py",
}
AP_SCRIPTS = {
    "pyramid_trt_multiscale": "scripts/stage3_trt_multiscale_ap_bridge_v3.py",
    "pyramid_tvm_fp16_bridge": "scripts/stage2_h800_fp16_rewritten_activation_bridge.py",
    "pyramid_tvm_int8_numeric_gate": "scripts/stage3_pyramid_tvm_int8_ap_numeric_gate_v3.py",
    "codriving_trt_multiscale": "scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py",
    "codriving_tvm_fp16_bridge": "scripts/stage3_codriving_tvm_fp16_ap_bridge_v3.py",
    "codriving_tvm_int8_numeric_gate": "scripts/stage3_codriving_tvm_int8_ap_numeric_gate_v3.py",
}
METRICS = ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
HASH_BINDINGS = (
    ("performance_result_json", "performance_result_sha256"),
    ("ap_report_path", "ap_report_sha256"),
)
LOCKED_FIELDS = ("manifest_job_id", *METRICS, "performance_result_sha256", "ap_report_sha256")
SOURCE_EVIDENCE_FIELDS = (
    "manifest_job_id", "terminal_status", *METRICS,
    "performance_result_json", "performance_result_sha256",
    "ap_report_path", "ap_report_sha256", "failure_reason",
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_evidence_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else REPO_ROOT / path


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in LOCKED_FIELDS)


def _source_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in SOURCE_EVIDENCE_FIELDS)


def _command_option(command: Sequence[Any], option: str) -> str | None:
    values = [str(value) for value in command]
    try:
        return values[values.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _report_path(command: Sequence[Any]) -> Path:
    for option in ("--report-json", "--out-json", "--export-report-json"):
        value = _command_option(command, option)
        if value:
            return _resolve_evidence_path(value).resolve()
    raise ValueError("full AP command lacks an output report path")


def audit_gold176(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    gold144_rows: Sequence[Mapping[str, Any]],
    targeted32_rows: Sequence[Mapping[str, Any]],
    performance_job_rows: Sequence[Mapping[str, Any]],
    ap_plan_rows: Sequence[Mapping[str, Any]],
    verify_evidence_files: bool,
) -> dict[str, Any]:
    jobs = manifest.get("jobs")
    if manifest.get("schema_version") != MANIFEST_SCHEMA or not isinstance(jobs, list):
        raise ValueError(f"expected {MANIFEST_SCHEMA}")
    if len(rows) != 176 or len(jobs) != 176:
        raise ValueError("Gold176 must contain exactly 176 rows and jobs")
    if any(row.get("schema_version") != ROW_SCHEMA for row in rows):
        raise ValueError(f"all rows must use {ROW_SCHEMA}")

    row_ids = [str(row.get("manifest_job_id") or "") for row in rows]
    job_ids = [str(job.get("job_id") or "") for job in jobs]
    if len(set(row_ids)) != 176 or len(set(job_ids)) != 176 or set(row_ids) != set(job_ids):
        raise ValueError("row and manifest identities must be unique and identical")
    jobs_by_id = {str(job["job_id"]): job for job in jobs}
    binding_fields = ("group_id", "model", "width", "q_mode", "dispatch_key", "capability_profile_id", "split")
    for row in rows:
        job = jobs_by_id[str(row["manifest_job_id"])]
        mismatch = [field for field in binding_fields if row.get(field) != job.get(field)]
        if mismatch:
            raise ValueError(f"row/manifest binding mismatch: {row['manifest_job_id']}:{mismatch}")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["group_id"]), []).append(row)
    if len(grouped) != 44:
        raise ValueError("Gold176 must contain exactly 44 groups")
    for group_id, group_rows in grouped.items():
        arms = {(str(row["dispatch_key"]), str(row["q_mode"])) for row in group_rows}
        if len(group_rows) != 4 or arms != ARMS:
            raise ValueError(f"incomplete four-arm group: {group_id}")
        if len({str(row["split"]) for row in group_rows}) != 1:
            raise ValueError(f"group crosses split boundary: {group_id}")

    split_groups = Counter(str(group_rows[0]["split"]) for group_rows in grouped.values())
    if dict(split_groups) != {"train": 38, "locked_holdout": 6}:
        raise ValueError(f"unexpected grouped split: {dict(split_groups)}")
    statuses = Counter(str(row.get("terminal_status")) for row in rows)
    if dict(statuses) != {"measured_success_gold": 174, "feasibility_failure": 2}:
        raise ValueError(f"unexpected terminal status distribution: {dict(statuses)}")

    measured_rows = [row for row in rows if row["terminal_status"] == "measured_success_gold"]
    for path_field, _ in HASH_BINDINGS:
        paths = [str(row.get(path_field) or "") for row in measured_rows]
        if len(set(paths)) != len(paths):
            raise ValueError(f"duplicate measured evidence path detected: {path_field}")

    evidence_checked = 0
    for row in rows:
        if row["terminal_status"] == "measured_success_gold":
            if not all(_finite(row.get(field)) for field in METRICS):
                raise ValueError(f"non-finite measured metric: {row['manifest_job_id']}")
            for path_field, hash_field in HASH_BINDINGS:
                digest = str(row.get(hash_field) or "")
                if not SHA_PATTERN.fullmatch(digest):
                    raise ValueError(f"invalid evidence SHA: {row['manifest_job_id']}:{hash_field}")
                if verify_evidence_files:
                    path = _resolve_evidence_path(row.get(path_field))
                    if not path.is_file() or sha256_file(path) != digest:
                        raise ValueError(f"evidence file/SHA mismatch: {row['manifest_job_id']}:{path}")
                    evidence_checked += 1
        elif any(row.get(field) is not None for field in (*METRICS, "performance_result_sha256", "ap_report_sha256")):
            raise ValueError(f"feasibility failure contains fabricated metrics: {row['manifest_job_id']}")

    base_locked = {
        str(row["manifest_job_id"]): _identity(row)
        for row in gold144_rows
        if row.get("split") == "locked_holdout"
    }
    final_locked = {
        str(row["manifest_job_id"]): _identity(row)
        for row in rows
        if row.get("split") == "locked_holdout"
    }
    if len(base_locked) != 24 or final_locked != base_locked:
        raise ValueError("Gold144 locked holdout evidence changed")

    base_all = {str(row["manifest_job_id"]): _source_identity(row) for row in gold144_rows}
    final_base = {
        str(row["manifest_job_id"]): _source_identity(row)
        for row in rows if row.get("source_pool") == "gold144"
    }
    if len(base_all) != 144 or final_base != base_all:
        raise ValueError("Gold144 source evidence changed during merge")

    final_by_id = {str(row["manifest_job_id"]): row for row in rows}
    if len(targeted32_rows) != 32:
        raise ValueError("targeted32 must contain exactly 32 rows")
    compare_fields = (
        "terminal_status", *METRICS,
        "performance_result_json", "performance_result_sha256",
        "ap_report_path", "ap_report_sha256",
    )
    for source in targeted32_rows:
        final = final_by_id.get(str(source["manifest_job_id"]))
        if final is None or any(final.get(field) != source.get(field) for field in compare_fields):
            raise ValueError(f"targeted32 evidence changed during merge: {source['manifest_job_id']}")

    target_ids = {str(row["manifest_job_id"]) for row in targeted32_rows}
    performance_by_manifest = {
        str(row.get("manifest_job_id") or ""): row for row in performance_job_rows
    }
    plan_by_manifest = {str(row.get("manifest_job_id") or ""): row for row in ap_plan_rows}
    if len(performance_by_manifest) != 32 or set(performance_by_manifest) != target_ids:
        raise ValueError("performance plan must bind exactly once to targeted32")
    if len(plan_by_manifest) != 32 or set(plan_by_manifest) != target_ids:
        raise ValueError("AP plan must bind exactly once to targeted32")
    manifest_by_id = {str(job["job_id"]): job for job in jobs}
    for manifest_id in target_ids:
        row = final_by_id[manifest_id]
        job = manifest_by_id[manifest_id]
        performance_job = performance_by_manifest[manifest_id]
        plan = plan_by_manifest[manifest_id]
        expected_performance_runner = PERFORMANCE_RUNNERS[(str(job["dispatch_key"]), str(job["q_mode"]))]
        expected_width_key = "x".join(str(value) for value in job["width"])
        performance_identity = {
            "group_id": performance_job.get("group_id"),
            "model": performance_job.get("model"),
            "width_key": performance_job.get("width_key"),
            "dispatch_key": performance_job.get("dispatch_key"),
            "q_mode": performance_job.get("q_mode"),
            "split": performance_job.get("split"),
            "runner_key": performance_job.get("runner_key"),
        }
        expected_performance_identity = {
            "group_id": job.get("group_id"),
            "model": job.get("model"),
            "width_key": expected_width_key,
            "dispatch_key": job.get("dispatch_key"),
            "q_mode": job.get("q_mode"),
            "split": job.get("split"),
            "runner_key": expected_performance_runner,
        }
        if performance_identity != expected_performance_identity or performance_job.get("source_contract") != job.get("source_contract"):
            raise ValueError(f"performance arm identity mismatch: {manifest_id}")
        expected_ap_runner = AP_RUNNERS[(str(job["model"]), expected_performance_runner)]
        plan_identity = {
            "model": plan.get("model"),
            "width": list(plan.get("width") or []),
            "q": plan.get("q"),
            "profile": plan.get("profile"),
            "runner_key": plan.get("runner_key"),
            "source_contract": plan.get("source_contract"),
        }
        expected_plan_identity = {
            "model": job.get("model"),
            "width": list(job.get("width") or []),
            "q": job.get("q_mode"),
            "profile": job.get("capability_profile_id"),
            "runner_key": expected_ap_runner,
            "source_contract": job.get("source_contract"),
        }
        if plan_identity != expected_plan_identity:
            raise ValueError(f"AP arm identity mismatch: {manifest_id}")
        if str(plan.get("performance_job_id")) != str(performance_job.get("job_id")):
            raise ValueError(f"AP/performance plan binding mismatch: {manifest_id}")
        command = list(performance_job.get("command") or [])
        expected_onnx = str((job.get("source_contract") or {}).get("onnx_path") or "")
        if len(command) < 2 or str(command[1]) != PERFORMANCE_SCRIPTS[expected_performance_runner]:
            raise ValueError(f"performance command runner mismatch: {manifest_id}")
        if _command_option(command, "--onnx") != expected_onnx:
            raise ValueError(f"performance ONNX binding mismatch: {manifest_id}")
        if expected_performance_runner.startswith("trt_"):
            if _command_option(command, "--precision") != str(job["q_mode"]):
                raise ValueError(f"TRT precision command mismatch: {manifest_id}")
            expected_calibration = str((job.get("source_contract") or {}).get("calibration_root") or "")
            if job["q_mode"] == "int8" and _command_option(command, "--calib-dir") != expected_calibration:
                raise ValueError(f"TRT calibration command mismatch: {manifest_id}")
        elif _command_option(command, "--width") != ",".join(str(value) for value in job["width"]):
            raise ValueError(f"TVM width command mismatch: {manifest_id}")
        full_command = list(plan.get("full_command") or [])
        if len(full_command) < 2 or str(full_command[1]) != AP_SCRIPTS[expected_ap_runner]:
            raise ValueError(f"AP command runner mismatch: {manifest_id}")
        if expected_ap_runner == "pyramid_tvm_fp16_bridge":
            if _command_option(full_command, "--artifact-input-dtype") != "float32":
                raise ValueError(f"AP input dtype command mismatch: {manifest_id}")
        elif _command_option(full_command, "--precision-tag") != str(job["q_mode"]):
            raise ValueError(f"AP precision command mismatch: {manifest_id}")
        performance_path = _resolve_evidence_path(row["performance_result_json"]).resolve()
        ap_path = _resolve_evidence_path(row["ap_report_path"]).resolve()
        repaired_row = row.get("dispatch_key") == "tvm_auto" and row.get("q_mode") == "int8"
        width_key = "x".join(str(value) for value in row["width"])
        expected_label = f"{row['model']}_{width_key}"
        performance_root = _resolve_evidence_path(performance_job.get("remote_artifact_root")).resolve()
        supplement_root = performance_root.parent
        expected_performance_dir = performance_root / "batch_01" / str(row["group_id"]) / expected_performance_runner
        if repaired_row:
            expected_repair_performance_suffix = (
                Path("tvm_int8_repair") / str(row["model"]) / width_key / "build"
                / f"{expected_label}_scaleaware" / "route_b_int8_auto_decomp_result.json"
            )
            expected_repair_ap_suffix = (
                Path("tvm_int8_repair") / str(row["model"]) / width_key
                / "ap_full" / "full_ap_eval_report.json"
            )
            if performance_path != supplement_root / expected_repair_performance_suffix:
                raise ValueError(f"repair performance identity mismatch: {manifest_id}")
            if ap_path != supplement_root / expected_repair_ap_suffix:
                raise ValueError(f"repair AP identity mismatch: {manifest_id}")
        elif str(performance_job.get("runner_key")).startswith("trt_"):
            expected_result = _command_option(command, "--out")
            deterministic_result = expected_performance_dir / "trt_profile_result.json"
            if not expected_result or performance_path != deterministic_result or performance_path != _resolve_evidence_path(expected_result).resolve():
                raise ValueError(f"TRT result identity mismatch: {manifest_id}")
        else:
            output_dir = _command_option(command, "--out-dir")
            result_filename = {
                "tvm_fp16": "route_b_fp16_auto_result.json",
                "tvm_int8": "route_b_int8_auto_decomp_result.json",
            }[expected_performance_runner]
            deterministic_result = expected_performance_dir / expected_label / result_filename
            if (
                not output_dir
                or _resolve_evidence_path(output_dir).resolve() != expected_performance_dir
                or _command_option(command, "--label") != expected_label
                or performance_path != deterministic_result
            ):
                raise ValueError(f"TVM result identity mismatch: {manifest_id}")
        if not repaired_row:
            expected_ap_suffix = Path("ap") / str(row["model"]) / width_key / str(row["q_mode"]) / str(row["capability_profile_id"]) / "full_1789" / "full_ap_eval_report.json"
            deterministic_ap_path = supplement_root / "ap_execution" / expected_ap_suffix
            if ap_path != deterministic_ap_path or ap_path != _report_path(full_command):
                raise ValueError(f"AP report identity mismatch: {manifest_id}")
            compiled = str(plan.get("compiled_artifact_path") or plan.get("compiled_artifact") or "")
            artifact_option = "--engine" if expected_performance_runner.startswith("trt_") else (
                "--artifact-path" if expected_ap_runner == "pyramid_tvm_fp16_bridge" else "--compiled-artifact"
            )
            if _command_option(full_command, artifact_option) != compiled:
                raise ValueError(f"AP compiled artifact command mismatch: {manifest_id}")
        if verify_evidence_files:
            performance_payload = json.loads(performance_path.read_text())
            if expected_performance_runner.startswith("trt_"):
                if performance_payload.get("precision") != str(row["q_mode"]) or performance_payload.get("onnx") != Path(expected_onnx).name:
                    raise ValueError(f"TRT result payload identity mismatch: {manifest_id}")
            elif performance_payload.get("onnx_path") != expected_onnx or list(performance_payload.get("width") or []) != list(row["width"]):
                raise ValueError(f"TVM result payload identity mismatch: {manifest_id}")
            if not repaired_row:
                report_payload = json.loads(ap_path.read_text())
                if expected_performance_runner.startswith("trt_") and report_payload.get("engine_path") != str(plan.get("compiled_artifact_path")):
                    raise ValueError(f"TRT AP payload artifact mismatch: {manifest_id}")

    repaired = [
        row for row in rows
        if row.get("source_pool") == "targeted32"
        and row.get("dispatch_key") == "tvm_auto"
        and row.get("q_mode") == "int8"
        and "/tvm_int8_repair/" in str(row.get("performance_result_json") or "")
    ]
    if len(repaired) != 8:
        raise ValueError(f"expected eight repaired targeted TVM INT8 rows, found {len(repaired)}")

    return {
        "schema_version": "stage35_gold176_integrity_audit_v1",
        "qualified": True,
        "rows": len(rows),
        "groups": len(grouped),
        "split_groups": dict(split_groups),
        "terminal_status_rows": dict(statuses),
        "locked_holdout_groups": len(base_locked) // 4,
        "targeted32_rows": len(targeted32_rows),
        "targeted32_repaired_tvm_int8_rows": len(repaired),
        "measured_rows_with_zero_ap70": sum(
            row.get("terminal_status") == "measured_success_gold" and float(row["ap70"]) == 0.0
            for row in rows
        ),
        "evidence_files_sha256_verified": evidence_checked,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold176-json", type=Path, required=True)
    parser.add_argument("--manifest176-json", type=Path, required=True)
    parser.add_argument("--gold144-json", type=Path, required=True)
    parser.add_argument("--targeted32-json", type=Path, required=True)
    parser.add_argument("--performance-jobs-jsonl", type=Path, required=True)
    parser.add_argument("--ap-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--skip-evidence-files", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_gold176(
        json.loads(args.gold176_json.read_text()),
        json.loads(args.manifest176_json.read_text()),
        gold144_rows=json.loads(args.gold144_json.read_text()),
        targeted32_rows=json.loads(args.targeted32_json.read_text()),
        performance_job_rows=[json.loads(line) for line in args.performance_jobs_jsonl.read_text().splitlines() if line.strip()],
        ap_plan_rows=[json.loads(line) for line in args.ap_plan_jsonl.read_text().splitlines() if line.strip()],
        verify_evidence_files=not args.skip_evidence_files,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
