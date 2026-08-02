#!/usr/bin/env python3
"""Build and validate the v2 gold cold-start 96-point measurement plan.

The search genome is only ``[w0, w1, w2] + q_mode + mixed_policy_id``.
Compiler/runtime choices remain measurement context through
``compiler_profile_id`` and must not leak into ``strategy_id``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from stage2_codriving_int8_provenance import report_has_valid_int8_calibration

import yaml


REPO = Path("/home/jichengzhi/V2X")
DEFAULT_OUT_DIR = REPO / "results/v2_gold_coldstart_96_20260708"
DEFAULT_PLAN_JSON = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_launch_manifest.json"
DEFAULT_PLAN_CSV = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_launch_manifest.csv"
DEFAULT_AUDIT_JSON = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_missing_failure_audit.json"
DEFAULT_AUDIT_CSV = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_missing_failure_audit.csv"
DEFAULT_TRAINING_JSON = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_training_table.json"
DEFAULT_TRAINING_CSV = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_training_table.csv"
DEFAULT_PARTIAL_JSON = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_partial_training_rows.json"
DEFAULT_PARTIAL_CSV = DEFAULT_OUT_DIR / "v2_gold_coldstart_96_partial_training_rows.csv"

REQUIRED_METRICS = ("latency_ms", "energy_j", "ap70")
JOB_ID_KEYS = ("width", "compiler_profile_id", "q_mode", "mixed_policy_id")
FORBIDDEN_STRATEGY_TOKENS = ("tvm", "trt", "cutlass", "backend")

DEFAULT_WIDTHS = (
    "16x32x64",
    "24x32x96",
    "32x32x128",
    "32x64x128",
    "48x96x192",
    "56x112x224",
    "64x96x192",
    "64x128x256",
    "24x64x128",
    "40x64x128",
    "48x64x128",
    "64x64x128",
)

DEFAULT_STRATEGY_COMBINATIONS = (
    {
        "combo_id": "tvm_fp32_baseline",
        "compiler_profile_id": "h800_tvm_fp32_baseline_20260708",
        "backend_context": "tvm",
        "q_mode": "fp32",
        "mixed_policy_id": "none",
        "realized_schedule_route": "tvm_fp32_baseline_default_or_metaschedule",
        "runner_kind": "tvm_fp32_baseline",
    },
    {
        "combo_id": "tvm_routeb_fp16",
        "compiler_profile_id": "h800_tvm_routeb_20260708",
        "backend_context": "tvm",
        "q_mode": "fp16",
        "mixed_policy_id": "none",
        "realized_schedule_route": "routeb_fp16_auto_tensorcore",
        "runner_kind": "route_b_fp16_auto",
    },
    {
        "combo_id": "tvm_routeb_int8_all",
        "compiler_profile_id": "h800_tvm_routeb_20260708",
        "backend_context": "tvm",
        "q_mode": "int8",
        "mixed_policy_id": "all_eligible_conv",
        "realized_schedule_route": "routeb_int8_auto_decomp_tensorcore",
        "runner_kind": "route_b_int8_auto_decomp",
    },
    {
        "combo_id": "tvm_routeb_int8_top25",
        "compiler_profile_id": "h800_tvm_routeb_20260708",
        "backend_context": "tvm",
        "q_mode": "int8",
        "mixed_policy_id": "top25_flops",
        "realized_schedule_route": "routeb_mixed_top25_flops_tensorcore",
        "runner_kind": "codriving_whole_engine_tc_mixed",
    },
    {
        "combo_id": "tvm_routeb_int8_top50",
        "compiler_profile_id": "h800_tvm_routeb_20260708",
        "backend_context": "tvm",
        "q_mode": "int8",
        "mixed_policy_id": "top50_flops",
        "realized_schedule_route": "routeb_mixed_top50_flops_tensorcore",
        "runner_kind": "codriving_whole_engine_tc_mixed",
    },
    {
        "combo_id": "trt_fp16",
        "compiler_profile_id": "h800_trt_20260708",
        "backend_context": "tensorrt",
        "q_mode": "fp16",
        "mixed_policy_id": "none",
        "realized_schedule_route": "trt_fp16_engine",
        "runner_kind": "trt_engine_fp16",
    },
    {
        "combo_id": "trt_int8_all",
        "compiler_profile_id": "h800_trt_20260708",
        "backend_context": "tensorrt",
        "q_mode": "int8",
        "mixed_policy_id": "all_eligible_conv",
        "realized_schedule_route": "trt_int8_engine",
        "runner_kind": "trt_engine_int8",
    },
    {
        "combo_id": "cutlass_fp16",
        "compiler_profile_id": "h800_tvm_cutlass_byoc_20260708",
        "backend_context": "tvm_cutlass_byoc",
        "q_mode": "fp16",
        "mixed_policy_id": "none",
        "realized_schedule_route": "cutlass_byoc_fp16_fullengine",
        "runner_kind": "cutlass_byoc_fp16",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_width(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        parts = [int(item) for item in value]
    elif isinstance(value, str):
        parts = [int(item.strip()) for item in value.replace("x", ",").split(",") if item.strip()]
    else:
        raise TypeError(f"unsupported width value: {value!r}")
    if len(parts) != 3:
        raise ValueError(f"width must have 3 parts, got {value!r}")
    return parts[0], parts[1], parts[2]


def width_str(width: Any) -> str:
    w0, w1, w2 = parse_width(width)
    return f"{w0}x{w1}x{w2}"


def strategy_id(q_mode: str, mixed_policy_id: str) -> str:
    q = str(q_mode).lower()
    mixed = str(mixed_policy_id).lower()
    if q in {"fp32", "fp16"}:
        mixed = "none"
    return f"q={q}|mixed={mixed}"


def job_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in JOB_ID_KEYS)


def job_id(width: str, combo: dict[str, Any]) -> str:
    safe_width = "x".join(f"{part:03d}" for part in parse_width(width))
    raw = "__".join(
        [
            "v2gold",
            safe_width,
            str(combo["compiler_profile_id"]),
            str(combo["q_mode"]),
            str(combo["mixed_policy_id"]),
        ]
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _copy_combo(combo: dict[str, Any]) -> dict[str, Any]:
    return {
        "combo_id": str(combo["combo_id"]),
        "compiler_profile_id": str(combo["compiler_profile_id"]),
        "backend_context": str(combo["backend_context"]),
        "q_mode": str(combo["q_mode"]).lower(),
        "mixed_policy_id": str(combo["mixed_policy_id"]).lower(),
        "realized_schedule_route": str(combo["realized_schedule_route"]),
        "runner_kind": str(combo["runner_kind"]),
    }


def build_plan(
    widths: Iterable[Any] = DEFAULT_WIDTHS,
    strategy_combinations: Iterable[dict[str, Any]] = DEFAULT_STRATEGY_COMBINATIONS,
) -> dict[str, Any]:
    width_values = [width_str(width) for width in widths]
    combos = [_copy_combo(combo) for combo in strategy_combinations]
    jobs: list[dict[str, Any]] = []

    for width in width_values:
        w0, w1, w2 = parse_width(width)
        for combo in combos:
            sid = strategy_id(combo["q_mode"], combo["mixed_policy_id"])
            jobs.append(
                {
                    "job_id": job_id(width, combo),
                    "width": width,
                    "w0": w0,
                    "w1": w1,
                    "w2": w2,
                    "combo_id": combo["combo_id"],
                    "compiler_profile_id": combo["compiler_profile_id"],
                    "backend_context": combo["backend_context"],
                    "q_mode": combo["q_mode"],
                    "mixed_policy_id": combo["mixed_policy_id"],
                    "strategy_id": sid,
                    "genome": [w0, w1, w2, combo["q_mode"], combo["mixed_policy_id"]],
                    "realized_schedule_route": combo["realized_schedule_route"],
                    "runner_kind": combo["runner_kind"],
                    "required_metrics": list(REQUIRED_METRICS),
                    "trusted_for_final_frontier": True,
                    "build_status": "pending",
                    "run_status": "pending",
                    "source_file": None,
                }
            )

    return {
        "schema_version": "v2_gold_coldstart_96_launch_manifest_v1",
        "created_at_utc": utc_now(),
        "measurement_contract": {
            "gold_rows_only": True,
            "required_metrics": list(REQUIRED_METRICS),
            "stop_condition": "96/96 rows complete with latency_ms, energy_j, and ap70",
            "historical_hand_rewrite_prior_allowed": False,
            "strategy_id_must_exclude_backend": True,
            "genome_schema": ["w0", "w1", "w2", "q_mode", "mixed_policy_id"],
        },
        "widths": width_values,
        "strategy_combinations": combos,
        "jobs": jobs,
    }


def _metric_missing(row: dict[str, Any], metric: str) -> bool:
    value = row.get(metric)
    if value in (None, ""):
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return True
    return False


def _is_success(value: Any) -> bool:
    return str(value).lower() in {"success", "ok", "pass", "passed", "true"}


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1"}
    return bool(value)


def _row_reasons(expected_job: dict[str, Any] | None, row: dict[str, Any], required_metrics: Iterable[str]) -> list[str]:
    reasons: list[str] = []
    for metric in required_metrics:
        if _metric_missing(row, metric):
            reasons.append(f"missing_metric:{metric}")

    sid = str(row.get("strategy_id", ""))
    if any(token in sid.lower() for token in FORBIDDEN_STRATEGY_TOKENS):
        reasons.append("backend_token_in_strategy_id")

    if _truthy(row.get("historical_prior_flag")):
        reasons.append("historical_prior_in_gold")

    if row.get("trusted_for_final_frontier") is not True:
        reasons.append("not_trusted_for_final_frontier")

    if not row.get("source_file"):
        reasons.append("missing_source_file")

    if not _metric_missing(row, "ap70") and not row.get("ap_source_file"):
        reasons.append("missing_ap_source_file")

    if not _is_success(row.get("build_status")):
        reasons.append("build_not_success")

    if not _is_success(row.get("run_status")):
        reasons.append("run_not_success")

    expected_sid = strategy_id(str(row.get("q_mode", "")), str(row.get("mixed_policy_id", "")))
    if sid != expected_sid:
        reasons.append("strategy_id_mismatch")

    if expected_job is None:
        reasons.append("unexpected_row")
    else:
        if str(row.get("compiler_profile_id")) != str(expected_job["compiler_profile_id"]):
            reasons.append("compiler_profile_id_mismatch")
        if str(row.get("realized_schedule_route", expected_job["realized_schedule_route"])) != str(
            expected_job["realized_schedule_route"]
        ):
            reasons.append("realized_schedule_route_mismatch")

    return reasons


def validate_training_rows(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_metrics = list(plan.get("measurement_contract", {}).get("required_metrics", REQUIRED_METRICS))
    expected_by_key = {job_key(job): job for job in plan["jobs"]}
    seen_keys: set[tuple[str, ...]] = set()
    invalid_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    complete_rows = 0

    for row in rows:
        key = job_key(row)
        seen_keys.add(key)
        expected_job = expected_by_key.get(key)
        reasons = _row_reasons(expected_job, row, required_metrics)
        if reasons:
            issue = {
                "job_key": dict(zip(JOB_ID_KEYS, key, strict=True)),
                "width": row.get("width"),
                "compiler_profile_id": row.get("compiler_profile_id"),
                "q_mode": row.get("q_mode"),
                "mixed_policy_id": row.get("mixed_policy_id"),
                "strategy_id": row.get("strategy_id"),
                "reasons": reasons,
            }
            invalid_rows.append(issue)
            if "build_not_success" in reasons or "run_not_success" in reasons:
                failed_rows.append(issue)
        else:
            complete_rows += 1

    missing_rows = []
    for key, job in sorted(expected_by_key.items()):
        if key not in seen_keys:
            missing_rows.append(
                {
                    "job_key": dict(zip(JOB_ID_KEYS, key, strict=True)),
                    "job_id": job["job_id"],
                    "width": job["width"],
                    "compiler_profile_id": job["compiler_profile_id"],
                    "q_mode": job["q_mode"],
                    "mixed_policy_id": job["mixed_policy_id"],
                    "strategy_id": job["strategy_id"],
                    "required_metrics": list(required_metrics),
                }
            )

    complete = (
        complete_rows == len(expected_by_key)
        and not missing_rows
        and not invalid_rows
        and not failed_rows
        and len(rows) == len(expected_by_key)
    )
    return {
        "schema_version": "v2_gold_coldstart_96_audit_v1",
        "created_at_utc": utc_now(),
        "complete": complete,
        "summary": {
            "expected_rows": len(expected_by_key),
            "observed_rows": len(rows),
            "complete_rows": complete_rows,
            "missing_rows": len(missing_rows),
            "invalid_rows": len(invalid_rows),
            "failed_rows": len(failed_rows),
        },
        "missing_rows": missing_rows,
        "invalid_rows": invalid_rows,
        "failed_rows": failed_rows,
    }


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _result_by_precision(payload: dict[str, Any], precision: str) -> dict[str, Any] | None:
    for item in payload.get("results", []):
        if str(item.get("precision", "")).lower() == precision:
            return item
    return None


def _base_row(job: dict[str, Any], source_file: Path) -> dict[str, Any]:
    return {
        **{key: job[key] for key in JOB_ID_KEYS},
        "combo_id": job["combo_id"],
        "backend_context": job["backend_context"],
        "strategy_id": job["strategy_id"],
        "realized_schedule_route": job["realized_schedule_route"],
        "latency_ms": None,
        "energy_j": None,
        "ap70": None,
        "trusted_for_final_frontier": True,
        "source_file": str(source_file),
        "build_status": "success",
        "run_status": "success",
        "historical_prior_flag": False,
    }


def _row_from_trt(job: dict[str, Any], results_root: Path) -> dict[str, Any] | None:
    precision = "fp16" if job["q_mode"] == "fp16" else "int8"
    path = results_root / "codriving_trt_qxs8_20260708" / f"codriving_trt_{job['width']}_{precision}_20260708.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    row = _base_row(job, path)
    row["latency_ms"] = _float_or_none(payload.get("lat_p50_ms") or payload.get("lat_mean_ms"))
    row["energy_j"] = _float_or_none(payload.get("energy_j"))
    build_success = bool(payload.get("build_success"))
    row["build_status"] = "success" if build_success else "failed"
    row["run_status"] = "success" if row["latency_ms"] is not None else "failed"
    return row


def _row_from_routeb_pure(job: dict[str, Any], results_root: Path) -> dict[str, Any] | None:
    energy_candidates = sorted(
        (results_root / "v2_gold_coldstart_96_20260708" / "tvm_energy_raw" / job["width"]).glob(
            "pure_both_energy_gpu*.json"
        )
    )
    path = energy_candidates[0] if energy_candidates else (
        results_root
        / "codriving_routeb_qxs8_backboneonly_20260708"
        / f"codriving_routeb_{job['width']}_backboneonly_20260708.json"
    )
    if not path.is_file():
        return None
    payload = load_json(path)
    precision = "fp16" if job["q_mode"] == "fp16" else "int8"
    result = _result_by_precision(payload, precision)
    if result is None:
        return None
    row = _base_row(job, path)
    row["latency_ms"] = _float_or_none(result.get("latency_ms_p50") or result.get("latency_ms_min"))
    energy = result.get("energy") if isinstance(result.get("energy"), dict) else {}
    row["energy_j"] = _float_or_none(
        result.get("energy_j") or result.get("joule_per_inference") or energy.get("joule_per_inference")
    )
    row["build_status"] = "success" if result.get("status") == "success" else "failed"
    row["run_status"] = "success" if row["latency_ms"] is not None and result.get("status") == "success" else "failed"
    return row


def _row_from_tvm_fp32_baseline(job: dict[str, Any], results_root: Path) -> dict[str, Any] | None:
    candidates = sorted(
        (results_root / "v2_gold_coldstart_96_20260708" / "tvm_energy_raw" / job["width"]).glob(
            "pure_both_energy_gpu*.json"
        )
    )
    if not candidates:
        return None
    path = candidates[0]
    payload = load_json(path)
    baseline = payload.get("baseline_no_tc") if isinstance(payload.get("baseline_no_tc"), dict) else {}
    if not baseline:
        return None
    row = _base_row(job, path)
    row["latency_ms"] = _float_or_none(baseline.get("latency_ms_p50") or baseline.get("latency_ms_min"))
    energy = baseline.get("energy") if isinstance(baseline.get("energy"), dict) else {}
    row["energy_j"] = _float_or_none(
        baseline.get("energy_j") or baseline.get("joule_per_inference") or energy.get("joule_per_inference")
    )
    row["build_status"] = "success" if baseline.get("status") == "success" else "failed"
    row["run_status"] = "success" if row["latency_ms"] is not None and baseline.get("status") == "success" else "failed"
    return row


def _mixed_globs(width: str, policy: str) -> list[str]:
    if policy == "top25_flops":
        return [f"codriving_routeb_{width}_mixed_top25_flops_gpu*_20260708.json"]
    if policy == "top50_flops":
        return [
            f"codriving_routeb_{width}_mixed_top50_flops_gpu*_20260708.json",
            f"codriving_routeb_{width}_mixed_top50_gpu*_20260708.json",
        ]
    return []


def _row_from_routeb_mixed(job: dict[str, Any], results_root: Path) -> dict[str, Any] | None:
    candidates: list[Path] = []
    energy_dir = results_root / "v2_gold_coldstart_96_20260708" / "tvm_energy_raw" / job["width"]
    if job["mixed_policy_id"] == "top25_flops":
        candidates.extend(sorted(energy_dir.glob("mixed_top25_flops_energy_gpu*.json")))
    elif job["mixed_policy_id"] == "top50_flops":
        candidates.extend(sorted(energy_dir.glob("mixed_top50_flops_energy_gpu*.json")))
    result_dir = results_root / "codriving_routeb_mixed_qxs_probe_20260708"
    for pattern in _mixed_globs(job["width"], job["mixed_policy_id"]):
        candidates.extend(sorted(result_dir.glob(pattern)))
    if not candidates:
        return None
    path = candidates[0]
    payload = load_json(path)
    result = _result_by_precision(payload, "mixed")
    if result is None:
        return None
    row = _base_row(job, path)
    row["latency_ms"] = _float_or_none(result.get("latency_ms_p50") or result.get("latency_ms_min"))
    energy = result.get("energy") if isinstance(result.get("energy"), dict) else {}
    row["energy_j"] = _float_or_none(
        result.get("energy_j") or result.get("joule_per_inference") or energy.get("joule_per_inference")
    )
    row["build_status"] = "success" if result.get("status") == "success" else "failed"
    row["run_status"] = "success" if row["latency_ms"] is not None and result.get("status") == "success" else "failed"
    return row


def _row_from_cutlass_fp16(job: dict[str, Any], results_root: Path) -> dict[str, Any] | None:
    energy_candidates = [
        path
        for path in sorted(
            (results_root / "v2_gold_coldstart_96_20260708" / "cutlass_energy_raw" / job["width"]).glob(
                "cutlass_fp16_energy_gpu*.json"
            )
        )
        if ".failed." not in path.name
    ]
    if energy_candidates:
        path = None
        payload = None
        for candidate in energy_candidates:
            candidate_payload = load_json(candidate)
            if candidate_payload.get("status") == "success":
                path = candidate
                payload = candidate_payload
                break
        if path is None or payload is None:
            path = energy_candidates[0]
            payload = load_json(path)
        row = _base_row(job, path)
        row["latency_ms"] = _float_or_none(payload.get("latency_ms_p50") or payload.get("latency_ms_min"))
        energy = payload.get("energy") if isinstance(payload.get("energy"), dict) else {}
        row["energy_j"] = _float_or_none(
            payload.get("energy_j") or payload.get("joule_per_inference") or energy.get("joule_per_inference")
        )
        row["build_status"] = "success" if payload.get("status") == "success" else "failed"
        row["run_status"] = "success" if row["latency_ms"] is not None and payload.get("status") == "success" else "failed"
        return row

    path = results_root / "codriving_qxs_g4_s5_buildrun_summary_20260708.json"
    if not path.is_file():
        return None
    payload = load_json(path)
    row_payload = None
    for item in payload.get("rows", []):
        if str(item.get("width_key")) == job["width"]:
            row_payload = item
            break
    if row_payload is None:
        return None
    row = _base_row(job, path)
    row["latency_ms"] = _float_or_none(row_payload.get("s5_fp16_cutlass_ms"))
    energy = row_payload.get("energy") if isinstance(row_payload.get("energy"), dict) else {}
    row["energy_j"] = _float_or_none(
        row_payload.get("energy_j") or row_payload.get("joule_per_inference") or energy.get("joule_per_inference")
    )
    row["build_status"] = "success" if row_payload.get("status") == "success" else "failed"
    row["run_status"] = "success" if row["latency_ms"] is not None and row_payload.get("status") == "success" else "failed"
    return row


def _ap_epoch(path: Path) -> int:
    match = re.search(r"_epoch(\d+)\.ya?ml$", path.name)
    return int(match.group(1)) if match else -1


def _latest_codriving_ap_file(width: str, results_root: Path) -> Path | None:
    ap_dir = results_root / "v2_gold_coldstart_96_20260708" / "codriving_ap_raw" / width
    if not ap_dir.is_dir():
        return None
    candidates = sorted(ap_dir.glob("eval_intermediate_epoch*.yaml"), key=_ap_epoch)
    if not candidates:
        candidates = sorted(ap_dir.glob("eval*.yaml"), key=_ap_epoch)
    return candidates[-1] if candidates else None


def _load_codriving_ap(path: Path) -> dict[str, float]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        "ap30": float(payload["ap30"]) if payload.get("ap30") is not None else None,
        "ap50": float(payload["ap_50"]) if payload.get("ap_50") is not None else None,
        "ap70": float(payload["ap_70"]) if payload.get("ap_70") is not None else None,
    }


def _trt_hybrid_ap_tag(row: dict[str, Any]) -> str | None:
    if str(row.get("backend_context", "")).lower() != "tensorrt":
        return None
    q_mode = str(row.get("q_mode", "")).lower()
    mixed = str(row.get("mixed_policy_id", "")).lower()
    if q_mode == "fp16" and mixed == "none":
        return "trt_fp16"
    if q_mode == "int8" and mixed == "all_eligible_conv":
        return "trt_int8_all"
    return None


def _latest_codriving_trt_hybrid_ap_file(row: dict[str, Any], results_root: Path) -> Path | None:
    tag = _trt_hybrid_ap_tag(row)
    if tag is None:
        return None
    width = width_str(row["width"])
    ap_dir = results_root / "v2_gold_coldstart_96_20260708" / "codriving_trt_hybrid_ap_raw" / width
    if not ap_dir.is_dir():
        return None
    candidates = sorted(ap_dir.glob(f"{tag}*.json"))
    if not candidates:
        return None
    final_candidates = [path for path in candidates if "final" in path.name.lower()]
    selected = final_candidates or candidates
    return max(selected, key=lambda path: path.stat().st_mtime)


def _load_codriving_trt_hybrid_ap(path: Path) -> dict[str, float] | None:
    payload = load_json(path)
    n_done = int(payload.get("n_done") or 0)
    n_trt_path = int(payload.get("n_trt_path") or 0)
    n_fallback_path = int(payload.get("n_fallback_path") or 0)
    if n_done <= 0 or n_trt_path != n_done or n_fallback_path != 0:
        return None
    ap70 = _float_or_none(payload.get("ap70"))
    if ap70 is None:
        return None
    return {
        "ap30": _float_or_none(payload.get("ap30")),
        "ap50": _float_or_none(payload.get("ap50")),
        "ap70": ap70,
    }


def _tvm_routeb_resnet_ap_tag(row: dict[str, Any]) -> str | None:
    if str(row.get("backend_context", "")).lower() != "tvm":
        return None
    if str(row.get("compiler_profile_id", "")) != "h800_tvm_routeb_20260708":
        return None
    q_mode = str(row.get("q_mode", "")).lower()
    mixed = str(row.get("mixed_policy_id", "")).lower()
    if q_mode == "fp16" and mixed == "none":
        return "tvm_routeb_fp16"
    if q_mode == "int8" and mixed == "all_eligible_conv":
        return "tvm_routeb_int8_all"
    if q_mode == "int8" and mixed == "top25_flops":
        return "tvm_routeb_int8_top25"
    if q_mode == "int8" and mixed == "top50_flops":
        return "tvm_routeb_int8_top50"
    return None


TVM_ROUTE_B_REPORT_CONTRACT = {
    "tvm_routeb_fp16": {"mode": "fp16", "precision": "fp16"},
    "tvm_routeb_int8_all": {"mode": "int8_all", "precision": "int8"},
    "tvm_routeb_int8_top25": {"mode": "mixed_top25_flops", "precision": "mixed"},
    "tvm_routeb_int8_top50": {"mode": "mixed_top50_flops", "precision": "mixed"},
}


def _latest_codriving_tvm_routeb_resnet_ap_file(row: dict[str, Any], results_root: Path) -> Path | None:
    tag = _tvm_routeb_resnet_ap_tag(row)
    if tag is None:
        return None
    width = width_str(row["width"])
    ap_dir = results_root / "v2_gold_coldstart_96_20260708" / "codriving_tvm_routeb_resnet_ap_raw" / width
    if not ap_dir.is_dir():
        return None
    candidates = sorted(ap_dir.glob(f"{tag}*.json"))
    if not candidates:
        return None
    final_candidates = [path for path in candidates if "final" in path.name.lower()]
    selected = final_candidates or candidates
    return max(selected, key=lambda path: path.stat().st_mtime)


def _load_codriving_tvm_routeb_resnet_ap(
    path: Path,
    *,
    expected_tag: str | None = None,
) -> dict[str, Any] | None:
    payload = load_json(path)
    if expected_tag is not None:
        contract = TVM_ROUTE_B_REPORT_CONTRACT.get(expected_tag)
        if contract is None or payload.get("tag") != expected_tag:
            return None
        if any(payload.get(field) != value for field, value in contract.items()):
            return None
    if payload.get("pipeline_scope") != "tvm_routeb_resnet_in_full_pytorch_eval":
        return None
    n_done = int(payload.get("n_done") or 0)
    n_tvm_path = int(payload.get("n_tvm_path") or 0)
    n_fallback_path = int(payload.get("n_fallback_path") or 0)
    if n_done <= 0 or n_tvm_path != n_done or n_fallback_path != 0:
        return None
    if not report_has_valid_int8_calibration(payload):
        return None
    ap70 = _float_or_none(payload.get("ap70"))
    if ap70 is None:
        return None
    return {
        "ap30": _float_or_none(payload.get("ap30")),
        "ap50": _float_or_none(payload.get("ap50")),
        "ap70": ap70,
        "pipeline_scope": payload.get("pipeline_scope"),
    }


def _row_accepts_fp_ap(row: dict[str, Any]) -> bool:
    return str(row.get("combo_id", "")).lower() in {"tvm_fp32_baseline", "cutlass_fp16"}


def apply_codriving_ap_overlays(rows: list[dict[str, Any]], results_root: Path = REPO / "results") -> None:
    ap_cache: dict[str, tuple[Path, dict[str, float]] | None] = {}
    trt_ap_cache: dict[tuple[str, str], tuple[Path, dict[str, float]] | None] = {}
    tvm_ap_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]] | None] = {}
    for row in rows:
        if not _row_accepts_fp_ap(row):
            pass
        else:
            width = width_str(row["width"])
            if width not in ap_cache:
                path = _latest_codriving_ap_file(width, results_root)
                ap_cache[width] = (path, _load_codriving_ap(path)) if path is not None else None
            cached = ap_cache[width]
            if cached is not None:
                path, metrics = cached
                row["ap70"] = metrics["ap70"]
                row["ap_source_file"] = str(path)

        tag = _trt_hybrid_ap_tag(row)
        if tag is not None:
            key = (width_str(row["width"]), tag)
            if key not in trt_ap_cache:
                path = _latest_codriving_trt_hybrid_ap_file(row, results_root)
                metrics = _load_codriving_trt_hybrid_ap(path) if path is not None else None
                trt_ap_cache[key] = (path, metrics) if path is not None and metrics is not None else None
            cached_trt = trt_ap_cache[key]
            if cached_trt is not None:
                path, metrics = cached_trt
                row["ap70"] = metrics["ap70"]
                row["ap_source_file"] = str(path)

        tvm_tag = _tvm_routeb_resnet_ap_tag(row)
        if tvm_tag is None:
            continue
        tvm_key = (width_str(row["width"]), tvm_tag)
        if tvm_key not in tvm_ap_cache:
            path = _latest_codriving_tvm_routeb_resnet_ap_file(row, results_root)
            metrics = (
                _load_codriving_tvm_routeb_resnet_ap(path, expected_tag=tvm_tag)
                if path is not None
                else None
            )
            tvm_ap_cache[tvm_key] = (path, metrics) if path is not None and metrics is not None else None
        cached_tvm = tvm_ap_cache[tvm_key]
        if cached_tvm is None:
            continue
        path, metrics = cached_tvm
        row["ap70"] = metrics["ap70"]
        row["ap_source_file"] = str(path)
        row["ap_pipeline_scope"] = metrics["pipeline_scope"]


def collect_existing_rows(plan: dict[str, Any], results_root: Path = REPO / "results") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        combo_id = str(job["combo_id"])
        row: dict[str, Any] | None = None
        if combo_id == "tvm_fp32_baseline":
            row = _row_from_tvm_fp32_baseline(job, results_root)
        elif combo_id in {"trt_fp16", "trt_int8_all"}:
            row = _row_from_trt(job, results_root)
        elif combo_id in {"tvm_routeb_fp16", "tvm_routeb_int8_all"}:
            row = _row_from_routeb_pure(job, results_root)
        elif combo_id in {"tvm_routeb_int8_top25", "tvm_routeb_int8_top50"}:
            row = _row_from_routeb_mixed(job, results_root)
        elif combo_id == "cutlass_fp16":
            row = _row_from_cutlass_fp16(job, results_root)
        if row is not None:
            rows.append(row)
    apply_codriving_ap_overlays(rows, results_root)
    return rows


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = load_json(path)
        if isinstance(payload, list):
            return list(payload)
        return list(payload.get("rows", []))

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _audit_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for status_key in ("missing_rows", "invalid_rows", "failed_rows"):
        for item in audit.get(status_key, []):
            row = {
                "audit_status": status_key.removesuffix("_rows"),
                "width": item.get("width"),
                "compiler_profile_id": item.get("compiler_profile_id"),
                "q_mode": item.get("q_mode"),
                "mixed_policy_id": item.get("mixed_policy_id"),
                "strategy_id": item.get("strategy_id"),
                "job_id": item.get("job_id"),
                "reasons": ";".join(item.get("reasons", [])),
            }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "validate", "import-existing"), default="plan")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--plan-csv", type=Path, default=None)
    parser.add_argument("--rows", type=Path, default=None, help="Training rows JSON/CSV for validate mode.")
    parser.add_argument("--training-json", type=Path, default=None)
    parser.add_argument("--training-csv", type=Path, default=None)
    parser.add_argument("--partial-json", type=Path, default=None)
    parser.add_argument("--partial-csv", type=Path, default=None)
    parser.add_argument("--audit-json", type=Path, default=None)
    parser.add_argument("--audit-csv", type=Path, default=None)
    args = parser.parse_args()

    plan_json = args.plan_json or args.out_dir / DEFAULT_PLAN_JSON.name
    plan_csv = args.plan_csv or args.out_dir / DEFAULT_PLAN_CSV.name
    audit_json = args.audit_json or args.out_dir / DEFAULT_AUDIT_JSON.name
    audit_csv = args.audit_csv or args.out_dir / DEFAULT_AUDIT_CSV.name
    training_json = args.training_json or args.out_dir / DEFAULT_TRAINING_JSON.name
    training_csv = args.training_csv or args.out_dir / DEFAULT_TRAINING_CSV.name
    partial_json = args.partial_json or args.out_dir / DEFAULT_PARTIAL_JSON.name
    partial_csv = args.partial_csv or args.out_dir / DEFAULT_PARTIAL_CSV.name

    if args.mode == "plan":
        plan = build_plan()
        audit = validate_training_rows(plan, [])
        write_json(plan_json, plan)
        write_csv(plan_csv, plan["jobs"])
        write_json(audit_json, audit)
        write_csv(audit_csv, _audit_rows(audit))
        print(f"wrote plan: {plan_json}")
        print(f"wrote audit: {audit_json}")
        print(f"jobs: {len(plan['jobs'])}")
        return

    if args.mode == "import-existing":
        plan = load_json(plan_json) if plan_json.is_file() else build_plan()
        rows = collect_existing_rows(plan, args.results_root)
        audit = validate_training_rows(plan, rows)
        write_json(partial_json, {"schema_version": "v2_gold_coldstart_96_partial_training_rows_v1", "rows": rows})
        write_csv(partial_csv, rows)
        write_json(audit_json, audit)
        write_csv(audit_csv, _audit_rows(audit))
        if audit["complete"]:
            write_json(training_json, {"schema_version": "v2_gold_coldstart_96_training_table_v1", "rows": rows})
            write_csv(training_csv, rows)
        print(f"imported_rows: {len(rows)}")
        print(f"complete: {audit['complete']}")
        print(json.dumps(audit["summary"], ensure_ascii=False, sort_keys=True))
        return

    if args.rows is None:
        raise SystemExit("--rows is required in validate mode")
    plan = load_json(plan_json)
    rows = load_rows(args.rows)
    audit = validate_training_rows(plan, rows)
    write_json(audit_json, audit)
    write_csv(audit_csv, _audit_rows(audit))
    if audit["complete"]:
        write_json(training_json, {"schema_version": "v2_gold_coldstart_96_training_table_v1", "rows": rows})
        write_csv(training_csv, rows)
    print(f"complete: {audit['complete']}")
    print(json.dumps(audit["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
