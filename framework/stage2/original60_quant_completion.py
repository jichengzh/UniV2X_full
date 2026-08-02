"""Completion queue helpers for original60 FP16/INT8 Stage2 quant work."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMPLETION_JOB_SCHEMA = "original60_fp16_int8_completion_job_v1"
COMPLETION_QUEUE_SCHEMA = "original60_fp16_int8_completion_queue_v1"
TARGET_PRECISIONS = ("fp16", "int8")
AXES = ("latency", "energy", "ap")
NATIVE_INT8_QUANT_METHOD = "h800_tvm_native_int8_backbone_subnet"
NATIVE_INT8_QUANT_SCOPE = "backbone_subnet_native_int8"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TVM_ROOT = _REPO_ROOT / "results" / "tvm_measurements"
DEFAULT_MODEL_ROOT = Path(os.environ.get("V2X_TVM_MODEL_ROOT", _DEFAULT_TVM_ROOT / "models"))
DEFAULT_WORKDIR_ROOT = Path(os.environ.get("V2X_TVM_WORKDIR_ROOT", _DEFAULT_TVM_ROOT / "workdirs"))


def _label(row: dict[str, Any]) -> str:
    explicit = str(row.get("label") or "")
    if explicit:
        return explicit
    software_point_id = str(row.get("software_point_id") or "")
    if software_point_id.startswith("original60:"):
        parts = software_point_id.split(":")
        if len(parts) > 1:
            return parts[1]
    candidate_id = str(row.get("candidate_id") or "")
    if candidate_id and ":" not in candidate_id:
        return candidate_id
    return ""


def _width(row: dict[str, Any]) -> list[int]:
    value = row.get("width")
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return [int(item) for item in value]
    software_point_id = str(row.get("software_point_id") or "")
    for token in software_point_id.split(":"):
        if "x" in token:
            parts = [part for part in token.split("x") if part]
            if parts and all(part.isdigit() for part in parts):
                return [int(part) for part in parts]
    return []


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        precision = str(row.get("precision") or "")
        if precision not in TARGET_PRECISIONS:
            continue
        label = _label(row)
        if not label:
            continue
        out[(label, precision)] = row
    return out


def _is_native_int8(row: dict[str, Any]) -> bool:
    return (
        str(row.get("quant_method") or "") == NATIVE_INT8_QUANT_METHOD
        and str(row.get("quant_scope") or "") == NATIVE_INT8_QUANT_SCOPE
        and str(row.get("engine_kind") or "") == "tvm_graph_executor"
    )


def _is_true_fp16(row: dict[str, Any]) -> bool:
    text = " ".join(
        str(row.get(key) or "").lower()
        for key in (
            "quant_method",
            "quality_gate_status",
            "layer_precision_summary",
            "measurement_source",
            "source_files",
            "notes",
        )
    )
    return "true_fp16" in text or "float16" in text


def _is_claimable_ap(row: dict[str, Any], precision: str) -> bool:
    if str(row.get("measurement_status") or "") != "measured":
        return False
    text = " ".join(str(value).lower() for value in row.values())
    if any(token in text for token in ("trt", "predicted", "interpolated", "model-fit", "model_fit")):
        return False
    if precision == "int8" and "int8" not in text:
        return False
    if precision == "fp16" and not ("fp16" in text or "float16" in text):
        return False
    return True


def _axis_status(
    *,
    axis: str,
    precision: str,
    row: dict[str, Any] | None,
) -> tuple[str, str | None]:
    if row is None:
        return "no_claim", "missing_original60_axis_row"
    status = str(row.get("measurement_status") or "no_claim")
    failure_reason = row.get("failure_reason")
    if status != "measured":
        return status, None if failure_reason is None else str(failure_reason)
    if bool(row.get("full_network_claim")):
        return "needs_remeasure", "full_network_claim_must_remain_false"
    if axis == "ap":
        if _is_claimable_ap(row, precision):
            return "measured", None
        return "no_claim", f"{precision}_ap_requires_compliant_true_eval_source"
    if precision == "int8" and not _is_native_int8(row):
        return "needs_native_int8_replace", "old_int8_route_requires_native_full_onnx_replacement"
    if precision == "fp16" and not _is_true_fp16(row):
        return "needs_true_fp16_remeasure", "fp16_row_lacks_true_fp16_precision_evidence"
    return "measured", None


def _required_actions(precision: str, statuses: dict[str, str]) -> list[str]:
    if precision == "fp16":
        mapping = {
            "latency": "run_true_fp16_latency",
            "energy": "run_true_fp16_energy",
            "ap": "run_true_fp16_ap_eval",
        }
    else:
        mapping = {
            "latency": "run_native_int8_full_onnx_latency",
            "energy": "run_native_int8_full_onnx_energy",
            "ap": "run_native_int8_ap_eval",
        }
    return [mapping[axis] for axis in AXES if statuses[axis] != "measured"]


def build_completion_jobs(
    *,
    latency_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
    created_at: str,
    model_root: Path = DEFAULT_MODEL_ROOT,
    workdir_root: Path = DEFAULT_WORKDIR_ROOT,
) -> list[dict[str, Any]]:
    indexes = {
        "latency": _index_rows(latency_rows),
        "energy": _index_rows(energy_rows),
        "ap": _index_rows(ap_rows),
    }
    labels = sorted(
        {
            label
            for index in indexes.values()
            for label, precision in index
            if precision in TARGET_PRECISIONS
        }
    )

    jobs: list[dict[str, Any]] = []
    for label in labels:
        for precision in TARGET_PRECISIONS:
            axis_rows = {
                axis: indexes[axis].get((label, precision))
                for axis in AXES
            }
            width = next(
                (
                    _width(row)
                    for row in axis_rows.values()
                    if row is not None and _width(row)
                ),
                [],
            )
            statuses: dict[str, str] = {}
            failure_reasons: list[str] = []
            for axis in AXES:
                status, reason = _axis_status(
                    axis=axis,
                    precision=precision,
                    row=axis_rows[axis],
                )
                statuses[axis] = status
                if reason and reason not in failure_reasons:
                    failure_reasons.append(reason)
            actions = _required_actions(precision, statuses)
            original60_candidate_id = next(
                (
                    str(row.get("original60_candidate_id"))
                    for row in axis_rows.values()
                    if row is not None and row.get("original60_candidate_id")
                ),
                f"original60:{label}",
            )
            jobs.append(
                {
                    "schema": COMPLETION_JOB_SCHEMA,
                    "job_id": f"original60_completion:{label}:{precision}",
                    "label": label,
                    "precision": precision,
                    "width": width,
                    "original60_candidate_id": original60_candidate_id,
                    "onnx_backbone_path": str(Path(model_root) / f"{label}_backbone.onnx"),
                    "workdir": str(Path(workdir_root) / label),
                    "latency_status": statuses["latency"],
                    "energy_status": statuses["energy"],
                    "ap_status": statuses["ap"],
                    "artifact_ready": False if actions else True,
                    "artifact_status": "pending_measurement" if actions else "complete",
                    "required_actions": actions,
                    "last_failure_reason": ";".join(failure_reasons) if failure_reasons else None,
                    "full_network_claim": False,
                    "axis_rows": {
                        axis: {
                            "measurement_status": str((axis_rows[axis] or {}).get("measurement_status") or "missing"),
                            "quality_gate_status": str((axis_rows[axis] or {}).get("quality_gate_status") or ""),
                            "quant_method": str((axis_rows[axis] or {}).get("quant_method") or ""),
                            "raw_artifact": (axis_rows[axis] or {}).get("raw_artifact"),
                        }
                        for axis in AXES
                    },
                    "created_at": created_at,
                }
            )
    return jobs


def summarize_completion_jobs(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    precision_counts = Counter(str(job["precision"]) for job in jobs)
    axis_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for job in jobs:
        for axis in AXES:
            axis_status_counts[axis][str(job[f"{axis}_status"])] += 1
    return {
        "schema": COMPLETION_QUEUE_SCHEMA,
        "total_jobs": len(jobs),
        "precision_counts": dict(sorted(precision_counts.items())),
        "jobs_requiring_action": sum(1 for job in jobs if job.get("required_actions")),
        "axis_status_counts": {
            axis: dict(sorted(counter.items()))
            for axis, counter in sorted(axis_status_counts.items())
        },
    }


def native_int8_label_widths_from_jobs(jobs: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for job in jobs:
        if str(job.get("precision") or "") != "int8":
            continue
        actions = set(str(action) for action in (job.get("required_actions") or []))
        if not {
            "run_native_int8_full_onnx_latency",
            "run_native_int8_full_onnx_energy",
        }.intersection(actions):
            continue
        label = str(job.get("label") or "")
        width = job.get("width")
        if not label or not isinstance(width, list):
            continue
        out[label] = [int(item) for item in width]
    return dict(sorted(out.items()))


def fp16_label_widths_from_jobs(jobs: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for job in jobs:
        if str(job.get("precision") or "") != "fp16":
            continue
        actions = set(str(action) for action in (job.get("required_actions") or []))
        if not {
            "run_true_fp16_latency",
            "run_true_fp16_energy",
        }.intersection(actions):
            continue
        label = str(job.get("label") or "")
        width = job.get("width")
        if not label or not isinstance(width, list):
            continue
        out[label] = [int(item) for item in width]
    return dict(sorted(out.items()))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _review_md(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# FP16/INT8 Original60 Completion Review",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- total_jobs: `{summary['total_jobs']}`",
        f"- jobs_requiring_action: `{summary['jobs_requiring_action']}`",
        f"- precision_counts: `{summary['precision_counts']}`",
        "",
        "## Jobs",
        "",
        "| label | precision | latency | energy | AP | required actions |",
        "|---|---|---|---|---|---|",
    ]
    for job in payload["jobs"]:
        lines.append(
            "| {label} | {precision} | {latency_status} | {energy_status} | {ap_status} | {actions} |".format(
                label=job["label"],
                precision=job["precision"],
                latency_status=job["latency_status"],
                energy_status=job["energy_status"],
                ap_status=job["ap_status"],
                actions=", ".join(job.get("required_actions") or []),
            )
        )
    return "\n".join(lines) + "\n"


def _gap_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        actions = list(job.get("required_actions") or [])
        if not actions:
            continue
        rows.append(
            {
                "job_id": job["job_id"],
                "label": job["label"],
                "precision": job["precision"],
                "width": job["width"],
                "latency_status": job["latency_status"],
                "energy_status": job["energy_status"],
                "ap_status": job["ap_status"],
                "required_actions": actions,
                "last_failure_reason": job.get("last_failure_reason"),
            }
        )
    return rows


def _gap_md(payload: dict[str, Any]) -> str:
    lines = [
        "# FP16/INT8 Original60 Completion Gap Report",
        "",
        f"- created_at: `{payload['created_at']}`",
        f"- gap_count: `{payload['gap_count']}`",
        "",
        "| label | precision | latency | energy | AP | reason |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {label} | {precision} | {latency_status} | {energy_status} | {ap_status} | {reason} |".format(
                label=row["label"],
                precision=row["precision"],
                latency_status=row["latency_status"],
                energy_status=row["energy_status"],
                ap_status=row["ap_status"],
                reason=row.get("last_failure_reason") or "",
            )
        )
    return "\n".join(lines) + "\n"


def write_completion_outputs(
    *,
    output_root: Path,
    jobs: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Path]:
    root = Path(output_root)
    queue_path = root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
    review_json = root / "exports/fp16_int8_original60_completion_review_latest.json"
    review_md = root / "exports/fp16_int8_original60_completion_review_latest.md"
    gap_json = root / "exports/fp16_int8_original60_gap_report_latest.json"
    gap_md = root / "exports/fp16_int8_original60_gap_report_latest.md"

    summary = summarize_completion_jobs(jobs)
    review_payload = {
        "schema": COMPLETION_QUEUE_SCHEMA,
        "created_at": created_at,
        "summary": summary,
        "jobs": jobs,
    }
    gap_rows = _gap_rows(jobs)
    gap_payload = {
        "schema": "original60_fp16_int8_completion_gap_report_v1",
        "created_at": created_at,
        "gap_count": len(gap_rows),
        "rows": gap_rows,
    }

    _write_jsonl(queue_path, jobs)
    _write_json(review_json, review_payload)
    _write_json(gap_json, gap_payload)
    review_md.parent.mkdir(parents=True, exist_ok=True)
    review_md.write_text(_review_md(review_payload), encoding="utf-8")
    gap_md.parent.mkdir(parents=True, exist_ok=True)
    gap_md.write_text(_gap_md(gap_payload), encoding="utf-8")
    return {
        "queue": queue_path,
        "review_json": review_json,
        "review_md": review_md,
        "gap_json": gap_json,
        "gap_md": gap_md,
    }
