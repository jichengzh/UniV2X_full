#!/usr/bin/env python3
"""Audit cross-time Gold128 latency/energy repeats against frozen Gold baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "stage35_gold128_repeat_audit_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coefficient_of_variation(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or not np.all(np.isfinite(array)) or float(np.mean(array)) == 0.0:
        raise ValueError("CV requires at least two finite, nonzero-mean measurements")
    return float(np.std(array, ddof=1) / abs(np.mean(array)) * 100.0)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _first_finite(values: Sequence[Any]) -> float | None:
    return next((float(value) for value in values if _finite(value)), None)


def _metrics_from_state(state: Mapping[str, Any]) -> tuple[float | None, float | None]:
    result_path = Path(str(state.get("result_json") or ""))
    expected_sha = str(state.get("result_sha256") or "")
    if not result_path.is_file() or not expected_sha:
        return None, None
    if sha256_file(result_path) != expected_sha:
        raise ValueError(f"repeat result SHA256 mismatch: {result_path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    latency_payload = payload.get("latency") if isinstance(payload.get("latency"), Mapping) else {}
    energy_payload = payload.get("energy") if isinstance(payload.get("energy"), Mapping) else {}
    return (
        _first_finite([
            payload.get("lat_p50_ms"), payload.get("latency_ms"),
            latency_payload.get("latency_ms_p50"), latency_payload.get("lat_p50_ms"),
        ]),
        _first_finite([
            payload.get("energy_j"), energy_payload.get("energy_j"), energy_payload.get("joules"),
            energy_payload.get("joules_per_inference"), energy_payload.get("joule_per_inference"),
            energy_payload.get("energy_J"),
        ]),
    )


def _result_is_bound_to_job(job: Mapping[str, Any], result_path: str) -> bool:
    command = list(map(str, job.get("command") or []))
    result = Path(result_path).resolve()
    if "--out" in command:
        return result == Path(command[command.index("--out") + 1]).resolve()
    if "--out-dir" in command:
        if "--label" not in command or len(command) < 2:
            return False
        output_dir = Path(command[command.index("--out-dir") + 1]).resolve()
        label = command[command.index("--label") + 1]
        runner = Path(command[1]).name
        filenames = {
            "stage2_route_b_fp16_auto_runner.py": "route_b_fp16_auto_result.json",
            "stage2_route_b_int8_auto_decomp.py": "route_b_int8_auto_decomp_result.json",
        }
        filename = filenames.get(runner)
        return filename is not None and result == output_dir / label / filename
    return False


def build_repeat_audit(
    jobs: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    ap_repeat_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    jobs_by_id = {str(row.get("job_id")): row for row in jobs}
    success_by_id: dict[str, Mapping[str, Any]] = {}
    for row in state_rows:
        if row.get("status") == "success":
            success_by_id[str(row.get("job_id"))] = row
    if len(jobs_by_id) != 32 or set(success_by_id) != set(jobs_by_id):
        raise ValueError(f"repeat audit requires 32 successful jobs, got jobs={len(jobs_by_id)} success={len(success_by_id)}")
    gold_by_id = {str(row.get("manifest_job_id")): row for row in gold_rows}

    rows: list[dict[str, Any]] = []
    for job_id, job in jobs_by_id.items():
        state = success_by_id[job_id]
        if not _result_is_bound_to_job(job, str(state.get("result_json") or "")):
            raise ValueError(f"repeat result is not bound to job output contract: {job_id}")
        manifest_id = str(job["manifest_job_id"])
        baseline = gold_by_id.get(manifest_id)
        if baseline is None or baseline.get("terminal_status") != "measured_success_gold":
            raise ValueError(f"missing measured Gold baseline for {manifest_id}")
        repeat_latency, repeat_energy = _metrics_from_state(state)
        values = {
            "baseline_latency_ms": baseline.get("latency_ms"),
            "repeat_latency_ms": repeat_latency,
            "baseline_energy_j": baseline.get("energy_j"),
            "repeat_energy_j": repeat_energy,
        }
        if not all(_finite(value) for value in values.values()):
            raise ValueError(f"non-finite repeat evidence for {job_id}: {values}")
        baseline_latency = float(values["baseline_latency_ms"])
        repeat_latency = float(values["repeat_latency_ms"])
        baseline_energy = float(values["baseline_energy_j"])
        repeat_energy = float(values["repeat_energy_j"])
        rows.append({
            "schema_version": SCHEMA,
            "job_id": job_id,
            "manifest_job_id": manifest_id,
            "group_id": job["group_id"],
            "repeat_category": job["repeat_category"],
            "runner_key": job["runner_key"],
            **values,
            "latency_relative_drift": (repeat_latency - baseline_latency) / baseline_latency,
            "energy_relative_drift": (repeat_energy - baseline_energy) / baseline_energy,
            "latency_cv_pct": coefficient_of_variation([baseline_latency, repeat_latency]),
            "energy_cv_pct": coefficient_of_variation([baseline_energy, repeat_energy]),
            "repeat_result_json": state.get("result_json"),
            "repeat_result_sha256": state.get("result_sha256"),
        })

    latency_cv = [float(row["latency_cv_pct"]) for row in rows]
    energy_cv = [float(row["energy_cv_pct"]) for row in rows]
    latency_distribution = _distribution(latency_cv)
    energy_distribution = _distribution(energy_cv)
    performance_checks = {
        "latency_cv_median_le_5pct": latency_distribution["median"] <= 5.0,
        "latency_cv_p90_le_10pct": latency_distribution["p90"] <= 10.0,
        "energy_cv_median_le_10pct": energy_distribution["median"] <= 10.0,
        "energy_cv_p90_le_20pct": energy_distribution["p90"] <= 20.0,
    }
    category_summary = {}
    for category in sorted({str(row["repeat_category"]) for row in rows}):
        selected = [row for row in rows if row["repeat_category"] == category]
        category_summary[category] = {
            "rows": len(selected),
            "latency_cv_pct": _distribution([float(row["latency_cv_pct"]) for row in selected]),
            "energy_cv_pct": _distribution([float(row["energy_cv_pct"]) for row in selected]),
        }
    performance_qualified = all(performance_checks.values())
    ap_status = "qualified" if ap_repeat_audit and ap_repeat_audit.get("qualified") is True else "not_measured"
    return {
        "schema_version": SCHEMA,
        "terminal_rows": len(rows),
        "performance_repeat_qualified": performance_qualified,
        "performance_checks": performance_checks,
        "latency_cv_pct": latency_distribution,
        "energy_cv_pct": energy_distribution,
        "category_summary": category_summary,
        "ap_repeat_status": ap_status,
        "ap_repeat_audit": dict(ap_repeat_audit) if ap_repeat_audit else None,
        "qualified": performance_qualified and ap_status == "qualified",
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--state-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--ap-repeat-audit-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_repeat_audit(
        [row for path in args.jobs_jsonl for row in read_jsonl(path)],
        [row for path in args.state_jsonl for row in read_jsonl(path)],
        json.loads(args.gold_json.read_text(encoding="utf-8")),
        ap_repeat_audit=json.loads(args.ap_repeat_audit_json.read_text(encoding="utf-8")) if args.ap_repeat_audit_json else None,
    )
    result["source_gold_sha256"] = sha256_file(args.gold_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
        writer.writeheader(); writer.writerows(result["rows"])
    print(json.dumps({key: result[key] for key in ("terminal_rows", "performance_repeat_qualified", "ap_repeat_status", "qualified")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
