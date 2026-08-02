#!/usr/bin/env python3
"""Aggregate F-Cooper TVM scheduler, source-reuse, and budget evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def verify_binding(binding: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(binding.get("path") or "")).resolve()
    expected = str(binding.get("sha256") or "")
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"artifact binding drift: {path}")
    return path, read_json(path)


def _empty_gpu() -> dict[str, Any]:
    return {
        "busy_seconds": 0.0,
        "jobs_started": 0,
        "job_attempts": [],
    }


def scheduler_summary(
    evidence_root: Path,
    *,
    expected_pairs: set[tuple[str, int]],
) -> dict[str, Any]:
    per_gpu = {str(index): _empty_gpu() for index in range(8)}
    components = []
    successful_pairs: set[tuple[str, int]] = set()
    earliest = math.inf
    latest = -math.inf
    total_gpu_seconds = 0.0
    total_retries = 0
    peak = 0
    nonterminal_components: list[tuple[str, set[tuple[str, int]]]] = []
    queue_component_count = 0
    queued_jobs = 0
    maximum_start_delay_seconds = 0.0
    for audit_path in sorted((evidence_root / "schedulers").glob("*/audit.json")):
        manifest_path = audit_path.parent / "manifest.json"
        state_path = audit_path.parent / "state.json"
        if not manifest_path.is_file() or not state_path.is_file():
            continue
        manifest = read_json(manifest_path)
        manifest_without_sha = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
        if (
            manifest.get("schema_version") != "fcooper_tvm_gpu_job_manifest_v1"
            or manifest.get("manifest_sha256")
            != canonical_sha256(manifest_without_sha)
        ):
            continue
        audit = read_json(audit_path)
        if audit.get("schema_version") != "fcooper_tvm_gpu_scheduler_audit_v1":
            raise ValueError(f"unexpected scheduler audit: {audit_path}")
        state = read_json(state_path)
        matched_jobs = []
        for job in manifest.get("jobs") or []:
            request_path = Path(str(job.get("request_json") or "")).resolve()
            if not request_path.is_file():
                continue
            request = read_json(request_path)
            rows = request.get("rows")
            row_index = int(job.get("row_index", -1))
            if (
                not isinstance(rows, list)
                or row_index < 0
                or row_index >= len(rows)
            ):
                continue
            row_id = str(
                rows[row_index].get("row_id")
                or rows[row_index].get("manifest_job_id")
                or ""
            )
            pair = (row_id, int(job.get("max_trials", -1)))
            if pair not in expected_pairs:
                continue
            state_job = (state.get("jobs") or {}).get(str(job.get("job_id")))
            if not isinstance(state_job, Mapping):
                raise ValueError(f"scheduler state lacks formal job: {pair}")
            if (state_job.get("input_snapshot") or {}).get("row_id") != row_id:
                raise ValueError(f"scheduler formal row binding drift: {pair}")
            matched_jobs.append((dict(job), dict(state_job), pair))
        if not matched_jobs:
            continue
        queue = audit.get("queue")
        if not isinstance(queue, Mapping):
            raise ValueError(f"scheduler audit lacks queue accounting: {audit_path}")
        queue_component_count += 1
        queued_jobs += int(queue.get("queued_jobs", len(matched_jobs)))
        maximum_start_delay_seconds = max(
            maximum_start_delay_seconds,
            float(queue.get("maximum_start_delay_seconds", 0.0)),
        )
        counts = audit.get("status_counts") or {}
        is_terminal = (
            int(counts.get("pending", -1)) == 0
            and int(counts.get("running", -1)) == 0
        )
        if not is_terminal:
            nonterminal_components.append(
                (
                    audit_path.parent.name,
                    {pair for _, _, pair in matched_jobs},
                )
            )
        matched_pair_labels = []
        for job, state_job, pair in matched_jobs:
            matched_pair_labels.append({"row_id": pair[0], "tvm_trials": pair[1]})
            if state_job.get("status") == "succeeded":
                successful_pairs.add(pair)
            attempts = state_job.get("attempt_history") or []
            total_retries += max(0, len(attempts) - 1)
            for attempt in attempts:
                gpu = int(attempt.get("gpu", -1))
                if gpu not in range(8):
                    raise ValueError(f"formal scheduler attempt has invalid GPU: {pair}")
                seconds_value = attempt.get("gpu_seconds")
                seconds = (
                    float(seconds_value)
                    if seconds_value is not None
                    else 0.0
                )
                target = per_gpu[str(gpu)]
                target["busy_seconds"] += seconds
                target["jobs_started"] += 1
                target["job_attempts"].append(
                    {
                        **dict(attempt),
                        "scheduler": audit_path.parent.name,
                        "job_id": job["job_id"],
                        "row_id": pair[0],
                        "tvm_trials": pair[1],
                    }
                )
                total_gpu_seconds += seconds
                started = attempt.get("started_wall_time")
                if started is not None:
                    earliest = min(earliest, float(started))
            finished = state_job.get("finished_wall_time")
            if finished is not None:
                latest = max(latest, float(finished))
        peak = max(
            peak,
            int((audit.get("effective_parallelism") or {}).get("peak_running_jobs", 0)),
        )
        components.append(
            {
                "scheduler": audit_path.parent.name,
                "audit_path": str(audit_path.resolve()),
                "audit_sha256": file_sha256(audit_path),
                "state_path": str(state_path.resolve()),
                "state_sha256": file_sha256(state_path),
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": file_sha256(manifest_path),
                "matched_formal_pairs": matched_pair_labels,
                "status_counts": counts,
                "terminal": is_terminal,
                "gpu_hours": sum(
                    float(attempt.get("gpu_seconds") or 0.0)
                    for _, state_job, _ in matched_jobs
                    for attempt in (state_job.get("attempt_history") or [])
                )
                / 3600.0,
                "wall_clock_seconds": float(audit.get("wall_clock_seconds") or 0.0),
            }
        )
    missing_pairs = expected_pairs - successful_pairs
    if missing_pairs:
        raise ValueError(
            "formal scheduler coverage is incomplete: "
            + ", ".join(f"{row_id}@{trials}" for row_id, trials in sorted(missing_pairs))
        )
    if not components or not math.isfinite(earliest) or latest < earliest:
        raise ValueError("no terminal scheduler evidence found")
    superseded_abandoned = sorted(
        name for name, pairs in nonterminal_components if pairs <= successful_pairs
    )
    unresolved_nonterminal = sorted(
        name for name, pairs in nonterminal_components if not pairs <= successful_pairs
    )
    elapsed = latest - earliest
    return {
        "component_count": len(components),
        "components": components,
        "per_gpu": per_gpu,
        "gpu_hours": total_gpu_seconds / 3600.0,
        "wall_clock_seconds": elapsed,
        "effective_parallelism": {
            "average_gpu_jobs": total_gpu_seconds / elapsed if elapsed > 0 else 0.0,
            "peak_running_jobs": peak,
        },
        "queue": {
            "component_count": queue_component_count,
            "queued_jobs": queued_jobs,
            "maximum_start_delay_seconds": maximum_start_delay_seconds,
        },
        "infrastructure_retry_count": total_retries,
        "nonterminal_or_abandoned_components": unresolved_nonterminal,
        "superseded_abandoned_components": superseded_abandoned,
        "formal_pair_coverage": {
            "expected": len(expected_pairs),
            "successful": len(successful_pairs),
            "missing": [],
        },
    }


def formal_measurement_summary(pools: Mapping[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = []
    pool_bindings = {}
    for name, path in sorted(pools.items()):
        resolved = Path(path).resolve()
        payload = read_json(resolved)
        if (
            payload.get("schema_version") != "fcooper_tvm_stage6_evidence_pool_v1"
            or payload.get("pool_name") != name
            or not isinstance(payload.get("rows"), list)
        ):
            raise ValueError(f"formal pool identity drift: {resolved}")
        rows.extend(dict(row) for row in payload["rows"])
        pool_bindings[name] = {
            "path": str(resolved),
            "sha256": file_sha256(resolved),
            "row_count": len(payload["rows"]),
        }
    identities = [
        (str(row.get("row_id") or ""), int(row.get("tvm_trials", -1)))
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("cross-pool measurement identity reuse is forbidden")
    source_rows = []
    for row in rows:
        source_binding = (row.get("artifacts") or {}).get("source_provenance")
        if source_binding is None:
            continue
        source_path, source = verify_binding(source_binding)
        if (
            source.get("passed") is not True
            or source.get("backend_neutral_only") is not True
            or source.get("trt_compiled_artifacts_reused") is not False
            or source.get("trt_performance_labels_reused") is not False
            or source.get("trt_predictions_or_ap_reused") is not False
        ):
            raise ValueError(f"source reuse policy drift: {source_path}")
        source_rows.append(source)
    reused = sum(source.get("source_reused_from_trt_v2") is True for source in source_rows)
    generated = sum(source.get("source_generated_in_tvm_v1") is True for source in source_rows)
    return (
        {
            "pool_bindings": pool_bindings,
            "row_observations": len(rows),
            "unique_row_trial_pairs": len(
                set(identities)
            ),
            "row_trial_pairs": [
                {"row_id": row_id, "tvm_trials": trials}
                for row_id, trials in sorted(identities)
            ],
            "tvm_trials": sum(int(row.get("tvm_trials", 0)) for row in rows),
        },
        {
            "audited_success_rows": len(source_rows),
            "reused_from_trt_v2_rows": reused,
            "generated_in_tvm_v1_rows": generated,
            "reuse_rate": reused / len(source_rows) if source_rows else 0.0,
            "recovery_training_seconds_saved": sum(
                float(source.get("recovery_training_seconds_saved") or 0.0)
                for source in source_rows
            ),
        },
    )


def build_resource_audit(
    *,
    evidence_root: Path,
    pools: Mapping[str, Path],
    outer_budget: int,
) -> dict[str, Any]:
    formal, source_reuse = formal_measurement_summary(pools)
    expected_pairs = {
        (str(row["row_id"]), int(row["tvm_trials"]))
        for row in formal["row_trial_pairs"]
    }
    scheduler = scheduler_summary(
        Path(evidence_root).resolve(),
        expected_pairs=expected_pairs,
    )
    return {
        "schema_version": "fcooper_tvm_stage6_resource_audit_v1",
        "passed": True,
        "outer_budget": int(outer_budget),
        "formal_measurement": formal,
        "source_reuse": source_reuse,
        "scheduler": {
            key: value
            for key, value in scheduler.items()
            if key != "per_gpu"
        },
        "per_gpu": scheduler["per_gpu"],
    }


def parse_pool(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--pool requires NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--pool", action="append", type=parse_pool, required=True)
    parser.add_argument("--outer-budget", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pools = dict(args.pool)
    if len(pools) != len(args.pool):
        raise ValueError("duplicate --pool name")
    audit = build_resource_audit(
        evidence_root=args.evidence_root,
        pools=pools,
        outer_budget=args.outer_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
