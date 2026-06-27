#!/usr/bin/env python3
"""Salvage canonical energy LUT rows from completed telemetry payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    append_jsonl,
    energy_lut_row,
    read_jsonl,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--latency-rows", action="append", required=True)
    parser.add_argument("--existing-energy-rows", action="append", default=[])
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--job-state")
    parser.add_argument("--created-at")
    return parser.parse_args()


def _read_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else float(value)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return None if value is None else int(value)


def _payload_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("telemetry_payload.json"))


def _state_event(
    state_path: Path | None,
    *,
    job_id: str,
    status: str,
    payload_path: Path,
    row_id: str | None = None,
) -> None:
    if state_path is None:
        return
    append_jsonl(
        state_path,
        {
            "schema": "lut_job_state_row_v1",
            "job_id": job_id,
            "status": status,
            "attempt": 0,
            "finished_at": utc_timestamp(),
            "payload_path": str(payload_path),
            "output_row_id": row_id,
        },
    )


def _source_files(payload: dict[str, Any], payload_path: Path) -> list[str]:
    files = [str(item) for item in payload.get("source_files", []) or []]
    payload_text = str(payload_path)
    if payload_text not in files:
        files.append(payload_text)
    return files


def build_salvaged_energy_row(
    *,
    payload: dict[str, Any],
    payload_path: Path,
    latency_row: dict[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    run_id = str(payload.get("run_id") or payload_path.parent.name)
    return energy_lut_row(
        config_id=str(latency_row["config_id"]),
        model=str(latency_row["model"]),
        manifest_digest=str(latency_row.get("manifest_digest") or "unknown"),
        candidate_id=str(latency_row["candidate_id"]),
        software_point_id=str(latency_row["software_point_id"]),
        dense_stage=str(latency_row["dense_stage"]),
        optimized_scope=str(latency_row.get("optimized_scope") or "rsu_dense_core"),
        width=list(latency_row["width"]),
        quant_policy=str(latency_row["quant_policy"]),
        schedule_policy=str(latency_row["schedule_policy"]),
        backend="h800_tvm_power_telemetry",
        measurement_status="measured",
        joule_per_inference=float(payload["joule_per_inference"]),
        watt_avg=_optional_float(payload, "watt_avg"),
        watt_p50=_optional_float(payload, "watt_p50"),
        watt_p90=_optional_float(payload, "watt_p90"),
        idle_watt_avg=_optional_float(payload, "idle_watt_avg"),
        idle_baseline_policy=payload.get("idle_baseline_policy", "subtract_idle_avg"),
        sample_window_ms=_optional_int(payload, "sample_window_ms"),
        telemetry_source=payload["telemetry_source"],
        power_cap_watt=_optional_float(payload, "power_cap_watt"),
        clock_policy=payload.get("clock_policy", "default"),
        latency_config_id=str(latency_row["config_id"]),
        latency_run_id=str(payload["latency_run_id"]),
        warmup_iters=int(payload.get("warmup_iters", latency_row.get("warmup_iters", 0))),
        measure_iters=int(payload.get("measure_iters", latency_row.get("measure_iters", 0))),
        repeat=int(payload.get("repeat", latency_row.get("repeat", 1))),
        provenance=payload.get(
            "provenance",
            "H800 power telemetry salvaged from payload after canonical row failure",
        ),
        run_id=run_id,
        created_at=created_at or str(payload.get("created_at") or utc_timestamp()),
        source_files=_source_files(payload, payload_path),
        raw_artifact=str(payload.get("raw_artifact") or payload_path.parent),
        row_source="salvaged_from_payload",
        measurement_run_id=run_id,
        notes=payload.get("notes", "salvaged energy payload; config inherited from latency row"),
    )


def main() -> int:
    args = parse_args()
    latency_rows = _read_many(args.latency_rows)
    latency_by_run = {str(row.get("run_id")): row for row in latency_rows if row.get("run_id")}
    existing_energy_rows = _read_many(args.existing_energy_rows)
    seen_run_ids = {str(row.get("run_id")) for row in existing_energy_rows if row.get("run_id")}
    seen_row_ids = {str(row.get("row_id")) for row in existing_energy_rows if row.get("row_id")}
    out_jsonl = Path(args.out_jsonl)
    state_path = Path(args.job_state) if args.job_state else None

    salvaged = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for payload_path in _payload_paths(Path(args.payload_root)):
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            latency_run_id = str(payload["latency_run_id"])
            for key in ("joule_per_inference", "telemetry_source"):
                if payload.get(key) is None:
                    raise KeyError(key)
            latency_row = latency_by_run[latency_run_id]
            row = build_salvaged_energy_row(
                payload=payload,
                payload_path=payload_path,
                latency_row=latency_row,
                created_at=args.created_at,
            )
            job_id = str(payload.get("job_id") or f"energy:{latency_row['config_id']}")
            _state_event(
                state_path,
                job_id=job_id,
                status="payload_ready",
                payload_path=payload_path,
            )
            if row["run_id"] in seen_run_ids or row["row_id"] in seen_row_ids:
                skipped += 1
                continue
            append_jsonl(out_jsonl, row)
            seen_run_ids.add(str(row["run_id"]))
            seen_row_ids.add(str(row["row_id"]))
            salvaged += 1
            _state_event(
                state_path,
                job_id=job_id,
                status="salvaged",
                payload_path=payload_path,
                row_id=str(row["row_id"]),
            )
            _state_event(
                state_path,
                job_id=job_id,
                status="succeeded",
                payload_path=payload_path,
                row_id=str(row["row_id"]),
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            skipped += 1
            failures.append({"payload": str(payload_path), "reason": str(exc)})

    print(
        json.dumps(
            {
                "schema": "stage2_energy_salvage_summary_v1",
                "payload_root": args.payload_root,
                "salvaged_rows": salvaged,
                "skipped_payloads": skipped,
                "failures": failures,
                "out_jsonl": str(out_jsonl),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
