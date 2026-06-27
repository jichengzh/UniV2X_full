#!/usr/bin/env python3
"""Import measured energy telemetry CSV into canonical Stage2 energy rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import energy_lut_row, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--backend", default="h800_tvm_power_telemetry")
    parser.add_argument("--measurement-status", default="measured")
    return parser.parse_args()


def _float_or_none(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _int_or_none(value: str | None) -> int | None:
    return None if value in (None, "") else int(value)


def main() -> int:
    args = parse_args()
    source = Path(args.source_csv)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            width = [int(value) for value in str(item["width"]).split("|") if value]
            raw_artifact = item.get("raw_artifact") or str(source)
            run_id = item.get("run_id") or f"import_energy_{source.stem}"
            rows.append(
                energy_lut_row(
                    config_id=item["config_id"],
                    model=item["model"],
                    manifest_digest=item.get("manifest_digest") or digest,
                    candidate_id=item["candidate_id"],
                    software_point_id=item["software_point_id"],
                    dense_stage=item["dense_stage"],
                    width=width,
                    quant_policy=item["quant_policy"],
                    schedule_policy=item["schedule_policy"],
                    backend=args.backend,
                    measurement_status=args.measurement_status,
                    joule_per_inference=float(item["joule_per_inference"]),
                    watt_avg=_float_or_none(item.get("watt_avg")),
                    watt_p50=_float_or_none(item.get("watt_p50")),
                    watt_p90=_float_or_none(item.get("watt_p90")),
                    idle_watt_avg=_float_or_none(item.get("idle_watt_avg")),
                    idle_baseline_policy=item.get("idle_baseline_policy")
                    or "subtract_idle_avg",
                    sample_window_ms=_int_or_none(item.get("sample_window_ms")) or 0,
                    telemetry_source=item["telemetry_source"],
                    power_cap_watt=_float_or_none(item.get("power_cap_watt")),
                    clock_policy=item.get("clock_policy") or "default",
                    latency_config_id=item.get("latency_config_id") or item["config_id"],
                    latency_run_id=item["latency_run_id"],
                    warmup_iters=_int_or_none(item.get("warmup_iters")) or 0,
                    measure_iters=_int_or_none(item.get("measure_iters")) or 0,
                    repeat=_int_or_none(item.get("repeat")) or 0,
                    provenance=item.get("provenance")
                    or "imported H800 power telemetry",
                    run_id=run_id,
                    measurement_run_id=run_id,
                    row_source="import_existing",
                    created_at=item.get("created_at") or created_at,
                    source_files=[str(source)],
                    raw_artifact=raw_artifact,
                    notes=item.get("notes")
                    or "imported measured telemetry; dense-core only",
                )
            )
    write_jsonl(args.out_jsonl, rows)
    print(f"stage2_import_energy_lut_ok rows={len(rows)} out={args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
