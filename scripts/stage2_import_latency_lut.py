#!/usr/bin/env python3
"""Import an existing Stage2 latency LUT JSON into canonical latency rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    latency_lut_row,
    stable_config_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--backend", default="h800_tvm")
    parser.add_argument("--measurement-status", default="measured")
    return parser.parse_args()


def _items(data: dict[str, object]) -> list[dict[str, object]]:
    value = data.get("widths")
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    value = data.get("grid")
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def main() -> int:
    args = parse_args()
    source = Path(args.source_json)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = json.loads(source.read_text(encoding="utf-8"))
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    rows = []
    for item in _items(data):
        width = [int(value) for value in item["num_filters"]]
        label = str(item.get("label", "_".join(str(value) for value in width)))
        software_point_id = f"{label}:w{'x'.join(str(value) for value in width)}:fp16"
        for schedule_policy, latency_key in (
            ("default", "default_us"),
            ("metaschedule_tuned", "tuned_us"),
        ):
            if item.get(latency_key) is None:
                continue
            measurement_status = args.measurement_status
            if item.get("_source") == "estimated_vol_power_law":
                measurement_status = "estimated"
            config_id = stable_config_id(
                model=args.model,
                candidate_id=label,
                software_point_id=software_point_id,
                quant_policy="fp16",
                schedule_policy=schedule_policy,
            )
            rows.append(
                latency_lut_row(
                    config_id=config_id,
                    model=args.model,
                    manifest_digest=digest,
                    candidate_id=label,
                    software_point_id=software_point_id,
                    dense_stage=str(item.get("dense_stage", "unknown")),
                    width=width,
                    quant_policy="fp16",
                    schedule_policy=schedule_policy,
                    backend=args.backend,
                    measurement_status=measurement_status,
                    latency_p50_us=float(item[latency_key]),
                    latency_p90_us=None,
                    latency_mean_us=None,
                    latency_std_us=None,
                    latency_min_us=None,
                    latency_max_us=None,
                    warmup_iters=0,
                    measure_iters=0,
                    repeat=0,
                    run_id=f"import_latency_{source.stem}",
                    created_at=created_at,
                    source_files=[str(source)],
                    raw_artifact=str(source),
                    provenance=str(
                        data.get("_source", data.get("_source_real", "imported_latency_lut"))
                    ),
                    notes="imported from existing Stage2 latency LUT; not a new generation job",
                )
            )
    write_jsonl(args.out_jsonl, rows)
    print(f"stage2_import_latency_lut_ok rows={len(rows)} out={args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
