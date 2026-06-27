#!/usr/bin/env python3
"""Update a Stage2 evidence registry from canonical LUT row JSONL files."""

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
    coverage_from_rows,
    energy_claim_allowed_from_rows,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--latency-rows")
    parser.add_argument("--ap-rows")
    parser.add_argument("--energy-rows")
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def _source_update(
    *,
    path: str,
    rows: list[dict[str, Any]],
    backend: str,
    measured_status: str = "measured",
    provenance: str,
) -> dict[str, Any]:
    coverage = coverage_from_rows(rows)
    status = measured_status if coverage["measured_cells"] > 0 else "not_done"
    return {
        "path": path,
        "measurement_status": status,
        "backend": backend if status == measured_status else "not_available",
        "coverage": coverage,
        "provenance": provenance,
    }


def _merge_source(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    merged.update(update)
    return merged


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    data = json.loads(registry_path.read_text(encoding="utf-8"))

    if args.latency_rows:
        rows = read_jsonl(args.latency_rows)
        data["latency_lut"] = _merge_source(
            dict(data.get("latency_lut", {})),
            _source_update(
                path=args.latency_rows,
                rows=rows,
                backend="h800_tvm",
                provenance="canonical latency_lut_rows_v1",
            ),
        )

    if args.ap_rows:
        rows = read_jsonl(args.ap_rows)
        data["ap_anchors"] = _merge_source(
            dict(data.get("ap_anchors", {})),
            _source_update(
                path=args.ap_rows,
                rows=rows,
                backend="model_eval",
                provenance="canonical ap_anchor_rows_v1",
            ),
        )

    if args.energy_rows:
        rows = read_jsonl(args.energy_rows)
        coverage = coverage_from_rows(rows)
        claim_allowed = energy_claim_allowed_from_rows(rows)
        data["energy_lut"] = _merge_source(
            dict(data.get("energy_lut", {})),
            {
                "path": args.energy_rows,
                "measurement_status": "measured" if claim_allowed else "not_done",
                "backend": "h800_tvm_power_telemetry"
                if claim_allowed
                else "not_available",
                "coverage": coverage,
                "provenance": "canonical energy_lut_rows_v1",
            },
        )

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"schema": "stage2_registry_update_summary_v1", "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
